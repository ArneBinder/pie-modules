import logging
from collections.abc import MutableMapping
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from pytorch_ie.core import PyTorchIEModel
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.nn import Parameter
from transformers import get_linear_schedule_with_warmup

from ..taskmodules.components.seq2seq import PointerNetworkSpanAndRelationEncoderDecoder
from .components.pointer_network.bart_as_pointer_network import BartAsPointerNetwork
from .components.pointer_network.metrics import AnnotationLayerMetric

logger = logging.getLogger(__name__)


STAGE_TRAIN = "train"
STAGE_VAL = "val"
STAGE_TEST = "test"


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flatten_dict_gen(d, parent_key, sep))


@PyTorchIEModel.register()
class SimplePointerNetworkModel(PyTorchIEModel):
    def __init__(
        self,
        model_name_or_path: str,
        target_token_ids: List[int],
        vocab_size: int,
        embedding_weight_mapping: Optional[Dict[int, List[int]]] = None,
        use_encoder_mlp: bool = False,
        annotation_encoder_decoder_name: str = "pointer_network_span_and_relation",
        annotation_encoder_decoder_kwargs: Optional[Dict[str, Any]] = None,
        metric_splits: List[str] = [STAGE_VAL, STAGE_TEST],
        # optimizer / scheduler
        lr: float = 5e-5,
        layernorm_decay: float = 0.001,
        warmup_proportion: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.lr = lr
        self.layernorm_decay = layernorm_decay
        self.warmup_proportion = warmup_proportion

        if annotation_encoder_decoder_name == "pointer_network_span_and_relation":
            self.annotation_encoder_decoder = PointerNetworkSpanAndRelationEncoderDecoder(
                **(annotation_encoder_decoder_kwargs or {}),
            )
        else:
            raise Exception(
                f"Unsupported annotation encoder decoder: {annotation_encoder_decoder_name}"
            )

        self.model = BartAsPointerNetwork.from_pretrained(
            model_name_or_path,
            # label id space
            bos_token_id=self.annotation_encoder_decoder.bos_id,
            eos_token_id=self.annotation_encoder_decoder.eos_id,
            pad_token_id=self.annotation_encoder_decoder.eos_id,
            label_ids=self.annotation_encoder_decoder.label_ids,
            # target token id space
            target_token_ids=target_token_ids,
            # mapping to better initialize the label embedding weights
            embedding_weight_mapping=embedding_weight_mapping,
            # other parameters
            use_encoder_mlp=use_encoder_mlp,
        )
        if not self.is_from_pretrained:
            self.model.resize_token_embeddings(vocab_size)
            self.model.overwrite_decoder_label_embeddings_with_mapping()

        # NOTE: This is not a ModuleDict, so this will not live on the torch device!
        self.metrics: Dict[str, AnnotationLayerMetric] = {
            stage: AnnotationLayerMetric(
                eos_id=self.annotation_encoder_decoder.eos_id,
                annotation_encoder_decoder=self.annotation_encoder_decoder,
            )
            for stage in metric_splits
        }

    def predict(self, inputs, num_beams=5, min_length=7, **kwargs) -> Dict[str, Any]:
        is_training = self.training
        self.eval()

        # num_beams=3, min_length=5, max_length=20)
        outputs = self.model.generate(
            inputs["src_tokens"], num_beams=num_beams, min_length=min_length, **kwargs
        )

        if is_training:
            self.train()
        return {"pred": outputs}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs, _ = batch
        pred = self.predict(inputs=inputs)
        return pred

    def forward(self, inputs, **kwargs):
        input_ids = inputs["src_tokens"]
        attention_mask = inputs["src_attention_mask"]
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def step(self, batch, stage: str) -> torch.FloatTensor:
        inputs, targets = batch
        if targets is None:
            raise ValueError("Targets must be provided for training or evaluation!")

        # Truncate the bos_id. The decoder input_ids will be created by the model
        # by shifting the labels one position to the right and adding the bos_id
        labels = targets["tgt_tokens"][:, 1:]

        outputs = self(inputs=inputs, labels=labels)
        loss = outputs.loss

        # show loss on each step only during training
        self.log(
            f"loss/{stage}", loss, on_step=(stage == STAGE_TRAIN), on_epoch=True, prog_bar=True
        )

        stage_metrics = self.metrics.get(stage, None)
        if stage_metrics is not None:
            prediction = self.predict(inputs)
            stage_metrics(prediction["pred"], targets["tgt_tokens"])

        return loss

    def training_step(self, batch, batch_idx) -> torch.FloatTensor:
        loss = self.step(batch, stage=STAGE_TRAIN)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.FloatTensor:
        loss = self.step(batch, stage=STAGE_VAL)

        return loss

    def test_step(self, batch, batch_idx) -> torch.FloatTensor:
        loss = self.step(batch, stage=STAGE_TEST)

        return loss

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end(stage=STAGE_TRAIN)

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(stage=STAGE_VAL)

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(stage=STAGE_TEST)

    def _on_epoch_end(self, stage: str) -> None:
        metrics = self.metrics.get(stage, None)
        if metrics is not None:
            metric_dict = metrics.get_metric(reset=True)
            metric_dict_flat = flatten_dict(d=metric_dict, sep="/")
            for k, v in metric_dict_flat.items():
                self.log(f"metric_{k}/{stage}", v, on_step=False, on_epoch=True, prog_bar=True)

    def encoder_decoder_params(self) -> Iterator[Tuple[str, Parameter]]:
        for name, param in self.named_parameters():
            if "encoder" in name or "decoder" in name:
                yield name, param

    def configure_optimizers(self) -> OptimizerLRScheduler:
        encoder_decoder_params = dict(self.encoder_decoder_params())

        # norm for not bart layer
        parameters = []
        params = {
            "lr": self.lr,
            "weight_decay": 1e-2,
            "params": [
                param
                for name, param in self.named_parameters()
                if name not in encoder_decoder_params
            ],
        }
        parameters.append(params)

        params = {
            "lr": self.lr,
            "weight_decay": 1e-2,
            "params": [
                param
                for name, param in encoder_decoder_params.items()
                if "layernorm" in name or "layer_norm" in name
            ],
        }
        parameters.append(params)

        params = {
            "lr": self.lr,
            "weight_decay": self.layernorm_decay,
            "params": [
                param
                for name, param in encoder_decoder_params.items()
                if not ("layernorm" in name or "layer_norm" in name)
            ],
        }
        parameters.append(params)

        optimizer = torch.optim.AdamW(parameters)

        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
