import copy
import logging
from collections.abc import MutableMapping
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from pytorch_ie import TaskModule
from pytorch_ie.core import PyTorchIEModel
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.nn import Parameter
from torchmetrics import Metric
from transformers import get_linear_schedule_with_warmup

from ..taskmodules import PointerNetworkTaskModule
from ..taskmodules.common import HasBuildMetric
from .components.pointer_network.bart_as_pointer_network import BartAsPointerNetwork

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


def get_layer_norm_parameters(
    named_parameters: Iterator[Tuple[str, Parameter]]
) -> Iterator[Parameter]:
    return (
        param for name, param in named_parameters if "layernorm" in name or "layer_norm" in name
    )


def get_non_layer_norm_parameters(
    named_parameters: Iterator[Tuple[str, Parameter]]
) -> Iterator[Parameter]:
    return (
        param
        for name, param in named_parameters
        if not ("layernorm" in name or "layer_norm" in name)
    )


@PyTorchIEModel.register()
class SimplePointerNetworkModel(PyTorchIEModel):
    def __init__(
        self,
        model_name_or_path: str,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        label_ids: List[int],
        target_token_ids: List[int],
        vocab_size: int,
        embedding_weight_mapping: Optional[Dict[int, List[int]]] = None,
        use_encoder_mlp: bool = False,
        taskmodule_config: Optional[Dict[str, Any]] = None,
        metric_splits: List[str] = [STAGE_VAL, STAGE_TEST],
        metric_intervals: Optional[Dict[str, int]] = None,
        # optimizer / scheduler
        lr: float = 5e-5,
        weight_decay: float = 1e-2,
        head_decay: Optional[float] = None,
        shared_decay: Optional[float] = None,
        encoder_layer_norm_decay: Optional[float] = 0.001,
        decoder_layer_norm_decay: Optional[float] = None,
        layernorm_decay: Optional[float] = 0.001,  # deprecated
        warmup_proportion: float = 0.0,
        # generation
        max_length: int = 512,
        num_beams: int = 4,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if layernorm_decay is not None:
            logger.warning(
                "layernorm_decay is deprecated, please use encoder_layernorm_decay instead!"
            )
            encoder_layer_norm_decay = layernorm_decay
        self.save_hyperparameters(ignore=["layernorm_decay"])

        # optimizer / scheduler
        self.lr = lr
        self.weight_decay = weight_decay
        self.head_decay = head_decay if head_decay is not None else self.weight_decay
        self.shared_decay = shared_decay if shared_decay is not None else self.weight_decay
        self.encoder_layer_norm_decay = (
            encoder_layer_norm_decay if encoder_layer_norm_decay is not None else self.weight_decay
        )
        self.decoder_layer_norm_decay = (
            decoder_layer_norm_decay if decoder_layer_norm_decay is not None else self.weight_decay
        )
        self.warmup_proportion = warmup_proportion

        # can be used to override the generation kwargs from the generation config that gets constructed from the
        # BartAsPointerNetwork config
        self.generation_kwargs = generation_kwargs or {}

        self.model = BartAsPointerNetwork.from_pretrained(
            model_name_or_path,
            # label id space (bos/eos/pad_token_id are also used for generation)
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            label_ids=label_ids,
            # target token id space
            target_token_ids=target_token_ids,
            # mapping to better initialize the label embedding weights
            embedding_weight_mapping=embedding_weight_mapping,
            # other parameters
            use_encoder_mlp=use_encoder_mlp,
            # generation
            forced_bos_token_id=None,  # to disable ForcedBOSTokenLogitsProcessor
            forced_eos_token_id=None,  # to disable ForcedEOSTokenLogitsProcessor
            max_length=max_length,
            num_beams=num_beams,
        )

        self.model.resize_token_embeddings(vocab_size)

        if not self.is_from_pretrained:
            self.model.overwrite_decoder_label_embeddings_with_mapping()

        self.metrics: Optional[Dict[str, Metric]]
        if taskmodule_config is not None:
            taskmodule_kwargs = copy.copy(taskmodule_config)
            taskmodule_kwargs.pop(TaskModule.config_type_key)
            taskmodule = PointerNetworkTaskModule(**taskmodule_kwargs)
            taskmodule.post_prepare()
            if not isinstance(taskmodule, HasBuildMetric):
                raise Exception(
                    f"taskmodule {taskmodule} does not implement HasBuildMetric interface"
                )
            # NOTE: This is not a ModuleDict, so this will not live on the torch device!
            self.metrics = {stage: taskmodule.build_metric(stage) for stage in metric_splits}
        else:
            self.metrics = None
        self.metric_intervals = metric_intervals or {}

    def predict(self, inputs, **kwargs) -> Dict[str, Any]:
        is_training = self.training
        self.eval()

        generation_kwargs = copy.deepcopy(self.generation_kwargs)
        generation_kwargs.update(kwargs)
        outputs = self.model.generate(inputs["src_tokens"], **generation_kwargs)

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

    def step(self, batch, stage: str, batch_idx: int) -> torch.FloatTensor:
        inputs, targets = batch
        if targets is None:
            raise ValueError("Targets must be provided for training or evaluation!")

        # Truncate the bos_id. The decoder input_ids will be created by the model
        # by shifting the labels one position to the right and adding the bos_id
        labels = targets["tgt_tokens"][:, 1:]
        decoder_attention_mask = targets["tgt_attention_mask"][:, 1:]

        outputs = self(inputs=inputs, labels=labels, decoder_attention_mask=decoder_attention_mask)
        loss = outputs.loss

        # show loss on each step only during training
        self.log(
            f"loss/{stage}", loss, on_step=(stage == STAGE_TRAIN), on_epoch=True, prog_bar=True
        )

        if self.metrics is not None:
            stage_metrics = self.metrics.get(stage, None)
            metric_interval = self.metric_intervals.get(stage, 1)
            if stage_metrics is not None and (batch_idx + 1) % metric_interval == 0:
                prediction = self.predict(inputs)
                # the format of expected needs to be the same as the format of prediction
                stage_metrics.update(prediction, {"pred": targets["tgt_tokens"]})

        return loss

    def training_step(self, batch, batch_idx) -> torch.FloatTensor:
        loss = self.step(batch, stage=STAGE_TRAIN, batch_idx=batch_idx)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.FloatTensor:
        loss = self.step(batch, stage=STAGE_VAL, batch_idx=batch_idx)

        return loss

    def test_step(self, batch, batch_idx) -> torch.FloatTensor:
        loss = self.step(batch, stage=STAGE_TEST, batch_idx=batch_idx)

        return loss

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end(stage=STAGE_TRAIN)

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(stage=STAGE_VAL)

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(stage=STAGE_TEST)

    def _on_epoch_end(self, stage: str) -> None:
        if self.metrics is not None:
            metrics = self.metrics.get(stage, None)
            if metrics is not None:
                metric_dict = metrics.compute()
                metrics.reset()
                metric_dict_flat = flatten_dict(d=metric_dict, sep="/")
                for k, v in metric_dict_flat.items():
                    self.log(f"metric_{k}/{stage}", v, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        parameters = []

        # head parameters
        params = {
            "lr": self.lr,
            "weight_decay": self.head_decay,
            "params": dict(self.model.head_named_params()).values(),
        }
        parameters.append(params)

        # decoder only layer norm parameters
        params = {
            "lr": self.lr,
            "weight_decay": self.decoder_layer_norm_decay,
            "params": get_layer_norm_parameters(self.model.decoder_only_named_params()),
        }
        parameters.append(params)

        # decoder only other parameters
        params = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "params": get_non_layer_norm_parameters(self.model.decoder_only_named_params()),
        }
        parameters.append(params)

        # encoder only layer norm parameters
        params = {
            "lr": self.lr,
            "weight_decay": self.encoder_layer_norm_decay,
            "params": get_layer_norm_parameters(self.model.encoder_only_named_params()),
        }
        parameters.append(params)

        # encoder only other parameters
        params = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "params": get_non_layer_norm_parameters(self.model.encoder_only_named_params()),
        }
        parameters.append(params)

        # encoder-decoder shared parameters
        params = {
            "lr": self.lr,
            "weight_decay": self.shared_decay,
            "params": dict(self.model.encoder_decoder_shared_named_params()).values(),
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
