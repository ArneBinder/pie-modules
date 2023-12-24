import copy
import logging
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional

import torch
from pytorch_ie import TaskModule
from pytorch_ie.core import PyTorchIEModel
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchmetrics import Metric
from transformers import get_linear_schedule_with_warmup

from ..taskmodules import PointerNetworkTaskModuleForEnd2EndRE
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


@PyTorchIEModel.register()
class SimplePointerNetworkModel(PyTorchIEModel):
    def __init__(
        self,
        base_model_config: Dict[str, Any],
        taskmodule_config: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        # metrics
        metric_splits: List[str] = [STAGE_VAL, STAGE_TEST],
        metric_intervals: Optional[Dict[str, int]] = None,
        use_prediction_for_metrics: bool = True,
        # optimizer / scheduler
        layernorm_decay: Optional[float] = 0.001,  # deprecated
        warmup_proportion: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if layernorm_decay is not None:
            logger.warning(
                "layernorm_decay is deprecated, please use base_model_kwargs.encoder_layer_norm_decay instead!"
            )
            base_model_config["encoder_layer_norm_decay"] = layernorm_decay
        self.save_hyperparameters(ignore=["layernorm_decay"])

        # optimizer / scheduler
        self.warmup_proportion = warmup_proportion

        # can be used to override the generation kwargs from the generation config that gets constructed from the
        # BartAsPointerNetwork config
        self.generation_kwargs = generation_kwargs or {}

        # TODO: Use AutoModelAsPointerNetwork when it is available
        self.model = BartAsPointerNetwork.from_pretrained(
            # generation
            forced_bos_token_id=None,  # to disable ForcedBOSTokenLogitsProcessor
            forced_eos_token_id=None,  # to disable ForcedEOSTokenLogitsProcessor
            **base_model_config,
        )

        self.model.adjust_original_model()

        self.use_prediction_for_metrics = use_prediction_for_metrics
        self.metric_intervals = metric_intervals or {}
        self.metrics: Dict[str, Metric] = {}
        if taskmodule_config is not None:
            # TODO: use AutoTaskModule.from_config() when it is available
            taskmodule_kwargs = copy.copy(taskmodule_config)
            taskmodule_kwargs.pop(TaskModule.config_type_key)
            taskmodule = PointerNetworkTaskModuleForEnd2EndRE(**taskmodule_kwargs)
            taskmodule.post_prepare()
            if not isinstance(taskmodule, HasBuildMetric):
                raise Exception(
                    f"taskmodule {taskmodule} does not implement HasBuildMetric interface"
                )
            # NOTE: This is not a ModuleDict, so this will not live on the torch device!
            self.metrics = {stage: taskmodule.build_metric(stage) for stage in metric_splits}

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

        stage_metrics = self.metrics.get(stage, None)
        metric_interval = self.metric_intervals.get(stage, 1)
        if stage_metrics is not None and (batch_idx + 1) % metric_interval == 0:
            if self.use_prediction_for_metrics:
                prediction = self.predict(inputs)
            else:
                # construct prediction from the model output
                logits = outputs.logits
                # get the indices (these are without the initial bos_ids, see above)
                indices = torch.argmax(logits, dim=-1)
                # re-add the bos_ids
                prediction_ids = torch.cat([targets["tgt_tokens"][:, :1], indices], dim=-1)
                prediction = {"pred": prediction_ids}
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
                # TODO: consider https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metriccollection
                #  and self.log_dict()
                metric_dict_flat = flatten_dict(d=metric_dict, sep="/")
                for k, v in metric_dict_flat.items():
                    self.log(f"metric_{k}/{stage}", v, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.model.configure_optimizer()

        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
