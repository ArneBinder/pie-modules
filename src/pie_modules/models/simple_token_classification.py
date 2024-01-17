import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import FloatTensor, LongTensor
from torchmetrics import Metric
from transformers import AutoConfig, AutoModelForTokenClassification, BatchEncoding
from transformers.modeling_outputs import TokenClassifierOutput
from typing_extensions import TypeAlias

from pie_modules.models.mixins import WithMetricsFromTaskModule

ModelInputsType: TypeAlias = BatchEncoding
ModelTargetsType: TypeAlias = LongTensor
ModelStepInputType: TypeAlias = Tuple[
    ModelInputsType,
    Optional[ModelTargetsType],
]
ModelOutputType: TypeAlias = LongTensor

TRAINING = "train"
VALIDATION = "val"
TEST = "test"

logger = logging.getLogger(__name__)


@PyTorchIEModel.register()
class SimpleTokenClassificationModel(
    PyTorchIEModel, RequiresModelNameOrPath, RequiresNumClasses, WithMetricsFromTaskModule
):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        learning_rate: float = 1e-5,
        label_pad_id: int = -100,
        taskmodule_config: Optional[Dict[str, Any]] = None,
        metric_stages: List[str] = [TRAINING, VALIDATION, TEST],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.label_pad_id = label_pad_id
        self.num_classes = num_classes

        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_classes)
        if self.is_from_pretrained:
            self.model = AutoModelForTokenClassification.from_config(config=config)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name_or_path, config=config
            )
        self.setup_metrics(metric_stages=metric_stages, taskmodule_config=taskmodule_config)

    def forward(
        self, inputs: ModelInputsType, labels: Optional[torch.LongTensor] = None
    ) -> TokenClassifierOutput:
        inputs_without_special_tokens_mask = {
            k: v for k, v in inputs.items() if k != "special_tokens_mask"
        }
        return self.model(labels=labels, **inputs_without_special_tokens_mask)

    def decode(
        self,
        logits: FloatTensor,
        attention_mask: LongTensor,
        special_tokens_mask: LongTensor,
    ) -> LongTensor:
        # get the max index for each token from the logits
        tags_tensor = torch.argmax(logits, dim=-1).to(torch.long)

        # mask out the padding and special tokens
        tags_tensor = tags_tensor.masked_fill(attention_mask == 0, self.label_pad_id)

        # mask out the special tokens
        tags_tensor = tags_tensor.masked_fill(special_tokens_mask == 1, self.label_pad_id)
        return tags_tensor

    def _step(
        self,
        stage: str,
        batch: ModelStepInputType,
        metric: Optional[Metric] = None,
    ) -> FloatTensor:
        inputs, targets = batch
        assert targets is not None, "targets has to be available for training"

        output = self(inputs, labels=targets)

        loss = output.loss
        # show loss on each step only during training
        self.log(
            f"loss/{stage}",
            loss,
            on_step=(stage == TRAINING),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if metric is not None:
            predicted_tags = self.decode(
                logits=output.logits,
                attention_mask=inputs["attention_mask"],
                special_tokens_mask=inputs["special_tokens_mask"],
            )
            metric.update(predicted_tags, targets)

        return loss

    def training_step(self, batch: ModelStepInputType, batch_idx: int) -> FloatTensor:
        return self._step(stage=TRAINING, batch=batch, metric=self.get_metric(stage=TRAINING))

    def validation_step(self, batch: ModelStepInputType, batch_idx: int) -> FloatTensor:
        return self._step(stage=VALIDATION, batch=batch, metric=self.get_metric(stage=VALIDATION))

    def test_step(self, batch: ModelStepInputType, batch_idx: int) -> FloatTensor:
        return self._step(stage=TEST, batch=batch, metric=self.get_metric(stage=TEST))

    def _on_epoch_end(self, stage: str, metric: Optional[Metric] = None) -> None:
        if metric is not None:
            values = metric.compute()
            log_kwargs = {"on_step": False, "on_epoch": True, "sync_dist": True}
            if isinstance(values, dict):
                for key, value in values.items():
                    self.log(f"metric/{key}/{stage}", value, **log_kwargs)
            else:
                metric_name = getattr(metric, "name", None) or type(metric).__name__
                self.log(f"metric/{metric_name}/{stage}", values, **log_kwargs)
            metric.reset()

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end(stage=TRAINING, metric=self.metric_train)

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(stage=VALIDATION, metric=self.metric_val)

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(stage=TEST, metric=self.metric_test)

    def predict(self, inputs: ModelInputsType, **kwargs) -> ModelOutputType:
        output = self(inputs)
        predicted_tags = self.decode(
            logits=output.logits,
            attention_mask=inputs["attention_mask"],
            special_tokens_mask=inputs["special_tokens_mask"],
        )
        return predicted_tags

    def predict_step(
        self, batch: ModelStepInputType, batch_idx: int, dataloader_idx: int
    ) -> LongTensor:
        inputs, targets = batch
        return self.predict(inputs=inputs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
