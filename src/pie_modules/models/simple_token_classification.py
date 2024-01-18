import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import FloatTensor, LongTensor
from transformers import AutoConfig, AutoModelForTokenClassification, BatchEncoding
from transformers.modeling_outputs import TokenClassifierOutput
from typing_extensions import TypeAlias

from pie_modules.models.mixins import WithMetricsFromTaskModule

ModelInputType: TypeAlias = BatchEncoding
ModelTargetType: TypeAlias = LongTensor
ModelStepInputType: TypeAlias = Tuple[
    ModelInputType,
    Optional[ModelTargetType],
]
ModelOutputType: TypeAlias = TokenClassifierOutput

TRAINING = "train"
VALIDATION = "val"
TEST = "test"

logger = logging.getLogger(__name__)


@PyTorchIEModel.register()
class SimpleTokenClassificationModel(
    PyTorchIEModel,
    RequiresModelNameOrPath,
    RequiresNumClasses,
    WithMetricsFromTaskModule[ModelInputType, ModelTargetType, ModelOutputType],
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
        self, inputs: ModelInputType, targets: Optional[ModelTargetType] = None
    ) -> ModelOutputType:
        inputs_without_special_tokens_mask = {
            k: v for k, v in inputs.items() if k != "special_tokens_mask"
        }
        return self.model(labels=targets, **inputs_without_special_tokens_mask)

    def decode(
        self,
        inputs: ModelInputType,
        outputs: ModelInputType,
    ) -> ModelTargetType:
        # get the max index for each token from the logits
        tags_tensor = torch.argmax(outputs.logits, dim=-1).to(torch.long)

        # mask out the padding and special tokens
        tags_tensor = tags_tensor.masked_fill(inputs["attention_mask"] == 0, self.label_pad_id)

        # mask out the special tokens
        tags_tensor = tags_tensor.masked_fill(
            inputs["special_tokens_mask"] == 1, self.label_pad_id
        )
        return tags_tensor

    def _step(
        self,
        stage: str,
        batch: ModelStepInputType,
    ) -> FloatTensor:
        inputs, targets = batch
        assert targets is not None, "targets has to be available for training"

        output = self(inputs, targets=targets)

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
        self.update_metric(inputs=inputs, outputs=output, targets=targets, stage=stage)

        return loss

    def training_step(self, batch: ModelStepInputType, batch_idx: int) -> FloatTensor:
        return self._step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: ModelStepInputType, batch_idx: int) -> FloatTensor:
        return self._step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: ModelStepInputType, batch_idx: int) -> FloatTensor:
        return self._step(stage=TEST, batch=batch)

    def predict(self, inputs: ModelInputType, **kwargs) -> ModelTargetType:
        outputs = self(inputs)
        predicted_tags = self.decode(inputs=inputs, outputs=outputs)
        return predicted_tags

    def predict_step(
        self, batch: ModelStepInputType, batch_idx: int, dataloader_idx: int
    ) -> ModelTargetType:
        inputs, targets = batch
        return self.predict(inputs=inputs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
