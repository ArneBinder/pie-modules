import logging
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_ie import AutoTaskModule
from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import FloatTensor, LongTensor, nn
from transformers import AutoConfig, AutoModelForTokenClassification, BatchEncoding
from transformers.modeling_outputs import TokenClassifierOutput
from typing_extensions import TypeAlias

from pie_modules.models.interface import RequiresTaskmoduleConfig

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
    PyTorchIEModel, RequiresModelNameOrPath, RequiresNumClasses, RequiresTaskmoduleConfig
):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        learning_rate: float = 1e-5,
        label_pad_id: int = -100,
        taskmodule_config: Optional[Dict[str, Any]] = None,
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

        # todo: use metric_val, metric_test, metric_train and remove _on_epoch_end()
        self.metrics = nn.ModuleDict()
        if taskmodule_config is not None:
            self.taskmodule = AutoTaskModule.from_config(taskmodule_config)
            # TODO: remove this once this is done in `TaskModule._from_config()`
            self.taskmodule.post_prepare()
            for stage in [TRAINING, VALIDATION, TEST]:
                stage_metric = self.taskmodule.configure_model_metric(stage=stage)
                if stage_metric is not None:
                    self.metrics[stage] = stage_metric
                else:
                    logger.warning(
                        f"The taskmodule {self.taskmodule.__class__.__name__} does not define a metric for stage "
                        f"'{stage}'."
                    )

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

    def step(
        self,
        stage: str,
        batch: ModelStepInputType,
    ) -> FloatTensor:
        inputs, targets = batch
        assert targets is not None, "targets has to be available for training"

        output = self(inputs, labels=targets)

        loss = output.loss
        # show loss on each step only during training
        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        if stage in self.metrics:
            metric = self.metrics[stage]
            predicted_tags = self.decode(
                logits=output.logits,
                attention_mask=inputs["attention_mask"],
                special_tokens_mask=inputs["special_tokens_mask"],
            )
            # TODO: do not do this here, but in the metric (current approach works just for token wise F1)
            mask = targets != self.label_pad_id
            predicted_tags_valid = predicted_tags[mask]
            targets_valid = targets[mask]
            metric(predicted_tags_valid, targets_valid)

        return loss

    def training_step(self, batch: ModelStepInputType, batch_idx: int) -> FloatTensor:
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: ModelStepInputType, batch_idx: int) -> FloatTensor:
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: ModelStepInputType, batch_idx: int) -> FloatTensor:
        return self.step(stage=TEST, batch=batch)

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

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end(stage=TRAINING)

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(stage=VALIDATION)

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(stage=TEST)

    def _on_epoch_end(self, stage: str) -> None:
        if stage in self.metrics:
            metric = self.metrics[stage]
            value = metric.compute()
            self.log(
                f"metric/{type(metric).__name__}/{stage}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            metric.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
