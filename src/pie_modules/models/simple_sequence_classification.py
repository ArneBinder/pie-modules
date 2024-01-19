import logging
from typing import Any, Iterator, MutableMapping, Optional, Tuple

import torch.nn
import torchmetrics
from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses
from torch import Tensor, nn
from torch.nn import Parameter
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing_extensions import TypeAlias

# The input to the forward method of this model. It is passed to
# the base transformer model. Can also contain additional arguments
# for the pooler (these need to be prefixed with "pooler_").
ModelInputType: TypeAlias = MutableMapping[str, Any]
# A dict with a single key "logits".
ModelOutputType: TypeAlias = SequenceClassifierOutputWithPast
# This contains the input and target tensors for a single training step.
ModelStepInputType: TypeAlias = Tuple[
    ModelInputType,  # input
    Optional[Tensor],  # targets
]

# stage names
TRAINING = "train"
VALIDATION = "val"
TEST = "test"


logger = logging.getLogger(__name__)


@PyTorchIEModel.register()
class SimpleSequenceClassificationModel(
    PyTorchIEModel, RequiresModelNameOrPath, RequiresNumClasses
):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        tokenizer_vocab_size: Optional[int] = None,
        ignore_index: Optional[int] = None,
        learning_rate: float = 1e-5,
        task_learning_rate: Optional[float] = None,
        warmup_proportion: float = 0.1,
        multi_label: bool = False,
        freeze_base_model: bool = False,
        base_model_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion
        self.freeze_base_model = freeze_base_model

        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_classes)
        if self.is_from_pretrained:
            self.model = AutoModelForSequenceClassification.from_config(config=config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, config=config
            )

        self.base_model_prefix = base_model_prefix or self.model.base_model_prefix

        if tokenizer_vocab_size is not None:
            self.model.resize_token_embeddings(tokenizer_vocab_size)

        if self.freeze_base_model:
            for name, param in self.base_model_named_parameters():
                param.requires_grad = False

        self.f1 = nn.ModuleDict(
            {
                f"stage_{stage}": torchmetrics.F1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    task="multilabel" if multi_label else "multiclass",
                )
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )

    def base_model_named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        base_model: torch.nn.Module = getattr(self.model, self.base_model_prefix, None)
        if base_model is None:
            raise ValueError(
                f"Base model with prefix '{self.base_model_prefix}' not found in {type(self.model).__name__}"
            )
        return base_model.named_parameters(prefix=f"{prefix}model.{self.base_model_prefix}")

    def task_named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        base_model_parameter_names = dict(self.base_model_named_parameters(prefix=prefix)).keys()
        for name, param in self.named_parameters(prefix=prefix):
            if name not in base_model_parameter_names:
                yield name, param

    def forward(self, inputs: ModelInputType) -> ModelOutputType:
        return self.model(**inputs)

    def step(self, stage: str, batch: ModelStepInputType):
        inputs, target = batch
        assert target is not None, "target has to be available for training"

        all_inputs = dict(inputs)
        all_inputs["labels"] = target
        output = self(all_inputs)
        loss = output.loss

        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        logits = output.logits
        f1 = self.f1[f"stage_{stage}"]
        f1(logits, target)
        # self.log(f"{stage}/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: ModelStepInputType, batch_idx: int):
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: ModelStepInputType, batch_idx: int):
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: ModelStepInputType, batch_idx: int):
        return self.step(stage=TEST, batch=batch)

    def configure_optimizers(self):
        if self.task_learning_rate is not None:
            base_model_params = [param for name, param in self.base_model_named_parameters()]
            task_params = [param for name, param in self.task_named_parameters()]
            optimizer = AdamW(
                [
                    {"params": base_model_params, "lr": self.learning_rate},
                    {"params": task_params, "lr": self.task_learning_rate},
                ]
            )
        else:
            optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
