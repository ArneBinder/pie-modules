from typing import Any, Dict, MutableMapping, Optional, Tuple

from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.nn import ModuleDict, functional
from torch.optim import Adam
from torchmetrics import F1Score
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    BatchEncoding,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from typing_extensions import TypeAlias

from pie_modules.models.interface import RequiresMaxInputLength

BatchOutput: TypeAlias = Dict[str, Any]

# The input to the forward method of this model. It is passed to
# the base transformer model.
ModelInputType: TypeAlias = MutableMapping[str, Any]
# The output of the forward method of this model.
ModelOutputType: TypeAlias = QuestionAnsweringModelOutput
# The input to the step methods, i.e. training_step, validation_step, test_step.
# It contains the input and target tensors for a single training step.
StepBatchEncoding: TypeAlias = Tuple[
    ModelInputType,
    Optional[Dict[str, Tensor]],
]


TRAINING = "train"
VALIDATION = "val"
TEST = "test"


@PyTorchIEModel.register()
class SimpleExtractiveQuestionAnsweringModel(
    PyTorchIEModel, RequiresModelNameOrPath, RequiresMaxInputLength
):
    """A PIE model for extractive question answering. It is a simple Pytorch-Lightning module that
    wraps around a question answering model from the Huggingface transformers library. The
    ExtractiveQuestionAnsweringTaskModule can be used create the input and target encodings as well
    as to decode the model output.

    Args:
        model_name_or_path: The name (Huggingface Hub model identifier) or local path of the model to use.
        max_input_length: The maximum length of the input sequence. Required for metric calculation.
        learning_rate: The learning rate to use for training. Defaults to 1e-5.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_input_length: int,
        learning_rate: float = 1e-5,
        warmup_proportion: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.max_input_length = max_input_length

        config = AutoConfig.from_pretrained(model_name_or_path)
        if self.is_from_pretrained:
            self.model = AutoModelForQuestionAnswering.from_config(config=config)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                model_name_or_path, config=config
            )

        self.f1_start: Dict[str, F1Score] = ModuleDict(
            {
                f"stage_{stage}": F1Score(task="multiclass", num_classes=max_input_length)
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )
        self.f1_end: Dict[str, F1Score] = ModuleDict(
            {
                f"stage_{stage}": F1Score(task="multiclass", num_classes=max_input_length)
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )

    def forward(self, inputs: BatchEncoding) -> ModelOutputType:
        return self.model(**inputs)

    def step(
        self,
        stage: str,
        batch: StepBatchEncoding,
    ) -> Tensor:
        inputs, targets = batch
        if targets is None:
            raise ValueError("targets has to be available for training, but it is None")

        output = self({**inputs, **targets})

        loss = output.loss
        # show loss on each step only during training
        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        start_positions = targets["start_positions"]
        end_positions = targets["end_positions"]
        start_logits = output.start_logits
        end_logits = output.end_logits

        sequence_length = inputs["input_ids"].size(1)
        f1_start = self.f1_end[f"stage_{stage}"]
        # We need to pad the logits to the max_input_length, otherwise the F1 metric complains
        # that the shape does not match the num_classes.
        start_logits_padded = functional.pad(
            start_logits, (0, self.max_input_length - sequence_length), value=float("-inf")
        )
        f1_start(start_logits_padded, start_positions)
        self.log(
            f"{stage}/f1_start",
            f1_start,
            on_step=(stage == TRAINING),
            on_epoch=True,
            prog_bar=True,
        )
        f1_end = self.f1_end[f"stage_{stage}"]
        # We need to pad the logits to the max_input_length, otherwise the F1 metric complains
        # that the shape does not match the num_classes.
        end_logits_padded = functional.pad(
            end_logits, (0, self.max_input_length - sequence_length), value=float("-inf")
        )
        f1_end(end_logits_padded, end_positions)
        self.log(
            f"{stage}/f1_end", f1_end, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True
        )
        # log f1 as simple average of start and end f1. we need to call compute() on the metric to get
        # the actual value, otherwise lightning complains that there is no model attribute with name "f1"
        f1_value = (f1_start.compute() + f1_end.compute()) / 2
        self.log(f"{stage}/f1", f1_value, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: StepBatchEncoding, batch_idx: int) -> Tensor:
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: StepBatchEncoding, batch_idx: int) -> Tensor:
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: StepBatchEncoding, batch_idx: int) -> Tensor:
        return self.step(stage=TEST, batch=batch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
