import abc
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from pytorch_ie import PyTorchIEModel
from typing_extensions import TypeAlias

from pie_modules.models.mixins import WithMetricsFromTaskModule

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
TargetType = TypeVar("TargetType")
StepInputType: TypeAlias = Tuple[
    InputType,
    Optional[TargetType],
]
StepOutputType = TypeVar("StepOutputType")

TRAINING = "train"
VALIDATION = "val"
TEST = "test"


class DefaultModel(
    PyTorchIEModel,
    WithMetricsFromTaskModule[InputType, TargetType, OutputType],
    Generic[InputType, OutputType, TargetType, StepOutputType],
    abc.ABC,
):
    def __init__(
        self,
        taskmodule_config: Optional[Dict[str, Any]] = None,
        metric_stages: List[str] = [TRAINING, VALIDATION, TEST],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.setup_metrics(metric_stages=metric_stages, taskmodule_config=taskmodule_config)

    def get_loss_from_outputs(self, outputs: OutputType) -> StepOutputType:
        if hasattr(outputs, "loss"):
            return outputs.loss
        else:
            raise ValueError(
                f"The model {self.__class__.__name__} does not define a 'loss' attribute in its output, "
                "so the loss cannot be automatically extracted from the outputs. Please override the"
                "get_loss_from_outputs() method for this model."
            )

    def log_loss(self, stage: str, loss: StepOutputType) -> None:
        # show loss on each step only during training
        self.log(
            f"loss/{stage}",
            loss,
            on_step=(stage == TRAINING),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def _step(
        self,
        stage: str,
        batch: StepInputType,
    ) -> StepOutputType:
        inputs, targets = batch
        assert targets is not None, "targets has to be available for training"

        outputs = self(inputs=inputs, targets=targets)

        self.update_metric(inputs=inputs, outputs=outputs, targets=targets, stage=stage)

        loss = self.get_loss_from_outputs(outputs=outputs)
        self.log_loss(stage=stage, loss=loss)

        return loss

    def training_step(self, batch: StepInputType, batch_idx: int) -> StepOutputType:
        return self._step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: StepInputType, batch_idx: int) -> StepOutputType:
        return self._step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: StepInputType, batch_idx: int) -> StepOutputType:
        return self._step(stage=TEST, batch=batch)

    def predict_step(
        self, batch: StepInputType, batch_idx: int, dataloader_idx: int
    ) -> TargetType:
        inputs, targets = batch
        return self.predict(inputs=inputs)
