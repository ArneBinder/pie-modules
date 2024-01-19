import abc
import logging
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pytorch_ie import AutoTaskModule
from pytorch_lightning import LightningModule
from torchmetrics import Metric, MetricCollection

from pie_modules.models.interface import RequiresTaskmoduleConfig

InputType = TypeVar("InputType")
TargetType = TypeVar("TargetType")
OutputType = TypeVar("OutputType")

TRAINING = "train"
VALIDATION = "val"
TEST = "test"

logger = logging.getLogger(__name__)


class WithMetricsFromTaskModule(
    LightningModule, RequiresTaskmoduleConfig, Generic[InputType, TargetType, OutputType], abc.ABC
):
    """A mixin for LightningModules that adds metrics from a taskmodule.

    The metrics are added to the LightningModule as attributes with the names metric_{stage} via
    setup_metrics method, where stage is one of "train", "val", or "test". The metrics are updated
    with the update_metric method and logged with the on_{stage}_epoch_end methods.
    """

    def setup_metrics(
        self, metric_stages: List[str], taskmodule_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Setup metrics for the given stages. If taskmodule_config is provided, the metrics are
        configured from the taskmodule. Otherwise, no metrics are available.

        Args:
            metric_stages: The stages for which to setup metrics. Must be one of "train", "val", or
                "test".
            taskmodule_config: The config for the taskmodule which can be obtained from the
                taskmodule.config property.
        """

        for stage in [TRAINING, VALIDATION, TEST]:
            self._set_metric(stage=stage, metric=None)
        if taskmodule_config is not None:
            taskmodule = AutoTaskModule.from_config(taskmodule_config)
            for stage in metric_stages:
                if stage not in [TRAINING, VALIDATION, TEST]:
                    raise ValueError(
                        f'metric_stages must only contain the values "{TRAINING}", "{VALIDATION}", and "{TEST}".'
                    )
                metric = taskmodule.configure_model_metric(stage=stage)
                if metric is not None:
                    self._set_metric(stage=stage, metric=metric)
                else:
                    logger.warning(
                        f"The taskmodule {taskmodule.__class__.__name__} does not define a metric for stage "
                        f"'{stage}'."
                    )
        else:
            logger.warning("No taskmodule_config was provided. Metrics will not be available.")

    def _get_metric(self, stage: str) -> Optional[Union[Metric, MetricCollection]]:
        return getattr(self, f"metric_{stage}")

    def _set_metric(self, stage: str, metric: Optional[Union[Metric, MetricCollection]]) -> None:
        setattr(self, f"metric_{stage}", metric)

    @abc.abstractmethod
    def predict(self, inputs: InputType, **kwargs) -> TargetType:
        """Predict the target for the given inputs."""
        pass

    @abc.abstractmethod
    def decode(self, inputs: InputType, outputs: OutputType) -> TargetType:
        """Decode the outputs of the model into the target format."""
        pass

    def update_metric(
        self,
        stage: str,
        inputs: InputType,
        targets: TargetType,
        outputs: Optional[OutputType] = None,
    ) -> None:
        """Update the metric for the given stage. If outputs is provided, the predictions are
        decoded from the outputs. Otherwise, the predictions are obtained by directly calling the
        predict method with the inputs (note that this causes the model to be called a second
        time). Finally, the metric is updated with the predictions and targets.

        Args:
            stage: The stage for which to update the metric. Must be one of "train", "val", or "test".
            inputs: The inputs to the model.
            targets: The targets for the inputs.
            outputs: The outputs of the model. They are decoded into predictions if provided. If
                outputs is None, the predictions are obtained by directly calling the predict method
                on the inputs.
        """

        metric = self._get_metric(stage=stage)
        if metric is not None:
            if outputs is not None:
                predictions = self.decode(inputs=inputs, outputs=outputs)
            else:
                predictions = self.predict(inputs=inputs)
            metric.update(predictions, targets)

    def log_metric(self, stage: str, reset: bool = True) -> None:
        """Log the metric for the given stage and reset it."""

        metric = self._get_metric(stage=stage)
        if metric is not None:
            values = metric.compute()
            log_kwargs = {"on_step": False, "on_epoch": True, "sync_dist": True}
            if isinstance(values, dict):
                for key, value in values.items():
                    self.log(f"metric/{key}/{stage}", value, **log_kwargs)
            else:
                metric_name = getattr(metric, "name", None) or type(metric).__name__
                self.log(f"metric/{metric_name}/{stage}", values, **log_kwargs)
            if reset:
                metric.reset()
