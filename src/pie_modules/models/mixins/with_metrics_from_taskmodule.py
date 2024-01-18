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
    def setup_metrics(
        self, metric_stages: List[str], taskmodule_config: Optional[Dict[str, Any]] = None
    ) -> None:
        for stage in [TRAINING, VALIDATION, TEST]:
            self.set_metric(stage=stage, metric=None)
        if taskmodule_config is not None:
            taskmodule = AutoTaskModule.from_config(taskmodule_config)
            for stage in metric_stages:
                if stage not in [TRAINING, VALIDATION, TEST]:
                    raise ValueError(
                        f'metric_stages must only contain the values "{TRAINING}", "{VALIDATION}", and "{TEST}".'
                    )
                metric = taskmodule.configure_model_metric(stage=stage)
                if metric is not None:
                    self.set_metric(stage=stage, metric=metric)
                else:
                    logger.warning(
                        f"The taskmodule {taskmodule.__class__.__name__} does not define a metric for stage "
                        f"'{stage}'."
                    )
        else:
            logger.warning("No taskmodule_config was provided. Metrics will not be available.")

    def get_metric(self, stage: str) -> Optional[Union[Metric, MetricCollection]]:
        return getattr(self, f"metric_{stage}")

    def set_metric(self, stage: str, metric: Optional[Union[Metric, MetricCollection]]) -> None:
        setattr(self, f"metric_{stage}", metric)

    def predict(self, inputs: InputType, **kwargs) -> TargetType:
        outputs = self(inputs)
        decoded_outputs = self.decode(inputs=inputs, outputs=outputs)
        return decoded_outputs

    @abc.abstractmethod
    def decode(self, inputs: InputType, outputs: OutputType) -> TargetType:
        pass

    def update_metric(
        self,
        stage: str,
        inputs: InputType,
        targets: TargetType,
        outputs: Optional[OutputType] = None,
    ) -> None:
        metric = self.get_metric(stage=stage)
        if metric is not None:
            if outputs is not None:
                predictions = self.decode(inputs=inputs, outputs=outputs)
            else:
                predictions = self.predict(inputs=inputs)
            metric.update(predictions, targets)

    def _on_epoch_end(self, stage: str) -> None:
        metric = self.get_metric(stage=stage)
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
        self._on_epoch_end(stage=TRAINING)

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(stage=VALIDATION)

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(stage=TEST)
