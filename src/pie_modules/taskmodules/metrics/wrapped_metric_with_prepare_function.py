import logging
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.wrappers.abstract import WrapperMetric

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WrappedMetricWithPrepareFunction(WrapperMetric, Generic[T]):
    """A wrapper around a metric that can be used with predictions and targets that are need to be
    prepared (e.g. un-batched) before passing them to the metric.

    Args:
        prepare_function: A function that prepares the input for the metric. It is called with
            the predictions as well as the targets.
        metric: The metric to wrap. It should be a subclass of torchmetrics.Metric.
    """

    def __init__(
        self,
        prepare_function: Callable[[T], Any],
        metric: Metric,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.prepare_function = prepare_function
        self.metric = metric
        self.name = name or type(self.metric).__name__

    def update(self, predictions: T, targets: T):
        super().update(predictions, targets)

    def compute(self) -> Any:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()

    @property
    def metric_state(self) -> Dict[str, Union[List[Tensor], Tensor]]:
        return self.metric.metric_state

    def forward(self, predictions: T, targets: T) -> Any:
        prediction_prepared = self.prepare_function(predictions)
        target_prepared = self.prepare_function(targets)
        return self.metric.forward(prediction_prepared, target_prepared)
