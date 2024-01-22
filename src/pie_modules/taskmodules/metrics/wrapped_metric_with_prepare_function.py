import logging
from typing import Any, Callable, Dict, Generic, List, TypeVar, Union

from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers.abstract import WrapperMetric

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WrappedMetricWithPrepareFunction(WrapperMetric, Generic[T]):
    """A wrapper around a metric that can be used with predictions and targets that are need to be
    prepared (e.g. un-batched) before passing them to the metric.

    Args:
        metric: The metric to wrap. It should be a subclass of torchmetrics.Metric.
        prepare_function: A function that prepares the input for the metric. It is called with
            the predictions as well as the targets.
        prepare_does_unbatch: If True, the prepare_function is expected to return an iterable of
            individual inputs. This can be used to un-batch the input before passing it to the
            wrapped metric.
    """

    def __init__(
        self,
        metric: Union[Metric, MetricCollection],
        prepare_function: Callable[[T], Any],
        prepare_does_unbatch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.prepare_function = prepare_function
        self.prepare_does_unbatch = prepare_does_unbatch

    def forward(self, prediction: T, target: T) -> Any:
        prediction_prepared = self.prepare_function(prediction)
        target_prepared = self.prepare_function(target)
        if self.prepare_does_unbatch:
            if len(prediction_prepared) != len(target_prepared):
                raise ValueError(
                    f"Number of prepared predictions ({len(prediction_prepared)}) and targets "
                    f"({len(target_prepared)}) do not match."
                )
            if len(prediction_prepared) == 0:
                raise ValueError("Empty batch.")
            results = []
            for prediction_str, target_str in zip(prediction_prepared, target_prepared):
                current_result = self.metric(prediction_str, target_str)
                results.append(current_result)
            return results
        else:
            return self.metric(prediction_prepared, target_prepared)

    def update(self, prediction: T, target: T) -> None:
        prediction_prepared = self.prepare_function(prediction)
        target_prepared = self.prepare_function(target)
        if self.prepare_does_unbatch:
            if len(prediction_prepared) != len(target_prepared):
                raise ValueError(
                    f"Number of prepared predictions ({len(prediction_prepared)}) and targets "
                    f"({len(target_prepared)}) do not match."
                )
            if len(prediction_prepared) == 0:
                raise ValueError("Empty batch.")
            for prediction_str, target_str in zip(prediction_prepared, target_prepared):
                self.metric.update(prediction_str, target_str)
        else:
            self.metric.update(prediction_prepared, target_prepared)

    def compute(self) -> Any:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()

    @property
    def metric_state(self) -> Dict[str, Union[List[Tensor], Tensor]]:
        if isinstance(self.metric, Metric):
            return self.metric.metric_state
        elif isinstance(self.metric, MetricCollection):
            return {
                metric_name: metric.metric_state for metric_name, metric in self.metric.items()
            }
        else:
            raise ValueError(f"Unsupported metric type: {type(self.metric)}")
