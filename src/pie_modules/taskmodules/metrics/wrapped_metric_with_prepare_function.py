import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Generic, TypeVar

from torchmetrics import Metric, MetricCollection

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WrappedMetricWithPrepareFunction(MetricCollection, Generic[T]):
    """A wrapper around a metric that can be used with predictions and targets that are need to be
    prepared (e.g. un-batched) before passing them to the metric.

    Args:
        metric: The metric to wrap. It should be a subclass of torchmetrics.Metric.
        prepare_function: A function that prepares the input for the metric. It is called with
            the predictions as well as the targets.
        prepare_does_unbatch: If True, the prepare_function is expected to return an iterable of
            individual inputs. This is used to unbatch the input before passing it to the wrapped
            metric.
    """

    def __init__(
        self,
        metric: Metric,
        prepare_function: Callable[[T], Any],
        prepare_does_unbatch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(metric, **kwargs)
        self.prepare_function = prepare_function
        self.prepare_does_unbatch = prepare_does_unbatch

    def forward(self, prediction: T, target: T) -> Dict[str, Any]:
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
            results = defaultdict(list)
            for prediction_str, target_str in zip(prediction_prepared, target_prepared):
                for k, v in super().forward(prediction_str, target_str).items():
                    results[k].append(v)
            mean_results = {k: sum(v) / len(v) for k, v in results.items()}
            return mean_results
        else:
            return super().forward(prediction_prepared, target_prepared)
