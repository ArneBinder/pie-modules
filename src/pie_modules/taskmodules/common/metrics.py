from typing import Any, Callable, Generic, Iterable, TypeVar

from torchmetrics import Metric

T = TypeVar("T")


class WrappedMetricWithUnbatchFunction(Metric, Generic[T]):
    """A wrapper around a metric that can be used with a batched input.

    Args:
        unbatch_function: A function that takes a batched input and returns an iterable of
            individual inputs. This is used to unbatch the input before passing it to the wrapped
            metric.
        metric: The metric to wrap. It should be a subclass of torchmetrics.Metric.
    """

    def __init__(
        self, unbatch_function: Callable[[T], Iterable[Any]], metric: Metric, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.unbatch_function = unbatch_function
        self.metric = metric

    def update(self, predictions: T, targets: T) -> None:
        prediction_list = self.unbatch_function(predictions)
        target_list = self.unbatch_function(targets)
        for prediction_str, target_str in zip(prediction_list, target_list):
            self.metric(prediction_str, target_str)

    def compute(self) -> Any:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()
