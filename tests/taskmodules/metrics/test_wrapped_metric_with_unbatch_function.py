import pytest
from torchmetrics import Metric

from pie_modules.taskmodules.metrics import WrappedMetricWithUnbatchFunction


class TestMetric(Metric):
    """A simple metric that computes the exact match ratio between predictions and targets."""

    def __init__(self):
        super().__init__()
        self.add_state("matching", default=[])

    def update(self, prediction: str, target: str):
        self.matching.append(prediction == target)

    def compute(self):
        # Note: returning NaN in the case of an empty list would be more correct, but
        #   returning 0.0 is more convenient for testing.
        return sum(self.matching) / len(self.matching) if self.matching else 0.0


@pytest.fixture(scope="module")
def wrapped_metric_with_unbatch_function():
    # just split the strings to unbatch the inputs
    return WrappedMetricWithUnbatchFunction(
        metric=TestMetric(), unbatch_function=lambda x: x.split()
    )


def test_wrapped_metric_with_unbatch_function(wrapped_metric_with_unbatch_function):
    metric = wrapped_metric_with_unbatch_function
    assert metric is not None
    assert metric.unbatch_function is not None
    assert metric.metric is not None

    assert metric.compute() == 0.0

    metric.reset()
    metric.update(predictions="abc", targets="abc")
    assert metric.compute() == 1.0

    metric.reset()
    metric.update(predictions="abc", targets="def")
    assert metric.compute() == 0.0

    metric.reset()
    metric.update(predictions="abc def", targets="abc def")
    assert metric.compute() == 1.0

    metric.reset()
    metric.update(predictions="abc def", targets="def abc")
    assert metric.compute() == 0.0

    metric.reset()
    metric.update(predictions="abc xyz", targets="def xyz")
    assert metric.compute() == 0.5


def test_wrapped_metric_with_unbatch_function_size_mismatch(wrapped_metric_with_unbatch_function):
    with pytest.raises(ValueError) as excinfo:
        wrapped_metric_with_unbatch_function.update(predictions="abc", targets="abc def")
    assert str(excinfo.value) == "Number of predictions (1) and targets (2) do not match."
