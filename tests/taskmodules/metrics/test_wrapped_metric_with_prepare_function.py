from torchmetrics import Metric

from pie_modules.taskmodules.metrics import WrappedMetricWithPrepareFunction


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


def test_metric():
    metric = WrappedMetricWithPrepareFunction(
        metric=TestMetric(),
        # just take the first "word" of each input
        prepare_function=lambda x: x.split()[0],
    )
    assert metric is not None
    assert metric.prepare_function is not None
    assert metric.metric is not None

    assert metric.compute() == 0.0

    metric.reset()
    metric(predictions="abc", targets="abc")
    assert metric.compute() == 1.0

    metric.reset()
    metric(predictions="abc", targets="def")
    assert metric.compute() == 0.0

    metric.reset()
    metric(predictions="abc def", targets="abc xyz")
    # we consider just the first word, so this is still 1.0
    assert metric.compute() == 1.0

    metric.reset()
    metric(predictions="abc def", targets="xyz def")
    assert metric.compute() == 0.0
