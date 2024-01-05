import math

import pytest
from pytorch_ie.annotations import LabeledSpan
from torchmetrics import Metric

from pie_modules.taskmodules.common import (
    PrecisionRecallAndF1ForLabeledAnnotations,
    WrappedMetricWithUnbatchFunction,
)


def test_precision_recall_and_f1_for_labeled_annotations():
    metric = PrecisionRecallAndF1ForLabeledAnnotations()
    assert metric.compute() == {"micro": {"f1": 0.0, "precision": 0.0, "recall": 0.0}}

    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="a")],
        predicted=[LabeledSpan(start=0, end=1, label="a")],
    )
    assert metric.compute() == {
        "a": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        "micro": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
    }

    metric.reset()
    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="a"), LabeledSpan(start=0, end=1, label="b")],
        predicted=[LabeledSpan(start=0, end=1, label="b"), LabeledSpan(start=0, end=1, label="c")],
    )
    assert metric.compute() == {
        "b": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        "a": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        "c": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        "micro": {"recall": 0.5, "precision": 0.5, "f1": 0.5},
    }

    # check deduplication in same update
    metric.reset()
    metric.update(
        gold=[
            LabeledSpan(start=0, end=1, label="a"),
            LabeledSpan(start=0, end=1, label="a"),
            LabeledSpan(start=0, end=1, label="b"),
        ],
        predicted=[
            LabeledSpan(start=0, end=1, label="b"),
            LabeledSpan(start=0, end=1, label="b"),
            LabeledSpan(start=0, end=1, label="c"),
        ],
    )
    assert metric.compute() == {
        "b": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        "a": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        "c": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        "micro": {"recall": 0.5, "precision": 0.5, "f1": 0.5},
    }

    # assert no deduplication over multiple updates
    metric.reset()
    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="a")],
        predicted=[LabeledSpan(start=0, end=1, label="b")],
    )
    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="b")],
        predicted=[LabeledSpan(start=0, end=1, label="a")],
    )
    assert metric.compute() == {
        "a": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
        "b": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
        "micro": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
    }


def test_precision_recall_and_f1_for_labeled_annotations_in_percent():
    metric = PrecisionRecallAndF1ForLabeledAnnotations(in_percent=True)

    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="a")],
        predicted=[LabeledSpan(start=0, end=1, label="a"), LabeledSpan(start=0, end=1, label="b")],
    )
    assert metric.compute() == {
        "b": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        "a": {"recall": 100.0, "precision": 100.0, "f1": 100.0},
        "micro": {"recall": 100.0, "precision": 50.0, "f1": 66.66666666666666},
    }


def test_precision_recall_and_f1_for_labeled_annotations_with_label_mapping():
    metric = PrecisionRecallAndF1ForLabeledAnnotations(
        label_mapping={"a": "label_a", "b": "label_b"}
    )

    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="a")],
        predicted=[LabeledSpan(start=0, end=1, label="a"), LabeledSpan(start=0, end=1, label="b")],
    )
    assert metric.compute() == {
        "label_a": {"f1": 1.0, "precision": 1.0, "recall": 1.0},
        "label_b": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
        "micro": {"f1": 0.6666666666666666, "precision": 0.5, "recall": 1.0},
    }


def test_precision_recall_and_f1_for_labeled_annotations_key_micro_error():
    metric = PrecisionRecallAndF1ForLabeledAnnotations()
    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="micro")],
        predicted=[],
    )
    with pytest.raises(ValueError) as excinfo:
        metric.compute()
    assert (
        str(excinfo.value)
        == "The key 'micro' was used as an annotation label, but it is reserved for the micro average. "
        "You can change which key is used for that with the 'key_micro' argument."
    )


@pytest.fixture(scope="module")
def wrapped_metric_with_unbatch_function():
    class ExactMatchMetric(Metric):
        """A simple metric that computes the exact match ratio between predictions and targets."""

        def __init__(self):
            super().__init__()
            self.add_state("matching", default=[])

        def update(self, prediction: str, target: str):
            self.matching.append(prediction == target)

        def compute(self):
            return sum(self.matching) / len(self.matching) if self.matching else float("nan")

    # just split the strings to unbatch the inputs
    return WrappedMetricWithUnbatchFunction(
        metric=ExactMatchMetric(), unbatch_function=lambda x: x.split()
    )


def test_wrapped_metric_with_unbatch_function(wrapped_metric_with_unbatch_function):
    metric = wrapped_metric_with_unbatch_function
    assert metric is not None
    assert metric.unbatch_function is not None
    assert metric.metric is not None

    metric_value = metric.compute()
    assert math.isnan(metric_value)

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
