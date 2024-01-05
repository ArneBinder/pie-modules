import json
from typing import Any, Dict, Tuple

import pytest
import torch
from pytorch_ie.annotations import LabeledSpan
from torchmetrics import Metric

from pie_modules.taskmodules.common import (
    PrecisionRecallAndF1ForLabeledAnnotations,
    WrappedLayerMetricsWithUnbatchAndDecodeWithErrorsFunction,
    WrappedMetricWithUnbatchFunction,
)


def test_precision_recall_and_f1_for_labeled_annotations():
    metric = PrecisionRecallAndF1ForLabeledAnnotations()
    assert metric.compute() == {"micro": {"f1": 0.0, "precision": 0.0, "recall": 0.0}}
    assert metric.metric_state == {
        "gold": [],
        "predicted": [],
        "correct": [],
        "idx": torch.tensor(0),
    }

    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="a")],
        predicted=[LabeledSpan(start=0, end=1, label="a")],
    )
    assert metric.metric_state == {
        "gold": [(0, LabeledSpan(start=0, end=1, label="a", score=1.0))],
        "predicted": [(0, LabeledSpan(start=0, end=1, label="a", score=1.0))],
        "correct": [(0, LabeledSpan(start=0, end=1, label="a", score=1.0))],
        "idx": torch.tensor(1),
    }
    assert metric.compute() == {
        "a": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
        "micro": {"recall": 1.0, "precision": 1.0, "f1": 1.0},
    }

    metric.reset()
    metric.update(
        gold=[LabeledSpan(start=0, end=1, label="a"), LabeledSpan(start=0, end=1, label="b")],
        predicted=[LabeledSpan(start=0, end=1, label="b"), LabeledSpan(start=0, end=1, label="c")],
    )
    assert set(metric.metric_state) == {"gold", "predicted", "correct", "idx"}
    assert metric.metric_state["idx"] == torch.tensor(1)
    assert set(metric.metric_state["gold"]) == {
        (0, LabeledSpan(start=0, end=1, label="a", score=1.0)),
        (0, LabeledSpan(start=0, end=1, label="b", score=1.0)),
    }
    assert set(metric.metric_state["predicted"]) == {
        (0, LabeledSpan(start=0, end=1, label="b", score=1.0)),
        (0, LabeledSpan(start=0, end=1, label="c", score=1.0)),
    }
    assert set(metric.metric_state["correct"]) == {
        (0, LabeledSpan(start=0, end=1, label="b", score=1.0)),
    }
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
    assert set(metric.metric_state) == {"gold", "predicted", "correct", "idx"}
    assert metric.metric_state["idx"] == torch.tensor(1)
    assert set(metric.metric_state["gold"]) == {
        (0, LabeledSpan(start=0, end=1, label="a", score=1.0)),
        (0, LabeledSpan(start=0, end=1, label="b", score=1.0)),
    }
    assert set(metric.metric_state["predicted"]) == {
        (0, LabeledSpan(start=0, end=1, label="b", score=1.0)),
        (0, LabeledSpan(start=0, end=1, label="c", score=1.0)),
    }
    assert set(metric.metric_state["correct"]) == {
        (0, LabeledSpan(start=0, end=1, label="b", score=1.0)),
    }
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
    assert set(metric.metric_state) == {"gold", "predicted", "correct", "idx"}
    assert metric.metric_state["idx"] == torch.tensor(2)
    assert set(metric.metric_state["gold"]) == {
        (0, LabeledSpan(start=0, end=1, label="a", score=1.0)),
        (1, LabeledSpan(start=0, end=1, label="b", score=1.0)),
    }
    assert set(metric.metric_state["predicted"]) == {
        (0, LabeledSpan(start=0, end=1, label="b", score=1.0)),
        (1, LabeledSpan(start=0, end=1, label="a", score=1.0)),
    }
    assert set(metric.metric_state["correct"]) == set()
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


@pytest.fixture(scope="module")
def wrapped_layer_metrics_with_unbatch_and_decode_with_errors_function():
    def decode_with_errors_function(x: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
        if x == "error":
            return {"entities": [], "relations": []}, {"dummy": 1}
        else:
            return json.loads(x), {"dummy": 0}

    layer_metrics = {
        "entities": TestMetric(),
        "relations": TestMetric(),
    }
    metric = WrappedLayerMetricsWithUnbatchAndDecodeWithErrorsFunction(
        layer_metrics=layer_metrics,
        unbatch_function=lambda x: x.split("\n"),
        decode_layers_with_errors_function=decode_with_errors_function,
    )
    return metric


def test_wrapped_layer_metrics_with_unbatch_and_decode_with_errors_function(
    wrapped_layer_metrics_with_unbatch_and_decode_with_errors_function,
):
    metric = wrapped_layer_metrics_with_unbatch_and_decode_with_errors_function
    assert metric is not None
    assert metric.unbatch_function is not None
    assert metric.decode_layers_with_errors_function is not None
    assert metric.layer_metrics is not None
    assert metric.metric_state == {
        "total": torch.tensor(0),
        "exact_encoding_matches": torch.tensor(0),
        "errors": [],
    }

    values = metric.compute()
    assert metric.metric_state
    assert values == {
        "decoding_errors": {"all": 0.0},
        "entities": 0.0,
        "exact_encoding_matches": 0.0,
        "relations": 0.0,
    }

    metric.reset()
    # Prediction and expected are the same.
    metric.update(
        prediction=json.dumps({"entities": ["E1"], "relations": ["R1"]}),
        expected=json.dumps({"entities": ["E1"], "relations": ["R1"]}),
    )
    assert metric.metric_state == {
        "total": torch.tensor(1),
        "exact_encoding_matches": torch.tensor(1),
        "errors": [("dummy", 0)],
    }
    values = metric.compute()
    assert values == {
        "decoding_errors": {"all": 0.0, "dummy": 0.0},
        "entities": 1.0,
        "exact_encoding_matches": 1.0,
        "relations": 1.0,
    }

    metric.reset()
    # Prediction and expected are different and there are multiple entries.
    # The first entry is an exact match, the second entry is not.
    metric.update(
        prediction=json.dumps({"entities": ["E1"], "relations": ["R1"]})
        + "\n"
        + json.dumps({"entities": ["E1"], "relations": ["R1"]}),
        expected=json.dumps({"entities": ["E1"], "relations": ["R1"]})
        + "\n"
        + json.dumps({"entities": ["E1"], "relations": ["R2"]}),
    )
    assert metric.metric_state == {
        "total": torch.tensor(2),
        "exact_encoding_matches": torch.tensor(1),
        "errors": [("dummy", 0), ("dummy", 0)],
    }
    values = metric.compute()
    assert values == {
        "decoding_errors": {"all": 0.0, "dummy": 0.0},
        "entities": 1.0,
        "exact_encoding_matches": 0.5,
        "relations": 0.5,
    }

    metric.reset()
    # Encoding error
    metric.update(
        prediction="error",
        expected=json.dumps({"entities": ["E1"], "relations": []}),
    )
    assert metric.metric_state == {
        "total": torch.tensor(1),
        "exact_encoding_matches": torch.tensor(0),
        "errors": [("dummy", 1)],
    }
    values = metric.compute()
    # In the case on an error, the decoding function returns adict with empty lists for entities and relations.
    # Thus, we get a perfect match for entities and a 0.0 match for relations.
    assert values == {
        "decoding_errors": {"all": 1.0, "dummy": 1.0},
        "entities": 0.0,
        "exact_encoding_matches": 0.0,
        "relations": 1.0,
    }

    # test mismatched number of predictions and targets
    metric.reset()
    with pytest.raises(ValueError) as excinfo:
        metric.update(
            prediction=json.dumps({"entities": ["E1"], "relations": ["R1"]}),
            expected=json.dumps({"entities": ["E1"], "relations": ["R1"]})
            + "\n"
            + json.dumps({"entities": ["E1"], "relations": ["R1"]}),
        )
    assert str(excinfo.value) == "Number of predictions (1) and targets (2) do not match."
