import pytest
import torch
from pytorch_ie.annotations import LabeledSpan

from pie_modules.taskmodules.metrics import PrecisionRecallAndF1ForLabeledAnnotations


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
