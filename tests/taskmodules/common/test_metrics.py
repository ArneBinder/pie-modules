import pytest
from torch import isnan, tensor
from torchmetrics.text import ROUGEScore

from pie_modules.taskmodules.common import WrappedMetricWithUnbatchFunction


@pytest.fixture(scope="module")
def metric():
    return WrappedMetricWithUnbatchFunction(metric=ROUGEScore(), unbatch_function=lambda x: x)


def test_metric(metric):
    assert metric is not None
    assert metric.unbatch_function is not None
    assert metric.metric is not None

    metric_values = metric.compute()
    assert len(metric_values) > 0
    assert all(isnan(value) for value in metric_values.values())


def test_metric_equal_predictions_and_targets(metric):
    metric.reset()
    metric.update(predictions=["test"], targets=["test"])
    assert metric.compute() == {
        "rouge1_fmeasure": tensor(1.0),
        "rouge1_precision": tensor(1.0),
        "rouge1_recall": tensor(1.0),
        "rouge2_fmeasure": tensor(0.0),
        "rouge2_precision": tensor(0.0),
        "rouge2_recall": tensor(0.0),
        "rougeL_fmeasure": tensor(1.0),
        "rougeL_precision": tensor(1.0),
        "rougeL_recall": tensor(1.0),
        "rougeLsum_fmeasure": tensor(1.0),
        "rougeLsum_precision": tensor(1.0),
        "rougeLsum_recall": tensor(1.0),
    }


def test_metric_different_predictions_and_targets(metric):
    metric.reset()
    metric.update(predictions=["test"], targets=["different"])
    assert metric.compute() == {
        "rouge1_fmeasure": tensor(0.0),
        "rouge1_precision": tensor(0.0),
        "rouge1_recall": tensor(0.0),
        "rouge2_fmeasure": tensor(0.0),
        "rouge2_precision": tensor(0.0),
        "rouge2_recall": tensor(0.0),
        "rougeL_fmeasure": tensor(0.0),
        "rougeL_precision": tensor(0.0),
        "rougeL_recall": tensor(0.0),
        "rougeLsum_fmeasure": tensor(0.0),
        "rougeLsum_precision": tensor(0.0),
        "rougeLsum_recall": tensor(0.0),
    }


def test_metric_partially_different_predictions_and_targets(metric):
    metric.reset()
    metric.update(predictions=["test"], targets=["test different"])
    metric_values = metric.compute()
    assert metric_values == {
        "rouge1_fmeasure": tensor(0.6666666865348816),
        "rouge1_precision": tensor(1.0),
        "rouge1_recall": tensor(0.5000),
        "rouge2_fmeasure": tensor(0.0),
        "rouge2_precision": tensor(0.0),
        "rouge2_recall": tensor(0.0),
        "rougeL_fmeasure": tensor(0.6666666865348816),
        "rougeL_precision": tensor(1.0),
        "rougeL_recall": tensor(0.5000),
        "rougeLsum_fmeasure": tensor(0.6666666865348816),
        "rougeLsum_precision": tensor(1.0),
        "rougeLsum_recall": tensor(0.5000),
    }
