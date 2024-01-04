import json

import pytest
import torch
from pytorch_ie import AutoTaskModule, TaskModule
from torch import isnan, tensor
from torchmetrics.text import ROUGEScore

from pie_modules.taskmodules.common import WrappedMetricWithUnbatchFunction
from tests.models.test_simple_generative_pointer_predict import (
    SCIARG_BATCH_PATH,
    SCIARG_BATCH_PREDICTION_PATH,
    TASKMODULE_PATH,
)


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


@pytest.fixture(scope="module")
def taskmodule() -> TaskModule:
    taskmodule: TaskModule = AutoTaskModule.from_pretrained(str(TASKMODULE_PATH))
    assert taskmodule.is_prepared
    return taskmodule


@pytest.fixture(scope="module")
def sciarg_batch_truncated():
    batch = ()
    for data_type in ["inputs", "targets"]:
        path = str(SCIARG_BATCH_PATH).format(type=data_type)
        with open(path) as f:
            data_json = json.load(f)
        data_truncated = {k: torch.tensor(v[:5]) for k, v in data_json.items()}
        batch += (data_truncated,)
    return batch


@pytest.fixture(scope="module")
def sciarg_batch_prediction():
    with open(SCIARG_BATCH_PREDICTION_PATH) as f:
        expected_prediction = json.load(f)
    return torch.tensor(expected_prediction)


def test_taskmodule_metric(taskmodule, sciarg_batch_truncated, sciarg_batch_prediction):
    metric = taskmodule.configure_model_metric("test")
    inputs, targets = sciarg_batch_truncated
    metric.update(sciarg_batch_prediction, targets["labels"])

    values = metric.compute()
    assert values == {
        "encoding_match": 0.4,
        "labeled_spans": {
            "own_claim": {"recall": 25.0, "precision": 6.6667, "f1": 10.5263},
            "background_claim": {"recall": 50.9804, "precision": 47.2727, "f1": 49.0566},
            "data": {"recall": 20.5882, "precision": 20.5882, "f1": 20.5882},
        },
        "labeled_spans/micro": {"recall": 37.6344, "precision": 29.4118, "f1": 33.0189},
        "binary_relations": {
            "contradicts": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            "parts_of_same": {"recall": 16.6667, "precision": 8.3333, "f1": 11.1111},
            "supports": {"recall": 8.5106, "precision": 8.6957, "f1": 8.6022},
            "semantically_same": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        },
        "binary_relations/micro": {"recall": 8.0645, "precision": 8.4746, "f1": 8.2645},
        "errors": {"correct": 0.977, "order": 0.023},
        "errors/all": 0.023,
    }

    assert set(metric.state) == {"layer_metrics", "total", "encoding_match", "errors"}
