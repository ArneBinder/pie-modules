import json
import logging
import os

import pytest
import torch
from pytorch_ie import AutoTaskModule, TaskModule

from pie_modules.models import SimpleGenerativeModel
from pie_modules.taskmodules import PointerNetworkTaskModuleForEnd2EndRE
from tests import DUMP_FIXTURE_DATA, FIXTURES_ROOT

logger = logging.getLogger(__name__)


# wandb run: https://wandb.ai/arne/dataset-sciarg-task-ner_re-training/runs/terkqzyn
MODEL_PATH = "/home/arbi01/projects/pie-document-level/models/dataset-sciarg/task-ner_re/2023-12-18_22-15-53"
MODEL_PATH_WITH_POSITION_ID_PATTERN = "/home/arbi01/projects/pie-document-level/models/dataset-sciarg/task-ner_re/2023-12-18_22-15-53_position_id_pattern"
TASKMODULE_PATH = (
    FIXTURES_ROOT
    / "taskmodules"
    / "pointer_network_taskmodule_for_end2end_re"
    / "sciarg_pretrained"
)
SCIARG_BATCH_PATH = (
    FIXTURES_ROOT
    / "taskmodules"
    / "pointer_network_taskmodule_for_end2end_re"
    / "sciarg_batch_encoding.{type}.json"
)

SCIARG_BATCH_PREDICTION_PATH = (
    FIXTURES_ROOT
    / "models"
    / "simple_pointer_network"
    / f"sciarg_batch_prediction_{os.path.split(MODEL_PATH)[-1]}.json"
)
SCIARG_BATCH_PREDICTION_WITH_POSITION_ID_PATTERN_PATH = (
    FIXTURES_ROOT
    / "models"
    / "simple_pointer_network"
    / f"sciarg_batch_prediction_{os.path.split(MODEL_PATH_WITH_POSITION_ID_PATTERN)[-1]}.json"
)


@pytest.fixture(scope="module")
def trained_model() -> SimpleGenerativeModel:
    model = SimpleGenerativeModel.from_pretrained(MODEL_PATH)  # , generation_kwargs=None)
    assert not model.training
    return model


@pytest.mark.slow
def test_trained_model(trained_model):
    assert trained_model is not None


@pytest.fixture(scope="module")
def loaded_taskmodule() -> TaskModule:
    taskmodule: TaskModule = AutoTaskModule.from_pretrained(str(TASKMODULE_PATH))
    assert taskmodule.is_prepared
    return taskmodule


@pytest.mark.slow
def test_loaded_taskmodule(loaded_taskmodule):
    assert loaded_taskmodule is not None


@pytest.mark.skipif(not DUMP_FIXTURE_DATA, reason="don't dump fixture data")
def test_dump_sciarg_batch(loaded_taskmodule):
    from pie_datasets import DatasetDict

    dataset = DatasetDict.load_dataset("pie/sciarg")
    dataset_converted = dataset.to_document_type(loaded_taskmodule.document_type)
    sciarg_document = dataset_converted["train"][0]

    loaded_taskmodule.tokenizer_kwargs["strict_span_conversion"] = False
    task_encodings = loaded_taskmodule.encode([sciarg_document], encode_target=True)
    batch = loaded_taskmodule.collate(task_encodings)
    for data_type, data in zip(["inputs", "targets"], batch):
        # Note: we don't dump the CPM_tag, because it's too large
        data_serializable = {k: v.tolist() for k, v in data.items() if k != "constraints"}
        with open(str(SCIARG_BATCH_PATH).format(type=data_type), "w") as f:
            json.dump(data_serializable, f, sort_keys=True)


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


def load_prediction(path):
    with open(path) as f:
        expected_prediction = json.load(f)
    return torch.tensor(expected_prediction)


@pytest.mark.slow
def test_sciarg_predict(trained_model, sciarg_batch_truncated, loaded_taskmodule):
    expected_prediction = load_prediction(SCIARG_BATCH_PREDICTION_PATH)
    torch.manual_seed(42)
    inputs, targets = sciarg_batch_truncated

    prediction = trained_model.predict(inputs)
    assert prediction is not None
    assert prediction.tolist() == expected_prediction.tolist()

    # calculate metrics just to check the scores
    metric = loaded_taskmodule.configure_model_metric(stage="test")
    metric.update(prediction, targets["labels"])
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


@pytest.mark.slow
def test_sciarg_predict_with_position_id_pattern(sciarg_batch_truncated, loaded_taskmodule):
    trained_model = SimpleGenerativeModel.from_pretrained(MODEL_PATH_WITH_POSITION_ID_PATTERN)
    assert trained_model is not None

    expected_prediction = load_prediction(SCIARG_BATCH_PREDICTION_WITH_POSITION_ID_PATTERN_PATH)
    torch.manual_seed(42)
    inputs, targets = sciarg_batch_truncated

    prediction = trained_model.predict(inputs, max_length=20)
    assert prediction is not None
    assert prediction.tolist() == expected_prediction.tolist()

    # calculate metrics just to check the scores
    metric = loaded_taskmodule.configure_model_metric(stage="test")
    metric.update(prediction, targets["labels"])

    values = metric.compute()
    assert values == {
        "binary_relations": {
            "contradicts": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
            "parts_of_same": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
            "semantically_same": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
            "supports": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
        },
        "binary_relations/micro": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
        "encoding_match": 0.4,
        "errors": {"correct": 0.1667, "len": 0.8333},
        "errors/all": 0.8333,
        "labeled_spans": {
            "background_claim": {"f1": 3.8462, "precision": 100.0, "recall": 1.9608},
            "data": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
            "own_claim": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
        },
        "labeled_spans/micro": {"f1": 2.1277, "precision": 100.0, "recall": 1.0753},
    }
