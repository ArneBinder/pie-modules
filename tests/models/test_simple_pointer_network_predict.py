import json
import logging
import os
from collections import defaultdict

import pytest
import torch
from pytorch_ie import AutoTaskModule, TaskModule
from pytorch_ie.annotations import BinaryRelation, LabeledSpan

from pie_modules.models import SimplePointerNetworkModel
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


@pytest.fixture(scope="module")
def trained_model() -> SimplePointerNetworkModel:
    model = SimplePointerNetworkModel.from_pretrained(MODEL_PATH)  # , generation_kwargs=None)
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
        data_serializable = {k: v.tolist() for k, v in data.items() if k != "CPM_tag"}
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


@pytest.fixture(scope="module")
def sciarg_batch_prediction():
    with open(SCIARG_BATCH_PREDICTION_PATH) as f:
        expected_prediction = json.load(f)
    return torch.tensor(expected_prediction)


@pytest.mark.slow
def test_sciarg_predict(
    trained_model, sciarg_batch_truncated, sciarg_batch_prediction, loaded_taskmodule
):
    torch.manual_seed(42)
    inputs, targets = sciarg_batch_truncated

    prediction = trained_model.predict(inputs)
    assert prediction is not None
    assert prediction.tolist() == sciarg_batch_prediction.tolist()


@pytest.mark.slow
def test_sciarg_predict_with_position_id_pattern(sciarg_batch_truncated, loaded_taskmodule):
    trained_model = SimplePointerNetworkModel.from_pretrained(MODEL_PATH_WITH_POSITION_ID_PATTERN)
    assert trained_model is not None

    torch.manual_seed(42)
    inputs, targets = sciarg_batch_truncated
    inputs_truncated = {k: v[:5] for k, v in inputs.items()}
    targets_truncated = {k: v[:5] for k, v in targets.items()}

    prediction = trained_model.predict(inputs_truncated, max_length=20)
    assert prediction is not None
    metric = loaded_taskmodule.configure_metric()
    metric.update(prediction, targets_truncated["tgt_tokens"])

    values = metric.compute()
    assert values == {
        "em": 0.4,
        "labeled_spans": {
            "background_claim": {"recall": 1.9608, "precision": 100.0, "f1": 3.8462},
            "data": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            "own_claim": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        },
        "labeled_spans/micro": {"recall": 1.0753, "precision": 50.0, "f1": 2.1053},
        "binary_relations": {
            "semantically_same": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            "supports": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            "contradicts": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            "parts_of_same": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        },
        "binary_relations/micro": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        "invalid": {"len": 0.2},
        "invalid/all": 0.2,
    }
