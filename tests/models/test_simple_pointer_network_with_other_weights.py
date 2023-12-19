import json
import logging
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass

import pytest
import torch
from pytorch_ie import AnnotationList, Document, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextBasedDocument
from pytorch_lightning import Trainer
from torch.optim import AdamW

from pie_modules.models import PointerNetworkModel, SimplePointerNetworkModel
from pie_modules.taskmodules import PointerNetworkTaskModule
from tests import DUMP_FIXTURE_DATA, FIXTURES_ROOT, _config_to_str

logger = logging.getLogger(__name__)


# wandb run: https://wandb.ai/arne/dataset-sciarg-task-ner_re-training/runs/2xhakq93
MODEL_PATH = "/home/arbi01/projects/pie-document-level/models/dataset-sciarg/task-ner_re/2023-12-14_00-25-37"
# wandb run: https://wandb.ai/arne/dataset-sciarg-task-ner_re-training/runs/y00unkeq
OTHER_MODEL_PATH = "/home/arbi01/projects/pie-document-level/models/dataset-sciarg/task-ner_re/2023-12-15_03-04-43"


@pytest.fixture(scope="module")
def trained_model() -> SimplePointerNetworkModel:
    model = SimplePointerNetworkModel.from_pretrained(MODEL_PATH)
    assert not model.training
    return model


@pytest.mark.slow
def test_trained_model(trained_model):
    assert trained_model is not None


@pytest.fixture(scope="module")
def loaded_taskmodule():
    taskmodule = PointerNetworkTaskModule.from_pretrained(MODEL_PATH)
    assert taskmodule.is_prepared
    return taskmodule


@pytest.mark.slow
def test_loaded_taskmodule(loaded_taskmodule):
    assert loaded_taskmodule is not None


@pytest.mark.skipif(not DUMP_FIXTURE_DATA, reason="don't dump fixture data")
def test_dump_sciarg_batch(loaded_taskmodule):
    from pie_datasets import DatasetDict

    dataset = DatasetDict.load_dataset("pie/sciarg", name="merge_fragmented_spans")
    dataset_converted = dataset.to_document_type(loaded_taskmodule.document_type)
    sciarg_document = dataset_converted["train"][0]

    task_encodings = loaded_taskmodule.encode([sciarg_document], encode_target=True)
    batch = loaded_taskmodule.collate(task_encodings)
    path = FIXTURES_ROOT / "models" / "pointer_network_model" / "sciarg_batch_with_targets.pkl"
    with open(path, "wb") as f:
        pickle.dump(batch, f)


@pytest.fixture(scope="module")
def sciarg_batch():
    with open(
        FIXTURES_ROOT / "models" / "pointer_network" / "sciarg_batch_with_targets.pkl",
        "rb",
    ) as f:
        batch = pickle.load(f)
    return batch


@pytest.fixture(scope="module")
def sciarg_batch_predictions():
    with open(
        FIXTURES_ROOT / "models" / "simple_pointer_network" / "sciarg_batch_predictions.json"
    ) as f:
        data = json.load(f)
    return data


@pytest.mark.slow
def test_sciarg_batch_predictions(sciarg_batch_predictions, loaded_taskmodule):
    annotations, errors = loaded_taskmodule.annotation_encoder_decoder.decode(
        sciarg_batch_predictions[2]
    )
    assert set(annotations) == {"labeled_spans", "binary_relations"}
    assert len(annotations["labeled_spans"]) == 2
    assert len(annotations["binary_relations"]) == 1


@pytest.mark.slow
def test_sciarg_predict_step(trained_model, sciarg_batch, sciarg_batch_predictions):
    torch.manual_seed(42)
    prediction = trained_model.predict_step(sciarg_batch, 0)
    assert prediction is not None
    assert prediction["pred"].tolist() == sciarg_batch_predictions


def convert_original_param_name(name: str) -> str:
    new_name = name.replace("encoder.bart_", "")
    new_name = new_name.replace("encoder.embed_tokens.", "shared.")
    new_name = re.sub(r"^decoder\.", "", new_name)
    if new_name.startswith("encoder_mlp."):
        return f"model.pointer_head.{new_name}"
    else:
        return f"model.model.{new_name}"


def replace_weights_with_other_model(model, other_model):
    if isinstance(other_model, PointerNetworkModel):
        other_params_name_mapped = {
            convert_original_param_name(name): param
            for name, param in other_model.named_parameters()
        }
    else:
        raise ValueError(f"Unknown model type: {type(other_model)}")
    param: torch.nn.Parameter
    for name, param in model.named_parameters():
        if name in other_params_name_mapped:
            other_param = other_params_name_mapped[name]
            assert param.shape == other_param.shape
            param.data = other_param.data
        else:
            logger.warning(f"Parameter {name} not found in other model!")


@pytest.mark.slow
def test_sciarg_predict_with_weights_from_other_model(
    trained_model, sciarg_batch, sciarg_batch_predictions, loaded_taskmodule
):
    replace_weights_with_other_model(
        trained_model, PointerNetworkModel.from_pretrained(OTHER_MODEL_PATH)
    )

    torch.manual_seed(42)
    inputs, targets = sciarg_batch
    generation_kwargs = {"num_beams": 5, "max_length": 512}
    prediction = trained_model.predict(inputs, **generation_kwargs)
    assert prediction is not None
    targets_list = targets["tgt_tokens"].tolist()
    prediction_list = prediction["pred"].tolist()

    tp = defaultdict(set)
    fp = defaultdict(set)
    fn = defaultdict(set)
    for i in range(len(targets_list)):
        current_targets = targets_list[i]
        current_predictions = prediction_list[i]
        annotations, errors = loaded_taskmodule.annotation_encoder_decoder.decode(
            current_predictions
        )
        (
            expected_annotations,
            expected_errors,
        ) = loaded_taskmodule.annotation_encoder_decoder.decode(current_targets)
        for layer_name in expected_annotations:
            tp[layer_name] |= set(annotations[layer_name]) & set(expected_annotations[layer_name])
            fp[layer_name] |= set(annotations[layer_name]) - set(expected_annotations[layer_name])
            fn[layer_name] |= set(expected_annotations[layer_name]) - set(annotations[layer_name])

    # check the numbers
    assert {layer_name: len(anns) for layer_name, anns in fp.items()} == {
        "labeled_spans": 120,
        "binary_relations": 71,
    }

    assert {layer_name: len(anns) for layer_name, anns in fn.items()} == {
        "labeled_spans": 111,
        "binary_relations": 80,
    }

    assert {layer_name: len(anns) for layer_name, anns in tp.items()} == {
        "labeled_spans": 24,
        "binary_relations": 2,
    }

    # check the actual annotations
