import json
import logging
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass

import pytest
import torch
from pytorch_ie import AnnotationList, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextBasedDocument
from torch.optim import AdamW

from pie_modules.models import PointerNetworkModel, SimplePointerNetworkModel
from pie_modules.taskmodules import PointerNetworkTaskModule

# from src.models.components.gmam.metrics import LabeledAnnotationScore, AnnotationLayerMetric
from tests import FIXTURES_ROOT, _config_to_str

logger = logging.getLogger(__name__)


# wandb run: https://wandb.ai/arne/dataset-sciarg-task-ner_re-training/runs/y00unkeq
MODEL_PATH = "/home/arbi01/projects/pie-document-level/models/dataset-sciarg/task-ner_re/2023-12-15_03-04-43"
# wandb run: https://wandb.ai/arne/dataset-sciarg-task-ner_re-training/runs/2xhakq93
OTHER_MODEL_PATH = "/home/arbi01/projects/pie-document-level/models/dataset-sciarg/task-ner_re/2023-12-14_00-25-37"


@pytest.fixture(scope="module")
def trained_model() -> PointerNetworkModel:
    model = PointerNetworkModel.from_pretrained(MODEL_PATH)
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
        FIXTURES_ROOT / "models" / "pointer_network_model" / "sciarg_batch_predictions.json"
    ) as f:
        data = json.load(f)
    return data


@pytest.mark.slow
def test_sciarg_batch_predictions(sciarg_batch_predictions, loaded_taskmodule):
    annotations, errors = loaded_taskmodule.annotation_encoder_decoder.decode(
        sciarg_batch_predictions[2]
    )
    assert set(annotations) == {"labeled_spans", "binary_relations"}
    assert len(annotations["labeled_spans"]) == 44
    assert len(annotations["binary_relations"]) == 21


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
    if isinstance(other_model, SimplePointerNetworkModel):
        params_name_mapping = {
            convert_original_param_name(name): name for name, param in model.named_parameters()
        }
        other_params_name_mapped = {
            params_name_mapping[name]: param for name, param in other_model.named_parameters()
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
        trained_model, SimplePointerNetworkModel.from_pretrained(OTHER_MODEL_PATH)
    )

    torch.manual_seed(42)
    inputs, targets = sciarg_batch
    # generation_kwargs = {"num_beams": 5, "max_length": 512}
    prediction = trained_model.predict(inputs)
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
        "labeled_spans": 101,
        "binary_relations": 63,
    }
    assert {layer_name: len(anns) for layer_name, anns in fn.items()} == {
        "labeled_spans": 93,
        "binary_relations": 73,
    }
    assert {layer_name: len(anns) for layer_name, anns in tp.items()} == {
        "labeled_spans": 41,
        "binary_relations": 9,
    }

    # check the actual annotations
    assert dict(fp) == {
        "labeled_spans": {
            LabeledSpan(start=776, end=787, label="background_claim", score=1.0),
            LabeledSpan(start=696, end=720, label="own_claim", score=1.0),
            LabeledSpan(start=23, end=39, label="own_claim", score=1.0),
            LabeledSpan(start=797, end=798, label="data", score=1.0),
            LabeledSpan(start=240, end=250, label="background_claim", score=1.0),
            LabeledSpan(start=236, end=538, label="own_claim", score=1.0),
            LabeledSpan(start=70, end=78, label="data", score=1.0),
            LabeledSpan(start=169, end=170, label="data", score=1.0),
            LabeledSpan(start=45, end=52, label="own_claim", score=1.0),
            LabeledSpan(start=217, end=228, label="data", score=1.0),
            LabeledSpan(start=42, end=45, label="data", score=1.0),
            LabeledSpan(start=719, end=734, label="own_claim", score=1.0),
            LabeledSpan(start=783, end=814, label="data", score=1.0),
            LabeledSpan(start=510, end=511, label="data", score=1.0),
            LabeledSpan(start=502, end=503, label="data", score=1.0),
            LabeledSpan(start=536, end=565, label="own_claim", score=1.0),
            LabeledSpan(start=149, end=167, label="background_claim", score=1.0),
            LabeledSpan(start=324, end=325, label="data", score=1.0),
            LabeledSpan(start=512, end=516, label="background_claim", score=1.0),
            LabeledSpan(start=636, end=664, label="own_claim", score=1.0),
            LabeledSpan(start=41, end=117, label="data", score=1.0),
            LabeledSpan(start=627, end=634, label="background_claim", score=1.0),
            LabeledSpan(start=443, end=453, label="background_claim", score=1.0),
            LabeledSpan(start=331, end=337, label="data", score=1.0),
            LabeledSpan(start=251, end=260, label="background_claim", score=1.0),
            LabeledSpan(start=78, end=261, label="own_claim", score=1.0),
            LabeledSpan(start=307, end=308, label="data", score=1.0),
            LabeledSpan(start=298, end=317, label="own_claim", score=1.0),
            LabeledSpan(start=171, end=173, label="data", score=1.0),
            LabeledSpan(start=408, end=433, label="own_claim", score=1.0),
            LabeledSpan(start=469, end=475, label="background_claim", score=1.0),
            LabeledSpan(start=463, end=464, label="data", score=1.0),
            LabeledSpan(start=883, end=965, label="own_claim", score=1.0),
            LabeledSpan(start=969, end=991, label="own_claim", score=1.0),
            LabeledSpan(start=164, end=166, label="data", score=1.0),
            LabeledSpan(start=258, end=270, label="background_claim", score=1.0),
            LabeledSpan(start=262, end=276, label="data", score=1.0),
            LabeledSpan(start=589, end=599, label="own_claim", score=1.0),
            LabeledSpan(start=72, end=86, label="own_claim", score=1.0),
            LabeledSpan(start=192, end=202, label="background_claim", score=1.0),
            LabeledSpan(start=50, end=51, label="data", score=1.0),
            LabeledSpan(start=533, end=556, label="own_claim", score=1.0),
            LabeledSpan(start=95, end=96, label="data", score=1.0),
            LabeledSpan(start=118, end=133, label="own_claim", score=1.0),
            LabeledSpan(start=499, end=500, label="data", score=1.0),
            LabeledSpan(start=799, end=814, label="background_claim", score=1.0),
            LabeledSpan(start=127, end=148, label="background_claim", score=1.0),
            LabeledSpan(start=388, end=389, label="data", score=1.0),
            LabeledSpan(start=348, end=371, label="own_claim", score=1.0),
            LabeledSpan(start=619, end=626, label="data", score=1.0),
            LabeledSpan(start=230, end=239, label="background_claim", score=1.0),
            LabeledSpan(start=659, end=665, label="own_claim", score=1.0),
            LabeledSpan(start=742, end=747, label="own_claim", score=1.0),
            LabeledSpan(start=174, end=176, label="data", score=1.0),
            LabeledSpan(start=290, end=308, label="own_claim", score=1.0),
            LabeledSpan(start=87, end=97, label="data", score=1.0),
            LabeledSpan(start=625, end=636, label="own_claim", score=1.0),
            LabeledSpan(start=161, end=188, label="own_claim", score=1.0),
            LabeledSpan(start=377, end=407, label="own_claim", score=1.0),
            LabeledSpan(start=142, end=158, label="own_claim", score=1.0),
            LabeledSpan(start=79, end=87, label="background_claim", score=1.0),
            LabeledSpan(start=158, end=179, label="background_claim", score=1.0),
            LabeledSpan(start=367, end=368, label="data", score=1.0),
            LabeledSpan(start=429, end=441, label="background_claim", score=1.0),
            LabeledSpan(start=34, end=46, label="background_claim", score=1.0),
            LabeledSpan(start=454, end=461, label="background_claim", score=1.0),
            LabeledSpan(start=378, end=398, label="own_claim", score=1.0),
            LabeledSpan(start=182, end=188, label="background_claim", score=1.0),
            LabeledSpan(start=201, end=217, label="background_claim", score=1.0),
            LabeledSpan(start=512, end=527, label="background_claim", score=1.0),
            LabeledSpan(start=24, end=33, label="background_claim", score=1.0),
            LabeledSpan(start=168, end=173, label="data", score=1.0),
            LabeledSpan(start=384, end=387, label="background_claim", score=1.0),
            LabeledSpan(start=47, end=49, label="data", score=1.0),
            LabeledSpan(start=737, end=738, label="data", score=1.0),
            LabeledSpan(start=543, end=554, label="data", score=1.0),
            LabeledSpan(start=152, end=156, label="own_claim", score=1.0),
            LabeledSpan(start=233, end=253, label="background_claim", score=1.0),
            LabeledSpan(start=96, end=132, label="own_claim", score=1.0),
            LabeledSpan(start=177, end=180, label="data", score=1.0),
            LabeledSpan(start=58, end=59, label="data", score=1.0),
            LabeledSpan(start=106, end=120, label="background_claim", score=1.0),
            LabeledSpan(start=618, end=691, label="own_claim", score=1.0),
            LabeledSpan(start=723, end=724, label="data", score=1.0),
            LabeledSpan(start=667, end=678, label="own_claim", score=1.0),
            LabeledSpan(start=726, end=741, label="data", score=1.0),
            LabeledSpan(start=927, end=913, label="own_claim", score=1.0),
            LabeledSpan(start=474, end=521, label="own_claim", score=1.0),
            LabeledSpan(start=180, end=200, label="background_claim", score=1.0),
            LabeledSpan(start=66, end=76, label="own_claim", score=1.0),
            LabeledSpan(start=486, end=487, label="data", score=1.0),
            LabeledSpan(start=157, end=160, label="data", score=1.0),
            LabeledSpan(start=507, end=510, label="own_claim", score=1.0),
            LabeledSpan(start=355, end=376, label="own_claim", score=1.0),
            LabeledSpan(start=85, end=99, label="own_claim", score=1.0),
            LabeledSpan(start=620, end=624, label="data", score=1.0),
            LabeledSpan(start=350, end=364, label="own_claim", score=1.0),
            LabeledSpan(start=136, end=148, label="background_claim", score=1.0),
            LabeledSpan(start=607, end=617, label="background_claim", score=1.0),
            LabeledSpan(start=528, end=535, label="own_claim", score=1.0),
            LabeledSpan(start=159, end=161, label="data", score=1.0),
        },
        "binary_relations": {
            BinaryRelation(
                head=LabeledSpan(start=307, end=308, label="data", score=1.0),
                tail=LabeledSpan(start=233, end=253, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=230, end=239, label="background_claim", score=1.0),
                tail=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=518, end=521, label="data", score=1.0),
                tail=LabeledSpan(start=512, end=516, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=726, end=741, label="data", score=1.0),
                tail=LabeledSpan(start=742, end=747, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=149, end=167, label="background_claim", score=1.0),
                tail=LabeledSpan(start=136, end=148, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=388, end=389, label="data", score=1.0),
                tail=LabeledSpan(start=384, end=387, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=367, end=368, label="data", score=1.0),
                tail=LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=169, end=170, label="data", score=1.0),
                tail=LabeledSpan(start=149, end=167, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=324, end=325, label="data", score=1.0),
                tail=LabeledSpan(start=258, end=270, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=512, end=516, label="background_claim", score=1.0),
                tail=LabeledSpan(start=528, end=532, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=177, end=180, label="data", score=1.0),
                tail=LabeledSpan(start=149, end=167, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=522, end=526, label="data", score=1.0),
                tail=LabeledSpan(start=512, end=516, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=158, end=179, label="background_claim", score=1.0),
                tail=LabeledSpan(start=127, end=148, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=164, end=166, label="data", score=1.0),
                tail=LabeledSpan(start=142, end=158, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=171, end=173, label="data", score=1.0),
                tail=LabeledSpan(start=149, end=167, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=174, end=176, label="data", score=1.0),
                tail=LabeledSpan(start=512, end=516, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=87, end=97, label="data", score=1.0),
                tail=LabeledSpan(start=72, end=86, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=24, end=33, label="background_claim", score=1.0),
                tail=LabeledSpan(start=34, end=46, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=307, end=308, label="data", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=47, end=49, label="data", score=1.0),
                tail=LabeledSpan(start=34, end=46, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=96, end=105, label="data", score=1.0),
                tail=LabeledSpan(start=136, end=148, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=723, end=724, label="data", score=1.0),
                tail=LabeledSpan(start=696, end=720, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=159, end=161, label="data", score=1.0),
                tail=LabeledSpan(start=142, end=158, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=343, end=347, label="data", score=1.0),
                tail=LabeledSpan(start=348, end=371, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
                tail=LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=262, end=276, label="data", score=1.0),
                tail=LabeledSpan(start=78, end=261, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=212, end=213, label="data", score=1.0),
                tail=LabeledSpan(start=192, end=202, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=331, end=337, label="data", score=1.0),
                tail=LabeledSpan(start=350, end=364, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=50, end=51, label="data", score=1.0),
                tail=LabeledSpan(start=34, end=46, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=528, end=535, label="own_claim", score=1.0),
                tail=LabeledSpan(start=512, end=527, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=502, end=503, label="data", score=1.0),
                tail=LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=510, end=511, label="data", score=1.0),
                tail=LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=619, end=626, label="data", score=1.0),
                tail=LabeledSpan(start=627, end=634, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=174, end=176, label="data", score=1.0),
                tail=LabeledSpan(start=149, end=167, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=384, end=387, label="background_claim", score=1.0),
                tail=LabeledSpan(start=390, end=403, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=42, end=45, label="data", score=1.0),
                tail=LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=95, end=96, label="data", score=1.0),
                tail=LabeledSpan(start=88, end=92, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=499, end=500, label="data", score=1.0),
                tail=LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=776, end=787, label="background_claim", score=1.0),
                tail=LabeledSpan(start=799, end=814, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=737, end=738, label="data", score=1.0),
                tail=LabeledSpan(start=719, end=734, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=168, end=173, label="data", score=1.0),
                tail=LabeledSpan(start=142, end=158, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=106, end=120, label="background_claim", score=1.0),
                tail=LabeledSpan(start=136, end=148, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=41, end=117, label="data", score=1.0),
                tail=LabeledSpan(start=139, end=160, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=463, end=464, label="data", score=1.0),
                tail=LabeledSpan(start=474, end=521, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=66, end=76, label="own_claim", score=1.0),
                tail=LabeledSpan(start=78, end=89, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
                tail=LabeledSpan(start=719, end=734, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=783, end=814, label="data", score=1.0),
                tail=LabeledSpan(start=757, end=778, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=454, end=461, label="background_claim", score=1.0),
                tail=LabeledSpan(start=469, end=475, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=486, end=487, label="data", score=1.0),
                tail=LabeledSpan(start=454, end=461, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=95, end=96, label="data", score=1.0),
                tail=LabeledSpan(start=454, end=461, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=797, end=798, label="data", score=1.0),
                tail=LabeledSpan(start=776, end=787, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=217, end=228, label="data", score=1.0),
                tail=LabeledSpan(start=204, end=213, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=408, end=433, label="own_claim", score=1.0),
                tail=LabeledSpan(start=474, end=521, label="own_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=182, end=188, label="background_claim", score=1.0),
                tail=LabeledSpan(start=192, end=202, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=742, end=747, label="own_claim", score=1.0),
                tail=LabeledSpan(start=749, end=756, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=58, end=59, label="data", score=1.0),
                tail=LabeledSpan(start=45, end=52, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=70, end=78, label="data", score=1.0),
                tail=LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=307, end=308, label="data", score=1.0),
                tail=LabeledSpan(start=310, end=322, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=620, end=624, label="data", score=1.0),
                tail=LabeledSpan(start=625, end=636, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=118, end=133, label="own_claim", score=1.0),
                tail=LabeledSpan(start=97, end=111, label="own_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=157, end=160, label="data", score=1.0),
                tail=LabeledSpan(start=152, end=156, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=161, end=188, label="own_claim", score=1.0),
                tail=LabeledSpan(start=298, end=317, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=177, end=180, label="data", score=1.0),
                tail=LabeledSpan(start=512, end=516, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
        },
    }

    assert dict(fn) == {
        "labeled_spans": {
            LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
            LabeledSpan(start=33, end=37, label="own_claim", score=1.0),
            LabeledSpan(start=274, end=291, label="background_claim", score=1.0),
            LabeledSpan(start=519, end=527, label="background_claim", score=1.0),
            LabeledSpan(start=113, end=117, label="data", score=1.0),
            LabeledSpan(start=378, end=401, label="own_claim", score=1.0),
            LabeledSpan(start=136, end=148, label="own_claim", score=1.0),
            LabeledSpan(start=906, end=911, label="own_claim", score=1.0),
            LabeledSpan(start=41, end=52, label="background_claim", score=1.0),
            LabeledSpan(start=401, end=413, label="data", score=1.0),
            LabeledSpan(start=57, end=60, label="data", score=1.0),
            LabeledSpan(start=74, end=86, label="own_claim", score=1.0),
            LabeledSpan(start=589, end=606, label="background_claim", score=1.0),
            LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
            LabeledSpan(start=719, end=731, label="background_claim", score=1.0),
            LabeledSpan(start=498, end=501, label="data", score=1.0),
            LabeledSpan(start=509, end=512, label="data", score=1.0),
            LabeledSpan(start=365, end=386, label="background_claim", score=1.0),
            LabeledSpan(start=203, end=210, label="background_claim", score=1.0),
            LabeledSpan(start=427, end=439, label="background_claim", score=1.0),
            LabeledSpan(start=411, end=428, label="background_claim", score=1.0),
            LabeledSpan(start=42, end=76, label="own_claim", score=1.0),
            LabeledSpan(start=43, end=45, label="data", score=1.0),
            LabeledSpan(start=488, end=491, label="data", score=1.0),
            LabeledSpan(start=34, end=52, label="background_claim", score=1.0),
            LabeledSpan(start=180, end=227, label="own_claim", score=1.0),
            LabeledSpan(start=618, end=631, label="own_claim", score=1.0),
            LabeledSpan(start=118, end=138, label="own_claim", score=1.0),
            LabeledSpan(start=70, end=75, label="data", score=1.0),
            LabeledSpan(start=41, end=47, label="own_claim", score=1.0),
            LabeledSpan(start=1007, end=1010, label="data", score=1.0),
            LabeledSpan(start=342, end=349, label="background_claim", score=1.0),
            LabeledSpan(start=605, end=615, label="data", score=1.0),
            LabeledSpan(start=366, end=369, label="data", score=1.0),
            LabeledSpan(start=23, end=40, label="own_claim", score=1.0),
            LabeledSpan(start=348, end=371, label="background_claim", score=1.0),
            LabeledSpan(start=523, end=529, label="data", score=1.0),
            LabeledSpan(start=212, end=225, label="background_claim", score=1.0),
            LabeledSpan(start=1012, end=1018, label="data", score=1.0),
            LabeledSpan(start=262, end=270, label="own_claim", score=1.0),
            LabeledSpan(start=88, end=97, label="data", score=1.0),
            LabeledSpan(start=441, end=455, label="background_claim", score=1.0),
            LabeledSpan(start=838, end=862, label="background_claim", score=1.0),
            LabeledSpan(start=54, end=57, label="data", score=1.0),
            LabeledSpan(start=109, end=113, label="data", score=1.0),
            LabeledSpan(start=408, end=423, label="background_claim", score=1.0),
            LabeledSpan(start=687, end=691, label="own_claim", score=1.0),
            LabeledSpan(start=323, end=326, label="data", score=1.0),
            LabeledSpan(start=93, end=99, label="own_claim", score=1.0),
            LabeledSpan(start=815, end=831, label="background_claim", score=1.0),
            LabeledSpan(start=230, end=234, label="data", score=1.0),
            LabeledSpan(start=485, end=488, label="data", score=1.0),
            LabeledSpan(start=618, end=634, label="background_claim", score=1.0),
            LabeledSpan(start=636, end=664, label="background_claim", score=1.0),
            LabeledSpan(start=330, end=337, label="background_claim", score=1.0),
            LabeledSpan(start=306, end=309, label="data", score=1.0),
            LabeledSpan(start=572, end=584, label="background_claim", score=1.0),
            LabeledSpan(start=736, end=739, label="data", score=1.0),
            LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
            LabeledSpan(start=722, end=725, label="data", score=1.0),
            LabeledSpan(start=501, end=504, label="data", score=1.0),
            LabeledSpan(start=578, end=588, label="data", score=1.0),
            LabeledSpan(start=207, end=209, label="data", score=1.0),
            LabeledSpan(start=217, end=228, label="own_claim", score=1.0),
            LabeledSpan(start=302, end=309, label="data", score=1.0),
            LabeledSpan(start=49, end=64, label="own_claim", score=1.0),
            LabeledSpan(start=509, end=516, label="background_claim", score=1.0),
            LabeledSpan(start=310, end=341, label="background_claim", score=1.0),
            LabeledSpan(start=682, end=691, label="own_claim", score=1.0),
            LabeledSpan(start=387, end=390, label="data", score=1.0),
            LabeledSpan(start=60, end=70, label="background_claim", score=1.0),
            LabeledSpan(start=143, end=174, label="background_claim", score=1.0),
            LabeledSpan(start=981, end=1006, label="background_claim", score=1.0),
            LabeledSpan(start=642, end=660, label="own_claim", score=1.0),
            LabeledSpan(start=480, end=508, label="background_claim", score=1.0),
            LabeledSpan(start=662, end=674, label="own_claim", score=1.0),
            LabeledSpan(start=210, end=211, label="data", score=1.0),
            LabeledSpan(start=862, end=865, label="data", score=1.0),
            LabeledSpan(start=787, end=814, label="background_claim", score=1.0),
            LabeledSpan(start=761, end=775, label="background_claim", score=1.0),
            LabeledSpan(start=415, end=454, label="own_claim", score=1.0),
            LabeledSpan(start=800, end=814, label="background_claim", score=1.0),
            LabeledSpan(start=94, end=97, label="data", score=1.0),
            LabeledSpan(start=704, end=728, label="own_claim", score=1.0),
            LabeledSpan(start=696, end=720, label="background_claim", score=1.0),
            LabeledSpan(start=235, end=239, label="data", score=1.0),
            LabeledSpan(start=796, end=799, label="data", score=1.0),
            LabeledSpan(start=76, end=78, label="data", score=1.0),
            LabeledSpan(start=454, end=475, label="background_claim", score=1.0),
            LabeledSpan(start=181, end=184, label="data", score=1.0),
            LabeledSpan(start=182, end=202, label="background_claim", score=1.0),
            LabeledSpan(start=626, end=633, label="own_claim", score=1.0),
            LabeledSpan(start=726, end=747, label="data", score=1.0),
        },
        "binary_relations": {
            BinaryRelation(
                head=LabeledSpan(start=60, end=70, label="background_claim", score=1.0),
                tail=LabeledSpan(start=74, end=86, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=365, end=386, label="background_claim", score=1.0),
                tail=LabeledSpan(start=427, end=439, label="background_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
                tail=LabeledSpan(start=519, end=527, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=57, end=60, label="data", score=1.0),
                tail=LabeledSpan(start=41, end=52, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=54, end=57, label="data", score=1.0),
                tail=LabeledSpan(start=41, end=52, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=96, end=105, label="data", score=1.0),
                tail=LabeledSpan(start=74, end=87, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=488, end=491, label="data", score=1.0),
                tail=LabeledSpan(start=454, end=475, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=70, end=75, label="data", score=1.0),
                tail=LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=488, end=491, label="data", score=1.0),
                tail=LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
                tail=LabeledSpan(start=719, end=731, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
                tail=LabeledSpan(start=761, end=775, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=302, end=309, label="data", score=1.0),
                tail=LabeledSpan(start=310, end=341, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=235, end=239, label="data", score=1.0),
                tail=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=605, end=615, label="data", score=1.0),
                tail=LabeledSpan(start=618, end=631, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=366, end=369, label="data", score=1.0),
                tail=LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
                tail=LabeledSpan(start=136, end=148, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=42, end=76, label="own_claim", score=1.0),
                tail=LabeledSpan(start=78, end=89, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
                tail=LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=212, end=213, label="data", score=1.0),
                tail=LabeledSpan(start=203, end=210, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=118, end=138, label="own_claim", score=1.0),
                tail=LabeledSpan(start=139, end=160, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=323, end=326, label="data", score=1.0),
                tail=LabeledSpan(start=326, end=350, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=181, end=184, label="data", score=1.0),
                tail=LabeledSpan(start=143, end=174, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=93, end=99, label="own_claim", score=1.0),
                tail=LabeledSpan(start=906, end=911, label="own_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=330, end=337, label="background_claim", score=1.0),
                tail=LabeledSpan(start=342, end=349, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=441, end=455, label="background_claim", score=1.0),
                tail=LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=230, end=234, label="data", score=1.0),
                tail=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=43, end=45, label="data", score=1.0),
                tail=LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=498, end=501, label="data", score=1.0),
                tail=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=207, end=209, label="data", score=1.0),
                tail=LabeledSpan(start=212, end=225, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=1007, end=1010, label="data", score=1.0),
                tail=LabeledSpan(start=1012, end=1018, label="data", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=33, end=37, label="own_claim", score=1.0),
                tail=LabeledSpan(start=93, end=99, label="own_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=626, end=633, label="own_claim", score=1.0),
                tail=LabeledSpan(start=687, end=691, label="own_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=94, end=97, label="data", score=1.0),
                tail=LabeledSpan(start=97, end=126, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=207, end=209, label="data", score=1.0),
                tail=LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=401, end=413, label="data", score=1.0),
                tail=LabeledSpan(start=415, end=454, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=292, end=298, label="background_claim", score=1.0),
                tail=LabeledSpan(start=274, end=291, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=518, end=521, label="data", score=1.0),
                tail=LabeledSpan(start=509, end=516, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=618, end=631, label="own_claim", score=1.0),
                tail=LabeledSpan(start=682, end=691, label="own_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=636, end=664, label="background_claim", score=1.0),
                tail=LabeledSpan(start=618, end=634, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=796, end=799, label="data", score=1.0),
                tail=LabeledSpan(start=800, end=814, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
                tail=LabeledSpan(start=182, end=202, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=578, end=588, label="data", score=1.0),
                tail=LabeledSpan(start=589, end=606, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=523, end=529, label="data", score=1.0),
                tail=LabeledSpan(start=530, end=542, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=88, end=97, label="data", score=1.0),
                tail=LabeledSpan(start=97, end=111, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=862, end=865, label="data", score=1.0),
                tail=LabeledSpan(start=838, end=862, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=522, end=526, label="data", score=1.0),
                tail=LabeledSpan(start=509, end=516, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=722, end=725, label="data", score=1.0),
                tail=LabeledSpan(start=696, end=720, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=509, end=512, label="data", score=1.0),
                tail=LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=662, end=674, label="own_claim", score=1.0),
                tail=LabeledSpan(start=642, end=660, label="own_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=262, end=270, label="own_claim", score=1.0),
                tail=LabeledSpan(start=255, end=261, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
                tail=LabeledSpan(start=480, end=508, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=113, end=117, label="data", score=1.0),
                tail=LabeledSpan(start=118, end=138, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=217, end=228, label="own_claim", score=1.0),
                tail=LabeledSpan(start=204, end=213, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=736, end=739, label="data", score=1.0),
                tail=LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=310, end=322, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=726, end=747, label="data", score=1.0),
                tail=LabeledSpan(start=749, end=756, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=76, end=78, label="data", score=1.0),
                tail=LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=352, end=365, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=509, end=516, label="background_claim", score=1.0),
                tail=LabeledSpan(start=528, end=532, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=387, end=390, label="data", score=1.0),
                tail=LabeledSpan(start=390, end=403, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=485, end=488, label="data", score=1.0),
                tail=LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                tail=LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=210, end=211, label="data", score=1.0),
                tail=LabeledSpan(start=212, end=225, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=501, end=504, label="data", score=1.0),
                tail=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=109, end=113, label="data", score=1.0),
                tail=LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=49, end=64, label="own_claim", score=1.0),
                tail=LabeledSpan(start=41, end=47, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=210, end=211, label="data", score=1.0),
                tail=LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=306, end=309, label="data", score=1.0),
                tail=LabeledSpan(start=310, end=322, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=485, end=488, label="data", score=1.0),
                tail=LabeledSpan(start=454, end=475, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=343, end=347, label="data", score=1.0),
                tail=LabeledSpan(start=348, end=371, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=390, end=403, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=427, end=439, label="background_claim", score=1.0),
                tail=LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
        },
    }

    assert dict(tp) == {
        "labeled_spans": {
            LabeledSpan(start=757, end=778, label="own_claim", score=1.0),
            LabeledSpan(start=212, end=213, label="data", score=1.0),
            LabeledSpan(start=88, end=92, label="background_claim", score=1.0),
            LabeledSpan(start=310, end=322, label="background_claim", score=1.0),
            LabeledSpan(start=292, end=298, label="background_claim", score=1.0),
            LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
            LabeledSpan(start=302, end=312, label="background_claim", score=1.0),
            LabeledSpan(start=317, end=329, label="background_claim", score=1.0),
            LabeledSpan(start=352, end=365, label="background_claim", score=1.0),
            LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
            LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
            LabeledSpan(start=204, end=213, label="own_claim", score=1.0),
            LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
            LabeledSpan(start=255, end=261, label="own_claim", score=1.0),
            LabeledSpan(start=530, end=542, label="own_claim", score=1.0),
            LabeledSpan(start=128, end=133, label="data", score=1.0),
            LabeledSpan(start=97, end=111, label="own_claim", score=1.0),
            LabeledSpan(start=96, end=105, label="data", score=1.0),
            LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
            LabeledSpan(start=97, end=126, label="background_claim", score=1.0),
            LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
            LabeledSpan(start=230, end=247, label="own_claim", score=1.0),
            LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
            LabeledSpan(start=749, end=756, label="own_claim", score=1.0),
            LabeledSpan(start=30, end=41, label="background_claim", score=1.0),
            LabeledSpan(start=53, end=73, label="background_claim", score=1.0),
            LabeledSpan(start=134, end=141, label="own_claim", score=1.0),
            LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
            LabeledSpan(start=518, end=521, label="data", score=1.0),
            LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
            LabeledSpan(start=343, end=347, label="data", score=1.0),
            LabeledSpan(start=122, end=127, label="data", score=1.0),
            LabeledSpan(start=249, end=254, label="data", score=1.0),
            LabeledSpan(start=522, end=526, label="data", score=1.0),
            LabeledSpan(start=74, end=87, label="background_claim", score=1.0),
            LabeledSpan(start=90, end=95, label="data", score=1.0),
            LabeledSpan(start=78, end=89, label="own_claim", score=1.0),
            LabeledSpan(start=139, end=160, label="own_claim", score=1.0),
            LabeledSpan(start=326, end=350, label="background_claim", score=1.0),
            LabeledSpan(start=390, end=403, label="background_claim", score=1.0),
            LabeledSpan(start=528, end=532, label="background_claim", score=1.0),
        },
        "binary_relations": {
            BinaryRelation(
                head=LabeledSpan(start=122, end=127, label="data", score=1.0),
                tail=LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=88, end=92, label="background_claim", score=1.0),
                tail=LabeledSpan(start=97, end=126, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=326, end=350, label="background_claim", score=1.0),
                tail=LabeledSpan(start=352, end=365, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=128, end=133, label="data", score=1.0),
                tail=LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=230, end=247, label="own_claim", score=1.0),
                tail=LabeledSpan(start=204, end=213, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=249, end=254, label="data", score=1.0),
                tail=LabeledSpan(start=255, end=261, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=90, end=95, label="data", score=1.0),
                tail=LabeledSpan(start=74, end=87, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=302, end=312, label="background_claim", score=1.0),
                tail=LabeledSpan(start=292, end=298, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=90, end=95, label="data", score=1.0),
                tail=LabeledSpan(start=78, end=89, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
        },
    }
