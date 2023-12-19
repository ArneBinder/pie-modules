import logging
import pickle
import re
from collections import defaultdict

import pytest
import torch
from pytorch_ie.annotations import BinaryRelation, LabeledSpan

from pie_modules.models import PointerNetworkModel, SimplePointerNetworkModel
from pie_modules.taskmodules import PointerNetworkTaskModule
from tests import DUMP_FIXTURE_DATA, FIXTURES_ROOT
from tests.models.test_pointer_network_model_predict import (
    MODEL_PATH as OTHER_MODEL_PATH,
)
from tests.models.test_simple_pointer_network_predict import MODEL_PATH

logger = logging.getLogger(__name__)


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
    trained_model, sciarg_batch, loaded_taskmodule
):
    replace_weights_with_other_model(
        trained_model, PointerNetworkModel.from_pretrained(OTHER_MODEL_PATH)
    )

    torch.manual_seed(42)
    inputs, targets = sciarg_batch
    inputs_truncated = {k: v[:5] for k, v in inputs.items()}
    targets_truncated = {k: v[:5] for k, v in targets.items()}
    generation_kwargs = {"num_beams": 4, "max_length": 512}
    prediction = trained_model.predict(inputs_truncated, **generation_kwargs)
    assert prediction is not None
    targets_list = targets_truncated["tgt_tokens"].tolist()
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
        "labeled_spans": 91,
        "binary_relations": 48,
    }
    assert {layer_name: len(anns) for layer_name, anns in fn.items()} == {
        "labeled_spans": 66,
        "binary_relations": 57,
    }
    assert {layer_name: len(anns) for layer_name, anns in tp.items()} == {
        "labeled_spans": 27,
        "binary_relations": 5,
    }

    # check the actual annotations
    assert dict(fp) == {
        "labeled_spans": {
            LabeledSpan(start=692, end=728, label="own_claim", score=1.0),
            LabeledSpan(start=533, end=556, label="own_claim", score=1.0),
            LabeledSpan(start=180, end=200, label="background_claim", score=1.0),
            LabeledSpan(start=797, end=798, label="data", score=1.0),
            LabeledSpan(start=365, end=370, label="background_claim", score=1.0),
            LabeledSpan(start=457, end=466, label="own_claim", score=1.0),
            LabeledSpan(start=639, end=660, label="own_claim", score=1.0),
            LabeledSpan(start=342, end=349, label="own_claim", score=1.0),
            LabeledSpan(start=367, end=368, label="data", score=1.0),
            LabeledSpan(start=585, end=599, label="own_claim", score=1.0),
            LabeledSpan(start=258, end=270, label="background_claim", score=1.0),
            LabeledSpan(start=236, end=246, label="own_claim", score=1.0),
            LabeledSpan(start=340, end=341, label="data", score=1.0),
            LabeledSpan(start=486, end=498, label="own_claim", score=1.0),
            LabeledSpan(start=372, end=376, label="data", score=1.0),
            LabeledSpan(start=271, end=289, label="background_claim", score=1.0),
            LabeledSpan(start=348, end=371, label="own_claim", score=1.0),
            LabeledSpan(start=114, end=133, label="background_claim", score=1.0),
            LabeledSpan(start=70, end=78, label="data", score=1.0),
            LabeledSpan(start=169, end=170, label="data", score=1.0),
            LabeledSpan(start=380, end=386, label="background_claim", score=1.0),
            LabeledSpan(start=42, end=45, label="data", score=1.0),
            LabeledSpan(start=501, end=502, label="data", score=1.0),
            LabeledSpan(start=230, end=239, label="background_claim", score=1.0),
            LabeledSpan(start=510, end=511, label="data", score=1.0),
            LabeledSpan(start=502, end=503, label="data", score=1.0),
            LabeledSpan(start=649, end=691, label="own_claim", score=1.0),
            LabeledSpan(start=799, end=814, label="own_claim", score=1.0),
            LabeledSpan(start=324, end=325, label="data", score=1.0),
            LabeledSpan(start=719, end=734, label="own_claim", score=1.0),
            LabeledSpan(start=79, end=87, label="background_claim", score=1.0),
            LabeledSpan(start=331, end=337, label="data", score=1.0),
            LabeledSpan(start=240, end=250, label="background_claim", score=1.0),
            LabeledSpan(start=632, end=641, label="own_claim", score=1.0),
            LabeledSpan(start=34, end=46, label="background_claim", score=1.0),
            LabeledSpan(start=636, end=664, label="own_claim", score=1.0),
            LabeledSpan(start=307, end=308, label="data", score=1.0),
            LabeledSpan(start=511, end=526, label="own_claim", score=1.0),
            LabeledSpan(start=372, end=376, label="own_claim", score=1.0),
            LabeledSpan(start=747, end=756, label="own_claim", score=1.0),
            LabeledSpan(start=486, end=487, label="data", score=1.0),
            LabeledSpan(start=407, end=432, label="own_claim", score=1.0),
            LabeledSpan(start=201, end=217, label="background_claim", score=1.0),
            LabeledSpan(start=635, end=638, label="data", score=1.0),
            LabeledSpan(start=153, end=81, label="own_claim", score=1.0),
            LabeledSpan(start=456, end=478, label="own_claim", score=1.0),
            LabeledSpan(start=107, end=126, label="background_claim", score=1.0),
            LabeledSpan(start=384, end=387, label="background_claim", score=1.0),
            LabeledSpan(start=441, end=455, label="data", score=1.0),
            LabeledSpan(start=233, end=253, label="background_claim", score=1.0),
            LabeledSpan(start=50, end=51, label="data", score=1.0),
            LabeledSpan(start=95, end=96, label="data", score=1.0),
            LabeledSpan(start=443, end=453, label="background_claim", score=1.0),
            LabeledSpan(start=627, end=634, label="own_claim", score=1.0),
            LabeledSpan(start=388, end=389, label="data", score=1.0),
            LabeledSpan(start=174, end=176, label="data", score=1.0),
            LabeledSpan(start=411, end=441, label="background_claim", score=1.0),
            LabeledSpan(start=350, end=364, label="own_claim", score=1.0),
            LabeledSpan(start=480, end=508, label="own_claim", score=1.0),
            LabeledSpan(start=127, end=144, label="background_claim", score=1.0),
            LabeledSpan(start=761, end=775, label="own_claim", score=1.0),
            LabeledSpan(start=619, end=633, label="own_claim", score=1.0),
            LabeledSpan(start=192, end=202, label="background_claim", score=1.0),
            LabeledSpan(start=227, end=235, label="data", score=1.0),
            LabeledSpan(start=136, end=148, label="background_claim", score=1.0),
            LabeledSpan(start=607, end=617, label="own_claim", score=1.0),
            LabeledSpan(start=175, end=181, label="own_claim", score=1.0),
            LabeledSpan(start=47, end=49, label="data", score=1.0),
            LabeledSpan(start=528, end=565, label="own_claim", score=1.0),
            LabeledSpan(start=153, end=174, label="own_claim", score=1.0),
            LabeledSpan(start=737, end=738, label="data", score=1.0),
            LabeledSpan(start=177, end=180, label="data", score=1.0),
            LabeledSpan(start=514, end=527, label="background_claim", score=1.0),
            LabeledSpan(start=667, end=674, label="own_claim", score=1.0),
            LabeledSpan(start=146, end=148, label="background_claim", score=1.0),
            LabeledSpan(start=601, end=615, label="own_claim", score=1.0),
            LabeledSpan(start=489, end=490, label="data", score=1.0),
            LabeledSpan(start=182, end=188, label="background_claim", score=1.0),
            LabeledSpan(start=589, end=606, label="own_claim", score=1.0),
            LabeledSpan(start=276, end=300, label="own_claim", score=1.0),
            LabeledSpan(start=310, end=326, label="own_claim", score=1.0),
            LabeledSpan(start=212, end=219, label="own_claim", score=1.0),
            LabeledSpan(start=149, end=167, label="background_claim", score=1.0),
            LabeledSpan(start=512, end=516, label="own_claim", score=1.0),
            LabeledSpan(start=782, end=794, label="own_claim", score=1.0),
            LabeledSpan(start=427, end=439, label="own_claim", score=1.0),
            LabeledSpan(start=251, end=260, label="background_claim", score=1.0),
            LabeledSpan(start=499, end=500, label="data", score=1.0),
            LabeledSpan(start=890, end=842, label="own_claim", score=1.0),
            LabeledSpan(start=619, end=626, label="data", score=1.0),
            LabeledSpan(start=221, end=225, label="data", score=1.0),
        },
        "binary_relations": {
            BinaryRelation(
                head=LabeledSpan(start=372, end=376, label="own_claim", score=1.0),
                tail=LabeledSpan(start=407, end=432, label="own_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=367, end=368, label="data", score=1.0),
                tail=LabeledSpan(start=258, end=270, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=441, end=455, label="data", score=1.0),
                tail=LabeledSpan(start=456, end=478, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=342, end=349, label="own_claim", score=1.0),
                tail=LabeledSpan(start=350, end=364, label="own_claim", score=1.0),
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
                head=LabeledSpan(start=331, end=337, label="data", score=1.0),
                tail=LabeledSpan(start=350, end=364, label="own_claim", score=1.0),
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
                head=LabeledSpan(start=489, end=490, label="data", score=1.0),
                tail=LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=365, end=370, label="background_claim", score=1.0),
                tail=LabeledSpan(start=380, end=386, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=174, end=176, label="data", score=1.0),
                tail=LabeledSpan(start=149, end=167, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=639, end=660, label="own_claim", score=1.0),
                tail=LabeledSpan(start=667, end=674, label="own_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=230, end=239, label="background_claim", score=1.0),
                tail=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=114, end=133, label="background_claim", score=1.0),
                tail=LabeledSpan(start=136, end=148, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=227, end=235, label="data", score=1.0),
                tail=LabeledSpan(start=212, end=219, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=510, end=511, label="data", score=1.0),
                tail=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=499, end=500, label="data", score=1.0),
                tail=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=456, end=478, label="own_claim", score=1.0),
                tail=LabeledSpan(start=480, end=508, label="own_claim", score=1.0),
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
                head=LabeledSpan(start=212, end=213, label="data", score=1.0),
                tail=LabeledSpan(start=192, end=202, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=182, end=188, label="background_claim", score=1.0),
                tail=LabeledSpan(start=192, end=202, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=302, end=309, label="data", score=1.0),
                tail=LabeledSpan(start=310, end=326, label="own_claim", score=1.0),
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
                head=LabeledSpan(start=47, end=49, label="data", score=1.0),
                tail=LabeledSpan(start=34, end=46, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=782, end=794, label="own_claim", score=1.0),
                tail=LabeledSpan(start=799, end=814, label="own_claim", score=1.0),
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
                head=LabeledSpan(start=372, end=376, label="data", score=1.0),
                tail=LabeledSpan(start=365, end=370, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=340, end=341, label="data", score=1.0),
                tail=LabeledSpan(start=331, end=337, label="data", score=1.0),
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
                head=LabeledSpan(start=502, end=503, label="data", score=1.0),
                tail=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=619, end=626, label="data", score=1.0),
                tail=LabeledSpan(start=627, end=634, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=501, end=502, label="data", score=1.0),
                tail=LabeledSpan(start=486, end=498, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=486, end=487, label="data", score=1.0),
                tail=LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=42, end=45, label="data", score=1.0),
                tail=LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=127, end=144, label="background_claim", score=1.0),
                tail=LabeledSpan(start=107, end=126, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=221, end=225, label="data", score=1.0),
                tail=LabeledSpan(start=212, end=219, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=737, end=738, label="data", score=1.0),
                tail=LabeledSpan(start=719, end=734, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=175, end=181, label="own_claim", score=1.0),
                tail=LabeledSpan(start=153, end=174, label="own_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=635, end=638, label="data", score=1.0),
                tail=LabeledSpan(start=639, end=660, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=324, end=325, label="data", score=1.0),
                tail=LabeledSpan(start=233, end=253, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=522, end=526, label="data", score=1.0),
                tail=LabeledSpan(start=512, end=516, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                tail=LabeledSpan(start=514, end=527, label="background_claim", score=1.0),
                label="parts_of_same",
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
                head=LabeledSpan(start=443, end=453, label="background_claim", score=1.0),
                tail=LabeledSpan(start=411, end=441, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=146, end=148, label="background_claim", score=1.0),
                tail=LabeledSpan(start=127, end=144, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=797, end=798, label="data", score=1.0),
                tail=LabeledSpan(start=782, end=794, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
        },
    }

    assert dict(fn) == {
        "labeled_spans": {
            LabeledSpan(start=230, end=234, label="data", score=1.0),
            LabeledSpan(start=323, end=326, label="data", score=1.0),
            LabeledSpan(start=342, end=349, label="background_claim", score=1.0),
            LabeledSpan(start=143, end=174, label="background_claim", score=1.0),
            LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
            LabeledSpan(start=485, end=488, label="data", score=1.0),
            LabeledSpan(start=480, end=508, label="background_claim", score=1.0),
            LabeledSpan(start=800, end=814, label="background_claim", score=1.0),
            LabeledSpan(start=687, end=691, label="own_claim", score=1.0),
            LabeledSpan(start=626, end=633, label="own_claim", score=1.0),
            LabeledSpan(start=348, end=371, label="background_claim", score=1.0),
            LabeledSpan(start=761, end=775, label="background_claim", score=1.0),
            LabeledSpan(start=212, end=225, label="background_claim", score=1.0),
            LabeledSpan(start=306, end=309, label="data", score=1.0),
            LabeledSpan(start=441, end=455, label="background_claim", score=1.0),
            LabeledSpan(start=736, end=739, label="data", score=1.0),
            LabeledSpan(start=326, end=350, label="background_claim", score=1.0),
            LabeledSpan(start=128, end=133, label="data", score=1.0),
            LabeledSpan(start=136, end=148, label="own_claim", score=1.0),
            LabeledSpan(start=498, end=501, label="data", score=1.0),
            LabeledSpan(start=501, end=504, label="data", score=1.0),
            LabeledSpan(start=182, end=202, label="background_claim", score=1.0),
            LabeledSpan(start=578, end=588, label="data", score=1.0),
            LabeledSpan(start=519, end=527, label="background_claim", score=1.0),
            LabeledSpan(start=528, end=532, label="background_claim", score=1.0),
            LabeledSpan(start=207, end=209, label="data", score=1.0),
            LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
            LabeledSpan(start=509, end=512, label="data", score=1.0),
            LabeledSpan(start=43, end=45, label="data", score=1.0),
            LabeledSpan(start=618, end=634, label="background_claim", score=1.0),
            LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
            LabeledSpan(start=488, end=491, label="data", score=1.0),
            LabeledSpan(start=330, end=337, label="background_claim", score=1.0),
            LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
            LabeledSpan(start=387, end=390, label="data", score=1.0),
            LabeledSpan(start=518, end=521, label="data", score=1.0),
            LabeledSpan(start=352, end=365, label="background_claim", score=1.0),
            LabeledSpan(start=636, end=664, label="background_claim", score=1.0),
            LabeledSpan(start=1007, end=1010, label="data", score=1.0),
            LabeledSpan(start=70, end=75, label="data", score=1.0),
            LabeledSpan(start=210, end=211, label="data", score=1.0),
            LabeledSpan(start=589, end=606, label="background_claim", score=1.0),
            LabeledSpan(start=572, end=584, label="background_claim", score=1.0),
            LabeledSpan(start=122, end=127, label="data", score=1.0),
            LabeledSpan(start=366, end=369, label="data", score=1.0),
            LabeledSpan(start=605, end=615, label="data", score=1.0),
            LabeledSpan(start=682, end=691, label="own_claim", score=1.0),
            LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
            LabeledSpan(start=719, end=731, label="background_claim", score=1.0),
            LabeledSpan(start=796, end=799, label="data", score=1.0),
            LabeledSpan(start=94, end=97, label="data", score=1.0),
            LabeledSpan(start=365, end=386, label="background_claim", score=1.0),
            LabeledSpan(start=1012, end=1018, label="data", score=1.0),
            LabeledSpan(start=618, end=631, label="own_claim", score=1.0),
            LabeledSpan(start=235, end=239, label="data", score=1.0),
            LabeledSpan(start=427, end=439, label="background_claim", score=1.0),
            LabeledSpan(start=642, end=660, label="own_claim", score=1.0),
            LabeledSpan(start=310, end=341, label="background_claim", score=1.0),
            LabeledSpan(start=76, end=78, label="data", score=1.0),
            LabeledSpan(start=203, end=210, label="background_claim", score=1.0),
            LabeledSpan(start=662, end=674, label="own_claim", score=1.0),
            LabeledSpan(start=411, end=428, label="background_claim", score=1.0),
            LabeledSpan(start=181, end=184, label="data", score=1.0),
            LabeledSpan(start=509, end=516, label="background_claim", score=1.0),
            LabeledSpan(start=704, end=728, label="own_claim", score=1.0),
            LabeledSpan(start=34, end=52, label="background_claim", score=1.0),
        },
        "binary_relations": {
            BinaryRelation(
                head=LabeledSpan(start=210, end=211, label="data", score=1.0),
                tail=LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=387, end=390, label="data", score=1.0),
                tail=LabeledSpan(start=390, end=403, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=366, end=369, label="data", score=1.0),
                tail=LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=626, end=633, label="own_claim", score=1.0),
                tail=LabeledSpan(start=687, end=691, label="own_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=212, end=213, label="data", score=1.0),
                tail=LabeledSpan(start=203, end=210, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=235, end=239, label="data", score=1.0),
                tail=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
                tail=LabeledSpan(start=480, end=508, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=230, end=234, label="data", score=1.0),
                tail=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=578, end=588, label="data", score=1.0),
                tail=LabeledSpan(start=589, end=606, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                tail=LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=207, end=209, label="data", score=1.0),
                tail=LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
                tail=LabeledSpan(start=182, end=202, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=605, end=615, label="data", score=1.0),
                tail=LabeledSpan(start=618, end=631, label="own_claim", score=1.0),
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
                head=LabeledSpan(start=498, end=501, label="data", score=1.0),
                tail=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=94, end=97, label="data", score=1.0),
                tail=LabeledSpan(start=97, end=126, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=509, end=516, label="background_claim", score=1.0),
                tail=LabeledSpan(start=528, end=532, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
                tail=LabeledSpan(start=719, end=731, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=330, end=337, label="background_claim", score=1.0),
                tail=LabeledSpan(start=342, end=349, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=427, end=439, label="background_claim", score=1.0),
                tail=LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=76, end=78, label="data", score=1.0),
                tail=LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=210, end=211, label="data", score=1.0),
                tail=LabeledSpan(start=212, end=225, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=796, end=799, label="data", score=1.0),
                tail=LabeledSpan(start=800, end=814, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
                tail=LabeledSpan(start=761, end=775, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=352, end=365, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=122, end=127, label="data", score=1.0),
                tail=LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
                label="supports",
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
                head=LabeledSpan(start=441, end=455, label="background_claim", score=1.0),
                tail=LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=292, end=298, label="background_claim", score=1.0),
                tail=LabeledSpan(start=274, end=291, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=43, end=45, label="data", score=1.0),
                tail=LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=636, end=664, label="background_claim", score=1.0),
                tail=LabeledSpan(start=618, end=634, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=522, end=526, label="data", score=1.0),
                tail=LabeledSpan(start=509, end=516, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=736, end=739, label="data", score=1.0),
                tail=LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=326, end=350, label="background_claim", score=1.0),
                tail=LabeledSpan(start=352, end=365, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=70, end=75, label="data", score=1.0),
                tail=LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=365, end=386, label="background_claim", score=1.0),
                tail=LabeledSpan(start=427, end=439, label="background_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
        },
    }

    assert dict(tp) == {
        "labeled_spans": {
            LabeledSpan(start=310, end=322, label="background_claim", score=1.0),
            LabeledSpan(start=292, end=298, label="background_claim", score=1.0),
            LabeledSpan(start=302, end=312, label="background_claim", score=1.0),
            LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
            LabeledSpan(start=53, end=73, label="background_claim", score=1.0),
            LabeledSpan(start=212, end=213, label="data", score=1.0),
            LabeledSpan(start=317, end=329, label="background_claim", score=1.0),
            LabeledSpan(start=88, end=92, label="background_claim", score=1.0),
            LabeledSpan(start=302, end=309, label="data", score=1.0),
            LabeledSpan(start=74, end=87, label="background_claim", score=1.0),
            LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
            LabeledSpan(start=96, end=105, label="data", score=1.0),
            LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
            LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
            LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
            LabeledSpan(start=343, end=347, label="data", score=1.0),
            LabeledSpan(start=522, end=526, label="data", score=1.0),
            LabeledSpan(start=274, end=291, label="background_claim", score=1.0),
            LabeledSpan(start=90, end=95, label="data", score=1.0),
            LabeledSpan(start=390, end=403, label="background_claim", score=1.0),
            LabeledSpan(start=454, end=475, label="background_claim", score=1.0),
            LabeledSpan(start=97, end=126, label="background_claim", score=1.0),
            LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
            LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
            LabeledSpan(start=408, end=423, label="background_claim", score=1.0),
            LabeledSpan(start=30, end=41, label="background_claim", score=1.0),
            LabeledSpan(start=109, end=113, label="data", score=1.0),
        },
        "binary_relations": {
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
                head=LabeledSpan(start=96, end=105, label="data", score=1.0),
                tail=LabeledSpan(start=74, end=87, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=109, end=113, label="data", score=1.0),
                tail=LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
        },
    }