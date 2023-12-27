import json
import logging
import os
import pickle
from collections import defaultdict

import pytest
import torch
from pytorch_ie.annotations import BinaryRelation, LabeledSpan

from pie_modules.models import SimplePointerNetworkModel
from pie_modules.taskmodules import PointerNetworkTaskModuleForEnd2EndRE
from tests import DUMP_FIXTURE_DATA, FIXTURES_ROOT

logger = logging.getLogger(__name__)


# wandb run: https://wandb.ai/arne/dataset-sciarg-task-ner_re-training/runs/terkqzyn
MODEL_PATH = "/home/arbi01/projects/pie-document-level/models/dataset-sciarg/task-ner_re/2023-12-18_22-15-53"
MODEL_PATH_WITH_POSITION_ID_PATTERN = "/home/arbi01/projects/pie-document-level/models/dataset-sciarg/task-ner_re/2023-12-18_22-15-53_position_id_pattern"
SCIARG_BATCH_PATH = (
    FIXTURES_ROOT
    / "taskmodules"
    / "pointer_network_taskmodule_for_end2end_re"
    / "sciarg_batch_encoding.json"
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
def loaded_taskmodule():
    taskmodule = PointerNetworkTaskModuleForEnd2EndRE.from_pretrained(MODEL_PATH)
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
    batch_serializable = tuple({k: v.tolist() for k, v in item.items()} for item in batch)
    with open(SCIARG_BATCH_PATH, "w") as f:
        json.dump(batch_serializable, f, sort_keys=True)


@pytest.fixture(scope="module")
def sciarg_batch_truncated():
    with open(SCIARG_BATCH_PATH) as f:
        batch_json = json.load(f)
    inputs_json, targets_json = batch_json
    inputs_truncated = {k: torch.tensor(v[:5]) for k, v in inputs_json.items()}
    targets_truncated = {k: torch.tensor(v[:5]) for k, v in targets_json.items()}
    return inputs_truncated, targets_truncated


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


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason=f"model not available: {MODEL_PATH}")
def test_sciarg_metric(loaded_taskmodule, sciarg_batch_truncated, sciarg_batch_prediction):
    inputs, targets = sciarg_batch_truncated
    metric = loaded_taskmodule.configure_metric()
    metric.update(sciarg_batch_prediction, targets["tgt_tokens"])

    values = metric.compute()

    tp = defaultdict(set)
    fp = defaultdict(set)
    fn = defaultdict(set)
    for layer_name, anns_dict in metric.layer_metrics.items():
        tp[layer_name].update(anns_dict.correct)
        fp[layer_name].update(set(anns_dict.predicted) - set(anns_dict.correct))
        fn[layer_name].update(set(anns_dict.gold) - set(anns_dict.correct))

    # check the numbers: tp, fp, fn
    assert {layer_name: len(anns) for layer_name, anns in tp.items()} == {
        "labeled_spans": 35,
        "binary_relations": 5,
    }
    assert {layer_name: len(anns) for layer_name, anns in fp.items()} == {
        "labeled_spans": 83,
        "binary_relations": 62,
    }
    assert {layer_name: len(anns) for layer_name, anns in fn.items()} == {
        "labeled_spans": 58,
        "binary_relations": 57,
    }

    # check the actual annotations
    assert dict(fp) == {
        "labeled_spans": {
            LabeledSpan(start=533, end=556, label="own_claim", score=1.0),
            LabeledSpan(start=372, end=376, label="data", score=1.0),
            LabeledSpan(start=776, end=787, label="background_claim", score=1.0),
            LabeledSpan(start=70, end=78, label="data", score=1.0),
            LabeledSpan(start=262, end=271, label="background_claim", score=1.0),
            LabeledSpan(start=236, end=246, label="own_claim", score=1.0),
            LabeledSpan(start=585, end=599, label="own_claim", score=1.0),
            LabeledSpan(start=873, end=913, label="own_claim", score=1.0),
            LabeledSpan(start=42, end=45, label="data", score=1.0),
            LabeledSpan(start=348, end=371, label="own_claim", score=1.0),
            LabeledSpan(start=158, end=179, label="background_claim", score=1.0),
            LabeledSpan(start=815, end=824, label="background_claim", score=1.0),
            LabeledSpan(start=528, end=633, label="own_claim", score=1.0),
            LabeledSpan(start=153, end=174, label="own_claim", score=1.0),
            LabeledSpan(start=149, end=167, label="background_claim", score=1.0),
            LabeledSpan(start=79, end=87, label="own_claim", score=1.0),
            LabeledSpan(start=251, end=260, label="background_claim", score=1.0),
            LabeledSpan(start=528, end=556, label="own_claim", score=1.0),
            LabeledSpan(start=572, end=584, label="own_claim", score=1.0),
            LabeledSpan(start=469, end=475, label="background_claim", score=1.0),
            LabeledSpan(start=180, end=200, label="background_claim", score=1.0),
            LabeledSpan(start=791, end=794, label="background_claim", score=1.0),
            LabeledSpan(start=365, end=370, label="background_claim", score=1.0),
            LabeledSpan(start=759, end=761, label="data", score=1.0),
            LabeledSpan(start=192, end=202, label="background_claim", score=1.0),
            LabeledSpan(start=114, end=133, label="background_claim", score=1.0),
            LabeledSpan(start=619, end=626, label="data", score=1.0),
            LabeledSpan(start=34, end=148, label="background_claim", score=1.0),
            LabeledSpan(start=46, end=59, label="own_claim", score=1.0),
            LabeledSpan(start=61, end=68, label="own_claim", score=1.0),
            LabeledSpan(start=797, end=798, label="data", score=1.0),
            LabeledSpan(start=826, end=829, label="background_claim", score=1.0),
            LabeledSpan(start=607, end=617, label="background_claim", score=1.0),
            LabeledSpan(start=127, end=148, label="background_claim", score=1.0),
            LabeledSpan(start=799, end=814, label="background_claim", score=1.0),
            LabeledSpan(start=230, end=239, label="background_claim", score=1.0),
            LabeledSpan(start=367, end=368, label="data", score=1.0),
            LabeledSpan(start=719, end=734, label="background_claim", score=1.0),
            LabeledSpan(start=854, end=913, label="own_claim", score=1.0),
            LabeledSpan(start=601, end=615, label="data", score=1.0),
            LabeledSpan(start=182, end=188, label="data", score=1.0),
            LabeledSpan(start=98, end=126, label="background_claim", score=1.0),
            LabeledSpan(start=240, end=250, label="background_claim", score=1.0),
            LabeledSpan(start=404, end=410, label="data", score=1.0),
            LabeledSpan(start=203, end=211, label="background_claim", score=1.0),
            LabeledSpan(start=429, end=441, label="background_claim", score=1.0),
            LabeledSpan(start=454, end=461, label="background_claim", score=1.0),
            LabeledSpan(start=737, end=738, label="data", score=1.0),
            LabeledSpan(start=510, end=511, label="data", score=1.0),
            LabeledSpan(start=502, end=503, label="data", score=1.0),
            LabeledSpan(start=201, end=217, label="background_claim", score=1.0),
            LabeledSpan(start=456, end=478, label="background_claim", score=1.0),
            LabeledSpan(start=512, end=527, label="background_claim", score=1.0),
            LabeledSpan(start=175, end=181, label="own_claim", score=1.0),
            LabeledSpan(start=24, end=33, label="background_claim", score=1.0),
            LabeledSpan(start=384, end=387, label="background_claim", score=1.0),
            LabeledSpan(start=182, end=183, label="data", score=1.0),
            LabeledSpan(start=143, end=149, label="own_claim", score=1.0),
            LabeledSpan(start=697, end=717, label="own_claim", score=1.0),
            LabeledSpan(start=212, end=219, label="background_claim", score=1.0),
            LabeledSpan(start=233, end=253, label="background_claim", score=1.0),
            LabeledSpan(start=619, end=664, label="data", score=1.0),
            LabeledSpan(start=307, end=308, label="data", score=1.0),
            LabeledSpan(start=489, end=490, label="data", score=1.0),
            LabeledSpan(start=627, end=634, label="background_claim", score=1.0),
            LabeledSpan(start=486, end=487, label="data", score=1.0),
            LabeledSpan(start=443, end=453, label="background_claim", score=1.0),
            LabeledSpan(start=558, end=570, label="own_claim", score=1.0),
            LabeledSpan(start=106, end=113, label="background_claim", score=1.0),
            LabeledSpan(start=589, end=606, label="own_claim", score=1.0),
            LabeledSpan(start=441, end=455, label="data", score=1.0),
            LabeledSpan(start=257, end=270, label="background_claim", score=1.0),
            LabeledSpan(start=350, end=364, label="background_claim", score=1.0),
            LabeledSpan(start=30, end=59, label="own_claim", score=1.0),
            LabeledSpan(start=95, end=96, label="data", score=1.0),
            LabeledSpan(start=271, end=289, label="background_claim", score=1.0),
            LabeledSpan(start=499, end=500, label="data", score=1.0),
            LabeledSpan(start=675, end=691, label="background_claim", score=1.0),
            LabeledSpan(start=642, end=691, label="own_claim", score=1.0),
            LabeledSpan(start=380, end=386, label="background_claim", score=1.0),
            LabeledSpan(start=388, end=389, label="data", score=1.0),
            LabeledSpan(start=136, end=148, label="background_claim", score=1.0),
            LabeledSpan(start=692, end=728, label="own_claim", score=1.0),
        },
        "binary_relations": {
            BinaryRelation(
                head=LabeledSpan(start=791, end=794, label="background_claim", score=1.0),
                tail=LabeledSpan(start=799, end=814, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=365, end=370, label="background_claim", score=1.0),
                tail=LabeledSpan(start=380, end=386, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=212, end=213, label="data", score=1.0),
                tail=LabeledSpan(start=317, end=329, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=619, end=664, label="data", score=1.0),
                tail=LabeledSpan(start=636, end=664, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=797, end=798, label="data", score=1.0),
                tail=LabeledSpan(start=791, end=794, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=486, end=487, label="data", score=1.0),
                tail=LabeledSpan(start=454, end=461, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=441, end=455, label="data", score=1.0),
                tail=LabeledSpan(start=456, end=478, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=210, end=211, label="data", score=1.0),
                tail=LabeledSpan(start=212, end=219, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=233, end=253, label="background_claim", score=1.0),
                tail=LabeledSpan(start=257, end=270, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=61, end=68, label="own_claim", score=1.0),
                tail=LabeledSpan(start=46, end=59, label="own_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=182, end=188, label="data", score=1.0),
                tail=LabeledSpan(start=192, end=202, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=367, end=368, label="data", score=1.0),
                tail=LabeledSpan(start=233, end=253, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
                tail=LabeledSpan(start=719, end=734, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=307, end=308, label="data", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=24, end=33, label="background_claim", score=1.0),
                tail=LabeledSpan(start=34, end=148, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=759, end=761, label="data", score=1.0),
                tail=LabeledSpan(start=761, end=775, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=342, end=349, label="background_claim", score=1.0),
                tail=LabeledSpan(start=350, end=364, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                tail=LabeledSpan(start=512, end=527, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=106, end=113, label="background_claim", score=1.0),
                tail=LabeledSpan(start=136, end=148, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
                tail=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=95, end=96, label="data", score=1.0),
                tail=LabeledSpan(start=88, end=92, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=776, end=787, label="background_claim", score=1.0),
                tail=LabeledSpan(start=799, end=814, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=456, end=478, label="background_claim", score=1.0),
                tail=LabeledSpan(start=480, end=508, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=404, end=410, label="data", score=1.0),
                tail=LabeledSpan(start=411, end=428, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=454, end=461, label="background_claim", score=1.0),
                tail=LabeledSpan(start=469, end=475, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=212, end=213, label="data", score=1.0),
                tail=LabeledSpan(start=203, end=211, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=499, end=500, label="data", score=1.0),
                tail=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=114, end=133, label="background_claim", score=1.0),
                tail=LabeledSpan(start=136, end=148, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=182, end=183, label="data", score=1.0),
                tail=LabeledSpan(start=175, end=181, label="own_claim", score=1.0),
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
                head=LabeledSpan(start=502, end=503, label="data", score=1.0),
                tail=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=143, end=149, label="own_claim", score=1.0),
                tail=LabeledSpan(start=153, end=174, label="own_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=601, end=615, label="data", score=1.0),
                tail=LabeledSpan(start=618, end=631, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=443, end=453, label="background_claim", score=1.0),
                tail=LabeledSpan(start=411, end=428, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=489, end=490, label="data", score=1.0),
                tail=LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=489, end=490, label="data", score=1.0),
                tail=LabeledSpan(start=469, end=475, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=578, end=588, label="data", score=1.0),
                tail=LabeledSpan(start=589, end=606, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=70, end=78, label="data", score=1.0),
                tail=LabeledSpan(start=61, end=68, label="own_claim", score=1.0),
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
                head=LabeledSpan(start=737, end=738, label="data", score=1.0),
                tail=LabeledSpan(start=719, end=734, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=42, end=45, label="data", score=1.0),
                tail=LabeledSpan(start=30, end=59, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=572, end=584, label="own_claim", score=1.0),
                tail=LabeledSpan(start=558, end=570, label="own_claim", score=1.0),
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
                head=LabeledSpan(start=127, end=148, label="background_claim", score=1.0),
                tail=LabeledSpan(start=98, end=126, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=873, end=913, label="own_claim", score=1.0),
                tail=LabeledSpan(start=854, end=913, label="own_claim", score=1.0),
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
                head=LabeledSpan(start=230, end=239, label="background_claim", score=1.0),
                tail=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=826, end=829, label="background_claim", score=1.0),
                tail=LabeledSpan(start=815, end=824, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=251, end=260, label="background_claim", score=1.0),
                tail=LabeledSpan(start=262, end=271, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
                tail=LabeledSpan(start=212, end=219, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=207, end=209, label="data", score=1.0),
                tail=LabeledSpan(start=212, end=219, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=367, end=368, label="data", score=1.0),
                tail=LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=619, end=626, label="data", score=1.0),
                tail=LabeledSpan(start=627, end=634, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
        },
    }

    assert dict(fn) == {
        "labeled_spans": {
            LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
            LabeledSpan(start=485, end=488, label="data", score=1.0),
            LabeledSpan(start=274, end=291, label="background_claim", score=1.0),
            LabeledSpan(start=519, end=527, label="background_claim", score=1.0),
            LabeledSpan(start=528, end=532, label="background_claim", score=1.0),
            LabeledSpan(start=306, end=309, label="data", score=1.0),
            LabeledSpan(start=626, end=633, label="own_claim", score=1.0),
            LabeledSpan(start=618, end=634, label="background_claim", score=1.0),
            LabeledSpan(start=687, end=691, label="own_claim", score=1.0),
            LabeledSpan(start=736, end=739, label="data", score=1.0),
            LabeledSpan(start=330, end=337, label="background_claim", score=1.0),
            LabeledSpan(start=128, end=133, label="data", score=1.0),
            LabeledSpan(start=501, end=504, label="data", score=1.0),
            LabeledSpan(start=498, end=501, label="data", score=1.0),
            LabeledSpan(start=572, end=584, label="background_claim", score=1.0),
            LabeledSpan(start=136, end=148, label="own_claim", score=1.0),
            LabeledSpan(start=509, end=512, label="data", score=1.0),
            LabeledSpan(start=589, end=606, label="background_claim", score=1.0),
            LabeledSpan(start=719, end=731, label="background_claim", score=1.0),
            LabeledSpan(start=43, end=45, label="data", score=1.0),
            LabeledSpan(start=302, end=309, label="data", score=1.0),
            LabeledSpan(start=488, end=491, label="data", score=1.0),
            LabeledSpan(start=365, end=386, label="background_claim", score=1.0),
            LabeledSpan(start=387, end=390, label="data", score=1.0),
            LabeledSpan(start=323, end=326, label="data", score=1.0),
            LabeledSpan(start=518, end=521, label="data", score=1.0),
            LabeledSpan(start=203, end=210, label="background_claim", score=1.0),
            LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
            LabeledSpan(start=1007, end=1010, label="data", score=1.0),
            LabeledSpan(start=70, end=75, label="data", score=1.0),
            LabeledSpan(start=605, end=615, label="data", score=1.0),
            LabeledSpan(start=30, end=41, label="background_claim", score=1.0),
            LabeledSpan(start=310, end=341, label="background_claim", score=1.0),
            LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
            LabeledSpan(start=509, end=516, label="background_claim", score=1.0),
            LabeledSpan(start=122, end=127, label="data", score=1.0),
            LabeledSpan(start=366, end=369, label="data", score=1.0),
            LabeledSpan(start=522, end=526, label="data", score=1.0),
            LabeledSpan(start=34, end=52, label="background_claim", score=1.0),
            LabeledSpan(start=94, end=97, label="data", score=1.0),
            LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
            LabeledSpan(start=235, end=239, label="data", score=1.0),
            LabeledSpan(start=796, end=799, label="data", score=1.0),
            LabeledSpan(start=1012, end=1018, label="data", score=1.0),
            LabeledSpan(start=143, end=174, label="background_claim", score=1.0),
            LabeledSpan(start=76, end=78, label="data", score=1.0),
            LabeledSpan(start=800, end=814, label="background_claim", score=1.0),
            LabeledSpan(start=181, end=184, label="data", score=1.0),
            LabeledSpan(start=348, end=371, label="background_claim", score=1.0),
            LabeledSpan(start=212, end=225, label="background_claim", score=1.0),
            LabeledSpan(start=109, end=113, label="data", score=1.0),
            LabeledSpan(start=662, end=674, label="own_claim", score=1.0),
            LabeledSpan(start=642, end=660, label="own_claim", score=1.0),
            LabeledSpan(start=441, end=455, label="background_claim", score=1.0),
            LabeledSpan(start=230, end=234, label="data", score=1.0),
            LabeledSpan(start=454, end=475, label="background_claim", score=1.0),
            LabeledSpan(start=182, end=202, label="background_claim", score=1.0),
            LabeledSpan(start=704, end=728, label="own_claim", score=1.0),
        },
        "binary_relations": {
            BinaryRelation(
                head=LabeledSpan(start=427, end=439, label="background_claim", score=1.0),
                tail=LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=390, end=403, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=636, end=664, label="background_claim", score=1.0),
                tail=LabeledSpan(start=618, end=634, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=488, end=491, label="data", score=1.0),
                tail=LabeledSpan(start=454, end=475, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=292, end=298, label="background_claim", score=1.0),
                tail=LabeledSpan(start=274, end=291, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=365, end=386, label="background_claim", score=1.0),
                tail=LabeledSpan(start=427, end=439, label="background_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
                tail=LabeledSpan(start=182, end=202, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=605, end=615, label="data", score=1.0),
                tail=LabeledSpan(start=618, end=631, label="own_claim", score=1.0),
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
                head=LabeledSpan(start=343, end=347, label="data", score=1.0),
                tail=LabeledSpan(start=348, end=371, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=43, end=45, label="data", score=1.0),
                tail=LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=94, end=97, label="data", score=1.0),
                tail=LabeledSpan(start=97, end=126, label="background_claim", score=1.0),
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
                tail=LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
                tail=LabeledSpan(start=761, end=775, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=70, end=75, label="data", score=1.0),
                tail=LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=662, end=674, label="own_claim", score=1.0),
                tail=LabeledSpan(start=642, end=660, label="own_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=306, end=309, label="data", score=1.0),
                tail=LabeledSpan(start=310, end=322, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=626, end=633, label="own_claim", score=1.0),
                tail=LabeledSpan(start=687, end=691, label="own_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
                tail=LabeledSpan(start=480, end=508, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=323, end=326, label="data", score=1.0),
                tail=LabeledSpan(start=326, end=350, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
                tail=LabeledSpan(start=136, end=148, label="own_claim", score=1.0),
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
                head=LabeledSpan(start=366, end=369, label="data", score=1.0),
                tail=LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
                tail=LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=387, end=390, label="data", score=1.0),
                tail=LabeledSpan(start=390, end=403, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=210, end=211, label="data", score=1.0),
                tail=LabeledSpan(start=212, end=225, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                tail=LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=210, end=211, label="data", score=1.0),
                tail=LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=485, end=488, label="data", score=1.0),
                tail=LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=352, end=365, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
        },
    }

    assert dict(tp) == {
        "labeled_spans": {
            LabeledSpan(start=408, end=423, label="background_claim", score=1.0),
            LabeledSpan(start=310, end=322, label="background_claim", score=1.0),
            LabeledSpan(start=292, end=298, label="background_claim", score=1.0),
            LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
            LabeledSpan(start=302, end=312, label="background_claim", score=1.0),
            LabeledSpan(start=317, end=329, label="background_claim", score=1.0),
            LabeledSpan(start=88, end=92, label="background_claim", score=1.0),
            LabeledSpan(start=636, end=664, label="background_claim", score=1.0),
            LabeledSpan(start=352, end=365, label="background_claim", score=1.0),
            LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
            LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
            LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
            LabeledSpan(start=578, end=588, label="data", score=1.0),
            LabeledSpan(start=96, end=105, label="data", score=1.0),
            LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
            LabeledSpan(start=207, end=209, label="data", score=1.0),
            LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
            LabeledSpan(start=427, end=439, label="background_claim", score=1.0),
            LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
            LabeledSpan(start=97, end=126, label="background_claim", score=1.0),
            LabeledSpan(start=411, end=428, label="background_claim", score=1.0),
            LabeledSpan(start=210, end=211, label="data", score=1.0),
            LabeledSpan(start=343, end=347, label="data", score=1.0),
            LabeledSpan(start=53, end=73, label="background_claim", score=1.0),
            LabeledSpan(start=90, end=95, label="data", score=1.0),
            LabeledSpan(start=480, end=508, label="background_claim", score=1.0),
            LabeledSpan(start=342, end=349, label="background_claim", score=1.0),
            LabeledSpan(start=74, end=87, label="background_claim", score=1.0),
            LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
            LabeledSpan(start=682, end=691, label="own_claim", score=1.0),
            LabeledSpan(start=618, end=631, label="own_claim", score=1.0),
            LabeledSpan(start=761, end=775, label="background_claim", score=1.0),
            LabeledSpan(start=326, end=350, label="background_claim", score=1.0),
            LabeledSpan(start=390, end=403, label="background_claim", score=1.0),
            LabeledSpan(start=212, end=213, label="data", score=1.0),
        },
        "binary_relations": {
            BinaryRelation(
                head=LabeledSpan(start=326, end=350, label="background_claim", score=1.0),
                tail=LabeledSpan(start=352, end=365, label="background_claim", score=1.0),
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
                head=LabeledSpan(start=88, end=92, label="background_claim", score=1.0),
                tail=LabeledSpan(start=97, end=126, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=90, end=95, label="data", score=1.0),
                tail=LabeledSpan(start=74, end=87, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=96, end=105, label="data", score=1.0),
                tail=LabeledSpan(start=74, end=87, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
        },
    }


@pytest.mark.slow
def test_sciarg_predict_with_position_id_pattern(sciarg_batch, loaded_taskmodule):
    trained_model = SimplePointerNetworkModel.from_pretrained(MODEL_PATH_WITH_POSITION_ID_PATTERN)
    assert trained_model is not None

    torch.manual_seed(42)
    inputs, targets = sciarg_batch
    inputs_truncated = {k: v[:5] for k, v in inputs.items()}
    targets_truncated = {k: v[:5] for k, v in targets.items()}

    prediction = trained_model.predict(inputs_truncated, max_length=20)
    assert prediction is not None
    metric = loaded_taskmodule.configure_metric()
    metric.update(prediction, targets_truncated["tgt_tokens"])

    values = metric.compute()
    assert values == {
        "em": 0.0,
        "em_original": 0.4,
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
