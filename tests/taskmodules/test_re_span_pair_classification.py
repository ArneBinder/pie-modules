import dataclasses
import logging
import pickle
import re
from dataclasses import dataclass

import pytest
import torch
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, NaryRelation
from pytorch_ie.core import Annotation, AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TextDocument
from torch import tensor
from torchmetrics import Metric, MetricCollection

from pie_modules.taskmodules import RETextClassificationWithIndicesTaskModule
from pie_modules.taskmodules.re_span_pair_classification import (
    RESpanPairClassificationTaskModule,
)
from pie_modules.taskmodules.re_text_classification_with_indices import (
    HEAD,
    TAIL,
    inner_span_distance,
    span_distance,
)
from pie_modules.utils import flatten_dict
from tests import _config_to_str
from tests.conftest import _TABULATE_AVAILABLE, TestDocument

CONFIGS = [{}]
CONFIGS_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIGS_DICT.keys())
def cfg(request):
    return CONFIGS_DICT[request.param]


@pytest.fixture(scope="module")
def unprepared_taskmodule(cfg):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RESpanPairClassificationTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        log_first_n_examples=10,
        **cfg
    )
    assert not taskmodule.is_from_pretrained

    return taskmodule


@pytest.fixture(scope="module")
def taskmodule(unprepared_taskmodule, documents) -> RESpanPairClassificationTaskModule:
    unprepared_taskmodule.prepare(documents)
    return unprepared_taskmodule


def test_taskmodule(taskmodule: RESpanPairClassificationTaskModule, documents):
    assert taskmodule.is_prepared

    assert taskmodule.relation_annotation == "relations"
    assert taskmodule.labels == ["org:founded_by", "per:employee_of", "per:founder"]
    assert taskmodule.entity_labels == ["ORG", "PER"]
    assert taskmodule.label_to_id == {
        "org:founded_by": 1,
        "per:employee_of": 2,
        "per:founder": 3,
        "no_relation": 0,
    }
    assert taskmodule.argument_markers == [
        "[/SPAN:ORG]",
        "[/SPAN:PER]",
        "[SPAN:ORG]",
        "[SPAN:PER]",
    ]
    assert taskmodule.tokenizer.additional_special_tokens == [
        "[SPAN:PER]",
        "[/SPAN:ORG]",
        "[/SPAN:PER]",
        "[SPAN:ORG]",
    ]
    assert taskmodule.tokenizer.additional_special_tokens_ids == [28996, 28997, 28998, 28999]


@pytest.fixture(scope="module")
def document(documents):
    result = documents[4]
    assert (
        result.metadata["description"]
        == "sentences with multiple relation annotations and cross-sentence relation"
    )
    return result


@pytest.fixture(scope="module")
def task_encodings(taskmodule, document):
    result = taskmodule.encode(document, encode_target=True)
    return result


def test_encode_input(task_encodings, document, taskmodule):
    assert task_encodings is not None
    assert len(task_encodings) == 1
    inputs = task_encodings[0].inputs
    tokens = taskmodule.tokenizer.convert_ids_to_tokens(inputs["input_ids"])
    assert tokens == [
        "[CLS]",
        "First",
        "sentence",
        ".",
        "[SPAN:PER]",
        "En",
        "##ti",
        "##ty",
        "G",
        "[/SPAN:PER]",
        "works",
        "at",
        "[SPAN:ORG]",
        "H",
        "[/SPAN:ORG]",
        ".",
        "And",
        "founded",
        "[SPAN:ORG]",
        "I",
        "[/SPAN:ORG]",
        ".",
        "[SEP]",
    ]
    span_tokens = [
        tokens[start:end]
        for start, end in zip(inputs["span_start_indices"], inputs["span_end_indices"])
    ]
    assert span_tokens == [
        ["[SPAN:PER]", "En", "##ti", "##ty", "G", "[/SPAN:PER]"],
        ["[SPAN:ORG]", "H", "[/SPAN:ORG]"],
        ["[SPAN:ORG]", "I", "[/SPAN:ORG]"],
    ]
    tuple_spans = [[span_tokens[idx] for idx in indices] for indices in inputs["tuple_indices"]]
    assert tuple_spans == [
        [
            ["[SPAN:PER]", "En", "##ti", "##ty", "G", "[/SPAN:PER]"],
            ["[SPAN:ORG]", "H", "[/SPAN:ORG]"],
        ],
        [
            ["[SPAN:PER]", "En", "##ti", "##ty", "G", "[/SPAN:PER]"],
            ["[SPAN:ORG]", "I", "[/SPAN:ORG]"],
        ],
        [["[SPAN:ORG]", "I", "[/SPAN:ORG]"], ["[SPAN:ORG]", "H", "[/SPAN:ORG]"]],
    ]


def test_encode_target(taskmodule, task_encodings):
    assert len(task_encodings) == 1
    targets = task_encodings[0].targets
    labels = [taskmodule.id_to_label[label] for label in targets["labels"].tolist()]
    assert labels == ["per:employee_of", "per:founder", "org:founded_by"]


def test_maybe_log_example(taskmodule, task_encodings, caplog):
    caplog.clear()
    with caplog.at_level(logging.INFO):
        taskmodule._maybe_log_example(task_encodings[0], target=task_encodings[0].targets)
    assert caplog.messages == [
        "*** Example ***",
        "doc id: train_doc5",
        "tokens: [CLS] First sentence . [SPAN:PER] En ##ti ##ty G [/SPAN:PER] works at [SPAN:ORG] H [/SPAN:ORG] . And founded [SPAN:ORG] I [/SPAN:ORG] . [SEP]",
        "input_ids: 101 1752 5650 119 28996 13832 3121 2340 144 28998 1759 1120 28999 145 28997 119 1262 1771 28999 146 28997 119 102",
        "relation 0: per:employee_of",
        "\targ 0: [SPAN:PER] En ##ti ##ty G [/SPAN:PER]",
        "\targ 1: [SPAN:ORG] H [/SPAN:ORG]",
        "relation 1: per:founder",
        "\targ 0: [SPAN:PER] En ##ti ##ty G [/SPAN:PER]",
        "\targ 1: [SPAN:ORG] I [/SPAN:ORG]",
        "relation 2: org:founded_by",
        "\targ 0: [SPAN:ORG] I [/SPAN:ORG]",
        "\targ 1: [SPAN:ORG] H [/SPAN:ORG]",
    ]


def test_collate(taskmodule, task_encodings):
    result = taskmodule.collate(task_encodings)
    assert result is not None
    inputs, targets = result
    assert set(inputs) == {
        "input_ids",
        "attention_mask",
        "span_start_indices",
        "span_end_indices",
        "tuple_indices",
    }
    torch.testing.assert_close(
        inputs["input_ids"],
        tensor(
            [
                [
                    101,
                    1752,
                    5650,
                    119,
                    28996,
                    13832,
                    3121,
                    2340,
                    144,
                    28998,
                    1759,
                    1120,
                    28999,
                    145,
                    28997,
                    119,
                    1262,
                    1771,
                    28999,
                    146,
                    28997,
                    119,
                    102,
                ]
            ]
        ),
    )
    torch.testing.assert_close(inputs["attention_mask"], torch.ones_like(inputs["input_ids"]))
    torch.testing.assert_close(inputs["span_start_indices"], tensor([[4, 12, 18]]))
    torch.testing.assert_close(inputs["span_end_indices"], tensor([[10, 15, 21]]))
    torch.testing.assert_close(inputs["tuple_indices"], tensor([[[0, 1], [0, 2], [2, 1]]]))
    assert set(targets) == {"labels"}
    torch.testing.assert_close(targets["labels"], tensor([[2, 3, 1]]))


@pytest.fixture
def model_output():
    return {
        "labels": torch.tensor([[2, 3, 1]]),
        "probabilities": torch.tensor(
            [
                [
                    # no_relation, org:founded_by, per:employee_of, per:founder
                    [0.1, 0.2, 0.6, 0.1],
                    [0.1, 0.2, 0.2, 0.5],
                    [0.2, 0.5, 0.2, 0.1],
                ]
            ]
        ),
    }


@pytest.fixture
def unbatched_model_outputs(taskmodule, model_output):
    return taskmodule.unbatch_output(model_output)


def test_unbatch_outputs(taskmodule, unbatched_model_outputs):
    assert len(unbatched_model_outputs) == 1
    result = unbatched_model_outputs[0]
    assert set(result) == {"labels", "probabilities"}
    assert result["labels"] == ["per:employee_of", "per:founder", "org:founded_by"]
    assert result["probabilities"] == [0.6000000238418579, 0.5, 0.5]


def test_create_annotations_from_output(
    taskmodule, unbatched_model_outputs, task_encodings, document
):
    result = list(
        taskmodule.create_annotations_from_output(
            task_encoding=task_encodings[0], task_output=unbatched_model_outputs[0]
        )
    )
    scores = [0.6000000238418579, 0.5, 0.5]
    for i, ((layer_name, predicted_relation), original_relation) in enumerate(
        zip(result, document.relations)
    ):
        assert layer_name == taskmodule.relation_annotation
        assert predicted_relation == original_relation.copy()
        assert predicted_relation.score == scores[i]
