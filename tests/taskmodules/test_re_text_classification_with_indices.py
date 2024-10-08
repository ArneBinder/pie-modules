import dataclasses
import logging
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, Union

import pytest
import torch
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, NaryRelation
from pytorch_ie.core import Annotation, AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TextDocument
from torch import tensor
from torchmetrics import Metric, MetricCollection

from pie_modules.taskmodules import RETextClassificationWithIndicesTaskModule
from pie_modules.taskmodules.re_text_classification_with_indices import (
    HEAD,
    TAIL,
    inner_span_distance,
    span_distance,
)
from pie_modules.utils import flatten_dict
from tests import _config_to_str
from tests.conftest import _TABULATE_AVAILABLE, TestDocument

CONFIGS = [
    {"add_type_to_marker": False, "append_markers": False},
    {"add_type_to_marker": True, "append_markers": False},
    {"add_type_to_marker": False, "append_markers": True},
    {"add_type_to_marker": True, "append_markers": True},
]
CONFIGS_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIGS_DICT.keys())
def cfg(request):
    return CONFIGS_DICT[request.param]


def test_taskmodule_with_deprecated_parameters(caplog):
    with caplog.at_level(logging.WARNING):
        tokenizer_name_or_path = "bert-base-cased"
        taskmodule = RETextClassificationWithIndicesTaskModule(
            tokenizer_name_or_path=tokenizer_name_or_path, label_to_id={"a": 0, "b": 1}
        )
        assert taskmodule.labels == ["a", "b"]
    # check the warning message
    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message
        == "The parameter label_to_id is deprecated and will be removed in a future version. Please use labels instead."
    )


@pytest.fixture(scope="module")
def unprepared_taskmodule(cfg):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations", tokenizer_name_or_path=tokenizer_name_or_path, **cfg
    )
    assert not taskmodule.is_from_pretrained

    return taskmodule


@pytest.fixture(scope="module")
def taskmodule(unprepared_taskmodule, documents):
    unprepared_taskmodule.prepare(documents)
    return unprepared_taskmodule


@pytest.fixture
def model_output():
    return {
        "labels": torch.tensor([1, 0, 2, 3, 1, 0, 0, 0]),
        "probabilities": torch.tensor(
            [
                # O, org:founded_by, per:employee_of, per:founder
                [0.1, 0.6, 0.1, 0.2],
                [0.5, 0.2, 0.2, 0.1],
                [0.1, 0.2, 0.6, 0.1],
                [0.1, 0.2, 0.2, 0.5],
                [0.2, 0.4, 0.3, 0.1],
                [0.5, 0.2, 0.2, 0.1],
                [0.6, 0.1, 0.2, 0.1],
                [0.5, 0.2, 0.2, 0.1],
            ]
        ),
    }


def test_prepared_taskmodule(taskmodule, documents):
    assert taskmodule.is_prepared

    assert taskmodule.entity_labels == ["ORG", "PER"]

    if taskmodule.append_markers:
        if taskmodule.add_type_to_marker:
            assert taskmodule.argument_markers == [
                "[/H:ORG]",
                "[/H:PER]",
                "[/H]",
                "[/T:ORG]",
                "[/T:PER]",
                "[/T]",
                "[H:ORG]",
                "[H:PER]",
                "[H=ORG]",
                "[H=PER]",
                "[H]",
                "[T:ORG]",
                "[T:PER]",
                "[T=ORG]",
                "[T=PER]",
                "[T]",
            ]
            assert taskmodule.argument_markers_to_id == {
                "[/H:ORG]": 28996,
                "[/H:PER]": 28997,
                "[/H]": 28998,
                "[/T:ORG]": 28999,
                "[/T:PER]": 29000,
                "[/T]": 29001,
                "[H:ORG]": 29002,
                "[H:PER]": 29003,
                "[H=ORG]": 29004,
                "[H=PER]": 29005,
                "[H]": 29006,
                "[T:ORG]": 29007,
                "[T:PER]": 29008,
                "[T=ORG]": 29009,
                "[T=PER]": 29010,
                "[T]": 29011,
            }

        else:
            assert taskmodule.argument_markers == [
                "[/H]",
                "[/T]",
                "[H=ORG]",
                "[H=PER]",
                "[H]",
                "[T=ORG]",
                "[T=PER]",
                "[T]",
            ]
            assert taskmodule.argument_markers_to_id == {
                "[/H]": 28996,
                "[/T]": 28997,
                "[H=ORG]": 28998,
                "[H=PER]": 28999,
                "[H]": 29000,
                "[T=ORG]": 29001,
                "[T=PER]": 29002,
                "[T]": 29003,
            }
    else:
        if taskmodule.add_type_to_marker:
            assert taskmodule.argument_markers == [
                "[/H:ORG]",
                "[/H:PER]",
                "[/H]",
                "[/T:ORG]",
                "[/T:PER]",
                "[/T]",
                "[H:ORG]",
                "[H:PER]",
                "[H]",
                "[T:ORG]",
                "[T:PER]",
                "[T]",
            ]
            assert taskmodule.argument_markers_to_id == {
                "[/H:ORG]": 28996,
                "[/H:PER]": 28997,
                "[/H]": 28998,
                "[/T:ORG]": 28999,
                "[/T:PER]": 29000,
                "[/T]": 29001,
                "[H:ORG]": 29002,
                "[H:PER]": 29003,
                "[H]": 29004,
                "[T:ORG]": 29005,
                "[T:PER]": 29006,
                "[T]": 29007,
            }
        else:
            assert taskmodule.argument_markers == ["[/H]", "[/T]", "[H]", "[T]"]
            assert taskmodule.argument_markers_to_id == {
                "[/H]": 28996,
                "[/T]": 28997,
                "[H]": 28998,
                "[T]": 28999,
            }

    assert taskmodule.label_to_id == {
        "org:founded_by": 1,
        "per:employee_of": 2,
        "per:founder": 3,
        "no_relation": 0,
    }
    assert taskmodule.id_to_label == {
        1: "org:founded_by",
        2: "per:employee_of",
        3: "per:founder",
        0: "no_relation",
    }


def test_config(taskmodule):
    config = taskmodule._config()
    assert config["taskmodule_type"] == "RETextClassificationWithIndicesTaskModule"
    assert taskmodule.PREPARED_ATTRIBUTES == ["labels", "entity_labels"]
    assert all(attribute in config for attribute in taskmodule.PREPARED_ATTRIBUTES)
    assert config["labels"] == ["org:founded_by", "per:employee_of", "per:founder"]
    assert config["entity_labels"] == ["ORG", "PER"]


@pytest.mark.parametrize("encode_target", [False, True])
def test_encode(taskmodule, documents, encode_target):
    task_encodings = taskmodule.encode(documents, encode_target=encode_target)

    assert len(task_encodings) == 7

    encoding = task_encodings[0]

    tokens = taskmodule.tokenizer.convert_ids_to_tokens(encoding.inputs["input_ids"])
    assert len(tokens) == len(encoding.inputs["input_ids"])

    if taskmodule.add_type_to_marker:
        assert tokens[:14] == [
            "[CLS]",
            "[H:PER]",
            "En",
            "##ti",
            "##ty",
            "A",
            "[/H:PER]",
            "works",
            "at",
            "[T:ORG]",
            "B",
            "[/T:ORG]",
            ".",
            "[SEP]",
        ]
    else:
        assert tokens[:14] == [
            "[CLS]",
            "[H]",
            "En",
            "##ti",
            "##ty",
            "A",
            "[/H]",
            "works",
            "at",
            "[T]",
            "B",
            "[/T]",
            ".",
            "[SEP]",
        ]
    if taskmodule.append_markers:
        assert len(tokens) == 14 + 4
        assert tokens[-4:] == ["[H=PER]", "[SEP]", "[T=ORG]", "[SEP]"]
    else:
        assert len(tokens) == 14

    if encode_target:
        assert encoding.targets == [2]
    else:
        assert not encoding.has_targets

        with pytest.raises(AssertionError, match=re.escape("task encoding has no target")):
            encoding.targets


@pytest.fixture(scope="module")
def batch(taskmodule, documents):
    documents = [documents[i] for i in [0, 1, 4]]
    task_encodings = taskmodule.encode(documents, encode_target=True)
    return taskmodule.collate(task_encodings[:2])


def test_collate(taskmodule, batch):
    inputs, targets = batch

    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert inputs["input_ids"].shape == inputs["attention_mask"].shape

    if taskmodule.append_markers:
        assert inputs["input_ids"].shape == (2, 25)
        if taskmodule.add_type_to_marker:
            torch.testing.assert_close(
                inputs.input_ids,
                torch.tensor(
                    [
                        [
                            101,
                            29003,
                            13832,
                            3121,
                            2340,
                            138,
                            28997,
                            1759,
                            1120,
                            29007,
                            139,
                            28999,
                            119,
                            102,
                            29005,
                            102,
                            29009,
                            102,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            101,
                            1752,
                            5650,
                            119,
                            29003,
                            13832,
                            3121,
                            2340,
                            144,
                            28997,
                            1759,
                            1120,
                            29007,
                            145,
                            28999,
                            119,
                            1262,
                            1771,
                            146,
                            119,
                            102,
                            29005,
                            102,
                            29009,
                            102,
                        ],
                    ]
                ),
            )
        else:
            torch.testing.assert_close(
                inputs.input_ids,
                torch.tensor(
                    [
                        [
                            101,
                            29000,
                            13832,
                            3121,
                            2340,
                            138,
                            28996,
                            1759,
                            1120,
                            29003,
                            139,
                            28997,
                            119,
                            102,
                            28999,
                            102,
                            29001,
                            102,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            101,
                            1752,
                            5650,
                            119,
                            29000,
                            13832,
                            3121,
                            2340,
                            144,
                            28996,
                            1759,
                            1120,
                            29003,
                            145,
                            28997,
                            119,
                            1262,
                            1771,
                            146,
                            119,
                            102,
                            28999,
                            102,
                            29001,
                            102,
                        ],
                    ]
                ),
            )
        torch.testing.assert_close(
            inputs.attention_mask,
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        )

    else:
        assert inputs["input_ids"].shape == (2, 21)

        if taskmodule.add_type_to_marker:
            torch.testing.assert_close(
                inputs.input_ids,
                torch.tensor(
                    [
                        [
                            101,
                            29003,
                            13832,
                            3121,
                            2340,
                            138,
                            28997,
                            1759,
                            1120,
                            29005,
                            139,
                            28999,
                            119,
                            102,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            101,
                            1752,
                            5650,
                            119,
                            29003,
                            13832,
                            3121,
                            2340,
                            144,
                            28997,
                            1759,
                            1120,
                            29005,
                            145,
                            28999,
                            119,
                            1262,
                            1771,
                            146,
                            119,
                            102,
                        ],
                    ]
                ),
            )
        else:
            torch.testing.assert_close(
                inputs.input_ids,
                torch.tensor(
                    [
                        [
                            101,
                            28998,
                            13832,
                            3121,
                            2340,
                            138,
                            28996,
                            1759,
                            1120,
                            28999,
                            139,
                            28997,
                            119,
                            102,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            101,
                            1752,
                            5650,
                            119,
                            28998,
                            13832,
                            3121,
                            2340,
                            144,
                            28996,
                            1759,
                            1120,
                            28999,
                            145,
                            28997,
                            119,
                            1262,
                            1771,
                            146,
                            119,
                            102,
                        ],
                    ]
                ),
            )
        torch.testing.assert_close(
            inputs.attention_mask,
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        )

    assert set(targets) == {"labels"}
    torch.testing.assert_close(targets["labels"], torch.tensor([2, 2]))


def test_unbatch_output(taskmodule, model_output):
    unbatched_outputs = taskmodule.unbatch_output(model_output)

    assert len(unbatched_outputs) == 8

    labels = [
        "org:founded_by",
        "no_relation",
        "per:employee_of",
        "per:founder",
        "org:founded_by",
        "no_relation",
        "no_relation",
        "no_relation",
    ]
    probabilities = [0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5]

    for output, label, probability in zip(unbatched_outputs, labels, probabilities):
        assert set(output.keys()) == {"labels", "probabilities"}
        assert output["labels"] == [label]
        assert output["probabilities"] == pytest.approx([probability])


@pytest.mark.parametrize("inplace", [False, True])
def test_decode(taskmodule, documents, model_output, inplace):
    # copy the documents, because the taskmodule may modify them
    documents = [documents[i].copy() for i in [0, 1, 4]]

    encodings = taskmodule.encode(documents, encode_target=False)
    unbatched_outputs = taskmodule.unbatch_output(model_output)
    decoded_documents = taskmodule.decode(
        task_encodings=encodings,
        task_outputs=unbatched_outputs,
        inplace=inplace,
    )

    assert len(decoded_documents) == len(documents)

    if inplace:
        assert {id(doc) for doc in decoded_documents} == {id(doc) for doc in documents}
    else:
        assert {id(doc) for doc in decoded_documents}.isdisjoint({id(doc) for doc in documents})

    expected_scores = [0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5]
    i = 0
    for document in decoded_documents:
        for relation_expected, relation_decoded in zip(
            document["entities"], document["entities"].predictions
        ):
            assert relation_expected.start == relation_decoded.start
            assert relation_expected.end == relation_decoded.end
            assert relation_expected.label == relation_decoded.label
            assert expected_scores[i] == pytest.approx(relation_decoded.score)
            i += 1

    if not inplace:
        for document in documents:
            assert not document["relations"].predictions


def test_encode_with_partition(documents):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        partition_annotation="sentences",
    )
    assert not taskmodule.is_from_pretrained
    taskmodule.prepare(documents)

    assert len(documents) == 7
    encodings = taskmodule.encode(documents)
    tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(encoding.inputs["input_ids"])
        for encoding in encodings
    ]
    assert len(encodings) == 5
    assert encodings[0].document != encodings[1].document
    assert encodings[1].document != encodings[2].document
    # the last document contains 3 valid relations
    assert encodings[2].document == encodings[3].document
    assert encodings[3].document == encodings[4].document
    assert tokens[0] == [
        "[CLS]",
        "[H]",
        "En",
        "##ti",
        "##ty",
        "A",
        "[/H]",
        "works",
        "at",
        "[T]",
        "B",
        "[/T]",
        ".",
        "[SEP]",
    ]
    assert tokens[1] == [
        "[CLS]",
        "[H]",
        "En",
        "##ti",
        "##ty",
        "G",
        "[/H]",
        "works",
        "at",
        "[T]",
        "H",
        "[/T]",
        ".",
        "[SEP]",
    ]
    assert tokens[2] == [
        "[CLS]",
        "[H]",
        "En",
        "##ti",
        "##ty",
        "M",
        "[/H]",
        "works",
        "at",
        "[T]",
        "N",
        "[/T]",
        ".",
        "[SEP]",
    ]
    assert tokens[3] == [
        "[CLS]",
        "And",
        "[H]",
        "it",
        "[/H]",
        "founded",
        "[T]",
        "O",
        "[/T]",
        "[SEP]",
    ]
    assert tokens[4] == [
        "[CLS]",
        "And",
        "[T]",
        "it",
        "[/T]",
        "founded",
        "[H]",
        "O",
        "[/H]",
        "[SEP]",
    ]


def test_encode_with_windowing(documents):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_window=12,
    )
    assert not taskmodule.is_from_pretrained
    taskmodule.prepare(documents)

    assert len(documents) == 7
    encodings = taskmodule.encode(documents)
    assert len(encodings) == 3
    for encoding in encodings:
        assert len(encoding.inputs["input_ids"]) <= taskmodule.max_window
    tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(encoding.inputs["input_ids"])
        for encoding in encodings
    ]
    assert tokens[0] == [
        "[CLS]",
        "at",
        "[T]",
        "H",
        "[/T]",
        ".",
        "And",
        "founded",
        "[H]",
        "I",
        "[/H]",
        "[SEP]",
    ]
    assert tokens[1] == [
        "[CLS]",
        ".",
        "And",
        "[H]",
        "it",
        "[/H]",
        "founded",
        "[T]",
        "O",
        "[/T]",
        ".",
        "[SEP]",
    ]
    assert tokens[2] == [
        "[CLS]",
        ".",
        "And",
        "[T]",
        "it",
        "[/T]",
        "founded",
        "[H]",
        "O",
        "[/H]",
        ".",
        "[SEP]",
    ]


def test_encode_with_allow_discontinuous_text(documents):
    tokenizer_name_or_path = "bert-base-cased"
    # tokenizer_name_or_path = "allenai/longformer-scico"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_window=12,
        allow_discontinuous_text=True,
    )

    assert not taskmodule.is_from_pretrained
    taskmodule.prepare(documents)

    assert len(documents) == 7
    encodings = taskmodule.encode(documents)
    assert len(encodings) == 3

    for encoding in encodings:
        assert len(encoding.inputs["input_ids"]) <= taskmodule.max_window
    tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(encoding.inputs["input_ids"])
        for encoding in encodings
    ]
    assert tokens == [
        ["[CLS]", "at", "[T]", "H", "[/T]", "[SEP]", "founded", "[H]", "I", "[/H]", "[SEP]"],
        ["[CLS]", "And", "[H]", "it", "[/H]", "founded", "[T]", "O", "[/T]", "[SEP]"],
        ["[CLS]", "And", "[T]", "it", "[/T]", "founded", "[H]", "O", "[/H]", "[SEP]"],
    ]


@pytest.fixture(scope="module")
def taskmodule_with_add_argument_indices(documents):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        add_argument_indices_to_input=True,
    )

    assert not taskmodule.is_from_pretrained
    taskmodule.prepare(documents)
    return taskmodule


@pytest.fixture(scope="module")
def encodings_with_argument_indices(taskmodule_with_add_argument_indices, documents):
    task_encodings = taskmodule_with_add_argument_indices.encode(documents, encode_target=True)
    assert len(task_encodings) == 7
    return task_encodings


def test_encode_with_add_argument_indices(encodings_with_argument_indices):
    encodings = encodings_with_argument_indices
    assert all(["pooler_start_indices" in encoding.inputs for encoding in encodings])
    assert all(["pooler_end_indices" in encoding.inputs for encoding in encodings])

    # just check the first and fourth encoding
    encoding = encodings[0]
    assert "pooler_start_indices" in encoding.inputs
    assert "pooler_end_indices" in encoding.inputs
    assert len(encoding.inputs["pooler_start_indices"]) == 2
    assert len(encoding.inputs["pooler_end_indices"]) == 2
    assert encoding.inputs["pooler_start_indices"] == [2, 10]
    assert encoding.inputs["pooler_end_indices"] == [6, 11]
    encoding = encodings[3]
    assert "pooler_start_indices" in encoding.inputs
    assert "pooler_end_indices" in encoding.inputs
    assert len(encoding.inputs["pooler_start_indices"]) == 2
    assert len(encoding.inputs["pooler_end_indices"]) == 2
    assert encoding.inputs["pooler_start_indices"] == [17, 11]
    assert encoding.inputs["pooler_end_indices"] == [18, 12]


@pytest.fixture(scope="module")
def batch_with_argument_indices(
    taskmodule_with_add_argument_indices, encodings_with_argument_indices
):
    # just take the first two encodings
    return taskmodule_with_add_argument_indices.collate(encodings_with_argument_indices[:2])


def test_collate_with_add_argument_indices(batch_with_argument_indices):
    inputs, targets = batch_with_argument_indices

    assert set(inputs) == {
        "input_ids",
        "attention_mask",
        "pooler_start_indices",
        "pooler_end_indices",
    }

    assert inputs["input_ids"].shape == inputs["attention_mask"].shape
    torch.testing.assert_close(
        inputs["input_ids"],
        torch.tensor(
            [
                [
                    101,
                    28998,
                    13832,
                    3121,
                    2340,
                    138,
                    28996,
                    1759,
                    1120,
                    28999,
                    139,
                    28997,
                    119,
                    102,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    144,
                    28996,
                    1759,
                    1120,
                    28999,
                    145,
                    28997,
                    119,
                    1262,
                    1771,
                    146,
                    119,
                    102,
                ],
            ]
        ),
    )
    torch.testing.assert_close(
        inputs["attention_mask"],
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ),
    )

    torch.testing.assert_close(inputs["pooler_start_indices"], torch.tensor([[2, 10], [5, 13]]))
    torch.testing.assert_close(inputs["pooler_end_indices"], torch.tensor([[6, 11], [9, 14]]))


def test_encode_input_multiple_relations_for_same_arguments(caplog):
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path="bert-base-cased",
    )
    document = TestDocument(text="A founded B.", id="multiple_relations_for_same_arguments")
    document.entities.append(LabeledSpan(start=0, end=1, label="PER"))
    document.entities.append(LabeledSpan(start=10, end=11, label="PER"))
    entities = document.entities
    assert str(entities[0]) == "A"
    assert str(entities[1]) == "B"
    document.relations.extend(
        [
            BinaryRelation(head=entities[0], tail=entities[1], label="per:founded_by"),
            BinaryRelation(head=entities[0], tail=entities[1], label="per:founder"),
        ]
    )
    taskmodule.prepare([document])
    encodings = taskmodule.encode_input(document)

    assert len(caplog.messages) == 1
    assert (
        caplog.messages[0]
        == "doc.id=multiple_relations_for_same_arguments: there are multiple relations with the same arguments "
        "(('head', LabeledSpan(start=0, end=1, label='PER', score=1.0)), "
        "('tail', LabeledSpan(start=10, end=11, label='PER', score=1.0))): previous label='per:founded_by' "
        "and current label='per:founder'. We only keep the first occurring relation which has the "
        "label='per:founded_by'."
    )

    assert len(encodings) == 1
    relation = encodings[0].metadata["candidate_annotation"]
    assert str(relation.head) == "A"
    assert str(relation.tail) == "B"
    assert relation.label == "per:founded_by"


def test_encode_input_argument_role_unknown(documents):
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path="bert-base-cased",
        # the tail argument is not in the role_to_marker
        argument_role_to_marker={HEAD: "H"},
    )
    taskmodule.prepare(documents)
    with pytest.raises(ValueError) as excinfo:
        taskmodule.encode_input(documents[1])
    assert (
        str(excinfo.value) == "role='tail' not in known roles=['head'] (did you initialise the "
        "taskmodule with the correct argument_role_to_marker dictionary?)"
    )


def test_encode_input_with_add_candidate_relations(documents):
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path="bert-base-cased",
        add_candidate_relations=True,
    )
    taskmodule.prepare(documents)
    documents_without_relations = []
    encodings = []
    # just take the first three documents
    for doc in documents[:3]:
        doc_without_relations = doc.copy()
        relations = list(doc_without_relations.relations)
        doc_without_relations.relations.clear()
        # re-add one relation to test if it is kept
        if len(relations) > 0:
            doc_without_relations.relations.append(relations[0])
        documents_without_relations.append(doc_without_relations)
        encodings.extend(taskmodule.encode(doc_without_relations))

    assert len(encodings) == 4
    relations = [encoding.metadata["candidate_annotation"] for encoding in encodings]
    texts = [encoding.document.text for encoding in encodings]
    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in relations]

    # There are no entities in the first document, so there are no created relation candidates

    # this relation was kept
    assert texts[0] == "Entity A works at B."
    assert relation_tuples[0] == ("Entity A", "per:employee_of", "B")

    # the following relations were added
    assert texts[1] == "Entity A works at B."
    assert relation_tuples[1] == ("B", "no_relation", "Entity A")
    assert texts[2] == "Entity C and D."
    assert relation_tuples[2] == ("Entity C", "no_relation", "D")
    assert texts[3] == "Entity C and D."
    assert relation_tuples[3] == ("D", "no_relation", "Entity C")


@pytest.fixture
def document_with_nary_relations():
    @dataclasses.dataclass
    class TestDocumentWithNaryRelations(TextBasedDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[NaryRelation] = annotation_field(target="entities")

    document = TestDocumentWithNaryRelations(
        text="Entity A works at B.", id="doc_with_nary_relations"
    )
    document.entities.append(LabeledSpan(start=0, end=8, label="PER"))
    document.entities.append(LabeledSpan(start=18, end=19, label="PER"))
    document.relations.append(
        NaryRelation(
            arguments=tuple(document.entities),
            roles=tuple(["head", "tail"]),
            label="per:employee_of",
        )
    )
    return document


def test_encode_input_with_add_candidate_relations_with_wrong_relation_type(
    document_with_nary_relations,
):
    doc = document_with_nary_relations

    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path="bert-base-cased",
        add_candidate_relations=True,
        argument_role_to_marker={HEAD: "H", "arg2": "T"},
    )
    taskmodule.prepare([doc])
    with pytest.raises(NotImplementedError) as excinfo:
        taskmodule.encode_input(doc)
    assert (
        str(excinfo.value)
        == "doc.id=doc_with_nary_relations: the taskmodule does not yet support adding relation candidates "
        "with argument roles other than 'head' and 'tail': ['arg2', 'head']"
    )


def test_encode_input_with_add_reversed_relations(documents):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        add_reversed_relations=True,
    )
    taskmodule.prepare(documents)
    encodings = []
    # just take the first three documents
    for doc in documents[:3]:
        encodings.extend(taskmodule.encode_input(doc))

    assert len(encodings) == 2
    texts = [encoding.document.text for encoding in encodings]
    relations = [encoding.metadata["candidate_annotation"] for encoding in encodings]
    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in relations]

    # There are no relations in the first and last document, so there are also no new reversed relations

    # this is the original relation
    assert texts[0] == "Entity A works at B."
    assert relation_tuples[0] == ("Entity A", "per:employee_of", "B")

    # this is the reversed relation
    assert texts[1] == "Entity A works at B."
    assert relation_tuples[1] == ("B", "per:employee_of_reversed", "Entity A")

    # test that an already reversed relation is not reversed again
    document = TestDocument(
        text="Entity A works at B.", id="doc_with_relation_with_reversed_suffix"
    )
    document.entities.extend(
        [LabeledSpan(start=0, end=8, label="PER"), LabeledSpan(start=18, end=19, label="PER")]
    )
    document.relations.append(
        BinaryRelation(
            head=document.entities[1],
            tail=document.entities[0],
            label=f"per:employee_of{taskmodule.reversed_relation_label_suffix}",
        )
    )
    with pytest.raises(ValueError) as excinfo:
        taskmodule.encode_input(document)
    assert str(excinfo.value) == (
        "doc.id=doc_with_relation_with_reversed_suffix: The relation has the label 'per:employee_of_reversed' "
        "which already ends with the reversed_relation_label_suffix='_reversed'. It looks like the relation is "
        "already reversed, which is not allowed."
    )


def test_prepare_with_add_reversed_relations_with_label_has_suffix():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        add_reversed_relations=True,
    )
    document = TestDocument(
        text="Entity A works at B.", id="doc_with_relation_with_reversed_suffix"
    )
    document.entities.extend(
        [LabeledSpan(start=0, end=8, label="PER"), LabeledSpan(start=18, end=19, label="PER")]
    )
    document.relations.append(
        BinaryRelation(
            head=document.entities[0],
            tail=document.entities[1],
            label=f"per:employee_of{taskmodule.reversed_relation_label_suffix}",
        )
    )

    with pytest.raises(ValueError) as excinfo:
        taskmodule.prepare([document])
    assert (
        str(excinfo.value)
        == "doc.id=doc_with_relation_with_reversed_suffix: the relation label 'per:employee_of_reversed' "
        "already ends with the reversed_relation_label_suffix '_reversed', this is not allowed because "
        "we would not know if we should strip the suffix and revert the arguments during inference or not"
    )


@pytest.mark.parametrize("reverse_symmetric_relations", [False, True])
def test_encode_input_with_add_reversed_relations_with_symmetric_relations(
    reverse_symmetric_relations, caplog
):
    document = TestDocument(
        text="Entity A is married with B, but likes C, who is married with D.",
        id="doc_with_symmetric_relation",
    )
    document.entities.extend(
        [
            LabeledSpan(start=0, end=8, label="PER"),
            LabeledSpan(start=25, end=26, label="PER"),
            LabeledSpan(start=38, end=39, label="PER"),
            LabeledSpan(start=61, end=62, label="PER"),
        ]
    )
    assert str(document.entities[0]) == "Entity A"
    assert str(document.entities[1]) == "B"
    assert str(document.entities[2]) == "C"
    assert str(document.entities[3]) == "D"
    document.relations.extend(
        [
            BinaryRelation(
                head=document.entities[0], tail=document.entities[1], label="per:is_married_with"
            ),
            BinaryRelation(
                head=document.entities[0], tail=document.entities[2], label="per:likes"
            ),
            BinaryRelation(
                head=document.entities[2], tail=document.entities[3], label="per:is_married_with"
            ),
            BinaryRelation(
                head=document.entities[3], tail=document.entities[2], label="per:is_married_with"
            ),
        ]
    )

    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        add_reversed_relations=True,
        symmetric_relations=["per:is_married_with"],
        reverse_symmetric_relations=reverse_symmetric_relations,
    )
    taskmodule.prepare([document])
    encodings = taskmodule.encode_input(document)
    relations = [encoding.metadata["candidate_annotation"] for encoding in encodings]
    relation_tuples = [
        (str(relation.head), relation.label, str(relation.tail)) for relation in relations
    ]
    if reverse_symmetric_relations:
        assert relation_tuples == [
            ("Entity A", "per:is_married_with", "B"),
            ("Entity A", "per:likes", "C"),
            ("C", "per:is_married_with", "D"),
            ("D", "per:is_married_with", "C"),
            ("B", "per:is_married_with", "Entity A"),
            ("C", "per:likes_reversed", "Entity A"),
        ]
        assert len(caplog.messages) == 2
        assert (
            caplog.messages[0]
            == "doc.id=doc_with_symmetric_relation: there is already a relation with reversed "
            "arguments=(('head', LabeledSpan(start=61, end=62, label='PER', score=1.0)), "
            "('tail', LabeledSpan(start=38, end=39, label='PER', score=1.0))) and label=per:is_married_with, "
            "so we do not add the reversed relation (with label per:is_married_with) for these arguments"
        )
        assert (
            caplog.messages[1]
            == "doc.id=doc_with_symmetric_relation: there is already a relation with reversed "
            "arguments=(('head', LabeledSpan(start=38, end=39, label='PER', score=1.0)), "
            "('tail', LabeledSpan(start=61, end=62, label='PER', score=1.0))) and label=per:is_married_with, "
            "so we do not add the reversed relation (with label per:is_married_with) for these arguments"
        )
    else:
        assert relation_tuples == [
            ("Entity A", "per:is_married_with", "B"),
            ("Entity A", "per:likes", "C"),
            ("C", "per:is_married_with", "D"),
            ("D", "per:is_married_with", "C"),
            ("C", "per:likes_reversed", "Entity A"),
        ]
        assert len(caplog.messages) == 0

    caplog.clear()
    document = TestDocument(
        text="Entity A is married with B.",
        id="doc_with_reversed_symmetric_relation",
    )
    document.entities.append(LabeledSpan(start=0, end=8, label="PER"))
    document.entities.append(LabeledSpan(start=25, end=26, label="PER"))
    document.relations.append(
        BinaryRelation(
            head=document.entities[1], tail=document.entities[0], label="per:is_married_with"
        )
    )
    encodings = taskmodule.encode_input(document)
    relations = [encoding.metadata["candidate_annotation"] for encoding in encodings]
    relation_tuples = [
        (str(relation.head), relation.label, str(relation.tail)) for relation in relations
    ]
    if reverse_symmetric_relations:
        assert len(relation_tuples) == 2
        assert relation_tuples[0] == ("B", "per:is_married_with", "Entity A")
        assert relation_tuples[1] == ("Entity A", "per:is_married_with", "B")
        assert len(caplog.messages) == 1
        assert (
            caplog.messages[0]
            == "doc.id=doc_with_reversed_symmetric_relation: The symmetric relation with label 'per:is_married_with' "
            "has arguments (('head', LabeledSpan(start=25, end=26, label='PER', score=1.0)), "
            "('tail', LabeledSpan(start=0, end=8, label='PER', score=1.0))) which are not sorted by their start "
            "and end positions. This may lead to problems during evaluation because we assume that the arguments "
            "of symmetric relations were sorted in the beginning and, thus, interpret relations where this is not "
            "the case as reversed. All reversed relations will get their arguments swapped during inference in "
            "the case of add_reversed_relations=True to remove duplicates. You may consider adding reversed "
            "versions of the *symmetric* relations on your own and then setting *reverse_symmetric_relations* "
            "to False."
        )
    else:
        assert len(relation_tuples) == 1
        assert relation_tuples[0] == ("B", "per:is_married_with", "Entity A")
        assert len(caplog.messages) == 0


def test_encode_input_with_add_reversed_relations_with_wrong_relation_type(
    document_with_nary_relations,
):
    doc = document_with_nary_relations
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path="bert-base-cased",
        add_reversed_relations=True,
        symmetric_relations=["per:employee_of"],
    )
    taskmodule.prepare([doc])
    with pytest.raises(NotImplementedError) as excinfo:
        taskmodule.encode_input(doc)
    assert (
        str(excinfo.value)
        == "doc.id=doc_with_nary_relations: the taskmodule does not yet support adding "
        "reversed relations for type: <class 'pytorch_ie.annotations.NaryRelation'>"
    )


def test_inner_span_distance_overlap():
    dist = inner_span_distance((0, 2), (1, 3))
    assert dist == -1


def test_span_distance_unknown_type():
    with pytest.raises(ValueError) as excinfo:
        span_distance((0, 1), (2, 3), "unknown")
    assert str(excinfo.value) == "unknown distance_type=unknown. use one of: inner"


def test_encode_input_with_max_argument_distance():
    document = TestDocument(
        text="Entity A works at B and C.", id="doc_with_three_entities_and_two_relations"
    )
    e0 = LabeledSpan(start=0, end=8, label="PER")
    e1 = LabeledSpan(start=18, end=19, label="PER")
    e2 = LabeledSpan(start=24, end=25, label="PER")
    document.entities.extend([e0, e1, e2])
    assert str(document.entities[0]) == "Entity A"
    assert str(document.entities[1]) == "B"
    assert str(document.entities[2]) == "C"
    document.relations.append(
        BinaryRelation(
            head=document.entities[0], tail=document.entities[1], label="per:employee_of"
        )
    )
    document.relations.append(
        BinaryRelation(
            head=document.entities[0], tail=document.entities[2], label="per:employee_of"
        )
    )
    dist_01 = span_distance((e0.start, e0.end), (e1.start, e1.end), "inner")
    dist_02 = span_distance((e0.start, e0.end), (e2.start, e2.end), "inner")
    assert dist_01 == 10
    assert dist_02 == 16

    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_argument_distance=10,
    )
    taskmodule.prepare([document])
    encodings = taskmodule.encode_input(document)

    # there are two relations, but only one is within the max_argument_distance
    assert len(encodings) == 1
    relation = encodings[0].metadata["candidate_annotation"]
    assert str(relation.head) == "Entity A"
    assert str(relation.tail) == "B"
    assert relation.label == "per:employee_of"


def test_encode_input_with_max_argument_distance_with_wrong_relation_type(
    document_with_nary_relations,
):
    doc = document_with_nary_relations
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path="bert-base-cased",
        max_argument_distance=10,
    )
    taskmodule.prepare([doc])
    with pytest.raises(NotImplementedError) as excinfo:
        encodings = taskmodule.encode_input(doc)
    assert (
        str(excinfo.value)
        == "doc.id=doc_with_nary_relations: the taskmodule does not yet support filtering "
        "relation candidates for type: <class 'pytorch_ie.annotations.NaryRelation'>"
    )


def test_encode_input_with_unknown_label(caplog):
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path="bert-base-cased",
        labels=["rel"],
        entity_labels=["a", "b"],
        collect_statistics=True,
    )
    taskmodule.post_prepare()

    doc = TestDocument(text="hello world", id="doc_with_unknown_label")
    doc.entities.append(LabeledSpan(start=0, end=5, label="a"))
    doc.entities.append(LabeledSpan(start=6, end=11, label="b"))
    doc.relations.append(
        BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="unknown")
    )

    task_encodings = taskmodule.encode_input(doc)
    assert len(task_encodings) == 0

    with caplog.at_level(logging.INFO):
        taskmodule.show_statistics()
    assert len(caplog.messages) == 1
    assert (
        caplog.messages[0] == "statistics:\n"
        "|                       |   unknown |\n"
        "|:----------------------|----------:|\n"
        "| available             |         1 |\n"
        "| skipped_unknown_label |         1 |"
    )


def test_encode_with_empty_partition_layer(documents):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        partition_annotation="sentences",
    )
    taskmodule.prepare(documents)
    documents_without_sentences = []
    # just take the first three documents
    for doc in documents[:3]:
        doc_without_sentences = doc.copy()
        doc_without_sentences.sentences.clear()
        documents_without_sentences.append(doc_without_sentences)

    encodings = taskmodule.encode(documents_without_sentences)
    # since there are no sentences, but we use partition_annotation="sentences",
    # there are no encodings
    assert len(encodings) == 0


def test_encode_nary_relatio():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        argument_role_to_marker={"r1": "R1", "r2": "R2", "r3": "R3"},
        # setting labels and entity_labels makes the taskmodule prepared
        labels=["rel"],
        entity_labels=["a", "b", "c"],
    )
    taskmodule._post_prepare()

    @dataclass
    class DocWithNaryRelation(TextDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[NaryRelation] = annotation_field(target="entities")

    doc = DocWithNaryRelation(text="hello my world")
    entity1 = LabeledSpan(start=0, end=5, label="a")
    entity2 = LabeledSpan(start=6, end=8, label="b")
    entity3 = LabeledSpan(start=9, end=14, label="c")
    doc.entities.extend([entity1, entity2, entity3])
    doc.relations.append(
        NaryRelation(
            arguments=tuple([entity1, entity2, entity3]),
            roles=tuple(["r1", "r2", "r3"]),
            label="rel",
        )
    )

    task_encodings = taskmodule.encode([doc])
    assert len(task_encodings) == 1
    encoding = task_encodings[0]
    assert encoding.document == doc
    assert encoding.document.text == "hello my world"
    rel = encoding.metadata["candidate_annotation"]
    assert str(rel.arguments[0]) == "hello"
    assert str(rel.arguments[1]) == "my"
    assert str(rel.arguments[2]) == "world"
    assert rel.label == "rel"


def test_encode_unknown_relation_type():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        # setting labels and entity_labels makes the taskmodule prepared
        labels=["has_wrong_type"],
        entity_labels=["a"],
    )
    taskmodule._post_prepare()

    @dataclass(frozen=True)
    class UnknownRelation(Annotation):
        arg: LabeledSpan
        label: str

    @dataclass
    class DocWithUnknownRelationType(TextDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[UnknownRelation] = annotation_field(target="entities")

    doc = DocWithUnknownRelationType(text="hello world")
    entity = LabeledSpan(start=0, end=1, label="a")
    doc.entities.append(entity)
    doc.relations.append(UnknownRelation(arg=entity, label="has_wrong_type"))

    with pytest.raises(NotImplementedError) as excinfo:
        taskmodule.encode([doc])
    assert str(excinfo.value).startswith(
        "the taskmodule does not yet support getting relation arguments for type: "
    ) and str(excinfo.value).endswith("<locals>.UnknownRelation'>")


def test_encode_with_unaligned_span(caplog):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        # setting v and entity_labels makes the taskmodule prepared
        labels=["rel"],
        entity_labels=["a"],
    )
    taskmodule._post_prepare()

    @dataclass
    class MyDocument(TextDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")

    doc = MyDocument(text="hello   space", id="doc1")
    entity1 = LabeledSpan(start=0, end=5, label="a")
    entity2 = LabeledSpan(start=7, end=13, label="a")
    entity3 = LabeledSpan(start=6, end=8, label="a")
    doc.entities.extend([entity1, entity2, entity3])
    # the start of entity2 is not aligned with a token, but this will get fixed
    assert str(entity2) == " space"
    doc.relations.append(BinaryRelation(head=entity1, tail=entity2, label="rel"))
    # entity3 can not get fixed because it contains only space
    assert str(entity3) == "  "
    doc.relations.append(BinaryRelation(head=entity1, tail=entity3, label="rel"))

    task_encodings = taskmodule.encode([doc])
    # the second relation is skipped because we can not get an aligned token span for it
    assert len(task_encodings) == 1
    task_encoding = task_encodings[0]
    tokens = taskmodule.tokenizer.convert_ids_to_tokens(task_encoding.inputs["input_ids"])
    assert tokens == ["[CLS]", "[H]", "hello", "[/H]", "[T]", "space", "[/T]", "[SEP]"]

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.messages[0]
        == "doc.id=doc1: Skipping invalid example, cannot get argument token slice for LabeledSpan(start=6, end=8, label='a', score=1.0): \"  \""
    )


def test_encode_with_log_first_n_examples(caplog):
    @dataclass
    class DocumentWithLabeledEntitiesAndRelations(TextDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")

    doc = DocumentWithLabeledEntitiesAndRelations(text="hello world", id="doc1")
    entity1 = LabeledSpan(start=0, end=5, label="a")
    entity2 = LabeledSpan(start=6, end=11, label="a")
    doc.entities.extend([entity1, entity2])
    doc.relations.append(BinaryRelation(head=entity1, tail=entity2, label="rel"))

    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
        log_first_n_examples=1,
    )
    taskmodule.prepare([doc])

    # we need to set the log level to INFO, otherwise the log messages are not captured
    with caplog.at_level(logging.INFO):
        task_encodings = taskmodule.encode([doc, doc], encode_target=True)

    # the second example is skipped because log_first_n_examples=1
    assert len(task_encodings) == 2
    assert len(caplog.records) == 5
    assert all([record.levelname == "INFO" for record in caplog.records])
    assert caplog.records[0].message == "*** Example ***"
    assert caplog.records[1].message == "doc id: doc1"
    assert caplog.records[2].message == "tokens: [CLS] [H] hello [/H] [T] world [/T] [SEP]"
    assert caplog.records[3].message == "input_ids: 101 28998 19082 28996 28999 1362 28997 102"
    assert caplog.records[4].message == "Expected label: ['rel'] (ids = [1])"


@pytest.mark.skipif(condition=not _TABULATE_AVAILABLE, reason="requires the 'tabulate' package")
def test_encode_with_collect_statistics(documents, caplog):
    taskmodule = RETextClassificationWithIndicesTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path="bert-base-cased",
        collect_statistics=True,
    )
    taskmodule.prepare(documents)
    # we need to set the log level to INFO, otherwise the log messages are not captured
    with caplog.at_level(logging.INFO):
        task_encodings = taskmodule.encode(documents)
    assert len(task_encodings) == 7

    assert len(caplog.messages) == 1
    assert caplog.messages[0] == (
        "statistics:\n"
        "|           |   org:founded_by |   per:employee_of |   per:founder |   all_relations |\n"
        "|:----------|-----------------:|------------------:|--------------:|----------------:|\n"
        "| available |                2 |                 3 |             2 |               7 |\n"
        "| used      |                2 |                 3 |             2 |               7 |\n"
        "| used %    |              100 |               100 |           100 |             100 |"
    )


def test_get_global_attention(taskmodule, batch, cfg):
    global_attention_mask = taskmodule._get_global_attention(input_ids=batch[0]["input_ids"])
    tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(token_ids)
        for token_ids in batch[0]["input_ids"].tolist()
    ]
    global_attention_tokens = [
        [tok for tok, m in zip(tkns, glob_attn_mask) if m]
        for tkns, glob_attn_mask in zip(tokens, global_attention_mask)
    ]
    pad_tok = taskmodule.tokenizer.pad_token
    not_global_attention_tokens = [
        [tok for tok, m in zip(tkns, glob_attn_mask) if not (m or tok == pad_tok)]
        for tkns, glob_attn_mask in zip(tokens, global_attention_mask)
    ]
    if not cfg.get("append_markers", False):
        torch.testing.assert_close(
            global_attention_mask,
            torch.tensor(
                [
                    [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                ]
            ),
        )
        assert not_global_attention_tokens == [
            ["En", "##ti", "##ty", "A", "works", "at", "B", ".", "[SEP]"],
            [
                "First",
                "sentence",
                ".",
                "En",
                "##ti",
                "##ty",
                "G",
                "works",
                "at",
                "H",
                ".",
                "And",
                "founded",
                "I",
                ".",
                "[SEP]",
            ],
        ]
    else:
        torch.testing.assert_close(
            global_attention_mask,
            torch.tensor(
                [
                    [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                ]
            ),
        )
        assert not_global_attention_tokens == [
            ["En", "##ti", "##ty", "A", "works", "at", "B", ".", "[SEP]", "[SEP]", "[SEP]"],
            [
                "First",
                "sentence",
                ".",
                "En",
                "##ti",
                "##ty",
                "G",
                "works",
                "at",
                "H",
                ".",
                "And",
                "founded",
                "I",
                ".",
                "[SEP]",
                "[SEP]",
                "[SEP]",
            ],
        ]

    if cfg == {"add_type_to_marker": False, "append_markers": False}:
        assert global_attention_tokens == [
            ["[CLS]", "[H]", "[/H]", "[T]", "[/T]"],
            ["[CLS]", "[H]", "[/H]", "[T]", "[/T]"],
        ]
    elif cfg == {"add_type_to_marker": True, "append_markers": False}:
        assert global_attention_tokens == [
            ["[CLS]", "[H:PER]", "[/H:PER]", "[T:ORG]", "[/T:ORG]"],
            ["[CLS]", "[H:PER]", "[/H:PER]", "[T:ORG]", "[/T:ORG]"],
        ]
    elif cfg == {"add_type_to_marker": False, "append_markers": True}:
        assert global_attention_tokens == [
            ["[CLS]", "[H]", "[/H]", "[T]", "[/T]", "[H=PER]", "[T=ORG]"],
            ["[CLS]", "[H]", "[/H]", "[T]", "[/T]", "[H=PER]", "[T=ORG]"],
        ]
    elif cfg == {"add_type_to_marker": True, "append_markers": True}:
        assert global_attention_tokens == [
            ["[CLS]", "[H:PER]", "[/H:PER]", "[T:ORG]", "[/T:ORG]", "[H=PER]", "[T=ORG]"],
            ["[CLS]", "[H:PER]", "[/H:PER]", "[T:ORG]", "[/T:ORG]", "[H=PER]", "[T=ORG]"],
        ]
    else:
        raise ValueError(f"unexpected config: {cfg}")


def get_metric_state(metric_or_collection: Union[Metric, MetricCollection]) -> Dict[str, Any]:
    if isinstance(metric_or_collection, Metric):
        return {k: v.tolist() for k, v in flatten_dict(metric_or_collection.metric_state).items()}
    elif isinstance(metric_or_collection, MetricCollection):
        return flatten_dict({k: get_metric_state(v) for k, v in metric_or_collection.items()})
    else:
        raise ValueError(f"unsupported type: {type(metric_or_collection)}")


def test_configure_model_metric(documents, taskmodule):
    task_encodings = taskmodule.encode(documents, encode_target=True)
    batch = taskmodule.collate(task_encodings)

    metric = taskmodule.configure_model_metric(stage="train")
    assert isinstance(metric, (Metric, MetricCollection))
    state = get_metric_state(metric)
    assert state == {
        "micro_without_tn/f1/tp": [0],
        "micro_without_tn/f1/fp": [0],
        "micro_without_tn/f1/tn": [0],
        "micro_without_tn/f1/fn": [0],
        "with_tn/f1_per_label/tp": [0, 0, 0, 0],
        "with_tn/f1_per_label/fp": [0, 0, 0, 0],
        "with_tn/f1_per_label/tn": [0, 0, 0, 0],
        "with_tn/f1_per_label/fn": [0, 0, 0, 0],
        "with_tn/macro/f1/tp": [0, 0, 0, 0],
        "with_tn/macro/f1/fp": [0, 0, 0, 0],
        "with_tn/macro/f1/tn": [0, 0, 0, 0],
        "with_tn/macro/f1/fn": [0, 0, 0, 0],
        "with_tn/micro/f1/tp": [0],
        "with_tn/micro/f1/fp": [0],
        "with_tn/micro/f1/tn": [0],
        "with_tn/micro/f1/fn": [0],
    }
    assert metric.compute() == {
        "no_relation/f1": tensor(0.0),
        "org:founded_by/f1": tensor(0.0),
        "per:employee_of/f1": tensor(0.0),
        "per:founder/f1": tensor(0.0),
        "macro/f1": tensor(0.0),
        "micro/f1": tensor(0.0),
        "micro_without_tn/f1": tensor(0.0),
    }

    targets = batch[1]
    metric.update(targets, targets)
    state = get_metric_state(metric)
    assert state == {
        "micro_without_tn/f1/tp": [7],
        "micro_without_tn/f1/fp": [0],
        "micro_without_tn/f1/tn": [21],
        "micro_without_tn/f1/fn": [0],
        "with_tn/f1_per_label/tp": [0, 2, 3, 2],
        "with_tn/f1_per_label/fp": [0, 0, 0, 0],
        "with_tn/f1_per_label/tn": [7, 5, 4, 5],
        "with_tn/f1_per_label/fn": [0, 0, 0, 0],
        "with_tn/macro/f1/tp": [0, 2, 3, 2],
        "with_tn/macro/f1/fp": [0, 0, 0, 0],
        "with_tn/macro/f1/tn": [7, 5, 4, 5],
        "with_tn/macro/f1/fn": [0, 0, 0, 0],
        "with_tn/micro/f1/tp": [7],
        "with_tn/micro/f1/fp": [0],
        "with_tn/micro/f1/tn": [21],
        "with_tn/micro/f1/fn": [0],
    }
    assert metric.compute() == {
        "no_relation/f1": tensor(0.0),
        "org:founded_by/f1": tensor(1.0),
        "per:employee_of/f1": tensor(1.0),
        "per:founder/f1": tensor(1.0),
        "macro/f1": tensor(1.0),
        "micro/f1": tensor(1.0),
        "micro_without_tn/f1": tensor(1.0),
    }

    metric.reset()
    modified_targets = {"labels": torch.tensor([2, 2, 3, 1, 2, 0, 1])}
    # three positive matches and one true negative
    random_predictions = {"labels": torch.tensor([1, 1, 3, 1, 2, 0, 0])}
    metric.update(random_predictions, modified_targets)
    state = get_metric_state(metric)
    assert state == {
        "micro_without_tn/f1/tp": [3],
        "micro_without_tn/f1/fp": [3],
        "micro_without_tn/f1/tn": [15],
        "micro_without_tn/f1/fn": [3],
        "with_tn/f1_per_label/tp": [1, 1, 1, 1],
        "with_tn/f1_per_label/fp": [1, 2, 0, 0],
        "with_tn/f1_per_label/tn": [5, 3, 4, 6],
        "with_tn/f1_per_label/fn": [0, 1, 2, 0],
        "with_tn/macro/f1/tp": [1, 1, 1, 1],
        "with_tn/macro/f1/fp": [1, 2, 0, 0],
        "with_tn/macro/f1/tn": [5, 3, 4, 6],
        "with_tn/macro/f1/fn": [0, 1, 2, 0],
        "with_tn/micro/f1/tp": [4],
        "with_tn/micro/f1/fp": [3],
        "with_tn/micro/f1/tn": [18],
        "with_tn/micro/f1/fn": [3],
    }
    # created with torch.set_printoptions(precision=6)
    torch.testing.assert_close(
        metric.compute(),
        {
            "no_relation/f1": tensor(0.666667),
            "org:founded_by/f1": tensor(0.400000),
            "per:employee_of/f1": tensor(0.500000),
            "per:founder/f1": tensor(1.0),
            "macro/f1": tensor(0.641667),
            "micro/f1": tensor(0.571429),
            "micro_without_tn/f1": tensor(0.500000),
        },
    )

    # ensure that the metric can be pickled
    pickle.dumps(metric)
