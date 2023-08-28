import logging
import re
from dataclasses import dataclass

import numpy
import pytest
import torch
from pytorch_ie.annotations import BinaryRelation, Label, LabeledSpan, NaryRelation
from pytorch_ie.core import Annotation, AnnotationList, annotation_field
from pytorch_ie.documents import TextDocument

from pie_models.taskmodules import RETextClassificationWithIndicesTaskModule
from pie_models.taskmodules.re_text_classification_with_indices import HEAD, TAIL
from tests import _config_to_str

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


@pytest.fixture(scope="module")
def unprepared_taskmodule(cfg):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path, **cfg
    )
    assert not taskmodule.is_from_pretrained

    return taskmodule


@pytest.fixture(scope="module")
def documents(dataset):
    return dataset["train"]


@pytest.fixture(scope="module")
def taskmodule(unprepared_taskmodule, documents):
    unprepared_taskmodule.prepare(documents)
    return unprepared_taskmodule


@pytest.fixture
def model_output():
    return {
        "logits": torch.from_numpy(
            numpy.log(
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
            )
        ),
    }


def test_prepared_taskmodule(taskmodule, documents):
    assert taskmodule.is_prepared

    assert taskmodule.entity_labels == ["ORG", "PER"]
    assert taskmodule.sep_token_id

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
    assert "label_to_id" in config
    assert config["label_to_id"] == {
        "org:founded_by": 1,
        "per:employee_of": 2,
        "per:founder": 3,
        "no_relation": 0,
    }

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


@pytest.mark.parametrize("encode_target", [False, True])
def test_collate(taskmodule, documents, encode_target):
    documents = [documents[i] for i in [0, 1, 4]]

    encodings = taskmodule.encode(documents, encode_target=encode_target)

    assert len(encodings) == 4
    if encode_target:
        assert all([encoding.has_targets for encoding in encodings])
    else:
        assert not any([encoding.has_targets for encoding in encodings])

    batch_encoding = taskmodule.collate(encodings[:2])
    inputs, targets = batch_encoding

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

    if encode_target:
        torch.testing.assert_close(targets, torch.tensor([2, 2]))
    else:
        assert targets is None


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
    documents = [documents[i] for i in [0, 1, 4]]

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
        tokenizer_name_or_path=tokenizer_name_or_path,
        partition_annotation="sentences",
    )
    assert not taskmodule.is_from_pretrained
    taskmodule.prepare(documents)

    assert len(documents) == 8
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
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_window=12,
    )
    assert not taskmodule.is_from_pretrained
    taskmodule.prepare(documents)

    assert len(documents) == 8
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


@pytest.mark.parametrize("add_argument_indices_to_input", [False, True])
def test_encode_with_add_argument_indices(documents, add_argument_indices_to_input):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        add_argument_indices_to_input=add_argument_indices_to_input,
    )

    assert not taskmodule.is_from_pretrained
    taskmodule.prepare(documents)
    task_encodings = taskmodule.encode(documents)
    assert len(task_encodings) == 7

    encoding = task_encodings[0]
    if taskmodule.add_argument_indices_to_input:
        assert "pooler_start_indices" in encoding.inputs
        assert "pooler_end_indices" in encoding.inputs
        assert len(encoding.inputs["pooler_start_indices"]) == 2
        assert len(encoding.inputs["pooler_end_indices"]) == 2
        assert encoding.inputs["pooler_start_indices"] == [2, 10]
        assert encoding.inputs["pooler_end_indices"] == [6, 11]

    else:
        assert "pooler_start_indices" not in encoding.inputs
        assert "pooler_end_indices" not in encoding.inputs

    encoding = task_encodings[3]
    if taskmodule.add_argument_indices_to_input:
        assert "pooler_start_indices" in encoding.inputs
        assert "pooler_end_indices" in encoding.inputs
        assert len(encoding.inputs["pooler_start_indices"]) == 2
        assert len(encoding.inputs["pooler_end_indices"]) == 2
        assert encoding.inputs["pooler_start_indices"] == [17, 11]
        assert encoding.inputs["pooler_end_indices"] == [18, 12]

    else:
        assert "pooler_start_indices" not in encoding.inputs
        assert "pooler_end_indices" not in encoding.inputs


@pytest.mark.parametrize("add_argument_indices_to_input", [False, True])
def test_collate_with_add_argument_indices(documents, add_argument_indices_to_input):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        add_argument_indices_to_input=add_argument_indices_to_input,
    )
    taskmodule.prepare(documents)
    encodings = taskmodule.encode(documents)
    if add_argument_indices_to_input:
        assert all(["pooler_start_indices" in encoding.inputs for encoding in encodings])
        assert all(["pooler_end_indices" in encoding.inputs for encoding in encodings])
    else:
        assert not any(["pooler_start_indices" in encoding.inputs for encoding in encodings])
        assert not any(["pooler_end_indices" in encoding.inputs for encoding in encodings])

    batch_encoding = taskmodule.collate(encodings[:2])
    inputs, targets = batch_encoding

    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert inputs["input_ids"].shape == inputs["attention_mask"].shape
    if add_argument_indices_to_input:
        assert "pooler_start_indices" in inputs
        assert "pooler_end_indices" in inputs

        torch.testing.assert_close(
            inputs["pooler_start_indices"], torch.tensor([[2, 10], [5, 13]])
        )
        torch.testing.assert_close(inputs["pooler_end_indices"], torch.tensor([[6, 11], [9, 14]]))

    else:
        assert "pooler_start_indices" not in inputs
        assert "pooler_end_indices" not in inputs


def test_relation_argument_role_unknown(documents):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        # the tail argument is not in the role_to_marker
        argument_role_to_marker={HEAD: "H"},
    )
    taskmodule.prepare(documents)
    with pytest.raises(ValueError) as excinfo:
        task_encodings = taskmodule.encode(documents)
    assert (
        str(excinfo.value)
        == "role=tail not in role_to_marker={'head': 'H'} (did you initialise the taskmodule with "
        "the correct argument_role_to_marker dictionary?)"
    )


def test_encode_with_create_relation_candidates(documents):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        create_relation_candidates=True,
    )
    taskmodule.prepare(documents)
    documents_without_relations = []
    # just take the first three documents
    for doc in documents[:3]:
        doc_without_relations = doc.copy()
        doc_without_relations.relations.clear()
        documents_without_relations.append(doc_without_relations)
    encodings = taskmodule.encode(documents_without_relations)
    assert len(encodings) == 4

    # There are no entities in the first document, so there are no created relation candidates

    encoding = encodings[0]
    assert encoding.document == documents_without_relations[1]
    assert encoding.document.text == "Entity A works at B."
    relation = encoding.metadata["candidate_annotation"]
    assert str(relation.head) == "Entity A"
    assert str(relation.tail) == "B"
    assert relation.label == "no_relation"

    encoding = encodings[1]
    assert encoding.document == documents_without_relations[1]
    assert encoding.document.text == "Entity A works at B."
    relation = encoding.metadata["candidate_annotation"]
    assert str(relation.head) == "B"
    assert str(relation.tail) == "Entity A"
    assert relation.label == "no_relation"

    encoding = encodings[2]
    assert encoding.document == documents_without_relations[2]
    assert encoding.document.text == "Entity C and D."
    relation = encoding.metadata["candidate_annotation"]
    assert str(relation.head) == "Entity C"
    assert str(relation.tail) == "D"
    assert relation.label == "no_relation"

    encoding = encodings[3]
    assert encoding.document == documents_without_relations[2]
    assert encoding.document.text == "Entity C and D."
    relation = encoding.metadata["candidate_annotation"]
    assert str(relation.head) == "D"
    assert str(relation.tail) == "Entity C"
    assert relation.label == "no_relation"


def test_encode_with_empty_partition_layer(documents):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
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


def test_encode_binary_relation_with_wrong_argument_type():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        # setting label_to_id and entity_labels makes the taskmodule prepared
        label_to_id={"has_wrong_arguments": 1},
        entity_labels=["a", "b"],
    )
    taskmodule._post_prepare()

    @dataclass
    class DocWithBinaryRelationAndWrongArgumentType(TextDocument):
        entities: AnnotationList[Label] = annotation_field()
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")

    doc = DocWithBinaryRelationAndWrongArgumentType(text="hello world")
    label_a = Label(label="a")
    label_b = Label(label="b")
    doc.entities.extend([label_a, label_b])
    doc.relations.append(BinaryRelation(head=label_a, tail=label_b, label="has_wrong_arguments"))

    with pytest.raises(ValueError) as excinfo:
        taskmodule.encode([doc])
    assert (
        str(excinfo.value)
        == "the taskmodule expects the relation arguments to be of type LabeledSpan, but got "
        "<class 'pytorch_ie.annotations.Label'> and <class 'pytorch_ie.annotations.Label'>"
    )


def test_encode_nary_relatio():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        argument_role_to_marker={"r1": "R1", "r2": "R2", "r3": "R3"},
        # setting label_to_id and entity_labels makes the taskmodule prepared
        label_to_id={"rel": 1},
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


def test_encode_nary_relation_with_wrong_argument_type():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        # setting label_to_id and entity_labels makes the taskmodule prepared
        label_to_id={"has_wrong_arguments": 1},
        entity_labels=["a", "b", "c"],
    )
    taskmodule._post_prepare()

    @dataclass
    class DocWithNaryRelationAndWrongArgumentType(TextDocument):
        entities: AnnotationList[Label] = annotation_field()
        relations: AnnotationList[NaryRelation] = annotation_field(target="entities")

    doc = DocWithNaryRelationAndWrongArgumentType(text="hello world")
    label_a = Label(label="a")
    label_b = Label(label="b")
    label_c = Label(label="c")
    doc.entities.extend([label_a, label_b, label_c])
    doc.relations.append(
        NaryRelation(
            arguments=tuple([label_a, label_b, label_c]),
            roles=tuple(["a", "b", "c"]),
            label="has_wrong_arguments",
        )
    )

    with pytest.raises(ValueError) as excinfo:
        taskmodule.encode([doc])
    assert (
        str(excinfo.value)
        == "the taskmodule expects the relation arguments to be of type LabeledSpan, but got "
        "[<class 'pytorch_ie.annotations.Label'>, <class 'pytorch_ie.annotations.Label'>, "
        "<class 'pytorch_ie.annotations.Label'>]"
    )


def test_encode_unknown_relation_type():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        # setting label_to_id and entity_labels makes the taskmodule prepared
        label_to_id={"has_wrong_type": 1},
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
        "the taskmodule does not yet support relations of type: "
    ) and str(excinfo.value).endswith("<locals>.UnknownRelation'>")


def test_encode_with_unaligned_span(caplog):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        # setting label_to_id and entity_labels makes the taskmodule prepared
        label_to_id={"rel": 1},
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
    doc.entities.extend([entity1, entity2])
    # the start of entity2 is not aligned with a token
    assert str(entity2) == " space"
    doc.relations.append(BinaryRelation(head=entity1, tail=entity2, label="rel"))

    task_encodings = taskmodule.encode([doc])
    # the relation is skipped because the start of entity2 is not aligned with a token
    assert len(task_encodings) == 0

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "Skipping invalid example doc1, cannot get argument token slice(s)"
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
        tokenizer_name_or_path=tokenizer_name_or_path,
        log_first_n_examples=1,
    )
    taskmodule.prepare([doc])

    # we need to set the log level to INFO, otherwise the log messages are not captured
    caplog.set_level(logging.INFO)
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
