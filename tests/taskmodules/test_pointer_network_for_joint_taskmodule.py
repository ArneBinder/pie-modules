import logging
from dataclasses import dataclass
from typing import Dict, List, Set

import pytest
import torch
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, Document, annotation_field
from pytorch_ie.documents import TextBasedDocument

from pie_modules.taskmodules import PointerNetworkForJointTaskModule
from tests import FIXTURES_ROOT

logger = logging.getLogger(__name__)

FIXTURES_DIR = FIXTURES_ROOT / "taskmodules" / "gmam_taskmodule"


def _config_to_str(cfg: Dict[str, str]) -> str:
    result = "-".join([f"{k}={cfg[k]}" for k in sorted(cfg)])
    return result


CONFIGS = [
    {"span_end_mode": "first_token_of_last_word"},
    {"span_end_mode": "last_token"},
]
CONFIGS_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}

DUMP_FIXTURE_DATA = False


@pytest.fixture(scope="module", params=CONFIGS_DICT.keys())
def config(request):
    return CONFIGS_DICT[request.param]


@pytest.fixture(scope="module")
def config_str(config):
    return _config_to_str(config)


@pytest.fixture(scope="module")
def simple_document():
    @dataclass
    class ExampleDocument(TextBasedDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
        sentences: AnnotationList[Span] = annotation_field(target="text")

    doc = ExampleDocument(text="This is a dummy text about nothing. Trust me.")
    span1 = LabeledSpan(start=10, end=20, label="content")
    span2 = LabeledSpan(start=27, end=34, label="topic")
    span3 = LabeledSpan(start=42, end=44, label="person")
    doc.entities.extend([span1, span2, span3])
    rel = BinaryRelation(head=span1, tail=span2, label="is_about")
    doc.relations.append(rel)
    no_rel = BinaryRelation(head=span1, tail=span3, label="no_relation")
    doc.relations.append(no_rel)
    sent1 = Span(start=0, end=35)
    sent2 = Span(start=36, end=45)
    doc.sentences.extend([sent1, sent2])
    return doc


def test_simple_document(simple_document):
    spans = simple_document.entities
    assert len(spans) == 3
    assert (str(spans[0]), spans[0].label) == ("dummy text", "content")
    assert (str(spans[1]), spans[1].label) == ("nothing", "topic")
    assert (str(spans[2]), spans[2].label) == ("me", "person")
    relations = simple_document.relations
    assert len(relations) == 2
    assert (str(relations[0].head), relations[0].label, str(relations[0].tail)) == (
        "dummy text",
        "is_about",
        "nothing",
    )
    assert (str(relations[1].head), relations[1].label, str(relations[1].tail)) == (
        "dummy text",
        "no_relation",
        "me",
    )
    sentences = simple_document.sentences
    assert len(sentences) == 2
    assert str(sentences[0]) == "This is a dummy text about nothing."
    assert str(sentences[1]) == "Trust me."


SIMPLE_CONFIGS = [{}, {"partition_layer_name": "sentences"}]
SIMPLE_CONFIG_DICT = {_config_to_str(cfg): cfg for cfg in SIMPLE_CONFIGS}


@pytest.fixture(scope="module", params=SIMPLE_CONFIG_DICT.keys())
def simple_config_str(request):
    return request.param


@pytest.fixture(scope="module")
def simple_config(simple_config_str):
    return SIMPLE_CONFIG_DICT[simple_config_str]


@pytest.fixture(scope="module")
def simple_taskmodule(simple_document, simple_config):
    taskmodule = PointerNetworkForJointTaskModule(
        text_field_name="text",
        span_layer_name="entities",
        relation_layer_name="relations",
        exclude_annotation_names={"relations": ["no_relation"]},
        **simple_config,
    )

    taskmodule.prepare(documents=[simple_document])
    return taskmodule


def test_simple_taskmodule(simple_taskmodule):
    tm = simple_taskmodule
    assert tm.prepared_attributes == {
        "span_labels": ["content", "person", "topic"],
        "relation_labels": ["is_about"],
    }
    assert tm.labels == ["content", "person", "topic", "is_about", "none"]
    assert tm.label_tokens == [
        "<<is_about>>",
        "<<content>>",
        "<<person>>",
        "<<topic>>",
        "<<none>>",
    ]
    assert tm.label_ids == [2, 3, 4, 5, 6]
    assert tm.target_tokens == [
        "<s>",
        "</s>",
        "<<topic>>",
        "<<none>>",
        "<<is_about>>",
        "<<person>>",
        "<<content>>",
    ]
    assert tm.target_token_ids == [0, 2, 50265, 50266, 50267, 50268, 50269]


@pytest.fixture()
def simple_encoded_inputs(simple_taskmodule, simple_document):
    return simple_taskmodule.encode_input(simple_document)


@pytest.fixture()
def simple_encoded_input(simple_encoded_inputs):
    return simple_encoded_inputs[0]


def test_simple_encoded_input(simple_encoded_input, simple_taskmodule):
    assert simple_encoded_input is not None
    if simple_taskmodule.partition_layer_name is None:
        assert simple_encoded_input.inputs == {
            "src_tokens": [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 3101, 162, 4, 2],
            "src_seq_len": 13,
        }
        assert set(simple_encoded_input.metadata) == {"token2char", "char2token", "tokenized_span"}
        token2char = simple_encoded_input.metadata["token2char"]
        assert token2char == [
            (0, 0),
            (0, 4),
            (5, 7),
            (8, 9),
            (10, 15),
            (16, 20),
            (21, 26),
            (27, 34),
            (34, 35),
            (36, 41),
            (42, 44),
            (44, 45),
            (0, 0),
        ]
        char2token = simple_encoded_input.metadata["char2token"]
        assert char2token == {
            0: [1],
            1: [1],
            2: [1],
            3: [1],
            5: [2],
            6: [2],
            8: [3],
            10: [4],
            11: [4],
            12: [4],
            13: [4],
            14: [4],
            16: [5],
            17: [5],
            18: [5],
            19: [5],
            21: [6],
            22: [6],
            23: [6],
            24: [6],
            25: [6],
            27: [7],
            28: [7],
            29: [7],
            30: [7],
            31: [7],
            32: [7],
            33: [7],
            34: [8],
            36: [9],
            37: [9],
            38: [9],
            39: [9],
            40: [9],
            42: [10],
            43: [10],
            44: [11],
        }
        assert simple_encoded_input.metadata.get("partition") is None
        tokenized_span = simple_encoded_input.metadata["tokenized_span"]
        text = simple_encoded_input.document.text
        assert (
            text[tokenized_span.start : tokenized_span.end]
            == "This is a dummy text about nothing. Trust me."
        )
    elif simple_taskmodule.partition_layer_name == "sentences":
        assert simple_encoded_input.inputs == {
            "src_tokens": [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 2],
            "src_seq_len": 10,
        }
        assert set(simple_encoded_input.metadata) == {
            "token2char",
            "char2token",
            "partition",
            "tokenized_span",
        }
        token2char = simple_encoded_input.metadata["token2char"]
        assert token2char == [
            (0, 0),
            (0, 4),
            (5, 7),
            (8, 9),
            (10, 15),
            (16, 20),
            (21, 26),
            (27, 34),
            (34, 35),
            (0, 0),
        ]
        char2token = simple_encoded_input.metadata["char2token"]
        assert char2token == {
            0: [1],
            1: [1],
            2: [1],
            3: [1],
            5: [2],
            6: [2],
            8: [3],
            10: [4],
            11: [4],
            12: [4],
            13: [4],
            14: [4],
            16: [5],
            17: [5],
            18: [5],
            19: [5],
            21: [6],
            22: [6],
            23: [6],
            24: [6],
            25: [6],
            27: [7],
            28: [7],
            29: [7],
            30: [7],
            31: [7],
            32: [7],
            33: [7],
            34: [8],
        }
        partition = simple_encoded_input.metadata.get("partition")
        assert (partition.start, partition.end) == (0, 35)
        tokenized_span = simple_encoded_input.metadata["tokenized_span"]
        text = simple_encoded_input.document.text
        assert (
            text[tokenized_span.start : tokenized_span.end]
            == "This is a dummy text about nothing."
        )
    else:
        raise Exception(f"unknown partition_layer_name: {simple_taskmodule.partition_layer_name}")


@pytest.fixture()
def simple_task_encodings(simple_taskmodule, simple_encoded_inputs):
    for encoded_input in simple_encoded_inputs:
        targets = simple_taskmodule.encode_target(encoded_input)
        encoded_input.targets = targets
    return simple_encoded_inputs


@pytest.fixture()
def simple_task_encoding(simple_taskmodule, simple_task_encodings):
    return simple_task_encodings[0]


def test_encode_target_with_dummy_relations(simple_task_encoding, simple_taskmodule):
    targets = simple_task_encoding.targets
    if simple_taskmodule.partition_layer_name is None:
        assert targets["tgt_tokens"] == [
            0,
            14,
            14,
            5,
            11,
            12,
            3,
            2,
            17,
            17,
            4,
            6,
            6,
            6,
            6,
            1,
        ]
        assert targets["tgt_seq_len"] == 16
        assert targets["CPM_tag"] == [
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    elif simple_taskmodule.partition_layer_name == "sentences":
        assert targets["tgt_tokens"] == [0, 14, 14, 5, 11, 12, 3, 2, 1]
        assert targets["tgt_seq_len"] == 9
        assert targets["CPM_tag"] == [
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    else:
        raise Exception(f"unknown partition_layer_name: {simple_taskmodule.partition_layer_name}")


@pytest.fixture()
def simple_batch(simple_taskmodule, simple_task_encodings):
    return simple_taskmodule.collate(simple_task_encodings)


def test_simple_collate(simple_batch, simple_taskmodule):
    inputs, targets = simple_batch
    if simple_taskmodule.partition_layer_name is None:
        torch.testing.assert_close(
            inputs["src_tokens"],
            torch.tensor([[0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 3101, 162, 4, 2]]),
        )
        torch.testing.assert_close(inputs["src_seq_len"], torch.tensor([13]))

        torch.testing.assert_close(
            targets["tgt_tokens"],
            torch.tensor([[0, 14, 14, 5, 11, 12, 3, 2, 17, 17, 4, 6, 6, 6, 6, 1]]),
        )
        torch.testing.assert_close(targets["tgt_seq_len"], torch.tensor([16]))
        torch.testing.assert_close(
            targets["CPM_tag"],
            torch.tensor(
                [
                    [
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ]
            ),
        )
    elif simple_taskmodule.partition_layer_name == "sentences":
        torch.testing.assert_close(
            inputs["src_tokens"],
            torch.tensor(
                [
                    [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 2],
                    [0, 18823, 162, 4, 2, 1, 1, 1, 1, 1],
                ]
            ),
        )
        torch.testing.assert_close(inputs["src_seq_len"], torch.tensor([10, 5]))

        torch.testing.assert_close(
            targets["tgt_tokens"],
            torch.tensor(
                [
                    [0, 14, 14, 5, 11, 12, 3, 2, 1],
                    [0, 9, 9, 4, 6, 6, 6, 6, 1],
                ]
            ),
        )
        torch.testing.assert_close(targets["tgt_seq_len"], torch.tensor([9, 9]))
        torch.testing.assert_close(
            targets["CPM_tag"],
            torch.tensor(
                [
                    [
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                    ],
                ]
            ),
        )
    else:
        raise Exception(f"unknown partition_layer_name: {simple_taskmodule.partition_layer_name}")


@pytest.fixture()
def simple_unbatched_output(simple_taskmodule, simple_batch):
    inputs, targets = simple_batch
    # because the model is trained to reproduce the target tokens, we can just use them as model prediction
    model_output = {"pred": targets["tgt_tokens"]}
    return simple_taskmodule.unbatch_output(model_output)


@pytest.fixture()
def simple_task_outputs(simple_unbatched_output):
    return simple_unbatched_output


@pytest.fixture()
def simple_task_output(simple_task_outputs):
    return simple_task_outputs[0]


def test_simple_task_output(simple_task_output, simple_taskmodule):
    if simple_taskmodule.partition_layer_name is None:
        assert simple_task_output.tolist() == [
            0,
            14,
            14,
            5,
            11,
            12,
            3,
            2,
            17,
            17,
            4,
            6,
            6,
            6,
            6,
            1,
        ]
    elif simple_taskmodule.partition_layer_name == "sentences":
        assert simple_task_output.tolist() == [0, 14, 14, 5, 11, 12, 3, 2, 1]
    else:
        raise Exception(f"unknown partition_layer_name: {simple_taskmodule.partition_layer_name}")


def _test_annotations_from_output(task_encodings, task_outputs, taskmodule, layer_names_expected):
    assert len(task_outputs) == len(task_encodings)

    # this needs to be outside the below loop because documents can contain duplicates
    # which would break the comparison when clearing predictions that were already added
    for task_encoding in task_encodings:
        for layer_name in layer_names_expected:
            task_encoding.document[layer_name].predictions.clear()

    layer_names: Set[str] = set()
    # Note: this list may contain duplicates!
    documents: List[Document] = []
    for i in range(len(task_outputs)):
        task_encoding = task_encodings[i]
        task_output = task_outputs[i]
        documents.append(task_encoding.document)

        for layer_name, annotation in taskmodule.create_annotations_from_output(
            task_encoding=task_encoding, task_output=task_output
        ):
            task_encoding.document[layer_name].predictions.append(annotation)
            layer_names.add(layer_name)

    assert layer_names == layer_names_expected

    for document in documents:
        for layer_name in layer_names:
            layer = {
                str(ann)
                for ann in document[layer_name].predictions
                if taskmodule._is_valid_annotation(ann)
            }
            layer_expected = {
                str(ann) for ann in document[layer_name] if taskmodule._is_valid_annotation(ann)
            }
            assert layer == layer_expected

    # this needs to be outside the above loop because documents can contain duplicates
    # which would break the comparison when clearing predictions too early
    for document in documents:
        for layer_name in layer_names:
            document[layer_name].predictions.clear()


def test_simple_annotations_from_output(
    simple_task_encodings, simple_task_outputs, simple_taskmodule
):
    _test_annotations_from_output(
        taskmodule=simple_taskmodule,
        task_encodings=simple_task_encodings,
        task_outputs=simple_task_outputs,
        layer_names_expected={"entities", "relations"},
    )
