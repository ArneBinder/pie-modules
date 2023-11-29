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

# FIXTURES_DIR = FIXTURES_ROOT / "taskmodules" / "gmam_taskmodule"

DUMP_FIXTURE_DATA = False


def _config_to_str(cfg: Dict[str, str]) -> str:
    result = "-".join([f"{k}={cfg[k]}" for k in sorted(cfg)])
    return result


CONFIGS = [{}, {"partition_layer_name": "sentences"}]
CONFIG_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIG_DICT.keys())
def config_str(request):
    return request.param


@pytest.fixture(scope="module")
def config(config_str):
    return CONFIG_DICT[config_str]


@pytest.fixture(scope="module")
def document():
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


def test_document(document):
    spans = document.entities
    assert len(spans) == 3
    assert (str(spans[0]), spans[0].label) == ("dummy text", "content")
    assert (str(spans[1]), spans[1].label) == ("nothing", "topic")
    assert (str(spans[2]), spans[2].label) == ("me", "person")
    relations = document.relations
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
    sentences = document.sentences
    assert len(sentences) == 2
    assert str(sentences[0]) == "This is a dummy text about nothing."
    assert str(sentences[1]) == "Trust me."


@pytest.fixture(scope="module")
def taskmodule(document, config):
    taskmodule = PointerNetworkForJointTaskModule(
        text_field_name="text",
        span_layer_name="entities",
        relation_layer_name="relations",
        exclude_annotation_names={"relations": ["no_relation"]},
        **config,
    )

    taskmodule.prepare(documents=[document])
    return taskmodule


def test_taskmodule(taskmodule):
    tm = taskmodule
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
def encoded_inputs(taskmodule, document):
    return taskmodule.encode_input(document)


@pytest.fixture()
def encoded_input(encoded_inputs):
    return encoded_inputs[0]


def test_encoded_input(encoded_input, taskmodule):
    assert encoded_input is not None
    if taskmodule.partition_layer_name is None:
        assert encoded_input.inputs == {
            "src_tokens": [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 3101, 162, 4, 2],
            "src_seq_len": 13,
        }
        assert set(encoded_input.metadata) == {"token2char", "char2token", "tokenized_span"}
        token2char = encoded_input.metadata["token2char"]
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
        char2token = encoded_input.metadata["char2token"]
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
        assert encoded_input.metadata.get("partition") is None
        tokenized_span = encoded_input.metadata["tokenized_span"]
        text = encoded_input.document.text
        assert (
            text[tokenized_span.start : tokenized_span.end]
            == "This is a dummy text about nothing. Trust me."
        )
    elif taskmodule.partition_layer_name == "sentences":
        assert encoded_input.inputs == {
            "src_tokens": [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 2],
            "src_seq_len": 10,
        }
        assert set(encoded_input.metadata) == {
            "token2char",
            "char2token",
            "partition",
            "tokenized_span",
        }
        token2char = encoded_input.metadata["token2char"]
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
        char2token = encoded_input.metadata["char2token"]
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
        partition = encoded_input.metadata.get("partition")
        assert (partition.start, partition.end) == (0, 35)
        tokenized_span = encoded_input.metadata["tokenized_span"]
        text = encoded_input.document.text
        assert (
            text[tokenized_span.start : tokenized_span.end]
            == "This is a dummy text about nothing."
        )
    else:
        raise Exception(f"unknown partition_layer_name: {taskmodule.partition_layer_name}")


@pytest.fixture()
def task_encodings(taskmodule, encoded_inputs):
    for encoded_input in encoded_inputs:
        targets = taskmodule.encode_target(encoded_input)
        encoded_input.targets = targets
    return encoded_inputs


@pytest.fixture()
def task_encoding(taskmodule, task_encodings):
    return task_encodings[0]


def test_encode_target_with_dummy_relations(task_encoding, taskmodule):
    targets = task_encoding.targets
    if taskmodule.partition_layer_name is None:
        assert targets["tgt_tokens"] == [0, 14, 14, 2, 11, 12, 6, 4, 17, 17, 5, 3, 3, 3, 3, 1]
        assert targets["tgt_seq_len"] == 16
        assert targets["CPM_tag"] == [
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    elif taskmodule.partition_layer_name == "sentences":
        assert targets["tgt_tokens"] == [0, 14, 14, 2, 11, 12, 6, 4, 1]
        assert targets["tgt_seq_len"] == 9
        assert targets["CPM_tag"] == [
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    else:
        raise Exception(f"unknown partition_layer_name: {taskmodule.partition_layer_name}")


@pytest.fixture()
def batch(taskmodule, task_encodings):
    return taskmodule.collate(task_encodings)


def test_collate(batch, taskmodule):
    inputs, targets = batch
    for tensor in inputs.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.int64
    for tensor in targets.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.int64
    inputs_lists = {k: inputs[k].tolist() for k in sorted(inputs)}
    targets_lists = {k: targets[k].tolist() for k in sorted(targets)}
    if taskmodule.partition_layer_name is None:
        assert inputs_lists == {
            "src_seq_len": [13],
            "src_tokens": [[0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 3101, 162, 4, 2]],
        }
        assert targets_lists == {
            "CPM_tag": [
                [
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ],
            "tgt_seq_len": [16],
            "tgt_tokens": [[0, 14, 14, 2, 11, 12, 6, 4, 17, 17, 5, 3, 3, 3, 3, 1]],
        }
    elif taskmodule.partition_layer_name == "sentences":
        assert inputs_lists == {
            "src_seq_len": [10, 5],
            "src_tokens": [
                [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 2],
                [0, 18823, 162, 4, 2, 1, 1, 1, 1, 1],
            ],
        }
        assert targets_lists == {
            "CPM_tag": [
                [
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                ],
            ],
            "tgt_seq_len": [9, 9],
            "tgt_tokens": [[0, 14, 14, 2, 11, 12, 6, 4, 1], [0, 9, 9, 5, 3, 3, 3, 3, 1]],
        }
    else:
        raise Exception(f"unknown partition_layer_name: {taskmodule.partition_layer_name}")


@pytest.fixture()
def unbatched_output(taskmodule, batch):
    inputs, targets = batch
    # because the model is trained to reproduce the target tokens, we can just use them as model prediction
    model_output = {"pred": targets["tgt_tokens"]}
    return taskmodule.unbatch_output(model_output)


@pytest.fixture()
def task_outputs(unbatched_output):
    return unbatched_output


@pytest.fixture()
def task_output(task_outputs):
    return task_outputs[0]


def test_task_output(task_output, taskmodule):
    output_list = task_output.tolist()
    if taskmodule.partition_layer_name is None:
        assert output_list == [0, 14, 14, 2, 11, 12, 6, 4, 17, 17, 5, 3, 3, 3, 3, 1]
    elif taskmodule.partition_layer_name == "sentences":
        assert output_list == [0, 14, 14, 2, 11, 12, 6, 4, 1]
    else:
        raise Exception(f"unknown partition_layer_name: {taskmodule.partition_layer_name}")


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


def test_annotations_from_output(task_encodings, task_outputs, taskmodule):
    _test_annotations_from_output(
        taskmodule=taskmodule,
        task_encodings=task_encodings,
        task_outputs=task_outputs,
        layer_names_expected={"entities", "relations"},
    )
