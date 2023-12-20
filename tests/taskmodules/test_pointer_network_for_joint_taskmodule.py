import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Set

import pytest
import torch
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, Document, annotation_field
from pytorch_ie.documents import TextBasedDocument

from pie_modules.taskmodules import PointerNetworkTaskModule
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
        sentences: AnnotationList[LabeledSpan] = annotation_field(target="text")

    doc = ExampleDocument(text="This is a dummy text about nothing. Trust me.")
    span1 = LabeledSpan(start=10, end=20, label="content")
    span2 = LabeledSpan(start=27, end=34, label="topic")
    span3 = LabeledSpan(start=42, end=44, label="person")
    doc.entities.extend([span1, span2, span3])
    assert str(span1) == "dummy text"
    assert str(span2) == "nothing"
    assert str(span3) == "me"
    rel = BinaryRelation(head=span1, tail=span2, label="is_about")
    doc.relations.append(rel)
    assert str(rel.label) == "is_about"
    assert str(rel.head) == "dummy text"
    assert str(rel.tail) == "nothing"

    no_rel = BinaryRelation(head=span1, tail=span3, label="no_relation")
    doc.relations.append(no_rel)
    assert str(no_rel.label) == "no_relation"
    assert str(no_rel.head) == "dummy text"
    assert str(no_rel.tail) == "me"

    sent1 = LabeledSpan(start=0, end=35, label="1")
    sent2 = LabeledSpan(start=36, end=45, label="2")
    doc.sentences.extend([sent1, sent2])
    assert str(sent1) == "This is a dummy text about nothing."
    assert str(sent2) == "Trust me."
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
    taskmodule = PointerNetworkTaskModule(
        annotation_encoder_decoder_kwargs={
            "span_layer_name": "entities",
            "relation_layer_name": "relations",
            "exclude_labels_per_layer": {"relations": ["no_relation"]},
        },
        annotation_field_mapping={
            "entities": "labeled_spans",
            "relations": "binary_relations",
        },
        **config,
    )

    taskmodule.prepare(documents=[document])
    return taskmodule


def test_taskmodule(taskmodule):
    # check the annotation_encoder_decoder
    annotation_encoder_decoder = taskmodule.annotation_encoder_decoder
    assert annotation_encoder_decoder.is_prepared
    assert annotation_encoder_decoder.prepared_attributes == {
        "labels_per_layer": {
            "entities": ["content", "person", "topic"],
            "relations": ["is_about"],
        },
    }
    assert annotation_encoder_decoder.layer_names == ["entities", "relations"]
    assert annotation_encoder_decoder.special_targets == ["<s>", "</s>"]
    assert annotation_encoder_decoder.labels == ["none", "content", "person", "topic", "is_about"]
    assert annotation_encoder_decoder.targets == [
        "<s>",
        "</s>",
        "none",
        "content",
        "person",
        "topic",
        "is_about",
    ]
    assert annotation_encoder_decoder.bos_id == 0
    assert annotation_encoder_decoder.eos_id == 1
    assert annotation_encoder_decoder.none_id == 2
    assert annotation_encoder_decoder.span_ids == [3, 4, 5]
    assert annotation_encoder_decoder.relation_ids == [6]
    assert annotation_encoder_decoder.label2id == {
        "content": 3,
        "is_about": 6,
        "none": 2,
        "person": 4,
        "topic": 5,
    }

    # check taskmodule properties
    assert taskmodule.prepared_attributes == {
        "annotation_encoder_decoder_kwargs": {
            "span_layer_name": "entities",
            "relation_layer_name": "relations",
            "exclude_labels_per_layer": {"relations": ["no_relation"]},
            "bos_token": "<s>",
            "eos_token": "</s>",
            "labels_per_layer": {
                "entities": ["content", "person", "topic"],
                "relations": ["is_about"],
            },
        }
    }
    assert taskmodule.label_embedding_weight_mapping == {
        50265: [45260],
        50266: [39763],
        50267: [354, 1215, 9006],
        50268: [5970],
        50269: [10166],
    }
    assert taskmodule.target_tokens == [
        "<s>",
        "</s>",
        "<<none>>",
        "<<content>>",
        "<<person>>",
        "<<topic>>",
        "<<is_about>>",
    ]
    assert taskmodule.target_token_ids == [0, 2, 50266, 50269, 50268, 50265, 50267]


def test_prepared_config(taskmodule, config):
    if config == {}:
        assert taskmodule._config() == {
            "taskmodule_type": "PointerNetworkTaskModule",
            "document_type": "pytorch_ie.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
            "tokenized_document_type": "pie_modules.documents.TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
            "tokenizer_name_or_path": "facebook/bart-base",
            "tokenizer_init_kwargs": None,
            "tokenizer_kwargs": None,
            "partition_layer_name": None,
            "annotation_field_mapping": {
                "entities": "labeled_spans",
                "relations": "binary_relations",
            },
            "annotation_encoder_decoder_name": "pointer_network_span_and_relation",
            "annotation_encoder_decoder_kwargs": {
                "span_layer_name": "entities",
                "relation_layer_name": "relations",
                "exclude_labels_per_layer": {"relations": ["no_relation"]},
                "bos_token": "<s>",
                "eos_token": "</s>",
                "labels_per_layer": {
                    "entities": ["content", "person", "topic"],
                    "relations": ["is_about"],
                },
            },
            "label_tokens": None,
            "label_representations": None,
            "max_target_length": None,
            "create_constraints": True,
            "log_first_n_examples": None,
        }
    elif config == {"partition_layer_name": "sentences"}:
        assert taskmodule._config() == {
            "taskmodule_type": "PointerNetworkTaskModule",
            "document_type": "pytorch_ie.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
            "tokenized_document_type": "pie_modules.documents.TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
            "tokenizer_name_or_path": "facebook/bart-base",
            "tokenizer_init_kwargs": None,
            "tokenizer_kwargs": None,
            "partition_layer_name": "sentences",
            "annotation_field_mapping": {
                "entities": "labeled_spans",
                "relations": "binary_relations",
            },
            "annotation_encoder_decoder_name": "pointer_network_span_and_relation",
            "annotation_encoder_decoder_kwargs": {
                "span_layer_name": "entities",
                "relation_layer_name": "relations",
                "exclude_labels_per_layer": {"relations": ["no_relation"]},
                "bos_token": "<s>",
                "eos_token": "</s>",
                "labels_per_layer": {
                    "entities": ["content", "person", "topic"],
                    "relations": ["is_about"],
                },
            },
            "label_tokens": None,
            "label_representations": None,
            "max_target_length": None,
            "create_constraints": True,
            "log_first_n_examples": None,
        }


@pytest.fixture()
def encoded_inputs(taskmodule, document):
    return taskmodule.encode_input(document)


@pytest.fixture()
def encoded_input(encoded_inputs):
    return encoded_inputs[0]


def test_encoded_input(encoded_input, taskmodule):
    assert encoded_input is not None
    tokens = taskmodule.tokenizer.convert_ids_to_tokens(encoded_input.inputs.src_tokens)
    if taskmodule.partition_layer_name is None:
        assert asdict(encoded_input.inputs) == {
            "src_tokens": [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 3101, 162, 4, 2],
            "src_attention_mask": [1] * 13,
            "src_seq_len": 13,
        }
    elif taskmodule.partition_layer_name == "sentences":
        assert asdict(encoded_input.inputs) == {
            "src_tokens": [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 2],
            "src_attention_mask": [1] * 10,
            "src_seq_len": 10,
        }
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
    targets = asdict(task_encoding.targets)
    if taskmodule.partition_layer_name is None:
        assert targets["tgt_tokens"] == [0, 14, 14, 5, 11, 12, 3, 6, 17, 17, 4, 2, 2, 2, 2, 1]
        assert targets["tgt_seq_len"] == 16
        assert targets["CPM_tag"] == [
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    elif taskmodule.partition_layer_name == "sentences":
        assert targets["tgt_tokens"] == [0, 14, 14, 5, 11, 12, 3, 6, 1]
        assert targets["tgt_seq_len"] == 9
        assert targets["CPM_tag"] == [
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
            "src_attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        }
        assert targets_lists == {
            "CPM_tag": [
                [
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ],
            "tgt_seq_len": [16],
            "tgt_tokens": [[0, 14, 14, 5, 11, 12, 3, 6, 17, 17, 4, 2, 2, 2, 2, 1]],
            "tgt_attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        }
    elif taskmodule.partition_layer_name == "sentences":
        assert inputs_lists == {
            "src_seq_len": [10, 5],
            "src_tokens": [
                [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 2],
                [0, 18823, 162, 4, 2, 1, 1, 1, 1, 1],
            ],
            "src_attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]],
        }
        assert targets_lists == {
            "CPM_tag": [
                [
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                ],
            ],
            "tgt_seq_len": [9, 9],
            "tgt_tokens": [[0, 14, 14, 5, 11, 12, 3, 6, 1], [0, 9, 9, 4, 2, 2, 2, 2, 1]],
            "tgt_attention_mask": [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
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
        assert output_list == [0, 14, 14, 5, 11, 12, 3, 6, 17, 17, 4, 2, 2, 2, 2, 1]
    elif taskmodule.partition_layer_name == "sentences":
        assert output_list == [0, 14, 14, 5, 11, 12, 3, 6, 1]
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
                if ann.label in taskmodule.annotation_encoder_decoder.labels_per_layer[layer_name]
            }
            layer_expected = {
                str(ann)
                for ann in document[layer_name]
                if ann.label in taskmodule.annotation_encoder_decoder.labels_per_layer[layer_name]
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
