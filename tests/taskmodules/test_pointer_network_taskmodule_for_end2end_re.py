import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Set

import pytest
import torch
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, Document, annotation_field
from pytorch_ie.documents import TextBasedDocument

from pie_modules.taskmodules import PointerNetworkTaskModuleForEnd2EndRE
from pie_modules.taskmodules.common.metrics import AnnotationLayerMetric
from pie_modules.taskmodules.pointer_network_taskmodule_for_end2end_re import (
    EncodingWithIdsAndOptionalCpmTag,
)

logger = logging.getLogger(__name__)

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
    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        span_layer_name="entities",
        relation_layer_name="relations",
        exclude_labels_per_layer={"relations": ["no_relation"]},
        annotation_field_mapping={
            "entities": "labeled_spans",
            "relations": "binary_relations",
        },
        create_constraints=True,
        tokenizer_kwargs={"strict_span_conversion": False},
        **config,
    )

    taskmodule.prepare(documents=[document])
    return taskmodule


def test_taskmodule(taskmodule):
    assert taskmodule.is_prepared
    assert taskmodule.prepared_attributes == {
        "labels_per_layer": {
            "entities": ["content", "person", "topic"],
            "relations": ["is_about"],
        },
    }
    assert taskmodule.layer_names == ["entities", "relations"]
    assert taskmodule.special_targets == ["<s>", "</s>"]
    assert taskmodule.labels == ["none", "content", "person", "topic", "is_about"]
    assert taskmodule.targets == [
        "<s>",
        "</s>",
        "none",
        "content",
        "person",
        "topic",
        "is_about",
    ]
    assert taskmodule.bos_id == 0
    assert taskmodule.eos_id == 1
    assert taskmodule.none_id == 2
    assert taskmodule.span_ids == [3, 4, 5]
    assert taskmodule.relation_ids == [6]
    assert taskmodule.label2id == {
        "content": 3,
        "is_about": 6,
        "none": 2,
        "person": 4,
        "topic": 5,
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
            "taskmodule_type": "PointerNetworkTaskModuleForEnd2EndRE",
            "span_layer_name": "entities",
            "relation_layer_name": "relations",
            "none_label": "none",
            "loop_dummy_relation_name": "loop",
            "labels_per_layer": {
                "entities": ["content", "person", "topic"],
                "relations": ["is_about"],
            },
            "exclude_labels_per_layer": {"relations": ["no_relation"]},
            "create_constraints": True,
            "document_type": "pytorch_ie.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
            "tokenized_document_type": "pie_modules.documents.TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
            "tokenizer_name_or_path": "facebook/bart-base",
            "tokenizer_init_kwargs": None,
            "tokenizer_kwargs": {"strict_span_conversion": False},
            "partition_layer_name": None,
            "annotation_field_mapping": {
                "entities": "labeled_spans",
                "relations": "binary_relations",
            },
            "label_tokens": None,
            "label_representations": None,
            "log_first_n_examples": None,
        }
    elif config == {"partition_layer_name": "sentences"}:
        assert taskmodule._config() == {
            "taskmodule_type": "PointerNetworkTaskModuleForEnd2EndRE",
            "span_layer_name": "entities",
            "relation_layer_name": "relations",
            "none_label": "none",
            "loop_dummy_relation_name": "loop",
            "labels_per_layer": {
                "entities": ["content", "person", "topic"],
                "relations": ["is_about"],
            },
            "exclude_labels_per_layer": {"relations": ["no_relation"]},
            "create_constraints": True,
            "document_type": "pytorch_ie.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
            "tokenized_document_type": "pie_modules.documents.TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
            "tokenizer_name_or_path": "facebook/bart-base",
            "tokenizer_init_kwargs": None,
            "tokenizer_kwargs": {"strict_span_conversion": False},
            "partition_layer_name": "sentences",
            "annotation_field_mapping": {
                "entities": "labeled_spans",
                "relations": "binary_relations",
            },
            "label_tokens": None,
            "label_representations": None,
            "log_first_n_examples": None,
        }
    else:
        raise Exception(f"unknown config: {config}")


@pytest.fixture()
def task_encoding_without_target(taskmodule, document):
    return taskmodule.encode_input(document)[0]


def test_input_encoding(task_encoding_without_target, taskmodule):
    assert task_encoding_without_target is not None
    tokens = taskmodule.tokenizer.convert_ids_to_tokens(
        task_encoding_without_target.inputs.src_tokens
    )
    if taskmodule.partition_layer_name is None:
        assert asdict(task_encoding_without_target.inputs) == {
            "src_tokens": [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 3101, 162, 4, 2],
            "src_attention_mask": [1] * 13,
        }
    elif taskmodule.partition_layer_name == "sentences":
        assert asdict(task_encoding_without_target.inputs) == {
            "src_tokens": [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 2],
            "src_attention_mask": [1] * 10,
        }
    else:
        raise Exception(f"unknown partition_layer_name: {taskmodule.partition_layer_name}")


@pytest.fixture()
def target_encoding(taskmodule, task_encoding_without_target):
    return taskmodule.encode_target(task_encoding_without_target)


def test_target_encoding(target_encoding, taskmodule):
    assert target_encoding is not None
    if taskmodule.partition_layer_name is None:
        assert asdict(target_encoding) == {
            "tgt_tokens": [0, 14, 14, 5, 11, 12, 3, 6, 17, 17, 4, 2, 2, 2, 2, 1],
            "CPM_tag": [
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
            ],
        }
    elif taskmodule.partition_layer_name == "sentences":
        assert asdict(target_encoding) == {
            "tgt_tokens": [0, 14, 14, 5, 11, 12, 3, 6, 1],
            "CPM_tag": [
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        }
    else:
        raise Exception(f"unknown partition_layer_name: {taskmodule.partition_layer_name}")


@pytest.fixture()
def task_encoding(task_encoding_without_target, target_encoding):
    task_encoding_without_target.targets = target_encoding
    return task_encoding_without_target


def test_maybe_log_example(taskmodule, task_encoding, caplog, config):
    original_log_first_n_examples = taskmodule.log_first_n_examples
    taskmodule.log_first_n_examples = 1
    caplog.clear()
    with caplog.at_level(logging.INFO):
        taskmodule.maybe_log_example(task_encoding)
    if config == {}:
        assert caplog.messages == [
            "*** Example ***",
            "doc.id:        None-tokenized-1-of-1",
            "src_token_ids: 0 713 16 10 34759 2788 59 1085 4 3101 162 4 2",
            "src_tokens:    <s> This Ġis Ġa Ġdummy Ġtext Ġabout Ġnothing . ĠTrust Ġme . </s>",
            "tgt_token_ids: 0 14 14 5 11 12 3 6 17 17 4 2 2 2 2 1",
            "tgt_tokens:    <s> 14 {Ġnothing} 14 {Ġnothing} topic 11 {Ġdummy} 12 {Ġtext} content is_about 17 {Ġme} 17 {Ġme} person none none none none </s>",
        ]
    elif config == {"partition_layer_name": "sentences"}:
        assert caplog.messages == [
            "*** Example ***",
            "doc.id:        None-tokenized-1-of-2",
            "src_token_ids: 0 713 16 10 34759 2788 59 1085 4 2",
            "src_tokens:    <s> This Ġis Ġa Ġdummy Ġtext Ġabout Ġnothing . </s>",
            "tgt_token_ids: 0 14 14 5 11 12 3 6 1",
            "tgt_tokens:    <s> 14 {Ġnothing} 14 {Ġnothing} topic 11 {Ġdummy} 12 {Ġtext} content is_about </s>",
        ]
    else:
        raise Exception(f"unknown config: {config}")

    # restore original value
    taskmodule.log_first_n_examples = original_log_first_n_examples


def test_maybe_log_example_disabled(taskmodule, task_encoding, caplog):
    original_log_first_n_examples = taskmodule.log_first_n_examples
    taskmodule.log_first_n_examples = None
    caplog.clear()
    with caplog.at_level(logging.INFO):
        taskmodule.maybe_log_example(task_encoding)
    assert caplog.record_tuples == []

    # restore original value
    taskmodule.log_first_n_examples = original_log_first_n_examples


@pytest.fixture()
def task_encodings(taskmodule, document):
    return taskmodule.encode(documents=[document], encode_target=True)


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
    return taskmodule.unbatch_output(targets["tgt_tokens"])


@pytest.fixture()
def task_outputs(unbatched_output):
    return unbatched_output


@pytest.fixture()
def task_output(task_outputs) -> EncodingWithIdsAndOptionalCpmTag:
    return task_outputs[0]


def test_task_output(task_output, taskmodule):
    output_list = task_output.tgt_tokens
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
                if ann.label in taskmodule.labels_per_layer[layer_name]
            }
            layer_expected = {
                str(ann)
                for ann in document[layer_name]
                if ann.label in taskmodule.labels_per_layer[layer_name]
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


def test_configure_metric(taskmodule):
    metric = taskmodule.configure_metric()
    assert metric is not None
    assert isinstance(metric, AnnotationLayerMetric)


def test_generation_kwargs(taskmodule):
    assert taskmodule.generation_kwargs == {
        "no_repeat_ngram_size": 7,
    }
