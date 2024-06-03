import dataclasses
import logging

import pytest
import torch
from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextBasedDocument
from torch import tensor

from pie_modules.taskmodules import RESpanPairClassificationTaskModule
from pie_modules.utils.span import distance
from tests import _config_to_str

TOKENIZER_NAME_OR_PATH = "bert-base-cased"

CONFIGS = [{}, {"partition_annotation": "sentences"}]
CONFIGS_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIGS_DICT.keys())
def cfg(request):
    return CONFIGS_DICT[request.param]


@pytest.fixture(scope="module")
def unprepared_taskmodule(cfg):
    taskmodule = RESpanPairClassificationTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH,
        log_first_n_examples=10,
        collect_statistics=True,
        **cfg,
    )
    assert not taskmodule.is_from_pretrained

    return taskmodule


@dataclasses.dataclass
class FixedTestDocument(TextBasedDocument):
    sentences: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@pytest.fixture(scope="module")
def fixed_documents(documents):
    result = []
    for document in documents:
        fixed_doc = document.copy(with_annotations=False).as_type(FixedTestDocument)
        for sentence in document.sentences:
            fixed_doc.sentences.append(
                LabeledSpan(start=sentence.start, end=sentence.end, label="sentence")
            )
        entity_mapping = {}
        for entity in document.entities:
            new_entity = entity.copy()
            fixed_doc.entities.append(new_entity)
            entity_mapping[entity] = new_entity
        for relation in document.relations:
            new_relation = relation.copy(
                head=entity_mapping[relation.head], tail=entity_mapping[relation.tail]
            )
            fixed_doc.relations.append(new_relation)
        result.append(fixed_doc)
    return result


@pytest.fixture(scope="module")
def taskmodule(unprepared_taskmodule, fixed_documents) -> RESpanPairClassificationTaskModule:
    unprepared_taskmodule.prepare(fixed_documents)
    return unprepared_taskmodule


def test_taskmodule(taskmodule: RESpanPairClassificationTaskModule):
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

    # because this is not the standard value for relation_annotation, we can not determine the document type
    assert taskmodule.document_type is None


@pytest.fixture(scope="module")
def document(fixed_documents):
    result = fixed_documents[4]
    assert (
        result.metadata["description"]
        == "sentences with multiple relation annotations and cross-sentence relation"
    )
    return result


def test_create_candidate_relations(taskmodule, document):
    # _create_candidate_relations requires normalized documents
    normalized_document = taskmodule.normalize_document(document)
    candidate_relations = taskmodule._create_candidate_relations(normalized_document)
    resolved_relations = [ann.resolve() for ann in candidate_relations]
    assert resolved_relations == [
        ("no_relation", (("PER", "Entity G"), ("ORG", "H"))),
        ("no_relation", (("PER", "Entity G"), ("ORG", "I"))),
        ("no_relation", (("ORG", "H"), ("PER", "Entity G"))),
        ("no_relation", (("ORG", "H"), ("ORG", "I"))),
        ("no_relation", (("ORG", "I"), ("PER", "Entity G"))),
        ("no_relation", (("ORG", "I"), ("ORG", "H"))),
    ]


def test_create_candidate_relations_with_max_distance(taskmodule, document):
    # _create_candidate_relations requires normalized documents
    normalized_document = taskmodule.normalize_document(document)
    candidate_relations = taskmodule._create_candidate_relations(
        normalized_document, max_argument_distance=10
    )
    resolved_relations = [ann.resolve() for ann in candidate_relations]
    assert resolved_relations == [
        ("no_relation", (("PER", "Entity G"), ("ORG", "H"))),
        ("no_relation", (("ORG", "H"), ("PER", "Entity G"))),
    ]
    distances = [
        distance(
            start_end=(rel.head.start, rel.head.end),
            other_start_end=(rel.tail.start, rel.tail.end),
            distance_type="inner",
        )
        for rel in candidate_relations
    ]
    assert distances == [10.0, 10.0]


@pytest.fixture(scope="module")
def task_encodings(taskmodule, document):
    result = taskmodule.encode(document, encode_target=True)
    return result


def test_encode_input(task_encodings, document, taskmodule, cfg):
    assert task_encodings is not None
    if cfg == {}:
        assert len(task_encodings) == 1
        inputs = task_encodings[0].inputs
        assert set(inputs) == {
            "input_ids",
            "attention_mask",
            "span_start_indices",
            "span_end_indices",
            "tuple_indices",
            "tuple_indices_mask",
        }
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
        tuple_spans = [
            [span_tokens[idx] for idx in indices] for indices in inputs["tuple_indices"]
        ]
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
        assert inputs["tuple_indices_mask"].tolist() == [True, True, True]
    elif cfg == {"partition_annotation": "sentences"}:
        assert len(task_encodings) == 1
        for idx, encoding in enumerate(task_encodings):
            inputs = encoding.inputs
            assert set(inputs) == {
                "input_ids",
                "attention_mask",
                "span_start_indices",
                "span_end_indices",
                "tuple_indices",
                "tuple_indices_mask",
            }
            tokens = taskmodule.tokenizer.convert_ids_to_tokens(inputs["input_ids"])
            span_tokens = [
                tokens[start:end]
                for start, end in zip(inputs["span_start_indices"], inputs["span_end_indices"])
            ]
            tuple_spans = [
                [span_tokens[idx] for idx in indices] for indices in inputs["tuple_indices"]
            ]
            if idx == 0:
                assert tokens == [
                    "[CLS]",
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
                    "[SEP]",
                ]
                assert span_tokens == [
                    ["[CLS]", "En", "##ti", "##ty", "G", "[/SPAN:PER]"],
                    ["[SPAN:ORG]", "H", "[/SPAN:ORG]"],
                ]
                assert tuple_spans == [
                    [
                        ["[CLS]", "En", "##ti", "##ty", "G", "[/SPAN:PER]"],
                        ["[SPAN:ORG]", "H", "[/SPAN:ORG]"],
                    ]
                ]
                assert inputs["tuple_indices_mask"].tolist() == [True]
            else:
                raise ValueError(f"unexpected idx: {idx}")
    else:
        raise ValueError(f"unexpected config: {cfg}")


def test_encode_target(taskmodule, task_encodings, cfg):
    if cfg == {}:
        assert len(task_encodings) == 1
        targets = task_encodings[0].targets
        labels = [taskmodule.id_to_label[label] for label in targets["labels"].tolist()]
        assert labels == ["per:employee_of", "per:founder", "org:founded_by"]
    elif cfg == {"partition_annotation": "sentences"}:
        assert len(task_encodings) == 1
        for idx, encoding in enumerate(task_encodings):
            targets = encoding.targets
            labels = [taskmodule.id_to_label[label] for label in targets["labels"].tolist()]
            if idx == 0:
                assert labels == ["per:employee_of"]
            else:
                raise ValueError(f"unexpected idx: {idx}")
    else:
        raise ValueError(f"unexpected config: {cfg}")


def test_encode_with_no_gold_relation(document):
    # create a new taskmodule that does create candidate relations
    taskmodule = RESpanPairClassificationTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH,
        create_candidate_relations=True,
        labels=["org:founded_by", "per:employee_of", "per:founder"],
        entity_labels=["ORG", "PER"],
    )
    taskmodule.post_prepare()
    # create a new document that has no relations
    document = document.copy()
    document.relations.clear()

    encodings = taskmodule.encode(document, encode_target=True)

    assert len(encodings) == 1
    encoding = encodings[0]
    # same number of candidate relations as there are labels
    assert len(encoding.metadata["candidate_relations"]) == encoding.targets["labels"].numel()
    assert all(rel.label == "no_relation" for rel in encoding.metadata["candidate_relations"])
    assert encoding.targets["labels"].tolist() == [0, 0, 0, 0, 0, 0]


def test_encode_with_multiple_gold_relations_with_same_arguments(document, caplog):
    # create a new taskmodule that does create candidate relations
    taskmodule = RESpanPairClassificationTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH,
        labels=["org:founded_by", "per:employee_of", "per:founder"],
        entity_labels=["ORG", "PER"],
    )
    taskmodule.post_prepare()
    # create a new document that has multiple relations with the same arguments
    document = document.copy()
    document.relations.clear()
    head = document.entities[0]
    tail = document.entities[1]
    document.relations.extend(
        [
            BinaryRelation(head=head, tail=tail, label="org:founded_by"),
            BinaryRelation(head=head, tail=tail, label="per:employee_of"),
        ]
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        encodings = taskmodule.encode(document, encode_target=True)
    assert len(caplog.messages) == 2
    assert (
        caplog.messages[0]
        == "skip the candidate relation because there are more than one gold relation for "
        "its args and roles: [BinaryRelation(head=LabeledSpan(start=5, end=10, label='PER', score=1.0), "
        "tail=LabeledSpan(start=13, end=15, label='ORG', score=1.0), label='org:founded_by', score=1.0), "
        "BinaryRelation(head=LabeledSpan(start=5, end=10, label='PER', score=1.0), "
        "tail=LabeledSpan(start=13, end=15, label='ORG', score=1.0), label='per:employee_of', score=1.0)]"
    )
    assert (
        caplog.messages[1]
        == "skip the candidate relation because there are more than one gold relation for "
        "its args and roles: [BinaryRelation(head=LabeledSpan(start=5, end=10, label='PER', score=1.0), "
        "tail=LabeledSpan(start=13, end=15, label='ORG', score=1.0), label='org:founded_by', score=1.0), "
        "BinaryRelation(head=LabeledSpan(start=5, end=10, label='PER', score=1.0), "
        "tail=LabeledSpan(start=13, end=15, label='ORG', score=1.0), label='per:employee_of', score=1.0)]"
    )

    assert len(encodings) == 1
    encoding = encodings[0]
    candidate_relations = encoding.metadata["candidate_relations"]
    # same number of candidate relations as there are labels
    assert len(candidate_relations) == encoding.targets["labels"].numel()
    assert candidate_relations[0].label == "org:founded_by"
    assert candidate_relations[1].label == "per:employee_of"
    assert encoding.targets["labels"].tolist() == [-100, -100]


def test_maybe_log_example(taskmodule, task_encodings, caplog, cfg):
    caplog.clear()
    if cfg == {}:
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
    elif cfg == {"partition_annotation": "sentences"}:
        with caplog.at_level(logging.INFO):
            taskmodule._maybe_log_example(task_encodings[0], target=task_encodings[0].targets)
        assert caplog.messages == [
            "*** Example ***",
            "doc id: train_doc5",
            "tokens: [CLS] En ##ti ##ty G [/SPAN:PER] works at [SPAN:ORG] H [/SPAN:ORG] . [SEP]",
            "input_ids: 101 13832 3121 2340 144 28998 1759 1120 28999 145 28997 119 102",
            "relation 0: per:employee_of",
            "\targ 0: [CLS] En ##ti ##ty G [/SPAN:PER]",
            "\targ 1: [SPAN:ORG] H [/SPAN:ORG]",
        ]
    else:
        raise ValueError(f"unexpected config: {cfg}")


def test_encode_with_statistics(taskmodule, fixed_documents, cfg, caplog):
    caplog.clear()
    with caplog.at_level(logging.INFO):
        taskmodule.encode(fixed_documents, encode_target=True)
    assert len(caplog.messages) > 0
    statistics = caplog.messages[-1]
    if cfg == {}:
        assert (
            statistics
            == """statistics:
|                     |   org:founded_by |   per:employee_of |   per:founder |
|:--------------------|-----------------:|------------------:|--------------:|
| available           |                2 |                 3 |             2 |
| available_tokenized |                2 |                 3 |             2 |
| used                |                2 |                 3 |             2 |"""
        )
    elif cfg == {"partition_annotation": "sentences"}:
        assert (
            statistics
            == """statistics:
|                     |   org:founded_by |   per:employee_of |   per:founder |
|:--------------------|-----------------:|------------------:|--------------:|
| available           |                2 |                 3 |             2 |
| available_tokenized |                1 |                 3 |             1 |
| used                |                1 |                 3 |             1 |"""
        )
    else:
        raise ValueError(f"unexpected config: {cfg}")


def test_collate(taskmodule, task_encodings, cfg):
    result = taskmodule.collate(task_encodings)
    assert result is not None
    inputs, targets = result
    assert set(inputs) == {
        "input_ids",
        "attention_mask",
        "span_start_indices",
        "span_end_indices",
        "tuple_indices",
        "tuple_indices_mask",
    }
    if cfg == {}:
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
        torch.testing.assert_close(inputs["tuple_indices_mask"], tensor([[True, True, True]]))
        assert set(targets) == {"labels"}
        torch.testing.assert_close(targets["labels"], tensor([[2, 3, 1]]))
    elif cfg == {"partition_annotation": "sentences"}:
        torch.testing.assert_close(
            inputs["input_ids"],
            tensor(
                [[101, 13832, 3121, 2340, 144, 28998, 1759, 1120, 28999, 145, 28997, 119, 102]]
            ),
        )
        torch.testing.assert_close(
            inputs["attention_mask"], tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        )
        torch.testing.assert_close(inputs["span_start_indices"], tensor([[0, 8]]))
        torch.testing.assert_close(inputs["span_end_indices"], tensor([[6, 11]]))
        torch.testing.assert_close(inputs["tuple_indices"], tensor([[[0, 1]]]))
        torch.testing.assert_close(inputs["tuple_indices_mask"], tensor([[True]]))
        assert set(targets) == {"labels"}
        torch.testing.assert_close(targets["labels"], tensor([[2]]))
    else:
        raise ValueError(f"unexpected config: {cfg}")


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


def test_configure_model_metrics(taskmodule, model_output):
    metrics = taskmodule.configure_model_metric(stage="train")
    assert metrics is not None
    metric_values = metrics(model_output, model_output)
    metric_values_converted = {key: value.item() for key, value in metric_values.items()}
    assert metric_values_converted == {
        "macro/f1": 1.0,
        "micro/f1": 1.0,
        "no_relation/f1": 0.0,
        "org:founded_by/f1": 1.0,
        "per:employee_of/f1": 1.0,
        "per:founder/f1": 1.0,
    }
