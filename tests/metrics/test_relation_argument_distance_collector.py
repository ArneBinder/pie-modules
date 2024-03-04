import dataclasses

import pytest
from pytorch_ie import Annotation, Document
from pytorch_ie.annotations import (
    BinaryRelation,
    LabeledSpan,
    MultiLabeledBinaryRelation,
)
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument

from pie_modules.annotations import LabeledMultiSpan
from pie_modules.metrics import RelationArgumentDistanceCollector


@dataclasses.dataclass
class TestDocument(TextBasedDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


def test_relation_argument_distance_collector():
    doc = TestDocument(
        text="This is the first entity. This is the second entity. " "This is the third entity."
    )
    doc.entities.append(LabeledSpan(start=0, end=25, label="entity"))
    doc.entities.append(LabeledSpan(start=26, end=52, label="entity"))
    doc.entities.append(LabeledSpan(start=53, end=78, label="entity"))
    doc.relations.append(
        BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="relation_label_1")
    )
    doc.relations.append(
        BinaryRelation(head=doc.entities[1], tail=doc.entities[2], label="relation_label_2")
    )

    statistic = RelationArgumentDistanceCollector(layer="relations")
    values = statistic(doc)
    assert values == {
        "ALL": {"len": 4, "mean": 52.0, "std": 0.0, "min": 52.0, "max": 52.0},
        "relation_label_1": {"len": 2, "mean": 52.0, "std": 0.0, "min": 52.0, "max": 52.0},
        "relation_label_2": {"len": 2, "mean": 52.0, "std": 0.0, "min": 52.0, "max": 52.0},
    }


def test_relation_argument_distance_collector_with_multi_span():
    doc = TestDocument(
        text="This is the first entity. This is the second entity. "
        "This is a filler. This is the third entity."
    )
    doc.entities.append(LabeledMultiSpan(slices=((0, 25),), label="entity"))
    doc.entities.append(LabeledMultiSpan(slices=((26, 52),), label="entity"))
    doc.entities.append(LabeledMultiSpan(slices=((53, 78),), label="entity"))
    doc.relations.append(
        MultiLabeledBinaryRelation(
            head=doc.entities[0], tail=doc.entities[1], label="relation_label_1"
        )
    )
    doc.relations.append(
        MultiLabeledBinaryRelation(
            head=doc.entities[1], tail=doc.entities[2], label="relation_label_2"
        )
    )

    # TypeError
    # argument distance calculation is not yet supported for <class 'pytorch_ie.annotations.MultiLabeledBinaryRelation'>
    statistic = RelationArgumentDistanceCollector(layer="relations")
    values = statistic(doc)
    assert values == {
        "ALL": {"len": 4, "mean": 52, "std": 0.0, "min": 52.0, "max": 52.0},
        "relation_label_1": {"len": 2, "mean": 52.0, "std": 0.0, "min": 52.0, "max": 52.0},
        "relation_label_2": {"len": 2, "mean": 52.0, "std": 0.0, "min": 52.0, "max": 52.0},
    }


def test_relation_argument_distance_collector_with_tokenize():
    doc = TestDocument(
        text="This is the first entity. This is the second entity. " "This is the third entity."
    )

    doc.entities.append(LabeledSpan(start=0, end=25, label="entity"))
    doc.entities.append(LabeledSpan(start=26, end=52, label="entity"))
    doc.entities.append(LabeledSpan(start=53, end=78, label="entity"))
    doc.relations.append(
        BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="relation_label_1")
    )
    doc.relations.append(
        BinaryRelation(head=doc.entities[1], tail=doc.entities[2], label="relation_label_2")
    )

    @dataclasses.dataclass
    class TokenizedTestDocument(TokenBasedDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")

    statistic = RelationArgumentDistanceCollector(
        layer="relations",
        tokenize=True,
        tokenizer="bert-base-uncased",
        tokenized_document_type=TokenizedTestDocument,
    )
    values = statistic(doc)
    assert values == {
        "ALL": {"len": 4, "mean": 12.0, "std": 0.0, "min": 12.0, "max": 12.0},
        "relation_label_1": {"len": 2, "mean": 12.0, "std": 0.0, "min": 12.0, "max": 12.0},
        "relation_label_2": {"len": 2, "mean": 12.0, "std": 0.0, "min": 12.0, "max": 12.0},
    }


def test_relation_argument_distance_collector_with_tokenize_missing_tokenizer():
    with pytest.raises(ValueError) as excinfo:
        RelationArgumentDistanceCollector(
            layer="relations",
            tokenize=True,
            tokenized_document_type=TokenBasedDocument,
        )
    assert (
        str(excinfo.value) == "tokenizer must be provided to calculate distance in means of tokens"
    )


def test_relation_argument_distance_collector_with_tokenize_missing_tokenized_document_type():
    with pytest.raises(ValueError) as excinfo:
        RelationArgumentDistanceCollector(
            layer="relations",
            tokenize=True,
            tokenizer="bert-base-uncased",
        )
    assert (
        str(excinfo.value)
        == "tokenized_document_type must be provided to calculate distance in means of tokens"
    )


def test_relation_argument_distance_collector_with_tokenize_wrong_document_type():
    @dataclasses.dataclass
    class TestDocument(Document):
        data: str
        entities: AnnotationList[LabeledSpan] = annotation_field(target="data")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")

    doc = TestDocument(
        data="This is the first entity. This is the second entity. " "This is the third entity."
    )

    doc.entities.append(LabeledSpan(start=0, end=25, label="entity"))
    doc.entities.append(LabeledSpan(start=26, end=52, label="entity"))
    doc.entities.append(LabeledSpan(start=53, end=78, label="entity"))
    doc.relations.append(
        BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="relation_label_1")
    )
    doc.relations.append(
        BinaryRelation(head=doc.entities[1], tail=doc.entities[2], label="relation_label_2")
    )

    @dataclasses.dataclass
    class TokenizedTestDocument(TokenBasedDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")

    #
    statistic = RelationArgumentDistanceCollector(
        layer="relations",
        tokenize=True,
        tokenizer="bert-base-uncased",
        tokenized_document_type=TokenizedTestDocument,
    )

    with pytest.raises(ValueError) as excinfo:
        statistic(doc)
    assert (
        str(excinfo.value)
        == "doc must be a TextBasedDocument to calculate distance in means of tokens"
    )


def test_relation_argument_distance_collector_with_tokenize_wrong_span_annotation_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class UnknownSpan(Annotation):
        start: int
        end: int

    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        labeled_multi_spans: AnnotationList[UnknownSpan] = annotation_field(target="text")
        binary_relations: AnnotationList[BinaryRelation] = annotation_field(
            target="labeled_multi_spans"
        )

    doc = TestDocument(text="First sentence. Entity M works at N. And it founded O.")
    doc.labeled_multi_spans.append(UnknownSpan(start=0, end=15))
    doc.labeled_multi_spans.append(UnknownSpan(start=16, end=24))
    doc.binary_relations.append(
        BinaryRelation(
            head=doc.labeled_multi_spans[0],
            tail=doc.labeled_multi_spans[1],
            label="relation_label_1",
        )
    )

    statistic = RelationArgumentDistanceCollector(layer="binary_relations")

    with pytest.raises(TypeError) as excinfo:
        statistic(doc)
    assert (
        str(excinfo.value)
        == "argument distance calculation is not yet supported for arguments other than Spans"
    )


def test_relation_argument_distance_collector_with_tokenize_wrong_relation_annotation_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class UnknownRelation(Annotation):
        head: Annotation
        tail: Annotation

    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        labeled_spans: AnnotationList[LabeledSpan] = annotation_field(target="text")
        not_binary_relations: AnnotationList[UnknownRelation] = annotation_field(
            target="labeled_spans"
        )

    doc = TestDocument(text="This is the first entity. This is the second entity.")
    doc.labeled_spans.append(LabeledSpan(start=0, end=25, label="entity"))
    doc.labeled_spans.append(LabeledSpan(start=26, end=52, label="entity"))
    doc.not_binary_relations.append(
        UnknownRelation(head=doc.labeled_spans[0], tail=doc.labeled_spans[1])
    )

    statistic = RelationArgumentDistanceCollector(layer="not_binary_relations")

    with pytest.raises(TypeError) as excinfo:
        statistic(doc)
    assert (
        str(excinfo.value)
        == f"argument distance calculation is not yet supported for {UnknownRelation}"
    )
