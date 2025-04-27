import dataclasses

import pytest
from pie_core import Annotation, AnnotationLayer, Document, annotation_field

from pie_modules.annotations import LabeledMultiSpan, LabeledSpan
from pie_modules.documents import TextBasedDocument, TokenBasedDocument
from pie_modules.metrics import SpanCoverageCollector


@dataclasses.dataclass
class TestDocument(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


def test_span_coverage_collector():
    doc = TestDocument(text="A and O.")
    doc.entities.append(LabeledSpan(start=0, end=1, label="entity"))
    doc.entities.append(LabeledSpan(start=6, end=7, label="entity"))

    statistic = SpanCoverageCollector(layer="entities")
    values = statistic(doc)
    assert values == {"len": 1, "max": 0.25, "mean": 0.25, "min": 0.25, "std": 0.0}


def test_span_coverage_collector_with_multi_span():
    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        entities: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="text")

    doc = TestDocument(text="A and O.")
    doc.entities.append(LabeledMultiSpan(slices=((0, 1),), label="entity"))
    doc.entities.append(LabeledMultiSpan(slices=((6, 7),), label="entity"))

    statistic = SpanCoverageCollector(layer="entities")
    values = statistic(doc)
    assert values == {
        "len": 1,
        "max": 0.25,
        "mean": 0.25,
        "min": 0.25,
        "std": 0.0,
    }


def test_span_coverage_collector_with_labels():
    doc = TestDocument(text="A and O.")
    doc.entities.append(LabeledSpan(start=0, end=1, label="entity"))
    doc.entities.append(LabeledSpan(start=6, end=7, label="no_entity"))

    statistic = SpanCoverageCollector(layer="entities", labels=["entity"])
    values = statistic(doc)
    assert values == {"len": 1, "max": 0.125, "mean": 0.125, "min": 0.125, "std": 0.0}


def test_span_coverage_collector_with_tokenize():
    doc = TestDocument(text="A and O.")
    doc.entities.append(LabeledSpan(start=0, end=1, label="entity"))
    doc.entities.append(LabeledSpan(start=6, end=7, label="entity"))

    @dataclasses.dataclass
    class TokenizedTestDocument(TokenBasedDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")

    statistic = SpanCoverageCollector(
        layer="entities",
        tokenize=True,
        tokenizer="bert-base-uncased",
        tokenized_document_type=TokenizedTestDocument,
    )
    values = statistic(doc)
    assert values == {
        "len": 1,
        "max": 0.3333333333333333,
        "mean": 0.3333333333333333,
        "min": 0.3333333333333333,
        "std": 0.0,
    }


def test_span_coverage_collector_with_tokenize_missing_tokenizer():
    with pytest.raises(ValueError) as excinfo:
        SpanCoverageCollector(
            layer="entities",
            tokenize=True,
            tokenized_document_type=TokenBasedDocument,
        )
    assert (
        str(excinfo.value)
        == "tokenizer must be provided to calculate the span coverage in means of tokens"
    )


def test_span_coverage_collector_with_tokenize_missing_tokenized_document_type():
    with pytest.raises(ValueError) as excinfo:
        SpanCoverageCollector(
            layer="entities",
            tokenize=True,
            tokenizer="bert-base-uncased",
        )
    assert (
        str(excinfo.value)
        == "tokenized_document_type must be provided to calculate the span coverage in means of tokens"
    )


def test_span_coverage_collector_with_tokenize_wrong_document_type():
    @dataclasses.dataclass
    class TestDocument(Document):
        data: str
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="data")

    doc = TestDocument(data="A and O")

    @dataclasses.dataclass
    class TokenizedTestDocument(TokenBasedDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")

    statistic = SpanCoverageCollector(
        layer="entities",
        tokenize=True,
        tokenizer="bert-base-uncased",
        tokenized_document_type=TokenizedTestDocument,
    )

    with pytest.raises(ValueError) as excinfo:
        statistic(doc)
    assert (
        str(excinfo.value)
        == "doc must be a TextBasedDocument to calculate the span coverage in means of tokens"
    )


def test_span_coverage_collector_with_tokenize_wrong_annotation_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class UnknownSpan(Annotation):
        start: int
        end: int

    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        labeled_multi_spans: AnnotationLayer[UnknownSpan] = annotation_field(target="text")

    doc = TestDocument(text="First sentence. Entity M works at N. And it founded O.")
    doc.labeled_multi_spans.append(UnknownSpan(start=16, end=24))

    statistic = SpanCoverageCollector(layer="labeled_multi_spans")

    with pytest.raises(TypeError) as excinfo:
        statistic(doc)
    assert (
        str(excinfo.value) == f"span coverage calculation is not yet supported for {UnknownSpan}"
    )
