import dataclasses

import pytest
from pie_core import AnnotationLayer, Document, annotation_field

from pie_modules.annotations import Label, LabeledSpan
from pie_modules.documents import TextBasedDocument, TokenBasedDocument
from pie_modules.metrics import SpanLengthCollector


@pytest.fixture
def documents():
    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    result = []
    doc = TestDocument(text="First sentence. Entity M works at N. And it founded O.")
    doc.entities.append(LabeledSpan(start=16, end=24, label="per"))
    assert str(doc.entities[0]) == "Entity M"
    doc.entities.append(LabeledSpan(start=34, end=35, label="org"))
    assert str(doc.entities[1]) == "N"
    doc.entities.append(LabeledSpan(start=41, end=43, label="per"))
    assert str(doc.entities[2]) == "it"
    doc.entities.append(LabeledSpan(start=52, end=53, label="org"))
    assert str(doc.entities[3]) == "O"

    result.append(doc)

    # construct another document, but with longer entities
    doc = TestDocument(
        text="“Making a super tasty alt-chicken wing is only half of it,” said Po Bronson, general partner at "
        "SOSV and managing director of IndieBio."
    )
    doc.entities.append(LabeledSpan(start=65, end=75, label="per"))
    assert str(doc.entities[0]) == "Po Bronson"
    doc.entities.append(LabeledSpan(start=96, end=100, label="org"))
    assert str(doc.entities[1]) == "SOSV"
    doc.entities.append(LabeledSpan(start=126, end=134, label="org"))
    assert str(doc.entities[2]) == "IndieBio"

    result.append(doc)

    return result


def test_documents(documents):
    pass


def test_span_length_collector(documents):
    statistic = SpanLengthCollector(layer="entities")
    values = statistic(documents)
    assert statistic._values == [[8, 1, 2, 1], [10, 4, 8]]
    assert values == {
        "len": 7,
        "max": 10,
        "mean": 4.857142857142857,
        "min": 1,
        "std": 3.481730744843983,
    }

    statistic = SpanLengthCollector(layer="entities", labels="INFERRED")
    values = statistic(documents)
    assert [dict(v) for v in statistic._values] == [
        {"per": [8, 2], "org": [1, 1]},
        {"per": [10], "org": [4, 8]},
    ]
    assert values == {
        "org": {"len": 4, "max": 8, "mean": 3.5, "min": 1, "std": 2.8722813232690143},
        "per": {"len": 3, "max": 10, "mean": 6.666666666666667, "min": 2, "std": 3.39934634239519},
    }


def test_span_length_collector_wrong_label_value():
    with pytest.raises(ValueError) as excinfo:
        SpanLengthCollector(layer="entities", labels="WRONG")
    assert str(excinfo.value) == "labels must be a list of strings or 'INFERRED'"


def test_span_length_collector_with_tokenize(documents):
    @dataclasses.dataclass
    class TokenizedTestDocument(TokenBasedDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")

    statistic = SpanLengthCollector(
        layer="entities",
        tokenize=True,
        tokenizer="bert-base-uncased",
        tokenized_document_type=TokenizedTestDocument,
    )
    values = statistic(documents)
    assert values == {
        "len": 7,
        "max": 3,
        "mean": 1.8571428571428572,
        "min": 1,
        "std": 0.8329931278350429,
    }


def test_span_length_collector_with_tokenize_missing_tokenizer():
    with pytest.raises(ValueError) as excinfo:
        SpanLengthCollector(
            layer="entities",
            tokenize=True,
            tokenized_document_type=TokenBasedDocument,
        )
    assert (
        str(excinfo.value)
        == "tokenizer must be provided to calculate the span length in means of tokens"
    )


def test_span_length_collector_with_tokenize_missing_tokenized_document_type():
    with pytest.raises(ValueError) as excinfo:
        SpanLengthCollector(
            layer="entities",
            tokenize=True,
            tokenizer="bert-base-uncased",
        )
    assert (
        str(excinfo.value)
        == "tokenized_document_type must be provided to calculate the span length in means of tokens"
    )


def test_span_length_collector_with_tokenize_wrong_document_type():
    @dataclasses.dataclass
    class TestDocument(Document):
        data: str
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="data")

    doc = TestDocument(data="First sentence. Entity M works at N. And it founded O.")
    doc.entities.append(LabeledSpan(start=16, end=24, label="per"))
    assert str(doc.entities[0]) == "Entity M"

    @dataclasses.dataclass
    class TokenizedTestDocument(TokenBasedDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")

    statistic = SpanLengthCollector(
        layer="entities",
        tokenize=True,
        tokenizer="bert-base-uncased",
        tokenized_document_type=TokenizedTestDocument,
    )

    with pytest.raises(ValueError) as excinfo:
        statistic(doc)
    assert (
        str(excinfo.value)
        == "doc must be a TextBasedDocument to calculate the span length in means of tokens"
    )


def test_span_length_collector_with_tokenize_wrong_annotation_type():
    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        label: AnnotationLayer[Label] = annotation_field()

    doc = TestDocument(text="First sentence. Entity M works at N. And it founded O.")
    doc.label.append(Label(label="example"))

    statistic = SpanLengthCollector(layer="label")

    with pytest.raises(TypeError) as excinfo:
        statistic(doc)
    assert (
        str(excinfo.value)
        == "span length calculation is not yet supported for <class 'pytorch_ie.annotations.Label'>"
    )
