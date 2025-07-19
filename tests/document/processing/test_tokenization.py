import dataclasses
from collections import defaultdict
from typing import Dict

import pytest
from pie_core import Annotation, AnnotationLayer, Document, annotation_field

from pie_modules.annotations import (
    BinaryRelation,
    Label,
    LabeledMultiSpan,
    LabeledSpan,
    Span,
)
from pie_modules.document.processing import (
    text_based_document_to_token_based,
    token_based_document_to_text_based,
)
from pie_modules.document.processing.tokenization import find_token_offset_mapping
from pie_modules.documents import TextBasedDocument, TokenBasedDocument
from tests.conftest import TestDocument


@dataclasses.dataclass
class TokenizedTestDocument(TokenBasedDocument):
    sentences: AnnotationLayer[Span] = annotation_field(target="tokens")
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TestDocumentWithMultiSpans(TextBasedDocument):
    entities: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TokenizedTestDocumentWithMultiSpans(TokenBasedDocument):
    entities: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="tokens")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@pytest.fixture
def text_document():
    doc = TestDocument(text="First sentence. Entity M works at N. And it founded O.")
    doc.sentences.extend([Span(start=0, end=15), Span(start=16, end=36), Span(start=37, end=54)])
    doc.entities.extend(
        [
            LabeledSpan(start=16, end=24, label="per"),
            LabeledSpan(start=34, end=35, label="org"),
            LabeledSpan(start=41, end=43, label="per"),
            LabeledSpan(start=52, end=53, label="org"),
        ]
    )
    doc.relations.extend(
        [
            BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="per:employee_of"),
            BinaryRelation(head=doc.entities[2], tail=doc.entities[3], label="per:founder"),
        ]
    )
    _test_text_document(doc)
    return doc


def _test_text_document(doc):
    assert str(doc.sentences[0]) == "First sentence."
    assert str(doc.sentences[1]) == "Entity M works at N."
    assert str(doc.sentences[2]) == "And it founded O."

    assert str(doc.entities[0]) == "Entity M"
    assert str(doc.entities[1]) == "N"
    assert str(doc.entities[2]) == "it"
    assert str(doc.entities[3]) == "O"

    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in doc.relations]
    assert relation_tuples == [("Entity M", "per:employee_of", "N"), ("it", "per:founder", "O")]


@pytest.fixture
def text_document_with_multi_spans():
    doc = TestDocumentWithMultiSpans(text="First sentence. Entity M works at N. And it founded O.")
    doc.entities.extend(
        [
            LabeledMultiSpan(slices=((16, 22), (23, 24)), label="per"),
            LabeledMultiSpan(slices=((34, 35),), label="org"),
            LabeledMultiSpan(slices=((41, 43),), label="per"),
            LabeledMultiSpan(slices=((52, 53),), label="org"),
        ]
    )
    doc.relations.extend(
        [
            BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="per:employee_of"),
            BinaryRelation(head=doc.entities[2], tail=doc.entities[3], label="per:founder"),
        ]
    )
    _test_text_document_with_multi_spans(doc)
    return doc


def _test_text_document_with_multi_spans(doc):
    assert str(doc.entities[0]) == "('Entity', 'M')"
    assert str(doc.entities[1]) == "('N',)"
    assert str(doc.entities[2]) == "('it',)"
    assert str(doc.entities[3]) == "('O',)"

    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in doc.relations]
    assert relation_tuples == [
        ("('Entity', 'M')", "per:employee_of", "('N',)"),
        ("('it',)", "per:founder", "('O',)"),
    ]


@pytest.fixture
def token_document():
    doc = TokenizedTestDocument(
        tokens=(
            "[CLS]",
            "First",
            "sentence",
            ".",
            "Entity",
            "M",
            "works",
            "at",
            "N",
            ".",
            "And",
            "it",
            "founded",
            "O",
            ".",
            "[SEP]",
        ),
    )
    doc.sentences.extend(
        [
            Span(start=1, end=4),
            Span(start=4, end=10),
            Span(start=10, end=15),
        ]
    )
    doc.entities.extend(
        [
            LabeledSpan(start=4, end=6, label="per"),
            LabeledSpan(start=8, end=9, label="org"),
            LabeledSpan(start=11, end=12, label="per"),
            LabeledSpan(start=13, end=14, label="org"),
        ]
    )
    doc.relations.extend(
        [
            BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="per:employee_of"),
            BinaryRelation(head=doc.entities[2], tail=doc.entities[3], label="per:founder"),
        ]
    )
    _test_token_document(doc)
    return doc


def _test_token_document(doc):
    assert str(doc.sentences[0]) == "('First', 'sentence', '.')"
    assert str(doc.sentences[1]) == "('Entity', 'M', 'works', 'at', 'N', '.')"
    assert str(doc.sentences[2]) == "('And', 'it', 'founded', 'O', '.')"

    assert str(doc.entities[0]) == "('Entity', 'M')"
    assert str(doc.entities[1]) == "('N',)"
    assert str(doc.entities[2]) == "('it',)"
    assert str(doc.entities[3]) == "('O',)"

    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in doc.relations]
    assert relation_tuples == [
        ("('Entity', 'M')", "per:employee_of", "('N',)"),
        ("('it',)", "per:founder", "('O',)"),
    ]


@pytest.fixture
def token_document_with_multi_spans():
    doc = TokenizedTestDocumentWithMultiSpans(
        tokens=(
            "[CLS]",
            "First",
            "sentence",
            ".",
            "Entity",
            "M",
            "works",
            "at",
            "N",
            ".",
            "And",
            "it",
            "founded",
            "O",
            ".",
            "[SEP]",
        ),
    )
    doc.entities.extend(
        [
            LabeledMultiSpan(slices=((4, 5), (5, 6)), label="per"),
            LabeledMultiSpan(slices=((8, 9),), label="org"),
            LabeledMultiSpan(slices=((11, 12),), label="per"),
            LabeledMultiSpan(slices=((13, 14),), label="org"),
        ]
    )
    doc.relations.extend(
        [
            BinaryRelation(head=doc.entities[0], tail=doc.entities[1], label="per:employee_of"),
            BinaryRelation(head=doc.entities[2], tail=doc.entities[3], label="per:founder"),
        ]
    )
    _test_token_document_with_multi_spans(doc)
    return doc


def _test_token_document_with_multi_spans(doc):
    assert str(doc.entities[0]) == "(('Entity',), ('M',))"
    assert str(doc.entities[1]) == "(('N',),)"
    assert str(doc.entities[2]) == "(('it',),)"
    assert str(doc.entities[3]) == "(('O',),)"

    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in doc.relations]
    assert relation_tuples == [
        ("(('Entity',), ('M',))", "per:employee_of", "(('N',),)"),
        ("(('it',),)", "per:founder", "(('O',),)"),
    ]


def test_find_token_offset_mapping(text_document, token_document):
    token_offset_mapping = find_token_offset_mapping(
        text=text_document.text, tokens=list(token_document.tokens)
    )
    assert token_offset_mapping == [
        (0, 0),
        (0, 5),
        (6, 14),
        (14, 15),
        (16, 22),
        (23, 24),
        (25, 30),
        (31, 33),
        (34, 35),
        (35, 36),
        (37, 40),
        (41, 43),
        (44, 51),
        (52, 53),
        (53, 54),
        (54, 54),
    ]


def _assert_added_annotations(
    document: Document,
    converted_document: Document,
    added_annotations: Dict[str, Dict[Annotation, Annotation]],
):
    for ann_field in document.annotation_fields():
        layer_name = ann_field.name
        text_annotations = document[layer_name]
        token_annotations = converted_document[layer_name]
        expected_mapping = dict(zip(text_annotations, token_annotations))
        assert len(expected_mapping) > 0
        assert added_annotations[layer_name] == expected_mapping


def test_text_based_document_to_token_based(text_document, token_document):
    added_annotations = defaultdict(dict)
    doc = text_based_document_to_token_based(
        text_document,
        tokens=list(token_document.tokens),
        result_document_type=TokenizedTestDocument,
        added_annotations=added_annotations,
    )
    _assert_added_annotations(text_document, doc, added_annotations)
    _test_token_document(doc)


def test_text_based_document_to_token_based_multi_span(
    text_document_with_multi_spans, token_document_with_multi_spans
):
    added_annotations = defaultdict(dict)
    doc = text_based_document_to_token_based(
        text_document_with_multi_spans,
        tokens=list(token_document_with_multi_spans.tokens),
        result_document_type=TokenizedTestDocumentWithMultiSpans,
        added_annotations=added_annotations,
    )
    _assert_added_annotations(text_document_with_multi_spans, doc, added_annotations)
    _test_token_document_with_multi_spans(doc)


def test_text_based_document_to_token_based_tokens_from_metadata(text_document, token_document):
    doc = text_document.copy()
    doc.metadata["tokens"] = list(token_document.tokens)
    result = text_based_document_to_token_based(
        doc,
        result_document_type=TokenizedTestDocument,
    )
    _test_token_document(result)


def test_text_based_document_to_token_based_missing_tokens_and_token_offset_mapping(
    text_document, token_document
):
    with pytest.raises(ValueError) as excinfo:
        text_based_document_to_token_based(
            text_document,
            result_document_type=TokenizedTestDocument,
        )
    assert (
        str(excinfo.value)
        == "tokens or token_offset_mapping must be provided to convert a text based document to "
        "token based, but got None for both"
    )


def test_text_based_document_to_token_based_tokens_from_metadata_are_different(
    text_document, token_document, caplog
):
    doc = text_document.copy()
    doc.metadata["tokens"] = list(token_document.tokens) + ["[PAD]"]
    with caplog.at_level("WARNING"):
        result = text_based_document_to_token_based(
            doc,
            tokens=list(token_document.tokens),
            result_document_type=TokenizedTestDocument,
        )
    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message
        == "tokens in metadata are different from new tokens, take the new tokens"
    )
    _test_token_document(result)


def test_text_based_document_to_token_based_offset_mapping_from_metadata(
    text_document, token_document
):
    doc = text_document.copy()
    doc.metadata["token_offset_mapping"] = find_token_offset_mapping(
        text=doc.text, tokens=list(token_document.tokens)
    )
    result = text_based_document_to_token_based(
        doc,
        result_document_type=TokenizedTestDocument,
    )
    _test_token_document(result)


def test_text_based_document_to_token_based_token_offset_mapping_from_metadata_is_different(
    text_document, token_document, caplog
):
    doc = text_document.copy()
    doc.metadata["token_offset_mapping"] = []
    with caplog.at_level("WARNING"):
        result = text_based_document_to_token_based(
            doc,
            token_offset_mapping=find_token_offset_mapping(
                text=doc.text, tokens=list(token_document.tokens)
            ),
            result_document_type=TokenizedTestDocument,
        )
    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message
        == "token_offset_mapping in metadata is different from the new token_offset_mapping, overwrite the metadata"
    )
    _test_token_document(result)


def test_text_based_document_to_token_based_space_span_strict(text_document, token_document):
    doc = TestDocument(text=text_document.text)
    # add a span that is empty
    doc.entities.append(LabeledSpan(start=5, end=6, label="unaligned"))
    assert str(doc.entities[0]) == " "

    with pytest.raises(ValueError) as excinfo:
        text_based_document_to_token_based(
            doc,
            tokens=list(token_document.tokens),
            result_document_type=TokenizedTestDocument,
        )
    assert (
        str(excinfo.value)
        == 'cannot find token span for character span: " ", text="First sentence. Entity M works at N. '
        'And it founded O.", token_offset_mapping=[(0, 0), (0, 5), (6, 14), (14, 15), (16, 22), (23, 24), '
        "(25, 30), (31, 33), (34, 35), (35, 36), (37, 40), (41, 43), (44, 51), (52, 53), (53, 54), (54, 54)]"
    )


def test_text_based_document_to_token_based_empty_span_strict(text_document, token_document):
    doc = TestDocument(text=text_document.text)
    # add a span that is empty
    doc.entities.append(LabeledSpan(start=3, end=3, label="empty"))
    assert str(doc.entities[0]) == ""

    with pytest.raises(ValueError) as excinfo:
        text_based_document_to_token_based(
            doc,
            tokens=list(token_document.tokens),
            result_document_type=TokenizedTestDocument,
        )
    assert (
        str(excinfo.value)
        == 'cannot find token span for character span: "", text="First sentence. Entity M works at N. '
        'And it founded O.", token_offset_mapping=[(0, 0), (0, 5), (6, 14), (14, 15), (16, 22), (23, 24), '
        "(25, 30), (31, 33), (34, 35), (35, 36), (37, 40), (41, 43), (44, 51), (52, 53), (53, 54), (54, 54)]"
    )


def test_text_based_document_to_token_based_space_span(text_document, token_document, caplog):
    doc = TestDocument(text=text_document.text)
    # add a span that is empty
    doc.entities.append(LabeledSpan(start=5, end=6, label="unaligned"))
    assert str(doc.entities[0]) == " "

    with caplog.at_level("WARNING"):
        tokenized_doc = text_based_document_to_token_based(
            doc,
            tokens=list(token_document.tokens),
            result_document_type=TokenizedTestDocument,
            strict_span_conversion=False,
        )
    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message
        == 'cannot find token span for character span " ", skip it (disable this warning with verbose=False)'
    )

    # check (de-)serialization
    tokenized_doc.copy()

    assert len(doc.entities) == 1
    # the unaligned span is not included in the tokenized document
    assert len(tokenized_doc.entities) == 0


def test_text_based_document_to_token_based_strip_span(text_document, token_document):
    doc = TestDocument(text=text_document.text)
    # add a span that is not aligned with the tokenization
    doc.entities.append(LabeledSpan(start=5, end=16, label="unaligned"))
    assert str(doc.entities[0]) == " sentence. "

    result_doc = text_based_document_to_token_based(
        doc,
        tokens=list(token_document.tokens),
        result_document_type=TokenizedTestDocument,
        strip_spans=True,
    )
    assert result_doc.entities.resolve() == [("unaligned", ("sentence", "."))]


def test_text_based_document_to_token_based_empty_multi_span_strict(
    token_document_with_multi_spans, text_document_with_multi_spans
):
    doc = TestDocumentWithMultiSpans(text=text_document_with_multi_spans.text)
    # add a multi span that is not aligned with the tokenization
    doc.entities.append(LabeledMultiSpan(slices=(), label="empty"))
    assert doc.entities.resolve() == [("empty", ())]

    with pytest.raises(ValueError) as excinfo:
        text_based_document_to_token_based(
            doc,
            tokens=list(token_document_with_multi_spans.tokens),
            result_document_type=TokenizedTestDocumentWithMultiSpans,
        )
    assert (
        str(excinfo.value)
        == 'cannot find token span for character span: "()", text="First sentence. Entity M works at N. '
        'And it founded O.", token_offset_mapping=[(0, 0), (0, 5), (6, 14), (14, 15), (16, 22), (23, 24), '
        "(25, 30), (31, 33), (34, 35), (35, 36), (37, 40), (41, 43), (44, 51), (52, 53), (53, 54), (54, 54)]"
    )


def test_text_based_document_to_token_based_space_multi_span_slice_strict(
    token_document_with_multi_spans, text_document_with_multi_spans
):
    doc = TestDocumentWithMultiSpans(text=text_document_with_multi_spans.text)
    # add a multi span that is not aligned with the tokenization
    doc.entities.append(LabeledMultiSpan(slices=((5, 6),), label="unaligned"))
    assert doc.entities.resolve() == [("unaligned", (" ",))]

    with pytest.raises(ValueError) as excinfo:
        text_based_document_to_token_based(
            doc,
            tokens=list(token_document_with_multi_spans.tokens),
            result_document_type=TokenizedTestDocumentWithMultiSpans,
        )
    assert (
        str(excinfo.value)
        == 'cannot find token span for character span: "(\' \',)", text="First sentence. Entity M works at N. '
        'And it founded O.", token_offset_mapping=[(0, 0), (0, 5), (6, 14), (14, 15), (16, 22), (23, 24), '
        "(25, 30), (31, 33), (34, 35), (35, 36), (37, 40), (41, 43), (44, 51), (52, 53), (53, 54), (54, 54)]"
    )


def test_text_based_document_to_token_based_empty_multi_span_slice_strict(
    token_document_with_multi_spans, text_document_with_multi_spans
):
    doc = TestDocumentWithMultiSpans(text=text_document_with_multi_spans.text)
    # add a multi span that is not aligned with the tokenization
    doc.entities.append(LabeledMultiSpan(slices=((3, 3),), label="empty_slice"))
    assert doc.entities.resolve() == [("empty_slice", ("",))]

    with pytest.raises(ValueError) as excinfo:
        text_based_document_to_token_based(
            doc,
            tokens=list(token_document_with_multi_spans.tokens),
            result_document_type=TokenizedTestDocumentWithMultiSpans,
        )
    assert (
        str(excinfo.value)
        == 'cannot find token span for character span: "(\'\',)", text="First sentence. Entity M works at N. '
        'And it founded O.", token_offset_mapping=[(0, 0), (0, 5), (6, 14), (14, 15), (16, 22), (23, 24), '
        "(25, 30), (31, 33), (34, 35), (35, 36), (37, 40), (41, 43), (44, 51), (52, 53), (53, 54), (54, 54)]"
    )


def test_token_based_document_to_text_based_strip_multi_span(
    token_document_with_multi_spans, text_document_with_multi_spans
):
    doc = TestDocumentWithMultiSpans(text=text_document_with_multi_spans.text)
    # add a span that is not aligned with the tokenization
    doc.entities.append(LabeledMultiSpan(slices=((5, 16),), label="unaligned"))
    assert doc.entities.resolve() == [("unaligned", (" sentence. ",))]

    result_doc = text_based_document_to_token_based(
        doc,
        tokens=list(token_document_with_multi_spans.tokens),
        result_document_type=TokenizedTestDocumentWithMultiSpans,
        strip_spans=True,
    )
    assert result_doc.entities.resolve() == [("unaligned", (("sentence", "."),))]


def test_text_based_document_to_token_based_wrong_annotation_type():
    @dataclasses.dataclass
    class WrongAnnotationType(TextBasedDocument):
        wrong_annotations: AnnotationLayer[Label] = annotation_field(target="text")

    doc = WrongAnnotationType(text="First sentence. Entity M works at N. And it founded O.")
    doc.wrong_annotations.append(Label(label="wrong"))

    with pytest.raises(TypeError) as excinfo:
        text_based_document_to_token_based(
            doc,
            result_document_type=TokenizedTestDocument,
            token_offset_mapping=[],
        )
    assert (
        str(excinfo.value)
        == "can not convert layers that target the text but contain non-span annotations, "
        "but found <class 'pie_modules.annotations.Label'>"
    )


def test_token_based_document_to_text_based(token_document, text_document):
    added_annotations = defaultdict(dict)
    result = token_based_document_to_text_based(
        token_document,
        text=text_document.text,
        result_document_type=TestDocument,
        added_annotations=added_annotations,
    )
    _assert_added_annotations(token_document, result, added_annotations)
    _test_text_document(result)


def test_token_based_document_to_text_based_multi_span(
    token_document_with_multi_spans, text_document_with_multi_spans
):
    added_annotations = defaultdict(dict)
    result = token_based_document_to_text_based(
        token_document_with_multi_spans,
        text=text_document_with_multi_spans.text,
        result_document_type=TestDocumentWithMultiSpans,
        added_annotations=added_annotations,
    )
    _assert_added_annotations(token_document_with_multi_spans, result, added_annotations)
    _test_text_document_with_multi_spans(result)


def test_token_based_document_to_text_based_join_tokens_with(text_document, token_document):
    result = token_based_document_to_text_based(
        token_document,
        join_tokens_with=" ",
        result_document_type=TestDocument,
    )

    sentences = [str(sentence) for sentence in result.sentences]
    assert sentences == ["First sentence .", "Entity M works at N .", "And it founded O ."]

    entities = [str(entity) for entity in result.entities]
    assert entities == ["Entity M", "N", "it", "O"]

    relation_tuples = [(str(rel.head), rel.label, str(rel.tail)) for rel in result.relations]
    assert relation_tuples == [("Entity M", "per:employee_of", "N"), ("it", "per:founder", "O")]


def test_token_based_document_to_text_based_missing_text(token_document):
    with pytest.raises(ValueError) as excinfo:
        token_based_document_to_text_based(
            token_document,
            result_document_type=TestDocument,
        )
    assert (
        str(excinfo.value)
        == "if join_tokens_with is None, text must be provided, but got None as well"
    )


def test_token_based_document_token_based_offset_mapping_from_metadata(
    token_document, text_document
):
    doc = token_document.copy()
    doc.metadata["token_offset_mapping"] = find_token_offset_mapping(
        text=text_document.text, tokens=list(doc.tokens)
    )
    result = token_based_document_to_text_based(
        doc,
        text=text_document.text,
        result_document_type=TestDocument,
    )
    _test_text_document(result)


def test_token_based_document_to_text_based_wrong_annotation_type():
    @dataclasses.dataclass
    class WrongAnnotationType(TokenBasedDocument):
        wrong_annotations: AnnotationLayer[Label] = annotation_field(target="tokens")

    doc = WrongAnnotationType(tokens=("Hallo", "World"))
    doc.wrong_annotations.append(Label(label="wrong"))

    with pytest.raises(TypeError) as excinfo:
        token_based_document_to_text_based(
            doc,
            text="Hello World",
            result_document_type=TestDocument,
        )
    assert (
        str(excinfo.value)
        == "can not convert layers that target the tokens but contain non-span annotations, "
        "but found <class 'pie_modules.annotations.Label'>"
    )
