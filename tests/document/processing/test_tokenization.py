import dataclasses
from collections import defaultdict

import pytest
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument
from transformers import AutoTokenizer, PreTrainedTokenizer

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
    tokenize_document,
)
from pie_modules.document.processing.tokenization import find_token_offset_mapping
from tests.conftest import TestDocument


@dataclasses.dataclass
class TokenizedTestDocument(TokenBasedDocument):
    sentences: AnnotationList[Span] = annotation_field(target="tokens")
    entities: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TestDocumentWithMultiSpans(TextBasedDocument):
    entities: AnnotationList[LabeledMultiSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TokenizedTestDocumentWithMultiSpans(TokenBasedDocument):
    entities: AnnotationList[LabeledMultiSpan] = annotation_field(target="tokens")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


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


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("bert-base-cased")


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


def test_text_based_document_to_token_based(text_document, token_document):
    added_annotations = defaultdict(list)
    doc = text_based_document_to_token_based(
        text_document,
        tokens=list(token_document.tokens),
        result_document_type=TokenizedTestDocument,
        added_annotations=added_annotations,
    )
    for ann_field in text_document.annotation_fields():
        layer_name = ann_field.name
        assert added_annotations[layer_name] == list(text_document[layer_name])
    _test_token_document(doc)


def test_text_based_document_to_token_based_multi_span(
    text_document_with_multi_spans, token_document_with_multi_spans
):
    added_annotations = defaultdict(list)
    doc = text_based_document_to_token_based(
        text_document_with_multi_spans,
        tokens=list(token_document_with_multi_spans.tokens),
        result_document_type=TokenizedTestDocumentWithMultiSpans,
        added_annotations=added_annotations,
    )
    for ann_field in text_document_with_multi_spans.annotation_fields():
        layer_name = ann_field.name
        assert added_annotations[layer_name] == list(text_document_with_multi_spans[layer_name])
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


def test_text_based_document_to_token_based_unaligned_span_strict(text_document, token_document):
    doc = TestDocument(text=text_document.text)
    # add a span that is not aligned with the tokenization
    doc.entities.append(LabeledSpan(start=0, end=6, label="unaligned"))
    assert str(doc.entities[0]) == "First "

    with pytest.raises(ValueError) as excinfo:
        text_based_document_to_token_based(
            doc,
            tokens=list(token_document.tokens),
            result_document_type=TokenizedTestDocument,
        )
    assert (
        str(excinfo.value)
        == 'cannot find token span for character span: "First ", text="First sentence. Entity M works at N. '
        'And it founded O.", token_offset_mapping=[(0, 0), (0, 5), (6, 14), (14, 15), (16, 22), (23, 24), '
        "(25, 30), (31, 33), (34, 35), (35, 36), (37, 40), (41, 43), (44, 51), (52, 53), (53, 54), (54, 54)]"
    )


def test_text_based_document_to_token_based_unaligned_span_not_strict(
    text_document, token_document, caplog
):
    doc = TestDocument(text=text_document.text)
    # add a span that is not aligned with the tokenization
    doc.entities.append(LabeledSpan(start=0, end=6, label="unaligned"))
    assert str(doc.entities[0]) == "First "

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
        == 'cannot find token span for character span "First ", skip it (disable this warning with verbose=False)'
    )

    # check (de-)serialization
    tokenized_doc.copy()

    assert len(doc.entities) == 1
    # the unaligned span is not included in the tokenized document
    assert len(tokenized_doc.entities) == 0


def test_text_based_document_to_token_based_wrong_annotation_type():
    @dataclasses.dataclass
    class WrongAnnotationType(TextBasedDocument):
        wrong_annotations: AnnotationList[Label] = annotation_field(target="text")

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
        "but found <class 'pytorch_ie.annotations.Label'>"
    )


def test_token_based_document_to_text_based(token_document, text_document):
    added_annotations = defaultdict(list)
    result = token_based_document_to_text_based(
        token_document,
        text=text_document.text,
        result_document_type=TestDocument,
        added_annotations=added_annotations,
    )
    for ann_field in token_document.annotation_fields():
        layer_name = ann_field.name
        assert added_annotations[layer_name] == list(token_document[layer_name])

    _test_text_document(result)


def test_token_based_document_to_text_based_multi_span(
    token_document_with_multi_spans, text_document_with_multi_spans
):
    added_annotations = defaultdict(list)
    result = token_based_document_to_text_based(
        token_document_with_multi_spans,
        text=text_document_with_multi_spans.text,
        result_document_type=TestDocumentWithMultiSpans,
        added_annotations=added_annotations,
    )
    for ann_field in token_document_with_multi_spans.annotation_fields():
        layer_name = ann_field.name
        assert added_annotations[layer_name] == list(token_document_with_multi_spans[layer_name])

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
        wrong_annotations: AnnotationList[Label] = annotation_field(target="tokens")

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
        "but found <class 'pytorch_ie.annotations.Label'>"
    )


def test_tokenize_document(text_document, tokenizer):
    tokenized_docs = tokenize_document(
        text_document,
        tokenizer=tokenizer,
        result_document_type=TokenizedTestDocument,
    )
    assert len(tokenized_docs) == 1
    tokenized_doc = tokenized_docs[0]

    # check (de-)serialization
    tokenized_doc.copy()

    assert (
        tokenized_doc.metadata["text"]
        == text_document.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == (
        "[CLS]",
        "First",
        "sentence",
        ".",
        "En",
        "##ti",
        "##ty",
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
    )
    assert len(tokenized_doc.sentences) == len(text_document.sentences) == 3
    sentences = [str(sentence) for sentence in tokenized_doc.sentences]
    assert sentences == [
        "('First', 'sentence', '.')",
        "('En', '##ti', '##ty', 'M', 'works', 'at', 'N', '.')",
        "('And', 'it', 'founded', 'O', '.')",
    ]
    assert len(tokenized_doc.entities) == len(text_document.entities) == 4
    entities = [str(entity) for entity in tokenized_doc.entities]
    assert entities == ["('En', '##ti', '##ty', 'M')", "('N',)", "('it',)", "('O',)"]
    assert len(tokenized_doc.relations) == len(text_document.relations) == 2
    relation_tuples = [
        (str(rel.head), rel.label, str(rel.tail)) for rel in tokenized_doc.relations
    ]
    assert relation_tuples == [
        ("('En', '##ti', '##ty', 'M')", "per:employee_of", "('N',)"),
        ("('it',)", "per:founder", "('O',)"),
    ]


def test_tokenize_document_max_length(text_document, tokenizer, caplog):
    caplog.clear()
    with caplog.at_level("WARNING"):
        tokenized_docs = tokenize_document(
            text_document,
            tokenizer=tokenizer,
            result_document_type=TokenizedTestDocument,
            # max_length is set to 10, so the document is split into two parts
            strict_span_conversion=False,
            max_length=10,
            return_overflowing_tokens=True,
        )
    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message
        == "could not convert all annotations from document with id=None to token based documents, missed annotations "
        "(disable this message with verbose=False):\n"
        "{\n"
        '  "relations": "{BinaryRelation(head=LabeledSpan(start=16, end=24, label=\'per\', score=1.0), '
        "tail=LabeledSpan(start=34, end=35, label='org', score=1.0), label='per:employee_of', score=1.0)}\",\n"
        '  "sentences": "{Span(start=16, end=36)}"\n'
        "}"
    )
    assert len(tokenized_docs) == 2
    tokenized_doc = tokenized_docs[0]

    # check (de-)serialization
    tokenized_doc.copy()

    assert (
        tokenized_doc.metadata["text"]
        == text_document.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == (
        "[CLS]",
        "First",
        "sentence",
        ".",
        "En",
        "##ti",
        "##ty",
        "M",
        "works",
        "[SEP]",
    )
    assert len(tokenized_doc.sentences) == 1
    sentences = [str(sentence) for sentence in tokenized_doc.sentences]
    assert sentences == ["('First', 'sentence', '.')"]
    assert len(tokenized_doc.entities) == 1
    entities = [str(entity) for entity in tokenized_doc.entities]
    assert entities == ["('En', '##ti', '##ty', 'M')"]
    assert len(tokenized_doc.relations) == 0

    tokenized_doc = tokenized_docs[1]

    # check (de-)serialization
    tokenized_doc.copy()

    assert (
        tokenized_doc.metadata["text"]
        == text_document.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == (
        "[CLS]",
        "at",
        "N",
        ".",
        "And",
        "it",
        "founded",
        "O",
        ".",
        "[SEP]",
    )
    assert len(tokenized_doc.sentences) == 1
    sentences = [str(sentence) for sentence in tokenized_doc.sentences]
    assert sentences == ["('And', 'it', 'founded', 'O', '.')"]
    assert len(tokenized_doc.entities) == 3
    entities = [str(entity) for entity in tokenized_doc.entities]
    assert entities == ["('N',)", "('it',)", "('O',)"]
    assert len(tokenized_doc.relations) == 1
    relation_tuples = [
        (str(rel.head), rel.label, str(rel.tail)) for rel in tokenized_doc.relations
    ]
    assert relation_tuples == [("('it',)", "per:founder", "('O',)")]


def test_tokenize_document_max_length_strict(text_document, tokenizer):
    with pytest.raises(ValueError) as excinfo:
        tokenize_document(
            text_document,
            tokenizer=tokenizer,
            result_document_type=TokenizedTestDocument,
            # max_length is set to 10, so the document is split into two parts
            strict_span_conversion=True,
            max_length=10,
            return_overflowing_tokens=True,
        )
    assert (
        str(excinfo.value)
        == "could not convert all annotations from document with id=None to token based documents, "
        "but strict_span_conversion is True, so raise an error, missed annotations:\n"
        "{\n"
        '  "relations": "{BinaryRelation(head=LabeledSpan(start=16, end=24, label=\'per\', score=1.0), '
        "tail=LabeledSpan(start=34, end=35, label='org', score=1.0), label='per:employee_of', score=1.0)}\",\n"
        '  "sentences": "{Span(start=16, end=36)}"\n'
        "}"
    )


def test_tokenize_document_partition(text_document, tokenizer):
    tokenized_docs = tokenize_document(
        text_document,
        tokenizer=tokenizer,
        result_document_type=TokenizedTestDocument,
        partition_layer="sentences",
    )
    assert len(tokenized_docs) == 3
    tokenized_doc = tokenized_docs[0]

    # check (de-)serialization
    tokenized_doc.copy()

    assert (
        tokenized_doc.metadata["text"]
        == text_document.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == ("[CLS]", "First", "sentence", ".", "[SEP]")
    assert len(tokenized_doc.sentences) == 1
    assert len(tokenized_doc.entities) == 0
    assert len(tokenized_doc.relations) == 0

    tokenized_doc = tokenized_docs[1]

    # check (de-)serialization
    tokenized_doc.copy()

    assert (
        tokenized_doc.metadata["text"]
        == text_document.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == (
        "[CLS]",
        "En",
        "##ti",
        "##ty",
        "M",
        "works",
        "at",
        "N",
        ".",
        "[SEP]",
    )
    assert len(tokenized_doc.sentences) == 1
    sentences = [str(sentence) for sentence in tokenized_doc.sentences]
    assert sentences == ["('En', '##ti', '##ty', 'M', 'works', 'at', 'N', '.')"]
    assert len(tokenized_doc.entities) == 2
    entities = [str(entity) for entity in tokenized_doc.entities]
    assert entities == ["('En', '##ti', '##ty', 'M')", "('N',)"]
    assert len(tokenized_doc.relations) == 1
    relation_tuples = [
        (str(rel.head), rel.label, str(rel.tail)) for rel in tokenized_doc.relations
    ]
    assert relation_tuples == [("('En', '##ti', '##ty', 'M')", "per:employee_of", "('N',)")]

    tokenized_doc = tokenized_docs[2]

    # check (de-)serialization
    tokenized_doc.copy()

    assert (
        tokenized_doc.metadata["text"]
        == text_document.text
        == "First sentence. Entity M works at N. And it founded O."
    )
    assert tokenized_doc.tokens == ("[CLS]", "And", "it", "founded", "O", ".", "[SEP]")
    assert len(tokenized_doc.sentences) == 1
    sentences = [str(sentence) for sentence in tokenized_doc.sentences]
    assert sentences == ["('And', 'it', 'founded', 'O', '.')"]
    assert len(tokenized_doc.entities) == 2
    entities = [str(entity) for entity in tokenized_doc.entities]
    assert entities == ["('it',)", "('O',)"]
    assert len(tokenized_doc.relations) == 1
    relation_tuples = [
        (str(rel.head), rel.label, str(rel.tail)) for rel in tokenized_doc.relations
    ]
    assert relation_tuples == [("('it',)", "per:founder", "('O',)")]


def test_tokenize_document_with_slow_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)
    text_document = TextBasedDocument(text="Alice has a cat. Bob has a dog.")

    tokenized_docs = tokenize_document(
        text_document, tokenizer=tokenizer, result_document_type=TokenBasedDocument
    )
    assert len(tokenized_docs) == 1


def test_tokenize_document_with_slow_tokenizer_and_windowing():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)
    text_document = TextBasedDocument(text="Alice has a cat. Bob has a dog.")

    tokenized_docs = tokenize_document(
        text_document,
        tokenizer=tokenizer,
        result_document_type=TokenBasedDocument,
        max_length=5,
        return_overflowing_tokens=True,
    )
    assert (
        len(tokenized_docs) == 3
    )  # the input text gets tokenized into 12 tokens and max_length is 5
