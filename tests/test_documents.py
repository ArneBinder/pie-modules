from pytorch_ie.annotations import BinaryRelation, LabeledSpan

from pie_modules.annotations import ExtractiveAnswer, Question
from pie_modules.documents import (
    ExtractiveQADocument,
    TokenDocumentWithLabeledPartitions,
    TokenDocumentWithLabeledSpans,
    TokenDocumentWithLabeledSpansAndBinaryRelations,
    TokenDocumentWithLabeledSpansAndLabeledPartitions,
    TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    TokenizedExtractiveQADocument,
)


def test_token_document_with_labeled_spans():
    doc = TokenDocumentWithLabeledSpans(
        tokens=("This", "is", "a", "sentence", "."), id="token_document_with_labeled_spans"
    )
    e1 = LabeledSpan(start=0, end=1, label="entity")
    doc.labeled_spans.append(e1)
    assert str(e1) == "('This',)"
    e2 = LabeledSpan(start=2, end=4, label="entity")
    doc.labeled_spans.append(e2)
    assert str(e2) == "('a', 'sentence')"

    # test (de-)serialization
    doc_copy = doc.copy()
    assert doc == doc_copy


def test_token_document_with_labeled_partitions():
    doc = TokenDocumentWithLabeledPartitions(
        tokens=(
            "This",
            "is",
            "a",
            "sentence",
            ".",
            "And",
            "this",
            "is",
            "another",
            "sentence",
            ".",
        ),
        id="token_document_with_labeled_partitions",
    )
    sent1 = LabeledSpan(start=0, end=5, label="sentence")
    doc.labeled_partitions.append(sent1)
    assert str(sent1) == "('This', 'is', 'a', 'sentence', '.')"
    sent2 = LabeledSpan(start=5, end=11, label="sentence")
    doc.labeled_partitions.append(sent2)
    assert str(sent2) == "('And', 'this', 'is', 'another', 'sentence', '.')"

    # test (de-)serialization
    doc_copy = doc.copy()
    assert doc == doc_copy


def test_token_document_with_labeled_spans_and_labeled_partitions():
    doc = TokenDocumentWithLabeledSpansAndLabeledPartitions(
        tokens=(
            "This",
            "is",
            "a",
            "sentence",
            ".",
            "And",
            "this",
            "is",
            "another",
            "sentence",
            ".",
        ),
        id="token_document_with_labeled_spans_and_labeled_partitions",
    )
    e1 = LabeledSpan(start=0, end=1, label="entity")
    doc.labeled_spans.append(e1)
    assert str(e1) == "('This',)"
    e2 = LabeledSpan(start=2, end=4, label="entity")
    doc.labeled_spans.append(e2)
    assert str(e2) == "('a', 'sentence')"
    sent1 = LabeledSpan(start=0, end=5, label="sentence")
    doc.labeled_partitions.append(sent1)
    assert str(sent1) == "('This', 'is', 'a', 'sentence', '.')"
    sent2 = LabeledSpan(start=5, end=11, label="sentence")
    doc.labeled_partitions.append(sent2)
    assert str(sent2) == "('And', 'this', 'is', 'another', 'sentence', '.')"

    # test (de-)serialization
    doc_copy = doc.copy()
    assert doc == doc_copy


def test_token_document_with_labeled_spans_and_binary_relations():
    doc = TokenDocumentWithLabeledSpansAndBinaryRelations(
        tokens=(
            "This",
            "is",
            "a",
            "sentence",
            ".",
            "And",
            "this",
            "is",
            "another",
            "sentence",
            ".",
        ),
        id="token_document_with_labeled_spans_and_binary_relations",
    )
    e1 = LabeledSpan(start=0, end=1, label="entity")
    doc.labeled_spans.append(e1)
    assert str(e1) == "('This',)"
    e2 = LabeledSpan(start=2, end=4, label="entity")
    doc.labeled_spans.append(e2)
    assert str(e2) == "('a', 'sentence')"
    r1 = BinaryRelation(head=e1, tail=e2, label="relation")
    doc.binary_relations.append(r1)
    assert str(r1.head) == "('This',)"
    assert str(r1.tail) == "('a', 'sentence')"
    assert r1.label == "relation"

    # test (de-)serialization
    doc_copy = doc.copy()
    assert doc == doc_copy


def test_token_document_with_labeled_spans_binary_relations_and_labeled_partitions():
    doc = TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
        tokens=(
            "This",
            "is",
            "a",
            "sentence",
            ".",
            "And",
            "this",
            "is",
            "another",
            "sentence",
            ".",
        ),
        id="token_document_with_labeled_spans_binary_relations_and_labeled_partitions",
    )
    e1 = LabeledSpan(start=0, end=1, label="entity")
    doc.labeled_spans.append(e1)
    assert str(e1) == "('This',)"
    e2 = LabeledSpan(start=2, end=4, label="entity")
    doc.labeled_spans.append(e2)
    assert str(e2) == "('a', 'sentence')"
    r1 = BinaryRelation(head=e1, tail=e2, label="relation")
    doc.binary_relations.append(r1)
    assert str(r1.head) == "('This',)"
    assert str(r1.tail) == "('a', 'sentence')"
    assert r1.label == "relation"
    sent1 = LabeledSpan(start=0, end=5, label="sentence")
    doc.labeled_partitions.append(sent1)
    assert str(sent1) == "('This', 'is', 'a', 'sentence', '.')"
    sent2 = LabeledSpan(start=5, end=11, label="sentence")
    doc.labeled_partitions.append(sent2)
    assert str(sent2) == "('And', 'this', 'is', 'another', 'sentence', '.')"

    # test (de-)serialization
    doc_copy = doc.copy()
    assert doc == doc_copy


def test_extractive_qa_document():
    doc = ExtractiveQADocument(
        text="This is a sentence. And that is another sentence.", id="extractive_qa_document"
    )
    q1 = Question(text="What is this?")
    doc.questions.append(q1)
    q2 = Question(text="What is that?")
    doc.questions.append(q2)

    a1 = ExtractiveAnswer(start=8, end=18, question=q1)
    doc.answers.append(a1)
    assert str(a1.question) == "What is this?"
    assert str(a1) == "a sentence"

    a2 = ExtractiveAnswer(start=32, end=48, question=q2)
    doc.answers.append(a2)
    assert str(a2.question) == "What is that?"
    assert str(a2) == "another sentence"

    # test (de-)serialization
    doc_copy = doc.copy()
    assert doc == doc_copy


def test_tokenized_extractive_qa_document():
    doc = TokenizedExtractiveQADocument(
        tokens=(
            "This",
            "is",
            "a",
            "sentence",
            ".",
            "And",
            "that",
            "is",
            "another",
            "sentence",
            ".",
        ),
        id="tokenized_extractive_qa_document",
    )
    q1 = Question(text="What is this?")
    doc.questions.append(q1)
    q2 = Question(text="What is that?")
    doc.questions.append(q2)

    a1 = ExtractiveAnswer(start=2, end=4, question=q1)
    doc.answers.append(a1)
    assert str(a1.question) == "What is this?"
    assert str(a1) == "('a', 'sentence')"

    a2 = ExtractiveAnswer(start=8, end=10, question=q2)
    doc.answers.append(a2)
    assert str(a2.question) == "What is that?"
    assert str(a2) == "('another', 'sentence')"

    # test (de-)serialization
    doc_copy = doc.copy()
    assert doc == doc_copy
