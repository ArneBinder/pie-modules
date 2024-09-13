from itertools import chain
from typing import List

import pytest
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

from pie_modules.document.processing.text_pair import (
    add_negative_coref_relations,
    construct_text_pair_coref_documents_from_partitions_via_relations,
)
from pie_modules.document.types import (
    BinaryCorefRelation,
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations,
)

SENTENCES = [
    "Entity A works at B.",
    "And she founded C.",
    "Bob loves his cat.",
    "She sleeps a lot.",
]


@pytest.fixture(scope="module")
def text_documents() -> List[TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions]:
    doc1 = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
        id="doc1", text=" ".join(SENTENCES[:2])
    )
    # add sentence partitions
    doc1.labeled_partitions.append(LabeledSpan(start=0, end=len(SENTENCES[0]), label="sentence"))
    doc1.labeled_partitions.append(
        LabeledSpan(
            start=len(SENTENCES[0]) + 1,
            end=len(SENTENCES[0]) + 1 + len(SENTENCES[1]),
            label="sentence",
        )
    )
    # add spans
    doc1.labeled_spans.append(LabeledSpan(start=0, end=8, label="PERSON"))
    doc1.labeled_spans.append(LabeledSpan(start=18, end=19, label="COMPANY"))
    doc1_sen2_offset = doc1.labeled_partitions[1].start
    doc1.labeled_spans.append(
        LabeledSpan(start=4 + doc1_sen2_offset, end=7 + doc1_sen2_offset, label="PERSON")
    )
    doc1.labeled_spans.append(
        LabeledSpan(start=16 + doc1_sen2_offset, end=17 + doc1_sen2_offset, label="COMPANY")
    )
    # add relation
    doc1.binary_relations.append(
        BinaryRelation(
            head=doc1.labeled_spans[0], tail=doc1.labeled_spans[2], label="semantically_same"
        )
    )

    doc2 = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
        id="doc2", text=" ".join(SENTENCES[2:4])
    )
    # add sentence partitions
    doc2.labeled_partitions.append(LabeledSpan(start=0, end=len(SENTENCES[2]), label="sentence"))
    doc2.labeled_partitions.append(
        LabeledSpan(
            start=len(SENTENCES[2]) + 1,
            end=len(SENTENCES[2]) + 1 + len(SENTENCES[3]),
            label="sentence",
        )
    )
    # add spans
    doc2.labeled_spans.append(LabeledSpan(start=0, end=3, label="PERSON"))
    doc2.labeled_spans.append(LabeledSpan(start=10, end=17, label="ANIMAL"))
    doc2_sen2_offset = doc2.labeled_partitions[1].start
    doc2.labeled_spans.append(
        LabeledSpan(start=0 + doc2_sen2_offset, end=3 + doc2_sen2_offset, label="ANIMAL")
    )
    # add relation
    doc2.binary_relations.append(
        BinaryRelation(
            head=doc2.labeled_spans[1], tail=doc2.labeled_spans[2], label="semantically_same"
        )
    )

    return [doc1, doc2]


def test_simple_text_documents(text_documents):
    assert len(text_documents) == 2
    doc = text_documents[0]
    # test serialization
    doc.copy()
    # test sentences
    assert doc.labeled_partitions.resolve() == [
        ("sentence", "Entity A works at B."),
        ("sentence", "And she founded C."),
    ]
    # test spans
    assert doc.labeled_spans.resolve() == [
        ("PERSON", "Entity A"),
        ("COMPANY", "B"),
        ("PERSON", "she"),
        ("COMPANY", "C"),
    ]
    # test relation
    assert doc.binary_relations.resolve() == [
        ("semantically_same", (("PERSON", "Entity A"), ("PERSON", "she")))
    ]

    doc = text_documents[1]
    # test serialization
    doc.copy()
    # test sentences
    assert doc.labeled_partitions.resolve() == [
        ("sentence", "Bob loves his cat."),
        ("sentence", "She sleeps a lot."),
    ]
    # test spans
    assert doc.labeled_spans.resolve() == [
        ("PERSON", "Bob"),
        ("ANIMAL", "his cat"),
        ("ANIMAL", "She"),
    ]
    # test relation
    assert doc.binary_relations.resolve() == [
        ("semantically_same", (("ANIMAL", "his cat"), ("ANIMAL", "She")))
    ]


def test_construct_text_pair_coref_documents_from_partitions_via_relations(text_documents):
    all_docs = {
        doc.id: doc
        for doc in construct_text_pair_coref_documents_from_partitions_via_relations(
            documents=text_documents, relation_label="semantically_same"
        )
    }
    assert set(all_docs) == {"doc2[0:18]+doc2[19:36]", "doc1[0:20]+doc1[21:39]"}

    doc = all_docs["doc2[0:18]+doc2[19:36]"]
    assert doc.text == "Bob loves his cat."
    assert doc.text_pair == "She sleeps a lot."
    assert doc.labeled_spans.resolve() == [("PERSON", "Bob"), ("ANIMAL", "his cat")]
    assert doc.labeled_spans_pair.resolve() == [("ANIMAL", "She")]
    assert doc.binary_coref_relations.resolve() == [
        ("coref", (("ANIMAL", "his cat"), ("ANIMAL", "She")))
    ]

    doc = all_docs["doc1[0:20]+doc1[21:39]"]
    assert doc.text == "Entity A works at B."
    assert doc.text_pair == "And she founded C."
    assert doc.labeled_spans.resolve() == [("PERSON", "Entity A"), ("COMPANY", "B")]
    assert doc.labeled_spans_pair.resolve() == [("PERSON", "she"), ("COMPANY", "C")]
    assert doc.binary_coref_relations.resolve() == [
        ("coref", (("PERSON", "Entity A"), ("PERSON", "she")))
    ]


@pytest.fixture(scope="module")
def positive_documents():
    doc1 = TextPairDocumentWithLabeledSpansAndBinaryCorefRelations(
        id="0", text="Entity A works at B.", text_pair="And she founded C."
    )
    doc1.labeled_spans.append(LabeledSpan(start=0, end=8, label="PERSON"))
    doc1.labeled_spans.append(LabeledSpan(start=18, end=19, label="COMPANY"))
    doc1.labeled_spans_pair.append(LabeledSpan(start=4, end=7, label="PERSON"))
    doc1.labeled_spans_pair.append(LabeledSpan(start=16, end=17, label="COMPANY"))
    doc1.binary_coref_relations.append(
        BinaryCorefRelation(head=doc1.labeled_spans[0], tail=doc1.labeled_spans_pair[0])
    )

    doc2 = TextPairDocumentWithLabeledSpansAndBinaryCorefRelations(
        id="0", text="Bob loves his cat.", text_pair="She sleeps a lot."
    )
    doc2.labeled_spans.append(LabeledSpan(start=0, end=3, label="PERSON"))
    doc2.labeled_spans.append(LabeledSpan(start=10, end=17, label="ANIMAL"))
    doc2.labeled_spans_pair.append(LabeledSpan(start=0, end=3, label="ANIMAL"))
    doc2.binary_coref_relations.append(
        BinaryCorefRelation(head=doc2.labeled_spans[1], tail=doc2.labeled_spans_pair[0])
    )

    return [doc1, doc2]


def test_positive_documents(positive_documents):
    assert len(positive_documents) == 2
    doc1, doc2 = positive_documents
    assert doc1.labeled_spans.resolve() == [("PERSON", "Entity A"), ("COMPANY", "B")]
    assert doc1.labeled_spans_pair.resolve() == [("PERSON", "she"), ("COMPANY", "C")]
    assert doc1.binary_coref_relations.resolve() == [
        ("coref", (("PERSON", "Entity A"), ("PERSON", "she")))
    ]

    assert doc2.labeled_spans.resolve() == [("PERSON", "Bob"), ("ANIMAL", "his cat")]
    assert doc2.labeled_spans_pair.resolve() == [("ANIMAL", "She")]
    assert doc2.binary_coref_relations.resolve() == [
        ("coref", (("ANIMAL", "his cat"), ("ANIMAL", "She")))
    ]


def test_construct_negative_documents(positive_documents):
    assert len(positive_documents) == 2
    docs = list(add_negative_coref_relations(positive_documents))
    TEXTS = [
        "Entity A works at B.",
        "And she founded C.",
        "Bob loves his cat.",
        "She sleeps a lot.",
    ]
    assert all(doc.text in TEXTS for doc in docs)
    assert all(doc.text_pair in TEXTS for doc in docs)

    all_texts = [(doc.text, doc.text_pair) for doc in docs]
    all_scores = [[coref_rel.score for coref_rel in doc.binary_coref_relations] for doc in docs]
    all_rels_resolved = [doc.binary_coref_relations.resolve() for doc in docs]

    all_rels_and_scores = [
        (texts, list(zip(scores, rels_resolved)))
        for texts, scores, rels_resolved in zip(all_texts, all_scores, all_rels_resolved)
    ]

    assert all_rels_and_scores == [
        (("And she founded C.", "And she founded C."), []),
        (
            ("And she founded C.", "Bob loves his cat."),
            [(0.0, ("coref", (("PERSON", "she"), ("PERSON", "Bob"))))],
        ),
        (
            ("And she founded C.", "Entity A works at B."),
            [
                (1.0, ("coref", (("PERSON", "she"), ("PERSON", "Entity A")))),
                (0.0, ("coref", (("COMPANY", "C"), ("COMPANY", "B")))),
            ],
        ),
        (("And she founded C.", "She sleeps a lot."), []),
        (
            ("Bob loves his cat.", "And she founded C."),
            [(0.0, ("coref", (("PERSON", "Bob"), ("PERSON", "she"))))],
        ),
        (("Bob loves his cat.", "Bob loves his cat."), []),
        (
            ("Bob loves his cat.", "Entity A works at B."),
            [(0.0, ("coref", (("PERSON", "Bob"), ("PERSON", "Entity A"))))],
        ),
        (
            ("Bob loves his cat.", "She sleeps a lot."),
            [(1.0, ("coref", (("ANIMAL", "his cat"), ("ANIMAL", "She"))))],
        ),
        (
            ("Entity A works at B.", "And she founded C."),
            [
                (1.0, ("coref", (("PERSON", "Entity A"), ("PERSON", "she")))),
                (0.0, ("coref", (("COMPANY", "B"), ("COMPANY", "C")))),
            ],
        ),
        (
            ("Entity A works at B.", "Bob loves his cat."),
            [(0.0, ("coref", (("PERSON", "Entity A"), ("PERSON", "Bob"))))],
        ),
        (("Entity A works at B.", "Entity A works at B."), []),
        (("Entity A works at B.", "She sleeps a lot."), []),
        (("She sleeps a lot.", "And she founded C."), []),
        (
            ("She sleeps a lot.", "Bob loves his cat."),
            [(1.0, ("coref", (("ANIMAL", "She"), ("ANIMAL", "his cat"))))],
        ),
        (("She sleeps a lot.", "Entity A works at B."), []),
        (("She sleeps a lot.", "She sleeps a lot."), []),
    ]
