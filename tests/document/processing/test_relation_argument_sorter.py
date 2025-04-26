import dataclasses
import logging

import pytest
from pytorch_ie.core import Annotation, AnnotationLayer, annotation_field

from pie_modules.annotations import (
    BinaryRelation,
    LabeledMultiSpan,
    LabeledSpan,
    NaryRelation,
)
from pie_modules.document.processing import RelationArgumentSorter
from pie_modules.document.processing.relation_argument_sorter import (
    construct_relation_with_new_args,
    get_relation_args,
)
from pie_modules.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansAndBinaryRelations,
)


@pytest.fixture
def document():
    doc = TextDocumentWithLabeledSpansAndBinaryRelations(
        text="Entity G works at H. And founded I."
    )
    doc.labeled_spans.append(LabeledSpan(start=0, end=8, label="PER"))
    assert str(doc.labeled_spans[0]) == "Entity G"
    doc.labeled_spans.append(LabeledSpan(start=18, end=19, label="ORG"))
    assert str(doc.labeled_spans[1]) == "H"
    doc.labeled_spans.append(LabeledSpan(start=33, end=34, label="ORG"))
    assert str(doc.labeled_spans[2]) == "I"

    return doc


def test_relation_argument_sorter(document):
    # these arguments are not sorted
    document.binary_relations.append(
        BinaryRelation(
            head=document.labeled_spans[1], tail=document.labeled_spans[0], label="worksAt"
        )
    )
    # these arguments are sorted
    document.binary_relations.append(
        BinaryRelation(
            head=document.labeled_spans[0], tail=document.labeled_spans[2], label="founded"
        )
    )

    arg_sorter = RelationArgumentSorter(relation_layer="binary_relations")
    doc_sorted_args = arg_sorter(document)

    assert document.text == doc_sorted_args.text
    assert document.labeled_spans == doc_sorted_args.labeled_spans
    assert len(doc_sorted_args.binary_relations) == len(document.binary_relations)

    # this relation should be sorted
    assert str(doc_sorted_args.binary_relations[0].head) == "Entity G"
    assert str(doc_sorted_args.binary_relations[0].tail) == "H"
    assert doc_sorted_args.binary_relations[0].label == "worksAt"

    # this relation should be the same as before
    assert str(doc_sorted_args.binary_relations[1].head) == "Entity G"
    assert str(doc_sorted_args.binary_relations[1].tail) == "I"
    assert doc_sorted_args.binary_relations[1].label == "founded"

    assert document != doc_sorted_args


@pytest.fixture
def document_with_nary_relation():
    @dataclasses.dataclass
    class TextDocumentWithLabeledSpansAndNaryRelations(TextDocumentWithLabeledSpans):
        nary_relations: AnnotationLayer[NaryRelation] = annotation_field(target="labeled_spans")

    doc = TextDocumentWithLabeledSpansAndNaryRelations(text="Entity G works at H. And founded I.")
    doc.labeled_spans.append(LabeledSpan(start=0, end=8, label="PER"))
    assert str(doc.labeled_spans[0]) == "Entity G"
    doc.labeled_spans.append(LabeledSpan(start=18, end=19, label="ORG"))
    assert str(doc.labeled_spans[1]) == "H"
    doc.labeled_spans.append(LabeledSpan(start=33, end=34, label="ORG"))
    assert str(doc.labeled_spans[2]) == "I"

    doc.nary_relations.append(
        NaryRelation(
            arguments=(doc.labeled_spans[0], doc.labeled_spans[1], doc.labeled_spans[2]),
            roles=("person", "worksAt", "founded"),
            label="event",
        )
    )

    return doc


def test_get_args_wrong_type(document_with_nary_relation):
    with pytest.raises(TypeError) as excinfo:
        get_relation_args(document_with_nary_relation.nary_relations[0])
    assert (
        str(excinfo.value)
        == "relation NaryRelation(arguments=(LabeledSpan(start=0, end=8, label='PER', score=1.0), "
        "LabeledSpan(start=18, end=19, label='ORG', score=1.0), LabeledSpan(start=33, end=34, "
        "label='ORG', score=1.0)), roles=('person', 'worksAt', 'founded'), label='event', score=1.0) "
        "has unknown type [<class 'pytorch_ie.annotations.NaryRelation'>], cannot get arguments from it"
    )


def test_construct_relation_with_new_args_wrong_type(document_with_nary_relation):
    with pytest.raises(TypeError) as excinfo:
        construct_relation_with_new_args(
            document_with_nary_relation.nary_relations[0],
            (
                document_with_nary_relation.labeled_spans[0],
                document_with_nary_relation.labeled_spans[1],
            ),
        )
    assert (
        str(excinfo.value)
        == "original relation NaryRelation(arguments=(LabeledSpan(start=0, end=8, label='PER', score=1.0), "
        "LabeledSpan(start=18, end=19, label='ORG', score=1.0), LabeledSpan(start=33, end=34, label='ORG', "
        "score=1.0)), roles=('person', 'worksAt', 'founded'), label='event', score=1.0) has unknown type "
        "[<class 'pytorch_ie.annotations.NaryRelation'>], cannot reconstruct it with new arguments"
    )


def test_relation_argument_sorter_with_label_whitelist(document):
    # argument of both relations are not sorted
    document.binary_relations.append(
        BinaryRelation(
            head=document.labeled_spans[1], tail=document.labeled_spans[0], label="worksAt"
        )
    )
    document.binary_relations.append(
        BinaryRelation(
            head=document.labeled_spans[2], tail=document.labeled_spans[0], label="founded"
        )
    )

    # we only want to sort the relations with the label "founded"
    arg_sorter = RelationArgumentSorter(
        relation_layer="binary_relations",
        label_whitelist=["founded"],
    )
    doc_sorted_args = arg_sorter(document)

    assert document.text == doc_sorted_args.text
    assert document.labeled_spans == doc_sorted_args.labeled_spans

    # this relation should be the same as before
    assert doc_sorted_args.binary_relations[0] == document.binary_relations[0]

    # this relation should be sorted
    assert doc_sorted_args.binary_relations[1] != document.binary_relations[1]
    assert str(doc_sorted_args.binary_relations[1].head) == "Entity G"
    assert str(doc_sorted_args.binary_relations[1].tail) == "I"
    assert doc_sorted_args.binary_relations[1].label == "founded"


def test_relation_argument_sorter_sorted_rel_already_exists_with_same_label(document, caplog):
    document.binary_relations.append(
        BinaryRelation(
            head=document.labeled_spans[1], tail=document.labeled_spans[0], label="worksAt"
        )
    )
    document.binary_relations.append(
        BinaryRelation(
            head=document.labeled_spans[0], tail=document.labeled_spans[1], label="worksAt"
        )
    )

    arg_sorter = RelationArgumentSorter(relation_layer="binary_relations")

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        doc_sorted_args = arg_sorter(document)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "do not add the new relation with sorted arguments, because it is already there: "
        "BinaryRelation(head=LabeledSpan(start=0, end=8, label='PER', score=1.0), "
        "tail=LabeledSpan(start=18, end=19, label='ORG', score=1.0), label='worksAt', score=1.0)"
    )

    assert document.text == doc_sorted_args.text
    assert document.labeled_spans == doc_sorted_args.labeled_spans

    # since there is already a relation with the same label and sorted arguments,
    # there should be just one relation in the end
    assert len(doc_sorted_args.binary_relations) == 1
    assert str(doc_sorted_args.binary_relations[0].head) == "Entity G"
    assert str(doc_sorted_args.binary_relations[0].tail) == "H"


def test_relation_argument_sorter_sorted_rel_already_exists_with_different_label(document):
    document.binary_relations.append(
        BinaryRelation(
            head=document.labeled_spans[1], tail=document.labeled_spans[0], label="worksAt"
        )
    )
    document.binary_relations.append(
        BinaryRelation(
            head=document.labeled_spans[0], tail=document.labeled_spans[1], label="founded"
        )
    )

    arg_sorter = RelationArgumentSorter(relation_layer="binary_relations")

    with pytest.raises(ValueError) as excinfo:
        arg_sorter(document)
    assert (
        str(excinfo.value)
        == "there is already a relation with sorted args (LabeledSpan(start=0, end=8, label='PER', score=1.0), "
        "LabeledSpan(start=18, end=19, label='ORG', score=1.0)) but with a different label: founded != worksAt"
    )


def test_relation_argument_sorter_with_dependent_layers():
    @dataclasses.dataclass(frozen=True)
    class Attribute(Annotation):
        annotation: Annotation
        label: str

    @dataclasses.dataclass
    class ExampleDocument(TextBasedDocument):
        labeled_spans: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
        binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(
            target="labeled_spans"
        )
        relation_attributes: AnnotationLayer[Attribute] = annotation_field(
            target="binary_relations"
        )

    doc = ExampleDocument(text="Entity G works at H. And founded I.")
    doc.labeled_spans.append(LabeledSpan(start=0, end=8, label="PER"))
    assert str(doc.labeled_spans[0]) == "Entity G"
    doc.labeled_spans.append(LabeledSpan(start=18, end=19, label="ORG"))
    assert str(doc.labeled_spans[1]) == "H"
    doc.binary_relations.append(
        BinaryRelation(head=doc.labeled_spans[1], tail=doc.labeled_spans[0], label="worksAt")
    )
    assert str(doc.binary_relations[0].head) == "H"
    assert str(doc.binary_relations[0].tail) == "Entity G"
    doc.relation_attributes.append(
        Attribute(annotation=doc.binary_relations[0], label="some_attribute")
    )

    arg_sorter = RelationArgumentSorter(relation_layer="binary_relations")

    doc_sorted_args = arg_sorter(doc)

    assert doc.text == doc_sorted_args.text
    assert doc.labeled_spans == doc_sorted_args.labeled_spans
    assert len(doc_sorted_args.relation_attributes) == len(doc.relation_attributes) == 1
    new_rel = doc_sorted_args.binary_relations[0]
    assert str(new_rel.head) == "Entity G"
    assert str(new_rel.tail) == "H"
    assert len(doc_sorted_args.relation_attributes) == len(doc.relation_attributes) == 1
    assert doc_sorted_args.relation_attributes[0].annotation == new_rel
    assert doc_sorted_args.relation_attributes[0].label == "some_attribute"


def test_relation_argument_sorter_with_labeled_multi_spans():
    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        labeled_multi_spans: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="text")
        binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(
            target="labeled_multi_spans"
        )

    doc = TestDocument(text="Karl The Big Heinz loves what he does.")
    karl = LabeledMultiSpan(slices=((0, 4), (13, 18)), label="PER")
    doc.labeled_multi_spans.append(karl)
    assert str(karl) == "('Karl', 'Heinz')"
    he = LabeledMultiSpan(slices=((30, 32),), label="PER")
    doc.labeled_multi_spans.append(he)
    assert str(he) == "('he',)"
    doc.binary_relations.append(BinaryRelation(head=he, tail=karl, label="coref"))

    arg_sorter = RelationArgumentSorter(relation_layer="binary_relations")
    doc_sorted_args = arg_sorter(doc)

    assert doc.text == doc_sorted_args.text
    assert doc.labeled_multi_spans == doc_sorted_args.labeled_multi_spans
    assert len(doc_sorted_args.binary_relations) == len(doc.binary_relations) == 1
    new_rel = doc_sorted_args.binary_relations[0]
    assert new_rel.head == karl
    assert new_rel.tail == he
    assert new_rel.label == "coref"
