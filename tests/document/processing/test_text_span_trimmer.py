import dataclasses

import pytest
from pytorch_ie.core import AnnotationList, annotation_field

from pie_modules.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pie_modules.document.processing import TextSpanTrimmer
from pie_modules.documents import TextBasedDocument


@dataclasses.dataclass
class DocumentWithEntitiesRelationsAndPartitions(TextBasedDocument):
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
    partitions: AnnotationList[LabeledSpan] = annotation_field(target="text")


@pytest.fixture
def document1() -> DocumentWithEntitiesRelationsAndPartitions:
    TEXT1 = "Jane lives in Berlin. this is a truncated sentence about Karl\n   "
    ENTITY_JANE_TEXT1 = LabeledSpan(start=0, end=4, label="person")
    ENTITY_BERLIN_TEXT1 = LabeledSpan(start=13, end=20, label="city")
    ENTITY_KARL_TEXT1 = LabeledSpan(start=57, end=61, label="person")
    ENTITY_EMPTY_TEXT1 = LabeledSpan(start=62, end=65, label="other")
    SENTENCE1_TEXT1 = LabeledSpan(start=0, end=21, label="sentence")
    SENTENCE2_TEXT1 = LabeledSpan(start=22, end=65, label="sentence")
    REL_JANE_LIVES_IN_BERLIN = BinaryRelation(
        head=ENTITY_JANE_TEXT1, tail=ENTITY_BERLIN_TEXT1, label="lives_in"
    )
    REL_KARL_HAS_NOTHING = BinaryRelation(
        head=ENTITY_KARL_TEXT1, tail=ENTITY_EMPTY_TEXT1, label="has_nothing"
    )

    document = DocumentWithEntitiesRelationsAndPartitions(text=TEXT1)
    document.entities.extend(
        [ENTITY_JANE_TEXT1, ENTITY_BERLIN_TEXT1, ENTITY_KARL_TEXT1, ENTITY_EMPTY_TEXT1]
    )
    document.partitions.extend([SENTENCE1_TEXT1, SENTENCE2_TEXT1])
    document.relations.extend([REL_JANE_LIVES_IN_BERLIN, REL_KARL_HAS_NOTHING])

    assert str(document.entities[0]) == "Jane"
    assert str(document.entities[1]) == " Berlin"
    assert str(document.entities[2]) == "Karl"
    assert str(document.entities[3]) == "   "
    assert str(document.partitions[0]) == "Jane lives in Berlin."
    assert str(document.partitions[1]) == "this is a truncated sentence about Karl\n   "

    assert str(document.relations[0].tail) == " Berlin"
    assert str(document.relations[0].head) == "Jane"
    assert str(document.relations[0].label) == "lives_in"
    assert str(document.relations[1].tail) == "   "
    assert str(document.relations[1].head) == "Karl"
    assert str(document.relations[1].label) == "has_nothing"

    return document


@pytest.mark.parametrize(
    "layer,skip_empty",
    [
        ("entities", False),
        ("partitions", False),
        ("partitions", True),
    ],
)
def test_text_span_trimmer(document1, layer, skip_empty):
    trimmer = TextSpanTrimmer(layer=layer, skip_empty=skip_empty)
    processed_document = trimmer(document1)

    assert len(document1.entities) == 4
    assert len(document1.relations) == 2
    assert len(processed_document.partitions) == len(document1.partitions) == 2

    if layer == "entities" and not skip_empty:
        assert len(processed_document.entities) == 4
        assert len(processed_document.relations) == 2
        assert str(processed_document.entities[0]) == "Jane"
        assert str(processed_document.entities[1]) == "Berlin"
        assert str(processed_document.entities[2]) == "Karl"
        assert str(processed_document.entities[3]) == ""
        assert str(processed_document.partitions[0]) == "Jane lives in Berlin."
        assert (
            str(processed_document.partitions[1]) == "this is a truncated sentence about Karl\n   "
        )
        assert str(processed_document.relations[0].tail) == "Berlin"
        assert str(processed_document.relations[0].head) == "Jane"
        assert str(processed_document.relations[0].label) == "lives_in"
        assert str(processed_document.relations[1].tail) == ""
        assert str(processed_document.relations[1].head) == "Karl"
        assert str(processed_document.relations[1].label) == "has_nothing"
    elif layer == "partitions":
        assert len(processed_document.entities) == 4
        assert str(processed_document.entities[0]) == "Jane"
        assert str(processed_document.entities[1]) == " Berlin"
        assert str(processed_document.entities[2]) == "Karl"
        assert str(processed_document.entities[3]) == "   "
        assert str(processed_document.partitions[0]) == "Jane lives in Berlin."
        assert str(processed_document.partitions[1]) == "this is a truncated sentence about Karl"
        assert str(processed_document.relations[0].tail) == " Berlin"
        assert str(processed_document.relations[0].head) == "Jane"
        assert str(processed_document.relations[0].label) == "lives_in"
        assert str(processed_document.relations[1].tail) == "   "
        assert str(processed_document.relations[1].head) == "Karl"
        assert str(processed_document.relations[1].label) == "has_nothing"
    else:
        raise ValueError(f"Unknown parameter combination: layer={layer}, skip_empty={skip_empty}")


def test_text_span_trimmer_remove_entity_of_relations(document1):
    trimmer = TextSpanTrimmer(layer="entities", skip_empty=True)
    with pytest.raises(ValueError) as excinfo:
        processed_document = trimmer(document1)
    assert (
        str(excinfo.value)
        == "Could not add annotation BinaryRelation(head=LabeledSpan(start=57, end=61, label='person', score=1.0), "
        "tail=LabeledSpan(start=62, end=65, label='other', score=1.0), label='has_nothing', score=1.0) "
        "to DocumentWithEntitiesRelationsAndPartitions because it depends on annotations that are not present "
        "in the document."
    )


@dataclasses.dataclass
class DocumentWithLabeledMultiSpansAndBinaryRelations(TextBasedDocument):
    labeled_multi_spans: AnnotationList[LabeledMultiSpan] = annotation_field(target="text")
    binary_relations: AnnotationList[BinaryRelation] = annotation_field(
        target="labeled_multi_spans"
    )


def test_text_span_trimmer_with_multi_spans():
    doc = DocumentWithLabeledMultiSpansAndBinaryRelations(
        text="Jane Doe lives in New The Big York."
    )
    jane_doe = LabeledMultiSpan(slices=((0, 4), (5, 9)), label="person", score=1.0)  # Jane Doe
    doc.labeled_multi_spans.append(jane_doe)
    assert str(jane_doe) == "('Jane', 'Doe ')"
    new_york = LabeledMultiSpan(slices=((17, 21), (30, 34)), label="city", score=0.9)  # New York
    doc.labeled_multi_spans.append(new_york)
    assert str(new_york) == "(' New', 'York')"
    lives_in = BinaryRelation(head=jane_doe, tail=new_york, label="lives_in", score=0.8)
    doc.binary_relations.append(lives_in)

    trimmer = TextSpanTrimmer(layer="labeled_multi_spans")
    processed_document = trimmer(doc)
    assert len(processed_document.labeled_multi_spans) == 2
    assert len(processed_document.binary_relations) == 1
    assert str(processed_document.labeled_multi_spans[0]) == "('Jane', 'Doe')"
    assert str(processed_document.labeled_multi_spans[1]) == "('New', 'York')"


@pytest.mark.parametrize("skip_empty,strict", [(True, True), (False, False)])
def test_text_span_trimmer_with_multi_spans_that_is_already_empty(skip_empty, strict):
    doc = DocumentWithLabeledMultiSpansAndBinaryRelations(
        text="Jane Doe lives in New The Big York."
    )
    empty = LabeledMultiSpan(slices=(), label="person")
    doc.labeled_multi_spans.append(empty)
    assert str(empty) == "()"
    new_york = LabeledMultiSpan(slices=((17, 21), (30, 34)), label="city", score=0.9)  # New York
    doc.labeled_multi_spans.append(new_york)
    assert str(new_york) == "(' New', 'York')"
    lives_in = BinaryRelation(head=empty, tail=new_york, label="lives_in", score=0.8)
    doc.binary_relations.append(lives_in)

    trimmer = TextSpanTrimmer(layer="labeled_multi_spans", skip_empty=skip_empty, strict=strict)
    if skip_empty and strict:
        with pytest.raises(ValueError) as excinfo:
            processed_document = trimmer(doc)
        assert (
            str(excinfo.value)
            == "Could not add annotation BinaryRelation(head=LabeledMultiSpan(slices=(), label='person', score=1.0), "
            "tail=LabeledMultiSpan(slices=((17, 21), (30, 34)), label='city', score=0.9), label='lives_in', score=0.8) "
            "to DocumentWithLabeledMultiSpansAndBinaryRelations because it depends on annotations that are not present "
            "in the document."
        )
    else:
        processed_document = trimmer(doc)
        if skip_empty:
            assert len(processed_document.labeled_multi_spans) == 1
            assert str(processed_document.labeled_multi_spans[0]) == "('New', 'York')"
            assert len(processed_document.binary_relations) == 0
        else:
            assert len(processed_document.labeled_multi_spans) == 2
            assert str(processed_document.labeled_multi_spans[0]) == "()"
            assert str(processed_document.labeled_multi_spans[1]) == "('New', 'York')"
            assert len(processed_document.binary_relations) == 1
            assert (
                processed_document.binary_relations[0].head
                == processed_document.labeled_multi_spans[0]
            )
            assert (
                processed_document.binary_relations[0].tail
                == processed_document.labeled_multi_spans[1]
            )
