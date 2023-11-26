import dataclasses

import pytest
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument

from pie_models.document.processing import TextSpanTrimmer


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
