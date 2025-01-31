import pytest
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

from pie_modules.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pie_modules.document.processing import SpansViaRelationMerger
from pie_modules.document.processing.merge_spans_via_relation import (
    _merge_spans_via_relation,
)
from pie_modules.documents import (
    TextDocumentWithLabeledMultiSpansAndBinaryRelations,
    TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
    TextDocumentWithLabeledSpansAndBinaryRelations,
)


@pytest.mark.parametrize(
    "create_multi_spans",
    [False, True],
)
def test_merge_spans_via_relation(create_multi_spans: bool):
    # we have 6 spans and 4 relations
    # spans 0, 2, 4 are connected via "link" relation, so they should be merged
    # spans 3, 5 are connected via "link" relation, but they do not have the same label,
    #   so they should not be merged. But the relation should be removed
    # spans 0, 3 are connected via "relation_x" relation, its head should be remapped to the new span
    spans = [
        LabeledSpan(start=0, end=1, label="label_a"),
        LabeledSpan(start=2, end=3, label="other"),
        LabeledSpan(start=4, end=5, label="label_a"),
        LabeledSpan(start=6, end=7, label="label_b"),
        LabeledSpan(start=8, end=9, label="label_a"),
        LabeledSpan(start=10, end=11, label="label_c"),
    ]
    relations = [
        BinaryRelation(head=spans[0], tail=spans[2], label="link"),
        BinaryRelation(head=spans[0], tail=spans[3], label="relation_x"),
        BinaryRelation(head=spans[2], tail=spans[4], label="link"),
        BinaryRelation(head=spans[3], tail=spans[5], label="link"),
    ]

    merged_spans, merged_relations = _merge_spans_via_relation(
        spans=spans,
        relations=relations,
        link_relation_label="link",
        create_multi_spans=create_multi_spans,
    )
    if create_multi_spans:
        head = LabeledMultiSpan(
            slices=(
                (0, 1),
                (4, 5),
                (8, 9),
            ),
            label="label_a",
        )
        tail = LabeledMultiSpan(slices=((6, 7),), label="label_b")
        assert merged_spans == {
            head,
            LabeledMultiSpan(slices=((2, 3),), label="other"),
            tail,
            LabeledMultiSpan(slices=((10, 11),), label="label_c"),
        }
    else:
        head = LabeledSpan(start=0, end=9, label="label_a")
        tail = LabeledSpan(start=6, end=7, label="label_b")
        assert merged_spans == {
            head,
            tail,
            LabeledSpan(start=2, end=3, label="other"),
            LabeledSpan(start=10, end=11, label="label_c"),
        }
    assert merged_relations == {BinaryRelation(head=head, tail=tail, label="relation_x")}


def sort_spans(spans):
    if len(spans) == 0:
        return []
    if isinstance(spans[0], LabeledSpan):
        return sorted(spans, key=lambda span: (span.start, span.end, span.label))
    else:
        return sorted(spans, key=lambda span: (span.slices, span.label))


def resolve_spans(spans):
    if len(spans) == 0:
        return []
    if isinstance(spans[0], LabeledSpan):
        return [(span.target[span.start : span.end], span.label) for span in spans]
    else:
        return [
            (tuple(span.target[start:end] for start, end in span.slices), span.label)
            for span in spans
        ]


@pytest.mark.parametrize("create_multi_spans", [False, True])
def test_spans_via_relation_merger(create_multi_spans):
    doc = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
        text="This text, however, is about nothing (see here)."
    )
    doc.labeled_partitions.append(LabeledSpan(start=0, end=48, label="sentence"))
    assert str(doc.labeled_partitions[0]) == "This text, however, is about nothing (see here)."
    doc.labeled_spans.extend(
        [
            LabeledSpan(start=0, end=9, label="claim"),
            LabeledSpan(start=11, end=18, label="other"),
            LabeledSpan(start=20, end=36, label="claim"),
            LabeledSpan(start=38, end=46, label="data"),
        ]
    )
    assert str(doc.labeled_spans[0]) == "This text"
    assert str(doc.labeled_spans[1]) == "however"
    assert str(doc.labeled_spans[2]) == "is about nothing"
    assert str(doc.labeled_spans[3]) == "see here"
    doc.binary_relations.extend(
        [
            BinaryRelation(head=doc.labeled_spans[0], tail=doc.labeled_spans[2], label="link"),
            BinaryRelation(head=doc.labeled_spans[3], tail=doc.labeled_spans[2], label="support"),
        ]
    )
    # after merging, that should be the same as in the gold data
    doc.binary_relations.predictions.extend(
        [
            BinaryRelation(head=doc.labeled_spans[0], tail=doc.labeled_spans[2], label="link"),
            BinaryRelation(head=doc.labeled_spans[3], tail=doc.labeled_spans[0], label="support"),
        ]
    )

    processor = SpansViaRelationMerger(
        relation_layer="binary_relations",
        link_relation_label="link",
        use_predicted_spans=False,
        create_multi_spans=create_multi_spans,
        result_document_type=(
            TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions
            if create_multi_spans
            else None
        ),
        result_field_mapping={
            "labeled_spans": "labeled_multi_spans" if create_multi_spans else "labeled_spans",
            "labeled_partitions": "labeled_partitions",
        },
    )
    result = processor(doc)
    if create_multi_spans:
        assert isinstance(
            result, TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions
        )
        sorted_spans = sort_spans(result.labeled_multi_spans)
        sorted_spans_resolved = resolve_spans(sorted_spans)
        assert sorted_spans_resolved == [
            (("This text", "is about nothing"), "claim"),
            (("however",), "other"),
            (("see here",), "data"),
        ]
    else:
        assert isinstance(result, TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions)
        sorted_spans = sort_spans(result.labeled_spans)
        sorted_spans_resolved = resolve_spans(sorted_spans)
        assert sorted_spans_resolved == [
            ("This text, however, is about nothing", "claim"),
            ("however", "other"),
            ("see here", "data"),
        ]
    # check gold and predicted relations
    for relations in [result.binary_relations, result.binary_relations.predictions]:
        assert len(relations) == 1
        assert relations[0].head == sorted_spans[2]
        assert relations[0].tail == sorted_spans[0]
        assert relations[0].label == "support"

    # check the labeled partitions
    assert len(result.labeled_partitions) == 1
    assert str(result.labeled_partitions[0]) == "This text, however, is about nothing (see here)."


def test_spans_via_relation_merger_create_multi_span_missing_result_document_type():
    with pytest.raises(ValueError) as exc_info:
        SpansViaRelationMerger(
            relation_layer="binary_relations",
            link_relation_label="link",
            create_multi_spans=True,
        )
    assert (
        str(exc_info.value) == "result_document_type must be set when create_multi_spans is True"
    )


def test_spans_via_relation_merger_with_result_document_type_missing_result_field_mapping():
    with pytest.raises(ValueError) as exc_info:
        SpansViaRelationMerger(
            relation_layer="binary_relations",
            link_relation_label="link",
            result_document_type=TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
        )
    assert (
        str(exc_info.value)
        == "result_field_mapping must be set when result_document_type is provided"
    )
