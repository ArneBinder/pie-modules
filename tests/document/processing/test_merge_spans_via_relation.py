import pytest

from pie_modules.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pie_modules.document.processing.merge_spans_via_relation import (
    _merge_spans_via_relation,
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
