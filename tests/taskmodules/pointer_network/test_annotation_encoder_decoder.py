import pytest
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span

from pie_modules.taskmodules.pointer_network.annotation_encoder_decoder import (
    BinaryRelationEncoderDecoder,
    LabeledSpanEncoderDecoder,
    SpanEncoderDecoder,
    SpanEncoderDecoderWithOffset,
)


@pytest.mark.parametrize("exclusive_end", [True, False])
def test_span_encoder_decoder(exclusive_end):
    """Test the SimpleSpanEncoderDecoder class."""

    encoder_decoder = SpanEncoderDecoder(exclusive_end)
    if exclusive_end:
        assert encoder_decoder.encode(Span(start=1, end=2)) == [1, 2]
        assert encoder_decoder.decode([1, 2]) == Span(start=1, end=2)
    else:
        assert encoder_decoder.encode(Span(start=1, end=2)) == [1, 1]
        assert encoder_decoder.decode([1, 1]) == Span(start=1, end=2)


@pytest.mark.parametrize("exclusive_end", [True, False])
def test_span_encoder_decoder_validate_encoding(exclusive_end):
    """Test the SimpleSpanEncoderDecoder class."""

    encoder_decoder = SpanEncoderDecoder(exclusive_end)

    if exclusive_end:
        assert encoder_decoder.validate_encoding([1, 2]) == set()
        assert encoder_decoder.validate_encoding([2, 1]) == {"order"}
        assert encoder_decoder.validate_encoding([1, 1]) == {"order"}
        assert encoder_decoder.validate_encoding([1]) == {"len"}
        assert encoder_decoder.validate_encoding([1, 2, 3]) == {"len"}
    else:
        assert encoder_decoder.validate_encoding([1, 1]) == set()
        assert encoder_decoder.validate_encoding([2, 1]) == {"order"}
        assert encoder_decoder.validate_encoding([1]) == {"len"}
        assert encoder_decoder.validate_encoding([1, 2, 3]) == {"len"}


def test_span_encoder_decoder_wrong_input():
    """Test the SimpleSpanEncoderDecoder class."""

    encoder_decoder = SpanEncoderDecoder()

    # test too many values
    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.decode([1, 2, 3])
    assert (
        str(excinfo.value)
        == "two values are required to decode as Span, but the encoding is: [1, 2, 3]"
    )

    # test too few values
    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.decode([1])
    assert (
        str(excinfo.value) == "two values are required to decode as Span, but the encoding is: [1]"
    )


def test_span_encoder_decoder_with_offset():
    """Test the SpanEncoderDecoderWithOffset class."""

    encoder_decoder = SpanEncoderDecoderWithOffset(offset=1)

    assert encoder_decoder.encode(Span(start=1, end=2)) == [2, 3]
    assert encoder_decoder.decode([2, 3]) == Span(start=1, end=2)


def test_span_encoder_decoder_with_offset_validate_encoding():
    """Test the SpanEncoderDecoderWithOffset class."""

    encoder_decoder = SpanEncoderDecoderWithOffset(offset=1)

    assert encoder_decoder.validate_encoding([2, 3]) == set()
    assert encoder_decoder.validate_encoding([3, 2]) == {"order"}
    assert encoder_decoder.validate_encoding([2, 2]) == {"order"}
    assert encoder_decoder.validate_encoding([2]) == {"len"}
    assert encoder_decoder.validate_encoding([2, 3, 4]) == {"len"}
    assert encoder_decoder.validate_encoding([0, 2]) == {"offset"}
    assert encoder_decoder.validate_encoding([0]) == {"len", "offset"}


@pytest.mark.parametrize("mode", ["indices_label", "label_indices"])
def test_labeled_span_encoder_decoder(mode):
    """Test the LabeledSpanEncoderDecoder class."""

    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode=mode,
    )

    if mode == "indices_label":
        assert encoder_decoder.encode(LabeledSpan(start=1, end=2, label="A")) == [3, 4, 0]
        assert encoder_decoder.decode([3, 4, 0]) == LabeledSpan(start=1, end=2, label="A")
    elif mode == "label_indices":
        assert encoder_decoder.encode(LabeledSpan(start=1, end=2, label="A")) == [0, 3, 4]
        assert encoder_decoder.decode([0, 3, 4]) == LabeledSpan(start=1, end=2, label="A")
    else:
        raise ValueError(f"unknown mode: {mode}")


@pytest.mark.parametrize("mode", ["indices_label", "label_indices"])
def test_labeled_span_encoder_decoder_validate_encoding(mode):
    """Test the LabeledSpanEncoderDecoder class."""

    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode=mode,
    )

    # Note: we first strip the label encoding and then validate the span encoding and the label encoding
    if mode == "indices_label":
        assert encoder_decoder.validate_encoding([3, 4, 0]) == set()
        assert encoder_decoder.validate_encoding([3, 4, 2]) == {"label"}
        assert encoder_decoder.validate_encoding([4, 3, 0]) == {"order"}
        assert encoder_decoder.validate_encoding([0, 3, 0]) == {"offset"}
        assert encoder_decoder.validate_encoding([3, 3, 2]) == {"order", "label"}
        assert encoder_decoder.validate_encoding([3, 0, 2]) == {"label", "offset"}
        assert encoder_decoder.validate_encoding([3, 0]) == {"len"}
        assert encoder_decoder.validate_encoding([3, 4, 5, 0]) == {"len"}
        assert encoder_decoder.validate_encoding([3, 2]) == {"label", "len"}
        assert encoder_decoder.validate_encoding([0, 2]) == {"label", "len", "offset"}
    elif mode == "label_indices":
        assert encoder_decoder.validate_encoding([0, 3, 4]) == set()
        assert encoder_decoder.validate_encoding([2, 3, 4]) == {"label"}
        assert encoder_decoder.validate_encoding([0, 4, 3]) == {"order"}
        assert encoder_decoder.validate_encoding([0, 3, 0]) == {"offset"}
        assert encoder_decoder.validate_encoding([2, 0, 3]) == {"label", "offset"}
        assert encoder_decoder.validate_encoding([0, 3]) == {"len"}
        assert encoder_decoder.validate_encoding([0, 3, 4, 5]) == {"len"}
        assert encoder_decoder.validate_encoding([2, 3]) == {"label", "len"}
        assert encoder_decoder.validate_encoding([0, 0]) == {"len", "offset"}
        assert encoder_decoder.validate_encoding([2, 0]) == {"label", "len", "offset"}
    else:
        raise ValueError(f"unknown mode: {mode}")


def test_labeled_span_encoder_decoder_unknown_mode():
    """Test the LabeledSpanEncoderDecoder class."""

    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="unknown",
    )
    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.encode(LabeledSpan(start=1, end=2, label="A"))
    assert str(excinfo.value) == "unknown mode: unknown"

    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.decode([0, 3, 4])
    assert str(excinfo.value) == "unknown mode: unknown"


@pytest.mark.parametrize(
    "mode", ["head_tail_label", "tail_head_label", "label_head_tail", "label_tail_head"]
)
def test_binary_relation_encoder_decoder(mode):
    """Test the BinaryRelationEncoderDecoder class."""

    label2id = {"A": 0, "B": 1, "C": 2}
    labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=labeled_span_encoder_decoder,
        tail_encoder_decoder=labeled_span_encoder_decoder,
        label2id=label2id,
        mode=mode,
    )

    if mode == "head_tail_label":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=3, end=4, label="B"),
                label="C",
            )
        ) == [4, 5, 0, 6, 7, 1, 2]
        assert encoder_decoder.decode([4, 5, 0, 6, 7, 1, 2]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=3, end=4, label="B"),
            label="C",
        )
    elif mode == "tail_head_label":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=3, end=4, label="B"),
                label="C",
            )
        ) == [6, 7, 1, 4, 5, 0, 2]
        assert encoder_decoder.decode([6, 7, 1, 4, 5, 0, 2]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=3, end=4, label="B"),
            label="C",
        )
    elif mode == "label_head_tail":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=3, end=4, label="B"),
                label="C",
            )
        ) == [2, 4, 5, 0, 6, 7, 1]
        assert encoder_decoder.decode([2, 4, 5, 0, 6, 7, 1]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=3, end=4, label="B"),
            label="C",
        )
    elif mode == "label_tail_head":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=3, end=4, label="B"),
                label="C",
            )
        ) == [2, 6, 7, 1, 4, 5, 0]
        assert encoder_decoder.decode([2, 6, 7, 1, 4, 5, 0]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=3, end=4, label="B"),
            label="C",
        )


@pytest.mark.parametrize(
    "mode", ["head_tail_label", "tail_head_label", "label_head_tail", "label_tail_head"]
)
def test_binary_relation_encoder_decoder_loop_relation(mode):
    """Test the BinaryRelationEncoderDecoder class."""

    # we use different label2id for head and tail to test the case where the head and tail
    # have different label sets
    head_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=3),
        label2id={"A": 1, "B": 2},
        mode="indices_label",
    )
    tail_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=3),
        label2id={"A": -1, "B": -2},
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=head_encoder_decoder,
        tail_encoder_decoder=tail_encoder_decoder,
        label2id={"N": 3},
        mode=mode,
        loop_dummy_relation_name="L",
        none_label="N",
    )

    if mode == "head_tail_label":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=1, end=2, label="A"),
                label="L",
            )
        ) == [4, 5, 1, 3, 3, 3, 3]
        assert encoder_decoder.decode([4, 5, 1, 3, 3, 3, 3]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=1, end=2, label="A"),
            label="L",
        )
    elif mode == "tail_head_label":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=1, end=2, label="A"),
                label="L",
            )
        ) == [4, 5, -1, 3, 3, 3, 3]
        assert encoder_decoder.decode([4, 5, -1, 3, 3, 3, 3]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=1, end=2, label="A"),
            label="L",
        )
    elif mode == "label_head_tail":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=1, end=2, label="A"),
                label="L",
            )
        ) == [3, 4, 5, 1, 3, 3, 3]
        assert encoder_decoder.decode([3, 4, 5, 1, 3, 3, 3]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=1, end=2, label="A"),
            label="L",
        )
    elif mode == "label_tail_head":
        assert encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=1, end=2, label="A"),
                label="L",
            )
        ) == [3, 4, 5, -1, 3, 3, 3]
        assert encoder_decoder.decode([3, 4, 5, -1, 3, 3, 3]) == BinaryRelation(
            head=LabeledSpan(start=1, end=2, label="A"),
            tail=LabeledSpan(start=1, end=2, label="A"),
            label="L",
        )
    else:
        raise ValueError(f"unknown mode: {mode}")


@pytest.mark.parametrize(
    "loop_dummy_relation_name,none_label",
    [("L", None), (None, "N")],
)
def test_binary_relation_encoder_decoder_only_loop_or_none_label_provided(
    loop_dummy_relation_name, none_label
):
    """Test the BinaryRelationEncoderDecoder class."""

    label2id = {"A": 0, "B": 1, "N": 2}
    labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=labeled_span_encoder_decoder,
        tail_encoder_decoder=labeled_span_encoder_decoder,
        label2id=label2id,
        mode="head_tail_label",
        loop_dummy_relation_name=loop_dummy_relation_name,
        none_label=none_label,
    )

    if loop_dummy_relation_name is not None:
        with pytest.raises(ValueError) as excinfo:
            encoder_decoder.encode(
                BinaryRelation(
                    head=LabeledSpan(start=1, end=2, label="A"),
                    tail=LabeledSpan(start=1, end=2, label="A"),
                    label=loop_dummy_relation_name,
                )
            )

        assert (
            str(excinfo.value)
            == "loop_dummy_relation_name is set, but none_label is not set: None"
        )
    elif none_label is not None:
        none_id = label2id[none_label]
        with pytest.raises(ValueError) as excinfo:
            encoder_decoder.decode([4, 5, 1, none_id, none_id, none_id, none_id])
        assert (
            str(excinfo.value)
            == "loop_dummy_relation_name is not set, but none_label=N was found in decoded encoding: "
            "[4, 5, 1, 2, 2, 2, 2] (label2id: {'A': 0, 'B': 1, 'N': 2}))"
        )
    else:
        raise ValueError("unknown setting")


@pytest.mark.parametrize(
    "loop_dummy_relation_name,none_label",
    [(None, None), ("L", "N")],
)
def test_binary_relation_encoder_decoder_unknown_mode(loop_dummy_relation_name, none_label):
    """Test the BinaryRelationEncoderDecoder class."""

    label2id = {"A": 0, "B": 1, "N": 2, "L": 3}
    labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=labeled_span_encoder_decoder,
        tail_encoder_decoder=labeled_span_encoder_decoder,
        label2id=label2id,
        mode="unknown",
        loop_dummy_relation_name=loop_dummy_relation_name,
        none_label=none_label,
    )
    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.encode(
            BinaryRelation(
                head=LabeledSpan(start=1, end=2, label="A"),
                tail=LabeledSpan(start=1, end=2, label="A"),
                label="L",
            )
        )
    assert str(excinfo.value) == "unknown mode: unknown"

    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.decode([2, 2, 2, 2, 2, 2, 2])
    assert str(excinfo.value) == "unknown mode: unknown"


def test_binary_relation_encoder_decoder_wrong_encoding_size():
    """Test the BinaryRelationEncoderDecoder class."""

    label2id = {"A": 0, "B": 1, "C": 2}
    labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=labeled_span_encoder_decoder,
        tail_encoder_decoder=labeled_span_encoder_decoder,
        label2id=label2id,
        mode="head_tail_label",
    )
    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.decode([1, 2, 3, 4, 5, 6])
    assert (
        str(excinfo.value)
        == "seven values are required to decode as BinaryRelation, but the encoding is: [1, 2, 3, 4, 5, 6]"
    )

    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.decode([1, 2, 3, 4, 5, 6, 7, 8])
    assert (
        str(excinfo.value)
        == "seven values are required to decode as BinaryRelation, but the encoding is: [1, 2, 3, 4, 5, 6, 7, 8]"
    )


@pytest.mark.parametrize(
    "mode", ["head_tail_label", "tail_head_label", "label_head_tail", "label_tail_head"]
)
def test_binary_relation_encoder_decoder_validate_encoding(mode):
    """Test the BinaryRelationEncoderDecoder class."""

    label2id = {"A": 0, "B": 1, "C": 2}
    labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=labeled_span_encoder_decoder,
        tail_encoder_decoder=labeled_span_encoder_decoder,
        label2id=label2id,
        mode=mode,
    )

    if mode in ["head_tail_label", "tail_head_label"]:
        assert encoder_decoder.validate_encoding([4, 5, 0, 6, 7, 1, 2]) == set()
        assert encoder_decoder.validate_encoding([4, 5, 0, 6, 7, 1, 3]) == {"label"}
        assert encoder_decoder.validate_encoding([4, 5, 0, 6, 7, 3, 2]) == {"label"}
        assert encoder_decoder.validate_encoding([1, 5, 0, 6, 7, 2, 2]) == {"offset"}
        assert encoder_decoder.validate_encoding([5, 4, 0, 6, 7, 1, 2]) == {"order"}
        assert encoder_decoder.validate_encoding([5, 4, 0, 0, 7, 3, 2]) == {
            "order",
            "label",
            "offset",
        }
        assert encoder_decoder.validate_encoding([5, 4, 0, 6, 7, 1, 2, 3]) == {"len"}
        assert encoder_decoder.validate_encoding([4, 5, 0, 4, 7, 1, 2]) == {"overlap"}
    elif mode in ["label_head_tail", "label_tail_head"]:
        assert encoder_decoder.validate_encoding([2, 4, 5, 0, 6, 7, 1]) == set()
        assert encoder_decoder.validate_encoding([3, 4, 5, 0, 6, 7, 1]) == {"label"}
        assert encoder_decoder.validate_encoding([2, 4, 5, 0, 6, 7, 3]) == {"label"}
        assert encoder_decoder.validate_encoding([2, 1, 5, 0, 6, 7, 2]) == {"offset"}
        assert encoder_decoder.validate_encoding([2, 5, 4, 0, 6, 7, 1]) == {"order"}
        assert encoder_decoder.validate_encoding([2, 5, 4, 0, 0, 7, 3]) == {
            "order",
            "label",
            "offset",
        }
        assert encoder_decoder.validate_encoding([3, 5, 4, 0, 6, 7, 1, 2]) == {"len"}
        assert encoder_decoder.validate_encoding([2, 4, 5, 0, 4, 7, 1]) == {"overlap"}
    else:
        raise ValueError(f"unknown mode: {mode}")
