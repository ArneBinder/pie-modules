import pytest
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span

from pie_modules.annotations import LabeledMultiSpan
from pie_modules.taskmodules.pointer_network.annotation_encoder_decoder import (
    BinaryRelationEncoderDecoder,
    DecodingEmptySpanException,
    DecodingLabelException,
    DecodingLengthException,
    DecodingNegativeIndexException,
    DecodingOrderException,
    DecodingSpanNestedException,
    DecodingSpanOverlapException,
    EncodingEmptySlicesException,
    EncodingEmptySpanException,
    IncompleteEncodingException,
    LabeledMultiSpanEncoderDecoder,
    LabeledSpanEncoderDecoder,
    SpanEncoderDecoder,
    SpanEncoderDecoderWithOffset,
    spans_are_nested,
    spans_have_overlap,
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


def test_span_encoder_decoder_empty_span():
    encoder_decoder = SpanEncoderDecoder()
    with pytest.raises(EncodingEmptySpanException) as excinfo:
        encoder_decoder.encode(Span(start=1, end=1))
    assert (
        str(excinfo.value)
        == "can not encode empty Span annotations, i.e. where the start index equals the end index"
    )


def test_span_encoder_decoder_wrong_length():
    """Test the SimpleSpanEncoderDecoder class."""

    encoder_decoder = SpanEncoderDecoder()
    with pytest.raises(DecodingLengthException) as excinfo:
        encoder_decoder.decode([1])
    assert (
        str(excinfo.value)
        == "two values are required to decode as Span, but encoding has length 1"
    )
    assert excinfo.value.identifier == "len"

    with pytest.raises(DecodingLengthException) as excinfo:
        encoder_decoder.decode([1, 2, 3])
    assert (
        str(excinfo.value)
        == "two values are required to decode as Span, but encoding has length 3"
    )
    assert excinfo.value.identifier == "len"


def test_span_encoder_decoder_wrong_order():
    """Test the SimpleSpanEncoderDecoder class."""

    encoder_decoder = SpanEncoderDecoder()

    with pytest.raises(DecodingOrderException) as excinfo:
        encoder_decoder.decode([3, 2])
    assert (
        str(excinfo.value)
        == "end index can not be smaller than start index, but got: start=3, end=2"
    )
    assert excinfo.value.identifier == "order"

    # zero-length span
    span = encoder_decoder.decode([1, 1])
    assert span is not None


def test_span_encoder_decoder_wrong_offset():
    """Test the SimpleSpanEncoderDecoder class."""

    encoder_decoder = SpanEncoderDecoder()

    with pytest.raises(DecodingNegativeIndexException) as excinfo:
        encoder_decoder.decode([-1, 2])
    assert str(excinfo.value) == "indices must be positive, but got: [-1, 2]"
    assert excinfo.value.identifier == "index"


@pytest.mark.parametrize("exclusive_end", [True, False])
def test_span_encoder_decoder_parse(exclusive_end):
    encoder_decoder = SpanEncoderDecoder(exclusive_end)
    if exclusive_end:
        assert encoder_decoder.parse([1, 2, 3, 4], [], 5) == (Span(start=1, end=2), [3, 4])
    else:
        assert encoder_decoder.parse([1, 1, 3, 4], [], 5) == (Span(start=1, end=2), [3, 4])


@pytest.mark.parametrize("exclusive_end", [True, False])
def test_span_encoder_decoder_parse_empty_span(exclusive_end):
    encoder_decoder = SpanEncoderDecoder(exclusive_end)
    encoding = [3, 3, 3, 4] if exclusive_end else [3, 2, 3, 4]
    with pytest.raises(DecodingEmptySpanException) as excinfo:
        encoder_decoder.parse(encoding, [], 5)
    assert excinfo.value.identifier == "empty_span"
    assert (
        str(excinfo.value)
        == "end index can not be equal to start index to decode as Span, but got: start=3, end=3"
    )
    assert excinfo.value.remaining == [3, 4]


@pytest.mark.parametrize("exclusive_end", [True, False])
def test_span_encoder_decoder_parse_wrong_order(exclusive_end):
    encoder_decoder = SpanEncoderDecoder(exclusive_end)
    encoding = [3, 2, 3, 4] if exclusive_end else [3, 1, 3, 4]
    with pytest.raises(DecodingOrderException) as excinfo:
        encoder_decoder.parse(encoding, [], 5)
    assert excinfo.value.identifier == "order"
    assert (
        str(excinfo.value)
        == "end index can not be smaller than start index, but got: start=3, end=2"
    )
    assert excinfo.value.remaining == [3, 4]


@pytest.mark.parametrize("exclusive_end", [True, False])
def test_span_encoder_decoder_parse_negative_index(exclusive_end):
    encoder_decoder = SpanEncoderDecoder(exclusive_end)
    encoding = [-1, 2, 3, 4] if exclusive_end else [-1, 1, 3, 4]
    with pytest.raises(DecodingNegativeIndexException) as excinfo:
        encoder_decoder.parse(encoding, [], 5)
    assert excinfo.value.identifier == "index"
    assert str(excinfo.value) == "indices must be positive, but got: start=-1, end=2"
    assert excinfo.value.remaining == [3, 4]


def test_spans_are_nested():
    # fully nested
    assert spans_are_nested(Span(start=1, end=4), Span(start=2, end=3))
    assert spans_are_nested(Span(start=2, end=3), Span(start=1, end=4))
    # nested with same start
    assert spans_are_nested(Span(start=1, end=3), Span(start=1, end=2))
    assert spans_are_nested(Span(start=1, end=2), Span(start=1, end=3))
    # nested with same end
    assert spans_are_nested(Span(start=2, end=4), Span(start=3, end=4))
    assert spans_are_nested(Span(start=3, end=4), Span(start=2, end=4))
    # nested with same start and end
    assert spans_are_nested(Span(start=1, end=3), Span(start=1, end=3))

    # not nested
    assert not spans_are_nested(Span(start=1, end=2), Span(start=3, end=4))
    # not nested, but touching
    assert not spans_are_nested(Span(start=1, end=2), Span(start=2, end=3))
    # not nested, but overlap
    assert not spans_are_nested(Span(start=1, end=3), Span(start=2, end=4))


def test_spans_have_overlap():
    # overlap, no touching
    assert spans_have_overlap(Span(start=1, end=3), Span(start=2, end=4))
    assert spans_have_overlap(Span(start=2, end=4), Span(start=1, end=3))
    # overlap, same start
    assert spans_have_overlap(Span(start=1, end=2), Span(start=1, end=3))
    assert spans_have_overlap(Span(start=1, end=3), Span(start=1, end=2))
    # overlap, same end
    assert spans_have_overlap(Span(start=2, end=3), Span(start=1, end=3))
    assert spans_have_overlap(Span(start=1, end=3), Span(start=2, end=3))
    # no overlap, not touching
    assert not spans_have_overlap(Span(start=1, end=2), Span(start=3, end=4))
    assert not spans_have_overlap(Span(start=3, end=4), Span(start=1, end=2))
    # no overlap, touching
    assert not spans_have_overlap(Span(start=1, end=2), Span(start=2, end=3))
    assert not spans_have_overlap(Span(start=2, end=3), Span(start=1, end=2))


@pytest.mark.parametrize("allow_nested", [True, False])
def test_span_encoder_decoder_parse_with_previous_annotations(allow_nested):
    encoder_decoder = SpanEncoderDecoder(allow_nested=allow_nested)
    expected_span = Span(start=1, end=3)
    remaining_encoding = [3, 4]
    # encoding of the expected span + remaining encoding
    encoding = [1, 3] + remaining_encoding
    other_span = Span(start=3, end=4)
    nested_span = Span(start=2, end=3)
    overlapping_span = Span(start=2, end=4)
    # other_span should not pose a problem in any case
    assert encoder_decoder.parse(encoding, [other_span], 5) == (expected_span, remaining_encoding)
    # nested_span should only be allowed if allow_nested=True
    if allow_nested:
        assert encoder_decoder.parse(encoding, [nested_span], 5) == (
            expected_span,
            remaining_encoding,
        )
    else:
        with pytest.raises(DecodingSpanNestedException) as excinfo:
            encoder_decoder.parse(encoding, [nested_span], 5)
        assert (
            str(excinfo.value) == f"the encoded span is nested in another span: {nested_span}. "
            f"You can set allow_nested=True to allow nested spans."
        )
        assert excinfo.value.remaining == remaining_encoding
    # overlapping_span is not allowed in any case
    with pytest.raises(DecodingSpanOverlapException) as excinfo:
        encoder_decoder.parse(encoding, [overlapping_span], 5)
    assert str(excinfo.value) == f"the encoded span overlaps with another span: {overlapping_span}"
    assert excinfo.value.remaining == remaining_encoding


@pytest.mark.parametrize(
    "exclusive_end,allow_nested", [(False, False), (False, True), (True, False), (True, True)]
)
def test_span_encoder_decoder_parse_incomplete_0(exclusive_end, allow_nested):
    encoder_decoder = SpanEncoderDecoder(exclusive_end=exclusive_end, allow_nested=allow_nested)
    # no previous annotations
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse([], [], 6)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    assert excinfo.value.follow_up_candidates == [0, 1, 2, 3, 4, 5]
    # previous annotation
    other_span = Span(start=2, end=4)
    encoded_other_span = encoder_decoder.encode(other_span)
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse([], [other_span], 6)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    if allow_nested:
        assert excinfo.value.follow_up_candidates == [0, 1, 2, 3, 4, 5]
    else:
        if exclusive_end:
            assert encoded_other_span == [2, 4]
        else:
            assert encoded_other_span == [2, 3]
        # index 3 is excluded because they are covered by the other_span
        # Note that index 2 is not included, despite being covered by the other_span,
        # because we allow to generate the exact same span again.
        assert excinfo.value.follow_up_candidates == [0, 1, 2, 4, 5]


@pytest.mark.parametrize(
    "allow_nested,exclusive_end", [(False, False), (False, True), (True, False), (True, True)]
)
def test_span_encoder_decoder_parse_incomplete_1(allow_nested, exclusive_end):
    encoder_decoder = SpanEncoderDecoder(exclusive_end=exclusive_end, allow_nested=allow_nested)
    encoding = [1]

    # no previous annotations
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding, [], 6)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect an end index, so the follow-up candidates ...
    if exclusive_end:
        # are bigger than 1, but smaller still in the range of the text length (equal or smaller than 5)
        assert excinfo.value.follow_up_candidates == [2, 3, 4, 5, 6]
    else:
        # bigger or equal to 1, but smaller still in the range of the text length (smaller than 5)
        assert excinfo.value.follow_up_candidates == [1, 2, 3, 4, 5]

    # previous annotations
    # a span before the current start index should not affect the follow-up candidates
    other_span_before = Span(start=0, end=1)
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding, [other_span_before], 6)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    if exclusive_end:
        assert excinfo.value.follow_up_candidates == [2, 3, 4, 5, 6]
    else:
        assert excinfo.value.follow_up_candidates == [1, 2, 3, 4, 5]

    # a span after the current start index should limit the follow-up candidates
    other_span_after = Span(start=2, end=4)
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding, [other_span_after], 6)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    if allow_nested:
        if exclusive_end:
            # 3 as end index is excluded because the resulting span [1, 3)
            # would have an overlap with the other_span_after
            assert excinfo.value.follow_up_candidates == [2, 4, 5, 6]
        else:
            # 2 as end index is excluded because the resulting span [1, 3)
            # would have an overlap with the other_span_after
            assert excinfo.value.follow_up_candidates == [1, 3, 4, 5]
    else:
        if exclusive_end:
            assert excinfo.value.follow_up_candidates == [2]
        else:
            assert excinfo.value.follow_up_candidates == [1]

    nesting_span = Span(start=0, end=2)
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding, [nesting_span], 6)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    if allow_nested:
        if exclusive_end:
            # only the span [1, 2) is allowed, so only 2 is a valid follow-up candidate
            assert excinfo.value.follow_up_candidates == [2]
        else:
            # only the span [1, 2) is allowed, so only 1 is a valid follow-up candidate
            assert excinfo.value.follow_up_candidates == [1]
    else:
        # We generate the exact same span again, so the follow-up candidates are the same as for the previous case
        if exclusive_end:
            assert excinfo.value.follow_up_candidates == [2]
        else:
            assert excinfo.value.follow_up_candidates == [1]


def test_span_encoder_decoder_with_offset():
    """Test the SpanEncoderDecoderWithOffset class."""

    encoder_decoder = SpanEncoderDecoderWithOffset(offset=1)

    assert encoder_decoder.encode(Span(start=1, end=2)) == [2, 3]
    assert encoder_decoder.decode([2, 3]) == Span(start=1, end=2)


def test_span_encoder_decoder_with_offset_parse():
    """Test the SpanEncoderDecoderWithOffset class."""
    encoder_decoder = SpanEncoderDecoderWithOffset(offset=1)
    expected_span = Span(start=1, end=3)
    encoded_span = encoder_decoder.encode(expected_span)

    # test without remaining encoding
    assert encoder_decoder.parse(encoded_span, [], 6) == (expected_span, [])

    # test with remaining encoding
    remaining_encoding = [3, 4]
    assert encoder_decoder.parse(encoded_span + remaining_encoding, [], 6) == (
        expected_span,
        remaining_encoding,
    )


def test_span_encoder_decoder_with_offset_parse_incomplete():
    encoder_decoder = SpanEncoderDecoderWithOffset(offset=1)
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse([2], [], 6)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    assert excinfo.value.follow_up_candidates == [3, 4, 5, 6, 7]


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
def test_labeled_span_encoder_decoder_wrong_label_encoding(mode):
    """Test the LabeledSpanEncoderDecoder class."""

    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode=mode,
    )

    if mode == "indices_label":
        with pytest.raises(DecodingLabelException) as excinfo:
            encoder_decoder.decode([2, 3, 4])
    elif mode == "label_indices":
        with pytest.raises(DecodingLabelException) as excinfo:
            encoder_decoder.decode([4, 2, 3])
    assert str(excinfo.value) == "unknown label id: 4 (label2id: {'A': 0, 'B': 1})"
    assert excinfo.value.identifier == "label"


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


@pytest.mark.parametrize("mode", ["indices_label", "label_indices"])
def test_labeled_span_encoder_decoder_parse(mode):
    """Test the LabeledSpanEncoderDecoder class."""

    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode=mode,
    )
    expected_span = LabeledSpan(start=1, end=3, label="A")
    encoded_span = encoder_decoder.encode(expected_span)
    remaining_encoding = [3, 4]
    # encoding of the expected span + remaining encoding
    encoding = encoded_span + remaining_encoding
    assert encoder_decoder.parse(encoding, [], 6) == (expected_span, remaining_encoding)


def test_labeled_span_encoder_decoder_parse_unknown_mode():
    """Test the LabeledSpanEncoderDecoder class."""

    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode="unknown",
    )
    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.parse([0, 3, 4], [], 6)
    assert str(excinfo.value) == "unknown mode: unknown"


@pytest.mark.parametrize("mode", ["label_indices", "indices_label"])
def test_labeled_span_encoder_decoder_parse_incomplete(mode):
    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
        mode=mode,
    )
    if mode == "label_indices":
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse([], [], 6)
        assert str(excinfo.value) == "the encoding has not enough values to decode as LabeledSpan"
        assert excinfo.value.follow_up_candidates == [0, 1]
    elif mode == "indices_label":
        encoded_span = encoder_decoder.span_encoder_decoder.encode(Span(start=1, end=2))
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoded_span, [], 6)
        assert str(excinfo.value) == "the encoding has not enough values to decode as LabeledSpan"
        assert excinfo.value.follow_up_candidates == [0, 1]
    else:
        raise ValueError(f"unknown mode: {mode}")


def test_labeled_multi_span_encoder_decoder():
    """Test the LabeledMultiSpanEncoderDecoder class."""

    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledMultiSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
    )
    # encode and decode a single span with two slices and label A
    span = LabeledMultiSpan(slices=((1, 2), (4, 6)), label="A")
    encoding = [3, 4, 6, 8, 0]
    assert encoder_decoder.encode(span) == encoding
    assert encoder_decoder.decode(encoding) == span

    # encoding empty slices are not allowed
    with pytest.raises(EncodingEmptySlicesException) as excinfo:
        encoder_decoder.encode(LabeledMultiSpan(slices=(), label="A"))
    assert str(excinfo.value) == "LabeledMultiSpan must have at least one slice to encode it."

    # decoding an odd number of encoding entries is required for decoding
    with pytest.raises(DecodingLengthException) as excinfo:
        encoder_decoder.decode([3, 4, 6, 8])
    assert (
        str(excinfo.value)
        == "an odd number of encoding entries is required for decoding a LabeledMultiSpan, but got 4"
    )


def test_labeled_multi_span_encoder_decoder_parse():
    """Test the LabeledMultiSpanEncoderDecoder class."""

    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledMultiSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
    )
    expected_span = LabeledMultiSpan(slices=((1, 2), (4, 6)), label="A")
    encoding = [3, 4, 6, 8, 0]
    remaining_encoding = [0, 1]
    # encoding of the expected span + remaining encoding
    assert encoder_decoder.parse(encoding + remaining_encoding, [], 10) == (
        expected_span,
        remaining_encoding,
    )


def test_labeled_multi_span_encoder_decoder_parse_incomplete():
    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledMultiSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
    )
    encoding = encoder_decoder.encode(LabeledMultiSpan(slices=((1, 2), (4, 6)), label="A"))
    assert encoding == [3, 4, 6, 8, 0]
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse([], [], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect a start index, so the follow-up candidates are [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # but we need to add the offset of 2
    assert excinfo.value.follow_up_candidates == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding[:1], [], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect an end index of a none-empty Span, i.e. [2, 3, 4, 5, 6, 7, 8, 9, 10],
    # but we need to add the offset of 2
    assert excinfo.value.follow_up_candidates == [4, 5, 6, 7, 8, 9, 10, 11, 12]

    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding[:2], [], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as LabeledMultiSpan"
    # we can follow up with 1) a label encoding, i.e. [0, 1], or 2) the start index of a new span encoding
    # not covered by the previous slice which is Span(1, 2), i.e. [0] + [2, 3, 4, 5, 6, 7, 8, 9, 10],
    # but we also allow to generate the exact same span again, so the index 1 is also allowed
    # and we need to add the offset of 2
    assert excinfo.value.follow_up_candidates == [0, 1] + [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding[:3], [], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect an end index of a none-empty Span, so the follow-up candidates are [7, 13)
    assert excinfo.value.follow_up_candidates == list(range(7, 13))


def test_labeled_multi_span_encoder_decoder_parse_incomplete_with_previous_annotations():
    label2id = {"A": 0, "B": 1}
    encoder_decoder = LabeledMultiSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(offset=len(label2id)),
        label2id=label2id,
    )
    # a span before the current start index should not affect the follow-up candidates
    other_span_before = LabeledMultiSpan(slices=((0, 1), (2, 3)), label="A")
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse([], [other_span_before], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect a start index which can be in between the slices of the other_span_before or after the last slice,
    # but we also allow to generate the exact same spans again, so the indices [0, 2] are also allowed
    # i.e. [0] + [1] + [2] + [3, 4, 5, 6, 7, 8, 9], but we need to add the offset of 2
    assert excinfo.value.follow_up_candidates == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # a span after the current start index should limit the follow-up candidates
    other_span_after = LabeledMultiSpan(slices=((5, 6), (7, 8)), label="A")
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse([3], [other_span_after], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect an end index of a none-empty Span and the span should not overlap with other_span_after,
    # so allowed end indices are [2, 3, 4, 5], but we need to add the offset of 2
    assert excinfo.value.follow_up_candidates == [4, 5, 6, 7]

    entangled_span = LabeledMultiSpan(slices=((0, 2), (4, 5)), label="A")
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse([4, 5], [entangled_span], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as LabeledMultiSpan"
    # we expect either a label encoding, i.e. [0, 1], or a new span encoding, but the start index is
    # not allowed to be within any of the slices of the entangled_span, i.e. [0, 1] and [4],
    # and not within the span  itself, i.e. [2, 3], but we allow to generate the exact same spans again,
    # so the indices 0, 4, and 2 are also allowed, so the allowed end indices are
    # [0] + [2] + [3] + [4] + [5, 6, 7, 8, 9], but we need to add the offset of 2
    assert excinfo.value.follow_up_candidates == [0, 1] + [2] + [4] + [5] + [6] + [7, 8, 9, 10, 11]


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
    with pytest.raises(DecodingLengthException) as excinfo:
        encoder_decoder.decode([1, 2, 3, 4, 5, 6])
    assert (
        str(excinfo.value)
        == "seven values are required to decode as BinaryRelation, but the encoding has length 6"
    )
    assert excinfo.value.identifier == "len"

    with pytest.raises(DecodingLengthException) as excinfo:
        encoder_decoder.decode([1, 2, 3, 4, 5, 6, 7, 8])
    assert (
        str(excinfo.value)
        == "seven values are required to decode as BinaryRelation, but the encoding has length 8"
    )
    assert excinfo.value.identifier == "len"


def test_binary_relation_encoder_decoder_wrong_label_index():
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
    with pytest.raises(DecodingLabelException) as excinfo:
        encoder_decoder.decode([1, 2, 3, 4, 5, 6, 7])
    assert str(excinfo.value) == "unknown label id: 7 (label2id: {'A': 0, 'B': 1, 'C': 2})"
    assert excinfo.value.identifier == "label"


@pytest.mark.parametrize(
    "mode", ["head_tail_label", "tail_head_label", "label_head_tail", "label_tail_head"]
)
def test_binary_relation_encoder_decoder_parse(mode):
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
    expected_relation = BinaryRelation(
        head=LabeledSpan(start=1, end=2, label="A"),
        tail=LabeledSpan(start=3, end=4, label="B"),
        label="C",
    )
    encoding = encoder_decoder.encode(expected_relation)
    remaining_encoding = [2, 3]
    # encoding of the expected relation + remaining encoding
    assert encoder_decoder.parse(encoding + remaining_encoding, [], 10) == (
        expected_relation,
        remaining_encoding,
    )


@pytest.mark.parametrize(
    "mode", ["head_tail_label", "tail_head_label", "label_head_tail", "label_tail_head"]
)
def test_binary_relation_encoder_decoder_parse_loop_dummy_relation(mode):
    """Test the BinaryRelationEncoderDecoder class."""

    label2id = {"A": 0, "B": 1, "C": 2, "N": 3}
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
        loop_dummy_relation_name="L",
        none_label="N",
    )
    expected_relation = BinaryRelation(
        head=LabeledSpan(start=1, end=2, label="A"),
        tail=LabeledSpan(start=1, end=2, label="A"),
        label="L",
    )
    encoding = encoder_decoder.encode(expected_relation)
    remaining_encoding = [6, 7, 8]
    # encoding of the expected relation + remaining encoding
    assert encoder_decoder.parse(encoding + remaining_encoding, [], 10) == (
        expected_relation,
        remaining_encoding,
    )


def test_binary_relation_encoder_decoder_parse_loop_dummy_relation_missing():
    """Test the BinaryRelationEncoderDecoder class."""

    label2id = {"A": 0, "B": 1, "C": 2, "N": 3}
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
        none_label="N",
    )
    expected_relation = BinaryRelation(
        head=LabeledSpan(start=1, end=2, label="A"),
        tail=LabeledSpan(start=1, end=2, label="A"),
        label="N",
    )
    encoding = encoder_decoder.encode(expected_relation)
    with pytest.raises(ValueError) as excinfo:
        encoder_decoder.parse(encoding, [], 10)
    assert (
        str(excinfo.value)
        == "loop_dummy_relation_name is not set, but none_label=N was found in the encoding: "
        "[5, 6, 0, 5, 6, 0, 3] (label2id: {'A': 0, 'B': 1, 'C': 2, 'N': 3}))"
    )


@pytest.mark.parametrize("mode", ["head_tail_label", "label_head_tail"])
def test_binary_relation_encoder_decoder_parse_incomplete(mode):
    span_label2id = {"A": 0, "B": 1}
    relation_label2id = {"C": 2, "N": 3}
    labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
        span_encoder_decoder=SpanEncoderDecoderWithOffset(
            offset=len(span_label2id) + len(relation_label2id)
        ),
        label2id=span_label2id,
        mode="indices_label",
    )
    encoder_decoder = BinaryRelationEncoderDecoder(
        head_encoder_decoder=labeled_span_encoder_decoder,
        tail_encoder_decoder=labeled_span_encoder_decoder,
        label2id=relation_label2id,
        mode=mode,
        loop_dummy_relation_name="loop",
        none_label="N",
    )
    # Note: we use a tail that comes before the head on purpose to test the restricted follow-up candidates
    relation = BinaryRelation(
        head=LabeledSpan(start=3, end=4, label="A"),
        tail=LabeledSpan(start=1, end=2, label="B"),
        label="C",
    )
    encoding = encoder_decoder.encode(relation)
    # just to see the encoding
    if mode.endswith("_label"):
        assert encoding == [7, 8, 0, 5, 6, 1, 2]
    elif mode.startswith("label_"):
        assert encoding == [2, 7, 8, 0, 5, 6, 1]
    else:
        raise ValueError(f"unknown mode: {mode}")

    # check missing start index of first argument
    if mode.endswith("_label"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse([], [], 10)
    elif mode.startswith("label_"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:1], [], 10)
    else:
        raise ValueError(f"unknown mode: {mode}")
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect a start index which can be any valid start index, i.e. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # but we need to add the offset of 4
    assert excinfo.value.follow_up_candidates == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    # check missing end index of first argument
    if mode.endswith("_label"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:1], [], 10)
    elif mode.startswith("label_"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:2], [], 10)
    else:
        raise ValueError(f"unknown mode: {mode}")
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect an end index of a none-empty Span, i.e. [4, 5, 6, 7, 8, 9, 10],
    # but we need to add the offset of 4
    assert excinfo.value.follow_up_candidates == [8, 9, 10, 11, 12, 13, 14]

    # check missing label of first argument
    if mode.endswith("_label"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:2], [], 10)
    elif mode.startswith("label_"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:3], [], 10)
    else:
        raise ValueError(f"unknown mode: {mode}")
    assert str(excinfo.value) == "the encoding has not enough values to decode as LabeledSpan"
    # we expect a label encoding
    assert excinfo.value.follow_up_candidates == [0, 1]

    # check missing start index of second argument
    if mode.endswith("_label"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:3], [], 10)
    elif mode.startswith("label_"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:4], [], 10)
    else:
        raise ValueError(f"unknown mode: {mode}")
    assert str(excinfo.value) == "the encoding has not enough values to decode as BinaryRelation"
    # we expect
    # 1) the none_label index, which is 3, or
    # 2) a start index which can be any index that is not covered by the first
    # argument span which is LabeledSpan(start=3, end=4, label="A"),
    # but we allow to generate the exact same span again, i.e. [0, 1, 2] + [3] + [4, 5, 6, 7, 8, 9],
    # but we need to add the offset of 4
    assert excinfo.value.follow_up_candidates == [3] + [4, 5, 6] + [7] + [8, 9, 10, 11, 12, 13]

    # check missing end index of second argument
    if mode.endswith("_label"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:4], [], 10)
    elif mode.startswith("label_"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:5], [], 10)
    else:
        raise ValueError(f"unknown mode: {mode}")
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect an end index of a none-empty Span and the span should not overlap with the first argument span
    # which is LabeledSpan(start=3, end=4, label="A"), so allowed end indices are [2, 3]
    # but we need to add the offset of 4
    assert excinfo.value.follow_up_candidates == [6, 7]

    # check missing label of second argument
    if mode.endswith("_label"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:5], [], 10)
    elif mode.startswith("label_"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:6], [], 10)
    else:
        raise ValueError(f"unknown mode: {mode}")
    assert str(excinfo.value) == "the encoding has not enough values to decode as LabeledSpan"
    # we expect a label encoding
    assert excinfo.value.follow_up_candidates == [0, 1]

    # check missing label
    if mode.endswith("_label"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:6], [], 10)
    elif mode.startswith("label_"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse([], [], 10)
    else:
        raise ValueError(f"unknown mode: {mode}")
    assert str(excinfo.value) == "the encoding has not enough values to decode as BinaryRelation"
    # we expect a label encoding
    # TODO: this should be only 2
    assert excinfo.value.follow_up_candidates == [2, 3]

    # check loop dummy relation
    encoding = encoder_decoder.encode(
        BinaryRelation(
            head=LabeledSpan(start=3, end=4, label="A"),
            tail=LabeledSpan(start=3, end=4, label="A"),
            label="loop",
        )
    )
    if mode.endswith("_label"):
        assert encoding == [7, 8, 0, 3, 3, 3, 3]
    elif mode.startswith("label_"):
        assert encoding == [3, 7, 8, 0, 3, 3, 3]
    else:
        raise ValueError(f"unknown mode: {mode}")

    # check missing second none id in second argument
    if mode.endswith("_label"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:4], [], 10)
    elif mode.startswith("label_"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:5], [], 10)
    else:
        raise ValueError(f"unknown mode: {mode}")
    assert str(excinfo.value) == "the encoding has not enough values to decode as BinaryRelation"
    # we expect the none_label index, which is 3
    assert excinfo.value.follow_up_candidates == [3]

    # check missing third none id in second argument
    if mode.endswith("_label"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:5], [], 10)
    elif mode.startswith("label_"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:6], [], 10)
    else:
        raise ValueError(f"unknown mode: {mode}")
    assert str(excinfo.value) == "the encoding has not enough values to decode as BinaryRelation"
    # we expect the none_label index, which is 3
    assert excinfo.value.follow_up_candidates == [3]

    # check missing none id as loop dummy relation id
    if mode.endswith("_label"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse(encoding[:6], [], 10)
    elif mode.startswith("label_"):
        with pytest.raises(IncompleteEncodingException) as excinfo:
            encoder_decoder.parse([], [], 10)
    else:
        raise ValueError(f"unknown mode: {mode}")
    assert str(excinfo.value) == "the encoding has not enough values to decode as BinaryRelation"
    # we expect the none_label index, which is 3
    # TODO: this should be only 3
    assert excinfo.value.follow_up_candidates == [2, 3]


def test_binary_relation_encoder_decoder_parse_incomplete_with_previous_annotations():
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
    relation = BinaryRelation(
        head=LabeledSpan(start=3, end=4, label="A"),
        tail=LabeledSpan(start=1, end=2, label="B"),
        label="C",
    )
    encoding = encoder_decoder.encode(relation)
    assert encoding == [6, 7, 0, 4, 5, 1, 2]
    surrounding_relation = BinaryRelation(
        head=LabeledSpan(start=0, end=1, label="A"),
        tail=LabeledSpan(start=6, end=7, label="B"),
        label="C",
    )

    # test the first span start index
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse([], [surrounding_relation], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect a start index which can be any index that is not covered by the surrounding_relation argument spans,
    # which are LabeledSpan(start=0, end=1, label="A") and LabeledSpan(start=6, end=7, label="B"),
    # but we allow to generate the exact same span again, i.e. [0] + [1, 2, 3, 4, 5] + [6] + [7, 8, 9]
    # but we need to add the offset of 3
    assert excinfo.value.follow_up_candidates == [3] + [4, 5, 6, 7, 8] + [9] + [10, 11, 12]

    # test the first span end index
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding[:1], [surrounding_relation], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect an end index of a none-empty Span and the span should not overlap with the
    # surrounding_relation argument spans, so allowed end indices are [4, 5, 6],
    # but we need to add the offset of 3
    assert excinfo.value.follow_up_candidates == [7, 8, 9]

    # test the first span label
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding[:2], [surrounding_relation], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as LabeledSpan"
    # we expect a label encoding
    assert excinfo.value.follow_up_candidates == [0, 1, 2]

    # test the second span start index
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding[:3], [surrounding_relation], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect a start index which can be any index that is not covered by the surrounding_relation argument spans,
    # which are LabeledSpan(start=0, end=1, label="A") and LabeledSpan(start=6, end=7, label="B"),
    # and not by the first argument span which is LabeledSpan(start=3, end=4, label="A"),
    # but we allow to generate the exact same spans again,
    # i.e. [0] + [1, 2] + [3] + [4, 5] + [6] + [7, 8, 9], but we need to add the offset of 3
    assert excinfo.value.follow_up_candidates == [3] + [4, 5] + [6] + [7, 8] + [9] + [10, 11, 12]

    # test the second span end index
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding[:4], [surrounding_relation], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as Span"
    # we expect an end index of a none-empty Span and the span should not overlap with the surrounding_relation
    # argument spans which are LabeledSpan(start=0, end=1, label="A") and LabeledSpan(start=6, end=7, label="B"),
    # or the first argument span which is LabeledSpan(start=3, end=4, label="A"), so allowed end indices are [2, 3],
    # but we need to add the offset of 3
    assert excinfo.value.follow_up_candidates == [5, 6]

    # test the second span label
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding[:5], [surrounding_relation], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as LabeledSpan"
    # we expect a label encoding
    assert excinfo.value.follow_up_candidates == [0, 1, 2]

    # test the relation label
    with pytest.raises(IncompleteEncodingException) as excinfo:
        encoder_decoder.parse(encoding[:6], [surrounding_relation], 10)
    assert str(excinfo.value) == "the encoding has not enough values to decode as BinaryRelation"
    # we expect a label encoding
    assert excinfo.value.follow_up_candidates == [0, 1, 2]
