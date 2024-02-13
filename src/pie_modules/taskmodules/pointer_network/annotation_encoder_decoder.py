import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from pytorch_ie import Annotation
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span

from pie_modules.annotations import LabeledMultiSpan
from pie_modules.taskmodules.common.interfaces import (
    DecodingException,
    EncodingException,
    GenerativeAnnotationEncoderDecoder,
)

logger = logging.getLogger(__name__)


class DecodingLengthException(DecodingException[List[int]]):
    identifier = "len"


class DecodingOrderException(DecodingException[List[int]]):
    identifier = "order"


class DecodingSpanOverlapException(DecodingException[List[int]]):
    identifier = "overlap"


class DecodingSpanNestedException(DecodingException[List[int]]):
    identifier = "nested"


class DecodingLabelException(DecodingException[List[int]]):
    identifier = "label"


class DecodingNegativeIndexException(DecodingException[List[int]]):
    identifier = "negative_index"


class DecodingEmptySpanException(DecodingException[List[int]]):
    identifier = "empty_span"


class IncompleteEncodingException(DecodingException[List[int]]):
    identifier = "incomplete"

    def __init__(self, message: str, encoding: List[int], follow_up_candidates: List[int]):
        super().__init__(message, encoding, remaining=[])
        self.follow_up_candidates = follow_up_candidates


class EncodingEmptySpanException(EncodingException[Span]):
    identifier = "empty_span"


class EncodingEmptySlicesException(EncodingException[LabeledMultiSpan]):
    identifier = "empty_slices"


def spans_have_overlap(span: Span, other_span: Span) -> bool:
    return (
        other_span.start <= span.start < other_span.end
        or other_span.start < span.end <= other_span.end
    )


def span_is_nested_in_other_span(span: Span, other_span: Span) -> bool:
    return (
        other_span.start <= span.start < other_span.end
        and other_span.start < span.end <= other_span.end
    )


def spans_are_nested(span: Span, other_span: Span) -> bool:
    return span_is_nested_in_other_span(
        span=span, other_span=other_span
    ) or span_is_nested_in_other_span(span=other_span, other_span=span)


class SpanEncoderDecoder(GenerativeAnnotationEncoderDecoder[Span, List[int]]):
    """An encoder-decoder for Spans.

    This encoder-decoder encodes a Span annotation as a list of two integers, the start and end index
    of the span. Note that the end index of the Span annotation is exclusive, i.e. the span covers the
    indices [start, end). However, the end index can be encoded as exclusive or inclusive, depending on
    the `exclusive_end` parameter. Note that empty spans are not allowed, i.e. the start index must be
    smaller than the end index.

    Args:
        exclusive_end (bool, optional): Whether the end index will be encoded as exclusive or inclusive, i.e. the
            encoded span covers the indices [encoded_start, encoded_end) or [encoded_start, encoded_end].
            Defaults to True.
        allow_nested (bool, optional): Whether nested spans are allowed during parsing. If set to False, parsing
            will raise an exception if a span is completely within another span. Defaults to False.
    """

    def __init__(self, exclusive_end: bool = True, allow_nested: bool = False):
        self.exclusive_end = exclusive_end
        self.allow_nested = allow_nested

    def encode(self, annotation: Span, metadata: Optional[Dict[str, Any]] = None) -> List[int]:
        if annotation.start == annotation.end:
            raise EncodingEmptySpanException(
                "can not encode empty Span annotations, i.e. where the start index equals the end index",
                annotation=annotation,
            )
        end_idx = annotation.end
        if not self.exclusive_end:
            end_idx -= 1
        return [annotation.start, end_idx]

    def decode(self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None) -> Span:
        if len(encoding) != 2:
            raise DecodingLengthException(
                f"two values are required to decode as Span, but encoding has length {len(encoding)}",
                encoding=encoding,
            )
        end_idx = encoding[1]
        if not self.exclusive_end:
            end_idx += 1
        if end_idx < encoding[0]:
            raise DecodingOrderException(
                f"end index can not be smaller than start index, but got: start={encoding[0]}, "
                f"end={end_idx}",
                encoding=encoding,
            )
        if any(idx < 0 for idx in encoding):
            raise DecodingNegativeIndexException(
                f"indices must be positive, but got: {encoding}", encoding=encoding
            )
        return Span(start=encoding[0], end=end_idx)

    def parse(
        self,
        encoding: List[int],
        decoded_annotations: List[Span],
        text_length: int,
    ) -> Tuple[Span, List[int]]:
        exclusive_end_offset = 0 if self.exclusive_end else 1
        # the encoding is incomplete if it is empty, collect follow-up candidate indices
        if len(encoding) == 0:
            if self.allow_nested:
                # everything is allowed
                follow_up_candidates = list(range(text_length))
            else:
                # exclude indices that are already covered by other annotations
                nested_indices: Set[int] = set()
                for previous_span in decoded_annotations:
                    # +1 because we allow to generate the exact same spans again
                    nested_indices.update(range(previous_span.start + 1, previous_span.end))
                follow_up_candidates = [
                    idx for idx in range(text_length) if idx not in nested_indices
                ]
            raise IncompleteEncodingException(
                "the encoding has not enough values to decode as Span",
                encoding=encoding,
                follow_up_candidates=follow_up_candidates,
            )
        # the encoding is incomplete if it has only one value, collect follow-up candidate indices
        elif len(encoding) == 1:
            covering_spans = {
                ann for ann in decoded_annotations if ann.start <= encoding[0] < ann.end
            }
            if self.allow_nested:
                # exclude spans that overlap other spans, i.e. if encoding[0] is in another span, the next
                # candidate should be also within this span
                if len(covering_spans) == 0:
                    # allow all indices outside spans, after the start index
                    nested_indices = set()
                    for span in decoded_annotations:
                        # -1 because the end is outside the span
                        nested_indices = nested_indices.union(set(range(span.start, span.end - 1)))

                    follow_up_candidates = [
                        idx + 1 - exclusive_end_offset
                        for idx in range(encoding[0], text_length)
                        if idx not in nested_indices
                    ]
                else:
                    # allow all indices that are within *all* covering spans, i.e. the smallest covering span,
                    # and after the start index
                    nested_indices = set(range(0, text_length))
                    for span in covering_spans:
                        # + 1 because we want to include the (exclusive) end index
                        nested_indices = nested_indices.intersection(
                            set(range(span.start, span.end + 1))
                        )
                    follow_up_candidates = [
                        idx - exclusive_end_offset for idx in nested_indices if idx > encoding[0]
                    ]
            elif len(covering_spans) > 0:
                if len(covering_spans) > 1:
                    raise ValueError(
                        "more than one covering span found, but allow_nested=False. This should not happen."
                    )
                covering_span = list(covering_spans)[0]
                # if we generated the start of an existing span, we need to generate the exact end next
                follow_up_candidates = [covering_span.end - exclusive_end_offset]
            else:
                # allow all indices after the start index and before the next span. we add a dummy span to
                # correctly handle the case where no other spans are present
                dummy_span = Span(start=text_length, end=text_length + 1)
                next_span_start = min(
                    ann.start
                    for ann in decoded_annotations + [dummy_span]
                    if encoding[0] <= ann.start
                )
                # +1 because we disallow empty spans
                min_index = encoding[0] + 1
                # +1 because the end index is exclusive
                max_index_exclusive = next_span_start + 1
                follow_up_candidates = list(
                    range(
                        min_index - exclusive_end_offset,
                        max_index_exclusive - exclusive_end_offset,
                    )
                )
            raise IncompleteEncodingException(
                "the encoding has not enough values to decode as Span",
                encoding=encoding,
                follow_up_candidates=follow_up_candidates,
            )
        # the encoding is complete, try to decode the span
        else:
            start_idx = encoding[0]
            end_idx = encoding[1]
            remaining = encoding[2:]
            # the end index for Span annotations is exclusive, so we need to add 1 to the end index
            if not self.exclusive_end:
                end_idx += 1
            if end_idx == start_idx:
                raise DecodingEmptySpanException(
                    "end index can not be equal to start index to decode as Span, but got: "
                    f"start={start_idx}, end={end_idx}",
                    encoding=encoding,
                    remaining=remaining,
                )
            if end_idx < start_idx:
                raise DecodingOrderException(
                    f"end index can not be smaller than start index, "
                    f"but got: start={start_idx}, end={end_idx}",
                    encoding=encoding,
                    remaining=remaining,
                )
            if any(idx < 0 for idx in [start_idx, end_idx]):
                raise DecodingNegativeIndexException(
                    f"indices must be positive, but got: start={start_idx}, end={end_idx}",
                    encoding=encoding,
                    remaining=remaining,
                )
            # check overlap and nesting with previously decoded spans
            span = Span(start=start_idx, end=end_idx)
            for previous_span in decoded_annotations:
                simple_previous_span = Span(start=previous_span.start, end=previous_span.end)
                if (
                    spans_have_overlap(span=span, other_span=simple_previous_span)
                    and span != simple_previous_span
                ):
                    if spans_are_nested(span=span, other_span=simple_previous_span):
                        if not self.allow_nested:
                            raise DecodingSpanNestedException(
                                f"the encoded span is nested in another span: {previous_span}. "
                                "You can set allow_nested=True to allow nested spans.",
                                encoding=encoding,
                                remaining=remaining,
                            )
                        else:
                            # this is allowed, so we just pass
                            pass
                    else:
                        raise DecodingSpanOverlapException(
                            f"the encoded span overlaps with another span: {previous_span}",
                            encoding=encoding,
                            remaining=remaining,
                        )
            return span, remaining


class SpanEncoderDecoderWithOffset(SpanEncoderDecoder):
    def __init__(self, offset: int, **kwargs):
        super().__init__(**kwargs)
        self.offset = offset

    def encode(self, annotation: Span, metadata: Optional[Dict[str, Any]] = None) -> List[int]:
        encoding = super().encode(annotation=annotation, metadata=metadata)
        return [x + self.offset for x in encoding]

    def decode(self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None) -> Span:
        encoding = [x - self.offset for x in encoding]
        return super().decode(encoding=encoding, metadata=metadata)

    def parse(
        self,
        encoding: List[int],
        decoded_annotations: List[Span],
        text_length: int,
    ) -> Tuple[Span, List[int]]:
        encoding_without_offset = [x - self.offset for x in encoding]
        try:
            span, remaining = super().parse(
                encoding=encoding_without_offset,
                decoded_annotations=decoded_annotations,
                text_length=text_length,
            )
        except IncompleteEncodingException as e:
            # we need to add the offset to the follow-up candidates
            follow_up_candidates = [x + self.offset for x in e.follow_up_candidates]
            raise IncompleteEncodingException(
                e.message, encoding=encoding, follow_up_candidates=follow_up_candidates
            )
        # use the original encoding, i.e. with any potential offset, to get the remaining encoding
        return span, encoding[len(encoding) - len(remaining) :]


class LabeledSpanEncoderDecoder(GenerativeAnnotationEncoderDecoder[LabeledSpan, List[int]]):
    def __init__(
        self,
        span_encoder_decoder: SpanEncoderDecoderWithOffset,
        label2id: Dict[str, int],
        mode: str,
    ):
        self.span_encoder_decoder = span_encoder_decoder
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.mode = mode

    def encode(
        self, annotation: LabeledSpan, metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        encoded_span = self.span_encoder_decoder.encode(annotation=annotation, metadata=metadata)
        encoded_label = self.label2id[annotation.label]
        if self.mode == "indices_label":
            return encoded_span + [encoded_label]
        elif self.mode == "label_indices":
            return [encoded_label] + encoded_span
        else:
            raise ValueError(f"unknown mode: {self.mode}")

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> LabeledSpan:
        if self.mode == "label_indices":
            encoded_label = encoding[0]
            encoded_span = encoding[1:]
        elif self.mode == "indices_label":
            encoded_label = encoding[-1]
            encoded_span = encoding[:-1]
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        decoded_span = self.span_encoder_decoder.decode(encoding=encoded_span, metadata=metadata)
        if encoded_label not in self.id2label:
            raise DecodingLabelException(
                f"unknown label id: {encoded_label} (label2id: {self.label2id})", encoding=encoding
            )
        result = LabeledSpan(
            start=decoded_span.start,
            end=decoded_span.end,
            label=self.id2label[encoded_label],
        )
        return result

    def _parse_label(self, encoding: List[int]) -> Tuple[str, List[int]]:
        if len(encoding) == 0:
            raise IncompleteEncodingException(
                "the encoding has not enough values to decode as LabeledSpan",
                encoding=encoding,
                follow_up_candidates=sorted(self.id2label.keys()),
            )
        label_encoding = encoding[0]
        remaining = encoding[1:]
        if label_encoding not in self.id2label:
            raise DecodingLabelException(
                f"unknown label id: {label_encoding} (label2id: {self.label2id})",
                encoding=encoding,
                remaining=remaining,
            )
        label = self.id2label[label_encoding]
        return label, remaining

    def parse(
        self,
        encoding: List[int],
        decoded_annotations: List[LabeledSpan],
        text_length: int,
    ) -> Tuple[LabeledSpan, List[int]]:
        if self.mode == "label_indices":
            label, remaining = self._parse_label(encoding)
        elif self.mode == "indices_label":
            label, remaining = None, encoding
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        span, remaining = self.span_encoder_decoder.parse(
            encoding=remaining, decoded_annotations=decoded_annotations, text_length=text_length
        )
        if label is None:
            label, remaining = self._parse_label(remaining)
        result = LabeledSpan(start=span.start, end=span.end, label=label)
        return result, remaining


class LabeledMultiSpanEncoderDecoder(
    GenerativeAnnotationEncoderDecoder[LabeledMultiSpan, List[int]]
):
    """An encoder-decoder for LabeledMultiSpans.

    To encode a LabeledMultiSpan, the slices (start-end-index-tuples) are encoded in order,
    followed by the label id. Note that we expect the MultiSpan to have at least one slice.
    """

    def __init__(
        self,
        span_encoder_decoder: SpanEncoderDecoderWithOffset,
        label2id: Dict[str, int],
    ):
        self.span_encoder_decoder = span_encoder_decoder
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def encode(
        self, annotation: LabeledMultiSpan, metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        if len(annotation.slices) == 0:
            raise EncodingEmptySlicesException(
                "LabeledMultiSpan must have at least one slice to encode it.",
                annotation=annotation,
            )
        encoding = []
        for start, end in annotation.slices:
            encoded_span = self.span_encoder_decoder.encode(
                annotation=Span(start=start, end=end), metadata=metadata
            )
            encoding.extend(encoded_span)
        encoding.append(self.label2id[annotation.label])
        return encoding

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> LabeledMultiSpan:
        if len(encoding) % 2 != 1:
            raise DecodingLengthException(
                f"an odd number of encoding entries is required for decoding a LabeledMultiSpan, "
                f"but got {len(encoding)}",
                encoding=encoding,
            )
        slices = []
        for i in range(0, len(encoding) - 1, 2):
            encoded_span = encoding[i : i + 2]
            span = self.span_encoder_decoder.decode(encoding=encoded_span, metadata=metadata)
            slices.append((span.start, span.end))
        label = self.id2label[encoding[-1]]
        return LabeledMultiSpan(slices=tuple(slices), label=label)

    def parse(
        self,
        encoding: List[int],
        decoded_annotations: List[LabeledMultiSpan],
        text_length: int,
    ) -> Tuple[LabeledMultiSpan, List[int]]:
        decoded_spans = []
        for ann in decoded_annotations:
            for start, end in ann.slices:
                decoded_spans.append(Span(start=start, end=end))

        slices: List[Tuple[int, int]] = []
        remaining = encoding
        while True:
            try:
                span, remaining = self.span_encoder_decoder.parse(
                    encoding=remaining, decoded_annotations=decoded_spans, text_length=text_length
                )
            except IncompleteEncodingException as e:
                # if the current remaining encoding was empty, but we already have slices,
                # we need to add the label ids to the follow-up candidates
                if len(remaining) == 0 and len(slices) > 0:
                    raise IncompleteEncodingException(
                        "the encoding has not enough values to decode as LabeledMultiSpan",
                        encoding=encoding,
                        follow_up_candidates=sorted(e.follow_up_candidates + list(self.id2label)),
                    )
                # otherwise (partial span or empty encoding), we just re-raise the exception
                else:
                    raise e
            slices.append((span.start, span.end))
            decoded_spans.append(span)
            if len(remaining) > 0 and remaining[0] in self.id2label:
                label = self.id2label[remaining[0]]
                break

        return LabeledMultiSpan(slices=tuple(slices), label=label), remaining[1:]


class BinaryRelationEncoderDecoder(GenerativeAnnotationEncoderDecoder[BinaryRelation, List[int]]):
    def __init__(
        self,
        head_encoder_decoder: GenerativeAnnotationEncoderDecoder[Annotation, List[int]],
        tail_encoder_decoder: GenerativeAnnotationEncoderDecoder[Annotation, List[int]],
        label2id: Dict[str, int],
        mode: str,
        loop_dummy_relation_name: Optional[str] = None,
        none_label: Optional[str] = None,
    ):
        self.head_encoder_decoder = head_encoder_decoder
        self.tail_encoder_decoder = tail_encoder_decoder
        self.loop_dummy_relation_name = loop_dummy_relation_name
        self.none_label = none_label
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.mode = mode

    def encode(
        self, annotation: BinaryRelation, metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        encoded_head = self.head_encoder_decoder.encode(annotation=annotation.head)
        encoded_tail = self.tail_encoder_decoder.encode(annotation=annotation.tail)

        if (
            self.loop_dummy_relation_name is not None
            and annotation.label == self.loop_dummy_relation_name
        ):
            if annotation.head != annotation.tail:
                raise ValueError(
                    f"expected head == tail for loop_dummy_relation, but got: {annotation.head}, "
                    f"{annotation.tail}"
                )
            if self.none_label is None:
                raise ValueError(
                    f"loop_dummy_relation_name is set, but none_label is not set: {self.none_label}"
                )
            none_id = self.label2id[self.none_label]
            encoded_none_argument = [none_id, none_id, none_id]
            if self.mode == "head_tail_label":
                return encoded_head + encoded_none_argument + [none_id]
            elif self.mode == "tail_head_label":
                return encoded_tail + encoded_none_argument + [none_id]
            elif self.mode == "label_head_tail":
                return [none_id] + encoded_head + encoded_none_argument
            elif self.mode == "label_tail_head":
                return [none_id] + encoded_tail + encoded_none_argument
            else:
                raise ValueError(f"unknown mode: {self.mode}")
        else:
            encoded_label = self.label2id[annotation.label]
            if self.mode == "tail_head_label":
                return encoded_tail + encoded_head + [encoded_label]
            elif self.mode == "head_tail_label":
                return encoded_head + encoded_tail + [encoded_label]
            elif self.mode == "label_head_tail":
                return [encoded_label] + encoded_head + encoded_tail
            elif self.mode == "label_tail_head":
                return [encoded_label] + encoded_tail + encoded_head
            else:
                raise ValueError(f"unknown mode: {self.mode}")

    def is_single_span_label(self, label: str) -> bool:
        return self.none_label is not None and label == self.none_label

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> BinaryRelation:
        if len(encoding) != 7:
            raise DecodingLengthException(
                f"seven values are required to decode as BinaryRelation, but the encoding has length {len(encoding)}",
                encoding=encoding,
            )
        if self.mode.endswith("_label"):
            encoded_label = encoding[6]
            encoded_arguments = encoding[:6]
            argument_mode = self.mode[: -len("_label")]
        elif self.mode.startswith("label_"):
            encoded_label = encoding[0]
            encoded_arguments = encoding[1:]
            argument_mode = self.mode[len("label_") :]
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        if encoded_label not in self.id2label:
            raise DecodingLabelException(
                f"unknown label id: {encoded_label} (label2id: {self.label2id})", encoding=encoding
            )
        label = self.id2label[encoded_label]
        if self.is_single_span_label(label=label):
            if argument_mode == "head_tail":
                span_encoder = self.head_encoder_decoder
            elif argument_mode == "tail_head":
                span_encoder = self.tail_encoder_decoder
            else:
                raise ValueError(f"unknown argument mode: {argument_mode}")
            encoded_span = encoded_arguments[:3]
            span = span_encoder.decode(encoding=encoded_span, metadata=metadata)
            if self.loop_dummy_relation_name is None:
                raise ValueError(
                    f"loop_dummy_relation_name is not set, but none_label={self.none_label} "
                    f"was found in decoded encoding: {encoding} (label2id: {self.label2id}))"
                )
            rel = BinaryRelation(head=span, tail=span, label=self.loop_dummy_relation_name)
        else:
            if argument_mode == "head_tail":
                encoded_head = encoded_arguments[:3]
                encoded_tail = encoded_arguments[3:]
            elif argument_mode == "tail_head":
                encoded_tail = encoded_arguments[:3]
                encoded_head = encoded_arguments[3:]
            else:
                raise ValueError(f"unknown argument mode: {argument_mode}")
            head = self.head_encoder_decoder.decode(encoding=encoded_head, metadata=metadata)
            tail = self.tail_encoder_decoder.decode(encoding=encoded_tail, metadata=metadata)
            rel = BinaryRelation(head=head, tail=tail, label=label)

        return rel

    def _parse_label(self, encoding: List[int]) -> Tuple[str, List[int]]:
        if len(encoding) == 0:
            raise IncompleteEncodingException(
                "the encoding has not enough values to decode as BinaryRelation",
                encoding=encoding,
                follow_up_candidates=sorted(self.id2label.keys()),
            )
        label_encoding = encoding[0]
        remaining = encoding[1:]
        if label_encoding not in self.id2label:
            raise DecodingLabelException(
                f"unknown label id: {label_encoding} (label2id: {self.label2id})",
                encoding=encoding,
                remaining=remaining,
            )
        label = self.id2label[label_encoding]
        return label, remaining

    def parse(
        self,
        encoding: List[int],
        decoded_annotations: List[BinaryRelation],
        text_length: int,
    ) -> Tuple[BinaryRelation, List[int]]:
        if self.mode.endswith("_label"):
            label, remaining = None, encoding
            argument_mode = self.mode[: -len("_label")]
        elif self.mode.startswith("label_"):
            label, remaining = self._parse_label(encoding)
            argument_mode = self.mode[len("label_") :]
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        if argument_mode == "head_tail":
            first_argument_encoder = self.head_encoder_decoder
            second_argument_encoder = self.tail_encoder_decoder
        elif argument_mode == "tail_head":
            first_argument_encoder = self.tail_encoder_decoder
            second_argument_encoder = self.head_encoder_decoder
        else:
            raise ValueError(f"unknown argument mode: {argument_mode}")

        decoded_arguments = []
        for rel in decoded_annotations:
            decoded_arguments.append(rel.head)
            decoded_arguments.append(rel.tail)

        first_argument, remaining = first_argument_encoder.parse(
            encoding=remaining, decoded_annotations=decoded_arguments, text_length=text_length
        )
        decoded_arguments.append(first_argument)
        try:
            second_argument, remaining = second_argument_encoder.parse(
                encoding=remaining, decoded_annotations=decoded_arguments, text_length=text_length
            )
        except DecodingException as e:
            if self.none_label is not None:
                none_id = self.label2id[self.none_label]
                if remaining[0:3] == [none_id] * 3:
                    second_argument = first_argument
                    remaining = remaining[3:]
                elif len(remaining) == 0 and isinstance(e, IncompleteEncodingException):
                    raise IncompleteEncodingException(
                        "the encoding has not enough values to decode as BinaryRelation",
                        encoding=encoding,
                        follow_up_candidates=sorted(e.follow_up_candidates + [none_id]),
                    )
                elif 0 < len(remaining) < 3 and remaining == [none_id] * len(remaining):
                    raise IncompleteEncodingException(
                        "the encoding has not enough values to decode as BinaryRelation",
                        encoding=encoding,
                        follow_up_candidates=[none_id],
                    )
                else:
                    raise e
            else:
                raise e

        if label is None:
            label, remaining = self._parse_label(remaining)

        if label == self.none_label:
            if self.loop_dummy_relation_name is None:
                raise ValueError(
                    f"loop_dummy_relation_name is not set, but none_label={self.none_label} "
                    f"was found in the encoding: {encoding} (label2id: {self.label2id}))"
                )
            label = self.loop_dummy_relation_name

        if argument_mode == "head_tail":
            rel = BinaryRelation(head=first_argument, tail=second_argument, label=label)
        elif argument_mode == "tail_head":
            rel = BinaryRelation(head=second_argument, tail=first_argument, label=label)
        else:
            raise ValueError(f"unknown argument mode: {argument_mode}")
        return rel, remaining
