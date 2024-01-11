import logging
from typing import Any, Dict, List, Optional, Set

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span

from pie_modules.taskmodules.common import AnnotationEncoderDecoder
from pie_modules.taskmodules.common.interfaces import DecodingException

logger = logging.getLogger(__name__)


class DecodingLengthException(DecodingException[List[int]]):
    identifier = "len"


class DecodingOrderException(DecodingException[List[int]]):
    identifier = "order"


class DecodingSpanOverlapException(DecodingException[List[int]]):
    identifier = "overlap"


class DecodingLabelException(DecodingException[List[int]]):
    identifier = "label"


class DecodingNegativeIndexException(DecodingException[List[int]]):
    identifier = "index"


class SpanEncoderDecoder(AnnotationEncoderDecoder[Span, List[int]]):
    def __init__(self, exclusive_end: bool = True):
        self.exclusive_end = exclusive_end

    def encode(self, annotation: Span, metadata: Optional[Dict[str, Any]] = None) -> List[int]:
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


class LabeledSpanEncoderDecoder(AnnotationEncoderDecoder[LabeledSpan, List[int]]):
    def __init__(
        self,
        span_encoder_decoder: AnnotationEncoderDecoder[Span, List[int]],
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


class BinaryRelationEncoderDecoder(AnnotationEncoderDecoder[BinaryRelation, List[int]]):
    def __init__(
        self,
        head_encoder_decoder: AnnotationEncoderDecoder[Span, List[int]],
        tail_encoder_decoder: AnnotationEncoderDecoder[Span, List[int]],
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
