import logging
from typing import Any, Dict, List, Optional

from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span

from pie_modules.taskmodules.common import AnnotationEncoderDecoder

logger = logging.getLogger(__name__)


class SimpleSpanEncoderDecoder(AnnotationEncoderDecoder[Span, List[int]]):
    def encode(self, span: Span, metadata: Optional[Dict[str, Any]] = None) -> Optional[List[int]]:
        return [span.start, span.end]

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Span]:
        if len(encoding) != 2:
            raise Exception(
                f"two values are required to decode as Span, but the encoding is: {encoding}"
            )
        return Span(start=encoding[0], end=encoding[1])


class SpanEncoderDecoderWithOffset(AnnotationEncoderDecoder[Span, List[int]]):
    def __init__(self, offset: int, span_end_mode: str = "last_token"):
        self.span_end_mode = span_end_mode
        self.offset = offset

    def encode(
        self, annotation: Span, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[List[int]]:
        if self.span_end_mode == "first_token_of_last_word":
            raise NotImplementedError("span_end_mode=first_token_of_last_word not implemented")
        elif self.span_end_mode == "last_token":
            return [annotation.start + self.offset, annotation.end + self.offset - 1]
        else:
            raise Exception(f"unknown span_end_mode: {self.span_end_mode}")

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Span]:
        if len(encoding) != 2:
            raise Exception(
                f"two values are required to decode as Span, but encoding is: {encoding}"
            )

        if self.span_end_mode == "first_token_of_last_word":
            raise NotImplementedError("span_end_mode=first_token_of_last_word not implemented")
        elif self.span_end_mode == "last_token":
            return Span(start=encoding[0] - self.offset, end=encoding[1] - self.offset + 1)
        else:
            raise Exception(f"unknown span_end_mode: {self.span_end_mode}")


class LabeledSpanEncoderDecoder(AnnotationEncoderDecoder[LabeledSpan, List[int]]):
    def __init__(
        self,
        span_encoder_decoder: AnnotationEncoderDecoder[Span, List[int]],
        label2id: Dict[str, int],
    ):
        self.span_encoder_decoder = span_encoder_decoder
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def encode(
        self, annotation: LabeledSpan, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[List[int]]:
        encoded_span = self.span_encoder_decoder.encode(annotation=annotation, metadata=metadata)
        if encoded_span is None:
            return None
        return encoded_span + [self.label2id[annotation.label]]

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> LabeledSpan:
        if len(encoding) != 3:
            raise Exception(
                f"three values are required to decode as LabeledSpan, but encoding is: {encoding}"
            )
        decoded_span = self.span_encoder_decoder.decode(encoding=encoding[:2], metadata=metadata)
        if decoded_span is None:
            raise Exception(f"failed to decode span from encoding: {encoding}")
        result = LabeledSpan(
            start=decoded_span.start,
            end=decoded_span.end,
            label=self.id2label[encoding[2]],
        )
        return result


class BinaryRelationEncoderDecoder(AnnotationEncoderDecoder[BinaryRelation, List[int]]):
    def __init__(
        self,
        head_encoder_decoder: AnnotationEncoderDecoder[Span, List[int]],
        tail_encoder_decoder: AnnotationEncoderDecoder[Span, List[int]],
        label2id: Dict[str, int],
        loop_dummy_relation_name: Optional[str] = "loop",
        none_label: str = "none",
    ):
        self.head_encoder_decoder = head_encoder_decoder
        self.tail_encoder_decoder = tail_encoder_decoder
        self.loop_dummy_relation_name = loop_dummy_relation_name
        self.none_label = none_label
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def encode(
        self, annotation: BinaryRelation, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[List[int]]:
        encoded_head = self.head_encoder_decoder.encode(annotation=annotation.head)
        encoded_tail = self.tail_encoder_decoder.encode(annotation=annotation.tail)
        if encoded_head is None or encoded_tail is None:
            raise Exception(f"failed to encode head or tail from annotation: {annotation}")
        if (
            self.loop_dummy_relation_name is None
            or annotation.label != self.loop_dummy_relation_name
        ):
            return encoded_tail + encoded_head + [self.label2id[annotation.label]]
        else:
            if encoded_head != encoded_tail:
                raise Exception(
                    f"expected encoded_head == encoded_tail for loop_dummy_relation, but got: {encoded_head}, "
                    f"{encoded_tail}"
                )
            none_id = self.label2id[self.none_label]
            return encoded_head + [none_id, none_id, none_id, none_id]

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> BinaryRelation:
        if len(encoding) != 7:
            raise Exception(
                f"seven values are required to decode as BinaryRelation, but encoding is: {encoding}"
            )

        decoded_tail = self.tail_encoder_decoder.decode(encoding=encoding[:3], metadata=metadata)
        label = self.id2label[encoding[6]]
        if label == self.none_label:
            decoded_head = decoded_tail
        else:
            decoded_head = self.head_encoder_decoder.decode(
                encoding=encoding[3:6], metadata=metadata
            )
        if decoded_head is None or decoded_tail is None:
            raise Exception(f"failed to decode head or tail from encoding: {encoding}")
        rel = BinaryRelation(head=decoded_head, tail=decoded_tail, label=label)
        return rel
