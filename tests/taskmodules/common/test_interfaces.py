from typing import Any, Dict, List, Set, Tuple

from pie_modules.annotations import Span
from pie_modules.taskmodules.common import AnnotationEncoderDecoder


def test_annotation_encoder_decoder():
    """Test the AnnotationEncoderDecoder class."""

    class SpanAnnotationEncoderDecoder(AnnotationEncoderDecoder[Span, Tuple[int, int]]):
        """A class that uses the AnnotationEncoderDecoder class."""

        def encode(self, annotation: Span, **kwargs) -> Tuple[int, int]:
            return annotation.start, annotation.end

        def decode(self, encoding: Tuple[int, int], **kwargs) -> Span:
            return Span(start=encoding[0], end=encoding[1])

        def validate_encoding(self, encoding: Tuple[int, int]) -> Set[str]:
            return {"order"} if encoding[0] > encoding[1] else set()

    encoder_decoder = SpanAnnotationEncoderDecoder()

    assert encoder_decoder.encode(Span(start=1, end=2)) == (1, 2)
    assert encoder_decoder.decode((1, 2)) == Span(start=1, end=2)
    assert encoder_decoder.validate_encoding((1, 2)) == set()
    assert encoder_decoder.validate_encoding((2, 1)) == {"order"}
