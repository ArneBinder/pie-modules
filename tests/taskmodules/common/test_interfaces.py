from typing import Any, Dict, List, Tuple

from pytorch_ie.annotations import Span
from torchmetrics import Metric

from pie_modules.taskmodules.common import (
    AnnotationEncoderDecoder,
    HasBuildMetric,
    HasDecodeAnnotations,
)


def test_annotation_encoder_decoder():
    """Test the AnnotationEncoderDecoder class."""

    class SpanAnnotationEncoderDecoder(AnnotationEncoderDecoder[Span, Tuple[int, int]]):
        """A class that uses the AnnotationEncoderDecoder class."""

        def encode(self, annotation: Span, **kwargs) -> Tuple[int, int]:
            return annotation.start, annotation.end

        def decode(self, encoding: Tuple[int, int], **kwargs) -> Span:
            return Span(start=encoding[0], end=encoding[1])

    encoder_decoder = SpanAnnotationEncoderDecoder()

    assert encoder_decoder.encode(Span(start=1, end=2)) == (1, 2)
    assert encoder_decoder.decode((1, 2)) == Span(start=1, end=2)


def test_has_decode_annotations():
    """Test the HasDecodeAnnotations class."""

    class MyAnnotationDecoder(HasDecodeAnnotations[List[int]]):
        """A class that uses the HasDecodeAnnotations class."""

        def decode_annotations(
            self, encoding: List[int], **kwargs
        ) -> Tuple[Dict[str, List[Span]], Any]:
            return {"spans": [Span(start=encoding[0], end=encoding[1])]}, {
                "too_long": len(encoding) > 2
            }

    my_class = MyAnnotationDecoder()

    assert my_class.decode_annotations([1, 2]) == (
        {"spans": [Span(start=1, end=2)]},
        {"too_long": False},
    )
    assert my_class.decode_annotations([1, 2, 3]) == (
        {"spans": [Span(start=1, end=2)]},
        {"too_long": True},
    )


def test_has_build_metric():
    """Test the HasBuildMetric class."""

    class MyMetric(Metric):
        """A dummy metric class."""

        def update(self, x):
            pass

        def compute(self):
            return 0

    class MyMetricBuilder(HasBuildMetric):
        """A class that uses the HasBuildMetric class."""

        def build_metric(self, stage: str = None):
            return MyMetric()

    my_builder = MyMetricBuilder()
    my_metric = my_builder.build_metric()
    assert isinstance(my_metric, Metric)
    assert my_metric.compute() == 0
