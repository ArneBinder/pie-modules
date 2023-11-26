from __future__ import annotations

import logging
from typing import TypeVar

from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationList, Document

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


def trim_text_spans(
    document: D,
    layer: str,
    skip_empty: bool = True,
    verbose: bool = True,
) -> D:
    """Remove the whitespace at the beginning and end of span annotations that target a text field.

    Args:
        document: The document to trim its span annotations.
        layer: The name of the span layer to trim.
        skip_empty: If True, empty spans will be skipped. Otherwise, an error will be raised.
        verbose: If True, log warnings for trimmed spans.

    Returns:
        The document with trimmed spans.
    """
    annotation_layer_names = {f.name for f in document.annotation_fields()}
    result = type(document).fromdict(
        {k: v for k, v in document.asdict().items() if k not in annotation_layer_names}
    )

    spans: AnnotationList[LabeledSpan] = document[layer]

    old2new_spans = {}
    removed_span_ids = []

    text = spans.target

    for span in spans:
        span_text = text[span.start : span.end]
        new_start = span.start + len(span_text) - len(span_text.lstrip())
        new_end = span.end - len(span_text) + len(span_text.rstrip())

        if new_end <= new_start:
            if skip_empty:
                if verbose:
                    logger.warning(
                        f'Span "{span}" is empty after trimming. Skipping it. (disable this warning with verbose=False)'
                    )
                removed_span_ids.append(span._id)
                continue
            else:
                if verbose:
                    logger.warning(
                        f'Span "{span}" is empty after trimming. Keep it. (disable this warning with verbose=False)'
                    )
                # if there was only whitespace, we create a span with length 0 at the start of the original span
                if new_end < new_start:
                    new_start = span.start
                    new_end = span.start

        new_span = LabeledSpan(
            start=new_start,
            end=new_end,
            label=span.label,
            score=span.score,
        )
        if (span.start != new_span.start or span.end != new_span.end) and verbose:
            logger.debug(
                f'Trimmed span "{span}" to "{new_span}" (disable this warning with verbose=False)'
            )
        old2new_spans[span._id] = new_span

    result[layer].extend(old2new_spans.values())
    result.add_all_annotations_from_other(
        document,
        override_annotations={layer: old2new_spans},
        removed_annotations={layer: set(removed_span_ids)},
        verbose=verbose,
        strict=True,
    )

    return result


class TextSpanTrimmer:
    """Remove the whitespace at the beginning and end of span annotations that target a text field.

    Args:
        layer: The name of the text span layer to trim.
        skip_empty: If True, empty spans will be skipped. Otherwise, an error will be raised.
        verbose: If True, log warnings for trimmed spans.
    """

    def __init__(
        self,
        layer: str,
        skip_empty: bool = True,
        verbose: bool = True,
    ):
        self.layer = layer
        self.skip_empty = skip_empty
        self.verbose = verbose

    def __call__(self, document: D) -> D:
        return trim_text_spans(
            document=document,
            layer=self.layer,
            skip_empty=self.skip_empty,
            verbose=self.verbose,
        )
