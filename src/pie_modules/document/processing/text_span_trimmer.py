from __future__ import annotations

import logging
from typing import TypeVar

from pytorch_ie.annotations import LabeledSpan, Span
from pytorch_ie.core import AnnotationList, Document

from pie_modules.annotations import LabeledMultiSpan

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


def trim_text_spans(
    document: D,
    layer: str,
    skip_empty: bool = True,
    strict: bool = True,
    verbose: bool = True,
) -> D:
    """Remove the whitespace at the beginning and end of span annotations that target a text field.

    Args:
        document: The document to trim its span annotations.
        layer: The name of the span layer to trim.
        skip_empty: If True, empty spans will be skipped. Otherwise, an error will be raised.
        strict: If True, raise an error if a removed span causes a removal of a relation or
            other annotation that depends on it.
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
        if isinstance(span, Span):
            starts_and_ends = [(span.start, span.end)]
            original_kwargs = {
                "start": span.start,
                "end": span.end,
            }
        elif isinstance(span, LabeledMultiSpan):
            starts_and_ends = list(span.slices)
            if len(starts_and_ends) == 0:
                if skip_empty:
                    if verbose:
                        logger.warning(
                            f'Span "{span}" is already empty (before trimming). Remove it because skip_empty=True. '
                            f"(disable this warning with verbose=False)"
                        )
                    removed_span_ids.append(span._id)
                else:
                    if verbose:
                        logger.warning(
                            f'Span "{span}" is already empty (before trimming). Keep it because skip_empty=False. '
                            f"(disable this warning with verbose=False)"
                        )
                    old2new_spans[span._id] = span.copy()
                continue
            original_kwargs = {
                "slices": span.slices,
            }
        else:
            raise ValueError(f"Unsupported span type: {type(span)}")
        new_starts_and_ends = []
        for start, end in starts_and_ends:
            span_text = text[start:end]
            new_start = start + len(span_text) - len(span_text.lstrip())
            new_end = end - len(span_text) + len(span_text.rstrip())

            if new_end <= new_start:
                if skip_empty:
                    continue
                else:
                    # if there was only whitespace, we create a span with length 0 at the start of the original span
                    if new_end < new_start:
                        new_start = span.start
                        new_end = span.start
            new_starts_and_ends.append((new_start, new_end))

        if skip_empty:
            if len(new_starts_and_ends) == 0:
                if verbose:
                    logger.warning(
                        f'Span "{span}" is empty after trimming. Skipping it. (disable this warning with verbose=False)'
                    )
                removed_span_ids.append(span._id)
                continue
        if isinstance(span, Span):
            if not len(new_starts_and_ends) == 1:
                raise ValueError(f"Expected one span, got {len(new_starts_and_ends)}")
            new_kwargs = {
                "start": new_starts_and_ends[0][0],
                "end": new_starts_and_ends[0][1],
            }
        elif isinstance(span, LabeledMultiSpan):
            new_kwargs = {
                "slices": tuple(new_starts_and_ends),
            }
        else:
            raise ValueError(f"Unsupported span type: {type(span)}")

        new_span = span.copy(**new_kwargs)
        if original_kwargs != new_kwargs and verbose:
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
        strict=strict,
    )

    return result


class TextSpanTrimmer:
    """Remove the whitespace at the beginning and end of span annotations that target a text field.

    Args:
        layer: The name of the text span layer to trim.
        skip_empty: If True, empty spans will be skipped. Otherwise, an error will be raised.
        strict: If True, raise an error if a removed span causes a removal of a relation or other
            annotation that depends on it.
        verbose: If True, log warnings for trimmed spans.
    """

    def __init__(
        self,
        layer: str,
        skip_empty: bool = True,
        strict: bool = True,
        verbose: bool = True,
    ):
        self.layer = layer
        self.skip_empty = skip_empty
        self.strict = strict
        self.verbose = verbose

    def __call__(self, document: D) -> D:
        return trim_text_spans(
            document=document,
            layer=self.layer,
            skip_empty=self.skip_empty,
            strict=self.strict,
            verbose=self.verbose,
        )
