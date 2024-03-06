from __future__ import annotations

from typing import Generic, TypeVar, overload

from pytorch_ie import Document

from pie_modules.annotations import LabeledMultiSpan, LabeledSpan, MultiSpan, Span

D = TypeVar("D", bound=Document)


@overload
def multi_span_to_span(multi_span: LabeledMultiSpan) -> LabeledSpan:
    ...


@overload
def multi_span_to_span(multi_span: MultiSpan) -> Span:
    ...


def multi_span_to_span(multi_span: MultiSpan) -> Span:
    """Convert a MultiSpan to a Span by taking the start and end of the first and last slice, i.e.
    create a span that covers all slices of the MultiSpan."""

    if len(multi_span.slices) == 0:
        raise ValueError("Cannot convert an empty MultiSpan to a Span.")
    slices_sorted = sorted(multi_span.slices)
    span_dict = multi_span.asdict()
    span_dict.pop("slices")
    span_dict.pop("_id")
    span_dict["start"] = slices_sorted[0][0]
    span_dict["end"] = slices_sorted[-1][1]
    if isinstance(multi_span, LabeledMultiSpan):
        return LabeledSpan(**span_dict)
    elif isinstance(multi_span, MultiSpan):
        return Span(**span_dict)
    else:
        raise ValueError(f"Unknown MultiSpan type: {type(multi_span)}")


class MultiSpanMerger(Generic[D]):
    """Merges MultiSpans in the given layer of the input document into Spans in the result
    document.

    Args:
        layer: The name of the annotation layer that contains the MultiSpans to merge.
        result_document_type: The type of the result document.
        result_field_mapping: A dictionary that maps the layer name of the input document to the
            layer name of the result document. If None, the layer name of the input document is used
            as the layer name of the result document. Required if the layer name of the result
            document is different from the layer name of the input document.
    """

    def __init__(
        self,
        layer: str,
        result_document_type: type[D],
        result_field_mapping: dict[str, str] | None = None,
    ):
        self.layer = layer
        self.result_document_type = result_document_type
        self.result_field_mapping = result_field_mapping or {}

    def __call__(self, document: Document) -> D:
        result: D = document.copy(with_annotations=False).as_type(
            new_type=self.result_document_type, field_mapping=self.result_field_mapping
        )
        target_layer_name = self.result_field_mapping.get(self.layer, self.layer)
        source_layer = document[self.layer]
        target_layer = result[target_layer_name]
        span_mapping: dict[int, Span] = {}
        # process gold annotations
        for multi_span in source_layer:
            new_span = multi_span_to_span(multi_span)
            span_mapping[multi_span._id] = new_span
            target_layer.append(new_span)
        # process predicted annotations
        for multi_span in source_layer.predictions:
            new_span = multi_span_to_span(multi_span)
            span_mapping[multi_span._id] = new_span
            target_layer.predictions.append(new_span)

        result.add_all_annotations_from_other(
            document, override_annotations={target_layer_name: span_mapping}
        )
        return result
