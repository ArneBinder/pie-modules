from __future__ import annotations

from typing import Generic, TypeVar

from pytorch_ie import Document

from pie_modules.annotations import LabeledMultiSpan, LabeledSpan, MultiSpan, Span

D = TypeVar("D", bound=Document)


def multi_span_to_span(multi_span: MultiSpan) -> Span:
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
