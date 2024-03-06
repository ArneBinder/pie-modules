from __future__ import annotations

from typing import Generic, TypeVar, get_args

from pytorch_ie import Annotation, Document

from pie_modules.annotations import MultiSpan, Span

ST = TypeVar("ST", bound=Span)


def multi_span_to_span(multi_span: MultiSpan, result_type: type[ST]) -> ST:
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
    return result_type(**span_dict)


AT = TypeVar("AT", bound=Annotation)


def get_layer_annotation_type(
    document_type: type[Document], layer_name: str, super_type: type[AT] = Annotation
) -> type[AT]:
    """Get the annotation type of the given layer in the given document type."""
    for field in document_type.annotation_fields():
        if field.name == layer_name:
            result = get_args(field.type)[0]
            if not issubclass(result, super_type):
                raise ValueError(
                    f"The layer {layer_name} in the document type {document_type} must be a subclass of {super_type}."
                )
            return result
    raise ValueError(
        f"The document type {document_type} does not have a layer with the name {layer_name}."
    )


D = TypeVar("D", bound=Document)


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
        # ensure that the layer exists and is a MultiSpan layer
        get_layer_annotation_type(type(document), layer_name=self.layer, super_type=MultiSpan)
        result: D = document.copy(with_annotations=False).as_type(
            new_type=self.result_document_type, field_mapping=self.result_field_mapping
        )
        target_layer_name = self.result_field_mapping.get(self.layer, self.layer)
        source_layer = document[self.layer]
        target_layer = result[target_layer_name]
        # get target annotation type from the result document type
        target_span_type = get_layer_annotation_type(
            self.result_document_type, layer_name=target_layer_name, super_type=Span
        )
        span_mapping: dict[int, Span] = {}
        # process gold annotations
        for multi_span in source_layer:
            new_span = multi_span_to_span(multi_span, result_type=target_span_type)
            span_mapping[multi_span._id] = new_span
            target_layer.append(new_span)
        # process predicted annotations
        for multi_span in source_layer.predictions:
            new_span = multi_span_to_span(multi_span, result_type=target_span_type)
            span_mapping[multi_span._id] = new_span
            target_layer.predictions.append(new_span)

        result.add_all_annotations_from_other(
            document, override_annotations={target_layer_name: span_mapping}
        )
        return result
