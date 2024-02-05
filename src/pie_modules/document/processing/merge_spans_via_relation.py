import logging
from typing import Sequence, Set, Tuple, TypeVar, Union

import networkx as nx
from pytorch_ie import AnnotationLayer
from pytorch_ie.core import Document

from pie_modules.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pie_modules.utils import resolve_type

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


def _merge_spans_via_relation(
    spans: Sequence[LabeledSpan],
    relations: Sequence[BinaryRelation],
    link_relation_label: str,
    create_multi_spans: bool = True,
) -> Tuple[Union[Set[LabeledSpan], Set[LabeledMultiSpan]], Set[BinaryRelation]]:
    # convert list of relations to a graph to easily calculate connected components to merge
    g = nx.Graph()
    link_relations = []
    other_relations = []
    for rel in relations:
        if rel.label == link_relation_label:
            link_relations.append(rel)
            # never merge spans that have not the same label
            if (
                not (isinstance(rel.head, LabeledSpan) or isinstance(rel.tail, LabeledSpan))
                or rel.head.label == rel.tail.label
            ):
                g.add_edge(rel.head, rel.tail)
            else:
                logger.debug(
                    f"spans to merge do not have the same label, do not merge them: {rel.head}, {rel.tail}"
                )
        else:
            other_relations.append(rel)

    span_mapping = {}
    connected_components: Set[LabeledSpan]
    for connected_components in nx.connected_components(g):
        # all spans in a connected component have the same label
        label = list(span.label for span in connected_components)[0]
        connected_components_sorted = sorted(connected_components, key=lambda span: span.start)
        if create_multi_spans:
            new_span = LabeledMultiSpan(
                slices=tuple((span.start, span.end) for span in connected_components_sorted),
                label=label,
            )
        else:
            new_span = LabeledSpan(
                start=min(span.start for span in connected_components_sorted),
                end=max(span.end for span in connected_components_sorted),
                label=label,
            )
        for span in connected_components_sorted:
            span_mapping[span] = new_span
    for span in spans:
        if span not in span_mapping:
            if create_multi_spans:
                span_mapping[span] = LabeledMultiSpan(
                    slices=((span.start, span.end),), label=span.label, score=span.score
                )
            else:
                span_mapping[span] = LabeledSpan(
                    start=span.start, end=span.end, label=span.label, score=span.score
                )

    new_spans = set(span_mapping.values())
    new_relations = {
        BinaryRelation(
            head=span_mapping[rel.head],
            tail=span_mapping[rel.tail],
            label=rel.label,
            score=rel.score,
        )
        for rel in other_relations
    }

    return new_spans, new_relations


class SpansViaRelationMerger:
    """Merge spans based on relations.

    This processor merges spans based on the relations with a specific label. The spans
    are merged into a single span if they are connected via a relation with the specified
    label. The processor can be used to merge spans based on predicted or gold relations.

    Args:
        relation_layer: The name of the relation layer in the document.
        link_relation_label: The label of the relation that should be used to merge spans.
        result_document_type: The type of the document to return. This can be a class or
            a string that can be resolved to a class. The class must be a subclass of
            `Document`.
        result_field_mapping: A mapping from the field names in the input document to the
            field names in the result document.
        create_multi_spans: Whether to create multi spans or not. If `True`, multi spans
            will be created, otherwise single spans that cover the merged spans will be
            created.
        use_predicted_spans: Whether to use the predicted spans or the gold spans when
            processing predictions.
    """

    def __init__(
        self,
        relation_layer: str,
        link_relation_label: str,
        result_document_type: Union[type[Document], str],
        result_field_mapping: dict[str, str],
        create_multi_spans: bool = True,
        use_predicted_spans: bool = True,
    ):
        self.relation_layer = relation_layer
        self.link_relation_label = link_relation_label
        self.result_document_type = resolve_type(
            result_document_type, expected_super_type=Document
        )
        self.result_field_mapping = result_field_mapping
        self.create_multi_spans = create_multi_spans
        self.use_predicted_spans = use_predicted_spans

    def __call__(self, document: D) -> D:
        relations: AnnotationLayer[BinaryRelation] = document[self.relation_layer]
        spans: AnnotationLayer[LabeledSpan] = document[self.relation_layer].target_layer

        # process gold annotations
        new_gold_spans, new_gold_relations = _merge_spans_via_relation(
            spans=spans,
            relations=relations,
            link_relation_label=self.link_relation_label,
            create_multi_spans=self.create_multi_spans,
        )

        # process predicted annotations
        new_pred_spans, new_pred_relations = _merge_spans_via_relation(
            spans=spans.predictions if self.use_predicted_spans else spans,
            relations=relations.predictions,
            link_relation_label=self.link_relation_label,
            create_multi_spans=self.create_multi_spans,
        )

        result = document.copy(with_annotations=False).as_type(new_type=self.result_document_type)
        span_layer_name = document[self.relation_layer].target_name
        result_span_layer_name = self.result_field_mapping[span_layer_name]
        result_relation_layer_name = self.result_field_mapping[self.relation_layer]
        result[result_span_layer_name].extend(new_gold_spans)
        result[result_relation_layer_name].extend(new_gold_relations)
        result[result_span_layer_name].predictions.extend(new_pred_spans)
        result[result_relation_layer_name].predictions.extend(new_pred_relations)

        # copy over remaining fields mentioned in result_field_mapping
        for field_name, result_field_name in self.result_field_mapping.items():
            if field_name not in [span_layer_name, self.relation_layer]:
                for ann in document[field_name]:
                    result[result_field_name].append(ann.copy())
                for ann in document[field_name].predictions:
                    result[result_field_name].predictions.append(ann.copy())
        return result
