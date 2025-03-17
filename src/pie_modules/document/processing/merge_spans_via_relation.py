import logging
from typing import Optional, Sequence, Set, Tuple, TypeVar, Union

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
    combine_scores_method: str = "mean",
) -> Tuple[Union[Set[LabeledSpan], Set[LabeledMultiSpan]], Set[BinaryRelation]]:
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX must be installed to use the SpansViaRelationMerger. "
            "You can install NetworkX with `pip install networkx`."
        )

    # convert list of relations to a graph to easily calculate connected components to merge
    g = nx.Graph()
    link_relations = []
    other_relations = []
    span2edge_relation = {}
    for rel in relations:
        if rel.label == link_relation_label:
            link_relations.append(rel)
            # never merge spans that have not the same label
            if (
                not (isinstance(rel.head, LabeledSpan) or isinstance(rel.tail, LabeledSpan))
                or rel.head.label == rel.tail.label
            ):
                g.add_edge(rel.head, rel.tail)
                span2edge_relation[rel.head] = rel
                span2edge_relation[rel.tail] = rel
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
        # get all relations that connect the spans in the connected component
        connected_component_relations_set = {
            span2edge_relation[span] for span in connected_components_sorted
        }
        # keep only n-1 relations (take the n-1 highest scoring relations) to not use more scores than necessary
        connected_component_relations = sorted(
            connected_component_relations_set, key=lambda rel: rel.score, reverse=True
        )[: len(connected_components) - 1]
        relation_scores = [rel.score for rel in connected_component_relations]
        span_scores = [span.score for span in connected_components_sorted]
        all_scores = relation_scores + span_scores
        if combine_scores_method == "mean":
            score = sum(all_scores) / len(all_scores)
        elif combine_scores_method == "product":
            score = 1.0
            for s in all_scores:
                score *= s
        else:
            raise ValueError(f'combine_scores_method="{combine_scores_method}" not supported')

        if create_multi_spans:
            new_span = LabeledMultiSpan(
                slices=tuple((span.start, span.end) for span in connected_components_sorted),
                label=label,
                score=score,
            )
        else:
            new_span = LabeledSpan(
                start=min(span.start for span in connected_components_sorted),
                end=max(span.end for span in connected_components_sorted),
                label=label,
                score=score,
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

    This processor merges spans based on binary relations. The spans are merged into a
    single span if they are connected via a relation with the specified link label. The
    processor handles both gold and predicted annotations.

    Args:
        relation_layer: The name of the relation layer in the document.
        link_relation_label: The label of the relation that should be used to merge spans.
        create_multi_spans: Whether to create multi spans or not. If `True`, multi spans
            will be created, otherwise single spans that cover the merged spans will be
            created.
        result_document_type: The type of the document to return. This can be a class or
            a string that can be resolved to a class. The class must be a subclass of
            `Document`. Required when `create_multi_spans` is `True`.
        result_field_mapping: A mapping from the field names in the input document to the
            field names in the result document. This is used to copy over fields from the
            input document to the result document. The keys are the field names in the
            input document and the values are the field names in the result document.
            Required when `result_document_type` is provided.
        use_predicted_spans: Whether to use the predicted spans or the gold spans when
            processing predictions.
        combine_scores_method: The method to combine the scores of the relations and the
            spans. The options are "mean" and "product". The default is "mean".
    """

    def __init__(
        self,
        relation_layer: str,
        link_relation_label: str,
        result_document_type: Optional[Union[type[Document], str]] = None,
        result_field_mapping: Optional[dict[str, str]] = None,
        create_multi_spans: bool = True,
        use_predicted_spans: bool = True,
        combine_scores_method: str = "mean",
    ):
        self.relation_layer = relation_layer
        self.link_relation_label = link_relation_label
        self.create_multi_spans = create_multi_spans
        if self.create_multi_spans:
            if result_document_type is None:
                raise ValueError(
                    "result_document_type must be set when create_multi_spans is True"
                )
        self.result_document_type: Optional[type[Document]]
        if result_document_type is not None:
            if result_field_mapping is None:
                raise ValueError(
                    "result_field_mapping must be set when result_document_type is provided"
                )
            self.result_document_type = resolve_type(
                result_document_type, expected_super_type=Document
            )
        else:
            self.result_document_type = None
        self.result_field_mapping = result_field_mapping or {}
        self.use_predicted_spans = use_predicted_spans
        self.combine_scores_method = combine_scores_method

    def __call__(self, document: D) -> D:
        relations: AnnotationLayer[BinaryRelation] = document[self.relation_layer]
        spans: AnnotationLayer[LabeledSpan] = document[self.relation_layer].target_layer

        # process gold annotations
        new_gold_spans, new_gold_relations = _merge_spans_via_relation(
            spans=spans,
            relations=relations,
            link_relation_label=self.link_relation_label,
            create_multi_spans=self.create_multi_spans,
            combine_scores_method=self.combine_scores_method,
        )

        # process predicted annotations
        new_pred_spans, new_pred_relations = _merge_spans_via_relation(
            spans=spans.predictions if self.use_predicted_spans else spans,
            relations=relations.predictions,
            link_relation_label=self.link_relation_label,
            create_multi_spans=self.create_multi_spans,
            combine_scores_method=self.combine_scores_method,
        )

        result = document.copy(with_annotations=False)
        if self.result_document_type is not None:
            result = result.as_type(new_type=self.result_document_type)
        span_layer_name = document[self.relation_layer].target_name
        result_span_layer_name = self.result_field_mapping.get(span_layer_name, span_layer_name)
        result_relation_layer_name = self.result_field_mapping.get(
            self.relation_layer, self.relation_layer
        )
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
