"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

import logging
from functools import partial
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import numpy as np
import torch
from pytorch_ie.annotations import (
    BinaryRelation,
    LabeledSpan,
    MultiLabeledBinaryRelation,
    NaryRelation,
    Span,
)
from pytorch_ie.core import (
    Annotation,
    AnnotationList,
    Document,
    TaskEncoding,
    TaskModule,
)
from pytorch_ie.documents import (
    TextDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from pytorch_ie.taskmodules.interface import ChangesTokenizerVocabSize
from pytorch_ie.utils.span import has_overlap, is_contained_in
from pytorch_ie.utils.window import get_window_around_slice
from torch import LongTensor
from torchmetrics import ClasswiseWrapper, F1Score, MetricCollection
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing_extensions import TypeAlias, TypeVar

from pie_modules.models.simple_sequence_classification import (
    InputType as ModelInputType,
)
from pie_modules.models.simple_sequence_classification import (
    TargetType as ModelTargetType,
)
from pie_modules.taskmodules.common.mixins import RelationStatisticsMixin
from pie_modules.taskmodules.metrics import WrappedMetricWithPrepareFunction
from pie_modules.utils.tokenization import (
    SpanNotAlignedWithTokenException,
    get_aligned_token_span,
)

InputEncodingType: TypeAlias = Dict[str, Any]
TargetEncodingType: TypeAlias = Sequence[int]
DocumentType: TypeAlias = TextDocument

TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]


class TaskOutputType(TypedDict, total=False):
    labels: Sequence[str]
    probabilities: Sequence[float]


TaskModuleType: TypeAlias = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
    Tuple[ModelInputType, Optional[ModelTargetType]],
    ModelTargetType,
    TaskOutputType,
]


HEAD = "head"
TAIL = "tail"
START = "start"
END = "end"


logger = logging.getLogger(__name__)


def _get_labels(model_output: ModelTargetType) -> LongTensor:
    return model_output["labels"]


def _get_labels_together_remove_none_label(
    predictions: ModelTargetType, targets: ModelTargetType, none_idx: int
) -> Tuple[LongTensor, LongTensor]:
    mask_not_both_none = (predictions["labels"] != none_idx) | (targets["labels"] != none_idx)
    predictions_not_none = predictions["labels"][mask_not_both_none]
    targets_not_none = targets["labels"][mask_not_both_none]
    return predictions_not_none, targets_not_none


def inner_span_distance(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> int:
    dist_start_other_end = abs(start_end[0] - other_start_end[1])
    dist_end_other_start = abs(start_end[1] - other_start_end[0])
    dist = min(dist_start_other_end, dist_end_other_start)
    if not has_overlap(start_end, other_start_end):
        return dist
    else:
        return -dist


def span_distance(
    start_end: Tuple[int, int], other_start_end: Tuple[int, int], distance_type: str
) -> int:
    if distance_type == "inner":
        return inner_span_distance(start_end, other_start_end)
    else:
        raise ValueError(f"unknown distance_type={distance_type}. use one of: inner")


class MarkerFactory:
    def __init__(self, role_to_marker: Dict[str, str]):
        self.role_to_marker = role_to_marker

    def _get_role_marker(self, role: str) -> str:
        return self.role_to_marker[role]

    def _get_marker(self, role: str, is_start: bool, label: Optional[str] = None) -> str:
        result = "["
        if not is_start:
            result += "/"
        result += self._get_role_marker(role)
        if label is not None:
            result += f":{label}"
        result += "]"
        return result

    def get_start_marker(self, role: str, label: Optional[str] = None) -> str:
        return self._get_marker(role=role, is_start=True, label=label)

    def get_end_marker(self, role: str, label: Optional[str] = None) -> str:
        return self._get_marker(role=role, is_start=False, label=label)

    def get_append_marker(self, role: str, label: Optional[str] = None) -> str:
        role_marker = self._get_role_marker(role)
        if label is None:
            return f"[{role_marker}]"
        else:
            return f"[{role_marker}={label}]"

    @property
    def all_roles(self) -> Set[str]:
        return set(self.role_to_marker)

    def get_all_markers(
        self,
        entity_labels: List[str],
        append_markers: bool = False,
        add_type_to_marker: bool = False,
    ) -> List[str]:
        result: Set[str] = set()
        if add_type_to_marker:
            none_and_labels = [None] + entity_labels
        else:
            none_and_labels = [None]
        for role in self.all_roles:
            # create start and end markers without label and for all labels, if add_type_to_marker
            for maybe_label in none_and_labels:
                result.add(self.get_start_marker(role=role, label=maybe_label))
                result.add(self.get_end_marker(role=role, label=maybe_label))
            # create append markers for all labels
            if append_markers:
                for entity_label in entity_labels:
                    result.add(self.get_append_marker(role=role, label=entity_label))

        # sort and convert to list
        return sorted(result)


class RelationArgument:
    def __init__(
        self,
        entity: LabeledSpan,
        role: str,
        token_span: Span,
        add_type_to_marker: bool,
        marker_factory: MarkerFactory,
    ) -> None:
        self.marker_factory = marker_factory
        if role not in self.marker_factory.all_roles:
            raise ValueError(
                f"role='{role}' not in known roles={sorted(self.marker_factory.all_roles)} (did you "
                f"initialise the taskmodule with the correct argument_role_to_marker dictionary?)"
            )

        self.entity = entity

        self.role = role
        self.token_span = token_span
        self.add_type_to_marker = add_type_to_marker

    @property
    def maybe_label(self) -> Optional[str]:
        return self.entity.label if self.add_type_to_marker else None

    @property
    def as_start_marker(self) -> str:
        return self.marker_factory.get_start_marker(role=self.role, label=self.maybe_label)

    @property
    def as_end_marker(self) -> str:
        return self.marker_factory.get_end_marker(role=self.role, label=self.maybe_label)

    @property
    def as_append_marker(self) -> str:
        # Note: we add the label in either case (we use self.entity.label instead of self.label)
        return self.marker_factory.get_append_marker(role=self.role, label=self.entity.label)

    def shift_token_span(self, value: int):
        self.token_span = Span(
            start=self.token_span.start + value, end=self.token_span.end + value
        )


def get_relation_argument_spans_and_roles(
    relation: Annotation,
) -> Tuple[Tuple[str, Annotation], ...]:
    if isinstance(relation, BinaryRelation):
        return (HEAD, relation.head), (TAIL, relation.tail)
    elif isinstance(relation, NaryRelation):
        # create unique order by sorting the arguments by their start and end positions and role
        sorted_args = sorted(
            zip(relation.roles, relation.arguments),
            key=lambda role_and_span: (
                role_and_span[1].start,
                role_and_span[1].end,
                role_and_span[0],
            ),
        )
        return tuple(sorted_args)
    else:
        raise NotImplementedError(
            f"the taskmodule does not yet support getting relation arguments for type: {type(relation)}"
        )


def construct_mask(input_ids: torch.LongTensor, positive_ids: List[Any]) -> torch.LongTensor:
    """Construct a mask for the input_ids where all entries in mask_ids are 1."""
    masks = [torch.nonzero(input_ids == marker_token_id) for marker_token_id in positive_ids]
    globs = torch.cat(masks)
    value = torch.ones(globs.shape[0], dtype=int)
    mask = torch.zeros(input_ids.shape, dtype=int)
    mask.index_put_(tuple(globs.t()), value)
    return mask


S = TypeVar("S", bound=Span)


def shift_span(span: S, offset: int) -> S:
    return span.copy(start=span.start + offset, end=span.end + offset)


@TaskModule.register()
class RETextClassificationWithIndicesTaskModule(
    RelationStatisticsMixin,
    TaskModuleType,
    ChangesTokenizerVocabSize,
):
    """Marker based relation extraction. This taskmodule prepares the input token ids in such a way
    that before and after the candidate head and tail entities special marker tokens are inserted.
    Then, the modified token ids can be simply passed into a transformer based text classifier
    model.

    parameters:

        partition_annotation: str, optional. If specified, LabeledSpan annotations with this name are
            expected to define partitions of the document that will be processed individually, e.g. sentences
            or sections of the document text.
        none_label: str, defaults to "no_relation". The relation label that indicate dummy/negative relations.
            Predicted relations with that label will not be added to the document(s).
        max_window: int, optional. If specified, use the tokens in a window of maximal this amount of tokens
            around the center of head and tail entities and pass only that into the transformer.
        create_relation_candidates: bool, defaults to False. If True, create relation candidates by pairwise
            combining all entities in the document and assigning the none_label. If the document already contains
            a relation with the entity pair, we do not add it again. If False, assume that the document already
            contains relation annotations including negative examples (i.e. relations with the none_label).
    """

    PREPARED_ATTRIBUTES = ["labels", "entity_labels"]

    def __init__(
        self,
        tokenizer_name_or_path: str,
        relation_annotation: str = "binary_relations",
        add_candidate_relations: bool = False,
        add_reversed_relations: bool = False,
        partition_annotation: Optional[str] = None,
        none_label: str = "no_relation",
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        multi_label: bool = False,
        labels: Optional[List[str]] = None,
        label_to_id: Optional[Dict[str, int]] = None,
        add_type_to_marker: bool = False,
        argument_role_to_marker: Optional[Dict[str, str]] = None,
        single_argument_pair: bool = True,
        append_markers: bool = False,
        entity_labels: Optional[List[str]] = None,
        reversed_relation_label_suffix: str = "_reversed",
        symmetric_relations: Optional[List[str]] = None,
        reverse_symmetric_relations: bool = True,
        max_argument_distance: Optional[int] = None,
        max_argument_distance_type: str = "inner",
        max_window: Optional[int] = None,
        allow_discontinuous_text: bool = False,
        log_first_n_examples: int = 0,
        add_argument_indices_to_input: bool = False,
        add_global_attention_mask_to_input: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if label_to_id is not None:
            logger.warning(
                "The parameter label_to_id is deprecated and will be removed in a future version. "
                "Please use labels instead."
            )
            id_to_label = {v: k for k, v in label_to_id.items()}
            # reconstruct labels from label_to_id. Note that we need to remove the none_label
            labels = [
                id_to_label[i] for i in range(len(id_to_label)) if id_to_label[i] != none_label
            ]
        self.save_hyperparameters(ignore=["label_to_id"])

        self.relation_annotation = relation_annotation
        self.add_candidate_relations = add_candidate_relations
        self.add_reversed_relations = add_reversed_relations
        self.padding = padding
        self.truncation = truncation
        self.labels = labels
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.multi_label = multi_label
        self.add_type_to_marker = add_type_to_marker
        self.single_argument_pair = single_argument_pair
        self.append_markers = append_markers
        self.entity_labels = entity_labels
        self.partition_annotation = partition_annotation
        self.none_label = none_label
        self.reversed_relation_label_suffix = reversed_relation_label_suffix
        self.symmetric_relations = set(symmetric_relations or [])
        self.reverse_symmetric_relations = reverse_symmetric_relations
        self.max_argument_distance = max_argument_distance
        self.max_argument_distance_type = max_argument_distance_type
        self.max_window = max_window
        self.allow_discontinuous_text = allow_discontinuous_text

        # overwrite None with 0 for backward compatibility
        self.log_first_n_examples = log_first_n_examples or 0
        self.add_argument_indices_to_input = add_argument_indices_to_input
        self.add_global_attention_mask_to_input = add_global_attention_mask_to_input
        if argument_role_to_marker is None:
            self.argument_role_to_marker = {HEAD: "H", TAIL: "T"}
        else:
            self.argument_role_to_marker = argument_role_to_marker

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # used when allow_discontinuous_text
        self.glue_token_ids = self._get_glue_token_ids()

        self.argument_markers = None

        self._logged_examples_counter = 0

    def _get_glue_token_ids(self):
        dummy_ids = self.tokenizer.build_inputs_with_special_tokens(
            token_ids_0=[-1], token_ids_1=[-2]
        )
        return dummy_ids[dummy_ids.index(-1) + 1 : dummy_ids.index(-2)]

    @property
    def document_type(self) -> Optional[Type[DocumentType]]:
        if self.partition_annotation is not None:
            dt = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions
        else:
            dt = TextDocumentWithLabeledSpansAndBinaryRelations
        if self.relation_annotation == "binary_relations":
            return dt
        else:
            logger.warning(
                f"relation_annotation={self.relation_annotation} is "
                f"not the default value ('binary_relations'), so the taskmodule {type(self).__name__} can not request "
                f"the usual document type for auto-conversion ({dt.__name__}) because this has the bespoken default "
                f"value as layer name instead of the provided one."
            )
            return None

    def get_relation_layer(self, document: Document) -> AnnotationList[BinaryRelation]:
        return document[self.relation_annotation]

    def get_entity_layer(self, document: Document) -> AnnotationList[LabeledSpan]:
        relations: AnnotationList[BinaryRelation] = self.get_relation_layer(document)
        return relations.target_layer

    def get_marker_factory(self) -> MarkerFactory:
        return MarkerFactory(role_to_marker=self.argument_role_to_marker)

    def _prepare(self, documents: Sequence[DocumentType]) -> None:
        entity_labels: Set[str] = set()
        relation_labels: Set[str] = set()
        for document in documents:
            relations: AnnotationList[BinaryRelation] = self.get_relation_layer(document)
            entities: AnnotationList[LabeledSpan] = self.get_entity_layer(document)

            for entity in entities:
                entity_labels.add(entity.label)

            for relation in relations:
                relation_labels.add(relation.label)
                if self.add_reversed_relations:
                    if relation.label.endswith(self.reversed_relation_label_suffix):
                        raise ValueError(
                            f"doc.id={document.id}: the relation label '{relation.label}' already ends with "
                            f"the reversed_relation_label_suffix '{self.reversed_relation_label_suffix}', "
                            f"this is not allowed because we would not know if we should strip the suffix and "
                            f"revert the arguments during inference or not"
                        )
                    if relation.label not in self.symmetric_relations:
                        relation_labels.add(relation.label + self.reversed_relation_label_suffix)

        if self.none_label in relation_labels:
            relation_labels.remove(self.none_label)

        self.labels = sorted(relation_labels)
        self.entity_labels = sorted(entity_labels)

    def encode(self, *args, **kwargs):
        self.reset_statistics()
        res = super().encode(*args, **kwargs)
        self.show_statistics()
        return res

    def _post_prepare(self):
        self.label_to_id = {label: i + 1 for i, label in enumerate(self.labels)}
        self.label_to_id[self.none_label] = 0
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.marker_factory = self.get_marker_factory()
        self.argument_markers = self.marker_factory.get_all_markers(
            append_markers=self.append_markers,
            add_type_to_marker=self.add_type_to_marker,
            entity_labels=self.entity_labels,
        )
        self.tokenizer.add_tokens(self.argument_markers, special_tokens=True)

        self.argument_markers_to_id = {
            marker: self.tokenizer.vocab[marker] for marker in self.argument_markers
        }

        self.argument_role2idx = {
            role: i for i, role in enumerate(sorted(self.marker_factory.all_roles))
        }

    def _add_reversed_relations(
        self,
        arguments2relation: Dict[Tuple[Tuple[str, Annotation], ...], Annotation],
        doc_id: Optional[str] = None,
    ) -> None:
        if self.add_reversed_relations:
            for arguments, rel in list(arguments2relation.items()):
                arg_roles, arg_spans = zip(*arguments)
                if isinstance(rel, BinaryRelation):
                    label = rel.label
                    if label in self.symmetric_relations and not self.reverse_symmetric_relations:
                        continue
                    if label.endswith(self.reversed_relation_label_suffix):
                        raise ValueError(
                            f"doc.id={doc_id}: The relation has the label '{label}' which already ends with the "
                            f"reversed_relation_label_suffix='{self.reversed_relation_label_suffix}'. "
                            f"It looks like the relation is already reversed, which is not allowed."
                        )
                    if rel.label not in self.symmetric_relations:
                        label += self.reversed_relation_label_suffix

                    reversed_rel = BinaryRelation(
                        head=rel.tail,
                        tail=rel.head,
                        label=label,
                        score=rel.score,
                    )
                    reversed_arguments = get_relation_argument_spans_and_roles(reversed_rel)
                    if reversed_arguments in arguments2relation:
                        prev_rel = arguments2relation[reversed_arguments]
                        prev_label = prev_rel.label
                        logger.warning(
                            f"doc.id={doc_id}: there is already a relation with reversed "
                            f"arguments={reversed_arguments} and label={prev_label}, so we do not add the reversed "
                            f"relation (with label {prev_label}) for these arguments"
                        )
                        if self.collect_statistics:
                            self.increase_counter(("skipped_reversed_same_arguments", rel.label))
                        continue
                    elif rel.label in self.symmetric_relations:
                        # warn if the original relation arguments were not sorted by their start and end positions
                        # in the case of symmetric relations
                        if not all(isinstance(arg_span, Span) for arg_span in arg_spans):
                            raise NotImplementedError(
                                f"doc.id={doc_id}: the taskmodule does not yet support adding reversed relations "
                                f"for symmetric relations with arguments that are no Spans: {arguments}"
                            )
                        args_sorted = sorted(
                            [rel.head, rel.tail], key=lambda span: (span.start, span.end)
                        )
                        if args_sorted != [rel.head, rel.tail]:
                            logger.warning(
                                f"doc.id={doc_id}: The symmetric relation with label '{label}' has arguments "
                                f"{arguments} which are not sorted by their start and end positions. "
                                f"This may lead to problems during evaluation because we assume that the "
                                f"arguments of symmetric relations were sorted in the beginning and, thus, interpret "
                                f"relations where this is not the case as reversed. All reversed relations will get "
                                f"their arguments swapped during inference in the case of add_reversed_relations=True "
                                f"to remove duplicates. You may consider adding reversed versions of the *symmetric* "
                                f"relations on your own and then setting *reverse_symmetric_relations* to False."
                            )
                            if self.collect_statistics:
                                self.increase_counter(
                                    ("used_not_sorted_reversed_arguments", rel.label)
                                )

                    arguments2relation[reversed_arguments] = reversed_rel
                else:
                    raise NotImplementedError(
                        f"doc.id={doc_id}: the taskmodule does not yet support adding reversed relations for type: "
                        f"{type(rel)}"
                    )

    def _add_candidate_relations(
        self,
        arguments2relation: Dict[Tuple[Tuple[str, Annotation], ...], Annotation],
        entities: Iterable[Span],
        doc_id: Optional[str] = None,
    ) -> None:
        if self.add_candidate_relations:
            if self.marker_factory.all_roles == {HEAD, TAIL}:
                # iterate over all possible argument candidates
                for head in entities:
                    for tail in entities:
                        if head != tail:
                            # Create a relation candidate with the none label. Otherwise, we use the existing relation.
                            new_relation = BinaryRelation(
                                head=head, tail=tail, label=self.none_label, score=1.0
                            )
                            new_relation_args = get_relation_argument_spans_and_roles(new_relation)
                            # we use the new relation only if there is no existing relation with the same arguments
                            if new_relation_args not in arguments2relation:
                                arguments2relation[new_relation_args] = new_relation
            else:
                raise NotImplementedError(
                    f"doc.id={doc_id}: the taskmodule does not yet support adding relation candidates "
                    f"with argument roles other than 'head' and 'tail': {sorted(self.marker_factory.all_roles)}"
                )

    def _filter_relations_by_argument_distance(
        self,
        arguments2relation: Dict[Tuple[Tuple[str, Annotation], ...], Annotation],
        doc_id: Optional[str] = None,
    ) -> None:
        if self.max_argument_distance is not None:
            for arguments, rel in list(arguments2relation.items()):
                if isinstance(rel, BinaryRelation):
                    if isinstance(rel.head, Span) and isinstance(rel.tail, Span):
                        dist = span_distance(
                            (rel.head.start, rel.head.end),
                            (rel.tail.start, rel.tail.end),
                            self.max_argument_distance_type,
                        )
                        if dist > self.max_argument_distance:
                            arguments2relation.pop(arguments)
                            self.collect_relation("skipped_argument_distance", rel)
                    else:
                        raise NotImplementedError(
                            f"doc.id={doc_id}: the taskmodule does not yet support filtering relation candidates "
                            f"with arguments of type: {type(rel.head)} and {type(rel.tail)}"
                        )
                else:
                    raise NotImplementedError(
                        f"doc.id={doc_id}: the taskmodule does not yet support filtering relation candidates for "
                        f"type: {type(rel)}"
                    )

    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        all_relations: Sequence[Annotation] = self.get_relation_layer(document)
        all_entities: Sequence[Span] = self.get_entity_layer(document)
        self.collect_all_relations("available", all_relations)

        partitions: Sequence[Span]
        if self.partition_annotation is not None:
            partitions = document[self.partition_annotation]
            if len(partitions) == 0:
                logger.warning(
                    f"the document {document.id} has no '{self.partition_annotation}' partition entries, "
                    f"no inputs will be created!"
                )
        else:
            # use single dummy partition
            partitions = [Span(start=0, end=len(document.text))]

        task_encodings: List[TaskEncodingType] = []
        for partition in partitions:
            # get all entities that are contained in the current partition
            entities: List[Span] = [
                entity
                for entity in all_entities
                if is_contained_in((entity.start, entity.end), (partition.start, partition.end))
            ]

            # create a mapping from relation arguments to the respective relation objects
            entities_set = set(entities)
            arguments2relation: Dict[Tuple[Tuple[str, Annotation], ...], Annotation] = {}
            for rel in all_relations:
                # skip relations with unknown labels
                if rel.label not in self.label_to_id:
                    self.collect_relation("skipped_unknown_label", rel)
                    continue

                arguments = get_relation_argument_spans_and_roles(rel)
                arg_roles, arg_spans = zip(*arguments)
                # filter out all relations that have arguments not in the current partition
                if all(arg_span in entities_set for arg_span in arg_spans):
                    # check if there are multiple relations with the same argument tuple
                    if arguments in arguments2relation:
                        prev_label = arguments2relation[arguments].label
                        logger.warning(
                            f"doc.id={document.id}: there are multiple relations with the same arguments {arguments}: "
                            f"previous label='{prev_label}' and current label='{rel.label}'. We only keep the first "
                            f"occurring relation which has the label='{prev_label}'."
                        )
                        self.collect_relation("skipped_same_arguments", rel)
                    else:
                        arguments2relation[arguments] = rel
                elif any(arg_span in entities_set for arg_span in arg_spans):
                    logger.warning(
                        f"doc.id={document.id}: there is a relation with label '{rel.label}' and arguments "
                        f"{arguments} that is only partially contained in the current partition. We skip this relation."
                    )
                    self.collect_relation("skipped_partially_contained", rel)

            self._add_reversed_relations(arguments2relation=arguments2relation, doc_id=document.id)
            self._add_candidate_relations(
                arguments2relation=arguments2relation, entities=entities, doc_id=document.id
            )
            self._filter_relations_by_argument_distance(
                arguments2relation=arguments2relation, doc_id=document.id
            )

            without_special_tokens = self.max_window is not None
            text = document.text[partition.start : partition.end]
            encoding = self.tokenizer(
                text,
                padding=False,
                truncation=self.truncation if self.max_window is None else False,
                max_length=self.max_length,
                is_split_into_words=False,
                return_offsets_mapping=False,
                add_special_tokens=not without_special_tokens,
            )

            for arguments, rel in arguments2relation.items():
                arg_roles, arg_spans = zip(*arguments)
                if not all(isinstance(arg, LabeledSpan) for arg in arg_spans):
                    # TODO: add test case for this
                    raise ValueError(
                        f"the taskmodule expects the relation arguments to be of type LabeledSpan, "
                        f"but got {[type(arg) for arg in arg_spans]}"
                    )

                arg_spans_partition = [
                    shift_span(span, offset=-partition.start) for span in arg_spans
                ]
                # map character spans to token spans
                try:
                    arg_token_spans = [
                        get_aligned_token_span(
                            encoding=encoding,
                            char_span=arg,
                        )
                        for arg in arg_spans_partition
                    ]
                # Check if the mapping was successful. It may fail (and is None) if any argument start or end does not
                # match a token start or end, respectively.
                except SpanNotAlignedWithTokenException as e:
                    span_original = shift_span(e.span, offset=partition.start)
                    # the span is not attached because we shifted it above, so we can not use str(e.span)
                    span_text = document.text[span_original.start : span_original.end]
                    logger.warning(
                        f"doc.id={document.id}: Skipping invalid example, cannot get argument token slice for "
                        f'{span_original}: "{span_text}"'
                    )
                    self.collect_relation("skipped_args_not_aligned", rel)
                    continue

                # create the argument objects
                args = [
                    RelationArgument(
                        entity=span,
                        role=role,
                        token_span=token_span,
                        add_type_to_marker=self.add_type_to_marker,
                        marker_factory=self.marker_factory,
                    )
                    for span, role, token_span in zip(arg_spans, arg_roles, arg_token_spans)
                ]

                input_ids = encoding["input_ids"]

                # windowing: we restrict the input to a window of a maximal size (max_window) with the arguments
                # of the candidate relation in the center (as much as possible)
                if self.max_window is not None:
                    # The actual number of tokens needs to be lower than max_window because we add two
                    # marker tokens (before / after) each argument and the default special tokens
                    # (e.g. CLS and SEP).
                    max_tokens = (
                        self.max_window
                        - len(args) * 2
                        - self.tokenizer.num_special_tokens_to_add()
                    )
                    # if we add the markers also to the end, this decreases the available window again by
                    # two tokens (marker + sep) per argument
                    if self.append_markers:
                        # TODO: add test case for this
                        max_tokens -= len(args) * 2

                    if self.allow_discontinuous_text:
                        max_tokens_per_argument = max_tokens // len(args)
                        max_tokens_per_argument -= len(self.glue_token_ids)
                        if any(
                            arg.token_span.end - arg.token_span.start > max_tokens_per_argument
                            for arg in args
                        ):
                            self.collect_relation("skipped_too_long_argument", rel)
                            continue

                        mask = np.zeros_like(input_ids)
                        for arg in args:
                            # if the input is already fully covered by one argument frame, we keep everything
                            if len(input_ids) <= max_tokens_per_argument:
                                mask[:] = 1
                                break
                            arg_center = (arg.token_span.end + arg.token_span.start) // 2
                            arg_frame_start = arg_center - max_tokens_per_argument // 2
                            # shift the frame to the right if it is out of bounds
                            if arg_frame_start < 0:
                                arg_frame_start = 0
                            arg_frame_end = arg_frame_start + max_tokens_per_argument
                            # shift the frame to the left if it is out of bounds
                            # Note that this can not cause to have arg_frame_start < 0 because we already
                            # checked that the frame is not larger than the input.
                            if arg_frame_end > len(input_ids):
                                arg_frame_end = len(input_ids)
                                arg_frame_start = arg_frame_end - max_tokens_per_argument
                            # still, a sanity check
                            if arg_frame_start < 0:
                                raise ValueError(
                                    f"arg_frame_start={arg_frame_start} < 0 after adjusting arg_frame_end={arg_frame_end}"
                                )
                            mask[arg_frame_start:arg_frame_end] = 1
                        offsets = np.cumsum(mask != 1)
                        arg_cluster_offset_values = set()
                        # sort by start indices
                        args_sorted = sorted(args, key=lambda x: x.token_span.start)
                        for arg in args_sorted:
                            offset = offsets[arg.token_span.start]
                            arg_cluster_offset_values.add(offset)
                            arg.shift_token_span(-offset)
                            # shift back according to inserted glue patterns
                            num_glues = len(arg_cluster_offset_values) - 1
                            arg.shift_token_span(num_glues * len(self.glue_token_ids))

                        new_input_ids: List[int] = []
                        for arg_cluster_offset_value in sorted(arg_cluster_offset_values):
                            if len(new_input_ids) > 0:
                                new_input_ids.extend(self.glue_token_ids)
                            segment_mask = offsets == arg_cluster_offset_value
                            segment_input_ids = [
                                input_id
                                for input_id, keep in zip(input_ids, mask & segment_mask)
                                if keep
                            ]
                            new_input_ids.extend(segment_input_ids)

                        input_ids = new_input_ids
                    else:
                        # the slice from the beginning of the first entity to the end of the second is required
                        slice_required = (
                            min(arg.token_span.start for arg in args),
                            max(arg.token_span.end for arg in args),
                        )
                        window_slice = get_window_around_slice(
                            slice=slice_required,
                            max_window_size=max_tokens,
                            available_input_length=len(input_ids),
                        )
                        # this happens if slice_required (all arguments) does not fit into max_tokens (the available window)
                        if window_slice is None:
                            self.collect_relation("skipped_too_long", rel)
                            continue

                        window_start, window_end = window_slice
                        input_ids = input_ids[window_start:window_end]

                        for arg in args:
                            arg.shift_token_span(-window_start)

                # collect all markers with their target positions, the source argument, and
                marker_ids_with_positions = []
                for arg in args:
                    marker_ids_with_positions.append(
                        (
                            self.argument_markers_to_id[arg.as_start_marker],
                            arg.token_span.start,
                            arg,
                            START,
                        )
                    )
                    marker_ids_with_positions.append(
                        (
                            self.argument_markers_to_id[arg.as_end_marker],
                            arg.token_span.end,
                            arg,
                            END,
                        )
                    )

                # create new input ids with the markers inserted and collect new mention offsets
                input_ids_with_markers = list(input_ids)
                offset = 0
                arg_start_indices = [-1] * len(self.argument_role2idx)
                arg_end_indices = [-1] * len(self.argument_role2idx)
                for marker_id, token_position, arg, marker_type in sorted(
                    marker_ids_with_positions, key=lambda id_pos: id_pos[1]
                ):
                    input_ids_with_markers = (
                        input_ids_with_markers[: token_position + offset]
                        + [marker_id]
                        + input_ids_with_markers[token_position + offset :]
                    )
                    offset += 1
                    if self.add_argument_indices_to_input:
                        idx = self.argument_role2idx[arg.role]
                        if marker_type == START:
                            if arg_start_indices[idx] != -1:
                                # TODO: add test case for this
                                raise ValueError(
                                    f"Trying to overwrite arg_start_indices[{idx}]={arg_start_indices[idx]} with "
                                    f"{token_position + offset} for document {document.id}"
                                )
                            arg_start_indices[idx] = token_position + offset
                        elif marker_type == END:
                            if arg_end_indices[idx] != -1:
                                # TODO: add test case for this
                                raise ValueError(
                                    f"Trying to overwrite arg_start_indices[{idx}]={arg_end_indices[idx]} with "
                                    f"{token_position + offset} for document {document.id}"
                                )
                            # -1 to undo the additional offset for the end marker which does not
                            # affect the mention offset
                            arg_end_indices[idx] = token_position + offset - 1

                if self.append_markers:
                    if self.tokenizer.sep_token is None:
                        # TODO: add test case for this
                        raise ValueError("append_markers is True, but tokenizer has no sep_token")
                    sep_token_id = self.tokenizer.vocab[self.tokenizer.sep_token]
                    for arg in args:
                        if without_special_tokens:
                            # TODO: add test case for this
                            input_ids_with_markers.append(sep_token_id)
                            input_ids_with_markers.append(
                                self.argument_markers_to_id[arg.as_append_marker]
                            )
                        else:
                            input_ids_with_markers.append(
                                self.argument_markers_to_id[arg.as_append_marker]
                            )
                            input_ids_with_markers.append(sep_token_id)

                # when windowing is used, we have to add the special tokens manually
                if without_special_tokens:
                    input_ids_with_markers = self.tokenizer.build_inputs_with_special_tokens(
                        token_ids_0=input_ids_with_markers
                    )

                inputs = {"input_ids": input_ids_with_markers}
                if self.add_argument_indices_to_input:
                    inputs["pooler_start_indices"] = arg_start_indices
                    inputs["pooler_end_indices"] = arg_end_indices
                task_encodings.append(
                    TaskEncoding(
                        document=document,
                        inputs=inputs,
                        metadata=({"candidate_annotation": rel}),
                    )
                )

                self.collect_relation("used", rel)

        return task_encodings

    def _maybe_log_example(
        self,
        task_encoding: TaskEncodingType,
        target: TargetEncodingType,
    ):
        """Maybe log the example."""

        # log the first n examples
        if self._logged_examples_counter < self.log_first_n_examples:
            input_ids = task_encoding.inputs["input_ids"]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            target_labels = [self.id_to_label[label_id] for label_id in target]
            logger.info("*** Example ***")
            logger.info("doc id: %s", task_encoding.document.id)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("Expected label: %s (ids = %s)", target_labels, target)

            self._logged_examples_counter += 1

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> TargetEncodingType:
        candidate_annotation = task_encoding.metadata["candidate_annotation"]
        if isinstance(candidate_annotation, (BinaryRelation, NaryRelation)):
            labels = [candidate_annotation.label]
        else:
            raise NotImplementedError(
                f"encoding the target with a candidate_annotation of another type than BinaryRelation or"
                f"NaryRelation is not yet supported. candidate_annotation has the type: "
                f"{type(candidate_annotation)}"
            )
        target = [self.label_to_id[label] for label in labels]

        self._maybe_log_example(task_encoding=task_encoding, target=target)

        return target

    def unbatch_output(self, model_output: ModelTargetType) -> Sequence[TaskOutputType]:
        unbatched_output = []
        if self.multi_label:
            raise NotImplementedError
        else:
            label_ids = model_output["labels"].detach().cpu().tolist()
            probabilities = model_output["probabilities"].detach().cpu().tolist()
            for batch_idx in range(len(label_ids)):
                label_id = label_ids[batch_idx]
                result: TaskOutputType = {
                    "labels": [self.id_to_label[label_id]],
                    "probabilities": [probabilities[batch_idx][label_id]],
                }
                unbatched_output.append(result)

        return unbatched_output

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncodingType,
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Union[BinaryRelation, MultiLabeledBinaryRelation, NaryRelation]]]:
        candidate_annotation = task_encoding.metadata["candidate_annotation"]
        new_annotation: Union[BinaryRelation, MultiLabeledBinaryRelation, NaryRelation]
        if self.multi_label:
            raise NotImplementedError
        else:
            label = task_output["labels"][0]
            probability = (
                task_output["probabilities"][0] if "probabilities" in task_output else 1.0
            )
            if isinstance(candidate_annotation, BinaryRelation):
                head = candidate_annotation.head
                tail = candidate_annotation.tail
                # Reverse predicted reversed relations back. Serialization will remove any duplicated relations.
                if self.add_reversed_relations:
                    # TODO: add test case for this
                    if label.endswith(self.reversed_relation_label_suffix):
                        label = label[: -len(self.reversed_relation_label_suffix)]
                        head, tail = tail, head
                    # If the predicted label is symmetric, we sort the arguments by its center.
                    elif label in self.symmetric_relations and self.reverse_symmetric_relations:
                        if not (isinstance(head, Span) and isinstance(tail, Span)):
                            raise ValueError(
                                f"the taskmodule expects the relation arguments of the candidate_annotation"
                                f"to be of type Span, but got head of type: {type(head)} and tail of type: "
                                f"{type(tail)}"
                            )
                        # use a unique order for the arguments: sort by start and end positions
                        head, tail = sorted([head, tail], key=lambda span: (span.start, span.end))
                new_annotation = BinaryRelation(
                    head=head, tail=tail, label=label, score=probability
                )
            elif isinstance(candidate_annotation, NaryRelation):
                # TODO: add test case for this
                if self.add_reversed_relations:
                    raise ValueError("can not reverse a NaryRelation")
                new_annotation = NaryRelation(
                    arguments=candidate_annotation.arguments,
                    roles=candidate_annotation.roles,
                    label=label,
                    score=probability,
                )
            else:
                raise NotImplementedError(
                    f"creating a new annotation from a candidate_annotation of another type than BinaryRelation is "
                    f"not yet supported. candidate_annotation has the type: {type(candidate_annotation)}"
                )
            if not (self.add_candidate_relations and label == self.none_label):
                yield self.relation_annotation, new_annotation

    def _get_global_attention(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        # we want to have global attention on all marker tokens and the cls token
        positive_token_ids = list(self.argument_markers_to_id.values()) + [
            self.tokenizer.cls_token_id
        ]
        global_attention_mask = construct_mask(
            input_ids=input_ids, positive_ids=positive_token_ids
        )
        return global_attention_mask

    def collate(
        self, task_encodings: Sequence[TaskEncodingType]
    ) -> Tuple[ModelInputType, Optional[ModelTargetType]]:
        input_features = [
            {"input_ids": task_encoding.inputs["input_ids"]} for task_encoding in task_encodings
        ]

        inputs: Dict[str, torch.LongTensor] = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if self.add_argument_indices_to_input:
            inputs["pooler_start_indices"] = torch.tensor(
                [task_encoding.inputs["pooler_start_indices"] for task_encoding in task_encodings]
            ).to(torch.long)
            inputs["pooler_end_indices"] = torch.tensor(
                [task_encoding.inputs["pooler_end_indices"] for task_encoding in task_encodings]
            ).to(torch.long)

        if self.add_global_attention_mask_to_input:
            inputs["global_attention_mask"] = self._get_global_attention(
                input_ids=inputs["input_ids"]
            )

        if not task_encodings[0].has_targets:
            return inputs, None

        target_list: List[TargetEncodingType] = [
            task_encoding.targets for task_encoding in task_encodings
        ]
        targets = torch.tensor(target_list, dtype=torch.int64)

        if not self.multi_label:
            targets = targets.flatten()

        return inputs, {"labels": targets}

    def configure_model_metric(self, stage: str) -> MetricCollection:
        if self.label_to_id is None:
            raise ValueError(
                "The taskmodule has not been prepared yet, so label_to_id is not known. "
                "Please call taskmodule.prepare(documents) before configuring the model metric "
                "or pass the labels to the taskmodule constructor an call taskmodule.post_prepare()."
            )
        # we use the length of label_to_id because that contains the none_label (in contrast to labels)
        labels = [self.id_to_label[i] for i in range(len(self.label_to_id))]
        common_metric_kwargs = {
            "num_classes": len(labels),
            "task": "multilabel" if self.multi_label else "multiclass",
        }
        return MetricCollection(
            {
                "with_tn": WrappedMetricWithPrepareFunction(
                    metric=MetricCollection(
                        {
                            "micro/f1": F1Score(average="micro", **common_metric_kwargs),
                            "macro/f1": F1Score(average="macro", **common_metric_kwargs),
                            "f1_per_label": ClasswiseWrapper(
                                F1Score(average=None, **common_metric_kwargs),
                                labels=labels,
                                postfix="/f1",
                            ),
                        }
                    ),
                    prepare_function=_get_labels,
                ),
                # We can not easily calculate the macro f1 here, because
                # F1Score with average="macro" would still include the none_label.
                "micro/f1_without_tn": WrappedMetricWithPrepareFunction(
                    metric=F1Score(average="micro", **common_metric_kwargs),
                    prepare_together_function=partial(
                        _get_labels_together_remove_none_label,
                        none_idx=self.label_to_id[self.none_label],
                    ),
                ),
            }
        )
