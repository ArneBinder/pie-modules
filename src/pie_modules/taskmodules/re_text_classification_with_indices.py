"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

import logging
from collections import defaultdict
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
import pandas as pd
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
from pytorch_ie.utils.span import get_token_slice, has_overlap, is_contained_in
from pytorch_ie.utils.window import get_window_around_slice
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from typing_extensions import TypeAlias

from pie_modules.models.sequence_classification import (
    ModelOutputType,
    ModelStepInputType,
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
    ModelStepInputType,
    ModelOutputType,
    TaskOutputType,
]


HEAD = "head"
TAIL = "tail"
START = "start"
END = "end"


logger = logging.getLogger(__name__)


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


class RelationArgument:
    def __init__(
        self,
        entity: LabeledSpan,
        role: str,
        token_span: Span,
        add_type_to_marker: bool,
        role_to_marker: Dict[str, str],
    ) -> None:
        self.entity = entity
        self.role_to_marker = role_to_marker
        if role not in self.role_to_marker:
            raise ValueError(
                f"role={role} not in role_to_marker={role_to_marker} (did you initialise the taskmodule "
                f"with the correct argument_role_to_marker dictionary?)"
            )
        self.role = role
        self.token_span = token_span
        self.add_type_to_marker = add_type_to_marker

    @property
    def as_start_marker(self) -> str:
        return self._get_marker(is_start=True)

    @property
    def as_end_marker(self) -> str:
        return self._get_marker(is_start=False)

    @property
    def role_marker(self) -> str:
        return self.role_to_marker[self.role]

    def _get_marker(self, is_start: bool = True) -> str:
        return f"[{'' if is_start else '/'}{self.role_marker}" + (
            f":{self.entity.label}]" if self.add_type_to_marker else "]"
        )

    @property
    def as_append_marker(self) -> str:
        return f"[{self.role_marker}={self.entity.label}]"

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


@TaskModule.register()
class RETextClassificationWithIndicesTaskModule(TaskModuleType, ChangesTokenizerVocabSize):
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
        log_first_n_examples: int = 0,
        add_argument_indices_to_input: bool = False,
        collect_statistics: bool = False,
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
        # overwrite None with 0 for backward compatibility
        self.log_first_n_examples = log_first_n_examples or 0
        self.add_argument_indices_to_input = add_argument_indices_to_input
        if argument_role_to_marker is None:
            self.argument_role_to_marker = {HEAD: "H", TAIL: "T"}
        else:
            self.argument_role_to_marker = argument_role_to_marker
        self.collect_statistics = collect_statistics

        self.argument_role2idx = {
            role: i for i, role in enumerate(sorted(self.argument_role_to_marker))
        }

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        self.argument_markers = None

        self._logged_examples_counter = 0

        self.reset_statistics()

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

    def reset_statistics(self):
        self._statistics = defaultdict(int)
        self._collected_relations: Dict[str, List[Annotation]] = defaultdict(list)

    def collect_relation(self, kind: str, relation: Annotation):
        if self.collect_statistics:
            self._collected_relations[kind].append(relation)

    def collect_all_relations(self, kind: str, relations: Iterable[Annotation]):
        if self.collect_statistics:
            self._collected_relations[kind].extend(relations)

    def finalize_statistics(self):
        if self.collect_statistics:
            all_relations = set(self._collected_relations["available"])
            used_relations = set(self._collected_relations["used"])
            skipped_other = all_relations - used_relations
            for key, rels in self._collected_relations.items():
                rels_set = set(rels)
                if key.startswith("skipped_"):
                    skipped_other -= rels_set
                elif key.startswith("used_"):
                    pass
                elif key in ["available", "used"]:
                    pass
                else:
                    raise ValueError(f"unknown key: {key}")
                for rel in rels_set:
                    self.increase_counter(key=(key, rel.label))
            for rel in skipped_other:
                self.increase_counter(key=("skipped_other", rel.label))

    def show_statistics(self):
        if self.collect_statistics:
            self.finalize_statistics()

            to_show = pd.Series(self._statistics)
            if len(to_show.index.names) > 1:
                to_show = to_show.unstack()
            logger.info(f"statistics:\n{to_show.to_markdown()}")

    def increase_counter(self, key: Tuple[Any, ...], value: Optional[int] = 1):
        if self.collect_statistics:
            key_str = tuple(str(k) for k in key)
            self._statistics[key_str] += value

    def encode(self, *args, **kwargs):
        self.reset_statistics()
        res = super().encode(*args, **kwargs)
        self.show_statistics()
        return res

    def construct_argument_markers(self) -> List[str]:
        # ignore the typing because we know that this is only called on a prepared taskmodule,
        # i.e. self.entity_labels is already set by _prepare or __init__
        entity_labels: List[str] = self.entity_labels  # type: ignore
        argument_markers: Set[str] = set()
        for arg_role, role_marker in self.argument_role_to_marker.items():
            for arg_pos in [START, END]:
                is_start = arg_pos == START
                argument_markers.add(f"[{'' if is_start else '/'}{role_marker}]")
                if self.add_type_to_marker:
                    for entity_type in entity_labels:
                        argument_markers.add(
                            f"[{'' if is_start else '/'}{role_marker}"
                            f"{':' + entity_type if self.add_type_to_marker else ''}]"
                        )
                if self.append_markers:
                    for entity_type in entity_labels:
                        argument_markers.add(f"[{role_marker}={entity_type}]")

        return sorted(list(argument_markers))

    def _post_prepare(self):
        self.label_to_id = {label: i + 1 for i, label in enumerate(self.labels)}
        self.label_to_id[self.none_label] = 0
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.argument_markers = self.construct_argument_markers()
        self.tokenizer.add_tokens(self.argument_markers, special_tokens=True)

        self.argument_markers_to_id = {
            marker: self.tokenizer.vocab[marker] for marker in self.argument_markers
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
            if set(self.argument_role_to_marker) == {HEAD, TAIL}:
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
                    f"with argument roles other than 'head' and 'tail': {sorted(self.argument_role_to_marker)}"
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

                # map character spans to token spans
                arg_token_slices_including_none = [
                    get_token_slice(
                        character_slice=(arg.start, arg.end),
                        char_to_token_mapper=encoding.char_to_token,
                        character_offset=partition.start,
                    )
                    for arg in arg_spans
                ]
                # Check if the mapping was successful. It may fail (and is None) if any argument start or end does not
                # match a token start or end, respectively.
                if any(token_slice is None for token_slice in arg_token_slices_including_none):
                    arg_spans_dict = {arg_span: str(arg_span) for arg_span in arg_spans}
                    logger.warning(
                        f"doc.id={document.id}: Skipping invalid example, cannot get argument token slices for "
                        f"{arg_spans_dict}"
                    )
                    self.collect_relation("skipped_args_not_aligned", rel)
                    continue

                # ignore the typing, because we checked for None above
                arg_token_slices: List[Tuple[int, int]] = arg_token_slices_including_none  # type: ignore

                # create the argument objects
                args = [
                    RelationArgument(
                        entity=span,
                        role=role,
                        token_span=Span(start=token_slice[0], end=token_slice[1]),
                        add_type_to_marker=self.add_type_to_marker,
                        role_to_marker=self.argument_role_to_marker,
                    )
                    for span, role, token_slice in zip(arg_spans, arg_roles, arg_token_slices)
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
        if isinstance(candidate_annotation, BinaryRelation):
            labels = [candidate_annotation.label]
        else:
            raise NotImplementedError(
                f"encoding the target with a candidate_annotation of another type than BinaryRelation is "
                f"not yet supported. candidate_annotation has the type: {type(candidate_annotation)}"
            )
        target = [self.label_to_id[label] for label in labels]

        self._maybe_log_example(task_encoding=task_encoding, target=target)

        return target

    def unbatch_output(self, model_output: ModelOutputType) -> Sequence[TaskOutputType]:
        logits = model_output["logits"]

        output_label_probs = logits.sigmoid() if self.multi_label else logits.softmax(dim=-1)
        output_label_probs = output_label_probs.detach().cpu().numpy()

        unbatched_output = []
        if self.multi_label:
            raise NotImplementedError
        else:
            label_ids = np.argmax(output_label_probs, axis=-1)
            for batch_idx, label_id in enumerate(label_ids):
                label = self.id_to_label[label_id]
                prob = float(output_label_probs[batch_idx, label_id])
                result: TaskOutputType = {
                    "labels": [label],
                    "probabilities": [prob],
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
            probability = task_output["probabilities"][0]
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
                if self.reversed_relation_label_suffix is not None:
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

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> ModelStepInputType:
        input_features = [
            {"input_ids": task_encoding.inputs["input_ids"]} for task_encoding in task_encodings
        ]

        inputs: Dict[str, torch.Tensor] = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if self.add_argument_indices_to_input:
            inputs["pooler_start_indices"] = torch.tensor(
                [task_encoding.inputs["pooler_start_indices"] for task_encoding in task_encodings]
            )
            inputs["pooler_end_indices"] = torch.tensor(
                [task_encoding.inputs["pooler_end_indices"] for task_encoding in task_encodings]
            )

        if not task_encodings[0].has_targets:
            return inputs, None

        target_list: List[TargetEncodingType] = [
            task_encoding.targets for task_encoding in task_encodings
        ]
        targets = torch.tensor(target_list, dtype=torch.int64)

        if not self.multi_label:
            targets = targets.flatten()

        return inputs, targets
