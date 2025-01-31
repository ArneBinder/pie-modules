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
from copy import deepcopy
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

import pandas as pd
import torch
from pytorch_ie.annotations import (
    BinaryRelation,
    LabeledSpan,
    MultiLabeledBinaryRelation,
    NaryRelation,
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
    TextDocumentWithLabeledPartitions,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from pytorch_ie.taskmodules.interface import ChangesTokenizerVocabSize
from tokenizers import AddedToken
from torch import BoolTensor, LongTensor, Tensor
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import ClasswiseWrapper, F1Score, Metric, MetricCollection
from transformers import AutoTokenizer
from typing_extensions import TypeAlias

from pie_modules.document.processing import (
    token_based_document_to_text_based,
    tokenize_document,
)
from pie_modules.documents import (
    TokenDocumentWithLabeledSpansAndBinaryRelations,
    TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from pie_modules.taskmodules.metrics import WrappedMetricWithPrepareFunction
from pie_modules.utils.span import distance as get_span_distance

PAD_VALUES = {
    "input_ids": 0,
    "attention_mask": 0,
    "span_start_indices": 0,
    "span_end_indices": 0,
    "tuple_indices": -1,
    "labels": -100,
    "tuple_indices_mask": False,
}
DTYPES = {
    "input_ids": torch.long,
    "attention_mask": torch.long,
    "span_start_indices": torch.long,
    "span_end_indices": torch.long,
    "tuple_indices": torch.long,
    "labels": torch.long,
    "tuple_indices_mask": torch.bool,
}


class InputEncodingType(TypedDict, total=False):
    # shape: (sequence_length,)
    input_ids: LongTensor
    # shape: (sequence_length,)
    attention_mask: LongTensor
    # shape: (num_entities,)
    span_start_indices: LongTensor
    # shape: (num_entities,)
    span_end_indices: LongTensor
    # list of lists of argument indices: [[head_idx, tail_idx], ...]
    # NOTE: these indices point into span_start_indices and span_end_indices!
    tuple_indices: LongTensor
    tuple_indices_mask: BoolTensor


class TargetEncodingType(TypedDict, total=False):
    # list of label indices: [label_idx, ...]
    labels: LongTensor


DocumentType: TypeAlias = TextDocument
TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]


class TaskOutputType(TypedDict, total=False):
    labels: Sequence[str]
    probabilities: Sequence[float]


class ModelInputType(TypedDict, total=False):
    input_ids: LongTensor
    attention_mask: LongTensor
    span_start_indices: LongTensor
    span_end_indices: LongTensor
    tuple_indices: LongTensor
    tuple_indices_mask: BoolTensor


class ModelTargetType(TypedDict, total=False):
    labels: LongTensor
    probabilities: LongTensor


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


def _get_label_ids_from_model_output(
    model_output: ModelTargetType,
) -> LongTensor:
    return model_output["labels"]


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


def construct_argument_marker(pos: str, label: Optional[str] = None, role: str = "SPAN") -> str:
    if pos not in [START, END]:
        raise ValueError(f"pos must be one of {START} or {END}, but got: {pos}")
    start_or_end_marker = "" if pos == START else "/"
    if label is not None:
        return f"[{start_or_end_marker}{role}:{label}]"
    else:
        return f"[{start_or_end_marker}{role}]"


def inject_markers_into_text(
    text: str, positions_and_markers: List[Tuple[int, str]]
) -> Tuple[str, Dict[int, int]]:
    offset = 0
    original2new_pos = dict()
    for original_pos, marker in sorted(positions_and_markers):
        text = text[: original_pos + offset] + marker + text[original_pos + offset :]
        offset += len(marker)
        original2new_pos[original_pos] = original_pos + offset
    return text, original2new_pos


def to_tensor(key: str, value: Any) -> Tensor:
    return torch.tensor(value, dtype=DTYPES[key])


def pad_or_stack(key: str, values: List[LongTensor]) -> Tensor:
    if key in PAD_VALUES:
        max_last_dim = None
        if key == "tuple_indices":
            max_last_dim = max(v.shape[-1] for v in values if len(v.shape) == 2)
            values = [v.reshape(-1) for v in values]
        result = pad_sequence(values, batch_first=True, padding_value=PAD_VALUES[key])
        if key == "tuple_indices":
            batch_size = len(values)
            result = result.reshape(batch_size, -1, max_last_dim)
    else:
        result = torch.stack(values, dim=0)
    return result


@TaskModule.register()
class RESpanPairClassificationTaskModule(TaskModuleType, ChangesTokenizerVocabSize):
    """Task module for relation extraction as span pair classification.

    This task module frames relation extraction as a span pair classification task where all candidate
    pairs in a given text are classified at once. The task module injects start and end markers for
    each entity (i.e. "[SPAN]" and "[/SPAN]") into the text and tokenizes the text (the markers are
    handled as special tokens, and thus, kept as they are). It then collects the start- and end-marker
    positions for each entity and constructs a model input encoding from the tokenized text and these
    positions. The model target encoding consists of a list of label indices and a list of tuples
    (head and tail) of argument indices that point into the start- and end-marker positions from the
    model inputs. The model output is expected to be of the same format as the model target encoding,
    but with probabilities for each label.

    This means, that the model should return only positive relations (argument indices + label) and
    discard all negative ones.

    Args:
        tokenizer_name_or_path: The name or path of the tokenizer to use.
        relation_annotation: The name of the annotation layer that contains the binary relations.
        partition_annotation: The name of the annotation layer that contains the labeled partitions.
            If provided, the task module expects the document to have a partition layer with the
            given name containing LabeledSpans. These entries are used to split the text into
            partitions, e.g. paragraphs or sentences, that are treated as separate documents during
            tokenization. Defaults to None.
        tokenize_kwargs: Additional keyword arguments passed to the tokenizer during tokenization.
        create_candidate_relations: Whether to create candidate relations for training. If True, the
            task module creates all possible pairs of entities in the text as candidate relations.
            Defaults to False.
        create_candidate_relations_kwargs: Additional keyword arguments passed to the method that
            creates the candidate relations (e.g. max_argument_distance). Defaults to None.
        labels: The list of relation labels. If not provided, the task module will collect the labels
            from the documents during preparation. Defaults to None.
        entity_labels: The list of entity labels. If not provided, the task module will collect the
            entity labels from the documents during preparation. Defaults to None.
        add_type_to_marker: Whether to add the entity type to the markers. If True, the markers will
            look like this: "[SPAN:entity_type]" and "[/SPAN:entity_type]" where entity_type is the
            type of the respective entity. Defaults to False.
        log_first_n_examples: The number of examples to log during training. If 0, no examples are logged.
            Defaults to 0.
        collect_statistics: Whether to collect statistics during preparation. If True, the task module
            will collect statistics about the available, used, and skipped relations. Defaults to False.
    """

    PREPARED_ATTRIBUTES = ["labels", "entity_labels"]

    def __init__(
        self,
        tokenizer_name_or_path: str,
        relation_annotation: str = "binary_relations",
        no_relation_label: str = "no_relation",
        partition_annotation: Optional[str] = None,
        tokenize_kwargs: Optional[Dict[str, Any]] = None,
        create_candidate_relations: bool = False,
        create_candidate_relations_kwargs: Optional[Dict[str, Any]] = None,
        labels: Optional[List[str]] = None,
        entity_labels: Optional[List[str]] = None,
        add_type_to_marker: bool = True,
        log_first_n_examples: int = 0,
        collect_statistics: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.relation_annotation = relation_annotation
        self.no_relation_label = no_relation_label
        self.tokenize_kwargs = tokenize_kwargs or {}
        self.create_candidate_relations = create_candidate_relations
        self.create_candidate_relations_kwargs = create_candidate_relations_kwargs or {}
        self.labels = labels
        self.add_type_to_marker = add_type_to_marker
        self.entity_labels = entity_labels
        self.partition_annotation = partition_annotation
        # overwrite None with 0 for backward compatibility
        self.log_first_n_examples = log_first_n_examples or 0
        self.collect_statistics = collect_statistics

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

    @property
    def tokenized_document_type(self) -> Type[TokenDocumentWithLabeledSpansAndBinaryRelations]:
        if self.partition_annotation is None:
            return TokenDocumentWithLabeledSpansAndBinaryRelations
        else:
            return TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions

    @property
    def normalized_document_type(self) -> Type[TextDocumentWithLabeledSpansAndBinaryRelations]:
        if self.partition_annotation is None:
            return TextDocumentWithLabeledSpansAndBinaryRelations
        else:
            return TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions

    def normalize_document(self, document) -> TextDocumentWithLabeledSpansAndBinaryRelations:
        span_layer_name = self.get_span_layer_name(document)
        field_mapping = {
            span_layer_name: "labeled_spans",
            self.relation_annotation: "binary_relations",
        }
        if self.partition_annotation is not None:
            field_mapping[self.partition_annotation] = "labeled_partitions"
        casted_document = document.as_type(
            self.normalized_document_type, field_mapping=field_mapping
        )
        return casted_document

    def get_relation_layer(self, document: Document) -> AnnotationList[BinaryRelation]:
        return document[self.relation_annotation]

    def get_span_layer_name(self, document: Document) -> str:
        return document[self.relation_annotation].target_name

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

        if self.no_relation_label in relation_labels:
            relation_labels.remove(self.no_relation_label)

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
            all_relations = set(self._collected_relations["available_tokenized"])
            used_relations = set(self._collected_relations["used"])
            skipped_other = all_relations - used_relations
            for key, rels in self._collected_relations.items():
                rels_set = set(rels)
                if key.startswith("skipped_"):
                    skipped_other -= rels_set
                elif key.startswith("used_"):
                    pass
                elif key in ["available", "available_tokenized", "used"]:
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

    def collect_argument_markers(self, entity_labels: Iterable[str]) -> List[str]:
        argument_markers: Set[str] = set()
        for arg_pos in [START, END]:
            if self.add_type_to_marker:
                for entity_label in entity_labels:
                    argument_markers.add(
                        construct_argument_marker(pos=arg_pos, label=entity_label)
                    )
            else:
                argument_markers.add(construct_argument_marker(pos=arg_pos))

        return sorted(list(argument_markers))

    def _post_prepare(self):
        self.label_to_id = {label: i + 1 for i, label in enumerate(self.labels)}
        self.label_to_id[self.no_relation_label] = 0
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.argument_markers = self.collect_argument_markers(self.entity_labels)
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.argument_markers}
        )
        if len(self.argument_markers) != num_added:
            logger.warning(
                f"expected to add {len(self.argument_markers)} argument markers, but added {num_added}. It seems "
                f"that the tokenizer already contains some of the argument markers."
            )

        self.argument_markers_to_id = {
            marker: self.tokenizer.vocab[marker] for marker in self.argument_markers
        }

    def _create_candidate_relations(
        self,
        document: TokenDocumentWithLabeledSpansAndBinaryRelations,
        max_argument_distance: Optional[int] = None,
        argument_distance_type: str = "inner",
    ) -> Sequence[Annotation]:
        # TODO: ensure that the relation layer type is BinaryRelation!
        labeled_spans = document.labeled_spans
        candidate_relations = []
        for i, head in enumerate(labeled_spans):
            for j, tail in enumerate(labeled_spans):
                if i == j:
                    continue
                rel = BinaryRelation(head=head, tail=tail, label=self.no_relation_label)
                if max_argument_distance is not None:
                    arg_distance = get_span_distance(
                        start_end=(head.start, head.end),
                        other_start_end=(tail.start, tail.end),
                        distance_type=argument_distance_type,
                    )
                    if arg_distance > max_argument_distance:
                        self.collect_relation("skipped_argument_distance", rel)
                        continue
                candidate_relations.append(rel)
        return candidate_relations

    def inject_markers_for_labeled_spans(
        self,
        document: TextDocumentWithLabeledSpansAndBinaryRelations,
    ) -> Tuple[TextDocumentWithLabeledSpansAndBinaryRelations, Dict[LabeledSpan, LabeledSpan]]:
        # collect markers and injection positions
        positions_and_markers = []
        for labeled_span in document.labeled_spans:
            label_or_none = labeled_span.label if self.add_type_to_marker else None
            start_marker = construct_argument_marker(pos=START, label=label_or_none)
            positions_and_markers.append((labeled_span.start, start_marker))
            end_marker = construct_argument_marker(pos=END, label=label_or_none)
            positions_and_markers.append((labeled_span.end, end_marker))

        if isinstance(document, TextDocumentWithLabeledPartitions):
            # create "dummy" markers for the partitions so that entries for these positions are created
            # in original2new_pos
            for labeled_partition in document.labeled_partitions:
                positions_and_markers.append((labeled_partition.start, ""))
                positions_and_markers.append((labeled_partition.end, ""))

        # inject markers into the text
        marked_text, original2new_pos = inject_markers_into_text(
            document.text, positions_and_markers
        )

        # construct new spans
        old2new_spans = dict()
        for labeled_span in document.labeled_spans:
            start = original2new_pos[labeled_span.start]
            end = original2new_pos[labeled_span.end]
            new_span = LabeledSpan(start=start, end=end, label=labeled_span.label)
            old2new_spans[labeled_span] = new_span

        # construct new relations
        old2new_relations = dict()
        for relation in document.binary_relations:
            if isinstance(relation, BinaryRelation):
                head = old2new_spans[relation.head]
                tail = old2new_spans[relation.tail]
                new_relation = BinaryRelation(head=head, tail=tail, label=relation.label)
            else:
                raise NotImplementedError(
                    f"the taskmodule does not yet support relations of type {type(relation)}"
                )
            old2new_relations[relation] = new_relation

        # construct new document
        new_document = type(document)(
            id=document.id,
            metadata=deepcopy(document.metadata),
            text=marked_text,
        )
        new_document.labeled_spans.extend(old2new_spans.values())
        new_document.binary_relations.extend(old2new_relations.values())
        if isinstance(document, TextDocumentWithLabeledPartitions):
            for labeled_partition in document.labeled_partitions:
                new_start = original2new_pos[labeled_partition.start]
                new_end = original2new_pos[labeled_partition.end]
                new_labeled_partitions = labeled_partition.copy(start=new_start, end=new_end)
                new_document.labeled_partitions.append(new_labeled_partitions)

        new2old_spans = {new_span: old_span for old_span, new_span in old2new_spans.items()}
        return new_document, new2old_spans

    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        self.collect_all_relations("available", self.get_relation_layer(document))

        # 1. inject start and end markers for each entity into the text
        #   - save mapping from new entities to original entities
        # 2. tokenize the text
        #   - add the marker tokens to the tokenizer as special tokens
        #   - tokenize with tokenize_document()
        # 3. get start- and end-token positions for each entity
        # 4. construct task encoding from tokenized text and entity positions

        normalized_document = self.normalize_document(document)
        document_with_markers, injected2original_spans = self.inject_markers_for_labeled_spans(
            normalized_document
        )
        all_added_annotations: List[Dict[str, Dict[Annotation, Annotation]]] = []
        tokenized_docs = tokenize_document(
            document_with_markers,
            tokenizer=self.tokenizer,
            result_document_type=self.tokenized_document_type,
            partition_layer=(
                "labeled_partitions" if self.partition_annotation is not None else None
            ),
            added_annotations=all_added_annotations,
            strict_span_conversion=False,
            **self.tokenize_kwargs,
        )

        task_encodings: List[TaskEncodingType] = []
        for tokenized_doc, tokenized_annotations in zip(tokenized_docs, all_added_annotations):
            self.collect_all_relations("available_tokenized", tokenized_doc.binary_relations)
            # collect start- and end-token positions for each entity
            span_start_indices = []
            span_end_indices = []
            for labeled_span in tokenized_doc.labeled_spans:
                # the start marker is one token before the start of the span
                span_start_indices.append(labeled_span.start - 1)
                # the end marker is one token after the end of the span, but the end index is exclusive
                span_end_indices.append(labeled_span.end)

            labeled_span2idx = {span: idx for idx, span in enumerate(tokenized_doc.labeled_spans)}
            tuple_indices = []  # list of lists of argument indices: [[head_idx, tail_idx], ...]
            if self.create_candidate_relations:
                candidate_relations = self._create_candidate_relations(
                    tokenized_doc, **self.create_candidate_relations_kwargs
                )
            else:
                candidate_relations = tokenized_doc.binary_relations

            # if there are no candidate relations, skip the whole (tokenized) document
            if len(candidate_relations) == 0:
                continue

            for relation in candidate_relations:
                current_args_indices = []
                for _, arg_span in get_relation_argument_spans_and_roles(relation):
                    arg_idx = labeled_span2idx[arg_span]
                    current_args_indices.append(arg_idx)
                tuple_indices.append(current_args_indices)

            encoding = tokenized_doc.metadata["tokenizer_encoding"]
            inputs = {
                "input_ids": encoding.ids,
                "attention_mask": encoding.attention_mask,
                "span_start_indices": span_start_indices,
                "span_end_indices": span_end_indices,
                "tuple_indices": tuple_indices,
                "tuple_indices_mask": [True] * len(tuple_indices),
            }
            inputs_tensors = {k: to_tensor(k, v) for k, v in inputs.items()}
            task_encodings.append(
                TaskEncoding(
                    document=document,
                    inputs=inputs_tensors,
                    metadata={
                        "tokenized_document": tokenized_doc,
                        "injected2original_spans": injected2original_spans,
                        "candidate_relations": candidate_relations,
                        "tokenized_annotations": tokenized_annotations,
                    },
                )
            )

        return task_encodings

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> TargetEncodingType:
        gold_relations = task_encoding.metadata["tokenized_document"].binary_relations
        gold_roles_and_args2relation = defaultdict(list)
        for relation in gold_relations:
            # If we manually set the labels, we only consider relations with a label in the label_to_id mapping
            # This allows us to ignore relations with certain labels during training.
            if relation.label in self.label_to_id:
                gold_roles_and_args2relation[
                    get_relation_argument_spans_and_roles(relation)
                ].append(relation)
        label_indices = []  # list of label indices
        candidate_relations = []
        for candidate_relation in task_encoding.metadata["candidate_relations"]:
            candidate_roles_and_args = get_relation_argument_spans_and_roles(candidate_relation)
            gold_relations = gold_roles_and_args2relation.get(candidate_roles_and_args, [])
            if len(gold_relations) == 0:
                label_idx = self.label_to_id[candidate_relation.label]
                self.collect_relation("used", candidate_relation)
            elif len(gold_relations) == 1:
                label_idx = self.label_to_id[gold_relations[0].label]
                self.collect_relation("used", gold_relations[0])
            else:
                # TODO: or should we add all gold relations with the same arguments?
                logger.warning(
                    f"skip the candidate relation because there are more than one gold relation "
                    f"for its args and roles: {gold_relations}"
                )
                for gold_relation in gold_relations:
                    self.collect_relation("skipped_same_arguments", gold_relation)
                label_idx = PAD_VALUES["labels"]

            label_indices.append(label_idx)
            candidate_relations.append(candidate_relation)

        task_encoding.metadata["candidate_relations"] = candidate_relations
        target: TargetEncodingType = {"labels": to_tensor("labels", label_indices)}

        self._maybe_log_example(task_encoding=task_encoding, target=target)

        return target

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
            logger.info("*** Example ***")
            logger.info(f"doc id: {task_encoding.document.id}")
            logger.info(f"tokens: {' '.join([x for x in tokens])}")
            logger.info(f"input_ids: {' '.join([str(x) for x in input_ids.tolist()])}")
            # target data
            span_start_indices = task_encoding.inputs["span_start_indices"]
            span_end_indices = task_encoding.inputs["span_end_indices"]
            labels = [self.id_to_label[label] for label in target["labels"].tolist()]
            for i, (label, tuple_indices) in enumerate(
                zip(labels, task_encoding.inputs["tuple_indices"])
            ):
                logger.info(f"relation {i}: {label}")
                for j, arg_idx in enumerate(tuple_indices):
                    arg_tokens = tokens[span_start_indices[arg_idx] : span_end_indices[arg_idx]]
                    logger.info(f"\targ {j}: {' '.join([str(x) for x in arg_tokens])}")

            self._logged_examples_counter += 1

    def collate(
        self, task_encodings: Sequence[TaskEncodingType]
    ) -> Tuple[ModelInputType, Optional[ModelTargetType]]:
        input_keys = task_encodings[0].inputs.keys()
        inputs: ModelInputType = {  # type: ignore
            key: pad_or_stack(key, [task_encoding.inputs[key] for task_encoding in task_encodings])
            for key in input_keys
        }

        targets: Optional[ModelTargetType] = None
        if task_encodings[0].has_targets:
            target_keys = task_encodings[0].targets.keys()
            targets: ModelTargetType = {  # type: ignore
                key: pad_or_stack(
                    key, [task_encoding.targets[key] for task_encoding in task_encodings]
                )
                for key in target_keys
            }

        return inputs, targets

    def unbatch_output(self, model_output: ModelTargetType) -> Sequence[TaskOutputType]:
        # shape: (batch_size, num_candidates)
        label_ids = model_output["labels"].detach().cpu().tolist()
        # shape: (batch_size, num_candidates, num_labels)
        all_probabilities = model_output["probabilities"].detach().cpu().tolist()
        unbatched_output = []
        for batch_idx in range(len(label_ids)):
            labels = []
            probabilities = []
            for label_id, probs in zip(label_ids[batch_idx], all_probabilities[batch_idx]):
                labels.append(self.id_to_label[label_id])
                probabilities.append(probs[label_id])
            entry: TaskOutputType = {
                "labels": labels,
                "probabilities": probabilities,
            }
            unbatched_output.append(entry)

        return unbatched_output

    def decode_annotations(
        self,
        task_output: TaskOutputType,
        task_encoding: TaskEncodingType,
    ) -> Dict[str, List[Annotation]]:
        char2token_spans = task_encoding.metadata["tokenized_annotations"]["labeled_spans"]
        token2char_spans = {v: k for k, v in char2token_spans.items()}
        injected2original_spans = task_encoding.metadata["injected2original_spans"]
        new_relations = []
        for candidate_relation, label, probability, is_valid in zip(
            task_encoding.metadata["candidate_relations"],
            task_output["labels"],
            task_output["probabilities"],
            task_encoding.inputs["tuple_indices_mask"],
        ):
            # exclude
            # - padding entries (is_valid=False)
            # - negative relations (if we have added them)
            if is_valid and (
                label != self.no_relation_label or not self.create_candidate_relations
            ):
                token_head, token_tail = candidate_relation.head, candidate_relation.tail
                char_head = token2char_spans[token_head]
                char_tail = token2char_spans[token_tail]
                original_head = injected2original_spans[char_head]
                original_tail = injected2original_spans[char_tail]
                new_annotation = candidate_relation.copy(
                    head=original_head, tail=original_tail, label=label, score=probability
                )
                new_relations.append(new_annotation)

        return {"binary_relations": new_relations}

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncodingType,
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Union[BinaryRelation, MultiLabeledBinaryRelation, NaryRelation]]]:
        decoded_annotations = self.decode_annotations(
            task_output=task_output, task_encoding=task_encoding
        )

        for relation in decoded_annotations["binary_relations"]:
            yield self.relation_annotation, relation

    def configure_model_metric(self, stage: str) -> Metric:
        if self.label_to_id is None:
            raise ValueError(
                "The taskmodule has not been prepared yet, so label_to_id is not known. "
                "Please call taskmodule.prepare(documents) before configuring the model metric "
                "or pass the labels to the taskmodule constructor an call taskmodule.post_prepare()."
            )
        labels = [self.id_to_label[i] for i in range(len(self.label_to_id))]
        common_metric_kwargs = {
            "num_classes": len(labels),
            "task": "multiclass",
            "ignore_index": PAD_VALUES["labels"],
        }
        return WrappedMetricWithPrepareFunction(
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
            prepare_function=_get_label_ids_from_model_output,
        )
