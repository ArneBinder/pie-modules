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
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from pytorch_ie.taskmodules.interface import ChangesTokenizerVocabSize
from torch import LongTensor
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
    # input_ids and attention_mask should come already padded from the tokenizer
    # "input_ids": 0,
    # "attention_mask": 0,
    "start_marker_positions": 0,
    "end_marker_positions": 0,
    "arg_indices": 0,
    "labels": -100,
}


class InputEncodingType(TypedDict, total=False):
    # shape: (sequence_length,)
    input_ids: LongTensor
    # shape: (sequence_length,)
    attention_mask: LongTensor
    # shape: (num_entities,)
    start_marker_positions: LongTensor
    # shape: (num_entities,)
    end_marker_positions: LongTensor
    # list of lists of argument indices: [[head_idx, tail_idx], ...]
    # NOTE: these indices point into start_marker_positions and end_marker_positions!
    arg_indices: LongTensor


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
    start_marker_positions: LongTensor
    end_marker_positions: LongTensor
    arg_indices: LongTensor


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


def _get_labels_from_model_output(
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


def construct_argument_marker(pos: str, label: Optional[str] = None, role: str = "ARG") -> str:
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


@TaskModule.register()
class RESpanPairClassificationTaskModule(TaskModuleType, ChangesTokenizerVocabSize):
    """Task module for relation extraction as span pair classification.

    This task module frames relation extraction as a span pair classification task where all candidate
    pairs in a given text are classified at once. The task module injects start and end markers for
    each entity into the text and tokenizes the text (the markers are handled as special tokens, and
    thus, kept as they are). It then collects the start- and end-marker positions for each entity and
    constructs a model input encoding from the tokenized text and these positions. The model target
    encoding consists of a list of label indices and a list of tuples (head and tail) of argument
    indices that point into the start- and end-marker positions from the model inputs. The model
    output is expected to be of the same format as the model target encoding, but with probabilities
    for each label.

    This means, that the model should return only positive relations (argument indices + label) and
    discard all negative ones.

    Args:
        tokenizer_name_or_path: The name or path of the tokenizer to use.
        relation_annotation: The name of the annotation layer that contains the binary relations.
        partition_annotation: The name of the annotation layer that contains the labeled partitions.
            If provided, the task module expects the document to have a partition layer with the
            given name. The partition layer is used to split the text into partitions, e.g. paragraphs
            or sentences, that are treated as separate documents during tokenization. Defaults to None.
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
        add_type_to_marker: Whether to add the entity type to the marker. If True, the marker will
            look like this: [START:entity_type] and [END:entity_type]. If False, the marker will look
            like this: [START] and [END]. Defaults to False.
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
        add_type_to_marker: bool = False,
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
        self.tokenizer.add_tokens(self.argument_markers, special_tokens=True)

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
            start_marker = construct_argument_marker(pos=START, label=labeled_span.label)
            positions_and_markers.append((labeled_span.start, start_marker))
            end_marker = construct_argument_marker(pos=END, label=labeled_span.label)
            positions_and_markers.append((labeled_span.end, end_marker))

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
        new_document = TextDocumentWithLabeledSpansAndBinaryRelations(
            id=document.id,
            metadata=deepcopy(document.metadata),
            text=marked_text,
        )
        new_document.labeled_spans.extend(old2new_spans.values())
        new_document.binary_relations.extend(old2new_relations.values())

        new2old_spans = {new_span: old_span for old_span, new_span in old2new_spans.items()}
        return new_document, new2old_spans

    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        all_relations: Sequence[Annotation] = self.get_relation_layer(document)
        self.collect_all_relations("available", all_relations)

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
        tokenized_docs = tokenize_document(
            document_with_markers,
            tokenizer=self.tokenizer,
            result_document_type=self.tokenized_document_type,
            partition_layer="labeled_partitions"
            if self.partition_annotation is not None
            else None,
            strict_span_conversion=False,
            # TODO: does this work as expected? e.g. when using return_overflowing_tokens?
            return_tensors="pt",
            **self.tokenize_kwargs,
        )

        task_encodings: List[TaskEncodingType] = []
        for tokenized_doc in tokenized_docs:
            # collect start- and end-token positions for each entity
            start_marker_positions = []
            end_marker_positions = []
            for labeled_span in tokenized_doc.labeled_spans:
                # the start marker is one token before the start of the span
                start_marker_positions.append(labeled_span.start - 1)
                # the end marker is one token after the end of the span, but the end index is exclusive
                end_marker_positions.append(labeled_span.end)

            labeled_span2idx = {span: idx for idx, span in enumerate(tokenized_doc.labeled_spans)}
            arg_indices = []  # list of lists of argument indices: [[head_idx, tail_idx], ...]
            if self.create_candidate_relations:
                candidate_relations = self._create_candidate_relations(
                    tokenized_doc, **self.create_candidate_relations_kwargs
                )
            else:
                candidate_relations = tokenized_doc.binary_relations
            for relation in candidate_relations:
                current_args_indices = []
                for _, arg_span in get_relation_argument_spans_and_roles(relation):
                    arg_idx = labeled_span2idx[arg_span]
                    current_args_indices.append(arg_idx)
                arg_indices.append(current_args_indices)

            # TODO: can we do this? i.e. converting to a dict and adding the new keys?
            inputs = dict(tokenized_doc.metadata["tokenizer_encoding"])
            inputs["start_marker_positions"] = torch.tensor(start_marker_positions)
            inputs["end_marker_positions"] = torch.tensor(end_marker_positions)
            inputs["arg_indices"] = torch.tensor(arg_indices).to(torch.long)

            task_encodings.append(
                TaskEncoding(
                    document=document,
                    inputs=inputs,
                    metadata={
                        "tokenized_document": tokenized_doc,
                        "injected2original_spans": injected2original_spans,
                        "candidate_relations": candidate_relations,
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
            gold_roles_and_args2relation[get_relation_argument_spans_and_roles(relation)].append(
                relation
            )
        label_indices = []  # list of label indices
        valid_candidate_relations = []
        for candidate_relation in task_encoding.metadata["candidate_relations"]:
            candidate_roles_and_args = get_relation_argument_spans_and_roles(candidate_relation)
            gold_relations = gold_roles_and_args2relation.get(candidate_roles_and_args, [])
            if len(gold_relations) == 0:
                label_indices.append(self.label_to_id[candidate_relation.label])
            elif len(gold_relations) == 1:
                label_indices.append(self.label_to_id[gold_relations[0].label])
            else:
                logger.warning(
                    f"skip the candidate relation because there are more than one gold relation "
                    f"for its args and roles: {gold_relations}"
                )
                for gold_relation in gold_relations:
                    self.collect_relation("skipped_same_arguments", gold_relation)
                continue
            valid_candidate_relations.append(candidate_relation)

        task_encoding.metadata["candidate_relations"] = valid_candidate_relations
        target: TargetEncodingType = {
            "labels": torch.tensor(label_indices).to(torch.long),
        }

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
            logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
            logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
            # target data
            start_marker_positions = task_encoding.inputs["start_marker_positions"]
            end_marker_positions = task_encoding.inputs["end_marker_positions"]
            labels = [self.id_to_label[label] for label in target["labels"]]
            for i, (label, arg_indices) in enumerate(
                zip(labels, task_encoding.inputs["arg_indices"])
            ):
                logger.info(f"relation {i}: {label}")
                for j, arg_idx in enumerate(arg_indices):
                    arg_tokens = tokens[
                        start_marker_positions[arg_idx] : end_marker_positions[arg_idx]
                    ]
                    logger.info(f"\targ {i}: {' '.join([str(x) for x in arg_tokens])}")

            self._logged_examples_counter += 1

    def pad_or_stack(self, key: str, values: List[LongTensor]) -> LongTensor:
        if key in PAD_VALUES:
            return pad_sequence(values, batch_first=True, padding_value=PAD_VALUES[key]).to(
                torch.long
            )
        else:
            return torch.stack(values, dim=0).to(torch.long)

    def collate(
        self, task_encodings: Sequence[TaskEncodingType]
    ) -> Tuple[ModelInputType, Optional[ModelTargetType]]:
        input_keys = task_encodings[0].inputs.keys()
        inputs: ModelInputType = {  # type: ignore
            key: self.pad_or_stack(
                key, [task_encoding.inputs[key] for task_encoding in task_encodings]
            )
            for key in input_keys
        }

        targets: Optional[ModelTargetType] = None
        if task_encodings[0].has_targets:
            target_keys = task_encodings[0].targets.keys()
            targets: ModelTargetType = {  # type: ignore
                key: self.pad_or_stack(
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
        tokenized_document: TokenDocumentWithLabeledSpansAndBinaryRelations,
    ) -> Dict[str, List[Annotation]]:
        new_relations = []
        for candidate_relation, label, probability in zip(
            tokenized_document.metadata["candidate_relations"],
            task_output["labels"],
            task_output["probabilities"],
        ):
            new_annotation = candidate_relation.copy(label=label, score=probability)
            new_relations.append(new_annotation)

        return {"binary_relations": new_relations}

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncodingType,
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Union[BinaryRelation, MultiLabeledBinaryRelation, NaryRelation]]]:
        tokenized_document = task_encoding.metadata["tokenized_document"]

        decoded_annotations = self.decode_annotations(
            task_output=task_output, tokenized_document=tokenized_document
        )

        # Note: token_based_document_to_text_based() does not yet consider predictions, so we need to clear
        # the main annotations and attach the predictions to that
        for layer_name, annotations in decoded_annotations.items():
            tokenized_document[layer_name].clear()
            for annotation in annotations:
                tokenized_document[layer_name].append(annotation)

        untokenized_document: TextDocumentWithLabeledSpansAndBinaryRelations = (
            token_based_document_to_text_based(
                tokenized_document, result_document_type=self.normalized_document_type
            )
        )

        injected2original_spans = task_encoding.metadata["injected2original_spans"]
        for relation in untokenized_document.binary_relations:
            # map back from spans over the marker-injected text to the original spans
            if isinstance(relation, BinaryRelation):
                original_head = injected2original_spans[relation.head]
                original_tail = injected2original_spans[relation.tail]
                new_relation = BinaryRelation(
                    head=original_head, tail=original_tail, label=relation.label
                )
            else:
                raise NotImplementedError(
                    f"the taskmodule does not yet support relations of type {type(relation)}"
                )
            yield self.relation_annotation, new_relation

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
            # do we want to ignore the no_relation_label?
            # "ignore_index": self.label_to_id[self.no_relation_label],
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
            prepare_function=_get_labels_from_model_output,
        )
