import dataclasses
import json
import logging
from collections import Counter, defaultdict
from functools import cmp_to_key
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch
from pytorch_ie import AnnotationLayer, Document
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import Annotation, TaskEncoding, TaskModule
from pytorch_ie.core.taskmodule import (
    InputEncoding,
    ModelBatchOutput,
    TargetEncoding,
    TaskBatchEncoding,
)
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument
from torchmetrics import Metric
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing_extensions import TypeAlias

# import for backwards compatibility (don't remove!)
from pie_modules.documents import (
    TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)

from ..document.processing import token_based_document_to_text_based, tokenize_document
from ..utils import resolve_type
from .common import BatchableMixin, HasBuildMetric, HasDecodeAnnotations
from .pointer_network.annotation_encoder_decoder import (
    BinaryRelationEncoderDecoder,
    LabeledSpanEncoderDecoder,
    SpanEncoderDecoderWithOffset,
)
from .pointer_network.metrics import AnnotationLayerMetric

logger = logging.getLogger(__name__)


DocumentType: TypeAlias = TextBasedDocument


@dataclasses.dataclass
class InputEncodingType(BatchableMixin):
    src_tokens: List[int]
    src_attention_mask: List[int]

    @property
    def src_seq_len(self) -> int:
        return len(self.src_tokens)


@dataclasses.dataclass
class EncodingWithIdsAndOptionalCpmTag(BatchableMixin):
    tgt_tokens: List[int]
    CPM_tag: Optional[List[List[int]]] = None

    @property
    def tgt_seq_len(self) -> int:
        return len(self.tgt_tokens)

    @property
    def tgt_attention_mask(self) -> List[int]:
        return [1] * len(self.tgt_tokens)


TargetEncodingType: TypeAlias = EncodingWithIdsAndOptionalCpmTag
TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]
TaskOutputType: TypeAlias = EncodingWithIdsAndOptionalCpmTag


def _span_is_in_partition(span: Span, partition: Optional[Span] = None):
    if partition is None:
        return True
    return (
        partition.start <= span.start < partition.end
        and partition.start < span.end <= partition.end
    )


def cmp_src_rel(v1: BinaryRelation, v2: BinaryRelation) -> int:
    if not all(isinstance(ann, LabeledSpan) for ann in [v1.head, v1.tail, v2.head, v2.tail]):
        raise Exception(f"expected LabeledSpan, but got: {v1}, {v2}")
    if v1.head.start == v2.head.start:  # v1[0]["from"] == v2[0]["from"]:
        return v1.tail.start - v2.tail.start  # v1[1]["from"] - v2[1]["from"]
    return v1.head.start - v2.head.start  # v1[0]["from"] - v2[0]["from"]


# TODO: use enable BucketSampler (just mentioning here because no better place available for now)
# see https://github.com/Lightning-AI/lightning/pull/13640#issuecomment-1199032224


@TaskModule.register()
class PointerNetworkTaskModuleForEnd2EndRE(
    HasBuildMetric,
    HasDecodeAnnotations[TaskOutputType],
    TaskModule[
        DocumentType,
        InputEncoding,
        TargetEncoding,
        TaskBatchEncoding,
        ModelBatchOutput,
        TaskOutputType,
    ],
):
    PREPARED_ATTRIBUTES = ["labels_per_layer"]

    def __init__(
        self,
        # specific for this use case
        span_layer_name: str = "labeled_spans",
        relation_layer_name: str = "binary_relations",
        none_label: str = "none",
        loop_dummy_relation_name: str = "loop",
        # generic pointer network
        label_tokens: Optional[Dict[str, str]] = None,
        label_representations: Optional[Dict[str, str]] = None,
        labels_per_layer: Optional[Dict[str, List[str]]] = None,
        exclude_labels_per_layer: Optional[Dict[str, List[str]]] = None,
        # target encoding
        create_constraints: bool = False,
        # tokenization
        document_type: str = "pytorch_ie.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
        tokenized_document_type: str = "pie_modules.documents.TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
        tokenizer_name_or_path: str = "facebook/bart-base",
        tokenizer_init_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        partition_layer_name: Optional[str] = None,
        annotation_field_mapping: Optional[Dict[str, str]] = None,
        # logging
        log_first_n_examples: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # tokenization
        self._document_type: Type[TextBasedDocument] = resolve_type(
            document_type, expected_super_type=TextBasedDocument
        )
        self._tokenized_document_type: Type[TokenBasedDocument] = resolve_type(
            tokenized_document_type, expected_super_type=TokenBasedDocument
        )
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            **(tokenizer_init_kwargs or {}),
        )
        self.annotation_field_mapping = annotation_field_mapping or dict()
        self.partition_layer_name = partition_layer_name

        # for this specific use case: end-to-end relation extraction
        self.span_layer_name = span_layer_name
        self.relation_layer_name = relation_layer_name
        self.none_label = none_label
        self.loop_dummy_relation_name = loop_dummy_relation_name
        # will be set in _post_prepare()
        self.relation_encoder_decoder: BinaryRelationEncoderDecoder

        # collected in prepare(), if not passed in
        self.labels_per_layer = labels_per_layer
        self.exclude_labels_per_layer = exclude_labels_per_layer or {}

        # how to encode and decode the annotations
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.label_tokens = label_tokens or dict()
        self.label_representations = label_representations or dict()

        # target encoding
        self.create_constraints = create_constraints
        self.pad_values = {
            "tgt_tokens": self.target_pad_id,
            "tgt_attention_mask": 0,
            "src_tokens": self.tokenizer.pad_token_id,
            "src_attention_mask": 0,
            "CPM_tag": -1,
        }
        self.dtypes = {
            "tgt_tokens": torch.int64,
            "tgt_attention_mask": torch.int64,
            "src_seq_len": torch.int64,
            "src_tokens": torch.int64,
            "src_attention_mask": torch.int64,
            "tgt_seq_len": torch.int64,
            "CPM_tag": torch.int64,
        }

        # logging
        self.log_first_n_examples = log_first_n_examples

    @property
    def document_type(self) -> Type[TextBasedDocument]:
        return self._document_type

    @property
    def tokenized_document_type(self) -> Type[TokenBasedDocument]:
        return self._tokenized_document_type

    @property
    def layer_names(self) -> List[str]:
        return [self.span_layer_name, self.relation_layer_name]

    @property
    def special_targets(self) -> list[str]:
        return [self.bos_token, self.eos_token]

    @property
    def special_target2id(self) -> Dict[str, int]:
        return {target: idx for idx, target in enumerate(self.special_targets)}

    @property
    def target_pad_id(self) -> int:
        return self.special_target2id[self.eos_token]

    @property
    def generation_kwargs(self) -> Dict[str, Any]:
        return {
            "no_repeat_ngram_size": 7,
            # TODO: add this when it looks really solid (currently strange behavior)
            # "prefix_allowed_tokens_fn": self._prefix_allowed_tokens_fn,
        }

    def _prepare(self, documents: Sequence[DocumentType]) -> None:
        # collect all labels
        labels: Dict[str, Set[str]] = {layer_name: set() for layer_name in self.layer_names}
        for doc in documents:
            for layer_name in self.layer_names:
                exclude_labels = self.exclude_labels_per_layer.get(layer_name, [])
                labels[layer_name].update(
                    ac.label for ac in doc[layer_name] if ac.label not in exclude_labels
                )

        self.labels_per_layer = {
            # sort labels to ensure deterministic order
            layer_name: sorted(labels)
            for layer_name, labels in labels.items()
        }

    def construct_label_token(self, label: str) -> str:
        return self.label_tokens.get(label, f"<<{label}>>")

    def get_label_representation(self, label: str) -> str:
        return self.label_representations.get(label, label)

    def _post_prepare(self) -> None:
        # set up labels
        if self.labels_per_layer is None:
            raise Exception("labels_per_layer is not defined. Call prepare() first or pass it in.")
        self.labels: List[str] = [self.none_label]
        for layer_name in self.layer_names:
            self.labels.extend(self.labels_per_layer[layer_name])
        if len(set(self.labels)) != len(self.labels):
            raise Exception(f"labels are not unique: {self.labels}")

        # set up targets and ids
        self.targets: List[str] = self.special_targets + self.labels
        self.target2id: Dict[str, int] = {target: idx for idx, target in enumerate(self.targets)}

        # generic ids
        self.eos_id: int = self.target2id[self.eos_token]
        self.bos_id: int = self.target2id[self.bos_token]

        # span and relation ids
        self.span_ids: List[int] = [
            self.target2id[label] for label in self.labels_per_layer[self.span_layer_name]
        ]
        self.relation_ids: List[int] = [
            self.target2id[label] for label in self.labels_per_layer[self.relation_layer_name]
        ]
        # the none id is used for the dummy relation which models out-of-relation spans
        self.none_id: int = self.target2id[self.none_label]

        # helpers (same as targets / target2id, but only for labels)
        self.label2id: Dict[str, int] = {label: self.target2id[label] for label in self.labels}
        self.id2label: Dict[int, str] = {idx: label for label, idx in self.label2id.items()}
        self.label_ids: List[int] = [self.label2id[label] for label in self.labels]

        # annotation-encoder-decoders
        span_encoder_decoder = SpanEncoderDecoderWithOffset(
            offset=self.pointer_offset, exclusive_end=False
        )
        labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
            span_encoder_decoder=span_encoder_decoder,
            label2id=self.label2id,
            mode="indices_label",
        )
        self.relation_encoder_decoder = BinaryRelationEncoderDecoder(
            head_encoder_decoder=labeled_span_encoder_decoder,
            tail_encoder_decoder=labeled_span_encoder_decoder,
            label2id=self.label2id,
            loop_dummy_relation_name=self.loop_dummy_relation_name,
            none_label=self.none_label,
            mode="tail_head_label",
        )

        label2token = {label: self.construct_label_token(label=label) for label in self.labels}
        if len(set(label2token.values())) != len(label2token):
            raise Exception(f"label2token values are not unique: {label2token}")

        already_in_vocab = [
            tok
            for tok in label2token.values()
            if self.tokenizer.convert_tokens_to_ids(tok) != self.tokenizer.unk_token_id
        ]
        if len(already_in_vocab) > 0:
            raise Exception(
                f"some special tokens to add (mapped label ids) are already in the tokenizer vocabulary, "
                f"this is not allowed: {already_in_vocab}. You may want to adjust the label2special_token mapping"
            )
        # sort by length, so that longer tokens are added first
        label_tokens_sorted = sorted(label2token.values(), key=lambda x: len(x), reverse=True)
        self.tokenizer.add_special_tokens(
            special_tokens_dict={"additional_special_tokens": label_tokens_sorted}
        )

        # target tokens are the special tokens plus the mapped label tokens
        self.target_tokens: List[str] = self.special_targets + [
            label2token[label] for label in self.labels
        ]
        self.target_token_ids: List[int] = self.tokenizer.convert_tokens_to_ids(self.target_tokens)

        # construct a mapping from label_token_id to token_ids that will be used to initialize the embeddings
        # of the labels
        self.label_embedding_weight_mapping = dict()
        for label, label_token in label2token.items():
            label_token_indices = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(label_token)
            )
            # sanity check: label_tokens should not be split up
            if len(label_token_indices) > 1:
                raise RuntimeError(f"{label_token} wrong split")
            else:
                label_token_idx = label_token_indices[0]

            label_representation = self.get_label_representation(label)
            source_indices = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(label_representation)
            )
            if self.tokenizer.unk_token_id in source_indices:
                raise RuntimeError(
                    f"tokenized label_token={label_token} [{source_indices}] contains unk_token"
                )
            self.label_embedding_weight_mapping[label_token_idx] = source_indices

    @property
    def pointer_offset(self) -> int:
        return len(self.targets)

    @property
    def target_ids(self) -> Set[int]:
        return set(range(self.pointer_offset))

    def build_metric(self, stage: Optional[str] = None) -> Metric:
        return AnnotationLayerMetric(
            eos_id=self.eos_id,
            taskmodule=self,
            layer_names=self.layer_names,
            decode_annotations_func=self.decode_annotations,
        )

    def validate_relation_encoding(
        self, encoding: List[int]
    ) -> Tuple[Optional[Tuple[int, int, int, int, int, int, int]], Optional[str]]:
        valid_encoding: Optional[Tuple[int, int, int, int, int, int, int]] = None
        skip = False
        error_type = None

        # if we got a relation id too early, we skip the tuple
        if len(encoding) != 7:
            skip = True
            error_type = "len"

        # we got a span-only tuple ...
        elif self.none_id in encoding:
            # TODO: assert that pair[2] in self.span_ids
            #  and pair[4..6] is self.none_id
            if not (
                encoding[2] in self.target_ids
                and encoding[5] in self.target_ids
                and encoding[6] in self.target_ids
            ):
                # TODO: set error_type = "cross"? or return another error
                skip = True
            else:
                skip = False
            # TODO: check start end indices (as for "order" below)

        # we got a relation tuple ...
        else:
            # check the offsets
            # TODO: check that at positions 0, 1, 3, and 4, we have offsets
            # if any end index is smaller than the start index, skip the tuple
            if encoding[0] > encoding[1] or encoding[3] > encoding[4]:
                skip = True
                error_type = "order"
            # if the spans overlap, skip the tuple
            elif not (encoding[1] < encoding[3] or encoding[0] > encoding[4]):
                skip = True
                error_type = "cover"

            # check the labels

            # if the label ids got mixed up (span vs. relation ids), skip the tuple
            if (
                encoding[2] in self.relation_ids
                or encoding[5] in self.relation_ids
                or encoding[6] in self.span_ids
            ):
                # Consider making an additional layer of restrictions to prevent misalignment
                # of the relationship and span tags (可以考虑做多一层限制，防止relation 和 span标签错位)
                skip = True
                error_type = "cross"
            # TODO: move above previous check and make the other a elif case to not count cross errors twice
            # if at the label positions we have no label ids, skip the tuple
            span_and_relation_ids = self.relation_ids + self.span_ids
            if (
                encoding[2] not in span_and_relation_ids
                or encoding[5] not in span_and_relation_ids
                or encoding[6] not in span_and_relation_ids
            ):
                # if not tag.issubset(self.relation_idx+self.span_idx):
                skip = True
                error_type = "cross"

        if not skip:
            # ignore type because of tuple length (already checked above)
            valid_encoding = tuple(encoding)  # type: ignore
        return valid_encoding, error_type

    def get_valid_relation_encodings(
        self,
        tag_seq: List[int],
    ) -> Tuple[List[Tuple[int, int, int, int, int, int, int]], Dict[str, int], List[int]]:
        errors = {
            "len": 0,
            "order": 0,
            "cross": 0,
            "cover": 0,
            # , "correct": 0 ,
        }

        encodings = []
        current_encoding: List[int] = []
        if len(tag_seq):
            for i in tag_seq:
                # a tuple will be terminated when the next id is a relation id
                # or when this is a none_id and the tuple has 6 entries (i.e. it will be completed with the next id)
                if i in self.relation_ids or (i == self.none_id and len(current_encoding) == 6):
                    current_encoding.append(i)

                    # check if the current_encoding is valid
                    valid_encoding, error_type = self.validate_relation_encoding(current_encoding)
                    if error_type is not None:
                        # TODO: count total amounts instead of returning bool values.
                        #  This requires to also count "correct".
                        errors[error_type] = 1

                    if valid_encoding is not None:
                        encodings.append(valid_encoding)
                    current_encoding = []
                else:
                    current_encoding.append(i)

        return encodings, errors, current_encoding

    def encode_annotations(
        self, layers: Dict[str, List[Annotation]], metadata: Optional[Dict[str, Any]] = None
    ) -> TaskOutputType:
        if not set(layers.keys()) == set(self.layer_names):
            raise Exception(f"unexpected layers: {layers.keys()}. expected: {self.layer_names}")

        if self.labels_per_layer is None:
            raise Exception("labels_per_layer is not defined. Call prepare() first or pass it in.")

        # encode relations
        all_relation_arguments = set()
        relation_encodings = dict()
        for rel in layers[self.relation_layer_name]:
            if not isinstance(rel, BinaryRelation):
                raise Exception(f"expected BinaryRelation, but got: {rel}")
            if rel.label in self.labels_per_layer[self.relation_layer_name]:
                encoded_relation = self.relation_encoder_decoder.encode(
                    annotation=rel, metadata=metadata
                )
                if encoded_relation is None:
                    raise Exception(f"failed to encode relation: {rel}")
                relation_encodings[rel] = encoded_relation
                all_relation_arguments.update([rel.head, rel.tail])

        # encode spans that are not arguments of any relation
        no_relation_spans = [
            span for span in layers[self.span_layer_name] if span not in all_relation_arguments
        ]
        for span in no_relation_spans:
            dummy_relation = BinaryRelation(
                head=span, tail=span, label=self.loop_dummy_relation_name
            )
            encoded_relation = self.relation_encoder_decoder.encode(
                annotation=dummy_relation, metadata=metadata
            )
            if encoded_relation is not None:
                relation_encodings[dummy_relation] = encoded_relation

        # sort relations by start indices of head and tail
        sorted_relations = sorted(relation_encodings, key=cmp_to_key(cmp_src_rel))

        # build target tokens
        tgt_tokens = [self.bos_id]
        for rel in sorted_relations:
            encoded_relation = relation_encodings[rel]
            tgt_tokens.extend(encoded_relation)
        tgt_tokens.append(self.eos_id)

        # sanity check
        _, encoding_errors, remaining = self.get_valid_relation_encodings(tag_seq=tgt_tokens[1:])
        if (
            not all(v == 0 for k, v in encoding_errors.items() if k != "correct")
            or len(remaining) > 0
        ):
            decoded, invalid = self.decode_annotations(
                EncodingWithIdsAndOptionalCpmTag(tgt_tokens), metadata=metadata
            )
            not_encoded = {}
            for layer_name in layers:
                # convert to dicts to make them comparable (original annotations are attached which breaks comparison)
                decoded_dicts = [ann.asdict() for ann in decoded[layer_name]]
                # filter annotations and convert to str to make them json serializable
                filtered = {
                    str(ann) for ann in layers[layer_name] if ann.asdict() not in decoded_dicts
                }
                if len(filtered) > 0:
                    not_encoded[layer_name] = list(filtered)
            if len(not_encoded) > 0:
                logger.warning(
                    f"encoding errors: {encoding_errors}, skipped annotations:\n"
                    f"{json.dumps(not_encoded, sort_keys=True, indent=2)}"
                )
            elif len([tag for tag in remaining if tag != self.eos_id]) > 0:
                logger.warning(
                    f"encoding errors: {encoding_errors}, remaining encoding ids: {remaining}"
                )

        # build CPM tag
        if self.create_constraints:
            if metadata is None or "src_len" not in metadata:
                raise Exception("metadata with 'src_len' is required to create constraints")
            src_len = metadata["src_len"]
            constraints = self.build_constraints(src_len=src_len, tgt_tokens=tgt_tokens)
            cpm_tag = constraints
        else:
            cpm_tag = None
        return EncodingWithIdsAndOptionalCpmTag(tgt_tokens=tgt_tokens, CPM_tag=cpm_tag)

    def decode_annotations(
        self, encoding: TaskOutputType, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, List[Annotation]], Any]:
        # strip the bos token
        relation_encodings, errors, remaining = self.get_valid_relation_encodings(
            tag_seq=encoding.tgt_tokens[1:]
        )
        relation_tuples: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
        entity_labels: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for relation_encoding in relation_encodings:
            rel = self.relation_encoder_decoder.decode(
                encoding=list(relation_encoding), metadata=metadata
            )
            head_span = (rel.head.start, rel.head.end)
            entity_labels[head_span].append(rel.head.label)

            if rel.label != self.loop_dummy_relation_name:
                tail_span = (rel.tail.start, rel.tail.end)
                entity_labels[tail_span].append(rel.tail.label)
                relation_tuples.append((head_span, tail_span, rel.label))
            else:
                assert rel.head == rel.tail

        # TODO: do we really need this majority vote? Why not keep both? Or parametrize this?
        entities: Dict[Tuple[int, int], LabeledSpan] = {}
        for (start, end), labels in entity_labels.items():
            c = Counter(labels)
            # if len(c) > 1:
            #    logger.warning(f"multiple labels for span, take the most common: {dict(c)}")
            most_common_label = c.most_common(1)[0][0]
            entities[(start, end)] = LabeledSpan(start=start, end=end, label=most_common_label)

        entity_layer = list(entities.values())
        relation_layer = [
            BinaryRelation(head=entities[head_span], tail=entities[tail_span], label=label)
            for head_span, tail_span, label in relation_tuples
        ]
        return {
            self.span_layer_name: entity_layer,
            self.relation_layer_name: relation_layer,
        }, errors

    # TODO: use torch instead of numpy?
    def _pointer_tag(
        self,
        last: List[int],
        t: int,
        idx: int,
        arr: np.ndarray,
    ) -> np.ndarray:
        if t == 0:  # start # c1 [0, 1]
            arr[: self.pointer_offset] = 0
        elif idx % 7 == 0:  # c1 [0,1, 23]
            arr[:t] = 0
        elif idx % 7 == 1:  # tc1 [0,1,23, tc] span标签设为1
            arr = np.zeros_like(arr, dtype=int)
            for i in self.span_ids:
                arr[i] = 1
        elif idx % 7 == 2:  # c2 [0,1,23,tc, 45]
            arr[: self.pointer_offset] = 0
            arr[last[-3] : last[-2]] = 0
        elif idx % 7 == 3:  # c2 [0,1,23,tc,45, 67]
            arr[:t] = 0
            if t < last[-4]:
                arr[last[-4] :] = 0
            else:
                arr[last[-4] : last[-3]] = 0
        elif idx % 7 == 4:  # tc2 [0,1,23,tc,45,67, tc]
            arr = np.zeros_like(arr, dtype=int)
            for i in self.span_ids:
                arr[i] = 1
        elif idx % 7 == 5:  # r [0,1,23,tc,45,67,tc, r]
            arr = np.zeros_like(arr, dtype=int)
            for i in self.relation_ids:
                arr[i] = 1
        elif idx % 7 == 6:  # next
            arr[: self.pointer_offset] = 0
        return arr

    def build_constraints(
        self,
        src_len: int,
        tgt_tokens: List[int],
    ) -> List[List[int]]:
        # strip the bos token
        target = tgt_tokens[1:]
        # pad for 0
        likely_hood = np.ones(src_len + self.pointer_offset, dtype=int)
        likely_hood[: self.pointer_offset] = 0
        CMP_tag: List[np.ndarray] = [likely_hood]
        for idx, t in enumerate(target[:-1]):
            last7 = target[idx - 7 if idx - 7 > 0 else 0 : idx + 1]
            likely_hood = np.ones(src_len + self.pointer_offset, dtype=int)
            tag = self._pointer_tag(last=last7, t=t, idx=idx, arr=likely_hood)
            tag[self.none_id] = 1
            CMP_tag.append(tag)
        last_end = np.zeros(src_len + self.pointer_offset, dtype=int)
        last_end[self.none_id] = 1
        last_end[target[-1]] = 1
        CMP_tag[-1] = last_end
        result = [i.tolist() for i in CMP_tag]
        return result

    def maybe_log_example(
        self,
        task_encoding: TaskEncodingType,
        targets: Optional[TargetEncodingType] = None,
    ):
        if self.log_first_n_examples is not None and self.log_first_n_examples > 0:
            tokenized_doc_id = task_encoding.metadata["tokenized_document"].id
            inputs = task_encoding.inputs
            targets = targets or task_encoding.targets
            src_token_ids = inputs.src_tokens
            src_tokens = self.tokenizer.convert_ids_to_tokens(src_token_ids)
            tgt_token_ids = targets.tgt_tokens
            tgt_tokens = [
                self.targets[tgt_token_id]
                if tgt_token_id < self.pointer_offset
                else str(tgt_token_id)
                + " {"
                + str(src_tokens[tgt_token_id - self.pointer_offset])
                + "}"
                for tgt_token_id in tgt_token_ids
            ]
            logger.info("*** Example ***")
            logger.info(f"doc.id:        {tokenized_doc_id}")
            logger.info(f"src_token_ids: {' '.join([str(i) for i in src_token_ids])}")
            logger.info(f"src_tokens:    {' '.join(src_tokens)}")
            logger.info(f"tgt_token_ids: {' '.join([str(i) for i in tgt_token_ids])}")
            logger.info(f"tgt_tokens:    {' '.join(tgt_tokens)}")
            self.log_first_n_examples -= 1

    def tokenize_document(self, document: DocumentType) -> List[TokenBasedDocument]:
        field_mapping = dict(self.annotation_field_mapping)
        if self.partition_layer_name is not None:
            field_mapping[self.partition_layer_name] = "labeled_partitions"
            partition_layer = "labeled_partitions"
        else:
            partition_layer = None
        casted_document = document.as_type(self.document_type, field_mapping=field_mapping)
        tokenized_docs = tokenize_document(
            casted_document,
            tokenizer=self.tokenizer,
            result_document_type=self.tokenized_document_type,
            partition_layer=partition_layer,
            **self.tokenizer_kwargs,
        )
        for idx, tokenized_doc in enumerate(tokenized_docs):
            tokenized_doc.id = f"{document.id}-tokenized-{idx+1}-of-{len(tokenized_docs)}"

        return tokenized_docs

    def encode_input(
        self, document: DocumentType, is_training: bool = False
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        tokenized_docs = self.tokenize_document(document)
        task_encodings: List[TaskEncodingType] = []
        for tokenized_doc in tokenized_docs:
            tokenizer_encoding = tokenized_doc.metadata["tokenizer_encoding"]
            task_encodings.append(
                TaskEncoding(
                    document=document,
                    inputs=InputEncodingType(
                        src_tokens=tokenizer_encoding.ids,
                        src_attention_mask=tokenizer_encoding.attention_mask,
                    ),
                    metadata={"tokenized_document": tokenized_doc},
                )
            )

        return task_encodings

    def get_mapped_layer(self, document: Document, layer_name: str) -> AnnotationLayer:
        if layer_name in self.annotation_field_mapping:
            layer_name = self.annotation_field_mapping[layer_name]
        return document[layer_name]

    def encode_target(self, task_encoding: TaskEncodingType) -> Optional[TargetEncodingType]:
        document = task_encoding.metadata["tokenized_document"]

        layers = {
            layer_name: self.get_mapped_layer(document, layer_name=layer_name)
            for layer_name in self.layer_names
        }
        result = self.encode_annotations(
            layers=layers,
            metadata={**task_encoding.metadata, "src_len": task_encoding.inputs.src_seq_len},
        )

        self.maybe_log_example(task_encoding=task_encoding, targets=result)
        return result

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> TaskBatchEncoding:
        if len(task_encodings) == 0:
            raise ValueError("no task_encodings available")
        inputs = InputEncodingType.batch(
            values=[x.inputs for x in task_encodings],
            dtypes=self.dtypes,
            pad_values=self.pad_values,
        )

        targets = None
        if task_encodings[0].has_targets:
            targets = TargetEncodingType.batch(
                values=[x.targets for x in task_encodings],
                dtypes=self.dtypes,
                pad_values=self.pad_values,
            )

        return inputs, targets

    def unbatch_output(self, model_output: ModelBatchOutput) -> Sequence[TaskOutputType]:
        # model_output just contains "pred"
        pred = model_output["pred"]
        batch_size = pred.size(0)
        result = [
            EncodingWithIdsAndOptionalCpmTag(pred[i].to(device="cpu").tolist())
            for i in range(batch_size)
        ]
        return result

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncodingType,
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Annotation]]:
        layers, errors = self.decode_annotations(
            encoding=task_output, metadata=task_encoding.metadata
        )
        tokenized_document = task_encoding.metadata["tokenized_document"]

        # Note: token_based_document_to_text_based() does not yet consider predictions, so we need to clear
        # the main annotations and attach the predictions to that
        for layer_name, annotations in layers.items():
            layer = self.get_mapped_layer(tokenized_document, layer_name=layer_name)
            layer.clear()
            layer.extend(annotations)

        untokenized_document = token_based_document_to_text_based(
            tokenized_document, result_document_type=self.document_type
        )

        for layer_name in layers:
            annotations = self.get_mapped_layer(untokenized_document, layer_name=layer_name)
            for annotation in annotations:
                yield layer_name, annotation.copy()
