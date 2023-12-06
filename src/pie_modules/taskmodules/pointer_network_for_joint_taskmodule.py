import dataclasses
import logging
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import (
    Annotation,
    AnnotationList,
    TaskEncoding,
    TaskModule,
    annotation_field,
)
from pytorch_ie.core.taskmodule import (
    InputEncoding,
    ModelBatchOutput,
    TargetEncoding,
    TaskBatchEncoding,
)
from pytorch_ie.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    TokenBasedDocument,
)
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing_extensions import TypeAlias

from ..document.processing import tokenize_document
from .components.seq2seq import (
    PointerNetworkSpanAndRelationEncoderDecoder,
    PointerNetworkSpanEncoderDecoder,
)

logger = logging.getLogger(__name__)


DocumentType: TypeAlias = TextBasedDocument


@dataclasses.dataclass
class InputEncodingType:
    src_tokens: List[int]
    src_seq_len: int


@dataclasses.dataclass
class TargetEncodingType:
    tgt_tokens: List[int]
    tgt_seq_len: int
    CPM_tag: Optional[List[int]] = None


TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]
TaskOutput: TypeAlias = torch.Tensor


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansAndBinaryRelations(TokenBasedDocument):
    labeled_spans: AnnotationList[LabeledSpan] = annotation_field(target="tokens")
    binary_relations: AnnotationList[BinaryRelation] = annotation_field(target="labeled_spans")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
    TokenDocumentWithLabeledSpansAndBinaryRelations
):
    labeled_partitions: AnnotationList[Span] = annotation_field(target="labeled_spans")


def _pad_tensor(tensor: torch.Tensor, target_shape: List[int], pad_value: float) -> torch.Tensor:
    shape = tensor.shape
    pad: List[int] = []
    for i, s in enumerate(shape):
        pad = [0, target_shape[i] - s] + pad
    result = F.pad(tensor, pad=pad, value=pad_value)
    assert list(result.shape) == target_shape
    return result


def ld2dl(
    list_of_dicts: Union[List[Dict[str, Any]], Sequence[Dict[str, Any]]],
    keys: Optional[str] = None,
    getter: Optional[Callable] = None,
) -> Dict[str, List[Any]]:
    if getter is None:

        def getter(x):
            return x

    keys = keys or getter(list_of_dicts[0]).keys()
    v = {k: [getter(dic)[k] for dic in list_of_dicts] for k in keys}
    return v


def _span_is_in_partition(span: Span, partition: Optional[Span] = None):
    if partition is None:
        return True
    return (
        partition.start <= span.start < partition.end
        and partition.start < span.end <= partition.end
    )


# TODO: use enable BucketSampler (just mentioning here because no better place available for now)
# see https://github.com/Lightning-AI/lightning/pull/13640#issuecomment-1199032224


@TaskModule.register()
class PointerNetworkForJointTaskModule(
    TaskModule[
        DocumentType,
        InputEncoding,
        TargetEncoding,
        TaskBatchEncoding,
        ModelBatchOutput,
        TaskOutput,
    ]
):
    PREPARED_ATTRIBUTES = ["span_labels", "relation_labels"]

    def __init__(
        self,
        tokenizer_name_or_path: str = "facebook/bart-base",
        span_labels: Optional[List[str]] = None,
        relation_labels: Optional[List[str]] = None,
        none_label: str = "none",
        # so that the label word can be initialized in a better embedding.
        label2special_token: Optional[Dict[str, str]] = None,
        # dummy relation type to encode entities that do not belong to any relation
        loop_dummy_relation_name: str = "loop",
        exclude_annotation_names: Optional[Dict[str, List[str]]] = None,
        span_end_mode: str = "last_token",
        tokenize_per_word: bool = False,
        text_field_name: str = "text",
        span_layer_name: str = "labeled_spans",
        relation_layer_name: str = "binary_relations",
        word_layer_name: Optional[str] = None,
        partition_layer_name: Optional[str] = None,
        log_first_n_examples: Optional[int] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_init_kwargs: Optional[Dict[str, Any]] = None,
        max_target_length: Optional[int] = None,
        create_constraints: bool = True,
        annotation_encoder_decoder_name: str = "gmam",
        annotation_encoder_decoder_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.label2special_token = label2special_token or {}
        self.span_labels = span_labels
        self.relation_labels = relation_labels
        self.none_label = none_label

        self.loop_dummy_relation_name = loop_dummy_relation_name
        self.exclude_annotation_names = exclude_annotation_names or dict()
        self.create_constraints = create_constraints

        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            **(tokenizer_init_kwargs or {}),
        )

        self.tokenizer_kwargs = tokenizer_kwargs or {}

        # how to encode the end of the span
        self.annotation_encoder_decoder_name = annotation_encoder_decoder_name
        self.annotation_encoder_decoder_kwargs = annotation_encoder_decoder_kwargs or {}

        self.span_end_mode = span_end_mode

        self.text_field_name = text_field_name
        self.span_layer_name = span_layer_name
        self.relation_layer_name = relation_layer_name
        self.partition_layer_name = partition_layer_name
        self.word_layer_name = word_layer_name

        self.tokenize_per_word = tokenize_per_word

        self.max_target_length = max_target_length

        # see fastNLP.core.batch.DataSetGetter
        # self.input_names = {"src_tokens", "src_seq_len", "tgt_tokens", "tgt_seq_len", "CPM_tag"}
        # self.target_names = {"src_seq_len", "tgt_tokens", "tgt_seq_len", "target_span"}

        self.pad_values = {
            # "tgt_tokens": 1,  # this will be set in _post_prepare()
            "src_tokens": self.tokenizer.pad_token_id,
            "CPM_tag": -1,
        }
        self.dtypes = {
            "tgt_tokens": torch.int64,
            "src_seq_len": torch.int64,
            "src_tokens": torch.int64,
            "tgt_seq_len": torch.int64,
            "CPM_tag": torch.int64,
        }

        self.log_first_n_examples = log_first_n_examples

    @property
    def document_type(self):
        if self.partition_layer_name is None:
            return TextDocumentWithLabeledSpansAndBinaryRelations
        else:
            return TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions

    @property
    def tokenized_document_type(self):
        if self.partition_layer_name is None:
            return TokenDocumentWithLabeledSpansAndBinaryRelations
        else:
            return TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions

    def _prepare(self, documents: Sequence[DocumentType]):
        span_labels: Set[str] = set()
        relation_labels: Set[str] = set()
        for doc in documents:
            span_labels.update(
                ac.label
                for ac in doc[self.span_layer_name]
                if ac.label not in self.exclude_annotation_names.get(self.span_layer_name, [])
            )
            relation_labels.update(
                rel.label
                for rel in doc[self.relation_layer_name]
                if rel.label not in self.exclude_annotation_names.get(self.relation_layer_name, [])
            )
        self.span_labels = sorted(span_labels)
        self.relation_labels = sorted(relation_labels)

    def _post_prepare(self):
        # we need the following:
        # 1. labels: entity and relation labels and the none label
        # 2. label tokens: labels encapsulated with "<<" and ">>"
        # 3. target tokens: "<bos>", "<eos>", and (2)
        # 4. target ids (3)
        # 5. token ids of (3)
        # + various mappings

        self.labels = self.span_labels + self.relation_labels + [self.none_label]
        self.label2token = {
            label: self.label2special_token.get(label, f"<<{label}>>") for label in self.labels
        }
        self.token2label = {v: k for k, v in self.label2token.items()}
        if len(self.label2token) != len(self.token2label):
            raise Exception(
                f"all entries in label2token need to map to different entries, which is not the case: "
                f"{self.label2token}"
            )
        self.label_tokens = sorted(self.label2token.values(), key=lambda x: len(x), reverse=True)
        already_in_vocab = [
            tok
            for tok in self.label_tokens
            if self.tokenizer.convert_tokens_to_ids(tok) != self.tokenizer.unk_token_id
        ]
        if len(already_in_vocab) > 0:
            raise Exception(
                f"some special tokens to add (mapped label ids) are already in the tokenizer vocabulary, "
                f"this is not allowed: {already_in_vocab}. You may want to adjust the label2special_token mapping"
            )
        # self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_label_ids
        self.tokenizer.add_special_tokens(
            special_tokens_dict={"additional_special_tokens": self.label_tokens}
        )

        self.label_token_ids = self.tokenizer.convert_tokens_to_ids(self.label_tokens)
        # this returns all the token ids that can occur in the output
        self.target_token_ids = [
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
        ] + sorted(self.label_token_ids)

        self.label2id: Dict[str, int] = {}
        self.target_token2id: Dict[str, int] = {}
        for idx, target_token_id in enumerate(self.target_token_ids):
            target_token = self.tokenizer.convert_ids_to_tokens(target_token_id)
            self.target_token2id[target_token] = idx
            if target_token in self.label_tokens:
                self.label2id[self.token2label[target_token]] = idx
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.bos_id = self.target_token2id[self.tokenizer.bos_token]
        self.eos_id = self.target_token2id[self.tokenizer.eos_token]
        self.span_ids = [self.label2id[i] for i in self.span_labels]
        self.relation_ids = [self.label2id[i] for i in self.relation_labels]
        self.none_ids = self.label2id[self.none_label]
        # Set to the id where eos is located (设置为eos所在的id)
        self.pad_id = self.eos_id

        # TODO: make that configurable and do not depend on << >> syntax
        self.embedding_weight_mapping = dict()
        for label_token in self.label_tokens:
            # sanity check: label_tokens should not be split up
            special_token_index = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(label_token)
            )
            if len(special_token_index) > 1:
                raise RuntimeError(f"{label_token} wrong split")
            else:
                special_token_index = special_token_index[0]

            assert label_token[:2] == "<<" and label_token[-2:] == ">>"
            source_indices = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(label_token[2:-2])
            )
            assert self.tokenizer.unk_token_id not in source_indices
            self.embedding_weight_mapping[special_token_index] = source_indices

        if self.annotation_encoder_decoder_name == "gmam":
            span_encoder_decoder = PointerNetworkSpanEncoderDecoder(
                span_end_mode=self.span_end_mode,
                pointer_offset=self.pointer_offset,
            )
            self.annotation_encoder_decoder = PointerNetworkSpanAndRelationEncoderDecoder(
                span_encoder_decoder=span_encoder_decoder,
                id2label=self.id2label,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                span_ids=self.span_ids,
                relation_ids=self.relation_ids,
                none_id=self.none_ids,
                **self.annotation_encoder_decoder_kwargs,
            )
        else:
            raise Exception(
                f"unknown annotation_encoder_decoder_name: {self.annotation_encoder_decoder_name}"
            )

    @property
    def target_tokens(self) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(self.target_token_ids)

    @property
    def target_ids(self) -> List[int]:
        return list(self.target_token2id.values())

    @property
    def label_ids(self):
        return sorted(self.label2id.values())

    @property
    def pointer_offset(self) -> int:
        return len(self.target_token_ids)

    @property
    def pad_id(self) -> int:
        v = self.pad_values.get("tgt_tokens", None)
        if v is None:
            raise Exception("pad value for tgt_tokens is not set")
        return v

    @pad_id.setter
    def pad_id(self, value):
        self.pad_values["tgt_tokens"] = value

    def maybe_log_example(
        self,
        task_encoding: TaskEncodingType,
        targets: Optional[TargetEncodingType] = None,
    ):
        if self.log_first_n_examples is not None and self.log_first_n_examples > 0:
            inputs = task_encoding.inputs
            targets = targets or task_encoding.targets
            src_token_ids = inputs.src_tokens
            src_tokens = self.tokenizer.convert_ids_to_tokens(src_token_ids)
            tgt_token_ids = targets.tgt_tokens
            tgt_tokens = [
                self.tokenizer.convert_ids_to_tokens(self.target_token_ids[tgt_token_id])
                if tgt_token_id < self.pointer_offset
                else str(tgt_token_id)
                + " {"
                + str(src_tokens[tgt_token_id - self.pointer_offset])
                + "}"
                for tgt_token_id in tgt_token_ids
            ]
            logger.info("*** Example ***")
            # logger.info(f"doc id: {task_encoding.document.id}")
            logger.info(f"src_token_ids: {' '.join([str(i) for i in src_token_ids])}")
            logger.info(f"src_tokens:    {' '.join(src_tokens)}")
            logger.info(f"tgt_token_ids: {' '.join([str(i) for i in tgt_token_ids])}")
            logger.info(f"tgt_tokens:    {' '.join(tgt_tokens)}")
            self.log_first_n_examples -= 1

    def encode_input(
        self, document: DocumentType, is_training: bool = False
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        task_encodings = []
        text = getattr(document, self.text_field_name)

        if self.word_layer_name is not None:
            char2word = {}
            word2char = []
            for word_idx, word in enumerate(document[self.word_layer_name]):
                word2char.append((word.start, word.end))
                for char_idx in range(word.start, word.end):
                    char2word[char_idx] = word_idx
        else:
            char2word = None
            word2char = None

        if self.partition_layer_name is not None:
            partitions = document[self.partition_layer_name]
        else:
            partitions = [None]
        for partition in partitions:
            tokenizer_encodings = []
            if self.tokenize_per_word:
                # TODO: implement truncation? striding?
                if len(self.tokenizer_kwargs) > 0:
                    raise NotImplementedError(
                        "tokenizer_kwargs not supported for tokenize_per_word"
                    )
                if self.word_layer_name not in document:
                    raise Exception(
                        f'the annotation layer "{self.word_layer_name}" that should contain word annotations '
                        f"is required if tokenize_per_word is enabled"
                    )
                token2char = []
                src_tokens = [self.tokenizer.bos_token_id]
                special_tokens_mask = [1]
                token2char.append((0, 0))
                for word in document[self.word_layer_name]:
                    if _span_is_in_partition(span=word, partition=partition):
                        word_str = text[word.start : word.end]
                        tokenizer_output = self.tokenizer(
                            word_str, return_offsets_mapping=True, add_special_tokens=False
                        )
                        token_ids = tokenizer_output.input_ids
                        for start, end in tokenizer_output.offset_mapping:
                            token2char.append((start + word.start, end + word.start))
                        src_tokens.extend(token_ids)
                        special_tokens_mask.extend([0] * len(token_ids))
                src_tokens.append(self.tokenizer.eos_token_id)
                special_tokens_mask.append(1)
                tokenizer_encodings.append((src_tokens, token2char, special_tokens_mask))
            else:
                text_partition = (
                    text[partition.start : partition.end] if partition is not None else text
                )
                tokenizer_output = self.tokenizer(
                    text_partition,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                    return_overflowing_tokens=True,
                    **self.tokenizer_kwargs,
                )
                for encoding in tokenizer_output.encodings:
                    src_tokens = encoding.ids
                    if partition is not None:
                        token2char = [
                            (start + partition.start, end + partition.start)
                            if not is_special_token
                            else (start, end)
                            for (start, end), is_special_token in zip(
                                encoding.offsets, encoding.special_tokens_mask
                            )
                        ]
                    else:
                        token2char = encoding.offsets
                    special_tokens_mask = encoding.special_tokens_mask
                    tokenizer_encodings.append((src_tokens, token2char, special_tokens_mask))

            for src_tokens, token2char, special_tokens_mask in tokenizer_encodings:
                src_seq_len = len(src_tokens)
                inputs = InputEncodingType(src_tokens=src_tokens, src_seq_len=src_seq_len)

                char2token = defaultdict(list)
                for token_idx, (char_start, char_end) in enumerate(token2char):
                    for char_idx in range(char_start, char_end):
                        char2token[char_idx].append(token_idx)

                no_special_token2char = [
                    start_end
                    for start_end, is_special_token in zip(token2char, special_tokens_mask)
                    if not is_special_token
                ]
                tokenized_span = Span(
                    start=no_special_token2char[0][0], end=no_special_token2char[-1][1]
                )
                metadata = {
                    "token2char": token2char,
                    "char2token": dict(char2token),
                    "tokenized_span": tokenized_span,
                }
                if partition is not None:
                    metadata["partition"] = partition

                if char2word is not None and word2char is not None:
                    metadata["char2word"] = char2word
                    metadata["word2char"] = word2char

                    word2token = []
                    for char_start, char_end in word2char:
                        token_start = char2token[char_start][0]
                        token_end = char2token[char_end - 1][-1] + 1
                        word2token.append((token_start, token_end))

                    token2word = {}
                    for word_idx, (token_start, token_end) in enumerate(word2token):
                        for token_idx in range(token_start, token_end):
                            token2word[token_idx] = word_idx
                    metadata["word2token"] = word2token
                    metadata["token2word"] = token2word

                task_encoding = TaskEncoding(
                    document=document,
                    inputs=inputs,
                    metadata=metadata,
                )
                task_encodings.append(task_encoding)

        return task_encodings

    def _is_valid_annotation(
        self,
        annotation: Annotation,
        partition: Optional[Span] = None,
        tokenized_span: Optional[Span] = None,
    ) -> bool:
        if isinstance(annotation, BinaryRelation):
            excluded_rel_names = set(
                self.exclude_annotation_names.get(self.relation_layer_name, [])
            )
            return (
                annotation.label not in excluded_rel_names
                and self._is_valid_annotation(
                    annotation.head, partition=partition, tokenized_span=tokenized_span
                )
                and self._is_valid_annotation(
                    annotation.tail, partition=partition, tokenized_span=tokenized_span
                )
            )
        elif isinstance(annotation, LabeledSpan):
            excluded_names = set(self.exclude_annotation_names.get(self.span_layer_name, []))
            return (
                _span_is_in_partition(span=annotation, partition=partition)
                and _span_is_in_partition(span=annotation, partition=tokenized_span)
                and annotation.label not in excluded_names
            )
        elif isinstance(annotation, Span):
            return _span_is_in_partition(
                span=annotation, partition=partition
            ) and _span_is_in_partition(span=annotation, partition=tokenized_span)
        else:
            raise Exception(f"annotation has unknown type: {annotation}")

    def encode_target(self, task_encoding: TaskEncodingType) -> Optional[TargetEncodingType]:
        document = task_encoding.document

        partition = task_encoding.metadata.get("partition", None)
        tokenized_span = task_encoding.metadata["tokenized_span"]
        valid_relations = [
            rel
            for rel in document[self.relation_layer_name]
            if self._is_valid_annotation(rel, partition=partition, tokenized_span=tokenized_span)
        ]
        valid_spans = [
            span
            for span in document[self.span_layer_name]
            if self._is_valid_annotation(span, partition=partition, tokenized_span=tokenized_span)
        ]

        tgt_tokens = self.annotation_encoder_decoder.encode(
            layers={"span": valid_spans, "relation": valid_relations},
            metadata=task_encoding.metadata,
        )

        if self.max_target_length is not None and len(tgt_tokens) > self.max_target_length:
            raise ValueError(
                f"target length {len(tgt_tokens)} exceeds max_target_length {self.max_target_length}"
            )

        constraints = None
        if self.create_constraints:
            constraints = self.annotation_encoder_decoder.build_constraints(
                src_len=task_encoding.inputs.src_seq_len,
                tgt_tokens=tgt_tokens,
            )
        result = TargetEncodingType(
            tgt_tokens=tgt_tokens, tgt_seq_len=len(tgt_tokens), CPM_tag=constraints
        )
        self.maybe_log_example(task_encoding=task_encoding, targets=result)
        return result

    def _pad_values(self, values: List[List], name: str, strategy: str = "longest"):
        if name not in self.pad_values:
            return values
        if not isinstance(values, list):
            return values
        if strategy != "longest":
            raise ValueError(f"unknown padding strategy: {strategy}")
        pad_value = self.pad_values[name]
        tensor_list = [torch.tensor(value_list) for value_list in values]
        shape_lists = list(zip(*[t.shape for t in tensor_list]))
        max_shape = [max(dims) for dims in shape_lists]
        padded = [
            _pad_tensor(tensor=t, target_shape=max_shape, pad_value=pad_value)
            for i, t in enumerate(tensor_list)
        ]
        return torch.stack(padded)

    def _to_tensor(
        self, values: Union[List, torch.Tensor], name: str
    ) -> Union[torch.Tensor, List]:
        if name not in self.dtypes:
            return values
        if not isinstance(values, torch.Tensor):
            tensor = torch.Tensor(values)
        else:
            tensor = values
        tensor = tensor.to(dtype=self.dtypes[name])
        return tensor

    def _prepare_values(self, values: List, name: str) -> Union[torch.Tensor, List]:
        maybe_padded = self._pad_values(values=values, name=name)
        maybe_tensor = self._to_tensor(values=maybe_padded, name=name)
        return maybe_tensor

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> TaskBatchEncoding:
        if len(task_encodings) == 0:
            raise ValueError("no task_encodings available")
        inputs = {
            k: self._prepare_values(values=v, name=k)
            for k, v in ld2dl(
                task_encodings, getter=lambda x: dataclasses.asdict(x.inputs)
            ).items()
        }

        targets = None
        if task_encodings[0].has_targets:
            targets = {
                k: self._prepare_values(values=v, name=k)
                for k, v in ld2dl(
                    task_encodings, getter=lambda x: dataclasses.asdict(x.targets)
                ).items()
            }

        return inputs, targets

    def unbatch_output(self, model_output: ModelBatchOutput) -> Sequence[TaskOutput]:
        # model_output just contains "pred"
        pred = model_output["pred"]
        batch_size = pred.size(0)
        result = [pred[i].to(device="cpu") for i in range(batch_size)]
        return result

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncodingType,
        task_output: TaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        layers, _errors = self.annotation_encoder_decoder.decode(
            targets=task_output.tolist(), metadata=task_encoding.metadata
        )

        for span in layers["span"]:
            yield self.span_layer_name, span

        for rel in layers["relation"]:
            yield self.relation_layer_name, rel
