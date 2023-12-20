import dataclasses
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import torch
from pytorch_ie import AnnotationLayer, Document
from pytorch_ie.annotations import Span
from pytorch_ie.core import Annotation, TaskEncoding, TaskModule
from pytorch_ie.core.taskmodule import (
    InputEncoding,
    ModelBatchOutput,
    TargetEncoding,
    TaskBatchEncoding,
)
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing_extensions import TypeAlias

from ..document.processing import token_based_document_to_text_based, tokenize_document
from ..utils import BatchableMixin, resolve_type
from .components.seq2seq import PointerNetworkSpanAndRelationEncoderDecoder

logger = logging.getLogger(__name__)


DocumentType: TypeAlias = TextBasedDocument


@dataclasses.dataclass
class InputEncodingType(BatchableMixin):
    src_tokens: List[int]
    src_attention_mask: List[int]

    @property
    def src_seq_len(self):
        return len(self.src_tokens)


@dataclasses.dataclass
class TargetEncodingType(BatchableMixin):
    tgt_tokens: List[int]
    tgt_attention_mask: List[int]
    CPM_tag: Optional[List[List[int]]] = None

    @property
    def tgt_seq_len(self):
        return len(self.tgt_tokens)


TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]
TaskOutput: TypeAlias = torch.Tensor


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
class PointerNetworkTaskModule(
    TaskModule[
        DocumentType,
        InputEncoding,
        TargetEncoding,
        TaskBatchEncoding,
        ModelBatchOutput,
        TaskOutput,
    ]
):
    # Note that we overwrite self.is_prepared with self.annotation_encoder_decoder.is_prepared, so it is fine
    # to initialize the taskmodule *with* a value for annotation_encoder_decoder_kwargs, but _prepare(documents)
    # gets still called when the taskmodule is used in a pipeline.
    PREPARED_ATTRIBUTES = ["annotation_encoder_decoder_kwargs"]

    def __init__(
        self,
        # tokenization
        document_type: str = "pytorch_ie.documents.TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
        tokenized_document_type: str = "pie_modules.documents.TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions",
        tokenizer_name_or_path: str = "facebook/bart-base",
        tokenizer_init_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        partition_layer_name: Optional[str] = None,
        annotation_field_mapping: Optional[Dict[str, str]] = None,
        # how to encode and decode the annotations
        annotation_encoder_decoder_name: str = "pointer_network_span_and_relation",
        annotation_encoder_decoder_kwargs: Optional[Dict[str, Any]] = None,
        label_tokens: Optional[Dict[str, str]] = None,
        label_representations: Optional[Dict[str, str]] = None,
        # target encoding
        max_target_length: Optional[int] = None,
        create_constraints: bool = False,
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

        # how to encode and decode the annotations
        self.annotation_encoder_decoder_name = annotation_encoder_decoder_name
        self.annotation_encoder_decoder_kwargs = annotation_encoder_decoder_kwargs or {}
        if self.annotation_encoder_decoder_name == "pointer_network_span_and_relation":
            # do not pass bos_token and eos_token directly as constructor arguments,
            # because they may be already in annotation_encoder_decoder_kwargs
            self.annotation_encoder_decoder_kwargs["bos_token"] = self.tokenizer.bos_token
            self.annotation_encoder_decoder_kwargs["eos_token"] = self.tokenizer.eos_token
            self.annotation_encoder_decoder = PointerNetworkSpanAndRelationEncoderDecoder(
                **self.annotation_encoder_decoder_kwargs
            )
        else:
            raise Exception(
                f"unknown annotation_encoder_decoder_name: {self.annotation_encoder_decoder_name}"
            )
        self.label_tokens = label_tokens or dict()
        self.label_representations = label_representations or dict()

        # target encoding
        self.max_target_length = max_target_length
        self.create_constraints = create_constraints
        self.pad_values = {
            "tgt_tokens": self.annotation_encoder_decoder.target_pad_id,
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
    def is_prepared(self):
        return self.annotation_encoder_decoder.is_prepared

    def _prepare(self, documents: Sequence[DocumentType]):
        self.annotation_encoder_decoder._prepare(documents=documents)

    def construct_label_token(self, label: str) -> str:
        return self.label_tokens.get(label, f"<<{label}>>")

    def get_label_representation(self, label: str) -> str:
        return self.label_representations.get(label, label)

    def _post_prepare(self):
        self.annotation_encoder_decoder._post_prepare()
        # This is a bit hacky, but we need to update the kwargs of the annotation-encoder-decoder with its
        # prepared attributes, because they are required when instantiating it from config, i.e. when
        # taskmodule._prepare(documents) is *not* called.
        self.annotation_encoder_decoder_kwargs.update(
            self.annotation_encoder_decoder.prepared_attributes
        )

        label2token = {
            label: self.construct_label_token(label=label)
            for label in self.annotation_encoder_decoder.labels
        }
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
        self.target_tokens: List[str] = self.annotation_encoder_decoder.special_targets + [
            label2token[label] for label in self.annotation_encoder_decoder.labels
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
                self.annotation_encoder_decoder.targets[tgt_token_id]
                if tgt_token_id < self.annotation_encoder_decoder.pointer_offset
                else str(tgt_token_id)
                + " {"
                + str(src_tokens[tgt_token_id - self.annotation_encoder_decoder.pointer_offset])
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
            for layer_name in self.annotation_encoder_decoder.layer_names
        }
        tgt_tokens = self.annotation_encoder_decoder.encode(
            layers=layers, metadata=task_encoding.metadata
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
            tgt_tokens=tgt_tokens,
            tgt_attention_mask=[1] * len(tgt_tokens),
            CPM_tag=constraints,
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
