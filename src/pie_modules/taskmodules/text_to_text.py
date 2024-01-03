import dataclasses
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import torch
from pytorch_ie import AnnotationLayer, Document
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

from pie_modules.annotations import AnnotationWithText
from pie_modules.document.processing import (
    token_based_document_to_text_based,
    tokenize_document,
)
from pie_modules.utils import resolve_type

from .common import BatchableMixin
from .common.utils import get_first_occurrence_index

logger = logging.getLogger(__name__)


DocumentType: TypeAlias = TextBasedDocument


@dataclasses.dataclass
class InputEncodingType(BatchableMixin):
    input_ids: List[int]
    attention_mask: List[int]


@dataclasses.dataclass
class TargetEncodingType(BatchableMixin):
    labels: List[int]
    # TODO: verify that decoder_attention_mask makes sense for AutoModelForSeq2SeqLM instances
    decoder_attention_mask: Optional[List[int]] = None

    # @property
    # def decoder_attention_mask(self) -> List[int]:
    #    return [1] * len(self.labels)


TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]
TaskOutputType: TypeAlias = TargetEncodingType


@TaskModule.register()
class TextToTextTaskModule(
    TaskModule[
        DocumentType,
        InputEncoding,
        TargetEncoding,
        TaskBatchEncoding,
        ModelBatchOutput,
        TaskOutputType,
    ],
):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        # input document
        document_type: str,  # e.g. "pie_modules.documents.TextDocumentWithAbstractiveSummary"
        tokenized_document_type: str,  # e.g. "pie_modules.documents.TokenDocumentWithQuestionsAndGenerativeAnswers",
        target_layer: str,  # e.g. "abstractive_summary"
        target_annotation_type: str,  # e.g. "pie_modules.annotations.AbstractiveSummary"
        # TODO: rename to "guidance_*" (also tests)?
        source_layer: Optional[str] = None,  # e.g. "questions" for question answering
        source_annotation_field: Optional[str] = None,  # e.g. "question" for question answering
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

        self.target_layer = target_layer
        self.source_layer = source_layer
        self.target_annotation_type: Type[AnnotationWithText] = resolve_type(
            target_annotation_type, expected_super_type=AnnotationWithText
        )
        self.source_annotation_field = source_annotation_field

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

        # target encoding
        self.pad_values = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
            "labels": self.tokenizer.pad_token_id,
            "decoder_attention_mask": 0,
        }
        self.dtypes = {
            "input_ids": torch.int64,
            "attention_mask": torch.int64,
            "labels": torch.int64,
            "decoder_attention_mask": torch.int64,
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
        return [self.target_layer]

    def get_mapped_layer(self, document: Document, layer_name: str) -> AnnotationLayer:
        if layer_name in self.annotation_field_mapping:
            layer_name = self.annotation_field_mapping[layer_name]
        return document[layer_name]

    @property
    def generation_config(self) -> Dict[str, Any]:
        return {}

    def maybe_log_example(
        self,
        task_encoding: TaskEncodingType,
        targets: Optional[TargetEncodingType] = None,
    ):
        if self.log_first_n_examples is not None and self.log_first_n_examples > 0:
            inputs = task_encoding.inputs

            logger.info(f"input_ids: {inputs.input_ids}")
            logger.info(f"attention_mask: {inputs.attention_mask}")
            if targets is not None or task_encoding.has_targets:
                targets = targets or task_encoding.targets
                logger.info(f"labels: {targets.labels}")
            self.log_first_n_examples -= 1

    def encode_annotations(
        self,
        layers: Dict[str, AnnotationLayer],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TargetEncodingType:
        target_annotations = []
        source_annotation = (
            metadata.get("source_annotation", None) if metadata is not None else None
        )
        if source_annotation is not None:
            if self.source_annotation_field is None:
                raise ValueError(
                    "source_annotation is available, but source_annotation_field is not set"
                )
            # filter annotations that belong to the source_annotation
            for target_annotation in layers[self.target_layer]:
                current_source_annotation = getattr(
                    target_annotation, self.source_annotation_field
                )
                if current_source_annotation == source_annotation:
                    target_annotations.append(target_annotation)
        else:
            target_annotations = layers[self.target_layer]

        if len(target_annotations) != 1:
            raise ValueError(
                f"target_annotations {self.target_layer} contains {len(target_annotations)} annotations, "
                f"but expected exactly one"
            )
        annotation = target_annotations[0]
        if isinstance(annotation, self.target_annotation_type):
            text = target_annotations[0].text
        else:
            raise ValueError(
                f"target_annotations {self.target_layer} contains an annotation of type {type(annotation)}, "
                f"but expected {self.target_annotation_type}"
            )
        encoding = self.tokenizer(text)
        return TargetEncodingType(
            labels=encoding["input_ids"], decoder_attention_mask=encoding["attention_mask"]
        )

    def decode_annotations(
        self, encoding: TaskOutputType, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, List[Annotation]], Any]:
        text = self.tokenizer.decode(encoding.labels, skip_special_tokens=True)
        annotation_kwargs = {}
        if self.source_annotation_field is not None:
            if metadata is None:
                raise ValueError(
                    "metadata is required to decode annotations with source_annotation_field"
                )
            source_annotation = metadata.get("source_annotation", None)
            if source_annotation is not None:
                if self.source_annotation_field is None:
                    raise ValueError(
                        "source_annotation is available, but source_annotation_field is not set"
                    )
                annotation_kwargs[self.source_annotation_field] = source_annotation

        decoded_layers = {
            self.target_layer: [self.target_annotation_type(text=text, **annotation_kwargs)]
        }
        # no error collection yet
        errors: Dict[str, Any] = {}
        return decoded_layers, errors

    def tokenize_document(
        self, document: DocumentType, source_text: Optional[str] = None
    ) -> List[TokenBasedDocument]:
        field_mapping = dict(self.annotation_field_mapping)
        if self.partition_layer_name is not None:
            field_mapping[self.partition_layer_name] = "labeled_partitions"
            partition_layer = "labeled_partitions"
        else:
            partition_layer = None
        casted_document = document.as_type(self.document_type, field_mapping=field_mapping)

        tokenizer_kwargs = dict(self.tokenizer_kwargs)
        if source_text is not None:
            tokenizer_kwargs["text"] = source_text
        tokenized_docs = tokenize_document(
            casted_document,
            tokenizer=self.tokenizer,
            result_document_type=self.tokenized_document_type,
            partition_layer=partition_layer,
            **tokenizer_kwargs,
        )
        for idx, tokenized_doc in enumerate(tokenized_docs):
            tokenized_doc.id = f"{document.id}-tokenized-{idx+1}-of-{len(tokenized_docs)}"

        return tokenized_docs

    def encode_input(
        self, document: DocumentType, is_training: bool = False
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        task_encodings: List[TaskEncodingType] = []
        if self.source_layer is None:
            source_annotations = [None]
        else:
            source_annotations = document[self.source_layer]
        for source_annotation in source_annotations:
            source_text = None
            if source_annotation is not None:
                # Here could also more sophisticated logic be implemented
                source_text = source_annotation.text
            tokenized_docs = self.tokenize_document(document, source_text=source_text)
            for tokenized_doc in tokenized_docs:
                tokenizer_encoding = tokenized_doc.metadata["tokenizer_encoding"]
                task_encodings.append(
                    TaskEncoding(
                        document=document,
                        inputs=InputEncodingType(
                            input_ids=tokenizer_encoding.ids,
                            attention_mask=tokenizer_encoding.attention_mask,
                        ),
                        metadata={
                            "tokenized_document": tokenized_doc,
                            "source_annotation": source_annotation,
                        },
                    )
                )

        return task_encodings

    def encode_target(self, task_encoding: TaskEncodingType) -> Optional[TargetEncodingType]:
        document = task_encoding.metadata["tokenized_document"]
        source_annotation = task_encoding.metadata["source_annotation"]

        layers = {
            layer_name: self.get_mapped_layer(document, layer_name=layer_name)
            for layer_name in self.layer_names
        }
        result = self.encode_annotations(
            layers=layers,
            metadata={**task_encoding.metadata, "source_annotation": source_annotation},
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
        batch_size = model_output.size(0)

        # We use the position after the first eos token as the seq_len.
        # Note that, if eos_id is not in model_output for a given batch item, the result will be
        # model_output.size(1) + 1 (i.e. seq_len + 1) for that batch item. This is fine, because we use the
        # seq_lengths just to truncate the output and want to keep everything if eos_id is not present.
        seq_lengths = get_first_occurrence_index(model_output, self.tokenizer.eos_token_id) + 1

        result = [
            TaskOutputType(model_output[i, : seq_lengths[i]].to(device="cpu").tolist())
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

        # TODO: implement configure_model_metric(self, stage: str) -> Optional[Metric]
