"""
workflow:
    Document
        -> (InputEncoding, TargetEncoding) -> TaskEncoding -> TaskBatchEncoding
            -> ModelBatchEncoding -> ModelBatchOutput
        -> TaskOutput
    -> Document
"""

import copy
import dataclasses
import logging
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

import torch
import torch.nn.functional as F
from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import LabeledSpan, Span
from pytorch_ie.core import TaskEncoding, TaskModule
from pytorch_ie.documents import (
    TextDocument,
    TextDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansAndLabeledPartitions,
    TokenBasedDocument,
)
from pytorch_ie.models.transformer_token_classification import (
    ModelOutputType,
    ModelStepInputType,
)
from pytorch_ie.utils.span import bio_tags_to_spans
from tokenizers import Encoding
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from typing_extensions import TypeAlias

from pie_modules.document.processing import (
    token_based_document_to_text_based,
    tokenize_document,
)

InputEncodingType: TypeAlias = Encoding
TargetEncodingType: TypeAlias = Sequence[int]
DocumentType: TypeAlias = TextDocument

TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]
TaskOutputType: TypeAlias = Dict[str, Any]

TaskModuleType: TypeAlias = TaskModule[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
    ModelStepInputType,
    ModelOutputType,
    TaskOutputType,
]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TokenDocumentWithLabeledSpans(TokenBasedDocument):
    labeled_spans: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansAndLabeledPartitions(TokenDocumentWithLabeledSpans):
    labeled_partitions: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")


@TaskModule.register()
class TokenClassificationTaskModule(TaskModuleType):
    # list of attribute names that need to be set by _prepare()
    PREPARED_ATTRIBUTES: List[str] = ["labels"]

    def __init__(
        self,
        tokenizer_name_or_path: str,
        span_annotation: str = "labeled_spans",
        partition_annotation: Optional[str] = None,
        label_pad_token_id: int = -100,
        labels: Optional[List[str]] = None,
        max_window: Optional[int] = None,
        window_overlap: int = 0,
        show_statistics: bool = False,
        include_ill_formed_predictions: bool = True,
        tokenize_kwargs: Optional[Dict[str, Any]] = None,
        pad_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # backwards compatibility
        tokenize_kwargs = copy.deepcopy(tokenize_kwargs) or {}
        if max_window is not None:
            tokenize_kwargs["max_length"] = max_window
            tokenize_kwargs["return_overflowing_tokens"] = True
            logger.warning(
                "The 'max_window' parameter is deprecated and will be removed in a future version. "
                "Please use the 'tokenize_kwargs[\"max_length\"]' parameter instead."
            )
        if window_overlap > 0:
            tokenize_kwargs["stride"] = window_overlap
            tokenize_kwargs["return_overflowing_tokens"] = True
            logger.warning(
                "The 'window_overlap' parameter is deprecated and will be removed in a future version. "
                "Please use the 'tokenize_kwargs[\"stride\"]' parameter instead."
            )

        self.save_hyperparameters(ignore=["max_window", "window_overlap"])

        self.span_annotation = span_annotation
        self.partition_annotation = partition_annotation
        self.labels = labels
        self.label_pad_token_id = label_pad_token_id
        self.show_statistics = show_statistics
        self.include_ill_formed_predictions = include_ill_formed_predictions
        self.tokenize_kwargs = tokenize_kwargs or {}
        self.pad_kwargs = pad_kwargs or {}

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    @property
    def document_type(self) -> Optional[Type[TextDocument]]:
        dt: Type[TextDocument]
        errors = []
        if self.span_annotation != "labeled_spans":
            errors.append(
                f"span_annotation={self.span_annotation} is not the default value ('labeled_spans')"
            )
        if self.partition_annotation is None:
            dt = TextDocumentWithLabeledSpans
        else:
            if self.partition_annotation != "labeled_partitions":
                errors.append(
                    f"partition_annotation={self.partition_annotation} is not the default value "
                    f"('labeled_partitions')"
                )
            dt = TextDocumentWithLabeledSpansAndLabeledPartitions

        if len(errors) == 0:
            return dt
        else:
            logger.warning(
                f"{' and '.join(errors)}, so the taskmodule {type(self).__name__} can not request "
                f"the usual document type ({dt.__name__}) for auto-conversion because this has the bespoken default "
                f"value as layer name(s) instead of the provided one(s)."
            )
            return None

    def get_span_layer(self, document: DocumentType) -> AnnotationLayer[LabeledSpan]:
        return document[self.span_annotation]

    def _prepare(self, documents: Sequence[DocumentType]) -> None:
        # collect all possible labels
        labels: Set[str] = set()
        for document in documents:
            spans: AnnotationLayer[LabeledSpan] = self.get_span_layer(document)

            for span in spans:
                labels.add(span.label)

        self.labels = sorted(labels)

    def _post_prepare(self):
        # creat the real token labels (BIO scheme) from the labels
        self.label_to_id = {"O": 0}
        current_id = 1
        for label in sorted(self.labels):
            for prefix in ["B", "I"]:
                self.label_to_id[f"{prefix}-{label}"] = current_id
                current_id += 1

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def encode_text(
        self, text, partition: Optional[Span] = None, add_special_tokens: bool = True
    ) -> BatchEncoding:
        if self.partition_annotation is not None and partition is None:
            raise ValueError("partitioning is enabled, but no partition is provided")

        text_partition = text[partition.start : partition.end] if partition is not None else text
        return self.tokenizer(
            text_partition,
            padding=False,
            truncation=False,
            max_length=None,
            is_split_into_words=False,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            add_special_tokens=add_special_tokens,
        )

    def encode_input(
        self,
        document: TextDocument,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        if self.partition_annotation is None:
            tokenized_document_type = TokenDocumentWithLabeledSpans
            casted_document_type = TextDocumentWithLabeledSpans
            field_mapping = {self.span_annotation: "labeled_spans"}
        else:
            tokenized_document_type = TokenDocumentWithLabeledSpansAndLabeledPartitions
            casted_document_type = TextDocumentWithLabeledSpansAndLabeledPartitions
            field_mapping = {
                self.span_annotation: "labeled_spans",
                self.partition_annotation: "labeled_partitions",
            }
        casted_document = document.as_type(casted_document_type, field_mapping=field_mapping)
        tokenized_docs = tokenize_document(
            casted_document,
            tokenizer=self.tokenizer,
            result_document_type=tokenized_document_type,
            partition_layer="labeled_partitions"
            if self.partition_annotation is not None
            else None,
            strict_span_conversion=False,
            **self.tokenize_kwargs,
        )

        task_encodings: List[TaskEncodingType] = []
        for tokenized_doc in tokenized_docs:
            task_encodings.append(
                TaskEncoding(
                    document=document,
                    inputs=tokenized_doc.metadata["tokenizer_encoding"],
                    metadata={"tokenized_document": tokenized_doc},
                )
            )

        return task_encodings

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> Optional[TargetEncodingType]:
        metadata = task_encoding.metadata
        tokenized_document = metadata["tokenized_document"]
        tokenizer_encoding: Encoding = tokenized_document.metadata["tokenizer_encoding"]

        tag_sequence = [
            None if tokenizer_encoding.special_tokens_mask[j] else "O"
            for j in range(len(tokenizer_encoding.ids))
        ]
        if self.labels is None:
            raise ValueError(
                "'labels' must be set before calling encode_target(). Was prepare() called on the taskmodule?"
            )
        for span in tokenized_document.labeled_spans:
            if span.label not in self.labels:
                continue
            start = span.start
            end = span.end
            if any(tag != "O" for tag in tag_sequence[start:end]):
                logger.warning(f"tag already assigned (current span has an overlap: {span}).")
                return None

            tag_sequence[start] = f"B-{span.label}"
            for j in range(start + 1, end):
                tag_sequence[j] = f"I-{span.label}"

        targets = [
            self.label_to_id[tag] if tag is not None else self.label_pad_token_id
            for tag in tag_sequence
        ]

        return targets

    def collate(self, task_encodings: Sequence[TaskEncodingType]) -> ModelStepInputType:
        input_ids = [task_encoding.inputs.ids for task_encoding in task_encodings]
        inputs = self.tokenizer.pad(
            {"input_ids": input_ids}, return_tensors="pt", **self.pad_kwargs
        )

        if not task_encodings[0].has_targets:
            return inputs, None

        tag_ids = [task_encoding.targets for task_encoding in task_encodings]
        targets = self.tokenizer.pad(
            {"input_ids": tag_ids}, return_tensors="pt", **self.pad_kwargs
        )["input_ids"]

        # set the padding label to the label_pad_token_id
        pad_mask = inputs["input_ids"] == self.tokenizer.pad_token_id
        targets[pad_mask] = self.label_pad_token_id

        return inputs, targets

    def unbatch_output(self, model_output: ModelOutputType) -> Sequence[TaskOutputType]:
        logits = model_output["logits"]
        probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()
        indices = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        tags = [[self.id_to_label[e] for e in b] for b in indices]
        return [{"tags": t, "probabilities": p} for t, p in zip(tags, probabilities)]

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncodingType,
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, LabeledSpan]]:
        tokenized_document = task_encoding.metadata["tokenized_document"]
        special_tokens_mask = tokenized_document.metadata["tokenizer_encoding"].special_tokens_mask

        tag_sequence = [
            "O" if is_special_token else tag
            for tag, is_special_token in zip(task_output["tags"], special_tokens_mask)
        ]

        # Note: token_based_document_to_text_based() does not yet consider predictions, so we need to clear
        # the main annotations and attach the predictions to that
        tokenized_document.labeled_spans.clear()
        for label, (start, end_inclusive) in bio_tags_to_spans(
            tag_sequence, include_ill_formed=self.include_ill_formed_predictions
        ):
            token_span_annotation = LabeledSpan(label=label, start=start, end=end_inclusive + 1)
            tokenized_document.labeled_spans.append(token_span_annotation)

        # we can not use self.document_type here because that may be None if self.span_annotation or
        # self.partition_annotation is not the default value
        document_type = (
            TextDocumentWithLabeledSpansAndLabeledPartitions
            if self.partition_annotation
            else TextDocumentWithLabeledSpans
        )
        untokenized_document: Union[
            TextDocumentWithLabeledSpans, TextDocumentWithLabeledSpansAndLabeledPartitions
        ] = token_based_document_to_text_based(
            tokenized_document, result_document_type=document_type
        )

        for span in untokenized_document.labeled_spans:
            # need to copy the span because it can be attached to only one document
            yield self.span_annotation, span.copy()
