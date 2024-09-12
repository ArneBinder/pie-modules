import logging
from collections import defaultdict
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import torch
from pytorch_ie import Annotation
from pytorch_ie.core import TaskEncoding, TaskModule
from pytorch_ie.taskmodules.interface import ChangesTokenizerVocabSize
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC
from transformers import AutoTokenizer
from typing_extensions import TypeAlias

from pie_modules.document.types import (
    BinaryCorefRelation,
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations,
)
from pie_modules.taskmodules.common.mixins import RelationStatisticsMixin
from pie_modules.taskmodules.metrics import WrappedMetricWithPrepareFunction
from pie_modules.utils import list_of_dicts2dict_of_lists

logger = logging.getLogger(__name__)

InputEncodingType: TypeAlias = Dict[str, Any]
TargetEncodingType: TypeAlias = Sequence[float]
DocumentType: TypeAlias = TextPairDocumentWithLabeledSpansAndBinaryCorefRelations

TaskEncodingType: TypeAlias = TaskEncoding[
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
]


class TaskOutputType(TypedDict, total=False):
    scores: Sequence[str]


ModelInputType: TypeAlias = Dict[str, torch.Tensor]
ModelTargetType: TypeAlias = Dict[str, torch.Tensor]
ModelOutputType: TypeAlias = Dict[str, torch.Tensor]

TaskModuleType: TypeAlias = TaskModule[
    # _InputEncoding, _TargetEncoding, _TaskBatchEncoding, _ModelBatchOutput, _TaskOutput
    DocumentType,
    InputEncodingType,
    TargetEncodingType,
    Tuple[ModelInputType, Optional[ModelTargetType]],
    ModelTargetType,
    TaskOutputType,
]


def _get_labels(model_output: ModelTargetType) -> torch.Tensor:
    return model_output["labels"]


@TaskModule.register()
class CrossTextBinaryCorefTaskModule(
    RelationStatisticsMixin, TaskModuleType, ChangesTokenizerVocabSize
):
    DOCUMENT_TYPE = DocumentType

    def __init__(
        self,
        tokenizer_name_or_path: str,
        add_negative_relations: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.add_negative_relations = add_negative_relations

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def _add_negative_relations(self, positives: Iterable[DocumentType]) -> Iterable[DocumentType]:
        positive_tuples = defaultdict(set)
        text2spans = defaultdict(set)
        for doc in positives:
            for labeled_span in doc.labeled_spans:
                text2spans[doc.text].add(labeled_span.copy())
            for labeled_span in doc.labeled_spans_pair:
                text2spans[doc.text_pair].add(labeled_span.copy())

            for coref in doc.binary_coref_relations:
                positive_tuples[(doc.text, doc.text_pair)].add(
                    (coref.head.copy(), coref.tail.copy())
                )
                positive_tuples[(doc.text_pair, doc.text)].add(
                    (coref.tail.copy(), coref.head.copy())
                )

        new_docs = []
        for text in sorted(text2spans):
            for text_pair in sorted(text2spans):
                current_positives = positive_tuples.get((text, text_pair), set())
                new_doc = TextPairDocumentWithLabeledSpansAndBinaryCorefRelations(
                    text=text, text_pair=text_pair
                )
                new_doc.labeled_spans.extend(
                    labeled_span.copy() for labeled_span in text2spans[text]
                )
                new_doc.labeled_spans_pair.extend(
                    labeled_span.copy() for labeled_span in text2spans[text_pair]
                )
                for s in sorted(new_doc.labeled_spans):
                    for s_p in sorted(new_doc.labeled_spans_pair):
                        # exclude relations to itself
                        if text == text_pair and s.copy() == s_p.copy():
                            continue
                        if s.label != s_p.label:
                            continue
                        score = 1.0 if (s.copy(), s_p.copy()) in current_positives else 0.0
                        new_coref_rel = BinaryCorefRelation(head=s, tail=s_p, score=score)
                        new_doc.binary_coref_relations.append(new_coref_rel)
                new_docs.append(new_doc)

        return new_docs

    def encode(self, documents: Union[DocumentType, Iterable[DocumentType]], **kwargs):
        self.reset_statistics()
        if self.add_negative_relations:
            if isinstance(documents, DocumentType):
                documents = [documents]
            documents = self._add_negative_relations(documents)

        result = super().encode(documents=documents, **kwargs)
        self.show_statistics()
        return result

    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        self.collect_all_relations(kind="available", relations=document.binary_coref_relations)
        tokenizer_kwargs = dict(
            padding=False,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_offsets_mapping=False,
            add_special_tokens=True,
        )
        encoding = self.tokenizer(text=document.text, **tokenizer_kwargs)
        encoding_pair = self.tokenizer(text=document.text_pair, **tokenizer_kwargs)

        task_encodings = []
        for coref_rel in document.binary_coref_relations:
            start = encoding.char_to_token(coref_rel.head.start)
            end = encoding.char_to_token(coref_rel.head.end - 1) + 1
            start_pair = encoding_pair.char_to_token(coref_rel.tail.start)
            end_pair = encoding_pair.char_to_token(coref_rel.tail.end - 1) + 1
            if any(offset is None for offset in [start, end, start_pair, end_pair]):
                logger.warning(
                    f"Could not get token offsets for arguments of coref relation: {coref_rel.resolve()}. Skip it."
                )
                self.collect_relation(kind="skipped_args_not_aligned", relation=coref_rel)
                continue
            task_encodings.append(
                TaskEncoding(
                    document=document,
                    inputs={
                        "encoding": encoding,
                        "encoding_pair": encoding_pair,
                        "pooler_start_indices": start,
                        "pooler_end_indices": end,
                        "pooler_pair_start_indices": start_pair,
                        "pooler_pair_end_indices": end_pair,
                    },
                    metadata={"candidate_annotation": coref_rel},
                )
            )
            self.collect_relation("used", coref_rel)
        return task_encodings

    def encode_target(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncodingType, TargetEncodingType],
    ) -> Optional[TargetEncodingType]:
        return task_encoding.metadata["candidate_annotation"].score

    def collate(
        self,
        task_encodings: Sequence[
            TaskEncoding[DocumentType, InputEncodingType, TargetEncodingType]
        ],
    ) -> Tuple[ModelInputType, Optional[ModelTargetType]]:
        inputs_dict = list_of_dicts2dict_of_lists(
            [task_encoding.inputs for task_encoding in task_encodings]
        )

        inputs = {
            k: self.tokenizer.pad(v, return_tensors="pt").data
            if k in ["encoding", "encoding_pair"]
            else torch.tensor(v)
            for k, v in inputs_dict.items()
        }
        for k, v in inputs.items():
            if k.startswith("pooler_") and k.endswith("_indices"):
                inputs[k] = v.unsqueeze(-1)

        if not task_encodings[0].has_targets:
            return inputs, None
        targets = {
            "labels": torch.tensor([task_encoding.targets for task_encoding in task_encodings])
        }
        return inputs, targets

    def configure_model_metric(self, stage: str) -> Metric:
        return WrappedMetricWithPrepareFunction(
            metric=MetricCollection({"auroc": BinaryAUROC(thresholds=None)}),
            prepare_function=_get_labels,
        )

    def unbatch_output(self, model_output: ModelTargetType) -> Sequence[TaskOutputType]:
        raise NotImplementedError()

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncodingType, TargetEncodingType],
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Annotation]]:
        raise NotImplementedError()
