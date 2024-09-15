import copy
import logging
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import torch
from pytorch_ie import Annotation
from pytorch_ie.annotations import Span
from pytorch_ie.core import TaskEncoding, TaskModule
from pytorch_ie.utils.window import get_window_around_slice
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC
from transformers import AutoTokenizer, BatchEncoding
from typing_extensions import TypeAlias

from pie_modules.documents import (
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
    score: float
    is_valid: bool


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


class SpanNotAlignedWithTokenException(Exception):
    def __init__(self, span):
        self.span = span


class SpanDoesNotFitIntoAvailableWindow(Exception):
    def __init__(self, span):
        self.span = span


def _get_labels(model_output: ModelTargetType) -> torch.Tensor:
    return model_output["labels"]


@TaskModule.register()
class CrossTextBinaryCorefTaskModule(RelationStatisticsMixin, TaskModuleType):
    """This taskmodule processes documents of type
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations in preparation for a
    SequencePairSimilarityModelWithPooler."""

    DOCUMENT_TYPE = DocumentType

    def __init__(
        self,
        tokenizer_name_or_path: str,
        max_window: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_window = max_window if max_window is not None else self.tokenizer.model_max_length
        self.available_window = self.max_window - self.tokenizer.num_special_tokens_to_add()
        self.num_special_tokens_before = len(self._get_special_tokens_before_input())

    def _get_special_tokens_before_input(self) -> List[int]:
        dummy_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=[-1])
        return dummy_ids[: dummy_ids.index(-1)]

    def encode(self, documents: Union[DocumentType, Iterable[DocumentType]], **kwargs):
        self.reset_statistics()
        result = super().encode(documents=documents, **kwargs)
        self.show_statistics()
        return result

    def truncate_encoding_around_span(
        self, encoding: BatchEncoding, char_span: Span
    ) -> Tuple[Dict[str, List[int]], Span]:
        input_ids = copy.deepcopy(encoding["input_ids"])

        token_start = encoding.char_to_token(char_span.start)
        token_end_before = encoding.char_to_token(char_span.end - 1)
        if token_start is None or token_end_before is None:
            raise SpanNotAlignedWithTokenException(span=char_span)
        token_end = token_end_before + 1

        # truncate input_ids and shift token_start and token_end
        if len(input_ids) > self.available_window:
            window_slice = get_window_around_slice(
                slice=[token_start, token_end],
                max_window_size=self.available_window,
                available_input_length=len(input_ids),
            )
            if window_slice is None:
                raise SpanDoesNotFitIntoAvailableWindow(span=(token_start, token_end))
            window_start, window_end = window_slice
            input_ids = input_ids[window_start:window_end]
            token_start -= window_start
            token_end -= window_start

        truncated_encoding = self.tokenizer.prepare_for_model(ids=input_ids)
        # shift indices because we added special tokens to the input_ids
        token_start += self.num_special_tokens_before
        token_end += self.num_special_tokens_before

        return truncated_encoding, Span(start=token_start, end=token_end)

    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        self.collect_all_relations(kind="available", relations=document.binary_coref_relations)
        tokenizer_kwargs = dict(
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        encoding = self.tokenizer(text=document.text, **tokenizer_kwargs)
        encoding_pair = self.tokenizer(text=document.text_pair, **tokenizer_kwargs)

        task_encodings = []
        for coref_rel in document.binary_coref_relations:
            # TODO: This can miss instances if both texts are the same. We could check that
            #   coref_rel.head is in document.labeled_spans (same for the tail), but would this
            #   slow down the encoding?
            if not (
                coref_rel.head.target == document.text
                or coref_rel.tail.target == document.text_pair
            ):
                raise ValueError(
                    f"It is expected that coref relations go from (head) spans over 'text' "
                    f"to (tail) spans over 'text_pair', but this is not the case for this "
                    f"relation (i.e. it points into the other direction): {coref_rel.resolve()}"
                )
            try:
                current_encoding, token_span = self.truncate_encoding_around_span(
                    encoding=encoding, char_span=coref_rel.head
                )
                current_encoding_pair, token_span_pair = self.truncate_encoding_around_span(
                    encoding=encoding_pair, char_span=coref_rel.tail
                )
            except SpanNotAlignedWithTokenException as e:
                logger.warning(
                    f"Could not get token offsets for argument ({e.span}) of coref relation: "
                    f"{coref_rel.resolve()}. Skip it."
                )
                self.collect_relation(kind="skipped_args_not_aligned", relation=coref_rel)
                continue
            except SpanDoesNotFitIntoAvailableWindow as e:
                logger.warning(
                    f"Argument span [{e.span}] does not fit into available token window "
                    f"({self.available_window}). Skip it."
                )
                self.collect_relation(
                    kind="skipped_span_does_not_fit_into_window", relation=coref_rel
                )
                continue

            task_encodings.append(
                TaskEncoding(
                    document=document,
                    inputs={
                        "encoding": current_encoding,
                        "encoding_pair": current_encoding_pair,
                        "pooler_start_indices": token_span.start,
                        "pooler_end_indices": token_span.end,
                        "pooler_pair_start_indices": token_span_pair.start,
                        "pooler_pair_end_indices": token_span_pair.end,
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
        label_ids = model_output["labels"].detach().cpu().tolist()
        probabilities = model_output["probabilities"].detach().cpu().tolist()
        result: List[TaskOutputType] = [
            {"is_valid": label_id != 0, "score": prob}
            for label_id, prob in zip(label_ids, probabilities)
        ]
        return result

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncodingType, TargetEncodingType],
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Annotation]]:
        if task_output["is_valid"]:
            score = task_output["score"]
            new_coref_rel = task_encoding.metadata["candidate_annotation"].copy(score=score)
            yield "binary_coref_relations", new_coref_rel
