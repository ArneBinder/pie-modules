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
from pytorch_ie.core import TaskEncoding, TaskModule
from transformers import AutoTokenizer
from typing_extensions import TypeAlias

from pie_modules.documents import (
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations,
)
from pie_modules.taskmodules.common.mixins import RelationStatisticsMixin

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
    label_pair: Tuple[str, str]
    entailment_probability_pair: Tuple[float, float]


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


@TaskModule.register()
class CrossTextBinaryCorefTaskModuleByNli(RelationStatisticsMixin, TaskModuleType):
    """This taskmodule processes documents of type
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations in preparation for a sequence
    classification model trained for NLI."""

    DOCUMENT_TYPE = DocumentType

    def __init__(
        self,
        tokenizer_name_or_path: str,
        labels: List[str],
        entailment_label: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.labels = labels
        self.entailment_label = entailment_label
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def _post_prepare(self):
        self.id_to_label = dict(enumerate(self.labels))
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.entailment_idx = self.label_to_id[self.entailment_label]

    def encode(self, documents: Union[DocumentType, Iterable[DocumentType]], **kwargs):
        self.reset_statistics()
        result = super().encode(documents=documents, **kwargs)
        self.show_statistics()
        return result

    def encode_input(
        self,
        document: DocumentType,
        is_training: bool = False,
    ) -> Optional[Union[TaskEncodingType, Sequence[TaskEncodingType]]]:
        self.collect_all_relations(kind="available", relations=document.binary_coref_relations)
        result = []
        for coref_rel in document.binary_coref_relations:
            head_text = str(coref_rel.head)
            tail_text = str(coref_rel.tail)
            task_encoding = TaskEncoding(
                document=document,
                inputs={"text": [head_text, tail_text], "text_pair": [tail_text, head_text]},
                metadata={"candidate_annotation": coref_rel},
            )
            result.append(task_encoding)
            self.collect_relation("used", coref_rel)
        return result

    def encode_target(
        self,
        task_encoding: TaskEncodingType,
    ) -> Optional[TargetEncodingType]:
        raise NotImplementedError()

    def collate(
        self,
        task_encodings: Sequence[
            TaskEncoding[DocumentType, InputEncodingType, TargetEncodingType]
        ],
    ) -> Tuple[ModelInputType, Optional[ModelTargetType]]:
        all_texts = []
        all_texts_pair = []
        for task_encoding in task_encodings:
            all_texts.extend(task_encoding.inputs["text"])
            all_texts_pair.extend(task_encoding.inputs["text_pair"])
        inputs = self.tokenizer(
            text=all_texts,
            text_pair=all_texts_pair,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        if not task_encodings[0].has_targets:
            return inputs, None
        raise NotImplementedError()

    def unbatch_output(self, model_output: ModelTargetType) -> Sequence[TaskOutputType]:
        probs_tensor = model_output["probabilities"]
        labels_tensor = model_output["labels"]

        bs, num_classes = probs_tensor.size()
        # Reshape the probs tensor to (bs/2, 2, num_classes)
        probs_paired = probs_tensor.view(bs // 2, 2, num_classes).detach().cpu().tolist()

        # Reshape the labels tensor to (bs/2, 2)
        labels_paired = labels_tensor.view(bs // 2, 2).detach().cpu().tolist()

        result = []
        for (label_id, label_id_pair), (probs_list, probs_list_pair) in zip(
            labels_paired, probs_paired
        ):
            task_output: TaskOutputType = {
                "label_pair": (self.id_to_label[label_id], self.id_to_label[label_id_pair]),
                "entailment_probability_pair": (
                    probs_list[self.entailment_idx],
                    probs_list_pair[self.entailment_idx],
                ),
            }
            result.append(task_output)
        return result

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncodingType, TargetEncodingType],
        task_output: TaskOutputType,
    ) -> Iterator[Tuple[str, Annotation]]:
        if all(label == self.entailment_label for label in task_output["label_pair"]):
            probs = task_output["entailment_probability_pair"]
            score = (probs[0] + probs[1]) / 2
            new_coref_rel = task_encoding.metadata["candidate_annotation"].copy(score=score)
            yield "binary_coref_relations", new_coref_rel
