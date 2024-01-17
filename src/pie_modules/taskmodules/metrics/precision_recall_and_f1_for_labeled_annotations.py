import logging
from collections import Counter
from typing import Any, Collection, Dict, Iterable, Optional, Union

import torch
from pytorch_ie import Annotation
from torch import LongTensor
from torchmetrics import Metric

from pie_modules.utils import flatten_nested_dict

logger = logging.getLogger(__name__)


class PrecisionRecallAndF1ForLabeledAnnotations(Metric):
    """Computes precision, recall and F1 for labeled annotations. Inputs and targets are lists of
    annotations. True positives are counted as the number of annotations that are the same in both
    inputs and targets calculated as exact matches via set operation, false positives and false
    negatives accordingly. The annotations are deduplicated for each instance. But if the same
    annotation occurs in different instances, it is counted as two separate annotations.

    Args:
        label_mapping: A dictionary mapping annotation labels to human-readable labels. If None,
            the annotation labels are used as they are. Can be used to map label ids to string labels.
        key_micro: The key to use for the micro-average in the metric result dictionary.
        in_percent: Whether to return the results in percent, i.e. values between 0 and 100 instead of
            between 0 and 1.
    """

    def __init__(
        self,
        label_mapping: Optional[Dict[Any, str]] = None,
        key_micro: Optional[str] = "micro",
        key_macro: Optional[str] = "macro",
        in_percent: bool = False,
        flatten_result_with_sep: Optional[str] = None,
    ):
        super().__init__()
        self.label_mapping = label_mapping
        self.key_micro = key_micro
        self.key_macro = key_macro
        self.in_percent = in_percent
        self.flatten_result_with_sep = flatten_result_with_sep

    def inc_counts(self, counts: LongTensor, key: Optional[str], prefix: str = "counts_"):
        full_key = prefix
        if key is not None:
            full_key += key

        if not hasattr(self, full_key):
            self.add_state(full_key, default=torch.zeros_like(counts), dist_reduce_fx="sum")

        prev_value = getattr(self, full_key)
        setattr(self, full_key, prev_value + counts)

    def get_counts(self, key_prefix: str = "counts_") -> Dict[Optional[str], LongTensor]:
        result = {}
        for k, v in self.metric_state.items():
            if k.startswith(key_prefix):
                key = k[len(key_prefix) :] or None
                result[key] = v
        return result

    def update(self, gold: Iterable[Annotation], predicted: Iterable[Annotation]) -> None:
        # remove duplicates within each list
        gold_set = set(gold)
        predicted_set = set(predicted)
        new_counts = self.calculate_counts(gold_set, predicted_set, gold_set & predicted_set)
        for k, v in new_counts.items():
            self.inc_counts(counts=v, key=k)

    def get_precision_recall_f1(self, n_gold_predicted_correct: LongTensor) -> Dict[str, float]:
        n_gold = n_gold_predicted_correct[0]
        n_predicted = n_gold_predicted_correct[1]
        n_correct = n_gold_predicted_correct[2]
        recall = 0.0 if n_gold == 0 else (n_correct / n_gold)
        precision = 0.0 if n_predicted == 0 else (n_correct / n_predicted)
        f1 = 0.0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)

        if self.in_percent:
            recall *= 100
            precision *= 100
            f1 *= 100
        return {
            "recall": torch.tensor(recall).to(self.device),
            "precision": torch.tensor(precision).to(self.device),
            "f1": torch.tensor(f1).to(self.device),
        }

    def get_label(self, annotation: Annotation) -> Optional[str]:
        label: Optional[str] = getattr(annotation, "label", None)
        if self.label_mapping is not None:
            return self.label_mapping[label]
        return label

    def calculate_counts(
        self,
        gold: Collection[Annotation],
        predicted: Collection[Annotation],
        correct: Collection[Annotation],
    ) -> Dict[Optional[str], LongTensor]:
        result = {}
        # per class
        gold_counter = Counter([self.get_label(ann) for ann in gold])
        predicted_counter = Counter([self.get_label(ann) for ann in predicted])
        correct_counter = Counter([self.get_label(ann) for ann in correct])
        for label in gold_counter.keys() | predicted_counter.keys():
            if self.key_micro is not None and label == self.key_micro:
                raise ValueError(
                    f"The key '{self.key_micro}' was used as an annotation label, but it is reserved for "
                    f"the micro average. You can change which key is used for that with the 'key_micro' argument."
                )
            result[label] = torch.tensor(
                [
                    gold_counter.get(label, 0),
                    predicted_counter.get(label, 0),
                    correct_counter.get(label, 0),
                ]
            ).to(device=self.device)

        # overall
        if self.key_micro is not None:
            result[self.key_micro] = torch.tensor([len(gold), len(predicted), len(correct)]).to(
                device=self.device
            )
        return result

    def compute(self) -> Union[Dict[str, Any], Dict[Optional[str], dict[str, float]]]:
        counts = self.get_counts()
        result = {label: self.get_precision_recall_f1(counts[label]) for label in counts.keys()}
        if self.key_macro is not None:
            result_without_micro = {
                k: v for k, v in result.items() if self.key_micro is None or k != self.key_micro
            }
            if len(result_without_micro) > 0:
                result[self.key_macro] = {
                    k: torch.stack([v[k] for v in result_without_micro.values()]).mean()
                    for k in ["precision", "recall", "f1"]
                }

        if self.flatten_result_with_sep is not None:
            return flatten_nested_dict(result, sep=self.flatten_result_with_sep)
        else:
            return result
