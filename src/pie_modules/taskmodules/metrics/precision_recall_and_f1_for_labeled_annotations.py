import logging
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from pytorch_ie import Annotation
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
        key_micro: str = "micro",
        in_percent: bool = False,
        flatten_result_with_sep: Optional[str] = None,
    ):
        super().__init__()
        self.label_mapping = label_mapping
        self.key_micro = key_micro
        self.in_percent = in_percent
        self.flatten_result_with_sep = flatten_result_with_sep
        self.add_state("gold", default=[])
        self.add_state("predicted", default=[])
        self.add_state("correct", default=[])
        self.add_state("idx", default=torch.tensor(0))

    def update(self, gold: Iterable[Annotation], predicted: Iterable[Annotation]) -> None:
        # remove duplicates within each list, but collect them with the instance idx to allow
        # for same annotations in different examples (otherwise they would be counted as one
        # because they are not attached to a specific document)
        gold_set = {(self.idx.item(), ann) for ann in set(gold)}
        predicted_set = {(self.idx.item(), ann) for ann in set(predicted)}
        self.gold.extend(gold_set)
        self.predicted.extend(predicted_set)
        self.correct.extend(gold_set & predicted_set)
        self.idx += 1

    def get_precision_recall_f1(
        self, n_gold: int, n_predicted: int, n_correct: int
    ) -> Dict[str, float]:
        recall = 0.0 if n_gold == 0 else (n_correct / n_gold)
        precision = 0.0 if n_predicted == 0 else (n_correct / n_predicted)
        f1 = 0.0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)

        if self.in_percent:
            recall *= 100
            precision *= 100
            f1 *= 100
        return {"recall": recall, "precision": precision, "f1": f1}

    def get_label(self, annotation: Annotation) -> Optional[str]:
        label: Optional[str] = getattr(annotation, "label", None)
        if self.label_mapping is not None:
            return self.label_mapping[label]
        return label

    def get_counts(
        self,
        gold: Sequence[Annotation],
        predicted: Sequence[Annotation],
        correct: Sequence[Annotation],
    ) -> Dict[Optional[str], Tuple[int, int, int]]:
        result = {}
        # per class
        gold_counter = Counter([self.get_label(ann) for idx, ann in gold])
        predicted_counter = Counter([self.get_label(ann) for idx, ann in predicted])
        correct_counter = Counter([self.get_label(ann) for idx, ann in correct])
        for label in gold_counter.keys() | predicted_counter.keys():
            if label == self.key_micro:
                raise ValueError(
                    f"The key '{self.key_micro}' was used as an annotation label, but it is reserved for "
                    f"the micro average. You can change which key is used for that with the 'key_micro' argument."
                )
            result[label] = (
                gold_counter.get(label, 0),
                predicted_counter.get(label, 0),
                correct_counter.get(label, 0),
            )

        # overall
        result[self.key_micro] = (len(gold), len(predicted), len(correct))
        return result

    def compute(self) -> Union[Dict[str, Any], Dict[Optional[str], dict[str, float]]]:
        counts = self.get_counts(self.gold, self.predicted, self.correct)
        result = {label: self.get_precision_recall_f1(*counts[label]) for label in counts.keys()}

        if self.flatten_result_with_sep is not None:
            return flatten_nested_dict(result, sep=self.flatten_result_with_sep)
        else:
            return result
