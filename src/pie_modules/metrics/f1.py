import logging
from collections import defaultdict
from functools import partial
from typing import Callable, Collection, Dict, Hashable, Optional, Tuple, Union

import pandas as pd
from pie_core import Annotation, Document, DocumentMetric
from pie_core.utils.hydra import resolve_target

logger = logging.getLogger(__name__)


def has_one_of_the_labels(ann: Annotation, label_field: str, labels: Collection[str]) -> bool:
    return getattr(ann, label_field) in labels


def has_this_label(ann: Annotation, label_field: str, label: str) -> bool:
    return getattr(ann, label_field) == label


class F1Metric(DocumentMetric):
    """Computes the (micro aggregated) F1 score for a given layer. If labels are provided, it also
    computes the F1 score for each label separately and the macro F1 score.

    Args:
        layer: The layer to compute the F1 score for.
        labels: If provided, calculate F1 score for each label.
        label_field: The field to use for the label. Defaults to "label".
        show_as_markdown: If True, logs the F1 score as markdown on the console when calling compute().
    """

    def __init__(
        self,
        layer: str,
        labels: Optional[Union[Collection[str], str]] = None,
        label_field: str = "label",
        show_as_markdown: bool = False,
        annotation_processor: Optional[Union[Callable[[Annotation], Hashable], str]] = None,
    ):
        super().__init__()
        self.layer = layer
        self.label_field = label_field
        self.show_as_markdown = show_as_markdown
        if isinstance(annotation_processor, str):
            self.annotation_processor = resolve_target(annotation_processor)
        else:
            self.annotation_processor = annotation_processor

        self.per_label = labels is not None
        self.infer_labels = False
        if self.per_label:
            if isinstance(labels, str):
                if labels != "INFERRED":
                    raise ValueError(
                        "labels can only be 'INFERRED' if per_label is True and labels is a string"
                    )
                self.labels = []
                self.infer_labels = True
            elif isinstance(labels, Collection):
                if not all(isinstance(label, str) for label in labels):
                    raise ValueError("labels must be a collection of strings")
                if "MICRO" in labels or "MACRO" in labels:
                    raise ValueError(
                        "labels cannot contain 'MICRO' or 'MACRO' because they are used to capture aggregated metrics"
                    )
                if len(labels) == 0:
                    raise ValueError("labels cannot be empty")
                self.labels = list(labels)
            else:
                raise ValueError("labels must be a string or a collection of strings")

    def reset(self) -> None:
        self.counts: Dict[str, Tuple[int, int, int]] = defaultdict(lambda: (0, 0, 0))

    def calculate_counts(
        self,
        document: Document,
        annotation_filter: Optional[Callable[[Annotation], bool]] = None,
        annotation_processor: Optional[Callable[[Annotation], Hashable]] = None,
    ) -> Tuple[int, int, int]:
        annotation_processor = annotation_processor or (lambda ann: ann)
        annotation_filter = annotation_filter or (lambda ann: True)
        predicted_annotations = {
            annotation_processor(ann)
            for ann in document[self.layer].predictions
            if annotation_filter(ann)
        }
        gold_annotations = {
            annotation_processor(ann) for ann in document[self.layer] if annotation_filter(ann)
        }
        tp = len([ann for ann in predicted_annotations & gold_annotations])
        fn = len([ann for ann in gold_annotations - predicted_annotations])
        fp = len([ann for ann in predicted_annotations - gold_annotations])
        return tp, fp, fn

    def add_counts(self, counts: Tuple[int, int, int], label: str) -> None:
        self.counts[label] = (
            self.counts[label][0] + counts[0],
            self.counts[label][1] + counts[1],
            self.counts[label][2] + counts[2],
        )

    def _update(self, document: Document) -> None:
        new_counts = self.calculate_counts(
            document=document,
            annotation_filter=(
                partial(has_one_of_the_labels, label_field=self.label_field, labels=self.labels)
                if self.per_label and not self.infer_labels
                else None
            ),
            annotation_processor=self.annotation_processor,
        )
        self.add_counts(new_counts, label="MICRO")
        if self.infer_labels:
            layer = document[self.layer]
            # collect labels from gold data and predictions
            for ann in list(layer) + list(layer.predictions):
                label = getattr(ann, self.label_field)
                if label not in self.labels:
                    self.labels.append(label)
        if self.per_label:
            for label in self.labels:
                new_counts = self.calculate_counts(
                    document=document,
                    annotation_filter=partial(
                        has_this_label, label_field=self.label_field, label=label
                    ),
                    annotation_processor=self.annotation_processor,
                )
                self.add_counts(new_counts, label=label)

    def _compute(self) -> Dict[str, Dict[str, float]]:
        res = dict()
        if self.per_label:
            res["MACRO"] = {"f1": 0.0, "p": 0.0, "r": 0.0}
        for label, counts in self.counts.items():
            tp, fp, fn = counts
            if tp == 0:
                p, r, f1 = 0.0, 0.0, 0.0
            else:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * p * r / (p + r)
            res[label] = {"f1": f1, "p": p, "r": r, "s": tp + fn}
            if self.per_label and label in self.labels:
                res["MACRO"]["f1"] += f1 / len(self.labels)
                res["MACRO"]["p"] += p / len(self.labels)
                res["MACRO"]["r"] += r / len(self.labels)
        if self.show_as_markdown:
            logger.info(f"\n{self.layer}:\n{pd.DataFrame(res).round(3).T.to_markdown()}")
        return res
