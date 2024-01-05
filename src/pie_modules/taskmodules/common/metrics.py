import copy
import logging
from collections import Counter, defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import torch
from pytorch_ie import Annotation
from torch.nn import ModuleDict
from torchmetrics import Metric

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WrappedMetricWithUnbatchFunction(Metric, Generic[T]):
    """A wrapper around a metric that can be used with a batched input.

    Args:
        unbatch_function: A function that takes a batched input and returns an iterable of
            individual inputs. This is used to unbatch the input before passing it to the wrapped
            metric.
        metric: The metric to wrap. It should be a subclass of torchmetrics.Metric.
    """

    def __init__(
        self, unbatch_function: Callable[[T], Iterable[Any]], metric: Metric, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.unbatch_function = unbatch_function
        self.metric = metric

    def update(self, predictions: T, targets: T) -> None:
        prediction_list = self.unbatch_function(predictions)
        target_list = self.unbatch_function(targets)
        for prediction_str, target_str in zip(prediction_list, target_list):
            self.metric(prediction_str, target_str)

    def compute(self) -> Any:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()


class PrecisionRecallAndF1ForLabeledAnnotations(Metric):
    def __init__(
        self,
        label_mapping: Optional[Dict[Any, str]] = None,
    ):
        super().__init__()
        self.label_mapping = label_mapping
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.gold: List[Tuple[int, Annotation]] = []
        self.predicted: List[Tuple[int, Annotation]] = []
        self.correct: List[Tuple[int, Annotation]] = []
        self.idx = 0

    @property
    def state(self) -> Dict[str, Any]:
        # copy to disallow modification of the state
        return {
            "gold": copy.deepcopy(self.gold),
            "predicted": copy.deepcopy(self.predicted),
            "correct": copy.deepcopy(self.correct),
        }

    def get_precision_recall_f1(
        self, n_gold: int, n_predicted: int, n_correct: int
    ) -> Dict[str, float]:
        recall = 0.0 if n_gold == 0 else (n_correct / n_gold)
        precision = 0.0 if n_predicted == 0 else (n_correct / n_predicted)
        f1 = 0.0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)

        recall *= 100
        precision *= 100
        f1 *= 100
        return {"recall": recall, "precision": precision, "f1": f1}

    def get_label(self, annotation: Annotation) -> Optional[str]:
        label: Optional[str] = getattr(annotation, "label", None)
        if self.label_mapping is not None:
            return self.label_mapping[label]
        return label

    def compute(self) -> Tuple[Dict[str, float], Dict[Optional[str], Dict[str, float]]]:
        # per class
        per_class: Dict[Optional[str], Dict[str, float]] = {}
        gold_counter = Counter([self.get_label(ann) for idx, ann in self.gold])
        predicted_counter = Counter([self.get_label(ann) for idx, ann in self.predicted])
        correct_counter = Counter([self.get_label(ann) for idx, ann in self.correct])
        for label in gold_counter.keys() | predicted_counter.keys():
            n_gold = gold_counter.get(label, 0)
            n_predicted = predicted_counter.get(label, 0)
            n_correct = correct_counter.get(label, 0)
            per_class[label] = self.get_precision_recall_f1(n_gold, n_predicted, n_correct)

        # overall
        n_gold = len(self.gold)
        n_predicted = len(self.predicted)
        n_correct = len(self.correct)
        overall = self.get_precision_recall_f1(n_gold, n_predicted, n_correct)

        return overall, per_class

    def update(self, gold: Iterable[Annotation], predicted: Iterable[Annotation]) -> None:
        # remove duplicates within each list, but collect them with the instance idx to allow
        # for same annotations in different examples (otherwise they would be counted as one
        # because they are not attached to a specific document)
        gold_set = {(self.idx, ann) for ann in set(gold)}
        predicted_set = {(self.idx, ann) for ann in set(predicted)}
        self.gold.extend(gold_set)
        self.predicted.extend(predicted_set)
        self.correct.extend(gold_set & predicted_set)
        self.idx += 1


U = TypeVar("U")


class WrappedLayerMetricsWithUnbatchAndDecodingFunction(Metric, Generic[T, U]):
    """A wrapper around annotation layer metrics that can be used with batched encoded annotations.

    Args:
        layer_metrics: A dictionary mapping layer names to annotation layer metrics. Each metric
            should be a subclass of torchmetrics.Metric and should take two sets of annotations as
            input.
        unbatch_function: A function that takes a batched input and returns an iterable of
            individual inputs. This is used to unbatch the input before passing it to the annotation
            decoding function (decode_annotations_with_errors_function).
        decode_annotations_with_errors_function: A function that takes an annotation encoding and
            returns a tuple of two dictionaries. The first dictionary maps layer names to a list of
            annotations. The second dictionary maps error names to the number of errors that were
            encountered while decoding the annotations.
        round_precision: The number of digits to round the results to. If None, no rounding is
            performed.
        error_key_correct: The key in the error dictionary whose value should be the number of *correctly*
            decoded annotations, so that the sum of all values in the error dictionary can be used to
            normalize the error counts. If None, the total number of training examples is used to
            normalize the error counts.
        collect_encoding_matches: Whether to collect the number of examples where the full target encoding
            was predicted correctly (exact matches).
    """

    def __init__(
        self,
        layer_metrics: Dict[str, Metric],
        unbatch_function: Callable[[T], Iterable[U]],
        decode_annotations_with_errors_function: Callable[
            [U], Tuple[Dict[str, Iterable[Annotation]], Dict[str, int]]
        ],
        round_precision: Optional[int] = 4,
        error_key_correct: Optional[str] = None,
        collect_encoding_matches: bool = True,
    ):
        super().__init__()

        self.key_error_correct = error_key_correct
        self.collect_encoding_matches = collect_encoding_matches
        self.round_precision = round_precision
        self.unbatch_function = unbatch_function
        self.decode_annotations_with_errors_func = decode_annotations_with_errors_function
        self.layer_metrics = ModuleDict(layer_metrics)

        self.reset()

    def update(self, prediction, expected):
        prediction_list = self.unbatch_function(prediction)
        expected_list = self.unbatch_function(expected)

        for expected_encoding, prediction_encoding in zip(expected_list, prediction_list):
            gold_annotations, _ = self.decode_annotations_with_errors_func(expected_encoding)
            predicted_annotations, predicted_errors = self.decode_annotations_with_errors_func(
                prediction_encoding
            )
            for k, v in predicted_errors.items():
                self.errors[k] += v

            for layer_name, metric in self.layer_metrics.items():
                metric.update(gold_annotations[layer_name], predicted_annotations[layer_name])

            if self.collect_encoding_matches:
                if isinstance(expected_encoding, torch.Tensor) and isinstance(
                    prediction_encoding, torch.Tensor
                ):
                    is_match = torch.equal(expected_encoding, prediction_encoding)
                else:
                    is_match = expected_encoding == prediction_encoding
                if is_match:
                    self.encoding_match += 1

            self.total += 1

    def reset(self):
        super().reset()

        for metric in self.layer_metrics.values():
            metric.reset()

        # total number of tuples
        self.total = 1e-13

        self.errors = defaultdict(int)
        # this contains the number of examples where the full target sequence was predicted correctly (exact matches)
        self.encoding_match = 0

    @property
    def state(self) -> Dict[str, Any]:
        # copy to disallow modification of the state
        return {
            "total": copy.copy(self.total),
            "errors": copy.deepcopy(self.errors),
            "encoding_match": copy.copy(self.encoding_match),
            "layer_metrics": {
                layer_name: metric.state for layer_name, metric in self.layer_metrics.items()
            },
        }

    def _nested_round(self, d: Dict[str, Any]) -> Dict[str, Any]:
        if self.round_precision is None:
            return d
        res: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                res[k] = self._nested_round(v)
            elif isinstance(v, float):
                res[k] = round(v, self.round_precision)
            else:
                res[k] = v
        return res

    def compute(self):
        res = {}

        if self.collect_encoding_matches:
            res["encoding_match"] = self.encoding_match / self.total

        for layer_name, metric in self.layer_metrics.items():
            overall_layer_info, layer_info = metric.compute()
            res[layer_name] = layer_info
            res[f"{layer_name}/micro"] = overall_layer_info

        # if errors contains a "correct" key, use that to normalize, otherwise use the number of training examples
        if self.key_error_correct in self.errors:
            errors_total = sum(self.errors.values())
        else:
            errors_total = self.total
        res["errors"] = {k: v / errors_total for k, v in self.errors.items()}
        res["errors/all"] = (
            sum(v for k, v in self.errors.items() if k != self.key_error_correct) / errors_total
        )

        res = self._nested_round(res)

        return res
