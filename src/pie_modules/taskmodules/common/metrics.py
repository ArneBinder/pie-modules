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

from pytorch_ie import Annotation, TaskModule
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

    def update(self, gold, predicted) -> None:
        # include idx to allow for same annotations in different examples (otherwise they would be counted as one
        # because they are not attached to a specific document)
        gold_set = {(self.idx, ann) for ann in gold}
        predicted_set = {(self.idx, ann) for ann in predicted}
        self.gold.extend(gold_set)
        self.predicted.extend(predicted_set)
        self.correct.extend(gold_set & predicted_set)

    def inc_idx(self, n: int = 1):
        self.idx += n


class AnnotationLayerMetric(Metric, Generic[T]):
    def __init__(
        self,
        taskmodule: TaskModule[Any, Any, Any, Any, Any, T],
        layer_names: List[str],
        decode_annotations_func: Callable[[T], Tuple[Dict[str, List[Annotation]], Dict[str, int]]],
        round_precision: Optional[int] = 4,
        key_invalid_correct: Optional[str] = None,
    ):
        super().__init__()
        self.taskmodule = taskmodule

        self.key_invalid_correct = key_invalid_correct
        self.layer_names = layer_names
        self.round_precision = round_precision
        self.decode_annotations_func = decode_annotations_func
        self.layer_metrics = ModuleDict(
            {
                layer_name: PrecisionRecallAndF1ForLabeledAnnotations()
                for layer_name in self.layer_names
            }
        )

        self.reset()

    def update(self, prediction, expected):
        prediction_list = self.taskmodule.unbatch_output(prediction)
        expected_list = self.taskmodule.unbatch_output(expected)

        for expected_encoding, prediction_encoding in zip(expected_list, prediction_list):
            gold_annotations, gold_errors = self.decode_annotations_func(expected_encoding)
            predicted_annotations, predicted_errors = self.decode_annotations_func(
                prediction_encoding
            )
            for k, v in predicted_errors.items():
                self.invalid[k] += v

            for layer_name, metric in self.layer_metrics.items():
                # remove duplicates from layer data
                gold_layer = set(gold_annotations[layer_name])
                pred_layer = set(predicted_annotations[layer_name])
                metric.update(gold_layer, pred_layer)
                metric.inc_idx()

            if expected_encoding == prediction_encoding:
                self.em += 1

            self.total += 1

    def reset(self):
        super().reset()

        for metric in self.layer_metrics.values():
            metric.reset()

        # total number of tuples
        self.total = 1e-13

        self.invalid = defaultdict(int)
        # this contains the number of examples where the full target sequence was predicted correctly (exact matches)
        self.em = 0

    @property
    def state(self) -> Dict[str, Any]:
        # copy to disallow modification of the state
        return {
            "total": copy.copy(self.total),
            "invalid": copy.deepcopy(self.invalid),
            "em": copy.copy(self.em),
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

        res["em"] = self.em / self.total

        for layer_name, metric in self.layer_metrics.items():
            overall_layer_info, layer_info = metric.compute()
            res[layer_name] = layer_info
            res[f"{layer_name}/micro"] = overall_layer_info

        # if invalid contains a "correct" key, use that to normalize, otherwise use the number of training examples
        if self.key_invalid_correct in self.invalid:
            invalid_total = sum(self.invalid.values())
        else:
            invalid_total = self.total
        res["invalid"] = {k: v / invalid_total for k, v in self.invalid.items()}
        res["invalid/all"] = (
            sum(v for k, v in self.invalid.items() if k != self.key_invalid_correct)
            / invalid_total
        )

        res = self._nested_round(res)

        return res
