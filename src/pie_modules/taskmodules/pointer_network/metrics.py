import logging
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import torch
from pytorch_ie import Annotation, TaskModule
from torch.nn import ModuleDict
from torchmetrics import Metric

logger = logging.getLogger(__name__)


class LabeledAnnotationScores(Metric):
    def __init__(
        self,
        label_mapping: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.label_mapping = label_mapping
        self.reset()

    def reset(self):
        super().reset()
        self.gold = []
        self.predicted = []
        self.correct = []

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

    def compute(self) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        # per class
        per_class: Dict[str, Dict[str, float]] = {}
        gold_counter = Counter([x.label for x in self.gold])
        predicted_counter = Counter([x.label for x in self.predicted])
        correct_counter = Counter([x.label for x in self.correct])
        for label in gold_counter.keys() | predicted_counter.keys():
            if self.label_mapping is not None:
                label = self.label_mapping[label]
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

    def update(self, gold, predicted):
        gold_set = set(gold)
        predicted_set = set(predicted)
        self.gold.extend(gold_set)
        self.predicted.extend(predicted_set)
        self.correct.extend(gold_set & predicted_set)


T = TypeVar("T")


class AnnotationLayerMetric(Metric, Generic[T]):
    def __init__(
        self,
        eos_id: int,
        taskmodule: TaskModule[Any, Any, Any, Any, Any, T],
        layer_names: List[str],
        decode_annotations_func: Callable[[T], Tuple[Dict[str, List[Annotation]], Dict[str, int]]],
        round_precision: Optional[int] = 4,
    ):
        super().__init__()
        self.taskmodule = taskmodule

        self.layer_names = layer_names
        self.round_precision = round_precision
        self.decode_annotations_func = decode_annotations_func
        self.eos_id = eos_id
        self.layer_metrics = ModuleDict(
            {layer_name: LabeledAnnotationScores() for layer_name in self.layer_names}
        )

        self.reset()

    def get_exact_matches(self, prediction, expected) -> int:
        bsz = prediction.size(0)

        # TODO: is this to get the first eos index? Note that we use eos also for padding...
        pred_eos_index = prediction.flip(dims=[1]).eq(self.eos_id).cumsum(dim=1).long()
        expected_eos_index = expected.flip(dims=[1]).eq(self.eos_id).cumsum(dim=1).long()

        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        expected_seq_len = (
            expected_eos_index.flip(dims=[1]).eq(expected_eos_index[:, -1:]).sum(dim=1)
        )  # bsz
        expected_seq_len = (expected_seq_len - 2).tolist()

        em = 0
        for i in range(bsz):
            # delete </s>
            # Note: I have absolutely no idea why this is not the same as:
            # expected[i, 1:expected_seq_len[i]]
            ts_tensor = expected[:, 1:][i, : expected_seq_len[i]]
            ps_tensor = prediction[:, 1:][i, : pred_seq_len[i]]
            if torch.equal(ts_tensor, ps_tensor):
                em += 1

        return em

    def update(self, prediction, expected):
        prediction_list = self.taskmodule.unbatch_output(prediction)
        expected_list = self.taskmodule.unbatch_output(expected)
        bsz = len(prediction_list)
        self.total += bsz

        for i in range(bsz):
            expected_encoding = expected_list[i]
            prediction_encoding = prediction_list[i]
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

            if expected_encoding == prediction_encoding:
                self.em += 1

        self.em_original += self.get_exact_matches(prediction, expected)

    def reset(self):
        super().reset()

        for metric in self.layer_metrics.values():
            metric.reset()

        # total number of tuples
        self.total = 1e-13

        self.invalid = defaultdict(int)
        # this contains the number of examples where the full target sequence was predicted correctly (exact matches)
        self.em = 0
        self.em_original = 0

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
        # TODO: remove if em (above) is correct
        res["em_original"] = self.em_original / self.total

        for layer_name, metric in self.layer_metrics.items():
            overall_layer_info, layer_info = metric.compute()
            res[layer_name] = layer_info
            res[layer_name + "/micro"] = overall_layer_info

        # if invalid contains a "correct" key, use that to normalize, otherwise use the number of training examples
        if "correct" in self.invalid:
            invalid_total = sum(self.invalid.values())
            # remove the "correct" entry to get the correct value for invalid/all below
            self.invalid.pop("correct")
        else:
            invalid_total = self.total
        res["invalid"] = {k: v / invalid_total for k, v in self.invalid.items()}
        res["invalid/all"] = sum(self.invalid.values()) / invalid_total

        res = self._nested_round(res)

        return res
