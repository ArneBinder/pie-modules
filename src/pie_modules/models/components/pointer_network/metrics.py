from collections import Counter, defaultdict
from typing import Dict, Optional, Tuple

import torch

from pie_modules.taskmodules.components.seq2seq import AnnotationEncoderDecoder


class LabeledAnnotationScore:
    def __init__(self, label_mapping: Optional[Dict[int, str]] = None):
        self.label_mapping = label_mapping
        self.reset()

    def reset(self):
        self.gold = []
        self.predicted = []
        self.correct = []

    def compute(self, n_gold: int, n_predicted: int, n_correct: int) -> Tuple[float, float, float]:
        recall = 0 if n_gold == 0 else (n_correct / n_gold)
        precision = 0 if n_predicted == 0 else (n_correct / n_predicted)
        f1 = 0.0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall * 100, precision * 100, f1 * 100

    def result(self) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        class_info: Dict[str, Dict[str, float]] = {}
        gold_counter = Counter([x.label for x in self.gold])
        predicted_counter = Counter([x.label for x in self.predicted])
        correct_counter = Counter([x.label for x in self.correct])
        for label, count in gold_counter.items():
            if self.label_mapping is not None:
                label = self.label_mapping[label]
            n_gold = count
            n_predicted = predicted_counter.get(label, 0)
            n_correct = correct_counter.get(label, 0)
            recall, precision, f1 = self.compute(n_gold, n_predicted, n_correct)
            class_info[label] = {
                "acc": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        n_gold = len(self.gold)
        n_predicted = len(self.predicted)
        n_correct = len(self.correct)
        recall, precision, f1 = self.compute(n_gold, n_predicted, n_correct)
        return {"acc": precision, "recall": recall, "f1": f1}, class_info

    def update(self, gold, predicted):
        gold = list(set(gold))
        predicted = list(set(predicted))
        self.gold.extend(gold)
        self.predicted.extend(predicted)
        self.correct.extend([pre_entity for pre_entity in predicted if pre_entity in gold])


class AnnotationLayerMetric:
    def __init__(
        self,
        eos_id: int,
        annotation_encoder_decoder: AnnotationEncoderDecoder,
    ):
        super().__init__()
        self.annotation_encoder_decoder = annotation_encoder_decoder
        self.eos_id = eos_id
        self.layer_metrics = {
            layer_name: LabeledAnnotationScore()
            for layer_name in self.annotation_encoder_decoder.layer_names
        }

        self.reset()

    def __call__(self, prediction, expected):
        bsz = prediction.size(0)
        self.total += bsz

        pred_eos_index = prediction.flip(dims=[1]).eq(self.eos_id).cumsum(dim=1).long()
        expected_eos_index = expected.flip(dims=[1]).eq(self.eos_id).cumsum(dim=1).long()

        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        expected_seq_len = (
            expected_eos_index.flip(dims=[1]).eq(expected_eos_index[:, -1:]).sum(dim=1)
        )  # bsz
        expected_seq_len = (expected_seq_len - 2).tolist()

        for i in range(bsz):
            # delete </s>
            # Note: I have absolutely no idea why this is not the same as:
            # expected[i, 1:expected_seq_len[i]]
            ts_tensor = expected[:, 1:][i, : expected_seq_len[i]]
            ps_tensor = prediction[:, 1:][i, : pred_seq_len[i]]
            if torch.equal(ts_tensor, ps_tensor):
                self.em += 1

            gold_annotations, gold_invalid = self.annotation_encoder_decoder.decode(
                expected[i].tolist()
            )
            predicted_annotations, invalid = self.annotation_encoder_decoder.decode(
                prediction[i].tolist()
            )
            for k, v in invalid.items():
                self.invalid[k] += v

            for layer_name, metric in self.layer_metrics.items():
                # remove duplicates from layer data
                gold_layer = set(gold_annotations[layer_name])
                pred_layer = set(predicted_annotations[layer_name])
                metric.update(gold_layer, pred_layer)

    def reset(self):
        for metric in self.layer_metrics.values():
            metric.reset()

        # total number of tuples
        self.total = 1e-13

        self.invalid = defaultdict(int)
        # this contains the number of examples where the full target sequence was predicted correctly
        self.em = 0

    def get_metric(self, reset=True):
        res = {}

        res["em"] = round(self.em / self.total, 4)

        for layer_name, metric in self.layer_metrics.items():
            overall_layer_info, layer_info = metric.result()
            res[layer_name] = layer_info
            res[layer_name + "/micro"] = overall_layer_info

        # if invalid contains a "total" key, use that to normalize, otherwise use the number of training examples
        invalid_total = self.invalid.pop("total", self.total)
        for k, v in self.invalid.items():
            res["invalid/" + k] = round(v / invalid_total, 4)
        res["invalid/all"] = round(sum(self.invalid.values()) / invalid_total, 4)

        if reset:
            self.reset()
        return res
