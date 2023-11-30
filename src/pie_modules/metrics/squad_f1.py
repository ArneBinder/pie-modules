import collections
import logging
import re
import string
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from pytorch_ie.core import DocumentMetric

from pie_modules.documents import ExtractiveQADocument

logger = logging.getLogger(__name__)


class SQuADF1(DocumentMetric):
    def __init__(
        self,
        no_answer_probability_threshold: float = 1.0,
        show_as_markdown: bool = False,
    ) -> None:
        super().__init__()
        self.no_answer_probability_threshold = no_answer_probability_threshold
        self.default_na_prob = 0.0
        self.show_as_markdown = show_as_markdown

    def reset(self):
        self.exact_scores = {}
        self.f1_scores = {}
        self.qas_id_to_has_answer = {}
        self.has_answer_qids = []
        self.no_answer_qids = []

    def _update(self, document: ExtractiveQADocument):
        gold_answers_for_questions = defaultdict(list)
        predicted_answers_for_questions = defaultdict(list)
        for ann in document.answers:
            gold_answers_for_questions[ann.question].append(ann)
        for ann in document.answers.predictions:
            predicted_answers_for_questions[ann.question].append(ann)

        for idx, question in enumerate(document.questions):
            if document.id is None:
                qas_id = f"text={document.text},question={question}"
            else:
                qas_id = document.id + f"_{idx}"

            self.qas_id_to_has_answer[qas_id] = bool(gold_answers_for_questions[question])
            if self.qas_id_to_has_answer[qas_id]:
                self.has_answer_qids.append(qas_id)
            else:
                self.no_answer_qids.append(qas_id)

            gold_answers = [
                str(answer)
                for answer in gold_answers_for_questions[question]
                if self.normalize_answer(str(answer))
            ]

            if not gold_answers:
                # For unanswerable questions, only correct answer is empty string
                gold_answers = [""]

            predicted_answers = predicted_answers_for_questions[question]
            if len(predicted_answers) == 0:
                prediction = ""
            else:
                prediction = str(max(predicted_answers, key=lambda ann: ann.score))

            self.exact_scores[qas_id] = max(
                self.compute_exact(a, prediction) for a in gold_answers
            )
            self.f1_scores[qas_id] = max(self.compute_f1(a, prediction) for a in gold_answers)

    def apply_no_ans_threshold(self, scores: Dict[str, float]) -> Dict[str, float]:
        new_scores = {}
        for qid, s in scores.items():
            no_prob = self.default_na_prob
            pred_na = no_prob > self.no_answer_probability_threshold
            if pred_na:
                new_scores[qid] = float(not self.qas_id_to_has_answer[qid])
            else:
                new_scores[qid] = s
        return new_scores

    def make_eval_dict(
        self, exact_scores: Dict[str, float], f1_scores: Dict[str, float], qid_list=None
    ) -> collections.OrderedDict:
        if not qid_list:
            total = len(exact_scores)
            return collections.OrderedDict(
                [
                    ("exact", 100.0 * sum(exact_scores.values()) / total),
                    ("f1", 100.0 * sum(f1_scores.values()) / total),
                    ("total", total),
                ]
            )
        else:
            total = len(qid_list)
            return collections.OrderedDict(
                [
                    ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                    ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                    ("total", total),
                ]
            )

    def merge_eval(
        self, main_eval: Dict[str, float], new_eval: Dict[str, float], prefix: str
    ) -> None:
        for k in new_eval:
            main_eval[f"{prefix}_{k}"] = new_eval[k]

    def _compute(self) -> Dict[str, Dict[str, float]]:
        exact_threshold = self.apply_no_ans_threshold(self.exact_scores)
        f1_threshold = self.apply_no_ans_threshold(self.f1_scores)

        evaluation = self.make_eval_dict(exact_threshold, f1_threshold)

        if self.has_answer_qids:
            has_ans_eval = self.make_eval_dict(
                exact_threshold, f1_threshold, qid_list=self.has_answer_qids
            )
            self.merge_eval(evaluation, has_ans_eval, "HasAns")

        if self.no_answer_qids:
            no_ans_eval = self.make_eval_dict(
                exact_threshold, f1_threshold, qid_list=self.no_answer_qids
            )
            self.merge_eval(evaluation, no_ans_eval, "NoAns")

        # return evaluation
        result = dict(evaluation)
        if self.show_as_markdown:
            logger.info(f"\n{pd.Series(result, name=self.current_split).round(3).to_markdown()}")
        return result

    def normalize_answer(self, s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(self, s: str) -> List[str]:
        if not s:
            return []
        return self.normalize_answer(s).split()

    def compute_exact(self, a_gold: str, a_pred: str) -> int:
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))

    def compute_f1(self, a_gold: str, a_pred: str) -> float:
        gold_toks = self.get_tokens(a_gold)
        pred_toks = self.get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
