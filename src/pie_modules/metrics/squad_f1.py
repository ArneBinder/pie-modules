import collections
import logging
import re
import string
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from pytorch_ie.core import DocumentMetric

from pie_modules.documents import TextDocumentWithQuestionsAndExtractiveAnswers

logger = logging.getLogger(__name__)


def prefix_keys(d: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in d.items()}


class SQuADF1(DocumentMetric):
    """Computes the F1 score for extractive question answering in the style of the official SQuAD
    evaluation script. The metric is computed for each document-question pair and then averaged
    over all these pairs.

    The code is a simplified version of
    https://github.com/huggingface/transformers/blob/ac5893756bafcd745d93a442cf36f984545dbad8/src/transformers/data/metrics/squad_metrics.py.

    Args:
        show_as_markdown: If True, the metric result is printed as markdown table when calling `compute()`.
            Default: False.
    """

    def __init__(self, show_as_markdown: bool = False) -> None:
        super().__init__()

        self.show_as_markdown = show_as_markdown

    def reset(self):
        self.exact_scores = {}
        self.f1_scores = {}
        self.qas_id_to_has_answer = {}
        self.has_answer_qids = []
        self.no_answer_qids = []

    def _update(self, document: TextDocumentWithQuestionsAndExtractiveAnswers):
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

    def make_eval_dict(self, qid_list=None) -> Dict[str, float]:
        if qid_list:
            exact_scores = [self.exact_scores[k] for k in qid_list]
            f1_scores = [self.f1_scores[k] for k in qid_list]
        else:
            exact_scores = self.exact_scores.values()
            f1_scores = self.f1_scores.values()

        total = len(exact_scores)
        return {
            "exact": 100.0 * sum(exact_scores) / total,
            "f1": 100.0 * sum(f1_scores) / total,
            "total": total,
        }

    def _compute(self) -> Dict[str, float]:
        evaluation = self.make_eval_dict()

        if self.has_answer_qids:
            has_ans_eval = prefix_keys(
                self.make_eval_dict(qid_list=self.has_answer_qids), "HasAns"
            )
            evaluation.update(has_ans_eval)

        if self.no_answer_qids:
            no_ans_eval = prefix_keys(self.make_eval_dict(qid_list=self.no_answer_qids), "NoAns")
            evaluation.update(no_ans_eval)

        # return evaluation
        if self.show_as_markdown:
            logger.info(
                f"\n{pd.Series(evaluation, name=self.current_split).round(3).to_markdown()}"
            )
        return evaluation

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
        gold_tokens = self.get_tokens(a_gold)
        pred_tokens = self.get_tokens(a_pred)
        common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
        num_same = sum(common.values())
        if len(gold_tokens) == 0 or len(pred_tokens) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_tokens == pred_tokens)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
