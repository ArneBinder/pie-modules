import dataclasses
from typing import Any, Callable, List, Sequence

from torchmetrics import Metric
from torchmetrics.text import ROUGEScore
from transformers import PreTrainedTokenizer

from pie_modules.taskmodules.common import EncodingWithLabelsAndDecoderAttentionMask


class RougeMetric(Metric):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        unbatch_func: Callable[[Any], Sequence[EncodingWithLabelsAndDecoderAttentionMask]],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.unbatch_func = unbatch_func
        self.tokenizer = tokenizer
        self.rouge_score = ROUGEScore()

    def update(self, predictions, targets) -> None:
        prediction_list = self.unbatch_func(predictions)
        target_list = self.unbatch_func(targets)
        prediction: EncodingWithLabelsAndDecoderAttentionMask
        target: EncodingWithLabelsAndDecoderAttentionMask
        for prediction, target in zip(prediction_list, target_list):
            prediction_str = self.tokenizer.decode(prediction.labels, skip_special_tokens=True)
            target_str = self.tokenizer.decode(target.labels, skip_special_tokens=True)
            self.rouge_score(prediction_str, target_str)

    def compute(self) -> Any:
        return self.rouge_score.compute()

    def reset(self) -> None:
        self.rouge_score.reset()
