from typing import Any, Callable, Dict, Generic, Iterable, Optional, Sequence, TypeVar

from torchmetrics import Metric
from torchmetrics.text import ROUGEScore
from transformers import PreTrainedTokenizer

T = TypeVar("T")


class TextMetric(Metric, Generic[T]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        unbatch_func: Callable[[T], Iterable[str]],
        rouge_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.unbatch_func = unbatch_func
        self.tokenizer = tokenizer
        self.rouge_score = ROUGEScore(**(rouge_kwargs or {}))

    def update(self, predictions: T, targets: T) -> None:
        prediction_list = self.unbatch_func(predictions)
        target_list = self.unbatch_func(targets)
        for prediction_str, target_str in zip(prediction_list, target_list):
            self.rouge_score(prediction_str, target_str)

    def compute(self) -> Any:
        return self.rouge_score.compute()

    def reset(self) -> None:
        self.rouge_score.reset()
