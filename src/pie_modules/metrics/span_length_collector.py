import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pytorch_ie.annotations import Span
from pytorch_ie.core import Document, DocumentStatistic
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument
from transformers import AutoTokenizer, PreTrainedTokenizer

from pie_modules.document.processing import tokenize_document
from pie_modules.utils import resolve_type

logger = logging.getLogger(__name__)


class SpanLengthCollector(DocumentStatistic):
    """Collects the lengths of Span annotations. If labels are provided, the lengths collected per
    label.

    If a tokenizer is provided, the span length is calculated in means of tokens, otherwise in
    means of characters.
    """

    DEFAULT_AGGREGATION_FUNCTIONS = ["len", "mean", "std", "min", "max"]

    def __init__(
        self,
        layer: str,
        tokenize: bool = False,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        tokenized_document_type: Optional[Union[str, Type[TokenBasedDocument]]] = None,
        labels: Optional[Union[List[str], str]] = None,
        label_attribute: str = "label",
        tokenize_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer = layer
        if isinstance(labels, str) and labels != "INFERRED":
            raise ValueError("labels must be a list of strings or 'INFERRED'")
        self.labels = labels
        self.label_field = label_attribute
        self.tokenize = tokenize
        if self.tokenize:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer must be provided to calculate the span length in means of tokens"
                )
            if isinstance(tokenizer, str):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.tokenizer = tokenizer
            if tokenized_document_type is None:
                raise ValueError(
                    "tokenized_document_type must be provided to calculate the span length in means of tokens"
                )
            self.tokenized_document_type = resolve_type(
                tokenized_document_type, expected_super_type=TokenBasedDocument
            )
            self.tokenize_kwargs = tokenize_kwargs or {}

    def _collect(self, doc: Document) -> Union[List[int], Dict[str, List[int]]]:
        docs: Union[List[Document], List[TokenBasedDocument]]
        if self.tokenize:
            if not isinstance(doc, TextBasedDocument):
                raise ValueError(
                    "doc must be a TextBasedDocument to calculate the span length in means of tokens"
                )
            docs = tokenize_document(
                doc,
                tokenizer=self.tokenizer,
                result_document_type=self.tokenized_document_type,
                **self.tokenize_kwargs,
            )
        else:
            docs = [doc]

        values: Dict[str, List[int]]
        if isinstance(self.labels, str):
            values = defaultdict(list)
        else:
            values = {label: [] for label in self.labels or ["ALL"]}
        for doc in docs:
            layer_obj = getattr(doc, self.layer)
            for span in layer_obj:
                if not isinstance(span, Span):
                    raise TypeError(
                        f"span length calculation is not yet supported for {type(span)}"
                    )
                length = span.end - span.start
                if self.labels is None:
                    label = "ALL"
                else:
                    label = getattr(span, self.label_field)
                values[label].append(length)

        return values if self.labels is not None else values["ALL"]
