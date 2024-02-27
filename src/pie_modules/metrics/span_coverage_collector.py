import logging
from typing import Any, Dict, List, Optional, Set, Type, Union

from pytorch_ie.annotations import Span
from pytorch_ie.core import Document, DocumentStatistic
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument
from transformers import AutoTokenizer, PreTrainedTokenizer

from pie_modules.annotations import LabeledMultiSpan
from pie_modules.document.processing import tokenize_document
from pie_modules.utils import resolve_type

logger = logging.getLogger(__name__)


class SpanCoverageCollector(DocumentStatistic):
    """Collects the coverage of Span annotations. It can handle overlapping spans.

    If a tokenizer is provided, the span coverage is calculated in means of tokens, otherwise in
    means of characters.

    Args:
        layer: The annotation layer of the document to calculate the span coverage for.
        tokenize: Whether to tokenize the document before calculating the span coverage. Default is False.
        tokenizer: The tokenizer to use for tokenization. Should be a PreTrainedTokenizer or a string
            representing the name of a pre-trained tokenizer, e.g. "bert-base-uncased". Required if
            tokenize is True.
        tokenized_document_type: The type of the tokenized document or a string that can be resolved
            to such a type. Required if tokenize is True.
        tokenize_kwargs: Additional keyword arguments for the tokenization.
        labels: If provided, only spans with these labels are considered.
        label_attribute: The attribute of the span to consider as label. Default is "label".
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
        self.labels = labels
        self.label_field = label_attribute
        self.tokenize = tokenize
        if self.tokenize:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer must be provided to calculate the span coverage in means of tokens"
                )
            if isinstance(tokenizer, str):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.tokenizer = tokenizer
            if tokenized_document_type is None:
                raise ValueError(
                    "tokenized_document_type must be provided to calculate the span coverage in means of tokens"
                )
            self.tokenized_document_type = resolve_type(
                tokenized_document_type, expected_super_type=TokenBasedDocument
            )
            self.tokenize_kwargs = tokenize_kwargs or {}

    def _collect(self, doc: Document) -> float:
        docs: Union[List[Document], List[TokenBasedDocument]]
        if self.tokenize:
            if not isinstance(doc, TextBasedDocument):
                raise ValueError(
                    "doc must be a TextBasedDocument to calculate the span coverage in means of tokens"
                )
            docs = tokenize_document(
                doc,
                tokenizer=self.tokenizer,
                result_document_type=self.tokenized_document_type,
                **self.tokenize_kwargs,
            )
            if len(docs) != 1:
                raise ValueError(
                    "tokenization of a single document must result in a single document to calculate the "
                    "span coverage correctly. Please check your tokenization settings, especially that "
                    "no windowing is applied because of max input length restrictions."
                )
            doc = docs[0]

        layer_obj = getattr(doc, self.layer)
        target = layer_obj.target
        covered_indices: Set[int] = set()
        for span in layer_obj:
            if self.labels is not None and getattr(span, self.label_field) not in self.labels:
                continue
            if isinstance(span, Span):
                covered_indices.update(range(span.start, span.end))
            elif isinstance(span, LabeledMultiSpan):
                for start, end in span.slices:
                    covered_indices.update(range(start, end))
            else:
                raise TypeError(f"span coverage calculation is not yet supported for {type(span)}")

        return len(covered_indices) / len(target)
