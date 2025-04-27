import logging
from typing import List, Optional, Set, Union

from pie_core import Document, DocumentStatistic

from pie_modules.annotations import LabeledMultiSpan, Span

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
        labels: Optional[Union[List[str], str]] = None,
        label_attribute: str = "label",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer = layer
        self.labels = labels
        self.label_field = label_attribute

    def _collect(self, doc: Document) -> float:
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
