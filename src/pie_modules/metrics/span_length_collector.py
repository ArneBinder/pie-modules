import logging
from collections import defaultdict
from typing import Dict, List, Optional, Union

from pie_core import Document, DocumentStatistic

from pie_modules.annotations import Span

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
        labels: Optional[Union[List[str], str]] = None,
        label_attribute: str = "label",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer = layer
        if isinstance(labels, str) and labels != "INFERRED":
            raise ValueError("labels must be a list of strings or 'INFERRED'")
        self.labels = labels
        self.label_field = label_attribute

    def _collect(self, doc: Document) -> Union[List[int], Dict[str, List[int]]]:
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
