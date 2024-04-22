from __future__ import annotations

import logging
from typing import TypeVar

import nltk
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.documents import TextDocumentWithLabeledPartitions

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=TextDocumentWithLabeledPartitions)


class NltkSentenceSplitter:
    def __init__(
        self,
        partition_layer_name: str = "labeled_partitions",
        text_field_name: str = "text",
        sentencizer_url: str = "tokenizers/punkt/PY3/english.pickle",
    ):
        self.partition_layer_name = partition_layer_name
        self.text_field_name = text_field_name
        self.sentencizer = nltk.data.load(sentencizer_url)

    def __call__(self, document: D) -> None:
        partition_layer = document[self.partition_layer_name]
        if len(partition_layer) > 0:
            logger.warning(
                f"Layer {self.partition_layer_name} in document {document.id} is not empty. "
                f"Clearing it before adding new sentence partitions."
            )
            partition_layer.clear()

        text: str = getattr(document, self.text_field_name)
        sentence_spans = self.sentencizer.span_tokenize(text)
        sentences = [
            LabeledSpan(start=start, end=end, label="sentence") for start, end in sentence_spans
        ]
        partition_layer.extend(sentences)
