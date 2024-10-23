from __future__ import annotations

import logging
from typing import TypeVar

from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.documents import TextDocumentWithLabeledPartitions

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=TextDocumentWithLabeledPartitions)


class NltkSentenceSplitter:
    """A document processor that adds sentence partitions to a TextDocumentWithLabeledPartitions document.
    It uses the NLTK Punkt tokenizer to split the text of the document into sentences. See
    https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.punkt.PunktSentenceTokenizer for more information.

    Args:
        partition_layer_name: The name of the partition layer to add the sentence partitions to. This layer
            must be an AnnotationLayer of LabeledSpan annotations.
        text_field_name: The name of the text field in the document to split into sentences.
        sentencizer_url: The URL to the NLTK Punkt tokenizer model.
        inplace: A boolean value that determines whether the sentence partitions are added to the input document
            or a new document is created.
    """

    def __init__(
        self,
        partition_layer_name: str = "labeled_partitions",
        text_field_name: str = "text",
        sentencizer_url: str = "tokenizers/punkt/PY3/english.pickle",
        inplace: bool = True,
    ):
        try:
            import nltk
        except ImportError:
            raise ImportError(
                "NLTK must be installed to use the NltkSentenceSplitter. "
                "You can install NLTK with `pip install nltk`."
            )

        self.partition_layer_name = partition_layer_name
        self.text_field_name = text_field_name
        self.inplace = inplace
        # download the NLTK Punkt tokenizer model
        nltk.download("punkt")
        self.sentencizer = nltk.data.load(sentencizer_url)

    def __call__(self, document: D) -> D:
        if not self.inplace:
            document = document.copy()

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

        return document


class FlairSegtokSentenceSplitter:
    """A document processor that adds sentence partitions to a TextDocumentWithLabeledPartitions document.
    It uses the Flair SegtokSentenceSplitter to split the text of the document into sentences. See
    https://github.com/flairNLP/flair/blob/master/flair/splitter.py for more information.

    Args:
        partition_layer_name: The name of the partition layer to add the sentence partitions to. This layer
            must be an AnnotationLayer of LabeledSpan annotations.
        text_field_name: The name of the text field in the document to split into sentences.
        inplace: A boolean value that determines whether the sentence partitions are added to the input document
            or a new document is created.
    """

    def __init__(
        self,
        partition_layer_name: str = "labeled_partitions",
        text_field_name: str = "text",
        inplace: bool = True,
    ):
        try:
            from flair.splitter import SegtokSentenceSplitter
        except ImportError:
            raise ImportError(
                "Flair must be installed to use the FlairSegtokSentenceSplitter. "
                "You can install Flair with `pip install flair`."
            )

        self.partition_layer_name = partition_layer_name
        self.text_field_name = text_field_name
        self.sentencizer = SegtokSentenceSplitter()
        self.inplace = inplace

    def __call__(self, document: D) -> D:
        if not self.inplace:
            document = document.copy()

        partition_layer = document[self.partition_layer_name]
        if len(partition_layer) > 0:
            logger.warning(
                f"Layer {self.partition_layer_name} in document {document.id} is not empty. "
                f"Clearing it before adding new sentence partitions."
            )
            partition_layer.clear()

        text: str = getattr(document, self.text_field_name)
        sentence_spans = self.sentencizer.split(text)
        sentences = [
            LabeledSpan(
                start=sentence.start_position,
                end=sentence.start_position + len(sentence.text),
                label="sentence",
            )
            for sentence in sentence_spans
        ]
        partition_layer.extend(sentences)

        return document
