import dataclasses
from typing import Any, Dict, Optional, Tuple

from pie_core import AnnotationLayer, Document, annotation_field
from typing_extensions import TypeAlias

from pie_modules.annotations import (
    AbstractiveSummary,
    BinaryCorefRelation,
    BinaryRelation,
    ExtractiveAnswer,
    GenerativeAnswer,
    Label,
    LabeledMultiSpan,
    LabeledSpan,
    MultiLabel,
    Question,
    Span,
)


@dataclasses.dataclass
class WithMetadata:
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class WithTokens:
    tokens: Tuple[str, ...]


@dataclasses.dataclass
class WithText:
    text: str


@dataclasses.dataclass
class TextBasedDocument(WithMetadata, WithText, Document):
    pass


@dataclasses.dataclass
class TokenBasedDocument(WithMetadata, WithTokens, Document):
    def __post_init__(self) -> None:

        # When used in a dataset, the document gets serialized to json like structure which does not know tuples,
        # so they get converted to lists. This is a workaround to automatically convert the "tokens" back to tuples
        # when the document is created from a dataset.
        if isinstance(self.tokens, list):
            object.__setattr__(self, "tokens", tuple(self.tokens))
        elif not isinstance(self.tokens, tuple):
            raise ValueError("tokens must be a tuple.")

        # Call the default document construction code
        super().__post_init__()


# backwards compatibility
TextDocument: TypeAlias = TextBasedDocument


@dataclasses.dataclass
class DocumentWithLabel(Document):
    label: AnnotationLayer[Label] = annotation_field()


@dataclasses.dataclass
class DocumentWithMultiLabel(Document):
    label: AnnotationLayer[MultiLabel] = annotation_field()


@dataclasses.dataclass
class TextDocumentWithLabel(DocumentWithLabel, TextBasedDocument):
    pass


@dataclasses.dataclass
class TextDocumentWithMultiLabel(DocumentWithMultiLabel, TextBasedDocument):
    pass


@dataclasses.dataclass
class TextDocumentWithLabeledPartitions(TextBasedDocument):
    labeled_partitions: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithSentences(TextBasedDocument):
    sentences: AnnotationLayer[Span] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithSpans(TextBasedDocument):
    spans: AnnotationLayer[Span] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithLabeledSpans(TextBasedDocument):
    labeled_spans: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithLabeledSpansAndLabeledPartitions(
    TextDocumentWithLabeledSpans, TextDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TextDocumentWithLabeledSpansAndSentences(
    TextDocumentWithLabeledSpans, TextDocumentWithSentences
):
    pass


@dataclasses.dataclass
class TextDocumentWithLabeledSpansAndBinaryRelations(TextDocumentWithLabeledSpans):
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(target="labeled_spans")


@dataclasses.dataclass
class TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
    TextDocumentWithLabeledSpansAndLabeledPartitions,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledPartitions,
):
    pass


@dataclasses.dataclass
class TextDocumentWithSpansAndBinaryRelations(TextDocumentWithSpans):
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(target="spans")


@dataclasses.dataclass
class TextDocumentWithSpansAndLabeledPartitions(
    TextDocumentWithSpans, TextDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TextDocumentWithSpansBinaryRelationsAndLabeledPartitions(
    TextDocumentWithSpansAndLabeledPartitions,
    TextDocumentWithSpansAndBinaryRelations,
    TextDocumentWithLabeledPartitions,
):
    pass


@dataclasses.dataclass
class TextDocumentWithLabeledMultiSpans(TextBasedDocument):
    labeled_multi_spans: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TextDocumentWithLabeledMultiSpansAndLabeledPartitions(
    TextDocumentWithLabeledMultiSpans, TextDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TextDocumentWithLabeledMultiSpansAndBinaryRelations(TextDocumentWithLabeledMultiSpans):
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(
        target="labeled_multi_spans"
    )


@dataclasses.dataclass
class TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions(
    TextDocumentWithLabeledMultiSpansAndLabeledPartitions,
    TextDocumentWithLabeledMultiSpansAndBinaryRelations,
):
    pass


@dataclasses.dataclass
class TextDocumentWithQuestionsAndExtractiveAnswers(TextBasedDocument):
    """A text based PIE document with annotations for extractive question answering."""

    questions: AnnotationLayer[Question] = annotation_field()
    # Note: We define the target fields / layers of the answers layer in the following way. Any answer
    # targets the text field and (one entry of) the questions layer, so named_targets needs to contain
    # these *values*. See ExtractiveAnswer.TARGET_NAMES for the required *keys* for named_targets.
    answers: AnnotationLayer[ExtractiveAnswer] = annotation_field(
        named_targets={"base": "text", "questions": "questions"}
    )


@dataclasses.dataclass
class TokenDocumentWithQuestionsAndExtractiveAnswers(TokenBasedDocument):
    """A tokenized PIE document with annotations for extractive question answering."""

    questions: AnnotationLayer[Question] = annotation_field()
    # Note: We define the target fields / layers of the answers layer in the following way. Any answer
    # targets the text field and (one entry of) the questions layer, so named_targets needs to contain
    # these *values*. See ExtractiveAnswer.TARGET_NAMES for the required *keys* for named_targets.
    answers: AnnotationLayer[ExtractiveAnswer] = annotation_field(
        named_targets={"base": "tokens", "questions": "questions"}
    )


# backwards compatibility
ExtractiveQADocument = TextDocumentWithQuestionsAndExtractiveAnswers
TokenizedExtractiveQADocument = TokenDocumentWithQuestionsAndExtractiveAnswers


@dataclasses.dataclass
class TokenDocumentWithLabeledSpans(TokenBasedDocument):
    labeled_spans: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TokenDocumentWithLabeledPartitions(TokenBasedDocument):
    labeled_partitions: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansAndLabeledPartitions(
    TokenDocumentWithLabeledSpans, TokenDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansAndBinaryRelations(TokenDocumentWithLabeledSpans):
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(target="labeled_spans")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
    TokenDocumentWithLabeledSpansAndBinaryRelations, TokenDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TextDocumentWithQuestionsAndGenerativeAnswers(TextBasedDocument):
    """A text based PIE document with annotations for generative question answering."""

    questions: AnnotationLayer[Question] = annotation_field()
    generative_answers: AnnotationLayer[GenerativeAnswer] = annotation_field(target="questions")


@dataclasses.dataclass
class TokenDocumentWithQuestionsAndGenerativeAnswers(TokenBasedDocument):
    """A tokenized PIE document with annotations for generative question answering."""

    questions: AnnotationLayer[Question] = annotation_field()
    generative_answers: AnnotationLayer[GenerativeAnswer] = annotation_field(target="questions")


@dataclasses.dataclass
class TextDocumentWithAbstractiveSummary(TextBasedDocument):
    """A text based PIE document with annotations for abstractive summarization."""

    abstractive_summary: AnnotationLayer[AbstractiveSummary] = annotation_field()


@dataclasses.dataclass
class TokenDocumentWithAbstractiveSummary(TokenBasedDocument):
    """A tokenized PIE document with annotations for abstractive summarization."""

    abstractive_summary: AnnotationLayer[AbstractiveSummary] = annotation_field()


@dataclasses.dataclass
class TokenDocumentWithLabeledMultiSpans(TokenBasedDocument):
    labeled_multi_spans: AnnotationLayer[LabeledMultiSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TokenDocumentWithLabeledMultiSpansAndLabeledPartitions(
    TokenDocumentWithLabeledMultiSpans, TokenDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TokenDocumentWithLabeledMultiSpansAndBinaryRelations(TokenDocumentWithLabeledMultiSpans):
    binary_relations: AnnotationLayer[BinaryRelation] = annotation_field(
        target="labeled_multi_spans"
    )


@dataclasses.dataclass
class TokenDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions(
    TokenDocumentWithLabeledMultiSpansAndLabeledPartitions,
    TokenDocumentWithLabeledMultiSpansAndBinaryRelations,
):
    pass


@dataclasses.dataclass
class WithTextPair:
    text_pair: str


@dataclasses.dataclass
class WithLabeledSpansPair(WithTextPair):
    labeled_spans_pair: AnnotationLayer[LabeledSpan] = annotation_field(target="text_pair")


@dataclasses.dataclass
class WithLabeledPartitionsPair(WithTextPair):
    labeled_partitions_pair: AnnotationLayer[LabeledSpan] = annotation_field(target="text_pair")


@dataclasses.dataclass
class TextPairBasedDocument(TextBasedDocument, WithTextPair):
    pass


@dataclasses.dataclass
class TextPairDocumentWithLabeledPartitions(
    WithLabeledPartitionsPair, TextPairBasedDocument, TextDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TextPairDocumentWithLabeledSpans(
    WithLabeledSpansPair, TextPairBasedDocument, TextDocumentWithLabeledSpans
):
    pass


@dataclasses.dataclass
class TextPairDocumentWithLabeledSpansAndLabeledPartitions(
    TextPairDocumentWithLabeledPartitions,
    TextPairDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansAndLabeledPartitions,
):
    pass


@dataclasses.dataclass
class TextPairDocumentWithLabeledSpansAndBinaryCorefRelations(
    TextPairDocumentWithLabeledSpans, TextDocumentWithLabeledSpans
):
    binary_coref_relations: AnnotationLayer[BinaryCorefRelation] = annotation_field(
        targets=["labeled_spans", "labeled_spans_pair"]
    )


@dataclasses.dataclass
class TextPairDocumentWithLabeledSpansSimilarityRelationsAndLabeledPartitions(
    TextPairDocumentWithLabeledSpansAndLabeledPartitions,
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations,
):
    pass
