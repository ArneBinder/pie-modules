import dataclasses

from pytorch_ie.core import AnnotationLayer, AnnotationList, annotation_field

# re-export all documents from pytorch_ie to have a single entry point
from pytorch_ie.documents import (
    TextBasedDocument,
    TextDocumentWithLabel,
    TextDocumentWithLabeledMultiSpans,
    TextDocumentWithLabeledMultiSpansAndBinaryRelations,
    TextDocumentWithLabeledMultiSpansAndLabeledPartitions,
    TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
    TextDocumentWithLabeledPartitions,
    TextDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansAndLabeledPartitions,
    TextDocumentWithLabeledSpansAndSentences,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    TextDocumentWithMultiLabel,
    TextDocumentWithSentences,
    TextDocumentWithSpans,
    TextDocumentWithSpansAndBinaryRelations,
    TextDocumentWithSpansAndLabeledPartitions,
    TextDocumentWithSpansBinaryRelationsAndLabeledPartitions,
    TokenBasedDocument,
)

from pie_modules.annotations import (
    AbstractiveSummary,
    BinaryCorefRelation,
    BinaryRelation,
    ExtractiveAnswer,
    GenerativeAnswer,
    LabeledMultiSpan,
    LabeledSpan,
    Question,
)


@dataclasses.dataclass
class TextDocumentWithQuestionsAndExtractiveAnswers(TextBasedDocument):
    """A text based PIE document with annotations for extractive question answering."""

    questions: AnnotationList[Question] = annotation_field()
    # Note: We define the target fields / layers of the answers layer in the following way. Any answer
    # targets the text field and (one entry of) the questions layer, so named_targets needs to contain
    # these *values*. See ExtractiveAnswer.TARGET_NAMES for the required *keys* for named_targets.
    answers: AnnotationList[ExtractiveAnswer] = annotation_field(
        named_targets={"base": "text", "questions": "questions"}
    )


@dataclasses.dataclass
class TokenDocumentWithQuestionsAndExtractiveAnswers(TokenBasedDocument):
    """A tokenized PIE document with annotations for extractive question answering."""

    questions: AnnotationList[Question] = annotation_field()
    # Note: We define the target fields / layers of the answers layer in the following way. Any answer
    # targets the text field and (one entry of) the questions layer, so named_targets needs to contain
    # these *values*. See ExtractiveAnswer.TARGET_NAMES for the required *keys* for named_targets.
    answers: AnnotationList[ExtractiveAnswer] = annotation_field(
        named_targets={"base": "tokens", "questions": "questions"}
    )


# backwards compatibility
ExtractiveQADocument = TextDocumentWithQuestionsAndExtractiveAnswers
TokenizedExtractiveQADocument = TokenDocumentWithQuestionsAndExtractiveAnswers


@dataclasses.dataclass
class TokenDocumentWithLabeledSpans(TokenBasedDocument):
    labeled_spans: AnnotationList[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TokenDocumentWithLabeledPartitions(TokenBasedDocument):
    labeled_partitions: AnnotationList[LabeledSpan] = annotation_field(target="tokens")


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansAndLabeledPartitions(
    TokenDocumentWithLabeledSpans, TokenDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TokenDocumentWithLabeledSpansAndBinaryRelations(TokenDocumentWithLabeledSpans):
    binary_relations: AnnotationList[BinaryRelation] = annotation_field(target="labeled_spans")


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
