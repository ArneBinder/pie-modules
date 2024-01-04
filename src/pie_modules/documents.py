import dataclasses

from pytorch_ie import AnnotationLayer
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument

from pie_modules.annotations import (
    AbstractiveSummary,
    ExtractiveAnswer,
    GenerativeAnswer,
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
