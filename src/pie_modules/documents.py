import dataclasses

from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument

from pie_models.annotations import ExtractiveAnswer, Question


@dataclasses.dataclass
class ExtractiveQADocument(TextBasedDocument):
    """A PIE document with annotations for extractive question answering."""

    questions: AnnotationList[Question] = annotation_field()
    # Note: We define the target fields / layers of the answers layer in the following way. Any answer
    # targets the text field and (one entry of) the questions layer, so named_targets needs to contain
    # these keys. See ExtractiveAnswer.TARGET_NAMES for the required *values* for named_targets.
    answers: AnnotationList[ExtractiveAnswer] = annotation_field(
        named_targets={"text": "base", "questions": "questions"}
    )


@dataclasses.dataclass
class TokenizedExtractiveQADocument(TokenBasedDocument):
    """A tokenized PIE document with annotations for extractive question answering."""

    questions: AnnotationList[Question] = annotation_field()
    # Note: We define the target fields / layers of the answers layer in the following way. Any answer
    # targets the text field and (one entry of) the questions layer, so named_targets needs to contain
    # these keys. See ExtractiveAnswer.TARGET_NAMES for the required *values* for named_targets.
    answers: AnnotationList[ExtractiveAnswer] = annotation_field(
        named_targets={"tokens": "base", "questions": "questions"}
    )
