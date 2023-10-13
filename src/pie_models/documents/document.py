import dataclasses

from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument

from .annotation import ExtractiveAnswer, Question


@dataclasses.dataclass
class ExtractiveQADocument(TextBasedDocument):
    """A PIE document with annotations for extractive question answering."""

    questions: AnnotationList[Question] = annotation_field()
    answers: AnnotationList[ExtractiveAnswer] = annotation_field(targets=["text", "questions"])


@dataclasses.dataclass
class TokenizedExtractiveQADocument(TokenBasedDocument):
    """A tokenized PIE document with annotations for extractive question answering."""

    questions: AnnotationList[Question] = annotation_field()
    answers: AnnotationList[ExtractiveAnswer] = annotation_field(targets=["tokens", "questions"])
