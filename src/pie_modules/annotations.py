import dataclasses
from typing import Optional, Tuple

# re-export all annotations from pytorch_ie to have a single entry point
from pytorch_ie.annotations import (
    BinaryRelation,
    Label,
    LabeledMultiSpan,
    LabeledSpan,
    MultiLabel,
    MultiLabeledBinaryRelation,
    MultiLabeledSpan,
    MultiSpan,
    NaryRelation,
    Span,
    _post_init_single_label,
)
from pytorch_ie.core import Annotation


@dataclasses.dataclass(eq=True, frozen=True)
class AnnotationWithText(Annotation):
    text: str


@dataclasses.dataclass(eq=True, frozen=True)
class Question(AnnotationWithText):
    """A question about a context."""

    def __str__(self) -> str:
        return self.text


@dataclasses.dataclass(eq=True, frozen=True)
class ExtractiveAnswer(Span):
    """An answer to a question."""

    # this annotation has two target fields
    TARGET_NAMES = ("base", "questions")

    question: Question
    # The score of the answer. This is not considered when comparing two answers (e.g. prediction with gold).
    score: Optional[float] = dataclasses.field(default=None, compare=False)

    def __str__(self) -> str:
        if not self.is_attached:
            return ""
        # we assume that the first target is the text
        context = self.named_targets["base"]
        return str(context[self.start : self.end])


@dataclasses.dataclass(eq=True, frozen=True)
class AbstractiveSummary(AnnotationWithText):
    """An abstractive summary."""

    score: Optional[float] = dataclasses.field(default=None, compare=False)


@dataclasses.dataclass(eq=True, frozen=True)
class GenerativeAnswer(AnnotationWithText):
    """An answer to a question."""

    score: Optional[float] = dataclasses.field(default=None, compare=False)
    question: Optional[Question] = None


@dataclasses.dataclass(eq=True, frozen=True)
class BinaryCorefRelation(BinaryRelation):
    label: str = "coref"
