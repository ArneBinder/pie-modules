import dataclasses
from typing import Optional, Tuple

# re-export all annotations from pytorch_ie to have a single entry point
from pytorch_ie.annotations import (
    BinaryRelation,
    Label,
    LabeledSpan,
    MultiLabel,
    MultiLabeledBinaryRelation,
    MultiLabeledSpan,
    NaryRelation,
    Span,
    _post_init_single_label,
)
from pytorch_ie.core import Annotation


def _post_init_multi_span(self):
    if isinstance(self.slices, list):
        object.__setattr__(self, "slices", tuple(tuple(s) for s in self.slices))


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
class MultiSpan(Annotation):
    slices: Tuple[Tuple[int, int], ...]

    def __post_init__(self) -> None:
        _post_init_multi_span(self)

    def __str__(self) -> str:
        if not self.is_attached:
            return super().__str__()
        return str(tuple(self.target[start:end] for start, end in self.slices))


@dataclasses.dataclass(eq=True, frozen=True)
class LabeledMultiSpan(MultiSpan):
    label: str
    score: float = dataclasses.field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        _post_init_single_label(self)
