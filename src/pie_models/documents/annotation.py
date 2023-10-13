import dataclasses
from typing import Optional

from pytorch_ie.annotations import Span
from pytorch_ie.core import Annotation


@dataclasses.dataclass(eq=True, frozen=True)
class Question(Annotation):
    """A question about a context."""

    text: str

    def __str__(self) -> str:
        return self.text


@dataclasses.dataclass(eq=True, frozen=True)
class ExtractiveAnswer(Span):
    """An answer to a question."""

    question: Question
    # The score of the answer. This is not considered when comparing two answers (e.g. prediction with gold).
    score: Optional[float] = dataclasses.field(default=None, compare=False)

    def __str__(self) -> str:
        if self.targets is None:
            return ""
        context = self.targets[0]
        return str(context[self.start : self.end])
