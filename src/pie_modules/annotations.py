import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from pie_core import Annotation


def _post_init_single_label(self):
    if not isinstance(self.label, str):
        raise ValueError("label must be a single string.")

    if not isinstance(self.score, float):
        raise ValueError("score must be a single float.")


def _post_init_multi_label(self):
    if self.score is None:
        score = tuple([1.0] * len(self.label))
        object.__setattr__(self, "score", score)

    if not isinstance(self.label, tuple):
        object.__setattr__(self, "label", tuple(self.label))

    if not isinstance(self.score, tuple):
        object.__setattr__(self, "score", tuple(self.score))

    if len(self.label) != len(self.score):
        raise ValueError(
            f"Number of labels ({len(self.label)}) and scores ({len(self.score)}) must be equal."
        )


def _post_init_multi_span(self):
    if isinstance(self.slices, list):
        object.__setattr__(self, "slices", tuple(tuple(s) for s in self.slices))


def _post_init_arguments_and_roles(self):
    if len(self.arguments) != len(self.roles):
        raise ValueError(
            f"Number of arguments ({len(self.arguments)}) and roles ({len(self.roles)}) must be equal"
        )
    if not isinstance(self.arguments, tuple):
        object.__setattr__(self, "arguments", tuple(self.arguments))
    if not isinstance(self.roles, tuple):
        object.__setattr__(self, "roles", tuple(self.roles))


@dataclass(eq=True, frozen=True)
class Label(Annotation):
    label: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        _post_init_single_label(self)

    def resolve(self) -> Any:
        return self.label


@dataclass(eq=True, frozen=True)
class MultiLabel(Annotation):
    label: Tuple[str, ...]
    score: Optional[Tuple[float, ...]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        _post_init_multi_label(self)

    def resolve(self) -> Any:
        return self.label


@dataclass(eq=True, frozen=True)
class Span(Annotation):
    start: int
    end: int

    def __str__(self) -> str:
        if not self.is_attached:
            return super().__str__()
        return str(self.target[self.start : self.end])

    def resolve(self) -> Any:
        if self.is_attached:
            return self.target[self.start : self.end]
        else:
            raise ValueError(f"{self} is not attached to a target.")


@dataclass(eq=True, frozen=True)
class LabeledSpan(Span):
    label: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        _post_init_single_label(self)

    def resolve(self) -> Any:
        return self.label, super().resolve()


@dataclass(eq=True, frozen=True)
class MultiLabeledSpan(Span):
    label: Tuple[str, ...]
    score: Optional[Tuple[float, ...]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        _post_init_multi_label(self)

    def resolve(self) -> Any:
        return self.label, super().resolve()


@dataclass(eq=True, frozen=True)
class MultiSpan(Annotation):
    slices: Tuple[Tuple[int, int], ...]

    def __post_init__(self) -> None:
        _post_init_multi_span(self)

    def __str__(self) -> str:
        if not self.is_attached:
            return super().__str__()
        return str(tuple(self.target[start:end] for start, end in self.slices))

    def resolve(self) -> Any:
        if self.is_attached:
            return tuple(self.target[start:end] for start, end in self.slices)
        else:
            raise ValueError(f"{self} is not attached to a target.")


@dataclass(eq=True, frozen=True)
class LabeledMultiSpan(MultiSpan):
    label: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        _post_init_single_label(self)

    def resolve(self) -> Any:
        return self.label, super().resolve()


@dataclass(eq=True, frozen=True)
class BinaryRelation(Annotation):
    head: Annotation
    tail: Annotation
    label: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        _post_init_single_label(self)

    def resolve(self) -> Any:
        return self.label, (self.head.resolve(), self.tail.resolve())


@dataclass(eq=True, frozen=True)
class MultiLabeledBinaryRelation(Annotation):
    head: Annotation
    tail: Annotation
    label: Tuple[str, ...]
    score: Optional[Tuple[float, ...]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        _post_init_multi_label(self)

    def resolve(self) -> Any:
        return self.label, (self.head.resolve(), self.tail.resolve())


@dataclass(eq=True, frozen=True)
class NaryRelation(Annotation):
    arguments: Tuple[Annotation, ...]
    roles: Tuple[str, ...]
    label: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        _post_init_arguments_and_roles(self)
        _post_init_single_label(self)

    def resolve(self) -> Any:
        return (
            self.label,
            tuple((role, arg.resolve()) for arg, role in zip(self.arguments, self.roles)),
        )


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
