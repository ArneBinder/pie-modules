from typing import TypeVar

from pytorch_ie.annotations import Span
from transformers import BatchEncoding

S = TypeVar("S", bound=Span)


class SpanNotAlignedWithTokenException(Exception):
    def __init__(self, span):
        self.span = span


def get_aligned_token_span(encoding: BatchEncoding, char_span: S) -> S:
    # find the start
    token_start = None
    token_end_before = None
    char_start = None
    for idx in range(char_span.start, char_span.end):
        token_start = encoding.char_to_token(idx)
        if token_start is not None:
            char_start = idx
            break

    if char_start is None:
        raise SpanNotAlignedWithTokenException(span=char_span)
    for idx in range(char_span.end - 1, char_start - 1, -1):
        token_end_before = encoding.char_to_token(idx)
        if token_end_before is not None:
            break

    if token_start is None or token_end_before is None:
        raise SpanNotAlignedWithTokenException(span=char_span)

    return char_span.copy(start=token_start, end=token_end_before + 1)
