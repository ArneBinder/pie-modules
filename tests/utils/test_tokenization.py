import pytest
from pytorch_ie.annotations import Span
from transformers import AutoTokenizer

from pie_modules.utils.tokenization import (
    SpanNotAlignedWithTokenException,
    get_aligned_token_span,
)


def test_get_aligned_token_span():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    text = "Hello, world!"
    encoding = tokenizer(text)
    tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids)
    assert tokens == ["[CLS]", "Hello", ",", "world", "!", "[SEP]"]

    # already aligned
    char_span = Span(0, 5)
    assert text[char_span.start : char_span.end] == "Hello"
    token_span = get_aligned_token_span(encoding=encoding, char_span=char_span)
    assert tokens[token_span.start : token_span.end] == ["Hello"]

    # end not aligned
    char_span = Span(5, 7)
    assert text[char_span.start : char_span.end] == ", "
    token_span = get_aligned_token_span(encoding=encoding, char_span=char_span)
    assert tokens[token_span.start : token_span.end] == [","]

    # start not aligned
    char_span = Span(6, 12)
    assert text[char_span.start : char_span.end] == " world"
    token_span = get_aligned_token_span(encoding=encoding, char_span=char_span)
    assert tokens[token_span.start : token_span.end] == ["world"]

    # start not aligned, end inside token
    char_span = Span(6, 8)
    assert text[char_span.start : char_span.end] == " w"
    token_span = get_aligned_token_span(encoding=encoding, char_span=char_span)
    assert tokens[token_span.start : token_span.end] == ["world"]

    # empty char span
    char_span = Span(2, 2)
    assert text[char_span.start : char_span.end] == ""
    with pytest.raises(SpanNotAlignedWithTokenException) as e:
        get_aligned_token_span(encoding=encoding, char_span=char_span)
    assert e.value.span == char_span

    # empty token span
    char_span = Span(6, 7)
    assert text[char_span.start : char_span.end] == " "
    with pytest.raises(SpanNotAlignedWithTokenException) as e:
        get_aligned_token_span(encoding=encoding, char_span=char_span)
    assert e.value.span == char_span
