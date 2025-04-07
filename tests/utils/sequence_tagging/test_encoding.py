import pytest

from pie_modules.utils.sequence_tagging import (
    tag_sequence_to_token_spans,
    token_spans_to_tag_sequence,
)
from pie_modules.utils.sequence_tagging.encoding import (
    _bioul_to_boul,
    _boul_to_bioul,
    labeled_spans_to_iob2,
)

BIOUL_ENCODING_NAME = "BIOUL"
IOB2_ENCODING_NAME = "IOB2"
BOUL_ENCODING_NAME = "BOUL"


@pytest.fixture
def special_tokens_masks():
    return [
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
    ]


@pytest.fixture
def true_tag_sequences():
    return {
        BIOUL_ENCODING_NAME: [
            [
                "O",
                "U-person",
                "O",
                "U-city",
                "O",
                "U-person",
                "O",
            ],
            [
                "O",
                "U-city",
                "O",
                "B-person",
                "I-person",
                "I-person",
                "L-person",
                "O",
            ],
        ],
        BOUL_ENCODING_NAME: [
            [
                "O",
                "U-person",
                "O",
                "U-city",
                "O",
                "U-person",
                "O",
            ],
            [
                "O",
                "U-city",
                "O",
                "B-person",
                "O",
                "O",
                "L-person",
                "O",
            ],
        ],
        IOB2_ENCODING_NAME: [
            [
                "O",
                "B-person",
                "O",
                "B-city",
                "O",
                "B-person",
                "O",
            ],
            [
                "O",
                "B-city",
                "O",
                "B-person",
                "I-person",
                "I-person",
                "I-person",
                "O",
            ],
        ],
    }


@pytest.mark.parametrize(
    "encoding",
    [IOB2_ENCODING_NAME, BIOUL_ENCODING_NAME, BOUL_ENCODING_NAME, None],
)
@pytest.mark.parametrize(
    "include_ill_formed",
    [True, False],
)
def test_spans_to_tag_sequence(
    encoding, special_tokens_masks, true_tag_sequences, include_ill_formed
):
    """Given labeled spans are converted into a tag sequence with the encoding scheme.

    The encoding scheme can only be either IOB2, BIOUL or BOUL.
    """
    labeled_spans = [
        [("person", (1, 2)), ("city", (3, 4)), ("person", (5, 6))],
        [("city", (1, 2)), ("person", (3, 7))],
    ]
    base_sequence_lengths = [len(special_tokens_masks[0]), len(special_tokens_masks[1])]
    if encoding is None:
        with pytest.raises(ValueError):
            labeled_span = labeled_spans[0]
            token_spans_to_tag_sequence(
                labeled_spans=labeled_span,
                base_sequence_length=base_sequence_lengths[0],
                coding_scheme="None",
            )
    else:
        labeled_span = labeled_spans[0]
        tag_sequence = token_spans_to_tag_sequence(
            labeled_spans=labeled_span,
            base_sequence_length=base_sequence_lengths[0],
            coding_scheme=encoding,
            include_ill_formed=include_ill_formed,
        )
        assert tag_sequence == true_tag_sequences[encoding][0]
        labeled_span = labeled_spans[1]
        tag_sequence = token_spans_to_tag_sequence(
            labeled_spans=labeled_span,
            base_sequence_length=base_sequence_lengths[1],
            coding_scheme=encoding,
            include_ill_formed=include_ill_formed,
        )
        assert tag_sequence == true_tag_sequences[encoding][1]


@pytest.mark.parametrize(
    "include_ill_formed",
    [True, False],
)
def test_labeled_spans_to_iob2(special_tokens_masks, true_tag_sequences, include_ill_formed):
    """Given labeled spans are converted into a tag sequence with the IOB2 encoding scheme.

    If the ill_formed sequences are allowed, they are kept in the tag sequence otherwise ignored.
    """
    labeled_spans = [
        # In this case if ill formed is included then we won't be able to fix it
        [("person", (1, 3)), ("city", (2, 4))],
        # In this case, if ill formed is included then we can still fix it
        [("city", (1, 7)), ("person", (3, 5))],
    ]
    tags = []
    for i, labeled_span in enumerate(labeled_spans):
        base_sequence_length = len(special_tokens_masks[i])
        tags.append(
            labeled_spans_to_iob2(
                labeled_spans=labeled_span,
                base_sequence_length=base_sequence_length,
                include_ill_formed=include_ill_formed,
            )
        )

    if include_ill_formed:
        assert tags[0] == ["O", "B-person", "B-city", "I-city", "O", "O", "O"]
        assert tags[1] == [
            "O",
            "B-city",
            "I-city",
            "B-person",
            "I-person",
            "I-city",
            "I-city",
            "O",
        ]
    else:
        assert tags[0] == ["O", "B-person", "I-person", "O", "O", "O", "O"]
        assert tags[1] == ["O", "B-city", "I-city", "I-city", "I-city", "I-city", "I-city", "O"]


@pytest.mark.parametrize(
    "encoding",
    [BIOUL_ENCODING_NAME, IOB2_ENCODING_NAME, BOUL_ENCODING_NAME, None],
)
def test_tag_sequence_to_span(encoding, true_tag_sequences):
    """Given tag sequence is converted into a span based on the type of encoding.

    If encoding is not of type IOB2, BIOUL or BOUL, a value error exception is generated. Ill
    formed tag sequences are fixed.
    """
    sequence_to_span = {
        BIOUL_ENCODING_NAME: [
            (
                ["O", "B-city", "O", "B-city", "U-city"],
                [
                    ("city", (1, 1)),
                    ("city", (3, 4)),
                ],
            ),
            (["B-city", "I-city", "I-person", "L-person"], [("city", (0, 1)), ("person", (2, 3))]),
            (["B-city", "I-city", "L-person"], [("city", (0, 1)), ("person", (2, 2))]),
            (["B-city", "I-city", "B-person"], [("city", (0, 1)), ("person", (2, 2))]),
            (["B-city", "I-city", "B-city"], [("city", (0, 2))]),
            (
                ["L-city", "I-city", "L-person"],
                [("city", (0, 0)), ("city", (1, 1)), ("person", (2, 2))],
            ),
            (["B-city", "I-city", "I-city"], [("city", (0, 2))]),
            (["B-city", "O", "U-city"], [("city", (0, 0)), ("city", (2, 2))]),
        ],
        BOUL_ENCODING_NAME: [
            (
                ["O", "B-city", "O", "B-city", "U-city"],
                [("city", (1, 4))],
            ),
            (["B-city", "O", "O", "L-person"], [("city", (0, 2)), ("person", (3, 3))]),
            (
                ["L-city", "O", "L-person"],
                [("city", (0, 0)), ("person", (1, 2))],
            ),
            (["B-city", "O", "O"], [("city", (0, 2))]),
            (
                ["B-city", "O", "U-city"],
                [
                    ("city", (0, 2)),
                ],
            ),
        ],
        IOB2_ENCODING_NAME: [
            (["O", "B-city", "O", "B-city", "I-city"], [("city", (1, 1)), ("city", (3, 4))]),
            (["B-city", "I-city", "I-person", "I-person"], [("city", (0, 1)), ("person", (2, 3))]),
            (["B-city", "I-city", "I-city", "B-person"], [("city", (0, 2)), ("person", (3, 3))]),
            (["B-city", "B-person"], [("city", (0, 0)), ("person", (1, 1))]),
        ],
    }
    if encoding is None:
        with pytest.raises(ValueError):
            tag_sequence = sequence_to_span[BIOUL_ENCODING_NAME][0][0]
            tag_sequence_to_token_spans(tag_sequence, coding_scheme=encoding)
    else:
        for tag_sequence, labeled_span in sequence_to_span[encoding]:
            computed_labeled_span = tag_sequence_to_token_spans(
                tag_sequence, coding_scheme=encoding
            )
            computed_labeled_span = sorted(computed_labeled_span, key=lambda x: x[1][0])
            assert computed_labeled_span == labeled_span


@pytest.mark.parametrize(
    "encoding",
    [BIOUL_ENCODING_NAME, IOB2_ENCODING_NAME, BOUL_ENCODING_NAME, None],
)
def test_tag_sequence_to_span_without_include_ill_formed(encoding, true_tag_sequences):
    """Given tag sequence is converted into a span based on the type of encoding.

    If the encoding is not of type IOB2, BIOUL or BOUL, a value error exception is generated. Ill
    formed tag sequence if any are removed.
    """
    sequence_to_span = {
        BIOUL_ENCODING_NAME: [
            (
                ["O", "B-city", "O", "B-city", "L-city"],
                [("city", (3, 4))],
            ),
            (["B-city", "L-city", "I-person", "L-person"], [("city", (0, 1))]),
            (
                ["B-city", "I-city", "L-city", "B-person", "U-person", "L-person"],
                [("city", (0, 2))],
            ),
            (
                ["B-city", "I-city", "L-city", "I-person", "U-person", "L-person"],
                [("city", (0, 2)), ("person", (4, 4))],
            ),
            (["B-city", "I-city", "B-city"], []),
        ],
        BOUL_ENCODING_NAME: [
            (
                ["O", "B-city", "O", "L-city", "U-city"],
                [("city", (1, 3)), ("city", (4, 4))],
            ),
            (["B-city", "O", "L-city", "L-person"], [("city", (0, 2))]),
            (
                ["B-city", "O", "L-city", "B-person", "U-city", "L-person"],
                [("city", (0, 2))],
            ),
            (
                ["B-city", "O", "L-city", "O", "U-city", "L-person"],
                [("city", (0, 2)), ("city", (4, 4))],
            ),
        ],
        IOB2_ENCODING_NAME: [
            (["O", "B-city", "O", "B-city", "I-city"], [("city", (1, 1)), ("city", (3, 4))]),
            (["B-city", "I-city", "I-person", "I-person"], [("city", (0, 1))]),
        ],
    }
    if encoding is None:
        with pytest.raises(ValueError):
            tag_sequence = sequence_to_span[BIOUL_ENCODING_NAME][0][0]
            tag_sequence_to_token_spans(
                tag_sequence, coding_scheme=encoding, include_ill_formed=False
            )
    else:
        for tag_sequence, labeled_span in sequence_to_span[encoding]:
            computed_labeled_span = tag_sequence_to_token_spans(
                tag_sequence, coding_scheme=encoding, include_ill_formed=False
            )
            computed_labeled_span = sorted(computed_labeled_span, key=lambda x: x[1][0])
            assert computed_labeled_span == labeled_span


def test_bioul_to_boul():
    """Given BIOUL sequence is converted to the BOUL sequence.

    All 'I' tags between 'B' and 'L' tag of the same label are converted to 'O' tag.
    """
    bioul_sequence = [
        "O",
        "B-background_claim",
        "I-background_claim",
        "I-background_claim",
        "L-background_claim",
        "O",
        "U-data",
        "O",
        "O",
    ]
    boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "O",
        "L-background_claim",
        "O",
        "U-data",
        "O",
        "O",
    ]
    new_tag_sequence = _bioul_to_boul(bioul_sequence)
    assert new_tag_sequence == boul_sequence


def test_boul_to_bioul():
    """Given BOUL sequence is converted to the BIOUL sequence.

    All 'O' tags between 'B' and 'L' tag of the same label are converted to 'I' tag of the same
    label.
    """
    boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "O",
        "L-background_claim",
        "O",
        "U-data",
        "O",
        "O",
    ]
    bioul_sequence = [
        "O",
        "B-background_claim",
        "I-background_claim",
        "I-background_claim",
        "L-background_claim",
        "O",
        "U-data",
        "O",
        "O",
    ]
    new_tag_sequence = _boul_to_bioul(boul_sequence)
    assert new_tag_sequence == bioul_sequence

    boul_sequence = [
        None,
        "U-data",
        "O",
        "O",
    ]
    bioul_sequence = [
        "U-data",
        "O",
        "O",
    ]
    new_tag_sequence = _boul_to_bioul(boul_sequence)
    assert new_tag_sequence == bioul_sequence
