import functools
from collections import defaultdict
from typing import Callable, Dict, Optional

import pytest

from pie_modules.annotations import BinaryRelation, LabeledSpan
from pie_modules.utils.sequence_tagging.convert_annotation import (
    convert_span_annotations_to_tag_sequence,
    convert_tag_sequence_to_span_annotations,
    span_annotations_to_labeled_spans,
)

ENTITY_ANNOTATION_NAME = "entities"
RELATION_ANNOTATION_NAME = "relations"
SENTENCE_ANNOTATION_NAME = "sentences"

BIOUL_ENCODING_NAME = "BIOUL"
IOB2_ENCODING_NAME = "IOB2"
BOUL_ENCODING_NAME = "BOUL"

TEXT1 = "Jane lives in Berlin. this is no sentence about Karl\n"
TEXT2 = "Seattle is a rainy city. Jenny Durkan is the city's mayor.\n"

ENTITY_JANE_TEXT1 = LabeledSpan(start=0, end=4, label="person")
ENTITY_BERLIN_TEXT1 = LabeledSpan(start=14, end=20, label="city")
ENTITY_KARL_TEXT1 = LabeledSpan(start=48, end=52, label="person")
SENTENCE1_TEXT1 = LabeledSpan(start=0, end=21, label="sentence")
REL_JANE_LIVES_IN_BERLIN = BinaryRelation(
    head=ENTITY_JANE_TEXT1, tail=ENTITY_BERLIN_TEXT1, label="lives_in"
)

ENTITY_SEATTLE_TEXT2 = LabeledSpan(start=0, end=7, label="city")
ENTITY_JENNY_TEXT2 = LabeledSpan(start=25, end=37, label="person")
SENTENCE1_TEXT2 = LabeledSpan(start=0, end=24, label="sentence")
SENTENCE2_TEXT2 = LabeledSpan(start=25, end=58, label="sentence")
REL_JENNY_MAYOR_OF_SEATTLE = BinaryRelation(
    head=ENTITY_JENNY_TEXT2, tail=ENTITY_SEATTLE_TEXT2, label="mayor_of"
)


@pytest.fixture
def char_to_token_mappings():
    # This is created using tokenizer
    # TEXTS = [TEXT1, TEXT2]
    # encodings = tokenizer(TEXTS, return_offsets_mapping=True)
    # result is slightly different from shown below as command below might create some None values
    # result = [{char_idx: encodings[i].char_to_token(char_idx) for char_idx in range(len(text))} for i, text in enumerate(TEXTS)]
    return {
        TEXT1: {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            5: 2,
            6: 2,
            7: 2,
            8: 2,
            9: 2,
            11: 3,
            12: 3,
            14: 4,
            15: 4,
            16: 4,
            17: 4,
            18: 4,
            19: 4,
            20: 5,
            22: 6,
            23: 6,
            24: 6,
            25: 6,
            27: 7,
            28: 7,
            30: 8,
            31: 8,
            33: 9,
            34: 9,
            35: 9,
            36: 9,
            37: 9,
            38: 9,
            39: 9,
            40: 9,
            42: 10,
            43: 10,
            44: 10,
            45: 10,
            46: 10,
            48: 11,
            49: 11,
            50: 11,
            51: 11,
        },
        TEXT2: {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            8: 2,
            9: 2,
            11: 3,
            13: 4,
            14: 4,
            15: 4,
            16: 4,
            17: 4,
            19: 5,
            20: 5,
            21: 5,
            22: 5,
            23: 6,
            25: 7,
            26: 7,
            27: 7,
            28: 7,
            29: 7,
            31: 8,
            32: 8,
            33: 9,
            34: 9,
            35: 9,
            36: 10,
            38: 11,
            39: 11,
            41: 12,
            42: 12,
            43: 12,
            45: 13,
            46: 13,
            47: 13,
            48: 13,
            49: 14,
            50: 15,
            52: 16,
            53: 16,
            54: 16,
            55: 16,
            56: 16,
            57: 17,
        },
    }


@pytest.fixture
def special_tokens_masks():
    return {
        TEXT1: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        TEXT2: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }


@pytest.fixture
def true_tag_sequences():
    return {
        BIOUL_ENCODING_NAME: {
            TEXT1: [
                "O",
                "U-person",
                "O",
                "O",
                "U-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "U-person",
                "O",
            ],
            TEXT2: [
                "O",
                "U-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-person",
                "I-person",
                "I-person",
                "L-person",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
            ],
        },
        BOUL_ENCODING_NAME: {
            TEXT1: [
                "O",
                "U-person",
                "O",
                "O",
                "U-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "U-person",
                "O",
            ],
            TEXT2: [
                "O",
                "U-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-person",
                "O",
                "O",
                "L-person",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
            ],
        },
        IOB2_ENCODING_NAME: {
            TEXT1: [
                "O",
                "B-person",
                "O",
                "O",
                "B-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-person",
                "O",
            ],
            TEXT2: [
                "O",
                "B-city",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-person",
                "I-person",
                "I-person",
                "I-person",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
            ],
        },
    }


def _char_to_token_mapper(
    char_idx: int,
    char_to_token_mapping: Dict[int, int],
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
) -> Optional[int]:
    if char_start is not None and char_idx < char_start:
        # return negative number to encode out-ot-window
        return -1
    if char_end is not None and char_idx >= char_end:
        # return negative number to encode out-ot-window
        return -2
    return char_to_token_mapping.get(char_idx, None)


def get_char_to_token_mapper(
    char_to_token_mapping: Dict[int, int],
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
) -> Callable[[int], Optional[int]]:
    return functools.partial(
        _char_to_token_mapper,
        char_to_token_mapping=char_to_token_mapping,
        char_start=char_start,
        char_end=char_end,
    )


def test_convert_span_annotations_to_tag_sequence(
    char_to_token_mappings, special_tokens_masks, true_tag_sequences
):
    # Document contains three entities, one relation and one partition. Entities will be converted into the tag sequences
    # with the IOB2 encoding scheme.
    entities = [ENTITY_JANE_TEXT1, ENTITY_BERLIN_TEXT1, ENTITY_KARL_TEXT1]
    char_to_token_mapper = get_char_to_token_mapper(
        char_to_token_mapping=char_to_token_mappings[TEXT1],
    )
    tag_sequence = convert_span_annotations_to_tag_sequence(
        span_annotations=entities,
        base_sequence_length=len(special_tokens_masks[TEXT1]),
        char_to_token_mapper=char_to_token_mapper,
    )
    assert tag_sequence == true_tag_sequences[IOB2_ENCODING_NAME][TEXT1]

    # Document contains two entities, one relation and two partition. Entities will be converted into the tag sequences
    # with the IOB2 encoding scheme.
    entities = [ENTITY_SEATTLE_TEXT2, ENTITY_JENNY_TEXT2]
    char_to_token_mapper = get_char_to_token_mapper(
        char_to_token_mapping=char_to_token_mappings[TEXT2],
    )
    tag_sequence = convert_span_annotations_to_tag_sequence(
        span_annotations=entities,
        base_sequence_length=len(special_tokens_masks[TEXT2]),
        char_to_token_mapper=char_to_token_mapper,
    )
    assert tag_sequence == true_tag_sequences[IOB2_ENCODING_NAME][TEXT2]


def test_span_annotations_to_labeled_spans_with_partition(char_to_token_mappings):
    # Document contains two entities, one relation and two partition. First partition is used to create the labeled
    # spans. Only one entity will be converted into the labeled span since it is the only entity inside the first
    # partition.
    entities = [ENTITY_SEATTLE_TEXT2, ENTITY_JENNY_TEXT2]
    partition = SENTENCE1_TEXT2
    stats = defaultdict(lambda: defaultdict(int))
    char_to_token_mapper = get_char_to_token_mapper(
        char_to_token_mapping=char_to_token_mappings[TEXT2],
    )
    labeled_spans = span_annotations_to_labeled_spans(
        span_annotations=entities,
        char_to_token_mapper=char_to_token_mapper,
        partition=partition,
        statistics=stats,
    )
    assert labeled_spans == [("city", (1, 2))]


def test_span_annotations_to_labeled_spans_with_out_of_window_tokens():
    # Indices of a span cannot be negative. Since the char_to_token_mapper always returns a negative value, it will
    # result in a ValueError.
    entities = [ENTITY_SEATTLE_TEXT2, ENTITY_JENNY_TEXT2]
    with pytest.raises(
        ValueError, match="start index: -1 or end index: -1 is negative. This is deprecated."
    ):
        span_annotations_to_labeled_spans(
            span_annotations=entities,
            char_to_token_mapper=lambda x: -1,
        )


def test_span_annotations_to_labeled_spans_with_reversed_span_indices():
    # The start index of a span cannot be greater than the end index. char_to_token_mapper returns 1 for the start index
    # and 0 for the end index, this results in a ValueError.

    entities = [ENTITY_JANE_TEXT1]
    with pytest.raises(ValueError, match="start index: 1 is after end index: 0."):
        span_annotations_to_labeled_spans(
            span_annotations=entities,
            char_to_token_mapper=lambda x: 1 if x == 0 else 0,
        )


@pytest.mark.parametrize(
    "with_statistics",
    [True, False],
)
def test_span_annotations_to_labeled_spans_with_unaligned_spans(
    char_to_token_mappings, with_statistics
):
    # Document contains an entity. However, the document starts with a whitespace which results in an unaligned span.
    # Therefore, statistics collected for the method will contain one entity skipped due to incorrect alignment and no
    # added entity.
    entities = [LabeledSpan(start=0, end=5, label="person")]
    char_to_token_mapper = get_char_to_token_mapper(
        char_to_token_mapping=char_to_token_mappings[TEXT1],
    )
    if with_statistics:
        stats = defaultdict(lambda: defaultdict(int))
        labeled_spans = span_annotations_to_labeled_spans(
            span_annotations=entities, char_to_token_mapper=char_to_token_mapper, statistics=stats
        )
        assert stats["skipped_unaligned"]["person"] == 1
        # There was only one span which is skipped due to alignment issue, therefore no added spans in stats
        assert "added" not in stats
    else:
        labeled_spans = span_annotations_to_labeled_spans(
            span_annotations=entities,
            char_to_token_mapper=char_to_token_mapper,
        )
        assert labeled_spans == []


def test_convert_tag_sequence_to_span_annotations(true_tag_sequences):
    # Given tag sequence encoded with the IOB2 encoding scheme is converted into the spans.
    token_offset_mapping = [
        (0, 0),
        (0, 7),
        (8, 10),
        (11, 12),
        (13, 18),
        (19, 23),
        (23, 24),
        (25, 30),
        (31, 33),
        (33, 36),
        (36, 37),
        (38, 40),
        (41, 44),
        (45, 49),
        (49, 50),
        (0, 0),
    ]
    spans = convert_tag_sequence_to_span_annotations(
        true_tag_sequences[IOB2_ENCODING_NAME][TEXT2],
        token_offset_mapping=token_offset_mapping,
        offset=0,
    )
    spans = sorted(spans, key=lambda x: x.start)
    span = spans[0]
    assert span.start == ENTITY_SEATTLE_TEXT2.start
    assert span.end == ENTITY_SEATTLE_TEXT2.end
    assert span.label == ENTITY_SEATTLE_TEXT2.label

    span = spans[1]
    assert span.start == ENTITY_JENNY_TEXT2.start
    assert span.end == ENTITY_JENNY_TEXT2.end
    assert span.label == ENTITY_JENNY_TEXT2.label
