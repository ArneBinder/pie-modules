import pytest

from pie_modules.utils.sequence_tagging.ill_formed import (
    InvalidTagSequence,
    fix_bioul,
    fix_boul,
    fix_iob2,
    remove_bioul,
    remove_boul,
    remove_iob2,
)

BIOUL_ENCODING_NAME = "BIOUL"
BOUL_ENCODING_NAME = "BOUL"


def test_fix_boul():
    # Incorrect Cases:
    # 1. If there is no L tag
    boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "O",
        "O",
    ]
    new_boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "O",
        "L-background_claim",
    ]
    # This case is handled in process_O(line 187) of fix_boul
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence
    # 2. If there is U tag within a span
    boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "U-background_claim",
        "L-background_claim",
        "O",
    ]
    new_boul_sequence = ["O", "B-background_claim", "O", "O", "L-background_claim", "O"]

    # This case is handled in process_O(line 196) of fix_boul
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-background_claim",
        "O",
        "U-data",
        "L-background_claim",
        "O",
    ]
    new_boul_sequence = [
        "B-background_claim",
        "L-background_claim",
        "U-data",
        "U-background_claim",
        "O",
    ]
    # This case is handled in process_O(line 200) of fix_boul
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "U-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "U-data",
        "B-background_claim",
        "L-background_claim",
    ]
    # This case is handled in fix_boul (line 258)
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "U-background_claim",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-background_claim",
        "O",
        "L-background_claim",
    ]
    # This case is handled in fix_boul (line 249)
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "U-data",
        "O",
    ]
    new_boul_sequence = [
        "U-data",
        "O",
    ]
    # This case is handled in fix_boul (line 211)
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-data",
        "L-data",
        "U-background_claim",
    ]
    # This case is handled in _process_B(line 54) of fix_boul
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence
    # 3. If span starts with L

    boul_sequence = [
        "O",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "U-background_claim",
    ]
    # This case is handled in fix_boul (line 233)
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "L-background_claim",
        "O",
        "O",
    ]
    new_boul_sequence = [
        "U-background_claim",
        "O",
        "O",
    ]
    # This case is handled in fix_boul (line 221)
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "L-background_claim",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-background_claim",
        "O",
        "L-background_claim",
    ]
    # This case is handled first handled line 221 and then line 249 of fix_boul
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-background_claim",
        "L-background_claim",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-background_claim",
        "O",
        "O",
        "L-background_claim",
    ]
    # This case is handled in fix_boul (line 246)
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "L-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "U-data",
        "B-background_claim",
        "L-background_claim",
    ]
    # This case is handled first handled line 221 and then line 259 of fix_boul
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-data",
        "L-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-data",
        "L-data",
        "B-background_claim",
        "L-background_claim",
    ]
    # This case is handled in fix_boul (line 259)
    new_tag_sequence = fix_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence


def test_fix_bioul():
    # Incorrect Cases:
    # 1. If there is no L tag
    bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "I-background_claim",
        "O",  # "L-background_claim",
    ]
    new_bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "L-background_claim",
        "O",  # "L-background_claim",
    ]
    # This case is handled in process_I(line 94) of fix_bioul
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    # 2. If there is U tag within a span
    bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "U-background_claim",
        "L-background_claim",
        "O",
    ]
    new_bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "I-background_claim",
        "L-background_claim",
        "O",
    ]
    # This case is handled in process_I(line 102) of fix_bioul
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "U-data",
        "L-background_claim",
        "O",
    ]
    new_bioul_sequence = [
        "B-background_claim",
        "L-background_claim",
        "U-data",
        "U-background_claim",
        "O",
    ]
    # This case is handled in process_I(line 105) and then line 139 of fix_bioul
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    # 3. If there is O tag within a span
    bioul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "I-background_claim",
        "L-background_claim",
    ]

    new_bioul_sequence = [
        "O",
        "U-background_claim",
        "O",
        "B-background_claim",
        "L-background_claim",
    ]
    # This case is handled in process_I(line 94) of fix_bioul
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    # 4. If span starts with I
    bioul_sequence = [
        "O",
        "I-background_claim",
        "I-background_claim",
        "I-background_claim",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "O",
        "B-background_claim",
        "I-background_claim",
        "I-background_claim",
        "L-background_claim",
    ]
    # This case is handled in line 154 of fix_bioul
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "U-background_claim",
    ]
    # This case is handled in line 139 of fix_bioul
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    bioul_sequence = [
        "B-data",
        "I-data",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "B-data",
        "L-data",
        "U-background_claim",
    ]
    # This case is handled in line 54 of fix_bioul
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "U-data",
        "O",
    ]
    # This case is handled in line 115 of fix_bioul
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == bioul_sequence

    bioul_sequence = [
        "U-data",
        "O",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "U-data",
        "O",
        "U-background_claim",
    ]
    # This case is handled in line 139 of fix_bioul
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "U-background_claim",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "B-background_claim",
        "L-background_claim",
    ]
    # This case is handled in line 143 of fix_bioul
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "B-background_claim",
        "L-background_claim",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "L-background_claim",
    ]
    # This case is handled in line 148 of fix_bioul
    new_tag_sequence = fix_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence


def test_fix_iob2():
    iob2_sequence = [
        "B-background_claim",
        "B-background_claim",
        "I-background_claim",
    ]
    new_iob2_sequence = [
        "B-background_claim",
        "B-background_claim",
        "I-background_claim",
    ]
    # This case is handled in line 304 and 306 of fix_iob2
    new_tag_sequence = fix_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence

    iob2_sequence = [
        "B-background_claim",
        "B-data",
        "I-background_claim",
    ]
    new_iob2_sequence = [
        "B-background_claim",
        "B-data",
        "B-background_claim",
    ]
    # This case is handled in line 308 of fix_iob2
    new_tag_sequence = fix_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence

    iob2_sequence = [
        "B-background_claim",
        "B-data",
        "I-data",
    ]
    new_iob2_sequence = [
        "B-background_claim",
        "B-data",
        "I-data",
    ]
    # This case is handled in line 304 and 306 of fix_iob2
    new_tag_sequence = fix_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence

    iob2_sequence = ["B-background_claim", "I-background_claim", "I-data", "O"]
    new_iob2_sequence = [
        "B-background_claim",
        "I-background_claim",
        "B-data",
        "O",
    ]
    # This case is handled in line 308 of fix_iob2
    new_tag_sequence = fix_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence

    iob2_sequence = [
        "I-background_claim",
        "I-background_claim",
        "O",
    ]
    new_iob2_sequence = [
        "B-background_claim",
        "I-background_claim",
        "O",
    ]
    # This case is handled in line 320 of fix_iob2
    new_tag_sequence = fix_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence

    iob2_sequence = [
        "B-background_claim",
        "I-background_claim",
        "B-data",
    ]
    new_iob2_sequence = [
        "B-background_claim",
        "I-background_claim",
        "B-data",
    ]
    # This case is handled in line 306 and 304 of fix_iob2
    new_tag_sequence = fix_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence


def test_remove_bioul():
    bioul_sequence = [
        "B-data",
        "I-data",
        "L-data",
        "O",
        "I-data",
        "B-data",
        "L-data",
    ]
    new_bioul_sequence = [
        "B-data",
        "I-data",
        "L-data",
        "O",
        "O",
        "B-data",
        "L-data",
    ]
    # This case is handled in line 368 of remove_bioul
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "B-data",
        "I-data",
        "L-data",
        "O",
        "I-data",
        "U-data",
        "L-data",
    ]
    new_bioul_sequence = [
        "B-data",
        "I-data",
        "L-data",
        "O",
        "O",
        "U-data",
        "O",
    ]
    # This case is handled in line 368 of remove_bioul
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "I-background_claim",
        "O",  # "L-background_claim",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",  # "L-background_claim",
    ]
    # This case is handled in line 362 of remove_bioul
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    # 2. If there is U tag within a span
    bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "U-background_claim",
        "L-background_claim",
        "O",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 362 of remove_bioul
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "B-background_claim",
        "I-background_claim",
        "U-data",
        "L-background_claim",
        "O",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 362 of remove_bioul
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    # 3. If there is O tag within a span
    bioul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "I-background_claim",
        "L-background_claim",
    ]

    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 362 of remove_bioul
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    # 4. If span starts with I
    bioul_sequence = [
        "O",
        "I-background_claim",
        "I-background_claim",
        "I-background_claim",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 362 of remove_bioul
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence
    bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 362 of remove_bioul
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "B-data",
        "I-data",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 362 of remove_bioul
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence

    bioul_sequence = [
        "U-data",
        "O",
        "L-background_claim",
    ]
    new_bioul_sequence = [
        "U-data",
        "O",
        "O",
    ]
    # This case is handled in line 368 of remove_bioul
    new_tag_sequence = remove_bioul(bioul_sequence)
    assert new_tag_sequence == new_bioul_sequence


def test_remove_boul():
    boul_sequence = [
        "B-data",
        "O",
        "L-data",
        "O",
        "O",
        "U-data",
    ]
    new_boul_sequence = [
        "B-data",
        "O",
        "L-data",
        "O",
        "O",
        "U-data",
    ]
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-data",
        "O",
        "L-data",
        "O",
        "B-data",
        "U-data",
        "L-data",
    ]
    new_boul_sequence = [
        "B-data",
        "O",
        "L-data",
        "O",
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 402 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "O",
        "O",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 402 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    # 2. If there is U tag within a span
    boul_sequence = [
        "O",
        "B-background_claim",
        "O",
        "U-background_claim",
        "L-background_claim",
        "O",
    ]
    new_boul_sequence = ["O", "O", "O", "O", "O", "O"]
    # This case is handled in line 402 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-background_claim",
        "O",
        "U-data",
        "L-background_claim",
        "O",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 402 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "U-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "U-data",
        "O",
        "O",
    ]
    # This case is handled in line 408 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "U-background_claim",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "U-background_claim",
        "O",
        "O",
    ]
    # This case is handled in line 408 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "U-data",
        "O",
    ]
    new_boul_sequence = [
        "U-data",
        "O",
    ]

    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 402 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "O",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 408 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "L-background_claim",
        "O",
        "O",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 408 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "L-background_claim",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 408 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-background_claim",
        "L-background_claim",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-background_claim",
        "L-background_claim",
        "O",
        "O",
    ]
    # This case is handled in line 408 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "L-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "O",
        "O",
        "O",
    ]
    # This case is handled in line 408 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence

    boul_sequence = [
        "B-data",
        "L-data",
        "O",
        "L-background_claim",
    ]
    new_boul_sequence = [
        "B-data",
        "L-data",
        "O",
        "O",
    ]
    # This case is handled in line 408 of remove_boul
    new_tag_sequence = remove_boul(boul_sequence)
    assert new_tag_sequence == new_boul_sequence


def test_remove_iob2():
    # All ill formed cases are handled in line 439 of remove_iob2
    iob2_sequence = [
        "B-data",
        "B-data",
        "I-data",
        "O",
    ]
    new_iob2_sequence = [
        "B-data",
        "B-data",
        "I-data",
        "O",
    ]

    new_tag_sequence = remove_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence

    iob2_sequence = [
        "B-data",
        "I-data",
        "I-background_claim",
        "O",
    ]
    new_iob2_sequence = [
        "B-data",
        "I-data",
        "O",
        "O",
    ]
    new_tag_sequence = remove_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence

    iob2_sequence = [
        "I-data",
        "B-data",
        "I-data",
        "O",
    ]
    new_iob2_sequence = [
        "O",
        "B-data",
        "I-data",
        "O",
    ]
    new_tag_sequence = remove_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence

    iob2_sequence = [
        "B-background_claim",
        "B-background_claim",
        "I-background_claim",
    ]
    new_iob2_sequence = [
        "B-background_claim",
        "B-background_claim",
        "I-background_claim",
    ]
    new_tag_sequence = remove_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence

    iob2_sequence = [
        "B-background_claim",
        "B-data",
        "I-background_claim",
    ]
    new_iob2_sequence = [
        "B-background_claim",
        "B-data",
        "O",
    ]
    new_tag_sequence = remove_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence

    iob2_sequence = [
        "B-background_claim",
        "B-data",
        "I-data",
    ]
    new_iob2_sequence = [
        "B-background_claim",
        "B-data",
        "I-data",
    ]
    new_tag_sequence = remove_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence

    iob2_sequence = ["B-background_claim", "I-background_claim", "I-data", "O"]
    new_iob2_sequence = [
        "B-background_claim",
        "I-background_claim",
        "O",
        "O",
    ]
    new_tag_sequence = remove_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence

    iob2_sequence = [
        "I-background_claim",
        "I-background_claim",
        "O",
    ]
    new_iob2_sequence = [
        "O",
        "O",
        "O",
    ]
    new_tag_sequence = remove_iob2(iob2_sequence)
    assert new_tag_sequence == new_iob2_sequence


def test_invalid_tag_sequence():
    # Here we test if the different fix encoding methods can detect and raise the invalid tag sequence error.
    iob2_sequence = [
        "B-background_claim",
        "I-background_claim",
        "L-background_claim",
    ]
    with pytest.raises(InvalidTagSequence):
        fix_iob2(iob2_sequence)

    bioul_sequence = [
        "U-background_claim",
        "M-background_claim",
    ]
    with pytest.raises(InvalidTagSequence, match=f"{bioul_sequence}"):
        fix_bioul(bioul_sequence)

    boul_sequence = [
        "U-background_claim",
        "M-background_claim",
    ]
    with pytest.raises(InvalidTagSequence):
        fix_boul(boul_sequence)
