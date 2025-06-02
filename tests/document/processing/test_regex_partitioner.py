import dataclasses
import json
import logging
from typing import Tuple

import pytest
from pie_core import AnnotationLayer, annotation_field

from pie_modules.annotations import LabeledSpan
from pie_modules.document.processing import RegexPartitioner
from pie_modules.document.processing.regex_partitioner import (
    _get_partitions_with_matcher,
)
from pie_modules.documents import TextBasedDocument


@dataclasses.dataclass
class TextDocumentWithPartitions(TextBasedDocument):
    partitions: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


def have_overlap(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> bool:
    other_start_overlaps = start_end[0] <= other_start_end[0] < start_end[1]
    other_end_overlaps = start_end[0] < other_start_end[1] <= start_end[1]
    start_overlaps_other = other_start_end[0] <= start_end[0] < other_start_end[1]
    end_overlaps_other = other_start_end[0] < start_end[1] <= other_start_end[1]
    return other_start_overlaps or other_end_overlaps or start_overlaps_other or end_overlaps_other


def test_regex_partitioner():
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. RegexPartitioner
    # partitions the text based on the given pattern. After partitioning, there are be four partitions with same label.
    document = TextDocumentWithPartitions(text=TEXT1)
    new_document = regex_partitioner(document)

    partitions = new_document.partitions
    labels = [partition.label for partition in partitions]
    assert len(partitions) == 4
    assert labels == ["partition"] * len(partitions)
    assert str(partitions[0]) == "This is initial text."
    assert str(partitions[1]) == "<start>Jane lives in Berlin. this is no sentence about Karl."
    assert (
        str(partitions[2]) == "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
    )
    assert str(partitions[3]) == "<end>Karl enjoys sunny days in Berlin."


def test_regex_partitioner_with_statistics(caplog):
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    TEXT2 = "This is initial text.<start>Lily is mother of Harry.<end>Beth greets Emma."

    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
        label_group_id=0,
        label_whitelist=["<start>", "<middle>", "<end>"],
        skip_initial_partition=True,
        collect_statistics=True,
    )

    # The document contains a text separated by some markers like <start>, <middle> and <end>. After partitioning, there
    # are three partitions excluding initial part. Therefore, document length is not be equal to sum of partitions.
    document = TextDocumentWithPartitions(text=TEXT1)
    caplog.set_level(logging.INFO)
    caplog.clear()
    regex_partitioner.enter_dataset(None)
    new_document = regex_partitioner(document)
    regex_partitioner.exit_dataset(None)
    partitions = new_document.partitions
    assert len(partitions) == 3

    assert len(caplog.records) == 1
    log_description, log_json = caplog.records[0].message.split("\n", maxsplit=1)
    assert log_description.strip() == "Statistics:"
    assert json.loads(log_json) == {
        "partition_lengths": {
            "min": 38,
            "max": 66,
            "mean": 54.666666666666664,
            "stddev": 12.036980056845191,
        },
        "num_partitions": {"min": 3, "max": 3, "mean": 3, "stddev": 0.0},
        "document_lengths": {"min": 185, "max": 185, "mean": 185, "stddev": 0.0},
    }

    # The document contains a text separated by some markers like <start> and <end>. RegexPartitioner appends statistics
    # from each document, therefore statistics contains information from previous document as well. After partitioning,
    # there are two partitions excluding initial part. Therefore, the sum of document lengths is not be equal to sum of
    # partitions.
    document = TextDocumentWithPartitions(text=TEXT2)
    caplog.set_level(logging.INFO)
    caplog.clear()
    regex_partitioner.enter_dataset(None)
    new_document = regex_partitioner(document)
    regex_partitioner.exit_dataset(None)
    partitions = new_document.partitions
    assert len(partitions) == 2

    assert len(caplog.records) == 1
    log_description, log_json = caplog.records[0].message.split("\n", maxsplit=1)
    assert log_description.strip() == "Statistics:"
    assert json.loads(log_json) == {
        "partition_lengths": {"min": 22, "max": 31, "mean": 26.5, "stddev": 4.5},
        "num_partitions": {"min": 2, "max": 2, "mean": 2, "stddev": 0.0},
        "document_lengths": {"min": 74, "max": 74, "mean": 74, "stddev": 0.0},
    }

    with pytest.raises(
        TypeError,
        match=r"type of given key \[<class 'str'>\] or value \[<class 'float'>\] is incorrect.",
    ):
        regex_partitioner.update_statistics("num_partitions", 1.0)

    regex_partitioner.show_statistics()


@pytest.mark.parametrize("label_whitelist", [["<start>", "<middle>", "<end>"], [], None])
@pytest.mark.parametrize("skip_initial_partition", [True, False])
def test_regex_partitioner_without_label_group_id(label_whitelist, skip_initial_partition):
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
        label_whitelist=label_whitelist,
        skip_initial_partition=skip_initial_partition,
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. Since label_group_id is
    # None, the partitions (if any) will have same label.
    document = TextDocumentWithPartitions(text=TEXT1)
    new_document = regex_partitioner(document)
    partitions = new_document.partitions
    assert [partition.label for partition in partitions] == ["partition"] * len(partitions)
    if skip_initial_partition:
        if label_whitelist == ["<start>", "<middle>", "<end>"] or label_whitelist == []:
            # Since label_group_id is None, no label will be created using the matched pattern. Therefore, the default
            # partition label is used but since it is not in label_whitelist, no partition is created.
            assert len(partitions) == 0
        else:  # label_whitelist is None
            # since label_whitelist and label_group_id is None and skip_initial_partition is True, three partitions are
            # created with the same label
            assert len(partitions) == 3
            assert (
                str(partitions[0])
                == "<start>Jane lives in Berlin. this is no sentence about Karl."
            )
            assert (
                str(partitions[1])
                == "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[2]) == "<end>Karl enjoys sunny days in Berlin."
    else:  # skip_initial_partition is False
        if label_whitelist == ["<start>", "<middle>", "<end>"] or label_whitelist == []:
            # Since label_group_id is None, no label will be created using the matched pattern. Therefore, the default
            # partition label is used but since it is not in label_whitelist, no partition is created.
            assert len(partitions) == 0
        else:  # label_whitelist is None
            # since label_whitelist and label_group_id is None and skip_initial_partition is False, four partitions are
            # created with the same label.
            assert len(partitions) == 4
            assert str(partitions[0]) == "This is initial text."
            assert (
                str(partitions[1])
                == "<start>Jane lives in Berlin. this is no sentence about Karl."
            )
            assert (
                str(partitions[2])
                == "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[3]) == "<end>Karl enjoys sunny days in Berlin."


@pytest.mark.parametrize(
    "label_whitelist", [["partition", "<start>", "<end>"], ["<start>", "<end>"], [], None]
)
@pytest.mark.parametrize("skip_initial_partition", [True, False])
def test_regex_partitioner_with_label_group_id(label_whitelist, skip_initial_partition):
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    regex_partitioner = RegexPartitioner(
        pattern="(<start>|<middle>|<end>)",
        label_group_id=0,
        label_whitelist=label_whitelist,
        skip_initial_partition=skip_initial_partition,
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. Possible partitions can
    # be four including the initial partition.
    document = TextDocumentWithPartitions(text=TEXT1)
    new_document = regex_partitioner(document)
    partitions = new_document.partitions
    labels = [partition.label for partition in partitions]
    if skip_initial_partition:
        if label_whitelist == ["<start>", "<end>"] or label_whitelist == [
            "partition",
            "<start>",
            "<end>",
        ]:
            # Since skip_initial_partition is True, therefore even if initial_partition_label is in label_whitelist, it
            # will not be added as a partition.
            assert len(partitions) == 2
            assert labels == ["<start>", "<end>"]
            assert (
                str(partitions[0])
                == "<start>Jane lives in Berlin. this is no sentence about Karl.<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[1]) == "<end>Karl enjoys sunny days in Berlin."
        elif label_whitelist == []:
            # Even though labels are created using label_group_id, label_whitelist is empty. Therefore, no partition will
            # be created.
            assert len(partitions) == 0
        else:  # label_whitelist is None
            # Since label_whitelist is None, all the labels formed using label_group_id will create a partition.
            assert len(partitions) == 3
            assert labels == ["<start>", "<middle>", "<end>"]
            assert (
                str(partitions[0])
                == "<start>Jane lives in Berlin. this is no sentence about Karl."
            )
            assert (
                str(partitions[1])
                == "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[2]) == "<end>Karl enjoys sunny days in Berlin."
    else:  # skip_initial_partition is False
        if label_whitelist == ["<start>", "<end>"]:
            # Though skip_initial_partition is False it is not in label_whitelist, therefore not added as a partition.
            assert len(partitions) == 2
            assert labels == ["<start>", "<end>"]
            assert (
                str(partitions[0])
                == "<start>Jane lives in Berlin. this is no sentence about Karl.<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[1]) == "<end>Karl enjoys sunny days in Berlin."
        elif label_whitelist == ["partition", "<start>", "<end>"]:
            # Since initial partition label is in label_whitelist, therefore it will form a partition in the document.
            assert len(partitions) == 3
            assert labels == ["partition", "<start>", "<end>"]
            assert str(partitions[0]) == "This is initial text."
            assert (
                str(partitions[1])
                == "<start>Jane lives in Berlin. this is no sentence about Karl.<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[2]) == "<end>Karl enjoys sunny days in Berlin."
        elif label_whitelist == []:
            # Even though labels are created using label_group_id, label_whitelist is empty. Therefore, no partition will
            # be created.
            assert len(partitions) == 0
        else:  # label_whitelist is None
            # Since label_whitelist is None, all the labels formed using label_group_id will create a partition. In
            # addition to that the initial partition will also be added to the document.
            assert len(partitions) == 4
            assert labels == ["partition", "<start>", "<middle>", "<end>"]
            assert str(partitions[0]) == "This is initial text."
            assert (
                str(partitions[1])
                == "<start>Jane lives in Berlin. this is no sentence about Karl."
            )
            assert (
                str(partitions[2])
                == "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
            )
            assert str(partitions[3]) == "<end>Karl enjoys sunny days in Berlin."


@pytest.mark.parametrize("label_whitelist", [["partition"], [], None])
@pytest.mark.parametrize("skip_initial_partition", [True, False])
def test_regex_partitioner_with_no_match_found(skip_initial_partition, label_whitelist):
    TEXT2 = "This is initial text.<start>Lily is mother of Harry.<end>Beth greets Emma."
    regex_partitioner = RegexPartitioner(
        pattern="(<middle>)",
        label_group_id=0,
        label_whitelist=label_whitelist,
        skip_initial_partition=skip_initial_partition,
    )
    # The document contains a text separated by some markers like <start> and <end>. Only possible partition in the
    # document based on the given pattern is the initial partition.
    document = TextDocumentWithPartitions(text=TEXT2)
    new_document = regex_partitioner(document)

    partitions = new_document.partitions
    if skip_initial_partition:
        # No matter what the value of label_whitelist is, there will be no partition created, since the given pattern
        # is not in the document and skip_initial_partition is True.
        if label_whitelist == ["partition"]:
            assert len(partitions) == 0
        elif label_whitelist == []:
            assert len(partitions) == 0
        else:  # label_whitelist is None
            assert len(partitions) == 0
    else:
        if label_whitelist == ["partition"]:
            # Since initial_partition_label is contained in label_whitelist, the initial partition will be added to the
            # document.
            assert len(partitions) == 1
            assert str(partitions[0]) == TEXT2
            assert partitions[0].label == "partition"
        elif label_whitelist == []:
            # Even though skip_initial_partition is False, initial_partition_label is not contained in label_whitelist.
            # Therefore, the initial partition will not be added to the document.
            assert len(partitions) == 0
        else:  # label_whitelist is None
            # Since label_whitelist is None and skip_initial_partition is False, the initial partition will be added to
            # the document.
            assert len(partitions) == 1
            assert str(partitions[0]) == TEXT2
            assert partitions[0].label == "partition"


def test_get_partitions_with_matcher():
    TEXT1 = (
        "This is initial text.<start>Jane lives in Berlin. this is no sentence about Karl."
        "<middle>Seattle is a rainy city. Jenny Durkan is the city's mayor."
        "<end>Karl enjoys sunny days in Berlin."
    )
    # The document contains a text separated by some markers like <start>, <middle> and <end>. finditer method is used
    # which returns non overlapping match from the text. Therefore, none of the partition created should have overlapped
    # span and all of them should be instances of LabeledSpan.
    document = TextDocumentWithPartitions(text=TEXT1)
    partitions = []
    for partition in _get_partitions_with_matcher(
        text=document.text,
        matcher_or_pattern="(<start>|<middle>|<end>)",
        label_group_id=0,
        label_whitelist=["<start>", "<middle>", "<end>"],
    ):
        assert isinstance(partition, LabeledSpan)
        for p in partitions:
            assert not have_overlap((p.start, p.end), (partition.start, partition.end))
        partitions.append(partition)


@pytest.mark.parametrize(
    "strip_whitespace, verbose",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_regex_partitioner_with_strip_whitespace(strip_whitespace, verbose, caplog):
    TEXT1 = (
        "\nThis is initial text. Jane lives in Berlin. this is no sentence about Karl.\n"
        "Seattle is a rainy city. Jenny Durkan is the city's mayor.\n\n"
        "Karl enjoys sunny days in Berlin.\n"
    )
    regex_partitioner = RegexPartitioner(
        pattern="\n",
        strip_whitespace=strip_whitespace,
        verbose=verbose,
    )
    document = TextDocumentWithPartitions(text=TEXT1)
    new_document = regex_partitioner(document)

    partitions = new_document.partitions
    labels = [partition.label for partition in partitions]
    if strip_whitespace:
        assert len(partitions) == 3
        assert labels == ["partition"] * len(partitions)
        assert (
            str(partitions[0])
            == "This is initial text. Jane lives in Berlin. this is no sentence about Karl."
        )
        assert str(partitions[1]) == "Seattle is a rainy city. Jenny Durkan is the city's mayor."
        assert str(partitions[2]) == "Karl enjoys sunny days in Berlin."
        if verbose:
            assert len(caplog.messages) == 3
            assert caplog.messages[0] == (
                "Found empty partition in text at [0:0] with potential label: 'partition'. It will be skipped."
            )
            assert caplog.messages[1] == (
                "Found empty partition in text at [135:136] with potential label: 'partition'. It will be skipped."
            )
            assert caplog.messages[2] == (
                "Found empty partition in text at [170:171] with potential label: 'partition'. It will be skipped."
            )
    else:
        assert len(partitions) == 5
        assert labels == ["partition"] * len(partitions)
        assert (
            str(partitions[0])
            == "\nThis is initial text. Jane lives in Berlin. this is no sentence about Karl."
        )
        assert str(partitions[1]) == "\nSeattle is a rainy city. Jenny Durkan is the city's mayor."
        assert str(partitions[2]) == "\n"
        assert str(partitions[3]) == "\nKarl enjoys sunny days in Berlin."
        assert str(partitions[4]) == "\n"
        if verbose:
            assert len(caplog.messages) == 1
            assert (
                caplog.messages[0]
                == "Found empty partition in text at [0:0] with potential label: 'partition'. It will be skipped."
            )
