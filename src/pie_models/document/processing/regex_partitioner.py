from __future__ import annotations

import json
import logging
import re
import statistics
from typing import Any, Callable, Iterable, Iterator, Match, TypeVar

from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import EnterDatasetMixin, ExitDatasetMixin
from pytorch_ie.documents import TextBasedDocument

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=TextBasedDocument)


def create_regex_matcher(pattern):
    return re.compile(pattern).finditer


def strip_span(start: int, end: int, text: str) -> tuple[int, int]:
    """This method strips the leading and trailing whitespaces from the span.

    :param start: An integer value that represents the start index of the span.
    :param end: An integer value that represents the end index of the span.
    :param text: A string value that represents the text from which the span is extracted.
    """
    span_text = text[start:end]
    new_start = start + len(span_text) - len(span_text.lstrip())
    new_end = end - len(span_text) + len(span_text.rstrip())
    # if the span is empty, then create a span of length 0 at the start index
    if new_start >= new_end:
        new_start = start
        new_end = start
    return new_start, new_end


def _get_partitions_with_matcher(
    text: str,
    matcher_or_pattern: Callable[[str], Iterable[Match]] | str,
    label_group_id: int | None = None,  # = 1,
    label_whitelist: list[str] | None = None,
    skip_initial_partition: bool = False,  # = True
    default_partition_label: str = "partition",
    initial_partition_label: str | None = None,
    strip_whitespace: bool = False,
    verbose: bool = True,
) -> Iterator[LabeledSpan]:
    """This method yields LabeledSpans as partitions of the given text. matcher is used to search
    for a pattern in the text. If the pattern is found, it returns a Match object that contains
    matched groups. A partition is then created using a span in the matched groups. The span of a
    partition starts from the first match (inclusive) and ends at the next match (exclusive) or at
    the end of the text. A partition is labeled either using the default_partition_label or using
    the list of labels available in label_whitelist. It should be noted that none of the partitions
    overlap.

    :param text: A text that is to be partitioned
    :param matcher_or_pattern: A method or a string. In the former case, that method is used to
        find a pattern in the text and return an iterator yielding the Match objects, e.g.
        re.compile(PATTERN).finditer. In the latter, the string is used as a pattern to find the
        matches in the text.
    :param label_group_id: An integer value (default:None) to select the desired match group from
        the Match object. This match group is then used to create a label for the partition.
    :param label_whitelist: An optional list of labels (default:None) which are allowed to form a
        partition if label_group_id is not None. label_whitelist is the whitelist for the labels
        created using label_group_id. If label_whitelist is None, then all the labels created using
        label_group_id will form a partition.
    :param skip_initial_partition: A boolean value (default:False) that prevents the initial
        partition to be saved.
    :param default_partition_label: A string value (default:partition) to be used as the default
        label for the parts if no label_group_id for the match object is provided.
    :param initial_partition_label: A string value (default:None) to be used as a label for the
        initial partition. This is only used when skip_initial_partition is False. If it is None
        then default_partition_label is used as initial_partition_label.
    """
    if isinstance(matcher_or_pattern, str):
        matcher = create_regex_matcher(matcher_or_pattern)
    else:
        matcher = matcher_or_pattern
    if initial_partition_label is None:
        initial_partition_label = default_partition_label
    previous_start = previous_label = None
    if not skip_initial_partition:
        if label_whitelist is None or initial_partition_label in label_whitelist:
            previous_start = 0
            previous_label = initial_partition_label
    for match in matcher(text):
        if label_group_id is not None:
            start = match.start(label_group_id)
            end = match.end(label_group_id)
            label = text[start:end]
        else:
            label = default_partition_label
        if label_whitelist is None or label in label_whitelist:
            if previous_start is not None and previous_label is not None:
                start = previous_start
                end = match.start()
                if strip_whitespace:
                    start, end = strip_span(start=start, end=end, text=text)
                if end - start == 0:
                    if verbose:
                        logger.warning(
                            f"Found empty partition in text at [{previous_start}:{match.start()}] "
                            f"with potential label: '{previous_label}'. It will be skipped."
                        )
                else:
                    span = LabeledSpan(start=start, end=end, label=previous_label)
                    yield span

            previous_start = match.start()
            previous_label = label

    if previous_start is not None and previous_label is not None:
        start = previous_start
        end = len(text)
        if strip_whitespace:
            start, end = strip_span(start=start, end=end, text=text)
        if end - start == 0:
            if verbose:
                logger.warning(
                    f"Found empty partition in text at [{previous_start}:{len(text)}] with potential label: "
                    f"'{previous_label}'. It will be skipped."
                )
        else:
            span = LabeledSpan(start=start, end=end, label=previous_label)
            yield span


class RegexPartitioner(EnterDatasetMixin, ExitDatasetMixin):
    """RegexPartitioner partitions a document into multiple partitions using a regular expression.
    For more information, refer to get_partitions_with_matcher() method.

    :param pattern: A regular expression to search for in the text. It is also included at the beginning of each partition.
    :param collect_statistics: A boolean value (default:False) that allows to collect relevant statistics of the
                                document after partitioning. When this parameter is enabled, following stats are
                                collected:
                                1. partition_lengths: list of lengths of all partitions
                                2. num_partitions: list of number of partitions in each document
                                3. document_lengths: list of document lengths
                                show_statistics can be used to get statistical insight over these lists.
    :param partitioner_kwargs: keyword arguments for get_partitions_with_matcher() method
    """

    def __init__(
        self,
        pattern: str,
        collect_statistics: bool = False,
        partition_layer_name: str = "partitions",
        text_field_name: str = "text",
        **partitioner_kwargs,
    ):
        self.matcher = create_regex_matcher(pattern)
        self.partition_layer_name = partition_layer_name
        self.text_field_name = text_field_name
        self.collect_statistics = collect_statistics
        self.reset_statistics()
        self.partitioner_kwargs = partitioner_kwargs

    def reset_statistics(self):
        self._statistics: dict[str, Any] = {
            "partition_lengths": [],
            "num_partitions": [],
            "document_lengths": [],
        }

    def show_statistics(self, description: str | None = None):
        description = description or "Statistics"
        statistics_show = {
            key: {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "stddev": statistics.pstdev(values),
            }
            for key, values in self._statistics.items()
        }

        logger.info(f"{description}: \n{json.dumps(statistics_show, indent=2)}")

    def update_statistics(self, key: str, value: int | str | list):
        if self.collect_statistics:
            if isinstance(value, list):
                self._statistics[key] += value
            elif isinstance(value, str) or isinstance(value, int):
                self._statistics[key].append(value)
            else:
                raise TypeError(
                    f"type of given key [{type(key)}] or value [{type(value)}] is incorrect."
                )

    def __call__(self, document: D) -> D:
        partition_lengths = []
        text: str = getattr(document, self.text_field_name)
        for partition in _get_partitions_with_matcher(
            text=text, matcher_or_pattern=self.matcher, **self.partitioner_kwargs
        ):
            document[self.partition_layer_name].append(partition)
            partition_lengths.append(partition.end - partition.start)

        if self.collect_statistics:
            self.update_statistics("num_partitions", len(document[self.partition_layer_name]))
            self.update_statistics("partition_lengths", partition_lengths)
            self.update_statistics("document_lengths", len(text))

        return document

    def enter_dataset(self, dataset, name: str | None = None) -> None:
        if self.collect_statistics:
            self.reset_statistics()

    def exit_dataset(self, dataset, name: str | None = None) -> None:
        if self.collect_statistics:
            self.show_statistics(description=name)
