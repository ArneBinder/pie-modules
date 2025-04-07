import logging
from typing import Callable, Counter, DefaultDict, Iterable, List, Optional, Tuple

from pie_modules.annotations import LabeledSpan, Span

from .encoding import (
    tag_sequence_to_token_spans,
    token_spans_to_tag_sequence,
)

logger = logging.getLogger(__name__)


TypedStringSpan = Tuple[str, Tuple[int, int]]


def convert_tag_sequence_to_span_annotations(
    tag_sequence: List[str],
    token_offset_mapping: List[Tuple[int, int]],
    offset: int = 0,
    encoding: str = "IOB2",
    include_ill_formed: bool = True,
    classes_to_ignore: Optional[List[str]] = None,
) -> List[LabeledSpan]:
    """Given a tag sequence corresponding to a coding scheme (IOB2, BIOUL and BOUL), this method
    converts it into the labeled spans.

    # Parameters
        tag_sequence : `List[str]`, required.
           The tag sequence encoded in IOB2, BIOUL or BOUL, e.g. ["B-PER", "O", "L-PER"].
        token_offset_mapping: List[Tuple[int, int]], required.
            List of tuples containing start and end indices at character level for each token in the tag sequence.
        offset: int, optional.
            character offset to adjust the character spans when mapped from token level using token_offset_mapping.
            For instance, offset can be the start index of the partition containing the tag sequence.
        encoding: str, optional (default = "IOB2"),
            type of encoding scheme
        include_ill_formed: bool, optional (default = True),
            The tag sequence might be ill formed, so based on value of this parameter, such sequence is either fixed
            (if True) or removed (if False)
        classes_to_ignore : `List[str]`, optional (default = `None`).
           A list of string class labels `excluding` the IOB2 tag
           which should be ignored when extracting spans.

    # Returns
       `List[LabeledSpan]`
            List of character level labeled span created from the tag sequence
    """
    spans = tag_sequence_to_token_spans(
        tag_sequence=tag_sequence,
        coding_scheme=encoding,
        classes_to_ignore=classes_to_ignore,
        include_ill_formed=include_ill_formed,
    )
    return [
        LabeledSpan(
            token_offset_mapping[start][0] + offset,
            token_offset_mapping[end][1] + offset,
            label,
        )
        for label, (start, end) in spans
    ]


def span_annotations_to_labeled_spans(
    span_annotations: Iterable[LabeledSpan],
    char_to_token_mapper: Callable[[int], Optional[int]],
    partition: Optional[Span] = None,
    statistics: Optional[DefaultDict[str, Counter]] = None,
) -> List[Tuple[str, Tuple[int, int]]]:
    """Given a list of span annotations, a character position to token mapper (as obtained from
    batch_encoding.char_to_token), create a list of labeled spans. If a partition is provided, only
    the tokens within that span are considered.

    # Parameters
        span_annotations : `Iterable[LabeledSpan]`, required.
            List of span annotations
        char_to_token_mapper: `Callable[[int], Optional[int]]`, required.
            A method that maps character indices to token indices.
        partition: `Span`, optional (default = None)
            A span representing the partition
        statistics: `DefaultDict[str, Counter]`, optional (default = None)
            A dictionary that collects statistics related to the labeled spans.
            It collects the count of the span labels which are skipped due to alignment issues
            and the spans labels which are added successfully.
    # Returns
        `List[LabeledSpan]`
            List of token level labeled span mapped from span_annotations using char_to_token_mapper
    """
    offset = partition.start if partition is not None else 0
    labeled_spans = []
    for span_annotation in span_annotations:
        if partition is not None and (
            span_annotation.start < partition.start or span_annotation.end > partition.end
        ):
            continue

        start_idx = char_to_token_mapper(span_annotation.start - offset)
        end_idx = char_to_token_mapper(span_annotation.end - 1 - offset)
        if start_idx is None or end_idx is None:
            if statistics is not None:
                statistics["skipped_unaligned"][span_annotation.label] += 1
            else:
                logger.warning(
                    f"Entity annotation does not start or end with a token, it will be skipped: {span_annotation}"
                )
            continue

        # negative numbers encode out-of-window tokens
        if start_idx < 0 or end_idx < 0:
            raise ValueError(
                f"start index: {start_idx} or end index: {end_idx} is negative. This is deprecated."
            )

        if start_idx > end_idx:
            raise ValueError(f"start index: {start_idx} is after end index: {end_idx}.")

        labeled_spans.append((span_annotation.label, (start_idx, end_idx + 1)))

        if statistics is not None:
            statistics["added"][span_annotation.label] += 1

    return labeled_spans


def convert_span_annotations_to_tag_sequence(
    span_annotations: Iterable[LabeledSpan],
    base_sequence_length: int,
    char_to_token_mapper: Callable[[int], Optional[int]],
    partition: Optional[LabeledSpan] = None,
    statistics: Optional[DefaultDict[str, Counter]] = None,
    include_ill_formed: bool = True,
    encoding: str = "IOB2",
) -> List[str]:
    """Given a list of span annotations, a character position to token mapper (as obtained from
    batch_encoding.char_to_token) and a special tokens mask, create a sequence of tags with the
    length of the special tokens mask.

    # Parameters
        span_annotations : `Iterable[LabeledSpan]`, required.
            List of span annotations
        base_sequence_length: int, required.
            length of the tag sequence created from the span annotations
        char_to_token_mapper: `Callable[[int], Optional[int]]`, required.
            A method that maps character indices to token indices.
        partition: `Span`, optional (default = None)
            A span representing the partition
        statistics: `DefaultDict[str, Counter]`, optional (default = None)
            A dictionary that collects statistics related to the labeled spans.
            It collects the count of the span labels which are skipped due to alignment issues
            and the spans labels which are added successfully.
        include_ill_formed: bool, optional (default = True),
            The tag sequence might be ill formed, so based on value of this parameter, such sequence is either fixed
            (if True) or removed (if False)
        encoding: str, optional (default = "IOB2"),
            type of encoding scheme

    # Returns
       `List[str]`
            List of encoded string representing the tag sequence of length base_sequence_length
    """

    labeled_spans = span_annotations_to_labeled_spans(
        span_annotations=span_annotations,
        char_to_token_mapper=char_to_token_mapper,
        partition=partition,
        statistics=statistics,
    )

    tag_sequence = token_spans_to_tag_sequence(
        labeled_spans=labeled_spans,
        base_sequence_length=base_sequence_length,
        coding_scheme=encoding,
        include_ill_formed=include_ill_formed,
    )

    return tag_sequence
