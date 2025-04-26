import logging
from typing import List, Optional, Tuple

from .ill_formed import fix_encoding, remove_encoding

logger = logging.getLogger(__name__)


TypedSpan = Tuple[int, Tuple[int, int]]
TypedStringSpan = Tuple[str, Tuple[int, int]]


def _iob2_to_bioul(tag_sequence: List[str]) -> List[str]:
    """Given a tag sequence encoded with IOB2 labels, recode to BIOUL.

    In the BIO or IBO2 scheme, I is a token inside a span, O is a token outside
    a span and B is the beginning of a span.

    In BIOUL scheme, I is a token inside a span, O is a token outside a span, B
    is the beginning of a span, U represents a unit span and L is the end of a
    span.

    This method expects that the tag sequence encoded with IOB2 scheme is free from
    the ill formed tag sequence.

    # Parameters

    tag_sequence : `List[str]`, required.
        The tag sequence encoded in IOB2, e.g. ["B-PER", "I-PER", "O"].

    # Returns

    bioul_sequence : `List[str]`
        The tag sequence encoded in BIOUL, e.g. ["B-PER", "L-PER", "O"].
    """

    def replace_label(full_label, new_label):
        # example: full_label = 'I-PER', new_label = 'U', returns 'U-PER'
        parts = list(full_label.partition("-"))
        parts[0] = new_label
        return "".join(parts)

    def pop_replace_append(in_stack, out_stack, new_label):
        # pop the last element from in_stack, replace the label, append
        # to out_stack
        tag = in_stack.pop()
        new_tag = replace_label(tag, new_label)
        out_stack.append(new_tag)

    def process_stack(stack, out_stack):
        # process a stack of labels, add them to out_stack
        if len(stack) == 1:
            # just a U token
            pop_replace_append(stack, out_stack, "U")
        else:
            # need to code as BIL
            recoded_stack = []
            pop_replace_append(stack, recoded_stack, "L")
            while len(stack) >= 2:
                pop_replace_append(stack, recoded_stack, "I")
            pop_replace_append(stack, recoded_stack, "B")
            recoded_stack.reverse()
            out_stack.extend(recoded_stack)

    # Process the tag_sequence one tag at a time, adding spans to a stack,
    # then recode them.
    bioul_sequence = []
    stack: List[str] = []

    for label in tag_sequence:
        if label == "O" and len(stack) == 0:
            bioul_sequence.append(label)
        elif label == "O" and len(stack) > 0:
            # need to process the entries on the stack plus this one
            process_stack(stack, bioul_sequence)
            bioul_sequence.append(label)
        elif label[0] == "I":
            this_type = label.partition("-")[2]
            prev_type = stack[-1].partition("-")[2]
            if this_type == prev_type:
                stack.append(label)
        else:  # label[0] == "B":
            stack.append(label)

    return bioul_sequence


def _bioul_to_boul(bioul_tags: List[str]) -> List[str]:
    """Given a tag sequence encoded with BIOUL labels, recode to BOUL. It replaces every occurrence
    of I-label with O. This method assumes that the tag sequence is not ill formed.

    # Parameters
    tag_sequence : `List[str]`, required.
        The tag sequence encoded in BIOUL, e.g. ["B-PER", "I-PER", "L-PER"].

    # Returns
    boul_sequence : `List[str]`
        The tag sequence encoded in BOUL, e.g. ["B-PER", "O", "L-PER"].
    """
    return ["O" if tag.startswith("I-") else tag for tag in bioul_tags]


def _iob2_to_boul(tag_sequence: List[str]) -> List[str]:
    """Given a tag sequence encoded with IOB2 labels, recode to BOUL. The tag sequence is first
    converted to the BIOUL scheme and then BIOUL is recoded to BOUL. This method assumes that the
    tag sequence is not ill formed.

    # Parameters
    tag_sequence : `List[str]`, required.
        The tag sequence encoded in IOB2, e.g. ["B-PER", "I-PER", "I-PER"].

    # Returns
    boul_sequence : `List[str]`
        The tag sequence encoded in BOUL, e.g. ["B-PER", "O", "L-PER"].
    """
    bioul_tags = _iob2_to_bioul(tag_sequence=tag_sequence)
    return _bioul_to_boul(bioul_tags)


def labeled_spans_to_iob2(
    labeled_spans: List[TypedStringSpan],
    base_sequence_length: int,
    include_ill_formed: bool = False,
) -> List[str]:
    """This method converts a list of spans given as (label, (start_idx, end_idx)) to the tag
    sequence with IOB2 encoding scheme.

    # Parameters
    labeled_spans : `List[TypedStringSpan]`, required.
        A list of tuples containing spans and their label, e.g. [(person,(1,2)]
    base_sequence_length: int, required.
        The length of base sequence
    include_ill_formed: bool, optional (False by default)
        IOB2 tag sequence created by spans might be ill formed. If this parameter is True then we keep such ill formed
        sequence otherwise we exclude them from resulting the tag sequence.

    # Returns
    tags : `List[str]`
        The tag sequence encoded in IOB2, e.g. ["O", "B-PER", "I-PER"].
    """
    # create IOB2 encoding
    tags = ["O"] * base_sequence_length
    labeled_spans = sorted(labeled_spans, key=lambda span_annot: span_annot[1][0])
    for i, (label, (start, end)) in enumerate(labeled_spans):
        previous_tags = tags[start:end]
        if previous_tags != ["O"] * len(previous_tags):
            # raise ValueError(f"tags already set [{previous_tags}], i.e. there is an annotation overlap")
            if not include_ill_formed:
                continue

        # create IOB2 encoding
        tags[start] = f"B-{label}"
        tags[start + 1 : end] = [f"I-{label}"] * (len(previous_tags) - 1)

    return tags


def _boul_to_bioul(tag_sequence: List[str]) -> List[str]:
    """Given a tag sequence encoded with BOUL labels, recode to BIOUL. This method assumes that the
    tag sequence is not ill formed.

    # Parameters
    tag_sequence : `List[str]`, required.
        The tag sequence encoded in BOUL, e.g. ["B-PER", "O", "L-PER"].

    # Returns
    bioul_tags : `List[str]`
        The tag sequence encoded in BIOUL, e.g. ["B-PER", "I-PER", "L-PER"].
    """
    bioul_tags = []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label is None:
            index += 1
            continue
        elif label[0] == "B":
            bioul_tags.append(label)
            tag_type = label[2:]
            index += 1
            label = tag_sequence[index]
            while label[0] != "L" and index < len(tag_sequence):
                bioul_tags.append(f"I-{tag_type}")
                index += 1
                label = tag_sequence[index]
            bioul_tags.append(label)  # append L
        else:
            bioul_tags.append(label)  # append O
        index += 1
    return bioul_tags


def iob2_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None,
) -> List[TypedStringSpan]:
    """Given a sequence corresponding to BIO or IOB2 tags, extracts spans. Spans are inclusive and
    can be of zero length, representing a single word span.

    # Parameters
    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the IOB2 tag
        which should be ignored when extracting the spans.
    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end_inclusive)).
        Note that the label `does not` contain any IOB2 tag prefixes.
    """
    spans = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == "B":
            start = index
            current_span_label = label.partition("-")[2]
            index += 1
            if index < len(tag_sequence):
                label = tag_sequence[index]
            else:
                # if B is last tag in the sequence
                spans.append((current_span_label, (start, start)))
                continue
            if label[0] == "B":
                # if Ba Bb or Ba Ba
                spans.append((current_span_label, (start, start)))
                continue
            while label[0] == "I" and index < len(tag_sequence):
                # loop can end if it encounters another B or O or end of tag_sequence
                index += 1
                if index < len(tag_sequence):
                    label = tag_sequence[index]
            index -= 1
            end = index
            spans.append((current_span_label, (start, end)))
        index += 1
    return [span for span in spans if span[0] not in classes_to_ignore]


def bioul_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None,
) -> List[TypedStringSpan]:
    """Given a sequence corresponding to BIOUL tags, extracts spans. Spans are inclusive and can be
    of zero length, representing a single word span.

    # Parameters
    tag_sequence : `List[str]`, required.
        The tag sequence encoded in BIOUL, e.g. ["B-PER", "L-PER", "O"].
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the IOB2 tag
        which should be ignored when extracting the spans.

    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end_inclusive)).
    """

    spans = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == "U":
            spans.append((label.partition("-")[2], (index, index)))
        elif label[0] == "B":
            start = index
            end = index
            current_span_label = label.partition("-")[2]
            while label[0] != "L" and index < len(tag_sequence):
                index += 1
                label = tag_sequence[index]
                if label[0] == "L":
                    end = index
            spans.append((current_span_label, (start, end)))
        index += 1
    return [span for span in spans if span[0] not in classes_to_ignore]


def boul_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None,
) -> List[TypedStringSpan]:
    """Given a sequence corresponding to BOUL tags, extracts spans. It converts BOUL tags to BIOUL
    tags and then BIOUL tags are converted to spans.

    # Parameters
    tag_sequence : `List[str]`, required.
        The tag sequence encoded in BOUL, e.g. ["B-PER", "O", "L-PER"].
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the IOB2 tag
        which should be ignored when extracting the spans.

    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end_inclusive)).
    """
    bioul_tags = _boul_to_bioul(tag_sequence=tag_sequence)
    return bioul_tags_to_spans(
        tag_sequence=bioul_tags,
        classes_to_ignore=classes_to_ignore,
    )


def token_spans_to_tag_sequence(
    labeled_spans: List[TypedStringSpan],
    base_sequence_length: int,
    coding_scheme: str = "IOB2",
    include_ill_formed: bool = True,
) -> List[str]:
    """This method converts a list of spans given as (label, (start_idx, end_idx)) to a tag
    sequence. First, spans are converted to IOB2 tag sequence and then the ill formed sequence are
    either fixed or removed. Finally, the IOB2 tag sequence is recoded to the required coding
    scheme.

    # Parameters
    labeled_spans : `List[TypedStringSpan]`, required.
        A list of tuples containing spans and their label, e.g. [(person,(1,2)]
    base_sequence_length: int, required.
        The length of base sequence
    coding_scheme: str, optional (default = "IOB2"),
        type of encoding scheme
    include_ill_formed: bool, optional (default = True),
        The tag sequence might be ill formed, so based on value of this parameter, such sequence is either fixed
        (if True) or removed (if False)

    # Returns
    tags : `List[str]`
        The tag sequence encoded in IOB2, e.g. ["O", "B-PER", "I-PER"].
    """
    tags = labeled_spans_to_iob2(
        labeled_spans=labeled_spans,
        base_sequence_length=base_sequence_length,
        include_ill_formed=include_ill_formed,
    )
    if include_ill_formed:
        tags = fix_encoding(tags, "IOB2")
    else:
        tags = remove_encoding(tags, "IOB2")

    # Recode the labels if necessary.
    if coding_scheme == "BIOUL":
        coded_tags = _iob2_to_bioul(tags) if tags is not None else None
    elif coding_scheme == "BOUL":
        coded_tags = _iob2_to_boul(tags) if tags is not None else None
    elif coding_scheme == "IOB2":
        coded_tags = tags
    else:
        raise ValueError(f"Unknown Coding scheme {coding_scheme}.")

    return coded_tags


def tag_sequence_to_token_spans(
    tag_sequence: List[str],
    coding_scheme: str = "IOB2",
    classes_to_ignore: Optional[List[str]] = None,
    include_ill_formed: bool = True,
) -> List[TypedStringSpan]:
    """Given a sequence corresponding to a coding scheme (IOB2, BIOUL and BOUL), this method
    converts it into the token spans. It first fixes or removes the ill formed tag sequence (if
    any) and then converts the tag sequence to the spans.

    # Parameters
       tag_sequence : `List[str]`, required.
           The tag sequence encoded in IOB2, BIOUL or BOUL, e.g. ["B-PER", "O", "L-PER"].
        coding_scheme: str, optional (default = "IOB2"),
            type of encoding scheme
        classes_to_ignore : `List[str]`, optional (default = `None`).
           A list of string class labels `excluding` the IOB2 tag
           which should be ignored when extracting spans.
        include_ill_formed: bool, optional (default = True),
            The tag sequence might be ill formed, so based on value of this parameter, such sequence is either fixed
            (if True) or removed (if False)
       # Returns
       spans : `List[TypedStringSpan]`
           The typed, extracted spans from the sequence, in the format (label, (span_start, span_end_inclusive)).
    """

    if include_ill_formed:
        new_tag_sequence = fix_encoding(tag_sequence, coding_scheme)
    else:
        new_tag_sequence = remove_encoding(tag_sequence, coding_scheme)
    if coding_scheme == "BIOUL":
        labeled_spans = bioul_tags_to_spans(
            new_tag_sequence,
            classes_to_ignore=classes_to_ignore,
        )
    elif coding_scheme == "BOUL":
        labeled_spans = boul_tags_to_spans(
            new_tag_sequence,
            classes_to_ignore=classes_to_ignore,
        )
    else:  # coding_scheme = "IOB2":
        labeled_spans = iob2_tags_to_spans(
            tag_sequence,
            classes_to_ignore=classes_to_ignore,
        )

    return labeled_spans
