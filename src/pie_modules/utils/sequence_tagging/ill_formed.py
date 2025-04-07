from typing import List


class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return " ".join(self.tag_sequence)


def _update_previous_label(tag_sequence, current_tag_type):
    """When the current tag type in a span is different from the next span tag type, we convert
    using the following rule :

    BIOUL: update Ba to Ua or Ia to La
    BOUL: update Ba to Ua or O to La
    """
    previous_label = tag_sequence[-1]
    previous_tag = previous_label[0]
    del tag_sequence[-1]
    new_tag = "U" if previous_tag == "B" else "L"
    tag_sequence.append(f"{new_tag}-{current_tag_type}")


def _process_B(tag_sequence, new_tag_sequence, label, index, process_tag):
    """Process a span starting with a B, it receives the given tag sequence, a new tag sequence
    where the fixed sequence is stored, current label, index and finally a method process_tag which
    can be process_I in case of BIOUL and process_O in case of BOUL."""
    current_tag_type = label.partition("-")[2]
    index += 1
    if index < len(tag_sequence):
        new_tag_sequence.append(label)
        label = tag_sequence[index]
    else:
        # if B is last tag in the tag sequence, then append it as U
        new_tag_sequence.append(f'U-{label.partition("-")[2]}')
        return index, None
    index, label = process_tag(index, label, current_tag_type)

    if label[0] == "L":
        # if we encounter L directly after B or after processing
        # I e.g: Ba L* or Ia L* in case of BIOUL
        # O e.g: Ba L* or Ba O in case of BOUL
        tag_type = label.partition("-")[2]
        if tag_type == current_tag_type:
            # BIOUL: Ba La or Ia La
            # BOUL: Ba La or Ba O La
            new_tag_sequence.append(label)
        else:
            # BIOUL : Ba Lb or Ia Lb to Ua Lb or La Lb respectively
            # BOUL : Ba Lb or O Lb to Ua Lb or La Lb respectively
            _update_previous_label(new_tag_sequence, current_tag_type)
            index -= 1
    return index, label


def fix_bioul(
    tag_sequence: List[str],
) -> List[str]:
    """In all the example shown in the method below we refer to BIOUL tags as tag and the attached
    type as tag type such as "Ba" means B as BIOUL tag and "a" as tag type.

    This method fix ill formed tag sequence by following some rules mentioned below:
    1. tag type takes precedence over tag that means if a sequence is Ba Ia Ua then it is converted to Ba Ia Ia
    2. individual occurrence of B or L in between other tag type is converted into U e.g: La Bb Ba to La Ub Ba
    3. encountering different tag type or O inside a span will end the span at last index
    There might be subcases of the above-mentioned rules which are described in in-line comments
    """

    def process_I(index, label, current_tag_type):
        while label[0] == "I" and index < len(tag_sequence):
            if label.partition("-")[2] != current_tag_type:
                # I with different tag type : Ba Ib or Ba Ia Ib convert to Ua Ib or Ba La Ib
                _update_previous_label(new_tag_sequence, current_tag_type)
                index -= 1
                return index, label
            else:
                # add I to new tag sequence
                new_tag_sequence.append(label)
            index += 1
            if index >= len(tag_sequence):
                # if we encounter last tag of sequence as Ia then convert it into La
                previous_label = new_tag_sequence[-1]
                previous_tag_type = previous_label.partition("-")[2]
                del new_tag_sequence[-1]
                new_tag_sequence.append(f"L-{previous_tag_type}")
                continue
            else:
                label = tag_sequence[index]
        if label[0] == "O":
            # if we encounter "O" after processing I then convert last I to L , e.g : Ba Ia Ia O to Ba Ia La O
            _update_previous_label(new_tag_sequence, current_tag_type)
            index -= 1
        if label[0] in ["U", "B"]:
            # if we encounter U or B after processing I
            tag_type = label.partition("-")[2]
            if tag_type == current_tag_type:
                # Ia Ua or Ia Ba to Ia Ia
                label = f"I-{tag_type}"
                index, label = process_I(index, label, current_tag_type)
            else:
                # Ia Ub or Ia Bb to La *b
                _update_previous_label(new_tag_sequence, current_tag_type)
                index -= 1
        return index, label

    new_tag_sequence = []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == "U":
            # span starts with U
            new_tag_sequence.append(label)
        elif label[0] == "B":
            # span starts with B
            index, label = _process_B(tag_sequence, new_tag_sequence, label, index, process_I)
        else:
            # if span starts with other than B and U
            if label != "O":
                # when span starts with I or L
                if label[0] == "L":
                    # if it is L then depending upon type of previous tag, L is handled differently
                    if index == 0:
                        tag_type = label.partition("-")[2]
                        new_tag_sequence.append(f"U-{tag_type}")
                    else:
                        previous_label = new_tag_sequence[-1]
                        previous_tag = previous_label[0]
                        previous_tag_type = (
                            previous_label.partition("-")[2] if previous_tag != "O" else None
                        )
                        if (
                            previous_tag in ["O", "U", "L"]
                            and previous_tag_type != label.partition("-")[2]
                        ):
                            # Ua Lb or O La or La Lb then convert to Ua Ub or 0 Ua or La Ub respectively
                            new_tag_sequence.append(f'U-{label.partition("-")[2]}')
                        elif previous_label == f'U-{label.partition("-")[2]}':
                            # Ua La to Ba La
                            del new_tag_sequence[-1]
                            new_tag_sequence.append(f"B-{previous_tag_type}")
                            new_tag_sequence.append(label)
                        else:  # previous_label == label:
                            # La La to Ia La
                            del new_tag_sequence[-1]
                            new_tag_sequence.append(f"I-{previous_tag_type}")
                            new_tag_sequence.append(label)
                elif label[0] == "I":
                    # if span starts with I, convert to B and start processing span
                    previous_tag_type = label.partition("-")[2]
                    label = f"B-{previous_tag_type}"
                    index, label = _process_B(
                        tag_sequence, new_tag_sequence, label, index, process_I
                    )
                else:
                    raise InvalidTagSequence(tag_sequence)
            else:
                # if it is "O"
                new_tag_sequence.append(label)
        index += 1
    return new_tag_sequence


def fix_boul(
    tag_sequence: List[str],
) -> List[str]:
    """In all the example shown in the method below we refer to BOUL tags as tag and the attached
    type as tag type such as "Ba" means B as BOUL tag and "a" as tag type.

    This method fix ill formed tag sequence by following some rules mentioned below:
    1. tag type takes precedence over tag that means if a sequence is Ba O Ua then it is converted to Ba O O, here
        O means that it is part of current span
    2. individual occurrence of B or L in between other tag type is converted into U e.g: La Bb Ba to La Ub Ba
    3. encountering different tag type inside a span will end the span at last index
    There might be subcases of the above-mentioned rules which are described in in-line comments
    """

    def process_O(index, label, current_tag_type):
        while label[0] == "O" and index < len(tag_sequence):
            new_tag_sequence.append(label)
            index += 1
            if index >= len(tag_sequence):
                # if tag sequence ends with an O then convert last O to L
                del new_tag_sequence[-1]
                new_tag_sequence.append(f"L-{current_tag_type}")
                continue
            else:
                label = tag_sequence[index]
        if label[0] in ["U", "B"]:
            # if there is a U or B within a span after O's
            tag_type = label.partition("-")[2]
            if tag_type == current_tag_type:
                # if it's of same tag type then convert to O and start processing O again
                label = "O"
                index, label = process_O(index, label, current_tag_type)
            else:
                # if it's of different tag type then end span at last index
                _update_previous_label(new_tag_sequence, current_tag_type)
                index -= 1
        return index, label

    new_tag_sequence = []
    index = 0

    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == "U":
            # if span starts with U
            new_tag_sequence.append(label)
        elif label[0] == "B":
            # if span starts with B
            index, label = _process_B(tag_sequence, new_tag_sequence, label, index, process_O)
        elif label[0] == "L":
            # if span starts with L, process span in backward direction
            i = len(new_tag_sequence) - 1
            current_tag_type = label.partition("-")[2]
            if i < 0:
                # if tag sequence starts with L tag, convert L to U, e.g: La *a .. to Ua *a ..
                new_tag_sequence.append(f"U-{current_tag_type}")
                index += 1
                continue
            else:
                label = new_tag_sequence[i]
            count = 1
            while label[0] == "O" and i >= 0:
                # traverse backward as far as O is there
                count += 1
                i -= 1
                if i < 0:
                    # if tag sequence starts with L tag, convert L to U, e.g: O O La *a .. to O O Ua *a ..
                    new_tag_sequence.append(f"U-{current_tag_type}")
                else:
                    label = new_tag_sequence[i]
            # Note : It cannot be the case that we encounter a B while traversing backward as this situation would
            # already be handled in process_B
            if label[0] in ["L", "U"]:
                # if L or U is encountered
                tag_type = label.partition("-")[2]
                if tag_type == current_tag_type:
                    # La O La or Ua O La
                    if label[0] == "L":
                        # LOL to OOL
                        new_tag_sequence[-count] = "O"
                        new_tag_sequence.append(f"L-{current_tag_type}")
                    else:
                        # UOL to BOL
                        new_tag_sequence[-count] = f"B-{current_tag_type}"
                        new_tag_sequence.append(f"L-{current_tag_type}")
                else:
                    if count == 1:
                        # Ua Lb to Ua Ub
                        new_tag_sequence.append(f"U-{current_tag_type}")
                    else:
                        # La O Lb to La Bb Lb and Ua O Lb to Ua Bb Lb
                        count -= 1
                        new_tag_sequence[-count] = f"B-{current_tag_type}"
                        new_tag_sequence.append(f"L-{current_tag_type}")
        else:
            if label != "O":
                raise InvalidTagSequence(tag_sequence)
            else:
                new_tag_sequence.append(label)
        index += 1
    return new_tag_sequence


def fix_iob2(tag_sequence: List[str]) -> List[str]:
    """In all the example shown in the method below we refer to IOB2 tags as tag and the attached
    type as tag type such as "Ba" means B as IOB2 tag and "a" as tag type.

    This method fix ill formed tag sequence by following some rules mentioned below:
    1. tag type takes precedence over tag that means if a sequence is Ba Ia Ib then it is converted to Ba Ia Bb
    2. If the span starts with I tag it will be converted to B e.g. if a sequence is Ia Bb Ib then it is converted to Ba Bb Ib
    """

    new_tag_sequence = []
    index = 0

    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == "B":
            # if span starts with B
            current_tag_type = label.partition("-")[2]
            new_tag_sequence.append(label)
            index += 1
            if index < len(tag_sequence):
                label = tag_sequence[index]
            if label[0] == "B":
                # when there is B after B, we process again from second B
                continue
            while label[0] == "I" and index < len(tag_sequence):
                if label.partition("-")[2] != current_tag_type:
                    # Ba Ia Ib to Ba Ia Bb
                    tag_sequence[index] = f'B-{label.partition("-")[2]}'
                    index -= 1
                    break
                new_tag_sequence.append(label)
                index += 1
                if index < len(tag_sequence):
                    label = tag_sequence[index]
            if index < len(tag_sequence) and label[0] != "I":
                # when while loop ends due to encountering something other than I
                continue
        elif label[0] == "I":
            # Ia Ia to Ba Ia
            current_tag_type = label.partition("-")[2]
            tag_sequence[index] = f"B-{current_tag_type}"
            index -= 1
        else:
            if label != "O":
                raise InvalidTagSequence(tag_sequence)
            else:
                new_tag_sequence.append(label)
        index += 1
    return new_tag_sequence


def remove_bioul(
    tag_sequence: List[str],
) -> List[str]:
    """Removes the ill formed tag sequence from the given sequence.

    Note: if a span do not start with B, but encounters a U then it is not removed
    e.g: BILOIUL converts to BILOOUO but
        BILBUL converts to BILOOO
    """
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == "U":
            index += 1
            continue
        elif label[0] == "B":
            start = index
            current_span_label = label.partition("-")[2]
            expected_label = f"I-{current_span_label}"
            index += 1
            label = tag_sequence[index]
            while label == expected_label and index < len(tag_sequence):
                index += 1
                if index < len(tag_sequence):
                    label = tag_sequence[index]
            if label != f"L-{current_span_label}":
                last = (
                    index if index >= len(tag_sequence) else index + 1
                )  # if current index is greater than sequence length
                for i in range(start, last):
                    tag_sequence[i] = "O"
            index += 1
        else:
            if label == "O":
                index += 1
                continue
            tag_sequence[index] = "O"
    return tag_sequence


def remove_boul(
    tag_sequence: List[str],
) -> List[str]:
    """Removes the ill formed tag sequence from the given sequence.

    Note: if a span do not start with B, but encounters a U then it is not removed
    e.g: BOLOOUL converts to BILOOUO but
         BILBUL converts to BILOOO
    """
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == "U":
            index += 1
            continue
        elif label[0] == "B":
            start = index
            current_span_label = label.partition("-")[2]
            expected_label = "O"
            index += 1
            label = tag_sequence[index]
            while label == expected_label and index < len(tag_sequence):
                index += 1
                if index < len(tag_sequence):
                    label = tag_sequence[index]
            if label != f"L-{current_span_label}":
                last = (
                    index if index >= len(tag_sequence) else index + 1
                )  # if current index is greater than sequence length
                for i in range(start, last):
                    tag_sequence[i] = "O"
            index += 1
        else:
            if label == "O":
                index += 1
                continue
            tag_sequence[index] = "O"
    return tag_sequence


def remove_iob2(
    tag_sequence: List[str],
) -> List[str]:
    """Removes the ill formed tag sequence from the given sequence.

    e.g: BaIaLa converts to BaIaO      BaIaBaBaIbBb converts to BaIaBaBaOBb
    """
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == "B":
            current_span_label = label.partition("-")[2]
            expected_label = f"I-{current_span_label}"
            index += 1
            if index < len(tag_sequence):
                label = tag_sequence[index]
            if label[0] == "B":
                continue
            while label == expected_label and index < len(tag_sequence):
                index += 1
                if index < len(tag_sequence):
                    label = tag_sequence[index]
        else:
            if label == "O":
                index += 1
                continue
            tag_sequence[index] = "O"
    return tag_sequence


def fix_encoding(
    tag_sequence: List[str],
    encoding: str,
) -> List[str]:
    """Given a tag sequence with it's encoding scheme, the ill formed sequence in fixed.

    Encoding can only be IOB2, BIOUL or BOUL.
    """
    if encoding == "BIOUL":
        return fix_bioul(tag_sequence)
    elif encoding == "BOUL":
        return fix_boul(tag_sequence)
    elif encoding == "IOB2":
        return fix_iob2(tag_sequence)
    else:
        raise ValueError(f"Unknown Coding scheme {encoding}.")


def remove_encoding(
    tag_sequence: List[str],
    encoding: str,
) -> List[str]:
    """Given a tag sequence with it's encoding scheme, the ill formed sequence in removed.

    Encoding can only be IOB2, BIOUL or BOUL.
    """
    if encoding == "BIOUL":
        return remove_bioul(tag_sequence)
    elif encoding == "BOUL":
        return remove_boul(tag_sequence)
    elif encoding == "IOB2":
        return remove_iob2(tag_sequence)
    else:
        raise ValueError(f"Unknown Coding scheme {encoding}.")
