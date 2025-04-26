from typing import Tuple


def is_contained_in(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> bool:
    """Check if one span is contained in another. The spans are defined by their start and end
    indices.

    Note that the span is considered to be contained in the other span if it is completely within
    the other span, including the case where they are identical.
    """
    return other_start_end[0] <= start_end[0] and start_end[1] <= other_start_end[1]


def are_nested(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> bool:
    """Check if two spans are nested. The spans are defined by their start and end indices.

    Note that spans are considered to be nested if one is completely contained in the other,
    including the case where they are identical.
    """
    return is_contained_in(start_end, other_start_end) or is_contained_in(
        other_start_end, start_end
    )


def have_overlap(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> bool:
    """Check if two spans have an overlap. The spans are defined by their start and end indices.

    Note that two spans that are touching each other are not considered to have an overlap. But two
    spans that are nested, including the case where they are identical, are considered to have an
    overlap.
    """

    other_start_overlaps = start_end[0] <= other_start_end[0] < start_end[1]
    other_end_overlaps = start_end[0] < other_start_end[1] <= start_end[1]
    start_overlaps_other = other_start_end[0] <= start_end[0] < other_start_end[1]
    end_overlaps_other = other_start_end[0] < start_end[1] <= other_start_end[1]
    return other_start_overlaps or other_end_overlaps or start_overlaps_other or end_overlaps_other


def distance_center(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> float:
    """Calculate the distance between the centers of two spans.

    The spans are defined by their start and end indices.
    """
    center = (start_end[0] + start_end[1]) / 2
    center_other = (other_start_end[0] + other_start_end[1]) / 2
    return abs(center - center_other)


def distance_outer(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> float:
    """Calculate the distance between the outer edges of two spans. The spans are defined by their
    start and end indices.

    In case of an overlap, the covered area is considered to be the distance.
    """
    _max = max(start_end[0], start_end[1], other_start_end[0], other_start_end[1])
    _min = min(start_end[0], start_end[1], other_start_end[0], other_start_end[1])
    return float(_max - _min)


def distance_inner(start_end: Tuple[int, int], other_start_end: Tuple[int, int]) -> float:
    """Calculate the distance between the inner edges of two spans. The spans are defined by their
    start and end indices.

    In case of an overlap, the negative of the overlapping area is considered to be the distance.
    """
    dist_start_other_end = abs(start_end[0] - other_start_end[1])
    dist_end_other_start = abs(start_end[1] - other_start_end[0])
    dist = float(min(dist_start_other_end, dist_end_other_start))
    if not have_overlap(start_end, other_start_end):
        return dist
    else:
        return -dist


def distance(
    start_end: Tuple[int, int], other_start_end: Tuple[int, int], distance_type: str
) -> float:
    """Calculate the distance between two spans based on the given distance type.

    Args:
        start_end: a tuple of two integers representing the start and end index of the first span
        other_start_end: a tuple of two integers representing the start and end index of the second span
        distance_type: the type of distance to calculate. One of: center, inner, outer
    """
    if distance_type == "center":
        return distance_center(start_end, other_start_end)
    elif distance_type == "inner":
        return distance_inner(start_end, other_start_end)
    elif distance_type == "outer":
        return distance_outer(start_end, other_start_end)
    else:
        raise ValueError(
            f"unknown distance_type={distance_type}. use one of: center, inner, outer"
        )
