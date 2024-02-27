import pytest

from pie_modules.utils.span import (
    are_nested,
    distance,
    distance_center,
    distance_inner,
    distance_outer,
    have_overlap,
)


def test_have_overlap():
    # no overlap, not touching
    assert not have_overlap((0, 1), (2, 3))
    assert not have_overlap((2, 3), (0, 1))
    # no overlap, touching
    assert not have_overlap((0, 1), (1, 2))
    assert not have_overlap((1, 2), (0, 1))
    # overlap, not touching
    assert have_overlap((0, 2), (1, 3))
    assert have_overlap((1, 3), (0, 2))
    # overlap, same start
    assert have_overlap((0, 2), (0, 3))
    assert have_overlap((0, 3), (0, 2))
    # overlap, same end
    assert have_overlap((0, 2), (1, 2))
    assert have_overlap((1, 2), (0, 2))
    # overlap, identical
    assert have_overlap((0, 1), (0, 1))


def test_are_nested():
    # no overlap, not touching
    assert not are_nested((0, 1), (2, 3))
    assert not are_nested((2, 3), (0, 1))
    # no overlap, touching
    assert not are_nested((0, 1), (1, 2))
    assert not are_nested((1, 2), (0, 1))
    # overlap, not touching
    assert not are_nested((0, 2), (1, 3))
    assert not are_nested((1, 3), (0, 2))
    # overlap, same start
    assert are_nested((0, 2), (0, 3))
    assert are_nested((0, 3), (0, 2))
    # overlap, same end
    assert are_nested((0, 2), (1, 2))
    assert are_nested((1, 2), (0, 2))
    # overlap, identical
    assert are_nested((0, 1), (0, 1))
    # nested, not touching
    assert are_nested((0, 3), (1, 2))
    assert are_nested((1, 2), (0, 3))


def test_distance_center():
    # no overlap, not touching
    assert distance_center((0, 1), (2, 3)) == 2.0
    assert distance_center((2, 3), (0, 1)) == 2.0
    # no overlap, touching
    assert distance_center((0, 1), (1, 2)) == 1.0
    assert distance_center((1, 2), (0, 1)) == 1.0
    # overlap, not touching
    assert distance_center((0, 2), (1, 3)) == 1.0
    assert distance_center((1, 3), (0, 2)) == 1.0
    # overlap, same start
    assert distance_center((0, 2), (0, 3)) == 0.5
    assert distance_center((0, 3), (0, 2)) == 0.5
    # overlap, same end
    assert distance_center((0, 2), (1, 2)) == 0.5
    assert distance_center((1, 2), (0, 2)) == 0.5
    # overlap, identical
    assert distance_center((0, 1), (0, 1)) == 0.0


def test_distance_inner():
    # no overlap, not touching
    assert distance_inner((0, 1), (2, 3)) == 1.0
    assert distance_inner((2, 3), (0, 1)) == 1.0
    # no overlap, touching
    assert distance_inner((0, 1), (1, 2)) == 0.0
    assert distance_inner((1, 2), (0, 1)) == 0.0
    # overlap, not touching
    assert distance_inner((0, 2), (1, 3)) == -1.0
    assert distance_inner((1, 3), (0, 2)) == -1.0
    # overlap, same start
    assert distance_inner((0, 2), (0, 3)) == -2.0
    assert distance_inner((0, 3), (0, 2)) == -2.0
    # overlap, same end
    assert distance_inner((0, 2), (1, 2)) == -1.0
    assert distance_inner((1, 2), (0, 2)) == -1.0
    # overlap, identical
    assert distance_inner((0, 1), (0, 1)) == -1.0


def test_distance_outer():
    # identical
    assert distance_outer((0, 1), (0, 1)) == 1.0
    # no overlap, not touching
    assert distance_outer((0, 1), (2, 3)) == 3.0
    assert distance_outer((2, 3), (0, 1)) == 3.0
    # no overlap, touching
    assert distance_outer((0, 1), (1, 2)) == 2.0
    assert distance_outer((1, 2), (0, 1)) == 2.0
    # overlap, not touching
    assert distance_outer((0, 2), (1, 3)) == 3.0
    assert distance_outer((1, 3), (0, 2)) == 3.0
    # overlap, same start
    assert distance_outer((0, 2), (0, 3)) == 3.0
    assert distance_outer((0, 3), (0, 2)) == 3.0
    # overlap, same end
    assert distance_outer((0, 2), (1, 2)) == 2.0
    assert distance_outer((1, 2), (0, 2)) == 2.0


@pytest.mark.parametrize(
    "distance_type",
    ["outer", "center", "inner", "unknown"],
)
def test_distance(distance_type):
    start_end = (0, 1)
    other_start_end = (2, 3)
    if distance_type != "unknown":
        distance(start_end, other_start_end, distance_type)
    else:
        with pytest.raises(ValueError) as excinfo:
            distance(start_end, other_start_end, distance_type)
        assert (
            str(excinfo.value) == "unknown distance_type=unknown. use one of: center, inner, outer"
        )
