import pytest

from pie_modules.utils.span import distance, distance_inner


def test_distance_inner_overlap():
    dist = distance_inner((0, 2), (1, 3))
    assert dist == -1


def test_distance_unknown_type():
    with pytest.raises(ValueError) as excinfo:
        distance((0, 1), (2, 3), "unknown")
    assert str(excinfo.value) == "unknown distance_type=unknown. use one of: center, inner, outer"
