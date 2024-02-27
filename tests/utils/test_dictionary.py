from pie_modules.utils import flatten_dict


def test_flatten_nested_dict():
    d = {"a": {"b": 1, "c": 2}, "d": 3}
    assert flatten_dict(d) == {"a/b": 1, "a/c": 2, "d": 3}
    assert flatten_dict(d, sep=".") == {"a.b": 1, "a.c": 2, "d": 3}
