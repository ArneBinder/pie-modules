from typing import Any, Iterable, MutableMapping, Tuple


def list_of_dicts2dict_of_lists(list_of_dicts: list[dict]) -> dict[str, list]:
    return {k: [d[k] for d in list_of_dicts] for k in list_of_dicts[0].keys()}


def _flatten_dict_gen(d, parent_key, sep) -> Iterable[Tuple[str, Any]]:
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from _flatten_dict_gen(v, new_key, sep=sep)
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "/"):
    """Flatten a nested dictionary.

    Example:
        d = {"a": {"b": 1, "c": 2}, "d": 3}
        flatten_nested_dict(d) == {"a/b": 1, "a/c": 2, "d": 3}
    """
    return dict(_flatten_dict_gen(d, parent_key, sep))
