from typing import Any, Iterable, MutableMapping, Optional, Tuple, Type, TypeVar, Union

from pytorch_ie import Document
from pytorch_ie.utils.hydra import resolve_target

T = TypeVar("T", bound=Document)
T_super = TypeVar("T_super", bound=Document)


def resolve_type(
    type_or_str: Union[str, Type[T]], expected_super_type: Optional[Type[T_super]] = None
) -> Type[T]:
    if isinstance(type_or_str, str):
        dt = resolve_target(type_or_str)  # type: ignore
    else:
        dt = type_or_str
    if not (
        isinstance(dt, type)
        and (expected_super_type is None or issubclass(dt, expected_super_type))
    ):
        raise TypeError(
            f"type must be a subclass of {expected_super_type} or a string that resolves to that, "
            f"but got {dt}"
        )
    return dt


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
