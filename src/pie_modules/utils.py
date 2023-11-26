from typing import Optional, Type, TypeVar, Union

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
