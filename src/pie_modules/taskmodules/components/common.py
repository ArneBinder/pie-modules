import abc
import dataclasses
import logging
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F
from pytorch_ie import Annotation
from torch import Tensor
from torchmetrics import Metric

logger = logging.getLogger(__name__)


AE = TypeVar("AE")
A = TypeVar("A", bound=Annotation)
ALE = TypeVar("ALE")


class AnnotationEncoderDecoder(abc.ABC, Generic[A, AE]):
    """Base class for annotation encoders and decoders."""

    @abc.abstractmethod
    def encode(self, annotation: A, metadata: Optional[Dict[str, Any]] = None) -> Optional[AE]:
        pass

    @abc.abstractmethod
    def decode(self, encoding: AE, metadata: Optional[Dict[str, Any]] = None) -> Optional[A]:
        pass


class AnnotationLayersEncoderDecoder(abc.ABC, Generic[ALE]):
    """Base class for annotation layer encoders and decoders."""

    @property
    @abc.abstractmethod
    def layer_names(self) -> List[str]:
        pass

    @abc.abstractmethod
    def encode(
        self, layers: Dict[str, List[Annotation]], metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ALE]:
        pass

    @abc.abstractmethod
    def decode(
        self, encoding: ALE, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, List[Annotation]], Any]:
        pass

    def get_metric(self, **kwargs) -> Metric:
        raise NotImplementedError

    def unbatch(self, prediction: Dict[str, Tensor]) -> List[ALE]:
        raise NotImplementedError


def _pad_tensor(tensor: Tensor, target_shape: List[int], pad_value: float) -> Tensor:
    """Pad a tensor to a target shape.

    Args:
        tensor: The tensor to pad.
        target_shape: The target shape.
        pad_value: The value to use for padding.

    Returns: The padded tensor.
    """

    shape = tensor.shape
    pad: List[int] = []
    for i, s in enumerate(shape):
        pad = [0, target_shape[i] - s] + pad
    result = F.pad(tensor, pad=pad, value=pad_value)

    return result


def maybe_pad_values(
    values: Any, pad_value: Optional[Any] = None, strategy: str = "longest"
) -> Any:
    """If an iterable of values is passed and a pad value is given, pad the values to the same
    length and create a tensor from them. Otherwise, return the values unchanged.

    Args:
        values: The values to pad.
        pad_value: The value to use for padding.
        strategy: The padding strategy. Currently only "longest" is supported.

    Returns: The padded values.
    """

    if pad_value is None:
        return values
    if not isinstance(values, Iterable):
        raise TypeError(f"values must be iterable to pad them, but got {type(values)}")
    if strategy != "longest":
        raise ValueError(f"unknown padding strategy: {strategy}")
    tensor_list = [torch.tensor(value_list) for value_list in values]
    shape_lists = list(zip(*[t.shape for t in tensor_list]))
    max_shape = [max(dims) for dims in shape_lists]
    padded = [
        _pad_tensor(tensor=t, target_shape=max_shape, pad_value=pad_value)
        for i, t in enumerate(tensor_list)
    ]
    return torch.stack(padded)


def maybe_to_tensor(
    values: Iterable[Any], dtype: Optional[torch.dtype] = None, pad_value: Optional[Any] = None
) -> Any:
    """If an iterable of values is passed and a dtype is given, convert the values to a tensor of
    the given type.

    Args:
        values: The values to convert.
        dtype: A dtype to convert the values to.
        pad_value: A pad value to use if the values are padded.

    Returns: A tensor or the values unchanged.
    """

    if all(v is None for v in values):
        return None
    if dtype is None:
        return values
    maybe_padded = maybe_pad_values(values=values, pad_value=pad_value)
    if not isinstance(maybe_padded, torch.Tensor):
        maybe_padded = torch.Tensor(maybe_padded)
    tensor = maybe_padded.to(dtype=dtype)
    return tensor


class BatchableMixin:
    """A mixin class that provides a batch method to batch a list of instances of the class.

    Example:
        >>> import dataclasses
        >>> from typing import List, Dict
        >>> import torch
        >>>
        >>> @dataclasses.dataclass
        >>> class Foo(BatchableMixin):
        >>>     a: List[int]
        >>>
        >>>   @property
        >>>   def len_a(self):
        >>>       return len(self.a)
        >>>
        >>> x = Foo(a=[1, 2, 3])
        >>> y = Foo(a=[4, 5])
        >>>
        >>> Foo.batch(values=[x, y], dtypes={"a": torch.int64, "len_a": torch.int64}, pad_values={"a": 0})
        {'a': tensor([[1, 2, 3],[4, 5, 0]]), 'len_a': tensor([3, 2])}
    """

    @classmethod
    def get_property_names(cls) -> List[str]:
        return [name for name in cls.__dict__ if isinstance(getattr(cls, name), property)]

    @classmethod
    def get_dataclass_field_names(cls) -> List[str]:
        if dataclasses.is_dataclass(cls):
            return [f.name for f in dataclasses.fields(cls)]
        else:
            return []

    @classmethod
    def get_attribute_names(cls) -> List[str]:
        return cls.get_property_names() + cls.get_dataclass_field_names()

    @classmethod
    def batch(
        cls,
        values: List[Any],
        dtypes: Dict[str, torch.dtype],
        pad_values: Dict[str, Any],
    ) -> Dict[str, Any]:
        attribute_names = cls.get_attribute_names()
        return {
            k: maybe_to_tensor(
                values=[getattr(x, k) for x in values],
                dtype=dtypes.get(k, None),
                pad_value=pad_values.get(k, None),
            )
            for k in attribute_names
        }
