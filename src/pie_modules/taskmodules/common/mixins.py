import dataclasses
import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_ie import Annotation
from torch import Tensor

logger = logging.getLogger(__name__)


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

    Note that the padding is done on all dimensions.

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
    """A mixin class that provides a batch method to batch a list of instances of the class. All
    attributes, but also property methods, are batched. The batch method returns a dictionary with
    all attribute / property names as keys. The values are tensors created from the stacked values
    of the attributes / properties. The tensors are padded to the length of the longest instance in
    the batch and converted to the given dtype.

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
            # Only batch attributes that are not None for any of the values.
            if not all(getattr(x, k) is None for x in values)
        }


class RelationStatisticsMixin:
    def __init__(self, collect_statistics: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.collect_statistics = collect_statistics
        self.reset_statistics()

    def reset_statistics(self):
        self._statistics = defaultdict(int)
        self._collected_relations: Dict[str, List[Annotation]] = defaultdict(list)

    def collect_relation(self, kind: str, relation: Annotation):
        if self.collect_statistics:
            self._collected_relations[kind].append(relation)

    def collect_all_relations(self, kind: str, relations: Iterable[Annotation]):
        if self.collect_statistics:
            self._collected_relations[kind].extend(relations)

    def finalize_statistics(self):
        if self.collect_statistics:
            all_relations = set(self._collected_relations["available"])
            used_relations = set(self._collected_relations["used"])
            skipped_other = all_relations - used_relations
            for key, rels in self._collected_relations.items():
                rels_set = set(rels)
                if key.startswith("skipped_"):
                    skipped_other -= rels_set
                elif key.startswith("used_"):
                    pass
                elif key in ["available", "used"]:
                    pass
                else:
                    raise ValueError(f"unknown key: {key}")
                for rel in rels_set:
                    # Set "no_relation" as label when the score is zero. We encode negative relations
                    # in such a way in the case of multi-label or binary (similarity for coref).
                    label = rel.label if rel.score > 0 else "no_relation"
                    self.increase_counter(key=(key, label))
            for rel in skipped_other:
                self.increase_counter(key=("skipped_other", rel.label))

    def show_statistics(self):
        if self.collect_statistics:
            self.finalize_statistics()

            to_show = pd.Series(self._statistics)
            if len(to_show.index.names) > 1:
                to_show = to_show.unstack()
            to_show = to_show.fillna(0)
            if to_show.columns.size > 1:
                to_show["all_relations"] = to_show.loc[:, to_show.columns != "no_relation"].sum(
                    axis=1
                )
            if "used" in to_show.index and "available" in to_show.index:
                to_show.loc["used %"] = (
                    100 * to_show.loc["used"] / to_show.loc["available"]
                ).round()
            logger.info(f"statistics:\n{to_show.to_markdown()}")

    def increase_counter(self, key: Tuple[Any, ...], value: Optional[int] = 1):
        if self.collect_statistics:
            key_str = tuple(str(k) for k in key)
            self._statistics[key_str] += value
