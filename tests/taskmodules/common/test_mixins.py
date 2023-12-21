import dataclasses
from typing import List

import torch

from pie_modules.taskmodules.common import BatchableMixin


def test_batchable_mixin():
    """Test the BatchableMixin class."""

    # pylint: disable=too-few-public-methods
    @dataclasses.dataclass
    class Foo(BatchableMixin):
        """A class that uses the BatchableMixin class."""

        a: List[int]

        @property
        def len_a(self):
            """Return the length of the list a."""
            return len(self.a)

    x = Foo(a=[1, 2, 3])
    y = Foo(a=[4, 5])

    batch = Foo.batch(
        values=[x, y], dtypes={"a": torch.int64, "len_a": torch.int64}, pad_values={"a": 0}
    )
    torch.testing.assert_allclose(batch["a"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
    torch.testing.assert_allclose(batch["len_a"], torch.tensor([3, 2]))
