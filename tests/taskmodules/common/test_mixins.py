import dataclasses
import logging
from typing import List

import torch
from pytorch_ie import Annotation

from pie_modules.taskmodules.common import BatchableMixin
from pie_modules.taskmodules.common.mixins import RelationStatisticsMixin


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
    torch.testing.assert_close(batch["a"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
    torch.testing.assert_close(batch["len_a"], torch.tensor([3, 2]))


def test_relation_statistics_mixin_show_statistics(caplog):
    """Test the RelationStatisticsMixin class."""

    class Foo(RelationStatisticsMixin):
        """A class that uses the RelationStatisticsMixin class."""

        pass

    @dataclasses.dataclass(eq=True, frozen=True)
    class TestAnnotation(Annotation):
        label: str
        score: float = dataclasses.field(default=1.0, compare=False)

    x = Foo(collect_statistics=True)

    relations = [
        TestAnnotation(label="A"),
        TestAnnotation(label="B"),
        TestAnnotation(label="C"),
        TestAnnotation(label="D"),
    ]
    x.collect_all_relations(kind="available", relations=relations)
    x.collect_relation(kind="skipped_test", relation=relations[1])
    x.collect_all_relations(kind="used", relations=[relations[0], relations[2]])
    with caplog.at_level(logging.INFO):
        x.show_statistics()
    assert caplog.messages[0] == (
        "statistics:\n"
        "|               |   A |   B |   C |   D |   all_relations |\n"
        "|:--------------|----:|----:|----:|----:|----------------:|\n"
        "| available     |   1 |   1 |   1 |   1 |               4 |\n"
        "| skipped_other |   0 |   0 |   0 |   1 |               1 |\n"
        "| skipped_test  |   0 |   1 |   0 |   0 |               1 |\n"
        "| used          |   1 |   0 |   1 |   0 |               2 |\n"
        "| used %        | 100 |   0 | 100 |   0 |              50 |"
    )
