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

    @dataclasses.dataclass
    class Foo(RelationStatisticsMixin):
        """A class that uses the RelationStatisticsMixin class."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    @dataclasses.dataclass(eq=True, frozen=True)
    class TestAnnotation(Annotation):
        label: str
        score: float = dataclasses.field(default=1.0, compare=False)

    x = Foo(collect_statistics=True)

    # Test with no relations collected
    x.collect_all_relations(kind="available", relations=[])
    x.collect_all_relations(kind="used", relations=[])
    with caplog.at_level(logging.INFO):
        x.show_statistics()
    assert caplog.messages[0] == ("statistics:\n" "| 0   |\n" "|-----|")

    # Test Regular case
    x.reset_statistics()
    relations = [
        TestAnnotation(label="A", score=1),
        TestAnnotation(label="B", score=0.5),
        TestAnnotation(label="C", score=0.3),
    ]
    x.collect_all_relations(kind="available", relations=relations)
    x.collect_relation(kind="skipped_test", relation=relations[1])
    x.collect_all_relations(
        kind="used",
        relations=set(x._collected_relations["available"])
        - set(x._collected_relations["skipped_test"]),
    )
    with caplog.at_level(logging.INFO):
        x.show_statistics()
    assert caplog.messages[1] == (
        "statistics:\n"
        "|              |   A |   B |   C |   all_relations |\n"
        "|:-------------|----:|----:|----:|----------------:|\n"
        "| available    |   1 |   1 |   1 |               3 |\n"
        "| skipped_test |   0 |   1 |   0 |               1 |\n"
        "| used         |   1 |   0 |   1 |               2 |\n"
        "| used %       | 100 |   0 | 100 |              67 |"
    )
