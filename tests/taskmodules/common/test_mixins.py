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
    # all available relations
    x.collect_all_relations(kind="available", relations=relations)
    # relations skipped for a reason ("test")
    x.collect_relation(kind="skipped_test", relation=relations[1])
    # mark two relations as used, one of them is skipped for another (unknown) reason
    x.collect_all_relations(kind="used", relations=[relations[0], relations[2]])

    with caplog.at_level(logging.INFO):
        x.show_statistics()
    assert caplog.messages[0] == (
        "Foo does not have a `none_label` attribute. "
        "Using 'no_relation' as the label for relations with score 0 in statistics. "
        "Set the `none_label` or `_statistics_none_label` attribute before using statistics or "
        "overwrite `get_none_label_for_statistics()` function to get rid of this message."
    )
    assert caplog.messages[1] == (
        "statistics:\n"
        "|               |   A |   B |   C |   D |   all_relations |\n"
        "|:--------------|----:|----:|----:|----:|----------------:|\n"
        "| available     |   1 |   1 |   1 |   1 |               4 |\n"
        "| skipped_other |   0 |   0 |   0 |   1 |               1 |\n"
        "| skipped_test  |   0 |   1 |   0 |   0 |               1 |\n"
        "| used          |   1 |   0 |   1 |   0 |               2 |\n"
        "| used %        | 100 |   0 | 100 |   0 |              50 |"
    )


def test_relation_statistics_mixin_show_statistics_no_relations(caplog):
    """Test the RelationStatisticsMixin class with 0 score prediction."""

    class Foo(RelationStatisticsMixin):
        """A class that uses the RelationStatisticsMixin class."""

        pass

    x = Foo(collect_statistics=True)

    # Test with no relations collected
    x.collect_all_relations(kind="available", relations=[])
    x.collect_all_relations(kind="used", relations=[])
    with caplog.at_level(logging.INFO):
        x.show_statistics()
    assert caplog.messages[0] == "statistics:\n" "| 0   |\n" "|-----|"


def test_relation_statistics_mixin_show_statistics_custom_none_label(caplog):
    """Test the RelationStatisticsMixin class with custom none_label."""

    class Foo(RelationStatisticsMixin):
        """A class that uses the RelationStatisticsMixin class.

        It also sets the `none_label` attribute which will be used by statistics.
        """

        def __init__(self, none_label: str = "no_relation", **kwargs):
            super().__init__(**kwargs)
            self.none_label = none_label

    @dataclasses.dataclass(eq=True, frozen=True)
    class TestAnnotation(Annotation):
        label: str
        score: float = dataclasses.field(default=1.0, compare=False)

    x = Foo(collect_statistics=True, none_label="None_Label")

    relations = [
        TestAnnotation(label="A", score=1.0),
        TestAnnotation(label="B", score=0.5),
        TestAnnotation(label="C", score=0.0),
        TestAnnotation(label="D", score=0.3),
    ]
    # all available relations
    x.collect_all_relations(kind="available", relations=relations)
    # relations skipped for a reason ("test")
    x.collect_relation(kind="skipped_test", relation=relations[1])
    # mark two relations as used, one of them is skipped for another (unknown) reason
    x.collect_all_relations(kind="used", relations=[relations[0], relations[2]])

    with caplog.at_level(logging.INFO):
        x.show_statistics()
    assert caplog.messages[0] == (
        "statistics:\n"
        "|               |   A |   B |   D |   None_Label |   all_relations |\n"
        "|:--------------|----:|----:|----:|-------------:|----------------:|\n"
        "| available     |   1 |   1 |   1 |            1 |               3 |\n"
        "| skipped_other |   0 |   0 |   1 |            0 |               1 |\n"
        "| skipped_test  |   0 |   1 |   0 |            0 |               1 |\n"
        "| used          |   1 |   0 |   0 |            1 |               1 |\n"
        "| used %        | 100 |   0 |   0 |          100 |              33 |"
    )
