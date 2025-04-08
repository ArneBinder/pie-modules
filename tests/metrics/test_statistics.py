from pie_modules.metrics.statistics import (
    DummyCollector,
    FieldLengthCollector,
    LabelCountCollector,
    SubFieldLengthCollector,
    TokenCountCollector,
)


def test_statistics(document_dataset):
    statistic = DummyCollector()
    values = statistic(document_dataset)
    assert values == {"test": {"sum": 2}, "train": {"sum": 7}, "val": {"sum": 2}}

    # note that we check for labels=["LOC", "PER", "ORG"], but the actual labels in the data are just ["PER", "ORG"]
    statistic = LabelCountCollector(field="entities", labels=["LOC", "PER", "ORG"])
    values = statistic(document_dataset)
    assert values == {
        "test": {
            "LOC": {"mean": 0.0, "std": 0.0, "min": 0, "max": 0, "len": 2, "sum": 0},
            "PER": {"mean": 0.5, "std": 0.5, "min": 0, "max": 1, "len": 2, "sum": 1},
            "ORG": {"mean": 1.0, "std": 1.0, "min": 0, "max": 2, "len": 2, "sum": 2},
        },
        "val": {
            "LOC": {"mean": 0.0, "std": 0.0, "min": 0, "max": 0, "len": 2, "sum": 0},
            "PER": {"mean": 0.5, "std": 0.5, "min": 0, "max": 1, "len": 2, "sum": 1},
            "ORG": {"mean": 1.0, "std": 1.0, "min": 0, "max": 2, "len": 2, "sum": 2},
        },
        "train": {
            "LOC": {"mean": 0.0, "std": 0.0, "min": 0, "max": 0, "len": 7, "sum": 0},
            "PER": {
                "mean": 1.0,
                "std": 0.5345224838248488,
                "min": 0,
                "max": 2,
                "len": 7,
                "sum": 7,
            },
            "ORG": {
                "mean": 1.2857142857142858,
                "std": 0.6998542122237652,
                "min": 0,
                "max": 2,
                "len": 7,
                "sum": 9,
            },
        },
    }

    statistic = LabelCountCollector(field="entities", labels="INFERRED")
    values = statistic(document_dataset)
    assert values == {
        "test": {"PER": {"max": 1, "len": 1, "sum": 1}, "ORG": {"max": 2, "len": 1, "sum": 2}},
        "val": {"PER": {"max": 1, "len": 1, "sum": 1}, "ORG": {"max": 2, "len": 1, "sum": 2}},
        "train": {"PER": {"max": 2, "len": 6, "sum": 7}, "ORG": {"max": 2, "len": 6, "sum": 9}},
    }

    statistic = FieldLengthCollector(field="text")
    values = statistic(document_dataset)
    assert values == {
        "test": {"mean": 34.5, "std": 16.5, "min": 18, "max": 51},
        "val": {"mean": 34.5, "std": 16.5, "min": 18, "max": 51},
        "train": {"mean": 29.714285714285715, "std": 15.153634978486993, "min": 15, "max": 54},
    }

    # this is not super useful, we just collect the lengths of the labels, but it is enough to test the code
    statistic = SubFieldLengthCollector(field="entities", subfield="label")
    values = statistic(document_dataset)
    assert values == {
        "test": {"mean": 3.0, "std": 0.0, "min": 3, "max": 3},
        "val": {"mean": 3.0, "std": 0.0, "min": 3, "max": 3},
        "train": {"mean": 3.0, "std": 0.0, "min": 3, "max": 3},
    }


def test_statistics_with_tokenize(document_dataset):
    statistic = TokenCountCollector(
        text_field="text",
        tokenizer="bert-base-uncased",
        tokenizer_kwargs=dict(add_special_tokens=False),
    )
    values = statistic(document_dataset)
    assert values == {
        "test": {"max": 13, "mean": 8.5, "min": 4, "std": 4.5},
        "train": {"max": 14, "mean": 8.285714285714286, "min": 4, "std": 3.5742845723419436},
        "val": {"max": 13, "mean": 8.5, "min": 4, "std": 4.5},
    }
