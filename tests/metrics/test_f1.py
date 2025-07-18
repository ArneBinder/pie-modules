from dataclasses import dataclass

import pytest
from pie_core import AnnotationLayer, annotation_field

from pie_modules.annotations import LabeledSpan
from pie_modules.documents import TextBasedDocument
from pie_modules.metrics import F1Metric


@pytest.fixture
def documents():
    @dataclass
    class TextDocumentWithEntities(TextBasedDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    # a test sentence with two entities
    doc1 = TextDocumentWithEntities(
        text="The quick brown fox jumps over the lazy dog.",
    )
    doc1.entities.append(LabeledSpan(start=4, end=19, label="animal"))
    doc1.entities.append(LabeledSpan(start=35, end=43, label="animal"))
    assert str(doc1.entities[0]) == "quick brown fox"
    assert str(doc1.entities[1]) == "lazy dog"

    # a second test sentence with a different text and a single entity (a company)
    doc2 = TextDocumentWithEntities(text="Apple is a great company.")
    doc2.entities.append(LabeledSpan(start=0, end=5, label="company"))
    assert str(doc2.entities[0]) == "Apple"

    documents = [doc1, doc2]

    # add predictions
    # correct
    documents[0].entities.predictions.append(LabeledSpan(start=4, end=19, label="animal"))
    # correct, but duplicate, this should not be counted
    documents[0].entities.predictions.append(LabeledSpan(start=4, end=19, label="animal"))
    # correct
    documents[0].entities.predictions.append(LabeledSpan(start=35, end=43, label="animal"))
    # wrong label
    documents[0].entities.predictions.append(LabeledSpan(start=35, end=43, label="cat"))
    # correct
    documents[1].entities.predictions.append(LabeledSpan(start=0, end=5, label="company"))
    # wrong span
    documents[1].entities.predictions.append(LabeledSpan(start=10, end=15, label="company"))

    return documents


def test_f1(documents):
    metric = F1Metric(layer="entities")
    metric(documents)
    # tp, fp, fn for micro
    assert dict(metric.counts) == {"MICRO": (3, 2, 0)}
    assert metric.compute() == {"MICRO": {"f1": 0.7499999999999999, "p": 0.6, "r": 1.0, "s": 3}}


def test_f1_per_label(documents):
    metric = F1Metric(layer="entities", labels=["animal", "company", "cat"])
    metric(documents)
    # tp, fp, fn for micro and per label
    assert dict(metric.counts) == {
        "MICRO": (3, 2, 0),
        "cat": (0, 1, 0),
        "company": (1, 1, 0),
        "animal": (2, 0, 0),
    }
    assert metric.compute() == {
        "MACRO": {"f1": 0.5555555555555556, "p": 0.5, "r": 0.6666666666666666},
        "MICRO": {"f1": 0.7499999999999999, "p": 0.6, "r": 1.0, "s": 3},
        "animal": {"f1": 1.0, "p": 1.0, "r": 1.0, "s": 2},
        "cat": {"f1": 0.0, "p": 0.0, "r": 0.0, "s": 0},
        "company": {"f1": 0.6666666666666666, "p": 0.5, "r": 1.0, "s": 1},
    }


def test_f1_per_label_inferred(documents):
    metric = F1Metric(layer="entities", labels="INFERRED")
    metric(documents)
    # tp, fp, fn for micro and per label
    assert dict(metric.counts) == {
        "MICRO": (3, 2, 0),
        "animal": (2, 0, 0),
        "company": (1, 1, 0),
        "cat": (0, 1, 0),
    }
    assert metric.compute() == {
        "MACRO": {"f1": 0.5555555555555556, "p": 0.5, "r": 0.6666666666666666},
        "MICRO": {"f1": 0.7499999999999999, "p": 0.6, "r": 1.0, "s": 3},
        "animal": {"f1": 1.0, "p": 1.0, "r": 1.0, "s": 2},
        "cat": {"f1": 0.0, "p": 0.0, "r": 0.0, "s": 0},
        "company": {"f1": 0.6666666666666666, "p": 0.5, "r": 1.0, "s": 1},
    }


def test_f1_per_label_no_labels(documents):
    with pytest.raises(ValueError) as excinfo:
        F1Metric(layer="entities", labels=[])
    assert str(excinfo.value) == "labels cannot be empty"


def test_f1_per_label_not_allowed():
    with pytest.raises(ValueError) as excinfo:
        F1Metric(layer="entities", labels=["animal", "MICRO"])
    assert (
        str(excinfo.value)
        == "labels cannot contain 'MICRO' or 'MACRO' because they are used to capture aggregated metrics"
    )


# def test_f1_show_as_markdown(documents, caplog):
#    metric = F1Metric(layer="entities", labels=["animal", "company", "cat"], show_as_markdown=True)
#    metric(documents)
#    caplog.set_level(logging.INFO)
#    caplog.clear()
#    metric.compute()
#    assert len(caplog.records) == 1
#    assert (
#        caplog.records[0].message == "\n"
#        "entities:\n"
#        "|         |    f1 |   p |     r |\n"
#        "|:--------|------:|----:|------:|\n"
#        "| MACRO   | 0.556 | 0.5 | 0.667 |\n"
#        "| MICRO   | 0.75  | 0.6 | 1     |\n"
#        "| animal  | 1     | 1   | 1     |\n"
#        "| company | 0.667 | 0.5 | 1     |\n"
#        "| cat     | 0     | 0   | 0     |"
#    )
