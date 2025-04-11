from dataclasses import dataclass

import pytest
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.core import AnnotationLayer, annotation_field
from pytorch_ie.documents import TextBasedDocument

from pie_modules.metrics import ConfusionMatrix
import logging


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
    # wrong label
    documents[0].entities.predictions.append(LabeledSpan(start=35, end=43, label="cat"))
    # correct
    documents[1].entities.predictions.append(LabeledSpan(start=0, end=5, label="company"))
    # wrong span
    documents[1].entities.predictions.append(LabeledSpan(start=10, end=15, label="company"))

    return documents


def test_confusion_matrix(documents):
    metric = ConfusionMatrix(layer="entities")
    metric(documents)
    # (gold_label, predicted_label): count
    assert dict(metric.counts) == {
        ("animal", "animal"): 1,
        ("animal", "cat"): 1,
        ("UNDETECTED", "company"): 1,
        ("company", "company"): 1,
    }
    assert metric.compute() == {
        "animal": {"animal": 1, "cat": 1},
        "UNDETECTED": {"company": 1},
        "company": {"company": 1},
    }


def test_undetected_is_gold_label(documents):
    metric = ConfusionMatrix(layer="entities", undetected_label="animal")
    with pytest.raises(ValueError) as exception:
        metric(documents)

    assert str(exception.value).startswith("The gold annotation has the label")


def test_unassignable_is_pred_label(documents):
    metric = ConfusionMatrix(layer="entities", unassignable_label="cat")
    with pytest.raises(ValueError) as exception:
        metric(documents)

    assert str(exception.value).startswith("The predicted annotation has the label")


@pytest.fixture
def documents_with_several_gold_labels(documents):
    doc1 = documents[0].copy()
    doc2 = documents[1].copy()
    doc1.entities.append(LabeledSpan(start=4, end=19, label="cat"))

    return [doc1, doc2]


def test_documents_with_several_gold_labels(documents_with_several_gold_labels, caplog):
    metric = ConfusionMatrix(layer="entities")
    with pytest.raises(ValueError):
        metric(documents_with_several_gold_labels)

    metric = ConfusionMatrix(layer="entities", strict=False)
    metric(documents_with_several_gold_labels)
    assert caplog.messages[0].startswith("The base annotation LabeledSpan(start=4, end=19, label='DUMMY_LABEL', score=1.0) has multiple gold labels")
    # I only assert the start of the message here. The full warning includes the list of labels
    # (e.g. ['cat', 'animal']), but the order of the list seems to me to be random.


@pytest.fixture
def documents_without_predictions(documents):
    doc1 = documents[0].copy()
    doc2 = documents[1].copy()

    doc1.entities.predictions.clear()
    doc2.entities.predictions.clear()

    return [doc1, doc2]


def test_documents_without_predictions(documents_without_predictions):
    metric = ConfusionMatrix(layer="entities")
    metric(documents_without_predictions)
    assert dict(metric.counts) == {('animal', 'UNASSIGNABLE'): 2, ('company', 'UNASSIGNABLE'): 1}


def test_show_as_markdown(documents, caplog):
    caplog.set_level(logging.INFO)
    metric = ConfusionMatrix(layer="entities", show_as_markdown=True)
    metric(documents)

    markdown = ['\nentities:\n|            |   animal |   cat |   company |\n|:-----------|---------:|------:|----------:|\n| animal     |        1 |     1 |         0 |\n| company    |        0 |     0 |         1 |\n| UNDETECTED |        0 |     0 |         1 |']

    assert caplog.messages == markdown


def test_annotation_processor_str(documents):
    # annotation_processor = ""
    # metric = ConfusionMatrix(layer="entities", annotation_processor=annotation_processor)
    # Do you have any ideas for a good processor to use here?
