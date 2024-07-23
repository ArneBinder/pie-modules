import pytest
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.documents import TextDocumentWithLabeledPartitions

from pie_modules.document.processing import (
    FlairSegtokSentenceSplitter,
    NltkSentenceSplitter,
)


@pytest.mark.parametrize("inplace", [True, False])
def test_nltk_sentence_splitter(caplog, inplace):
    doc = TextDocumentWithLabeledPartitions(
        text="This is a test sentence. This is another one.", id="test_doc"
    )
    # add a dummy text partition to trigger the warning (see below)
    doc.labeled_partitions.append(LabeledSpan(start=0, end=len(doc.text), label="text"))
    caplog.clear()
    # create the sentence splitter
    sentence_splitter = NltkSentenceSplitter(inplace=inplace)
    # call the sentence splitter
    result = sentence_splitter(doc)
    if inplace:
        assert result is doc
    else:
        assert result is not doc
        doc = result
    # check the log message
    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message == "Layer labeled_partitions in document test_doc is not empty. "
        "Clearing it before adding new sentence partitions."
    )
    # check the result
    assert len(doc.labeled_partitions) == 2
    assert str(doc.labeled_partitions[0]) == "This is a test sentence."
    assert str(doc.labeled_partitions[1]) == "This is another one."


@pytest.mark.parametrize("inplace", [True, False])
def test_flair_segtok_sentence_splitter(caplog, inplace):
    doc = TextDocumentWithLabeledPartitions(
        text="This is a test sentence. This is another one.", id="test_doc"
    )
    # add a dummy text partition to trigger the warning (see below)
    doc.labeled_partitions.append(LabeledSpan(start=0, end=len(doc.text), label="text"))
    caplog.clear()
    # create the sentence splitter
    sentence_splitter = FlairSegtokSentenceSplitter(inplace=inplace)
    # call the sentence splitter
    result = sentence_splitter(doc)
    if inplace:
        assert result is doc
    else:
        assert result is not doc
        doc = result
    # check the log message
    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message == "Layer labeled_partitions in document test_doc is not empty. "
        "Clearing it before adding new sentence partitions."
    )
    # check the result
    assert len(doc.labeled_partitions) == 2
    assert str(doc.labeled_partitions[0]) == "This is a test sentence."
    assert str(doc.labeled_partitions[1]) == "This is another one."
