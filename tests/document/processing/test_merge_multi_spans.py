from pie_modules.annotations import BinaryRelation, LabeledMultiSpan, LabeledSpan
from pie_modules.document.processing import MultiSpanMerger
from pie_modules.documents import (
    TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)


def test_multi_span_merger():
    doc = TextDocumentWithLabeledMultiSpansBinaryRelationsAndLabeledPartitions(
        text='He lives in New "Never Sleeping" York.'
    )
    he = LabeledMultiSpan(slices=((0, 2),), label="PER")
    doc.labeled_multi_spans.append(he)
    assert str(he) == "('He',)"
    new_york = LabeledMultiSpan(slices=((12, 15), (33, 37)), label="LOC")
    doc.labeled_multi_spans.append(new_york)
    assert str(new_york) == "('New', 'York')"
    # same as new_york but with a different end
    new_york_prediction = LabeledMultiSpan(slices=((12, 15), (33, 38)), label="LOC")
    doc.labeled_multi_spans.predictions.append(new_york_prediction)
    assert str(new_york_prediction) == "('New', 'York.')"
    lives_in = BinaryRelation(head=he, tail=new_york, label="lives_in")
    doc.binary_relations.append(lives_in)
    sentence = LabeledSpan(start=0, end=len(doc.text), label="SENTENCE")
    doc.labeled_partitions.append(sentence)
    assert str(sentence) == 'He lives in New "Never Sleeping" York.'

    multi_span_merger = MultiSpanMerger(
        result_document_type=TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
        layer="labeled_multi_spans",
        result_field_mapping={
            "labeled_multi_spans": "labeled_spans",
        },
    )
    result = multi_span_merger(doc)
    assert len(result.labeled_spans) == 2
    assert str(result.labeled_spans[0]) == "He"
    assert result.labeled_spans[0].label == he.label == "PER"
    assert str(result.labeled_spans[1]) == 'New "Never Sleeping" York'
    assert result.labeled_spans[1].label == new_york.label == "LOC"
    assert len(result.labeled_spans.predictions) == 1
    assert str(result.labeled_spans.predictions[0]) == 'New "Never Sleeping" York.'
    assert result.labeled_spans.predictions[0].label == new_york_prediction.label == "LOC"
    assert len(result.binary_relations) == 1
    assert result.binary_relations[0].head == result.labeled_spans[0]
    assert result.binary_relations[0].tail == result.labeled_spans[1]
    assert result.binary_relations[0].label == lives_in.label == "lives_in"
    assert len(result.labeled_partitions) == 1
    assert str(result.labeled_partitions[0]) == 'He lives in New "Never Sleeping" York.'
    assert result.labeled_partitions[0].label == sentence.label == "SENTENCE"
