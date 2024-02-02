import json
from typing import Dict, Optional

import pytest
from pytorch_ie import Annotation

from pie_modules.annotations import LabeledMultiSpan


def _test_annotation_reconstruction(
    annotation: Annotation, annotation_store: Optional[Dict[int, Annotation]] = None
):
    ann_str = json.dumps(annotation.asdict())
    annotation_reconstructed = type(annotation).fromdict(
        json.loads(ann_str), annotation_store=annotation_store
    )
    assert annotation_reconstructed == annotation


def test_labeled_multi_span():
    labeled_multi_span1 = LabeledMultiSpan(slices=((1, 2), (3, 4)), label="label1")
    assert labeled_multi_span1.slices == ((1, 2), (3, 4))
    assert labeled_multi_span1.label == "label1"
    assert labeled_multi_span1.score == pytest.approx(1.0)

    labeled_multi_span2 = LabeledMultiSpan(
        slices=((5, 6), (7, 8)),
        label="label2",
        score=0.5,
    )
    assert labeled_multi_span2.slices == ((5, 6), (7, 8))
    assert labeled_multi_span2.label == "label2"
    assert labeled_multi_span2.score == pytest.approx(0.5)

    assert labeled_multi_span2.asdict() == {
        "_id": labeled_multi_span2._id,
        "slices": ((5, 6), (7, 8)),
        "label": "label2",
        "score": 0.5,
    }

    _test_annotation_reconstruction(labeled_multi_span2)
