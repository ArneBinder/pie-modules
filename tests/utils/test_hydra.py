import dataclasses

import pytest
from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import LabeledSpan, Span
from pytorch_ie.documents import TextBasedDocument

from pie_modules.utils import resolve_type


@dataclasses.dataclass
class TestDocumentWithEntities(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TestDocumentWithSentences(TextBasedDocument):
    sentences: AnnotationLayer[Span] = annotation_field(target="text")


def test_resolve_document_type():
    assert resolve_type(TestDocumentWithEntities) == TestDocumentWithEntities
    assert (
        resolve_type("tests.utils.test_hydra.TestDocumentWithEntities") == TestDocumentWithEntities
    )
    with pytest.raises(TypeError) as exc_info:
        resolve_type("tests.utils.test_hydra.test_resolve_document_type")
    assert str(exc_info.value).startswith(
        "type must be a subclass of None or a string that resolves to that, but got "
        "<function test_resolve_document_type"
    )

    assert (
        resolve_type(TestDocumentWithEntities, expected_super_type=TextBasedDocument)
        == TestDocumentWithEntities
    )
    with pytest.raises(TypeError) as exc_info:
        resolve_type(TestDocumentWithEntities, expected_super_type=TestDocumentWithSentences)
    assert (
        str(exc_info.value)
        == f"type must be a subclass of {TestDocumentWithSentences} or a string "
        f"that resolves to that, but got {TestDocumentWithEntities}"
    )
