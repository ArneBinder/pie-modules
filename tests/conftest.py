import dataclasses
import json

import pkg_resources
import pytest
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument

from tests import DUMP_FIXTURE_DATA, FIXTURES_ROOT

_TABULATE_AVAILABLE = "tabulate" in {pkg.key for pkg in pkg_resources.working_set}


@dataclasses.dataclass
class TestDocument(TextBasedDocument):
    sentences: AnnotationList[Span] = annotation_field(target="text")
    entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")


def example_to_doc_dict(example):
    doc = TestDocument(text=example["text"], id=example["id"])
    doc.metadata = dict(example["metadata"])
    sentences = [Span.fromdict(dct) for dct in example["sentences"]]
    entities = [LabeledSpan.fromdict(dct) for dct in example["entities"]]
    relations = [
        BinaryRelation(head=entities[rel["head"]], tail=entities[rel["tail"]], label=rel["label"])
        for rel in example["relations"]
    ]
    for sentence in sentences:
        doc.sentences.append(sentence)

    for entity in entities:
        doc.entities.append(entity)

    for relation in relations:
        doc.relations.append(relation)

    return doc.asdict()


SPLIT_SIZES = {"train": 6, "validation": 2, "test": 2}


@pytest.fixture(scope="module")
def document_dataset():
    result = {}
    for path in (FIXTURES_ROOT / "hf_datasets" / "json").iterdir():
        loaded_data = json.load(open(path))["data"]
        docs = [TestDocument.fromdict(example_to_doc_dict(ex)) for ex in loaded_data]
        result[path.stem] = docs
    return result


@pytest.fixture(scope="module")
def documents(document_dataset):
    return document_dataset["train"]


def test_documents(documents):
    assert len(documents) == 8
    assert all(isinstance(doc, TestDocument) for doc in documents)


def test_dont_dump_fixture_data():
    # this test is here to make sure that we don't dump the fixture on CI
    assert not DUMP_FIXTURE_DATA
