import dataclasses

import pkg_resources
import pytest
from datasets import load_dataset
from pytorch_ie import DatasetDict
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument

from tests import FIXTURES_ROOT

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


@pytest.fixture(scope="session")
def hf_dataset():
    result = load_dataset(
        "json",
        field="data",
        data_dir=str(FIXTURES_ROOT / "hf_datasets" / "json"),
    )

    return result


def test_hf_dataset(hf_dataset):
    assert hf_dataset is not None
    assert set(hf_dataset) == set(SPLIT_SIZES)
    for split in hf_dataset:
        assert len(hf_dataset[split]) == SPLIT_SIZES[split]


@pytest.fixture(scope="session")
def dataset(hf_dataset):
    mapped_dataset = hf_dataset.map(example_to_doc_dict)
    dataset = DatasetDict.from_hf(hf_dataset=mapped_dataset, document_type=TestDocument)
    return dataset


def test_dataset(dataset):
    assert dataset is not None
    assert set(dataset) == set(SPLIT_SIZES)
    for split in dataset:
        assert len(dataset[split]) == SPLIT_SIZES[split]
    # try getting a split
    d_train = dataset["train"]
    assert d_train is not None
    # try getting a document
    doc0 = d_train[0]
    assert doc0 is not None
    assert isinstance(doc0, TestDocument)
