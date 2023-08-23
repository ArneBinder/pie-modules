import dataclasses

import pytest
from datasets import load_dataset
from pytorch_ie import DatasetDict
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import AnnotationList, annotation_field
from pytorch_ie.documents import TextBasedDocument

from tests import FIXTURES_ROOT


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


@pytest.fixture(scope="session")
def dataset():
    hf_dataset = load_dataset(
        "json",
        field="data",
        data_dir=str(FIXTURES_ROOT / "hf_datasets" / "json"),
    )
    mapped_dataset = hf_dataset.map(example_to_doc_dict)
    dataset = DatasetDict.from_hf(hf_dataset=mapped_dataset, document_type=TestDocument)
    assert dataset is not None
    # try getting a split
    d_train = dataset["train"]
    assert d_train is not None
    # try getting a document
    doc0 = d_train[0]
    assert doc0 is not None
    assert isinstance(doc0, TestDocument)
    return dataset
