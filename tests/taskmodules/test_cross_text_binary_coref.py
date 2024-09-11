from typing import Any, Dict, Union

import pytest
import torch.testing
from pytorch_ie.annotations import LabeledSpan
from torchmetrics import Metric, MetricCollection

from pie_modules.document.types import (
    BinaryCorefRelation,
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations,
)
from pie_modules.taskmodules import CrossTextBinaryCorefTaskModule
from pie_modules.utils import flatten_dict, list_of_dicts2dict_of_lists
from tests import _config_to_str

TOKENIZER_NAME_OR_PATH = "bert-base-cased"

CONFIGS = [
    {},
    # {"add_negative_relations": True},
]
CONFIGS_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIGS_DICT.keys())
def config(request):
    return CONFIGS_DICT[request.param]


@pytest.fixture(scope="module")
def positive_documents():
    doc1 = TextPairDocumentWithLabeledSpansAndBinaryCorefRelations(
        id="0", text="Entity A works at B.", text_pair="And she founded C."
    )
    doc1.labeled_spans.append(LabeledSpan(start=0, end=8, label="PERSON"))
    doc1.labeled_spans.append(LabeledSpan(start=18, end=19, label="COMPANY"))
    doc1.labeled_spans_pair.append(LabeledSpan(start=4, end=7, label="PERSON"))
    doc1.labeled_spans_pair.append(LabeledSpan(start=16, end=17, label="COMPANY"))
    doc1.binary_coref_relations.append(
        BinaryCorefRelation(head=doc1.labeled_spans[0], tail=doc1.labeled_spans_pair[0])
    )

    doc2 = TextPairDocumentWithLabeledSpansAndBinaryCorefRelations(
        id="0", text="Bob loves his cat.", text_pair="She sleeps a lot."
    )
    doc2.labeled_spans.append(LabeledSpan(start=0, end=3, label="PERSON"))
    doc2.labeled_spans.append(LabeledSpan(start=10, end=17, label="ANIMAL"))
    doc2.labeled_spans_pair.append(LabeledSpan(start=0, end=3, label="ANIMAL"))
    doc2.binary_coref_relations.append(
        BinaryCorefRelation(head=doc2.labeled_spans[1], tail=doc2.labeled_spans_pair[0])
    )

    return [doc1, doc2]


def test_positive_documents(positive_documents):
    assert len(positive_documents) == 2
    doc1, doc2 = positive_documents
    assert doc1.labeled_spans.resolve() == [("PERSON", "Entity A"), ("COMPANY", "B")]
    assert doc1.labeled_spans_pair.resolve() == [("PERSON", "she"), ("COMPANY", "C")]
    assert doc1.binary_coref_relations.resolve() == [
        ("coref", (("PERSON", "Entity A"), ("PERSON", "she")))
    ]

    assert doc2.labeled_spans.resolve() == [("PERSON", "Bob"), ("ANIMAL", "his cat")]
    assert doc2.labeled_spans_pair.resolve() == [("ANIMAL", "She")]
    assert doc2.binary_coref_relations.resolve() == [
        ("coref", (("ANIMAL", "his cat"), ("ANIMAL", "She")))
    ]


@pytest.fixture(scope="module")
def unprepared_taskmodule(config):
    taskmodule = CrossTextBinaryCorefTaskModule(
        tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH, **config
    )
    assert not taskmodule.is_from_pretrained

    return taskmodule


@pytest.fixture(scope="module")
def taskmodule(unprepared_taskmodule, positive_documents):
    unprepared_taskmodule.prepare(positive_documents)
    return unprepared_taskmodule


@pytest.fixture(scope="module")
def documents_with_negatives(taskmodule, positive_documents):
    return list(taskmodule._add_negative_relations(positive_documents))


def test_construct_negative_documents(positive_documents, documents_with_negatives):
    assert len(positive_documents) == 2
    TEXTS = [
        "Entity A works at B.",
        "And she founded C.",
        "Bob loves his cat.",
        "She sleeps a lot.",
    ]
    assert len(documents_with_negatives) == 12
    all_scores = [
        [coref_rel.score for coref_rel in doc.binary_coref_relations]
        for doc in documents_with_negatives
    ]
    assert documents_with_negatives[0].text == TEXTS[1]
    assert documents_with_negatives[0].text_pair == TEXTS[2]
    assert documents_with_negatives[0].binary_coref_relations.resolve() == [
        ("coref", (("PERSON", "she"), ("PERSON", "Bob"))),
        ("coref", (("PERSON", "she"), ("ANIMAL", "his cat"))),
        ("coref", (("COMPANY", "C"), ("PERSON", "Bob"))),
        ("coref", (("COMPANY", "C"), ("ANIMAL", "his cat"))),
    ]
    assert all_scores[0] == [0.0, 0.0, 0.0, 0.0]

    assert documents_with_negatives[1].text == TEXTS[1]
    assert documents_with_negatives[1].text_pair == TEXTS[0]
    assert documents_with_negatives[1].binary_coref_relations.resolve() == [
        ("coref", (("PERSON", "she"), ("PERSON", "Entity A"))),
        ("coref", (("PERSON", "she"), ("COMPANY", "B"))),
        ("coref", (("COMPANY", "C"), ("PERSON", "Entity A"))),
        ("coref", (("COMPANY", "C"), ("COMPANY", "B"))),
    ]
    assert all_scores[1] == [1.0, 0.0, 0.0, 0.0]

    assert documents_with_negatives[2].text == TEXTS[1]
    assert documents_with_negatives[2].text_pair == TEXTS[3]
    assert documents_with_negatives[2].binary_coref_relations.resolve() == [
        ("coref", (("PERSON", "she"), ("ANIMAL", "She"))),
        ("coref", (("COMPANY", "C"), ("ANIMAL", "She"))),
    ]
    assert all_scores[2] == [0.0, 0.0]

    assert documents_with_negatives[3].text == TEXTS[2]
    assert documents_with_negatives[3].text_pair == TEXTS[1]
    assert documents_with_negatives[3].binary_coref_relations.resolve() == [
        ("coref", (("PERSON", "Bob"), ("PERSON", "she"))),
        ("coref", (("PERSON", "Bob"), ("COMPANY", "C"))),
        ("coref", (("ANIMAL", "his cat"), ("PERSON", "she"))),
        ("coref", (("ANIMAL", "his cat"), ("COMPANY", "C"))),
    ]
    assert all_scores[3] == [0.0, 0.0, 0.0, 0.0]

    assert documents_with_negatives[4].text == TEXTS[2]
    assert documents_with_negatives[4].text_pair == TEXTS[0]
    assert documents_with_negatives[4].binary_coref_relations.resolve() == [
        ("coref", (("PERSON", "Bob"), ("PERSON", "Entity A"))),
        ("coref", (("PERSON", "Bob"), ("COMPANY", "B"))),
        ("coref", (("ANIMAL", "his cat"), ("PERSON", "Entity A"))),
        ("coref", (("ANIMAL", "his cat"), ("COMPANY", "B"))),
    ]
    assert all_scores[4] == [0.0, 0.0, 0.0, 0.0]

    assert documents_with_negatives[5].text == TEXTS[2]
    assert documents_with_negatives[5].text_pair == TEXTS[3]
    assert documents_with_negatives[5].binary_coref_relations.resolve() == [
        ("coref", (("PERSON", "Bob"), ("ANIMAL", "She"))),
        ("coref", (("ANIMAL", "his cat"), ("ANIMAL", "She"))),
    ]
    assert all_scores[5] == [0.0, 1.0]

    assert documents_with_negatives[6].text == TEXTS[0]
    assert documents_with_negatives[6].text_pair == TEXTS[1]
    assert documents_with_negatives[6].binary_coref_relations.resolve() == [
        ("coref", (("PERSON", "Entity A"), ("PERSON", "she"))),
        ("coref", (("PERSON", "Entity A"), ("COMPANY", "C"))),
        ("coref", (("COMPANY", "B"), ("PERSON", "she"))),
        ("coref", (("COMPANY", "B"), ("COMPANY", "C"))),
    ]
    assert all_scores[6] == [1.0, 0.0, 0.0, 0.0]

    assert documents_with_negatives[7].text == TEXTS[0]
    assert documents_with_negatives[7].text_pair == TEXTS[2]
    assert documents_with_negatives[7].binary_coref_relations.resolve() == [
        ("coref", (("PERSON", "Entity A"), ("PERSON", "Bob"))),
        ("coref", (("PERSON", "Entity A"), ("ANIMAL", "his cat"))),
        ("coref", (("COMPANY", "B"), ("PERSON", "Bob"))),
        ("coref", (("COMPANY", "B"), ("ANIMAL", "his cat"))),
    ]
    assert all_scores[7] == [0.0, 0.0, 0.0, 0.0]

    assert documents_with_negatives[8].text == TEXTS[0]
    assert documents_with_negatives[8].text_pair == TEXTS[3]
    assert documents_with_negatives[8].binary_coref_relations.resolve() == [
        ("coref", (("PERSON", "Entity A"), ("ANIMAL", "She"))),
        ("coref", (("COMPANY", "B"), ("ANIMAL", "She"))),
    ]
    assert all_scores[8] == [0.0, 0.0]

    assert documents_with_negatives[9].text == TEXTS[3]
    assert documents_with_negatives[9].text_pair == TEXTS[1]
    assert documents_with_negatives[9].binary_coref_relations.resolve() == [
        ("coref", (("ANIMAL", "She"), ("PERSON", "she"))),
        ("coref", (("ANIMAL", "She"), ("COMPANY", "C"))),
    ]
    assert all_scores[9] == [0.0, 0.0]

    assert documents_with_negatives[10].text == TEXTS[3]
    assert documents_with_negatives[10].text_pair == TEXTS[2]
    assert documents_with_negatives[10].binary_coref_relations.resolve() == [
        ("coref", (("ANIMAL", "She"), ("PERSON", "Bob"))),
        ("coref", (("ANIMAL", "She"), ("ANIMAL", "his cat"))),
    ]
    assert all_scores[10] == [0.0, 1.0]

    assert documents_with_negatives[11].text == TEXTS[3]
    assert documents_with_negatives[11].text_pair == TEXTS[0]
    assert documents_with_negatives[11].binary_coref_relations.resolve() == [
        ("coref", (("ANIMAL", "She"), ("PERSON", "Entity A"))),
        ("coref", (("ANIMAL", "She"), ("COMPANY", "B"))),
    ]
    assert all_scores[11] == [0.0, 0.0]


@pytest.fixture(scope="module")
def task_encodings_without_target(taskmodule, documents_with_negatives):
    task_encodings = taskmodule.encode_input(documents_with_negatives[0])
    return task_encodings


def test_encode_input(task_encodings_without_target, taskmodule):
    task_encodings = task_encodings_without_target
    convert_ids_to_tokens = taskmodule.tokenizer.convert_ids_to_tokens

    inputs_dict = list_of_dicts2dict_of_lists(
        [task_encoding.inputs for task_encoding in task_encodings]
    )
    tokens = [convert_ids_to_tokens(encoding["input_ids"]) for encoding in inputs_dict["encoding"]]
    tokens_pair = [
        convert_ids_to_tokens(encoding["input_ids"]) for encoding in inputs_dict["encoding_pair"]
    ]
    assert tokens == [
        ["[CLS]", "And", "she", "founded", "C", ".", "[SEP]"],
        ["[CLS]", "And", "she", "founded", "C", ".", "[SEP]"],
        ["[CLS]", "And", "she", "founded", "C", ".", "[SEP]"],
        ["[CLS]", "And", "she", "founded", "C", ".", "[SEP]"],
    ]
    assert tokens_pair == [
        ["[CLS]", "Bob", "loves", "his", "cat", ".", "[SEP]"],
        ["[CLS]", "Bob", "loves", "his", "cat", ".", "[SEP]"],
        ["[CLS]", "Bob", "loves", "his", "cat", ".", "[SEP]"],
        ["[CLS]", "Bob", "loves", "his", "cat", ".", "[SEP]"],
    ]
    span_tokens = [
        toks[start:end]
        for toks, start, end in zip(tokens, inputs_dict["start"], inputs_dict["end"])
    ]
    span_tokens_pair = [
        toks[start:end]
        for toks, start, end in zip(
            tokens_pair, inputs_dict["start_pair"], inputs_dict["end_pair"]
        )
    ]
    assert span_tokens == [["she"], ["she"], ["C"], ["C"]]
    assert span_tokens_pair == [["Bob"], ["his", "cat"], ["Bob"], ["his", "cat"]]


def test_encode_target(task_encodings_without_target, taskmodule):
    target = taskmodule.encode_target(task_encodings_without_target[0])
    assert target == 0.0


@pytest.fixture(scope="module", params=[False, True])
def batch(taskmodule, positive_documents, documents_with_negatives, request):
    if request.param:
        original_value = taskmodule.add_negative_relations
        taskmodule.add_negative_relations = True
        task_encodings = taskmodule.encode(positive_documents, encode_target=True)[:4]
        taskmodule.add_negative_relations = original_value
    else:
        task_encodings = taskmodule.encode(documents_with_negatives[0], encode_target=True)
    result = taskmodule.collate(task_encodings)
    return result


def test_collate(batch, taskmodule):
    assert batch is not None
    inputs, targets = batch
    assert inputs is not None
    assert set(inputs) == {"encoding", "encoding_pair", "start", "end", "start_pair", "end_pair"}
    torch.testing.assert_close(
        inputs["encoding"]["input_ids"],
        torch.tensor(
            [
                [101, 1262, 1131, 1771, 140, 119, 102],
                [101, 1262, 1131, 1771, 140, 119, 102],
                [101, 1262, 1131, 1771, 140, 119, 102],
                [101, 1262, 1131, 1771, 140, 119, 102],
            ]
        ),
    )
    torch.testing.assert_close(
        inputs["encoding"]["token_type_ids"], torch.zeros_like(inputs["encoding"]["input_ids"])
    )
    torch.testing.assert_close(
        inputs["encoding"]["attention_mask"], torch.ones_like(inputs["encoding"]["input_ids"])
    )

    torch.testing.assert_close(
        inputs["encoding_pair"]["input_ids"],
        torch.tensor(
            [
                [101, 3162, 7871, 1117, 5855, 119, 102],
                [101, 3162, 7871, 1117, 5855, 119, 102],
                [101, 3162, 7871, 1117, 5855, 119, 102],
                [101, 3162, 7871, 1117, 5855, 119, 102],
            ]
        ),
    )
    torch.testing.assert_close(
        inputs["encoding_pair"]["token_type_ids"],
        torch.zeros_like(inputs["encoding_pair"]["input_ids"]),
    )
    torch.testing.assert_close(
        inputs["encoding_pair"]["attention_mask"],
        torch.ones_like(inputs["encoding_pair"]["input_ids"]),
    )

    torch.testing.assert_close(inputs["start"], torch.tensor([2, 2, 4, 4]))
    torch.testing.assert_close(inputs["end"], torch.tensor([3, 3, 5, 5]))
    torch.testing.assert_close(inputs["start_pair"], torch.tensor([1, 3, 1, 3]))
    torch.testing.assert_close(inputs["end_pair"], torch.tensor([2, 5, 2, 5]))

    torch.testing.assert_close(targets, torch.tensor([0.0, 0.0, 0.0, 0.0]))


def get_metric_state(metric_or_collection: Union[Metric, MetricCollection]) -> Dict[str, Any]:
    if isinstance(metric_or_collection, Metric):
        return {
            k: [vv.tolist() for vv in v]
            for k, v in flatten_dict(metric_or_collection.metric_state).items()
        }
    elif isinstance(metric_or_collection, MetricCollection):
        return flatten_dict({k: get_metric_state(v) for k, v in metric_or_collection.items()})
    else:
        raise ValueError(f"unsupported type: {type(metric_or_collection)}")


def test_configure_metric(taskmodule, batch):
    metric = taskmodule.configure_model_metric(stage="train")

    assert isinstance(metric, (Metric, MetricCollection))
    state = get_metric_state(metric)
    assert state == {"auroc/preds": [], "auroc/target": []}

    # targets = batch[1]
    targets = torch.tensor([0.0, 1.0, 0.0, 0.0])
    metric.update(targets, targets)

    state = get_metric_state(metric)
    assert state == {"auroc/preds": [[0.0, 1.0, 0.0, 0.0]], "auroc/target": [[0.0, 1.0, 0.0, 0.0]]}

    assert metric.compute() == {"auroc": torch.tensor(1.0)}

    # torch.rand_like(targets)
    random_targets = torch.tensor([0.2703, 0.6812, 0.2582, 0.8030])
    metric.update(random_targets, targets)
    state = get_metric_state(metric)
    assert state == {
        "auroc/preds": [
            [0.0, 1.0, 0.0, 0.0],
            [0.2703000009059906, 0.6812000274658203, 0.2581999897956848, 0.8029999732971191],
        ],
        "auroc/target": [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    }

    assert metric.compute() == {"auroc": torch.tensor(0.9166666269302368)}
