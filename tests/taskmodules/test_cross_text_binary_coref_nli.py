import logging
from typing import Any, Dict, Union

import pytest
import torch.testing
from pytorch_ie.annotations import LabeledSpan
from torch import tensor
from torchmetrics import Metric, MetricCollection

from pie_modules.document.processing.text_pair import add_negative_coref_relations
from pie_modules.documents import (
    BinaryCorefRelation,
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations,
)
from pie_modules.taskmodules import CrossTextBinaryCorefTaskModuleByNli
from pie_modules.utils import flatten_dict, list_of_dicts2dict_of_lists
from tests import FIXTURES_ROOT, _config_to_str

TOKENIZER_NAME_OR_PATH = "bert-base-cased"
DOC_IDX_WITH_TASK_ENCODINGS = 2

CONFIGS = [
    {},
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
    taskmodule = CrossTextBinaryCorefTaskModuleByNli(
        tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH,
        labels=["entailment", "neutral", "contradiction"],
        entailment_label="entailment",
        **config,
    )
    assert not taskmodule.is_from_pretrained

    return taskmodule


@pytest.fixture(scope="module")
def taskmodule(unprepared_taskmodule, positive_documents):
    unprepared_taskmodule.prepare(positive_documents)
    return unprepared_taskmodule


@pytest.fixture(scope="module")
def task_encodings_without_target(taskmodule, positive_documents):
    task_encodings = taskmodule.encode_input(positive_documents[0])
    return task_encodings


def test_encode_input(task_encodings_without_target, taskmodule):
    task_encodings = task_encodings_without_target
    convert_ids_to_tokens = taskmodule.tokenizer.convert_ids_to_tokens
    assert len(task_encodings) == 1
    task_encoding = task_encodings[0]
    tokens = [convert_ids_to_tokens(input_ids) for input_ids in task_encoding.inputs["input_ids"]]
    assert len(tokens) == 2
    assert tokens[0] == ["[CLS]", "En", "##ti", "##ty", "A", "[SEP]", "she", "[SEP]"]
    assert tokens[1] == ["[CLS]", "she", "[SEP]", "En", "##ti", "##ty", "A", "[SEP]"]


def test_encode_with_collect_statistics(taskmodule, positive_documents, caplog):
    documents_with_negatives = add_negative_coref_relations(positive_documents)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        original_values = taskmodule.collect_statistics
        taskmodule.collect_statistics = True
        taskmodule.encode(documents_with_negatives, encode_target=False)
        taskmodule.collect_statistics = original_values

    assert len(caplog.messages) == 1
    assert (
        caplog.messages[0] == "statistics:\n"
        "|           |   coref |   no_relation |   all_relations |\n"
        "|:----------|--------:|--------------:|----------------:|\n"
        "| available |       4 |             6 |               4 |\n"
        "| used      |       4 |             6 |               4 |\n"
        "| used %    |     100 |           100 |             100 |"
    )


@pytest.fixture(scope="module")
def task_encodings(taskmodule, positive_documents):
    return taskmodule.encode(positive_documents, encode_target=False)


@pytest.fixture(scope="module")
def batch(taskmodule, task_encodings):
    result = taskmodule.collate(task_encodings)
    return result


def test_collate(batch, taskmodule):
    assert batch is not None
    inputs, targets = batch
    assert inputs is not None
    assert set(inputs) == {
        "input_ids",
        "token_type_ids",
        "attention_mask",
    }
    torch.testing.assert_close(
        inputs["input_ids"],
        torch.tensor(
            [
                [101, 13832, 3121, 2340, 138, 102, 1131, 102],
                [101, 1131, 102, 13832, 3121, 2340, 138, 102],
                [101, 1117, 5855, 102, 1153, 102, 0, 0],
                [101, 1153, 102, 1117, 5855, 102, 0, 0],
            ]
        ),
    )
    torch.testing.assert_close(
        inputs["token_type_ids"],
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0],
            ]
        ),
    )
    torch.testing.assert_close(
        inputs["attention_mask"],
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
            ]
        ),
    )

    assert targets is None


@pytest.fixture(scope="module")
def model_output():
    return {
        "labels": torch.tensor([0, 0, 1, 2]),
        "probabilities": torch.tensor(
            [
                # O, org:founded_by, per:employee_of, per:founder
                [0.4, 0.2, 0.3],
                [0.6, 0.1, 0.1],
                [0.2, 0.5, 0.2],
                [0.2, 0.1, 0.6],
            ]
        ),
    }


@pytest.fixture(scope="module")
def unbatched_output(taskmodule, model_output):
    return taskmodule.unbatch_output(model_output=model_output)


def test_unbatch_output(unbatched_output, taskmodule):
    assert len(unbatched_output) == 2
    assert unbatched_output == [
        {
            "entailment_probability_pair": (0.4000000059604645, 0.6000000238418579),
            "label_pair": ("entailment", "entailment"),
        },
        {
            "entailment_probability_pair": (0.20000000298023224, 0.20000000298023224),
            "label_pair": ("neutral", "contradiction"),
        },
    ]


def test_create_annotation_from_output(taskmodule, task_encodings, unbatched_output):
    all_new_annotations = []
    for task_encoding, task_output in zip(task_encodings, unbatched_output):
        for new_annotation in taskmodule.create_annotations_from_output(
            task_encoding=task_encoding, task_output=task_output
        ):
            all_new_annotations.append(new_annotation)
    assert all(layer_name == "binary_coref_relations" for layer_name, ann in all_new_annotations)
    resolve_annotations_with_scores = [
        (round(ann.score, 4), ann.resolve()) for layer_name, ann in all_new_annotations
    ]
    assert resolve_annotations_with_scores == [
        (0.5, ("coref", (("PERSON", "Entity A"), ("PERSON", "she"))))
    ]


def get_metric_state(metric_or_collection: Union[Metric, MetricCollection]) -> Dict[str, Any]:
    if isinstance(metric_or_collection, Metric):
        return flatten_dict(metric_or_collection.metric_state)
    elif isinstance(metric_or_collection, MetricCollection):
        return flatten_dict({k: get_metric_state(v) for k, v in metric_or_collection.items()})
    else:
        raise ValueError(f"unsupported type: {type(metric_or_collection)}")


def test_configure_metric(taskmodule, batch):
    metric = taskmodule.configure_model_metric(stage="train")

    assert isinstance(metric, (Metric, MetricCollection))
    state = get_metric_state(metric)
    torch.testing.assert_close(
        state,
        {
            "continuous/auroc/preds": [],
            "continuous/auroc/target": [],
            "continuous/avg-P/preds": [],
            "continuous/avg-P/target": [],
            "continuous/f1/fn": tensor([0]),
            "continuous/f1/fp": tensor([0]),
            "continuous/f1/tn": tensor([0]),
            "continuous/f1/tp": tensor([0]),
        },
    )

    # targets = batch[1]
    targets = {
        "scores": torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }
    metric.update(targets, targets)

    state = get_metric_state(metric)
    torch.testing.assert_close(
        state,
        {
            "continuous/auroc/preds": [tensor([0.0, 1.0, 0.0, 0.0])],
            "continuous/auroc/target": [tensor([0.0, 1.0, 0.0, 0.0])],
            "continuous/avg-P/preds": [tensor([0.0, 1.0, 0.0, 0.0])],
            "continuous/avg-P/target": [tensor([0.0, 1.0, 0.0, 0.0])],
            "continuous/f1/tp": tensor([1]),
            "continuous/f1/fp": tensor([0]),
            "continuous/f1/tn": tensor([3]),
            "continuous/f1/fn": tensor([0]),
        },
    )

    torch.testing.assert_close(
        metric.compute(),
        {"auroc": tensor(1.0), "avg-P": tensor(1.0), "f1": tensor(1.0)},
    )

    # torch.rand_like(targets)
    random_targets = {
        "scores": torch.tensor([0.2703, 0.6812, 0.2582, 0.9030]),
    }
    metric.update(random_targets, targets)
    state = get_metric_state(metric)
    torch.testing.assert_close(
        state,
        {
            "continuous/auroc/preds": [
                tensor([0.0, 1.0, 0.0, 0.0]),
                tensor([0.2703, 0.6812, 0.2582, 0.9030]),
            ],
            "continuous/auroc/target": [
                tensor([0.0, 1.0, 0.0, 0.0]),
                tensor([0.0, 1.0, 0.0, 0.0]),
            ],
            "continuous/avg-P/preds": [
                tensor([0.0, 1.0, 0.0, 0.0]),
                tensor([0.2703, 0.6812, 0.2582, 0.9030]),
            ],
            "continuous/avg-P/target": [
                tensor([0.0, 1.0, 0.0, 0.0]),
                tensor([0.0, 1.0, 0.0, 0.0]),
            ],
            "continuous/f1/tp": tensor([1]),
            "continuous/f1/fp": tensor([1]),
            "continuous/f1/tn": tensor([5]),
            "continuous/f1/fn": tensor([1]),
        },
    )

    torch.testing.assert_close(
        metric.compute(),
        {"auroc": tensor(0.91666663), "avg-P": tensor(0.83333337), "f1": tensor(0.50000000)},
    )
