import json
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
from pie_modules.taskmodules import CrossTextBinaryCorefTaskModule
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
    file_name = (
        FIXTURES_ROOT / "taskmodules" / "cross_text_binary_coref" / "documents_with_negatives.json"
    )

    # result = list(add_negative_relations(positive_documents))
    # result_json = [doc.asdict() for doc in result]
    # with open(file_name, "w") as f:
    #    json.dump(result_json, f, indent=2)

    with open(file_name) as f:
        result_json = json.load(f)
    result = [
        TextPairDocumentWithLabeledSpansAndBinaryCorefRelations.fromdict(doc_json)
        for doc_json in result_json
    ]

    return result


@pytest.fixture(scope="module")
def task_encodings_without_target(taskmodule, documents_with_negatives):
    task_encodings = taskmodule.encode_input(documents_with_negatives[DOC_IDX_WITH_TASK_ENCODINGS])
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
    ]
    assert tokens_pair == [
        ["[CLS]", "En", "##ti", "##ty", "A", "works", "at", "B", ".", "[SEP]"],
        ["[CLS]", "En", "##ti", "##ty", "A", "works", "at", "B", ".", "[SEP]"],
    ]
    span_tokens = [
        toks[start:end]
        for toks, start, end in zip(
            tokens, inputs_dict["pooler_start_indices"], inputs_dict["pooler_end_indices"]
        )
    ]
    span_tokens_pair = [
        toks[start:end]
        for toks, start, end in zip(
            tokens_pair,
            inputs_dict["pooler_pair_start_indices"],
            inputs_dict["pooler_pair_end_indices"],
        )
    ]
    assert span_tokens == [["she"], ["C"]]
    assert span_tokens_pair == [["En", "##ti", "##ty", "A"], ["B"]]


def test_encode_target(task_encodings_without_target, taskmodule):
    targets = [
        taskmodule.encode_target(task_encoding) for task_encoding in task_encodings_without_target
    ]
    assert targets == [1.0, 0.0]


def test_encode_with_collect_statistics(taskmodule, positive_documents, caplog):
    documents_with_negatives = add_negative_coref_relations(positive_documents)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        original_values = taskmodule.collect_statistics
        taskmodule.collect_statistics = True
        taskmodule.encode(documents_with_negatives, encode_target=True)
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


def test_encode_with_windowing(documents_with_negatives, caplog):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = CrossTextBinaryCorefTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_window=4,
        collect_statistics=True,
    )
    assert not taskmodule.is_from_pretrained
    taskmodule.prepare(documents_with_negatives)

    assert len(documents_with_negatives) == 16
    caplog.clear()
    with caplog.at_level(logging.INFO):
        task_encodings = taskmodule.encode(documents_with_negatives)
    assert len(caplog.messages) > 0
    assert (
        caplog.messages[-1] == "statistics:\n"
        "|                                       |   coref |   no_relation |   all_relations |\n"
        "|:--------------------------------------|--------:|--------------:|----------------:|\n"
        "| available                             |       4 |             6 |               4 |\n"
        "| skipped_span_does_not_fit_into_window |       2 |             2 |               2 |\n"
        "| used                                  |       2 |             4 |               2 |\n"
        "| used %                                |      50 |            67 |              50 |"
    )

    assert len(task_encodings) == 6
    for task_encoding in task_encodings:
        for k, v in task_encoding.inputs["encoding"].items():
            assert len(v) <= taskmodule.max_window
        for k, v in task_encoding.inputs["encoding_pair"].items():
            assert len(v) <= taskmodule.max_window


@pytest.fixture(scope="module")
def task_encodings(taskmodule, documents_with_negatives):
    return taskmodule.encode(
        documents_with_negatives[DOC_IDX_WITH_TASK_ENCODINGS], encode_target=True
    )


@pytest.fixture(scope="module")
def batch(taskmodule, task_encodings):
    result = taskmodule.collate(task_encodings)
    return result


def test_collate(batch, taskmodule):
    assert batch is not None
    inputs, targets = batch
    assert inputs is not None
    assert set(inputs) == {
        "pooler_end_indices",
        "encoding_pair",
        "pooler_pair_end_indices",
        "pooler_start_indices",
        "encoding",
        "pooler_pair_start_indices",
    }
    torch.testing.assert_close(
        inputs["encoding"]["input_ids"],
        torch.tensor(
            [[101, 1262, 1131, 1771, 140, 119, 102], [101, 1262, 1131, 1771, 140, 119, 102]]
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
                [101, 13832, 3121, 2340, 138, 1759, 1120, 139, 119, 102],
                [101, 13832, 3121, 2340, 138, 1759, 1120, 139, 119, 102],
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

    torch.testing.assert_close(inputs["pooler_start_indices"], torch.tensor([[2], [4]]))
    torch.testing.assert_close(inputs["pooler_end_indices"], torch.tensor([[3], [5]]))
    torch.testing.assert_close(inputs["pooler_pair_start_indices"], torch.tensor([[1], [7]]))
    torch.testing.assert_close(inputs["pooler_pair_end_indices"], torch.tensor([[5], [8]]))

    torch.testing.assert_close(targets, {"scores": torch.tensor([1.0, 0.0])})


@pytest.fixture(scope="module")
def unbatched_output(taskmodule):
    model_output = {
        "scores": torch.tensor([0.5338148474693298, 0.9866107940673828]),
    }
    return taskmodule.unbatch_output(model_output=model_output)


def test_unbatch_output(unbatched_output, taskmodule):
    assert len(unbatched_output) == 2
    assert unbatched_output == [
        {"is_similar": False, "score": 0.5338148474693298},
        {"is_similar": True, "score": 0.9866107702255249},
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
        (0.9866, ("coref", (("COMPANY", "C"), ("COMPANY", "B")))),
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
            "discrete/f1_per_label/tp": tensor([0, 0]),
            "discrete/f1_per_label/fp": tensor([0, 0]),
            "discrete/f1_per_label/tn": tensor([0, 0]),
            "discrete/f1_per_label/fn": tensor([0, 0]),
            "discrete/macro/f1/tp": tensor([0, 0]),
            "discrete/macro/f1/fp": tensor([0, 0]),
            "discrete/macro/f1/tn": tensor([0, 0]),
            "discrete/macro/f1/fn": tensor([0, 0]),
            "discrete/micro/f1/tp": tensor([0]),
            "discrete/micro/f1/fp": tensor([0]),
            "discrete/micro/f1/tn": tensor([0]),
            "discrete/micro/f1/fn": tensor([0]),
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
            "discrete/f1_per_label/tp": tensor([3, 1]),
            "discrete/f1_per_label/fp": tensor([0, 0]),
            "discrete/f1_per_label/tn": tensor([1, 3]),
            "discrete/f1_per_label/fn": tensor([0, 0]),
            "discrete/macro/f1/tp": tensor([3, 1]),
            "discrete/macro/f1/fp": tensor([0, 0]),
            "discrete/macro/f1/tn": tensor([1, 3]),
            "discrete/macro/f1/fn": tensor([0, 0]),
            "discrete/micro/f1/tp": tensor([4]),
            "discrete/micro/f1/fp": tensor([0]),
            "discrete/micro/f1/tn": tensor([4]),
            "discrete/micro/f1/fn": tensor([0]),
        },
    )

    torch.testing.assert_close(
        metric.compute(),
        {
            "auroc": tensor(1.0),
            "no_relation/f1": tensor(1.0),
            "coref/f1": tensor(1.0),
            "macro/f1": tensor(1.0),
            "micro/f1": tensor(1.0),
        },
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
            "discrete/f1_per_label/tp": tensor([5, 1]),
            "discrete/f1_per_label/fp": tensor([1, 1]),
            "discrete/f1_per_label/tn": tensor([1, 5]),
            "discrete/f1_per_label/fn": tensor([1, 1]),
            "discrete/macro/f1/tp": tensor([5, 1]),
            "discrete/macro/f1/fp": tensor([1, 1]),
            "discrete/macro/f1/tn": tensor([1, 5]),
            "discrete/macro/f1/fn": tensor([1, 1]),
            "discrete/micro/f1/tp": tensor([6]),
            "discrete/micro/f1/fp": tensor([2]),
            "discrete/micro/f1/tn": tensor([6]),
            "discrete/micro/f1/fn": tensor([2]),
        },
    )

    torch.testing.assert_close(
        metric.compute(),
        {
            "auroc": tensor(0.916667),
            "no_relation/f1": tensor(0.833333),
            "coref/f1": tensor(0.500000),
            "macro/f1": tensor(0.666667),
            "micro/f1": tensor(0.750000),
        },
    )
