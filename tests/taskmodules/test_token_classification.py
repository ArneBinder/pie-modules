import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import LabeledSpan
from pytorch_ie.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansAndLabeledPartitions,
)
from transformers import BatchEncoding

from pie_modules.taskmodules import TokenClassificationTaskModule


def _config_to_str(cfg: Dict[str, Any]) -> str:
    # Converts a configuration dictionary to a string representation
    result = "-".join([f"{k}={cfg[k]}" for k in sorted(cfg)])
    return result


CONFIGS: List[Dict[str, Any]] = [
    {},
    {"max_window": 8},
    {"max_window": 8, "window_overlap": 2},
    {"partition_annotation": "sentences"},
]

CONFIGS_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIGS_DICT.keys())
def config(request):
    """
    - Provides clean and readable test configurations.
    - Yields config dictionaries from the CONFIGS list to produce clean test case identifiers.

    """
    return CONFIGS_DICT[request.param]


@pytest.fixture(scope="module")
def config_str(config):
    # Fixture returning a string representation of the config
    return _config_to_str(config)


@pytest.fixture(scope="module")
def unprepared_taskmodule(config):
    """
    - Prepares a task module with the specified tokenizer and configuration.
    - Sets up the task module with a unprepared state for testing purposes.

    """
    return TokenClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased", span_annotation="entities", **config
    )


@dataclass
class ExampleDocument(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    sentences: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@pytest.fixture(scope="module")
def documents():
    """
    - Creates example documents with predefined texts.
    - Assigns labels to the documents for testing purposes.

    """
    doc1 = ExampleDocument(text="Mount Everest is the highest peak in the world.", id="doc1")
    doc1.entities.append(LabeledSpan(start=0, end=13, label="head"))
    assert str(doc1.entities[0]) == "Mount Everest"
    doc2 = ExampleDocument(text="Alice loves reading books. Bob enjoys playing soccer.", id="doc2")
    doc2.entities.append(LabeledSpan(start=0, end=5, label="head"))
    assert str(doc2.entities[0]) == "Alice"
    doc2.sentences.append(LabeledSpan(start=27, end=53, label="sentence"))
    assert str(doc2.sentences[0]) == "Bob enjoys playing soccer."
    return [doc1, doc2]


def test_taskmodule(unprepared_taskmodule):
    assert unprepared_taskmodule is not None


@pytest.fixture(scope="module")
def taskmodule(unprepared_taskmodule, documents):
    """
    - Prepares the task module with the given documents, i.e. collect available label values.
    - Calls the necessary methods to prepare the task module with the documents.
    - Calls _prepare(documents) and then _post_prepare()

    """
    unprepared_taskmodule.prepare(documents)
    return unprepared_taskmodule


def test_prepare(taskmodule):
    assert taskmodule is not None
    assert taskmodule.is_prepared
    assert taskmodule.label_to_id == {"O": 0, "B-head": 1, "I-head": 2}
    assert taskmodule.id_to_label == {0: "O", 1: "B-head", 2: "I-head"}


def test_config(taskmodule):
    config = taskmodule._config()
    assert config["taskmodule_type"] == "TokenClassificationTaskModule"
    assert "labels" in config
    assert config["labels"] == ["head"]


@pytest.fixture(scope="module")
def task_encodings_without_targets(taskmodule, documents):
    """
    - Generates task encodings for all the documents, but without associated targets.
    """
    return taskmodule.encode(documents, encode_target=False)


def test_task_encodings_without_targets(task_encodings_without_targets, taskmodule, config):
    tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(task_encoding.inputs.ids)
        for task_encoding in task_encodings_without_targets
    ]

    # If config is empty
    if config == {}:
        assert tokens == [
            [
                "[CLS]",
                "mount",
                "everest",
                "is",
                "the",
                "highest",
                "peak",
                "in",
                "the",
                "world",
                ".",
                "[SEP]",
            ],
            [
                "[CLS]",
                "alice",
                "loves",
                "reading",
                "books",
                ".",
                "bob",
                "enjoys",
                "playing",
                "soccer",
                ".",
                "[SEP]",
            ],
        ]

    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == {"max_window": 8, "window_overlap": 2}:
        for t in tokens:
            assert len(t) <= 8

        assert tokens == [
            ["[CLS]", "mount", "everest", "is", "the", "highest", "peak", "[SEP]"],
            ["[CLS]", "highest", "peak", "in", "the", "world", ".", "[SEP]"],
            ["[CLS]", "alice", "loves", "reading", "books", ".", "bob", "[SEP]"],
            ["[CLS]", ".", "bob", "enjoys", "playing", "soccer", ".", "[SEP]"],
        ]

    # If config has the specified value (max_window=8)
    elif config == {"max_window": 8}:
        for t in tokens:
            assert len(t) <= 8

        assert tokens == [
            ["[CLS]", "mount", "everest", "is", "the", "highest", "peak", "[SEP]"],
            ["[CLS]", "in", "the", "world", ".", "[SEP]"],
            ["[CLS]", "alice", "loves", "reading", "books", ".", "bob", "[SEP]"],
            ["[CLS]", "enjoys", "playing", "soccer", ".", "[SEP]"],
        ]

    # If config has the specified value (partition_annotation=sentences)
    elif config == {"partition_annotation": "sentences"}:
        assert tokens

    else:
        raise ValueError(f"unknown config: {config}")


@pytest.fixture(scope="module")
def task_encodings(taskmodule, documents):
    return taskmodule.encode(documents, encode_target=True)


def test_task_encodings(task_encodings, taskmodule, config):
    tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(task_encoding.inputs.ids)
        for task_encoding in task_encodings
    ]
    labels_tokens = [
        [taskmodule.id_to_label[x] if x != -100 else "<pad>" for x in task_encoding.targets]
        for task_encoding in task_encodings
    ]
    assert len(labels_tokens) == len(tokens)

    tokens_with_labels = list(zip(tokens, labels_tokens))

    for tokens, labels in tokens_with_labels:
        assert len(tokens) == len(labels)

    # If config is empty
    if config == {}:
        assert tokens_with_labels == [
            (
                [
                    "[CLS]",
                    "mount",
                    "everest",
                    "is",
                    "the",
                    "highest",
                    "peak",
                    "in",
                    "the",
                    "world",
                    ".",
                    "[SEP]",
                ],
                ["<pad>", "B-head", "I-head", "O", "O", "O", "O", "O", "O", "O", "O", "<pad>"],
            ),
            (
                [
                    "[CLS]",
                    "alice",
                    "loves",
                    "reading",
                    "books",
                    ".",
                    "bob",
                    "enjoys",
                    "playing",
                    "soccer",
                    ".",
                    "[SEP]",
                ],
                ["<pad>", "B-head", "O", "O", "O", "O", "O", "O", "O", "O", "O", "<pad>"],
            ),
        ]

    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == {"max_window": 8, "window_overlap": 2}:
        for tokens, labels in tokens_with_labels:
            assert len(tokens) <= 8

        assert tokens_with_labels == [
            (
                ["[CLS]", "mount", "everest", "is", "the", "highest", "peak", "[SEP]"],
                ["<pad>", "B-head", "I-head", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "highest", "peak", "in", "the", "world", ".", "[SEP]"],
                ["<pad>", "O", "O", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "alice", "loves", "reading", "books", ".", "bob", "[SEP]"],
                ["<pad>", "B-head", "O", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", ".", "bob", "enjoys", "playing", "soccer", ".", "[SEP]"],
                ["<pad>", "O", "O", "O", "O", "O", "O", "<pad>"],
            ),
        ]

    # If config has the specified value (max_window=8)
    elif config == {"max_window": 8}:
        for tokens, labels in tokens_with_labels:
            assert len(tokens) <= 8

        assert tokens_with_labels == [
            (
                ["[CLS]", "mount", "everest", "is", "the", "highest", "peak", "[SEP]"],
                ["<pad>", "B-head", "I-head", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "in", "the", "world", ".", "[SEP]"],
                ["<pad>", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "alice", "loves", "reading", "books", ".", "bob", "[SEP]"],
                ["<pad>", "B-head", "O", "O", "O", "O", "O", "<pad>"],
            ),
            (
                ["[CLS]", "enjoys", "playing", "soccer", ".", "[SEP]"],
                ["<pad>", "O", "O", "O", "O", "<pad>"],
            ),
        ]

    # If config has the specified value (partition_annotation=sentences)
    elif config == {"partition_annotation": "sentences"}:
        assert tokens_with_labels == [
            (
                ["[CLS]", "bob", "enjoys", "playing", "soccer", ".", "[SEP]"],
                ["<pad>", "O", "O", "O", "O", "O", "<pad>"],
            )
        ]

    else:
        raise ValueError(f"unknown config: {config}")


@pytest.fixture(scope="module")
def task_encodings_for_batch(task_encodings, config):
    task_encodings_by_document = defaultdict(list)
    for task_encoding in task_encodings:
        task_encodings_by_document[task_encoding.document.id].append(task_encoding)

    if "partition_annotation" in config:
        return task_encodings_by_document["doc2"][0], task_encodings_by_document["doc2"][0]
    else:
        return task_encodings_by_document["doc1"][0], task_encodings_by_document["doc2"][0]


@pytest.fixture(scope="module")
def batch(taskmodule, task_encodings_for_batch, config):
    return taskmodule.collate(task_encodings_for_batch)


def test_collate(batch, config):
    """
    - Test the collate function that creates batch encodings based on the specified configuration.

    - Parameters:
        batch (tuple): A tuple containing the batch encoding and other metadata.
        config (dict): A dictionary containing configuration settings for the collation.
    """
    assert batch is not None
    assert len(batch) == 2
    batch_encoding, _ = batch

    # If config is empty
    if config == {}:
        input_ids_expected = torch.tensor(
            [
                [101, 4057, 23914, 2003, 1996, 3284, 4672, 1999, 1996, 2088, 1012, 102],
                [101, 5650, 7459, 3752, 2808, 1012, 3960, 15646, 2652, 4715, 1012, 102],
            ],
            dtype=torch.int64,
        )
        attention_mask_expected = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.int64,
        )

        encoding_expected = BatchEncoding(
            data={
                "input_ids": input_ids_expected,
                "attention_mask": attention_mask_expected,
            }
        )
        torch.testing.assert_close(batch_encoding.input_ids, encoding_expected.input_ids)
        torch.testing.assert_close(batch_encoding.attention_mask, encoding_expected.attention_mask)

    # If config has the specified values (max_window=8, window_overlap=2)
    elif config == {"max_window": 8, "window_overlap": 2}:
        input_ids_expected = torch.tensor(
            [
                [101, 4057, 23914, 2003, 1996, 3284, 4672, 102],
                [101, 5650, 7459, 3752, 2808, 1012, 3960, 102],
            ],
            dtype=torch.int64,
        )
        attention_mask_expected = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64
        )
        encoding_expected = BatchEncoding(
            data={
                "input_ids": input_ids_expected,
                "attention_mask": attention_mask_expected,
            }
        )
        torch.testing.assert_close(batch_encoding.input_ids, encoding_expected.input_ids)
        torch.testing.assert_close(batch_encoding.attention_mask, encoding_expected.attention_mask)

    # If config has the specified values (max_window=8)
    elif config == {"max_window": 8}:
        input_ids_expected = torch.tensor(
            [
                [101, 4057, 23914, 2003, 1996, 3284, 4672, 102],
                [101, 5650, 7459, 3752, 2808, 1012, 3960, 102],
            ],
            dtype=torch.int64,
        )
        attention_mask_expected = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64
        )
        encoding_expected = BatchEncoding(
            data={
                "input_ids": input_ids_expected,
                "attention_mask": attention_mask_expected,
            }
        )
        torch.testing.assert_close(batch_encoding.input_ids, encoding_expected.input_ids)
        torch.testing.assert_close(batch_encoding.attention_mask, encoding_expected.attention_mask)

    # If config has the specified value (partition_annotation=sentences)
    elif config == {"partition_annotation": "sentences"}:
        input_ids_expected = torch.tensor(
            [[101, 3960, 15646, 2652, 4715, 1012, 102], [101, 3960, 15646, 2652, 4715, 1012, 102]],
            dtype=torch.int64,
        )
        token_type_ids_expected = torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.int64
        )
        attention_mask_expected = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.int64
        )
        encoding_expected = BatchEncoding(
            data={
                "input_ids": input_ids_expected,
                "attention_mask": attention_mask_expected,
            }
        )

        torch.testing.assert_close(batch_encoding.input_ids, encoding_expected.input_ids)
        torch.testing.assert_close(batch_encoding.attention_mask, encoding_expected.attention_mask)
    else:
        raise ValueError(f"unknown config: {config}")

    assert set(batch_encoding.data) == set(encoding_expected.data)


# This is not used, but can be used to create a batch of task encodings with targets for the unbatched_outputs fixture.
@pytest.fixture(scope="module")
def real_model_output(batch, taskmodule):
    from pytorch_ie.models import TransformerTokenClassificationModel

    model = TransformerTokenClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=len(taskmodule.label_to_id),
    )
    inputs, targets = batch
    result = model(inputs)
    return result


@pytest.fixture(scope="module")
def model_output(config, batch):
    # create "perfect" output from targets
    targets = batch[1].clone()
    targets[targets == -100] = 0
    one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=3).float() * 0.99 + 0.005
    # convert to logits (logit = log(p/(1-p)))
    logits = torch.log(one_hot_targets / (1 - one_hot_targets))
    return {"logits": logits}


@pytest.fixture(scope="module")
def unbatched_outputs(taskmodule, model_output):
    return taskmodule.unbatch_output(model_output)


def test_unbatched_output(unbatched_outputs, config):
    """
    - Test the unbatched outputs generated by the model.

    - Parameters:
        unbatched_outputs (list): List of unbatched outputs from the model.
        config (dict): The configuration to check different cases.

    - Perform assertions for each unbatched output based on the given configuration.
    """
    assert unbatched_outputs is not None
    assert len(unbatched_outputs) == 2

    # Based on the config, perform assertions for each unbatched output
    if config == {}:
        # Assertions for the first unbatched output
        assert unbatched_outputs[0]["tags"] == [
            "O",
            "B-head",
            "I-head",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
        ]
        np.testing.assert_almost_equal(
            unbatched_outputs[0]["probabilities"].round(4),
            np.array(
                [
                    [0.9999, 0.0, 0.0],
                    [0.0, 0.9999, 0.0],
                    [0.0, 0.0, 0.9999],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        # Assertions for the second unbatched output
        assert unbatched_outputs[1]["tags"] == [
            "O",
            "B-head",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
        ]
        np.testing.assert_almost_equal(
            unbatched_outputs[1]["probabilities"].round(4),
            np.array(
                [
                    [0.9999, 0.0, 0.0],
                    [0.0, 0.9999, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

    elif config == {"max_window": 8, "window_overlap": 2}:
        # Assertions for the first unbatched output
        assert unbatched_outputs[0]["tags"] == ["O", "B-head", "I-head", "O", "O", "O", "O", "O"]
        np.testing.assert_almost_equal(
            unbatched_outputs[0]["probabilities"].round(4),
            np.array(
                [
                    [0.9999, 0.0, 0.0],
                    [0.0, 0.9999, 0.0],
                    [0.0, 0.0, 0.9999],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        # Assertions for the second unbatched output
        assert unbatched_outputs[1]["tags"] == ["O", "B-head", "O", "O", "O", "O", "O", "O"]
        np.testing.assert_almost_equal(
            unbatched_outputs[1]["probabilities"].round(4),
            np.array(
                [
                    [0.9999, 0.0, 0.0],
                    [0.0, 0.9999, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

    elif config == {"max_window": 8}:
        # Assertions for the first unbatched output
        assert unbatched_outputs[0]["tags"] == ["O", "B-head", "I-head", "O", "O", "O", "O", "O"]
        np.testing.assert_almost_equal(
            unbatched_outputs[0]["probabilities"].round(4),
            np.array(
                [
                    [0.9999, 0.0, 0.0],
                    [0.0, 0.9999, 0.0],
                    [0.0, 0.0, 0.9999],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

        # Assertions for the second unbatched output
        assert unbatched_outputs[1]["tags"] == ["O", "B-head", "O", "O", "O", "O", "O", "O"]
        np.testing.assert_almost_equal(
            unbatched_outputs[1]["probabilities"].round(4),
            np.array(
                [
                    [0.9999, 0.0, 0.0],
                    [0.0, 0.9999, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

    elif config == {"partition_annotation": "sentences"}:
        # Assertions for the first unbatched output
        assert unbatched_outputs[0]["tags"] == ["O", "O", "O", "O", "O", "O", "O"]
        np.testing.assert_almost_equal(
            unbatched_outputs[0]["probabilities"].round(4),
            np.array(
                [
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

        # Assertions for the second unbatched output
        assert unbatched_outputs[1]["tags"] == ["O", "O", "O", "O", "O", "O", "O"]
        np.testing.assert_almost_equal(
            unbatched_outputs[1]["probabilities"].round(4),
            np.array(
                [
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                    [0.9999, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

    else:
        raise ValueError(f"unknown config: {config}")


@pytest.fixture(scope="module")
def annotations_from_output(taskmodule, task_encodings_for_batch, unbatched_outputs, config):
    """
    - Converts the inputs (task_encoding_without_targets) and the respective model outputs (unbatched_outputs)
    into human-readable  annotations.

    """

    named_annotations_per_document = defaultdict(list)
    for task_encoding, task_output in zip(task_encodings_for_batch, unbatched_outputs):
        annotations = taskmodule.create_annotations_from_output(task_encoding, task_output)
        named_annotations_per_document[task_encoding.document.id].extend(list(annotations))
    return named_annotations_per_document


def test_annotations_from_output(annotations_from_output, config, documents):
    """
    - Test the annotations generated from the output.

    - Parameters:
        annotations_from_output (list): List of annotations from the model output.
        config (dict): The configuration to check different cases.

    - For each configuration, check the first two entries from annotations_from_output for both documents.
    """
    assert annotations_from_output is not None  # Check that annotations_from_output is not None
    # Sort the annotations in each document by start and end positions
    annotations_from_output = {
        doc_id: sorted(annotations, key=lambda x: (x[0], x[1].start, x[1].end))
        for doc_id, annotations in annotations_from_output.items()
    }
    documents_by_id = {doc.id: doc for doc in documents}
    documents_with_annotations = []
    resolved_annotations = defaultdict(list)
    # Check that the number of annotations is correct
    for doc_id, layer_names_and_annotations in annotations_from_output.items():
        new_doc = documents_by_id[doc_id].copy()
        for layer_name, annotation in layer_names_and_annotations:
            assert layer_name == "entities"
            assert isinstance(annotation, LabeledSpan)
            new_doc.entities.predictions.append(annotation)
            resolved_annotations[doc_id].append(str(annotation))
        documents_with_annotations.append(new_doc)

    resolved_annotations = dict(resolved_annotations)
    # Check based on the config
    if config == {}:
        # Assertions for the first document
        assert resolved_annotations == {"doc1": ["Mount Everest"], "doc2": ["Alice"]}

    elif config == {"max_window": 8, "window_overlap": 2}:
        # Assertions for the first document
        assert resolved_annotations == {"doc1": ["Mount Everest"], "doc2": ["Alice"]}

    elif config == {"max_window": 8}:
        # Assertions for the first document
        assert resolved_annotations == {"doc1": ["Mount Everest"], "doc2": ["Alice"]}

    elif config == {"partition_annotation": "sentences"}:
        # Assertions for the first document
        assert resolved_annotations == {}

    else:
        raise ValueError(f"unknown config: {config}")


def test_document_type():
    taskmodule = TokenClassificationTaskModule(tokenizer_name_or_path="bert-base-uncased")
    assert taskmodule.document_type == TextDocumentWithLabeledSpans


def test_document_type_with_partitions():
    taskmodule = TokenClassificationTaskModule(
        tokenizer_name_or_path="bert-base-uncased", partition_annotation="labeled_partitions"
    )
    assert taskmodule.document_type == TextDocumentWithLabeledSpansAndLabeledPartitions


def test_document_type_with_non_default_span_annotation(caplog):
    with caplog.at_level(logging.WARNING):
        taskmodule = TokenClassificationTaskModule(
            tokenizer_name_or_path="bert-base-uncased", span_annotation="entities"
        )
    assert taskmodule.document_type is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "span_annotation=entities is not the default value ('labeled_spans'), so the taskmodule "
        "TokenClassificationTaskModule can not request the usual document type "
        "(TextDocumentWithLabeledSpans) for auto-conversion because this has the bespoken default value "
        "as layer name(s) instead of the provided one(s)."
    )


def test_document_type_with_non_default_partition_annotation(caplog):
    with caplog.at_level(logging.WARNING):
        taskmodule = TokenClassificationTaskModule(
            tokenizer_name_or_path="bert-base-uncased", partition_annotation="sentences"
        )
    assert taskmodule.document_type is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "partition_annotation=sentences is not the default value ('labeled_partitions'), "
        "so the taskmodule TokenClassificationTaskModule can not request the usual document type "
        "(TextDocumentWithLabeledSpansAndLabeledPartitions) for auto-conversion because this has "
        "the bespoken default value as layer name(s) instead of the provided one(s)."
    )


def test_document_type_with_non_default_span_and_partition_annotation(caplog):
    with caplog.at_level(logging.WARNING):
        taskmodule = TokenClassificationTaskModule(
            tokenizer_name_or_path="bert-base-uncased",
            span_annotation="entities",
            partition_annotation="sentences",
        )
    assert taskmodule.document_type is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "span_annotation=entities is not the default value ('labeled_spans') and "
        "partition_annotation=sentences is not the default value ('labeled_partitions'), "
        "so the taskmodule TokenClassificationTaskModule can not request the usual document "
        "type (TextDocumentWithLabeledSpansAndLabeledPartitions) for auto-conversion because "
        "this has the bespoken default value as layer name(s) instead of the provided one(s)."
    )
