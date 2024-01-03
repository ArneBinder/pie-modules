import pytest
import torch

from pie_modules.models import SimpleGenerativeModel
from pie_modules.taskmodules import TextToTextTaskModule

MODEL_ID = "google/t5-efficient-tiny-nl2"


@pytest.fixture(scope="module")
def taskmodule():
    return TextToTextTaskModule(
        tokenizer_name_or_path=MODEL_ID,
        document_type="pie_modules.documents.TextDocumentWithAbstractiveSummary",
        target_layer="abstractive_summary",
        target_annotation_type="pie_modules.annotations.AbstractiveSummary",
        tokenized_document_type="pie_modules.documents.TokenDocumentWithAbstractiveSummary",
    )


@pytest.fixture(scope="module")
def model(taskmodule):
    return SimpleGenerativeModel(
        base_model_type="transformers.AutoModelForSeq2SeqLM",
        base_model_config=dict(pretrained_model_name_or_path=MODEL_ID),
        taskmodule_config=taskmodule._config(),
        learning_rate=1e-3,
        optimizer_type="torch.optim.Adam",
    )


def test_model(model):
    assert model is not None
    assert model.model is not None
    assert model.taskmodule is not None


@pytest.fixture(scope="module")
def batch(model):
    inputs = {
        "input_ids": torch.tensor(
            [
                [100, 19, 3, 9, 794, 1708, 1, 0, 0, 0, 0, 0],
                [100, 19, 430, 794, 1708, 84, 19, 3, 9, 720, 1200, 1],
            ]
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        ),
    }

    targets = {
        "labels": torch.tensor([[3, 9, 1708, 1, 0], [3, 9, 1200, 1708, 1]]),
        "decoder_attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
    }

    return inputs, targets


def test_batch(batch, taskmodule):
    inputs, targets = batch
    input_ids_tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs["input_ids"]
    ]
    assert input_ids_tokens == [
        [
            "▁This",
            "▁is",
            "▁",
            "a",
            "▁test",
            "▁document",
            "</s>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
        ],
        [
            "▁This",
            "▁is",
            "▁another",
            "▁test",
            "▁document",
            "▁which",
            "▁is",
            "▁",
            "a",
            "▁bit",
            "▁longer",
            "</s>",
        ],
    ]

    labels_tokens = [
        taskmodule.tokenizer.convert_ids_to_tokens(labels) for labels in targets["labels"]
    ]
    assert labels_tokens == [
        ["▁", "a", "▁document", "</s>", "<pad>"],
        ["▁", "a", "▁longer", "▁document", "</s>"],
    ]


def test_training_step(batch, model):
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(10.146586418151855))

    model.on_train_epoch_end()


def test_validation_step(batch, model):
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(10.146586418151855))

    model.on_validation_epoch_end()


def test_test_step(batch, model):
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(10.146586418151855))

    model.on_test_epoch_end()


def test_predict_step(batch, model):
    predictions = model.predict_step(batch, batch_idx=0)
    assert predictions is not None
    torch.testing.assert_close(
        predictions,
        torch.tensor(
            [
                [32099, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [
                    32099,
                    19,
                    3,
                    9,
                    248,
                    194,
                    12,
                    129,
                    25,
                    708,
                    5,
                    37,
                    166,
                    794,
                    1708,
                    19,
                    3,
                    9,
                    794,
                ],
            ]
        ),
    )

    predicted_tokens = [
        model.taskmodule.tokenizer.convert_ids_to_tokens(prediction) for prediction in predictions
    ]
    assert predicted_tokens == [
        [
            "<extra_id_0>",
            "</s>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
        ],
        [
            "<extra_id_0>",
            "▁is",
            "▁",
            "a",
            "▁great",
            "▁way",
            "▁to",
            "▁get",
            "▁you",
            "▁started",
            ".",
            "▁The",
            "▁first",
            "▁test",
            "▁document",
            "▁is",
            "▁",
            "a",
            "▁test",
        ],
    ]
