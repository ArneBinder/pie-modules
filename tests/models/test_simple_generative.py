import math

import pytest
import torch

from pie_modules.models import SimpleGenerativeModel
from pie_modules.models.simple_generative import STAGE_TEST, STAGE_VAL
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
        # use a strange learning rate to make sure it is passed through
        learning_rate=13e-3,
        optimizer_type="torch.optim.Adam",
    )


def test_model(model):
    assert model is not None
    assert model.model is not None
    assert model.taskmodule is not None


def test_model_without_taskmodule(caplog):
    with caplog.at_level("WARNING"):
        model = SimpleGenerativeModel(
            base_model_type="transformers.AutoModelForSeq2SeqLM",
            base_model_config=dict(pretrained_model_name_or_path=MODEL_ID),
        )
    assert model is not None
    assert len(caplog.messages) == 2
    assert (
        caplog.messages[0]
        == "No taskmodule is available, so no metrics will be created. Please set taskmodule_config to a "
        "valid taskmodule config to use metrics."
    )
    assert (
        caplog.messages[1]
        == "No taskmodule is available, so no generation config will be created. Consider setting "
        "taskmodule_config to a valid taskmodule config to use specific setup for generation."
    )


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
    model.train()
    torch.manual_seed(42)
    metric = model.get_metric(STAGE_VAL, batch_idx=0)
    metric.reset()
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(8.98222827911377))

    metric_values = metric.compute()
    metric_values_float = {key: value.item() for key, value in metric_values.items()}

    # we do not collect metrics during training, so all entries should be NaN
    assert len(metric_values_float) > 0
    assert all([math.isnan(value) for value in metric_values_float.values()])

    model.on_train_epoch_end()


def test_validation_step(batch, model):
    model.eval()
    torch.manual_seed(42)
    metric = model.get_metric(STAGE_VAL, batch_idx=0)
    metric.reset()
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(10.146586418151855))

    metric_values = metric.compute()
    metric_values_float = {key: value.item() for key, value in metric_values.items()}
    assert metric_values_float == {
        "rouge1_fmeasure": 0.1111111119389534,
        "rouge1_precision": 0.06666667014360428,
        "rouge1_recall": 0.3333333432674408,
        "rouge2_fmeasure": 0.0,
        "rouge2_precision": 0.0,
        "rouge2_recall": 0.0,
        "rougeL_fmeasure": 0.1111111119389534,
        "rougeL_precision": 0.06666667014360428,
        "rougeL_recall": 0.3333333432674408,
        "rougeLsum_fmeasure": 0.0555555559694767,
        "rougeLsum_precision": 0.03333333507180214,
        "rougeLsum_recall": 0.1666666716337204,
    }

    model.on_validation_epoch_end()


def test_test_step(batch, model):
    model.eval()
    torch.manual_seed(42)
    metric = model.get_metric(STAGE_TEST, batch_idx=0)
    metric.reset()
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(10.146586418151855))

    metric_values = metric.compute()
    metric_values_float = {key: value.item() for key, value in metric_values.items()}
    assert metric_values_float == {
        "rouge1_fmeasure": 0.1111111119389534,
        "rouge1_precision": 0.06666667014360428,
        "rouge1_recall": 0.3333333432674408,
        "rouge2_fmeasure": 0.0,
        "rouge2_precision": 0.0,
        "rouge2_recall": 0.0,
        "rougeL_fmeasure": 0.1111111119389534,
        "rougeL_precision": 0.06666667014360428,
        "rougeL_recall": 0.3333333432674408,
        "rougeLsum_fmeasure": 0.0555555559694767,
        "rougeLsum_precision": 0.03333333507180214,
        "rougeLsum_recall": 0.1666666716337204,
    }

    model.on_test_epoch_end()


def test_predict_step(batch, model):
    model.eval()
    torch.manual_seed(42)
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


def test_configure_optimizers(model):
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 13e-3
    assert len(optimizer.param_groups) == 1
    param_group = optimizer.param_groups[0]
    assert len(param_group["params"]) == 47
