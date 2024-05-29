import math
from typing import List, Optional

import pytest
import torch
from pytorch_lightning import Trainer
from torch.optim import Optimizer

from pie_modules.models import SimpleGenerativeModel
from pie_modules.models.common import TESTING, VALIDATION
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
        text_metric_type="torchmetrics.text.ROUGEScore",
    )


@pytest.fixture(scope="module")
def model(taskmodule) -> SimpleGenerativeModel:
    return SimpleGenerativeModel(
        base_model_type="transformers.AutoModelForSeq2SeqLM",
        base_model_config=dict(pretrained_model_name_or_path=MODEL_ID),
        # only use predictions for metrics in test stage to cover all cases (default is all stages)
        metric_call_predict=[TESTING],
        taskmodule_config=taskmodule.config,
        # use a strange learning rate to make sure it is passed through
        learning_rate=13e-3,
        optimizer_type="torch.optim.Adam",
    )


def test_model(model):
    assert model is not None
    assert model.model is not None
    assert model.taskmodule is not None
    named_parameters = dict(model.named_parameters())
    parameter_means = {k: v.sum().mean().item() for k, v in named_parameters.items()}
    parameter_means_expected = {
        "model.shared.weight": -3213379.5,
        "model.encoder.block.0.layer.0.SelfAttention.q.weight": 1.4103267192840576,
        "model.encoder.block.0.layer.0.SelfAttention.k.weight": -99.39199829101562,
        "model.encoder.block.0.layer.0.SelfAttention.v.weight": -122.13032531738281,
        "model.encoder.block.0.layer.0.SelfAttention.o.weight": 56.757991790771484,
        "model.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": -361.335693359375,
        "model.encoder.block.0.layer.0.layer_norm.weight": 57.98170471191406,
        "model.encoder.block.0.layer.1.DenseReluDense.wi.weight": 908.3580932617188,
        "model.encoder.block.0.layer.1.DenseReluDense.wo.weight": 44.574432373046875,
        "model.encoder.block.0.layer.1.layer_norm.weight": 308.4140625,
        "model.encoder.block.1.layer.0.SelfAttention.q.weight": -5.168671607971191,
        "model.encoder.block.1.layer.0.SelfAttention.k.weight": -113.32646179199219,
        "model.encoder.block.1.layer.0.SelfAttention.v.weight": -168.3751678466797,
        "model.encoder.block.1.layer.0.SelfAttention.o.weight": 31.725296020507812,
        "model.encoder.block.1.layer.0.layer_norm.weight": 103.01976776123047,
        "model.encoder.block.1.layer.1.DenseReluDense.wi.weight": 318.46270751953125,
        "model.encoder.block.1.layer.1.DenseReluDense.wo.weight": -145.50942993164062,
        "model.encoder.block.1.layer.1.layer_norm.weight": 504.828125,
        "model.encoder.final_layer_norm.weight": 333.976318359375,
        "model.decoder.block.0.layer.0.SelfAttention.q.weight": 2.7615132331848145,
        "model.decoder.block.0.layer.0.SelfAttention.k.weight": 45.513877868652344,
        "model.decoder.block.0.layer.0.SelfAttention.v.weight": -8.495063781738281,
        "model.decoder.block.0.layer.0.SelfAttention.o.weight": 137.4838409423828,
        "model.decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": -75.12335205078125,
        "model.decoder.block.0.layer.0.layer_norm.weight": 50.14404296875,
        "model.decoder.block.0.layer.1.EncDecAttention.q.weight": 0.5128352642059326,
        "model.decoder.block.0.layer.1.EncDecAttention.k.weight": -9.239250183105469,
        "model.decoder.block.0.layer.1.EncDecAttention.v.weight": -71.90103149414062,
        "model.decoder.block.0.layer.1.EncDecAttention.o.weight": 175.31907653808594,
        "model.decoder.block.0.layer.1.layer_norm.weight": 16.86767578125,
        "model.decoder.block.0.layer.2.DenseReluDense.wi.weight": 330.0903015136719,
        "model.decoder.block.0.layer.2.DenseReluDense.wo.weight": 882.955322265625,
        "model.decoder.block.0.layer.2.layer_norm.weight": 764.71484375,
        "model.decoder.block.1.layer.0.SelfAttention.q.weight": 4.039071083068848,
        "model.decoder.block.1.layer.0.SelfAttention.k.weight": 27.05837631225586,
        "model.decoder.block.1.layer.0.SelfAttention.v.weight": -25.415212631225586,
        "model.decoder.block.1.layer.0.SelfAttention.o.weight": -265.144775390625,
        "model.decoder.block.1.layer.0.layer_norm.weight": 285.88543701171875,
        "model.decoder.block.1.layer.1.EncDecAttention.q.weight": -8.171785354614258,
        "model.decoder.block.1.layer.1.EncDecAttention.k.weight": 87.50703430175781,
        "model.decoder.block.1.layer.1.EncDecAttention.v.weight": -160.01060485839844,
        "model.decoder.block.1.layer.1.EncDecAttention.o.weight": -6.447116851806641,
        "model.decoder.block.1.layer.1.layer_norm.weight": 19.337772369384766,
        "model.decoder.block.1.layer.2.DenseReluDense.wi.weight": -1200.2591552734375,
        "model.decoder.block.1.layer.2.DenseReluDense.wo.weight": 2665.6357421875,
        "model.decoder.block.1.layer.2.layer_norm.weight": 1890.19287109375,
        "model.decoder.final_layer_norm.weight": 214.18069458007812,
    }
    assert set(parameter_means) == set(parameter_means_expected)
    for k in parameter_means:
        torch.testing.assert_close(
            torch.tensor(parameter_means[k]), torch.tensor(parameter_means_expected[k]), msg=k
        )


def test_model_pickleable(model):
    import pickle

    pickle.dumps(model)


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
        == "No taskmodule is available, so no metrics are set up. Please provide a "
        "taskmodule_config to enable metrics for stages ['train', 'val', 'test']."
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
    metric = model._get_metric(VALIDATION, batch_idx=0)
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
    metric = model._get_metric(VALIDATION, batch_idx=0)
    metric.reset()
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(10.146586418151855))

    metric_values = metric.compute()
    metric_values_float = {key: value.item() for key, value in metric_values.items()}
    assert metric_values_float == {
        "rouge1_fmeasure": 0.0,
        "rouge1_precision": 0.0,
        "rouge1_recall": 0.0,
        "rouge2_fmeasure": 0.0,
        "rouge2_precision": 0.0,
        "rouge2_recall": 0.0,
        "rougeL_fmeasure": 0.0,
        "rougeL_precision": 0.0,
        "rougeL_recall": 0.0,
        "rougeLsum_fmeasure": 0.0,
        "rougeLsum_precision": 0.0,
        "rougeLsum_recall": 0.0,
    }

    model.on_validation_epoch_end()


def test_test_step(batch, model):
    model.eval()
    torch.manual_seed(42)
    metric = model._get_metric(TESTING, batch_idx=0)
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
    labels = predictions["labels"]
    assert labels is not None
    torch.testing.assert_close(
        labels,
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
        model.taskmodule.tokenizer.convert_ids_to_tokens(label) for label in labels
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


@pytest.fixture(scope="module")
def optimizer(model):
    return model.configure_optimizers()


def test_optimizer(optimizer):
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 13e-3
    assert len(optimizer.param_groups) == 1
    param_group = optimizer.param_groups[0]
    assert len(param_group["params"]) == 47


def _assert_optimizer(
    actual: Optimizer,
    expected: Optimizer,
    allow_mismatching_param_group_keys: Optional[List[str]] = None,
):
    allow_mismatching_param_group_key_set = set(allow_mismatching_param_group_keys or [])
    assert actual is not None
    assert isinstance(actual, type(expected))
    assert actual.defaults == expected.defaults
    assert len(actual.param_groups) == len(expected.param_groups)
    for actual_param_group, expected_param_group in zip(
        actual.param_groups, expected.param_groups
    ):
        actual_keys = set(actual_param_group) - allow_mismatching_param_group_key_set
        expected_keys = set(expected_param_group) - allow_mismatching_param_group_key_set
        assert actual_keys == expected_keys
        for key in actual_keys:
            # also include the key in the comparison to have it in the assertion error message
            assert (key, actual_param_group[key]) == (key, expected_param_group[key])


def test_configure_optimizers_with_warmup(model, optimizer):
    backup_value = model.warmup_proportion
    model.warmup_proportion = 0.1
    model.trainer = Trainer(max_epochs=10)
    optimizer_and_schedular = model.configure_optimizers()
    assert optimizer_and_schedular is not None
    assert isinstance(optimizer_and_schedular, tuple)
    assert len(optimizer_and_schedular) == 2
    optimizers, schedulers = optimizer_and_schedular
    assert len(optimizers) == 1
    _assert_optimizer(
        optimizers[0], optimizer, allow_mismatching_param_group_keys=["initial_lr", "lr"]
    )
    assert len(schedulers) == 1
    assert set(schedulers[0]) == {"scheduler", "interval"}
    scheduler = schedulers[0]["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert scheduler.optimizer is optimizers[0]
    assert scheduler.base_lrs == [13e-3]

    model.warmup_proportion = backup_value
