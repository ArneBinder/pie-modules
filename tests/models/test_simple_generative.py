import math
from typing import List, Optional

import pytest
import torch
from pytorch_lightning import Trainer
from torch.optim import Optimizer

from pie_modules.models import SimpleGenerativeModel
from pie_modules.models.common import TESTING, VALIDATION
from pie_modules.taskmodules import TextToTextTaskModule
from tests.models import trunc_number

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
    parameter_means = {k: trunc_number(v.mean().item(), 8) for k, v in named_parameters.items()}
    parameter_means_expected = {
        "model.decoder.block.0.layer.0.SelfAttention.k.weight": 0.00069448,
        "model.decoder.block.0.layer.0.SelfAttention.o.weight": 0.00209783,
        "model.decoder.block.0.layer.0.SelfAttention.q.weight": 4.213e-05,
        "model.decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": -0.58690118,
        "model.decoder.block.0.layer.0.SelfAttention.v.weight": -0.00012962,
        "model.decoder.block.0.layer.0.layer_norm.weight": 0.19587516,
        "model.decoder.block.0.layer.1.EncDecAttention.k.weight": -0.00014097,
        "model.decoder.block.0.layer.1.EncDecAttention.o.weight": 0.00267515,
        "model.decoder.block.0.layer.1.EncDecAttention.q.weight": 7.82e-06,
        "model.decoder.block.0.layer.1.EncDecAttention.v.weight": -0.00109712,
        "model.decoder.block.0.layer.1.layer_norm.weight": 0.06588935,
        "model.decoder.block.0.layer.2.DenseReluDense.wi.weight": 0.00125919,
        "model.decoder.block.0.layer.2.DenseReluDense.wo.weight": 0.0033682,
        "model.decoder.block.0.layer.2.layer_norm.weight": 2.98716735,
        "model.decoder.block.1.layer.0.SelfAttention.k.weight": 0.00041287,
        "model.decoder.block.1.layer.0.SelfAttention.o.weight": -0.00404578,
        "model.decoder.block.1.layer.0.SelfAttention.q.weight": 6.163e-05,
        "model.decoder.block.1.layer.0.SelfAttention.v.weight": -0.0003878,
        "model.decoder.block.1.layer.0.layer_norm.weight": 1.11673998,
        "model.decoder.block.1.layer.1.EncDecAttention.k.weight": 0.00133525,
        "model.decoder.block.1.layer.1.EncDecAttention.o.weight": -9.837e-05,
        "model.decoder.block.1.layer.1.EncDecAttention.q.weight": -0.00012469,
        "model.decoder.block.1.layer.1.EncDecAttention.v.weight": -0.00244156,
        "model.decoder.block.1.layer.1.layer_norm.weight": 0.07553817,
        "model.decoder.block.1.layer.2.DenseReluDense.wi.weight": -0.00457862,
        "model.decoder.block.1.layer.2.DenseReluDense.wo.weight": 0.01016859,
        "model.decoder.block.1.layer.2.layer_norm.weight": 7.3835659,
        "model.decoder.final_layer_norm.weight": 0.83664333,
        "model.encoder.block.0.layer.0.SelfAttention.k.weight": -0.0015166,
        "model.encoder.block.0.layer.0.SelfAttention.o.weight": 0.00086605,
        "model.encoder.block.0.layer.0.SelfAttention.q.weight": 2.151e-05,
        "model.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": -2.8229351,
        "model.encoder.block.0.layer.0.SelfAttention.v.weight": -0.00186356,
        "model.encoder.block.0.layer.0.layer_norm.weight": 0.22649103,
        "model.encoder.block.0.layer.1.DenseReluDense.wi.weight": 0.00346511,
        "model.encoder.block.0.layer.1.DenseReluDense.wo.weight": 0.00017003,
        "model.encoder.block.0.layer.1.layer_norm.weight": 1.20474243,
        "model.encoder.block.1.layer.0.SelfAttention.k.weight": -0.00172922,
        "model.encoder.block.1.layer.0.SelfAttention.o.weight": 0.00048408,
        "model.encoder.block.1.layer.0.SelfAttention.q.weight": -7.886e-05,
        "model.encoder.block.1.layer.0.SelfAttention.v.weight": -0.0025692,
        "model.encoder.block.1.layer.0.layer_norm.weight": 0.40242096,
        "model.encoder.block.1.layer.1.DenseReluDense.wi.weight": 0.00121483,
        "model.encoder.block.1.layer.1.DenseReluDense.wo.weight": -0.00055507,
        "model.encoder.block.1.layer.1.layer_norm.weight": 1.97198486,
        "model.encoder.final_layer_norm.weight": 1.30459499,
        "model.shared.weight": -0.39069545,
    }
    assert parameter_means == parameter_means_expected


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
