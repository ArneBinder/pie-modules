from typing import Dict

import pytest
import torch
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import LambdaLR
from transformers.modeling_outputs import SequenceClassifierOutput

from pie_modules.models import SimpleSequenceClassificationModel
from pie_modules.models.simple_sequence_classification import OutputType

NUM_CLASSES = 4


@pytest.fixture
def inputs() -> Dict[str, torch.LongTensor]:
    result_dict = {
        "input_ids": torch.tensor(
            [
                [
                    101,
                    28998,
                    13832,
                    3121,
                    2340,
                    138,
                    28996,
                    1759,
                    1120,
                    28999,
                    139,
                    28997,
                    119,
                    102,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    144,
                    28996,
                    1759,
                    1120,
                    28999,
                    145,
                    28997,
                    119,
                    1262,
                    1771,
                    146,
                    119,
                    102,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    144,
                    28996,
                    1759,
                    1120,
                    145,
                    119,
                    1262,
                    1771,
                    28999,
                    146,
                    28997,
                    119,
                    102,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    13832,
                    3121,
                    2340,
                    144,
                    1759,
                    1120,
                    28999,
                    145,
                    28997,
                    119,
                    1262,
                    1771,
                    28998,
                    146,
                    28996,
                    119,
                    102,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    150,
                    28996,
                    1759,
                    1120,
                    28999,
                    151,
                    28997,
                    119,
                    1262,
                    1122,
                    1771,
                    152,
                    119,
                    102,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    13832,
                    3121,
                    2340,
                    150,
                    1759,
                    1120,
                    151,
                    119,
                    1262,
                    28998,
                    1122,
                    28996,
                    1771,
                    28999,
                    152,
                    28997,
                    119,
                    102,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    13832,
                    3121,
                    2340,
                    150,
                    1759,
                    1120,
                    151,
                    119,
                    1262,
                    28999,
                    1122,
                    28997,
                    1771,
                    28998,
                    152,
                    28996,
                    119,
                    102,
                ],
            ]
        ).to(torch.long),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ).to(torch.long),
    }

    return result_dict


@pytest.fixture
def targets() -> Dict[str, torch.LongTensor]:
    return {"labels": torch.tensor([0, 1, 2, 3, 1, 2, 3]).to(torch.long)}


@pytest.fixture
def model() -> SimpleSequenceClassificationModel:
    torch.manual_seed(42)
    result = SimpleSequenceClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
    )
    return result


def test_model_pickleable(model):
    import pickle

    pickle.dumps(model)


@pytest.fixture
def model_output(model, inputs) -> OutputType:
    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    return model(inputs)


def test_forward(model_output, inputs):
    batch_size = inputs["input_ids"].shape[0]
    assert isinstance(model_output, SequenceClassifierOutput)
    assert set(model_output) == {"logits"}
    logits = model_output["logits"]

    assert logits.shape == (batch_size, NUM_CLASSES)

    torch.testing.assert_close(
        logits,
        torch.tensor(
            [
                [
                    -0.0780562311410904,
                    0.2885679602622986,
                    -0.1652916818857193,
                    0.25686803460121155,
                ],
                [
                    -0.07416453957557678,
                    0.28354859352111816,
                    -0.18583300709724426,
                    0.2679266333580017,
                ],
                [
                    -0.07721473276615143,
                    0.27951815724372864,
                    -0.18438687920570374,
                    0.26015764474868774,
                ],
                [
                    -0.07416942715644836,
                    0.2880846858024597,
                    -0.18872812390327454,
                    0.2668967545032501,
                ],
                [
                    -0.06794390082359314,
                    0.2791520059108734,
                    -0.18853652477264404,
                    0.2560432553291321,
                ],
                [
                    -0.06797368824481964,
                    0.28091782331466675,
                    -0.18849357962608337,
                    0.24799709022045135,
                ],
                [
                    -0.06416021287441254,
                    0.2858850657939911,
                    -0.19061337411403656,
                    0.25447627902030945,
                ],
            ]
        ),
    )


def test_decode(model, model_output, inputs):
    decoded = model.decode(inputs=inputs, outputs=model_output)
    assert isinstance(decoded, dict)
    assert set(decoded) == {"labels", "probabilities"}
    labels = decoded["labels"]
    assert labels.shape == (inputs["input_ids"].shape[0],)
    torch.testing.assert_close(
        labels,
        torch.tensor([1, 1, 1, 1, 1, 1, 1]),
    )
    probabilities = decoded["probabilities"]
    assert probabilities.shape == (inputs["input_ids"].shape[0], NUM_CLASSES)
    torch.testing.assert_close(
        probabilities,
        torch.tensor(
            [
                [
                    0.21020983159542084,
                    0.30330243706703186,
                    0.19264918565750122,
                    0.2938385605812073,
                ],
                [0.21131442487239838, 0.3021913170814514, 0.18898707628250122, 0.2975071966648102],
                [0.2114931344985962, 0.30215057730674744, 0.1899993121623993, 0.29635706543922424],
                [
                    0.21120351552963257,
                    0.30340734124183655,
                    0.18834275007247925,
                    0.29704639315605164,
                ],
                [
                    0.21349377930164337,
                    0.3020835518836975,
                    0.18923982977867126,
                    0.29518282413482666,
                ],
                [0.2138788104057312, 0.30317223072052, 0.18959489464759827, 0.2933540642261505],
                [0.21387633681297302, 0.30351871252059937, 0.18847113847732544, 0.294133722782135],
            ]
        ),
    )


@pytest.fixture
def batch(inputs, targets):
    return inputs, targets


def test_training_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.3877977132797241))


def test_validation_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.3877977132797241))


def test_test_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.3877977132797241))


def test_base_model_named_parameters(model):
    base_model_named_parameters = dict(model.base_model_named_parameters())
    assert set(base_model_named_parameters) == {
        "model.bert.pooler.dense.bias",
        "model.bert.encoder.layer.0.intermediate.dense.weight",
        "model.bert.encoder.layer.0.intermediate.dense.bias",
        "model.bert.encoder.layer.1.attention.output.dense.weight",
        "model.bert.encoder.layer.1.attention.output.LayerNorm.weight",
        "model.bert.encoder.layer.1.attention.self.query.weight",
        "model.bert.encoder.layer.1.output.dense.weight",
        "model.bert.encoder.layer.0.output.dense.bias",
        "model.bert.encoder.layer.1.intermediate.dense.bias",
        "model.bert.encoder.layer.1.attention.self.value.bias",
        "model.bert.encoder.layer.0.attention.output.dense.weight",
        "model.bert.encoder.layer.0.attention.self.query.bias",
        "model.bert.encoder.layer.0.attention.self.value.bias",
        "model.bert.encoder.layer.1.output.dense.bias",
        "model.bert.encoder.layer.1.attention.self.query.bias",
        "model.bert.encoder.layer.1.attention.output.LayerNorm.bias",
        "model.bert.encoder.layer.0.attention.self.query.weight",
        "model.bert.encoder.layer.0.attention.output.LayerNorm.bias",
        "model.bert.encoder.layer.0.attention.self.key.bias",
        "model.bert.encoder.layer.1.intermediate.dense.weight",
        "model.bert.encoder.layer.1.output.LayerNorm.bias",
        "model.bert.encoder.layer.1.output.LayerNorm.weight",
        "model.bert.encoder.layer.0.attention.self.key.weight",
        "model.bert.encoder.layer.1.attention.output.dense.bias",
        "model.bert.encoder.layer.0.attention.output.dense.bias",
        "model.bert.embeddings.LayerNorm.bias",
        "model.bert.encoder.layer.0.attention.self.value.weight",
        "model.bert.encoder.layer.0.attention.output.LayerNorm.weight",
        "model.bert.embeddings.token_type_embeddings.weight",
        "model.bert.encoder.layer.0.output.LayerNorm.weight",
        "model.bert.embeddings.position_embeddings.weight",
        "model.bert.encoder.layer.1.attention.self.key.bias",
        "model.bert.embeddings.LayerNorm.weight",
        "model.bert.encoder.layer.0.output.LayerNorm.bias",
        "model.bert.encoder.layer.1.attention.self.key.weight",
        "model.bert.pooler.dense.weight",
        "model.bert.encoder.layer.0.output.dense.weight",
        "model.bert.embeddings.word_embeddings.weight",
        "model.bert.encoder.layer.1.attention.self.value.weight",
    }


def test_task_named_parameters(model):
    task_named_parameters = dict(model.task_named_parameters())
    assert set(task_named_parameters) == {
        "model.classifier.weight",
        "model.classifier.bias",
    }


def test_configure_optimizers_with_warmup():
    model = SimpleSequenceClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
    )
    model.trainer = Trainer(max_epochs=10)
    optimizers_and_schedulers = model.configure_optimizers()
    assert len(optimizers_and_schedulers) == 2
    optimizers, schedulers = optimizers_and_schedulers
    assert len(optimizers) == 1
    assert len(schedulers) == 1
    optimizer = optimizers[0]
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-05
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["eps"] == 1e-08

    scheduler = schedulers[0]
    assert isinstance(scheduler, dict)
    assert set(scheduler) == {"scheduler", "interval"}
    assert isinstance(scheduler["scheduler"], LambdaLR)


def test_configure_optimizers_with_task_learning_rate(monkeypatch):
    model = SimpleSequenceClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        learning_rate=1e-5,
        task_learning_rate=1e-3,
        # disable warmup to make sure the scheduler is not added which would set the learning rate
        # to 0
        warmup_proportion=0.0,
    )
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    # base model parameters
    param_group = optimizer.param_groups[0]
    assert len(param_group["params"]) == 39
    assert param_group["lr"] == 1e-5
    # classifier head parameters
    param_group = optimizer.param_groups[1]
    assert len(param_group["params"]) == 2
    assert param_group["lr"] == 1e-3
    # ensure that all parameters are covered
    assert set(optimizer.param_groups[0]["params"] + optimizer.param_groups[1]["params"]) == set(
        model.parameters()
    )


def test_freeze_base_model(monkeypatch, inputs, targets):
    model = SimpleSequenceClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        freeze_base_model=True,
        # disable warmup to make sure the scheduler is not added which would set the learning rate
        # to 0
        warmup_proportion=0.0,
    )
    base_model_params = [param for name, param in model.base_model_named_parameters()]
    task_params = [param for name, param in model.task_named_parameters()]
    assert len(base_model_params) + len(task_params) == len(list(model.parameters()))
    for param in base_model_params:
        assert not param.requires_grad
    for param in task_params:
        assert param.requires_grad


def test_base_model_named_parameters_wrong_prefix(monkeypatch):
    model = SimpleSequenceClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        base_model_prefix="wrong_prefix",
    )
    with pytest.raises(ValueError) as excinfo:
        model.base_model_named_parameters()
    assert (
        str(excinfo.value)
        == "Base model with prefix 'wrong_prefix' not found in BertForSequenceClassification"
    )
