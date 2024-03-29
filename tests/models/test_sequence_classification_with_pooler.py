from typing import Dict

import pytest
import torch
from pytorch_lightning import Trainer
from torch import LongTensor
from torch.optim.lr_scheduler import LambdaLR
from transformers.modeling_outputs import SequenceClassifierOutput

from pie_modules.models import SequenceClassificationModelWithPooler
from pie_modules.models.sequence_classification_with_pooler import OutputType

NUM_CLASSES = 4
POOLER = "start_tokens"


@pytest.fixture
def inputs() -> Dict[str, LongTensor]:
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
        "pooler_start_indices": torch.tensor(
            [[2, 10], [5, 13], [5, 17], [17, 11], [5, 13], [14, 18], [18, 14]]
        ).to(torch.long),
        "pooler_end_indices": torch.tensor(
            [[6, 11], [9, 14], [9, 18], [18, 12], [9, 14], [15, 19], [19, 15]]
        ).to(torch.long),
    }

    return result_dict


@pytest.fixture
def targets() -> Dict[str, LongTensor]:
    return {"labels": torch.tensor([0, 1, 2, 3, 1, 2, 3]).to(torch.long)}


@pytest.fixture
def model() -> SequenceClassificationModelWithPooler:
    torch.manual_seed(42)
    result = SequenceClassificationModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        pooler=POOLER,
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
    batch_size, seq_len = inputs["input_ids"].shape

    assert isinstance(model_output, SequenceClassifierOutput)
    assert set(model_output) == {"logits"}
    logits = model_output["logits"]

    assert logits.shape == (batch_size, NUM_CLASSES)

    torch.testing.assert_close(
        logits,
        torch.tensor(
            [
                [-0.5805037021636963, 0.12570726871490479, 1.187800407409668, 0.5867480635643005],
                [-0.5103899836540222, -0.4129180312156677, 1.222808599472046, 0.767367422580719],
                [
                    -0.5193025469779968,
                    0.007931053638458252,
                    1.2698432207107544,
                    0.6175908446311951,
                ],
                [
                    -0.10545363277196884,
                    -0.17329390347003937,
                    1.101582407951355,
                    0.49733155965805054,
                ],
                [
                    -0.48656341433525085,
                    -0.4286993145942688,
                    1.2574571371078491,
                    0.7629366517066956,
                ],
                [
                    -0.3718412220478058,
                    0.09046845138072968,
                    0.8015384674072266,
                    0.24329520761966705,
                ],
                [-0.20474043488502502, -0.1895218938589096, 0.8438000679016113, 0.441173791885376],
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
        torch.tensor([2, 2, 2, 2, 2, 2, 2]),
    )
    probabilities = decoded["probabilities"]
    assert probabilities.shape == (inputs["input_ids"].shape[0], NUM_CLASSES)
    torch.testing.assert_close(
        probabilities.round(decimals=4),
        torch.tensor(
            [
                [0.0826, 0.1675, 0.4844, 0.2655],
                [0.0881, 0.0971, 0.4986, 0.3162],
                [0.0848, 0.1436, 0.5073, 0.2643],
                [0.1407, 0.1315, 0.4706, 0.2572],
                [0.0887, 0.0940, 0.5076, 0.3096],
                [0.1304, 0.2070, 0.4215, 0.2412],
                [0.1476, 0.1498, 0.4211, 0.2815],
            ]
        ),
    )


def test_decode_with_multi_label(model_output, inputs):
    torch.manual_seed(42)
    model = SequenceClassificationModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        pooler=POOLER,
        multi_label=True,
    )
    decoded = model.decode(inputs=inputs, outputs=model_output)
    assert isinstance(decoded, dict)
    assert set(decoded) == {"labels", "probabilities"}
    labels = decoded["labels"]
    assert labels.shape == (inputs["input_ids"].shape[0], NUM_CLASSES)
    torch.testing.assert_close(
        labels,
        torch.tensor(
            [
                [0, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
            ]
        ),
    )
    probabilities = decoded["probabilities"]
    assert probabilities.shape == (inputs["input_ids"].shape[0], NUM_CLASSES)
    torch.testing.assert_close(
        probabilities.round(decimals=4),
        torch.tensor(
            [
                [0.3588, 0.5314, 0.7663, 0.6426],
                [0.3751, 0.3982, 0.7726, 0.6830],
                [0.3730, 0.5020, 0.7807, 0.6497],
                [0.4737, 0.4568, 0.7506, 0.6218],
                [0.3807, 0.3944, 0.7786, 0.6820],
                [0.4081, 0.5226, 0.6903, 0.5605],
                [0.4490, 0.4528, 0.6993, 0.6085],
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
    torch.testing.assert_close(loss, torch.tensor(1.6224687099456787))


def test_validation_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.6224687099456787))


def test_test_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.6224687099456787))


def test_base_model_named_parameters(model):
    base_model_named_parameters = dict(model.base_model_named_parameters())
    assert set(base_model_named_parameters) == {
        "model.pooler.dense.bias",
        "model.encoder.layer.0.intermediate.dense.weight",
        "model.encoder.layer.0.intermediate.dense.bias",
        "model.encoder.layer.1.attention.output.dense.weight",
        "model.encoder.layer.1.attention.output.LayerNorm.weight",
        "model.encoder.layer.1.attention.self.query.weight",
        "model.encoder.layer.1.output.dense.weight",
        "model.encoder.layer.0.output.dense.bias",
        "model.encoder.layer.1.intermediate.dense.bias",
        "model.encoder.layer.1.attention.self.value.bias",
        "model.encoder.layer.0.attention.output.dense.weight",
        "model.encoder.layer.0.attention.self.query.bias",
        "model.encoder.layer.0.attention.self.value.bias",
        "model.encoder.layer.1.output.dense.bias",
        "model.encoder.layer.1.attention.self.query.bias",
        "model.encoder.layer.1.attention.output.LayerNorm.bias",
        "model.encoder.layer.0.attention.self.query.weight",
        "model.encoder.layer.0.attention.output.LayerNorm.bias",
        "model.encoder.layer.0.attention.self.key.bias",
        "model.encoder.layer.1.intermediate.dense.weight",
        "model.encoder.layer.1.output.LayerNorm.bias",
        "model.encoder.layer.1.output.LayerNorm.weight",
        "model.encoder.layer.0.attention.self.key.weight",
        "model.encoder.layer.1.attention.output.dense.bias",
        "model.encoder.layer.0.attention.output.dense.bias",
        "model.embeddings.LayerNorm.bias",
        "model.encoder.layer.0.attention.self.value.weight",
        "model.encoder.layer.0.attention.output.LayerNorm.weight",
        "model.embeddings.token_type_embeddings.weight",
        "model.encoder.layer.0.output.LayerNorm.weight",
        "model.embeddings.position_embeddings.weight",
        "model.encoder.layer.1.attention.self.key.bias",
        "model.embeddings.LayerNorm.weight",
        "model.encoder.layer.0.output.LayerNorm.bias",
        "model.encoder.layer.1.attention.self.key.weight",
        "model.pooler.dense.weight",
        "model.encoder.layer.0.output.dense.weight",
        "model.embeddings.word_embeddings.weight",
        "model.encoder.layer.1.attention.self.value.weight",
    }


def test_task_named_parameters(model):
    task_named_parameters = dict(model.task_named_parameters())
    assert set(task_named_parameters) == {
        "classifier.weight",
        "pooler.pooler.missing_embeddings",
        "classifier.bias",
    }


def test_configure_optimizers_with_warmup():
    model = SequenceClassificationModelWithPooler(
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
    model = SequenceClassificationModelWithPooler(
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
    model = SequenceClassificationModelWithPooler(
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
