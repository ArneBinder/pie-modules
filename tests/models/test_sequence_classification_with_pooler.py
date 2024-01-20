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
        probabilities,
        torch.tensor(
            [
                [
                    0.08264221996068954,
                    0.16745781898498535,
                    0.48435819149017334,
                    0.26554182171821594,
                ],
                [0.0881081372499466, 0.09712868928909302, 0.498579740524292, 0.3161833882331848],
                [
                    0.08477703481912613,
                    0.14363254606723785,
                    0.5073344111442566,
                    0.26425591111183167,
                ],
                [
                    0.14074258506298065,
                    0.13151122629642487,
                    0.47058120369911194,
                    0.25716495513916016,
                ],
                [0.08874323219060898, 0.0940297544002533, 0.5076375603675842, 0.3095895051956177],
                [
                    0.13036882877349854,
                    0.20699138939380646,
                    0.42146939039230347,
                    0.24117043614387512,
                ],
                [0.1475677490234375, 0.1498306840658188, 0.42108210921287537, 0.28151947259902954],
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
