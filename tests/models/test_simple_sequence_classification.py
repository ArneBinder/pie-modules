import pytest
import torch
import transformers
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    SequenceClassifierOutputWithPast,
)

from pie_modules.models import SimpleSequenceClassificationModel


@pytest.fixture
def inputs():
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
        ),
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
        ),
        "pooler_start_indices": torch.tensor(
            [[2, 10], [5, 13], [5, 17], [17, 11], [5, 13], [14, 18], [18, 14]]
        ),
        "pooler_end_indices": torch.tensor(
            [[6, 11], [9, 14], [9, 18], [18, 12], [9, 14], [15, 19], [19, 15]]
        ),
    }

    return result_dict


@pytest.fixture
def targets():
    return torch.tensor([0, 1, 2, 3, 1, 2, 3])


@pytest.fixture(params=["cls_token"])  # , "mention_pooling", "start_tokens"])
def pooler_type(request):
    return request.param


def get_model(
    monkeypatch,
    pooler_type,
    batch_size,
    seq_len,
    num_classes,
    add_dummy_linear=False,
    model_type="bert",
    base_model_prefix="dummy_linear",
    **model_kwargs,
):
    class MockConfig:
        def __init__(
            self, hidden_size: int = 10, classifier_dropout: float = 0.1, model_type="bert"
        ) -> None:
            self.hidden_size = hidden_size
            self.model_type = model_type
            if self.model_type == "distilbert":
                self.seq_classif_dropout = classifier_dropout
            elif self.model_type == "albert":
                self.classifier_dropout_prob = classifier_dropout
            else:
                self.classifier_dropout = classifier_dropout

    class MockModel(torch.nn.Module):
        def __init__(
            self, batch_size, seq_len, hidden_size, num_classes, add_dummy_linear
        ) -> None:
            super().__init__()
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.hidden_size = hidden_size
            self.num_classes = num_classes
            if add_dummy_linear:
                self.dummy_linear = torch.nn.Linear(self.hidden_size, 99)
            self.classifier = torch.nn.Linear(self.hidden_size, self.num_classes)

        def __call__(self, *args, **kwargs):
            logits = torch.rand(self.batch_size, self.num_classes)
            loss = torch.rand(1)[0]
            return SequenceClassifierOutputWithPast(logits=logits, loss=loss)

        def resize_token_embeddings(self, new_num_tokens):
            pass

    hidden_size = 10
    tokenizer_vocab_size = 30000

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda model_name_or_path: MockConfig(
            hidden_size=hidden_size, classifier_dropout=0.1, model_type=model_type
        ),
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda model_name_or_path, config: MockModel(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_classes=num_classes,
            add_dummy_linear=add_dummy_linear,
        ),
    )

    # set seed to make the classifier deterministic
    torch.manual_seed(42)
    result = SimpleSequenceClassificationModel(
        model_name_or_path="some-model-name",
        num_classes=num_classes,
        tokenizer_vocab_size=tokenizer_vocab_size,
        ignore_index=0,
        # pooler=pooler_type,
        # disable warmup because it would require a trainer and a datamodule to get the total number of training steps
        warmup_proportion=0.0,
        base_model_prefix=base_model_prefix,
        **model_kwargs,
    )
    assert not result.is_from_pretrained

    return result


@pytest.fixture
def model(monkeypatch, pooler_type, inputs, targets):
    return get_model(
        monkeypatch=monkeypatch,
        pooler_type=pooler_type,
        batch_size=inputs["input_ids"].shape[0],
        seq_len=inputs["input_ids"].shape[1],
        num_classes=int(max(targets) + 1),
    )


def test_forward(inputs, model, pooler_type):
    batch_size, seq_len = inputs["input_ids"].shape
    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    output = model.forward(inputs)
    assert set(output) == {"logits", "loss"}
    logits = output["logits"]

    assert logits.shape == (batch_size, 4)

    if pooler_type == "cls_token":
        torch.testing.assert_close(
            logits,
            torch.tensor(
                [
                    [
                        0.8822692632675171,
                        0.9150039553642273,
                        0.38286375999450684,
                        0.9593056440353394,
                    ],
                    [
                        0.3904482126235962,
                        0.600895345211029,
                        0.2565724849700928,
                        0.7936413288116455,
                    ],
                    [
                        0.9407714605331421,
                        0.13318592309951782,
                        0.9345980882644653,
                        0.5935796499252319,
                    ],
                    [
                        0.8694044351577759,
                        0.5677152872085571,
                        0.7410940527915955,
                        0.42940449714660645,
                    ],
                    [
                        0.8854429125785828,
                        0.5739044547080994,
                        0.2665800452232361,
                        0.6274491548538208,
                    ],
                    [
                        0.26963168382644653,
                        0.4413635730743408,
                        0.2969208359718323,
                        0.831685483455658,
                    ],
                    [
                        0.10531491041183472,
                        0.26949483156204224,
                        0.3588126301765442,
                        0.19936376810073853,
                    ],
                ]
            ),
        )
    else:
        raise ValueError(f"Unknown pooler type: {pooler_type}")


@pytest.fixture
def batch(inputs, targets):
    return (inputs, targets)


def test_step(batch, model, pooler_type):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.step("train", batch)
    if pooler_type == "cls_token":
        torch.testing.assert_close(loss, torch.tensor(0.5471915602684021))
    else:
        raise ValueError(f"Unknown pooler type: {pooler_type}")


def test_training_step(batch, model):
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None


def test_validation_step(batch, model):
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None


def test_test_step(batch, model):
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None


def test_configure_optimizers(model):
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-05
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["eps"] == 1e-08


def test_configure_optimizers_with_task_learning_rate(monkeypatch):
    model = get_model(
        monkeypatch=monkeypatch,
        pooler_type="cls_token",
        batch_size=7,
        seq_len=22,
        num_classes=4,
        add_dummy_linear=True,
        task_learning_rate=0.1,
    )
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    param_group = optimizer.param_groups[0]
    assert param_group["lr"] == 1e-05
    # the dummy linear from the mock base model has 2 parameters
    assert len(param_group["params"]) == 2
    assert param_group["params"][0].shape == torch.Size([99, 10])
    assert param_group["params"][1].shape == torch.Size([99])
    param_group = optimizer.param_groups[1]
    assert param_group["lr"] == 0.1
    # the classifier head has 2 parameters
    assert len(param_group["params"]) == 2
    assert param_group["params"][0].shape == torch.Size([4, 10])
    assert param_group["params"][1].shape == torch.Size([4])


def test_freeze_base_model(monkeypatch, inputs, targets):
    # set seed to make the classifier deterministic
    model = get_model(
        monkeypatch,
        pooler_type="cls_token",
        batch_size=7,
        seq_len=22,
        num_classes=4,
        add_dummy_linear=True,
        freeze_base_model=True,
    )
    base_model_params = list(model.model.dummy_linear.parameters())
    task_parameters = list(model.model.classifier.parameters())
    # the dummy linear from the mock base model has 2 parameters
    assert len(base_model_params) == 2
    for param in base_model_params:
        assert not param.requires_grad


@pytest.mark.parametrize(
    "model_type", ["bert", "albert", "distilbert", "roberta", "deberta", "electra", "xlm-roberta"]
)
def test_config_model_classifier_dropout(monkeypatch, model_type):
    model = get_model(
        monkeypatch,
        pooler_type="cls_token",
        batch_size=7,
        seq_len=22,
        num_classes=4,
        model_type=model_type,
    )
    assert model is not None


def test_base_model_named_parameters(monkeypatch):
    model = get_model(
        monkeypatch,
        pooler_type="cls_token",
        batch_size=7,
        seq_len=22,
        num_classes=4,
        add_dummy_linear=True,
    )
    base_model_named_parameters = model.base_model_named_parameters()
    assert len(base_model_named_parameters) == 2
    assert base_model_named_parameters[0][0] == "model.dummy_linear.weight"
    assert base_model_named_parameters[1][0] == "model.dummy_linear.bias"


def test_base_model_named_parameters_no_prefix(monkeypatch):
    model = get_model(
        monkeypatch,
        pooler_type="cls_token",
        batch_size=7,
        seq_len=22,
        num_classes=4,
        add_dummy_linear=True,
        base_model_prefix=None,
    )
    with pytest.raises(ValueError) as excinfo:
        model.base_model_named_parameters()
    assert (
        str(excinfo.value) == "base_model_prefix has to be set to select the base model parameters"
    )


def test_base_model_named_parameters_wrong_prefix(monkeypatch):
    model = get_model(
        monkeypatch,
        pooler_type="cls_token",
        batch_size=7,
        seq_len=22,
        num_classes=4,
        add_dummy_linear=True,
        base_model_prefix="wrong_prefix",
    )
    with pytest.raises(ValueError) as excinfo:
        model.base_model_named_parameters()
    assert (
        str(excinfo.value)
        == "No base model parameters found. Is base_model_prefix=wrong_prefix for MockModel correct?"
    )
