import pytest
import torch
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPooling

from pie_models.models import SequenceClassificationModel


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


@pytest.fixture(params=["cls_token", "mention_pooling", "start_tokens"])
def pooler_type(request):
    return request.param


def get_model(
    monkeypatch,
    pooler_type,
    batch_size,
    seq_len,
    num_classes,
    add_dummy_linear=False,
    **model_kwargs,
):
    class MockConfig:
        def __init__(self, hidden_size: int = 10, classifier_dropout: float = 1.0) -> None:
            self.hidden_size = hidden_size
            self.classifier_dropout = classifier_dropout

    class MockModel(torch.nn.Module):
        def __init__(self, batch_size, seq_len, hidden_size, add_dummy_linear) -> None:
            super().__init__()
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.hidden_size = hidden_size
            if add_dummy_linear:
                self.dummy_linear = torch.nn.Linear(self.hidden_size, 99)

        def __call__(self, *args, **kwargs):
            last_hidden_state = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
            return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state)

        def resize_token_embeddings(self, new_num_tokens):
            pass

    hidden_size = 10
    tokenizer_vocab_size = 30000

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda model_name_or_path: MockConfig(hidden_size=hidden_size, classifier_dropout=1.0),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda model_name_or_path, config: MockModel(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            add_dummy_linear=add_dummy_linear,
        ),
    )

    # set seed to make the classifier deterministic
    torch.manual_seed(42)
    result = SequenceClassificationModel(
        model_name_or_path="some-model-name",
        num_classes=num_classes,
        tokenizer_vocab_size=tokenizer_vocab_size,
        ignore_index=0,
        pooler=pooler_type,
        # disable warmup because it would require a trainer and a datamodule to get the total number of training steps
        warmup_proportion=0.0,
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
        num_classes=max(targets) + 1,
    )


def test_forward(inputs, model, pooler_type):
    batch_size, seq_len = inputs["input_ids"].shape
    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    output = model.forward(inputs)
    assert set(output) == {"logits"}
    logits = output["logits"]

    assert logits.shape == (batch_size, 4)

    if pooler_type == "cls_token":
        torch.testing.assert_close(
            logits,
            torch.tensor(
                [
                    [
                        1.1392780542373657,
                        0.3456377387046814,
                        -0.2497227042913437,
                        0.7444550395011902,
                    ],
                    [
                        0.5357180833816528,
                        0.2821698784828186,
                        -0.2038819044828415,
                        0.572994589805603,
                    ],
                    [
                        0.9346485733985901,
                        0.30422478914260864,
                        -0.30148035287857056,
                        0.5211007595062256,
                    ],
                    [
                        0.6590405106544495,
                        0.19314078986644745,
                        -0.16493697464466095,
                        0.4076993465423584,
                    ],
                    [
                        0.45989543199539185,
                        0.6741790175437927,
                        -0.5770877003669739,
                        0.6692476272583008,
                    ],
                    [
                        0.6195374131202698,
                        0.2635548710823059,
                        -0.30177515745162964,
                        0.469584584236145,
                    ],
                    [
                        0.3063261806964874,
                        0.4225810766220093,
                        -0.33908766508102417,
                        0.547605037689209,
                    ],
                ]
            ),
        )
    elif pooler_type == "start_tokens":
        torch.testing.assert_close(
            logits,
            torch.tensor(
                [
                    [
                        0.28298091888427734,
                        -0.16016829013824463,
                        -0.0061178505420684814,
                        -0.9113596081733704,
                    ],
                    [
                        0.25231730937957764,
                        -0.10320538282394409,
                        -0.24115778505802155,
                        -0.444608211517334,
                    ],
                    [
                        0.38434189558029175,
                        -0.6033056974411011,
                        -0.2230159193277359,
                        -0.7633178234100342,
                    ],
                    [
                        0.38798075914382935,
                        -0.24908408522605896,
                        -0.18538469076156616,
                        -0.46249520778656006,
                    ],
                    [
                        0.3724908232688904,
                        -0.37968626618385315,
                        -0.06202100217342377,
                        -0.9619439244270325,
                    ],
                    [
                        0.35508251190185547,
                        -0.43490925431251526,
                        -0.1275191456079483,
                        -1.0958456993103027,
                    ],
                    [
                        0.21123524010181427,
                        -0.445130854845047,
                        -0.024221107363700867,
                        -0.9154050350189209,
                    ],
                ]
            ),
        )
    elif pooler_type == "mention_pooling":
        torch.testing.assert_close(
            logits,
            torch.tensor(
                [
                    [
                        0.44285398721694946,
                        -0.2843514680862427,
                        -0.14574748277664185,
                        -0.6859760880470276,
                    ],
                    [
                        0.4771193563938141,
                        -0.572691798210144,
                        -0.2516267001628876,
                        -1.1906213760375977,
                    ],
                    [
                        0.35073068737983704,
                        -0.24320194125175476,
                        0.028778836131095886,
                        -0.9210844039916992,
                    ],
                    [
                        0.38173380494117737,
                        -0.44920578598976135,
                        -0.2865368127822876,
                        -0.5453884601593018,
                    ],
                    [
                        0.4319082498550415,
                        -0.42361000180244446,
                        -0.15595994889736176,
                        -0.8779217600822449,
                    ],
                    [
                        0.1349087655544281,
                        -0.3701835870742798,
                        -0.37905648350715637,
                        -0.7094825506210327,
                    ],
                    [
                        0.4267612099647522,
                        -0.39893561601638794,
                        -0.32917478680610657,
                        -0.7729455828666687,
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
        torch.testing.assert_close(loss, torch.tensor(1.4302054643630981))
    elif pooler_type == "start_tokens":
        torch.testing.assert_close(loss, torch.tensor(1.514051079750061))
    elif pooler_type == "mention_pooling":
        torch.testing.assert_close(loss, torch.tensor(1.5418565273284912))
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
    base_model_params = list(model.model.parameters())
    # the dummy linear from the mock base model has 2 parameters
    assert len(base_model_params) == 2
    for param in base_model_params:
        assert not param.requires_grad
