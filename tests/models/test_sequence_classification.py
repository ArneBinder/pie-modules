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
    model_type="bert",
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
        lambda model_name_or_path: MockConfig(
            hidden_size=hidden_size, classifier_dropout=0.1, model_type=model_type
        ),
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
        num_classes=int(max(targets) + 1),
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
                        1.25642395019531250000,
                        0.39357030391693115234,
                        -0.29225713014602661133,
                        0.79580175876617431641,
                    ],
                    [
                        0.53891825675964355469,
                        0.34787857532501220703,
                        -0.24634249508380889893,
                        0.59947609901428222656,
                    ],
                    [
                        0.74169844388961791992,
                        0.30519056320190429688,
                        -0.55728095769882202148,
                        0.49557113647460937500,
                    ],
                    [
                        0.35605597496032714844,
                        0.19517414271831512451,
                        -0.00861304998397827148,
                        0.65302681922912597656,
                    ],
                    [
                        0.29554772377014160156,
                        0.71216350793838500977,
                        -0.62688910961151123047,
                        0.92307460308074951172,
                    ],
                    [
                        0.67893451452255249023,
                        0.30236703157424926758,
                        -0.35009318590164184570,
                        0.49039006233215332031,
                    ],
                    [
                        0.33092185854911804199,
                        0.47906285524368286133,
                        -0.39155155420303344727,
                        0.57707947492599487305,
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
                        0.28744211792945861816,
                        -0.07656848430633544922,
                        0.06205615401268005371,
                        -0.94508385658264160156,
                    ],
                    [
                        0.29196885228157043457,
                        0.02899619936943054199,
                        -0.21342960000038146973,
                        -0.36053514480590820312,
                    ],
                    [
                        0.36177605390548706055,
                        -0.64715611934661865234,
                        -0.26786345243453979492,
                        -0.80762034654617309570,
                    ],
                    [
                        0.42720466852188110352,
                        -0.25489825010299682617,
                        -0.19527786970138549805,
                        -0.49900960922241210938,
                    ],
                    [
                        0.20688854157924652100,
                        -0.29307979345321655273,
                        -0.12208836525678634644,
                        -0.89110243320465087891,
                    ],
                    [
                        0.35013008117675781250,
                        -0.49105945229530334473,
                        -0.18206793069839477539,
                        -1.19002366065979003906,
                    ],
                    [
                        0.31203818321228027344,
                        -0.37706983089447021484,
                        0.07198116183280944824,
                        -0.81837034225463867188,
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
                        0.48370990157127380371,
                        -0.27870815992355346680,
                        -0.13999497890472412109,
                        -0.73714041709899902344,
                    ],
                    [
                        0.54668211936950683594,
                        -0.29652747511863708496,
                        -0.26315566897392272949,
                        -0.95955950021743774414,
                    ],
                    [
                        0.22266633808612823486,
                        -0.24484989047050476074,
                        -0.03910681605339050293,
                        -0.94041651487350463867,
                    ],
                    [
                        0.42026358842849731445,
                        -0.47725573182106018066,
                        -0.30766916275024414062,
                        -0.59111309051513671875,
                    ],
                    [
                        0.23630522191524505615,
                        -0.29734912514686584473,
                        -0.30620723962783813477,
                        -0.75251650810241699219,
                    ],
                    [
                        0.14205220341682434082,
                        -0.39235562086105346680,
                        -0.41546288132667541504,
                        -0.77219748497009277344,
                    ],
                    [
                        0.44709876179695129395,
                        -0.20209559798240661621,
                        -0.18925097584724426270,
                        -0.64976799488067626953,
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
        torch.testing.assert_close(loss, torch.tensor(1.417838096618652))
    elif pooler_type == "start_tokens":
        torch.testing.assert_close(loss, torch.tensor(1.498929619789123))
    elif pooler_type == "mention_pooling":
        torch.testing.assert_close(loss, torch.tensor(1.489617109298706))
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
