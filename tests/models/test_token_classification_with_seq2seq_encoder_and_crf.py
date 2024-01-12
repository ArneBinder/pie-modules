import pytest
import torch
import transformers
from pytorch_ie.taskmodules import TransformerTokenClassificationTaskModule
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from pie_modules.models import TokenClassificationModelWithSeq2SeqEncoderAndCrf
from pie_modules.models.components.seq2seq_encoder import RNN_TYPE2CLASS
from pie_modules.models.token_classification_with_seq2seq_encoder_and_crf import (
    HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE,
)
from tests import _config_to_str

CONFIGS = [{}, {"use_crf": False}]
CONFIG_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIG_DICT.keys())
def config_str(request):
    return request.param


@pytest.fixture(scope="module")
def config(config_str):
    return CONFIG_DICT[config_str]


@pytest.mark.skip(reason="Only to recreate the batch if taskmodule has changed")
def test_batch(documents, batch):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TransformerTokenClassificationTaskModule(
        entity_annotation="entities",
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    taskmodule.prepare(documents)
    encodings = taskmodule.encode(documents, encode_target=True, as_dataset=True)
    batch_from_documents = taskmodule.collate(encodings[:4])

    inputs, targets = batch
    inputs_from_documents, targets_from_documents = batch_from_documents
    torch.testing.assert_close(inputs["input_ids"], inputs_from_documents["input_ids"])
    torch.testing.assert_close(inputs["attention_mask"], inputs_from_documents["attention_mask"])
    torch.testing.assert_close(targets, targets_from_documents)


@pytest.fixture
def batch():
    inputs = {
        "input_ids": torch.tensor(
            [
                [101, 138, 1423, 5650, 119, 102, 0, 0, 0, 0, 0, 0],
                [101, 13832, 3121, 2340, 138, 1759, 1120, 139, 119, 102, 0, 0],
                [101, 13832, 3121, 2340, 140, 1105, 141, 119, 102, 0, 0, 0],
                [101, 1752, 5650, 119, 13832, 3121, 2340, 142, 1105, 143, 119, 102],
            ]
        ).to(torch.long),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ),
    }
    targets = torch.tensor(
        [
            [-100, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100],
            [-100, 3, 4, 4, 4, 0, 0, 1, 0, -100, -100, -100],
            [-100, 3, 4, 4, 4, 0, 1, 0, -100, -100, -100, -100],
            [-100, 0, 0, 0, 3, 4, 4, 4, 0, 1, 0, -100],
        ]
    )
    return inputs, targets


def get_model(
    monkeypatch,
    model_type,
    num_classes,
    batch_size,
    seq_len,
    add_dummy_linear=False,
    **model_kwargs,
):
    class MockConfig:
        def __init__(
            self,
            hidden_size: int = 10,
            classifier_dropout: float = 0.1,
            model_type=model_type,
        ) -> None:
            self.hidden_size = hidden_size
            self.model_type = model_type
            classifier_dropout_attr = HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE.get(
                model_type, None
            )
            if classifier_dropout_attr is not None:
                setattr(self, classifier_dropout_attr, classifier_dropout)

    class MockModel(torch.nn.Module):
        def __init__(self, batch_size, seq_len, hidden_size, add_dummy_linear) -> None:
            super().__init__()
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.hidden_size = hidden_size
            if add_dummy_linear:
                self.dummy_linear = torch.nn.Linear(self.hidden_size, 99)

        def __call__(self, *args, **kwargs):
            last_hidden_state = torch.FloatTensor(
                torch.rand(self.batch_size, self.seq_len, self.hidden_size)
            )
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=last_hidden_state,
            )

    hidden_size = 10

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
    result = TokenClassificationModelWithSeq2SeqEncoderAndCrf(
        model_name_or_path="bert",
        num_classes=num_classes,
        warmup_proportion=0.0,
        ignore_index=-100,
        **model_kwargs,
    )
    assert not result.is_from_pretrained

    return result


@pytest.fixture
def model(monkeypatch, batch, config) -> TokenClassificationModelWithSeq2SeqEncoderAndCrf:
    inputs, targets = batch
    seq2seq_dict = {
        "type": "linear",
        "out_features": 10,
    }
    model = get_model(
        monkeypatch=monkeypatch,
        model_type="bert",
        batch_size=inputs["input_ids"].shape[0],
        seq_len=inputs["input_ids"].shape[1],
        num_classes=int(torch.max(targets) + 1),
        seq2seq_encoder=seq2seq_dict,
        **config,
    )
    return model


@pytest.mark.parametrize(
    "model_type", list(HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE) + ["unknown"]
)
def test_config_model_classifier_dropout(monkeypatch, model_type):
    if model_type in HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE:
        model = get_model(
            monkeypatch=monkeypatch,
            model_type=model_type,
            batch_size=4,
            seq_len=10,
            num_classes=5,
        )
        assert model is not None
        assert isinstance(model, TokenClassificationModelWithSeq2SeqEncoderAndCrf)
    else:
        with pytest.raises(ValueError):
            model = get_model(
                monkeypatch=monkeypatch,
                model_type=model_type,
                batch_size=4,
                seq_len=10,
                num_classes=5,
            )


@pytest.mark.parametrize("seq2seq_enc_type", list(RNN_TYPE2CLASS))
def test_seq2seq_classification_head(monkeypatch, seq2seq_enc_type):
    seq2seq_dict = {
        "type": seq2seq_enc_type,
        "hidden_size": 10,
    }
    model = get_model(
        monkeypatch,
        model_type="bert",
        batch_size=4,
        seq_len=10,
        num_classes=5,
        freeze_base_model=True,
        seq2seq_encoder=seq2seq_dict,
    )

    assert model is not None
    assert isinstance(model, TokenClassificationModelWithSeq2SeqEncoderAndCrf)
    assert isinstance(model.seq2seq_encoder.rnn, RNN_TYPE2CLASS[seq2seq_enc_type])


def test_freeze_base_model(monkeypatch):
    model = get_model(
        monkeypatch,
        model_type="bert",
        batch_size=4,
        seq_len=10,
        num_classes=5,
        add_dummy_linear=True,
        freeze_base_model=True,
    )
    base_model_params = dict(model.model.named_parameters(prefix="model"))
    assert len(base_model_params) > 0
    for param in base_model_params.values():
        assert not param.requires_grad
    task_params = {
        name: param for name, param in model.named_parameters() if name not in base_model_params
    }
    assert len(task_params) > 0
    for param in task_params.values():
        assert param.requires_grad


def test_tune_base_model(monkeypatch):
    model = get_model(
        monkeypatch,
        model_type="bert",
        batch_size=4,
        seq_len=10,
        num_classes=5,
        add_dummy_linear=True,
        freeze_base_model=False,
    )
    base_model_params = dict(model.model.named_parameters(prefix="model"))
    assert len(base_model_params) > 0
    for param in base_model_params.values():
        assert param.requires_grad
    task_params = {
        name: param for name, param in model.named_parameters() if name not in base_model_params
    }
    assert len(task_params) > 0
    for param in task_params.values():
        assert param.requires_grad


def test_forward(batch, model):
    inputs, targets = batch
    batch_size, seq_len = inputs["input_ids"].shape
    num_classes = int(torch.max(targets) + 1)

    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    output = model.forward(inputs)
    assert set(output) == {"logits"}
    logits = output["logits"]
    assert logits.shape == (batch_size, seq_len, num_classes)
    # check the first batch entry
    torch.testing.assert_close(
        logits[0],
        torch.tensor(
            [
                [
                    0.5665707588195801,
                    -0.2936006188392639,
                    0.6836847066879272,
                    -0.8939268589019775,
                    0.07820314168930054,
                ],
                [
                    0.24385401606559753,
                    -0.16216063499450684,
                    0.603381872177124,
                    -0.9981566667556763,
                    0.18869800865650177,
                ],
                [
                    0.4842372238636017,
                    -0.08902332186698914,
                    0.3403770625591278,
                    -0.6819748878479004,
                    0.08253003656864166,
                ],
                [
                    0.04966197907924652,
                    -0.32139256596565247,
                    0.5452693700790405,
                    -1.0789272785186768,
                    0.019466541707515717,
                ],
                [
                    0.2986313998699188,
                    -0.24962571263313293,
                    0.5462854504585266,
                    -0.9566639065742493,
                    0.11756543815135956,
                ],
                [
                    0.48104530572891235,
                    -0.06864413619041443,
                    0.5099285840988159,
                    -0.8464795351028442,
                    -0.02686941623687744,
                ],
                [
                    0.2378637194633484,
                    -0.04355515539646149,
                    0.40335217118263245,
                    -0.8488281965255737,
                    0.15444552898406982,
                ],
                [
                    0.20840872824192047,
                    -0.2112131416797638,
                    0.46829894185066223,
                    -0.744867742061615,
                    0.25981956720352173,
                ],
                [
                    0.232809916138649,
                    0.0795224979519844,
                    0.3929428160190582,
                    -0.7533693313598633,
                    0.2106921672821045,
                ],
                [
                    0.1851186901330948,
                    -0.1450422704219818,
                    0.2970556616783142,
                    -0.7189601063728333,
                    0.19078847765922546,
                ],
                [
                    0.48274922370910645,
                    0.11437579989433289,
                    0.3290281593799591,
                    -0.5824930667877197,
                    0.22612173855304718,
                ],
                [
                    0.2992321252822876,
                    -0.10336627066135406,
                    0.3752039670944214,
                    -0.7740738391876221,
                    0.2034076750278473,
                ],
            ]
        ),
    )

    # check the sums per sequence
    torch.testing.assert_close(
        logits.sum(1),
        torch.tensor(
            [
                [
                    3.770183563232422,
                    -1.4937254190444946,
                    5.494809150695801,
                    -9.878721237182617,
                    1.7048689126968384,
                ],
                [
                    4.664599418640137,
                    -0.9193091988563538,
                    5.376185417175293,
                    -8.726889610290527,
                    1.5600370168685913,
                ],
                [
                    4.443399906158447,
                    -0.6660721302032471,
                    5.095791816711426,
                    -8.66091537475586,
                    2.086989402770996,
                ],
                [
                    4.494462013244629,
                    -0.5705814957618713,
                    6.037875175476074,
                    -8.867290496826172,
                    1.831087589263916,
                ],
            ]
        ),
    )


def test_step(batch, model, config):
    torch.manual_seed(42)
    loss = model.step("train", batch)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(56.7139))
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.570178747177124))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_training_step(batch, model, config):
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(56.78658676147461))
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.5575151443481445))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_training_step_without_attention_mask(batch, model, config):
    inputs, targets = batch
    inputs_without_attention_mask = {k: v for k, v in inputs.items() if k != "attention_mask"}
    loss = model.training_step(batch=(inputs_without_attention_mask, targets), batch_idx=0)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(72.02947235107422))
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.5575151443481445))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_validation_step(batch, model, config):
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(56.78658676147461))
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.5575151443481445))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_test_step(batch, model, config):
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(56.78658676147461))
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.5575151443481445))
    else:
        raise ValueError(f"Unknown config: {config}")


@pytest.mark.parametrize("test_step", [False, True])
def test_predict_and_predict_step(model, batch, config, test_step):
    torch.manual_seed(42)
    if test_step:
        predictions = model.predict_step(batch, batch_idx=0, dataloader_idx=0)
    else:
        predictions = model.predict(batch[0])
    assert predictions.shape == batch[1].shape
    if config == {}:
        torch.testing.assert_close(
            predictions,
            torch.tensor(
                [
                    [2, 2, 0, 2, 2, 2, -100, -100, -100, -100, -100, -100],
                    [2, 0, 2, 2, 0, 2, 2, 2, 0, 2, -100, -100],
                    [2, 0, 2, 0, 2, 0, 2, 0, 2, -100, -100, -100],
                    [2, 0, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2],
                ]
            ),
        )
    elif config == {"use_crf": False}:
        torch.testing.assert_close(
            predictions,
            torch.tensor(
                [
                    [2, 2, 0, 2, 2, 2, -100, -100, -100, -100, -100, -100],
                    [0, 0, 2, 2, 0, 2, 2, 2, 2, 2, -100, -100],
                    [2, 2, 2, 0, 2, 2, 0, 0, 0, -100, -100, -100],
                    [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                ]
            ),
        )
    else:
        raise ValueError(f"Unknown config: {config}")


def test_predict_without_attention_mask(model, batch, config):
    inputs, targets = batch
    inputs_without_attention_mask = {k: v for k, v in inputs.items() if k != "attention_mask"}
    torch.manual_seed(42)
    predictions = model.predict(inputs_without_attention_mask)
    assert predictions.shape == targets.shape
    if config == {}:
        torch.testing.assert_close(
            predictions,
            torch.tensor(
                [
                    [2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 0, 2],
                    [2, 0, 2, 2, 0, 2, 2, 2, 0, 2, 0, 2],
                    [2, 0, 2, 0, 2, 2, 0, 2, 0, 2, 0, 2],
                    [2, 0, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2],
                ]
            ),
        )
    elif config == {"use_crf": False}:
        torch.testing.assert_close(
            predictions,
            torch.tensor(
                [
                    [2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2],
                    [0, 0, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2],
                    [2, 2, 2, 0, 2, 2, 0, 0, 0, 2, 2, 2],
                    [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                ]
            ),
        )
    else:
        raise ValueError(f"Unknown config: {config}")


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
        model_type="bert",
        batch_size=4,
        seq_len=10,
        num_classes=5,
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
    # the classifier head has 5 parameters
    assert len(param_group["params"]) == 5
    assert param_group["params"][0].shape == torch.Size([5, 10])
    assert param_group["params"][1].shape == torch.Size([5])
