import pytest
import torch
from pytorch_lightning import Trainer

from pie_modules.models import TokenClassificationModelWithSeq2SeqEncoderAndCrf
from pie_modules.models.common import TESTING, TRAINING, VALIDATION
from pie_modules.taskmodules import LabeledSpanExtractionByTokenClassificationTaskModule
from tests import _config_to_str

CONFIGS = [{}, {"use_crf": False}]
CONFIG_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIG_DICT.keys())
def config_str(request):
    return request.param


@pytest.fixture(scope="module")
def config(config_str):
    return CONFIG_DICT[config_str]


@pytest.fixture
def taskmodule_config():
    return {
        "taskmodule_type": "LabeledSpanExtractionByTokenClassificationTaskModule",
        "tokenizer_name_or_path": "bert-base-cased",
        "span_annotation": "entities",
        "partition_annotation": None,
        "label_pad_id": -100,
        "labels": ["ORG", "PER"],
        "include_ill_formed_predictions": True,
        "tokenize_kwargs": None,
        "pad_kwargs": None,
        "log_precision_recall_metrics": True,
    }


def test_taskmodule_config(documents, taskmodule_config):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = LabeledSpanExtractionByTokenClassificationTaskModule(
        span_annotation="entities",
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    taskmodule.prepare(documents)
    assert taskmodule.config == taskmodule_config


def test_batch(documents, batch, taskmodule_config):
    taskmodule = LabeledSpanExtractionByTokenClassificationTaskModule.from_config(
        taskmodule_config
    )
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
        "special_tokens_mask": torch.tensor(
            [
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
    }
    targets = {
        "labels": torch.tensor(
            [
                [-100, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100],
                [-100, 3, 4, 4, 4, 0, 0, 1, 0, -100, -100, -100],
                [-100, 3, 4, 4, 4, 0, 1, 0, -100, -100, -100, -100],
                [-100, 0, 0, 0, 3, 4, 4, 4, 0, 1, 0, -100],
            ]
        )
    }
    return inputs, targets


@pytest.fixture
def model(batch, config, taskmodule_config) -> TokenClassificationModelWithSeq2SeqEncoderAndCrf:
    seq2seq_dict = {
        "type": "linear",
        "out_features": 10,
    }
    torch.manual_seed(42)
    model = TokenClassificationModelWithSeq2SeqEncoderAndCrf(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=5,
        seq2seq_encoder=seq2seq_dict,
        taskmodule_config=taskmodule_config,
        metric_stages=["val", "test"],
        **config,
    )
    return model


def test_model(model, config):
    assert model is not None
    named_parameters = dict(model.named_parameters())
    parameter_means = {k: round(v.mean().item(), 8) for k, v in named_parameters.items()}
    parameter_means_expected = {
        "model.embeddings.word_embeddings.weight": 0.00311526,
        "model.embeddings.position_embeddings.weight": 5.503e-05,
        "model.embeddings.token_type_embeddings.weight": -0.00154195,
        "model.embeddings.LayerNorm.weight": 1.31234503,
        "model.embeddings.LayerNorm.bias": -0.02946087,
        "model.encoder.layer.0.attention.self.query.weight": -0.00039493,
        "model.encoder.layer.0.attention.self.query.bias": 0.01857448,
        "model.encoder.layer.0.attention.self.key.weight": 0.00038633,
        "model.encoder.layer.0.attention.self.key.bias": 0.00205573,
        "model.encoder.layer.0.attention.self.value.weight": 4.223e-05,
        "model.encoder.layer.0.attention.self.value.bias": 0.00654173,
        "model.encoder.layer.0.attention.output.dense.weight": 3.018e-05,
        "model.encoder.layer.0.attention.output.dense.bias": 0.0007209,
        "model.encoder.layer.0.attention.output.LayerNorm.weight": 1.19983101,
        "model.encoder.layer.0.attention.output.LayerNorm.bias": 0.06087145,
        "model.encoder.layer.0.intermediate.dense.weight": -0.00117319,
        "model.encoder.layer.0.intermediate.dense.bias": -0.12199588,
        "model.encoder.layer.0.output.dense.weight": -0.00022129,
        "model.encoder.layer.0.output.dense.bias": -0.00130311,
        "model.encoder.layer.0.output.LayerNorm.weight": 1.24196482,
        "model.encoder.layer.0.output.LayerNorm.bias": 0.00529504,
        "model.encoder.layer.1.attention.self.query.weight": -0.00073216,
        "model.encoder.layer.1.attention.self.query.bias": -0.03583976,
        "model.encoder.layer.1.attention.self.key.weight": 0.00013331,
        "model.encoder.layer.1.attention.self.key.bias": 0.00450627,
        "model.encoder.layer.1.attention.self.value.weight": 0.00010125,
        "model.encoder.layer.1.attention.self.value.bias": -0.0007095,
        "model.encoder.layer.1.attention.output.dense.weight": -2.43e-05,
        "model.encoder.layer.1.attention.output.dense.bias": 0.00414465,
        "model.encoder.layer.1.attention.output.LayerNorm.weight": 1.03773439,
        "model.encoder.layer.1.attention.output.LayerNorm.bias": 0.04432376,
        "model.encoder.layer.1.intermediate.dense.weight": -0.00134408,
        "model.encoder.layer.1.intermediate.dense.bias": -0.12472576,
        "model.encoder.layer.1.output.dense.weight": -5.323e-05,
        "model.encoder.layer.1.output.dense.bias": 0.00067709,
        "model.encoder.layer.1.output.LayerNorm.weight": 1.01716208,
        "model.encoder.layer.1.output.LayerNorm.bias": -0.04744423,
        "model.pooler.dense.weight": 0.00012952,
        "model.pooler.dense.bias": -0.00520781,
        "seq2seq_encoder.weight": -0.00439181,
        "seq2seq_encoder.bias": 0.01506435,
        "classifier.weight": 0.00453858,
        "classifier.bias": 0.13226676,
    }
    if config.get("use_crf", True):
        parameter_means_expected.update(
            {
                "crf.start_transitions": 0.0436484,
                "crf.end_transitions": -0.00948838,
                "crf.transitions": 0.00680668,
            }
        )
    assert parameter_means == parameter_means_expected


def test_model_pickleable(model):
    import pickle

    pickle.dumps(model)


def test_freeze_base_model():
    model = TokenClassificationModelWithSeq2SeqEncoderAndCrf(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=5,
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


def test_tune_base_model():
    model = TokenClassificationModelWithSeq2SeqEncoderAndCrf(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=5,
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
    num_classes = int(torch.max(targets["labels"]) + 1)

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
                    0.2983264923095703,
                    0.2825356125831604,
                    0.07071954011917114,
                    0.18499699234962463,
                    0.33343958854675293,
                ],
                [
                    0.040096089243888855,
                    0.30112865567207336,
                    -0.04935203492641449,
                    0.21017718315124512,
                    0.09015939384698868,
                ],
                [
                    0.06823589652776718,
                    0.10778219252824783,
                    -0.15539433062076569,
                    0.5013774633407593,
                    -0.40293776988983154,
                ],
                [
                    -0.056031033396720886,
                    0.5694245100021362,
                    0.4017048478126526,
                    0.057996511459350586,
                    -0.05365113914012909,
                ],
                [
                    -0.03769463300704956,
                    0.4193854331970215,
                    0.27049922943115234,
                    0.1521742194890976,
                    0.30935269594192505,
                ],
                [
                    0.24729765951633453,
                    0.20631808042526245,
                    0.15089815855026245,
                    0.567837119102478,
                    -0.2030831128358841,
                ],
                [
                    0.21842727065086365,
                    0.01828605681657791,
                    0.2380525916814804,
                    0.2717846930027008,
                    0.32562902569770813,
                ],
                [
                    0.09538321942090988,
                    0.2377324104309082,
                    -0.03270860016345978,
                    0.26029419898986816,
                    0.13534477353096008,
                ],
                [
                    -0.0025518983602523804,
                    0.11825758963823318,
                    0.039080917835235596,
                    0.3058757483959198,
                    -0.08563672006130219,
                ],
                [
                    -0.022532835602760315,
                    0.05094567686319351,
                    0.006704658269882202,
                    0.37255239486694336,
                    -0.16184081137180328,
                ],
                [
                    0.08076323568820953,
                    -0.08618196099996567,
                    -0.021112561225891113,
                    0.4702887535095215,
                    -0.24479801952838898,
                ],
                [
                    0.08460880815982819,
                    -0.12512007355690002,
                    -0.25200968980789185,
                    0.48507219552993774,
                    -0.17945758998394012,
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
                    1.0143283605575562,
                    2.100494146347046,
                    0.6670827269554138,
                    3.8404273986816406,
                    -0.1374797224998474,
                ],
                [
                    4.008062839508057,
                    -5.860606670379639,
                    4.935301303863525,
                    6.916836261749268,
                    1.7625821828842163,
                ],
                [
                    2.8068482875823975,
                    -7.062673091888428,
                    4.238302707672119,
                    8.642420768737793,
                    0.19193750619888306,
                ],
                [
                    0.5242522358894348,
                    -3.213310480117798,
                    4.6281819343566895,
                    7.656630992889404,
                    -0.40581274032592773,
                ],
            ]
        ),
    )


def test_step(batch, model, config):
    torch.manual_seed(42)
    loss = model._step("train", batch)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(59.975791931152344))
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.6528030633926392))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_training_step_and_on_epoch_end(batch, model, config):
    metric = model._get_metric(TRAINING, batch_idx=0)
    assert metric is None
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(59.42658996582031))
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.6708829402923584))
    else:
        raise ValueError(f"Unknown config: {config}")

    model.on_train_epoch_end()


def test_training_step_without_attention_mask(batch, model, config):
    inputs, targets = batch
    inputs_without_attention_mask = {k: v for k, v in inputs.items() if k != "attention_mask"}
    loss = model.training_step(batch=(inputs_without_attention_mask, targets), batch_idx=0)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(77.98155975341797))
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.6701185703277588))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_validation_step_and_on_epoch_end(batch, model, config):
    metric = model._get_metric(VALIDATION, batch_idx=0)
    metric.reset()
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    metric_values = {k: v.item() for k, v in metric.compute().items()}
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(59.42658996582031))
        assert metric_values == {
            "token/macro/f1": 0.3919413983821869,
            "token/micro/f1": 0.5333333611488342,
            "span/PER/f1": 0.0833333358168602,
            "span/PER/recall": 0.0476190485060215,
            "span/PER/precision": 0.3333333432674408,
            "span/ORG/f1": 0.0,
            "span/ORG/recall": 0.0,
            "span/ORG/precision": 0.0,
            "span/macro/f1": 0.0416666679084301,
            "span/macro/precision": 0.1666666716337204,
            "span/macro/recall": 0.02380952425301075,
            "span/micro/f1": 0.06666667014360428,
            "span/micro/recall": 0.0416666679084301,
            "span/micro/precision": 0.1666666716337204,
        }
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.6708829402923584))
        assert metric_values == {
            "token/macro/f1": 0.14374999701976776,
            "token/micro/f1": 0.2666666805744171,
            "span/PER/f1": 0.0,
            "span/PER/recall": 0.0,
            "span/PER/precision": 0.0,
            "span/ORG/f1": 0.0,
            "span/ORG/recall": 0.0,
            "span/ORG/precision": 0.0,
            "span/macro/f1": 0.0,
            "span/macro/precision": 0.0,
            "span/macro/recall": 0.0,
            "span/micro/f1": 0.0,
            "span/micro/recall": 0.0,
            "span/micro/precision": 0.0,
        }
    else:
        raise ValueError(f"Unknown config: {config}")

    model.on_validation_epoch_end()


def test_test_step_and_on_epoch_end(batch, model, config):
    metric = model._get_metric(TESTING, batch_idx=0)
    metric.reset()
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    metric_values = {k: v.item() for k, v in metric.compute().items()}
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(59.42658996582031))
        assert metric_values == {
            "token/macro/f1": 0.3919413983821869,
            "token/micro/f1": 0.5333333611488342,
            "span/ORG/f1": 0.0,
            "span/ORG/recall": 0.0,
            "span/ORG/precision": 0.0,
            "span/PER/f1": 0.0833333358168602,
            "span/PER/recall": 0.0476190485060215,
            "span/PER/precision": 0.3333333432674408,
            "span/macro/f1": 0.0416666679084301,
            "span/macro/precision": 0.1666666716337204,
            "span/macro/recall": 0.02380952425301075,
            "span/micro/f1": 0.06666667014360428,
            "span/micro/recall": 0.0416666679084301,
            "span/micro/precision": 0.1666666716337204,
        }
    elif config == {"use_crf": False}:
        torch.testing.assert_close(loss, torch.tensor(1.6708829402923584))
        assert metric_values == {
            "token/macro/f1": 0.14374999701976776,
            "token/micro/f1": 0.2666666805744171,
            "span/ORG/f1": 0.0,
            "span/ORG/recall": 0.0,
            "span/ORG/precision": 0.0,
            "span/PER/f1": 0.0,
            "span/PER/recall": 0.0,
            "span/PER/precision": 0.0,
            "span/macro/f1": 0.0,
            "span/macro/precision": 0.0,
            "span/macro/recall": 0.0,
            "span/micro/f1": 0.0,
            "span/micro/recall": 0.0,
            "span/micro/precision": 0.0,
        }
    else:
        raise ValueError(f"Unknown config: {config}")

    model.on_test_epoch_end()


@pytest.mark.parametrize("test_step", [False, True])
def test_predict_and_predict_step(model, batch, config, test_step):
    torch.manual_seed(42)
    if test_step:
        predictions = model.predict_step(batch, batch_idx=0, dataloader_idx=0)
    else:
        predictions = model.predict(batch[0])

    assert set(predictions) == {"labels", "probabilities"}
    labels = predictions["labels"]
    probabilities = predictions["probabilities"]
    if config == {}:
        torch.testing.assert_close(
            labels,
            torch.tensor(
                [
                    [-100, 1, 3, 1, 1, -100, -100, -100, -100, -100, -100, -100],
                    [-100, 3, 4, 4, 0, 3, 3, 3, 2, -100, -100, -100],
                    [-100, 3, 4, 3, 3, 3, 3, 3, -100, -100, -100, -100],
                    [-100, 3, 2, 4, 3, 3, 3, 3, 3, 3, 3, -100],
                ]
            ),
        )
    elif config == {"use_crf": False}:
        torch.testing.assert_close(
            labels,
            torch.tensor(
                [
                    [-100, 1, 3, 1, 1, -100, -100, -100, -100, -100, -100, -100],
                    [-100, 3, 4, 3, 3, 3, 3, 3, 2, -100, -100, -100],
                    [-100, 3, 4, 3, 3, 3, 3, 3, -100, -100, -100, -100],
                    [-100, 3, 2, 4, 3, 3, 3, 3, 3, 2, 3, -100],
                ]
            ),
        )
    else:
        raise ValueError(f"Unknown config: {config}")

    assert labels.shape == batch[1]["labels"].shape
    torch.testing.assert_close(
        probabilities[:2].round(decimals=4),
        torch.tensor(
            [
                [
                    [0.2123, 0.2090, 0.1691, 0.1896, 0.2199],
                    [0.1835, 0.2382, 0.1678, 0.2175, 0.1929],
                    [0.1997, 0.2078, 0.1597, 0.3080, 0.1247],
                    [0.1521, 0.2844, 0.2405, 0.1705, 0.1525],
                    [0.1523, 0.2406, 0.2073, 0.1842, 0.2155],
                    [0.2048, 0.1966, 0.1860, 0.2822, 0.1305],
                    [0.1997, 0.1635, 0.2037, 0.2107, 0.2223],
                    [0.1904, 0.2195, 0.1675, 0.2245, 0.1981],
                    [0.1834, 0.2070, 0.1912, 0.2497, 0.1688],
                    [0.1831, 0.1971, 0.1886, 0.2719, 0.1593],
                    [0.2021, 0.1710, 0.1825, 0.2984, 0.1459],
                    [0.2090, 0.1694, 0.1492, 0.3119, 0.1605],
                ],
                [
                    [0.1950, 0.1239, 0.2854, 0.2325, 0.1632],
                    [0.2324, 0.1133, 0.1760, 0.2818, 0.1965],
                    [0.1906, 0.1211, 0.2027, 0.2170, 0.2687],
                    [0.2018, 0.1164, 0.2073, 0.2418, 0.2327],
                    [0.2354, 0.0762, 0.2061, 0.2774, 0.2050],
                    [0.1968, 0.0876, 0.2437, 0.3027, 0.1693],
                    [0.2154, 0.0789, 0.2183, 0.3195, 0.1680],
                    [0.2011, 0.0958, 0.2537, 0.2560, 0.1934],
                    [0.1979, 0.1001, 0.2898, 0.2209, 0.1913],
                    [0.2338, 0.0861, 0.2225, 0.3663, 0.0913],
                    [0.2280, 0.0760, 0.2654, 0.2864, 0.1441],
                    [0.2413, 0.0705, 0.2240, 0.2984, 0.1658],
                ],
            ]
        ),
    )


def test_configure_optimizers(model):
    model.trainer = Trainer(max_epochs=10)
    optimizer_and_schedular = model.configure_optimizers()
    assert optimizer_and_schedular is not None
    optimizers, schedulers = optimizer_and_schedular

    assert len(optimizers) == 1
    optimizer = optimizers[0]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-05
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["eps"] == 1e-08

    assert len(schedulers) == 1
    scheduler = schedulers[0]
    assert isinstance(scheduler["scheduler"], torch.optim.lr_scheduler.LambdaLR)


def test_configure_optimizers_with_task_learning_rate():
    model = TokenClassificationModelWithSeq2SeqEncoderAndCrf(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=5,
        warmup_proportion=0.0,
        task_learning_rate=1e-4,
    )
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    # check that all parameters are in the optimizer
    assert set(optimizer.param_groups[0]["params"]) | set(
        optimizer.param_groups[1]["params"]
    ) == set(model.parameters())

    # base model parameters
    param_group = optimizer.param_groups[0]
    assert param_group["lr"] == 1e-05
    assert len(param_group["params"]) == 39

    # task parameters
    param_group = optimizer.param_groups[1]
    assert param_group["lr"] == 1e-04
    assert len(param_group["params"]) == 5
