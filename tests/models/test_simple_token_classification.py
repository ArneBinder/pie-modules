import pytest
import torch

from pie_modules.models import SimpleTokenClassificationModel
from pie_modules.taskmodules import TokenClassificationTaskModule
from tests import _config_to_str

CONFIGS = [{}]
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
        "taskmodule_type": "TokenClassificationTaskModule",
        "tokenizer_name_or_path": "bert-base-cased",
        "span_annotation": "entities",
        "partition_annotation": None,
        "label_pad_id": -100,
        "labels": ["ORG", "PER"],
        "include_ill_formed_predictions": True,
        "tokenize_kwargs": {},
        "pad_kwargs": None,
    }


def test_taskmodule_config(documents, taskmodule_config):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = TokenClassificationTaskModule(
        span_annotation="entities",
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    taskmodule.prepare(documents)
    assert taskmodule.config == taskmodule_config


def test_batch(documents, batch, taskmodule_config):
    taskmodule = TokenClassificationTaskModule.from_config(taskmodule_config)
    encodings = taskmodule.encode(documents, encode_target=True)
    # just take the first 4 encodings
    batch_from_documents = taskmodule.collate(encodings[:4])

    inputs, targets = batch
    inputs_from_documents, targets_from_documents = batch_from_documents
    assert set(inputs) == set(inputs_from_documents)
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
    targets = torch.tensor(
        [
            [-100, 0, 0, 0, 0, -100, -100, -100, -100, -100, -100, -100],
            [-100, 3, 4, 4, 4, 0, 0, 1, 0, -100, -100, -100],
            [-100, 3, 4, 4, 4, 0, 1, 0, -100, -100, -100, -100],
            [-100, 0, 0, 0, 3, 4, 4, 4, 0, 1, 0, -100],
        ]
    )
    return inputs, targets


@pytest.fixture
def model(monkeypatch, batch, config, taskmodule_config) -> SimpleTokenClassificationModel:
    torch.manual_seed(42)
    model = SimpleTokenClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=5,
        taskmodule_config=taskmodule_config,
        metric_stages=["val", "test"],
    )
    return model


def test_forward(batch, model):
    inputs, targets = batch
    batch_size, seq_len = inputs["input_ids"].shape
    num_classes = model.config["num_classes"]

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
                    0.07148821651935577,
                    -0.2565232515335083,
                    -0.25278958678245544,
                    0.2990874648094177,
                    -0.03310558199882507,
                ],
                [
                    0.07846908271312714,
                    -0.17958512902259827,
                    -0.24599787592887878,
                    0.27696123719215393,
                    0.07664960622787476,
                ],
                [
                    0.13290363550186157,
                    -0.13444341719150543,
                    -0.07887572050094604,
                    0.317052960395813,
                    0.2816348373889923,
                ],
                [
                    0.08293254673480988,
                    -0.24898692965507507,
                    -0.08886243402957916,
                    0.10302959382534027,
                    0.31092968583106995,
                ],
                [
                    0.05878816172480583,
                    -0.2312331348657608,
                    -0.0873665064573288,
                    0.2766477167606354,
                    0.03432014584541321,
                ],
                [
                    -0.1116400808095932,
                    -0.26766031980514526,
                    -0.13703128695487976,
                    0.19948995113372803,
                    0.07068736851215363,
                ],
                [
                    -0.026067664846777916,
                    -0.18476778268814087,
                    -0.310282438993454,
                    0.3037613034248352,
                    -0.11853311210870743,
                ],
                [
                    -0.024998387321829796,
                    -0.2244686782360077,
                    -0.29393917322158813,
                    0.23167231678962708,
                    -0.15282289683818817,
                ],
                [
                    0.10277961194515228,
                    -0.2738752067089081,
                    -0.08948110044002533,
                    0.14838886260986328,
                    -0.08609344065189362,
                ],
                [
                    0.07655282318592072,
                    -0.3150433897972107,
                    -0.07492146641016006,
                    0.18046391010284424,
                    -0.11110746115446091,
                ],
                [
                    0.004683507606387138,
                    -0.30919894576072693,
                    -0.0968572124838829,
                    0.1935732066631317,
                    -0.18172965943813324,
                ],
                [
                    -0.042572494596242905,
                    -0.23680375516414642,
                    -0.14623208343982697,
                    0.1462315022945404,
                    -0.2074090838432312,
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
                    0.4033188819885254,
                    -2.8625898361206055,
                    -1.902637004852295,
                    2.6763601303100586,
                    -0.1165795847773552,
                ],
                [
                    -1.3821518421173096,
                    -3.8267252445220947,
                    -1.0482028722763062,
                    0.2934025824069977,
                    -3.01615047454834,
                ],
                [
                    -1.3901684284210205,
                    -4.28705358505249,
                    0.5455893278121948,
                    1.5018802881240845,
                    -3.5158305168151855,
                ],
                [
                    -0.3989315330982208,
                    -3.5138368606567383,
                    -0.02564099431037903,
                    2.212425470352173,
                    -1.5391907691955566,
                ],
            ]
        ),
    )


def test_training_step(batch, model, config):
    assert model.metric_train is None
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.676901936531067))


def test_validation_step(batch, model, config):
    model.metric_val.reset()
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.676901936531067))
    metric_value = model.metric_val.compute()
    assert metric_value == {"metric/MulticlassF1Score/val": torch.tensor(0.04210526496171951)}


def test_test_step(batch, model, config):
    model.metric_test.reset()
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.676901936531067))
    metric_value = model.metric_test.compute()
    assert metric_value == {"metric/MulticlassF1Score/test": torch.tensor(0.04210526496171951)}


@pytest.mark.parametrize("test_step", [False, True])
def test_predict_and_predict_step(model, batch, config, test_step):
    torch.manual_seed(42)
    if test_step:
        predictions = model.predict_step(batch, batch_idx=0, dataloader_idx=0)
    else:
        predictions = model.predict(batch[0])
    assert predictions.shape == batch[1].shape
    torch.testing.assert_close(
        predictions,
        torch.tensor(
            [
                [-100, 3, 3, 4, 3, -100, -100, -100, -100, -100, -100, -100],
                [-100, 3, 3, 2, 0, 2, 2, 2, 3, -100, -100, -100],
                [-100, 3, 2, 2, 3, 3, 2, 3, -100, -100, -100, -100],
                [-100, 3, 4, 3, 2, 3, 2, 3, 3, 2, 3, -100],
            ]
        ),
    )


def test_configure_optimizers(model):
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 1e-05
    assert len(optimizer.param_groups) == 1
    assert len(optimizer.param_groups[0]["params"]) > 0
    assert set(optimizer.param_groups[0]["params"]) == set(model.parameters())
