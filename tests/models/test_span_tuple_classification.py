import pytest
import torch
from pytorch_lightning import Trainer
from torch import tensor

from pie_modules.models import SpanTupleClassificationModel
from pie_modules.models.common import TESTING, TRAINING, VALIDATION
from pie_modules.taskmodules import RESpanPairClassificationTaskModule
from tests import _config_to_str

CONFIGS = [{}]
CONFIG_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}
NUM_CLASSES = 5


@pytest.fixture(scope="module", params=CONFIG_DICT.keys())
def config_str(request):
    return request.param


@pytest.fixture(scope="module")
def config(config_str):
    return CONFIG_DICT[config_str]


@pytest.fixture
def taskmodule_config():
    return {
        "taskmodule_type": "RESpanPairClassificationTaskModule",
        "tokenizer_name_or_path": "bert-base-cased",
        "relation_annotation": "relations",
        "no_relation_label": "no_relation",
        "partition_annotation": None,
        "tokenize_kwargs": None,
        "create_candidate_relations": False,
        "create_candidate_relations_kwargs": None,
        "labels": ["org:founded_by", "per:employee_of", "per:founder"],
        "entity_labels": ["ORG", "PER"],
        "add_type_to_marker": True,
        "log_first_n_examples": 0,
        "collect_statistics": False,
    }


def test_taskmodule_config(documents, taskmodule_config):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RESpanPairClassificationTaskModule(
        relation_annotation="relations",
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    taskmodule.prepare(documents)
    assert taskmodule.config == taskmodule_config


def test_batch(documents, batch, taskmodule_config):
    taskmodule = RESpanPairClassificationTaskModule.from_config(taskmodule_config)
    encodings = taskmodule.encode(documents, encode_target=True, as_dataset=True)
    batch_from_documents = taskmodule.collate(encodings[:4])

    inputs, targets = batch
    inputs_from_documents, targets_from_documents = batch_from_documents
    assert set(inputs) == set(inputs_from_documents)
    for key in inputs:
        torch.testing.assert_close(inputs[key], inputs_from_documents[key])
    assert set(targets) == set(targets_from_documents)
    for key in targets:
        torch.testing.assert_close(targets[key], targets_from_documents[key])


@pytest.fixture
def batch():
    inputs = {
        "input_ids": tensor(
            [
                [
                    101,
                    28996,
                    13832,
                    3121,
                    2340,
                    138,
                    28998,
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
                    28996,
                    13832,
                    3121,
                    2340,
                    144,
                    28998,
                    1759,
                    1120,
                    28999,
                    145,
                    28997,
                    119,
                    1262,
                    1771,
                    28999,
                    146,
                    28997,
                    119,
                    102,
                    0,
                    0,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28996,
                    13832,
                    3121,
                    2340,
                    150,
                    28998,
                    1759,
                    1120,
                    28999,
                    151,
                    28997,
                    119,
                    1262,
                    28996,
                    1122,
                    28998,
                    1771,
                    28999,
                    152,
                    28997,
                    119,
                    102,
                ],
            ]
        ),
        "attention_mask": tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ),
        "span_start_indices": tensor([[1, 9, 0, 0], [4, 12, 18, 0], [4, 12, 17, 21]]),
        "span_end_indices": tensor([[7, 12, 0, 0], [10, 15, 21, 0], [10, 15, 20, 24]]),
        "tuple_indices": tensor(
            [[[0, 1], [-1, -1], [-1, -1]], [[0, 1], [0, 2], [2, 1]], [[0, 1], [2, 3], [3, 2]]]
        ),
        "tuple_indices_mask": tensor(
            [[True, False, False], [True, True, True], [True, True, True]]
        ),
    }
    targets = {"labels": tensor([[2, -100, -100], [2, 3, 1], [2, 3, 1]])}
    return inputs, targets


@pytest.fixture
def model(batch, config, taskmodule_config) -> SpanTupleClassificationModel:
    torch.manual_seed(42)
    model = SpanTupleClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        taskmodule_config=taskmodule_config,
        metric_stages=["val", "test"],
        **config,
    )
    return model


def test_model_pickleable(model):
    import pickle

    pickle.dumps(model)


def test_freeze_base_model():
    model = SpanTupleClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
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
    model = SpanTupleClassificationModel(
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


@pytest.mark.parametrize(
    "span_embedding_mode", ["start_and_end_token", "start_token", "end_token"]
)
@pytest.mark.parametrize(
    "tuple_embedding_mode", ["concat", "multiply2_and_concat", "index_0", "index_1"]
)
def test_forward_embeddings(batch, taskmodule_config, span_embedding_mode, tuple_embedding_mode):
    torch.manual_seed(42)
    simple_model = SpanTupleClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        # disable the tuple mlp to allow for checking the intermediate embeddings via the indices
        tuple_entry_hidden_dim=None,
        taskmodule_config=taskmodule_config,
        span_embedding_mode=span_embedding_mode,
        tuple_embedding_mode=tuple_embedding_mode,
    )

    inputs, targets = batch
    batch_size, seq_len = inputs["input_ids"].shape
    num_classes = int(torch.max(targets["labels"]) + 1)

    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    # return embeddings to check the logits
    output = simple_model.forward(inputs, return_embeddings=True)
    assert set(output) == {"logits", "last_hidden_state", "span_embeddings", "tuple_embeddings"}
    logits_flat = output["logits"]
    assert len(logits_flat.shape) == 2
    assert logits_flat.shape[-1] == NUM_CLASSES

    # check span_embeddings: they should be the entries of last_hidden_state at the
    #  span_start_indices and span_end_indices
    for batch_idx in range(batch_size):
        for j, (start, end) in enumerate(
            zip(inputs["span_start_indices"][batch_idx], inputs["span_end_indices"][batch_idx])
        ):
            current_expected_span_embedding_list = []
            if simple_model.span_embedding_mode == "start_and_end_token":
                current_expected_span_embedding_list.append(
                    output["last_hidden_state"][batch_idx, start]
                )
                current_expected_span_embedding_list.append(
                    output["last_hidden_state"][batch_idx, end]
                )
            elif simple_model.span_embedding_mode == "start_token":
                current_expected_span_embedding_list.append(
                    output["last_hidden_state"][batch_idx, start]
                )
            elif simple_model.span_embedding_mode == "end_token":
                current_expected_span_embedding_list.append(
                    output["last_hidden_state"][batch_idx, end]
                )
            else:
                raise ValueError(
                    f"Unknown span_embedding_mode: {simple_model.span_embedding_mode}"
                )
            expected_current_span_embedding = torch.concat(
                current_expected_span_embedding_list, dim=-1
            )
            current_span_embeddings = output["span_embeddings"][batch_idx, j]
            torch.testing.assert_close(current_span_embeddings, expected_current_span_embedding)

    # check tuple_embeddings: they should be the entries of span_embeddings at the tuple_indices
    tuple_idx = 0
    for batch_idx in range(batch_size):
        for indices, is_valid in zip(
            inputs["tuple_indices"][batch_idx], inputs["tuple_indices_mask"][batch_idx]
        ):
            if is_valid:
                current_expected_tuple_embedding_list = [
                    output["span_embeddings"][batch_idx, idx] for idx in indices
                ]
                if simple_model.tuple_embedding_mode == "concat":
                    expected_current_tuple_embedding = torch.concat(
                        current_expected_tuple_embedding_list, dim=-1
                    )
                elif simple_model.tuple_embedding_mode == "multiply2_and_concat":
                    expected_current_tuple_embedding = torch.cat(
                        [
                            current_expected_tuple_embedding_list[0]
                            * current_expected_tuple_embedding_list[1],
                            current_expected_tuple_embedding_list[0],
                            current_expected_tuple_embedding_list[1],
                        ],
                        dim=-1,
                    )
                elif simple_model.tuple_embedding_mode.startswith("index_"):
                    idx = int(simple_model.tuple_embedding_mode.split("_")[1])
                    expected_current_tuple_embedding = current_expected_tuple_embedding_list[idx]
                else:
                    raise ValueError(
                        f"Unknown tuple_embedding_mode: {simple_model.tuple_embedding_mode}"
                    )
                current_tuple_embedding = output["tuple_embeddings"][tuple_idx]
                torch.testing.assert_close(
                    current_tuple_embedding, expected_current_tuple_embedding
                )
                tuple_idx += 1


def test_forward_logits(batch, model):
    inputs, targets = batch

    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    # return embeddings to check the logits
    output = model.forward(inputs)
    assert set(output) == {"logits"}
    logits_flat = output["logits"]
    assert len(logits_flat.shape) == 2
    assert logits_flat.shape[-1] == NUM_CLASSES
    # check the actual logits
    torch.testing.assert_close(
        logits_flat,
        tensor(
            [
                [
                    0.04708037152886391,
                    -0.1854250431060791,
                    -0.1840103566646576,
                    0.3704131245613098,
                    0.18601544201374054,
                ],
                [
                    -0.03788023442029953,
                    -0.10608164966106415,
                    -0.02290359139442444,
                    0.28895634412765503,
                    0.17962007224559784,
                ],
                [
                    -0.059875741600990295,
                    -0.1754310429096222,
                    -0.10405908524990082,
                    0.35208940505981445,
                    0.19879946112632751,
                ],
                [
                    0.15815359354019165,
                    0.026701007038354874,
                    0.03577664494514465,
                    0.0583316795527935,
                    0.15422050654888153,
                ],
                [
                    -0.06102393940091133,
                    -0.08837515115737915,
                    0.01728247106075287,
                    0.31482765078544617,
                    0.11037502437829971,
                ],
                [
                    -0.09222549945116043,
                    -0.024930231273174286,
                    -0.04674120619893074,
                    0.2863592207431793,
                    0.11059696227312088,
                ],
                [
                    0.12158188223838806,
                    -0.1837768256664276,
                    -0.25936028361320496,
                    0.1423916220664978,
                    -0.002427928149700165,
                ],
            ]
        ),
    )


def test_step(batch, model, config):
    torch.manual_seed(42)
    loss = model._step("train", batch)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(1.624246597290039))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_training_step_and_on_epoch_end(batch, model, config):
    metric = model._get_metric(TRAINING, batch_idx=0)
    assert metric is None
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(1.624246597290039))
    else:
        raise ValueError(f"Unknown config: {config}")

    model.on_train_epoch_end()


def test_validation_step_and_on_epoch_end(batch, model, config):
    metric = model._get_metric(VALIDATION, batch_idx=0)
    metric.reset()
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    metric_values = {k: v.item() for k, v in metric.compute().items()}
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(1.624246597290039))
        assert metric_values == {
            "macro/f1": 0.125,
            "micro/f1": 0.2857142984867096,
            "no_relation/f1": 0.0,
            "org:founded_by/f1": 0.0,
            "per:employee_of/f1": 0.0,
            "per:founder/f1": 0.5,
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
        torch.testing.assert_close(loss, torch.tensor(1.624246597290039))
        assert metric_values == {
            "macro/f1": 0.125,
            "micro/f1": 0.2857142984867096,
            "no_relation/f1": 0.0,
            "org:founded_by/f1": 0.0,
            "per:employee_of/f1": 0.0,
            "per:founder/f1": 0.5,
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
        # TODO: adjust when model is fixed
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
    else:
        raise ValueError(f"Unknown config: {config}")

    assert labels.shape == batch[1]["labels"].shape
    # TODO: adjust when model is fixed
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
    model = SpanTupleClassificationModel(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
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
    assert len(param_group["params"]) == 6
