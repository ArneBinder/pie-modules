import logging
from dataclasses import dataclass

import pytest
import torch
from pytorch_ie import AnnotationList, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextBasedDocument
from pytorch_lightning import Trainer
from torch.optim import AdamW

from pie_modules.models import SimpleGenerativeModel
from pie_modules.models.base_models import BartAsPointerNetwork
from pie_modules.taskmodules import PointerNetworkTaskModuleForEnd2EndRE
from tests import _config_to_str

# just the default config for now
CONFIGS = [{}, {"decoder_position_id_pattern": [0, 0, 1, 0, 0, 1, 1]}]
CONFIG_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}

TASKMODULE_CONFIGS = [{}, {"create_constraints": True}, {"constrained_generation": True}]
TASKMODULE_CONFIG_DICT = {_config_to_str(cfg): cfg for cfg in TASKMODULE_CONFIGS}

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", params=CONFIG_DICT.keys())
def config_str(request):
    return request.param


@pytest.fixture(scope="module")
def config(config_str):
    return CONFIG_DICT[config_str]


@pytest.fixture(scope="module", params=TASKMODULE_CONFIG_DICT.keys())
def taskmodule_config_str(request):
    return request.param


@pytest.fixture(scope="module")
def taskmodule_config(taskmodule_config_str):
    return TASKMODULE_CONFIG_DICT[taskmodule_config_str]


@pytest.fixture(scope="module")
def document():
    @dataclass
    class ExampleDocument(TextBasedDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
        sentences: AnnotationList[LabeledSpan] = annotation_field(target="text")

    doc = ExampleDocument(text="This is a dummy text about nothing. Trust me.")
    span1 = LabeledSpan(start=10, end=20, label="content")
    span2 = LabeledSpan(start=27, end=34, label="topic")
    span3 = LabeledSpan(start=42, end=44, label="person")
    doc.entities.extend([span1, span2, span3])
    assert str(span1) == "dummy text"
    assert str(span2) == "nothing"
    assert str(span3) == "me"
    rel = BinaryRelation(head=span1, tail=span2, label="is_about")
    doc.relations.append(rel)
    assert str(rel.label) == "is_about"
    assert str(rel.head) == "dummy text"
    assert str(rel.tail) == "nothing"

    no_rel = BinaryRelation(head=span1, tail=span3, label="no_relation")
    doc.relations.append(no_rel)
    assert str(no_rel.label) == "no_relation"
    assert str(no_rel.head) == "dummy text"
    assert str(no_rel.tail) == "me"

    sent1 = LabeledSpan(start=0, end=35, label="1")
    sent2 = LabeledSpan(start=36, end=45, label="2")
    doc.sentences.extend([sent1, sent2])
    assert str(sent1) == "This is a dummy text about nothing."
    assert str(sent2) == "Trust me."
    return doc


@pytest.fixture(scope="module")
def taskmodule(document, taskmodule_config):
    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        tokenizer_name_or_path="facebook/bart-base",
        span_layer_name="entities",
        relation_layer_name="relations",
        exclude_labels_per_layer={"relations": ["no_relation"]},
        annotation_field_mapping={
            "entities": "labeled_spans",
            "relations": "binary_relations",
        },
        partition_layer_name="sentences",
        # disable strict_span_conversion, this effects only the no_relation annotation
        tokenizer_kwargs={"strict_span_conversion": False},
        **taskmodule_config,
    )

    taskmodule.prepare(documents=[document])

    return taskmodule


def test_taskmodule(taskmodule):
    assert taskmodule.is_prepared


@pytest.fixture(scope="module")
def model(taskmodule, config) -> SimpleGenerativeModel:
    torch.manual_seed(42)
    model = SimpleGenerativeModel(
        base_model_type=BartAsPointerNetwork,
        base_model_config=dict(
            pretrained_model_name_or_path="sshleifer/distilbart-xsum-12-1",
            bos_token_id=taskmodule.bos_id,
            eos_token_id=taskmodule.eos_id,
            pad_token_id=taskmodule.eos_id,
            label_ids=taskmodule.label_ids,
            target_token_ids=taskmodule.target_token_ids,
            embedding_weight_mapping=taskmodule.label_embedding_weight_mapping,
            max_length=512,
            num_beams=4,
            **config,
        ),
        taskmodule_config=taskmodule._config(),
    )
    # set model to training mode, otherwise model.encoder.bart_encoder.training will be False!
    model.train()
    return model


def test_model(model):
    assert model is not None


# not used
@pytest.fixture(scope="module")
def batch(taskmodule, document):
    task_encodings = taskmodule.encode(documents=[document], encode_target=True)
    batch = taskmodule.collate(task_encodings)
    return batch


def test_batch(batch, config, taskmodule):
    assert batch is not None
    inputs, targets = batch
    assert inputs is not None
    assert set(inputs) == {"input_ids", "attention_mask"}
    torch.testing.assert_close(
        inputs["input_ids"],
        torch.tensor(
            [[0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 2], [0, 18823, 162, 4, 2, 1, 1, 1, 1, 1]]
        ),
    )
    torch.testing.assert_close(
        inputs["attention_mask"],
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]),
    )

    assert targets is not None
    if taskmodule.create_constraints:
        assert set(targets) == {"constraints", "labels", "decoder_attention_mask"}
    else:
        assert set(targets) == {"labels", "decoder_attention_mask"}
    torch.testing.assert_close(
        targets["labels"],
        torch.tensor([[14, 14, 5, 11, 12, 3, 6, 1], [9, 9, 4, 2, 2, 2, 2, 1]]),
    )
    torch.testing.assert_close(
        targets["decoder_attention_mask"],
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]),
    )

    if taskmodule.create_constraints:
        torch.testing.assert_close(
            targets["constraints"],
            torch.tensor(
                [
                    [
                        [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    [
                        [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, -1, -1, -1, -1, -1],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1],
                    ],
                ]
            ),
        )


def test_forward_without_labels(model, batch):
    inputs, targets = batch
    with pytest.raises(ValueError) as excinfo:
        model(inputs)
    assert str(excinfo.value) == "decoder_input_ids has to be set!"


def test_training_step(model, batch, config, taskmodule):
    torch.manual_seed(42)
    assert model.training
    loss = model.training_step(batch, 0)
    if taskmodule.create_constraints:
        if config == {}:
            torch.testing.assert_close(loss, torch.tensor(6.148426055908203))
        elif config == {"decoder_position_id_pattern": [0, 0, 1, 0, 0, 1, 1]}:
            torch.testing.assert_close(loss, torch.tensor(6.367420196533203))
        else:
            raise ValueError(f"Unknown config: {config}")
    else:
        if config == {}:
            torch.testing.assert_close(loss, torch.tensor(3.702044725418091))
        elif config == {"decoder_position_id_pattern": [0, 0, 1, 0, 0, 1, 1]}:
            torch.testing.assert_close(loss, torch.tensor(3.945438861846924))
        else:
            raise ValueError(f"Unknown config: {config}")


def test_validation_step(model, batch, config, taskmodule):
    torch.manual_seed(42)
    model.eval()
    assert not model.training
    loss = model.validation_step(batch, 0)
    if taskmodule.create_constraints:
        if config == {}:
            torch.testing.assert_close(loss, torch.tensor(6.278500556945801))
        elif config == {"decoder_position_id_pattern": [0, 0, 1, 0, 0, 1, 1]}:
            torch.testing.assert_close(loss, torch.tensor(6.575843334197998))
        else:
            raise ValueError(f"Unknown config: {config}")
    else:
        if config == {}:
            torch.testing.assert_close(loss, torch.tensor(3.883049488067627))
        elif config == {"decoder_position_id_pattern": [0, 0, 1, 0, 0, 1, 1]}:
            torch.testing.assert_close(loss, torch.tensor(4.204827308654785))
        else:
            raise ValueError(f"Unknown config: {config}")


def test_test_step(model, batch, config, taskmodule):
    decoder_position_id_pattern = model.model.config.decoder_position_id_pattern
    torch.manual_seed(42)
    model.eval()
    assert not model.training
    model.metrics["test"].reset()
    loss = model.test_step(batch, 0)
    values = model.metrics["test"].compute()

    if taskmodule.create_constraints:
        if config == {}:
            torch.testing.assert_close(loss, torch.tensor(6.278500556945801))
        elif config == {"decoder_position_id_pattern": [0, 0, 1, 0, 0, 1, 1]}:
            torch.testing.assert_close(loss, torch.tensor(6.575843334197998))
        else:
            raise ValueError(f"Unknown config: {config}")
    else:
        if config == {}:
            torch.testing.assert_close(loss, torch.tensor(3.883049488067627))
        elif config == {"decoder_position_id_pattern": [0, 0, 1, 0, 0, 1, 1]}:
            torch.testing.assert_close(loss, torch.tensor(4.204827308654785))
        else:
            raise ValueError(f"Unknown config: {config}")

    if taskmodule.constrained_generation and decoder_position_id_pattern == [0, 0, 1, 0, 0, 1, 1]:
        assert values == {
            "exact_encoding_matches": 0.0,
            "decoding_errors": {"correct": 1.0, "all": 0.0},
            "entities": {
                "person": {"recall": 100.0, "precision": 33.3333, "f1": 50.0},
                "content": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
                "topic": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
                "micro": {"recall": 33.3333, "precision": 14.2857, "f1": 20.0},
            },
            "relations": {
                "is_about": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
                "micro": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            },
        }
    elif taskmodule.constrained_generation and decoder_position_id_pattern is None:
        assert values == {
            "exact_encoding_matches": 0.0,
            "decoding_errors": {"correct": 1.0, "all": 0.0},
            "entities": {
                "person": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
                "content": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
                "topic": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
                "micro": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            },
            "relations": {
                "is_about": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
                "micro": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            },
        }
    else:
        assert values == {
            "exact_encoding_matches": 0.0,
            "decoding_errors": {"all": 0.0},
            "entities": {
                "person": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
                "content": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
                "topic": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
                "micro": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            },
            "relations": {
                "is_about": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
                "micro": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            },
        }


def test_test_step_without_use_prediction_for_metrics(taskmodule, batch):
    torch.manual_seed(42)
    model = SimpleGenerativeModel(
        base_model_type=BartAsPointerNetwork,
        base_model_config=dict(
            pretrained_model_name_or_path="sshleifer/distilbart-xsum-12-1",
            bos_token_id=taskmodule.bos_id,
            eos_token_id=taskmodule.eos_id,
            pad_token_id=taskmodule.eos_id,
            label_ids=taskmodule.label_ids,
            target_token_ids=taskmodule.target_token_ids,
            embedding_weight_mapping=taskmodule.label_embedding_weight_mapping,
            max_length=512,
            num_beams=4,
        ),
        taskmodule_config=taskmodule._config(),
        warmup_proportion=0.1,
        use_prediction_for_metrics=False,
    )
    torch.manual_seed(42)
    model.eval()
    assert not model.training
    model.metrics["test"].reset()
    loss = model.test_step(batch, 0)
    if taskmodule.create_constraints:
        torch.testing.assert_close(loss, torch.tensor(6.278500556945801))
    else:
        torch.testing.assert_close(loss, torch.tensor(3.883049488067627))
    values = model.metrics["test"].compute()
    assert values == {
        "exact_encoding_matches": 0.0,
        "decoding_errors": {"all": 0.0},
        "entities": {
            "person": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            "content": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            "topic": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            "micro": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        },
        "relations": {
            "is_about": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
            "micro": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
        },
    }


def test_predict_step(model, batch, config, taskmodule):
    torch.manual_seed(42)
    output = model.predict_step(batch, 0)
    assert output is not None
    if taskmodule.constrained_generation:
        if config == {}:
            torch.testing.assert_close(
                output,
                torch.tensor(
                    [
                        [8, 9, 5, 10, 12, 5, 6, 10, 12, 5, 13, 13, 5, 6, 1],
                        [8, 9, 4, 10, 10, 3, 6, 9, 9, 3, 10, 10, 3, 6, 1],
                    ]
                ),
            )
        elif config == {"decoder_position_id_pattern": [0, 0, 1, 0, 0, 1, 1]}:
            torch.testing.assert_close(
                output,
                torch.tensor(
                    [
                        [8, 9, 5, 10, 12, 5, 6, 10, 12, 5, 13, 13, 5, 6, 1],
                        [8, 9, 4, 10, 10, 5, 6, 9, 9, 4, 8, 8, 4, 6, 1],
                    ]
                ),
            )
        else:
            raise ValueError(f"Unknown config: {config}")
    else:
        if config == {}:
            torch.testing.assert_close(
                output,
                torch.tensor(
                    [
                        [8, 9, 10, 12, 13, 10, 12, 12, 13, 10, 1],
                        [8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 1],
                    ]
                ),
            )
        elif config == {"decoder_position_id_pattern": [0, 0, 1, 0, 0, 1, 1]}:
            torch.testing.assert_close(
                output,
                torch.tensor(
                    [[8, 9, 10, 12, 13, 10, 12, 12, 13, 10, 1], [8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 1]]
                ),
            )
        else:
            raise ValueError(f"Unknown config: {config}")


@pytest.fixture(scope="module")
def default_model(taskmodule, config) -> SimpleGenerativeModel:
    torch.manual_seed(42)
    model = SimpleGenerativeModel(
        base_model_type=BartAsPointerNetwork,
        base_model_config=dict(
            pretrained_model_name_or_path="sshleifer/distilbart-xsum-12-1",
            bos_token_id=taskmodule.bos_id,
            eos_token_id=taskmodule.eos_id,
            pad_token_id=taskmodule.eos_id,
            label_ids=taskmodule.label_ids,
            target_token_ids=taskmodule.target_token_ids,
            embedding_weight_mapping=taskmodule.label_embedding_weight_mapping,
            max_length=512,
            num_beams=4,
        ),
        taskmodule_config=taskmodule._config(),
    )
    # set model to training mode, otherwise model.encoder.bart_encoder.training will be False!
    model.train()
    return model


def test_on_train_epoch_end(default_model, config):
    torch.manual_seed(42)
    default_model.on_train_epoch_end()


def test_on_validation_epoch_end(default_model, config):
    torch.manual_seed(42)
    default_model.on_validation_epoch_end()


def test_on_test_epoch_end(default_model, config):
    torch.manual_seed(42)
    default_model.on_test_epoch_end()


def test_configure_optimizers(default_model, config):
    model = default_model
    optimizers = model.configure_optimizers()
    assert isinstance(optimizers, AdamW)
    assert len(optimizers.param_groups) == 6
    assert all(param_group["lr"] == 5e-05 for param_group in optimizers.param_groups)
    all_param_shapes = [
        [tuple(p.shape) for p in param_group["params"]] for param_group in optimizers.param_groups
    ]

    # check that all parameters are covered
    all_params = set(model.parameters())
    all_params_in_param_groups = set()
    for param_group in optimizers.param_groups:
        all_params_in_param_groups.update(param_group["params"])
    assert all_params_in_param_groups == all_params

    # head parameters
    assert optimizers.param_groups[0]["weight_decay"] == 0.01
    # per default, it is with encoder_mlp
    assert all_param_shapes[0] == [(1024, 1024), (1024,), (1024, 1024), (1024,)]

    # decoder layer norm only parameters
    assert optimizers.param_groups[1]["weight_decay"] == 0.01 == model.model.config.weight_decay
    assert len(all_param_shapes[1]) == 8

    # decoder only other parameters
    assert optimizers.param_groups[2]["weight_decay"] == 0.01 == model.model.config.weight_decay
    assert len(all_param_shapes[2]) == 21

    # layer norm encoder only parameters
    assert (
        optimizers.param_groups[3]["weight_decay"]
        == 0.001
        == model.model.config.encoder_layer_norm_decay
    )
    assert len(all_param_shapes[3]) == 50

    # remaining encoder only parameters
    assert optimizers.param_groups[4]["weight_decay"] == 0.01 == model.model.config.weight_decay
    assert len(all_param_shapes[4]) == 145

    # encoder-decoder shared parameters (embed_tokens.weight)
    assert optimizers.param_groups[5]["weight_decay"] == 0.01 == model.model.config.weight_decay
    assert len(all_param_shapes[5]) == 1


def test_configure_optimizers_with_warmup_proportion(default_model):
    model = default_model
    original_warmup_proportion = default_model.warmup_proportion
    default_model.warmup_proportion = 0.1

    model.trainer = Trainer(max_epochs=10)
    optimizers_and_schedulers = model.configure_optimizers()
    assert optimizers_and_schedulers is not None
    assert isinstance(optimizers_and_schedulers, tuple) and len(optimizers_and_schedulers) == 2

    optimizers, schedulers = optimizers_and_schedulers
    assert isinstance(optimizers[0], torch.optim.Optimizer)
    assert set(schedulers[0]) == {"scheduler", "interval"}
    schedular = schedulers[0]["scheduler"]
    assert isinstance(schedular, torch.optim.lr_scheduler.LRScheduler)

    # restore original value
    default_model.warmup_proportion = original_warmup_proportion
