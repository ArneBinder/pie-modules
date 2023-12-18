import logging
from dataclasses import dataclass

import pytest
import torch
from pytorch_ie import AnnotationList, Document, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextBasedDocument
from pytorch_lightning import Trainer
from torch.optim import AdamW

from pie_modules.models import SimplePointerNetworkModel
from pie_modules.taskmodules import PointerNetworkTaskModule
from tests import _config_to_str

# just the default config for now
CONFIGS = [{}, {"use_encoder_mlp": True}]
CONFIG_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", params=CONFIG_DICT.keys())
def config_str(request):
    return request.param


@pytest.fixture(scope="module")
def config(config_str):
    return CONFIG_DICT[config_str]


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
def taskmodule(document):
    taskmodule = PointerNetworkTaskModule(
        annotation_encoder_decoder_kwargs={
            "span_layer_name": "entities",
            "relation_layer_name": "relations",
            "exclude_labels_per_layer": {"relations": ["no_relation"]},
        },
        annotation_field_mapping={
            "entities": "labeled_spans",
            "relations": "binary_relations",
        },
    )

    taskmodule.prepare(documents=[document])

    return taskmodule


def test_taskmodule(taskmodule):
    # check the annotation_encoder_decoder
    annotation_encoder_decoder = taskmodule.annotation_encoder_decoder
    assert annotation_encoder_decoder.is_prepared
    assert annotation_encoder_decoder.prepared_attributes == {
        "labels_per_layer": {
            "entities": ["content", "person", "topic"],
            "relations": ["is_about"],
        },
    }
    assert annotation_encoder_decoder.layer_names == ["entities", "relations"]
    assert annotation_encoder_decoder.special_targets == ["<s>", "</s>"]
    assert annotation_encoder_decoder.labels == ["none", "content", "person", "topic", "is_about"]
    assert annotation_encoder_decoder.targets == [
        "<s>",
        "</s>",
        "none",
        "content",
        "person",
        "topic",
        "is_about",
    ]
    assert annotation_encoder_decoder.bos_id == 0
    assert annotation_encoder_decoder.eos_id == 1
    assert annotation_encoder_decoder.none_id == 2
    assert annotation_encoder_decoder.span_ids == [3, 4, 5]
    assert annotation_encoder_decoder.relation_ids == [6]
    assert annotation_encoder_decoder.label2id == {
        "content": 3,
        "is_about": 6,
        "none": 2,
        "person": 4,
        "topic": 5,
    }

    # check taskmodule properties
    assert taskmodule.prepared_attributes == {
        "annotation_encoder_decoder_kwargs": {
            "span_layer_name": "entities",
            "relation_layer_name": "relations",
            "exclude_labels_per_layer": {"relations": ["no_relation"]},
            "bos_token": "<s>",
            "eos_token": "</s>",
            "labels_per_layer": {
                "entities": ["content", "person", "topic"],
                "relations": ["is_about"],
            },
        }
    }
    assert taskmodule.label_embedding_weight_mapping == {
        50265: [45260],
        50266: [39763],
        50267: [354, 1215, 9006],
        50268: [5970],
        50269: [10166],
    }
    assert taskmodule.target_tokens == [
        "<s>",
        "</s>",
        "<<none>>",
        "<<content>>",
        "<<person>>",
        "<<topic>>",
        "<<is_about>>",
    ]
    assert taskmodule.target_token_ids == [0, 2, 50266, 50269, 50268, 50265, 50267]


@pytest.fixture(scope="module")
def model(taskmodule, config) -> SimplePointerNetworkModel:
    torch.manual_seed(42)
    model = SimplePointerNetworkModel(
        model_name_or_path="sshleifer/distilbart-xsum-12-1",
        target_token_ids=taskmodule.target_token_ids,
        vocab_size=len(taskmodule.tokenizer),
        embedding_weight_mapping=taskmodule.label_embedding_weight_mapping,
        annotation_encoder_decoder_name=taskmodule.annotation_encoder_decoder_name,
        annotation_encoder_decoder_kwargs=taskmodule.annotation_encoder_decoder_kwargs,
        **config,
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


def test_batch(batch, config):
    assert batch is not None
    inputs, targets = batch
    assert inputs is not None
    assert set(inputs) == {"src_tokens", "src_seq_len", "src_attention_mask"}
    torch.testing.assert_close(
        inputs["src_tokens"],
        torch.tensor([[0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 3101, 162, 4, 2]]),
    )
    torch.testing.assert_close(
        inputs["src_attention_mask"],
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
    )
    torch.testing.assert_close(
        inputs["src_seq_len"],
        torch.tensor([13]),
    )

    assert targets is not None
    assert set(targets) == {"tgt_tokens", "tgt_attention_mask", "tgt_seq_len", "CPM_tag"}
    torch.testing.assert_close(
        targets["tgt_tokens"],
        torch.tensor([[0, 14, 14, 5, 11, 12, 3, 6, 17, 17, 4, 2, 2, 2, 2, 1]]),
    )
    torch.testing.assert_close(
        targets["tgt_attention_mask"],
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
    )
    torch.testing.assert_close(
        targets["tgt_seq_len"],
        torch.tensor([16]),
    )


def test_forward_without_labels(model, batch):
    inputs, targets = batch
    with pytest.raises(ValueError) as excinfo:
        model(inputs)
    assert str(excinfo.value) == "decoder_input_ids has to be set!"


def test_training_step(model, batch, config):
    torch.manual_seed(42)
    assert model.training
    loss = model.training_step(batch, 0)
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(5.422972202301025))
    elif config == {"use_encoder_mlp": True}:
        torch.testing.assert_close(loss, torch.tensor(4.601412773132324))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_validation_step(model, batch, config):
    torch.manual_seed(42)
    model.eval()
    assert not model.training
    loss = model.validation_step(batch, 0)
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(5.610896587371826))
    elif config == {"use_encoder_mlp": True}:
        torch.testing.assert_close(loss, torch.tensor(4.802009105682373))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_test_step(model, batch, config):
    torch.manual_seed(42)
    model.eval()
    assert not model.training
    loss = model.test_step(batch, 0)
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(5.610896587371826))
    elif config == {"use_encoder_mlp": True}:
        torch.testing.assert_close(loss, torch.tensor(4.802009105682373))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_configure_optimizers(model, config):
    optimizers = model.configure_optimizers()
    assert isinstance(optimizers, AdamW)
    assert len(optimizers.param_groups) == 5
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
    if config == {}:  # no encoder_mlp
        assert len(all_param_shapes[0]) == 0
    elif config == {"use_encoder_mlp": True}:
        assert all_param_shapes[0] == [(1024, 1024), (1024,), (1024, 1024), (1024,)]
    else:
        raise ValueError(f"Unknown config: {config}")

    # decoder parameters
    assert optimizers.param_groups[1]["weight_decay"] == 0.01
    assert len(all_param_shapes[1]) == 29

    # layer norm encoder only parameters
    assert optimizers.param_groups[2]["weight_decay"] == 0.001 == model.layernorm_decay
    assert len(all_param_shapes[2]) == 50

    # remaining encoder only parameters
    assert optimizers.param_groups[3]["weight_decay"] == 0.01
    assert len(all_param_shapes[3]) == 145

    # encoder-decoder shared parameters (embed_tokens.weight)
    assert optimizers.param_groups[4]["weight_decay"] == 0.01
    assert len(all_param_shapes[4]) == 1


def test_configure_optimizers_with_warmup_proportion(taskmodule, config):
    torch.manual_seed(42)
    model = SimplePointerNetworkModel(
        model_name_or_path="sshleifer/distilbart-xsum-12-1",
        target_token_ids=taskmodule.target_token_ids,
        vocab_size=len(taskmodule.tokenizer),
        embedding_weight_mapping=taskmodule.label_embedding_weight_mapping,
        annotation_encoder_decoder_name=taskmodule.annotation_encoder_decoder_name,
        annotation_encoder_decoder_kwargs=taskmodule.annotation_encoder_decoder_kwargs,
        warmup_proportion=0.1,
    )
    # set model to training mode, otherwise model.encoder.bart_encoder.training will be False!
    model.train()

    model.trainer = Trainer(max_epochs=10)
    optimizers_and_schedulars = model.configure_optimizers()
    assert optimizers_and_schedulars is not None
    assert isinstance(optimizers_and_schedulars, tuple) and len(optimizers_and_schedulars) == 2

    optimizers, schedulers = optimizers_and_schedulars
    assert isinstance(optimizers[0], torch.optim.Optimizer)
    assert set(schedulers[0]) == {"scheduler", "interval"}
    schedular = schedulers[0]["scheduler"]
    assert isinstance(schedular, torch.optim.lr_scheduler.LRScheduler)


# wandb run: https://wandb.ai/arne/dataset-sciarg-task-ner_re-training/runs/2xhakq93
MODEL_PATH = "/home/arbi01/projects/pie-document-level/models/dataset-sciarg/task-ner_re/2023-12-14_00-25-37"


@pytest.fixture(scope="module")
def trained_model() -> SimplePointerNetworkModel:
    model = SimplePointerNetworkModel.from_pretrained(MODEL_PATH)
    assert not model.training
    return model


@pytest.mark.slow
def test_trained_model(trained_model):
    assert trained_model is not None


@pytest.fixture(scope="module")
def loaded_taskmodule():
    taskmodule = PointerNetworkTaskModule.from_pretrained(MODEL_PATH)
    assert taskmodule.is_prepared
    return taskmodule


@pytest.mark.slow
def test_loaded_taskmodule(loaded_taskmodule):
    assert loaded_taskmodule is not None


@pytest.fixture(scope="module")
def sciarg_dataset(loaded_taskmodule):
    from pie_datasets import DatasetDict

    dataset = DatasetDict.load_dataset("pie/sciarg", name="merge_fragmented_spans")
    dataset_converted = dataset.to_document_type(loaded_taskmodule.document_type)
    return dataset_converted


@pytest.fixture(scope="module")
def sciarg_document(sciarg_dataset) -> Document:
    return sciarg_dataset["train"][0]


@pytest.mark.slow
def test_sciarg_document(sciarg_document):
    assert sciarg_document is not None


@pytest.fixture(scope="module")
def sciarg_batch(sciarg_document, loaded_taskmodule):
    task_encodings = loaded_taskmodule.encode([sciarg_document])
    batch = loaded_taskmodule.collate(task_encodings)
    return batch


@pytest.fixture(scope="module")
def sciarg_batch_predictions():
    return [
        [0, 0, 12, 5, 2, 2, 2, 5, 3, 3, 7, 3, 3, 3, 6, 20, 29, 5, 2, 2],
        [0, 0, 5, 87, 91, 5, 9, 87, 84, 5, 80, 318, 5, 6, 95, 1, 1, 1, 1, 1],
        [0, 0, 55, 3, 57, 58, 4, 9, 44, 55, 3, 60, 60, 4, 9, 63, 82, 3, 2, 2],
        [0, 0, 68, 5, 52, 54, 4, 9, 56, 68, 5, 71, 77, 5, 6, 71, 77, 3, 80, 2],
        [0, 0, 183, 5, 153, 158, 5, 7, 163, 183, 5, 185, 190, 5, 7, 195, 210, 5, 192, 2],
        [0, 0, 56, 5, 59, 73, 5, 9, 95, 108, 5, 5, 2, 2, 2, 162, 165, 5, 167, 2],
        [0, 0, 61, 3, 65, 65, 4, 9, 55, 61, 3, 68, 68, 4, 9, 82, 95, 5, 97, 2],
        [0, 0, 48, 5, 52, 85, 5, 6, 88, 98, 5, 100, 104, 4, 9, 106, 125, 5, 2, 2],
        [0, 0, 9, 3, 3, 3, 7, 3, 3, 9, 3, 10, 53, 3, 9, 50, 53, 3, 2, 2],
        [0, 0, 5, 374, 190, 4, 9, 952, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]


@pytest.mark.slow
def test_sciarg_batch_predictions(sciarg_batch_predictions, loaded_taskmodule):
    annotations, errors = loaded_taskmodule.annotation_encoder_decoder.decode(
        sciarg_batch_predictions[2]
    )
    assert set(annotations) == {"labeled_spans", "binary_relations"}
    assert len(annotations["labeled_spans"]) == 44
    assert len(annotations["binary_relations"]) == 21


@pytest.mark.slow
def test_sciarg_predict_step(trained_model, sciarg_batch, sciarg_batch_predictions):
    torch.manual_seed(42)
    prediction = trained_model.predict_step(sciarg_batch, 0)
    assert prediction is not None
    assert prediction["pred"].tolist() == sciarg_batch_predictions
