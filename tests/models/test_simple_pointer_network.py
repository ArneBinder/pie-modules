import json
import logging
import pickle
from collections import defaultdict
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
from tests import DUMP_FIXTURE_DATA, FIXTURES_ROOT, _config_to_str

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


@pytest.mark.skipif(not DUMP_FIXTURE_DATA, reason="don't dump fixture data")
def test_dump_sciarg_batch(loaded_taskmodule):
    from pie_datasets import DatasetDict

    dataset = DatasetDict.load_dataset("pie/sciarg", name="merge_fragmented_spans")
    dataset_converted = dataset.to_document_type(loaded_taskmodule.document_type)
    sciarg_document = dataset_converted["train"][0]

    task_encodings = loaded_taskmodule.encode([sciarg_document], encode_target=True)
    batch = loaded_taskmodule.collate(task_encodings)
    path = FIXTURES_ROOT / "models" / "pointer_network_model" / "sciarg_batch_with_targets.pkl"
    with open(path, "wb") as f:
        pickle.dump(batch, f)


@pytest.fixture(scope="module")
def sciarg_batch():
    with open(
        FIXTURES_ROOT / "models" / "pointer_network" / "sciarg_batch_with_targets.pkl",
        "rb",
    ) as f:
        batch = pickle.load(f)
    return batch


@pytest.fixture(scope="module")
def sciarg_batch_predictions():
    # TODO: why are there two leading zeros (bos_id)?
    with open(
        FIXTURES_ROOT / "models" / "simple_pointer_network" / "sciarg_batch_predictions.json"
    ) as f:
        data = json.load(f)
    return data


@pytest.mark.slow
def test_sciarg_batch_predictions(sciarg_batch_predictions, loaded_taskmodule):
    annotations, errors = loaded_taskmodule.annotation_encoder_decoder.decode(
        sciarg_batch_predictions[2]
    )
    assert set(annotations) == {"labeled_spans", "binary_relations"}
    assert len(annotations["labeled_spans"]) == 2
    assert len(annotations["binary_relations"]) == 1


@pytest.mark.slow
def test_sciarg_predict_step(trained_model, sciarg_batch, sciarg_batch_predictions):
    torch.manual_seed(42)
    prediction = trained_model.predict_step(sciarg_batch, 0)
    assert prediction is not None
    assert prediction["pred"].tolist() == sciarg_batch_predictions


@pytest.mark.slow
def test_sciarg_predict(trained_model, sciarg_batch, sciarg_batch_predictions, loaded_taskmodule):
    torch.manual_seed(42)
    inputs, targets = sciarg_batch
    generation_kwargs = {"num_beams": 5, "max_length": 512}
    prediction = trained_model.predict(inputs, **generation_kwargs)
    assert prediction is not None
    targets_list = targets["tgt_tokens"].tolist()
    prediction_list = prediction["pred"].tolist()

    tp = defaultdict(set)
    fp = defaultdict(set)
    fn = defaultdict(set)
    for i in range(len(targets_list)):
        current_targets = targets_list[i]
        current_predictions = prediction_list[i]
        annotations, errors = loaded_taskmodule.annotation_encoder_decoder.decode(
            current_predictions
        )
        (
            expected_annotations,
            expected_errors,
        ) = loaded_taskmodule.annotation_encoder_decoder.decode(current_targets)
        for layer_name in expected_annotations:
            tp[layer_name] |= set(annotations[layer_name]) & set(expected_annotations[layer_name])
            fp[layer_name] |= set(annotations[layer_name]) - set(expected_annotations[layer_name])
            fn[layer_name] |= set(expected_annotations[layer_name]) - set(annotations[layer_name])

    # check the numbers
    assert {layer_name: len(anns) for layer_name, anns in fp.items()} == {
        "labeled_spans": 108,
        "binary_relations": 65,
    }
    assert {layer_name: len(anns) for layer_name, anns in fn.items()} == {
        "labeled_spans": 106,
        "binary_relations": 78,
    }
    assert {layer_name: len(anns) for layer_name, anns in tp.items()} == {
        "labeled_spans": 29,
        "binary_relations": 4,
    }

    # check the actual annotations
    assert dict(fp) == {
        "labeled_spans": {
            LabeledSpan(start=742, end=747, label="own_claim", score=1.0),
            LabeledSpan(start=79, end=92, label="background_claim", score=1.0),
            LabeledSpan(start=161, end=188, label="own_claim", score=1.0),
            LabeledSpan(start=523, end=529, label="own_claim", score=1.0),
            LabeledSpan(start=388, end=389, label="data", score=1.0),
            LabeledSpan(start=607, end=617, label="background_claim", score=1.0),
            LabeledSpan(start=142, end=158, label="own_claim", score=1.0),
            LabeledSpan(start=45, end=52, label="background_claim", score=1.0),
            LabeledSpan(start=127, end=148, label="background_claim", score=1.0),
            LabeledSpan(start=230, end=239, label="background_claim", score=1.0),
            LabeledSpan(start=174, end=176, label="data", score=1.0),
            LabeledSpan(start=799, end=814, label="background_claim", score=1.0),
            LabeledSpan(start=87, end=97, label="data", score=1.0),
            LabeledSpan(start=378, end=398, label="own_claim", score=1.0),
            LabeledSpan(start=153, end=174, label="own_claim", score=1.0),
            LabeledSpan(start=106, end=167, label="background_claim", score=1.0),
            LabeledSpan(start=317, end=318, label="data", score=1.0),
            LabeledSpan(start=510, end=521, label="own_claim", score=1.0),
            LabeledSpan(start=-8, end=-7, label="own_claim", score=1.0),
            LabeledSpan(start=203, end=211, label="background_claim", score=1.0),
            LabeledSpan(start=966, end=995, label="data", score=1.0),
            LabeledSpan(start=429, end=441, label="background_claim", score=1.0),
            LabeledSpan(start=454, end=461, label="background_claim", score=1.0),
            LabeledSpan(start=214, end=220, label="background_claim", score=1.0),
            LabeledSpan(start=152, end=156, label="own_claim", score=1.0),
            LabeledSpan(start=204, end=261, label="own_claim", score=1.0),
            LabeledSpan(start=217, end=228, label="background_claim", score=1.0),
            LabeledSpan(start=737, end=738, label="data", score=1.0),
            LabeledSpan(start=543, end=554, label="data", score=1.0),
            LabeledSpan(start=227, end=235, label="own_claim", score=1.0),
            LabeledSpan(start=542, end=555, label="own_claim", score=1.0),
            LabeledSpan(start=177, end=180, label="data", score=1.0),
            LabeledSpan(start=233, end=253, label="background_claim", score=1.0),
            LabeledSpan(start=212, end=219, label="own_claim", score=1.0),
            LabeledSpan(start=474, end=521, label="own_claim", score=1.0),
            LabeledSpan(start=723, end=724, label="data", score=1.0),
            LabeledSpan(start=726, end=741, label="data", score=1.0),
            LabeledSpan(start=106, end=120, label="background_claim", score=1.0),
            LabeledSpan(start=627, end=634, label="background_claim", score=1.0),
            LabeledSpan(start=96, end=116, label="own_claim", score=1.0),
            LabeledSpan(start=443, end=453, label="background_claim", score=1.0),
            LabeledSpan(start=46, end=59, label="own_claim", score=1.0),
            LabeledSpan(start=463, end=464, label="data", score=1.0),
            LabeledSpan(start=61, end=68, label="own_claim", score=1.0),
            LabeledSpan(start=310, end=341, label="own_claim", score=1.0),
            LabeledSpan(start=746, end=756, label="own_claim", score=1.0),
            LabeledSpan(start=157, end=160, label="data", score=1.0),
            LabeledSpan(start=350, end=364, label="background_claim", score=1.0),
            LabeledSpan(start=85, end=99, label="own_claim", score=1.0),
            LabeledSpan(start=555, end=560, label="own_claim", score=1.0),
            LabeledSpan(start=340, end=353, label="own_claim", score=1.0),
            LabeledSpan(start=528, end=535, label="own_claim", score=1.0),
            LabeledSpan(start=791, end=794, label="own_claim", score=1.0),
            LabeledSpan(start=969, end=1006, label="own_claim", score=1.0),
            LabeledSpan(start=680, end=681, label="own_claim", score=1.0),
            LabeledSpan(start=224, end=228, label="background_claim", score=1.0),
            LabeledSpan(start=236, end=246, label="own_claim", score=1.0),
            LabeledSpan(start=696, end=720, label="own_claim", score=1.0),
            LabeledSpan(start=192, end=202, label="background_claim", score=1.0),
            LabeledSpan(start=112, end=117, label="data", score=1.0),
            LabeledSpan(start=499, end=500, label="data", score=1.0),
            LabeledSpan(start=348, end=371, label="own_claim", score=1.0),
            LabeledSpan(start=511, end=734, label="own_claim", score=1.0),
            LabeledSpan(start=136, end=148, label="background_claim", score=1.0),
            LabeledSpan(start=619, end=626, label="data", score=1.0),
            LabeledSpan(start=78, end=185, label="own_claim", score=1.0),
            LabeledSpan(start=175, end=181, label="own_claim", score=1.0),
            LabeledSpan(start=797, end=798, label="data", score=1.0),
            LabeledSpan(start=776, end=787, label="background_claim", score=1.0),
            LabeledSpan(start=705, end=718, label="background_claim", score=1.0),
            LabeledSpan(start=189, end=203, label="background_claim", score=1.0),
            LabeledSpan(start=367, end=368, label="data", score=1.0),
            LabeledSpan(start=223, end=225, label="data", score=1.0),
            LabeledSpan(start=340, end=341, label="data", score=1.0),
            LabeledSpan(start=719, end=734, label="own_claim", score=1.0),
            LabeledSpan(start=704, end=718, label="background_claim", score=1.0),
            LabeledSpan(start=70, end=78, label="data", score=1.0),
            LabeledSpan(start=169, end=170, label="data", score=1.0),
            LabeledSpan(start=158, end=179, label="background_claim", score=1.0),
            LabeledSpan(start=185, end=201, label="own_claim", score=1.0),
            LabeledSpan(start=434, end=437, label="data", score=1.0),
            LabeledSpan(start=502, end=503, label="data", score=1.0),
            LabeledSpan(start=324, end=325, label="data", score=1.0),
            LabeledSpan(start=58, end=59, label="data", score=1.0),
            LabeledSpan(start=168, end=173, label="background_claim", score=1.0),
            LabeledSpan(start=149, end=167, label="background_claim", score=1.0),
            LabeledSpan(start=182, end=183, label="data", score=1.0),
            LabeledSpan(start=298, end=317, label="own_claim", score=1.0),
            LabeledSpan(start=512, end=516, label="background_claim", score=1.0),
            LabeledSpan(start=454, end=498, label="background_claim", score=1.0),
            LabeledSpan(start=408, end=433, label="own_claim", score=1.0),
            LabeledSpan(start=331, end=337, label="data", score=1.0),
            LabeledSpan(start=171, end=173, label="data", score=1.0),
            LabeledSpan(start=72, end=86, label="own_claim", score=1.0),
            LabeledSpan(start=469, end=475, label="background_claim", score=1.0),
            LabeledSpan(start=236, end=237, label="data", score=1.0),
            LabeledSpan(start=164, end=166, label="data", score=1.0),
            LabeledSpan(start=507, end=510, label="own_claim", score=1.0),
            LabeledSpan(start=863, end=864, label="data", score=1.0),
            LabeledSpan(start=118, end=133, label="own_claim", score=1.0),
            LabeledSpan(start=262, end=276, label="data", score=1.0),
            LabeledSpan(start=258, end=270, label="background_claim", score=1.0),
            LabeledSpan(start=719, end=728, label="background_claim", score=1.0),
            LabeledSpan(start=694, end=703, label="data", score=1.0),
            LabeledSpan(start=247, end=258, label="data", score=1.0),
            LabeledSpan(start=310, end=913, label="own_claim", score=1.0),
            LabeledSpan(start=95, end=96, label="data", score=1.0),
            LabeledSpan(start=221, end=223, label="background_claim", score=1.0),
        },
        "binary_relations": {
            BinaryRelation(
                head=LabeledSpan(start=682, end=691, label="own_claim", score=1.0),
                tail=LabeledSpan(start=680, end=681, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=-8, end=-7, label="own_claim", score=1.0),
                tail=LabeledSpan(start=528, end=535, label="own_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=340, end=341, label="data", score=1.0),
                tail=LabeledSpan(start=512, end=516, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=463, end=464, label="data", score=1.0),
                tail=LabeledSpan(start=474, end=521, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=236, end=237, label="data", score=1.0),
                tail=LabeledSpan(start=340, end=353, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=636, end=664, label="background_claim", score=1.0),
                tail=LabeledSpan(start=627, end=634, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=203, end=211, label="background_claim", score=1.0),
                tail=LabeledSpan(start=221, end=223, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=342, end=349, label="background_claim", score=1.0),
                tail=LabeledSpan(start=350, end=364, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=61, end=68, label="own_claim", score=1.0),
                tail=LabeledSpan(start=46, end=59, label="own_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=324, end=325, label="data", score=1.0),
                tail=LabeledSpan(start=233, end=253, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=434, end=437, label="data", score=1.0),
                tail=LabeledSpan(start=408, end=433, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=331, end=337, label="data", score=1.0),
                tail=LabeledSpan(start=317, end=329, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=443, end=453, label="background_claim", score=1.0),
                tail=LabeledSpan(start=429, end=441, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=757, end=778, label="own_claim", score=1.0),
                tail=LabeledSpan(start=749, end=756, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=510, end=521, label="own_claim", score=1.0),
                tail=LabeledSpan(start=507, end=510, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=127, end=148, label="background_claim", score=1.0),
                tail=LabeledSpan(start=158, end=179, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=106, end=120, label="background_claim", score=1.0),
                tail=LabeledSpan(start=136, end=148, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=454, end=461, label="background_claim", score=1.0),
                tail=LabeledSpan(start=469, end=475, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=694, end=703, label="data", score=1.0),
                tail=LabeledSpan(start=704, end=718, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=164, end=166, label="data", score=1.0),
                tail=LabeledSpan(start=142, end=158, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=171, end=173, label="data", score=1.0),
                tail=LabeledSpan(start=106, end=167, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=262, end=276, label="data", score=1.0),
                tail=LabeledSpan(start=204, end=261, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=95, end=96, label="data", score=1.0),
                tail=LabeledSpan(start=79, end=92, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=157, end=160, label="data", score=1.0),
                tail=LabeledSpan(start=152, end=156, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=182, end=183, label="data", score=1.0),
                tail=LabeledSpan(start=185, end=201, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=212, end=213, label="data", score=1.0),
                tail=LabeledSpan(start=214, end=220, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=966, end=995, label="data", score=1.0),
                tail=LabeledSpan(start=310, end=913, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=863, end=864, label="data", score=1.0),
                tail=LabeledSpan(start=969, end=1006, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=174, end=176, label="data", score=1.0),
                tail=LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=70, end=78, label="data", score=1.0),
                tail=LabeledSpan(start=61, end=68, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=247, end=258, label="data", score=1.0),
                tail=LabeledSpan(start=236, end=246, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=543, end=554, label="data", score=1.0),
                tail=LabeledSpan(start=555, end=560, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=87, end=97, label="data", score=1.0),
                tail=LabeledSpan(start=72, end=86, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=523, end=529, label="own_claim", score=1.0),
                tail=LabeledSpan(start=530, end=542, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=189, end=203, label="background_claim", score=1.0),
                tail=LabeledSpan(start=78, end=185, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=207, end=209, label="data", score=1.0),
                tail=LabeledSpan(start=227, end=235, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=737, end=738, label="data", score=1.0),
                tail=LabeledSpan(start=719, end=734, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=378, end=398, label="own_claim", score=1.0),
                tail=LabeledSpan(start=542, end=555, label="own_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=168, end=173, label="background_claim", score=1.0),
                tail=LabeledSpan(start=142, end=158, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=619, end=626, label="data", score=1.0),
                tail=LabeledSpan(start=607, end=617, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=797, end=798, label="data", score=1.0),
                tail=LabeledSpan(start=791, end=794, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=726, end=741, label="data", score=1.0),
                tail=LabeledSpan(start=742, end=747, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=317, end=318, label="data", score=1.0),
                tail=LabeledSpan(start=230, end=247, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=302, end=309, label="data", score=1.0),
                tail=LabeledSpan(start=310, end=341, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=388, end=389, label="data", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=161, end=188, label="own_claim", score=1.0),
                tail=LabeledSpan(start=298, end=317, label="own_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=169, end=170, label="data", score=1.0),
                tail=LabeledSpan(start=149, end=167, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
                tail=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=217, end=228, label="background_claim", score=1.0),
                tail=LabeledSpan(start=204, end=213, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=499, end=500, label="data", score=1.0),
                tail=LabeledSpan(start=454, end=498, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=746, end=756, label="own_claim", score=1.0),
                tail=LabeledSpan(start=511, end=734, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=112, end=117, label="data", score=1.0),
                tail=LabeledSpan(start=139, end=160, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=367, end=368, label="data", score=1.0),
                tail=LabeledSpan(start=258, end=270, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=177, end=180, label="data", score=1.0),
                tail=LabeledSpan(start=192, end=202, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=719, end=728, label="background_claim", score=1.0),
                tail=LabeledSpan(start=705, end=718, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=230, end=239, label="background_claim", score=1.0),
                tail=LabeledSpan(start=224, end=228, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=723, end=724, label="data", score=1.0),
                tail=LabeledSpan(start=696, end=720, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=58, end=59, label="data", score=1.0),
                tail=LabeledSpan(start=45, end=52, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=343, end=347, label="data", score=1.0),
                tail=LabeledSpan(start=348, end=371, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=223, end=225, label="data", score=1.0),
                tail=LabeledSpan(start=212, end=219, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=118, end=133, label="own_claim", score=1.0),
                tail=LabeledSpan(start=97, end=111, label="own_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=776, end=787, label="background_claim", score=1.0),
                tail=LabeledSpan(start=799, end=814, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=502, end=503, label="data", score=1.0),
                tail=LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=175, end=181, label="own_claim", score=1.0),
                tail=LabeledSpan(start=153, end=174, label="own_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=134, end=141, label="own_claim", score=1.0),
                tail=LabeledSpan(start=96, end=116, label="own_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
        },
    }

    assert dict(fn) == {
        "labeled_spans": {
            LabeledSpan(start=626, end=633, label="own_claim", score=1.0),
            LabeledSpan(start=93, end=99, label="own_claim", score=1.0),
            LabeledSpan(start=726, end=747, label="data", score=1.0),
            LabeledSpan(start=54, end=57, label="data", score=1.0),
            LabeledSpan(start=109, end=113, label="data", score=1.0),
            LabeledSpan(start=441, end=455, label="background_claim", score=1.0),
            LabeledSpan(start=390, end=403, label="background_claim", score=1.0),
            LabeledSpan(start=838, end=862, label="background_claim", score=1.0),
            LabeledSpan(start=323, end=326, label="data", score=1.0),
            LabeledSpan(start=230, end=234, label="data", score=1.0),
            LabeledSpan(start=528, end=532, label="background_claim", score=1.0),
            LabeledSpan(start=815, end=831, label="background_claim", score=1.0),
            LabeledSpan(start=485, end=488, label="data", score=1.0),
            LabeledSpan(start=618, end=634, label="background_claim", score=1.0),
            LabeledSpan(start=310, end=322, label="background_claim", score=1.0),
            LabeledSpan(start=306, end=309, label="data", score=1.0),
            LabeledSpan(start=330, end=337, label="background_claim", score=1.0),
            LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
            LabeledSpan(start=736, end=739, label="data", score=1.0),
            LabeledSpan(start=572, end=584, label="background_claim", score=1.0),
            LabeledSpan(start=217, end=228, label="own_claim", score=1.0),
            LabeledSpan(start=501, end=504, label="data", score=1.0),
            LabeledSpan(start=49, end=64, label="own_claim", score=1.0),
            LabeledSpan(start=578, end=588, label="data", score=1.0),
            LabeledSpan(start=43, end=45, label="data", score=1.0),
            LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
            LabeledSpan(start=488, end=491, label="data", score=1.0),
            LabeledSpan(start=203, end=210, label="background_claim", score=1.0),
            LabeledSpan(start=662, end=674, label="own_claim", score=1.0),
            LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
            LabeledSpan(start=411, end=428, label="background_claim", score=1.0),
            LabeledSpan(start=642, end=660, label="own_claim", score=1.0),
            LabeledSpan(start=387, end=390, label="data", score=1.0),
            LabeledSpan(start=509, end=516, label="background_claim", score=1.0),
            LabeledSpan(start=1007, end=1010, label="data", score=1.0),
            LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
            LabeledSpan(start=30, end=41, label="background_claim", score=1.0),
            LabeledSpan(start=310, end=341, label="background_claim", score=1.0),
            LabeledSpan(start=53, end=73, label="background_claim", score=1.0),
            LabeledSpan(start=704, end=728, label="own_claim", score=1.0),
            LabeledSpan(start=60, end=70, label="background_claim", score=1.0),
            LabeledSpan(start=862, end=865, label="data", score=1.0),
            LabeledSpan(start=522, end=526, label="data", score=1.0),
            LabeledSpan(start=981, end=1006, label="background_claim", score=1.0),
            LabeledSpan(start=480, end=508, label="background_claim", score=1.0),
            LabeledSpan(start=787, end=814, label="background_claim", score=1.0),
            LabeledSpan(start=378, end=401, label="own_claim", score=1.0),
            LabeledSpan(start=90, end=95, label="data", score=1.0),
            LabeledSpan(start=74, end=87, label="background_claim", score=1.0),
            LabeledSpan(start=94, end=97, label="data", score=1.0),
            LabeledSpan(start=88, end=97, label="data", score=1.0),
            LabeledSpan(start=800, end=814, label="background_claim", score=1.0),
            LabeledSpan(start=235, end=239, label="data", score=1.0),
            LabeledSpan(start=696, end=720, label="background_claim", score=1.0),
            LabeledSpan(start=76, end=78, label="data", score=1.0),
            LabeledSpan(start=33, end=37, label="own_claim", score=1.0),
            LabeledSpan(start=181, end=184, label="data", score=1.0),
            LabeledSpan(start=401, end=413, label="data", score=1.0),
            LabeledSpan(start=454, end=475, label="background_claim", score=1.0),
            LabeledSpan(start=136, end=148, label="own_claim", score=1.0),
            LabeledSpan(start=182, end=202, label="background_claim", score=1.0),
            LabeledSpan(start=113, end=117, label="data", score=1.0),
            LabeledSpan(start=906, end=911, label="own_claim", score=1.0),
            LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
            LabeledSpan(start=274, end=291, label="background_claim", score=1.0),
            LabeledSpan(start=519, end=527, label="background_claim", score=1.0),
            LabeledSpan(start=408, end=423, label="background_claim", score=1.0),
            LabeledSpan(start=74, end=86, label="own_claim", score=1.0),
            LabeledSpan(start=88, end=92, label="background_claim", score=1.0),
            LabeledSpan(start=57, end=60, label="data", score=1.0),
            LabeledSpan(start=41, end=52, label="background_claim", score=1.0),
            LabeledSpan(start=722, end=725, label="data", score=1.0),
            LabeledSpan(start=128, end=133, label="data", score=1.0),
            LabeledSpan(start=498, end=501, label="data", score=1.0),
            LabeledSpan(start=589, end=606, label="background_claim", score=1.0),
            LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
            LabeledSpan(start=719, end=731, label="background_claim", score=1.0),
            LabeledSpan(start=42, end=76, label="own_claim", score=1.0),
            LabeledSpan(start=509, end=512, label="data", score=1.0),
            LabeledSpan(start=96, end=105, label="data", score=1.0),
            LabeledSpan(start=365, end=386, label="background_claim", score=1.0),
            LabeledSpan(start=180, end=227, label="own_claim", score=1.0),
            LabeledSpan(start=618, end=631, label="own_claim", score=1.0),
            LabeledSpan(start=427, end=439, label="background_claim", score=1.0),
            LabeledSpan(start=41, end=47, label="own_claim", score=1.0),
            LabeledSpan(start=118, end=138, label="own_claim", score=1.0),
            LabeledSpan(start=97, end=126, label="background_claim", score=1.0),
            LabeledSpan(start=23, end=40, label="own_claim", score=1.0),
            LabeledSpan(start=70, end=75, label="data", score=1.0),
            LabeledSpan(start=34, end=52, label="background_claim", score=1.0),
            LabeledSpan(start=518, end=521, label="data", score=1.0),
            LabeledSpan(start=210, end=211, label="data", score=1.0),
            LabeledSpan(start=262, end=270, label="own_claim", score=1.0),
            LabeledSpan(start=415, end=454, label="own_claim", score=1.0),
            LabeledSpan(start=122, end=127, label="data", score=1.0),
            LabeledSpan(start=366, end=369, label="data", score=1.0),
            LabeledSpan(start=605, end=615, label="data", score=1.0),
            LabeledSpan(start=143, end=174, label="background_claim", score=1.0),
            LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
            LabeledSpan(start=796, end=799, label="data", score=1.0),
            LabeledSpan(start=523, end=529, label="data", score=1.0),
            LabeledSpan(start=1012, end=1018, label="data", score=1.0),
            LabeledSpan(start=348, end=371, label="background_claim", score=1.0),
            LabeledSpan(start=761, end=775, label="background_claim", score=1.0),
            LabeledSpan(start=212, end=225, label="background_claim", score=1.0),
            LabeledSpan(start=687, end=691, label="own_claim", score=1.0),
        },
        "binary_relations": {
            BinaryRelation(
                head=LabeledSpan(start=230, end=234, label="data", score=1.0),
                tail=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=365, end=386, label="background_claim", score=1.0),
                tail=LabeledSpan(start=427, end=439, label="background_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=636, end=664, label="background_claim", score=1.0),
                tail=LabeledSpan(start=618, end=634, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=113, end=117, label="data", score=1.0),
                tail=LabeledSpan(start=118, end=138, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=88, end=97, label="data", score=1.0),
                tail=LabeledSpan(start=97, end=111, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
                tail=LabeledSpan(start=719, end=731, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=70, end=75, label="data", score=1.0),
                tail=LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=33, end=37, label="own_claim", score=1.0),
                tail=LabeledSpan(start=93, end=99, label="own_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=181, end=184, label="data", score=1.0),
                tail=LabeledSpan(start=143, end=174, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=518, end=521, label="data", score=1.0),
                tail=LabeledSpan(start=509, end=516, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=523, end=529, label="data", score=1.0),
                tail=LabeledSpan(start=530, end=542, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=605, end=615, label="data", score=1.0),
                tail=LabeledSpan(start=618, end=631, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=54, end=57, label="data", score=1.0),
                tail=LabeledSpan(start=41, end=52, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=441, end=455, label="background_claim", score=1.0),
                tail=LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=43, end=45, label="data", score=1.0),
                tail=LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=230, end=247, label="own_claim", score=1.0),
                tail=LabeledSpan(start=204, end=213, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=235, end=239, label="data", score=1.0),
                tail=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=618, end=631, label="own_claim", score=1.0),
                tail=LabeledSpan(start=682, end=691, label="own_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=302, end=309, label="data", score=1.0),
                tail=LabeledSpan(start=310, end=341, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=488, end=491, label="data", score=1.0),
                tail=LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=662, end=674, label="own_claim", score=1.0),
                tail=LabeledSpan(start=642, end=660, label="own_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=214, end=228, label="background_claim", score=1.0),
                tail=LabeledSpan(start=182, end=202, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=210, end=211, label="data", score=1.0),
                tail=LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=109, end=113, label="data", score=1.0),
                tail=LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=212, end=213, label="data", score=1.0),
                tail=LabeledSpan(start=203, end=210, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=352, end=365, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=722, end=725, label="data", score=1.0),
                tail=LabeledSpan(start=696, end=720, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=306, end=309, label="data", score=1.0),
                tail=LabeledSpan(start=310, end=322, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=501, end=504, label="data", score=1.0),
                tail=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=94, end=97, label="data", score=1.0),
                tail=LabeledSpan(start=97, end=126, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
                tail=LabeledSpan(start=480, end=508, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=122, end=127, label="data", score=1.0),
                tail=LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
                tail=LabeledSpan(start=136, end=148, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=726, end=747, label="data", score=1.0),
                tail=LabeledSpan(start=749, end=756, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=42, end=76, label="own_claim", score=1.0),
                tail=LabeledSpan(start=78, end=89, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=522, end=526, label="data", score=1.0),
                tail=LabeledSpan(start=509, end=516, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=488, end=491, label="data", score=1.0),
                tail=LabeledSpan(start=454, end=475, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                tail=LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=509, end=512, label="data", score=1.0),
                tail=LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=76, end=78, label="data", score=1.0),
                tail=LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=128, end=133, label="data", score=1.0),
                tail=LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=118, end=138, label="own_claim", score=1.0),
                tail=LabeledSpan(start=139, end=160, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=485, end=488, label="data", score=1.0),
                tail=LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=862, end=865, label="data", score=1.0),
                tail=LabeledSpan(start=838, end=862, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=330, end=337, label="background_claim", score=1.0),
                tail=LabeledSpan(start=342, end=349, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=796, end=799, label="data", score=1.0),
                tail=LabeledSpan(start=800, end=814, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=61, end=68, label="background_claim", score=1.0),
                tail=LabeledSpan(start=46, end=59, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=736, end=739, label="data", score=1.0),
                tail=LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=578, end=588, label="data", score=1.0),
                tail=LabeledSpan(start=589, end=606, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=49, end=64, label="own_claim", score=1.0),
                tail=LabeledSpan(start=41, end=47, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=96, end=105, label="data", score=1.0),
                tail=LabeledSpan(start=74, end=87, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=387, end=390, label="data", score=1.0),
                tail=LabeledSpan(start=390, end=403, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=292, end=298, label="background_claim", score=1.0),
                tail=LabeledSpan(start=274, end=291, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=509, end=516, label="background_claim", score=1.0),
                tail=LabeledSpan(start=528, end=532, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=88, end=92, label="background_claim", score=1.0),
                tail=LabeledSpan(start=97, end=126, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=427, end=439, label="background_claim", score=1.0),
                tail=LabeledSpan(start=461, end=478, label="background_claim", score=1.0),
                label="contradicts",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=310, end=322, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=626, end=633, label="own_claim", score=1.0),
                tail=LabeledSpan(start=687, end=691, label="own_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=60, end=70, label="background_claim", score=1.0),
                tail=LabeledSpan(start=74, end=86, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=217, end=228, label="own_claim", score=1.0),
                tail=LabeledSpan(start=204, end=213, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=90, end=95, label="data", score=1.0),
                tail=LabeledSpan(start=74, end=87, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=1007, end=1010, label="data", score=1.0),
                tail=LabeledSpan(start=1012, end=1018, label="data", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=498, end=501, label="data", score=1.0),
                tail=LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=343, end=347, label="data", score=1.0),
                tail=LabeledSpan(start=348, end=371, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=57, end=60, label="data", score=1.0),
                tail=LabeledSpan(start=41, end=52, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=210, end=211, label="data", score=1.0),
                tail=LabeledSpan(start=212, end=225, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=485, end=488, label="data", score=1.0),
                tail=LabeledSpan(start=454, end=475, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=262, end=270, label="own_claim", score=1.0),
                tail=LabeledSpan(start=255, end=261, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=366, end=369, label="data", score=1.0),
                tail=LabeledSpan(start=369, end=383, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=207, end=209, label="data", score=1.0),
                tail=LabeledSpan(start=227, end=235, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=747, end=756, label="background_claim", score=1.0),
                tail=LabeledSpan(start=761, end=775, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
                tail=LabeledSpan(start=519, end=527, label="background_claim", score=1.0),
                label="parts_of_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=401, end=413, label="data", score=1.0),
                tail=LabeledSpan(start=415, end=454, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=323, end=326, label="data", score=1.0),
                tail=LabeledSpan(start=326, end=350, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=93, end=99, label="own_claim", score=1.0),
                tail=LabeledSpan(start=906, end=911, label="own_claim", score=1.0),
                label="semantically_same",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=207, end=209, label="data", score=1.0),
                tail=LabeledSpan(start=212, end=225, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=390, end=403, label="background_claim", score=1.0),
                tail=LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
        },
    }

    assert dict(tp) == {
        "labeled_spans": {
            LabeledSpan(start=97, end=111, label="own_claim", score=1.0),
            LabeledSpan(start=139, end=160, label="own_claim", score=1.0),
            LabeledSpan(start=302, end=309, label="data", score=1.0),
            LabeledSpan(start=326, end=350, label="background_claim", score=1.0),
            LabeledSpan(start=343, end=347, label="data", score=1.0),
            LabeledSpan(start=249, end=254, label="data", score=1.0),
            LabeledSpan(start=230, end=247, label="own_claim", score=1.0),
            LabeledSpan(start=757, end=778, label="own_claim", score=1.0),
            LabeledSpan(start=505, end=509, label="background_claim", score=1.0),
            LabeledSpan(start=682, end=691, label="own_claim", score=1.0),
            LabeledSpan(start=90, end=95, label="data", score=1.0),
            LabeledSpan(start=749, end=756, label="own_claim", score=1.0),
            LabeledSpan(start=317, end=329, label="background_claim", score=1.0),
            LabeledSpan(start=292, end=298, label="background_claim", score=1.0),
            LabeledSpan(start=342, end=349, label="background_claim", score=1.0),
            LabeledSpan(start=302, end=312, label="background_claim", score=1.0),
            LabeledSpan(start=114, end=120, label="background_claim", score=1.0),
            LabeledSpan(start=636, end=664, label="background_claim", score=1.0),
            LabeledSpan(start=352, end=365, label="background_claim", score=1.0),
            LabeledSpan(start=476, end=485, label="background_claim", score=1.0),
            LabeledSpan(start=204, end=213, label="own_claim", score=1.0),
            LabeledSpan(start=134, end=141, label="own_claim", score=1.0),
            LabeledSpan(start=207, end=209, label="data", score=1.0),
            LabeledSpan(start=290, end=305, label="background_claim", score=1.0),
            LabeledSpan(start=530, end=542, label="own_claim", score=1.0),
            LabeledSpan(start=78, end=89, label="own_claim", score=1.0),
            LabeledSpan(start=491, end=498, label="background_claim", score=1.0),
            LabeledSpan(start=255, end=261, label="own_claim", score=1.0),
            LabeledSpan(start=212, end=213, label="data", score=1.0),
        },
        "binary_relations": {
            BinaryRelation(
                head=LabeledSpan(start=90, end=95, label="data", score=1.0),
                tail=LabeledSpan(start=78, end=89, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=302, end=312, label="background_claim", score=1.0),
                tail=LabeledSpan(start=292, end=298, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=326, end=350, label="background_claim", score=1.0),
                tail=LabeledSpan(start=352, end=365, label="background_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
            BinaryRelation(
                head=LabeledSpan(start=249, end=254, label="data", score=1.0),
                tail=LabeledSpan(start=255, end=261, label="own_claim", score=1.0),
                label="supports",
                score=1.0,
            ),
        },
    }
