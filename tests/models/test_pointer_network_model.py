import logging
from dataclasses import dataclass

import pytest
import torch
from pytorch_ie import AnnotationList, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextBasedDocument
from torch.optim import AdamW

from pie_modules.models import PointerNetworkModel
from pie_modules.taskmodules import PointerNetworkTaskModule

# from src.models.components.gmam.metrics import LabeledAnnotationScore, AnnotationLayerMetric
from tests import _config_to_str

# from tests.taskmodules.test_gmam_taskmodule import FIXTURES_DIR as TASKMODULE_FIXTURE_DIR


CONFIGS = [
    {"use_encoder_mlp": True},
    {"biloss": False, "decode_mask": False, "replace_pos": False, "use_encoder_mlp": False},
    {"biloss": False, "decode_mask": False, "replace_pos": False, "use_encoder_mlp": True},
]
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
def model(taskmodule, config) -> PointerNetworkModel:
    torch.manual_seed(42)
    model = PointerNetworkModel(
        bart_model="sshleifer/distilbart-xsum-12-1",
        target_token_ids=taskmodule.target_token_ids,
        pad_token_id=taskmodule.tokenizer.pad_token_id,
        vocab_size=len(taskmodule.tokenizer),
        embedding_weight_mapping=taskmodule.label_embedding_weight_mapping,
        decoder_type="avg_score",
        copy_gate=False,
        use_recur_pos=False,
        # replace_pos=True,
        position_type=0,
        max_length=10,
        max_len_a=0.5,
        num_beams=4,
        do_sample=False,
        repetition_penalty=1,
        length_penalty=1.0,
        restricter=None,
        # decode_mask=True,
        # biloss=True,
        lr=5e-5,
        max_target_positions=512,
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
    assert set(targets) == {"tgt_tokens", "tgt_seq_len", "CPM_tag"}
    torch.testing.assert_close(
        targets["tgt_tokens"],
        torch.tensor([[0, 14, 14, 5, 11, 12, 3, 6, 17, 17, 4, 2, 2, 2, 2, 1]]),
    )
    torch.testing.assert_close(
        targets["tgt_seq_len"],
        torch.tensor([16]),
    )
    torch.testing.assert_close(
        targets["CPM_tag"],
        torch.tensor(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
        ),
    )


def test_training_step(model, batch, config):
    torch.manual_seed(42)
    assert model.training
    loss = model.training_step(batch, 0)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.dim() == 0
    if config == {"use_encoder_mlp": True}:
        torch.testing.assert_close(loss, torch.tensor(5.364923000335693))
    elif config == {
        "biloss": False,
        "decode_mask": False,
        "replace_pos": False,
        "use_encoder_mlp": False,
    }:
        torch.testing.assert_close(loss, torch.tensor(5.140683650970459))
    elif config == {
        "biloss": False,
        "decode_mask": False,
        "replace_pos": False,
        "use_encoder_mlp": True,
    }:
        torch.testing.assert_close(loss, torch.tensor(4.4815826416015625))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_validation_step(model, batch, config):
    torch.manual_seed(42)
    model.eval()
    assert not model.training
    loss = model.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.dim() == 0
    if config == {"use_encoder_mlp": True}:
        torch.testing.assert_close(loss, torch.tensor(5.8239617347717285))
    elif config == {
        "biloss": False,
        "decode_mask": False,
        "replace_pos": False,
        "use_encoder_mlp": False,
    }:
        torch.testing.assert_close(loss, torch.tensor(5.610896587371826))
    elif config == {
        "biloss": False,
        "decode_mask": False,
        "replace_pos": False,
        "use_encoder_mlp": True,
    }:
        torch.testing.assert_close(loss, torch.tensor(4.803609371185303))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_predict_step(model, batch):
    torch.manual_seed(42)
    prediction = model.predict_step(batch, 0)
    assert prediction is not None


def test_metric_val(model, batch):
    metric = model.metrics["val"]
    metric.reset()
    inputs, targets = batch
    prediction = model.predict(inputs)
    metric(prediction["pred"], targets["tgt_tokens"])

    values = metric.get_metric(reset=True)
    assert values is not None
    assert values == {
        "em": 0.0,
        "entities": {
            "person": {"acc": 0, "recall": 0.0, "f1": 0.0},
            "topic": {"acc": 0, "recall": 0.0, "f1": 0.0},
            "content": {"acc": 0, "recall": 0.0, "f1": 0.0},
        },
        "entities/micro": {"acc": 0, "recall": 0.0, "f1": 0.0},
        "relations": {"is_about": {"acc": 0, "recall": 0.0, "f1": 0.0}},
        "relations/micro": {"acc": 0, "recall": 0.0, "f1": 0.0},
        "invalid/len": 0.0,
        "invalid/order": 0.0,
        "invalid/cross": 0.0,
        "invalid/cover": 0.0,
        "invalid/all": 0.0,
    }


def test_head_parameters(model, config):
    parameter_shapes = {name: param.shape for name, param in model.head_parameters.items()}

    # TODO: should the parameters of the "decoder.decoder" be included here?
    if config == {"use_encoder_mlp": True}:
        assert parameter_shapes == {
            "decoder.decoder.embed_positions.weight": torch.Size([1026, 1024]),
            "decoder.decoder.layers.0.self_attn.k_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.self_attn.k_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn.v_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.self_attn.v_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn.q_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.self_attn.q_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn.out_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.self_attn.out_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn_layer_norm.weight": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn_layer_norm.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn.k_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.encoder_attn.k_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn.v_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.encoder_attn.v_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn.q_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.encoder_attn.q_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn.out_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.encoder_attn.out_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn_layer_norm.weight": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn_layer_norm.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.fc1.weight": torch.Size([4096, 1024]),
            "decoder.decoder.layers.0.fc1.bias": torch.Size([4096]),
            "decoder.decoder.layers.0.fc2.weight": torch.Size([1024, 4096]),
            "decoder.decoder.layers.0.fc2.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.final_layer_norm.weight": torch.Size([1024]),
            "decoder.decoder.layers.0.final_layer_norm.bias": torch.Size([1024]),
            "decoder.decoder.layernorm_embedding.weight": torch.Size([1024]),
            "decoder.decoder.layernorm_embedding.bias": torch.Size([1024]),
            "decoder.bi_encoder_mlp.0.weight": torch.Size([1024, 1024]),
            "decoder.bi_encoder_mlp.0.bias": torch.Size([1024]),
            "decoder.bi_encoder_mlp.3.weight": torch.Size([1024, 1024]),
            "decoder.bi_encoder_mlp.3.bias": torch.Size([1024]),
            "decoder.encoder_mlp.0.weight": torch.Size([1024, 1024]),
            "decoder.encoder_mlp.0.bias": torch.Size([1024]),
            "decoder.encoder_mlp.3.weight": torch.Size([1024, 1024]),
            "decoder.encoder_mlp.3.bias": torch.Size([1024]),
        }
    elif config == {
        "biloss": False,
        "decode_mask": False,
        "replace_pos": False,
        "use_encoder_mlp": False,
    }:
        assert parameter_shapes == {
            "decoder.decoder.embed_positions.weight": torch.Size([1026, 1024]),
            "decoder.decoder.embed_positions_replace.weight": torch.Size([1026, 1024]),
            "decoder.decoder.layers.0.self_attn.k_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.self_attn.k_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn.v_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.self_attn.v_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn.q_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.self_attn.q_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn.out_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.self_attn.out_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn_layer_norm.weight": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn_layer_norm.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn.k_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.encoder_attn.k_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn.v_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.encoder_attn.v_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn.q_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.encoder_attn.q_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn.out_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.encoder_attn.out_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn_layer_norm.weight": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn_layer_norm.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.fc1.weight": torch.Size([4096, 1024]),
            "decoder.decoder.layers.0.fc1.bias": torch.Size([4096]),
            "decoder.decoder.layers.0.fc2.weight": torch.Size([1024, 4096]),
            "decoder.decoder.layers.0.fc2.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.final_layer_norm.weight": torch.Size([1024]),
            "decoder.decoder.layers.0.final_layer_norm.bias": torch.Size([1024]),
            "decoder.decoder.layernorm_embedding.weight": torch.Size([1024]),
            "decoder.decoder.layernorm_embedding.bias": torch.Size([1024]),
            "decoder.bi_encoder_mlp.0.weight": torch.Size([1024, 1024]),
            "decoder.bi_encoder_mlp.0.bias": torch.Size([1024]),
            "decoder.bi_encoder_mlp.3.weight": torch.Size([1024, 1024]),
            "decoder.bi_encoder_mlp.3.bias": torch.Size([1024]),
        }
    elif config == {
        "biloss": False,
        "decode_mask": False,
        "replace_pos": False,
        "use_encoder_mlp": True,
    }:
        assert parameter_shapes == {
            "decoder.decoder.embed_positions.weight": torch.Size([1026, 1024]),
            "decoder.decoder.embed_positions_replace.weight": torch.Size([1026, 1024]),
            "decoder.decoder.layers.0.self_attn.k_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.self_attn.k_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn.v_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.self_attn.v_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn.q_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.self_attn.q_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn.out_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.self_attn.out_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn_layer_norm.weight": torch.Size([1024]),
            "decoder.decoder.layers.0.self_attn_layer_norm.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn.k_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.encoder_attn.k_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn.v_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.encoder_attn.v_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn.q_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.encoder_attn.q_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn.out_proj.weight": torch.Size([1024, 1024]),
            "decoder.decoder.layers.0.encoder_attn.out_proj.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn_layer_norm.weight": torch.Size([1024]),
            "decoder.decoder.layers.0.encoder_attn_layer_norm.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.fc1.weight": torch.Size([4096, 1024]),
            "decoder.decoder.layers.0.fc1.bias": torch.Size([4096]),
            "decoder.decoder.layers.0.fc2.weight": torch.Size([1024, 4096]),
            "decoder.decoder.layers.0.fc2.bias": torch.Size([1024]),
            "decoder.decoder.layers.0.final_layer_norm.weight": torch.Size([1024]),
            "decoder.decoder.layers.0.final_layer_norm.bias": torch.Size([1024]),
            "decoder.decoder.layernorm_embedding.weight": torch.Size([1024]),
            "decoder.decoder.layernorm_embedding.bias": torch.Size([1024]),
            "decoder.bi_encoder_mlp.0.weight": torch.Size([1024, 1024]),
            "decoder.bi_encoder_mlp.0.bias": torch.Size([1024]),
            "decoder.bi_encoder_mlp.3.weight": torch.Size([1024, 1024]),
            "decoder.bi_encoder_mlp.3.bias": torch.Size([1024]),
            "decoder.encoder_mlp.0.weight": torch.Size([1024, 1024]),
            "decoder.encoder_mlp.0.bias": torch.Size([1024]),
            "decoder.encoder_mlp.3.weight": torch.Size([1024, 1024]),
            "decoder.encoder_mlp.3.bias": torch.Size([1024]),
        }
    else:
        raise ValueError(f"Unknown config: {config}")


def test_bart_layernorm_parameters(model):
    parameter_shapes = {
        name: param.shape
        for name, param in model.base_model_layernorm_parameters.items()
        if param.requires_grad
    }
    assert parameter_shapes == {
        "encoder.bart_encoder.layers.0.self_attn_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.0.self_attn_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.0.final_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.0.final_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.1.self_attn_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.1.self_attn_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.1.final_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.1.final_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.2.self_attn_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.2.self_attn_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.2.final_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.2.final_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.3.self_attn_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.3.self_attn_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.3.final_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.3.final_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.4.self_attn_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.4.self_attn_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.4.final_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.4.final_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.5.self_attn_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.5.self_attn_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.5.final_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.5.final_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.6.self_attn_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.6.self_attn_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.6.final_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.6.final_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.7.self_attn_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.7.self_attn_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.7.final_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.7.final_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.8.self_attn_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.8.self_attn_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.8.final_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.8.final_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.9.self_attn_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.9.self_attn_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.9.final_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.9.final_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.10.self_attn_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.10.self_attn_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.10.final_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.10.final_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.11.self_attn_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.11.self_attn_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.11.final_layer_norm.weight": torch.Size([1024]),
        "encoder.bart_encoder.layers.11.final_layer_norm.bias": torch.Size([1024]),
        "encoder.bart_encoder.layernorm_embedding.weight": torch.Size([1024]),
        "encoder.bart_encoder.layernorm_embedding.bias": torch.Size([1024]),
    }


def test_other_bart_parameters(model):
    parameter_shapes = {
        name: param.shape
        for name, param in model.base_model_other_parameters.items()
        if param.requires_grad
    }
    assert parameter_shapes == {
        "encoder.bart_encoder.embed_tokens.weight": torch.Size([50270, 1024]),
        "encoder.bart_encoder.embed_positions.weight": torch.Size([1026, 1024]),
        "encoder.bart_encoder.layers.0.self_attn.k_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.0.self_attn.k_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.0.self_attn.v_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.0.self_attn.v_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.0.self_attn.q_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.0.self_attn.q_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.0.self_attn.out_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.0.self_attn.out_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.0.fc1.weight": torch.Size([4096, 1024]),
        "encoder.bart_encoder.layers.0.fc1.bias": torch.Size([4096]),
        "encoder.bart_encoder.layers.0.fc2.weight": torch.Size([1024, 4096]),
        "encoder.bart_encoder.layers.0.fc2.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.1.self_attn.k_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.1.self_attn.k_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.1.self_attn.v_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.1.self_attn.v_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.1.self_attn.q_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.1.self_attn.q_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.1.self_attn.out_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.1.self_attn.out_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.1.fc1.weight": torch.Size([4096, 1024]),
        "encoder.bart_encoder.layers.1.fc1.bias": torch.Size([4096]),
        "encoder.bart_encoder.layers.1.fc2.weight": torch.Size([1024, 4096]),
        "encoder.bart_encoder.layers.1.fc2.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.2.self_attn.k_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.2.self_attn.k_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.2.self_attn.v_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.2.self_attn.v_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.2.self_attn.q_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.2.self_attn.q_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.2.self_attn.out_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.2.self_attn.out_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.2.fc1.weight": torch.Size([4096, 1024]),
        "encoder.bart_encoder.layers.2.fc1.bias": torch.Size([4096]),
        "encoder.bart_encoder.layers.2.fc2.weight": torch.Size([1024, 4096]),
        "encoder.bart_encoder.layers.2.fc2.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.3.self_attn.k_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.3.self_attn.k_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.3.self_attn.v_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.3.self_attn.v_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.3.self_attn.q_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.3.self_attn.q_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.3.self_attn.out_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.3.self_attn.out_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.3.fc1.weight": torch.Size([4096, 1024]),
        "encoder.bart_encoder.layers.3.fc1.bias": torch.Size([4096]),
        "encoder.bart_encoder.layers.3.fc2.weight": torch.Size([1024, 4096]),
        "encoder.bart_encoder.layers.3.fc2.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.4.self_attn.k_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.4.self_attn.k_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.4.self_attn.v_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.4.self_attn.v_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.4.self_attn.q_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.4.self_attn.q_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.4.self_attn.out_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.4.self_attn.out_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.4.fc1.weight": torch.Size([4096, 1024]),
        "encoder.bart_encoder.layers.4.fc1.bias": torch.Size([4096]),
        "encoder.bart_encoder.layers.4.fc2.weight": torch.Size([1024, 4096]),
        "encoder.bart_encoder.layers.4.fc2.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.5.self_attn.k_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.5.self_attn.k_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.5.self_attn.v_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.5.self_attn.v_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.5.self_attn.q_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.5.self_attn.q_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.5.self_attn.out_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.5.self_attn.out_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.5.fc1.weight": torch.Size([4096, 1024]),
        "encoder.bart_encoder.layers.5.fc1.bias": torch.Size([4096]),
        "encoder.bart_encoder.layers.5.fc2.weight": torch.Size([1024, 4096]),
        "encoder.bart_encoder.layers.5.fc2.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.6.self_attn.k_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.6.self_attn.k_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.6.self_attn.v_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.6.self_attn.v_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.6.self_attn.q_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.6.self_attn.q_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.6.self_attn.out_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.6.self_attn.out_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.6.fc1.weight": torch.Size([4096, 1024]),
        "encoder.bart_encoder.layers.6.fc1.bias": torch.Size([4096]),
        "encoder.bart_encoder.layers.6.fc2.weight": torch.Size([1024, 4096]),
        "encoder.bart_encoder.layers.6.fc2.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.7.self_attn.k_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.7.self_attn.k_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.7.self_attn.v_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.7.self_attn.v_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.7.self_attn.q_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.7.self_attn.q_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.7.self_attn.out_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.7.self_attn.out_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.7.fc1.weight": torch.Size([4096, 1024]),
        "encoder.bart_encoder.layers.7.fc1.bias": torch.Size([4096]),
        "encoder.bart_encoder.layers.7.fc2.weight": torch.Size([1024, 4096]),
        "encoder.bart_encoder.layers.7.fc2.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.8.self_attn.k_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.8.self_attn.k_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.8.self_attn.v_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.8.self_attn.v_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.8.self_attn.q_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.8.self_attn.q_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.8.self_attn.out_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.8.self_attn.out_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.8.fc1.weight": torch.Size([4096, 1024]),
        "encoder.bart_encoder.layers.8.fc1.bias": torch.Size([4096]),
        "encoder.bart_encoder.layers.8.fc2.weight": torch.Size([1024, 4096]),
        "encoder.bart_encoder.layers.8.fc2.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.9.self_attn.k_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.9.self_attn.k_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.9.self_attn.v_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.9.self_attn.v_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.9.self_attn.q_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.9.self_attn.q_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.9.self_attn.out_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.9.self_attn.out_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.9.fc1.weight": torch.Size([4096, 1024]),
        "encoder.bart_encoder.layers.9.fc1.bias": torch.Size([4096]),
        "encoder.bart_encoder.layers.9.fc2.weight": torch.Size([1024, 4096]),
        "encoder.bart_encoder.layers.9.fc2.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.10.self_attn.k_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.10.self_attn.k_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.10.self_attn.v_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.10.self_attn.v_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.10.self_attn.q_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.10.self_attn.q_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.10.self_attn.out_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.10.self_attn.out_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.10.fc1.weight": torch.Size([4096, 1024]),
        "encoder.bart_encoder.layers.10.fc1.bias": torch.Size([4096]),
        "encoder.bart_encoder.layers.10.fc2.weight": torch.Size([1024, 4096]),
        "encoder.bart_encoder.layers.10.fc2.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.11.self_attn.k_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.11.self_attn.k_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.11.self_attn.v_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.11.self_attn.v_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.11.self_attn.q_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.11.self_attn.q_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.11.self_attn.out_proj.weight": torch.Size([1024, 1024]),
        "encoder.bart_encoder.layers.11.self_attn.out_proj.bias": torch.Size([1024]),
        "encoder.bart_encoder.layers.11.fc1.weight": torch.Size([4096, 1024]),
        "encoder.bart_encoder.layers.11.fc1.bias": torch.Size([4096]),
        "encoder.bart_encoder.layers.11.fc2.weight": torch.Size([1024, 4096]),
        "encoder.bart_encoder.layers.11.fc2.bias": torch.Size([1024]),
    }
