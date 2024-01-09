import pytest
import torch
from transformers import (
    BartModel,
    BeamSearchScorer,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)
from transformers.generation import BeamSearchEncoderDecoderOutput

from pie_modules.models.base_models import (
    BartAsPointerNetwork,
    BartModelWithDecoderPositionIds,
)
from tests import _config_to_str

# TODO: is there an even smaller model we can use?
MODEL_NAME_OR_PATH = "sshleifer/distilbart-xsum-12-1"
CONFIGS = [{}, {"decoder_position_id_pattern": [0, 0, 1, 0, 0, 1, 1]}]
CONFIG_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIG_DICT.keys())
def config_str(request):
    return request.param


@pytest.fixture(scope="module")
def config(config_str):
    return CONFIG_DICT[config_str]


@pytest.fixture(scope="module")
def document():
    from pytorch_ie.annotations import BinaryRelation, LabeledSpan
    from pytorch_ie.documents import (
        TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
    )

    doc = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(
        text="This is a dummy text about nothing. Trust me."
    )
    span1 = LabeledSpan(start=10, end=20, label="content")
    span2 = LabeledSpan(start=27, end=34, label="topic")
    span3 = LabeledSpan(start=42, end=44, label="person")
    doc.labeled_spans.extend([span1, span2, span3])
    assert str(span1) == "dummy text"
    assert str(span2) == "nothing"
    assert str(span3) == "me"
    rel = BinaryRelation(head=span1, tail=span2, label="is_about")
    doc.binary_relations.append(rel)
    assert str(rel.label) == "is_about"
    assert str(rel.head) == "dummy text"
    assert str(rel.tail) == "nothing"

    sent1 = LabeledSpan(start=0, end=35, label="1")
    sent2 = LabeledSpan(start=36, end=45, label="2")
    doc.labeled_partitions.extend([sent1, sent2])
    assert str(sent1) == "This is a dummy text about nothing."
    assert str(sent2) == "Trust me."
    return doc


@pytest.fixture(scope="module")
def taskmodule(document):
    from pie_modules.taskmodules import PointerNetworkTaskModuleForEnd2EndRE

    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        tokenizer_name_or_path=MODEL_NAME_OR_PATH,
        partition_layer_name="labeled_partitions",
        create_constraints=True,
    )

    taskmodule.prepare(documents=[document])

    return taskmodule


@pytest.fixture(scope="module")
def model(config) -> BartAsPointerNetwork:
    model_name_or_path = MODEL_NAME_OR_PATH

    torch.random.manual_seed(42)
    model = BartAsPointerNetwork.from_pretrained(
        model_name_or_path,
        # label id space
        bos_token_id=0,  # taskmodule.bos_id,
        eos_token_id=1,  # taskmodule.eos_id,
        pad_token_id=1,  # taskmodule1.eos_id,
        # target token id space
        target_token_ids=[0, 2, 50266, 50269, 50268, 50265, 50267],  # taskmodule.target_token_ids,
        # mapping to better initialize the label embedding weights
        # taken from taskmodule.label_embedding_weight_mapping
        embedding_weight_mapping={
            50266: [39763],
            50269: [10166],
            50268: [5970],
            50265: [45260],
            50267: [354, 1215, 9006],
        },
        **config,
    )

    return model


def test_model(model):
    assert isinstance(model, BartAsPointerNetwork)
    if model.config.decoder_position_id_pattern is None:
        assert isinstance(model.model, BartModel)
    else:
        assert isinstance(model.model, BartModelWithDecoderPositionIds)


@pytest.fixture(scope="module")
def batch():
    inputs = {
        "input_ids": torch.tensor(
            [
                [0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 2],
                [0, 18823, 162, 4, 2, 1, 1, 1, 1, 1],
            ]
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
        ),
    }
    targets = {
        "labels": torch.tensor([[14, 14, 5, 11, 12, 3, 6, 1], [9, 9, 4, 2, 2, 2, 2, 1]]),
        "decoder_attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]
        ),
    }
    return inputs, targets


@pytest.fixture(scope="module")
def batch_with_constraints(batch):
    constraints = torch.tensor(
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
    )
    targets_with_constraints = {**batch[1], "constraints": constraints}
    return batch[0], targets_with_constraints


@pytest.mark.skip(reason="This is just to show how to create the batch.")
def test_batch_with_constraints(batch_with_constraints, taskmodule, document):
    inputs, targets = batch_with_constraints
    task_encodings = taskmodule.encode([document], encode_target=True)
    batch_from_documents = taskmodule.collate(task_encodings)
    inputs_from_documents, targets_from_documents = batch_from_documents
    for key in inputs:
        torch.testing.assert_close(inputs[key], inputs_from_documents[key])

    for key in targets:
        torch.testing.assert_close(targets[key], targets_from_documents[key])


@pytest.fixture(scope="module")
def decoder_input_ids(model):
    # taken from batch[1]["labels"]
    labels = torch.tensor([[14, 14, 5, 11, 12, 3, 6, 1], [9, 9, 4, 2, 2, 2, 2, 1]])
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=labels)
    return decoder_input_ids


def test_prepare_decoder_input_ids_from_labels(decoder_input_ids):
    assert decoder_input_ids.shape == (2, 8)
    torch.testing.assert_close(
        decoder_input_ids,
        torch.tensor([[0, 14, 14, 5, 11, 12, 3, 6], [0, 9, 9, 4, 2, 2, 2, 2]]),
    )


def test_forward(model, batch, decoder_input_ids):
    inputs, targets = batch
    torch.manual_seed(42)
    outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
    assert outputs.loss is None
    assert outputs.logits is not None
    # shape: (batch_size, output_seq_len, target_size=num_target_ids+num_offsets)
    assert outputs.logits.shape == (2, 8, 17)
    torch.testing.assert_close(
        # check only the first sequence output
        outputs.logits[:, 0, :],
        torch.tensor(
            [
                [
                    -1.0000000138484279e24,
                    3.682220935821533,
                    -0.9326803088188171,
                    -0.22636914253234863,
                    -0.48131969571113586,
                    0.19212311506271362,
                    0.909582257270813,
                    -0.26033997535705566,
                    4.807479381561279,
                    1.8076038360595703,
                    1.1034351587295532,
                    -0.13411812484264374,
                    1.15517258644104,
                    0.4741376042366028,
                    -0.458639532327652,
                    2.8401029109954834,
                    -1.0000000331813535e32,
                ],
                [
                    -1.0000000138484279e24,
                    3.8057942390441895,
                    -1.143894076347351,
                    -0.827788233757019,
                    -0.21552489697933197,
                    -0.2897048890590668,
                    0.34937697649002075,
                    -0.6374335289001465,
                    3.0700652599334717,
                    1.6204571723937988,
                    3.234084367752075,
                    -1.0000000331813535e32,
                    -1.0000000331813535e32,
                    -1.0000000331813535e32,
                    -1.0000000331813535e32,
                    -1.0000000331813535e32,
                    -1.0000000331813535e32,
                ],
            ]
        ),
    )


def test_forward_with_labels(model, batch):
    inputs, targets = batch
    targets_without_constraints = {
        key: value for key, value in targets.items() if key != "constraints"
    }
    assert set(inputs) == {"input_ids", "attention_mask"}
    assert set(targets_without_constraints) == {"labels", "decoder_attention_mask"}
    torch.manual_seed(42)
    outputs = model(**inputs, **targets_without_constraints)
    loss = outputs.loss
    if isinstance(model.model, BartModel):
        torch.testing.assert_close(loss, torch.tensor(3.883049488067627))
    elif isinstance(model.model, BartModelWithDecoderPositionIds):
        torch.testing.assert_close(loss, torch.tensor(4.204827308654785))
    else:
        raise ValueError(f"Unknown model type {type(model.model)}")


def test_forward_with_labels_and_constraints(model, batch_with_constraints):
    inputs, targets = batch_with_constraints
    assert set(inputs) == {"input_ids", "attention_mask"}
    assert set(targets) == {"labels", "decoder_attention_mask", "constraints"}
    torch.manual_seed(42)
    outputs = model(**inputs, **targets)
    loss = outputs.loss
    if isinstance(model.model, BartModel):
        torch.testing.assert_close(loss, torch.tensor(6.278500556945801))
    elif isinstance(model.model, BartModelWithDecoderPositionIds):
        torch.testing.assert_close(loss, torch.tensor(6.575843334197998))
    else:
        raise ValueError(f"Unknown model type {type(model.model)}")


@pytest.fixture(scope="module")
def empty_decoder_input_ids(batch, model):
    inputs, targets = batch
    batch_size, seq_len = inputs["input_ids"].shape
    decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long) * model.config.bos_token_id
    torch.testing.assert_close(
        decoder_input_ids,
        torch.tensor([[0], [0]]),
    )
    return decoder_input_ids


@pytest.fixture(scope="module")
def encoder_outputs(model, batch):
    inputs, targets = batch
    torch.manual_seed(42)
    encoder_outputs = model.get_encoder()(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )
    return encoder_outputs


@pytest.fixture(scope="module")
def prepared_encoder_decoder_kwargs_for_generation(
    model, batch, empty_decoder_input_ids, encoder_outputs
):
    model_kwargs = {
        "attention_mask": batch[0]["attention_mask"],
        "output_attentions": False,
        "output_hidden_states": False,
        "use_cache": True,
    }
    torch.manual_seed(42)
    prepared_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
        inputs_tensor=batch[0]["input_ids"],
        model_kwargs=model_kwargs,
        model_input_name="input_ids",
    )
    return prepared_kwargs


def test_prepare_encoder_decoder_kwargs_for_generation(
    prepared_encoder_decoder_kwargs_for_generation, batch, encoder_outputs
):
    model_kwargs = {
        "attention_mask": batch[0]["attention_mask"],
        "output_attentions": False,
        "output_hidden_states": False,
        "use_cache": True,
    }

    assert set(prepared_encoder_decoder_kwargs_for_generation) == set(model_kwargs) | {
        "encoder_input_ids",
        "encoder_attention_mask",
        "encoder_outputs",
    }
    torch.testing.assert_close(
        prepared_encoder_decoder_kwargs_for_generation["encoder_input_ids"],
        batch[0]["input_ids"],
    )
    torch.testing.assert_close(
        prepared_encoder_decoder_kwargs_for_generation["encoder_attention_mask"],
        batch[0]["attention_mask"],
    )
    torch.testing.assert_close(
        prepared_encoder_decoder_kwargs_for_generation["encoder_outputs"].last_hidden_state,
        encoder_outputs.last_hidden_state,
    )


def test_prepare_inputs_for_generation(
    model,
    prepared_encoder_decoder_kwargs_for_generation,
    empty_decoder_input_ids,
    batch,
    encoder_outputs,
):
    result = model.prepare_inputs_for_generation(
        decoder_input_ids=empty_decoder_input_ids, **prepared_encoder_decoder_kwargs_for_generation
    )
    result_keys = {
        "input_ids",
        "attention_mask",
        "encoder_outputs",
        "decoder_input_ids",
        "decoder_attention_mask",
        "past_key_values",
        "use_cache",
        "head_mask",
        "decoder_head_mask",
        "cross_attn_head_mask",
    }
    if isinstance(model.model, BartModelWithDecoderPositionIds):
        result_keys.add("decoder_position_ids")
    assert set(result) == result_keys
    torch.testing.assert_close(
        result["input_ids"],
        batch[0]["input_ids"],
    )
    torch.testing.assert_close(
        result["attention_mask"],
        batch[0]["attention_mask"],
    )
    torch.testing.assert_close(
        result["encoder_outputs"].last_hidden_state,
        encoder_outputs.last_hidden_state,
    )
    torch.testing.assert_close(
        result["decoder_input_ids"],
        empty_decoder_input_ids,
    )
    assert result["decoder_attention_mask"] is None
    assert result["past_key_values"] is None
    assert result["use_cache"] is True
    assert result["head_mask"] is None
    assert result["decoder_head_mask"] is None
    assert result["cross_attn_head_mask"] is None
    if "decoder_position_ids" in result:
        torch.testing.assert_close(result["decoder_position_ids"], torch.tensor([[0], [0]]))


def test_prepare_inputs_for_generation_with_past_key_values(
    model,
    prepared_encoder_decoder_kwargs_for_generation,
    batch,
    encoder_outputs,
):
    # shallow copy to avoid changing the original dict
    kwargs = dict(prepared_encoder_decoder_kwargs_for_generation)
    kwargs["decoder_input_ids"] = torch.tensor(
        [
            [0, 8, 9],
            [0, 8, 10],
            [0, 8, 15],
            [0, 8, 8],
            [0, 9, 10],
            [0, 8, 12],
            [0, 8, 9],
            [0, 8, 10],
            [0, 9, 10],
            [0, 8, 8],
            [0, 9, 9],
            [0, 8, 6],
        ]
    )
    # 12 is batch_size (2) * num_beams (6),
    # 16 is number of encoder / decoder attention heads,
    # 2 is the length of already generated tokens / 10 is the length of the encoder input,
    # 64 seems to be the size of the hidden states
    dummy_past_key_values = (
        torch.zeros((12, 16, 2, 64)),
        torch.zeros((12, 16, 2, 64)),
        torch.zeros((12, 16, 10, 64)),
        torch.zeros((12, 16, 10, 64)),
    )

    result = model.prepare_inputs_for_generation(past_key_values=dummy_past_key_values, **kwargs)
    if isinstance(model.model, BartModel):
        assert len(result) == 10
    elif isinstance(model.model, BartModelWithDecoderPositionIds):
        assert len(result) == 11
    else:
        raise ValueError(f"Unknown model type {type(model.model)}")
    torch.testing.assert_close(
        result["input_ids"],
        batch[0]["input_ids"],
    )
    torch.testing.assert_close(
        result["attention_mask"],
        batch[0]["attention_mask"],
    )
    torch.testing.assert_close(
        result["encoder_outputs"].last_hidden_state,
        encoder_outputs.last_hidden_state,
    )
    torch.testing.assert_close(
        result["decoder_input_ids"],
        # just the last id for each entry
        torch.tensor([[9], [10], [15], [8], [10], [12], [9], [10], [10], [8], [9], [6]]),
    )
    assert result["decoder_attention_mask"] is None
    assert result["past_key_values"] is dummy_past_key_values
    assert result["use_cache"] is True
    assert result["head_mask"] is None
    assert result["decoder_head_mask"] is None
    assert result["cross_attn_head_mask"] is None
    if "decoder_position_ids" in result:
        torch.testing.assert_close(
            result["decoder_position_ids"],
            # originally this was 0 from the pattern, but got shifted for the position-bos and position-pad indices
            torch.tensor([[2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2]]),
        )


def test_generate(model, batch, empty_decoder_input_ids):
    inputs, targets = batch
    batch_size, seq_len = inputs["input_ids"].shape
    torch.manual_seed(42)
    outputs = model.generate(**inputs)
    if isinstance(model.model, BartModel):
        assert outputs.shape == (batch_size, 12)
        torch.testing.assert_close(
            outputs,
            torch.tensor(
                [
                    [0, 8, 9, 10, 10, 12, 13, 10, 12, 12, 15, 1],
                    [0, 8, 9, 9, 9, 8, 9, 6, 6, 6, 10, 1],
                ]
            ),
        )
    elif isinstance(model.model, BartModelWithDecoderPositionIds):
        assert outputs.shape == (batch_size, 12)
        torch.testing.assert_close(
            outputs,
            torch.tensor(
                [
                    [0, 8, 9, 10, 12, 13, 10, 12, 12, 13, 15, 1],
                    [0, 8, 8, 8, 9, 9, 9, 8, 9, 8, 10, 1],
                ]
            ),
        )
    else:
        raise ValueError(f"Unknown model type {type(model.model)}")


def test_head_named_params(model):
    parameter_shapes = {name: tuple(param.shape) for name, param in model.head_named_params()}
    assert parameter_shapes == {
        "pointer_head.encoder_mlp.0.bias": (1024,),
        "pointer_head.encoder_mlp.0.weight": (1024, 1024),
        "pointer_head.encoder_mlp.3.bias": (1024,),
        "pointer_head.encoder_mlp.3.weight": (1024, 1024),
    }


def test_encoder_only_named_params(model):
    parameter_shapes = {
        name: tuple(param.shape) for name, param in model.encoder_only_named_params()
    }
    assert len(parameter_shapes) == 195
    assert parameter_shapes == {
        "model.encoder.embed_positions.weight": (1026, 1024),
        "model.encoder.layers.0.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.0.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.0.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.0.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.0.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.0.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.0.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.0.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.0.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.0.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.0.fc1.weight": (4096, 1024),
        "model.encoder.layers.0.fc1.bias": (4096,),
        "model.encoder.layers.0.fc2.weight": (1024, 4096),
        "model.encoder.layers.0.fc2.bias": (1024,),
        "model.encoder.layers.0.final_layer_norm.weight": (1024,),
        "model.encoder.layers.0.final_layer_norm.bias": (1024,),
        "model.encoder.layers.1.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.1.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.1.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.1.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.1.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.1.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.1.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.1.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.1.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.1.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.1.fc1.weight": (4096, 1024),
        "model.encoder.layers.1.fc1.bias": (4096,),
        "model.encoder.layers.1.fc2.weight": (1024, 4096),
        "model.encoder.layers.1.fc2.bias": (1024,),
        "model.encoder.layers.1.final_layer_norm.weight": (1024,),
        "model.encoder.layers.1.final_layer_norm.bias": (1024,),
        "model.encoder.layers.2.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.2.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.2.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.2.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.2.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.2.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.2.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.2.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.2.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.2.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.2.fc1.weight": (4096, 1024),
        "model.encoder.layers.2.fc1.bias": (4096,),
        "model.encoder.layers.2.fc2.weight": (1024, 4096),
        "model.encoder.layers.2.fc2.bias": (1024,),
        "model.encoder.layers.2.final_layer_norm.weight": (1024,),
        "model.encoder.layers.2.final_layer_norm.bias": (1024,),
        "model.encoder.layers.3.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.3.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.3.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.3.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.3.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.3.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.3.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.3.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.3.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.3.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.3.fc1.weight": (4096, 1024),
        "model.encoder.layers.3.fc1.bias": (4096,),
        "model.encoder.layers.3.fc2.weight": (1024, 4096),
        "model.encoder.layers.3.fc2.bias": (1024,),
        "model.encoder.layers.3.final_layer_norm.weight": (1024,),
        "model.encoder.layers.3.final_layer_norm.bias": (1024,),
        "model.encoder.layers.4.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.4.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.4.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.4.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.4.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.4.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.4.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.4.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.4.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.4.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.4.fc1.weight": (4096, 1024),
        "model.encoder.layers.4.fc1.bias": (4096,),
        "model.encoder.layers.4.fc2.weight": (1024, 4096),
        "model.encoder.layers.4.fc2.bias": (1024,),
        "model.encoder.layers.4.final_layer_norm.weight": (1024,),
        "model.encoder.layers.4.final_layer_norm.bias": (1024,),
        "model.encoder.layers.5.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.5.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.5.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.5.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.5.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.5.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.5.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.5.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.5.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.5.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.5.fc1.weight": (4096, 1024),
        "model.encoder.layers.5.fc1.bias": (4096,),
        "model.encoder.layers.5.fc2.weight": (1024, 4096),
        "model.encoder.layers.5.fc2.bias": (1024,),
        "model.encoder.layers.5.final_layer_norm.weight": (1024,),
        "model.encoder.layers.5.final_layer_norm.bias": (1024,),
        "model.encoder.layers.6.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.6.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.6.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.6.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.6.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.6.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.6.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.6.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.6.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.6.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.6.fc1.weight": (4096, 1024),
        "model.encoder.layers.6.fc1.bias": (4096,),
        "model.encoder.layers.6.fc2.weight": (1024, 4096),
        "model.encoder.layers.6.fc2.bias": (1024,),
        "model.encoder.layers.6.final_layer_norm.weight": (1024,),
        "model.encoder.layers.6.final_layer_norm.bias": (1024,),
        "model.encoder.layers.7.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.7.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.7.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.7.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.7.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.7.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.7.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.7.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.7.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.7.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.7.fc1.weight": (4096, 1024),
        "model.encoder.layers.7.fc1.bias": (4096,),
        "model.encoder.layers.7.fc2.weight": (1024, 4096),
        "model.encoder.layers.7.fc2.bias": (1024,),
        "model.encoder.layers.7.final_layer_norm.weight": (1024,),
        "model.encoder.layers.7.final_layer_norm.bias": (1024,),
        "model.encoder.layers.8.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.8.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.8.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.8.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.8.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.8.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.8.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.8.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.8.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.8.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.8.fc1.weight": (4096, 1024),
        "model.encoder.layers.8.fc1.bias": (4096,),
        "model.encoder.layers.8.fc2.weight": (1024, 4096),
        "model.encoder.layers.8.fc2.bias": (1024,),
        "model.encoder.layers.8.final_layer_norm.weight": (1024,),
        "model.encoder.layers.8.final_layer_norm.bias": (1024,),
        "model.encoder.layers.9.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.9.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.9.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.9.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.9.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.9.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.9.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.9.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.9.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.9.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.9.fc1.weight": (4096, 1024),
        "model.encoder.layers.9.fc1.bias": (4096,),
        "model.encoder.layers.9.fc2.weight": (1024, 4096),
        "model.encoder.layers.9.fc2.bias": (1024,),
        "model.encoder.layers.9.final_layer_norm.weight": (1024,),
        "model.encoder.layers.9.final_layer_norm.bias": (1024,),
        "model.encoder.layers.10.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.10.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.10.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.10.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.10.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.10.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.10.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.10.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.10.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.10.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.10.fc1.weight": (4096, 1024),
        "model.encoder.layers.10.fc1.bias": (4096,),
        "model.encoder.layers.10.fc2.weight": (1024, 4096),
        "model.encoder.layers.10.fc2.bias": (1024,),
        "model.encoder.layers.10.final_layer_norm.weight": (1024,),
        "model.encoder.layers.10.final_layer_norm.bias": (1024,),
        "model.encoder.layers.11.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.11.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.11.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.11.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.11.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.11.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.11.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.11.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.11.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.11.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.11.fc1.weight": (4096, 1024),
        "model.encoder.layers.11.fc1.bias": (4096,),
        "model.encoder.layers.11.fc2.weight": (1024, 4096),
        "model.encoder.layers.11.fc2.bias": (1024,),
        "model.encoder.layers.11.final_layer_norm.weight": (1024,),
        "model.encoder.layers.11.final_layer_norm.bias": (1024,),
        "model.encoder.layernorm_embedding.weight": (1024,),
        "model.encoder.layernorm_embedding.bias": (1024,),
    }


def test_decoder_only_named_params(model):
    parameter_shapes = {
        name: tuple(param.shape) for name, param in model.decoder_only_named_params()
    }
    assert len(parameter_shapes) == 29
    assert parameter_shapes == {
        "model.decoder.embed_positions.weight": (1026, 1024),
        "model.decoder.layers.0.self_attn.k_proj.weight": (1024, 1024),
        "model.decoder.layers.0.self_attn.k_proj.bias": (1024,),
        "model.decoder.layers.0.self_attn.v_proj.weight": (1024, 1024),
        "model.decoder.layers.0.self_attn.v_proj.bias": (1024,),
        "model.decoder.layers.0.self_attn.q_proj.weight": (1024, 1024),
        "model.decoder.layers.0.self_attn.q_proj.bias": (1024,),
        "model.decoder.layers.0.self_attn.out_proj.weight": (1024, 1024),
        "model.decoder.layers.0.self_attn.out_proj.bias": (1024,),
        "model.decoder.layers.0.self_attn_layer_norm.weight": (1024,),
        "model.decoder.layers.0.self_attn_layer_norm.bias": (1024,),
        "model.decoder.layers.0.encoder_attn.k_proj.weight": (1024, 1024),
        "model.decoder.layers.0.encoder_attn.k_proj.bias": (1024,),
        "model.decoder.layers.0.encoder_attn.v_proj.weight": (1024, 1024),
        "model.decoder.layers.0.encoder_attn.v_proj.bias": (1024,),
        "model.decoder.layers.0.encoder_attn.q_proj.weight": (1024, 1024),
        "model.decoder.layers.0.encoder_attn.q_proj.bias": (1024,),
        "model.decoder.layers.0.encoder_attn.out_proj.weight": (1024, 1024),
        "model.decoder.layers.0.encoder_attn.out_proj.bias": (1024,),
        "model.decoder.layers.0.encoder_attn_layer_norm.weight": (1024,),
        "model.decoder.layers.0.encoder_attn_layer_norm.bias": (1024,),
        "model.decoder.layers.0.fc1.weight": (4096, 1024),
        "model.decoder.layers.0.fc1.bias": (4096,),
        "model.decoder.layers.0.fc2.weight": (1024, 4096),
        "model.decoder.layers.0.fc2.bias": (1024,),
        "model.decoder.layers.0.final_layer_norm.weight": (1024,),
        "model.decoder.layers.0.final_layer_norm.bias": (1024,),
        "model.decoder.layernorm_embedding.weight": (1024,),
        "model.decoder.layernorm_embedding.bias": (1024,),
    }


def test_encoder_decoder_shared_named_params(model):
    parameter_shapes = {
        name: tuple(param.shape) for name, param in model.encoder_decoder_shared_named_params()
    }
    assert len(parameter_shapes) == 1
    assert parameter_shapes == {"model.shared.weight": (50270, 1024)}


def test_base_model_named_params(model):
    parameter_shapes = {
        name: tuple(param.shape) for name, param in model.base_model_named_params()
    }
    assert len(parameter_shapes) == 225
    encoder_only_parameter_shapes = {
        name: tuple(param.shape) for name, param in model.encoder_only_named_params()
    }
    decoder_only_parameter_shapes = {
        name: tuple(param.shape) for name, param in model.decoder_only_named_params()
    }
    shared_parameter_shapes = {
        name: tuple(param.shape) for name, param in model.encoder_decoder_shared_named_params()
    }
    expected_parameter_shapes = {
        **encoder_only_parameter_shapes,
        **decoder_only_parameter_shapes,
        **shared_parameter_shapes,
    }

    assert parameter_shapes == expected_parameter_shapes


def test_configure_optimizer(model):
    optimizer = model.configure_optimizer()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 0.001
    assert optimizer.defaults["weight_decay"] == model.config.weight_decay == 0.01
    assert len(optimizer.param_groups) == 6
    assert all(param_group["lr"] == model.config.lr for param_group in optimizer.param_groups)

    # head parameters
    assert optimizer.param_groups[0]["weight_decay"] == model.config.weight_decay == 0.01
    # decoder only layer norm parameters
    assert optimizer.param_groups[1]["weight_decay"] == model.config.weight_decay == 0.01
    # decoder only other parameters
    assert optimizer.param_groups[2]["weight_decay"] == model.config.weight_decay == 0.01
    # encoder only layer norm parameters
    assert (
        optimizer.param_groups[3]["weight_decay"] == model.config.encoder_layer_norm_decay == 0.001
    )
    # encoder only other parameters
    assert optimizer.param_groups[4]["weight_decay"] == model.config.weight_decay == 0.01
    # encoder-decoder shared parameters
    assert optimizer.param_groups[5]["weight_decay"] == model.config.weight_decay == 0.01

    all_optimized_parameters = set()
    for param_group in optimizer.param_groups:
        all_optimized_parameters.update(set(param_group["params"]))
    assert len(all_optimized_parameters) == 229
    # check that all model parameters are covered
    all_model_parameters = {param for name, param in model.named_parameters()}
    assert all_optimized_parameters == all_model_parameters


ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)


@pytest.mark.skip("This is just a showcase of how beam search works")
def test_bart_pointer_network_beam_search(model, taskmodule):
    encoder_input_str = ARTICLE_TO_SUMMARIZE  # "translate English to German: How old are you?"
    encoder_input_tokenized = taskmodule.tokenizer(encoder_input_str, return_tensors="pt")

    # lets run beam search using 3 beams
    num_beams = 3
    # define decoder start token ids
    decoder_input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    decoder_input_ids = decoder_input_ids * model.config.decoder_start_token_id

    # add encoder_outputs to model keyword arguments
    encoder = model.get_encoder()
    encoder_input_ids = encoder_input_tokenized.input_ids.repeat_interleave(num_beams, dim=0)
    encoder_attention_mask = encoder_input_tokenized.attention_mask.repeat_interleave(
        num_beams, dim=0
    )
    encoder_outputs = encoder(encoder_input_ids, return_dict=True)
    model_kwargs = {
        "encoder_outputs": encoder_outputs,
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
    }

    # instantiate beam scorer
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        num_beams=num_beams,
        device=model.device,
    )

    # instantiate logits processors
    logits_processor = LogitsProcessorList(
        [
            MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ]
    )

    outputs = model.beam_search(
        decoder_input_ids,
        beam_scorer,
        logits_processor=logits_processor,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        **model_kwargs,
    )

    torch.testing.assert_close(
        outputs,
        torch.tensor(
            [[0, 28, 41, 35, 33, 36, 17, 33, 36, 17, 33, 36, 17, 33, 36, 17, 33, 36, 37, 1]]
        ),
    )

    # result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # assert result == [
    #    " power lines in California have been shut down after a power provider said it was due to high winds."
    # ]


@pytest.mark.skip("This is just a showcase of how beam search with scores works")
def test_bart_pointer_network_generate_with_scores(model, taskmodule):
    encoder_input_str = ARTICLE_TO_SUMMARIZE  # "translate English to German: How old are you?"
    inputs = taskmodule.tokenizer(encoder_input_str, max_length=1024, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"],
        num_beams=3,
        min_length=5,
        max_length=20,
        return_dict_in_generate=True,
        output_scores=True,
    )
    assert isinstance(outputs, BeamSearchEncoderDecoderOutput)
    torch.testing.assert_close(outputs.sequences_scores, torch.tensor([-6.784079074859619]))
    torch.testing.assert_close(
        outputs.sequences,
        torch.tensor(
            [[0, 28, 41, 35, 33, 36, 17, 48, 36, 17, 33, 55, 35, 33, 17, 48, 55, 35, 48, 36]]
        ),
    )

    # result = tokenizer.batch_decode(
    #    summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # assert result == [" power lines in California have been shut down on Friday."]
