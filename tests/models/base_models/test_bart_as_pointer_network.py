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

# this is a small model that can be used for testing
MODEL_NAME_OR_PATH = "sshleifer/bart-tiny-random"
DECODER_POSITION_ID_PATTERN = [0, 0, 1, 0, 0, 1, 1]
CONFIGS = [{}, {"decoder_position_id_mode": "pattern"}]
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
        pad_token_id=1,  # taskmodule.eos_id,
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
        decoder_position_id_pattern=DECODER_POSITION_ID_PATTERN,
        **config,
    )

    return model


def test_model(model, config):
    assert model is not None
    named_parameters = dict(model.named_parameters())
    parameter_means = {k: v.sum().mean().item() for k, v in named_parameters.items()}
    parameter_means_expected = {
        "model.shared.weight": -17.04595184326172,
        "model.encoder.embed_positions.weight": -3.2610244750976562,
        "model.encoder.layers.0.self_attn.k_proj.weight": -0.2634662091732025,
        "model.encoder.layers.0.self_attn.k_proj.bias": 0.0,
        "model.encoder.layers.0.self_attn.v_proj.weight": -0.31434395909309387,
        "model.encoder.layers.0.self_attn.v_proj.bias": 0.0,
        "model.encoder.layers.0.self_attn.q_proj.weight": -0.5630431771278381,
        "model.encoder.layers.0.self_attn.q_proj.bias": 0.0,
        "model.encoder.layers.0.self_attn.out_proj.weight": -0.061936795711517334,
        "model.encoder.layers.0.self_attn.out_proj.bias": 0.0,
        "model.encoder.layers.0.self_attn_layer_norm.weight": 24.0,
        "model.encoder.layers.0.self_attn_layer_norm.bias": 0.0,
        "model.encoder.layers.0.fc1.weight": -0.3323897421360016,
        "model.encoder.layers.0.fc1.bias": 0.0,
        "model.encoder.layers.0.fc2.weight": 0.5965678095817566,
        "model.encoder.layers.0.fc2.bias": 0.0,
        "model.encoder.layers.0.final_layer_norm.weight": 24.0,
        "model.encoder.layers.0.final_layer_norm.bias": 0.0,
        "model.encoder.layers.1.self_attn.k_proj.weight": -0.4510859251022339,
        "model.encoder.layers.1.self_attn.k_proj.bias": 0.0,
        "model.encoder.layers.1.self_attn.v_proj.weight": 0.0683179721236229,
        "model.encoder.layers.1.self_attn.v_proj.bias": 0.0,
        "model.encoder.layers.1.self_attn.q_proj.weight": 0.3944129943847656,
        "model.encoder.layers.1.self_attn.q_proj.bias": 0.0,
        "model.encoder.layers.1.self_attn.out_proj.weight": 0.6753036975860596,
        "model.encoder.layers.1.self_attn.out_proj.bias": 0.0,
        "model.encoder.layers.1.self_attn_layer_norm.weight": 24.0,
        "model.encoder.layers.1.self_attn_layer_norm.bias": 0.0,
        "model.encoder.layers.1.fc1.weight": 0.2978851795196533,
        "model.encoder.layers.1.fc1.bias": 0.0,
        "model.encoder.layers.1.fc2.weight": -0.07737221568822861,
        "model.encoder.layers.1.fc2.bias": 0.0,
        "model.encoder.layers.1.final_layer_norm.weight": 24.0,
        "model.encoder.layers.1.final_layer_norm.bias": 0.0,
        "model.encoder.layernorm_embedding.weight": 24.0,
        "model.encoder.layernorm_embedding.bias": 0.0,
        "model.decoder.embed_positions.weight": -3.141465187072754,
        "model.decoder.layers.0.self_attn.k_proj.weight": -0.6152989268302917,
        "model.decoder.layers.0.self_attn.k_proj.bias": 0.0,
        "model.decoder.layers.0.self_attn.v_proj.weight": 0.29129207134246826,
        "model.decoder.layers.0.self_attn.v_proj.bias": 0.0,
        "model.decoder.layers.0.self_attn.q_proj.weight": 0.18711398541927338,
        "model.decoder.layers.0.self_attn.q_proj.bias": 0.0,
        "model.decoder.layers.0.self_attn.out_proj.weight": -0.11605234444141388,
        "model.decoder.layers.0.self_attn.out_proj.bias": 0.0,
        "model.decoder.layers.0.self_attn_layer_norm.weight": 24.0,
        "model.decoder.layers.0.self_attn_layer_norm.bias": 0.0,
        "model.decoder.layers.0.encoder_attn.k_proj.weight": -0.24503424763679504,
        "model.decoder.layers.0.encoder_attn.k_proj.bias": 0.0,
        "model.decoder.layers.0.encoder_attn.v_proj.weight": -0.23326119780540466,
        "model.decoder.layers.0.encoder_attn.v_proj.bias": 0.0,
        "model.decoder.layers.0.encoder_attn.q_proj.weight": -0.20252904295921326,
        "model.decoder.layers.0.encoder_attn.q_proj.bias": 0.0,
        "model.decoder.layers.0.encoder_attn.out_proj.weight": 0.5707532167434692,
        "model.decoder.layers.0.encoder_attn.out_proj.bias": 0.0,
        "model.decoder.layers.0.encoder_attn_layer_norm.weight": 24.0,
        "model.decoder.layers.0.encoder_attn_layer_norm.bias": 0.0,
        "model.decoder.layers.0.fc1.weight": 0.32172316312789917,
        "model.decoder.layers.0.fc1.bias": 0.0,
        "model.decoder.layers.0.fc2.weight": -0.007703527808189392,
        "model.decoder.layers.0.fc2.bias": 0.0,
        "model.decoder.layers.0.final_layer_norm.weight": 24.0,
        "model.decoder.layers.0.final_layer_norm.bias": 0.0,
        "model.decoder.layers.1.self_attn.k_proj.weight": -0.44178277254104614,
        "model.decoder.layers.1.self_attn.k_proj.bias": 0.0,
        "model.decoder.layers.1.self_attn.v_proj.weight": -0.4103395938873291,
        "model.decoder.layers.1.self_attn.v_proj.bias": 0.0,
        "model.decoder.layers.1.self_attn.q_proj.weight": 0.7464197874069214,
        "model.decoder.layers.1.self_attn.q_proj.bias": 0.0,
        "model.decoder.layers.1.self_attn.out_proj.weight": -0.39272433519363403,
        "model.decoder.layers.1.self_attn.out_proj.bias": 0.0,
        "model.decoder.layers.1.self_attn_layer_norm.weight": 24.0,
        "model.decoder.layers.1.self_attn_layer_norm.bias": 0.0,
        "model.decoder.layers.1.encoder_attn.k_proj.weight": -0.3977898061275482,
        "model.decoder.layers.1.encoder_attn.k_proj.bias": 0.0,
        "model.decoder.layers.1.encoder_attn.v_proj.weight": -0.5307084321975708,
        "model.decoder.layers.1.encoder_attn.v_proj.bias": 0.0,
        "model.decoder.layers.1.encoder_attn.q_proj.weight": -0.48503491282463074,
        "model.decoder.layers.1.encoder_attn.q_proj.bias": 0.0,
        "model.decoder.layers.1.encoder_attn.out_proj.weight": 0.46500909328460693,
        "model.decoder.layers.1.encoder_attn.out_proj.bias": 0.0,
        "model.decoder.layers.1.encoder_attn_layer_norm.weight": 24.0,
        "model.decoder.layers.1.encoder_attn_layer_norm.bias": 0.0,
        "model.decoder.layers.1.fc1.weight": 0.5949467420578003,
        "model.decoder.layers.1.fc1.bias": 0.0,
        "model.decoder.layers.1.fc2.weight": -0.37737518548965454,
        "model.decoder.layers.1.fc2.bias": 0.0,
        "model.decoder.layers.1.final_layer_norm.weight": 24.0,
        "model.decoder.layers.1.final_layer_norm.bias": 0.0,
        "model.decoder.layernorm_embedding.weight": 24.0,
        "model.decoder.layernorm_embedding.bias": 0.0,
        "pointer_head.encoder_mlp.0.weight": -0.28922533988952637,
        "pointer_head.encoder_mlp.0.bias": 0.0,
        "pointer_head.encoder_mlp.3.weight": 0.038049373775720596,
        "pointer_head.encoder_mlp.3.bias": 0.0,
    }
    assert set(parameter_means) == set(parameter_means_expected)
    for k in parameter_means:
        torch.testing.assert_close(
            torch.tensor(parameter_means[k]), torch.tensor(parameter_means_expected[k]), msg=k
        )
    assert isinstance(model, BartAsPointerNetwork)
    if config == {}:
        assert isinstance(model.model, BartModel)
    elif config == {"decoder_position_id_mode": "pattern"}:
        assert isinstance(model.model, BartModelWithDecoderPositionIds)
    else:
        raise ValueError(f"Unknown config: {config}")


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


def test_forward(model, batch, decoder_input_ids, config):
    inputs, targets = batch
    torch.manual_seed(42)
    outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
    assert outputs.loss is None
    assert outputs.logits is not None
    # shape: (batch_size, output_seq_len, target_size=num_target_ids+num_offsets)
    assert outputs.logits.shape == (2, 8, 17)
    # check exact values only for the first sequence output
    torch.testing.assert_close(
        outputs.logits[:, 0, :],
        torch.tensor(
            [
                [
                    -1.0000000138484279e24,
                    -0.23238050937652588,
                    0.2958170473575592,
                    0.05529244616627693,
                    0.042530909180641174,
                    0.10081343352794647,
                    -0.07145103812217712,
                    0.1258169263601303,
                    -0.08469001948833466,
                    0.09224237501621246,
                    0.022736266255378723,
                    -0.056650951504707336,
                    0.00475323386490345,
                    0.0679982528090477,
                    0.04947580397129059,
                    -0.08609804511070251,
                    -1.0000000331813535e32,
                ],
                [
                    -1.0000000138484279e24,
                    -0.23274853825569153,
                    0.2960396707057953,
                    0.05556505173444748,
                    0.04273711144924164,
                    0.10071950405836105,
                    -0.07135685533285141,
                    0.12565924227237701,
                    0.06317455321550369,
                    0.06436678022146225,
                    -0.07508281618356705,
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
    # check the sum of all logits
    if config == {}:
        torch.testing.assert_close(
            outputs.logits.sum(0).sum(0),
            torch.tensor(
                [
                    -1.6000000221574846e25,
                    -0.9064984917640686,
                    1.1896746158599854,
                    0.979636013507843,
                    0.1837124079465866,
                    1.3070943355560303,
                    -0.12108190357685089,
                    0.625571072101593,
                    -0.12742891907691956,
                    0.6052249073982239,
                    -0.4059778153896332,
                    -8.000000265450828e32,
                    -8.000000265450828e32,
                    -8.000000265450828e32,
                    -8.000000265450828e32,
                    -8.000000265450828e32,
                    -1.6000000530901656e33,
                ]
            ),
        )
    elif config == {"decoder_position_id_mode": "pattern"}:
        torch.testing.assert_close(
            outputs.logits.sum(0).sum(0),
            torch.tensor(
                [
                    -1.6000000221574846e25,
                    -0.5539568662643433,
                    0.7004714012145996,
                    1.5720455646514893,
                    -0.3760949671268463,
                    0.7738710641860962,
                    -0.10904453694820404,
                    0.6459492444992065,
                    -0.013624418526887894,
                    0.48998790979385376,
                    -0.5045579075813293,
                    -8.000000265450828e32,
                    -8.000000265450828e32,
                    -8.000000265450828e32,
                    -8.000000265450828e32,
                    -8.000000265450828e32,
                    -1.6000000530901656e33,
                ]
            ),
        )
    else:
        raise ValueError(f"Unknown config: {config}")


def test_forward_with_labels(model, batch, config):
    inputs, targets = batch
    targets_without_constraints = {
        key: value for key, value in targets.items() if key != "constraints"
    }
    assert set(inputs) == {"input_ids", "attention_mask"}
    assert set(targets_without_constraints) == {"labels", "decoder_attention_mask"}
    torch.manual_seed(42)
    outputs = model(**inputs, **targets_without_constraints)
    loss = outputs.loss
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(2.4492502212524414))
    elif config == {"decoder_position_id_mode": "pattern"}:
        torch.testing.assert_close(loss, torch.tensor(2.421541690826416))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_forward_with_labels_and_constraints(model, batch_with_constraints, config):
    inputs, targets = batch_with_constraints
    assert set(inputs) == {"input_ids", "attention_mask"}
    assert set(targets) == {"labels", "decoder_attention_mask", "constraints"}
    torch.manual_seed(42)
    outputs = model(**inputs, **targets)
    loss = outputs.loss
    if config == {}:
        torch.testing.assert_close(loss, torch.tensor(4.772888660430908))
    elif config == {"decoder_position_id_mode": "pattern"}:
        torch.testing.assert_close(loss, torch.tensor(4.747316360473633))
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
    config,
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
    if model.pointer_head.use_prepared_position_ids:
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
    if config == {}:
        assert "decoder_position_ids" not in result
    elif config == {"decoder_position_id_mode": "pattern"}:
        torch.testing.assert_close(result["decoder_position_ids"], torch.tensor([[0], [0]]))
    else:
        raise ValueError(f"Unknown config: {config}")


def test_prepare_inputs_for_generation_with_past_key_values(
    model,
    prepared_encoder_decoder_kwargs_for_generation,
    batch,
    encoder_outputs,
    config,
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
    if config == {}:
        assert len(result) == 10
    elif config == {"decoder_position_id_mode": "pattern"}:
        assert len(result) == 11
    else:
        raise ValueError(f"Unknown config: {config}")
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


def test_generate(model, batch, empty_decoder_input_ids, config):
    inputs, targets = batch
    batch_size, seq_len = inputs["input_ids"].shape
    torch.manual_seed(42)
    outputs = model.generate(**inputs)
    if config == {}:
        assert outputs.shape == (batch_size, 20)  # note that 20 is the model.config.max_length
        torch.testing.assert_close(
            outputs,
            torch.tensor(
                [
                    [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                ]
            ),
        )
    elif config == {"decoder_position_id_mode": "pattern"}:
        assert outputs.shape == (batch_size, 20)  # note that 20 is the model.config.max_length
        torch.testing.assert_close(
            outputs,
            torch.tensor(
                [
                    [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                ]
            ),
        )
    else:
        raise ValueError(f"Unknown config: {config}")


def test_head_named_params(model):
    parameter_shapes = {name: tuple(param.shape) for name, param in model.head_named_params()}
    assert parameter_shapes == {
        "pointer_head.encoder_mlp.0.bias": (24,),
        "pointer_head.encoder_mlp.0.weight": (24, 24),
        "pointer_head.encoder_mlp.3.bias": (24,),
        "pointer_head.encoder_mlp.3.weight": (24, 24),
    }


def test_encoder_only_named_params(model):
    parameter_shapes = {
        name: tuple(param.shape) for name, param in model.encoder_only_named_params()
    }
    assert len(parameter_shapes) == 35
    assert parameter_shapes == {
        "model.encoder.embed_positions.weight": (1026, 24),
        "model.encoder.layernorm_embedding.bias": (24,),
        "model.encoder.layernorm_embedding.weight": (24,),
        "model.encoder.layers.0.fc1.bias": (16,),
        "model.encoder.layers.0.fc1.weight": (16, 24),
        "model.encoder.layers.0.fc2.bias": (24,),
        "model.encoder.layers.0.fc2.weight": (24, 16),
        "model.encoder.layers.0.final_layer_norm.bias": (24,),
        "model.encoder.layers.0.final_layer_norm.weight": (24,),
        "model.encoder.layers.0.self_attn.k_proj.bias": (24,),
        "model.encoder.layers.0.self_attn.k_proj.weight": (24, 24),
        "model.encoder.layers.0.self_attn.out_proj.bias": (24,),
        "model.encoder.layers.0.self_attn.out_proj.weight": (24, 24),
        "model.encoder.layers.0.self_attn.q_proj.bias": (24,),
        "model.encoder.layers.0.self_attn.q_proj.weight": (24, 24),
        "model.encoder.layers.0.self_attn.v_proj.bias": (24,),
        "model.encoder.layers.0.self_attn.v_proj.weight": (24, 24),
        "model.encoder.layers.0.self_attn_layer_norm.bias": (24,),
        "model.encoder.layers.0.self_attn_layer_norm.weight": (24,),
        "model.encoder.layers.1.fc1.bias": (16,),
        "model.encoder.layers.1.fc1.weight": (16, 24),
        "model.encoder.layers.1.fc2.bias": (24,),
        "model.encoder.layers.1.fc2.weight": (24, 16),
        "model.encoder.layers.1.final_layer_norm.bias": (24,),
        "model.encoder.layers.1.final_layer_norm.weight": (24,),
        "model.encoder.layers.1.self_attn.k_proj.bias": (24,),
        "model.encoder.layers.1.self_attn.k_proj.weight": (24, 24),
        "model.encoder.layers.1.self_attn.out_proj.bias": (24,),
        "model.encoder.layers.1.self_attn.out_proj.weight": (24, 24),
        "model.encoder.layers.1.self_attn.q_proj.bias": (24,),
        "model.encoder.layers.1.self_attn.q_proj.weight": (24, 24),
        "model.encoder.layers.1.self_attn.v_proj.bias": (24,),
        "model.encoder.layers.1.self_attn.v_proj.weight": (24, 24),
        "model.encoder.layers.1.self_attn_layer_norm.bias": (24,),
        "model.encoder.layers.1.self_attn_layer_norm.weight": (24,),
    }


def test_decoder_only_named_params(model):
    parameter_shapes = {
        name: tuple(param.shape) for name, param in model.decoder_only_named_params()
    }
    assert len(parameter_shapes) == 55
    assert parameter_shapes == {
        "model.decoder.embed_positions.weight": (1026, 24),
        "model.decoder.layernorm_embedding.bias": (24,),
        "model.decoder.layernorm_embedding.weight": (24,),
        "model.decoder.layers.0.encoder_attn.k_proj.bias": (24,),
        "model.decoder.layers.0.encoder_attn.k_proj.weight": (24, 24),
        "model.decoder.layers.0.encoder_attn.out_proj.bias": (24,),
        "model.decoder.layers.0.encoder_attn.out_proj.weight": (24, 24),
        "model.decoder.layers.0.encoder_attn.q_proj.bias": (24,),
        "model.decoder.layers.0.encoder_attn.q_proj.weight": (24, 24),
        "model.decoder.layers.0.encoder_attn.v_proj.bias": (24,),
        "model.decoder.layers.0.encoder_attn.v_proj.weight": (24, 24),
        "model.decoder.layers.0.encoder_attn_layer_norm.bias": (24,),
        "model.decoder.layers.0.encoder_attn_layer_norm.weight": (24,),
        "model.decoder.layers.0.fc1.bias": (16,),
        "model.decoder.layers.0.fc1.weight": (16, 24),
        "model.decoder.layers.0.fc2.bias": (24,),
        "model.decoder.layers.0.fc2.weight": (24, 16),
        "model.decoder.layers.0.final_layer_norm.bias": (24,),
        "model.decoder.layers.0.final_layer_norm.weight": (24,),
        "model.decoder.layers.0.self_attn.k_proj.bias": (24,),
        "model.decoder.layers.0.self_attn.k_proj.weight": (24, 24),
        "model.decoder.layers.0.self_attn.out_proj.bias": (24,),
        "model.decoder.layers.0.self_attn.out_proj.weight": (24, 24),
        "model.decoder.layers.0.self_attn.q_proj.bias": (24,),
        "model.decoder.layers.0.self_attn.q_proj.weight": (24, 24),
        "model.decoder.layers.0.self_attn.v_proj.bias": (24,),
        "model.decoder.layers.0.self_attn.v_proj.weight": (24, 24),
        "model.decoder.layers.0.self_attn_layer_norm.bias": (24,),
        "model.decoder.layers.0.self_attn_layer_norm.weight": (24,),
        "model.decoder.layers.1.encoder_attn.k_proj.bias": (24,),
        "model.decoder.layers.1.encoder_attn.k_proj.weight": (24, 24),
        "model.decoder.layers.1.encoder_attn.out_proj.bias": (24,),
        "model.decoder.layers.1.encoder_attn.out_proj.weight": (24, 24),
        "model.decoder.layers.1.encoder_attn.q_proj.bias": (24,),
        "model.decoder.layers.1.encoder_attn.q_proj.weight": (24, 24),
        "model.decoder.layers.1.encoder_attn.v_proj.bias": (24,),
        "model.decoder.layers.1.encoder_attn.v_proj.weight": (24, 24),
        "model.decoder.layers.1.encoder_attn_layer_norm.bias": (24,),
        "model.decoder.layers.1.encoder_attn_layer_norm.weight": (24,),
        "model.decoder.layers.1.fc1.bias": (16,),
        "model.decoder.layers.1.fc1.weight": (16, 24),
        "model.decoder.layers.1.fc2.bias": (24,),
        "model.decoder.layers.1.fc2.weight": (24, 16),
        "model.decoder.layers.1.final_layer_norm.bias": (24,),
        "model.decoder.layers.1.final_layer_norm.weight": (24,),
        "model.decoder.layers.1.self_attn.k_proj.bias": (24,),
        "model.decoder.layers.1.self_attn.k_proj.weight": (24, 24),
        "model.decoder.layers.1.self_attn.out_proj.bias": (24,),
        "model.decoder.layers.1.self_attn.out_proj.weight": (24, 24),
        "model.decoder.layers.1.self_attn.q_proj.bias": (24,),
        "model.decoder.layers.1.self_attn.q_proj.weight": (24, 24),
        "model.decoder.layers.1.self_attn.v_proj.bias": (24,),
        "model.decoder.layers.1.self_attn.v_proj.weight": (24, 24),
        "model.decoder.layers.1.self_attn_layer_norm.bias": (24,),
        "model.decoder.layers.1.self_attn_layer_norm.weight": (24,),
    }


def test_encoder_decoder_shared_named_params(model):
    parameter_shapes = {
        name: tuple(param.shape) for name, param in model.encoder_decoder_shared_named_params()
    }
    assert len(parameter_shapes) == 1
    assert parameter_shapes == {"model.shared.weight": (50270, 24)}


def test_base_model_named_params(model):
    parameter_shapes = {
        name: tuple(param.shape) for name, param in model.base_model_named_params()
    }
    assert len(parameter_shapes) == 91
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
    assert len(all_optimized_parameters) > 0
    # check that all model parameters are covered
    all_model_parameters = {param for name, param in model.named_parameters()}
    assert all_optimized_parameters == all_model_parameters


# note that this is only used for the tests below which are marked as slow
# and are primarily meant to show how beam search works
@pytest.fixture(scope="module")
def pretrained_model() -> BartAsPointerNetwork:
    torch.random.manual_seed(42)
    model = BartAsPointerNetwork.from_pretrained(
        "sshleifer/distilbart-xsum-12-1",
        # label id space
        bos_token_id=0,  # taskmodule.bos_id,
        eos_token_id=1,  # taskmodule.eos_id,
        pad_token_id=1,  # taskmodule.eos_id,
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
        decoder_position_id_mode="pattern",
        decoder_position_id_pattern=[0, 0, 1, 0, 0, 1, 1],
    )

    return model


ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)


@pytest.mark.slow
def test_bart_pointer_network_beam_search(pretrained_model, taskmodule):
    model = pretrained_model
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
    torch.manual_seed(42)
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

    torch.manual_seed(42)
    outputs = model.beam_search(
        decoder_input_ids,
        beam_scorer,
        logits_processor=logits_processor,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        max_length=20,
        **model_kwargs,
    )

    torch.testing.assert_close(
        outputs,
        torch.tensor(
            [[0, 10, 30, 44, 45, 15, 16, 17, 33, 15, 16, 17, 33, 33, 33, 35, 33, 35, 33, 35]]
        ),
    )

    # result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # assert result == [
    #    " power lines in California have been shut down after a power provider said it was due to high winds."
    # ]


@pytest.mark.slow
def test_bart_pointer_network_generate_with_scores(pretrained_model, taskmodule):
    model = pretrained_model
    encoder_input_str = ARTICLE_TO_SUMMARIZE  # "translate English to German: How old are you?"
    inputs = taskmodule.tokenizer(encoder_input_str, max_length=1024, return_tensors="pt")

    torch.manual_seed(42)
    outputs = model.generate(
        inputs["input_ids"],
        num_beams=3,
        min_length=5,
        max_length=20,
        return_dict_in_generate=True,
        output_scores=True,
    )
    assert isinstance(outputs, BeamSearchEncoderDecoderOutput)
    torch.testing.assert_close(outputs.sequences_scores, torch.tensor([-7.126733303070068]))
    torch.testing.assert_close(
        outputs.sequences,
        torch.tensor(
            [[0, 10, 30, 44, 45, 15, 16, 17, 33, 15, 16, 35, 33, 15, 35, 41, 35, 33, 33, 33]]
        ),
    )

    # result = tokenizer.batch_decode(
    #    summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # assert result == [" power lines in California have been shut down on Friday."]
