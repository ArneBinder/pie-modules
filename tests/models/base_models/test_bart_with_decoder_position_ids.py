import pytest
import torch
from torch.nn import Embedding
from transformers import BartConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput,
)
from transformers.models.bart.modeling_bart import (
    BartEncoder,
    BartLearnedPositionalEmbedding,
)

from pie_modules.models.base_models.bart_with_decoder_position_ids import (
    BartDecoderWithPositionIds,
    BartLearnedPositionalEmbeddingWithPositionIds,
    BartModelWithDecoderPositionIds,
)


def test_bart_learned_positional_embedding_with_position_ids():
    # Arrange
    torch.manual_seed(42)
    model = BartLearnedPositionalEmbeddingWithPositionIds(10, 6)
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    position_ids_original = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    position_ids_different = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]])

    # Act
    original = model(input_ids=input_ids)
    replaced_original = model(input_ids=input_ids, position_ids=position_ids_original)
    replaced_different = model(input_ids=input_ids, position_ids=position_ids_different)

    # Assert
    assert original.shape == (1, 10, 6)
    assert replaced_original.shape == (1, 10, 6)
    torch.testing.assert_close(original, replaced_original)
    assert replaced_different.shape == (1, 10, 6)
    assert not torch.allclose(original, replaced_different)


@pytest.fixture(scope="module")
def bart_config():
    return BartConfig(
        vocab_size=30,
        d_model=10,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=20,
        decoder_ffn_dim=20,
        max_position_embeddings=10,
    )


@pytest.fixture(scope="module")
def bart_decoder_with_position_ids(bart_config):
    return BartDecoderWithPositionIds(config=bart_config)


def test_bart_decoder_with_position_ids(bart_decoder_with_position_ids):
    assert bart_decoder_with_position_ids is not None


def test_bart_decoder_with_position_ids_get_input_embeddings(bart_decoder_with_position_ids):
    input_embeddings = bart_decoder_with_position_ids.get_input_embeddings()
    assert input_embeddings is not None
    assert isinstance(input_embeddings, Embedding)
    assert input_embeddings.embedding_dim == 10
    assert input_embeddings.num_embeddings == 30


def test_bart_decoder_with_position_ids_set_input_embeddings(bart_decoder_with_position_ids):
    original_input_embeddings = bart_decoder_with_position_ids.get_input_embeddings()
    torch.manual_seed(42)
    new_input_embeddings = Embedding(
        original_input_embeddings.num_embeddings, original_input_embeddings.embedding_dim
    )
    bart_decoder_with_position_ids.set_input_embeddings(new_input_embeddings)
    input_embeddings = bart_decoder_with_position_ids.get_input_embeddings()
    assert input_embeddings == new_input_embeddings
    assert input_embeddings is not original_input_embeddings
    # recover original input embeddings
    bart_decoder_with_position_ids.set_input_embeddings(original_input_embeddings)


def test_bart_decoder_with_position_ids_forward(bart_decoder_with_position_ids):
    # Arrange
    model = bart_decoder_with_position_ids
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    position_ids_original = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    position_ids_different = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]])

    # Act
    torch.manual_seed(42)
    original = model(input_ids=input_ids)
    torch.manual_seed(42)
    replaced_original = model(input_ids=input_ids, position_ids=position_ids_original)
    torch.manual_seed(42)
    replaced_different = model(input_ids=input_ids, position_ids=position_ids_different)

    # Assert
    assert isinstance(original, BaseModelOutputWithPastAndCrossAttentions)
    assert original.last_hidden_state.shape == (1, 8, 10)
    assert isinstance(replaced_original, BaseModelOutputWithPastAndCrossAttentions)
    torch.testing.assert_close(original.last_hidden_state, replaced_original.last_hidden_state)

    assert isinstance(replaced_different, BaseModelOutputWithPastAndCrossAttentions)
    assert replaced_different.last_hidden_state.shape == (1, 8, 10)
    assert not torch.allclose(original.last_hidden_state, replaced_different.last_hidden_state)


@pytest.fixture(scope="module")
def bart_model_with_decoder_position_ids(bart_config):
    torch.manual_seed(42)
    return BartModelWithDecoderPositionIds(config=bart_config)


def test_bart_model_with_decoder_position_ids(bart_model_with_decoder_position_ids):
    assert bart_model_with_decoder_position_ids is not None


def test_bart_model_with_decoder_position_ids_get_input_embeddings(
    bart_model_with_decoder_position_ids,
):
    input_embeddings = bart_model_with_decoder_position_ids.get_input_embeddings()
    assert input_embeddings is not None
    assert isinstance(input_embeddings, Embedding)
    assert input_embeddings.embedding_dim == 10
    assert input_embeddings.num_embeddings == 30


def test_bart_model_with_decoder_position_ids_set_input_embeddings(
    bart_model_with_decoder_position_ids,
):
    original_input_embeddings = bart_model_with_decoder_position_ids.get_input_embeddings()
    torch.manual_seed(42)
    new_input_embeddings = Embedding(
        original_input_embeddings.num_embeddings, original_input_embeddings.embedding_dim
    )
    bart_model_with_decoder_position_ids.set_input_embeddings(new_input_embeddings)
    input_embeddings = bart_model_with_decoder_position_ids.get_input_embeddings()
    assert input_embeddings == new_input_embeddings
    assert input_embeddings is not original_input_embeddings
    # recover original input embeddings
    bart_model_with_decoder_position_ids.set_input_embeddings(original_input_embeddings)


def test_bart_model_with_decoder_position_ids_get_encoder(bart_model_with_decoder_position_ids):
    encoder = bart_model_with_decoder_position_ids.get_encoder()
    assert encoder is not None
    assert isinstance(encoder, BartEncoder)


def test_bart_model_with_decoder_position_ids_get_decoder(bart_model_with_decoder_position_ids):
    decoder = bart_model_with_decoder_position_ids.get_decoder()
    assert decoder is not None
    assert isinstance(decoder, BartDecoderWithPositionIds)


@pytest.mark.parametrize(
    "return_dict",
    [True, False],
)
def test_bart_model_with_decoder_position_forward(
    bart_model_with_decoder_position_ids, return_dict
):
    # Arrange
    model = bart_model_with_decoder_position_ids
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    position_ids_original = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    position_ids_different = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]])

    # Act
    torch.manual_seed(42)
    original = model(input_ids=input_ids, return_dict=return_dict)[0]
    torch.manual_seed(42)
    replaced_original = model(
        input_ids=input_ids, decoder_position_ids=position_ids_original, return_dict=return_dict
    )[0]
    torch.manual_seed(42)
    replaced_different = model(
        input_ids=input_ids, decoder_position_ids=position_ids_different, return_dict=return_dict
    )[0]

    # Assert
    assert isinstance(original, torch.FloatTensor)
    assert original.shape == (1, 8, 10)
    torch.testing.assert_close(
        original[0, :5, :3],
        torch.tensor(
            [
                [0.7961970567703247, 1.2232387065887451, 0.7286717295646667],
                [0.034051503986120224, -0.9746682047843933, -0.700711190700531],
                [0.1363907903432846, -0.4540761113166809, -1.2949464321136475],
                [1.1136258840560913, -0.1388537585735321, 1.538393259048462],
                [-1.1127841472625732, 0.22768200933933258, 1.6438117027282715],
            ]
        ),
    )
    torch.testing.assert_close(
        original.sum(dim=-1),
        torch.tensor(
            [
                [
                    -2.384185791015625e-07,
                    -4.76837158203125e-07,
                    -2.682209014892578e-07,
                    2.086162567138672e-07,
                    5.960464477539063e-08,
                    5.960464477539063e-08,
                    0.0,
                    0.0,
                ]
            ]
        ),
    )
    assert isinstance(replaced_original, torch.FloatTensor)
    torch.testing.assert_close(original, replaced_original)

    assert isinstance(replaced_different, torch.FloatTensor)
    assert replaced_different.shape == (1, 8, 10)
    torch.testing.assert_close(
        replaced_different[0, :5, :3],
        torch.tensor(
            [
                [0.7961970567703247, 1.2232387065887451, 0.7286717295646667],
                [0.1183161735534668, -0.7555443048477173, -1.230163812637329],
                [1.2578136920928955, 0.18759475648403168, -0.1578090786933899],
                [0.5176712870597839, 0.9378399848937988, 1.3435578346252441],
                [0.6121589541435242, -1.0105386972427368, 2.361997365951538],
            ]
        ),
    )
    torch.testing.assert_close(
        replaced_different.sum(dim=-1),
        torch.tensor(
            [
                [
                    -2.384185791015625e-07,
                    -4.76837158203125e-07,
                    -2.682209014892578e-07,
                    2.086162567138672e-07,
                    5.960464477539063e-08,
                    5.960464477539063e-08,
                    0.0,
                    0.0,
                ]
            ]
        ),
    )
    assert not torch.allclose(replaced_different, original)
