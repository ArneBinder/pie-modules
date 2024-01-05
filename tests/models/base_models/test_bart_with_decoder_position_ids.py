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


def test_bart_model_with_decoder_position_forward(bart_model_with_decoder_position_ids):
    # Arrange
    model = bart_model_with_decoder_position_ids
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    position_ids_original = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    position_ids_different = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]])

    # Act
    torch.manual_seed(42)
    original = model(input_ids=input_ids)
    torch.manual_seed(42)
    replaced_original = model(input_ids=input_ids, decoder_position_ids=position_ids_original)
    torch.manual_seed(42)
    replaced_different = model(input_ids=input_ids, decoder_position_ids=position_ids_different)

    # Assert
    assert isinstance(original, Seq2SeqModelOutput)
    assert original.last_hidden_state.shape == (1, 8, 10)
    assert isinstance(replaced_original, Seq2SeqModelOutput)
    torch.testing.assert_close(original.last_hidden_state, replaced_original.last_hidden_state)

    assert isinstance(replaced_different, Seq2SeqModelOutput)
    assert replaced_different.last_hidden_state.shape == (1, 8, 10)
    assert not torch.allclose(original.last_hidden_state, replaced_different.last_hidden_state)
