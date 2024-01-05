import torch

from pie_modules.models.base_models.bart_with_decoder_position_ids import (
    BartLearnedPositionalEmbeddingWithPositionIds,
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
