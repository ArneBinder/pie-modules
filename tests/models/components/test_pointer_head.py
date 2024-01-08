import pytest
import torch
from torch import nn

from pie_modules.models.components.pointer_head import PointerHead


def get_pointer_head(**kwargs):
    torch.manual_seed(42)
    return PointerHead(
        embeddings=nn.Embedding(10, 20),
        # no continuous label ids
        label_ids=[3, 5, 4],
        eos_id=1,
        pad_id=2,
        # bos, eos, pad, 3 x label ids
        target_token_ids=[1, 2, 3, 6, 7, 8],
        embedding_weight_mapping={
            "5": [3, 4],
            "6": [2],
        },
        use_encoder_mlp=True,
        use_constraints_encoder_mlp=True,
        decoder_position_id_pattern=[0, 1, 1, 2],
        # increase_position_ids_per_record=True,
        **kwargs
    )


def test_get_pointer_head():
    pointer_head = get_pointer_head()
    assert pointer_head is not None
    assert pointer_head.use_prepared_position_ids


def test_set_embeddings():
    pointer_head = get_pointer_head()
    original_embeddings = pointer_head.embeddings
    new_embeddings = nn.Embedding(10, 20)
    pointer_head.set_embeddings(new_embeddings)
    assert pointer_head.embeddings is not None
    assert pointer_head.embeddings != original_embeddings
    assert pointer_head.embeddings == new_embeddings


def test_overwrite_embeddings_with_mapping():
    pointer_head = get_pointer_head()
    original_embeddings_weight = pointer_head.embeddings.weight.clone()
    pointer_head.overwrite_embeddings_with_mapping()
    assert pointer_head.embeddings is not None
    assert not torch.equal(pointer_head.embeddings.weight, original_embeddings_weight)
    torch.testing.assert_allclose(
        pointer_head.embeddings.weight[5], original_embeddings_weight[[3, 4]].mean(dim=0)
    )
    torch.testing.assert_allclose(
        pointer_head.embeddings.weight[6], original_embeddings_weight[[2]].mean(dim=0)
    )


@pytest.mark.parametrize(
    "use_attention_mask",
    [True, False],
)
def test_prepare_decoder_input_ids(use_attention_mask):
    pointer_head = get_pointer_head()
    encoder_input_ids = torch.tensor(
        [
            [100, 101, 102, 103, 104, 105],
            [200, 201, 202, 203, 0, 0],
        ]
    ).to(torch.long)
    input_ids = torch.tensor(
        [
            [0, 3, 4, 5, 6, 7],
            [0, 3, 9, 1, 2, 2],
        ]
    ).to(torch.long)
    attention_mask = (
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0],
            ]
        ).to(torch.long)
        if use_attention_mask
        else None
    )
    # to recap, the target2token_id mapping is (bos, eos, pad, 3 x label ids)
    torch.testing.assert_allclose(pointer_head.target2token_id, torch.tensor([1, 2, 3, 6, 7, 8]))
    # 3 labels + bos / pad
    assert pointer_head.pointer_offset == 6

    prepared_decoder_input_ids = pointer_head.prepare_decoder_input_ids(
        input_ids=input_ids, encoder_input_ids=encoder_input_ids, attention_mask=attention_mask
    )
    assert prepared_decoder_input_ids is not None
    assert prepared_decoder_input_ids.shape == input_ids.shape
    assert prepared_decoder_input_ids.tolist() == [[1, 6, 7, 8, 100, 101], [1, 6, 203, 2, 3, 3]]


def test_prepare_decoder_input_ids_out_of_bounds():
    pointer_head = get_pointer_head()
    # 3 labels + bos / pad
    assert pointer_head.pointer_offset == 6
    encoder_input_ids = torch.tensor(
        [
            [100, 101, 102],
        ]
    ).to(torch.long)
    input_ids = torch.tensor(
        [
            # 9 is out of bounds: > pointer_head.pointer_offset + len(encoder_input_ids)
            [0, 9],
        ]
    ).to(torch.long)

    with pytest.raises(ValueError) as excinfo:
        pointer_head.prepare_decoder_input_ids(
            input_ids=input_ids, encoder_input_ids=encoder_input_ids
        )
    assert str(excinfo.value) == (
        "encoder_input_ids_index.max() [3] must be smaller than encoder_input_length [3]!"
    )


@pytest.mark.parametrize(
    "use_attention_mask,increase_position_ids_per_record",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_prepare_decoder_position_ids(use_attention_mask, increase_position_ids_per_record):
    pointer_head = get_pointer_head(
        increase_position_ids_per_record=increase_position_ids_per_record
    )
    input_ids = torch.tensor(
        [
            [0, 3, 4, 5, 6, 7],
            [0, 3, 9, 1, 2, 2],
        ]
    ).to(torch.long)
    attention_mask = (
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0],
            ]
        ).to(torch.long)
        if use_attention_mask
        else None
    )
    # to recap, the target2token_id mapping is (bos, eos, pad, 3 x label ids)
    torch.testing.assert_allclose(pointer_head.target2token_id, torch.tensor([1, 2, 3, 6, 7, 8]))
    # 3 labels + bos / pad
    assert pointer_head.pointer_offset == 6

    prepared_decoder_position_ids = pointer_head.prepare_decoder_position_ids(
        input_ids=input_ids, attention_mask=attention_mask
    )
    assert prepared_decoder_position_ids is not None
    assert prepared_decoder_position_ids.shape == input_ids.shape
    if not increase_position_ids_per_record:
        if use_attention_mask:
            assert prepared_decoder_position_ids.tolist() == [
                [0, 2, 3, 3, 4, 2],
                [0, 2, 3, 3, 1, 1],
            ]
        else:
            assert prepared_decoder_position_ids.tolist() == [
                [0, 2, 3, 3, 4, 2],
                [0, 2, 3, 3, 4, 2],
            ]
    else:
        # the ids (except for bos and pad) get increased by 3 per record (which has length 4)
        if use_attention_mask:
            assert prepared_decoder_position_ids.tolist() == [
                [0, 2, 3, 3, 4, 5],
                [0, 2, 3, 3, 1, 1],
            ]
        else:
            assert prepared_decoder_position_ids.tolist() == [
                [0, 2, 3, 3, 4, 5],
                [0, 2, 3, 3, 4, 5],
            ]


def test_prepare_decoder_inputs():
    pointer_head = get_pointer_head()
    encoder_input_ids = torch.tensor(
        [
            [100, 101, 102, 103, 104, 105],
            [200, 201, 202, 203, 0, 0],
        ]
    ).to(torch.long)
    input_ids = torch.tensor(
        [
            [0, 3, 4, 5, 6, 7],
            [0, 3, 9, 1, 2, 2],
        ]
    ).to(torch.long)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ]
    ).to(torch.long)
    # to recap, the target2token_id mapping is (bos, eos, pad, 3 x label ids)
    torch.testing.assert_allclose(pointer_head.target2token_id, torch.tensor([1, 2, 3, 6, 7, 8]))
    # 3 labels + bos / pad
    assert pointer_head.pointer_offset == 6

    decoder_inputs = pointer_head.prepare_decoder_inputs(
        input_ids=input_ids, encoder_input_ids=encoder_input_ids, attention_mask=attention_mask
    )
    assert set(decoder_inputs) == {"input_ids", "position_ids", "attention_mask"}
    assert decoder_inputs["input_ids"] is not None
    assert decoder_inputs["position_ids"] is not None
    assert decoder_inputs["input_ids"].shape == input_ids.shape
    assert decoder_inputs["position_ids"].shape == input_ids.shape
    assert decoder_inputs["input_ids"].tolist() == [[1, 6, 7, 8, 100, 101], [1, 6, 203, 2, 3, 3]]
    assert decoder_inputs["position_ids"].tolist() == [[0, 2, 3, 3, 4, 2], [0, 2, 3, 3, 1, 1]]
