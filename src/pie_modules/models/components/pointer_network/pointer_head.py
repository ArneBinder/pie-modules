from typing import List, Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import BartDecoder
from transformers.utils import logging

logger = logging.get_logger(__name__)


class PointerHead(torch.nn.Module):
    # Copy and generate,
    def __init__(
        self,
        decoder,
        # label space
        label_ids: List[int],
        eos_id: int,
        pad_id: int,
        # target token space
        target_token_ids: List[int],
        # other parameters
        use_encoder_mlp: bool = False,
    ):
        super().__init__()

        if not isinstance(decoder, BartDecoder):
            raise ValueError("PointerHead only works with BartDecoder!")
        self.decoder = decoder

        self.pad_id = pad_id
        self.eos_id = eos_id

        target2token_id = torch.LongTensor(target_token_ids)
        self.register_buffer("target2token_id", target2token_id)
        self.label_token_ids = [target_token_ids[label_id] for label_id in label_ids]

        self.pointer_offset = len(target2token_id)

        if self.eos_id >= self.pointer_offset:
            raise ValueError(
                f"eos_id [{self.eos_id}] must be smaller than pointer_offset [{self.pointer_offset}]!"
            )
        self.eos_token_id = target_token_ids[self.eos_id]
        if self.pad_id >= self.pointer_offset:
            raise ValueError(
                f"pad_id [{self.pad_id}] must be smaller than pointer_offset [{self.pointer_offset}]!"
            )
        self.pad_token_id = target_token_ids[self.pad_id]

        hidden_size = self.decoder.embed_tokens.weight.size(1)
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )

    def output_size(self):
        return len(self.target_token_ids)

    def prepare_decoder_input_ids(
        self,
        input_ids: torch.LongTensor,
        encoder_input_ids: torch.LongTensor,
    ):
        mapping_token_mask = input_ids.lt(self.pointer_offset)
        mapped_tokens = input_ids.masked_fill(input_ids.ge(self.pointer_offset), 0)
        tag_mapped_tokens = self.target2token_id[mapped_tokens]

        encoder_input_ids_index = input_ids - self.pointer_offset
        encoder_input_ids_index = encoder_input_ids_index.masked_fill(
            encoder_input_ids_index.lt(0), 0
        )
        # TODO: parametrize this (use max input length)
        assert encoder_input_ids_index.max() < 1024
        word_mapped_tokens = encoder_input_ids.gather(index=encoder_input_ids_index, dim=1)

        decoder_input_ids = torch.where(
            mapping_token_mask, tag_mapped_tokens, word_mapped_tokens
        )  # bsz x max_len
        # attention_mask = input_ids.ne(self.pad_token_id)  # inverted tgt_pad_mask?
        cumsum = input_ids.eq(self.pad_id).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])
        decoder_input_ids = decoder_input_ids.masked_fill(tgt_pad_mask, self.pad_token_id)

        # TODO: why was this in the original code?
        # decoder_input_ids = decoder_input_ids[:, :-1]

        return decoder_input_ids

    def decoder_forward(self, input_ids, encoder_input_ids, **kwargs):
        modified_decoder_input_ids = self.prepare_decoder_input_ids(
            input_ids=input_ids,
            encoder_input_ids=encoder_input_ids,
        )

        decoder_outputs = self.decoder(input_ids=modified_decoder_input_ids, **kwargs)
        return decoder_outputs

    def forward(
        self,
        last_hidden_state,
        encoder_last_hidden_state,
        encoder_input_ids,
        encoder_attention_mask,
        labels: Optional[torch.LongTensor] = None,
    ):
        # assemble the logits
        logits = last_hidden_state.new_full(
            (
                last_hidden_state.size(0),
                last_hidden_state.size(1),
                self.pointer_offset + encoder_input_ids.size(-1),
            ),
            fill_value=-1e24,
        )

        # eos and tag scores depend only on the decoder output
        eos_scores = F.linear(
            last_hidden_state,
            self.decoder.embed_tokens.weight[[self.eos_token_id]],
        )  # bsz x max_len x 1
        label_scores = F.linear(
            last_hidden_state, self.decoder.embed_tokens.weight[self.label_token_ids]
        )  # bsz x max_len x num_class

        # the pointer depends on the src token embeddings, the encoder output and the decoder output
        # bsz x max_bpe_len x hidden_size
        # src_outputs = state.encoder_output
        src_outputs = encoder_last_hidden_state
        if hasattr(self, "encoder_mlp"):
            src_outputs = self.encoder_mlp(src_outputs)

        # mask = state.encoder_mask.eq(0)
        input_embed = self.decoder.embed_tokens(
            encoder_input_ids
        )  # bsz x max_word_len x hidden_size
        # bsz = encoder_input_ids.size(0)
        # position_embed = torch.stack(
        #    [self.encoder_embed_positions(encoder_input_ids)] * bsz, dim=0
        # )

        word_scores = torch.einsum(
            "blh,bnh->bln", last_hidden_state, src_outputs
        )  # bsz x max_len x max_word_len
        gen_scores = torch.einsum(
            "blh,bnh->bln", last_hidden_state, input_embed
        )  # bsz x max_len x max_word_len
        # positions_scores = torch.einsum(
        #    "blh,bnh->bln", hidden_state, position_embed
        # )  # bsz x max_len x max_word_len

        # if self.position_type == 9:
        #    avg_word_scores = (positions_scores + word_scores) / 2
        # elif self.position_type == 10:
        #    avg_word_scores = positions_scores
        # else:
        avg_word_scores = (gen_scores + word_scores) / 2
        # TODO: what exactly does this mask?
        mask = encoder_attention_mask.eq(0)
        mask = mask.unsqueeze(1)
        # TODO: what are 2 and 1?
        mask = mask.__or__(encoder_input_ids.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        avg_word_scores = avg_word_scores.masked_fill(mask, -1e32)
        # word_scores = word_scores.masked_fill(mask, -1e32)

        # Note: logits[:, :, 0] contains the score for the bos token which should be never generated!
        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2 : self.pointer_offset] = label_scores
        logits[:, :, self.pointer_offset :] = avg_word_scores

        loss = None
        if labels is not None:
            # labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            # TODO: any masking for the padding needed?
            logits_resized = logits.view(-1, logits.size(-1))
            labels_resized = labels.view(-1)
            loss = loss_fct(logits_resized, labels_resized)

        return logits, loss
