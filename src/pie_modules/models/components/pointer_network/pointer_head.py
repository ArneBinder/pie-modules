from typing import Dict, List, Optional

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
        decoder_position_id_pattern: Optional[List[int]] = None,
        increase_position_ids_per_record: bool = False,
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

        self.position_id_pattern = decoder_position_id_pattern
        if decoder_position_id_pattern is not None:
            self.register_buffer(
                "decoder_position_id_pattern", torch.tensor(decoder_position_id_pattern)
            )
        self.increase_position_ids_per_record = increase_position_ids_per_record

    @property
    def has_position_id_pattern(self):
        return hasattr(self, "decoder_position_id_pattern")

    def output_size(self):
        return len(self.target_token_ids)

    def prepare_decoder_input_ids(
        self,
        input_ids: torch.LongTensor,
        encoder_input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
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
        # cumsum = input_ids.eq(self.pad_id).flip(dims=[1]).cumsum(dim=-1)
        # tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])
        # decoder_input_ids = decoder_input_ids.masked_fill(tgt_pad_mask, self.pad_token_id)

        # during training, the attention mask available
        if attention_mask is not None:
            decoder_input_ids = decoder_input_ids.masked_fill(
                ~attention_mask.bool(), self.pad_token_id
            )

        # TODO: why was this in the original code?
        # decoder_input_ids = decoder_input_ids[:, :-1]

        return decoder_input_ids

    def prepare_decoder_position_ids(
        self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None
    ):
        bsz, tokens_len = input_ids.size()
        pattern_len = len(self.decoder_position_id_pattern)
        # the number of full and partly records. note that tokens_len includes the bos token
        repeat_num = (tokens_len - 2) // pattern_len + 1
        position_ids = self.decoder_position_id_pattern.repeat(bsz, repeat_num)

        if self.increase_position_ids_per_record:
            # TODO: check this
            reshape_pos = position_ids.view(bsz, -1, pattern_len)
            shift_pos = reshape_pos.size(1)  # TODO: isn't this the same as repeat_num?
            add_shift_pos = (
                torch.range(0, shift_pos - 1, device=reshape_pos.device)
                .repeat(bsz)
                .view(bsz, -1)
                .unsqueeze(-1)
            )
            # TODO: add_shift_pos *= max(self.decoder_position_id_pattern) + 1?
            reshape_pos = add_shift_pos + reshape_pos
            position_ids = reshape_pos.view(bsz, -1).long()
        # use start_position_id=0
        start_pos = torch.zeros(bsz, 1, dtype=position_ids.dtype, device=position_ids.device)
        # shift by 2 to account for start_position_id=0 and pad_position_id=1
        all_position_ids = torch.cat([start_pos, position_ids + 2], dim=-1)
        all_position_ids_truncated = all_position_ids[:bsz, :tokens_len]

        # during training, the attention mask is not None
        if attention_mask is not None:
            # pad with pad_position_id=1
            return all_position_ids_truncated.masked_fill(~attention_mask.bool(), 1)
        else:
            return all_position_ids_truncated

    def overwrite_decoder_label_embeddings_with_mapping(
        self, label_embedding_mapping: Dict[int, List[int]], encoder_weights: torch.Tensor
    ):
        """Overwrite the decoder label embeddings with embeddings from an encoder. This is useful
        if the label vocabulary is a subset of the source vocabulary. In this case, the embeddings
        of the label tokens will be initialized with the average of the embeddings of the source
        tokens.

        :param label_embedding_mapping: a mapping from label token ids to source token ids
        :param encoder_weights: the encoder weights
        :return: None
        """

        if label_embedding_mapping is None:
            raise ValueError("No label_embedding_mapping provided!")
        for special_token_index, source_indices in label_embedding_mapping.items():
            embed = encoder_weights.data[source_indices[0]]
            for i in source_indices[1:]:
                embed += self.decoder.embed_tokens.weight.data[i]
            embed /= len(source_indices)
            self.decoder.embed_tokens.weight.data[special_token_index] = embed

    def decoder_forward(
        self,
        input_ids: torch.LongTensor,
        encoder_input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        modified_decoder_input_ids = self.prepare_decoder_input_ids(
            input_ids=input_ids,
            encoder_input_ids=encoder_input_ids,
            attention_mask=attention_mask,
        )
        if self.has_position_id_pattern:
            kwargs["position_ids"] = self.prepare_decoder_position_ids(
                input_ids=modified_decoder_input_ids,
                attention_mask=attention_mask,
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
        decoder_attention_mask: Optional[torch.LongTensor] = None,
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
            loss_fct = CrossEntropyLoss()
            logits_resized = logits.reshape(-1, logits.size(-1))
            labels_resized = labels.reshape(-1)
            if decoder_attention_mask is None:
                raise ValueError("decoder_attention_mask must be provided to compute the loss!")
            mask_resized = decoder_attention_mask.reshape(-1)
            labels_masked = labels_resized.masked_fill(
                ~mask_resized.to(torch.bool), loss_fct.ignore_index
            )
            loss = loss_fct(logits_resized, labels_masked)

        return logits, loss
