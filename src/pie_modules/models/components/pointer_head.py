from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
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
        use_constraints_encoder_mlp: bool = False,
        decoder_position_id_pattern: Optional[List[int]] = None,
        increase_position_ids_per_record: bool = False,
    ):
        super().__init__()

        # if not isinstance(decoder, BartDecoder):
        #    raise ValueError("PointerHead only works with BartDecoder!")
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
        if use_constraints_encoder_mlp:
            self.constraints_encoder_mlp = nn.Sequential(
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
        encoder_input_length = encoder_input_ids.size(1)
        if encoder_input_ids_index.max() >= encoder_input_length:
            raise ValueError(
                f"encoder_input_ids_index.max() [{encoder_input_ids_index.max()}] must be smaller "
                f"than encoder_input_length [{encoder_input_length}]!"
            )

        word_mapped_tokens = encoder_input_ids.gather(index=encoder_input_ids_index, dim=1)

        decoder_input_ids = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)

        # during training, the attention mask is available
        if attention_mask is not None:
            decoder_input_ids = decoder_input_ids.masked_fill(
                ~attention_mask.bool(), self.pad_token_id
            )

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
            position_ids_reshaped = position_ids.view(bsz, -1, pattern_len)
            add_shift_pos = (
                torch.range(0, repeat_num - 1, device=position_ids_reshaped.device)
                .repeat(bsz)
                .view(bsz, -1)
                .unsqueeze(-1)
            )
            # multiply by the highest position id in the pattern so that the position ids are unique
            # for any decoder_position_id_pattern across all records
            add_shift_pos *= max(self.decoder_position_id_pattern) + 1
            position_ids_reshaped = add_shift_pos + position_ids_reshaped
            position_ids = position_ids_reshaped.view(bsz, -1).long()
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
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if self.has_position_id_pattern:
            if position_ids is None:
                position_ids = self.prepare_decoder_position_ids(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            # shallow copy kwargs to avoid modifying the original dict
            kwargs = {**kwargs, "position_ids": position_ids}

        modified_decoder_input_ids = self.prepare_decoder_input_ids(
            input_ids=input_ids,
            encoder_input_ids=encoder_input_ids,
            attention_mask=attention_mask,
        )

        decoder_outputs = self.decoder(
            input_ids=modified_decoder_input_ids, attention_mask=attention_mask, **kwargs
        )

        return decoder_outputs

    def forward(
        self,
        last_hidden_state,
        encoder_last_hidden_state,
        encoder_input_ids,
        encoder_attention_mask,
        labels: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        constraints: Optional[torch.LongTensor] = None,
    ):
        decoder_embed_tokens = self.decoder.embed_tokens

        # assemble the logits
        logits = last_hidden_state.new_full(
            (
                last_hidden_state.size(0),
                last_hidden_state.size(1),
                self.pointer_offset + encoder_input_ids.size(-1),
            ),
            fill_value=-1e24,
        )

        # eos and label scores depend only on the decoder output
        # bsz x max_len x 1
        eos_scores = F.linear(last_hidden_state, decoder_embed_tokens.weight[[self.eos_token_id]])
        label_embeddings = decoder_embed_tokens.weight[self.label_token_ids]
        # bsz x max_len x num_class
        label_scores = F.linear(last_hidden_state, label_embeddings)

        # the pointer depends on the src token embeddings, the encoder output and the decoder output
        # bsz x max_bpe_len x hidden_size
        src_outputs = encoder_last_hidden_state
        if getattr(self, "encoder_mlp", None) is not None:
            src_outputs = self.encoder_mlp(src_outputs)

        # bsz x max_word_len x hidden_size
        input_embed = decoder_embed_tokens(encoder_input_ids)

        # bsz x max_len x max_word_len
        word_scores = torch.einsum("blh,bnh->bln", last_hidden_state, src_outputs)
        gen_scores = torch.einsum("blh,bnh->bln", last_hidden_state, input_embed)
        avg_word_scores = (gen_scores + word_scores) / 2

        # TODO: what exactly does this mask? Masking special tokens?
        mask = encoder_attention_mask.eq(0)
        mask = mask.unsqueeze(1)
        # TODO: what are 2 and 1 (for ge)? Note that 2 is the eos_token_id of Bart.
        mask = mask.__or__(encoder_input_ids.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        avg_word_scores = avg_word_scores.masked_fill(mask, -1e32)

        # Note: logits[:, :, 0] contains the score for the bos token which should be never generated!
        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2 : self.pointer_offset] = label_scores
        logits[:, :, self.pointer_offset :] = avg_word_scores

        loss = None
        # compute the loss if labels are provided
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

        # compute the constraints loss if constraints are provided
        if constraints is not None:
            if getattr(self, "constraints_encoder_mlp", None) is not None:
                # TODO: is it fine to apply constraints_encoder_mlp to both src_outputs and label_embeddings?
                #  This is what the original code seems to do, but this is different from the usage of encoder_mlp.
                constraints_src_outputs = self.constraints_encoder_mlp(src_outputs)
                constraints_label_embeddings = self.constraints_encoder_mlp(label_embeddings)
            else:
                constraints_src_outputs = src_outputs
                constraints_label_embeddings = label_embeddings
            constraints_label_scores = F.linear(last_hidden_state, constraints_label_embeddings)
            # bsz x max_len x max_word_len
            constraints_word_scores = torch.einsum(
                "blh,bnh->bln", last_hidden_state, constraints_src_outputs
            )
            constraints_logits = last_hidden_state.new_full(
                (
                    last_hidden_state.size(0),
                    last_hidden_state.size(1),
                    self.pointer_offset + encoder_input_ids.size(-1),
                ),
                fill_value=-1e24,
            )
            constraints_logits[:, :, 2 : self.pointer_offset] = constraints_label_scores
            constraints_logits[:, :, self.pointer_offset :] = constraints_word_scores

            mask = constraints >= 0
            constraints_logits_valid = constraints_logits[mask]
            constraints_valid = constraints[mask]
            loss_c = F.binary_cross_entropy(
                torch.sigmoid(constraints_logits_valid), constraints_valid.float()
            )

            if loss is None:
                loss = loss_c
            else:
                loss += loss_c

        return logits, loss
