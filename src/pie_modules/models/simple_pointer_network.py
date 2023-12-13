import copy
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BartConfig, BartModel, BartPreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bart.modeling_bart import BartDecoder, shift_tokens_right
from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)

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


class BartAsPointerConfig(BartConfig):
    def __init__(
        self,
        # label space ids (note that all target ids are: [bos_id, eos_id] + label_ids)
        label_ids: Optional[List[int]] = None,
        # mapping from label space ids to target token ids
        target_token_ids: Optional[List[int]] = None,
        # other parameters
        use_encoder_mlp: bool = True,
        max_target_positions: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # we use the bos_id as the decoder_start_token_id
        self.decoder_start_token_id = self.bos_token_id

        self.label_ids = label_ids
        self.target_token_ids = target_token_ids

        self.use_encoder_mlp = use_encoder_mlp
        self.max_target_positions = max_target_positions


class BartAsPointerNetwork(BartPreTrainedModel):
    config_class = BartAsPointerConfig
    base_model_prefix = "model"
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "lm_head.weight",
    ]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config: BartAsPointerConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )
        # TODO: remove and adjust get_output_embeddings()?
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # TODO: add content to here
        self.pointer_head = PointerHead(
            decoder=self.model.decoder,
            target_token_ids=self.model.config.target_token_ids,
            label_ids=self.model.config.label_ids,
            eos_id=self.model.config.eos_token_id,
            pad_id=self.model.config.pad_token_id,
            use_encoder_mlp=self.model.config.use_encoder_mlp,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()
        # TODO: return self.pointer_decoder?
        # return self.pointer_decoder

    def resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros(
                (1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device
            )
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.model.config.pad_token_id, self.model.config.decoder_start_token_id
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.model.config.use_cache
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # this contains the new input_ids
        modified_decoder_input_ids = self.pointer_head.prepare_decoder_input_ids(
            input_ids=decoder_input_ids,
            encoder_input_ids=input_ids,
        )

        if not isinstance(encoder_outputs, (BaseModelOutput, tuple)):
            raise ValueError(
                f"Inconsistent encoder_outputs: either should be of type BaseModelOutput or tuple, but it is "
                f"'{type(encoder_outputs)}'."
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.model.decoder(
            input_ids=modified_decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        if not isinstance(decoder_outputs, BaseModelOutputWithPastAndCrossAttentions):
            raise ValueError(
                "Inconsistent output: The output of the model decoder should be of type "
                f"`Seq2SeqLMOutput`, but is of type `{type(decoder_outputs)}`."
            )
        if not isinstance(encoder_outputs, BaseModelOutput):
            raise ValueError(
                "Inconsistent output: The output of the model encoder should be of type "
                f"`BaseModelOutputWithPastAndCrossAttentions`, but is of type `{type(encoder_outputs)}`."
            )
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""Labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*): Labels
        for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100`
        are ignored (masked), the loss is only computed for the tokens with labels in `[0, ...,
        config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning(
                    "The `use_cache` argument is changed to `False` since `labels` is provided."
                )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model_forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not isinstance(outputs, Seq2SeqModelOutput):
            raise ValueError(
                "Inconsistent output: The output of the model forward should be of type "
                f"`Seq2SeqLMOutput`, but is of type `{type(outputs)}`."
            )
        lm_logits, masked_lm_loss = self.pointer_head(
            last_hidden_state=outputs.last_hidden_state,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_input_ids=input_ids,
            encoder_attention_mask=attention_mask,
            labels=labels,
        )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        encoder_input_ids,
        encoder_attention_mask,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": encoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        result = super()._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )
        # add items that are needed for pointer network
        result["encoder_input_ids"] = inputs_tensor
        result["encoder_attention_mask"] = result["attention_mask"]
        return result
