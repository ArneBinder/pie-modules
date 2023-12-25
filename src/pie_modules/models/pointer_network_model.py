import copy
from collections.abc import MutableMapping
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_ie import TaskModule
from pytorch_ie.core import PyTorchIEModel
from torch import nn
from torch.nn import Parameter
from torchmetrics import Metric
from transformers import get_linear_schedule_with_warmup
from typing_extensions import TypeAlias

from ..taskmodules import PointerNetworkTaskModuleForEnd2EndRE
from ..taskmodules.common import HasBuildMetric
from .components.pointer_network.generator import SequenceGenerator
from .components.pointer_network.interface import Seq2SeqEncoder, State
from .components.pointer_network.losses import Seq2SeqLoss
from .components.pointer_network.modeling_bart import (
    BartDecoder,
    BartEncoder,
    BartModel,
    LearnedPositionalEmbedding,
)
from .components.pointer_network.utils import seq_len_to_mask

GmamModelStepBatchEncoding: TypeAlias = Tuple[Dict[str, Any], Dict[str, Any]]


def get_layernorm_parameters(named_parameters: Dict[str, Parameter]) -> Dict[str, Parameter]:
    return {
        name: param
        for name, param in named_parameters.items()
        if "layernorm" in name or "layer_norm" in name
    }


def get_non_layernorm_parameters(named_parameters: Dict[str, Parameter]) -> Dict[str, Parameter]:
    return {
        name: param
        for name, param in named_parameters.items()
        if not ("layernorm" in name or "layer_norm" in name)
    }


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flatten_dict_gen(d, parent_key, sep))


class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        encoder_output = self.bart_encoder(
            input_ids=src_tokens, attention_mask=mask, return_dict=True, output_hidden_states=True
        )
        encoder_hidden_states = encoder_output.last_hidden_state
        hidden_states = encoder_output.hidden_states
        return encoder_hidden_states, mask, hidden_states


class CaGFBartDecoder(torch.nn.Module):
    # Copy and generate,
    def __init__(
        self,
        decoder,
        pad_token_id,
        target_token_ids,
        label_ids,
        eos_id,
        pad_id,
        encoder_embed_positions,
        use_encoder_mlp=False,
        position_type=0,
        replace_pos=True,
        max_target_positions=None,
    ):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        self.encoder_embed_positions = encoder_embed_positions
        max_target_positions = max_target_positions or self.decoder.max_target_positions
        causal_mask = torch.zeros(max_target_positions, max_target_positions).fill_(float("-inf"))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer("causal_masks", causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_token_ids = [target_token_ids[label_id] for label_id in label_ids]
        target2token_id = torch.LongTensor(target_token_ids)
        self.eos_token_id = target_token_ids[eos_id]
        self.pad_id = pad_id
        self.register_buffer("target2token_id", target2token_id)
        self.pointer_offset = len(target2token_id)  # plus one (加上一个)
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.bi_encoder_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        self.position_type = position_type
        self.replace_pos = replace_pos

        if replace_pos:
            if position_type == 0:
                self.decoder.embed_positions_replace.weight = self.decoder.embed_positions.weight
                repeat_pos = torch.tensor([2, 2, 3, 2, 2, 3, 3])  # pad 0 start 1
            elif position_type == 1:
                repeat_pos = torch.tensor([2, 3, 4, 2, 3, 4, 4])  # pad 0 start 1
            elif position_type == 2:
                repeat_pos = torch.tensor([2, 3, 4, 5, 6, 7, 8])  # pad 0 start 1
            elif position_type == 3:
                repeat_pos = torch.tensor([2, 3, 4, 2, 3, 4, 5])  # pad 0 start 1
            elif position_type == 4:
                repeat_pos = torch.tensor([2, 2, 2, 2, 2, 2, 2])
            elif position_type == 5:
                repeat_pos = torch.tensor([2, 2, 2, 2, 2, 2, 2])
            elif position_type == 6:
                repeat_pos = torch.tensor([2, 2, 3, 2, 2, 3, 4])
            elif position_type == 7:
                self.replace_pos = False
                self.decoder.embed_positions.reset_parameters()
                repeat_pos = torch.tensor([2, 2, 2, 2, 2, 2, 2])  # not used
            elif position_type == 8:
                self.decoder.embed_positions_replace.weight = self.decoder.embed_positions.weight
                repeat_pos = torch.tensor([2, 2, 3, 2, 2, 3, 3])  # pad 0 start 1
            elif position_type in [9, 10]:
                orig_decoder_embed_positions_replace = self.decoder.embed_positions_replace
                self.decoder.embed_positions_replace = LearnedPositionalEmbedding(
                    num_embeddings=(
                        orig_decoder_embed_positions_replace.num_embeddings
                        - orig_decoder_embed_positions_replace.padding_idx
                    )
                    * 2,
                    embedding_dim=orig_decoder_embed_positions_replace.embedding_dim,
                    padding_idx=orig_decoder_embed_positions_replace.padding_idx,
                )
                # TODO: will this be still connected to self.encoder_embed_positions and self.decoder.embed_positions?
                self.decoder.embed_positions_replace.weight = torch.nn.Parameter(
                    torch.concat(
                        [self.encoder_embed_positions.weight, self.decoder.embed_positions.weight],
                        dim=0,
                    )
                )
                repeat_pos = None
            else:
                raise ValueError(f"position_type {position_type} not supported")
            # 2 2 3 2 2 3 3
            # 2 3 4 5 6 7 8
            # 2 3 4 2 3 4 4
            # 2 3 4 2 3 4 5
            # 1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3
            pad_pos = torch.tensor(0)
            start_pos = torch.tensor(1)
            if repeat_pos is not None:
                self.register_buffer("repeat_pos", repeat_pos)
            self.register_buffer("pad_pos", pad_pos)
            self.register_buffer("start_pos", start_pos)

    def prepare_RPE(self, tokens, mapping_token_mask, src_tokens_index, tag_mapped_tokens=None):
        if self.position_type in [9, 10]:
            bsz, seq_len = tokens.size()[:2]
            target_pos = torch.stack(
                [torch.arange(seq_len, dtype=torch.long, device=tokens.device) + 1024] * bsz,
                dim=0,
            )
            return torch.where(mapping_token_mask, target_pos, src_tokens_index)

        if tag_mapped_tokens is None:
            mapped_tokens = tokens.masked_fill(tokens.ge(self.pointer_offset), 0)
            tag_mapped_tokens = self.target2token_id[mapped_tokens]
        bsz, tokens_len = tokens.size()
        repeat_num = tokens_len // 7 + 1  # 1 if int((tokens_len-1)/7)== 0 else
        pos_tokens = self.repeat_pos.repeat(bsz, repeat_num)
        if self.position_type == 4:
            reshape_pos = pos_tokens.view(bsz, -1, 7)
            shift_pos = reshape_pos.size(1)
            add_shift_pos = torch.range(0, shift_pos - 1).repeat(bsz).view(bsz, -1).unsqueeze(-1)
            reshape_pos = add_shift_pos.to(reshape_pos.device) + reshape_pos
            pos_tokens = reshape_pos.view(bsz, -1).long()
        pos_tokens = torch.cat([self.start_pos.repeat(bsz, 1), pos_tokens], dim=-1)
        pos_tokens = pos_tokens[:bsz, :tokens_len]
        # TODO: what is 2 again? pad_idx?
        pos_tokens = pos_tokens.masked_fill(tag_mapped_tokens.eq(2), self.pad_pos.data)
        return pos_tokens

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        encoder_input_ids,
        CPM_tag=None,
        generate=False,
        past_key_values=None,
    ):
        cumsum = input_ids.eq(self.pad_id).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        mapping_token_mask = input_ids.lt(self.pointer_offset)
        mapped_tokens = input_ids.masked_fill(input_ids.ge(self.pointer_offset), 0)
        tag_mapped_tokens = self.target2token_id[mapped_tokens]

        encoder_input_ids_index = input_ids - self.pointer_offset  # bsz x num_src_token
        encoder_input_ids_index = encoder_input_ids_index.masked_fill(
            encoder_input_ids_index.lt(0), 0
        )

        assert encoder_input_ids_index.max() < 1024
        word_mapped_tokens = encoder_input_ids.gather(index=encoder_input_ids_index, dim=1)

        decoder_input_ids = torch.where(
            mapping_token_mask, tag_mapped_tokens, word_mapped_tokens
        )  # bsz x max_len
        decoder_input_ids = decoder_input_ids.masked_fill(tgt_pad_mask, self.pad_token_id)
        if self.replace_pos:
            pos_tokens = self.prepare_RPE(
                tokens=decoder_input_ids,
                mapping_token_mask=mapping_token_mask,
                tag_mapped_tokens=tag_mapped_tokens,
                src_tokens_index=encoder_input_ids_index,
            )
        else:
            pos_tokens = None

        if not generate:
            if pos_tokens is not None:
                positions = pos_tokens[:, :-1]
            else:
                positions = None
            decoder_input_ids = decoder_input_ids[:, :-1]
            decoder_pad_mask = decoder_input_ids.eq(self.pad_token_id)  # decoder需要让pad位置为1
            decoder_output = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_padding_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_pad_mask,
                decoder_causal_mask=self.causal_masks[
                    : decoder_input_ids.size(1), : decoder_input_ids.size(1)
                ],
                past_key_values=past_key_values,
                return_dict=True,
                pos_emb=positions,
            )
        else:
            assert CPM_tag is None
            positions = pos_tokens

            decoder_output = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_padding_mask=encoder_padding_mask,
                decoder_padding_mask=None,
                decoder_causal_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                pos_emb=positions,
            )
        hidden_state = decoder_output.last_hidden_state  # bsz x max_len x hidden_size

        # assemble the logits
        logits = hidden_state.new_full(
            (
                hidden_state.size(0),
                hidden_state.size(1),
                self.pointer_offset + encoder_input_ids.size(-1),
            ),
            fill_value=-1e24,
        )

        # eos and tag scores depend only on the decoder output
        eos_scores = F.linear(
            hidden_state,
            self.decoder.embed_tokens.weight[[self.eos_token_id]],
        )  # bsz x max_len x 1
        label_scores = F.linear(
            hidden_state, self.decoder.embed_tokens.weight[self.label_token_ids]
        )  # bsz x max_len x num_class

        # the pointer depends on the src token embeddings, the encoder output and the decoder output
        # bsz x max_bpe_len x hidden_size
        # src_outputs = state.encoder_output
        src_outputs = encoder_hidden_states
        if hasattr(self, "encoder_mlp"):
            src_outputs = self.encoder_mlp(src_outputs)

        # mask = state.encoder_mask.eq(0)
        mask = encoder_padding_mask.eq(0)
        mask = mask.unsqueeze(1)
        input_embed = self.decoder.embed_tokens(
            encoder_input_ids
        )  # bsz x max_word_len x hidden_size
        bsz = encoder_input_ids.size(0)
        position_embed = torch.stack(
            [self.encoder_embed_positions(encoder_input_ids)] * bsz, dim=0
        )

        word_scores = torch.einsum(
            "blh,bnh->bln", hidden_state, src_outputs
        )  # bsz x max_len x max_word_len
        gen_scores = torch.einsum(
            "blh,bnh->bln", hidden_state, input_embed
        )  # bsz x max_len x max_word_len
        positions_scores = torch.einsum(
            "blh,bnh->bln", hidden_state, position_embed
        )  # bsz x max_len x max_word_len

        if self.position_type == 9:
            avg_word_scores = (positions_scores + word_scores) / 2
        elif self.position_type == 10:
            avg_word_scores = positions_scores
        else:
            avg_word_scores = (gen_scores + word_scores) / 2
        # TODO: what are 2 and 1?
        mask = mask.__or__(encoder_input_ids.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        avg_word_scores = avg_word_scores.masked_fill(mask, -1e32)
        word_scores = word_scores.masked_fill(mask, -1e32)

        # Note: logits[:, :, 0] contains the score for the bos token which should be never generated!
        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2 : self.pointer_offset] = label_scores
        logits[:, :, self.pointer_offset :] = avg_word_scores

        constrain_logits = None
        constrain_tag = None
        if CPM_tag is not None:
            # if hasattr(self, "bi_encoder_mlp"):
            bi_outputs = self.bi_encoder_mlp(src_outputs)
            bi_label_scores = F.linear(
                hidden_state,
                self.bi_encoder_mlp(self.decoder.embed_tokens.weight[self.label_token_ids]),
            )
            bi_logits = torch.einsum(
                "blh,bnh->bln", hidden_state, bi_outputs
            )  # bsz x max_len x max_word_len
            constrain_logits = hidden_state.new_full(
                (
                    hidden_state.size(0),
                    hidden_state.size(1),
                    self.pointer_offset + encoder_input_ids.size(-1),
                ),
                fill_value=-1e24,
            )
            constrain_logits[:, :, 2 : self.pointer_offset] = bi_label_scores
            constrain_logits[:, :, self.pointer_offset :] = bi_logits
            constrain_tag = CPM_tag.float()[..., 2:]
            constrain_logits = constrain_logits[..., 2:]

        return logits, (constrain_logits, constrain_tag), None, decoder_output.past_key_values

    def decode(self, tokens, state):
        voc_logits, _, token_cls_scores, past_key_values = self(
            input_ids=tokens,
            generate=True,
            encoder_hidden_states=state.encoder_output,
            encoder_padding_mask=state.encoder_mask,
            encoder_input_ids=state.src_tokens,
            past_key_values=state.past_key_values,
        )
        state.past_key_values = past_key_values
        voc_logits = voc_logits[:, -1]
        return voc_logits, None, token_cls_scores


class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new


@PyTorchIEModel.register()
class PointerNetworkModel(PyTorchIEModel):
    """用于封装Seq2SeqModel使其可以做生成任务."""

    def __init__(
        self,
        bart_model: str,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        none_id: int,
        span_ids: List[int],
        relation_ids: List[int],
        label_ids: List[int],
        target_token_ids: List[int],
        vocab_size: int,
        pad_token_id: int,
        embedding_weight_mapping: Optional[Dict[int, List[int]]] = None,
        decoder_type=None,
        copy_gate: bool = False,
        use_encoder_mlp: bool = False,
        use_recur_pos: bool = False,
        tag_first: int = False,
        replace_pos: bool = True,
        position_type: int = 0,
        max_target_positions: Optional[int] = None,
        max_length: int = 30,
        max_len_a: float = 0.0,
        num_beams: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1,
        length_penalty: float = 1.0,
        restricter: Optional[Callable] = None,
        decode_mask: bool = True,
        metric_splits: List[str] = ["val", "test"],
        metric_intervals: Optional[Dict[str, int]] = None,
        taskmodule_config: Optional[Dict[str, Any]] = None,
        # added for the loss
        biloss: int = True,
        # added for the optimizer / scheduler
        lr: float = 5e-5,
        layernorm_decay: float = 0.001,
        warmup_proportion: float = 0.0,
        **kwargs,
    ):
        """:param int,None bos_id: 句子开头的token id :param int,None eos_id: 句子结束的token id :param int
        max_length: 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len :param float max_len_a:
        每句话的decode长度为max_length + max_len_a*src_len。 如果不为0，需要保证State中包含encoder_mask :param int
        num_beams: beam search的大小 :param bool do_sample: 是否通过采样的方式生成 :param float temperature:
        只有在do_sample为True才有意义 :param int top_k: 只从top_k中采样 :param float top_p:
        只从top_p的token中采样，nucles sample :param float repetition_penalty: 多大程度上惩罚重复的token :param
        float length_penalty: 对长度的惩罚，小于1鼓励长句，大于1鼓励短剧 :param int pad_id:
        当某句话生成结束之后，之后生成的内容用pad_token_id补充."""
        super().__init__(**kwargs)

        self.save_hyperparameters()

        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(vocab_size)
        encoder = model.encoder
        decoder = model.decoder

        label_token_ids = [target_token_ids[label_id] for label_id in label_ids]

        if use_recur_pos:
            decoder.set_position_embedding(label_token_ids[0], tag_first)

        if embedding_weight_mapping is not None:
            for special_token_index, source_indices in embedding_weight_mapping.items():
                embed = model.encoder.embed_tokens.weight.data[source_indices[0]]
                for i in source_indices[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(source_indices)
                model.decoder.embed_tokens.weight.data[int(special_token_index)] = embed

        self.encoder = FBartEncoder(encoder)
        if decoder_type is None:
            assert copy_gate is False
            raise NotImplementedError
        elif decoder_type == "avg_score":
            self.decoder = CaGFBartDecoder(
                decoder=decoder,
                encoder_embed_positions=model.encoder.embed_positions,
                pad_token_id=pad_token_id,
                target_token_ids=target_token_ids,
                label_ids=label_ids,
                eos_id=eos_id,
                pad_id=pad_id,
                use_encoder_mlp=use_encoder_mlp,
                position_type=position_type,
                replace_pos=replace_pos,
                max_target_positions=max_target_positions,
            )
            self.decoder.relation_ids = relation_ids
            self.decoder.span_ids = span_ids
            self.decoder.none_ids = none_id
        else:
            raise RuntimeError("Unsupported feature.")

        self.generator = SequenceGenerator(
            decoder=self.decoder,
            max_length=max_length,
            max_len_a=max_len_a,
            num_beams=num_beams,
            do_sample=do_sample,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=pad_id,
            restricter=restricter,
            decode_mask=decode_mask,
        )

        self.losses = {
            "train": Seq2SeqLoss(biloss=biloss),
            "val": Seq2SeqLoss(biloss=biloss),
            "test": Seq2SeqLoss(biloss=biloss),
        }
        self.metrics: Optional[Dict[str, Metric]] = None
        if taskmodule_config is not None:
            taskmodule_kwargs = copy.copy(taskmodule_config)
            taskmodule_kwargs.pop(TaskModule.config_type_key)
            taskmodule = PointerNetworkTaskModuleForEnd2EndRE(**taskmodule_kwargs)
            taskmodule.post_prepare()
            if not isinstance(taskmodule, HasBuildMetric):
                raise Exception(
                    f"taskmodule {taskmodule} does not implement HasBuildMetric interface"
                )
            # NOTE: This is not a ModuleDict, so this will not live on the torch device!
            self.metrics = {stage: taskmodule.build_metric(stage) for stage in metric_splits}
        else:
            self.metrics = None
        self.metric_intervals = metric_intervals or {}

        self.lr = lr
        self.layernorm_decay = layernorm_decay
        self.warmup_proportion = warmup_proportion

    def prepare_state(self, src_tokens, src_seq_len=None):
        encoder_hidden_states, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(
            encoder_output=encoder_hidden_states,
            encoder_mask=encoder_mask,
            src_tokens=src_tokens,
            src_embed_outputs=src_embed_outputs,
        )
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def forward(
        self,
        src_tokens,
        tgt_tokens,
        src_attention_mask=None,
        tgt_attention_mask=None,
        CPM_tag=None,
        src_seq_len=None,
    ):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens=src_tokens, src_seq_len=src_seq_len)
        decoder_output = self.decoder(
            input_ids=tgt_tokens,
            CPM_tag=CPM_tag,
            encoder_hidden_states=state.encoder_output,
            encoder_padding_mask=state.encoder_mask,
            encoder_input_ids=state.src_tokens,
            generate=False,
        )
        if isinstance(decoder_output, torch.Tensor):
            return {"pred": decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {
                "pred": decoder_output[0],
                "constrain_pred": decoder_output[1],
                # TODO: remove? (was added for testing)
                "state": state,
            }
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")

    def predict(self, inputs, **kwargs):
        """给定source的内容，输出generate的内容.

        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor src_seq_len: bsz
        :return:
        """
        src_tokens = inputs["src_tokens"]
        src_seq_len = inputs.get("src_seq_len", None)
        is_training = self.training
        self.eval()
        state = self.prepare_state(src_tokens=src_tokens, src_seq_len=src_seq_len)
        result = self.generator.generate(state, src_seq_len)
        if is_training:
            self.train()
        return {"pred": result}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs, _ = batch
        pred = self.predict(inputs=inputs)
        return pred

    def step(self, batch: GmamModelStepBatchEncoding, stage: str, batch_idx: int):
        inputs, targets = batch
        batch_size = inputs["src_tokens"].shape[0]
        criterion = self.losses.get(stage, None)
        loss = None
        if criterion is not None:
            output = self.forward(
                tgt_tokens=targets["tgt_tokens"],
                CPM_tag=targets.get("CPM_tag", None),
                **inputs,
            )
            loss = criterion(output, targets).mean()
            self.log(
                name=f"loss/{stage}",
                value=loss,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        if stage == "train" and loss is None:
            raise Exception("loss is not allowed to be None for the training step")

        if self.metrics is not None:
            stage_metrics = self.metrics.get(stage, None)
            metric_interval = self.metric_intervals.get(stage, 1)
            if stage_metrics is not None and (batch_idx + 1) % metric_interval == 0:
                prediction = self.predict(inputs)
                # the format of expected needs to be the same as the format of prediction
                stage_metrics.update(prediction, {"pred": targets["tgt_tokens"]})

        return loss

    def training_step(self, batch: GmamModelStepBatchEncoding, batch_idx):
        loss = self.step(batch, stage="train", batch_idx=batch_idx)

        return loss

    def validation_step(self, batch: GmamModelStepBatchEncoding, batch_idx):
        loss = self.step(batch, stage="val", batch_idx=batch_idx)

        return loss

    def test_step(self, batch: GmamModelStepBatchEncoding, batch_idx):
        loss = self.step(batch, stage="test", batch_idx=batch_idx)

        return loss

    def on_train_epoch_end(self):
        self._on_epoch_end(stage="train")

    def on_validation_epoch_end(self):
        self._on_epoch_end(stage="val")

    def on_test_epoch_end(self):
        self._on_epoch_end(stage="test")

    def _on_epoch_end(self, stage: str):
        if self.metrics is not None:
            metrics = self.metrics.get(stage, None)
            if metrics is not None:
                metric_dict = metrics.compute()
                metrics.reset()
                metric_dict_flat = flatten_dict(d=metric_dict, sep="/")
                for k, v in metric_dict_flat.items():
                    self.log(f"metric_{k}/{stage}", v, on_step=False, on_epoch=True, prog_bar=True)

    @property
    def head_parameters(self) -> Dict[str, Any]:
        return {
            name: param
            for name, param in self.named_parameters()
            if not ("bart_encoder" in name or "decoder.decoder" in name)
        }

    @property
    def decoder_only_parameters(self) -> Dict[str, Parameter]:
        return {
            name: param
            for name, param in self.named_parameters()
            if ("decoder.decoder" in name and "embed_tokens" not in name)
        }

    @property
    def encoder_only_parameters(self) -> Dict[str, Parameter]:
        return {
            name: param
            for name, param in self.named_parameters()
            if ("bart_encoder" in name and "embed_tokens" not in name)
        }

    @property
    def shared_encoder_decoder_parameters(self) -> Dict[str, Parameter]:
        return {name: param for name, param in self.named_parameters() if "embed_tokens" in name}

    def configure_optimizers(self):
        # head parameters
        parameters = []
        params = {
            "lr": self.lr,
            "weight_decay": 1e-2,
            "params": list(self.head_parameters.values()),
        }
        parameters.append(params)

        # decoder only parameters
        params = {
            "lr": self.lr,
            "weight_decay": 1e-2,
            "params": list(self.decoder_only_parameters.values()),
        }
        parameters.append(params)

        # encoder only layer norm parameters
        params = {
            "lr": self.lr,
            "weight_decay": self.layernorm_decay,
            "params": list(get_layernorm_parameters(self.encoder_only_parameters).values()),
        }
        parameters.append(params)

        # encoder only other parameters
        params = {
            "lr": self.lr,
            "weight_decay": 1e-2,
            "params": list(get_non_layernorm_parameters(self.encoder_only_parameters).values()),
        }
        parameters.append(params)

        # shared parameters
        params = {
            "lr": self.lr,
            "weight_decay": 1e-2,
            "params": list(self.shared_encoder_decoder_parameters.values()),
        }
        parameters.append(params)

        optimizer = torch.optim.AdamW(parameters)

        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
