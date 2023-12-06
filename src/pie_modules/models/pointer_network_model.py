from collections import Counter, defaultdict
from collections.abc import MutableMapping
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_ie.core import PyTorchIEModel
from torch import nn
from transformers import get_linear_schedule_with_warmup
from typing_extensions import TypeAlias

from ..taskmodules.components.seq2seq import (
    AnnotationEncoderDecoder,
    PointerNetworkSpanAndRelationEncoderDecoder,
)
from .components.pointer_network.generator import SequenceGenerator
from .components.pointer_network.interface import Seq2SeqDecoder, Seq2SeqEncoder, State
from .components.pointer_network.losses import Seq2SeqLoss
from .components.pointer_network.modeling_bart import (
    BartDecoder,
    BartEncoder,
    BartModel,
    LearnedPositionalEmbedding,
)
from .components.pointer_network.utils import seq_len_to_mask

GmamModelStepBatchEncoding: TypeAlias = Tuple[Dict[str, Any], Dict[str, Any]]


class LabeledAnnotationScore:
    def __init__(self, label_mapping: Optional[Dict[int, str]] = None):
        self.label_mapping = label_mapping
        self.reset()

    def reset(self):
        self.gold = []
        self.predicted = []
        self.correct = []

    def compute(self, n_gold: int, n_predicted: int, n_correct: int) -> Tuple[float, float, float]:
        recall = 0 if n_gold == 0 else (n_correct / n_gold)
        precision = 0 if n_predicted == 0 else (n_correct / n_predicted)
        f1 = 0.0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall * 100, precision * 100, f1 * 100

    def result(self) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        class_info: Dict[str, Dict[str, float]] = {}
        gold_counter = Counter([x.label for x in self.gold])
        predicted_counter = Counter([x.label for x in self.predicted])
        correct_counter = Counter([x.label for x in self.correct])
        for label, count in gold_counter.items():
            if self.label_mapping is not None:
                label = self.label_mapping[label]
            n_gold = count
            n_predicted = predicted_counter.get(label, 0)
            n_correct = correct_counter.get(label, 0)
            recall, precision, f1 = self.compute(n_gold, n_predicted, n_correct)
            class_info[label] = {
                "acc": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        n_gold = len(self.gold)
        n_predicted = len(self.predicted)
        n_correct = len(self.correct)
        recall, precision, f1 = self.compute(n_gold, n_predicted, n_correct)
        return {"acc": precision, "recall": recall, "f1": f1}, class_info

    def update(self, gold, predicted):
        gold = list(set(gold))
        predicted = list(set(predicted))
        self.gold.extend(gold)
        self.predicted.extend(predicted)
        self.correct.extend([pre_entity for pre_entity in predicted if pre_entity in gold])


class AnnotationLayerMetric:
    def __init__(
        self,
        eos_id: int,
        annotation_encoder_decoder: AnnotationEncoderDecoder,
    ):
        super().__init__()
        self.annotation_encoder_decoder = annotation_encoder_decoder
        self.eos_id = eos_id
        self.layer_metrics = {
            layer_name: LabeledAnnotationScore()
            for layer_name in self.annotation_encoder_decoder.layer_names
        }

        self.reset()

    def __call__(self, prediction, expected):
        bsz = prediction.size(0)
        self.total += bsz

        pred_eos_index = prediction.flip(dims=[1]).eq(self.eos_id).cumsum(dim=1).long()
        expected_eos_index = expected.flip(dims=[1]).eq(self.eos_id).cumsum(dim=1).long()

        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        expected_seq_len = (
            expected_eos_index.flip(dims=[1]).eq(expected_eos_index[:, -1:]).sum(dim=1)
        )  # bsz
        expected_seq_len = (expected_seq_len - 2).tolist()

        for i in range(bsz):
            # delete </s>
            # Note: I have absolutely no idea why this is not the same as:
            # expected[i, 1:expected_seq_len[i]]
            ts_tensor = expected[:, 1:][i, : expected_seq_len[i]]
            ps_tensor = prediction[:, 1:][i, : pred_seq_len[i]]
            if torch.equal(ts_tensor, ps_tensor):
                self.em += 1

            gold_annotations, gold_invalid = self.annotation_encoder_decoder.decode(
                expected[i].tolist()
            )
            predicted_annotations, invalid = self.annotation_encoder_decoder.decode(
                prediction[i].tolist()
            )
            for k, v in invalid.items():
                self.invalid[k] += v

            for layer_name, metric in self.layer_metrics.items():
                # remove duplicates from layer data
                gold_layer = set(gold_annotations[layer_name])
                pred_layer = set(predicted_annotations[layer_name])
                metric.update(gold_layer, pred_layer)

    def reset(self):
        for metric in self.layer_metrics.values():
            metric.reset()

        # total number of tuples
        self.total = 1e-13

        self.invalid = defaultdict(int)
        # this contains the number of examples where the full target sequence was predicted correctly
        self.em = 0

    def get_metric(self, reset=True):
        res = {}

        res["em"] = round(self.em / self.total, 4)

        for layer_name, metric in self.layer_metrics.items():
            overall_layer_info, layer_info = metric.result()
            res[layer_name] = layer_info
            res[layer_name + "/micro"] = overall_layer_info

        # if invalid contains a "total" key, use that to normalize, otherwise use the number of training examples
        invalid_total = self.invalid.pop("total", self.total)
        for k, v in self.invalid.items():
            res["invalid/" + k] = round(v / invalid_total, 4)
        res["invalid/all"] = round(sum(self.invalid.values()) / invalid_total, 4)

        if reset:
            self.reset()
        return res


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
        dict = self.bart_encoder(
            input_ids=src_tokens, attention_mask=mask, return_dict=True, output_hidden_states=True
        )
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states


class CaGFBartDecoder(Seq2SeqDecoder):
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
        # TODO: move into prepare_RPE
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
        tokens,
        state,
        CPM_tag=None,
        generate=False,
    ):
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        cumsum = tokens.eq(self.pad_id).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        mapping_token_mask = tokens.lt(self.pointer_offset)
        mapped_tokens = tokens.masked_fill(tokens.ge(self.pointer_offset), 0)
        tag_mapped_tokens = self.target2token_id[mapped_tokens]

        src_tokens_index = tokens - self.pointer_offset  # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        assert src_tokens_index.max() < 1024
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(
            mapping_token_mask, tag_mapped_tokens, word_mapped_tokens
        )  # bsz x max_len
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)
        if self.replace_pos:
            pos_tokens = self.prepare_RPE(
                tokens=tokens,
                mapping_token_mask=mapping_token_mask,
                tag_mapped_tokens=tag_mapped_tokens,
                src_tokens_index=src_tokens_index,
            )
        else:
            pos_tokens = None
        if not generate:
            # assert CPM_tag is not None
            # bsz,input_d,_ = tokens.shape()
            if pos_tokens is not None:
                positions = pos_tokens[:, :-1]
            else:
                positions = None
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(
                input_ids=tokens,
                encoder_hidden_states=encoder_outputs,
                encoder_padding_mask=encoder_pad_mask,
                decoder_padding_mask=decoder_pad_mask,
                decoder_causal_mask=self.causal_masks[: tokens.size(1), : tokens.size(1)],
                return_dict=True,
                pos_emb=positions,
            )
        else:
            assert CPM_tag is None
            positions = pos_tokens
            past_key_values = state.past_key_values
            dict = self.decoder(
                input_ids=tokens,
                encoder_hidden_states=encoder_outputs,
                encoder_padding_mask=encoder_pad_mask,
                decoder_padding_mask=None,
                decoder_causal_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                pos_emb=positions,
            )
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if generate:
            state.past_key_values = dict.past_key_values

        # assemble the logits
        logits = hidden_state.new_full(
            (
                hidden_state.size(0),
                hidden_state.size(1),
                self.pointer_offset + src_tokens.size(-1),
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
        src_outputs = state.encoder_output
        if hasattr(self, "encoder_mlp"):
            src_outputs = self.encoder_mlp(src_outputs)

        # if hasattr(self, "bi_encoder_mlp"):
        bi_outputs = self.bi_encoder_mlp(src_outputs)
        bi_label_scores = F.linear(
            hidden_state,
            self.bi_encoder_mlp(self.decoder.embed_tokens.weight[self.label_token_ids]),
        )

        mask = state.encoder_mask.eq(0)
        mask = mask.unsqueeze(1)
        input_embed = self.decoder.embed_tokens(src_tokens)  # bsz x max_word_len x hidden_size
        bsz = src_tokens.size(0)
        position_embed = torch.stack([self.encoder_embed_positions(src_tokens)] * bsz, dim=0)

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
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        avg_word_scores = avg_word_scores.masked_fill(mask, -1e32)
        word_scores = word_scores.masked_fill(mask, -1e32)

        # Note: logits[:, :, 0] contains the score for the bos token which should be never generated!
        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2 : self.pointer_offset] = label_scores
        logits[:, :, self.pointer_offset :] = avg_word_scores

        bi_logits = torch.einsum(
            "blh,bnh->bln", hidden_state, bi_outputs
        )  # bsz x max_len x max_word_len
        constrain_logits = hidden_state.new_full(
            (
                hidden_state.size(0),
                hidden_state.size(1),
                self.pointer_offset + src_tokens.size(-1),
            ),
            fill_value=-1e24,
        )
        constrain_logits[:, :, 2 : self.pointer_offset] = bi_label_scores
        constrain_logits[:, :, self.pointer_offset :] = bi_logits
        constrain_tag = None
        if CPM_tag is not None:
            constrain_tag = CPM_tag.float()[..., 2:]
            constrain_logits = constrain_logits[..., 2:]

        return (logits, (constrain_logits, constrain_tag), None)

    def decode(self, tokens, state):
        voc_logits, _, token_cls_scores = self(tokens, state, generate=True)
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
        label_ids: List[int],
        pad_id: int,
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
        # seq2seq_model: Seq2SeqModel END
        max_length: int = 30,
        max_len_a: float = 0.0,
        num_beams: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1,
        length_penalty: float = 1.0,
        restricter: Optional[Callable] = None,
        decode_mask: bool = True,
        metric_splits: List[str] = ["val", "test"],
        annotation_encoder_decoder_name: str = "gmam",
        annotation_encoder_decoder_kwargs: Optional[Dict[str, Any]] = None,
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
                decoder,
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
        else:
            raise RuntimeError("Unsupported feature.")

        self.generator = SequenceGenerator(
            self.decoder,
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

        self.losses = {"train": {"seq2seq": Seq2SeqLoss(biloss=biloss)}}

        # also allow "gmam_annotation_encoder_decoder" for backward compatibility
        if annotation_encoder_decoder_name in ["gmam_annotation_encoder_decoder", "gmam"]:
            annotation_encoder_decoder = PointerNetworkSpanAndRelationEncoderDecoder(
                bos_id=bos_id,
                eos_id=eos_id,
                **(annotation_encoder_decoder_kwargs or {}),
            )
            self.decoder.relation_ids = annotation_encoder_decoder.relation_ids
            self.decoder.span_ids = annotation_encoder_decoder.span_ids
            self.decoder.none_ids = annotation_encoder_decoder.none_id
        else:
            raise Exception(
                f"Unsupported annotation encoder decoder: {annotation_encoder_decoder_name}"
            )
        # NOTE: This is not a ModuleDict, so this will not live on the torch device!
        self.metrics = {
            stage: AnnotationLayerMetric(
                eos_id=eos_id,
                annotation_encoder_decoder=annotation_encoder_decoder,
            )
            for stage in metric_splits
        }

        self.lr = lr
        self.layernorm_decay = layernorm_decay
        self.warmup_proportion = warmup_proportion

    def prepare_state(self, src_tokens, src_seq_len=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def forward(self, src_tokens, tgt_tokens, CPM_tag=None, src_seq_len=None):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens=src_tokens, src_seq_len=src_seq_len)
        decoder_output = self.decoder(tokens=tgt_tokens, state=state, CPM_tag=CPM_tag)
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

    def step(self, batch: GmamModelStepBatchEncoding, stage: str):
        inputs, targets = batch
        batch_size = inputs["src_tokens"].shape[0]
        losses = self.losses.get(stage, {})
        if len(losses) > 0:
            output = self.forward(
                tgt_tokens=targets["tgt_tokens"],
                CPM_tag=targets.get("CPM_tag", None),
                **inputs,
            )
            loss_dict = {k: loss(output, targets).mean() for k, loss in losses.items()}
            for k, v in loss_dict.items():
                self.log(
                    name=f"loss/{k}/{stage}",
                    value=v,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch_size,
                )
            loss = sum(loss_dict.values())
        else:
            loss = None
            if stage == "train":
                raise Exception("loss is not allowed to be None for the training step")

        metrics = self.metrics.get(stage, None)
        if metrics is not None:
            prediction = self.predict(inputs)
            metrics(prediction["pred"], targets["tgt_tokens"])

        return loss

    def training_step(self, batch: GmamModelStepBatchEncoding, batch_idx):
        loss = self.step(batch, stage="train")

        return loss

    def validation_step(self, batch: GmamModelStepBatchEncoding, batch_idx):
        loss = self.step(batch, stage="val")

        return loss

    def test_step(self, batch: GmamModelStepBatchEncoding, batch_idx):
        loss = self.step(batch, stage="test")

        return loss

    def on_train_epoch_end(self):
        self._on_epoch_end(stage="train")

    def on_validation_epoch_end(self):
        self._on_epoch_end(stage="val")

    def on_test_epoch_end(self):
        self._on_epoch_end(stage="test")

    def _on_epoch_end(self, stage: str):
        metrics = self.metrics.get(stage, None)
        if metrics is not None:
            metric_dict = metrics.get_metric(reset=True)
            metric_dict_flat = flatten_dict(d=metric_dict, sep="/")
            for k, v in metric_dict_flat.items():
                self.log(f"metric_{k}/{stage}", v, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # norm for not bart layer
        parameters = []
        params = {"lr": self.lr, "weight_decay": 1e-2}
        params["params"] = [
            param
            for name, param in self.named_parameters()
            if not ("bart_encoder" in name or "bart_decoder" in name)
        ]
        parameters.append(params)

        params = {"lr": self.lr, "weight_decay": 1e-2}
        params["params"] = []
        for name, param in self.named_parameters():
            if ("bart_encoder" in name or "bart_decoder" in name) and not (
                "layernorm" in name or "layer_norm" in name
            ):
                params["params"].append(param)
        parameters.append(params)

        params = {"lr": self.lr, "weight_decay": self.layernorm_decay}
        params["params"] = []
        for name, param in self.named_parameters():
            if ("bart_encoder" in name or "bart_decoder" in name) and (
                "layernorm" in name or "layer_norm" in name
            ):
                params["params"].append(param)
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
