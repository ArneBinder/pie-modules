from typing import Optional, Union

import torch

# from fastNLP import seq_len_to_mask
# from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
# from torch import nn
# from fastNLP.models.seq2seq_model import Seq2SeqModel
# from fastNLP.modules.decoder.seq2seq_decoder import Seq2SeqDecoder, State
# from fastNLP.models import Seq2SeqModel


class State:
    def __init__(self, encoder_output=None, encoder_mask=None, **kwargs):
        """Each Decoder has a corresponding State object to carry the encoder output and the decode
        state up to the current point in time.

        :param Union[torch.Tensor, list, tuple] encoder_output: if not None, the element inside
        must be a torch.Tensor, with the first dimension being batch by default :param
        Union[torch.Tensor, list, tuple] encoder_mask: if not None, the element inside must be a
        torch.Tensor, with the first dimension being batch by default
        :param kwargs:
        """
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self._decode_length = 0

    @property
    def num_samples(self) -> Optional[int]:
        """The returned value is the number of samples in the encoder state, which is mainly used
        to determine the size of the batch when Generate.
        (返回的State中包含的是多少个sample的encoder状态，主要用于Generate的时候确定batch的大小。)

        :return:
        """
        if self.encoder_output is not None:
            return self.encoder_output.size(0)
        else:
            return None

    @property
    def decode_length(self) -> int:
        """The decoder will only start decoding from the token after decode_length, 0 means it has
        not started decoding yet. (当前Decode到哪个token了，decoder只会从decode_length之后的token开始decode,
        为0说明还没开始decode。)

        :return:
        """
        return self._decode_length

    @decode_length.setter
    def decode_length(self, value: int):
        self._decode_length = value

    def _reorder_state(
        self, state: Union[torch.Tensor, list, tuple], indices: torch.LongTensor, dim: int = 0
    ):
        if isinstance(state, torch.Tensor):
            state = state.index_select(index=indices, dim=dim)
        elif isinstance(state, list):
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = self._reorder_state(state[i], indices, dim)
        elif isinstance(state, tuple):
            tmp_list = []
            for i in range(len(state)):
                assert state[i] is not None
                tmp_list.append(self._reorder_state(state[i], indices, dim))
            state = tuple(tmp_list)
        else:
            raise TypeError(f"Cannot reorder data of type:{type(state)}")

        return state

    def reorder_state(self, indices: torch.LongTensor):
        if self.encoder_mask is not None:
            self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        if self.encoder_output is not None:
            self.encoder_output = self._reorder_state(self.encoder_output, indices)


class Seq2SeqEncoder(torch.nn.Module):
    """所有Sequence2Sequence Encoder的基类。需要实现forward函数."""

    def __init__(self):
        super().__init__()

    def forward(self, tokens, seq_len):
        """

        :param torch.LongTensor tokens: bsz x max_len, encoder的输入
        :param torch.LongTensor seq_len: bsz
        :return:
        """
        raise NotImplementedError


class Seq2SeqDecoder(torch.nn.Module):
    """Sequence-to-Sequence
    Decoder的基类。一定需要实现forward、decode函数，剩下的函数根据需要实现。每个Seq2SeqDecoder都应该有相应的State对象
    用来承载该Decoder所需要的Encoder输出、Decoder需要记录的历史信息(例如LSTM的hidden信息)。"""

    def __init__(self):
        super().__init__()

    def forward(self, tokens, state, **kwargs):
        """

        :param torch.LongTensor tokens: bsz x max_len
        :param State state: state包含了encoder的输出以及decode之前的内容
        :return: 返回值可以为bsz x max_len x vocab_size的Tensor，也可以是一个list，但是第一个元素必须是词的预测分布
        """
        raise NotImplementedError

    def reorder_states(self, indices, states):
        """根据indices重新排列states中的状态，在beam search进行生成时，会用到该函数。

        :param torch.LongTensor indices:
        :param State states:
        :return:
        """
        assert isinstance(
            states, State
        ), f"`states` should be of type State instead of {type(states)}"
        states.reorder_state(indices)

    def init_state(self, encoder_output, encoder_mask):
        """初始化一个state对象，用来记录了encoder的输出以及decode已经完成的部分。

        :param Union[torch.Tensor, list, tuple] encoder_output: 如果不为None，内部元素需要为torch.Tensor,
        默认其中第一维是batch     维度 :param Union[torch.Tensor, list, tuple] encoder_mask:
        如果部位None，内部元素需要torch.Tensor, 默认其中第一维是batch     维度
        :param kwargs:
        :return: State, 返回一个State对象，记录了encoder的输出
        """
        state = State(encoder_output, encoder_mask)
        return state

    def decode(self, tokens, state):
        """根据states中的内容，以及tokens中的内容进行之后的生成。

        :param torch.LongTensor tokens: bsz x max_len, 截止到上一个时刻所有的token输出。
        :param State state: 记录了encoder输出与decoder过去状态
        :return: torch.FloatTensor: bsz x vocab_size, 输出的是下一个时刻的分布
        """
        outputs = self(state=state, tokens=tokens)
        if isinstance(outputs, torch.Tensor):
            return outputs[:, -1]
        else:
            raise RuntimeError(
                "Unrecognized output from the `forward()` function. Please override the `decode()` function."
            )


class Seq2SeqModel(torch.nn.Module):
    def __init__(self, encoder: Seq2SeqEncoder, decoder: Seq2SeqDecoder):
        """可以用于在Trainer中训练的Seq2Seq模型。正常情况下，继承了该函数之后，只需要实现classmethod build_model即可。如果需要使用该模型
        进行生成，需要把该模型输入到 :class:`~fastNLP.models.SequenceGeneratorModel` 中。在本模型中，forward()会把encoder后的
        结果传入到decoder中，并将decoder的输出output出来。

        :param encoder: Seq2SeqEncoder 对象，需要实现对应的forward()函数，接受两个参数，第一个为bsz x max_len的source tokens, 第二个为
            bsz的source的长度；需要返回两个tensor: encoder_outputs: bsz x max_len x hidden_size, encoder_mask: bsz x max_len
            为1的地方需要被attend。如果encoder的输出或者输入有变化，可以重载本模型的prepare_state()函数或者forward()函数
        :param decoder: Seq2SeqDecoder 对象，需要实现init_state()函数，输出为两个参数，第一个为bsz x max_len x hidden_size是
            encoder的输出; 第二个为bsz x max_len，为encoder输出的mask，为0的地方为pad。若decoder需要更多输入，请重载当前模型的
            prepare_state()或forward()函数
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_tokens, tgt_tokens, src_seq_len=None, tgt_seq_len=None):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens, src_seq_len)
        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {"pred": decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {"pred": decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")

    def prepare_state(self, src_tokens, src_seq_len=None):
        """调用encoder获取state，会把encoder的encoder_output,
        encoder_mask直接传入到decoder.init_state中初始化一个state.

        :param src_tokens:
        :param src_seq_len:
        :return:
        """
        encoder_output, encoder_mask = self.encoder(src_tokens, src_seq_len)
        state = self.decoder.init_state(encoder_output, encoder_mask)
        return state

    @classmethod
    def build_model(cls, *args, **kwargs):
        """需要实现本方法来进行Seq2SeqModel的初始化.

        :return:
        """
        raise NotImplementedError
