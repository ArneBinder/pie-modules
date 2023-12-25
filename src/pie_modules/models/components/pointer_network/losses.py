import inspect
from collections import defaultdict

import torch

# from fastNLP import LossBase
import torch.nn.functional as F

from .utils import (
    _build_args,
    _check_arg_dict_list,
    _CheckError,
    _CheckRes,
    _get_func_signature,
    seq_len_to_mask,
)

# from fastNLP import seq_len_to_mask


class LossBase:
    r"""所有loss的基类。如果需要结合到Trainer之中需要实现get_loss方法."""

    def __init__(self):
        self._param_map = {}  # key是fun的参数，value是以该值从传入的dict取出value
        self._checked = False

    @property
    def param_map(self):
        if len(self._param_map) == 0:  # 如果为空说明还没有初始化
            func_spect = inspect.getfullargspec(self.get_loss)
            func_args = [arg for arg in func_spect.args if arg != "self"]
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    def get_loss(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return: torch.Tensor
        """
        raise NotImplementedError

    def _init_param_map(self, key_map=None, **kwargs):
        r"""检查key_map和其他参数map，并将这些映射关系添加到self._param_map.

        :param dict key_map: 表示key的映射关系
        :param kwargs: key word args里面的每一个的键-值对都会被构造成映射关系
        :return: None
        """
        value_counter = defaultdict(set)
        if key_map is not None:
            if not isinstance(key_map, dict):
                raise TypeError(f"key_map must be `dict`, got {type(key_map)}.")
            for key, value in key_map.items():
                if value is None:
                    self._param_map[key] = key
                    continue
                if not isinstance(key, str):
                    raise TypeError(f"key in key_map must be `str`, not `{type(key)}`.")
                if not isinstance(value, str):
                    raise TypeError(f"value in key_map must be `str`, not `{type(value)}`.")
                self._param_map[key] = value
                value_counter[value].add(key)
        for key, value in kwargs.items():
            if value is None:
                self._param_map[key] = key
                continue
            if not isinstance(value, str):
                raise TypeError(f"in {key}={value}, value must be `str`, not `{type(value)}`.")
            self._param_map[key] = value
            value_counter[value].add(key)
        for value, key_set in value_counter.items():
            if len(key_set) > 1:
                raise ValueError(
                    f"Several parameters:{key_set} are provided with one output {value}."
                )

        # check consistence between signature and _param_map
        func_spect = inspect.getfullargspec(self.get_loss)
        func_args = [arg for arg in func_spect.args if arg != "self"]
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.get_loss)}. Please check the "
                    f"initialization parameters, or change its signature."
                )

        # evaluate should not have varargs.
        # if func_spect.varargs:
        #     raise NameError(f"Delete `*{func_spect.varargs}` in {get_func_signature(self.get_loss)}(Do not use "
        #                     f"positional argument.).")

    def __call__(self, pred_dict, target_dict, check=False):
        r"""
        :param dict pred_dict: 模型的forward函数返回的dict
        :param dict target_dict: DataSet.batch_y里的键-值对所组成的dict
        :param Boolean check: 每一次执行映射函数的时候是否检查映射表，默认为不检查
        :return:
        """

        if not self._checked:
            # 1. check consistence between signature and _param_map
            func_spect = inspect.getfullargspec(self.get_loss)
            func_args = {arg for arg in func_spect.args if arg != "self"}
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.get_loss)}.")

            # 2. only part of the _param_map are passed, left are not
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {
                input_arg: func_arg for func_arg, input_arg in self._param_map.items()
            }

        mapped_pred_dict = {}
        mapped_target_dict = {}
        for input_arg, mapped_arg in self._reverse_param_map.items():
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
            if input_arg in target_dict:
                mapped_target_dict[mapped_arg] = target_dict[input_arg]

        # missing
        if not self._checked:
            duplicated = []
            for input_arg, mapped_arg in self._reverse_param_map.items():
                if input_arg in pred_dict and input_arg in target_dict:
                    duplicated.append(input_arg)
            check_res = _check_arg_dict_list(self.get_loss, [mapped_pred_dict, mapped_target_dict])
            # replace missing.
            missing = check_res.missing
            replaced_missing = list(missing)
            for idx, func_arg in enumerate(missing):
                # Don't delete `` in this information, nor add ``
                replaced_missing[idx] = (
                    f"{self._param_map[func_arg]}" + f"(assign to `{func_arg}` "
                    f"in `{self.__class__.__name__}`)"
                )

            check_res = _CheckRes(
                missing=replaced_missing,
                unused=check_res.unused,
                duplicated=duplicated,
                required=check_res.required,
                all_needed=check_res.all_needed,
                varargs=check_res.varargs,
            )

            if check_res.missing or check_res.duplicated:
                raise _CheckError(
                    check_res=check_res, func_signature=_get_func_signature(self.get_loss)
                )
            self._checked = True

        refined_args = _build_args(self.get_loss, **mapped_pred_dict, **mapped_target_dict)

        loss = self.get_loss(**refined_args)
        self._checked = True

        return loss


class LossInForward(LossBase):
    r"""从forward()函数返回结果中获取loss."""

    def __init__(self, loss_key="loss"):
        r"""

        :param str loss_key: 在forward函数中loss的键名，默认为loss
        """
        super().__init__()
        if not isinstance(loss_key, str):
            raise TypeError(f"Only str allowed for loss_key, got {type(loss_key)}.")
        self.loss_key = loss_key

    def get_loss(self, **kwargs):
        if self.loss_key not in kwargs:
            check_res = _CheckRes(
                missing=[
                    self.loss_key + f"(assign to `{self.loss_key}` in `{self.__class__.__name__}`"
                ],
                unused=[],
                duplicated=[],
                required=[],
                all_needed=[],
                varargs=[],
            )
            raise _CheckError(
                check_res=check_res, func_signature=_get_func_signature(self.get_loss)
            )
        return kwargs[self.loss_key]

    def __call__(self, pred_dict, target_dict, check=False):
        loss = self.get_loss(**pred_dict)

        if not (isinstance(loss, torch.Tensor) and len(loss.size()) == 0):
            if not isinstance(loss, torch.Tensor):
                raise TypeError(f"Loss excepted to be a torch.Tensor, got {type(loss)}")
            loss = torch.sum(loss) / (loss.view(-1)).size(0)
            # raise RuntimeError(f"The size of loss excepts to be torch.Size([]), got {loss.size()}")

        return loss


class Seq2SeqLoss(LossBase):
    def __init__(self, biloss):
        super().__init__()
        self.biloss = biloss

    def get_loss(self, tgt_tokens, tgt_seq_len, pred, constrain_pred):
        """

        :param tgt_tokens: bsz x max_len, [sos, tokens, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
        loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))
        if self.biloss:
            unlikely_label = constrain_pred[1]
            unlikely_label = unlikely_label[mask.eq(0), :]
            pred = constrain_pred[0][mask.eq(0), :]
            active_unlikely = unlikely_label.ge(0).view(-1)  # 取label 0和1 （invalid/valid）
            active_pred = pred.view(-1)[active_unlikely]
            active_unlikely_label = unlikely_label.view(-1)[active_unlikely]
            input = F.sigmoid(active_pred)
            loss_c = F.binary_cross_entropy(target=active_unlikely_label, input=input)
            return loss + loss_c
        else:
            return loss


def _prepare_losser(losses):
    if losses is None:
        losses = LossInForward()
        return losses
    elif isinstance(losses, LossBase):
        return losses
    else:
        raise TypeError(f"Type of loss should be `LossBase`, got {type(losses)}")
