import inspect
from abc import abstractmethod
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from src.models.components.gmam.utils import (
    _build_args,
    _check_arg_dict_list,
    _CheckError,
    _CheckRes,
    _get_func_signature,
)


def build_pair(
    tag_seq: List[int],
    target_ids: List[int],
    span_ids: List[int],
    relation_ids: List[int],
    none_id: int,
    if_skip_cross: bool = True,
) -> Tuple[List[Tuple[int, int, int, int, int, int, int]], Tuple[int, int, int, int]]:
    invalid_len = 0
    invalid_order = 0
    invalid_cross = 0
    invalid_cover = 0
    skip = False
    pairs = []
    cur_pair: List[int] = []
    if len(tag_seq):
        for i in tag_seq:
            if i in relation_ids or (i == none_id and len(cur_pair) == 6):
                cur_pair.append(i)
                if len(cur_pair) != 7:
                    skip = True
                    invalid_len = 1
                elif none_id in cur_pair:
                    # tag = set([cur_pair[2], cur_pair[5], cur_pair[6]])
                    if not (
                        cur_pair[2] in target_ids
                        and cur_pair[5] in target_ids
                        and cur_pair[6] in target_ids
                    ):
                        # if not tag.issubset(add_token):
                        skip = True
                    else:
                        skip = False
                else:  # The decoding length is correct (解码长度正确)
                    # Check for correct position (检查位置是否正确) <s1,e1,t1,s2,e2,t2,t3>
                    if cur_pair[0] > cur_pair[1] or cur_pair[3] > cur_pair[4]:
                        skip = True
                        invalid_order = 1
                    elif not (cur_pair[1] < cur_pair[3] or cur_pair[0] > cur_pair[4]):
                        skip = True
                        invalid_cover = 1
                    if (
                        cur_pair[2] in relation_ids
                        or cur_pair[5] in relation_ids
                        or cur_pair[6] in span_ids
                    ):
                        if (
                            if_skip_cross
                        ):  # Consider making an additional layer of restrictions to prevent misalignment of the relationship and span tags (可以考虑做多一层限制，防止relation 和 span标签错位)
                            skip = True
                        invalid_cross = 1
                    # tag = set([cur_pair[2], cur_pair[5], cur_pair[6]])
                    RC_idx = relation_ids + span_ids
                    if not (
                        cur_pair[2] in RC_idx and cur_pair[5] in RC_idx and cur_pair[6] in RC_idx
                    ):
                        # if not tag.issubset(self.relation_idx+self.span_idx):
                        skip = True
                        invalid_cross = 1

                if skip:
                    skip = False
                else:
                    if len(cur_pair) != 7:
                        raise Exception(f"expected 7 entries, but got: {cur_pair}")
                    pairs.append(tuple(cur_pair))
                cur_pair = []
            else:
                cur_pair.append(i)
    # pairs = list(set(pairs))

    # ignore type because of tuple length
    return pairs, (invalid_len, invalid_order, invalid_cross, invalid_cover)  # type: ignore


class MetricBase:
    r"""所有metrics的基类,所有的传入到Trainer, Tester的Metric需要继承自该对象，需要覆盖写入evaluate(), get_metric()方法。

        evaluate(xxx)中传入的是一个batch的数据。

        get_metric(xxx)当所有数据处理完毕，调用该方法得到最终的metric值

    以分类问题中，Accuracy计算为例
    假设model的forward返回dict中包含 `pred` 这个key, 并且该key需要用于Accuracy::

        class Model(nn.Module):
            def __init__(xxx):
                # do something
            def forward(self, xxx):
                # do something
                return {'pred': pred, 'other_keys':xxx} # pred's shape: batch_size x num_classes

    假设dataset中 `label` 这个field是需要预测的值，并且该field被设置为了target
    对应的AccMetric可以按如下的定义, version1, 只使用这一次::

        class AccMetric(MetricBase):
            def __init__(self):
                super().__init__()

                # 根据你的情况自定义指标
                self.corr_num = 0
                self.total = 0

            def evaluate(self, label, pred): # 这里的名称需要和dataset中target field与model返回的key是一样的，不然找不到对应的value
                # dev或test时，每个batch结束会调用一次该方法，需要实现如何根据每个batch累加metric
                self.total += label.size(0)
                self.corr_num += label.eq(pred).sum().item()

            def get_metric(self, reset=True): # 在这里定义如何计算metric
                acc = self.corr_num/self.total
                if reset: # 是否清零以便重新计算
                    self.corr_num = 0
                    self.total = 0
                return {'acc': acc} # 需要返回一个dict，key为该metric的名称，该名称会显示到Trainer的progress bar中


    version2，如果需要复用Metric，比如下一次使用AccMetric时，dataset中目标field不叫label而叫y，或者model的输出不是pred::

        class AccMetric(MetricBase):
            def __init__(self, label=None, pred=None):
                # 假设在另一场景使用时，目标field叫y，model给出的key为pred_y。则只需要在初始化AccMetric时，
                #   acc_metric = AccMetric(label='y', pred='pred_y')即可。
                # 当初始化为acc_metric = AccMetric()，即label=None, pred=None, fastNLP会直接使用'label', 'pred'作为key去索取对
                #   应的的值
                super().__init__()
                self._init_param_map(label=label, pred=pred) # 该方法会注册label和pred. 仅需要注册evaluate()方法会用到的参数名即可
                # 如果没有注册该则效果与version1就是一样的

                # 根据你的情况自定义指标
                self.corr_num = 0
                self.total = 0

            def evaluate(self, label, pred): # 这里的参数名称需要和self._init_param_map()注册时一致。
                # dev或test时，每个batch结束会调用一次该方法，需要实现如何根据每个batch累加metric
                self.total += label.size(0)
                self.corr_num += label.eq(pred).sum().item()

            def get_metric(self, reset=True): # 在这里定义如何计算metric
                acc = self.corr_num/self.total
                if reset: # 是否清零以便重新计算
                    self.corr_num = 0
                    self.total = 0
                return {'acc': acc} # 需要返回一个dict，key为该metric的名称，该名称会显示到Trainer的progress bar中


    ``MetricBase`` 将会在输入的字典 ``pred_dict`` 和 ``target_dict`` 中进行检查.
    ``pred_dict`` 是模型当中 ``forward()`` 函数或者 ``predict()`` 函数的返回值.
    ``target_dict`` 是DataSet当中的ground truth, 判定ground truth的条件是field的 ``is_target`` 被设置为True.

    ``MetricBase`` 会进行以下的类型检测:

    1. self.evaluate当中是否有varargs, 这是不支持的.
    2. self.evaluate当中所需要的参数是否既不在 ``pred_dict`` 也不在 ``target_dict`` .
    3. self.evaluate当中所需要的参数是否既在 ``pred_dict`` 也在 ``target_dict`` .

    除此以外，在参数被传入self.evaluate以前，这个函数会检测 ``pred_dict`` 和 ``target_dict`` 当中没有被用到的参数
    如果kwargs是self.evaluate的参数，则不会检测


    self.evaluate将计算一个批次(batch)的评价指标，并累计。 没有返回值
    self.get_metric将统计当前的评价指标并返回评价结果, 返回值需要是一个dict, key是指标名称，value是指标的值
    """

    def __init__(self):
        self._param_map = {}  # key is param in function, value is input param.
        self._checked = False
        self._metric_name = self.__class__.__name__

    @property
    def param_map(self):
        if len(self._param_map) == 0:  # 如果为空说明还没有初始化
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = [arg for arg in func_spect.args if arg != "self"]
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_metric(self, reset=True):
        raise NotImplementedError

    def set_metric_name(self, name: str):
        r"""设置metric的名称，默认是Metric的class name.

        :param str name:
        :return: self
        """
        self._metric_name = name
        return self

    def get_metric_name(self):
        r"""返回metric的名称.

        :return:
        """
        return self._metric_name

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
        func_spect = inspect.getfullargspec(self.evaluate)
        func_args = [arg for arg in func_spect.args if arg != "self"]
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.evaluate)}. Please check the "
                    f"initialization parameters, or change its signature."
                )

    def __call__(self, pred_dict, target_dict):
        r"""
        这个方法会调用self.evaluate 方法.
        在调用之前，会进行以下检测:
            1. self.evaluate当中是否有varargs, 这是不支持的.
            2. self.evaluate当中所需要的参数是否既不在``pred_dict``也不在``target_dict``.
            3. self.evaluate当中所需要的参数是否既在``pred_dict``也在``target_dict``.

            除此以外，在参数被传入self.evaluate以前，这个函数会检测``pred_dict``和``target_dict``当中没有被用到的参数
            如果kwargs是self.evaluate的参数，则不会检测
        :param pred_dict: 模型的forward函数或者predict函数返回的dict
        :param target_dict: DataSet.batch_y里的键-值对所组成的dict(即is_target=True的fields的内容)
        :return:
        """

        if not self._checked:
            if not callable(self.evaluate):
                raise TypeError(
                    f"{self.__class__.__name__}.evaluate has to be callable, not {type(self.evaluate)}."
                )
            # 1. check consistence between signature and _param_map
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = {arg for arg in func_spect.args if arg != "self"}
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.evaluate)}.")

            # 2. only part of the _param_map are passed, left are not
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {
                input_arg: func_arg for func_arg, input_arg in self._param_map.items()
            }

        # need to wrap inputs in dict.
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
            check_res = _check_arg_dict_list(self.evaluate, [mapped_pred_dict, mapped_target_dict])
            # only check missing.
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
                    check_res=check_res, func_signature=_get_func_signature(self.evaluate)
                )
            self._checked = True
        refined_args = _build_args(self.evaluate, **mapped_pred_dict, **mapped_target_dict)

        self.evaluate(**refined_args)

        return


class Seq2SeqBinaryRelationMetric(MetricBase):
    def __init__(
        self,
        eos_id: int,
        span_ids: List[int],
        relation_ids: List[int],
        none_ids: Union[List[int], int],
        target_token2id: Dict[str, int],
    ):
        super().__init__()
        self.eos_id = eos_id
        self.target_token2id = target_token2id

        self.target_id2token = dict(zip(target_token2id.values(), target_token2id.keys()))
        if isinstance(none_ids, int):
            self.none_id = none_ids
        elif isinstance(none_ids, (list, tuple)):
            assert len(none_ids) == 1
            self.none_id = none_ids[0]
        self.span_ids = span_ids
        self.relation_ids = relation_ids
        self.target_ids = list(self.target_id2token)
        self.none_label = self.target_id2token[self.none_id]

        self.span_metric = LabeledSequenceScore()

        self.relation_metric = LabeledSequenceScore()

        self.reset()

    def evaluate(self, pred, tgt_tokens):
        bsz = pred.size(0)
        self.total += bsz

        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # delete </s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = (
            target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1)
        )  # bsz
        target_seq_len = (target_seq_len - 2).tolist()
        for i in range(bsz):
            ts_tensor = tgt_tokens[i, : target_seq_len[i]]
            ps_tensor = pred[i, : pred_seq_len[i]]

            if torch.equal(ts_tensor, ps_tensor):
                self.em += 1

            target_tuples, _ = build_pair(
                tag_seq=ts_tensor.tolist(),
                target_ids=self.target_ids,
                span_ids=self.span_ids,
                relation_ids=self.relation_ids,
                none_id=self.none_id,
            )

            # Calculate the invalid for each of the three cases (分别计算三种情况的invalid)
            predicted_tuples, invalid = build_pair(
                tag_seq=ps_tensor.tolist(),
                target_ids=self.target_ids,
                span_ids=self.span_ids,
                relation_ids=self.relation_ids,
                none_id=self.none_id,
            )
            self.invalid += sum(invalid)
            self.invalid_len += invalid[0]
            self.invalid_order += invalid[1]
            self.invalid_cross += invalid[2]
            self.invalid_cover += invalid[3]

            # convert label ids to strings
            target_tuples = [
                tuple(self.target_id2token.get(x, x) for x in t) for t in target_tuples
            ]
            predicted_tuples = [
                tuple(self.target_id2token.get(x, x) for x in t) for t in predicted_tuples
            ]

            # collect span data
            span_target = [tuple(t[0:3]) for t in target_tuples] + [
                tuple(t[3:6]) for t in target_tuples
            ]
            span_pred = [tuple(t[0:3]) for t in predicted_tuples] + [
                tuple(t[3:6]) for t in predicted_tuples
            ]

            # remove none entries and duplicates from span data
            span_target = {t for t in span_target if t[-1] != self.none_label}
            span_pred = {t for t in span_pred if t[-1] != self.none_label}
            # collect counts
            self.span_tp += len(span_target & span_pred)
            self.span_fn += len(span_target - span_pred)
            self.span_fp += len(span_pred - span_target)

            self.span_metric.update(span_target, span_pred)

            # remove none entries and duplicates from full relation data
            relation_target = {t for t in target_tuples if self.none_label not in t}
            relation_pred = {t for t in predicted_tuples if self.none_label not in t}
            # collect counts
            self.triple_tp += len(relation_pred & relation_target)
            self.triple_fp += len(relation_pred - relation_target)
            self.triple_fn += len(relation_target - relation_pred)

            self.relation_metric.update(relation_target, relation_pred)

    def reset(self):
        self.span_metric.reset()
        self.relation_metric.reset()

        self.triple_fp = 0
        self.triple_tp = 0
        self.triple_fn = 0

        self.span_fp = 0
        self.span_tp = 0
        self.span_fn = 0

        # total number of tuples
        self.total = 1e-13

        self.invalid = 0
        self.invalid_len = 0
        self.invalid_order = 0
        self.invalid_cross = 0
        self.invalid_cover = 0
        # this contains the number of examples where the full target sequence was predicted correctly
        self.em = 0

    def get_metric(self, reset=True):
        res = {}

        # TODO: remove? duplicate of relation metrics
        f, pre, rec = compute_f_pre_rec(1, self.triple_tp, self.triple_fn, self.triple_fp)
        res["triple/f"] = round(f, 4) * 100
        res["triple/r"] = round(rec, 4) * 100
        res["triple/p"] = round(pre, 4) * 100

        # TODO: remove? duplicate of entity metrics
        f, pre, rec = compute_f_pre_rec(1, self.span_tp, self.span_fn, self.span_fp)
        res["span/f"] = round(f, 4) * 100
        res["span/r"] = round(rec, 4) * 100
        res["span/p"] = round(pre, 4) * 100

        res["em"] = round(self.em / self.total, 4)
        res["invalid/all"] = round(self.invalid / self.total, 4)

        overall_span_info, span_info = self.span_metric.result()
        res["entity"] = span_info
        res["entity/micro"] = overall_span_info

        overall_relation_info, relation_info = self.relation_metric.result()
        res["relation"] = relation_info
        res["relation/micro"] = overall_relation_info

        res["invalid/len"] = round(self.invalid_len / self.total, 4)
        res["invalid/order"] = round(self.invalid_order / self.total, 4)
        res["invalid/cross"] = round(self.invalid_cross / self.total, 4)
        res["invalid/cover"] = round(self.invalid_cover / self.total, 4)
        if reset:
            self.reset()
        return res


def compute_f_pre_rec(beta_square, tp, fn, fp):
    r"""

    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * tp / (fn + fp + (1 + beta_square) * tp + 1e-13)

    return f, pre, rec


class LabeledSequenceScore:
    def __init__(self, id2label: Optional[Dict[int, str]] = None, label_pos: int = -1):
        self.label_pos = label_pos
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.gold = []
        self.predicted = []
        self.correct = []

    def compute(self, n_gold, n_predicted, n_correct):
        recall = 0 if n_gold == 0 else (n_correct / n_gold)
        precision = 0 if n_predicted == 0 else (n_correct / n_predicted)
        f1 = 0.0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall * 100, precision * 100, f1 * 100

    def result(self):
        class_info = {}
        gold_counter = Counter([x[self.label_pos] for x in self.gold])
        predicted_counter = Counter([x[self.label_pos] for x in self.predicted])
        correct_counter = Counter([x[self.label_pos] for x in self.correct])
        for label, count in gold_counter.items():
            if self.id2label is not None:
                label = self.id2label[label]
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


def _prepare_metrics(metrics):
    r"""Prepare list of Metric based on input :param metrics:

    :return: List[fastNLP.MetricBase]
    """
    _metrics = []
    if metrics:
        if isinstance(metrics, list):
            for metric in metrics:
                if isinstance(metric, type):
                    metric = metric()
                if isinstance(metric, MetricBase):
                    metric_name = metric.__class__.__name__
                    if not callable(metric.evaluate):
                        raise TypeError(
                            f"{metric_name}.evaluate must be callable, got {type(metric.evaluate)}."
                        )
                    if not callable(metric.get_metric):
                        raise TypeError(
                            f"{metric_name}.get_metric must be callable, got {type(metric.get_metric)}."
                        )
                    _metrics.append(metric)
                else:
                    raise TypeError(
                        f"The type of metric in metrics must be `MetricBase`, not `{type(metric)}`."
                    )
        elif isinstance(metrics, MetricBase):
            _metrics = [metrics]
        else:
            raise TypeError(
                f"The type of metrics should be `list[MetricBase]` or `MetricBase`, "
                f"got {type(metrics)}."
            )
    return _metrics
