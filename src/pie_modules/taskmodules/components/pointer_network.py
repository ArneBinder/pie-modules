import logging
from collections import Counter, defaultdict
from functools import cmp_to_key
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from pytorch_ie import Document, PreparableMixin
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import Annotation

from pie_modules.taskmodules.components.common import (
    AnnotationEncoderDecoder,
    AnnotationLayersEncoderDecoder,
)

logger = logging.getLogger(__name__)


def cmp_src_rel(v1: BinaryRelation, v2: BinaryRelation) -> int:
    if not all(isinstance(ann, LabeledSpan) for ann in [v1.head, v1.tail, v2.head, v2.tail]):
        raise Exception(f"expected LabeledSpan, but got: {v1}, {v2}")
    if v1.head.start == v2.head.start:  # v1[0]["from"] == v2[0]["from"]:
        return v1.tail.start - v2.tail.start  # v1[1]["from"] - v2[1]["from"]
    return v1.head.start - v2.head.start  # v1[0]["from"] - v2[0]["from"]


class SimpleSpanEncoderDecoder(AnnotationEncoderDecoder[Span, List[int]]):
    def encode(self, span: Span, metadata: Optional[Dict[str, Any]] = None) -> Optional[List[int]]:
        return [span.start, span.end]

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Span]:
        if len(encoding) != 2:
            raise Exception(
                f"two values are required to decode as Span, but the encoding is: {encoding}"
            )
        return Span(start=encoding[0], end=encoding[1])


class SpanEncoderDecoderWithOffset(AnnotationEncoderDecoder[Span, List[int]]):
    def __init__(self, offset: int, span_end_mode: str = "last_token"):
        self.span_end_mode = span_end_mode
        self.offset = offset

    def encode(
        self, annotation: Span, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[List[int]]:
        if self.span_end_mode == "first_token_of_last_word":
            raise NotImplementedError("span_end_mode=first_token_of_last_word not implemented")
        elif self.span_end_mode == "last_token":
            return [annotation.start + self.offset, annotation.end + self.offset - 1]
        else:
            raise Exception(f"unknown span_end_mode: {self.span_end_mode}")

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Span]:
        if len(encoding) != 2:
            raise Exception(
                f"two values are required to decode as Span, but encoding is: {encoding}"
            )

        if self.span_end_mode == "first_token_of_last_word":
            raise NotImplementedError("span_end_mode=first_token_of_last_word not implemented")
        elif self.span_end_mode == "last_token":
            return Span(start=encoding[0] - self.offset, end=encoding[1] - self.offset + 1)
        else:
            raise Exception(f"unknown span_end_mode: {self.span_end_mode}")


class LabeledSpanEncoderDecoder(AnnotationEncoderDecoder[LabeledSpan, List[int]]):
    def __init__(
        self,
        span_encoder_decoder: AnnotationEncoderDecoder[Span, List[int]],
        label2id: Dict[str, int],
    ):
        self.span_encoder_decoder = span_encoder_decoder
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def encode(
        self, annotation: LabeledSpan, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[List[int]]:
        encoded_span = self.span_encoder_decoder.encode(annotation=annotation, metadata=metadata)
        if encoded_span is None:
            return None
        return encoded_span + [self.label2id[annotation.label]]

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> LabeledSpan:
        if len(encoding) != 3:
            raise Exception(
                f"three values are required to decode as LabeledSpan, but encoding is: {encoding}"
            )
        decoded_span = self.span_encoder_decoder.decode(encoding=encoding[:2], metadata=metadata)
        if decoded_span is None:
            raise Exception(f"failed to decode span from encoding: {encoding}")
        result = LabeledSpan(
            start=decoded_span.start,
            end=decoded_span.end,
            label=self.id2label[encoding[2]],
        )
        return result


class BinaryRelationEncoderDecoder(AnnotationEncoderDecoder[BinaryRelation, List[int]]):
    def __init__(
        self,
        head_encoder_decoder: AnnotationEncoderDecoder[Span, List[int]],
        tail_encoder_decoder: AnnotationEncoderDecoder[Span, List[int]],
        label2id: Dict[str, int],
        loop_dummy_relation_name: Optional[str] = "loop",
        none_label: str = "none",
    ):
        self.head_encoder_decoder = head_encoder_decoder
        self.tail_encoder_decoder = tail_encoder_decoder
        self.loop_dummy_relation_name = loop_dummy_relation_name
        self.none_label = none_label
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def encode(
        self, annotation: BinaryRelation, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[List[int]]:
        encoded_head = self.head_encoder_decoder.encode(annotation=annotation.head)
        encoded_tail = self.tail_encoder_decoder.encode(annotation=annotation.tail)
        if encoded_head is None or encoded_tail is None:
            raise Exception(f"failed to encode head or tail from annotation: {annotation}")
        if (
            self.loop_dummy_relation_name is None
            or annotation.label != self.loop_dummy_relation_name
        ):
            return encoded_tail + encoded_head + [self.label2id[annotation.label]]
        else:
            if encoded_head != encoded_tail:
                raise Exception(
                    f"expected encoded_head == encoded_tail for loop_dummy_relation, but got: {encoded_head}, "
                    f"{encoded_tail}"
                )
            none_id = self.label2id[self.none_label]
            return encoded_head + [none_id, none_id, none_id, none_id]

    def decode(
        self, encoding: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> BinaryRelation:
        if len(encoding) != 7:
            raise Exception(
                f"seven values are required to decode as BinaryRelation, but encoding is: {encoding}"
            )

        decoded_tail = self.tail_encoder_decoder.decode(encoding=encoding[:3], metadata=metadata)
        label = self.id2label[encoding[6]]
        if label == self.none_label:
            decoded_head = decoded_tail
        else:
            decoded_head = self.head_encoder_decoder.decode(
                encoding=encoding[3:6], metadata=metadata
            )
        if decoded_head is None or decoded_tail is None:
            raise Exception(f"failed to decode head or tail from encoding: {encoding}")
        rel = BinaryRelation(head=decoded_head, tail=decoded_tail, label=label)
        return rel


class PointerNetworkSpanAndRelationEncoderDecoder(
    AnnotationLayersEncoderDecoder[List[int]],
    PreparableMixin,
):
    PREPARED_ATTRIBUTES = ["labels_per_layer"]

    def __init__(
        self,
        bos_token: str,
        eos_token: str,
        span_layer_name: str = "labeled_spans",
        relation_layer_name: str = "binary_relations",
        none_label: str = "none",
        loop_dummy_relation_name: str = "loop",
        ignore_error_types: Optional[List[str]] = None,
        span_encoder_decoder_name: str = "span_encoder_decoder_with_offset",
        span_encoder_decoder_kwargs: Optional[Dict[str, Any]] = None,
        labels_per_layer: Optional[Dict[str, List[str]]] = None,
        exclude_labels_per_layer: Optional[Dict[str, List[str]]] = None,
    ):
        self.labels_per_layer = labels_per_layer
        self.exclude_labels_per_layer = exclude_labels_per_layer or {}
        self.none_label = none_label
        self.span_layer_name = span_layer_name
        self.relation_layer_name = relation_layer_name
        self.span_encoder_decoder_name = span_encoder_decoder_name
        self.span_encoder_decoder_kwargs = span_encoder_decoder_kwargs

        # self.span_encoder_decoder: SpanEncoderDecoder
        self.labeled_span_encoder_decoder: LabeledSpanEncoderDecoder
        self.relation_encoder_decoder: BinaryRelationEncoderDecoder

        self.loop_dummy_relation_name = loop_dummy_relation_name
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.ignore_error_types = ignore_error_types or []

        if self.is_prepared:
            self._post_prepare()

    @property
    def layer_names(self) -> List[str]:
        return [self.span_layer_name, self.relation_layer_name]

    @property
    def special_targets(self) -> list[str]:
        return [self.bos_token, self.eos_token]

    @property
    def special_target2id(self) -> Dict[str, int]:
        return {target: idx for idx, target in enumerate(self.special_targets)}

    @property
    def target_pad_id(self) -> int:
        return self.special_target2id[self.eos_token]

    @property
    def generation_kwargs(self) -> Dict[str, Any]:
        return {
            "no_repeat_ngram_size": 7,
            # TODO: add this when it looks really solid (currently strange behavior)
            # "prefix_allowed_tokens_fn": self._prefix_allowed_tokens_fn,
        }

    def get_min_pointer_idx_from_input_ids(
        self, input_ids: List[int], current_tuple_size: int
    ) -> int:
        if len(input_ids) == 0:
            raise Exception(f"input_ids is empty: {input_ids}")

        # only bos token
        if len(input_ids) == 1:
            return 0

        # create a dummy value that is larger than any value in input_ids to get never selected
        max_dummy_value = max(input_ids) + 1
        padded_input_ids = input_ids + [max_dummy_value] * (7 - current_tuple_size)
        last_tuple = padded_input_ids[-7:]
        (
            last_tail_start,
            last_tail_end,
            last_tail_label,
            last_head_start,
            last_head_end,
            last_head_label,
            last_rel_label,
        ) = last_tuple
        candidate_indices = [last_tail_start, last_head_start]
        # we only consider the previous tuple if it is available and the start idx of the last head
        # was not yet generated
        if len(input_ids) > 7 and last_head_start == max_dummy_value:
            previous_tuple = padded_input_ids[-14:-7]
            (
                previous_tail_start,
                previous_tail_end,
                previous_tail_label,
                previous_head_start,
                previous_head_end,
                previous_head_label,
                previous_rel_label,
            ) = previous_tuple
            candidate_indices += [previous_tail_start, previous_head_start]
        return min(idx for idx in candidate_indices if idx != self.label2id[self.none_label])

    def _prefix_allowed_tokens_fn(self, batch_id: int, input_ids: torch.Tensor) -> List[int]:
        # if we get eos, we can only continue with eos (TODO: should be pad instead)
        if input_ids[-1] == self.eos_id:
            return [self.eos_id]

        # this will be 0 to 6 (inclusive)
        current_tuple_size = (input_ids.size(0) - 1) % 7

        # TODO: use get_min_pointer_idx_from_input_ids(). but this decreases performance for now...
        min_pointer_idx = self.pointer_offset
        # min_pointer_idx = self.get_min_pointer_idx_from_input_ids(input_ids.tolist(), current_tuple_size)

        # TODO: parameterize 1024 (should be max encoder input size!)
        max_pointer_idx = 1024 + self.pointer_offset

        none_id = self.label2id[self.none_label]
        # if we got a tuple with a none label, we can only continue with a none label
        current_tuple = input_ids[-current_tuple_size:].tolist()
        if 3 < len(current_tuple) < 7 and current_tuple[3] == none_id:
            return [none_id]

        # next id should be a pointer index, eos or none
        if current_tuple_size in [0, 1, 3, 4]:
            allowed_ids = list(range(min_pointer_idx, max_pointer_idx))

            # add eos to allow to terminate
            if current_tuple_size == 0:
                allowed_ids += [self.eos_id]
            # add none to allow to continue with a none label
            elif current_tuple_size == 3:
                allowed_ids += [none_id]

            return allowed_ids
        # next id should be a span label index
        elif current_tuple_size in [2, 5]:
            return self.span_ids
        # next id should be a relation label index
        elif current_tuple_size == 6:
            # TODO: why do we need to allow the span_ids here? removing them decreases performance for now...
            return self.relation_ids + self.span_ids
        else:
            raise Exception(f"unexpected current_tuple_size: {current_tuple_size}")

    def _prepare(self, documents: Sequence[Document]):
        span_labels: Set[str] = set()
        relation_labels: Set[str] = set()
        for doc in documents:
            span_labels.update(
                ac.label
                for ac in doc[self.span_layer_name]
                if ac.label not in self.exclude_labels_per_layer.get(self.span_layer_name, [])
            )
            relation_labels.update(
                rel.label
                for rel in doc[self.relation_layer_name]
                if rel.label not in self.exclude_labels_per_layer.get(self.relation_layer_name, [])
            )
        self.labels_per_layer = {
            self.span_layer_name: sorted(span_labels),
            self.relation_layer_name: sorted(relation_labels),
        }

    def _post_prepare(self) -> None:
        # set up labels
        if self.labels_per_layer is None:
            raise Exception("labels_per_layer is not defined. Call prepare() first or pass it in.")
        self.labels: List[str] = [self.none_label]
        for layer_name in self.layer_names:
            self.labels.extend(self.labels_per_layer[layer_name])
        if len(set(self.labels)) != len(self.labels):
            raise Exception(f"labels are not unique: {self.labels}")

        # set up targets and ids
        self.targets: List[str] = self.special_targets + self.labels
        self.target2id: Dict[str, int] = {target: idx for idx, target in enumerate(self.targets)}

        # generic ids
        self.eos_id: int = self.target2id[self.eos_token]
        self.bos_id: int = self.target2id[self.bos_token]

        # span and relation ids
        self.span_ids: List[int] = [
            self.target2id[label] for label in self.labels_per_layer[self.span_layer_name]
        ]
        self.relation_ids: List[int] = [
            self.target2id[label] for label in self.labels_per_layer[self.relation_layer_name]
        ]
        # the none id is used for the dummy relation which models out-of-relation spans
        self.none_id: int = self.target2id[self.none_label]

        # span encoder decoder
        self.span_encoder_decoder: Union[SimpleSpanEncoderDecoder, SpanEncoderDecoderWithOffset]
        if self.span_encoder_decoder_name == "span_encoder_decoder_with_offset":
            self.span_encoder_decoder = SpanEncoderDecoderWithOffset(
                offset=self.pointer_offset, **(self.span_encoder_decoder_kwargs or {})
            )
        elif self.span_encoder_decoder_name == "simple":
            self.span_encoder_decoder = SimpleSpanEncoderDecoder(
                **(self.span_encoder_decoder_kwargs or {})
            )
        else:
            raise Exception(f"unknown span_encoder_decoder_name: {self.span_encoder_decoder_name}")

        # helpers (same as targets / target2id, but only for labels)
        self.label2id: Dict[str, int] = {label: self.target2id[label] for label in self.labels}
        self.id2label: Dict[int, str] = {idx: label for label, idx in self.label2id.items()}
        self.label_ids: List[int] = [self.label2id[label] for label in self.labels]

        self.labeled_span_encoder_decoder = LabeledSpanEncoderDecoder(
            span_encoder_decoder=self.span_encoder_decoder,
            label2id=self.label2id,
        )
        self.relation_encoder_decoder = BinaryRelationEncoderDecoder(
            head_encoder_decoder=self.labeled_span_encoder_decoder,
            tail_encoder_decoder=self.labeled_span_encoder_decoder,
            label2id=self.label2id,
            loop_dummy_relation_name=self.loop_dummy_relation_name,
            none_label=self.none_label,
        )

    @property
    def pointer_offset(self) -> int:
        return len(self.targets)

    @property
    def target_ids(self) -> Set[int]:
        return set(range(self.pointer_offset))

    def sanitize_sequence(
        self,
        tag_seq: List[int],
    ) -> Tuple[List[Tuple[int, int, int, int, int, int, int]], Dict[str, int]]:
        # TODO: count total amounts instead of returning bool values.
        #  This requires to also count "total" (maybe also "skipped" and "correct").
        invalid = {
            "len": 0,
            "order": 0,
            "cross": 0,
            "cover": 0,
        }  # , "total": 0 , "skipped": 0, "correct": 0}
        skip = False
        pairs = []
        cur_pair: List[int] = []
        if len(tag_seq):
            for i in tag_seq:
                if i in self.relation_ids or (i == self.none_id and len(cur_pair) == 6):
                    cur_pair.append(i)
                    if len(cur_pair) != 7:
                        skip = True
                        invalid["len"] = 1
                    elif self.none_id in cur_pair:
                        # tag = set([cur_pair[2], cur_pair[5], cur_pair[6]])
                        if not (
                            cur_pair[2] in self.target_ids
                            and cur_pair[5] in self.target_ids
                            and cur_pair[6] in self.target_ids
                        ):
                            # if not tag.issubset(add_token):
                            skip = True
                        else:
                            skip = False
                    else:  # The decoding length is correct (解码长度正确)
                        # Check for correct position (检查位置是否正确) <s1,e1,t1,s2,e2,t2,t3>
                        if cur_pair[0] > cur_pair[1] or cur_pair[3] > cur_pair[4]:
                            if "cover" not in self.ignore_error_types:
                                skip = True
                            invalid["order"] = 1
                        elif not (cur_pair[1] < cur_pair[3] or cur_pair[0] > cur_pair[4]):
                            skip = True
                            invalid["cover"] = 1
                        if (
                            cur_pair[2] in self.relation_ids
                            or cur_pair[5] in self.relation_ids
                            or cur_pair[6] in self.span_ids
                        ):
                            # Consider making an additional layer of restrictions to prevent misalignment
                            # of the relationship and span tags (可以考虑做多一层限制，防止relation 和 span标签错位)
                            if "cross" not in self.ignore_error_types:
                                skip = True
                            invalid["cross"] = 1
                        # tag = set([cur_pair[2], cur_pair[5], cur_pair[6]])
                        RC_idx = self.relation_ids + self.span_ids
                        if not (
                            cur_pair[2] in RC_idx
                            and cur_pair[5] in RC_idx
                            and cur_pair[6] in RC_idx
                        ):
                            # if not tag.issubset(self.relation_idx+self.span_idx):
                            skip = True
                            invalid["cross"] = 1

                    if skip:
                        skip = False
                        # invalid["skipped"] += 1
                    else:
                        if len(cur_pair) != 7:
                            raise Exception(f"expected 7 entries, but got: {cur_pair}")
                        pairs.append(tuple(cur_pair))
                        # invalid["correct"] += 1
                    cur_pair = []
                else:
                    cur_pair.append(i)

        # invalid["total"] = invalid["correct"] + invalid["skipped"]

        # ignore type because of tuple length
        return pairs, invalid  # type: ignore

    def encode(
        self, layers: Dict[str, List[Annotation]], metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        if not set(layers.keys()) == set(self.layer_names):
            raise Exception(f"unexpected layers: {layers.keys()}. expected: {self.layer_names}")

        if self.labels_per_layer is None:
            raise Exception("labels_per_layer is not defined. Call prepare() first or pass it in.")

        # encode relations
        all_relation_arguments = set()
        relation_encodings = dict()
        for rel in layers[self.relation_layer_name]:
            if not isinstance(rel, BinaryRelation):
                raise Exception(f"expected BinaryRelation, but got: {rel}")
            if rel.label in self.labels_per_layer[self.relation_layer_name]:
                encoded_relation = self.relation_encoder_decoder.encode(
                    annotation=rel, metadata=metadata
                )
                if encoded_relation is None:
                    raise Exception(f"failed to encode relation: {rel}")
                relation_encodings[rel] = encoded_relation
                all_relation_arguments.update([rel.head, rel.tail])

        # encode spans that are not arguments of any relation
        no_relation_spans = [
            span for span in layers[self.span_layer_name] if span not in all_relation_arguments
        ]
        for span in no_relation_spans:
            dummy_relation = BinaryRelation(
                head=span, tail=span, label=self.loop_dummy_relation_name
            )
            encoded_relation = self.relation_encoder_decoder.encode(
                annotation=dummy_relation, metadata=metadata
            )
            if encoded_relation is not None:
                relation_encodings[dummy_relation] = encoded_relation

        # sort relations by start indices of head and tail
        sorted_relations = sorted(relation_encodings, key=cmp_to_key(cmp_src_rel))

        # build target tokens
        tgt_tokens = [self.bos_id]
        for rel in sorted_relations:
            encoded_relation = relation_encodings[rel]
            tgt_tokens.extend(encoded_relation)
        tgt_tokens.append(self.eos_id)

        # sanity check
        _, invalid = self.sanitize_sequence(tag_seq=tgt_tokens[1:])
        if not all(v == 0 for k, v in invalid.items() if k not in self.ignore_error_types):
            decoded, invalid = self.decode(tgt_tokens, metadata=metadata)
            logger.warning(f"invalid: {invalid}, decoded: {decoded}")

        return tgt_tokens

    def decode(
        self, targets: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, List[Annotation]], Any]:
        # strip the bos token
        ps, _errors = self.sanitize_sequence(tag_seq=targets[1:])
        relation_tuples: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
        entity_labels: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for tup in ps:
            rel = self.relation_encoder_decoder.decode(encoding=list(tup), metadata=metadata)
            head_span = (rel.head.start, rel.head.end)
            entity_labels[head_span].append(rel.head.label)

            if rel.label != self.none_label:
                tail_span = (rel.tail.start, rel.tail.end)
                entity_labels[tail_span].append(rel.tail.label)
                relation_tuples.append((head_span, tail_span, rel.label))
            else:
                assert rel.head == rel.tail

        entities: Dict[Tuple[int, int], LabeledSpan] = {}
        for (start, end), labels in entity_labels.items():
            c = Counter(labels)
            # if len(c) > 1:
            #    logger.warning(f"multiple labels for span, take the most common: {dict(c)}")
            most_common_label = c.most_common(1)[0][0]
            entities[(start, end)] = LabeledSpan(start=start, end=end, label=most_common_label)

        entity_layer = list(entities.values())
        relation_layer = [
            BinaryRelation(head=entities[head_span], tail=entities[tail_span], label=label)
            for head_span, tail_span, label in relation_tuples
        ]
        return {
            self.span_layer_name: entity_layer,
            self.relation_layer_name: relation_layer,
        }, _errors

    def _pointer_tag(
        self,
        last: List[int],
        t: int,
        idx: int,
        arr: np.ndarray,
    ) -> np.ndarray:
        if t == 0:  # start # c1 [0, 1]
            arr[: self.pointer_offset] = 0
        elif idx % 7 == 0:  # c1 [0,1, 23]
            arr[:t] = 0
        elif idx % 7 == 1:  # tc1 [0,1,23, tc] span标签设为1
            arr = np.zeros_like(arr, dtype=int)
            for i in self.span_ids:
                arr[i] = 1
        elif idx % 7 == 2:  # c2 [0,1,23,tc, 45]
            arr[: self.pointer_offset] = 0
            arr[last[-3] : last[-2]] = 0
        elif idx % 7 == 3:  # c2 [0,1,23,tc,45, 67]
            arr[:t] = 0
            if t < last[-4]:
                arr[last[-4] :] = 0
            else:
                arr[last[-4] : last[-3]] = 0
        elif idx % 7 == 4:  # tc2 [0,1,23,tc,45,67, tc]
            arr = np.zeros_like(arr, dtype=int)
            for i in self.span_ids:
                arr[i] = 1
        elif idx % 7 == 5:  # r [0,1,23,tc,45,67,tc, r]
            arr = np.zeros_like(arr, dtype=int)
            for i in self.relation_ids:
                arr[i] = 1
        elif idx % 7 == 6:  # next
            arr[: self.pointer_offset] = 0
        return arr

    def build_constraints(
        self,
        src_len: int,
        tgt_tokens: List[int],
    ) -> List[List[int]]:
        # strip the bos token
        target = tgt_tokens[1:]
        # pad for 0
        likely_hood = np.ones(src_len + self.pointer_offset, dtype=int)
        likely_hood[: self.pointer_offset] = 0
        CMP_tag: List[np.ndarray] = [likely_hood]
        for idx, t in enumerate(target[:-1]):
            last7 = target[idx - 7 if idx - 7 > 0 else 0 : idx + 1]
            likely_hood = np.ones(src_len + self.pointer_offset, dtype=int)
            tag = self._pointer_tag(last=last7, t=t, idx=idx, arr=likely_hood)
            tag[self.none_id] = 1
            CMP_tag.append(tag)
        last_end = np.zeros(src_len + self.pointer_offset, dtype=int)
        last_end[self.none_id] = 1
        last_end[target[-1]] = 1
        CMP_tag[-1] = last_end
        result = [i.tolist() for i in CMP_tag]
        return result
