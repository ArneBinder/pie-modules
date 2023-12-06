import abc
import logging
from collections import Counter, defaultdict
from functools import cmp_to_key
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import Annotation

logger = logging.getLogger(__name__)

# ====================  INTERFACE  ====================


class SpanEncoderDecoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, span: Span, metadata: Optional[Dict[str, Any]] = None) -> Optional[List[int]]:
        pass

    @abc.abstractmethod
    def decode(self, targets: List[int], metadata: Optional[Dict[str, Any]] = None) -> Span:
        pass


class AnnotationEncoderDecoder(abc.ABC):
    layer_names: List[str]

    @abc.abstractmethod
    def encode(
        self, layers: Dict[str, List[Annotation]], metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        pass

    @abc.abstractmethod
    def decode(
        self, targets: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, List[Annotation]], Any]:
        pass


class PointerNetworkEncoderDecoder(AnnotationEncoderDecoder):
    @abc.abstractmethod
    def build_constraints(
        self,
        src_len: int,
        tgt_tokens: List[int],
    ) -> List[List[int]]:
        pass


# ====================  IMPLEMENTATIONS  ====================


def cmp_src_rel(v1: BinaryRelation, v2: BinaryRelation) -> int:
    if not all(isinstance(ann, LabeledSpan) for ann in [v1.head, v1.tail, v2.head, v2.tail]):
        raise Exception(f"expected LabeledSpan, but got: {v1}, {v2}")
    if v1.head.start == v2.head.start:  # v1[0]["from"] == v2[0]["from"]:
        return v1.tail.start - v2.tail.start  # v1[1]["from"] - v2[1]["from"]
    return v1.head.start - v2.head.start  # v1[0]["from"] - v2[0]["from"]


class SimpleSpanEncoderDecoder(SpanEncoderDecoder):
    def encode(self, span: Span, metadata: Optional[Dict[str, Any]] = None) -> List[int]:
        return [span.start, span.end]

    def decode(self, targets: List[int], metadata: Optional[Dict[str, Any]] = None) -> Span:
        assert len(targets) == 2
        return Span(start=targets[0], end=targets[1])


class PointerNetworkSpanEncoderDecoder(SpanEncoderDecoder):
    def __init__(self, span_end_mode: str, pointer_offset: int = 0):
        self.span_end_mode = span_end_mode
        self.pointer_offset = pointer_offset

    def encode(self, span: Span, metadata: Optional[Dict[str, Any]] = None) -> Optional[List[int]]:
        # map word indices (open interval) to src_token indices (closed interval, end index points to beginning
        # of the last word!)
        if metadata is None:
            raise Exception("encoding with SpanEncoderDecoder requires metadata")
        char_start, char_end = span.start, span.end
        if char_start not in metadata["char2token"]:
            return None
        token_start = metadata["char2token"][char_start][0]

        if self.span_end_mode == "first_token_of_last_word":
            if "char2word" not in metadata or "word2token" not in metadata:
                raise Exception(
                    "encoding with span_end_mode=first_token_of_last_word requires char2word and word2token mappings"
                )
            word_end = metadata["char2word"][char_end - 1] + 1
            token_end = metadata["word2token"][word_end - 1][0]
        elif self.span_end_mode == "last_token":
            if char_end - 1 not in metadata["char2token"]:
                return None
            token_end = metadata["char2token"][char_end - 1][-1]
        else:
            raise Exception(f"unknown span_end_mode: {self.span_end_mode}")

        return [token_start + self.pointer_offset, token_end + self.pointer_offset]

    def decode(self, targets: List[int], metadata: Optional[Dict[str, Any]] = None) -> Span:
        if metadata is None:
            raise Exception("decoding with SpanEncoderDecoder requires metadata")
        assert len(targets) == 2
        token_start, token_end = targets[0] - self.pointer_offset, targets[1] - self.pointer_offset
        char_start = metadata["token2char"][token_start][0]

        if self.span_end_mode == "first_token_of_last_word":
            if "char2word" not in metadata or "word2token" not in metadata:
                raise Exception(
                    "decoding with span_end_mode=first_token_of_last_word requires token2word and word2char mappings"
                )
            word_end = metadata["token2word"][token_end] + 1
            char_end = metadata["word2char"][word_end - 1][1]
        elif self.span_end_mode == "last_token":
            char_end = metadata["token2char"][token_end][1]
        else:
            raise Exception(f"unknown span_end_mode: {self.span_end_mode}")

        return Span(char_start, char_end)


class PointerNetworkSpanAndRelationEncoderDecoder(PointerNetworkEncoderDecoder):
    span_layer_name = "span"
    relation_layer_name = "relation"
    layer_names = [span_layer_name, relation_layer_name]

    def __init__(
        self,
        id2label: Dict[int, str],
        bos_id: int,
        eos_id: int,
        span_ids: List[int],
        relation_ids: List[int],
        none_id: int,
        loop_dummy_relation_name: str = "loop",
        ignore_error_types: Optional[List[str]] = None,
        span_encoder_decoder: Optional[SpanEncoderDecoder] = None,
    ):
        self.span_encoder_decoder = span_encoder_decoder or SimpleSpanEncoderDecoder()
        self.id2label = id2label
        self.label2id: Dict[str, int] = {v: k for k, v in id2label.items()}
        self.loop_dummy_relation_name = loop_dummy_relation_name
        self.none_label = self.id2label[none_id]
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.span_ids = span_ids
        self.relation_ids = relation_ids
        self.none_id = none_id
        self.ignore_error_types = ignore_error_types or []

    @property
    def target_ids(self):
        return [self.bos_id, self.eos_id] + self.relation_ids + [self.none_id] + self.span_ids

    @property
    def pointer_offset(self) -> int:
        return len(self.target_ids)

    def sanitize_sequence(
        self,
        tag_seq: List[int],
        target_ids: List[int],
        span_ids: List[int],
        relation_ids: List[int],
        none_id: int,
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
                if i in relation_ids or (i == none_id and len(cur_pair) == 6):
                    cur_pair.append(i)
                    if len(cur_pair) != 7:
                        skip = True
                        invalid["len"] = 1
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
                            if "cover" not in self.ignore_error_types:
                                skip = True
                            invalid["order"] = 1
                        elif not (cur_pair[1] < cur_pair[3] or cur_pair[0] > cur_pair[4]):
                            skip = True
                            invalid["cover"] = 1
                        if (
                            cur_pair[2] in relation_ids
                            or cur_pair[5] in relation_ids
                            or cur_pair[6] in span_ids
                        ):
                            # Consider making an additional layer of restrictions to prevent misalignment
                            # of the relationship and span tags (可以考虑做多一层限制，防止relation 和 span标签错位)
                            if "cross" not in self.ignore_error_types:
                                skip = True
                            invalid["cross"] = 1
                        # tag = set([cur_pair[2], cur_pair[5], cur_pair[6]])
                        RC_idx = relation_ids + span_ids
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

    def encode_labeled_span(
        self, labeled_span: LabeledSpan, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[List[int]]:
        encoded_span = self.span_encoder_decoder.encode(span=labeled_span, metadata=metadata)
        if encoded_span is None:
            return None

        return encoded_span + [self.label2id[labeled_span.label]]

    def decode_labeled_span(
        self, targets: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> LabeledSpan:
        if len(targets) != 3:
            raise Exception(
                f"three target values are required to decode as LabeledSpan, but targets is: {targets}"
            )
        decoded_span = self.span_encoder_decoder.decode(targets=targets[:2], metadata=metadata)
        result = LabeledSpan(
            start=decoded_span.start,
            end=decoded_span.end,
            label=self.id2label[targets[2]],
        )
        return result

    def encode_relation(
        self, rel: BinaryRelation, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[List[int]]:
        encoded_head = self.encode_labeled_span(labeled_span=rel.head, metadata=metadata)
        encoded_tail = self.encode_labeled_span(labeled_span=rel.tail, metadata=metadata)

        if encoded_head is None or encoded_tail is None:
            if encoded_head:
                logger.warning(f"encoded_head is None: {rel.head}")
            if encoded_tail:
                logger.warning(f"encoded_tail is None: {rel.tail}")
            return None

        if rel.label == self.loop_dummy_relation_name:
            assert encoded_head == encoded_tail
            none_id = self.label2id[self.none_label]
            target_span = encoded_head + [none_id, none_id, none_id, none_id]
        else:
            label_id = self.label2id[rel.label]
            target_span = encoded_tail + encoded_head + [label_id]

        return target_span

    def decode_relation(
        self, targets, metadata: Optional[Dict[str, Any]] = None
    ) -> BinaryRelation:
        # sent1 target
        # sent2 src
        rel_label = self.id2label[targets[6]]
        decoded_tail = self.decode_labeled_span(targets=targets[0:3], metadata=metadata)
        if rel_label == self.none_label:
            decoded_head = decoded_tail
        else:
            decoded_head = self.decode_labeled_span(targets=targets[3:6], metadata=metadata)
        rel = BinaryRelation(head=decoded_head, tail=decoded_tail, label=rel_label)
        return rel

    def encode(
        self, layers: Dict[str, List[Annotation]], metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        if not set(layers.keys()) == set(self.layer_names):
            raise Exception(f"unexpected layers: {layers.keys()}. expected: {self.layer_names}")
        spans = layers[self.layer_names[0]]
        relations = layers[self.layer_names[1]]

        all_relation_arguments = set(chain(*[(rel.head, rel.tail) for rel in relations]))
        dummy_loop_relations = [
            BinaryRelation(head=span, tail=span, label=self.loop_dummy_relation_name)
            for span in spans
            if span not in all_relation_arguments
        ]
        relations_with_dummies = list(relations) + dummy_loop_relations

        sorted_relations = sorted(relations_with_dummies, key=cmp_to_key(cmp_src_rel))

        tgt_tokens = [self.bos_id]
        for rel in sorted_relations:
            new_target_span = self.encode_relation(rel=rel, metadata=metadata)
            if new_target_span is not None:
                tgt_tokens.extend(new_target_span)
        tgt_tokens.append(self.eos_id)

        # sanity check
        _, invalid = self.sanitize_sequence(
            tag_seq=tgt_tokens[1:],
            target_ids=self.target_ids,
            span_ids=self.span_ids,
            relation_ids=self.relation_ids,
            none_id=self.none_id,
        )
        if not all(v == 0 for k, v in invalid.items() if k not in self.ignore_error_types):
            decoded, invalid = self.decode(tgt_tokens, metadata=metadata)
            logger.warning(f"invalid: {invalid}, decoded: {decoded}")

        return tgt_tokens

    def decode(
        self, targets: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, List[Annotation]], Any]:
        ps, _errors = self.sanitize_sequence(
            # strip the bos token
            tag_seq=targets[1:],
            target_ids=self.target_ids,
            span_ids=self.span_ids,
            relation_ids=self.relation_ids,
            none_id=self.none_id,
        )
        relation_tuples: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
        entity_labels: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for tup in ps:
            rel = self.decode_relation(targets=tup, metadata=metadata)
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
