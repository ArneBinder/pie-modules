import dataclasses
import logging
from collections import Counter, defaultdict
from functools import cmp_to_key
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_ie.annotations import BinaryRelation, LabeledSpan, Span
from pytorch_ie.core import (
    Annotation,
    AnnotationList,
    Document,
    TaskEncoding,
    TaskModule,
    annotation_field,
)
from pytorch_ie.core.taskmodule import (
    InputEncoding,
    ModelBatchOutput,
    TargetEncoding,
    TaskBatchEncoding,
)
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)


class SpanEncoderDecoderInterface:
    def encode(self, span: Span, metadata: Optional[Dict[str, Any]] = None) -> Optional[List[int]]:
        raise NotImplementedError

    def decode(self, targets: List[int], metadata: Optional[Dict[str, Any]] = None) -> Span:
        raise NotImplementedError


class SimpleSpanEncoderDecoder(SpanEncoderDecoderInterface):
    def encode(self, span: Span, metadata: Optional[Dict[str, Any]] = None) -> List[int]:
        return [span.start, span.end]

    def decode(self, targets: List[int], metadata: Optional[Dict[str, Any]] = None) -> Span:
        assert len(targets) == 2
        return Span(start=targets[0], end=targets[1])


class SpanEncoderDecoder(SpanEncoderDecoderInterface):
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


class AnnotationEncoderDecoderInterface:
    layer_names: List[str]

    def __init__(self, span_encoder_decoder: Optional[SpanEncoderDecoder] = None):
        self.span_encoder_decoder = span_encoder_decoder or SimpleSpanEncoderDecoder()

    def encode(
        self, layers: Dict[str, List[Annotation]], metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        raise NotImplementedError

    def decode(
        self, targets: List[int], metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, List[Annotation]], Any]:
        raise NotImplementedError


class AnnotationEncoderDecoder(AnnotationEncoderDecoderInterface):
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
        **kwargs,
    ):
        super().__init__(**kwargs)
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


@dataclasses.dataclass
class DocumentType(Document):
    text: str
    words: AnnotationList[Span] = annotation_field(target="text")
    spans: AnnotationList[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationList[BinaryRelation] = annotation_field(target="spans")
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


TaskOutput: TypeAlias = torch.Tensor


def _pad_tensor(tensor: torch.Tensor, target_shape: List[int], pad_value: float) -> torch.Tensor:
    shape = tensor.shape
    pad: List[int] = []
    for i, s in enumerate(shape):
        pad = [0, target_shape[i] - s] + pad
    result = F.pad(tensor, pad=pad, value=pad_value)
    assert list(result.shape) == target_shape
    return result


# aspect with relation  component_r
# opinion without relation component


def ld2dl(
    list_of_dicts: Union[List[Dict[str, Any]], Sequence[Dict[str, Any]]],
    keys: Optional[str] = None,
    getter: Optional[Callable] = None,
) -> Dict[str, List[Any]]:
    if getter is None:

        def getter(x):
            return x

    keys = keys or getter(list_of_dicts[0]).keys()
    v = {k: [getter(dic)[k] for dic in list_of_dicts] for k in keys}
    return v


def cmp_src(v1, v2):
    if v1[0]["from"] == v2[0]["from"]:
        return v1[1]["from"] - v2[1]["from"]
    return v1[0]["from"] - v2[0]["from"]


def cmp_src_rel(v1: BinaryRelation, v2: BinaryRelation):
    if v1.head.start == v2.head.start:  # v1[0]["from"] == v2[0]["from"]:
        return v1.tail.start - v2.tail.start  # v1[1]["from"] - v2[1]["from"]
    return v1.head.start - v2.head.start  # v1[0]["from"] - v2[0]["from"]


def cmp_tg(v1, v2):
    if v1[1]["from"] == v2[1]["from"]:
        return v1[0]["from"] - v2[0]["from"]
    return v1[1]["from"] - v2[1]["from"]


def cmp_tg_rel(v1: BinaryRelation, v2: BinaryRelation):
    if v1.tail.start == v2.tail.start:  # v1[1]["from"] == v2[1]["from"]:
        return v1.head.start - v2.head.start  # v1[0]["from"] - v2[0]["from"]
    return v1.tail.start - v2.tail.start  # v1[1]["from"] - v2[1]["from"]


def _relations_to_sources_targets(relations):
    component_src_r = []
    component_tg = []
    for rel in relations:
        src_component = {
            "from": rel.head.start,
            "to": rel.head.end,
            "component": rel.head.label,
            "polarity": rel.label,
            "term": rel.head.target[rel.head.start : rel.head.end],
        }
        trg_component = {
            "from": rel.tail.start,
            "to": rel.tail.end,
            "component": rel.tail.label,
            "term": rel.tail.target[rel.tail.start : rel.tail.end],
        }
        component_src_r.append(src_component)
        component_tg.append(trg_component)

    sources_targets = list(zip(component_src_r, component_tg))
    return sources_targets


def _span_is_in_partition(span: Span, partition: Optional[Span] = None):
    if partition is None:
        return True
    return (
        partition.start <= span.start < partition.end
        and partition.start < span.end <= partition.end
    )


def pointer_tag(
    last: List[int],
    t: int,
    idx: int,
    arr: np.ndarray,
    span_ids: List[int],
    relation_ids: List[int],
    shift: int,
) -> np.ndarray:
    if t == 0:  # start # c1 [0, 1]
        arr[:shift] = 0
    elif idx % 7 == 0:  # c1 [0,1, 23]
        arr[:t] = 0
    elif idx % 7 == 1:  # tc1 [0,1,23, tc] span标签设为1
        arr = np.zeros_like(arr, dtype=int)
        for i in span_ids:
            arr[i] = 1
    elif idx % 7 == 2:  # c2 [0,1,23,tc, 45]
        arr[:shift] = 0
        arr[last[-3] : last[-2]] = 0
    elif idx % 7 == 3:  # c2 [0,1,23,tc,45, 67]
        arr[:t] = 0
        if t < last[-4]:
            arr[last[-4] :] = 0
        else:
            arr[last[-4] : last[-3]] = 0
    elif idx % 7 == 4:  # tc2 [0,1,23,tc,45,67, tc]
        arr = np.zeros_like(arr, dtype=int)
        for i in span_ids:
            arr[i] = 1
    elif idx % 7 == 5:  # r [0,1,23,tc,45,67,tc, r]
        arr = np.zeros_like(arr, dtype=int)
        for i in relation_ids:
            arr[i] = 1
    elif idx % 7 == 6:  # next
        arr[:shift] = 0
    return arr


def CPM_prepare(
    src_len: int,
    target: list,
    span_ids: List[int],
    relation_ids: List[int],
    none_ids: int,
    shift: int = 0,
) -> List[List[int]]:
    # pad for 0
    likely_hood = np.ones(src_len + shift, dtype=int)
    likely_hood[:shift] = 0
    CMP_tag: List[np.ndarray] = [likely_hood]
    for idx, t in enumerate(target[:-1]):
        last7 = target[idx - 7 if idx - 7 > 0 else 0 : idx + 1]
        likely_hood = np.ones(src_len + shift, dtype=int)
        tag = pointer_tag(last7, t, idx, likely_hood, span_ids, relation_ids, shift)
        tag[none_ids] = 1
        CMP_tag.append(tag)
    last_end = np.zeros(src_len + shift, dtype=int)
    last_end[none_ids] = 1
    last_end[target[-1]] = 1
    CMP_tag[-1] = last_end
    result = [i.tolist() for i in CMP_tag]
    return result


# TODO: use enable BucketSampler (just mentioning here because no better place available for now)
# see https://github.com/Lightning-AI/lightning/pull/13640#issuecomment-1199032224


@TaskModule.register()
class PointerNetworkForJointTaskModule(TaskModule):
    PREPARED_ATTRIBUTES = ["span_labels", "relation_labels"]

    def __init__(
        self,
        tokenizer_name_or_path: str = "facebook/bart-base",
        span_labels: Optional[List[str]] = None,
        relation_labels: Optional[List[str]] = None,
        none_label: str = "none",
        # so that the label word can be initialized in a better embedding.
        label2special_token: Optional[Dict[str, str]] = None,
        # dummy relation type to encode entities that do not belong to any relation
        loop_dummy_relation_name: str = "loop",
        exclude_annotation_names: Optional[Dict[str, List[str]]] = None,
        span_end_mode: str = "last_token",
        tokenize_per_word: bool = False,
        text_field_name: str = "raw_words",
        span_layer_name: str = "argument_components",
        relation_layer_name: str = "relations",
        word_layer_name: Optional[str] = None,
        partition_layer_name: Optional[str] = None,
        log_first_n_examples: Optional[int] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_init_kwargs: Optional[Dict[str, Any]] = None,
        max_target_length: Optional[int] = None,
        create_constraints: bool = True,
        annotation_encoder_decoder_name: str = "gmam",
        annotation_encoder_decoder_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.label2special_token = label2special_token or {}
        self.span_labels = span_labels
        self.relation_labels = relation_labels
        self.none_label = none_label

        self.loop_dummy_relation_name = loop_dummy_relation_name
        self.exclude_annotation_names = exclude_annotation_names or dict()
        self.create_constraints = create_constraints

        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            **(tokenizer_init_kwargs or {}),
        )

        self.tokenizer_kwargs = tokenizer_kwargs or {}

        # how to encode the end of the span
        self.annotation_encoder_decoder_name = annotation_encoder_decoder_name
        self.annotation_encoder_decoder_kwargs = annotation_encoder_decoder_kwargs or {}

        self.span_end_mode = span_end_mode

        self.text_field_name = text_field_name
        self.span_layer_name = span_layer_name
        self.relation_layer_name = relation_layer_name
        self.partition_layer_name = partition_layer_name
        self.word_layer_name = word_layer_name

        self.tokenize_per_word = tokenize_per_word

        self.max_target_length = max_target_length

        # see fastNLP.core.batch.DataSetGetter
        # self.input_names = {"src_tokens", "src_seq_len", "tgt_tokens", "tgt_seq_len", "CPM_tag"}
        # self.target_names = {"src_seq_len", "tgt_tokens", "tgt_seq_len", "target_span"}

        self.pad_values = {
            # "tgt_tokens": 1,  # this will be set in _post_prepare()
            "src_tokens": self.tokenizer.pad_token_id,
            "CPM_tag": -1,
        }
        self.dtypes = {
            "tgt_tokens": torch.int64,
            "src_seq_len": torch.int64,
            "src_tokens": torch.int64,
            "tgt_seq_len": torch.int64,
            "CPM_tag": torch.int64,
        }

        self.log_first_n_examples = log_first_n_examples

    def _prepare(self, documents: Sequence[DocumentType]):
        span_labels: Set[str] = set()
        relation_labels: Set[str] = set()
        for doc in documents:
            span_labels.update(
                ac.label
                for ac in doc[self.span_layer_name]
                if ac.label not in self.exclude_annotation_names.get(self.span_layer_name, [])
            )
            relation_labels.update(
                rel.label
                for rel in doc[self.relation_layer_name]
                if rel.label not in self.exclude_annotation_names.get(self.relation_layer_name, [])
            )
        self.span_labels = sorted(span_labels)
        self.relation_labels = sorted(relation_labels)

    def _post_prepare(self):
        # we need the following:
        # 1. labels: entity and relation labels and the none label
        # 2. label tokens: labels encapsulated with "<<" and ">>"
        # 3. target tokens: "<bos>", "<eos>", and (2)
        # 4. target ids (3)
        # 5. token ids of (3)
        # + various mappings

        self.labels = self.span_labels + self.relation_labels + [self.none_label]
        self.label2token = {
            label: self.label2special_token.get(label, f"<<{label}>>") for label in self.labels
        }
        self.token2label = {v: k for k, v in self.label2token.items()}
        if len(self.label2token) != len(self.token2label):
            raise Exception(
                f"all entries in label2token need to map to different entries, which is not the case: "
                f"{self.label2token}"
            )
        self.label_tokens = sorted(self.label2token.values(), key=lambda x: len(x), reverse=True)
        already_in_vocab = [
            tok
            for tok in self.label_tokens
            if self.tokenizer.convert_tokens_to_ids(tok) != self.tokenizer.unk_token_id
        ]
        if len(already_in_vocab) > 0:
            raise Exception(
                f"some special tokens to add (mapped label ids) are already in the tokenizer vocabulary, "
                f"this is not allowed: {already_in_vocab}. You may want to adjust the label2special_token mapping"
            )
        # self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_label_ids
        self.tokenizer.add_special_tokens(
            special_tokens_dict={"additional_special_tokens": self.label_tokens}
        )

        self.label_token_ids = self.tokenizer.convert_tokens_to_ids(self.label_tokens)
        # this returns all the token ids that can occur in the output
        self.target_token_ids = [
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
        ] + sorted(self.label_token_ids)

        self.label2id: Dict[str, int] = {}
        self.target_token2id: Dict[str, int] = {}
        for idx, target_token_id in enumerate(self.target_token_ids):
            target_token = self.tokenizer.convert_ids_to_tokens(target_token_id)
            self.target_token2id[target_token] = idx
            if target_token in self.label_tokens:
                self.label2id[self.token2label[target_token]] = idx
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.bos_id = self.target_token2id[self.tokenizer.bos_token]
        self.eos_id = self.target_token2id[self.tokenizer.eos_token]
        self.span_ids = [self.label2id[i] for i in self.span_labels]
        self.relation_ids = [self.label2id[i] for i in self.relation_labels]
        self.none_ids = self.label2id[self.none_label]
        # Set to the id where eos is located (设置为eos所在的id)
        self.pad_id = self.eos_id

        # TODO: make that configurable and do not depend on << >> syntax
        self.embedding_weight_mapping = dict()
        for label_token in self.label_tokens:
            # sanity check: label_tokens should not be split up
            special_token_index = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(label_token)
            )
            if len(special_token_index) > 1:
                raise RuntimeError(f"{label_token} wrong split")
            else:
                special_token_index = special_token_index[0]

            assert label_token[:2] == "<<" and label_token[-2:] == ">>"
            source_indices = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(label_token[2:-2])
            )
            assert self.tokenizer.unk_token_id not in source_indices
            self.embedding_weight_mapping[special_token_index] = source_indices

        if self.annotation_encoder_decoder_name == "gmam":
            span_encoder_decoder = SpanEncoderDecoder(
                span_end_mode=self.span_end_mode,
                pointer_offset=self.pointer_offset,
            )
            self.annotation_encoder_decoder = AnnotationEncoderDecoder(
                span_encoder_decoder=span_encoder_decoder,
                id2label=self.id2label,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                span_ids=self.span_ids,
                relation_ids=self.relation_ids,
                none_id=self.none_ids,
                **self.annotation_encoder_decoder_kwargs,
            )
        else:
            raise Exception(
                f"unknown annotation_encoder_decoder_name: {self.annotation_encoder_decoder_name}"
            )

    @property
    def target_tokens(self) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(self.target_token_ids)

    @property
    def target_ids(self) -> List[int]:
        return list(self.target_token2id.values())

    @property
    def label_ids(self):
        return sorted(self.label2id.values())

    @property
    def pointer_offset(self) -> int:
        return len(self.target_token_ids)

    @property
    def pad_id(self) -> int:
        v = self.pad_values.get("tgt_tokens", None)
        if v is None:
            raise Exception("pad value for tgt_tokens is not set")
        return v

    @pad_id.setter
    def pad_id(self, value):
        self.pad_values["tgt_tokens"] = value

    def maybe_log_example(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
        targets: Optional[TargetEncoding] = None,
    ):
        if self.log_first_n_examples is not None and self.log_first_n_examples > 0:
            inputs = task_encoding.inputs
            targets = targets or task_encoding.targets
            src_token_ids = inputs["src_tokens"]
            src_tokens = self.tokenizer.convert_ids_to_tokens(src_token_ids)
            tgt_token_ids = targets["tgt_tokens"]
            tgt_tokens = [
                self.tokenizer.convert_ids_to_tokens(self.target_token_ids[tgt_token_id])
                if tgt_token_id < self.pointer_offset
                else str(tgt_token_id)
                + " {"
                + str(src_tokens[tgt_token_id - self.pointer_offset])
                + "}"
                for tgt_token_id in tgt_token_ids
            ]
            logger.info("*** Example ***")
            # logger.info(f"doc id: {task_encoding.document.id}")
            logger.info(f"src_token_ids: {' '.join([str(i) for i in src_token_ids])}")
            logger.info(f"src_tokens:    {' '.join(src_tokens)}")
            logger.info(f"tgt_token_ids: {' '.join([str(i) for i in tgt_token_ids])}")
            logger.info(f"tgt_tokens:    {' '.join(tgt_tokens)}")
            self.log_first_n_examples -= 1

    def encode_input(
        self, document: DocumentType, is_training: bool = False
    ) -> Optional[
        Union[
            TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
            Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
        ]
    ]:
        task_encodings = []
        text = getattr(document, self.text_field_name)

        if self.word_layer_name is not None:
            char2word = {}
            word2char = []
            for word_idx, word in enumerate(document[self.word_layer_name]):
                word2char.append((word.start, word.end))
                for char_idx in range(word.start, word.end):
                    char2word[char_idx] = word_idx
        else:
            char2word = None
            word2char = None

        if self.partition_layer_name is not None:
            partitions = document[self.partition_layer_name]
        else:
            partitions = [None]
        for partition in partitions:
            tokenizer_encodings = []
            if self.tokenize_per_word:
                # TODO: implement truncation? striding?
                if len(self.tokenizer_kwargs) > 0:
                    raise NotImplementedError(
                        "tokenizer_kwargs not supported for tokenize_per_word"
                    )
                if self.word_layer_name not in document:
                    raise Exception(
                        f'the annotation layer "{self.word_layer_name}" that should contain word annotations '
                        f"is required if tokenize_per_word is enabled"
                    )
                token2char = []
                src_tokens = [self.tokenizer.bos_token_id]
                special_tokens_mask = [1]
                token2char.append((0, 0))
                for word in document[self.word_layer_name]:
                    if _span_is_in_partition(span=word, partition=partition):
                        word_str = text[word.start : word.end]
                        tokenizer_output = self.tokenizer(
                            word_str, return_offsets_mapping=True, add_special_tokens=False
                        )
                        token_ids = tokenizer_output.input_ids
                        for start, end in tokenizer_output.offset_mapping:
                            token2char.append((start + word.start, end + word.start))
                        src_tokens.extend(token_ids)
                        special_tokens_mask.extend([0] * len(token_ids))
                src_tokens.append(self.tokenizer.eos_token_id)
                special_tokens_mask.append(1)
                tokenizer_encodings.append((src_tokens, token2char, special_tokens_mask))
            else:
                text_partition = (
                    text[partition.start : partition.end] if partition is not None else text
                )
                tokenizer_output = self.tokenizer(
                    text_partition,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                    return_overflowing_tokens=True,
                    **self.tokenizer_kwargs,
                )
                for encoding in tokenizer_output.encodings:
                    src_tokens = encoding.ids
                    if partition is not None:
                        token2char = [
                            (start + partition.start, end + partition.start)
                            if not is_special_token
                            else (start, end)
                            for (start, end), is_special_token in zip(
                                encoding.offsets, encoding.special_tokens_mask
                            )
                        ]
                    else:
                        token2char = encoding.offsets
                    special_tokens_mask = encoding.special_tokens_mask
                    tokenizer_encodings.append((src_tokens, token2char, special_tokens_mask))

            for src_tokens, token2char, special_tokens_mask in tokenizer_encodings:
                src_seq_len = len(src_tokens)
                inputs = {"src_tokens": src_tokens, "src_seq_len": src_seq_len}

                char2token = defaultdict(list)
                for token_idx, (char_start, char_end) in enumerate(token2char):
                    for char_idx in range(char_start, char_end):
                        char2token[char_idx].append(token_idx)

                no_special_token2char = [
                    start_end
                    for start_end, is_special_token in zip(token2char, special_tokens_mask)
                    if not is_special_token
                ]
                tokenized_span = Span(
                    start=no_special_token2char[0][0], end=no_special_token2char[-1][1]
                )
                metadata = {
                    "token2char": token2char,
                    "char2token": dict(char2token),
                    "tokenized_span": tokenized_span,
                }
                if partition is not None:
                    metadata["partition"] = partition

                if char2word is not None and word2char is not None:
                    metadata["char2word"] = char2word
                    metadata["word2char"] = word2char

                    word2token = []
                    for char_start, char_end in word2char:
                        token_start = char2token[char_start][0]
                        token_end = char2token[char_end - 1][-1] + 1
                        word2token.append((token_start, token_end))

                    token2word = {}
                    for word_idx, (token_start, token_end) in enumerate(word2token):
                        for token_idx in range(token_start, token_end):
                            token2word[token_idx] = word_idx
                    metadata["word2token"] = word2token
                    metadata["token2word"] = token2word

                task_encoding = TaskEncoding(
                    document=document,
                    inputs=inputs,
                    metadata=metadata,
                )
                task_encodings.append(task_encoding)

        return task_encodings

    def _is_valid_annotation(
        self,
        annotation: Annotation,
        partition: Optional[Span] = None,
        tokenized_span: Optional[Span] = None,
    ) -> bool:
        if isinstance(annotation, BinaryRelation):
            excluded_rel_names = set(
                self.exclude_annotation_names.get(self.relation_layer_name, [])
            )
            return (
                annotation.label not in excluded_rel_names
                and self._is_valid_annotation(
                    annotation.head, partition=partition, tokenized_span=tokenized_span
                )
                and self._is_valid_annotation(
                    annotation.tail, partition=partition, tokenized_span=tokenized_span
                )
            )
        elif isinstance(annotation, LabeledSpan):
            excluded_names = set(self.exclude_annotation_names.get(self.span_layer_name, []))
            return (
                _span_is_in_partition(span=annotation, partition=partition)
                and _span_is_in_partition(span=annotation, partition=tokenized_span)
                and annotation.label not in excluded_names
            )
        elif isinstance(annotation, Span):
            return _span_is_in_partition(
                span=annotation, partition=partition
            ) and _span_is_in_partition(span=annotation, partition=tokenized_span)
        else:
            raise Exception(f"annotation has unknown type: {annotation}")

    def encode_target(
        self, task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding]
    ) -> Optional[TargetEncoding]:
        document = task_encoding.document

        partition = task_encoding.metadata.get("partition", None)
        tokenized_span = task_encoding.metadata["tokenized_span"]
        valid_relations = [
            rel
            for rel in document[self.relation_layer_name]
            if self._is_valid_annotation(rel, partition=partition, tokenized_span=tokenized_span)
        ]
        valid_spans = [
            span
            for span in document[self.span_layer_name]
            if self._is_valid_annotation(span, partition=partition, tokenized_span=tokenized_span)
        ]

        tgt_tokens = self.annotation_encoder_decoder.encode(
            layers={"span": valid_spans, "relation": valid_relations},
            metadata=task_encoding.metadata,
        )

        if self.max_target_length is not None and len(tgt_tokens) > self.max_target_length:
            raise ValueError(
                f"target length {len(tgt_tokens)} exceeds max_target_length {self.max_target_length}"
            )

        result = {
            "tgt_tokens": tgt_tokens,
            "tgt_seq_len": len(tgt_tokens),
        }
        if self.create_constraints:
            CPM_tag = CPM_prepare(
                src_len=task_encoding.inputs["src_seq_len"],
                # strip the bos token
                target=tgt_tokens[1:],
                # shift=target_shift,
                shift=self.pointer_offset,
                span_ids=self.span_ids,
                relation_ids=self.relation_ids,
                none_ids=self.none_ids,
            )
            assert CPM_tag is not None
            result["CPM_tag"] = CPM_tag
        self.maybe_log_example(task_encoding=task_encoding, targets=result)
        return result

    def _pad_values(self, values: List[List], name: str, strategy: str = "longest"):
        if name not in self.pad_values:
            return values
        if not isinstance(values, list):
            return values
        if strategy != "longest":
            raise ValueError(f"unknown padding strategy: {strategy}")
        pad_value = self.pad_values[name]
        tensor_list = [torch.tensor(value_list) for value_list in values]
        shape_lists = list(zip(*[t.shape for t in tensor_list]))
        max_shape = [max(dims) for dims in shape_lists]
        padded = [
            _pad_tensor(tensor=t, target_shape=max_shape, pad_value=pad_value)
            for i, t in enumerate(tensor_list)
        ]
        return torch.stack(padded)

    def _to_tensor(
        self, values: Union[List, torch.Tensor], name: str
    ) -> Union[torch.Tensor, List]:
        if name not in self.dtypes:
            return values
        if not isinstance(values, torch.Tensor):
            tensor = torch.Tensor(values)
        else:
            tensor = values
        tensor = tensor.to(dtype=self.dtypes[name])
        return tensor

    def _prepare_values(self, values: List, name: str) -> Union[torch.Tensor, List]:
        maybe_padded = self._pad_values(values=values, name=name)
        maybe_tensor = self._to_tensor(values=maybe_padded, name=name)
        return maybe_tensor

    def collate(
        self, task_encodings: Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]
    ) -> TaskBatchEncoding:
        if len(task_encodings) == 0:
            raise ValueError("no task_encodings available")
        inputs = {
            k: self._prepare_values(values=v, name=k)
            for k, v in ld2dl(task_encodings, getter=lambda x: x.inputs).items()
        }

        targets = None
        if task_encodings[0].has_targets:
            targets = {
                k: self._prepare_values(values=v, name=k)
                for k, v in ld2dl(task_encodings, getter=lambda x: x.targets).items()
            }

        return inputs, targets

    def unbatch_output(self, model_output: ModelBatchOutput) -> Sequence[TaskOutput]:
        # model_output just contains "pred"
        pred = model_output["pred"]
        batch_size = pred.size(0)
        result = [pred[i].to(device="cpu") for i in range(batch_size)]
        return result

    def create_annotations_from_output(
        self,
        task_encoding: TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
        task_output: TaskOutput,
    ) -> Iterator[Tuple[str, Annotation]]:
        layers, _errors = self.annotation_encoder_decoder.decode(
            targets=task_output.tolist(), metadata=task_encoding.metadata
        )

        for span in layers["span"]:
            yield self.span_layer_name, span

        for rel in layers["relation"]:
            yield self.relation_layer_name, rel
