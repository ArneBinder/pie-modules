import copy
import logging
import random
from collections import defaultdict
from collections.abc import Iterator
from itertools import chain
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar

from pytorch_ie.annotations import LabeledSpan, Span
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansAndBinaryRelations,
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from tqdm import tqdm

from pie_modules.documents import (
    BinaryCorefRelation,
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations,
)
from pie_modules.utils.span import are_nested

logger = logging.getLogger(__name__)

S = TypeVar("S", bound=Span)
S2 = TypeVar("S2", bound=Span)


def _span2partition_mapping(spans: Iterable[S], partitions: Iterable[S2]) -> Dict[S, S2]:
    result = {}
    for span in spans:
        for partition in partitions:
            if are_nested(
                start_end=(span.start, span.end), other_start_end=(partition.start, partition.end)
            ):
                result[span] = partition
                break
    return result


def _span_copy_shifted(span: S, offset: int) -> S:
    return span.copy(start=span.start + offset, end=span.end + offset)


def _construct_text_pair_coref_documents_from_partitions_via_relations(
    document: TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions, relation_label: str
) -> List[TextPairDocumentWithLabeledSpansAndBinaryCorefRelations]:
    span2partition = _span2partition_mapping(
        spans=document.labeled_spans, partitions=document.labeled_partitions
    )
    partition2spans = defaultdict(list)
    for span, partition in span2partition.items():
        partition2spans[partition].append(span)

    texts2docs_and_span_mappings: Dict[
        Tuple[str, str],
        Tuple[
            TextPairDocumentWithLabeledSpansAndBinaryCorefRelations,
            Dict[LabeledSpan, LabeledSpan],
            Dict[LabeledSpan, LabeledSpan],
        ],
    ] = dict()
    result = []
    for rel in document.binary_relations:
        if rel.label != relation_label:
            continue

        if rel.head not in span2partition:
            raise ValueError(f"head not in any partition: {rel.head}")
        head_partition = span2partition[rel.head]
        text = document.text[head_partition.start : head_partition.end]

        if rel.tail not in span2partition:
            raise ValueError(f"tail not in any partition: {rel.tail}")
        tail_partition = span2partition[rel.tail]
        text_pair = document.text[tail_partition.start : tail_partition.end]

        if (text, text_pair) in texts2docs_and_span_mappings:
            new_doc, head_spans_mapping, tail_spans_mapping = texts2docs_and_span_mappings[
                (text, text_pair)
            ]
        else:
            if document.id is not None:
                doc_id = (
                    f"{document.id}[{head_partition.start}:{head_partition.end}]"
                    f"+{document.id}[{tail_partition.start}:{tail_partition.end}]"
                )
            else:
                doc_id = None
            new_doc = TextPairDocumentWithLabeledSpansAndBinaryCorefRelations(
                id=doc_id,
                text=text,
                text_pair=text_pair,
                metadata={
                    "original_doc_id": document.id,
                    "original_doc_id_pair": document.id,
                    "original_doc_span": {
                        "start": head_partition.start,
                        "end": head_partition.end,
                    },
                    "original_doc_span_pair": {
                        "start": tail_partition.start,
                        "end": tail_partition.end,
                    },
                },
            )

            head_spans_mapping = {
                span: _span_copy_shifted(span=span, offset=-head_partition.start)
                for span in partition2spans[head_partition]
            }
            new_doc.labeled_spans.extend(head_spans_mapping.values())

            tail_spans_mapping = {
                span: _span_copy_shifted(span=span, offset=-tail_partition.start)
                for span in partition2spans[tail_partition]
            }
            new_doc.labeled_spans_pair.extend(tail_spans_mapping.values())

            texts2docs_and_span_mappings[(text, text_pair)] = (
                new_doc,
                head_spans_mapping,
                tail_spans_mapping,
            )
            result.append(new_doc)

        coref_rel = BinaryCorefRelation(
            head=head_spans_mapping[rel.head], tail=tail_spans_mapping[rel.tail], score=1.0
        )
        new_doc.binary_coref_relations.append(coref_rel)

    return result


def construct_text_pair_coref_documents_from_partitions_via_relations(
    documents: Iterable[TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions], **kwargs
) -> Iterator[TextPairDocumentWithLabeledSpansAndBinaryCorefRelations]:
    for doc in documents:
        yield from _construct_text_pair_coref_documents_from_partitions_via_relations(
            document=doc, **kwargs
        )


def shift_span(span: S, offset: int) -> S:
    return span.copy(start=span.start + offset, end=span.end + offset)


def construct_text_document_from_text_pair_coref_document(
    document: TextPairDocumentWithLabeledSpansAndBinaryCorefRelations,
    glue_text: str,
    no_relation_label: str,
    relation_label_mapping: Optional[Dict[str, str]] = None,
) -> TextDocumentWithLabeledSpansAndBinaryRelations:
    if document.text == document.text_pair:
        new_doc = TextDocumentWithLabeledSpansAndBinaryRelations(
            id=document.id, metadata=copy.deepcopy(document.metadata), text=document.text
        )
        old2new_spans: Dict[LabeledSpan, LabeledSpan] = {}
        new2new_spans: Dict[LabeledSpan, LabeledSpan] = {}
        for old_span in chain(document.labeled_spans, document.labeled_spans_pair):
            new_span = old_span.copy()
            # when detaching / copying the span, it may be the same as a previous span from the other
            new_span = new2new_spans.get(new_span, new_span)
            new2new_spans[new_span] = new_span
            old2new_spans[old_span] = new_span
    else:
        new_doc = TextDocumentWithLabeledSpansAndBinaryRelations(
            text=document.text + glue_text + document.text_pair,
            id=document.id,
            metadata=copy.deepcopy(document.metadata),
        )
        old2new_spans = {}
        old2new_spans.update({span: span.copy() for span in document.labeled_spans})
        offset = len(document.text) + len(glue_text)
        old2new_spans.update(
            {span: shift_span(span.copy(), offset) for span in document.labeled_spans_pair}
        )

    # sort to make order deterministic
    new_doc.labeled_spans.extend(
        sorted(old2new_spans.values(), key=lambda s: (s.start, s.end, s.label))
    )
    for old_rel in document.binary_coref_relations:
        label = old_rel.label if old_rel.score > 0.0 else no_relation_label
        if relation_label_mapping is not None:
            label = relation_label_mapping.get(label, label)
        new_rel = old_rel.copy(
            head=old2new_spans[old_rel.head],
            tail=old2new_spans[old_rel.tail],
            label=label,
            score=1.0,
        )
        new_doc.binary_relations.append(new_rel)

    return new_doc


def add_negative_coref_relations(
    documents: Iterable[TextPairDocumentWithLabeledSpansAndBinaryCorefRelations],
    max_num_negatives: Optional[int] = None,
    downsampling_factor: Optional[float] = None,
    random_seed: Optional[int] = None,
    enforce_same_original_doc_id: bool = False,
    enforce_different_original_doc_id: bool = False,
) -> Iterable[TextPairDocumentWithLabeledSpansAndBinaryCorefRelations]:
    positive_tuples = defaultdict(set)
    text2spans = defaultdict(set)
    text2original_doc_id = dict()
    text2span = dict()
    for doc in documents:
        for labeled_span in doc.labeled_spans:
            text2spans[doc.text].add(labeled_span.copy())
        for labeled_span in doc.labeled_spans_pair:
            text2spans[doc.text_pair].add(labeled_span.copy())

        for coref in doc.binary_coref_relations:
            positive_tuples[(doc.text, doc.text_pair)].add((coref.head.copy(), coref.tail.copy()))
            positive_tuples[(doc.text_pair, doc.text)].add((coref.tail.copy(), coref.head.copy()))
        text2original_doc_id[doc.text] = doc.metadata.get("original_doc_id")
        text2original_doc_id[doc.text_pair] = doc.metadata.get("original_doc_id_pair")
        text2span[doc.text] = doc.metadata.get("original_doc_span")
        text2span[doc.text_pair] = doc.metadata.get("original_doc_span_pair")

    new_docs = []
    new_rels2new_docs = {}
    positive_rels = []
    negative_rels = []
    for text in tqdm(sorted(text2spans)):
        original_doc_id = text2original_doc_id[text]
        for text_pair in sorted(text2spans):
            original_doc_id_pair = text2original_doc_id[text_pair]
            if enforce_same_original_doc_id:
                if original_doc_id is None or original_doc_id_pair is None:
                    raise ValueError(
                        "enforce_same_original_doc_id is set, but original_doc_id(_pair) is None"
                    )
                if original_doc_id != original_doc_id_pair:
                    continue
            if enforce_different_original_doc_id:
                if original_doc_id is None or original_doc_id_pair is None:
                    raise ValueError(
                        "enforce_different_original_doc_id is set, but original_doc_id(_pair) is None"
                    )
                if original_doc_id == original_doc_id_pair:
                    continue
            current_positives = positive_tuples.get((text, text_pair), set())
            new_doc = TextPairDocumentWithLabeledSpansAndBinaryCorefRelations(
                text=text,
                text_pair=text_pair,
                metadata={
                    "original_doc_id": original_doc_id,
                    "original_doc_id_pair": original_doc_id_pair,
                    "original_doc_span": text2span[text],
                    "original_doc_span_pair": text2span[text_pair],
                },
            )
            new_doc.labeled_spans.extend(labeled_span.copy() for labeled_span in text2spans[text])
            new_doc.labeled_spans_pair.extend(
                labeled_span.copy() for labeled_span in text2spans[text_pair]
            )
            for s in sorted(new_doc.labeled_spans):
                for s_p in sorted(new_doc.labeled_spans_pair):
                    # exclude relations to itself
                    if text == text_pair and s.copy() == s_p.copy():
                        continue
                    if s.label != s_p.label:
                        continue
                    score = 1.0 if (s.copy(), s_p.copy()) in current_positives else 0.0
                    new_coref_rel = BinaryCorefRelation(head=s, tail=s_p, score=score)
                    # new_doc.binary_coref_relations.append(new_coref_rel)
                    new_rels2new_docs[new_coref_rel] = new_doc
                    if score > 0.0:
                        positive_rels.append(new_coref_rel)
                    else:
                        negative_rels.append(new_coref_rel)
            new_docs.append(new_doc)

    for rel in positive_rels:
        new_rels2new_docs[rel].binary_coref_relations.append(rel)

    if max_num_negatives is None:
        # Downsampling of negatives. This requires positive instances!
        if downsampling_factor is not None:
            if len(positive_rels) == 0:
                raise ValueError(
                    f"downsampling [factor={downsampling_factor}] is enabled, "
                    f"but no positive relations are available to calculate max_num_negatives"
                )

            max_num_negatives = int(len(positive_rels) * downsampling_factor)
            if max_num_negatives == 0:
                logger.warning(
                    f"downsampling with factor={downsampling_factor} and number of "
                    f"positive relations={len(positive_rels)} does not produce any negatives"
                )
    elif downsampling_factor is not None:
        raise ValueError(
            f"setting max_num_negatives [{max_num_negatives}] and [{downsampling_factor}] "
            f"simultaneously is ambiguous and not allowed"
        )

    if max_num_negatives is not None:
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(negative_rels)
        negative_rels = negative_rels[:max_num_negatives]

    for rel in negative_rels:
        new_rels2new_docs[rel].binary_coref_relations.append(rel)

    docs_with_rels = [doc for doc in new_docs if len(doc.binary_coref_relations) > 0]
    logger.info(
        f"constructed {len(negative_rels)} negative for {len(positive_rels)} "
        f"positive relations in {len(docs_with_rels)} documents"
    )
    return docs_with_rels
