from collections import defaultdict
from collections.abc import Iterator
from typing import Dict, Iterable, List, Tuple, TypeVar

from pytorch_ie.annotations import LabeledSpan, Span
from pytorch_ie.documents import (
    TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
)
from tqdm import tqdm

from pie_modules.documents import (
    BinaryCorefRelation,
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations,
)
from pie_modules.utils.span import are_nested

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
                id=doc_id, text=text, text_pair=text_pair
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


def add_negative_coref_relations(
    documents: Iterable[TextPairDocumentWithLabeledSpansAndBinaryCorefRelations], **kwargs
) -> Iterable[TextPairDocumentWithLabeledSpansAndBinaryCorefRelations]:
    positive_tuples = defaultdict(set)
    text2spans = defaultdict(set)
    for doc in documents:
        for labeled_span in doc.labeled_spans:
            text2spans[doc.text].add(labeled_span.copy())
        for labeled_span in doc.labeled_spans_pair:
            text2spans[doc.text_pair].add(labeled_span.copy())

        for coref in doc.binary_coref_relations:
            positive_tuples[(doc.text, doc.text_pair)].add((coref.head.copy(), coref.tail.copy()))
            positive_tuples[(doc.text_pair, doc.text)].add((coref.tail.copy(), coref.head.copy()))

    new_docs = []
    for text in tqdm(sorted(text2spans)):
        for text_pair in sorted(text2spans):
            current_positives = positive_tuples.get((text, text_pair), set())
            new_doc = TextPairDocumentWithLabeledSpansAndBinaryCorefRelations(
                text=text, text_pair=text_pair
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
                    new_doc.binary_coref_relations.append(new_coref_rel)
            new_docs.append(new_doc)

    return new_docs
