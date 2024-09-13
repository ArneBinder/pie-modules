from collections import defaultdict
from typing import Iterable

from pie_modules.document.types import (
    BinaryCorefRelation,
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations,
)


def add_negative_relations(
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
    for text in sorted(text2spans):
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
