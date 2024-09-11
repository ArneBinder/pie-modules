import dataclasses

from pytorch_ie import AnnotationLayer, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import (
    TextBasedDocument,
    TextDocumentWithLabeledPartitions,
    TextDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansAndLabeledPartitions,
)


@dataclasses.dataclass
class WithTextPair:
    text_pair: str


@dataclasses.dataclass
class WithLabeledSpansPair(WithTextPair):
    labeled_spans_pair: AnnotationLayer[LabeledSpan] = annotation_field(target="text_pair")


@dataclasses.dataclass
class WithLabeledPartitionsPair(WithTextPair):
    labeled_partitions_pair: AnnotationLayer[LabeledSpan] = annotation_field(target="text_pair")


@dataclasses.dataclass
class TextPairBasedDocument(TextBasedDocument, WithTextPair):
    pass


@dataclasses.dataclass
class TextPairDocumentWithLabeledPartitions(
    WithLabeledPartitionsPair, TextPairBasedDocument, TextDocumentWithLabeledPartitions
):
    pass


@dataclasses.dataclass
class TextPairDocumentWithLabeledSpans(
    WithLabeledSpansPair, TextPairBasedDocument, TextDocumentWithLabeledSpans
):
    pass


@dataclasses.dataclass
class TextPairDocumentWithLabeledSpansAndLabeledPartitions(
    TextPairDocumentWithLabeledPartitions,
    TextPairDocumentWithLabeledSpans,
    TextDocumentWithLabeledSpansAndLabeledPartitions,
):
    pass


@dataclasses.dataclass(eq=True, frozen=True)
class BinaryCorefRelation(BinaryRelation):
    label: str = "coref"


@dataclasses.dataclass
class TextPairDocumentWithLabeledSpansAndBinaryCorefRelations(
    TextPairDocumentWithLabeledSpans, TextDocumentWithLabeledSpans
):
    binary_coref_relations: AnnotationLayer[BinaryCorefRelation] = annotation_field(
        targets=["labeled_spans", "labeled_spans_pair"]
    )


@dataclasses.dataclass
class TextPairDocumentWithLabeledSpansSimilarityRelationsAndLabeledPartitions(
    TextPairDocumentWithLabeledSpansAndLabeledPartitions,
    TextPairDocumentWithLabeledSpansAndBinaryCorefRelations,
):
    pass
