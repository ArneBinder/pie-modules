from __future__ import annotations

import logging
from typing import TypeVar

from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.core import Annotation, AnnotationList, Document

from pie_modules.annotations import LabeledMultiSpan

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


def get_relation_args(relation: Annotation) -> tuple[Annotation, ...]:
    if isinstance(relation, BinaryRelation):
        return relation.head, relation.tail
    else:
        raise TypeError(
            f"relation {relation} has unknown type [{type(relation)}], cannot get arguments from it"
        )


def sort_annotations(annotations: tuple[Annotation, ...]) -> tuple[Annotation, ...]:
    if len(annotations) <= 1:
        return annotations
    if all(isinstance(ann, LabeledSpan) for ann in annotations):
        return tuple(sorted(annotations, key=lambda ann: (ann.start, ann.end, ann.label)))
    elif all(isinstance(ann, LabeledMultiSpan) for ann in annotations):
        return tuple(sorted(annotations, key=lambda ann: (ann.slices, ann.label)))
    else:
        raise TypeError(
            f"annotations {annotations} have unknown types [{set(type(ann) for ann in annotations)}], "
            f"cannot sort them"
        )


def construct_relation_with_new_args(
    relation: Annotation, new_args: tuple[Annotation, ...]
) -> BinaryRelation:
    if isinstance(relation, BinaryRelation):
        return BinaryRelation(
            head=new_args[0],
            tail=new_args[1],
            label=relation.label,
            score=relation.score,
        )
    else:
        raise TypeError(
            f"original relation {relation} has unknown type [{type(relation)}], "
            f"cannot reconstruct it with new arguments"
        )


class RelationArgumentSorter:
    """Sorts the arguments of the relations in the given relation layer. The sorting is done by the
    start and end positions of the arguments. The relations with the same sorted arguments are
    merged into one relation.

    Args:
        relation_layer: the name of the relation layer
        label_whitelist: if not None, only the relations with the label in the whitelist are sorted
        verbose: if True, log warnings for relations with sorted arguments that are already present
    """

    def __init__(
        self,
        relation_layer: str,
        label_whitelist: list[str] | None = None,
        verbose: bool = True,
    ):
        self.relation_layer = relation_layer
        self.label_whitelist = label_whitelist
        self.verbose = verbose

    def __call__(self, doc: D) -> D:
        rel_layer: AnnotationList[BinaryRelation] = doc[self.relation_layer]
        args2relations: dict[tuple[LabeledSpan, ...], BinaryRelation] = {
            get_relation_args(rel): rel for rel in rel_layer
        }

        old2new_annotations = {}
        new_annotations = []
        for args, rel in args2relations.items():
            if self.label_whitelist is not None and rel.label not in self.label_whitelist:
                # just add the relations whose label is not in the label whitelist (if a whitelist is present)
                old2new_annotations[rel._id] = rel.copy()
                new_annotations.append(old2new_annotations[rel._id])
            else:
                args_sorted = sort_annotations(args)
                if args == args_sorted:
                    # if the relation args are already sorted, just add the relation
                    old2new_annotations[rel._id] = rel.copy()
                    new_annotations.append(old2new_annotations[rel._id])
                else:
                    if args_sorted not in args2relations:
                        old2new_annotations[rel._id] = construct_relation_with_new_args(
                            rel, args_sorted
                        )
                        new_annotations.append(old2new_annotations[rel._id])
                    else:
                        prev_rel = args2relations[args_sorted]
                        if prev_rel.label != rel.label:
                            raise ValueError(
                                f"there is already a relation with sorted args {args_sorted} "
                                f"but with a different label: {prev_rel.label} != {rel.label}"
                            )
                        else:
                            logger.warning(
                                f"do not add the new relation with sorted arguments, because it is already there: "
                                f"{prev_rel}"
                            )
                            # we use the previous relation with sorted arguments to re-map any annotations that
                            # depend on the current relation
                            old2new_annotations[rel._id] = prev_rel.copy()

        result = doc.copy(with_annotations=False)
        result[self.relation_layer].extend(new_annotations)
        result.add_all_annotations_from_other(
            doc,
            override_annotations={self.relation_layer: old2new_annotations},
            verbose=self.verbose,
            strict=True,
        )
        return result
