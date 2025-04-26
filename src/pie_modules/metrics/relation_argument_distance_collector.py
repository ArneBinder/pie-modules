from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union

from pytorch_ie.core import Document, DocumentStatistic
from pytorch_ie.utils.hydra import resolve_target
from transformers import AutoTokenizer, PreTrainedTokenizer

from pie_modules.annotations import BinaryRelation, NaryRelation, Span
from pie_modules.document.processing import tokenize_document
from pie_modules.documents import TextBasedDocument, TokenBasedDocument
from pie_modules.utils.span import distance


class RelationArgumentDistanceCollector(DocumentStatistic):
    """Collects the distances between the arguments of relation annotations. For n-ary relations,
    the distances between all pairs of arguments are collected. The distances can be calculated in
    means of characters or tokens.

    Args:
        layer: The relation annotation layer of the document to collect the distances from.
        distance_type: The type of distance to calculate. Can be "outer", "inner", or "center".
        tokenize: Whether to tokenize the document before calculating the distance.
        tokenizer: The tokenizer to use for tokenization. If a string is provided, the tokenizer is
            loaded from the Hugging Face model hub. Required if tokenize is True.
        tokenized_document_type: The type of document to return after tokenization. Required if
            tokenize is True.
        key_all: The key to use for the aggregation of all values.
    """

    DEFAULT_AGGREGATION_FUNCTIONS = ["len", "mean", "std", "min", "max"]

    def __init__(
        self,
        layer: str,
        distance_type: str = "outer",
        tokenize: bool = False,
        tokenize_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        tokenized_document_type: Optional[Union[str, Type[TokenBasedDocument]]] = None,
        key_all: str = "ALL",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer = layer
        self.distance_type = distance_type
        self.key_all = key_all
        self.tokenize = tokenize
        self.tokenize_kwargs = tokenize_kwargs or {}
        if self.tokenize:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer must be provided to calculate distance in means of tokens"
                )
            if isinstance(tokenizer, str):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.tokenizer = tokenizer
            if tokenized_document_type is None:
                raise ValueError(
                    "tokenized_document_type must be provided to calculate distance in means of tokens"
                )
            self.tokenized_document_type: Type[TokenBasedDocument] = resolve_target(
                tokenized_document_type
            )

    def _collect(self, doc: Document) -> Dict[str, List[float]]:
        if self.tokenize:
            if not isinstance(doc, TextBasedDocument):
                raise ValueError(
                    "doc must be a TextBasedDocument to calculate distance in means of tokens"
                )
            docs = tokenize_document(
                doc,
                tokenizer=self.tokenizer,
                result_document_type=self.tokenized_document_type,
                **self.tokenize_kwargs,
            )
        else:
            docs = [doc]
        values: Dict[str, List[float]] = defaultdict(list)
        for doc in docs:
            layer_obj = getattr(doc, self.layer)

            for binary_relation in layer_obj:
                if isinstance(binary_relation, BinaryRelation):
                    args = [binary_relation.head, binary_relation.tail]
                    label = binary_relation.label
                elif isinstance(binary_relation, NaryRelation):
                    args = binary_relation.arguments
                    label = binary_relation.label
                else:
                    raise TypeError(
                        f"argument distance calculation is not yet supported for {type(binary_relation)}"
                    )
                if any(not isinstance(arg, Span) for arg in args):
                    raise TypeError(
                        "argument distance calculation is not yet supported for arguments other than Spans"
                    )
                # collect distances between all pairs of arguments
                for idx1, arg1 in enumerate(args):
                    for idx2, arg2 in enumerate(args):
                        if idx1 == idx2:
                            continue
                        d = distance(
                            start_end=(arg1.start, arg1.end),
                            other_start_end=(arg2.start, arg2.end),
                            distance_type=self.distance_type,
                        )

                        values[label].append(d)

        if self.key_all in values:
            raise ValueError(
                f'key key_all="{self.key_all}" is reserved for the aggregation of all values. Please '
                f"choose another value for key_all which is not used as a label of any annotation."
            )
        labels = list(values)
        for label in labels:
            values[self.key_all].extend(values[label])

        # improve order of entries for histogram plotting: first, the key_all entry, then the rest,
        # so that not the key_all entry is plotted last and thus covers all other entries
        return {label: values[label] for label in [self.key_all] + labels}
