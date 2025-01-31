import functools
import json
import logging
from collections import defaultdict
from copy import copy, deepcopy
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pytorch_ie.core import Annotation
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument
from transformers import PreTrainedTokenizer

from pie_modules.annotations import MultiSpan, Span
from pie_modules.utils import resolve_type

logger = logging.getLogger(__name__)

ToD = TypeVar("ToD", bound=TokenBasedDocument)
TeD = TypeVar("TeD", bound=TextBasedDocument)


def find_token_offset_mapping(text: str, tokens: Iterable[str]) -> List[Tuple[int, int]]:
    """Find the token offset mapping for a given text and tokens. If a token is not found in the
    text, the token offset mapping will be (idx, idx) for this token. So, this works also if
    special tokens are not part of the text.

    Args:
        text (str): The text.
        tokens (Iterable[str]): The tokens.

    Returns:
        List[Tuple[int, int]]: The token offset mapping.
    """

    token_offset_mapping = []
    start = 0
    for token in tokens:
        new_start = text.find(token, start)
        if new_start == -1:
            token_offset_mapping.append((start, start))
            continue
        end = new_start + len(token)
        token_offset_mapping.append((new_start, end))
        start = end
    return token_offset_mapping


def get_stripped_offsets(start: int, end: int, string: str) -> Tuple[int, int]:
    """Get the stripped offsets of a span in a string, i.e. the start and end index of the span
    without leading and trailing whitespaces. If the span is only whitespaces, a tuple is returned
    where start > end.

    Args:
        start (int): The start index.
        end (int): The end index.
        string (str): The string.

    Returns:
        Tuple[int, int]: The stripped offsets.
    """
    span_str = string[start:end]
    left_offset = len(span_str) - len(span_str.lstrip())
    right_offset = len(span_str) - len(span_str.rstrip())

    return start + left_offset, end - right_offset


def char_span_to_token_span(
    span: Annotation, char_to_token: Callable[[int], Optional[int]], strip_span: bool = False
) -> Optional[Union[Span, MultiSpan]]:
    if isinstance(span, Span):
        if strip_span:
            base_text = span.targets[0]
            if not isinstance(base_text, str):
                raise TypeError(
                    f"The first target of a text targeting span must be a string, but found {type(base_text)} as first "
                    f"target type. Can not convert the span {span}."
                )
            char_start, char_end = get_stripped_offsets(span.start, span.end, base_text)
        else:
            char_start, char_end = span.start, span.end
        # we can not convert empty and invalid spans
        if char_start >= char_end:
            return None
        start_token_idx = char_to_token(char_start)
        end_token_idx_inclusive = char_to_token(char_end - 1)
        if start_token_idx is None or end_token_idx_inclusive is None:
            return None
        return span.copy(start=start_token_idx, end=end_token_idx_inclusive + 1)
    elif isinstance(span, MultiSpan):
        if strip_span:
            base_text = span.targets[0]
            if not isinstance(base_text, str):
                raise TypeError(
                    f"The first target of a text targeting span must be a string, but found {type(base_text)} as first "
                    f"target type. Can not convert the span {span}."
                )
            stripped_slices = [
                get_stripped_offsets(start, end, base_text) for start, end in span.slices
            ]
        else:
            stripped_slices = span.slices
        # remove empty and invalid slices
        stripped_slices = [(start, end) for start, end in stripped_slices if start < end]
        if len(stripped_slices) == 0:
            return None
        slices_inclusive_end = [
            (char_to_token(start), char_to_token(end - 1)) for start, end in stripped_slices
        ]
        if any(start is None or end is None for start, end in slices_inclusive_end):
            return None
        return span.copy(
            slices=tuple(
                # ignore type because we checked that start and end are not None
                (start, inclusive_end + 1)  # type: ignore
                for start, inclusive_end in slices_inclusive_end
            )
        )
    else:
        raise TypeError(
            f"can not convert layers that target the text but contain non-span annotations, but found {type(span)}"
        )


def token_span_to_char_span(
    span: Annotation, token_offset_mapping: List[Tuple[int, int]]
) -> Optional[Union[Span, MultiSpan]]:
    if isinstance(span, Span):
        start_char_idx = token_offset_mapping[span.start][0]
        end_char_idx = token_offset_mapping[span.end - 1][1]
        return span.copy(start=start_char_idx, end=end_char_idx)
    elif isinstance(span, MultiSpan):
        slices = [
            (token_offset_mapping[start][0], token_offset_mapping[end - 1][1])
            for start, end in span.slices
        ]
        return span.copy(slices=slices)
    else:
        raise TypeError(
            f"can not convert layers that target the tokens but contain non-span annotations, but found {type(span)}"
        )


def span_sort_key(span: Union[Span, MultiSpan]) -> Tuple[int, ...]:
    if isinstance(span, Span):
        return span.start, span.end
    elif isinstance(span, MultiSpan):
        result: List[int] = []
        for start, end in span.slices:
            result.extend((start, end))
        return tuple(result)
    else:
        raise TypeError(f"can not sort {type(span)}")


def text_based_document_to_token_based(
    doc: TextBasedDocument,
    result_document_type: Union[Type[ToD], str],
    tokens: Optional[List[str]] = None,
    token_offset_mapping: Optional[List[Tuple[int, int]]] = None,
    char_to_token: Optional[Callable[[int], Optional[int]]] = None,
    strip_spans: bool = False,
    strict_span_conversion: bool = True,
    verbose: bool = True,
    added_annotations: Optional[Dict[str, Dict[Annotation, Annotation]]] = None,
) -> ToD:
    """Convert a text based document to a token based document. Uses either tokens,
    token_offset_mapping or char_to_token (provided explicitly or from the document metadata) to
    convert the annotations that target the text and all that depend on these (also adds all
    remaining annotations).

    Args:
        doc (TextBasedDocument): The text based document.
        result_document_type (Union[Type[ToD], str]): The type of the token based document.
        tokens (Optional[List[str]], optional): The tokens. If None, the tokens are taken from the
            metadata. Defaults to None.
        token_offset_mapping (Optional[List[Tuple[int, int]]], optional): The token offset mapping.
            If None, the token offset mapping is taken from the metadata. Defaults to None.
        char_to_token (Optional[Callable[[int], Optional[int]]], optional): The char to token function.
            If None, the char to token function is constructed from the token offset mapping. Defaults
            to None.
        strict_span_conversion (bool, optional): If True, raise an error if not all annotations can
            be converted to token based documents. Defaults to True.
        strip_spans (bool, optional): If True, strip the whitespace from the character spans before
            converting them to token spans. Defaults to False.
        verbose (bool, optional): If True, log warnings if annotations can not be converted. Defaults
            to True.
        added_annotations (Optional[Dict[str, Dict[Annotation, Annotation]]], optional): Pass an empty
            dictionary to collect the added annotations. Defaults to None.

    Returns:
        ToD: The token based document of type result_document_type with the converted annotations.
    """

    document_type = resolve_type(
        type_or_str=result_document_type, expected_super_type=TokenBasedDocument
    )

    metadata = deepcopy(doc.metadata)

    if tokens is None:
        tokens = doc.metadata.get("tokens")
    elif "tokens" in metadata and metadata["tokens"] != tokens:
        logger.warning("tokens in metadata are different from new tokens, take the new tokens")

    # save text, token_offset_mapping and char_to_token (if available) in metadata
    metadata["text"] = doc.text
    token_offset_mapping_lists: Optional[List[List[int]]]
    if token_offset_mapping is None:
        token_offset_mapping_lists = metadata.get("token_offset_mapping")
        if token_offset_mapping_lists is not None:
            token_offset_mapping = [tuple(offsets) for offsets in token_offset_mapping_lists]  # type: ignore
    else:
        # convert offset tuples to lists because serialization and deserialization again
        # will produce lists in any way (json does not know tuples)
        token_offset_mapping_lists = [list(offsets) for offsets in token_offset_mapping]
        if (
            "token_offset_mapping" in metadata
            and metadata["token_offset_mapping"] != token_offset_mapping_lists
        ):
            logger.warning(
                "token_offset_mapping in metadata is different from the new token_offset_mapping, "
                "overwrite the metadata"
            )
        metadata["token_offset_mapping"] = token_offset_mapping_lists

    if tokens is None:
        if token_offset_mapping is not None:
            tokens = [doc.text[start:end] for start, end in token_offset_mapping]
        else:
            raise ValueError(
                "tokens or token_offset_mapping must be provided to convert a text based document to token based, "
                "but got None for both"
            )

    if char_to_token is None:
        char_to_token = metadata.get("char_to_token")
    else:
        if "char_to_token" in metadata and metadata["char_to_token"] != char_to_token:
            logger.warning(
                "char_to_token in metadata is different from the new char_to_token, overwrite the metadata"
            )
        metadata["char_to_token"] = char_to_token

    # construct the char_to_token function, if not provided, from the token_offset_mapping
    if char_to_token is None:
        if token_offset_mapping is None:
            token_offset_mapping = find_token_offset_mapping(text=doc.text, tokens=tokens)
        char_to_token_dict: Dict[int, int] = {}
        for token_idx, (start, end) in enumerate(token_offset_mapping):
            for char_idx in range(start, end):
                char_to_token_dict[char_idx] = token_idx

        def char_to_token(char_idx: int) -> Optional[int]:
            return char_to_token_dict.get(char_idx)

    result = document_type(tokens=tuple(tokens), id=doc.id, metadata=metadata)

    text_targeting_layers = [
        annotation_field.name
        for annotation_field in doc.annotation_fields()
        if "text" in annotation_field.metadata["targets"]
    ]

    override_annotations: Dict[str, Dict[int, Annotation]] = {}
    removed_annotations: Dict[str, Set[int]] = defaultdict(set)
    for text_targeting_layer_name in text_targeting_layers:
        override_annotations[text_targeting_layer_name] = {}
        for char_span in doc[text_targeting_layer_name]:
            token_span = char_span_to_token_span(char_span, char_to_token, strip_spans)
            if token_span is None:
                if strict_span_conversion:
                    raise ValueError(
                        f'cannot find token span for character span: "{char_span}", text="{doc.text}", '
                        f"token_offset_mapping={token_offset_mapping}"
                    )
                else:
                    if verbose:
                        logger.warning(
                            f'cannot find token span for character span "{char_span}", skip it (disable this '
                            f"warning with verbose=False)"
                        )
                removed_annotations[text_targeting_layer_name].add(char_span._id)
            else:
                override_annotations[text_targeting_layer_name][char_span._id] = token_span
                if added_annotations is not None:
                    added_annotations.setdefault(text_targeting_layer_name, {})[
                        char_span
                    ] = token_span
        valid_spans = set(override_annotations[text_targeting_layer_name].values())
        result[text_targeting_layer_name].extend(sorted(valid_spans, key=span_sort_key))

    added_annotations_from_remaining_layers = result.add_all_annotations_from_other(
        doc,
        override_annotations=override_annotations,
        removed_annotations=removed_annotations,
        strict=strict_span_conversion,
        verbose=verbose,
    )
    if added_annotations is not None:
        for layer_name, orig_ann_id2new_ann in added_annotations_from_remaining_layers.items():
            ann_id2ann = {
                ann._id: ann for ann in list(doc[layer_name]) + list(doc[layer_name].predictions)
            }
            annotation_mapping = {
                ann_id2ann[orig_ann_id]: new_ann
                for orig_ann_id, new_ann in orig_ann_id2new_ann.items()
            }
            added_annotations.setdefault(layer_name, {}).update(annotation_mapping)

    return result


def token_based_document_to_text_based(
    doc: TokenBasedDocument,
    result_document_type: Union[Type[TeD], str],
    text: Optional[str] = None,
    token_offset_mapping: Optional[List[Tuple[int, int]]] = None,
    join_tokens_with: Optional[str] = None,
    strict_span_conversion: bool = True,
    verbose: bool = True,
    added_annotations: Optional[Dict[str, Dict[Annotation, Annotation]]] = None,
) -> TeD:
    """Convert a token based document to a text based document. Uses either text,
    token_offset_mapping or char_to_token (provided explicitly or from the document metadata) to
    convert the annotations that target the tokens and all that depend on these (also adds all
    remaining annotations).

    Args:
        doc (TokenBasedDocument): The token based document.
        result_document_type (Union[Type[TeD], str]): The type of the text based document.
        text (Optional[str], optional): The text. If None, constructed from the tokens (requires
            join_tokens_with) or taken from the metadata. Defaults to None.
        token_offset_mapping (Optional[List[Tuple[int, int]]], optional): The token offset mapping.
            If None, the token offset mapping is constructed from the tokens (requires join_tokens_with)
            or taken from the metadata. Defaults to None.
        join_tokens_with (Optional[str], optional): The token separator. If no text is provided, the
            text and token offset mapping are constructed from the tokens by joining them with this
            separator. Defaults to None.
        strict_span_conversion (bool, optional): If True, raise an error if not all annotations
            can be converted to text based documents. Defaults to True.
        verbose (bool, optional): If True, log warnings if annotations can not be converted.
            Defaults to True.
        added_annotations (Optional[Dict[str, Dict[Annotation, Annotation]]], optional): Pass an
            empty dictionary to collect the added annotations. Defaults to None.

    Returns:
        TeD: The text based document of type result_document_type with the converted annotations.
    """

    document_type = resolve_type(
        type_or_str=result_document_type, expected_super_type=TextBasedDocument
    )

    # if a token_separator is provided, we construct the text from the tokens
    if text is None and join_tokens_with is not None:
        start = 0
        token_offset_mapping = []
        tokens = doc.tokens
        for token in tokens:
            end = start + len(token)
            token_offset_mapping.append((start, end))
            # we add the separator after each token
            start = end + len(join_tokens_with)
        text = join_tokens_with.join(tokens)

    # otherwise we try to use the text from the metadata
    if text is None:
        text = doc.metadata.get("text")

    if text is None:
        raise ValueError(
            "if join_tokens_with is None, text must be provided, but got None as well"
        )

    token_offset_mapping_lists = (
        doc.metadata.get("token_offset_mapping")
        if token_offset_mapping is None
        else token_offset_mapping
    )
    if token_offset_mapping_lists is None:
        token_offset_mapping = find_token_offset_mapping(text=text, tokens=doc.tokens)
    else:
        # we convert the token_offset_mapping to tuples because the token_offset_mapping
        # in the metadata is a list of lists, but we need a list of tuples
        token_offset_mapping = [tuple(offsets) for offsets in token_offset_mapping_lists]  # type: ignore

    result = document_type(text=text, id=doc.id, metadata=deepcopy(doc.metadata))
    result.metadata["tokens"] = list(doc.tokens)
    # convert offset tuples to lists because serialization and deserialization again
    # will produce lists in any way (json does not know tuples)
    token_offset_mapping_lists = [list(offsets) for offsets in token_offset_mapping]
    if (
        "token_offset_mapping" in doc.metadata
        and doc.metadata["token_offset_mapping"] != token_offset_mapping_lists
    ):
        logger.warning(
            "token_offset_mapping in metadata is different from the new token_offset_mapping, "
            "overwrite the metadata"
        )
    result.metadata["token_offset_mapping"] = token_offset_mapping_lists

    token_targeting_layers = [
        annotation_field.name
        for annotation_field in doc.annotation_fields()
        if "tokens" in annotation_field.metadata["targets"]
    ]

    override_annotations: Dict[str, Dict[int, Annotation]] = {}
    removed_annotations: Dict[str, Set[int]] = defaultdict(set)
    for token_targeting_layer_name in token_targeting_layers:
        override_annotations[token_targeting_layer_name] = {}
        for token_span in doc[token_targeting_layer_name]:
            char_span = token_span_to_char_span(token_span, token_offset_mapping)
            override_annotations[token_targeting_layer_name][token_span._id] = char_span
            if added_annotations is not None:
                added_annotations.setdefault(token_targeting_layer_name, {})[
                    token_span
                ] = char_span
        valid_spans = set(override_annotations[token_targeting_layer_name].values())
        result[token_targeting_layer_name].extend(sorted(valid_spans, key=span_sort_key))

    added_annotations_from_remaining_layers = result.add_all_annotations_from_other(
        doc,
        override_annotations=override_annotations,
        removed_annotations=removed_annotations,
        strict=strict_span_conversion,
        verbose=verbose,
    )
    if added_annotations is not None:
        for layer_name, orig_ann_id2new_ann in added_annotations_from_remaining_layers.items():
            ann_id2ann = {
                ann._id: ann for ann in list(doc[layer_name]) + list(doc[layer_name].predictions)
            }
            annotation_mapping = {
                ann_id2ann[orig_ann_id]: new_ann
                for orig_ann_id, new_ann in orig_ann_id2new_ann.items()
            }
            added_annotations.setdefault(layer_name, {}).update(annotation_mapping)

    return result


def tokenize_document(
    doc: TextBasedDocument,
    tokenizer: PreTrainedTokenizer,
    result_document_type: Type[ToD],
    partition_layer: Optional[str] = None,
    strip_spans: bool = False,
    strict_span_conversion: bool = True,
    added_annotations: Optional[List[Dict[str, Dict[Annotation, Annotation]]]] = None,
    verbose: bool = True,
    **tokenize_kwargs,
) -> List[ToD]:
    """Tokenize a document with a given tokenizer and return a list of token based documents. The
    document is tokenized in partitions if a partition layer is provided. The annotations that
    target the text are converted to target the tokens and also all dependent annotations are
    converted.

    Args:
        doc (TextBasedDocument): The document to tokenize.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        result_document_type (Type[ToD]): The exact type of the token based documents.
        partition_layer (Optional[str], optional): The layer to use for partitioning the document. If None, the whole
            document is tokenized. Defaults to None.
        strip_spans (bool, optional): If True, strip the whitespace from the character spans before converting them to
            token spans. Defaults to False.
        strict_span_conversion (bool, optional): If True, raise an error if not all annotations can be converted to
            token based documents. Defaults to True.
        added_annotations (Optional[List[Dict[str, Dict[Annotation, Annotation]]]], optional): Pass an empty list to
            collect the added annotations. Defaults to None.
        verbose (bool, optional): If True, log warnings if annotations can not be converted. Defaults to True.

    Returns:
        List[ToD]: The token based documents of type result_document_type with the converted annotations.
    """

    added_annotation_lists: Dict[str, List[Annotation]] = defaultdict(list)
    result = []
    partitions: Iterable[Span]
    if partition_layer is None:
        partitions = [Span(start=0, end=len(doc.text))]
    else:
        partitions = doc[partition_layer]
    for partition in partitions:
        text = doc.text[partition.start : partition.end]
        current_tokenize_kwargs = copy(tokenize_kwargs)
        if "text" in tokenize_kwargs:
            current_tokenize_kwargs["text_pair"] = text
            sequence_index = 1
        else:
            current_tokenize_kwargs["text"] = text
            sequence_index = 0
        tokenized_text = tokenizer(**current_tokenize_kwargs)
        for batch_encoding in tokenized_text.encodings:
            token_offset_mapping = batch_encoding.offsets
            char_to_token: Optional[Callable[[int], Optional[int]]]
            char_to_token = functools.partial(
                batch_encoding.char_to_token, sequence_index=sequence_index
            )
            token_offset_mapping = [
                offsets if s_id == sequence_index else (0, 0)
                for s_id, offsets in zip(batch_encoding.sequence_ids, token_offset_mapping)
            ]
            if partition.start > 0:
                token_offset_mapping = [
                    (start + partition.start, end + partition.start)
                    for start, end in token_offset_mapping
                ]
                char_to_token = None
            current_added_annotations: Dict[str, Dict[Annotation, Annotation]] = defaultdict(dict)
            tokenized_document = text_based_document_to_token_based(
                doc,
                tokens=batch_encoding.tokens,
                result_document_type=result_document_type,
                token_offset_mapping=token_offset_mapping,
                char_to_token=char_to_token,
                strict_span_conversion=False,
                strip_spans=strip_spans,
                verbose=False,
                added_annotations=current_added_annotations,
            )
            tokenized_document.metadata["tokenizer_encoding"] = batch_encoding
            result.append(tokenized_document)
            for k, v in current_added_annotations.items():
                added_annotation_lists[k].extend(v)
            if added_annotations is not None:
                added_annotations.append(current_added_annotations)

    missed_annotations = defaultdict(set)
    if strict_span_conversion or verbose:
        # We check the annotations with respect to the layers of the result_document_type.
        # Note that the original document may have more layers, but since result documents
        # are of type result_document_type, we only check the layers of this type.
        for annotation_field in result_document_type.annotation_fields():
            # do not check the partition layer because the partitions are not required later on
            # and entries get quite probably removed when windowing is applied, so this just pollutes the logs
            if annotation_field.name != partition_layer:
                current_missed_annotations = set(doc[annotation_field.name]) - set(
                    added_annotation_lists[annotation_field.name]
                )
                if len(current_missed_annotations) > 0:
                    missed_annotations[annotation_field.name] = current_missed_annotations

    if len(missed_annotations) > 0:
        missed_annotations_simplified = {k: str(v) for k, v in missed_annotations.items()}
        if strict_span_conversion:
            raise ValueError(
                f"could not convert all annotations from document with id={doc.id} to token based documents, "
                f"but strict_span_conversion is True, so raise an error, "
                f"missed annotations:\n{json.dumps(missed_annotations_simplified, sort_keys=True, indent=2)}"
            )
        else:
            if verbose:
                logger.warning(
                    f"could not convert all annotations from document with id={doc.id} to token based documents, "
                    f"missed annotations (disable this message with verbose=False):\n"
                    f"{json.dumps(missed_annotations_simplified, sort_keys=True, indent=2)}"
                )

    return result
