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

from pytorch_ie.annotations import Span
from pytorch_ie.core import Annotation
from pytorch_ie.documents import TextBasedDocument, TokenBasedDocument
from transformers import PreTrainedTokenizer

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


def text_based_document_to_token_based(
    doc: TextBasedDocument,
    result_document_type: Union[Type[ToD], str],
    tokens: Optional[List[str]] = None,
    token_offset_mapping: Optional[List[Tuple[int, int]]] = None,
    char_to_token: Optional[Callable[[int], Optional[int]]] = None,
    strict_span_conversion: bool = True,
    verbose: bool = True,
    added_annotations: Optional[Dict[str, List[Annotation]]] = None,
) -> ToD:
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
        char_span: Span
        for char_span in doc[text_targeting_layer_name]:
            if not isinstance(char_span, Span):
                raise TypeError(
                    f"can not convert layers that target the text but contain non-span annotations, "
                    f"but found {type(char_span)} in layer {text_targeting_layer_name}"
                )
            start_token_idx = char_to_token(char_span.start)
            end_token_idx_inclusive = char_to_token(char_span.end - 1)
            if start_token_idx is None or end_token_idx_inclusive is None:
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
                token_span = char_span.copy(start=start_token_idx, end=end_token_idx_inclusive + 1)
                override_annotations[text_targeting_layer_name][char_span._id] = token_span
                if added_annotations is not None:
                    added_annotations[text_targeting_layer_name].append(char_span)
        valid_spans = set(override_annotations[text_targeting_layer_name].values())
        result[text_targeting_layer_name].extend(sorted(valid_spans, key=lambda span: span.start))

    added_annotations_from_remaining_layers = result.add_all_annotations_from_other(
        doc,
        override_annotations=override_annotations,
        removed_annotations=removed_annotations,
        strict=strict_span_conversion,
        verbose=verbose,
    )
    if added_annotations is not None:
        for layer_name, annotations in added_annotations_from_remaining_layers.items():
            added_annotations[layer_name].extend(annotations)

    return result


def token_based_document_to_text_based(
    doc: TokenBasedDocument,
    result_document_type: Union[Type[TeD], str],
    text: Optional[str] = None,
    token_offset_mapping: Optional[List[Tuple[int, int]]] = None,
    join_tokens_with: Optional[str] = None,
    strict_span_conversion: bool = True,
    verbose: bool = True,
    added_annotations: Optional[Dict[str, List[Annotation]]] = None,
) -> TeD:
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
            if not isinstance(token_span, Span):
                raise TypeError(
                    f"can not convert layers that target the tokens but contain non-span annotations, "
                    f"but found {type(token_span)} in layer {token_targeting_layer_name}"
                )
            start_char_idx = token_offset_mapping[token_span.start][0]
            end_char_idx = token_offset_mapping[token_span.end - 1][1]

            char_span = token_span.copy(start=start_char_idx, end=end_char_idx)
            override_annotations[token_targeting_layer_name][token_span._id] = char_span
            if added_annotations is not None:
                added_annotations[token_targeting_layer_name].append(token_span)
        valid_spans = set(override_annotations[token_targeting_layer_name].values())
        result[token_targeting_layer_name].extend(sorted(valid_spans, key=lambda span: span.start))

    added_annotations_from_remaining_layers = result.add_all_annotations_from_other(
        doc,
        override_annotations=override_annotations,
        removed_annotations=removed_annotations,
        strict=strict_span_conversion,
        verbose=verbose,
    )
    if added_annotations is not None:
        for layer_name, annotations in added_annotations_from_remaining_layers.items():
            added_annotations[layer_name].extend(annotations)

    return result


def tokenize_document(
    doc: TextBasedDocument,
    tokenizer: PreTrainedTokenizer,
    result_document_type: Type[ToD],
    partition_layer: Optional[str] = None,
    strict_span_conversion: bool = True,
    verbose: bool = True,
    **tokenize_kwargs,
) -> List[ToD]:
    added_annotations: Dict[str, List[Annotation]] = defaultdict(list)
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
            tokenized_document = text_based_document_to_token_based(
                doc,
                tokens=batch_encoding.tokens,
                result_document_type=result_document_type,
                token_offset_mapping=token_offset_mapping,
                char_to_token=char_to_token,
                strict_span_conversion=False,
                verbose=False,
                added_annotations=added_annotations,
            )
            tokenized_document.metadata["tokenizer_encoding"] = batch_encoding
            result.append(tokenized_document)

    missed_annotations = defaultdict(set)
    if strict_span_conversion or verbose:
        for annotation_field in doc.annotation_fields():
            current_missed_annotations = set(doc[annotation_field.name]) - set(
                added_annotations[annotation_field.name]
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
