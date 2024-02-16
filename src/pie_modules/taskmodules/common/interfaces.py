import abc
import logging
from collections import defaultdict
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from pytorch_ie import Annotation

logger = logging.getLogger(__name__)

# Annotation Encoding type: encoding for a single annotation
AE = TypeVar("AE")
# Annotation type
A = TypeVar("A", bound=Annotation)
# Annotation Collection Encoding type: encoding for a collection of annotations,
# e.g. all relevant annotations for a document
ACE = TypeVar("ACE")


class DecodingException(Exception, Generic[AE], abc.ABC):
    """Exception raised when decoding fails."""

    identifier: str

    def __init__(self, message: str, encoding: AE, remaining: Optional[AE] = None):
        self.message = message
        self.encoding = encoding
        self.remaining = remaining


class EncodingException(Exception, Generic[A], abc.ABC):
    """Exception raised when encoding fails."""

    identifier: str

    def __init__(self, message: str, annotation: A):
        self.message = message
        self.annotation = annotation


class AnnotationEncoderDecoder(abc.ABC, Generic[A, AE]):
    """Base class for annotation encoders and decoders."""

    @abc.abstractmethod
    def encode(self, annotation: A, metadata: Optional[Dict[str, Any]] = None) -> AE:
        pass

    @abc.abstractmethod
    def decode(self, encoding: AE, metadata: Optional[Dict[str, Any]] = None) -> A:
        pass


class GenerativeAnnotationEncoderDecoder(AnnotationEncoderDecoder[A, AE], abc.ABC):
    """Base class for generative annotation encoders and decoders."""

    @abc.abstractmethod
    def parse(self, encoding: AE, decoded_annotations: List[A], text_length: int) -> Tuple[A, AE]:
        """Parse the encoding and return the decoded annotation and the remaining encoding."""
        pass


class GenerativeAnnotationEncoderDecoderWithParseWithErrors(
    Generic[A], GenerativeAnnotationEncoderDecoder[A, List[int]], abc.ABC
):
    KEY_INVALID_CORRECT = "correct"

    def parse_with_error_handling(
        self,
        encoding: List[int],
        input_length: int,
        stop_ids: List[int],
        errors: Optional[Dict[str, int]] = None,
        decoded_annotations: Optional[List[A]] = None,
    ) -> Tuple[List[A], Dict[str, int], List[int]]:
        errors = errors or defaultdict(int)
        decoded_annotations = decoded_annotations or []
        valid_encoding: A
        successfully_decoded: List[int] = []
        remaining = encoding
        prev_len = len(remaining)
        while len(remaining) > 0:
            if remaining[0] in stop_ids:
                # we discard everything after any stop id
                break
            try:
                valid_encoding, remaining = self.parse(
                    encoding=remaining,
                    decoded_annotations=decoded_annotations,
                    text_length=input_length,
                )
                decoded_annotations.append(valid_encoding)
                errors[self.KEY_INVALID_CORRECT] += 1
                successfully_decoded = encoding[: len(encoding) - len(remaining)]
            except DecodingException as e:
                if e.remaining is None:
                    raise ValueError(f"decoding exception did not return remaining encoding: {e}")
                errors[e.identifier] += 1
                remaining = e.remaining

            # if we did not consume any ids, we discard the first remaining one
            if len(remaining) == prev_len:
                logger.warning(
                    f"parse did not consume any ids, discarding first id from {remaining}"
                )
                remaining = remaining[1:]
            prev_len = len(remaining)

        return decoded_annotations, dict(errors), encoding[len(successfully_decoded) :]
