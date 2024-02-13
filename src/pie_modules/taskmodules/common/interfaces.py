import abc
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from pytorch_ie import Annotation

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
