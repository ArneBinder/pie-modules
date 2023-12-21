import abc
import logging
from typing import Any, Dict, Generic, Optional, TypeVar

from pytorch_ie import Annotation

logger = logging.getLogger(__name__)

# Annotation Encoding type: encoding for a single annotation
AE = TypeVar("AE")
# Annotation type
A = TypeVar("A", bound=Annotation)


class AnnotationEncoderDecoder(abc.ABC, Generic[A, AE]):
    """Base class for annotation encoders and decoders."""

    @abc.abstractmethod
    def encode(self, annotation: A, metadata: Optional[Dict[str, Any]] = None) -> Optional[AE]:
        pass

    @abc.abstractmethod
    def decode(self, encoding: AE, metadata: Optional[Dict[str, Any]] = None) -> Optional[A]:
        pass
