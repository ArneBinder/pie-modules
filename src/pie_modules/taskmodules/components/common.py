import abc
import logging
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from pytorch_ie import Annotation

logger = logging.getLogger(__name__)


AE = TypeVar("AE")
A = TypeVar("A", bound=Annotation)


class AnnotationEncoderDecoder(abc.ABC, Generic[A, AE]):
    """Base class for annotation encoders and decoders."""

    @abc.abstractmethod
    def encode(self, annotation: A, metadata: Optional[Dict[str, Any]] = None) -> Optional[AE]:
        pass

    @abc.abstractmethod
    def decode(self, encoding: AE, metadata: Optional[Dict[str, Any]] = None) -> Optional[A]:
        pass


class AnnotationLayersEncoderDecoder(abc.ABC, Generic[AE]):
    """Base class for annotation layer encoders and decoders."""

    @property
    @abc.abstractmethod
    def layer_names(self) -> List[str]:
        pass

    @abc.abstractmethod
    def encode(
        self, layers: Dict[str, List[Annotation]], metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[AE]:
        pass

    @abc.abstractmethod
    def decode(
        self, encoding: AE, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, List[Annotation]], Any]:
        pass
