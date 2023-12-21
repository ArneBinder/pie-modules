import abc
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from pytorch_ie import Annotation
from torchmetrics import Metric

# Annotation Encoding type: encoding for a single annotation
AE = TypeVar("AE")
# Annotation type
A = TypeVar("A", bound=Annotation)
# Annotation Collection Encoding type: encoding for a collection of annotations,
# e.g. all relevant annotations for a document
ACE = TypeVar("ACE")


class AnnotationEncoderDecoder(abc.ABC, Generic[A, AE]):
    """Base class for annotation encoders and decoders."""

    @abc.abstractmethod
    def encode(self, annotation: A, metadata: Optional[Dict[str, Any]] = None) -> Optional[AE]:
        pass

    @abc.abstractmethod
    def decode(self, encoding: AE, metadata: Optional[Dict[str, Any]] = None) -> Optional[A]:
        pass


class HasDecodeAnnotations(abc.ABC, Generic[ACE]):
    """Interface for modules that can decode annotations."""

    @abc.abstractmethod
    def decode_annotations(
        self, encoding: ACE, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, List[Annotation]], Any]:
        pass


# TODO: move into pytorch_ie
class HasBuildMetric(abc.ABC):
    """Interface for modules that can build metrics."""

    @abc.abstractmethod
    def build_metric(self, stage: Optional[str] = None) -> Metric:
        pass
