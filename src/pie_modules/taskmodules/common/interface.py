import abc
from typing import Optional

from torchmetrics import Metric


# TODO: move to pytorch_ie
class HasConfigureMetric(abc.ABC):
    """Interface for modules that can configure a metric."""

    @abc.abstractmethod
    def configure_metric(self, stage: Optional[str] = None) -> Optional[Metric]:
        pass
