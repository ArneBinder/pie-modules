import logging
from typing import Any, Dict, List, Optional

import torch
from pytorch_ie import AutoTaskModule
from torchmetrics import Metric

from pie_modules.models.interface import RequiresTaskmoduleConfig

TRAINING = "train"
VALIDATION = "val"
TEST = "test"

logger = logging.getLogger(__name__)


class WithMetricsFromTaskModule(torch.nn.Module, RequiresTaskmoduleConfig):
    def setup_metrics(
        self, metric_stages: List[str], taskmodule_config: Optional[Dict[str, Any]] = None
    ) -> None:
        for stage in [TRAINING, VALIDATION, TEST]:
            setattr(self, f"metric_{stage}", None)
        if taskmodule_config is not None:
            taskmodule = AutoTaskModule.from_config(taskmodule_config)
            for stage in metric_stages:
                if stage not in [TRAINING, VALIDATION, TEST]:
                    raise ValueError(
                        f'metric_stages must only contain the values "{TRAINING}", "{VALIDATION}", and "{TEST}".'
                    )
                stage_metric = taskmodule.configure_model_metric(stage=stage)
                if stage_metric is not None:
                    setattr(self, f"metric_{stage}", stage_metric)
                else:
                    logger.warning(
                        f"The taskmodule {taskmodule.__class__.__name__} does not define a metric for stage "
                        f"'{stage}'."
                    )
        else:
            logger.warning("No taskmodule_config was provided. Metrics will not be available.")

    def get_metric(self, stage: str) -> Optional[Metric]:
        return getattr(self, f"metric_{stage}")
