import copy
import logging
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional, Set, Type, Union

import torch
from pytorch_ie.auto import AutoTaskModule
from pytorch_ie.core import PyTorchIEModel
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.optim import Optimizer
from torchmetrics import Metric, MetricCollection
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from pie_modules.models.interface import RequiresTaskmoduleConfig
from pie_modules.utils import resolve_type

logger = logging.getLogger(__name__)


STAGE_TRAIN = "train"
STAGE_VAL = "val"
STAGE_TEST = "test"


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flatten_dict_gen(d, parent_key, sep))


@PyTorchIEModel.register()
class SimpleGenerativeModel(PyTorchIEModel, RequiresTaskmoduleConfig):
    """This model is a simple wrapper around a generative model from Huggingface transformers. That
    means, its predict() and predict_step() methods will call the generate() method of the base
    model.

    If a taskmodule config is provided, the taskmodule will be instantiated and used to create metrics and
    a generation config with its configure_model_metric() and configure_model_generation() methods,
    respectively.

    If the base model has a configure_optimizer() method, this will be used to create the optimizer. Otherwise,
    the optimizer_type and learning_rate will be used to create an optimizer.

    Args:
        base_model_type: The type of the base model, e.g. "transformers.AutoModelForSeq2SeqLM". It should have a
            from_pretrained() method.
        base_model_config: A dictionary with the keyword arguments that will be passed to the from_pretrained()
            method of the base model.
        override_generation_kwargs: The generation config for the base model. This will override the generation config
            from the taskmodule, if one is provided.
        taskmodule_config: The config for the taskmodule. This will be used to create metrics and a generation
            config, if the taskmodule has a configure_model_metric() and configure_model_generation() method,
            respectively.
        metric_stages: A list of stage names, i.e. a subset of ("train", "val", "test"), for which metrics will be
            created. Requires a taskmodule with a configure_model_metric() method.
        metric_intervals: The intervals at which the metrics will be computed. This is a dictionary with the metric
            stages ("train", "val", "test") as key and the interval as value. The interval determines how often the
            metric will be computed, i.e. if the interval is 1, the metric will be computed for each batch. If the
            interval is 2, the metric will be computed just for every second batch, i.e. the all other batches will
            be skipped. This is useful to speed up training, because computing the metrics can be expensive.
        use_prediction_for_metrics: Whether to use the generated prediction (e.g. via beam search) for the
            metric calculation. Otherwise, the argmax of the logits from the model output will be used, which is
            much less compute intense but amy overestimate the performance because each token is individually
            predicted from the prefix of gold tokens.
            The value can be a bool or a list of stage names. If this is True, this is equivalent to setting this to
            the list of all metric stages. If it is False, this disables the use of predictions for all stages.
        warmup_proportion: The proportion of the training steps that will be used for the warmup of the learning rate
            scheduler.
        learning_rate: The learning rate for the optimizer. If the base model has a configure_optimizer() method, this
            will be ignored.
        optimizer_type: The type of the optimizer. If the base model has a configure_optimizer() method, this will be
            ignored.
        **kwargs: Additional keyword arguments that will be passed to the PyTorchIEModel constructor.
    """

    def __init__(
        self,
        # base model setup
        base_model_type: str,
        base_model_config: Dict[str, Any],
        # generation
        override_generation_kwargs: Optional[Dict[str, Any]] = None,
        # metrics
        taskmodule_config: Optional[Dict[str, Any]] = None,
        metric_stages: List[str] = [STAGE_VAL, STAGE_TEST],
        metric_intervals: Optional[Dict[str, int]] = None,
        use_prediction_for_metrics: Union[bool, List[str]] = True,
        # scheduler / optimizer
        warmup_proportion: float = 0.0,
        # important: the following entries are only used if the base model does not have a configure_optimizer method!
        learning_rate: Optional[float] = None,
        optimizer_type: Optional[Union[str, Type[Optimizer]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # optimizer / scheduler
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.warmup_proportion = warmup_proportion

        # Note: We do not set expected_super_type=PreTrainedModel for resolve_type() because
        #   AutoModel* classed such as AutoModelForSeq2SeqLM do not inherit from that.
        resolved_base_model_type: Type[PreTrainedModel] = resolve_type(base_model_type)
        self.model = resolved_base_model_type.from_pretrained(**base_model_config)

        self.use_prediction_for_metrics: Set[str]
        if isinstance(use_prediction_for_metrics, bool):
            self.use_prediction_for_metrics = (
                set(metric_stages) if use_prediction_for_metrics else set()
            )
        else:
            self.use_prediction_for_metrics = set(use_prediction_for_metrics)
        missed_stages = self.use_prediction_for_metrics - set(metric_stages)
        if len(missed_stages) > 0:
            raise ValueError(
                f"There are stages in use_prediction_for_metrics that are not in metric_stages: "
                f"{missed_stages}. Available metric splits: {metric_stages}."
            )

        if taskmodule_config is not None:
            self.taskmodule = AutoTaskModule.from_config(taskmodule_config)
        else:
            self.taskmodule = None

        self.metric_intervals = metric_intervals or {}
        self.configure_metrics(metric_stages=metric_stages)

        self.generation_config = self.configure_generation(**(override_generation_kwargs or {}))

    def configure_metrics(self, metric_stages: List[str]) -> None:
        if self.taskmodule is not None:
            for stage in metric_stages:
                stage_metric = self.taskmodule.configure_model_metric(stage=stage)
                if stage_metric is not None:
                    self.set_metric(stage=stage, metric=stage_metric)
                else:
                    logger.warning(
                        f"The taskmodule {self.taskmodule.__class__.__name__} does not define a metric for stage "
                        f"'{stage}'."
                    )
        else:
            logger.warning(
                "No taskmodule is available, so no metrics will be created. Please set taskmodule_config to a valid "
                "taskmodule config to use metrics."
            )

    def configure_generation(self, **kwargs) -> Dict[str, Any]:
        if self.taskmodule is not None:
            # get the generation config from the taskmodule
            generation_config = self.taskmodule.configure_model_generation()
        else:
            logger.warning(
                "No taskmodule is available, so no generation config will be created. Consider "
                "setting taskmodule_config to a valid taskmodule config to use specific setup for generation."
            )
            generation_config = {}
        generation_config.update(kwargs)
        return generation_config

    def predict(self, inputs, **kwargs) -> torch.LongTensor:
        is_training = self.training
        self.eval()

        generation_kwargs = copy.deepcopy(self.generation_config)
        generation_kwargs.update(kwargs)
        outputs = self.model.generate(**inputs, **generation_kwargs)

        if is_training:
            self.train()

        # TODO: move into base model? or does this work for "all" generative models?
        # strip the bos_id
        if isinstance(outputs, torch.Tensor):
            return outputs[:, 1:]
        else:
            raise ValueError(f"Unsupported output type: {type(outputs)}")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs, _ = batch
        prediction = self.predict(inputs=inputs)
        return prediction

    def forward(self, inputs, **kwargs):
        return self.model(**inputs, **kwargs)

    def set_metric(self, stage: str, metric: Union[Metric, MetricCollection]) -> None:
        if stage not in [STAGE_TRAIN, STAGE_VAL, STAGE_TEST]:
            raise ValueError(
                f"unknown metric stage: {stage}. metric_stages must only contain the values "
                f'"{STAGE_TRAIN}", "{STAGE_VAL}", and "{STAGE_TEST}".'
            )
        setattr(self, f"metric_{stage}", metric)

    def get_metric(self, stage: str, batch_idx: int) -> Optional[Union[Metric, MetricCollection]]:
        stage_metrics = getattr(self, f"metric_{stage}", None)
        metric_interval = self.metric_intervals.get(stage, 1)
        if stage_metrics is not None and (batch_idx + 1) % metric_interval == 0:
            return stage_metrics
        else:
            return None

    def step(self, batch, stage: str, batch_idx: int) -> torch.FloatTensor:
        inputs, targets = batch
        if targets is None:
            raise ValueError("Targets must be provided for training or evaluation!")

        outputs = self(inputs=inputs, **targets)
        loss = outputs.loss

        # show loss on each step only during training
        self.log(
            f"loss/{stage}", loss, on_step=(stage == STAGE_TRAIN), on_epoch=True, prog_bar=True
        )

        metric = self.get_metric(stage=stage, batch_idx=batch_idx)
        if metric is not None:
            if stage in self.use_prediction_for_metrics:
                prediction = self.predict(inputs)
            else:
                # construct prediction from the model output
                logits = outputs.logits
                # get the indices (these are without the initial bos_ids, see above)
                prediction = torch.argmax(logits, dim=-1)
            # the format of expected needs to be the same as the format of prediction
            metric(prediction, targets["labels"])
            log_kwargs = {"on_step": False, "on_epoch": True, "sync_dist": True}
            if isinstance(metric, Metric):
                key = getattr(metric, "name", None) or f"metric/{type(metric).__name__}/{stage}"
                self.log(key, value=metric, **log_kwargs)
            elif isinstance(metric, MetricCollection):
                self.log_dict(metric, **log_kwargs)
            else:
                raise ValueError(
                    f"metric must be an instance of torchmetrics.Metric or torchmetrics.MetricCollection, but is "
                    f"of type {type(metric)}."
                )

        return loss

    def training_step(self, batch, batch_idx) -> torch.FloatTensor:
        loss = self.step(batch, stage=STAGE_TRAIN, batch_idx=batch_idx)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.FloatTensor:
        loss = self.step(batch, stage=STAGE_VAL, batch_idx=batch_idx)

        return loss

    def test_step(self, batch, batch_idx) -> torch.FloatTensor:
        loss = self.step(batch, stage=STAGE_TEST, batch_idx=batch_idx)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if hasattr(self.model, "configure_optimizer") and callable(self.model.configure_optimizer):
            if self.learning_rate is not None:
                raise ValueError(
                    f"learning_rate is set to {self.learning_rate}, but the *base model* ({type(self.model)}) has a "
                    f"configure_optimizer method. Please set learning_rate to None and configure the optimizer "
                    f"inside the *base model*."
                )
            optimizer = self.model.configure_optimizer()
        else:
            logger.warning(
                f"The model does not have a configure_optimizer method. Creating an optimizer of "
                f"optimizer_type={self.optimizer_type} with the learning_rate={self.learning_rate} instead."
            )
            if self.optimizer_type is None:
                raise ValueError(
                    f"optimizer_type is None, but the *base model* ({type(self.model)}) does not have a "
                    f"configure_optimizer method. Please set the optimizer_type to a valid optimizer type, "
                    f"e.g. optimizer_type=torch.optim.Adam."
                )
            resolved_optimizer_type = resolve_type(
                self.optimizer_type, expected_super_type=Optimizer
            )
            optimizer = resolved_optimizer_type(self.parameters(), lr=self.learning_rate)

        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
