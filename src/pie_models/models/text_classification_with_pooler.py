import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, MutableMapping, Optional, Tuple, Union

import torch
import torchmetrics
from pytorch_ie.core import PyTorchIEModel
from torch import Tensor, cat, nn
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup
from typing_extensions import TypeAlias

# The input to the forward method of this model. It is passed to
# the base transformer model. Can also contain additional arguments
# for the pooler (these need to be prefixed with "pooler_").
ModelInputType: TypeAlias = MutableMapping[str, Any]
# A dict with a single key "logits".
ModelOutputType: TypeAlias = Dict[str, Tensor]
# This contains the input and target tensors for a single training step.
StepInputType: TypeAlias = Tuple[
    ModelInputType,  # input
    Optional[Tensor],  # targets
]

# stage names
TRAINING = "train"
VALIDATION = "val"
TEST = "test"

# classification head input types for classifier_head_input_type
CLS_TOKEN = "cls_token"  # CLS token
START_TOKENS = "start_tokens"  # MTB start tokens concat
MENTION_POOLING = "mention_pooling"  # mention token pooling and concat


logger = logging.getLogger(__name__)


def pool_cls(hidden_state: Tensor, **kwargs) -> Tensor:
    return hidden_state[:, 0, :]


class Pooler(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError


class AtIndexPooler(Pooler):
    """Pooler that takes the hidden state at a given index.

    Args:
        input_dim: The input dimension of the hidden state.
        num_indices: The number of indices to pool.
        offset: The offset to add to the indices.
    """

    def __init__(self, input_dim: int, num_indices: int = 2, offset: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_indices = num_indices
        self.offset = offset
        self.missing_embeddings = nn.Parameter(torch.empty(num_indices, self.input_dim))
        nn.init.normal_(self.missing_embeddings)

    def forward(self, hidden_state: Tensor, indices: Tensor, **kwargs) -> Tensor:
        batch_size, seq_len, hidden_size = hidden_state.shape
        if indices.shape[1] != self.num_indices:
            raise ValueError(
                f"number of indices [{indices.shape[1]}] has to be the same as num_types [{self.num_indices}]."
            )

        # times num_types due to concat
        result = torch.zeros(
            batch_size, hidden_size * self.num_indices, device=hidden_state.device
        )
        for batch_idx, current_indices in enumerate(indices):
            current_embeddings = [
                hidden_state[batch_idx, current_indices[i] + self.offset, :]
                if current_indices[i] >= 0
                else self.missing_embeddings[i]
                for i in range(self.num_indices)
            ]
            result[batch_idx] = cat(current_embeddings, 0)
        return result

    @property
    def output_dim(self) -> int:
        return self.input_dim * self.num_indices


class ArgumentWrappedPooler(Pooler):
    """Wraps a pooler and maps the arguments to the pooler.

    Args:
        pooler: The pooler to wrap.
        argument_mapping: A mapping from the arguments of the forward method to the arguments of the pooler.
    """

    def __init__(self, pooler: nn.Module, argument_mapping: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self.pooler = pooler
        self.argument_mapping = argument_mapping

    def forward(self, hidden_state: Tensor, **kwargs) -> Tensor:
        pooler_kwargs = {}
        for k, v in kwargs.items():
            if k in self.argument_mapping:
                pooler_kwargs[self.argument_mapping[k]] = v
        return self.pooler(hidden_state, **pooler_kwargs)

    @property
    def output_dim(self) -> int:
        return self.pooler.output_dim


class SpanMaxPooler(Pooler):
    """Pooler that takes the max hidden state over a span.

    Args:
        input_dim: The input dimension of the hidden state.
        num_indices: The number of indices to pool.
    """

    def __init__(self, input_dim: int, num_indices: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_indices = num_indices
        self.missing_embeddings = nn.Parameter(torch.empty(num_indices, self.input_dim))
        nn.init.normal_(self.missing_embeddings)

    def forward(
        self, hidden_state: Tensor, start_indices: Tensor, end_indices: Tensor, **kwargs
    ) -> Tensor:
        batch_size, seq_len, hidden_size = hidden_state.shape
        if start_indices.shape != end_indices.shape:
            raise ValueError(
                f"start_indices shape [{start_indices.shape}] has to be the same as end_indices shape "
                f"[{end_indices.shape}]."
            )
        if start_indices.shape[1] != self.num_indices:
            raise ValueError(
                f"number of indices [{start_indices.shape[1]}] has to be the same as num_types [{self.num_indices}]."
            )

        # times num_indices due to concat
        result = torch.zeros(
            batch_size, hidden_size * self.num_indices, device=hidden_state.device
        )
        for batch_idx in range(batch_size):
            current_start_indices = start_indices[batch_idx]
            current_end_indices = end_indices[batch_idx]
            current_embeddings = [
                torch.amax(
                    hidden_state[batch_idx, current_start_indices[i] : current_end_indices[i], :],
                    0,
                )
                if current_start_indices[i] >= 0 and current_end_indices[i] >= 0
                else self.missing_embeddings[i]
                for i in range(self.num_indices)
            ]
            result[batch_idx] = cat(current_embeddings, 0)

        return result

    @property
    def output_dim(self) -> int:
        return self.input_dim * self.num_indices


def get_pooler_and_output_size(config: Dict[str, Any], input_dim: int) -> Tuple[Callable, int]:
    pooler_config = dict(config)
    pooler_type = pooler_config.pop("type", CLS_TOKEN)
    if pooler_type == CLS_TOKEN:
        return pool_cls, input_dim
    elif pooler_type == START_TOKENS:
        pooler = AtIndexPooler(input_dim=input_dim, offset=-1, **pooler_config)
        pooler_wrapped = ArgumentWrappedPooler(
            pooler=pooler, argument_mapping={"start_indices": "indices"}
        )
        return pooler_wrapped, pooler.output_dim
    elif pooler_type == MENTION_POOLING:
        pooler = SpanMaxPooler(input_dim=input_dim, **pooler_config)
        return pooler, pooler.output_dim
    else:
        raise ValueError(f"Unknown pooler type {pooler_type}")


@PyTorchIEModel.register()
class TextClassificationModelWithPooler(PyTorchIEModel):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        tokenizer_vocab_size: int,
        ignore_index: Optional[int] = None,
        learning_rate: float = 1e-5,
        task_learning_rate: float = 1e-4,
        warmup_proportion: float = 0.1,
        freeze_model: bool = False,
        multi_label: bool = False,
        t_total: Optional[int] = None,
        pooler: Optional[Union[Dict[str, Any], str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if t_total is not None:
            logger.warning(
                "t_total is deprecated, we use estimated_stepping_batches from the pytorch lightning trainer instead"
            )

        self.save_hyperparameters(ignore=["t_total"])

        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion

        config = AutoConfig.from_pretrained(model_name_or_path)
        if self.is_from_pretrained:
            self.model = AutoModel.from_config(config=config)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.model.resize_token_embeddings(tokenizer_vocab_size)

        # if freeze_model:
        #     for param in self.model.parameters():
        #         param.requires_grad = False

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        if isinstance(pooler, str):
            pooler = {"type": pooler}
        self.pooler_config = pooler or {}
        self.pooler, pooler_output_dim = get_pooler_and_output_size(
            config=self.pooler_config,
            input_dim=config.hidden_size,
        )
        self.classifier = nn.Linear(pooler_output_dim, num_classes)

        self.loss_fct = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()

        self.f1 = nn.ModuleDict(
            {
                f"stage_{stage}": torchmetrics.F1Score(
                    num_classes=num_classes, ignore_index=ignore_index
                )
                for stage in [TRAINING, VALIDATION, TEST]
            }
        )

    def forward(self, inputs: ModelInputType) -> ModelOutputType:
        pooler_inputs = {}
        model_inputs = {}
        for k, v in inputs.items():
            if k.startswith("pooler_"):
                pooler_inputs[k[len("pooler_") :]] = v
            else:
                model_inputs[k] = v

        output = self.model(**model_inputs)

        hidden_state = output.last_hidden_state

        pooled_output = self.pooler(hidden_state, **pooler_inputs)

        logits = self.classifier(pooled_output)
        return {"logits": logits}

    def step(self, stage: str, batch: StepInputType):
        input_, target = batch
        assert target is not None, "target has to be available for training"

        logits = self(input_)["logits"]

        loss = self.loss_fct(logits, target)

        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        f1 = self.f1[f"stage_{stage}"]
        f1(logits, target)
        self.log(f"{stage}/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: StepInputType, batch_idx: int):
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: StepInputType, batch_idx: int):
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: StepInputType, batch_idx: int):
        return self.step(stage=TEST, batch=batch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer

        # param_optimizer = list(self.named_parameters())
        # # TODO: this needs fixing (does not work models other than BERT)
        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in param_optimizer if "bert" in n]},
        #     {
        #         "params": [p for n, p in param_optimizer if "bert" not in n],
        #         "lr": self.task_learning_rate,
        #     },
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, int(self.t_total * self.warmup_proportion), self.t_total
        # )
        # return [optimizer], [scheduler]
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
