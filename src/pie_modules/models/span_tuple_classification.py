import logging
from dataclasses import dataclass
from typing import Iterator, MutableMapping, Optional, Tuple, TypeVar, Union

import torch
from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses
from torch import FloatTensor, LongTensor, Tensor, nn
from torch.nn import Parameter
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup
from transformers.utils import ModelOutput
from typing_extensions import TypeAlias

from .common import ModelWithBoilerplate


@dataclass
class SpanPairClassifierOutput(ModelOutput):
    """Base class for outputs of span pair classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, num_input_pairs, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# model inputs / outputs / targets
InputType: TypeAlias = MutableMapping[str, LongTensor]
OutputType: TypeAlias = SpanPairClassifierOutput
TargetType: TypeAlias = MutableMapping[str, Union[LongTensor, FloatTensor]]
# step inputs (batch) / outputs (loss)
StepInputType: TypeAlias = Tuple[InputType, TargetType]
StepOutputType: TypeAlias = FloatTensor


HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE = {
    "albert": "classifier_dropout_prob",
    "distilbert": "seq_classif_dropout",
}

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Tensor)


def get_embeddings_at_indices(embeddings: T, indices: LongTensor) -> T:
    # embeddings: (bs, seq_len, hidden_size)
    # indices: (bs, num_indices)
    hidden_size = embeddings.size(-1)
    # Expand dimensions of start_marker_positions to match hidden_states
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
    # result: (bs, num_indices, hidden_size)
    result = embeddings.gather(1, indices_expanded)
    return result


@PyTorchIEModel.register()
class SpanTupleClassificationModel(
    ModelWithBoilerplate[InputType, OutputType, TargetType, StepOutputType],
    RequiresModelNameOrPath,
    RequiresNumClasses,
):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        tokenizer_vocab_size: Optional[int] = None,
        classifier_dropout: Optional[float] = None,
        learning_rate: float = 1e-5,
        task_learning_rate: Optional[float] = None,
        warmup_proportion: float = 0.1,
        multi_label: bool = False,
        multi_label_threshold: float = 0.5,
        span_pooler_mode: str = "start_and_end_token",
        tuple_pooler_mode: str = "concat",
        num_tuple_entries: int = 2,
        freeze_base_model: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion
        self.freeze_base_model = freeze_base_model

        config = AutoConfig.from_pretrained(model_name_or_path)
        if self.is_from_pretrained:
            self.model = AutoModel.from_config(config=config)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, config=config)

        if tokenizer_vocab_size is not None:
            self.model.resize_token_embeddings(tokenizer_vocab_size)

        if self.freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad = False

        if classifier_dropout is None:
            # Get the classifier dropout value from the Huggingface model config.
            # This is a bit of a mess since some Configs use different variable names or change the semantics
            # of the dropout (e.g. DistilBert has one dropout prob for QA and one for Seq classification, and a
            # general one for embeddings, encoder and pooler).
            classifier_dropout_attr = HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE.get(
                config.model_type, "classifier_dropout"
            )
            classifier_dropout = getattr(config, classifier_dropout_attr) or 0.0
        self.dropout = nn.Dropout(classifier_dropout)

        self.span_pooler_mode = span_pooler_mode
        if self.span_pooler_mode in ["start_token", "end_token"]:
            span_pooler_factor = 1
        elif self.span_pooler_mode in ["start_and_end_token"]:
            span_pooler_factor = 2
        else:
            raise ValueError(f"Invalid value for use_markers: {self.span_pooler_mode}")

        self.num_tuple_entries = num_tuple_entries
        self.tuple_pooler_mode = tuple_pooler_mode
        if self.tuple_pooler_mode == "concat":
            tuple_pooler_factor = num_tuple_entries
        else:
            raise ValueError(f"Invalid value for tuple_pooler_mode: {self.tuple_pooler_mode}")

        # TODO: do sth more sophisticated here
        self.classifier = nn.Linear(
            config.hidden_size * tuple_pooler_factor * span_pooler_factor, num_classes
        )
        self.multi_label = multi_label
        self.multi_label_threshold = multi_label_threshold
        self.loss_fct = nn.BCEWithLogitsLoss() if self.multi_label else nn.CrossEntropyLoss()

    def span_pooler(self, hidden_states, span_start_indices, span_end_indices):
        """Pool the hidden states for the spans using the specified mode.

        Args:
            hidden_states: The hidden states from the transformer model. shape: (batch_size, seq_len, hidden_size)
            span_start_indices: The indices of the start tokens of the spans. shape: (batch_size, num_spans)
            span_end_indices: The indices of the end tokens of the spans. shape: (batch_size, num_spans)

        Returns:
            The pooled span embeddings. shape: (batch_size, num_spans, hidden_size)
        """

        if self.span_pooler_mode == "start_token":
            span_embeddings = get_embeddings_at_indices(hidden_states, span_start_indices)
        elif self.span_pooler_mode == "end_token":
            span_embeddings = get_embeddings_at_indices(hidden_states, span_end_indices)
        elif self.span_pooler_mode == "start_and_end_token":
            span_embeddings = torch.cat(
                [
                    get_embeddings_at_indices(hidden_states, span_start_indices),
                    get_embeddings_at_indices(hidden_states, span_end_indices),
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Invalid value for use_markers: {self.span_pooler_mode}")
        span_embeddings = torch.cat(span_embeddings, dim=-1)
        return span_embeddings

    def tuple_pooler(self, span_embeddings, tuple_indices):
        """Pool the span embeddings for the tuples.

        Args:
            span_embeddings: The span embeddings. shape: (batch_size, num_spans, span_embedding_size)
            tuple_indices: The indices of the spans in the tuples. shape: (batch_size, num_tuples, num_tuple_entries)

        Returns:
            The pooled tuple embeddings. shape: (batch_size, num_tuples, num_tuple_entries * span_embedding_size)
        """

        if not tuple_indices.shape[-1] == self.num_tuple_entries:
            raise ValueError(
                f"Number of entries in tuple_indices should be equal to num_tuple_entries={self.num_tuple_entries}"
            )
        tuple_embeddings = []
        for i in range(tuple_indices.shape[-1]):
            current_embeddings = get_embeddings_at_indices(span_embeddings, tuple_indices[:, i])
            tuple_embeddings.append(current_embeddings)
        if self.tuple_pooler_mode == "concat":
            tuple_embeddings = torch.cat(tuple_embeddings, dim=-1)
        else:
            raise ValueError(f"Invalid value for tuple_pooler_mode: {self.tuple_pooler_mode}")
        return tuple_embeddings

    def pooler(self, hidden_states, tuple_indices, **span_pooler_inputs):
        # get the span embeddings from the hidden states and the start and end marker positions
        span_embeddings = self.span_pooler(hidden_states, **span_pooler_inputs)
        tuple_embeddings = self.tuple_pooler(span_embeddings, tuple_indices)
        return tuple_embeddings

    def forward(self, inputs: InputType, targets: Optional[TargetType] = None) -> OutputType:
        pooler_inputs = {}
        model_inputs = {}
        for k, v in inputs.items():
            if k in ["span_start_indices", "span_end_indices", "tuple_indices"]:
                pooler_inputs[k] = v
            else:
                model_inputs[k] = v

        output = self.model(**model_inputs)

        hidden_state = output.last_hidden_state

        pooled_output = self.pooler(hidden_state, **pooler_inputs)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        result = {"logits": logits}
        if targets is not None:
            labels = targets["labels"]
            loss = self.loss_fct(logits, labels)
            result["loss"] = loss

        return SpanPairClassifierOutput(**result)

    def decode(self, inputs: InputType, outputs: OutputType) -> TargetType:
        if not self.multi_label:
            labels = torch.argmax(outputs.logits, dim=-1).to(torch.long)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        else:
            probabilities = torch.sigmoid(outputs.logits)
            labels = (probabilities > self.multi_label_threshold).to(torch.long)
        return {"labels": labels, "probabilities": probabilities}

    def base_model_named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        if prefix:
            prefix = f"{prefix}."
        return self.model.named_parameters(prefix=f"{prefix}model")

    def task_named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        if prefix:
            prefix = f"{prefix}."
        base_model_parameter_names = dict(self.base_model_named_parameters(prefix=prefix)).keys()
        for name, param in self.named_parameters(prefix=prefix):
            if name not in base_model_parameter_names:
                yield name, param

    def configure_optimizers(self):
        if self.task_learning_rate is not None:
            base_model_params = (param for name, param in self.base_model_named_parameters())
            task_params = (param for name, param in self.task_named_parameters())
            optimizer = AdamW(
                [
                    {"params": base_model_params, "lr": self.learning_rate},
                    {"params": task_params, "lr": self.task_learning_rate},
                ]
            )
        else:
            optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
