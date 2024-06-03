import logging
from dataclasses import dataclass
from typing import Iterator, List, MutableMapping, Optional, Tuple, TypeVar, Union

import torch
from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, nn
from torch.nn import Dropout, Parameter
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup
from transformers.utils import ModelOutput
from typing_extensions import TypeAlias

from .common import ModelWithBoilerplate


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0, activation=nn.GELU()):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.f = activation
        self.dropout = Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.f(self.linear(x))
        x = self.dropout(x)
        return x


@dataclass
class SpanPairClassifierOutput(ModelOutput):
    """Base class for outputs of span pair classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(num_valid_input_pairs_in_batch, config.num_labels)`):
            Classification scores (before SoftMax).
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
            The last hidden state of the transformer model. Returned if `return_embeddings=True`.
        span_embeddings (`torch.FloatTensor` of shape `(batch_size, num_spans, span_embedding_dim)`, *optional*):
            The embeddings of the spans. Returned if `return_embeddings=True`.
        tuple_embeddings (`torch.FloatTensor` of shape `(num_valid_input_pairs_in_batch, tuple_embedding_dim)`, *optional*):
            The embeddings of the tuples. Returned if `return_embeddings=True`.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    span_embeddings: Optional[torch.FloatTensor] = None
    tuple_embeddings: Optional[torch.FloatTensor] = None


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
    """A span tuple classification model that uses a pooler to get a representation of the input
    spans and then applies a linear classifier to that representation. The pooler can be configured
    via the `span_embedding_mode` and `tuple_embedding_mode` arguments. It expects the input to
    contain the indices of the start and end tokens of the spans (for the span pooler) and the
    indices of the spans in the tuples to classify (for the tuple pooler).

    Args:
        model_name_or_path: The name or path of the HuggingFace model to use.
        num_classes: The number of classes for the classification task.
        span_embedding_mode: The mode to pool the hidden states for the spans. One of "start_token",
            "end_token", "start_and_end_token".
        tuple_embedding_mode: The mode to pool the span embeddings for the tuples. Possible values are
            "concat" (concatenate the embeddings of the tuple entries), "multiply2_and_concat"
            (multiply the embeddings of the first two entries and concatenate them with the
            embeddings of the first two entries) and "index_{idx}" (use the embedding of the entry
            at index {idx} as the tuple embedding). Note that "multiply2_and_concat" requires
            `num_tuple_entries=2`. Default: "multiply2_and_concat".
        num_tuple_entries: The number of entries in the tuples.
        tuple_entry_hidden_dim: If provided, the tuple entries (i.e. the span embeddings at the tuple indices)
            are mapped to this dimensionality before combining them. Default: 768.
        tokenizer_vocab_size: The size of the tokenizer vocabulary. If provided, the model's
            tokenizer embeddings are resized to this size.
        classifier_dropout: The dropout probability for the classifier. If not provided, the
            dropout probability is taken from the Huggingface model config.
        learning_rate: The learning rate for the optimizer.
        task_learning_rate: The learning rate for the task-specific parameters. If None, the
            learning rate for all parameters is set to `learning_rate`.
        warmup_proportion: The proportion of steps to warm up the learning rate.
        multi_label: If True, the model is trained as a multi-label classifier.
        multi_label_threshold: The threshold for the multi-label classifier, i.e. the probability
            above which a class is predicted.
        freeze_base_model: If True, the base model parameters are frozen.
        label_pad_value: The padding value for the labels.
        probability_pad_value: The padding value for the probabilities.
        **kwargs: Additional keyword arguments passed to the parent class,
            see :class:`ModelWithBoilerplate`.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        span_embedding_mode: str = "start_and_end_token",
        tuple_embedding_mode: str = "multiply2_and_concat",
        num_tuple_entries: int = 2,
        tuple_entry_hidden_dim: Optional[int] = 768,
        tokenizer_vocab_size: Optional[int] = None,
        classifier_dropout: Optional[float] = None,
        learning_rate: float = 1e-5,
        task_learning_rate: Optional[float] = None,
        warmup_proportion: float = 0.1,
        multi_label: bool = False,
        multi_label_threshold: float = 0.5,
        freeze_base_model: bool = False,
        label_pad_value: int = -100,
        probability_pad_value: float = -1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.task_learning_rate = task_learning_rate
        self.warmup_proportion = warmup_proportion
        self.freeze_base_model = freeze_base_model
        self.label_pad_value = label_pad_value
        self.probability_pad_value = probability_pad_value

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

        # embedder for the spans
        self.span_embedding_mode = span_embedding_mode
        if self.span_embedding_mode in ["start_token", "end_token"]:
            self.span_embedding_dim = self.model.config.hidden_size
        elif self.span_embedding_mode in ["start_and_end_token"]:
            self.span_embedding_dim = self.model.config.hidden_size * 2
        else:
            raise ValueError(f"Invalid value for span_embedding_mode: {self.span_embedding_mode}")

        # embedder for the tuples
        self.num_tuple_entries = num_tuple_entries
        self.tuple_entry_hidden_dim = tuple_entry_hidden_dim
        if self.tuple_entry_hidden_dim is not None:
            self.tuple_entry_embedders = nn.ModuleList(
                [
                    MLP(self.span_embedding_dim, self.tuple_entry_hidden_dim)
                    for _ in range(num_tuple_entries)
                ]
            )
            tuple_entry_dim = self.tuple_entry_hidden_dim
        else:
            self.tuple_entry_embedders = None
            tuple_entry_dim = self.span_embedding_dim
        self.tuple_embedding_mode = tuple_embedding_mode
        if self.tuple_embedding_mode == "concat":
            tuple_embedding_dim = tuple_entry_dim * self.num_tuple_entries
        elif self.tuple_embedding_mode == "multiply2_and_concat":
            if self.num_tuple_entries != 2:
                raise ValueError(
                    "tuple_embedding_mode='multiply2_and_concat' requires num_tuple_entries=2"
                )
            tuple_embedding_dim = tuple_entry_dim * 3
        elif self.tuple_embedding_mode.startswith("index_"):
            idx = int(self.tuple_embedding_mode.split("_")[1])
            if idx >= self.num_tuple_entries:
                raise ValueError(
                    f"Invalid index IDX={idx} for tuple_embedding_mode='index_IDX'. "
                    f"Number of entries in tuple: {self.num_tuple_entries}"
                )
            tuple_embedding_dim = tuple_entry_dim
        else:
            raise ValueError(
                f"Invalid value for tuple_embedding_mode: {self.tuple_embedding_mode}"
            )

        # classifier
        # TODO: do sth more sophisticated here
        self.classifier = nn.Linear(tuple_embedding_dim, num_classes)

        self.multi_label = multi_label
        self.multi_label_threshold = multi_label_threshold
        self.loss_fct = nn.BCEWithLogitsLoss() if self.multi_label else nn.CrossEntropyLoss()

    def span_embedder(
        self,
        hidden_state: FloatTensor,
        span_start_indices: LongTensor,
        span_end_indices: LongTensor,
    ) -> FloatTensor:
        """Create the span embeddings from the hidden states and the span start and end indices.

        Args:
            hidden_state: The last hidden state from the transformer model. shape: (batch_size, seq_len, hidden_size)
            span_start_indices: The indices of the start tokens of the spans. shape: (batch_size, num_spans)
            span_end_indices: The indices of the end tokens of the spans. shape: (batch_size, num_spans)

        Returns:
            The pooled span embeddings. shape: (batch_size, num_spans, hidden_size)
        """

        if self.span_embedding_mode == "start_token":
            span_embeddings = get_embeddings_at_indices(hidden_state, span_start_indices)
        elif self.span_embedding_mode == "end_token":
            span_embeddings = get_embeddings_at_indices(hidden_state, span_end_indices)
        elif self.span_embedding_mode == "start_and_end_token":
            span_embeddings = torch.cat(
                [
                    get_embeddings_at_indices(hidden_state, span_start_indices),
                    get_embeddings_at_indices(hidden_state, span_end_indices),
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Invalid value for span_embedding_mode: {self.span_embedding_mode}")

        return span_embeddings

    def tuple_embedder(
        self,
        span_embeddings: FloatTensor,
        tuple_indices: LongTensor,
        tuple_indices_mask: BoolTensor,
    ) -> FloatTensor:
        """Create the tuple embeddings from the span embeddings and the tuple indices.

        Args:
            span_embeddings: The span embeddings. shape: (batch_size, num_spans, span_embedding_size)
            tuple_indices: The indices of the spans in the tuples. shape: (batch_size, num_tuples, num_tuple_entries)
            tuple_indices_mask: A mask indicating which tuples are valid. shape: (batch_size, num_tuples)

        Returns:
            The pooled tuple embeddings. shape: (num_tuples_in_batch, num_tuple_entries * span_embedding_size)
        """

        if not tuple_indices.shape[-1] == self.num_tuple_entries:
            raise ValueError(
                f"Number of entries in tuple_indices should be equal to num_tuple_entries={self.num_tuple_entries}"
            )
        batch_size, max_num_spans = span_embeddings.shape[:2]
        # we need to add the batch offsets to the tuple indices to get the correct indices in the
        # flattened span_embeddings
        batch_offsets = (
            torch.arange(batch_size, device=tuple_indices.device).unsqueeze(-1).unsqueeze(-1)
            * max_num_spans
        )
        tuple_indices_with_offsets = tuple_indices + batch_offsets
        # shape: (num_tuples_in_batch, num_entries)
        valid_tuple_indices_flat = tuple_indices_with_offsets[tuple_indices_mask]

        # we need to flatten the span_embeddings to get the embeddings at the tuple indices
        # shape: (batch_size * num_spans, span_embedding_size)
        span_embeddings_flat = span_embeddings.view(-1, span_embeddings.size(-1))

        # map the span embeddings individually for each tuple entry
        # each entry has the shape: (batch_size * num_spans, tuple_entry_dim)
        if self.tuple_entry_embedders is not None:
            span_embeddings_mapped = [
                mlp(span_embeddings_flat) for mlp in self.tuple_entry_embedders
            ]
        else:
            span_embeddings_mapped = [span_embeddings_flat] * self.num_tuple_entries

        tuple_embeddings_list: List[FloatTensor] = []
        for i in range(self.num_tuple_entries):
            # shape: (num_tuples_in_batch)
            current_tuple_indices = valid_tuple_indices_flat[:, i]
            # get the embeddings that were mapped with the mlp for the current entry
            # shape: (batch_size * num_spans, tuple_entry_dim)
            span_embeddings_mapped_for_entry = span_embeddings_mapped[i]
            # shape: (num_tuples_in_batch, tuple_entry_dim)
            current_embeddings = span_embeddings_mapped_for_entry[current_tuple_indices]
            tuple_embeddings_list.append(current_embeddings)
        if self.tuple_embedding_mode == "concat":
            tuple_embeddings = torch.cat(tuple_embeddings_list, dim=-1).to(span_embeddings.dtype)
        elif self.tuple_embedding_mode == "multiply2_and_concat":
            tuple_embeddings = torch.cat(
                [
                    tuple_embeddings_list[0] * tuple_embeddings_list[1],
                    tuple_embeddings_list[0],
                    tuple_embeddings_list[1],
                ],
                dim=-1,
            )
        elif self.tuple_embedding_mode.startswith("index_"):
            index = int(self.tuple_embedding_mode.split("_")[1])
            tuple_embeddings = tuple_embeddings_list[index]
        else:
            raise ValueError(
                f"Invalid value for tuple_embedding_mode: {self.tuple_embedding_mode}"
            )
        return tuple_embeddings

    def forward(
        self,
        inputs: InputType,
        targets: Optional[TargetType] = None,
        return_embeddings: bool = False,
    ) -> OutputType:
        span_embedder_inputs = {}
        tuple_embedder_inputs = {}
        base_model_inputs = {}
        for k, v in inputs.items():
            if k.startswith("span_"):
                span_embedder_inputs[k] = v
            elif k.startswith("tuple_"):
                tuple_embedder_inputs[k] = v
            else:
                base_model_inputs[k] = v

        output = self.model(**base_model_inputs)
        last_hidden_state = self.dropout(output.last_hidden_state)

        # get the span embeddings from the hidden states and the start and end marker positions
        span_embeddings = self.span_embedder(
            hidden_state=last_hidden_state, **span_embedder_inputs
        )
        # get the tuple embeddings from the span embeddings and the tuple indices
        # Note that this flattens the batch dimension to not compute embeddings for padding tuples!
        tuple_embeddings_flat = self.tuple_embedder(
            span_embeddings=span_embeddings, **tuple_embedder_inputs
        )

        logits_valid = self.classifier(tuple_embeddings_flat)

        result = {"logits": logits_valid}
        if targets is not None:
            labels = targets["labels"]
            mask = inputs["tuple_indices_mask"]
            valid_labels = labels[mask]
            loss = self.loss_fct(logits_valid, valid_labels)
            result["loss"] = loss

        if return_embeddings:
            result["last_hidden_state"] = last_hidden_state
            result["tuple_embeddings"] = tuple_embeddings_flat
            result["span_embeddings"] = span_embeddings

        return SpanPairClassifierOutput(**result)

    def decode(self, inputs: InputType, outputs: OutputType) -> TargetType:
        if not self.multi_label:
            labels_flat = torch.argmax(outputs.logits, dim=-1).to(torch.long)
            probabilities_flat = torch.softmax(outputs.logits, dim=-1)
        else:
            probabilities_flat = torch.sigmoid(outputs.logits)
            labels_flat = (probabilities_flat > self.multi_label_threshold).to(torch.long)

        # re-construct the original shape
        mask = inputs["tuple_indices_mask"]
        # create "empty" labels and probabilities tensors
        labels = (
            torch.ones(mask.shape, dtype=torch.long, device=labels_flat.device)
            * self.label_pad_value
        )
        prob_shape = list(mask.shape) + [probabilities_flat.shape[-1]]
        probabilities = (
            torch.ones(prob_shape, dtype=torch.float, device=probabilities_flat.device)
            * self.probability_pad_value
        )
        # fill in the valid values
        labels[mask] = labels_flat
        probabilities[mask] = probabilities_flat

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
