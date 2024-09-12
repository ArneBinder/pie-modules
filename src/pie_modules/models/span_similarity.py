import logging
from typing import Any, Dict, Iterator, MutableMapping, Optional, Tuple, Union

import torch
from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath
from torch import FloatTensor, LongTensor, nn
from torch.nn import BCELoss, Parameter
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput
from typing_extensions import TypeAlias

from .common import ModelWithBoilerplate
from .components.pooler import get_pooler_and_output_size

# model inputs / outputs / targets
InputType: TypeAlias = MutableMapping[str, LongTensor]
OutputType: TypeAlias = SequenceClassifierOutput
TargetType: TypeAlias = MutableMapping[str, Union[LongTensor, FloatTensor]]
# step inputs (batch) / outputs (loss)
StepInputType: TypeAlias = Tuple[InputType, TargetType]
StepOutputType: TypeAlias = FloatTensor


HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE = {
    "albert": "classifier_dropout_prob",
    "distilbert": "seq_classif_dropout",
}

logger = logging.getLogger(__name__)


@PyTorchIEModel.register()
class SpanSimilarityModel(
    ModelWithBoilerplate[InputType, OutputType, TargetType, StepOutputType],
    RequiresModelNameOrPath,
):
    """TODO.

    Args:
        model_name_or_path: The name or path of the HuggingFace model to use.
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
        pooler: The pooler configuration. If None, CLS token pooling is used.
        freeze_base_model: If True, the base model parameters are frozen.
        base_model_prefix: The prefix of the base model parameters when using a task_learning_rate
            or freeze_base_model. If None, the base_model_prefix of the model is used.
        **kwargs: Additional keyword arguments passed to the parent class,
            see :class:`ModelWithBoilerplate`.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_vocab_size: Optional[int] = None,
        similarity_threshold: float = 0.5,
        classifier_dropout: Optional[float] = None,
        learning_rate: float = 1e-5,
        task_learning_rate: Optional[float] = None,
        warmup_proportion: float = 0.1,
        # TODO: use "mention_pooling" per default? But this requires
        #  to also set num_indices=1 in the pooler_config
        pooler: Optional[Union[Dict[str, Any], str]] = None,
        freeze_base_model: bool = False,
        hidden_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.similarity_threshold = similarity_threshold
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

        if isinstance(pooler, str):
            pooler = {"type": pooler}
        if pooler is not None:
            if pooler["type"] == "mention_pooling":
                # we have only one index (span) per input to pool
                pooler["num_indices"] = 1
        self.pooler_config = pooler or {}
        self.pooler, pooler_output_dim = get_pooler_and_output_size(
            config=self.pooler_config,
            input_dim=config.hidden_size,
        )
        if hidden_dim is not None:
            self.classifier = nn.Linear(pooler_output_dim, hidden_dim)
        else:
            self.classifier = None

        # TODO: is this ok?
        self.loss_fct = BCELoss()

    def get_pooled_output(self, model_inputs, pooler_inputs):
        output = self.model(**model_inputs)
        hidden_state = output.last_hidden_state
        pooled_output = self.pooler(hidden_state, **pooler_inputs)
        pooled_output = self.dropout(pooled_output)
        if self.classifier is not None:
            return self.classifier(pooled_output)
        return pooled_output

    def forward(
        self,
        inputs: InputType,
        targets: Optional[TargetType] = None,
        return_hidden_states: bool = False,
    ) -> OutputType:
        model_inputs = None
        model_inputs_pair = None
        pooler_inputs = {}
        pooler_inputs_pair = {}
        for k, v in inputs.items():
            if k.startswith("pooler_") and k.endswith("_pair"):
                k_target = k[len("pooler_") : -len("_pair")]
                pooler_inputs_pair[k_target] = v
            elif k.startswith("pooler_"):
                k_target = k[len("pooler_") :]
                pooler_inputs[k_target] = v
            elif k == "encoding":
                model_inputs = v
            elif k == "encoding_pair":
                model_inputs_pair = v
            else:
                raise ValueError(f"unexpected model input: {k}")

        pooled_output = self.get_pooled_output(model_inputs, pooler_inputs)
        pooled_output_pair = self.get_pooled_output(model_inputs_pair, pooler_inputs_pair)

        logits = torch.nn.functional.cosine_similarity(pooled_output, pooled_output_pair)

        result = {"logits": logits}
        if targets is not None:
            labels = targets["labels"]
            loss = self.loss_fct(logits, labels)
            result["loss"] = loss
        if return_hidden_states:
            raise NotImplementedError("return_hidden_states is not yet implemented")

        return SequenceClassifierOutput(**result)

    def decode(self, inputs: InputType, outputs: OutputType) -> TargetType:
        labels = (outputs.logits >= self.similarity_threshold).to(torch.long)

        return {"labels": labels, "probabilities": outputs.logits}

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
