import logging
from typing import Any, Dict, MutableMapping, Optional, Tuple, Union

import torchmetrics
from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses
from torch import Tensor, nn
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup
from typing_extensions import TypeAlias

from .components.pooler import get_pooler_and_output_size

# The input to the forward method of this model. It is passed to
# the base transformer model. Can also contain additional arguments
# for the pooler (these need to be prefixed with "pooler_").
ModelInputType: TypeAlias = MutableMapping[str, Any]
# A dict with a single key "logits".
ModelOutputType: TypeAlias = Dict[str, Tensor]
# This contains the input and target tensors for a single training step.
ModelStepInputType: TypeAlias = Tuple[
    ModelInputType,  # input
    Optional[Tensor],  # targets
]

# stage names
TRAINING = "train"
VALIDATION = "val"
TEST = "test"

HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE = {
    "albert": "classifier_dropout_prob",
    "distilbert": "seq_classif_dropout",
}

logger = logging.getLogger(__name__)


@PyTorchIEModel.register()
class SequenceClassificationModel(PyTorchIEModel, RequiresModelNameOrPath, RequiresNumClasses):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        tokenizer_vocab_size: Optional[int] = None,
        ignore_index: Optional[int] = None,
        classifier_dropout: Optional[float] = None,
        learning_rate: float = 1e-5,
        task_learning_rate: Optional[float] = None,
        warmup_proportion: float = 0.1,
        multi_label: bool = False,
        pooler: Optional[Union[Dict[str, Any], str]] = None,
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
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    task="multilabel" if multi_label else "multiclass",
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

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        return {"logits": logits}

    def step(self, stage: str, batch: ModelStepInputType):
        input_, target = batch
        assert target is not None, "target has to be available for training"

        logits = self(input_)["logits"]

        loss = self.loss_fct(logits, target)

        self.log(f"{stage}/loss", loss, on_step=(stage == TRAINING), on_epoch=True, prog_bar=True)

        f1 = self.f1[f"stage_{stage}"]
        f1(logits, target)
        self.log(f"{stage}/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: ModelStepInputType, batch_idx: int):
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: ModelStepInputType, batch_idx: int):
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: ModelStepInputType, batch_idx: int):
        return self.step(stage=TEST, batch=batch)

    def configure_optimizers(self):
        if self.task_learning_rate is not None:
            all_params = dict(self.named_parameters())
            base_model_params = dict(self.model.named_parameters(prefix="model"))
            task_params = {k: v for k, v in all_params.items() if k not in base_model_params}
            optimizer = AdamW(
                [
                    {"params": base_model_params.values(), "lr": self.learning_rate},
                    {"params": task_params.values(), "lr": self.task_learning_rate},
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
