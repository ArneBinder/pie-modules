import logging
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_ie import AutoTaskModule
from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import FloatTensor, LongTensor, Tensor, nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
from transformers import (
    AutoConfig,
    AutoModel,
    BatchEncoding,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import TokenClassifierOutput
from typing_extensions import TypeAlias

from .components.seq2seq_encoder import build_seq2seq_encoder
from .interface import RequiresTaskmoduleConfig

ModelInputsType: TypeAlias = BatchEncoding
ModelTargetsType: TypeAlias = LongTensor
ModelStepInputType: TypeAlias = Tuple[
    ModelInputsType,
    Optional[ModelTargetsType],
]
ModelOutputType: TypeAlias = LongTensor

HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE = {
    "bert": "hidden_dropout_prob",
    "roberta": "hidden_dropout_prob",
    "albert": "classifier_dropout_prob",
    "distilbert": "seq_classif_dropout",
    "deberta-v2": "hidden_dropout_prob",
    "longformer": "hidden_dropout_prob",
}

TRAINING = "train"
VALIDATION = "val"
TEST = "test"

logger = logging.getLogger(__name__)


@PyTorchIEModel.register()
class TokenClassificationModelWithSeq2SeqEncoderAndCrf(
    PyTorchIEModel, RequiresNumClasses, RequiresModelNameOrPath, RequiresTaskmoduleConfig
):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        learning_rate: float = 1e-5,
        task_learning_rate: Optional[float] = None,
        label_pad_id: int = -100,
        ignore_index: Optional[int] = None,
        special_token_label_id: int = 0,
        classifier_dropout: Optional[float] = None,
        use_crf: bool = True,
        freeze_base_model: bool = False,
        warmup_proportion: float = 0.1,
        seq2seq_encoder: Optional[Dict[str, Any]] = None,
        taskmodule_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.ignore_index = ignore_index
        self.special_token_label_id = special_token_label_id

        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.task_learning_rate = task_learning_rate
        self.label_pad_id = label_pad_id
        self.num_classes = num_classes

        config = AutoConfig.from_pretrained(model_name_or_path)
        if self.is_from_pretrained:
            self.model = AutoModel.from_config(config=config)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, config=config)

        if freeze_base_model:
            self.model.requires_grad_(False)

        hidden_size = config.hidden_size
        self.seq2seq_encoder = None
        if seq2seq_encoder is not None:
            self.seq2seq_encoder, hidden_size = build_seq2seq_encoder(
                config=seq2seq_encoder, input_size=hidden_size
            )

        if classifier_dropout is None:
            # Get the classifier dropout value from the Huggingface model config.
            # This is a bit of a mess since some Configs use different variable names or change the semantics
            # of the dropout (e.g. DistilBert has one dropout prob for QA and one for Seq classification, and a
            # general one for embeddings, encoder and pooler).
            classifier_dropout_attr = HF_MODEL_TYPE_TO_CLASSIFIER_DROPOUT_ATTRIBUTE.get(
                config.model_type, "classifier_dropout"
            )
            if hasattr(config, classifier_dropout_attr):
                classifier_dropout = getattr(config, classifier_dropout_attr)
            else:
                raise ValueError(
                    f"The config {type(config),__name__} loaded from {model_name_or_path} has no attribute "
                    f"{classifier_dropout_attr}"
                )
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier = nn.Linear(hidden_size, num_classes)

        self.crf = CRF(num_tags=num_classes, batch_first=True) if use_crf else None

        self.metrics = {}
        if taskmodule_config is not None:
            self.taskmodule = AutoTaskModule.from_config(taskmodule_config)
            for stage in [TRAINING, VALIDATION, TEST]:
                stage_metric = self.taskmodule.configure_model_metric(stage=stage)
                if stage_metric is not None:
                    self.metrics[stage] = stage_metric
                else:
                    logger.warning(
                        f"The taskmodule {self.taskmodule.__class__.__name__} does not define a metric for stage "
                        f"'{stage}'."
                    )

    def decode(
        self,
        logits: FloatTensor,
        attention_mask: LongTensor,
        special_tokens_mask: LongTensor,
    ) -> LongTensor:
        attention_mask_bool = attention_mask.to(torch.bool)
        if self.crf is not None:
            decoded_tags = self.crf.decode(emissions=logits, mask=attention_mask_bool)
            # pad the decoded tags to the length of the logits to have the same shape as when not using the crf
            seq_len = logits.shape[1]
            padded_tags = [
                tags + [self.label_pad_id] * (seq_len - len(tags)) for tags in decoded_tags
            ]
            tags_tensor = torch.tensor(padded_tags, device=logits.device).to(torch.long)
        else:
            # get the max index for each token from the logits
            tags_tensor = torch.argmax(logits, dim=-1).to(torch.long)
        # set the padding and special tokens to the label_pad_id
        mask = attention_mask_bool & ~special_tokens_mask.to(torch.bool)
        tags_tensor = tags_tensor.masked_fill(~mask, self.label_pad_id)
        return tags_tensor

    def forward(
        self, inputs: ModelInputsType, labels: Optional[LongTensor] = None
    ) -> TokenClassifierOutput:
        inputs_without_special_tokens_mask = {
            k: v for k, v in inputs.items() if k != "special_tokens_mask"
        }
        outputs = self.model(**inputs_without_special_tokens_mask)
        sequence_output = outputs[0]

        if self.seq2seq_encoder is not None:
            sequence_output = self.seq2seq_encoder(sequence_output)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.crf is not None:
                # Overwrite the padding labels with ignore_index. Note that this is different from the
                # attention_mask, because the attention_mask includes special tokens, whereas the labels
                # are set to label_pad_id also for special tokens (e.g. [CLS]). We need handle all
                # occurrences of label_pad_id because usually that index is out of range with respect to
                # the number of logits in which case the crf would complain. However, we can not simply
                # pass a mask to the crf that also masks out the special tokens, because the crf does not
                # allow the first token to be masked out.
                mask_pad_or_special = labels == self.label_pad_id
                labels_valid = labels.masked_fill(mask_pad_or_special, self.special_token_label_id)
                # the crf expects a bool mask
                if "attention_mask" in inputs:
                    mask_bool = inputs["attention_mask"].to(torch.bool)
                else:
                    mask_bool = None
                log_likelihood = self.crf(emissions=logits, tags=labels_valid, mask=mask_bool)
                loss = -log_likelihood
            else:
                loss_fct = CrossEntropyLoss(ignore_index=self.label_pad_id)
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def step(self, stage: str, batch: ModelStepInputType) -> FloatTensor:
        inputs, targets = batch
        assert targets is not None, "targets have to be available for training"

        output = self(inputs, labels=targets)

        loss = output.loss
        # show loss on each step only during training
        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == TRAINING),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        metric = self.metrics.get(stage, None)
        if metric is not None:
            predicted_tags = self.decode(
                logits=output.logits,
                attention_mask=inputs["attention_mask"],
                special_tokens_mask=inputs["special_tokens_mask"],
            )
            metric = self.metrics[stage]
            metric(predicted_tags, targets)
            self.log(
                f"metric/{type(metric)}/{stage}",
                metric,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return loss

    def training_step(self, batch: ModelStepInputType, batch_idx: int) -> FloatTensor:
        return self.step(stage=TRAINING, batch=batch)

    def validation_step(self, batch: ModelStepInputType, batch_idx: int) -> FloatTensor:
        return self.step(stage=VALIDATION, batch=batch)

    def test_step(self, batch: ModelStepInputType, batch_idx: int) -> FloatTensor:
        return self.step(stage=TEST, batch=batch)

    def predict(self, inputs: ModelInputsType, **kwargs) -> LongTensor:
        output = self(inputs)
        predicted_tags = self.decode(
            logits=output.logits,
            attention_mask=inputs["attention_mask"],
            special_tokens_mask=inputs["special_tokens_mask"],
        )
        return predicted_tags

    def predict_step(
        self, batch: ModelStepInputType, batch_idx: int, dataloader_idx: int
    ) -> LongTensor:
        inputs, targets = batch
        return self.predict(inputs=inputs)

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end(stage=TRAINING)

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(stage=VALIDATION)

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(stage=TEST)

    def _on_epoch_end(self, stage: str) -> None:
        metric = self.metrics.get(stage, None)
        if metric is not None:
            value = metric.compute()
            self.log(
                f"metric/{type(metric)}/{stage}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            metric.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.task_learning_rate is not None:
            all_params = dict(self.named_parameters())
            base_model_params = dict(self.model.named_parameters(prefix="model"))
            task_params = {k: v for k, v in all_params.items() if k not in base_model_params}
            optimizer = torch.optim.AdamW(
                [
                    {"params": base_model_params.values(), "lr": self.learning_rate},
                    {"params": task_params.values(), "lr": self.task_learning_rate},
                ]
            )
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        if self.warmup_proportion > 0.0:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(stepping_batches * self.warmup_proportion), stepping_batches
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
