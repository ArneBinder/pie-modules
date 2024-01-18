import logging
from typing import Optional, Tuple

import torch
from pytorch_ie.core import PyTorchIEModel
from pytorch_ie.models.interface import RequiresModelNameOrPath, RequiresNumClasses
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import FloatTensor, LongTensor
from transformers import AutoConfig, AutoModelForTokenClassification, BatchEncoding
from transformers.modeling_outputs import TokenClassifierOutput
from typing_extensions import TypeAlias

from pie_modules.models.default_model import DefaultModel

# mode inputs / outputs / targets
InputType: TypeAlias = BatchEncoding
OutputType: TypeAlias = TokenClassifierOutput
TargetType: TypeAlias = LongTensor
# step inputs / outputs
StepInputType: TypeAlias = Tuple[InputType, Optional[TargetType]]
StepOutputType: TypeAlias = FloatTensor


logger = logging.getLogger(__name__)


@PyTorchIEModel.register()
class SimpleTokenClassificationModel(
    DefaultModel[InputType, OutputType, TargetType, StepOutputType],
    RequiresModelNameOrPath,
    RequiresNumClasses,
):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        learning_rate: float = 1e-5,
        label_pad_id: int = -100,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.label_pad_id = label_pad_id
        self.num_classes = num_classes

        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_classes)
        if self.is_from_pretrained:
            self.model = AutoModelForTokenClassification.from_config(config=config)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name_or_path, config=config
            )

    def forward(self, inputs: InputType, targets: Optional[TargetType] = None) -> OutputType:
        inputs_without_special_tokens_mask = {
            k: v for k, v in inputs.items() if k != "special_tokens_mask"
        }
        return self.model(labels=targets, **inputs_without_special_tokens_mask)

    def decode(self, inputs: InputType, outputs: InputType) -> TargetType:
        # get the max index for each token from the logits
        tags_tensor = torch.argmax(outputs.logits, dim=-1).to(torch.long)

        # mask out the padding and special tokens
        tags_tensor = tags_tensor.masked_fill(inputs["attention_mask"] == 0, self.label_pad_id)

        # mask out the special tokens
        tags_tensor = tags_tensor.masked_fill(
            inputs["special_tokens_mask"] == 1, self.label_pad_id
        )
        return tags_tensor

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
