import dataclasses
from typing import List, Optional

from .mixins import BatchableMixin


@dataclasses.dataclass
class EncodingWithLabelsAndDecoderAttentionMask(BatchableMixin):
    labels: List[int]
    decoder_attention_mask: Optional[List[int]] = None

    # @property
    # def decoder_attention_mask(self) -> List[int]:
    #    return [1] * len(self.labels)
