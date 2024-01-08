import math
from typing import Callable, List

import torch
from transformers import LogitsProcessor, add_start_docstrings
from transformers.generation.logits_process import LOGITS_PROCESSOR_INPUTS_DOCSTRING


class PrefixConstrainedLogitsProcessorWithMaximum(LogitsProcessor):
    r"""This is similar to [`PrefixConstrainedLogitsProcessor`] but the constraint function gets the
    maximum possible index as input. This is useful for Pointer Network where the generated token
    can be an index into the input which depends on the length of that input.

    Args:
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor, int], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.
    """

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.LongTensor, int], List[int]],
        num_beams: int,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(
            input_ids.view(-1, self._num_beams, input_ids.shape[-1])
        ):
            for beam_id, sent in enumerate(beam_sent):
                mask[
                    batch_id * self._num_beams + beam_id,
                    self._prefix_allowed_tokens_fn(batch_id, sent, mask.size(1)),
                ] = 0

        return scores + mask
