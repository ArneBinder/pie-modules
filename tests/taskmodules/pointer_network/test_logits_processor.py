import torch

from pie_modules.taskmodules.pointer_network.logits_processor import (
    PrefixConstrainedLogitsProcessorWithMaximum,
)


def test_prefix_constrained_logits_processor_with_maximum():
    def allow_last_three(batch_id, sent, max_index):
        return list(range(max_index - 3, max_index))

    logits_processor = PrefixConstrainedLogitsProcessorWithMaximum(
        prefix_allowed_tokens_fn=allow_last_three, num_beams=1
    )

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]]).to(dtype=torch.long)
    scores = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0]]).to(dtype=torch.float)
    new_scores = logits_processor(input_ids, scores)
    assert new_scores.shape == scores.shape
    torch.testing.assert_close(
        new_scores,
        torch.tensor(
            [[-float("inf"), -float("inf"), -float("inf"), -float("inf"), 0.9, 0.9, 0.0]]
        ),
    )
