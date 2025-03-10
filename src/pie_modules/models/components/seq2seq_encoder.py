import logging
from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import FloatTensor, LongTensor, Tensor, nn

logger = logging.getLogger(__name__)

RNN_TYPE2CLASS = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
ACTIVATION_TYPE2CLASS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}


class RNNWrapper(nn.Module):
    def __init__(self, rnn: nn.Module):
        super().__init__()
        self.rnn = rnn

    def forward(self, *args, **kwargs) -> Tensor:
        return self.rnn(*args, **kwargs)[0]

    @property
    def output_size(self) -> int:
        if self.rnn.bidirectional:
            return self.rnn.hidden_size * 2
        else:
            return self.rnn.hidden_size


class ConcatenatedSequencesWrapper(nn.Module):
    """Wrapper for a module that processes concatenated sequences.

    The input tensor is expected to have the shape (batch_size, sequence_length, input_size) and
    multiple sequences are concatenated along the batch dimension. The module is expected to
    process the concatenated sequences and return the processed sequences with the same shape. The
    processed sequences are then separated back to the original sequences and returned as the
    output tensor.
    """

    def __init__(self, module: nn.Module, module_output_size: int):
        super().__init__()
        self.module = module
        self.module_output_size = module_output_size

    def forward(
        self, values: FloatTensor, sequence_ids: LongTensor, *args, **kwargs
    ) -> FloatTensor:
        results = torch.zeros(
            values.size(0), values.size(1), self.module_output_size, device=values.device
        )
        for seq_idx in torch.unique(sequence_ids):
            # get values for the current sequence (from multiple batch entries)
            mask = sequence_ids == seq_idx
            # shape: (num_selected, sequence_length, input_size)
            selected_values = values[mask]
            # flatten the batch dimension
            concatenated_sequence = selected_values.view(-1, selected_values.size(-1))
            # (num_selected * sequence_length, input_size) -> (num_selected * sequence_length, output_size)
            processed_sequence = self.module(
                concatenated_sequence.unsqueeze(0), *args, **kwargs
            ).squeeze(0)
            # restore the batch dimension: (num_selected, sequence_length, output_size)
            reconstructed_sequence = processed_sequence.view(
                selected_values.size(0), selected_values.size(1), processed_sequence.size(-1)
            )
            # store the processed sequence back to the results tensor at the correct batch indices
            results[mask] = reconstructed_sequence
        return results


def build_seq2seq_encoder(
    config: Dict[str, Any], input_size: int
) -> Tuple[Optional[nn.Module], int]:
    # copy the config to avoid side effects
    config = copy(config)
    seq2seq_encoder_type = config.pop("type", None)
    if seq2seq_encoder_type is None:
        logger.warning(
            f"seq2seq_encoder_type is not specified in the seq2seq_encoder: {config}. "
            f"Do not build this seq2seq_encoder."
        )
        return None, input_size

    if seq2seq_encoder_type == "sequential":
        modules: List[nn.Module] = []
        output_size = input_size
        for key, subconfig in config.items():
            module, output_size = build_seq2seq_encoder(subconfig, input_size)
            if module is not None:
                modules.append(module)
            input_size = output_size

        seq2seq_encoder = nn.Sequential(*modules)
    elif seq2seq_encoder_type == "concatenate_sequences":
        submodule, output_size = build_seq2seq_encoder(config["module"], input_size)
        seq2seq_encoder = ConcatenatedSequencesWrapper(submodule, output_size)
    elif seq2seq_encoder_type in RNN_TYPE2CLASS:
        rnn_class = RNN_TYPE2CLASS[seq2seq_encoder_type]
        seq2seq_encoder = RNNWrapper(rnn_class(input_size=input_size, batch_first=True, **config))
        output_size = seq2seq_encoder.output_size
    elif seq2seq_encoder_type == "linear":
        seq2seq_encoder = nn.Linear(in_features=input_size, **config)
        output_size = seq2seq_encoder.out_features
    elif seq2seq_encoder_type in ACTIVATION_TYPE2CLASS:
        activation_class = ACTIVATION_TYPE2CLASS[seq2seq_encoder_type]
        seq2seq_encoder = activation_class(**config)
        output_size = input_size
    elif seq2seq_encoder_type == "dropout":
        seq2seq_encoder = nn.Dropout(**config)
        output_size = input_size
    elif seq2seq_encoder_type == "none":
        seq2seq_encoder = None
        output_size = input_size
    else:
        raise ValueError(f"Unknown seq2seq_encoder_type: {seq2seq_encoder_type}")

    return seq2seq_encoder, output_size
