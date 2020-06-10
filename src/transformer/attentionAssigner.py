import torch.nn as nn
import torch

from transformer.conv_encoder import Conv1d
from utils.utils import sequence_mask


class Attention_Assigner(nn.Module):
    """atteniton assigner of CIF including self-attention and feed forward.
    """

    def __init__(self, d_input, d_hidden, context_width, layer_num, dropout=0.1):
        super().__init__()
        # parameters
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.layer_num = layer_num
        self.context_width = context_width

        self.conv = Conv1d(d_input, d_hidden, layer_num, context_width,
                           pad='same', name='assigner')
        self.linear = nn.Linear(d_hidden, 1)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns:
            enc_output: N x T x H
        """
        x, input_lengths = self.conv(padded_input, input_lengths)
        alphas = self.linear(x).squeeze(-1)
        alphas = torch.sigmoid(alphas)
        pad_mask = sequence_mask(input_lengths)

        return alphas * pad_mask
