import torch.nn as nn
import torch

from transformer.conv_encoder import Conv1d
from utils.utils import sequence_mask


class Attention_Assigner(nn.Module):
    """atteniton assigner of CIF including self-attention and feed forward.
    """

    def __init__(self, d_input, d_hidden, w_context, n_layers, dropout=0.1):
        super().__init__()
        # parameters
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.w_context = w_context

        self.conv = Conv1d(d_input, d_hidden, n_layers, w_context,
                           pad='same', name='assigner')
        self.dropout = nn.Dropout(p=dropout)
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
        x = self.dropout(x)
        alphas = self.linear(x).squeeze(-1)
        alphas = torch.sigmoid(alphas)
        pad_mask = sequence_mask(input_lengths)

        return alphas * pad_mask

    def tail_fixing(alpha):

        return alpha


class Attention_Assigner_Big(Attention_Assigner):
    """atteniton assigner of CIF including self-attention and feed forward.
    """

    def __init__(self, d_input, d_hidden, w_context, n_layers, dropout=0.1):
        nn.Module.__init__(self)
        # parameters
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.w_context = w_context

        self.conv_1 = Conv1d(d_input, d_hidden, n_layers, w_context,
                           pad='same', name='assigner1')
        self.conv_2 = Conv1d(d_input, 2 * d_hidden, n_layers, w_context,
                           pad='same', name='assigner2')
        self.linear_1 = nn.Linear(2 * d_hidden, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, 1)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns:
            enc_output: N x T x H
        """
        x, input_lengths = self.conv_1(padded_input, input_lengths)
        x, input_lengths = self.conv_2(x, input_lengths)
        x = self.dropout_1(x)
        x = self.linear_1(x)
        x = self.dropout_2(x)
        x = self.linear_2(x).squeeze(-1)
        alphas = torch.sigmoid(x)
        pad_mask = sequence_mask(input_lengths)

        return alphas * pad_mask


class Attention_Assigner_RNN(Attention_Assigner):
    """atteniton assigner of CIF including self-attention and feed forward.
    """

    def __init__(self, d_input, d_hidden, w_context, n_layers, dropout=0.1):
        nn.Module.__init__(self)
        # parameters
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.w_context = w_context

        self.conv = Conv1d(d_input, d_hidden, n_layers, w_context,
                           pad='same', name='assigner')
        self.rnn = nn.LSTM(input_size=d_hidden, hidden_size=d_hidden//4,
                           bidirectional=True, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_hidden//2, 1)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns:
            enc_output: N x T x H
        """
        x, input_lengths = self.conv(padded_input, input_lengths)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        alphas = self.linear(x).squeeze(-1)
        alphas = torch.sigmoid(alphas)
        pad_mask = sequence_mask(input_lengths)

        return alphas * pad_mask
