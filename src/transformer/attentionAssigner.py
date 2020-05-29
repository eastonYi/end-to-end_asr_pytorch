import torch.nn as nn


class Attention_Assigner(nn.Module):
    """atteniton assigner of CIF including self-attention and feed forward.
    """

    def __init__(self, d_input, d_hidden, dropout=0.1):
        super().__init__()
        # parameters
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.dropout = dropout

        # use linear transformation with layer norm to replace input embedding
        padding = None
        self.conv1 = nn.Conv1d(d_input, d_hidden, kernel_size=3, strides=1, padding=padding)
        self.layer_norm1 = nn.LayerNorm(d_hidden)
        self.conv2 = nn.Conv1d(d_hidden, d_hidden, kernel_size=3, strides=1, padding=padding)
        self.layer_norm2 = nn.LayerNorm(d_hidden)
        self.linear = nn.Linear(d_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns:
            enc_output: N x T x H
        """

        x = self.conv1(padded_input)
        x = self.layer_norm1(x)
        x = self.conv2(x)
        x = self.layer_norm2(x)
        alphas = self.linear(x)

        return alphas
