import torch.nn as nn

from transformer.module import PositionalEncoding
from transformer.encoder import EncoderLayer
from utils.utils import sequence_mask, get_attn_pad_mask


class Encoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, n_src, n_layers, n_head, d_model, d_inner, dropout=0.1):
        super().__init__()
        # parameters
        self.token_emb = nn.Embedding(n_src, d_model)
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_model = d_model
        self.d_input = n_src
        self.d_output = d_model
        self.d_inner = d_inner
        self.dropout_rate = dropout

        # use linear transformation with layer norm to replace input embedding
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns:
            enc_output: N x T x H
        """
        # Prepare masks
        non_pad_mask = sequence_mask(input_lengths).unsqueeze(-1)
        length = padded_input.size(1)
        slf_attn_mask = get_attn_pad_mask(input_lengths, length)

        # Forward
        enc_output = self.dropout(
            self.layer_norm_in(self.token_emb(padded_input)) +
            self.positional_encoding(padded_input))

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        return enc_output
