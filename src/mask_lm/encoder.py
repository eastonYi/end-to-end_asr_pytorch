import torch.nn as nn

from transformer.attention import MultiHeadAttention
from transformer.module import PositionalEncoding, PositionwiseFeedForward
from utils.utils import sequence_mask, get_attn_pad_mask


class Encoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, n_src, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, dropout=0.1, pe_maxlen=5000):
        super().__init__()
        # parameters
        self.token_emb = nn.Embedding(n_src, d_model)
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_input = n_src
        self.d_output = d_model
        self.d_inner = d_inner
        self.dropout_rate = dropout
        self.pe_maxlen = pe_maxlen

        # use linear transformation with layer norm to replace input embedding
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
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


class EncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output

    def forward_cache(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input[:, -1: , :], enc_input, enc_input)

        enc_output = self.pos_ffn(enc_output)

        return enc_output
