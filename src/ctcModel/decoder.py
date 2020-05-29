import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import IGNORE_ID, sequence_mask, get_subsequent_mask, pad_list


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, n_tgt_vocab, d_input):
        super().__init__()
        # parameters
        self.n_tgt_vocab = n_tgt_vocab
        self.dim_output = n_tgt_vocab

        self.tgt_word_prj = nn.Linear(d_input, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

    def forward(self, encoder_padded_outputs, encoder_input_lengths):
        """
        Args:
            padded_input: N x To
            encoder_padded_outputs: N x Ti x H

        Returns:
        """
        # Get Deocder Input and Output

        # Prepare masks
        mask = sequence_mask(encoder_input_lengths) # B x T

        # B x T x V
        mask = mask.view(mask.shape[0], mask.shape[1], 1).repeat(1, 1, self.dim_output)

        # before softmax
        logits = self.tgt_word_prj(encoder_padded_outputs) * mask
        len_logits = encoder_input_lengths

        return logits, len_logits
