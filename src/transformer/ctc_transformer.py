import torch.nn as nn

from transformer import Transformer


class CTC_Transformer(Transformer):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.ctc_fc = nn.Linear(encoder.dim_output, decoder.dim_output, bias=False)

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        ctc_pred = self.ctc_fc(encoder_padded_outputs)
        ctc_pred_len = input_lengths
        # pred is score before softmax
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)

        return [ctc_pred_len, ctc_pred, pred], gold
