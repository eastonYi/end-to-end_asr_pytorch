import torch
import torch.nn as nn


class Mask_LM(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc = nn.Linear(encoder.d_output, encoder.d_input, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def token_mask(self, padded_input, p=0.05, M=10):
        """
        Args:
            padded_input: N x Ti x D
        """
        B, T = padded_input.size(0), padded_input.size(1)
        mask_index = torch.rand((B, T)).cuda() > p # 1110111...
        _mask_index = mask_index.clone()

        for i in range(M):
            _mask_index = torch.cat([_mask_index[:, 1:], _mask_index[:, 0].unsqueeze(1)], 1)
            mask_index *= _mask_index

        # mask_index:  1110000111
        # ~mask_index: 0001111000
        masked_input = torch.where(
            mask_index,
            padded_input,
            torch.zeros_like(padded_input))

        return masked_input, ~mask_index


    def forward(self, padded_input, input_lengths, padded_target=None):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        masked_input, masked_index = self.token_mask(padded_input)
        encoder_padded_outputs = self.encoder(masked_input, input_lengths)
        # pred is score before softmax

        logits_AE = self.fc(encoder_padded_outputs)

        if padded_target:
            logits = self.decoder(padded_target, encoder_padded_outputs,
                                  input_lengths)
        else:
            logits = None

        return logits_AE, logits, masked_index

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        encoder_outputs = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 args)
        return nbest_hyps

    @classmethod
    def create_model(cls, args):
        from ctcModel.decoder import Decoder
        from mask_lm.encoder import Encoder

        encoder = Encoder(args.n_src, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(args.n_tgt, args.d_model)

        model = cls(encoder, decoder)

        return model

    @classmethod
    def load_model(cls, path, args):
        model = cls(*cls.create_model(args))

        package = torch.load(path, map_location=lambda storage, loc: storage)
        model.load_state_dict(package['state_dict'])

        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package
