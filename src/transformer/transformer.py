import torch
import torch.nn as nn


class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)
        return pred, gold

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
        from transformer.decoder import Decoder
        from transformer.encoder import Encoder

        encoder = Encoder(args.d_input * args.LFR_m, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(args.sos_id, args.eos_id, args.vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)

        return encoder, decoder

    @classmethod
    def load_model(cls, path, args):
        encoder, decoder = cls.create_model(args)
        model = cls(encoder, decoder)

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


class CTC_Transformer(Transformer):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.ctc_fc = nn.Linear(encoder.d_output, decoder.d_output, bias=False)

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
        pred = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)

        return ctc_pred_len, ctc_pred, pred


class Conv_CTC_Transformer(CTC_Transformer):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, conv_encoder, encoder, decoder):
        super().__init__(encoder, decoder)
        self.conv_encoder = conv_encoder

    def forward(self, features, len_features, targets):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        conv_outputs, len_sequence = self.conv_encoder(features, len_features)
        encoder_outputs = self.encoder(conv_outputs, len_sequence)

        ctc_logits = self.ctc_fc(encoder_outputs)
        len_ctc_logits = len_sequence

        logits, targets_eos = self.decoder(targets, encoder_outputs, len_sequence)

        return ctc_logits, len_ctc_logits, logits, targets_eos

    def recognize(self, feature, len_feature, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        conv_outputs, len_sequence = self.conv_encoder(feature.unsqueeze(0), len_feature)
        encoder_outputs = self.encoder(conv_outputs, len_sequence)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0], char_list, args)

        return nbest_hyps

    @classmethod
    def create_model(cls, args):
        from transformer.decoder import Decoder
        from transformer.encoder import Encoder
        from transformer.conv_encoder import Conv2dSubsample

        conv_encoder = Conv2dSubsample(args.d_input * args.LFR_m, args.d_model,
                                       n_layers=args.n_conv_layers)
        encoder = Encoder(args.d_model, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(args.sos_id, args.eos_id, args.vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)

        return conv_encoder, encoder, decoder

    @classmethod
    def load_model_from_package(cls, package, args):

        conv_encoder, encoder, decoder = cls.create_model(args)
        model = cls(conv_encoder, encoder, decoder)
        model.load_state_dict(package['state_dict'])

        return model
