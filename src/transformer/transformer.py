import torch
import torch.nn as nn

from utils.utils import spec_aug


class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, decoder, spec_aug_cfg=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.spec_aug_cfg = spec_aug_cfg

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, len_features, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        if self.spec_aug_cfg:
            features, len_features = spec_aug(features, len_features, self.spec_aug_cfg)

        encoder_padded_outputs = self.encoder(features, len_features)
        # pred is score before softmax
        logits, targets_eos = self.decoder(padded_target, encoder_padded_outputs,
                                           len_features)
        return logits, targets_eos

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

        encoder = Encoder(d_input=args.d_input * args.LFR_m,
                          n_layers=args.n_layers_enc,
                          n_head=args.n_head,
                          d_model=args.d_model,
                          d_inner=args.d_inner,
                          dropout=args.dropout)
        decoder = Decoder(sos_id=args.sos_id,
                          eos_id=args.eos_id,
                          n_tgt_vocab=args.vocab_size,
                          n_layers=args.n_layers_dec,
                          n_head=args.n_head,
                          d_model=args.d_model,
                          d_inner=args.d_inner,
                          dropout=args.dropout)

        model = cls.create_model(encoder, decoder)

        return model

    @classmethod
    def load_model(cls, path, args):
        model = cls.create_model(args)

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

    def __init__(self, encoder, decoder, spec_aug_cfg=None):
        super().__init__(encoder, decoder, spec_aug_cfg)
        self.ctc_fc = nn.Linear(encoder.d_output, decoder.d_output, bias=False)

    def forward(self, features, len_features, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        if self.spec_aug_cfg:
            features, len_features = spec_aug(features, len_features, self.spec_aug_cfg)

        encoder_padded_outputs = self.encoder(features, len_features)
        ctc_pred = self.ctc_fc(encoder_padded_outputs)
        ctc_pred_len = len_features
        # pred is score before softmax
        pred = self.decoder(padded_target, encoder_padded_outputs, len_features)

        return ctc_pred_len, ctc_pred, pred


class Conv_CTC_Transformer(CTC_Transformer):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, conv_encoder, encoder, decoder, spec_aug_cfg=None):
        super().__init__(encoder, decoder, spec_aug_cfg)
        self.conv_encoder = conv_encoder

    def forward(self, features, len_features, targets, spec_aug_cfg=False):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        if self.spec_aug_cfg:
            features, len_features = spec_aug(features, len_features, self.spec_aug_cfg)

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
        # conv_outputs, len_sequence = self.conv_encoder(feature.unsqueeze(0), len_feature)
        conv_outputs, len_sequence = self.conv_encoder(feature, len_feature)
        encoder_outputs = self.encoder(conv_outputs, len_sequence)
        nbest_hyps = self.decoder.recognize(encoder_outputs[0], char_list, args)
        # hyp_ids, scores = self.decoder.beam_search(encoder_outputs, len_sequence, 5, 100)

        return nbest_hyps

    def batch_recognize(self, features, len_features, beam_size):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        conv_outputs, len_sequences = self.conv_encoder(features, len_features)
        encoder_outputs = self.encoder(conv_outputs, len_sequences)
        nbest_hyps = self.decoder.batch_beam_decode(encoder_outputs, len_sequences, beam_size)

        return nbest_hyps


    @classmethod
    def create_model(cls, args):
        from transformer.decoder import Decoder
        from transformer.encoder import Encoder
        from transformer.conv_encoder import Conv2dSubsample

        conv_encoder = Conv2dSubsample(d_input=args.d_input * args.LFR_m,
                                       d_model=args.d_model,
                                       n_layers=args.n_conv_layers)
        encoder = Encoder(d_input=args.d_model,
                          n_layers=args.n_layers_enc,
                          n_head=args.n_head,
                          d_model=args.d_model,
                          d_inner=args.d_inner,
                          dropout=args.dropout)
        decoder = Decoder(sos_id=args.sos_id,
                          eos_id=args.eos_id,
                          n_tgt_vocab=args.vocab_size,
                          n_layers=args.n_layers_dec,
                          n_head=args.n_head,
                          d_model=args.d_model,
                          d_inner=args.d_inner,
                          dropout=args.dropout)

        model = cls(conv_encoder, encoder, decoder, spec_aug_cfg=args.spec_aug_cfg)

        return model

    @classmethod
    def load_model_from_package(cls, package, args):

        model = cls.create_model(args, spec_aug_cfg=args.spec_aug_cfg)
        model.load_state_dict(package['state_dict'])

        return model
