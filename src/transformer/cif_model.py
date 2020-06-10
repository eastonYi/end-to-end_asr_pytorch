import torch
import torch.nn as nn

from conv_encoder import Conv2dSubsample as Conv_Encoder
from decoder import Decoder_CIF as Decoder
from encoder import Encoder
from attentionAssigner import Attention_Assigner


class CIF_Model(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, conv_encoder, encoder, assigner, decoder):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.encoder = encoder
        self.assigner = assigner
        self.decoder = decoder
        self.ctc_fc = nn.Linear(encoder.dim_output, decoder.dim_output, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, len_features, targets, threshold=0.95):
        """
        Args:
            features: N x T x D
            len_sequence: N
            padded_targets: N x To
        """
        conv_outputs, len_sequence = self.conv_encoder(features, len_features)
        encoder_outputs, *_ = self.encoder(conv_outputs, len_sequence)

        ctc_logits = self.ctc_fc(encoder_outputs)
        len_ctc_logits = len_sequence

        alpha = self.assigner(encoder_outputs, len_sequence)

        # sum
        _num = alpha.sum(-1)
        # scaling
        num = (targets > 0).float().sum(-1)
        alpha *= (num / _num)[:, None].repeat(1, alpha.size(1))

        # cif
        l = self.cif(encoder_outputs, alpha, threshold=threshold)

        logits = self.decoder(l, targets)

        return ctc_logits, len_ctc_logits, _num, num, logits

    def cif(self, hidden, alphas, threshold, log=False):
        batch_size, len_time, hidden_size = hidden.size()

        # loop varss
        integrate = torch.zeros([batch_size]).cuda()
        frame = torch.zeros([batch_size, hidden_size]).cuda()
        # intermediate vars along time
        list_fires = []
        list_frames = []

        for t in range(len_time):
            alpha = alphas[:, t]
            distribution_completion = torch.ones([batch_size]).cuda() - integrate

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate > threshold
            integrate = torch.where(fire_place,
                                    integrate - torch.ones([batch_size]).cuda(),
                                    integrate)
            cur = torch.where(fire_place,
                              distribution_completion,
                              alpha)
            remainds = alpha - cur

            frame += cur[:, None] * hidden[:, t, :]
            list_frames.append(frame)
            frame = torch.where(fire_place[:, None].repeat(1, hidden_size),
                                remainds[:, None] * hidden[:, t, :],
                                frame)
            if log:
                print('t: {}\t{:.3f} -> {:.3f}|{:.3f}'.format(
                    t, integrate[0].numpy(), cur[0].numpy(), remainds[0].numpy()))

        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        list_ls = []
        len_labels = (alphas.sum(-1) - 0.001).ceil().int()
        max_label_len = len_labels.max()
        for b in range(batch_size):
            fire = fires[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.where(fire > threshold)[0])
            pad_l = torch.zeros([max_label_len - l.size(0), hidden_size]).cuda()
            list_ls.append(torch.cat([l, pad_l], 0))

        if log:
            print('fire:\n', fires.numpy())

        return torch.stack(list_ls, 0)

    def recognize(self, input, input_length, char_list, args, threshold=0.95):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        conv_padded_outputs, input_length = self.conv_encoder(input, input_length)
        encoder_outputs, *_ = self.encoder(input.unsqueeze(0), input_length)
        alpha = self.assigner(encoder_outputs, input_length)

        l = self.cif(encoder_outputs, alpha, threshold=threshold)

        nbest_hyps = self.decoder.recognize_beam(l, char_list, args)

        return nbest_hyps

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model, LFR_m, LFR_n = cls.load_model_from_package(package)
        return model, LFR_m, LFR_n

    @classmethod
    def load_model_from_package(cls, package):
        conv_encoder = Conv_Encoder(
                        package['d_input'],
                        package['n_layers_enc'],
                        package['n_head'],
                        package['d_k'],
                        package['d_v'],
                        package['d_model'],
                        package['d_inner'],
                        dropout=package['dropout'],
                        pe_maxlen=package['pe_maxlen'])
        encoder = Encoder(
                        package['d_input'],
                        package['n_layers_enc'],
                        package['n_head'],
                        package['d_k'],
                        package['d_v'],
                        package['d_model'],
                        package['d_inner'],
                        dropout=package['dropout'],
                        pe_maxlen=package['pe_maxlen'])
        assigner = Attention_Assigner(
                        package['d_model'],
                        package['d_model'],
                        package['context_width'],
                        package['num_assigner_layers'])
        decoder = Decoder(
                        package['sos_id'],
                        package['vocab_size'],
                        package['d_word_vec'],
                        package['n_layers_dec'],
                        package['n_head'],
                        package['d_k'],
                        package['d_v'],
                        package['d_model'],
                        package['d_inner'],
                        dropout=package['dropout'],
                        tgt_emb_prj_weight_sharing=package['tgt_emb_prj_weight_sharing'],
                        pe_maxlen=package['pe_maxlen'])
        model = cls(conv_encoder, encoder, assigner, decoder)
        model.load_state_dict(package['state_dict'])
        LFR_m, LFR_n = package['LFR_m'], package['LFR_n']
        return model, LFR_m, LFR_n

    @staticmethod
    def serialize(model, optimizer, epoch, LFR_m, LFR_n, tr_loss=None, cv_loss=None):
        package = {
            # Low Frame Rate Feature
            'LFR_m': LFR_m,
            'LFR_n': LFR_n,
            # encoder
            'd_input': model.encoder.d_input,
            'n_layers_enc': model.encoder.n_layers,
            'n_head': model.encoder.n_head,
            'd_k': model.encoder.d_k,
            'd_v': model.encoder.d_v,
            'd_model': model.encoder.d_model,
            'd_inner': model.encoder.d_inner,
            'dropout': model.encoder.dropout_rate,
            'pe_maxlen': model.encoder.pe_maxlen,
            # assigner
            'context_width': model.assigner.context_width,
            'num_assigner_layers': model.assigner.layer_num,
            # decoder
            'sos_id': model.decoder.sos_id,
            'vocab_size': model.decoder.n_tgt_vocab,
            'd_word_vec': model.decoder.d_word_vec,
            'n_layers_dec': model.decoder.n_layers,
            'tgt_emb_prj_weight_sharing': model.decoder.tgt_emb_prj_weight_sharing,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package
