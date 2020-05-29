import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder
from attentionAssigner import Attention_Assigner


class CIF_Model(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, assigner, decoder):
        super().__init__()
        self.encoder = encoder
        self.assigner = assigner
        self.decoder = decoder

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def cif(self, hidden, alphas, threshold, log=False):
        batch_size, len_time, hidden_size = hidden.shape

        # loop vars
        integrate = torch.zeros([batch_size])
        frame = torch.zeros([batch_size, hidden_size])
        # intermediate vars along time
        list_fires = []
        list_frames = []

        for t in range(len_time):
            alpha = alphas[:, t]
            distribution_completion = torch.ones([batch_size]) - integrate

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate > threshold
            integrate = torch.where(fire_place,
                                 x=integrate - torch.ones([batch_size]),
                                 y=integrate)
            cur = torch.where(fire_place,
                           x=distribution_completion,
                           y=alpha)
            remainds = alpha - cur

            frame += cur[:, None] * hidden[:, t, :]
            list_frames.append(frame)
            frame = torch.where(torch.tile(fire_place[:, None], [1, hidden_size]),
                             x=remainds[:, None] * hidden[:, t, :],
                             y=frame)
            if log:
                print('t: {}\t{:.3f} -> {:.3f}|{:.3f}'.format(t, integrate[0].numpy(), cur[0].numpy(), remainds[0].numpy()))

        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        list_ls = []
        # len_labels = torch.cast(torch.round(torch.reduce_sum(alphas, -1)), torch.int32)
        len_labels = torch.cast(torch.math.ceil(torch.reduce_sum(alphas, -1)-0.001), torch.int32)
        max_label_len = torch.reduce_max(len_labels)
        for b in range(batch_size):
            fire = fires[b, :]
            l = torch.gather_nd(frames[b, :, :], torch.where(fire > threshold))
            pad_l = torch.zeros([max_label_len-l.shape[0], hidden_size])
            list_ls.append(torch.concat([l, pad_l], 0))

        if log:
            print('fire:\n', fires.numpy())

        return torch.stack(list_ls, 0)

    def forward(self, padded_input, input_lengths, padded_target, threshold=0.95):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax

        alpha = self.assigner(encoder_padded_outputs)
        # sum
        _num = alpha.sum(-1)
        # scaling
        num = (padded_target > 0).float().sum(-1)
        alpha *= torch.tile((num / _num)[:, None], [1, alpha.shape[1]])

        # cif
        l = self.cif(encoder_padded_outputs, alpha, threshold=threshold)

        pred, gold, *_ = self.decoder(padded_target, l, input_lengths)

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
        encoder_outputs, *_ = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 args)
        return nbest_hyps

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model, LFR_m, LFR_n = cls.load_model_from_package(package)
        return model, LFR_m, LFR_n

    @classmethod
    def load_model_from_package(cls, package):
        encoder = Encoder(package['d_input'],
                          package['n_layers_enc'],
                          package['n_head'],
                          package['d_k'],
                          package['d_v'],
                          package['d_model'],
                          package['d_inner'],
                          dropout=package['dropout'],
                          pe_maxlen=package['pe_maxlen'])
        decoder = Decoder(package['sos_id'],
                          package['eos_id'],
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
                          pe_maxlen=package['pe_maxlen'],
                          )
        model = cls(encoder, decoder)
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
            # decoder
            'sos_id': model.decoder.sos_id,
            'eos_id': model.decoder.eos_id,
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
