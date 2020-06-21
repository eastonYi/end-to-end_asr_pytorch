import torch
import torch.nn as nn


class CIF_Model(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, conv_encoder, encoder, assigner, decoder):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.encoder = encoder
        self.assigner = assigner
        self.decoder = decoder
        self.ctc_fc = nn.Linear(encoder.d_output, decoder.d_output, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, len_features, targets, threshold=0.95, random_scale=False):
        """
        Args:
            features: N x T x D
            len_sequence: N
            padded_targets: N x To
        """
        conv_outputs, len_sequence = self.conv_encoder(features, len_features)
        encoder_outputs = self.encoder(conv_outputs, len_sequence)

        ctc_logits = self.ctc_fc(encoder_outputs)
        len_ctc_logits = len_sequence

        alpha = self.assigner(encoder_outputs, len_sequence)

        # sum
        _num = alpha.sum(-1)
        # scaling
        num = (targets > 0).float().sum(-1)
        if random_scale:
            # random (-0.5, 0.5]
            num_noise = num + torch.rand(alpha.size(0)).cuda() - 0.5
        alpha *= (num_noise / _num)[:, None].repeat(1, alpha.size(1))

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
        len_labels = torch.round(alphas.sum(-1)).int()
        max_label_len = len_labels.max()
        for b in range(batch_size):
            fire = fires[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.where(fire > threshold)[0])
            pad_l = torch.zeros([max_label_len - l.size(0), hidden_size]).cuda()
            list_ls.append(torch.cat([l, pad_l], 0))

        if log:
            print('fire:\n', fires.numpy())

        return torch.stack(list_ls, 0)

    def recognize(self, input, input_length, char_list, args, threshold=0.95, target_num=None):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        conv_padded_outputs, input_length = self.conv_encoder(input.unsqueeze(0), input_length)
        encoder_outputs = self.encoder(conv_padded_outputs, input_length)

        alpha = self.assigner(encoder_outputs, input_length)
        if target_num:
            _num = alpha.sum(-1)
            num = target_num
            alpha *= (num / _num)[:, None].repeat(1, alpha.size(1))

        l = self.cif(encoder_outputs, alpha, threshold=threshold)

        nbest_hyps = self.decoder.recognize_beam(l, char_list, args)
        # nbest_hyps = self.decoder.recognize_beam_cache(l, char_list, args)

        return nbest_hyps

    @classmethod
    def create_model(cls, args):
        from transformer.conv_encoder import Conv2dSubsample
        from transformer.encoder import Encoder
        # from transformer.attentionAssigner import Attention_Assigner
        from transformer.attentionAssigner import Attention_Assigner_RNN as Attention_Assigner
        from transformer.decoder import Decoder_CIF as Decoder

        conv_encoder = Conv2dSubsample(args.d_input * args.LFR_m, args.d_model,
                                       n_layers=args.n_conv_layers)
        encoder = Encoder(args.d_model, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        assigner = Attention_Assigner(d_input=args.d_model,
                                      d_hidden=args.d_assigner_hidden,
                                      w_context=args.w_context,
                                      n_layers=args.n_assigner_layers)
        decoder = Decoder(args.sos_id, args.vocab_size, args.d_model, args.n_layers_dec,
                          args.n_head, args.d_k, args.d_v, args.d_model,
                          args.d_inner, dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)

        return conv_encoder, encoder, assigner, decoder

    @classmethod
    def load_model(cls, path, args):

        # creat mdoel
        conv_encoder, encoder, assigner, decoder = cls.create_model(args)
        model = cls(conv_encoder, encoder, assigner, decoder)

        # load params
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model.load_state_dict(package['state_dict'])

        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package
