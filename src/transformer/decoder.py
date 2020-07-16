import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.attention import MultiheadAttention
from transformer.encoder import EncoderLayer
from transformer.module import PositionalEncoding, PositionwiseFeedForward
from utils.utils import get_attn_key_pad_mask, get_attn_pad_mask, get_subsequent_mask, pad_list


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, sos_id, eos_id, n_tgt_vocab, n_layers, n_head,
                 d_model, d_inner, dropout=0.1):
        super().__init__()
        # parameters
        self.sos_id = sos_id  # Start of Sentence
        self.eos_id = eos_id  # End of Sentence
        self.n_tgt_vocab = n_tgt_vocab
        self.d_word_vec = d_model
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_output = n_tgt_vocab
        self.dropout = dropout

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, self.d_word_vec)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, dropout=dropout)
            for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

    def preprocess(self, targets):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label
        """
        ys = [y[y != 0] for y in targets]  # parse padded ys
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, 0)
        ys_out_pad = pad_list(ys_out, 0)
        assert ys_in_pad.size() == ys_out_pad.size()

        return ys_in_pad, ys_out_pad

    def forward(self, targets, encoder_padded_outputs, encoder_input_lengths):
        """
        Args:
            padded_input: N x To
            encoder_padded_outputs: N x Ti x H

        Returns:
        """
        # Get Deocder Input and Output
        targets_sos, targets_eos = self.preprocess(targets)

        # Prepare masks
        non_pad_mask = (targets_sos > 0).unsqueeze(-1)

        slf_attn_mask_subseq = get_subsequent_mask(targets_sos)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=targets_sos,
                                                     seq_q=targets_sos,
                                                     pad_idx=0)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        output_length = targets_sos.size(1)
        dec_enc_attn_mask = get_attn_pad_mask(encoder_input_lengths, output_length)

        # Forward
        dec_output = self.dropout(self.tgt_word_emb(targets_sos) +
                                  self.positional_encoding(targets_sos))

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output, encoder_padded_outputs,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

        # before softmax
        logits = self.tgt_word_prj(dec_output)

        return logits, targets_eos

    def step_forward(self, ys, encoder_outputs):
        # -- Prepare masks
        non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # 1xix1
        slf_attn_mask = get_subsequent_mask(ys)

        # -- Forward
        dec_output = self.tgt_word_emb(ys) + self.positional_encoding(ys)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output, encoder_outputs,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=None)

        seq_logit = self.tgt_word_prj(dec_output[:, -1])

        local_scores = F.log_softmax(seq_logit, dim=1)

        return local_scores

    def recognize_beam(self, encoder_outputs, char_list, args):
        """Beam search, decode one utterence now.
        Args:
            encoder_outputs: T x H
            char_list: list of character
            args: args.beam

        Returns:
            nbest_hyps:
        """
        # search params
        beam = args.beam_size
        nbest = args.nbest
        if args.decode_max_len == 0:
            maxlen = encoder_outputs.size(0)
        else:
            maxlen =  min(args.decode_max_len, encoder_outputs.size(0))

        encoder_outputs = encoder_outputs.unsqueeze(0)

        # prepare sos
        ys = torch.ones(1, 1).fill_(self.sos_id).type_as(encoder_outputs).long()

        # yseq: 1xT
        hyp = {'score': 0.0, 'yseq': ys}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                ys = hyp['yseq']  # 1 x i

                local_scores = self.step_forward(ys, encoder_outputs)

                # topk scores
                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1)

                for j in range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = torch.ones(1, (1+ys.size(1))).type_as(encoder_outputs).long()
                    new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq']
                    new_hyp['yseq'][:, ys.size(1)] = int(local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

            hyps = sorted(hyps_best_kept,
                          key=lambda x: x['score'],
                          reverse=True)[:beam]

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'] = torch.cat([hyp['yseq'],
                                             torch.ones(1, 1).fill_(self.eos_id).type_as(encoder_outputs).long()], dim=1)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][0, -1] == self.eos_id:
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            hyps = remained_hyps
            if len(hyps) > 0:
                print('remeined hypothes: ' + str(len(hyps)))
            else:
                print('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                print('hypo: ' + ''.join([char_list[int(x)]
                                          for x in hyp['yseq'][0, 1:]]))
        # end for i in range(maxlen)
        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
            :min(len(ended_hyps), nbest)]
        # compitable with LAS implementation
        for hyp in nbest_hyps:
            hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()

        return [hyp['yseq'] for hyp in nbest_hyps], [len(hyp['yseq']) for hyp in nbest_hyps]


class Decoder_CIF(Decoder):
    """Encoder of Transformer including self-attention and feed forward.
    """
    def __init__(self, sos_id, n_tgt_vocab, n_layers, n_head, d_model, d_inner, dropout=0.1):
        # parameters
        nn.Module.__init__(self)
        # parameters
        self.sos_id = sos_id  # Start of Sentence
        self.n_tgt_vocab = n_tgt_vocab
        self.d_word_vec = d_model
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_output = n_tgt_vocab
        self.dropout = dropout

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, self.d_word_vec)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, dropout=dropout)
            for _ in range(n_layers)])
        self.input_affine = nn.Linear(2*d_model, d_model, bias=False)

        self.tgt_word_prj = nn.Linear(2*d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

    def preprocess(self, target):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label
        """
        pad_mask = (target > 0).long()
        sos = torch.ones(target.size(0), 1).fill_(self.sos_id).long().cuda()
        ys_in = torch.cat([sos, target[:, :-1]], 1)
        ys_in *= pad_mask

        return ys_in

    def forward(self, encoded_attentioned, target):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns:
            enc_output: N x T x H
        """
        # Prepare masks
        ys_in = self.preprocess(target)
        non_pad_mask = (target > 0).unsqueeze(-1)

        slf_attn_mask_subseq = get_subsequent_mask(ys_in)
        slf_attn_mask_keypad = get_attn_key_pad_mask(
            seq_k=ys_in, seq_q=ys_in, pad_idx=0)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        ys_in_emb = self.dropout(self.tgt_word_emb(ys_in) + self.positional_encoding(ys_in))

        dec_output = self.input_affine(torch.cat([encoded_attentioned, ys_in_emb], -1))

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        dec_output = torch.cat([encoded_attentioned, dec_output], -1)

        logits = self.tgt_word_prj(dec_output)

        return logits

    def step_forward(self, ys, encoded_attentioned, t):
        # -- Prepare masks
        non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # 1xix1
        slf_attn_mask = get_subsequent_mask(ys)

        # -- Forward
        target_emb = self.tgt_word_emb(ys) + self.positional_encoding(ys)
        dec_output = self.input_affine(torch.cat([encoded_attentioned[:, :t+1, :], target_emb], -1))

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        dec_output = torch.cat([encoded_attentioned[:, :t+1, :], dec_output], -1)

        seq_logit = self.tgt_word_prj(dec_output[:, -1])

        local_scores = F.log_softmax(seq_logit, dim=1)

        return local_scores

    def recognize_beam(self, encoded_attentioned, char_list, args):
        """Beam search, decode one utterence now.
        Args:
            encoder_outputs: T x H
            char_list: list of character
            args: args.beam

        Returns:
            nbest_hyps:
        """
        # search params
        beam = args.beam_size
        nbest = args.nbest
        maxlen = encoded_attentioned.size(1)

        # prepare sos
        ys = torch.ones(1, 1).fill_(self.sos_id).long().cuda()

        # yseq: 1xT
        hyp = {'score': 0.0, 'yseq': ys}
        hyps = [hyp]

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                ys = hyp['yseq']  # 1 x i
                local_scores = self.step_forward(ys, encoded_attentioned, i)

                # topk scores
                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1)

                for j in range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = torch.ones(1, (1 + ys.size(1))).long().cuda()
                    new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq']
                    new_hyp['yseq'][:, ys.size(1)] = int(local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

            hyps = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

        # end for i in range(maxlen)
        nbest_hyps = sorted(hyps, key=lambda x: x['score'], reverse=True)[:min(len(hyps), nbest)]

        # compitable with LAS implementation
        for hyp in nbest_hyps:
            hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()
            # print('hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

        return [hyp['yseq'] for hyp in nbest_hyps], [len(hyp['yseq']) for hyp in nbest_hyps]

    def step_forward_cache(self, ys, enc_attentioned, dec_cache, t):
        # -- Forward
        target_emb = self.tgt_word_emb(ys) + self.positional_encoding(ys)
        dec_output = self.input_affine(torch.cat([enc_attentioned[:, :t+1, :], target_emb], -1))
        new_cache = []

        for i, dec_layer in enumerate(self.layer_stack):
            dec_output = dec_layer.forward_cache(dec_output)
            dec_output = torch.cat([dec_cache[:, :, i, :], dec_output], axis=1)
            new_cache.append(dec_output.unsqueeze(2))

        new_cache = torch.cat(new_cache, axis=2)  # [batch_size, n_step, num_blocks, num_hidden]

        dec_output = torch.cat([enc_attentioned[:, :t+1, :], dec_output], -1)

        seq_logit = self.tgt_word_prj(dec_output[:, -1])

        local_scores = F.log_softmax(seq_logit, dim=1)

        return local_scores, new_cache

    def recognize_beam_cache(self, encoded_attentioned, char_list, args):
        """Beam search, decode one utterence now.
        Args:
            encoder_outputs: T x H
            char_list: list of character
            args: args.beam

        Returns:
            nbest_hyps:
        """
        # search params
        beam = args.beam_size
        nbest = args.nbest
        maxlen = encoded_attentioned.size(1)

        # prepare sos
        ys = torch.ones(1, 1).fill_(self.sos_id).long().cuda()

        # yseq: 1xT
        hyp = {'score': 0.0, 'yseq': ys,
               'cache': torch.zeros([1, 0, self.n_layers, self.d_model]).cuda()}
        hyps = [hyp]

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                ys = hyp['yseq']  # 1 x i
                local_scores, cache_decoder = self.step_forward_cache(
                    ys, encoded_attentioned, hyp['cache'], i)

                # topk scores
                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1)

                for j in range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = torch.ones(1, (1 + ys.size(1))).long().cuda()
                    new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq']
                    new_hyp['yseq'][:, ys.size(1)] = int(local_best_ids[0, j])
                    new_hyp['cache'] = cache_decoder
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

            hyps = sorted(hyps_best_kept,
                          key=lambda x: x['score'],
                          reverse=True)[:beam]

        # end for i in range(maxlen)
        nbest_hyps = sorted(hyps, key=lambda x: x['score'], reverse=True)[:min(len(hyps), nbest)]

        # compitable with LAS implementation
        for hyp in nbest_hyps:
            hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()
            print('hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

        return [hyp['yseq'] for hyp in nbest_hyps], [len(hyp['yseq']) for hyp in nbest_hyps]


    def recognize_batch_beam_cache(self, encoded_attentioned, char_list, args):
        """Beam search, decode one utterence now.
        Args:
            encoder_outputs: T x H
            char_list: list of character
            args: args.beam

        Returns:
            nbest_hyps:
        """
        # search params
        batch = encoded_attentioned.size(0)
        beam = args.beam_size
        nbest = args.nbest
        maxlen = encoded_attentioned.size(1)

        # prepare sos
        ys = torch.ones(batch * beam, 1).fill_(self.sos_id).long().cuda()

        # yseq: 1xT
        hyp = {'score': 0.0,
               'yseq': ys,
               'cache': torch.zeros([batch * beam, 0, self.n_layers, self.d_model]).cuda()}
        hyps = [hyp]

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                ys = hyp['yseq']  # 1 x i
                local_scores, cache_decoder = self.step_forward_cache(
                    ys, encoded_attentioned, hyp['cache'], i)

                # topk scores
                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1)

                for j in range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = torch.ones(1, (1 + ys.size(1))).long().cuda()
                    new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq']
                    new_hyp['yseq'][:, ys.size(1)] = int(local_best_ids[0, j])
                    new_hyp['cache'] = cache_decoder
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

            hyps = sorted(hyps_best_kept,
                          key=lambda x: x['score'],
                          reverse=True)[:beam]

        # end for i in range(maxlen)
        nbest_hyps = sorted(hyps, key=lambda x: x['score'], reverse=True)[:min(len(hyps), nbest)]

        # compitable with LAS implementation
        for hyp in nbest_hyps:
            hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()
            print('hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

        return [hyp['yseq'] for hyp in nbest_hyps], [len(hyp['yseq']) for hyp in nbest_hyps]


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.enc_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output

    def forward_cache(self, dec_input, enc_output,
                      non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input[:, -1: , :], dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output
