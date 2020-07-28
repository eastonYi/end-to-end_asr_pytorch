#!/usr/bin/env python3
from collections import defaultdict


def pad_list(xs, pad_value, max_len=None):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    lengths = torch.tensor([x.size(0) for x in xs]).long()
    max_len = lengths.max() if not max_len else max_len
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad, lengths


def process_dict(dict_path):
    with open(dict_path, 'rb') as f:
        dictionary = f.readlines()
    char_list = [entry.decode('utf-8').split(' ')[0]
                 for entry in dictionary]
    sos_id = char_list.index('<sos>')
    eos_id = char_list.index('<eos>')

    return char_list, sos_id, eos_id


def load_vocab(path, vocab_size=None):
    with open(path, encoding='utf8') as f:
        vocab = [line.strip().split()[0] for line in f]
    vocab = vocab[:vocab_size] if vocab_size else vocab
    id_unk = vocab.index('<unk>')
    token2idx = defaultdict(lambda: id_unk)
    idx2token = defaultdict(lambda: '<unk>')
    token2idx.update({token: idx for idx, token in enumerate(vocab)})
    idx2token.update({idx: token for idx, token in enumerate(vocab)})
    idx2token[token2idx['<pad>']] = ''
    idx2token[token2idx['<blk>']] = ''
    idx2token[token2idx['<unk>']] = '<UNK>'
    idx2token[token2idx['<sos>']] = ''
    idx2token[token2idx['<eos>']] = ''

    assert len(token2idx) == len(idx2token)

    return token2idx, idx2token


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    char_list, sos_id, eos_id = process_dict(path)
    print(char_list, sos_id, eos_id)

# * ------------------ recognition related ------------------ *
def parse_ctc_hypothesis(hyp, char_list):
    """Function to parse hypothesis

    :param list hyp: recognition hypothesis
    :param list char_list: list of characters
    :return: recognition text strinig
    :return: recognition token strinig
    :return: recognition tokenid string
    """
    tokenid_as_list = list(map(int, hyp))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, 0.0


def parse_hypothesis(hyp, char_list):
    """Function to parse hypothesis

    :param list hyp: recognition hypothesis
    :param list char_list: list of characters
    :return: recognition text strinig
    :return: recognition token strinig
    :return: recognition tokenid string
    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp['score'])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score


def ids2str(hyps_ints, idx2token):
    list_res = []
    hyp = ''
    for hyp_ints, length in zip(*hyps_ints):
        for idx in hyp_ints[:length]:
            hyp += idx2token[idx]
        list_res.append(hyp)

    return list_res


# -- Transformer Related --
import torch


def pad_to_batch(xs, pad_value):
    """
    xs: nested list [[...], [...], ...]
    """
    lens = [len(x) for x in xs]
    max_len = max(lens)

    for l, x in zip(lens, xs):
        x.extend([pad_value] * (max_len - l))

    return torch.tensor(xs)


def sequence_mask(lengths, maxlen=None, dtype=torch.float):
    if maxlen is None:
        maxlen = lengths.max()
    mask = torch.ones((len(lengths), maxlen),
                      device=lengths.device,
                      dtype=torch.uint8).cumsum(dim=1) <= lengths.unsqueeze(0).t()

    return mask.type(dtype)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.le(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_attn_pad_mask(input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    non_pad_mask = sequence_mask(input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask < 1.0
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)

    return attn_mask


def spec_aug(padded_features, feature_lengths, config):
    # print('using spec_aug:', config)
    freq_mask_num, freq_mask_width, time_mask_num, time_mask_width = (int(i) for i in config.split('-'))
    freq_means = torch.mean(padded_features, dim=-1)
    time_means = (torch.sum(padded_features, dim=1)
            /feature_lengths[:, None].float()) # Note that features are padded with zeros.

    B, T, V = padded_features.shape
    # mask freq
    for _ in range(time_mask_num):
        fs = (freq_mask_width * torch.rand(size=[B],
            device=padded_features.device, requires_grad=False)).long()
        f0s = ((V-fs).float()*torch.rand(size=[B],
            device=padded_features.device, requires_grad=False)).long()
        for b in range(B):
            padded_features[b, :, f0s[b]:f0s[b]+fs[b]] = freq_means[b][:, None]

    # mask time
    for _ in range(time_mask_num):
        ts = (time_mask_width * torch.rand(size=[B],
            device=padded_features.device, requires_grad=False)).long()
        t0s = ((feature_lengths-ts).float()*torch.rand(size=[B],
            device=padded_features.device, requires_grad=False)).long()
        for b in range(B):
            padded_features[b, t0s[b]:t0s[b]+ts[b], :] = time_means[b][None, :]

    return padded_features, feature_lengths
