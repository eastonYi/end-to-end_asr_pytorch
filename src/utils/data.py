"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.
"""
import json
import os
import numpy as np
import torch
import torch.utils.data as data
import kaldi_io

from utils.utils import pad_list


class AudioDataset(data.Dataset):
    """
    TODO: this is a little HACK now, put batch_size here now.
          remove batch_size to dataloader later.
    """

    def __init__(self, data_json_path, batch_size, max_length_in, max_length_out,
                 num_batches=0, batch_frames=0):
        # From: espnet/src/asr/asr_utils.py: make_batchset()
        """
        Args:
            data: espnet/espnet json format file.
            num_batches: for debug. only use num_batches minibatch but not all.
        """
        super().__init__()
        try:
            with open(data_json_path, 'rb') as f:
                data = json.load(f)['utts']
        except:
            data = {}
            for fpathe, _, fs in os.walk(os.path.dirname(data_json_path)):   # os.walk获取所有的目录
                for f in fs:
                    if f.endswith('.json'):  # 判断是否是".sfx"结尾
                        filename = os.path.join(fpathe, f)
                        with open(filename, 'rb') as f:
                            data = dict(list(data.items()) +
                                        list(json.load(f)['utts'].items()))

        list_to_pop = []
        for key, sample in data.items():
            len_x = int(sample['input'][0]['shape'][0])
            len_y = int(sample['output'][0]['shape'][0])
            if len_x / len_y < 5.0:
                list_to_pop.append(key)

        [data.pop(i) for i in list_to_pop]
        # sort it by input lengths (long to short)
        sorted_data = sorted(data.items(), key=lambda data: int(
            data[1]['input'][0]['shape'][0]), reverse=True)
        # change batchsize depending on the input and output length
        minibatch = []
        # Method 1: Generate minibatch based on batch_size
        # i.e. each batch contains #batch_size utterances
        if batch_frames == 0:
            start = 0
            while True:
                ilen = int(sorted_data[start][1]['input'][0]['shape'][0])
                olen = int(sorted_data[start][1]['output'][0]['shape'][0])
                factor = max(int(ilen / max_length_in), int(olen / max_length_out))
                # if ilen = 1000 and max_length_in = 800
                # then b = batchsize / 2
                # and max(1, .) avoids batchsize = 0
                b = max(1, int(batch_size / (1 + factor)))
                end = min(len(sorted_data), start + b)
                minibatch.append(sorted_data[start:end])
                # DEBUG
                # total= 0
                # for i in range(start, end):
                #     total += int(sorted_data[i][1]['input'][0]['shape'][0])
                # print(total, end-start)
                if end == len(sorted_data):
                    break
                start = end
        # Method 2: Generate minibatch based on batch_frames
        # i.e. each batch contains approximately #batch_frames frames
        else:  # batch_frames > 0
            print("NOTE: Generate minibatch based on batch_frames.")
            print("i.e. each batch contains approximately #batch_frames frames")
            start = 0
            while True:
                total_frames = 0
                end = start
                while total_frames < batch_frames and end < len(sorted_data):
                    ilen = int(sorted_data[end][1]['input'][0]['shape'][0])
                    total_frames += ilen
                    end += 1
                # print(total_frames, end-start)
                minibatch.append(sorted_data[start:end])
                if end == len(sorted_data):
                    break
                start = end
        if num_batches > 0:
            minibatch = minibatch[:num_batches]
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, token2idx=None, LFR_m=1, LFR_n=1, label_type='token', **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = LFRCollate(token2idx, label_type, LFR_m=LFR_m, LFR_n=LFR_n)


class LFRCollate(object):
    """Build this wrapper to pass arguments(LFR_m, LFR_n) to _collate_fn"""
    def __init__(self, token2idx, label_type, LFR_m=1, LFR_n=1):
        self.token2idx = token2idx
        self.label_type = label_type
        self.LFR_m = LFR_m
        self.LFR_n = LFR_n

    def __call__(self, batch):
        return _collate_fn(batch, self.token2idx, self.label_type, LFR_m=self.LFR_m, LFR_n=self.LFR_n)


def _collate_fn(batch, token2idx, label_type, LFR_m=1, LFR_n=1):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        xs_pad: N x Ti x D, torch.Tensor
        ilens : N, torch.Tentor
        ys_pad: N x To, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1
    batch = load_inputs_and_targets(batch[0], token2idx, label_type,
        LFR_m=LFR_m, LFR_n=LFR_n)
    xs, ys = batch

    # TODO: perform subsamping

    # get batch of lengths of input sequences
    ilens = np.array([x.shape[0] for x in xs])

    # perform padding and convert to tensor
    xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0)
    ilens = torch.from_numpy(ilens)
    ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], 0)

    return xs_pad, ilens, ys_pad


# ------------------------------ utils ------------------------------------
def load_inputs_and_targets(batch, token2idx, label_type, LFR_m, LFR_n):
    # From: espnet/src/asr/asr_utils.py: load_inputs_and_targets
    # load acoustic features and target sequence of token ids
    # for b in batch:
    #     print(b[1]['input'][0]['feat'])
    xs = [kaldi_io.read_mat(b[1]['input'][0]['feat']) for b in batch]
    ys = [b[1]['output'][0][label_type].split() for b in batch]

    if LFR_m != 1 or LFR_n != 1:
        # xs = build_LFR_features(xs, LFR_m, LFR_n)
        xs = [build_LFR_features(x, LFR_m, LFR_n) for x in xs]

    # get index of non-zero length samples
    nonzero_idx = filter(lambda i: len(ys[i]) > 0, range(len(xs)))
    # sort in input lengths
    nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
    if len(nonzero_sorted_idx) != len(xs):
        print("warning: Target sequences include empty token")

    # remove zero-lenght samples
    xs = [xs[i] for i in nonzero_sorted_idx]
    ys = [np.fromiter(map(lambda x: token2idx[x], ys[i]), dtype=np.int64)
          for i in nonzero_sorted_idx]

    return xs, ys


def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.

    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i*n:i*n+m]))
        else: # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)
    #     LFR_inputs_batch.append(np.vstack(LFR_inputs))
    # return LFR_inputs_batch
