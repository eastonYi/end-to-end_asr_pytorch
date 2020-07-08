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
import numpy as np
import torch
import torch.utils.data as data

from utils.utils import pad_list


class VQ_Dataset(data.Dataset):
    """
    TODO: this is a little HACK now, put batch_size here now.
          remove batch_size to dataloader later.
    """

    def __init__(self, data_path, batch_size, max_length_in, max_length_out,
                 down_sample_rate=1, num_batches=0, batch_frames=0):
        # From: espnet/src/asr/asr_utils.py: make_batchset()
        """
        Args:
            data: espnet/espnet json format file.
            num_batches: for debug. only use num_batches minibatch but not all.
        """
        super().__init__()
        minibatch = []
        one_batch = []
        num_frames = 0
        for sample in self.sample_iter(data_path, max_length_in):
            one_batch.append(sample)
            num_frames += len(sample)
            if num_frames > batch_frames:
                minibatch.append(one_batch[:-2])
                one_batch = [one_batch[-1]]
                num_frames = len(one_batch[0])
        if one_batch:
            minibatch.append(one_batch)
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

    def sample_iter(self, data_path, max_length):
        with open(data_path) as f:
            for line in f:
                tokens = line.strip().split()
                for i in range(0, len(tokens), max_length):
                    yield tokens[i: i+max_length]


class VQ_DataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, token2idx=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = LFRCollate(token2idx)


class LFRCollate(object):
    """Build this wrapper to pass arguments(LFR_m, LFR_n) to _collate_fn"""
    def __init__(self, token2idx):
        self.token2idx = token2idx

    def __call__(self, batches):
        return _collate_fn(batches, self.token2idx)


def _collate_fn(batches, token2idx):
    ys = [np.fromiter((token2idx[token] for token in batch), dtype=np.int64)
          for batch in batches[0]]

    ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], 0)

    return ys_pad
