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
import torch
import torch.utils.data as data

from utils.utils import pad_list


class VQ_Dataset(data.Dataset):
    """
    TODO: this is a little HACK now, put batch_size here now.
          remove batch_size to dataloader later.
    """

    def __init__(self, data_path, token2idx, max_length_in, max_length_out,
                 down_sample_rate=1, batch_frames=0):
        # From: espnet/src/asr/asr_utils.py: make_batchset()
        """
        Args:
            data: espnet/espnet json format file.
            num_batches: for debug. only use num_batches minibatch but not all.
        """
        super().__init__()
        all_batches = []
        one_batch = []
        num_frames = 0
        for sample in self.sample_iter(data_path, token2idx, max_length_in):
            one_batch.append(sample)
            num_frames += len(sample)
            if num_frames > batch_frames:
                all_batches.append(one_batch[:-2])
                one_batch = [one_batch[-1]]
                num_frames = len(one_batch[0])
        if one_batch:
            all_batches.append(one_batch)
        self.all_batches = all_batches

    def __getitem__(self, index):
        return self.all_batches[index]

    def __len__(self):
        return len(self.all_batches)

    def sample_iter(self, data_path, token2idx, max_length):
        with open(data_path) as f:
            for line in f:
                uttid, tokens = line.strip().split(maxsplit=1)
                tokens = tokens.split()
                for i in range(0, len(tokens), max_length):
                    yield [token2idx[token] for token in tokens[i: i+max_length]]

def f_x_pad(batch):
    return pad_list([torch.tensor(sample).long() for sample in batch[0]], 0)


class VQ_Pred_Dataset(data.Dataset):
    """
    TODO: this is a little HACK now, put batch_size here now.
          remove batch_size to dataloader later.
    """

    def __init__(self, f_vq, f_trans, tokenIn2idx, tokenOut2idx, batch_size,
                 max_length_in, max_length_out, down_sample_rate=1):
        # From: espnet/src/asr/asr_utils.py: make_batchset()
        """
        Args:
            data: espnet/espnet json format file.
            num_batches: for debug. only use num_batches minibatch but not all.
        """
        super().__init__()
        self.all_batches = []
        one_batch = []

        with open(f_vq) as f1, open(f_trans) as f2:
            for vq, trans in zip(f1, f2):
                uttid, vq = vq.strip().split(maxsplit=1)
                _uttid, trans = trans.strip().split(maxsplit=1)
                assert uttid == _uttid
                x = [tokenIn2idx[token] for token in vq.split()]
                y = [tokenOut2idx[token] for token in trans.split()]
                one_batch.append([x, y])

                if len(one_batch) >= batch_size:
                    self.all_batches.append(one_batch)
                    one_batch = []

    def __getitem__(self, index):
        return self.all_batches[index]

    def __len__(self):
        return len(self.all_batches)


def f_xy_pad(batch):
    xs_pad = pad_list([torch.tensor(sample[0]).long() for sample in batch[0]], 0)
    ys_pad = pad_list([torch.tensor(sample[1]).long() for sample in batch[0]], 0)
    # xs_pad = pad_to_batch([sample for sample in batch[0][0]], 0)
    # ys_pad = pad_to_batch([sample for sample in batch[0][1]], 0)

    return xs_pad, ys_pad
