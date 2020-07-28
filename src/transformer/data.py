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
    def __init__(self, json_path, token2idx, batch_size=None, frames_size=None,
                 len_in_max=None, len_out_max=None, rate_in_out=None):
        # From: espnet/src/asr/asr_utils.py: make_batchset()
        """
        Args:
            len_in_max:     filter input seq length > len_in_max
            len_out_max:    filter output seq length > len_out_max
            rate_in_out:    tuple, filter len_seq_in/len_seq_out not in rate_in_out
            batch_size:     batched by batch_size
            frames_size:    batched by frames_size
            log:            logging if samples filted
        """
        super().__init__()
        try:
            # json_path is a single file
            with open(json_path) as f:
                data = json.load(f)
        except:
            # json_path is a dir where *.json in
            data = []
            for dir, _, fs in os.walk(os.path.dirname(json_path)):   # os.walk获取所有的目录
                for f in fs:
                    if f.endswith('.json'):  # 判断是否是".json"结尾
                        filename = os.path.join(dir, f)
                        with open(filename) as f:
                            data.extend(json.load(f))

        # filter samples
        list_to_pop = []
        for i, sample in enumerate(data):
            len_x = sample['input_length']
            len_y = sample['output_length']
            if not 0 < len_x <= len_in_max:
                list_to_pop.append(i)
            elif not 0 < len_y <= len_out_max:
                list_to_pop.append(i)
            elif rate_in_out and not (rate_in_out[0] <= (len_x / len_y) <= rate_in_out[1]):
                list_to_pop.append(i)

            # gen token_ids
            sample['token_ids'] = [token2idx[token] for token in sample['token'].split()]

        print('filtered {} samples:\n{}'.format(
            len(list_to_pop), ', '.join(data[i]['uttid'] for i in list_to_pop)))
        list_to_pop.reverse()
        [data.pop(i) for i in list_to_pop]

        # sort it by input lengths (long to short)
        data_sorted = sorted(data, key=lambda data: sample['input_length'], reverse=True)
        # change batchsize depending on the input and output length
        minibatch = []
        # Method 1: Generate minibatch based on batch_size
        # i.e. each batch contains #batch_size utterances
        if batch_size:
            start = 0
            while True:
                end = start
                ilen = data_sorted[end]['input_length']
                olen = data_sorted[end]['output_length']
                factor = max(int(ilen / len_in_max), int(olen / len_out_max))
                b = max(1, int(batch_size / (1 + factor)))
                end = min(len(data_sorted), start + b)
                minibatch.append(data_sorted[start:end])
                if end == len(data_sorted):
                    break
                start = end
        # Method 2: Generate minibatch based on frames_size
        # i.e. each batch contains approximately #frames_size frames
        elif frames_size:  # frames_size > 0
            print("NOTE: Generate minibatch based on frames_size.")
            print("i.e. each batch contains approximately #frames_size frames")
            start = 0
            while True:
                total_frames = 0
                end = start
                while total_frames < frames_size and end < len(data_sorted):
                    ilen = data_sorted[end]['input_length']
                    total_frames += ilen
                    end += 1
                minibatch.append(data_sorted[start:end])
                if end == len(data_sorted):
                    break
                start = end
        else:
            assert batch_size or frames_size
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class batch_generator(object):
    def __init__(self, sos_id=None, eos_id=None, LFR_m=1, LFR_n=1):
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.LFR_m = LFR_m
        self.LFR_n = LFR_n

    def __call__(self, batch):
        uttids, xs, ys = [], [], []
        for sample in batch[0]:
            uttids.append(sample['uttid'])

            x = build_LFR_features(kaldi_io.read_mat(sample['feat']), self.LFR_m, self.LFR_n)
            xs.append(torch.tensor(x).float())

            y = sample['token_ids']
            ys.append(torch.tensor(y).long())

        xs_pad, len_xs = pad_list(xs, 0)
        ys_pad, len_ys = pad_list(ys, 0)

        return uttids, xs_pad, len_xs, ys_pad, len_ys



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
