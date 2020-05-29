#/usr/bin/python
#encoding=utf-8

#greedy decoder and beamsearch decoder for ctc

import torch
import numpy as np


class Decoder(object):
    "解码器基类定义，作用是将模型的输出转化为文本使其能够与标签计算正确率"
    def __init__(self, space_idx=1, blank_index=0):
        '''
        int2char     :     将类别转化为字符标签
        space_idx    :     空格符号的索引，如果为为-1，表示空格不是一个类别
        blank_index  :     空白类的索引，默认设置为0
        '''
        self.space_idx = space_idx
        self.blank_index = blank_index

    def __call__(self, prob_tensor, frame_seq_len=None):
        return self.decode(prob_tensor, frame_seq_len)

    def decode(self):
        "解码函数，在GreedyDecoder和BeamDecoder继承类中实现"
        raise NotImplementedError;

    def ctc_reduce_map(self, batch_samples, lengths):
        """
        inputs:
            batch_samples: size x time
        return:
            (padded_samples, mask): (size x max_len, size x max_len)
                                     max_len <= time
        """
        sents = []
        for align, length in zip(batch_samples, lengths):
            sent = []
            tmp = None
            for token in align[:length]:
                if token != self.blank_index and token != tmp:
                    sent.append(token)
                tmp = token
            sents.append(sent)

        return self.padding_list_seqs(sents, dtype=np.int32, pad=0)

    def padding_list_seqs(self, list_seqs, dtype=np.float32, pad=0.):
        len_x = [len(s) for s in list_seqs]

        size_batch = len(list_seqs)
        maxlen = max(len_x)

        shape_feature = tuple()
        for s in list_seqs:
            if len(s) > 0:
                shape_feature = np.asarray(s).shape[1:]
                break

        x = (np.ones((size_batch, maxlen) + shape_feature) * pad).astype(dtype)
        for idx, s in enumerate(list_seqs):
            x[idx, :len(s)] = s

        return x, len_x


class GreedyDecoder(Decoder):
    "直接解码，把每一帧的输出概率最大的值作为输出值，而不是整个序列概率最大的值"
    def decode(self, prob_tensor, frame_seq_len):
        '''解码函数
        Args:
            prob_tensor   :   网络模型输出
            frame_seq_len :   每一样本的帧数
        Returns:
            解码得到的string，即识别结果
        '''
        _, decoded = torch.max(prob_tensor, 2)
        decoded = decoded.view(decoded.size(0), decoded.size(1))

        return self.ctc_reduce_map(decoded, frame_seq_len)


class BeamDecoder(Decoder):
    "Beam search 解码。解码结果为整个序列概率的最大值"
    def __init__(self, beam_width = 200, blank_index = 0, space_idx = -1, lm_path=None, lm_alpha=0.01):
        self.beam_width = beam_width
        super().__init__(pace_idx=space_idx, blank_index=blank_index)

        import sys
        sys.path.append('../')
        import utils.BeamSearch as uBeam
        import utils.NgramLM as uNgram
        lm = uNgram.LanguageModel(arpa_file=lm_path)
        self._decoder = uBeam.ctcBeamSearch(beam_width, lm, lm_alpha=lm_alpha, blank_index = blank_index)

    def decode(self, prob_tensor, frame_seq_len=None):
        '''解码函数
        Args:
            prob_tensor   :   网络模型输出
            frame_seq_len :   每一样本的帧数
        Returns:
            res           :   解码得到的string，即识别结果
        '''
        probs = prob_tensor.transpose(0, 1)
        probs = torch.exp(probs)
        res = self._decoder.decode(probs, frame_seq_len)
        return res


if __name__ == '__main__':
    decoder = Decoder('abcde', 1, 2)
    print(decoder._convert_to_strings([[1,2,1,0,3],[1,2,1,1,1]]))
