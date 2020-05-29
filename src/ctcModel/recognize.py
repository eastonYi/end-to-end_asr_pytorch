#!/usr/bin/env python
import argparse
import json
import time
import torch
import kaldi_io

from ctcModel.ctc_model import CTC_Model
from utils.utils import load_vocab, ids2str
from data.data import build_LFR_features

parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Decoding.")
# data
parser.add_argument('--recog-json', type=str, required=True,
                    help='Filename of recognition data (json)')
parser.add_argument('--dict', type=str, required=True,
                    help='Dictionary which should include <unk> <sos> <eos>')
parser.add_argument('--output', type=str, required=True,
                    help='Filename of result label data (json)')
# model
parser.add_argument('--model-path', type=str, required=True,
                    help='Path to model file created by training')
# decode
parser.add_argument('--beam-size', default=1, type=int,
                    help='Beam size')
parser.add_argument('--nbest', default=1, type=int,
                    help='Nbest size')
parser.add_argument('--print-freq', default=1, type=int,
                    help='print_freq')
parser.add_argument('--decode-max-len', default=0, type=int,
                    help='Max output length. If ==0 (default), it uses a '
                    'end-detect function to automatically find maximum '
                    'hypothesis lengths')


def recognize(args):
    model, LFR_m, LFR_n = CTC_Model.load_model(args.model_path)
    print(model)
    model.eval()
    model.cuda()
    token2idx, idx2token = load_vocab(args.dict)
    blank_index = token2idx['<blk>']

    if args.beam_size == 1:
        from ctcModel.ctc_infer import GreedyDecoder

        decode = GreedyDecoder(space_idx=0, blank_index=blank_index)
    else:
        from ctcModel.ctc_infer import BeamDecoder

        decode = BeamDecoder(beam_width=args.beam_size, blank_index=blank_index, space_idx=0)

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    # decode each utterance
    with torch.no_grad(), open(args.output, 'w') as f:
        for idx, name in enumerate(js.keys(), 1):
            print('(%d/%d) decoding %s' %
                  (idx, len(js.keys()), name), flush=True)
            input = kaldi_io.read_mat(js[name]['input'][0]['feat'])  # TxD
            input = build_LFR_features(input, LFR_m, LFR_n)
            input = torch.from_numpy(input).float()
            input_length = torch.tensor([input.size(0)], dtype=torch.int)
            input = input.cuda()
            input_length = input_length.cuda()
            hyps_ints = model.recognize(input, input_length, decode, args)
            hyp = ids2str(hyps_ints, idx2token)[0]
            f.write(name + ' ' + hyp + '\n')


if __name__ == "__main__":
    args = parser.parse_args()
    print(args, flush=True)
    recognize(args)
