#!/usr/bin/env python
import argparse
import json
import os
import time
import torch
import kaldi_io

from utils.utils import load_vocab, ids2str, add_results_to_json
from utils.data import build_LFR_features

parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Decoding.")
# data
parser.add_argument('--type', type=str, required=True,
                    help='test or infer')
parser.add_argument('--recog-json', type=str, required=True,
                    help='Filename of recognition data (json)')
parser.add_argument('--structure', type=str, default='transformer',
                    help='transformer transformer-ctc conv-transformer-ctc')
parser.add_argument('--model-path', type=str, required=True,
                    help='Filename of recognition data (json)')
parser.add_argument('--vocab', type=str, required=True,
                    help='Dictionary which should include <unk> <sos> <eos>')
parser.add_argument('--output', type=str, required=True,
                    help='Filename of result label data (json)')
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


def test(args):
    if args.structure == 'transformer':
        from transformer.Transformer import Transformer
    elif args.structure == 'transformer-ctc':
        from transformer.Transformer import CTC_Transformer as Transformer
    elif args.structure == 'conv-transformer-ctc':
        from transformer.Transformer import Conv_CTC_Transformer as Transformer

    model, LFR_m, LFR_n = Transformer.load_model(args.model_path)
    print(model)
    model.eval()
    model.cuda()
    token2idx, idx2token = load_vocab(args.vocab)

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    new_js = {}

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
            hyps_ints = model.recognize(input, input_length, idx2token, args)
            hyp = ids2str(hyps_ints, idx2token)[0]
            f.write(name + ' ' + hyp + '\n')
    #         new_js[name] = add_results_to_json(js[name], hyps_ints, idx2token)
    #
    # with open(args.output, 'wb') as f:
    #     f.write(json.dumps({'utts': new_js}, indent=4,
    #                        sort_keys=True).encode('utf_8'))


def infer(args):
    return


if __name__ == "__main__":
    args = parser.parse_args()
    print(args, flush=True)
    if args.type == 'test':
        test(args)
    elif args.type == 'infer':
        infer(args)
