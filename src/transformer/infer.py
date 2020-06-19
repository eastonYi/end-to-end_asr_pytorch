#!/usr/bin/env python
import argparse
import json
import torch
import kaldi_io

from utils.utils import load_vocab, ids2str
from utils.data import build_LFR_features


parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Decoding.")
# data
parser.add_argument('--type', type=str, required=True,
                    help='test or infer')
parser.add_argument('--recog-json', type=str, required=True,
                    help='Filename of recognition data (json)')
parser.add_argument('--model-path', type=str, required=True,
                    help='Filename of recognition data (json)')
parser.add_argument('--vocab', type=str, required=True,
                    help='Dictionary which should include <unk> <sos> <eos>')
parser.add_argument('--output', type=str, required=True,
                    help='Filename of result label data (json)')

# Low Frame Rate (stacking and skipping frames)
parser.add_argument('--LFR_m', default=4, type=int,
                    help='Low Frame Rate: number of frames to stack')
parser.add_argument('--LFR_n', default=3, type=int,
                    help='Low Frame Rate: number of frames to skip')

# Network architecture
parser.add_argument('--structure', type=str, default='transformer',
                    help='transformer transformer-ctc conv-transformer-ctc')
# conv_encoder
parser.add_argument('--n_conv_layers', default=3, type=int,
                    help='Dimension of key')
# encoder
parser.add_argument('--d_input', default=80, type=int,
                    help='Dim of encoder input (before LFR)')
parser.add_argument('--n_layers_enc', default=6, type=int,
                    help='Number of encoder stacks')
parser.add_argument('--n_head', default=8, type=int,
                    help='Number of Multi Head Attention (MHA)')
parser.add_argument('--d_k', default=64, type=int,
                    help='Dimension of key')
parser.add_argument('--d_v', default=64, type=int,
                    help='Dimension of value')
parser.add_argument('--d_model', default=512, type=int,
                    help='Dimension of model')
parser.add_argument('--d_inner', default=2048, type=int,
                    help='Dimension of inner')
parser.add_argument('--dropout', default=0.1, type=float,
                    help='Dropout rate')
parser.add_argument('--pe_maxlen', default=5000, type=int,
                    help='Positional Encoding max len')
# assigner
parser.add_argument('--w_context', default=3, type=int,
                    help='Positional Encoding max len')
parser.add_argument('--d_assigner_hidden', default=512, type=int,
                    help='Positional Encoding max len')
parser.add_argument('--n_assigner_layers', default=3, type=int,
                    help='Positional Encoding max len')
# decoder
parser.add_argument('--d_word_vec', default=512, type=int,
                    help='Dim of decoder embedding')
parser.add_argument('--n_layers_dec', default=6, type=int,
                    help='Number of decoder stacks')
parser.add_argument('--tgt_emb_prj_weight_sharing', default=0, type=int,
                    help='share decoder embedding with decoder projection')

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
        from transformer.Transformer import Transformer as Model
    elif args.structure == 'transformer-ctc':
        from transformer.Transformer import CTC_Transformer as Model
    elif args.structure == 'conv-transformer-ctc':
        from transformer.Transformer import Conv_CTC_Transformer as Model
    elif args.structure == 'cif':
        from transformer.CIF_Model import CIF_Model as Model

    token2idx, idx2token = load_vocab(args.vocab)
    args.sos_id = token2idx['<sos>']
    args.eos_id = token2idx['<eos>']
    args.vocab_size = len(token2idx)

    model = Model.load_model(args.model_path, args)
    print(model)
    model.eval()
    model.cuda()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    # decode each utterance
    with torch.no_grad(), open(args.output, 'w') as f:
        for idx, name in enumerate(js.keys(), 1):
            print('(%d/%d) decoding %s' %
                  (idx, len(js.keys()), name), flush=True)
            input = kaldi_io.read_mat(js[name]['input'][0]['feat'])  # TxD
            input = build_LFR_features(input, args.LFR_m, args.LFR_n)
            input = torch.from_numpy(input).float()
            input_length = torch.tensor([input.size(0)], dtype=torch.int)
            input = input.cuda()
            input_length = input_length.cuda()
            # hyps_ints = model.recognize(input, input_length, idx2token, args)
            hyps_ints = model.recognize_beam_cache(input, input_length, idx2token, args)
            hyp = ids2str(hyps_ints, idx2token)[0]
            f.write(name + ' ' + hyp + '\n')


def infer(args):
    return


if __name__ == "__main__":
    args = parser.parse_args()
    print(args, flush=True)
    if args.type == 'test':
        test(args)
    elif args.type == 'infer':
        infer(args)
