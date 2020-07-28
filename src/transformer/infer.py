#!/usr/bin/env python
import argparse
import json
import torch
import kaldi_io
import time
from torch.utils.data import DataLoader

from utils.utils import load_vocab, ids2str
from utils.data import build_LFR_features
from transformer.data import AudioDataset, batch_generator


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
parser.add_argument('--label_type', type=str, default='token',
                    help='label_type')
parser.add_argument('--num-workers', default=4, type=int,
                    help='Number of workers to generate minibatch')

# Low Frame Rate (stacking and skipping frames)
parser.add_argument('--LFR_m', default=1, type=int,
                    help='Low Frame Rate: number of frames to stack')
parser.add_argument('--LFR_n', default=1, type=int,
                    help='Low Frame Rate: number of frames to skip')
parser.add_argument('--spec_aug_cfg', default=None, type=str,
                    help='spec_aug_cfg')

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
parser.add_argument('--d_model', default=512, type=int,
                    help='Dimension of model')
parser.add_argument('--d_inner', default=2048, type=int,
                    help='Dimension of inner')
parser.add_argument('--dropout', default=0.1, type=float,
                    help='Dropout rate')
# assigner
parser.add_argument('--w_context', default=3, type=int,
                    help='Positional Encoding max len')
parser.add_argument('--d_assigner_hidden', default=512, type=int,
                    help='Positional Encoding max len')
parser.add_argument('--n_assigner_layers', default=3, type=int,
                    help='Positional Encoding max len')
# decoder
parser.add_argument('--n_layers_dec', default=6, type=int,
                    help='Number of decoder stacks')

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

    cur_time = time.time()
    # decode each utterance

    test_dataset = AudioDataset('/home/easton/projects/OpenASR/egs/aishell1/data/test.json',
                                token2idx, frames_size=1000,
                                len_in_max=1999, len_out_max=99)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             collate_fn=batch_generator(),
                             num_workers=args.num_workers)
    # test_loader = AudioDataLoader(test_dataset, batch_size=1,
    #                             token2idx=token2idx,
    #                             label_type=args.label_type,
    #                             num_workers=args.num_workers,
    #                             LFR_m=args.LFR_m, LFR_n=args.LFR_n)

    def process_batch(hyps, scores, idx2token, fw):
        for nbest, nscore in zip(hyps, scores):
            for n, (hyp, score) in enumerate(zip(nbest, nscore)):
                hyp = hyp.tolist()
                try:
                    eos = hyp.index(3)
                except:
                    eos = None

                hyp = ''.join(idx2token[i] for i in hyp[:eos])
                print("top{}: {} score: {:.3f}\n".format(n+1, hyp, score))
                if n == 0:
                    fw.write("{} {}\n".format('uttid', hyp))

    with torch.no_grad(), open(args.output, 'w') as fw:
        for data in test_loader:
            uttids, xs_pad, len_xs, ys_pad, len_ys = data
            xs_pad = xs_pad.cuda()
            ys_pad = ys_pad.cuda()
            hyps_ints, len_decoded_sorted, scores = model.batch_recognize(
                xs_pad, len_xs, args.beam_size)

            process_batch(hyps_ints.cpu().numpy(), scores.cpu().numpy(), idx2token, fw)
            # f.write(uttids[0] + ' ' + hyp + '\n')

    # with torch.no_grad(), open(args.output, 'w') as f:
    #     for idx, uttid in enumerate(js.keys(), 1):
    #         input = kaldi_io.read_mat(js[uttid]['input'][0]['feat'])  # TxD
    #         input = build_LFR_features(input, args.LFR_m, args.LFR_n)
    #         input = torch.from_numpy(input).float()
    #         input_length = torch.tensor([input.size(0)], dtype=torch.int)
    #         input = input.cuda()
    #         input_length = input_length.cuda()
    #         # hyps_ints = model.recognize(input, input_length, idx2token, args,
    #         #                             target_num=len(js[uttid]['output']['tokenid'].split()))
    #         hyps_ints = model.recognize(input, input_length, idx2token, args)
    #         # hyps_ints = model.recognize_beam_cache(input, input_length, idx2token, args)
    #         hyp = ids2str(hyps_ints, idx2token)[0]
    #         print(hyp)
    #         f.write(uttid + ' ' + hyp + '\n')
    #         used_time = time.time() - cur_time
    #         print('({}/{}) use time {:.2f}s {}: {}'.format(
    #             idx, len(js.keys()), used_time, uttid, hyp), flush=True)
    #         cur_time = time.time()


def infer(args):
    return


if __name__ == "__main__":
    args = parser.parse_args()
    print(args, flush=True)
    if args.type == 'test':
        test(args)
    elif args.type == 'infer':
        infer(args)
