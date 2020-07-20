#!/usr/bin/env python
import argparse
import torch
from torch.utils.data import DataLoader

from utils.utils import load_vocab
from transformer.optimizer import TransformerOptimizer
from mask_lm.data import VQ_Dataset, f_x_pad, VQ_Pred_Dataset, f_xy_pad


parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Training "
    "(Transformer framework).")
# General config
# Task related
parser.add_argument('--train_src', type=str, default=None,
                    help='Filename of train label data (json)')
parser.add_argument('--valid_src', type=str, default=None,
                    help='Filename of validation label data (json)')
parser.add_argument('--train_tgt', type=str, default=None,
                    help='Filename of train label data (json)')
parser.add_argument('--valid_tgt', type=str, default=None,
                    help='Filename of validation label data (json)')
parser.add_argument('--vocab_src', type=str, required=True,
                    help='Dictionary which should include <unk>')
parser.add_argument('--vocab_tgt', type=str, required=True,
                    help='Dictionary which should include <unk> <sos> <eos>')

parser.add_argument('--num_workers', default=1, type=int,
                    help='Dimension of model')

parser.add_argument('--maxlen_in', default=1000, type=int,
                    help='Dimension of model')
parser.add_argument('--down_sample_rate', default=8, type=int,
                    help='Dimension of model')
# Network architecture
# conv_encoder
parser.add_argument('--n_conv_layers', default=3, type=int,
                    help='Dimension of key')
# encoder
# TODO: automatically infer input dim
parser.add_argument('--structure', type=str, default='BERT',
                    help='')
parser.add_argument('--mode', type=str, default='pre-train',
                    help='')
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

# decoder
parser.add_argument('--tgt_emb_prj_weight_sharing', default=0, type=int,
                    help='share decoder embedding with decoder projection')
# Loss
parser.add_argument('--label_smoothing', default=0.1, type=float,
                    help='label smoothing')

# Training config
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size')
parser.add_argument('--batch_frames', default=0, type=int,
                    help='Batch frames. If this is not 0, batch size will make no sense')
parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                    help='Batch size is reduced if the input sequence length > ML')
parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                    help='Batch size is reduced if the output sequence length > ML')
parser.add_argument('--num-workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--k', default=1.0, type=float,
                    help='tunable scalar multiply to learning rate')
parser.add_argument('--warmup_steps', default=4000, type=int,
                    help='warmup steps')
# save and load model
parser.add_argument('--pretrain', default=None, type=str,
                    help='Location to save epoch models')
parser.add_argument('--save-folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--_continue', default='', type=int,
                    help='Continue from checkpoint model')
parser.add_argument('--model-path', default='last.model',
                    help='Location to save best validation model')

parser.add_argument('--print-freq', default=10, type=int,
                    help='Frequency of printing training infomation')


def pre_train(args):
    # Construct Solver
    # data
    token2idx_src, idx2token_src = load_vocab(args.vocab_src)
    token2idx_tgt, idx2token_tgt = load_vocab(args.vocab_tgt)
    args.n_src = len(idx2token_src)
    args.n_tgt = len(idx2token_tgt)

    tr_dataset = VQ_Dataset(args.train_json, token2idx_src,
                            args.maxlen_in, args.maxlen_out,
                            down_sample_rate=args.down_sample_rate,
                            batch_frames=args.batch_frames)
    cv_dataset = VQ_Dataset(args.valid_json, token2idx_src,
                            args.maxlen_in, args.maxlen_out,
                            down_sample_rate=args.down_sample_rate,
                            batch_frames=args.batch_frames)
    tr_loader = DataLoader(tr_dataset, batch_size=1,
                           collate_fn=f_x_pad,
                           num_workers=args.num_workers,
                           shuffle=args.shuffle)
    cv_loader = DataLoader(cv_dataset, batch_size=1,
                           collate_fn=f_x_pad,
                           num_workers=args.num_workers)
    # load dictionary and generate char_list, sos_id, eos_id
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    if args.structure == 'BERT':
        from mask_lm.Mask_LM import Mask_LM as Model
        from mask_lm.solver import VQ_Finetune_Solver as Solver

        model = Model.create_model(args)

    print(model)
    model.cuda()

    # optimizer
    optimizier = TransformerOptimizer(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        args.k,
        args.d_model,
        args.warmup_steps)

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


def finetune(args):
    # Construct Solver
    # data
    token2idx_src, idx2token_src = load_vocab(args.vocab_src)
    token2idx_tgt, idx2token_tgt = load_vocab(args.vocab_tgt)
    args.n_src = len(idx2token_src)
    args.n_tgt = len(idx2token_tgt)

    tr_dataset = VQ_Pred_Dataset(args.train_src, args.train_tgt,
                                 token2idx_src, token2idx_tgt,
                                 args.batch_size, args.maxlen_in, args.maxlen_out,
                                 down_sample_rate=args.down_sample_rate)
    cv_dataset = VQ_Pred_Dataset(args.valid_src, args.valid_tgt,
                                 token2idx_src, token2idx_tgt,
                                 args.batch_size, args.maxlen_in, args.maxlen_out,
                                 down_sample_rate=args.down_sample_rate)
    tr_loader = DataLoader(tr_dataset, batch_size=1,
                           collate_fn=f_xy_pad,
                           num_workers=args.num_workers,
                           shuffle=args.shuffle)
    cv_loader = DataLoader(cv_dataset, batch_size=1,
                           collate_fn=f_xy_pad,
                           num_workers=args.num_workers)

    # load dictionary and generate char_list, sos_id, eos_id
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    if args.structure == 'BERT':
        from mask_lm.Mask_LM import Mask_LM as Model
        from mask_lm.solver import Mask_LM_Solver as Solver

        model = Model.create_model(args)

    print(model)
    model.cuda()

    # optimizer
    optimizier = TransformerOptimizer(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        args.k,
        args.d_model,
        args.warmup_steps)

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    if args.mode == 'pre-train':
        pre_train(args)
    elif args.mode == 'finetune':
        finetune(args)
