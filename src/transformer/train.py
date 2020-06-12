#!/usr/bin/env python
import argparse
import torch

from utils.data import AudioDataLoader, AudioDataset
from utils.utils import load_vocab
from transformer.optimizer import TransformerOptimizer


parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Training "
    "(Transformer framework).")
# General config
# Task related
parser.add_argument('--train-json', type=str, default=None,
                    help='Filename of train label data (json)')
parser.add_argument('--valid-json', type=str, default=None,
                    help='Filename of validation label data (json)')
parser.add_argument('--vocab', type=str, required=True,
                    help='Dictionary which should include <unk> <sos> <eos>')
# Low Frame Rate (stacking and skipping frames)
parser.add_argument('--LFR_m', default=4, type=int,
                    help='Low Frame Rate: number of frames to stack')
parser.add_argument('--LFR_n', default=3, type=int,
                    help='Low Frame Rate: number of frames to skip')
# Network architecture

# conv_encoder
parser.add_argument('--n_conv_layers', default=3, type=int,
                    help='Dimension of key')
# encoder
# TODO: automatically infer input dim
parser.add_argument('--structure', type=str, default='transformer',
                    help='transformer transformer-ctc conv-transformer-ctc')
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
parser.add_argument('--n_assigner_layers', default=3, type=int,
                    help='Positional Encoding max len')
# decoder
parser.add_argument('--d_word_vec', default=512, type=int,
                    help='Dim of decoder embedding')
parser.add_argument('--n_layers_dec', default=6, type=int,
                    help='Number of decoder stacks')
parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int,
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
parser.add_argument('--batch-size', default=32, type=int,
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
parser.add_argument('--save-folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--_continue', default='', type=int,
                    help='Continue from checkpoint model')
parser.add_argument('--model-path', default='last.model',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print-freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_lr', dest='visdom_lr', type=int, default=0,
                    help='Turn on visdom graphing learning rate')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom-id', default='Transformer training',
                    help='Identifier for visdom run')


def main(args):
    # Construct Solver
    # data
    # char_list, sos_id, eos_id = process_dict(args.dict)
    token2idx, idx2token = load_vocab(args.vocab)
    vocab_size = len(token2idx)
    sos_id = token2idx['<sos>']
    eos_id = token2idx['<eos>']

    tr_dataset = AudioDataset(args.train_json, args.batch_size,
                              args.maxlen_in, args.maxlen_out,
                              batch_frames=args.batch_frames)
    cv_dataset = AudioDataset(args.valid_json, args.batch_size,
                              args.maxlen_in, args.maxlen_out,
                              batch_frames=args.batch_frames)
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                token2idx=token2idx,
                                num_workers=args.num_workers,
                                shuffle=args.shuffle,
                                LFR_m=args.LFR_m, LFR_n=args.LFR_n)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                token2idx=token2idx,
                                num_workers=args.num_workers,
                                LFR_m=args.LFR_m, LFR_n=args.LFR_n)
    # load dictionary and generate char_list, sos_id, eos_id
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    if args.structure == 'transformer':
        from transformer.decoder import Decoder
        from transformer.encoder import Encoder
        from transformer.Transformer import Transformer
        from transformer.solver import Transformer_Solver as Solver

        encoder = Encoder(args.d_input * args.LFR_m, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)
        model = Transformer(encoder, decoder)

    elif args.structure == 'transformer-ctc':
        from transformer.decoder import Decoder
        from transformer.encoder import Encoder
        from transformer.Transformer import CTC_Transformer as Transformer
        from transformer.solver import Transformer_CTC_Solver as Solver

        encoder = Encoder(args.d_input * args.LFR_m, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)
        model = Transformer(encoder, decoder)

    elif args.structure == 'conv-transformer-ctc':
        from transformer.decoder import Decoder
        from transformer.encoder import Encoder
        from transformer.Transformer import Conv_CTC_Transformer as Transformer
        from transformer.solver import Transformer_CTC_Solver as Solver
        from transformer.conv_encoder import Conv2dSubsample

        conv_encoder = Conv2dSubsample(args.d_input * args.LFR_m, args.d_model,
                                       n_layers=args.n_conv_layers)
        encoder = Encoder(args.d_model, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)
        model = Transformer(conv_encoder, encoder, decoder)

    elif args.structure == 'cif':
        from transformer.conv_encoder import Conv2dSubsample
        from transformer.encoder import Encoder
        from transformer.attentionAssigner import Attention_Assigner
        from transformer.decoder import Decoder_CIF as Decoder
        from transformer.CIF_Model import CIF_Model
        from transformer.solver import CIF_Solver as Solver

        conv_encoder = Conv2dSubsample(args.d_input * args.LFR_m,
                                       args.d_model,
                                       n_layers=args.n_conv_layers)
        encoder = Encoder(args.d_model, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        assigner = Attention_Assigner(d_input=args.d_model, d_hidden=args.d_model,
                                      w_context=args.w_context,
                                      n_layers=args.n_assigner_layers)
        decoder = Decoder(sos_id, vocab_size, args.d_word_vec, args.n_layers_dec,
                          args.n_head, args.d_k, args.d_v, args.d_model,
                          args.d_inner, dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)
        model = CIF_Model(conv_encoder, encoder, assigner, decoder)

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
    main(args)
