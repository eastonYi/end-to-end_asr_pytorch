#!/bin/bash
SRC_ROOT=$PWD/../../src
export PYTHONPATH=$SRC_ROOT:$PYTHONPATH

gpu_id=0
stage='train'

structure='BERT'
mode='finetune'
model_src=$SRC_ROOT/mask_lm
dumpdir=/data3/easton/data/Librispeech/wav2vec
vocab_src=/data3/easton/data/Librispeech/wav2vec/vocab_500.txt
vocab_tgt=/data3/easton/data/Librispeech/org_data/subword_3727.vocab
pretrain=/home/easton/projects/Speech-Transformer-pytorch/egs/libri_wav2vec/exp/bert/last.model
expdir=exp/finetune # tag for managing experiments.
decode_dir=${expdir}/decode_dev_beam${beam_size}
model=last.model
# Training config
epochs=100
continue=0
print_freq=100
batch_size=4
maxlen_in=1000
maxlen_out=150

# optimizer
k=0.2
warmup_steps=4000

# Decode config
shuffle=1

# Network architecture
# Conv encoder
n_conv_layers=1

# Encoder
n_layers_enc=10
n_head=8
d_model=512
d_inner=2048
dropout=0.1

# Loss
label_smoothing=0.1

train_src=${dumpdir}/train-960/train.src.with_uttid
train_tgt=${dumpdir}/train-960/train-100.subword

dev_src=${dumpdir}/valid/valid.src.with_uttid
dev_tgt=${dumpdir}/valid/valid.subword

if [ $stage = 'train' ];then
    echo "stage 3: Network Training"
    mkdir -p ${expdir}
    CUDA_VISIBLE_DEVICES=${gpu_id} python $model_src/train.py \
            --train_src $train_src \
            --valid_src $dev_src \
            --train_tgt $train_tgt \
            --valid_tgt $dev_tgt \
            --vocab_src ${vocab_src} \
            --vocab_tgt ${vocab_tgt} \
            --structure ${structure} \
            --mode ${mode} \
            --n_conv_layers $n_conv_layers \
            --n_layers_enc $n_layers_enc \
            --n_head $n_head \
            --d_model $d_model \
            --d_inner $d_inner \
            --dropout $dropout \
            --label_smoothing ${label_smoothing} \
            --epochs $epochs \
            --shuffle $shuffle \
            --batch_size $batch_size \
            --maxlen-in $maxlen_in \
            --maxlen-out $maxlen_out \
            --k $k \
            --warmup_steps $warmup_steps \
            --pretrain $pretrain \
            --save-folder ${expdir} \
            --_continue ${continue} \
            --print-freq ${print_freq}
fi


if [ $stage = 'test' ];then
    echo "stage 4: Decoding"
    mkdir -p ${decode_dir}
    export PYTHONWARNINGS="ignore"
    CUDA_VISIBLE_DEVICES=${gpu_id} python $model_src/infer.py \
            --type ${stage} \
            --recog-json ${feat_test_dir}/data.json \
            --vocab_src ${vocab_src} \
            --vocab_tgt ${vocab_tgt} \
            --structure ${structure} \
            --label_type ${label_type} \
            --d_input $d_input \
            --n_conv_layers $n_conv_layers \
            --n_layers_enc $n_layers_enc \
            --d_assigner_hidden $d_assigner_hidden \
            --n_assigner_layers $n_assigner_layers \
            --n_head $n_head \
            --d_model $d_model \
            --d_inner $d_inner \
            --dropout $dropout \
            --n_layers_dec $n_layers_dec \
            --model-path ${expdir}/${model} \
            --output ${decode_dir}/hyp \
            --beam-size $beam_size \
            --nbest $nbest \
            --decode-max-len $decode_max_len
fi
