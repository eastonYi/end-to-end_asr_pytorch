#!/bin/bash
SRC_ROOT=$PWD/../../src
export PYTHONPATH=$SRC_ROOT:$PYTHONPATH

gpu_id=1
stage='train'

structure='cif'
label_type='phone'
model_src=$SRC_ROOT/transformer
dumpdir=/data3/easton/data/CALLHOME_Multilingual/dump   # directory to dump full features
vocab=/data3/easton/data/CALLHOME_Multilingual/dump/phone.vocab
expdir=exp/cif # tag for managing experiments.
decode_dir=${expdir}/decode_dev_beam${beam_size}
model=last.model
# Training config
epochs=100
continue=0
print_freq=100
batch_frames=20000
maxlen_in=800
maxlen_out=150

# optimizer
k=0.2
warmup_steps=10000

# Decode config
shuffle=1
beam_size=5
nbest=1
decode_max_len=100

# Feature configuration
LFR_m=1  # Low Frame Rate: number of frames to stack
LFR_n=1  # Low Frame Rate: number of frames to skip
spec_aug_cfg=2-27-2-40

# Network architecture
# Conv encoder
n_conv_layers=2

# Encoder
d_input=80
n_layers_enc=10
n_head=8
d_model=512
d_inner=2048
dropout=0.1

# assigner
context_width=5
d_assigner_hidden=512
n_assigner_layers=1

# Decoder
n_layers_dec=1

# Loss
label_smoothing=0.1
lambda_qua=0.0001

feat_train_dir=${dumpdir}/train; mkdir -p ${feat_train_dir}
feat_test_dir=${dumpdir}/test; mkdir -p ${feat_test_dir}
feat_dev_dir=${dumpdir}/dev; mkdir -p ${feat_dev_dir}

if [ $stage = 'train' ];then
    echo "stage 3: Network Training"
    mkdir -p ${expdir}
    CUDA_VISIBLE_DEVICES=${gpu_id} python $model_src/train.py \
            --train-json ${feat_train_dir}/data.json \
            --valid-json ${feat_dev_dir}/data.json \
            --vocab ${vocab} \
            --structure ${structure} \
            --label_type ${label_type} \
            --LFR_m ${LFR_m} \
            --LFR_n ${LFR_n} \
            --spec_aug_cfg ${spec_aug_cfg} \
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
            --label_smoothing ${label_smoothing} \
            --lambda_qua ${lambda_qua} \
            --epochs $epochs \
            --shuffle $shuffle \
            --batch_frames $batch_frames \
            --maxlen-in $maxlen_in \
            --maxlen-out $maxlen_out \
            --k $k \
            --warmup_steps $warmup_steps \
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
            --vocab $vocab \
            --structure ${structure} \
            --label_type ${label_type} \
            --LFR_m ${LFR_m} \
            --LFR_n ${LFR_n} \
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
