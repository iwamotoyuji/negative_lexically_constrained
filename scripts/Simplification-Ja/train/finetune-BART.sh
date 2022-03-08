#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-JaBART
RESULT_DIR=../../../results/Simplification-Ja

BART_SCALE=base
EXP_NAME=BART_base
GPU_ID=0
SEED=1
while getopts d:g:hln:p:r:s: OPT
do
    case $OPT in
    d ) DATASETS_DIR=$OPTARG
        ;;
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-d DATASETS_DIR] [-g GPU_ID] [-l BART_SCALE=large] [-n EXP_NAME] [-p PRE_TRAINED_DIR] [-r RESULT_DIR] [-s SEED]" 1>&2
        exit 1
        ;;
    l ) BART_SCALE=large
        ;;
    n ) EXP_NAME=$OPTARG
        ;;
    p ) PRE_TRAINED_DIR=$OPTARG
        ;;
    r ) RESULT_DIR=$OPTARG
        ;;
    s ) SEED=$OPTARG
        ;;
    esac
done

preprocessed_dir=${DATASETS_DIR}/SNOW/bpeBART-${BART_SCALE}/fairseq-preprocess
save_dir=${RESULT_DIR}/${EXP_NAME}
rm -fr $save_dir
mkdir -p $save_dir

echo Training server name : `hostname` > ${save_dir}/train.log
CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-train $preprocessed_dir \
    --prepend-bos \
    --arch bart_${BART_SCALE} \
    --restore-file ${PRE_TRAINED_DIR}/japanese_bart_${BART_SCALE}_2.0/bart_model.pt \
    --task translation_from_pretrained_bart \
    --source-lang complex --target-lang simple \
    --optimizer adam --adam-betas '{0.9, 0.98}' --adam-eps 1e-06 \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
    --lr 3e-5 --min-lr -1 --lr-scheduler polynomial_decay --warmup-updates 500 \
    --dropout 0.2 --weight-decay 0.0001 --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --fp16 --seed $SEED \
    --max-tokens 4096 --update-freq 1 --max-update 30000 --total-num-update 30000 \
    --save-interval-updates 1000 --validate-interval 9999 --validate-interval-updates 1000 \
    --save-dir ${save_dir}/checkpoints --no-epoch-checkpoints \
    --ddp-backend no_c10d \
    --encoder-normalize-before --decoder-normalize-before \
    --log-format simple \
    2>&1 | tee -a ${save_dir}/train.log
