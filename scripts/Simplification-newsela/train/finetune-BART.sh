#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART
RESULT_DIR=../../../results/Simplification-newsela

BPE_TOKENS=BART
BART_SCALE=base
EXP_NAME=BART
GPU_ID=0
SEED=1
while getopts b:d:g:hln:p:r:s: OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-d DATASETS_DIR] [-g GPU_ID] [-l BART_SCALE=large] [-n EXP_NAME] [-p PRE_TRAINED_DIR] [-r RESULT_DIR] [-s SEED]" 1>&2
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

preprocessed_dir=${DATASETS_DIR}/newsela-auto/bpe${BPE_TOKENS}/fairseq-preprocess-BART-${BART_SCALE}
save_dir=${RESULT_DIR}/newsela-auto/${EXP_NAME}
rm -fr $save_dir
mkdir -p $save_dir

echo Training server name : `hostname` > ${save_dir}/train.log
CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-train $preprocessed_dir \
    --arch bart_${BART_SCALE} \
    --restore-file ${PRE_TRAINED_DIR}/bart.${BART_SCALE}/model.pt \
    --task translation \
    --source-lang complex --target-lang simple \
    --truncate-source \
    --share-all-embeddings --share-decoder-input-output-embed --layernorm-embedding \
    --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-08 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --lr 3e-5 --lr-scheduler polynomial_decay --warmup-updates 500 \
    --dropout 0.2 --weight-decay 0.0001 --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --fp16 --seed $SEED \
    --max-tokens 4096 --update-freq 1 --max-update 30000 --total-num-update 30000 \
    --save-interval-updates 1000 --validate-interval 9999 --validate-interval-updates 1000 \
    --save-dir ${save_dir}/checkpoints --no-epoch-checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --log-format simple \
    2>&1 | tee -a ${save_dir}/train.log
#    --patience 5
#    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#    --eval-bleu \
#    --eval-bleu-args '{"beam": 5, "lenpen": 1.0}' \
#    --eval-bleu-detok moses \
#    --eval-bleu-remove-bpe \
#--required-batch-size-multiple 1 --attention-dropout 0.1 \