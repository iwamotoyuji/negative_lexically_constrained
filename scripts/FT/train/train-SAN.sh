#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
RESULT_DIR=../../../results/FT

BPE_TOKENS=16000
DOMAIN=Combo
EXP_NAME=RNN
GPU_ID=0
SEED=1
while getopts b:d:efg:hn:r:s: OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    e ) DOMAIN=Entertainment_Music
        ;;
    f ) DOMAIN=Family_Relationships
        ;;
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-d DATASETS_DIR] [-e DOMAIN=Entertainment_Music] [-f DOMAIN=Family_Relationships] [-g GPU_ID] [-n EXP_NAME] [-r RESULT_DIR] [-s SEED]" 1>&2
        exit 1
        ;;
    n ) EXP_NAME=$OPTARG
        ;;
    r ) RESULT_DIR=$OPTARG
        ;;
    s ) SEED=$OPTARG
        ;;
    esac
done

preprocessed_dir=${DATASETS_DIR}/GYAFC/${DOMAIN}/bpe${BPE_TOKENS}/bidirectional/fairseq-preprocess
save_dir=${RESULT_DIR}/${DOMAIN}/${EXP_NAME}
rm -fr $save_dir
mkdir -p $save_dir

echo Training server name : `hostname` > ${save_dir}/train.log
CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-train $preprocessed_dir \
    --arch transformer \
    --share-all-embeddings --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --reset-optimizer --reset-dataloader --reset-meters \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --dropout 0.2 --weight-decay 0.0001 --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --fp16 --seed $SEED \
    --max-tokens 4096 --patience 5 \
    --save-interval-updates 1000 --validate-interval 9999 --validate-interval-updates 1000 \
    --save-dir ${save_dir}/checkpoints --keep-interval-updates 20 --no-epoch-checkpoints \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --log-format simple \
    2>&1 | tee -a ${save_dir}/train.log
    