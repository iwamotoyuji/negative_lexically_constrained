#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART
RESULT_DIR=../../../results/FT

BPE_TOKENS=BART
BART_SCALE=base
DOMAIN=Combo
EXP_NAME=BART
GPU_ID=0
SEED=1
while getopts b:d:efg:hln:p:r:s: OPT
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
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-d DATASETS_DIR] [-e DOMAIN=Entertainment_Music] [-f DOMAIN=Family_Relationships] [-g GPU_ID] [-l BART_SCALE=large] [-n EXP_NAME] [-p PRE_TRAINED_DIR] [-r RESULT_DIR]" 1>&2
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

preprocessed_dir=${DATASETS_DIR}/GYAFC/${DOMAIN}/bpe${BPE_TOKENS}/bidirectional/fairseq-preprocess-BART-${BART_SCALE}
save_dir=${RESULT_DIR}/${DOMAIN}/${EXP_NAME}
rm -fr $save_dir
mkdir -p $save_dir

TOTAL_NUM_UPDATES=40000
WARMUP_UPDATES=500
LR=3e-5
MAX_TOKENS=4096
UPDATE_FREQ=1

echo Training server name : `hostname` > ${save_dir}/train.log
CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-train $preprocessed_dir \
    --arch bart_${BART_SCALE} \
    --restore-file ${PRE_TRAINED_DIR}/bart.${BART_SCALE}/model.pt \
    --task translation \
    --source-lang src --target-lang tgt \
    --truncate-source \
    --share-all-embeddings --share-decoder-input-output-embed --layernorm-embedding \
    --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-08 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --lr $LR --lr-scheduler polynomial_decay --warmup-updates $WARMUP_UPDATES \
    --dropout 0.2 --weight-decay 0.0001 --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --fp16 --seed $SEED \
    --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ \
    --max-epoch 20 --total-num-update $TOTAL_NUM_UPDATES --patience 5 \
    --save-interval-updates 1000 --validate-interval 9999 --validate-interval-updates 1000 \
    --save-dir ${save_dir}/checkpoints --keep-interval-updates 20 --no-epoch-checkpoints \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --log-format simple \
    2>&1 | tee -a ${save_dir}/train.log
#--required-batch-size-multiple 1 --attention-dropout 0.1 \