#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART
RESULT_DIR=../../../results/Simplification-newsela
UTILS_DIR=../../../utils

BPE_TOKENS=16000
BART_SCALE=base
CONSTRAINT=0
EXP_NAME=RNN
GPU_ID=0
MODEL=checkpoint_best.pt
MODE=test
while getopts b:c:d:g:hlm:n:p:r:u:v OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    c ) CONSTRAINT=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-c CONSTRAINT] [-d DATASETS_DIR] [-g GPU_ID] [-l BART_SCALE=large] [-m MODEL] [-n EXP_NAME] [-p PRE_TRAINED_DIR] [-r RESULT_DIR] [-u UTILS_DIR] [-v MODE=valid]" 1>&2
        exit 1
        ;;
    l ) BART_SCALE=large
        ;;
    m ) MODEL=$OPTARG
        ;;
    n ) EXP_NAME=$OPTARG
        ;;
    p ) PRE_TRAINED_DIR=$OPTARG
        ;;
    r ) RESULT_DIR=$OPTARG
        ;;    
    u ) UTILS_DIR=$OPTARG
        ;;
    v ) MODE=valid
        ;;
    esac
done


DETOKENIZER=${UTILS_DIR}/mosesdecoder/scripts/tokenizer/detokenizer.perl
BART_BPE_DECODER=${PRE_TRAINED_DIR}/multiprocessing_bpe_decoder.py

# --- Download utils ---
if type easse > /dev/null 2>&1; then
    echo "[Info] easse already exists, skipping download"
else
    echo "[Info] easse not exists, start installation to ${UTILS_DIR}/easse"
    pushd $UTILS_DIR
    git clone https://github.com/feralvam/easse.git
    cd easse
    pip install -e .
    popd
fi


if [ $EXP_NAME == "RNN_ensemble" ]; then
    output_dir=${RESULT_DIR}/newsela-auto/RNN_ensemble/${MODE}
    model=${RESULT_DIR}/newsela-auto/RNN_SEED11/checkpoints/checkpoint_7_28000.pt:${RESULT_DIR}/newsela-auto/RNN_SEED22/checkpoints/checkpoint_4_17000.pt:${RESULT_DIR}/newsela-auto/RNN_SEED33/checkpoints/checkpoint_5_19000.pt:${RESULT_DIR}/newsela-auto/RNN_SEED44/checkpoints/checkpoint_7_30000.pt
elif [ $EXP_NAME == "SAN_ensemble" ]; then
    output_dir=${RESULT_DIR}/newsela-auto/SAN_ensemble/${MODE}
    model=${RESULT_DIR}/newsela-auto/SAN_SEED11/checkpoints/checkpoint_7_29000.pt:${RESULT_DIR}/newsela-auto/SAN_SEED22/checkpoints/checkpoint_7_28000.pt:${RESULT_DIR}/newsela-auto/SAN_SEED33/checkpoints/checkpoint_7_28000.pt:${RESULT_DIR}/newsela-auto/SAN_SEED44/checkpoints/checkpoint_7_30000.pt
elif [ $EXP_NAME == "BART_base_ensemble" ]; then
    output_dir=${RESULT_DIR}/newsela-auto/BART_base_ensemble/${MODE}
    model=${RESULT_DIR}/newsela-auto/BART_base_SEED11/checkpoints/checkpoint_2_9000.pt:${RESULT_DIR}/newsela-auto/BART_base_SEED22/checkpoints/checkpoint_2_9000.pt:${RESULT_DIR}/newsela-auto/BART_base_SEED33/checkpoints/checkpoint_4_18000.pt:${RESULT_DIR}/newsela-auto/BART_base_SEED44/checkpoints/checkpoint_5_23000.pt
elif [ $EXP_NAME == "BART_large_ensemble" ]; then
    output_dir=${RESULT_DIR}/newsela-auto/BART_large_ensemble/${MODE}
    model=${RESULT_DIR}/newsela-auto/BART_large_SEED11/checkpoints/checkpoint_2_9000.pt:${RESULT_DIR}/newsela-auto/BART_large_SEED22/checkpoints/checkpoint_2_9000.pt:${RESULT_DIR}/newsela-auto/BART_large_SEED33/checkpoints/checkpoint_4_15000.pt:${RESULT_DIR}/newsela-auto/BART_large_SEED44/checkpoints/checkpoint_5_21000.pt
else
    output_dir=${RESULT_DIR}/newsela-auto/${EXP_NAME}/${MODE}
    model=${RESULT_DIR}/newsela-auto/${EXP_NAME}/checkpoints/${MODEL}
fi
mkdir -p $output_dir

input_dir=${DATASETS_DIR}/newsela-auto/bpe${BPE_TOKENS}
reference_dir=${DATASETS_DIR}/newsela-auto/tok
reference=${reference_dir}/${MODE}.simple

if [ $BPE_TOKENS == "BART" ]; then
    preprocessed_dir=${DATASETS_DIR}/newsela-auto/bpe${BPE_TOKENS}/fairseq-preprocess-BART-${BART_SCALE}
else
    preprocessed_dir=${DATASETS_DIR}/newsela-auto/bpe${BPE_TOKENS}/fairseq-preprocess
fi
if [ $CONSTRAINT == 0 ]; then
    input=${input_dir}/${MODE}.complex
    result=${output_dir}/result
    log=${output_dir}/eval.log
    arg=""
else
    input=${input_dir}/${MODE}-add-NLC_${CONSTRAINT}.complex
    result=${output_dir}/result-NLC_${CONSTRAINT}
    log=${output_dir}/eval-NLC_${CONSTRAINT}.log
    arg="--constraints"
    paste -d "" ${input_dir}/${MODE}.complex ${input_dir}/NLC_${CONSTRAINT}-for-${MODE}.complex > $input
fi
CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-interactive $preprocessed_dir \
    --input $input \
    --path $model \
    --buffer-size 1024 --batch-size 128 \
    --nbest 1 --max-len-b 50 \
    --beam 5 --lenpen 1.0 --remove-bpe $arg \
    2>&1 | tee ${result}.txt
grep ^S ${result}.txt | cut -f2- > ${result}.src
grep ^H ${result}.txt | cut -f3- > ${result}.sys

if [ $BPE_TOKENS == "BART" ]; then
    cp ${result}.sys ${result}.tmp.sys
    python $BART_BPE_DECODER \
        --encoder-json ${PRE_TRAINED_DIR}/encoder.json --vocab-bpe ${PRE_TRAINED_DIR}/vocab.bpe \
        --inputs ${result}.tmp.sys --outputs ${result}.sys \
        --workers 60 --keep-empty
fi

cat ${result}.sys | perl -C $DETOKENIZER -l en > ${result}.detok.sys
cat ${result}.src | perl -C $DETOKENIZER -l en > ${result}.detok.src
echo model name : $MODEL >> $log
easse evaluate -t custom -m 'bleu,sari' \
    --refs_sents_paths $reference \
    --orig_sents_path ${result}.detok.src \
    --sys_sents_path ${result}.detok.sys 2>&1 | tee -a $log
