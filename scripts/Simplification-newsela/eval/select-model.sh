#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART
RESULT_DIR=../../../results/Simplification-newsela
UTILS_DIR=../../../utils

BPE_TOKENS=16000
BART_SCALE=base
EXP_NAME=RNN
GPU_ID=0
while getopts b:d:g:hln:p:r:tu: OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-d DATASETS_DIR] [-g GPU_ID] [-l BART_SCALE=large] [-n EXP_NAME] [-p PRE_TRAINED_DIR] [-r RESULT_DIR] [-u UTILS_DIR]" 1>&2
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
    u ) UTILS_DIR=$OPTARG
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


input_dir=${DATASETS_DIR}/newsela-auto/bpe${BPE_TOKENS}
reference_dir=${DATASETS_DIR}/newsela-auto/tok
output_dir=${RESULT_DIR}/newsela-auto/${EXP_NAME}/valid
mkdir -p $output_dir

input=${input_dir}/valid.complex
reference=${reference_dir}/valid.simple
log=${RESULT_DIR}/newsela-auto/${EXP_NAME}/select-model.log

if [ $BPE_TOKENS == "BART" ]; then
    preprocessed_dir=${DATASETS_DIR}/newsela-auto/bpe${BPE_TOKENS}/fairseq-preprocess-BART-${BART_SCALE}
else
    preprocessed_dir=${DATASETS_DIR}/newsela-auto/bpe${BPE_TOKENS}/fairseq-preprocess
fi

for model_name in ${RESULT_DIR}/newsela-auto/${EXP_NAME}/checkpoints/*.pt; do
    echo model name : `basename $model_name` >> $log
    CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-interactive $preprocessed_dir \
        --input $input \
        --path $model_name \
        --buffer-size 1024 --batch-size 128 \
        --nbest 1 --max-len-b 50 \
        --beam 5 --lenpen 1.0 --remove-bpe \
        > ${output_dir}/result.txt
    grep ^S ${output_dir}/result.txt | cut -f2- > ${output_dir}/result.src
    grep ^H ${output_dir}/result.txt | cut -f3- > ${output_dir}/result.sys

    if [ $BPE_TOKENS == "BART" ]; then
        cp ${output_dir}/result.sys ${output_dir}/result.tmp.sys
        python $BART_BPE_DECODER \
            --encoder-json ${PRE_TRAINED_DIR}/encoder.json --vocab-bpe ${PRE_TRAINED_DIR}/vocab.bpe \
            --inputs ${output_dir}/result.tmp.sys --outputs ${output_dir}/result.sys \
            --workers 60 --keep-empty
    fi

    cat ${output_dir}/result.sys | perl -C $DETOKENIZER -l en > ${output_dir}/result.detok.sys
    cat ${output_dir}/result.src | perl -C $DETOKENIZER -l en > ${output_dir}/result.detok.src
    easse evaluate -t custom -m 'bleu,sari' \
        --refs_sents_paths $reference \
        --orig_sents_path ${output_dir}/result.detok.src \
        --sys_sents_path ${output_dir}/result.detok.sys 2>&1 | tee -a $log
done
