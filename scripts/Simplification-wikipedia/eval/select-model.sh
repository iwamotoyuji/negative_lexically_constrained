#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART
RESULT_DIR=../../../results/Simplification-wikipedia
UTILS_DIR=../../../utils

BPE_TOKENS=16000
BART_SCALE=base
EXP_NAME=RNN
GPU_ID=0
TRAIN_SET=wiki-large
TEST_SET=asset
while getopts ab:d:g:hln:p:r:tu: OPT
do
    case $OPT in
    a ) TRAIN_SET=wiki-auto
        ;;
    b ) BPE_TOKENS=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-a TRAIN_SET=wiki-auto] [-b BPE_TOKENS] [-d DATASETS_DIR] [-g GPU_ID] [-l BART_SCALE=large] [-n EXP_NAME] [-p PRE_TRAINED_DIR] [-r RESULT_DIR] [-t TEST_SET=turkcorpus] [-u UTILS_DIR]" 1>&2
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
    t ) TEST_SET=turkcorpus
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


output_dir=${RESULT_DIR}/${TRAIN_SET}/${EXP_NAME}/select-model
mkdir -p $output_dir

input_dir=${DATASETS_DIR}/${TEST_SET}/bpe${BPE_TOKENS}-from-${TRAIN_SET}
reference_dir=${DATASETS_DIR}/${TEST_SET}/tok
if [ $TEST_SET == "asset" ]; then
    references=${reference_dir}/valid.orig.simple.0,${reference_dir}/valid.orig.simple.1,${reference_dir}/valid.orig.simple.2,${reference_dir}/valid.orig.simple.3,${reference_dir}/valid.orig.simple.4,${reference_dir}/valid.orig.simple.5,${reference_dir}/valid.orig.simple.6,${reference_dir}/valid.orig.simple.7,${reference_dir}/valid.orig.simple.8,${reference_dir}/valid.orig.simple.9
else
    references=${reference_dir}/valid.orig.simple.0,${reference_dir}/valid.orig.simple.1,${reference_dir}/valid.orig.simple.2,${reference_dir}/valid.orig.simple.3,${reference_dir}/valid.orig.simple.4,${reference_dir}/valid.orig.simple.5,${reference_dir}/valid.orig.simple.6,${reference_dir}/valid.orig.simple.7
fi

if [ $BPE_TOKENS == "BART" ]; then
    preprocessed_dir=${DATASETS_DIR}/${TRAIN_SET}/bpe${BPE_TOKENS}/fairseq-preprocess-BART-${BART_SCALE}
else
    preprocessed_dir=${DATASETS_DIR}/${TRAIN_SET}/bpe${BPE_TOKENS}/fairseq-preprocess
fi

input=${input_dir}/valid.complex
result=${output_dir}/${TEST_SET}
log=${output_dir}/${TEST_SET}.log

for model_name in ${RESULT_DIR}/${TRAIN_SET}/${EXP_NAME}/checkpoints/*.pt; do
    echo model name : `basename $model_name` >> $log
    CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-interactive $preprocessed_dir \
        --input $input \
        --path $model_name \
        --buffer-size 1024 --batch-size 64 \
        --nbest 1 --max-len-b 50 \
        --beam 5 --lenpen 1.0 --remove-bpe \
        > ${result}.txt
    grep ^H ${result}.txt | cut -f3- > ${result}.sys

    if [ $BPE_TOKENS == "BART" ]; then
        cp ${result}.sys ${result}.tmp.sys
        python $BART_BPE_DECODER \
            --encoder-json ${PRE_TRAINED_DIR}/encoder.json --vocab-bpe ${PRE_TRAINED_DIR}/vocab.bpe \
            --inputs ${result}.tmp.sys --outputs ${result}.sys \
            --workers 60 --keep-empty
    fi

    cat ${result}.sys | perl -C $DETOKENIZER -l en > ${result}.detok.sys

    easse evaluate -t custom -m 'bleu,sari' \
        --refs_sents_paths $references \
        --orig_sents_path ${reference_dir}/valid.orig.complex \
        --sys_sents_path ${result}.detok.sys \
        2>&1 | tee -a $log
done
