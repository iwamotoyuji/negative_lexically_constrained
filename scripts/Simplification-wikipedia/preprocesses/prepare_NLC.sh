#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
NLC_MAKER_DIR=../../NLC_maker
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART

BPE_TOKENS=16000
CONSTRAINT=0.5
MODE=test
TRAIN_SET=wiki-large
while getopts ab:c:d:hn:p:v OPT
do
    case $OPT in
    a ) TRAIN_SET=wiki-auto
        ;;
    b ) BPE_TOKENS=$OPTARG
        ;;
    c ) CONSTRAINT=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-a TRAIN_SET=wiki-auto] [-b BPE_TOKENS] [-c CONSTRAINT] [-d DATASETS_DIR] [-n NLC_MAKER_DIR] [-p PRE_TRAINED_DIR] [-v MODE=valid]" 1>&2
        exit 0
        ;;
    n ) NLC_MAKER_DIR=$OPTARG
        ;;
    p ) PRE_TRAINED_DIR=$OPTARG
        ;;
    v ) MODE=valid
        ;;
    esac
done


# --- Make PMI dict ---
python ${NLC_MAKER_DIR}/make_PMI_dict.py \
    ${DATASETS_DIR}/${TRAIN_SET}/tok/train.complex \
    ${DATASETS_DIR}/${TRAIN_SET}/tok/train.simple \
    ${DATASETS_DIR}/${TRAIN_SET}/PMI_dict.pickle \
    --ignore_word_cnt 3

# --- Make constraint ---
if [ $BPE_TOKENS == "BART" ]; then
    for dataset in asset turkcorpus; do
        python ${NLC_MAKER_DIR}/make_constraint_file_for_BART.py \
            ${DATASETS_DIR}/${dataset}/tok/${MODE}.complex \
            ${DATASETS_DIR}/${dataset}/bpe${BPE_TOKENS}-from-${TRAIN_SET}/NLC_${CONSTRAINT}-for-${MODE}.complex \
            ${DATASETS_DIR}/${TRAIN_SET}/PMI_dict.pickle \
            --encoder-json ${PRE_TRAINED_DIR}/encoder.json --vocab-bpe ${PRE_TRAINED_DIR}/vocab.bpe \
            --theta $CONSTRAINT
    done
else
    for dataset in asset turkcorpus; do
        python ${NLC_MAKER_DIR}/make_constraint_file.py \
            ${DATASETS_DIR}/${dataset}/bpe${BPE_TOKENS}-from-${TRAIN_SET}/${MODE}.complex \
            ${DATASETS_DIR}/${dataset}/bpe${BPE_TOKENS}-from-${TRAIN_SET}/NLC_${CONSTRAINT}-for-${MODE}.complex \
            ${DATASETS_DIR}/${TRAIN_SET}/PMI_dict.pickle \
            --theta $CONSTRAINT
    done
fi
