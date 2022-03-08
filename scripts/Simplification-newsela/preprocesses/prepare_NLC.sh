#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
NLC_MAKER_DIR=../../NLC_maker
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART

BPE_TOKENS=16000
CONSTRAINT=0.5
MODE=test
while getopts b:c:d:hn:p:v OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    c ) CONSTRAINT=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-c CONSTRAINT] [-d DATASETS_DIR] [-n NLC_MAKER_DIR] [-p PRE_TRAINED_DIR] [-v MODE=valid]" 1>&2
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
    ${DATASETS_DIR}/newsela-auto/tok/train.complex \
    ${DATASETS_DIR}/newsela-auto/tok/train.simple \
    ${DATASETS_DIR}/newsela-auto/PMI_dict.pickle \
    --ignore_word_cnt 3

# --- Make constraint ---
if [ $BPE_TOKENS == "BART" ]; then
    python ${NLC_MAKER_DIR}/make_constraint_file_for_BART.py \
        ${DATASETS_DIR}/newsela-auto/tok/${MODE}.complex \
        ${DATASETS_DIR}/newsela-auto/bpe${BPE_TOKENS}/NLC_${CONSTRAINT}-for-${MODE}.complex \
        ${DATASETS_DIR}/newsela-auto/PMI_dict.pickle \
        --encoder-json ${PRE_TRAINED_DIR}/encoder.json --vocab-bpe ${PRE_TRAINED_DIR}/vocab.bpe \
        --theta $CONSTRAINT
else
    python ${NLC_MAKER_DIR}/make_constraint_file.py \
        ${DATASETS_DIR}/newsela-auto/bpe${BPE_TOKENS}/${MODE}.complex \
        ${DATASETS_DIR}/newsela-auto/bpe${BPE_TOKENS}/NLC_${CONSTRAINT}-for-${MODE}.complex \
        ${DATASETS_DIR}/newsela-auto/PMI_dict.pickle \
        --theta $CONSTRAINT
fi
