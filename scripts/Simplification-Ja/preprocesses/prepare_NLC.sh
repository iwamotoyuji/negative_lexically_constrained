#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
NLC_MAKER_DIR=../../NLC_maker

BART_SCALE=base
BPE_TOKENS=8000
CONSTRAINT=0.5
MODE=test
while getopts b:c:d:hln:p:v OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    c ) CONSTRAINT=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-c CONSTRAINT] [-d DATASETS_DIR] [-l BART_SCALE=large] [-n NLC_MAKER_DIR] [-v MODE=valid]" 1>&2
        exit 0
        ;;
    l ) BART_SCALE=large
        ;;
    n ) NLC_MAKER_DIR=$OPTARG
        ;;
    v ) MODE=valid
        ;;
    esac
done


# --- Make PMI dict ---
python ${NLC_MAKER_DIR}/make_PMI_dict.py \
    ${DATASETS_DIR}/SNOW/tok/train.complex \
    ${DATASETS_DIR}/SNOW/tok/train.simple \
    ${DATASETS_DIR}/SNOW/PMI_dict.pickle \
    --ignore_word_cnt 3

# --- Make constraint ---
if [ $BPE_TOKENS == "BART" ]; then
    bpe_path=${DATASETS_DIR}/SNOW/bpeBART-${BART_SCALE}
    args="--sentencepiece"
else
    bpe_path=${DATASETS_DIR}/SNOW/bpe${BPE_TOKENS}
    args=""
fi

python ${NLC_MAKER_DIR}/make_constraint_file.py \
    ${bpe_path}/${MODE}.complex \
    ${bpe_path}/NLC_${CONSTRAINT}-for-${MODE}.complex \
    ${DATASETS_DIR}/SNOW/PMI_dict.pickle \
    --theta $CONSTRAINT $args
