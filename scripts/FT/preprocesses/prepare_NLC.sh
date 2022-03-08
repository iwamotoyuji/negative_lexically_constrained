#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
NLC_MAKER_DIR=../../NLC_maker
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART

BPE_TOKENS=16000
CONSTRAINT=0.5
while getopts b:c:d:hn:p: OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    c ) CONSTRAINT=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-c CONSTRAINT] [-d DATASETS_DIR] [-n NLC_MAKER_DIR] [-p PRE_TRAINED_DIR]" 1>&2
        exit 0
        ;;
    n ) NLC_MAKER_DIR=$OPTARG
        ;;
    p ) PRE_TRAINED_DIR=$OPTARG
        ;;
    esac
done


for domain in Entertainment_Music Family_Relationships Combo; do
    # --- Make PMI dict ---
    python ${NLC_MAKER_DIR}/make_PMI_dict.py \
        ${DATASETS_DIR}/GYAFC/${domain}/tok/train.informal \
        ${DATASETS_DIR}/GYAFC/${domain}/tok/train.formal \
        ${DATASETS_DIR}/GYAFC/${domain}/PMI_dict.pickle \
        --ignore_word_cnt 3
    
    # --- Make constraint ---
    if [ $BPE_TOKENS == "BART" ]; then
        python ${NLC_MAKER_DIR}/make_constraint_file_for_BART.py \
            ${DATASETS_DIR}/GYAFC/${domain}/tok/test-to-formal.informal \
            ${DATASETS_DIR}/GYAFC/${domain}/bpe${BPE_TOKENS}/NLC_${CONSTRAINT}-${domain}.informal \
            ${DATASETS_DIR}/GYAFC/${domain}/PMI_dict.pickle \
            --encoder-json ${PRE_TRAINED_DIR}/encoder.json --vocab-bpe ${PRE_TRAINED_DIR}/vocab.bpe \
            --theta $CONSTRAINT
        if [ $domain == "Combo" ]; then
            for d in Entertainment_Music Family_Relationships; do
                ln -sf ../../${d}/bpe${BPE_TOKENS}/NLC_${CONSTRAINT}-${d}.informal ${DATASETS_DIR}/GYAFC/${domain}/bpe${BPE_TOKENS}/NLC_${CONSTRAINT}-${d}.informal
            done
        fi
    else
        # --- Make constraint ---
        python ${NLC_MAKER_DIR}/make_constraint_file.py \
            ${DATASETS_DIR}/GYAFC/${domain}/bpe${BPE_TOKENS}/test-to-formal.informal \
            ${DATASETS_DIR}/GYAFC/${domain}/bpe${BPE_TOKENS}/NLC_${CONSTRAINT}-${domain}.informal \
            ${DATASETS_DIR}/GYAFC/${domain}/PMI_dict.pickle \
            --theta $CONSTRAINT
        if [ $domain == "Combo" ]; then
            for d in Entertainment_Music Family_Relationships; do
                python ${NLC_MAKER_DIR}/make_constraint_file.py \
                    ${DATASETS_DIR}/GYAFC/${domain}/bpe${BPE_TOKENS}/test-to-formal-${d}.informal \
                    ${DATASETS_DIR}/GYAFC/${domain}/bpe${BPE_TOKENS}/NLC_${CONSTRAINT}-${d}.informal \
                    ${DATASETS_DIR}/GYAFC/${d}/PMI_dict.pickle \
                    --theta $CONSTRAINT
            done
        fi
    fi
done
    