#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets

BPE_TOKENS=16000
while getopts b:d:h OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-d DATASETS_DIR]" 1>&2
        exit 0
        ;;
    esac
done


for domain in Entertainment_Music Family_Relationships Combo; do
    bpe_path=${DATASETS_DIR}/GYAFC/${domain}/bpe${BPE_TOKENS}
    bidirectional_path=${DATASETS_DIR}/GYAFC/${domain}/bpe${BPE_TOKENS}/bidirectional
    mkdir -p $bidirectional_path
    if [ -d $bpe_path ]; then
        echo "[Info] Make bidirectional data to ${bidirectional_path} from ${bpe_path}"
    else
        echo "[Error] Not exist ${bpe_path}"
        exit 1
    fi

    if [ $BPE_TOKENS == "BART" ]; then
        formal_tag="201"
        informal_tag="202"
    else
        formal_tag="<F>"
        informal_tag="<I>"
    fi

    # --- train data ---
    sed "s/^/${formal_tag} /g" ${bpe_path}/train.informal > ${bidirectional_path}/train.src
    sed "s/^/${informal_tag} /g" ${bpe_path}/train.formal >> ${bidirectional_path}/train.src
    cat ${bpe_path}/train.formal > ${bidirectional_path}/train.tgt
    cat ${bpe_path}/train.informal >> ${bidirectional_path}/train.tgt

    # --- dev data ---
    sed "s/^/${formal_tag} /g" ${bpe_path}/dev-to-formal.informal > ${bidirectional_path}/dev.src
    sed "s/^/${informal_tag} /g" ${bpe_path}/dev-to-informal.formal >> ${bidirectional_path}/dev.src
    cat ${bpe_path}/dev-to-formal.formal > ${bidirectional_path}/dev.tgt
    cat ${bpe_path}/dev-to-informal.informal >> ${bidirectional_path}/dev.tgt

    # --- test data ---
    sed "s/^/${formal_tag} /g" ${bpe_path}/test-to-formal.informal > ${bidirectional_path}/test-to-formal.informal
    sed "s/^/${informal_tag} /g" ${bpe_path}/test-to-informal.formal > ${bidirectional_path}/test-to-informal.formal

    if [ $domain == "Combo" ]; then
        for d in Entertainment_Music Family_Relationships; do
            sed "s/^/${formal_tag} /g" ${bpe_path}/test-to-formal-${d}.informal > ${bidirectional_path}/test-to-formal-${d}.informal
            sed "s/^/${informal_tag} /g" ${bpe_path}/test-to-informal-${d}.formal > ${bidirectional_path}/test-to-informal-${d}.formal
        done
    fi
done