#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-JaBART

BART_SCALE=base
BPE_TOKENS=8000
while getopts b:d:hl OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-d DATASETS_DIR] [-l BART_SCALE=large] [-p PRE_TRAINED_DIR]" 1>&2
        exit 0
        ;;
    l ) BART_SCALE=large
        ;;
    esac
done


train_text=${DATASETS_DIR}/SNOW/bpe${BPE_TOKENS}/train
valid_text=${DATASETS_DIR}/SNOW/bpe${BPE_TOKENS}/valid
src_lang=complex
tgt_lang=simple

if [ $BPE_TOKENS == "BART" ]; then
    preprocessed_dir=${DATASETS_DIR}/SNOW/bpeBART-${BART_SCALE}/fairseq-preprocess
    rm -fr $preprocessed_dir
    mkdir -p $preprocessed_dir
    fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
        --trainpref ${DATASETS_DIR}/SNOW/bpeBART-${BART_SCALE}/train --validpref ${DATASETS_DIR}/SNOW/bpeBART-${BART_SCALE}/valid \
        --destdir $preprocessed_dir \
        --workers 60 \
        --srcdict ${PRE_TRAINED_DIR}/japanese_bart_${BART_SCALE}_2.0/dict.txt --tgtdict ${PRE_TRAINED_DIR}/japanese_bart_${BART_SCALE}_2.0/dict.txt
else
    preprocessed_dir=${DATASETS_DIR}/SNOW/bpe${BPE_TOKENS}/fairseq-preprocess
    rm -fr $preprocessed_dir
    mkdir -p $preprocessed_dir
    fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
        --trainpref ${DATASETS_DIR}/SNOW/bpe${BPE_TOKENS}/train --validpref ${DATASETS_DIR}/SNOW/bpe${BPE_TOKENS}/valid \
        --destdir $preprocessed_dir \
        --workers 60 \
        --joined-dictionary
fi