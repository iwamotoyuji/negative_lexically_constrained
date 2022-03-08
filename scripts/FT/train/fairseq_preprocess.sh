#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART

BPE_TOKENS=16000
BART_SCALE=base
DOMAIN=Combo
while getopts b:d:efhlp: OPT
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
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-d DATASETS_DIR] [-e DOMAIN=Entertainment_Music] [-f DOMAIN=Family_Relationships] [-l BART_SCALE=large] [-p PRE_TRAINED_DIR]" 1>&2
        exit 0
        ;;
    l ) BART_SCALE=large
        ;;
    p ) PRE_TRAINED_DIR=$OPTARG
        ;;
    esac
done

# --- Download BART ---
mkdir -p $PRE_TRAINED_DIR
pushd $PRE_TRAINED_DIR
if [ -d ./bart.base ]; then
    echo "[Info] Pre-trained BART base model already exists, skipping download"
else
    echo "[Info] Downloading Pre-trained BART base model"
    wget https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
    tar -xzvf bart.base.tar.gz
fi
if [ -d ./bart.large ]; then
    echo "[Info] Pre-trained BART large model already exists, skipping download"
else
    wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
    tar -xzvf bart.large.tar.gz
fi
popd


train_text=${DATASETS_DIR}/GYAFC/${DOMAIN}/bpe${BPE_TOKENS}/bidirectional/train
valid_text=${DATASETS_DIR}/GYAFC/${DOMAIN}/bpe${BPE_TOKENS}/bidirectional/dev
src_lang=src
tgt_lang=tgt

if [ $BPE_TOKENS == "BART" ]; then
    preprocessed_dir=${DATASETS_DIR}/GYAFC/${DOMAIN}/bpe${BPE_TOKENS}/bidirectional/fairseq-preprocess-BART-${BART_SCALE}
    rm -fr $preprocessed_dir
    mkdir -p $preprocessed_dir
    fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
        --trainpref $train_text --validpref $valid_text \
        --destdir $preprocessed_dir \
        --workers 60 \
        --srcdict ${PRE_TRAINED_DIR}/bart.${BART_SCALE}/dict.txt --tgtdict ${PRE_TRAINED_DIR}/bart.${BART_SCALE}/dict.txt
else
    preprocessed_dir=${DATASETS_DIR}/GYAFC/${DOMAIN}/bpe${BPE_TOKENS}/bidirectional/fairseq-preprocess
    rm -fr $preprocessed_dir
    mkdir -p $preprocessed_dir
    fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
        --trainpref $train_text --validpref $valid_text \
        --destdir $preprocessed_dir \
        --workers 60 \
        --joined-dictionary
fi