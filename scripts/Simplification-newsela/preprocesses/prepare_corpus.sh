#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
RESOURCE_DIR=../../../resources
UTILS_DIR=/home/iwamoto/utils

while getopts d:hr:u: OPT
do
    case $OPT in
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-d DATASETS_DIR] [-r RESOURCE_DIR] [-u UTILS_DIR]" 1>&2
        exit 0
        ;;
    r ) RESOURCE_DIR=$OPTARG
        ;;
    u ) UTILS_DIR=$OPTARG
        ;;
    esac
done


NORMALIZER=${UTILS_DIR}/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl
TOKENIZER=${UTILS_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl
DETOKENIZER=${UTILS_DIR}/mosesdecoder/scripts/tokenizer/detokenizer.perl

# --- Download utils ---
mkdir -p $UTILS_DIR
pushd $UTILS_DIR
if [ -d ./mosesdecoder ]; then
    echo "[Info] mosesdecoder already exists, skipping download"
else
    echo "[Info] Cloning Moses github repository (for tokenization scripts)..."
    git clone https://github.com/moses-smt/mosesdecoder.git    
fi

if [ -f ./script.converter.distribution/z2h-utf8.pl ]; then
    echo "[Info] z2h already exists, skipping download"
else
    wget http://lotus.kuee.kyoto-u.ac.jp/WAT/WAT2019/baseline/script.converter.distribution.tar.gz
    tar -zxvf script.converter.distribution.tar.gz
fi
popd


orig_path=${RESOURCE_DIR}/newsela-auto/ACL2020
tok_path=${DATASETS_DIR}/newsela-auto/tok
mkdir -p $tok_path

# --- Preprocess newsela-auto ---
echo "[Info] Tokenizing newsela-auto train data"
cat ${orig_path}/train.src | $NORMALIZER -l en > ${tok_path}/train.complex
cat ${orig_path}/train.dst | $NORMALIZER -l en > ${tok_path}/train.simple

echo "[Info] Tokenizing newsela-auto valid data..."
cat ${orig_path}/valid.src | $NORMALIZER -l en > ${tok_path}/valid.complex
cat ${orig_path}/valid.dst > ${tok_path}/valid.simple

echo "[Info] Tokenizing newsela-auto test data..."
cat ${orig_path}/test.src | $NORMALIZER -l en > ${tok_path}/test.complex
cat ${orig_path}/test.dst > ${tok_path}/test.simple
