#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART
UTILS_DIR=../../../utils

BPE_TOKENS=16000
TRAIN_SET=wiki-large
while getopts ab:d:hp:u: OPT
do
    case $OPT in
    a ) TRAIN_SET=wiki-auto
        ;;
    b ) BPE_TOKENS=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-a TRAIN_SET=wiki-auto] [-b BPE_TOKENS] [-d DATASETS_DIR] [-p PRE_TRAINED_DIR] [-u UTILS_DIR]" 1>&2
        exit 0
        ;;
    p ) PRE_TRAINED_DIR=$OPTARG
        ;;
    u ) UTILS_DIR=$OPTARG
        ;;
    esac
done


FASTBPE=${UTILS_DIR}/fastBPE/fast
BART_BPE_ENCODER=${PRE_TRAINED_DIR}/multiprocessing_bpe_encoder.py

# --- Download utils ---
mkdir -p $UTILS_DIR
pushd $UTILS_DIR
if [ -d ./fastBPE ]; then
    echo "[Info] fastBPE already exists, skipping download"
else
    echo "[Info] Cloning fastBPE repository (for BPE pre-processing)..."
    git clone https://github.com/glample/fastBPE.git
fi
if [ -f ./fastBPE/fast ]; then
    echo "[Info] fastBPE already exists, skipping install"
else
    cd ./fastBPE
    g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
    cd ..
    if ! [[ -f ./fastBPE/fast ]]; then
        echo "[Error] fastBPE not successfully installed, abort."
        exit 1
    fi
fi
popd

# --- Download BART ---
mkdir -p $PRE_TRAINED_DIR
pushd $PRE_TRAINED_DIR
if [ -f ./encoder.json ]; then
    echo "[Info] BART encoder.json already exists, skipping download"
else
    echo "[Info] Downloading BART encoder.json"
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
fi
if [ -f ./vocab.bpe ]; then
    echo "[Info] BART vocab.bpe already exists, skipping download"
else
    echo "[Info] Downloading BART vocab.bpe"
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
fi
popd


tok_path=${DATASETS_DIR}/${TRAIN_SET}/tok
bpe_path=${DATASETS_DIR}/${TRAIN_SET}/bpe${BPE_TOKENS}
mkdir -p $bpe_path

# --- Apply BPE to train ---
if [ $BPE_TOKENS == "BART" ]; then
    for file in train.complex train.simple; do
        python $BART_BPE_ENCODER \
            --encoder-json ${PRE_TRAINED_DIR}/encoder.json --vocab-bpe ${PRE_TRAINED_DIR}/vocab.bpe \
            --inputs ${tok_path}/${file} --outputs ${bpe_path}/${file} \
            --workers 60 --keep-empty
    done

    # --- Apply codes to valid and test ---
    for dataset in asset turkcorpus; do
        tok_path=${DATASETS_DIR}/${dataset}/tok
        bpe_path=${DATASETS_DIR}/${dataset}/bpe${BPE_TOKENS}-from-${TRAIN_SET}
        mkdir -p $bpe_path
        for file in valid.complex valid-all.complex valid-all.simple test.complex; do
            python $BART_BPE_ENCODER \
                --encoder-json ${PRE_TRAINED_DIR}/encoder.json --vocab-bpe ${PRE_TRAINED_DIR}/vocab.bpe \
                --inputs ${tok_path}/${file} --outputs ${bpe_path}/${file} \
                --workers 60 --keep-empty
        done
    done
else
    BPE_CODE=${bpe_path}/code
    BPE_VOCAB=${bpe_path}/vocab
    
    # --- Learn BPE ---
    $FASTBPE learnbpe $BPE_TOKENS ${tok_path}/train.complex ${tok_path}/train.simple > $BPE_CODE

    # --- Apply codes to train ---
    $FASTBPE applybpe ${bpe_path}/train.complex ${tok_path}/train.complex $BPE_CODE
    $FASTBPE applybpe ${bpe_path}/train.simple ${tok_path}/train.simple $BPE_CODE

    # --- Get train vocabulary ---
    $FASTBPE getvocab ${bpe_path}/train.complex ${bpe_path}/train.simple > ${BPE_VOCAB}.joined_dict

    # --- Apply codes to valid and test ---
    for dataset in asset turkcorpus; do
        tok_path=${DATASETS_DIR}/${dataset}/tok
        bpe_path=${DATASETS_DIR}/${dataset}/bpe${BPE_TOKENS}-from-${TRAIN_SET}
        mkdir -p $bpe_path
        for file in valid.complex valid-all.complex valid-all.simple test.complex; do
            $FASTBPE applybpe ${bpe_path}/${file} ${tok_path}/${file} $BPE_CODE ${BPE_VOCAB}.joined_dict
            $FASTBPE applybpe ${bpe_path}/${file} ${tok_path}/${file} $BPE_CODE ${BPE_VOCAB}.joined_dict
        done
    done
fi
