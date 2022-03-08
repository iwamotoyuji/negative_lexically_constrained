#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
UTILS_DIR=../../../utils

BPE_TOKENS=8000
while getopts b:d:hu: OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-d DATASETS_DIR] [-u UTILS_DIR]" 1>&2
        exit 0
        ;;
    u ) UTILS_DIR=$OPTARG
        ;;
    esac
done


FASTBPE=${UTILS_DIR}/fastBPE/fast

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

tok_path=${DATASETS_DIR}/SNOW/tok
bpe_path=${DATASETS_DIR}/SNOW/bpe${BPE_TOKENS}
mkdir -p $bpe_path

BPE_CODE=${bpe_path}/code
BPE_VOCAB=${bpe_path}/vocab

# --- Learn BPE ---
$FASTBPE learnbpe $BPE_TOKENS ${tok_path}/train.complex ${tok_path}/train.simple > $BPE_CODE

# --- Apply codes to train ---
$FASTBPE applybpe ${bpe_path}/train.complex ${tok_path}/train.complex $BPE_CODE
$FASTBPE applybpe ${bpe_path}/train.simple ${tok_path}/train.simple $BPE_CODE

# --- Get train vocabulary ---
$FASTBPE getvocab ${bpe_path}/train.complex ${bpe_path}/train.simple > ${BPE_VOCAB}.joined_dict

# --- Apply codes to valid ---
$FASTBPE applybpe ${bpe_path}/valid.complex ${tok_path}/valid.complex $BPE_CODE ${BPE_VOCAB}.joined_dict
$FASTBPE applybpe ${bpe_path}/valid.simple ${tok_path}/valid.simple $BPE_CODE ${BPE_VOCAB}.joined_dict

# --- Apply codes to test ---
$FASTBPE applybpe ${bpe_path}/test.complex ${tok_path}/test.complex $BPE_CODE ${BPE_VOCAB}.joined_dict
$FASTBPE applybpe ${bpe_path}/test.simple ${tok_path}/test.simple.6 $BPE_CODE ${BPE_VOCAB}.joined_dict
