#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART
UTILS_DIR=../../../utils

BPE_TOKENS=16000
while getopts b:d:hp:u: OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-d DATASETS_DIR] [-p PRE_TRAINED_DIR] [-u UTILS_DIR]" 1>&2
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


for domain in Entertainment_Music Family_Relationships Combo; do
    # --- Path settings ---
    tok_path=${DATASETS_DIR}/GYAFC/${domain}/tok
    bpe_path=${DATASETS_DIR}/GYAFC/${domain}/bpe${BPE_TOKENS}
    mkdir -p $bpe_path

    if [ $BPE_TOKENS == "BART" ]; then
        # --- Apply codes to train and dev ---
        for file in train dev-to-formal dev-to-informal; do
            for lang in informal formal; do
                python $BART_BPE_ENCODER \
                    --encoder-json ${PRE_TRAINED_DIR}/encoder.json --vocab-bpe ${PRE_TRAINED_DIR}/vocab.bpe \
                    --inputs ${tok_path}/${file}.${lang} --outputs ${bpe_path}/${file}.${lang} \
                    --workers 60 --keep-empty
            done
        done

        # --- Apply codes to test ---
        python $BART_BPE_ENCODER \
            --encoder-json ${PRE_TRAINED_DIR}/encoder.json --vocab-bpe ${PRE_TRAINED_DIR}/vocab.bpe \
            --inputs ${tok_path}/test-to-formal.informal --outputs ${bpe_path}/test-to-formal.informal \
            --workers 60 --keep-empty
        python $BART_BPE_ENCODER \
            --encoder-json ${PRE_TRAINED_DIR}/encoder.json --vocab-bpe ${PRE_TRAINED_DIR}/vocab.bpe \
            --inputs ${tok_path}/test-to-informal.formal --outputs ${bpe_path}/test-to-informal.formal \
            --workers 60 --keep-empty

        if [ $domain == "Combo" ]; then
            for d in Entertainment_Music Family_Relationships; do
                ln -sf ../../${d}/bpe${BPE_TOKENS}/test-to-formal.informal ${bpe_path}/test-to-formal-${d}.informal
                ln -sf ../../${d}/bpe${BPE_TOKENS}/test-to-informal.formal ${bpe_path}/test-to-informal-${d}.formal
            done
        fi
    else
        BPE_CODE=${bpe_path}/code
        BPE_VOCAB=${bpe_path}/vocab
        
        # --- Learn BPE ---
        $FASTBPE learnbpe $BPE_TOKENS ${tok_path}/train.informal ${tok_path}/train.formal > $BPE_CODE

        # --- Apply codes to train ---
        $FASTBPE applybpe ${bpe_path}/train.informal ${tok_path}/train.informal $BPE_CODE
        $FASTBPE applybpe ${bpe_path}/train.formal ${tok_path}/train.formal $BPE_CODE

        # --- Get train vocabulary ---
        $FASTBPE getvocab ${bpe_path}/train.informal ${bpe_path}/train.formal > ${BPE_VOCAB}.joined_dict

        # --- Apply codes to dev ---
        for file in dev-to-formal dev-to-informal; do
            for lang in informal formal; do
                $FASTBPE applybpe ${bpe_path}/${file}.${lang} ${tok_path}/${file}.${lang} $BPE_CODE ${BPE_VOCAB}.joined_dict
            done
        done

        # --- Apply codes to test ---
        $FASTBPE applybpe ${bpe_path}/test-to-formal.informal ${tok_path}/test-to-formal.informal $BPE_CODE ${BPE_VOCAB}.joined_dict
        $FASTBPE applybpe ${bpe_path}/test-to-informal.formal ${tok_path}/test-to-informal.formal $BPE_CODE ${BPE_VOCAB}.joined_dict

        if [ $domain == "Combo" ]; then
            for d in Entertainment_Music Family_Relationships; do
                tok_path=${DATASETS_DIR}/GYAFC/${d}/tok
                $FASTBPE applybpe ${bpe_path}/test-to-formal-${d}.informal ${tok_path}/test-to-formal.informal $BPE_CODE ${BPE_VOCAB}.joined_dict
                $FASTBPE applybpe ${bpe_path}/test-to-informal-${d}.formal ${tok_path}/test-to-informal.formal $BPE_CODE ${BPE_VOCAB}.joined_dict
            done
        fi
    fi
done
