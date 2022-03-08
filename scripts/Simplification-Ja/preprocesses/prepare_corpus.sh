#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-JaBART
RESOURCE_DIR=../../../resources

BART_SCALE=base
while getopts d:hlr: OPT
do
    case $OPT in
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-d DATASETS_DIR] [-l BART_SCALE=large] [-r RESOURCE_DIR]" 1>&2
        exit 0
        ;;
    l ) BART_SCALE=large
        ;;
    r ) RESOURCE_DIR=$OPTARG
        ;;
    esac
done


EXTRUCTOR=$(dirname $0)/SNOW_extructor.py
SPLITER=$(dirname $0)/split_train_val.py
PREPROCESSER=$(dirname $0)/jaBART_preprocess.py
bpe_model=${PRE_TRAINED_DIR}/japanese_bart_${BART_SCALE}_2.0/sp.model
bpe_dict=${PRE_TRAINED_DIR}/japanese_bart_${BART_SCALE}_2.0/dict.txt

# --- Download JaBART ---
mkdir -p $PRE_TRAINED_DIR
pushd $PRE_TRAINED_DIR
if [ -d ./japanese_bart_${BART_SCALE}_2.0 ]; then
    echo "[Info] JaBART already exists, skipping download"
else
    echo "[Info] Downloading JaBART..."
    wget http://lotus.kuee.kyoto-u.ac.jp/nl-resource/JapaneseBARTPretrainedModel/japanese_bart_${BART_SCALE}_2.0.tar.gz
    tar -zxvf japanese_bart_${BART_SCALE}_2.0.tar.gz
fi

orig_path=${RESOURCE_DIR}/SNOW
tok_path=${DATASETS_DIR}/SNOW/tok
bpe_path=${DATASETS_DIR}/SNOW/bpeBART-${BART_SCALE}
mkdir -p ${orig_path}/T15 ${orig_path}/T23 $tok_path $bpe_path

# --- Preprocess T15 ---
echo "[Info] Extructing T15 data..."
python $EXTRUCTOR ${orig_path}/T15-2020.1.7.xlsx ${orig_path}/T15/train

# --- Preprocess T23 ---
echo "[Info] Extructing T23 data..."
python $EXTRUCTOR ${orig_path}/T23-2020.1.7.xlsx ${orig_path}/T23/train
python $EXTRUCTOR ${orig_path}/T23-2020.1.7.xlsx ${orig_path}/T23/test --test

cat ${orig_path}/T15/train.complex ${orig_path}/T23/train.complex > ${orig_path}/train.complex
cat ${orig_path}/T15/train.simple ${orig_path}/T23/train.simple > ${orig_path}/train.simple

python $SPLITER ${orig_path}/train ${orig_path}/splited-train ${orig_path}/splited-valid -s complex -t simple

# --- Tokenize T15 ---
echo "[Info] Tokenizing train data"
python $PREPROCESSER ${orig_path}/splited-train.complex ${tok_path}/train.complex ${bpe_path}/train.complex --bpe_model $bpe_model --bpe_dict $bpe_dict
python $PREPROCESSER ${orig_path}/splited-train.simple ${tok_path}/train.simple ${bpe_path}/train.simple --bpe_model $bpe_model --bpe_dict $bpe_dict

echo "[Info] Tokenizing valid data"
python $PREPROCESSER ${orig_path}/splited-valid.complex ${tok_path}/valid.complex ${bpe_path}/valid.complex --bpe_model $bpe_model --bpe_dict $bpe_dict
python $PREPROCESSER ${orig_path}/splited-valid.simple ${tok_path}/valid.simple ${bpe_path}/valid.simple --bpe_model $bpe_model --bpe_dict $bpe_dict

echo "[Info] Tokenizing test data"
python $PREPROCESSER ${orig_path}/T23/test.complex ${tok_path}/test.complex ${bpe_path}/test.complex --bpe_model $bpe_model --bpe_dict $bpe_dict
for i in $(seq 0 6); do
    python $PREPROCESSER ${orig_path}/T23/test.simple.${i} ${tok_path}/test.simple.${i} ${bpe_path}/test.simple.${i} --bpe_model $bpe_model --bpe_dict $bpe_dict
done
