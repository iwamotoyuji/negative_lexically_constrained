#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART
RESULT_DIR=../../../results/Simplification-wikipedia
UTILS_DIR=/home/iwamoto/utils

BPE_TOKENS=16000
BART_SCALE=base
CONSTRAINT=0
EXP_NAME=RNN
GPU_ID=0
MODEL=checkpoint_best.pt
MODE=test
TRAIN_SET=wiki-large
TEST_SET=asset
while getopts ab:c:d:g:hlm:n:p:r:tu:v OPT
do
    case $OPT in
    a ) TRAIN_SET=wiki-auto
        ;;
    b ) BPE_TOKENS=$OPTARG
        ;;
    c ) CONSTRAINT=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-a TRAIN_SET=wiki-auto] [-b BPE_TOKENS] [-c CONSTRAINT] [-d DATASETS_DIR] [-g GPU_ID] [-l BART_SCALE=large] [-m MODEL] [-n EXP_NAME] [-p PRE_TRAINED_DIR] [-r RESULT_DIR] [-t TEST_SET=turkcorpus] [-u UTILS_DIR] [-v MODE=valid]" 1>&2
        exit 1
        ;;
    l ) BART_SCALE=large
        ;;
    m ) MODEL=$OPTARG
        ;;
    n ) EXP_NAME=$OPTARG
        ;;
    p ) PRE_TRAINED_DIR=$OPTARG
        ;;
    r ) RESULT_DIR=$OPTARG
        ;;
    t ) TEST_SET=turkcorpus
        ;;        
    u ) UTILS_DIR=$OPTARG
        ;;
    v ) MODE=valid
        ;;
    esac
done


DETOKENIZER=${UTILS_DIR}/mosesdecoder/scripts/tokenizer/detokenizer.perl
BART_BPE_DECODER=${PRE_TRAINED_DIR}/multiprocessing_bpe_decoder.py

# --- Download utils ---
if type easse > /dev/null 2>&1; then
    echo "[Info] easse already exists, skipping download"
else
    echo "[Info] easse not exists, start installation to ${UTILS_DIR}/easse"
    pushd $UTILS_DIR
    git clone https://github.com/feralvam/easse.git
    cd easse
    pip install -e .
    popd
fi


if [ $TEST_SET == "asset" ]; then
    if [ $EXP_NAME == "RNN_ensemble" ]; then
        output_dir=${RESULT_DIR}/${TRAIN_SET}/RNN_ensemble/${MODE}
        model=${RESULT_DIR}/${TRAIN_SET}/RNN_SEED11/checkpoints/checkpoint_7_30000.pt:${RESULT_DIR}/${TRAIN_SET}/RNN_SEED22/checkpoints/checkpoint_4_17000.pt:${RESULT_DIR}/${TRAIN_SET}/RNN_SEED33/checkpoints/checkpoint_3_10000.pt:${RESULT_DIR}/${TRAIN_SET}/RNN_SEED44/checkpoints/checkpoint_4_17000.pt
    elif [ $EXP_NAME == "SAN_ensemble" ]; then
        output_dir=${RESULT_DIR}/${TRAIN_SET}/SAN_ensemble/${MODE}
        model=${RESULT_DIR}/${TRAIN_SET}/SAN_SEED11/checkpoints/checkpoint_7_30000.pt:${RESULT_DIR}/${TRAIN_SET}/SAN_SEED22/checkpoints/checkpoint_6_26000.pt:${RESULT_DIR}/${TRAIN_SET}/SAN_SEED33/checkpoints/checkpoint_7_28000.pt:${RESULT_DIR}/${TRAIN_SET}/SAN_SEED44/checkpoints/checkpoint_7_30000.pt
    elif [ $EXP_NAME == "BART_base_ensemble" ]; then
        output_dir=${RESULT_DIR}/${TRAIN_SET}/BART_base_ensemble/${MODE}
        model=${RESULT_DIR}/${TRAIN_SET}/BART_base_SEED11/checkpoints/checkpoint_5_22000.pt:${RESULT_DIR}/${TRAIN_SET}/BART_base_SEED22/checkpoints/checkpoint_4_15000.pt:${RESULT_DIR}/${TRAIN_SET}/BART_base_SEED33/checkpoints/checkpoint_4_16000.pt:${RESULT_DIR}/${TRAIN_SET}/BART_base_SEED44/checkpoints/checkpoint_6_28000.pt
    elif [ $EXP_NAME == "BART_large_ensemble" ]; then
        output_dir=${RESULT_DIR}/${TRAIN_SET}/BART_large_ensemble/${MODE}
        model=${RESULT_DIR}/${TRAIN_SET}/BART_large_SEED11/checkpoints/checkpoint_5_22000.pt:${RESULT_DIR}/${TRAIN_SET}/BART_large_SEED22/checkpoints/checkpoint_5_19000.pt:${RESULT_DIR}/${TRAIN_SET}/BART_large_SEED33/checkpoints/checkpoint_4_16000.pt:${RESULT_DIR}/${TRAIN_SET}/BART_large_SEED44/checkpoints/checkpoint_6_28000.pt
    else
        output_dir=${RESULT_DIR}/${TRAIN_SET}/${EXP_NAME}/${MODE}
        model=${RESULT_DIR}/${TRAIN_SET}/${EXP_NAME}/checkpoints/${MODEL}
    fi
else
    if [ $EXP_NAME == "RNN_ensemble" ]; then
        output_dir=${RESULT_DIR}/${TRAIN_SET}/RNN_ensemble/${MODE}
        model=${RESULT_DIR}/${TRAIN_SET}/RNN_SEED11/checkpoints/checkpoint_7_28000.pt:${RESULT_DIR}/${TRAIN_SET}/RNN_SEED22/checkpoints/checkpoint_4_15000.pt:${RESULT_DIR}/${TRAIN_SET}/RNN_SEED33/checkpoints/checkpoint_6_23000.pt:${RESULT_DIR}/${TRAIN_SET}/RNN_SEED44/checkpoints/checkpoint_7_28000.pt
    elif [ $EXP_NAME == "SAN_ensemble" ]; then
        output_dir=${RESULT_DIR}/${TRAIN_SET}/SAN_ensemble/${MODE}
        model=${RESULT_DIR}/${TRAIN_SET}/SAN_SEED11/checkpoints/checkpoint_7_28000.pt:${RESULT_DIR}/${TRAIN_SET}/SAN_SEED22/checkpoints/checkpoint_7_29000.pt:${RESULT_DIR}/${TRAIN_SET}/SAN_SEED33/checkpoints/checkpoint_7_30000.pt:${RESULT_DIR}/${TRAIN_SET}/SAN_SEED44/checkpoints/checkpoint_7_28000.pt
    elif [ $EXP_NAME == "BART_base_ensemble" ]; then
        output_dir=${RESULT_DIR}/${TRAIN_SET}/BART_base_ensemble/${MODE}
        model=${RESULT_DIR}/${TRAIN_SET}/BART_base_SEED11/checkpoints/checkpoint_6_26000.pt:${RESULT_DIR}/${TRAIN_SET}/BART_base_SEED22/checkpoints/checkpoint_6_25000.pt:${RESULT_DIR}/${TRAIN_SET}/BART_base_SEED33/checkpoints/checkpoint_6_27000.pt:${RESULT_DIR}/${TRAIN_SET}/BART_base_SEED44/checkpoints/checkpoint_6_27000.pt
    elif [ $EXP_NAME == "BART_large_ensemble" ]; then
        output_dir=${RESULT_DIR}/${TRAIN_SET}/BART_large_ensemble/${MODE}
        model=${RESULT_DIR}/${TRAIN_SET}/BART_large_SEED11/checkpoints/checkpoint_6_26000.pt:${RESULT_DIR}/${TRAIN_SET}/BART_large_SEED22/checkpoints/checkpoint_4_17000.pt:${RESULT_DIR}/${TRAIN_SET}/BART_large_SEED33/checkpoints/checkpoint_6_27000.pt:${RESULT_DIR}/${TRAIN_SET}/BART_large_SEED44/checkpoints/checkpoint_6_27000.pt
    else
        output_dir=${RESULT_DIR}/${TRAIN_SET}/${EXP_NAME}/${MODE}
        model=${RESULT_DIR}/${TRAIN_SET}/${EXP_NAME}/checkpoints/${MODEL}
    fi
fi
mkdir -p $output_dir

input_dir=${DATASETS_DIR}/${TEST_SET}/bpe${BPE_TOKENS}-from-${TRAIN_SET}
reference_dir=${DATASETS_DIR}/${TEST_SET}/tok
if [ $TEST_SET == "asset" ]; then
    references=${reference_dir}/${MODE}.orig.simple.0,${reference_dir}/${MODE}.orig.simple.1,${reference_dir}/${MODE}.orig.simple.2,${reference_dir}/${MODE}.orig.simple.3,${reference_dir}/${MODE}.orig.simple.4,${reference_dir}/${MODE}.orig.simple.5,${reference_dir}/${MODE}.orig.simple.6,${reference_dir}/${MODE}.orig.simple.7,${reference_dir}/${MODE}.orig.simple.8,${reference_dir}/${MODE}.orig.simple.9
else
    references=${reference_dir}/${MODE}.orig.simple.0,${reference_dir}/${MODE}.orig.simple.1,${reference_dir}/${MODE}.orig.simple.2,${reference_dir}/${MODE}.orig.simple.3,${reference_dir}/${MODE}.orig.simple.4,${reference_dir}/${MODE}.orig.simple.5,${reference_dir}/${MODE}.orig.simple.6,${reference_dir}/${MODE}.orig.simple.7
fi

if [ $BPE_TOKENS == "BART" ]; then
    preprocessed_dir=${DATASETS_DIR}/${TRAIN_SET}/bpe${BPE_TOKENS}/fairseq-preprocess-BART-${BART_SCALE}
else
    preprocessed_dir=${DATASETS_DIR}/${TRAIN_SET}/bpe${BPE_TOKENS}/fairseq-preprocess
fi

if [ $CONSTRAINT == 0 ]; then
    input=${input_dir}/${MODE}.complex
    result=${output_dir}/${TEST_SET}
    log=${output_dir}/eval.log
    arg=""
else
    input=${input_dir}/${MODE}-add-NLC_${CONSTRAINT}.complex
    result=${output_dir}/${TEST_SET}-NLC_${CONSTRAINT}
    log=${output_dir}/eval-NLC_${CONSTRAINT}.log
    arg="--constraints"
    paste -d "" ${input_dir}/${MODE}.complex ${input_dir}/NLC_${CONSTRAINT}-for-${MODE}.complex > $input
fi

CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-interactive $preprocessed_dir \
    --input $input \
    --path $model \
    --buffer-size 1024 --batch-size 64 \
    --nbest 1 --max-len-b 50 \
    --beam 5 --lenpen 1.0 --remove-bpe $arg \
    2>&1 | tee ${result}.txt
grep ^H ${result}.txt | cut -f3- > ${result}.sys

if [ $BPE_TOKENS == "BART" ]; then
    cp ${result}.sys ${result}.tmp.sys
    python $BART_BPE_DECODER \
        --encoder-json ${PRE_TRAINED_DIR}/encoder.json --vocab-bpe ${PRE_TRAINED_DIR}/vocab.bpe \
        --inputs ${result}.tmp.sys --outputs ${result}.sys \
        --workers 60 --keep-empty
fi

cat ${result}.sys | perl -C $DETOKENIZER -l en > ${result}.detok.sys

echo model name : $MODEL >> $log
easse evaluate -t custom -m 'bleu,sari' \
    --refs_sents_paths $references \
    --orig_sents_path ${reference_dir}/${MODE}.orig.complex \
    --sys_sents_path ${result}.detok.sys \
    2>&1 | tee -a $log
