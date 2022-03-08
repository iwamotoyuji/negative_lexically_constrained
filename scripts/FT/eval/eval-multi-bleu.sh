#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
PRE_TRAINED_DIR=../../../pre-trained/fairseq-BART
RESULT_DIR=../../../results/FT
UTILS_DIR=../../../utils

BPE_TOKENS=16000
BART_SCALE=base
CONSTRAINT=0
DOMAIN=Combo
EXP_NAME=RNN
GPU_ID=0
while getopts b:c:d:efg:hln:p:r:u: OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    c ) CONSTRAINT=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    e ) DOMAIN=Entertainment_Music
        ;;
    f ) DOMAIN=Family_Relationships
        ;;
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-c CONSTRAINT] [-d DATASETS_DIR] [-e DOMAIN=Entertainment_Music] [-f DOMAIN=Family_Relationships] [-g GPU_ID] [-l BART_SCALE=large] [-n EXP_NAME] [-p PRE_TRAINED_DIR] [-r RESULT_DIR] [-u UTILS_DIR]" 1>&2
        exit 1
        ;;
    l ) BART_SCALE=large
        ;;
    n ) EXP_NAME=$OPTARG
        ;;
    p ) PRE_TRAINED_DIR=$OPTARG
        ;;        
    r ) RESULT_DIR=$OPTARG
        ;;
    u ) UTILS_DIR=$OPTARG
        ;;
    esac
done


DETOKENIZER=${UTILS_DIR}/mosesdecoder/scripts/tokenizer/detokenizer.perl
BLEU=$(dirname $0)/multi-bleu-detok.perl
BART_BPE_DECODER=${PRE_TRAINED_DIR}/multiprocessing_bpe_decoder.py


if [ $EXP_NAME == "RNN_ensemble" ]; then
    output_dir=${RESULT_DIR}/${DOMAIN}/RNN_ensemble
    model=${RESULT_DIR}/${DOMAIN}/RNN_SEED11/checkpoints/checkpoint_best.pt:${RESULT_DIR}/${DOMAIN}/RNN_SEED22/checkpoints/checkpoint_best.pt:${RESULT_DIR}/${DOMAIN}/RNN_SEED33/checkpoints/checkpoint_best.pt:${RESULT_DIR}/${DOMAIN}/RNN_SEED44/checkpoints/checkpoint_best.pt
elif [ $EXP_NAME == "SAN_ensemble" ]; then
    output_dir=${RESULT_DIR}/${DOMAIN}/SAN_ensemble
    model=${RESULT_DIR}/${DOMAIN}/SAN_SEED11/checkpoints/checkpoint_best.pt:${RESULT_DIR}/${DOMAIN}/SAN_SEED22/checkpoints/checkpoint_best.pt:${RESULT_DIR}/${DOMAIN}/SAN_SEED33/checkpoints/checkpoint_best.pt:${RESULT_DIR}/${DOMAIN}/SAN_SEED44/checkpoints/checkpoint_best.pt
elif [ $EXP_NAME == "BART_base_ensemble" ]; then
    output_dir=${RESULT_DIR}/${DOMAIN}/BART_base_ensemble
    model=${RESULT_DIR}/${DOMAIN}/BART_base_SEED11/checkpoints/checkpoint_best.pt:${RESULT_DIR}/${DOMAIN}/BART_base_SEED22/checkpoints/checkpoint_best.pt:${RESULT_DIR}/${DOMAIN}/BART_base_SEED33/checkpoints/checkpoint_best.pt:${RESULT_DIR}/${DOMAIN}/BART_base_SEED44/checkpoints/checkpoint_best.pt
elif [ $EXP_NAME == "BART_large_ensemble" ]; then
    output_dir=${RESULT_DIR}/${DOMAIN}/BART_large_ensemble
    model=${RESULT_DIR}/${DOMAIN}/BART_large_SEED11/checkpoints/checkpoint_best.pt:${RESULT_DIR}/${DOMAIN}/BART_large_SEED22/checkpoints/checkpoint_best.pt:${RESULT_DIR}/${DOMAIN}/BART_large_SEED33/checkpoints/checkpoint_best.pt:${RESULT_DIR}/${DOMAIN}/BART_large_SEED44/checkpoints/checkpoint_best.pt
else
    output_dir=${RESULT_DIR}/${DOMAIN}/${EXP_NAME}
    model=${output_dir}/checkpoints/checkpoint_best.pt
fi
mkdir -p $output_dir

input_dir=${DATASETS_DIR}/GYAFC/${DOMAIN}/bpe${BPE_TOKENS}/bidirectional

if [ $BPE_TOKENS == "BART" ]; then
    preprocessed_dir=${input_dir}/fairseq-preprocess-BART-${BART_SCALE}
else
    preprocessed_dir=${input_dir}/fairseq-preprocess
fi

if [ $DOMAIN == "Combo" ]; then
    for d in Entertainment_Music Family_Relationships; do
        reference_dir=${DATASETS_DIR}/GYAFC/${d}/tok

        if [ $CONSTRAINT == 0 ]; then
            input=${input_dir}/test-to-formal-${d}.informal
            result=${output_dir}/${d}
            log=${output_dir}/eval-${d}.log
            arg=""
        else
            input=${input_dir}/test-to-formal-${d}-add-NLC_${CONSTRAINT}.informal
            result=${output_dir}/${d}-NLC_${CONSTRAINT}
            log=${output_dir}/eval-${d}-NLC_${CONSTRAINT}.log
            arg="--constraints"
            paste -d "" ${input_dir}/test-to-formal-${d}.informal ${DATASETS_DIR}/GYAFC/${DOMAIN}/bpe${BPE_TOKENS}/NLC_${CONSTRAINT}-${d}.informal > $input
        fi

        CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-interactive $preprocessed_dir \
            --input $input \
            --path $model \
            --buffer-size 1024 --batch-size 128 \
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
        $BLEU ${reference_dir}/test-to-formal.orig.formal0 \
            ${reference_dir}/test-to-formal.orig.formal1 \
            ${reference_dir}/test-to-formal.orig.formal2 \
            ${reference_dir}/test-to-formal.orig.formal3 \
            < ${result}.detok.sys  2>&1 | tee -a $log 
    done
else
    reference_dir=${DATASETS_DIR}/GYAFC/${DOMAIN}/tok
    if [ $CONSTRAINT == 0 ]; then
        input=${input_dir}/test-to-formal.informal
        result=${output_dir}/result
        log=${output_dir}/eval.log
        arg=""
    else
        input=${input_dir}/test-to-formal-add-NLC_${CONSTRAINT}.informal
        result=${output_dir}/result-NLC_${CONSTRAINT}
        log=${output_dir}/eval-NLC_${CONSTRAINT}.log
        arg="--constraints"
        paste -d "" ${input_dir}/test-to-formal.informal ${DATASETS_DIR}/GYAFC/${DOMAIN}/bpe${BPE_TOKENS}/NLC_${CONSTRAINT}.informal > $input
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-interactive $preprocessed_dir \
        --input $input \
        --path $model \
        --buffer-size 1024 --batch-size 128 \
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
    $BLEU ${reference_dir}/test-to-formal.orig.formal0 \
        ${reference_dir}/test-to-formal.orig.formal1 \
        ${reference_dir}/test-to-formal.orig.formal2 \
        ${reference_dir}/test-to-formal.orig.formal3 \
        < ${result}.detok.sys  2>&1 | tee -a $log 
fi
