#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
RESULT_DIR=../../../results/Simplification-Ja
UTILS_DIR=../../../utils

BPE_TOKENS=8000
BART_SCALE=base
EXP_NAME=RNN
GPU_ID=0
while getopts b:d:g:hln:r:u: OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-d DATASETS_DIR] [-g GPU_ID] [-l BART_SCALE=large] [-n EXP_NAME] [-p PRE_TRAINED_DIR] [-r RESULT_DIR] [-u UTILS_DIR]" 1>&2
        exit 1
        ;;
    l ) BART_SCALE=large
        ;;
    n ) EXP_NAME=$OPTARG
        ;;
    r ) RESULT_DIR=$OPTARG
        ;;
    u ) UTILS_DIR=$OPTARG
        ;;
    esac
done


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


output_dir=${RESULT_DIR}/${EXP_NAME}/select-model
mkdir -p $output_dir

reference_dir=${DATASETS_DIR}/SNOW/tok
references=${reference_dir}/valid.simple

if [ $BPE_TOKENS == "BART" ]; then
    preprocessed_dir=${DATASETS_DIR}/SNOW/bpeBART-${BART_SCALE}/fairseq-preprocess
    gen_args="--task translation_from_pretrained_bart --prepend-bos"
else
    preprocessed_dir=${DATASETS_DIR}/SNOW/bpe${BPE_TOKENS}/fairseq-preprocess
    gen_args=""
fi

result=${output_dir}/result
log=${output_dir}/select_model.log

for model_name in ${RESULT_DIR}/${EXP_NAME}/checkpoints/*.pt; do
    echo model name : `basename $model_name` >> $log
    CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-generate $preprocessed_dir \
        --gen-subset valid -s complex -t simple \
        --path $model_name \
        --batch-size 64 \
        --nbest 1 --beam 5 --lenpen 1.0 \
        --max-len-b 50 --remove-bpe "@@ " $gen_args \
        > ${result}.txt
    grep ^H ${result}.txt | cut -c 3- | sort -n | cut -f 3- > ${result}.sys

    if [ $BPE_TOKENS == "BART" ]; then
        cp ${result}.sys ${result}.sys.tmp
        cat ${result}.sys.tmp | sed 's/<<unk>>/<unk>/g' | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^[ \t]*//g' > ${result}.sys
    fi

    easse evaluate -t custom -m 'bleu,sari' \
        --refs_sents_paths $references \
        --orig_sents_path ${reference_dir}/valid.complex \
        --sys_sents_path ${result}.sys \
        --tokenizer none \
        2>&1 | tee -a $log
done
