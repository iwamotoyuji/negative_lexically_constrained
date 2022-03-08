#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
RESULT_DIR=../../../results/Simplification-Ja
UTILS_DIR=../../../utils

BPE_TOKENS=8000
BART_SCALE=base
CONSTRAINT=0
EXP_NAME=RNN
GPU_ID=0
MODEL=checkpoint_best.pt
MODE=test
while getopts b:c:d:g:hlm:n:p:r:tu:v OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    c ) CONSTRAINT=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-c CONSTRAINT] [-d DATASETS_DIR] [-g GPU_ID] [-l BART_SCALE=large] [-m MODEL] [-n EXP_NAME] [-r RESULT_DIR] [-u UTILS_DIR] [-v MODE=valid]" 1>&2
        exit 1
        ;;
    l ) BART_SCALE=large
        ;;
    m ) MODEL=$OPTARG
        ;;
    n ) EXP_NAME=$OPTARG
        ;;
    r ) RESULT_DIR=$OPTARG
        ;;      
    u ) UTILS_DIR=$OPTARG
        ;;
    v ) MODE=valid
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


if [ $EXP_NAME == "RNN_ensemble" ]; then
    output_dir=${RESULT_DIR}/RNN_ensemble/${MODE}
    model=${RESULT_DIR}/RNN_SEED11/checkpoints/checkpoint_45_12000.pt:${RESULT_DIR}/RNN_SEED22/checkpoints/checkpoint_45_12000.pt:${RESULT_DIR}/RNN_SEED33/checkpoints/checkpoint_37_10000.pt:${RESULT_DIR}/RNN_SEED44/checkpoints/checkpoint_56_15000.pt
elif [ $EXP_NAME == "SAN_ensemble" ]; then
    output_dir=${RESULT_DIR}/SAN_ensemble/${MODE}
    model=${RESULT_DIR}/SAN_SEED11/checkpoints/checkpoint_26_7000.pt:${RESULT_DIR}/SAN_SEED22/checkpoints/checkpoint_23_6000.pt:${RESULT_DIR}/SAN_SEED33/checkpoints/checkpoint_41_11000.pt:${RESULT_DIR}/SAN_SEED44/checkpoints/checkpoint_85_23000.pt
elif [ $EXP_NAME == "BART_base_ensemble" ]; then
    output_dir=${RESULT_DIR}/BART_base_ensemble/${MODE}
    model=${RESULT_DIR}/BART_base_SEED11/checkpoints/checkpoint_44_13000.pt:${RESULT_DIR}/BART_base_SEED22/checkpoints/checkpoint_51_15000.pt:${RESULT_DIR}/BART_base_SEED33/checkpoints/checkpoint_34_10000.pt:${RESULT_DIR}/BART_base_SEED44/checkpoints/checkpoint_31_9000.pt
elif [ $EXP_NAME == "BART_large_ensemble" ]; then
    output_dir=${RESULT_DIR}/BART_large_ensemble/${MODE}
    model=${RESULT_DIR}/BART_large_SEED11/checkpoints/checkpoint_24_7000.pt:${RESULT_DIR}/BART_large_SEED22/checkpoints/checkpoint_24_7000.pt:${RESULT_DIR}/BART_large_SEED33/checkpoints/checkpoint_21_6000.pt:${RESULT_DIR}/BART_large_SEED44/checkpoints/checkpoint_31_9000.pt
else
    output_dir=${RESULT_DIR}/${EXP_NAME}/${MODE}
    model=${RESULT_DIR}/${EXP_NAME}/checkpoints/${MODEL}
fi

mkdir -p $output_dir

reference_dir=${DATASETS_DIR}/SNOW/tok
if [ $MODE == "valid" ]; then
    references=${reference_dir}/${MODE}.simple
else
    references=${reference_dir}/${MODE}.simple.0,${reference_dir}/${MODE}.simple.1,${reference_dir}/${MODE}.simple.2,${reference_dir}/${MODE}.simple.3,${reference_dir}/${MODE}.simple.4,${reference_dir}/${MODE}.simple.5,${reference_dir}/${MODE}.simple.6
fi

if [ $BPE_TOKENS == "BART" ]; then
    input_dir=${DATASETS_DIR}/SNOW/bpeBART-${BART_SCALE}
    preprocessed_dir=${input_dir}/fairseq-preprocess
else
    input_dir=${DATASETS_DIR}/SNOW/bpe${BPE_TOKENS}
    preprocessed_dir=${input_dir}/fairseq-preprocess
fi

if [ $CONSTRAINT == 0 ]; then
    input=${input_dir}/${MODE}.complex
    result=${output_dir}/result
    log=${output_dir}/eval.log
    nlc_args=""
else
    input=${input_dir}/${MODE}-add-NLC_${CONSTRAINT}.complex
    result=${output_dir}/result-NLC_${CONSTRAINT}
    log=${output_dir}/eval-NLC_${CONSTRAINT}.log
    nlc_args="--constraints"
    paste -d "" ${input_dir}/${MODE}.complex ${input_dir}/NLC_${CONSTRAINT}-for-${MODE}.complex > $input
fi

if [ $BPE_TOKENS == "BART" ]; then
    tagged_input=${input}.tagged
    sed "s/^/<s> /g" $input > $tagged_input
    input=$tagged_input
fi

CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-interactive $preprocessed_dir \
    --input $input \
    --path $model \
    --batch-size 64 --buffer-size 1024 \
    --nbest 1 --beam 5 --lenpen 1.0 \
    --max-len-b 50 --remove-bpe "@@ " $nlc_args \
    > ${result}.txt
grep ^H ${result}.txt | cut -f 3- > ${result}.sys

if [ $BPE_TOKENS == "BART" ]; then
    cp ${result}.sys ${result}.sys.tmp
    cat ${result}.sys.tmp | sed 's/<<unk>>/<unk>/g' | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^[ \t]*//g' > ${result}.sys
fi

echo model name : $MODEL >> $log
easse evaluate -t custom -m 'bleu,sari' \
    --refs_sents_paths $references \
    --orig_sents_path ${reference_dir}/${MODE}.complex \
    --sys_sents_path ${result}.sys \
    --tokenizer none \
    2>&1 | tee -a $log
