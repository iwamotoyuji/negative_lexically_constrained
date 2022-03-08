#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../../datasets
RESOURCE_DIR=../../../resources
UTILS_DIR=../../../utils

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


combo_tok_path=${DATASETS_DIR}/GYAFC/Combo/tok
rm -rf $combo_tok_path
mkdir -p $combo_tok_path

for domain in Entertainment_Music Family_Relationships; do
    orig_path=${RESOURCE_DIR}/GYAFC_Corpus/${domain}
    tok_path=${DATASETS_DIR}/GYAFC/${domain}/tok
    tmp_path=${DATASETS_DIR}/GYAFC/${domain}/tmp
    mkdir -p $tok_path ${tmp_path}/train ${tmp_path}/tune ${tmp_path}/test

	# --- Tokenization ---
	for file in train/informal train/formal tune/informal tune/informal.ref0 tune/informal.ref1 tune/informal.ref2 tune/informal.ref3 tune/formal tune/formal.ref0 tune/formal.ref1 tune/formal.ref2 tune/formal.ref3 test/informal test/formal; do
		echo "[Info] Tokenizing ${orig_path}/${file} to ${tmp_path}/${file}..."
		cat ${orig_path}/${file} | \
            $NORMALIZER -l en | \
			$TOKENIZER -l en -a -no-escape -threads 8 -lines 100000 > ${tmp_path}/${file}
	done

    # --- train data ---
	for lang in informal formal; do
        cat ${tmp_path}/train/${lang} > ${tok_path}/train.${lang}
        cat ${tmp_path}/train/${lang} >> ${combo_tok_path}/train.${lang}
	done

    # --- dev test data ---
	for lang in informal formal; do
		if [ $lang = informal ]; then
		    tolang=formal
		elif [ $lang = formal ]; then
			tolang=informal
		fi
		cat ${tmp_path}/tune/${lang} ${tmp_path}/tune/${lang} ${tmp_path}/tune/${lang} ${tmp_path}/tune/${lang} > ${tok_path}/dev-to-${tolang}.${lang}
        cat ${tok_path}/dev-to-${tolang}.${lang} >> ${combo_tok_path}/dev-to-${tolang}.${lang}
		cat ${tmp_path}/tune/${tolang}.ref0 ${tmp_path}/tune/${tolang}.ref1 ${tmp_path}/tune/${tolang}.ref2 ${tmp_path}/tune/${tolang}.ref3 > ${tok_path}/dev-to-${tolang}.${tolang}
        cat ${tok_path}/dev-to-${tolang}.${tolang} >> ${combo_tok_path}/dev-to-${tolang}.${tolang}
        
        cat ${tmp_path}/test/${lang} > ${tok_path}/test-to-${tolang}.${lang}
        cat ${tmp_path}/test/${lang} >> ${combo_tok_path}/test-to-${tolang}.${lang}
		for i in $(seq 0 3); do
			cat ${orig_path}/test/${tolang}.ref${i} > ${tok_path}/test-to-${tolang}.orig.${tolang}${i}
            cat ${orig_path}/test/${tolang}.ref${i} >> ${combo_tok_path}/test-to-${tolang}.orig.${tolang}${i}
		done
	done
done

for lang in informal formal; do
    paste -d'\n' ${combo_tok_path}/test-to-${lang}.orig.${lang}0 ${combo_tok_path}/test-to-${lang}.orig.${lang}1 ${combo_tok_path}/test-to-${lang}.orig.${lang}2 ${combo_tok_path}/test-to-${lang}.orig.${lang}3 > ${combo_tok_path}/test-to-${lang}.orig.${lang}-all
done
