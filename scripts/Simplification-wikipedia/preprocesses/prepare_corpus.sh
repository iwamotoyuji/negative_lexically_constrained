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

mkdir -p $RESOURCE_DIR
pushd $RESOURCE_DIR
# --- Download wiki-auto ---
if [ -d ./wiki-auto ]; then
    echo "[Info] wiki-auto git already exists, skipping download"
else
    echo "[Info] Cloning simplification git..."
    git clone https://github.com/chaojiang06/wiki-auto.git
fi
pushd ./wiki-auto/wiki-auto/all_data
if [ -f ./wiki-auto-part-1-data.json ]; then
    echo "[Info] wiki-auto-part1 data already exists, skipping download"
else
    echo "[Info] Downloading wiki-auto-part1 data..."
    wget https://www.dropbox.com/sh/ohqaw41v48c7e5p/AAATBDhU1zpdcT5x5WgO8DMaa/wiki-auto-all-data/wiki-auto-part-1-data.json
fi
if [ -f ./wiki-auto-part-2-data.json ]; then
    echo "[Info] wiki-auto-part2 data already exists, skipping download"
else
    echo "[Info] Downloading wiki-auto-part2 data..."
    wget https://www.dropbox.com/sh/ohqaw41v48c7e5p/AAATgPkjo_tPt9z12vZxJ3MRa/wiki-auto-all-data/wiki-auto-part-2-data.json
fi
popd

# --- Download wikilarge ---
if [ -d ./data-simplification ]; then
    echo "[Info] wikilarge data already exists, skipping download"
else
    echo "[Info] Downloading wikilarge data..."
    wget https://github.com/louismartin/dress-data/raw/master/data-simplification.tar.bz2
    tar -jxvf data-simplification.tar.bz2
fi

# --- Download ASSET ---
if [ -d ./asset ]; then
    echo "[Info] asset git already exists, skipping download"
else
    echo "[Info] Cloning ASSET git..."
    git clone https://github.com/facebookresearch/asset.git
fi

# --- Download turkcorpus ---
if [ -d ./simplification ]; then
    echo "[Info] simplification git already exists, skipping download"
else
    echo "[Info] Cloning simplification git..."
    git clone https://github.com/cocoxu/simplification.git
fi
popd

# --- Preprocess wiki-auto ---
echo "[Info] Tokenizing wiki-auto train data"
orig_path=${RESOURCE_DIR}/wiki-auto/wiki-auto/ACL2020
tok_path=${DATASETS_DIR}/wiki-auto/tok
mkdir -p $tok_path

cat ${orig_path}/train.src | \
    $NORMALIZER -l en | \
    $TOKENIZER -l en -a -no-escape -threads 8 -lines 100000 > ${tok_path}/train.complex
cat ${orig_path}/train.dst | \
    $NORMALIZER -l en | \
    $TOKENIZER -l en -a -no-escape -threads 8 -lines 100000 > ${tok_path}/train.simple

# --- Preprocess wikilarge ---
echo "[Info] Tokenizing wikilarge train data..."
orig_path=${RESOURCE_DIR}/data-simplification/wikilarge
tok_path=${DATASETS_DIR}/wiki-large/tok
mkdir -p $tok_path

cat ${orig_path}/wiki.full.aner.ori.train.src | \
    $NORMALIZER -l en | \
	$TOKENIZER -l en -a -no-escape -threads 8 -lines 100000 > ${tok_path}/train.complex
cat ${orig_path}/wiki.full.aner.ori.train.dst | \
    $NORMALIZER -l en | \
	$TOKENIZER -l en -a -no-escape -threads 8 -lines 100000 > ${tok_path}/train.simple

# --- Preprocess ASSET ---
echo "[Info] Tokenizing ASSET valid data..."
orig_path=${RESOURCE_DIR}/asset/dataset
tok_path=${DATASETS_DIR}/asset/tok
mkdir -p $tok_path

cat ${orig_path}/asset.valid.orig > ${tok_path}/valid.orig.complex
cat ${orig_path}/asset.valid.orig | \
    $NORMALIZER -l en | \
	$TOKENIZER -l en -a -no-escape -threads 8 -lines 100000 > ${tok_path}/valid.complex
for i in $(seq 0 9); do
    cat ${orig_path}/asset.valid.simp.${i} > ${tok_path}/valid.orig.simple.${i}
done

rm -f ${tok_path}/valid-all.complex
rm -f ${tok_path}/valid-all.simple
for i in $(seq 0 9); do
    cat ${tok_path}/valid.complex >> ${tok_path}/valid-all.complex
    cat ${orig_path}/asset.valid.simp.${i} | \
        $NORMALIZER -l en | \
        $TOKENIZER -l en -a -no-escape -threads 8 -lines 100000 >> ${tok_path}/valid-all.simple
done

echo "[Info] Tokenizing ASSET test data..."
cat ${orig_path}/asset.test.orig > ${tok_path}/test.orig.complex
cat ${orig_path}/asset.test.orig | \
    $NORMALIZER -l en | \
    $TOKENIZER -l en -a -no-escape -threads 8 -lines 100000 > ${tok_path}/test.complex
for i in $(seq 0 9); do
    cat ${orig_path}/asset.test.simp.${i} > ${tok_path}/test.orig.simple.${i}
done

# --- Preprocess turkcorpus ---
echo "[Info] Tokenizing turkcorpus valid data..."
orig_path=${RESOURCE_DIR}/simplification/data/turkcorpus
tok_path=${DATASETS_DIR}/turkcorpus/tok
mkdir -p $tok_path

cat ${orig_path}/tune.8turkers.tok.norm | perl -C $DETOKENIZER -l en > ${tok_path}/valid.orig.complex
cat ${orig_path}/tune.8turkers.tok.norm > ${tok_path}/valid.complex
for i in $(seq 0 7); do
    cat ${orig_path}/tune.8turkers.tok.turk.${i} | perl -C $DETOKENIZER -l en > ${tok_path}/valid.orig.simple.${i}
done

rm -f ${tok_path}/valid-all.complex
rm -f ${tok_path}/valid-all.simple
for i in $(seq 0 7); do
    cat ${orig_path}/tune.8turkers.tok.norm >> ${tok_path}/valid-all.complex
    cat ${orig_path}/tune.8turkers.tok.turk.${i} >> ${tok_path}/valid-all.simple
done

echo "[Info] Tokenizing turkcorpus test data..."
cat ${orig_path}/test.8turkers.tok.norm | perl -C $DETOKENIZER -l en > ${tok_path}/test.orig.complex
cat ${orig_path}/test.8turkers.tok.norm > ${tok_path}/test.complex
for i in $(seq 0 7); do
    cat ${orig_path}/test.8turkers.tok.turk.${i} | perl -C $DETOKENIZER -l en > ${tok_path}/test.orig.simple.${i}
done
