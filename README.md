# 概要
- スタイル変換において，出力文生成時に特定の単語に対し負の制約をかけることで，その単語を出力しないようにする．
- 入力文コーパスと出力文コーパスに対して，PMI を用いることで，負の制約をかける単語を決定する．
- Formality Transfer (En)，Simplification (En)，SImplification (Ja) において，RNN, SAN, BART の3つのモデルで実験し，負の語彙制約の有効性を調査

# 導入
- fairseq を https://github.com/pytorch/fairseq/pull/2958 を参考にして改造し，ビームサーチに負の語彙制約を適用できるようにしています．公式の fairseq では動作しません．
``` bash
git clone https://github.com/iwamotoyuji/fairseq-nlc.git -b nlc
cd fairseq-nlc
pip install --editable ./
cd ..
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
            --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
            --global-option="--fast_multihead_attn" ./
```

# Formality Transfer
## データの準備
- データセットは GYAFC の Entertainment_Music と Family_Relationships の両方を用います．
- tokenizer fastBPE 等は自動でダウンロードされます．
``` bash
pushd scripts/FT/preprocesses
bash prepare_corpus.sh -r ${GYAFCがあるフォルダ}
# RNN および SAN
bash apply_bpe.sh
bash insert_tags_for_bidirectional.sh
# BART
bash apply_bpe.sh -b BART
bash insert_tags_for_bidirectional.sh -b BART
popd
```

## モデルの訓練
``` bash
pushd scripts/FT/train
# RNN
bash fairseq_preprocess.sh
bash train-RNN.sh -n ${実験名} -g ${GPUのID} -s ${SEED値}
# SAN
bash fairseq_preprocess.sh
bash train-SAN.sh -n ${実験名} -g ${GPUのID} -s ${SEED値}
# BART-base
bash fairseq_preprocess.sh -b BART
bash finetune-BART.sh -n ${実験名} -g ${GPUのID} -s ${SEED値}
# BART-large
bash fairseq_preprocess.sh -b BART -l large
bash finetune-BART.sh -n ${実験名} -g ${GPUのID} -s ${SEED値} -l large
popd
```

## モデルの選択
- 使用するモデルは validation データにおける BLEU が最大のモデル (checkpoint_best.pt) を使用します．

## 負の語彙制約の生成
- 閾値 Θ 以上の PMI を有する単語を制約対象とします．
- Formality Transfer では先行研究に倣い，閾値を 0.5 とする．
``` bash
pushd scripts/FT/preprocesses
# RNN および SAN
bash prepare_NLC.sh -c ${閾値 Θ}
# BART
bash prepare_NLC.sh -c ${閾値 Θ} -b BART
popd
```

## モデルの評価
``` bash
pushd scripts/FT/eval
# RNN および SAN
bash eval-multi-bleu.sh -n ${実験名} -g ${GPUのID}               # 制約なし
bash eval-multi-bleu.sh -n ${実験名} -g ${GPUのID} -c ${閾値 Θ}  # 制約あり
# BART-base
bash eval-multi-bleu.sh -n ${実験名} -g ${GPUのID} -b BART               # 制約なし
bash eval-multi-bleu.sh -n ${実験名} -g ${GPUのID} -c ${閾値 Θ} -b BART  # 制約あり
# BART-large
bash eval-multi-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -l large               # 制約なし
bash eval-multi-bleu.sh -n ${実験名} -g ${GPUのID} -c ${閾値 Θ} -b BART -l large  # 制約あり
popd
```

# Simplification
## データの準備
- 訓練データセットは https://github.com/chaojiang06/wiki-auto が自動でダウンロードされます．
- 評価データセットは https://github.com/facebookresearch/asset.git と https://github.com/cocoxu/simplification.git が自動でダウンロードされます．
- tokenizer fastBPE 等は自動でダウンロードされます．
``` bash
pushd scripts/Simplification-wikipedia/preprocesses
bash prepare_corpus.sh -r ${データセットを保存する(されている)フォルダ}
# RNN および SAN
bash apply_bpe.sh -a
# BART
bash apply_bpe.sh -b BART -a
popd
```

## モデルの訓練
``` bash
pushd scripts/Simplification-wikipedia/train
# RNN
bash fairseq_preprocess.sh -a
bash train-RNN.sh -n ${実験名} -g ${GPUのID} -s ${SEED値} -a
# SAN
bash fairseq_preprocess.sh -a
bash train-SAN.sh -n ${実験名} -g ${GPUのID} -s ${SEED値} -a
# BART-base
bash fairseq_preprocess.sh -b BART -a
bash finetune-BART.sh -n ${実験名} -g ${GPUのID} -s ${SEED値} -a
# BART-large
bash fairseq_preprocess.sh -b BART -l large -a
bash finetune-BART.sh -n ${実験名} -g ${GPUのID} -s ${SEED値} -l large -a
popd
```

## モデルの選択
- 使用するモデルは validation データにおける SARI が最大のモデルを使用します．
- fairseq-interactive を用いていますが，fairseq-generate を用いた方が高速なので，改造することを推奨．その場合，Simplification-Ja を参考にしてください．
- results/Simplification-wikipedia/wiki-auto/実験名/select-model/テストセット.log を見てスコアの高いモデルを選択します．
``` bash
pushd scripts/Simplification-wikipedia/eval
# RNN および SAN
bash select-model.sh -n ${実験名} -g ${GPUのID} -a     # asset
bash select-model.sh -n ${実験名} -g ${GPUのID} -a -t  # turkcorpus
# BART-base
bash select-model.sh -n ${実験名} -g ${GPUのID} -b BART -a     # asset
bash select-model.sh -n ${実験名} -g ${GPUのID} -b BART -a -t  # turkcorpus
# BART-large
bash select-model.sh -n ${実験名} -g ${GPUのID} -b BART -l -a     # asset
bash select-model.sh -n ${実験名} -g ${GPUのID} -b BART -l -a -t  # turkcorpus
popd
```

## 負の語彙制約の生成
- 閾値 Θ 以上の PMI を有する単語を制約対象とします．
- Simplification では，閾値 Θ を validation により決定します．
``` bash
pushd scripts/Simplification-wikipedia/preprocesses
# RNN および SAN
bash prepare_NLC.sh -c ${閾値 Θ} -a     # test用
bash prepare_NLC.sh -c ${閾値 Θ} -a -v  # validation用
# BART
bash prepare_NLC.sh -c ${閾値 Θ} -b BART -a     # test用
bash prepare_NLC.sh -c ${閾値 Θ} -b BART -a -v  # validation用
popd
```

## モデルの評価
- 閾値 Θ を決定するために validation を行う際は -v オプションをつけてください．
``` bash
pushd scripts/Simplification-wikipedia/eval
# RNN および SAN
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -a -m ${モデル名}                  # 制約なし asset
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -a -t -m ${モデル名}               # 制約なし turkcorpus
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -a -m ${モデル名} -c ${閾値 Θ}     # 制約あり asset
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -a -t -m ${モデル名} -c ${閾値 Θ}  # 制約あり turkcorpus
# BART-base
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -a -m ${モデル名}                  # 制約なし asset
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -a -t -m ${モデル名}               # 制約なし turkcorpus
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -a -m ${モデル名} -c ${閾値 Θ}     # 制約あり asset
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -a -t -m ${モデル名} -c ${閾値 Θ}  # 制約あり turkcorpus
# BART-large
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -l -a -m ${モデル名}                  # 制約なし asset
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -l -a -t -m ${モデル名}               # 制約なし turkcorpus
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -l -a -m ${モデル名} -c ${閾値 Θ}     # 制約あり asset
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -l -a -t -m ${モデル名} -c ${閾値 Θ}  # 制約あり turkcorpus
popd
```

# Simplification-Ja
## 導入
- 日本語の Simplification では [黒橋研が作成したfairseq](https://nlp.ist.i.kyoto-u.ac.jp/?BART%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB) を改造しています．公式の fairseq や上記で用いた fairseq では動作しません．
- 必要なライブラリも [ここ](https://github.com/utanaka2000/fairseq/blob/japanese_bart_pretrained_model/JAPANESE_BART_README.md) に書いてあるので，適宜インストールしてください．
``` bash
git clone https://github.com/iwamotoyuji/fairseq-nlc.git -b japanese_bart_pretrained_model_nlc fairseq-nlc-ja
cd fairseq-nlc-ja
pip install --editable ./
cd ..
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
            --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
            --global-option="--fast_multihead_attn" ./
```

## データの準備
- 訓練データセットは [SNOW T15 やさしい日本語コーパス](https://www.jnlp.org/GengoHouse/snow/t15) と [SNOW T23 やさしい日本語コーパス](https://www.jnlp.org/GengoHouse/snow/t23) を用います．
- tokenizer fastBPE 等は自動でダウンロードされます．
``` bash
pushd scripts/Simplification-Ja/preprocesses
# RNN, SAN
bash prepare_corpus.sh -r ${SNOW があるフォルダ}
bash apply_bpe.sh 
# BART-base
bash prepare_corpus.sh -r ${SNOW があるフォルダ}
# BART-large
bash bash prepare_corpus.sh -r ${SNOW があるフォルダ} -l
popd
```

## モデルの訓練
``` bash
pushd scripts/Simplification-Ja/train
# RNN
bash fairseq_preprocess.sh
bash train-RNN.sh -n ${実験名} -g ${GPUのID} -s ${SEED値} -a
# SAN
bash fairseq_preprocess.sh
bash train-SAN.sh -n ${実験名} -g ${GPUのID} -s ${SEED値} -a
# BART-base
bash fairseq_preprocess.sh -b BART
bash finetune-BART.sh -n ${実験名} -g ${GPUのID} -s ${SEED値} -a
# BART-large
bash fairseq_preprocess.sh -b BART -l large
bash finetune-BART.sh -n ${実験名} -g ${GPUのID} -s ${SEED値} -l large -a
popd
```

## モデルの選択
- 使用するモデルは validation データにおける SARI が最大のモデルを使用します．
- results/Simplification-wikipedia/wiki-auto/実験名/select-model/テストセット.log を見てスコアの高いモデルを選択する．
``` bash
pushd scripts/Simplification-Ja/eval
# RNN および SAN
bash select-model.sh -n ${実験名} -g ${GPUのID}
# BART-base
bash select-model.sh -n ${実験名} -g ${GPUのID} -b BART
# BART-large
bash select-model.sh -n ${実験名} -g ${GPUのID} -b BART -l
popd
```

## 負の語彙制約の生成
- 閾値 Θ 以上の PMI を有する単語を制約対象とします．
- Simplification では，閾値 Θ を validation により決定します．
``` bash
pushd scripts/Simplification-Ja/preprocesses
# RNN および SAN
bash prepare_NLC.sh -c ${閾値 Θ}     # test用
bash prepare_NLC.sh -c ${閾値 Θ} -v  # validation用
# BART-base
bash prepare_NLC.sh -c ${閾値 Θ} -b BART     # test用
bash prepare_NLC.sh -c ${閾値 Θ} -b BART -v  # validation用
# BART-large
bash prepare_NLC.sh -c ${閾値 Θ} -b BART -l     # test用
bash prepare_NLC.sh -c ${閾値 Θ} -b BART -l -v  # validation用
popd
```

## モデルの評価
- 閾値 Θ を決定するために validation を行う際は -v オプションをつけてください．
``` bash
pushd scripts/Simplification-Ja/eval
# RNN および SAN
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -m ${モデル名}               # 制約なし
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -m ${モデル名} -c ${閾値 Θ}  # 制約あり
# BART-base
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -m ${モデル名}               # 制約なし
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -m ${モデル名} -c ${閾値 Θ}  # 制約あり
# BART-large
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -l -m ${モデル名}               # 制約なし
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -l -m ${モデル名} -c ${閾値 Θ}  # 制約あり
popd
```