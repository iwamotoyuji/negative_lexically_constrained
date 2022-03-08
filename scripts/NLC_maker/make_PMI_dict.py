import argparse
import math
import pickle
#from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_file_path')
    parser.add_argument('tgt_file_path')
    parser.add_argument('output_file_path')
    parser.add_argument('--ignore_word_cnt', type=int, default=3)
    args = parser.parse_args()

    return args


def main():
    opt = parse_args()

    word2cnt = {}
    all_word_cnt = 0
    src_word2cnt = {}
    src_all_word_cnt = 0
    with open(opt.src_file_path) as f:
        for sent in f:
            #words = sent.strip().lower().split(' ')
            words = sent.strip().split(' ')
            for word in words:
                if word in word2cnt:
                    word2cnt[word] += 1
                    src_word2cnt[word] += 1
                else:
                    word2cnt[word] = 1
                    src_word2cnt[word] = 1
                all_word_cnt += 1
                src_all_word_cnt += 1

    with open(opt.tgt_file_path) as f:
        for sent in f:
            #words = sent.strip().lower().split(' ')
            words = sent.strip().split(' ')
            for word in words:
                if word in word2cnt:
                    word2cnt[word] += 1
                else:
                    word2cnt[word] = 1
                all_word_cnt += 1

    PMI_dict = {}
    for key, value in src_word2cnt.items():
        if word2cnt[key] > opt.ignore_word_cnt:
            p_word = word2cnt[key] / all_word_cnt
            p_word_given_src = value / src_all_word_cnt
            PMI_dict[key] = math.log(p_word_given_src / p_word)

    with open(opt.output_file_path, mode='wb') as f:
        pickle.dump(PMI_dict, f)
        print(f"[Info] Dumped PMT dict to {opt.output_file_path}")


if __name__ == "__main__":
    main()