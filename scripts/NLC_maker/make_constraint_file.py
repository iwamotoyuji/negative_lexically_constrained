import argparse
import pickle
from re import compile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_bpe_file_path')
    parser.add_argument('output_file_path')
    parser.add_argument('PMI_dict_path')
    parser.add_argument('--theta', type=float, default=0.5)
    parser.add_argument('--sentencepiece', action='store_true')
    args = parser.parse_args()

    return args


def main():
    opt = parse_args()

    with open(opt.PMI_dict_path, 'rb') as f:
        PMI_dict = pickle.load(f)

    is_bpe = compile("^(?!▁).+$") if opt.sentencepiece else compile(".*?@@$")
    bpe_decode = compile(" ") if opt.sentencepiece else compile("@@ |@@ ?$")
    check_next_word = 1 if opt.sentencepiece else 0
    constrain_text_list = []
    with open(opt.input_bpe_file_path) as f:
        for sent in f:
            words = sent.strip().split()
            constrain_words = []
            i = 0
            while i < len(words):
                if (i + check_next_word) < len(words) and is_bpe.match(words[i + check_next_word]):
                    bpe_word = words[i]
                    while True:
                        i += 1
                        bpe_word += f" {words[i]}"
                        if not ((i + check_next_word) < len(words) and is_bpe.match(words[i + check_next_word])):
                            break
                    orig_word = bpe_decode.sub('', bpe_word)
                else:
                    bpe_word = words[i]
                    orig_word = words[i]

                if opt.sentencepiece:
                    assert orig_word[0] == '▁'
                    orig_word = orig_word[1:]
                #if orig_word.lower() in PMI_dict and bpe_word not in constrain_words and PMI_dict[orig_word.lower()] > opt.theta:
                if orig_word in PMI_dict and bpe_word not in constrain_words and PMI_dict[orig_word] > opt.theta:
                    constrain_words.append(bpe_word)
                i += 1
                
            if constrain_words:
                constrain_text = '\t##' + '\t'.join(constrain_words)
            else:
                constrain_text = ""
            constrain_text_list.append(constrain_text)

    with open(opt.output_file_path, 'w') as f:
        f.write('\n'.join(constrain_text_list))
    print(f"[Info] Dumped negative constrained inputs to {opt.output_file_path}")


if __name__ == "__main__":
    main()