import argparse
import pickle
from re import compile
from fairseq.data.encoders.gpt2_bpe import get_encoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_path')
    parser.add_argument('output_file_path')
    parser.add_argument('PMI_dict_path')
    parser.add_argument('--encoder-json', help="path to encoder.json")
    parser.add_argument('--vocab-bpe', type=str, help="path to vocab.bpe")
    parser.add_argument('--theta', type=float, default=0.5)
    args = parser.parse_args()

    return args


def main():
    opt = parse_args()
    encoder = get_encoder(opt.encoder_json, opt.vocab_bpe)

    with open(opt.PMI_dict_path, 'rb') as f:
        PMI_dict = pickle.load(f)

    constrain_text_list = []
    with open(opt.input_file_path) as f:
        for sent in f:
            words = sent.strip().split()
            constrain_words = []
            for orig_word in words:
                if orig_word in PMI_dict and PMI_dict[orig_word] > opt.theta:
                    # BART outputs " word" (space + word) as well as "word", so add both to the constraints.
                    bpe_ids = encoder.encode(orig_word)
                    space_bpe_ids = encoder.encode(' ' + orig_word)
                    bpe_word = ' '.join(list(map(str, bpe_ids)))
                    space_bpe_word = ' '.join(list(map(str, space_bpe_ids)))
                    if bpe_word not in constrain_words:
                       constrain_words.append(bpe_word)
                    if space_bpe_ids not in constrain_words:
                        constrain_words.append(space_bpe_word)
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