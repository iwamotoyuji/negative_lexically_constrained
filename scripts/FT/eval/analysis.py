import argparse
from re import compile
from typing import Sequence, Optional
from sacrebleu.metrics import BLEU, BLEUScore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_dir', default="../../../datasets")
    parser.add_argument('--result_dir', default="../../../results/FT")
    parser.add_argument('--bpe_tokens', default="16000")
    parser.add_argument('-e', '--exp_name', default="RNN_SEED11")
    parser.add_argument('-c', '--constraint', type=float, default=0.5)
    args = parser.parse_args()

    return args


def get_metric(smooth_method: str = 'exp',
               smooth_value: float = None,
               lowercase: bool = False,
               tokenize=BLEU.TOKENIZER_DEFAULT,
               use_effective_order: bool = True) -> BLEU:
    
    metric = BLEU(
        lowercase=lowercase, tokenize=tokenize, force=False,
        smooth_method=smooth_method, smooth_value=smooth_value,
        effective_order=use_effective_order)
    return metric

def sentence_bleu(hypothesis: str, references: Sequence[str], metric: BLEU) -> BLEUScore:
    return metric.sentence_score(hypothesis, references)


def main():
    opt = parse_args()

    metric = get_metric()
    bpe_decode = compile("@@ |@@ ?$")
    domain = "Combo"
    output_dir = f"{opt.result_dir}/{domain}/{opt.exp_name}"

    for d in ["Entertainment_Music", "Family_Relationships"]:
        reference_dir = f"{opt.datasets_dir}/GYAFC/{d}/tok"
        nlc = f"{opt.datasets_dir}/GYAFC/{d}/bpe{opt.bpe_tokens}/NLC_{opt.constraint}-{d}.informal"
        base_sys_out = f"{output_dir}/result-{d}.detok.sys"
        nlc_sys_out = f"{output_dir}/result-{d}-NLC_{opt.constraint}.detok.sys"

        analysis_results = ["base_sys_out\tNLC_sys_out\tbase_BLEU\tNLC_BLEU\tBLEU_diff\tconstraints\toriginal\treference0\treference1\treference2\treference3\n"]
        with open(base_sys_out) as base_sys_f, \
             open(nlc_sys_out) as nlc_sys_f, \
             open(nlc) as nlc_f, \
             open(f"{reference_dir}/test-to-formal.informal") as orig_f, \
             open(f"{reference_dir}/test-to-formal.orig.formal0") as ref0_f, \
             open(f"{reference_dir}/test-to-formal.orig.formal1") as ref1_f, \
             open(f"{reference_dir}/test-to-formal.orig.formal2") as ref2_f, \
             open(f"{reference_dir}/test-to-formal.orig.formal3") as ref3_f:
            for lines in zip(base_sys_f, nlc_sys_f, nlc_f, orig_f, ref0_f, ref1_f, ref2_f, ref3_f):
                base_sys_line, nlc_sys_line, nlc_line, orig_line, ref0_line, ref1_line, ref2_line, ref3_line = (line.strip() for line in lines)

                nlc_line = bpe_decode.sub('', nlc_line)
                nlc_line = nlc_line.replace('##', '').replace('\t', ',')
                if nlc_line == '':
                    nlc_line = 'None'

                base_bleu = sentence_bleu(base_sys_line, [ref0_line, ref1_line, ref2_line, ref3_line], metric)
                nlc_bleu = sentence_bleu(nlc_sys_line, [ref0_line, ref1_line, ref2_line, ref3_line], metric)
                bleu_diff = nlc_bleu.score - base_bleu.score
                analysis_results.append(f"{base_sys_line}\t{nlc_sys_line}\t{base_bleu.score}\t{nlc_bleu.score}\t{bleu_diff}\t{nlc_line}\t{orig_line}\t{ref0_line}\t{ref1_line}\t{ref2_line}\t{ref3_line}\n")
                
        output_file_name = f"{output_dir}/analysis-{d}.tsv"
        with open(output_file_name, 'w') as output_f:
            output_f.writelines(analysis_results)
        print(f"[Info] dumped tsv file to {output_file_name}")

if __name__ == "__main__":
    main()