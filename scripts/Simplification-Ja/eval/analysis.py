import sys
import argparse
from re import compile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_dir', default="../../../datasets")
    parser.add_argument('--result_dir', default="../../../results/Simplification-Ja")
    parser.add_argument('--bpe_tokens', default="8000")
    parser.add_argument('--easse_dir', default="~/utils/easse")
    parser.add_argument('-e', '--exp_name', default="RNN_SEED11")
    parser.add_argument('-c', '--constraint', type=float, default=0.5)
    args = parser.parse_args()

    return args


def main():
    opt = parse_args()
    sys.path.append(opt.easse_dir)
    from easse.sari import corpus_sari

    bpe_decode = compile("@@ |@@ ?$")

    output_dir = f"{opt.result_dir}/{opt.exp_name}/test"
    reference_dir = f"{opt.datasets_dir}/SNOW/tok"
    nlc = f"{opt.datasets_dir}/SNOW/bpe{opt.bpe_tokens}/NLC_{opt.constraint}-for-test.complex"
    base_sys_out = f"{output_dir}/result.sys"
    nlc_sys_out = f"{output_dir}/result-NLC_{opt.constraint}.sys"

    analysis_results = ["base_sys_out\tNLC_sys_out\tbase_SARI\tNLC_SARI\tSARI_diff\tconstraints\toriginal\treference0\treference1\treference2\treference3\treference4\treference5\treference6\n"]
    with open(base_sys_out) as base_sys_f, \
         open(nlc_sys_out) as nlc_sys_f, \
         open(nlc) as nlc_f, \
         open(f"{reference_dir}/test.complex") as orig_f, \
         open(f"{reference_dir}/test.simple.0") as ref0_f, \
         open(f"{reference_dir}/test.simple.1") as ref1_f, \
         open(f"{reference_dir}/test.simple.2") as ref2_f, \
         open(f"{reference_dir}/test.simple.3") as ref3_f, \
         open(f"{reference_dir}/test.simple.4") as ref4_f, \
         open(f"{reference_dir}/test.simple.5") as ref5_f, \
         open(f"{reference_dir}/test.simple.6") as ref6_f:
        for lines in zip(base_sys_f, nlc_sys_f, nlc_f, orig_f, ref0_f, ref1_f, ref2_f, ref3_f, ref4_f, ref5_f, ref6_f):
            base_sys_line, nlc_sys_line, nlc_line, orig_line, ref0_line, ref1_line, ref2_line, ref3_line, ref4_line, ref5_line, ref6_line \
                = (line.strip() for line in lines)

            nlc_line = bpe_decode.sub('', nlc_line)
            nlc_line = nlc_line.replace('##', '').replace('\t', ',')
            if nlc_line == '':
                nlc_line = 'None'

            base_sari = corpus_sari(
                orig_sents=[orig_line],
                sys_sents=[base_sys_line],
                refs_sents=[
                    [ref0_line], [ref1_line], [ref2_line], [ref3_line], [ref4_line], [ref5_line], [ref6_line]
                ]
            )
            nlc_sari = corpus_sari(
                orig_sents=[orig_line],
                sys_sents=[nlc_sys_line],
                refs_sents=[
                    [ref0_line], [ref1_line], [ref2_line], [ref3_line], [ref4_line], [ref5_line], [ref6_line]
                ]
            )
        
            sari_diff = nlc_sari - base_sari
            result = f"{base_sys_line}\t{nlc_sys_line}\t{base_sari}\t{nlc_sari}\t{sari_diff}\t{nlc_line}\t{orig_line}\t{ref0_line}\t{ref1_line}\t{ref2_line}\t{ref3_line}\t{ref4_line}\t{ref5_line}\t{ref6_line}\n"
            analysis_results.append(result)

    output_file_name = f"{output_dir}/analysis.tsv"
    with open(output_file_name, 'w') as output_f:
        output_f.writelines(analysis_results)
    print(f"[Info] dumped tsv file to {output_file_name}")


if __name__ == "__main__":
    main()