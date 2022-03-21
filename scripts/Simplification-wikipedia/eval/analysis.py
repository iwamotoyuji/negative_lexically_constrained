import sys
import argparse
from re import compile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_dir', default="../../../datasets")
    parser.add_argument('--result_dir', default="../../../results/Simplification-wikipedia")
    parser.add_argument('--bpe_tokens', default="16000")
    parser.add_argument('--easse_dir', default="~/utils/easse")
    parser.add_argument('-e', '--exp_name', default="RNN_SEED11")
    parser.add_argument('-c', '--constraint', type=float, default=0.1)
    parser.add_argument('-t', '--turkcorpus', action='store_true')
    args = parser.parse_args()

    return args


def main():
    opt = parse_args()
    sys.path.append(opt.easse_dir)
    from easse.sari import corpus_sari

    if opt.turkcorpus:
        test_set = "turkcorpus"
        num_ref = 8
    else:
        test_set = "asset"
        num_ref = 10

    bpe_decode = compile("@@ |@@ ?$")

    output_dir = f"{opt.result_dir}/wiki-auto/{opt.exp_name}/test"
    reference_dir = f"{opt.datasets_dir}/{test_set}/tok"
    nlc = f"{opt.datasets_dir}/{test_set}/bpe{opt.bpe_tokens}-from-wiki-auto/NLC_{opt.constraint}-for-test.complex"
    base_sys_out = f"{output_dir}/{test_set}.detok.sys"
    nlc_sys_out = f"{output_dir}/{test_set}-NLC_{opt.constraint}.detok.sys"

    with open(base_sys_out) as base_sys_f:
        base_sys_lines = base_sys_f.readlines()
    with open(nlc_sys_out) as nlc_sys_f:
        nlc_sys_lines = nlc_sys_f.readlines()
    with open(nlc) as nlc_f:
        nlc_lines = nlc_f.readlines()
    with open(f"{reference_dir}/test.orig.complex") as orig_f:
        orig_lines = orig_f.readlines()
    
    tsv_head = "base_sys_out\tNLC_sys_out\tbase_SARI\tNLC_SARI\tSARI_diff\tconstraints\toriginal"
    ref_lines_list = []
    ref_line_list = []
    for i in range(num_ref):
        tsv_head += f"\treference{i}"
        ref_line_list.append(None)
        with open(f"{reference_dir}/test.orig.simple.{i}") as ref_f:
            ref_lines = ref_f.readlines()
            ref_lines_list.append(ref_lines)
    tsv_head += '\n'

    analysis_results = [tsv_head]
    num_lines = len(base_sys_lines)
    for line_id in range(num_lines):
        base_sys_line = base_sys_lines[line_id].strip()
        nlc_sys_line = nlc_sys_lines[line_id].strip()
        orig_line = orig_lines[line_id].strip()
        nlc_line = nlc_lines[line_id].strip()
        nlc_line = bpe_decode.sub('', nlc_line)
        nlc_line = nlc_line.replace('##', '').replace('\t', ',')
        if nlc_line == '':
            nlc_line = 'None'        

        for i in range(num_ref):
            ref_line_list[i] = ref_lines_list[i][line_id].strip()

        base_sari = corpus_sari(
            orig_sents=[orig_line],
            sys_sents=[base_sys_line],
            refs_sents=[[ref_line_list[i]] for i in range(num_ref)],
        )
        nlc_sari = corpus_sari(
            orig_sents=[orig_line],
            sys_sents=[nlc_sys_line],
            refs_sents=[[ref_line_list[i]] for i in range(num_ref)],
        )
        
        sari_diff = nlc_sari - base_sari
        result = f"{base_sys_line}\t{nlc_sys_line}\t{base_sari}\t{nlc_sari}\t{sari_diff}\t{nlc_line}\t{orig_line}"
        for i in range(num_ref):
            result += f"\t{ref_line_list[i]}"
        result += '\n'
        analysis_results.append(result)

    output_file_name = f"{output_dir}/analysis-{test_set}.tsv"
    with open(output_file_name, 'w') as output_f:
        output_f.writelines(analysis_results)
    print(f"[Info] dumped tsv file to {output_file_name}")


if __name__ == "__main__":
    main()