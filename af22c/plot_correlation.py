import argparse
from af22c.proteome import Proteome, ProteomePLDDTs, ProteomeCorrelation, ProteomeSETHPreds

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('proteome_dir')
    parser.add_argument('data_dir')
    parser.add_argument('plddts_path')
    parser.add_argument('seth_path')
    args = parser.parse_args()

    proteome = Proteome.from_folder(args.proteome_dir, args.data_dir)
    plddts = ProteomePLDDTs.from_file(plDDT_path=args.plddts_path)
    seths = ProteomeSETHPreds.from_file(SETH_preds_path=args.seth_path)
    correlation = ProteomeCorrelation(proteome, plddts, seths)

    print(correlation.plot_mean_pearson_corr_mat(min_q_len=10, max_q_len=5000, min_n_seq=100, max_n_seq=5000))