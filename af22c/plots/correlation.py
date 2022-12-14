#!/usr/bin/env python

import argparse
from af22c.proteome import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory to put the plot")
    parser.add_argument("neff_dir")
    parser.add_argument("neff_naive_dir")
    parser.add_argument("neff_hhsuite_dir")
    parser.add_argument("plddts_path")
    parser.add_argument("seth_path")
    parser.add_argument("maxzs_path")
    parser.add_argument("msa_sizes_dir")
    args = parser.parse_args()

    neffs = ProteomeNeffs.from_directory(args.neff_dir)
    neffs_naive = ProteomeNeffsNaive.from_directory(args.neff_naive_dir)
    neffs_hhsuite = ProteomeNeffsHHsuite.from_directory(args.neff_hhsuite_dir)
    plddts = ProteomePLDDTs.from_file(args.plddts_path)
    seths = ProteomeSETHPreds.from_file(args.seth_path)
    maxzs = ProteomeMaxZs.from_directory(args.maxzs_path)
    msa_sizes = ProteomeMSASizes.from_file(args.msa_sizes_dir)
    correlation = ProteomeCorrelation([plddts, seths, neffs, neffs_naive, neffs_hhsuite, maxzs], msa_sizes)

    correlation.plot_mean_pearson_corr_mat(
        args.data_dir,
        "HUMAN",
        min_q_len=10,
        max_q_len=300,
        min_n_seq=100,
        max_n_seq=2000,
    )
