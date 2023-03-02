#!/usr/bin/env python3

from af22c.plots import *
from af22c.proteome import *


def plot(proteome_name, set_name, correlation):
    ax = plot_msa_sizes_with_correlation(correlation, aggregate="mean")
    ax.get_figure().savefig(f"data/corr_by_size/{proteome_name}_{set_name}_msa_sizes_with_correlation_mean.png")

    ax = plot_msa_sizes_with_correlation(correlation, aggregate="median")
    ax.get_figure().savefig(f"data/corr_by_size/{proteome_name}_{set_name}_msa_sizes_with_correlation_median.png")

    ax = plot_msa_sizes_with_correlation(correlation, aggregate="std")
    ax.get_figure().savefig(f"data/corr_by_size/{proteome_name}_{set_name}_msa_sizes_with_correlation_std.png")

    ax = plot_msa_sizes_with_correlation(correlation, aggregate="abs_mean")
    ax.get_figure().savefig(f"data/corr_by_size/{proteome_name}_{set_name}_msa_sizes_with_correlation_abs_mean.png")

    ax = plot_msa_sizes_with_correlation(correlation, aggregate="abs_median")
    ax.get_figure().savefig(f"data/corr_by_size/{proteome_name}_{set_name}_msa_sizes_with_correlation_abs_median.png")


if __name__ == "__main__":
    proteome_name = "UP000005640_9606"
    neffs = ProteomeNeffs.from_directory(f"data/cluster/{proteome_name}/neffs")
    neffs_naive = ProteomeNeffsNaive.from_directory(f"data/cluster/{proteome_name}/neffs_naive")
    neffs_hhsuite = ProteomeNeffsHHsuite.from_directory(f"data/cluster/{proteome_name}/neffs_hhsuite")
    neffs_mmseqs = ProteomeNeffsMMseqs.from_directory(f"data/cluster/{proteome_name}/neffs_mmseqs/", only_half_scores=True)
    plddts = ProteomePLDDTs.from_file(f"data/{proteome_name}_HUMAN_v3_plddts_fltrd.json")
    seths = ProteomeSETHPreds.from_file("data/Human_SETH_preds.txt")
    msa_sizes = ProteomeMSASizes.from_file(f"data/cluster/{proteome_name}/msa_sizes.csv")

    correlation = ProteomeCorrelation([plddts, seths, neffs_naive, neffs_hhsuite, neffs_mmseqs], msa_sizes)
    set_name = "15k"
    plot(proteome_name, set_name, correlation)

    correlation = ProteomeCorrelation([plddts, seths, neffs, neffs_naive, neffs_hhsuite, neffs_mmseqs], msa_sizes)
    set_name = "300_Spartans"
    plot(proteome_name, set_name, correlation)