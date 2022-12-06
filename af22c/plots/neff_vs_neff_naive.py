#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import logging

from af22c.load_msa import calc_naive_neff_by_id
from af22c.neff_cache_or_calc import NeffCacheOrCalc


def plot_neff_vs_neff_naive(
    ax, naive_neff_scores: list[float], neff_scores: list[float], plot_difference=True
):
    """
    Create a plot on `ax` that shows the difference between the naive Neff score calculation and an actual calculation.

    Parameters
    ----------
    plot_difference : bool
        If this flag is set, the per-residue difference is plotted as a separate line
    """
    # aggregate data
    feature_names = ["Neff", "Neff naive"]
    features = [neff_scores, naive_neff_scores]
    if plot_difference:
        feature_names.append("Neff - Neff naive")
        features.append([y - x for x, y in zip(neff_scores, naive_neff_scores)])

    # combine in dataframe for seaborn
    df = pd.DataFrame(
        np.array(
            features
        ).transpose(),  # wants one features per column, not per row -> transpose
        columns=feature_names,
    )

    # plot all features
    for name in feature_names:
        sns.lineplot(df, x=df.index, y=name, ax=ax, label=name)


def main():
    parser = argparse.ArgumentParser(
        prog=__name__,
        description="Plot the difference between an actual Neff score calculation and a naive one (only count gaps in "
        "an MSA column).",
    )
    parser.add_argument(
        "-f",
        "--proteome-file",
        help="Path to a .tar or .tar.gz file containing MSAs of a proteome",
        default="data/UP000005640_9606.tar",
    )
    parser.add_argument(
        "-p",
        "--protein-id",
        help="UniProt identifier of the protein to calculate scores on",
        default="Q96QF7",
    )
    parser.add_argument(
        "-c",
        "--cache-file",
        help="Select a cache file for precomputed Neff scores",
        default="data/UP000005640_9606_neff_cache.tar",
    )
    out_default_path = "data/plot_neff_vs_neff_naive_PROTID.png"
    parser.add_argument(
        "-o",
        "--out-file",
        help=f"Location for the plotted image. If no location is specified, the default path `{out_default_path}` is "
        f"taken, where `PROTID` is replaced by the specified UniProt identifier.",
        default=out_default_path,
    )
    args = parser.parse_args()

    neff_src = NeffCacheOrCalc(
        proteome_filename=args.proteome_file,
        cache_filename=args.cache_file,
    )

    # obtain scores
    neffs = neff_src.get_neffs(args.protein_id)
    neffs_naive = calc_naive_neff_by_id(args.proteome_file, args.protein_id)

    # calculate correlation
    pearson = np.corrcoef(np.stack((neffs, neffs_naive)))[0, 1]

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(
        title=f"per-residue Neff vs. naive version for {args.protein_id}, pearson correlation = {pearson:.04f}",
        xlim=(0, len(neffs)),
        xlabel="residue number",
        ylabel="score",
    )
    ax.grid()
    plot_neff_vs_neff_naive(ax, neffs, neffs_naive)
    out_file = args.out_file.replace("PROTID", args.protein_id)
    ax.legend()
    plt.savefig(out_file)

    logging.info(f"plotted Neff scores of {args.protein_id} to {out_file}")


if __name__ == "__main__":
    main()
