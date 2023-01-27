#!/usr/bin/env python3
import logging
import os

import matplotlib.pyplot as plt
import argparse
from af22c.utils import readable_enumeration
from af22c.plots import style
from af22c.plots.per_protein_score import plot_multiple_scores_on_one_axis
from af22c.proteome import *


if __name__ == "__main__":
    # parameter parsing
    # -----------------
    SCORENAME_TO_CLASS = {
        "neffs": lambda dirname: ProteomeNeffs.from_directory(dirname),
        "neffs_hhsuite": lambda dirname: ProteomeNeffsHHsuite.from_directory(dirname),
        "neffs_mmseqs": lambda dirname: ProteomeNeffsMMseqs.from_directory(dirname),
        "gapcount": lambda dirname: ProteomeNeffsNaive.from_directory(dirname),
    }
    SUPPORTED_DIRNAMES = list(SCORENAME_TO_CLASS.keys())
    parser = argparse.ArgumentParser(
        description=f"Plot per residue scores in a single PNG file. Scores can be obtained from the following "
                    f"subfolders of `data_src`: {readable_enumeration(SUPPORTED_DIRNAMES)}."
    )
    parser.add_argument(
        "data_src",
        help="The root directory in which the scores can be found. (The directory could contain the following "
             "subfolders from which scores are collected: mmseqs, neffs, neffs_hhsuite, gapcount, ...)",
    )
    parser.add_argument(
        "out_dir",
        help="Plot is stored in `out_dir`/human_`PROT`_per_res.png",
    )
    parser.add_argument(
        "prot_id",
        help="Specify UniProt identifier for which all scores should be collected",
        metavar="PROT",
    )
    args = parser.parse_args()
    # -----------------

    # find relevant subdirs
    def collect_scores(path) -> list[ProteomewidePerResidueMetric]:
        scores = []
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if filename.endswith("_msa_size.csv"):
                ... # skip, cannot use the file in this context
            elif filename.startswith("UP") and os.path.isdir(filepath):
                # search proteome subdirectory
                scores.extend(collect_scores(filepath))
            elif filename in SUPPORTED_DIRNAMES:
                fn = SCORENAME_TO_CLASS[filename]
                score = fn(filepath)
                scores.append(score)
        return scores

    scores = collect_scores(args.data_src)
    with style():
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_multiple_scores_on_one_axis(ax, scores, args.prot_id)

        # TODO: where to obtain this from?
        proteome_name = "human"

        figpath = os.path.join(args.out_dir, f"{proteome_name}_{args.prot_id}_per_res.png")
        fig.savefig(figpath)
