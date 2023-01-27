#!/usr/bin/env python3

import argparse
import json
import os

from af22c.plots.msa_sizes import plot_msa_sizes
from af22c.proteome import ProteomeMSAs, ProteomeMSASizes
from pathlib import Path
import logging


def show_duplicates(proteome):
    for m in proteome.get_msas():
        m.examine_duplicates()


if __name__ == "__main__":
    # parameter parsing
    # -----------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_src",
        help="Either a protein source (to calculate sizes on demand; can be .tar[.gz] archive or folder containing .a3m"
             " files) or a CSV sizes containing the protein size information (then, no on-demand calculation is "
             "possible)",
    )
    parser.add_argument(
        "data_dir",
        help="Used for caching when `data_src` points to an archive or folder; also used for storing the output images "
             "when using the -p option.",
    )
    parser.add_argument(
        "-s",
        "--subsets-file",
        type=argparse.FileType("r"),
        help="Specify ranges for protein subsets. The file must be a .json file with one array in it. The elements of "
             "these arrays must be objects with the following fields: \"name\" for subset name, \"num_seqs_range\" to "
             "specify an array with min/max entries of # of sequences, and \"query_length_range\" to specify an array "
             "of min/max entries of query length",
        metavar="SUBSF",
    )
    parser.add_argument("-c", "--compute_size", action="store_true")
    parser.add_argument("-p", "--plot_size", action="store_true")
    parser.add_argument("-d", "--duplicates", action="store_true")
    args = parser.parse_args()
    # -----------------

    # data extraction
    # -----------------
    if not args.data_src.endswith(".csv"):
        proteome = ProteomeMSAs.from_file(args.data_src)
        sizes = ProteomeMSASizes.from_msas(proteome, args.data_dir)
    else:
        sizes = ProteomeMSASizes.from_file(args.data_src)
    # -----------------

    # main functionality
    # -----------------
    if args.compute_size:
        sizes.precompute_msa_sizes()
    if args.plot_size:
        subsets = None
        if args.subsets_file:
            subsets = json.load(args.subsets_file)

        # create plots
        for scale_ext, scale_opt in [("", False), ("_logscale", True)]:
            jg = plot_msa_sizes(sizes, log_scale=scale_opt, subsets=subsets)

            # save plot
            data_dir = Path(args.data_dir)
            name = "human"  # TODO: add support for other proteomes as well?
            fig_dir = data_dir / "plots"
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = fig_dir / f"{name}_msa_size_scatter{scale_ext}.png"
            jg.savefig(fig_path)

            logging.info(f"saved figure to {fig_path}")
    if args.duplicates:
        show_duplicates(sizes)
    # -----------------
