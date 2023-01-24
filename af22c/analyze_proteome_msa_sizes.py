#!/usr/bin/env python3

import argparse

from af22c.plots.msa_sizes import plot_msa_sizes
from af22c.proteome import ProteomeMSAs, ProteomeMSASizes
from pathlib import Path
import logging
import matplotlib.pyplot as plt


def show_duplicates(proteome):
    for m in proteome.get_msas():
        m.examine_duplicates()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "proteome_src",
        help="Proteome source folder or archive (.tar or .tar.gz) containing MSAs",
    )
    parser.add_argument("data_dir")
    parser.add_argument("-c", "--compute_size", action="store_true")
    parser.add_argument("-p", "--plot_size", action="store_true")
    parser.add_argument("-d", "--duplicates", action="store_true")
    args = parser.parse_args()

    proteome = ProteomeMSAs.from_file(args.proteome_src)
    sizes = ProteomeMSASizes.from_msas(proteome, args.data_dir)
    if args.compute_size:
        sizes.precompute_msa_sizes()
    if args.plot_size:
        # create plot
        jg = plot_msa_sizes(sizes)

        # save plot
        data_dir = Path(args.data_dir)
        name = "human"  # TODO: add support for other proteomes as well?
        fig_path = data_dir / "plots" / f"{name}_msa_size_scatter.png"
        jg.savefig(fig_path)

        logging.info(f"saved figure to {fig_path}")
    if args.duplicates:
        show_duplicates(sizes)
