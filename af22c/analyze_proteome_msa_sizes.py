#!/usr/bin/env python3

import argparse
from af22c.proteome import ProteomeMSAs, ProteomeMSASizes


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
        sizes.plot_msa_sizes()
    if args.duplicates:
        show_duplicates(sizes)
