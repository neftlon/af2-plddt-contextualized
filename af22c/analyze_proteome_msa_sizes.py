#!/usr/bin/env python3

import argparse
from af22c.proteome import Proteome


def show_duplicates(proteome):
    for m in proteome.get_msas():
        m.examine_duplicates()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('proteome_dir')
    parser.add_argument('data_dir')
    parser.add_argument('-c', '--compute_size', action='store_true')
    parser.add_argument('-p', '--plot_size', action='store_true')
    parser.add_argument('-d', '--duplicates', action='store_true')
    args = parser.parse_args()

    proteome = Proteome.from_folder(args.proteome_dir, args.data_dir)
    if args.compute_size:
        proteome.compute_msa_sizes()
    if args.plot_size:
        proteome.plot_msa_sizes()
    if args.duplicates:
        show_duplicates(proteome)
