#!/usr/bin/env python

import argparse
from af22c.proteome import ProteomeMSAs, ProteomeMSASizes, ProteomeNeffs,\
    ProteomeNeffsNaive
from af22c.utils import add_msa_size_limit_options, size_limits_to_dict
import logging
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("proteome_msas_dir")
    parser.add_argument("data_dir")
    parser.add_argument("-p", "--protein_ids_file", default=None, type=str)
    parser.add_argument("--msa_sizes_file", default=None, type=str)
    parser = add_msa_size_limit_options(parser)
    parser.add_argument("-d", "--dry_run", action="store_true")
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--mode", default="ref", type=str)  # "ref" or "naive"
    args = parser.parse_args()

    proteome = ProteomeMSAs.from_directory(args.proteome_msas_dir)
    uniprot_ids = proteome.get_uniprot_ids()
    logging.info(f"found {len(uniprot_ids)} MSAs")

    # Filter by size
    limits = size_limits_to_dict(args)
    if any(limits.values()):
        if args.msa_sizes_file:
            msa_sizes = ProteomeMSASizes.from_file(args.msa_sizes_file)
        else:
            msa_sizes = ProteomeMSASizes.from_msas(proteome, args.data_dir)
            msa_sizes.precompute_msa_sizes()
        ids_to_process = msa_sizes.get_uniprot_ids_in_size(**limits)
        uniprot_ids &= ids_to_process

    # Filter by protein IDs
    if args.protein_ids_file:
        ids_df = pd.read_csv(args.protein_ids_file, header=None)
        ids_to_process = set(ids_df.squeeze('columns'))
        uniprot_ids &= ids_to_process

    logging.info(f"{len(uniprot_ids)} MSAs left after filtering")

    # Select mode
    if args.mode == "ref":
        data_dir = str(Path(args.data_dir) / "neffs")
        neffs = ProteomeNeffs.from_msas(proteome, data_dir)
    elif args.mode == "naive":
        data_dir = str(Path(args.data_dir) / "neffs_naive")
        neffs = ProteomeNeffsNaive.from_msas(proteome, data_dir)
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    logging.info(f"computing Neffs with mode '{args.mode}'")

    if not args.dry_run:
        for uniprot_id in uniprot_ids:
            neffs.compute_scores_by_id(uniprot_id, args.overwrite)
