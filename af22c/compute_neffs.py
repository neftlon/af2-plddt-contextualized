#!/usr/bin/env python

import argparse
from af22c.proteome import ProteomeMSAs, ProteomeMSASizes, ProteomeNeffs,\
    ProteomeNeffsNaive
import logging
import math
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("proteome_msas_dir")
    parser.add_argument("data_dir")
    parser.add_argument("-p", "--protein_ids_file", default=None, type=str)
    parser.add_argument("--msa_sizes_file", default=None, type=str)
    parser.add_argument("-n", "--max_n_sequences", default=math.nan, type=int)
    parser.add_argument("-l", "--max_query_length", default=math.nan, type=int)
    parser.add_argument("-m", "--min_n_sequences", default=math.nan, type=int)
    parser.add_argument("-k", "--min_query_length", default=math.nan, type=int)
    parser.add_argument("-d", "--dry_run", action="store_true")
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--mode", default="ref", type=str)  # "ref" or "naive"
    args = parser.parse_args()

    proteome = ProteomeMSAs.from_directory(args.proteome_msas_dir)
    uniprot_ids = proteome.get_uniprot_ids()
    logging.info(f"found {len(uniprot_ids)} MSAs")

    # Filter by size
    size_limits = (args.min_query_length, args.max_query_length,
                   args.min_n_sequences, args.max_n_sequences)
    if not all(math.isnan(v) for v in size_limits):
        if args.msa_sizes_file:
            msa_sizes = ProteomeMSASizes.from_file(args.msa_sizes_file)
        else:
            msa_sizes = ProteomeMSASizes.from_msas(proteome, args.data_dir)
            msa_sizes.precompute_msa_sizes()
        ids_to_process = msa_sizes.get_uniprot_ids_in_size(
            min_q_len=args.min_query_length,
            max_q_len=args.max_query_length,
            min_n_seq=args.min_n_sequences,
            max_n_seq=args.max_n_sequences
        )
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
