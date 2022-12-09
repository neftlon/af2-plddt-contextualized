import argparse
from af22c.proteome import ProteomeMSAs, ProteomeMSASizes, ProteomeMaxZs
import logging
import math
from pathlib import Path
import json

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("proteome_msas_dir")
    parser.add_argument("data_dir")
    parser.add_argument("msa_sizes_file")
    parser.add_argument("-n", "--max_n_sequences", default=math.inf, type=int)
    parser.add_argument("-l", "--max_query_length", default=math.inf, type=int)
    parser.add_argument("-m", "--min_n_sequences", default=0, type=int)
    parser.add_argument("-k", "--min_query_length", default=0, type=int)
    parser.add_argument("-d", "--dry_run", action="store_true")
    args = parser.parse_args()

    proteome = ProteomeMSAs.from_directory(args.proteome_msas_dir)
    msa_sizes = ProteomeMSASizes.from_file(args.msa_sizes_file)
    maxzs = ProteomeMaxZs.from_msas(proteome, str(Path(args.data_dir) / "maxzs"))

    uniprot_ids = proteome.get_uniprot_ids()
    msa_sizes_ids = msa_sizes.get_uniprot_ids()
    logging.info(
        f"found {len(uniprot_ids)} MSAs and sizes for {len(msa_sizes_ids)} MSAs, "
        f"selecting subset ..."
    )
    ids_in_size = msa_sizes.get_uniprot_ids_in_size(
        min_q_len=args.min_query_length,
        max_q_len=args.max_query_length,
        min_n_seq=args.min_n_sequences,
        max_n_seq=args.max_n_sequences,
    )
    uniprot_ids &= ids_in_size
    logging.info(
        f"computing scores for {len(uniprot_ids)} MSAs "
        f"with {args.min_query_length} <= query length <= {args.max_query_length} "
        f"and {args.min_n_sequences} <= number of sequences <= {args.max_n_sequences} ..."
    )

    if not args.dry_run:
        for uniprot_id in uniprot_ids:
            try:
                maxzs.compute_scores_by_id(uniprot_id, overwrite_scores=True)
            except ValueError as e:
                logging.error(f"Unable to compute maxzs scores for {uniprot_id}. \n {e}")
