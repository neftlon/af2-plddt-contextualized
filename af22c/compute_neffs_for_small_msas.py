import argparse
from af22c.proteome import Proteome
import logging
import math

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("proteome_dir")
    parser.add_argument("data_dir")
    parser.add_argument("-n", "--max_n_sequences", default=math.inf, type=int)
    parser.add_argument("-l", "--max_query_length", default=math.inf, type=int)
    parser.add_argument("-m", "--min_n_sequences", default=0, type=int)
    parser.add_argument("-k", "--min_query_length", default=0, type=int)
    parser.add_argument("-d", "--dry_run", action="store_true")
    args = parser.parse_args()

    proteome = Proteome.from_folder(args.proteome_dir, args.data_dir)

    msa_sizes = proteome.get_msa_sizes()
    uniprot_ids = proteome.get_uniprot_ids()
    logging.info(
        f"found {len(uniprot_ids)} MSAs and sizes for {len(msa_sizes)} MSAs, "
        f"selecting subset ..."
    )
    subset_ids = proteome.get_uniprot_ids_in_size(
        min_q_len=args.min_query_length,
        max_q_len=args.max_query_length,
        min_n_seq=args.min_n_sequences,
        max_n_seq=args.max_n_sequences,
    )
    logging.info(
        f"computing Neffs for {len(subset_ids)} MSAs "
        f"with {args.min_query_length} <= query length <= {args.max_query_length} "
        f"and {args.min_n_sequences} <= number of sequences <= {args.max_n_sequences} ..."
    )

    if not args.dry_run:
        for uniprot_id in subset_ids:
            proteome.compute_neff_by_id(uniprot_id)
            proteome.compute_neff_naive_by_id(uniprot_id)
