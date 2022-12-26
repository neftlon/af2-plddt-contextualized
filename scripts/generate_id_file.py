import argparse
from af22c.proteome import ProteomeMSASizes
import logging
import math
import random
import pandas as pd

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("msa_sizes_file")
    parser.add_argument("result_file")
    parser.add_argument("-n", "--max_n_sequences", default=math.inf, type=int)
    parser.add_argument("-l", "--max_query_length", default=math.inf, type=int)
    parser.add_argument("-m", "--min_n_sequences", default=0, type=int)
    parser.add_argument("-k", "--min_query_length", default=0, type=int)
    parser.add_argument("-b", "--big_representative_only", action="store_true", default=False)
    args = parser.parse_args()

    limits = args.min_query_length, args.max_query_length, args.min_n_sequences, args.max_n_sequences
    msa_sizes = ProteomeMSASizes.from_file(args.msa_sizes_file)

    if args.big_representative_only:
        # Assumption 100 away from max doesn't make much of a difference
        args.min_n_sequences = max(0, args.max_n_sequences - 100)
        args.min_query_length = max(0, args.max_query_length - 100)
        uniprot_ids = msa_sizes.get_uniprot_ids_in_size(args.min_query_length, args.max_query_length,
                                                        args.min_n_sequences, args.max_n_sequences)
        if len(uniprot_ids) < 1:
            raise ValueError("No MSA in this size range!")
        representative = random.choice(tuple(uniprot_ids))
        logging.info(
            f"Found representative MSA with "
            f"{msa_sizes[representative][1]} query length and "
            f"{msa_sizes[representative][0]} sequences."
        )
        pd.Series(representative).to_csv(args.result_file, index=False, header=False)
    else:
        uniprot_ids = msa_sizes.get_uniprot_ids_in_size(args.min_query_length, args.max_query_length, args.min_n_sequences, args.max_n_sequences)
        logging.info(
            f"Found {len(uniprot_ids)} MSAs with "
            f"query length in [{args.min_query_length}, {args.max_query_length}] and "
            f"number of sequences in [{args.min_n_sequences}, {args.max_n_sequences}]."
        )
        pd.Series(list(uniprot_ids)).to_csv(args.result_file, index=False, header=False)
