import argparse
from af22c.proteome import ProteomeMSASizes
import logging
import math
import random
import pandas as pd
import numpy as np

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("msa_sizes_file")
    parser.add_argument("result_file")
    parser.add_argument("-n", "--max_n_sequences", default=math.inf, type=int)
    parser.add_argument("-l", "--max_query_length", default=math.inf, type=int)
    parser.add_argument("-m", "--min_n_sequences", default=0, type=int)
    parser.add_argument("-k", "--min_query_length", default=0, type=int)
    parser.add_argument("-b", "--big_representative_only", action="store_true",
                        default=False)
    parser.add_argument("-s", "--random_sample_size", default=math.nan,
                        type=int)
    args = parser.parse_args()

    limits = {
        "min_q_len": args.min_query_length,
        "max_q_len": args.max_query_length,
        "min_n_seq": args.min_n_sequences,
        "max_n_seq": args.max_n_sequences
    }
    msa_sizes = ProteomeMSASizes.from_file(args.msa_sizes_file)

    if args.big_representative_only:
        uniprot_ids = list(msa_sizes.get_uniprot_ids_in_size(**limits))  # cast to list to make it indexable
        prot_sizes = np.array([msa_sizes[prot_id] for prot_id in uniprot_ids])

        # Get the biggest representative
        max_idx = np.argmax(prot_sizes[:, 0] * prot_sizes[:, 1])  # n_seq * q_len provides the "size" of the MSA

        representative = uniprot_ids[max_idx]
        logging.info(
            f"Found representative MSA with "
            f"{msa_sizes[representative][1]} query length and "
            f"{msa_sizes[representative][0]} sequences."
        )
        id_list = pd.Series([representative])
    elif not math.isnan(args.random_sample_size):
        uniprot_ids = msa_sizes.get_uniprot_ids_in_size(**limits)
        if len(uniprot_ids) < args.random_sample_size:
            raise ValueError("Not enough MSA in this size range!")
        uniprot_ids = random.sample(list(uniprot_ids), args.random_sample_size)
        prot_sizes = np.array([msa_sizes[prot_id] for prot_id in uniprot_ids])
        logging.info(f"Selected a random sample with the following statistics: "
                     f"mean query length: {prot_sizes[:, 1].mean():.2f}, "
                     f"min query length: {prot_sizes[:, 1].min()}, "
                     f"max query length: {prot_sizes[:, 1].max()}, "
                     f"mean number of sequences: {prot_sizes[:, 0].mean():.2f}, "
                     f"min number of sequences: {prot_sizes[:, 0].min()}, "
                     f"max number of sequences: {prot_sizes[:, 0].max()}")
        id_list = pd.Series(uniprot_ids)
    else:
        uniprot_ids = msa_sizes.get_uniprot_ids_in_size(**limits)
        logging.info(
            f"Found {len(uniprot_ids)} MSAs in size range {limits}."
        )
        id_list = pd.Series(list(uniprot_ids))

    # Write to file
    id_list.to_csv(args.result_file, index=False, header=False)
