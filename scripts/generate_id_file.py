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
    parser.add_argument("-s", "--sample_size", default=None, type=int)
    parser.add_argument("-b", "--biggest_only", action="store_true",
                        default=False)
    args = parser.parse_args()

    limits = {
        "min_q_len": args.min_query_length,
        "max_q_len": args.max_query_length,
        "min_n_seq": args.min_n_sequences,
        "max_n_seq": args.max_n_sequences
    }
    msa_sizes = ProteomeMSASizes.from_file(args.msa_sizes_file)
    uniprot_ids = list(msa_sizes.get_uniprot_ids_in_size(**limits))

    if len(uniprot_ids) < 1:
        raise ValueError("No MSAs found in size range.")
    elif args.sample_size and len(uniprot_ids) < args.sample_size:
        raise ValueError("Not enough MSAs in this size range!")

    logging.info(
        f"Found {len(uniprot_ids)} MSAs in size range {limits}."
    )

    if args.biggest_only:
        n = args.sample_size if args.sample_size else 1

        # Load all MSA sizes into a DataFrame
        size_by_id = {prot_id: msa_sizes[prot_id] for prot_id in uniprot_ids}
        size_df = pd.DataFrame.from_dict(size_by_id, orient='index',
                                         columns=['n_seq', 'q_len'])

        # Find the biggest s MSAs
        size_df["combined_size"] = size_df["n_seq"] * size_df["q_len"]
        size_df.sort_values(by="combined_size", ascending=False, inplace=True)
        size_df.drop(columns=["combined_size"], inplace=True)
        uniprot_ids = size_df.index[:n].tolist()

        logging.info(
            f"Selected {n} biggest MSA(s), with top size(s): \n"
            f"{size_df.head(min(5, n))}"
        )
    elif args.sample_size:
        uniprot_ids = random.sample(uniprot_ids, args.sample_size)
        prot_sizes = np.array([msa_sizes[prot_id] for prot_id in uniprot_ids])
        logging.info(f"Selected a random sample of size {args.sample_size} "
                     f"with the following statistics:\n"
                     f"mean query length: {prot_sizes[:, 1].mean():.2f},\n"
                     f"min query length: {prot_sizes[:, 1].min()},\n"
                     f"max query length: {prot_sizes[:, 1].max()},\n"
                     f"mean number of sequences: {prot_sizes[:, 0].mean():.2f},\n"
                     f"min number of sequences: {prot_sizes[:, 0].min()},\n"
                     f"max number of sequences: {prot_sizes[:, 0].max()}")

    # Write to file
    pd.Series(uniprot_ids).to_csv(args.result_file, index=False, header=False)
