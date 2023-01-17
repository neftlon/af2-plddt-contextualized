#!/usr/bin/env python

import argparse
import logging
import pandas as pd
from af22c.proteome import ProteomeMSASizes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("msa_sizes_file")
    parser.add_argument("protein_ids_file")

    args = parser.parse_args()

    msa_sizes = ProteomeMSASizes.from_file(args.msa_sizes_file)
    ids_df = pd.read_csv(args.protein_ids_file, header=None)
    ids_to_process = set(ids_df.squeeze('columns'))

    size_by_id = {prot_id: msa_sizes[prot_id] for prot_id in ids_to_process}
    size_df = pd.DataFrame.from_dict(size_by_id, orient='index',
                                     columns=['n_seq', 'q_len'])

    print(size_df)

