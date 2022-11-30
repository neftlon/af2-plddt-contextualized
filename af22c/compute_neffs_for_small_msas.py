import argparse
from af22c.proteome import Proteome
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('proteome_dir')
    parser.add_argument('data_dir')
    parser.add_argument('-n', '--n_sequences')
    parser.add_argument('-l', '--query_length')
    parser.add_argument('-d', '--dry_run', action='store_true')
    args = parser.parse_args()

    proteome = Proteome.from_folder(args.proteome_dir, args.data_dir)

    msa_sizes = proteome.get_msa_sizes()
    logging.info(f"found {len(msa_sizes)} MSAs, selecting subset ...")
    subset_small = msa_sizes[(msa_sizes["query_length"] <= int(args.query_length))
                             & (msa_sizes["sequence_count"] <= int(args.n_sequences))]
    logging.info(f"computing Neffs for {len(subset_small)} MSAs with query length <= {int(args.query_length)} "
                 f"and number of sequences <= {int(args.n_sequences)} ...")

    if not args.dry_run:
        for uniprot_id in subset_small["uniprot_id"]:
            proteome.compute_neff_by_id(uniprot_id)
            proteome.compute_neff_naive_by_id(uniprot_id)



