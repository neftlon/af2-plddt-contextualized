#!/usr/bin/env python3

import tarfile
import logging
import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from af22c.load_msa import apply_by_id, get_a3m_size
from af22c.neff_cache_or_calc import NeffCacheOrCalc


def main():
    # setup cache handler
    neff_src = NeffCacheOrCalc(
        proteome_filename="data/UP000005640_9606.tar",
        cache_filename="data/UP000005640_9696_neff_cache.tar",
    )

    # calculate which IDs are available
    logging.debug("finding available protein IDs")
    with tarfile.open(neff_src.proteome_filename) as f:
        filenames = f.getnames()
        proteome_name = neff_src.get_raw_proteome_name()
        protein_msa_files = [fn for fn in filenames if fn.startswith(f"{proteome_name}/msas/") and fn.endswith(".a3m")]
        avail_prot_ids = [os.path.splitext(os.path.basename(fn))[0] for fn in protein_msa_files]
    logging.debug(f"found {len(avail_prot_ids)} proteins to look at")

    logging.info(f"examining MSA sizes")
    msa_sizes = []
    for uniprot_id in tqdm(avail_prot_ids):
        msa_sizes.append(apply_by_id(get_a3m_size, neff_src.proteome_filename, uniprot_id))

    size_df = pd.DataFrame(np.array(msa_sizes), columns=["query_length", "sequence_count"])
    fig, ax = plt.subplots()
    sns.scatterplot(data=size_df, x='sequence_count', y='query_length', ax=ax)
    ax.set(xlabel='Number of Sequences in MSA', ylabel='Length of Query')
    plt.savefig("data/msa_size_scatter.png")

if __name__=='__main__':
    main()