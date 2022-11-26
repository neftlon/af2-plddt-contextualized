#!/usr/bin/env python3

import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from af22c.proteome import Proteome


def compute_and_store_msa_sizes(proteome):
    p = Path('data') / f'{proteome.name}_msa_size.csv'
    logging.info(f"examining MSA sizes ...")
    size_df = pd.DataFrame(np.array(proteome.get_msa_sizes()), columns=["query_length", "sequence_count"])
    size_df.to_csv(p, index=False)
    logging.info(f"written MSA sizes to {p}")


def plot_msa_sizes(filename):
    csv_path = Path(filename)
    size_df = pd.read_csv(csv_path)
    fig, ax = plt.subplots()
    sns.scatterplot(data=size_df, x='sequence_count', y='query_length', ax=ax)
    ax.set(xlabel='Number of Sequences in MSA', ylabel='Length of Query')
    plt.savefig(Path('data') / f'{csv_path.stem}_scatter.png')


def show_duplicates(proteome):
    for m in proteome.get_msas():
        m.examine_duplicates()


if __name__=='__main__':
    proteome = Proteome.from_folder('data/UP000005640_9606', name='UP000005640_9606')
    # TODO specify by argument what happens (show plot, just compute, ...?)
    # show_duplicates(proteome)
    compute_and_store_msa_sizes(proteome)