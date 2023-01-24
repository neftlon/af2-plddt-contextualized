import seaborn as sns
from af22c.proteome import ProteomeMSASizes


def plot_msa_sizes(msa_sizes: ProteomeMSASizes, uniprot_ids: set[str] = None) -> sns.JointGrid:
    """
    Create a scatter plot showing a datapoint for a set of proteins depending on MSA meta information.

    The location of each datapoint/protein is determined by the two meta-parameters:
    * number of sequences in the MSA of the respective protein and
    * number of AAs in protein (query length).
    These two parameters are projected on the coordinate axes of the plot.
    """
    # prefiltering of proteins
    msa_sizes_df = msa_sizes.get_msa_sizes()
    if uniprot_ids:
        if uniprot_ids - set(msa_sizes_df['uniprot_id']):
            raise ValueError('Not all specified uniprot ids where found in the precomputed msa sizes file.')
        msa_sizes_df = msa_sizes_df.loc[msa_sizes_df['uniprot_id'].isin(uniprot_ids)]

    # actual plotting
    sns.set_style("whitegrid")
    p = sns.jointplot(data=msa_sizes_df, x="query_length", y="sequence_count")
    p.set_axis_labels(
        "Number of Amino Acids in Query", "Number of Sequences in MSA"
    )
    return p
