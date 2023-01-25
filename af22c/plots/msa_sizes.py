import seaborn as sns
import matplotlib.patches as patches
from af22c.proteome import ProteomeMSASizes
from af22c.plots import style


def plot_msa_sizes(
    msa_sizes: ProteomeMSASizes,
    uniprot_ids: set[str] = None,
    log_scale: bool = True,
    subsets: list[dict] = None,
) -> sns.JointGrid:
    """
    Create a scatter plot showing a datapoint for a set of proteins depending on MSA meta information.

    The location of each datapoint/protein is determined by the two meta-parameters:
    * number of sequences in the MSA of the respective protein and
    * number of AAs in protein (query length).
    These two parameters are projected on the coordinate axes of the plot.
    """
    with style(start_idx=2) as c:
        dataset_color = c()

        # prefiltering of proteins
        msa_sizes_df = msa_sizes.get_msa_sizes()
        if uniprot_ids:
            if uniprot_ids - set(msa_sizes_df['uniprot_id']):
                raise ValueError('Not all specified uniprot ids where found in the precomputed msa sizes file.')
            msa_sizes_df = msa_sizes_df.loc[msa_sizes_df['uniprot_id'].isin(uniprot_ids)]

        # actual plotting
        #sns.set_style("whitegrid")
        marginal_kws = {
            "bins": 25,
            "fill": True,
            "color": dataset_color,
        }
        if log_scale:
            marginal_kws["log_scale"] = True
        p = sns.jointplot(
            data=msa_sizes_df,
            x="query_length",
            y="sequence_count",
            marginal_kws=marginal_kws,
            facecolor=dataset_color,
        )
        p.set_axis_labels(
            "Number of Amino Acids in Query", "Number of Sequences in MSA"
        )
        if log_scale:
            p.ax_joint.set_xscale("log")
            p.ax_joint.set_yscale("log")

        if subsets:
            rects, names = [], []
            for idx, subset in enumerate(subsets):
                name, (num_seqs_min, num_seqs_max), (query_length_min, query_length_max) = (
                    subset["name"], subset["num_seqs_range"], subset["query_length_range"]
                )
                rect = patches.Rectangle(
                    (query_length_min, num_seqs_min),
                    query_length_max - query_length_min,
                    num_seqs_max - num_seqs_min,
                    linewidth=4,
                    edgecolor=c(),
                    facecolor="none",
                )
                p.ax_joint.add_patch(rect)
                rects.append(rect)
                names.append(name)
            p.ax_joint.legend(rects, names)

        return p
