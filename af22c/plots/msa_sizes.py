from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns
import matplotlib.patches as patches
from af22c.proteome import ProteomeMSASizes
from af22c.plots import style


def plot_msa_sizes(
    msa_sizes: ProteomeMSASizes,
    uniprot_ids: set[str] = None,
    log_scale: bool = True,
    subsets: list[dict] = None,
    magnification_subset_name: str = None,
    histplot: bool = True,
    kdeplot: bool = True,
) -> sns.JointGrid:
    """
    Create a scatter plot showing a datapoint for a set of proteins depending on MSA meta information.

    The location of each datapoint/protein is determined by the two meta-parameters:
    * number of sequences in the MSA of the respective protein and
    * number of AAs in protein (query length).
    These two parameters are projected on the coordinate axes of the plot.
    """
    with style(start_idx=2) as c:
        sns.set_theme(style="dark")

        dataset_color = ".15" #c()

        # prefiltering of proteins
        msa_sizes_df = msa_sizes.get_msa_sizes()
        if uniprot_ids:
            if uniprot_ids - set(msa_sizes_df['uniprot_id']):
                raise ValueError('Not all specified uniprot ids where found in the precomputed msa sizes file.')
            msa_sizes_df = msa_sizes_df.loc[msa_sizes_df['uniprot_id'].isin(uniprot_ids)]

        # actual plotting
        marginal_kws = {
            "bins": 25,
            "fill": True,
            "color": dataset_color,
        }
        if log_scale:
            marginal_kws["log_scale"] = True
        # this function automatically creates the scatter plot
        p = sns.jointplot(
            data=msa_sizes_df,
            x="query_length",
            y="sequence_count",
            marginal_kws=marginal_kws,
            color=dataset_color,
            s=5,
        )
        if histplot:
            p.plot_joint(sns.histplot, bins=50, pthresh=.1, cmap="mako")
        if kdeplot:
            p.plot_joint(sns.kdeplot, levels=5, color="w", linewidths=1)
        p.set_axis_labels(
            "Number of Amino Acids in Query", "Number of Sequences in MSA"
        )
        if log_scale:
            p.ax_joint.set_xscale("log")
            p.ax_joint.set_yscale("log")

        if subsets:
            rects, names = [], []
            for idx, subset in enumerate(subsets):
                color = c()
                name, (num_seqs_min, num_seqs_max), (query_length_min, query_length_max) = (
                    subset["name"], subset["num_seqs_range"], subset["query_length_range"]
                )
                if magnification_subset_name is not None and name == magnification_subset_name:
                    mask = (
                        (query_length_min <= msa_sizes_df["query_length"]) &
                        (msa_sizes_df["query_length"] <= query_length_max) &
                        (num_seqs_min <= msa_sizes_df["sequence_count"]) &
                        (msa_sizes_df["sequence_count"] <= num_seqs_max)
                    )
                    zoomed_df = msa_sizes_df[mask]
                    ax_inset = inset_axes(p.ax_joint, width=2.0, height=2.0,
                                          bbox_to_anchor=(.4,.2,.6,.8),
                                          bbox_transform=p.ax_joint.transAxes, loc=3)
                    ax_inset.set_xlim(query_length_min,query_length_max)
                    ax_inset.set_ylim(num_seqs_min,num_seqs_max)
                    for val in ax_inset.spines.values():
                        val.set(color=color, linewidth=4)

                    sns.scatterplot(
                        data=zoomed_df,
                        x="query_length",
                        y="sequence_count",
                        color=dataset_color,
                        s=5,
                        ax=ax_inset,
                    )
                    if histplot:
                        sns.histplot(
                            data=zoomed_df,
                            x="query_length",
                            y="sequence_count",
                            log_scale=(True, False),
                            bins=50,
                            pthresh=.1,
                            cmap="mako",
                            ax=ax_inset,
                        )
                    if kdeplot:
                        sns.kdeplot(
                            data=zoomed_df,
                            x="query_length",
                            y="sequence_count",
                            log_scale=(True, False),
                            levels=5,
                            color="w",
                            linewidths=1,
                            ax=ax_inset,
                        )
                    ax_inset.set_xlabel("Log-scale Number of\nAmino Acids in Query")
                    ax_inset.set_ylabel(None)
                    ax_inset.set_xscale("log")

                    mark_inset(p.ax_joint, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

                rect = patches.Rectangle(
                    (query_length_min, num_seqs_min),
                    query_length_max - query_length_min,
                    num_seqs_max - num_seqs_min,
                    linewidth=4,
                    edgecolor=color,
                    facecolor="none",
                )
                p.ax_joint.add_patch(rect)
                rects.append(rect)
                names.append(name)
            p.ax_joint.legend(rects, names)

        return p
