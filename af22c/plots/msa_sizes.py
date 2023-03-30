import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Union
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from af22c.proteome import ProteomeMSASizes, ProteomeCorrelation
from af22c.plots.style import style
from af22c.plots.per_proteome_score import style_plots


def plot_msa_sizes(
    msa_sizes: ProteomeMSASizes,
    uniprot_ids: set[str] = None,
    log_scale: Union[bool, tuple[bool, bool]] = True,
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
        sns.set_theme(style="darkgrid")

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
            marginal_kws["log_scale"] = log_scale
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

        # decide which axis should use logscale
        if isinstance(log_scale, bool) and log_scale:
            log_scale = (True,)*2
        if isinstance(log_scale, tuple) and len(log_scale) == 2:
            xlog, ylog = log_scale
            if xlog:
                p.ax_joint.set_xscale("log")
            if ylog:
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


def get_msa_size_bins(msa_sizes_df, n_bins=50):
    min_q_len = msa_sizes_df['query_length'].min()
    max_q_len = msa_sizes_df['query_length'].max()
    min_n_seq = msa_sizes_df['sequence_count'].min()
    max_n_seq = msa_sizes_df['sequence_count'].max()
    q_len_bins = np.linspace(min_q_len, max_q_len, n_bins)
    n_seq_bins = np.linspace(min_n_seq, max_n_seq, n_bins)
    return q_len_bins, n_seq_bins


def get_corrs_in_bin(
        q_len_bin: tuple[int, int],
        n_seq_bin: tuple[int, int],
        corrs,
        msa_sizes_df,
        corr_uniprot_ids_in_order: list[str]
) -> list[float]:
    # corrs np.array of shape (1, n_uniprot_ids)

    # Expand limits
    q_len_bin_min, q_len_bin_max = q_len_bin
    n_seq_bin_min, n_seq_bin_max = n_seq_bin

    # Select proteins in bin
    q_len_mask = ((msa_sizes_df['query_length'] >= q_len_bin_min)
                  & (msa_sizes_df['query_length'] < q_len_bin_max))
    n_seq_mask = ((msa_sizes_df['sequence_count'] >= n_seq_bin_min)
                  & (msa_sizes_df['sequence_count'] < n_seq_bin_max))
    mask = q_len_mask & n_seq_mask
    msa_sizes_df_ids_in_bin = msa_sizes_df.loc[mask, 'uniprot_id']

    # Extract correlations for proteins in bin
    corrs_in_bin = []
    for prot_id in msa_sizes_df_ids_in_bin:
        corrs_in_bin.append(corrs[corr_uniprot_ids_in_order.index(prot_id)])
    return corrs_in_bin


def get_binned_corrs(
        q_len_bins,
        n_seq_bins,
        corrs,
        msa_sizes_df,
        corr_uniprot_ids_in_order
) -> list[list[list[float]]]:
    binned_corrs = []
    for i in range(len(n_seq_bins) - 1):
        binned_corrs.append([])
        for j in range(len(q_len_bins) - 1):
            n_seq_bin = (n_seq_bins[i], n_seq_bins[i + 1])
            q_len_bin = (q_len_bins[j], q_len_bins[j + 1])
            corrs_in_bin = get_corrs_in_bin(q_len_bin,
                                            n_seq_bin,
                                            corrs,
                                            msa_sizes_df,
                                            corr_uniprot_ids_in_order)
            binned_corrs[i].append(corrs_in_bin)
    return list(reversed(binned_corrs))


def plot_msa_sizes_with_correlation(
    correlations: ProteomeCorrelation,
    uniprot_ids: set[str] = None,
    aggregate: str = "mean",
) -> sns.JointGrid:

    msa_sizes_df = correlations.msa_sizes.get_msa_sizes()

    # Excluding proteins not in correlation object
    msa_sizes_ids = correlations.msa_sizes.get_uniprot_ids()
    corr_uniprot_ids = correlations.get_uniprot_ids()
    if corr_uniprot_ids - msa_sizes_ids:
        raise ValueError('Not all uniprot ids from the correlation file are '
                         'present in the msa sizes file.')
    msa_sizes_df = msa_sizes_df.loc[
        msa_sizes_df['uniprot_id'].isin(corr_uniprot_ids)
    ]

    # Excluding proteins not in uniprot_ids
    if uniprot_ids:
        if uniprot_ids - set(msa_sizes_df['uniprot_id']):
            raise ValueError('Not all specified uniprot ids where found in '
                             'the precomputed msa sizes file.')
        msa_sizes_df = msa_sizes_df.loc[
            msa_sizes_df['uniprot_id'].isin(uniprot_ids)
        ]
    msa_sizes_df.reset_index(drop=True, inplace=True)

    # Compute correlations
    corrs_stack, df_index, ids_in_order = correlations.get_pearson_corr_stack()

    # Bin proteins by msa size
    q_len_bins, n_seq_bins = get_msa_size_bins(msa_sizes_df)

    # Plot for each score combination
    n = len(correlations.scores)
    fig, axs = plt.subplots(n - 1, n - 1)  # n-1, because the diagonal is not drawn
    for i in range(0, n - 1):
        for j in range(0, i + 1):
            corrs = corrs_stack[:, i + 1, j].flatten()
            binned_corrs = get_binned_corrs(q_len_bins,
                                            n_seq_bins,
                                            corrs,
                                            msa_sizes_df,
                                            ids_in_order)
            for k in range(len(n_seq_bins) - 1):
                for l in range(len(q_len_bins) - 1):
                    vmin = -1
                    vmax = 1
                    if aggregate == "mean":
                        binned_corrs[k][l] = np.mean(binned_corrs[k][l])
                    elif aggregate == "median":
                        binned_corrs[k][l] = np.median(binned_corrs[k][l])
                    elif aggregate == "abs_mean":
                        binned_corrs[k][l] = np.mean(np.abs(binned_corrs[k][l]))
                        vmin = 0
                    elif aggregate == "abs_median":
                        binned_corrs[k][l] = np.median(
                            np.abs(binned_corrs[k][l])
                        )
                        vmin = 0
                    elif aggregate == "std":
                        binned_corrs[k][l] = np.std(binned_corrs[k][l])
                        vmin = 0
                    else:
                        raise ValueError(f"Invalid aggregation method "
                                         f"{aggregate}.")
            binned_corrs = np.array(binned_corrs)

            sns.heatmap(binned_corrs,
                        cmap='mako',
                        vmin=vmin,
                        vmax=vmax,
                        ax=axs[i, j])

    style_plots(axs, df_index[0], df_index[1])
    return fig