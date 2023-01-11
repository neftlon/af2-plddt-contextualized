import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from af22c.proteome import ProteomeCorrelation


def style_plots(axs, x_labels, y_labels, x_ticks=None, y_ticks=None, x_limits=None, y_limits=None):
    n = len(axs)
    for i in range(0, n):
        for j in range(0, n):
            if i >= j:
                axs[i, j].set_xticklabels([])
                axs[i, j].set_ylim(y_limits)
                axs[i, j].set_xlim(x_limits)
                axs[i, j].set_yticklabels([])
                axs[i, j].set_ylabel('')
            else:
                axs[i, j].axis('off')

    # Add x labels to the bottom of the matrix
    for i in range(n):
        axs[-1, i].set_xlabel(x_labels[i])
        if x_ticks:
            axs[-1, i].set_xticks(x_ticks)
            axs[-1, i].set_xticklabels(x_ticks)

    # Add y labels to the left of the matrix and ticks to the leftmost plot
    for i in range(n):
        axs[i, 0].set_ylabel(y_labels[i+1])  # +1, because the diagonal is not drawn
        if y_ticks:
            axs[i, 0].set_yticks(y_ticks)
            axs[i, 0].set_yticklabels(y_ticks)


def plot_correlation_boxplots(corr: ProteomeCorrelation):
    p_corr_array, df_index = corr.get_pearson_corr_stack()
    n = len(corr.scores)
    fig, axs = plt.subplots(n-1, n-1)  # n-1, because the diagonal is not drawn
    for i in range(0, n-1):
        for j in range(0, i+1):
            sns.boxplot(p_corr_array[:, i+1, j].flatten(), ax=axs[i, j], width=0.5)

    style_plots(axs, df_index[0], df_index[1], y_ticks=[-1, 0, 1], y_limits=(-1, 1))
    return fig


def plot_correlation_histograms(corr: ProteomeCorrelation):
    p_corr_array, df_index = corr.get_pearson_corr_stack()
    n = len(corr.scores)
    fig, axs = plt.subplots(n-1, n-1)  # n-1, because the diagonal is not drawn
    for i in range(0, n-1):
        for j in range(0, i+1):
            sns.histplot(p_corr_array[:, i+1, j].flatten(), ax=axs[i, j])

    style_plots(axs, df_index[0], df_index[1], x_ticks=[-1, 0, 1], x_limits=(-1, 1))
    return fig


def plot_correlation_means(ax: plt.Axes, corr: ProteomeCorrelation):
    # TODO Fix NaN values when including small msas, problem probably in computation. Size range limits are set to
    #  avoid those errors.
    p_corr_array, df_index = corr.get_pearson_corr_stack(min_q_len=10,
                                                         max_q_len=300,
                                                         min_n_seq=100,
                                                         max_n_seq=2000)
    p_corr_mean = pd.DataFrame(
        np.mean(p_corr_array, axis=0), index=df_index[0], columns=df_index[1]
    )

    mask = np.triu(np.ones_like(p_corr_mean, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(p_corr_mean, annot=True, mask=mask, cmap=cmap, vmin=-1, vmax=1, ax=ax)


def plot_whole_dataset_scatter(corr: ProteomeCorrelation):
    obs_dict = {score.metric_name: [] for score in corr.scores}
    for prot_id in corr.get_uniprot_ids():
        for score in corr.scores:
            obs_dict[score.metric_name] += score[prot_id]
    obs_df = pd.DataFrame(obs_dict)

    return sns.pairplot(obs_df, corner=True, kind='hist')
