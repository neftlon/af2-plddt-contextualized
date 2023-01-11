import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from af22c.proteome import ProteomeCorrelation


def plot_correlation_boxplots(corr: ProteomeCorrelation):
    p_corr_array, _ = corr.get_pearson_corr_stack()
    n = len(corr.scores)
    fig, axs = plt.subplots(n, n)
    for i in range(1, n):
        for j in range(0, i):
            sns.boxplot(p_corr_array[:, i, j].flatten(), ax=axs[i, j])
    return fig


def plot_correlation_histograms(corr: ProteomeCorrelation):
    p_corr_array, _ = corr.get_pearson_corr_stack()
    n = len(corr.scores)
    fig, axs = plt.subplots(n, n)
    for i in range(1, n):
        for j in range(0, i):
            sns.histplot(p_corr_array[:, i, j].flatten(), ax=axs[i, j])
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
