import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from af22c.proteome import ProteomewidePerResidueMetric, ProteomeCorrelation


def plot_per_protein_score_distribution(
    ax: plt.Axes,
    score: ProteomewidePerResidueMetric,
    uniprot_id: str,
    # TODO: scores should keep track of their limits/colors themselves
    limits: tuple[float, float] = None,
    **kwargs
):
    if limits:
        ax.set_ylim(limits[0], limits[1])

    df = pd.DataFrame({score.metric_name: score[uniprot_id]})
    sns.boxplot(data=df, y=score.metric_name, **kwargs)


def plot_multiple_scores_in_one(
    axs: tuple[plt.Axes],
    scores: list[ProteomewidePerResidueMetric],
    uniprot_id: str,
    # TODO: scores should keep track of their limits/colors themselves
    limits: dict[str: tuple[float, float]],
    colors: dict[str: str],
):
    df = pd.DataFrame({score.metric_name: score[uniprot_id] for score in scores})
    for score, ax in zip(scores, axs):
        ax.set_xlim(0, len(score[uniprot_id]))
        if score.metric_name in limits and limits[score.metric_name]:
            ax.set_ylim(*limits[score.metric_name])
        color = colors[score.metric_name] if score.metric_name in colors else None
        sns.lineplot(data=df, x=df.index, y=score.metric_name, ax=ax, color=color)


def plot_pairwise_correlation(ax: plt.Axes, corr: ProteomeCorrelation, uniprot_id: str):
    corr_df = corr.get_pearson_corr(uniprot_id)
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_df, annot=True, mask=mask, cmap=cmap, vmin=-1, vmax=1, ax=ax)


def plot_pairwise_scatter(corr: ProteomeCorrelation, uniprot_id: str) -> sns.PairGrid:
    corr_df = corr.generate_observation_df(uniprot_id)
    return sns.pairplot(corr_df, corner=True)
