import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from af22c.proteome import ProteomewidePerResidueMetric, ProteomeCorrelation, ProteomeNeffsMMseqs


def plot_per_protein_score_distribution(
    ax: plt.Axes,
    score: ProteomewidePerResidueMetric,
    uniprot_id: str,
    **kwargs
):
    limits = score.limits
    if limits:
        ax.set_ylim(limits[0], limits[1])

    df = pd.DataFrame({score.metric_name: score[uniprot_id]})
    sns.boxplot(data=df, y=score.metric_name, color=score.color, **kwargs)


def plot_multiple_scores_on_multiple_axis(
    axs: tuple[plt.Axes],
    scores: list[ProteomewidePerResidueMetric],
    uniprot_id: str
):
    df = pd.DataFrame({score.metric_name: score[uniprot_id] for score in scores})
    for score, ax in zip(scores, axs):
        ax.set_xlim(0, len(score[uniprot_id]))
        if score.limits:
            ax.set_ylim(*score.limits)
        sns.lineplot(data=df, x=df.index, y=score.metric_name, ax=ax, color=score.color)


def plot_multiple_scores_on_one_axis(
    ax: plt.Axes,
    scores: list[ProteomewidePerResidueMetric],
    uniprot_id: str,
):
    """Plot multiple scores in one diagram. Scores are normalized to [0,1]."""
    for score in scores:
        items = score[uniprot_id]
        name, color, limits = score.metric_name, score.color, score.limits

        # TODO: hack, for some reason, mmseqs items are still too long -- crop them if possible. when this issue is
        # resolved, this code can go.
        if isinstance(score, ProteomeNeffsMMseqs):
            other_items = next(t[uniprot_id] for t in scores if not isinstance(t, ProteomeNeffsMMseqs))
            if other_items:
                items = items[:len(other_items)]

        # try to normalize items first by given limits, then by dataset extrema
        upper, lower = max(items), min(min(items), 0)
        if isinstance(limits, tuple):
            if limits[0] is not None:
                lower = limits[0]
            if limits[1] is not None:
                upper = limits[1]
        items = [(item - lower) / (upper - lower) for item in items]

        ax.plot(items, color=color, label=f"{name}")
    ax.legend()
    ax.set_title("Normalized per-residue scores for %s" % uniprot_id)
    ax.set_xlabel("Residue index in protein sequence")


def plot_pairwise_correlation(ax: plt.Axes, corr: ProteomeCorrelation, uniprot_id: str):
    corr_df = corr.get_pearson_corr(uniprot_id)
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_df, annot=True, mask=mask, cmap=cmap, vmin=-1, vmax=1, ax=ax)


def plot_pairwise_scatter(corr: ProteomeCorrelation, uniprot_id: str) -> sns.PairGrid:
    corr_df = corr.generate_observation_df(uniprot_id)
    return sns.pairplot(corr_df, corner=True)
