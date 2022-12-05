import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statistics


def plot_pairwise_correlation(
    ax: plt.Axes, datasets: list[list[float]], labels: list[str]
):
    """
    Plot pairwise correlation between datasets.

    Raises a `ValueError` if the supplied number of datasets does not match the supplied number of labels.
    """
    if len(datasets) != len(labels):
        raise ValueError(
            f"number of datasets ({len(datasets)}) does not match number of labels ({len(labels)}"
        )

    # calculate pairwise correlation
    pairwise_correlation = np.zeros((4, 4))
    for i, xs in enumerate(datasets):
        for j, ys in enumerate(datasets):
            pairwise_correlation[i, j] = round(statistics.correlation(xs, ys), 2)

    # plot pairwise correlation
    color_map = LinearSegmentedColormap.from_list("", ["yellow", "blue", "yellow"])
    im = ax.imshow(
        pairwise_correlation, cmap=color_map, vmin=-1, vmax=1
    )  # plot correlation matrix

    ax.set_xticks(np.arange(len(datasets)), labels=labels)
    ax.set_yticks(np.arange(len(datasets)), labels=labels)
    ax.grid(False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(datasets)):
        for j in range(len(datasets)):
            corr = pairwise_correlation[i, j]
            text = ax.text(
                j, i, corr, ha="center", va="center", color=(0.5 - 0.5 * corr,) * 3
            )
