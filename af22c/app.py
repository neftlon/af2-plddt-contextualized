#!/usr/bin/env python3

import streamlit as st
from af22c.plots import *
from af22c.proteome import *


if __name__ == "__main__":
    sns.set_theme(context="paper", style="whitegrid", palette="deep")

    proteome_name = "UP000005640_9606"
    neffs = ProteomeNeffs.from_directory(f"data/cluster/{proteome_name}/neffs")
    neffs_naive = ProteomeNeffsNaive.from_directory(f"data/cluster/{proteome_name}/neffs_naive")
    plddts = ProteomePLDDTs.from_file(f"data/{proteome_name}_HUMAN_v3_plddts_fltrd.json")
    seths = ProteomeSETHPreds.from_file("data/Human_SETH_preds.txt")
    msa_sizes = ProteomeMSASizes.from_file(f"data/cluster/{proteome_name}_msa_size.csv")
    correlation = ProteomeCorrelation([plddts, seths, neffs, neffs_naive], msa_sizes)

    # look into the overlap between the UniProt identifiers of the source datasets
    shared_prot_ids = correlation.get_uniprot_ids()

    with st.sidebar:
        """## Parameters"""
        prot_id = st.selectbox(
            "Which protein would you like to look at?", shared_prot_ids
        )
        proteome_wide = st.checkbox("Show proteome wide analysis")

    """## Per-protein metrics"""

    colors = ["blue", "orange", "green", "brown"]
    for score, column, color in zip(correlation.scores, st.columns(4), colors):
        metric_name = score.metric_name
        with column:
            f"""### {metric_name}"""

            # TODO: maybe remove precision from Neffs and Neff scores
            st.metric(f"mean {metric_name}", f"{np.mean(score[prot_id]):0.04f}")
            st.metric(f"std {metric_name}", f"{np.std(score[prot_id]):0.04f}")
            fig, ax = plt.subplots()
            fig.set_size_inches(0.5, 2.5)

            plot_per_protein_score_distribution(ax, score, prot_id, color=color)

            st.pyplot(fig)

    """## Per-residue scores"""
    fig, axs = plt.subplots(nrows=4, figsize=(8, 8))
    plot_multiple_scores_in_one(axs, correlation.scores, prot_id, colors)
    st.pyplot(fig)

    """
    ## Correlation matrix
    
    Pairwise Pearson correlation coefficient between two values.
    
    White means a correlation value of `1.0`, darker patches mean less correlation.
    
    Values are rounded to two digits.
    """

    fig, ax = plt.subplots()
    plot_pairwise_correlation(ax, correlation, prot_id)
    fig.tight_layout()
    st.pyplot(fig)

    """
    ## Pairwise scatterplots of all metrics
    """
    fig = plot_pairwise_scatter(correlation, prot_id)
    fig.tight_layout()
    st.pyplot(fig)

    if proteome_wide:
        n_mismatch_res = 0  # TODO: obtain this from proteome module

        # TODO: add very cool heatmap plots
        """
        ### Disregarded proteins
        """
        col_left, col_right = st.columns(2)
        with col_left:
            n_missing_seth = len(plddts.get_uniprot_ids() - seths.get_uniprot_ids())
            st.metric(
                "UniProt identifiers only appearing in \npLDDT scores", n_missing_seth
            )

            n_missing_af2 = len(seths.get_uniprot_ids() - plddts.get_uniprot_ids())
            st.metric(
                "UniProt identifiers only appearing in \nSETH predictions",
                n_missing_af2,
            )

            st.metric(
                "Number of proteins disregarded due to mismatch in number of residues:",
                n_mismatch_res,
            )

        with col_right:
            num_prot_ids = len(plddts.get_uniprot_ids() | seths.get_uniprot_ids())
            st.metric("Total number of UniProt identifiers", num_prot_ids)
            sum_disregarded = (
                num_prot_ids - n_missing_af2 - n_missing_seth - n_mismatch_res
            )
            st.metric("Number of used UniProt identifiers", sum_disregarded)