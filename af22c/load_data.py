#!/usr/bin/env python3

import os
import json
import streamlit as st
import statistics

from af22c.load_msa import calc_naive_neff_by_id
from af22c.neff_cache_or_calc import NeffCacheOrCalc
from af22c.plot_pairwise_correlation import plot_pairwise_correlation


@st.experimental_singleton(suppress_st_warning=True)
def load_plddt_scores(path):
    """Load pLDDT scores into a `dict` from a `.json` file mapping UniProt identifiers to a list o per-residue pLDDT
    scores"""
    with open(path) as infile:
        return json.load(infile)


@st.experimental_singleton
def load_seth_preds(path):
    """Load SETH predictions into a `dict` mapping UniProt identifiers to a list of per-residue CheZOD scores"""
    with open(path) as infile:
        lines = infile.readlines()
        headers = lines[::2]
        disorders = lines[1::2]
        proteome_seth_preds = {}
        for header, disorder in zip(headers, disorders):
            uniprot_id = header.split("|")[1]
            disorder = list(map(float, disorder.split(", ")))
            proteome_seth_preds[uniprot_id] = disorder
        return proteome_seth_preds


def compute_proteome_wide_corr(path, plddts, seth_preds, shared_prot_ids):
    # Count how many proteins do not have equal number of residues in plddts and seth_preds.
    n_mismatch_res = 0
    mismatched_prots = []
    for prot_id in shared_prot_ids:
        if len(plddts[prot_id]) != len(seth_preds[prot_id]):
            n_mismatch_res = n_mismatch_res + 1
            mismatched_prots.append(
                (prot_id, len(plddts[prot_id]), len(seth_preds[prot_id]))
            )
    st.table(mismatched_prots)

    # Initialize stat vectors
    n = len(shared_prot_ids) - n_mismatch_res
    pearson = np.zeros(n)
    spearman_rho = np.zeros(n)
    spearman_pval = np.zeros(n)

    i = 0
    for prot_id in shared_prot_ids:
        # Construct observation matrix for current protein ID
        prot_plddts = np.array(plddts[prot_id])
        prot_pred_dis = np.array(seth_preds[prot_id])
        if prot_plddts.shape != prot_pred_dis.shape:
            continue
        obs_mat = np.stack([prot_plddts, prot_pred_dis], axis=0)

        # Compute stats
        pearson[i] = np.corrcoef(obs_mat)[0, 1]
        spearman_rho[i], spearman_pval[i] = stats.spearmanr(prot_plddts, prot_pred_dis)

        i = i + 1

    proteome_wide_stats_dict = {
        "pearson": pearson.tolist(),
        "spearman_rho": spearman_rho.tolist(),
        "spearman_pval": spearman_pval.tolist(),
        "n_mismatch_res": n_mismatch_res,
    }
    with open(path, "w") as outfile:
        json.dump(proteome_wide_stats_dict, outfile)
    return proteome_wide_stats_dict


def load_proteome_wide_corr(path, plddts, seth_preds, shared_prot_ids):
    try:
        with open(path) as infile:
            proteome_wide_stats_dict = json.load(infile)
            return proteome_wide_stats_dict
    except:
        return compute_proteome_wide_corr(path, plddts, seth_preds, shared_prot_ids)


if __name__ == "__main__":
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", palette="deep")

    data_dir = "./data"
    proteome_name = "UP000005640_9606_HUMAN_v3"
    proteome_wide_stats_ending = "_proteome_stats.json"
    plddts_fltrd_ending = "_plddts_fltrd.json"
    plddts_filename = proteome_name + plddts_fltrd_ending
    seth_preds_filename = "Human_SETH_preds.txt"

    neff_src = NeffCacheOrCalc(
        proteome_filename="data/UP000005640_9606.tar",
        cache_filename="data/UP000005640_9606_neff_cache.tar",
    )

    print("loading per-protein scores")
    plddts = load_plddt_scores(os.path.join(data_dir, plddts_filename))
    seth_preds = load_seth_preds(os.path.join(data_dir, seth_preds_filename))

    # look into the overlap between the UniProt identifiers of the source datasets
    plddts_ids = set(plddts.keys())
    seth_preds_ids = set(seth_preds.keys())
    shared_prot_ids = plddts_ids & seth_preds_ids

    with st.sidebar:
        """## Parameters"""
        prot_id = st.selectbox(
            "Which protein would you like to look at?", shared_prot_ids
        )
        prot_plddts = np.array(plddts[prot_id])
        prot_pred_dis = np.array(seth_preds[prot_id])
        prot_neffs = np.array(neff_src.get_neffs(prot_id))
        prot_neffs_naive = np.array(
            calc_naive_neff_by_id(neff_src.proteome_filename, prot_id)
        )
        proteome_wide = st.checkbox("Show proteome wide analysis")

    # Construct DataFrame for visualization with seaborn
    df = pd.DataFrame(
        np.array(
            [prot_plddts, prot_pred_dis, prot_neffs, prot_neffs_naive]
        ).transpose(),
        columns=["pLDDT score", "pred. disorder", "Neff", "Neff naive"],
    )

    """## Per-protein metrics"""

    # TODO(johannes): think about merging these cols into one plot for "comparable" scales
    col_plddts, col_pred_dis, col_neff, col_neff_naive = st.columns(4)
    with col_plddts:
        """### pLDDT"""
        st.metric("mean pLDDT", f"{np.mean(prot_plddts):0.04f}")
        st.metric("std pLDDT", f"{np.std(prot_plddts):0.04f}")
        fig, ax = plt.subplots()
        fig.set_size_inches(0.5, 2.5)
        ax.set_ylim(0, 100)
        sns.boxplot(data=df, y="pLDDT score")
        st.pyplot(fig)

    with col_pred_dis:
        """### Pred. dis."""
        st.metric(f"mean predicted disorder", f"{np.mean(prot_pred_dis):0.04f}")
        st.metric(f"std predicted disorder", f"{np.std(prot_pred_dis):0.04f}")
        fig, ax = plt.subplots()
        fig.set_size_inches(0.5, 2.5)
        ax.set_ylim(-20, 20)
        sns.boxplot(data=df, y="pred. disorder", color="orange")
        st.pyplot(fig)

    with col_neff:
        """### Neff"""
        st.metric(f"mean Neff", f"{int(np.mean(prot_neffs))}")
        st.metric(f"std Neff", f"{int(np.std(prot_neffs))}")
        fig, ax = plt.subplots()
        fig.set_size_inches(0.5, 2.5)
        sns.boxplot(data=df, y="Neff", color="green")
        st.pyplot(fig)

    with col_neff_naive:
        """### Neff naive"""
        st.metric(f"mean Neff naive", f"{int(np.mean(prot_neffs_naive))}")
        st.metric(f"std Neff naive", f"{int(np.std(prot_neffs_naive))}")
        fig, ax = plt.subplots()
        fig.set_size_inches(0.5, 2.5)
        sns.boxplot(data=df, y="Neff naive", color="brown")
        st.pyplot(fig)

    """## Per-residue scores"""
    fig, (ax_plddt, ax_pred_dis, ax_neff, ax_neff_naive) = plt.subplots(
        nrows=4, figsize=(8, 8)
    )
    ax_plddt.set_xlim(0, len(prot_plddts))
    ax_plddt.set_ylim(0, 100)
    sns.lineplot(data=df, x=df.index, y="pLDDT score", ax=ax_plddt)
    ax_pred_dis.set_xlim(0, len(prot_pred_dis))
    ax_pred_dis.set_ylim(-20, 20)
    sns.lineplot(
        data=df, x=df.index, y="pred. disorder", ax=ax_pred_dis, color="orange"
    )
    ax_neff.set_xlim(0, len(prot_neffs))
    sns.lineplot(data=df, x=df.index, y="Neff", ax=ax_neff, color="green")
    ax_neff_naive.set_xlim(0, len(prot_neffs_naive))
    sns.lineplot(data=df, x=df.index, y="Neff naive", ax=ax_neff_naive, color="brown")
    st.pyplot(fig)

    """
    ## Correlation matrix
    
    Pairwise Pearson correlation coefficient between two values.
    
    White means a correlation value of `1.0`, darker patches mean less correlation.
    
    Values are rounded to two digits.
    """

    fig, ax = plt.subplots()

    values = [prot_plddts, prot_pred_dis, prot_neffs, prot_neffs_naive]
    labels = ["pLDDT", "pred. dis.", "Neff", "Neff naive"]

    plot_pairwise_correlation(ax, values, labels)

    fig.tight_layout()
    st.pyplot(fig)

    """
    ## Scatterplot of pLDDT and predicted disorder
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(-20, 20)
    sns.scatterplot(df, x="pLDDT score", y="pred. disorder")
    st.pyplot(fig)

    """
    ## Dataset pairwise correlation

    The following stats show the correlation between the pLDDT and the SETH predictions for the selected protein.
    """

    obs_mat = np.stack([prot_plddts, prot_pred_dis], axis=0)

    def covariance():
        cov_mat = np.cov(obs_mat)
        st.metric("Covariance", f"{cov_mat[0, 1]:0.04f}")

    def pearson_corr():
        corr_mat = np.corrcoef(obs_mat)
        st.metric("Pearson correlation", f"{corr_mat[0, 1]:0.04f}")

    def spearman_corr():
        rho, pval = stats.spearmanr(prot_plddts, prot_pred_dis)
        st.metric("Spearman correlation rho", f"{rho:0.04f}")
        st.metric("Spearman p-value", f"{pval:0.04f}")

    col_left, col_right = st.columns(2)
    with col_left:
        covariance()
        pearson_corr()

    with col_right:
        spearman_corr()

    if proteome_wide:
        proteome_stats_path = os.path.join(
            data_dir, proteome_name + proteome_wide_stats_ending
        )
        proteome_wide_stats_dict = load_proteome_wide_corr(
            proteome_stats_path, plddts, seth_preds, shared_prot_ids
        )
        n_mismatch_res = proteome_wide_stats_dict.pop("n_mismatch_res")
        proteome_wide_stats = pd.DataFrame(proteome_wide_stats_dict)

        """
        ## Proteome wide distribution of correlation

        The following stats show how the pairwise correlation between pLDDT and SETH predictions is distributed over the whole proteome.
        """
        col_left, col_right = st.columns(2)
        with col_left:
            """### Pearson"""
            st.metric(
                "mean Pearson correlation",
                f"{np.median(proteome_wide_stats['pearson']):0.04f}",
            )
            st.metric(
                "std Pearson correlation",
                f"{np.std(proteome_wide_stats['pearson']):0.04f}",
            )

            fig, ax = plt.subplots()
            fig.set_size_inches(1, 4)
            ax.set_ylim(-1, 1)
            sns.boxplot(data=proteome_wide_stats, y="pearson")
            st.pyplot(fig)

        with col_right:
            """### Spearman"""
            st.metric(
                "mean Spearman correlation",
                f"{np.median(proteome_wide_stats['spearman_rho']):0.04f}",
            )
            st.metric(
                "std Spearman correlation",
                f"{np.std(proteome_wide_stats['spearman_rho']):0.04f}",
            )

            fig, ax = plt.subplots()
            ax.set_ylim(-1, 1)
            fig.set_size_inches(1, 4)
            sns.boxplot(data=proteome_wide_stats, y="spearman_rho")
            st.pyplot(fig)

            st.metric(
                "mean Spearman p-value",
                f"{np.median(proteome_wide_stats['spearman_pval']):0.04f}",
            )
            st.metric(
                "std Spearman p-value",
                f"{np.std(proteome_wide_stats['spearman_pval']):0.04f}",
            )

            fig, ax = plt.subplots()
            ax.set_ylim(0, 1)
            fig.set_size_inches(1, 4)
            sns.boxplot(data=proteome_wide_stats, y="spearman_pval")
            st.pyplot(fig)

        """
        ### Disregarded proteins
        """
        col_left, col_right = st.columns(2)
        with col_left:
            n_missing_seth = len(plddts_ids - seth_preds_ids)
            st.metric(
                "UniProt identifiers only appearing in \npLDDT scores", n_missing_seth
            )

            n_missing_af2 = len(seth_preds_ids - plddts_ids)
            st.metric(
                "UniProt identifiers only appearing in \nSETH predictions",
                n_missing_af2,
            )

            st.metric(
                "Number of proteins disregarded due to mismatch in number of residues:",
                n_mismatch_res,
            )

        with col_right:
            num_prot_ids = len(plddts_ids | seth_preds_ids)
            st.metric("Total number of UniProt identifiers", num_prot_ids)
            sum_disregarded = (
                num_prot_ids - n_missing_af2 - n_missing_seth - n_mismatch_res
            )
            st.metric("Number of used UniProt identifiers", sum_disregarded)
