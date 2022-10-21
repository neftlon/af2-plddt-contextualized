#!/usr/bin/env python3

import os
import json
from tkinter import Y
import streamlit as st


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


@st.experimental_singleton
def compute_proteome_wide_corr(plddts, seth_preds, shared_prot_ids):
    # Count how many proteins do not have equal number of residues in plddts and seth_preds.
    n_mismatch_res = 0
    mismatched_prots = []
    for prot_id in shared_prot_ids:
        if len(plddts[prot_id]) != len(seth_preds[prot_id]):
            n_mismatch_res = n_mismatch_res + 1
            mismatched_prots.append((prot_id, len(plddts[prot_id]), len(seth_preds[prot_id])))
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
            # TODO Fix! AF2 predictions are partitioned in fragments of 1400 residues, if the protein has more than 2700 residues. We have to recombine them first. Here is a workaround that disregards those for now. We also have to fix this in the picker. (See varadi2022_AlphaFoldProteinStructure)
            continue
        obs_mat = np.stack([prot_plddts, prot_pred_dis], axis=0)

        # Compute stats
        pearson[i] = np.corrcoef(obs_mat)[0, 1]
        spearman_rho[i], spearman_pval[i] = stats.spearmanr(prot_plddts, prot_pred_dis)

        i = i + 1

    proteome_wide_stats = {"pearson": pearson,
                           "spearman_rho": spearman_rho,
                           "spearman_pval": spearman_pval}
    return (pd.DataFrame(proteome_wide_stats), n_mismatch_res)


if __name__ == "__main__":
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", palette="deep")

    data_dir = "./data"
    plddts_filename = "UP000005640_9606_HUMAN_v3_plddts_defrag.json"
    seth_preds_filename = "Human_SETH_preds.txt"

    print("loading per-protein scores")
    plddts = load_plddt_scores(os.path.join(data_dir, plddts_filename))
    seth_preds = load_seth_preds(os.path.join(data_dir, seth_preds_filename))

    # look into the overlap between the UniProt identifiers of the source datasets
    plddts_ids = set(plddts.keys())
    seth_preds_ids = set(seth_preds.keys())
    shared_prot_ids = plddts_ids & seth_preds_ids

    with st.sidebar:
        """## Parameters"""
        prot_id = st.selectbox("Which protein would you like to look at?", shared_prot_ids)
        prot_plddts = np.array(plddts[prot_id])
        prot_pred_dis = np.array(seth_preds[prot_id])
        proteome_wide = st.checkbox("Show proteome wide analysis")

    # Construct DataFrame for visualization with seaborn
    df = pd.DataFrame(
            np.array([prot_plddts, prot_pred_dis]).transpose(),
            columns=["pLDDT score", "pred. disorder"],
        )

    """## Per-protein metrics"""

    col_plddts, col_pred_dis = st.columns(2)
    with col_plddts:
        """### pLDDT"""
        st.metric("mean pLDDT", f"{np.mean(prot_plddts):0.04f}")
        st.metric("std pLDDT", f"{np.std(prot_plddts):0.04f}")
        fig, ax = plt.subplots()
        ax.set_ylim(0, 100)
        sns.boxplot(data=df, y="pLDDT score")
        st.pyplot(fig)

    with col_pred_dis:
        """### Predicted disorder"""
        st.metric(f"mean predicted disorder",
                  f"{np.mean(prot_pred_dis):0.04f}")
        st.metric(f"std predicted disorder",
                  f"{np.std(prot_pred_dis):0.04f}")
        fig, ax = plt.subplots()
        ax.set_ylim(-20, 20)
        sns.boxplot(data=df, y="pred. disorder")
        st.pyplot(fig)

    """## Per-residue scores"""
    fig, ax = plt.subplots()
    sns.lineplot(df)
    st.pyplot(fig)


    """
    ## Scatterplot of pLDDT and predicted disorder
    """
    fig, ax = plt.subplots()
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
        proteome_wide_stats, n_mismatch_res = compute_proteome_wide_corr(plddts, seth_preds, shared_prot_ids)

        """
        ## Proteome wide distribution of correlation

        The following stats show how the pairwise correlation between pLDDT and SETH predictions is distributed over the whole proteome.
        """
        col_left, col_right = st.columns(2)
        with col_left:
            """### Pearson"""
            st.metric("mean Pearson correlation",
                      f"{np.median(proteome_wide_stats['pearson']):0.04f}")
            st.metric("std Pearson correlation",
                      f"{np.std(proteome_wide_stats['pearson']):0.04f}")

            fig, ax = plt.subplots()
            ax.set_ylim(-1, 1)
            sns.boxplot(data=proteome_wide_stats, y="pearson")
            st.pyplot(fig)

        with col_right:
            """### Spearman"""
            st.metric("mean Spearman correlation",
                      f"{np.median(proteome_wide_stats['spearman_rho']):0.04f}")
            st.metric("std Spearman correlation",
                      f"{np.std(proteome_wide_stats['spearman_rho']):0.04f}")

            fig, ax = plt.subplots()
            ax.set_ylim(-1, 1)
            sns.boxplot(data=proteome_wide_stats, y="spearman_rho")
            st.pyplot(fig)

            st.metric("mean Spearman p-value",
                      f"{np.median(proteome_wide_stats['spearman_pval']):0.04f}")
            st.metric("std Spearman p-value",
                      f"{np.std(proteome_wide_stats['spearman_pval']):0.04f}")

            fig, ax = plt.subplots()
            ax.set_ylim(0, 1)
            sns.boxplot(data=proteome_wide_stats, y="spearman_pval")
            st.pyplot(fig)

        """
        ### Disregarded proteins
        """
        col_left, col_right = st.columns(2)
        with col_left:
            n_missing_seth = len(plddts_ids - seth_preds_ids)
            st.metric("UniProt identifiers only appearing in \nplDDT scores", n_missing_seth)

            n_missing_af2 = len(seth_preds_ids - plddts_ids)
            st.metric("UniProt identifiers only appearing in \nSETH predictions", n_missing_af2)

            st.metric("Number of proteins disregarded due to mismatch in number of residues:", n_mismatch_res)

        with col_right:
            num_prot_ids = len(plddts_ids | seth_preds_ids)
            st.metric("Total number of UniProt identifiers", num_prot_ids)
            sum_disregarded = num_prot_ids - n_missing_af2 - n_missing_seth - n_mismatch_res
            st.metric("Number of used UniProt identifiers", sum_disregarded)
