#!/usr/bin/env python3

import os
import json
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


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    data_dir = "./data"
    plddts_filename = "UP000005640_9606_HUMAN_v3_plddts.json"
    seth_preds_filename = "Human_SETH_preds.txt"

    print("loading per-protein scores")
    plddts = load_plddt_scores(os.path.join(data_dir, plddts_filename))
    seth_preds = load_seth_preds(os.path.join(data_dir, seth_preds_filename))

    # look into the overlap between the UniProt identifiers of the source datasets
    plddts_ids = set(plddts.keys())
    seth_preds_ids = set(seth_preds.keys())
    shared_prot_ids = plddts_ids & seth_preds_ids
    only_once = plddts_ids ^ seth_preds_ids
    if only_once:
        num_prot_ids = len(plddts_ids | seth_preds_ids)
        print(f" {len(plddts_ids - seth_preds_ids)}/{num_prot_ids} UniProt identifiers appear only in pLDDT scores "
              f"file, but not in the SETH predictions")
        print(f" {len(seth_preds_ids - plddts_ids)}/{num_prot_ids} UniProt identifiers appear only in SETH "
              f"predictions, but not in the pLDDT scores file")
    else:
        print("per-protein scores have have equal UniProt identifiers, nice!")

    """## Parameters"""
    prot_id = st.selectbox("Which protein would you like to look at?", shared_prot_ids)
    prot_plddts = plddts[prot_id]
    prot_pred_dis = seth_preds[prot_id]
    scale_factor = st.slider("Which scale factor should be applied to the predicted disorder score?", 0.1, 10.0, 5.0, 0.1)  # to map from range [0;20] to [0;100]
    scaled_prot_pred_dis = scale_factor * np.array(prot_pred_dis)

    """## Per-protein metrics"""
    st.metric("mean pLDDT", np.mean(prot_plddts))
    st.metric(f"mean scaled {scale_factor}x predicted disorder", np.mean(scaled_prot_pred_dis))

    """## Per-residue metrics"""
    n = len(prot_plddts)
    df = pd.DataFrame(
        np.array([prot_plddts, scaled_prot_pred_dis]).transpose(),
        columns=["pLDDT score", "scaled pred. disorder"],
    )
    st.line_chart(df)
