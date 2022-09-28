#!/usr/bin/env python3

import os
import json


def load_plddt_scores(path):
    """Load pLDDT scores into a `dict` from a `.json` file mapping UniProt identifiers to a list o per-residue pLDDT
    scores"""
    with open(path) as infile:
        return json.load(infile)


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
    import itertools
    import matplotlib.pyplot as plt

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

    print("plotting results")
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    for prot_id in itertools.islice(shared_prot_ids, 10):
        prot_plddts = plddts[prot_id]
        prot_pred_dis = seth_preds[prot_id]

        ax.scatter(range(len(prot_plddts)), prot_plddts, label=prot_id)
    ax.legend(title="per-residue pLDDT")
    plt.show()
