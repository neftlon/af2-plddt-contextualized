import pandas as pd
import numpy as np

BLOSUM62_probs = pd.read_table("data/blosum62qij.txt", sep="\\s+", comment="#")
BLOSUM62_probs.index = BLOSUM62_probs.columns
BLOSUM62_probs = pd.DataFrame(np.tril(BLOSUM62_probs.values) + np.triu(BLOSUM62_probs.values.T, 1),
                              columns=BLOSUM62_probs.columns,
                              index=BLOSUM62_probs.index)
PROT_SEQ_AAS = BLOSUM62_probs.index


def get_normalized_similarity_matrix() -> np.ndarray:
    # TODO return dataframe and refactor access in similarity_matrix.py
    return BLOSUM62_probs.values


def get_background_distribution(normalized_sim_matrix: np.ndarray) -> np.ndarray:
    return np.sum(BLOSUM62_probs.values, axis=0)
