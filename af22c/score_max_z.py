from af22c.load_msa import MultipleSeqAlign
from af22c.similarity_matrix import PROT_SEQ_AAS, get_background_distribution, get_normalized_similarity_matrix
import numpy as np

AA_TO_INT = {aa: index for index, aa in enumerate(PROT_SEQ_AAS)}
INT_TO_AA = {index: aa for index, aa in enumerate(PROT_SEQ_AAS)}
DEFAULT_SIMILARITY_MATRIX = get_normalized_similarity_matrix()


def calc_relative_freqs(msa: MultipleSeqAlign, colidx: int) -> np.ndarray:
    J = len(PROT_SEQ_AAS)
    res = np.zeros(J)
    for match in msa.matches:
        aa = match[colidx]
        if aa in PROT_SEQ_AAS:
            res[AA_TO_INT[aa]] += 1
    return res


def calc_covariance_matrix(
    relative_freqs: np.ndarray, bg_dist: np.ndarray, num_observed_symbols: int
) -> np.ndarray:
    assert relative_freqs.ndim == 1
    assert bg_dist.ndim == 1

    J = len(PROT_SEQ_AAS)
    res = np.empty((J, J))
    for i, j in np.ndindex(J, J):
        kronecker = 1.0 if i == j else 0.0
        res[i, j] = (
            relative_freqs[i] * (kronecker - relative_freqs[j])
        ) / num_observed_symbols
    return res


def calc_max_z(
    msa: MultipleSeqAlign, sim_matrix=DEFAULT_SIMILARITY_MATRIX, bg_dist=None
) -> list[float]:
    """
    Calculate the maxZ score introduced by [1].

    [1] Ahola, V., Aittokallio, T., Vihinen, M. et al. A statistical score for assessing the quality of multiple
    sequence alignments. BMC Bioinformatics 7, 484 (2006). https://doi.org/10.1186/1471-2105-7-484
    """
    # TODO: add citation to README.md?
    # NOTE: this code is taken from https://github.com/mcm2020/average-maxZ-score/blob/master/averagemaxz/maxz.py
    if not bg_dist:
        bg_dist = get_background_distribution(sim_matrix)

    # constants
    J = len(PROT_SEQ_AAS)  # number of available symbols

    # actual maxZ score calculation
    Zs = np.zeros(len(msa.query_seq))
    for colidx in range(len(msa.query_seq)):
        # calculate observed symbol frequencies
        sym_freqs = {aa: 0 for aa in PROT_SEQ_AAS}  # symbol frequencies n_i
        if msa.query_seq[colidx] not in PROT_SEQ_AAS:
            raise ValueError("Query sequence contains a non AA residue.")
        sym_freqs[msa.query_seq[colidx]] = 1  # Add residue appearance from query.
        for match in msa.matches:
            sym = match.aligned_seq[colidx]
            if sym in PROT_SEQ_AAS:
                sym_freqs[sym] += 1

        n = sum(sym_freqs.values())  # actual number of symbols observed, N - gaps
        b = np.array(
            [sym_freq / n for sym_freq in sym_freqs.values()]
        )  # maximum likelihood estimator for beta vector

        relative_freqs = calc_relative_freqs(msa, colidx)
        covariance_matrix = calc_covariance_matrix(bg_dist, bg_dist, n)

        # calculate actual Z scores
        Zis = np.zeros(J)
        for i in range(J):
            ci = sim_matrix[i, :]  # c_i is "the ith row of C"
            numerator = ci.T @ (b - bg_dist)
            denominator = np.sqrt(ci.T @ covariance_matrix @ ci)
            Zi = numerator / denominator
            Zis[i] = Zi
        Z = np.max(Zis)
        Zs[colidx] = Z
    return Zs.tolist()
