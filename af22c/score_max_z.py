from af22c.load_msa import MultipleSeqAlign
import numpy as np

# TODO: move this to a constants file
# A list of standard amino acid names
# NOTE: taken from https://biopython.org/docs/1.76/api/Bio.Alphabet.IUPAC.html#Bio.Alphabet.IUPAC.IUPACProtein
PROT_SEQ_AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INT = {aa: index for index, aa in enumerate(PROT_SEQ_AAS)}
INT_TO_AA = {index: aa for index, aa in enumerate(PROT_SEQ_AAS)}
DEFAULT_SIMILARITY_MATRIX = np.array(
    [
        [
            0.0215,
            0.0016,
            0.0022,
            0.003,
            0.0016,
            0.0058,
            0.0011,
            0.0032,
            0.0033,
            0.0044,
            0.0013,
            0.0019,
            0.0022,
            0.0019,
            0.0023,
            0.0063,
            0.0037,
            0.0051,
            0.0004,
            0.0013,
        ],
        [
            0.0016,
            0.0119,
            0.0004,
            0.0004,
            0.0005,
            0.0008,
            0.0002,
            0.0011,
            0.0005,
            0.0016,
            0.0004,
            0.0004,
            0.0004,
            0.0003,
            0.0004,
            0.001,
            0.0009,
            0.0014,
            0.0001,
            0.0003,
        ],
        [
            0.0022,
            0.0004,
            0.0213,
            0.0049,
            0.0008,
            0.0025,
            0.001,
            0.0012,
            0.0024,
            0.0015,
            0.0005,
            0.0037,
            0.0012,
            0.0016,
            0.0016,
            0.0028,
            0.0019,
            0.0013,
            0.0002,
            0.0006,
        ],
        [
            0.003,
            0.0004,
            0.0049,
            0.0161,
            0.0009,
            0.0019,
            0.0014,
            0.0012,
            0.0041,
            0.002,
            0.0007,
            0.0022,
            0.0014,
            0.0035,
            0.0027,
            0.003,
            0.002,
            0.0017,
            0.0003,
            0.0009,
        ],
        [
            0.0016,
            0.0005,
            0.0008,
            0.0009,
            0.0183,
            0.0012,
            0.0008,
            0.003,
            0.0009,
            0.0054,
            0.0012,
            0.0008,
            0.0005,
            0.0005,
            0.0009,
            0.0012,
            0.0012,
            0.0026,
            0.0008,
            0.0042,
        ],
        [
            0.0058,
            0.0008,
            0.0025,
            0.0019,
            0.0012,
            0.0378,
            0.001,
            0.0014,
            0.0025,
            0.0021,
            0.0007,
            0.0029,
            0.0014,
            0.0014,
            0.0017,
            0.0038,
            0.0022,
            0.0018,
            0.0004,
            0.0008,
        ],
        [
            0.0011,
            0.0002,
            0.001,
            0.0014,
            0.0008,
            0.001,
            0.0093,
            0.0006,
            0.0012,
            0.001,
            0.0004,
            0.0014,
            0.0005,
            0.001,
            0.0012,
            0.0011,
            0.0007,
            0.0006,
            0.0002,
            0.0015,
        ],
        [
            0.0032,
            0.0011,
            0.0012,
            0.0012,
            0.003,
            0.0014,
            0.0006,
            0.0184,
            0.0016,
            0.0114,
            0.0025,
            0.001,
            0.001,
            0.0009,
            0.0012,
            0.0017,
            0.0027,
            0.012,
            0.0004,
            0.0014,
        ],
        [
            0.0033,
            0.0005,
            0.0024,
            0.0041,
            0.0009,
            0.0025,
            0.0012,
            0.0016,
            0.0161,
            0.0025,
            0.0009,
            0.0024,
            0.0016,
            0.0031,
            0.0062,
            0.0031,
            0.0023,
            0.0019,
            0.0003,
            0.001,
        ],
        [
            0.0044,
            0.0016,
            0.0015,
            0.002,
            0.0054,
            0.0021,
            0.001,
            0.0114,
            0.0025,
            0.0371,
            0.0049,
            0.0014,
            0.0014,
            0.0016,
            0.0024,
            0.0024,
            0.0033,
            0.0095,
            0.0007,
            0.0022,
        ],
        [
            0.0013,
            0.0004,
            0.0005,
            0.0007,
            0.0012,
            0.0007,
            0.0004,
            0.0025,
            0.0009,
            0.0049,
            0.004,
            0.0005,
            0.0004,
            0.0007,
            0.0008,
            0.0009,
            0.001,
            0.0023,
            0.0002,
            0.0006,
        ],
        [
            0.0019,
            0.0004,
            0.0037,
            0.0022,
            0.0008,
            0.0029,
            0.0014,
            0.001,
            0.0024,
            0.0014,
            0.0005,
            0.0141,
            0.0009,
            0.0015,
            0.002,
            0.0031,
            0.0022,
            0.0012,
            0.0002,
            0.0007,
        ],
        [
            0.0022,
            0.0004,
            0.0012,
            0.0014,
            0.0005,
            0.0014,
            0.0005,
            0.001,
            0.0016,
            0.0014,
            0.0004,
            0.0009,
            0.0191,
            0.0008,
            0.001,
            0.0017,
            0.0014,
            0.0012,
            0.0001,
            0.0005,
        ],
        [
            0.0019,
            0.0003,
            0.0016,
            0.0035,
            0.0005,
            0.0014,
            0.001,
            0.0009,
            0.0031,
            0.0016,
            0.0007,
            0.0015,
            0.0008,
            0.0073,
            0.0025,
            0.0019,
            0.0014,
            0.0012,
            0.0002,
            0.0007,
        ],
        [
            0.0023,
            0.0004,
            0.0016,
            0.0027,
            0.0009,
            0.0017,
            0.0012,
            0.0012,
            0.0062,
            0.0024,
            0.0008,
            0.002,
            0.001,
            0.0025,
            0.0178,
            0.0023,
            0.0018,
            0.0016,
            0.0003,
            0.0009,
        ],
        [
            0.0063,
            0.001,
            0.0028,
            0.003,
            0.0012,
            0.0038,
            0.0011,
            0.0017,
            0.0031,
            0.0024,
            0.0009,
            0.0031,
            0.0017,
            0.0019,
            0.0023,
            0.0126,
            0.0047,
            0.0024,
            0.0003,
            0.001,
        ],
        [
            0.0037,
            0.0009,
            0.0019,
            0.002,
            0.0012,
            0.0022,
            0.0007,
            0.0027,
            0.0023,
            0.0033,
            0.001,
            0.0022,
            0.0014,
            0.0014,
            0.0018,
            0.0047,
            0.0125,
            0.0036,
            0.0003,
            0.0009,
        ],
        [
            0.0051,
            0.0014,
            0.0013,
            0.0017,
            0.0026,
            0.0018,
            0.0006,
            0.012,
            0.0019,
            0.0095,
            0.0023,
            0.0012,
            0.0012,
            0.0012,
            0.0016,
            0.0024,
            0.0036,
            0.0196,
            0.0004,
            0.0015,
        ],
        [
            0.0004,
            0.0001,
            0.0002,
            0.0003,
            0.0008,
            0.0004,
            0.0002,
            0.0004,
            0.0003,
            0.0007,
            0.0002,
            0.0002,
            0.0001,
            0.0002,
            0.0003,
            0.0003,
            0.0003,
            0.0004,
            0.0065,
            0.0009,
        ],
        [
            0.0013,
            0.0003,
            0.0006,
            0.0009,
            0.0042,
            0.0008,
            0.0015,
            0.0014,
            0.001,
            0.0022,
            0.0006,
            0.0007,
            0.0005,
            0.0007,
            0.0009,
            0.001,
            0.0009,
            0.0015,
            0.0009,
            0.0102,
        ],
    ]
)


def calc_background_distribution(sim_matrix: np.ndarray) -> np.ndarray:
    return np.sum(sim_matrix, axis=0)


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
        bg_dist = calc_background_distribution(sim_matrix)

    # constants
    J = len(PROT_SEQ_AAS)  # number of available symbols

    # actual maxZ score calculation
    Zs = np.zeros(len(msa.query_seq))
    for colidx in range(len(msa.query_seq)):
        # calculate observed symbol frequencies
        sym_freqs = {aa: 0 for aa in PROT_SEQ_AAS}  # symbol frequencies n_i
        if msa.query_seq[colidx] not in PROT_SEQ_AAS:
            # TODO how should we handle this?
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
