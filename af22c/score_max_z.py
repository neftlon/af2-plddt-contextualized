from af22c.load_msa import MultipleSeqAlign
import numpy as np

# TODO: move this to a constants file
# A list of standard amino acid names
# NOTE: taken from https://biopython.org/docs/1.76/api/Bio.Alphabet.IUPAC.html#Bio.Alphabet.IUPAC.IUPACProtein
PROT_SEQ_AAS = list("ACDEFGHIKLMNPQRSTVWY")


def calc_max_z(proteome_filename: str, uniprot_id: str) -> list[float]:
    """
    Calculate the maxZ score introduced by [1].

    [1] Ahola, V., Aittokallio, T., Vihinen, M. et al. A statistical score for assessing the quality of multiple
    sequence alignments. BMC Bioinformatics 7, 484 (2006). https://doi.org/10.1186/1471-2105-7-484
    """
    # TODO: add citation to README.md?
    msa = MultipleSeqAlign.from_a3m("")

    # constants
    J = len(PROT_SEQ_AAS)
    N = len(msa.matches)

    C = np.eye(J)  # similarity matrix (TODO: does this need to come from a PSSM?)

    beta0 = np.zeros(
        J
    )  # TODO: find this, "degree of positional conservation w.r.t. a predefined background distribution"
    covariance0 = np.zeros(
        (J, J)
    )  # TODO: implement this: "equation (3) with beta_j replaced by beta_j0"

    Zs = np.zeros(len(msa.query_seq))
    for colidx in range(len(msa.query_seq)):
        # calculate observed symbol frequencies
        sym_freqs = {aa: 0 for aa in PROT_SEQ_AAS}  # symbol frequencies n_i
        for match in msa.matches:
            sym = match.aligned_seq[colidx]
            sym_freqs[sym] += 1

        n = sum(sym_freqs.values())  # actual number of symbols observed, N - gaps
        beta_mls = [
            sym_freq / n for sym_freq in sym_freqs
        ]  # maximum likelihood estimator for beta vector
        covariance = ()  # TODO: this matrix should be the same for each column

        # calculate actual Z scores
        Zis = np.zeros(J)
        for i in range(J):
            ci = C[i, :]  # c_i is "the ith row of C"
            Zi = (ci.T @ (beta_mls - beta0)) / np.sqrt(ci.T @ covariance0 @ ci)
        Z = np.max(Zis)
        Zs[colidx] = Z
    return Zs.tolist()
