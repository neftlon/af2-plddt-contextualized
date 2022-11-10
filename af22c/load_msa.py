#!/usr/bin/env python3

import io
import json

from Bio import SeqIO
from Bio.Seq import Seq
from collections import namedtuple
from dataclasses import dataclass, field
from tqdm import tqdm
import tarfile
import logging
import string
import numpy as np

MsaMatchAttribs = namedtuple("MsaMatchAttribs", [
    "target_id",
    "aln_score",
    "seq_identity",
    "eval",
    "qstart",
    "qend",
    "qlen",
    "tstart",
    "tend",
    "tlen",
])


# Generate translation table for lowercase removal
LOWERCASE_DEL_TABLE = str.maketrans('', '', string.ascii_lowercase)


@dataclass
class MsaMatch:
    """
    MsaMatch object containing the parsed header field in attribs, the original sequence in orig_seq    and the sequence without insertions.
    """
    attribs: MsaMatchAttribs
    orig_seq: str
    aligned_seq: str = field(init=False)

    def __post_init__(self):
        # Convert to String and back in order to use string translation method instead of
        # Biopython translation. This is still way faster than filtering and joining.
        self.aligned_seq = Seq(str(self.orig_seq).translate(LOWERCASE_DEL_TABLE))

    def __getitem__(self, item: int):
        """Get a residue by index. (Including gaps from MSA.)"""
        return self.aligned_seq[item]

    def __str__(self):
        return self.aligned_seq

    def __len__(self):
        return len(self.aligned_seq)


def extract_query_and_matches(a3m: io.TextIOBase) -> tuple[str, str, list[MsaMatch]]:
    # TODO(johannes): do we really want _this_ parameter? or should we accept filename strings as well?
    seqs = list(SeqIO.parse(a3m, "fasta"))
    query = seqs[0]  # first sequence is the query sequence
    matches = []
    for idx, seq in tqdm(enumerate(seqs[1:]), desc='Loading MSAs', total=len(seqs)-1):
        raw_attribs = seq.description.split("\t")
        # TODO(johannes): Sometimes (for instance in Q9A7K5.a3m) the MSA file contains the same (presumable) query
        # sequence at least twice. What purpose does this serve? The code below currently skips these duplications, but
        # this is probably just wrong behavior.
        if len(raw_attribs) != len(MsaMatchAttribs._fields):
            logging.warning(f"a3m file contains a match at index {idx} (of {len(seqs)} matches) that contains not the "
                            f"required (={len(MsaMatchAttribs._fields)}) number of fields: {len(raw_attribs)}. "
                            f"match description: \"{seq.description}\"")
            continue

        attribs = MsaMatchAttribs(*raw_attribs)
        match = MsaMatch(attribs, seq.seq)
        matches.append(match)
    return query.id, query.seq, matches


def hamming_distance(s1, s2) -> int:
    """Return the Hamming distance between equal-length sequences.

    Taken from: https://en.wikipedia.org/wiki/Hamming_distance"""
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length.")
    return sum(np.array(list(s1)) != np.array(list(s2)))


def normalized_hamming_distance(s1, s2) -> float:
    return hamming_distance(s1, s2) / len(s1)


def get_n_eff(query, matches: list[MsaMatch], theta_id=0.2) -> int:
    num_matches = len(matches)
    n_eff = 0
    for s in tqdm(range(num_matches), desc='Compute Neffs'):
        inv_pi_s = 0.0
        for t in range(num_matches):
            s_seq = matches[s].aligned_seq
            t_seq = matches[t].aligned_seq
            dist = normalized_hamming_distance(s_seq, t_seq)
            if dist < theta_id:
                inv_pi_s += 1.0
        pi_s = 1.0 / inv_pi_s
        n_eff += pi_s
    return n_eff

def seq_identity_vectorized(msa):
    msa_vec = np.array([list(seq) for seq in msa])

    # Initialize with NaN in order to avoid silent errors
    pair_seq_ident = np.empty((len(msa), len(msa)))
    pair_seq_ident[:] = np.nan

    for i, s1 in tqdm(enumerate(msa_vec), desc='Compute seq_ident', total=len(msa)):
        for j, s2 in enumerate(msa_vec):
            # TODO (@Simon) only run j till j==i and set
            # upper triangle matrix to same value
            # TODO Try to sum after loop, i.e. trade memory for speed
            pair_seq_ident[i, j] = np.sum(s1 == s2)

    return pair_seq_ident / len(msa[0])

def get_depth(query, matches: list[MsaMatch], seq_id=0.8):
    msa = [query] + matches
    pair_seq_id = seq_identity_vectorized(msa)
    n_eff_weights = np.zeros(len(msa))
    for i in range(len(msa)):
        n_eff_weights[i] = sum(map(int, pair_seq_id[i] >= 0.8))
    inv_n_eff_weights = 1 / n_eff_weights


    n_non_gaps = np.zeros(len(query)) 
    for i, m in enumerate(msa):
        for c in range(len(query)):
            n_non_gaps[c] += int(m[c] != '-') * inv_n_eff_weights[i]
    return n_non_gaps
    

def proteome_wide_analysis():
    proteome_msas_filename = "data/UP000001816_190650.tar.gz"
    with tarfile.open(proteome_msas_filename) as tar:
        logging.debug(f"opened {proteome_msas_filename}")
        names = tar.getnames()
        logging.debug(f"found around {len(names)} .a3m files containing MSAs")
        depths = {}
        # TODO: this loop looks like a bottleneck -- make it parallel?
        for name in tqdm(names):
            if name.endswith(".a3m"):
                # process .a3m member
                with tar.extractfile(name) as raw_a3m:
                    with io.TextIOWrapper(raw_a3m) as a3m:
                        query_id, query_seq, matches = extract_query_and_matches(a3m)
                        prot_depths = get_depth(query_seq, matches)
                        depths[query_id] = prot_depths

        with open("data/UP000001816_190650_depths.json", "w") as f:
            json.dump(depths, f)


# experiments with MSA files
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    #proteome_wide_analysis()
    with open("data/Q9A7K5.a3m") as a3m:
        query_id, query_seq, matches = extract_query_and_matches(a3m)
        depths = get_depth(query_seq, matches)
        print(depths)
