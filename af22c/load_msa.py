#!/usr/bin/env python3

import io
import json
import os.path

from Bio import SeqIO
from Bio.Seq import Seq
from collections import namedtuple
from dataclasses import dataclass, field
from tqdm import tqdm
import tarfile
import logging
import string
import numpy as np
from functools import lru_cache
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

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


@lru_cache(None)
def warn_once(msg: str):
    """
    Print only log message only once.

    Code taken from: https://stackoverflow.com/a/66062313
    """
    logging.warning(msg)

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


def extract_query_and_matches(a3m_handle) -> tuple[str, str, list[MsaMatch]]:
    # TODO(johannes): do we really want _this_ parameter? or should we accept filename strings as well?
    seqs = list(SeqIO.parse(a3m_handle, "fasta"))
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
    """Per protein N_eff score."""
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


def compute_pairwise_seq_ident(combined_args):
    i, msa_vec = combined_args
    s1 = msa_vec[i]
    result = [np.sum(s1 == s2) for s2 in msa_vec]
    return result
    #for j, s2 in enumerate(msa_vec):
        # TODO (@Simon) only run j till j==i and set
        # upper triangle matrix to same value
        # TODO Try to sum after loop, i.e. trade memory for speed
    #    row[j] = np.sum(s1 == s2)


def seq_identity_parallel(msa):
    msa_vec = np.array([list(seq) for seq in msa])
    with ProcessPoolExecutor() as ppe:
        pairwise_input = zip(range(len(msa_vec)), repeat(msa_vec))
        pair_seq_ident = list(tqdm(
            ppe.map(compute_pairwise_seq_ident, pairwise_input),
            desc='Compute seq_ident',
            total=len(msa_vec),
        ))
        return np.array(pair_seq_ident) / len(msa[0])

def get_depth(query, matches: list[MsaMatch], seq_id=0.8):
    msa = [query] + matches
    pair_seq_id = seq_identity_parallel(msa)

    n_eff_weights = np.zeros(len(msa))
    for i in tqdm(range(len(msa)), desc="calculating Neff weights", total=len(msa)):
        n_eff_weights[i] = sum(map(int, pair_seq_id[i] >= seq_id))
    inv_n_eff_weights = 1 / n_eff_weights

    n_non_gaps = np.zeros(len(query))
    for c in tqdm(range(len(query)), desc="counting gaps", total=len(query)):
        for i, m in enumerate(msa):
            n_non_gaps[c] += int(m[c] != '-') * inv_n_eff_weights[i]
    return n_non_gaps


def get_names_from_tarfile(tar_filename: str) -> list[str]:
    with tarfile.open(tar_filename) as tar:
        logging.debug(f"opened {tar_filename}")
        names = tar.getnames()
        logging.debug(f"found around {len(names)} .a3m files containing MSAs")
        return names


def get_neff_by_id(tar_filename, uniprot_id) -> np.ndarray:
    if tar_filename.endswith(".tar.gz"):
        warn_once(f"iterating a .tar.gz file is much slower than just using the already-deflated .tar file")

    # NOTE: This function expects the `tar_filename` to be in a special format to work with a generated .tar file. The
    # MSAs are expected to be in a subdirectory called `f"{proteome_name}/msas/"`.

    proteome_names, _ = os.path.splitext(os.path.basename(tar_filename))
    a3m_subdir = os.path.join(proteome_names, "msas")

    with tarfile.open(tar_filename) as tar:
        a3m_filename = os.path.join(a3m_subdir, f"{uniprot_id}.a3m")
        a3m = tar.extractfile(a3m_filename)
        with io.TextIOWrapper(a3m, encoding="utf-8") as a3m:
            query_id, query_seq, matches = extract_query_and_matches(a3m)
            depths = get_depth(query_seq, matches)
            return depths
    

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
    prot_id = "A0A0A0MRZ7"
    depths = get_neff_by_id("data/UP000005640_9606.tar", prot_id)
    print(f"{prot_id}: {depths}")
    """
    with open("data/Q9A7K5.a3m") as a3m:
        query_id, query_seq, matches = extract_query_and_matches(a3m)
        depths = get_depth(query_seq, matches)
        print(depths)
    """
