#!/usr/bin/env python3

import io
import json
import os.path
import sys

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
from itertools import repeat, chain
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from typing import Callable


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
    logging.info(" loading MSAs")
    for idx, seq in tqdm(enumerate(seqs[1:]), total=len(seqs)-1):
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
    logging.info(" computing Neffs")
    for s in tqdm(range(num_matches)):
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
    # Prepare msa for vector operations.
    msa_vec = np.array([list(seq) for seq in msa])

    # Compute the number of identical residues for all pairs once
    # and build an upper right triangle matrix from it.
    n_ident_res = np.zeros((len(msa), len(msa)))
    for i, s in tqdm(enumerate(msa_vec), desc='Compute n_ident_res (s/it decreasing)', total=len(msa)):
        # For each sequence, count the number of identical residues in all following sequences.
        n_ident_res[i, i:] = np.sum(msa_vec[i:] == s, axis=1)

    # Fill the lower left triangle matrix with the values from the upper right triangle.
    n_ident_res += n_ident_res.T - np.diag(n_ident_res.diagonal())

    return n_ident_res / len(msa[0])


def compute_pairwise_seq_ident(combined_args):
    i, msa_vec = combined_args
    s1 = msa_vec[i]
    result = [np.sum(s1 == s2) for s2 in msa_vec]
    return result


def one_against_many_res_id(combined_args):
    i, msa_vec = combined_args
    return np.sum(msa_vec[i:] == msa_vec[i], axis=1)


def batched_one_against_many_res_id(combined_args):
    """
    Compute the number of identical residues between a batch of input sequences and a complete MSA.
    """
    idx, msa_vec = combined_args
    return [np.sum(msa_vec[i:] == msa_vec[i], axis=1) for i in idx]


def flatten_list_of_lists(l):
    """
    Faster than list comprehension solutions.
    """
    return list(chain.from_iterable(l))


def sort_by_idx(idx, l):
    """
    Sort a list by an index given as list of indices.
    """
    return [x for _, x in sorted(zip(idx, l))]


def seq_identity_parallel(msa):
    # Prepare msa for vector operations.
    msa_vec = np.array([list(seq) for seq in msa])

    # Distribute the sequence indices over multiple batches.
    # TODO use default number of workers (i.e. number of cpus) and read it out
    #  for splitting the MSA in batches.
    n_batches = mp.cpu_count()
    batch_idxs = [range(i, len(msa_vec), n_batches) for i in range(n_batches)]

    with ProcessPoolExecutor(max_workers=n_batches) as ppe:
        pairwise_input = zip(batch_idxs, repeat(msa_vec))
        # TODO add some kind of progress indicator for the batched
        #  multiprocessing, maybe report from called function at specific
        #  intervals and accumulate progress here?
        #  Beware, with the current batching early iterations are slower than later ones.
        #  This could be changed by shuffling and then reordering.
        logging.info(f" mapping one-against-many to {n_batches} jobs")
        batched_n_ident_res_list = list(tqdm(
            ppe.map(batched_one_against_many_res_id, pairwise_input)
        ))

        n_ident_res_list_ur = flatten_list_of_lists(batched_n_ident_res_list)
        idx = flatten_list_of_lists(batch_idxs)

        n_ident_res_list_ur = sort_by_idx(idx, n_ident_res_list_ur)

        # Build an upper triangle matrix from the computed list of lists, where each list
        # represents one row of the upper right triangle matrix.
        n_ident_res_vec = np.zeros((len(msa), len(msa)))
        n_ident_res_vec[np.triu_indices_from(n_ident_res_vec)] = flatten_list_of_lists(
            n_ident_res_list_ur
        )

        # Fill the lower left triangle matrix with the values from the upper right triangle
        n_ident_res_vec += n_ident_res_vec.T - np.diag(n_ident_res_vec.diagonal())

        return np.array(n_ident_res_vec) / len(msa[0])


def get_depth(query, matches: list[MsaMatch], seq_id=0.8):
    msa = [query] + matches
    pair_seq_id = seq_identity_parallel(msa)

    n_eff_weights = np.zeros(len(msa))
    logging.info(" calulating Neff weights...")
    for i in tqdm(range(len(msa)), total=len(msa)):
        n_eff_weights[i] = sum(map(int, pair_seq_id[i] >= seq_id))
    inv_n_eff_weights = 1 / n_eff_weights

    n_non_gaps = np.zeros(len(query))
    logging.info(" counting gaps...")
    for c in tqdm(range(len(query)), total=len(query)):
        for i, m in enumerate(msa):
            n_non_gaps[c] += int(m[c] != '-') * inv_n_eff_weights[i]
    return n_non_gaps


def get_depth_naive(query, matches: list[MsaMatch]):
    msa = [query] + matches

    n_non_gaps = np.zeros(len(query))
    logging.info(" counting gaps...")
    for c in tqdm(range(len(query)), total=len(query)):
        for i, m in enumerate(msa):
            n_non_gaps[c] += int(m[c] != '-')
    return n_non_gaps


def get_names_from_tarfile(tar_filename: str) -> list[str]:
    with tarfile.open(tar_filename) as tar:
        logging.debug(f"opened {tar_filename}")
        names = tar.getnames()
        logging.debug(f"found around {len(names)} .a3m files containing MSAs")
        return names


def calc_neff_by_id(tar_filename: str, uniprot_id: str) -> list[float]:
    """Calculate Neff scores for a single protein (by its UniProt identifier)."""

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
            # NOTE: since Neff scores usually are in the area of 1k to 10k, rounding to `int` here should be sufficient
            return list(map(lambda f: round(f), depths.tolist()))


def calc_naive_neff_by_id(tar_filename: str, uniprot_id: str) -> list[float]:
    """Calculate naive Neff scores (=count gaps in MSA) for a single protein (found by its UniProt identifier)."""

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
            naive_depths = get_depth_naive(query_seq, matches)
            # NOTE: since Neff scores usually are in the area of 1k to 10k, rounding to `int` here should be sufficient
            return list(map(lambda f: round(f), naive_depths.tolist()))


def get_depths_naive_by_a3m(a3m_handle) -> list[int]:
    """Calculate MSA size for a single protein given as a3m file handle."""
    _, query_seq, matches = extract_query_and_matches(a3m_handle)
    depths = get_depth_naive(query_seq, matches)
    return depths.tolist()


def get_a3m_size(a3m_handle) -> tuple[int, int]:
    """Calculate MSA size for a single protein given as a3m file handle."""
    _, query_seq, matches = extract_query_and_matches(a3m_handle)
    return len(query_seq), len(matches) + 1  # +1 for query sequence


def apply_by_id(func: Callable, tar_filename: str, uniprot_id: str):
    """
    Apply func to an .a3m-file stored in the given tar. The .a3m-file is identified by uniprot_id.
    """
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
            return func(a3m)


def proteome_wide_analysis():
    proteome_msas_filename = "data/UP000001816_190650.tar.gz"
    with tarfile.open(proteome_msas_filename) as tar:
        logging.debug(f"opened {proteome_msas_filename}")
        names = tar.getnames()
        logging.debug(f"found around {len(names)} .a3m files containing MSAs")
        depths = {}
        # TODO: this loop looks like a bottleneck -- make it parallel?
        for name in tqdm(names, desc="iterating file in .tar"):
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

    # if we have the HUMAN proteome tar file available, try to load a protein from there
    human_path = "data/UP000005640_9606.tar"
    if os.path.exists(human_path):
        logging.info("HUMAN proteome available, loading example Neff scores")
        depths = calc_neff_by_id(human_path, "A0A0A0MRZ9")
        print(depths)
        # Code example: How to compute msa size
        # seq_len, num_seq = apply_by_id(get_a3m_size, human_path, "A0A0A0MRZ9")

        # Code example: How to compute naive depths
        # depths_naive = apply_by_id(get_depths_naive_by_a3m, human_path, "A0A0A0MRZ9")
        # print(depths_naive)
        sys.exit(0)  # don't run anything afterwards

    # prot_id = "A0A0A0MRZ7"
    # depths = get_neff_by_id("data/UP000005640_9606.tar", prot_id)
    # print(f"{prot_id}: {depths}")
    with open("data/Q9A7K5.a3m") as a3m:
        query_id, query_seq, matches = extract_query_and_matches(a3m)
        depths = get_depth(query_seq, matches)
        print(depths)
