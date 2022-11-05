#!/usr/bin/env python3

import io
import json

from Bio import SeqIO
from collections import namedtuple
from dataclasses import dataclass
import tarfile
import logging
import tqdm

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

@dataclass
class MsaMatch:
    attribs: MsaMatchAttribs
    gapped_seq: str

    def __getitem__(self, item: int):
        """Get a residue by index. (Including gaps from MSA.)"""
        return self.gapped_seq[item]


def extract_query_and_matches(a3m: io.TextIOBase) -> tuple[str, str, list[MsaMatch]]:
    # TODO(johannes): do we really want _this_ parameter? or should we accept filename strings as well?

    seqs = list(SeqIO.parse(a3m, "fasta"))
    query = seqs[0]  # first sequence is the query sequence
    matches = []
    for idx, seq in enumerate(seqs[1:]):
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
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


def normalized_hamming_distance(s1, s2) -> float:
    return hamming_distance(s1, s2) / len(s1)


def get_depth(query, matches: list[MsaMatch], theta_id=0.2) -> int:
    """
    Get the MSA for each column.

    Depth is calculated based on the N_eff score introduced in [1]. This score was also used in AlphaFold 2 for
    assessing MSA quality.

    [1] Tianqi Wu, Jie Hou, Badri Adhikari, Jianlin Cheng, Analysis of several key factors influencing deep
    learning-based inter-residue contact prediction, Bioinformatics, Volume 36, Issue 4, 15 February 2020,
    Pages 1091–1098, https://doi.org/10.1093/bioinformatics/btz679
    """
    num_matches = len(matches)
    n_eff = 0
    for s in range(num_matches):
        inv_pi_s = 0.0
        for t in range(num_matches):
            s_seq = matches[s].gapped_seq[:len(query)]
            t_seq = matches[t].gapped_seq[:len(query)]
            dist = normalized_hamming_distance(s_seq, t_seq)
            if dist < theta_id:
                inv_pi_s += 1.0
        pi_s = 1.0 / inv_pi_s
        n_eff += pi_s
    return n_eff


def proteome_wide_analysis():
    proteome_msas_filename = "data/UP000001816_190650.tar.gz"
    with tarfile.open(proteome_msas_filename) as tar:
        logging.debug(f"opened {proteome_msas_filename}")
        names = tar.getnames()
        logging.debug(f"found around {len(names)} .a3m files containing MSAs")
        depths = {}
        # TODO: this loop looks like a bottleneck -- make it parallel?
        for name in tqdm.tqdm(names):
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
