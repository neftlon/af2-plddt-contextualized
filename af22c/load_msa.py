#!/usr/bin/env python3

import io
from Bio import SeqIO
from typing import Union, Tuple, List
from collections import namedtuple
from dataclasses import dataclass
import tarfile
import logging
import itertools

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


def extract_query_and_matches(a3m: io.TextIOBase) -> Tuple[str, str, List[MsaMatch]]:
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


def proteome_wide_analysis():
    proteome_msas_filename = "data/UP000001816_190650.tar.gz"
    with tarfile.open(proteome_msas_filename) as tar:
        logging.debug(f"opened {proteome_msas_filename}")
        names = tar.getnames()
        logging.debug(f"found around {len(names)} .a3m files containing MSAs")
        for name in names:
            if name.endswith(".a3m"):
                # process .a3m member
                with tar.extractfile(name) as raw_a3m:
                    with io.TextIOWrapper(raw_a3m) as a3m:
                        query_id, query_seq, matches = extract_query_and_matches(a3m)
                        # TODO: do something with matches


# experiments with MSA files
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with open("data/Q9A7K5.a3m") as a3m:
        query_id, query_seq, matches = extract_query_and_matches(a3m)
