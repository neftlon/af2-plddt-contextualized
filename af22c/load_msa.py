#!/usr/bin/env python3
from Bio import SeqIO
from collections import namedtuple
from dataclasses import dataclass
import tarfile

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

# experiments with MSA files
if __name__ == "__main__":
    with tarfile.open("data/UP000001816_190650.tar.gz") as tar:
        names = tar.getnames()
        for name in names:
            print(name)

