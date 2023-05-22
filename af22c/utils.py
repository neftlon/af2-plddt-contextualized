"""
This module contains some functionality that is used in different scripts and can therefore be reused.
"""

import tarfile
import os
import io
from functools import lru_cache
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import IO, NamedTuple
from argparse import ArgumentParser
from dataclasses import dataclass


LimitsType = tuple[float | int | None, float | int | None] | None

# TODO: something like this probably exists, find out where something like this is.
class TarLoc(NamedTuple):
    """Specify the location of a file that is located in a (potentially compressed) .tar archive."""
    tar_filename: str # location of the .tar file itself
    filename: str # location of the file inside the .tar file

@dataclass
class ProteomeTar:
    """
    Describe a .tar file containing multiple MSAs.
    
    The .tar file is expected to contain multiple MSAs files, each of which containing a UniProt identifier in their name.
    """
    tar_filename: str
    proteome_name: str

    def __init__(self, tar_filename: str, proteome_name: str = None):
        self.tar_filename = tar_filename
        if proteome_name is None: # try to extract proteome name from tar filename if not given
            proteome_name, _ = os.path.splitext(os.path.basename(self.tar_filename))
        self.proteome_name = proteome_name

        if tar_filename.endswith(".tar.gz"):
            warn_once(
                f"iterating a .tar.gz file is much slower than just using the already-deflated .tar file"
            )
    
    def get_protein_location(self, uniprot_id: str) -> TarLoc:
        """Get the location of the MSA file for a specific protein."""
        filename = os.path.join(self.proteome_name, "msas", f"{uniprot_id}.a3m")
        return TarLoc(self.tar_filename, filename)

@lru_cache(None)
def warn_once(msg: str):
    """
    Print only log message only once.

    Code taken from: https://stackoverflow.com/a/66062313
    """
    logging.warning(msg)


def get_raw_proteome_name(proteome_filename: str) -> str:
    """Take a full proteome filename and only select the proteome name"""
    return os.path.basename(proteome_filename).split(".")[
        0
    ]  # the proteome name comes before the .tar.gz extension.


def get_protein_ids(proteome_filename: str) -> list[str]:
    """Get a list of UniProt IDs that the corresponding proteome file contains"""
    with tarfile.open(proteome_filename) as tar:
        raise NotImplementedError("`get_protein_ids` is not implemented yet")


@contextmanager
def as_handle(filething, mode="r", **kwargs) -> IO:
    """
    Transparent context manager for working with files. You can pass a filename as a `str`, `Path` or an actual file
    object to this function. The function will try to create a handle if necessary.
    """
    if isinstance(filething, TarLoc):
        with tarfile.open(filething.tar_filename) as tar:
            with tar.extractfile(filething.filename) as fh:
                with io.TextIOWrapper(fh, encoding="utf-8") as wrapper:
                    yield wrapper
    elif isinstance(filething, io.TextIOWrapper):
        yield filething
    else:
        try:
            with open(filething, mode=mode, **kwargs) as fh:
                yield fh
        except TypeError:
            yield filething


def add_msa_size_limit_options(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("-n", "--max_n_sequences", default=None, type=int)
    parser.add_argument("-l", "--max_query_length", default=None, type=int)
    parser.add_argument("-m", "--min_n_sequences", default=None, type=int)
    parser.add_argument("-k", "--min_query_length", default=None, type=int)
    return parser


def size_limits_to_dict(args) -> dict:
    return {
        "min_q_len": args.min_query_length,
        "max_q_len": args.max_query_length,
        "min_n_seq": args.min_n_sequences,
        "max_n_seq": args.max_n_sequences
    }


def readable_enumeration(items: list[str], coord_conj: str = "and", oxford_comma: bool = True) -> str:
    if len(items) < 3:
        return f" {coord_conj} ".join(items)
    *a, b = items
    return f"{', '.join(a)}{',' if oxford_comma else ''} {coord_conj} {b}"
