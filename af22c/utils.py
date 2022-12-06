"""
This module contains some functionality that is used in different scripts and can therefore be reused.
"""

import tarfile
import os
from functools import lru_cache
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import IO


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
    try:
        with open(filething, mode=mode, **kwargs) as fh:
            yield fh
    except TypeError:
        yield filething
