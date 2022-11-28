"""
This module contains some functionality that is used in different scripts and can therefore be reused.
"""

import tarfile
import os


def get_raw_proteome_name(proteome_filename: str) -> str:
    """Take a full proteome filename and only select the proteome name"""
    return os.path.splitext(os.path.basename(proteome_filename))[0]


def get_protein_ids(proteome_filename: str) -> list[str]:
    """Get a list of UniProt IDs that the corresponding proteome file contains"""
    with tarfile.open(proteome_filename) as tar:
        raise NotImplemented("`get_protein_ids` is not implemented yet")
