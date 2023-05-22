import io
import json
import os.path
import sys

import tarfile

def calc_neff_by_id(tar_filename: str, uniprot_id: str) -> list[float]:
    """Calculate Neff scores for a single protein (by its UniProt identifier)."""

    if tar_filename.endswith(".tar.gz"):
        warn_once(
            f"iterating a .tar.gz file is much slower than just using the already-deflated .tar file"
        )

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
        warn_once(
            f"iterating a .tar.gz file is much slower than just using the already-deflated .tar file"
        )

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
