#!/usr/bin/env python3

import os
import re
import tarfile
import gzip
import tempfile

import numpy as np
import tqdm
import json
from Bio.PDB.PDBParser import PDBParser
from concurrent.futures import ProcessPoolExecutor


UNIPROT_ID_EXTRACTOR = re.compile(
    ".*(af|AF)-(?P<uniprotid>([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-(f|F)[0-9]+)?)-model_v[0-9]+\.pdb\.gz"
)


def get_structure_from_lines(pdb_parser, id, pdb_lines):
    """Emulate the `get_structure` method of the `PDBParser` to process an in-memory PDB file since the function only
    allows for passing a filename as an argument"""
    pdb_parser.header = None
    pdb_parser.trailer = None
    pdb_parser.structure_builder.init_structure(id)
    pdb_parser._parse(pdb_lines)
    pdb_parser.structure_builder.set_header(pdb_parser.header)
    return pdb_parser.structure_builder.get_structure()


def get_plddts(struc):
    """Get pLDDT scores for each atom inside the structure of a PDB file's structure"""
    scores = []
    for model in struc:
        for chain in model:
            for res in chain:
                # NOTE: according to AlphaFold 2 database's FAQ [1], the pLDDT score is stored inside the
                # b-factor field of the PDB file.
                #
                # [1] https://alphafold.ebi.ac.uk/faq#faq-5
                plddts = [float(atom.get_bfactor()) for atom in res]
                mean_plddt = round(sum(plddts) / len(plddts), 2)
                scores.append(mean_plddt)
    return scores


def extract_pdb_gzs_from_tar_file(tar_filename, dst_dirname):
    """Extract all .pdb.gz files from a .tar file into an existing destination directory at `dst_dirname`"""
    with tarfile.open(tar_filename) as tar:
        members = tar.getmembers()
        # only take PDB files, ignore CIF
        members = filter(lambda member: member.name.endswith(".pdb.gz"), members)
        tar.extractall(dst_dirname, members)


def extract_plddts_from_pdb_gz(filename):
    """Extract the pLDDT scores from a compressed PDB (.pdb.gz) file."""
    with open(filename, "rb") as pdb_gz:
        # decompress PDB, obtain its structure information, and extract pLDDT scores
        pdb_gz = pdb_gz.read()
        pdb_bytes = gzip.decompress(pdb_gz)
        pdb = pdb_bytes.decode()
        pdb_lines = pdb.split("\n")
        pdb_parser = PDBParser()
        struc = get_structure_from_lines(pdb_parser, filename, pdb_lines)
        pdb_plddts = get_plddts(struc)

        matches = UNIPROT_ID_EXTRACTOR.match(filename)
        if not matches:
            raise Exception(
                f"was not able to extract UniProt ID from filename {filename}"
            )

        uniprot_id = matches.group("uniprotid")
        return uniprot_id, pdb_plddts


def extract_structure_from_pdb_gz(filename):
    """Extract the pLDDT scores from a compressed PDB (.pdb.gz) file."""
    with open(filename, "rb") as pdb_gz:
        # decompress PDB, obtain its structure information, and extract pLDDT scores
        pdb_gz = pdb_gz.read()
        pdb_bytes = gzip.decompress(pdb_gz)
        pdb = pdb_bytes.decode()
        pdb_lines = pdb.split("\n")
        pdb_parser = PDBParser()
        struc = get_structure_from_lines(pdb_parser, filename, pdb_lines)
        return struc


def extract_plddts_from_pdb_gzs(pdb_gzs_path):
    """Extract pLDDT scores from each compressed PDB file (.pdb.gz) inside a given directory `pdb_gzs_path` and
    return a `dict` mapping from UniProt identifier to a `list` of per-residue pLDDT scores (`float` from 0 to 100)."""
    filenames = os.listdir(pdb_gzs_path)
    filenames = map(
        lambda filename: os.path.join(pdb_gzs_path, filename), filenames
    )  # reconstruct full path
    filenames = list(filenames)

    # since the `extract_plddts_from_pdb_gz` function is CPU-bound, try to run it in parallel.
    with ProcessPoolExecutor() as ppe:
        return dict(
            tqdm.tqdm(
                ppe.map(extract_plddts_from_pdb_gz, filenames), total=len(filenames)
            )
        )


if __name__ == "__main__":
    # find available data files
    data_dir = "./data"
    names, paths = [], []
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and path.endswith(".tar"):
            name, _ = os.path.splitext(name)  # take name w/o file extension
            names.append(name)
            paths.append(path)

    print("found the following structure files:")
    for name, path in zip(names, paths):
        print(f" {name} at {path}")

    # collect pLDDT scores from PDB files
    for name, path in zip(names, paths):
        print(f"parsing {name}")

        # extract .pdb.gz files into its own folder
        with tempfile.TemporaryDirectory() as pdb_gzs_path:
            print(f" extracting TAR file items to {pdb_gzs_path}")
            extract_pdb_gzs_from_tar_file(path, pdb_gzs_path)

            # extract .pdb.gz files into .pdb files
            print(f" extracting pLDDT scores from files in {pdb_gzs_path}")
            proteome_plddts = extract_plddts_from_pdb_gzs(pdb_gzs_path)

            # write pLDDT scores to .json
            plddts_path = os.path.join(data_dir, name + "_plddts.json")
            print(
                f" writing output file to {plddts_path} containing pLDDT scores for {len(proteome_plddts)} proteins"
            )
            with open(plddts_path, "w") as outfile:
                json.dump(proteome_plddts, outfile)
            print(" done!")
