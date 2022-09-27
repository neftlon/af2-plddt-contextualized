#!/usr/bin/env python3

import os
import sys
import tarfile
import gzip
import tqdm
import tempfile
import json
from Bio.PDB.PDBParser import PDBParser


def get_structure_from_lines(pdb_parser, id, pdb_lines):
    """emulate the `get_structure` method of the `PDBParser` to process an in-memory PDB file since the function only allows for passing a filename as an argument"""
    pdb_parser.header = None
    pdb_parser.trailer = None
    pdb_parser.structure_builder.init_structure(id)
    pdb_parser._parse(pdb_lines)
    pdb_parser.structure_builder.set_header(pdb_parser.header)
    return pdb_parser.structure_builder.get_structure()


def get_plddts(struc):
    """get pLDDT scores for each atom inside the structure of a PDB file's structure"""
    scores = []
    for model in struc:
        for chain in model:
            for res in chain:
                # TODO(johannes): In [1], always the first atom in `res` is picked. However, shouldn't we look 
                # for the C-alpha atom instead of blindly picking the first one? According to [2], pLDDT
                # "estimates whether the predicted residue has similar distances to neighboring C-alpha atoms".
                #
                # [1] https://github.com/Rostlab/TMvis/blob/afc65d099012ba6e1aed928f76e8ea033210f8e3/TMvis/main.py#L156-L159
                # [2] https://www.rbvi.ucsf.edu/chimerax/data/pae-apr2022/pae.html#:~:text=Per%2Dresidue%20confidence%20scores%20(pLDDT,below%2050%20indicating%20low%20confidence.
                atom = res["CA"]

                # NOTE: according to AlphaFold 2 database's FAQ [1], the pLDDT score is stored inside the
                # b-factor field of the PDB file. 
                # 
                # [1] https://alphafold.ebi.ac.uk/faq#faq-5 
                plddt = float(atom.get_bfactor())
                scores.append(plddt)
    return scores


def extract_plddts_from_tarfile_item(params):
    tar, filename = params
    with tar.extractfile(filename) as pdb_gz:
        # decompress PDB, obtain its structure information, and extract pLDDT scores
        pdb_gz = pdb_gz.read()
        pdb_bytes = gzip.decompress(pdb_gz)
        pdb = pdb_bytes.decode()
        pdb_lines = pdb.split("\n")
        pdb_parser = PDBParser()
        struc = get_structure_from_lines(pdb_parser, filename, pdb_lines)
        pdb_plddts = get_plddts(struc)

        # TODO(johannes): is this a correct way to extract the UniProt identifier?
        uniprot_id = filename.split("-")[1]
        return uniprot_id, pdb_plddts
    return None

if __name__ == "__main__":
    # find available data files
    data_dir = "./data"
    names, paths = [], []
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and path.endswith(".tar"):
            names.append(name)
            paths.append(path)

    print("found the following structure files:")
    for name, path in zip(names, paths):
        print(f" {name} at {path}")

    # collect pLDDT scores from PDB files
    for name, path in zip(names, paths):
        print(f"parsing {name}")
        num_pdb_files = 0
        proteome_plddts = {}
        with tarfile.open(path) as tar:
            print(" opened TAR file")
            filenames = tar.getnames()
            filenames = list(filter(lambda filename: filename.endswith(".pdb.gz"), filenames))  # only take PDB files, ignore CIF
            print(f" found {len(filenames)} files in TAR file")
            for filename in tqdm.tqdm(filenames):
                if pair := extract_plddts_from_tarfile_item((tar, filename)):
                    uniprot_id, pdb_plddts = pair
                    proteome_plddts[uniprot_id] = pdb_plddts
                    num_pdb_files += 1
                else:
                    print(f" ERROR: failed to process {filename}", file=sys.stderr)
        print(f" found and processed {num_pdb_files} PDB files from TAR file")

        # store cached pLDDT scores as a file
        outfilename = os.path.join(data_dir, "plddts.json")
        with open(outfilename, "w") as outfile:
            json.dump(proteome_plddts, outfile)
        print(f" wrote {outfilename} containing extracted pLDDT scores")

