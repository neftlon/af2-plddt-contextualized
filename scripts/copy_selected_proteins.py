#!/usr/bin/env python

"""
Copy selected protein MSAs from a source .tar to a folder. The MSAs to copy are specified by the UniProt ID of
the respective protein. These IDs need to be listed in a selection file.
"""
import os
import sys
from pathlib import Path
import tarfile

from tqdm import tqdm

from af22c.proteome import ProteomeMSAs

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: %s selectionfile sourcefile destdir" % sys.argv[0])
        sys.exit(2)
    selection_filename, source_filename, dest_dirname = sys.argv[1:]
    msas = ProteomeMSAs.from_archive(source_filename)
    with open(selection_filename) as selection:
        for protein_id in tqdm(selection.readlines()):
            protein_id = protein_id.strip()
            raw_msa = msas.get_raw_msa_by_id(protein_id)
            out_filename = os.path.join(dest_dirname, f"{protein_id}.a3m")
            Path(out_filename).write_text(raw_msa)
