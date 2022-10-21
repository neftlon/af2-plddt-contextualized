#!/usr/bin/env python3 

import os
import tarfile
import tempfile
from af22c.extract_plddts import extract_structure_from_pdb_gz 

def get_titin_structures():
    with tarfile.open("data/UP000005640_9606_HUMAN_v3.tar") as tar:
        print("finding members")
        members = tar.getmembers()
        titin_members = list(filter(lambda member: ("Q8WZ42" in member.name) and member.name.endswith(".pdb.gz"), members))
        with tempfile.TemporaryDirectory() as temp_path:
            print(f"extracting all titin members to {temp_path}")
            tar.extractall(path=temp_path, members=titin_members)
            print(f"extracted, now taking a look at {len(titin_members)} PDB files")
            for member in titin_members:
                member_path = os.path.join(temp_path, member.name)
                #uniprot_id, pdb_plddts = extract_plddts_from_pdb_gz()
                #print(uniprot_id)
                print(f" extracting member {member_path}")
                yield extract_structure_from_pdb_gz(member_path)


if __name__ == "__main__":
    pass

