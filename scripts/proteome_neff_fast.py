#!/usr/bin/env python3

import os
import subprocess as sp
from tqdm import tqdm

NEFFFAST = "./scripts/neff_gpu.py"
PROTEOME_NAME="UP000005640_9606"
TARFILE_NAME=f"./data/{PROTEOME_NAME}.tar"
TARGET_DIR="./data/test"

# extract protein from tar file
tar_protein_filenames = sp.run(
  f"tar -tf {TARFILE_NAME}", 
  shell=True,
  capture_output=True,
)
assert tar_protein_filenames.returncode == 0, "failed to get protein names"
tar_protein_names = [
  os.path.splitext(os.path.basename(n))[0]
  for n in tar_protein_filenames.stdout.decode("ascii").splitlines()
  if "/msas/" in n and n.endswith(".a3m")
]

# see which proteins have already been processed
existing_protein_names = [
  os.path.splitext(os.path.basename(n))[0]
  for n in os.listdir(TARGET_DIR)
]
print(f"found {len(existing_protein_names)} already processed proteins")

to_process = list(set(tar_protein_names) - set(existing_protein_names))
print(f"{len(to_process)} proteins need to be processed")

# run Neff calculation for each file
num_success,num_failed = 0,0
for protein_name in (pbar := tqdm(to_process)):
  pbar.set_description("Neff'ing %s, %dP, %dF" % (protein_name, num_success,num_failed))
  outfilename = os.path.join(TARGET_DIR, f"{protein_name}.json")
  cmd = (
    f"tar -xOf {TARFILE_NAME} {PROTEOME_NAME}/msas/{protein_name}.a3m | "
    f"{NEFFFAST} -m - -o {outfilename}"
  )
  res = sp.run(cmd, shell=True, capture_output=True)
  if res.returncode != 0:
    num_failed += 1
    with open(os.path.join(TARGET_DIR, "failed.txt"), "a") as f:
      f.write(protein_name + "\n")
  else:
    num_success += 1

