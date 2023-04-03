#!/usr/bin/env python3

# pick K proteins randomly, run benchmarks for different methods, and store the results of each run in a CV file

import argparse
import os
import pandas as pd
import subprocess as sp
import random
import tempfile
from timeit import timeit
from tqdm import tqdm

# TODO: check whether docker containers are already available on _this_ system

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out", type=str, default="./data/bench.csv", help="CSV output file for benchmark results")
parser.add_argument("--mmseqs", type=str, default=os.environ.get("MMSEQS"), help="specify mmseqs executable location")
args = parser.parse_args()

PROTEOME_NAME = "UP000005640_9606"
PROTEOME_FILE = f"./data/{PROTEOME_NAME}.tar"
SCRIPTS = [
  #"./scripts/neff_gpu.py",
  "docker run -i --rm neff-mmseqs neff_mmseqs.py",
]
K = 50 # number of sampled proteins
random.seed(42) # reproducibility

script_args = "--batch-size 16384"
if args.mmseqs is not None:
  script_args += f" --mmseqs {args.mmseqs}"

# extract ids from tar file 
tarres = sp.run(
  # list tar file contents and remove those ending with a "/" (=directories)
  f"tar -tf {PROTEOME_FILE} | grep -e \"[^/]$\"",
  shell=True, capture_output=True, check=True
)
all_protein_names = [
  os.path.splitext(os.path.basename(n))[0]
  for n in tarres.stdout.decode().split()
  if n.startswith(PROTEOME_NAME) and "/msas/" in n # sanity check for file layout
]

# pick proteins
protein_names = random.choices(all_protein_names, k=K)

# run benchmarks
data = []
for protein_name in tqdm(protein_names, desc="timing proteins"):
  with tempfile.NamedTemporaryFile() as tf:
    # put .a3m file temp file
    sp.run(
      f"tar -xOf {PROTEOME_FILE} {PROTEOME_NAME}/msas/{protein_name}.a3m > {tf.name}",
      shell=True, check=True,
    )
    
    # run scripts
    for script in (pbar := tqdm(SCRIPTS, leave=False)):
      pbar.set_description("checking %s" % script)
      duration = -1
      try:
        duration = timeit(lambda: sp.run(
          # the pipe is required such that `cat` is executed on _this_ host machine,
          # whilst the script might be in a container.
          f"cat {tf.name} | {script} - - {script_args}",
          shell=True, capture_output=True, check=True,
        ), number=1)
      except sp.CalledProcessError as e:
        # use `repr` to escape string for CSV file
        status = f"returncode={e.returncode},output={repr(e.output)}"
      else:
        status = "PASS" # mark run as "no error"
      finally:
        data.append({
          "protein": protein_name,
          "script": script,
          "run_seconds": duration,
          "status": status,
        })
df = pd.DataFrame(data)
df.to_csv(args.out, index=False)

