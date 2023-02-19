#!/usr/bin/env python3

import sys
import os
import subprocess as sp
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
  description="generate per-residue Neff scores blazingly fast on the GPU",
)
parser.add_argument(
  "-f", "--source-file", type=str, metavar="SRC",
  default="./data/UP000005640_9606.tar",
  help="specify a source to load MSAs from. this can be\n(1) either a folder with .a3ms in it or\n"
       "(2) a .tar or .tar.gz archive (.tar.gz is slow, therefore not recommended). if an archive "
       "is specified, the .a3m files are expected to be in a subfolder and can be found using "
       "this path: \n f\"{PROTEOME_NAME}/msas/{protein_name}.a3m\" inside the archive. "
       "default filename is %(default)s",
)
parser.add_argument(
  "-o", "--out-dir", type=str, metavar="OUTDIR",
  default="./data/test",
  help="specify a directory that will contain output generated by this utility. this includes a "
       ".json file for each protein present in the SRC. also, a failed.txt file is written that "
       "contains (line-seperated) protein IDs that NEFFF failed to process. (potenially due to "
       "CUDA OOM or other errors.) default directory is %(default)s",
)
parser.add_argument(
  "-n", "--neff-fast", "--nefffast", type=str, metavar="NEFFF",
  default="./scripts/neff_gpu.py",
  help="specify the script that actually generates the Neff scores for each protein. (default "
       "script path %(default)s)",
)
parser.add_argument(
  "-d", "--device", type=str, metavar="DEVICE", default="cuda", 
  help="device passed to NEFFF, can be cuda, cpu, or any other valid pytorch "
       "device. (default is %(default)s)"
)
parser.add_argument(
  "-b", "--batch-size", "--batchsize", type=int, metavar="B", default=2**12,
  help="batch size passed to NEFFF, set to high number (>2**14) if much gpu memory "
       "is available and to lower value if not so much memory is available (<2**12)."
)
parser.add_argument(
  "-p", "--proteome-name", "--proteome", type=str, metavar="P", default="infer",
  help="optionally specify the proteome name. set to \"infer\" to try to infer the PROTEOME_NAME "
       "from the archive name. this is used to locate the folder containing the \"/msas/\" "
       " subfolder containing the .a3ms inside the archive. see the --source-file option. (for "
       "folder-based SRC, this option is not relevant.)"
)
args = parser.parse_args()

# extract protein from archive or folder 
tar_protein_filenames = sp.run(
  f"tar -tf {args.source_file}", 
  shell=True,
  capture_output=True,
)
assert tar_protein_filenames.returncode == 0, "failed to get protein names"
tar_stdout_lines = tar_protein_filenames.stdout.decode("ascii").splitlines()
tar_protein_names = [
  os.path.splitext(os.path.basename(n))[0]
  for n in tar_stdout_lines if "/msas/" in n and n.endswith(".a3m")
]

# if proteome name is infer, try to to infer the proteome name from the archive
if args.proteome_name == "infer":
  # NB: this is a very hacking way. tar -t probably works the same on a lot of linux machines, 
  # but does it guarantee that the order of outputs? in the following line, it is required that
  # the first output line of tar -t is the base directory containing all the MSAs. right of the 
  # bat, I can imagine 1000&1 ways how this can go wrong.
  args.proteome_name = tar_stdout_lines[0][:-1]
  print("tried to infer proteome name:", args.proteome_name)

# see which proteins have already been processed
existing_protein_names = [
  os.path.splitext(os.path.basename(n))[0]
  for n in os.listdir(args.out_dir)
]
print(f"found {len(existing_protein_names)} already processed proteins")

to_process = list(set(tar_protein_names) - set(existing_protein_names))
print(f"{len(to_process)} proteins need to be processed")

# run Neff calculation for each file
num_success,num_failed = 0,0
for protein_name in (pbar := tqdm(to_process)):
  pbar.set_description("Neff'ing %s, %dP, %dF" % (protein_name, num_success,num_failed))
  outfilename = os.path.join(args.out_dir, f"{protein_name}.json")
  cmd = (
    f"tar -xOf {args.source_file} {args.proteome_name}/msas/{protein_name}.a3m | "
    f"{args.neff_fast} -m - -o {outfilename} --batch-size {args.batch_size} --device {args.device}"
  )
  res = sp.run(cmd, shell=True, capture_output=True)
  if res.returncode != 0:
    num_failed += 1
    with open(os.path.join(args.out_dir, "failed.txt"), "a") as f:
      f.write(protein_name + "\n")
    tqdm.write("protein %s failed due to \n%s" % (protein_name,res.stderr))
  else:
    num_success += 1

