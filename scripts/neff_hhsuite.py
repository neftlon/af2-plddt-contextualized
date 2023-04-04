#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
import subprocess as sp
import struct
import sys

# parse arguments
parser = argparse.ArgumentParser(description="generate per-residue Neff scores with hhmake")
parser.add_argument("INFILE",type=str,help="input .a3m MSA file")
parser.add_argument("OUTFILE",type=argparse.FileType("w"),help="output .json file containing per-residue scores")
parser.add_argument(
  "--hhmake",type=str,default=dict(os.environ).get("HHMAKE","hhmake"),
  help="specify hhmake executable.",
)
compli = "exists for compliance, is ignored."
parser.add_argument("-d", "--device", type=str, default=None, metavar="?", required=False, help=compli)
parser.add_argument("-b", "--batch-size", type=int, default=None, metavar="?", required=False, help=compli)
parser.add_argument("-l", "--gpu-mem-limit", type=str, required=False, metavar="?", default=None, help=compli)
args = parser.parse_args()

# compute Neff scores with hhmake
inarg = args.INFILE
if inarg == "-":
  inarg = "stdin" # make compliant with hhsuite
hhmakeres = sp.run(
  f"{args.hhmake} -id 100 -diff inf -seq 2 -i {inarg} -o stdout",
  shell=True, stdout=sp.PIPE, stderr=sp.PIPE,
)
if hhmakeres.returncode != 0:
  print("error: to run hhmake.",file=sys.stderr)
  print("!!! cmd stdout !!!",hhmakeres.stdout.decode(),sep="\n",file=sys.stderr)
  print("!!! cmd stderr !!!",hhmakeres.stderr.decode(),sep="\n",file=sys.stderr)
  sys.exit(-1)

# extract Neff scores from hhmake's output
reader = csv.reader(hhmakeres.stdout.decode().split("\n"), delimiter="\t")
neff_list = []
# Skip all entries until 'HMM' field is found and then three more
for l in reader:
    if l and l[0].startswith('HMM'):
        next(reader)
        next(reader)
        next(reader)
        break
# Read entry from 8th column of every third line
for l in reader:
    if len(l) >= 8: # ensure index 7 is available
        neff_list.append(int(l[7]))
        next(reader)
        next(reader)

print(json.dumps(neff_list),file=args.OUTFILE) # use print instead of json.dump to not have null-byte at the end

