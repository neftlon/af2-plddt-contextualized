#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess as sp
import struct
import sys
from tempfile import TemporaryDirectory

# parse arguments
parser = argparse.ArgumentParser(description="generate Neff scores from an MSA using mmseqs")
parser.add_argument("INFILE",type=argparse.FileType(),help="input .a3m MSA file")
parser.add_argument("OUTFILE",type=argparse.FileType("w"),help="output .json file containing per-residue scores")
parser.add_argument(
  "--mmseqs",type=str,default=dict(os.environ).get("MMSEQS","mmseqs"),
  help="specify mmseqs executable. note that the version of mmseqs must be "
       "such that the profile2neff command is available!",
)
args = parser.parse_args()

with TemporaryDirectory() as tempdir:
  infile = args.INFILE.read()
  
  # this hack creates a valid mmseqs db from a single .a3m file in 3 steps
  # (by default mmseqs cannot use .a3m files as db input)
  # 1) copy original a3m file and append a null byte
  dbfilename = os.path.join(tempdir,"msaDb")
  with open(dbfilename, "wb") as dbfile:
    dbfile.write(infile.encode())
    dbfile.write(struct.pack("B", 0))
  # 2) write a .dbtype file specifying 0xb = 0d11 as mode, which indicates a3m
  with open(f"{dbfilename}.dbtype", "wb") as dbtype:
    dbtype.write(struct.pack("<Bxxx", 0xb))
  # 3) write a .index file specifying a single entry
  with open(f"{dbfilename}.index", "w") as dbindex:
    size = len(infile)  # write size of ORIGINAL file
    dbindex.write("\t".join(["0", "0", str(size)]) + "\n")
  
  # create an mmseqs profile
  profile_filename = os.path.join(tempdir,"profileDb")
  profileret = sp.run(
    "%s msa2profile %s %s" % (args.mmseqs,dbfilename,profile_filename),
    shell=True,capture_output=True,
  )
  if profileret.returncode != 0:
    print("error: failed to convert MSA to profile.",file=sys.stderr)
    print("!!! cmd stdout !!!",profileret.stdout.decode(),sep="\n",file=sys.stderr)
    print("!!! cmd stderr !!!",profileret.stderr.decode(),sep="\n",file=sys.stderr)
    sys.exit(-1)
  
  # compute Neff scores
  mmseqs_neff_filename = os.path.join(tempdir,"neff.txt")
  neffret = sp.run(
    "%s profile2neff %s %s" % (args.mmseqs,profile_filename,mmseqs_neff_filename),
    shell=True,capture_output=True,
  )
  if neffret.returncode != 0:
    print("error: failed to extract Neff scores from profileDb.",file=sys.stderr)
    print("!!! cmd stdout !!!",profileret.stdout.decode(),sep="\n",file=sys.stderr)
    print("!!! cmd stderr !!!",profileret.stderr.decode(),sep="\n",file=sys.stderr)
    sys.exit(-1)
  
  # extract Neff scores from first line of output
  # TODO: handle multiple sequences in one .a3m file!
  scores = [float(f) for f in open(mmseqs_neff_filename).read().splitlines()[1].split("\t")]
  print(json.dumps(scores),file=args.OUTFILE) # use print instead of json.dump to not have null-byte at the end
  
