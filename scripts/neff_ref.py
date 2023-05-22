#!/usr/bin/env python3

import json
import sys
import os
import argparse
from af22c import neff_ref

def main(args=sys.argv):
  # parse arguments
  parser = argparse.ArgumentParser(description="generate per-residue Neff scores blazingly fast with mmseqs")
  parser.add_argument("INFILE",type=argparse.FileType(),help="input .a3m MSA file")
  parser.add_argument("OUTFILE",type=argparse.FileType("w",encoding="ascii"),help="output .json file containing per-residue scores")
  parser.add_argument(
    "--mmseqs",type=str,default=dict(os.environ).get("MMSEQS","mmseqs"),
    help="specify mmseqs executable. note that the version of mmseqs must be "
         "such that the profile2neff command is available!",
  )
  compli = "exists for compliance, is ignored."
  parser.add_argument("-d", "--device", type=str, default=None, metavar="?", required=False, help=compli)
  parser.add_argument("-b", "--batch-size", type=int, default=None, metavar="?", required=False, help=compli)
  parser.add_argument("-l", "--gpu-mem-limit", type=str, required=False, metavar="?", default=None, help=compli)
  args = parser.parse_args(args[1:])
  
  scores = neff_ref(args.INFILE)
  args.OUTFILE.write(json.dumps(scores.tolist()))
  args.OUTFILE.write("\n")
  
if __name__ == "__main__":
  main()

