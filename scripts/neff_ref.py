#!/usr/bin/env python3

import json
import sys
import argparse
from af22c.proteome import MultipleSeqAlign

def main(args=sys.argv):
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--msa", metavar="MSAFILE", type=str, required=True)
  parser.add_argument("-o", "--outfile", metavar="OUTJSON", type=argparse.FileType("w",encoding="ascii"), default=sys.stdout)
  args = parser.parse_args(args[1:])
  
  msa = MultipleSeqAlign.from_a3m(args.msa)
  scores = msa.compute_neff()
  args.outfile.write(json.dumps(scores.tolist()))
  args.outfile.write("\n")
  
if __name__ == "__main__":
  main()

