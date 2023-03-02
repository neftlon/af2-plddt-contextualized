#!/usr/bin/env python3

"""collect json files containing Neff scores from a directory, 
round them (don't need so many (un)important decimals), and put them in a 
single json file."""

import argparse
from dataclasses import dataclass
import json
import os
import subprocess as sp
import sys
from tqdm import tqdm

# parse arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("SOURCE",type=str,default="./data/test",help="source directory or archive (.tar or .tar.gz)")
parser.add_argument("DESTFILE",type=argparse.FileType("w"),help="output .json file")
args = parser.parse_args()

# look which files should be processed
srcmode = None
prot_files = []
if any(args.SOURCE.endswith(ext) for ext in [".tar",".tar.gz"]):
  srcmode = "archive"
  tarres = sp.run(f"tar -tf {args.SOURCE}",shell=True,capture_output=True)
  if tarres.returncode == 0:
    prot_files = tarres.stdout.decode().split()
  else:
    print("error: failed to get files from input archive",file=sys.stderr)
    print("!!! cmd stdout !!!",tarres.stdout.decode(),sep="\n",file=sys.stderr)
    print("!!! cmd stderr !!!",tarres.stderr.decode(),sep="\n",file=sys.stderr)
    sys.exit(-1)
elif os.path.isdir(args.SOURCE):
  srcmode = "dir"
  prot_files = os.listdir(args.SOURCE)
if srcmode is None:
  print("error: failed to determine input file type")
  sys.exit(-1)

# gather entries
entries,xs = {},[]
for fn in tqdm(prot_files, desc="gathering entries"):
  if fn.endswith(".json"):
    try:
      if srcmode == "dir":
        path = os.path.join(args.SOURCE, fn)
        with open(path) as f:
          xs = json.load(f)
      elif srcmode == "archive":
        z = "z" if args.SOURCE.endswith(".gz") else ""
        tarres = sp.run(f"tar -{z}xOf {args.SOURCE} {fn}",shell=True,capture_output=True,check=True)
        xs = json.loads(tarres.stdout.decode())
    except json.decoder.JSONDecodeError:
      tqdm.write("failed to load %s: JSONDecodeError" % fn)
    except sp.CalledProcessError as tarerr:
      tqdm.write(f"error: failed to get {fn} from input archive")
      tqdm.write("!!! cmd stdout !!!\n" + tarerr.stdout.decode())
      tqdm.write("!!! cmd stderr !!!\n" + tarerr.stderr.decode())
    else:
      entries[fn[:-5]] = [round(x,2) for x in xs]

# write out file in compact json representation
json.dump(entries,args.DESTFILE,separators=(',', ':'))
print("dumped entries to %s" % args.DESTFILE.name)

