#!/usr/bin/env python3

# count number of structure prediction files with more than one fragment
# usage:
# tar -tf data/UP000005640_9606_HUMAN_v3.tar | ./scripts/afdb_fragmentation.py -

import tarfile
import argparse
import re
from collections import defaultdict

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
  "NAMES", type=argparse.FileType(),
  help="specify a list of structure prediction files, specify - to read from stdin"
)
parser.add_argument(
  "-r", "--regex", type=str,
  help="specify a regex string to parse `id` and `fragnum` from",
  default=r"AF-(?P<id>([A-Z]|[0-9])+)-F(?P<fragnum>[0-9]+)-model_v[0-9]+\.pdb\.gz"
)
args = parser.parse_args()

# create mapping "protein id" to list of "fragment numbers"
expr = re.compile(args.regex)
id2frags = defaultdict(list) # mapping: "protein id":str -> list["fragment number":int]
for name in args.NAMES:
  m = expr.match(name)
  if m is not None:
    # check that required groups are available
    assert (
      "id" in m.groupdict() and "fragnum" in m.groupdict()
    ), "regular expression must include `id` and `fragnum` groups!"
    # add fragnum to respective protein
    id2frags[m["id"]] += [int(m["fragnum"])]

# check that there are no duplicates
id2count = {} # mapping: "protein id":str -> "number of fragments":int
for protid, fragnums in id2frags.items():
  if len(fragnums) != len(set(fragnums)):
    print(f"error: protein {protid} contains multiple fragments with same name {fragnums}")
  else:
    id2count[protid] = len(fragnums)

# print statistics
print("total number of proteins:", len(id2count))
counts = defaultdict(int)
for protid, numfrags in id2count.items():
  counts[numfrags] += 1
print("number of fragments\toccurrences")
for numfrags, numoccurrences in counts.items():
  print(numfrags, numoccurrences, sep="\t")
