#!/usr/bin/env python3

"""collect json files containing Neff scores from a directory, 
round them (don't need so many (un)important decimals), and put them in a 
single json file."""

import os
import sys
import json
from tqdm import tqdm

SOURCE_DIR="./data/test"
PROTEOME_NAME="UP000005640_9606"
TARGET_FILE=f"./data/{PROTEOME_NAME}_neff_fast.json"

# gather entries
entries = {}
for fn in tqdm(os.listdir(SOURCE_DIR), desc="gathering entries"):
  if fn.endswith(".json"):
    path = os.path.join(SOURCE_DIR, fn)
    with open(path) as f:
      try:
        xs = json.load(f)
      except json.decoder.JSONDecodeError:
        tqdm.write("failed to load %s: JSONDecodeError" % path)
      else:
        entries[fn[:-5]] = [round(x,2) for x in xs]

# write out file in compact json representation
with open(TARGET_FILE,"w") as f:
  json.dump(entries,f,separators=(',', ':'))
print("dumped entries to %s" % TARGET_FILE)

