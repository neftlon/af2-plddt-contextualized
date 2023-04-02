import os
import json

input_dir = "data/cluster/UP000005640_9606/neffs_naive"
output_file = "data/cluster/UP000005640_9606/UP000005640_9606_neff_naive.json"

data = {}
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename)) as f:
            file_prefix = os.path.splitext(filename)[0]
            data[file_prefix] = json.load(f)

with open(output_file, "w") as f:
    json.dump(data, f)
