import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="Input directory")
parser.add_argument("output_file", help="Output file")
args = parser.parse_args()
input_dir = args.input_dir
output_file = args.output_file

data = {}
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename)) as f:
            file_prefix = os.path.splitext(filename)[0]
            data[file_prefix] = json.load(f)

with open(output_file, "w") as f:
    json.dump(data, f)
