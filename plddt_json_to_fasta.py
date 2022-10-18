#!/usr/bin/env python3

"""
Write the mapping (UniProtID -> plDDT scores) from a .json file to a .fasta file.
If there is no out file specified, the .fasta contents are dumped to stdout.

The output file contains two lines for each protein, the first contains the UniProtID and the second line contains a
list of comma-separated plDDT values.

Example:
```
>X6R8D5
43.9,47.84,53.25,47.35,58.27,42.5,45.51,52.87,45.72 ...
>W6CW81
45.73,47.9,55.16,55.1,52.65,55.1,58.14,59.75,58.63 ...
...
```

Usage:
{} <infile.json> [outfile.fasta]
"""

import json
import sys


def main():
    # chicken out early if # of arguments is invalid
    args = sys.argv
    if len(args) not in [2, 3]:
        print(__doc__.format(sys.argv[0]))
        return

    with open(args[1]) as infile:
        # open the output file, if available
        outfile = open(args[2], "w") if len(args) == 3 else sys.stdout

        # dump the data into the output stream
        data = json.load(infile)
        assert type(data) == dict
        for prot_id, plddts in data.items():
            plddts = list(map(str, plddts))
            plddts = ",".join(plddts)
            print(f">{prot_id}\n{plddts}", file=outfile)

        # don't close stdout
        if len(args) == 3:
            outfile.close()


if __name__ == "__main__":
    main()
