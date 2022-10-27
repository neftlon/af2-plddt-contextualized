#!/usr/bin/env python3

"""
This script extracts fragments of proteins from AlphaFold's database that contain only one fragment.

(Proteins containing more fragment cannot be concatenated trivially since the different fragments overlap each other.)
"""

import json
import itertools
import argparse
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Only preserve proteins that consist of one fragment in the output file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # print argument defaults
    )
    parser.add_argument(
        "--infile",
        "-i",
        "--in",
        type=argparse.FileType("r"),
        default="data/UP000005640_9606_HUMAN_v3_plddts.json",
        help="location of input .json file; the file containing proteins with AlphaFold's `-Fn` extension",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        "--out",
        type=argparse.FileType("w"),
        default="data/UP000005640_9606_HUMAN_v3_plddts_fltrd.json",
        help="location of output .json file; the file containing proteins without AlphaFold's `-Fn` extension",
    )
    args = parser.parse_args()

    logging.debug("loading proteins")
    plddts = json.load(args.infile)  # mapping: UniProtID (per fragment) -> list of pLDDT scores
    result = {}  # mapping: UniProtID -> list of pLDDT scores

    # store per-protein fragments
    logging.debug("storing per-protein fragments")
    frags_by_id = {}
    for full_id in plddts:
        if "-F" in full_id:
            uniprot_id, fn = full_id.split("-F")
            fn = int(fn)

            if uniprot_id not in frags_by_id:
                frags_by_id[uniprot_id] = {}

            assert fn not in frags_by_id[uniprot_id]
            frags_by_id[uniprot_id][fn] = plddts[full_id]
        else:
            # append to result directly if the protein is not fragmented
            assert full_id not in result, "since `plddts` is already a `dict`, the key should not appear twice!"
            result[full_id] = plddts[full_id]

    # filter fragments
    logging.debug("filtering fragments")
    for prot_id, frags in frags_by_id.items():
        frags = sorted(frags.items(), key=lambda kv: kv[0])
        frags = list(map(lambda kv: kv[1], frags))

        # do not include proteins with more than one fragment
        if len(frags) > 1:
            continue

        frags = itertools.chain(*frags)
        frags = list(frags)
        assert prot_id not in result
        result[prot_id] = frags

    logging.debug("dumping filtered proteins")
    json.dump(result, args.outfile)
            

