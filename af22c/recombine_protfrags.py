#!/usr/bin/env python3

import json
import itertools

if __name__ == "__main__":
    # TODO(johannes): Make this independent of single files
    infilename = "data/UP000005640_9606_HUMAN_v3_plddts.json"
    outfilename = "data/UP000005640_9606_HUMAN_v3_plddts_defrag.json"

    with open(infilename) as infile:
        plddts = json.load(infile)

        # store per-protein fragments
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
                # TODO: append to result directly
                raise Exception(f"protein id {full_id} does not contain a \"-F\"!")

        # concatenate fragments
        result = {}
        for prot_id, frags in frags_by_id.items():
            frags = sorted(frags.items(), key=lambda kv: kv[0])
            frags = map(lambda kv: kv[1], frags)
            frags = itertools.chain(*frags)
            frags = list(frags)
            assert prot_id not in result
            result[prot_id] = frags

        with open(outfilename, "w") as outfile:
            json.dump(result, outfile)
            

