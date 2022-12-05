#!/usr/bin/env python3

import io
import tarfile


if __name__ == "__main__":
    proteome_msas_filename = "data/UP000001816_190650.tar.gz"
    with tarfile.open(proteome_msas_filename) as tar:
        print(f"opened {proteome_msas_filename}")
        names = tar.getnames()
        print(f"found around {len(names)} .a3m files containing MSAs")
        depths = {}
        for name in names:
            if name.endswith(".a3m"):
                # process .a3m member
                with tar.extractfile(name) as raw_a3m:
                    with io.TextIOWrapper(raw_a3m) as a3m:
                        contents = a3m.read()
                        lines = contents.splitlines()
                        marker_count = sum(line == lines[0] for line in lines)
                        print(f'ID "{lines[0]}" appears {marker_count} times')
