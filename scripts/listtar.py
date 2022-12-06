#!/usr/bin/env python3

"""
List the contents of a tar file using the `tarfile.list` function.
"""

import tarfile
import sys


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <filename.tar>")
        sys.exit(-1)
    with tarfile.open(sys.argv[1]) as f:
        f.list()
