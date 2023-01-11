#!/usr/bin/env python

import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: %s selectionfile sourcedir" % sys.argv[0])
        sys.exit(2)
    selection_filename, source_dirname = sys.argv[1:]
