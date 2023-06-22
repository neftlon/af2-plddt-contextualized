"""
This file contains code for calling mmseqs from our framework to calculate Neff
scores. Call `neff` from the base module with mode set to mmseqs to use this
functionality.
"""

import os
import subprocess as sp
import sys
import struct
from tempfile import TemporaryDirectory
from .utils import as_handle

def mmseqs_neff_from_a3m(a3m_file_or_path, mmseqs: str = "mmseqs"):
  with as_handle(a3m_file_or_path) as a3m_handle:
    with TemporaryDirectory() as tempdir:
      infile = a3m_handle.read()
      
      # this hack creates a valid mmseqs db from a single .a3m file in 3 steps
      # (by default mmseqs cannot use .a3m files as db input)
      # 1) copy original a3m file and append a null byte
      dbfilename = os.path.join(tempdir,"msaDb")
      with open(dbfilename, "wb") as dbfile:
        dbfile.write(infile.encode())
        dbfile.write(struct.pack("B", 0))
      # 2) write a .dbtype file specifying 0xb = 0d11 as mode, which indicates a3m
      with open(f"{dbfilename}.dbtype", "wb") as dbtype:
        dbtype.write(struct.pack("<Bxxx", 0xb))
      # 3) write a .index file specifying a single entry
      with open(f"{dbfilename}.index", "w") as dbindex:
        size = len(infile)  # write size of ORIGINAL file
        dbindex.write("\t".join(["0", "0", str(size)]) + "\n")
      
      # create an mmseqs profile
      profile_filename = os.path.join(tempdir,"profileDb")
      profileret = sp.run(
        "%s msa2profile %s %s" % (mmseqs,dbfilename,profile_filename),
        shell=True,stdout=sp.PIPE,stderr=sp.PIPE,
      )
      if profileret.returncode != 0:
        raise RuntimeError(
          f"failed to convert MSA to profile.\n"
          f"cmd stdout:\n{profileret.stdout.decode()}\n"
          f"cmd stderr:\n{profileret.stderr.decode()}"
        )
      
      # compute Neff scores
      mmseqs_neff_filename = os.path.join(tempdir,"neff.txt")
      neffret = sp.run(
        "%s profile2neff %s %s" % (mmseqs,profile_filename,mmseqs_neff_filename),
        shell=True,stdout=sp.PIPE,stderr=sp.PIPE,
      )
      if neffret.returncode != 0:
        raise RuntimeError(
          f"failed to extract Neff scores from profileDb.\n"
          f"cmd stdout:\n{profileret.stdout.decode()}\n"
          f"cmd stderr:\n{profileret.stderr.decode()}"
        )
      
      # extract Neff scores from first line of output
      # TODO: handle multiple sequences in one .a3m file!
      return [
        float(f) for f in open(mmseqs_neff_filename).read().splitlines()[1].split("\t")
      ]

