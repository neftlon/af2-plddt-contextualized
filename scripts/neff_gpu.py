#!/usr/bin/env python3

import io
import json
import sys
import tarfile
import argparse
import torch
from af22c import neff, gapcount

def open_a3m(archive_path, a3m_path):
  """
  create a handle from an a3m file in a tar file.
  the a3m_path is expected to be in the archive.
  """
  tar = tarfile.open(archive_path)
  a3m = tar.extractfile(a3m_path)
  res = io.TextIOWrapper(a3m, encoding="utf-8")
  return res

def main(args=sys.argv):
  parser = argparse.ArgumentParser(
    description="generate per-residue Neff scores blazingly fast on the GPU",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "INFILE", type=str,#argparse.FileType("r",encoding="ascii"), 
    help="location of the .a3m file containing the MSA. "
         "if an archive is given, this is interpreted to be relative to the archive root.",
  )
  parser.add_argument("OUTFILE", type=str, default="-", help="output .json file containing scores")
  parser.add_argument("-a", "--archive", metavar="TARFILE", type=str, required=False)
  parser.add_argument(
    "-m", "--mode", metavar="MODE", type=str, choices=["neff", "gapcount"], default="neff", 
    help="pick one of two computation modes (MODE=\"neff\" or MODE=\"gapcount\") for scores: (neff) score "       
         "calculation with weighting by pairwise sequence identity and (gapcount) score calculation "
         "without expensive weighting, only count gaps in each MSA column"
  )
  parser.add_argument("-d", "--device", metavar="DEV", type=str, default="cuda" if torch.cuda.is_available() else None,
                      help="pytorch device to run calculations on; options may include \"cpu\" or \"cuda\"")
  parser.add_argument("-b", "--batch-size", metavar="N", type=int, default=2**12,
                      help="specify the number of sequence pairs included in one seqeuence identity calculation batch.\n"
                           "(higher=faster, but also more (GPU) memory usage, lower=slower, but less (GPU) memory usage)")
  parser.add_argument("-l", "--gpu-mem-limit", metavar="M", type=str, default=None,
                      help="specify maximum gpu memory that should be utilized for calculations. should be given in the following form: f\"{num}{unit}\" where unit can be g for gigabytes, m for megabytes, k for kilobytes, or empty string for bytes; num specifies how many of the respective unit can be allocated. (don't specify the option to make all memory available)")
  parser.add_argument("--mmseqs",type=str,required=False,help="exists for compliance, is ignored.")
  parser.add_argument("-v", "--verbose", default=False, action="store_true")
  args = parser.parse_args(args[1:])
  
  # NB: torch.cuda.set_per_process_memory_fraction fails when passing it a torch.device at
  # the moment. therefore obtain the default gpu index here and set the memory fraction of
  # this device.
  gpuindex = torch.cuda.current_device() 
  # convert desired device to torch device
  args.device = torch.device(args.device,gpuindex)
  if args.device.type != "cuda" and args.verbose:
    print("warning: gpu not found, expect decrease in execution speed")
  
  # cap available gpu memory if desired
  if str(args.gpu_mem_limit).lower() != "none": # okok, passing "--gpu-mem-limit None" is also valid for no specifying a limit
    mul = 1 # memory limit num multiplier
    if args.gpu_mem_limit[-1] in "gmk":
      mul = {"k":2**10,"m":2**20,"g":2**30}[args.gpu_mem_limit[-1]]
      args.gpu_mem_limit = args.gpu_mem_limit[:-1]
    limitbytes = int(args.gpu_mem_limit) * mul
    totalbytes = torch.cuda.get_device_properties(args.device).total_memory
    if limitbytes < totalbytes:
      frac = limitbytes / totalbytes
      torch.cuda.set_per_process_memory_fraction(frac,args.device)
      if args.verbose:
        print("trying to cap gpu memory to",limitbytes,
              "bytes on gpu %s, this fraction is" % str(args.device),frac)
    elif args.verbose:
      print("warning: specified gpu memory limit (%d) is more than total gpu memory bytes "
            "(%d); therefore no limit will be set." % (limitbytes,totalbytes))
  
  # open input file depending on arguments
  infile = None
  if args.archive is not None:
    infile = open_a3m(args.archive,args.INFILE)
  else:
    infile = sys.stdin if args.INFILE == "-" else open(args.INFILE)

  if infile:
    # run calculations (either calculate full neff scores or only count gaps)
    scores = {
      "neff": lambda: neff(infile, device=args.device, batch_size=args.batch_size, verbose=args.verbose).tolist(),
      "gapcount": lambda: gapcount(infile, device=args.device, batch_size=args.batch_size, verbose=args.verbose).tolist(),
    }[args.mode]()
    contents = json.dumps(scores)
    
    # write outfile if there was no exception
    outfile = sys.stdout if args.OUTFILE == "-" else open(args.OUTFILE, "w")
    print(contents, file=outfile)
    if args.OUTFILE != "-": # don't close stdout
      outfile.close()
    infile.close()
  else:
    print("error: unable to generate Neff scores, input file is None",file=sys.stderr)

if __name__ == "__main__":
  main()

