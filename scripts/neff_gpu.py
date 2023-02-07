#!/usr/bin/env python3

import io
import json
import os
import sys
import tarfile
import argparse
import time
from string import ascii_lowercase
from itertools import chain
import torch
from tqdm import tqdm
from af22c.proteome import MultipleSeqAlign
from af22c.utils import as_handle

def loadmsa(path, stoi):
  """load an MSA as an encoded tensor from a file-like object and a mapping dict stoi"""
  with as_handle(path) as f:
    lines = f.readlines()
    assert not len(lines) & 1, f"MSA at {path} should contain an even number of lines"
    query_id = lines[0].split()[0]
    query_len = len([() for ch in lines[1].strip() if not ch in ascii_lowercase])
    
    # count header lines that are not just the query ID (1+ because first query is included in encoded msa)
    num_seqs = 1 + sum([1 for line in lines[::2] if not line[:-1] == query_id and line.startswith(">")])
    
    # encode msa from lines
    encmsa = torch.zeros((num_seqs,query_len), dtype=torch.uint8)
    idx = 0
    for hdr, seq in zip(lines[::2], lines[1::2]):
      # skip query that appears multiple times in the file
      if idx > 1 and hdr[:-1] == query_id:
         continue
      # encode sequence
      encmsa[idx,:] = torch.tensor(
        # NB: stoi comes from outside!
        [stoi[ch] for ch in seq[:-1] if not ch in ascii_lowercase],
        dtype=torch.uint8,
      )
      idx += 1
    return encmsa

def pwseq(encmsa, device=None, batch_size=2**12, **kwargs):
  """return pairwise sequence identity calculated with pytorch"""
  num_seqs,query_len = encmsa.shape
  
  # calculate all pairs for which pairwise sequence identities need to be calculated
  pairs = torch.triu_indices(*(num_seqs,)*2, 1, device=device).T
  
  # each batch should yield a matrix with batch_size elements
  num_batches = (len(pairs) + batch_size - 1) // batch_size
  num_full_batches = len(pairs) // batch_size
  
  batch_pairs = pairs[:-(len(pairs)%batch_size)]
  rest_pairs = pairs[-(len(pairs)%batch_size):]
  
  # put pairs into batches
  batches = batch_pairs.view(num_full_batches, -1, 2)
  if num_batches != num_full_batches:
    batches = chain(batches, [rest_pairs])
    
  # one batch contains batch_size many pairs, which yields batch_size many similarity scores because the 
  # similarity matrix is symmetric.
  bpwseq = torch.eye(num_seqs, device=device) # matrix containing similarity scores for two sequences
  for batch_pairs in tqdm(batches, total=num_batches, desc="running batches"):
    # extract sequences in batch
    batch_seqs = encmsa[batch_pairs]

    # calculate pairwise distances 
    batch_pwdists = torch.sum(batch_seqs[:,0,:] != batch_seqs[:,1,:], axis=-1)
    batch_pwseq = 1 - batch_pwdists / query_len
    
    # put at right location in result matrix (and make symmetric)
    bpwseq[batch_pairs[:,0],batch_pairs[:,1]] = batch_pwseq
    bpwseq[batch_pairs[:,1],batch_pairs[:,0]] = batch_pwseq
  
  return bpwseq
  
def gapcount(encmsa, weights=None, gaptok=None, stoi=None, nongap=False, **kwargs):
  """
  calculate number of gaps in each msa column. returns a vector of length query_len.
  weights can be specified to weight each sequence in the msa column.
  if nongap=False (default), gaps will be counted; otherwise non-gaps will be counted.
  the gaps are identified by gaptok, which can be the token id for gaps. alternatively, a
  dictionary stoi can be supplied, where the gaptok is looked up.
  """
  if gaptok is None:
    assert stoi is not None
    gaptok = stoi['-']
  gapindicator = encmsa != gaptok if nongap else encmsa == gaptok
  if weights is not None:
    return weights @ gapindicator.float()
  return torch.sum(gapindicator, dim=0)

def neff(msa, pwseqfn=pwseq, seqid_thres=0.8, **kwargs):
  """calculate neff scores for an encoded msa."""
  device = kwargs.get("device")
  
  # encoding/decoding  
  vocab = '-ACDEFGHIKLMNPQRSTVWXY' # TODO: do we want to infer vocab?
  stoi = {c:i for i, c in enumerate(vocab)}
  itos = {i:c for c, i in stoi.items()}
  gaptok = stoi['-']
  
  # encode msa if necessary
  if isinstance(msa, MultipleSeqAlign):
    # collect all sequences, including query
    allseqs = [msa.query_seq] + [match.aligned_seq for match in msa.matches]
    encmsa = torch.zeros((len(msa.matches)+1, len(msa.query_seq)))
    for seqidx, seq in tqdm(enumerate(allseqs), desc="converting msa"):
      for colidx, colval in enumerate(seq):
        encmsa[seqidx, colidx] = stoi[colval]
  elif isinstance(msa, torch.Tensor):
    encmsa = msa
  elif isinstance(msa, str) or isinstance(msa, io.TextIOWrapper):
    encmsa = loadmsa(msa, stoi)
  else:
    raise ValueError(f"unable to interpret type of input msa ({type(msa)=})")
  
  # put msa on desired device, if it is not there already
  if encmsa.device != device:
    encmsa = encmsa.to(device)
  
  num_seqs, query_len = encmsa.shape
  eq = pwseqfn(encmsa, **kwargs)
  # calculate neff weights (dim can be 0 or 1, does not matter because pwseq is symmetric)
  neffweights = 1 / torch.sum(eq >= seqid_thres, dim=0)
  return gapcount(encmsa, weights=neffweights, nongap=True, gaptok=gaptok, **kwargs)

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
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-m", "--msa", metavar="MSAFILE", type=argparse.FileType("r",encoding="ascii"))
  parser.add_argument("-am", "--archive-msa", metavar="MSAINARCHIVE", type=str, required=False)
  parser.add_argument("-a", "--archive", metavar="TARFILE", type=str, required=False)
  parser.add_argument("-o", "--outfile", metavar="OUTJSON", type=argparse.FileType("w",encoding="ascii"), default=sys.stdout)
  parser.add_argument("-d", "--device", metavar="DEV", type=str, default="cuda" if torch.cuda.is_available() else None,
                      help="pytorch device to run calculations on; options may include \"cpu\" or \"cuda\"")
  parser.add_argument("-b", "--batch-size", metavar="N", type=int, default=2**12,
                      help="specify the number of sequence pairs included in one seqeuence identity calculation batch.\n"
                           "(higher=faster, but also more (GPU) memory usage, lower=slower, but less (GPU) memory usage)")
  args = parser.parse_args(args[1:])
  
  if args.device != "cuda":
    print("warning: gpu not found, expect decrease in execution speed")
  
  # open input file depending on arguments
  infile = None
  if args.msa is not None:
    infile = args.msa
    if args.archive is not None or args.archive_msa is not None:
      print("warning: --archive-msa/-am/--archive/-a options will be ignored when specifying an MSAFILE")
  elif args.archive is not None and args.archive_msa is not None:
    infile = open_a3m(args.archive,args.archive_msa)
  else:
    print("error: please specify either an MSA or an archive with path to the MSA in the archive")

  if infile:
    # run calculations
    scores = neff(infile, device=args.device, batch_size=args.batch_size).tolist()
    args.outfile.write(json.dumps(scores))
    args.outfile.write("\n")
    infile.close()

if __name__ == "__main__":
  main()

