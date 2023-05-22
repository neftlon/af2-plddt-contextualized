from af22c.msa import as_encmsa
from itertools import chain
import numpy as _ # prevent "Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library"
import torch
from tqdm import tqdm

def pwseq(msa, device=None, batch_size=2**12, verbose=True, **kwargs):
  """return pairwise sequence identity calculated with pytorch"""
  with as_encmsa(msa) as encmsa:
    num_seqs,query_len = encmsa.shape
    
    # calculate all pairs for which pairwise sequence identities need to be calculated
    pairs = torch.triu_indices(*(num_seqs,)*2, 1, device=device).T
    
    # each batch should yield a matrix with batch_size elements
    num_batches = (len(pairs) + batch_size - 1) // batch_size
    num_full_batches = len(pairs) // batch_size
    
    batch_pairs = pairs[:-(len(pairs)%batch_size)]
    rest_pairs = pairs[-(len(pairs)%batch_size):]
    
    # put pairs into batches
    batches = batch_pairs.view(max(num_full_batches,1), -1, 2)
    if num_batches != num_full_batches:
      batches = chain(batches, [rest_pairs])
      
    # one batch contains batch_size many pairs, which yields batch_size many similarity scores because the 
    # similarity matrix is symmetric.
    bpwseq = torch.eye(num_seqs, device=device) # matrix containing similarity scores for two sequences
    
    for batch_pairs in (
      tqdm(batches, total=num_batches, desc="running batches")
      if verbose else
      batches
    ):
      # extract sequences in batch
      batch_seqs = encmsa.data[batch_pairs]

      # calculate pairwise distances 
      batch_pwdists = torch.sum(batch_seqs[:,0,:] != batch_seqs[:,1,:], axis=-1)
      batch_pwseq = 1 - batch_pwdists / query_len
      
      # put at right location in result matrix (and make symmetric)
      bpwseq[batch_pairs[:,0],batch_pairs[:,1]] = batch_pwseq
      bpwseq[batch_pairs[:,1],batch_pairs[:,0]] = batch_pwseq
    
    return bpwseq
  
def gapcount(msa, weights=None, nongap=False, **kwargs):
  """
  calculate number of gaps in each msa column. returns a vector of length query_len.
  weights can be specified to weight each sequence in the msa column.
  if nongap=False (default), gaps will be counted; otherwise non-gaps will be counted.
  """
  with as_encmsa(msa) as encmsa:
    gapindicator = encmsa.data != encmsa.gaptok if nongap else encmsa.data == encmsa.gaptok
    if weights is not None:
      return weights @ gapindicator.float()
    return torch.sum(gapindicator, dim=0)

def neff(msa, pwseqfn=pwseq, seqid_thres=0.8, **kwargs):
  """calculate neff scores for an encoded msa."""
  device = kwargs.get("device")
  
  with as_encmsa(msa) as encmsa:
    # put msa on desired device, if it is not there already
    if encmsa.device != device:
      encmsa = encmsa.to(device)
    
    num_seqs, query_len = encmsa.shape
    eq = pwseqfn(encmsa, **kwargs)
    # calculate neff weights (dim can be 0 or 1, does not matter because pwseq is symmetric)
    neffweights = 1 / torch.sum(eq >= seqid_thres, dim=0)
    return gapcount(encmsa, weights=neffweights, nongap=True, **kwargs)