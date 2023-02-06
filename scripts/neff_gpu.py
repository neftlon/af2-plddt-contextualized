import torch
from af22c.proteome import MultipleSeqAlign
from tqdm import tqdm

def pwseq(encmsa, device=None, batch_size=100000, **kwargs):
  """return pairwise sequence identity calculated with pytorch"""
  num_seqs,query_len = encmsa.shape
  
  # calculate all pairs for which pairwise sequence identities need to be calculated
  pairs = []
  for i in tqdm(range(num_seqs), desc="gathering pairs"):
    for j in range(i+1, num_seqs):
      pairs.append((i,j))
  pairs = torch.tensor(pairs)
  
  # each batch should yield a matrix with batch_size elements
  num_batches = (len(pairs) + batch_size - 1) // batch_size
  
  # one batch contains batch_size many pairs, which yields batch_size many similarity scores because the 
  # similarity matrix is symmetric.
  bpwseq = torch.eye(num_seqs) # matrix containing similarity scores for two sequences
  for batch_idx in tqdm(range(num_batches), desc="running batches"):
    # calculate similarity scores for a batch
    pairs_idx = torch.arange(batch_idx*batch_size, min((batch_idx + 1)*batch_size, len(pairs)))
    batch_pairs = pairs[pairs_idx]
    batch_pairs_flat = batch_pairs.view(-1)
    
    # calculate sequences in batch
    batch_seqs = encmsa[batch_pairs_flat]
    batch_seqs = batch_seqs.view(-1, 2, query_len)

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
  """
  calculate neff scores for an encoded msa.
  """
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
  else:
    encmsa = msa
  
  num_seqs, query_len = encmsa.shape
  eq = pwseqfn(encmsa, **kwargs)
  # calculate neff weights (dim can be 0 or 1, does not matter because pwseq is symmetric)
  neffweights = 1 / torch.sum(eq >= seqid_thres, dim=0)
  return gapcount(encmsa, weights=neffweights, nongap=True, gaptok=gaptok, **kwargs)

