from af22c.utils import as_handle, TarLoc
from contextlib import contextmanager
import io
from string import ascii_lowercase
import numpy as _ # prevent "Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library"
import torch
from typing import NamedTuple

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
        [stoi[ch] for ch in seq.rstrip() if not ch in ascii_lowercase],
        dtype=torch.uint8,
      )
      idx += 1
    return encmsa

class EncMsa(NamedTuple):
  """Store an encoded MSA with vocabulary and means of encoding"""
  vocab: list[str] # list of characters used in the MSA, including gaps
  stoi: dict # mapping: "residue character" -> int
  itos: dict # mapping: int -> "residue character"
  data: torch.Tensor # (N, L) matrix, where N is the number of sequences and L is the length of the query sequence
  gaptok: int # integer version of the gap token
  
  @property
  def shape(self):
    return self.data.shape
  
  @property
  def device(self):
    return self.data.device
  
  def to(self, device):
    """mimic torch behavior and move data to `device`. (return a new tuple since `self` is immutable.)"""
    params = self._asdict()
    params["data"] = self.data.to(device)
    return EncMsa(**params)
  
  @classmethod
  def from_thing(cls, thing):
    """create a new EncMsa from a torch.Tensor, file, or list of strs"""
    # vocabulary and conversion
    vocab = '-ACDEFGHIKLMNPQRSTVWXY' # TODO: do we want to infer vocab?
    stoi = {c:i for i, c in enumerate(vocab)}
    itos = {i:c for c, i in stoi.items()}
    gaptok = stoi['-']
    
    # try to convert the MSA if not done yet
    if isinstance(thing, torch.Tensor):
      encmsa = thing
    elif (
      isinstance(thing, str) or
      isinstance(thing, io.TextIOWrapper) or
      isinstance(thing, TarLoc)
    ):
      encmsa = loadmsa(thing, stoi)
    elif isinstance(thing, list):
      encseqs = []
      for idx, seq in enumerate(thing):
        if not isinstance(seq, str):
          raise ValueError(f"unable to interpret type {type(seq)} of sequence #{idx}")
        if not all(ch in vocab for ch in seq):
          raise ValueError(f"sequence #{idx} contains invalid characters")
        encseqs.append([stoi[ch] for ch in seq if not ch in ascii_lowercase])
      encmsa = torch.tensor(encseqs, dtype=torch.uint8)
    else:
      raise ValueError(f"unable to interpret type of input as msa ({type(thing)=})")
    return cls(vocab, stoi, itos, encmsa, gaptok)

@contextmanager
def as_encmsa(thing):
  """convert "any"`thing` (that is convertible to MSA) to an MSA"""
  if isinstance(thing, EncMsa): # invariant against `EncMsa`s
    yield thing
  else:
    yield EncMsa.from_thing(thing)
