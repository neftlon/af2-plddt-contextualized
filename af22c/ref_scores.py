from typing import Callable, IO, Optional, NamedTuple
from dataclasses import dataclass, field
import multiprocessing as mp
from tqdm import tqdm
import string
from af22c.utils import as_handle
from itertools import repeat, chain
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import defaultdict

# NOTE: for pickle-ability, this function must live outside of neff_ref
def _batched_one_against_many_res_id(combined_args):
    """
    Compute the number of identical residues between a batch of input sequences and a complete MSA.
    """
    idx, msa_vec = combined_args
    return [np.sum(msa_vec[i:] == msa_vec[i], axis=1) for i in idx]

def neff_ref(msa, mode="neff"):
    """
    Original implementation of Neff calculation. 

    This function is here for legacy reasons, it can be used to compare the speed difference with the faster GPU-based version.

    Args:
        msa: The filename or location of an .a3m file containing an MSA
        mode: The mode to use for Neff calculation. Can be either "neff" or "gapcount". Defaults to "neff".

    Returns:
        A list of Neff scores, one score for each position in the MSA.

    Examples:
        >>> from af22c.utils import ProteomeTar
        >>> from af22c.neff_ref import neff_ref
        >>> # open a .tar file containing human protein MSAs
        >>> human = ProteomeTar("data/UP000005640_9606.tar")
        >>> p59090 = human.get_protein_location("P59090")
        >>> # calculate Neff scores for P59090
        >>> neff_ref(p59090)
    """

    # TODO: this function is the only place that makes the project depend on Biopython. -> decide how we want to manage the sitation with this dependency
    from Bio import SeqIO
    from Bio.Seq import Seq

    # manual .a3m file parsing
    class MsaMatchAttribs(NamedTuple):
        """Attributes from a match in a .a3m file"""
        aln_score: int
        seq_identity: float
        eval: float
        qstart: int
        qend: int
        qlen: int
        tstart: int
        tend: int
        tlen: int

    # Generate translation table for lowercase removal
    LOWERCASE_DEL_TABLE = str.maketrans("", "", string.ascii_lowercase)

    @dataclass
    class MsaMatch:
        """
        MsaMatch object containing the parsed header field in attribs, the original sequence in orig_seq    and the sequence without insertions.
        """
        target_id: str
        attribs: Optional[MsaMatchAttribs]
        orig_seq: Seq  # with insertions
        aligned_seq: Seq = field(init=False)  # without insertions

        def __post_init__(self):
            # Convert to String and back in order to use string translation method instead of
            # Biopython translation. This is still way faster than filtering and joining.
            self.aligned_seq = Seq(str(self.orig_seq).translate(LOWERCASE_DEL_TABLE))

        def __getitem__(self, item: int):
            """Get a residue by index. (Including gaps from MSA.)"""
            return self.aligned_seq[item]

        def __str__(self):
            return str(self.aligned_seq)

        def __len__(self):
            return len(self.aligned_seq)

    def flatten_list_of_lists(l):
        """
        Faster than list comprehension solutions.
        """
        return list(chain.from_iterable(l))

    def sort_by_idx(idx, l):
        """
        Sort a list by an index given as list of indices.
        """
        return [x for _, x in sorted(zip(idx, l))]

    def seq_identity_parallel(msa):
        # Prepare msa for vector operations.
        msa_vec = np.array([list(seq) for seq in msa])

        # Distribute the sequence indices over multiple batches.
        # TODO use default number of workers (i.e. number of cpus) and read it out
        #  for splitting the MSA in batches.
        n_batches = mp.cpu_count()
        batch_idxs = [range(i, len(msa_vec), n_batches) for i in range(n_batches)]

        with ProcessPoolExecutor(max_workers=n_batches) as ppe:
            pairwise_input = zip(batch_idxs, repeat(msa_vec))
            # TODO add some kind of progress indicator for the batched
            #  multiprocessing, maybe report from called function at specific
            #  intervals and accumulate progress here?
            #  Beware, with the current batching early iterations are slower than later ones.
            #  This could be changed by shuffling and then reordering.
            logging.info(f" mapping one-against-many to {n_batches} jobs")
            batched_n_ident_res_list = list(
                tqdm(ppe.map(_batched_one_against_many_res_id, pairwise_input))
            )

            # TODO The following generation of the full matrix from the upper triangle matrix is another bottleneck.
            #  We should speed it up for computing bigger MSAs.
            n_ident_res_list_ur = flatten_list_of_lists(batched_n_ident_res_list)
            idx = flatten_list_of_lists(batch_idxs)

            n_ident_res_list_ur = sort_by_idx(idx, n_ident_res_list_ur)

            # Build an upper triangle matrix from the computed list of lists, where each list
            # represents one row of the upper right triangle matrix.
            n_ident_res_vec = np.zeros((len(msa), len(msa)))
            n_ident_res_vec[np.triu_indices_from(n_ident_res_vec)] = flatten_list_of_lists(
                n_ident_res_list_ur
            )

            # Fill the lower left triangle matrix with the values from the upper right triangle
            n_ident_res_vec += n_ident_res_vec.T - np.diag(n_ident_res_vec.diagonal())

            return np.array(n_ident_res_vec) / len(msa[0])

    def extract_query_and_matches(a3m_handle) -> tuple[str, str, list[MsaMatch]]:
        # TODO(johannes): do we really want _this_ parameter? or should we accept filename strings as well?
        seqs = list(SeqIO.parse(a3m_handle, "fasta"))
        query = seqs[0]  # first sequence is the query sequence
        matches = []
        logging.info("loading MSA")
        for idx, seq in tqdm(enumerate(seqs[1:]), total=len(seqs) - 1):
            raw_attribs = seq.description.split("\t")
            target_id, *remaining_attribs = raw_attribs

            # TODO(johannes): Sometimes (for instance in Q9A7K5.a3m) the MSA file contains the same (presumable) query
            # sequence at least twice. What purpose does this serve? The code below currently skips these duplications, but
            # this is probably just wrong behavior.
            if (target_id == query.id) and not remaining_attribs:
                logging.warning(f"query_id {query.id} appearing for a second time in .a3m file, skipping sequence")
                continue

            attribs = None
            if len(remaining_attribs) == len(MsaMatchAttribs._fields):
                # typecast fields from str to their respective type before creating the match attribs object
                attribs = MsaMatchAttribs(*[
                    MsaMatchAttribs.__annotations__[field_name](field_value)
                    for field_name, field_value in zip(MsaMatchAttribs._fields, remaining_attribs)
                ])
            else:
                logging.warning(
                    f"a3m file contains a match at index {idx} (of {len(seqs)} matches) that contains not the "
                    f"required (={len(MsaMatchAttribs._fields)}) number of fields: {len(raw_attribs)}. "
                    f'match description: "{seq.description}"'
                )

            match = MsaMatch(target_id, attribs, seq.seq)
            matches.append(match)
        return query.id, query.seq, matches

    @dataclass
    class MultipleSeqAlign:
        """
        This class manages MSAs.
        Insertions are removed from matches upon loading.
        """

        query_id: str
        query_seq: Seq
        matches: list[MsaMatch]

        @classmethod
        def from_a3m(cls, filething: str | Path | IO):
            with as_handle(filething) as a3m:
                # TODO make extract_query_and_matches a method of the MSA class
                return cls(*extract_query_and_matches(a3m))

        # TODO Implement sanity check looking for 'X', '-' and other strange things in query sequence as well as
        #  consistent sequence lengths.

        def get_size(self) -> tuple[int, int]:
            return len(self.query_seq), len(self.matches) + 1  # +1 for query sequence

        def vectorize(self):
            # TODO test
            msa = [self.query_seq] + self.matches
            return np.array([list(seq) for seq in msa])

        def compute_neff(self):
            seq_id=0.8
            msa = [self.query_seq] + self.matches
            pair_seq_id = seq_identity_parallel(msa)

            n_eff_weights = np.zeros(len(msa))
            logging.info(" calulating Neff weights...")
            for i in tqdm(range(len(msa)), total=len(msa)):
                n_eff_weights[i] = sum(map(int, pair_seq_id[i] >= seq_id))
            inv_n_eff_weights = 1 / n_eff_weights

            n_non_gaps = np.zeros(len(self.query_seq))
            logging.info(" counting gaps...")
            for c in tqdm(range(len(self.query_seq)), total=len(self.query_seq)):
                for i, m in enumerate(msa):
                    n_non_gaps[c] += int(m[c] != "-") * inv_n_eff_weights[i]
            return n_non_gaps

        def compute_neff_naive(self):
            msa = [self.query_seq] + self.matches

            n_non_gaps = np.zeros(len(self.query_seq))
            logging.info(" counting gaps...")
            for c in tqdm(range(len(self.query_seq)), total=len(self.query_seq)):
                for i, m in enumerate(msa):
                    n_non_gaps[c] += int(m[c] != "-")
            return n_non_gaps

        def examine_duplicates(self):
            logging.info(f"looking for duplicates in MSA ...")
            # TODO find out why duplicates occur
            seqs_by_id = defaultdict(list)
            seqs_by_id[self.query_id] = [str(self.query_seq)]
            for m in self.matches:
                seqs_by_id[m.attribs.target_id].append(str(m.aligned_seq))

            n_dupl = 0
            for prot_id, seqs in seqs_by_id.items():
                if len(seqs) > 1:
                    n_dupl += 1
                    seqs_string = "\n".join(seqs)
                    logging.info(
                        f"The ID {prot_id} appears more than once with these sequences:\n"
                        f"{seqs_string}"
                    )
            if n_dupl:
                logging.info(
                    f"In total {n_dupl} of {len(seqs_by_id)} IDs occur more than once!"
                )
            else:
                logging.info(f"No duplicates found!")
            return n_dupl
        
    msa = MultipleSeqAlign.from_a3m(msa)
    if mode == "neff":
        return msa.compute_neff()
    elif mode == "gapcount" or mode == "naive":
        return msa.compute_neff_naive()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    