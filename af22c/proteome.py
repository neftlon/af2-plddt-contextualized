import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass, field
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Generator
import json

from af22c.load_msa import MultipleSeqAlign


@dataclass
class Proteome:
    """
    This class manages MSAs for whole Proteomes.

    The proteome is expected to be a folder with the following layout:

    ```python
    msas_for_protein_path = f"{proteome_name}/msas/{uniprot_id}.a3m"
    ```
    """
    name: str
    path: Path
    msa_path: Path
    data_dir: Path
    msa_sizes_path: Path = field(init=False)
    neff_dir: Path = field(init=False)
    neff_naive_dir: Path = field(init=False)

    def __post_init__(self):
        self.msa_sizes_path = self.data_dir / f"{self.name}_msa_size.csv"
        self.neff_dir = self.data_dir / self.name / 'neffs'
        self.neff_naive_dir = self.data_dir / self.name / 'neffs_naive'

    @classmethod
    def from_folder(cls, path: str, data_dir: str = 'data'):
        proteome_path = Path(path)
        return cls(proteome_path.name,
                   proteome_path,
                   proteome_path / 'msas',
                   Path(data_dir)
                   )

    def get_msa_by_id(self, uniprot_id: str) -> MultipleSeqAlign:
        return MultipleSeqAlign.from_a3m(self.msa_path / f'{uniprot_id}.a3m')

    def get_msas(self) -> Generator[MultipleSeqAlign, None, None]:
        logging.info(f"iterating over MSAs ...")
        for uniprot_id in tqdm(self.get_uniprot_ids()):
            yield self.get_msa_by_id(uniprot_id)

    def get_uniprot_ids(self, mode='msa_available') -> set[str]:
        # Select which path to search for what kind of files.
        if mode == 'msa_available':
            search_dir = self.msa_path
            suffix = ".a3m"
        elif mode == 'neff_available':
            search_dir = self.neff_dir
            suffix = ".json"
        elif mode == 'neff_naive_available':
            search_dir = self.neff_naive_dir
            suffix = ".json"
        else:
            raise KeyError(f"Unknown mode '{mode}'.")

        # Search path and extract IDs from filenames
        filenames = [f for f in search_dir.iterdir() if f.is_file()]
        return {f.stem for f in filenames if f.suffix == suffix}

    def get_uniprot_ids_in_size(self, min_q_len=0, max_q_len=np.inf, min_n_seq=0, max_n_seq=np.inf) -> set[str]:
        msa_sizes = self.get_msa_sizes()
        in_size = msa_sizes[(msa_sizes["query_length"] <= max_q_len)
                            & (msa_sizes["sequence_count"] <= max_n_seq)
                            & (msa_sizes["query_length"] >= min_q_len)
                            & (msa_sizes["sequence_count"] >= min_n_seq)]
        return set(in_size["uniprot_id"])

    def _store_neffs(self, path, neffs):
        # NOTE: since Neff scores usually are in the area of 1k to 10k, rounding to `int` here should be sufficient
        neffs = list(map(lambda f: round(f), neffs.tolist()))

        with path.open(mode='w+') as p:
            json.dump(neffs, p)

    def compute_neff_by_id(self, uniprot_id: str):
        self.neff_dir.mkdir(parents=True, exist_ok=True)
        neff_path = self.neff_dir / f"{uniprot_id}.json"
        if neff_path.is_file():
            logging.info(f"neffs for {uniprot_id} are already cached, skipped computation")
        else:
            logging.info(f"computing neffs for {uniprot_id} ...")
            msa = self.get_msa_by_id(uniprot_id)
            neffs = msa.compute_neff()
            self._store_neffs(neff_path, neffs)

    def compute_neff_naive_by_id(self, uniprot_id: str):
        self.neff_naive_dir.mkdir(parents=True, exist_ok=True)
        neff_path = self.neff_naive_dir / f"{uniprot_id}.json"
        if neff_path.is_file():
            logging.info(f"naive neffs for {uniprot_id} are already cached, skipped computation")
        else:
            logging.info(f"computing naive neffs for {uniprot_id} ...")
            msa = self.get_msa_by_id(uniprot_id)
            neffs = msa.compute_neff_naive()
            self._store_neffs(neff_path, neffs)

    def _store_msa_sizes(self, msa_sizes: list[list]):
        size_df = pd.DataFrame(msa_sizes, columns=["uniprot_id", "query_length", "sequence_count"])
        size_df.to_csv(self.msa_sizes_path, mode='a', header=not self.msa_sizes_path.is_file(), index=False)

    def compute_msa_sizes(self):
        logging.info(f"computing MSA sizes ...")
        msa_sizes = []
        for i, msa in enumerate(self.get_msas()):
            q_len, n_seq = msa.get_size()
            q_id = msa.query_id
            msa_sizes.append([q_id, q_len, n_seq])

            # Write to file every 100 iterations
            if i % 100 == 0:
                self._store_msa_sizes(msa_sizes)
                msa_sizes = []
        if msa_sizes:
            self._store_msa_sizes(msa_sizes)
        logging.info(f"wrote MSA sizes to {self.msa_sizes_path}")

    def get_msa_sizes(self) -> pd.DataFrame:
        if self.msa_sizes_path.is_file():
            logging.info(f"msa size file found, loading ...")
            return pd.read_csv(self.msa_sizes_path)
        else:
            logging.info(f"no msa size file found")
            self.compute_msa_sizes()
            return pd.read_csv(self.msa_sizes_path)

    def plot_msa_sizes(self):
        fig_path = self.data_dir / f'{self.name}_msa_size_scatter.png'
        msa_sizes = self.get_msa_sizes()
        sns.set_style('whitegrid')
        p = sns.jointplot(data=msa_sizes, x='sequence_count', y='query_length')
        p.set_axis_labels('Number of Sequences in MSA', 'Number of Amino Acids in Query')
        p.savefig(fig_path)
        logging.info(f"saved figure to {fig_path}")


