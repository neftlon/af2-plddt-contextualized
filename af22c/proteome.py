import pandas as pd
import seaborn as sns
from dataclasses import dataclass, field
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Generator

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

    def __post_init__(self):
        self.msa_sizes_path = self.data_dir / f"{self.name}_msa_size.csv"

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

    def get_uniprot_ids(self) -> set[str]:
        filenames = [f for f in self.msa_path.iterdir() if f.is_file()]
        return {f.stem for f in filenames if f.suffix == ".a3m"}

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
        fig, ax = plt.subplots()
        sns.scatterplot(data=self.get_msa_sizes(), x='sequence_count', y='query_length', ax=ax)
        ax.set(xlabel='Number of Sequences in MSA', ylabel='Length of Query')
        plt.savefig(fig_path)
        logging.info(f"saved figure to {fig_path}")


