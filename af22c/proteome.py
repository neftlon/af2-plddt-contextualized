import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass, field
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Generator
import json
from abc import abstractmethod, ABC

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
        # TODO the behaviour of loading from msa_sizes and not from available MSAs could be confusing. Fix!
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

    def _load_neffs(self, path: Path):
        with path.open() as p:
            return json.load(p)

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
    def get_neff_by_id(self, uniprot_id: str) -> list:
        neff_path = self.neff_dir / f"{uniprot_id}.json"
        try:
            return self._load_neffs(neff_path)
        except FileNotFoundError:
            # TODO compute and store Neff if not found
            raise FileNotFoundError(f"Neff file for {uniprot_id} not found. Compute Neffs first!")

    def get_neff_naive_by_id(self, uniprot_id: str) -> list:
        neff_naive_path = self.neff_naive_dir / f"{uniprot_id}.json"
        try:
            return self._load_neffs(neff_naive_path)
        except FileNotFoundError:
            # TODO compute and store Neff naive if not found
            raise FileNotFoundError(f"Neff naive file for {uniprot_id} not found. Compute Neffs first!")

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
        p = sns.jointplot(data=msa_sizes, x='query_length', y='sequence_count')
        p.set_axis_labels('Number of Amino Acids in Query', 'Number of Sequences in MSA')
        p.savefig(fig_path)
        logging.info(f"saved figure to {fig_path}")

@dataclass
class ProteomeMetric(ABC):
    # metric_by_id: dict[str: list[float] | tuple[float, float]]
    @abstractmethod
    def get_uniprot_ids(self) -> set[str]:
        """
        Get all uniprot ids for which the metric is available without further computation.
        I.e. it can be loaded from disk.
        """
        ...

    @abstractmethod
    def __getitem__(self, item: str) -> list[float]:
        """
        Item is a uniprot id. The returned list of floats contains the metric for each AA in the specified protein.
        """
        ...


@dataclass
class ProteomePLDDTs(ProteomeMetric):
    """
    This class manages plDDTs for whole proteomes.

    The plDDTs are expected to be stored in a .json file in a dictionary by ID like:

    ```python
    plDDTs = {prot_id1: [...], prot_id2: [...]}
    ```
    """
    plddts_by_id: dict[str: list[float]]

    @classmethod
    def from_file(cls, plddt_path: str):
        path = Path(plddt_path)
        with path.open() as p:
            return cls(json.load(p))

    def get_uniprot_ids(self):
        return set(self.plddts_by_id.keys())

    def __getitem__(self, item):
        return self.plddts_by_id[item]


@dataclass
class ProteomeSETHPreds(ProteomeMetric):
    """
    This class manages SETH predictions for whole Proteomes.

    The SETH_preds are expected to be stored in a .json file in a dictionary by ID like:

    ```python
    SETH_preds = {prot_id1: [...], prot_id2: [...]}
    ```
    """
    seth_preds_by_id: dict[str: list[float]]

    @classmethod
    def from_file(cls, seth_preds_path: str):
        path = Path(seth_preds_path)
        with path.open() as p:
            lines = p.readlines()
            headers = lines[::2]
            disorders = lines[1::2]
            proteome_seth_preds = {}
            for header, disorder in zip(headers, disorders):
                uniprot_id = header.split("|")[1]
                disorder = list(map(float, disorder.split(", ")))
                proteome_seth_preds[uniprot_id] = disorder
            return cls(proteome_seth_preds)

    def get_uniprot_ids(self):
        return set(self.seth_preds_by_id.keys())

    def __getitem__(self, item):
        return self.seth_preds_by_id[item]


@dataclass
class ProteomeCorrelation:
    msas: Proteome
    plddts: ProteomePLDDTs
    seth_preds: ProteomeSETHPreds

    def _get_shared_ids(self) -> set[str]:
        neff_ids = self.msas.get_uniprot_ids(mode='neff_available')
        neff_naive_ids = self.msas.get_uniprot_ids(mode='neff_naive_available')
        plddt_ids = self.plddts.get_uniprot_ids()
        seth_pred_ids = self.seth_preds.get_uniprot_ids()

        shared_ids = neff_ids & neff_naive_ids & plddt_ids & seth_pred_ids
        not_shared_ids = (neff_ids | neff_naive_ids | plddt_ids | seth_pred_ids) - shared_ids
        logging.info(f"Disregarding {len(not_shared_ids)} proteins since they are not available in all datasets.")
        return shared_ids

    def _get_length_consistent_ids(self) -> set[str]:
        mismatched_len_ids = set()
        shared_ids = self._get_shared_ids()
        for prot_id in shared_ids:
            reference_len = len(self.msas.get_neff_by_id(prot_id))
            is_mismatch = (reference_len != len(self.plddts[prot_id])
                           or reference_len != len(self.seth_preds[prot_id])
                           or reference_len != len(self.msas.get_neff_naive_by_id(prot_id)))
            if is_mismatch:
                mismatched_len_ids.add(prot_id)
        logging.info(f"Disregarding {len(mismatched_len_ids)} proteins due to sequence length mismatches.")
        return shared_ids - mismatched_len_ids

    def get_uniprot_ids(self) -> set[str]:
        return self._get_length_consistent_ids()

    def _generate_observation_df(self, uniprot_id) -> pd.DataFrame:
        obs_dict = {'pLDDTs': self.plddts[uniprot_id],
                    'pred. dis.': self.seth_preds[uniprot_id],
                    'Neff': self.msas.get_neff_by_id(uniprot_id),
                    'Neff naive': self.msas.get_neff_naive_by_id(uniprot_id)}
        return pd.DataFrame(obs_dict)

    def get_pearson_corr(self, uniprot_id) -> pd.DataFrame:
        obs_df = self._generate_observation_df(uniprot_id)
        return obs_df.corr()

    def plot_mean_pearson_corr_mat(self, min_q_len=0, max_q_len=np.inf, min_n_seq=0, max_n_seq=np.inf):
        prot_ids = self.get_uniprot_ids()
        prot_ids = prot_ids & self.msas.get_uniprot_ids_in_size(min_q_len=min_q_len,
                                                                max_q_len=max_q_len,
                                                                min_n_seq=min_n_seq,
                                                                max_n_seq=max_n_seq)
        df_index = None
        p_corr_list = []
        for prot_id in prot_ids:
            p_corr = self.get_pearson_corr(prot_id)
            if not df_index:
                df_index = (p_corr.index, p_corr.columns)
            p_corr_list.append(p_corr.to_numpy())
        p_corr_array = np.stack(p_corr_list)
        # TODO Fix NaN values when including small msas, problem probably in computation
        # TODO return the following DataFrame and do plotting somewhere else
        p_corr_mean = pd.DataFrame(np.mean(p_corr_array, axis=0), index=df_index[0], columns=df_index[1])

        fig_path = self.msas.data_dir / f'{self.msas.name}_mean_pearson_corr.png'
        sns.set_style('whitegrid')
        mask = np.triu(np.ones_like(p_corr_mean, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        p = sns.heatmap(p_corr_mean, annot=True, mask=mask, cmap=cmap, vmin=-1, vmax=1)
        p.get_figure().savefig(fig_path)
        logging.info(f"saved figure to {fig_path}")
