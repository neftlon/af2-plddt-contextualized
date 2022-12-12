import io
import os
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass, field
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Generator, Any, Callable
import json
from abc import abstractmethod, ABC

from af22c.load_msa import MultipleSeqAlign
from af22c.score_max_z import calc_max_z
from af22c.utils import get_raw_proteome_name


class Proteome(ABC):
    @abstractmethod
    def get_uniprot_ids(self) -> set[str]:
        """
        Get all uniprot ids for which the metric is available without further computation.
        I.e. it can be loaded from disk.
        """
        ...


class ProteomewidePerProteinMetric(Proteome):
    @abstractmethod
    def __getitem__(self, uniprot_id: str) -> Any:
        ...


@dataclass
class ProteomewidePerResidueMetric(Proteome):
    metric_name: str = field(init=False)

    def __post_init__(self):
        self.init_metric_name()

    @abstractmethod
    def init_metric_name(self):
        ...

    @abstractmethod
    def __getitem__(self, uniprot_id: str) -> list[float | int]:
        """Return a list of scores (per-residue/AA) for a given UniProt identifier."""
        ...


@dataclass
class ProteomeMSAs(Proteome):
    """
    This class manages MSAs for whole Proteomes.

    The proteome is expected to be a folder with the following layout:

    ```python
    msas_for_protein_path = f"{proteome_name}/msas/{uniprot_id}.a3m"
    ```
    """

    class MSAProvider(ABC):
        @abstractmethod
        def get_by_id(self, uniprot_id: str) -> MultipleSeqAlign:
            ...

        @abstractmethod
        def get_msas(self) -> Generator[MultipleSeqAlign, None, None]:
            ...

        @abstractmethod
        def get_uniprot_ids(self) -> set[str]:
            ...

        @abstractmethod
        def get_name(self) -> str:
            ...

    @dataclass
    class DirectoryMSAProvider(MSAProvider):
        name: str  # proteome name
        msa_path: Path  # MSA path (directory containing MSAs)

        def get_by_id(self, uniprot_id: str) -> MultipleSeqAlign:
            return MultipleSeqAlign.from_a3m(self.msa_path / f"{uniprot_id}.a3m")

        def get_msas(self) -> Generator[MultipleSeqAlign, None, None]:
            for uniprot_id in tqdm(self.get_uniprot_ids()):
                yield self.get_by_id(uniprot_id)

        def get_uniprot_ids(self) -> set[str]:
            # Search path and extract IDs from filenames
            filenames = [f for f in self.msa_path.iterdir() if f.is_file()]
            return {f.stem for f in filenames if f.suffix == ".a3m"}

        def get_name(self) -> str:
            return self.name

    @dataclass
    class ArchiveMSAProvider(MSAProvider):
        name: str  # proteome name
        msa_path: Path  # MSA path (path to archive containing MSAs)
        a3m_subdir: Path = field(init=False)

        def __post_init__(self):
            # TODO: not all archive files have the same layout, make this parameterizable
            self.a3m_subdir = Path(self.name) / "msas"

        def _extract_msa(self, tar: tarfile.TarFile, uniprot_id: str):
            a3m_filename = os.path.join(self.a3m_subdir, f"{uniprot_id}.a3m")
            a3m = tar.extractfile(a3m_filename)
            with io.TextIOWrapper(a3m, encoding="utf-8") as a3m:
                return MultipleSeqAlign.from_a3m(a3m)

        def get_by_id(self, uniprot_id: str) -> MultipleSeqAlign:
            with tarfile.open(self.msa_path) as tar:
                return self._extract_msa(tar, uniprot_id)

        def get_msas(self) -> Generator[MultipleSeqAlign, None, None]:
            uniprot_ids = self.get_uniprot_ids()
            with tarfile.open(self.msa_path) as tar:
                for uniprot_id in uniprot_ids:
                    yield self._extract_msa(tar, uniprot_id)

        def get_uniprot_ids(self) -> set[str]:
            a3m_subdir = str(
                self.a3m_subdir
            )  # convert to string only once and not multiple times
            with tarfile.open(self.msa_path) as tar:
                filenames = tar.getnames()
                # all files must be within a specified subdirectory of the archive, and they need to be .a3m files
                filenames = [
                    filename
                    for filename in filenames
                    if filename.startswith(a3m_subdir) and filename.endswith(".a3m")
                ]
                # use filename without path and extension as UniProt ID
                return {
                    os.path.basename(filename).split(".")[0] for filename in filenames
                }

        def get_name(self) -> str:
            return self.name

    msa_provider: MSAProvider

    @classmethod
    def from_directory(cls, msas_path: str):
        msas_path = Path(msas_path)
        return cls(cls.DirectoryMSAProvider(msas_path.name, msas_path))

    @classmethod
    def from_archive(cls, proteome_path: str):
        if proteome_path.endswith(".tar.gz"):
            logging.warning(
                f"Trying to create an `ProteomeMSA` file from a compressed file {proteome_path}. This is possible, but "
                f"extremely slow and therefore not recommended. We suggest unpacking the file, for instance to a .tar "
                f"archive or to a directory (then, `ProteomeMSA.from_directory` must be used to create this object)."
            )
        proteome_name = get_raw_proteome_name(proteome_path)
        proteome_path = Path(proteome_path)
        return cls(cls.ArchiveMSAProvider(proteome_name, proteome_path))

    @classmethod
    def from_file(cls, path: str):
        """
        Create a `ProteomeMSAs` object without explicitly specifying a filetype.

        This method decides whether it should call `ProteomeMSAs.from_directory` or `ProteomeMSAs.from_archive`.
        """
        if path.endswith(".tar") or path.endswith(".tar.gz"):
            return cls.from_archive(path)
        return cls.from_directory(path)

    def __getitem__(self, uniprot_id: str) -> MultipleSeqAlign:
        return self.msa_provider.get_by_id(uniprot_id)

    def get_msas(self) -> Generator[MultipleSeqAlign, None, None]:
        return self.msa_provider.get_msas()

    def get_uniprot_ids(self) -> set[str]:
        return self.msa_provider.get_uniprot_ids()

    def get_name(self) -> str:
        return self.msa_provider.get_name()


@dataclass
class ProteomeMSASizes(ProteomewidePerProteinMetric):
    class CSVMSASizeProvider:
        def __init__(self, path: Path):
            self.msa_sizes = None  # initialized by call to self.load
            self.filepath = path
            self.load()

        def load(self):
            if not self.filepath.is_file():
                raise FileNotFoundError(
                    f"CSV file {self.filepath} is not available, please precompute or compute on demand "
                    f"(using `ProteomeMSASizes.from_msas`)"
                )
            self.msa_sizes = pd.read_csv(self.filepath)

            # check that the CSV contains the right fields
            required_fields = {"uniprot_id", "query_length", "sequence_count"}
            actual_fields = set(self.msa_sizes.columns)
            if not required_fields <= actual_fields:
                raise ValueError(
                    f"CSV file {self.filepath} does not contain the right column names to build a `ProteomeMSASizes` "
                    f"object.\n"
                    f"Expected column names:       {required_fields}\n"
                    f"Actual column names in file: {actual_fields}"
                )

        def __getitem__(self, uniprot_id) -> tuple[int, int]:
            size = self.msa_sizes[self.msa_sizes["uniprot_id"] == uniprot_id]
            n_seq, q_len = size["sequence_count"], size["query_length"]
            if not isinstance(n_seq, int) or not isinstance(q_len, int):
                raise ValueError(
                    f"The type of the dimensions is ({type(n_seq)}, {type(q_len)} and not (int, int). "
                    f"Maybe the uniprot_id {uniprot_id} appears multiple times in the "
                    f"msa sizes .csv file?"
                )
            return n_seq, q_len

        def get_uniprot_ids(self) -> set[str]:
            return set(self.msa_sizes["uniprot_id"])

        def get_uniprot_ids_in_size(
            self, min_q_len=0, max_q_len=np.inf, min_n_seq=0, max_n_seq=np.inf
        ) -> set[str]:
            in_size = self.msa_sizes[
                (self.msa_sizes["query_length"] <= max_q_len)
                & (self.msa_sizes["sequence_count"] <= max_n_seq)
                & (self.msa_sizes["query_length"] >= min_q_len)
                & (self.msa_sizes["sequence_count"] >= min_n_seq)
            ]
            return set(in_size["uniprot_id"])

        def get_msa_sizes(self) -> pd.DataFrame:
            return self.msa_sizes

        def precompute_msa_sizes(self):
            raise Exception(
                "Cannot compute MSA sizes, no MSAs available. "
                "Please provide MSAs on initialization by using `ProteomeMSASizes.from_msas`."
            )

    class ComputingMSASizeProvider(CSVMSASizeProvider):
        """Compute MSA sizes from a proteome on demand."""

        def __init__(
            self,
            proteome_msas: ProteomeMSAs,
            cache_file_path: Path,
            write_csv_on_demand: bool,
        ):
            # NOTE: this field need to be set BEFORE initializing the superclass because the overwritten `load` method
            # of THIS subclass uses the attribute. otherwise, the `variable` will be referenced before assignment and
            # everything goes down the drain.
            self.write_csv_on_demand = write_csv_on_demand

            super().__init__(cache_file_path)
            self.proteome_msas = proteome_msas

        def load(self):
            try:
                super().load()
            except FileNotFoundError:
                if self.write_csv_on_demand:
                    # CSV file was not found, create a new one
                    self._store_msa_sizes([])  # write at least the header
                    super().load()
                else:
                    raise  # we cannot help -- sorry, let the caller handle the exception

        def _store_msa_sizes(self, msa_sizes: list[list]):
            if not self.write_csv_on_demand:
                raise IOError(
                    f"Writing CSV file {self.filepath} is not allowed, write_csv_on_demand flag was set to False!"
                )
            size_df = pd.DataFrame(
                msa_sizes, columns=["uniprot_id", "query_length", "sequence_count"]
            )
            size_df.to_csv(
                self.filepath,
                mode="a",
                header=not self.filepath.is_file(),
                index=False,
            )

        def compute_msa_size(self, uniprot_id: str) -> tuple[int, int]:
            msa = self.proteome_msas[uniprot_id]
            return msa.get_size()

        def __getitem__(self, uniprot_id: str) -> tuple[int, int]:
            try:
                return super().__getitem__(uniprot_id)
            except ValueError:
                q_len, n_seq = self.compute_msa_size(uniprot_id)
                if self.write_csv_on_demand:
                    self._store_msa_sizes([[uniprot_id, q_len, n_seq]])
                    super().load()  # reload CSV for future calls to __getitem__
                else:
                    logging.warning(
                        f"Computed MSA size for {uniprot_id}, but did not store changes in {self.filepath} "
                        f"since the write_csv_on_demand flag was set to False"
                    )
                return q_len, n_seq

        def precompute_msa_sizes(self):
            logging.info("Deleting cache file")
            os.remove(self.filepath)

            logging.info(f"Computing MSA sizes ...")
            msa_sizes = []
            for i, msa in enumerate(self.proteome_msas.get_msas()):
                q_len, n_seq = msa.get_size()
                q_id = msa.query_id
                msa_sizes.append([q_id, q_len, n_seq])

                # Write to file every 100 iterations
                if i % 100 == 0:
                    self._store_msa_sizes(msa_sizes)
                    msa_sizes = []
            if msa_sizes:
                self._store_msa_sizes(msa_sizes)
            logging.info(f"Wrote MSA sizes to {self.filepath}")

            super().load()

    msa_size_provider: CSVMSASizeProvider

    @classmethod
    def from_file(cls, path: str):
        return cls(cls.CSVMSASizeProvider(Path(path)))

    @classmethod
    def from_msas(
        cls, proteome_msas: ProteomeMSAs, data_dir="data", write_csv_on_demand=True
    ):
        # TODO: figure out whether we want to pass the data_dir here as a parameter
        proteome_name = proteome_msas.get_name()
        data_dir = Path(data_dir)
        cache_file_path = data_dir / f"{proteome_name}_msa_size.csv"
        return cls(
            cls.ComputingMSASizeProvider(
                proteome_msas,
                cache_file_path,
                write_csv_on_demand=write_csv_on_demand,
            )
        )

    def __getitem__(self, uniprot_id: str) -> tuple[int, int]:
        """Return a tuple containing `(n_sequences, len_query)` for a given UniProt identifier."""
        return self.msa_size_provider[uniprot_id]

    def get_uniprot_ids(self) -> set[str]:
        return self.msa_size_provider.get_uniprot_ids()

    def get_uniprot_ids_in_size(
        self, min_q_len=0, max_q_len=np.inf, min_n_seq=0, max_n_seq=np.inf
    ) -> set[str]:
        # TODO(johannes): can we also just delegate using **kwargs (problem: method parameters are not written here)
        return self.msa_size_provider.get_uniprot_ids_in_size(
            min_q_len, max_q_len, min_n_seq, max_n_seq
        )

    def get_msa_sizes(self) -> pd.DataFrame:
        return self.msa_size_provider.get_msa_sizes()

    def precompute_msa_sizes(self):
        """
        Precompute MSA sizes and store them in the referenced .csv file.
        """
        self.msa_size_provider.precompute_msa_sizes()

    def plot_msa_sizes(self, data_dir="data", name="human", uniprot_ids: set[str] = None):
        data_dir = Path(data_dir)
        fig_path = data_dir / f"{name}_msa_size_scatter.png"
        msa_sizes = self.get_msa_sizes()
        if uniprot_ids:
            # TODO raise exception if not all uniprot IDs are in msa_sizes as this can lead to unexpected behaviour
            msa_sizes = msa_sizes.loc[msa_sizes['uniprot_id'].isin(uniprot_ids)]
        sns.set_style("whitegrid")
        p = sns.jointplot(data=msa_sizes, x="query_length", y="sequence_count")
        p.set_axis_labels(
            "Number of Amino Acids in Query", "Number of Sequences in MSA"
        )
        p.savefig(fig_path)
        logging.info(f"saved figure to {fig_path}")


@dataclass
class ProteomeScores(ProteomewidePerResidueMetric):
    # TODO: add precompute_scores method

    @dataclass
    class ScoresFromDirProvider:
        scores_dir: Path

        def get_uniprot_ids(self) -> set[str]:
            # Search path and extract IDs from filenames
            filenames = [f for f in self.scores_dir.iterdir() if f.is_file()]
            return {f.stem for f in filenames if f.suffix == ".json"}

        def __getitem__(self, uniprot_id) -> list[float | int]:
            cache_path = self.scores_dir / f"{uniprot_id}.json"
            try:
                return self._load_scores(cache_path)
            except FileNotFoundError:
                # TODO compute and store scores if not found
                raise FileNotFoundError(
                    f"Score file for {uniprot_id} not found. Compute scores first!"
                )

        @staticmethod
        def _load_scores(path: Path):
            with path.open() as p:
                return json.load(p)

        def compute_scores_by_id(self, uniprot_id: str, overwrite_scores=False):
            raise Exception(
                "Cannot compute MSA scores, no MSAs available. "
                "Please provide MSAs on initialization by using the appropriate "
                "`Proteome[SCORE NAME].from_msas`."
            )

    @dataclass
    class ScoresFromProteomeProvider(ScoresFromDirProvider):
        proteome_msas: ProteomeMSAs
        write_scores_on_demand: bool
        compute_scores_fn: Callable

        def __getitem__(self, uniprot_id: str) -> list[float | int]:
            try:
                return super()[uniprot_id]
            except FileNotFoundError:
                self.compute_scores_by_id(uniprot_id)
                return super()[uniprot_id]

        def compute_scores_by_id(self, uniprot_id: str, overwrite_scores=False):
            if not self.write_scores_on_demand:
                raise IOError(
                    f"Writing scores to {self.scores_dir} is not allowed, write_scores_on_demand flag was set to False!"
                )

            self.scores_dir.mkdir(parents=True, exist_ok=True)
            scores_path = self.scores_dir / f"{uniprot_id}.json"
            if scores_path.is_file() and not overwrite_scores:
                logging.info(
                    f"Scores for {uniprot_id} are already cached, skipped computation"
                )
            else:
                logging.info(f"Computing scores for {uniprot_id} ...")
                msa = self.proteome_msas[uniprot_id]
                scores = self.compute_scores_fn(msa)
                with scores_path.open(mode="w+") as p:
                    json.dump(scores, p)

    score_provider: ScoresFromDirProvider

    @classmethod
    def from_directory(cls, path: str):
        return cls(cls.ScoresFromDirProvider(Path(path)))

    @classmethod
    def from_msas(
        cls, proteome_msas: ProteomeMSAs, scores_dir: str, write_scores_on_demand=True
    ):
        scores_dir = Path(scores_dir)
        return cls(
            cls.ScoresFromProteomeProvider(
                scores_dir, proteome_msas, write_scores_on_demand, cls.compute_scores
            )
        )

    def get_uniprot_ids(self) -> set[str]:
        return self.score_provider.get_uniprot_ids()

    def __getitem__(self, uniprot_id):
        return self.score_provider[uniprot_id]

    def compute_scores_by_id(self, uniprot_id: str, overwrite_scores=False):
        self.score_provider.compute_scores_by_id(uniprot_id, overwrite_scores)

    @staticmethod
    @abstractmethod
    def compute_scores(msa: MultipleSeqAlign) -> list[float | int]:
        ...


class ProteomeNeffs(ProteomeScores):
    def init_metric_name(self):
        self.metric_name = "Neff"

    @staticmethod
    def compute_scores(msa: MultipleSeqAlign) -> list[int]:
        # NOTE: since Neff scores usually are in the area of 1k to 10k, rounding to `int` here should be sufficient
        neffs = msa.compute_neff()
        return list(map(lambda f: round(f), neffs.tolist()))


class ProteomeNeffsNaive(ProteomeScores):
    def init_metric_name(self):
        self.metric_name = "Neff naive"

    @staticmethod
    def compute_scores(msa: MultipleSeqAlign) -> list[int]:
        # NOTE: since Neff scores usually are in the area of 1k to 10k, rounding to `int` here should be sufficient
        neffs_naive = msa.compute_neff_naive()
        return list(map(lambda f: round(f), neffs_naive.tolist()))


class ProteomeMaxZs(ProteomeScores):
    def init_metric_name(self):
        self.metric_name = "maxZ"

    @staticmethod
    def compute_scores(msa: MultipleSeqAlign) -> list[float]:
        # TODO: take care of the parameters to this function
        return list(map(lambda f: round(f, 2), calc_max_z(msa)))


@dataclass
class ProteomePLDDTs(ProteomewidePerResidueMetric):
    """
    This class manages plDDTs for whole proteomes.

    The plDDTs are expected to be stored in a .json file in a dictionary by ID like:

    ```python
    plDDTs = {prot_id1: [...], prot_id2: [...]}
    ```
    """

    plddts_by_id: dict[str : list[float]]

    @classmethod
    def from_file(cls, plddt_path: str):
        path = Path(plddt_path)
        with path.open() as p:
            return cls(json.load(p))

    def init_metric_name(self):
        self.metric_name = "pLDDT"

    def get_uniprot_ids(self):
        return set(self.plddts_by_id.keys())

    def __getitem__(self, uniprot_id):
        return self.plddts_by_id[uniprot_id]


@dataclass
class ProteomeSETHPreds(ProteomewidePerResidueMetric):
    """
    This class manages SETH predictions for whole Proteomes.

    The SETH_preds are expected to be stored in a .json file in a dictionary by ID like:

    ```python
    SETH_preds = {prot_id1: [...], prot_id2: [...]}
    ```
    """

    seth_preds_by_id: dict[str : list[float]]

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

    def init_metric_name(self):
        self.metric_name = "pred. dis."

    def get_uniprot_ids(self):
        return set(self.seth_preds_by_id.keys())

    def __getitem__(self, uniprot_id):
        return self.seth_preds_by_id[uniprot_id]


@dataclass
class ProteomeCorrelation:
    scores: list[ProteomewidePerResidueMetric]
    msa_sizes: ProteomeMSASizes

    def _get_shared_ids(self) -> set[str]:
        score_ids = [score.get_uniprot_ids() for score in self.scores]

        all_ids = set().union(*score_ids)
        shared_ids = all_ids.intersection(*score_ids)

        if self.msa_sizes:
            ids = self.msa_sizes.get_uniprot_ids()
            all_ids |= ids
            shared_ids &= ids

        not_shared_ids = all_ids - shared_ids

        logging.info(
            f"Disregarding {len(not_shared_ids)} proteins since they are not available in all datasets."
        )
        return shared_ids

    def _get_length_consistent_ids(self) -> set[str]:
        mismatched_len_ids = set()
        shared_ids = self._get_shared_ids()
        for prot_id in shared_ids:
            lens = [len(score[prot_id]) for score in self.scores]
            is_mismatch = len(set(lens)) > 1
            if is_mismatch:
                mismatched_len_ids.add(prot_id)
        logging.info(
            f"Disregarding {len(mismatched_len_ids)} proteins due to sequence length mismatches."
        )
        return shared_ids - mismatched_len_ids

    def get_uniprot_ids(self) -> set[str]:
        return self._get_length_consistent_ids()

    def _generate_observation_df(self, uniprot_id) -> pd.DataFrame:
        obs_dict = {score.metric_name: score[uniprot_id] for score in self.scores}
        return pd.DataFrame(obs_dict)

    def get_pearson_corr(self, uniprot_id) -> pd.DataFrame:
        obs_df = self._generate_observation_df(uniprot_id)
        return obs_df.corr()

    def plot_mean_pearson_corr_mat(
        self,
        data_dir,
        name,
        min_q_len=0,
        max_q_len=np.inf,
        min_n_seq=0,
        max_n_seq=np.inf,
    ):
        # compute protein IDs
        prot_ids = self.get_uniprot_ids()

        if (
            min_q_len != 0
            or max_q_len != np.inf
            or min_n_seq != 0
            or max_n_seq != np.inf
        ):
            # we can only limit the range of protein sizes, if there is an MSASizes object present
            # in `scores`
            if self.msa_sizes:
                # we found a `ProteomeMSASizes` score, use it to only retain protein IDs that
                # lay within the given ranges.
                prot_ids &= self.msa_sizes.get_uniprot_ids_in_size(
                    min_q_len=min_q_len,
                    max_q_len=max_q_len,
                    min_n_seq=min_n_seq,
                    max_n_seq=max_n_seq,
                )
            else:
                logging.warning(
                    "According to the parameters, this call to `plot_mean_pearson_corr_mat` "
                    "should only include proteins of given sizes into analysis. "
                    "However, this requires at least one `ProteomeMSASizes` score to be "
                    "present in the scores list. Since there is none, the list of proteins "
                    "will NOT be shrunken by the given range constraints. "
                    "Consider adding a `ProteomeMSASizes` scores to the list of scores."
                )

        # compute pairwise correlation
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
        p_corr_mean = pd.DataFrame(
            np.mean(p_corr_array, axis=0), index=df_index[0], columns=df_index[1]
        )

        # draw figure
        fig_path = Path(data_dir) / f"{name}_mean_pearson_corr.png"
        sns.set_style("whitegrid")
        mask = np.triu(np.ones_like(p_corr_mean, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        p = sns.heatmap(p_corr_mean, annot=True, mask=mask, cmap=cmap, vmin=-1, vmax=1)
        p.get_figure().savefig(fig_path)
        logging.info(f"saved figure to {fig_path}")

        # collect all per residue scores over the given subset of the proteome
        obs_dict = {score.metric_name: [] for score in self.scores}
        for prot_id in prot_ids:
            for score in self.scores:
                obs_dict[score.metric_name] += score[prot_id]
        obs_df = pd.DataFrame(obs_dict)

        # draw hist plot
        fig_path = Path(data_dir) / f"{name}_all_proteins_hist.png"
        p = sns.pairplot(obs_df, corner=True, kind='hist')
        p.savefig(fig_path)
        logging.info(f"saved figure to {fig_path}.")

        # draw correlation boxplots
        n = len(self.scores)
        fig, axs = plt.subplots(n, n)
        for i in range(1, n):
            for j in range(0, i):
                sns.boxplot(p_corr_array[:, i, j].flatten(), ax=axs[i, j])
        fig_path = Path(data_dir) / f"{name}_pearson_corr_boxplot.png"
        fig.savefig(fig_path)
        logging.info(f"saved figure to {fig_path}.")

        # draw correlation histograms
        n = len(self.scores)
        fig, axs = plt.subplots(n, n)
        for i in range(1, n):
            for j in range(0, i):
                sns.histplot(p_corr_array[:, i, j].flatten(), ax=axs[i, j])
        fig_path = Path(data_dir) / f"{name}_pearson_corr_hist.png"
        fig.savefig(fig_path)
        logging.info(f"saved figure to {fig_path}.")
