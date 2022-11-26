from dataclasses import dataclass
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
    # TODO data_path for storing results

    @classmethod
    def from_folder(cls, path: str, name='Proteome'):
        proteome_path = Path(path)
        return cls(name, proteome_path, proteome_path / 'msas')

    def get_msa_by_id(self, uniprot_id: str) -> MultipleSeqAlign:
        return MultipleSeqAlign.from_a3m(self.msa_path / f'{uniprot_id}.a3m')

    def get_msas(self) -> Generator[MultipleSeqAlign, None, None]:
        logging.info(f"iterating over MSAs ...")
        for uniprot_id in tqdm(self.get_uniprot_ids()):
            yield self.get_msa_by_id(uniprot_id)

    def get_uniprot_ids(self) -> list[str]:
        filenames = [f for f in self.msa_path.iterdir() if f.is_file()]
        return [f.stem for f in filenames if f.suffix == ".a3m"]

    def get_msa_sizes(self) -> list[tuple[int, int]]:
        return [msa.get_size() for msa in self.get_msas()]
