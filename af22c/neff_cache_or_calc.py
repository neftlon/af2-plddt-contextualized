#!/usr/bin/env python3

import json
import tarfile
import tempfile
import logging
import os.path
from dataclasses import dataclass
import signal

from af22c.load_msa import warn_once, calc_neff_by_id


@dataclass
class NeffCacheOrCalc:
    """
    This class manages Neff scores without having the user worry about whether they have already been calculated

    Proteome file
    -------------

    The proteome file is expected to be a .tar or .tar.gz file with the following layout:

    ```python
    msas_for_protein_path = f"{proteome_name}/msas/{uniprot_id}.a3m"
    ```

    Cache file
    ----------

    The cache file is expected to have a similar layout. It is expected to contain .json files containing an array of
    Neff scores for each protein.

    ```python
    neffs_for_protein_path = f"{proteome_name}/neffs/{uniprot_id}.json"
    ```
    """
    proteome_filename: str
    cache_filename: str

    def get_neffs_for_protein_path(self, uniprot_id: str) -> str:
        proteome_name, _ = os.path.splitext(os.path.basename(self.proteome_filename))
        return f"{proteome_name}/neffs/{uniprot_id}.json"

    def get_from_cache(self, uniprot_id: str) -> list[float]:
        """
        Get Neff scores for a protein from the cache file.

        If the cache file is not available or the cache file does not contain scores for the requested protein, `None`
        is returned.
        """
        if not os.path.exists(self.cache_filename):
            logging.debug(f"cache file {self.cache_filename} does not exist")
            return None  # TODO: do we need a different return type for this?

        if self.cache_filename.endswith(".gz"):
            warn_once(f"cache file {self.cache_filename} is compressed. this may dramatically slow down processing. "
                      f"consider extracting it to a .tar file.")

        neffs_for_protein_path = self.get_neffs_for_protein_path(uniprot_id)
        with tarfile.open(self.cache_filename) as tar:
            names = tar.getnames()

            # check if protein has been cached
            if neffs_for_protein_path not in names:
                return None

            # load scores from cache file if available
            with tar.extractfile(neffs_for_protein_path) as neffs_json:
                return json.load(neffs_json)

    def store_in_cache(self, uniprot_id: str, scores: list[float]):
        """Store Neff scores for a protein in cache file"""
        # write scores to temp file
        with tempfile.TemporaryFile() as temp:
            enc = json.dumps(scores).encode()
            temp.write(enc)
            enc_size = temp.tell()  # obtain written size
            temp.seek(0)  # reset reading head of temp file

            # write scores from temp file to cache archive
            with tarfile.open(self.cache_filename, "w") as tar:
                neff_for_protein_path = self.get_neffs_for_protein_path(uniprot_id)
                info = tarfile.TarInfo(neff_for_protein_path)
                info.size = enc_size
                tar.addfile(info, temp)

    def get_neffs(self, uniprot_id: str) -> list[float]:
        """Get Neff scores for a protein."""
        if scores := self.get_from_cache(uniprot_id):
            return scores

        scores = calc_neff_by_id(self.proteome_filename, uniprot_id)
        self.store_in_cache(uniprot_id, scores)
        return scores


if __name__ == "__main__":
    neff_src = NeffCacheOrCalc(
        proteome_filename="data/UP000005640_9606.tar",
        cache_filename="data/UP000005640_9696_neff_cache.tar",
    )
    scores = neff_src.get_neffs("A0A0A0MRZ9")
    print(scores)

    def keyboard_interrupt_handler(signum, frame):
        if signum == signal.SIGINT:
            print("keyboard interrupt caught!")
            sys.exit(0)
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)
