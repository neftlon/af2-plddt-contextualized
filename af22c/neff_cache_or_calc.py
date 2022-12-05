#!/usr/bin/env python3

import sys
import json
import tarfile
import tempfile
import logging
import os.path
from dataclasses import dataclass
import signal
import time
from types import NoneType

from af22c.load_msa import calc_neff_by_id
from af22c.utils import get_raw_proteome_name, warn_once


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

    def get_raw_proteome_name(self) -> str:
        return get_raw_proteome_name(self.proteome_filename)

    def get_neffs_for_protein_path(self, uniprot_id: str) -> str:
        proteome_name = self.get_raw_proteome_name()
        return f"{proteome_name}/neffs/{uniprot_id}.json"

    def get_from_cache(self, uniprot_id: str) -> list[float]:
        """
        Get Neff scores for a protein from the cache file.

        If the cache file is not available or the cache file does not contain scores for the requested protein, `None`
        is returned.
        """
        # check whether the cache file has been specified
        if isinstance(self.cache_filename, NoneType):
            return None  # TODO: do we need a different return type for this?

        if not os.path.exists(self.cache_filename):
            logging.debug(f"cache file {self.cache_filename} does not exist")
            return None  # TODO: do we need a different return type for this?

        if self.cache_filename.endswith(".gz"):
            warn_once(
                f"cache file {self.cache_filename} is compressed. this may dramatically slow down processing. "
                f"consider extracting it to a .tar file."
            )

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
        # don't try to store anything in cache, if the cache file location has not been specified
        if isinstance(self.cache_filename, NoneType):
            warn_once(
                f"cannot store precomputed Neff scores for {uniprot_id}, cache file was not specified"
            )
            return

        # write scores to temp file
        with tempfile.TemporaryFile() as temp:
            enc = json.dumps(scores).encode()
            temp.write(enc)
            enc_size = temp.tell()  # obtain written size
            temp.seek(0)  # reset reading head of temp file

            # write scores from temp file to cache archive
            with tarfile.open(self.cache_filename, "a") as tar:
                neff_for_protein_path = self.get_neffs_for_protein_path(uniprot_id)
                info = tarfile.TarInfo(neff_for_protein_path)
                info.size = enc_size
                info.mtime = time.time()  # nicer than only having zeros in there...
                tar.addfile(info, temp)

    def get_neffs(self, uniprot_id: str) -> list[float]:
        """Get Neff scores for a protein."""
        # return cached scores if possible
        if scores := self.get_from_cache(uniprot_id):
            return scores

        # if Neffs are not cached, calculate them from proteome and store them in cache
        scores = calc_neff_by_id(self.proteome_filename, uniprot_id)
        self.store_in_cache(uniprot_id, scores)
        return scores


def main():
    keyboard_interrupt_caught = False
    is_in_write = False
    num_neff_files_written = 0

    def exit_message():
        logging.info(
            f"program is shutting down, wrote {num_neff_files_written}/{len(ids_to_process)} Neff files"
        )

    def keyboard_interrupt_handler(signum, frame):
        nonlocal keyboard_interrupt_caught
        if signum == signal.SIGINT:
            if is_in_write:
                logging.info(
                    f"received Ctrl-C, but program is currently writing to a file. finishing write first..."
                )
                keyboard_interrupt_caught = True
            else:
                logging.info(f"received Ctrl-C, aborting calculations...")
                exit_message()
                sys.exit(0)

    # setup signal handler
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    # setup cache handler
    neff_src = NeffCacheOrCalc(
        proteome_filename="data/UP000005640_9606.tar",
        cache_filename="data/UP000005640_9606_neff_cache.tar",
    )

    # calculate which IDs are available
    logging.debug("finding available protein IDs")
    with tarfile.open(neff_src.proteome_filename) as f:
        filenames = f.getnames()
        proteome_name = neff_src.get_raw_proteome_name()
        protein_msa_files = [
            fn
            for fn in filenames
            if fn.startswith(f"{proteome_name}/msas/") and fn.endswith(".a3m")
        ]
        avail_prot_ids = [
            os.path.splitext(os.path.basename(fn))[0] for fn in protein_msa_files
        ]
    logging.debug(f"found {len(avail_prot_ids)} proteins to look at")

    # calculate which IDs are already cached
    cached_prot_ids = []
    if os.path.exists(neff_src.cache_filename):
        logging.debug("finding protein IDs that already have cached Neff scores")
        with tarfile.open(neff_src.cache_filename) as f:
            filenames = f.getnames()
            proteome_name = neff_src.get_raw_proteome_name()
            protein_msa_files = [
                fn
                for fn in filenames
                if fn.startswith(f"{proteome_name}/neffs/") and fn.endswith(".json")
            ]
            cached_prot_ids = [
                os.path.splitext(os.path.basename(fn))[0] for fn in protein_msa_files
            ]
    else:
        logging.debug(
            "unable to find cache file, will create one when writing first list of Neff scores"
        )
    logging.debug(f"found {len(cached_prot_ids)} proteins that are already cached")

    # calculate which proteins need to be cached
    ids_to_process = set(avail_prot_ids) - set(cached_prot_ids)

    for uniprot_id in ids_to_process:
        # break early if we caught an interrupt
        if keyboard_interrupt_caught:
            break

        # the long calculation can be aborted
        logging.info(f"processing {uniprot_id}")
        logging.info(f" calculating scores...")
        scores = calc_neff_by_id(neff_src.proteome_filename, uniprot_id)

        # write outputs
        is_in_write = True
        logging.info(f" storing scores...")
        neff_src.store_in_cache(uniprot_id, scores)
        num_neff_files_written += 1
        logging.info(f" ---> done")
        is_in_write = False


if __name__ == "__main__":
    main()
