# AlphaFold2 pLDDT scores contextualized

This project aims at understanding and quantifying the relationship betweeen pLDDTs, MSAs and embedding-based disorder
predictions.

Therefore we developed a Python library and accompanying scripts to calculate Neff (number of effective sequences) and gap count scores.

The commands described in the following subchapters are meant to be executed in a shell in the project root directory.

## Installation

The project uses Python for scripts. The root directory contains a `environment.yml` file that can be used to create an
appropriate environment.

It is recommended to use [conda](https://docs.conda.io/en/latest/), you can create a new requirement with all dependencies installed like the following:

```bash
conda create --file=environment.yml
conda activate env
```

Note that our project requires Python 3.10.

Install the `af22c` package in development mode with the following command in the project root directory:
```bash
pip install -e .
```

## Usage

Our code can be used either as a Python library or as standalone programs. In the following, we present common usecases for both methods.

### Library

You can use our library to calculate Neff or gapcount scores from an MSA in the following way. Specify a torch device in the functions to use GPU acceleration.

```python
>>> from af22c import neff, gapcount
>>> msa = ["SEQWENCE","--QWENCE","--QWEN--","--QEEN--"]; print("\n".join(msa))
SEQWENCE
--QWENCE
--QWEN--
--QEEN--
>>> # calculate Neff and gapcount scores for the MSA
>>> neff(msa, device="cuda")
tensor([1., 1., 3., 3., 3., 3., 2., 2.])
>>> gapcount(msa, device="cuda")
tensor([3, 3, 0, 0, 0, 0, 2, 2])
```

In our case, we have MSAs for the entire human proteome. They are inside a .tar file. With our library, you can extract .a3m MSA files from a .tar file and use it as an input for score calculation. In the following, we show how you can calculate an protein-specific average Neff score for [APP](https://en.wikipedia.org/wiki/Amyloid-beta_precursor_protein).

```python
>>> from af22c import neff
>>> from af22c.utils import ProteomeTar
>>> human = ProteomeTar("data/UP000005640_9606.tar")
>>> app = human.get_protein_location("Q92624")
>>> neff(app, device="cuda").mean()
tensor(209.4100, device='cuda:0')
```

_Hint:_ To get output while running, you can specify `verbose=True`. For debugging CUDA out-of-memory issues when running `neff`, you can lower the `batch_size` (default: `2**12`) parameter used for pairwise sequence identity calculation.

_TODO: say that we also have ref impl available_

### Scripts

We also provide scripts to extract Neff scores on the command line.

```bash
./scripts/neff_gpu.py ./data/A0A0A0MRZ7.a3m /tmp/neff.json
cat /tmp/neff.json # only truncated output shown:
[1564.3165283203125, 1660.9517822265625, 1906.4017333984375, 1983.4613037109375, 2283.94921875, 2345.59912109375, 2614.16015625, 2852.3935546875, 2907.48046875, 3112.87890625, 3278.29541015625, 3376.743896484375, 3506.353759765625, ...]
```

_TODO: say that we also have support for .tar files containing entire proteomes._

## pLDDT

Our project deals with the comparison between Neff, SETH predictions, and pLDDT scores. Therefore, we need utilities to obtain and handle these pLDDT scores. We include our utility programs in this repository. In the following, we will show how one can gather pLDDT scores from AlphaFold 2's database.

For using the download script, the `wget` utility must be installed on the system.

### 1. Data download

The application relies on data from the [AlphaFold 2 database](https://alphafold.ebi.ac.uk/) ([[1]](#1) and [[2]](#2))
and precomputed disorder predictions from SETH [[3]](#3). The `download_human.sh` script downloads both datasets into
the `./data` subdirectory. (If not yet present, the directory is created for you.)

```shell
./af22c/download_human.sh
```

Before downloading, the script checks whether the data has changed on the server. To force a download, delete the
corresponding file in the `./data` subdirectory.

### 2. pLDDT score extraction

The pLDDT scores are stored [alongside the 3D residue position](https://alphafold.ebi.ac.uk/faq#faq-5) inside the PDB
files predicted by AlphaFold 2. The `extract_plddts.py` utility extracts and stores them in a `..._plddts.json` file
inside the `./data` subdirectory.

```shell
./scripts/extract_plddts.py
```

Note, that the program extracts the `.tar` file downloaded from the AlphaFold 2 database inside a temporary directory.
Since every `.pdb.gz` file from the archive is copied, this temporarily requires as much addition disk memory as the
initial download. This happens to ease the use of multiple processes using a Python `ProcessPoolExecutor`.

### 3. Filter pLDDTs for fragmented proteins 

If proteins have more than 2700 residues, they are predicted in [fragments by AF2](https://alphafold.ebi.ac.uk/faq).
This also means, the pLDDTs are stored in multiple files, one for each fragment. The `filter_protfrags.py` utility 
filters the pLDDTs to only contain non-fragmented proteins and stores the filtered pLDDTs in a `..._plddts_fltrd.json`
file inside the `./data` subdirectory.

```shell
./scripts/filter_protfrags.py
```

## Miscellaneous

The `environment.yml` file can be created using this command:

```shell
conda env export | head -n -1 > environment.yml
```

The call to `head` removes the platform-specific `prefix` line generated by conda's export command.

## References

<a id="1">[1]</a>
Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583–589 (2021). https://doi.org/10.1038/s41586-021-03819-2

<a id="2">[2]</a>
Mihaly Varadi, Stephen Anyango, Mandar Deshpande, Sreenath Nair, Cindy Natassia, Galabina Yordanova, David Yuan, Oana Stroe, Gemma Wood, Agata Laydon, Augustin Žídek, Tim Green, Kathryn Tunyasuvunakool, Stig Petersen, John Jumper, Ellen Clancy, Richard Green, Ankur Vora, Mira Lutfi, Michael Figurnov, Andrew Cowie, Nicole Hobbs, Pushmeet Kohli, Gerard Kleywegt, Ewan Birney, Demis Hassabis, Sameer Velankar, AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models, Nucleic Acids Research, Volume 50, Issue D1, 7 January 2022, Pages D439–D444, https://doi.org/10.1093/nar/gkab1061

<a id="3">[3]</a>
Ilzhoefer, D., Heinzinger, M. & Rost, B. 2022. SETH predicts nuances of residue disorder from protein embeddings, https://doi.org/10.1101/2022.06.23.497276 (Preprint)
