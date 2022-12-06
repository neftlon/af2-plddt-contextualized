# Tools

This directory contains some scripts. The following table describes what they do.

| Filename                                                                         | Description                                                                                                          |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| [`find_files_with_duplicate_ids.py`](find_files_with_duplicate_ids.py)           | Look through structure/.a3m files in a proteome and find protein IDs that were split into multiple fragments.        |
| [`listtar.py`](listtar.py)                                                       | List the contents of a .tar or .tar.gz file using Python's builtin `tarfile` library.                                |
| [`plddt_json_to_fasta.py`](plddt_json_to_fasta.py)                               | Convert a file containing pLDDT scores for each protein from the JSON format (used in our analysis) to FASTA format. |
| [`plot_multiple_neff_vs_neff_naive.bash`](plot_multiple_neff_vs_neff_naive.bash) | Plot Neff and Neff naive scores for different (random selection) proteins of the human proteome.                     |

If not specified otherwise, all scripts are expected to be executed from the project root directory.
