#!/bin/bash

# execute this script from within the project's root directory

mkdir -p ./data
pushd ./data
# download the human proteome predictions from the AlphaFold 2 database
wget -Nc https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v3.tar
# download precomputed SETH predictions for the human proteome
wget -Nc -O Human_SETH_preds.txt https://zenodo.org/record/6673817/files/Human_SETH_preds.txt?download=1
popd
