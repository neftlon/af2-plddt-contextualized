#!/bin/sh
# downloads the human proteome from the AlphaFold 2 database to the ./data directory

mkdir -p ./data
pushd ./data
wget https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v3.tar
popd

