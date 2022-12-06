#!/usr/bin/env bash
# make Python imports work from current directory
export PYTHONPATH="${PYTHONPATH}:`pwd`"

./af22c/plots/neff_vs_neff_naive.py -p A8MX80
./af22c/plots/neff_vs_neff_naive.py -p Q8IUR7
./af22c/plots/neff_vs_neff_naive.py -p O43242
./af22c/plots/neff_vs_neff_naive.py -p P06310
./af22c/plots/neff_vs_neff_naive.py -p Q96QF7
