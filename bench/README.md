# Benchmarks

This directory contains code for comparing the performance of mmseqs, hhsuite, gapcount, and neff_gpu.py using isolated containers.

You can run benchmarks with the `bench.py` scripts. (Please run the script from the project's root directory (not this one, the one above).) Note that the script expects the containers to be built in advance, as described in this [readme](../docker/README.md).

```bash
# run benchmarks
./bench/bench.py
# analyze run-time in notebook
jupyter notebook bench/analysis.ipynb
```

