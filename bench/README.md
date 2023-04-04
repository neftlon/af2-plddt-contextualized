# Benchmarks

This directory contains code for comparing the performance of mmseqs and neff_gpu.py.

* You can run benchmarks with the `bench.py` scripts. (Please run the script from the project's root directory (not this one, the one above).)

## Build containers

We provide docker containers with the environment necessary to benchmark each metric. In the following, commands for building and running these containers are shown. All commands are expected to be ran from within the project's root directory.

```bash
# build mmseqs container
docker build -t neff-mmseqs:latest -f docker/mmseqs/Dockerfile .
# build "our" container
docker build -t neff-gpu:latest -f docker/neffgpu/Dockerfile .
```
