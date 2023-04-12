This file contains instructions to build containers containing Neff calculation scripts using [mmseqs2](https://mmseqs.com/), [hhsuite](https://github.com/soedinglab/hh-suite), our original CPU-based Neff script, and our GPU-based Neff calculation script. The containers can be built using the following commands. All commands are expected to be ran from within the project's root directory.

```bash
# build mmseqs container
docker build -t neff-mmseqs:latest -f docker/mmseqs/Dockerfile .
# build "our" container, it also contains the reference script
docker build -t neff-gpu:latest -f docker/neffgpu/Dockerfile .
# build hhsuite container
docker build -t neff-hhsuite:latest -f docker/hhsuite/Dockerfile .
```

Please note that `neff-gpu` requires the `--gpus all` flag to be available for running on an accelerator. See [this](https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu) for more information.
