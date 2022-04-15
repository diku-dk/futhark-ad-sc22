# futhark-ad-sc22
This is the artifact for the Futhark AD submission to SC22; this
repository contains all source code and data required to reproduce
figures 6/7/8/9/11/12/13 in the Futhark AD submission to SC22, "AD for
an Array Language with Nested Parallelism".

The artifact is distributed as two a Docker containers, one for
running on Nvidia GPUs (CUDA) and one on AMD GPUs (ROCm). The artifact
can also be run without Docker, although there are many dependencies.
A section below describes how to do this.

## Requirements
### Hardware
* An x86_64 CPU.
* A modern Nvidia GPU or AMD GPU; the benchmarks in the paper were
  peformed with an A100 and a 2080 Ti (on a few select benchmarks) on
  the Nvidia side and an MI100 on the AMD side. Most of the benchmarks
  require large amounts of video memory (up to 30 GiB). These benchmarks
  will fail on GPUs with an insufficient amount of memory; in these
  cases the table corresponding to the benchmark will not be
  reproduced.
* ~20 GiB of free disk space.

### Software
Running the container requires an x86_64 system running a Linux
distribution with support for Docker. For the Nvidia container, the
Nvidia Container Toolkit is also necessary; please see Nvidia's
[installation
guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
for exact requirements and installation instructions.

## Setting up the Docker container
### GitHub Container Registry
The suggested way to obtain the container image is to pull from
the GitHub Container Registry. For the CUDA container, run

    docker pull ghcr.io/diku-dk/futhark-ad-sc22:cuda
 
and for the ROCm container, run

    docker pull ghcr.io/diku-dk/futhark-ad-sc22:rocm
    
Note: you may need to run docker as root with `sudo`.
    
### Building
Alternatively, the container image may be built by running

    docker build -t ghcr.io/diku-dk/futhark-ad-sc22:[cuda|rocm] .

within the root directory of this repository. Note that this will
likely take a long time and that the Dockerfile is not deterministic;
it is **strongly** recommended to pull from the GitHub Container Registry.

## Running experiments
The CUDA container may be run interactively with

    docker run --rm -it --gpus all ghcr.io/diku-dk/futhark-ad-sc22:cuda

and the ROCm contaier may be run interactively with

    docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video  ghcr.io/diku-dk/futhark-ad-sc22:rocm

### Reproducing individual tables
To reproduce the figures, run the corresponding `make` command in the
`/home/bench/futhark-ad-sc22` directory of the Docker container (this
is the default working directory):

* **Figure 6**: `make figure_6`.

* **Figure 7**: `make figure_7`.  Note that the Enzyme overheads are
  not computed by this artifact, but simply copied from the Enzyme
  paper for reference.
  
* **Figure 8**: `make figure_8`. Note that Figure 8 requires ~30 GiB
  of video memory to reproduce.

* **Figure 9**: `make figure_9`.

* **Figure 11**: `make figure_11`.

* **Figures 12 and 13**: `make figure_12_13`.

Note: Some of the figures in the paper contain results from multiple
machines.  The commands above only produce results for a single
machine (the one you are running on).  A comparison between machines
is not a contribution of the paper, so the artifact doesn't deal with
it.

Note: Only figures 8,9, and 11 and 12 are supported on the ROCm
container.  Constructing the other figures on the ROCm container has
undefined behavior.

## Running without the Docker container

It is possible to run the artifact without using the Docker container,
although it is somewhat intricate.  If any of the below seems wrong or
confusing, you can always peruse [Dockerfile.cuda](Dockerfile.cuda) or
[Dockerfile.rocm](Dockerfile.rocm) to see how the containers themselves
are constructed.

### Dependencies

You need the following components:

* [Git Large File Storage](https://git-lfs.github.com/) (available in
  many package managers).

* A working CUDA or OpenCL on your system.  The environment variables
  `LIBRARY_PATH`, `LD_LIBRARY_PATH`, and `CPATH` must be set such that
  including and linking against OpenCL/CUDA libraries works.  On most
  systems with CUDA this means:

  ```
  $ export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
  $ export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
  $ export CPATH=/usr/local/cuda/include:$CPATH
  ```

  But note that some systems installs CUDA in weird locations.

* [The Nix package manager.](https://nixos.org/download.html)
* Python 3.8 and `pip` (available in basically all package managers).
* Several Python packages. On CUDA, these are installable with:

  ```
  $ pip3 install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
  $ pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
  $ pip3 install futhark-data prettytable
  ```
 On ROCm, with:

  ```
  RUN pip3 install --upgrade torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm4.5.2
  RUN pip3 install futhark-data prettytable
   ```

### Preparation

Clone this repository and then initialise the submodules containing
ADBench and the Futhark compiler itself:

```
$ git lfs install # If you have not used git-lfs before.
$ git clone https://github.com/diku-dk/futhark-ad-sc22
$ cd futhark-ad-sc22
$ git submodule update --init ADBench
$ git submodule update --init futhark
```

**Optional:** recompile the Futhark compiler binary (but the included
one works as well): `make -B bin/futhark`.

### Running targets

Before running, you must set the environment variable `GPU` to either
`A100` if you have an NVIDIA GPU, or `MI100` if you have an AMD GPU:

```
export GPU=[A100|MI100]
```

After this, you should be able to use the Makefile targets [listed
above](https://github.com/diku-dk/futhark-ad-sc22#reproducing-individual-tables)
to reproduce the individual tables.

## Manifest
This section describes every top-level directory and its purpose.

* `ADBench/`: a Git submodule containing a fork of the main ADBench
  repository, with Futhark implementations added.  We use only a small
  amount of ADBench, but it is simpler to include all of it than to
  try to exfiltrate the pertinent parts.  The Futhark implementations
  reside in `ADBench/src/cpp/modules/futhark`.

* `benchmarks/`: contains the source code and data for all benchmarks
  except for the ADBench benchmark.

* `bin/`: precompiled binaries and scripts used in the artifact.

* `futhark/`: a Git submodule containing the Futhark compiler extended
  with support for AD.  This is the compiler used for the artifact,
  and can be used to (re)produce the `bin/futhark` executable with `make bin/futhark -B`.

* `originals/`: original sources for the `lbm`, `rsbench` and `xsbench`
  benchmarks to compare against.

* `tmp/`: used for storing raw results from running some of the benchmarks.
  (ADBench contains its own temporary directory, select other benchmarks store
  results in other folders.)
