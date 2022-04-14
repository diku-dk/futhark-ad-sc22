# futhark-ad-sc22
This is the artifact for the Futhark AD submission to SC22; this
repository contains all source code and data required to reproduce the
figures in the Futhark AD submission to SC22, "AD for an Array
Language with Nested Parallelism".

The artifact is distributed as a Docker container.

## Requirements
### Hardware
* An x86_64 CPU.
* A modern Nvidia GPU; the benchmarks in the paper were peformed with
  an A100 and a 2080 Ti. Some of the datasets (e.g., dataset `D_2` for
  kmeans, see Fig. 8 of the submission) require large amounts of video
  memory (~30 GiB) due to manifestation of large intermediate arrays.
  These benchmarks will fail on GPUs with an insufficient amount of
  memory; in these cases the table corresponding to the benchmark will
  not be reproduced.
* ~20 GiB of free disk space.

### Software
Running the container requires an x86_64 system running a Linux
distribution with support for the Nvidia Container Toolkit; please see
Nvidia's [installation
guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
for exact requirements and installation instructions.

## Setting up the Docker container
### GitHub Container Registry
The suggested way to obtain the container image is to pull from
the GitHub Container Registry:

    docker pull ghcr.io/diku-dk/futhark-ad-sc22:latest
    
Note: you may need to run docker as root with `sudo`.
    
### Building
Alternatively, the container image may be built by running

    docker build -t ghcr.io/diku-dk/futhark-ad-sc22:latest .

within the root directory of this repository. Note that this will
likely take a long time and that the Dockerfile is not deterministic;
it is **strongly** suggested to pull from the GitHub Container Registry.

## Running experiments
The docker container may be run interactively with

    docker run --rm -it --gpus all ghcr.io/diku-dk/futhark-ad-sc22:latest

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
machines.  The commands below only produce results for a single
machine (the one you are running on).  A comparison between machines
is not a contribution of the paper, so the artifact doesn't deal with
it.

## Manifest
This section describes every top-level directory and its purpose.

* `ADBench/`: a Git submodule containing a fork of the main ADBench
  repository, with Futhark implementations added.  We use only a small
  amount of ADBench, but it is simpler to include all of it than to
  try to exfiltrate the pertinent parts.  The Futhark implementations
  reside in `ADBench/src/cpp/modules/futhark`.

* `bin/`: precompiled binaries and scripts used in the artifact.

* `futhark/`: a Git submodule containing the Futhark compiler extended
  with support for AD.  This is the compiler used for the artifact,
  and can be used to (re)produce the `bin/futhark` executable with `make bin/futhark -B`.

* `tmp/`: used for storing raw results from running some of the benchmarks.
  (ADBench contains its own temporary directory, select other benchmarks store
  results in other folders.)
  
* `originals/`: original sources for the `lbm`, `rsbench` and `xsbench`
  benchmarks to compare against.
  
* `benchmarks/`: contains the source code and data for all benchmarks
  except for the ADBench benchmark.
  
