# futhark-ad-sc22
Artifact for the Futhark AD submission to SC22. This repository
contains all source code and data required to reproduce the figures in
the Futhark AD submission to SC22, "AD for an Array Language with
Nested Parallelism".

The artifact is distributed as a Docker container.

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

* `tmp/`: used for storing raw results from running the benchmarks.
  (ADBench contains its own temporary directory.)
  
* `benchmarks/`: contains the source code and data for all benchmarks
  except for the ADBench benchmark.
  
## Requirements
Running the container requires a working installation of the Nvidia
Container Toolkit; please see Nvidia's [installation
guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).


## Setting up the Docker container
### Docker Hub
The suggested way to obtain the container image is to pull from
[the repository on Docker Hub](https://hub.docker.com/r/zfnmxt/futhark-ad-sc22):

    docker pull zfnmxt/futhark-ad-sc22
    
### Building
Alternatively, the container image may be built by running

    docker build -t zfnmxt/futhark-ad-sc22 .
    
within the root directory of this repository. Note that this will
likely take a very long time.

## Running experiments
The docker container may be run interactively with

    docker run -it --gpus all zfnmxt/futhark-ad-sc22

### Reproducing individual tables
To reproduce the figures, run the corresponding `make` command in the
`/home/bench/futhark-ad-sc22` directory of the Docker container (this
is the default working directory).

* **Figure 6**: `make figure_6`.

* **Figure 7**: `make figure_7`.  Note that the Enzyme overheads are
  not computed by this artifact, but simply copied from the Enzyme
  paper for reference.
  
* **Figure 8**: `make figure_8`.

* **Figure 9**: `make figure_9`.

* **Figure 11**: `make figure_11`.

* **Figures 12 and 13**: `make figure_12_13`.

Note: Some of the figures in the paper contain results from multiple
machines.  The commands below only produce results for a single
machine (the one you are running on).  A comparison between machines
is not a contribution of the paper, so the artifact doesn't deal with
it.
