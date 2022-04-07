# futhark-ad-sc22
Artifact for the Futhark AD submission to SC22

# Set-up
1. Set the `PYTHON` env variable to be your Python binary. Note that Python 3.7 or greater is required:
```
export PYTHON=python
```
2. Execute [`setup-python.sh`](setup-python.sh) to set-up a Python virtual environment with the appropriate packages:
```
$ sh setup-python.sh
```
3. Launch the Python virtual environment:
```
$ source .venv/bin/activate
```
4. If benchmarking on the A100 or 2080Ti, set the `GPU` env variable appropriately (so the correct
tuning files and options are used):
```
$ export GPU=A100
```
or
```
$ export GPU=2080TI
```

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
