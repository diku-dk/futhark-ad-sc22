# futhark-ad-sc22
Artifact for the Futhark AD submission to SC22

# Set-up
1. Set the `PYTHON` env variable to be your Python binary. Note that Python 3.7 or greater is required:
```
export PYTHON=python
```
2. Execute [`setup-python.sh`](setup-python.sh) to set-up a Python virtual environment with the appropriate packages:
```
$ chmod +x setup-python.sh
$ ./setup-python.sh
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
