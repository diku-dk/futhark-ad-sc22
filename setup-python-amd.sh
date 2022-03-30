#!/bin/sh
if [ ! -d ".venv" ] 
then 
  $PYTHON -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade -r requirements-amd.txt  --extra-index-url https://download.pytorch.org/whl/rocm4.2


