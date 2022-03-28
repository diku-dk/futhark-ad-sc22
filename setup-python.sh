#!/bin/sh
if [ ! -d ".venv" ] 
then 
  $PYTHON -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_releases.html --extra-index-url https://download.pytorch.org/whl/cu113


