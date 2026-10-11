#!/bin/sh
# Run a list of experiment stages one after another (each stage is resumable).
cd "$(dirname "$0")/.."
for st in "$@"; do python3 scripts/run_experiments.py "$st" 2>&1 | grep --line-buffered -e '^\[' -e Error -e error; done
echo CHAIN_DONE
