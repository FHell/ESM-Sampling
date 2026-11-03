#!/bin/bash
# Run the PyPSA-Eur workflow with every cache inside the project folder.
cd /home/hellmann/Repos/ESM-Sampling/external/pypsa-eur
export XDG_CACHE_HOME=/home/hellmann/Repos/ESM-Sampling/external/cache/xdg
export UV_CACHE_DIR=/home/hellmann/Repos/ESM-Sampling/external/cache/uv
export MPLCONFIGDIR=/home/hellmann/Repos/ESM-Sampling/external/cache/mpl
export TMPDIR=/home/hellmann/Repos/ESM-Sampling/external/cache/tmp; mkdir -p $TMPDIR
export PATH=$PWD/.venv/bin:$PATH
snakemake "$@" --storage-cached-http-cache "$XDG_CACHE_HOME/snakemake-pypsa-eur"
