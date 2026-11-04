#!/bin/bash
# Build job (LP solves with HiGHS, weather-model fits, data preparation, PyPSA-Eur solves):
#   sbatch hpc/jobs/build.sbatch <campaign> <config> script <path> [args...]
#SBATCH --account=coen
#SBATCH --qos=short
#SBATCH --job-name=esm-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
bash "$ESM_PROJ/campaigns/$1/code/hpc/run.sh" "$@"
