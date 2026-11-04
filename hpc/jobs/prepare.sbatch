#!/bin/bash
# Environment preparation as a batch job on a login node (QOS io: internet, 4 cores):
#   sbatch hpc/jobs/prepare.sbatch [<code dir>]
#SBATCH --account=icone
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --job-name=esm-prep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
export JULIA_NUM_PRECOMPILE_TASKS=4
bash "${1:-$ESM_PROJ/ESM-Sampling}/hpc/prepare_env.sh" "${1:-$ESM_PROJ/ESM-Sampling}"
