#!/bin/bash
# Build the model files on the cluster (needs internet: QOS io on a login node, 4 cores, 3 days;
# requeued if interrupted, resumes from the kept downloads and snakemake outputs):
#   sbatch hpc/jobs/data.sbatch small|pypsa256|pypsa256ts|all  (pypsa256ts needs no internet, only the pypsa256 build)
#SBATCH --account=icone
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --requeue
#SBATCH --job-name=esm-data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
bash "$ESM_PROJ/ESM-Sampling/hpc/prepare_data.sh" "${1:-all}"
