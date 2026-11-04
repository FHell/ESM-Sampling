#!/bin/bash
# Short analysis job (fast start in the priority QOS):
#   sbatch hpc/jobs/analysis.sbatch <campaign> <config> script <path> [args...]
#SBATCH --account=coen
#SBATCH --partition=priority
#SBATCH --qos=priority
#SBATCH --job-name=esm-ana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
bash "$ESM_PROJ/campaigns/$1/code/hpc/run.sh" "$@"
