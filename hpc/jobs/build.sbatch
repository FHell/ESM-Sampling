#!/bin/bash
# Build job, preemptible (LP solves with HiGHS, weather-model fits, data preparation, PyPSA-Eur
# solves); a preempted job is requeued and starts over (finished cache entries are kept):
#   sbatch [-c <cores>] hpc/jobs/build.sbatch <campaign> <config> script <path> [args...]
# HiGHS uses ESM_LP_THREADS = the allocated cores.
#SBATCH --account=icone
#SBATCH --partition=priority
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --job-name=esm-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
exec bash "$ESM_PROJ/campaigns/$1/code/hpc/run.sh" "$@"
