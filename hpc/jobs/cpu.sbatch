#!/bin/bash
# CPU sampler run (docs/system/hpc.md §5):  sbatch hpc/jobs/cpu.sbatch <campaign> <config> [cpu|script <path> args...]
# Threads = --cpus-per-task (override on the command line: sbatch -c 16 ...). Longer than 24 h:
# RESUBMIT=<n> sbatch ... chains continuation jobs that resume from the checkpoint.
#SBATCH --account=coen
#SBATCH --qos=short
#SBATCH --job-name=esm-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
bash "$ESM_PROJ/campaigns/$1/code/hpc/run.sh" "$@"
