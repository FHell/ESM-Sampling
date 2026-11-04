#!/bin/bash
# CPU sampler run, preemptible (the default; docs/system/hpc.md §5):
#   sbatch [-c <cores>] [--mem-per-cpu=<MB>] hpc/jobs/cpu.sbatch <campaign> <config> [cpu|script <path> args...]
# Runs in standby on a shared node and is requeued automatically when preempted; it resumes from
# the latest state snapshot. Threads follow the allocation (hpc/run.sh): Julia threads = cores,
# BLAS/OpenMP 1. RESUBMIT=<n> sbatch ... chains continuations past the time limit.
#SBATCH --account=icone
#SBATCH --partition=priority
#SBATCH --qos=standby
#SBATCH --requeue
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
