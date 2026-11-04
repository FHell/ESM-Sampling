#!/bin/bash
# CPU sampler run, preemptible (the default; docs/system/hpc.md §5):
#   sbatch [-c <cores>] [--mem-per-cpu=<MB>] hpc/jobs/cpu.sbatch <campaign> <config> [cpu|script <path> args...]
# Runs in standby on a shared node and is requeued automatically when preempted; it resumes from
# the latest state snapshot. Threads follow the allocation (hpc/run.sh): Julia threads = cores,
# BLAS/OpenMP 1. Past the time limit a continuation is submitted automatically (run.sh;
# RESUBMIT=<n> sbatch ... limits their number, default 10).
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
export ESM_JOB_SCRIPT="$ESM_PROJ/campaigns/$1/code/hpc/jobs/cpu.sbatch" ESM_JOB_ARGS="$*"   # continuations (run.sh)
exec bash "$ESM_PROJ/campaigns/$1/code/hpc/run.sh" "$@"
