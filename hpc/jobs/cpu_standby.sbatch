#!/bin/bash
# Preemptible CPU run in the standby QOS (requeued on preemption; resumes from the checkpoint).
#   sbatch hpc/jobs/cpu_standby.sbatch <campaign> <config> [cpu|script <path> args...]
#SBATCH --account=coen
#SBATCH --partition=priority
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --job-name=esm-sby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
bash "$ESM_PROJ/campaigns/$1/code/hpc/run.sh" "$@"
