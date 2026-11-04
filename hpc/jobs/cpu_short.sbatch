#!/bin/bash
# CPU run with guaranteed runtime (QOS short, not preempted), for when preemption would hurt:
#   sbatch [-c <cores>] hpc/jobs/cpu_short.sbatch <campaign> <config> [cpu|script <path> args...]
#SBATCH --account=icone
#SBATCH --qos=short
#SBATCH --job-name=esm-cpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
bash "$ESM_PROJ/campaigns/$1/code/hpc/run.sh" "$@"
