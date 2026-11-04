#!/bin/bash
# Device run with guaranteed runtime (gpushort, not preempted):
#   sbatch hpc/jobs/gpu_short.sbatch <campaign> <config> [device|script <path> args...]
# One long-lived process batching chains and replicas (Reactant compiles 2–4 min per process).
#SBATCH --account=coen
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --gres=gpu:1
#SBATCH --job-name=esm-gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
export TMPDIR=/localscratch/$USER/$SLURM_JOB_ID; mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
c="$1"; cfg="$2"; shift 2
bash "$ESM_PROJ/campaigns/$c/code/hpc/run.sh" "$c" "$cfg" "${@:-device}"
