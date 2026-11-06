#!/bin/bash
# Device tests (arrays and the compile pass on Reactant's CPU backend) on a compute node, one
# Julia process per spec (device/test/run_devtests.sh; docs/system/testing.md):
#   sbatch [--time=…] [--mem=…] hpc/jobs/devtest.sbatch <code dir> <spec> [<spec> ...]
# <code dir> is a checkout prepared by hpc/prepare_env.sh (its data/ link and the precompiled
# device environment), e.g. a worktree of a branch under $ESM_PROJ/devtest/. Not a campaign:
# nothing is collected; the log is $ESM_SCRATCH/logs/devtests-<job id>.log and the Slurm output.
# The specs run one after another; submit several jobs for independent groups (priority: at
# most 5 jobs). Not requeued: a test cannot resume.
#SBATCH --account=icone
#SBATCH --partition=priority
#SBATCH --qos=priority
#SBATCH --no-requeue
#SBATCH --job-name=esm-devtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
code="${1:?usage: devtest.sbatch <code dir> <spec> ...}"; shift
source "$code/hpc/env.sh"; esm_load_julia || exit 2
export JULIA_PKG_OFFLINE=true
export DEVTEST_LOG="$ESM_SCRATCH/logs/devtests-${SLURM_JOB_ID:-local}.log"
exec bash "$code/device/test/run_devtests.sh" "$@"
