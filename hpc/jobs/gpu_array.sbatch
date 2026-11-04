#!/bin/bash
# Job array of device (Reactant, H100) runs, preemptible (gpupreempt, requeued on preemption): one
# run per line of a list file, each line "[NAME=value ...] <config> [device|script <path> args...]"
# (leading NAME=value tokens are environment settings of that run, e.g. FLOAT=32 FAMILIES=all;
# recorded in run.toml). One GPU per task.
#   sbatch --time=<estimate> --array=1-<n>%<concurrent> hpc/jobs/gpu_array.sbatch <campaign> <list file>
# The list path is relative to the campaign code (e.g. campaigns/H5/list.txt of the clone).
#SBATCH --account=icone
#SBATCH --partition=gpu
#SBATCH --qos=gpupreempt
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --job-name=esm-garr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%A_%a.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
code="$ESM_PROJ/campaigns/$1/code"
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$code/$2")
[ -n "$line" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $2" >&2; exit 2; }
read -r -a args <<< "$line"
envs=()
while [ ${#args[@]} -gt 0 ] && [[ "${args[0]}" =~ ^[A-Z_][A-Z0-9_]*= ]]; do envs+=("${args[0]}"); args=("${args[@]:1}"); done
[ ${#envs[@]} -gt 0 ] && export "${envs[@]}" ESM_LINE_ENV="${envs[*]}"
[ ${#args[@]} -lt 2 ] && args+=(device)
export TMPDIR=/localscratch/$USER/$SLURM_JOB_ID ESM_TMPDIR_CLEANUP=1; mkdir -p "$TMPDIR"   # removed by run.sh
exec bash "$code/hpc/run.sh" "$1" "${args[@]}"
