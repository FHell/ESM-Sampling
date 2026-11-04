#!/bin/bash
# Job array, preemptible: one run per line of a list file, each line "<config> [cpu|script <path> args...]".
#   sbatch --array=1-<n>%<concurrent> [-c <cores>] hpc/jobs/cpu_array.sbatch <campaign> <list file>
# The list path is relative to the campaign code (e.g. campaigns/H3/list.txt of the clone).
#SBATCH --account=icone
#SBATCH --partition=priority
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --job-name=esm-arr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%A_%a.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
code="$ESM_PROJ/campaigns/$1/code"
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$code/$2")
[ -n "$line" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $2" >&2; exit 2; }
read -r -a args <<< "$line"
bash "$code/hpc/run.sh" "$1" "${args[@]}"
