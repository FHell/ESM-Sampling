#!/bin/bash
# Campaign H0, CPU part: the prepared environment works offline on a compute node.
#   sbatch hpc/jobs/h0_cpu.sbatch H0
# Runs the deterministic test suite offline, then a short CPU smoke run through the wrapper.
#SBATCH --account=icone
#SBATCH --partition=priority
#SBATCH --qos=priority
#SBATCH --job-name=esm-h0cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
code="$ESM_PROJ/campaigns/$1/code"
source "$code/hpc/env.sh"; esm_load_julia || exit 2
export JULIA_PKG_OFFLINE=true
cd "$code"
echo "== julia $(julia --startup-file=no -e 'print(VERSION)') on $(hostname)"
julia --project=. -t 8 -e 'using ESMSampling; println("ESMSampling loads offline")'
EXACT_SKIP=1 julia --project=. -t 8 test/runtests.jl || { echo "H0: test suite failed; smoke run skipped" >&2; exit 1; }
bash "$code/hpc/run.sh" "$1" hpc/configs/h0_smoke.toml cpu
