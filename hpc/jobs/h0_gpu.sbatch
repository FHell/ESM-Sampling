#!/bin/bash
# Campaign H0, GPU part: Reactant finds the H100 with the CUDA runtime fetched on a login node,
# offline; then a short family profile on the GPU.
#   sbatch hpc/jobs/h0_gpu.sbatch H0
#SBATCH --account=icone
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --gres=gpu:1
#SBATCH --job-name=esm-h0gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --chdir=/p/tmp/hellmann/esm
#SBATCH --output=/p/tmp/hellmann/esm/logs/%x-%j.out
ESM_PROJ=${ESM_PROJ:-/p/projects/coen/esm}
code="$ESM_PROJ/campaigns/$1/code"
source "$code/hpc/env.sh"; esm_load_julia || exit 2
export JULIA_PKG_OFFLINE=true TMPDIR=/localscratch/$USER/$SLURM_JOB_ID; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
cd "$code"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
julia --project=device -t 16 -e '
    using Reactant
    Reactant.set_default_backend("gpu")
    println("Reactant devices: ", Reactant.devices())
    x = Reactant.to_rarray(rand(1000)); f(x) = sum(abs2, x)
    println("compiled sum on the GPU: ", (@jit f(x)))'
FAMILIES=fast NSWEEPS=2 julia --project=device -t 16 device/scripts/profile_families.jl configs/device40_6h_gpu.toml "$ESM_SCRATCH/runs/$1/h0_gpu_profile"
