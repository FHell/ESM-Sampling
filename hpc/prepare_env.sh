#!/bin/bash
# Prepare the environment of a code directory on a LOGIN node (internet), docs/system/hpc.md §3:
# instantiate and precompile both Julia projects into the shared depot, sync the Python
# environment, and record the environment. Compute nodes have no internet; jobs run offline.
#
#   bash hpc/prepare_env.sh [<code dir>]       (default: $ESM_CODE, the clone)
#
# First time: clone the repository (the script does it if $ESM_CODE does not exist).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/env.sh"
esm_dirs

code="${1:-$ESM_CODE}"
if [ ! -d "$code/.git" ] && [ ! -f "$code/.git" ]; then
    [ "$code" = "$ESM_CODE" ] || { echo "prepare_env: $code is not a git checkout" >&2; exit 1; }
    git clone "$ESM_REMOTE" "$ESM_CODE"
fi
cd "$code"
esm_load_julia

# data: the checkout reads data/ from the project space
[ -e data ] || ln -s "$ESM_PROJ/data" data

# Reactant: request the CUDA runtime variant (a login node has no GPU to detect); ESMDevice: run its
# precompile workload (it traces the device programs once at precompile time, so every job saves the
# Julia compilation of the tracing code; off by default on development machines)
cat > device/LocalPreferences.toml <<EOF
[Reactant_jll]
gpu = "$REACTANT_GPU"
gpu_version = "$REACTANT_GPU_VERSION"

[ESMDevice]
precompile_workload = true
EOF

export JULIA_NUM_PRECOMPILE_TASKS=${JULIA_NUM_PRECOMPILE_TASKS:-8}   # shared login node
unset JULIA_PKG_OFFLINE
nice -n 10 julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
nice -n 10 julia --project=device -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

if command -v uv >/dev/null 2>&1 && [ -f uv.lock ]; then
    nice -n 10 uv sync --frozen
else
    echo "prepare_env: uv or uv.lock not found; Python environment not synced" >&2
fi

# record
sha() { [ -f "$1" ] && sha256sum "$1" | cut -c1-16 || echo "none"; }
{
    echo "[[prepared]]"
    echo "code = \"$code\""
    echo "commit = \"$(git rev-parse HEAD)\""
    echo "date = \"$(date -Iseconds)\""
    echo "host = \"$(hostname)\""
    echo "julia = \"$(julia --startup-file=no -e 'print(VERSION)')\""
    echo "julia_module = \"${JULIA_BIN:-$JULIA_MODULE}\""
    echo "manifest = \"$(sha Manifest.toml)\""
    echo "manifest_device = \"$(sha device/Manifest.toml)\""
    echo "uv_lock = \"$(sha uv.lock)\""
    echo "reactant_gpu = \"$REACTANT_GPU $REACTANT_GPU_VERSION\""
    echo
} >> "$ESM_PROJ/env/ENVIRONMENT.toml"
echo "prepare_env: done for $code ($(git rev-parse --short HEAD)); recorded in $ESM_PROJ/env/ENVIRONMENT.toml"
echo "Next: model files (sbatch hpc/jobs/data.sbatch small), then campaign H0: bash hpc/new_campaign.sh H0;"
echo "      sbatch hpc/jobs/h0_cpu.sbatch H0; sbatch hpc/jobs/h0_gpu.sbatch H0 (logs in $ESM_SCRATCH/logs/)"
