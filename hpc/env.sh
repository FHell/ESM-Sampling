# Environment of every job and of the preparation on the PIK cluster (docs/system/hpc.md).
# Source it:  source hpc/env.sh
# Everything can be overridden by setting the variable before sourcing.

# --- who and where -------------------------------------------------------------------------
export ESM_USER=${ESM_USER:-hellmann}
export ESM_PROJECT=${ESM_PROJECT:-coen}
export ESM_ACCOUNT=${ESM_ACCOUNT:-icone}                      # Slurm account (--account); the project directory is coen
export ESM_PROJ=${ESM_PROJ:-/p/projects/$ESM_PROJECT/esm}      # permanent, backed up
export ESM_SCRATCH=${ESM_SCRATCH:-/p/tmp/$ESM_USER/esm}       # fast, purged when inactive
export ESM_CODE=${ESM_CODE:-$ESM_PROJ/ESM-Sampling}           # the git clone; results are committed here
export ESM_REMOTE=${ESM_REMOTE:-https://github.com/FHell/ESM-Sampling}
export ESM_ERA5=${ESM_ERA5:-/p/projects/climate_data_central/reanalysis/ERA5}   # shared, read-only: never scan it recursively

# --- Julia -----------------------------------------------------------------------------------
# The Manifests are resolved with Julia 1.12.2, the cluster's module julia/1.12.2. JULIA_BIN (a
# julia binary) replaces the module if set.
export JULIA_MODULE=${JULIA_MODULE:-julia/1.12.2}
export JULIA_DEPOT_PATH=${ESM_JULIA_DEPOT:-$ESM_PROJ/env/julia_depot}
export JULIA_CPU_TARGET=${JULIA_CPU_TARGET:-"generic;znver4,clone_all"}   # login (EPYC 9354) and compute (9554): Zen 4
export JULIA_HISTORY=$ESM_PROJ/env/julia_history
# Reactant chooses its XLA runtime variant (CPU only / CUDA) when it is installed, from the GPU it
# sees; login nodes have none, so the CUDA variant is requested explicitly (prepare_env.sh also
# writes it into device/LocalPreferences.toml). 12.9 for CUDA 12.x drivers, 13.1 for 13.x; the
# cluster's CUDA module is cuda/13.4.3, so the GPU nodes' driver serves CUDA 13 (H0 checks it with
# nvidia-smi). Reactant brings its own CUDA runtime: the jobs do not load the cuda module.
export REACTANT_GPU=${REACTANT_GPU:-cuda}
export REACTANT_GPU_VERSION=${REACTANT_GPU_VERSION:-13.1}

# --- Python (data preparation, PyPSA-Eur) ---------------------------------------------------
# Package indexes from the user's shell (e.g. UV_INDEX for a PyTorch wheel index) reach the jobs
# through Slurm's environment export and would redirect the resolution of the pinned PyPSA-Eur
# requirements; only the default index (PyPI) and uv.lock are used here.
unset UV_INDEX UV_INDEX_URL UV_EXTRA_INDEX_URL UV_DEFAULT_INDEX UV_FIND_LINKS UV_INDEX_STRATEGY \
      PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS
export UV_CACHE_DIR=$ESM_PROJ/env/uv_cache
export UV_PROJECT_ENVIRONMENT=$ESM_PROJ/env/venv
export UV_PYTHON_INSTALL_DIR=$ESM_PROJ/env/uv_python               # uv-managed Python builds
export XDG_CACHE_HOME=$ESM_PROJ/env/xdg_cache
export MPLCONFIGDIR=$ESM_PROJ/env/mpl

# --- caches (content-addressed; shared by all runs; the mirror is the durable fallback) ------
export ESM_CACHE_DIR=${ESM_CACHE_DIR:-$ESM_SCRATCH/cache}
export ESM_CACHE_MIRROR=${ESM_CACHE_MIRROR:-$ESM_PROJ/cache}

# --- threads: set per job from the allocation by hpc/run.sh (shared nodes); outside a job ------
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

esm_load_julia() {
    if [ -n "${JULIA_BIN:-}" ]; then
        export PATH="$(dirname "$JULIA_BIN"):$PATH"
    elif type module >/dev/null 2>&1; then
        module load "$JULIA_MODULE" || { echo "env.sh: module $JULIA_MODULE not found; set JULIA_MODULE or JULIA_BIN" >&2; return 1; }
    fi
    local v
    v=$(julia --startup-file=no -e 'print(VERSION.major, ".", VERSION.minor)') || return 1
    if [ "$v" != "1.12" ]; then
        echo "env.sh: julia $v found, the Manifests need 1.12 (set JULIA_MODULE or JULIA_BIN)" >&2
        return 1
    fi
}

esm_dirs() {
    mkdir -p "$ESM_PROJ"/{env/julia_depot,env/uv_cache,env/xdg_cache,env/mpl,data/raw,data/processed,cache,archive,ledger,campaigns} \
             "$ESM_SCRATCH"/{cache,runs,logs}
}
