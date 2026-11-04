#!/bin/bash
# The run wrapper (docs/system/hpc.md §5–§7): the only entry point of a job that runs the sampler
# or a script. Called from the job templates in hpc/jobs/, inside a Slurm allocation.
#
#   bash hpc/run.sh <campaign> <config.toml> [cpu | device | script <path> [args...]]
#
# - code: the campaign's pinned worktree $ESM_PROJ/campaigns/<campaign>/code (must be clean);
# - run directory $ESM_SCRATCH/runs/<campaign>/<run id> on /p/tmp: the config (as given and as
#   run, with out, cache, checkpoint and stop file set), logs, outputs, checkpoint;
# - run id: $RUN_ID if set (continuations), else <campaign>-<config name>-<job id>[_<task>],
#   which a Slurm requeue keeps, so a requeued job resumes from its checkpoint;
# - SIGTERM (preemption, scancel) and SIGUSR1 (#SBATCH --signal=B:USR1@<lead>, before the time
#   limit) create the stop file; the driver then checkpoints at a sweep boundary and exits with
#   75; with RESUBMIT=<n> > 0 the wrapper submits a continuation job (afterany dependency);
# - at the end: the run record (run.toml), sacct figures, archive copy without checkpoints,
#   and one line in the runs index ($ESM_PROJ/ledger/runs.jsonl and the campaign's runs.jsonl).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/env.sh"
esm_dirs

campaign="${1:?usage: run.sh <campaign> <config> [cpu|device|script <path> args...]}"
config="${2:?usage: run.sh <campaign> <config> [cpu|device|script <path> args...]}"
mode="${3:-cpu}"
shift $(( $# < 3 ? $# : 3 ))

code="$ESM_PROJ/campaigns/$campaign/code"
[ -d "$code" ] || { echo "run.sh: no campaign $campaign (hpc/new_campaign.sh)" >&2; exit 2; }
cd "$code"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "run.sh: the campaign code $code has uncommitted changes; refusing to run" >&2; exit 2
fi
commit=$(git rev-parse HEAD)
esm_load_julia || exit 2
export JULIA_PKG_OFFLINE=true UV_OFFLINE=1 ESM_IGNORE_SIGTERM=1

jobid="${SLURM_JOB_ID:-local$$}"
task="${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
cfgname="$(basename "$config" .toml)"
run_id="${RUN_ID:-$campaign-$cfgname-${SLURM_ARRAY_JOB_ID:-$jobid}$task}"
rundir="$ESM_SCRATCH/runs/$campaign/$run_id"
mkdir -p "$rundir"
# Threads follow the allocation: nodes are shared, and libraries that size themselves from the
# node's 128 cores would oversubscribe the cores this job was given. Julia threads = allocated
# cores; BLAS and OpenMP single-threaded (their parallel regions would multiply with Julia's
# threads) unless a job sets ESM_BLAS_THREADS / ESM_OMP_THREADS; HiGHS gets the allocated cores
# (ESM_LP_THREADS; it runs inside one Julia task).
threads="${ESM_JULIA_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export JULIA_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="${ESM_BLAS_THREADS:-1}" MKL_NUM_THREADS="${ESM_BLAS_THREADS:-1}"
export OMP_NUM_THREADS="${ESM_OMP_THREADS:-1}"
export ESM_LP_THREADS="${ESM_LP_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

# the config as given and as run (absolute paths into the run directory and the caches)
[ -f "$rundir/config.orig.toml" ] || cp "$config" "$rundir/config.orig.toml"
# only the keys this code version defines (older code rejects unknown keys)
known=$(grep -oE '^[[:space:]]+(cache_dir|cache_mirror|checkpoint|stop_file)::' src/run.jl | tr -d ' :' | tr '\n' ',')
julia --startup-file=no -e '
    using TOML
    c = TOML.parsefile(ARGS[1]); rd = ARGS[2]; known = split(ARGS[5], ",")
    c["out"] = joinpath(rd, "out.h5")
    set(k, v) = k in known && (c[k] = v)
    set("cache_dir", ARGS[3]); set("cache_mirror", ARGS[4])
    set("checkpoint", joinpath(rd, "checkpoint.jls")); set("stop_file", joinpath(rd, "STOP"))
    open(io -> TOML.print(io, c), joinpath(rd, "config.toml"), "w")' \
    "$rundir/config.orig.toml" "$rundir" "$ESM_CACHE_DIR" "$ESM_CACHE_MIRROR" "$known" || exit 2
rm -f "$rundir/STOP"

case "$mode" in
    cpu)    cmd=(julia --project=. -t "$threads" scripts/run.jl "$rundir/config.toml") ;;
    device) cmd=(julia --project=device -t "$threads" device/scripts/run_device.jl "$rundir/config.toml") ;;
    script) s="${1:?run.sh: script mode needs a path}"; shift
            cmd=(julia --project=. -t "$threads" "$s" "$rundir/config.toml" "$@") ;;
    *)      echo "run.sh: mode $mode: cpu, device or script" >&2; exit 2 ;;
esac

# --- run record (start) ---------------------------------------------------------------------
start=$(date -Iseconds)
restarts="${SLURM_RESTART_COUNT:-0}"
{
    echo "[[attempt]]"
    echo "run_id = \"$run_id\""
    echo "campaign = \"$campaign\""
    echo "config = \"$config\""
    echo "config_sha = \"$(sha256sum "$rundir/config.orig.toml" | cut -c1-16)\""
    echo "commit = \"$commit\""
    echo "mode = \"$mode\""
    echo "command = \"${cmd[*]}\""
    echo "job_id = \"$jobid\""
    echo "array_task = \"${SLURM_ARRAY_TASK_ID:-}\""
    echo "partition = \"${SLURM_JOB_PARTITION:-}\""
    echo "qos = \"${SLURM_JOB_QOS:-}\""
    echo "nodes = \"${SLURM_JOB_NODELIST:-$(hostname)}\""
    echo "cpus = ${SLURM_CPUS_PER_TASK:-$threads}"
    echo "threads = \"julia $threads, blas $OPENBLAS_NUM_THREADS, omp $OMP_NUM_THREADS, lp $ESM_LP_THREADS\""
    echo "gpus = \"${SLURM_GPUS_ON_NODE:-${CUDA_VISIBLE_DEVICES:-}}\""
    echo "mem_mb = \"${SLURM_MEM_PER_NODE:-}\""
    echo "restart_count = $restarts"
    echo "start = \"$start\""
} >> "$rundir/run.toml"

# --- run, with the stop-file trap ------------------------------------------------------------
signalled=0
trap 'touch "$rundir/STOP"; signalled=1' TERM USR1
if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-$threads}" --export=ALL "${cmd[@]}" >> "$rundir/run.log" 2>&1 &
else
    "${cmd[@]}" >> "$rundir/run.log" 2>&1 &
fi
pid=$!
# a trapped signal interrupts `wait` (status > 128) while the driver keeps running: wait again
while true; do
    wait "$pid"; code_exit=$?
    kill -0 "$pid" 2>/dev/null || break
done
trap - TERM USR1

# finished; checkpointed (snapshot written, exit 75); preempted (signalled, no snapshot: Slurm
# requeues it, or it resumes from an earlier snapshot); failed
if   [ "$code_exit" -eq 0 ];  then status=finished
elif [ "$code_exit" -eq 75 ]; then status=checkpointed
elif [ "$signalled" -eq 1 ];  then status=preempted
else status=failed
fi
end=$(date -Iseconds)

# --- sacct of the step, outputs ---------------------------------------------------------------
acct=""
if [ -n "${SLURM_JOB_ID:-}" ] && command -v sacct >/dev/null; then
    acct=$(sacct -j "$SLURM_JOB_ID" -n -P --format=JobID,Elapsed,MaxRSS,TotalCPU,ConsumedEnergyRaw 2>/dev/null | tail -n 1)
fi
IFS='|' read -r _ a_elapsed a_maxrss a_cpu a_energy <<< "${acct:-||||}"
{
    echo "end = \"$end\""
    echo "exit = $code_exit"
    echo "status = \"$status\""
    echo "elapsed = \"$a_elapsed\""
    echo "max_rss = \"$a_maxrss\""
    echo "total_cpu = \"$a_cpu\""
    echo "energy_j = \"$a_energy\""
    echo "outputs = ["
    find "$rundir" -maxdepth 1 -type f ! -name 'checkpoint*' ! -name STOP -printf '    "%f %s",\n'
    echo "]"
    echo
} >> "$rundir/run.toml"

# --- archive and runs index (finished or failed runs; checkpointed and preempted runs continue) -
jstr() { printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')"; }
line="{\"run_id\":$(jstr "$run_id"),\"campaign\":$(jstr "$campaign"),\"config\":$(jstr "$config"),\"commit\":$(jstr "$commit"),\"mode\":$(jstr "$mode"),\"job_id\":$(jstr "$jobid"),\"status\":$(jstr "$status"),\"exit\":$code_exit,\"start\":$(jstr "$start"),\"end\":$(jstr "$end"),\"restarts\":$restarts,\"cpus\":$threads,\"elapsed\":$(jstr "$a_elapsed"),\"max_rss\":$(jstr "$a_maxrss"),\"energy_j\":$(jstr "$a_energy"),\"rundir\":$(jstr "$rundir")}"
(
    flock 9
    echo "$line" >> "$ESM_PROJ/ledger/runs.jsonl"
    echo "$line" >> "$ESM_PROJ/campaigns/$campaign/runs.jsonl"
) 9>> "$ESM_PROJ/ledger/.lock"
if [ "$status" = finished ] || [ "$status" = failed ]; then
    mkdir -p "$ESM_PROJ/archive/$campaign/$run_id"
    rsync -a --exclude 'checkpoint*' --exclude STOP "$rundir/" "$ESM_PROJ/archive/$campaign/$run_id/"
fi

# --- continuation ------------------------------------------------------------------------------
if [ "$status" = checkpointed ] && [ "${RESUBMIT:-0}" -gt 0 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    script="${ESM_JOB_SCRIPT:-$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=//p')}"
    sbatch --dependency=afterany:"$SLURM_JOB_ID" \
           --export=ALL,RUN_ID="$run_id",RESUBMIT=$(( RESUBMIT - 1 )) "$script" "$campaign" "$config" \
        && echo "run.sh: continuation of $run_id submitted ($(( RESUBMIT - 1 )) left)"
fi
echo "run.sh: $run_id $status (exit $code_exit); record $rundir/run.toml"
exit $code_exit
