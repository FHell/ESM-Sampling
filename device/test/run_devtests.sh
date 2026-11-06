#!/bin/bash
# Device tests, one fresh Julia process per test file and case (running them all in one session
# exhausts memory: 17.5 GB). Locally and in a Slurm job (hpc/jobs/devtest.sbatch).
#
#   bash device/test/run_devtests.sh <spec> [<spec> ...]
#
# A spec is a file in device/test/ with optional environment variables after a colon, separated
# by commas:
#   test_comp.jl:CASE=G
#   test_replica_compiled.jl:ESM_DEVICE_COMPILE=1,ESM_DEVICE_FAMILIES=all
#   test_reactant_fams.jl:CASE=A,FAMS=all
#   comp:G            (short for test_comp.jl:CASE=G)
#
# Environment: DEVTEST_LOG (log file, default results/devtests-<date>.log), DEVTEST_THREADS
# (default: SLURM_CPUS_PER_TASK, else 4), DEVTEST_TIMEOUT (seconds per process, default 5400),
# DEVTEST_MEM (memory cap per process outside Slurm, via systemd-run if available; default 12G;
# in a Slurm job the job's --mem is the cap). Exit status: the number of failed specs.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
mkdir -p results
LOG=${DEVTEST_LOG:-$root/results/devtests-$(date +%Y%m%d-%H%M%S).log}
THREADS=${DEVTEST_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
TIMEOUT=${DEVTEST_TIMEOUT:-5400}
MEM=${DEVTEST_MEM:-12G}

cap=()
if [ -z "${SLURM_JOB_ID:-}" ] && command -v systemd-run >/dev/null 2>&1 \
   && systemd-run --user --scope -q true 2>/dev/null; then
    cap=(systemd-run --user --scope -q -p MemoryMax=$MEM -p MemorySwapMax=2G)
fi
nicecmd=()
[ -z "${SLURM_JOB_ID:-}" ] && nicecmd=(nice -n 19)     # this machine is shared (CLAUDE.md)

echo "devtests in $root at $(git rev-parse --short HEAD 2>/dev/null) on $(hostname), $THREADS threads; log $LOG"
failed=0
for spec in "$@"; do
    case "$spec" in comp:*) spec="test_comp.jl:CASE=${spec#comp:}" ;; esac
    file=${spec%%:*}
    vars=()
    [ "$file" != "$spec" ] && IFS=, read -r -a vars <<< "${spec#*:}"
    [ -f "device/test/$file" ] || { echo "== $spec: no file device/test/$file" | tee -a "$LOG"; failed=$((failed + 1)); continue; }
    echo "start $spec $(date +%T)" >> "$LOG"
    t0=$(date +%s)
    "${cap[@]}" env "${vars[@]}" timeout "$TIMEOUT" "${nicecmd[@]}" \
        julia --project=device -t "$THREADS" "device/test/$file" >> "$LOG" 2>&1
    rc=$?
    [ $rc -ne 0 ] && failed=$((failed + 1))
    echo "== $spec exit=$rc $(( $(date +%s) - t0 )) s" | tee -a "$LOG"
done
echo "ALL DONE $(date +%T): $failed of $# failed" | tee -a "$LOG"
exit $failed
