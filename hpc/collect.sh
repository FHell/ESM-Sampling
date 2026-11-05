#!/bin/bash
# Collect a campaign's small results into the clone and commit them (docs/system/hpc.md §6).
# Results travel back through git: the user pushes from the cluster and pulls locally. Large
# outputs stay in $ESM_PROJ/archive/<campaign>/.
#
#   bash hpc/collect.sh <campaign> [<glob of result files, default: summary* *.csv *.toml *.txt>]
#
# Copies, from every archived run of the campaign, the run record (run.toml), the config as run
# and the files matching the globs into $ESM_CODE/campaigns/<campaign>/runs/<run id>/, plus the
# campaign's runs index; refuses files above ESM_COLLECT_MAX_MB (default 5) and a campaign total
# above ESM_COLLECT_TOTAL_MB (default 50); commits in the clone. It never pushes.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/env.sh"

id="${1:?usage: collect.sh <campaign> [globs...]}"; shift || true
globs=("$@"); [ ${#globs[@]} -gt 0 ] || globs=('summary*' '*.csv' '*.txt')
maxmb=${ESM_COLLECT_MAX_MB:-5}; totmb=${ESM_COLLECT_TOTAL_MB:-50}
arch="$ESM_PROJ/archive/$id"
dest="$ESM_CODE/campaigns/$id"
[ -d "$arch" ] || { echo "collect: no archive $arch" >&2; exit 1; }
mkdir -p "$dest/runs"

skipped=0
for run in "$arch"/*/; do
    rid=$(basename "$run"); out="$dest/runs/$rid"; mkdir -p "$out"
    for f in run.toml config.toml config.orig.toml stages.csv summary_device.toml; do [ -f "$run/$f" ] && cp "$run/$f" "$out/"; done   # always: the run record and the cost inputs
    for g in "${globs[@]}"; do
        for f in "$run"/$g; do
            [ -f "$f" ] || continue
            sz=$(stat -c %s "$f")
            if [ "$sz" -gt $(( maxmb * 1024 * 1024 )) ]; then
                echo "collect: skipped $(basename "$f") of $rid ($(( sz / 1024 / 1024 )) MB > $maxmb MB); it stays in the archive" >&2
                skipped=$(( skipped + 1 )); continue
            fi
            cp "$f" "$out/"
        done
    done
done
[ -f "$ESM_PROJ/campaigns/$id/runs.jsonl" ] && cp "$ESM_PROJ/campaigns/$id/runs.jsonl" "$dest/runs.jsonl"

total=$(du -sm "$dest" | cut -f1)
if [ "$total" -gt "$totmb" ]; then
    echo "collect: $dest is $total MB > $totmb MB; nothing committed (narrow the globs or raise ESM_COLLECT_TOTAL_MB)" >&2
    exit 1
fi
# the cost table (wall, allocated core-hours, CPU hours, GPU-hours, energy per run, stage and campaign)
if esm_load_julia >/dev/null 2>&1; then
    julia --startup-file=no "$here/cost_table.jl" "$dest" || echo "collect: cost table failed (see above); results collected without it" >&2
else
    echo "collect: no julia; cost table not written (run hpc/cost_table.jl $dest later)" >&2
fi
cd "$ESM_CODE"
git add "campaigns/$id"
if git diff --cached --quiet; then
    echo "collect: nothing new for $id"
else
    n=$(ls "$dest/runs" | wc -l)
    git commit -q -m "Campaign $id: results of $n runs (collected on $(hostname) $(date -I))"
    echo "collect: committed campaigns/$id ($total MB, $n runs, $skipped files skipped). Review, then push yourself."
fi
