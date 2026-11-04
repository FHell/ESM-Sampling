#!/bin/bash
# Create a campaign on a LOGIN node (docs/system/hpc.md §2, §6): a git worktree of the clone at a
# pinned commit (the code every run of the campaign uses), its prepared environment, its run and
# archive directories, and its record directory in the clone (campaigns/<id>/, committed).
#
#   bash hpc/new_campaign.sh <campaign id, e.g. H2> [<commit-ish, default: HEAD of the clone>]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/env.sh"
esm_dirs

id="${1:?usage: new_campaign.sh <campaign id> [<commit-ish>]}"
rev="${2:-HEAD}"
[[ "$id" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "new_campaign: campaign id may contain only letters, digits, _ . -" >&2; exit 1; }

cdir="$ESM_PROJ/campaigns/$id"
[ -e "$cdir/code" ] && { echo "new_campaign: $cdir/code exists already" >&2; exit 1; }
commit=$(git -C "$ESM_CODE" rev-parse "$rev")
mkdir -p "$cdir"
git -C "$ESM_CODE" worktree add --detach "$cdir/code" "$commit"
bash "$here/prepare_env.sh" "$cdir/code"

mkdir -p "$ESM_SCRATCH/runs/$id" "$ESM_PROJ/archive/$id"
rec="$ESM_CODE/campaigns/$id"
mkdir -p "$rec"
if [ ! -f "$rec/campaign.toml" ]; then
    cat > "$rec/campaign.toml" <<EOF
# Campaign $id (docs/campaigns.md, row $id; docs/system/hpc.md)
id = "$id"
commit = "$commit"
created = "$(date -Iseconds)"
code = "$cdir/code"
runs = "$ESM_SCRATCH/runs/$id"
archive = "$ESM_PROJ/archive/$id"
question = ""        # fill in: the question of the row in docs/campaigns.md
design = ""          # fill in: configurations, sizes, seeds, resources
EOF
fi
echo "new_campaign: $id at $(git -C "$ESM_CODE" rev-parse --short "$commit"); code in $cdir/code"
echo "Fill in $rec/campaign.toml, then submit jobs from hpc/jobs/ with CAMPAIGN=$id."
