#!/usr/bin/env bash
# Copy ERA5 hourly single-level fields from the PIK climate_data_central mount, one
# monthly global file at a time, cut each to the European box and delete the global
# copy. Sequential, niced, resumable (finished months are skipped).
#
#   scripts/fetch_era5.sh 2013 2003 2010 2017
#
# Remote layout: $REMOTE/<variable>/<variable>_YYYYMM.nc, global 0.25 deg, ~1.1 GB
# (t2m), 1.5 GB (u10, v10), 0.7 GB (ssrd) per month. Through the gvfs sftp mount
# one file takes ~5 min (~4 MB/s), so one year (48 files, ~57 GB) takes ~4 h.
#
# Rules: no listing or globbing on the mount (file names are constructed), one copy
# at a time, never process a file on the mount in place.
# Stop: kill the PID in data/raw/era5/fetch.pid (the current month is redone on restart).
set -u
cd "$(dirname "$0")/.."
REMOTE=${REMOTE:-/run/user/1000/gvfs/sftp:host=hpc/p/projects/climate_data_central/reanalysis/ERA5}
BASE=data/raw/era5
TMP=$BASE/tmp
OUT=$BASE/europe
MAX_GB=${MAX_GB:-80}        # local budget for data/raw/era5
MIN_FREE_GB=${MIN_FREE_GB:-40}
VARS="2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind surface_solar_radiation_downwards"
mkdir -p "$TMP" "$OUT"
echo $$ > "$BASE/fetch.pid"
log() { echo "$(date '+%F %T') $*" | tee -a "$BASE/fetch.log"; }

for year in "$@"; do
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    for v in $VARS; do
      f=${v}_${year}${m}.nc
      dst=$OUT/$v/$f
      [ -s "$dst" ] && continue
      mkdir -p "$OUT/$v"
      used=$(du -s --block-size=1G "$BASE" | cut -f1)
      free=$(df --output=avail --block-size=1G . | tail -1 | tr -d ' ')
      if [ "$used" -ge "$MAX_GB" ] || [ "$free" -le "$MIN_FREE_GB" ]; then
        log "STOP: budget (used ${used} GB of $MAX_GB, free ${free} GB)"; exit 2
      fi
      if [ ! -s "$TMP/$f" ]; then
        t0=$(date +%s)
        if ! ionice -c3 nice -n 19 cp "$REMOTE/$v/$f" "$TMP/$f.part"; then
          log "FAIL copy $v/$f"; rm -f "$TMP/$f.part"; exit 1
        fi
        mv "$TMP/$f.part" "$TMP/$f"
        log "copied $v/$f $(stat -c %s "$TMP/$f") B in $(( $(date +%s) - t0 )) s"
      fi
      if nice -n 19 .venv/bin/python -I scripts/era5_subset.py "$TMP/$f" "$dst" >> "$BASE/fetch.log" 2>&1; then
        rm -f "$TMP/$f"
      else
        log "FAIL subset $f (global copy kept in $TMP)"; exit 1
      fi
    done
  done
  log "year $year complete"
done
rm -f "$BASE/fetch.pid"
log "done: $*"
