#!/usr/bin/env bash
# Wait for scripts/fetch_era5.sh to finish (its PID file disappears), then rebuild
# data/processed/model256_era5.h5 from all complete ERA5 years (niced).
#   setsid nohup scripts/after_fetch_era5.sh > data/raw/era5/after_fetch.log 2>&1 &
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
while [ -e data/raw/era5/fetch.pid ] && kill -0 "$(cat data/raw/era5/fetch.pid)" 2>/dev/null; do sleep 60; done
if ! tail -1 data/raw/era5/fetch.log | grep -q "^.* done:"; then
  echo "$(date '+%F %T') fetch did not finish cleanly; not rebuilding"; exit 1
fi
echo "$(date '+%F %T') rebuilding model256_era5.h5"
nice -n 19 .venv/bin/python -I scripts/prepare_era5.py --force 2>&1 | grep -v "^INFO"
echo "$(date '+%F %T') finished with status ${PIPESTATUS[0]}"
