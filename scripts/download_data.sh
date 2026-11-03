#!/usr/bin/env bash
# Download the PyPSA-Eur example network and the OPSD weather/load data into data/raw.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
cd data/raw
get() { [ -s "$2" ] || curl -sSL -o "$2" "$1"; }
get https://data.pypsa.org/networks/examples/latest/carbon_management.nc carbon_management.nc
get https://data.open-power-system-data.org/weather_data/2020-09-16/weather_data.csv weather_data.csv
get https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv time_series_60min_singleindex.csv
ls -la
