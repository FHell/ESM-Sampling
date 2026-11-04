#!/bin/bash
# Build the model files on the cluster from public sources (docs/system/data_sources.md §5,
# docs/system/hpc.md §4). Needs internet: run it on a login node or as hpc/jobs/data.sbatch
# (QOS io). Writes into $ESM_PROJ/data/ (through the clone's data/ link) and records every file
# in $ESM_PROJ/data/processed/PROVENANCE.toml.
#
#   bash hpc/prepare_data.sh small      model.h5 (40 nodes) and model76.h5 (76 nodes)      [internet: io]
#   bash hpc/prepare_data.sh pypsa256   the PyPSA-Eur build esm256, then model256.h5     [internet: io]
#   bash hpc/prepare_data.sh all        small, then pypsa256                             [internet: io]
#   bash hpc/prepare_data.sh era5 [years, default 2003,2010,2013,2017]
#        model256_era5.h5 from the cluster's ERA5 folder: no internet, needs pypsa256 first;
#        run it on a compute node (hpc/jobs/era5.sbatch, standby). The window year 2013 must be
#        among the years.
#
# Re-running is safe: downloads, NASA POWER points (data/raw/power) and snakemake outputs are
# kept and reused; a requeued job continues where it stopped.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/env.sh"
esm_dirs
what="${1:?usage: prepare_data.sh small|pypsa256|all}"
cd "$ESM_CODE"
[ -e data ] || ln -s "$ESM_PROJ/data" data
cores="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py() { nice -n 10 uv run --frozen python "$@"; }

record() {   # file, command
    local f="$1"; shift
    {
        echo "[[file]]"
        echo "path = \"$f\""
        echo "sha256 = \"$(sha256sum "$f" | cut -c1-16)\""
        echo "bytes = $(stat -c %s "$f")"
        echo "commit = \"$(git rev-parse HEAD)\""
        echo "command = \"$*\""
        echo "date = \"$(date -Iseconds)\""
        echo "host = \"$(hostname)\""
        echo
    } >> "$ESM_PROJ/data/processed/PROVENANCE.toml"
}

build_small() {
    bash scripts/download_data.sh
    # model.h5: the 10-country example network (40 nodes), then the model-completion datasets
    py scripts/prepare_data.py --raw data/raw --out data/processed/model.h5
    py scripts/prepare_data.py --raw data/raw --out data/processed/model.h5 --augment
    record data/processed/model.h5 "prepare_data.py --out data/processed/model.h5; prepare_data.py --augment"
    # model76.h5: 27 countries of the example network (its 34 without AL, BA, GB, ME, MK, RS, EU)
    local c76="AT,BE,BG,CH,CZ,DE,DK,EE,ES,FI,FR,GR,HR,HU,IE,IT,LT,LU,LV,NL,NO,PL,PT,RO,SE,SI,SK"
    py scripts/prepare_data.py --raw data/raw --out data/processed/model76.h5 --countries "$c76"
    record data/processed/model76.h5 "prepare_data.py --out data/processed/model76.h5 --countries $c76"
}

build_pypsa256() {
    local pe="$ESM_PROJ/external/pypsa-eur"
    local rev; rev=$(cut -d' ' -f1 configs/pypsa-eur/REVISION)
    if [ ! -d "$pe/.git" ]; then
        git clone https://github.com/PyPSA/pypsa-eur.git "$pe"
    fi
    git -C "$pe" fetch -q origin && git -C "$pe" checkout -q "$rev"
    cp configs/pypsa-eur/config.esm.yaml "$pe/config/config.esm.yaml"
    # its Python environment, frozen from the local build (configs/pypsa-eur/requirements.txt)
    if [ ! -x "$pe/.venv/bin/snakemake" ]; then
        uv venv --python 3.12 "$pe/.venv"
        uv pip install --python "$pe/.venv/bin/python" -r configs/pypsa-eur/requirements.txt
    fi
    mkdir -p "$ESM_PROJ/external/cache"/{xdg,tmp,mpl}
    (
        cd "$pe"
        export XDG_CACHE_HOME="$ESM_PROJ/external/cache/xdg" TMPDIR="$ESM_PROJ/external/cache/tmp" \
               MPLCONFIGDIR="$ESM_PROJ/external/cache/mpl" PATH="$pe/.venv/bin:$PATH"
        nice -n 10 snakemake resources/esm256/networks/composed_2050.nc --cores "$cores" \
            --configfile config/config.esm.yaml --rerun-incomplete \
            --storage-cached-http-cache "$ESM_PROJ/external/cache/xdg/snakemake-pypsa-eur"
    )
    py scripts/prepare_data.py --raw data/raw --out data/processed/model256.h5 \
        --network "$pe/resources/esm256/networks/composed_2050.nc"
    record data/processed/model256.h5 "PyPSA-Eur $rev, snakemake composed_2050.nc (config.esm.yaml); prepare_data.py --network composed_2050.nc --out data/processed/model256.h5"
}

build_era5() {
    local years="${1:-2003,2010,2013,2017}"
    local pe="$ESM_PROJ/external/pypsa-eur"
    local out="$ESM_PROJ/data/raw/era5/europe"
    [ -f data/processed/model256.h5 ] && [ -d "$pe/resources/esm256" ] \
        || { echo "prepare_data.sh era5: run pypsa256 first (model256.h5 and the esm256 resources)" >&2; exit 2; }
    [[ ",$years," == *",2013,"* ]] || { echo "prepare_data.sh era5: the years must include 2013 (the calibration window)" >&2; exit 2; }
    # European cut-outs of the monthly global files, read in place from the shared folder: one file
    # at a time, paths constructed (never a listing or glob of the folder), finished months kept
    local vars="2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind surface_solar_radiation_downwards"
    local y m v f dst
    for y in ${years//,/ }; do
        for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
            for v in $vars; do
                f=${v}_${y}${m}.nc; dst="$out/$v/$f"
                [ -s "$dst" ] && continue
                mkdir -p "$out/$v"
                ionice -c3 nice -n 10 uv run --frozen python -I scripts/era5_subset.py "$ESM_ERA5/$v/$f" "$dst.part" \
                    && mv "$dst.part" "$dst" || { echo "prepare_data.sh era5: subset of $v/$f failed" >&2; rm -f "$dst.part"; exit 1; }
            done
        done
        echo "prepare_data.sh era5: year $y cut"
    done
    py scripts/prepare_era5.py --base data/processed/model256.h5 --out data/processed/model256_era5.h5 \
        --era5 "$out" --years "$years" --network "$pe/resources/esm256/networks/composed_2050.nc" \
        --resources "$pe/resources/esm256" --raw data/raw --force
    record data/processed/model256_era5.h5 "era5_subset.py from $ESM_ERA5 ($years); prepare_era5.py --base model256.h5 --years $years"
}

case "$what" in
    small)    build_small ;;
    pypsa256) build_pypsa256 ;;
    all)      build_small; build_pypsa256 ;;
    era5)     build_era5 "${2:-}" ;;
    *)        echo "prepare_data.sh: small, pypsa256, all or era5 [years]" >&2; exit 2 ;;
esac
echo "prepare_data.sh: done ($what); provenance in $ESM_PROJ/data/processed/PROVENANCE.toml"
