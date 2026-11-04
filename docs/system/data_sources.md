# Data sources and provenance

This document records where every input of the model comes from: the PyPSA-Eur networks and
the configuration of the PyPSA-Eur build, the NASA POWER weather, the OPSD load and
temperature data, and the ERA5 weather with its retrieval and processing. It ends with the
provenance of each model input file. The extraction into the model's HDF5 files is described in
[model.md](model.md) §5, their layout in [data.md](data.md).

## 1. Networks

**PyPSA example network** `carbon_management.nc` (downloaded by `scripts/download_data.sh` from
`data.pypsa.org/networks/examples/latest/`; network name "Hybrid Scenario from
https://www.nature.com/articles/s41560-025-01752-6"). A sector-coupled 2050 net-zero network
with 91 AC buses and 168 snapshots at 3-hour resolution, 1–21 May 2013. Only its electricity
layer is used (`extract_network` in `scripts/prepare_data.py`): AC lines with their optimized
ratings `s_nom_opt`, DC links with `p_nom_opt`, onwind, offwind and solar from the generators
with their potentials, gas from the OCGT links (capex and efficiency, fuel price plus CO₂
cost at the configured price), fixed nuclear and run-of-river, battery and H₂ from store plus
charger and discharger links at fixed energy-to-power ratios, pumped hydro and reservoir hydro
from the storage units. It gives `model.h5` (10 countries, 40 nodes) and `model76.h5` (27
countries, 76 nodes).

**PyPSA-Eur build** of the 10-country region at 256 clusters:

- **Code**: a clone of PyPSA-Eur in `external/pypsa-eur` at revision
  `8d4b7d3073e027f52aabcd4675afcb47b767cdc5` (recorded in `configs/pypsa-eur/REVISION`).
- **Configuration**: `configs/pypsa-eur/config.esm.yaml` (a copy of
  `external/pypsa-eur/config/config.esm.yaml`): run name `esm256`; countries DE, FR, BE, NL,
  LU, CH, AT, CZ, PL, DK; snapshots of the year 2013; sector coupling disabled; 256 clusters;
  CO₂ budget 5 % in 2050; HiGHS as solver.
- **Run**: `external/run_pypsa_eur.sh` runs snakemake with every cache inside the project folder
  (`external/cache/`). The output used is the composed, unsolved network
  `external/pypsa-eur/resources/esm256/networks/composed_2050.nc`, plus the resources
  `availability_matrix_<tech>.nc`, `pop_layout_total.nc` and `onshore_regions.geojson` for the
  ERA5 mapping (§4).
- **Extraction** (`extract_network_elec`, `prepare_data.py --network …`): existing line and
  link ratings (`s_nom`, `p_nom`), renewable potentials, extendable gas (OCGT, CCGT), fixed
  nuclear and run-of-river, battery and H₂ (store plus links), pumped hydro and reservoir hydro,
  and the brownfield fleets (coal, lignite, oil, biomass, waste, geothermal). Result:
  `model256.h5` with 256 nodes, 480 lines and 4 DC links.

## 2. NASA POWER weather

Hourly point data from the NASA POWER API (community RE, UTC) at the coordinates of every
network node, 2001–2022: 50 m wind speed `WS50M` (MERRA-2), all-sky surface irradiance
`ALLSKY_SFC_SW_DWN` (CERES) and 2 m temperature `T2M`. Responses are cached in
`data/raw/power/`.

**Conversion** (`build_weather`). Wind: a generic power curve at 100 m hub height (wind
extrapolated with shear exponent 1/7; cut-in 3, rated 12, cut-out 25 m/s), smoothed to
represent the spread of a fleet. Solar: capacity factor $\mathrm{clip}(s\cdot\mathrm{GHI}/1000,
0, 1)$. One scale $s$ per generator is calibrated so that the mean capacity factor over the
network's snapshot window equals the mean PyPSA-Eur `p_max_pu` of that generator (weighted by
potential). The window mean is not monotone in $s$ for wind (cut-out at 25 m/s), so the scale
is the smallest root on $[0.2, 4]$ (`smallest_root`: a grid of 39 points scanned upwards to
the first sign change, then `brentq` in that cell); without a root, the grid point of the
largest window mean, and the build reports how many generators miss their target by more than
0.005. Every year has 8760 hours; 29 February is dropped.

> **Status: deviation.** The model files built so far, locally and on the cluster, predate the
> smallest-root calibration: the scale was bracketed on $[0.2, 4]$ and set to 4 whenever the
> window mean at 4 lay below the target, which puts the node deep in the cut-out regime, where
> its output falls as the wind rises (correlation with the wind speed about −0.6). Affected:
> `model.h5` 9 of 19 offwind and 1 of 40 onwind nodes, `model76.h5` 10 of 51 and 2 of 76,
> `model256.h5` 49 of 64 and 33 of 256 (2013 offwind mean 0.282 against a target of 0.434).
> A rebuild changes only these wind columns. The files need rebuilding ([hpc.md](hpc.md)
> §8.3, D1 and D2); `model256_era5.h5` is not affected (its weather is ERA5's).

## 3. Load

National load models fitted to OPSD data (both downloaded by `download_data.sh`):
`time_series_60min_singleindex.csv` (ENTSO-E actual load) and `weather_data.csv`
(population-weighted national temperature). Per country, load relative to its mean is regressed
on hour-of-week indicators, their products with one annual harmonic, and heating and cooling
degrees ($\max(15-T,0)$, $\max(T-22,0)$) of a 24-hour exponentially weighted temperature, over
2015–2019. The model is then driven by each node's own temperature and scaled so that the
node's mean load over the network's snapshot window equals the network's nodal mean load.

## 4. ERA5

**Source.** The ERA5 reanalysis on the PIK cluster, reached through a gvfs sftp mount at
`/run/user/1000/gvfs/sftp:host=hpc/p/projects/climate_data_central/reanalysis/ERA5`: one file
per variable and month, `<variable>/<variable>_YYYYMM.nc`, 1940-01 to 2026-01, global 0.25°
(1440 × 721), hourly, float32. The fields used are `10m_u_component_of_wind`,
`10m_v_component_of_wind`, `2m_temperature` and `surface_solar_radiation_downwards`; 100 m
wind and direct or top-of-atmosphere radiation are not on the mount. The four fields take
about 57 GB per year, and the mount copies at about 4 MB/s, about 4 hours per year.

**Retrieval.** `scripts/fetch_era5.sh <years>` copies one global monthly file at a time, cuts
it to the box lon −10.5…25.5, lat 40.5…59.5 (145 × 77 cells, covering every onshore and
offshore region of the 10 countries) with `scripts/era5_subset.py`, and deletes the global
copy. It is sequential, niced, resumable (finished months are skipped), never lists or
processes files on the mount, and stops at a local disk budget. The cut-outs (about 0.6 GB per
year for the four fields) are in `data/raw/era5/europe/<variable>/`, the log in
`data/raw/era5/fetch.log`. `scripts/after_fetch_era5.sh` waits for a clean end of the fetch and
then rebuilds the model file. The planned years are 2013 (the snapshot year of the PyPSA-Eur
build, needed for the calibration), 2003 (hot summer), 2010 (cold, calm winter) and 2017
(January low-wind period); all 22 years would take about 90 hours of transfer and 13 GB
locally.

**Processing** (`scripts/prepare_era5.py`). Everything except the weather is copied from
`model256.h5`; the script checks that nodes, generator rows and nodal mean loads agree with a
fresh extraction of `composed_2050.nc`. Per bus:

- **renewables**: the capacity factor per ERA5 cell, averaged over the cells of the bus with the
  PyPSA-Eur availability matrices as weights (eligible area per 0.3° cutout cell and bus, times
  $\cos$ latitude, remapped conservatively to the 0.25° grid; offwind = ac + dc + floating);
  wind from the 10 m speed $|u_{10},v_{10}|$ extrapolated to 100 m with exponent 1/7 and the same
  power curve; PV from $\mathrm{GHI} = \mathrm{ssrd}/3600$ with the same model;
- **calibration**: the scale per generator as for POWER (§2, `smallest_root`), so the 2013 mean
  equals the PyPSA-Eur target;
- **temperature** for the load model: the population-weighted mean over the cells whose centre
  lies in the bus's onshore region (at least one cell per region);
- **time**: ssrd accumulates over the hour ending at its time stamp and is shifted to the hour's
  start; 29 February is dropped.

The calibrated scales are stored in `weather_calibration/<carrier>_scale` and the source in the
attributes `weather_source` and `weather_base`.

**Scope.** The box covers the 10-country region of the PyPSA-Eur build, and the processing maps
the cells onto its 256 buses, so ERA5 exists for `model256.h5` only (`model256_era5.h5`); the
40- and 76-node files keep NASA POWER. Larger networks need a new build and a wider box.

**On the cluster** the cut-outs are made on a compute node from the ERA5 folder in place
(`hpc/prepare_data.sh era5 [years]`, `hpc/jobs/era5.sbatch`, after the `pypsa256` build;
[hpc.md](hpc.md) §4, §8.3), with the same `era5_subset.py` and `prepare_era5.py`; the four
default years take about 8 minutes. `prepare_era5.py` also writes
`model256_era5_comparison.csv`, ERA5 against the base file per carrier and node.

**The built file** (cluster, 2003, 2010, 2013, 2017; 117 MB): no NaN in the cut-outs, every
area weight inside the box, population cells in every region; every scale away from the bounds
and every generator within 0.005 of its 2013 target (median scales onwind 1.29, offwind 1.02,
solar 0.96). Against the NASA POWER file `model256.h5`: solar (hourly correlation 0.995, the
same mean) and load (0.999) agree; onwind correlates at 0.90 and varies more (aggregate
coefficient of variation 0.66 against 0.51; 10 m wind extrapolated to 100 m); offwind differs
(mean 0.418 against 0.285, hourly correlation −0.76 in aggregate), which is the calibration
defect of the POWER file (§2), not ERA5.

> **Status: missing.** The comparison against POWER files rebuilt with the corrected
> calibration, and of extremes per node, has not been made; `configs/full256_era5.toml` has not
> been run.

## 5. Provenance of the model files

| file | network | weather | years | build |
|---|---|---|---|---|
| `model.h5` | example network, 10 countries | NASA POWER, calibrated on 1–21 May 2013 | 2001–2022 | `download_data.sh`, `prepare_data.py`, `--augment` |
| `model76.h5` | example network, 27 countries | NASA POWER, as above | 2001–2022 | `prepare_data.py --countries AT,BE,BG,CH,CZ,DE,DK,EE,ES,FI,FR,GR,HR,HU,IE,IT,LT,LU,LV,NL,NO,PL,PT,RO,SE,SI,SK` (the example network's 34 countries without AL, BA, GB, ME, MK, RS and the pseudo-country EU; the list is the file's `countries` attribute) |
| `model256.h5` | PyPSA-Eur build `esm256` | NASA POWER, calibrated on 2013 | 2001–2022 | `prepare_data.py --network composed_2050.nc` |
| `model256_pypsa.h5` | as `model256.h5` | the network's own series (PyPSA-Eur's atlite profiles and loads), not calibrated: renewable rows are the `p_nom_max`-weighted mean `p_max_pu` of the PyPSA generators they merge (solar with solar-hsat, the three offwind kinds), nodal load the sum of the node's loads | 2013 (one year, the build's snapshots) | `prepare_data.py --network composed_2050.nc --pypsa-series` (no downloads; the build checks the written series against the network); on the cluster `hpc/prepare_data.sh pypsa256ts` |
| `model256_era5.h5` | as `model256.h5` | ERA5, calibrated on 2013 | the years of the build (on the cluster 2003, 2010, 2013, 2017) | locally `fetch_era5.sh` (copies over the sftp mount), `prepare_era5.py`; on the cluster `hpc/prepare_data.sh era5`, reading the ERA5 folder in place |

On the cluster the same builds are scripted (`hpc/prepare_data.sh`, [hpc.md](hpc.md) §4), with
the PyPSA-Eur environment recreated from `configs/pypsa-eur/requirements.txt` (frozen from the
local build), and each file recorded in `data/processed/PROVENANCE.toml`.

Common to all: CO₂ price 150 €/t in the marginal costs (attribute `co2_price`), battery 6 h and
H₂ 168 h energy-to-power ratios, constant availabilities of nuclear 0.9 and run-of-river 0.5,
reservoir inflow 0.3 of rated power, fixed thermal fleets (where present) at availability
0.9, gas and storage power bounded by 1.5
times the nodal peak load, DC link efficiencies $0.98\cdot0.977^{\text{length}/1000\,\text{km}}$
(the networks' links are lossless), line loss coefficients from resistance and nominal voltage.
