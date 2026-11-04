# First full-resolution test: 256 nodes × hourly with ERA5 weather

Status: prepared, not run. Config `configs/full256_era5.toml`, data
`data/processed/model256_era5.h5` (built by `scripts/prepare_era5.py`).

@@STATUS@@

## 1. ERA5 on the PIK mount

`/run/user/1000/gvfs/sftp:host=hpc/p/projects/climate_data_central/reanalysis/ERA5`
(listed with 6 non-recursive `ls -l`; listings saved in `data/raw/era5/listing_*.txt`):

* layout `<long variable name>/<long variable name>_YYYYMM.nc`, 1940-01 … 2026-01;
* **global** 0.25° (1440 × 721, lon 0…359.75, lat 90…−90), hourly, NetCDF4 from
  cfgrib, float32, zlib level 1, chunks (124, 121, 240);
* available single-level fields relevant here: `10m_u_component_of_wind`,
  `10m_v_component_of_wind`, `2m_temperature`, `surface_solar_radiation_downwards`
  (plus dewpoint, pressure, precipitation, thermal radiation, …).
  **No 100 m wind, no direct/TOA radiation (`fdir`, `tisr`)**; the CHANGELOG says
  most other variables (rsdsdir, rsdt, sfcWind, …) were moved to the tape archive
  (ask Christoph Menz, PIK).
* sizes per month: t2m ≈ 1.1 GB, u10 and v10 ≈ 1.5 GB, ssrd ≈ 0.7 GB, i.e.
  **≈ 57 GB per year** for the four fields, ≈ 1.25 TB for 2001–2022.
* throughput of the gvfs sftp mount, one sequential `cp`: **≈ 3.9 MB/s**
  (1.1 GB in 284 s, 1.5 GB in 389 s), i.e. **≈ 4.1 h per year**.

Because the files are global and the transfer is slow, only four years are copied
for the first test: **2013** (the PyPSA-Eur snapshot year, needed for the
calibration), **2003** (hot summer), **2010** (cold, calm winter), **2017**
(January Dunkelflaute). Each global month is copied, cut to the box lon −10.5…25.5,
lat 40.5…59.5 (145 × 77 cells; covers all onshore and offshore regions of the 10
countries, the French Atlantic EEZ reaches −9.9°) and deleted
(`scripts/fetch_era5.sh` → `scripts/era5_subset.py`). The cut-outs are ≈ 13 MB per
variable and month (≈ 0.6 GB per year for all four), so local disk use stays below
≈ 1.6 GB plus one global file in `data/raw/era5/tmp`.

Local files: `data/raw/era5/europe/<variable>/<variable>_YYYYMM.nc`
(dims time, lat, lon increasing; variables t2m [K], u10, v10 [m/s], ssrd [J/m²]),
log `data/raw/era5/fetch.log`, remote README/CHANGELOG copies in `data/raw/era5/`.

Extending to more years (≈ 4 h per year, sequential, resumable, skips finished
months; stop with `kill $(cat data/raw/era5/fetch.pid)`):

    setsid nohup scripts/fetch_era5.sh 2001 2005 2008 2015 2019 2022 \
        > data/raw/era5/fetch.stdout 2>&1 &
    nice -n 19 .venv/bin/python -I scripts/prepare_era5.py --force

All 22 years would take ≈ 90 h of transfer (≈ 1.25 TB through the mount) but only
≈ 13 GB locally. Faster alternatives (cut-outs made on the HPC side, or reading
the box only) need someone on the cluster and were not tried.

## 2. Processing (`scripts/prepare_era5.py`)

* **Everything but the weather is copied unchanged from `model256.h5`**
  (network, generators, storage, brownfield fleets, attributes); only `weather/*`
  and the `years` attribute are replaced, plus attributes `weather_source`,
  `weather_base` and the group `weather_calibration/<carrier>_scale` (the
  calibrated scale per generator; ignored by `load_system`). Node lists of each
  field are identical to `model256.h5`. The script recomputes the network
  extraction of `prepare_data.py` from `composed_2050.nc` (only for the targets)
  and asserts that nodes, generator rows and nodal mean loads agree with the base
  file.
* **Spatial mapping (instead of one POWER point at the bus):**
  - onwind, offwind, solar: capacity factor per ERA5 cell, then the mean over
    the cells of the bus weighted by the PyPSA-Eur availability matrices
    (`availability_matrix_<tech>.nc`, eligible area per 0.3° cutout cell and bus,
    times cos(lat)), remapped conservatively to the 0.25° grid; offwind = ac + dc +
    float. Median 34 cells per onshore bus (11–108), 30 per offshore bus (4–307).
  - temperature for the load model: population weighted (`pop_layout_total.nc`)
    mean over the cells whose centre lies in the bus's onshore region
    (`onshore_regions.geojson`); every region has at least one cell (median 10).
* **Conversion as in `prepare_data.py`:** the same smoothed power curve at 100 m;
  **10 m wind** (|u10, v10|) extrapolated to 100 m with shear exponent 1/7 (the
  POWER file used WS50M); PV cf = clip(s·GHI/1000, 0, 1) with GHI = ssrd/3600;
  scale s per generator calibrated so that the 2013 mean equals the PyPSA-Eur
  `p_max_pu` mean; the same OPSD-fitted national load models (hour of week ×
  seasonal harmonics + heating/cooling degrees on the 24 h-EWM of temperature),
  driven by node temperature (per year, the EWM restarts on Jan 1) and scaled to
  the nodal mean load of 2013.
* **Calibration fix.** `prepare_data.calibrate` tests `f(4) < 0` before
  bracketing and returns s = 4 when the target is above the capacity factor at
  s = 4. For wind f is not monotone (cut-out at 25 m/s), so nodes with a large
  target land deep in the cut-out regime. In `model256.h5` this happened at
  most offwind nodes: the 2013 offwind mean is 0.282 against a target of 0.434
  (49 of 64 nodes off by more than 0.01, e.g. BE0 4: target 0.527, POWER 0.313
  at s = 4, the curve peaks at 0.725 near s = 2.1), and 30 of 256 onwind nodes are
  off by more than 0.01. `prepare_era5.py` scans s upwards on a grid and takes the
  smallest root (`calibrated_series`). `prepare_data.py` was not changed; the
  POWER files `model256.h5`, `model76.h5`, `model.h5` presumably carry the same
  bias (to be fixed in the main session, see §7).
* **Time conventions:** ssrd is the accumulation over the hour ending at
  valid_time; it is shifted to the hour's start (the diurnal maximum over Germany
  then sits at 11 UTC, as in the POWER file). Feb 29 dropped, 8760 h per year.

Run time ≈ @@RUNTIME@@ for @@NYEARS@@, peak memory ≈ @@MEM@@ (4 BLAS threads, niced).

## 3. ERA5 vs NASA POWER

@@TABLE@@

## 4. Config `configs/full256_era5.toml`

The options of `configs/flow256.toml` (flow basis, `flow_theta = "haar"`, local
kernel with the default `fine_kernel = "auto"` → flow moves, SoC storage,
renewables as load, heuristic start, no bias, no α = 0 run), plus
`measure = "physical"`, the model completion (`brownfield = true`,
`losses = true`, `thermal = "soft"`, `link_losses = true`,
`uc = ["gas", "coal", "lignite", "nuclear"]`; all keys exist in `Config`), and

    levels = [[10, 24], [256, 24], [256, 6], [256, 1]]
    J = [2, 2, 2]

`check_levels` passes (reference level 1 = 256 × daily); every h divides 8760
and the previous h. Sample counts are those of a smoke test: `warmup0 = n0 = 5`,
`mlda_warmup = 1`, `nsamples = 3`.

## 5. Resource estimate

Coordinates per level from `Layout(E.vspecs, M, h)` with the completion options
(measured, `build_energy_model` ≈ 20 s):

| level | T | k coords | v coords | per step | one Float64 state |
|---|---|---|---|---|---|
| 10 × 24 h | 365 | 56 | 48 180 | 132 | 0.4 MB |
| 256 × 24 h | 365 | 1 344 | 744 600 | 2 040 | 6 MB |
| 256 × 6 h | 1 460 | 1 344 | 2 978 400 | 2 040 | 24 MB |
| 256 × 1 h | 8 760 | 1 344 | 17 870 400 | 2 040 | 143 MB |

(Without the completion it is 1 280 per step at 256 nodes.) The weather state at
the finest level is (256 · 3 + 64) series × 8760 = 7.3·10⁶ values (58 MB in
Float64); the model file holds Y × 7.3·10⁶ Float32 (117 MB for 4 years, 641 MB
for 22).

**Time per sweep.** Measured anchors: 256 nodes × daily, local kernel, SoC +
renewables as load, 1 280 coordinates per step: 13.5 s per sweep on 4 niced
threads (`docs/reactant_performance.md` §1); flow sweeps cost 1.8–2.7× a local
sweep (`docs/setup_overview.md` §9); cost is linear in nodes × steps
(`docs/local_kernel.md` §5: 40 → 76 nodes 0.63 → 1.29 s). Scaling to 24× the
steps and 2 040/1 280 = 1.6× the coordinates per step:

* local sweep at 256 × 1 h: 13.5 s × 24 × 1.6 ≈ **8–9 min**;
* flow sweep: ≈ **15–25 min** on 4 threads (UC and loss segments may add to the
  per-move cost; not measured);
* 6 h level: ≈ 1/6 of that (3–4 min per flow sweep), daily level ≈ 1/24.

**The smoke run** (3 samples after 1 MLDA warmup step; J = 2 per level, p_fine =
0.7) needs ≈ 4–6 finest sweeps plus ≈ 10 sweeps at 6 h and ≈ 20 at daily, i.e.
roughly **1.5–3 h** of sampling, plus setup: weather model fit at hourly on 4
years (not measured at this size), the heuristic start and the construction of
the local kernel and flow families at T = 8760. Equilibration would need
≈ 25 sweeps (≈ 8–10 h) and is not the goal of this test.

**Memory.** A few tens of copies of the finest state (chain states per level,
proposals, gradients/caches, weather per level) at 143 + 58 MB each give
**≈ 3–8 GB**; the local kernel's per-step move tables scale with T and are the
main unknown. Watch the RSS during setup.

## 6. Commands

Smoke check of the data file (done, §3):

    nice -n 19 julia --project -t 4 -e 'using ESMSampling; sys = ESMSampling.load_system("data/processed/model256_era5.h5"); println(sys.N, " ", sys.years)'

Test run (not started; wait for the 4-year file, see status above):

    mkdir -p results
    nice -n 19 julia --project -t 4 scripts/run.jl configs/full256_era5.toml 2>&1 | tee results/full256_era5.log

## 7. Open issues

@@OPEN@@
