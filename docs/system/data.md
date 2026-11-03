# Data and outputs

This document gives the layout of the model input files (`data/processed/*.h5`), the
available input files, and the layout of the HDF5 outputs of each driver. The pipeline that
builds the inputs is in [model.md](model.md) §5, the run configuration in
[config.md](config.md).

## 1. Input files

| file | network | built by |
|---|---|---|
| `model.h5` | 40 nodes, 79 lines, 8 DC links: the default 10-country region of the PyPSA example network `carbon_management` (2050 net zero, electricity layer); NASA POWER weather 2001–2022; no brownfield fleets | `prepare_data.py` |
| `model76.h5` | 76 nodes, 130 lines, 27 DC links: 27 countries of the same example network | `prepare_data.py --countries …` |
| `model256.h5` | 256 nodes, 480 lines, 4 DC links: the PyPSA-Eur workflow (`external/pypsa-eur`, OSM base network clustered to 256 nodes, electricity only) for the 10-country region; brownfield fleets; NASA POWER weather | `prepare_data.py --network …` |
| `model256_era5.h5` | the 256-node network with ERA5 weather for selected years | `prepare_era5.py` |
| `backup/` | backups of the input files without the model-completion datasets | — |

## 2. Input layout

Written by h5py in row-major order; Julia reads the axes reversed. Node and unit indices are
0-based in the file (`load_system` adds 1).

| path | content |
|---|---|
| attributes | `source_network`, `countries`, `years` (weather years), `co2_price` (€/t contained in the marginal costs) |
| `network/nodes`, `country`, `x`, `y`, `mean_load` | buses |
| `network/line_bus0`, `line_bus1`, `line_b`, `line_s` | merged AC lines: ends, susceptance (normalized to mean 1), rating (MW) |
| `network/line_alpha`, `line_length` | quadratic loss coefficient (1/MW) and length |
| `network/link_bus0`, `link_bus1`, `link_cap`, `link_eta`, `link_length` | DC links |
| `generators/carrier`, `bus`, `invest`, `kbar`, `k`, `capex`, `mc`, `avail` | one unit per (node, carrier): investable flag, potential, fixed capacity, annualized capex (€/MW/a), marginal cost (€/MWh, including CO₂), constant availability |
| `generators/k0`, `mc0`, `ef`, `ef0` | existing capacity, old-vintage marginal cost, emission factors (t/MWh) of new and old vintage |
| `storage/carrier`, `bus`, `invest`, `kbar`, `k`, `capex`, `mc`, `hours`, `eta_in`, `eta_out`, `inflow` | one unit per (node, carrier); capex includes the energy part; `inflow` per unit power |
| `brownfield/generators/…` | fixed brownfield fleets (same columns as `generators`), used with `brownfield = true` |
| `weather/<field>/nodes`, `weather/<field>/data` | per field (`load`, `onwind`, `offwind`, `solar`): node indices and a Float32 tensor (year, hour, node) in the file, i.e. `data[node, hour, year]` in Julia, 8760 hours per year |

## 3. Outputs

Matrices of traces are stored quantity × sample in Julia (sample × quantity when read with
h5py). `quantity_names` names the rows.

**CPU driver and device runs** (`chains_report`):

| path | content |
|---|---|
| `quantity_names` | monitored quantities ([sampler.md](sampler.md) §9) |
| `chains/<c>/Q`, `chains/<c>/start` | trace and start kind of chain `c` |
| `convergence/rhat`, `ess_bulk`, `ess_tail`, `mcse_rel`, `mean`, `sd`, `pass` | per quantity |
| `convergence/warmup_detected`, `window_start`, `converged`, `thresholds` | run-level verdict and the thresholds used |
| `post/weights`, `post/ess`, `post/Q_weighted` | bias post-correction ($\alpha = 1$) |

**CPU driver only** (`write_run_outputs`, appended to the same file):

| path | content |
|---|---|
| `prices/sampled`, `prices/lp` | sampled prices per node and finest step over all chains (€/MWh); LP duals when the reference LP (`init = "lp"`) is at the finest level |
| `prices/regions/<country\|system>/{mean,load_weighted,p_shed,p_dump}/{mean,se,trace}`, `regions`, `seasons` | sampled prices and shortage/surplus frequencies per region and meteorological season |
| `bias/theta`, `bias/a`, `bias/c`, `bias/online_updates` | bias coefficients (after the post-correction refit when `bias_post`) and the observation counts of the online refits |
| `co2/price`, `co2/price_trace` | tuned CO₂ price and its warmup trace (`co2_cap > 0`) |
| `adaptive/chain<c>/level<ℓ>/updates`, `transport_modes`, `shapes/<kind>_L<L>_season<s>`, `response/{nodes,beta_deficit,beta_surplus,units_per_node}` | learned quantities of the adaptive kernel per chain (season `s`: DJF, MAM, JJA, SON) |
| `weather/transition`, `weather/em_loglik` | fitted regime transition matrix and EM trace |
| `calibration/beta`, `lp_objective`, `lp_capacity` | temperatures $(\beta_k,\beta_v)$ and the reference LP |
| `config/levels` | the levels |

## 4. Caches

`<dirname(out)>/cache/weather_<hash>.jls` (fitted weather model) and
`<dirname(out)>/cache/lp_<hash>.h5` (reference LP solution); see
[architecture.md](architecture.md) §5 for the keys. Deleting them forces a refit or re-solve.
