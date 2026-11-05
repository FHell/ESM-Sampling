# Data and outputs

This document gives the layout of the model input files (`data/processed/*.h5`), the
available input files, and the layout of the HDF5 outputs of each driver. The pipeline that
builds the inputs is in [model.md](model.md) §5, the provenance of the sources in
[data_sources.md](data_sources.md), the run configuration in [config.md](config.md).

## 1. Input files

| file | network | built by |
|---|---|---|
| `model.h5` | 40 nodes, 79 lines, 8 DC links: the default 10-country region of the PyPSA example network `carbon_management` (2050 net zero, electricity layer); NASA POWER weather 2001–2022; no brownfield fleets | `prepare_data.py` |
| `model76.h5` | 76 nodes, 130 lines, 27 DC links: 27 countries of the same example network | `prepare_data.py --countries …` |
| `model256.h5` | 256 nodes, 480 lines, 4 DC links: the PyPSA-Eur workflow (`external/pypsa-eur`, OSM base network clustered to 256 nodes, electricity only) for the 10-country region; brownfield fleets; NASA POWER weather | `prepare_data.py --network …` |
| `model256_era5.h5` | the 256-node network with ERA5 weather for selected years | `prepare_era5.py` |
| `model256_pypsa.h5` | the 256-node network with the network's own time series as the weather: one year (2013), the renewable `p_max_pu` per generator row and the nodal loads of `composed_2050.nc`; the same nodes, lines, links, generator and storage rows as `model256.h5` (the dispatchable boxes `kbar` follow its own peak load) | `prepare_data.py --network … --pypsa-series` |
| `backup/` | backups of the input files without the model-completion datasets | — |

## 2. Input layout

Written by h5py in row-major order; Julia reads the axes reversed. Node and unit indices are
0-based in the file (`load_system` adds 1).

| path | content |
|---|---|
| attributes | `source_network`, `countries`, `years` (weather years), `co2_price` (€/t contained in the marginal costs), `weather_source` (`NASA POWER and the OPSD load model` or `network time series`) |
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
| `convergence/post_weight_ess_ok` | 1 if the post-correction weight ESS is at least 30 % of the samples (or there is no post-correction) |
| `post/weights`, `post/ess`, `post/Q_weighted` | bias post-correction ($\alpha = 1$) |
| `post/bias_used`, `post/bias_refit` | $\hat b$ in use and the refitted $\tilde b$ at every production sample (the log weights are their difference) |

**Device runs** (appended by `run_device`): `device_profile/<part>` (seconds per host
checkpoint (`mh`) or per step (`replicas`) of every timed part, `rebuild` the residual rebuild) and
`device_profile/compile_<program>` (compile seconds); Metropolis driver: `run/restarts`,
`run/resumed_at_checkpoint` (host checkpoint of the last resume, −1: none) and
`device/residual_drift` (chain × host checkpoint: max |r_cache − r_rebuilt| in MW before the rebuild,
[device.md](device.md) §2); replica driver: the outputs of the CPU replica driver (`replicas/*`,
`run/*`; the replica records at the host checkpoints only; the probes those of the device probe,
`replicas/<c>/probe/it` with the step of the copy, [device.md](device.md) §6.2),
`replicas/<c>/residual_drift` (replica × host checkpoint, as above), `replicas/record_every`
(= `device_sweeps`) and `replicas/probe_slots` (probe slots per chain on the device, 0 or 1).

**CPU driver only** (`write_run_outputs`, appended to the same file):

| path | content |
|---|---|
| `prices/sampled`, `prices/lp` | sampled prices per node and finest step over all chains (€/MWh); LP duals when the reference LP (`init = "lp"`) is at the finest level |
| `prices/regions/<country\|system>/{mean,load_weighted,p_shed,p_dump}/{mean,se,trace}`, `regions`, `seasons` | sampled prices and shortage/surplus frequencies per region and meteorological season |
| `bias/theta`, `bias/a`, `bias/c`, `bias/online_updates` | bias coefficients (after the post-correction refit when `bias_post`; `a`, `c` per capacity coefficient of the bias level) and the observation counts of the online refits |
| `bias/level`, `bias/Dbar`, `bias/sample_steps` | level of the bias (0-based; the finest after the lift), the load normalization $\bar D$, the number of sample steps of $\bar{\mathbb E}$ |
| `co2/price`, `co2/price_trace` | tuned CO₂ price and its warmup trace (`co2_cap > 0`) |
| `adaptive/chain<c>/level<ℓ>/updates`, `transport_modes`, `shapes/<kind>_L<L>_season<s>`, `response/{nodes,beta_deficit,beta_surplus,units_per_node}` | learned quantities of the adaptive kernel per chain (season `s`: DJF, MAM, JJA, SON) |
| `weather/transition`, `weather/em_loglik` | fitted regime transition matrix and EM trace |
| `calibration/beta`, `lp_objective`, `lp_capacity` | temperatures $(\beta_k,\beta_v)$ and the reference LP |
| `config/levels` | the levels |
| `run/restarts`, `run/resumed_at_sweep` | resumes from a snapshot (0 for a straight run) and the fine sweep counter of the last one (−1: none, or resumed in the level-0 warmup) |

**Replica driver** (`sampler = "replicas"`, `run_replicas`): `chains_report` as above on the
replica mean of the monitored quantities per step (no `post/*`), then (`write_replica_outputs`):

| path | content |
|---|---|
| `replicas/carriers`, `replicas/feature_names`, `replicas/M`, `replicas/kappa_s` | carrier names of the k fields, names of the replica weather features, replicas per chain, $\kappa_s$ per k field |
| `replicas/tilt` | $(\lambda, \mu_\Phi, \sigma_\Phi)$ (NaN without a declared $\Phi$) |
| `replicas/<c>/ktot` | carrier totals of the shared capacities per step (MW; k field × step) |
| `replicas/<c>/capex`, `op`, `shed`, `penalty`, `pfcost` | capex per step; per replica (replica × step) opex, shed (MWh), penalty and penalty-free cost (capex + generation + storage) |
| `replicas/<c>/weights`, `active`, `since`, `phi`, `moved` | per replica and step: drift weight $\omega_j$, active flag, steps since its last burst, $\Phi(x_j)$, whether its weather moved |
| `replicas/<c>/rs`, `fk` | per k field and step: interior-node means of the drift score $r\bar s_k$ and of $\beta_k\partial_kK$ |
| `replicas/<c>/eta_w`, `nclip`, `time` | Langevin step per k field, clipped drifts, (wall time, time of the replica loop) per step |
| `replicas/<c>/features` | replica weather features per step, stacked replica by replica (feature × replica, step): annual mean load, capacity factors, residual load at the start capacities (mean, peak step, worst 7-step mean; GW), regime frequencies, regime switches |
| `replicas/<c>/probe/it` | per probe: (step, replica, adapted 0/1) |
| `replicas/<c>/probe/gap`, `rs_eq`, `rs_in`, `fk` | per k field and probe: gap/$\beta_k\partial K$, equilibrated and in-flight $r\hat s_k$, $\beta_k\partial K$ (interior means) |
| `replicas/<c>/probe/cost`, `eta_before`, `eta_after` | (opex, opex at equilibrium, shed, shed at equilibrium) of the probed replica; the step before and after the probe |
| `weather/transition`, `calibration/beta`, `config/levels`, `run/restarts`, `run/resumed_at_sweep` | as for the chains driver (the step counter of the last resume) |

`summary_replicas.txt` next to `out` (collected by `hpc/collect.sh`): over the production
steps per chain the carrier totals (mean, sd, last), the last Langevin steps, the drift and the
probes' equilibrated scores and gaps in units of $\beta_k\partial K$, capex, penalty-free cost,
opex, shed, the mean number of active replicas and the replica weather features (mean, sd).

## 4. Caches

`<cache>/weather_<hash>.jls` (fitted weather model, Julia Serialization) and
`<cache>/lp_<hash>.h5` (reference LP solution), where `<cache>` is `ESM_CACHE_DIR`, else
`cache_dir`, else `<dirname(out)>/cache` ([config.md](config.md)); see
[architecture.md](architecture.md) §5 for the keys. Deleting them forces a refit or re-solve.
A miss is first looked up in the mirror (`ESM_CACHE_MIRROR`, `cache_mirror`) and copied in.
Entries are written as `.<name>.tmp-<pid>-<random>` in the same directory and renamed into
place; leftover temporary files are from killed writers and can be deleted.

## 5. State snapshots

The file `checkpoint` of the CPU and device drivers ([hpc.md](hpc.md) §7), HDF5, replaced atomically at
every write. Vectors are the coefficient vectors of the state at its level (the layouts of
[model.md](model.md) §2).

| path | content |
|---|---|
| `version` | layout version (1) |
| `phase`, `it`, `where` | `"level0"` or `"fine"`; sweeps completed in the phase (fine: above `mlda_warmup` is production); a readable description |
| `restarts`, `written` | resumes before the run that wrote it; Unix time of the write |
| `co2_price` | the CO₂ price μ at the time (tuned during the fine warmup with `co2_cap > 0`) |
| `config/seed`, `config/levels` | checked on resume |
| `nchains` | number of saved chain states (1 in the level-0 warmup) |
| `chains/<c>/level`, `m`, `z0`, `w/<j>`, `k`, `v` | the state: level ℓ, regimes, level-0 weather, innovations of levels 1…ℓ, capacity and operation coefficients |
| `chains/<c>/logtarget` | its log target at the time (for checks; recomputed on resume) |
| `chains/<c>/start`, `chains/<c>/Q` | fine phase: start kind and the monitored trace so far (quantity × sweep) |
| `co2/price_trace` | the CO₂ price trace of the fine warmup so far |
| `sampler/sigma_scale`, `sampler/gibbs/{wv,wz,wk}` | level-0 phase: carrier-scaling step sizes and slice widths |
| `device/driver` | device runs of the Metropolis driver (`device_sampler = "mh"`): `"mh"`; `it` counts host checkpoints |
| `device/residual_drift` | the same runs: the residual drift so far (chain × host checkpoint), continued after a resume |

**Replica driver** (`write_replica_snapshot`, also of the device replica driver, so either
driver resumes from the other's snapshots; `read_snapshot` of the chains driver refuses it):

| path | content |
|---|---|
| `version`, `driver` | layout version (1), `"replicas"` |
| `phase`, `it`, `where`, `restarts`, `written`, `co2_price`, `config/seed`, `config/levels`, `nchains` | as above; `phase = "replicas"`, `it` the replica steps completed (above `mlda_warmup` is production; 0: after the initialisation) |
| `chains/<c>/start`, `replicas`, `Q` | start kind, number of replicas, the monitored trace so far |
| `chains/<c>/eta_w`, `since`, `window`, `active` | Langevin step per k field; per replica steps since its last burst, whether it had a regime window, active flag |
| `chains/<c>/trace/<key>`, `chains/<c>/probe/<key>` | the replica and probe records so far (the keys of `replicas/<c>/...` above) |
| `chains/<c>/replica/<j>/level`, `m`, `z0`, `w/<i>`, `k`, `v` | the state of replica j |

## 6. LP baselines (campaigns H2, H2b)

Small text files in the run directory (`dirname(out)`), collected by `hpc/collect.sh`
([validation.md](validation.md) §2.7). Costs in €/a, capacities in MW (storage: power), energies
in MWh/a.

**`scripts/lp_baseline.jl`** (our LP on historical years; `scripts/lp_reference.jl` writes the
older single-LP `summary_lp.toml` of the sampled-weather reference LP in the same place):

| file | content |
|---|---|
| `summary_lp.toml` | `years`, `weights`, `status`; `objective_eur_per_a` (capex + Σ w opex), `capex_eur_per_a`, `opex_eur_per_a`, `opex_by_year_eur`; `sampler_cost_eur_per_a` (objective + the cycling cost: about the sampler's cost at the LP solution, validation.md §2.7); `mean_stats` (weighted means of the per-year statistics); `capacity.<k field>.total_MW` and `.by_country_MW`; free-coordinate counts; LP size (`lp_variables`, `lp_constraints`), `seconds_setup`, `seconds_optimum`, `maxrss_MB`, threads; `ignored_model_terms`; commit, run id, data hash |
| `years.csv` | per year: `year, weight, opex_eur, co2_t, curtail_MWh, curtail_cost_eur, storage_cycling_MWh, cycling_loss_MWh, cycling_cost_eur, dump_MWh, energy_MWh_<carrier>…, shed_MWh` (storage energy: discharge) |
| `capacity.csv` | `carrier, country, MW` at the optimum |
| `mga.csv` | one row per MGA solve: `eps, target, sense, status, attempts, cost_eur, cost_rel_excess, total_MW, seconds` (status `OPTIMAL`, `feasible:<status>` for a primal-feasible point within the bound, or a failure with NaN values) |
| `mga_capacity.csv` | `eps, target, sense, carrier, country, MW`: every carrier's capacity per country at each extreme |
| `summary_mga.toml` | `envelope.eps_<ε>.<k field>`: `min_MW`, `optimum_MW`, `max_MW`, `status`; total time and memory |

**`scripts/pypsa_baseline.py`** (PyPSA on its network): `summary_pypsa.toml` (resolution,
snapshots, `total_cost_eur_per_a` = capex + opex in PyPSA's statistics, `objective_eur_per_a`,
`fixed_capex_eur_per_a` of the existing fleet, `co2_price_eur_per_t` = the cap's dual,
`shed_MWh`, `energy_MWh` per carrier, `capacity.<group>.total_MW`, `.by_country_MW`, LP size,
times, memory), `capacity_pypsa.csv`, and with ε `mga_pypsa.csv`
(`eps, target, sense, status, cost_eur, cost_rel_excess, total_MW, seconds`),
`mga_capacity_pypsa.csv`, `summary_mga_pypsa.toml`. Groups: `onwind`, `solar`, `offwind`,
`gas`, `battery`, `H2` as in [validation.md](validation.md) §2.7.
