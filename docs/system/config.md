# Configuration reference

This document lists every key of `Config` (`src/run.jl`) with its default and meaning, and
states which values define the system as it is run. A configuration file is TOML with a
subset of these keys (`load_config`; unknown keys are an error, and `check_config` rejects
values the code does not support). Examples are in `configs/`. The model features behind the
keys are explained in [model.md](model.md), the sampler options in [sampler.md](sampler.md).

## 1. System settings

The defaults are the values of the system:

| key | system value | default |
|---|---|---|
| `basis` | `"haar"` or `"flow"` | `"haar"` |
| `storage` | `"soc"` (the only form) | `"soc"` |
| `renewables` | `"load"` (the only form) | `"load"` |
| `init` | `"heuristic"` (large networks) or `"lp"` (small) | `"lp"` (the default data file is the 40-node model) |
| `measure` | open design choice; the configurations use `"physical"` | `"utilization"` (kept until the choice is made) |
| `nchains` | several chains (the configurations use 4) | 4 |
| `bias_online` | refits during the finest-level warmup, frozen afterwards | `true` |
| `sampler` | `"replicas"` (the replica–Langevin sampler, [sampler.md](sampler.md) §11) | `"chains"` (kept until the chains driver and the bias are removed) |

All constraints are priced (there are no hard walls). In the chains driver the fine-level
transition is `fine_local_step`, and the bias is refitted during the warmup only and frozen for
production; the replica driver uses the operation sweep of the same fine kernel and no bias.

System configurations: `configs/complete40.toml` (all model terms, adaptive kernel, α = 1);
the replica-sampler configurations of campaign H4 (`campaigns/H4/*.toml`: 40 nodes daily and
6-hourly, `sampler = "replicas"`, no bias).
`configs/chains40.toml` and `configs/device40.toml` are **mechanics tests only**: they combine
α = 0, the physical measure and capacity scaling moves, in which combination capacities can
run away; they exercise the drivers and diagnostics and are not physically meaningful runs.
`configs/haar_poc.toml` and `configs/flow.toml` are the base configurations of the tests and
benchmarks; `configs/flow256.toml` and `configs/full256_era5.toml` are smoke configurations of
the 256-node network.

## 2. All keys

**Files and run.**

| key | default | meaning |
|---|---|---|
| `data` | `"data/processed/model.h5"` | model input file ([data.md](data.md)) |
| `out` | `"results/run.h5"` | output file |
| `cache_dir` | `""` | directory of the weather-model and LP caches; `""`: `<dirname(out)>/cache/`; the environment variable `ESM_CACHE_DIR`, if set and non-empty, overrides it ([hpc.md](hpc.md) §4) |
| `cache_mirror` | `""` | read-only fallback of the caches: an entry missing in the cache but present here is copied in instead of recomputed; `""`: none; overridden by `ESM_CACHE_MIRROR` |
| `checkpoint` | `""` | CPU and device drivers: the state snapshot file, written at sweep boundaries and resumed from if it exists at start ([hpc.md](hpc.md) §7, layout in [data.md](data.md) §5); `""`: off |
| `checkpoint_minutes` | 30.0 | ... a snapshot every this many minutes of wall time (and once after the level-0 warmup) |
| `stop_file` | `""` | ... if this file exists at a sweep boundary: write a snapshot, remove the file, exit with code 75 |
| `seed` | 1 | master seed (chains use derived seeds; after the r-th resume, re-seeded from seed and r) |
| `verbose` | `true` | progress messages |

**Environment variables** (they are not keys; [hpc.md](hpc.md) §3–§7):

| variable | effect |
|---|---|
| `ESM_CACHE_DIR` | overrides `cache_dir` when set and non-empty |
| `ESM_CACHE_MIRROR` | overrides `cache_mirror` when set and non-empty |
| `ESM_LP_THREADS` | HiGHS's `threads` option in every LP (`solve_lp`, the LP baselines, `scripts/pypsa_baseline.py`); unset: HiGHS's default. HiGHS's global thread scheduler is reset when it is set (otherwise a `threads` different from HiGHS's default makes the solve fail) |
| `ESM_IGNORE_SIGTERM` | `1`: `scripts/run.jl` and `scripts/lp_baseline.jl` ignore SIGTERM (the job wrapper stops the run through the stop file) |
| `ESM_PYPSA_EUR` | the PyPSA-Eur clone with its build (default `$ESM_PROJ/external/pypsa-eur`); `${ESM_PYPSA_EUR}` in the `network` key of an H2b configuration |

**LP baselines.** `scripts/lp_baseline.jl` reads the usual keys (`data`, `levels`: its LP is at the reference level, the first spatially complete level; the model terms; `out`: its outputs go to `dirname(out)`); the cases are arguments `years=`, `eps=`, `carriers=`, `time_limit=` ([validation.md](validation.md) §2.7). `scripts/pypsa_baseline.py` reads its own TOML keys (`network`, `resolution`, `co2`, `load_shedding`, `line_expansion`, `snapshots`, `out`), documented in the script and in `campaigns/H2b/configs/`.

**Hierarchy and bases.**

| key | default | meaning |
|---|---|---|
| `levels` | `[(10, 24), (40, 24)]` | levels $(M_\ell, h_\ell)$, coarse to fine ([model.md](model.md) §2.3); the default suits the 40-node data (countries daily → nodes daily) |
| `basis` | `"haar"` | `"haar"` (graph Haar) or `"flow"` (Haar plus transport modes) |
| `flow_theta` | `"nested"` | `basis = "flow"`: modes as the coarse angle coordinates (`"nested"`) or only as moves (`"haar"`, large networks) |

**Model.**

| key | default | meaning |
|---|---|---|
| `c_shed`, `c_dump` | 3000, 5 | balance prices (€/MWh) |
| `c_viol` | 3000 | price of constraint violations (€/MWh per unit of excess) |
| `measure` | `"utilization"` | reference measure of dispatchable units: `"utilization"` or `"physical"` |
| `storage` | `"soc"` | storage form; only `"soc"` (state-of-charge trajectory) exists |
| `renewables` | `"load"` | renewables as load (output $Ak$, curtailment as dump); the only form |
| `co2_price` | NaN | canonical CO₂ price (€/t); NaN uses the price contained in the data file |
| `co2_cap` | 0 | > 0: tune the CO₂ price during the finest-level warmup to this mean annual emission (t/a), frozen for production; CPU driver (on the mean over chains) |
| `brownfield` | `false` | existing capacities as floors, capex on $k-k_0$, merged vintages, fixed thermal fleets |
| `losses`, `loss_segments` | `false`, 4 | quadratic line losses; interpolation cells per rating |
| `thermal`, `thermal_band` | `"hard"`, 0.1 | `"soft"` thermal limits; band width as a fraction of the rating |
| `link_losses` | `false` | lossy DC links |
| `uc` | `[]` | carriers with fleet unit commitment |

**Weather model.**

| key | default | meaning |
|---|---|---|
| `K` | 4 | number of weather regimes |

**Temperatures.**

| key | default | meaning |
|---|---|---|
| `beta_v`, `beta_k` | 4e-5, 4e-8 | operational and investment temperatures (1/€) |
| `calibrate_beta` | `false` | derive β from the reference LP by equipartition instead |
| `eps_v`, `eps_k` | 0.01, 0.05 | target slacks of the calibration (fractions of the optimal cost) |

**Start.**

| key | default | meaning |
|---|---|---|
| `init` | `"lp"` | `"lp"` (reference LP at the first complete level) or `"heuristic"` (no LP) |
| `descent_iters` | 4000 | descent of the level-0 operation before the level-0 warmup (0: off) |

**Fine-level kernel.**

| key | default | meaning |
|---|---|---|
| `fine_kernel` | `"auto"` | `"local"`, `"flow"`, `"adaptive"`; `"auto"` is `"flow"` with a flow basis, else `"local"` |
| `local_extra` | 4.0 | random moves per node and step and sweep, on top of the atoms |
| `cell_level_power` | 1.0 | cell moves: partition level drawn $\propto$ (number of cells)$^p$ |
| `xloc_partner_hops` | 1 | localized pairs: partner within this many hops |
| `xloc_ball_hops` | 0 | localized pairs: re-routing ball radius (0: the pair only) |
| `kop` | `false` | capacity moves with their operation ([moves/capacity.md](moves/capacity.md)) in every fine step, after `kscale` and `kpeak` |
| `storage_long` | 1.0 | probability per operation sweep of the long-range storage sweep (`slong`, `slevel`; [moves/storage.md](moves/storage.md)), every kernel |
| `storage_long_moves` | 32 | ... `slong` moves per storage unit and sweep |
| `storage_long_hours` | 24.0 | ... only for storage units with at least this many hours of storage (batteries and small PHS are box-limited); also the units of the storage pairs |
| `storage_xpair` | 0.1 | rate of the single-instant storage pair `sxpair` in the local sweep, on the scale of the plain families (the other families keep their absolute rates; about 60 moves per unit and sweep at 40 nodes daily); every kernel ([moves/storage.md](moves/storage.md)) |
| `storage_xlong` | 8 | interval storage pairs `sxlong` per storage unit in the long-range storage sweep |
| `storage_xpair_hops` | 1 | storage pairs: partner units at the unit's own node or at nodes within this many hops |
| `storage_xpair_ball` | 1 | ... flow re-routed on the nodes within this many hops of the pair (the leak beyond is compensated pro rata) |
| `fine_weather` | 0.0 | fraction of the weather coordinates moved per fine step (0: weather fixed at the fine level) |
| `fine_weather_blocks` | 1.0 | with `fine_weather > 0`: joint $z_0$ block moves (16, 64, 365 level-0 steps) per fine step, in expectation ([moves/weather.md](moves/weather.md)) |
| `fine_weather_block_steps` | 600 | ... only of the lengths whose window spans at most this many finest-level steps: all three daily, 16 and 64 at 3 h and 6 h, 16 at 1 h |

**Bias.**

| key | default | meaning |
|---|---|---|
| `use_bias` | `true` | $\alpha = 1$ (learned bias) or $\alpha = 0$ |
| `bias_samples` | 16 | exact weather draws for $\bar{\mathbb E}\hat g$, at the level of the bias (redrawn at the finest level when the bias is lifted) |
| `bias_sample_steps` | 4096 | steps of those draws kept for $\bar{\mathbb E}\hat g$ (uniformly, weighted; 0: all) |
| `bias_table_res` | 100 | grid points per smoothing width $s$ of the tables of $F_1$, $F_2$ |
| `bias_monotone` | `true` | monotone, bounded implied price in the $\hat g$ fit |
| `bias_warmstart` | 20.0 | weight (in samples) of the LP-score warm start; `init = "lp"` only; 0: off |
| `bias_start`, `bias_every` | 150, 25 | level-0 warmup: first learning iteration, refit interval |
| `bias_online` | `true` | online refits from the pooled scores of the finest chains during the finest-level warmup; frozen afterwards (all drivers) |
| `bias_online_every` | 1 | observe every n-th finest step |
| `bias_online_min` | 16 | no online refit before this many observations (refits at $2^j\ge$ it); the lifted fit stays until then |
| `bias_trust` | 1.0 | trust region of the online refits: the change of $\hat b$ minus its mean over the last 32 observed states is at most this times the spread of $\hat b$ over them (at least the number of capacity coordinates); 0: off |
| `bias_post` | `true` | post-correction by reweighting at $\alpha = 1$ |

**Replica–Langevin sampler** (`sampler = "replicas"`, `run_replicas`, [sampler.md](sampler.md)
§11). The chain keys apply: `nchains` replica sets, starts cycled from `chain_starts`,
`mlda_warmup` and `nsamples` count replica steps; `fine_kernel`, `local_extra`, the storage keys
and the `xloc` keys define the operation sweep; `use_bias`, the bias keys, `kop` and the
fine-weather keys are not used.

| key | default | meaning |
|---|---|---|
| `sampler` | `"chains"` | CPU driver: `"chains"` (`run_chains`) or `"replicas"` (`run_replicas`) |
| `replicas` | 8 | weather replicas per chain, sharing the capacities |
| `replica_init_sweeps` | 40 | operation sweeps per replica at fixed $k$ after its first weather draw |
| `replica_inner` | 1 | operation sweeps per replica and step (scores averaged over them) |
| `replica_theta` | 0.0 | continuous pCN angle per step (0: off) |
| `replica_burst_every` | 40 | a pCN burst per replica every this many steps, staggered over the replicas (0: off) |
| `replica_burst_angle` | 0.3 | the burst angle $\varphi$ |
| `replica_window` | 60 | regime window length $B$ in level-0 steps (0: no window moves) |
| `replica_window_prob` | 0.5 | probability of a window move with a burst (continuous mode: per step) |
| `replica_ffbs` | `false` | centred FFBS of the regimes with every burst |
| `replica_compensate` | `true` | compensated initialisation of the dispatch after a weather step |
| `replica_gate_min`, `replica_gate_max` | 10, 30 | gate after a pCN burst: no weight before `min` steps, then once the relaxation indicator is below the tolerance, in any case from `max` steps on |
| `replica_gate_min_window`, `replica_gate_max_window` | 20, 60 | the same after a burst with a regime window |
| `replica_gate_win` | 8 | relaxation indicator: compares the halves of the last this many steps |
| `replica_gate_tol` | 0.3 | ... active below this difference of the normalised carrier scores (units of $\beta_k\partial K$) |
| `replica_probe_every` | 20 | lag probe every this many steps (0: off) |
| `replica_probe_len`, `replica_probe_tail` | 30, 15 | probe: operation sweeps at frozen $(k,x)$, of which the last `tail` give the equilibrated score |
| `replica_probe_win` | 10 | in-flight score: mean over this many active steps of the probed replica (a replica needs that many) |
| `replica_probe_avg` | 3 | probes averaged per step adaptation |
| `langevin_eta` | 5e-4 | initial Langevin step $\eta_w$ in $w=\log(k+\kappa_s)$, every k field |
| `langevin_eta_min`, `langevin_eta_max` | 2e-5, 2e-3 | bounds of the adapted step |
| `langevin_adapt` | `true` | adapt $\eta_w$ per k field from the probes (warmup only; ×0.7 / ×1.2) |
| `langevin_gap_hi`, `langevin_gap_lo` | 0.10, 0.05 | shrink above, grow below this mean $\lvert$gap$\rvert/\beta_k\partial K$ |
| `langevin_kappa` | `{}` | $\kappa_s$ per carrier (MW), a TOML table such as `{ gas = 300.0 }`; carriers not given: onwind 500, solar 1000, offwind 200, gas 300, battery 100, H2 100 |
| `langevin_kappa_default` | 100.0 | $\kappa_s$ of other carriers |
| `langevin_kmin` | 1.0 | reflection at $k = \kappa_{\min}$ (MW) |
| `langevin_clip` | 0.02 | drift clip per step in $w$ |
| `langevin_box` | `true` | reflection at the upper capacity bound |
| `tilt_functional` | `"resload_mean"` | declared weather functional $\Phi$ of the stress tilt ([ensemble.md](ensemble.md) §4.5); `"none"`: no $\Phi$ |
| `tilt_lambda` | 0.0 | tilt strength $\lambda$ (0: the untilted drift, bitwise) |
| `tilt_samples` | 200 | exact weather draws for the standardisation of $\Phi$ |

**Run lengths.**

| key | default | meaning |
|---|---|---|
| `warmup0` | 1500 | level-0 warmup iterations (bias learning) |
| `mlda_warmup` | 100 | finest-level warmup sweeps (all drivers; device: checkpoints) |
| `nsamples` | 300 | finest-level production sweeps (device: checkpoints) |

**Chains and convergence.**

| key | default | meaning |
|---|---|---|
| `nchains` | 4 | finest-level chains (CPU: `run_chains`; device: chains) |
| `chain_starts` | `["low", "high", "heuristic", "heuristic"]` | starts, cycled: capacities ×0.5, ×2, as set up |
| `rhat_max` | 1.01 | convergence: rank-normalized split-$\hat R$ below this |
| `ess_min` | 400 | bulk and tail ESS at least this |
| `mcse_max` | 0.05 | MCSE of the mean / posterior sd at most this |
| `rhat_screen` | 1.05 | threshold of the sliding-window warmup detection |

**Device.**

| key | default | meaning |
|---|---|---|
| `device_backend` | `"gpu"` | Reactant backend, `"gpu"` or `"cpu"` |
| `device_sweeps` | 10 | `mh`: operation sweeps per device call (one capacity pass and one host checkpoint after each); `replicas`: replica steps per host checkpoint (monitored quantities, replica records and snapshot boundary; `mlda_warmup` and `nsamples` must be multiples of it) |
| `device_sampler` | `"mh"` | capacity sampler of the device driver: `"mh"` (Metropolis capacity pass with the bias, [device.md](device.md) §5.1) or `"replicas"` (the replica–Langevin sampler with the batch on the device, §5.2, §6.2; its settings are the keys of the CPU sampler: `replicas`, `replica_*`, `langevin_*`, `tilt_*` above; the lag probe runs in a probe slot on the device, `replica_probe_len` and `replica_probe_tail` rounded up to whole steps of `replica_inner` sweeps) |
| `kmove_nprop` | 2 | Metropolis proposals per unit and capacity pass |
| `kmove_sigma` | 0.3 | scaling-move proposal scale (log capacity) |
| `kpeak_sigma` | 0.05 | peak-set proposal scale, fraction of the unit's start capacity (at least 100 MW) |

**Obsolete keys.** `kernel`, `p_fine`, `alpha0` and `regularized` are accepted with a warning
when they carry the value the code always uses (`"local"`, `1.0`, `false`, `true`), so that
older files still load; any other value is an error. Remove them from configuration files.
