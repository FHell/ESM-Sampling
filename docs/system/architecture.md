# Architecture

This document maps the system onto the code: the packages and files, the main types, the
call graph of a run in each driver, the caches, threading and colouring, the outputs and
the tests. What the code computes is described in [ensemble.md](ensemble.md),
[model.md](model.md), [weather_model.md](weather_model.md) and [sampler.md](sampler.md); the
device port in [device.md](device.md).

## 1. Repository layout

| path | content |
|---|---|
| `src/` | package `ESMSampling`: model, weather, sampler, drivers |
| `device/` | package `ESMDevice` (depends on `ESMSampling` by path, and on Reactant 0.2.304): the GPU port |
| `test/`, `device/test/` | tests ([testing.md](testing.md)) |
| `configs/` | TOML run configurations ([config.md](config.md)) |
| `scripts/` | driver script (`run.jl`), data preparation (`download_data.sh`, `prepare_data.py`, the ERA5 scripts), benchmarks (`bench_perf.jl`, `prof_kernel.jl`, `regress_cmp.jl`, `xloc_eff.jl`, `xloc_leak.jl`) |
| `data/raw/`, `data/processed/` | downloaded sources and model input files ([data.md](data.md), provenance in [data_sources.md](data_sources.md)); not in git |
| `results/` | run outputs and the caches (`results/cache/`); not in git |
| `external/pypsa-eur/` | the PyPSA-Eur workflow used to build the 256-node network ([data_sources.md](data_sources.md)) |
| `external/reactant-bench/` | Reactant prototypes and benchmarks |
| `work/perf/` | a working copy used for CPU performance work |

## 2. Files of `src/`

In include order (`src/ESMSampling.jl`):

| file | content |
|---|---|
| `spatial.jl` | `ModeBasis`; graph Haar bases (`haar_basis`, `haar_field_basis`, weighted for capacities); identity basis of the links; component indicators; projections |
| `layout.jl` | `FieldSpec` (kinds `KFIELD`, `STEP`, `INSTANT`), `Layout`, `Transfer`, `split_state`/`merge_state`, `restrict_to`/`prolong_to` |
| `system.jl` | `System`, `load_system` (HDF5 input), `subsystem` (sub-networks for tests) |
| `flowbasis.jl` | `FlowBasis`: proxy capacities and injections, `CongestionADMM`, transport modes, angle basis, storage shapes |
| `energy.jl` | `GenGroup`, `StorGroup`, `LineTerms`, `EnergyModel`, `build_energy_model`, `energy_cost` (with gradients), `log_measure`, `set_co2_price!` |
| `lp.jl` | the reference capacity-expansion LP (JuMP/HiGHS), `LPSolution`, cache, `interior_center`, `to_modes` |
| `weather.jl` | `WeatherModel`, `DetailModel`, fit (EM, details), exact sampling, `weather_nodes`, cache; `season`, the one season definition (DJF, MAM, JJA, SON) |
| `bias.jl` | `BiasModel`: $\hat g$, $\hat h$, $\hat b$, score regression, monotone QP |
| `hierarchy.jl` | `Problem`, `State`, `logtarget`, `make_state`, `bias_term`, `level0_potential` (pre-descent) |
| `bias_online.jl` | `OnlineBias`, `online_bias_step!` |
| `bias_post.jl` | `PostBias`, `post_correct` |
| `level0.jl` | level-0 slice updates of the bias warmup (`GibbsState`, `SweepCtx`, `StepWS`, `step_term`, `update_colour!`, `update_k!`, `slice_coord!`) |
| `local.jl` | the local exact kernel: `GridSolver`, `LocalModel`, `LocalCost`, `MoveWS`, `eval_move!`, line sampler, move preparation, `local_sweep!`, `kscale_sweep!`, `LocalKernel`, `fine_local_step`, `local_base_step`, block-move workspaces and `commit_block!` |
| `weather_moves.jl` | level-0 weather move with pro-rata shear |
| `weather_fine.jl` | fine-level weather moves |
| `flowmoves.jl` | `FlowFrames`, transport families, storage multiscale sweep, `flow_sweep!`, localized modes |
| `adaptive.jl` | `AdaptiveModes`, learned families, `adaptive_sweep!`, `kpeak_sweep!` |
| `sampler.jl` | `ChainSampler` (the kernels of one chain), `quantities`, carrier scaling `scale_step`, `base_step` |
| `diagnostics.jl` | single-chain ESS (`ess`), `PriceAcc`, `RegionPriceAcc` |
| `convergence.jl` | multi-chain diagnostics (rank-normalized $\hat R$, bulk/tail ESS, MCSE, warmup detection) |
| `run.jl` | `Config`, `load_config`, `setup`, `heuristic_start`, the level-0 chain `run_level0!`, `predescend`, `new_sampler`, `run` |
| `chains.jl` | the CPU driver `run_chains`, `chain_start`, `monitor_quantities`, `chains_report`, `write_run_outputs` |

`device/src/`: `ESMDevice.jl` (device model, state, exact cost), `terms.jl` (term batch,
dense sampler), `rounds.jl` (plain families, schedule, device loop), `comp.jl` (compensated
families), `cells.jl` (cell templates), `kmove.jl` (`kscale`/`kpeak` Metropolis), `kren.jl`
(renewable capacity moves, peak sets), `bias_dev.jl` (device bias), `driver.jl`
(`run_device`).

## 3. Main types

**Model and target.**

- `System`: network, units, weather tensors and data attributes, read from the HDF5 file.
- `FieldSpec`, `ModeBasis`, `Layout`, `Transfer`: a field and its sparse basis (`U`, its
  transpose `Ut`, ranks per level, Haar cell labels), the coefficient layout of a level, and the
  transfer between consecutive levels.
- `EnergyModel`: the fields (`kspecs`, `vspecs`, `wspecs`), generator and storage groups, line
  terms, prices, options (`measure`, `brownfield`; `storage = :soc`, `renewables = :load` and
  `regularized = true` are fixed and kept as fields for the device checks), the partitions and
  the optional `FlowBasis`.
- `WeatherModel`: specs, layouts and transfers of the weather fields, climatology and scales,
  the switching VAR, the `DetailModel`s, the historical data per level and the responsibilities.
- `BiasModel`: basis knots, coefficients $\theta$, $a$, $c$, the bias samples, the regression
  accumulators and the `active` flag of level-0 learning.
- `Problem`: energy model, weather model, levels with their layouts and transfers,
  $\beta_k$, $\beta_v$, the bias and `use_bias` ($\alpha$).
- `State`: level `ℓ`, regimes `m`, `z0`, innovations `ws`, anomalies `xs` per level, coefficient
  vectors `k` and `v`, the cached log target `L` (without the weather prior) and `CostParts`.

**Sampler (CPU).**

- `ChainSampler` (`new_sampler`): the kernels of one chain: the `LocalKernel` of the fine
  level and, for the level-0 bias warmup, the `GibbsState` (slice widths) and the step sizes of
  the carrier scaling move.
- `LocalKernel`: per-level `LocalModel`s (built lazily under a lock), per-thread `MoveWS`,
  `FlowFrames` and `AdaptiveModes` per level, block workspaces, the fine kernel mode and the
  weather fraction.
- `LocalModel`: static tables of a level: roles and buses of every field, node–line incidence,
  balance cells, units per node, `GridSolver`, cell tables for cell moves, `xloc` templates.
- `LocalCost`: the caches of a state at fixed $(k,x)$: node values of every operation field,
  clamped availabilities, clamped capacities, residual $r$, line flows, cell sums, raw weather.
- `MoveWS`: a prepared move (node entries, coefficient changes, touched lines, nodes, cells and
  storage units, kinks and their requirement-tracking data, weather entries, breakpoints) plus
  scratch for the line sampler and recorded terms (`EntTerm`, `LineTerm`, `StuTerm`, `CellTerm`).
- `StepMove`, `BlockMove`: a move at one step, or over consecutive steps (linked SoC steps,
  storage blocks, weather windows), behind the same line-sampler interface.
- `FlowFrames`: per-step predicted binding lines, face-projected modes and proxy classes, keyed
  by a hash of $(k,x)$. `AdaptiveModes`: learned directions, shapes and response coefficients.
- `OnlineBias`, `PostBias`: observation counts and refit schedule; recorded samples for the
  post-correction. `PriceAcc`, `RegionPriceAcc`: sampled price accumulators.

**Device.** `DeviceModel`, `DeviceIndex`, `DeviceState`, `FlatState`; `TermBatch` with
`GenTerms`, `UCTerms`, `LinkTerms`, `LineTermsD`, `SocTerms`, `Kinks`; `Consts`, `StaticParams`,
`ChainParams`; `Topo` and the schedule (rounds, host and device index tables); `KParams`,
`RenParams`/`RenState`; `BiasDev`/`BiasState`. See [device.md](device.md).

## 4. Call graph of a run

**Entry.** `scripts/run.jl <config.toml>` → `load_config` (with `check_config`) → `run(cfg)`,
which is `run_chains(cfg)`. Device runs start from `device/scripts/run_device.jl` →
`ESMDevice.run_device(cfg)`.

**Setup** (`setup`, shared by all drivers): `load_system` → `build_energy_model` (with
`flow_basis` for `basis = "flow"`) → `check_levels` → `weather_model_cached` →
`sample_weather` → start: `solve_lp_cached` (`init = "lp"`) or `heuristic_start` → β
(fixed, or calibrated from the LP) → `init_bias` → `build_problem` → `warmstart_bias!` (LP
start at $\alpha = 1$) → the start state at the finest level by prolongation from the
reference level (`make_state`).

**Level-0 warmup** (at $\alpha = 1$, shared by the CPU and device drivers): `new_sampler` →
`restrict_state` to level 0 → `predescend` (`descend!` on `level0_potential`, operation only)
→ `run_level0!` with `learn = true` for `warmup0` iterations, each a `base_step`
(`local_base_step`: local operation sweep, weather slice updates, weather shear, capacity
slice updates, FFBS, carrier scaling, in random order), with bias observations and refits →
`bias.active = false`.

**CPU driver** (`run_chains`): `setup` → level-0 warmup → `chain_start` per chain →
`new_sampler` per chain → per sweep and chain `fine_local_step`, during the warmup
`online_bias_step!` (all chains re-evaluated after a refit) and, with `co2_cap > 0`, the CO₂
price update on the chain mean (all chains re-evaluated); `monitor_quantities`; in
production `post_record!` and the sampled prices (`accumulate_prices!`,
`accumulate_region_prices!`) → `chains_report` (diagnostics, post-correction, HDF5) →
`write_run_outputs` (prices, bias, CO₂ price, learned quantities, weather model,
calibration).

**Fine step** (`fine_local_step`): `local_model` → for the flow and adaptive kernels
`FlowFrames` (refreshed) and `AdaptiveModes` → in random order the operation sweep
(`local_sweep!`, `flow_sweep!` or `adaptive_sweep!`) and the capacity moves
(`kscale_sweep!` then `kpeak_sweep!`) → `maybe_weather!` → `fine_weather_sweep!`.

**Operation sweep** (`local_sweep!`): `LocalCost` → per parity class, threaded over its steps:
build the move list → per move `clear!`, `prep_*` (direction and compensation) → `finalize!`
(touch sets, primary breakpoints) → `sample_line!` (`walk!` over `eval_move!`) → Jacobian step
(`logjac`) → `commit!`/`commit_block!` → `evaluate!` of the state.

**Device driver** (`run_device`): `setup` → level-0 warmup at $\alpha = 1$ → `chain_start` →
`pack`, `to_device`, `plan` → compile `run_sweeps` and `capacity_pass` → per checkpoint: sweeps,
capacity pass, `unflat!`, `to_host!`, online bias, `monitor_quantities`, `post_record!` →
`chains_report`.

## 5. Caches

- **Weather model**: `<dirname(out)>/cache/weather_<hash>.jls`, keyed by a content fingerprint of
  the weather tensors, the levels, `K`, the seed and the bases.
- **Reference LP**: `<dirname(out)>/cache/lp_<hash>.h5`, keyed by a fingerprint of the network,
  units and weather, the reference level, seed, `K`, prices, levels, storage and renewable forms,
  basis, and the model options that change the LP.
- **In memory**: `LocalModel` per level (lazy), `LocalCost` per sweep (rebuilt from the state;
  incremental within a sweep), `FlowFrames` keyed by $(k,x)$, recorded walk terms per move, column
  prefixes of the bases (`colcache`), storage shapes per (step, length, cell) in the flow basis.

## 6. Threading and colouring

Julia threads (`-t`) parallelize within a sweep; the chains of the CPU driver run one after
another. In `local_sweep!` the steps of one parity class run in parallel with `:static`
scheduling, each step with its own `Xoshiro` stream seeded from the chain's generator and a
per-thread `MoveWS` (two per thread in the SoC form, for the linked step). Storage and
learned-shape block sweeps colour blocks by parity of the block index; fine weather moves run
blocks with equal index modulo a spacing in parallel, with workspaces handed out through a
channel; the level-0 slice updates use five colour classes of steps. Capacity moves and
level-0 weather moves at $\alpha = 1$ run in sequence. On the device, rounds are built from
time classes, per-node unit slots, a distance-2 node colouring and an edge colouring of the
links ([device.md](device.md) §4).

## 7. Outputs

Each driver writes one HDF5 file (`out`); the layout is in [data.md](data.md) §3.
Costs per move and per sweep, scaling and device throughput are in
[performance.md](performance.md).

## 8. Tests

See [testing.md](testing.md) for the strategy. CPU: `test/runtests.jl` (level transfers,
FFBS, cost gradients, the Haar basis, and the included files `test_local.jl`, `test_flow.jl`,
`test_bias.jl`, `test_model.jl`, `test_convergence.jl`), plus `test/test_exact.jl` (end-to-end
exactness, run separately). Device: the files in `device/test/`, each run on its own. A new move family
follows the checklist in [moves/adding_a_move.md](moves/adding_a_move.md).
