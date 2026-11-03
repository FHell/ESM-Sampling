# Architecture

This document maps the system onto the code: the packages and files, the main types, the
call graph of a run in each driver mode, the caches, threading and colouring, the outputs and
the tests. A final, clearly marked section lists legacy code that still exists but is not
part of the system. What the code computes is described in [ensemble.md](ensemble.md),
[model.md](model.md), [weather_model.md](weather_model.md) and [sampler.md](sampler.md); the
device port in [device.md](device.md).

## 1. Repository layout

| path | content |
|---|---|
| `src/` | package `ESMSampling`: model, weather, sampler, drivers |
| `device/` | package `ESMDevice` (depends on `ESMSampling` by path, and on Reactant 0.2.304): the GPU port |
| `test/`, `device/test/` | tests ([testing.md](testing.md)) |
| `configs/` | TOML run configurations ([config.md](config.md)) |
| `scripts/` | driver script, data preparation, benchmarks, plotting |
| `data/raw/`, `data/processed/` | downloaded sources and model input files ([data.md](data.md), provenance in [data_sources.md](data_sources.md)); not in git |
| `results/` | run outputs and the caches (`results/cache/`); not in git |
| `external/pypsa-eur/` | the PyPSA-Eur workflow used to build the 256-node network ([data_sources.md](data_sources.md)) |
| `external/reactant-bench/` | Reactant prototypes and benchmarks |
| `work/perf/` | a working copy used for CPU performance work |
| `toy/` | legacy toy models (§9) |

## 2. Files of `src/`

In include order (`src/ESMSampling.jl`):

| file | content |
|---|---|
| `spatial.jl` | `ModeBasis`; graph Haar bases (`haar_basis`, `haar_field_basis`, weighted for capacities); Laplace modes; projections |
| `layout.jl` | `FieldSpec` (kinds `KFIELD`, `STEP`, `INSTANT`), `Layout`, `Transfer`, `split_state`/`merge_state`, `restrict_to`/`prolong_to` |
| `system.jl` | `System`, `load_system` (HDF5 input), `subsystem` (sub-networks for tests) |
| `flowbasis.jl` | `FlowBasis`: proxy capacities and injections, `CongestionADMM`, transport modes, angle basis, storage shapes |
| `energy.jl` | `GenGroup`, `StorGroup`, `LineTerms`, `EnergyModel`, `build_energy_model`, `energy_cost` (with gradients), `log_measure`, `set_co2_price!` |
| `constraints.jl` | `Constraints` for hard walls (empty with regularized constraints) |
| `lp.jl` | the reference capacity-expansion LP (JuMP/HiGHS), `LPSolution`, cache |
| `weather.jl` | `WeatherModel`, `DetailModel`, fit (EM, details), exact sampling, `weather_nodes`, cache |
| `hmc.jl` | reflective dynamics on a polytope; `descend!` serves the optional pre-descent |
| `bias.jl` | `BiasModel`: $\hat g$, $\hat h$, $\hat b$, score regression, monotone QP |
| `hierarchy.jl` | `Problem`, `State`, `logtarget`, `make_state`, `bias_term` |
| `bias_online.jl` | `OnlineBias`, `online_bias_step!` |
| `bias_post.jl` | `PostBias`, `post_correct` |
| `mlda.jl` | `MLDASampler` (the sampler-state container of every CPU driver), `quantities`, `base_step`, `level_step`, carrier scaling `scale_step`; MLDA proper is legacy |
| `gibbs.jl` | level-0 slice updates (`GibbsState`, `SweepCtx`, `StepWS`, `step_term`, `update_colour!`, `update_k!`, `slice_coord!`) |
| `pairs.jl` | legacy slice pair moves |
| `local.jl` | the local exact kernel: `GridSolver`, `LocalModel`, `LocalCost`, `MoveWS`, `eval_move!`, line sampler, move preparation, `local_sweep!`, `kscale_sweep!`, `LocalKernel`, `fine_local_step`, `local_base_step` |
| `weather_moves.jl` | level-0 weather move with pro-rata shear |
| `weather_fine.jl` | fine-level weather moves |
| `flowmoves.jl` | `FlowFrames`, transport families, storage multiscale sweep, `flow_sweep!`, localized modes |
| `timehier.jl` | prolongation rules in time and intra-block diagnostics (legacy) |
| `adaptive.jl` | `AdaptiveModes`, learned families, `adaptive_sweep!`, `kpeak_sweep!` |
| `decompose.jl` | decomposed LP start (legacy) |
| `diagnostics.jl` | single-chain ESS and split-$\hat R$, `PriceAcc`, `RegionPriceAcc` |
| `convergence.jl` | multi-chain diagnostics (rank-normalized $\hat R$, bulk/tail ESS, MCSE, warmup detection) |
| `run.jl` | `Config`, `load_config`, `setup`, `heuristic_start`, level-0 chain, `run` |
| `chains.jl` | multi-chain mode: `run_chains`, `chain_start`, `monitor_quantities`, `chains_report` |

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
  terms, prices, options (`measure`, `storage`, `renewables`, `regularized`, `brownfield`), the
  partitions and the optional `FlowBasis`.
- `WeatherModel`: specs, layouts and transfers of the weather fields, climatology and scales,
  the switching VAR, the `DetailModel`s, the historical data per level and the responsibilities.
- `BiasModel`: basis knots, coefficients $\theta$, $a$, $c$, the bias samples, the regression
  accumulators and the `active` flag of level-0 learning.
- `Problem`: energy model, weather model, levels with their layouts, transfers and constraints,
  $\beta_k$, $\beta_v$, the bias and `use_bias` ($\alpha$).
- `State`: level `ℓ`, regimes `m`, `z0`, innovations `ws`, anomalies `xs` per level, coefficient
  vectors `k` and `v`, the cached log target `L` (without the weather prior) and `CostParts`.

**Sampler (CPU).**

- `MLDASampler`: the state container of a chain's kernels (base kernel choice, `GibbsState`,
  `LocalKernel`, carrier-scaling scales, `p_fine`, statistics); its MLDA fields are legacy.
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

**Entry.** `scripts/run.jl <config.toml>` → `load_config` → `run(cfg)`; with `nchains > 0`,
`run` hands over to `run_chains`. Device runs start from `device/scripts/run_device.jl` →
`ESMDevice.run_device(cfg)`.

**Setup** (`setup`, shared by all drivers): `load_system` → `build_energy_model` (with
`flow_basis` for `basis = "flow"`) → `check_levels` → `weather_model_cached` →
`sample_weather` → start: `solve_lp_cached` (`init = "lp"`) or `heuristic_start` → β
(fixed, or calibrated from the LP) → `init_bias` → `build_problem` → `warmstart_bias!` (LP
start at $\alpha = 1$) → the start state at the finest level by prolongation from the
reference level (`make_state`).

**Single-chain driver** (`run`, `nchains = 0`): `new_sampler` → level-0 restriction and
`predescend` → `run_level0!` with learning (warmup, bias) → `run_level0!` (level-0 production
chain of `n0` iterations) → exact weather draws for the diagnostics → finest level:
`mlda_warmup` + `nsamples` calls of `level_step`, which with `p_fine = 1` is always
`fine_local_step`, each followed by `online_bias_step!` and, during warmup, the CO₂ price
update; production records quantities, post-correction samples and sampled prices → the
$\alpha = 0$ level-0 comparison (`alpha0`) → `weather_cost_map` → HDF5.

**Fine step** (`fine_local_step`): `local_model` → for the flow and adaptive kernels
`FlowFrames` (refreshed) and `AdaptiveModes` → in random order the operation sweep
(`local_sweep!`, `flow_sweep!` or `adaptive_sweep!`) and the capacity moves
(`kscale_sweep!` then `kpeak_sweep!`) → `maybe_weather!` → `fine_weather_sweep!`.

**Operation sweep** (`local_sweep!`): `LocalCost` → per parity class, threaded over its steps:
build the move list → per move `clear!`, `prep_*` (direction and compensation) → `finalize!`
(touch sets, primary breakpoints) → `sample_line!` (`walk!` over `eval_move!`) → Jacobian step
(`logjac`) → `commit!`/`commit_block!` → `evaluate!` of the state.

**Multi-chain mode** (`run_chains`): `setup` → level-0 warmup at $\alpha = 1$ →
`chain_start` per chain → `new_sampler` per chain with `p_fine = 1` → per sweep and chain
`level_step`, `online_bias_step!` (warmup only; all chains re-evaluated after a refit),
`monitor_quantities`, `post_record!` → `chains_report` (diagnostics, post-correction, HDF5).

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

Julia threads (`-t`) parallelize within a sweep; chains of the multi-chain mode run one after
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
`scripts/plot_results.py` plots the single-chain output. Costs per move and per sweep, scaling
and device throughput are in [performance.md](performance.md).

## 8. Tests

See [testing.md](testing.md) for the strategy. CPU: `test/runtests.jl` (bases and transfers,
FFBS, the Haar basis, and the included files `test_local.jl`, `test_flow.jl`, `test_bias.jl`,
`test_model.jl`, `test_convergence.jl`), plus `test/test_exact.jl` (end-to-end exactness, run
separately). Device: the files in `device/test/`, each run on its own. A new move family
follows the checklist in [moves/adding_a_move.md](moves/adding_a_move.md).

## 9. Legacy code (not part of the system)

The following code still exists and runs, but is not part of the system described in these
documents. It is kept for reference or possible later use; nothing in the system depends on
it except where noted.

- **Multilevel delayed acceptance** (`src/mlda.jl`: `mlda_core`, `mlda_step`,
  `perturb_details`, `pcn`, `detail_scales`, `LevelStats`, `violated_fields`). Coarse chains on a
  split state with delayed acceptance of fine proposals. Reached through `level_step` when
  `p_fine < 1`; configuration keys `J`, `frac`, `ϱ`, `p_fine`; outputs `pairs/`, `mlda/`,
  `estimate/multilevel`. The system sets `p_fine = 1` (the multi-chain and device drivers do so
  always). Its toy studies are in `toy/`.
- **Other base kernels.** `kernel = "gibbs"` (coordinate slice Gibbs at level 0, `gibbs_step`,
  and `fine_gibbs_step` at finer levels) and `kernel = "hmc"` (reflective HMC, `hmc.jl`, with
  `nleap`, `mass_windows`, `p_ffbs`, `p_scale`). The system uses `kernel = "local"`, whose level-0
  kernel reuses the slice updates of `gibbs.jl` and whose optional pre-descent uses `descend!`.
- **Slice pair moves** (`src/pairs.jl`: `pair_sweep!`, `storage_shift_sweep!`), superseded by the
  exact pair families.
- **Temporal block moves** (`block_sweep!`, `prep_block!` in `local.jl`;
  `temporal_multigrid_sweep!` in `gibbs.jl`): time-constant directions over dyadic blocks. Only
  experiment scripts call them. (`BlockMove` itself is part of the system.)
- **Decomposed LP start** (`src/decompose.jl`, `init = "decomposed"`, `decomp_cap`,
  `decomp_block_days`).
- **Prolongation rules and intra-block diagnostics** (`src/timehier.jl`) and
  `init_by_prolongation` in `run.jl`; used by experiment scripts.
- **Alternative model forms** kept as options on the CPU: hard walls (`regularized = false`,
  with `constraints.jl` and the feasibility test in `runtests.jl`), Laplace bases
  (`basis = "laplace"`), the LP storage form (`storage = "lp"`) and renewables with utilization
  fields (`renewables = "dispatch"`). The local kernel needs the Haar basis and regularized
  constraints; the device supports only the SoC form and renewables as load.
- **Parts of the single-chain driver** beyond the fine chain: the level-0 production chain
  (`n0`) and the $\alpha = 0$ comparison at level 0 (`alpha0`), the weather–cost map
  (`costmap/`), and the multilevel estimator.
- **Experiment scripts** `scripts/exp_*.jl`, `scripts/compare_kernels.jl`, `scripts/scaling.jl`.
  The benchmark scripts (`bench_perf.jl`, `prof_kernel.jl`, `regress_cmp.jl`, `xloc_eff.jl`,
  `xloc_leak.jl`, `device/scripts/throughput.jl`) are current tools.
