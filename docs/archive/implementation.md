# Algorithms and implementation

Companion to [mathematical_formulation.md](mathematical_formulation.md). It
describes the code as it stands, how the pieces fit, what has been measured,
and what is missing (§9).

---

## 1. Layout and running

```
scripts/download_data.sh      PyPSA network + OPSD files → data/raw
scripts/prepare_data.py       network extraction, NASA POWER download, CF calibration, load model → data/processed/model.h5
src/ESMSampling.jl            Julia package (modules below)
scripts/run.jl                driver: julia -t 8 --project=. scripts/run.jl configs/<cfg>.toml
scripts/compare_kernels.jl    level-0 Gibbs vs HMC from the same start (short runs)
scripts/scaling.jl            LP/sampler scaling benchmark (written, not run)
scripts/plot_results.py       figures from a run's HDF5 output (not updated to the newest outputs)
configs/quick.toml            2 levels, Laplace modes
configs/demo.toml             4 levels, Laplace modes (4 → 12 → 40 modes daily → 40 modes 6 h)
configs/haar_poc.toml         3 levels, graph Haar (countries daily → nodes daily → nodes 6 h), α = 0
test/runtests.jl              unit tests
```

Python (uv): pypsa, h5py, scipy, matplotlib. Julia 1.12: HDF5, JuMP, HiGHS
(+ stdlib). Data and results are git-ignored.

---

## 2. Data pipeline (`scripts/prepare_data.py`)

1. Load `carbon_management.nc`, `calculate_dependent_values()`; map every bus
   to its AC location; keep AC buses of the selected countries.
2. Lines merged per pair (Σ 1/x_pu_eff, Σ s_nom_opt), DC links merged per pair.
3. Generators per (node, carrier); OCGT links → gas generators; batteries and H₂
   assembled from store + charger + discharger; PHS/hydro storage units.
4. Fetch NASA POWER per node (4 parallel requests, cached in `data/raw/power`).
5. Calibrate CF scale per generator by `brentq` against the network's mean
   `p_max_pu` over its snapshot window; fit the national load models; drive
   them with node temperatures; scale to the nodal loads.
6. Write HDF5: `network/*`, `generators/*`, `storage/*`,
   `weather/<field>/{nodes,data}` with data `[year, hour, node]` (Julia reads it
   reversed as `[node, hour, year]`).

---

## 3. Core data structures (Julia)

| type | file | content |
|---|---|---|
| `System` | `system.jl` | raw arrays from HDF5 |
| `ModeBasis` | `spatial.jl` | `U` (support × r, dense), `support`, `rank_at[M]` |
| `FieldSpec` | `layout.jl` | name, kind (`KFIELD`/`STEP`/`INSTANT`), basis, node box |
| `Layout` | `layout.jl` | `(M, h, T)`, modes per field, offsets into the flat vector |
| `Transfer` | `layout.jl` | ratio r, Helmert matrix, detail offsets, `dblock` (coarse block of each detail coordinate) |
| `EnergyModel` | `energy.jl` | specs for k, v, weather; generator/storage groups; prices; `regularized`, `κ0` |
| `Constraints` | `constraints.jl` | coordinate boxes + sparse rows `A`, `Aᵀ`, `b` (empty when regularized) |
| `WeatherModel` | `weather.jl` | layouts/transfers, climatology per level, MS-VAR parameters, detail models, data per level, responsibilities |
| `BiasModel` | `bias.jl` | ĝ coefficients, ĥ coefficients, exact weather samples, LS accumulators |
| `Problem` | `hierarchy.jl` | everything above plus per-level layouts, transfers, constraints, β, bias flag |
| `State` | `hierarchy.jl` | level, regimes `m`, `z0`, innovations `ws`, weather per level `xs`, `k`, `v`, cached `L`, `CostParts` |
| `MLDASampler` | `mlda.jl` | subchain lengths, HMC state, Gibbs state, proposal scales, statistics, estimator pairs |

All states are flat vectors in mode coordinates; `fieldview` gives the
(modes × time) matrix of a field; `nodevals` = `U[:,1:m] * Z`.

---

## 4. Energy model evaluation (`energy.jl`, `constraints.jl`)

* `energy_cost(E, lk, lv, zk, zv, Xn; grad)` evaluates capex and operational
  cost at a level for raw node weather `Xn` and returns `CostParts` and, with
  `grad = true`, analytic gradients w.r.t. k (capex and operation separately),
  v and the raw node weather. Kinks use subgradients (sign, indicator).
  Per-group gradient loops sit behind function barriers (`gen_grad`,
  `stor_grad`); one evaluation with gradient at level 0 costs ≈ 3 ms.
* `ishard(E, lay, specs, f)`: hard walls only if `regularized = false` and the
  field is complete at the level; otherwise clamp + penalty.
* Balance pricing on the projected residual uses the load field's basis
  (`U_m U_mᵀ r`, dense).
* `build_constraints` (hard mode only): node rows for complete fields
  (single-entry rows become boxes), coefficient boxes for incomplete fields,
  θ bounds from 2·(N−c) small LPs (`theta_mode_bounds`).

---

## 5. Level-0 kernels

### 5.1 Coordinate Gibbs (`gibbs.jl`, default)

* **Locality.** `step_term(P, ℓ, ctx, ws, zv, x, z0, m, τ)` returns
  β_v·cost_τ (+ ĝ_τ − log p_τ(z0|m) at ℓ = 0). The sum over τ equals the
  potential minus k-only terms. A step coordinate touches term τ, an instant
  coordinate terms τ−1 and τ, a weather coordinate τ and τ+1.
* **Colouring.** Steps τ ≡ c (mod 5) are updated in parallel
  (`Threads.@threads :static`, per-thread `StepWS` buffers, per-step seeds).
  `T` must be divisible by 5 for the cyclic wrap to be safe (365 and 1460 are).
* **Coordinate updates.** Neal's slice sampler (stepping out, at most 16 steps,
  shrinkage). Widths per field adapted in warmup to 3 × mean |move|; v and
  weather widths are shared across levels.
* **Reversibility.** One `gibbs_step` applies the moves {5 colour classes,
  k-slice block, FFBS, carrier scaling} in a uniformly random order, and the
  coordinates of a step in random order. That is a mixture over orderings of
  reversible updates, so it is π₀-reversible, as required for an MLDA coarse
  kernel.
* **k block.** Slice sampling of each level-0 k coordinate with full
  re-evaluation of L₀ (24–30 coordinates).
* **Cost.** ≈ 0.5–1.1 s per sweep on 8 threads at 36k–74k coordinates
  (`step_term` is allocation-free after the last change; timings above predate
  it).

### 5.1a Balance-preserving and node-level moves (`pairs.jl`)

At a spatially complete level, `pair_sweep!` updates, in parallel over time
steps, along directions defined in node space and mapped to the Haar
coefficients by the rows of the bases:

* pairs: shift t MW of injection between two units at one node, or between
  nodes with the flow re-routed by Δθ = t L⁺(e_n − e_m) (nodal residuals
  unchanged); optional partner weighting towards interior units with an MH
  correction for the state-dependent selection;
* single-unit moves at one node (change only that node's residual);
* node-potential moves (one node's θ).

`storage_shift_sweep!` adds intertemporal storage moves (charge at τ₁,
discharge at τ₂, energy carried in between, partners balance both steps).
These kernels are used by `scripts/exp_pairs.jl`, `exp_scaling.jl` and
`exp_weather_step.jl`; they are not yet wired into `run.jl`.

### 5.2 Reflective HMC (`hmc.jl`, `kernel = "hmc"`)

Leapfrog whose drift is the exact billiard in {lo ≤ q ≤ hi, Aq ≤ b}: event
queue (binary heap with versions), lazy position sync, metric reflection
`p ← p − 2(aᵀv)/(aᵀM⁻¹a) a`; dual averaging (target 0.65), diagonal mass from
Welford windows. With regularized Ω there are no walls and it is plain HMC on
a piecewise-linear potential. Observed: step size collapses to ~1e-5,
weather frozen; kept for comparison only.

### 5.3 Other moves

* `ffbs`: exact regime path given z0.
* `scale_step`: carrier scaling (formulation §7), log-scale step adapted to
  acceptance 0.3.
* `descend!` / `predescend`: Adam-normalized reflective descent of v only (k, x
  fixed), used once before warmup.

---

## 6. MLDA (`mlda.jl`)

* `mlda_core(rng, s, J, ops)`: generic two-level DA step (split, random-length
  subchain via `ops.step`, independent fine proposal, assemble, accept with
  exp(Δ(s) − Δ(ψ)), optional continuation to J for the estimator). Verified on
  a toy.
* `mlda_step` builds `ops` for the energy problem:
  * weather innovations: pCN with ϱ·scale on a random subset of coarse blocks;
  * k and v details: Gaussian random walk, per-field scale (5 % of the detail
    spread of the initial state) × an adapted global multiplier (Robbins–Monro
    on the log scale, target 0.25), on the same random subset of blocks;
  * records acceptance, infeasibility and the violated fields (hard mode).
* `level_step` at ℓ ≥ 1 is a random-scan mixture: with probability `p_fine` a
  `fine_gibbs_step` (Gibbs sweep over v at fixed (k, x) at level ℓ), otherwise
  an MLDA step. Both are π_ℓ-reversible.
* Estimator: Q₀ from a separate level-0 production chain plus, per level, the
  mean of Q_ℓ(s_ℓ) − Q_{ℓ−1}(ψ_end) over recorded pairs.

---

## 7. Driver (`run.jl`)

`setup`: load data → energy model → weather model (EM ~10 s) → exact weather
draw → LP at the reference level (HiGHS IPM without crossover, ~4 min at 40
nodes daily; cached in `results/cache/lp_<hash>.h5`) → β (fixed or calibrated)
→ bias (+ LP warm start if α = 1) → `Problem` → initial finest state.

`run`: pre-descent → level-0 warmup (kernel adaptation, bias learning when
α = 1) → bias frozen → level-0 production chain → exact weather draws for the
§v1 6.4 diagnostic → MLDA warmup and production on the finest level (from the
initial finest state) → estimator → optional α = 0 level-0 chain → weather-cost
map from ĝ, ĥ → HDF5 output (`quantity_names`, traces per level, pairs,
acceptance, estimates, bias coefficients, cost map, weather transition matrix,
β, LP objective/capacities).

Key config keys: `levels`, `basis`, `kernel`, `regularized`, `use_bias`,
`beta_v`, `beta_k`, `calibrate_beta`, `J`, `p_fine`, `frac`, `ϱ`, `p_ffbs`,
`p_scale`, `warmup0`, `n0`, `mlda_warmup`, `nsamples`, `descent_iters`,
`bias_*`, `c_shed`, `c_dump`, `c_viol`, `K`.

---

## 8. Measurements so far (40 nodes)

| run | result |
|---|---|
| LP, 40 nodes daily | 9.5e10 €/a total (7.8e10 capex): 351 GW onwind, 538 GW solar, 71 GW offwind, 103 GW gas, no battery/H₂; 230–420 s |
| HMC, level 0, hard constraints, Laplace modes (4) | acceptance 0.6–0.8, ε → 1e-5, capacities and weather nearly frozen |
| 4-level demo, hard constraints | MLDA acceptance ℓ1 0.25, ℓ2 ≈ 0.01, ℓ3 ≈ 0.03; proposals at complete levels almost all infeasible (pinned bounds) |
| Gibbs vs HMC, α = 0, 3 min each | Gibbs: log target −7.8e6 → −3.4e5, weather moves; HMC: −1.0e7 → −3.5e6, weather frozen. Neither equilibrated. |
| Gibbs vs HMC, α = 1 | both ran away (bias tail leak; earlier also an improper ĥ, fixed by clipping) |
| Haar PoC, α = 0, 3 levels | no infeasible proposals; MLDA acceptance 0 at both levels in 30 steps; level 0 total 1.75e11 (capex 1.38e11, α = 0 overbuild expected), finest total 6.3e11 still falling (shed 166 TWh from the time-prolonged start) |

---

## 9. Missing algorithmic and implementation pieces

Status update: items marked **done** are implemented in the sparse/local round
(see [local_kernel.md](local_kernel.md) and [physical_formulation.md](physical_formulation.md)).

**Base sampler (v2 §5.1)**
1. **done** (operation): exact inversion of the piecewise-exponential line
   conditionals for every local move (`sample_line!`); weather and k coordinates
   still use slice sampling.
2. **done**: balance-preserving directions (unit pairs, flow re-routing with a
   sparse grid solver, cell pairs, storage–generator pairs in the SoC form,
   temporal block moves).
3. **done**: investment moves in physical coordinates per node with the
   Jacobian (`kscale_sweep!`); reference-measure option `measure = "physical"`.
4. Sheared weather coordinates and elliptical slice sampling: open. Weather is
   still updated by coordinate slice at fixed v (level 0) and by MLDA proposals.
5. Piecewise-deterministic samplers (Zig-Zag, BPS): open (the exact line
   conditionals make them less attractive).

**Hierarchy and MLDA**
6. **done** for the operation (no widths needed); fine levels: local sweeps
   relax a level from its own LP in ≈ 25 sweeps at 24/12/6 h. Starts that are
   coarse in time remain slow (soft dump/cycling directions).
7. Gaussian level priors and pCN for k and v details: open (the overcomplete
   cell and block moves play the role of coarse moves inside the fine level).
8. F̂_ℓ adaptive error model: open.
9. Validated estimator on the energy problem: open.

**Performance**
10. **done**: sparse Haar bases, cell-mean projection, sparse flow re-routing,
    O(move) incremental costs. Remaining dense objects: the level-0 weather VAR
    matrices (small) and the θ mode bounds (hard mode only).
11. No GPU yet; threads in the colour classes of all sweeps.
12. LP: one weather year, perfect foresight; IPM without crossover.

**Diagnostics and outputs**
13. **partly done**: `ess`, `split_rhat`; not yet called from `run`.
14. **partly done**: sampled nodal prices (`PriceAcc`); not yet written by `run`.

**Tests**
15. **done** for the bases and the local kernel (`test/test_local.jl`); still
    missing for `gibbs.jl` step terms, weather details and bias regressions.

**Formulation** (new): storage as a state-of-charge trajectory
(`storage = "soc"`), capacity as the maximum of dispatch (not yet), balance
elimination by a dependent unit (not yet), sampling-oriented formulations of
the PyPSA components (docs/sampling_formulations.md, in progress).
