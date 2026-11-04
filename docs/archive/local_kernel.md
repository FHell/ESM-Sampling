# Sparse bases and the local exact kernel

Status after the implementation round that started from "sparse over a
complete basis first". Code: `src/spatial.jl` (bases), `src/local.jl` (kernel),
`src/diagnostics.jl`, options in `src/run.jl` (`kernel = "local"`, `measure`,
`storage`). Tests: `test/test_local.jl` (included in `test/runtests.jl`).

## 1. Sparse graph-Haar bases

* `ModeBasis` stores `U` as a sparse matrix (plus its transpose for row access,
  a cache of column prefixes and, for Haar levels, the cell labels and sizes).
  No dense support × modes matrices are left in the model.
* `haar_basis(partitions, support)` builds the Haar basis of a field directly on
  its support (no Gram–Schmidt of global vectors). Inside each parent cell the
  children are ordered along the Fiedler vector of their quotient graph and
  contrasted as a balanced binary tree, so the vectors of one parent have
  O(size · log #children) nonzeros (a 40-node field: ≤ 271 nonzeros instead of
  1600).
* Potentials use the partition (AC component → country → node) without the
  component indicators (gauge-free by construction).
* The level projection of the residual (balance at a coarse level) is the cell
  mean, O(N) (`project!`), instead of U Uᵀ r.
* The flow re-routing of cross-node pairs uses sparse Cholesky factors of the
  grounded Laplacian per AC component (`GridSolver`, pure-Julia triangular
  solves, thread-safe) instead of a dense L⁺.

## 2. Local incremental cost

`LocalCost` caches, at one level and for fixed (k, x), the node values of every
v field, the residual r (N × T), the line flows and the residual sums of the
balance cells. Every cost term of step τ is local: a unit term (generator,
storage, link), a line term, and a cell balance term Δt·n_c·φ(S_c/n_c). A move is
a sparse direction given both as node entries and as coefficient changes; its
cost change is evaluated in O(size of the move), and committed in the same
time. Consistency with `energy_cost` is tested to round-off for all move
families and both storage forms.

## 3. Moves

| family | direction | where |
|---|---|---|
| atom | one basis vector (the coordinate update) | all levels |
| unit | one unit at one node | complete fields |
| pair | two units at one node, injection-balanced | complete |
| xpair | two units at different nodes, flow re-routed by Δθ = L⁺(e_n − e_m) | complete |
| potential | one node's angle | complete θ |
| cell | one carrier over a whole cell of a coarser partition (multigrid atom; the overcomplete Haar frame) | all levels |
| cellpair | two carriers over a cell, cell sum of the residual unchanged (transported coarse direction) | all levels |
| spair | (SoC form) state of charge at τ with generator compensation at τ−1 and τ | complete |
| tpot | transport move (angle of one node) with pro-rata re-dispatch of all generators at every affected node, σ_i = p_i/P fixed; Jacobian Π(P(t)/P)^{m−1} by an independence Metropolis step after the exact draw (the core move of the transport-first formulation, sampling_formulations.md §8) | complete |
| block families | any of the above, constant over a dyadic block of steps (temporal multigrid) | `block_sweep!` |
| kscale | (k, ω) ↦ (e^u k, e^{−u} ω) for one unit, generalized Gibbs with Jacobian | complete k |

Selection uses only (k, x), which the operational sweep holds fixed, so every
move is π-reversible; the sweep order is a random permutation of the move list,
steps of one parity class are updated in parallel.

**Exact line conditionals.** Along a move the cost is piecewise linear in t,
so the conditional is piecewise exponential. Primary breakpoints: clamp points
of moved node values and limit points of moved flows. Between consecutive
primary breakpoints the residual cell sums and the storage dynamics residuals
are linear, so their sign changes are found exactly. The conditional is sampled
by segment masses and inversion, walking out from t = 0 until the potential is
40 nats above its running minimum (neglected mass < e⁻⁴⁰). No step widths: the
same sampler serves a node-level unit move and a network-wide cell move,
whose natural scales differ by orders of magnitude. A Kolmogorov–Smirnov test
against quadrature is in the tests.

## 4. Options

* `measure = "physical"`: reference measure in MW for dispatchable units
  (docs/physical_formulation.md §2).
* `storage = "soc"`: storage as a state-of-charge trajectory (one field per
  carrier, charge/discharge derived; docs/physical_formulation.md §4). A change of
  the state of charge at instant τ moves the injection at τ−1 and τ; the kernel
  evaluates such moves as linked two-step blocks.
* `renewables = "load"`: renewables as negative load (output A·k exactly,
  curtailment is the negative nodal residual priced at c_dump; no ω fields;
  sampling_formulations.md R1).
* `kernel = "local"`: level 0 mixes the local sweep with the weather colour
  classes (coordinate slice on z0 at fixed v), the k block, FFBS and the carrier
  scaling move; levels ≥ 1 use local sweeps + kscale as the fixed-level kernel
  inside MLDA.

## 5. Results (40 nodes, Haar, α = 0, capacities from the daily LP, one weather draw)

Equilibrium excess of the operational cost over the level's LP optimum, in
units of the equipartition prediction n_pinned/β_v (in brackets: in units of
(n_pinned + N·T)/β_v, which also counts the balance kinks), from the level's
own LP solution (`scripts/exp_local.jl`, 100 sweeps, 8 threads):

| finest step | coordinates | after 25 sweeps | last quarter | s/sweep | old node kernel (50 its) |
|---|---|---|---|---|---|
| 24 h | 2.1·10⁵ | 1.56 | 1.40 [1.07] | 0.63 | 1.31, 1.7 s/it |
| 12 h | 4.1·10⁵ | 1.92 | 1.91 [1.34] | 1.20 | 1.85, 3.4 s/it |
| 6 h | 8.3·10⁵ | 1.84 | 1.96 [1.43] | 3.3* | 1.83, 6.6 s/it |

| 76 nodes, 24 h | 4.3·10⁵ | 3.17 (12 sweeps) | 2.08 [1.45], still falling at 50 | 1.29 | node 1.27 at 8.7 s/it; coord 1.75 at 18.2 s/it |

(*) run concurrently with another 7-thread job.

Cost per sweep now scales linearly with the network: 40 → 76 nodes is 0.63 →
1.29 s/sweep (the dense-basis code went from 1.7 to 8.7 s/it, ×5). A sweep of the local kernel
contains every atom of every step plus 160 random moves per step; it is 2–3×
cheaper than an iteration of the old node kernel. The stationary level is
reached after ≈ 25 sweeps at every resolution (the old runs were still rising).

Starts far from the optimum (the daily LP of the setup carried through the
hierarchy): 24 h 458 → 19.8 after 100 sweeps; 12 h 352 → 16.1; 6 h 259 → 10.8
(old node kernel: 13.7 at 12 h and 21.9 at 6 h after 50 its, i.e. about the
same per second). In a second far-start test (`scripts/exp_farstart.jl`,
another LP draw) shed relaxes within ≈ 50 sweeps (1 min) and the cost settles at
1.35×, but dump and storage cycling keep drifting for minutes: they are the
soft directions (5 €/MWh, round-trip losses). Temporal block moves did not help
at equal wall time (1.32× vs 1.35× after 150 s).

## 6. Diagnostics and sampled prices

`ess` (Geyer initial monotone sequence), `split_rhat`, and sampled nodal
prices: by the score identity ∂F/∂d_{n,τ} = E_π[∂C/∂d_{n,τ}] = Δt·E[c_shed 1(r̄>0)
− c_dump 1(r̄<0)], so the sampled price is the imbalance probability weighted by
the shed and dump prices (`PriceAcc`, `accumulate_prices!`). The driver writes
ESS and split-R̂ per quantity and the sampled prices (with the LP duals when the
reference level is the finest level).

`scripts/exp_prices.jl` (40 nodes × 24 h, LP capacities, 200 sweeps from the LP,
prices averaged over the last 150): mean 88 €/MWh sampled vs 72 €/MWh LP duals
(load-weighted 95 vs 79); correlation 0.54 over node × step, 0.78 over node
means, 0.62 over the daily system mean. The sampled prices have the heavier
upper tail expected at finite β_v (90 % / 99 % quantiles 195 / 536 vs 121 / 124
€/MWh): a price is the shed probability times c_shed, and near-optimal states
shed occasionally where the LP sits exactly at the margin.

## 7. Formulation comparison (far start, 40 nodes × 24 h, 150 s, 8 threads)

`scripts/exp_farstart.jl` with `STORAGE`, `RENEW`, `MEASURE` (same weather and
capacities; LP optimum of the generation cost 1.938·10¹⁰ €). Excess ratios are
not comparable across formulations (different pinned counts; with renewables
as load, curtailment is priced as dump), so the table shows the parts:

| formulation | sweeps in 150 s | generation cost − LP | storage cost | dump [MWh] (drift) | penalty |
|---|---|---|---|---|---|
| LP storage, ω renewables | 214 | 1.13·10⁹ | 1.61·10⁶ (rising) | 1.7·10⁷ (rising) | 4.5·10⁸ |
| SoC storage | 206 | 7.3·10⁸ | 1.08·10⁶ (flat) | 2.1·10⁷ (rising) | 1.7·10⁸ |
| SoC + renewables as load | 255 | 4.8·10⁸ | 1.13·10⁶ (flat) | 9.6·10⁷ incl. curtailment (flat) | 1.4·10⁸ |

The two changes of the physical formulation reduce the excess generation cost
2.4× at equal wall time and remove both drifts (storage cycling, curtail-vs-dump).

**Pro-rata transport moves** (`tpot`, `tsoc`, weight 0.3 each, SoC + renewables as
load): no gain at fixed (k, x) — after 75 s generation cost 1.9859·10¹⁰ vs
1.9852·10¹⁰ without them (and fewer sweeps per second). The operational block
near the optimum is already served by the unit/pair moves; the value of the
pro-rata compensation is expected where weather or capacities move (the shear),
which is how `weather_shear_sweep!` uses it.

**Physical measure with investment moves at α = 0** (`MEASURE=physical`, local +
`kscale`): capacities run away (battery 0.46 → 412 GW, H₂ → 107 GW, gas 96 → 225
GW in 150 s). In ρ₀ the capacity marginal is e^{−β_k K}·Z, and under the physical
measure every flat dispatch direction (an idle storage unit's trajectory) adds
log k per step: the entropic force n_flat/k beats β_k c by orders of magnitude.
This is the ρ₀ pathology of sampling_formulations.md §0.1; the model is ρ₁
(per-year perfect-foresight dispatch, investment against the weather
distribution), where the bias carries the k-dependence of log Z and the force is
scaled by β_k/β_v. Capacity moves must therefore be tested at α = 1, not α = 0.

## 8. Weather moves with shear (level 0)

`weather_shear_sweep!` (`src/weather_moves.jl`) moves every level-0 weather
coordinate together with a pro-rata re-dispatch of the dispatchable carriers of
each affected balance cell (cell-constant utilization changes, in the span of
the level; σ invariant along the line, Jacobian Π(P_c(t)/P_c)^{m_c−1}); the line
target is the full level-0 term (cost, bias, weather prior of τ and τ+1). It is
part of the level-0 kernel of `kernel = "local"`. First test at α = 0: after
150 sweeps of coordinate Gibbs, one shear sweep raised L₀ from −1.19·10⁸ to
−9.7·10⁶: the coordinate kernel had not equilibrated, and the sheared moves let
the weather reach low-cost patterns at once. At α = 0 that is the ρ₀ tilt of the
weather marginal (correct for ρ₀, not the model); the informative test is at
α = 1 with the bias, still to be done.
