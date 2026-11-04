# Overnight results

Logs in `results/toy/` and `results/`. Toy model and earlier tests:
[toy_mlda.md](toy_mlda.md), [toy_results.md](toy_results.md).

## 1. NCMC as the coarse step (toy, `toy/run_toy3.jl`)

Coarse step: log-scale the slack totals of M cells by $e^{\eta/K}$ per stage
(shares transported, slack node absorbs, Jacobian in the work), relax between
stages with a π-reversible kernel, accept with $e^{-W}$ (fine potential). The
chain alternates one base sweep and one NCMC step. 12 s per configuration.

**Uncongested hard toy, base/relaxation = local moves** (alone: no relaxation in
15 s, ESS/s ≈ 1):

| perturbation | K = 1 | K = 4 | K = 16 |
|---|---|---|---|
| merit cells, M = 1 | relax 0.68 s, ESS/s 12 / 6, acc 0.66 | 2.6 s, 2 / 3, 0.71 | 9.2 s, 1.6 / 1.3, 0.79 |
| merit cells, M = 16 | 0.42 s, 23 / 15, acc 0.43 | 1.7 s, 6 / 4, 0.51 | 5.1 s, 3 / 2, 0.61 |
| geographic cells, M = 16 | 0.40 s, 30 / 25, acc 0.44 | 1.7 s, 7 / 6, 0.46 | 7.2 s, 3 / 2, 0.46 |

(ESS/s for objective / west-half slack.) The single-stage move (K = 1), i.e. a
plain Metropolis move that scales cell totals with transported shares, gives the
whole gain: 20–30× in ESS/s and relaxation in under a second. More stages raise
acceptance only slightly and cost K sweeps each.

**Congested regularized toy** (interface W→E binding): with local or slack-pair
base kernels no NCMC variant relaxes the chain within 12 s; single-cell moves are
accepted 0.3–0.5, global moves (M = 16) ≈ 0. The interface coupling is not
addressed by cell scaling; the active-set-following mixture kernel remains the
only one that works (relax < 1 sweep, see toy_results.md part E).

**Conclusion.** NCMC adds little beyond its first stage here. The useful
ingredient is the transported (share-preserving) perturbation, which is a valid
MH move with its Jacobian and needs no coarse model.

## 2. Toy scaling: convergence vs finest scale (`toy/run_toy4.jl`)

Grid side n = 8 … 32 (N = 64 … 1024), 600 sweeps. "relax" = sweeps from the far
start to within 3 sd of the reference; ESS per 1000 sweeps (objective, west
slack).

**Uncongested hard toy**

| N | local | slack pairs | MLDA redraw, merit cells (no fine sweep) | MLDA redraw, geographic cells |
|---|---|---|---|---|
| 64 | > 600 | 4 sweeps, 386 / 316 | 5, 134 / 164 | 1, 16 / 18 |
| 256 | > 600 | 7, 464 / 622 | 5, 187 / 199 | 5, 59 / 59 |
| 576 | > 600 | 8, 484 / 552 | 17, 138 / 176 | 12, 1.7 / 1.7 |
| 1024 | > 600 | 7, 542 / 502 | 15, 221 / 169 | 6, 1.7 / 1.7 |

**Congested hard toy** (interface = half the west export)

| N | local | slack pairs | zone pairs | mixture |
|---|---|---|---|---|
| 64 | > 600 | > 600 | > 600 (west ESS 11) | 6 sweeps, 357 / 584 |
| 256 | > 600 | > 600 | > 600 (west ESS 9) | 9, 345 / 951 |
| 576 | > 600 | > 600 | > 600 (west ESS 20) | 19, 392 / 1000 |
| 1024 | > 600 | > 600 | > 600 (west ESS 5) | 10, 320 / 956 |

Cost per sweep grows linearly with N for all kernels (≈ 4 ms per sweep at
N = 1024).

**Conclusion.** Kernels whose directions follow the active set have convergence
per sweep that is independent of N: total cost O(N) per effective sample.
MLDA with merit-order cells (full-space coarse chain, shares redrawn) is also
N-independent; with geographic cells it degrades with N. Local moves never relax.

## 3. Energy model: balance-preserving pair moves (40 nodes × daily)

`src/pairs.jl`, `scripts/exp_pairs.jl`. Weather and capacities fixed at the LP's;
predicted equilibrium opex 1.791·10¹⁰ €/a (excess 9.1·10⁸). A pair move shifts
$t$ MW of injection between two units at one time step, at the same node or across
nodes with the flow re-routed by $\Delta\theta=tL^+(e_n-e_m)$; nodal residuals are
unchanged (checked: shed and dump identical to 7 digits after 1526 moves). 240 s
per run, 8 threads.

| kernel | LP start: excess / predicted | 30 %-push start: excess / predicted |
|---|---|---|
| coordinate Gibbs (Haar coefficients) | 2.16 | 67.0 |
| + same-node pairs | 2.25 | 70.4 |
| + same- and cross-node pairs | **1.91** | **42.8** |
| + pairs, partner weighted to interior units (MH-corrected) | 2.35 (half the iterations) | 65.0 |
| pairs only (not ergodic: cannot change total injection) | 58 | 908 |
| + node-level single-unit and node-potential moves (+ pairs) | 1.84 | **16.8** |
| node-level moves + pairs, coordinate sweep only every 5th iteration | **1.75** | 21.4 |
| same + intertemporal storage shifts (charge at τ₁, discharge at τ₂, partners balance) | 1.77 | 22.1 |

**Reading.**

* The coordinates of a spatially complete Haar level are cell contrasts: one
  coordinate move changes a unit at several nodes and so the residual of several
  nodes at once. Moves that change a single node's residual (one unit at one
  node; one node potential) are what the near-optimal geometry allows, and they
  give the largest gain from the far start (67 → 17).
* Balance-preserving pairs help on top of that; weighting the partner towards
  interior units did not pay for its cost.
* Intertemporal storage shifts did not change the result, so the remaining slow
  relaxation from the far start is not (only) the storage trajectory. The far
  start chain is still relaxing steadily (2.9× → 1.9× target opex between
  iterations 50 and 100), i.e. slow but not stuck.
* From the LP start every kernel ends near 1.8–2.2× the predicted equilibrium
  excess after 4 minutes, still drifting down slowly.

**Self-consistency of the target.** The equipartition prediction counts pinned
coordinates of the LP solution but not the balance kinks: every balanced
node-time is also a cost-confining direction of the priced model (N·T more
directions). With the traces of the LP-start chains, E[β excess] ≈ 6.4–6.7·10⁴
against Var(βC) ≈ 5.6·10⁴ for the node kernel (for a product of exponentials both
equal the number of confined directions), and 50 907 = 36 307 pinned + 14 600
balance kinks. The node-kernel chains are therefore within ≈ 25–30 % of
equilibrium; the coordinate-only chains still drift (their variance estimate is
dominated by the trend).

## 4. Energy model: convergence vs finest time step (`scripts/exp_scaling.jl`)

40 nodes, capacities from the daily LP, one weather draw, Haar hierarchy. For each
finest step h a reference operational LP at fixed capacities (IPM, relative
interior) gives C*_h and the pinned count. Two starts: the level's own LP solution
(2 % interior push) and the daily LP prolonged in time. 50 iterations per chain,
8 threads. Excess relative to the pinned-count prediction after 1 / 12 / 25 / 50
iterations:

| h | coordinates | reference LP | kernel, start | excess/predicted | s per iteration |
|---|---|---|---|---|---|
| 24 | 2.1·10⁵ | 51 s, pinned 54 382 | coordinate, own LP | 19.1 / 2.91 / 2.00 / **1.49** | 3.1 |
| 24 | | | node, own LP | 14.8 / 2.11 / 1.47 / **1.31** | 1.7 |
| 24 | | | node, prolonged (= own) | 16.1 / 2.09 / 1.45 / 1.27 | 1.7 |
| 12 | 4.1·10⁵ | 87 s, pinned 68 612 | coordinate, own LP | 16.9 / 3.01 / 2.35 / **2.00** | 6.4 |
| 12 | | | node, own LP | 13.3 / 2.40 / 1.91 / **1.85** | 3.4 |
| 12 | | | node, prolonged daily | 244 / 73.5 / 32.1 / 13.7 | 3.5 |
| 6 | 8.3·10⁵ | 239 s, pinned 150 187 | coordinate, own LP | 8.6 / 2.29 / 2.08 / **1.96** | 12.5 |
| 6 | | | node, own LP | 7.1 / 2.07 / 1.81 / **1.83** | 6.6 |
| 6 | | | node, prolonged daily | 285 / 138 / 58.8 / 21.9 | 6.9 |

Including the balance kinks in the count (+N·T: 14 600, 29 200, 58 400) the node
kernel after 50 iterations is at ≈ 1.03×, 1.30× and 1.32× of the corrected
prediction for 24 h, 12 h and 6 h.

**Reading.**

* From the level's own LP, convergence per iteration depends only weakly on the
  resolution (the excess ratio after 50 iterations is about the same at 12 h and
  6 h), while the cost per iteration grows linearly with T. This matches the toy:
  with local, balance-respecting moves, total work per effective sample scales
  like the number of coordinates.
* The node-level kernel is about twice as fast per iteration as coordinate Gibbs
  and converges somewhat better per iteration.
* Starting from a coarse-in-time solution prolonged to the fine level is a far
  start: intra-day variation of load and solar against constant daily dispatch.
  Its relaxation gets slower with finer resolution (13.7× at 12 h, 21.9× at 6 h
  after 50 iterations). Coarse-to-fine initialization therefore needs either an
  LP at the finest level or a better prolongation (dispatch that follows the
  intra-day weather), not just piecewise-constant interpolation.
* The reference LP itself scales super-linearly here (51 s → 87 s → 239 s from
  2·10⁵ to 8·10⁵ coordinates), while a sampler iteration scales linearly.

### Spatial scaling point: 76 nodes (27 countries), daily

`configs/haar76.toml` (data `data/processed/model76.h5`, built with
`prepare_data.py --countries …`; 130 lines, 27 DC links, Nordic and Irish AC
islands). Reference operational LP 208 s, 4.3·10⁵ coordinates.

| nodes | kernel (own LP start) | excess/predicted after 1 / 12 / 25 / 50 iterations | s per iteration |
|---|---|---|---|
| 40 | coordinate | 19.1 / 2.91 / 2.00 / 1.49 | 3.1 |
| 40 | node | 14.8 / 2.11 / 1.47 / 1.31 | 1.7 |
| 76 | coordinate | 22.0 / 3.06 / 2.23 / 1.75 | 18.2 |
| 76 | node | 17.6 / 2.56 / 1.61 / **1.27** | 8.7 |

Convergence per iteration does not degrade with the network size. The cost per
iteration grows faster than linearly (×5 for ×2 coordinates) because the spatial
bases are stored densely (each step evaluation is O(N²) per field); with sparse
Haar transforms it would be linear.

## 5. Energy model: relaxation after a weather move (`scripts/exp_weather_step.jl`)

40 nodes × daily; equilibrate 240 s from the LP, then +3σ network-mean load for
days 10–23 (opex jumps by ≈ 1·10¹⁰ €/a), relax 180 s. Baseline: warm-started
dual simplex re-solve 1.1 s, LP Δopex 2.95·10⁸ €.

| kernel | time to within 10 % of the jump | chain Δ mean opex |
|---|---|---|
| multigrid-in-time kernel (earlier run, R3) | 136 s | 1.73·10⁹ (5.9× LP) |
| coordinate Gibbs | 55 s | 5.9·10⁸ (2.0× LP) |
| node-level kernel | 76 s | 4.8·10⁸ (1.6× LP) |

The chain's cost change is now within a factor 1.6–2 of the LP's (part of the
difference is the entropy term of F at finite β, part residual lag), but one
weather step still costs ≈ 50–70× a warm LP re-solve at this size.

## 6. Unit tests

All pass, including a new test of the graph-Haar basis (orthonormal, nested,
details local to one cell with zero sum).

## 7. Summary

* **What converges.** In the toy and in the energy model, convergence is decided
  by whether the move directions respect the constraints that are active at the
  near-optimal vertex (pinned units, balanced node-times, binding interfaces),
  not by the number of levels. Exchanges with the marginal node (toy), the
  active-set mixture (congested toy) and node-level single-unit / node-potential
  moves with balance-preserving pairs (energy model) relax in a number of sweeps
  that is roughly independent of the problem size; local network moves, geometric
  multigrid directions and Haar-coefficient coordinate moves do not, or only
  slowly.
* **MLDA.** It speeds up convergence substantially only as a full-space coarse
  chain whose coarse model encodes the active set (merit-order cells; in the toy
  10–20× over the best single-level kernel per second when no fine sweep is
  needed). With geographic cells, or with a coarse model that ignores a binding
  constraint, its acceptance is ≈ 0. NCMC adds little beyond a single transported
  stage, which is simply a Metropolis move along a scale-covariant direction.
* **Energy model, fixed weather and capacities.** The node-level kernel reaches
  within ≈ 3 % (24 h) to ≈ 30 % (12 h, 6 h) of the equipartition prediction
  (counting balance kinks) in 50 iterations from the level's own LP solution, at a
  cost per iteration linear in the number of time steps (1.7 / 3.4 / 6.6 s at
  2 / 4 / 8·10⁵ coordinates on 8 threads). From 40 to 76 nodes the convergence
  per iteration is unchanged (1.31× → 1.27×); the cost per iteration grows ×5
  because of the dense spatial bases. Far starts — including a coarse-in-time
  solution prolonged to finer steps — relax much more slowly, and more slowly the
  finer the time step.
* **Weather steps.** After a 14-day +3σ load change the node kernel relaxes in
  ≈ 1 min and estimates the cost change within 1.6× of the LP; a warm LP re-solve
  takes 1.1 s at this size. The fast block is not yet competitive with warm LP
  re-solves per weather sample at 40 nodes × daily; its advantage can only appear
  where LPs scale worse than the linear-cost sweeps (the reference LP took 51 s,
  87 s and 239 s at 24 h, 12 h and 6 h, i.e. super-linear).

**Open next steps.** (1) Active-set-aware directions in the energy model beyond
the node level: price-setting units per node-time as partners (the toy's
mixture), selected with the MH correction but cheaper than the weighting tried
here. (2) A prolongation in time that keeps dispatch balanced at the finer steps
(e.g. marginal units follow the intra-day residual), to make coarse-to-fine starts
near ones. (3) The MLDA coarse model "fine model with reduced costs averaged
within merit-order cells" requires the cells to follow the active set; in the
energy model this means clustering unit-time pairs by reduced cost, which the
LP duals at a coarse level could provide.
