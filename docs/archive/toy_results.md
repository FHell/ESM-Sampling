# Tests of the alternatives on the toy

Model and analysis: [toy_mlda.md](toy_mlda.md). Code: `toy/Toy.jl`,
`toy/run_toy.jl`; log: `results/toy/run_toy.log`.

**Setting.** 16×16 grid ($N=256$), $a$ = west–east gradient + noise (slack node
$i^*$ in the east), $c_i=1$ ($S=256$), $\beta=200/\bar r$ (near-vertex regime),
hard constraints or regularized with $\mu=30\max r$ (and $3\max r$ in part B).
Every move samples its 1-D line conditional exactly (truncated or
piecewise-exponential inversion), so there are no step sizes. Ground truth comes
from an exact sampler (independent exponentials or asymmetric Laplace variables,
with rejection on the slack node).

---

## A. Operation at fixed weather: which directions mix

One sweep = $N$ line moves (plus all quadtree cells for the multigrid variant).
Far start: equal slack on all nodes. Equilibrium runs start from an exact draw.

| directions | hard: sweeps to relax | hard: ESS per 1000 sweeps (excess, west slack) | regularized: sweeps to relax | regularized: ESS (excess, west slack) |
|---|---|---|---|---|
| local network moves, $\Delta s=-Le_i$ (coordinate Gibbs in $x$) | > 1500 | 23, 14 | > 1500 | 63, 10 |
| local + quadtree cell shifts $-L\mathbf 1_C$ (multigrid) | > 1500 | 85, 4 | > 1500 | 48, 9 |
| neighbour pairs $e_i-e_j$ (re-routing along an edge) | > 1500 | 4, 11 | > 1500 | 33, 9 |
| random global pairs | 158 | 17, 109 | 236 | 8, 51 |
| **pairs with the slack node $e_j-e_{i^*}$** | **9** | **413, 334** | **6** | **465, 454** |

Exact equilibrium excess: 255 (hard), 251 (regularized). The first three samplers
also show biased equilibrium means after 1500 sweeps (239–281), i.e. they have
not mixed.

**Reading.** Network-local moves and geometric multigrid directions are both
blocked by the pinned constraints: every such direction moves several pinned
slacks at once. The direction that exchanges slack between a pinned node and the
price-setting node is a coordinate of the near-vertex product measure, and with it
the chain is essentially independent after one sweep. What matters is the
direction set relative to the LP basis (which node is marginal), not depth or
geometry.

## B. MLDA variants (proposals from exact equilibrium states)

16 cells of about 16 nodes, the slack node in its own cell. Relative spread of
reduced costs per cell, $\Delta r/\bar r$: geographic 4×4 blocks 0.28,
merit-order cells (sorted by $r$) 0.05. $M$ = number of cells moved by the
coarse proposal; "acc" = mean acceptance.

| variant | cells | M = 1 | M = 2 | M = 4 | M = 8 | M = 16 |
|---|---|---|---|---|---|---|
| Galerkin coarse model, details fixed — hard | geo | 0 | 0 | 0 | 0 | 0 |
| Galerkin, details fixed — reg. μ = 30 | geo | 0 (log α ≈ −356) | 0 | 0 | 0 | 0 (−7.2·10³) |
| entropy-corrected coarse (Gamma), details fixed — hard | geo / merit | 0.11 / 0.14 | 0.03 / 0.03 | 0.003 / 0.004 | 0 | 0 |
| entropy-corrected, details fixed — reg. μ = 30 | geo / merit | 0.13 / 0.17 | 0.03 / 0.03 | 0 / 0.004 | 0 | 0 |
| entropy-corrected, details fixed — reg. μ = 3 | geo / merit | 0.31 / 0.31 | 0.15 / 0.12 | 0.03 / 0.04 | 0.003 / 0.001 | 0 |
| **full-space coarse chain, shares transported (§5a) — hard** | **geo** | 0.71 | 0.67 | 0.45 | 0.27 | 0.08 |
| **full-space coarse chain, shares transported — hard** | **merit** | **0.97** | **0.90** | **0.85** | **0.77** | **0.53** |

Non-equilibrium switching (NCMC; entropy-corrected proposal, geographic cells,
$K$ stages of uniform shifts each followed by one within-cell pair sweep):

| | K = 1 | 4 | 16 | 64 |
|---|---|---|---|---|
| hard, M = 1 | 0.12 | 0.12 | 0.16 | 0.12 |
| hard, M = 16 | 0 | 0 | 0 | 0 |
| reg. μ = 30, M = 1 | 0.17 | 0.19 | 0.40 | 0.49 |
| reg. μ = 30, M = 16 | 0 (log α −927) | 0 (−545) | 0 (−152) | 0 (−48) |
| reg. μ = 3, M = 1 | 0.33 | 0.45 | 0.57 | 0.59 |
| reg. μ = 3, M = 16 | 0 | 0 | 0.001 | 0.018 |

**Reading.**

* (M1) is confirmed: a coarse model built from the cost alone is rejected at
  every $M$, hundreds of nats per cell.
* (M2) is confirmed: with the entropy correction, the acceptance with fixed
  details is a product of per-cell factors of 0.1–0.3 and is zero beyond a few
  cells. Regularization helps only through a small μ, and merit-order cells do
  not help when details are fixed.
* The full-space coarse chain with transported shares is the only variant that
  accepts global moves, and only with merit-order cells (0.53 at M = 16, against
  0.08 for geographic cells), as predicted in §5a. Measured acceptances are lower
  than the Gaussian estimate $2\Phi(-\sqrt v/2)$ (0.89 and 0.43), but the ordering
  and the dependence on $\Delta r/\bar r$ agree.
* NCMC with uniform shifts helps single-cell moves under regularization
  (0.17 → 0.49) but not global ones, and not at all with hard walls. With hard
  walls the uniform shift has a constant failure probability per unit of moved
  total: the relaxation re-randomizes the smallest share before every stage, so
  more stages do not reduce the risk. A transporting perturbation (shares)
  inside NCMC would avoid this; it was not tested.

## C. Weather tilt by mean force (regularized toy)

Weather prior $c\sim\mathcal N(\mathbf 1,\Sigma_c)$, $\sigma_c=0.2$,
exponential correlation of length 3. Slow block: pCN step for $c$ with force
$\lambda\hat g$; fast block: 3 sweeps of slack-node pairs to relax after the
weather move, then $\hat g=$ mean over 3 sweeps of $-\mu\,\mathbb 1\{s<0\}$
(score identity). Target: $\mathcal N(\mathbf 1-\lambda\Sigma_cr,\Sigma_c)$
(near-vertex $\nabla F=-r$). λ chosen for a mean shift of $0.5\sigma_c$; the
α = 0 joint chain would tilt with β, 2677 times more strongly.

| λ | mean shift per node | target | error | force vs −r |
|---|---|---|---|---|
| 0 | 0.024 | 0 | ‖Δc‖/(σ_c√N) = 0.15 (Monte Carlo error ≈ 0.1) | corr 0.96 |
| 0.0635 | 0.123 | 0.100 | 31 % of the shift | corr 0.95, magnitude error 26 % |

**Reading.** λ = 0 leaves the weather distribution unbiased, independent of the
fast chain. A mild tilt is reproduced in direction and roughly in size by forces
averaged over a few fast sweeps; the remaining error is dominated by the noise of
the force estimator, whose signal comes from rare violations (probability
$r/\mu$ per node). Errors enter damped by λ, as intended, instead of amplified by
β.

---

## Conclusions for the energy model

1. **Moves aligned with the LP basis beat everything else by orders of
   magnitude.** Exchange with the price-setting unit (balance-preserving,
   merit-order aware) mixes in a few sweeps; local, geometric-multigrid and
   generic pair moves do not relax in 1500 sweeps. In the energy model this means
   moves that exchange dispatch between pinned units and the marginal units of
   the node/time, with flows re-routed accordingly; the marginal units change with
   the weather, so the direction set has to follow the current active set.
2. **MLDA works only as a full-space coarse chain that transports details**, with
   cells grouped by merit-order position. Fixed details fail regardless of
   regularization, depth or switching.
3. **NCMC helps local moves under regularization only**; its perturbation should
   transport rather than shift.
4. **Mean-force tilting works** and is unbiased at λ = 0; the force estimator's
   variance (rare-event signal) is the quantity to improve, for example by
   Rao–Blackwellizing the violation probability from the line conditionals.

The toy is easier than the energy model in one respect: its near-vertex measure
is exactly a product in the slack-node pair coordinates, and the marginal node
does not change with the weather. Both are what has to be engineered in the
full model.
