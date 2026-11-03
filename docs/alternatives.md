# Alternatives after the first implementation and the two reviews

Context: [mathematical_formulation.md](mathematical_formulation.md),
[implementation.md](implementation.md), [plan.md](plan.md), setup notes
[v2](../near_optimal_sampling_setup_v2.md). Units below: β_v = 4·10⁻⁵ €⁻¹,
β_k = 4·10⁻⁸ €⁻¹, 1 nat of β_v ≙ 25 k€, typical cost spread across weather
years ΔF ≈ 10⁹ €.

Requirements stated by the user:

* a **joint probability** over investments k, weather x and operation v;
* **no solve of the fine LP per weather sample** (prohibitive at high
  spatio-temporal resolution);
* **exploit the spatio-temporal hierarchy**;
* ability to **bias the weather toward cost-driving patterns** later
  (tilted weather ρ_λ(x|k) ∝ ρˣ(x) e^{λF(x,k)}, λ > 0).

---

## 0. The structural obstacle

In any chain over (x, v) whose density contains e^{−β_v C_x(k,v)}, the
marginal of x carries Z_{β_v}(x,k) = e^{−β_v F(x,k)}: a weather tilt of strength
β_v ΔF ≈ 4·10⁴ nats. To obtain ρˣ (λ = 0) or a mild tilt (λ ≈ 10⁻⁹ €⁻¹) the
density must cancel this through a bias accurate to O(1) nat, i.e. F to 25 k€
uniformly over the weather distribution. This holds for every sampler,
including MLDA; the observed +117σ load excursion at "α = 1" and the
degeneracy of α = 0 are consequences. Errors in Λ̂(k) by contrast enter ρ_K
damped by β_k/β_v = 10⁻³.

## 1. Alternatives that the reviewers proposed (recorded, not adopted)

| | idea | buys | costs |
|---|---|---|---|
| A | β_v → ∞: ρ_K ∝ exp(−β_k[c·k + E_x C*(x,k)]) by LP recourse (SAA, Benders, dual-subgradient MALA); operation post hoc | log-concave ρ_K, errors damped, MLDA over LP resolutions has O(1) increments | **one LP per weather sample and k** — excluded by the requirements |
| B | finite β_v, weather drawn exactly, v-chains at fixed (x,k), learn only Λ̂(k) (+ entropy correction) | errors damped by 10⁻³, no ĝ | Λ̂ needs a design over k; tilting needs F(x) |
| C | ĝ = −β_v C*(x,k) + ŝ with C* by LP | tail-safe bias | LP per x |
| D | integrate dispatch at fixed nodal injection analytically (smoothed merit order) | smaller, smoother fast block | still ~10⁴ dims |
| E | null-space coordinates for exact balance | removes N·T hyperplane-pinned directions | x-dependent parametrization |
| F | logit coordinates for box-bounded fields (v2 §4.5) | box-pinned directions become unit-width | storage/line constraints remain |

## 2. Proposed formulation: one joint target, fast/slow sampling by mean forces

**Target** (joint, unchanged in spirit):
$$
\rho_\lambda(k,x,v)=\rho_K(k)\,\rho_\lambda(x\mid k)\,\pi(v\mid x,k),\qquad
\rho_\lambda(x\mid k)\propto\rho^x(x)\,e^{\lambda F(x,k)},
$$
λ = 0 is α = 1; λ > 0 tilts toward expensive weather; ρ_K at β_k.

**Sampler**: a multiscale process.

* **Fast** v at β_v given (x, k): exact/tuning-free operational moves (§3).
* **Slow** x: moves that preserve the weather prior (pCN / elliptical slice in
  the non-centered weather coordinates (m, z₀, w_ℓ); FFBS for m), driven by the
  force λ∇_xF.
* **Slowest** k: Langevin / HMC / MALA in ≈ 240 dimensions on the smooth
  landscape β_k(c_k + E_x∇_kF).

**Mean forces** from the score identities (v2 §6.2), averaged over the
fast chain — no log Z, no LP:
$$
\nabla_xF(x,k)=\mathbb E_{\pi(\cdot|x,k)}[\nabla_xC_x(k,v)],\qquad
\nabla_kF(x,k)=\mathbb E_{\pi(\cdot|x,k)}[\nabla_kC_x(k,v)].
$$

* Force errors reach the weather marginal multiplied by λ and the investment
  marginal by β_k — **damped**, not amplified.
* At λ = 0 the x-moves ignore v: the weather marginal is ρˣ **by construction**.
* The tilt is a single force term; λ ladders give the cumulant generating
  function of F by thermodynamic integration, d/dλ log E e^{λF} = E_λ F.
* Metropolis corrections for x or k moves can use F-differences obtained by
  integrating the mean force along the move (with a noise-penalty correction).

Approximate elements: finite time-scale separation (v must re-equilibrate after
each slow move) and noisy forces. Diagnostics: λ = 0 weather marginal vs exact
draws; invariance under halving slow step sizes / doubling fast sweeps;
fluctuation–dissipation d⟨C⟩/dβ_v = −Var C at fixed (x, k).

Literature analogues: temperature-accelerated MD (Maragliano & Vanden-Eijnden
2006), driven adiabatic free-energy dynamics (Abrams & Tuckerman 2008),
adaptive biasing force, heterogeneous multiscale methods (E & Engquist),
multilevel Monte Carlo (Giles).

## 3. Where the spatio-temporal hierarchy enters

The same nested coefficients (graph-Haar cells in space, blocks/hierarchical
surpluses in time) serve three roles.

| | fast block v | slow block x | slow block k |
|---|---|---|---|
| representation | nested Haar/block coefficients | multilevel weather model (m, z₀, w_ℓ) | capacity coefficients: cell totals, then allocations |
| role | **move directions** (multigrid Monte Carlo): exact line sampling of every coefficient under the *fine* cost; V-cycle relaxation after slow moves; locality of re-equilibration | **step frequency per level** (multiple time stepping: coarse coefficients often, fine details rarely) | same |
| forces | — | adjoint restriction of the fine score (cell/block sums) | cell sums of node forces |
| cross-level estimation | — | **MLMC** forces ∇F_L ≈ ∇F_0 + Σ(∇F_ℓ − ∇F_{ℓ−1}) with coupled chains (shared slow state, coarse v = restriction of fine v or common random numbers) | same |

Notes:

* Weather cannot be moved into the fast block (it would pick up the
  e^{−β_vF} tilt). At λ = 0 fine weather details can be drawn from the prior.
* Seasonal storage and inter-country exchange are the slow modes of the fast
  block; coarse temporal/spatial directions exist precisely to relax them.
* A change of a fine weather detail in cell c, block j only perturbs v around
  (c, j): re-equilibration is local.

## 4. MLDA with more levels: what it can and cannot do

The level increment is extensive, Δ_ℓ = Σ_b δ_b over blocks b. A global coarse
subchain moves all B blocks, so
Var[Δ_ℓ(s) − Δ_ℓ(ψ)] ≈ B σ_b² and the acceptance decays like exp(−c B σ_b²).

* More, gentler levels (binary in space by spectral bisection, factor 2–3 in
  time, ≈ 11 levels at 40 nodes × hourly) reduce σ_b²; B (≈ 1.5·10⁴ blocks per
  level at 40 nodes × daily) is unchanged. Depth alone does not help.
* **Local (blocked) MLDA**: pick a block (colour-parallel), run a short coarse
  subchain on its coarse coefficients with neighbours fixed, propose that
  block's fine details, accept with the local increment Δ_b. Reversible for the
  block conditional; acceptance O(1) per level if σ_b² = O(1).
* Then it is a close relative of hierarchical-direction line sampling (no
  rejection, no coarse model). Local DA pays off only where the coarse local
  model is much cheaper than the fine local evaluation.
* It does not address §0 (a target property).

## 5. Experiments chosen to test the key assumptions (small, 40 nodes)

E1. **Fast block at fixed (x, k)**: coordinate Gibbs on v at nodes × daily
    from the LP solution; does the operational cost settle near
    C*_op + n_pinned/β_v, how fast, and do two starts agree?
E2. **Mean force from the fast chain**: the force on the coarse weather
    coefficients estimated by averaging scores; its Monte Carlo error and
    autocorrelation, and agreement with the zero-temperature (LP-dual) force.
E3. **Extensivity of the MLDA discrepancy**: from an equilibrated fine state,
    the distribution of log α for a single-block coarse move with fine details
    fixed (σ_b²), versus a full coarse sweep (B blocks).

---

## 6. Results of E1–E3 (40 nodes, Haar, α = 0, weather and capacities fixed at the LP's)

Script `scripts/exp_fastblock.jl`, log `results/exp_fastblock.log`, data
`results/exp_fastblock.h5`; 8 threads, ≈ 3 s per Gibbs sweep at
2.1·10⁵ operational coordinates (nodes × daily). LP opex 1.700·10¹⁰ €/a,
36 307 pinned coordinates → predicted equilibrium excess n/β_v = 9.1·10⁸ €.

**E1 — fast block at fixed (x, k).** Two chains, 120 sweeps each.

| start | after 1 sweep | mean of 2nd half (two quarters) | excess over LP |
|---|---|---|---|
| LP + 2 % push to the box centre | 4.6·10¹⁰ | 1.882·10¹⁰ (1.886, 1.877) | 1.8·10⁹ (2× predicted), still falling |
| LP + 30 % push | 8.0·10¹¹ | 5.96·10¹⁰ (6.25, 5.67) | 4.3·10¹⁰, still falling |

Relaxation from a nearby steep region is fast (within 10 % of the target cost
in ≈ 20 sweeps); the residual approach and relaxation from a large global
displacement are slow — the slow modes of the fast block (year-long storage
trajectories, exchange patterns). Two starts do not agree after 120 sweeps.
Not converged. Note that this is ~6 min per weather sample against a 4 min
cold LP solve; a warm-started LP re-solve would be faster still.

**E2 — mean force on the 4 × 365 network-mean weather coordinates.**
Per-coordinate scores are O(10³) nats per σ (as estimated in §0); their
sweep-to-sweep autocorrelation is short (τ ≈ 1 sweep) and the Monte Carlo error
of the second-half mean is 10–40 %. But the chain's mean force correlates only
0.5–0.7 with the LP-dual force, differs from it by more than its own norm, and
the two starts disagree by 140 %: the forces are dominated by the
non-equilibrium state (lag bias), not by noise. (The LP force is itself one
subgradient at a degenerate vertex, so some difference is expected.)

**E3 — MLDA discrepancy for coarse moves with fine details fixed.** Coarse
(countries × daily) conditional slice moves of n random coordinates; fine
model = nodes × daily.

| n moved | median log α | mean log α | mean acceptance |
|---|---|---|---|
| 1 | −29 | −495 | 0.21 |
| 4 | −9.5·10² | −5.4·10³ | 0 |
| 16 | −8.9·10³ | −1.7·10⁴ | 0 |
| 64 | −5.4·10⁴ | −1.0·10⁵ | 0 |
| 256 | −2.3·10⁵ | −2.7·10⁵ | 0 |
| 1024 | −1.1·10⁶ | −1.2·10⁶ | 0 |
| full sweep (60 590) | −5.0·10⁷ | | 0 |

log α is extensive (≈ −10³ nats per moved coarse coordinate) and dominated by
its mean, not its spread: the coarse model's conditional scale is far wider
than the fine model's along the same direction (the coarse model does not see
which fine coordinates are pinned), so a coarse move pushes pinned fine
coordinates into the penalties. Even single-coordinate local DA accepts only
≈ 20 %. This confirms §4 and the reviewer's point that the failure is a mean
shift.

## 7. Review of this note (Fable, 2nd round) — main corrections

1. **λ > 0 breaks the k block**: forces averaged under ρ_λ give E_λ∇_kF, i.e. a
   planner facing tilted weather. Either declare that target ("stressed
   planner") or run two weather replicas (untilted feeds k, tilted answers the
   weather question).
2. **Regime moves** in non-centered coordinates change x through A^m, c^m, Q^m,
   so they need F-differences at λ > 0. Use FFBS in centered coordinates
   (m | actual z₀ and details; x unchanged) interleaved with non-centered pCN
   (ASIS).
3. **MLMC coupling**: "coarse v = restriction of fine v" samples R_#π_ℓ, not
   π_{ℓ−1}, and biases the telescope. Couple through the **nested weather**
   x_ℓ = (x_{ℓ−1}, w_ℓ); v-chains need coupling only for variance (prolongation
   warm start, common random numbers). MLMC uses increments linearly, so the
   extensivity that kills MLDA is harmless; level variances decay only if the
   coarse energy model converges.
4. **F values** (tilting by reweighting, λ-ladder, cost map) need more than
   gradients: trajectory TI, or **TI across resolution**,
   dF/ds = E_s[C^{(L)}(v) − C^{(0)}(Rv)] along C_s = (1−s)C^{(0)}∘R + sC^{(L)},
   anchored by one cheap coarse LP. Required accuracy is loose (≈ 10⁸–10⁹ €).
5. **Time-scale separation**: a coarse weather move shifts the whole year's
   storage schedule, so one effective weather sample costs ≥ τ_global sweeps —
   to be compared with a warm-started LP re-solve. π(v|x,k) is log-concave
   (no metastability, only conditioning); storage moves must be active-set
   adapted (shifts between consecutive pinned instants). Lag diagnostic:
   dissipated work (Crooks forward/backward).
6. **Local MLDA** is valid but fails by mean shift at pinned blocks (E3
   confirms); line sampling along hierarchical directions under the fine cost
   is the right use of the hierarchy.
7. §0 is **double intractability**; exchange algorithm, pseudo-marginal 1/Z
   estimators, replicas, fractional powers and changes of variables do not
   remove it. §2 is an approximate-exchange / double-MH scheme (bias bounds:
   Alquier et al. 2016; Medina-Aguayo et al. 2016).
8. Metropolis via integrated mean force is the Ceperley–Dewing penalty method:
   exact only for unbiased Gaussian noise; treat as approximate. For k use
   stochastic-gradient Langevin (bias damped by β_k²) rather than HMC.
9. Proposed **E0**: ΔF between two exact weather years by TI along a straight
   x-path with the fast chain, against the LP difference.

## 8. Relaxing after a coarse move (question from the user)

Letting the fine chain relax after a coarse move *before* the accept/reject
makes the fine proposal depend on the coarse move and invalidates the MLDA
ratio. The exact version is **non-equilibrium candidate Monte Carlo** (NCMC,
Nilmeier–Crooks–Minh–Chodera 2011; Jarzynski/AIS): switch from the old to the
new coarse state through intermediate targets with fine relaxation sweeps in
between, accept with exp(−W) for the accumulated work W. Acceptance tends to
exp(−W_diss) → 1 for slow switching. Dissipated work is extensive like Δ, so
global coarse moves need switching times growing with the number of blocks;
local (colour-parallel) moves are cheap. E3's ≈ 10³ nats per coarse coordinate
at instantaneous switching and E1's relaxation times give the scale: per-block
switching over tens of sweeps would be needed. The relaxer should be exact line
Gibbs (gradient methods collapse on the kinks).

## 9. What the evidence says so far

* The fast block is the bottleneck for every variant (mean forces, MLMC,
  NCMC, TI): its global modes relax over ≫ 100 sweeps at 40 nodes × daily.
  Improving it (hierarchical directions under the fine cost, active-set storage
  shifts, flow re-routing, prolongation warm starts) comes first.
* MLDA with coarse models that do not see the fine active set cannot accept
  coarse moves, regardless of depth or locality; the hierarchy should enter as
  move directions (fast block), nested-weather MLMC (forces) and TI across
  resolution (F values).
* Before building the slow dynamics, E0 and a direct τ_global measurement
  decide whether the fast chain can beat a warm-started LP re-solve per
  weather sample.

## 10. Results of R1–R4 (relaxation and the LP baseline)

Scripts `scripts/exp_relax.jl` (R1–R3) and `scripts/exp_lp_warm.jl` (R4);
40 nodes × daily, α = 0, weather and capacities at the LP's, 8 threads.
Predicted equilibrium opex 1.791·10¹⁰ €/a (LP 1.700·10¹⁰ + 9.1·10⁸).

| | result |
|---|---|
| R1, far start (30 % push), 10 min each | plain coordinate Gibbs: 197 sweeps, opex 2.61× target; + temporal multigrid directions on the 10 coarse spatial modes: 34 iterations (≈ 7 sweeps each), 3.13× target. Multigrid directions in time did not speed up relaxation per wall time. |
| R2, LP start, multigrid kernel, 10 min | opex 1.910·10¹⁰ ± 0.8·10⁸, excess 2.3× the equipartition prediction (still approaching) |
| R3, +3σ network-mean load for 14 days | opex jump 9.9·10⁹; back within 10 % of the jump after 136 s; chain Δopex 1.73·10⁹ |
| R4, operational LP at fixed capacities | cold dual simplex 21 s; **warm re-solve after the same load change 1.1 s**; LP Δopex 2.95·10⁸ |

Consequences:

* At this size a warm-started LP re-solve is about 100× faster per weather
  sample than relaxing the fast chain, and the chain's estimate of the cost
  change is about 6× too large (it has not equilibrated). The fast block, not the
  evaluation of the target, is the bottleneck.
* Moves along fixed coarse directions (spatial or temporal multigrid) do not
  remove that bottleneck; the toy analysis (docs/toy_mlda.md §4–§5a) explains
  why: directions that ignore the active set push pinned coordinates into the
  penalties. Active-set-aware moves (transport along the merit order, as in
  toy §5a) are the candidate.
