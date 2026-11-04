# Plan: from proof of concept to an H100-scale sampler

Status and gaps are recorded in [mathematical_formulation.md](mathematical_formulation.md)
(§9) and [implementation.md](implementation.md) (§9). The target design is
[near_optimal_sampling_setup_v2.md](../../near_optimal_sampling_setup_v2.md). This
plan orders the remaining work so that every stage has a checkable result
before the next one builds on it.

Guiding decisions (already taken):

* Constraints are **regularized** (priced with physical slopes), not hard walls.
* β is **fixed by experience**, not calibrated.
* Base sampler: **exact/tuning-free Gibbs-type moves**, not HMC.
* Spatial layers: **local graph-Haar partitions** (+ optional few large-scale
  modes at level 0), time layers: block averages / hierarchical surpluses.
* Target hardware: **H100** (FP64, 80 GB, multi-GPU) via **Reactant.jl**; the
  current CPU Julia code becomes the reference implementation.

---

## Stage 0 — Freeze a reference and a test ladder (CPU, small)

Goal: a set of small problems with known answers that every later version must
reproduce.

| test problem | known answer | checks |
|---|---|---|
| T0: fixed (k, x), operation only, 3 nodes × 1 week, daily → 6 h | exact: brute-force 1-D quadrature is impossible, but E[C] − C* ≈ n_conf/β_v at large β and the marginal of single coordinates against long single-site Gibbs runs | operational kernel correctness |
| T1: same, 40 nodes × 1 year daily | LP optimum + equipartition excess; agreement of two independent kernels (coordinate Gibbs vs. directional Gibbs) | operational kernel at scale |
| T2: k free, x fixed (perfect foresight), α irrelevant | LP capacities as β_k, β_v → ∞; ρ_K at moderate β by long runs | investment moves |
| T3: weather only (no energy model) | exact samples from the weather model | weather moves, FFBS, §6.4 diagnostic machinery |
| T4: toy linear-Gaussian "energy model" with analytic log Z(x,k) | exact b(x,k) and ρ_1 | bias learning end-to-end, α = 1 |
| T5: 2-level and 3-level toy hierarchies (exists for MLDA core) | exact Gaussian | MLDA, estimator |

Deliverables: `test/ladder/` with T0–T5, a script printing pass/fail and
effective sample sizes; unit tests for Gibbs step-term consistency, sweep
reversibility (detailed balance on a tiny discrete-valued version), Haar
basis, weather details, bias regressions, LP conversion, scaling move.

## Stage 1 — Fix the statistical core on CPU

1.1 **Bias with safe tails.** ĝ as −β_v Δt D̄ Σ_t ∫ p(u) du with p
non-decreasing from −c_dump to c_shed (non-negative ramp weights, fixed sum),
fitted by non-negative least squares on the scores; add block-local (country)
features; ĥ on clipped coarse capacities with a curvature prior. Check on T4
and on the α = 1 weather diagnostic (regime frequencies and coarse moments
within Monte Carlo error of exact draws).

1.2 **Exact line sampling.** Replace slice sampling by exact inversion of the
piecewise-exponential 1-D conditionals: collect breakpoints of all terms the
coordinate touches (clamps, residual sign changes, penalty kinks), evaluate
the affine pieces, sample segment by CDF, then within the segment. Check
against slice sampling on T0/T1 (same marginals).

1.3 **Direction set** (v2 §5.1): unit pairs at a node (balance-preserving),
flow re-routing (θ move + compensating injections), intertemporal storage
shifts, investment–dispatch moves per node in physical coordinates with the
k^{−n} Jacobian. Random scan with tuned direction weights. Check ESS/s
against coordinate Gibbs on T1/T2.

1.4 **Weather moves in sheared coordinates** (ṽ = v − S x) with elliptical
slice sampling on the Gaussian weather blocks. Check on T3 (exact) and T1
with free weather (mixing of coarse weather coefficients).

1.5 **Hierarchy** per v2 §4: graph Haar levels (network → countries → areas
→ nodes; areas from spectral bisection), optional level-0 Laplace modes,
L-orthogonal θ splitting, genuinely aggregated coarse network models (coarse
Laplacian, cell-summed balance, per-class capacities), optional Gaussian level
priors (ε, κ) and pCN on all fine coefficients. Check (A1)-type consistency
and MLDA acceptance per level on T1-sized problems.

1.6 **Fine-level warmup and adaptation.** Per-level slice/direction scales;
initialize fine levels by prolongation + fixed-level sweeps until the level's
cost trace is stationary; then switch on MLDA. Check: finest-level cost within
equipartition excess of the LP at α = 0 with fixed k (T1).

1.7 **Estimator and diagnostics** in the driver: R̂ over chains, ESS,
multilevel estimator with confidence intervals, §6.4 diagnostic with a
pass/fail threshold, nodal prices as quantities.

Exit criterion for Stage 1: on the 40-node × daily → 6 h problem, α = 1
chains pass the weather diagnostic, finest-level MLDA acceptance > 0.05,
R̂ < 1.05 for capacity totals and total cost over 8 chains.

## Stage 2 — Reactant.jl implementation (GPU-ready)

2.1 **Data model**: one `Model` of plain arrays — fields padded to all nodes
with masks, shape (chains × fields × nodes × steps), FP64; index arrays for
lines, links, cells; Haar parent maps per level; stacked weather detail
parameters indexed by group.

2.2 **Traced kernels** (pure functions, one compiled set per level):
cost and per-block Δ; Haar restrict/prolong; Enzyme gradients for the scores
(replacing hand-written gradients); batched colour-class sweep with exact line
sampling; direction moves; weather detail synthesis; ĝ/Ēĝ.

2.3 **Host orchestration**: MLDA recursion (`mlda_core` interface kept),
adaptation, FFBS (or a small traced kernel), bias regression on batched
scores, I/O.

2.4 **Equivalence tests** against the CPU reference on the Stage-0 ladder
(bitwise-close costs and gradients; statistically equal samples).

2.5 **Performance**: measure cost evaluations/s and sweeps/s with 1, 16, 64,
256 chains on CPU and H100; check XLA compile times per shape set; keep the
number of distinct shapes small.

## Stage 3 — Scale data and model

3.1 **Network**: full PyPSA-Eur (~250 nodes, or the 91-node example with all
countries), all electricity-sector technologies; optionally lines and storage
ratios as investments (finite-difference boundary terms, §v1 6.2).

3.2 **Weather**: ERA5 (via atlite, needs CDS credentials) for 40–60 years at
node level, replacing NASA POWER; hourly resolution; load anomalies beyond
temperature; held-out validation (regime persistence, correlations per level,
Dunkelflaute frequency); interannual factor; random-effects variant (v2 §3.5)
as an alternative.

3.3 **Initialization at scale**: LP only at a coarse level (or GPU PDLP),
fine levels reached by prolongation and sampling.

3.4 **Multi-GPU**: chains sharded across 8 GPUs; optional time-axis sharding.

## Stage 4 — Science outputs

ρ_K marginals and near-optimal trade-offs; α = 0 vs α = 1 comparison;
weather-cost map and per-regime Dunkelflaute classification; perfect-foresight
variant (§v1 1.6); sensitivity to β, c_shed, c_dump.

---

## Dependencies and order

Stage 0 → 1.1, 1.2 (independent) → 1.3, 1.4 → 1.5 → 1.6 → 1.7 → Stage 2
(2.1 can start after 1.2) → Stage 3 → Stage 4. Stage 2 should not start before
1.2, because the kernel shapes depend on the line-sampling design.

## Main risks

| risk | mitigation |
|---|---|
| Fine-level MLDA acceptance stays ≈ 0 even with regularized Ω | fixed-level sweeps carry the fine chain; ε > 0 priors and aggregated coarse models to reduce Δ variance; fall back to a single-level GPU sampler at the finest level with multilevel only for bias learning |
| Bias learning at α = 1 unstable | monotone-price ĝ; T4 toy first; batched scores from many chains |
| Reactant/XLA limits (scatter, sort, while on GPU; compile time) | spike in 2.2 on the cost kernel first; keep shapes few; fallback to KernelAbstractions.jl/CUDA.jl kernels |
| Exact line sampling expensive (many breakpoints per coordinate with dense modes) | Haar locality keeps breakpoints O(local units) |
| Data access (ERA5 credentials) | NASA POWER as interim; 22 years |
