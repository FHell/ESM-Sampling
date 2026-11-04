# Model completion (todo items 17, 18, 19)

The remaining model items, how each is implemented, and why each stays exact for
the sampler. All new terms are options; with the defaults the model is the one
used so far.

| config key | default | meaning |
|---|---|---|
| `co2_price` | NaN (the data's 150 €/t) | canonical CO₂ price μ (€/t) |
| `co2_cap` | 0 (off) | mean-cap tuning of μ during the finest-level warmup, then frozen (§1) |
| `brownfield` | false | existing capacities as floors, merged vintages, fixed thermal fleets |
| `losses`, `loss_segments` | false, 4 | quadratic line losses (piecewise-linear interpolant) |
| `thermal`, `thermal_band` | "hard", 0.1 | "soft": convex overload band above the rating |
| `link_losses` | false | signed lossy DC links |
| `uc` | [] | carriers with fleet unit commitment, e.g. ["gas", "coal", "lignite", "nuclear"] |
| `fine_weather` | 0 (off) | fraction of the weather coordinates moved per fine-level step |

The data files gained the needed arrays additively (`scripts/prepare_data.py --augment`;
existing arrays unchanged, backups in `data/processed/backup/`): line loss
coefficients α (from r/v_nom², merged parallel lines weighted by the flow split),
link efficiencies (0.98 · 0.977^(length/1000) for the lossless PyPSA-Eur DC links),
existing capacities k0, old-vintage marginal costs and emission factors, and the
brownfield fixed fleets of the 256-node network (coal 41 GW, lignite 38 GW, oil,
biomass, waste, geothermal). Cache keys of the weather model and the LP are now
content fingerprints, so additive data changes do not invalidate them.

## 1. Canonical CO₂ price (item 17)

Every generator carries its emission factor (t/MWh_el, per vintage) and its marginal
cost without CO₂; the effective cost is mc = mc_fuel + μ·ef (`set_co2_price!`). This
is the canonical ensemble of sampling_formulations.md §6: local, parallel, emissions
fluctuate across weather years. Annual emissions are reported (`CostParts.co2`,
quantity `co2_t`). Tuning μ to a mean cap is a stochastic approximation of the target
itself (the target changes with μ), so it runs during the finest-level warmup only:
μ ← μ exp(0.5 n^−0.6 clamp(E_n/Γ − 1, ±1)), then frozen for production.

## 2. Merged co-located vintages and brownfield (item 17)

With `brownfield = true`:

* **Investable units:**
  - existing capacity k0 is the lower box of k (no retirement);
  - capex is charged on k − k0 (sunk existing capacity);
  - the upper box is raised to at least 1.1·k0.
* **Merit order:** output is dispatched cheapest vintage first. The cost of output p is
  convex piecewise linear, C(p) = c_lo min(p, A q_lo) + c_hi (p − A q_lo)₊ (`gen_vint`).
  It is the zero-temperature merit order of the two co-located components and
  replaces their flat split (E3). Emissions follow the same split.
* **Sampler:** the kink is one more primary breakpoint of the generator entries (also
  on both sides of a T2 kink). The capacity moves (`kscale`, `kpeak`) evaluate the
  kinked cost with the kink moving with k.
* **Fixed thermal fleets** are added as further carriers (fixed k, own marginal cost
  and emission factor).

## 3. Line losses and soft thermal limits (item 17)

Both are functions of the relative flow u = |f|/s on one shared grid
(`LineTerms`):

* **Losses:**
  - α s² L(u), with L the piecewise-linear interpolant of u² on a uniform grid of
    `loss_segments` cells per rating (exact at the grid points, linear beyond 1.5 s);
  - drawn half at each end of the line, so they enter the nodal balance;
  - interpolation error ≤ α s²/(8K²).
* **Soft limits:** the marginal overload price rises linearly from 0 at s to c_viol at
  (1 + band)·s, then stays at c_viol (ψ interpolated on `thermal_segments` points of
  the band). A convex overload cost instead of the c_viol wall.

Both are convex and piecewise linear in f, so the exact piecewise-exponential line
sampler stays exact; their kinks (f = 0 and ±U_j s) are primary breakpoints. Exact
quadratic losses would need truncated-Gaussian pieces in every flow move and a
quadratic residual. The interpolant keeps the sampler and the GPU design unchanged
and is a modelling choice of the same kind as PyPSA's secants, at a much finer grid.

**Signed lossy links.** One signed flow u per link, losses (1 − η)|u| at the
receiving end (`link_inj`). The kink at u = 0 is a primary breakpoint.

**Requirement-tracking kinks.** The exact T2 compensation of transport moves keeps
the shares σ and the excess e = P − Q₊. It needs the actual requirement Q along the
move. With losses or lossy links that requirement is piecewise linear in t, so
Q_k(t) = Q_k + Δr_n(t) is evaluated from all entries except the compensating units
(two-pass `eval_move!`). Its sign change is a secondary breakpoint, and so are the
clamps of the compensating units. A kink of one secondary quantity bends others, so
for these moves the walk follows events in order: one-sided slopes from a probe
step extrapolate every secondary exactly up to the first kink. The Jacobian uses
P(t) = P + φ_k(t) from an evaluation at the drawn t. Tests: sampled line draws
against quadrature (KS < 0.007 at 40 000 draws), incremental costs against the
global evaluation.

## 4. Fleet unit commitment (item 17)

PyPSA's linearized UC per committable carrier:

* **Commitment field:** c = κ/k ∈ [0, 1], an instant field (one value per step,
  coupled to the next step).
* **Output:** y ∈ [g_min c, c], with the walls priced at c_viol.
* **Costs:** no-load cost a·A k c per hour; start-up cost s·A k (c_{τ+1} − c_τ)₊.
* **Parameters:** (g_min, s, a) default per carrier (`UC_DEFAULTS`: gas 0.3/40/4, coal
  0.4/80/3, lignite 0.5/100/3, nuclear 0.5/150/1, …). They are assumptions; PyPSA-Eur
  has no UC data for these fleets.
* **Moves:**
  - atoms of the commitment field move it at fixed output;
  - in every other move of a committable unit at the main step, the commitment follows
    the output with the headroom kept (`ws.follow`; for T2 moves the same kinked map,
    so the invariants and the Jacobian are unchanged);
  - the linked previous step of the SoC form does not follow, so parallel classes stay
    independent;
  - capacity moves scale the commitment with the output (Jacobian counts both fields).
* **Physical measure:** counts the commitment coordinates.
* **LP:** linear relaxation with committed capacity, start-up and no-load terms.

## 5. Balance frame (item 17, first bullet)

"Balance eliminated by a dependent unit, or the full transport-first layout": both
are reparametrizations (sampling_formulations.md §1, §8). The sampler realizes them
in the move frame instead of the state layout. Exact T2 moves keep (σ, e) and are
the transport-first coordinates as moves. The node-value state stays, which is also
the layout of the Reactant design. Nothing further to implement.

## 6. Fine-level weather moves (item 18)

With MLDA shelved, the fine chain must move the weather itself (before, the fine
weather moved only through MLDA proposals with acceptance ≈ 0). `src/weather_fine.jl`:

* **Regimes, centered.** With the anomalies of every level fixed, m ~ p(m | x) by FFBS.
  The emissions are the level-0 VAR terms plus, for each level-0 step, the detail
  densities N(ξ; Aρ, LLᵀ) of all its descendants. The innovations are then recomputed
  as w = L⁻¹(ξ − Aρ). Cost and bias are unchanged.
  - *Observation:* given the fine weather the regimes are almost exactly identified.
    In the tests no regime changed, so regime mixing at the fine level is slow.
  - *Follow-up if needed:* a non-centered regime move (w fixed, x changes, MH with the
    cost).
* **Innovations of every level and level-0 anomalies, non-centered.** One coordinate
  moves by t.
  - *Propagation:* the change of the finest anomalies is linear in t (exact windowed
    propagation through the detail regressions, ± one coarse step per level), so the
    raw node weather is affine in t.
  - *Weather entries:* the data-map kinks (load ≥ 0, availability ∈ [0, 1]) are primary
    breakpoints, and the residual and renewable output change follow in the move.
  - *Compensation:* dispatchable generators at every affected node follow with exact
    T2 on the actual requirement (requirement-tracking kinks). Weather-dependent units
    are excluded, since their κ changes with the weather.
  - *Prior:* exact on the line with the Gaussian term: w ~ N(0,1); z0 has the VAR terms
    of its step and the next, a quadratic in t (`z0_quad`). The segments are truncated
    Gaussians (`sample_segments_gauss`).
  - *Correction:* the T2 Jacobian and, for z0, the bias change enter an independence
    Metropolis step.
* **Parallelism:** moves in blocks j with equal j mod M (M from the window spread)
  run in parallel; z0 moves with the learned bias run in sequence.
* **Use:** `fine_weather > 0` (kernel = "local") adds a weather sweep to every
  fine-level step. Without MLDA set `p_fine = 1`.
* **Tests (`test/test_model.jl`):**
  - incremental cost against the global evaluation;
  - caches, the non-centered representation and the regime step reproduce the anomalies;
  - line draws against quadrature with the prior;
  - z0 quadratic against `logp_z0`;
  - invariance of the weather kernel under p(m, z0, w), from exact draws (§6.1).

### 6.1 Invariance test

Start from many independent exact draws, apply two weather sweeps (β_v = 0, no
compensation), and compare statistics with fresh exact draws. This holds whatever the
mixing speed. A single long chain is not a usable check here: the annual statistics
of the AR(1) weather have ESS 3–9 over 250 sweeps. Result: see the run log in the
final report of this round.

## 7. Aggregated sampled prices (item 19)

`RegionPriceAcc` (diagnostics.jl): per sample and per (region, meteorological season)
the mean marginal price, the load-weighted price, and the shortage and surplus
frequencies. Regions are countries plus the whole system. The driver writes means,
standard errors (from the ESS) and the traces to `prices/regions/<country|system>/…`.

## 8. LP

The LP is a start and a small-model reference only. It includes vintages, lossy links,
the CO₂ price and the relaxed UC. It excludes losses and soft limits: the LP keeps hard
limits and is lossless. With UC it is slow (312 s at 10 cells × daily), so it is not
developed further. Large models use the heuristic start.
