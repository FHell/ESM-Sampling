# Mixing diagnosis of the CPU sampler (40 nodes, daily)

Report of an overnight diagnosis (2026-11-03/04) of why the 4-chain screening
`configs/chains40.toml` does not converge (max R̂ 4.6, bulk ESS 4–18 after 100 sweeps). It
describes experiments and prototypes; it is a development note, not a description of the
system (`docs/system/` is). Scripts are in `scripts/mixing/`, outputs in `results/mixing/`.
Two small changes in `src/` were made after the legacy removal was merged (commit bbbb97c), as
agreed with the main session: the `kpeak` slice-width fix (§6, `src/adaptive.jl`) and the
storage multiscale sweep in the `local` kernel (§6.2, `src/local.jl`), with their
`docs/system/` updates (and the outdated test-margin notes in `testing.md`/`roadmap.md`);
the runs before that merge (baseline, `shared_base`) used the old code, `own_kop` the new code
with the old `kpeak`, all later runs the new code with the fix.

**Dependence on the old bias.** Everything except §4.4 and §5.4 is at α = 0 or under the weather
prior alone and does not depend on the bias. §4.4 and §5.4 use the level-0 bias of the code
before the bias rewrite (branch `bias-merge`), learned in a shortened warmup and frozen; that
bias will be gone, so those two sections describe what a fast capacity move and the weather
moves do with *a* poor bias, not with the new one, and need repeating.

## Summary

* **The screening measured a transient.** At α = 0 every chain was still relaxing after the
  100 sweeps of the screening, and still after 600 (§2): gas was eliminated (269 → 6 GW),
  batteries ran into their box (465 GW), H₂ kept growing, renewables froze in chain-specific
  states (between-chain sd 0.5–0.65 in log k per unit), and the weather was locked in
  chain-specific states up to 18 prior sd from the prior.
* **Why the capacities do not move (§3).** `kscale` moves a capacity at fixed physical
  dispatch. For a renewable unit this hits the node balance at every step: dumping upward
  (0.26 nats/MW), shedding downward at the step with the least slack; the conditional is
  ≈ 4 MW wide whatever the unit's size (the shed-limited downward side also at α = 1), a
  median 1.6–6.5e-4 in log k per sweep. For
  stores and gas it allows an overshoot of 1/(β_k c) ≈ 44–570 MW above the current use,
  and the operation can only grow into it: a ratchet of about one capex scale per sweep.
  Mean forces on the log capacities are large (≈ 3e4 nats per unit log-scale of the wind
  carriers at sweep 600): the target is strongly sloped, not flat; the drift is slow
  because of the moves.
* **Where the α = 0 target goes is entropic (§4.2).** The zero-temperature α = 0 optimum (LP,
  capex × 1e-3) has 0.3 GW of batteries, the chains 465 GW. Under the physical measure each
  store gets a force n_free/k from its unconfined SoC volume, 15× the capex slope at 10 GW;
  under the utilization measure the force −n_conf/k drives storage toward zero and splits
  the chains (11–223 GW of batteries, H₂ at the κmin floor in 3 of 4 chains). α = 0 with
  β_k/β_v = 1e-3 is not a useful screening target (the main session has meanwhile marked
  `chains40.toml` as a mechanics test).
* **`kop` (§4)**: a new capacity move at fixed utilizations, with the operation of the unit
  scaled and the node's dispatchable generators following share-preservingly (exact,
  checked). At α = 0 it shortens the transient 2–4-fold and halves R̂ of storage,
  solar and capex at equal CPU, but it cannot move renewables in a gas-free system, where
  nothing follows a capacity change on the critical days; neither can a capacity transport
  pair with the power flow following (§4.3).
  At α = 1 with the old level-0 bias it exposed that bias at once: ĥ had negative curvature,
  and one `kop` sweep moved solar to 4.8 TW and b̂ by 2.7e8 nats (§4.4); the capacity part of
  any bias must be confining before wide capacity moves can be used there (to be rechecked
  with the rewritten bias).
* **Coarse weather (§5).** Under the prior alone the fine sampler's single z0 moves need at
  least 1.6e5 sweeps per independent annual load mean (the 10 load rows are collinear,
  R² = 0.994; persistence 0.8–0.93) and 470–930 sweeps for the wind and solar means; shifts
  of 16–365 days along the joint principal directions raise the rates by 2–5 orders of
  magnitude per move and 6–700× per CPU second. Under the cost they are as confined as the
  single moves (α = 0: a step shrinking like 1/B). The regimes are identified by z0 alone
  (0.1–0.5 % uncertainty per day under the prior), so no regime move that keeps z0 can help.
* **Seasonal storage operation (§6.2).** In the `local` kernel the H₂ state of charge is moved
  only by single-step moves and stayed at a mean of 0.03–0.09 for 300–600 sweeps in every run;
  with the storage multiscale sweep it relaxes to 0.28–0.31 within 100 sweeps at 8 % of the
  CPU time. Now part of the `local` kernel in `src/`.
* **A bug fixed in `src/`**: `kpeak` used a slice width proportional to the current
  capacity, which breaks the reversibility of the stepping-out rule (§6). The end-to-end
  exactness test passes with either version, i.e. it is not sensitive to this.

## 1. Setup and tools

* **Runs.** `scripts/mixing/driver.jl` mirrors `run_chains` at α = 0 (same starts ×0.5, ×2,
  heuristic, heuristic; same samplers; the `fine_local_step` structure of the `local`
  kernel) and records per sweep and chain: all 219 node capacities, the change of every
  log-capacity by the `kscale` pass and by the `kpeak` pass, the 36 z0 row means and their
  seasonal means, regimes, cost parts, the 84 monitored quantities and the wall time of
  every component. Extra (prototype) moves are appended to each fine step.
  `scripts/mixing/run_exp.jl <name> <sweeps> <extras> key=value…` runs a variant
  (configuration overrides of `chains40.toml`, prototype moves).
* **Forces** (`scripts/mixing/forces.jl`). For a unit with u = log k the derivative of the
  log marginal of u is the expectation over π(v | x, k) of the derivative of the line
  log-density at u = 0, in either parametrization of a capacity change: `kscale` (physical
  dispatch fixed, group Jacobian (1 − n)u, measure term) or `kop` (utilizations fixed,
  pro-rata compensation, §4). Both expectations must agree when the operation is in
  equilibrium given the capacities. Decomposed into capex, measure, cost and Jacobian parts.
* **Expected squared jumps** (`scripts/mixing/prior_spectral.jl`). For a Gaussian target with
  precision Λ and exact conditional draws along directions d_j (chosen uniformly), the
  expected squared jump of a linear functional aᵀx per move relative to twice its variance is
  rate = mean_j (d_jᵀa)²/(d_jᵀΛd_j) / (aᵀΛ⁻¹a). For a reversible chain the integrated
  autocorrelation time of aᵀx is at least (1 + ρ₁)/(1 − ρ₁) ≈ 2/rate moves (Jensen on the
  spectral measure), so 2/rate is a rigorous lower bound on the moves per independent sample.
* **Files.** `scripts/mixing/`: `common.jl` (setup), `driver.jl` (instrumented chains),
  `run_baseline.jl`, `run_exp.jl`, `run_a1.jl`, `run_a1w.jl`, `run_a1_weather.jl` (runs),
  `kop.jl`, `kxpair.jl`, `wblock.jl` (prototype moves), `forces.jl`, `landscape.jl`
  (forces, thermodynamic integration along a carrier; not run, the single-point forces
  sufficed), `prior_spectral.jl`, `prior_mixing.jl` (weather prior), `weather_esjd.jl`,
  `analyze.jl`, `check_moves.jl`, `test_exact_oldkpeak.jl`. Outputs in `results/mixing/`
  (`<run>.h5` traces, `<run>_chains.jls` end states, `<run>.log`).
* **Consistency checks** of every prototype (`scripts/mixing/check_moves.jl`, and `check = true`
  in the moves), in the style of `test/test_local.jl` and `test/test_flow.jl`: incremental
  log-target change against the global `evaluate!`, caches against a rebuilt `LocalCost`,
  anomalies against a fresh `weather_at_level`, and a curve check (from the moved state the
  move with −u returns exactly to the start, its line log-density is the negative of the
  forward one, the pro-rata coordinates (σ, e) are unchanged along the curve). All pass to
  round-off (cache differences ≤ 1e-10 MW, log-target differences below 1e-6 relative).

## 2. The baseline: 600 sweeps of the screening configuration

`scripts/mixing/run_baseline.jl`: `chains40.toml` (α = 0, local
kernel, fine weather 0.05, own weather draw per chain), 4 chains × 600 sweeps, 0.84 CPU s
per chain and sweep on 4 threads (34 min). Time shares: operation sweep 69 %, `kpeak` 14 %,
fine weather 14 %, `kscale` 3 %.

**The screening was a transient, not a mixing measurement.** Over the 100 sweeps of the
original screening every chain was still relaxing in the same direction (opex fell from
1e11 to 3e9 €/a, batteries grew 30×, H₂ 200×, gas fell). After 600 sweeps (window 301–600):

| carrier | `kscale` median \|Δlog k\| per sweep (rms) | `kpeak` | spread of log k between chains (sd, median unit) | R̂ of the total | drift of log total per sweep | totals per chain (GW) |
|---|---|---|---|---|---|---|
| onwind | 2.5e-4 (3.5e-3) | – | 0.50 | 4.0 | +5e-5 | 500, 358, 362, 300 |
| solar | 1.6e-4 (5.3e-2) | – | 0.65 | 4.6 | −3e-5 | 1810, 1420, 1250, 1290 |
| offwind | 6.5e-4 (0.15) | – | 0.61 | 4.4 | +7e-5 | 66, 49, 65, 46 |
| gas | 0.11 (0.43) | 3.5e-3 | 0.39 | 2.0 | −7e-3 | 6.7, 6.8, 5.6, 5.4 |
| battery | 2.1e-3 (7e-3) | – | 0.007 | 2.9 | +1.2e-4 | 469, 457, 458, 449 |
| H₂ | 4.7e-3 (0.015) | – | 0.24 | 2.8 | +7e-4 | 168, 137, 159, 141 |

(Start: onwind 181, solar 909, gas 269, battery 13, H₂ 0.5 GW for the heuristic chains.)
The worst R̂ on the window are regime 4 (9.1), the solar total (4.6), per-country onwind
and solar (4.1–4.6), the z0 annual means (4.2–4.6), the mean line loading (4.3) and the
capex (4.1); all bulk ESS are 4–5. What happened:

* **Gas was eliminated** (269 → 6 GW in every chain) and **batteries ran into their upper
  box** (465 GW in total; 449–469 GW, between-chain spread 0.7 %). H₂ is still growing
  (+0.07 % per sweep).
* **Renewables froze.** A `kscale` move of a renewable unit changes its log capacity by a
  median 2–6e-4 per sweep, while the four chains differ by sd 0.5–0.65 in log k per unit.
  A random walk with that step needs ~(0.5/3e-4)² ≈ 3e6 sweeps (four weeks of CPU per chain)
  to cross the spread. The
  device run (`results/device40.h5`) shows the same: with Metropolis proposals of scale
  0.3 in log k, renewable capacities in three of four chains did not change at all.
* **The weather is locked** in chain-specific, strongly tilted states: the annual means of
  the z0 field-mean rows at sweep 600 are, per chain (load, onwind, offwind, solar):
  (0.28, −0.98, −1.16, −1.01), (−0.12, −1.18, 1.51, −0.31), (−0.59, −1.03, 0.36, −1.43),
  (−0.43, −0.66, 1.92, 0.56), against prior standard deviations of (0.38, 0.23, 0.11, 0.34):
  up to 18 prior sd from zero (offwind), and almost unchanged between sweeps 300 and 600.
  Regime 2 has a frequency of 84–95 % in every chain against 31 % under the prior.

## 3. Why the capacity moves do not move the capacities

### 3.1 Renewables: the fixed-dispatch conditional is a few MW wide

`kscale` moves a capacity at fixed physical dispatch. A renewable unit treated as load has
no dispatch coordinate, so the move changes its node's balance by A_τ δ at every step. The
residual of a balanced node fluctuates thermally on the dump side with scale
1/(β_v c_dump Δt) = 208 MW and on the shed side with 1/(β_v c_shed Δt) = 0.35 MW. Hence:

* upward the extra output is dumped, β_v c_dump Δt Σ_τ A_τ ≈ 0.26 nats per MW for solar
  (Σ A ≈ 55 full-load days), a scale of ≈ 4 MW;
* downward the move hits shedding (3000 €/MWh) at the step with the smallest dump slack
  relative to A_τ, the minimum of 365 slacks of mean ≈ 208/A_τ MW, again ≈ 4 MW.

The width is independent of the unit's size (10–30 GW): |Δlog k| ≈ 4 MW / 20 GW = 2e-4,
exactly the measured median 1.6–6.5e-4. The downward limit does not depend on α either (the
bias is smooth in k, the shed kink is not); at α = 1 the upward dump slope is partly
cancelled by the bias, which can only widen the upward side. At α = 0 the marginal has a strong slope (§3.3), so the unit drifts at about
σ²f/2 per sweep (σ ≈ 4 MW, f ≈ 0.08 nats/MW for onwind at the baseline end state), i.e.
≈ 0.6 MW per sweep: thousands of sweeps per GW. At α = 1 the marginal width of a
renewable unit is set by capex at β_k, 1/(β_k c) ≈ 240 MW (onwind) to 600 MW (solar), so
the same 4 MW steps would need (600/4)² ≈ 2e4 sweeps per independent sample.

### 3.2 Dispatchable capacities and storage: the ratchet

For a dispatchable unit or a store, `kscale` keeps the physical dispatch (the stored energy)
fixed, so k can move freely above what the dispatch uses, with density ∝ exp(−β_k c k)
under the physical measure (the measure term cancels the group Jacobian up to
n κ0/(k + κ0)): an exponential overshoot of 1/(β_k c) = 570 MW (gas), 415 MW (battery),
44 MW (H₂) above the current peak use, and below it the box penalty. The operation can
then only grow into the new capacity. Capacity and operation advance together by at most
about one capex scale per sweep, whatever the force: the measured battery growth during
the transient was ≈ 100 MW per unit and sweep (4 GW per sweep in total), H₂ ≈ 20 MW per
unit and sweep. Gas fell from 269 to 6 GW over 600 sweeps by the same ratchet as the
storage took over its peak dispatch. `kpeak` (absorb variant, the only one in the `local`
kernel) moved gas by a median 3.5e-3 in log k per sweep and costs 14 % of the CPU time.

### 3.3 The forces: the target is strongly sloped, not flat

Mean forces dℓ/du on the common log-scale u of a carrier at the end state of a baseline
chain (sum over its units; mean over 9 operation states at fixed (k, x), two sweeps apart;
± standard error; `forces.jl`), in both parametrizations:

| carrier | `kscale` parametrization | `kop` parametrization | sd of the force between operation states |
|---|---|---|---|
| onwind | +2.8e4 ± 1.3e4 | +3.1e4 ± 1.0e4 | 3–4e4 |
| offwind | +3.9e4 ± 1.0e4 | +2.4e4 ± 0.4e4 | 1–3e4 |
| solar | −0.6e4 ± 1.8e4 | +1.1e4 ± 1.1e4 | 3–5e4 |
| battery | +3.7e3 ± 3.4e3 | +5.7e3 ± 4.0e3 | 1e4 |
| H₂ | +1.2e3 ± 6.5e3 | +1.5e4 ± 0.3e4 | 1–2e4 |
| gas | +30 ± 60 | −90 ± 40 | 100–200 |

A force of 3e4 per unit u means that scaling all onwind units by 1 % raises the log
marginal by ≈ 300 nats: after 600 sweeps the wind capacities are still far below where the
α = 0 target puts them, and the drift is blocked by the moves of §3.1. The force at fixed
operation fluctuates by 3–5e4 between operation states (the kinked balance), which is the
same roughness that confines the line moves. The two parametrizations agree within the
errors except for H₂, whose operation (seasonal state of charge, moved only by single-step
SoC atoms and `spair` in the `local` kernel) is itself not in equilibrium given k.

Per unit the parts of the force are (means over units): capex −60 to −130 per unit u for
every carrier; measure term (physical measure, n k/(k + κ0) with n = 365) +265 (battery,
many at the box where it vanishes), +338 (H₂), +360 (gas), in the `kop` parametrization.
For gas it is cancelled by the cost term (−362): the dispatch is cost-confined at every step,
and by equipartition each confined coordinate carries β_v·cost = 1 on average, i.e. −n in
total. For H₂ and batteries it is not cancelled (−50 to −60): their state of charge is free
at most steps, which is the entropic force of §4.2.

## 4. Prototype: capacity moves with their operation (`kop`)

`scripts/mixing/kop.jl`. The missing direction of §3 is a capacity change that the unit's own
operation follows. `kop` moves a capacity at fixed **utilization** coordinates:

* k_u → k_u e^u for one unit, or for all units of a carrier with a shared u (renewables as
  load and stores; output A k for renewables; for a store the SoC fraction is fixed, so the
  stored energy and the charge and discharge powers scale with k);
* at the unit's node and every step the requirement of the other dispatchable generators
  changes by −Δinj_τ, and they follow share-preservingly, P(u) = (Q − Δinj)₊ + e, p_i = σ_i P
  with (σ, e) fixed: the coordinate line of the pro-rata reparametrization of
  sampler.md §5.2, applied at all 365 steps of the node at once;
* line density in u: π(k(u), v(u)) · Π_units k_i(u)/k_i(0) · Π_{n,τ} |P_{n,τ}(u)/P_{n,τ}(0)|^{m−1}
  (Jacobian of the scaling group on the scaled capacities, Jacobian of the pro-rata
  coordinates; the physical measure term n log(k + κ0) is part of π), including capex, the
  box penalty, every operating cost term of the unit and its compensators, the balance, and
  at α = 1 the bias change;
* drawn by slice sampling in u with the fixed width 0.3 (u is a group parameter, so the
  stepping-out rule is the same from every point of the curve), truncated to k ≥ κmin;
  selection of the units by kind and k ≥ κmin, of the compensators by |A k| ≥ κmin at
  fixed k: both invariant along the curve.

One pass (every renewable and storage unit once, every such carrier once, random order)
costs about 0.15 s per chain at 40 nodes daily (219 one-dimensional slice draws over 365
steps), against 0.58 s for the operation sweep. Checks (`check_moves.jl`): incremental against
global log target, caches against a rebuild, and the curve check (reverse move returns
exactly, line densities antisymmetric, pro-rata coordinates invariant), for single units and
whole carriers, u ∈ {0.3, −0.5, 1.2, −2}, all to round-off.

### 4.1 Results at α = 0

`own_kop`: the baseline configuration plus one `kop` pass per fine step (300 sweeps, 0.79 CPU
s per chain-sweep against 0.84 for the baseline, i.e. equal cost within the load noise of the
machine; `kop` takes 18 % of the step). Window 151–300 against the baseline's 301–600:

| quantity | baseline, 600 sweeps: R̂ | own_kop, 300 sweeps: R̂ | between-chain sd of log k per unit (baseline → kop) |
|---|---|---|---|
| onwind total | 4.0 | 4.0 | 0.50 → 0.20 |
| solar total | 4.6 | 2.8 | 0.65 → 0.43 |
| offwind total | 4.4 | 4.3 | 0.61 → 0.36 |
| gas total | 2.0 | 1.6 | 0.39 → 0.37 |
| battery total | 2.9 | 1.5 | 0.007 → 0.004 |
| H₂ total | 2.8 | 2.0 | 0.24 → 0.10 |
| capex | 4.1 | 2.4 | |
| opex | 2.2 | 1.9 | |

During the transient `kop` does what it is built for: after 25 sweeps the carrier totals of
the four chains were within 5 % of each other (baseline: a factor 1.4), the log targets within
4e4 nats (baseline 6.6e5). Sweeps until the battery total reached 440 GW: 72–99 (baseline
145–457); H₂ 130 GW: 88–110 (286–449); gas below 20 GW: 153–196 (195–421). After 300 sweeps
H₂ was at 2.1–2.2e5 MW (baseline 1.4–1.7e5 after 600, still growing). The moves are
heavy-tailed: median |u| 3–14e-4 per renewable unit, but rms 0.07–0.28 (solar), i.e.
occasional jumps of tens of percent that carry the relaxation. In the gas-free late state
`kop` is not much wider than `kscale` (median |u| per unit at the baseline end state:
onwind 7.8e-4 against 2.6e-4, solar 2.9e-4 against 1.8e-4, offwind 1.3e-3 against 5.7e-4;
storage 1.1–1.4e-3 against 2.1–4.4e-3): with no dispatchable generator left at a node the
pro-rata compensation is empty and an extra MW of renewable output can only be dumped or
shed, so the renewables remain unconverged at both ends of the run.

### 4.2 Where the α = 0 chains are going: entropy, not cost

The zero-temperature limit of the α = 0 target at fixed weather is the capacity-expansion LP
with capex scaled by β_k/β_v = 1e-3 (ensemble.md §9). Solved at the shared weather of the
`shared_*` runs (`solve_lp`, 6 min): onwind 586 GW, solar 570 GW, offwind 88 GW, gas 0,
battery 0.3 GW, H₂ 48 GW. The chains instead approach batteries at their box (465 GW),
solar 1.2–2.3 TW and H₂ > 200 GW. The difference is entropic. Under the physical measure
every store carries log J = 365 log(k + κ0); wherever its state of charge is not confined
by the cost (at a node in surplus, cycling costs only c_dump times the losses, a scale of
≈ 50 GWh), the operation volume grows like k^n_free, a force n_free/k on the capacity:
365/k against the capex slope β_k c = 2.4e-3 per MW of a battery; the two balance at k ≈ 150 GW,
far beyond the box. More renewables create more surplus, more free storage volume and more
free dispatch volume; at α = 0 these forces act at full strength against a capex that the
target weights 1000 times less than the operation. The overbuilt end state is the target's
finite-temperature behaviour, not a sampler artefact; how slowly it is reached is the
sampler's (§3). (`configs/chains40.toml` is now marked as a mechanics test for this reason.)

What this means for the ensemble: the measure-induced capacity is where the entropic force
of n free (or confined) coordinates balances capex. At α = 0 the force acts fully,
n/k = β_k c, so k ≈ n/(β_k c) ≈ 150 GW per battery (beyond every box). In ρ_1 with an exact
bias it is damped by r = β_k/β_v, r n/k = β_k c, so k ≈ n/(β_v c) ≈ 150 MW per battery, 16 MW per H₂
store, 200 MW per gas unit (n = 365): a small effect, set by the operational temperature
and the number of steps, not by β_k. So at α = 1 the choice of reference measure moves the
capacities by O(n/(β_v c)) per unit, which grows linearly with the time resolution (8760
steps at hourly resolution: 3.6 GW per battery), and the roadmap item on scaling β_v with
the number of confined coordinates applies here too.

**The other reference measure has the opposite artefact.** `shared_util_kop`
(`measure = "utilization"`, shared weather, `kop`, 300 sweeps): batteries leave the box but the
chains split, 11.5, 223, 15 and 136 GW after 300 sweeps (each still growing slowly, chain 3
stuck at 15 GW since sweep 100), and H₂ collapses to the κmin floor (≈ 1 MW per unit,
41 MW in total) in three chains while it grows to 3.6 GW in the fourth. In utilization
coordinates every SoC coordinate confined by the cost occupies a volume ∝ 1/k, an entropic
force −n_conf/k (ensemble.md §5) that diverges toward small k: going from 1 GW to 1 MW gains
up to 365·log 1000 ≈ 2500 nats per store. Against the operational value of storage at α = 0
this gives two separated regions (storage near zero, storage large), an entropic barrier that
no local capacity move crosses. Neither reference measure gives an α = 0 storage marginal
that resembles the LP. In ρ_1 both effects are damped by r = 1e-3 if the bias carries them:
the utilization-measure force is contained in the capacity scores at fixed utilizations
(−β_v ∂_k C at fixed ω averages to it), the physical-measure term ∂_k log J is not (§6).

### 4.3 A capacity transport pair does not help either (negative result)

`scripts/mixing/kxpair.jl` moves renewable capacity of one carrier between the ends (n, m) of
an AC line, k_n + t, k_m − ρt (ρ the ratio of annual capacity factors), with the power flow
following at every step through the whole network (Δθ_τ = g_τ t L⁺(e_n − e_m), g_τ the mean of
the two output changes), so each node's balance changes only by half the profile mismatch.
Everything is affine in t with fixed coefficients (no Jacobian); exact slice draw; checks
passed (caches to 2e-10, incremental against global log target). At the baseline end state
the median |t| was 2.5 MW (40 moves, all accepted), the same as `kscale`. The profile
mismatch of neighbouring nodes is enough to hit the shed side (0.35 MW thermal scale) on the
critical days. In a gas-free system a capacity move can only be wide if something with
slack follows it exactly on the critical days; in the α = 0 end state nothing does. With
dispatchable generation present (α = 1, or realistic capex) the pro-rata compensation of
`kop` provides exactly that.

A non-equilibrium candidate move (change the capacity in n small increments with operation
sweeps in between and accept with the accumulated work) does not help either at this
roughness: with a force sd of σ_f ≈ 5e3 per unit u for a single renewable unit (§3.3) and full
relaxation between increments, the work variance of a step Δu is about σ_f²Δu²/n, so a 1 %
change needs n ≈ 2500 increments, i.e. 2500 operation sweeps per move. The force has to be
made smooth by an exact compensation, not averaged.

### 4.4 At α = 1 a fast capacity move exposes the bias at once (old level-0 bias)

`run_a1.jl` set up α = 1 as `run_chains` does (level-0 warmup and bias learning, shortened to
300 iterations because the full 1500 take 80 min at 3.2 s per iteration; then frozen), with
the utilization measure. The learned ĝ is bounded as designed (implied price −5 €/MWh
for u₁ ≤ −0.5, +3.7 to +5.2 €/MWh above, far below fuel or shed prices after this short
warmup), but ĥ, the separable quadratic in the level-0
capacity coefficients, has negative curvature coefficients (c down to −5.5e6 per unit κ²),
so −(1 − r)ĥ is anti-confining inside the clamps. One `kop` sweep from the heuristic start
(checks passed, bias included) moved solar from 0.9 to 4.8 TW (above its box) and onwind
from 181 to 743 GW, and changed b̂ by −2.7e8 nats, against +6e4 nats of capex and −1.6e5 of
operating cost: the log target rose by 2.7e8 through the bias alone. This is the same
mechanism as the reported storage runaway after a refit, only immediate: with `kscale` the
ratchet of §3.2 hides it for hundreds of sweeps. The α = 1 capacity runs were therefore
stopped; they would have measured the bias, not the sampler. Any wide capacity move (`kop`,
a Metropolis form on the device, parallel tempering) needs a capacity part of the bias that
is confining (c ≥ 0 or a trust region around the fitted states, the roadmap item on
regularizing ĥ) before it can be used at α = 1.

## 5. The coarse weather

### 5.1 Under the prior alone: the z0 coordinate moves cannot mix the annual means

This is the best case for the weather moves: an exact bias and a perfect compensation
would leave the weather prior as the line density of every compensated weather move. The
level-0 prior (regime-switching VAR, 36 rows = 10 load, 10 onwind, 6 offwind, 10 solar
Haar modes, 365 daily steps) has two properties that defeat single-coordinate moves:

* **Persistence**: spectral radius of Φ_k 0.82–0.93 (correlation times of 10–30 days).
* **Collinear rows**: the innovation of the load field-mean row is predicted by the other
  load rows with R² = 0.994 (correlations −0.95, −0.89, −0.82 with load rows 4, 8, 9). Its
  conditional variance given the other rows is 0.010–0.018 against an innovation variance
  of 2.2–4.7; the diagonal of the precision of z0 is 78 for load row 1 against 1.1 for the
  solar field-mean row.

Expected-squared-jump rates (§1) per move for the annual and seasonal means of the
field-mean rows, at a prior regime draw (`scripts/mixing/prior_spectral.jl`; families:
`single` = one z0 coordinate, the fine sampler's z0 move; `blkB` = a constant shift of one
row over B consecutive days; `vblkB` = a constant shift over B days along one of the six
leading eigenvectors of the prior covariance of the 36 annual means, `results/mixing/z0dirs.jls`):

| functional | single | blk4 | blk16 | blk64 | blk365 | vblk16 | vblk64 | vblk365 |
|---|---|---|---|---|---|---|---|---|
| annual mean, load | 2.0e-8 | 8.6e-8 | 3.4e-7 | 1.3e-6 | 3.9e-6 | 3.3e-4 | 1.8e-3 | 6.8e-3 |
| annual mean, onwind | 3.3e-6 | 1.6e-5 | 5.1e-5 | 1.3e-4 | 3.5e-4 | 1.0e-3 | 3.6e-3 | 1.1e-2 |
| annual mean, offwind | 6.5e-6 | 3.2e-5 | 1.3e-4 | 4.9e-4 | 1.5e-3 | 4.1e-4 | 1.5e-3 | 4.5e-3 |
| annual mean, solar | 3.7e-6 | 3.2e-5 | 1.6e-4 | 4.8e-4 | 1.3e-3 | 5.4e-4 | 2.6e-3 | 8.7e-3 |
| winter mean, load | 2.1e-8 | 8.8e-8 | 3.2e-7 | 7.6e-7 | 1.8e-7 | 3.2e-4 | 1.1e-3 | 3.3e-4 |
| winter mean, onwind | 3.3e-6 | 1.6e-5 | 5.1e-5 | 7.4e-5 | 1.2e-5 | 7.7e-4 | 1.8e-3 | 4.1e-4 |

With `fine_weather = 0.05` the fine step makes 657 z0 moves per sweep, so the
integrated autocorrelation time of the annual means is at least 2/(rate·657) sweeps:
**1.6e5 sweeps for the load, 930 for onwind, 470 for offwind and 820 for solar**, before any
cost enters (for the load 37 CPU hours per chain and independent sample at 0.84 s per sweep). Single-row block shifts help the wind and solar rows (×100 at B = 365) but not
the load (×200 only, still 1e3 sweeps per move-equivalent), because one load row cannot move
without the others. Shifts along the joint principal directions (`vblk`) raise the load
rate by 4–5 orders of magnitude: about 300 moves of `vblk365` or 1000 of `vblk64` per
independent annual mean, and a mixture of B = 16, 64, 365 covers the seasonal means
(winter-mean rates 3e-4 to 1.8e-3). Prior-only simulations (`prior_mixing.jl`, 3000 sweeps)
agree qualitatively (the single-move chains did not decorrelate the load mean at all).

Per CPU second, with the costs of the compensated moves at the fine level measured in §5.2
(1.3 ms for a single z0 move, 44, 166 and 885 ms for joint blocks of 16, 64 and 365 days at
40 nodes), the prior-only rates become (annual means, per CPU s): load 1.5e-5 (single) against
7.5e-3, 1.1e-2, 7.7e-3 (`vblk16/64/365`), i.e. 500–700 times faster; onwind 2.5e-3 against
2.2e-2 at B = 64 (9×); solar 2.8e-3 against 1.6e-2 (6×). This is the ceiling for an exact
bias with a compensation the operation follows; §5.2 shows what the cost does to it.

### 5.2 At the fine level at α = 0: the cost confines every weather move

At the end state of a baseline chain (sweep 600), per move (`scripts/mixing/wblock.jl`,
`weather_esjd.jl`; joint compensated moves with the exact prior quadratic and the
requirement-tracking compensation, consistency checks passed):

| family | accepted | median \|t\| | CPU per move | squared change of the annual means per move / prior variance (load, onwind, offwind, solar) |
|---|---|---|---|---|
| single z0 coordinate | 0.88 | 0.059 | 1.3 ms | 3e-9, 3e-9, 0, 1e-8 |
| one row, B = 16 | 0.42 | 6.6e-3 | 57 ms | 8e-8, 1e-7, 3e-5, 8e-8 |
| joint direction, B = 16 | 0.42 | 3.4e-3 | 44 ms | 2e-7, 1e-8, 4e-9, 9e-8 |
| joint direction, B = 64 | 0.38 | 7.0e-4 | 166 ms | 8e-8, 4e-8, 4e-7, 1e-8 |
| joint direction, B = 365 | 0.20 | 2.2e-4 | 885 ms | 1e-9, 2e-8, 3e-8, 1e-10 |

Over the second half of the baseline the annual means moved per sweep by a mean squared
jump of 5e-6 (load), 2e-5 (onwind), 7e-4 (offwind) and 3e-6 (solar) times their prior
variance, i.e. about 1.8e5, 4.6e4, 1.4e3 and 3.0e5 sweeps of random walk per prior sd (with
`kop`: 7.7e4, 9.6e3, 880, 1.7e5): the load is at its prior-only bound (§5.1), the wind and
solar means are 50–360 times slower than the prior alone would allow.

At α = 0 the weather is tilted by β_v times the cost (ensemble.md §4.1) and, with gas
eliminated, a weather change at a node has nothing to compensate it except shedding or
dumping: a block move's line is confined by the cost of all its days, so its step shrinks
like 1/B and the gain of the block disappears. Per CPU second the single moves are better.
The weather at α = 0 is therefore not a mixing problem that better weather moves can solve:
the conditional of the weather given the operation is narrow (cost-confined), the tilted
weather marginal sits up to 18 prior sd away from the prior (§2), and the chains are in
different tilted states. Two consequences:

* At α = 0 the weather diagnostics (z0 annual means, regimes) measure the tilt, not the
  weather model; the screening should use `fine_weather = 0` (shared weather) or α = 1.
* At α = 1 the bias removes the mean force on z0 but not the confinement at fixed operation;
  wide compensated weather moves need a compensation that the operation would actually
  follow (storage and transport, not only dispatchable generators, which α = 0 removes), and
  the joint low-frequency directions of §5.1.

### 5.3 Regimes

At the baseline end state the centered FFBS (`fine_regime_step!`) changed 0 of 365 days in
10 calls. The non-centered regime block move (innovations fixed, the regimes of a window of
B days drawn from their exact prior conditional given z0 and the neighbours, operation
following by the pro-rata map, Metropolis with exp(−β_v ΔC) and the Jacobian; checks passed)
changed the proposed regimes in 6–22 % of the proposals (the tilted z0 itself identifies the
regimes) and was never accepted: the median cost of a proposal was 4.7e4–9.2e4 nats for
B = 1–8, because the regime enters the fine detail model (groups (regime, season, phase)):
a switch moves the 30 new spatial modes of every field by (A_g′ − A_g)ρ + (L_g′ − L_g)w, and
with z₁ far out in the tail of the regression this is a large weather change. Both
parametrizations are blocked: given the fine anomalies the regime is determined (centered),
and changing it with the innovations fixed moves the weather by much more than the cost
allows (non-centered). The regime frequencies can only move with the weather itself.

The identification is a property of the weather model, not of the α = 0 tilt: for exact
prior draws, the posterior probability that a day's regime differs from the drawn one is
0.1–0.5 % given z0 alone and 0–0.2 % given z0 and the fine details (forward–backward on
`regime_loglik` and `detail_loglik!`, five draws). The 36-dimensional regime-specific VAR
emissions separate the regimes so sharply that the regime path is practically a function of
z0. Regime mixing is therefore z0 mixing; a regime move that keeps z0 (centered or
non-centered at the fine level) cannot help, and the regime frequencies are not an
independent convergence diagnostic. Options: accept this and rely on collective z0 moves
(§5.1) followed by FFBS; or make the regimes less sharply identified in the model (tied or
shrunk covariances across regimes, fewer regime-specific dimensions, regime-free detail
groups); a move that changes m together with z0 (regenerating z0 on a window from the new
regime's VAR with its innovations fixed) changes the weather by O(1) sd per day and was not
pursued, given the cost confinement of §5.2.

### 5.4 At α = 1 with the old level-0 bias (to be repeated with the new bias)

`run_a1w.jl`: α = 1 with the bias of §4.4, capacities fixed at the heuristic start (so that ĥ
plays no role), own weather draws, operation and fine weather moves only; stopped after 10
sweeps. The weather ran away at once: the annual solar mean went from 3–10 to 38–51 prior sd
in 10 sweeps, onwind to 3–9 sd, offwind to 4–6 sd, with 2–140 GWh of shed left. The bias's
implied price of the residual load (≤ 5.2 €/MWh, §4.4) cannot cancel a tilt driven by
shedding at 3000 €/MWh, so with that bias α = 1 tilts the weather like α = 0; the weather
moves themselves are wide when the force is large (z0 moved by several sd per sweep).

The same run showed a cost problem of the old code: the fine weather sweep took 5.4 s per
chain and sweep at α = 1 against 0.12 s at α = 0, because every z0 move evaluated the full
bias twice (with the 16-sample mean Ē ĝ). Since ĝ is a sum of per-step terms and Ē ĝ, ĥ do
not depend on the weather, the change of b̂ in a z0 move needs only the touched step; a
step-local version (tested equal to 3.5e-9 against the full difference, 1000× cheaper per
move) brought the sweep to 0.14 s. It was not kept: the `bias-merge` branch replaces it by an
incremental Δb̂ from a per-state bias cache for every fine weather move.

## 6. Other findings

* **`kpeak` slice width (bug, fixed).** `kpeak_sweep!` drew δ by slice sampling with the
  stepping-out width w·k0, k0 the *current* capacity. The width then differs between the
  points of the line, and the stepping-out procedure is no longer reversible (Neal 2003
  needs the same interval rule from every point of the slice): a small but systematic
  violation of invariance, growing with the size of the moves. Fixed in `src/adaptive.jl`
  (slice in u = log((k0 + δ)/k0) with the fixed width 0.1 and the Jacobian e^u; the
  Metropolis form used by the device was already correct) and in
  `docs/system/moves/capacity.md`. `kscale` is unaffected (fixed width in log k). Tests:
  §6.1.
* **`kpeak` at α = 0** takes 11–14 % of the CPU time and moved gas by a median 3.5e-3 (log k)
  per sweep in the baseline; its learned variant does not exist in the `local` kernel.
* **Device capacity moves**: one Metropolis scale `kmove_sigma` = 0.3 (log k) for all units.
  The conditional of a renewable unit at fixed dispatch is ≈ 3e-4 wide (§3.1), so the
  acceptance is of order 1e-3 and the renewables stay frozen (`results/device40.h5`: three of
  four chains unchanged). Per-carrier scales adapted during the warmup (acceptance ≈ 0.3) are
  needed at least; better a Metropolis form of `kop` (§7).
* **Seasonal storage in the `local` kernel.** With `basis = "haar"` the fine kernel is the
  plain local sweep (`fine_local_step` takes the flow branch only with a flow basis), so the
  H₂ state of charge is moved only by single-step SoC atoms and `spair`. The H₂ SoC mean has
  R̂ 3.3–4.4 in every run, and the H₂ forces disagree between the two parametrizations (§3.3),
  i.e. the seasonal operation is not in equilibrium at fixed capacities. `storage_ms_sweep!`
  (dyadic plateau moves) works without a flow basis; its effect is in §6.2.
* **Bias and capacity mixing (not worked on, noted).** (i) Under the physical measure the
  capacity scores used for ĥ omit ∂_k log J (ensemble.md §4.4); at α = 1 the entropic force
  n_free/k of §4.2 on every store then acts unreduced, which is enough to explain storage
  running away at α = 1. (ii) The online refits during the warmup fit scores from states
  of the capacity transient (§2: hundreds of sweeps from the starts), i.e. from a
  distribution the target does not have; a fit only after the capacity transient (fast with
  `kop`) would extrapolate less. (iii) With the bias frozen after the warmup, the
  post-correction refits on the same data and its weights are uniform (main session's
  observation): it is then a no-op and cannot flag a poor bias.

### 6.1 Tests of the `kpeak` fix

`test/runtests.jl` passes in full after the fix and again on the final working tree with
both `src/` changes (bases, local moves 700/700, flow design 186/186 including the `kpeak`
cache and cost consistency with the learned variant, bias, model completion, convergence
diagnostics; `results/mixing/test_runtests.log`, `results/mixing/test_runtests_final.log`).
`test/test_exact.jl` passes: operation part |z| ≤ 1.01, capacity part (`kscale` against
`kscale` + `kpeak`) op 0.91, shed 2.05, gas capacity 0.49, mean SoC 2.13, line flow −1.16.
The same test with the pre-fix `kpeak` (the old method evaluated into the module,
`scripts/mixing/test_exact_oldkpeak.jl`) also passes (op 0.95, shed 0.68, gas 0.64, SoC
0.95, line −1.21): the end-to-end test is not sensitive enough to see this violation, and the
z ≈ −3.6…−3.9 margins recorded in `testing.md` and `roadmap.md` are no longer reproduced by
the current test code with either version (the two notes were updated).

### 6.2 Storage multiscale moves in the `local` kernel

`shared_kop_sms`: shared weather, `kop`, plus one `storage_ms_sweep!` per fine step (dyadic
plateau and shape moves of the states of charge over blocks of 2–128 days; it needs no
flow basis), 200 sweeps, 8 % of the CPU time of a step. The mean H₂ state of charge settles
at 0.28–0.31 in all four chains within the first 100 sweeps, while every run without these
moves keeps it at 0.03–0.09 for 300–600 sweeps (baseline 0.034–0.048, `shared_base`
0.032–0.056, `own_kop` 0.051–0.092; and `shared_kop`, identical except for these moves, 0.065–0.11 after 200 sweeps). The H₂ total rose to 2.45–2.48e5 MW
(between-chain spread 1 %; `shared_kop` 2.24–2.32e5), and the log targets ended 3–4e4 nats higher
than in `shared_kop` in every chain. CPU: 1.05 against 0.84 s per chain-sweep in these two runs
(the sweep ran every step here; the `src/` version runs it with probability 1/2). The seasonal operation of the H₂ stores was therefore far from
its equilibrium in every `local`-kernel run: their state of charge cannot be moved over
months by single-step SoC atoms and `spair`, and the capacity forces on H₂ (§3.3) were
measured on an unrelaxed operation.

## 7. Recommendations for the main session

In order of expected effect. "Done" items are uncommitted changes in main's working tree for
the main session to merge; the others are for the main session to integrate (prototypes and
checks in `scripts/mixing/`).

1. **Do not screen convergence at α = 0 with β_k/β_v = 1e-3.** The α = 0 target is dominated
   by effects that are not the model: an operation weighted 1000× against capex, the
   weather tilted by up to 18 prior sd (§2), and entropic capacity forces of either sign
   depending on the reference measure (§4.2). Its capacity marginal is far from the LP and,
   under the utilization measure, split into separated storage regions. Screen at α = 1 with
   a frozen, stable bias, or, for the operation and the capacity moves alone, at fixed shared
   weather (`fine_weather = 0`).
2. **Add `kop` (capacity with its operation) to the fine step**, for renewables and stores,
   one move per unit and one per carrier per fine step, after `kscale`/`kpeak`
   (`scripts/mixing/kop.jl`: `kop_term`, `kop_eval`, `kop_move!`, `kop_sweep!`; the bias change
   is included). Cost ≈ 20 % of a fine step at 40 nodes daily. Checks to port into
   `test/test_local.jl`: incremental against global log target, caches, the curve check of
   `scripts/mixing/check_moves.jl` (`check_kop`). Document it in `moves/capacity.md` and
   `moves/README.md`. On the device a Metropolis form with a per-carrier scale fits the
   capacity pass (per unit: its 365 steps, its node's dispatchable generators, the balance).
   At α = 1 it requires a confining capacity part of the bias first (§4.4); check this first
   with the rewritten bias, e.g. one `kop` pass from the heuristic start and the change of b̂.
3. **Device capacity scales**: replace the single `kmove_sigma` by per-carrier scales adapted
   during the warmup; at α = 0 a renewable unit needs ≈ 3e-4 in log k at fixed dispatch.
4. **Joint low-frequency weather moves**: generalize `weather_coord_move!` to a direction
   over a block of level-0 steps (`z0_dir_move!` in `scripts/mixing/wblock.jl`: exact VAR
   quadratic of the block, windowed propagation, requirement-tracking compensation, bias in
   the Metropolis step), with fixed directions from the prior (leading eigenvectors of the
   covariance of the annual means, `results/mixing/z0dirs.jls`, computed by
   `scripts/mixing/prior_spectral.jl`) and B ∈ {16, 64, 365}. Under the prior they are the
   only moves that can mix the load annual mean (§5.1); at the α = 0 equilibrium the cost
   confines them as much as the single moves (§5.2). Add them with a low rate once the bias
   is stable and measure again at α = 1 (§5.4 had only the old bias).
5. **Regimes**: no sampler fix; they are identified by z0 in the weather model (§5.3). If
   regime mixing matters, change the model (tied or shrunk regime covariances, regime-free
   detail groups), and drop the regime frequencies from the convergence verdict or treat them
   as functions of z0.
6. **Done: seasonal storage in the `local` kernel.** `fine_local_step` now runs
   `storage_ms_sweep!` after the plain local sweep with probability 1/2, as the flow kernel
   does (`src/local.jl`; `sampler.md` §2 and §4.2, `moves/storage.md`, `moves/README.md`,
   `architecture.md` updated). It needs no flow basis (plateau moves only) and costs ≈ 8 % of
   a step; without it the H₂ state of charge does not relax (§6.2). Checked: six fine steps on
   `chains40.toml`, log target and cached operating cost consistent with a fresh evaluation;
   `test/runtests.jl` covers `storage_ms_sweep!` itself (`test_flow.jl`).
7. **Bias** (for the redesign): include ∂_k log J(k) in the capacity scores when
   `measure = "physical"` (otherwise the entropic storage force acts unreduced at α = 1);
   fit only after the capacity transient; note that with a frozen bias the post-correction
   weights are uniform.
8. **Done: `kpeak` slice width** (§6).
9. **Cost of the bias in weather moves**: with the old code a fine weather sweep at α = 1 cost
   45× the α = 0 sweep because each z0 move evaluated the full bias twice (§5.4). The
   incremental Δb̂ of the `bias-merge` branch addresses this; its per-move cost should be
   compared with the step-local evaluation (≈ 0.01 ms per z0 move at 40 nodes daily).

## 8. Follow-up: `kop` in `src/` (branch `kop`)

`src/kop.jl` implements recommendation 2 as a move family (`kop_sweep!`, called after `kscale`
and `kpeak` in `fine_local_step` for all three kernels when `kop = true`, default false;
`docs/system/moves/capacity.md`). It is the prototype of §4 with the new bias: the bias change
of a unit move is `bias_delta_k` on the per-step `BiasCache`, that of a carrier move the units
in sequence on a copy of the cache (exact; the commit applies the same sequence to the cache).
`test/test_kop.jl` checks, at α = 0 and α = 1 on the 40-node 6-hourly test configuration,
the line density against the global log target, the bias change against the difference of
`bias_term`, the local and bias caches against rebuilds after commits and after a sweep, the
reverse move, the antisymmetry of the line densities and the invariance of (σ, e). A pass
costs about as much as a `kscale` pass (0.44 against 0.48 s at 40 nodes × 6 h).

**α = 0, with and without `kop`, new code** (`scripts/mixing/run_kop_ab.jl`, `chains40.toml`,
4 chains, `fine_local_step` itself): `ab_kop` 125 sweeps (180 CPU s per chain, median step
1.21 s), `ab_base` 150 sweeps (159 s, median step 0.89 s), i.e. about equal CPU.

| landmark | with `kop` (sweep, per chain) | without |
|---|---|---|
| battery total ≥ 440 GW | 87, 59, 82, 84 | –, –, –, 132 |
| H₂ total ≥ 130 GW | 100, 98, 87, 87 | never (≤ 113 GW at 150) |
| gas total ≤ 60 GW | 85, –, 124, 62 | 107, –, –, – |
| log targets at the end | 2.31–2.48e5 (spread 1.7e4) | 1.99–2.54e5 (spread 5.5e4) |

The transient is again about twice as fast per CPU second and the chains are closer to each
other; R̂ on the second half is uninformative in either run (the chains are still relaxing:
renewables 3.6–4.6 with `kop`, 4.5–4.6 without). The renewables end higher with `kop` (solar
2.0–2.7 TW against 1.2–1.7 TW), in the direction of the forces of §3.3.

**α = 1 smoke test with the new bias** (`scripts/mixing/run_kop_a1smoke.jl`: `chains40.toml` at
α = 1, level-0 warmup shortened to 300 iterations, bias lifted to the fine level and frozen,
heuristic start). One standalone `kop` pass (184 moves) raised the log target by 3.16e6:
−Δb̂ +1.52e6, −β_vΔC +1.64e6 (the heuristic start sheds), −β_kΔK −2.8e4, Δlog J +3.6e4; onwind
went from 181 to 1065 GW (above its 932 GW box), solar from 0.91 to 2.43 TW, batteries from 13
to 44 GW. Ten fine steps from the same start:

| sweep | b̂ with `kop` | b̂ without | gas with / without (GW) |
|---|---|---|---|
| 1 | −2.19e7 | −8.20e6 | 386 / 386 |
| 2 | −2.93e7 | −1.43e7 | 430 / 484 |
| 5 | −3.15e7 | −2.44e7 | 534 / 576 |
| 10 | −3.20e7 | −2.62e7 | 616 / 548 |

b̂ runs away by 2.6–3.2e7 nats within ten sweeps with or without `kop`; the log target is then
dominated by −b̂ (β_vC ≈ 1e6, β_kK ≈ 1e4 at sweep 10). `kop` makes the first sweep faster (b̂
−2.2e7 against −8.2e6) but is not the cause: `kscale`/`kpeak` follow the same direction a few
sweeps later (gas 270 → 550–620 GW, solar 1.7–2.1 TW, H₂ 10–52 GW). With the bias cache an
α = 1 fine step costs 0.8–1.0 s per chain at 40 nodes daily, about the α = 0 cost.

The runaway is entirely in ĥ (a rerun with the same seeds, parts of b̂ from a fresh cache):

| state | ĝ | −r Ē ĝ | (1 − r) ĥ | firm capacity |
|---|---|---|---|---|
| heuristic start | −3.9e4 | +40 | +1.7e5 | 329 GW |
| after one `kop` pass | −1.2e5 | +120 | −1.3e6 | 362 GW |
| after 10 fine steps with `kop` | −1.6e5 | +129 | −3.19e7 | 625 GW |
| after 10 fine steps without | −1.4e5 | +122 | −2.61e7 | 487 GW |

ĝ moves by about 1e5 nats and Ē ĝ by nothing, while (1 − r)ĥ falls by 2.6–3.2e7: the
capacity part of the bias is not confining in the directions the capacity moves take, as
with the old bias (§4.4). `kop = true` has to wait for a confining ĥ.

**What the new bias changes in this diagnosis.** §4.4 and §5.4 (old level-0 bias) are
superseded by the smoke test above. The capacity scores now include ∂_k log J
(`bias_online.jl`), which addresses §6 (i). The cost of the bias in the fine weather moves
(§5.4, recommendation 9) is now the incremental `bias_delta_weather` of the bias cache.
Everything at α = 0 and under the weather prior (§2–§4.3, §5.1–§5.3) does not involve the bias
and stands.

## 9. Follow-up: joint z0 block moves in `src/` (branch `wblock`)

`src/weather_blocks.jl` implements recommendation 4: `z0_block_sweep!`, called at the end of
`fine_weather_sweep!` with `fine_weather_blocks` moves per fine step in expectation (default
1.0, only with `fine_weather > 0`; docs/system/moves/weather.md). Lengths 16, 64 and 365
level-0 steps with probabilities ∝ 1/B, uniform starts, directions the six leading
eigenvectors of the regime-averaged long-run covariance of the VAR, built from the weather
model (`z0_block_directions`; their overlaps with the prior-covariance directions of §5.1 are
0.94–0.998 and their prior rates within about 10 %). At α = 1 the bias-free exact line draw is
followed by the Metropolis correction e^{−Δb̂} from the bias cache (`bias_delta_weather` over
the block's window), like the coordinate moves. `test/test_wblock.jl`: directions orthonormal;
the VAR quadratic against `logp_z0`; at α = 0 and α = 1 the line density against the global
log target, Δb̂ against the `bias_term` difference, anomalies against `weather_at_level`, the
local and bias caches against rebuilds, the reverse move with the opposite line density; the
sweeps; and 20000 repeated moves of one block against quadrature of its line density with the
bias correction (KS 0.010, against 0.22 for the bias-free conditional).

**Measurement** (`scripts/mixing/wblock_esjd.jl`): prior-only expected-squared-jump rates per
move (three prior regime draws) times the cost per move at an equilibrated α = 0 state of the
40-node daily model (`chains40.toml`, baseline end state): a z0 coordinate move costs 0.18 ms
(657 per sweep, run in parallel), a block move 24, 108 and 598 ms for B = 16, 64 and 365
(accepted 0.45, 0.33, 0/4), 59 ms per move in the mixture.

| functional (field-mean row) | coordinate, per CPU s | blocks, per CPU s | gain |
|---|---|---|---|
| annual mean, load | 1.1e-4 | 1.4e-2 | 135 |
| seasonal means, load | 1.1–1.2e-4 | 0.7–1.6e-2 | 58–136 |
| annual mean, onwind | 1.8e-2 | 3.1e-2 | 1.7 |
| annual mean, solar | 2.1e-2 | 2.3e-2 | 1.1 |
| annual mean, offwind | 3.6e-2 | 1.1e-2 | 0.3 |

The blocks are the move for the load means, the bottleneck of §5.1; for wind and solar they
are about as efficient as the coordinate moves, for offwind less; both run. At the default
rate the load annual-mean rate per sweep rises from 1.3e-5 (657 coordinate moves) to 8.5e-4,
i.e. the prior-only bound on its autocorrelation time falls from 1.6e5 to about 2400 sweeps,
for about 40 ms per sweep (5 % of a fine step at the α = 0 equilibrium; the 20 weather sweeps
measured took 145 ms without and 188 ms with one block move). At α = 0 the moves are
cost-confined (median |t| of accepted moves 2.4e-3, 1.0e-3, 4.8e-4 in the direction units), as
in §5.2. Far from equilibrium (the relaxed heuristic start) a block move is several times more
expensive (45, 700 and 6200 ms), because the walk crosses many more compensation kinks.

**At 6-hourly resolution** (`scripts/mixing/wblock_cost6h.jl`, `complete40.toml` at α = 0: 40
nodes × 6 h, adaptive kernel, all model terms; a state after 15 fine steps from the heuristic
start; 4 threads): a fine step takes 19.9 s, its weather sweep 5.7 s. Block moves cost 0.43 s
(B = 16, window 80 finest steps), 1.88 s (B = 64, 272 steps) and 35.1 s (B = 365, 1476 steps)
per move, accepted 2/12, 2/6 and 0/3. With all three lengths the mixture costs 1.88 s per move
(9.5 % of a fine step), two thirds of it in the rare 365-day moves, each of which makes its fine
step almost three times longer. The lengths are therefore capped by their window at the finest
level (`fine_weather_block_steps` = 600 finest steps: all three daily, 16 and 64 at 3 and 6 h,
16 at 1 h); at 6 h the default then costs 0.72 s per move, 3.6 % of a fine step. Under the
prior the mixture without B = 365 has about 30 % lower rates per move for the annual load mean
(5.8e-4 against 8.4e-4), which at the daily costs gives the same rate per CPU second (1.4e-2)
and at the 6-hourly costs a 1.8 times higher one.
