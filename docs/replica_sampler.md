# Replica Langevin sampler: prototype and measurements

Development note (2026-11-04) on the prototype of the sampler that replaces the learned bias:
M weather replicas share the capacities; the weather of every replica follows a
prior-reversible process; the operation of every replica follows its weather and the
capacities by operation sweeps; the capacities take a Langevin step on the replica-averaged
sampled score. Code: `scripts/replicas/replicas.jl` (the sampler), `scripts/replicas/run_replicas.jl`
(the chains40 run). Ledger rows E21–E24 ([experiments.md](experiments.md)). The design principle
is [system/ensemble.md](system/ensemble.md) §3.1: no exact density for (k, v), the weather
marginal exactly ρ^x.

## 1. Structure

State: shared capacities k and M replicas (m_j, z0_j, w_j, v_j), j = 1..M. One step:

1. **Weather**, per replica, from its own random stream: a pCN step in the non-centred
   coordinates (continuous mode: angle θ every step; burst mode: angle φ every `burst_every`
   steps, the replicas staggered), regime window moves (below), and the compensated
   initialisation of the dispatch (the weather-independent dispatchable units of a node take up
   the change of its residual, increases in proportion to headroom, decreases in proportion to
   output).
2. **Operation**, per replica: `inner` operation sweeps (local sweep, storage multiscale with
   probability 1/2), with the capacity scores r·ŝ_k = r(−β_v∂_kC + ∂_k log J) after each.
3. **Capacities**: one Langevin step in w = log(k + κ_s) per unit, drift from −β_k∂K + r·s̄_k
   with s̄_k = Σ_j ω_j ŝ_k^{(j)}, one noise draw, applied to every replica with its dispatch
   following (fixed output for gas, the kop curve for renewables and storage), as in E19.
   The weights ω_j are 1/M_active over the replicas whose last weather burst is at least `gate`
   steps old (all replicas in continuous mode), times the stress tilt (§4).
4. **Lag probe** every `probe_every` steps (§3).

## 2. The weather process leaves ρ^x invariant

**Coordinates.** Write the level-0 VAR in innovation form: z0(1) = L₀e₁ and
z0(τ) = Φ_{m_τ}z0(τ−1) + c_{m_τ}(τ) + L_{m_τ}e_τ, with L_k L_kᵀ = Σ_k, L₀L₀ᵀ = Σ₀ and
c_k(τ) the harmonic mean terms. For a fixed regime path m the map e ↦ z0 is affine and
invertible (block lower triangular with blocks L_k), so its Jacobian is constant in e. The fine
levels are already non-centred, ξ_ℓ = Aρ + Lw_ℓ. In u = (m, e, w) the prior is the product
$$
p(u) = p_{\rm Markov}(m)\ \mathcal N(e;0,I)\ \prod_\ell \mathcal N(w_\ell;0,I),
$$
and x = X(u) is deterministic; X pushes p forward to ρ^x because it is the generator of
`sample_weather` itself (checked bitwise below).

**Blocks.**

- *(e, w) given m:* the pCN step (e, w) → cos θ (e, w) + sin θ ξ, ξ ~ N(0, I), with transition
  density q(u → u′) = N(u′; cos θ u, sin²θ I). Then
  N(u)q(u → u′) ∝ exp(−(|u|² + |u′|² − 2 cos θ u·u′)/(2 sin²θ)) is symmetric in (u, u′), so
  the kernel is reversible for N(0, I). Since (e, w) is independent of m under p, the kernel is
  reversible for p(u). In the z0 coordinates it is the pCN step of the conditional Gaussian
  p(z0 | m): z0′ − μ_m = cos θ (z0 − μ_m) + sin θ C_m^{1/2}ξ, with μ_m the mean path.
- *m given (e, w), window move:* a window [a, b] of B level-0 steps with a uniform start,
  independent of the state; m[a..b] redrawn from p_Markov(m[a..b] | m[a−1], m[b+1]) by backward
  messages β_τ = P^{b+1−τ}e_{m_{b+1}} and forward sampling (π₀ at τ = 1). This is the exact Gibbs
  conditional of the prior, because in u the regimes are a priori independent of (e, w); a
  mixture over windows of Gibbs kernels is reversible. z0 and the details follow from X at fixed
  (e, w).
- *m given x, centred FFBS* (`regmode = :ffbs`, `fine_regime_step!`): the exact Gibbs
  conditional p(m | x₀, …, x_L) of the prior, the innovations recomputed; x does not change.
  It is available but rarely changes the regimes (E10) and costs 0.27 s per call at 40 nodes.

Every block is reversible for the prior, so their composition (in any fixed order, or chosen
by a state-independent schedule such as the bursts) leaves it invariant. **None of the blocks
reads k or v**, and no other move of the sampler changes (m, z0, w): every replica's weather is
an autonomous Markov chain whose stationary law is ρ^x, and started from an exact prior draw its
marginal is ρ^x at every step, whatever the capacities and the operation do. The compensated
initialisation and the score gating act on v and on the weights of the capacity drift only;
the gate depends on the step count, not on the weather, so it selects replicas independently
of x.

**Tests (E21).** Deterministic: the generator in innovation form reproduces `sample_weather`
bitwise from the same random stream (regimes equal, max |Δz0| = 0); the round trip
z0 → e → z0 is exact to 1e-14; log p(z0 | m) + |e|²/2 is constant over z0 at fixed m to
1.7e-9 nats (constant Jacobian); the pCN balance identity holds to 1e-13; the window draw of a
3-step window matches the enumerated conditional (χ² = 1.85 on 7 df, 40 000 draws).
Long run: the weather process alone (θ = 0.3, a window move of B = 60 every step, 6000 steps)
against 2000 exact draws, for annual mean load, the mean capacity factors of onwind, solar and
offwind, the residual load at the heuristic capacities (annual mean, peak day, worst 7-day
mean), the regime frequencies and the number of regime switches: all means within |z| ≤ 1.7
(batch-means errors), all standard deviations within 5 %. With B = 7 the same run fails on
the regime frequencies (z up to 8): the window chain of the regimes mixes in about 700 window
moves at B = 7, 110 at B = 30, 45 at B = 60 and 7–17 at B = 120 (pure regime chain); the window
length sets the regime mixing.

## 3. Dispatch lag against weather speed (E22)

From an equilibrated chains40 state (k fixed, one replica, compensated initialisation), 120
steps of one operation sweep per step, probes at steps 40, 80, 120: gap = (in-flight score −
equilibrated score)/β_k∂K per carrier (onwind, solar, offwind, gas, battery, H₂; noise about
±0.05 for renewables and ±0.15 for gas, from the θ = 0 control):

| θ per sweep | gas gap | renewables gap | storage gap | shed (MWh, steps 41–120) | opex (1e10 €/a) |
|---|---|---|---|---|---|
| 0 | +0.04 | ≤ 0.02 | ≤ 0.01 | 2.5e3 | 2.36 |
| 0.01 | −0.18 | ≤ 0.07 | ≤ 0.02 | 2.5e3 | 2.39 |
| 0.02 | −0.04 | ≤ 0.11 | ≤ 0.04 | 6.7e3 | 2.44 |
| 0.03 | +0.34 | ≤ 0.08 | ≤ 0.05 | 2.5e4 | 2.51 |
| 0.1 | +5.3 | 0.15–0.75 | 0.2 | 7.7e5 | 3.28 |
| 0.1, no compensation | +4.6 | 0.5–1.1 | −0.6, +0.2 | 8.1e5 | 3.54 |
| 0.1, window move B = 60 every 10 steps | +11 | 0.5–1.4 | 0.3–0.4 | 2.2e6 | 4.38 |
| 0.3 | +21 | 1.6–2.7 | 0.8–0.9 | 6.2e6 | 6.20 |

The gas score is the sensitive one: at equilibrium its marginal value is close to zero, and
any transient shed raises it. A gap below about 10 % needs θ ≲ 0.02 per operation sweep. The
pCN autocorrelation time of a linear weather feature is (1 + cos θ)/(1 − cos θ) ≈ 4/θ² steps,
so continuous pCN within the lag bound needs about 10⁴ operation sweeps per independent
weather year (θ = 0.02), 25 times the cost of a full refresh with its 100–150 sweeps of
relaxation (E20).

A single burst relaxes much faster than continuous motion of the same total size: after one
pCN step of φ = 0.3 (compensated), shed returns to its baseline within 20 sweeps (1.6e6 →
2.2e3 MWh) and the gas score within about 10 (0.018 → 7.5e-4 → noise). Per operation sweep a
burst of φ = 0.3 relaxed for 20 sweeps decorrelates the weather by (1 − cos φ)/20 ≈ 2.2e-3,
eleven times more than continuous pCN at θ = 0.02 (θ²/2 = 2e-4). The relaxation time grows
slowly with the burst size (20 sweeps at φ = 0.3, 100–150 at φ = π/2), so larger bursts are
more efficient per sweep, provided the scores of a replica are not used while it relaxes. This
is why the run uses staggered bursts with score gating rather than continuous pCN.

## 4. Stress tilt

Declared functional Φ(x): the annual mean residual load D_t − Σ k_ref a_t (k_ref the heuristic
capacities, fixed), standardised by its prior mean 55.48 GW and sd 3.54 GW. Replica weights
ω_j ∝ exp(λΦ(x_j)) over the active replicas, self-normalised, enter only the capacity drift:
the capacities are designed against the tilted weather law ρ^x e^{λΦ}/Z (estimated by
self-normalised importance weights over the replicas, bias O(1/M)), while every replica's
weather keeps the marginal ρ^x. At λ = 0 the weights are exactly 1/M_active (exp(0) = 1), and
a run with Φ declared at λ = 0 reproduces the run without Φ bitwise (checked: equal k and v
after 3–4 steps); the weather paths of runs with λ = 0 and λ = 2 are identical (they do not
depend on k).

**Demonstration (E24).** From the end of the low run of §5, 50 steps at a fixed ηw = 2e-4 for
every carrier (no adaptation, no probes), λ = 0 and λ = 2 with the same random streams. The
replica weather paths are identical in both arms (bitwise). With the gate only 1.5 of the 4
replicas are active on average, so the weight ESS is 1.2; the replicas' mean Φ over the run
are +0.60, +0.52, +1.21, −1.66 sd. The tilt raises the weighted Φ seen by the drift from 0.27
to 0.56 sd and the drift score r·s̄ by 0.7 % (onwind), 0.6 % (solar), 4.3 % (offwind), 5.8 %
(gas), 1.1 % (battery) and −1.1 % (H₂) of β_k∂K; after 50 steps gas is 2.0 GW, onwind 1.6 GW and
offwind 0.2 GW higher than at λ = 0, capex 0.32e9 €/a higher. The direction is the expected one
(more firm and wind capacity against years with more residual load); its size is limited by
the small number of active replicas and the short run.

## 5. Run on chains40 (E23)

Setup: `configs/chains40.toml` without bias (40 nodes daily, physical measure); starts at the
heuristic capacities ×0.5 (low) and ×2 (high), 3.3–3.97× apart per carrier; M = 4 replicas
per chain with fresh prior weather draws, compensated, 40 operation sweeps each at fixed k;
then 330 steps of one operation sweep per replica; κ_s = (500, 1000, 200, 300, 100, 100) MW,
clip 0.02 per step, reflection at the box; weather bursts φ = 0.3 every 40 steps per replica
(staggered by 10), a regime window of B = 30 with half the bursts, gate 25 steps; probes every
20 steps (30 sweeps, mean of the last 15 against the replica's last in-flight scores); ηw
starting at 5e-4, ×0.7 when |gap| > 0.10, ×1.2 when |gap| < 0.05, within [2e-5, 2e-3].
Each run took 27 min (1631 s and 1656 s, at a machine load of about 7 of 16): 5.0 s per step,
of which about 1 s per replica sweep (sweep plus scores, weather step, Langevin update) and
1.1 s per step for the probes.

**Carrier totals (GW; onwind, solar, offwind, gas, battery, H₂) and penalty-free cost**
(capex + generation + storage, mean over the replicas):

| step | low | pf cost low | high | pf cost high |
|---|---|---|---|---|
| 1 | 92, 463, 17, 135, 6.8, 0.3 | 1.43e11 | 357, 1797, 65, 451, 26.9, 0.9 | 1.64e11 |
| 50 | 162, 753, 30, 214, 7.3, 1.0 | 1.31e11 | 299, 981, 45, 314, 25.8, 1.0 | 1.31e11 |
| 100 | 217, 762, 29, 215, 8.0, 1.1 | 1.24e11 | 313, 805, 39, 268, 24.6, 0.9 | 1.24e11 |
| 200 | 300, 686, 25, 201, 8.1, 1.3 | 1.16e11 | 314, 737, 38, 248, 24.1, 1.4 | 1.21e11 |
| 330 | 339, 668, 23, 200, 8.1, 2.0 | 1.14e11 | 347, 675, 37, 233, 23.6, 2.4 | 1.18e11 |

Onwind and solar meet (2 % and 1 % apart at the end, from 3.9× apart); offwind (1.6×), gas
(1.17×), battery (2.9×) and H₂ (1.2×) do not. Over the last 100 steps the penalty-free cost of
the active replicas is 1.129e11 (low) and 1.176e11 €/a (high), capex 7.72e10 and 8.40e10; the
shed of the active replicas is 0.65 and 1.2 TWh per year, against 0.002 TWh for one replica
at fixed weather (E19 B, E22 θ = 0): the bursts, the regime windows and the moving capacities
keep the replicas' dispatch out of equilibrium.

**Equilibrated scores** (probes of the second half, mean ⟨r ŝ_k⟩ / ⟨β_k ∂K⟩ on interior
nodes), low and high: onwind 1.12 and 1.13, solar 0.87 and 0.88, offwind 0.89 and 0.98, gas
−0.06 and 0.00, battery 0.37 and 0.45, H₂ 0.23 and 0.21. Both chains agree: onwind is still
below its optimum and solar above it (both still moving the right way at the end), and gas is
overbuilt with an equilibrated marginal value of zero, as in E19.

**Lag probe.** The in-flight gas score exceeds the equilibrated one by +0.6 to +2 β_k∂K during
the first 100 steps of the low chain (transient shed), and by +0.3 to +0.8 in the high chain;
the battery gap is −0.3 to −0.8 throughout. The adaptation therefore drives the gas and battery
steps to the floor 2e-5 in both chains by step 220, after which they barely move (gas ±0.5 GW
over the last 50 steps): the probe cannot tell the lag caused by the capacity step from the
lag caused by the weather bursts and the regime windows (the probes run 30 steps after a
burst), and on noise of about ±0.15 for gas a single probe decides. The renewable steps settle
at 0.7–5e-4, H₂ at the ceiling 2e-3 (its gap stays below 5 %). The second-half mean gaps:
low −0.06, −0.02, −0.02, +0.26, −0.56, 0.00; high −0.14, −0.06, −0.10, −0.06, −0.35, −0.01.

**Weather of the replicas** against 2000 prior draws (all replica steps pooled; each replica
has about 8 bursts in the run, a correlation of cos(0.3)^8 ≈ 0.7 between its first and last
weather, so the pooled values represent about 4–6 independent years per chain): annual mean
load 203.4 and 202.6 GW (prior 203.8 ± 1.6), onwind capacity factor 0.237 and 0.223 (0.237 ±
0.016), solar 0.107 and 0.107 (0.107 ± 0.003), offwind 0.315 and 0.319 (0.316 ± 0.006),
residual load mean 54.7 and 56.1 GW (55.5 ± 3.5), peak 194 and 195 GW (201 ± 11), worst
7-day mean 170 and 171 GW (175 ± 10): all within the spread of a handful of prior draws, as
they must be, since the weather does not see k or v (§2).

**Active replicas.** With bursts every 40 steps and a gate of 25, a replica contributes scores
15 steps out of 40: on average 1.5 of the 4 replicas enter the drift, so the capacity step
averages over fewer weather years than M suggests.

## 6. What to change before it becomes the system's sampler

- **Weather speed and gating.** Continuous pCN is too slow at an acceptable lag (§3). Measure the
  relaxation time n(φ) of a burst per size (and of a regime window per B) and choose φ to
  maximise (1 − cos φ)/n(φ); gate on a relaxation indicator of the replica (its shed and gas
  score back to their pre-burst level) rather than on a fixed step count, as long as the
  indicator does not depend on the weather itself (or accept and declare that dependence); run
  more replicas than are needed active (e.g. M = 8 with 4 active), or bursts less often, so that
  the drift averages over several years.
- **Regimes.** Window moves of B ≥ 60 for the regime frequencies to mix within a run; they are
  larger shocks than the pCN bursts and need their own gate.
- **Lag probe.** Separate the capacity lag from the weather lag: probe only replicas long after
  their burst, or measure the weather-only gap at frozen k as a reference and adapt on the
  difference; decide per carrier on a running mean of several probes, not on one (noise ±0.15
  for gas); probes cost 30 sweeps each, 20 % of the run here.
- **Gas.** In every run (E19, E23) the equilibrated gas marginal value is about zero against its
  capex: gas is overbuilt and only the drift towards lower gas, slowed by the lag-inflated
  in-flight scores and the step floor, removes it. A gas step at fixed output with a larger
  step once its probes are clean, or a Rao–Blackwellised gas score, would help.
- **Dispatch tracking.** Transient shed of the active replicas stays at 0.65–1.2 TWh/a against
  0.002 TWh/a at fixed weather; better storage moves (seasonal storage, E17) and a compensation
  that uses storage and imports, not only the node's gas, would shorten every relaxation.
- **Code.** Move the sampler into `src/` (replica state, weather process, Langevin step, probe,
  tilt) with the CPU exactness tests of CLAUDE.md (the weather kernels' invariance tests above
  as test cases), then the device form.
