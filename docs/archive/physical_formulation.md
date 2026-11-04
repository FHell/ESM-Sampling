# Physical formulation: undoing the LP linearization

> **Corrections** (from the component-by-component review,
> [sampling_formulations.md](sampling_formulations.md) §10): the numbers in §2
> (gas "as if 45× capex") and §3 (idle margin ≈ 250 MW, collapse of k) hold for
> the α = 0 ensemble ρ₀ only. In the two-temperature target ρ₁ the entropic force
> carries a factor β_k/β_v (≈ 10 % of capex at daily resolution, ≈ 2.5× at hourly
> resolution with β_v fixed), and the capacity is pinned to the peak dispatch
> within ≈ 1/(β_v c) ≈ 0.6 MW, so moves of k at fixed dispatch (`kscale`) do not
> mix capacities at α = 1; transported peak-set moves are needed. Renewables are
> better treated as negative load than kept in the utilization measure (§2). The
> dependent-unit frame (§5) has a 1/|η| Jacobian with efficiencies, and the
> current priced balance already is a frame with shed/dump as the dependent unit.

The LP needs every relation as a linear inequality: dispatch p ≤ A·k,
charge/discharge/state-of-charge with a dynamics inequality, balance as an
equality with shed/dump slacks. A sampler needs none of that. This note records
which parts of the original physics can be restored and what that does to the
target. The status of each item in the code is at the end.

## 1. The product structure p = A·k·ω (in the code)

Utilization coordinates ω ∈ [0,1] make the capacity bound a box that does not
depend on k. The scaling group (k, ω) ↦ (e^u k, e^{−u} ω) leaves the dispatch
unchanged inside the box and is the ridge direction of the (k, v) target; the
investment moves of `src/local.jl` (`kscale_sweep!`) sample along it exactly (generalized
Gibbs with the Jacobian).

## 2. The reference measure decides the capacity marginal

The near-optimal ensemble π ∝ e^{−β_k c·k − β_v C(p)} needs a reference
measure on (k, v). Two natural choices differ by a Jacobian per unit with n
dispatch coordinates:

  dp = (A k)ⁿ dω .

**Utilization measure (dk dω, the current default).** Where dispatch is
confined by the operational cost (pinned at a bound or at the price-setting
level, with a width δp set by β_v and the cost slopes and independent of k),
its ω-width is δp/(A k). The confined volume is therefore ∝ k^{−n_conf}: an
entropic force −n_conf/k against capacity, on top of the capex force −β_k c.
For a gas unit with n_conf ≈ T = 365 daily steps and k = 2 GW,

  n_conf / k ≈ 0.18 per MW   vs   β_k c = 4·10⁻⁸ · 10⁵ = 4·10⁻³ per MW,

i.e. the unit behaves as if its capacity cost ≈ 45× its capex. The force is
bounded in effect by the shed cost at the peak steps (β_v c_shed Δt ≈ 2.9 per MW
and step), so it shifts the capex–shed margin rather than collapsing capacity,
but it is not economics.

**Physical measure (dk dp).** Confined dispatch has a k-independent volume, so
the capacity marginal reflects capex against operational cost only. The code
option `measure = "physical"` (`log_measure` in `src/energy.jl`) adds n_u log(k_u + κ₀) to
log π for every dispatchable investable unit (generators without
weather-dependent availability, and storage with its 3T coordinates); the
investment moves include the same term.

Renewables keep the utilization measure in the code: their physical dispatch
space has zero width wherever A = 0, so a physical measure would make the
weather prior pick up Σ log A terms (degenerate night-time coordinates). A
smooth variant log(A k + κ₀) is possible but not implemented.

## 3. Investment as the maximum of dispatch

Under the physical measure the capacity of a dispatchable unit can be
integrated out exactly. With M(p) = max_τ p_τ (hard bound p ≤ k),

  ∫_M^∞ e^{−β_k c k} dk = e^{−β_k c M} / (β_k c),

so the marginal of the dispatch is

  π(p) ∝ exp(−β_k c · max_τ p_τ − β_v C(p)),

capex charged on the peak dispatch, and the capacity given the dispatch is
k = M + Exp(β_k c): the peak plus an idle margin of mean 1/(β_k c) (≈ 250 MW
at β_k = 4·10⁻⁸ /€ and c = 10⁵ €/MW/a). For renewables k ≥ max_τ p_τ/A_τ,
equivalently.

Under the utilization measure the same integral is ∫_M^∞ e^{−β_k c k} k^{−n} dk
≈ e^{−β_k c M} M^{1−n}/(n−1): the peak is charged the extra entropic factor of §2,
and the idle margin is ≈ M/n.

Consequences for sampling:
* no capacity variable and no capacity constraint: the dispatch is the state;
* the capex term couples the steps of a unit only through its maximum, so it
  sits at the peak steps — the investment of a dispatchable unit is decided by
  the weather of its peak residual-load hours, which is exactly the
  cost-driving weather the bias is meant to find;
* step-local moves are unchanged except at the argmax: a move at τ changes the
  capex iff it changes the maximum, which needs the top-two values per unit; in
  a parallel colour class the steps of one unit interact through the maximum
  (a per-class reduction, or one class per unit's peak step);
* renewables are better left in (k, ω): their dispatch sits at A·k at most
  steps (no curtailment), a line in p-space, and a change of capacity is a
  collective move of all steps (the scaling move).

## 4. Storage as a state-of-charge trajectory

The LP has charge p⁺, discharge p⁻ and state of charge e with
e_{τ+1} − e_τ ≤ Δt(η⁺p⁺ − p⁻/η⁻ + q k). Physically there is one trajectory e_τ;
with d_τ = h(e_{τ+1} − e_τ)/Δt − q (utilization units),

  p⁺_τ = (d_τ)₊/η⁺ ,   p⁻_τ = min((−d_τ)₊ η⁻, 1)  (spill beyond the power limit),

the dynamics hold identically, simultaneous charge and discharge (a flat,
wasteful direction of the LP form) disappears, and the storage dimension drops
from 3T to T. The remaining constraints are e ∈ [0,1] and p⁺ ≤ 1. The LP's
optional spillage below the power limit is lost (it only matters as an
alternative to dumping, for units with inflow). In the sampler a change of e at
one instant moves the injection at two consecutive steps, so storage moves
become two-step block moves (`BlockMove` in `src/local.jl` supports this).

## 5. Balance

The balance is an equality priced at every node and step (one kink each, the
N·T kinks that the equipartition count must include). It can be eliminated by
choosing a dependent unit per node and step whose dispatch absorbs the
residual (the frame of the previous discussion): the balance kink becomes the
bound kink of that unit, aligned with a coordinate when the dependent unit is
the price-setting one. The choice may depend on (k, x) without any correction.

## 6. Beyond the LP

Nothing in the sampler needs convexity: part-load efficiency curves, minimum
stable generation, start-up costs, quadratic line losses or AC power flow can
enter the cost directly, at the price of losing the LP as a reference.

## Status

| item | code |
|---|---|
| utilization coordinates, scaling moves | yes (`kscale_sweep!`) |
| physical measure for dispatchable units | option `measure = "physical"` |
| capacity as max of dispatch (collapsed k) | not yet |
| storage as SoC trajectory | option `storage = "soc"` (spill beyond the power limit priced for units without inflow) |
| renewables as negative load | option `renewables = "load"` |
| balance elimination by a dependent unit | not yet |
