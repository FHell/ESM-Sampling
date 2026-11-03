# Sampling-friendly formulations of further components

This document is the reference for model components and formulation variants that the code
does not have yet: for each, how PyPSA states it as an LP, the form that suits exact line
sampling and a defensible reference measure, and what it would cost. Every item carries
"Status: not implemented". The components that are implemented are described in
[model.md](model.md), the ensemble and the reference measure in [ensemble.md](ensemble.md);
the moves that would serve these forms in [moves/README.md](moves/README.md).

## 1. Criteria

A formulation for sampling is judged by its coordinates, its constraints and its reference
measure, not only by its zero-temperature optimum:

- **No spurious capacity dependence.** The width of a cost-confined direction should not
  depend on $k$; otherwise $\log Z$ carries an entropic force on capacities.
- **No spurious weather dependence.** The entropy should contain no per-step weather terms such
  as $\sum_\tau\log A_\tau(x)$; the bias would have to learn them.
- **No artificial flat directions.** Redundant LP variables (simultaneous charge and discharge,
  curtailment against dump, the split among identical co-located units, antiparallel lossy
  links) add a volume proportional to capacity at every step and are slow directions.
- **Consistency** under time and space aggregation and under splitting a unit; fix the
  technology granularity once (one unit per node, carrier and cost class).
- **Locality.** The log reference measure should have the locality of the cost.
- **Weather only in the cost.** A constraint that depends on $x$ breaks the score identity of
  the bias; price it instead.

The Jacobian rules that decide whether a change of coordinates changes the target: a bijection
$v=\phi(z)$ carries $|\det D\phi|$; eliminating a variable by an equality with coefficient
$\eta$ carries $1/|\eta|$ (1 for unit coefficients); a group move carries the group's Jacobian;
a direction chosen from the current state is valid only if it is invariant along the line.

## 2. Balance

The implemented priced balance $\Delta t\,[c_{\rm shed}(r)_++c_{\rm dump}(r)_-]$ equals a
$\delta$-balance with shedding and curtailment as backstop units, integrated out. Its
coordinates make every single-coordinate move hit the shed slope (a width of about 0.35 MW at
the default temperature), which the pair and compensated moves work around.

**Price-setting dependent unit.** Choose per node and step a dependent unit $j(n,\tau)$ from
$(k,x)$ (for example the merit-order unit at the prices of a cheap coarse LP) and use
coordinates $(p_{-j}, r)$ with $p_j$ determined by the balance. With unit coefficients the
Jacobian is 1, so the target is unchanged; a single coordinate then sees its reduced cost
$c_i - c_j$ instead of $c_{\rm shed}$, pair moves become coordinate moves, and an angle move
updates the dependent units of the node and its neighbours in $O(\deg)$. With a conversion
unit as $j$ the factor $1/|\eta_j|$ enters; if $\eta_j$ depends on the weather (heat-pump COP)
that factor violates the weather criterion, so prefer units with constant coefficients.
Status: not implemented (the share-preserving and learned compensations realize a frame of
this kind as moves, not as coordinates).

**Haar balance frame.** In Haar coordinates of the residual, a cell sum is absorbed by the
cell's price-setting unit and each contrast by the angle difference across the cut between the
children. A constant Jacobian for fixed lines, so a reparametrization matching the cell moves.
Status: not implemented as coordinates.

**Elastic demand.** Replace the kink at $r=0$ by a convex demand curve
$\int_0^{r_+}\mathrm{VoLL}(s)\,ds$ with VoLL rising to $c_{\rm shed}$: a heterogeneous value of
lost load. It changes the target, removes one confined direction per node and step, and smooths
the load tail of the bias. Status: not implemented.

## 3. Lines and transformers

**Exact quadratic losses.** $r_e f_e^2$ split half to each end, strictly convex, breaking the
degeneracy between parallel paths and removing interpolation kinks. Line conditionals become
piecewise exp-quadratic, sampled exactly with truncated-Gaussian pieces at the cost of erf
evaluations. Status: not implemented (the model uses a piecewise-linear interpolant on a fine
grid).

**Dynamic line rating** $s(x)$. As a hard limit it makes the feasible set weather-dependent
and breaks the score identity; as a priced overload it stays in the cost and is harmless.
Status: not implemented.

**Line expansion.** Three choices:

- *Fixed impedance* (PyPSA): with the physical flow measure and $s_e\ge s_e^0$, integrating out
  the rating charges $\beta_k c_e\max(s_e^0,\max_\tau|f_{e\tau}|)$, a brownfield hinge on the
  peak flow (for the single-temperature ensembles; at $\alpha=1$ the rating is pinned to the
  peak flow as dispatchable capacities are and needs explicit capacity moves).
- *Impedance scaling with capacity* (parallel circuits): $b_e = b_e^0 s_e/s_e^0$ turns the limit
  into an $s$-independent box $|b_e^0\Delta\theta_e|\le s_e^0$, so $s_e$ acts like a capacity and
  $\Delta\theta_e$ like a utilization; a change of $s_e$ at fixed angles changes two residuals at
  every step.
- *Reference measure*: with scaled impedances $d\theta$ and the injection measure differ by
  $\det L_{\rm red}(s)$ per step (the matrix-tree theorem), an extensive term
  $T\log\det L_{\rm red}(s)$; for extendable lines use the injection measure.

Status: not implemented (line ratings are fixed).

**Transformers and phase shifters.** Transformers are lines ($x$ and tap ratio enter $b$). A
phase shifter adds one box coordinate per step, frees its cycles and touches two residuals.
Status: not implemented.

**Alternative flow coordinates.** PTDFs (no angle coordinates, dense in the injections) suit
coarse spatial levels with few cells. Tree plus cycle flows, $f = f_{\rm tree}(P) + C\phi$, suit
transport-model parts (DC links, pipelines, phase shifters), where a cycle move keeps every
balance at the cost of its length. A soft Kirchhoff law (Gaussian loop flows) changes the target
and is useful only for tempering. Status: not implemented.

## 4. Generators

**Binary unit commitment.** The feasible set $\{0\}\cup[\underline g k,\bar g k]$ needs a
reference $w_0\delta_0 + dp$ with an arbitrary weight $w_0$ in MW, reversible-jump moves between
the strata, and birth and death moves of whole on-runs for minimum up and down times. Status:
not implemented (the model has the continuous fleet commitment).

**Part load and nonconvex heat rates.** Any piecewise-linear cost in (output, committed
capacity) keeps exact line sampling; nonconvexity makes the operational measure multimodal and
loses the LP as a reference. Status: not implemented.

**Ramping** $|p_\tau-p_{\tau-1}|\le\rho k$. As walls on differences it couples neighbouring
steps (two time classes suffice); binding only at sub-daily steps. Priced as wear,
$c_\rho|\Delta p|$, it is one more kink; at coarse steps it can be left out. Status: not
implemented.

**Retirement.** With fixed operation and maintenance cost on existing capacity, the capacity
cost becomes convex piecewise linear in $k$ around $k_0$ instead of the hinge $c(k-k_0)_+$, and
the brownfield floor is lifted. Status: not implemented.

**Capacity integrated out.** Under the physical measure the capacity of a dispatchable unit can
be integrated out: $\int_M^\infty e^{-\beta_k ck}dk = e^{-\beta_k cM}/(\beta_k c)$ with $M$ the
peak dispatch, so the dispatch carries capex on its maximum and $k = M + \mathrm{Exp}(\beta_k c)$.
A move at one step then changes the capex only if it changes the maximum (a top-two cache per
unit gives one extra breakpoint). This is exact for the single-temperature ensembles. At
$\alpha = 1$ it would need $\partial_k\log Z$ to relative accuracy $\beta_k/\beta_v$, so there
the capacity stays explicit with the peak-set move. Status: not implemented.

## 5. Storage

**Independent energy capacity.** With energy capacity $E$ separate from the power rating, no
standing loss and a cyclic year, a level offset $e\to e+a$ is a flat direction of length
$E - R$ ($R$ the range of the trajectory). Integrating out $E\ge R$ with the offset gives
$e^{-\beta_k c_E R}/(\beta_k c_E)^2$: capex on the range. The offset can be sampled exactly as one
coordinate with a range cache. With standing losses the map from power to state of charge is
invertible and there is no offset. Status: not implemented (fixed energy-to-power ratios).

**Standing losses** $e_{\tau+1} = \eta_{\rm st}e_\tau + \dots$: change the state-of-charge
derivative to $e_{\tau+1}-\eta_{\rm st}e_\tau$; the rest of the state-of-charge form is
unchanged. Status: not implemented.

**Spill.** Energy beyond the power limit is priced as a violation for units without inflow and
spilled for units with inflow. Hydro at a full reservoir must generate or spill; pricing that
spill at $c_{\rm dump}$ would make it equivalent to the nodal dump. Status: not implemented
beyond the power-limit rule.

**Plateau moves with range checks.** A state-of-charge increase over an interval of instants
moves energy between two distant steps; with a segment tree of the trajectory the box check is
$O(\log T)$ instead of $O(L)$. Status: not implemented (the block moves evaluate every instant).

## 6. Links and sector coupling

**Multi-bus links.** Throughput $p_0$ in input MW is withdrawn at bus 0 and $\eta_i(\tau)p_0$
delivered at bus $i$ (negative $\eta$ for consumption). One coordinate enters several balances,
so a link move is a chain move with compensation at each bus, $O(\text{buses})$ per step. In input
MW the reference does not depend on $x$; a link as the dependent unit of bus $i$ brings
$1/|\eta_i(\tau)|$, which for a heat pump with $\mathrm{COP}(T_\tau)$ puts
$-\sum_\tau\log\mathrm{COP}(x_\tau)$ into the entropy. Prefer dependent units with constant
coefficients at sector buses (stores, boilers, resistive heaters), and assign them so that the
elimination is triangular. Status: not implemented.

**Pipelines and antiparallel links** (H₂, gas): one signed flow with loss $(1-\eta)|u|$, as for
the DC links. Status: not implemented (no pipelines; the DC links already use this form).

**Combined heat and power:** one fuel coordinate plus the operating point in the heat–power
polygon, with the back-pressure and extraction lines as walls. Status: not implemented.

**Stock buses** (H₂, CO₂ atmosphere): never give them a priced shed or dump; a dump at the CO₂
bus would be free removal. Status: not implemented (no sector buses).

## 7. Global constraints

PyPSA's per-year LP does not distinguish the ensembles below; the weather ensemble forces a
choice.

| form | locality | meaning |
|---|---|---|
| per-year cap $\mathbb 1\{E(v)\le\Gamma\}$ | one global wall; steps of a parallel class coupled through the remaining budget | the cap binds in every weather year, an implicit price spiking in bad years |
| canonical price, cost $+\mu E(v)$ | local, parallel | emissions fluctuate across years |
| canonical price tuned to the mean cap | local | the multi-year analogue of a cap with banking |
| random price, $\mu$ an auxiliary variable | $v\mid\mu$ local, $\mu\mid v$ exact in one dimension | a mixture of canonical ensembles, not a cap |

The model implements the canonical price, optionally tuned to a mean cap. Status: the per-year
cap and the random price are not implemented. Per-step budget augmentations
($\sum_\tau b_\tau = \Gamma$) introduce an entropy $(\Gamma - E)^{T-1}$ that pushes emissions
down unless divided out, which is global again; avoid them.

**Constraints on capacities** (expansion limits, technology limits, cost budgets) are walls in
the low-dimensional capacity space; keep them hard, with cached sums, and truncate the capacity
moves at them. Status: not implemented.

## 8. Capacities and levels

**Reference.** Lebesgue in MW. A log-uniform reference $dk/k$ is singular at the legitimate
value $k=0$ and implies a scale-free prior; do not use it.

**The capacity floor.** Under the physical measure a unit with $k=0$ has an empty dispatch
range, so the floor $\kappa_0$ in the penalty weights and in $\log(k+\kappa_0)$ is not needed;
penalties become $c_{\rm viol}$ times the distance in MW. Status: not implemented ($\kappa_0 = 1$
MW is used throughout).

**Coarse levels.** A coarse capacity $K_c = \sum_{n\in c}k_n$ under Lebesgue measure on the fine
capacities has fiber volume $K_c^{n_c-1}/(n_c-1)!$; a coarse model without it is biased toward
small coarse capacities by $(n_c-1)\log K_c$ (alternatively, move in shares $k_n = K_c s_n$ with
Jacobian $K_c^{n_c-1}$). In time, a coarse step carries one confined direction where the fine
level carries $r$ (the block ratio), so the coarse level is $r$ times too cold in excess cost;
fix it with $\beta_\ell = \beta_v/r_\ell$ or a fiber entropy. This matters for the coarse
level-0 chain that learns the bias. Status: not implemented.

## 9. Where an LP form biases the ensemble

A checklist for every new component; the first column says what to avoid, the second whether the
current model has it.

| LP feature | effect on the ensemble | current model |
|---|---|---|
| normalized coordinates (utilization, shares) | $-\log(\text{scale})$ per confined coordinate: entropic forces on capacities, weather terms in the entropy | open choice of the reference measure |
| redundant variables (3 storage fields, curtailment vs dump, split of co-located units, two vintages, antiparallel links, separate charger and discharger) | flat, slow directions with volume ∝ capacity | removed by the SoC form, renewables as load, merged vintages, signed links, fixed energy-to-power ratios |
| capacity as a separate upper bound | the capacity–dispatch ridge; at $\alpha=1$ capacity pinned to the peak | handled by the scaling and peak-set moves |
| per-year hard caps | implicit weather-dependent prices | canonical price instead |
| piecewise-linear approximations | artificial kinks | fine-grid loss interpolant |
| Lebesgue per snapshot under weights or aggregation | coarse steps effectively colder | relevant for coarse levels only (§8) |
| one temperature for both stages | operating cost weighted $\beta_v/\beta_k$ times capex at $\alpha = 0$ | two temperatures with the bias |
