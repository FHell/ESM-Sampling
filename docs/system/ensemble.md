# The sampled ensemble

This document derives the probability measure the sampler targets: the joint law of
weather $x$, capacities $k$ and operation $v$ in a two-temperature ensemble. Which properties
of this measure are fixed and which are modelling choices is stated in §3.1: only the weather
prior is fixed; the decisions need a near-optimal density in which more expensive
configurations are less likely, not one exact density. It covers the operational Gibbs measure
per weather year, the investment measure, how the ensemble is sampled (weather replicas from
the prior, a Langevin step of the capacities on the replica-averaged sampled score, §4), the
reference measure, the regularized constraints and the zero-temperature limits. The terms of the cost are listed in [model.md](model.md), the
weather prior in [weather_model.md](weather_model.md), and how the measure is sampled in
[sampler.md](sampler.md). Symbols are collected in [glossary.md](glossary.md).

## 1. Ingredients

- **Weather** $x$: one year of load and renewable availability at every node, at the
  finest time step. Its law $\rho^x$ is the fitted multilevel weather model. The sampler
  works in the non-centered coordinates $(m, z_0, w_1,\dots,w_L)$ of that model, in which
  $x = X(m,z_0,w)$ is a deterministic function and the prior density is
  $p^w(m,z_0,w) = p(m)\,p(z_0\mid m)\prod_{\ell}\mathcal N(w_\ell;0,I)$.
- **Capacities** $k$: the node capacities of every investable unit (generator power,
  storage power at a fixed energy-to-power ratio). They are first-stage decisions, fixed
  before the weather is known.
- **Operation** $v$: voltage angles, link flows, dispatch utilizations, unit-commitment
  levels and states of charge for every step of the year. It is second stage: chosen with
  perfect foresight of the whole weather year $x$.
- **Costs.** The annualized capital cost $K(k)$ (€/a) and the annual operating cost
  $C_x(k,v)$ (€/a), a sum over the $T$ steps of length $\Delta t$ with $T\Delta t = 8760$ h.
  Both include the priced constraint violations of §6.
- **Reference measure** $\mu(dv\mid k) = J(k)\,dv$ on the operation, with $dv$ Lebesgue
  measure in the coordinates of [model.md](model.md) and $J(k)$ a capacity-dependent density
  (§5). Capacities carry Lebesgue measure $dk$.
- **Temperatures** $\beta_v$ for operation and $\beta_k$ for investment, in €⁻¹, with
  ratio $r = \beta_k/\beta_v$. The defaults are $\beta_v = 4\cdot10^{-5}$ and
  $\beta_k = 4\cdot10^{-8}$, so $r = 10^{-3}$.

## 2. Operation: a Gibbs measure per weather year

For a fixed weather year $x$ and capacities $k$, the operation is near-optimal at
temperature $\beta_v$:
$$
\pi(dv\mid x,k) = \frac{e^{-\beta_v C_x(k,v)}\,J(k)\,dv}{Z_x(k)},\qquad
Z_x(k) = \int e^{-\beta_v C_x(k,v)}\,J(k)\,dv,\qquad
F_x(k) = -\beta_v^{-1}\log Z_x(k).
$$
$F_x(k)$ is the operational free energy of the year: the minimum operating cost plus an
entropic correction of order $n_{\rm conf}/\beta_v$, where $n_{\rm conf}$ counts the
cost-confined directions of $v$ (each carries about $1/\beta_v$ of expected excess cost,
the equipartition rule). Because the dispatch of a year sees the whole year, storage and
unit commitment are operated with perfect foresight within the year.

## 3. Investment against the weather distribution

Capacities are chosen against the expected free energy over the weather distribution:
$$
\rho_K(dk) \propto \exp\!\Big(-\beta_k\big[K(k) + \mathbb E_{x\sim\rho^x}F_x(k)\big]\Big)\,dk
 = \exp\!\big(-\beta_k K(k) + r\,\Lambda(k)\big)\,dk,\qquad
\Lambda(k) = \mathbb E_{x\sim\rho^x}\log Z_x(k).
$$
The target is the product of the three stages,
$$
\rho_1(dx,dk,dv) = \rho^x(dx)\,\rho_K(dk)\,\pi(dv\mid x,k).
$$
Under $\rho_1$ the weather is independent of the decisions and distributed as $\rho^x$, the
operation is Gibbs given $(x,k)$, and the capacities follow the expected-cost measure. This
is the model: perfect-foresight dispatch per weather year, investment chosen against the
weather distribution, each at its own temperature.

### 3.1 What is fixed and what is a modelling choice

The density of the decisions is not a ground truth. The aim is a near-optimal ensemble:
configurations close to the cost optimum, with more expensive ones less likely. The
temperatures, the reference measure (§5) and the regularized constraints (§6) are modelling
choices, and $\rho_K\,\pi$ is one reasonable member of a family of such ensembles, not the
law the sampler must reproduce to the nat. Approximations and adaptations that change the
sampled density of $(k,v)$ are therefore acceptable as long as they keep its defining
properties:

- the density decreases with the expected total cost $K(k)+\mathbb E_xF_x(k)$ (capacities)
  and with the operating cost given $(x,k)$ (operation), at the stated temperatures;
- it concentrates near the optimum;
- chains from different starts agree on it, and it is stable under the sampler.

A sampler design is judged by these properties, not by its distance from one exact density
(for example the weight ESS of a reweighting to $\rho_1$).

**The weather is the exception.** Its marginal must be the prior $\rho^x$: tilting it toward
cheap years would change the climate the system is designed against, not the cost
preference. A deliberate bias toward expensive weather (stress tests) is an explicit,
controllable tilt with a known weight, separate from the capacity and operation steps, never a
side effect of them (§4.5).

## 4. How the ensemble is sampled: weather replicas and a Langevin capacity step

### 4.1 The ensemble without a correction

The density that a sampler can evaluate locally is
$$
\rho_0(dx,dk,dv) \propto \rho^x(dx)\,e^{-\beta_k K(k)}\,e^{-\beta_v C_x(k,v)}\,J(k)\,dk\,dv .
$$
Integrating out $v$ shows that it is not the model: the weather is tilted, $x\mid k \propto
\rho^x(x)\,Z_x(k)$, toward cheap years, with a strength of $\beta_v$ times the spread of $F_x$
over the weather (about $4\cdot10^4$ nats at the defaults), and the capacity marginal is
$e^{-\beta_k K(k)}\,\mathbb E_x Z_x(k)$, an annealed average in which operating cost is weighted
$\beta_v/\beta_k = 10^3$ times more than capital cost. $\rho_1$ is $\rho_0$ reweighted by
$e^{-b(x,k)}$ with $b = \log Z_x(k) - r\Lambda(k)$, which would have to be known to about one
nat across the visited weather and capacities; that is out of reach at $r = 10^{-3}$ (§4.6). The
sampler therefore does not sample one joint density of $(x,k,v)$. It samples the three stages
of $\rho_1$ by three coupled processes, each with the property §3.1 asks of it.

### 4.2 The capacity score

The capacity marginal $\rho_K\propto\exp(-\beta_kK(k)+r\Lambda(k))$ has the gradient
$$
\nabla_k\log\rho_K = -\beta_k\,\partial_kK(k) + r\,\mathbb E_{x\sim\rho^x}\,\partial_k\log Z_x(k),\qquad
\partial_k\log Z_x(k) = \mathbb E_{\pi(v\mid x,k)}\big[\hat s_k\big],\quad
\hat s_k = -\beta_v\,\partial_kC_x(k,v) + \partial_k\log J(k),
$$
by the score identity (every constraint is priced, §6, so differentiation under the integral
has no boundary terms). $\hat s_k$ is the **sampled capacity score** of a state: an estimate of
$\partial_k\log Z_x(k)$ whose mean is exact when the dispatch is distributed as $\pi(v\mid x,k)$.
With $M$ weather years $x_j\sim\rho^x$, each with its own dispatch $v_j$,
$$
\bar s_k = \sum_j\omega_j\,\hat s_k^{(j)},\qquad \sum_j\omega_j = 1,
$$
estimates $\mathbb E_x\partial_k\log Z_x(k)$, and $-\beta_k\partial_kK + r\bar s_k$ estimates
$\nabla_k\log\rho_K$.

### 4.3 The sampled process

The state is the shared capacities $k$ and $M$ **weather replicas** $(m_j, z_{0,j}, w_j, v_j)$,
$j = 1..M$ (`replicas`, default 8). One step:

1. **Weather**, per replica: a step of a Markov kernel that leaves the weather prior invariant
   and reads neither $k$ nor any $v$. It works in the non-centred coordinates $u = (m, e, w)$, $e$
   the innovations of the level-0 VAR, in which the prior is
   $p_{\rm Markov}(m)\,\mathcal N(e;0,I)\prod_\ell\mathcal N(w_\ell;0,I)$: pCN steps of $(e,w)$,
   regime window draws from the Markov prior given the regimes outside the window, optionally
   the centred FFBS ([moves/weather.md](moves/weather.md)). Every replica has its own random
   stream, and the schedule of the steps (staggered bursts) does not depend on the state.
2. **Operation**, per replica: operation sweeps at fixed $(x_j, k)$, each leaving
   $\pi(v\mid x_j,k)$ invariant (the fine kernel's sweep, [sampler.md](sampler.md) §4), after a
   compensated initialisation of $v_j$ when its weather moved.
3. **Capacities**: one Langevin step in $w = \log(k+\kappa_s)$ per unit ($\kappa_s$ a per-carrier
   offset), the exact change of variables of $\rho_K$ from $k$ to $w$,
   $$
   w' = w + \eta_w\big[(k+\kappa_s)(-\beta_k\partial_kK + r\bar s_k) + 1\big] + \sqrt{2\eta_w}\,\xi,
   $$
   with the drift clipped per step and reflection at $k = \kappa_{\min}$ and at the upper bound of
   the box. The weights $\omega_j$ are uniform over the replicas whose dispatch has relaxed after
   their last weather burst (the gate, [sampler.md](sampler.md) §11.3), times the stress tilt
   (§4.5).
4. **Dispatch following**: the new $k$ is applied to every replica with its dispatch following:
   dispatchable generators at fixed output, renewables and storage at fixed utilization with the
   node's dispatchable generators following share-preservingly (the maps of `kscale` and `kop`,
   [moves/capacity.md](moves/capacity.md)).

### 4.4 What the process samples

- **Weather: exactly the prior.** Every replica's weather is an autonomous Markov chain with
  stationary law $\rho^x$: its kernel is a composition of prior-reversible blocks chosen by a
  state-independent schedule, and no other part of the step changes $(m, e, w)$. Started from an
  exact prior draw, its marginal is $\rho^x$ at every step, whatever the capacities and the
  operation do. The weather paths of different replicas are independent.
- **Operation: the Gibbs measure, up to the lag.** At fixed $(x_j,k)$ the operation sweeps
  converge to $\pi(v\mid x_j,k)$. The capacities and the weather move between sweeps, so the
  dispatch lags behind its equilibrium; the lag probe measures the resulting gap in the scores
  and keeps it below a tolerance through the step size ([sampler.md](sampler.md) §11.4).
- **Capacities: a near-optimal density of the form of $\rho_K$.** With the exact score the step is
  the unadjusted Langevin algorithm for $\rho_K$ in $w$. Its stationary law differs from $\rho_K$ by
  the discretization ($O(\eta_w)$), by the noise of $\bar s_k$ (finitely many replicas and sweeps:
  a stochastic-gradient Langevin step without correction, which widens the density), by the
  dispatch lag, and by the gate's selection of replicas. These change the density of $(k,v)$, not
  the properties §3.1 asks for: the drift points down the expected total cost, and the density
  concentrates near its optimum at the temperature $\beta_k$.

> **Status: deviation.** The gate selects replicas by a relaxation indicator computed from their
> state, so the replicas that enter the drift at a given step are not an unconditional sample of
> the weather: years whose dispatch relaxes slowly enter later (at most `replica_gate_max` steps
> after their burst). This tilts the weather average of the drift, not the weather of any
> replica. Its size is not measured.

### 4.5 Stress tilt

A deliberate bias toward expensive weather is an explicit replica weight,
$\omega_j\propto\exp(\lambda\,\Phi(x_j))$ over the active replicas (self-normalised), with a declared
functional $\Phi$ (`tilt_functional = "resload_mean"`: the annual mean residual load at the fixed
reference capacities of the set-up start, standardised by its prior mean and sd from
`tilt_samples` exact draws) and the strength $\lambda$ (`tilt_lambda`). The capacities are then
designed against $\rho^x e^{\lambda\Phi}/Z$ (estimated by self-normalised importance weights over
the replicas, bias $O(1/M)$), while every replica's weather keeps the marginal $\rho^x$. At
$\lambda = 0$, the default, the weights are exactly the untilted ones, and the step is bitwise the
untilted step.

### 4.6 Retired: the learned bias

The earlier design sampled the joint density $\rho_1 = \rho_0e^{-b}$ with a learned bias $\hat b$
(regressed on the sampled scores, lifted, refitted online, post-corrected by reweighting). Its
error enters the weather marginal undamped and had to stay below about one nat; at $r = 10^{-3}$
the regression could not reach that, and a bias error of millions of nats reversed the cost
preference of the capacities on `chains40` (ledger E14–E16). It is retired as part of the target
([../retired.md](../retired.md)); its derivation is archived in
[../archive/learned_bias.md](../archive/learned_bias.md).

> **Status: deviation.** The bias code (`src/bias.jl`, `bias_online.jl`, `bias_post.jl`, the
> level-0 warmup, `device/src/bias_dev.jl`) and the capacity moves `kscale`/`kpeak`/`kop` as moves
> of a joint density remain for the chains driver (`sampler = "chains"`, the `Config` default),
> which samples $\rho_0$ (`use_bias = false`) or the bias ensemble. The system's sampler is
> `sampler = "replicas"` (CPU); the device has no replica sampler yet.

## 5. The reference measure

The ensemble is defined only once a reference measure on $v$ is fixed. The code offers two
(`measure`):

- **Utilization** (`"utilization"`): Lebesgue measure in the dimensionless utilization
  coordinates ($\omega\in[0,1]$ for dispatch, $e\in[0,1]$ for states of charge), $J = 1$.
- **Physical** (`"physical"`): Lebesgue measure in MW and MWh for dispatchable units,
  $dp = k\,d\omega$. Per investable dispatchable unit $u$ with $n_u$ utilization coordinates
  at the level, this adds $\log J(k) = \sum_u n_u\log(k_u+\kappa_0)$. Generators count $T$
  coordinates ($2T$ with unit commitment), storage $T$ in the state-of-charge form.
  Generators with weather-dependent availability keep the utilization measure, and
  renewables treated as load have no dispatch coordinate at all. $\kappa_0 = 1$ MW keeps the
  term finite at $k = 0$.

The choice does not change $\pi(v\mid x,k)$ and does not change the weather marginal,
because $J$ depends on $k$ only. It changes $\log Z$ by $\log J(k)$ and
therefore the capacity marginal. In utilization coordinates the confined dispatch of a unit
occupies a volume proportional to $k^{-n_{\rm conf}}$, an entropic force against capacity
of size $n_{\rm conf}/k$; the physical measure removes it. In $\rho_1$ any such force reaches
$\rho_K$ only with the factor $r$ (in the Langevin drift through $r\hat s_k$, which contains
$\partial_k\log J$); in $\rho_0$ it acts at full strength.

> **Open design choice.** Which reference defines the near-optimal ensemble (utilization
> shares or physical units, per technology class) is not settled. The sampler and its moves
> work for either choice; only the Jacobians of the capacity moves and the term
> $\log J(k)$ depend on it. The configurations in `configs/` use `"physical"`; the `Config`
> default is `"utilization"`.

## 6. Regularized constraints

No constraint is a hard wall. Every physical limit is priced at a
large slope, so the target is positive everywhere:

- **Balance.** The nodal residual $r_{n,\tau}$ (demand minus supply, including line and link
  flows, losses and storage) is not forced to zero. Its cost is
  $\Delta t\,[c_{\rm shed}(r)_+ + c_{\rm dump}(r)_-]$, with $c_{\rm shed} = 3000$ €/MWh (value
  of lost load) and $c_{\rm dump} = 5$ €/MWh. Shedding is equivalent to a backstop generator
  at every node; dumping also covers the curtailment of renewables treated as load.
- **Boxes and limits.** Utilizations outside $[0,1]$, link flows outside $[-1,1]$, states of
  charge outside $[0,1]$, storage power beyond its rating, line flows beyond the rating and
  commitment walls are priced at $c_{\rm viol} = 3000$ €/MWh per MWh of excess. For units the
  excess is measured with the capacity $k+\kappa_0$ (so the penalty does not vanish at
  $k = 0$), for lines and links with their rating. The physical cost is evaluated on the
  clamped values, so outside the box only the penalty grows.
- **Capacities** outside $[\underline k,\bar k]$ pay, on top of their capex, $\gamma_k = 10$
  times the capex of the excess; the operation sees the clamped capacity.

Three properties follow. The target is proper and positive, so no move proposal is
infeasible. No constraint is a wall, so the score identities of §4.2 (and the weather
score $\nabla_x\log Z_x = \mathbb E_\pi[-\beta_v\nabla_xC]$) have no boundary terms (this needs $c_{\rm dump}$ finite: otherwise $r \ge 0$ would be a hard,
weather-dependent wall). And every cost term is convex and piecewise linear in each
coordinate, which makes the exact line sampler of [sampler.md](sampler.md) possible. As
$c_{\rm shed}, c_{\rm dump}, c_{\rm viol}\to\infty$ the hard-constrained LP is recovered.

## 7. The densities as implemented

**Replica sampler** (`sampler = "replicas"`, `src/replicas.jl`). No joint density is evaluated.
Per replica, the weather kernels use the prior $p^w(m,z_0,w)$ in the coordinates $(m, e, w)$
(`z0_innovations`, `pcn_weather`, `regime_window`); the operation sweeps use the conditional log
density at fixed $(x_j,k)$,
$$
\log\pi(v\mid x_j,k) = -\beta_v C_{x_j}(k,v) + \text{const},
$$
which is `logtarget` (`src/hierarchy.jl`) at $\alpha = 0$ up to terms constant in $v$; and the
Langevin step uses $-\beta_k\partial_kK$ and $\hat s_k = -\beta_v\partial_kC + \partial_k\log J$ of
every replica (`capacity_scores`, the gradients of `energy_cost` and `log_measure`). $-\beta_k
\partial_kK + \hat s_k$ is the gradient in $k$ of `logtarget` at $\alpha = 0$ at fixed $(v, x)$.

**Chains driver** (`sampler = "chains"`; §4.6). With $\alpha\in\{0,1\}$ (`use_bias`), the sampled
density in the non-centred weather coordinates is
$$
\log\pi(m,z_0,w,k,v) = \log p^w(m,z_0,w) - \beta_k K(k) - \beta_v C_{X(m,z_0,w)}(k,v)
 - \alpha\,\hat b(x_{\ell_b},k_{\ell_b}) + \log J(k) + \text{const},
$$
with $x_{\ell_b}$, $k_{\ell_b}$ the restrictions of the weather and the capacities to the level of
the bias ([../archive/learned_bias.md](../archive/learned_bias.md)). `logtarget` returns it
without the weather prior; the prior enters the weather moves directly (`weather_moves.jl`,
`weather_fine.jl`). The cost $C$ is `energy_cost` in `src/energy.jl` (or its incremental form
`LocalCost`/`eval_move!` in `src/local.jl`), $J$ is `log_measure`, and $\hat b$ is `bias_term`
(in the moves its incremental form `BiasCache`).

## 8. Temperatures

$\beta_v$ and $\beta_k$ set the operational and the investment slack. By equipartition, the
expected excess over the optimum is about $n_v^\ast/\beta_v$ for operation and
$n_k/\beta_k$ for investment, with $n_v^\ast$, $n_k$ the numbers of cost-confined
coordinates. Both are fixed in the configuration (`beta_v`, `beta_k`). The option
`calibrate_beta` instead derives them from the reference LP: $\beta = n/(\varepsilon\,C^\ast)$
for target slacks `eps_v`, `eps_k`, with $n$ counted from the coordinates pinned in the
interior-point solution. A fixed $\beta_v$ does not scale with the number of steps, so the
operational slack per unit of cost grows with the time resolution.

## 9. Zero-temperature limits

| ensemble and limit | limit object |
|---|---|
| $\pi(v\mid x,k)$, $\beta_v\to\infty$ | the dispatch optimum of year $x$ at capacities $k$, $\min_v C_x(k,v)$; $F_x(k)\to\min_v C_x(k,v)$ |
| $\rho_1$, $\beta_v\to\infty$ then $\beta_k\to\infty$ | the two-stage stochastic program $\min_k K(k) + \mathbb E_x\min_v C_x(k,v)$; with the empirical distribution of the historical years in place of $\rho^x$ this is the multi-year stacked LP (shared capacities, dispatch per year) |
| single-temperature perfect foresight, $\beta_k=\beta_v=\beta$, $\alpha=0$, conditional on $x$, $\beta\to\infty$ | the single-year capacity-expansion LP of year $x$ |
| $\alpha=0$ with $\beta_k\ne\beta_v$, conditional on $x$ | $\min_k K(k) + (\beta_v/\beta_k)F_x(k)$: operating cost overweighted by $\beta_v/\beta_k$, not a PyPSA-Eur problem |
| finite $\beta$ | a near-optimal ensemble; its spread corresponds to MGA envelopes at a matched cost slack |

The perfect-foresight ensemble with $k$ as a second-stage variable,
$\rho^{\rm pf}(dx,dk,dv)\propto\rho^x(dx)\,e^{-\beta(K+C_x)}$, ranks weather years by how
expensive they are even when planned for. It is not the model, but it is the ensemble whose
limits are the single-year LPs used in [validation.md](validation.md).
