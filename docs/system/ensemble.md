# The sampled ensemble

This document derives the probability measure the sampler targets: the joint law of
weather $x$, capacities $k$ and operation $v$ in a two-temperature ensemble with a learned
bias. It covers the operational Gibbs measure per weather year, the investment measure, the
role of the bias $\hat b(x,k)$, the reference measure, the regularized constraints and the
zero-temperature limits. The terms of the cost are listed in [model.md](model.md), the
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

## 4. Why a bias is needed

### 4.1 The ensemble without bias ($\alpha = 0$)

The density that a sampler can evaluate locally is
$$
\rho_0(dx,dk,dv) \propto \rho^x(dx)\,e^{-\beta_k K(k)}\,e^{-\beta_v C_x(k,v)}\,J(k)\,dk\,dv .
$$
Integrating out $v$ shows what it does: the weather is tilted, $x\mid k \propto
\rho^x(x)\,Z_x(k)$, toward cheap years, with a strength of $\beta_v$ times the spread of
$F_x$ over the weather (about $4\cdot10^4$ nats at the defaults). The capacity marginal is
$e^{-\beta_k K(k)}\,\mathbb E_x Z_x(k)$, an annealed average in which operating cost is
weighted $\beta_v/\beta_k = 10^3$ times more than capital cost. $\rho_0$ is not the model.
In the code it is the case `use_bias = false`.

### 4.2 The exact relation

Define
$$
b(x,k) = \log Z_x(k) - r\,\Lambda(k).
$$
Then
$$
\rho_1(dx,dk,dv) \propto \rho^x(dx)\,e^{-\beta_k K(k)}\,e^{-\beta_v C_x(k,v)}\,J(k)\,e^{-b(x,k)}\,dk\,dv .
$$
To check this, integrate out $v$: it produces $Z_x(k)$, which cancels against
$e^{-\log Z_x(k)}$ and leaves $\rho^x(dx)\,e^{-\beta_k K(k) + r\Lambda(k)}$, which is
$\rho^x\otimes\rho_K$. So $\rho_1$ is $\rho_0$ reweighted by $e^{-b}$, and $b$ is the only
normalizing quantity that has to be learned. Its weather dependence is the weather
dependence of $\log Z_x(k)$; its capacity dependence is $(1-r)$ times the capacity
dependence of $\log Z$ plus terms that are constant in $x$.

### 4.3 Accuracy requirements

If the bias in use is $\hat b = b + \varepsilon$, the sampled $(x,k)$ marginal is
$\rho^x\otimes\rho_K$ multiplied by $e^{-\varepsilon(x,k)}$. Write
$\varepsilon = [\hat g + \hat h - \log Z] - r[\bar{\mathbb E}\hat g + \hat h - \Lambda]$
with the decomposition of §4.4. The first bracket is the model error of $\log Z_x(k)$ and
enters undamped: its weather-dependent part tilts the weather marginal and its
capacity-dependent part distorts $\rho_K$. The second bracket, the error of the
weather-averaged term, is damped by $r = 10^{-3}$. The bias therefore has to reproduce
differences of $\log Z_x(k)$ across the visited weather and capacities to about one nat.

### 4.4 The learned bias $\hat b$

The code (`src/bias.jl`) uses the decomposition $\log Z_x(k) \approx \hat g(x,k) + \hat h(k)
+ \text{const}$, which gives
$$
\hat b(x,k) = \hat g(x,k) - r\,\bar{\mathbb E}_{x'}\hat g(x',k) + (1-r)\,\hat h(k).
$$
$\bar{\mathbb E}$ is the mean over a fixed set of exact weather draws (`bias_samples`,
default 16), drawn once when the bias is set up.

**Weather part.** $\hat g$ is local in time, a sum over the level-0 steps $t$ (length
$\Delta t_0$) of a function of two system features:
$$
\hat g(x,k) = \Delta t_0\sum_{t}\theta^\top\phi(u_{1,t},u_{2,t}),\qquad
u_{1,t} = \frac{D_t - R_t(k)}{\bar D},\qquad u_{2,t} = u_{1,t} - \frac{F(k)}{\bar D}.
$$
$D_t$ is the system load after the data map, $R_t(k)$ the available output of every
generator except gas (renewables at their weather availability, fixed fleets at their
constant availability), $F(k)$ the firm capacity (gas plus storage power) and $\bar D$ the
mean system load of the bias samples. The basis $\phi$ is $[u_1,\ \text{ramps}(u_1),\ u_2,\
\text{ramps}(u_2)]$ with softplus ramps $s\,\mathrm{softplus}((u-\tau_j)/s)$, $s=0.05$, at
knots $\tau_j$ in $[-1.5,0.75]$ for $u_1$ and $[-0.6,0.4]$ for $u_2$.

**Score identity.** Because no constraint depends on $x$ (§6), differentiating $\log Z$
under the integral gives
$$
\nabla_x\log Z_x(k) = \mathbb E_{\pi(v\mid x,k)}\big[-\beta_v\nabla_x C_x(k,v)\big].
$$
$\hat g$ is linear in $\theta$, so $\theta$ is a least-squares fit of $\nabla_{z_0}\hat g$ to
per-sample scores $s_z = -\beta_v\nabla_{z_0}C$ in the level-0 weather coordinates
(accumulated normal equations, small ridge).

**Monotone-price constraint** (`bias_monotone = true`, the default). The derivative
$\partial\hat g/\partial D_t = -\beta_v\Delta t_0\,\lambda(u_t)$ defines an implied price
$\lambda$ of residual load. The fit is solved as a small convex QP under the constraints
that $\lambda$ is non-decreasing (every ramp coefficient $\le 0$) and bounded in
$[-c_{\rm dump}, c_{\rm shed}]$. $\hat g$ can then grow at most at physical price slopes,
which keeps it from leaking into the load tail where there are few samples.

**Capacity part.** $\hat h(k) = \sum_i a_i\kappa_i + \tfrac12 c_i\kappa_i^2$ is separable and
quadratic in the scaled, clamped level-0 capacity coefficients $\kappa_i$. Per coordinate,
the residual capacity score $s_{k,i} - \partial_i\hat g$, with $s_k = -\beta_v\partial_k C$,
is regressed linearly on $\kappa_i$ (slope only until $\kappa_i$ has varied). Evaluating
$\hat h$ on clamped coefficients keeps it from outweighing the capacity box penalty.

**Where the bias is evaluated.** $\hat b$ is evaluated on the level-0 restriction of the
state: the level-0 weather $z_0$ and the level-0 capacity coefficients $k_0$ (the coefficient
prefix of $k$). Fine-level scores are carried to these coordinates by the adjoint of the
level transfers.

> **Status: deviation.** Because $\hat b$ depends only on $(z_0, k_0)$, the weather
> dependence of $\log Z$ through the finer weather details $w_1,\dots,w_L$, and its capacity
> dependence through the capacity details beyond level 0, are not corrected. In those
> directions the sampled ensemble is tilted as in $\rho_0$. Whether a richer $\hat b$ is
> needed is open (see [roadmap.md](roadmap.md)).

> **Status: deviation.** Under the physical reference measure (§5),
> $\partial_k\log Z = \mathbb E[-\beta_v\partial_k C] + \partial_k\log J(k)$. The capacity
> scores used for $\hat h$ contain only the first term, so $\hat h$ misses $\partial_k\log J$
> when `measure = "physical"`.

How $\hat b$ is learned during a run (level-0 warmup, online refits on a geometric
schedule, post-correction by reweighting) is described in [sampler.md](sampler.md) §6.

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

The choice does not change $\pi(v\mid x,k)$ and, with an exact bias, does not change the
weather marginal, because $J$ depends on $k$ only. It changes $\log Z$ by $\log J(k)$ and
therefore the capacity marginal. In utilization coordinates the confined dispatch of a unit
occupies a volume proportional to $k^{-n_{\rm conf}}$, an entropic force against capacity
of size $n_{\rm conf}/k$; the physical measure removes it. In $\rho_1$ any such force reaches
$\rho_K$ only with the factor $r$; in $\rho_0$ it acts at full strength.

> **Open design choice.** Which reference defines the near-optimal ensemble (utilization
> shares or physical units, per technology class) is not settled. The sampler and its moves
> work for either choice; only the Jacobians of the capacity moves and the term
> $\log J(k)$ depend on it. The configurations in `configs/` use `"physical"`; the `Config`
> default is `"utilization"`.

## 6. Regularized constraints

No constraint is a hard wall (`regularized = true`). Every physical limit is priced at a
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
infeasible. No constraint depends on $x$, so the score identity of §4.4 has no boundary
terms (this needs $c_{\rm dump}$ finite: otherwise $r \ge 0$ would be a hard,
weather-dependent wall). And every cost term is convex and piecewise linear in each
coordinate, which makes the exact line sampler of [sampler.md](sampler.md) possible. As
$c_{\rm shed}, c_{\rm dump}, c_{\rm viol}\to\infty$ the hard-constrained LP is recovered.

## 7. The density as implemented

With $\alpha\in\{0,1\}$ (`use_bias`), the sampled density in the non-centered weather
coordinates is
$$
\log\pi(m,z_0,w,k,v) = \log p^w(m,z_0,w) - \beta_k K(k) - \beta_v C_{X(m,z_0,w)}(k,v)
 - \alpha\,\hat b(z_0,k_0) + \log J(k) + \text{const}.
$$
`logtarget` in `src/hierarchy.jl` returns this quantity without the weather prior; the prior
enters the weather moves directly (`weather_moves.jl`, `weather_fine.jl`). The cost
$C$ is `energy_cost` in `src/energy.jl` (or its incremental form `LocalCost`/`eval_move!` in
`src/local.jl`), $J$ is `log_measure`, and $\hat b$ is `bias_term`.

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
