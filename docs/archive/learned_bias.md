# The learned bias (retired)

Archived 2026-11-04 from [../system/ensemble.md](../system/ensemble.md) §4 (commit fb98531),
when the learned bias was retired as part of the target ([../retired.md](../retired.md)): the
derivation of the bias $b(x,k) = \log Z_x(k) - r\Lambda(k)$, its accuracy requirements and the
form of the learned $\hat b$ (`src/bias.jl`). The code still exists for the chains driver
(`sampler = "chains"`, `use_bias = true`) until it is removed; the system's sampler is the
replica–Langevin sampler of [../system/ensemble.md](../system/ensemble.md) §4 and
[../system/sampler.md](../system/sampler.md) §11. Section numbers below are those of the old
ensemble.md (§4.1–§4.4); references to §5–§9 point to the current ensemble.md.

## 4. Why a bias was needed

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

Under §3.1 the capacity part of this requirement is stricter than the model needs: a
distortion $e^{-\varepsilon}$ of $\rho_K$ is acceptable if the result still prefers cheaper
capacities near the optimum. It is not acceptable when $\varepsilon$ reverses that preference;
a bias error of millions of nats did so on `chains40`, where the frozen bias paid a chain to
shed load ([roadmap.md](roadmap.md) §1). The weather part of the requirement stands in full,
since the weather marginal must be $\rho^x$.

### 4.4 The learned bias $\hat b$

Under the physical reference measure (§5), $J$ depends on $k$ only, so
$\log Z_x(k) = \log J(k) + \log Z^u_x(k)$ with $Z^u$ the partition function in utilization
coordinates, and the exact bias contains $(1-r)\log J(k)$ exactly. The code (`src/bias.jl`)
carries that term exactly and uses the decomposition
$\log Z^u_x(k) \approx \hat g(x,k) + \hat h(k) + \text{const}$, which gives
$$
\hat b(x,k) = \hat g(x,k) - r\,\bar{\mathbb E}_{x'}\hat g(x',k) + (1-r)\,\big[\hat h(k) + \log J(k)\big].
$$
The entropic force of the reference measure, $n_u/(k_u+\kappa_0)$ per unit, is then damped to
$r\,n_u/(k_u+\kappa_0)$ in the target exactly, whatever the fit. Under the utilization measure,
and at level 0 (where no field is complete), $\log J = 0$.
$\hat b$ is evaluated at one level $\ell_b$ of the hierarchy, on the restriction of the state
to that level: the anomalies $x_{\ell_b}$ of the weather and the capacity coefficients
$k_{\ell_b}$. During the level-0 warmup $\ell_b = 0$; before the finest chains start, the bias
is lifted to their level (below), which is spatially complete, so that $\hat b$ sees the full
weather of the finest level and the node capacity of every unit.

**Weather part.** $\hat g$ is local in time, a sum over the steps $t$ of level $\ell_b$ (length
$\Delta t_b$) of two functions of two system features:
$$
\hat g(x,k) = \Delta t_b\sum_{t}\big[F_1(u_{1,t}) + F_2(u_{2,t})\big],\qquad
u_{1,t} = \frac{D_t - R_t(k)}{\bar D},\qquad u_{2,t} = u_{1,t} - \frac{F(k)}{\bar D}.
$$
$D_t$ is the system load of the level's node weather after the data map, $R_t(k)$ the available
output of every generator except gas (renewables at their weather availability, fixed fleets at
their constant availability) at the clamped node capacities of the level, $F(k)$ the firm
capacity (gas plus storage power) and $\bar D$ the mean system load of the level-0 bias samples.
The two functions are ramp sums,
$$
F_1(u) = \theta_{1}u + \sum_j\theta_{1j}\,s\,\mathrm{softplus}\big((u-\tau_j)/s\big),\qquad
F_2(u) = \theta_{2}u + \sum_j\theta_{2j}\,s\,\mathrm{softplus}\big((u-\tau'_j)/s\big),
$$
with $s=0.05$ and knots $\tau_j$ in $[-1.5,0.75]$ for $u_1$ and $\tau'_j$ in $[-0.6,0.4]$ for
$u_2$: the basis $\phi = [u_1,\ \text{ramps}(u_1),\ u_2,\ \text{ramps}(u_2)]$ with
coefficients $\theta$.

**Tables.** $F_1$ and $F_2$ are evaluated from tables (`RampTable`): the piecewise-linear
interpolant of the analytic ramp sum on a uniform grid of spacing $s/$`bias_table_res` over the
knot range widened by $20s$ on both sides, continued linearly with the end slopes beyond it
(they equal the asymptotic slopes of the ramp sum to $e^{-20}$). The tables are rebuilt
whenever $\theta$ changes, and they define $\hat g$ everywhere: global and incremental
evaluation, CPU and device. A capacity move re-evaluates $\hat g$ and $\bar{\mathbb E}\hat g$ at
every step of the level and at every sample step, which a table lookup makes cheap (an
evaluation of the ramp sums costs 22 softplus per point). The score regression keeps the
analytic basis for its design matrix.

**Weather average.** $\bar{\mathbb E}$ is a fixed weighted set of exact weather steps at the same
level $\ell_b$: of the $S\,T$ steps of $S$ = `bias_samples` exact draws of the weather at that
level, $n$ = `bias_sample_steps` are kept (uniformly without replacement), each with the weight
$\Delta t_b\,T/n$, so that $\bar{\mathbb E}\hat g = \Delta t_b\,(T/n)\sum_j[F_1+F_2]$ is an
unbiased estimate of the mean of $\hat g$ over the draws. Because $\hat g$ is a sum of per-step
terms, a subsample of steps suffices, and it is a fixed function of $k$, which is all the target
needs. $\bar{\mathbb E}\hat g$ must use the same $\hat g$ (same level) as the chain's own: through
the factor $r$ the capacity marginal sees $r(\bar{\mathbb E}\hat g + \hat h)\approx-\beta_k\,
\mathbb E_xF_x(k)$, the expected operating cost at the investment temperature, of the order of
the capex, and sub-daily residual-load variability (solar cycle, storage) changes $\hat g$ a
lot between daily and 6-hourly steps.

**Score identity.** Because no constraint depends on $x$ (§6), differentiating $\log Z$
under the integral gives
$$
\nabla_x\log Z_x(k) = \mathbb E_{\pi(v\mid x,k)}\big[-\beta_v\nabla_x C_x(k,v)\big].
$$
$\hat g$ is linear in $\theta$, so $\theta$ is a least-squares fit of $\nabla\hat g$ to per-sample
scores $s_x = -\beta_v\nabla C$ in the weather state coordinates of level $\ell_b$: $z_0$ and
the detail coordinates $\xi_1,\dots,\xi_{\ell_b}$ of the level transfers (the gradient w.r.t.
the anomalies of the level, `weather_grad_to_x`, split by the adjoints of the transfers,
`weather_coords_grad`). The design matrix is the Jacobian of the analytic $\hat g$ in the same
coordinates; the normal equations are accumulated with a small ridge.

**Monotone-price constraint** (`bias_monotone = true`, the default). The derivative
$\partial\hat g/\partial D_t = -\beta_v\Delta t_b\,\lambda(u_t)$ defines an implied price
$\lambda$ of residual load. The fit is solved as a small convex QP under the constraints
that $\lambda$ is non-decreasing (every ramp coefficient $\le 0$) and bounded in
$[-c_{\rm dump}, c_{\rm shed}]$. $\hat g$ can then grow at most at physical price slopes,
which keeps it from leaking into the load tail where there are few samples. $F_1$ and $F_2$
are then concave, so the secant slopes of their tables are non-increasing and the implied
price of the tabulated $\hat g$ is monotone as well. In addition the linear coefficient of
$u_2$ is non-positive, $\theta_{u_2}\le 0$: the shortfall part $-F_2'/(\beta_v\bar D)$ of the
price is non-negative, so firm capacity never has a negative value,
$\partial\hat g/\partial F = -(\Delta t_b/\bar D)\sum_t F_2'(u_{2,t})\ge 0$. The weather
scores do not separate $\theta_{u_1}$ from $\theta_{u_2}$ (their design columns are equal), so
without the constraint the ridge splits them equally, and a positive $\theta_{u_2}$ makes the
target push firm capacity up in surplus at about $\Delta t_b T\theta_{u_2}/\bar D$ nats per MW.

**Capacity part.** $\hat h(k) = \sum_i h_i(\kappa_i)$ is separable in the scaled, clamped
capacity coefficients $\kappa_i$ of the level (all coefficients of the level's layout). Each
$h_i$ is a fitted quadratic that can neither turn over inside the box nor run outward beyond its
data (`hcoord`). With $[\kappa^-_i,\kappa^+_i]$ the range of $\kappa_i$ in the regression data
and $q(\kappa) = a\kappa + \tfrac12 c\kappa^2$, flat beyond its vertex $\kappa^* = -a/c$ when
$c < 0$ ($C^1$, the value $-a^2/(2c)$ there; `hsat`),
$$
h_i(\kappa) = \begin{cases}
q(\kappa^+) + \max\big(q'(\kappa^+), 0\big)(\kappa-\kappa^+) & \kappa > \kappa^+,\\
q(\kappa) & \kappa^-\le\kappa\le\kappa^+,\\
q(\kappa^-) + \min\big(q'(\kappa^-), 0\big)(\kappa-\kappa^-) & \kappa < \kappa^-,
\end{cases}
$$
so that outside its data $e^{-\hat h}$ never grows away from the data. Per coordinate, the
residual capacity score $s_{k,i} - \partial_i\log J - \partial_i\hat g$, with
$$
s_k = -\beta_v\,\partial_k C + \partial_k\log J(k)
$$
(the score of $\log Z$; the measure term is `log_measure_grad`, zero under the utilization
measure and for fields that are not complete at the level, and it is subtracted again because
$\hat b$ carries it exactly), is regressed linearly on $\kappa_i$ (slope only until $\kappa_i$
has varied); the regression sums also keep the minimum and maximum of $\kappa_i$, which become
the data range at a refit (no data: an unbounded range, and $a = c = 0$). Evaluating $\hat h$
on clamped coefficients keeps it from outweighing the capacity box penalty.

Why this form: the score regression only knows $\log Z$ where the chains have been. A quadratic
fitted from states far apart is concave ($\log Z^u$ is) with its maximum inside the box, and
beyond the maximum $e^{-\hat h}$ grows with the capacity; a linear or convex extrapolation with
a slope that points away from the data does the same. Either makes the target run outward in
that capacity until the box stops it, by up to $10^9$ nats. Saturated beyond the vertex and
not pushing outward beyond the data, $\hat h$ leaves the target there to the cost and the
damped measure term: in utilization coordinates $\log Z^u$ decreases with a capacity beyond its
need (the volume of the confined dispatch shrinks like $k^{-n_{\rm conf}}$), so the target is
confined, and the chains extend the data range only where the cost leads them. A linear
continuation with the unclipped edge slope would still run away whenever the vertex lies inside
the data.

**Lift** (`lift_bias!`). The level-0 warmup learns at $\ell_b = 0$. Before the finest chains
start, the bias is lifted to their level $L$: $\theta$ and $\bar D$ are kept; $a$, $c$, the data
ranges of $\hat h$ and the coefficient scales are mapped to the fine layout (the level-0
coefficients are the prefix of each field's coefficients, the new detail coefficients start at
$a = c = 0$, so $\hat h$ of capacities without details beyond level 0 is unchanged); the
regression sums are reset, so the fine refits use fine data only; and the sample steps of
$\bar{\mathbb E}$ are redrawn at level $L$. The online refits of the finest warmup then fit at
level $L$, the first after `bias_online_min` observations (sampler.md §7.1).

How $\hat b$ is learned during a run (level-0 warmup, lift, online refits on a geometric
schedule, post-correction by reweighting) is described in [sampler.md](sampler.md) §7.1, and
how the moves evaluate it incrementally in [sampler.md](sampler.md) §6–§7.

