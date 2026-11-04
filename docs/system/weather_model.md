# The multilevel weather model

This document describes the weather prior $\rho^x$ of the ensemble: the fields it covers, the
standardized anomaly coordinates, the regime-switching VAR on the coarsest level, the
Gaussian detail models of the finer levels, the non-centered coordinates the sampler uses,
the map to node-level load and availability, and how the model is fitted and cached. Code:
`src/weather.jl` (model, fit, exact sampling), `src/weather_fine.jl` and
`src/weather_moves.jl` (moves). The moves are described in
[moves/weather.md](moves/weather.md); the level structure is shared with the energy model,
see [model.md](model.md) §2; the data source in [data.md](data.md).

## 1. Fields and data

The weather consists of four fields (`E.wspecs`, built in `build_energy_model`):

- `load` on every node;
- `onwind`, `offwind`, `solar` on the nodes that have a generator of that carrier.

The input file holds, per field, hourly values for every node of its support and every
historical year (`weather/<field>/data`, 8760 h per year with 29 February dropped; 22 years
2001–2022 in the NASA POWER files). Load is in MW, availability is a capacity factor. Each
field is expanded in the same spatial basis as the energy model fields on its support (graph
Haar for `basis = "haar"` or `"flow"`), so a coarse weather level is exactly the coarse
model data.

## 2. Standardized anomalies

For a hierarchy with finest step $h_L$, every field $f$ is first aggregated to step means of
$h_L$ hours. Then, per node and step:

- the **climatology** $\bar x$ is the mean over the years for each phase of the day and day
  of the year, smoothed with a circular window of $\pm 7$ days;
- the **anomaly** is $X - \bar x$, and the **scale** $s_f$ is its standard deviation over all
  nodes, steps and years (one scalar per field);
- the **standardized anomaly** $x_f = U_f^\top (X - \bar x)/s_f$ is the field's coefficient
  vector in its finest spatial basis $U_f$.

The coarser levels are obtained by the level restriction (`split_state`): the spatial
coefficient prefix and block means in time. The climatology is restricted the same way.
Node-level physical weather at level $\ell$ is
$$
X_f = U_f^{(\ell)}\big(\bar x_f^{(\ell)} + s_f\,x_f^{(\ell)}\big)\qquad(\texttt{weather\_nodes}).
$$

## 3. Coarsest level: regime-switching VAR

The level-0 coordinates $z_0(\tau)\in\mathbb R^{n_0}$ (the level-0 coefficients of all fields,
stacked, at level-0 step $\tau$) follow a vector autoregression whose parameters switch with
a hidden regime $m_\tau\in\{1,\dots,K\}$ (`K`, default 4):
$$
m_\tau\mid m_{\tau-1}\sim P_{m_{\tau-1},\cdot},\qquad
z_0(\tau)\mid m_\tau, z_0(\tau-1)\sim\mathcal N\!\big(\Phi_{m}z_0(\tau-1) + \mu_{m}
 + a_{m}\cos\tfrac{2\pi\tau}{T_0} + b_{m}\sin\tfrac{2\pi\tau}{T_0},\ \Sigma_{m}\big),
$$
with $m = m_\tau$, $T_0$ the number of level-0 steps per year, and $z_0(1)\sim\mathcal N(0,\Sigma_0)$,
$m_1\sim\pi_0$. The harmonic terms give every regime a seasonal mean. Conditional on the
regime path everything is Gaussian. Each year is an independent sequence (no interannual
latent factor).

**Fit** (`fit_switching_var`): EM over all years, initialized from k-means labels of the
7-step smoothed coarse weather. The M step is a responsibility-weighted ridge regression
for $(\Phi_m,\mu_m,a_m,b_m)$ and the residual covariance $\Sigma_m$; the E step is
forward–backward. $\Sigma_0$ is the empirical covariance of all $z_0$. The responsibilities
$\gamma$ of the final E step are kept for the detail fit.

## 4. Finer levels: conditional Gaussian details

Going from level $\ell$ to $\ell+1$ adds, per field, the detail coordinates $\xi$ of the
level transfer ([model.md](model.md) §2): time details of the existing spatial modes (the
$r-1$ orthonormal Helmert contrasts of each coarse block of $r = h_\ell/h_{\ell+1}$ fine
steps) and all $r$ fine steps of the newly added spatial modes. Each detail block is an
affine Gaussian function of nearby coarse values:

- **time details of existing mode $i$** in coarse block $j$:
  $\xi^T_{i,\cdot,j} = A^T_{i,g}\,\rho + L^T_{i,g}\,w$ with
  $\rho = (1,\ z_i(j-1),\ z_i(j),\ z_i(j+1))$, the coarse values of the same mode in the
  block and its neighbours;
- **steps of a new mode $i$**: $\xi^S_{i,\cdot,j} = A^S_{i,g}\,(1,\ z_1(j)) + L^S_{i,g}\,w$,
  driven by the field's first (field-mean) coarse mode.

The group $g$ is the triple (regime of the block's level-0 ancestor step, meteorological
season of that step, phase of the day of block $j$ when the coarse step is shorter than a
day). $L$ is lower triangular. Given the coarse level and the regimes, detail blocks are
independent.

**Fit** (`fit_details`): for every field, part, mode and group, a weighted least-squares
regression with the regime responsibilities as weights, shrunk toward the regression
pooled over all groups (pooled statistics enter with the weight of 20 samples), plus a small
ridge; $L$ is the Cholesky factor of the residual covariance. The data are the restrictions
of the historical anomalies to every level.

## 5. Non-centered coordinates

The sampler's weather state is $(m, z_0, w_1,\dots,w_L)$ with $w_\ell\sim\mathcal N(0,I)$
independent of everything coarser. The anomalies of every level follow deterministically:
$\xi_{\ell+1} = A\rho + Lw_{\ell+1}$ and $x_{\ell+1} = \mathrm{merge}(x_\ell,\xi_{\ell+1})$
(`details_from_w`, `weather_at_level`). The prior density is
$$
p^w(m,z_0,w) = \pi_0(m_1)\prod_{\tau>1}P_{m_{\tau-1}m_\tau}\ \cdot\ p(z_0\mid m)\ \cdot\
\prod_{\ell=1}^{L}\mathcal N(w_\ell;0,I).
$$
It is projective by construction: the level-$\ell$ marginal is the same formula truncated at
$\ell$, so no level needs a separate normalization. Exact draws are cheap
(`sample_weather`); they serve as starts, as the samples $\bar{\mathbb E}$ of the bias, and as
the reference of the weather diagnostics. The `State` stores $(m, z_0, w)$ and the anomalies
$x_0,\dots,x_\ell$ of every level (`s.xs`).

## 6. The data map

The energy model reads the node-level physical weather through a clamp
(`datamap`): load $\bar d = (X_{\rm load})_+$ and availability
$\bar a = \min(\max(X,0),1)$. The Gaussian model can produce negative load or availability
outside $[0,1]$; the data map makes the cost well defined, and its kinks are breakpoints of
the exact weather moves.

## 7. Fit, cache and seeds

`weather_model_cached` fits the model with its own random stream (seed + 7919, so a cached
and a fresh fit agree) and stores the fitted parameters in
`<cache>/weather_<hash>.jls` (the cache directory of [architecture.md](architecture.md) §5). The key is a hash of a content fingerprint of
the weather tensors in the data file, the levels, `K`, the seed and the spatial bases, so
additive changes to the data file do not invalidate it. The fit uses all years.

> **Status: missing.** There is no held-out validation of the fitted model (regime
> frequencies and persistence, correlations per level, frequency of multi-day low-wind cold
> events on years left out of the fit), and no per-year latent factor for interannual
> variability.

## 8. How the sampler moves the weather

| where | move | what it changes |
|---|---|---|
| level-0 warmup chain | FFBS of $m$ given $z_0$ | regimes (exact Gibbs) |
| level-0 warmup chain | slice update of each $z_0$ coordinate at fixed operation | coarse weather |
| level-0 warmup chain | $z_0$ coordinate with pro-rata re-dispatch (shear) | coarse weather and dispatch together |
| fine level (`fine_weather > 0`) | FFBS of $m$ given the anomalies of all levels, innovations recomputed | regimes (centered) |
| fine level | one innovation $w_\ell$ or one $z_0$ coordinate, exact line draw with the Gaussian prior and share-preserving compensation | weather details or coarse weather, with dispatch following |
| fine level | $z_0$ shifted along a fixed joint direction over 16, 64 or 365 level-0 steps, the same line draw and compensation | annual and seasonal means of the coarse weather |

Details are in [moves/weather.md](moves/weather.md). At the fine level the regimes are
almost determined by the anomalies, so the centered regime update rarely changes them.

> **Status: missing.** A non-centered regime move at the fine level (innovations fixed,
> anomalies change with the regime, Metropolis with the cost). Weather moves on the device.
