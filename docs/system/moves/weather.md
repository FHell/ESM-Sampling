# Weather moves

Weather moves change the weather state $(m, z_0, w)$ of the non-centered weather model
([../weather_model.md](../weather_model.md)) together with the dispatch, so that a change of
load or availability is not blocked by the balance. This document covers the fine-level moves
(`fine_weather > 0`) and the moves of the level-0 warmup chain. See [README.md](README.md) and
[../sampler.md](../sampler.md) §7.

## Fine level (`fine_weather_sweep!`)

**Regimes, centered.** With the anomalies $x_0,\dots,x_L$ of every level held fixed, the regime
path is drawn exactly by forward filtering, backward sampling from $p(m\mid x)$: the level-0
VAR emissions plus, per level-0 step, the detail densities $\mathcal N(\xi;A\rho,LL^\top)$ of
all its descendants (which include the Jacobian of the map to the innovations). The
innovations are then recomputed, $w = L^{-1}(\xi - A\rho)$. Cost and bias do not change. It is
an exact Gibbs update in the centered coordinates $(m,x)$; since the anomalies almost determine
the regimes, it rarely changes them.

**Innovations and level-0 anomalies, non-centered.** One coordinate moves by $t$: an innovation
of level $\ell\ge1$ (prior $\mathcal N(0,1)$) or a level-0 anomaly (prior: the VAR terms of its
step and the next, a quadratic in $t$). With the other innovations fixed, the change propagates
linearly through the detail regressions to the finest level (a window of $\pm1$ coarse step per
level), so the raw node weather is affine in $t$ and the cost is piecewise linear, with the
data-map kinks (load $\ge0$, availability in $[0,1]$) as primary breakpoints. The dispatchable
units at every node whose requirement changes follow share-preservingly on the actual
requirement (requirement-tracking kinks); weather-dependent units are excluded from the
compensation because their injection scale changes with the weather. $t$ is drawn exactly with
the Gaussian prior on the line (truncated Gaussian segments), then the Jacobian and, at
$\alpha = 1$, the bias change $e^{-\Delta\hat b}$ enter one Metropolis step (an independence
step with the exact bias-free conditional as proposal; on rejection the state stays). The bias
is evaluated at the finest level, so every coordinate, innovation or $z_0$, moves it; its change
is that of $\hat g$ at the steps of the window ($D_t$ and $R_t$ from the data-mapped weather
entries, `bias_delta_weather`), read from and committed to the bias cache of the fine step.

**Selection and schedule.** The regime update first; then, from the finest level down to level
0, a random sample of a fraction `fine_weather` of the coordinates of each level. Moves whose
coarse blocks are far enough apart (block index modulo a spacing derived from the window
spread) run in parallel, at every level including $z_0$: their windows, and so their bias
changes, are disjoint. **Cost.** A block move over the window of steps the coordinate reaches at
the finest level, plus compensation at its nodes, plus one table lookup per window step for the
bias.

## Level-0 warmup chain

- **FFBS** of the regimes given $z_0$ (exact Gibbs).
- **Coordinate slice updates** of every $z_0$ coordinate at fixed operation, by slice sampling
  on the level-0 terms it enters (steps $\tau$ and $\tau+1$: operating cost, the per-step part of
  $\hat g$, the VAR prior), five colour classes of steps in parallel, widths adapted during the
  warmup.
- **Weather with pro-rata shear** (`weather_shear_sweep!`): one $z_0$ coordinate together with a
  re-dispatch of the dispatchable carriers of every affected balance cell in proportion to
  their current output, as a cell-constant utilization change. The shares are invariant along
  the line, so the direction is; the line target includes the Jacobian
  $\prod_c|1+g_ct/P_c|^{m_c-1}$ and is sampled by slice sampling.

**Device.** No weather moves; device chains keep the weather of their start.

> **Status: missing.** A non-centered regime move at the fine level (innovations fixed, anomalies
> changing with the regime, Metropolis with the cost), for faster regime mixing.
