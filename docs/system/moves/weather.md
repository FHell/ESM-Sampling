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
0, a random sample of a fraction `fine_weather` of the coordinates of each level; then the
block moves of $z_0$ below. Moves whose
coarse blocks are far enough apart (block index modulo a spacing derived from the window
spread) run in parallel, at every level including $z_0$: their windows, and so their bias
changes, are disjoint. **Cost.** A block move over the window of steps the coordinate reaches at
the finest level, plus compensation at its nodes, plus one table lookup per window step for the
bias.

**Joint low-frequency blocks of $z_0$** (`z0_block_sweep!`, `fine_weather_blocks` per fine
step). The rows of $z_0$ are strongly coupled (the load modes are nearly collinear, the VAR is
persistent), so a coordinate move barely changes an annual or seasonal mean. A block move shifts
$z_0$ along a fixed joint direction $v$ of all rows over the level-0 steps $a..b$,
$$
z_0(\tau)\to z_0(\tau) + t\,v,\qquad \tau = a,\dots,b,
$$
with the innovations of the finer levels fixed, so the finest anomalies change by the windowed
propagation (affine in $t$). The prior on the line is the exact quadratic of the VAR emissions
of the steps $a..b+1$; compensation, exact line draw with the Gaussian prior, the Jacobian and,
at $\alpha = 1$, the Metropolis bias correction (`bias_delta_weather` over the window) are those
of the coordinate moves. The directions are the six leading eigenvectors of the regime-averaged
long-run covariance of the VAR, $\sum_k\pi_k(I-\Phi_k)^{-1}\Sigma_k(I-\Phi_k)^{-\top}$ with the
stationary regime frequencies $\pi$ (`z0_block_directions`, built from the weather model): the
directions in which block means fluctuate most.

**Selection.** In sequence after the coordinate moves, `fine_weather_blocks` moves per sweep in
expectation (the integer part, plus one with the probability of the remainder); per move a
length $B\in\{16, 64, 365\}$ level-0 steps with probability $\propto 1/B$ among the lengths
whose window at the finest level (the block plus one coarse step on each side per level)
spans at most `fine_weather_block_steps` = 600 finest steps (`z0_block_lengths`: all three at
a daily finest level, 16 and 64 at 3 h and 6 h, 16 at 1 h), a start uniform over the year (the
block clipped at its end) and a direction uniform among the six: all independent of the state.
**Cost.** As a coordinate move whose window spans the block, growing faster than linearly with
the window: at 40 nodes daily about 24, 108 and 600 ms for $B$ = 16, 64 and 365 at an
equilibrated state (60 ms per move in the mixture, several times more far from equilibrium); at
40 nodes × 6 h (`complete40.toml`, all model terms, after 15 fine steps) 0.43, 1.9 and 35 s, so
that the default lengths 16 and 64 cost 0.7 s per move, 3.6 % of a 20 s fine step.

**Use.** Under the weather prior alone, the block mixture moves the annual and seasonal load
means 60–140 times faster per CPU second than the coordinate moves (which need at least about
$10^5$ sweeps per independent annual load mean at `fine_weather = 0.05`), wind and solar about
as fast, offwind slower; both families run. Where the cost confines the weather (at
$\alpha = 0$), the blocks are confined too and cost about 5 % of a fine step at the default.

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

## Prior-reversible weather process (replica sampler)

In the sampler without a learned bias ([../roadmap.md](../roadmap.md) §1) the weather of every
replica follows a process that never reads the capacities or the operation, so that its
marginal is exactly $\rho^x$. It works in the fully non-centred coordinates $u = (m, e, w)$,
where $e$ are the innovations of the level-0 VAR, $z_0(1) = L_0e_1$,
$z_0(\tau) = \Phi_{m_\tau}z_0(\tau-1) + c_{m_\tau}(\tau) + L_{m_\tau}e_\tau$ ($L_kL_k^\top=\Sigma_k$),
and the prior is $p_{\rm Markov}(m)\,\mathcal N(e;0,I)\,\prod_\ell\mathcal N(w_\ell;0,I)$:

- **pCN step** of $(e, w)$ at fixed $m$: $(e,w)\to\cos\theta\,(e,w)+\sin\theta\,\xi$,
  $\xi\sim\mathcal N(0,I)$; reversible for $\mathcal N(0,I)$, which is independent of $m$.
- **Regime window**: the regimes of a window of $B$ level-0 steps (uniform start) redrawn from
  their Markov prior given the regimes outside it, at fixed $(e,w)$; the exact prior Gibbs
  conditional, since $m$ is a priori independent of $(e,w)$. The window length sets the regime
  mixing (about 45 window moves per independent regime frequency at $B = 60$, 700 at $B = 7$).
- Alternatively the centred FFBS above (exact, leaves $x$ unchanged, rarely changes $m$).

After a weather change the dispatch is compensated (the node's weather-independent
dispatchable units take up the change of the node residual). The dispatch lags a moving
weather: a gap below 10 % of $\beta_k\partial K$ in the capacity scores needs $\theta\lesssim0.02$
per operation sweep, about $10^4$ sweeps per independent weather year, while a single pCN step
of $\varphi = 0.3$ relaxes in about 20 sweeps. The replica sampler therefore moves the weather in
staggered bursts and gives a replica's scores zero weight for a fixed number of steps after its
burst (a schedule that does not depend on the weather). The invariance argument, its tests and
the measurements are in [../../replica_sampler.md](../../replica_sampler.md).

> **Status: prototype.** In `scripts/replicas/replicas.jl` (`pcn_weather`, `regime_window`,
> `weather_step`, `compensate!`), not in `src/` and not on the device.
