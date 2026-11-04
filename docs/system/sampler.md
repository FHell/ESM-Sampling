# The sampler

This document describes how the ensemble of [ensemble.md](ensemble.md) is sampled: the
fine-level transition as a mixture of exact line moves, the exact sampler of a line
conditional, why each move leaves the target invariant, the compensation rules, adaptation,
the capacity and weather moves, how the bias is learned and corrected, and multi-chain runs
with their convergence diagnostics. The individual move families are in
[moves/README.md](moves/README.md); the code structure in
[architecture.md](architecture.md); the GPU version in [device.md](device.md).

## 1. Structure of a run

1. **Setup** (`setup`): read the data, build the energy model, fit or load the weather model,
   draw an initial weather year, and build a start state at the finest level, either from the
   reference LP at the first spatially complete level (`init = "lp"`, small models) or from a
   heuristic (`init = "heuristic"`: renewables at a proxy share of their potential, gas at the
   nodal peak load, batteries at 5 % of it, interior dispatch).
2. **Level-0 warmup** (only at $\alpha = 1$): a chain on the coarsest level learns the bias
   $\hat b$ from its scores, which is then lifted to the finest level (§7.1).
3. **Finest-level chains**: `mlda_warmup` sweeps of warmup, in which the adaptive modes,
   the online bias and the optional CO₂ price adapt, then `nsamples` production sweeps with
   the bias and the CO₂ price frozen. `nchains` chains in the CPU driver (`run`, §9), many
   chains as a batch on the device ([device.md](device.md)).
4. **Outputs**: traces of monitored quantities, sampled prices, the learned quantities,
   diagnostics and the bias post-correction ([data.md](data.md)).

The finest level is spatially complete, and the coarser levels enter the fine chain only
through the move set: coarse spatial and temporal directions are moves of the fine state.

## 2. The fine-level transition

One transition (`fine_local_step`) of a finest-level state is the composition of

- an **operation sweep** with capacities and weather fixed (§4),
- a pass of **capacity moves**, `kscale` then `kpeak`, one move per investable unit, then with
  `kop = true` the `kop` moves (§6),

in random order (probability 1/2 each), followed by a **weather sweep** if
`fine_weather > 0` (§7). The operation sweep is one of three kernels (`fine_kernel`):

| `fine_kernel` | operation sweep | families |
|---|---|---|
| `"local"` | `local_sweep!` | atoms and the plain local families, then with probability 1/2 the storage multiscale sweep, then the long-range storage sweep |
| `"flow"` (the `"auto"` choice with `basis = "flow"`) | `flow_sweep!` | plain plus transport families, then (SoC form) with probability 1/2 the storage multiscale sweep, then the long-range storage sweep |
| `"adaptive"` | `adaptive_sweep!` | plain, transport and learned families, storage multiscale, long-range storage, learned temporal shapes; then observation and adaptation |

Each component leaves the target invariant, so their composition does.

## 3. Exact line moves

### 3.1 The line conditional

Every move is a one-dimensional curve through the current state, $s(t)$ with $s(0)$ the
current state, and $t$ is drawn from the target restricted to the curve:
$$
p(t)\ \propto\ \exp\!\big(-\beta_v\,\Delta C(t) - q(t)\big)\,J_{\rm curve}(t),
$$
with $\Delta C(t) = C(s(t)) - C(s(0))$, $q(t)$ the change of the weather prior (weather
moves only, a quadratic $\tfrac12\kappa t^2 + bt$), and $J_{\rm curve}$ the Jacobian of the
curve's parametrization (compensated moves only). For capacity moves $\beta_k\Delta K$, the
bias change and the measure term are part of the exponent.

Every cost term is convex and piecewise linear in its arguments, and the arguments are
piecewise linear in $t$, so $\Delta C(t)$ is piecewise linear and $e^{-\beta_v\Delta C}$ is
piecewise exponential. No step width is involved: the same sampler serves a single-unit
move and a network-wide cell move, whose natural scales differ by orders of magnitude.

### 3.2 Breakpoints

- **Primary breakpoints** are known from the prepared move before any evaluation: the clamp
  points of every moved node value (0 and 1, $-1$ for links, the merit-order kink of two
  vintages, zero flow of a lossy link), the knots of every moved line flow ($\pm U_j s_e$
  of the loss and overload interpolants, and $f = 0$ with losses), the kink $t^\ast=-Q/g$ of
  every share-preserving compensation together with the clamp points on both sides of it,
  and the data-map kinks of moved weather values.
- **Secondary breakpoints** are the zeros of the secondary quantities: the storage dynamics
  quantities (the charge/discharge switch $d = 0$ and the power limits) and the
  unit-commitment wall arguments, which are linear between consecutive primary breakpoints, and
  the balance residuals of the touched nodes (cell sums at coarse levels), which are linear
  only between consecutive kinks of the storage injections: a storage term's injection kinks
  at the zeros of its own dynamics quantities and bends the residual it feeds inside a primary
  segment. Within a primary segment the walk therefore first places the zeros of the storage
  quantities (exact by linear interpolation), then interpolates the zeros of the residuals on
  the pieces between them, where every secondary quantity is linear (`walk!`,
  `storage_mask!`, `piece_roots!`).

### 3.3 The walk

`sample_line!` walks outward from $t = 0$ in both directions over the primary breakpoints in
order (short lists sorted, long ones popped from a heap), evaluating $\Delta C$ at each and
inserting the secondary zeros in between. A walk stops when the potential exceeds its
running minimum by `maxnats` = 40, so the neglected mass is below $e^{-40}$. Beyond the last
breakpoint it steps outward with a step length derived from a probe of the outgoing slope;
a line that does not rise within 80 such steps is improper and the move returns $t = 0$.
Then the segment masses $\log\int e^{-g s}ds$ are computed stably (`logseg`), a segment is
chosen by inversion, and $t$ is drawn inside it from the truncated exponential
(`trexp_offset`). With a Gaussian prior on the line (weather moves) each segment is a
truncated Gaussian (`sample_segments_gauss`, exact `trnorm`).

**Requirement-tracking compensation.** With line losses or lossy links the requirement that
a share-preserving compensation follows is itself piecewise linear in $t$, so the kinks are
not known in advance and a kink of one secondary quantity bends others. For these moves the
walk proceeds event by event: at the current point a probe step gives the one-sided slopes
of all secondary quantities, which are extrapolated up to the earliest root; the root is
corrected by the secant through the current point and the extrapolated one (the probe slope
carries the round-off of a difference over a step of $10^{-7}$ of the segment), evaluated, and
becomes the new current point (`walk!`, the `nonlin` branch).

> **Status: deviation.** The secant is exact only if the event's quantity is linear between
> the current point and the extrapolated one; where it, or a quantity it depends on, bends at
> the root, the point lands about $10^{-7}$ of the segment off the kink, and the potential is up
> to about $10^{-5}$ nats off its linear interpolation there (weather moves with many events,
> `test/test_moves.jl`; [testing.md](testing.md) §5). Repeating the secant from the new point
> made it worse on other lines (up to 9e-4 nats; not analysed).

### 3.4 The Jacobian step

For curves whose parametrization has a non-constant Jacobian (share-preserving
compensation, §5), $t$ is first drawn from the piecewise-exponential conditional without the
Jacobian, then accepted with probability $\min(1, J_{\rm curve}(t)/J_{\rm curve}(0))$; on
rejection $t = 0$. This is an independence Metropolis step on the line with the exact
conditional as proposal, so it leaves the line target with the Jacobian invariant.

## 4. Operation sweeps

### 4.1 One sweep

`local_sweep!` updates the operation with $(k,x)$ fixed. For every step $\tau$ the move list
contains every basis atom of every operation field at $\tau$ plus `local_extra`$\cdot N$
random moves (default $4N$), each drawn from the family list with fixed probabilities; the
list is shuffled and each move is drawn exactly along its line. In the state-of-charge form
a move at $\tau$ is a linked two-step move over $\tau-1$ and $\tau$, because a state of
charge enters the storage terms of both steps.

Steps are updated in parallel by parity class (odd and even steps, three classes when $T$ is
odd): a move at $\tau$ reads and writes only steps $\tau-1$ and $\tau$, so moves at steps of
one class are conditionally independent. The classes run in random order, every step has its
own random stream seeded from the chain's generator, and per-thread workspaces (`MoveWS`)
hold only scratch.

### 4.2 Move families and rates

The family list and its rates depend on the kernel (§2):

| kernel | families (rate) |
|---|---|
| plain | `unit` 0.3, `pair` 0.25, `xpair` 0.03, `xlocc` 0.08, `xloc` 0.04, `potential` 0.1, `cellpair` 0.1, `cell` 0.1, and `spair` 0.3 in the SoC form; `sxpair` at `storage_xpair` (0.1) on top, in every kernel |
| flow | plain, plus `tpot` 0.1, `tmode` 0.05, `tcut` 0.15, `tline` 0.05, `tmode0` 0.05, `tline0` 0.05, `tsoc` 0.1 (SoC form) |
| adaptive | flow, plus `amode` 0.03, `acut` 0.1, `amode0` 0.03 and the learned-compensation variants `tmodeL` 0.03, `tcutL` 0.1, `tlineL` 0.05, `tpotL` 0.05, `amodeL` 0.02, `acutL` 0.08, `tsocL` 0.05 |

The number of random moves is scaled so that the plain families keep their absolute rates;
`sxpair` is added on top of each list with the number scaled again, so that it leaves every other
rate unchanged.
A new family is added by the checklist in [moves/adding_a_move.md](moves/adding_a_move.md).
All three kernels add the **storage multiscale sweep** (`storage_ms_sweep!`,
probability 1/2 per sweep: plateau and shape moves over dyadic blocks), the **long-range
storage sweep** (`storage_long_sweep!`, probability `storage_long` per sweep: `slong` plateaus
over any cyclic interval and the level move `slevel`, per storage unit, cells in parallel; then
the interval storage pairs `sxlong` in rounds of disjoint footprints;
[moves/storage.md](moves/storage.md)) and, in the adaptive kernel, the **learned temporal
shapes** (`adaptive_block_sweep!`). Blocks of one length are updated in parallel by parity of
the block index.

### 4.3 Locality

Every cost term of a step is local: a unit term, a line term, a balance term per node. A move
is a sparse direction given both as node-value entries and as coefficient changes; its cost
change is evaluated in time proportional to its size from the caches of `LocalCost` (node
values, residuals, line flows, cell sums), and its commit updates coefficients and caches
together. For the plain model, the terms touched by a move are recorded at $t = 0$ and
reused for every evaluation of the walk (`eval_cached!`); moves with the model-completion
terms (losses, soft limits, lossy links, vintages, commitment, weather entries) use the plain
evaluation. Measured costs per move are in [performance.md](performance.md).

## 5. Invariance and compensation

### 5.1 Why each move is invariant

A line draw is a Gibbs update along a curve, valid when the curve and its parametrization are
the same from every point on it. Three constructions guarantee this:

1. **Fixed directions.** The direction depends only on quantities that the move does not
   change: the capacities and the weather (fixed during an operation sweep), the network,
   fixed family probabilities, and tables that are frozen between adaptation updates (modes,
   shapes, response coefficients, flow frames computed from $(x,k)$). Selection rules such
   as "skip if the unit's injection scale $|\kappa| = |A k| < \kappa_{\min}$" also depend on
   $(k,x)$ only.
2. **Coordinate lines of a reparametrization.** The share-preserving compensation (§5.2)
   uses the current dispatch, but only through quantities that stay constant along the
   curve. The curve is a coordinate line in coordinates $(\sigma, e, t, \dots)$, and the
   Jacobian of that coordinate change enters through §3.4.
3. **Group moves.** The capacity moves act by a one-parameter group on the state and draw
   the group parameter from the generalized Gibbs conditional, which carries the group's
   Jacobian (Liu and Sabatti, 2000).

Within a step the moves run in a random order drawn independently of the state, and the
number and families of the random moves are drawn independently of the state, so a sweep is
a composition and mixture of invariant updates.

### 5.2 Compensation rules

A transport move (an angle, a transport mode, a state of charge in time) changes the
requirement at the nodes it touches. What the other units do is the compensation rule:

- **None.** The residuals absorb the change: the move exports from surplus into shortage.
  Plain angle moves (`potential`) and the families with suffix `0`.
- **Additive.** A coarse move adds the same change to every unit of a cell or block, and the
  finer structure is kept as an offset (`cell`, Haar atoms, storage plateaus).
- **Unit.** A named unit at the node takes the change (`spair`; the device's UNIT rule).
- **Share-preserving pro-rata** (`prorata!`). At node $n$ and step $\tau$, let $S_n$ be the
  dispatchable generators with $|\kappa_i|\ge\kappa_{\min}$, $p_i = \kappa_i\omega_i$ their
  unclamped outputs, $P = \sum p_i$, and $Q = r_{n,\tau} + \sum\kappa_i\,\mathrm{cl}(\omega_i)$
  the requirement they face, which changes by exactly $g\,t$ along the move. The move keeps
  the shares $\sigma_i = p_i/P$ and the excess $e = P - Q_+$ fixed and sets
  $$P(t) = (Q + g t)_+ + e,\qquad p_i(t) = \sigma_i P(t).$$
  At deficit nodes ($Q>0$) the dispatch follows the requirement and the residual stays; once
  the node is in surplus the dispatch stays at $e$ and the change goes into the residual.
  The entries are kinked at $t^\ast = -Q/g$, a primary breakpoint. $(\sigma, e)\leftrightarrow p$
  is a bijection at fixed $Q$, so the curve is a coordinate line, with Jacobian
  $$J_{\rm curve}(t) = \prod_{n}\Big|\frac{P_n(t)}{P_n(0)}\Big|^{|S_n|-1}.$$
  With losses or lossy links $Q(t)$ is the actual requirement along the move
  (requirement-tracking variant, §3.3), and the Jacobian is evaluated from the move at the
  drawn $t$. Without them $Q$ must change by exactly $g\,t$; for storage moves, whose actual
  injection is bent by efficiencies, power limits and the box, $Q$ carries the nominal
  injection of the moved units (`soc_qadj`, [moves/storage.md](moves/storage.md)).
- **Learned linear** (`compensate_learned!`, families with suffix `L`). Unit $i$ at node $n$
  changes its output by $\beta_{n,c,i}\,g$ and the residual takes $(1-\sum_i\beta_{n,c,i})g$,
  with coefficients learned by regressing each unit's output on the node requirement over the
  chain's own states, separately for the proxy classes $c$ (deficit or surplus, from the
  flow frames at $(x,k)$). The coefficients are fixed between updates and independent of $v$,
  so the direction is constant along the line and no Jacobian enters. They capture merit
  order (one unit taking most of a change), which pro-rata cannot.

## 6. Capacity moves

At the complete level, one `kscale` and one `kpeak` move per investable unit, in random
order (`kscale_sweep!`, `kpeak_sweep!`; [moves/capacity.md](moves/capacity.md)):

- **`kscale`**: the scaling group $(k,\omega)\mapsto(e^u k, \omega\,c(k)/c(e^uk))$ of one
  unit ($c$ the clamp to the capacity box; $(e^uk, e^{-u}\omega)$ inside it), which keeps its
  dispatch fixed, also for a raw capacity above its box; $u$ from the generalized Gibbs
  conditional with factor $e^{u}(c(k)/c(e^uk))^{n}$ by slice sampling.
- **`kpeak`**: $k\to k+\delta$ for a dispatchable unit together with its dispatch on the
  node's peak steps (top 5 % of the proxy requirement, the output there changes by
  $A(c(k+\delta)-c(k))$), the residual or the other units absorbing the change; $\delta$ from
  the generalized Gibbs conditional with $(c(k)/c(k+\delta))^{T}$ by slice sampling.

- **`kop`** (`kop_sweep!`, with `kop = true`): $k\to e^u k$ for one renewable or storage unit,
  or for all units of such a carrier with a common $u$, at fixed utilizations, so that the
  unit's operation scales with its capacity; the dispatchable generators at its node follow
  share-preservingly (the pro-rata rule of §5.2 at every step), with the group Jacobian and
  the pro-rata Jacobian in the line density; $u$ by slice sampling. One move per unit and one
  per carrier per pass.

All include capex, the bias change and the measure term. At $\alpha = 1$ a capacity is held
within about $1/(\beta_v c)$ of its peak dispatch, so moves of $k$ at fixed dispatch move it
little; `kpeak` moves capacity and peak dispatch together. A renewable capacity at fixed
dispatch is held by its node balance to a few MW; `kop` moves it with the dispatchable
generation of its node following.

**Incremental bias.** The bias is evaluated at the finest level, so a capacity proposal changes
$\hat g$ and $\bar{\mathbb E}\hat g$ at every step and sample step. A per-state cache
(`BiasCache`, built once per fine step and shared by the capacity and weather moves, since the
operation sweep changes neither $k$ nor $x$) holds the features $D_t$, $R_t$, $u_{1,t}$ and the
table values $F_1$, $F_2$ per step for the chain's weather and per sample step, the firm
capacity, the raw and clamped node capacities, the clamped coefficients of $\hat h$ and the
three sums. A proposal of one unit (`bias_delta_k`) changes $R_t$ by $A_{i,t}\Delta$ at every
step for a non-gas generator (steps with $A_{i,t}=0$ are skipped), the firm capacity for gas and
storage, and the clamped coefficients of its node for $\hat h$; the commit updates the cache and
recomputes the sums from the per-step values. The measure term of `kpeak` is the unit's term
of $\log J$ alone. A `kop` carrier move changes several units at once; its bias change is the
sum of the unit changes applied in sequence to a copy of the cache (exact, since each unit
change is evaluated from the updated cache), and the commit applies the same sequence to the
cache itself. The global `evaluate!` equals the incremental sums to round-off.

## 7. Weather and bias

### 7.1 Learning the bias

- **Level-0 warmup** (`run_level0!` with `learn = true`, `warmup0` iterations). A chain on
  level 0 with the level-0 kernel: in random order, a local operation sweep, slice updates of
  the level-0 weather coordinates at fixed operation (five colour classes of steps), the
  weather move with pro-rata shear, slice updates of the level-0 capacity coefficients, FFBS of
  the regimes, and the carrier scaling move ([moves/weather.md](moves/weather.md),
  [moves/capacity.md](moves/capacity.md)). The bias is evaluated at level 0 ($\ell_b = 0$;
  the per-step part of $\hat g$ enters the step terms of the slice updates). From iteration
  `bias_start` its scores enter the bias regression every iteration, with a refit every
  `bias_every` iterations. With `init = "lp"` the zero-temperature scores of the reference LP
  (its duals) enter first as pseudo-observations of weight `bias_warmstart`. An optional
  reflective descent of the operation (`descent_iters`) precedes the warmup.
- **Lift** (`lift_bias!`, both drivers, [ensemble.md](ensemble.md) §4.4). Before the finest
  chains start, the bias is lifted to their level: $\theta$ and $\bar D$ kept, $a$ and $c$
  prolongated (new detail coefficients zero), the regression sums reset, the sample steps of
  $\bar{\mathbb E}$ redrawn at the finest level from a stream of their own (seed + 4243), so
  that both drivers get the same ones.
- **Online refits** (`bias_online`, default on). In the finest chains the scores
  $s_x = -\beta_v\nabla C$ in the weather state coordinates of the finest level ($z_0$ and all
  detail coordinates) and $s_k = -\beta_v\partial_k C + \partial_k\log J$ on all capacity
  coordinates of the level enter the regression every `bias_online_every` steps during the
  finest-level warmup. The coefficients are refitted (monotone QP) only when the observation
  count reaches $2^j\ge$ `bias_online_min` (default 16): the sums are reset at the lift, and a
  refit from a few cold-start states would replace the lifted level-0 fit by an extrapolation of
  them, so the lifted fit stays until enough fine data exist. Both drivers pool the scores of
  all chains and freeze the bias at the end of the warmup, so production samples one fixed
  target. After a refit every chain's state is re-evaluated.
- **Trust region** (`bias_trust` $=\tau$, default 1; 0 turns it off). A refit replaces the
  coefficients $(\theta, a, c)$ by the convex combination $(1-\lambda)$ old $+\ \lambda$ new with
  the largest $\lambda\le1$ for which the change of $\hat b$ minus its mean over the
  last 32 observed states stays below $\tau\max(S, n_k)$, $S$ the spread (max − min) of the old
  $\hat b$ over those states and $n_k$ the number of capacity coordinates (a floor for an unfitted
  bias). $\hat g$, $\bar{\mathbb E}\hat g$ and $\log J$ are linear in the coefficients and
  $\hat h$ nearly so, so $\lambda$ is the bound over the deviation of the full step, halved until
  it holds (`trust_step!`). A convex combination of two fits satisfies the linear constraints of
  the monotone fit. The bias stays a fixed function between refits and is frozen after the
  warmup. With $\tau = 1$ a refit can at most double the spread of $\hat b$ over the recent
  states; a full refit from a few states far from equilibrium would change it by 10 to 50 times
  that, and with the wide capacity move `kop` such a step lets a capacity run away (gas to tens
  of TW in a test run).
- **Post-correction** (`bias_post`, default on). Every production sample records what is
  needed to evaluate $\hat b$ and to refit it at the finest level, not the raw weather: the
  features $u_{1,t}$ per step, the clamped node capacities and clamped capacity coefficients,
  the bias in use, and the sufficient statistics of its scores (its contribution to the normal
  equations). After the run, $\tilde b$ is refitted from the warmup's sums plus the production
  statistics and every sample gets the self-normalized weight
  $w_i\propto\exp(\hat b_{\rm used}(x_i,k_i)-\tilde b(x_i,k_i))$. The weight ESS measures how
  far the bias in use was from the final one; weighted quantities and weather diagnostics
  are written next to the unweighted ones. Weights cannot repair regions the chain did not
  visit.

> **Status: open.** With the confined $\hat h$, the exact measure term, $\theta_{u_2}\le0$, the
> refit floor and the trust region, short $\alpha = 1$ runs of `complete40` (2 chains, 400
> level-0 iterations, 16 + 10 fine sweeps) stay bounded: the level-0 chain no longer blows up
> after its refits, capacities, costs and storage settle, and the frozen $\hat b$ does not drift
> in production. With `kop` one chain's gas still drifts outward slowly (1.2 to 1.4 TW over 18
> sweeps): inside its data range a linear $\hat h$ cannot follow the convex
> $\log Z^u\sim -n_{\rm conf}\log k$, so it pushes toward the upper edge of the range, which the
> next refit extends. The bias is far from converged in such runs: refits still change it by
> $10^6$ nats over the visited states, and the post-correction weight ESS is 1.

### 7.2 Weather moves at the fine level

With `fine_weather` $= f > 0$, every fine step ends with `fine_weather_sweep!`: the centered
regime update (FFBS given the anomalies of all levels), then a fraction $f$ of the innovation
coordinates of every level and of the level-0 coordinates, each moved by an exact line draw
with its Gaussian prior and share-preserving compensation of the dispatchable units at every
node whose requirement changes. At $\alpha = 1$ every weather coordinate moves $\hat b$ (it is
evaluated at the finest level): the exact draw of the bias-free line conditional is followed by
a Metropolis correction with $e^{-\Delta\hat b}$ together with the Jacobian (an independence
Metropolis step with the exact conditional as proposal), and on rejection $t = 0$. The bias
change comes from the bias cache: $D_t$ and $R_t$ change at the steps of the move's window,
$\bar{\mathbb E}\hat g$ and $\hat h$ do not change (`bias_delta_weather`); an accepted move
updates the cache at those steps. Because the change is local to the window, moves in coarse
blocks far enough apart run in parallel at every level, $z_0$ included; the sums of the cache
are recomputed after the sweep. Then `fine_weather_blocks` joint block moves of $z_0$ per sweep
(in expectation) shift $z_0$ along a fixed joint direction over 16, 64 or 365 level-0 steps,
with the exact VAR quadratic of the block and the same compensation and bias correction, in
sequence ([moves/weather.md](moves/weather.md)); they mix the annual and seasonal means, which
the coordinate moves barely move. With `fine_weather = 0` the weather stays at its initial draw
at the fine level.

### 7.3 Canonical CO₂ price

With `co2_cap > 0` the CPU driver tunes $\mu$ during the finest-level warmup by
$\mu\leftarrow\mu\exp(0.5\,n^{-0.6}\,\mathrm{clamp}(E_n/\Gamma - 1,\pm1))$ after sweep $n$
($E_n$ the current annual emissions averaged over the chains, $\Gamma$ the cap), re-evaluates
every chain, and freezes $\mu$ for production.

> **Status: deviation.** The device driver ignores `co2_cap`.

## 8. Adaptation

Adaptation follows diminishing adaptation (Roberts and Rosenthal, 2007). The adaptive kernel
(`AdaptiveModes`) observes its own state after every sweep: the angle fluctuation covariance
(running mean per node and step, weight $1/n$), the intra-block fluctuation profiles of
dispatch and state of charge per dyadic block length and meteorological season, and the moments of
(node requirement, unit output) per node and proxy class. At observation counts $2^j\ge4$ it
recomputes the transport directions (leading eigenvectors of the angle covariance), their
localized Haar pieces, the temporal shapes and the response coefficients $\beta$. Every
kernel in the sequence is invariant for the target, the fixed families always stay in the
mixture, and the change between updates vanishes as the estimates converge. The online bias
uses the same geometric schedule. Both stop at the end of the finest-level warmup
(`mlda_warmup`): the production sweeps use one fixed kernel and one fixed target. Slice widths
of the level-0 kernel adapt during the level-0 warmup only.

## 9. Multi-chain runs and convergence

The CPU driver `run` (`run_chains`) runs `nchains` $= K$ finest-level chains for `mlda_warmup` +
`nsamples` sweeps each (one after another per sweep on the CPU, each sweep threaded
internally; as the batch dimension on the device). Starts are cycled from `chain_starts`:
`"low"` and `"high"` scale the start capacities by 0.5 and 2 (clamped to the boxes), the others
use the set-up start. With fine weather moves every chain also draws its own weather year;
otherwise all chains share the weather of the set-up state. Every chain has its own sampler
and random stream; the bias is shared (§7.1).

The monitored quantities are the capacity totals per carrier and per country, capex, opex,
total cost, shed, penalty, regime frequencies, the annual means of the level-0 weather,
emissions, the mean line loading, the mean H₂ state of charge, and the mean and load-weighted
sampled price. The diagnostics (`src/convergence.jl`, Vehtari et al. 2021) are the
rank-normalized split-$\hat R$ (maximum of the bulk and the folded version), bulk and tail
ESS, and the MCSE of the mean relative to the posterior standard deviation, computed on the
window after the warmup. The warmup end is detected as the first checkpoint from which the
sliding-window $\hat R$ stays below `rhat_screen`. A run converges when every quantity has
$\hat R <$ `rhat_max` (1.01), bulk and tail ESS $\ge$ `ess_min` (400) and MCSE/sd $\le$
`mcse_max` (0.05). The criteria and their use are in [validation.md](validation.md).

## 10. Sampled prices

By the score identity, the marginal price at node $n$ and step $\tau$ is
$\partial F/\partial d_{n,\tau} = \Delta t\,\mathbb E_\pi[c_{\rm shed}\mathbb 1(r>0) -
c_{\rm dump}\mathbb 1(r<0)]$: the sampled price is the imbalance probability weighted by the
shed and dump prices. `PriceAcc` accumulates it per node and step and `RegionPriceAcc` per
country, system and meteorological season, over the production sweeps of all chains of the
CPU driver. At finite $\beta_v$ the sampled prices have a
heavier upper tail than LP duals, because near-optimal states shed occasionally where the LP
sits at the margin.
