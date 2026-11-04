# Storage moves: transport in time

Storage moves shift energy in time by changing a state-of-charge trajectory, with the
dispatch at the storage unit's node compensating at every step whose power changes. This
document covers the single-instant move `tsoc` and its learned variant, the storage
multiscale plateau and shape moves over dyadic blocks, the long-range plateaus `slong` and
the level move `slevel`, and the storage pairs across nodes `sxpair` and `sxlong`. All need the state-of-charge storage form (`storage = "soc"`). The
single-instant storage–generator pair `spair` is in [pairs.md](pairs.md), the learned shapes in
[learned.md](learned.md).

**Requirement of the compensation.** The share-preserving compensation of every storage move
(`prorata!`, [../sampler.md](../sampler.md) §5.2) keeps the shares and the excess over a
requirement $\tilde Q$ that must change by exactly $g\,t$ along the move. The actual
injection of a storage unit is bent by its efficiencies, its power limits and the box, so
$\tilde Q$ uses the nominal injection of the moved units instead,
$\tilde Q = Q + \sum_u \big(\kappa_u (Y_{u,\tau+1} - Y_{u,\tau}) - k_u(p^+_u - p^-_u)\big)$ with
$\kappa_u = k_u h_u/\Delta t$ and the unclamped states of charge $Y$ (`soc_qadj`); the cost
still uses the actual injection. With requirement-tracking kinks (losses, lossy links) the
actual requirement is tracked and no correction is needed. Without it the curve through
$s(t)$ is not the curve through $s$ (a plateau moved back from $s(t)$ misses $s$ by up to 0.9
in the coefficients, and the Jacobians of the two directions do not cancel);
`test/test_local.jl` checks this reversibility for `tsoc`, the plateaus and `slong`.

## `tsoc` and `tsocL`: one instant

**Direction.** A node $n$ uniformly and one of its storage units; its state of charge at
instant $\tau$ rises by $t$. That is extra charging in step $\tau-1$ (requirement $+\kappa_s
t$) and less withdrawal in step $\tau$ ($-\kappa_s t$), with $\kappa_s = kh/\Delta t$
(efficiencies left to the cost). The dispatchable generators at $n$ follow at both steps:
share-preservingly for `tsoc` (two kinks, Jacobian step), with the learned coefficients for
`tsocL` (no Jacobian).

**Selection.** Uniform node and unit; $\kappa_s<\kappa_{\min}$ skips. **Cost.** A linked
two-step move at one node. Rates: `tsoc` 0.1 (flow and adaptive kernels), `tsocL` 0.05
(adaptive). **Device:** `:tsoc` and `:tsocL` (no requirement tracking for `:tsoc`).

## Plateau and shape moves over dyadic blocks

`storage_ms_sweep!` runs, with probability 1/2 per sweep of every fine kernel, a
sweep of block moves. For each dyadic length $L\in\{2,4,\dots,128\}$ with $L<T/2$ (random
order), a random offset defines blocks of $L$ steps; blocks of one parity run in parallel,
with 8 moves per block. Each move acts on one storage unit or, with probability 1/2, on all
units of one storage carrier in one cell (an equal utilization change for every unit of the
class).

- **Plateau.** The state of charge at every interior instant of the block rises by $t$:
  energy moves from the step before the block to its last step. Only the two end steps change
  power.
- **Shape.** With probability 1/2 (when a flow basis exists), the change is a profile $\phi$
  over the $L+1$ instants of the block, zero at both ends, taken from the proxy storage shapes
  of the unit's cell (leading principal components of the proxy intra-block state of charge,
  [../model.md](../model.md) §2.5). Every step of the block changes power.

**Compensation.** Share-preserving pro-rata at every step whose power changes, with the
Jacobian step; in the adaptive kernel, half of the sweeps use the learned coefficients
instead once they exist. **Selection.** Lengths, offsets, units and shapes depend only on
fixed probabilities, $(k,x)$ and the precomputed shapes. **Cost.** $L+1$ linked steps per move.

**Use.** States of charge relax slowly under single-instant moves because their changes have
to propagate step by step; block moves move inter-block energy in one draw. The box of the
state of charge limits the inter-block range of short storage and the power limit bounds the
amplitudes, so which lengths matter depends on the storage class.

**Device.** `:sblk{L}` (share-preserving) and `:sblkL{L}` (linear), $L$ = 2, 4, 8, 16, 32, for
one storage unit (`device/src/blocks.jl`).
* *Profile.* Each site has a profile from a host table: the plateau and optional shapes with
  zero ends.
* *Compensation.* It runs at every step whose power changes.
* *Rounds.* Blocks of one parity × one storage slot per node.

> **Status: deviation.** The device's share-preserving rule of `:tsoc` and `:sblk{L}` uses the
> actual requirement, not the nominal one above.
>
> **Status: missing.** The cell–class aggregates of the block moves; the proxy and learned
> shapes as device tables (any shape can be set with `set_profiles!`); the driver does not
> schedule the block families yet ([../device.md](../device.md)).

## `slong` and `slevel`: long-range plateaus and the level of a trajectory

`storage_long_sweep!` (`src/storage_long.jl`) runs once per operation sweep with probability
`storage_long` in every kernel (after the storage multiscale sweep).

- **`slong`.** For one storage unit, the state of charge rises by $t$ at the cyclic interval of
  instants $a+1,\dots,a+L$: energy moves from step $a$ to step $b = a+L$, wrapping around the
  year. Only the two end steps change power; the dispatch at the unit's node compensates there
  (share-preserving pro rata with its Jacobian step, or in the adaptive kernel half of the
  sweeps the learned rule).
- **`slevel`.** Every instant rises by $t$ ($L = T$): the whole cyclic trajectory shifts. No
  power changes while the trajectory stays in the box, so the line is flat between the box
  contacts and no compensation enters.

**Selection.** Every storage unit with at least `storage_long_hours` hours of storage and
$\kappa_s \ge \kappa_{\min}$ gets one `slevel` and `storage_long_moves` `slong` moves per sweep,
with $a$ uniform on $1,\dots,T$ and $L$ uniform on $1,\dots,T-1$, in random order. The moves of
one balance cell run on one thread, the cells in parallel: a move touches only its unit, the
dispatch at its node and the balance of the node's cell. The selection uses only $(k,x)$ and
fixed probabilities.

**Cost.** The interior steps (both instants raised) are a compact list of storage terms, not
step workspaces. For $t$ in the interval over which every raised instant stays in $[0,1]$
their cost change is exactly zero and their secondaries are constant, so a walk evaluation
costs the two end steps; beyond it each interior step whose instants leave the box is
evaluated exactly (storage term and the balance of its cell), and its clamp points are primary
breakpoints. At 40 nodes daily one move costs about 30 µs for hydro (the interval always
holds), 50 µs for the large PHS units and 130 µs for H₂ (box contacts within the year); a
sweep with 32 moves per unit costs about 0.1 s at 4 threads. Batteries and the small PHS units
(6 h and less) are left out by `storage_long_hours = 24`: their trajectories touch the box
within days, so long plateaus are box-limited and expensive.

**Use.** The seasonal storage directions are nearly flat between walls: moving energy between
two steps changes the cost only at those steps, and the level of a trajectory changes it only
at the box. A plateau's step is limited by the headroom of its two end steps, whatever its
length, so the seasonal modes need many long moves per unit and sweep; the dyadic blocks give
about 0.1 moves per unit and sweep at $L$ = 64–128, at 1.3 ms each, and never move the level.
On `chains40` the sweep adds 11–16 % to an operation sweep and brings the integrated
autocorrelation times of the hydro level and totals from 60–90 sweeps to 1–3 and of the H₂
totals and patterns from 20–90 to 5–15 (ledger E21).

> **Status: missing.** The hydro seasonal shape and the large PHS units still relax over about
> 100 sweeps (ledger E21): the end steps of `slong` stop where the node's own dispatchable
> units reach their capacity. Carrying the change to a neighbour through the `xlocc`
> templates gave smaller steps and was not kept.

**Device.** Missing.

## `sxpair` and `sxlong`: storage pairs across nodes

"Store more here, less there, with the flows compensating": a second storage unit, not a
generator, absorbs the end-step imbalance (`src/storage_cross.jl`).

**Direction.** Unit A's state of charge rises by $t$ (a fraction of $E_A = k_A h_A$) and unit
B's falls by $\rho t$, $\rho = E_A/E_B$, so the two shift the same nominal energy. At every step
whose power changes the nominal injection changes by $+c\,t$ at A and $-c\,t$ at B,
$c = \pm\kappa_A$ ($\kappa_A = E_A/\Delta t$).

- **`sxpair`** (single instant): the instants $\tau$, steps $\tau-1$ ($c = -\kappa_A$) and
  $\tau$ ($c = +\kappa_A$); a family of the local sweep (linked two-step move).
- **`sxlong`** (interval): the cyclic instants $a+1,\dots,a+L$, $1\le L\le T-1$, end steps $a$
  ($c=-\kappa_A$) and $b = a+L$ ($c = +\kappa_A$); built as `slong`, with the interior steps of both
  units as compact terms (exactly zero cost change while their instants stay in the box).

**Partners and selection.** Only units with at least `storage_long_hours` hours of storage
(`LM.sxunits`; κ ≥ κ_min). A is uniform among them (`sxpair`) or every one of them gets
`storage_xlong` interval moves per sweep (`sxlong`, $a$ uniform on $1..T$, $L$ uniform on
$1..T-1$). B is another such unit at A's node or at a partner node of A's node (within
`storage_xpair_hops` hops, the `xloc` partner sets), chosen with probability
$\propto \min(k_A, k_B)$ (`sx_partner`): the end steps stop at the power headroom of the weaker
unit, so a pair of a 3 GW reservoir with a 20 MW one moves little. Any carriers pair (hydro with
hydro, PHS or H₂). The selection uses only $k$, the network and fixed probabilities.

**Compensation.** At each power-changing step:

- **Flows.** For a cross-node pair, the angle change $c\,\Delta\theta$ of the pair's ball
  template (`xloc_directions` with `storage_xpair_hops` and `storage_xpair_ball`, built once per
  level: `LM.sxpart`, `sxnod`, `sxval`, `sxlkn`, `sxlkg`) carries the nominal change from B to A;
  every node of the ball keeps its residual. The leak into the first layer outside the ball is
  compensated share-preservingly at every leak node (`prorata!`), as in `xlocc`. With a 1-hop
  ball about a quarter of the dipole leaks (median at 40 nodes; half with the pair only, a tenth
  with 2 hops).
- **Losses.** The actual injection of a unit differs from its nominal one by its efficiencies.
  For a unit that only discharges (reservoir hydro, $\eta_{\rm in} = 0$) the expected loss is the
  fixed share $\lambda = 1-\eta_{\rm out}$ of the nominal change; the dispatch at the unit's node
  follows $\lambda c$ (A) and $-\lambda c$ (B) share-preservingly, with the requirement
  $\tilde Q = Q + I - (1-\lambda)N$ ($I$, $N$ the unit's actual and nominal injection with the
  unclamped state of charge; `sx_qloss`, the $\lambda$-form of `soc_qadj`), which changes by
  exactly $\lambda c\,t$. In the discharge regime the residuals at A's and B's nodes then do not
  change. Units that can charge have $\lambda = 0$: half of their steps charge and half
  discharge, so no fixed share fits; their losses go into the residuals of their nodes. Without
  the loss compensation the 10 % hydro loss went into shed at B's node and halved the hydro
  steps (E27). A same-node pair has no flows and one loss compensation with $(\lambda_A -
  \lambda_B)c$.

In the adaptive kernel half of the long-range sweeps use the learned rule for the leak and
loss compensations instead. **Jacobian.** The product of the pro-rata factors of all leak and
loss compensations at both steps (independence Metropolis step). The curve through $s(t)$ is the
curve through $s$, also across the compensation kinks (`test/test_local.jl`).

**Scheduling.** `sxpair` runs with the steps of one parity class in parallel (it touches two
steps). An `sxlong` move touches every step of its two units, so the interval pairs run in
rounds of moves with disjoint footprints (the balance cells of the ball, the other ends of its
lines and the leak nodes, `LM.sxfoot`; first fit in the random order of the moves,
`sx_rounds`), after the `slong` moves of the long-range sweep; the coefficient changes are
summed per thread (the node values of different cells share coarse Haar coefficients).

**Cost.** `sxpair`: two steps with the ball's lines and the leak nodes. `sxlong`: as `slong`
for two units, about 0.1 ms per move at 40 nodes daily. At the defaults (`storage_xpair` 0.1 on
the plain-family scale, about 60 moves per unit and sweep; `storage_xlong` 8) the two add about
6 % and 9 % to an operation sweep of `chains40`.

**Use.** See ledger E27.

**Device.** Missing.
