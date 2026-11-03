# Storage moves: transport in time

Storage moves shift energy in time by changing a state-of-charge trajectory, with the
dispatch at the storage unit's node compensating at every step whose power changes. This
document covers the single-instant move `tsoc` and its learned variant, and the storage
multiscale plateau and shape moves over dyadic blocks. All need the state-of-charge storage
form (`storage = "soc"`). The single-instant storage–generator pair `spair` is in
[pairs.md](pairs.md), the learned shapes in [learned.md](learned.md).

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

`storage_ms_sweep!` runs, with probability 1/2 per sweep of the flow and adaptive kernels, a
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

> **Status: missing.** The cell–class aggregates of the block moves; the proxy and learned
> shapes as device tables (any shape can be set with `set_profiles!`); the driver does not
> schedule the block families yet ([../device.md](../device.md)).
