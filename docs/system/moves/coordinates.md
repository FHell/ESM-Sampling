# Coordinate moves: atoms, units, angles

Single-coordinate moves are the irreducible core of the move set: together they reach every
direction of the operation, so every richer family is an addition on top of them. This
document covers the basis atoms of every operation field, the single-unit move, the node-angle
move and the commitment atoms. See [README.md](README.md) for the other families and
[../sampler.md](../sampler.md) for the line sampler.

## `atom`: one basis vector of a field

**Direction.** The coefficient $j$ of field $f$ at step $\tau$ changes by $t$; the node values
change by $t$ times the basis column $U_{\cdot j}$. At the complete level the finest Haar
atoms touch a few nodes (a contrast of two children), the coarse ones a whole cell
(a contrast of two halves of a country, or of two countries). Fields: angles, link
utilizations, dispatch utilizations, commitment, states of charge. A state-of-charge atom at
instant $\tau$ moves the storage terms of steps $\tau-1$ and $\tau$ and is evaluated as a
linked two-step move.

**Selection and invariance.** Systematic: every atom of every field at every step, once per
sweep, in a random order shared with the random moves. The direction is a fixed basis vector.

**Compensation.** None: the residuals absorb the injection change, so atoms change the
balance. That is what relaxes far starts (shed and dump), which balance-preserving moves
cannot remove.

**Cost.** Proportional to the atom's support. **Device:** the finest atoms are replaced by
node coordinates of the same fields (`:gen` for dispatch, `:uc`, `:soc`, `:link`), a
reparametrization with constant Jacobian. Coarse atoms of the dispatch fields are planned as
cell templates ([cells.md](cells.md)).

## `unit`: one unit at one node

**Direction.** A node $n$ uniformly, one of its units uniformly (a dispatchable generator, or a
storage charge or discharge field in the LP storage form), and the node value of that unit at
step $\tau$ moves by $t$. Only fields complete at the level.

**Compensation.** None. **Cost.** One unit term and one balance term. Rate 0.3 in every kernel.
On the device the `:gen` family is this move as a systematic coordinate sweep.

## `potential`: the angle of one node

**Direction.** $\theta_n$ at step $\tau$ moves by $t$; the flows of the lines at $n$ change,
and the residuals at $n$ and its neighbours absorb them. Node chosen uniformly; needs the
angle field complete.

**Compensation.** None: the move exports from one node into its neighbours and is pinned by
the balance where both sides have generation to spare. The compensated version is `tpot`
([transport.md](transport.md)).

**Cost.** The node's lines and the balance terms of their ends. Rate 0.1. **Device:**
`:theta`, scheduled by a distance-2 colouring of the node graph.

## Commitment atoms

With fleet unit commitment (`uc`), atoms of a commitment field move the committed fraction
at fixed output, which changes the no-load, start-up and wall terms of steps $\tau-1$ and
$\tau$. In every other move that changes a committable unit's output at the main step, the
commitment follows the output with the headroom kept (`ws.follow`); in the share-preserving
moves it follows with the same kinked map, so the invariants and the Jacobian are unchanged.
The linked previous step of a two-step move does not follow, so that parallel step classes
stay independent. **Device:** `:uc` (commitment at fixed output), and `:gen` lets the
commitment follow.
