# Transport moves in space

Transport moves change line flows along a direction in angle space, with the dispatch at the
affected nodes compensating by a rule: share-preserving pro-rata (with the Jacobian step),
learned linear coefficients, or none. This document covers the node-angle move `tpot`, the
face-projected transport modes `tmode`, the binding-line normals `tline`, the localized modes
`tcut`, their uncompensated and learned-compensation variants. They need the angle field
complete at the level. See [README.md](README.md), the compensation rules in
[../sampler.md](../sampler.md) §5.2, and the transport basis in [../model.md](../model.md)
§2.5.

## Frames from $(x,k)$

The flow kernels keep per-step **frames** (`FlowFrames`), recomputed whenever $(k,x)$
changes and never from $v$: the proxy injections at the current weather and capacities are
projected onto the line limits by the batched ADMM, which gives the predicted binding lines of
every step, and the transport modes are projected onto the null space of the binding rows'
loadings, $d\mapsto(I - W_B^\top(W_BW_B^\top)^+W_B)\,d$. The frames also give the proxy class
(surplus or deficit) of every node and step used by the learned compensation. Because they
depend only on $(k,x)$, every move built from them is exact whatever their quality.

## The families

**`tpot`.** The angle of a uniformly chosen node $n$ at step $\tau$ moves by $t$. The lossless
requirement change $g_m$ at $n$ and its neighbours (from the changed line flows) is
compensated share-preservingly at each of those nodes, with one kink per node and the Jacobian
step. Cost: the node's lines and $\deg(n)+1$ compensation sites. Rate 0.1 (flow and adaptive
kernels). Device: `:tpot`, scheduled by a distance-2 colouring; refuses models with line
losses or lossy links, which need requirement tracking.

**`tmode`.** One transport mode $k$ uniformly, projected onto the face of the step's predicted
binding lines; share-preserving compensation at every node whose requirement changes.
Dense: every node and line, $O(N + n_{\rm lines})$ per move. Rate 0.05. Device: missing (a
dense family at a low rate in the design).

**`tline`.** One predicted binding line $e=(a,b)$ of the step uniformly; the direction is the
injection pair $+1$ at $a$, $-1$ at $b$ re-routed through the network, $\Delta\theta =
L^+(e_a - e_b)$, with share-preserving compensation at both ends. It moves flow along the
normal of a binding constraint. Dense re-routing, $O(N)$. Rate 0.05. Device: missing
(planned as a local re-routing template).

**`tcut`.** A localized transport move. Each transport mode is expanded in the sparse
graph-Haar angle basis, $d = Hc$, and its dominant atoms are kept (those carrying 90 % of
$\|c\|^2$, at most 8 per mode; `localize`). A move picks a mode uniformly and one of its atoms
with probability $\propto c_j^2$, and moves the angles along that atom. A coarse Haar atom is
constant on each of the two cells it contrasts, so only lines leaving those cells change flow,
and the compensation is needed only at their boundary nodes. Share-preserving compensation
with the Jacobian. At 40 nodes a `tcut` draw takes about 46 µs against 10 ms for a full
`tmode`. Rate 0.15. Device: missing (planned as templates built on the host).

**`tmode0`, `tline0`.** The `tmode` and `tline` directions without compensation: the residuals
at the affected nodes absorb the change, so these moves export from surplus into shortage.
Rates 0.05 each.

**Learned-compensation variants** (adaptive kernel): `tpotL` (0.05), `tmodeL` (0.03),
`tlineL` (0.05), `tcutL` (0.1). The same directions with the learned linear rule instead of
pro-rata: each unit at an affected node takes $\beta_{n,c,i}$ of the requirement change, the
residual the rest; no kink and no Jacobian. They become active once the response coefficients
have been learned (before that the coefficients are an equal split). Device: `:tpotL`, with
static coefficients (equal split, or values passed in).

## Use

Near equilibrium at fixed $(k,x)$ the plain families already mix the operation, and the
transport families add little per second at 40 nodes. They pay off on larger networks (at
256 nodes daily a flow sweep does about twice the work of a plain sweep, at 2.7 times its
cost) and wherever the weather or the capacities move, because compensation lets transport
follow without shedding. Moves that hold every residual fixed cannot remove the shed that
dominates far starts; relaxation from such starts comes from the plain families.
