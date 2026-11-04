# Pair moves: balanced shifts between two units

Pair moves shift injection between two units so that the nodal residuals stay fixed inside
the boxes. This document covers the same-node pair, the cross-node pair with global flow
re-routing, its localized versions, and the storage–generator pair across two steps. The storage pairs
across nodes (`sxpair`, `sxlong`), which use the localized re-routing templates of `xloc`, are in
[storage.md](storage.md). See [README.md](README.md) for the other families.

## `pair`: two units at one node

**Direction.** A node $n$ and two distinct units $u_1, u_2$ at it, uniformly; their
utilizations change by $\Delta\omega_1 = t/\kappa_1$ and $\Delta\omega_2 = -t/\kappa_2$, so
$t$ MW of injection move from $u_2$ to $u_1$. The node residual is unchanged while both stay
in their boxes.

**Selection.** Units and $\kappa = Ak$ depend on $(k,x)$ only; the move is skipped if
$|\kappa_i| < \kappa_{\min}$. **Compensation.** None needed. **Cost.** Two unit terms and
one balance term. **Use.** Re-dispatch between technologies at a node (merit order) near
equilibrium. Rate 0.25. **Device:** `:pair`, every pair of dispatchable units at a node, masked
where $\kappa < \kappa_{\min}$.

## `xpair`: two units at different nodes

**Direction.** Nodes $n\ne m$ in the same AC component, one unit at each, injection $+t$ at
$n$ and $-t$ at $m$ as for `pair`, and the angle change $\Delta\theta = t\,L^{+}(e_n - e_m)$
that re-routes the flow through the network ($L$ the line Laplacian), so every nodal
residual is unchanged. The re-routing is solved with sparse Cholesky factors of the grounded
Laplacian per component (`GridSolver`).

**Cost.** The angle change is dense: every line of the component, $O(N)$ per move. Rate 0.03.
**Device:** missing; to be replaced by local re-routing templates.

## `xloc` and `xlocc`: localized cross-node pairs

**Direction.** A node $n$ uniformly, a partner $m$ uniformly among the nodes within
`xloc_partner_hops` hops (default 1), one unit at each, injections $\pm t$ as for `xpair`.
The re-routing is solved only on the ball $B$ of nodes within `xloc_ball_hops` hops of $n$ or
$m$ (default 0: the pair only), with zero angle change outside $B$:
$L_{BB}\Delta\theta_B = e_n - e_m$. Every node of $B$ keeps its residual; a small dipole leaks
into the first layer of nodes outside $B$. The templates (partners, ball angles, leak nodes
and their requirement changes) depend only on the network and are built once
(`xloc_directions`). If $B$ is the whole component the exact global re-routing is used.

**Compensation.** `xloc`: none, the leak goes into the residuals of the leak nodes. `xlocc`:
the leak is compensated share-preservingly at every leak node, with the Jacobian step.

**Cost.** The lines of the ball and its boundary; at 256 nodes 4.7–8 times more mixing per
microsecond than `xpair`. Rates `xlocc` 0.08, `xloc` 0.04. **Device:** missing (planned as
local re-routing templates).

## `spair`: storage–generator pair across two steps (SoC form)

**Direction.** A node $n$, one of its storage units and two of its units $u_1$, $u_2$. The
state of charge at instant $\tau$ rises by $t$ (as a fraction of $hk$), which is extra
charging in step $\tau-1$ and less net withdrawal in step $\tau$; unit $u_1$ covers it at
$\tau-1$ with $\Delta\omega_1 = \kappa_s t/\kappa_1$ and unit $u_2$ gives way at $\tau$ with
$\Delta\omega_2 = -\kappa_s t/\kappa_2$, where $\kappa_s = kh/\Delta t$. Efficiencies are
left to the cost, so the direction does not depend on $v$.

**Compensation.** Unit rule. **Cost.** A linked two-step move: two storage terms, two unit
terms, two balance terms. Rate 0.3 in the SoC form. **Device:** `:spair`, with the units of
each site chosen on the host.
