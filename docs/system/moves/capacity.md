# Capacity moves

Capacity moves change investments together with the operation they constrain, as group
actions sampled from their generalized Gibbs conditionals. This document covers the
fine-level scaling move `kscale` and the peak-set move `kpeak`, and the two capacity moves of
the level-0 warmup chain. See [README.md](README.md), [../sampler.md](../sampler.md) §6, and
[../ensemble.md](../ensemble.md) §5 for the reference measure that enters their Jacobians.

## `kscale`: the scaling group of one unit

**Direction.** For one investable unit at the complete level, the scaling group
$(k, y_1,\dots,y_n)\mapsto(e^u k, e^{-u}y_1,\dots,e^{-u}y_n)$ acts on its capacity and all its
utilization coordinates: $n = T$ for a dispatchable generator ($2T$ with unit commitment,
whose commitment scales with the output), $T$ for storage in the SoC form ($3T$ in the LP
form), $n = 0$ for a renewable unit treated as load. Inside the boxes every injection and state
of charge is unchanged, so only capex, penalty weights, clamps and the bias change.

**Conditional.** $u$ is drawn from $\propto\pi(g_u s)\,e^{(1-n)u}$ (Liu and Sabatti, 2000),
including $\beta_k\Delta K$, $\beta_v\Delta C$, the bias change, and with
`measure = "physical"` the change of $n\log(k+\kappa_0)$; by slice sampling (stepping out with
width 0.3, shrinkage), truncated to $ke^u\ge\kappa_{\min}$ so that the selection rule
$k\ge\kappa_{\min}$ is invariant. A Metropolis form with fixed-scale proposals
$u' = u+\sigma z$ (`mh = true`) mirrors the device kernel.

**Selection.** Every investable unit whose capacity and utilization fields are complete, once
per pass, in random order. **Cost.** $O(T)$ terms of one unit plus its node's balance terms;
the bias change at $\alpha = 1$ costs an evaluation of $\hat b$.

**Use.** At $\alpha = 0$ and for renewables it moves capacities freely. At $\alpha = 1$ a
dispatchable capacity is pinned to its peak dispatch within about $1/(\beta_v c)$ (below 1 MW at
the defaults), so moves at fixed dispatch move it little; that is what `kpeak` is for.

**Device.** Metropolis form (`kscale_mh!`, `kscale_mh_loop`) for dispatchable and storage
units, and `kren_mh_loop` for renewables treated as load (Jacobian $e^{u}$; the move changes
the net load, the renewable running cost and capex); units in a fixed order, chains as the
batch dimension, with the bias at $\alpha = 1$. Renewables run first in a capacity pass,
because they move the net load and therefore the peak sets.

## `kpeak`: capacity with its peak dispatch

**Direction.** For one investable dispatchable unit at node $n$: the peak set $S$ is the top 5 %
of the steps by the proxy requirement at $n$ (load minus renewable output $Ak$), which
depends on $(x, k_{\rm ren})$ only and is unchanged by the move. Then
$$
k\to k+\delta,\qquad \omega'_\tau = \frac{\omega_\tau k+\delta}{k+\delta}\ (\tau\in S),\qquad
\omega'_\tau = \frac{\omega_\tau k}{k+\delta}\ (\tau\notin S),
$$
so the output rises by $A\delta$ on the peak steps and is unchanged elsewhere; the commitment
follows by the same affine map. On the peak steps either the residual absorbs the extra output
(**absorb**) or the other units at the node give way with their learned deficit-class
coefficients, renormalized over the positive ones (**learned**; only in the adaptive kernel,
once the coefficients exist, with probability 1/2).

**Conditional.** The maps form a one-parameter translation group, so $\delta$ is drawn from
$\propto\pi(g_\delta s)\,(k/(k+\delta))^{T}$ ($2T$ with unit commitment), including capex, the
operating cost, the bias and the measure term; by slice sampling with width $0.1k$, truncated
to $k+\delta\ge\kappa_{\min}$. A Metropolis form with a fixed per-unit scale mirrors the device.

**Cost.** $O(T)$ per unit (plus the compensating units in the learned variant) and a bias
evaluation at $\alpha = 1$. **Use.** The capacity move of $\alpha = 1$; gas capacity moves by
1–3 GW per sweep at 40 nodes.

**Device.** Metropolis form (`kpeak_mh_loop`), absorb variant only, per-unit scale
`kpeak_sigma` times the unit's initial capacity (at least 100 MW), peak sets recomputed on the
device from the net load after the renewable moves (`peak_sets_dev`), with the bias at
$\alpha = 1$.

## Level-0 capacity moves

The level-0 warmup chain, whose capacity fields are coarse, uses two other moves:

- **capacity slice updates** (`update_k!`): each level-0 capacity coefficient in random order by
  slice sampling on the full level-0 target, widths adapted during the warmup;
- **carrier scaling** (`scale_step`): one investable carrier at random, its capacity
  coefficients multiplied by $c$ and the coefficients of its utilization, commitment and
  storage fields divided by $c$, $\log c\sim\mathcal N(0,\sigma_f^2)$, Metropolis with the
  Jacobian $c^{m_k - \sum m_\omega T}$; $\sigma_f$ adapted toward acceptance 0.3 during the warmup.
