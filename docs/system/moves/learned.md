# Learned moves

The adaptive kernel (`fine_kernel = "adaptive"`, `src/adaptive.jl`) learns move directions
from the chain's own fluctuations and adds them to the mixture: transport directions in
space, their localized pieces, and temporal shapes for storage and dispatch. This document
covers these families and the learned compensation coefficients they share with the
`…L` transport families. Directions are recomputed only on the geometric schedule of
diminishing adaptation ([../sampler.md](../sampler.md) §8); between updates they are fixed
tables, so every move is exact. See [README.md](README.md) for the other families.

## What is learned

`AdaptiveModes` observes the state after every adaptive sweep:

- the **angle fluctuation covariance** $C_\theta = \mathbb E[(\theta_\tau-\bar\theta_\tau)
  (\theta_\tau-\bar\theta_\tau)^\top]$ with a running mean per node and step (angles taken off
  their component means);
- per dyadic length $L\in\{2,4,8,16,32\}$ ($L<T/2$) and **season class** (the meteorological
  season DJF, MAM, JJA, SON of the block's start step, `season` in `weather.jl`, the same
  seasons as the weather model and the sampled prices), the second moments of intra-block profiles of the
  dispatch (block mean removed) and of the state of charge (linear interpolation of the end
  values removed);
- per node and proxy class (deficit, surplus from the frames), the running moments of
  (requirement $Q$, unit outputs $p_i$), for the **response coefficients**
  $\beta_{n,c,i} = \mathrm{Cov}(p_i,Q)/\mathrm{Var}(Q)$ (initially an equal split).

At observation counts $2^j\ge4$ (and never after `freeze`, by default never) it recomputes the
$K=8$ leading eigenvectors of $C_\theta$ (orthogonal to the component constants) and their
localized Haar pieces, the two leading eigenvectors of each profile second moment (state of
charge shapes with zero ends), and the coefficients $\beta$ (where at least 10 observations
and a nonzero requirement variance exist). The update counts are written to the output.

## The families

**`amode`, `amode0`, `amodeL`.** A learned transport direction chosen uniformly, projected per
step onto the face of the predicted binding lines (frames from $(x,k)$). `amode` compensates
share-preservingly (Jacobian step), `amode0` does not compensate, `amodeL` uses the learned
coefficients. Dense, $O(N + n_{\rm lines})$. Rates 0.03, 0.03, 0.02.

**`acut`, `acutL`.** Localized learned transport: one dominant Haar atom of a learned
direction, as `tcut` ([transport.md](transport.md)), with share-preserving or learned
compensation. Rates 0.1, 0.08.

**Learned storage shapes.** In `adaptive_block_sweep!`, a storage move over a block of a
learned length with a learned state-of-charge shape of the block's season class, for one unit
or a cell–class aggregate, share-preserving compensation at every step (learned compensation
in half of the moves once the coefficients exist). It is the shape move of
[storage.md](storage.md) with learned instead of proxy shapes.

**Learned dispatch shapes.** In the same sweep, two generator units at one node (nodes with at
least two complete dispatchable units), with changes $+\phi_j t/\kappa_1$ and $-\phi_j
t/\kappa_2$ at step $j$ of the block, $\phi$ a learned dispatch shape: a balance-neutral
re-dispatch with an intra-day or multi-day profile. No compensation needed.

Blocks of the learned shapes start day-aligned half of the time and at a random offset
otherwise (a state-independent choice); blocks of one parity run in parallel, 8 moves per
block, storage or dispatch shapes with probability 1/2 each.

## Use

The learned compensation is the largest single gain of the adaptive kernel: from far starts at
6 h resolution it halves the remaining excess at equal time compared with share-preserving
compensation alone, because it picks up merit order. The learned directions matter where the
proxy modes miss the actual fluctuations.

**Device.** None of these families is ported. Adaptation stays on the host in the design:
statistics pooled over chains, updates at $2^j$, uploaded as tables ([../device.md](../device.md)).
