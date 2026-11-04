# Move families

This directory describes every move family of the sampler, grouped by what the moves do. Each
family is an exact line move ([../sampler.md](../sampler.md) §3) along a direction whose
selection depends only on the capacities $k$, the weather $x$, the network and tables that are
frozen between adaptation updates, which is why it leaves the target invariant
([../sampler.md](../sampler.md) §5). Each document gives, per family, the direction, the
selection rule, the compensation and Jacobian, the cost of one move, where it helps, its
family name and whether the device port has it.

| document | families | role |
|---|---|---|
| [coordinates.md](coordinates.md) | `atom`, `unit`, `potential`; commitment atoms | single coordinates: the irreducible core of the move set |
| [pairs.md](pairs.md) | `pair`, `xpair`, `xloc`, `xlocc`, `spair` | injection-balanced shifts between two units, at one node, across nodes, or across two steps |
| [cells.md](cells.md) | `cell`, `cellpair` | coarse spatial directions embedded at the fine level (multigrid atoms) |
| [transport.md](transport.md) | `tpot`, `tmode`, `tline`, `tcut`, `tmode0`, `tline0`, and `tpotL`, `tmodeL`, `tlineL`, `tcutL` | transport in space with share-preserving, learned or no compensation |
| [storage.md](storage.md) | `tsoc`, `tsocL`, storage plateau and shape moves, `slong`, `slevel` | transport in time: states of charge over one instant, a dyadic block, any cyclic interval, or the whole trajectory |
| [learned.md](learned.md) | `amode`, `amode0`, `amodeL`, `acut`, `acutL`, learned storage and dispatch shapes | directions learned from the chain's own fluctuations |
| [capacity.md](capacity.md) | `kscale`, `kpeak`, `kop`; level-0 carrier scaling and capacity slice updates | capacities |
| [weather.md](weather.md) | regime FFBS, weather coordinate moves, joint $z_0$ blocks, weather shear; the prior-reversible weather process of the replica sampler (pCN, regime windows) | weather |

How to add a family, with its tests and its device port: [adding_a_move.md](adding_a_move.md).

## Where each family runs

| family | CPU kernel (`fine_kernel`) | device |
|---|---|---|
| atoms of every field | all | finest coordinates: `:gen`, `:uc`, `:soc`, `:link`; coarse atoms missing |
| `unit` | all | as `:gen` |
| `potential` | all | `:theta` |
| `pair` | all | `:pair` |
| `xpair`, `xloc`, `xlocc` | all | missing (planned as local re-routing templates) |
| `spair` | all (SoC form) | `:spair` |
| `cell`, `cellpair` | all | `:cell{q}`, `:cellpair{q}` (level-bucketed cell templates) |
| `tpot`, `tmode`, `tline`, `tcut`, `tmode0`, `tline0`, `tsoc` | flow, adaptive | `:tpot`, `:tsoc` (no requirement tracking); others missing |
| `tpotL`, `tmodeL`, `tlineL`, `tcutL`, `tsocL` | adaptive | `:tpotL`, `:tsocL` (static coefficients); others missing |
| storage plateau and shape | all | `:sblk{L}`, `:sblkL{L}` (one unit; aggregates missing) |
| `slong`, `slevel` | all, with `storage_long > 0` (SoC form, complete level) | missing |
| `amode`, `amode0`, `amodeL`, `acut`, `acutL`, learned shapes | adaptive | missing |
| `kscale`, `kpeak` | all, at the complete level | Metropolis forms, renewables included (`kren`); the learned `kpeak` variant missing |
| `kop` | all, at the complete level, with `kop = true` | missing |
| fine-level weather moves | all, with `fine_weather > 0` | missing |
| replica weather process (pCN, regime windows; not a line move) | replica sampler (`sampler = "replicas"`) | missing |
| joint $z_0$ blocks | all, with `fine_weather > 0` and `fine_weather_blocks > 0` | missing |
| level-0 moves (weather slice and shear, FFBS, capacity slice, carrier scaling) | level-0 warmup chain | not ported (host side) |

The device families `:spair`, `:tsoc(L)`, `:tpot(L)` and the cell templates are scheduled by the
device driver with `families = :all` (`device_families`); the default is the plain families. The
block families exist but are not scheduled yet ([../device.md](../device.md)). The device port is under active
development; its current state is recorded in `docs/reactant_phase3.md`.

## Common vocabulary

- **Injection scale** $\kappa$ of a unit: injection per unit utilization, $Ak$ for a
  generator, $-k$ for storage charging, $+k$ for discharging (`κunit`). Moves skip units with
  $|\kappa| < \kappa_{\min} = 1$ MW.
- **Requirement** $Q$ at a node: residual plus the clamped output of its dispatchable units,
  the demand those units face.
- **Compensation rules**: none, additive, unit, share-preserving pro-rata (with Jacobian),
  learned linear ([../sampler.md](../sampler.md) §5.2).
- **Rates** are the relative weights of the random moves in a step; every step also gets every
  basis atom once per sweep ([../sampler.md](../sampler.md) §4.2).
