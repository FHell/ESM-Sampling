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
| [storage.md](storage.md) | `tsoc`, `tsocL`, storage plateau and shape moves, `slong`, `slevel`, `sxpair`, `sxlong` | transport in time: states of charge over one instant, a dyadic block, any cyclic interval, or the whole trajectory; and between two storage units across nodes with the flows re-routed |
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
| `xpair`, `xloc`, `xlocc` | all | `:xpair`, `:xloc`, `:xlocc` (re-routing templates, generator units) |
| `spair` | all (SoC form) | `:spair` |
| `cell`, `cellpair` | all | `:cell{q}`, `:cellpair{q}` (level-bucketed cell templates) |
| `tpot`, `tmode`, `tline`, `tcut`, `tmode0`, `tline0`, `tsoc` | flow, adaptive | `:tpot`, `:tsoc` (no requirement tracking); `tmode`, `tline` as `:tdir` and `tmode0`, `tline0` as `:tdir0` (dense angle directions drawn on the host); `tcut` as `:tcut`, `:tcutw` (Haar atoms) |
| `tpotL`, `tmodeL`, `tlineL`, `tcutL`, `tsocL` | adaptive | `:tpotL`, `:tsocL`; `tmodeL`, `tlineL` as `:tdirL`; coefficients learned on the host (`run_device(…; adapt=true)`); `tcutL` as `:tcutL`, `:tcutLw` |
| storage plateau and shape | all | `:sblk{L}`, `:sblkL{L}` (one unit), `:sblkg{L}`, `:sblkgL{L}` (cell–class groups) |
| `slong`, `slevel` | all, with `storage_long > 0` (SoC form, complete level) | `:slong`, `:slongL` (L ≥ 2) |
| `sxpair` | all, with `storage_xpair > 0` (SoC form, complete level) | missing |
| `sxlong` | all, in the long-range storage sweep with `storage_xlong > 0` | missing |
| `amode`, `amode0`, `amodeL` | adaptive | as `:tdir`, `:tdir0`, `:tdirL` (modes learned on the host) |
| `acut`, `acutL` | adaptive | atoms of the learned pieces in `:tcut…` |
| learned storage shapes | adaptive | profile columns of `:sblk{L}` per season class, learned on the host |
| learned dispatch shapes | adaptive | `:dshp{L}` |
| `kscale`, `kpeak` | all, at the complete level | Metropolis forms, renewables included (`kren`); the learned `kpeak` variant missing |
| `kop` | all, at the complete level, with `kop = true` | not planned: the capacity moves are replaced by a Langevin step on the sampled scores (its device form comes later; [../../worklog.md](../../worklog.md)) |
| fine-level weather moves | all, with `fine_weather > 0` | not planned: the weather moves to a prior-reversible pCN process on the host ([../../worklog.md](../../worklog.md)) |
| replica weather process (pCN, regime windows; not a line move) | replica sampler (`sampler = "replicas"`) | missing |
| joint $z_0$ blocks | all, with `fine_weather > 0` and `fine_weather_blocks > 0` | not planned (as above) |
| level-0 moves (weather slice and shear, FFBS, capacity slice, carrier scaling) | level-0 warmup chain | not ported (host side) |

The device driver schedules the compensated, block, cell and dense angle families with
`families = :all` (`device_families`); the default is the plain families. The dense angle families
have sites only with the host adaptation (`adapt = true`), which draws their directions
([../device.md](../device.md) §3, §6).

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
