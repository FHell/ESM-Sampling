# The GPU port (ESMDevice)

This document describes the device version of the fine-level sampler in `device/`
(package `ESMDevice`, Reactant.jl/XLA): its scope, state layout, move representation and exact
line sampler, the round schedule, the capacity pass, the host loop, and what is done and what
remains. The port is under active development; the detailed state of the current phase is
kept by the device work in `docs/reactant_phase3.md`, and this document must be updated with
it. The CPU sampler it reproduces is in [sampler.md](sampler.md), the verification rules in
[testing.md](testing.md).

## 1. Scope

The device runs the finest, spatially complete level with many chains as a batch dimension,
in the formulation the system uses: `storage = "soc"`, `renewables = "load"`,
`regularized = true`, with every model-completion term (losses and soft limits as hinge
sums, lossy links, merged vintages, fleet unit commitment, the CO₂ price through the
marginal costs). The measure enters only the Jacobians of the capacity moves.

Everything else stays on the CPU host: setup, the weather model, the reference LP and the
heuristic start, the level-0 warmup that learns the bias, adaptation and bias refits,
checkpoints and I/O. The LP storage form, renewables with utilization fields, hard walls and
coarse levels are not ported.

## 2. State

Node values only, all FP64, laid out `[chain, step, item]` with the chain index fastest
(`DeviceState`): angles $(C,T,N)$, dispatch utilizations and commitment $(C,T,U_g)$, states
of charge $(C,T,U_s)$, link utilizations $(C,T,U_l)$, raw capacities $(C,U_g)$ and $(C,U_s)$.
The weather enters as the net load `dnet` (load minus renewable output $Ak$, from the host)
with the renewable running cost and capex. The device loop works on flattened
$(C, T\cdot X)$ matrices plus one residual cache $r$ (`FlatState`); there is no
mode-coefficient vector. `pack` builds the static model (`DeviceModel`) of a complete level,
`to_device`/`to_host!` convert between `State`s and the device state (exact at the complete
level, where every basis is orthonormal), and `cost_parts` evaluates the cost of every chain
($=$ `energy_cost` to round-off).

## 3. Moves and the exact line sampler

A move is a set of **sites**, one per chain and site index, each a line move
$t\mapsto s + t\,d$ whose cost change is a sum of piecewise-linear **terms** of a few linear
arguments (`TermBatch`: generator box terms with the vintage kink, commitment terms, link
terms, line terms with hinge knots, SoC terms, balance slots). A compensation site expands a
requirement change by one rule: none, unit, linear (learned coefficients) or share-preserving
(kinked arguments $x_0 + at + a_k\varphi_k(t)$ plus the Jacobian step).

`sample_dense` is exact without truncation:
* it sorts the primary points of every site, including the kinks and the clamp points on both
  pieces of every kinked argument;
* it keeps the K smallest sign changes of every secondary quantity per segment (`nroots`). K = 2
  for the plain families, where at most two are proven per term type; K = 4 for the compensated,
  cell and block families;
* it evaluates the potential at all points and inverts the segment masses. The masses are
  anchored at each segment's lower end, and offsets of decreasing segments at the right end.

**Guard.** A site whose line has more sign changes than K in some secondary quantity stays at
$t = 0$. The count is a property of the whole line, so this keeps the target invariant. The
tests report how often it happens.

An improper ray gives $t = 0$, as on the CPU. The share-preserving families apply the Jacobian
step (`jac_accept`, with the site's third uniform). The commit is a scatter-add into the state
and $r$.

| device family | CPU family | rule |
|---|---|---|
| `:gen` | finest dispatch atoms, `unit` (commitment follows) | none |
| `:uc` | commitment atoms | none |
| `:pair` | `pair` | balanced |
| `:soc` | finest SoC atoms | none |
| `:link` | link atoms | none |
| `:theta` | `potential` | none |
| `:spair` | `spair` | unit |
| `:tsoc`, `:tsocL` | `tsoc`, `tsocL` | share-preserving, linear |
| `:tpot`, `:tpotL` | `tpot`, `tpotL` | share-preserving, linear |
| `:cell{q}`, `:cellu{q}` | `cell` at partition level q (`u`: committable carriers) | additive |
| `:cellpair{q}`, `:cellpairu{q}` | `cellpair` at level q | balanced |
| `:sblk{L}`, `:sblkL{L}` ($L$ = 2, 4, 8, 16, 32) | storage plateau and shapes (`prep_storage_ms!`), one unit | share-preserving, linear |

**Choices made on the host.** Choices that do not depend on the state are drawn when the
schedule is built: the unit pair of `:spair`, the carriers of the cell moves, and the block
offset and profile.

**Cell families.** They are bucketed by level and padded to the level's largest cell. Level q
gets rounds ∝ (cells)^(cell_power−1), which reproduces the CPU's level probabilities.

**Block profiles.** The profile table holds the plateau and optional shapes with zero ends
(`set_profiles!`).

**Requirement tracking.** The share-preserving families refuse models with line losses or lossy
links, whose compensation needs requirement tracking on the CPU (an event walk). With those
terms the device uses the linear rule.

**Linear rule.** Its coefficients are host tables (`StaticParams.β1`, `β2`, equal split by
default), with the proxy class `ChainParams.cls` from `proxy_class`.

> **Status: deviation.** The term loops are unrolled at trace time, so the compiled program grows
> with a family's term count: `:sblkL4` takes 464 s and 11 GB to compile on the CPU backend, and
> `:tpotL` does not compile in 20 minutes at 12.7 GB. Large families therefore need term-stacked
> evaluation (a term axis in the arrays) or the slope-jump form before they can run compiled.

> **Status: deviation.** Cell–class aggregates of the block moves (one class in one cell) are not
> ported; the large coarse cells and long blocks are costly in the dense form (about 1 700
> points per site for a 40-node cell), so the slope-jump form is needed there.

## 4. Schedule

A **round** is a set of conditionally independent sites: two sites can share a round if the
terms they touch are disjoint. Time classes (odd and even steps, three when $T$ is odd)
separate the steps a SoC or commitment term couples; per node one unit slot (or pair) per
round; a distance-2 colouring of the node graph for angle moves; an edge colouring of the
links on their buses. Large rounds are split into chunks. `check_conflicts` verifies every
round on the host. A sweep runs every family's rounds in a fixed order; each round leaves the
target invariant, so the sweep does. All rounds of `ns` sweeps run in one compiled program
(`run_sweeps`: device loops `@trace for` over sweeps and rounds, the sweep count traced) with
Reactant's RNG; for comparisons the uniforms can be passed in as arrays.

> **Status: deviation.** Parallel lines are not supported by the angle family (none exist in
> the 40- and 256-node networks).

## 5. Capacity pass

After every `device_sweeps` operation sweeps, one compiled capacity pass runs, in order:
renewable scaling moves (`kren_mh_loop`), the peak sets recomputed from the net load
(`peak_sets_dev`), scaling moves of the dispatchable and storage units (`kscale_mh_loop`) and
peak-set moves of the dispatchable units (`kpeak_mh_loop`, absorb variant). All are Metropolis
with fixed-scale proposals (`kmove_sigma` for the log-scale, `kpeak_sigma` times the unit's
start capacity, at least 100 MW, for $\delta$), `kmove_nprop` proposals per unit, units in
sequence, chains in parallel; proposal normals and uniforms are drawn on the host. At
$\alpha = 1$ the bias change is computed incrementally from per-chain aggregates (`BiasDev`,
`BiasState`: the level-0 features $u_1,u_2$ of the chain's weather and of the exact samples,
the level-0 capacities and clamped coefficients). The capacity-dependent chain parameters are
then recomputed on the device.

## 6. Host loop

`run_device(cfg)` (script `device/scripts/run_device.jl`, configuration e.g.
`configs/device40.toml`; `run_device(cfg; merged=true, families=FAMILIES)`):

1. `setup` on the CPU, and at $\alpha = 1$ the level-0 warmup that learns the bias;
2. `nchains` chain starts as in the multi-chain mode, all with the weather of the set-up state;
3. `pack`, `to_device`, `plan` (schedule and index tables), upload, and one compile on the
   backend `device_backend`: `sweep_program`, the operation sweeps (traced count) followed by the
   capacity pass, as one program per checkpoint (`merged = false`: two programs, as the
   profiler uses);
4. `mlda_warmup + nsamples` checkpoints: sweeps and capacity pass, download, conversion to
   `State`s, online bias refits during the warmup (pooled over chains, frozen afterwards, the
   bias aggregates re-packed after a refit), monitored quantities, post-correction records;
5. `chains_report`: the same diagnostics and HDF5 output as the CPU multi-chain mode.

## 7. Status

| part | state |
|---|---|
| device model, exact cost, round trip | done |
| term batch, exact dense sampler, plain families, rounds, compiled multi-sweep program | done |
| compensated families `:spair`, `:tsoc(L)`, `:tpot(L)` | verified on arrays; `:spair`, `:tsoc(L)` also compiled (CPU backend); scheduled with `families = :all` |
| cell templates (all partition levels) | verified on arrays; scheduled with `families = :all` |
| storage blocks, one unit (plateau, shapes) | verified on arrays; `:sblkL4` compiled; not yet scheduled |
| stacked term evaluation (program size independent of the term count) | missing; needed to compile `:tpot(L)`, coarse cells, long blocks |
| one program per checkpoint (operation sweeps + capacity pass) | done (`sweep_program`) |
| capacity pass (`kren`, `kscale`, `kpeak` absorb) with the bias, host loop | done |
| templates for Haar angle atoms and localized proxy and learned modes (`tcut`, `acut`) | missing |
| local re-routing templates in place of `xpair`, `xloc(c)`, `tline` | missing |
| block aggregates (cell–class groups), learned shapes | missing |
| dense transport modes (`tmode`, `amode`) | missing; low rate, may be dropped |
| slope-jump sampler for large moves | missing |
| adaptation on the host with table uploads (modes, shapes, response coefficients); learned `kpeak` | missing |
| weather moves with compensation as a device family | missing |
| fused kernels, FP32 intermediates, tiling, multi-GPU, full-Europe memory plan | missing |

**Reactant constraints met so far.** Inside `@trace for`, every captured scalar becomes
traced, so structs captured by a loop are type-parametric in their scalars and static flags
are passed as `Val`; a traced loop index used for indexing needs `@allowscalar`; traced
ranges (`v[q:q]`) fail on Reactant 0.2.304; `abs(a*b)` of two distinct block arguments is
miscompiled to `a*b` (use `max(x,-x)`); FMA contraction makes exact zeros of padded sites
inexact. The residual cache is not rebuilt inside the device loop (its drift is about
$10^{-10}$ MW per 10–100 sweeps).

**Throughput** (development GPU, an RTX 3050 Ti laptop part with slow FP64; details and
estimates in [performance.md](performance.md) §7): about
$4.8\cdot10^5$ line moves per second at 40 nodes × 6 h with 4 chains against
$1.8\cdot10^5$ for the CPU plain kernel on 8 threads; $4.0\cdot10^5$ at 256 nodes daily with
one chain against $8.6\cdot10^4$. Four chains at 256 nodes exceed its 4 GB.
