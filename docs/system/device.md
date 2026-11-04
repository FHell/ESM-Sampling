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
in the formulation of the system: state-of-charge storage, renewables as load, all
constraints priced, with every model-completion term (losses and soft limits as hinge
sums, lossy links, merged vintages, fleet unit commitment, the CO₂ price through the
marginal costs). The measure enters only the Jacobians of the capacity moves.

Everything else stays on the CPU host: setup, the weather model, the reference LP and the
heuristic start, the level-0 warmup that learns the bias, adaptation and bias refits,
checkpoints and I/O. Coarse levels are not ported.

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

**Term axis.** Every field of a term type is one $(C, B, 1, n)$ array. The compensated, block,
cell and dense angle families build them directly (`device/src/stack.jl`): one gather of a
$(B, n)$ index table per field (`G4`, `S4`), and one scatter-add per state array in the commit
(`scadd4`). The plain families list their few terms and the `TermBatch` constructor stacks them
(`stack_terms`). Evaluation then works on $(C, B, K, n)$ for $K$ points and reduces over $n$, so
neither the trace nor the compiled program grows with the term count.

Static per-column metadata is recorded at trace time:
* the balance slot of every injection;
* the kink id of every kinked argument;
* whether an argument moves: a slope that is the literal `0.0` of the builder (or `nothing` in a
  stacked builder) is static;
* which unit columns can hold a unit with fleet commitment (`UC_COLS`, from the schedule): the
  commitment terms of the other columns are identically zero and are left out.

The balance slots sum their columns with a static padded column table (an appended zero column,
one gather and one sum; no 0/1 matmul, so a padded term cannot turn an infinity into NaN).

The line sampler has two forms, both exact without truncation, with the same points, the same
uniforms and the same draw up to round-off. `plan` chooses one per family (`line_draw`).

**Points.** Both forms use the same breakpoints of $\beta\Delta C(t)$:
* the primary points: the clamp points of moving arguments only, the kinks, and the clamp points
  on both pieces of every kinked argument;
* the sign changes of every secondary quantity, vectorized over the quantity axis (`rootsK`).
  K of them are kept per quantity:
  - 1 for a quantity of exactly one moving argument: it is monotone in $t$. This covers the
    SoC quantities with one moving end, and the commitment quantities with one moving
    argument;
  - none for a quantity of static arguments only;
  - the family's K otherwise (`nroots`): 2 for the plain families, where at most two are proven
    per term type, and 4 for the compensated, cell and block families.

A term's own secondary quantities are linear between its primary points. A balance-slot residual
is not: it also bends where a SoC injection kinks ($d = 0$ and $-d\,\eta_{out} = 1$), which are
roots of the SoC quantities. Its roots are therefore searched on the grid refined by those roots.
Before this two-stage search the dense form (and the CPU walk, `walk!`) interpolated the residual
across such a kink and missed it; the test of linearity between consecutive points
(`test/test_jump.jl`) catches this.

**Dense form** (`sample_dense`). Every term column is evaluated at every point, the segment
masses are inverted: anchored at each segment's lower end, offsets of decreasing segments at the
right end (`sample_core`). The cost per site is points × columns.

**Slope-jump form** (`sample_jump`, `device/src/jump.jl`). Every column is evaluated only at its
own points:
1. per column: 0, its primary points and the roots of its secondary quantities (searched on its
   own points), sorted;
2. per column: the cost change at these points (and one beyond each end), the slope of every
   local segment, the slope jump at every point and the slope just right of 0;
3. per balance slot: the points of every column injecting into it and 0, and the residual $r$
   there (one evaluation). $r$ is linear between them, so the balance cost
   $\Delta t\,(c_{shed} r_+ + c_{dump} r_-)$ bends only at those points and where $r$ crosses 0.
   At a column's point $p$ its jump is $\Delta t\,w(r(p))$ × that column's injection jump,
   $w = c_{shed}$ for $r \ge 0$ and $-c_{dump}$ below: it is added to the column's own entry. The
   crossings are found on the slot's points sorted with $r$ as payload (a change of the side
   $r \ge 0$ between the ends of a segment, or between a ray's end and infinity) and become
   entries with the jump $\Delta t\,(w_{after} - w_{before})\,r'$ (`slot_cross`). The same side
   convention in both makes the jumps add up to the cost's slopes also where $r$ vanishes at a
   point. Slot points are therefore no entries of their own;
4. the (point, jump) entries of a site: every entry at exactly 0 is folded into one, the rest
   sorted with the jumps as payload (one stable sort), and only the first `cap` kept
   (`JUMP_CAP`, set by `plan` from the sites of three rounds: 1.25 × the largest count + 8);
5. the segment slopes from the slope right of 0 outwards, the potential integrated outwards from
   $c(0) = 0$; every partial sum accumulates from 0 outwards, so no value near 0 carries the
   round-off of far, large terms, and a slope within the round-off of its sum is 0 (a flat stretch
   stays exactly flat). Then the dense form's inverse CDF (`sample_core1`).

Steps 4 and 5 work with the entry axis first, $(N, C, B)$ (`jump_points1`, `jump_values1`,
`sample_core1`): the entry axis is then the fastest in memory, so the payload sort and the scans
need no transposes. Reactant lowers `cumsum` to a reduce-window over the whole axis ($N^2$ work);
the scans of traced arrays run in two levels instead (`cumsum1`: blocks of `SCANBLK` = 32, then the
block totals, joined by a shift, so a prefix of exact zeros stays exactly zero). Point lists of at
most `SMALLSORT` = 8 entries per column are sorted by a compare-exchange network (`sort3`, min/max
pairs that fuse into their producers) instead of a sort op.

The cost per site is the sum of the columns' local points plus a sort of the entries.

**Form choice.** The plain families use the dense form. A compensated, cell or block family uses
the slope-jump form if its dense points × columns are at least `JUMP_RATIO` (4) × its
slope-jump entries (`plan(...; jump=:auto)`; `:none`, `:all` or a list of families override).
At 40 nodes daily: slope-jump for `:tsoc`, `:tpot`, all `:sblk…`, `:cell1`, `:cell2`,
`:cellpair1`, `:cellpair2`; dense for `:spair0`, `:cell3`, `:cellpair3`.

**Guards.** A site stays at $t = 0$:
* if some secondary quantity has more sign changes than kept;
* in the slope-jump form, if its count of real candidates reaches the cap. Padded candidates
  sit at exactly 0. The count includes every entry $\neq 0$ and every real candidate at 0
  (denominator $\neq 0$).

Both counts are properties of the whole line, so the guards keep the target invariant. The tests
report how often they act.

An improper ray gives $t = 0$, as on the CPU. The share-preserving families apply the Jacobian
step (`jac_accept`, with the site's third uniform). The commit is a scatter-add into the state
and $r$.

| device family | CPU family | rule |
|---|---|---|
| `:gen` | finest dispatch atoms, `unit` (commitment follows) | none |
| `:gen0` | the same for units without fleet commitment: no UC terms, all steps in one round | none |
| `:uc` | commitment atoms | none |
| `:pair` | `pair` | balanced |
| `:pair0` | `pair` of two units without fleet commitment (no UC terms, all steps in one round) | balanced |
| `:soc` | finest SoC atoms | none |
| `:link` | link atoms | none |
| `:theta` | `potential` | none |
| `:theta3`, `:theta5` | `potential` at nodes of degree ≤ 3 and 4–5, padded to their own degree (`:theta` then keeps degree ≥ 6) | none |
| `:spair` | `spair` (with `:spair0` in the list: pairs with a committable unit) | unit |
| `:spair0` | `spair` of two units without fleet commitment (no UC terms) | unit |
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

> **Status: deviation.** The cell pair at the coarsest level (`:cellpairu1`, two carriers over the
> whole 40-node network: 80 generator and 160 commitment terms) compiles in about 290 s at an
> 8 GB peak on the CPU backend, also in the slope-jump form, and is left out of the run set. It is
> verified on arrays (`test_jump.jl`) and compiled against arrays; the dense diagnostics of the CPU
> comparison (`test_comp.jl`) exceed 10 GB for it.

> **Status: deviation.** Cell–class aggregates of the block moves (one class in one cell) are not
> ported.

## 4. Schedule

A **round** is a set of conditionally independent sites: two sites can share a round if the
terms they touch are disjoint. Time classes (odd and even steps, three when $T$ is odd)
separate the steps a SoC or commitment term couples; per node one unit slot (or pair) per
round; a distance-2 colouring of the node graph for angle moves; an edge colouring of the
links on their buses. Large rounds are split into chunks (`bmax`, and the memory cap below).
`check_conflicts` verifies every round on the host. A sweep runs every family's rounds in a fixed order; each round leaves the
target invariant, so the sweep does. All rounds of `ns` sweeps run in one compiled program
(`run_sweeps`: device loops `@trace for` over sweeps and rounds, the sweep count traced) with
Reactant's RNG; for comparisons the uniforms can be passed in as arrays.

**Family sets.** `FAMILIES` (`:gen`, `:uc`, `:pair`, `:soc`, `:link`, `:theta`) is the
reference set the tests use. Device runs use `FAMILIES_FAST`, the same moves split by term
structure: `:gen0`/`:pair0` for units without fleet commitment (no UC terms, which are
identically zero for them, and no coupling of $\tau-1$ and $\tau$, so one round takes all
steps; `:gen`/`:pair` then keep the committable units and pairs), and the angle moves
bucketed by node degree (`THETA_BUCKETS`), each padded to its own largest degree. The cost of
a site is roughly points × terms, so padding to the global maximum degree costs about the
square of the ratio. `families = :all` (`device_families`) adds:
* `:spair0`, `:spair`;
* `:tsoc`, `:tpot` (`:tsocL`, `:tpotL` with requirement tracking);
* the storage blocks of `BLOCK_SCHED` ($L$ = 2, 4, 8, 16, 32; `:sblk…`, or `:sblkL…` with tracking);
* the cell families of the partition, except `:cellpairu1` (§3).

**Memory cap.** The largest tensor of a round has $C \cdot B \cdot e$ elements. Here $e$ is
points × columns in the dense form (`dense_size`). In the slope-jump form it is the larger of
the entries and the columns × local points of a term type (`jump_size`). Both are static per
family (`round_elements`).

`plan` splits the rounds of a compensated, cell or block family into even chunks, so that this
tensor stays within `DENSE_BUDGET`. The budget is 256 MB by default; it is the keyword `budget` of
`plan` and `device_setup`, and `BUDGET_MB` in `profile_families.jl`. A compiled dense round holds
about six such tensors (CPU backend: about 1.4 GB above the base at the default).

At 40 nodes daily with 4 chains, per site:

| family | dense points × columns | slope-jump entries (cap) | sites per round |
|---|---|---|---|
| `:tsoc` | 101 × 16 | 218 (106) | 7 280 |
| `:tpot` | 520 × 87 | 1 478 (360) | 1 274 |
| `:sblk2`, `:sblk4`, `:sblk8`, `:sblk16`, `:sblk32` | 227 × 27, 395 × 43, 731 × 75, 1 403 × 139, 2 747 × 267 | 473, 743, 1 283, 2 363, 4 523 (123, 143, 183, 263, 423) | 3 640, 1 800, 920, 440, 200 |
| `:cellpair1` | 403 × 80 | 842 (395) | 182 |

The slope-jump families are not split at this size. The plain families stay below 40 points ×
7 columns.

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
$\alpha = 1$ the bias, lifted to the chains' level ([ensemble.md](ensemble.md) §4.4), changes
incrementally from per-chain aggregates (`pack_bias` from the CPU bias cache; `BiasDev`,
`BiasState`): the features $u_1,u_2$ per finest step $(C,T_L)$ of the chain's weather and per
sample step of $\bar{\mathbb E}$, their sums of $F_1+F_2$, the raw node capacities of the
investable units whose capacity enters $\hat b$, and the clamped coefficients of $\hat h$. A
proposal of one unit changes its node capacity only (the level is spatially complete): $R_t$ by
$A_t\Delta$ at every step for a non-gas generator (the chain's availability and the sample
steps'), the firm capacity for gas and storage, and the clamped coefficients of its node.
$F_1$, $F_2$ come from the CPU's tables, evaluated by a gather of two table entries per point
(`tab_dev`, the arithmetic of the CPU's `tabval`). After an online refit the aggregates and
tables are re-packed and uploaded (same shapes, no recompile). The capacity-dependent chain
parameters are then recomputed on the device.

## 6. Host loop

`run_device(cfg)` (script `device/scripts/run_device.jl`, configuration e.g.
`configs/device40.toml`; `run_device(cfg; merged=false, families=…)`):

1. `setup` on the CPU, and at $\alpha = 1$ the level-0 warmup that learns the bias and its lift to
   the chains' level (`lift_bias!`, as the CPU driver);
2. `nchains` chain starts as in the CPU driver, all with the weather of the set-up state;
3. `pack`, `to_device`, `plan` (schedule and index tables), upload, and two compiles on the
   backend `device_backend`: the operation sweeps (traced count) and the capacity pass.
   `merged = true` compiles them as one program per checkpoint (`sweep_program`, bitwise equal).
   It takes about 5× longer to compile for one dispatch saved per checkpoint, so it is not the
   default;
4. `mlda_warmup + nsamples` checkpoints: sweeps and capacity pass, download, conversion to
   `State`s, online bias refits during the warmup (pooled over chains, frozen afterwards, the
   bias aggregates re-packed after a refit), monitored quantities, post-correction records;
5. `chains_report`: the same diagnostics and HDF5 output as the CPU driver (without the
   additional outputs of `write_run_outputs`).

The loop times every part per call (operation sweeps, capacity pass, host draws, download,
host work) and writes them to `device_profile/*` in the output file, with the compile times.
`device_setup(cfg)` builds everything up to the compile and is shared with the profilers:
`device/scripts/profile_device.jl` (Reactant's profiler on both programs: compile time split
into tracing, MLIR passes and XLA; run time; kernel and framework-op statistics; xplane traces
for xprof/Perfetto in `results/profiles/`) and `device/scripts/profile_families.jl` (every
operation family alone: rounds, sites, time per sweep and per move, its five most expensive
kernels with occupancy). The profiler needs programs compiled with `sync = true` and
`donated_args = :none`.

## 7. Status

| part | state |
|---|---|
| device model, exact cost, round trip | done |
| term batch, exact dense sampler, plain families, rounds, compiled multi-sweep program | done |
| compensated families `:spair`, `:spair0`, `:tsoc(L)`, `:tpot(L)` | verified on arrays and compiled (CPU backend); scheduled with `families = :all` |
| cell templates (all partition levels) | verified on arrays; compiled except `:cellpairu1` (see §3); scheduled with `families = :all` except `:cellpairu1` |
| storage blocks, one unit (plateau, shapes) | verified on arrays ($L$ = 2–32) and compiled; scheduled with `families = :all` |
| memory cap of a round (`DENSE_BUDGET`) | done |
| slope-jump form of the line sampler, per-family choice (`JUMP_RATIO`) | done; verified on arrays against the CPU and the dense form, Reactant (CPU backend) against arrays |
| term axis (program size independent of the term count) | done |
| one program per checkpoint (operation sweeps + capacity pass) | done (`sweep_program`, `merged = true`); not the default (compile time) |
| capacity pass (`kren`, `kscale`, `kpeak` absorb) with the bias, host loop | done |
| templates for Haar angle atoms and localized proxy and learned modes (`tcut`, `acut`) | missing |
| local re-routing templates in place of `xpair`, `xloc(c)`, `tline` | missing |
| block aggregates (cell–class groups), learned shapes | missing |
| dense transport modes (`tmode`, `amode`) | missing; low rate, may be dropped |
| adaptation on the host with table uploads (modes, shapes, response coefficients); learned `kpeak` | missing |
| weather moves with compensation as a device family | not planned: the weather moves to a prior-reversible pCN process on the host, the capacities to a Langevin step ([../worklog.md](../worklog.md)); `kop` and the device bias (`bias_dev.jl`, kept as it is until it is retired) are not extended |
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
