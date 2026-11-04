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

Two capacity samplers exist (`device_sampler`): the replica–Langevin sampler
([../replica_sampler.md](../replica_sampler.md); §5.2, §6.2), which replaces the learned bias,
and the Metropolis capacity pass with the bias (§5.1, §6.1), which is retired with the bias.

Everything else stays on the CPU host: setup, the weather model and the replicas' weather
process, the reference LP and the heuristic start, the level-0 warmup that learns the bias,
adaptation and bias refits, snapshots and I/O. Coarse levels are not ported.

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

**Term axis.** The builders list a site's terms; the `TermBatch` constructor stacks every field
of a term type into one $(C, B, 1, n)$ array (`stack_terms`). Evaluation then works on
$(C, B, K, n)$ for $K$ points and reduces over $n$, so the compiled program does not grow with
the term count.

Static per-column metadata is recorded at trace time:
* the balance slot of every injection;
* the kink id of every kinked argument;
* whether an argument moves: a slope that is the literal `0.0` of the builder is static.

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
3. per balance slot: the points of every column injecting into it and 0, the residual there, its
   roots, then the balance cost and its jumps in the same way;
4. the (point, jump) entries of a site: every entry at exactly 0 is folded into one, the rest
   sorted with the jumps as payload (one stable sort), and only the first `cap` kept
   (`JUMP_CAP`, set by `plan` from the sites of three rounds: 1.25 × the largest count + 8);
5. the segment slopes from the slope right of 0 outwards, the potential integrated outwards from
   $c(0) = 0$; every partial sum accumulates from 0 outwards, so no value near 0 carries the
   round-off of far, large terms, and a slope within the round-off of its sum is 0 (a flat stretch
   stays exactly flat). Then the dense form's inverse CDF (`sample_core`).

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

## 5. Capacity moves

### 5.1 Metropolis capacity pass (`device_sampler = "mh"`)

After every `device_sweeps` operation sweeps, one compiled capacity pass runs, in order:
renewable scaling moves (`kren_mh_loop`), the peak sets recomputed from the net load
(`peak_sets_dev`), scaling moves of the dispatchable and storage units (`kscale_mh_loop`) and
peak-set moves of the dispatchable units (`kpeak_mh_loop`, absorb variant). All are Metropolis
with fixed-scale proposals (`kmove_sigma` for the log-scale, `kpeak_sigma` times the unit's
start capacity, at least 100 MW, for $\delta$), `kmove_nprop` proposals per unit, units in
sequence, chains in parallel; proposal normals and uniforms are drawn on the host. At
$\alpha = 1$ the bias, lifted to the chains' level ([../archive/learned_bias.md](../archive/learned_bias.md) §4.4), changes
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

### 5.2 Replica–Langevin step (`device_sampler = "replicas"`)

The sampler of [sampler.md](sampler.md) §11 (CPU: `src/replicas.jl`): M weather replicas per chain
share the capacities; the capacities take a Langevin step on the replica-averaged sampled score.
Code: `device/src/score.jl`, `device/src/langevin_dev.jl`, the renewable units in `device/src/ren.jl`.

**Replica batch.** The batch dimension is $C$ = chains × M, the replicas of a chain contiguous
($c = j + (\text{chain}-1)M$). Every replica has its own net load, dispatch, states of charge and
residual; the capacities `kg`, `ks`, `RS.kr` are stored per replica and equal within a chain.
Reductions over a chain's replicas are reshapes $(C,U)\to(M,C/M,U)$ and sums (`rep_sum`); a
chain's new capacities go back to its replicas by a broadcast (`rep_bcast`).

**Score** (`cap_score`). Per replica and investable unit, at fixed utilizations,
$s_k = -\beta_v\,\partial_k C(k,v) + \partial_k\log J(k)$ in node coordinates, and the capex force
$\beta_k\,\partial_k K$ (with the box penalty $\gamma_k$). The terms are those of the CPU's capacity
gradient (`energy_cost(…; grad=true)`: `gen_grad` with the vintage kink, `uc_grad`, `soc_grad`,
renewables at $y = 1$; the balance through its subgradient
$\Delta t\,(c_{\rm shed}[r>0] - c_{\rm dump}[r<0])$ at the unit's node, from the residual cache)
plus `log_measure_grad`; a unit outside its box gets a zero operation score. Elementwise maps, sums
over the steps and one-hot contractions only. `score_acc` adds the score of every replica after
each of the `inner` operation sweeps of a step (weight 1/`inner`). In the Langevin coordinates
the score is $(k+\kappa_s)\,s_k$ (`wscore`).

**Step** (`lgv_w`, `lgv_pass`). Per chain and unit, $w = \log(k+\kappa_s)$ ($\kappa_s$ per carrier,
$\eta_w$ per chain and carrier, `LgvParams`):
$$
w' = w + \mathrm{clip}\big(\eta_w((k+\kappa_s)g + 1)\big) + \sqrt{2\eta_w}\,\xi,\qquad
g = -\beta_k\partial K + r\,\bar s_k,\quad r = \beta_k/\beta_v,
$$
$\bar s_k = \sum_j \omega_j s_k^{(j)}$ with the replica weights $\omega$ (`W`, uploaded per step: the
CPU's `drift_weights` over the replicas active by their gate, with the tilt). Reflection at
$k = \kappa_{\min}$ and, with `langevin_box`, at the upper bound of the box (`langevin_update`);
$\eta_w = 0$ freezes a unit (fixed capacities always). The noise $\xi$ is drawn
by Reactant's RNG in runs (`lgv_pass_rng`) and passed in for the tests.

**Dispatch following** (`lgv_commit`), in every replica, in the order of `follow_capacities!`:
1. dispatchable units at fixed output: utilizations and commitment scaled by $c(k_0)/c(k_1)$ (the
   commit map of `kscale`); the residual takes the output lost where the scaled utilization
   leaves $[0,1]$;
2. renewables (one carrier at a time; its units sit at distinct nodes, so a carrier is one
   vectorized update) and then storage, at fixed utilization (output $A k$; the state-of-charge
   fractions), the node's dispatchable generators following on the kop curve
   ([moves/capacity.md](moves/capacity.md)): with $P$, $Q$ the compensators' output and
   requirement and $e = P - Q_+$, the new output $P_1 = (Q-\Delta\mathrm{inj})_+ + e$ is shared as
   $y_i = \sigma_i P_1/\kappa_i$, $\sigma_i = \kappa_i y_i/P$, for the compensators with
   $|Ak|\ge\kappa_{\min}$ (`kop_follow`); a unit with $k_0 < \kappa_{\min}$ moves alone. The renewable
   output change moves the net load, the renewable running cost and its capex.

The chain parameters are then recomputed with the proxy class of the new net load.

**Weather upload** (`weather_arrays`, `weather_update`). After the host's weather step of some
replicas, their load less the fixed renewables' output $D_0$ $(C,T,N)$, its running cost and the
clamped availability $A$ $(C,T,U_r)$ are uploaded with a 0/1 mask of the changed replicas. The device
sets the net load $D_0 - H(A\,c(k_r))$ and the renewable running cost from the current renewable
capacities, shifts the residual by the change of the net load and (`comp`) re-initialises the
dispatch compensated (`compensate_weather!`: at every node and step the dispatchable
generators take up the change, increases in proportion to headroom, decreases in proportion to
output).

**Verification** (`device/test/test_langevin_dev.jl`, 40 nodes × 6 h, all model terms, 2 chains ×
2 replicas): on arrays against the CPU, the scores to $7\cdot10^{-13}$ (relative) and the forces to
$10^{-15}$; the step with the same normals against `langevin_capacities`/`follow_capacities!`
(capacities to $2\cdot10^{-14}$, utilizations $2\cdot10^{-14}$, residual $4\cdot10^{-11}$ MW, net load,
running cost and capex to round-off); the weather upload against `compensate_weather!` (utilizations
$2\cdot10^{-15}$, residual $2\cdot10^{-11}$ MW, proxy classes equal). Compiled by Reactant on the CPU
backend, the same comparisons hold, and compiled and array outputs agree to round-off.

A capacity exactly on its box bound (the clamped chain starts) can be inside the box for the
device's node values and outside for the CPU's coefficients (one round trip apart), so the
scores of such a unit differ in the first step; the Langevin step moves the capacity off the
bound.

## 6. Host loop

### 6.1 Metropolis driver (`device_sampler = "mh"`)

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
### 6.2 Replica driver (`device_sampler = "replicas"`)

`run_device(cfg)` with `device_sampler = "replicas"` calls `run_replicas_device`
(`device/src/replica_run.jl`, configuration e.g. `configs/device40_replica.toml`): the driver of
`run_replicas` (src/replicas.jl, [sampler.md](sampler.md) §11) with the replica batch on the device,
and its settings (`replicas`, `replica_*`, `langevin_*`, `tilt_*`; [config.md](config.md)):

1. `replica_device_setup`: `setup` (the bias is not used); per chain its start and `init_replicas`
   without its sweeps (fresh weather draws from the replicas' own streams, the compensated
   initialisation, `ReplicaSet`), or the replicas of the snapshot; `pack`, `plan`, the device arrays
   and four programs (operation sweeps, score accumulation, Langevin step with the dispatch
   following, weather upload), compiled, or with `compile = false` the same functions on Julia
   arrays; then `replica_init_sweeps` operation sweeps on the device;
2. per step (`replica_device_step!`):
   - `replica_device_weather!`: per replica the CPU's `replica_weather!` without its compensation
     (the schedule, the window and FFBS, the gate reset at a burst; `weather_process_step`), the
     upload of the changed replicas and the compensation on the device;
   - `replica_device_operation!`: `replica_inner` × (operation sweep, scores) on the device;
   - `replica_device_capacity!`: the scores downloaded into the CPU's per-field layout
     (`unit_map`), the gates (`update_gate!`), the probe windows and `drift_weights` on the host,
     the Langevin step on the device; every `replica_probe_every` steps per chain `lag_probe` on the
     host CPU (the candidate replica downloaded, the chain's probe stream) and in the warmup
     `adapt_step!`, the new steps uploaded;
3. a host checkpoint every `device_sweeps` steps: download, `State`s, `replica_monitor` and
   `replica_record`, the snapshot boundary (§6.3);
4. `chains_report` over the chains, `write_replica_outputs`, `write_replica_summary`, the timings.

The random streams are those of `run_replicas` for the weather (bitwise the same weather paths
from the same configuration) and the probes; the operation sweeps and the Langevin noise use two
device streams.

> **Status: deviation.** The monitored quantities and the replica records are taken at the host
> checkpoints only (every `device_sweeps` steps), not every step as on the CPU. The lag probe runs
> on the host CPU (`replica_probe_len` sweeps of one replica per chain), synchronously.

**Verification.** `device/test/test_replica_driver.jl` (arrays, against the CPU): the weather paths
and gate schedules of a run equal those of `run_replicas` bitwise; per step after the operation
sweeps, the downloaded scores equal `capacity_scores` ($6\cdot10^{-13}$), the gates and weights those
of `update_gate!` and `drift_weights`, and the Langevin step with the same normals plus the
dispatch following those of `langevin_capacities` and `follow_capacities!` (capacities $7\cdot10^{-13}$,
utilizations $2\cdot10^{-10}$, residual $2\cdot10^{-10}$ MW); a probe step adapts and uploads the
per-chain steps; stop (exit 75) and resume in the CPU layout.
`device/test/test_replica_compiled.jl` compares the four programs compiled by Reactant with the
arrays (and runs the compiled driver) when `ESM_DEVICE_COMPILE=1`.

### 6.3 Snapshots and resume

Both drivers follow the CPU driver's contract ([hpc.md](hpc.md) §7): at a host checkpoint, every
`checkpoint_minutes` and when `stop_file` exists, a snapshot is written atomically. The Metropolis
driver writes the chains layout ([data.md](data.md) §5) plus a group `device/`
(`device/src/snapshot_dev.jl`); the replica driver writes the layout of the CPU replica driver
(`write_replica_snapshot`, `boundary!`; also once after the initialisation), so each replica
driver resumes from the other's snapshots. With the stop file the driver
removes it and returns exit code 75 (`device/scripts/run_device.jl` exits with it and ignores
SIGTERM with `ESM_IGNORE_SIGTERM=1`). If
the snapshot exists at start, the driver resumes from its states: the device arrays, plan and
programs are rebuilt from them (the residual cache recomputed), the counters continue, Reactant's
RNG and the host streams (proposal draws, every replica's weather) are re-seeded from `seed` and
the restart count; the output records `run/restarts` and `run/resumed_at_checkpoint`
(Metropolis) or `run/resumed_at_sweep` (replicas, the step). A
resumed Metropolis run with the bias lifts the set-up bias (as the CPU driver). The grace time
of a preemption must cover one host checkpoint interval (`device_sweeps` steps) plus the write.
The device driver builds no caches of its own: `setup` uses `cache_dir` and `cache_mirror` as on
the CPU.

### 6.4 Setup and profilers

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
| nominal storage requirement of the share-preserving `:tsoc`, `:sblk{L}` (CPU `soc_qadj`, [moves/storage.md](moves/storage.md)) | deviation: the device uses the actual requirement; its comparisons against the CPU for these share-preserving families no longer hold until it adds the correction |
| long-range storage moves `slong`, `slevel` | missing |
| storage pairs `sxpair`, `sxlong` (re-routing templates, leak and loss compensation, footprint rounds) | missing |
| memory cap of a round (`DENSE_BUDGET`) | done |
| slope-jump form of the line sampler, per-family choice (`JUMP_RATIO`) | done; verified on arrays against the CPU and the dense form, Reactant (CPU backend) against arrays |
| term axis (program size independent of the term count) | done |
| one program per checkpoint (operation sweeps + capacity pass) | done (`sweep_program`, `merged = true`); not the default (compile time) |
| capacity pass (`kren`, `kscale`, `kpeak` absorb) with the bias, host loop | done (retired with the bias) |
| replica–Langevin capacity step: score, step in $w$, dispatch following (fixed output, kop curve), replica batch, weather upload with compensation | done; verified on arrays against the CPU and the prototype, compiled (CPU backend) against arrays |
| replica driver on the CPU sampler (`replicas`, `replica_*`, `langevin_*`, `tilt_*`; weather process, gate, probe and adaptation, weights, snapshot layout) | done; verified on arrays against `run_replicas`; compiled check written (`test_replica_compiled.jl`), not run |
| snapshots and resume of both drivers, exit code 75 | done; smoke-tested through `hpc/run.sh` (device mode, CPU backend) |
| templates for Haar angle atoms and localized proxy and learned modes (`tcut`, `acut`) | missing |
| local re-routing templates in place of `xpair`, `xloc(c)`, `tline` | missing |
| block aggregates (cell–class groups), learned shapes | missing |
| dense transport modes (`tmode`, `amode`) | missing; low rate, may be dropped |
| adaptation on the host with table uploads (modes, shapes, response coefficients); learned `kpeak` | missing |
| weather moves with compensation as a device family | missing |
| replica–Langevin sampler ([sampler.md](sampler.md) §11): the weather process per replica (pCN, regime windows), the compensated initialisation, the capacity scores, the Langevin update, the dispatch-following maps (fixed output, kop curve), the gate and the probe; replicas as a batch dimension | missing; CPU reference `src/replicas.jl` with explicit-noise building blocks (`pcn_weather`, `regime_window`, `langevin_update`) |
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
