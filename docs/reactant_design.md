# Reactant-first design of the fine-level sampler

A design for a GPU (XLA, Reactant.jl) version of the model and the fine-level
sampler, with many chains. It is written from the code in `src/` (energy.jl,
local.jl, flowmoves.jl, flowbasis.jl, adaptive.jl, bias*.jl, hierarchy.jl,
mlda.jl, run.jl) and the notes [setup_overview.md](archive/setup_overview.md),
[local_kernel.md](archive/local_kernel.md), [flow_hierarchy.md](archive/flow_hierarchy.md),
[sampling_formulations.md](archive/sampling_formulations.md) and
[reactant_performance.md](archive/reactant_performance.md). Nothing here changes the
target. Every simplification below either reparametrizes with a constant
Jacobian or changes only the move set, and every move set listed keeps the
coordinate moves, so the containment argument of setup_overview.md §6 still
holds.

**Summary**
* **Scope.** Only the spatially complete fine level goes on the device, in the
  formulation the runs use: SoC storage, renewables as load, regularized
  constraints. Coarse levels, MLDA, initialization and the LP stay on the CPU, as
  they are now.
* **State.** Node values only: θ, ω, e and u, plus one residual cache r. There
  is no mode-coefficient vector, no flow cache and no cell-sum cache. All of it is
  FP64 and laid out `[chain, step, item]`, with the chain index fastest.
* **Moves.** One generic representation. A move is a sparse primary direction
  (θ, unit, SoC and link entries) plus a list of *compensation sites*
  (node, step, g), each expanded by one rule: none, unit, linear (learned) or T2.
  Families differ only in how they fill this record. A family-agnostic exact
  padded line sampler draws t: a dense form for small moves and a slope-jump form
  for large ones.
* **Schedule.** Rounds of conditionally independent sites: all steps, all chains,
  and every node of a colour class at once. There are about 100 rounds per sweep
  at 256 nodes, against about 4 200 sequential rounds in the CPU schedule. The
  number of rounds depends on the colourings, not on N. A sweep is one compiled
  program with loops on the device and counter-based (Philox) random numbers.
* **Host.** Adaptation updates at sweeps 2^j, bias refits, template and colouring
  construction, MLDA and the coarse levels, and I/O. Per sweep only a few KB of
  observables cross the bus.

## 1. Scope and assumptions

* **Device kernel.** The operational block at fixed (k, x), plus the capacity
  moves, at the complete level (M = N, finest h). The flags are fixed:
  `storage = soc`, `renewables = load`, `regularized = true`. `measure` enters only
  the scalar Jacobian of the capacity moves.
* **Not ported:**
  - the LP storage form;
  - ω fields for renewables;
  - hard walls;
  - incomplete (coarse) fields with cell balances and mode boxes;
  - Gibbs/HMC level-0 kernels.

  These stay CPU-only.
* **Sizes used below.**
  - 256-node PyPSA-Eur, daily: N = 256, 480 lines, 4 links, 390 dispatchable
    units (≤ 3 per node: gas, nuclear, ror) and 632 storage units (≤ 4 per node).
    The maximum degree is 8, and a greedy distance-2 colouring needs 11 colours.
  - 40 nodes: maximum degree 7, 9 colours.
  - Full Europe, assumed: N ≈ 4 000, T = 8 760, about 6 000 dispatchable and
    10 000 storage units.

## 2. State representation (question 1)

### 2.1 Coordinates: node values, not mode coefficients

At the complete level every v-field basis is orthonormal: sparse graph-Haar,
with θ orthogonal to the component constants. Lebesgue measure on the
coefficients is therefore Lebesgue measure on the node values (θ taken modulo
the component constants). The device state is the node values. Coefficients are
computed only for output and for the CPU interface.

* **θ.** Stored as node angles with a free gauge. A line along e_n in R^N
  projects onto the line along P⊥e_n with the same parameter t, so the kernel on
  the quotient is the CPU's `potential` move exactly. Gauge drift is harmless in
  FP64; the host re-centres the angles at checkpoints.
* **Dropped caches.**
  - `F` is recomputed from θ: f = b(θ_a − θ_b), two gathers.
  - `S` equals r at the complete level, where the cells are the nodes.
  - The v vector and every `Ut` row walk in `commit!` (`addz_node!`, `zidx/zval`)
    go too.

  This halves the memory and removes the sparse-row indirections that dominate
  small moves on the CPU.

### 2.2 Arrays

Julia column-major `[c, τ, i]`: chain fastest, then step, then item.
C = chains, T = steps, and a dimension of 1 means a per-chain vector.

| array | shape | dtype | content |
|---|---|---|---|
| `θ` | (C, T, N) | F64 | node angles |
| `ω` | (C, T, U_g) | F64 | dispatchable utilizations, units ordered by node |
| `e` | (C, T, U_s) | F64 | SoC utilization at instant τ (cyclic) |
| `u` | (C, T, U_l) | F64 | link utilization ∈ [−1, 1] |
| `r` | (C, T, N) | F64 | residual cache (demand − supply); rebuilt from scratch every sweep |
| `dnet` | (C, T, N) | F64 | load − Σ A·k_ren at the chain's (x, k_ren); refreshed when x or k_ren change |
| `kg`, `ks` | (C, U_g), (C, U_s) | F64 | raw capacities; clamped values and κ = avail·k derived |
| `cls` | (C, T, N) | UInt8 | proxy class (deficit or surplus), from x only (§7) |
| `ctr` | (C,) and global | UInt32 | Philox counters |

**Static model, shared by all chains (Int32 or F64):**
* **Lines:** `line_a`, `line_b`, `line_bsus`, `line_s`.
* **Units:** `gen_node`, `gen_mc`, `gen_avail`; `st_node`, `st_h`, `st_ηin`,
  `st_ηout`, `st_qt`, `st_mc`; `link_a`, `link_b`, `link_cap`.
* **Per-node tables, padded with 0 = none:** `node_gens` (N, G), `node_st`
  (N, S), `node_lines` (N, D) with a sign, and `node_nbrs` (N, D). At 256 nodes
  G = 3, S = 4 and D = 8.
* **Scalars:** Δt, c_shed, c_dump, c_viol, κ0, β_v.

Dispatchable units have constant availability once renewables are load (ror,
nuclear and gas have `wf = 0`). So κ = avail·k is per chain, not per step, and
no availability array enters the hot loop.

**Why the chain index is fastest.** A move's index structure is shared by all
chains. Only values differ: the selections are drawn with chain-free keys (§4.4).
Every gather and scatter of a template index therefore touches C contiguous
words, and that is coalesced for C ≥ 32. With few chains, as at full Europe, the
step index is next. Rounds always cover all steps (or one parity), so
consecutive threads walk consecutive τ. A parity-split time order (odd steps,
then even steps) makes parity rounds contiguous too. That is a tuning option,
not part of the interface.

**Memory per chain (FP64):**
* 256 × daily: 1 800 items × 365 × 8 B ≈ 5 MB, so 256 chains take ≈ 1.3 GB.
* Full Europe × hourly: ≈ 28 000 items × 8 760 × 8 B ≈ 2 GB. That is about half
  of reactant_performance.md's 5–8 GB, because v, F and S are gone. Add the round
  workspace, which must be tiled (§4.5), and an 80 GB H100 holds ≈ 20–30 chains.

### 2.3 Precision

| quantity | precision | reason |
|---|---|---|
| state θ, ω, e, u; cache r; dnet; k | FP64 | `y += t·a` accumulates over 10⁶+ updates. The r kink must be placed to ≪ w(c_shed) = 0.35 MW (daily) against 10⁴ MW sums. T2 invariance of (σ, e) is tested to 10⁻¹² |
| direction coefficients from tables (learned β, shapes, templates, modes, Haar atoms) | FP32 storage | any fixed direction is valid. The rounded value is the direction, provided evaluation and commit use the same value |
| breakpoint locations (b − y₀)/a | FP64 subtraction, then FP32 | the one cancellation-prone step |
| potential at points, sorting, prefix sums, log-masses (slope-jump form) | FP32 allowed | built as slope × length from t = 0, so there is no c(t) − c(0) cancellation; ≈ 10⁻³ nats needed |
| dense-form term differences | FP64, or slope form | c(t) − c(0) of terms around 10³ nats |
| final in-segment offset, committed t | FP64 | it enters the state |
| uniforms | FP64 (53 bits from 2 Philox words) | inverse-CDF tails |
| accumulators (adaptation moments, prices, bias scores) | FP64 | long running means |

FP64 is the default: an H100 runs FP64 at half the FP32 rate and is
bandwidth-bound here anyway. FP32 intermediates are an option for the jump form
and for development on consumer GPUs, where FP64 runs at 1/64 of FP32.

## 3. Moves (question 2)

### 3.1 One generic move record

Every fine-level family of the CPU code has the same shape:

  (sparse primary direction) + (compensation at nodes whose requirement changes).

At fixed (k, x) the primary part is a static index pattern, a template, with
coefficients from (k, x) or the frozen tables. The compensation is a per-node rule.

```
MoveBatch (one round, B sites per chain, one size bucket)
  base step     τb::(B,)                          shared by all chains
  θ entries     (n, dτ)::(B,Eθ), dθ::(C|1,B,Eθ)
  unit entries  (kind∈{gen,link,soc}, idx, dτ)::(B,Eu), a::(C|1,B,Eu)
  comp sites    (node, dτ)::(B,Q), g::(C|1,B,Q), rule::(B,)   rule ∈ {NONE, UNIT(j), LINEAR, T2}
```

**Expansion of a compensation site** (n, τ, g), where g is the requirement change
per unit t. For θ directions, g = Σ_lines b·Δθ is static. For SoC entries
g = ±κ_s with κ_s = k·h/Δt:
* **NONE:** the residual absorbs the change (`tmode0`, `potential`, Haar θ atoms).
* **UNIT(j):** unit slot j at n takes −g/κ_j (the current `spair`).
* **LINEAR:** unit i takes β_{n,cls(n,τ),i}·g/κ_i, with β read from the learned
  table (G, N, 2), and the residual takes the rest (`compensate_learned!`). The
  direction is constant along the line, so there is no Jacobian.
* **T2:** one kink per site, with Q = r + Σκ clamp(ω), P = Σκω, e = P − Q₊ and
  kinked unit coefficients ak_i = ω_i/P (`prorata!`). The Jacobian term is
  (P, e, Q, g, m − 1).

The record has fixed sizes per family, so all sites of a round share one bucket.

### 3.2 Term batch: what the line sampler sees

`gather_terms` turns a MoveBatch into the terms of ΔC(t). There are four types,
each piecewise linear in t:

| term | argument(s) along t | cost | feeds balance slot |
|---|---|---|---|
| box (gen, link) | y = y₀ + a t (+ ak φ_k(t), T2) | `gen_cost`: Δt·mc·κ·clamp + c_viol·Δt·(k+κ0)·dist; link: c_viol·Δt·cap·dist | ±κ·Δclamp(y) |
| line | f = f₀ + df·t, df = b(dθ_a − dθ_b) | c_viol·Δt·(\|f\| − s)₊ | ±(f − f₀) at both ends |
| SoC (unit, step) | e_τ, e_{τ+1}, each linear | `soc_cost`: clamps, charge or discharge, power limit, spill rule, box penalty of e_τ | k(pc − mc) |
| balance (node, step) | r = r₀ + Σ contributions | Δt(c_shed r₊ + c_dump r₋) | — |

A θ entry appears only through its lines. A SoC entry appears only through its
two SoC terms. Lines with df = 0 are dropped when the template is built (the
CPU drops them at run time).

**Padded sizes per site.** E box entries, L lines, S SoC terms, R balance slots,
K kinks. The slot indices are local and shared across chains; the values have
shape (C, B, ·).

### 3.3 Family map: CPU → GPU

| CPU families | GPU family (rule) | E / L / S / R / K at 256 nodes | decision |
|---|---|---|---|
| atoms of ω, e, u fields; `unit` | **coord** (all coordinates, every sweep) for the finest atoms; the coarse Haar atoms of the dispatch fields stay as **cell templates** (see below) | gen 1/0/0/1/0; SoC 0/0/2/2/0; link 1/0/0/2/0 | merged at the finest scale. *Correction (main session):* the coarse dispatch atoms are not dropped; they are cell-constant changes, the same as `cell` moves |
| `pair` | **pair** (two slots at one node, ±1/κ) | 2/0/0/1/0 | kept |
| `potential`, `tpot`, `tpotL` | **theta(n)** with NONE / LINEAR / T2 | NONE 0/8/0/9/0; LINEAR 27/8/0/9/0; T2 27/8/0/9/9 | merged |
| `spair`, `tsoc`, `tsocL` | **soc(n)** with UNIT / LINEAR / T2 | UNIT 2/0/2/2/0; LINEAR 6/0/2/2/0; T2 6/0/2/2/2 | merged |
| θ Haar atoms, `tcut`, `tcutL`, `acut`, `acutL` | **template** (static sparse θ pattern plus boundary sites; NONE / LINEAR / T2) | bucketed by template size | merged; built on the host from the Haar basis and `localize` of the proxy and learned modes |
| `tline`, `tline0`; `xpair` | **template** with a *local* reroute: L⁺(e_a − e_b) solved on a k-hop ball (zero angle change outside), the ball's boundary nodes as compensation sites | ≈ 20–60 entries | replaced. Any fixed direction is exact; the dense O(N) reroute goes |
| `splateau`, `sshape`, `ashape`, `adshape` | **block(L)** with a profile over L + 1 instants or L steps, for a SoC unit, a cell-class aggregate or a generator pair; compensation sites at the end steps (plateau) or at every step (shape) | plateau 2G/0/L+1/L+1/(2 if T2); shape GL/0/L+1/L+1/(L if T2) | merged into one family with a profile table |
| `tmode`, `tmode0`, `amode`, `amode0` | **dense** (direction table D (N, K, T), face-projected) | ≈ N·G / n_l / 0 / N / N | kept at a low rate in the largest bucket; candidate to drop after the Phase 0 ablation |
| `cell`, `cellpair` at the complete level, and the coarse Haar atoms of the dispatch fields | **template** (cell-constant change of one carrier, or two carriers with the cell injection balanced; optional LINEAR/T2 compensation at the cell's nodes), with levels chosen with fixed probabilities ∝ number of cells (todo item 6) | ≈ cell size | *Correction (main session):* **kept**. They are the multigrid part of the move set, sparse within a cell, and linear in the cell size. Phase 0 sets their rates; it does not decide whether they exist |
| time-constant block families (`batom`, `bunit`, `bpair`, `bsoc`, `bcell`, `bcellpair`) | — | — | dropped (shelved; no gain at equal time) |
| `kscale`, `kpeak` | **kmove** (group action plus Jacobian, MH with fixed proposals, §3.6) | O(T) reduction per unit | kept, sequential over units |

The remaining set (coord, pair, theta, soc, template, block, dense, kmove)
contains every coordinate direction. So it is irreducible for the same reasons as
the CPU set, and the learned families stay optional extras on top.

### 3.4 Exact padded line sampler

There are two implementations behind one interface:

```
sample_line(tb::TermBatch, u::(C,B,2)) -> (t::(C,B), dr::(C,B,R))
```

1. **Dense form** (E + L + 2S + R ≤ ≈ 32, which covers ≈ 90 % of the moves). This
   is the prototype in `external/reactant-bench/linesampler.jl`, extended to all
   four term types:
   * primary points: clamp points, line limits, the SoC clamp points of both
     arguments, the kinks t* = −Q/g and the 4 side clamps per kinked entry;
   * a sort per site;
   * evaluate the balance arguments r_j and the SoC d_s at all points;
   * a root inside each segment for r_j = 0 and for d_s ∈ {0, η_in, −1/η_out};
   * the merged sort;
   * the potential at all points, `logseg` masses, the two rays, log-sum-exp,
     inverse CDF and the truncated-exponential offset.

   There is no 40-nat truncation, so it is exact, where the CPU walk truncates at
   e⁻⁴⁰. An improper ray (slope ≤ 0) returns t = 0, as on the CPU.
2. **Slope-jump form** (large buckets: templates, block(L ≥ 8), dense). Cost is
   O(K log K) and there is no walk:
   * *Local kinks.* Each box entry, line and SoC term emits its own (t,
     Δslope, Δr_slot) triples, at most 5, 2 and 4 + 3 × 5 respectively. The SoC
     thresholds come from a fixed-size local root search over its ≤ 5 linear
     pieces.
   * *Balance roots.* Per slot, the contributor kinks (≤ 2(G + D + 2S) ≈ 40) get
     a local sort and a prefix sum of dr/dt. That gives r_j at its kinks, the
     roots, and slope jumps Δt(c_shed + c_dump)|dr_j/dt|.
   * *One global sort per site* of all triples plus t = 0. The slope on each
     segment is the left-ray slope plus the cumulative jumps. The potential is the
     cumulative sum of slope × length, re-anchored to U(0) = 0. Then masses and
     inversion as in the dense form.
   * *The committed values.* t, and dr_j(t) for the commit, evaluated locally.
3. **Jacobian step** (T2 only):
   `jac_accept(tb, t, u)`: log J = Σ_k (m_k − 1) log|((Q_k + g_k t)₊ + e_k)/P_k|, and
   t ← 0 if u ≥ J. This is the independence Metropolis step of `local_sweep!`.
4. **Commit:** `commit(state, mb, t, dr)`, conflict-free by colouring. It applies
   scatter-adds to θ (dθ·t), ω and u (a·t + ak·φ_k(t)), e (de·t) and r (dr).

### 3.5 Compensation details that change on the GPU

* **κ_min selection rules.** Currently pairs are skipped when |κ| < κ_min. They
  become masks: a masked site has t = 0. These depend on (k, x) only, as on the
  CPU.
* **The proxy class `cls`** of the learned rule comes from x (and the proxy
  k̂), not from the current k (§7). The CPU uses `FF.surplus` from `frame_injections`
  at the current k.
* **T2 per site:** one kink and ≤ G kinked entries. This matches the CPU's
  "one kink per entry" assertion when sites are distinct nodes. Two sites at the
  same node and step never occur inside one move: templates merge the g of
  duplicate boundary nodes when they are built.

### 3.6 Capacity moves (kmove)

* **What changes per unit.** For one unit at node n, the move changes every step
  at n and the bias. The cost change is an O(T) reduction over the gen or SoC
  terms and the balance terms of n, which is contiguous in τ. The bias depends on
  k only through level-0 aggregates: R_t(k) is linear in k with precomputed
  per-unit weights w_{i,t}, plus the firm capacity. ĥ is separable in the
  level-0 k coordinates, and one unit changes only O(depth) of them. So Δb̂ for
  one unit is O(T₀ · (1 + n_samp)) with cached u₁, u₂.
* **Sampling.** On the GPU this is Metropolis with fixed-scale proposals on u
  (kscale) or δ (kpeak), m proposals per unit per sweep, with the generalized-Gibbs
  Jacobian in the target. That replaces `slice_scalar`, whose stepping-out and
  shrink loops are data-dependent. The proposal scale is adapted per carrier on
  the host with diminishing adaptation.
* **Order.** Units run sequentially, because the bias couples them, and each
  move is batched over chains. At 256 nodes that is about 1 000 units ×
  (a few kernels of a few µs) ≈ 10–30 ms per sweep. Without the bias (α = 0),
  units at distinct nodes are independent and run in parallel.
* **Peak sets** (top q of the proxy requirement at n). They come from (x, k_ren):
  one sort per (chain, node) along T, recomputed after a k_ren move.

## 4. Schedule (question 3)

### 4.1 Independence rules

Two sites can share a round if their footprints (the terms they touch) are
disjoint. Sites at different steps conflict only through SoC terms, which couple
τ − 1 and τ.

| round kind | sites per chain | footprint | colouring |
|---|---|---|---|
| coord-gen(j) | nodes with ≥ j generators × all τ | gen term, balance (n, τ) | none (slot j) |
| coord-soc(j, p) | nodes with ≥ j storage units × τ ∈ parity p | SoC terms τ−1, τ; balance (n, τ−1), (n, τ) | time parity; 3 classes when T is odd (`local_sweep!`) |
| coord-link(c) | links of edge colour c × all τ | link term, balance at both buses | edge colouring on the buses |
| pair | nodes with ≥ 2 generators × all τ | 2 gen terms, balance (n, τ) | none |
| theta(c, rule) | nodes of colour c × all τ | lines at n; balance and generators on N[n] | distance-2 colouring (G²): 9 colours at 40 nodes, 11 at 256 |
| soc(p, rule) | one storage unit per node × τ ∈ p | as coord-soc plus generators at n | time parity |
| template(c) | templates of colour c × all τ | support lines, boundary nodes | greedy colouring of the template conflict graph (host) |
| block(L, o, p, target) | blocks of parity p × nodes (unit) or cells (aggregate) | steps b₀…b₀+L at n or in the cell | block parity × cell |
| dense(k) | all τ | the whole network at τ | steps independent (θ only) |
| kmove | one unit, all chains | all τ at n, bias | sequential |

Drawing all sites of a round jointly is exact. They are conditionally
independent given the rest, so the joint update is the product of their exact
line kernels.

### 4.2 Rounds per sweep (256 × daily, estimate)

| kind | rounds | sites per round per chain |
|---|---|---|
| coord-gen | 3 | ≈ 130 × 365 |
| coord-soc | 4 × 3 = 12 | ≈ 160 × 183 |
| coord-link | 2 | 2 × 365 |
| pair | 2 | ≈ 100 × 365 |
| theta: NONE, then LINEAR or T2 | 2 × 11 | ≈ 23 × 365 |
| soc (UNIT and LINEAR/T2) | 2 × 3 | ≈ 160 × 183 |
| template | ≈ 20 | ≈ 5 × 365 |
| block (7 lengths × 2 parities × unit/aggregate) | ≈ 28 | 256 × T/2L |
| dense | 2–4 | 365 |
| **total** | **≈ 100** + the kmove loop | |

A sweep updates every coordinate once plus about the CPU's 4N random moves per
step. That is ≈ 0.6–1·10⁶ moves per chain-sweep, as on the CPU (770 k).

**Width.**
* *One chain at 256 × daily:* rounds of 10⁴–5·10⁴ sites, which is
  under-occupied, so run 64–256 chains.
* *Full Europe × hourly with one chain:* theta rounds have ≈ 2·10⁶ sites. A
  single chain then saturates the GPU, and rounds must be tiled.
* *Latency.* Rounds per sweep grow only with the colour counts (≈ 15–20 θ
  colours expected at full Europe), not with N. At ≈ 10–20 µs of launches per
  round inside a device loop, the floor is ≈ 2–5 ms per sweep.

**Throughput estimate.** This uses the per-move costs of reactant_performance.md
§2: naive XLA on an H100 runs at ≈ 20 ns for E ≤ 4 and ≈ 150 ns for E = 16.
* *The size mix.* After the heavy moves are localized, most moves are
  coord/pair/soc (E ≤ 6). Theta-LINEAR and templates run at E ≈ 20–60.
* *256 × daily:* ≈ 30–80 ms per chain-sweep naive. That is 10× better than that
  note's estimate for the *current* move mix, because the 10 % dense moves that
  took 80 % of the time are gone. With fused kernels it is ≈ 3–10 ms.
* *Equal work per sweep, not equal mixing.* Phase 0 decides whether the reduced
  set mixes as well.

### 4.3 One compiled program per sweep

```
sweep(ds, dm, dp, tab, sched, ctr) -> (ds, stats)
  for kind in (coord, pair, theta, soc, template, block, dense)   # static, unrolled: one body per kind
      @trace for r in 1:sched.nrounds[kind]                       # device loop; count is a traced scalar
          sites = dynamic_slice(sched.sites[kind], r)             # class padded to its max size, with mask
          mb = build_kind(ds, dm, dp, tab, sites, key(ctr, kind, r))
          tb = gather_terms(mb, ds, dm, dp)
          t, dr = sample_line(tb, uniforms(ctr, kind, r))
          kind_has_T2 && (t = jac_accept(tb, t, uniforms(...)))
          ds = commit(ds, mb, t, dr)                              # functional update: fresh arrays
      end
  end
  ds = rebuild_r(ds, dm, dp)                                     # cache drift reset, O(C·T·(N + n_l + U))
  @trace for i in 1:n_kunits; ds = kmove(ds, ..., i); end
```

* **Program size** is O(number of kinds), not O(rounds). Colour classes are rows
  of padded index tables, so new templates, colourings or learned tables never
  change shapes: they are padded to fixed maxima chosen at setup.
* **Loop-carried state** is returned as new arrays each iteration. The
  prototype's `@trace for` failed with "aliasing pattern of loop-carried
  variables changed" when it mutated in place.
* **Order and reversibility.** Kinds run in a fixed order. With `palindrome =
  true` the sweep runs kinds 1…n and then n…1, which makes it reversible: each
  round is reversible, because exact line draws and Jacobian MH are. That is
  needed only where MLDA wants a reversible base kernel. Otherwise the fixed order
  is π-invariant.
* **Wide rounds** (full Europe) are tiled inside the round with an inner
  `@trace for` over site chunks of about 2²⁰ sites. Chunks of one colour class are
  independent, so this does not change the kernel.

### 4.4 Random numbers

**Generator.** Philox4x32-10, written as an elementwise traced function. 32-bit
`mulhi` comes from a UInt64 product. The same code runs on CPU arrays, so the
CPU array version and the GPU version draw identical numbers.

**Keys and counters.**
* *Selections* (which pair, unit slot, template, shape, block offset) use
  key = (seed, 0) and counter = (sweep, kind, round, site). They are shared by
  all chains. They are independent of v, so each chain's kernel is the CPU kernel.
  In stationarity, shared selections induce no dependence between chains: given
  the selection, each kernel preserves π. Out of stationarity they correlate the
  chains slightly; for R̂ during burn-in, use chain-specific selection keys.
* *Line-draw and Jacobian uniforms* use key = (seed, chain) and the same counter
  with lanes 1–3.
* *Reproducibility.* Results are reproducible and independent of tiling and
  batch size. This allows the bitwise-identical CPU/GPU tests of §8.

## 5. Host and device (question 4)

| piece | where | when | transfer |
|---|---|---|---|
| model packing, unit and node tables, colourings (G², links), templates (Haar θ atoms, localized modes, local reroutes), schedule | host | setup; templates again at adaptation updates | up: index tables (≤ MB), fixed maximum shapes |
| state, caches, per-chain parameters | device | always | checkpoints only |
| operational sweep (all families) | device | every sweep | — |
| capacity moves including Δb̂ | device | every sweep | — |
| residual rebuild, cost parts, sampled-price accumulators | device | every sweep | down: cost parts and capacity totals, ≈ 100 B per chain |
| frames: congestion ADMM, binding mask, `cls`, face-projected dense modes D | device (ADMM is already array-first) | when x changes (§7) | up: the chain's x node arrays |
| adaptation accumulators: θ second moment (N × N; pooled over chains), intra-block shape moments, response moments (Q, p) per node and class | device | every sweep (`observe!`, `observe_response!`) | down at sweeps 2^j only |
| adaptation updates: eigenvectors, `localize`, β = cov/var, shapes; proposal scales of kmove | host | sweeps 2^j (diminishing; optional freeze) | up: tables of fixed shape |
| bias scores s_z, s_k | device (Enzyme through Reactant, or the explicit gradients ported from `energy_cost`) | every thinned sample | down: n₀·T₀ + n_{k0} per chain |
| bias regression and monotone QP refit | host | observation counts 2^j | up: θ̂, â, ĉ (tiny) |
| MLDA, coarse levels, level-0 weather chain, init (LP, heuristic, prolongation) | host (current CPU code) | per MLDA step or at init | up or down: one chain's state when MLDA swaps it |
| ESS, R̂, HDF5 output | host | every n sweeps and at the end | down: observables |

* **Asynchronous updates.** Host updates at 2^j work on a snapshot. The new
  tables take effect at the next sweep boundary while the device keeps sweeping
  (PJRT dispatch is asynchronous). The kernels then depend on the history with a
  delay, which diminishing adaptation allows.
* **Pooled adaptation.** All chains share the learned tables and pool their
  statistics. Treat the C chains as one chain on the product space: every kernel
  is π^{⊗C}-invariant for fixed tables, so Roberts & Rosenthal applies unchanged.
  Pooling learns C times faster, and shared tables keep the gathers coalesced.
* **MLDA.** Fine-level MLDA acceptance is ≈ 0 (setup_overview.md §3), so the GPU
  product is the many-chain fine sampler. MLDA is shelved; if it is revisited, it
  would use an overcomplete state (coarse and fine coordinates kept together,
  todo), stay on the host and call the device sweep for its `p_fine` steps. Moving x at the fine level is open
  (todo 18) and is not solved by the port. The `template` and `block` machinery
  is where a weather-with-compensation family would plug in. Weather enters dnet
  and the Gaussian prior, which is not piecewise linear, so that family needs MH
  or slice sampling, not the exact sampler.
* **Reactant.ProbProg** is not used for the exact kernel. Optionally,
  `mcmc_logpdf` NUTS can serve the smooth level-0 weather block on the host side
  later.

## 6. Kernel list

C = chains, B = sites per round per chain, and E, L, S, R, K are the bucket
sizes. All functions are pure (they return new arrays) and run unchanged on
`Array` (CPU reference) and `ConcreteRArray` (traced).

| kernel | signature | shapes and notes |
|---|---|---|
| `pack` | `pack(P, ℓ) -> DevModel, DevTables, Schedule` | host. Index tables Int32 and coefficients F32/F64. Colourings and templates padded to maxima |
| `to_device`, `to_host` | `State ↔ DevState` | coefficients ↔ node values via the orthonormal bases; θ gauge from the component means |
| `rebuild_r` | `(ds, dm, dp) -> r` | (C, T, N): dnet + scatter-add of line flows (from θ), links, `soc_cost` injections, −κ·clamp(ω) |
| `cost_parts` | `(ds, dm, dp) -> (C, 6)` | gen, storage, shed, dump, penalty, op: the term sum of `cache_cost`. Also the reference for the tests |
| `build_coord`, `build_pair`, `build_theta`, `build_soc`, `build_template`, `build_block`, `build_dense` | `(ds, dm, dp, tab, sites, key) -> MoveBatch` | selection from Philox; coefficients from κ, β and the profiles; compensation-site expansion |
| `gather_terms` | `(mb, ds, dm, dp) -> TermBatch` | gathers y₀, f₀ = b(θ_a − θ_b), e₀, e₁, r₀, Q, P |
| `sample_dense`, `sample_jump` | `(tb, u) -> (t::(C,B), dr::(C,B,R))` | §3.4 |
| `jac_accept` | `(tb, t, u) -> t` | T2 rounds only |
| `commit` | `(ds, mb, t, dr) -> ds` | scatter-add, conflict-free |
| `kmove` | `(ds, dm, dp, bias, unit, key) -> ds` | O(T) reductions plus O(T₀ n_samp) bias aggregates |
| `frames` | `(xnodes, dm) -> (cls, mask, D)` | batched ADMM over C·T columns plus masked SPD projection (`project_modes`) |
| `observe` | `(acc, ds, dm, dp) -> acc` | θ moments (GEMM over T), shape moments, response moments, prices |
| `sweep` | `(ds, dm, dp, tab, sched, ctr) -> (ds, stats)` | §4.3, one executable per (C, T, bucket set) |
| `check_conflicts` (debug) | `(mb) -> max writes per target` | scatter-add of ones per round, which must be ≤ 1. Run on every schedule in the tests |

## 7. Model-side changes (question 5)

These simplify the GPU version without changing the target, or they are choices
that are free for exactness:

1. **Node-value state at the complete level** (§2.1). It drops v, F and S and all
   `Ut` walks. There is no Jacobian change, because the bases are orthonormal.
2. **One code path:** SoC storage, renewables as load, regularized constraints.
   * *Renewables as load* removes ≈ 30 % of the coordinates, every A·k product in
     the hot loop and the curtailment-split flat directions. The output A·k sits
     in dnet, precomputed per chain when x or k_ren change.
   * *SoC storage* gives 1 coordinate per unit and step instead of 3, and no
     dynamics-penalty terms. Its cost is the two-step coupling: parity classes,
     with 3 when T is odd, and two SoC terms per SoC entry. The balance is clearly
     positive.

   Both formulation choices help the GPU version, not only the mixing.
3. **Fixed-size per-node slot tables:** G = 3, S = 4, D = 8 at 256 nodes. Merge
   co-located vintages and duplicate units when the data are built (one unit per
   carrier per node is already asserted), and keep G small. If full-Europe
   degrees reach 15 or more, use degree classes for the theta buckets.
4. **Frames, proxy classes and peak sets from x only** (with the proxy k̂ and
   the current k_ren). They are refreshed when x changes, not after every capacity
   move. Selection is still independent of v, so this is exact; only efficiency is
   at stake, and Phase 0 checks it.
5. **Compensation as node rules** (NONE, UNIT, LINEAR, T2). Make LINEAR, the
   learned rule, the default: it has no kink and no Jacobian, and it was the
   fastest variant (flow_hierarchy.md §15). T2 stays as an option; the Phase 0
   ablation decides whether it earns its kink tables.
6. **Localize everything dense.**
   * local-ball reroutes replace L⁺ in `xpair` and `tline`;
   * Haar-atom templates replace the full modes;
   * the dense modes run at a low rate, or not at all.

   This is the same recommendation as reactant_performance.md §5, and it pays on
   the CPU as well (work/perf).
7. **Keep the priced balance (B0) and the current coordinates; do not adopt the
   full T2 state layout for v1.** A T2 state removes r but moves (m − 1) log P
   into every θ line. Those line conditionals are no longer piecewise
   exponential, and the derived-capacity max couplings are global in T. The
   compensation-site record gives the same moves inside the exact piecewise
   sampler.
8. **Stay piecewise linear for v1.** Quadratic losses, soft thermal limits and
   elastic demand would need truncated-Gaussian pieces (erf) or an MH correction.
   Signed lossy links (one interior kink at u = 0) fit the box-term type as they
   are.
9. **No per-year global constraints.** A CO₂ cap would couple all steps; a
   canonical CO₂ price (sampling_formulations.md rec. 6) keeps the steps
   independent.
10. **Capacity moves by MH with fixed proposals** instead of slice sampling. The
    bias written through level-0 aggregates keeps Δb̂ O(T₀) per unit.

## 8. Phases, acceptance tests and effort (question 6)

### Work rules (user, for every phase)

1. **Correctness first on plain Julia arrays.** The device code is array code, so it runs
   without Reactant. It is compared with the CPU reference: the same mock random numbers on
   both sides (a fixed sequence; a mock `AbstractRNG` on the CPU, input arrays on the device),
   the same prepared moves, a few steps. Compare exactly where the arithmetic is the same and
   to round-off otherwise. No compile is needed for this loop.
2. **Tracing second, with Reactant on the CPU backend** (`Reactant.set_default_backend("cpu")`).
   This skips the CUDA/ptxas stage and avoids the 4 GB memory and the slow FP64 of the laptop
   GPU. The compiled result must equal the Julia-array result for the same inputs.
3. **The GPU only for timings**, at the end of a work item.
   *Measured:* the backend hardly changes the compile time (the peak-set program: 73 s on the
   CPU backend, 80 s on the GPU); tracing and Reactant's own passes dominate, and every fresh
   Julia process adds 1.5–2 min of start-up and package loading. So iterate in one persistent
   Julia session (the Julia MCP server, or a REPL) and compile only once the array version
   passes.
4. **No statistical tests** (KS, z-tests, many-chain MCMC runs) for the port. Tests take seconds
   to minutes. The statistical acceptance items listed below are replaced by these comparisons.

**Tracing rules found so far:**
* Inside `@trace for`, every captured `Int`, `Float64` or `Bool` becomes traced:
  - structs captured by a loop must be type-parametric in their scalar types too (`KParams`);
  - static flags go in as `Val`;
  - loop counts come from array shapes.
* A traced loop index used as an array index needs `Reactant.@allowscalar idx[j]`.
* `abs(a*b)` is miscompiled to `a*b` when the operands are distinct block arguments (function
  arguments, loop-carried values): Enzyme-JAX's NonNegativeResultAnalysis treats such a product as
  a square and `AbsPositiveSimplify` drops the `abs` (upstream issue EnzymeAD/Enzyme-JAX#2648, fix
  PR #2649 approved but not merged as of 2026-11; it then needs a Reactant_jll release). Use
  `max(x, -x)`. Related open soundness issues of the same analyses: Enzyme-JAX #2651, #2656, #2673.
* Dynamic indexing with a traced integer works (`v[q]` under `@allowscalar`, `X[:, :, q]`,
  `H[q, :]`). A traced range (`v[q:q]`) fails in `get_slice_stride` on Reactant 0.2.304, the
  newest version in the local registry; use integer indices and reshape.
* See also "Phase 0 spike: results".


**References.**
* *The CPU code* (`energy_cost`, `cache_cost`, `eval_move!`, `local_sweep!`,
  `adaptive_sweep!`, `kscale_sweep!`, `kpeak_sweep!`) and `test/test_exact.jl`
  (3 nodes × 73 steps).
* *Many chains make the statistical tests stronger.* With 512 GPU chains
  against a long CPU chain the z-tests are tight, and per-chain means support
  two-sample KS tests.

| phase | content | acceptance | effort |
|---|---|---|---|
| **0. Decisions on the CPU, plus a Reactant spike** | ablations on the current code (short, niced): (a) the rates of the cell templates and the level distribution (kept in every case); (b) `xpair`/`tline` → local reroute templates or `tpotL`; (c) a node-coordinate sweep for the finest dispatch atoms (the coarse atoms stay as cell templates); (d) frames, `cls` and peak sets from (x, k̂); (e) LINEAR-only vs T2 compensation; (f) no dense `tmode`/`amode`. Spike: `@trace for` with loop-carried scatter-add on (C, T, N), plus `stablehlo.sort` and Philox in a loop | excess/pred per sweep (40 nodes × 6 h far start; 256 × daily) within ≈ 20 % of the current adaptive kernel; the spike compiles and runs 100 rounds | 1–2 weeks |
| **1. Device model and exact cost** | `pack`, `to_device`/`to_host`, `rebuild_r`, `cost_parts`, Philox, gradients (Enzyme) for the bias scores | `cost_parts` = `energy_cost` parts ≤ 10⁻¹² relative on random states (3, 40, 256 nodes); r = `LocalCost.r`; gradients = `energy_cost(grad=true)` ≤ 10⁻¹⁰; round trip State → device → State exact; Philox matches the Random123 test vectors | 2–3 weeks |
| **2. Line sampler and plain rounds** | MoveBatch/TermBatch, `sample_dense`, `sample_jump`, `commit`; the coord, pair and theta-NONE families; colourings; the compiled sweep with device loops; runs on CPU arrays and on GPU | (i) ΔC(t) at random t = CPU `eval_move!` for the same direction, ≤ 10⁻⁹, per family; (ii) KS against quadrature for both forms, including stiff (c_shed) and SoC two-step cases (as `check_cpu.jl`); (iii) after 100 sweeps, the r cache against `rebuild_r` ≤ 10⁻⁸ MW and `cost_parts` against the CPU on the downloaded state; (iv) `check_conflicts` = 1 on every round; (v) test_exact setup: 512 GPU chains against the CPU `plain` kernel, \|z\| < 4 on op, shed, mean SoC and line flow; (vi) CPU-array and GPU runs identical to round-off for the same seeds; (vii) throughput at 256 × daily with 64 chains reported | 3–5 weeks |
| **3. Compensated and multiscale moves** | the rules UNIT and LINEAR, then T2 (kinks plus `jac_accept`); theta and soc compensated; templates (Haar θ atoms, localized proxy and learned modes, local reroutes, **cell templates for the dispatch fields at all coarser levels**); block profiles; dense (if kept). Per-family diagnostics (todo item 29) are shelved; the device kernels should still record cheap per-family counters (time share, Jacobian acceptance), so they can be added later | T2: (σ, e) invariant across the kink ≤ 10⁻¹², and the Metropolis line chain reproduces π·J (KS); test_exact operation test with the full GPU family set against the CPU reference, \|z\| < 4; 40 nodes × 6 h far start: excess/pred against sweeps no worse than the CPU adaptive kernel within noise | 4–6 weeks |
| **4. Capacities, adaptation, bias, host loop** | `kmove` (kscale and kpeak), level-0 bias aggregates, `observe`, host updates at 2^j with table upload, online bias scores and refits, outputs as in `run.jl` | test_exact capacity test (kscale against kscale + kpeak) on the GPU, \|z\| < 4; Δb̂ = CPU `bias_term` differences ≤ 10⁻¹⁰; fed the same states, the host updates reproduce the CPU's β, shapes and modes; α = 1 end to end at 40 nodes; HDF5 output readable by `plot_results.py` | 2–4 weeks |
| **5. Performance and scale** | fused kernels for `sample_dense` + `commit` (CUDA.jl or KernelAbstractions traced by Reactant), FP32 intermediates in the jump form, tiling, parity-split time layout, chain sharding over GPUs, full-Europe memory plan; then weather-with-compensation moves at the fine level as a new family | all statistical tests of phases 2–4 unchanged; targets at 256 × daily with 64 chains: ≥ 50× over 4 CPU threads naive, ≥ 300× fused; full Europe × hourly: one sweep in ≤ 20 s on one H100 | 3–6 weeks |

* **Shelved phase 4b, delayed acceptance with surrogates** (todo item 30, shelved
  for now), only for families the diagnostics show to be expensive: a surrogate line draw (e.g. a
  linearized bias change for kmove, coarse-time costs for long storage blocks)
  followed by a Metropolis correction with the exact change, as extra batched
  kernels in the same program. ≈ 1–2 weeks.
* **No MLDA phase.** MLDA is shelved (todo); the hierarchy acts through the
  overcomplete fine-level moves. **Adaptation and learning stay on the host**:
  modes, shapes, β and bias refits at 2^j, with statistics pooled over chains and
  uploaded as tables.
* **Total:** ≈ 15–26 weeks for one developer who knows the code.
* **First milestone:** the plain kernel with many chains on 256 × daily, at the
  end of phase 2, after ≈ 6–10 weeks. It is already useful for todo items 2 and 9
  (multi-chain R̂ from several starts; sweeps to equilibrium against network size).

## 9. Open risks

* **Reactant control flow.**
  - *Loop-carried scatter state:* it failed once in the prototype (aliasing).
  - *Dynamic slicing of padded class tables* inside `@trace for`.
  - *Compile time* with about 8 loop bodies and big buckets: minutes is
    acceptable, recompiles are not.

  Phase 0 spikes this before anything else.
* **Naive XLA is ≈ 10× off the hardware.** It materializes the per-move
  intermediates. The headline numbers need fused kernels traced into the program,
  and that embedding is to be verified for the Reactant version in use.
* **Mixing per sweep of the reduced move set.**
  - *Dropped families:* `cell`/`cellpair` at the complete level, dense reroutes,
    atoms of the dispatch fields.
  - *Frames from x only.*
  - *LINEAR instead of T2.*

  Any of these could cost mixing. Phase 0 measures it; nothing is dropped
  without an ablation.
* **Robustness of the slope-jump sampler:** near-equal breakpoints, huge slope
  ratios (c_shed/c_dump = 600), FP32 intermediates and improper rays. It needs its
  own KS and ΔC tests, beyond the ones for the dense form.
* **Silent bias from colouring bugs.** Overlapping footprints in a round break
  exactness without any error. `check_conflicts` must run on every new schedule
  and template set, including the learned templates after each update.
* **Sequential capacity loop.** At full Europe (≈ 20 000 investable units) it is
  ≈ 0.5–1 s per sweep. That is acceptable against a 10–20 s operational sweep, but
  it grows if capacity mixing needs several moves per unit and sweep.
* **Memory and tiling at full Europe.** Round workspaces scale with sites ×
  bucket size, so wide rounds must be tiled. That leaves ≤ 20–30 chains per 80 GB.
* **The sampler problems that remain.**
  - *Weather at the fine level:* MLDA acceptance ≈ 0 and no fine weather family.
  - *Capacity pinning at ρ₁.*
  - *Post-correction of the bias.*

  The port does not solve these. Many chains do make two things natural: pooled
  adaptation, and replica ideas such as a shared k across weather replicas with a
  mean force. Both are research questions outside this design.
* **Pooled adaptation** across chains is covered by the product-space argument.
  test_exact checks only one chain, so the GPU test must also run with pooling
  switched on.


## Review notes (main session)

The corrections below are adopted in the family map (§3.3) and the phase plan (§8).

* **MLDA is shelved**, so there is no coarse chain to port. The hierarchy acts
  through the overcomplete fine-level moves. If delayed acceptance with
  surrogates is added for expensive move families (docs/archive/todo.md item 30), the
  surrogate evaluations are extra batched kernels in the same program.
* **Keep the coarse cell and cell-pair moves and the dispatch atoms** as templates
  until the phase-0 ablation shows they do not matter. They are the multigrid
  part of the move set. They are sparse within a cell, and their cost is linear
  in the cell size once levels are chosen by size (docs/archive/todo.md item 6).
* The other simplifications are justified as stated:
  - representation changes (node-value state, the generic move record, FP64,
    shared keys, learned compensation as the default) do not change the target
    or the mixing;
  - removed families (time-constant block moves) showed no gain;
  - local-ball re-routes replace the dense ones after the ablation.

## Amendment: model completion and merged CPU performance work (main session)

The model gained terms after this design was written (docs/archive/model_completion.md).
They change the device cost and the line sampler as follows.

| term | device form | sampler |
|---|---|---|
| line losses (interpolant of u² on a fixed grid), soft overload band | per-line piecewise-linear table in u = \|f\|/s, shared grid `U` (≈ 10 points); losses enter both end residuals | the grid crossings are primary breakpoints: a fixed number per moved line, padded like the clamp points |
| signed lossy links | `link_inj(η, u)`, one kink at u = 0 | one more primary breakpoint per moved link |
| merged vintages | `gen_vint`: one kink per unit at y = q_lo/k | one more primary breakpoint per generator entry (both sides of a T2 kink) |
| fleet unit commitment | an instant-like field per committable carrier (start-up couples τ and τ+1, like SoC); walls y ∈ [g_min c, c] | walls and start-up kinks are secondary quantities; the commitment follows the output in main-step moves (headroom kept), never in the linked previous step |
| requirement-tracking T2 kinks (needed with losses, lossy links and weather moves) | Q_k(t) = Q_k + Δr_n(t) from a first pass over the non-compensating entries | the secondaries are only piecewise linear between primaries: event-driven walk (probe slopes, earliest root). On the device either the same event loop with a bounded event count, or LINEAR compensation for these moves (direction constant, no kink tracking). Decide in phase 3 |
| fine-level weather moves | windowed propagation of one weather coordinate to the finest anomalies (dense small matrices per level); weather entries per (node, step) with data-map kinks | Gaussian prior on the line: truncated-Gaussian segments (`logΦdiff`, `trnorm`); regimes by centered FFBS on the host |
| CO₂ price, brownfield floors | parameters only | — |

**Effect on the phases.**
* **Phase 1:** `cost_parts` must reproduce `energy_cost` including all these terms.
* **Phase 2:** the line-term tables and the extra primary breakpoints are added.
* **Phase 3:** UC and requirement-tracking kinks are added.
* **Phase 5:** the weather moves are listed there already.

**CPU reference.** The merged CPU kernel (perf_notes.md) is the reference for timings.
Its recorded-term evaluation applies to the plain model only (`cacheable`).

## Phase 0 spike: results (main session)

`device/spike.jl`, Reactant with XLA on the GPU of this machine (device 0; the ptxas messages show CUDA compilation).
The spike is a device loop (`@trace for`, 100 rounds) with loop-carried state:
* a (C, T, N) array updated by a scatter-add, written as a contraction with a one-hot (B, N) table;
* a sort per (chain, step) and a running accumulator;
* uniforms from Reactant's RNG (`ReactantRNG`, `rng_bit_generator`), its state carried by the loop.

**Results:**
* It compiles (≈ 28 s the first time, 2 s for the second variant) and runs at ≈ 100 µs per
  round at C = 16, T = 64, B = 24.
* With given uniforms it is bitwise identical to the same code on Julia arrays.
* The "aliasing pattern of loop-carried variables changed" failure of the earlier
  prototype did not recur.

**RNG decision.** Use Reactant's RNG instead of a hand-written Philox: Philox code
in array form overflowed the tracer's stack, and the built-in generator is native
XLA. The device stream then differs from the CPU stream, so correctness against
the CPU is checked in two ways:
* bitwise (or to round-off), with the same mock uniforms passed to the CPU reference and to the device for a few steps.

*Correction (user):* no statistical end-to-end tests (KS, z-tests, many-chain MCMC runs) for the port at this stage. The CPU implementation is the reference; the comparisons above replace the statistical acceptance tests of §8.

The Philox test vectors of phase 1 are dropped.

**Tracing rules found:**
1. Build every shape-dependent array outside `@trace for`: captured integers become traced numbers inside the loop.
2. Pass static shapes as `Val((…))`.
3. Never capture a lazy `reshape` of a traced array; materialize it first.
4. Convert the traced loop index through array arithmetic, e.g. `(zeros_int .+ k) .% UInt32`.
5. Model structs must be type-parametric in their array types to convert with `to_rarray`. Compile-time constant models (closures) make XLA fold the large constant contractions and compile very slowly, so the model is a traced argument.
6. A compiled call updates its input arrays in place (donation): the host copies what it keeps.
