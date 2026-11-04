# CPU performance work (docs/todo.md items 5–7)

Working copy `work/perf`, to be merged into the main `src/`. Two agents worked here; the
first one (no notes) did part of item 7 and a first version of item 5/6, the second one
reviewed it, finished it and continued. Everything below is the state of this copy.

**Merge warning.** The main `src/local.jl`, `src/energy.jl`, `src/adaptive.jl`, `src/run.jl`
were extended in parallel (line losses, lossy links, requirement-tracking kinks, fleet unit
commitment, `gen_cost(g, i, …)`). This copy is based on the main state of 15:19. The
diff of whole files is therefore misleading; merge function by function using the list
below. The one place that needs thought is the cached walk evaluation (§3.1): it mirrors
the plain `eval_move!` term by term and must be extended (or switched off) for the new
terms — see "Merging the cached evaluation" at the end.

## 1. Item 5 — localized cross-node pair moves

`src/local.jl`

* `xloc_directions(sys, GS, nodelines; rp=1, rb=0)` (new): for every node n its partners m
  (same AC component, 1 ≤ hops ≤ `rp`) and per pair a fixed re-routing template: the angle
  change on the ball B (nodes within `rb` hops of n or m) solving L_BB Δθ_B = e_n − e_m with
  Δθ = 0 outside B (grounded solve with zero Dirichlet boundary; dense Cholesky of the small
  ball system at construction). Every node of B keeps its residual; the dipole leaks into
  the first layer outside B, which is returned per pair as (nodes, requirement change per
  unit t). If B is the whole component the exact global re-routing is used (no leak).
  Network-only data, independent of the state.
* `LocalModel`: new fields `xpart`, `xnod`, `xval` (partners and templates), `xlkn`,
  `xlkg` (leak nodes and their requirement changes); new keyword arguments
  `cell_power=1.0, xr_partner=1, xr_ball=0`.
* `prep_xloc!(rng, ws, P, LM, LC, τ, jac; κmin)` (new), two families:
  * `xloc` (`jac === nothing`; also reachable via `prep_random!`): n uniform, partner
    uniform among `xpart[n]`, one random unit at each node (±1/κ as in `xpair`), the
    template angles; the leak goes into the residuals at the leak nodes.
  * `xlocc`: the same plus exact pro-rata (T2) compensation of the leak at every leak node
    (`prorata!`, kinked entries, Jacobian terms in `jac`; the Jacobian MH correction is
    applied in `local_sweep!`, `:xlocc` added to that family list).
  Exactness: selection (n, partner, units) and the template depend only on the network and
  (k, x) (κ); `xloc` is a plain line move, `xlocc` has exactly the structure of `tpot`
  (σ fixed along the line, Jacobian by an independence MH step). Directions are not
  balance-preserving outside the ball (`xloc`) — allowed.
* `plain_families(E)` (new): the default plain families, used by `local_sweep!` (default)
  and `flow_families` (flowmoves.jl). The cross-node pair rate 0.15 is split
  `xpair 0.03` (global dense re-routing, kept at a low rate), `xlocc 0.08`, `xloc 0.04`.
  All other rates unchanged (so `flow_families`' multiplier and the adaptive sweep's
  `wb` are unchanged).

Choice of the ball radius (`scripts/xloc_leak.jl`, `scripts/xloc_eff.jl`): the Dirichlet
truncation leaks a lot (the dipole flow crossing a circle of radius r decays only like
1/r on a meshed grid):

| ball radius rb | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| leak (Σ\|Δr\|/2 per unit), 40 nodes | 0.49 | 0.24 | 0.12 | 0.05 | 0.01 |
| leak, 256 nodes | 0.50 | 0.29 | 0.19 | 0.13 | 0.10 |
| template nodes (mean), 256 nodes | 2 | 7.6 | 17 | 31 | 47 |

so uncompensated templates are limited by the balance kinks of the leak nodes. Efficiency
per CPU time (single thread, heuristic start + short relaxation; t is the injection shift
in MW; median |t| is the robust measure, E[t²] is dominated by the far-from-equilibrium
start):

| 256 nodes | µs/move | accept | median \|t\| (MW) | E[t²]/µs | median²/µs |
|---|---|---|---|---|---|
| xpair (global) | 68.7 | 1.00 | 12.6 | 391 | 2.3 |
| xloc rb=0 | 6.4 | 1.00 | 5.9 | 339 | 5.4 |
| xloc rb=1 | 9.2 | 1.00 | 7.5 | 461 | 6.1 |
| xloc rb=2 | 13.6 | 1.00 | 9.1 | 489 | 6.1 |
| xlocc rb=0 | 8.6 | 0.98 | 9.6 | 3096 | 10.8 |
| xlocc rb=1 | 13.3 | 0.98 | 10.1 | 1373 | 7.6 |
| xlocc rb=2 | 18.6 | 0.99 | 11.9 | 1290 | 7.7 |

At 40 nodes the global `xpair` is still the most efficient per µs (E[t²]/µs 2.9·10⁴ vs
9.3·10³ for `xlocc` rb=0): the network is small and the dense re-routing cheap. At 256
nodes `xlocc` with rb = 0 is 4.7× (median) to 8× (E[t²]) more efficient than `xpair`, and its
cost does not grow with N. Hence the default rb = 0 (ball = the pair; flow re-routed on
the lines of n and m, leak compensated at their neighbours) and partners = graph
neighbours. Both are config keys (below).

## 2. Item 6 — cell moves sized by level

`src/local.jl`

* `LocalModel.qcum` (new field): cumulative level probabilities of the cell moves, level q
  ∝ (number of cells at q)^`cell_power` (default 1: ∝ number of cells, finer cells
  favoured; 0 gives back the uniform level).
* `draw_level(rng, LM)` (new) replaces `rand(rng, 1:LM.qlev)` in `prep_random!`
  (`cell`, `cellpair`) and `prep_block!` (`bcell`, `bcellpair`). Fixed probabilities, so
  the selection stays independent of v.

At 40 nodes (levels 1 / 10 / 40 cells) the levels get 2 % / 20 % / 78 %; at 256 nodes
(1 / 10 / 256) 0.4 % / 3.7 % / 96 %. Per-move cost of `cell` 7.3 → 3.4 µs, `cellpair`
12.7 → 5.2 µs at 40 nodes.

## 3. Item 7 — allocation-free, faster move evaluation (mathematics unchanged)

### 3.1 Line sampler (`src/local.jl`)

* `EntTerm`, `LineTerm`, `StuTerm`, `CellTerm` (new isbits structs) and
  `eval_move!(…; mode)`: mode 1 evaluates at t = 0 and records the compact per-term data
  (value at 0, slope, parameters) in `ws.tent/tline/tstu/tcell`; mode 2
  (`eval_cached!`, new) evaluates from the records with the *same arithmetic* (bitwise
  identical results; asserted in `test_local.jl`). Only in the SoC form (the LP storage
  form keeps the plain path). `ws.trec` marks valid records; `clear!` invalidates them,
  mode 2 without valid records falls back to the plain evaluation.
* `walk!`/`sample_line!`: one recording per line draw (`mv_fill!`, shared by the right
  and the left walk via `sec0`), all further evaluations cached (`mv_evalc!`). No
  `filter`/`sort`/`vcat`/`sortperm`/new `roots` per move: scratch in `LineBuf` (new
  struct, one per `MoveWS` and per `BlockMove`). Breakpoints: ≤ 16 sorted (insertion),
  longer lists heapified in place and popped in walking order (`heapify!`, `heappop!`,
  `siftdown!`, `hlt`: the same sequence as a full sort, O(n + p log n), no allocation;
  Base's `partialsort!` allocates). Secondary roots: `insort!` (insertion sort).
* `commit!`, `commit_block!`: the commit evaluation at the drawn t uses the recorded terms
  (mode 2) and then invalidates them.
* `BlockMove`: new field `lb::LineBuf` (old 8-argument constructor kept);
  `blockmove(wss, own, P, LM, LC)` (new) builds a block move on the persistent buffers
  (`bsec`, `bbps`, `bτs`, `lb`) of workspace `own`; used by `local_sweep!`,
  `block_sweep!`, `storage_ms_sweep!`, `adaptive_block_sweep!`.

### 3.2 Move preparation

* `reroute!(x, GS, n, m, b, y)` (new allocation-free method; the 4-argument form remains):
  scratch `ws.rb`, `ws.ry`, result in `ws.nbuf` (`prep_reroute!`, `prep_tline!`).
* `prorata!`: no temporary arrays; units from `LM.genunits` (new field: per node, generator
  units with a complete field); same sums in the same order.
* `prep_tpot!`: requirement changes in `ws.gbuf` on the touched nodes only (`ws.gnodes`,
  sorted, so the same ascending node order as the old dense loop).
* `prep_tdir!` (flowmoves.jl): `ws.gbuf` instead of `zeros(N)`.
* `prep_tsparse!` (flowmoves.jl): no Dicts/Sets: `ws.nbuf` for the angles, `ws.gbuf` +
  `ws.gnodes` (first-touch order) for the requirement changes, `ws.lseen`/`ws.lseenl` for
  the visited lines. (Node order of the `prorata!` calls changes from Dict order to
  first-touch order: same move, round-off-level differences.)
* `prep_tcut!` (flowmoves.jl): atom draw without `cumsum` (same comparisons), views
  instead of slices of the Haar column.
* `prep_storage_ms!` (flowmoves.jl): per-node requirement changes in `wq.gbuf`/`wq.gnodes`
  instead of a Dict (round-off-level differences in the order).
* `prep_spair!`, `prep_tsoc!`, `prep_block!` (`bsoc`): storage units of a node from
  `LM.nodestor` (new field) instead of a comprehension per move.
* `adaptive_block_sweep!` (adaptive.jl): `LM.genunits[n]` instead of a comprehension.
* `storage_ms_sweep!`, `adaptive_block_sweep!`: the single-unit selection reuses a vector.

### 3.3 Sweeps: type stability of the threaded loops (largest single gain)

`LC`, `wsets`, `pools` (and `families` in `local_sweep!`) were reassigned before being
captured by the `Threads.@threads` closure, so Julia boxed them (`Core.Box`): every use
inside the move loop was type-unstable (dynamic dispatch, boxed return values). Now single
assignments with new names (`LCc`, `wsl::Vector{MoveWS}`, `pl::Vector{BlockPool}`,
`fl`, `fams::Vector{Symbol}`) in `local_sweep!`, `block_sweep!`, `storage_ms_sweep!`,
`adaptive_block_sweep!`. `pools` passed as `Vector{Any}` (`LocalKernel.pools`) is converted
once per sweep. Per-step move lists and Jacobian buffers live in the workspace
(`ws.mvbuf`, `ws.jac`).

### 3.4 New `MoveWS` fields

`lb, gbuf, gnodes, nbuf, rb, ry, lseen, lseenl, tent, tline, tstu, tcell, trec, bsec,
bbps, bτs, jac, mvbuf` (all scratch; `gbuf` and `lseen` are kept zero/false between
uses).

## 4. Config keys and struct changes (merge checklist)

* `Config` (run.jl): `cell_level_power = 1.0`, `xloc_partner_hops = 1`,
  `xloc_ball_hops = 0`; passed to `LocalKernel(P; …, lmkw=(cell_power, xr_partner, xr_ball))`.
* `LocalKernel`: new last field `lmkw::NamedTuple`; `local_model` passes it to
  `LocalModel`. Other direct `LocalModel(P, ℓ)` calls use the defaults.
* `LocalModel`: new fields `qcum, genunits, nodestor, xpart, xnod, xval, xlkn, xlkg`
  (appended at the end), keyword arguments `cell_power, xr_partner, xr_ball`.
* `MoveWS`: fields of §3.4 appended at the end. `BlockMove`: field `lb` appended.
* New move families `:xloc`, `:xlocc` (default mixes changed: `plain_families`).

## 5. Measurements

`scripts/bench_perf.jl` (heuristic start, levels [(10, 24), (N, 24)], SoC + renewables as
load, Haar angles, daily). `fam`: µs per move of one family at 1 thread (a local sweep
with only that family, atoms off, minus the final `evaluate!`; per-step overhead
included). `sweep`: s/sweep at 4 threads (`nice 19`, the user's machine was busy in
parallel, so ±10 %). Setup/refit times are not included anywhere.

**Baselines.** The main copy changed during this work (unit commitment, losses, … added
around 16:30) and got slower by itself (≈ 1.2–1.5× per move at 40 nodes; it was also not
loadable for a while). Two "before" columns therefore: *main (orig.)* = the main state
this copy was made from (40-node family numbers measured at 15:46, 40-node sweeps at
≈ 16:20, before the main edits), and *main (now)* = a frozen snapshot of the main `src/`
taken at ≈ 16:45 (includes the new features, switched off). The 256-node "before" is only
available for *main (now)*; the speed-up against the original main there is
correspondingly smaller (estimate: divide by ≈ 1.3–1.5).

### Per family, µs per move (1 thread)

| family | 40: main (orig.) | 40: main (now) | 40: perf | speed-up vs orig. | 256: main (now) | 256: perf | speed-up |
|---|---|---|---|---|---|---|---|
| atom | 22.4 | 27.0 | 14.8 | 1.5× | 90.9 | 47.9 | 1.9× |
| unit | 3.7 | 3.9 | 2.4 | 1.6× | 3.9 | 2.4 | 1.6× |
| pair | 4.9 | 5.0 | 3.4 | 1.5× | 5.9 | 3.6 | 1.6× |
| xpair (global) | 19.5 | 26.6 | 12.2 | 1.6× | 134.9 | 62.5 | 2.2× |
| xlocc (new) | — | — | 6.9 | | — | 7.4 | 18× vs xpair (now) |
| xloc (new) | — | — | 4.9 | | — | 4.9 | 28× vs xpair (now) |
| potential | 5.3 | 5.7 | 3.7 | 1.4× | 5.6 | 3.6 | 1.5× |
| cellpair | 12.7 | 14.5 | 4.0 | 3.2× | 98.9 | 5.3 | 19× |
| cell | 7.3 | 8.3 | 2.7 | 2.7× | 38.3 | 3.1 | 12× |
| spair | 7.5 | 8.1 | 5.1 | 1.5× | 8.5 | 5.7 | 1.5× |
| tpot | 12.2 | 13.6 | 7.7 | 1.6× | 14.1 | 8.2 | 1.7× |
| tmode | 79.7 | 102.6 | 34.5 | 2.3× | 978.8 | 275.8 | 3.5× |
| tcut | 30.0 | 33.7 | 15.8 | 1.9× | 98.3 | 34.3 | 2.9× |
| tline | 23.2 | 33.0 | 15.8 | 1.5× | 200.2 | 79.5 | 2.5× |
| tmode0 | 35.8 | 69.1 | 25.7 | 1.4× | 1269.7 | 378.6 | 3.4× |
| tline0 | 16.9 | 23.2 | 11.3 | 1.5× | 134.4 | 62.4 | 2.2× |
| tsoc | 9.6 | 10.1 | 6.2 | 1.5× | 12.0 | 7.3 | 1.7× |

The cell/cellpair gains are mostly item 6 (fewer coarse cells); everything else is item 7
(same moves, same arithmetic). Allocations per move in a tight single-thread loop are 0 B
for every family (before: 3–70 KB at 256 nodes; `bench_perf.jl fam` still shows ≈ 60–300 B
per move, which is the per-step overhead of the sweep: move list, block move, Xoshiro,
statistics).

### Per sweep, s/sweep (4 threads)

| sweep | 40: main (orig.) | 40: main (now) | 40: perf | vs orig. | vs now | 256: main (now) | 256: perf | vs now |
|---|---|---|---|---|---|---|---|---|
| local_sweep!, default families | 0.672 | 1.037 | 0.510 | 1.3× | 2.0× | 16.0 | 7.65 | 2.1× |
| local_sweep!, flow families | 0.869 | 1.338 | 0.554 | 1.6× | 2.4× | 32.6 | 12.0 | 2.7× |
| storage_ms_sweep! | 0.102 | 0.127 | 0.072 | 1.4× | 1.8× | 1.05 | 0.55 | 1.9× |
| block_sweep! | 0.245 | 0.382 | 0.180 | 1.4× | 2.1× | 10.4 | 1.60 | 6.5× |

The default-family sweep changes the move mix (xpair → xloc/xlocc) as well; the
flow-family sweep at 256 nodes is the production fine-level sweep (`flow_sweep!`).
At 256 nodes the atoms are now the largest item (467 200 atom moves per sweep at
48 µs ≈ 22 s single-thread, i.e. ≈ 5.6 s of the 7.65 s default sweep at 4 threads);
the dense transport modes `tmode`/`tmode0` (≈ 300 µs) and `tline` are next.

Allocations per sweep (real, from `@allocated` in a script, 40 nodes, 2 threads): default
local sweep 9.6 MB, flow-family sweep 14.6 MB, mostly `LocalCost` and `evaluate!` (main:
0.87 GB and 1.24 GB per sweep from `bench_perf.jl`). `bench_perf.jl sweep` still prints
≈ 420 MB for the *default* line in this copy; that is an artefact of the benchmark
closure (the same call measured directly allocates 9.6 MB).

**Bitwise regression.** `bench_perf.jl regress` (fixed seeds, 2 threads) against the main
copy of 15:19–16:20 (`scripts/regress_cmp.jl`): plain families (unit, pair, xpair,
potential, spair) and the flow families without tcut/cell/cellpair give *identical* v
after a sweep each; the storage multiscale sweep differs at 1.6·10⁻¹⁵ (node order of the
pro-rata calls, Dict → list). So the cached evaluation, the heap-ordered walk, the commit
from cached terms, the buffer changes and the unboxing do not change any number.

## 6. Tests (final state, 4 threads, nice 19)

* `test/test_exact.jl`: operation part passes, z = 1.05 (op), 0.83 (shed), −1.69 (mean
  SoC), −0.84 (line flow); the full adaptive chain uses the new default mix (xpair, xloc,
  xlocc; on the 3-node network rb = 0 gives a truncated template with a leak to the third
  node, so the localized branch is exercised). Capacity part (known to be inconclusive):
  passes, z = 0.05, 2.96, −0.21, 1.13, −0.71.
* `test/test_local.jl`: all pass (33 + 12 + 349 + 676). Additions in this copy:
  - `xloc` and `xlocc` in the cost-consistency loop (Δ from `eval_move!`/`mv_eval!` vs
    full `evaluate!` after the commit, both storage forms, 300 trials per level);
  - SoC form: the cached walk evaluation equals the plain one bitwise
    (`mv_eval!(bm, 0; mode=1); mv_eval!(bm, t; mode=2)` vs `mv_eval!(bm, t)`, value and
    secondary quantities) for every trial;
  - SoC form: KS test of the linked two-step line sampler (cached walk, heap) against
    quadrature;
  - the localized templates keep the balance except at n, m and the listed leak nodes
    (rb = 0 and 1); level probabilities of the cell moves are a proper cdf.
* `test/test_flow.jl`: all pass (2 + 186).
* Bitwise regression against the main copy: see §5.

## 7. Scripts (new, in `scripts/`)

* `bench_perf.jl` (`fam`, `sweep`, `regress` modes), `regress_cmp.jl`: the measurements
  of §5 (work with both copies: run from `work/perf` with `--project=<copy>`).
* `prof_kernel.jl`: flat profile of a local sweep.
* `xloc_leak.jl`: leak and size of the localized templates per ball radius.
* `xloc_eff.jl`: efficiency of xpair vs xloc/xlocc (µs/move, median |t|, E[t²]).

## 8. Not done / next candidates

* Atoms are now the largest cost at 256 nodes (coarse Haar atoms of θ and of the
  dispatch touch many lines/nodes); they are the coordinate updates, so only the
  per-term evaluation cost can be reduced further (e.g. a per-move local cell index to
  avoid the `dS` scatter, storage parameters inside `StuTerm`).
* Dense transport modes (`tmode`, `tmode0`, ≈ 300 µs at 256 nodes) and `tline`
  (dense re-routing) are unchanged in structure; `tcut` is their localized form.
  `prep_amode!`/`project_face` still allocate a dense B × N matrix per move (low rate).
* The LP storage form does not use the cached evaluation (plain path, as before).
* `prep_block!` builds closures with boxed captured variables (per block move, small).

## Merging the cached evaluation

The recorded terms mirror the plain `eval_move!` of this copy: generators
`gen_cost(mc, A, k, w, …)`, lossless links `u = cap·clamp(y)`, lines
`cv Δt max(|f| − s, 0)` without losses, SoC storage `soc_cost`, cell terms. The main copy
now has `gen_cost(g, i, …)`, `link_inj(η, …)`, `line_eval(LT, …)` with losses,
requirement-tracking kinks (`kdyn`, second pass) and unit-commitment entries
(`uc_entry!`). When merging, either extend `EntTerm`/`LineTerm` and `eval_cached!` with
exactly the same expressions (record the t = 0 values, evaluate the same formulas), or
record only when the move has none of the new terms: e.g. in `eval_move!` set
`rec = mode == 1 && !ws.anydyn && <no UC entry> && <lossless lines/links>` and leave
`ws.trec = false` otherwise (mode 2 then falls back to the plain path automatically).
The bitwise check in `test/test_local.jl` (`Δc == Δ && secc == mv_sec(bm)`) catches any
mismatch; run it with the new features switched on.
