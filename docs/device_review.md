# Review of the Reactant port (device/), 2026-11-04

Development note: a fresh-eyes review (Fable) of the device code on main at 9a64d54, read-only, with small trace-only probes. The fixes are assigned in [worklog.md](worklog.md) (2026-11-04, night); the system documents describe the code as built.


Reviewer: a fresh-eyes pass over `device/src/*.jl`, `device/test/*.jl`, the documents (device.md, sampler.md §3/§11,
performance.md §7, testing.md) and the CPU code they mirror (`src/local.jl`, `storage_long.jl`, `replicas.jl`,
`energy.jl`, `kop.jl`). Read-only; nothing in the repository was changed. Scratch files (probe scripts, outputs) are in
`/tmp/claude-1000/-home-hellmann-Repos-ESM-Sampling/22beeddd-edd7-46d6-bac0-5fd06a2f554b/scratchpad/fable_review/`.

Severity: **H** (changes results or blocks the port), **M** (fragile, costly, or a silent loss), **L** (note).

---

## 1. Findings, ranked

### H1. Every host-drawn choice of the schedule is drawn once per run from `MersenneTwister(1)` and never redrawn

- `device/src/rounds.jl:195` (`schedule(…; rng=MersenneTwister(1))`), `rounds.jl:772-773` (`plan(…; rng=MersenneTwister(1))`);
  both drivers call `plan` without an `rng`: `device/src/driver.jl:228`, `device/src/replica_run.jl:136`.
- The choices frozen for the whole run (hundreds of sweeps × checkpoints): the `:spair` partner pair per (unit, step)
  (`comp.jl:47`), the cell carrier(s) per (cell, step) (`cells.jl:69,72`), the block start offset and the profile per site
  (`blocks.jl:62,76,137,149`), the dispatch-shape pair and profile (`dshape.jl:43,59`), the sparse sites (`sparse.jl:56`),
  and for `:slong` the interval `(a, L)` of every unit (`slong.jl:34`). Only the dense angle directions are ever
  refreshed (`tdir_refresh`, and only inside `adapt_step!` on an `AdaptiveModes` update or with frames: `driver.jl:168,179`).
  After a resume the same `MersenneTwister(1)` reproduces the same frozen set.
- The CPU draws all of these per move (`src/local.jl:1608-1656` `prep_random!`, `prep_spair!`, `storage_long.jl:277-282`
  `(a, L)` per move) or per sweep (`src/flowmoves.jl:283` block offset).
- Effect: each round is still a π-invariant kernel, so invariance holds, but the device kernel is a fixed cycle over a tiny,
  seed-independent subset of the CPU's move mixture. Concrete scenario: `:slong` at 40 nodes daily gives every hydro/H₂ unit
  exactly the same 4 intervals `(a, a+L)` per sweep for the entire run (the CPU: 32 fresh intervals per unit and sweep,
  `src/run.jl:28`); every storage block keeps one offset and one profile forever; a (storage unit, step) `:spair` site
  always pairs the same two generators. Mixing of the seasonal and transport-in-time directions is far below what the
  CPU/device comparisons of single rounds measure; nothing in the current tests (single rounds with given uniforms) can
  see this.
- Fix (no recompile needed if the table shapes stay): redraw the host choices at every host checkpoint from a restart-seeded
  stream and re-upload the tables, exactly like `tdir_refresh` — pad each family to a fixed site count B over draws (blocks:
  the maximum over offsets, the `on` mask covers the rest; slong/spair/cells: B is already fixed). Better still, move the
  cheap choices into the program and draw them from the `ReactantRNG` per round: a per-site integer from a candidate table
  (pairs, carriers, profile ids), and for `:slong` the `(a, L)` pair with the column tables computed by index arithmetic
  (`mod1(a + j, T) + (s − 1)T` from an iota; `wa = (mod(τ − a − 1, T) < L)`), which also removes the 5·B·T·R host tables.

### H2. The `:slong` compile time is Julia inference of Reactant's varargs `cat`, not XLA

- Culprits: `device/src/slong.jl:85` `rep(x) = cat(fill(y, T)...; dims=4)` (7 calls, T = 1460 operands each), and the commit
  path `terms.jl:639-644` `slot_changes` (T one-column slices) + `comp.jl:347` `dr4(dr) = cat(dr...; dims=3)` (T operands)
  used by `slong.jl:138` (and with N or Ns operands by `tdir.jl:138`, `sparse.jl:222`, `cells.jl:175`, `blocks.jl:276`,
  `dshape.jl:136`).
- Reactant routes these through `Base._cat_t(dims, T, X::TracedRArray...)` (`~/.julia/packages/Reactant/1C1cP/src/TracedRArray.jl:531`):
  `cat_size_shape`/`promote_eltype` recurse over the varargs tuple under Reactant's interpreter, so inference cost grows
  super-linearly with the operand count. Measured here (trace only, `@code_hlo optimize=false`, Reactant 0.2.304,
  `scratchpad/fable_review/cat_trace2.jl`), operands materialized 4-d traced arrays as after `c4`:
  T = 8: 0.63 s; 64: 10.2 s; 128: 34.7 s (≈ quadratic); 365: not finished after 230 s (killed), RSS 2.3 GB at 128 and
  rising. With lazily reshaped operands T = 365 ends in a `StackOverflowError` in `Base._cat_size_shape` through
  `call_with_reactant` (`cat_trace.jl`). Plain Julia arrays do the same `cat` in 0.14 s at T = 1460 (Base falls back to
  dynamic dispatch), so the cost is specific to the traced path. Extrapolated to T = 1460 this is the hour/15 GB scale
  reported for `:slong`; each distinct operand type/count is its own specialization (`rep` once, `dr4` once more).
- The `:slong` probe at 2 sites (`slong_trace.jl`, T = 1460, CPU backend): setup 45 s, plan 82 s (slope-jump form, cap 7 308
  entries = 5T + 8 as expected), and the trace alone had not finished after 410 s at 7 GB RSS when the 600 s limit killed it — the XLA passes
  were never reached. Apart from these concatenates the program is O(1) in T: every term type is one stacked
  (C, B, ·, T) array, the slot sums are one gather + one sum, the global part is one payload sort.
- Fix: never splat O(T) or O(N) operands into `cat`.
  1. `rep` → a broadcast (`y .+ zeros(1, 1, 1, T)`), or keep the per-unit fields (C, B, 1, 1) and let broadcasting work
     (only `soc_qadj4`'s `[:, :, :, 1:2]` slices need the two end columns explicitly);
  2. let `slot_changes` return the (C, B, 1, R) array `inj` itself and pass it through `do_round`; delete `dr4`; the
     2-slot families index `inj[:, :, :, j:j]`;
  3. a helper `catn(xs::Vector, dim)` = `Reactant.Ops.concatenate(xs, dim)` for traced arrays (a Vector argument, no
     varargs) and `cat(xs...; dims)` for Arrays, used by `stk` (`terms.jl:178`), `sort3` (`jump.jl:131`), `jump_points1`
     (`jump.jl:484`), `dense_points`/`root_count`, `addz!`, `uc_concat`, and the two `rep`s in `blocks.jl:202,267`.
  Expected compile of `:slong` afterwards: the `:sblk32` scale (about 15 s). Check: trace time of one round at 2 sites
  before/after, and the op counts of the unoptimized HLO (the probe script prints them).

### H3. The residual cache `r` is never rebuilt inside a run; fatal for the Float32 build

- `r` is built once (`rounds.jl:111-114` `flat_state`) and carried through every sweep, `capacity_pass` (`driver.jl:63-80`),
  `lgv_pass`/`weather_update` and every checkpoint; `rebuild_r` is used nowhere else (`grep rebuild_r device/src`).
  FP64: ~1e-10 MW per 100 sweeps (test_rounds), fine. Float32 (`ESMDevice.F32`): the scatter-adds round at ~1e-2 MW at
  |r| ≈ 1e5 MW, so after a few checkpoints the balance potential `Δt c_shed r₊` is off by O(10–100 €) per node-step and
  the shed/dump roots sit at the wrong t; the error grows with √(moves).
- Fix: rebuild r from the node arrays on the device at every host checkpoint (the residual of `cost_parts` on the node
  arrays is already traceable: one call at the end of `capacity_pass`/`lgv_pass`, or a small fourth program), and report
  `max |r_cache − r_rebuilt|` per checkpoint. Mandatory before any F32 result is trusted.

### M1. Float32 build: TF32 matmuls and a driver that does not convert

- (a) Reactant's `dot_general_precision` defaults to `PrecisionConfig.DEFAULT` (`Reactant/src/Configuration.jl:30`); on
  Ampere-class GPUs XLA then runs F32 `dot_general` in TF32 (10-bit mantissa). The one-hot contractions `tmul`/`tmulT`
  (`core.jl:223-225`) carry the whole residual and score arithmetic: `cost_parts`/`rebuild_r` (`core.jl:247-308`),
  `cap_score` (`score.jl:83,92,98`), `kop_follow`/`disp_follow`/`weather_comp` (`langevin_dev.jl:99,112-125,313-319`),
  `unit_node_r` (`kmove.jl`). A node sum of 1e4 MW would be off by ~10 MW. Fix: `Reactant.with_config(;
  dot_general_precision=PrecisionConfig.HIGHEST)` around the F32 compiles, or replace the one-hot matmuls by
  gathers/scatter-adds (what full Europe needs anyway, `core.jl` header). FP64 is unaffected.
- (b) `replica_device_setup` uploads with plain `Reactant.to_rarray` (`replica_run.jl:144`), without `ftconv`, so the F32
  replica driver feeds Float64 state, tables and weather into a Float32-literal program: a mixed-precision (mostly
  Float64) program, not the F32 build; `device_setup` does it right (`driver.jl:244`). The same `R` is used for every
  per-step upload (`W`, `chg`, `D0h`, `Ah`, `LP`).

### M2. Replica driver: plan options, family set, host probe and host weather arrays

- `replica_run.jl:136` calls `plan` without `jump`/`backend` → `JUMP_RATIO["cpu"]` is used on the GPU (form choice 4–11 %
  slower for `tsoc`, `sblk2`, `cell3`, `cellpair3`, performance.md §7). Pass `backend=cfg.device_backend` as `driver.jl:228`.
- The default operation set of the replica driver is `FAMILIES_FAST` (`replica_run.jl:329`, `driver.jl:283`, script default
  `FAMILIES=fast`): no transport in time, blocks, cells or slong, while the probe (`replica_run.jl:307`) runs the CPU kernel
  with all CPU families. The "lag gap" then measures the weak device kernel more than the Langevin lag and the adaptation
  shrinks η_w accordingly. Replica runs should use `families=:all` (and the configs/campaign lists should say so), or the
  probe should run the device families.
- The lag probe runs on the host CPU, synchronously: `probe_len` = 30 CPU sweeps of one replica per chain every 20 steps
  (`replica_run.jl:299-315`); at 40 × 6 h that is ~100 s per chain per probe against a device step of 0.03–0.1 s on an H100,
  so the probe dominates the wall time of a run with 4 chains. Idea: give every chain one extra replica slot that is a
  frozen copy (η-less by construction — it shares the chain's k, but its score gets weight 0 in `W`; the in-flight window is
  taken from the copy's scores over the probe length), so the probe is a free by-product of the existing programs; or
  a second `run_sweeps` compile on a `C = nchains` batch.
- `replica_device_weather!` (`replica_run.jl:228-233`) recomputes `weather_arrays` and `load_shares` for all replicas
  on the host and uploads the full (C, T, N) and (C, T, U_r) arrays whenever any replica moved (every ≈ n_b/M steps).
  At 256 nodes × 1 460 × C = 64 that is ~0.2 GB per upload plus `weather_nodes` for 64 replicas. Compute and upload only
  the moved replicas (an index list and a device-side scatter into resident D0/A).
- Minor: `dev[:acc0]()` allocates and uploads a fresh `ScoreAcc` every step; `sync!` after every program serialises the
  score program with the next sweep (fine for timing, costs pipelining).

### M3. The slope-jump cap is fixed from the initial state and guard activity is invisible in runs

- `rounds.jl:655-666`: `cap = 1.25 × max real-candidate count over three probed rounds of the initial state + 8`. The count
  is state-dependent (roots appear and vanish, the vintage kink mask `hv`, UC quantities), so sites can exceed it later in
  the run and are then held at t = 0 (invariant, but a silent loss of moves). `sample_jump` returns `over`, `do_round`
  (`rounds.jl:694-700`) discards it, and the drivers never report guard rates. Suggest: carry one counter per family in the
  loop state (a scalar sum of `over` per round), write it with the device profile at each checkpoint, and choose the margin
  generously once (changing `cap` changes shapes, i.e. recompiles).

### M4. Trace-time state lives in module globals mutated by every `plan`

- `UC_COLS`, `JUMP_CAP`, `JUMP_FAMILIES` (`rounds.jl:622-634`), `CELLINFO`, `CELL_BUCKETED` (`cells.jl:51,85`),
  `TDIR_TOPO/TDIR_SOURCE/TDIR_CHAINS` (`tdir.jl:20-28`), `SPARSE_SOURCE`, `BLOCK_GROUPS`, `PROFILES`, `DSHAPES`,
  `SCHED_FAMILIES` are read inside `build` at trace time (`ucon`, `jumpcap`, `jumpform`, `profiles`, `block_weights4`) and
  overwritten by every `plan`. Two plans in one process (tests that build several, `device_setup` followed by
  `replica_device_setup`, the profilers) change what a later compile bakes in; `CELL_BUCKETED` is latched to `true` and
  never reset, so a `:cell` plan after a `:cell1` plan gets per-level `Nc` and inconsistent column lengths in `host_table`.
- Fix: return them from `plan` (fields of `pl`) and hand them to `build` as static parts of the table NamedTuple
  (`tab.uc_cols`, `tab.cap`, `Val` flags), so a compiled program depends only on its arguments.

### M5. Snapshot and resume: small gaps

- Metropolis driver: `device_boundary!` is atomic and happens after the download, so host and device agree; but the
  tdir host stream `hr = MersenneTwister(cfg.seed + 101)` (`driver.jl:319`) is not restart-seeded (a resumed run replays the
  pre-resume direction sequence, contrary to the "never a replay" contract), and the schedule stream (H1) is the same
  after a resume. `HostAdapt`, the learned β, shapes and profile tables restart from scratch on resume (the CPU contract
  says so; worth a line in device.md §6.3).
- Replica driver: after `replica_weather!` with `ctlw.comp = false` the host `State.v` is stale until the next download;
  every consumer downloads first (probe, checkpoint, snapshot), so this is consistent today but easy to break — a comment
  at `replica_device_weather!` would help. A stop during the compile or `replica_init_sweeps` has no snapshot yet
  (restart from scratch, acceptable). The preemption grace must cover a host checkpoint that can contain a multi-minute
  host probe (M2).

### L1. Flat/improper decisions are not exactly line-invariant

- `terms.jl:599-603` (1e-12 relative to the potential at the ray ends) and `jump.jl:456-467` (16 ε × the sum of the
  columns' absolute slopes right of the current point, `s0abs`) decide "flat ray → t = 0" with a tolerance that depends on
  the current point; from another point of the same line the decision can differ. Measure-zero in practice and the CPU's
  80-step rule has the same character; note only.

### L2. `:slong` level move and move counts

- The level move (`lev`) still carries 2G generator columns and two commitment sets with zero slopes (`slong.jl:54-58,
  93-126`), flagged as moving by the static metadata (`mvk`), so they add primaries and root searches for nothing. A
  separate `:slevel` family with SoC terms only halves its cost.
- `SLONG_MOVES = 4` plateaus per unit and sweep against the CPU's 32 (`src/run.jl:28`), and L ≥ 2: a weaker seasonal
  kernel by design; state it in device.md §3.

### L3. Documents

- device.md §7: the row "replica–Langevin sampler … missing" (line 519) contradicts the "done" row (line 507) and §5.2/§6.2;
  §2 says "all FP64" while the F32 build exists; the slope-jump cap and the frozen schedule choices (H1) are not stated.

---

## 2. Fresh ideas for `:slong`

1. **Remove the varargs `cat`s (H2)** — the only structural blocker; afterwards the slope-jump program is already O(1) in T
   per op (≈ 20 T entries per site, one (np·T) residual evaluation, one payload sort of ≈ 5 T kept entries).
2. **Draw `(a, L)` inside the program** (H1): the `ce0/ce1/cr/wa/wb` tables become index arithmetic on an iota over the
   steps; host data shrinks to the unit list; fresh intervals every sweep; the same mechanism serves the block offsets and
   profile ids.
3. **Single-column slots need no slot machinery**: T − 2 of a slong site's balance slots are fed by exactly one SoC column
   (the end slots have the G compensators). For such a slot the residual crossing is a one-column problem on the column's
   own local points (the injection is piecewise linear between them), so the crossings and their jumps can be computed
   column-wise in `jump_parts` without `slotgather`, the (C, B, np, R) residual evaluation `r1(Pu)` and the (np, C, B, R)
   payload sort of the slot lists — only the two end slots keep the general path. This removes the largest per-site tensors
   of slong (np·T) and applies to every family whose slots have one column (`:soc`, the SoC slots of `:tsoc`).
4. **A per-site fused kernel** (KernelAbstractions, `raise=true`) that sorts the site's entries in shared memory and does the
   prefix sums and the inverse-CDF draw in one pass: replaces the global payload sort and the two-level scans (the largest
   kernels of the slope-jump form) for every family; ~7 k entries per slong site fit one block. A big effort; do 1–3 first.
5. A traced loop over step chunks is only needed for memory (not at 40 nodes); after 1 the tensor sizes are the limit, not
   the program size.

---

## 3. Checked and found fine

- **Breakpoints and roots.** Primaries: clamp points of moving arguments only, both pieces of kinked arguments, the kink
  t* = −Q/g, line knots ±U_j s and f = 0, SoC 0/1 clamps (`terms.jl:350-420`, `jump.jl:62-80`); secondary roots per
  quantity with the K-root guard (`rootsK`), 1 root for monotone quantities; the two-stage residual search with SoC
  injections (`terms.jl:515-544`). Slope-jump: balance jumps attached to column entries with the r ≥ 0 side convention;
  a touch-and-return at a point yields two crossings whose jumps are summed (`slot_cross`, `xs .== xi`); `jump_values1`
  integrates outward from c(0) = 0 with a round-off-aware flat test; the cap guard counts real candidates (line-invariant).
- **Ties and padding.** Stable payload sorts on both paths (Julia's default sort is stable since 1.9; `Reactant.Ops.sort(…;
  is_stable=true)`), zero-length segments carry no mass (`lmf = -Inf`), all entries at exactly 0 folded into one, padded
  candidates at 0; a site at a box bound (a real candidate at exactly 0) is counted. Empty rounds are dropped
  (`schedule`), padded sites have all slopes zero → improper → t = 0, their gathers hit column 1 and their scatters add
  exact zeros (`msk`).
- **Jacobian and compensation.** `logjac`/`jac_accept` = CPU `logjac` with `u3 ≥ exp(min(0, log J)) → t = 0`; `prorata4`
  (P, Q = r + Σ κ clamp ω (+ qadj), e, nm = nu − 1, act = g ≠ 0 ∧ nu > 0 ∧ |P| ≥ 1e-6) = `prorata!`; shares σ/κ = ω/P;
  `soc_qadj4` = `soc_qadj` (unclamped e-difference, clamped power), added only in the non-tracking rule; the linear rule
  β g/κ with the proxy class.
- **`:slong` semantics vs `prep_slong!`.** Raised instants a+1 … a+L through `wa/wb = raised(τ), raised(τ+1)`; +κ_s at a,
  −κ_s at b; the level move raises everything without compensation; cyclic `colτ`/`mod1` everywhere (also `:soc` at τ = 1,
  `nextτ`); UC follows at the end steps and the step before (as `:tpot`); the footprints are disjoint across nodes so the
  rounds are conflict-free; `κ_s ≥ κ_min` per chain and the 24 h selection; L on 2 … T−1 is a documented deviation.
- **Line draw arithmetic.** `trexp_d`/`trexp_end` = `trexp_offset` with the CPU's mirrored anchor for g < 0 (`cpumap`),
  `logseg_d` = `logseg`, masses anchored at the lower end, improper rays → t = 0 as the CPU.
- **Cost terms.** `cost_parts`/`eval_terms` term by term = `energy.jl` (`soc_cost`, the vintage merit order, `uc_cost`,
  link lossy injections, line hinges), `capacities` capex with γ_k and brownfield = `to_device`.
- **Replica step.** `lgv_w` = `langevin_update` (clip, reflection at κ_min and at the box from below, η = 0 frozen);
  `rep_sum`/`rep_bcast` keep k bitwise equal within a chain; `disp_follow` = `set_dispatchable_capacity!`;
  `kop_follow` = the kop curve (`P1 = (Q − Δinj)₊ + e`, σ = κω/P, |κ| ≥ κ_min, k₀ ≥ κ_min); `weather_update`
  (dnet = D0 − H(A c(k_r)), residual shift, `weather_comp` = `compensate_weather!`); the score terms and the
  physical-measure counts T(1 + uc), T; the gates/weights/probe on the host reuse the CPU functions on downloaded states.
- **RNG.** One (C, B, 3) draw per round from the `ReactantRNG` inside the traced loop (state carried across rounds and
  calls, confirmed by the tests), a separate Langevin stream (`seed + [0, 1]`), both re-seeded with the restart count;
  the mock path indexes `U[:, :, :, j + off]`; u3 is always drawn (the CPU draws it conditionally; same decision).
- **Reactant hygiene.** `@trace for` bodies index only with dynamic slices of the traced index; all static `for`s run over
  static counts (hinge knots, D lines, NR, nprop, carrier groups); loop-captured structs are type-parametric (`HingeT`,
  `Consts`, `KParams`, `LgvParams`, `RenParams`); flags as `Val`; `@allowscalar` for unit indices in `kmove`; `rs`
  materializes reshapes; `pin` barriers on t and root lists; `max(x, −x)` for `abs`; every device array is an argument;
  donated inputs are always reassigned from the outputs in both driver loops; tables re-uploaded with unchanged shapes.
- **Memory cap and conflicts.** `round_elements`/`DENSE_BUDGET` chunking; `check_conflicts` on every family's tables in the
  tests; the time classes separate the τ−1/τ couplings (checked for `:gen`, `:soc`, `:spair`, blocks' parity).
- **Zero-capacity units.** Masked by κ_min wherever a direction divides by κ (`pair`, `spair`, `cellpair`, compensators,
  `kop_follow`); elsewhere a zero-κ term only carries its box penalty (w = κ₀), harmless.
