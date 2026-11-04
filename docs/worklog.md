# Work log

What was done, by date, with the commits, and the open to-dos that have not been tackled yet.
Design-level open problems are in [system/roadmap.md](system/roadmap.md), experiments in
[experiments.md](experiments.md); this file is the running record of the work and its
concrete next steps. Update it at the end of every work session.

## Open to-dos

**Sampler design (decided direction, not yet built):**
- Capacities by a Langevin step on the sampled scores instead of the learned bias, weather from
  its prior by refresh ([system/ensemble.md](system/ensemble.md) §3.1,
  [system/roadmap.md](system/roadmap.md) §1). Tested (ledger E19, E20): renewables meet from
  different starts, but the dispatch lag biases the stationary capacities unless the step per
  operation sweep is ≲ 1e-4, and a weather refresh costs about 100 daily sweeps. The user
  approved the structure (2026-11-04). Replica prototype built and measured (ledger E22–E25,
  [replica_sampler.md](replica_sampler.md)): prior-reversible weather process exact and tested;
  continuous pCN too slow at an acceptable dispatch lag, so gated bursts; onwind and solar meet
  from opposite starts, offwind, gas and batteries not yet, gas overbuilt by the dispatch lag.
  Next: separate the weather lag from the capacity lag in the probe (probe only replicas long
  after their burst, or subtract a θ = 0 reference) and average several probes; choose the
  burst size and gate from a relaxation measurement per burst size; more replicas than active
  ones; regime windows of B ≥ 60 with their own gate; a better gas score; then move the
  sampler into `src/`. Then: rewrite ensemble.md §4 and sampler.md §6–7 for the new design, and retire the bias
  code (ĝ, ĥ, tables, lift, refits, trust region, post-correction, level-0 warmup,
  `device/src/bias_dev.jl`) and the capacity moves `kscale`/`kpeak`/`kop` as targets of the
  α = 0 density; record the learned bias in [retired.md](retired.md) with the reason (the
  precision it needs at r = 1e-3, ledger E16).
- An explicit, controllable tilt of the weather toward expensive years (stress tests): built in
  the replica prototype as replica weights exp(λΦ) in the capacity drift (E25); to decide: the
  functional Φ, and whether M = 4 replicas give usable weights (self-normalised, ESS).
- Rao-Blackwellized capacity scores (averages over the exact line conditionals) to cut the
  per-sweep score noise.
- Cross-node storage move: store at A, release at B, flows re-routed with the `xloc` templates
  (single-instant and `slong`-style interval forms), to avoid the node's own gas wall at the end
  steps; after the `storage_long` merge.
- Seasonal storage at fixed (k, x) after the long-range moves (ledger E21): the hydro seasonal
  shape, the large PHS units and opex still drift over 1000 sweeps; their moves stop where the
  node's own gas saturates at an end step. Next: a run of 3000 sweeps with the new kernel to see
  whether opex settles; a compensation that spreads over the node's gas and storage units or
  over several steps; the device port of `slong`/`slevel` and of the nominal storage
  requirement (`soc_qadj`) for `:tsoc` and `:sblk`.

**Decisions taken:**
- The learned bias is retired as part of the target (user, 2026-11-04): work proceeds on the
  replica–Langevin structure. A learned weather-response term may return only as an optional
  control variate for the between-year variance of the Langevin drift (no effect on the
  target; to be measured once the replica sampler is in `src/`).
- `storage_long` stays, on by default (user, 2026-11-04); merged. The `soc_qadj` correction
  changes `tsoc` and the storage plateaus, so the device comparisons of the share-preserving
  `:tsoc`/`:sblk` fail until the device adds it.

**Decisions waiting for the user:**
- z0 block moves are on by default (`fine_weather_blocks = 1`), a behaviour change for
  `chains40` and `complete40`.
- If any of the learned bias is kept: the bias safeguard choices (exact log J term, trust
  region on by default, θ_u₂ ≤ 0); `complete40.toml` has `warmup0` below `bias_start`.

**Device port:**
- Fine-level weather moves: not needed any more (the weather moves on the host by a
  prior-reversible pCN process). Device agent (2026-11-04) on, in order: GPU form choice and
  the `tpot` padding; restructuring so XLA does not materialize large intermediates
  (points × columns tensors, sort payloads, concatenations, transposes); a uniform Float32
  switch (FP64 stays the tested default; whether Float32 is a problem is tested empirically
  once everything is built); the learned and adaptive families. Hand-written fused kernels
  deferred (the estimates promise large gains; feasibility note requested).
- `:cellpairu1` compiles in about 290 s at 8 GB and stays out of the run set.
- Device form of the Langevin capacity step (replaces the Metropolis capacity pass and the
  planned `kop` port and per-carrier scales).
- Re-measure the compile times on a quiet machine (239 s / 149 s were measured under load).

**HPC (PIK cluster, [system/hpc.md](system/hpc.md); campaigns H0–H9 in the ledger):**
- `cache_dir` key or `ESM_CACHE_DIR`: one shared, content-addressed cache with atomic writes
  (now per run directory).
- Checkpoint and resume with SIGTERM handling in `run_chains` and `run_device`.
- `hpc/` scripts: environment preparation on a login node, the run wrapper, job templates, the
  run record and the runs index.
- Julia 1.12 on the cluster (the guide lists 1.11.7; the Manifests need 1.12); the Reactant CUDA
  artifact fetched on a login node without a GPU; the ERA5 folder's path (the user will look it
  up).
- Request `/p/projects` space; fill in the project and account names in hpc.md.

**Tests and validation:**
- Move-consistency suite `test/test_moves.jl` (agent, 2026-11-04): walk linearity, incremental =
  global, curve reversibility, caches, for every family; it replaces `test_exact.jl` as the gate
  (CLAUDE.md, testing.md, adding_a_move.md updated). No new statistical tests until the
  performance work is much further (user, 2026-11-04).
- Exactness tests do not detect small invariance violations (missed the `kpeak` width and the
  storage walk bug): longer reference runs (ledger P6).
- Sweeps to equilibrium against network size and resolution; 256-node ERA5 run (ledger P7, P8).

**Housekeeping:**
- `CLAUDE.md` still says many `Config` defaults are legacy values; check and update.
- `docs/reactant_design.md` mentions archived notes by bare file name in its text.
- Worktrees to remove when done: `.claude/worktrees/gpu-timing` (GPU timing copy), and the
  locked worktrees of finished agents whose branches are merged (review 72e12c2, storage
  09790f6; their outputs are copied to `results/review/` and `results/smix/`).
- Julia MCP sessions of agents restart at nice 0 with 16 threads unless every call passes the
  nice wrapper as `julia_cmd`; a CLAUDE.md note or a wrapper default would prevent it.

## Log

### 2026-11-04 (seasonal storage)

- Long-range storage moves `slong` (plateau over any cyclic interval, energy between two
  arbitrary steps) and `slevel` (shift of a whole trajectory), `src/storage_long.jl`, in every
  kernel (`storage_long`, `storage_long_moves`, `storage_long_hours`); the nominal storage
  requirement `soc_qadj` in the share-preserving compensation of `tsoc`, the storage plateaus
  and `slong` (the curve was not a coordinate line before); tests in `test_local.jl`,
  `test_model.jl`, `test_exact.jl` (ledger E21).

### 2026-11-04 (afternoon, replica prototype)

- Replica Langevin sampler prototype (`scripts/replicas/`): M weather replicas sharing k,
  prior-reversible weather process (pCN in the non-centred innovations, regime window draws),
  compensated dispatch, gated weather bursts, lag probe with per-carrier step adaptation, stress
  tilt; prior-invariance argument and tests, dispatch-lag scan, low/high run on chains40, tilt
  demonstration (ledger E22–E25, [replica_sampler.md](replica_sampler.md)).

### 2026-11-04 (overnight session and morning)

**Device port and performance**
- Reactant profiling tools (`device/scripts/profile_device.jl`, `profile_families.jl`); a device
  call is 98 % operation sweeps.
- Structure-specialised plain families `:gen0`, `:pair0`, degree-bucketed `:theta` (7916ee0):
  plain sweep 2.26 → 1.00 s on the laptop GPU.
- Term axis of the dense sampler (c11c906): fast set 1.00 → 0.57 s per sweep, `spair`
  2.55 → 1.00 s; end-to-end call of 10 sweeps 22.6 → 6.2 s.
- Memory caps per round (`DENSE_BUDGET`) and storage blocks and `:spair0` in the run set
  (4fd3d2c): the full run set fits the 4 GB card, 54 s per sweep.
- Slope-jump form of the line sampler (ad73f13): full run set 54 → 23 s per sweep; `tpot`
  17.7 → 8.3 s, `sblk16` 15.2 → 1.6 s; the dense form's storage-kink flaw fixed too.
- First compile is mostly one-time Julia JIT (1222643). Numbers in
  [system/performance.md](system/performance.md) §7.

**Correctness**
- CPU line walk wrong on storage lines (balance roots misplaced across storage kinks, up to
  262 nats on 13–18 % of lines): fixed, with a test (71f5720). Found by the device port.
- `kpeak` slice width not reversible: fixed (ee53043).

**Mixing**
- Diagnosis of the CPU sampler at 40 nodes ([mixing_diagnosis.md](mixing_diagnosis.md)): the
  screening was a transient; capacity moves confined; z0 means and regimes slow.
- Storage multiscale sweep in the `local` kernel (ee53043); `kop` capacity move, default off
  (8c67ff2); joint low-frequency z0 block moves, default on, window-capped (ce0ccf7).
- Operation mixing at fixed (k, x): scores fast once equilibrated, seasonal storage slow
  (0493e77, ledger E17).

**Bias and the α = 1 ensemble**
- Level-general bias at the fine level (f46c088) and safeguards (36c5026); bounded runs, weight
  ESS 1, chains far apart (ledger E14, E15).
- Fresh-eye review: the learned bias cannot reach the needed precision at r = 1e-3; Langevin
  capacity step recommended (ledger E16, branch 72e12c2).
- Design principle recorded: no exact density for the decisions, the weather prior fixed
  (ensemble.md §3.1, 50afdd6).

**Docs**
- Box fix: `kscale`/`kpeak` scale utilizations against the clamped capacity, so capacities
  above their box return to it (74f9bb0, merged 489f806; ledger E18).
- Langevin capacity test and weather-refresh relaxation (ledger E19, E20; scripts in
  `scripts/langevin/`).
- Superseded notes archived in `docs/archive/` (5037234); the experiment ledger (e7bb129); this
  work log; the list of retired approaches (`docs/retired.md`).

### Before 2026-11-04

Earlier work (formulations, flow and adaptive kernels, the legacy removal, the first device
phases) is recorded in the archived notes ([archive/README.md](archive/README.md)) and in the
git history up to bbbb97c (merge of the legacy removal).
