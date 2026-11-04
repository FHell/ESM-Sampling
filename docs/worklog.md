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
  operation sweep is ≲ 1e-4, and a weather refresh costs about 100 daily sweeps. Replica
  prototype built and measured (ledger E21–E24, [replica_sampler.md](replica_sampler.md)):
  prior-reversible weather process exact and tested; continuous pCN too slow at an acceptable
  dispatch lag, so gated bursts; next: separate the weather lag from the capacity lag in the
  probe (probe only replicas long after their burst, or subtract a θ = 0 reference), choose
  the burst size and gate from a relaxation measurement per burst size, regime windows of
  B ≥ 60 with their own gate, then move the sampler into `src/`. Then: rewrite ensemble.md §4 and sampler.md §6–7 for the new design, and retire the bias
  code (ĝ, ĥ, tables, lift, refits, trust region, post-correction, level-0 warmup,
  `device/src/bias_dev.jl`) and the capacity moves `kscale`/`kpeak`/`kop` as targets of the
  α = 0 density; record the learned bias in [retired.md](retired.md) with the reason (the
  precision it needs at r = 1e-3, ledger E16).
- An explicit, controllable tilt of the weather toward expensive years (stress tests): built in
  the replica prototype as replica weights exp(λΦ) in the capacity drift (E24); to decide: the
  functional Φ, and whether M = 4 replicas give usable weights (self-normalised, ESS).
- Rao-Blackwellized capacity scores (averages over the exact line conditionals) to cut the
  per-sweep score noise.
- Seasonal storage relaxes slowly at fixed (k, x) (τ ≳ 100 sweeps for hydro, PHS, H₂; ledger
  E17): a longer measurement (3000–5000 sweeps), then longer storage blocks or a seasonal
  plateau move per region in the CPU kernel.

**Decisions waiting for the user:**
- z0 block moves are on by default (`fine_weather_blocks = 1`), a behaviour change for
  `chains40` and `complete40`.
- If any of the learned bias is kept: the bias safeguard choices (exact log J term, trust
  region on by default, θ_u₂ ≤ 0); `complete40.toml` has `warmup0` below `bias_start`.
- Next device task: the GPU form choice (`tsoc` back to the dense form, about 1 s per sweep;
  the padded commitment columns and the payload sort in `tpot`, 20–30 %), or the port of the
  fine-level weather moves (uncompensated coordinate moves first).

**Device port:**
- Fine-level weather moves (missing); later the weather refresh in place of them.
- `:cellpairu1` compiles in about 290 s at 8 GB and stays out of the run set.
- Device form of the Langevin capacity step (replaces the Metropolis capacity pass and the
  planned `kop` port and per-carrier scales).
- Re-measure the compile times on a quiet machine (239 s / 149 s were measured under load).

**Tests and validation:**
- Exactness tests do not detect small invariance violations (missed the `kpeak` width and the
  storage walk bug): longer reference runs (ledger P6).
- Sweeps to equilibrium against network size and resolution; 256-node ERA5 run (ledger P7, P8).

**Housekeeping:**
- `CLAUDE.md` still says many `Config` defaults are legacy values; check and update.
- `docs/reactant_design.md` mentions archived notes by bare file name in its text.
- Worktrees to remove when done: `.claude/worktrees/gpu-timing`,
  `.claude/worktrees/agent-af45abb1d0d9c1d90` (review scripts, branch 72e12c2, to merge or drop).
- Julia MCP sessions of agents restart at nice 0 with 16 threads unless every call passes the
  nice wrapper as `julia_cmd`; a CLAUDE.md note or a wrapper default would prevent it.

## Log

### 2026-11-04 (afternoon, replica prototype)

- Replica Langevin sampler prototype (`scripts/replicas/`): M weather replicas sharing k,
  prior-reversible weather process (pCN in the non-centred innovations, regime window draws),
  compensated dispatch, gated weather bursts, lag probe with per-carrier step adaptation, stress
  tilt; prior-invariance argument and tests, dispatch-lag scan, low/high run on chains40, tilt
  demonstration (ledger E21–E24, [replica_sampler.md](replica_sampler.md)).

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
