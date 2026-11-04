# Work log

What was done, by date, with the commits, and the open to-dos that have not been tackled yet.
Design-level open problems are in [system/roadmap.md](system/roadmap.md), experiments in
[experiments.md](experiments.md) and campaigns in [campaigns.md](campaigns.md); this file is the running record of the work and its
concrete next steps. Update it at the end of every work session.

## Open to-dos

**Sampler design (decided direction, not yet built):**
- Replica–Langevin sampler in `src/replicas.jl` (`sampler = "replicas"`, sampler.md §11,
  ensemble.md §4), with deterministic tests (`test/test_replicas.jl`, `:wproc` in
  `test_moves.jl`). Next: campaign H4 on the cluster (prepared: low/high starts, 40 nodes daily
  M = 8 and 4, 6-hourly M = 8; to be submitted on the rebuilt `model.h5` with the
  wind-calibration fix, after which the prototype's carrier results, offwind in particular, are
  rechecked); from its results the burst size and gate defaults, then 256 nodes
  daily and the comparison with H2. Open: gas overbuilt in the prototype (a gas step of its own or
  a Rao–Blackwellised score), the size of the gate's weather dependence, seasonal storage after a
  burst. Then retire the bias code (ĝ, ĥ, tables, lift, refits, trust region, post-correction,
  level-0 warmup, `device/src/bias_dev.jl`) and `kscale`/`kpeak`/`kop` as moves of the α = 0
  density, and make `sampler = "replicas"` the default.
- An explicit, controllable tilt of the weather toward expensive years (stress tests): built in
  `src/replicas.jl` as replica weights exp(λΦ) in the capacity drift (`tilt_lambda`, Φ the
  standardised annual mean residual load); to decide: the functional Φ, and whether M = 8
  replicas give usable weights (self-normalised, ESS); campaign H9.
- Rao-Blackwellized capacity scores (averages over the exact line conditionals) to cut the
  per-sweep score noise.
- Seasonal storage at fixed (k, x) after the long-range moves and the storage pairs (ledger
  E21, E27): per-unit hydro and PHS shapes relax about twice as fast with `sxpair`/`sxlong`
  (equal CPU, chain 4), but the large hydro units and opex still drift over 3000 sweeps; the
  walls are now the reservoirs' zero discharge at the end steps. Next: campaign H10, redesigned to make its start states on the cluster; a direction
  whose power change is spread over several steps where the reservoir discharges (selection
  must not depend on v); the device port of `slong`/`slevel`, of the storage pairs and of the
  nominal storage requirement (`soc_qadj`) for `:tsoc` and `:sblk`.

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
- Plan (user, 2026-11-04): implement all missing families on plain arrays against the CPU
  (`slong`/`slevel`, storage pairs, Haar angle atoms and `tcut`/`acut`, local re-routing
  templates, block aggregates, `adshape`; not the learned `kpeak`, weather moves, fused kernels,
  multi-GPU), then one Reactant compile pass (CPU backend) over all device code, then a review
  of the existing and new Reactant code by a Fable agent; GPU timings from H5 on the cluster.
  The precompile workload stays only if it is a clean win. (Of the port agent's 6.2 h, 3.3 h were
  waits on background compile and timing scripts; few compiles broke.)
- Fine-level weather moves: not needed any more (the weather moves on the host by a
  prior-reversible pCN process). Device agent (2026-11-04) on, in order: GPU form choice and
  the `tpot` padding; restructuring so XLA does not materialize large intermediates
  (points × columns tensors, sort payloads, concatenations, transposes); a uniform Float32
  switch (FP64 stays the tested default; whether Float32 is a problem is tested empirically
  once everything is built); the learned and adaptive families. Hand-written fused kernels
  deferred (the estimates promise large gains; feasibility note requested).
- `:cellpairu1` compiles in about 290 s at 8 GB and stays out of the run set.
- Device form of the replica sampler: the CPU reference is `src/replicas.jl` (weather process
  with explicit noise, `langevin_update`, `follow_capacities!`; architecture.md §2); it replaces
  the Metropolis capacity pass and the planned `kop` port and per-carrier scales.
- Re-measure the compile times on a quiet machine (239 s / 149 s were measured under load).

**HPC (PIK cluster, [system/hpc.md](system/hpc.md); campaigns H0–H9 in the ledger):**
- `cache_dir` key or `ESM_CACHE_DIR`: one shared, content-addressed cache with atomic writes
  (now per run directory).
- Checkpoint and resume with SIGTERM handling in `run_chains` and `run_device`.
- `hpc/` scripts built (env, prepare, campaigns, run wrapper, collection into git, job
  templates, H0 checks; wrapper mechanics tested locally); cluster values filled in (user
  hellmann, project directory coen, Slurm account icone, julia/1.12.2, CUDA 13 → Reactant 13.1, ERA5 at
  /p/projects/climate_data_central/reanalysis/ERA5, clone at /p/projects/coen/esm/ESM-Sampling).
  Model files built on the cluster (`hpc/prepare_data.sh`, `hpc/jobs/data.sbatch`), no manual
  copy. H0 done except the SIGTERM and snapshot-resume tests (a test bug, fixed in f89c77e):
  rerun as H0b. Next on the cluster: the data jobs (2902883 small, 2902884 pypsa256), H0b, H10;
  then H2 with `scripts/lp_reference.jl` (MGA alternatives still to be scripted).
- Device driver checkpoint (after the device agent's work).

**Data:**
- The POWER model files carry the old wind calibration (scale 4, deep in cut-out, at 9 of 19
  offwind nodes of `model.h5`, 49 of 64 of `model256.h5`; [system/data_sources.md](system/data_sources.md)
  §2). Fixed in `prepare_data.py` (`smallest_root`, shared with the ERA5 build); a fixed
  build of `model.h5` changes exactly the 10 affected columns; the fixed builds of all three
  files (in the session scratchpad) have no scale at 4, every wind series rising with its wind
  speed, and every generator within 0.005 of its target (`model256.h5` offwind mean 0.436,
  ERA5 0.418). To do: replace the local files
  once the running agents are done (they read `data/` through their worktree links), and
  rebuild on the cluster
  (hpc.md §8.3, D1 and D2). Then recheck the physical conclusions of the experiments (the
  ledger's note), in particular offwind in E22–E25.

**Tests and validation:**
- Move-consistency suite `test/test_moves.jl` built and merged (2026-11-04); it replaces
  `test_exact.jl` as the gate (CLAUDE.md, testing.md, adding_a_move.md). No new statistical tests until the
  performance work is much further (user, 2026-11-04).
- The event walk of requirement-tracking moves places kinks about 1e-7 of the segment off (up
  to ~1e-5 nats between walk points on weather moves; `test_moves.jl` reports it as broken);
  an iterated secant made other lines worse. Needs a root placement that is exact where a
  quantity bends at its own root.
- `test_moves.jl` is order-dependent: the families run in sequence on one shared state, and
  entries placed before others changed their outcome (with the storage pairs early, `kopC`'s
  `lo` check missed by 2.1e-12 against an absolute 1e-12, and `wz0`'s sweep accepted no move).
  The storage pairs now run last; better: a fresh copy of the setup state per family, and
  relative tolerances.
- Exactness tests do not detect small invariance violations (missed the `kpeak` width and the
  storage walk bug): longer reference runs (ledger P6).
- Sweeps to equilibrium against network size and resolution; 256-node ERA5 run (ledger P7, P8).

**Housekeeping:**
- `CLAUDE.md` still says many `Config` defaults are legacy values; check and update.
- `docs/reactant_design.md` mentions archived notes by bare file name in its text.
- Worktrees to remove when done: `.claude/worktrees/gpu-timing` (GPU timing copy), and the
  locked worktrees of finished agents whose branches are merged (review 72e12c2, storage
  09790f6; their outputs are copied to `results/review/` and `results/smix/`; storage pairs
  a2ce14d, whose raw runs are in its `results/sxmix/`).
- Julia MCP sessions of agents restart at nice 0 with 16 threads unless every call passes the
  nice wrapper as `julia_cmd`; a CLAUDE.md note or a wrapper default would prevent it.

## Log

### 2026-11-04 (time limits)

- The time-limit signal (`--signal=B:USR1@900`) went to the templates' batch shell, which ran
  `run.sh` as a child: no stop file, no snapshot. The templates now `exec` `run.sh`, which
  records the signal (`TERM` or `USR1`). A short-lived automatic resubmission past the time
  limit was removed again: jobs are never chained past a QOS limit (user); every job asks for a
  `--time` covering its estimate (standby up to 7 days), H4 for 2 days (CLAUDE.md, hpc.md §5).
  Simulated locally with Slurm stubs.
  CLAUDE.md: the sampler described as weather replicas with a Langevin capacity step.

### 2026-11-04 (ERA5 built; wind calibration fixed)

- `model256_era5.h5` built on the cluster (job 2903048, 8 minutes, 2003, 2010, 2013, 2017;
  every generator within 0.005 of its 2013 target); its comparison with `model256.h5` showed
  POWER offwind anti-correlated with ERA5 (−0.76). Cause, checked node by node against the POWER
  points: `calibrate` returned the scale 4 whenever the window mean at 4 lay below the target,
  deep in cut-out (`model.h5`: 9 of 19 offwind, 1 of 40 onwind; `model76.h5`: 10 of 51, 2 of 76;
  `model256.h5`: 49 of 64, 33 of 256; every such node correlates negatively with its wind
  speed). `prepare_data.py` now takes the smallest root (`smallest_root`, also used by
  `prepare_era5.py`, identical results there) and reports target misses; a rebuild of
  `model.h5` changes exactly the 10 affected columns.
- `hpc.md` §8: the run book (what runs in which order). Rule (user): campaigns are
  self-contained, every input made on the cluster by versioned scripts, no files copied
  between the machines (CLAUDE.md, hpc.md §6); H10 is redesigned accordingly.

### 2026-11-04 (storage pairs across nodes)

- `sxpair` (single instant, local-sweep family, `storage_xpair` 0.1) and `sxlong` (cyclic
  interval in the long-range storage sweep, `storage_xlong` 8 per unit): a second storage unit at
  the same or a partner node absorbs the end-step imbalance, flows re-routed on 1-hop ball
  templates (`storage_xpair_hops`, `storage_xpair_ball`), leak and reservoir losses compensated
  share-preservingly (`sx_qloss`), interval pairs in rounds of disjoint footprints
  (`src/storage_cross.jl`); entries in `test_moves.jl`, checks in `test_local.jl`; E21 protocol
  on chain 4 for 3000 sweeps (ledger E27); the long runs as campaign H10 (`campaigns/H10/`,
  `scripts/mixing/opmix_hpc.jl`, smoke-tested locally through `hpc/run.sh` with a stop and a
  resume).

### 2026-11-04 (replica–Langevin sampler in src/)

Commits d833b50, f7d58c3, b7dcc27, a2e4c84, 515b1a1 (branch of the replica agent).

- `src/replicas.jl`: the replica–Langevin sampler as the system's CPU sampler (`run_replicas`,
  `sampler = "replicas"`): the prior-reversible weather process with explicit-noise building
  blocks (pCN in the innovations, regime windows, FFBS option), the capacity score, the Langevin
  update in w with its reflections, the dispatch-following maps, the gate on a measured
  relaxation indicator (bounded by `replica_gate_max`), a lag probe of relaxed replicas averaged
  over `replica_probe_avg` probes, 8 replicas by default, the stress tilt (λ = 0 bitwise
  untilted), snapshots and resume, records and `summary_replicas.txt`. `operation_sweep!`
  factored out of `fine_local_step` (bitwise the same steps). Config keys `sampler`,
  `replica_*`, `langevin_*`, `tilt_*`.
- Tests: `test/test_replicas.jl` (weather-process identities, score against finite differences,
  Langevin update, following maps, gate/weights/adaptation, one step at λ = 0 and 2, snapshots);
  the `:wproc` entry of `test_moves.jl`.
- Campaign H4 prepared (`campaigns/H4/`), smoke-tested through `hpc/run.sh` in a simulated
  layout (preempted twice, resumed from the snapshot).
- Docs: ensemble.md §4 rewritten for the sampled ensemble (the bias in a retired note,
  archived to `archive/learned_bias.md`, entry in retired.md), sampler.md §11, architecture,
  config, data, testing §6, moves/weather.md, roadmap §1.

### 2026-11-04 (deterministic move checks)

- `test/test_moves.jl`: one table-driven harness of single-move exactness checks for every
  family (walk linearity, incremental = global, reversibility with the same choices, excess,
  round trip, shifted density and inverse Jacobians, slice width and bound, Metropolis
  negation, caches after a sweep), on `haar_poc` level 1 and a 5-node all-terms sub-network;
  about 10 s after the setups (ledger E26, testing.md §5). Test hooks: `probe`/`fix` and more
  `rec` fields in `kscale_sweep!`/`kpeak_sweep!`, `weather_coord_prepare!`/`_commit!`,
  `prep_adshape!`.
- Found and fixed: `kop` without the commitment walls of its compensators (`kop_uc`); the walk's
  storage mask with commitment walls before the storage secondaries (`MoveWS.soff`). Open: the
  event walk's root placement (up to ~1e-5 nats, sampler.md §3.3).

### 2026-11-04 (HPC: first data jobs fixed)

- `data.sbatch small` (job 2902805) failed in `prepare_data.py`: `write_completion` checked
  `col in f["generators"]` before a fresh file had that group (only `--augment` worked). It
  now checks the full path (`"generators/k0" in f`). The failed run's 18 KB partial
  `model.h5` is overwritten on rerun.
- `data.sbatch pypsa256` (job 2902812) failed in `uv pip install` of the PyPSA-Eur
  requirements: the user's shell exports `UV_INDEX` (a PyTorch wheel index), Slurm passes it
  into the job, and uv used that index first (`certifi==2026.10.23` not found). `hpc/env.sh`
  now unsets the inherited uv/pip index variables.
- Fixes ba6caee (merged a8673c8); resubmitted as jobs 2902883 (`small`) and 2902884
  (`pypsa256`).

### 2026-11-04 (HPC: shared cache and state snapshots)

- CPU driver for the cluster (`src/checkpoint.jl`, [system/hpc.md](system/hpc.md) §4, §7):
  shared cache (`cache_dir`/`ESM_CACHE_DIR`, read-only mirror `cache_mirror`/`ESM_CACHE_MIRROR`,
  atomic writes of every entry); HDF5 state snapshots in `run_chains` (`checkpoint`,
  `checkpoint_minutes`, `stop_file`; resume from the saved states with re-seeded streams;
  exit code 75 after a stop; `ESM_IGNORE_SIGTERM` in `scripts/run.jl`); `ESM_LP_THREADS` for
  HiGHS; tests in `test/test_hpc.jl`. The device driver's snapshot is still missing.

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
