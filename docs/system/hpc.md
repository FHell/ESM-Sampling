# Runs on the PIK HPC cluster

How the sampler is run on the PIK cluster (HPC2024, Slurm): the environment, where code, data,
caches and results live, the job types, checkpointing, how every run is recorded, and how
results come back. The scripts are in `hpc/`; the campaigns are rows H0–H9 of the experiment
ledger ([../experiments.md](../experiments.md)). User `hellmann`, project directory `coen`
(`/p/projects/coen`), Slurm account `icone`.

> **Status: partial.** The scripts exist, and the CPU driver's shared cache (§4) and state
> snapshots with resume (§7) are built; the device driver has no snapshots yet (§7). Nothing
> has run on the cluster yet; campaign H0 is the first check.

## 1. The cluster in brief

| resource | what | limits that matter here |
|---|---|---|
| login nodes `login[01-03]` | 64 cores (EPYC 9354, Zen 4), 768 GB, **internet** | interactive work, compiling, short tests; QOS `io` (partition `io`) runs batch jobs there: 4 cores, 3 days, 5 concurrent jobs |
| CPU nodes (`standard`, 180; `priority`, 60) | 128 cores (2 × EPYC 9554, Zen 4), 768 GB, no internet | QOS `short` 24 h (default), `medium` 7 days, `long` 30 days but at most 32 cores per job; `priority` 24 h, starts within minutes, at most 64 cores and 5 jobs; `standby` preemptible |
| GPU nodes (`gpu`, 12) | 4 × H100 80 GB, 128 cores, 1.5 TB, `/localscratch` NVMe ~3 TB | QOS `gpushort` 24 h, `gpumedium` 7 days, `gpulong` 30 days, `gpupreempt` preemptible; `--gres=gpu:N` |
| `$HOME` | backed up, small quota | no job I/O |
| `/p/projects/coen` | backed up | the permanent home of this project on the cluster |
| `/p/tmp/hellmann` | fast parallel FS, not backed up, **inactive files are deleted** | job working directories |
| `/localscratch` (GPU nodes) | per job, cleared at job end | temporary and compile files |
| `/p/projects/climate_data_central/reanalysis/ERA5` | shared, read-only | ERA5 for the weather inputs (§4) |

Fair share charges *allocated* cores × wall time (memory is free, default ~6 GB per core) with
a 7-day half-life. Preemption (standby, gpupreempt) and cancellation send SIGTERM, then SIGKILL
5 minutes later. Licensed solvers are not available: every LP is solved with HiGHS (the Julia
code uses JuMP + HiGHS; PyPSA-Eur's `config.esm.yaml` sets `solving: solver: highs`). Software:
`module load julia/1.12.2` (the Manifests' version), `cuda/13.4.3` (not loaded: Reactant
brings its own CUDA runtime), a local `uv`.

## 2. Storage layout

```
/p/projects/coen/esm/                       permanent, backed up        ($ESM_PROJ)
├── ESM-Sampling/                           the git clone (main): code, and the committed results  ($ESM_CODE)
│   ├── data -> /p/projects/coen/esm/data   (link made by prepare_env.sh; git-ignored)
│   └── campaigns/⟨id⟩/                     committed: campaign.toml, runs.jsonl, runs/⟨run id⟩/ (records, small results)
├── campaigns/⟨id⟩/
│   ├── code/                               git worktree of the clone at the campaign's pinned commit
│   └── runs.jsonl                          the campaign's runs index
├── env/
│   ├── julia_depot/                        JULIA_DEPOT_PATH (packages, artifacts incl. XLA/CUDA, precompile caches)
│   ├── uv_cache/  venv/  xdg_cache/  mpl/  Python environment and tool caches
│   └── ENVIRONMENT.toml                    one entry per prepared code directory (§3)
├── data/{raw,processed}/                   inputs, read-only for jobs
├── cache/                                  durable mirror of the computed caches ($ESM_CACHE_MIRROR)
├── archive/⟨id⟩/⟨run id⟩/                  finished runs without checkpoints
└── ledger/runs.jsonl                       runs index of all campaigns

/p/tmp/hellmann/esm/                        fast, purged when inactive  ($ESM_SCRATCH)
├── cache/                                  the cache jobs read and write ($ESM_CACHE_DIR)
├── runs/⟨id⟩/⟨run id⟩/                     working directory of a run: configs, log, outputs, checkpoint, run.toml
└── logs/                                   Slurm output files
```

- **Code, environment and data never live on `/p/tmp`:** its automatic deletion would remove a
  depot or a data file, and offline jobs would fail.
- **One pinned code version per campaign:** `hpc/new_campaign.sh` makes a worktree of the clone
  at a commit; every run of the campaign uses it, and `hpc/run.sh` refuses to run if it has
  uncommitted changes. The clone itself can move on (and receives the results).
- **Caches are content-addressed,** so all runs share one cache; a purged entry is recomputed,
  not wrong. The mirror on `/p/projects` is the durable fallback.

## 3. Environment, prepared on a login node

Compute nodes have no internet, so everything is downloaded, instantiated and precompiled on a
login node, and jobs run offline. `hpc/env.sh` holds every path and variable (each can be
overridden before sourcing it); `hpc/prepare_env.sh [<code dir>]` prepares a code directory:

1. clones the repository if the clone does not exist;
2. links `data/` to the project's data;
3. writes `device/LocalPreferences.toml` with `[Reactant_jll] gpu = "cuda"`, `gpu_version =
   "13.1"`: Reactant picks its XLA runtime (CPU only or CUDA) when it is installed, from the GPU
   it detects, and a login node has none (`REACTANT_GPU`, `REACTANT_GPU_VERSION` in `env.sh`;
   12.9 for CUDA 12 drivers);
4. `Pkg.instantiate()` and `Pkg.precompile()` for both projects, with
   `JULIA_CPU_TARGET="generic;znver4,clone_all"` (login and compute nodes are Zen 4) and 8
   precompile tasks;
5. `uv sync --frozen` for the Python environment;
6. appends the record to `env/ENVIRONMENT.toml`: commit, Julia version and module, Manifest
   and lock hashes, the Reactant variant, date, host.

`hpc/jobs/prepare.sbatch` runs the same as a batch job in the `io` QOS. Jobs set
`JULIA_PKG_OFFLINE=true` and `UV_OFFLINE=1`.

## 4. Inputs and caches

- **Model files** (`data/processed/model*.h5`, 0.1–0.6 GB each) are built on the cluster from
  public sources, so their provenance is a commit plus recorded inputs
  ([data_sources.md](data_sources.md) §5): `hpc/prepare_data.sh small` (the example network,
  OPSD data and NASA POWER points: `model.h5`, `model76.h5`) and `hpc/prepare_data.sh pypsa256`
  (PyPSA-Eur at the pinned revision in `configs/pypsa-eur/REVISION`, its Python environment
  recreated from `configs/pypsa-eur/requirements.txt`, the snakemake target
  `composed_2050.nc` of `config.esm.yaml`, then `model256.h5`). Both need internet and run in
  the `io` QOS (`hpc/jobs/data.sbatch`, requeued if interrupted; downloads, POWER points and
  snakemake outputs are kept, so a rerun continues). Every built file is recorded in
  `data/processed/PROVENANCE.toml` (path, hash, size, commit, command, date, host).
- **ERA5** is read in place from `/p/projects/climate_data_central/reanalysis/ERA5` (`ESM_ERA5`),
  which compute nodes see directly, so no copy and no internet are needed:
  `hpc/prepare_data.sh era5 [years]` (`hpc/jobs/era5.sbatch`, standby on a compute node, after
  `pypsa256`, whose PyPSA-Eur resources it needs) cuts the monthly global files
  (`<variable>/<variable>_YYYYMM.nc`: `2m_temperature`, `10m_u_component_of_wind`,
  `10m_v_component_of_wind`, `surface_solar_radiation_downwards`) to the European box
  (`scripts/era5_subset.py`, into `data/raw/era5/europe/`), one file at a time with
  constructed paths (never a listing, glob or recursive scan of the folder), keeping finished
  months, then runs `scripts/prepare_era5.py` for `model256_era5.h5`. Default years 2003,
  2010, 2013, 2017; 2013, the calibration window, must be among them.
- **Weather-model and LP caches** (`weather_<hash>.jls`, `lp_<hash>.h5`, content-hashed
  keys; [architecture.md](architecture.md) §5). Their directory is the environment variable
  `ESM_CACHE_DIR` if set and non-empty, else the key `cache_dir`, else `dirname(out)/cache`;
  jobs set `ESM_CACHE_DIR=/p/tmp/hellmann/esm/cache`, so all runs share it. The read-only
  mirror is `ESM_CACHE_MIRROR` (or the key `cache_mirror`), set to
  `/p/projects/coen/esm/cache`: an entry missing in the cache but present in the mirror
  is copied into the cache and used, instead of being recomputed. Every entry is written
  under a temporary name in the cache directory (`.<name>.tmp-<pid>-<random>`) and renamed
  into place, so concurrent array tasks see a complete entry or none; a task that is killed
  while writing leaves only its temporary file, which is never read (and can be deleted). Two
  tasks that miss the same entry at once both compute it and the later rename wins (the
  entries are identical). An unreadable entry is recomputed and replaced, with a warning.
  The wrapper sets both in the run's config as well. At the end of a campaign its cache
  entries are copied to the mirror (`rsync -a /p/tmp/hellmann/esm/cache/ /p/projects/coen/esm/cache/`).
- **Compilation.** A Reactant process spends 2–4 minutes tracing and compiling before the first
  sweep, so GPU jobs are long-lived processes that batch many chains and replicas.

## 5. Job types and the wrapper

**Preemptible by default.** Jobs run in `standby` (CPU) or `gpupreempt` (GPU) whenever possible,
with `--requeue`: a preempted job is requeued automatically, keeps its run id and resumes from
the latest state snapshot (§7); a bit of lost work does not matter. Every job is constructed so
that it can be restarted this way. Guaranteed-runtime templates exist for the cases where
preemption would hurt.

**Shared nodes.** Jobs take a part of a node, not the whole node. Thread counts therefore follow
the allocation (`hpc/run.sh`), because libraries that size themselves from the node's 128 cores
would oversubscribe the cores the job was given: Julia threads = `--cpus-per-task`
(`ESM_JULIA_THREADS` to change); BLAS (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) and OpenMP
single-threaded unless a job sets `ESM_BLAS_THREADS` / `ESM_OMP_THREADS` (their parallel
regions would multiply with Julia's threads); HiGHS `threads` = `--cpus-per-task`
(`ESM_LP_THREADS`). Memory: the default of ~6 GB per core, or `--mem-per-cpu` / `--mem` on the
command line. The run record notes the thread plan.

| template (`hpc/jobs/`) | resources | QOS | used for |
|---|---|---|---|
| `prepare.sbatch` | 4 cores on a login node | `io` | environment preparation (§3) |
| `data.sbatch` | 4 cores on a login node, 3 days, requeued | `io` | building the model files that need downloads (§4) |
| `era5.sbatch` | 4 cores, 48 GB, 24 h, requeued | `standby` | `model256_era5.h5` from the cluster's ERA5 folder (§4) |
| `h0_cpu.sbatch`, `h0_gpu.sbatch` | 8 cores; 1 H100 + 16 cores | `priority`; `gpushort` | campaign H0: offline load, deterministic tests, a smoke run; Reactant on the H100, a family profile |
| `cpu.sbatch` | 8 cores (`-c` to change), 24 h, requeued | `standby` | CPU sampler runs (the default) |
| `cpu_array.sbatch` | one run per line of a list file, requeued | `standby` | many configurations or seeds (`--array=1-n%k`) |
| `build.sbatch` | 16 cores, 12 h, requeued | `standby` | LP solves (e.g. `scripts/lp_reference.jl`), weather-model fits, data preparation, PyPSA-Eur solves; a preempted build starts over, finished cache entries are kept |
| `gpu.sbatch` | 1 H100, 16 cores, 160 GB, 24 h, requeued | `gpupreempt` | device runs |
| `cpu_short.sbatch`, `gpu_short.sbatch` | as above, not preempted | `short`, `gpushort` | when preemption would hurt |
| `analysis.sbatch` | 4 cores, 2 h | `priority` | diagnostics, summaries |

Every template calls the wrapper of the campaign's pinned code:

```bash
sbatch hpc/jobs/cpu.sbatch <campaign> <config> [cpu | device | script <path> [args...]]
```

`hpc/run.sh` (paths in the config are relative to the campaign code):

1. checks that the campaign code has no uncommitted changes to tracked files (untracked files
   are fine), loads `julia/1.12.2`, sets the offline variables and
   `ESM_IGNORE_SIGTERM=1`;
2. makes the run directory `runs/⟨id⟩/⟨run id⟩/` with the run id `⟨id⟩-⟨config⟩-⟨job id⟩[_⟨task⟩]`
   (or `$RUN_ID`), which a Slurm requeue keeps, so a requeued job resumes;
3. copies the config (`config.orig.toml`) and writes the config as run (`config.toml`: `out`,
   `cache_dir`, `cache_mirror`, `checkpoint`, `stop_file` set to the run directory and caches);
4. appends the start of an attempt to `run.toml`: run id, campaign, config and its hash,
   commit, Julia version, the input data file and its hash (the model files are not in git),
   mode, command, job id, array task, partition, QOS, nodes, cores, GPUs, memory,
   restart count, start;
5. sets the threads from the allocation (§5) and runs the driver under `srun`; SIGTERM
   (preemption, `scancel`) and SIGUSR1 (the templates' `--signal=B:USR1@⟨lead⟩` before the time
   limit) create the stop file;
6. at the end completes the attempt: exit code, status (`finished`, `checkpointed` = exit 75 after a
   snapshot, `preempted` = signalled without a snapshot, `failed`), the `sacct` figures (elapsed, MaxRSS, CPU time, energy) and the output files;
   appends one line to the runs indexes (global and the campaign's, under a lock); archives
   finished and failed runs without checkpoints;
7. a run stopped by the time limit (exit 75) with `RESUBMIT=n > 0` submits its own
   continuation (afterany dependency, same run id, `n−1` left); a preempted run is requeued by
   Slurm instead. Runs longer than 24 hours are such chains, not `medium`/`long` jobs (`long`
   allows only 32 cores per job).

The wrapper was tested locally on 2026-11-04 with the CPU driver on `hpc/configs/h0_smoke.toml`:
a SIGTERM to the wrapper during the fine warmup gave a snapshot, exit 75 and the status
`checkpointed`; the same run id started again resumed from the snapshot (fine warmup, sweep 1
of 4) and finished with exit 0.

## 6. Records, the ledger, and results through git

- **Run record** `run.toml` (one `[[attempt]]` per job attempt) and **runs index** lines in
  `ledger/runs.jsonl` and `campaigns/⟨id⟩/runs.jsonl`, failed runs included.
- **Campaign:** `hpc/new_campaign.sh ⟨id⟩ [⟨commit⟩]` makes the pinned worktree, prepares it, and
  starts `campaigns/⟨id⟩/campaign.toml` in the clone (id, commit, paths; question and design to
  fill in from the ledger row).
- **Results come back through git.** `hpc/collect.sh ⟨id⟩ [globs]` copies, for every archived
  run, `run.toml`, the configs and the small result files (default `summary*`, `*.csv`,
  `*.txt`) into `campaigns/⟨id⟩/runs/⟨run id⟩/` of the clone, with the campaign's runs index,
  refuses files above 5 MB and a campaign above 50 MB (`ESM_COLLECT_MAX_MB`,
  `ESM_COLLECT_TOTAL_MB`), and commits in the clone. It never pushes: the user reviews and
  pushes from the cluster, and pulls locally. Large outputs stay in `archive/`.
- **Experiment ledger** ([../experiments.md](../experiments.md)): one row per campaign, whose
  record column points to `campaigns/⟨id⟩/`. The work log notes what was submitted.

## 7. State snapshots and resume

The jobs run preemptible (`standby`, requeued automatically), and losing the work since the
last snapshot is acceptable, so the CPU driver writes **snapshots of the chain states**, not
a full-fidelity checkpoint: plain HDF5 data that survives code and Julia versions (layout in
[data.md](data.md) §5).

- **Keys** ([config.md](config.md)): `checkpoint` (the snapshot file; empty: off),
  `checkpoint_minutes` (30), `stop_file`. The wrapper points `checkpoint` at
  `runs/⟨id⟩/⟨run id⟩/snapshot.h5` and `stop_file` at a file in the run directory.
- **When** (`run_chains`, at sweep boundaries: after a level-0 warmup iteration or a fine
  sweep over all chains): every `checkpoint_minutes` of wall time; once after the level-0
  warmup, before the first fine sweep; and when the stop file exists, after which the driver
  removes the stop file and exits with **code 75** (`scripts/run.jl` exits with the driver's
  code: 0 finished, 75 stopped after a snapshot, anything else a failure). The snapshot is
  written to a temporary name, flushed (`fsync`) and renamed over the previous one.
- **Contents:** the phase (`level0` or `fine`; a fine sweep counter above `mlda_warmup` is
  production) and the sweep counter; per chain the state (level, regimes `m`, `z0`, the
  innovations `w` per level, `k`, `v`) and its start kind; the monitored traces so far; the
  CO₂ price and its tuning trace; in the level-0 warmup the step sizes of the carrier scaling
  and the slice widths. Not the learned modes, the bias, the online-bias observations, the
  post-correction records or the price accumulators.
- **Resume:** if the snapshot file exists at start, the driver sets up as usual (the caches
  make that quick), loads the states, logs the restart count and the position, and continues
  the phase and the sweep counter: the chains restart from exactly the saved states, with
  random streams re-seeded from `seed` and the restart count (never a replay of the old
  stream), kernels rebuilt (adaptation starts afresh in the rest of the warmup and is frozen
  in production) and the caches recomputed from the states. A resume after the level-0
  warmup lifts the set-up bias, not the one learned in the warmup (the bias is being
  retired); the sampled prices and the post-correction cover the production sweeps since the
  last start; the output records `run/restarts` and `run/resumed_at_sweep`. A snapshot is
  refused if its levels or seed differ from the configuration.
- **SIGTERM.** Slurm sends SIGTERM to every process of a preempted job and SIGKILL after the
  grace time. With `ESM_IGNORE_SIGTERM=1`, `scripts/run.jl` ignores SIGTERM
  (`ESMSampling.ignore_sigterm`: `SIG_IGN` and SIGTERM unblocked on the main thread; Julia's
  own signal listener would otherwise receive it with `sigwait` and exit); the wrapper's trap
  creates the stop file, and the driver writes the snapshot at the next sweep boundary and
  exits with 75. The grace time must cover one sweep plus the write.

> **Status: missing (device).** `run_device` writes no snapshot yet. On the device the state is
> downloaded at each host checkpoint already; the snapshot adds the file write.

Snapshots are also outputs: a later experiment can start from them. They stay in the run
directory and are not archived (their size grows with the model).

## 8. First steps on the cluster

1. After the next push: `git -C /p/projects/coen/esm/ESM-Sampling pull` on a login node.
2. `bash hpc/prepare_env.sh` on a login node (or `sbatch hpc/jobs/prepare.sbatch`).
3. Build the model files: `sbatch hpc/jobs/data.sbatch small` (hours), and later
   `sbatch hpc/jobs/data.sbatch pypsa256` (the PyPSA-Eur build, longer), then
   `sbatch hpc/jobs/era5.sbatch` for the ERA5 variant.
4. `bash hpc/new_campaign.sh H0`, then `sbatch hpc/jobs/h0_cpu.sbatch H0` and
   `sbatch hpc/jobs/h0_gpu.sbatch H0`; read the logs in `/p/tmp/hellmann/esm/logs/`.
5. H2 (reference LPs): `bash hpc/new_campaign.sh H2`, then per configuration
   `sbatch hpc/jobs/build.sbatch H2 <config> script scripts/lp_reference.jl`, then
   `bash hpc/collect.sh H2 'summary*'` and push.
