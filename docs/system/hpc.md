# Runs on the PIK HPC cluster

How the sampler is run on the PIK cluster (HPC2024, Slurm): the environment, where code, data,
caches and results live, the job types, checkpointing, how every run is recorded, and how
results come back. The scripts are in `hpc/`; the campaigns are the rows of the campaign
list ([../campaigns.md](../campaigns.md)). User `hellmann`, project directory `coen`
(`/p/projects/coen`), Slurm account `icone`.

> **Status: partial.** The scripts exist and have run: the environment (H0: offline CPU and
> GPU jobs), the model files and the ERA5 file. Open: the SIGTERM and snapshot-resume tests on
> a compute node (H0b), the rebuild of the POWER model files with the corrected calibration
> (§8.3).

## 1. The cluster in brief

| resource | what | limits that matter here |
|---|---|---|
| login nodes `login[01-03]` | 64 cores (EPYC 9354, Zen 4), 768 GB, **internet** | interactive work, compiling, short tests; QOS `io` (partition `io`) runs batch jobs there: 4 cores, 3 days, 5 concurrent jobs |
| CPU nodes (`standard`, 180; `priority`, 60) | 128 cores (2 × EPYC 9554, Zen 4), 768 GB, no internet | QOS `short` 24 h (default), `medium` 7 days, `long` 30 days but at most 32 cores per job; `priority` 24 h, starts within minutes, at most 64 cores and 5 jobs; `standby` 7 days, preemptible (requeued with `--requeue`) |
| GPU nodes (`gpu`, 12) | 4 × H100 80 GB, 128 cores, 1.5 TB, `/localscratch` NVMe ~3 TB | QOS `gpushort` 24 h, `gpumedium` 7 days, `gpulong` 30 days, `gpupreempt` 30 days, preemptible; `--gres=gpu:N` |
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
   limit) create the stop file. `B:` sends the signal to the batch shell only, so the templates
   `exec` the wrapper: it is the batch shell;
6. at the end completes the attempt: exit code, status (`finished`, `checkpointed` = exit 75 after a
   snapshot, `preempted` = signalled without a snapshot, `failed`), the signal (`TERM`, `USR1`
   or none), the `sacct` figures (elapsed, MaxRSS, CPU time, energy) and the output files;
   appends one line to the runs indexes (global and the campaign's, under a lock); archives
   finished and failed runs without checkpoints;
7. a preempted run (SIGTERM) is requeued by Slurm (`--requeue`, same job id and run id) and
   resumes from its snapshot. Jobs are never resubmitted to get past a QOS time limit: every job
   asks for a `--time` that covers the run's estimate, in the QOS that allows it, preferably
   preemptible (`standby` up to 7 days, `gpupreempt` up to 30 days; for guaranteed runtime
   `medium`/`gpumedium` 7 days, `long` 30 days at most 32 cores). The templates' 24 h is a
   default for short runs; a campaign row gives `sbatch --time=…` where its runs are longer.
   Runs longer than a week are a design question, settled with the user before submitting. A
   run that reaches its time limit (SIGUSR1) stops with a snapshot (status `checkpointed`,
   signal `USR1`), so the work is kept; nothing is resubmitted.

The wrapper was tested locally on 2026-11-04 with the CPU driver on `hpc/configs/h0_smoke.toml`:
a SIGTERM to the wrapper during the fine warmup gave a snapshot, exit 75 and the status
`checkpointed`; the same run id started again resumed from the snapshot (fine warmup, sweep 1
of 4) and finished with exit 0. The signal path through the templates was simulated locally
with Slurm stubs (`cpu_array.sbatch`, a script-mode line): SIGUSR1 to the batch shell gave the
snapshot exit, status `checkpointed`, signal `USR1`, and no `sbatch`; a second attempt with
the same run id, sent SIGTERM, reused the run directory (a second attempt in `run.toml`).

## 6. Records, the ledger, and results through git

- **Self-contained campaigns.** A campaign needs nothing but its pinned commit and the cluster's
  own model files: every input (model files, start states, reference solutions) is made on the
  cluster by versioned, documented scripts, either by the data preparation (§8.3) or by a step
  of the campaign itself, submitted with a dependency on it. No files are copied between the
  local machine and the cluster; code goes up and results come back through git. A campaign
  does not pin a model file's hash; its run records carry the hash (`data_sha`), and the
  provenance file says how that file was built.
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
- **Campaign list** ([../campaigns.md](../campaigns.md)): one row per campaign, whose
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

**Device driver** (`run_device`, both samplers; [device.md](device.md) §6.3): the same keys, file
layout (plus a group `device/`), stop file, exit code 75 and `ESM_IGNORE_SIGTERM`
(`device/scripts/run_device.jl`); the boundaries are the host checkpoints (every
`device_sweeps` operation sweeps or replica steps), so the grace time must cover one of them
plus the write. On resume the device arrays and programs are rebuilt from the saved states
(the compile is repeated: 2–6 minutes) and Reactant's RNG is re-seeded. The replica driver
also writes a snapshot at the end of the run. Tested locally on 2026-11-04 through
`hpc/run.sh` in `device` mode on the CPU backend (`hpc/configs/device_smoke.toml`, the
simulated layout of §5).

The replica driver (`sampler = "replicas"`) follows the same contract with its own layout
(every replica's state, the Langevin steps and the gates; [data.md](data.md) §5,
[sampler.md](sampler.md) §11.5).

Snapshots are also outputs: a later experiment can start from them. They stay in the run
directory and are not archived (their size grows with the model).

## 8. What runs in which order

All commands on a login node in the clone, `cd /p/projects/coen/esm/ESM-Sampling`; logs in
`/p/tmp/hellmann/esm/logs/`. Each step names what it needs; steps without a dependency
between them can be submitted together.

### 8.1 Once: the clone and its environment

1. `git clone https://github.com/FHell/ESM-Sampling /p/projects/coen/esm/ESM-Sampling` (or let
   `prepare_env.sh` do it).
2. `bash hpc/prepare_env.sh` (or `sbatch hpc/jobs/prepare.sbatch`): Julia packages of both
   projects into the shared depot, precompiled; the Python environment (`uv sync`); Reactant's
   CUDA preferences.

### 8.2 After every pull

1. `git pull` (after a local push; results from the cluster go the other way, §8.5).
2. `bash hpc/prepare_env.sh` again if `Manifest.toml`, `device/Manifest.toml` or `uv.lock`
   changed (`git diff --stat HEAD@{1} -- Manifest.toml device/Manifest.toml uv.lock`). The
   data jobs and `run.sh` use the clone; campaigns use their own worktree (§6), prepared when
   they are created, so a pull never changes a running campaign.

### 8.3 Once: the model files

Each build appends a record to `/p/projects/coen/esm/data/processed/PROVENANCE.toml`; a
requeued or resubmitted job continues from its kept downloads and outputs.

| step | command | needs | makes | time |
|---|---|---|---|---|
| D1 | `sbatch hpc/jobs/data.sbatch small` | internet (QOS io) | `model.h5` (40 nodes), `model76.h5` (76 nodes), NASA POWER weather 2001–2022 | hours (limit 3 days) |
| D2 | `sbatch hpc/jobs/data.sbatch pypsa256` | internet (QOS io); independent of D1 | the PyPSA-Eur build `esm256`, then `model256.h5` (256 nodes, NASA POWER weather) | the longest (limit 3 days) |
| D3 | `sbatch --dependency=afterok:<job id of D2> hpc/jobs/era5.sbatch [years]` | D2 (`model256.h5` and the `esm256` resources) | `model256_era5.h5` (256 nodes, ERA5 weather) | not measured yet (limit 24 h, continues when requeued) |

D1 and D2 can be submitted together. Files built without the corrected wind calibration
([data_sources.md](data_sources.md) §2; their `PROVENANCE.toml` commit has no `smallest_root`
in `scripts/prepare_data.py`) need D1 and D2 again; a rerun of
D2 reuses the PyPSA-Eur build and only redoes `model256.h5`. D3 does not depend on the POWER
weather and need not be rerun. Check a build's log for the line per carrier
`… miss their window target by more than 0.005` (zero in the local builds of all three POWER
files). D3 reads the cluster's ERA5 folder in place on a compute
node (§4); its years are a choice: the default 2003, 2010, 2013, 2017 is enough for a first
full-resolution test, while the weather model is fitted on all years of the file
(`src/weather.jl`), so a production run wants more years (2013 must always be among
them). Each year reads about 57 GB from the shared folder, one file at a time.

ERA5 is mapped onto the 256-node network only: the cut-out box (lon −10.5…25.5,
lat 40.5…59.5) covers the 10-country region, and `model.h5` and `model76.h5` keep their NASA
POWER weather ([data_sources.md](data_sources.md) §4–5). `model256_era5.h5` is the input of
the full-resolution run, 256 nodes × hourly (`configs/full256_era5.toml`). Larger networks
(512, 1024 nodes, full Europe) need a new PyPSA-Eur build and a wider box
([roadmap.md](roadmap.md) §6).

### 8.4 Campaigns

The rows of [../campaigns.md](../campaigns.md) give each campaign's commands; the pattern is
`bash hpc/new_campaign.sh <id>` (worktree at the clone's current commit), the row's `sbatch`
lines, and `bash hpc/collect.sh <id> [<glob>]` when the runs are done. The order of the
current ones, with what each needs:

| campaign | needs | then |
|---|---|---|
| H0b (SIGTERM and snapshot tests) | D1: the snapshot test is skipped without `model.h5` | `bash hpc/new_campaign.sh H0b`; `sbatch hpc/jobs/h0_cpu.sbatch H0b` |
| H4 (replica sampler, far-apart starts) | D1 rebuilt with the corrected wind calibration | `bash hpc/new_campaign.sh H4`; `sbatch --time=2-00:00:00 --array=1-6%6 hpc/jobs/cpu_array.sbatch H4 campaigns/H4/list.txt` (6-hourly lines estimated 12–20 h); `bash hpc/collect.sh H4` |
| H10 (storage pairs) | D1 (rebuilt); being redesigned to make its start states on the cluster | as in its row, once prepared again |
| H2 and the other LP campaigns | D1 for 40 and 76 nodes, D2 for 256, D3 for ERA5 | as in their rows |
| the full-resolution run | D3 | not yet a campaign ([roadmap.md](roadmap.md) §6) |

### 8.5 Results back

1. `bash hpc/collect.sh <id> [<glob>]` on the cluster commits the run records and small results
   to `campaigns/<id>/` in the clone (§6); it never pushes.
2. The user pushes from the cluster; locally `git pull` (or fetch and merge) brings them in, and
   the campaign's row in `campaigns.md` gets its result.

