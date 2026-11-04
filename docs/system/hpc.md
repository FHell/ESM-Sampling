# Runs on the PIK HPC cluster

How the sampler is run on the PIK cluster (HPC2024, Slurm): the environment, where code, data,
caches and results live, the job types, checkpointing, and how every run is recorded. The
campaigns themselves are rows H0–H9 of the experiment ledger
([../experiments.md](../experiments.md)). Facts about the cluster are from the HPC2024 user
guide; values marked ⟨…⟩ are to be filled in.

> **Status: missing.** Nothing of this exists yet; this document is the intended shape. The
> code changes it needs are listed in §8.

## 1. The cluster in brief

| resource | what | limits that matter here |
|---|---|---|
| login nodes `login[01-03]` | 64 cores (EPYC 9354, Zen 4), 768 GB, **internet** | interactive work, compiling, short tests; QOS `io` (partition `io`) runs batch jobs there: 4 cores, 3 days, 5 concurrent jobs |
| CPU nodes (`standard`, 180; `priority`, 60) | 128 cores (2 × EPYC 9554, Zen 4), 768 GB, no internet | QOS `short` 24 h (default), `medium` 7 days, `long` 30 days but at most 32 cores per job; `priority` 24 h, starts within minutes, at most 64 cores and 5 jobs; `standby` preemptible |
| GPU nodes (`gpu`, 12) | 4 × H100 80 GB, 128 cores, 1.5 TB, `/localscratch` NVMe ~3 TB | QOS `gpushort` 24 h, `gpumedium` 7 days, `gpulong` 30 days, `gpupreempt` preemptible; request GPUs with `--gres=gpu:N` |
| `$HOME` | backed up, small quota | no job I/O |
| `/p/projects/⟨project⟩` | backed up, by ticket | the permanent home of this project on the cluster |
| `/p/tmp/$USER` | fast parallel FS, not backed up, **inactive files are deleted** | job working directories |
| `/localscratch` (GPU nodes) | per job, cleared at job end | staging and compile scratch |
| ERA5 folder ⟨path⟩ | read-only, shared on the cluster | the ERA5 cut-outs for the weather inputs (§4) |

Fair share charges *allocated* cores × wall time (memory is free, default ~6 GB per core) with
a 7-day half-life. Preemption (standby, gpupreempt, and cancellation at the time limit) sends
SIGTERM, then SIGKILL 5 minutes later. Licensed solvers are not available: every LP is solved
with HiGHS (the Julia code uses JuMP + HiGHS; PyPSA-Eur's `config.esm.yaml` sets
`solving: solver: highs`).

## 2. Storage layout

```
/p/projects/⟨project⟩/esm/                 permanent, backed up
├── code/                                  git clone; one worktree per campaign, pinned commit
├── env/
│   ├── julia_depot/                       JULIA_DEPOT_PATH (packages, artifacts incl. XLA/CUDA, precompile caches)
│   ├── uv_cache/  venv/                   UV_CACHE_DIR, the Python environment (data preparation)
│   └── ENVIRONMENT.toml                   module versions, Julia version, Manifest hashes, prepared on, by
├── data/
│   ├── raw/                               downloaded inputs (OPSD, PyPSA-Eur bundles)
│   └── processed/                         model*.h5 (40, 76, 256 nodes, ERA5 variants), read-only for jobs
├── cache/                                 durable mirror of the computed caches (weather models, reference LPs)
├── archive/⟨campaign⟩/⟨run_id⟩/           finished runs: outputs, run record, logs (no checkpoints)
└── ledger/runs.jsonl                      one line per run (§6)

/p/tmp/$USER/esm/                          fast, purged when inactive
├── cache/                                 the caches jobs read and write (content-hashed keys)
└── runs/⟨run_id⟩/                         working directory: config copy, checkpoints, logs, outputs
```

- **Code, environment and data never live on `/p/tmp`:** its automatic deletion would remove a
  depot or a data file that has not been touched for a while, and jobs would fail offline.
- **Caches are content-addressed** (hash of the model options, levels, weather seed), so all
  runs share one cache directory, and a purged cache is recomputed, not wrong. At the end of a
  campaign the cache entries it used are copied to the durable mirror; jobs fall back to the
  mirror before recomputing.
- **Outputs are written to the run directory** on `/p/tmp`; the job epilogue copies the
  outputs and the run record (not the checkpoints) to `archive/`, and appends the ledger line.
- **GPU jobs** stage the model file to `/localscratch/$USER/$SLURM_JOB_ID` (it is read once,
  so this matters little) and put scratch files of the compiler there.

## 3. Environment, prepared on a login node

Compute nodes have no internet, so everything is downloaded, instantiated and precompiled on a
login node, and jobs run with downloads switched off.

**Julia.** The Manifests are resolved with Julia 1.12.2 and the projects require stdlib
versions of 1.12, so the cluster needs a Julia 1.12 module (`module avail julia`; the guide
lists 1.11.7). If none exists, install a 1.12 binary with juliaup into `env/` on the login node
(the cluster has internet there), or ask for the module.

```bash
# login node, once per environment version
module load julia/⟨1.12.x⟩
export JULIA_DEPOT_PATH=/p/projects/⟨project⟩/esm/env/julia_depot
export JULIA_CPU_TARGET="generic;znver4,clone_all"   # login (EPYC 9354) and compute (9554) are both Zen 4
export JULIA_NUM_PRECOMPILE_TASKS=8                   # be gentle with the shared login node
cd /p/projects/⟨project⟩/esm/code
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=device -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

- **Reactant on the GPU.** Reactant fetches its XLA runtime (and the CUDA libraries) as
  artifacts when the project is instantiated, and chooses the variant by the platform it
  detects. On a login node without a GPU it may pick the CPU-only variant; the CUDA variant
  must be selected explicitly (a Reactant preference in `device/LocalPreferences.toml`) before
  instantiating. ⟨To verify in H0.⟩ The cluster's `module load cuda/12.6.0` is not needed if
  Reactant brings its own CUDA runtime; the NVIDIA driver on the GPU nodes must be new enough.
- **In jobs:** the same `JULIA_DEPOT_PATH`, plus `JULIA_PKG_OFFLINE=true` (a missing package
  fails loudly instead of trying the network), `julia -t $SLURM_CPUS_PER_TASK`, and the run
  wrapper of §5.
- **Precompile once per code version:** a changed `src/` triggers precompilation of
  ESMSampling in the first job, which is fine (a minute) but concurrent array tasks would all
  do it. The preparation script therefore precompiles after every checkout of a campaign
  commit.

**Python** (data preparation, PyPSA-Eur): with the local `uv`, `UV_CACHE_DIR` and the venv in
`env/`, `uv sync` on the login node; jobs set `UV_OFFLINE=1`. PyPSA-Eur's retrieval rules need
internet: run them in a job with `--partition=io --qos=io` (or locally, then `rsync`); the
solving rules run on compute nodes with HiGHS.

**Record.** `env/ENVIRONMENT.toml` holds the Julia and module versions, the Manifest hashes,
the uv lock hash, the date and who prepared it; every run record refers to it (§6).

## 4. Inputs and caches

- **Model files** (`data/processed/model*.h5`, 0.1–0.6 GB each) are built locally or in a
  preparation job and copied to `data/processed/` with `rsync -avzP`.
- **ERA5.** The cluster has a shared, read-only ERA5 folder (⟨path⟩), so no ERA5 download is
  needed: `scripts/prepare_era5.py` runs as a build job on a compute node, reads only the
  variables, years and region it needs, sequentially, and writes the processed weather into
  `data/processed/` (e.g. `model256_era5.h5`). It never scans the folder recursively (no
  `find` or `grep` over it) and runs one such job at a time, to keep the load on the shared
  storage low.
- **Weather-model and LP caches.** The code now writes them to `dirname(cfg.out)/cache`, i.e.
  per run directory. **Status: missing:** a `cache_dir` configuration key (or `ESM_CACHE_DIR`
  environment variable) so that all runs share `/p/tmp/$USER/esm/cache`, with a read-only
  fallback to the durable mirror, and writes to a temporary name followed by an atomic rename
  so that concurrent array tasks cannot read half-written entries.
- **Reference LPs** (the zero-temperature optimum per network, resolution and weather year,
  for validation and for the starts) are computed once per campaign in a preparation job
  (HiGHS, one job per case, threads = cores requested) and cached.
- **Compilation.** A Reactant process spends 2–4 minutes tracing and compiling before the
  first sweep (most of it once per process; [performance.md](performance.md) §7). GPU jobs are
  therefore long-lived processes that batch many chains and replicas, not arrays of short
  jobs. Whether XLA's persistent compilation cache can be used (on `/localscratch` or the
  depot) is to be checked in H0.

## 5. Job types

| type | resources | QOS | used for |
|---|---|---|---|
| **prepare** | login node or `io` job (4 cores) | `io` | instantiate, precompile, downloads, PyPSA-Eur retrieval |
| **build** | 1 node part, 16–64 cores | `short` | PyPSA-Eur solves (HiGHS), weather-model fits, reference LPs |
| **CPU chain** | 1 task, `--cpus-per-task` = threads of the chain (4–32, from H1), memory by model size | `short`, with restart chaining; `standby` when restartable | CPU sampler runs; job arrays over chains, seeds or configurations |
| **GPU** | `--gres=gpu:1`, 16 cores, 128–256 GB | `gpushort`, `gpumedium`; `gpupreempt` when restartable | device runs: C chains × M weather replicas batched on one H100 |
| **analysis** | 1–4 cores | `priority` | diagnostics, summaries, plots |

**The run wrapper** (`hpc/run.sh`, missing) is the only entry point of a job:

1. loads the modules and the environment of §3, creates `runs/⟨run_id⟩/` on `/p/tmp`, copies
   the configuration there, and refuses to run from a dirty or unpinned code tree;
2. writes the start of the run record (§6);
3. runs the driver with the run directory as its output and checkpoint location, and forwards
   SIGTERM to it (Slurm's 5-minute grace) so that it writes a checkpoint and exits;
4. on exit, completes the run record with `sacct` (elapsed, MaxRSS, CPU efficiency,
   ConsumedEnergy), copies outputs and record to `archive/`, and appends the ledger line.

**Longer than 24 hours.** Runs that exceed a QOS limit are a chain of jobs
(`sbatch --dependency=afterany:⟨jobid⟩`) that resume from the checkpoint, rather than
`medium`/`long` jobs (`long` allows only 32 cores per job). The same mechanism makes
`standby` and `gpupreempt` usable: a preempted job is requeued and resumes.

## 6. Run records and the ledger

- **Run id:** `⟨campaign⟩-⟨config⟩-⟨yyyymmdd-hhmm⟩-⟨jobid⟩[_⟨array index⟩]`.
- **Run record** (`runs/⟨run_id⟩/run.toml`): campaign id (ledger H-row), configuration (copy
  and hash), git commit, environment record, Slurm job id, partition, QOS, nodes, cores, GPUs,
  memory, seeds, start, end, exit status, checkpoint/resume count, the `sacct` summary, and
  the output files with their sizes.
- **Runs index** (`ledger/runs.jsonl`): one JSON line per finished or failed run, the record's
  fields; the campaign's analysis reads it to select runs. Failed runs are recorded too.
- **Experiment ledger** ([../experiments.md](../experiments.md)): one row per campaign (H0–H9),
  with the question, the design, the one-line result and the record (archive path, the runs
  index filter). Results summaries are copied back to the local repository under
  `results/hpc/⟨campaign⟩/` with `rsync`; the work log ([../worklog.md](../worklog.md)) notes
  what was submitted and when.

## 7. Checkpoints

**Status: missing.** Neither driver writes a restartable checkpoint. Needed: every ~15–30
minutes and on SIGTERM, an atomic write (temporary file, then rename) to the run directory of
everything a resumed run needs — chain states, sampler adaptation (step sizes, learned modes),
RNG states, the Langevin step sizes and lag-probe state, replica weather and weights, the
sweep counter and the monitored traces so far — and at start-up a resume when a checkpoint
exists. On the device, the state is downloaded at each host checkpoint already; the checkpoint
adds the file write and the RNG keys.

## 8. Code changes needed before the first campaign

1. `cache_dir` key or `ESM_CACHE_DIR`, shared cache with atomic writes (§4).
2. Checkpoint and resume in `run_chains` and `run_device`, with SIGTERM handling (§7).
3. `hpc/` scripts: `prepare_env.sh` (login node), `run.sh` (wrapper), job templates per type
   (§5), `record.jl` or a small Python helper for the run record and ledger line (§6).
4. A Julia 1.12 module or binary on the cluster, and the Reactant CUDA artifact on a login
   node (§3), verified in H0.
