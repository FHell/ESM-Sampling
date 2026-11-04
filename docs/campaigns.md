# Measurement campaigns

Campaigns produce results: validation against LP baselines, scaling and throughput, long
convergence runs, production. They run on the PIK cluster ([system/hpc.md](system/hpc.md)):
prepared and smoke-tested locally, then submitted there. Exploratory work that decides what to
build is in [experiments.md](experiments.md); the line is not strict.

**Life of a campaign.**
1. Prepare, self-contained ([system/hpc.md](system/hpc.md) §6): every input made on the
   cluster by a versioned script (the data preparation, or a first step of the campaign),
   nothing copied from the local machine; configs, the script for `hpc/run.sh`, the list file
   for the array template, and
   `campaigns/⟨id⟩/campaign.toml` (question, design) in the repository; a row here, status
   `prepared`; a local smoke run of a few minutes.
2. On the cluster: `bash hpc/new_campaign.sh ⟨id⟩` (pins the code at a commit), then the
   `sbatch` commands of the row; status `running`.
3. `bash hpc/collect.sh ⟨id⟩` commits the run records and small results to `campaigns/⟨id⟩/`;
   push, pull locally; status `done`, the one-line result filled in.

What each campaign needs first (environment, model files, copied inputs) and the order of the
one-time steps: [system/hpc.md](system/hpc.md) §8.

| id | status | question | design | how to run | result | record |
|---|---|---|---|---|---|---|
| H0 | done, except two tests (rerun as H0b) | does the environment work offline on CPU and GPU nodes | precompiled depot from a login node; offline load, CPU tests and a smoke run through `run.sh`; Reactant on the H100 and a fast-family profile | `sbatch hpc/jobs/h0_cpu.sbatch H0`, `sbatch hpc/jobs/h0_gpu.sbatch H0` | GPU (job 2902915, 4.6 min): offline, CUDA 13.1 runtime on driver 580.126 (CUDA 13.0), fast families 0.025 s/sweep at 40 × 6 h with 4 chains (laptop 0.57 s), compile 88 s then 4–12 s per family. CPU (job 2902913 on csp14c04, 12 min): offline load, 33 test sets passed; one check in `test_hpc.jl` failed because jobs set `ESM_CACHE_DIR` (a test bug, fixed in f89c77e), which skipped the SIGTERM and snapshot-resume tests; the smoke run through `run.sh` finished with a complete run record and a snapshot. Open: the two skipped tests; the XLA persistent cache | `campaigns/H0/`; `/p/tmp/hellmann/esm/runs/H0/` |
| H0b | prepared | do the snapshot-resume and SIGTERM tests pass on a compute node | H0's CPU job at the commit with the `test_hpc.jl` fix (new campaign, so H0 stays pinned to its commit) | `bash hpc/new_campaign.sh H0b`, `sbatch hpc/jobs/h0_cpu.sbatch H0b` | — | `campaigns/H0b/` |
| H1 | planned | CPU throughput per chain against threads and chains per node | 40 and 256 nodes, daily and 6 h; threads 4–64 per chain, 1–16 chains per node; sweeps/s and memory | — | — | [system/hpc.md](system/hpc.md) §5 |
| H2 | being prepared | reference LP optima (HiGHS) per network, resolution and weather year | 40, 256 nodes (later 512, 1024); daily, 6 h, 1 h; one build job per case; cached | (agent preparing: historical years, stacked optima, MGA envelopes) | — | [system/validation.md](system/validation.md) |
| H3 | planned | autocorrelation times of the operation at fixed (k, x), long runs | E17 protocol, 5000+ sweeps, job array over chains, sizes and resolutions | — | — | [experiments.md](experiments.md) E17 |
| H4 | prepared (smoke-tested locally 2026-11-04: line 1 through the array template and `run.sh`, init 305 s, ~7.5 s per step with 8 replicas on 4 loaded threads, preempted twice, snapshot and exit 75 each time, resumed at step 10); to be submitted on the rebuilt `model.h5` (wind-calibration fix, [system/data_sources.md](system/data_sources.md) §2), the current file calibrates 9 of 19 offwind nodes and 1 onwind node into cut-out | does the replica–Langevin sampler (`sampler = "replicas"`) bring chains from far-apart starts to the same capacities, and at what cost per step | 40 nodes (chains40 model, no bias), one chain per run, low (×0.5) and high (×2) heuristic starts; daily M = 8 and M = 4 (1500 + 3000 steps), 6-hourly M = 8 (800 + 1600 steps); all replica keys at their defaults; 8 cores each; then 256 nodes daily and the comparison with H2 | `bash hpc/new_campaign.sh H4`, then `sbatch --time=2-00:00:00 --array=1-6%6 hpc/jobs/cpu_array.sbatch H4 campaigns/H4/list.txt`; `bash hpc/collect.sh H4` collects `summary_replicas.txt` | — | `campaigns/H4/` (configs, `list.txt`, `campaign.toml`); [system/sampler.md](system/sampler.md) §11 |
| H5 | planned | device throughput on the H100 | chains × replicas batch scaling, FP64 against the Float32 switch, memory per chain, 40 and 256 nodes, all families | the replica driver: `sbatch hpc/jobs/gpu.sbatch H5 configs/device40_replica.toml device` (nchains, `device_replica.M` varied; per-part seconds per host checkpoint in `device_profile/*`: weather, operation, score, langevin); the operation families with `profile_families.jl` | — | [system/performance.md](system/performance.md) §7 |
| H6 | planned | sweeps to equilibrium against network size and finest step | 40, 256, 512, 1024 nodes; 24, 12, 6, 3, 1 h | — | — | [system/roadmap.md](system/roadmap.md) §7 (with memory for full Europe hourly) |
| H7 | planned | first full-resolution run, 256 nodes hourly with ERA5 | ERA5 from the cluster's ERA5 folder (`scripts/prepare_era5.py` as a build job), `configs/full256_era5.toml` | after `data.sbatch pypsa256` and `era5.sbatch` | — | [archive/full_resolution_run.md](archive/full_resolution_run.md) |
| H8 | deferred (no new statistical tests for now) | sensitivity of the exactness tests | long reference runs as job arrays | — | — | [experiments.md](experiments.md) P6 |
| H9 | planned | design under stressed weather | the explicit weather tilt (λ > 0) once built | — | — | [system/ensemble.md](system/ensemble.md) §3.1 |
| H10 | being redesigned (start states made on the cluster) | do the cross-node storage pairs (`sxpair`, `sxlong`) relax the seasonal storage at fixed (k, x) faster than the E21 kernel, on both E15 chains and over 5000 sweeps | E17/E21 protocol (`scripts/mixing/opmix_hpc.jl` through `run.sh` script mode), `chains40`, start states = E15 end states (chains 4 and 1), kernels `slong` (E21), `sx` (defaults: 8 `sxlong` per unit, `sxpair` 0.1), `sx16` (16, 0.2); 5000 sweeps, 6 array tasks of 8 cores; snapshots every 10 min and at the stop file, exact resume; summaries: τ, drift per quantity and per storage unit, cost per sweep, per-sweep trace | `bash hpc/new_campaign.sh H10`; copy the start states and the model file from the local machine (`campaigns/H10/campaign.toml`, `inputs`: the chains file into the campaign code, `model.h5` as `data/processed/model_e15.h5`, which `campaigns/H10/chains40_e15.toml` reads); `sbatch --array=1-6%6 hpc/jobs/cpu_array.sbatch H10 campaigns/H10/list.txt`; afterwards `bash hpc/collect.sh H10 'summary*'` | — | [campaigns/H10/campaign.toml](../campaigns/H10/campaign.toml), [experiments.md](experiments.md) E27 |
