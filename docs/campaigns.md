# Measurement campaigns

Campaigns produce results: validation against LP baselines, scaling and throughput, long
convergence runs, production. They run on the PIK cluster ([system/hpc.md](system/hpc.md)):
prepared and smoke-tested locally, then submitted there. Exploratory work that decides what to
build is in [experiments.md](experiments.md); the line is not strict.

**Life of a campaign.**
1. Prepare: configs, the script for `hpc/run.sh`, the list file for the array template, and
   `campaigns/⟨id⟩/campaign.toml` (question, design) in the repository; a row here, status
   `prepared`; a local smoke run of a few minutes.
2. On the cluster: `bash hpc/new_campaign.sh ⟨id⟩` (pins the code at a commit), then the
   `sbatch` commands of the row; status `running`.
3. `bash hpc/collect.sh ⟨id⟩` commits the run records and small results to `campaigns/⟨id⟩/`;
   push, pull locally; status `done`, the one-line result filled in.

| id | status | question | design | how to run | result | record |
|---|---|---|---|---|---|---|
| H0 | GPU half done; CPU half running | does the environment work offline on CPU and GPU nodes | precompiled depot from a login node; offline load, CPU tests and a smoke run through `run.sh`; Reactant on the H100 and a fast-family profile | `sbatch hpc/jobs/h0_cpu.sbatch H0`, `sbatch hpc/jobs/h0_gpu.sbatch H0` | GPU (job 2902915, 4.6 min): offline, CUDA 13.1 runtime on driver 580.126 (CUDA 13.0), fast families 0.025 s/sweep at 40 × 6 h with 4 chains (laptop 0.57 s), compile 88 s then 4–12 s per family; XLA persistent compilation cache not exercised yet | `/p/tmp/hellmann/esm/runs/H0/h0_gpu_profile/report.txt`; [system/performance.md](system/performance.md) §7 |
| H1 | planned | CPU throughput per chain against threads and chains per node | 40 and 256 nodes, daily and 6 h; threads 4–64 per chain, 1–16 chains per node; sweeps/s and memory | — | — | [system/hpc.md](system/hpc.md) §5 |
| H2 | being prepared | reference LP optima (HiGHS) per network, resolution and weather year | 40, 256 nodes (later 512, 1024); daily, 6 h, 1 h; one build job per case; cached | (agent preparing: historical years, stacked optima, MGA envelopes) | — | [system/validation.md](system/validation.md) |
| H3 | planned | autocorrelation times of the operation at fixed (k, x), long runs | E17 protocol, 5000+ sweeps, job array over chains, sizes and resolutions | — | — | [experiments.md](experiments.md) E17 |
| H4 | prepared (smoke-tested locally 2026-11-04) | does the replica–Langevin sampler (`sampler = "replicas"`) bring chains from far-apart starts to the same capacities, and at what cost per step | 40 nodes (chains40 model, no bias), one chain per run, low (×0.5) and high (×2) heuristic starts; daily M = 8 and M = 4 (1500 + 3000 steps), 6-hourly M = 8 (800 + 1600 steps); all replica keys at their defaults; 8 cores each; then 256 nodes daily and the comparison with H2 | `bash hpc/new_campaign.sh H4`, then `sbatch --array=1-6%6 hpc/jobs/cpu_array.sbatch H4 campaigns/H4/list.txt`; `bash hpc/collect.sh H4` collects `summary_replicas.txt` | — | `campaigns/H4/` (configs, `list.txt`, `campaign.toml`); [system/sampler.md](system/sampler.md) §11 |
| H5 | planned | device throughput on the H100 | chains × replicas batch scaling, FP64 against the Float32 switch, memory per chain, 40 and 256 nodes, all families | — | — | [system/performance.md](system/performance.md) §7 |
| H6 | planned | sweeps to equilibrium against network size and finest step | 40, 256, 512, 1024 nodes; 24, 12, 6, 3, 1 h | — | — | [system/roadmap.md](system/roadmap.md) §7 (with memory for full Europe hourly) |
| H7 | planned | first full-resolution run, 256 nodes hourly with ERA5 | ERA5 from the cluster's ERA5 folder (`scripts/prepare_era5.py` as a build job), `configs/full256_era5.toml` | after `data.sbatch pypsa256` and `era5.sbatch` | — | [archive/full_resolution_run.md](archive/full_resolution_run.md) |
| H8 | deferred (no new statistical tests for now) | sensitivity of the exactness tests | long reference runs as job arrays | — | — | [experiments.md](experiments.md) P6 |
| H9 | planned | design under stressed weather | the explicit weather tilt (λ > 0) once built | — | — | [system/ensemble.md](system/ensemble.md) §3.1 |
