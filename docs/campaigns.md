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
| H2 | prepared | LP baselines of our model: single-year optima on the historical years 2001–2022 and their spread, the stacked multi-year optimum and how it settles with the number of years, MGA envelopes (min/max total capacity per carrier at cost ≤ (1+ε)C\*), along the resolution ladder | `scripts/lp_baseline.jl` on `campaigns/H2/configs/` (default model terms), runs on the **rebuilt** model files (wind-calibration fix); three arrays by resources: `list_small.txt` (24: 40 × 24 h single years, stacked 2009–13, MGA 2013 ε 1/2/5/10 %; 5 min and 4 GB per LP), `list_medium.txt` (71: 40 × 6 h and 256 × 24 h single years, 40 × 24 h stacked 2004–13, MGA 40 × 6 h and 256 × 24 h at 2013 and 40 × 24 h stacked 2009–13 at ε 2/5 %; ≲ 14 h, ≲ 33 GB), `list_large.txt` (7: 40 × 1 h years 2003/10/13/17 ~80 GB ~20 h, 40 × 24 h stacked 22 y ~72 GB ~17 h, 40 × 6 h stacked 2009–13 ~65 GB ~14 h, 256 × 6 h 2013 ~85 GB ~24 h); estimates from 40 × 24 h measured locally (317 s, 4.2 GB, 2.4e5 variables) with memory 1 GB + 13 kB/variable and time ∝ size^1.7; details in `campaigns/H2/campaign.toml` | `bash hpc/new_campaign.sh H2`; from the clone: `sbatch --array=1-24%8 -c 4 --mem-per-cpu=6G hpc/jobs/cpu_array.sbatch H2 campaigns/H2/list_small.txt`; `sbatch --array=1-71%10 -c 8 --mem-per-cpu=6G hpc/jobs/cpu_array.sbatch H2 campaigns/H2/list_medium.txt`; `sbatch --array=1-7%4 -c 8 --mem-per-cpu=16G --partition=standard --qos=short --no-requeue hpc/jobs/cpu_array.sbatch H2 campaigns/H2/list_large.txt`; then `bash hpc/collect.sh H2` | — | `campaigns/H2/`; [system/validation.md](system/validation.md) §2.7 |
| H3 | planned | autocorrelation times of the operation at fixed (k, x), long runs | E17 protocol, 5000+ sweeps, job array over chains, sizes and resolutions | — | — | [experiments.md](experiments.md) E17 |
| H4 | planned | the replica–Langevin sampler at scale | after the prototype is merged: 40 nodes daily → 6 h → 256 nodes; low/high starts; against H2 | — | — | [worklog.md](worklog.md) |
| H5 | planned | device throughput on the H100 | chains × replicas batch scaling, FP64 against the Float32 switch, memory per chain, 40 and 256 nodes, all families | — | — | [system/performance.md](system/performance.md) §7 |
| H6 | planned | sweeps to equilibrium against network size and finest step | 40, 256, 512, 1024 nodes; 24, 12, 6, 3, 1 h | — | — | [system/roadmap.md](system/roadmap.md) §7 (with memory for full Europe hourly) |
| H7 | planned | first full-resolution run, 256 nodes hourly with ERA5 | ERA5 from the cluster's ERA5 folder (`scripts/prepare_era5.py` as a build job), `configs/full256_era5.toml` | after `data.sbatch pypsa256` and `era5.sbatch` | — | [archive/full_resolution_run.md](archive/full_resolution_run.md) |
| H8 | deferred (no new statistical tests for now) | sensitivity of the exactness tests | long reference runs as job arrays | — | — | [experiments.md](experiments.md) P6 |
| H9 | planned | design under stressed weather | the explicit weather tilt (λ > 0) once built | — | — | [system/ensemble.md](system/ensemble.md) §3.1 |
