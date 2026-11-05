# Runbook: running the campaigns on the PIK cluster

For an agent (or person) working on a login node of the PIK cluster in the clone
`/p/projects/coen/esm/ESM-Sampling`. It says what to run, in which order, what each step
needs, how to check it, and how results come back. Background: [docs/system/hpc.md](../docs/system/hpc.md)
(environment, wrapper, records, §8 the order in brief), the campaign rows in
[docs/campaigns.md](../docs/campaigns.md), and CLAUDE.md.

## 0. Rules

- **Never push.** Commit results in the clone (`hpc/collect.sh` does it); the user reviews and
  pushes. Never copy files from or to another machine: every input is built here by the
  versioned scripts.
- **QOS and time:** every job asks for a `--time` that covers its estimate (the commands below
  carry them). Preemptible QOS first (`standby` up to 7 days, `gpupreempt` up to 30 days);
  `medium` only where a job cannot resume (single LPs). **Never resubmit or chain jobs to get
  past a time limit.** A job that hits its limit stops with a snapshot (`run.toml`: status
  `checkpointed`, signal `USR1`); report it, and resubmit it only with a longer `--time` in a QOS
  that allows the whole remaining run, with `RUN_ID=<its run id>` to resume
  (`sbatch --export=ALL,RUN_ID=<id> --time=… hpc/jobs/<template> …`).
- **Preemption is normal:** standby and gpupreempt jobs are requeued by Slurm and resume from
  their snapshot (a new `[[attempt]]` in `run.toml`). Nothing to do.
- **Login nodes:** only light work (git, `new_campaign.sh`, `collect.sh`, sbatch). Everything
  heavy goes through `sbatch`. Shared nodes: the templates set the thread counts.
- **Code changes:** none on the cluster unless a step is broken; then make the smallest fix on a
  branch, commit it, report it, and wait for the user before running campaigns on it.
- **Record** what you submitted (job ids) and what came back in `docs/worklog.md` (a dated
  entry) and the campaign's row in `docs/campaigns.md` (status, one-line result), committed.

## 1. Facts

| what | value |
|---|---|
| clone | `/p/projects/coen/esm/ESM-Sampling` (`$ESM_CODE`) |
| project space | `/p/projects/coen/esm` (`$ESM_PROJ`): data, env, campaigns, archive, cache mirror |
| scratch | `/p/tmp/hellmann/esm` (`$ESM_SCRATCH`): run directories, logs (`logs/`), cache |
| account | `icone` (all templates) |
| logs | `/p/tmp/hellmann/esm/logs/<job name>-<job id>.out` |
| run records | `/p/tmp/hellmann/esm/runs/<campaign>/<run id>/run.toml`, `run.log`, `stages.csv` |
| runs index | `/p/projects/coen/esm/ledger/runs.jsonl`, `…/campaigns/<id>/runs.jsonl` |
| archive | `/p/projects/coen/esm/archive/<campaign>/<run id>/` (finished and failed runs) |

## 2. The dependency graph

```
                         S0 pull (+ prepare_env.sh only if a lock file changed)
          ┌──────────────┬────────────┬───────────────┬──────────────────────────┐
          D1 small       D2 pypsa256  D4 pypsa256ts   H2b (PyPSA on its network: needs nothing new)
          │              │            │
          ├─ H0b tests (CPU) ─┐       │
          ├─ H0b device smoke (GPU)   │
          ├─ H10 stage 1 (LP) │       │
          ├─ H2 list_small    │       │
          │                   ▼       │
          │        H4 (after H0b tests pass)
          │        H10 stage 2 (after H10 stage 1 and H0b tests)
          └────────── H2 list_medium, list_large (after D1, D2, D4)
                             H5 (GPU throughput): after the device work is merged and
                             pushed (a later pull) and the H0b device smoke passed
```

| step | needs | QOS / resources | estimate | output |
|---|---|---|---|---|
| S0 | – | login | minutes | the clone at the pushed commit |
| D1 `data.sbatch small` | S0 | io, 4 cores | hours | `model.h5`, `model76.h5` (wind calibration fixed) |
| D2 `data.sbatch pypsa256` | S0 | io, 4 cores | hours (the PyPSA-Eur build is reused) | `model256.h5` |
| D4 `data.sbatch pypsa256ts` | S0 (the existing esm256 network) | io, 4 cores | under an hour | `model256_pypsa.h5` (PyPSA's own series) |
| H0b tests | D1 | priority, 8 cores | under 1 h | the deterministic suite on a compute node, incl. SIGTERM and snapshot-resume |
| H0b device smoke | D1 | gpushort, 1 GPU | under 1 h | the device driver through `run.sh`, GPU memory log |
| H2b | S0 | standby / medium | 1–7 days per line | PyPSA's optimum and MGA envelopes |
| H10 stage 1 | D1 | standby, build | ≤ 2 h | the reference LP in the shared cache |
| H10 stage 2 | H10 stage 1, H0b tests | standby | 1–2 h per line | storage-pair mixing |
| H4 | D1, H0b tests | standby, 8 cores | 6–20 h per line | replica sampler from far starts |
| H2 list_small | D1 | standby, 4 cores | ≤ 12 h | 40-node daily LPs |
| H2 list_medium / list_large | D1, D2, D4 | standby / medium | ≤ 2 / ≤ 4 days | 40-node 6 h, 256-node LPs, stacked years |
| H5 | device merge pulled, H0b device smoke | gpupreempt, 1 GPU | ≤ 12 h per line | GPU throughput, memory, Float32 |

The dependencies are enforced with `--dependency=afterok:<job ids>`, so almost everything is
submitted in one go (§3). `afterok` on the H0b test job works as a gate: `h0_cpu.sbatch` exits
non-zero when a test fails.

## 3. Submitting (one pass, with dependencies)

```bash
cd /p/projects/coen/esm/ESM-Sampling
git pull
git diff --stat HEAD@{1} -- Manifest.toml device/Manifest.toml uv.lock   # non-empty: bash hpc/prepare_env.sh first

# the campaigns' pinned worktrees (each runs prepare_env on its worktree; light)
for c in H0b H2 H2b H4 H10; do bash hpc/new_campaign.sh $c; done

# data (io QOS, at most 5 concurrent io jobs)
D1=$(sbatch --parsable hpc/jobs/data.sbatch small)
D2=$(sbatch --parsable hpc/jobs/data.sbatch pypsa256)
D4=$(sbatch --parsable hpc/jobs/data.sbatch pypsa256ts)

# independent of the rebuild: PyPSA on its own network
sbatch --time=5-00:00:00 --array=1-6%6 -c 4 --mem-per-cpu=8G hpc/jobs/cpu_array.sbatch H2b campaigns/H2b/list.txt
sbatch --time=7-00:00:00 --partition=standard --qos=medium --no-requeue --array=1-2 -c 8 --mem-per-cpu=8G \
       hpc/jobs/cpu_array.sbatch H2b campaigns/H2b/list_6h.txt

# after D1: the checks and the 40-node work
T0=$(sbatch --parsable --dependency=afterok:$D1 hpc/jobs/h0_cpu.sbatch H0b)
sbatch --dependency=afterok:$D1 --time=01:00:00 hpc/jobs/gpu_short.sbatch H0b hpc/configs/device_smoke.toml device
L1=$(sbatch --parsable --dependency=afterok:$D1 --time=02:00:00 hpc/jobs/build.sbatch H10 configs/chains40.toml script scripts/lp_reference.jl)
sbatch --dependency=afterok:$D1 --time=12:00:00 --array=1-23%8 -c 4 --mem-per-cpu=6G hpc/jobs/cpu_array.sbatch H2 campaigns/H2/list_small.txt

# after the H0b tests pass
sbatch --dependency=afterok:$T0 --time=2-00:00:00 --array=1-6%6 hpc/jobs/cpu_array.sbatch H4 campaigns/H4/list.txt
sbatch --dependency=afterok:$L1:$T0 --time=12:00:00 --array=1-6%6 hpc/jobs/cpu_array.sbatch H10 campaigns/H10/list.txt

# after all data
sbatch --dependency=afterok:$D1:$D2:$D4 --time=2-00:00:00 --array=1-46%10 -c 8 --mem-per-cpu=6G \
       hpc/jobs/cpu_array.sbatch H2 campaigns/H2/list_medium.txt
sbatch --dependency=afterok:$D1:$D2:$D4 --time=4-00:00:00 --partition=standard --qos=medium --no-requeue \
       --array=1-10%4 -c 8 --mem-per-cpu=24G hpc/jobs/cpu_array.sbatch H2 campaigns/H2/list_large.txt

echo "D1=$D1 D2=$D2 D4=$D4 T0=$T0 L1=$L1"     # note the ids in the worklog
```

Held, not submitted: `campaigns/H2/list_mga.txt` (our MGA; PyPSA's in H2b is the reference) and
`campaigns/H2b/list_hold.txt` (beyond a week by estimate). H5 comes after the next pull that
brings the device work (§5).

If a dependency fails (`squeue` shows `DependencyNeverSatisfied`), the dependent jobs never
start: read the failed job's log, report, `scancel` the stuck dependents, and resubmit them after
the fix.

## 4. Checks after each step

- **D1, D2, D4:** the log ends with `prepare_data.sh: done (…)`; the build lines report
  `0 miss their window target` per carrier (wind calibration); a new entry per file in
  `/p/projects/coen/esm/data/processed/PROVENANCE.toml`.
- **H0b tests:** the log shows every test set passing (the broken markers of `test_moves.jl` and
  `test_bias.jl` are expected), then the smoke run's `run.sh: … finished`.
- **H0b device smoke:** `run.toml` status `finished`; `gpu_mem.csv` and `summary_device.toml` in the
  run directory; `run.log` without errors.
- **Every campaign run:** `run.toml` (one `[[attempt]]` per attempt, the last with status
  `finished`), `stages.csv` (wall and CPU per stage), the summaries named in its row.
- **Monitoring:** `squeue -u hellmann`, `sacct -u hellmann -S today --format=JobID,JobName,State,Elapsed,ExitCode`,
  the logs, and `tail -f …/run.log` of a run directory.

## 5. H5 (GPU throughput), later

After the device work package is merged and pushed and pulled here, and the H0b device smoke
passed:

```bash
bash hpc/new_campaign.sh H5
sbatch --time=12:00:00 --array=1-12%4 hpc/jobs/gpu_array.sbatch H5 campaigns/H5/list.txt
```

Its list lines carry `XLA_REACTANT_GPU_PREALLOCATE=false` (so `gpu_mem.csv` shows the program's
memory) and `FLOAT=32` or `FAMILIES=all` where needed.

## 6. Results back

When a campaign's runs are finished (or failed):

```bash
bash hpc/collect.sh <campaign>          # copies records and small results, writes cost.csv, commits
git log -1 --stat                        # review
```

`campaigns/<id>/cost.csv` gives per campaign, run and stage (setup, LP optimum, each MGA solve,
burn-in, warmup, production, device compiles): wall hours, allocated core-hours (what fair share
charges), consumed CPU hours, GPU-hours and energy, summed over all attempts. Then fill in the
campaign's row in `docs/campaigns.md` (status `done`, one-line result) and a worklog entry,
commit, and tell the user to push.

## 7. When something fails

- **A run failed** (`run.toml` status `failed`): `run.log` in the run directory (or the archive)
  and the job log. Report the error with the run id; do not rerun blindly.
- **Out of memory** (sacct `OUT_OF_MEMORY`): resubmit that array task with more
  `--mem-per-cpu` (`--array=<task>`), and note it.
- **Time limit** (`checkpointed`, signal `USR1`): see §0. The estimate was wrong; report the new
  estimate.
- **A dependency never satisfied:** §3.
- **The wrapper refuses to run** ("uncommitted changes"): the campaign worktree has modified
  tracked files; do not edit worktrees.
