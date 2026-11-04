# Ledger of experiments

One row per experiment: the question, the setup, the result in one line, and where the full
record lives (notes, logs, commits). Results that changed the system are described in
`docs/system/`; this ledger records how they were obtained. Add a row when an experiment is
planned, and update it when it has run. Outputs under `results/` are not versioned.

## Run

| id | date | question | setup | result | record |
|---|---|---|---|---|---|
| E1 | ≤ 2026-11-03 | earlier development experiments (toy MLDA, far starts, kernels, pairs, scaling, prices) | various | MLDA abandoned; flow and adaptive kernels; see the notes | [archive/overnight_results.md](archive/overnight_results.md), [archive/toy_results.md](archive/toy_results.md), [archive/perf_notes.md](archive/perf_notes.md); logs `results/exp_*.log` |
| E2 | 2026-11-04 | where does a device run spend its time | `profile_device.jl`, `configs/device40_6h_gpu.toml`, GPU | operation sweeps 98 % of a call; capacity pass 0.04 s | [system/performance.md](system/performance.md) §7 |
| E3 | 2026-11-04 | cost per device family; effect of structure-specialised families, the term axis, memory caps, the slope-jump form | `profile_families.jl`, same config, GPU | fast set 2.26 → 1.00 → 0.57 s/sweep; full run set 54 → 23 s/sweep; `tsoc` slower in the slope-jump form on the GPU | [system/performance.md](system/performance.md) §7; `results/profiles/`, `.claude/worktrees/gpu-timing/results/profiles/` |
| E4 | 2026-11-04 | first compile against retrace of the device programs | Reactant CPU backend, `@code_hlo` twice in one process | capacity pass 118 → 7.5 s, operation 124 → 35 s: one-time Julia JIT | [system/performance.md](system/performance.md) §7 |
| E5 | 2026-11-04 | end-to-end device loop | `run_device.jl`, same config, GPU | 22.6 → 9.8 → 6.2 s per call of 10 sweeps | [system/performance.md](system/performance.md) §7 |
| E6 | 2026-11-04 | is the multi-chain screening at α = 0 measuring mixing | `chains40.toml`, 600 sweeps, 4 chains | no: a transient; storage runs to its box, weather tilted up to 18 sd | [mixing_diagnosis.md](mixing_diagnosis.md) §2 |
| E7 | 2026-11-04 | why capacity moves do not move capacities | conditionals and mean forces at the E6 states | renewables confined to about 4 MW by the node balance; storage and gas ratchet; forces 3e4 nats per unit log-scale | [mixing_diagnosis.md](mixing_diagnosis.md) §3 |
| E8 | 2026-11-04 | capacity moves with their operation (`kop`) | prototype, then `src/kop.jl`; `chains40`, α = 0, equal CPU | transient about 2× faster per CPU second; merged, default off | [mixing_diagnosis.md](mixing_diagnosis.md) §4, §8 |
| E9 | 2026-11-04 | capacity transport pair | prototype | median step 2.5 MW: no gain (negative) | [mixing_diagnosis.md](mixing_diagnosis.md) §4.3 |
| E10 | 2026-11-04 | weather mixing under the prior, at α = 0, and of the regimes | ESJD lower bounds; fine-level α = 0 runs; FFBS | z0 load means need ≥ 1.6e5 sweeps; weather moves cost-confined at α = 0; regimes are a function of z0 | [mixing_diagnosis.md](mixing_diagnosis.md) §5 |
| E11 | 2026-11-04 | joint low-frequency z0 block moves | `src/weather_blocks.jl`; prior ESJD; cost at 40 × 6 h | load annual/seasonal means 58–136× per CPU second; a 365-day block costs 35 s at 6 h → window cap 600 steps | [mixing_diagnosis.md](mixing_diagnosis.md) §9 |
| E12 | 2026-11-04 | storage multiscale moves in the `local` kernel; `kpeak` reversibility fix | `chains40` | H₂ SoC relaxes (0.07 → 0.3); `test_exact.jl` cannot see the `kpeak` violation | [mixing_diagnosis.md](mixing_diagnosis.md) §6 |
| E13 | 2026-11-04 | is the CPU line walk exact on storage lines | midpoint linearity of the potential between walk points, `haar_poc` level 1 | no: 13–18 % of storage lines wrong by up to 262 nats; fixed (71f5720), test added | commit 71f5720; [system/sampler.md](system/sampler.md) §3.2 |
| E14 | 2026-11-04 | is the level-general bias stable at α = 1, and with which safeguards | `complete40` copies, 2 chains, 16 + 10 sweeps (cold start, warmup0 = 400, safeguard variants B–E) | runaway without safeguards; bounded with them, weight ESS 1 | `results/stability/`; [system/sampler.md](system/sampler.md) §7.1, [system/roadmap.md](system/roadmap.md) §1 |
| E15 | 2026-11-04 | α = 1 with the full stack | `chains40`, warmup0 = 400, 4 chains, 40 + 40 sweeps, `kop` off/on | bounded; bias pays one chain to shed; solar 6–9 sd high; R̂ 3–5; weight ESS 1; total costs 1.3–3.2e11 €/a across chains | [mixing_diagnosis.md](mixing_diagnosis.md) §10; `results/mixing/a1/` |
| E16 | 2026-11-04 | fresh-eye review of the learned bias at α = 1, and of tempering with a shared MLP bias | instrumented rerun of E15; score series at fixed (x, k); SGLD prototype | construction sound but needs 5–10 orders more score precision than available; high chain held by a box defect; Langevin step on capacities recommended | branch `worktree-agent-af45abb1d0d9c1d90` (72e12c2), its `results/review/` |

## Running and planned

| id | status | question | design | record |
|---|---|---|---|---|
| P1 | running | does a Langevin step on the capacities, driven by the sampled scores, bring chains from different starts together | `chains40`, 400+ sweeps from low and high starts; one weather year, then 4 replicas; stationary condition ⟨r·s_k⟩ = β_k∂K per carrier | agent branch (to be merged) |
| P2 | running | fix of capacities stuck above their box (`kscale`, `kpeak`) | regression test: a state above the box must move toward it | agent branch |
| P3 | running | how many operation sweeps does the dispatch need after a fresh weather draw | swap the weather of an equilibrated state, follow the cost to its plateau; daily, then 6 h | agent branch |
| P4 | running | integrated autocorrelation times of the operation at fixed (k, x): capacity scores per carrier, opex, shed, stored energy per region and quarter; slow space-time storage directions | `chains40`, E15 end states (heuristic and low chains), 1000 operation sweeps each | this ledger, when done |
| P5 | planned | `tsoc` dense against slope-jump on the GPU; `tpot` without the padded commitment columns | `profile_families.jl` | [system/performance.md](system/performance.md) §7 |
| P6 | planned | sensitivity of the exactness tests | longer reference runs, references equilibrated in the storage and dump directions | [system/roadmap.md](system/roadmap.md) §1 |
| P7 | planned | sweeps to equilibrium against network size; wall time to stationarity against the finest step; memory for full Europe hourly | [system/roadmap.md](system/roadmap.md) §7 | — |
| P8 | prepared | first full-resolution run, 256 nodes hourly with ERA5 | `configs/full256_era5.toml` | [archive/full_resolution_run.md](archive/full_resolution_run.md), [system/roadmap.md](system/roadmap.md) §6 |
