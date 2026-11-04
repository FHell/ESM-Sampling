# Performance reference

This document collects the measured cost of the sampler: the work per sweep, the time per
move of each family, the time per sweep and how it scales with the network size and the
number of steps, the overhead of the model-completion terms, the device throughput, memory,
and estimates for larger cases. Every number states where and when it was measured; numbers
without a measurement are marked as estimates. What the moves do is in
[sampler.md](sampler.md) and [moves/README.md](moves/README.md); the device design in
[device.md](device.md).

## 1. Machines and conditions

All measurements are from 2026-11-03 on the development workstation: Intel Core i7-11800H
(8 cores, 16 hardware threads), 30 GB RAM, NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB, about
190 GB/s, FP64 at 1/64 of the FP32 rate). CPU runs used `nice -n 19` while the machine was in
use for other work, so times scatter by about ±10 %. **The GPU numbers are development numbers
on this laptop GPU**, whose FP64 rate is about 0.08 TFLOP/s; they show the port works and
where its time goes, not what a data-centre GPU would do. Setup, refit and compile times are
excluded unless stated.

The standard benchmark case (`scripts/bench_perf.jl`): heuristic start, levels
`[(10, 24), (N, 24)]`, SoC storage, renewables as load, sparse Haar angles, daily steps,
without the model-completion terms, on the 40-node (`model.h5`) and 256-node (`model256.h5`)
networks.

## 2. Work per sweep

| | 40 nodes × daily | 256 nodes × daily |
|---|---|---|
| operation coordinates per step | 233 | 1 280 |
| coordinates per sweep | 85 045 | 467 200 |
| line moves per sweep (atoms + $4N$ random moves per step) | 133 000 | 770 000 |
| evaluations of $\Delta C(t)$ per move: mean / median / p99 | 14 / 10 / 91 | 18 / 11 / 122 |
| share of the sweep time in the line sampler | 78 % | 87 % |

(Instrumented runs, 1 thread, `external/reactant-bench/instrument.jl`.) The CPU kernel is
latency- and overhead-bound, not bandwidth-bound: about $10^4$ flops per move at roughly
0.5 GFLOP/s per core, with a per-chain working set of 12–20 MB at 256 nodes daily that nearly
fits in the last-level cache. Move evaluation and line draws allocate nothing (0 B per move in
a tight loop).

## 3. Time per move, by family

µs per move at 1 thread, including the per-step overhead of the sweep (`bench_perf.jl fam`):

| family | 40 × daily | 256 × daily |
|---|---|---|
| `atom` | 14.8 | 47.9 |
| `unit` | 2.4 | 2.4 |
| `pair` | 3.4 | 3.6 |
| `xpair` (global re-routing) | 12.2 | 62.5 |
| `xlocc` | 6.9 | 7.4 |
| `xloc` | 4.9 | 4.9 |
| `potential` | 3.7 | 3.6 |
| `cellpair` | 4.0 | 5.3 |
| `cell` | 2.7 | 3.1 |
| `spair` | 5.1 | 5.7 |
| `tpot` | 7.7 | 8.2 |
| `tmode` | 34.5 | 275.8 |
| `tcut` | 15.8 | 34.3 |
| `tline` | 15.8 | 79.5 |
| `tmode0` | 25.7 | 378.6 |
| `tline0` | 11.3 | 62.4 |
| `tsoc` | 6.2 | 7.3 |

Local families cost the same at both sizes. The atoms (the coarse Haar atoms touch many lines
and nodes) and the dense families (`xpair`, `tmode`, `tline`) grow with $N$. A full transport
mode at 40 nodes touches 79 lines, 108 entries and 473 breakpoints and costs about 10 ms per
exact draw; its localized piece (`tcut`) touches 12 lines, 39 entries and 136 breakpoints.

## 4. Time per sweep

s per sweep at 4 threads (`bench_perf.jl sweep`):

| sweep | 40 × daily | 256 × daily |
|---|---|---|
| `local_sweep!`, plain families | 0.51 | 7.65 |
| `local_sweep!`, plain and flow families (the operation part of `flow_sweep!`) | 0.55 | 12.0 |
| `storage_ms_sweep!` | 0.072 | 0.55 |
| `storage_long_sweep!` (32 `slong` + 1 `slevel` per unit with ≥ 24 h; measured in the E21 setup, not with `bench_perf.jl`) | 0.10 | — |
| ... its `sxlong` phase (8 interval storage pairs per unit with ≥ 24 h, rounds of disjoint footprints; E27 setup) | 0.055 | — |
| `sxpair` in `local_sweep!` at `storage_xpair` = 0.1 (about 3 650 moves; E27 setup) | 0.035 | — |

One `slong` move costs about 30 µs for hydro, 50 µs for the large PHS units and 130 µs for H₂
at 40 nodes daily (one thread; the interior steps are evaluated only where they leave the box),
against 1.3 ms for an `L = 128` plateau of `storage_ms_sweep!`.

At 256 nodes the atoms take about 5.6 s of the 7.65 s plain sweep (467 200 atom moves at
48 µs on one thread); the dense transport modes and `tline` come next. One adaptive sweep
costs about 2.2 times a plain local sweep at 40 nodes × 6 h.

**Capacity and weather passes with the bias** (40 nodes × 6 h, all model terms, 219 investable
units, 4 threads, `bias_sample_steps` = 4096): a `kscale` pass takes 0.15 s at $\alpha = 1$
against 0.06 s at $\alpha = 0$, a `kpeak` pass 0.53 s against 0.54 s, a weather sweep at
`fine_weather = 0.05` 3.6 s against 3.8 s. The bias change comes from the incremental bias cache
(two table lookups per step and sample step); building the cache and a global evaluation of
$\hat b$ cost about 4 ms each.

**Model-completion terms** (all on: losses, soft limits, lossy links, vintages, commitment),
40 nodes × 6 h, 8 threads: a fine step takes 9.2 s against 4.0 s for the plain model, and a
weather sweep at `fine_weather = 0.05` 2.3 s against 1.2 s. Most of the difference is that
moves touching these terms use the plain evaluation instead of the recorded-term evaluation
([roadmap.md](roadmap.md) §4).

## 5. Scaling with nodes and steps

- **Steps.** Moves per sweep are proportional to $T\cdot$(coordinates per step $+\,4N$), and
  the per-move cost does not depend on $T$, so the sweep time is linear in $T$ at fixed
  network.
- **Nodes.** Coordinates per step grow linearly with $N$. The per-move cost of the local
  families is constant, that of the atoms and the dense families grows with $N$: from 40 to
  256 nodes the plain sweep grows 15 times (0.51 to 7.65 s) for 6.4 times the nodes. The dense
  families must be localized before full-Europe sizes.
- **Model completion** raises the coordinates per step at 256 nodes from 1 280 to 2 040
  (commitment fields, brownfield fleets).

**Estimate** (not measured): 256 nodes × hourly with all model terms, from the 256-node daily
anchors scaled by 24 in $T$ and 1.6 in coordinates per step: 5–8 min per operation sweep on 4
threads with cached evaluation, up to about twice that with the plain evaluation of the
completion terms.

## 6. Mixing per sweep

These are properties of the move set, measured at 40 nodes unless stated, $\alpha = 0$, fixed
$(k,x)$:

- From the level's own LP solution, the operational excess reaches its stationary level in
  about 25 local sweeps at 24, 12 and 6 h steps.
- From far starts at 6 h, after about 33 sweeps the excess over the equipartition prediction
  is about 61 (local), 51 (flow with share-preserving compensation), 38 (adaptive) and 21
  (adaptive with learned compensation).
- At 256 nodes daily a flow sweep does about the work of two plain local sweeps, at 1.6 times
  the cost per sweep (§4).

## 7. Device

**Throughput of the plain device families** (`device/scripts/throughput.jl`, all model terms
on, GPU: one compiled `run_sweeps` call with Reactant's RNG; CPU: `local_sweep!` with the plain
families on 8 threads). The move sets differ, so compare line moves per second:

| model | chains | GPU s/sweep | GPU line moves/s | CPU s/sweep | CPU line moves/s |
|---|---|---|---|---|---|
| 40 × 6 h ($T$ = 1460) | 4 | 3.75 | $4.8\cdot10^5$ | 3.34 | $1.8\cdot10^5$ |
| 40 × 6 h | 2 | 1.88 | $4.8\cdot10^5$ | | |
| 256 × daily, rounds ≤ 8 192 sites | 1 | 2.87 | $4.0\cdot10^5$ | 12.1 | $8.6\cdot10^4$ |
| 256 × daily, 4 chains, rounds ≤ 32 768 sites | 4 | out of memory (4 GB) | | | |

Per family at 40 × 6 h with 4 chains: `gen` $1.0\cdot10^6$, `uc` $1.5\cdot10^6$, `pair`
$2.7\cdot10^5$, `soc` $1.1\cdot10^6$, `link` $4.0\cdot10^6$, `theta` $1.2\cdot10^5$ line moves/s.
The angle family takes half the sweep at 40 nodes, because the dense form evaluates every
hinge knot of every incident line (about 150 primary points per site). The moves per second do
not grow with the number of chains: the laptop GPU is saturated by its FP64 arithmetic.
Reactant's CPU backend runs the 40-node sweep at $5.4\cdot10^4$ line moves/s.

**Profile of a device run** (2026-11-04, `device/scripts/profile_device.jl` and
`profile_families.jl`, `configs/device40_6h_gpu.toml`: 40 nodes × 6 h, $T$ = 1460, 4 chains,
α = 0, no UC, no losses, GPU). Per call of 10 operation sweeps: operation sweeps 22.6 s,
capacity pass 0.04 s, host draws 0.001 s, download 0.05 s, host checkpoint 0.28 s, so the
operation sweeps are 98 % of the run. Per family, seconds per sweep (4 chains) and µs per line
move:

| family set | family | s/sweep | µs/move |
|---|---|---|---|
| `FAMILIES` | `gen` | 0.425 | 1.04 |
| | `pair` | 0.695 | 3.72 |
| | `soc` | 0.573 | 0.84 |
| | `link` | 0.012 | 0.25 |
| | `theta` | 0.552 | 2.42 |
| | sum | 2.26 | |
| `FAMILIES_FAST` | `gen0` | 0.047 | 0.12 |
| | `pair0` | 0.054 | 0.29 |
| | `soc` | 0.200 | 0.29 |
| | `link` | 0.011 | 0.24 |
| | `theta3`, `theta5`, `theta` | 0.048, 0.137, 0.069 | 0.64, 1.17, 1.96 |
| | sum | 0.57 | |
| compensated | `spair` | 1.00 | 1.47 |
| cells | `cell1` | 0.119 | 20.4 |
| | `cellpair1` | 0.274 | 47.0 |
| | `cell2`, `cell3` | 0.162, 0.028 | 2.78, 0.12 |
| | `cellpair2`, `cellpair3` | 0.260, 0.027 | 5.57, 0.16 |
| compensated, dense, capped at 256 MB per round | `spair0` | 0.425 | 0.62 |
| | `tsoc` | 2.46 | 3.60 |
| | `tpot` | 17.7 | 77.5 |
| storage blocks, dense, capped | `sblk2`, `sblk4` | 4.10, 5.07 | 12.0, 29.9 |
| | `sblk8`, `sblk16` | 8.12, 15.2 | 95.8, 360 |
| slope-jump form (ad73f13) | `tsoc` | 3.41 | 4.99 |
| | `tpot` | 8.26 | 36.3 |
| | `sblk2`, `sblk4`, `sblk8` | 2.79, 2.10, 1.82 | 8.19, 12.4, 21.5 |
| | `sblk16`, `sblk32` | 1.56, 1.60 | 37.0, 75.9 |
| | `cell1`, `cell2` | 0.050, 0.133 | 8.57, 2.28 |
| | `cellpair1`, `cellpair2` | 0.079, 0.184 | 13.6, 3.93 |

With every family of the run set (`device_families`) a sweep costs about 23 s in the slope-jump
form (ad73f13, with `sblk32` added to the set), against about 54 s in the dense form at 4fd3d2c.
In the dense form `tpot` and the storage blocks took 50 s: their cost per site grows with
points × terms (`tpot` 520 points × 87 columns, `sblk16` 1403 × 139), and the memory cap
(`DENSE_BUDGET`, 256 MB) split their rounds into many small ones (`tpot` 312 rounds of 183
sites). In the slope-jump form `tpot` (8.3 s) and `tsoc` (3.4 s) lead; their most expensive
kernel is the payload sort of the (point, jump) entries (`sort_0_1`, 1.2 s per sweep in `tpot`
at 33 % occupancy). `tsoc` is slower on the GPU than in the dense form (3.41 against 2.46 s),
although the static choice (`plan(...; jump=:auto)`, by the dense/slope-jump entry ratio) picks
the slope-jump form for it: on the CPU backend it is 2.6× faster.

**Slope-jump form** (§3 of [device.md](device.md); Reactant's CPU backend, `configs/device40.toml`:
40 nodes daily, $T$ = 365, 4 chains, 4 threads; one family alone, seconds per sweep, dense →
slope-jump): `tpot` 53.6 → 13.8, `sblk8` 27.0 → 1.70, `sblk16` 43.5 → 1.39, `sblk32` 71.5 →
1.16; `tsoc` 11.3 → 4.4, `sblk2` 10.7 → 3.4, `sblk4` 15.4 → 2.2, `cell1` 0.48 → 0.04,
`cellpair1` 1.85 → 0.08, `cell2` 0.67 → 0.17, `cellpair2` 2.20 → 0.34 (the second group measured
before the last round-off changes of the slope-jump values, which add two prefix sums); `spair0`
0.79 → 1.41, so it stays dense. The dense rows are at the 256 MB cap (`tpot` 87 rounds of 182
sites); the slope-jump rounds are not split.
In `tpot` the commitment columns (48 per site, 19 local points each) and the eight balance slots
(52 points each) take 0.15 s each of the 0.45 s of a round, the payload sort of the 1 476
entries 0.07 s.

The rows are for the dense sampler with the term axis (static zero slopes, per-quantity root
counts). Before it, the `FAMILIES_FAST` sweep took 1.00 s (`soc` 0.58 s, `pair0` 0.070 s,
`theta5` 0.155 s) and `spair` 2.55 s; the family rows of `FAMILIES` are from that version. The
first family profiled in a process runs up to 1.4× slower kernels (GPU clock warm-up); the
`gen0` row is from a second process. The remaining small kernels with low occupancy are a sort
over a 2–4-element axis per site (about 5 ms per sweep in `gen0` and `pair0`) and the
categorical draw over the points (`triton_softmax`, 17 % occupancy, 5–10 ms).

The whole device loop with `FAMILIES_FAST` (`run_device`, same configuration): 6.2 s per call
of 10 operation sweeps, of which the operation sweeps 5.77 s, capacity pass 0.15 s, download
0.05 s, host checkpoint 0.24 s (before the term axis 9.8 s per call; with `FAMILIES` 22.6 s).

In every family the most expensive kernels are the potential at all points of the dense form
and the secondary quantities at the extended primaries. Both are FP64-arithmetic bound on this
card (about 15 GFLOP/s achieved), so the cost of a site is about points × terms. Without fleet
commitment the generator sites of `FAMILIES` carried two identically zero UC terms (32 points
instead of 8), which `:gen0`/`:pair0` remove.

**Compile times.** Reactant's compile of the four-chain operation program at 40 × 6 h: 160 s,
of which tracing 130 s, MLIR passes 23 s and XLA 7.5 s; the capacity pass 100 s (tracing 90 s).
Most of the tracing time is Julia's own compilation of the tracing code, which a process does
once: traced a second time in the same process (CPU backend, `@code_hlo`), the capacity pass
takes 7.5 s instead of 118 s and the operation program 35 s instead of 124 s; a new closure over
the same methods (an edited driver argument) 7 s. Iterating on the moves in one persistent
session therefore costs seconds per recompile for the capacity pass. The remaining 35 s of the
operation program grow with the number of terms of a family, because the term loops are
unrolled at trace time. The six-family sweep compiles in 3–4 min on either backend; one family alone
in 4–40 s; the peak-set capacity program in about 75 s. Every fresh Julia process adds 1.5–2
min of start-up and package loading.

**Padded line sampler prototype** (`external/reactant-bench/linesampler.jl`, laptop GPU,
time per move at the largest batch): FP32 0.18 µs (4 entries, 4 cells) and 1.3 µs (16/8);
FP64 0.56 µs (4/4) and 4.4 µs (16/8). The naive XLA program is bound by the bandwidth of its
per-move intermediates (about 15 GFLOP/s); there is a floor of 0.3–0.4 ms per host-dispatched
round, which the device loops avoid.

**Estimates for an H100** (not measured; bandwidth ratio of about 17, FP64 at half rate): the
present program 30–100 times faster than on the laptop GPU before fusion; about 20 ns per
small move with naive XLA and 1–3 ns with fused kernels; at 256 nodes daily with 64–256
chains, 50–100 times the throughput of 4 CPU threads; full Europe (about 4 000 nodes) ×
hourly with localized moves, 10–20 s per sweep (2–5 s fused) against about 30 min on 4
threads, at most 8–12 chains per 80 GB.

**First H100 measurement** (campaign H0, PIK cluster, job 2902915, 2026-11-04; `profile_families.jl`
with `FAMILIES=fast`, `configs/device40_6h_gpu.toml`: 40 nodes × 6 h, $T$ = 1460, 4 chains, FP64,
the code of the term-axis/slope-jump version as on the laptop): the seven fast families together
0.025 s per sweep, `soc` the most expensive at 0.0087 s, against 0.57 s and 0.20 s on the laptop
GPU, about 23 times faster. Compile: 88 s for the first family (including the one-off start-up
compilation), 4–12 s for each further family (laptop: 113–147 s and 4–59 s). The node's driver
(580.126, CUDA 13.0) runs Reactant's CUDA 13.1 runtime. The full run set (compensated families,
blocks, cells) and batch scaling are campaign H5.

## 8. Offline parts and memory

- **Setup at 256 nodes**: `build_energy_model` with the completion options about 20 s. The
  congestion ADMM of the transport basis: about 56 GFLOP for 400 iterations at 256 × daily,
  0.4–0.6 s on 4 BLAS threads (analytic estimate).
- **Reference LP**: about 60–80 s at 40 nodes × daily; with the relaxed unit commitment 312 s at
  10 cells × daily. Large models use the heuristic start.
- **State size at 256 nodes** with the completion terms (2 040 coordinates per step): one
  Float64 state is 6 MB daily, 24 MB at 6 h and 143 MB hourly; the finest weather state at
  hourly resolution is 58 MB. A run holds a few tens of such copies, 3–8 GB at hourly
  resolution (estimate).
- **Device memory**: a 256-node daily chain needs about 20 MB of state, but the round
  workspace is about $C\cdot B\cdot K$ doubles (chains × sites × terms): on the 4 GB GPU four
  chains did not fit with rounds of 32 768 sites and need smaller rounds (at most 8 192 sites)
  or rounds tiled by chain (not measured). Full Europe hourly needs an estimated 5–8 GB per
  chain in FP64.
