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

At 256 nodes the atoms take about 5.6 s of the 7.65 s plain sweep (467 200 atom moves at
48 µs on one thread); the dense transport modes and `tline` come next. One adaptive sweep
costs about 2.2 times a plain local sweep at 40 nodes × 6 h. A `kpeak` pass takes about
0.1 s per sweep at 40 nodes daily without the bias and about 1.8 s with the bias evaluation at
$\alpha = 1$.

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

**Compile times.** The six-family sweep compiles in 3–4 min on either backend; one family alone
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
