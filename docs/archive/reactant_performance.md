# What a Reactant.jl (XLA, GPU) port of the sampler could gain

A rough estimate, anchored on instrumented CPU runs and a small GPU prototype.
Prototype and scripts: `external/reactant-bench/` (own `Project.toml` with
Reactant; nothing under `src/` was changed). The GPU here is an RTX 3050 Ti
Laptop (4 GB, ≈ 190 GB/s, FP64 at 1/64 of FP32). H100 figures are
extrapolated (≈ 3.35 TB/s HBM, FP64 ≈ 34 TFLOP/s non-tensor and ≈ 67 on tensor
cores).

**Summary**
* At 256 nodes × daily, one chain: about **10–25× over 4 CPU threads**. It is
  latency-bound by the sequential rounds of the colour classes, so it is only
  about 1× a full 64-core node.
* With 64–256 chains batched at 256 nodes: about **50–100× over 4 threads**,
  and **5–10× over a 64-core node**. That rises to 30–60× with fused move
  kernels.
* Full Europe × hourly is throughput-bound and memory-limited, at ≤ 8–12 chains
  per 80 GB GPU. With localized moves the estimate is **10–20 s per sweep on one
  H100**, against ≈ 30 min on 4 threads and ≈ 2–3 min on a 64-core node. Fused
  kernels would bring it to 2–5 s.
* The move mix matters as much as the hardware. At 256 nodes, 10 % of the moves
  (dense xpair reroutes and coarse cell moves) take 80 % of the time, and their
  cost grows like N per move. Without localizing them, full Europe is out of
  reach on CPU and GPU alike.

## 1. Work per sweep (measured, CPU, 1 thread)

Measurements come from `external/reactant-bench/instrument.jl`, which wraps
`sample_line!` and `mv_eval!` at run time. The setup is SoC storage, renewables
as load, a heuristic start and default local families. Each line move is a
linked two-step block (SoC form).

| per move | 40 nodes × daily | 256 nodes × daily |
|---|---|---|
| coordinates (per step) | 85 045 (233) | 467 200 (1 280) |
| line moves per sweep | 133 k | 770 k |
| s/sweep at 1 thread (4 threads, niced) | 2.9 (≈ 0.8) | 50–56 (13.5) |
| µs per move, all-in | 21.7 | 65–72 |
| evaluations of ΔC(t): mean / median / p99 | 14.1 / 10 / 91 | 17.5 / 11 / 122 |
| primary breakpoints: mean / median / p99 | 18.9 / 6 / 160 | 72.8 / 6 / 940 |
| entries: mean / median / p99 | 7.0 / 3 / 40 | 24 / 3 / 256 |
| lines, nodes (= cells), secondary quantities: mean | 4.9, 9.4, 22 | 27, 26, 43 |
| share of time in the line sampler | 78 % | 87 % |

The 256-node run, split by move size:

| entries | share of moves | share of time | evaluations | breakpoints |
|---|---|---|---|---|
| 1–2 | 43.5 % | 4.2 % | 9.1 | 4.8 |
| 3–8 | 36.5 % | 7.2 % | 14.8 | 9.9 |
| 9–32 | 9.6 % | 8.3 % | 33.6 | 33 |
| 33–128 | 3.1 % | 20.8 % | 80.6 | 132 |
| > 128 | 7.3 % | 59.5 % | 32.6 | 821 |

**The CPU kernel is latency- and overhead-bound, not bandwidth-bound.**
* *Work per move.* A move touches about 50 items at 40 nodes and about 150 at
  256 nodes (entries, lines, nodes, cells, secondaries). It evaluates them 14–17
  times, at about 10 flops per item.
* *Achieved rate.* That is ≈ 10⁴ flops per move in 22 µs, ≈ 0.5 GFLOP/s per
  core, which is 1–2 % of FP64 peak.
* *Bytes per move.* About 0.6–2.5 KB per move, mostly L1/L2-resident across the
  evaluations.
* *Working set.* The state and caches of one chain at 256 × daily are about
  12–20 MB, so the whole working set nearly fits in L3. Sweep traffic is
  0.1–2 GB from cache, and DRAM bandwidth is irrelevant.
* *Where the time goes.* Indirections (`Vector{Vector}`, role branches),
  allocations in `walk!`/`sample_line!` (`filter`, `sort!`, `vcat`, a new
  `roots` array per call), dynamic dispatch on the mover, and the sequential
  breakpoint walk.
* *Small moves are slow too.* Even 1–2-entry moves cost ≈ 6 µs. A tight padded
  CPU rewrite would itself gain an estimated 3–10×. Keep this in mind when
  reading any GPU/CPU ratio.

**Sequential depth.** A sweep is 2 parity classes × (atoms + 4N extras per
step) sequential moves: ≈ 730 rounds at 40 nodes and ≈ 4 200 rounds at 256
nodes. Each round is a batch of T/2 = 182 steps (daily) or 4 380 steps
(hourly).

## 2. What maps to XLA and what does not

**Maps well:**
* The congestion ADMM (already dense and batched over steps).
* Frame projection (masked small SPD solves batched over steps).
* The cost evaluation of one move at fixed points (gather + elementwise +
  segment-sum).
* Sorting fixed-size arrays (`stablehlo.sort` along a dimension).
* Log-sum-exp, `cumsum` and inverse sampling.
* Counter-based RNG.
* Batching over the steps of a colour class, over chains, and over the nodes of
  an independent set.

**Maps poorly:**
* The data-dependent breakpoint *walk*: early exit at 40 nats and a variable
  number of evaluations.
* Variable-length moves (1 to 368 entries in one family mix).
* Dict- and push!-based move preparation (`prep_tpot!`, `prep_storage_ms!`,
  `prorata!`).
* Dense reroutes with a variable support.
* Scatter commits with conflicts. They are conflict-free only if the moves of a
  round have disjoint supports.

**Batched formulation (implemented in `external/reactant-bench/linesampler.jl`).**
A round has B moves, each with E padded entries (y₀, slope a, box [0, 1],
inside and violation slopes, injection κ, cell slot) and C padded cells
(S₀, n_c):

1. Primary points: 0 and the 2E clamp points (padding gives a = 0, and its
   points collapse onto 0, which makes zero-length segments of zero mass). The
   points are sorted per move.
2. Lin(t) and S_c(t) are evaluated at the 2E + 1 points, plus one point beyond
   each end, which gives the slopes of the two outer rays.
3. For every segment (2E finite segments and 2 rays) and every cell, the root of
   S_c is computed if it falls inside the segment. The C + 2 sub-points are
   sorted.
4. Each sub-segment's log-mass is computed exactly, with stable `logseg`; a ray
   has mass −c − log g. Then log-sum-exp, `cumsum`, the inverse index via
   `sum(cum .< u)` with a one-hot pick, and the truncated-exponential offset. A
   ray with g ≤ 0 means an improper line, and the move returns t = 0.
5. Commit: y += t·a and S_c += ΔS_c by gather/scatter.

The scheme is exact with no truncation; the CPU walk cuts at 40 nats. A
Kolmogorov–Smirnov test against quadrature passes (KS distance 0.0009 and
0.0012 with B = 2·10⁵, at β = 1 and β = 0.05; the 95 % threshold is 0.003;
`check_cpu.jl`).

**Cost of the naive dense version.** It is O(K · S) per move (K points times S
touched items) plus O(K · C²). That is fine for E ≤ 16. For the heavy moves
(K ≈ 800 breakpoints) it must be replaced by a **slope-jump form**:
* every separable term (unit, line, link) becomes (breakpoint, slope jump)
  pairs;
* every cell or storage-dynamics term is built by a segmented sort and scan per
  cell;
* the roots of each cell sum add jumps;
* one global sort and a prefix sum give ΔC at all points.

That costs O(K log K), with no walk and fixed shapes per size bucket.

**Measured on the RTX 3050 Ti (ms per round; the CPU column is the same array
code on 1 thread, which is a weak baseline):**

| precision, E/C | B = 182 | B = 4 096 | B = 65 536 (16 384 for FP64 E = 16) |
|---|---|---|---|
| FP32, 4/4 | 0.30 ms (1.7 µs/move) | 0.87 ms (212 ns) | 11.5 ms (**176 ns/move**) |
| FP32, 16/8 | 0.59 ms | 5.6 ms (1.36 µs) | 85.5 ms (**1.31 µs/move**) |
| FP64, 4/4 | 0.43 ms | 3.2 ms (782 ns) | 36.6 ms (558 ns) |
| FP64, 16/8 | 1.43 ms | 19.9 ms (4.9 µs) | 71.8 ms (4.4 µs) |

* *Compile time.* The first `@compile` took 56–96 s, mostly Reactant start-up.
  Each further shape compiled in under 2 s.
* *Latency floor.* There is a floor of 0.3–0.4 ms per host-dispatched round.
* *An unfused program.* At large B the naive program runs at only ≈ 15 GFLOP/s,
  because XLA writes the per-move intermediates (E × C × points arrays, sort
  inputs) to HBM. It is bandwidth-bound on intermediates, not on the state.
* *Device loop.* A device-loop variant (`@trace for` over rounds) failed to
  trace, with "aliasing pattern of loop-carried variables changed". The loop
  needs the state passed as fresh arrays each iteration. Not pursued, so the
  in-graph round latency is not measured. The expected figure for a
  device-resident loop on an H100 is ≈ 20–100 µs per round, from a few dozen
  kernels at ≈ 1–4 µs each.

**Extrapolation to an H100, FP64.** In the bandwidth-bound regime, take the
ratio ≈ 17× and FP64 doubling the bytes:
* naive XLA: ≈ **20 ns/move** for small moves (E ≤ 4) and ≈ 150 ns/move for
  E = 16;
* fused, one thread per move with the intermediates in registers: the HBM
  traffic is only the gather/scatter (≈ 0.3–1 KB per move), with 3–10 k flops
  and about 50 FP64 transcendentals. That gives an arithmetic intensity of
  ≈ 10 flop/B, near the H100's FP64 balance point, and **≈ 1–3 ns/move**.

Against the real CPU kernel, a small move costs ≈ 6–12 µs on one thread: 300–600×
per move against one core on the naive version, and 10–40× against a 64-core
node. Fused kernels in Reactant mean custom kernels traced into the program
(Reactant can embed CUDA.jl/KernelAbstractions kernels); that is to be checked
for the version in use.

## 3. Expected speed-ups

The CPU anchors are 4 threads niced, as measured, and a 64-core node estimated
at 0.7 parallel efficiency.

| case | CPU | H100, naive XLA | H100, fused kernels | limited by |
|---|---|---|---|---|
| 256 × daily, 1 chain, current schedule (4 200 rounds) | 13.5 s (4 thr), ≈ 1.2 s (64 cores) | 0.5–1.5 s: **10–25×** vs 4 thr, ≈ 1× vs node | 0.2–0.5 s | latency of the rounds (4 200 × 0.1–0.3 ms); heavy moves ≈ 0.2 s |
| 256 × daily, 1 chain, restructured schedule (≈ 800 rounds, §5) | — | 0.15–0.4 s | 0.05–0.15 s | latency, then heavy moves |
| 256 × daily, 64–256 chains | 13.5 s per chain-sweep (4 thr) | 0.15–0.3 s per chain-sweep: **50–100×** vs 4 thr, 5–10× vs node | 0.02–0.05 s: 30–60× vs node | heavy (dense) moves; memory is no issue (≈ 20 MB per chain) |
| full Europe (N ≈ 4 000) × hourly, current move mix | ≈ 90 thread-h per sweep (dense reroutes ∝ N per move) | hours | — | dense moves: localize first |
| full Europe × hourly, localized moves (3–5·10⁸ moves per sweep) | ≈ 30 min (4 thr), ≈ 2–3 min (64 cores) | 10–20 s: **100–200×** vs 4 thr, **8–15×** vs node | 2–5 s: 30–60× vs node | throughput and HBM; 5–8 GB per chain in FP64, so ≤ 8–12 chains per 80 GB |

**What limits it:**
* **Colour-class serialization.** Only the T/2 steps of one parity class are
  independent, and every move inside a step is sequential. At daily resolution
  that is 182-wide rounds, far too narrow for a GPU, so a single chain stays
  latency-bound. Chains, or spatial colouring (§5), give the width.
* **Padding waste.** Move sizes are heavy-tailed, with a median of 3 entries and
  a p99 of 256. Padding one round to its maximum wastes up to ≈ 100×. Rounds
  must therefore be family- and size-synchronous: the same atom index j at every
  step of a class, and buckets for the extras.
* **Memory per chain** at full Europe: v and Y each 1.4–2.8 GB, plus r, S and
  the availabilities, ≈ 5–8 GB in FP64. Many chains need many GPUs. FP32 is
  usable for the move-local ΔC, which is a sum of per-term differences, but not
  for the incremental caches.
* **Compile time.** 1–2 min the first time, then seconds per shape bucket.
  Shapes are static: E, C, the number of breakpoints and B per bucket. Retracing
  is needed only when T, N or the bucket set changes.
* **Kernel launch.** 0.3–0.4 ms per host-driven round was measured. The sweep
  must run as one device program (`@trace for` / `while`), with all rounds of a
  sweep in one executable.

**Congestion ADMM (analytic; the benchmark did not finish in the time box).**
A dense iteration costs 2T[(N + n_k)² + 2(n_l + n_k)(N + n_k)] flops: 0.14 GFLOP
at 256 × daily, so 56 GFLOP for 400 iterations. That is ≈ 0.4–0.6 s on 4 BLAS
threads and ≈ 3–10 ms on an H100 (≈ 50–100×). At full Europe × hourly, with the
incidence products done as sparse differences, the dense M⁻¹ product costs
2.8·10¹¹ per iteration, 10¹⁴ in total: 2–5 s on an H100, against ≈ 10 min on
4 threads and ≈ 1 min on a node. This piece ports as is.

## 4. Reactant's probabilistic-programming layer (`Reactant.ProbProg`)

From the tutorials (MCMC, Running and resuming chains, Sampling, the API page;
stable docs, Nov 2026):

* **Kernels:**
  - `mh`: Metropolis–Hastings that regenerates the selected sites from the
    prior;
  - `mcmc` with `algorithm = :HMC | :NUTS` on traced models;
  - `mcmc_logpdf`: HMC/NUTS on a user log-density;
  - step-size and mass-matrix adaptation;
  - `run_chain` with chunking, resumable through `MCMCState`.

  Gibbs, slice sampling, custom kernels or proposals, and kernel composition are
  not provided. The docs show alternating single-site `mh` calls and recommend a
  joint `select` instead. NUTS needs an explicit step size.
* **Models:** plain Julia functions with
  `ProbProg.sample(rng, dist; symbol)`. There are few built-in distributions
  (Normal, Exponential, LogNormal, Bernoulli). Custom samplers come with
  `logpdf`, `support` and `bounds`; the support is used to unconstrain the site
  for HMC. Traces are `Trace`/`Constraint`/`Selection` with addresses, flattened
  to tensors.
* **Compilation:** whole inference programs compile as one executable
  (`@compile optimize = :probprog`), and "inner loops run on device without
  Julia round-trips". Iteration counts are passed as `ConcreteRNumber`.
* **Many chains:** batching or vmap over chains is not documented.

**Implications for this sampler:**
* *Exact line moves.* Our exact piecewise-exponential moves, breakpoint
  handling, Metropolis-within-Gibbs Jacobian steps and colour-class schedules
  are not expressible as ProbProg kernels.
* *HMC/NUTS.* The potential is piecewise linear with slopes from 5 to 3 000 €/MWh.
  Gradients are piecewise constant, and the kinks make the energy error large
  unless the steps are tiny. That is what `src/hmc.jl` ran into, and HMC/NUTS
  does not replace the exact kernel.
* *What ProbProg is useful for:*
  - a pattern for whole-chain device compilation: the RNG threaded through,
    `@trace` loops, `ConcreteRNumber` counts, state kept as tensors;
  - `mcmc_logpdf` NUTS on the smooth blocks, such as the level-0 weather
    latents, or capacities with the bias at α = 1, as an optional extra kernel;
  - `mcmc_summary` for diagnostics.
* *The port itself.* It should be written in plain Reactant (traced arrays,
  `@compile`, `@trace for`), with an explicit chain dimension. ProbProg does not
  change the performance estimate. It lowers the integration cost only for the
  smooth blocks.

## 5. Recommendations

1. **Change the algorithm first; this pays on CPU too.**
   * *Localize the heavy moves.* Truncate the xpair reroute to a graph
     neighbourhood or a Haar-local PTDF; a truncated direction is still an exact
     line. Size cell moves by level.
   * *Make rounds family-synchronous.* Use one random order of atoms and families
     shared by all steps of a parity class. The order is still independent of v,
     so every kernel stays π-invariant.
   * *Add spatial colouring inside a step.* Unit, pair and fine-atom moves at
     the nodes of an independent set of the line graph (distance 2) touch
     disjoint terms, so they are conditionally independent and can be drawn
     jointly. This cuts the rounds per sweep from ≈ 4 200 to ≈ 500–1 000 at
     256 nodes and makes rounds 10–100× wider.

   Expected effort: 1–2 weeks. Expected gain: 2–5× on CPU already.
2. **Port the local kernel as padded size buckets, with chains as a batch
   dimension.**
   * *Data layout:* fields × max support × T × chains, plus unit → node, line →
     node and cell label arrays, all as static index arrays (no Dicts). Use
     counter-based RNG per (sweep, round, move). Keep the caches in FP64 and do
     the move-local arithmetic in FP32/FP64.
   * *Small moves (E ≤ 8):* the dense-evaluate sampler prototyped here.
   * *Large moves:* the slope-jump form (sort + segmented scan).
   * *One executable per sweep:* run all rounds in a device loop.
   * *Order of work:* first the 256-node daily case with 64–256 chains, where
     the gain is largest (50–100× over 4 threads). Then fuse the hot kernels,
     since naive XLA stays ≈ 10× off the hardware.

   Expected effort: 4–8 weeks for the local families, plus 2–4 weeks for the
   T2 compensation (kinks, Jacobian MH) and the storage multiscale moves.
3. **Port the array-first pieces as they are.** These are the congestion ADMM,
   the frame projection and the PCA (expected ≈ 50–100×, a few days). Keep the
   learned-mode updates (`adaptive.jl`, every 2^j sweeps) on the host. Optional:
   replace the walk outright with a fixed grid plus exact correction, or with
   fixed-size Metropolis-within-Gibbs proposals. Both have fixed shapes, but they
   lose the step-free property that makes the same sampler serve node and network
   moves. The padded exact sampler above keeps it at comparable cost, so it is
   the better choice.

**Assumptions:**
* full Europe ≈ 4 000 nodes with 5–10 coordinates per node and step;
* moves per sweep ≈ (coordinates + 4N) per step × T;
* the H100 is bandwidth-bound for the naive XLA version, with the ratio taken
  from the measured FP32 runs here (17× bandwidth, 2× for FP64 bytes);
* a 64-core node gets 0.7 of linear scaling;
* the round latency is 0.1–0.3 ms host-driven and 20–100 µs in a device loop
  (estimated, not measured).

## Files

* `external/reactant-bench/instrument.jl`: per-move statistics (run-time
  wrappers).
* `instr256.log`, `instr40flow.log`: the raw statistics.
* `linesampler.jl`: the padded batched exact line sampler, the same code for
  CPU arrays and Reactant.
* `check_cpu.jl`: the KS check against quadrature.
* `bench.jl`, `bench_line.log`: the GPU vs CPU timings. ADMM and the device-loop
  variant are not finished.
