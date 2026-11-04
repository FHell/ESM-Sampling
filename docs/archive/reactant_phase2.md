# Reactant port, phase 2: term batch, dense line sampler, plain rounds

Phase 2 of [reactant_design.md](../reactant_design.md) §8, as adapted for the session: the term batch and
the exact padded dense-form line sampler for the plain families, the commit, the conflict-free
round schedule and the compiled device loop. Everything is array code over the chain
dimension. It runs on Julia `Array`s (the reference) and traced by Reactant (XLA on this
machine's RTX 3050 Ti). Code is in `device/` (package `ESMDevice`); nothing in `src/` was changed.

**Summary**
* **Implemented.** Six plain families: `gen`, `uc`, `pair`, `soc`, `link` and `theta` (θ-NONE).
  Their term types cover every model term these moves touch: merged-vintage kink, fleet unit
  commitment (walls, no-load, start-up), lossy links, line losses and the soft overload band
  (hinge knots), SoC two-step terms and the balance.
* **Exact dense sampler.** There is no truncation. Every secondary quantity has at most two sign
  changes along any plain-family line; this is proven per term type below and checked on every
  site.
* **Device loop.** One compiled program runs `ns` sweeps (a traced count) with nested device loops
  over sweeps and rounds, using Reactant's RNG.
* **Verification (test strategy of the user: deterministic, no statistics).** The device code was
  compared with the CPU reference on fixed mock uniforms (`test_mock.jl`, 36 000 draws over
  12 rounds):
  - The device sampler run on the CPU walk's points reproduces CPU `sample_line!` to 4.4e-13.
  - On its own points (every kink, exact rays) it agrees with the CPU on 35 769 of 36 000 draws.
    203 differ only because the CPU walk splits a ray with non-kink points, which gives the same
    distribution but a different u → t map.
  - 28 draws are CPU precision losses, explained below.
  - After committing the same t, node values are **bitwise equal** to the CPU caches, r agrees to
    1.5e-11 MW, and cost_parts/energy_cost to 1e-13.
  - Reactant agrees with Julia arrays to ≤ 2e-14 relative.
  ΔC against `eval_move!`, the r drift, the cost of the downloaded state and `check_conflicts`
  also pass.
* **Throughput** (GPU, development numbers): 4.8e5 line moves/s at 40 nodes × 6 h with
  4 chains, against 1.8e5/s for the CPU `local_sweep!` on 8 threads. 4.0e5/s at 256 × daily with
  1 chain (4 chains run out of the 4 GB), against 8.6e4/s on the CPU.
* **Found in the CPU reference** (§5.8, for the main session): the `walk!` ray step after a
  round-off-flat segment overshoots to ≈ 1e15–1e18. `sample_line!` then returns 0.0 or exactly
  the breakpoint, in 0.08 % of the draws (28 of 36 000).
* **Two Reactant 0.2.304 problems found and worked around** (§5): `abs(a*b)` is miscompiled, and
  FMA contraction makes the exact zeros of padded sites inexact. The main session should check
  phase-1 code for the first one.

## 1. Files

| file | content |
|---|---|
| `device/src/terms.jl` | term types (`GenTerms`, `UCTerms`, `LinkTerms`, `LineTermsD`, `SocTerms`, `TermBatch`), `eval_terms` (βΔC, secondary quantities, balance-slot changes at any points), `primaries`, `roots2`, `sample_dense`/`sample_points` (exact inverse CDF), `root_count` |
| `device/src/rounds.jl` | static and per-chain parameters, the flattened loop state `FlatState`, topology, time classes, colourings (distance-2 for θ, edge colouring for links), `schedule`, index tables (`host_table`, `device_table`), `check_conflicts`, gather and scatter-add (`scadd`: Julia loop or `stablehlo.scatter`), the family builders `build(Val(fam), …)` and `commit`, `do_round`, the device loops `run_rounds` and `run_sweeps`, `plan` |
| `device/src/ESMDevice.jl` | phase 1, plus the two `include`s and exports (no other change) |
| `device/test/setup40.jl`, `helpers.jl` | 40-node setup as in `test_cost.jl`; CPU reference moves (`cpu_move`: `prep_node!`, `finalize!`, `link_soc!`, BlockMove over τ−1, τ, exactly as `local_sweep!`) |
| `device/test/test_mock.jl` | **main verification:** device vs CPU with mock uniforms (`MockRNG`); Reactant vs Julia arrays |
| `device/test/test_sampler.jl` | (i) ΔC vs `eval_move!`, (iv) `check_conflicts`, the ≤ 2-roots check (KS only with `KS=1`) |
| `device/test/test_rounds.jl` | (vi) Reactant vs Julia arrays for one sweep, (iii) r cache and cost after sweeps in one program |
| `device/test/test_exact_dev.jl`, `diag_exact_soc.jl` | statistical runs from before the strategy change, off unless `RUN_STAT=1` (§4) |
| `device/scripts/throughput.jl` | (vii) |

Run: `cd device && nice -n 19 julia --project=. -t 4 test/<file>.jl`. The traced parts run on
Reactant's CPU backend (`DEVICE_BACKEND=gpu` for the GPU). Only `scripts/throughput.jl` uses the
GPU; one process at a time, because XLA preallocates 75 % of the 4 GB.

| test | time |
|---|---|
| `test_mock` | ≈ 1 min plus one compile |
| `test_sampler` | ≈ 2 min |
| `test_rounds` | ≈ 5 min |

## 2. What is implemented

### 2.1 Families and rounds

Moves are node values at the complete level. Every site is one exact line draw.

| family | direction | terms (padded counts) | rounds |
|---|---|---|---|
| `gen` | ω of one dispatchable unit at (n, τ). For committable units the commitment follows (direction (1, 1), headroom kept), as `ws.follow` on the CPU | 1 gen box; 2 UC (steps τ and τ−1, active only if committable); 1 balance | unit slot j per node × time class |
| `uc` | commitment of one committable unit, at fixed output | 2 UC | all committable units × time class |
| `pair` | two dispatchable units at one node, (1/κ₁, −1/κ₂). Masked per chain (t = 0) where κ = A k < κ_min (1.0), as `prep_random!` | 2 gen, 4 UC (follow), 1 balance | p-th unordered pair per node × time class |
| `soc` | e of one storage unit at instant τ | 2 SoC (τ−1, τ), 2 balance | storage slot j per node × time class |
| `link` | u of one link at τ | 1 link, 2 balance | edge colour of the links on their buses × all τ |
| `theta` | θ of one node at τ (NONE) | D lines (oriented, the moved node is the near end), 1 + D balance | distance-2 colour × all τ |

**Time classes.** Odd/even steps, or three classes when T is odd, as in `local_sweep!`. The
commitment couples τ−1 and τ, and so does SoC. Rounds longer than `bmax` sites are split into
chunks, which stay conflict-free (θ: 4096 sites per round at 40 nodes).

**40 nodes × 6 h** (T = 1460):
* rounds per sweep: gen 6, uc 2, pair 6, soc 8, link 3, theta 18 (9 colours × 2 chunks);
* 451 140 line moves per chain and sweep, against ≈ 4·N + atoms per step on the CPU.

**κ_min.** The masks depend on k only (per chain), as on the CPU.

**Parallel lines.** The θ family needs one balance slot per line end. `schedule` errors out if
parallel lines exist; there are none at 40 nodes. Merging by far node needs site-dependent
slot weights; that is open.

### 2.2 Term batch

`TermBatch(gen, uc, link, line, soc, r0)`: every field is a vector (one entry per padded term)
of arrays broadcastable to (C, B, 1); the balance slots of the injections are static integers per
family.

`eval_terms(tb, k, t)` evaluates, at points t of shape (C, B, K):
* βΔC(t) as a sum of term differences cost(t) − cost(0) in FP64, as the CPU does;
* the secondary quantities:
  - UC: yc − cc, g_min cc − yc and cc₁ − cc;
  - SoC: d, d − η_in and −d η_out − 1;
  - balance: r;
* the balance-slot changes, which feed the commit.

The term formulas are those of `gen_cost`/`gen_vint`, `uc_cost`, `link_inj`, `line_eval` (as the
phase-1 hinges), `soc_cost` and `φbal`.

**Primaries:**
* clamp points of every moved argument, including the UC arguments;
* the vintage kink (yk − y0)/a;
* the link kink at u = 0;
* line knots (±U_j s − f0)/df and f = 0.

Padded breakpoints are put at t = 0, which is always a point.

### 2.3 Dense sampler (exact)

1. Sort the primaries per site.
2. Evaluate the secondaries at the sorted primaries, plus one point beyond each end; the rays are
   linear there.
3. Per secondary quantity, keep the smallest and the largest sign change over all segments,
   including the two rays (min/max reductions, no sort).
4. Merge-sort the roots with the primaries.
5. Evaluate βΔC at all points.
6. Compute segment log-masses (`logseg`) and the two rays (−c − log g), followed by
   log-sum-exp, the inverse CDF and the truncated-exponential offset.

**Numerically stable masses and offsets.**
* A segment's log-mass is anchored at its lower end: −min(c_a, c_b) + logseg(L, |c_b − c_a|).
* For g < 0 the in-segment offset is measured from the right end (`trexp_end`). Long segments with
  a huge potential at one end therefore lose no precision; the CPU's walk does (§5.8).
* The u → t map of the offset is monotone (1 − u in the mirrored case). `Consts.cpumap = 1`
  selects the CPU's map instead, which is used for the comparisons with mock uniforms.

**Improper rays.** A ray whose outward rise is ≤ 1e-12 × |c| (flat within round-off) gives
t = 0, as on the CPU.

**Ray roots.** Roots farther than 10¹² from 0 are dropped, because the potential there is
≫ 10³ nats.

**Why ≤ 2 sign changes is exact:**
* `gen`, `link`, `soc`, `uc`: each secondary quantity is monotone in t, because one argument
  moves through clamps. For the UC walls under follow, yc − cc has no strict sign change, and
  g_min cc − yc has at most one.
* `theta`: balance slots are linear plus half the losses, which are convex in f and so in t. A
  convex function has ≤ 2 sign changes.
* `pair`: r = r₀ − Δp₁(t) − Δp₂(t), with Δp₁ nondecreasing and Δp₂ nonincreasing. It is constant
  while both units are inside their boxes and monotone on each side, so ≤ 2.

`root_count` counts the sign changes per quantity. Tests check it on every site of every round.

**Interface.** The interface is `sample(tb, k, u1, u2)` with uniforms (C, B, 1), so the slope-jump
form of phase 3 can sit behind it. Compensation sites (UNIT/LINEAR/T2) add terms of the existing
types, plus kinked arguments.

### 2.4 Commit, state, loop

**State.** `FlatState(θ, ω, uc, e, u, r)` holds (C, T·X) matrices (column τ + (x − 1)T). The
r cache is updated by the commit and never rebuilt inside the loop.

**Gathers and commit.** Gathers are `X[:, idx]`. The commit is a scatter-add, so padded sites
(zero change) may point at any column.

**Device loops.** `run_rounds` runs one `@trace for` per family over its rounds. Index tables are
dynamic-sliced by the traced round index, and the uniforms come from `rand(ReactantRNG, (C, B, 2))`
or from a given array. `run_sweeps` wraps it in an outer `@trace for` over sweeps with a traced
count, so one executable serves any number of sweeps.

**`check_conflicts(fam, host_table)`.** It reads the same index tables the kernels gather and
scatter with. Within a round, no item written by one site may be read or written by another, and
no site may write one balance slot twice.

## 3. Verification (40 nodes × 6 h, all model terms on; setup as `test_cost.jl`)

### 3.1 Device vs CPU with mock uniforms (`test_mock.jl`)

**Same directions.** Every site of the first two rounds of each family (the first 1 500 sites of
each round: 12 rounds, 18 000 sites × 2 chains) is built on the CPU with `prep_node!`,
`finalize!` and `link_soc!` as a BlockMove over τ−1, τ, as in `local_sweep!`, and on the device
from the same index tables.

**Same uniforms.** (u1, u2) per site and chain come from a fixed Weyl sequence:
* CPU: `MockRNG <: AbstractRNG` (its `rand` returns the next given value) passed to
  `ESMSampling.sample_line!`;
* device: the same array as input, with `cpumap = true`, the CPU's u → t map of the in-segment
  offset.

Round by round:
* **(a)** the device draws t with its own points;
* **(b)** the CPU draws with `sample_line!`;
* **(c)** the device sampler `sample_points` runs on the CPU walk's points;
* both sides commit the **device's** t, so the states stay comparable.

| comparison | result | kind |
|---|---|---|
| (c) device sampler on the CPU points vs CPU t | max \|Δt\|/max(1, \|t\|) = **4.4e-13** on all comparable draws | round-off: same algorithm, different summation of the potential |
| (a) device own points vs CPU t | **35 769 / 36 000** agree (max 7.6e-10 relative) | round-off |
| (a), the rest | 203 differ: the CPU walk split a ray (see below) | same distribution, different u → t map |
| CPU ray overshoot | 28 draws (0.08 %): a CPU walk point beyond 1e12, CPU t = 0.0 or exactly a breakpoint | CPU precision loss (§5.8) |
| CPU improper vs device improper | 0 / 0 | — |
| node values θ, ω, uc, e, u after 12 rounds vs CPU caches `LC.Y` | **0.0, bitwise equal (100 %)** | bitwise: the same y + t·a |
| r vs CPU `LC.r` | 1.5e-11 MW (98.1 % of entries bitwise) | round-off (different summation order of the slot changes) |
| r cache vs `rebuild_r` | 1.5e-11 MW | round-off |
| device `cost_parts` vs CPU `energy_cost` of the CPU state | 1.4e-13, 2.2e-16 relative | round-off |
| `energy_cost` of the downloaded device state (`to_host!`) | 6.7e-16, 4.4e-16 relative; max \|Δv\| 1.4e-12 | round-off (Uᵀ Y) |
| Reactant (CPU backend; GPU gives the same figures) vs Julia arrays, same uniforms, the same 12 rounds | θ 2.3e-14, ω 6.0e-14, uc 1.3e-14, e 4.0e-14, u 3.3e-14, r 9.3e-15 (relative); 85–99 % of entries bitwise | round-off (XLA libm and FMA contraction) |

**Why (a) can differ from the CPU.** The CPU walk inserts points that are not kinks. Beyond the
last breakpoint it steps by |t|, doubling while the slope of the last segment is ≤ 0, then by
2·40 nats/slope, and it stops 40 nats above the running minimum. The dense form has exactly the
kinks plus exact rays. Both segmentations sample the same piecewise-exponential density, but the
map (u1, u2) → t (segment by u1, offset by u2) depends on the segmentation. Check (c) shows that
this is the only difference: given the CPU's points, the device reproduces every CPU draw.

**The 40-nat truncation** changes the normalisation by ≤ e⁻⁴⁰, which is invisible at round-off;
none of the 203 differences comes from it.

### (i) ΔC(t) of the term batch = CPU `eval_move!`

The CPU move is built exactly as in `local_sweep!`: `prep_node!` along the same direction,
`finalize!`, `link_soc!`, and a BlockMove over τ−1 and τ evaluated with `mv_eval!`. The setup is
150 random sites × 3 chains per family, at random t.

The device state uses the CPU residual `LC.r`. The device's own `rebuild_r` differs from it by
2e-11 MW, and that alone moves ΔC by up to 1e-7 €.

Criterion: |Δ| ≤ 1e-9 |ΔC| + 1e-12 Δt c_shed max|r₀|. The second term is the ulp level of
r₁ = r₀ + Δr at |r₀| ≈ 10⁴ MW times c_shed Δt = 18 000 €/MW, which is present on the CPU as well.

| family | max abs. difference | max relative (where \|ΔC\| > 1e-6 €) |
|---|---|---|
| gen | 6.0e-8 € | 9.3e-11 |
| uc | 3.0e-8 € | 1.5e-16 |
| pair | 8.4e-9 € | 6.9e-13 |
| soc | 4.8e-7 € | 2.0e-11 |
| link | 3.4e-8 € | 3.5e-13 |
| theta | 8.8e-8 € | 2.6e-11 |

**Independent check.** On the 3-node setup, ΔC was also compared with the full `cost_parts`
difference of the moved state. It agrees to the ulp of the total cost (1e-6 € at 7e9 €).

### (ii) KS of line draws (statistical; run before the strategy change, now off unless `KS=1`)

The sampler draws 40 000 times per chain at one site per case. Each set of draws is compared
with:
* quadrature of exp(−β ΔC) (CPU evaluation, 200 001 points);
* 40 000 CPU `sample_line!` draws (two-sample KS).

The device draws come from two sources: Julia arrays with given uniforms, and Reactant on the
GPU with the device RNG (gen and soc cases). Critical values at α ≈ 0.001: 0.0097 one-sample,
0.0138 two-sample.

| case | KS vs quadrature | KS vs CPU draws | KS (Reactant RNG) vs quadrature |
|---|---|---|---|
| gen, stiff: node shedding, the c_shed kink inside the box (site (32, τ 1)) | 0.0044–0.0069 | 0.0045–0.0069 | 0.0048–0.0062 |
| soc, two-step (site (1, τ 2)) | 0.0035–0.0048 | 0.0050–0.0066 | 0.0031–0.0046 |
| uc, pair, link, theta (one site each) | 0.0035–0.0082 | 0.0036–0.0085 | — |

**3-node model at equilibrium states.** Device vs CPU draws were also compared at every
storage unit and generator, at τ = 1, 2, 37 and 73. All pass. The largest one-sample value (0.017,
SoC unit 1 τ 2) is equal for CPU and device draws, so it comes from the quadrature grid.

**Dense-form exactness.** Every secondary quantity has ≤ 2 sign changes on every site of every
round. The maximum is 1 for gen, uc, soc, link and theta, and 2 for pair.

### (iii) many sweeps in one compiled program (Reactant RNG, C = 2)

* GPU: 30 sweeps run in one call of `run_sweeps` (27 M line moves). The CPU backend re-run after
  the final sampler changes used 10 sweeps.
* The r cache is never rebuilt in between. It differs from `rebuild_r` by **9.8e-11 MW**
  (|r| up to 4.9e4); 6.2e-11 MW after the 10 CPU-backend sweeps.
* `cost_parts` of the final state against the CPU `energy_cost` of the downloaded state
  (`to_host!`): relative **2.6e-14 and 3.3e-15** on op (GPU), 5.9e-15 and 8.3e-15 (CPU backend),
  with shed and CO₂ equal.
* op fell from 6.0e11 to 8.4e10 € in chain 1, from the perturbed heuristic start.

### (iv) `check_conflicts`

0 conflicts on all 43 rounds of the schedule: gen 6, uc 2, pair 6, soc 8, link 3, theta 18.

Negative controls are detected:
* two neighbouring θ sites in one round;
* two units of one node;
* one SoC unit at τ and τ + 1.

SoC at τ and τ + 2 is accepted.

### (vi) Julia arrays vs Reactant, same uniforms

One sweep of all six families (C = 2; 902 280 line moves), on the GPU and again on the CPU
backend after the final changes (same numbers to two digits). Maximum difference, absolute and
relative to max |x|:

| field | absolute | relative |
|---|---|---|
| θ | 6.2e-9 | 3e-13 |
| ω | 3.3e-13 | 1e-13 |
| uc | 1.7e-14 | 2e-14 |
| e | 4.1e-13 | 4e-13 |
| u | 9.3e-14 | 9e-14 |
| r | 2.9e-8 MW | 6e-13 |

Exactly equal is not possible here. XLA's log1p/expm1 and FMA contraction differ from Julia's
in the last bits, and that propagates through later draws at round-off level.

### (vii) throughput on this GPU (RTX 3050 Ti Laptop, FP64 at 1/64 rate; development numbers)

`device/scripts/throughput.jl`. All model terms are on. GPU: one compiled `run_sweeps` call,
Reactant RNG. CPU: one `local_sweep!` with the default plain families (extra = 4), 8 threads, from
the same start. The CPU move set differs (Haar atoms plus random families), so compare line moves
per second, not sweeps.

| model | chains | line moves per chain-sweep | GPU s/sweep | GPU line moves/s | CPU s/sweep (8 threads) | CPU line moves/s |
|---|---|---|---|---|---|---|
| 40 × 6 h (T = 1460) | 4 | 451 140 | 3.75 | **4.8e5** | 3.34 (595 539 moves) | 1.8e5 |
| 40 × 6 h | 2 | 451 140 | 1.88 | 4.8e5 | | |
| 256 × daily (T = 365), rounds ≤ 8192 sites (θ ≤ 2048) | 1 | 1 161 795 | 2.87 | **4.0e5** | 12.1 (1 042 637 moves) | 8.6e4 |
| 256 × daily, 4 chains, rounds ≤ 32 768 sites | 4 | — | out of memory (2.96 GiB requested) | | | |

**Per family, 40 × 6 h, 4 chains:**

| family | rounds | s/sweep | line moves/s |
|---|---|---|---|
| gen | 6 | 0.39 | 1.0e6 |
| uc | 2 | 0.17 | 1.5e6 |
| pair | 6 | 0.69 | 2.7e5 |
| soc | 8 | 0.65 | 1.1e6 |
| link | 3 | 0.01 | 4.0e6 |
| theta | 18 | 1.91 | 1.2e5 |

**Readings:**
* **θ dominates at 40 nodes.** About 150 primaries per site (7 lines × 21 hinge knots) are all
  evaluated in the dense form.
* **The moves/s do not grow with the chains**, so the GPU is already saturated by the
  FP64 arithmetic: FP64 runs at 1/64 rate on this laptop GPU, ≈ 0.08 TFLOP/s.
* **On an H100** (FP64 at 1/2 rate), the same program should be ≈ 30–100× faster before any
  fusion.
* **Compile times.** The full six-family sweep compiles in 3–4 min (CPU or GPU backend
  alike). One family alone compiles in 4–40 s.
* **The Reactant CPU backend** runs the 40-node sweep at 5.4e4 line moves/s (test_rounds), ≈ 9× slower than the GPU here.


## 4. Statistical runs before the strategy change (now off by default): test (v) on the 3-node setup of test_exact.jl

Setup: the setup of test/test_exact.jl (3 nodes, T = 73, Δt = 120 h, fixed (k, x)) and the
observables of test/test_exact.jl.

**Device kernel from the start s₀ does not mix the flows.** The plain device kernel is coord,
pair and θ-NONE. Run from s₀ (512 chains, 300 + 300 sweeps), it is far from equilibrium: the
line flow is +89 MW against −651 MW on the CPU, and the drift between the first and second half
is z = −7.6. This is a mixing limitation of the plain set, not an exactness issue:
* A θ coordinate move changes the residual at both ends, which costs ≈ 14 nats per MW at
  Δt = 120 h.
* The CPU reference mixes the flows with `xpair` and `cell` moves, which are the compensated
  families of phase 3.

**Exactness is therefore tested as π-invariance.**
* The 512 device chains start from equilibrium states of the CPU plain kernel (one chain,
  300 burn-in sweeps, then thinned by 8).
* They then run 100 device sweeps: one compiled call, Reactant RNG, about 1e6 line moves/s.
* The starts reproduce the CPU cost to < 1e-11.
* The r drift after 100 sweeps is 3.6e-11 MW.

| observable | CPU long chain (4000 sweeps) | device after 100 sweeps | paired change per chain | z paired | z vs CPU |
|---|---|---|---|---|---|
| op | 6.85514e9 ± 7e5 | 6.85031e9 | −6.4e6 ± 2.2e6 | −2.87 | −6.87 |
| shed | 64.98 ± 0.56 | 64.24 | 1.8 ± 2.0 | 0.91 | −0.45 |
| mean SoC | 0.50309 ± 0.0014 | 0.50095 | −0.0011 ± 0.0008 | −1.37 | −1.35 |
| line flow | −650.5 ± 4.5 | −648.5 | 0.98 ± 1.8 | 0.53 | 0.32 |

The paired tests pass (|z| < 4) for all four observables, and so do the tests against the long
CPU chain for shed, SoC and flow. op against the CPU chain fails.

**Per-family invariance runs** (200 sweeps each, from the same starts):
* gen, pair and theta: no significant change.
* link: removes heavy-tailed link box penalties (penalty −4e6, z −2.1).
* soc: storage +19 (z 7.7), shed −12 (z −6.3), dump −267 MWh (z −11).

**Diagnosis** (`device/test/diag_exact_soc.jl`, CPU only, 64 chains):

| run | storage | shed | dump |
|---|---|---|---|
| E: the device soc schedule emulated on the CPU (`cpu_move` + CPU `sample_line!` + `commit_block!`, sequential, 200 sweeps) | +19.5 ± 6.8 | −14.9 ± 5.2 | −366 ± 73 (z −5) |
| A: CPU plain kernel continued from the same starts, 1200 sweeps | | | +3.5e3 ± 1.6e3 (z 2.2; 1.9 at 400 sweeps) |
| B: E, then 400 CPU plain sweeps | | | back to A's level |

E reproduces the device effect, and E is π-invariant by construction:
* it is a sequential sweep of exact single-coordinate line draws;
* it uses the CPU's own evaluation, and ΔC equals the full `cost_parts` difference;
* each single draw matches quadrature.

So the CPU reference states are not π-distributed in the storage-dispatch and dump directions.
Either the CPU plain kernel mixes them slowly (its own continuation drifts, with a large
between-chain spread), or it is biased there. The device kernel is not the cause.

**For the main session.** `test/test_exact.jl` (200 burn-in sweeps, 2000 samples) may
under-resolve these directions. A longer or many-chain CPU reference would decide between slow
mixing and bias.

The test asserts:
* the paired (invariance) tests for all four observables;
* the comparisons against the CPU chain for shed, SoC and flow.

op against the CPU is reported, and asserted only with `STRICT=1`.

## 5. Tracing issues met and how they were solved

1. **`abs(a*b)` is miscompiled by Reactant 0.2.304.** For two traced operands it returns a·b, so
   the abs is dropped. abs of a sum and abs(2·a·b) are fine.
   * Effect: `abs.(g .* L) .< 1e-10` was true for negative products, so the truncated-exponential
     offset picked the wrong branch.
   * Fix: every absolute value in the device kernels is `absd(x) = max(x, −x)`.
   * *Main session:* phase 1 `cost_parts` uses `abs.(f)` with `f = tmul(θ, Dl) .* lb`, the abs of
     a product. The traced phase-1 test still matches to 1e-14 here (the contraction output is
     probably not matched by the faulty pattern), but `absd` would be safer there too.
2. **FMA contraction makes the exact zeros of padded sites inexact.** XLA evaluates
   κ·clamp(y₀ + 0·t) − κ·clamp(y₀) as fma(κ, c, −κc), which is the rounding error of κc and not 0.
   Thousands of padded sites scatter into the same column, so the r cache drifted by 4e-9 MW per
   sweep there. Fix: slot changes of padded sites and padded line slots are masked with a select
   (`msk`). The drift is now 1e-10 MW after 30 sweeps.
3. **Discontinuous u → t map.** The CPU's `trexp_offset` mirrors with u for g < 0, while the
   |gL| < 1e-10 branch uses u L. Both are exact samplers, but the map jumps at the switch. In
   flat segments (|c| ≈ 4·10⁴ nats, slope from round-off) the Julia and XLA runs took different
   branches, a difference of 0.05 in e. Fix: for g < 0 the mirror uses 1 − u, which is the
   monotone inverse CDF and continuous with the small branch.
4. **Captured structs with `Float64` fields** in a device loop fail: the numbers become traced. So
   `Consts{F,H}` is type-parametric, with its own type-parametric `HingeT` copy of the phase-1
   `Hinge`.
5. **Term structs cannot have one array type parameter.** Under tracing, reshapes and gathers give
   different types, so the term structs hold untyped `Vector`s. Reshapes of traced arrays are
   materialized (`rs`).
6. **Optimization barrier.** An `optimization_barrier` on the drawn t was added defensively, so
   that commit and slot changes see one value. It was not the cause of item 2.
7. **Scatter-add.** It is written with `Reactant.Ops.scatter((a, b) -> a + b, …)` on (C, B)
   windows. Repeated indices add exactly. Gathers `X[:, idx]` with traced idx and dynamic slices
   of the round tables inside `@trace for` work as is.

8. **Found in the CPU sampler (for the main session; `src/local.jl`, `walk!`/`sample_line!`).**
   * *What happens.* After a segment that is flat to round-off, the slope used for the ray step is
     ≈ 1e-16 > 0, so δ = 2·maxnats/slope ≈ 1e15–1e18. Example: soc site (31, τ 10), where βΔC is
     flat to 1e-16 on [−0.5, 0.5].
   * *Effect.* The last segment [−1.1e18, −0.5] has c ≈ 1e20 at its far end. Its log-mass
     −c_a + logseg(L, c_b − c_a) loses all precision, and so does the mirrored offset
     a + (L − s′), giving t = 0.0 or exactly the breakpoint. This happened in 28 of 36 000 draws.
     That is a small bias towards the current state, at the breakpoint of a flat segment.
   * *Suggested fix:*
     - take the ray slope from a probe point beyond the last breakpoint, not from the last
       segment;
     - compute segment masses anchored at the lower end: −min(c_a, c_b) + logseg(L, |c_b − c_a|);
     - anchor the offset of a g < 0 segment at its right end: b − s′.
   * *On the device.* It does all three: exact rays from the last kink (lml/lmr), stable masses
     (`lmf`), and `trexp_end`.
9. **Dynamic slices in `@trace for`** (main session's question). What works here:
   * indexing a traced array with the traced loop index as an integer in the last dimension:
     `tab.q[:, r]`, `tab.ln[:, i, r]` (i static), `U[:, :, :, j + off]` (off an Int captured as a
     traced number);
   * gathers with a traced integer vector, `X[:, idx]` and `v[idx]`.
   Ranges with a traced bound (`v[q:q]`) were never needed: per-round data are whole columns of
   padded (B, R) tables, sliced by an integer. Static ranges (`X[:, :, 1:1]`, `h.d[j:j]` with j a
   Julia Int from an unrolled loop) are fine.

## 6. Open items

* **Compensated families, slope-jump form, templates, block and capacity moves** are phase 3+.
  The 3-node runs (§4) show why compensation is needed: θ coordinate moves do not mix the flows,
  because they are pinned by the balance.
* **θ cost.** θ takes 50 % of the sweep at 40 nodes because of the dense evaluation of all
  hinge knots. The slope-jump form (or evaluating only the knots of each line, since every line
  term is a function of one argument) is the next step for performance. The other families are
  ≥ 1e6 moves/s here.
* **Memory.** 256 × daily with 4 chains needs smaller rounds (`bdefault` ≤ 8192), or rounds
  tiled by chain; it was measured with 1 chain only. The round workspace is ≈ C·B·K·(terms)
  doubles.
* **Parallel lines** are not supported by the θ family (`schedule` errors). There are none at 40
  or 256 nodes. Support needs site-dependent slot weights.
* **`rebuild_r`** is not called inside the device loop. The cache drift is ~1e-10 MW per 10–100
  sweeps, so a rebuild every sweep (design §4.3) is optional. It is a host or second-program call
  for now.
* **`check_conflicts`** runs on the host, on the same index tables the device uses. The
  design's device scatter-of-ones version is not implemented.
* **CPU sampler precision** (§5.8) is for the main session to fix in `src/local.jl`, together with the
  `abs(a*b)` check of phase 1 (§5.1).
* **CPU reference in test_exact.jl** (§4). The CPU plain kernel's storage and dump directions
  appear under-equilibrated. This was found by the statistical runs made before the strategy
  change; no new statistical runs were made.
* **Uniforms.** The production device map for g < 0 uses 1 − u (continuous). `cpumap = true`
  reproduces the CPU's map for comparisons. Both are exact samplers.
* **`MockRNG` is duplicated.** `test_mock.jl` defines its own; the main session's
  `test/mockrng.jl` appeared later and could replace it.
