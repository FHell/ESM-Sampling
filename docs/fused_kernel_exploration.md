# Fused kernels for the device port: an exploration (2026-11-05)

Development note, work in progress on a side branch (not merged, nothing under `device/src/`
changed). Code in `device/experimental/`: its own environment (`Project.toml`, `Manifest.toml`:
ESMDevice developed into it, CUDA.jl 6.2.2 and KernelAbstractions 0.9.45 added, Reactant kept at
0.2.304 / Reactant_jll 0.0.438), the kernel `fused_jump.jl`, the check-and-timing script
`tpot_fused.jl`. The kernel is verified against the package's draw on the GPU (§3); on the
development GPU it is slower than the XLA global part it replaces (§3, §5).

## 1. Which family, and why

The candidates, with the numbers of [system/performance.md](system/performance.md) §7
(laptop GPU, 40 nodes × 6 h, 4 chains, slope-jump form at ad73f13; a sweep of the full run set
about 23 s):

| family | s/sweep | share | form | worst fit to XLA |
|---|---|---|---|---|
| `tpot` | 8.3 | 36 % | slope-jump, 1 478 entries per site (cap 360), 87 columns | the payload sort of the entries is its most expensive kernel (1.2 s per sweep at 33 % occupancy); about ten two-level scans over the entry axis, each a materialized (N, C, B) tensor |
| `sblk2 … sblk32` | 9.9 together (1.6–2.8 each) | 43 % | slope-jump, 473–4 523 entries (cap 123–423) | the same global part |
| `tsoc` | 3.4 | 15 % | slope-jump, 218 entries (cap 106) | the same; slower than its dense form on the GPU |
| `soc` | 0.2 | 1 % (35–47 % of the fast set) | dense, ≤ 40 points × 7 columns | many small launches; a site fits one thread |

Choice: **`tpot`**, and in it the part that every slope-jump family shares, the per-site
"global part" of the draw: fold the entries at 0, sort them with the jumps as payload, cap,
accumulate the segment slopes and the potential from 0 outwards, inverse CDF
(`jump_points1`, `jump_values1`, `sample_core1` in `device/src/jump.jl`). Reasons: the largest
single family (36 %); its most expensive kernel is exactly this part; one site's entries (≤ 1 478,
24 KB) fit one thread block's shared memory, so the sort, the scans and the draw map onto one
block without any global intermediate; and the kernel is family-independent, so the same kernel
serves the blocks, `tsoc`, the cells, `slong` and the storage pairs (together about 94 % of the
sweep), each of which keeps its own per-column term evaluation (`jump_parts`) in XLA.

The per-column part stays in XLA for now: it is one fusion per term type, well vectorized, and
fusing it means re-expressing every term type's local points, roots and jumps as scalar code.

## 2. Route and constraints

KernelAbstractions kernel launched inside the Reactant-traced program (Reactant's
`ReactantKernelAbstractionsExt` + `ReactantCUDAExt`): the kernel is compiled by GPUCompiler to
LLVM bitcode and embedded as an `enzymexla.kernel_call` with the block and thread dimensions and
the shared-memory size as operands; traced arrays are passed as `CuTracedArray`, traced scalars
as `CuTracedRNumber`; arrays written by the kernel become outputs of the call. Checked on this
machine (Reactant 0.2.304, CUDA.jl 6.2.2, driver 580.95, RTX 3050 Ti):

* a kernel with `@localmem` shared arrays and `@synchronize` barriers inside `while` loops
  compiles and runs correctly on the GPU inside the program; `@synchronize` inside `@inline`
  helper functions called from the kernel works as well (the barrier is CUDA's method-table
  override, which applies to everything the kernel compiles);
* `raise = true` (the kernel raised to HLO, the route to the CPU backend) **fails** for such
  kernels ("failed to raise func": the raising pass does not handle the `scf.while` loops with
  barriers and address-space-3 memory). Raising covers elementwise, affine kernels (Reactant's
  own tests); a sort/scan kernel cannot be tested on the CPU backend this way. The array-level
  check of the testing contract is therefore a Julia reference of the same per-site algorithm
  (`fused_draw_ref`, the same `fused_tail!` code as the kernel's serial mode), and the compiled
  check runs on the GPU only;
* KernelAbstractions' own CPU backend cannot run the kernel either (`@synchronize` inside
  control flow is not supported there);
* `@Const` arguments support linear indexing only (2-d indexing of a `Const` falls back to a
  dynamic call and fails to compile); the kernel indexes linearly;
* `@index(Local, Linear)` is not an `Int` on the device: a helper declared with `li::Int` fails
  to compile with only "unsupported call to jl_f_throw_methoderror" at the call site. Leave
  helper arguments unannotated;
* CUDA.jl must be loaded for the trace even when the target is not CUDA; it is functional here
  (CUDA 13 runtime, the artifacts were already in the depot);
* static shared memory is limited to 48 KB per block: for `tpot` at the script's size
  (N = 920, cap 362, 256 threads) the kernel needs 33 KB (sort keys 1 024 × 12 B, five scratch
  vectors of cap + 2, three reduction buffers); at the production size (N = 1 478, cap 360) about
  38 KB; `sblk32` (4 523 entries) needs the 8 192-slot sort buffer (98 KB) and so dynamic shared
  memory or a two-pass sort;
* FP64 throughout, as the package; a Float32 build would halve the shared memory. The
  development GPU runs FP64 at 1/64 of its FP32 rate, which bounds the kernel's arithmetic
  (bitonic compare-swaps, scan additions, the transcendental functions of the draw) and makes
  its timings unrepresentative of the cluster's GPUs (FP64 at 1/2 rate on A100/H100);
* the first compile of a program with the kernel takes about two minutes (the XLA service and
  the GPUCompiler pipeline start); later compiles of changed kernels take about 10 s.

Kernel design (`fused_draw_kernel!`): one block per (chain, site), NT threads; load the entries
with their original index as a stable tie-breaker (entries at exactly 0 become +Inf and their
jumps are folded into one entry at 0 with index 0, as `jump_points1`); bitonic sort over the
next power of two; cap; Inf entries moved onto the largest finite point with jump 0; the
real-candidate count, J0 and |J0| by a block reduction; then the tail (`jump_values1` +
`sample_core1`) over the ≤ cap + 2 entries, block-parallel (mode 0): the four prefix/suffix sums
of the jumps and their absolute values, the segment slopes with the flat-slope tolerances, the
two masked sums of the segment masses and the values at the points (Hillis–Steele scans in shared
memory, two elements per thread, so cap + 2 ≤ 2·NT), the rays, the segment log-masses (the
`log`/`exp` work spread over the threads), the maximum and the cumulative masses, the selected
segment by a block count, and thread 1 only for the final truncated-exponential draw. Mode 3
runs the same tail serially on thread 1 (`fused_tail!`, the reference's code); modes 1 and 2
skip the tail and the sort for timings. The partial sums use other association orders than the
two-level scan `cumsum1` and the serial reference, so "equal to round-off", not bitwise.

The wrapper `sample_jump_fused(tb, k, u1, u2; K, cap, nt, mode)` calls `jump_parts`, lays the
entries out as (N, C, B) (the package's own layout, the entry axis fastest, so a site's entries
are contiguous) and launches the kernel; on Julia arrays it runs the reference. `do_round_fused`
is `do_round` with this draw; the Jacobian step, slot changes and commit are unchanged.

## 3. Status and results

| step | state |
|---|---|
| environment with CUDA.jl and KernelAbstractions, Reactant unchanged | done |
| toolchain: a shared-memory/barrier kernel inside a Reactant program on the GPU | works |
| raising such a kernel to HLO for the CPU backend | fails (constraint above) |
| kernel (serial and block-parallel tail), Julia reference, wrapper (`fused_jump.jl`) | done, verified below |
| arrays: `sample_jump` against the reference on a `tpot` round | passes (400 and 2 000 sites) |
| GPU: the fused draw against the reference, the array draw and the XLA GPU draw; `do_round_fused` against `do_round` (t and the loop state) | passes |
| timings of the draws and the rounds on the development GPU | done at 400 and 2 000 sites; the kernel is slower than XLA's global part here |
| timings on a datacenter GPU (full-rate FP64) | not run |

Setup of the script (`C=2 H=24 NSITE=200`, then `NSITE=1000`): 40 nodes, T = 365, 2 chains,
`tpot` rounds of 200 (87 rounds) or 1 000 sites (29 rounds), N = 920 entries per site, cap 362,
K = 4, NT = 256; mock uniforms (a Weyl sequence) on both sides.

Correctness (maximum relative difference of t, |Δt| / max(1, |t|)):

| check | 400 sites | 2 000 sites |
|---|---|---|
| arrays: reference (`fused_draw_ref`) vs `sample_jump` | 9.8e-13 (0 sites > 1e-12) | 1.0e-12 |
| GPU fused draw vs the reference | 2.0e-12 (9 sites > 1e-12, 0 > 1e-9) | 2.0e-12 |
| GPU fused draw vs `sample_jump` on arrays | 1.5e-12 | — |
| GPU fused draw vs the XLA draw on the GPU | 1.2e-12 | 1.1e-12 |
| the XLA GPU draw vs its own array form (the GPU's own round-off) | 1.6e-12 | 2.3e-12 |
| parallel-tail kernel vs serial-tail kernel | 1.9e-15 | — |
| guards | equal, none set (every count below the cap) | equal, none set |

The fused kernel is within the round-off the XLA draw itself shows between the GPU and the CPU
arrays. Round level (400 sites, `do_round_fused` vs `do_round` on the GPU): t after the Jacobian
step 1.2e-12 (6 of 400 sites > 1e-12, none > 1e-9), 381 of 400 sites moved in both; the
committed loop state θ 1.2e-12 relative (7.9e-10 absolute), ω and uc 2.1e-13, r 9e-14, e and u
exactly equal.

Timings (development GPU, `sync = true`, minimum of 5; the machine is shared, medians are up to
25 % higher):

| program | 400 sites | 2 000 sites (1 820 real) |
|---|---|---|
| XLA round (`do_round`) | 11.6 ms | — |
| fused round, parallel tail | 11.5 ms | — |
| fused round, serial tail | 22.9 ms | — |
| XLA draw alone | 8.4 ms | 37.4 ms |
| fused draw, serial tail | 16.4 ms | 69.8 ms |
| fused draw, parallel tail | 11.4 ms | 44.5 ms |
| fused draw without the tail (mode 1) | 8.0 ms | 35.1 ms |
| fused draw without tail and sort (mode 2) | 6.9 ms | 22.2 ms |
| per-column part alone (`jump_parts` with its outputs copied out) | 6.0 ms | 29.9 ms |

Kernel shares: serial tail 8.3 ms / 34.7 ms, parallel tail 3.3 ms / 9.4 ms, bitonic sort
1.1 ms / 12.9 ms, load and reductions 0.8 ms / (not separable: at 2 000 sites the parts-only
program is slower than the mode-2 program, so it is no lower bound there; the entry outputs it
copies are 29 MB). Per real site at 2 000 sites: the fused draw 24.5 µs, the XLA draw 20.5 µs.
The kernel's global part costs about 2× XLA's at 400 sites (5.3 vs 2.3 ms) and 1.2–1.5× at
2 000 sites (the draw is 7 ms slower; the sort plus the parallel tail is 22 ms against XLA's
global part of at most 15 ms). At round level the two programs are on par at 400 sites (11.5 vs
11.6 ms) although the draws alone differ by 3 ms; the round's other steps (Jacobian, slot
changes, commit) are the same code on both sides, so the XLA global part is dearer inside the
round than in the draw-alone program (its fusions and scheduling differ), which the draw-alone
numbers do not show. The round comparison at 2 000 sites was not run. The ratio does not improve with size, so both sides are
throughput-bound here, and on this GPU the kernel's FP64 arithmetic (28 k compare-swaps per site
in the 1 024-slot bitonic sort against about 9 k for an O(N log N) sort of 921 entries; the
Hillis–Steele scans with n log n additions) is what it pays for; the XLA version is bound by the
memory traffic of its materialized (N, C, B) tensors, which a datacenter GPU does not make
cheaper, while it makes the FP64 arithmetic 32× cheaper relative to FP32.

## 4. Next steps

1. Timings on the cluster (A100/H100): the script as is, `C=4` and the plan's round size; the
   number that decides whether the global-part kernel pays on the target hardware.
2. The sort, now the largest part of the kernel: sort only the N + 1 real slots (a bitonic
   network on the padded power of two does 11 % more work at N = 920 but 78 % more at N = 4 523),
   or a block-wide merge/radix sort of the keys (O(N log N)); pack (point, index) into one 64-bit
   key where the points' bit patterns order like the values (sign-flip trick) to halve the
   shared-memory traffic of the compare-swaps.
3. Work-efficient scans (Blelloch, n additions and 2 log n barriers) instead of Hillis–Steele;
   the two serial 256-element reductions by thread 1 (the maximum, the count) as tree reductions.
4. The same kernel on `sblk8`/`sblk16` (`FAM=sblk8`) and `tsoc`; `sblk32` needs dynamic
   shared memory or a two-pass sort.
5. Fusing the per-column part: the scalar term functions already exist (`gen_c`, `uc_val`,
   `soc_c`, `link_c`, `bal_val`, the kinked argument `argloc`), so a per-column thread of the
   block can compute its local points, its cost jumps and its slot injections and append its
   entries to the shared list; the balance-slot crossings need the slot's points sorted with the
   residual (a second, smaller in-block sort per slot). This removes the materialized (N, C, B)
   entry tensors altogether, which is where the memory traffic of the XLA form goes. Estimate:
   one to two weeks for the share-preserving families (`tpot`, `tsoc`, blocks), given the
   verified global part; the plain dense families (`soc`, `gen0`, `theta`) are a separate,
   simpler kernel (one thread per site, everything in registers) that would matter for the fast
   set (35–47 % of it in `soc`).
6. Testing contract for kernels (as applied here): the Julia reference of the per-site algorithm
   is the array-level check (no raising), and the GPU-compiled kernel is compared against the
   reference, the package's array draw and the package's XLA draw on the same inputs and mock
   uniforms; FP64 and the GPU's libm differ from the CPU's by ulps, and the XLA draw's own
   GPU-vs-array difference (about 2e-12 relative in t) sets the scale of "equal to round-off".

## 5. Recommendation

The route works: KernelAbstractions kernels with shared memory and barriers run inside the
Reactant program on the GPU without touching the package, the global part of the slope-jump draw
fits one block per site, and the kernel reproduces the package's draw and the committed state to
the GPU's own round-off. The CPU backend cannot run such kernels (raising fails), so the testing
rules for kernels must accept a Julia reference as the array level and the GPU as the only
compiled check.

The global-part kernel alone does not pay on the development GPU: its FP64 arithmetic is
throughput-bound at the 1/64 FP64 rate, and it is 1.2–2× slower than XLA's global part. Whether
it pays on the cluster's GPUs is step 1 above and cannot be settled here. The larger step, fusing
the per-column term evaluation into the same block (step 5), removes the materialized entry
tensors that the XLA form is bound by and is where the "1–3 ns per move" estimates of
[system/performance.md](system/performance.md) come from; it is a one-to-two-week effort per
term structure, and the verified global part is its foundation.
