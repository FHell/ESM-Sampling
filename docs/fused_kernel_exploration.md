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
* a kernel program compiled on a Reactant CPU client *without* raising does run: Reactant lowers
  the kernel call to a host LLVM module (`llvm.` ops in the module, no `kernel_call`). This works
  for the barrier-free one-thread-per-site kernel (§6) and aborts the process for a kernel with
  `@synchronize` (LLVM cannot select the nvvm barrier intrinsic for the host);
* `@private` scratch (`MArray`) works on the GPU route; a closure that assigns a variable of the
  enclosing function's name boxes it, which shows up as "dynamic function invocation (call to +)"
  in the kernel compile;
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

### 3.1 Other properties of the `tpot` kernel: compile time, memory, program size

Measured in one warm process (`Reactant.Profiler.@timed_compile`: tracing, MLIR passes, XLA;
the compiled executable's memory statistics through `PjRtLoadedExecutableGetCompiledMemoryStats`,
the device allocator's peak through `XLA.allocatorstats` after `PjRtDeviceClearMemoryStats`, with
preallocation on; op count = `stablehlo`/`enzymexla`/`chlo` ops of the optimized module before
XLA). Both rounds at 400 and 2 000 sites:

| property | XLA round | fused round |
|---|---|---|
| compile, warm (trace / MLIR / XLA), 400 sites | 10.7 s (0.8 / 7.4 / 2.5) | 10.8 s (0.75 / 8.2 / 1.9) |
| compile, warm, 2 000 sites | 11.3 s (1.2 / 7.3 / 2.7) | 10.7 s (1.5 / 7.2 / 2.0) |
| compile, cold process | 171 s, of which 161 s Julia compiling the traced code | 17.1 s after the XLA round (shares its code; 8.2 s tracing incl. the kernel's GPUCompiler pass) |
| ops of the optimized module (both sizes) | 3 359 | 2 678 (−20 %); module text 645 KB vs 290 KB (the kernel's LLVM bitcode is embedded) |
| temporaries / peak, 400 sites | 21.2 / 44.6 MB | 18.8 / 42.9 MB |
| temporaries / peak, 2 000 sites | 114.2 / 142.2 MB | 93.9 / 131.5 MB |
| allocator peak above the resident buffers, 400 / 2 000 | 24.9 / 117.9 MB | 22.5 / 97.6 MB |

(Arguments 23.3 / 37.4 MB, outputs 1.2 MB, the same on both sides.) The compile time is the
same: the MLIR passes dominate and are the same program apart from the draw; the XLA stage is
0.6–0.7 s shorter with the kernel. The temporaries per site are 57 KB in XLA and 47 KB fused
(about 7.4 and 6.4 entry tensors of 920 × 8 B): the kernel removes the sort's and the scans'
(N, C, B) tensors but still takes the entries (qa, Ja) as materialized inputs, and the per-column
part's intermediates are the larger share. For `DENSE_BUDGET` this means about 18 % more sites
per round at the same memory; it does not change what fits per GPU in a meaningful way.

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
   Barrier-free kernels (§6) additionally compile and run on the CPU backend without raising,
   which restores the normal contract (arrays → CPU backend → GPU) for them.
7. The dense kernel on `gen0` and `theta` (the same pattern as `soc`: a per-site function over
   the stacked term fields, the family's primaries and quantities): not started. Raising the
   `soc` kernel to HLO would need a scratch without `alloca`/`memcpy` (tuples, fixed-count loops
   over `NP`) and is worth one attempt, since the raised form would also run as XLA fusions.

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

The one-thread-per-site kernel of the dense families (§6) is the easier and better-behaved kind:
the same draw to 1e-14, a fifth of the memory, a quarter of the ops, half the compile time, and
a CPU-backend check without raising. Its speed is the same as XLA's on this GPU because both are
bound by the same FP64 transcendentals; the memory and launch savings it brings are what counts
on a full-rate FP64 GPU, which again only the cluster can show.

## 6. A second kernel: the dense draw of `soc`, one thread per site

`device/experimental/fused_soc.jl` (kernel `soc_draw_kernel!`, per-site code `soc_site!`,
reference `soc_draw_ref`, wrapper `sample_soc_fused`, round `do_round_soc`) and the script
`soc_fused.jl`. The plain SoC family is the other regime: a site is 2 SoC columns (one moving
argument each) and 2 balance slots, its dense draw 1 + 4 primaries, 6 single roots of the SoC
quantities, 2 roots per slot of the residual on the refined grid (the two-stage search of
`dense_points`), 15 points, the inverse CDF over 17. One thread does the whole draw with the
points in per-thread scratch (`@private`, an `MArray`), the term values recomputed where needed,
no shared memory and no barriers; the sort is an insertion sort with fixed trip counts, the roots
follow `rootsK`'s rules (the smallest distinct sign changes, the ray-segment conditions,
`RAYMAX`), the potential is evaluated at the points directly and the draw is `sample_core`'s
arithmetic. The static column metadata (which arguments move, the slots) is passed as `Val`s.

Two pitfalls found: a closure that assigns a variable of the enclosing function's name (the
accumulator `acc`) captures and boxes it, which the GPU compiler reports as dynamic calls to `+`
(the CPU reference ran at 100 µs per site with 18 KB of allocations; 2.4 µs and none after
renaming); and the roots of successive quantities need their own write offset in the point list.

Results (`C=2 H=24`, the plan's `soc` rounds: 12 rounds of 7 280 sites, 14 560 per round; mock
uniforms):

| check | result |
|---|---|
| arrays: the reference vs `line_draw` (the package's dense draw) | bitwise equal on all 14 560 sites; the 15 points per site bitwise equal to `dense_points` |
| GPU kernel vs the reference | 6.1e-14 relative (8 615 sites differ bitwise: the GPU's libm and FMA contraction) |
| GPU kernel vs the XLA draw on the GPU | 1.1e-14 (494 sites differ bitwise) |
| the XLA GPU draw vs its array form | 6.1e-14 (8 717 differ bitwise) |
| `do_round_soc` vs `do_round` (GPU) | t 1.1e-14; θ, ω, uc, u exactly equal; e 1.1e-14; r 4.3e-14 relative |
| `raise = true` on the CPU backend | fails (the pass pipeline "all" fails; the dumped module stops at LLVM-level ops of the per-thread scratch and the tuples: 15 `llvm.alloca`, 17 `llvm.intr.memcpy`, 15 bounds-check traps, and the loops over the point count are not affine) |
| the CPU backend without raising (Reactant compiles the kernel as a host LLVM module on a CPU client) | compiles (0.9 s) and runs: 3.2e-14 from the reference (11 149 sites differ bitwise: XLA's CPU libm), guards equal |

The host route is the project's normal contract for this kind of kernel (arrays → compiled on
the CPU backend → GPU), without raising. It is limited to kernels without barriers: the `tpot`
kernel on the same route aborts the process in LLVM's host code generation ("Cannot select:
intrinsic llvm.nvvm.barrier.cta.sync.aligned.all"), so shared-memory/barrier kernels keep the
Julia reference as their array level and the GPU as their only compiled check.

| property (14 560 sites) | XLA | kernel |
|---|---|---|
| draw alone, min of 5 | 6.63 ms (0.46 µs per site) | 6.76 ms |
| round, min of 5 | 6.97 ms (0.48 µs per site) | 7.00 ms |
| compile, warm (trace / MLIR / XLA) | round 3.7 s (0.63 / 1.9 / 1.1); draw 10.1 s with the first Julia compile of the dense code | round 1.8 s (0.25 / 1.4 / 0.21); draw 3.1 s (1.2 incl. GPUCompiler / 1.4 / 0.43) |
| ops of the optimized module | 1 119 | 286 (−74 %) |
| temporaries / peak | 18.7 / 26.8 MB | 3.7 / 11.9 MB |
| allocator peak above the resident buffers | 21.6 MB | 6.7 MB |

(A build-only program takes 0.15 ms, a trivial program 0.13 ms: the launch overhead is not what
the 6.6 ms are.) The draw costs the same in both forms on this GPU, 0.46 µs per site: the dense
form evaluates the same term values and the same transcendental functions at the same 17 points
as the kernel, and that FP64 arithmetic (about 70 `log`/`exp`/`expm1`/`log1p` per site) is the
bound at the 1/64 FP64 rate, not the launches or the memory traffic the kernel removes. The
kernel's gains here are the other properties: a fifth of the temporaries, a quarter of the ops,
half the compile time, and the same draw within 1e-14. On a GPU with full-rate FP64 the
arithmetic shrinks 32× against the memory traffic, and the XLA form's 18.7 MB of temporaries and
about 1 100 ops (some 30 launches) become the cost; that is the case to measure on the cluster.

## 7. Both kernels in Float32 (the local proxy for cheap arithmetic)

The development GPU runs FP32 at 64× its FP64 rate, so Float32 is the local stand-in for "the
arithmetic is cheap" (as FP64 is on an H100 at half the FP32 rate): it shows whether the kernels
beat the XLA forms once arithmetic is no longer the bound. Both kernels are now generic in the
element type (`T = eltype` of the output: shared memory, per-thread scratch and every literal
sized or typed from it; the package functions they call are taken from the build of that type,
`fmod(T)`: `ESMDevice` for Float64, `ESMDevice.F32` for Float32, whose code carries Float32
literals). The XLA side is the package's Float32 build `ESMDevice.F32` (its own plan, model,
state and index; compiled under `ftcompile`, contractions at HIGHEST precision), so both sides
see the same Float32 inputs. Script: `device/experimental/fp32_fused.jl`. Two more pitfalls:
KernelAbstractions hoists `@localmem`/`@private` allocations to the top of the kernel body, so
their element type must be an expression of an argument (`eltype(tout)`), not a local assigned
later (that read an undefined variable, the kernel trapped at its start and Reactant then dropped
the whole kernel call as dead); and helper functions need their own `T`.

Correctness in Float32 has no bitwise or 1e-12 contract (the project has no Float32 exactness
tests by design). Reported: max relative |Δt| / max(1, |t|), the sites above 1e-4 and 1e-2, and
guard agreement; the Float64 draws on the same sites are the reference, and sites above 1e-2 are
the proxy for a flipped discrete choice (another segment).

| tpot, Float32 (C = 2) | 400 sites | 2 000 sites (1 820 real) |
|---|---|---|
| arrays: XLA form vs FP64 reference | 3.1e-3 (25 > 1e-4, 0 > 1e-2) | — |
| arrays: kernel reference vs FP64 reference | 3.1e-3 (11, 0) | — |
| arrays: kernel vs XLA form | 7.5e-4 (20, 0) | 1.2e-3 (94, 0; GPU kernel vs array form) |
| GPU: kernel vs XLA GPU draw | 14 (118 > 1e-4, 42 > 1e-2) | 1.1e-3 (54, 0) |
| GPU: kernel vs FP64 reference | 3.1e-3 (34, 0) | 2.8e-3 (81, 0) |
| GPU: XLA draw vs FP64 reference | 10 (132, 42) | 2.8e-3 (79, 0) |
| GPU: XLA draw vs its own array form | 10 (137, 42) | 1.4e-3 (95, 0) |
| GPU: kernel vs its own array form | 5.4e-4 (33, 0) | — |
| guards | 0 / 0, no flips | 0 / 0, no flips |
| rounds: t kernel vs XLA; state | 14 (109, 39); θ 14, ω 0.17, r 0.98 (the 39 flipped sites) | 1.1e-3 (54, 0); θ 1.1e-3, ω 9e-5, r 1.2e-4, e and u exact |

The Float32 round-off scale of the tpot draw is about 3e-3 at the worst site (its accumulated
sums over 360 entries), and the kernel sits on it at both sizes. The XLA Float32 program at 400
sites does not: it departs from its own array form at 42 of 400 sites by up to 10× (a different
segment), the same program at 2 000 sites does not (1.4e-3). This is on XLA's side (the kernel
on the same inputs stays within 5e-4 of its array form); which lowering differs at the small
batch (the sort or the scans) was not traced. In Float32 the kernel is the more faithful of the
two forms at that size.

| tpot, Float32 timings (min of 5) | 400 sites | 2 000 sites |
|---|---|---|
| XLA round / fused round | 3.2 / 2.7 ms (1.20×) | 14.8 / 15.4 ms (0.96×) |
| XLA draw / fused draw | 3.0 / 2.6 ms (1.14×) | 14.0 / 14.6 ms (0.96×) |
| parts only; kernel without tail; without tail and sort | 2.2; 2.1; 1.8 ms | 11.0; 9.5; 8.4 ms |
| kernel tail / sort | 0.5 / 0.2 ms | 5.1 / 1.1 ms |
| XLA global part (draw − mode 2) | ≥ 1.1 ms | ≥ 5.7 ms |
| compile, warm (trace / MLIR / XLA): XLA round, fused round | 11.0 (0.7 / 7.3 / 2.4), 11.2 (1.7 / 7.0 / 1.9) s | 11.7 (0.7 / 7.3 / 2.9), 12.0 (1.6 / 7.6 / 2.2) s |
| temporaries / peak: XLA round, fused round | 11.1 / 31.8, 12.1 / 32.7 MB | 52.7 / 82.8, 62.4 / 95.8 MB |

For the FP64 numbers see §3 (400: round 11.6 / 11.5 ms, draw 8.4 / 11.4 ms; 2 000: draw 37.4 /
44.5 ms). Float32 makes everything about 3–4× faster on this GPU; the kernel's global part fell
from 5.3 to about 0.7 ms at 400 sites (7×) while XLA's fell from 2.3 to 1.1 ms (2×), so at 400
sites the kernel's part is now the cheaper one and the round is 1.2× faster; at 2 000 sites the
two global parts cost the same (about 6 ms) and the round is a wash. The per-column part
(`jump_parts`, in XLA on both sides) is 70–75 % of the Float32 round, so the round cannot move
much either way. The Float32 temporaries of the fused round are not smaller than XLA's (the
kernel's inputs are materialized; XLA's own intermediates shrink with the element size).

| soc, Float32 (14 560 sites) | result |
|---|---|
| arrays: XLA form vs FP64 reference | 4.1e-5 (0 > 1e-4) |
| arrays: kernel reference vs FP64 reference | 6.3e-6 (0 > 1e-4) |
| arrays: kernel vs XLA form | 4.0e-5 (0 > 1e-4; 10 268 of 14 560 differ bitwise) |
| GPU draws, rounds, timings, memory | **not run**: the Float32 kernel fails to compile for the GPU (an `InvalidIRError`, dynamic dispatch inside the per-site closures `d1`/`inj`/`zf` of `soc_site!` with Float32 arguments; the Float64 kernel compiles and the Float32 reference runs on arrays). Open: make those closures type-stable in Float32 (the mixed Float32/Float64 arithmetic must come from a remaining Float64 value in the term functions or the closures' captured constants), then run `fp32_fused.jl`'s soc part |

Inference for the H100 (not measured): the H100 runs FP64 at half its FP32 rate, so the FP64
kernels there sit between this GPU's Float32 and Float64 regimes, nearer the Float32 one; the
Float32 ratios (tpot: 1.2× at 400 sites, 1.0× at 2 000; the kernel's global part 2× cheaper
than XLA's at small batches, equal at larger ones) are the better predictor of the FP64 ratios
there than this GPU's FP64 ratios (0.5–1.0×). That leaves the tpot global-part kernel as a modest
gain at best, bounded by the per-column part; the per-column fusion (§4 step 5) remains the step
that changes the balance, and the cluster measurement remains the decision.
