# Fused kernels for the device port: an exploration (2026-11-05)

Development note, work in progress on a side branch (not merged, nothing under `device/src/`
changed). Code in `device/experimental/`: its own environment (`Project.toml`, `Manifest.toml`:
ESMDevice developed into it, CUDA.jl 6.2.2 and KernelAbstractions 0.9.45 added, Reactant kept at
0.2.304 / Reactant_jll 0.0.438), the kernel `fused_jump.jl`, the check-and-timing script
`tpot_fused.jl`. The session was cut short before the prototype ran against the package's draw;
§4 says what is verified and what is not.

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
Its share is the first number the script measures (the round with and without the global part).

## 2. Route and constraints

KernelAbstractions kernel launched inside the Reactant-traced program (Reactant's
`ReactantKernelAbstractionsExt` + `ReactantCUDAExt`): the kernel is compiled by GPUCompiler to
LLVM bitcode and embedded as an `enzymexla.kernel_call` with the block and thread dimensions and
the shared-memory size as operands; traced arrays are passed as `CuTracedArray`, traced scalars
as `CuTracedRNumber`; arrays written by the kernel become outputs of the call. Checked on this
machine (Reactant 0.2.304, CUDA.jl 6.2.2, driver 580.95, RTX 3050 Ti):

* a kernel with `@localmem` shared arrays and `@synchronize` barriers inside `while` loops
  (a block-wide Hillis–Steele scan) compiles in 2.2 s and runs correctly on the GPU
  (error 1.4e-14 against `cumsum`);
* `raise = true` (the kernel raised to HLO, the route to the CPU backend) **fails** for that
  kernel ("failed to raise func": the raising pass does not handle the `scf.while` loops with
  barriers and address-space-3 memory). Raising covers elementwise, affine kernels (Reactant's
  own tests); a sort/scan kernel cannot be tested on the CPU backend this way. The array-level
  check of the testing contract therefore has to be a Julia reference of the same per-site
  algorithm (`fused_draw_ref`, the same `fused_tail!` code as the kernel), and the compiled
  check runs on the GPU only;
* KernelAbstractions' own CPU backend cannot run the kernel either (`@synchronize` inside
  control flow is not supported there);
* `@Const` arguments support linear indexing only (2-d indexing of a `Const` falls back to a
  dynamic call and fails to compile); the kernel indexes linearly;
* CUDA.jl must be loaded for the trace even when the target is not CUDA; it is functional here
  (CUDA 13 runtime, the artifacts were already in the depot);
* static shared memory is limited to 48 KB per block: for `tpot` (N = 1 478, cap 360, 256
  threads) the kernel needs about 38 KB (sort keys 2 048 × 12 B, five scratch vectors of
  cap + 2, three reduction buffers); `sblk32` (4 523 entries) needs the 8 192-slot sort buffer
  (98 KB) and so dynamic shared memory or a two-pass sort;
* FP64 throughout, as the package; a Float32 build would halve the shared memory.

Kernel design (`fused_draw_kernel!`): one block per (chain, site), NT threads; load the entries
with their original index as a stable tie-breaker (entries at exactly 0 become +Inf and their
jumps are folded into one entry at 0 with index 0, as `jump_points1`); bitonic sort over the
next power of two; cap; Inf entries moved onto the largest finite point with jump 0; the
real-candidate count, J0 and |J0| by a block reduction; then thread 1 runs the serial tail
(`fused_tail!`: the prefix and suffix sums from 0 outwards, the flat-slope tolerances, the rays,
the segment log-masses, the inverse CDF, the truncated-exponential offset, the improper-ray
rule), ≤ cap + 2 entries, the same arithmetic as `jump_values1`/`sample_core1` with the partial
sums accumulated serially (another association order than the two-level scan `cumsum1`, so
"equal to round-off", not bitwise). The serial tail is the part to parallelize next (block
scans), it costs about cap × a few dependent operations per site.

The wrapper `sample_jump_fused(tb, k, u1, u2; K, cap, nt)` calls `jump_parts`, lays the entries
out as (N, C, B) (the package's own layout, the entry axis fastest, so a site's entries are
contiguous) and launches the kernel; on Julia arrays it runs the reference. `do_round_fused` is
`do_round` with this draw; the Jacobian step, slot changes and commit are unchanged.

## 3. Status

| step | state |
|---|---|
| environment with CUDA.jl and KernelAbstractions, Reactant unchanged | done |
| toolchain: a shared-memory/barrier kernel inside a Reactant program on the GPU | works |
| raising such a kernel to HLO for the CPU backend | fails (constraint above) |
| kernel, Julia reference, wrapper (`fused_jump.jl`) | written, not yet run |
| arrays: `sample_jump` against `fused_draw_ref` on a `tpot` round; GPU: `do_round` against `do_round_fused`; timings of the two rounds and of `jump_parts` alone (`tpot_fused.jl`) | script written; its first run stopped at the data path (a worktree has no `data/`; fixed to fall back to the main checkout's `model.h5`); no result yet |

Expected from the design, to be confirmed by the script: the global part of a `tpot` round is
about 25–35 % of its time (the payload sort 15 % plus the scans and the slot-crossing sort);
the fused kernel for it should take a few milliseconds per round (about 20 k sites, 40 blocks
resident per SM pair at 38 KB shared memory), so the round should speed up by roughly that
share (1.3–1.5×), and the same for every slope-jump family. A larger gain needs the per-column
part fused as well.

## 4. Next steps

1. Run `device/experimental/tpot_fused.jl` (from the repository root,
   `timeout 600 nice -n 19 julia --project=device/experimental -t 4 device/experimental/tpot_fused.jl`;
   `C`, `H`, `NSITE`, `NT`, `NREP`, `FAM` in the environment; start with `C=2 H=24 NSITE=200`):
   the arrays comparison, the GPU comparison (t and the loop state after the commit) and the
   three timings. Fix what it finds (the kernel has not been compiled yet).
2. Parallelize the tail (block scans with a temp array, as the scan probe); measure the share
   of the serial tail first.
3. The same kernel on `sblk8`/`sblk16` (`FAM=sblk8`) and `tsoc`; `sblk32` needs dynamic
   shared memory.
4. Fusing the per-column part: the scalar term functions already exist (`gen_c`, `uc_val`,
   `soc_c`, `link_c`, `bal_val`, the kinked argument `argloc`), so a per-column thread of the
   block can compute its local points, its cost jumps and its slot injections and append its
   entries to the shared list; the balance-slot crossings need the slot's points sorted with the
   residual (a second, smaller in-block sort per slot). Estimate: one to two weeks for the
   share-preserving families (`tpot`, `tsoc`, blocks), given the verified global part; the plain
   dense families (`soc`, `gen0`, `theta`) are a separate, simpler kernel (one thread per site,
   everything in registers) that would matter for the fast set (35–47 % of it in `soc`).
5. Testing contract for kernels: the Julia reference of the per-site algorithm is the
   array-level check (no raising), and the GPU-compiled kernel is compared against both the
   reference and the package's XLA draw on the same inputs; FP64 and the GPU's libm differ from
   the CPU's by ulps, so "equal to round-off".

## 5. Recommendation

The route is viable: KernelAbstractions kernels with shared memory and barriers run inside the
Reactant program on the GPU without touching the package, and the global part of the slope-jump
draw is a natural first kernel because it is shared by the 94 % of the sweep that the
compensated, block and cell families take. Its payoff alone is bounded by that part's share
(expected 1.3–1.5× per slope-jump family); the larger step, fusing the per-column term
evaluation into the same block, is where the "1–3 ns per move" estimates of
[system/performance.md](system/performance.md) come from and is a one-to-two-week effort per
term structure once the global part is verified. The CPU backend cannot run such kernels
(raising fails), so the testing rules for kernels must accept a Julia reference as the array
level and the GPU as the only compiled check.
