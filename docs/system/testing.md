# Testing strategy

This document describes how the code is tested: the exactness tests of the CPU sampler, from
deterministic consistency checks to an end-to-end comparison of kernels, and the
verification of the device port against the CPU reference with mock random numbers. It lists
what each test file checks and how to run the tests. The validation of results (convergence,
comparison with PyPSA-Eur) is in [validation.md](validation.md); the code structure in
[architecture.md](architecture.md).

## 1. Principles

**Deterministic checks are the gate.** The sampler does not target one exact density
([ensemble.md](ensemble.md) §3.1); it is validated against LP-based results
([validation.md](validation.md)). What the tests guarantee is that every move is a correct line
move for the density it is written for, and that is checked without statistics, in seconds,
by the move-consistency suite `test/test_moves.jl` (being built; Status: missing): for every
family, the potential is linear between the walk's points, the incremental cost equals the
global difference, the curve is reversible (the same discrete choices at the new state, −t
returns to the start, inverse Jacobians, equal slice widths in the slice coordinate, symmetric
Metropolis proposals), and the caches equal a rebuild after a threaded sweep. These are the
conditions that the bugs of 2026-11-04 violated (the `kpeak` slice width, misplaced balance
roots on storage lines, the non-reversible storage compensation); none of them was found by
the statistical test below. No new statistical tests are added until the performance work is
much further. Besides the suite, the existing deterministic checks: the incremental cost change of a move equals the global
cost difference; caches stay equal to a rebuild from the coefficients; gradients match
finite differences; the invariants of compensated moves (shares and excess) hold across their
kinks; non-centered weather coordinates reproduce the anomalies.

**Existing distribution checks of single draws.** A single line draw is compared with
quadrature of its conditional (Kolmogorov–Smirnov over many draws of a fixed move), including
the stiff shed case, the two-step SoC case, requirement-tracking kinks, commitment and the
Gaussian weather prior.

**End-to-end comparison, optional diagnostic** (`test/test_exact.jl`, not a gate): on a 3-node sub-network at a 120-hour step
(73 steps) with fixed weather, the full adaptive kernel (all local, flow, learned, long-range
storage and capacity moves, with adaptation) must reproduce the equilibrium of a simple reference chain for a set
of observables within Monte Carlo error ($|z| < 4$ with ESS-based standard errors). The
reference for the operation is the atoms plus the plain linear families, which are exact by
construction (atoms alone do not mix in usable time); the capacity test compares `kscale`
against `kscale` + `kpeak`.

> **Status: deviation.** The capacity part of `test/test_exact.jl` depends on the seed: run
> alone with a fixed seed it gives identical results on main and with the long-range storage
> moves (it does not use them), and it passes with seed 21 ($|z|\le 1.84$) but fails with seed
> 22 (cost $z = -5.6$, gas capacity $-3.9$, line flow $2.7$) on the same code (2026-11-04). Its
> chains do not mix within 2000 steps and its error bars, from the chains' own ESS, are too
> small, so it decides neither whether `kpeak` is exact nor whether a change broke something.
> It needs references that have converged (longer runs, several seeds, or an exact
> enumeration on a smaller problem; ledger P6, H8). The operation test passes.

**The device port is verified against the CPU reference, not statistically.** The same
prepared moves and the same mock random numbers drive both sides (a mock `AbstractRNG` on the
CPU, input arrays on the device), for a few rounds; then $t$, node values, caches and costs
are compared exactly where the arithmetic is the same and to round-off otherwise. Order:
(1) the device code on plain Julia arrays against the CPU, (2) the same code compiled by
Reactant on the CPU backend, which must equal (1), (3) the GPU only for timings. No KS tests,
z-tests or many-chain runs for the port; tests take seconds to minutes. The production device
RNG is Reactant's own (`ReactantRNG`), whose stream differs from the CPU's, which is why the
comparison uses given uniforms.

## 2. CPU tests

| file | checks |
|---|---|
| `test/runtests.jl` | level transfers on graph Haar bases (split/merge round trip, block means commute with restriction); FFBS against the exact regime posterior; cost gradients against finite differences on the 40-node data; graph Haar bases |
| `test/test_local.jl` | sparse Haar field bases (orthonormality, sparsity, cell projections, gauge-free angles); segment helpers (`logseg`, truncated exponential); local moves on the 40-node model: cache cost = global cost, `xloc` templates keep the balance except at the leak nodes, cell-level probabilities, recorded walk evaluation bitwise equal to the plain one, incremental cost = global, cache consistency, line draws against quadrature (single-step and linked two-step), the potential linear between the walk's points on lines with storage terms (`spair`, `tsoc`, linked storage moves), scaling moves keep the caches consistent; long-range storage moves (`slong`, `slevel`): incremental cost = global, recorded = plain evaluation, lazy interior = full evaluation, caches after `storage_long_sweep!`, reversibility of the curve and inverse Jacobians (also for `tsoc` and the storage plateaus, across the compensation kink), walk linearity, line draws against quadrature |
| `test/test_flow.jl` | batched congestion ADMM against a QP; transport modes and angle basis orthonormal and gauge-free; storage shapes; face projection of the modes; flow moves: incremental = global, shares and excess invariant across the kink, caches; adaptive modes update on schedule and give gauge-free directions and zero-end shapes; `kpeak` keeps the caches consistent |
| `test/test_bias.jl` | the monotone fit gives a non-decreasing implied price within $[-c_{\rm dump}, c_{\rm shed}]$ and $\theta_{u_2}\le0$ (a non-increasing $F_2$); the per-coordinate $\hat h$ (`hsat`, `hcoord`): the quadratic, flat beyond the vertex, continued outside the data range with a clipped slope that never points away from the data, continuous, derivative against differences; $\hat b$ carries $(1-r)\log J$ exactly (global value, cache) and $\hat h$ regresses $s_k-\partial_k\log J-\partial_k\hat g$ with the data range in its sums; the refit floor (`bias_online_min`) and the trust region (the step is a convex combination of the old and the new fit, within the bound over the recent states, satisfies the monotone constraints; a wide region takes the full step); the tables of $F_1$, $F_2$ equal the analytic ramp sums at the grid points, are within the interpolation bound between them, continue linearly and have non-increasing secant slopes; the lift keeps $\theta$, $\bar D$ and the prefix coefficients (and $\hat h$ of capacities without details), resets the sums and redraws the sample steps; the bias cache, with $\hat h$ coordinates in every branch near the state: incremental $\Delta\hat b$ = global difference for capacity changes of renewable, gas and storage units and for fine weather moves of every level, the cache after commits equals a rebuild, and after `kscale`, `kpeak` and weather sweeps equals `bias_term`; $\partial_k\log J$, the design $\nabla\hat g$ in the weather state coordinates and $\partial_k\hat g$ (analytic $\hat g$), $\partial\hat h$ against finite differences, the measure term in $s_k$; post-correction records reproduce $\hat b$; the Metropolis bias correction of a fine weather move: repeated moves of one coordinate against quadrature of the line target with the bias (KS), which differs from the bias-free conditional |
| `test/test_kbox.jl` | `kscale` and `kpeak` at the capacity box, at $\alpha = 0$ and $\alpha = 1$, from a state with raw capacities above their box: per move (inside the box, above it and across its edge; dispatchable and storage units) the line density without its Jacobian against the change of the global log target, the reverse move returns to the start with the opposite density and Jacobian, the local and bias caches equal a rebuild; one `kscale` and one `kpeak` sweep bring the raw capacities above the box toward it |
| `test/test_model.jl` | all model-completion terms on (40 nodes): incremental = global for random local and transport moves, caches, line draws with requirement-tracking kinks and with commitment against quadrature, long-range storage moves with requirement-tracking end steps (incremental = global, caches, a line draw against quadrature), level-0 gradients against finite differences, fine-level weather moves (incremental = global, caches, the non-centered representation and the regime step reproduce the anomalies, line draws with the Gaussian prior), the local $z_0$ quadratic against `logp_z0` |
| `test/test_hpc.jl` | durable files of the CPU driver: an interrupted cache write leaves neither the entry nor a temporary file, a killed writer's temporary file is never read, a complete write appears under its name only, a mirror entry is copied in byte for byte (the mirror untouched), an unreadable entry is recomputed, the environment overrides the cache keys; a process with `ignore_sigterm` survives SIGTERM and exits with its own code (75); a short α = 1 run (2 chains, 40 nodes daily) stopped by the stop file in the level-0 warmup, the fine warmup and production (exit code 75 each time, the stop file consumed, the restart count in the snapshot), each resume starting from exactly the saved states, the last finishing with exit code 0 with the traces before the stops kept; the cache of a resumed run filled from the mirror, byte-identical, no temporary files left |
| `test/test_convergence.jl` | multi-chain diagnostics on synthetic chains: iid passes, AR chains give the expected ESS, location and scale differences raise $\hat R$, random walks fail, warmup detection |
| `test/test_exact.jl` | end-to-end comparison of chains (§1), an optional diagnostic, not a gate (seed-dependent); about 2–3 minutes; skipped with `EXACT_SKIP=1` |

Tests that need `data/processed/model.h5` are skipped if it is absent. The CPU tests a new move
family needs are listed in [moves/adding_a_move.md](moves/adding_a_move.md) §4.

## 3. Device tests (`device/test/`)

| file | checks |
|---|---|
| `test_cost.jl` | device cost = `energy_cost` (all terms), residual = `LocalCost.r`, on arrays and traced |
| `test_sampler.jl` | term-batch $\Delta C(t)$ = CPU `eval_move!` per family; conflict-freeness of every round; at most two sign changes of every secondary quantity on every site (KS only with `KS=1`) |
| `test_mock.jl` | the main comparison: every site of two rounds of every plain family with the same uniforms on both sides; $t$ per draw, node values, $r$ and costs after committing; Reactant against Julia arrays |
| `test_rounds.jl` | one sweep on arrays and traced with the same uniforms; many sweeps in one compiled program: $r$ cache drift and cost of the downloaded state |
| `test_comp.jl` | compensated families (`:spair`, `:tsocL`, `:tpotL` with all terms; `:tsoc`, `:tpot` without losses and lossy links), storage blocks (`:sblkL2/4`; `:sblk2/4`), cell templates (all levels): the same explicit CPU moves and uniforms, $t$ (own points, and the device sampler on the CPU walk's points), the Jacobian decision, node values, $r$ and costs after committing, the guard rate; then Reactant against arrays |
| `test_program.jl` | the merged program (operation sweeps + capacity pass) against the two compiled calls; with `FULL=1` all families in it |
| `test_jump.jl` | the two forms of the line sampler on every round of the compensated, block and cell families (models with and without losses), Julia arrays: the potential linear between consecutive points of either form (midpoints evaluated directly), the slope-jump potential at its points equal to the direct evaluation, the same $t$ from both forms with the same uniforms, the guard rates |
| `test_kmove.jl`, `test_kpeak.jl`, `test_kren.jl` | capacity moves against the CPU Metropolis forms with the same proposals |
| `test_kbox_dev.jl` | `kscale` and `kpeak` from a state with raw capacities above their box against the CPU Metropolis forms (arrays and compiled) |
| `test_kmove_bias.jl`, `test_bias_dev.jl` | with the bias lifted to the chains' level: capacity moves with the bias against the CPU Metropolis forms (arrays and compiled); the device bias change of the same proposals against the CPU bias cache (`bias_delta_k`) and against differences of `bias_term`, the aggregates after commits against a fresh pack, and the compiled `delta_bias` against the arrays |
| `test_driver_parts.jl` | device chain parameters = host chain parameters |
| `test_exact_dev.jl`, `diag_exact_soc.jl` | statistical runs; off unless `RUN_STAT=1` |

Shared setups: `setup40.jl`, `setup_all_terms.jl` (40 nodes, all model terms), `helpers.jl`
(CPU reference moves built as in `local_sweep!`), `mockrng.jl` (`MockRNG`). The device tests of
a new family: [moves/adding_a_move.md](moves/adding_a_move.md) §6.

## 4. Running

```bash
nice -n 19 julia --project=. -t 4 test/runtests.jl
nice -n 19 julia --project=. -t 4 test/test_exact.jl      # optional diagnostic, not a gate
cd device && nice -n 19 julia --project=. -t 4 test/test_mock.jl
```

A smoke run of the CPU driver: `scripts/run.jl` on a copy of `configs/chains40.toml` (a
mechanics configuration) with small `mlda_warmup` and `nsamples`.

Device tests run traced code on Reactant's CPU backend by default. Each fresh Julia process
costs minutes of start-up and compilation, so iterate in one persistent Julia session and
compile only once the array version passes.

> **Status: open.** The capacity part of `test/test_exact.jl` cannot detect invariance
> violations of the capacity moves (it missed the `kpeak` slice-width bug, and its result
> depends on the seed, see above); converged references would make it a test
> ([roadmap.md](roadmap.md)).
