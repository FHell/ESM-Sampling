# Testing strategy

This document describes how the code is tested: the exactness tests of the CPU sampler, from
deterministic consistency checks to an end-to-end comparison of kernels, and the
verification of the device port against the CPU reference with mock random numbers. It lists
what each test file checks and how to run the tests. The validation of results (convergence,
comparison with PyPSA-Eur) is in [validation.md](validation.md); the code structure in
[architecture.md](architecture.md).

## 1. Principles

**Exactness first, deterministically.** Every move must leave the target invariant. Most of
that is checked without statistics: the incremental cost change of a move equals the global
cost difference; caches stay equal to a rebuild from the coefficients; gradients match
finite differences; the invariants of compensated moves (shares and excess) hold across their
kinks; non-centered weather coordinates reproduce the anomalies.

**Distribution checks where a draw is random.** A single line draw is compared with
quadrature of its conditional (Kolmogorov–Smirnov over many draws of a fixed move), including
the stiff shed case, the two-step SoC case, requirement-tracking kinks, commitment and the
Gaussian weather prior.

**End-to-end exactness** (`test/test_exact.jl`): on a 3-node sub-network at a 120-hour step
(73 steps) with fixed weather, the full adaptive kernel (all local, flow, learned and capacity
moves, with adaptation) must reproduce the equilibrium of a simple reference chain for a set
of observables within Monte Carlo error ($|z| < 4$ with ESS-based standard errors). The
reference for the operation is the atoms plus the plain linear families, which are exact by
construction (atoms alone do not mix in usable time); the capacity test compares `kscale`
against `kscale` + `kpeak`.

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
| `test/test_local.jl` | sparse Haar field bases (orthonormality, sparsity, cell projections, gauge-free angles); segment helpers (`logseg`, truncated exponential); local moves on the 40-node model: cache cost = global cost, `xloc` templates keep the balance except at the leak nodes, cell-level probabilities, recorded walk evaluation bitwise equal to the plain one, incremental cost = global, cache consistency, line draws against quadrature (single-step and linked two-step), scaling moves keep the caches consistent |
| `test/test_flow.jl` | batched congestion ADMM against a QP; transport modes and angle basis orthonormal and gauge-free; storage shapes; face projection of the modes; flow moves: incremental = global, shares and excess invariant across the kink, caches; adaptive modes update on schedule and give gauge-free directions and zero-end shapes; `kpeak` keeps the caches consistent |
| `test/test_bias.jl` | the monotone fit gives a non-decreasing implied price within $[-c_{\rm dump}, c_{\rm shed}]$; the tables of $F_1$, $F_2$ equal the analytic ramp sums at the grid points, are within the interpolation bound between them, continue linearly and have non-increasing secant slopes; the lift keeps $\theta$, $\bar D$ and the prefix coefficients (and $\hat h$ of capacities without details), resets the sums and redraws the sample steps; the bias cache: incremental $\Delta\hat b$ = global difference for capacity changes of renewable, gas and storage units and for fine weather moves of every level, the cache after commits equals a rebuild, and after `kscale`, `kpeak` and weather sweeps equals `bias_term`; $\partial_k\log J$, the design $\nabla\hat g$ in the weather state coordinates and $\partial_k\hat g$ (analytic $\hat g$), $\partial\hat h$ against finite differences, the measure term in $s_k$; post-correction records reproduce $\hat b$; the Metropolis bias correction of a fine weather move: repeated moves of one coordinate against quadrature of the line target with the bias (KS), which differs from the bias-free conditional |
| `test/test_model.jl` | all model-completion terms on (40 nodes): incremental = global for random local and transport moves, caches, line draws with requirement-tracking kinks and with commitment against quadrature, level-0 gradients against finite differences, fine-level weather moves (incremental = global, caches, the non-centered representation and the regime step reproduce the anomalies, line draws with the Gaussian prior), the local $z_0$ quadratic against `logp_z0` |
| `test/test_convergence.jl` | multi-chain diagnostics on synthetic chains: iid passes, AR chains give the expected ESS, location and scale differences raise $\hat R$, random walks fail, warmup detection |
| `test/test_exact.jl` | end-to-end exactness (§1); about 2–3 minutes; skipped with `EXACT_SKIP=1` |

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
| `test_kmove.jl`, `test_kpeak.jl`, `test_kren.jl` | capacity moves against the CPU Metropolis forms with the same proposals |
| `test_kmove_bias.jl`, `test_bias_dev.jl` | with the bias lifted to the chains' level: capacity moves with the bias against the CPU Metropolis forms (arrays and compiled); the device bias change of the same proposals against the CPU bias cache (`bias_delta_k`) and against differences of `bias_term`, the aggregates after commits against a fresh pack, and the compiled `delta_bias` against the arrays |
| `test_driver_parts.jl` | device chain parameters = host chain parameters |
| `test_exact_dev.jl`, `diag_exact_soc.jl` | statistical runs; off unless `RUN_STAT=1` |

Shared setups: `setup40.jl`, `setup_all_terms.jl` (40 nodes, all model terms), `helpers.jl`
(CPU reference moves built as in `local_sweep!`), `mockrng.jl` (`MockRNG`). The device tests of
a new family: [moves/adding_a_move.md](moves/adding_a_move.md) §6.

## 4. Running

```bash
nice -n 19 julia --project=. -t 4 test/runtests.jl
nice -n 19 julia --project=. -t 4 test/test_exact.jl
cd device && nice -n 19 julia --project=. -t 4 test/test_mock.jl
```

A smoke run of the CPU driver: `scripts/run.jl` on a copy of `configs/chains40.toml` (a
mechanics configuration) with small `mlda_warmup` and `nsamples`.

Device tests run traced code on Reactant's CPU backend by default. Each fresh Julia process
costs minutes of start-up and compilation, so iterate in one persistent Julia session and
compile only once the array version passes.

> **Status: open.** The capacity part of `test/test_exact.jl` passes with $|z|\le 2.1$ (cost
> 0.91, shed 2.05, gas capacity 0.49, mean SoC 2.13, line flow −1.16 on 2026-11-04). It did not
> distinguish the `kpeak` slice sampler before and after the width fix (both pass, $|z|\le 1.2$
> with the old width), so it is not sensitive to small invariance violations of the capacity
> moves; longer reference runs would make it so ([roadmap.md](roadmap.md)).
