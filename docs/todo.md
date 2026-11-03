# To-do list

Status as of this round. `[x]` done, `[~]` partly done, `[ ]` open. The design is
described in [setup_overview.md](setup_overview.md).

## Done in this round (for reference)

- [x] Sparse graph-Haar bases (binary contrasts, weighted Haar for capacities), cell-mean balance projection, sparse grid solver
- [x] Local exact kernel: incremental costs, exact piecewise-exponential line draws, overcomplete move families
- [x] Physical formulation options: SoC storage, renewables as negative load, physical measure
- [x] Flow-based hierarchy: congestion-aware proxy transport modes (batched ADMM), active-set frames, transport and storage multiscale moves
- [x] Exact T2 compensation (kinked), uncompensated transport families
- [x] Adaptive learned modes with diminishing adaptation (transport directions, seasonal day-aligned shapes)
- [x] Learned linear compensation (response coefficients per node and proxy class)
- [x] Localized transport moves (dominant Haar atoms of proxy and learned modes)
- [x] Monotone-price bias fit; online bias learning with diminishing adaptation
- [x] Transported peak-set capacity moves (α = 1)
- [x] Adaptive solver as a driver option; learned quantities written to the output
- [x] LP-free heuristic start; weather model cache; 256-node PyPSA-Eur network
- [x] Diagnostics: ESS, split-R̂, sampled nodal prices
- [x] Reactant performance assessment ([reactant_performance.md](reactant_performance.md))

## Correctness and validation

1. [x] End-to-end exactness test (`test/test_exact.jl`, ≈ 2–3 min, 3 nodes × 120 h). The operation part at fixed (k, x) passes: full adaptive kernel vs atoms + plain linear families, all |z| < 1.7. Atoms alone do not mix and are not a usable reference. The capacity part (kscale vs kscale + kpeak) passes |z| < 4 but sits at z ≈ −3.6/−3.9 for cost and shed; longer runs are checking it
2. [~] LP-free convergence criterion (implemented: `src/convergence.jl`, `nchains` mode in `src/chains.jl`, `configs/chains40.toml`; first screening at 40 nodes daily, α = 0, 4 chains × 100 sweeps: nothing converged, max R̂ 4.6, bulk ESS 4–18. Slowest: capacities, the annual weather means and the regimes, then the operation through them): several chains from different starts, rank-normalized split-R̂, bulk/tail ESS and MCSE of cost, capacities, shed and prices (plan: [validation_plan.md](validation_plan.md) Part A)
34. [ ] Comparison with PyPSA-Eur, including the weather averaging (single-year and stacked multi-year LPs, MGA envelopes, then the original workflow on ERA5): [validation_plan.md](validation_plan.md) Part B
3. [~] α = 1 end to end: bias learned from the chain alone (mechanics work); weather diagnostic against exact draws still to do
4. [x] Post-correction of the bias (`bias_post`, `src/bias_post.jl`): b̃ refitted after the run from all scores, self-normalized weights exp(b̂_used − b̃), weight ESS, weighted quantities and weather diagnostics against exact draws in `post/…`. First short α = 1 run (40 nodes, 10 + 12 fine steps): weight ESS 1 of 12. One online refit (after observation 16) moved b̂ at the sampled states by ≈ 10⁷ nats, mostly through ĥ(k) at capacities far from the fit. The same instability let the storage capacities run away in that run (batteries 67 → 453 GW).
33. [ ] Stabilize the α = 1 bias early in a run: a long enough level-0 warmup before fine-level refits, damping or a trust region for the refits (limit the change of b̂ at recent states), and regularization of ĥ's slopes

## Sampler efficiency (CPU)

5. [x] Localized cross-node pair moves: `xloc` and the compensated `xlocc` (T2 at the leak nodes); at 256 nodes 4.7–8× more move per µs than the global `xpair` ([perf_notes.md](perf_notes.md))
6. [x] Cell and cell-pair moves sized by level: level ∝ (number of cells)^`cell_level_power`
7. [x] Allocation-free move evaluation: 0 B per move, sweeps 1.3–2.7× faster (40 and 256 nodes), merged into main. The recorded-term (cached) evaluation is used only for the plain model; moves with the model-completion terms take the plain path (`cacheable`)
31. [ ] Recorded-term evaluation for the model-completion terms (losses, soft limits, lossy links, vintages, commitment, weather entries). Measured at 40 nodes × 6 h, 8 threads: fine step 9.2 s with all terms vs 4.0 s plain (weather sweep at 5 %: 2.3 s vs 1.2 s)
8. [~] Faster intraday relaxation from far starts: seasonal day-aligned shapes with learned compensation done; learned dispatch shapes with learned compensation open

## Samplers

26. [ ] Parallel tempering: chains at a ladder of (β_v, β_k) temperatures with swaps; possibly tempering only the operational temperature, or the bias strength

## Experiments (after everything else; no expensive runs before)

9. [ ] Sweeps to equilibrium against network size (40 / 256, then 512 / 1024 nodes), with criterion 2
10. [ ] Capacity mixing at α = 1 with peak-set moves in a full run
11. [ ] Time hierarchy: coarse-heavy multigrid schedule; wall time to stationarity at 24/12/6/3/1 h
12. [ ] Memory estimate for full Europe hourly
27. [ ] Final profiling, once the known bottlenecks are fixed

## Formulation and modeling

16. [ ] Reference measure: open design question (physical vs utilization)
17. [x] From the formulation review ([model_completion.md](model_completion.md)):
    - balance frame: realized as exact T2 moves, the state stays nodal (§5);
    - signed lossy links; merged co-located vintages with brownfield floors and fixed thermal fleets;
    - canonical CO₂ price, emissions output, optional mean-cap tuning during warmup;
    - quadratic losses (piecewise-linear interpolant on a fine grid) and soft thermal limits; requirement-tracking T2 kinks;
    - fleet-level unit commitment (default parameters are assumptions)
18. [x] Weather moves at fine levels: centered regime FFBS, non-centered innovation and z0 moves with exact compensated line draws (`fine_weather`)
19. [x] Aggregated outputs of sampled nodal prices (per country and system, per season)
32. [ ] Non-centered regime moves at the fine level (the centered update hardly changes m given the fine weather)

## GPU / Reactant

20. [ ] Shared random move order per parity class; node colouring within a step
21. [ ] Port the array-first parts (ADMM, frame projection, principal components)
22. [~] Port the local kernel with chains as a batch dimension (padded size buckets, one compiled program per sweep). Done: phase 1 (device model, exact cost), phase 2 (term batch, exact dense line sampler, plain families gen/uc/pair/soc/link/θ, conflict-free rounds, compiled multi-sweep program; [reactant_phase2.md](reactant_phase2.md)), capacity moves kscale/kpeak with the bias on the device (`device/src/kmove.jl`, `bias_dev.jl`). All verified against the CPU reference with mock random numbers, on arrays and compiled. In progress: phase 3 (compensated and multiscale moves, one program with the capacity moves)
35. [ ] Check whether the CPU reference states of `test/test_exact.jl` are equilibrated in the storage/dump directions (the phase-2 agent saw exact SoC coordinate sweeps move dump by −366 ± 73 from them); a longer CPU reference run would tell slow mixing from bias
28. [x] Before 21/22: Reactant-first redesign review of model and sampler ([reactant_design.md](reactant_design.md))

## Data and scale

23. [ ] ERA5 weather for full Europe
24. [ ] Larger PyPSA-Eur network builds (512 / 1024 / full)
25. [ ] Reference LPs on HPC

## Shelved

Ideas kept for later, each with the condition under which it comes back.

**MLDA and coarse-level chains** (revisit only if the hierarchical overcomplete moves turn out not to be enough for convergence):

- **MLDA** (split-state: coarse chain on the coarse coefficients, fine details kept and perturbed; fine-level acceptance ≈ 0). The coarse levels stay for initialization and, possibly, multilevel estimators. If MLDA is revisited, consider an **overcomplete state**: coarse and fine coordinates both kept as a redundant representation, so that coarse chains act on the full state instead of splitting it into coarse part and details. That avoids the fixed-details problem behind the ≈ 0 acceptance.
- 15. Coarse-level consistency: a temperature per level, fiber entropies, peak factors for coarse capex. Needed for MLDA and multilevel estimators, not for the fine sampler.

**Move-set refinements** (revisit if convergence at scale is not good enough, or the final profile shows one family dominating):

- 29. Per-family diagnostics: mean |t| relative to the direction's natural scale (blocked directions), acceptance of the Jacobian Metropolis steps, share of the sweep time.
- 13. Learned *nonlinear* transported details as move compensation, only if item 29 shows blocked families (linear learned compensation is done). Independent of MLDA, but gated on 29.
- 30. Delayed acceptance with surrogates for expensive moves: line draw on a cheap surrogate, then a Metropolis correction with the exact change. Candidates: capacity moves at α = 1 (bias evaluation), multi-step storage moves at hourly resolution, fine-level weather moves. A cost reduction, not a mixing device.

**Initialization:**

- 14. Learned conditional prolongation. Independent of MLDA, but the prolongation rules tried all relax to about the same level within 25 sweeps, so there is little to gain.
- Recursive decomposed LP start (slower than the full LP at 40 nodes; the cheap variant is noted in flow_hierarchy.md §14).

**Other:**

- Temporal block moves (no gain at equal time).
