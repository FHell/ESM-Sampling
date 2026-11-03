# The sampler as it stands: overview

One page per topic, with pointers to the detailed notes. Code paths in brackets.

## 1. Target

Weather x (multilevel regime-switching weather model), capacities k, dispatch v:

  π(x, k, v) ∝ p(x) · exp(−β_k capex(k) − β_v C_x(k, v) − b̂(x, k)).

* **Dispatch has perfect foresight per weather year:** C_x is the year's
  operational cost, near-optimal at temperature β_v given (k, x). Investment is
  chosen against the weather distribution, i.e. the two-temperature ensemble ρ₁.
  The learned bias b̂ removes the weather dependence of log Z_x(k).
* **Constraints are regularized:** every violation is priced at physical slopes,
  so there are no hard walls.
* **β is fixed by experience.**
* **The reference measure is an open design question.** The current choice:
  physical units for transport and storage; utilization or MW for dispatch
  (`measure`). See [sampling_formulations.md](sampling_formulations.md) §0 and
  [physical_formulation.md](physical_formulation.md).

## 2. Flow-based formulation ([flow_hierarchy.md](flow_hierarchy.md) §1)

* **Coordinates per step:**
  - angles θ (KVL exact, flows b(θ_a − θ_b)) and signed link flows;
  - storage state of charge e (`storage = "soc"`; charge and discharge derived
    from differences);
  - dispatchable generator utilizations ω.

  Renewables are negative load: output A·k, curtailment is nodal dump
  (`renewables = "load"`).
* **Balance** is priced, with one residual r per node (shed if positive, dump if
  negative).
* **The transport-first parametrization is the frame of the moves.**
  - At every node and step, the requirement Q = r + Σ κ clamp(ω) and the
    generator output P = Σ κω define shares σ = p/P and excess e = P − Q₊.
  - (σ, e) ↔ p is a bijection at fixed Q. A transport move at fixed (σ, e) is
    an exact coordinate line, with a kink at Q = 0 and Jacobian |P(t)/P|^{m−1}.

## 3. Bases and hierarchy

* **Space:** sparse graph-Haar bases on nested partitions (network → country →
  [regions of ≤ cap nodes] → node). Inside each parent the contrasts form a
  balanced binary tree ordered by the Fiedler vector. Capacities use weighted
  Haar (cell-constant k/k̄), so coarse capacities are spread by potential.
  [`spatial.jl`]
* **Transport modes:** from a congestion-aware proxy (weather + unextended
  network, batched ADMM projection per step), a PCA of line loadings. They are
  either the coarse angle coordinates of a nested basis (`flow_theta = "nested"`)
  or only moves (`"haar"`, large networks). Run-time active-set frames from
  (x, k) project them onto the predicted binding face. [`flowbasis.jl`,
  `flowmoves.jl`]
* **Time:** levels with block means (power fields) and decimation with linear
  interpolation (state of charge); dyadic blocks for moves.
* **Levels (M_ℓ, h_ℓ) and MLDA** [`mlda.jl`]. **MLDA is shelved** (docs/todo.md):
  the hierarchy acts through the overcomplete fine-level moves. If it is revisited,
  it should use an overcomplete state (coarse and fine coordinates kept together)
  instead of the split state. The split-state design that was tried:
  - coarse chains on the coarse state;
  - fine details kept and perturbed (random walk, pCN), not redrawn;
  - delayed acceptance.

  With additive details the fine-level acceptance is ≈ 0 (the toy explains why:
  extensivity, entropy mismatch). The levels are used now for:
  - initialization;
  - multilevel estimators;
  - coarse moves embedded at the fine level (§4).

## 4. Fine-level sampler: an overcomplete move set ([local_kernel.md](local_kernel.md))

Every move is a line in the fine state space, sampled exactly from its
piecewise-exponential conditional (breakpoints at clamps, limits, cell-sum and
dynamics sign changes, compensation kinks). There are no step widths. Selection
depends only on (k, x), the frozen learned quantities and fixed probabilities,
so every move is π-invariant. The cost of a move is local, through incremental
node, line and cell caches.

| scale | families |
|---|---|
| fine, local | Haar atoms, single unit, same-node pair, cross-node pair (flow re-routed), node angle |
| coarse in space | cell (one carrier over a cell of any coarser partition), cell pair (two carriers, cell sum of the residual unchanged) |
| coarse in time | block versions of the above (dyadic blocks); storage plateau (energy moved between two blocks) and intra-block storage shape |
| transport | `tpot` (node angle), `tmode` (transport mode, face-projected), `tline` (binding-line normal), `tsoc` (SoC at an instant) |
| learned | `amode` (learned transport direction), `ashape` (learned storage shape), `adshape` (learned balance-neutral dispatch pair shape) |
| investment | `kscale` (k and its utilizations along the scaling group, with Jacobian); `kpeak` (k with the dispatch on the node's peak steps, residual or learned compensation; the α = 1 capacity move) |
| localized transport | `tcut`/`acut` (dominant Haar atoms of proxy or learned modes; cut transfers, compensation only at cell boundaries) |
| weather (level 0) | coordinate slice and weather with pro-rata shear |
| weather (fine levels, `fine_weather`) | centered regime FFBS; non-centered moves of the innovations of every level and of z0, exact line draws with the Gaussian prior, exact T2 compensation on the actual requirement ([model_completion.md](model_completion.md) §6) |
| localized pairs | `xloc`, `xlocc` (re-routing on a small ball; `xlocc` with T2 at the leak nodes) ([perf_notes.md](perf_notes.md)) |

## 5. Compensation of coarse and transport moves

What the fine structure the move does not set does when the move happens:

| mode | rule | exactness |
|---|---|---|
| none | the residuals absorb (export from surplus into shortage) | direction constant |
| additive | details kept as offsets (Haar atoms, cells, plateaus) | direction constant |
| T2 exact | shares σ and excess e kept: dispatch follows the requirement pro rata at deficit nodes and stops at surplus nodes; kink at Q = 0 | invariance of (σ, e); Jacobian by an independence Metropolis step |
| T2 requirement-tracking | as T2, with the actual requirement along the move (piecewise linear with losses, lossy links, weather): the kink is a secondary breakpoint, walked event by event | invariance of (σ, e); Jacobian at the drawn t |
| learned linear | dispatch responds to a requirement change with learned coefficients β_{n,c,i} (regression of each unit's output on the node requirement from the chain, per node and proxy class c = surplus or deficit from (x, k)); 1 − Σβ goes to the residual | coefficients fixed between updates and independent of v; no Jacobian |

## 6. Adaptation with diminishing adaptation (`adaptive.jl`, `bias.jl`)

* **Online bias** (`bias_online`, on by default): scores of the finest chain
  (−β_v ∇C carried to level-0 coordinates) enter the bias regression after every
  step. Monotone refits happen at observation counts 2^j (optional freeze). This
  matters most where there is no LP warm start, i.e. on the large models.

* **Learned from the chain's own states:**
  - the angle fluctuation covariance (transport directions);
  - intra-block fluctuation profiles of dispatch and state of charge, per dyadic
    length (temporal shapes);
  - the response coefficients β of §5.
* **Diminishing adaptation:** running means with weight 1/n; recomputed at
  sweeps 2^j only; optional freeze.
* **Exactness:** every kernel in the sequence is π-invariant. With the fixed
  families always present, diminishing adaptation plus containment gives the
  right limit (Roberts & Rosenthal 2007).

## 7. Initialization

* **`init = "lp"`:** the full LP at the first complete level (small models only).
* **`init = "heuristic"`:** proxy capacities, gas at nodal peak, interior dispatch.
* **`init = "decomposed"`, shelved** ([flow_hierarchy.md](flow_hierarchy.md) §14): at 40 nodes
  it takes 113 s, more than the full LP, and the start is poor, so the heuristic
  start plus relaxation is used instead. The design: recursive small LPs along the hierarchy
  [`decompose.jl`]:
  - countries: joint investment and operation, transport between cells;
  - per parent cell: children with the parent's capacity totals and the base-line
    flows already assigned fixed as border injections, transfers between
    children free, base flows by capacity share, DC lines with KVL at node level;
  - hourly operation LPs per region and block, with SoC fixed at the block ends;
  - global angles from the injections.

  Every LP is bounded in size, and each depth is parallel.
* **Prolongation of a coarse state:**
  - block-constant;
  - T2 (shares and excess kept);
  - T2 clamped to the unit boxes;
  - learned conditional details (planned).

## 8. Diagnostics and outputs

* ESS and split-R̂ per quantity.
* Sampled nodal prices E[c_shed 1(r > 0) − c_dump 1(r < 0)], compared with the LP
  duals where available.
* Weather diagnostics.
* Multilevel estimator pairs.

## 9. Status in numbers (40 nodes unless noted; details in the notes)

* **Model completion** ([model_completion.md](model_completion.md)): line losses and
  soft limits, lossy links, merged vintages with brownfield, canonical CO₂ price, fleet
  unit commitment and fine-level weather moves are options of the same exact kernel.
* **CPU kernel after the performance work** ([perf_notes.md](perf_notes.md)): sweeps
  1.3–2.7× faster at 40 and 256 nodes, 0 B per move.
* **GPU port** ([reactant_design.md](reactant_design.md)): phase 0 (spike) and phase 1
  (device model and exact cost, all terms, = CPU to 10⁻¹²) done.

* **Local kernel:** about 25 sweeps to stationarity from the level's own LP at
  24/12/6 h. Cost per sweep is linear in nodes × steps.
* **Formulation:** SoC storage + renewables as load cut the excess generation
  cost 2.4× at equal time and remove the slow drifts.
* **Flow sweep with exact T2:**
  - 40 nodes, 6 h: slightly better per sweep than local;
  - 256 nodes, daily: about 2× better per sweep;
  - in both cases 1.8–2.7× more expensive per sweep.
* **Adaptive learned moves, 6 h far start:** the best per sweep (38× after 32
  sweeps vs about 61× local), but 2.2× the cost per sweep.
* **Time axis:**
  - coarse and intra-block observables decorrelate in 2–4 sweeps at 24/6/3 h
    (block lengths 1/4/8);
  - intra-block details are only moderately compressible at 3 h;
  - relaxation from prolonged starts is the slow part.

## 10. Open

* **Post-correction of the bias** (implemented: `bias_post`; see todo item 4 for the first result). Record (x-features, k, scores) per
  sample, refit b̃ after the run from all scores (monotone QP), reweight with
  w = exp(b̂ − b̃), report the weight ESS, and check the weather marginal against
  exact draws before and after, optionally with a density-ratio correction on
  coarse weather features. The limits: the weights degenerate if b̂ is off by
  more than a few nats, and reweighting cannot repair regions the chain did not
  visit.

* Sweeps to equilibrium against network size (40 / 256 / 512+ nodes) from the heuristic start.
* Learned conditional prolongation.
* Localized (sparse) transport modes to cut the cost per flow sweep.
* Capacity mixing at α = 1 in a full run with a learned bias (the moves exist).
* GPU port (Reactant): the flow basis and frames are already array-first; the
  local moves need the padded, batched form.
* Full-Europe runs (heuristic start; reference LPs on HPC; ERA5 weather).
