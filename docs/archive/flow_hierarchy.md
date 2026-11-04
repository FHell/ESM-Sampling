# Flow-based hierarchical design

The sampler's variables are energy transport in space (potentials, links)
and in time (states of charge). Shares say which local technology covers the
resulting requirement. Investment follows from the dispatch, plus renewable
capacities as primary variables. The hierarchy is built from the weather data
and the unextended network, never from an LP solution. Background:
[sampling_formulations.md](sampling_formulations.md) §8 (transport-first T2),
Hofmann et al., EPL 124, 18005 (2018) (principal flow patterns).

## 1. Fine level (the exact target)

Per step τ:
* angles θ (KVL exact, f_e = b_e(θ_a − θ_b)), signed link flows u;
* storage states of charge e_s (SoC form: charge or discharge derived from
  e_{τ+1} − e_τ);
* dispatchable dispatch p_i = κ_i ω_i;
* renewables as negative load, with output A·k_ren;
* capacities k, explicit (α = 1: walls plus the bias).

The balance is priced, with shed and dump as the dependent unit (one residual
r per node: shed if positive, dump if negative). The transport-first
parametrization is realized as the frame of the moves (§6), exactly:
* at node n and step τ, P = Σ_{S_n} p_i over the generators, and the requirement
  Q = r + Σ_{S_n} κ_i clamp(ω_i), which changes by exactly g·t along a transport
  move;
* along the line, the shares σ_i = p_i/P and the excess e = P − Q₊ are held
  fixed (e may have either sign; −e is the shed while Q > 0), and
  P(t) = (Q + g t)₊ + e. (σ, e) ↔ p is a bijection at fixed Q, so the line is a
  coordinate line;
* at **deficit nodes** (Q > 0) dispatch follows the requirement pro rata and the
  residual stays fixed;
* at **surplus nodes** (Q ≤ 0) dispatch stays at e and the transport change goes
  into the residual (dump);
* dispatch is piecewise linear in t, with one known kink per node at
  t* = −Q/g, which is a primary breakpoint of the exact line sampler;
* the target carries |P(t)/P|^{m_n−1}, applied by an independence Metropolis step
  after the exact piecewise-exponential draw. Checked: (σ, e) invariant across
  the kink to 10⁻¹², and the Metropolis line chain reproduces π·J (KS 0.005).

Uncompensated transport families (`tmode0`, `tline0`, and the plain potential
moves) run alongside: the residuals at both ends absorb the change. These are the
moves that export from surplus into shortage.

**Open design question: the reference measure.** Which Lebesgue measure defines
the near-optimal ensemble (utilization shares vs physical MW/MWh, per
technology class; renewables as load removes the question for them) changes the
entropy terms in log Z and therefore the capacity and weather marginals
(sampling_formulations.md §0). The choices here (physical units for transport
and storage, utilization or physical for dispatch via `measure`) are a
reasonable starting point, not a settled decision. Everything below is a
reparametrization or a move set and works for either choice; only the
Jacobians of §1 and of the investment moves depend on it.

## 2. Transport basis (offline)

1. **Proxy capacities k̂.** Existing capacities for the fixed fleet. Investable
   renewables are scaled from their potentials so that their annual energy is a
   target share of the load (default 0.7).
2. **Proxy injections.** For every weather year and step: Δ_n = Σ_ren A k̂ − d_n,
   balanced copper-plate, p_n = Δ_n − (d̄_n/Σd̄)·ΣΔ.
3. **Congestion projection, one QP per step.** min ‖Δp‖² s.t.
   p + Δp = Lθ + B_C u, |b_e(θ_a − θ_b)| ≤ s_e, |u| ≤ ū. The output is the
   constrained angles θ_t and the binding set of every step.
4. **Transport modes.** PCA of the loading samples ℓ_t = diag(b/s) Kᵀ θ_t.
   - The top K loading patterns q_k are mapped back to angles,
     d_k = (WᵀW)⁺Wᵀq_k with W = diag(b/s)Kᵀ.
   - These are orthonormalized against the component constants.
   - K is chosen by explained loading variance (95 %, at most `K_max`).
5. **Angle basis.** The modes first, then the graph-Haar angle vectors with the
   modes projected out (Gram–Schmidt). At spatially coarse levels θ has the K
   modes. At the complete level it has all N − n_comp coordinates.

## 3. Active-set frame (run time, from (x, k) only)

For each step τ of a level, the QP of §2.3 is run on the current weather and
capacities (proxy injections at the current k), giving the predicted binding set
B_τ. The transport modes projected onto the null space of the binding rows,
d ↦ (I − W_Bᵀ(W_B W_Bᵀ)⁺W_B)d, move transport along the constraint face. The
binding lines themselves get normal directions: an injection pair at the line's
ends with re-routed flows. Cached per step, and recomputed only when x_τ or k
change. Since the frame never depends on v, every move stays exact.

## 4. Shares

Graph-Haar cells (network → country → node) for the dispatch fields.
* At coarse levels, dispatch is cell-constant (country shares), and the balance
  is the cell sum.
* Share moves: same-node pairs, and cell pairs (two carriers over a cell,
  injection-balanced).
* The pro-rata frame σ is the default compensation of every transport move.

## 5. Storage multiscale

Per unit, at every dyadic time level L:

  e(t) = interpolation of boundary values + Σ_k a_{b,k} φ_{L,k}(t) + detail.

* **Boundary states** (energy moved between blocks): decimation, exact.
* **Shapes φ_{L,k}.** Leading temporal principal components of the proxy
  intra-block SoC: the cumulative sum of the cell's proxy residual minus its
  block mean. Per cell and level, they vanish at the block boundaries.
* **Aggregation over a cell and storage class.** Weights E_s/ΣE give an equal
  utilization change for every unit of the class in the cell.
* **The box decides which part matters.** [0, E_s] bounds the inter-block
  range of short storage, and the power limit bounds the amplitudes.

## 6. Moves at the fine level (multigrid sampler)

| family | direction | compensation |
|---|---|---|
| atoms, unit, pair, cellpair | sparse basis vectors, units, share swaps | balance-neutral by construction (pairs) |
| `tmode` | transport mode k at τ, projected onto the face of B_τ | exact T2 (§1) at every node |
| `tline` | normal of a predicted binding line (injection pair at its ends) | exact T2 at both ends |
| `tmode0`, `tline0` | the same directions | none (residuals absorb) |
| `tpot` | angle of one node | exact T2 at the node and its neighbours |
| `tsoc` | SoC of one unit at one instant | exact T2 at τ−1 and τ |
| `splateau` | boundary SoC of a dyadic block, for one unit or a cell-class aggregate | pro rata at the two end steps |
| `sshape` | shape mode φ_{L,k} inside a block, for one unit or a cell-class aggregate | pro rata at every step of the block |

Compensated moves carry the Jacobian of §1.

## 7. Coarse model (levels ℓ < finest)

* **Space and time:** cells (countries) by blocks of r steps.
* **Coordinates:** the K transport modes; cell-constant dispatch; SoC
  cell-constant in utilization (the aggregate, with weights E_s), decimated at
  block boundaries.
* **Balance** on the cells.
* **Restriction and prolongation:** the nested bases and the existing time
  transfers.
* **Short storage at coarse time levels.** Its value lies inside blocks. The
  intra-block shapes enter the fine level as moves (§6), and the coarse model
  carries the block energy balance only.
* **Uses:** initialization (coarse equilibration, then prolongation), MLDA
  proposals, MLMC level differences.

## 8. Precomputed vs run time

* **Offline:** k̂, the proxy QPs, the transport modes, the angle basis and the
  storage shapes.
* **Run time:** the binding sets and projected modes per step, recomputed when
  x or k change.
* **Exactness:** nothing depends on v, so every move is exact for the fine
  target, whatever the quality of the heuristics. The heuristics only affect
  efficiency.

## 9. Reactant.jl / GPU layout

Written array-first from the start:
* **Congestion projection** (`CongestionADMM`, `project_congestion`):
  - batched over all steps as matrix columns;
  - one (N + n_links)² inverse precomputed;
  - every iteration is two matmuls, a clamp and an add;
  - static shapes and no solver library, so it can be traced as is.
* **Frames** (`project_modes`): masks (lines × T) instead of index lists. The
  projection is a masked small SPD solve per step, written as a dense batched
  operation.
* **Mode and coefficient tables:** dense N × K × T and m_θ × K × T arrays.
* **Proxy ensemble, PCA and storage shapes:** offline, dense linear algebra on
  the host.

**CPU reference to be batched:** the local exact moves. The traced form applies
one move family to every step of a colour class at once.
* Padded direction arrays: a fixed maximum number of entries per move, i.e.
  pro rata at ≤ deg + 1 nodes × the maximum units per node.
* Breakpoints padded to a fixed maximum, then sorted.
* Segment masses with a log-sum-exp scan, inverse sampling and the Jacobian
  acceptance, all vectorized.
* Data layout: padded per-field node arrays (fields × max support × T),
  unit→node and line→node index arrays, cells as label arrays with segment sums
  (scatter-add).
* Counter-based random numbers per (sweep, step, move).

The remaining Dicts in move preparation (`prep_tpot!`, `prep_storage_ms!`) are
CPU conveniences and get replaced by the padded index arrays.

## 10. Code

`src/flowbasis.jl` (§2, §5 shapes, §3 frames), `src/flowmoves.jl` (§6), option
`basis = "flow"` (§2.5), `kernel = "local"` with the flow families,
`configs/flow.toml`. Tests: `test/test_flow.jl`. Results: §10.

## 11. First results (40 nodes, SoC + renewables as load, 240 s per variant)

* **Transport basis:** K = 4 loading modes explain 95.4 % of the proxy loading
  second moment (71 %, 91 %, 94 %, 95 %). The maximum binding frequency of a
  line in the proxy ensemble is 17 %. At 24 h, 219 of 365 steps have
  predicted binding lines.
* **Correctness:**
  - the ADMM projection agrees with a QP solver on > 97 % of the binding sets;
  - transport and storage moves preserve every nodal residual (|Δr| < 10⁻¹¹ MW);
  - cost changes match the full evaluation.
* **24 h, far start:** local sweep 1.77×, flow sweep 1.79× (140 vs 261 sweeps).
  The same equilibrium, no gain at fixed (k, x).
* **6 h, start prolonged from daily (excess 373×):** local sweep 13.3× after
  240 s, flow sweep 70.8×, worse per sweep as well.
  - Pro-rata transport and storage moves hold every residual fixed, so they
    cannot remove the shed that dominates such starts.
  - Relaxation needs residual-changing moves (units, atoms); the flow families
    are for transport mixing near equilibrium and for compensation under weather
    and capacity moves.
  - A schedule that starts with the plain local families and switches on the
    flow families later is the obvious fix.
* **Coarse capacities:** prolonging level-0 capacities spreads a country's
  capacity uniformly over its nodes, which violates node potentials (capex with
  clamp penalty 1.7·10¹² € after prolongation). Coarse capacity coordinates
  should be weighted by the potentials (weighted Haar, cell-constant k/k̄); not
  yet done.
* **Network size:** 40 nodes is too small to judge the transport hierarchy (4
  modes, few binding lines); the next test needs a few hundred nodes.

### 11a. Exact compensation, uncompensated families alongside (6 h, far start, 4 threads at nice 19)

| | sweeps in 240 s | excess/pred at 240 s | per sweep |
|---|---|---|---|
| local | 56 | 26.9 | 70.8 at 28, 54.7 at 35 |
| flow (exact T2 + `tmode0`/`tline0`, plain rates kept) | 32 | 51.8 | 51.8 at 32 |

With the exact T2 compensation and the plain families at their usual rates, the
flow sweep is now slightly better **per sweep** than the local sweep (it was 2×
worse with the pro-rata-everywhere rule). It is still worse per second, because
a flow sweep costs 1.8× a local sweep (storage multiscale moves and the dense
transport directions). At 40 nodes the transport hierarchy has little to win;
the test that matters is the 256-node network.

### 11b. 256 nodes (PyPSA-Eur, 10 countries, daily, no reference LP)

Built from PyPSA-Eur (OSM base network clustered to 256 nodes,
`external/pypsa-eur/config/config.esm.yaml`; electricity only; NASA POWER weather
2001–2022 per node). The model has 256 nodes, 480 lines and 4 DC links within the
region, and a mean load of 176 GW. Heuristic start (`init = "heuristic"`): proxy
renewables, gas at the nodal peak load, interior dispatch. Sparse Haar angles with
the transport modes as moves (`flow_theta = "haar"`). 4 threads at nice 19, about
400 s per variant:

| | sweeps | s/sweep | operating cost after 12 sweeps | at the end |
|---|---|---|---|---|
| local | 30 | 13.5 | ≈ 5.9·10¹⁰ (11 sweeps) | 4.44·10¹⁰ (30 sweeps) |
| flow | 12 | 36 | 4.62·10¹⁰ | 4.62·10¹⁰ (12 sweeps) |

* **Per sweep, the flow sweep is about 2× as effective at 256 nodes** (12 flow
  sweeps ≈ 25 local sweeps), against about 1.1× at 40 nodes. The transport
  hierarchy matters more on the larger network.
* **Per second the two are about even**, because a flow sweep costs 2.7× a local
  sweep. The cost is in the dense transport directions: every `tmode`/`tline`
  move touches all nodes and lines plus the compensation at every node.
  Localizing the modes (thresholding, or graph-Haar-compressed modes) is the
  obvious optimization.

## 14. Recursive decomposed LP start (shelved)

`init = "decomposed"` (`src/decompose.jl`):
* the top LP over countries (transport model);
* per parent cell, an LP over its children with capacity totals and assigned
  border flows fixed, KVL at node level;
* operation LPs per block (one block-diagonal LP over all regions);
* global angles from the injections.

At 40 nodes × daily it takes 113 s:

| LP | time |
|---|---|
| top, 10 cells × 365 | 24 s |
| Germany, 14 nodes × 365 | 32 s |
| France, 13 nodes × 365 | 46 s (interior point failed, simplex retry) |

That is more than the full LP (≈ 60–80 s): HiGHS's interior point handles these
badly scaled year-long LPs poorly however they are split. The start is also poor
(total cost 1.99·10¹¹ vs 9.4·10¹⁰ €), because the decoupled transfers violate line
limits once the flows are recomputed globally. Shelved in favour of the heuristic
start (§11b). If far starts become the bottleneck at 256+ nodes, the cheap variant
to try is the top LP at a 3–7 day step, a heuristic split of the country capacities
by potential × capacity factor, and parallel weekly operation LPs.

## 15. Learned linear compensation (6 h, far start, 40 nodes, 4 threads at nice 19, ≈ 300 s)

| variant | sweeps | excess/pred at ≈ 33 sweeps | at the end |
|---|---|---|---|
| local (earlier run) | 69 | ≈ 61 | 17.5 |
| flow, exact T2 (earlier run) | 42 | ≈ 51 | 34.2 |
| adaptive, T2 compensation only | 42 | 37.8 | 22.3 |
| adaptive + learned compensation | 43 | 21.4 | **11.2** |

The learned response coefficients (per node and proxy class, regression of each
generator's output on the node requirement) halve the remaining excess at equal
time. With them the adaptive solver is the fastest variant per second as well as
per sweep. The coefficients pick up merit-order behaviour: at the node with the
most units, one unit takes 95 % of requirement changes in deficit hours and gas
none. Pro rata cannot represent that.

## 16. Localized transport moves

A transport mode (proxy or learned) is expanded in the sparse graph-Haar angle
basis. Its dominant atoms are kept: those carrying 90 % of ‖c‖², at most 8 per
mode. A localized move (`tcut`, `tcutL`, `acut`, `acutL`) picks one atom with
probability ∝ c_j². A coarse Haar atom is constant inside each of the two cells it
contrasts, so its flows change only on lines leaving those cells, and the
compensation (T2 or learned) is needed only at their boundary nodes. Lines with
zero flow change are dropped from a move before any evaluation. Full modes stay
in the mixture at a lower rate.

40 nodes × daily, 300 moves each:

| move | lines | entries | breakpoints | time per exact draw |
|---|---|---|---|---|
| full mode (`tmode`) | 79 | 108 | 473 | 10.3 ms |
| localized (`tcut`) | 12 | 39 | 136 | 46 µs |

## 17. Seasonal, day-aligned learned shapes

The intra-block shapes are learned per season class (from the block's start step,
which depends on τ only). Blocks are day-aligned half of the time. Storage-shape
moves also use the learned compensation, which makes them dispatch–storage pair
moves.

## 18. α = 1 pieces

* **Monotone-price bias** (`bias_monotone = true`, `monotone_fit` in
  `src/bias.jl`): the ĝ fit is a small QP whose implied price is non-decreasing
  and bounded by [−c_dump, c_shed]. 40 nodes, LP warm-start scores, implied price
  over u ∈ [−2, 2]:
  - unconstrained: −6755 … 1552 €/MWh, non-monotone, −6755 €/MWh at high residual
    load (the tail leak);
  - monotone: −2.2 … 582 €/MWh, non-decreasing.
* **Transported peak-set capacity moves** (`kpeak_sweep!`):
  - k → k + δ for a dispatchable unit together with its dispatch at the node's
    peak steps (top 5 % of the proxy requirement, from x and k_ren);
  - at the peak steps either the residual absorbs or the other units give way with
    the learned coefficients;
  - a group action, so a generalized Gibbs draw with the Jacobian (k/(k + δ))^T,
    capex, the bias and the reference measure in the target.

  40 nodes: 40 units moved per sweep, gas capacity moving by 1–3 GW per sweep
  (≈ 0.1 s per sweep without the bias, ≈ 1.8 s with the bias evaluation), caches
  consistent. Part of the fine-level kernel after the scaling moves.
