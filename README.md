# ESM-Sampling

Near-optimal sampling of energy system decisions under weather uncertainty,
implementing [`near_optimal_sampling_setup.md`](near_optimal_sampling_setup.md)
on a PyPSA-Eur network. The multilevel hierarchy uses **Laplace modes of the
network** in space and block coarsening in time, with an arbitrary number of
levels.

Detailed notes: [docs/mathematical_formulation.md](docs/mathematical_formulation.md) (model, layers, gaps) and [docs/implementation.md](docs/implementation.md) (algorithms, code, measurements, gaps).

## Data

`scripts/download_data.sh` and `scripts/prepare_data.py` (run with `uv`) build
`data/processed/model.h5`:

| source | used for |
|---|---|
| PyPSA example network `carbon_management` (PyPSA-Eur, 91 AC nodes, 2050 net-zero, sector coupled) | electricity layer at node resolution: AC lines (`x_pu_eff`, `s_nom_opt`), DC links, generator/storage costs and potentials, fixed fleet (nuclear, run-of-river, hydro, PHS) |
| NASA POWER hourly point data at each node, 2001–2022 (MERRA-2 50 m wind, CERES irradiance, 2 m temperature) | node-level availability (calibrated to the network's `p_max_pu` over its snapshot window) and temperature for load |
| OPSD `weather_data` + `time_series` (ENTSO-E load 2015–2019) | national temperature → load models, driven by node temperatures and scaled to the network's nodal loads |

Default region: DE, FR, BE, NL, LU, CH, AT, CZ, PL, DK → 40 nodes, 79 lines,
8 DC links, 22 weather years (`--countries` selects another region).

Investable: onwind, offwind, solar, gas (OCGT, CO₂ price 150 €/t), battery
(6 h) and H₂ (168 h) storage power at fixed energy-to-power ratio. Lines and
DC links are fixed at their optimized ratings (§6.2: these would shape Ω(k)).

```bash
uv sync
scripts/download_data.sh
uv run python scripts/prepare_data.py
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
julia --project=. scripts/run.jl configs/quick.toml      # smoke test
julia --project=. scripts/run.jl configs/demo.toml       # four levels
uv run python scripts/plot_results.py results/demo.h5 results/figures
```

## Code map (`src/`)

| file | content | setup § |
|---|---|---|
| `spatial.jl` | Laplace modes of the capacity-weighted AC+DC graph; nested orthonormal bases per field support (sequential Gram–Schmidt); gauge modes of θ removed per AC component | 4.1 |
| `layout.jl` | fields, level layouts `(M_ℓ, h_ℓ)`, splitting `s_{ℓ+1} = (s_ℓ, φ_ℓ)`: block means + Helmert contrasts (step fields), decimation + interpolation residuals (instant fields), appended modes (space) | 3.1, 4.1 |
| `energy.jl` | utilization coordinates, priced balance, cost and analytic gradients w.r.t. v, k and the raw weather | 2 |
| `constraints.jl` | Ω_ℓ: exact node constraints for spatially complete fields, outer coefficient boxes otherwise | 2.3, 4.3 |
| `lp.jl` | capacity-expansion LP for initialization and β calibration | 1.5 |
| `weather.jl` | climatology/anomalies, regime-switching VAR(1) (EM, FFBS), responsibility-weighted detail regressions, exact sampling | 3 |
| `hmc.jl` | reflective HMC: exact event-driven billiard on the polytope, dual averaging, diagonal mass adaptation | 5.1 |
| `bias.jl` | ĝ, ĥ, b̂ and the score-identity regressions | 6 |
| `hierarchy.jl` | level targets π_ℓ, states, increments Δ_ℓ | 4.2, 4.4 |
| `mlda.jl` | MLDA recursion with pCN/RW fine-mode proposals, estimator pairs | 5.2, 5.3 |
| `run.jl` | driver, outputs, diagnostics, weather-cost map | 6.4, 7 |

## Hierarchy

A level is `(M_ℓ, h_ℓ)`: the first `M_ℓ` Laplace modes of the network and a
step of `h_ℓ` hours (any number of levels; `M` non-decreasing, `h_ℓ` divisible
by `h_ℓ₊₁`; one level must contain all modes, from where only time is refined).
Every node field is expanded in the nested basis of its support, so

* restriction in space is a prefix of the mode coefficients, prolongation is
  `U[:, 1:m]`; carrier totals depend only on the first (constant) mode and so
  factor through the coarsest level;
* weather uses the same restriction as the energy model (§3.1): load and
  availability fields are expanded in the bases of their supports;
* θ is expanded without the per-AC-component constants (gauge, §2.2).

Coarse levels are free approximations (§4.2). Where a field is spatially
incomplete, node-level boxes do not project to modes, so:

* hard constraints: an outer box per mode coefficient (zonotope bound of the
  node box; LP bound of the line-limit polytope for θ) — this keeps (A1),
  *fine feasible ⇒ coarse feasible*, exact;
* cost: evaluated on the clamped prolongation; node-level violations
  (utilizations/capacities outside their box, storage creating energy, line
  overloads) are priced at `c_viol` (= value of lost load);
* balance: priced on the projection `U_m U_mᵀ r` of the nodal residual onto
  the level's load modes — the Laplace-mode analogue of aggregating the
  balance over a cluster; exact where the level is complete.

## Design choices beyond the setup notes

* **Temperatures (§1.5, open choice 1).** The LP is solved by interior point
  without crossover, so its solution is in the relative interior of the optimal
  face. `n_v*` is the number of coordinates *pinned* there (the cost-confining
  directions, each carrying 1/β of excess, as in `min c·v` on a box where all d
  coordinates are pinned and `E[excess] = d/β`); unpinned ones span the flat
  optimal face and carry none. This reads §1.5's "not pinned" the other way
  round — please check which is intended. Interior investment coordinates
  count ½ (quadratic confinement).
* **Bias warm start.** Before the online regression, ĝ and ĥ are fitted once to
  the zero-temperature scores of the LP (nodal prices and availability/capacity
  shadow prices), restricted to level-0 coordinates. The online regressions
  use the score identity in the *state* coordinates (z0 and level-0 k modes).
* **Carrier scaling move.** In utilization coordinates investments are pinned
  given dispatch (`p = ā k ω`). Level 0 therefore adds a reversible move that
  scales a carrier's capacity coefficients by c and its utilization
  coefficients by 1/c (nodal dispatch unchanged; Jacobian `c^(m_k − m_ω T)`).
* **Pre-descent.** The level-0 chain starts from a reflective (billiard)
  descent of v at fixed (k, x).
* **Fixed.** Line ratings and storage energy-to-power ratios (they shape Ω(k),
  §6.2); interannual factor of the weather model; F̂_ℓ = 0; ĝ uses system-level
  residual-load features only.
* **Data.** 22 weather years (2001–2022) instead of 60; load is modelled from
  temperature (no load anomalies beyond temperature and calendar).

## Status (proof of concept)

Latest results: [docs/overnight_results.md](docs/overnight_results.md) (NCMC,
toy scaling, balance-preserving and node-level moves in the energy model,
convergence vs time step and network size); toy model and MLDA analysis in
[docs/toy_mlda.md](docs/toy_mlda.md) and [docs/toy_results.md](docs/toy_results.md).

The current notes are [`near_optimal_sampling_setup_v2.md`](near_optimal_sampling_setup_v2.md).
Defaults now follow them where implemented:

* `regularized = true`: all constraints are priced (slope `c_viol`, capacity
  floor `κ0`), so MLDA proposals are never infeasible.
* `kernel = "gibbs"`: level-0 base kernel = random-scan mixture of coordinate
  slice Gibbs over (v, z0) in parallel colour classes of time steps, slice
  updates of the k coordinates, FFBS and the carrier (investment–dispatch)
  scaling move (`src/gibbs.jl`). `kernel = "hmc"` keeps reflective HMC.
* `basis = "haar"`: nested partition network → countries → nodes with local
  size-weighted Haar–Helmert contrasts (v2 §4.1 with ε = 0, without the
  level-0 Laplace modes); `basis = "laplace"` keeps Laplace modes.
* Levels ≥ 1 mix MLDA steps with Gibbs sweeps of the operation at fixed level
  (`p_fine`, v2 §5.2).
* β is fixed in the config (`beta_v`, `beta_k`); `calibrate_beta = true`
  restores the LP equipartition calibration.

`configs/haar_poc.toml` (levels countries/daily → nodes/daily → nodes/6 h,
α = 0) runs end to end in ~7 min on 8 threads. Observed so far:

* coordinate Gibbs equilibrates level 0 far faster than HMC and is the only
  kernel that moves the weather (`scripts/compare_kernels.jl`, α = 0);
* with regularized constraints no proposal is infeasible, but MLDA acceptance
  at levels 1–2 is still 0 in the short run: the finest chain has not
  equilibrated (fine Gibbs sweeps at 8·10⁵ coordinates with level-0 slice
  widths are slow);
* at α = 1 the learned bias leaks in the load tail and drives the weather to
  extremes; ĝ needs the monotone-price form with asymptote c_shed (not yet
  implemented).
