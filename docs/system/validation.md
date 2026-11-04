# Validation strategy

This document states how results are validated: when a run counts as converged without a
reference LP, and how the sampled ensembles are compared with PyPSA-Eur's own optimization,
including the averaging over weather. Since the decisions have no exact target density
([ensemble.md](ensemble.md) §3.1), the reference is LP-based: the optimum (zero-temperature
limits, §2) and the near-optimal alternatives that LP methods generate (MGA envelopes,
§2.1). Statistical tests of the sampler against an exact density are not part of the
validation. The tests of the code itself are in
[testing.md](testing.md); the ensemble and its limits in [ensemble.md](ensemble.md) §9; the
diagnostics' implementation in [sampler.md](sampler.md) §9.

## 1. Convergence without a reference

**Several chains from dispersed starts.** A single chain cannot show convergence. Every
validation run uses $K\ge4$ chains of the finest-level kernel whose starts are spread wider
than the target: capacities at 0.5 and 2 times the heuristic start (clamped to the boxes), the
heuristic start itself, and on small models the LP optimum; the operation at the interior
centre; with fine-level weather moves, a separate exact weather draw per chain. On the CPU
the chains run one after another (`nchains`); on the GPU they are the chain dimension, which
makes 64–512 chains cheap.

**Monitored quantities.** System totals (total cost, capex, opex, shed, emissions, penalty);
capacities per carrier and per country; operation (mean line loading, mean H₂ state of
charge); prices (mean and load-weighted sampled price; per node, country and season in the
CPU driver's outputs); weather (annual means of the level-0 fields, regime frequencies),
whose reference is known exactly from draws of the weather model (at $\alpha = 1$, after the
post-correction).

**Statistics and criterion.** Per quantity, over the chains after the warmup (Vehtari et al.
2021): rank-normalized split-$\hat R$ (bulk and folded), bulk ESS, tail ESS (indicators of the
5 % and 95 % quantiles), and the MCSE of the mean relative to the posterior standard
deviation.

| verdict | $\hat R$ | bulk and tail ESS (total) | MCSE/sd |
|---|---|---|---|
| converged | < 1.01 | ≥ 400 | ≤ 0.05 |
| screening (short runs) | < 1.05 | ≥ 100 | — |

The warmup ends at the first checkpoint from which the $\hat R$ over a sliding window (the
second half of the sweeps so far) stays below the screening threshold. **Sweeps to
equilibrium**, the measure of sampler efficiency against network size, is that first sweep.

**Adaptation and the bias.** Diminishing adaptation guarantees the limit, not a fixed target
during adaptation, so at $\alpha = 1$ the online bias and the adaptive modes must be frozen
before the measurement window starts. The post-correction weights enter the weighted
estimates; their ESS is reported next to the chain ESS, and a weight ESS below 30 % of the
samples counts as not converged (`convergence/post_weight_ess_ok`). The weights compare the
bias in use with a refit of it, not with the exact $b$, so a high weight ESS does not show
that the bias is right. Since the decisions have no exact target density
([ensemble.md](ensemble.md) §3.1), the criteria that matter are that the sampled density
decreases with cost, concentrates near the optimum, agrees between chains from different
starts, and leaves the weather at its prior. The CPU driver freezes the
bias, the canonical CO₂ price and the adaptive modes at the end of the warmup; the device
driver has no adaptive modes and freezes the bias.

**A necessary check for the operation.** The excess operating cost over its running minimum,
against the equipartition prediction $n_{\rm conf}/\beta_v$ (one $1/\beta_v$ per coordinate
confined by a piecewise-linear cost), is cheap and needs no LP. It is necessary for
stationarity, not sufficient.

## 2. Comparison with PyPSA-Eur

### 2.1 Principles

**Data flow one way**, from PyPSA-Eur into our model, through the extractor
(`scripts/prepare_data.py`). PyPSA-Eur never consumes our results; every comparison solves
PyPSA's problem on its side and ours on the imported data.

**What corresponds to what.** PyPSA-Eur solves LPs: zero-temperature optima on historical
years. Our runs are near-optimal ensembles at finite temperature on weather drawn from a
fitted model. The objects that should agree:

| ours | PyPSA-Eur counterpart | relation |
|---|---|---|
| dispatch at fixed $(x,k)$, $\beta_v\to\infty$ | dispatch LP of that year at fixed capacities | sampled means converge to the LP optimum, differences $O(1/\beta_v)$; sampled prices to the duals |
| $(k,v)$ at fixed $x$, single temperature $\beta_k=\beta_v\to\infty$, $\alpha = 0$ | single-year capacity-expansion LP | the same limit; with $\beta_k\ne\beta_v$ the sampled $k$ minimizes $K + (\beta_v/\beta_k)F_x$ instead |
| $k$ over the weather at a single temperature, $\alpha = 0$ | distribution of the single-year optima $k^\ast$(year) over the historical years | distribution against distribution |
| $\alpha = 1$, $\beta_v\to\infty$ then $\beta_k\to\infty$ | multi-year stacked LP: shared capacities, dispatch per year, equal year weights | the two-stage stochastic program, with the historical years in place of the weather model |
| spread of the $\alpha = 1$ ensemble at finite $\beta$ | MGA envelopes (`optimize_mga`) at a cost slack $\varepsilon$ | at matched mean excess cost ($\varepsilon\approx\mathbb E[\text{excess}]/C^\ast$) the sampled capacity ranges lie inside the envelope |

The weather enters in two ways that must be kept apart: our runs draw it from the fitted
model, PyPSA uses the historical years. §2.4 separates the weather-model error from the rest.

### 2.2 Same network, PyPSA-Eur as the source: the import and the formulation

PyPSA-Eur solves its own network configured to our scope (electricity only, the same cluster
count, no line expansion, matched CO₂ treatment: an emission price, or its cap against our
`co2_cap`). The extractor imports that network together with its own time series for the
snapshot year (the generators' `p_max_pu` and the loads) instead of the NASA POWER weather.
On the imported data we solve our LP and run our sampler cold. Our LP optimum is compared with
PyPSA's objective, capacities per carrier and country, dispatch, and duals against prices.
Every difference must be explained by an aggregation of the import (one unit per node and
carrier, fixed energy-to-power ratios, merged vintages); each is switched off on one side
where possible, so that its size is measured.

### 2.3 Low-temperature limit: the sampler's landscape

At a fixed weather year, our sampler at $\beta$, $10\beta$ and $100\beta$ (single temperature
for the capacity comparison, §2.1) should approach PyPSA's single-year optimum with
differences shrinking like $1/\beta$, and the sampled nodal prices should approach the LP
duals (the CPU driver writes `prices/lp` when the reference LP is at the finest
level). Line losses and soft limits can be switched on in both models (PyPSA's
`transmission_losses` is a piecewise-linear loss model, close to our interpolant). This
connects the regularized, priced target to the hard-constrained LP. Size: 40 nodes × 6 h,
locally.

### 2.4 Averaging over the weather

1. PyPSA single-year optima, one composed network per weather year (ERA5 cutouts), each
   imported as in §2.2: the distribution of $k^\ast$(year), system cost and prices.
2. PyPSA stacked optimum over 5, 10 and more years, at a rung of the resolution ladder where
   PyPSA solves: how fast the stochastic-program solution settles with the number of years.
3. Ours at $\alpha = 0$, single temperature, cold: the capacity distribution over the weather
   against step 1.
4. Ours at $\alpha = 1$, cold, after the bias is stabilized and post-corrected: mean $k$
   against the stacked optimum; the opex distribution over the weather against the per-year
   opex of the stacked solution; regional and seasonal prices against the stacked duals
   averaged over the years.
5. Attribution: the stacked solution $k^\ast$ evaluated on weather drawn from our model
   (dispatch LPs or our sampler at fixed $k^\ast$, cold) and on the historical years. If the
   annual cost and shortage distributions agree, differences in steps 3–4 come from the
   sampler and the temperature; otherwise from the weather model.
6. Finite temperature: MGA envelopes at $\varepsilon$ = 0.5, 1, 2 and 5 % around the stacked
   optimum, minimum and maximum per carrier and country, against our $\alpha = 1$ ensemble at
   the matched mean excess.

### 2.5 The resolution ladder

PyPSA-Eur does not converge at full resolution, even on the HPC. The comparison therefore runs
on a ladder of resolutions PyPSA can solve: spatial (e.g. 37, 64, 128 clusters), temporal
(PyPSA-Eur's own aggregation, 3 h or segmentation), and years (single years, and stacked
multi-year where feasible). On every rung both sides use the same imported network and the
metrics of §2.6. Our full-resolution run has no PyPSA counterpart; it is checked for
consistency with the trends along the ladder (capacities, costs and prices against
resolution), and it is the case the sampler is built for.

Settings matched on every rung: the CO₂ treatment (PyPSA-Eur's cap against our price or
`co2_cap`), line expansion off, the brownfield fleets, and load shedding (PyPSA-Eur's option
against our $c_{\rm shed}$). Our weather model is fitted to the imported years, so both sides
use the same weather source.

### 2.6 Metrics

Per country and season where meaningful: total system cost with the capex/opex split;
capacities per carrier and country, including storage power and energy; emissions and the
realized CO₂ price; mean and load-weighted prices and price duration curves; shortage and
curtailment; line loading distributions and cross-border flows; the seasonal H₂ state of
charge. Our uncertainties are the MCSE of §1 plus the ensemble spread; PyPSA's are the
year-to-year spread and the MGA envelope.

### 2.7 The LP baselines as built

**Ours** (`src/lp_baseline.jl`, `scripts/lp_baseline.jl`, campaign H2). The capacity-expansion
LP of `solve_lp` (`src/lp.jl`), built by `lp_build` from one operation block per weather year
(`lp_operation!`) on shared capacities:

- **Weather.** A historical year of the model file (`historical_weather`): the hourly fields
  `weather/<field>/data` averaged over the steps of the level (`aggregate`, as in the
  weather-model fit), then the data map (load ≥ 0, availabilities in [0, 1]). At a spatially
  complete level this is the weather model's training data of that year at that level, to
  round-off (`test/test_lp_baseline.jl`). The LP's level is the configuration's reference
  level, the first spatially complete level of `levels`: a 6 h LP is a configuration with
  `levels = [[10, 24], [40, 6]]` (the coarse level only makes the configuration valid).
- **Stacked optimum** (`solve_lp_years`): capacities shared, dispatch per year, objective
  capex + Σ_y w_y opex_y with equal weights $w_y = 1/Y$ (an annual cost). Storage and fleet
  commitment are cyclic within each year: every year starts and ends at the same state of
  charge, with no carry-over between years, as in a single-year LP and in the sampler's
  annual weather draws; the years are exchangeable scenarios of the two-stage problem, not a
  chronological sequence. The per-year duals are reported per unit weight (the year's own
  prices). One year is the single-year LP; the same year twice gives the same optimum.
- **MGA** (`mga_sweep!`): on the solved model, the bound capex + Σ w opex ≤ (1 + ε) C* is
  added and, for each ε and carrier (k field), the total capacity Σ_i k_f[i] is minimised and
  maximised; the model is reused (only the bound and the objective change). The capacity per
  country is recorded at every extreme. The bound is written in units of
  $S = 10^{\lfloor\log_{10}C^\ast\rfloor-4}$ €/a, so that its right-hand side is 1e4–1e5 like the
  other rows': unscaled (right-hand side 1e11), HiGHS's interior-point solvers HiPO and IPX
  stall on the maximisations at 40 nodes daily ("no progress", an infeasible point). A solve
  that does not end `OPTIMAL` is repeated with IPX without presolve; a primal-feasible point
  within the bound (relative 1e-6) is kept with the status `feasible:<status>` (HiPO often
  ends a capacity objective with `OTHER_ERROR` at such a point: dual residuals after
  postsolve), anything else is recorded as failed.

> **Status: partially verified.** On the 5-node test problem every MGA solve ends optimal. At
> 40 nodes daily (2013, ε = 5 %) with the scaled bound: onwind min `OPTIMAL` (−45 %, 8 min),
> onwind max (+50 %) and gas max (+117 %) feasible points exactly at the bound, HiPO status
> `OTHER_ERROR`, 5–7 min each; the maxima of the other carriers are untested locally (three of
> them together did not finish in 29 min on the loaded laptop). Without the scaling the
> maxima failed. Untried: IPX with crossover, a dual-simplex re-solve warm-started from a
> basis (it needs a crossover of the optimum first), tighter capacity boxes.

**MGA with open solvers: what others report.** PyPSA's `optimize_mga` adds the same kind of
budget, on `n.statistics` capex + opex with the fixed costs on both sides, and recommends
slacks of 2–20 % ([PyPSA docs, MGA](https://docs.pypsa.org/latest/user-guide/optimization/modelling-to-generate-alternatives/));
its `optimize_mga_in_direction` explores the near-optimal space along weighted directions.
Neumann & Brown ([2021, EPSR 190](https://arxiv.org/abs/1910.01891)) bound carrier and region
groups with an ε-constraint method on a PyPSA-Eur model; they
report no open-solver settings. PyPSA's troubleshooting guide notes that HiGHS can struggle on
large problems and recommends the interior-point method without crossover and clipping tiny
`p_max_pu` values ([PyPSA troubleshooting](https://docs.pypsa.org/stable/user-guide/troubleshooting/)).
Bröchin, Pickering, Tröndle & Pfenninger ([2022, arXiv:2211.12299](https://arxiv.org/abs/2211.12299))
name the numerical range, sparsity and degeneracy as what makes energy-system LPs hard, and
improve tractability with an automatic rescaling of the model; barrier without crossover is
fast but changes the structure of the optimal solution. Funplex
([2024, arXiv:2406.19809](https://arxiv.org/abs/2406.19809)) explores near-optimal spaces with
a multi-objective simplex to avoid the independent re-solves of standard MGA. We found no
report of HiGHS-specific MGA failures or fixes (searches of the PyPSA docs and issues, Calliope
SPORES, HiGHS); the budget-row scaling above is our own fix, in line with the rescaling
advice.
- **Not in the LP:** line losses and soft thermal limits (hard limits; a configuration with
  them is solved without them and the summary lists them as ignored). Renewable curtailment is
  free in the LP; the sampler's energy model charges curtailed output as nodal dump at
  $c_{
m dump}$ plus the renewable's marginal cost, so the sampler's cost at the LP solution
  is the LP objective plus `curtail_cost_eur` (about 0.6 % of the cost at 40 nodes daily; the
  test checks the identity to 1e-5).

**PyPSA-Eur's** (`scripts/pypsa_baseline.py`, campaign H2b). PyPSA solves the composed network
of the build (`resources/esm256/networks/composed_2050.nc`, 2013) with HiGHS, configured to our
scope (§2.2): lines and DC links at their current ratings, a load-shedding generator at every
bus at our $c_{
m shed}$, the build's CO₂ cap (or none, or a price), the time series averaged
over blocks of 24, 6, 3 or 1 h (PyPSA-Eur's `average_every_nhours`). MGA as PyPSA's
`optimize_mga` (budget on capex + opex including the fixed cost of the existing fleet), on one
model with the budget's right-hand side and the objective changed per solve; carrier groups
mapped to our k fields (onwind; solar and solar-hsat; the three offwind kinds; CCGT and OCGT
as gas; the battery discharger and the H₂ fuel cell as the storage powers). PyPSA's budget includes the fixed cost of the existing fleet (`total_cost_eur_per_a`), ours does not (greenfield objective), so an ε of H2b corresponds to about ε · total/objective of H2; both numbers are in `summary_pypsa.toml`. The local smoke (14 daily snapshots) ends every MGA solve optimal with the scaled budget.

Outputs: [data.md](data.md) §6. The two campaigns are independent; the comparison of their
results is a separate step.

> **Status: partial.** Built: the historical-year, stacked and MGA LPs of our model (H2) and
> PyPSA's optimum and MGA on its network (H2b), prepared as cluster campaigns
> ([campaigns.md](../campaigns.md)); not yet run on the cluster. Missing: the PyPSA-Eur
> time-series import mode of the extractor (so that our LP runs on PyPSA's own time series
> for §2.2; until then H2 uses the NASA POWER weather of our model files and H2b PyPSA's
> own atlite series (ERA5, SARAH), and the comparison of the two mixes the weather source with the
> formulation), PyPSA solves of several weather years (one composed network per year), the
> stacked PyPSA optimum, ERA5 cutouts for many years (HPC), the comparison step itself, and the
> stabilization of the $\alpha = 1$ bias. What exists besides: the convergence diagnostics and
> the multi-chain CPU driver, the LP-dual output for §2.3, and the ERA5 weather import
> (`prepare_era5.py`).
