# Validation strategy

This document states how results are validated: when a run counts as converged without a
reference LP, and how the sampled ensembles are compared with PyPSA-Eur's own optimization,
including the averaging over weather. The tests of the code itself are in
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
estimates; their ESS is reported next to the chain ESS, and a weight ESS below about 30 % of
the samples counts as not converged.

> **Status: deviation.** The drivers freeze the bias at the end of the warmup, but
> the adaptive modes (`AdaptiveModes`, `freeze` defaults to never) keep updating on their
> geometric schedule during production, and the convergence verdict does not yet include the
> weight-ESS condition.

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

> **Status: missing.** None of §2 has been carried out. Missing pieces: the PyPSA-Eur
> time-series import mode of the extractor, PyPSA-Eur solves of the imported networks,
> stacked multi-year and MGA runs, ERA5 cutouts for many years (HPC), and the stabilization of
> the $\alpha = 1$ bias. What exists: the convergence diagnostics and the multi-chain CPU driver, the
> LP-dual output for §2.3, and the ERA5 weather import (`prepare_era5.py`).
