# Validation plan: convergence criteria and comparison with PyPSA-Eur

Two open questions:
1. When has a run converged, without a reference LP? (todo item 2)
2. Do the sampled ensembles agree with what PyPSA-Eur's own optimization gives,
   in particular once the weather is averaged?

Part B is the plan for the second question.

## A. Convergence criteria (LP-free)

### A.1 Multiple chains from dispersed starts

A single chain cannot show convergence. Every validation run uses K ≥ 4
independent chains of the finest-level kernel. The starts are deliberately spread
wider than the target:

| start | capacities | operation | weather |
|---|---|---|---|
| low | heuristic × 0.5 | interior centre | exact draw, seed 1 |
| high | heuristic × 2 (clamped to the boxes) | interior centre | exact draw, seed 2 |
| heuristic | heuristic | interior centre | exact draw, seed 3 |
| LP (small models only) | LP optimum | LP dispatch | exact draw, seed 4 |

Different seeds also give different RNG streams. On the CPU the chains run one
after another, or one per thread group. On the GPU they are the chain dimension of
the device kernel, so the GPU port makes K = 64–512 cheap.

### A.2 Monitored quantities

* **System:** total, capex, opex, shed (MWh), emissions (t), penalty.
* **Capacities:** total per carrier; per country for gas, onwind, solar, battery and H₂.
* **Operation:** mean line loading (|f|/s); storage throughput per carrier;
  mean SoC of H₂ (the slowest storage).
* **Prices:** sampled prices per country and season, mean and load-weighted (item 19
  output).
* **Weather (with fine weather moves):**
  - annual means of the level-0 mode 1 per field;
  - regime frequencies.

  Their reference is known exactly: draws from the weather model (at α = 1, after
  the bias post-correction).

### A.3 Statistics

Compute these per quantity, over the K chains after the warmup, with the
diagnostics of Vehtari et al. (2021):
* rank-normalized split-R̂ (the existing `split_rhat` is the plain split-R̂ of a single chain);
* bulk ESS;
* tail ESS (indicators of the 5 % and 95 % quantiles);
* the MCSE of the mean relative to the posterior standard deviation.

**Criterion:**
* **Converged:** max R̂ < 1.01, bulk and tail ESS ≥ 400 in total, and MCSE/sd ≤ 0.05
  for every monitored quantity.
* **Screening thresholds (short runs):** R̂ < 1.05 and ESS ≥ 100.

**Warmup and adaptation.** The warmup ends at the first sweep from which R̂ over a
sliding window (the second half of the sweeps so far) stays below the threshold. At
α = 1 the online bias and the adaptive modes must be frozen before the measurement
window starts: diminishing adaptation guarantees the limit, not a fixed target
during adaptation. The bias post-correction weights (item 4) enter the weighted
estimates. Their ESS is reported next to the chain ESS, and a weight ESS below about
30 % of the samples counts as not converged.

**Sweeps to equilibrium (todo item 9)** is then measured as the first sweep at
which the sliding-window R̂ < 1.05, per network size.

**A cheap LP-free stationarity check for the operational block.** The excess cost
over the running minimum, against its equipartition prediction
(n_confined / 2β_v), the `excess/pred` used in the earlier experiments. It is
necessary for convergence but not sufficient.

### A.4 Implementation

* **CPU:** `src/convergence.jl` with the rank-normalized R̂, bulk and tail ESS and
  MCSE. A driver mode `nchains = K` with `starts = [...]` writes the per-chain traces
  (`chains/<k>/Q`) and a `convergence` group (R̂, ESS, MCSE per quantity, warmup
  sweep, verdict).
* **GPU:** the same statistics on the chain dimension of the device traces (host
  side, from the downloaded per-sweep quantities).
* **Cost:**
  - at 40 nodes × 6 h, 4 chains × 500 fine steps take 2–5 h on 8 CPU threads;
  - at 256 nodes the CPU is only enough for screening; the decisive runs belong on
    the GPU.

## B. Comparison with PyPSA-Eur

### B.0 What corresponds to what

PyPSA-Eur solves an LP: the cost optimum at zero temperature. Our runs are
near-optimal ensembles at temperatures (β_k, β_v). The objects that should agree:

| ours | PyPSA counterpart | relation |
|---|---|---|
| (k, v) at a fixed weather year x, β → ∞ | single-year LP for that year | the sampled means converge to the LP optimum, with differences O(1/β) |
| α = 0 (ρ₀: k adapts to each x), β → ∞ | distribution of the single-year LP optima k*(year) over the historical years | the distribution of k over the weather against the distribution over the years |
| α = 1 (ρ₁: one k against the weather distribution, perfect-foresight dispatch per year), β → ∞ | multi-year stacked LP: capacities shared, dispatch per year, equal year weights | the stochastic program min_k capex(k) + E_x min_v C_x(k, v), with the empirical distribution of the historical years in place of the weather model |
| spread of the ensemble at finite β | MGA envelopes (`optimize_mga`) at a cost slack ε | at matched mean excess cost (ε ≈ E[excess]/C*), the sampled capacity ranges should lie inside the MGA envelope |

The weather enters in two ways that must be kept apart:
* our runs draw weather from the fitted multilevel model;
* PyPSA uses the historical years.

B.3 separates the weather-model error from everything else.

### B.1 Same network, PyPSA as the source: validates the import and the formulation

The data flows one way, from PyPSA-Eur to our model, as in the pipeline.
* **PyPSA-Eur solves its own network**, configured to our scope:
  - electricity only, the same cluster count, no line expansion;
  - the CO₂ treatment matched: an emission price, or the cap with our `co2_cap`.
* **New `prepare_data.py --pypsa-timeseries` mode:** the extractor also takes that
  network's time series for its snapshot year (the generators' `p_max_pu`, the
  loads), instead of the NASA POWER weather.
* **On the imported data** we solve our Julia LP and run our sampler cold (B.2).

**Checks:**
1. Our LP optimum against PyPSA's objective, capacities per carrier and country,
   dispatch, and duals against prices.
2. Every difference must be explained by an aggregation of the import: one unit per
   node and carrier, fixed energy-to-power ratios, the merged vintages. Each of these
   is switched off where possible (e.g. keep PyPSA's storage ratios free on the PyPSA
   side), so that its size is measured.

This checks the extractor, the LP and the cost bookkeeping against an independent
solve without a second translation layer.

### B.2 Low-temperature limit: validates the sampler's landscape

At a fixed weather year (x fixed), run our sampler at β, 10β and 100β. The sampled
mean capacities and costs should approach the PyPSA single-year optimum, with
differences shrinking like 1/β. The sampled nodal prices should approach the LP
duals; the driver already writes `prices/lp`. This connects our regularized,
penalized target to PyPSA's hard-constrained LP.

* **Options:** losses and soft limits can be switched on in both models (PyPSA's
  `transmission_losses` is a piecewise-linear loss model, close to our interpolant).
* **Size:** 40 nodes × 6 h locally.

### B.3 Averaging over the weather

1. **PyPSA single-year optima**, one composed PyPSA-Eur network per weather year (ERA5
   cutouts, built on the HPC), each imported as in B.1: the distribution of k*(year),
   of the system cost and of the prices. Locally, until the cutouts exist: the one
   available year (2013) plus our POWER years for our side only.
2. **PyPSA stacked optimum** over 5, 10 and more years, at a rung of the resolution
   ladder where PyPSA solves (B.4). This shows how fast the stochastic-program
   solution settles with the number of years.
   - *40 nodes:* daily or 6-hourly is local work (22 × 1 460 snapshots).
   - *256 nodes:* 6-hourly on the HPC.
3. **Ours, α = 0, cold:** the capacity distribution over the weather against the
   distribution from step 1.
4. **Ours, α = 1, cold** (after the bias stabilization of todo item 33 and the
   post-correction):
   - mean k against the stacked optimum of step 2;
   - the opex distribution over the weather against the per-year opex of the stacked
     solution;
   - the regional and seasonal prices (item 19 outputs) against the stacked LP's
     duals averaged over the years.
5. **Weather model against history**, to attribute the differences:
   - evaluate the stacked solution k* on weather drawn from our model (dispatch
     LPs at fixed k*, or our sampler at fixed k*, cold), and on the historical years;
   - compare the annual cost and shortage distributions.

   If they agree, differences in steps 3–4 come from the sampler and the
   temperature; otherwise from the weather model.
6. **Finite temperature:** MGA envelopes at ε = 0.5, 1, 2 and 5 % around the stacked
   optimum, min and max per carrier and country, against our α = 1 ensemble at the
   matched mean excess.

### B.4 Resolution ladder (HPC, once ERA5 is available)

PyPSA-Eur does not converge at full resolution even on the HPC, so the comparison
runs on a ladder of resolutions that PyPSA can solve:
* **Spatial:** e.g. 37, 64, 128 clusters.
* **Temporal:** PyPSA-Eur's own aggregation (3 h, or segmentation).
* **Years:** single years, and stacked multi-year where feasible.

On every rung both sides run on the same network (imported as in B.1), and the B.5
metrics are compared. Our full-resolution run has no PyPSA counterpart. It is checked
for consistency with the trends along the ladder (capacities, costs and prices
against resolution), and that is the case the sampler is meant for.

**Settings to match on every rung:**
* the CO₂ treatment: PyPSA-Eur's cap of 5 % of 1990 against our canonical price (use
  `co2_cap` on our side, or an emission price in PyPSA-Eur);
* line expansion off (the `lv` setting);
* the brownfield fleets;
* load shedding (PyPSA-Eur's option against our c_shed).

**Our weather model** is fitted to the imported years, so both sides use the same
weather source. The POWER-based files stay for development.

### B.5 Metrics

Report each per country and season where meaningful:
* **Costs:** total system cost, with the capex and opex split.
* **Capacities:** per carrier and country, including storage power and energy.
* **Emissions** and the realized CO₂ price.
* **Prices:** mean and load-weighted marginal prices; price duration curves.
* **Shortage and curtailment.**
* **Network:** line loading distribution and cross-border flows.
* **Storage:** seasonal H₂ storage behaviour (SoC profile).

Uncertainties on our side are the MCSE from Part A, plus the ensemble spread. On the
PyPSA side the year-to-year spread, and the MGA envelope for near-optimality.

## C. Order and effort

| step | where | needs | effort |
|---|---|---|---|
| A.4 convergence diagnostics + `nchains` driver | local | — | 2–3 days |
| B.1 PyPSA-Eur solve of its own network (2013, a small cluster count) + import of its time series + our LP and cold sampler | local | the PyPSA-Eur workflow in `external/pypsa-eur` (built for 256 clusters; a smaller cluster count needs a rerun of the clustering rules) | 3–4 days |
| B.2 low-temperature limit (40 nodes × 6 h) | local | A.4 | 1–2 days of runs |
| B.3 steps 1–2 (one year locally; more years need the cutouts) | local, then HPC | B.1 | 1 day locally |
| todo 33 (bias stabilization), then B.3 steps 3–6 at 40 nodes | local or GPU | A.4, B.1 | 1–2 weeks |
| B.3 with many years; B.4 ladder | HPC (ERA5 cutouts, PyPSA-Eur runs) | network access to the HPC | after the data |

B.1 and B.2, and B.3 for the one available year, need nothing beyond this machine.
They validate the import, the model and the sampler before the HPC work, so the ladder
only has to explain resolution and weather-year effects.
