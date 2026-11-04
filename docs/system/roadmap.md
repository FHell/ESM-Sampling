# Roadmap and open problems

This document lists the open work: missing pieces, known deviations of the code from the
intended system, and planned experiments, grouped by area. It is the place where the roadmap
lives; each item links to the document that describes the affected part. Items are removed
when done, and the affected document is updated in the same change.

## 1. Correctness of the ensemble

- **Bias stabilization at $\alpha = 1$.** A long enough level-0 warmup before fine-level
  refits (`complete40.toml` learns nothing at level 0: `warmup0` < `bias_start`), dropping the
  cold-start observations from the sums, damping or a trust region for refits ($\hat b$ is
  linear in $(\theta, a, c)$, so a step toward the new fit can be scaled in closed form to
  bound its change at recent states, keeping the monotone constraints), and a form of $\hat h$
  that does not turn over inside the capacity box (no interior maximum of the quadratic, or a
  saturating form); the linear coefficients of $u_1$ and $u_2$ need a constraint (for instance
  that firm capacity has no negative value in surplus, $\theta_{u_2}\le0$) because the weather
  scores do not separate them. A refit currently moves $\hat b$ by $10^7$–$10^8$ nats at the visited states of
  a cold start, and storage runs to its bound after it ([sampler.md](sampler.md) §7.1).
- **$\alpha = 1$ end to end.** The weather marginal against exact draws, before and after the
  post-correction.
- **Exactness test sensitivity.** The capacity part of `test/test_exact.jl` passes ($|z|\le 2.1$)
  but did not detect the `kpeak` slice-width violation (fixed 2026-11-04); longer runs and
  reference states equilibrated in the storage and dump directions would make it sensitive to
  such errors ([testing.md](testing.md)).

## 2. Validation

- **Convergence at scale.** A first screening at 40 nodes daily ($\alpha = 0$, 4 chains × 100
  sweeps) does not converge (max $\hat R$ 4.6, bulk ESS 4–18); the slowest quantities are the
  capacities, the annual weather means and the regimes. Runs long enough for the criterion of
  [validation.md](validation.md) §1 are needed, on the device for 256 nodes.
- **Freeze adaptation before the measurement window** in the drivers (the adaptive
  modes keep updating), and add the weight-ESS condition to the verdict.
- **Comparison with PyPSA-Eur** ([validation.md](validation.md) §2): the time-series import mode
  of the extractor, PyPSA-Eur solves on the imported networks, the low-temperature limit,
  single-year and stacked multi-year optima, MGA envelopes, the resolution ladder.
- **Weather model validation** on held-out years ([weather_model.md](weather_model.md) §7).

## 3. Model and formulation

- **Reference measure:** decide utilization against physical units per technology class
  ([ensemble.md](ensemble.md) §5).
- **Investable line ratings and storage energy-to-power ratios** (currently fixed; they shape
  the feasible operation).
- **Interannual variability** in the weather model (a per-year latent factor).
- **`co2_cap`** in the device driver.
- **Temperature scaling with resolution:** a fixed $\beta_v$ makes the operational slack per
  unit of cost grow with the number of steps; decide whether $\beta_v$ should scale with the
  number of confined coordinates.

## 4. Sampler efficiency (CPU)

- **Recorded-term evaluation for the model-completion terms** (losses, soft limits, lossy
  links, vintages, commitment, weather entries); with all terms a fine step at 40 nodes × 6 h
  costs about 2.3 times the plain one.
- **Learned dispatch shapes with learned compensation** for faster intraday relaxation from
  far starts.
- **`kop` on by default** once a stable $\alpha = 1$ run with it exists: `kop = false` for now,
  because a wide capacity move follows any non-confining direction of the bias's capacity
  part at once ([moves/capacity.md](moves/capacity.md)).
- **Non-centered regime moves at the fine level** ([moves/weather.md](moves/weather.md)).
- **Parallel tempering:** chains at a ladder of $(\beta_v,\beta_k)$, possibly tempering only
  $\beta_v$ or the bias strength, with swaps.
- **Kept for later, with the condition under which they come back:** per-family diagnostics
  (mean move length against the direction's scale, Jacobian acceptance, time share) if
  convergence at scale is not good enough or one family dominates the profile; learned
  nonlinear compensation if those diagnostics show blocked families; delayed acceptance with
  surrogates for expensive moves (capacity moves at $\alpha = 1$, long storage blocks at hourly
  resolution, fine weather moves) as a cost reduction; learned conditional prolongation for
  starts.

## 5. GPU port

The remaining parts of [device.md](device.md) §7: scheduling the compensated families and the
cell templates in the driver; the `kop` capacity move (a Metropolis form with per-carrier
proposal scales in the capacity pass); templates for Haar angle atoms, localized modes and local
re-routing; block profiles; host-side adaptation with table uploads; weather moves; then
performance (fused kernels, FP32 intermediates, tiling, multi-GPU) and the full-Europe memory
plan. Also: the array-first offline parts (ADMM, frame projection, principal components) on
the device, and a shared random move order per parity class with node colouring within a
step.

## 6. Data and scale

- ERA5 weather for full Europe; larger PyPSA-Eur builds (512, 1024 nodes, full); reference
  LPs on the HPC.
- The first full-resolution run, 256 nodes × hourly with ERA5 weather and all model terms
  (`configs/full256_era5.toml`), is prepared and not run: estimated 15–25 minutes per flow
  sweep on 4 threads and 3–8 GB of memory.

## 7. Experiments

After the items above: sweeps to equilibrium against network size (40, 256, then 512 and
1024 nodes); capacity mixing at $\alpha = 1$ with peak-set moves in a full run; wall time to
stationarity against the finest step (24, 12, 6, 3, 1 h); a memory estimate for full Europe
hourly; final profiling once the known bottlenecks are fixed.
