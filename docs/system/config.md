# Configuration reference

This document lists every key of `Config` (`src/run.jl`) with its default and meaning, and
states which values define the system as it is run. A configuration file is TOML with a
subset of these keys (`load_config`; unknown keys are an error). Examples are in `configs/`.
The model features behind the keys are explained in [model.md](model.md), the sampler options
in [sampler.md](sampler.md).

## 1. System settings

Several defaults are legacy values kept for old configurations. The system as documented
uses:

| key | system value | default |
|---|---|---|
| `basis` | `"haar"` or `"flow"` | `"laplace"` |
| `kernel` | `"local"` | `"gibbs"` |
| `storage` | `"soc"` | `"lp"` |
| `renewables` | `"load"` | `"dispatch"` |
| `regularized` | `true` | `true` |
| `p_fine` | `1.0` (implied in the multi-chain and device drivers) | `0.5` |
| `init` | `"heuristic"` (large networks) or `"lp"` (small) | `"lp"` |
| `measure` | open design choice; the configurations use `"physical"` | `"utilization"` |

> **Status: deviation.** The defaults of `Config` do not describe the system; every system
> configuration sets the keys above explicitly. See `configs/complete40.toml` (all model
> terms, adaptive kernel), `configs/chains40.toml` (multi-chain screening) and
> `configs/device40.toml` (device run).

## 2. All keys

**Files and run.**

| key | default | meaning |
|---|---|---|
| `data` | `"data/processed/model.h5"` | model input file ([data.md](data.md)) |
| `out` | `"results/run.h5"` | output file; caches go to `<dirname(out)>/cache/` |
| `seed` | 1 | master seed (chains use derived seeds) |
| `verbose` | `true` | progress messages |

**Hierarchy and bases.**

| key | default | meaning |
|---|---|---|
| `levels` | `[(4,24), (12,24), (40,24), (40,6)]` | levels $(M_\ell, h_\ell)$, coarse to fine ([model.md](model.md) §2.3) |
| `basis` | `"laplace"` | `"haar"`, `"flow"` (Haar plus transport modes) or `"laplace"` (legacy) |
| `flow_theta` | `"nested"` | `basis = "flow"`: modes as the coarse angle coordinates (`"nested"`) or only as moves (`"haar"`, large networks) |

**Model.**

| key | default | meaning |
|---|---|---|
| `c_shed`, `c_dump` | 3000, 5 | balance prices (€/MWh) |
| `c_viol` | 3000 | price of constraint violations (€/MWh per unit of excess) |
| `regularized` | `true` | price all constraints instead of hard walls |
| `measure` | `"utilization"` | reference measure of dispatchable units: `"utilization"` or `"physical"` |
| `storage` | `"lp"` | `"soc"` (state-of-charge trajectory) or `"lp"` |
| `renewables` | `"dispatch"` | `"load"` (output $Ak$, curtailment as dump) or `"dispatch"` |
| `co2_price` | NaN | canonical CO₂ price (€/t); NaN uses the price contained in the data file |
| `co2_cap` | 0 | > 0: tune the CO₂ price during the finest-level warmup to this mean annual emission (t/a); single-chain driver only |
| `brownfield` | `false` | existing capacities as floors, capex on $k-k_0$, merged vintages, fixed thermal fleets |
| `losses`, `loss_segments` | `false`, 4 | quadratic line losses; interpolation cells per rating |
| `thermal`, `thermal_band` | `"hard"`, 0.1 | `"soft"` thermal limits; band width as a fraction of the rating |
| `link_losses` | `false` | lossy DC links |
| `uc` | `[]` | carriers with fleet unit commitment |

**Weather model.**

| key | default | meaning |
|---|---|---|
| `K` | 4 | number of weather regimes |

**Temperatures.**

| key | default | meaning |
|---|---|---|
| `beta_v`, `beta_k` | 4e-5, 4e-8 | operational and investment temperatures (1/€) |
| `calibrate_beta` | `false` | derive β from the reference LP by equipartition instead |
| `eps_v`, `eps_k` | 0.01, 0.05 | target slacks of the calibration (fractions of the optimal cost) |

**Start.**

| key | default | meaning |
|---|---|---|
| `init` | `"lp"` | `"lp"` (reference LP at the first complete level) or `"heuristic"` (no LP); other values are legacy |
| `descent_iters` | 4000 | reflective descent of the level-0 operation before the level-0 warmup (0: off) |

**Fine-level kernel.**

| key | default | meaning |
|---|---|---|
| `kernel` | `"gibbs"` | `"local"` (the system); other values select legacy kernels |
| `fine_kernel` | `"auto"` | `"local"`, `"flow"`, `"adaptive"`; `"auto"` is `"flow"` with a flow basis, else `"local"` |
| `local_extra` | 4.0 | random moves per node and step and sweep, on top of the atoms |
| `cell_level_power` | 1.0 | cell moves: partition level drawn $\propto$ (number of cells)$^p$ |
| `xloc_partner_hops` | 1 | localized pairs: partner within this many hops |
| `xloc_ball_hops` | 0 | localized pairs: re-routing ball radius (0: the pair only) |
| `fine_weather` | 0.0 | fraction of the weather coordinates moved per fine step (0: weather fixed at the fine level) |
| `p_fine` | 0.5 | probability of a fine-kernel step; the rest go to a legacy kernel ([architecture.md](architecture.md) §9); system: 1 |

**Bias.**

| key | default | meaning |
|---|---|---|
| `use_bias` | `true` | $\alpha = 1$ (learned bias) or $\alpha = 0$ |
| `bias_samples` | 16 | exact weather draws for $\bar{\mathbb E}\hat g$ |
| `bias_monotone` | `true` | monotone, bounded implied price in the $\hat g$ fit |
| `bias_warmstart` | 20.0 | weight (in samples) of the LP-score warm start; `init = "lp"` only; 0: off |
| `bias_start`, `bias_every` | 150, 25 | level-0 warmup: first learning iteration, refit interval |
| `bias_online` | `true` | online refits from the finest chain's scores |
| `bias_online_every` | 1 | observe every n-th finest step |
| `bias_freeze` | `typemax` | single-chain driver: no refits after this many observations (multi-chain and device: end of warmup) |
| `bias_post` | `true` | post-correction by reweighting at $\alpha = 1$ |

**Run lengths.**

| key | default | meaning |
|---|---|---|
| `warmup0` | 1500 | level-0 warmup iterations (bias learning) |
| `n0` | 1500 | single-chain driver: level-0 production iterations |
| `mlda_warmup` | 100 | finest-level warmup sweeps (all drivers; device: checkpoints) |
| `nsamples` | 300 | finest-level production sweeps (device: checkpoints) |
| `alpha0` | `true` | single-chain driver: also run an $\alpha = 0$ level-0 chain |

**Multi-chain mode and convergence.**

| key | default | meaning |
|---|---|---|
| `nchains` | 0 | > 0: run this many finest-level chains (`run_chains`; device: chains) |
| `chain_starts` | `["low", "high", "heuristic", "heuristic"]` | starts, cycled: capacities ×0.5, ×2, as set up |
| `rhat_max` | 1.01 | convergence: rank-normalized split-$\hat R$ below this |
| `ess_min` | 400 | bulk and tail ESS at least this |
| `mcse_max` | 0.05 | MCSE of the mean / posterior sd at most this |
| `rhat_screen` | 1.05 | threshold of the sliding-window warmup detection |

**Device.**

| key | default | meaning |
|---|---|---|
| `device_backend` | `"gpu"` | Reactant backend, `"gpu"` or `"cpu"` |
| `device_sweeps` | 10 | operation sweeps per device call (one capacity pass and one checkpoint after each) |
| `kmove_nprop` | 2 | Metropolis proposals per unit and capacity pass |
| `kmove_sigma` | 0.3 | scaling-move proposal scale (log capacity) |
| `kpeak_sigma` | 0.05 | peak-set proposal scale, fraction of the unit's start capacity (at least 100 MW) |

**Legacy keys** of code outside the system: `J`, `frac`, `ϱ`, `nleap`, `mass_windows`,
`p_ffbs`, `p_scale`, `decomp_cap`, `decomp_block_days`; see [architecture.md](architecture.md) §9.
