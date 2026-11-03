# The energy model as implemented

This document describes the energy system model whose cost enters the ensemble: the state
(fields, spatial bases, levels in space and time, node values and coefficients), every term
of the log-density, the optional model features with their configuration keys, and the data
pipeline from PyPSA-Eur and the weather sources. The ensemble built from this cost is in
[ensemble.md](ensemble.md); all configuration keys are in [config.md](config.md); the file
layouts in [data.md](data.md). Code: `src/system.jl` (data), `src/spatial.jl`,
`src/layout.jl` (bases and levels), `src/energy.jl` (model and cost), `src/flowbasis.jl`
(transport basis).

## 1. Network and units

The model is the electricity layer of a PyPSA-Eur network (`System`, read by `load_system`):

- **nodes** (AC buses, with country and coordinates);
- **AC lines** with normalized susceptance $b_e$, rating $s_e$ and loss coefficient
  $\alpha_e$; parallel lines are merged; line ratings are fixed;
- **DC links** with capacity and efficiency $\eta_l$; fixed;
- **generators**, one unit per (node, carrier): investable renewables (onwind, offwind,
  solar) with potential $\bar k$ and weather availability, investable gas, fixed nuclear and
  run-of-river with constant availability, and optional brownfield fleets (coal, lignite,
  oil, biomass, waste, geothermal);
- **storage**, one unit per (node, carrier): investable battery (6 h) and H₂ (168 h) with a
  fixed energy-to-power ratio, fixed pumped hydro and reservoir hydro (with constant inflow).

A unit's capacity $k$ is its power rating in MW; storage energy is $h\,k$ with $h$ the
energy-to-power ratio in hours. Capex is annualized €/MW/a (for storage including the
energy part).

## 2. State and bases

### 2.1 Fields

Every node-indexed quantity is a **field** (`FieldSpec`): a name, a kind, a spatial basis on
its support, and a node-level box $[\underline y,\bar y]$.

| field | kind | support | box | meaning |
|---|---|---|---|---|
| `k_<carrier>` | capacity | units of an investable carrier | $[k_0, \bar k]$ | capacity in MW ($k_0 = 0$ without brownfield) |
| `theta` | step | all nodes | none | voltage angle |
| `g_<carrier>` | step | units of a dispatchable carrier | $[0,1]$ | utilization $\omega$ |
| `uc_<carrier>` | instant | committable units | $[0,1]$ | committed fraction $c$ |
| `links` | step | links (identity basis) | $[-1,1]$ | signed link utilization $u$ |
| `se_<carrier>` | instant | storage units | $[0,1]$ | state of charge as a fraction of $h k$ |
| `sp_`, `sm_<carrier>` | step | storage units | $[0,1]$ | charge, discharge utilization (`storage = "lp"` only) |
| `load`, `onwind`, `offwind`, `solar` | step | weather supports | none | weather fields ([weather_model.md](weather_model.md)) |

A step field has one value per step; an instant field has one value per instant (the start
of each step), and the year is cyclic. Renewables treated as load (`renewables = "load"`)
have no utilization field.

### 2.2 Spatial bases

With `basis = "haar"` or `"flow"` every field is expanded in a sparse orthonormal **graph
Haar basis** on its support, for the nested partition network → countries → nodes
(`haar_basis`). The first vectors are the normalized indicators of the coarsest cells; then,
level by level, each parent cell gets contrasts between its children arranged as a balanced
binary tree (children ordered along the Fiedler vector of their quotient graph). Each vector
lives in one parent cell, and the first $n_q$ vectors span exactly the cell-constant fields
of partition level $q$, so the orthogonal projection onto a level is the cell mean.

- **Capacities** use a weighted Haar basis with the node potentials $\bar k$ as weights, so
  a coarse capacity coefficient spreads a cell's capacity in proportion to potential.
- **Angles** use the partition AC component → country → node without the component
  indicators, which removes the gauge (a constant angle per AC component).
- **Flow basis** (`basis = "flow"`): additionally, a small set of transport modes is computed
  offline from the weather data and the unextended network (§2.5). With
  `flow_theta = "nested"` the modes are the first angle coordinates and the remaining angle
  vectors are the Haar vectors with the modes projected out (dense); with
  `flow_theta = "haar"` the angles keep the sparse Haar basis and the modes are used only as
  move directions (large networks).
- `basis = "laplace"` (Laplace modes of the network, nested by Gram–Schmidt) is legacy; the
  local kernel needs the Haar basis ([architecture.md](architecture.md) §9).

### 2.3 Levels

A **level** is a pair $(M_\ell, h_\ell)$: the number of global basis vectors kept (for the Haar
basis: $1$, the number of countries, or $N$) and the step length in hours. A field keeps the
basis vectors spanned by the first $M_\ell$ global ones, so coarse coefficients are a prefix of
fine coefficients. Levels must refine ($M$ non-decreasing, $h_\ell$ divisible by $h_{\ell+1}$,
$8760/h$ integer), and from the first level containing all nodes on only time is refined
(`check_levels`). Between levels, step fields are block averaged in time with orthonormal
Helmert contrasts as details, instant fields are decimated with the deviations from linear
interpolation as details (`Transfer`, `split_state`, `merge_state`).

The sampler runs on the **finest level**, which is complete in space. Coarser levels are
used for the level-0 chain that learns the bias, for the reference LP, and for
initialization ([sampler.md](sampler.md)). A typical hierarchy is
`levels = [[10, 24], [40, 24]]` (countries daily, nodes daily) or with a further `[40, 6]`.

### 2.4 Coordinates and node values

The canonical state of a level is the flat coefficient vector of every field
(`State.k`, `State.v`), laid out by `Layout`. Node values are $Y_f = U_f^{(\ell)} Z_f$
(`nodevals`). At the complete level every basis is orthonormal and complete, so node values
and coefficients are equivalent; the local kernel caches the node values (`LocalCost`) and
updates coefficients and node values together. The device port stores node values only.

### 2.5 Transport basis (offline, `basis = "flow"`)

`flow_basis` builds proxy capacities (fixed fleet as is, investable renewables scaled from
their potentials to a 70 % energy share), proxy injections per step for a few historical
years (renewable output minus load, balanced copper-plate), and projects them onto the
unextended network's line limits by a batched ADMM (`CongestionADMM`). The leading principal
components of the projected line loadings (95 % of the loading variance, at most 12) are
mapped back to angle directions and orthonormalized against the component constants. The
same proxy ensemble gives the intra-block storage shapes (`storage_shapes`). Nothing here
depends on the sampled state.

## 3. Cost terms

All terms below are at the finest (complete) level, per step $\tau$ of length $\Delta t$,
and summed over the cyclic year. $\mathrm{cl}(y)$ is $y$ clamped to its box,
$w = \mathrm{cl}(k)+\kappa_0$ the penalty weight of a unit, $c_v = c_{\rm viol}$. The
reference evaluation is `energy_cost`; the local kernel evaluates the same terms
incrementally (`cache_cost` equals `energy_cost` to round-off).

**Capex.** $K(k) = \sum_u c_u k_u + \gamma_k\sum_u c_u|k_u-\mathrm{cl}(k_u)|$, with
`brownfield` minus the sunk $\sum_u c_u k_{0,u}$. It enters as $-\beta_k K$.

**Dispatchable generators.** Output $p = A\,\mathrm{cl}(k)\,\mathrm{cl}(\omega)$, with $A$ the
constant availability (or the weather availability $\bar a$ for a weather-dependent unit
when `renewables = "dispatch"`). Cost $\Delta t\,C_{\rm vint}(p) + c_v\Delta t\,w\,|\omega -
\mathrm{cl}(\omega)|$. Without vintages $C_{\rm vint}(p) = \mathrm{mc}\,p$. With
`brownfield`, a unit holds the existing capacity $k_0$ (old vintage) and $k - k_0$ (new) and
dispatches in merit order:
$C_{\rm vint}(p) = c_{\rm lo}\min(p, Aq_{\rm lo}) + c_{\rm hi}(p - Aq_{\rm lo})_+$ with
$q_{\rm lo}$ the capacity of the cheaper vintage (`gen_vint`). Marginal costs are
$\mathrm{mc} = \mathrm{mc}_{\rm fuel} + \mu\,\mathrm{ef}$ for the canonical CO₂ price $\mu$ and
the emission factor ef of each vintage (`set_co2_price!`). Emissions are reported.

**Renewables as load** (`renewables = "load"`). Output is $\bar a\,\mathrm{cl}(k)$ exactly,
with its (small) running cost; curtailment is the negative nodal residual and is priced as
dump. With `renewables = "dispatch"` they are dispatchable units with a utilization field.

**Fleet unit commitment** (`uc`). For a committable unit, the committed fraction
$c\in[0,1]$ is an instant field. Costs: no-load $\Delta t\,a\,A k\,\mathrm{cl}(c)$, start-up
$s\,Ak\,(\mathrm{cl}(c_{\tau+1}) - \mathrm{cl}(c_\tau))_+$, and the walls
$c_v\Delta t\,w\,[(\mathrm{cl}(\omega)-\mathrm{cl}(c))_+ + (g_{\min}\mathrm{cl}(c)-\mathrm{cl}(\omega))_+ +
|c-\mathrm{cl}(c)|]$. The parameters $(g_{\min}, s, a)$ per carrier are assumptions
(`UC_DEFAULTS`: gas 0.3/40/4, coal 0.4/80/3, lignite 0.5/100/3, nuclear 0.5/150/1, …).

**Storage, state-of-charge form** (`storage = "soc"`). One trajectory $e_\tau\in[0,1]$ per
unit. With $d_\tau = h(\mathrm{cl}(e_{\tau+1})-\mathrm{cl}(e_\tau))/\Delta t - q$ ($q$ the
inflow per unit power), charging is $p^+ = d/\eta^+$ for $d\ge0$ and discharge
$p^- = \min(-d\,\eta^-, 1)$ otherwise (energy beyond the power limit is spilled). The
injection is $k(p^- - p^+)$. Cost: $\Delta t\,\mathrm{mc}\,k(p^++p^-)$ plus the penalties
$c_v w h|e - \mathrm{cl}(e)|$ (energy outside the box), $c_v\Delta t\,w(p^+-1)_+$ (charging
beyond the rating), $c_v\Delta t\,w(-d\eta^- - 1)_+$ for units without inflow (discharge beyond
the rating), and $c_v\Delta t\,w\,d$ for units that cannot charge ($\eta^+ = 0$). Simultaneous
charge and discharge cannot occur.

**Storage, LP form** (`storage = "lp"`, legacy). Separate charge, discharge and SoC fields
with the dynamics inequality priced at $c_v$.

**Links.** Signed flow $u\,\mathrm{cap}$ with $u$ clamped to $[-1,1]$, losses $(1-\eta)|u\,\mathrm{cap}|$
at the receiving end (`link_inj`; $\eta = 1$ unless `link_losses`), penalty
$c_v\Delta t\,\mathrm{cap}\,|u - \mathrm{cl}(u)|$.

**Lines.** Flow $f_e = b_e(\theta_a-\theta_b)$ (Kirchhoff's voltage law by construction).
With $u_e = |f_e|/s_e$: overload cost $c_v\Delta t\,s_e\,\psi(u_e)$ and losses
$\alpha_e s_e^2 L(u_e)$ MW drawn half at each end (`LineTerms`). Hard limits:
$\psi = (u-1)_+$. Soft limits (`thermal = "soft"`): the marginal overload price rises
linearly from 0 at $s$ to $c_v$ at $(1+\delta)s$ and stays there ($\delta$ = `thermal_band`).
Losses (`losses`): $L$ is the piecewise-linear interpolant of $u^2$ on a grid of
`loss_segments` = $K$ cells per rating, exact at the grid points, linear beyond $1.5\,s$, with
interpolation error at most $\alpha s^2/(8K^2)$. All line terms are convex and piecewise
linear in $f$, so their kinks are breakpoints of the exact line sampler rather than a
departure from it.

**Balance.** The nodal residual
$$
r_{n,\tau} = \bar d_{n,\tau} + \sum_{e\ni n}\big(\pm f_e + \tfrac12\text{loss}_e\big) + \text{link injections}
 + \text{storage charging} - \text{generation}
$$
costs $\Delta t\,[c_{\rm shed}(r)_+ + c_{\rm dump}(r)_-]$. At a spatially coarse level the
residual is first projected onto the level (the cell mean for Haar bases), so only
imbalances the level resolves are priced.

**Penalty weights and the capacity floor.** Box penalties are weighted by
$w = \mathrm{cl}(k)+\kappa_0$ with $\kappa_0 = 1$ MW, so a unit with zero capacity still pays
for leaving its box.

**Reference measure.** $+\log J(k)$ with `measure = "physical"`
([ensemble.md](ensemble.md) §5); zero for `"utilization"`.

**Bias.** $-\hat b(z_0, k_0)$ at $\alpha = 1$ (`use_bias = true`)
([ensemble.md](ensemble.md) §4.4).

**Weather prior.** $\log p^w(m,z_0,w)$ ([weather_model.md](weather_model.md) §5).

The total cost is reported as `CostParts`: capex, operating cost, its generator and storage
parts, shed and dump (MWh), the penalty (€) and the emissions (t CO₂).

## 4. Optional features

Every feature is an option of the same model and the same exact kernel. Defaults are those
of `Config` (`src/run.jl`); see [config.md](config.md) for the keys that define the system
as it is run.

| key | default | feature |
|---|---|---|
| `storage` | `"lp"` | `"soc"`: state-of-charge storage (the system form); `"lp"`: charge/discharge/SoC fields |
| `renewables` | `"dispatch"` | `"load"`: renewables as negative load (the system form) |
| `measure` | `"utilization"` | `"physical"`: reference measure in MW for dispatchable units (open choice) |
| `regularized` | `true` | all constraints priced; `false` gives hard walls (legacy) |
| `basis`, `flow_theta` | `"laplace"`, `"nested"` | spatial basis (§2.2) |
| `co2_price` | NaN (the file's price, 150 €/t) | canonical CO₂ price $\mu$ |
| `co2_cap` | 0 (off) | tune $\mu$ during the finest-level warmup so that the mean emissions equal the cap, then freeze |
| `brownfield` | `false` | existing capacities as floors, capex on $k-k_0$, merged vintages, fixed thermal fleets; the upper box is raised to at least $1.1\,k_0$ |
| `losses`, `loss_segments` | `false`, 4 | quadratic line losses (piecewise-linear interpolant) |
| `thermal`, `thermal_band` | `"hard"`, 0.1 | soft thermal limits |
| `link_losses` | `false` | signed lossy DC links |
| `uc` | `[]` | carriers with fleet unit commitment, e.g. `["gas", "nuclear"]` |
| `c_shed`, `c_dump`, `c_viol` | 3000, 5, 3000 €/MWh | balance and violation prices |

Fixed in code (not in `Config`): $\gamma_k = 10$, $\kappa_0 = 1$ MW, the soft-limit
interpolation points (`thermal_segments = 4`), the unit-commitment parameters.

> **Status: deviation.** Line ratings and storage energy-to-power ratios are fixed, not
> investable. The reference LP (`src/lp.jl`) has hard limits and no losses, so with
> `losses` or `thermal = "soft"` it is a start, not the zero-temperature limit of the
> sampled model.

## 5. Data pipeline

The data flow one way, from PyPSA-Eur and the weather sources into one HDF5 file per network
(`data/processed/*.h5`, layout in [data.md](data.md)):

1. `scripts/download_data.sh` fetches the PyPSA example network `carbon_management.nc`,
   the OPSD `weather_data` (national temperatures) and `time_series` (ENTSO-E load
   2015–2019).
2. `scripts/prepare_data.py` (run with `uv`) extracts the electricity layer of a network for
   a set of countries (default DE, FR, BE, NL, LU, CH, AT, CZ, PL, DK): from the example
   network (`model.h5`, 40 nodes) or from a composed PyPSA-Eur network given with
   `--network` (`model256.h5`, the 256-cluster build in `external/pypsa-eur`). It merges
   parallel lines, takes link capacities, one generator and one storage unit per (node,
   carrier), costs, potentials and fixed fleets, and the loss coefficients and link
   efficiencies.
3. Weather: NASA POWER hourly point data at every node, 2001–2022 (50 m wind, irradiance,
   2 m temperature). Capacity factors come from a smoothed power curve and a simple PV model,
   with one scale per generator calibrated so that the mean over the network's snapshot
   window equals the mean PyPSA-Eur `p_max_pu`. Load comes from a national temperature
   model fitted to ENTSO-E load (hour-of-week × seasonal harmonics plus heating and cooling
   degrees), driven by node temperatures and scaled to the network's nodal mean load.
4. Model-completion data: the line loss coefficient is $\alpha = r/v_{\rm nom}^2$ per line,
   combined over merged parallel lines with their flow split,
   $\alpha = \sum_i\alpha_i(b_i/\sum_j b_j)^2$; DC link efficiencies are
   $0.98\cdot0.977^{\,\text{length}/1000\,\text{km}}$ (the PyPSA-Eur links are lossless);
   existing capacities, old-vintage marginal costs and emission factors come from the
   network; fixed thermal fleets have availability 0.9. `--augment` adds these datasets to
   an existing file without changing its other contents.
5. `scripts/prepare_era5.py` replaces the weather of a file by ERA5 (hourly, 0.25°), with
   capacity factors averaged over each bus by the PyPSA-Eur availability matrices and
   population-weighted temperatures (`model256_era5.h5`).

Upper boxes of gas and storage power are 1.5 times the nodal peak load. Weather-model and
LP caches are keyed by content fingerprints of these files.
