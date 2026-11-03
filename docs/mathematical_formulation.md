# From a PyPSA LP to a joint weather × decision sampling problem — mathematical formulation

This note describes the model as it is implemented in this repository. It
follows the setup notes ([v1](../near_optimal_sampling_setup.md),
[v2](../near_optimal_sampling_setup_v2.md)) and records where the
implementation makes its own choices or leaves something out. Section numbers
§v1.x / §v2.x refer to those notes. Gaps are collected in §9.

---

## 1. Starting point: the PyPSA capacity-expansion LP

The source is the PyPSA example network `carbon_management` (a PyPSA-Eur
hybrid 2050 net-zero scenario: 91 AC nodes, sector coupled, 168 snapshots of
3 h). Only its **electricity layer** is used, at the original node resolution,
restricted to a region (default DE, FR, BE, NL, LU, CH, AT, CZ, PL, DK: 40
nodes, 79 AC lines, 8 DC links).

### 1.1 What is taken from the network

| element | treatment |
|---|---|
| AC buses | nodes $n=1..N$ (`location`/`country` attributes) |
| AC lines | parallel lines between the same pair merged: $b_e=\sum 1/x^{\rm pu}_{\rm eff}$ (normalized to mean 1, so θ is in MW), rating $s_e=\sum s^{\rm opt}_{\rm nom}$; **fixed** |
| DC links | merged per pair, bidirectional capacity = max of the two directions; **fixed** |
| onwind, offwind, solar (+ rooftop) | one generator per (node, carrier); investable, $\bar k$ = summed `p_nom_max`, capex capacity-weighted |
| OCGT links gas→AC | investable dispatchable generator: capex $c^{\rm link}/\eta$ per MW$_{\rm el}$, marginal cost $(c_{\rm gas}+0.198\,t/{\rm MWh}\cdot p_{\rm CO_2})/\eta+c^{\rm VOM}$, $p_{\rm CO_2}=150$ €/t; $\bar k=1.5\times$ nodal peak load |
| nuclear, run-of-river | fixed fleet (`p_nom_opt`), constant availability 0.9 and 0.5 |
| battery (store + charger/discharger) | investable storage power $k^P$, energy $k^E=6\,{\rm h}\cdot k^P$, capex = charger + discharger/η + 6 h·store |
| H₂ (store + electrolysis/fuel cell) | same with 168 h |
| PHS, hydro storage units | fixed; hydro gets a constant inflow $0.3\,k^P$; units with < 0.5 h storage dropped |
| electricity loads | `electricity` + `industry electricity` + `agriculture electricity`, mean over the snapshot window per node |

Investments are a box $0\le k\le\bar k$; line ratings and energy-to-power
ratios are fixed (they would shape $\Omega(k)$, §v1 6.2).

### 1.2 The LP in physical units

For one weather year at step length $\Delta t$ (§6.1 uses it):
$$
\min_{k,p,\theta,u,p^\pm,e,r^\pm}\ \langle c_k,k\rangle+\Delta t\sum_t\Big[\sum_g c_g p_{g,t}+\sum_s c_s(p^+_{s,t}+p^-_{s,t})+\sum_n(c_{\rm shed}r^+_{n,t}+c_{\rm dump}r^-_{n,t})\Big]
$$
subject to $0\le p_{g,t}\le\bar a_{g,t}k_g$, $|b_e(\theta_a-\theta_b)|\le s_e$,
$|u_l|\le\bar u_l$, $0\le p^\pm_s\le k^P_s$, $0\le e_s\le h_sk^P_s$,
$e_{s,t+1}-e_{s,t}\le\Delta t(\eta^+p^+_{s,t}-p^-_{s,t}/\eta^-+q_s k^P_s)$ (cyclic),
and the balance $\bar d_{n,t}+(B f)_{n,t}+(B_Cu)_{n,t}+(H(p^+-p^-))_{n,t}-(Gp)_{n,t}=r^+_{n,t}-r^-_{n,t}$.
With $c_{\rm shed}=3000$, $c_{\rm dump}=5$ €/MWh this is the PyPSA problem with
backstop shedding and dumping at every node.

### 1.3 Reformulation as an energy model for sampling (§v1 2)

Operation is written in **utilization coordinates**
$p=\bar a\,k\,\omega_G$, $u=\bar u\,\omega_C$, $p^\pm=k^P\omega_\pm$, $e=h\,k^P\omega_e$,
$f_e=b_e(\theta_a-\theta_b)$. With the energy-to-power ratio fixed, the storage
inequality becomes $k$-independent,
$$
g_{s,t}:=h_s\frac{\omega_{e,s,t+1}-\omega_{e,s,t}}{\Delta t}-\eta^+_s\omega_{+,s,t}+\frac{\omega_{-,s,t}}{\eta^-_s}-q_s\ \le 0,
$$
so $\Omega$ depends neither on $x$ nor on $k$. The balance is priced (§v1 2.4):
$r=\bar d+Lθ+B_Cu+H(p^+-p^-)-Gp$, cost
$C_x(k,v)=\langle c_G,p\rangle+\langle c_S,p^++p^-\rangle+c_{\rm shed}\langle\mathbf 1,(r)_+\rangle+c_{\rm dump}\langle\mathbf 1,(r)_-\rangle$
($\Delta t$-weighted). Weather enters only through $\bar d=(d)_+$ and $\bar a={\rm clip}(a,0,1)$.

**Regularized Ω (default).** Following the design intent that constraints are
regularized rather than exact, every hard constraint is replaced by a priced
violation with physical slope $c_{\rm viol}$ (default $=c_{\rm shed}$):

| constraint | penalty per step (× Δt) |
|---|---|
| $0\le\omega_G\le1$ | $c_{\rm viol}(k_g+\kappa_0)\,{\rm dist}(\omega_G,[0,1])$, cost evaluated on ${\rm clip}(\omega_G)$ |
| $-1\le\omega_C\le1$ | $c_{\rm viol}\bar u_l\,{\rm dist}(\omega_C,[-1,1])$ |
| $0\le\omega_\pm\le1$ | $c_{\rm viol}(k^P+\kappa_0)\,{\rm dist}$ |
| $0\le\omega_e\le1$ | $c_{\rm viol}(k^P+\kappa_0)h\,{\rm dist}$ (energy, no Δt) |
| storage dynamics | $c_{\rm viol}(k^P+\kappa_0)(g)_+$ on clipped values |
| line ratings | $c_{\rm viol}(|f_e|-s_e)_+$ |
| $0\le k\le\bar k$ | $\gamma_k\langle c_k,{\rm dist}(k,[0,\bar k])\rangle$ added to capex ($\gamma_k=10$), operation sees ${\rm clip}(k)$ |

$\kappa_0=1$ MW keeps the target proper for units with zero capacity (otherwise
their utilization would be unconfined). With `regularized = false` the hard
polytope is used instead (§4.4).

---

## 2. Target measures

Exactly as §v1 1: $Z(x,k)=\int e^{-\beta_vC_x(k,v)}dv$,
$\rho_1=\rho_K\rho^x\pi(dv|x,k)$ and
$\rho_1\propto e^{-\beta_k\langle c_k,k\rangle}\rho^x e^{-\beta_vC}e^{-b(x,k)}$ with
$b=\log Z-(\beta_k/\beta_v)\Lambda$, $\Lambda=\mathbb E_{x'}\log Z(x',k)$.
α = 0 drops $b$.

**Temperatures** are fixed by experience: $\beta_v=4\cdot10^{-5}$,
$\beta_k=4\cdot10^{-8}$ €⁻¹. The equipartition calibration of §v1 1.5 is
available as an option. It counts the coordinates *pinned* at the
interior-point (relative-interior) LP solution, because in
$\min c\cdot v$ on a box with $c>0$ all $d$ coordinates are pinned and
$\mathbb E[\text{excess}]=d/\beta$; unpinned coordinates span the flat optimal
face and carry no excess. This reads §v1 1.5 ("not pinned") the other way round.

---

## 3. Weather

### 3.1 Data

* NASA POWER hourly point data at every node, 2001–2022 (22 years, Feb 29
  dropped): MERRA-2 wind at 50 m, CERES irradiance, 2 m temperature.
* Wind CF: smoothed generic power curve at 100 m (shear 1/7); solar CF:
  ${\rm clip}(\alpha\,{\rm GHI}/1000)$. The scale α is calibrated per generator
  so that the mean over the network's snapshot window equals the mean PyPSA-Eur
  `p_max_pu` of that generator (offshore at the node's land coordinate).
* Load: per country $\ell_t/\bar\ell=$ hour-of-week × (1, cos, sin of day of
  year) + heating/cooling degree terms of the 24 h-EWMA temperature, fitted on
  ENTSO-E load 2015–2019 against OPSD population-weighted temperature, then
  driven by the **node** temperature and scaled to the network's nodal load.

### 3.2 Anomalies and coordinates

At the finest time step $h_L$: climatology $\bar x$ = hour-of-year mean over
the years, smoothed over ±7 days per step of day; anomaly $x'=x-\bar x$;
standardized by one scalar per weather field $s_f$ (std of node anomalies).
Weather fields: load (all nodes), onwind, offwind, solar (on their generator
supports). They are expanded in the same generating system as the decisions
(§4), so coarse weather is the restriction of fine weather; node-level data at
level ℓ are $U_{f,\le m_\ell}\,(R_\ell\bar x+s_f x'_\ell)$ followed by the data map.

### 3.3 Model (§v1 3.2, non-centered)

* **Coarsest level**: regime-switching VAR(1) on the level-0 coordinates
  $z_\tau\in\mathbb R^{n_0}$ (all fields' level-0 modes stacked),
  $z_\tau\mid m_\tau,z_{\tau-1}\sim\mathcal N(\Phi_{m}z_{\tau-1}+M_mh(\tau),\Sigma_m)$,
  $h(\tau)=(1,\cos 2\pi\tau/T_0,\sin2\pi\tau/T_0)$, $z_1\sim\mathcal N(0,\Sigma_0)$
  with $\Sigma_0$ the pooled covariance; $K=4$ regimes, Markov chain $(P,\pi_0)$.
  Fitted by EM (k-means initialization on 7-day smoothed states, ridge 1e-2).
* **Details** level ℓ → ℓ+1, per field $f$, coarse block $j$, regime $m$ of the
  ancestor level-0 step, meteorological season $s$ and phase of day $p$:
  * time details of an existing mode $i$: $\xi=A\rho+L w$,
    $\rho=(1,z_i(j-1),z_i(j),z_i(j+1))$;
  * the steps of a newly added mode $i$: $\rho=(1,z_1(j))$ (the field's first,
    i.e. mean, coordinate);
  * $A,L$ from responsibility-weighted least squares, $LL^\top=Q$, with a
    pooled-statistics prior of 20 pseudo-observations per group.
  Detail blocks are independent given the coarse level; fields and modes are
  independent of each other at the detail level.
* Weather state at level ℓ: $(m,z_0,w_1,\dots,w_\ell)$ with
  $p^w_\ell=p(m)p(z_0|m)\prod_j\mathcal N(w_j;0,I)$ (projective, §v1 3.3).

---

## 4. Spatio-temporal layers

### 4.1 Fields

Every quantity indexed by space and time is a **field**: a support (set of
nodes, or of DC links), a kind (k: time-independent; step; instant) and a
node-level box. Fields: θ; $\omega_G$ per generator carrier; $\omega_C$; per
storage carrier $\omega_+,\omega_-$ (step) and $\omega_e$ (instant); $k$ per
investable carrier; the four weather fields.

### 4.2 Spatial generating systems

A global ordered orthonormal system $U_g=(g_1,\dots,g_N)$ of $\mathbb R^N$ is
chosen; for a field with support $S$ the basis $U_S$ is the sequential
Gram–Schmidt orthonormalization of the restricted vectors $g_j|_S$, $j=1,2,\dots$
(dropping dependent ones). `rank_at[M]` counts the basis vectors spanned by the
first $M$ global vectors, so the level-ℓ basis is a **prefix**
$U_S[:,1{:}m_\ell]$ and the bases are nested. θ additionally has the
per-AC-component constants (gauge) projected out. DC links use the identity.

Two systems are implemented:

* **Laplace modes** (`basis = "laplace"`): eigenvectors of the
  capacity-weighted Laplacian of the AC+DC graph (weights $s_e$, $\bar u_l$), in
  ascending eigenvalue. Dense.
* **Graph Haar** (`basis = "haar"`, v2 §4.1 with ε = 0): for nested partitions
  (network → countries → nodes), the constant followed, per level and per parent
  cell, by sequential size-weighted Helmert contrasts between child cells,
  $h\propto\mathbf 1_{A}/|A|-\mathbf 1_{B}/|B|$ (A = union of the first $j$
  children, B = child $j+1$). Each vector is supported on one parent cell; the
  first $M_\ell$ vectors span the cell-constant fields of partition ℓ. Level-0
  large-scale Laplace modes (v2 §4.1, ℓ = 0) are **not** included.

### 4.3 Temporal splitting and levels

A level is $(M_\ell,h_\ell)$: the first $M_\ell$ global spatial vectors and a
step of $h_\ell$ hours ($M$ non-decreasing, $h_\ell$ divisible by $h_{\ell+1}$,
year = 8760 h, cyclic). Going from ℓ to ℓ+1 with ratio $r=h_\ell/h_{\ell+1}$:

* **step fields**: block mean of each existing mode plus the orthonormal
  Helmert contrasts $H\in\mathbb R^{r\times(r-1)}$ within the block;
* **instant fields**: decimation (values at block-start instants) plus the
  deviations of the interior instants from linear interpolation (hierarchical
  surpluses);
* **new spatial modes**: all their fine steps.

The fine state is a linear bijection $s_{\ell+1}=(s_\ell,\xi_\ell)$ with
constant Jacobian; $\xi_\ell$ is stored per field as
$[\xi^T\,(m_c,r-1,T_c);\ \xi^S\,(m_f-m_c,r,T_c)]$. This is v2 §4.6 (ε = 0).

### 4.4 Level models

$C^{(\ell)}$ is evaluated on the **prolongation** of the level-ℓ state to nodes
at the level's time step: node values $Y=U_{\le m_\ell}Z$, then

* fields whose node constraints are not hard are clipped for the cost and their
  violations priced (§1.3);
* the balance is priced on the projection of the nodal residual onto the
  level's load modes, $\tilde r=U_{\le m}U_{\le m}^\top r$, the analogue of
  aggregating the balance over a cluster; with the Haar system this is the cell
  mean, and since $\varphi(r)=c_{\rm shed}r_++c_{\rm dump}r_-$ is positively
  homogeneous, $\sum_n\varphi(\tilde r_n)=\sum_{\rm cells}\varphi(\sum_{n\in c}r_n)$,
  exactly the coarse balance of §v1 2.4;
* at a spatially complete level the model is exact.

With hard constraints (`regularized = false`), complete fields keep their
exact node constraints and incomplete fields get an outer coefficient box
(zonotope bound of the node box; for θ, LP bounds over the line-limit
polytope), which keeps (A1) — fine feasible ⇒ coarse feasible — exact.

**Observed consequence.** With hard constraints and fixed fine details, a
coarse move almost always leaves Ω wherever dispatch or capacity is pinned at a
bound (in equilibrium within ~$1/(\beta_v\cdot{\rm slope})\approx10^{-5}$ of it):
fibers are empty in the typical case, not the exceptional one. This is why the
regularized formulation is the default.

### 4.5 Level targets and increments

$$
\pi_\ell(s_\ell)\propto e^{-\beta_k\,{\rm capex}_\ell}\,p^w_\ell(x_\ell)\,e^{-\beta_vC^{(\ell)}}\,e^{-\hat b(x_0,k_0)},
$$
with $\hat b$ evaluated on the level-0 restriction of $(x,k)$ at every level
(so $\hat b_{\ell+1}-\hat b_\ell=0$) and $\hat F_\ell=0$. States cache
$L_\ell=\log\pi_\ell-\log p^w_\ell$; the increment is
$\Delta_\ell(s)=L_{\ell}(R s)-L_{\ell+1}(s)$ and MLDA accepts with
$\exp(\Delta(s)-\Delta(\psi))$.

The reference measure is Lebesgue in utilization/mode coordinates for $k$ and
$v$ (κ = 1 trivially, ε = 0). There are no Gaussian level priors on $k$ or $v$.

---

## 5. Bias (§v1 6)

$$
\hat b(x,k)=\hat g(x,k)-\tfrac{\beta_k}{\beta_v}\,\bar{\mathbb E}_{x'}\hat g(x',k)+\Big(1-\tfrac{\beta_k}{\beta_v}\Big)\hat h(k),
$$
i.e. $\hat b_1=\hat g-\bar{\mathbb E}\hat g$ and $\hat\Lambda=\bar{\mathbb E}\hat g+\hat h$.

* $\hat g=\sum_t\Delta t\,\theta^\top\phi(u_{1,t},u_{2,t})$ with system residual
  load $u_1=(D_t-R_t(k))/\bar D$ ($R_t$: availability × capacity of all
  non-dispatchable generators) and shortfall $u_2=u_1-{\rm firm}(k)/\bar D$
  (gas + storage power); $\phi$ = linear term plus softplus ramps (width 0.05)
  at thresholds −1.5:0.25:0.75 (u₁) and −0.6:0.1:0.4 (u₂). Capacities enter
  clipped.
* $\bar{\mathbb E}$ over 16 fixed exact draws from the weather model.
* $\hat h(\kappa)=\sum_i a_i\kappa_i+\tfrac12c_i\kappa_i^2$ on the level-0
  coefficients of the **clipped** node capacities, scaled.
* **Regression (score identities, §v1 6.2) in state coordinates**: $\nabla_{z_0}\hat g$ on
  $-\beta_v\nabla_{z_0}C$ (the identity holds for the state coordinates, i.e.
  for the projection of the node gradient onto the modes), and
  $\nabla_\kappa\hat h$ on $-\beta_v\partial_kC-\partial_k\hat g$; running least
  squares, frozen after the warmup.
* **Warm start**: one observation with weight 20 from the LP's zero-temperature
  scores ($\partial F/\partial\bar d=-$balance duals, $\partial F/\partial\bar a=k\,\mu$,
  $\partial F/\partial k=\sum\mu\,\bar a$ from the capacity-constraint duals μ),
  restricted to level 0 (block sums in time, mode prefix in space).

**Known failure.** In the load tail $\partial_x\log Z\to-\beta_v c_{\rm shed}\Delta t$
per MW (≈ 3 per MW-day). Any slope mismatch of $\hat g$ there is linear in $x$
at that scale and outweighs the quadratic weather prior; at α = 1 the chain
drove the load anomaly to +117σ. A structurally safe form is
$\hat g=-\beta_v\Delta t\,\bar D\sum_t\int^{u_t}p(u)\,du$ with $p$
non-decreasing from $-c_{\rm dump}$ to exactly $c_{\rm shed}$ (non-negative
ramp weights summing to $c_{\rm shed}+c_{\rm dump}$), which cancels the tails by
construction. Not implemented.

---

## 6. Initialization and quantities

### 6.1 Initial state

One exact weather draw; the LP of §1.2 at the first spatially complete level
(default daily) for that year; conversion to utilizations (0.5 where the
capacity is zero); push into the interior, $v\leftarrow0.98v^\ast+0.02v_c$
($v_c$: ω = 0.5, ω₋ = 0.1, θ = 0, links 0), $k$ clipped to $[10^{-3},1-10^{-3}]\bar k$;
prolongation (zero details) to finer time levels and restriction to coarser
levels. Prolongation in time of a feasible daily state is feasible at finer
steps, but its cost is poor (intra-day load and solar variation against
constant daily dispatch).

### 6.2 Quantities of interest

$Q(s)$: capacity totals per investable carrier (the first Haar or Laplace
coordinate is the normalized support indicator, so totals depend only on the
coarsest level), capex, opex, total, shed energy, penalty, regime frequencies,
mean of the first weather coordinate per field. Multilevel estimator §v1 5.3
with subchains continued to $J_\ell$.

---

## 7. Investment–dispatch geometry

In utilization coordinates investments are pinned given dispatch
($p=\bar a k\omega$). For a carrier with capacity field coefficients $z_k$ and
utilization coefficients $z_\omega$, the map $(z_k,z_\omega)\mapsto(cz_k,z_\omega/c)$
keeps every nodal dispatch fixed; with $\log c\sim\mathcal N(0,\sigma^2)$ it is
an MH move with Jacobian $c^{\,m_k-\sum m_\omega T}$ (storage: $\omega_\pm,\omega_e$
scaled together; the ratio-form storage constraint is invariant). This is the
per-carrier version of v2 §2.6 ("change $k$ at fixed $p$").

---

## 8. What has been verified

* Nested bases orthonormal and nested; θ basis orthogonal to the constants.
* Split/merge is an exact bijection; block means commute with restriction.
* (A1) for the hard polytope: restrictions of a feasible finest state are feasible.
* Analytic gradients of the cost (v, k, weather) against finite differences.
* Σ of the per-step Gibbs terms equals the full level-0 potential (to 1e-15, interactive check).
* Reflective HMC reproduces a truncated Gaussian on a polytope.
* FFBS samples the exact regime posterior (enumeration).
* The generic MLDA core is exact on a three-level Gaussian toy with wrong
  coarse models, a random-walk mode and a pCN mode.
* LP duals satisfy the KKT relation $-\partial F/\partial k\approx c_k$ for the built technologies.

---

## 9. Gaps in the formulation

1. **Coarse spatial energy model.** Coarse levels evaluate the node-level model
   on the prolongation (clipped, projected balance) rather than a genuinely
   aggregated network with coarse components (v2 §4.6 "Space": $E_X^\top LD_X=0$,
   $E_X^\top G=G_cR_G$, per-class capacities). With Haar the prolongation is
   piecewise constant, which gives the Galerkin coarse Laplacian for flows, but
   line ratings are still checked line by line.
2. **θ splitting** uses the Euclidean (Haar or Laplace) orthogonality, not the
   $L$-orthogonal splitting of v2 §4.2/4.6.
3. **Generating system.** Only ε = 0. No overcomplete systems, no level priors
   $Q_\ell=\sigma^2(I-M)+\epsilon^2M$, no κ = 0 Gaussian reference or κ = 1
   correction (v2 §4.2–4.3); level-0 Laplace modes are not combined with the
   partition levels; no areas between countries and nodes.
4. **Logit-space expansion** of bounded fields (v2 §4.5) not implemented.
5. **Weather model.** No interannual factor; no nested random-effects variant
   (v2 §3.5); details independent across fields and modes; no validation on
   held-out years (regime persistence, correlations per level, low-wind cold
   spells); 22 instead of 60 years; load anomalies are temperature-driven only;
   Gaussian anomalies clipped by the data map bias the mean CF upward (onwind
   0.25 vs 0.22 in the data).
6. **Bias.** System-level features only (no nodal/block-local ĝ); tail leak
   (§5) unresolved; no level corrections $\hat g_\ell-\hat g_{\ell-1}$; ĥ's
   curvature poorly identified; $\hat F_\ell=0$.
7. **Fixed capacities** for lines and energy-to-power ratios (boundary terms
   of §v1 6.2 not estimated).
8. **Perfect-foresight variant** (§v1 1.6) and the per-regime Dunkelflaute
   classification beyond the crude per-regime ĝ average are not implemented.
9. **Data simplifications.** Region subset; today's network loads with 2050
   costs; offshore wind at the land coordinate (only the mean is calibrated);
   hydro inflow constant; run-of-river availability constant.
10. **Convergence** of any chain has not been demonstrated; the α = 1 target
    has not been sampled successfully.
