# Sampling-friendly formulations of the PyPSA components

This note goes through PyPSA's components one at a time. For each it compares the LP
form with formulations whose coordinates, constraints and reference measure
suit sampling. It follows [physical_formulation.md](physical_formulation.md)
and corrects it in places (§10). Code references are to `src/energy.jl` and
`src/local.jl`.

**Notation.** Node $n$, step $\tau$ (length $\Delta t$), instant $\tau$ for states
of charge. $d$ is load, $A_{g\tau}\in[0,1]$ availability, $k$ capacity [MW],
$p$ dispatch [MW], $e$ state of charge [MWh], $\theta$ angles,
$f=WB^\top\theta$ line flows, $u$ link flows. The residual
$r=d+L\theta+B_Cu+H(p^+-p^-)-Gp$ is positive for a shortage. Reference numbers
(40 nodes, daily, $T=365$):
$\beta_v=4\cdot10^{-5}$, $\beta_k=4\cdot10^{-8}$ €⁻¹, $c_{\rm shed}=3000$,
$c_{\rm dump}=5$ €/MWh, gas capex $c=4.4\cdot10^4$ €/MW/a.

A direction confined at a kink with one-sided slope $s$ [€/MWh] at one step has width
$$w(s)=\frac1{\beta_v s\,\Delta t}:\qquad w(c_{\rm shed})=0.35\ {\rm MW},\quad w(10)=104\ {\rm MW},\quad w(c_{\rm dump})=208\ {\rm MW}.$$

---


> **Model decision (user, after review):** the model is perfect foresight for the
> dispatch *per weather year*, with investments chosen against the weather
> distribution. That is the two-temperature target ρ₁ (dispatch near-optimal
> given (k, x) for the whole year x; capacity marginal e^{−β_k(K + E_x F_x(k))}).
> The ensembles ρ₀ and ρ^pf (one temperature, investments with foresight of a
> single year) are not the model; results below that hold "for ρ₀/ρ^pf"
> (collapse of k, the 570 MW idle margin) are computational devices at most,
> e.g. for initialization. The ρ₁ statements (§0.4 pinning, transported
> peak-set capacity moves with b̂/mean-force acceptance) are the relevant ones.

## 0. What the reference measure does, and which one to use

### 0.1 Entropy enters at order ε_v, and how it reaches ρ_K depends on the ensemble

Take a reference $m(dv\mid k)$. If the near-optimal operational set is a vertex
plus a flat face,
$$
\log Z_m(x,k)=-\beta_vC^*(x,k)+S_m(x,k)+o(1),\qquad
S_m=\sum_{i\in\rm conf}\log\frac{\mu_i}{\beta_v s_i}+\log{\rm vol}_m(\text{flat face}),
$$
where $s_i$ are the reduced-cost slopes of the confined directions and $\mu_i$ is
the density of $m$ along direction $i$, per physical unit (MW). The reference
therefore changes $F=-\beta_v^{-1}\log Z$ only at order $n_{\rm conf}/\beta_v\approx\varepsilon_v$,
the operational slack. How much it changes the capacity marginal depends on the ensemble:

| ensemble | capacity marginal | entropic force on $k$ |
|---|---|---|
| $\rho_0\propto e^{-\beta_kK-\beta_vC}$ (also the perfect-foresight $\rho^{\rm pf}$) | $e^{-\beta_kK-\beta_v\mathbb E F}$, so **opex is weighted $\beta_v/\beta_k=10^3$ times capex** | $\partial_k S_m$, full strength |
| $\rho_1$ (target) | $e^{-\beta_k(K+\mathbb E F)}$ | $(\beta_k/\beta_v)\,\partial_k\mathbb E S_m$ |
| profile ($\beta_v\to\infty$) | $e^{-\beta_k(K+\mathbb E C^*)}$ | none |

The profile ensemble is the reference-free definition of near-optimal
investments: the analogue of MGA, uniform in $k$ on the ε-near-optimal set
with $v$ re-optimized. Finite $\beta_v$ is a computational device and a model of
operational slack. **A good reference makes $\partial_k S_m$ and $\partial_x S_m$
small and economically meaningful.** Size of the force for utilization
coordinates (§0.3): per confined dispatch coordinate it is $-1/k$, so

$$\frac{\text{entropic force}}{\text{capex force}}=\frac{n_{\rm conf}}{\beta_v c\,k}\ (\rho_1)\quad\text{or}\quad\frac{n_{\rm conf}}{\beta_k c\,k}\ (\rho_0).$$

For gas with $n_{\rm conf}\approx T$ and $k=2$ GW this is **0.10 at $\rho_1$ (daily),
2.5 at $\rho_1$ (hourly, β_v unchanged), and 104 at $\rho_0$**. The ratio is
resolution-independent only if $\beta_v\propto n_v$ (equipartition calibration, v2 §1.5).
The current fixed $\beta_v$ does not satisfy this.

### 0.2 Criteria for a defensible entropy

* **E1 (no spurious $k$-dependence).** The width of a cost-confined direction should not depend on $k$.
  Then $\partial_kS_m$ comes only from reduced costs and from genuine flat faces.
* **E2 (no spurious $x$-dependence).** $S_m$ should not contain terms like $\sum_\tau\log A_\tau(x)$.
  The bias $b$ has to cancel the $x$-dependence of $\log Z$ to O(1) nat
  (alternatives.md §0). Every per-step weather term in $S_m$ is something $\hat b$ must
  learn, and the system-level $\hat g$ cannot represent it.
* **E3 (no artificial flat directions).** A redundant variable of the LP
  (simultaneous charge and discharge, curtailment versus dump, the split among
  identical co-located units, antiparallel lossy links) adds a volume
  ∝ capacity at every step. That is an extensive entropic force with no economic content,
  and it is also a slow direction for the sampler.
* **E4 (consistency).** The reference should behave predictably under time aggregation,
  spatial aggregation and splitting a unit into two (§7.3). Exact invariance under
  splitting is impossible for any product Lebesgue measure, so fix the technology
  granularity once: one unit per (node, carrier, cost class).
* **E5 (locality).** $\log m$ should be a sum of terms with the same locality as
  the cost.

**Choice.** Use Lebesgue measure in physical units (MW for powers and flows,
MWh for states of charge) on a **non-redundant** set of physical coordinates.
Treat the balance as $\delta(r)$, with shedding and curtailment as bounded units
(§1). Use Lebesgue measure in MW for capacities. This satisfies E1–E3 for
confined coordinates. Utilization coordinates violate E1 and E2. Logit and
multiscale-Gaussian references (v2 §4.3–4.5) are proper by construction but
replace the near-optimal volume by an arbitrary prior. Use them as sampler
coordinates, with their Jacobian (κ = 1), not as the target.

### 0.3 Jacobian rules used below

1. **Bijection** $v=\phi(z)$: the density in $z$ is $\pi(\phi(z))|\det D\phi|$.
   Example: $p=A k\omega$ gives $dp=(Ak)\,d\omega$ per coordinate.
2. **Elimination by an equality** $Mv=r$, with dependent coordinates $v_D$ and
   free coordinates $v_F$:
   $\delta(Mv-r)\,dv=|\det M_D|^{-1}dv_F$. With unit coefficients this is 1 for any
   choice of $D$. With efficiencies, $1/|\eta|$ per eliminated conversion.
   This is $\lim_{c\to\infty}$ of the priced balance, up to a constant.
3. **Integrating out** a variable exactly (collapse) leaves the marginal on the
   remaining variables unchanged. A collapsed sampler mixes at least as well as
   the uncollapsed one (Liu 1994).
4. **Group moves** $z\mapsto g_uz$: generalized Gibbs with factor
   $|\det Dg_u|$ (Liu–Sabatti). This is `kscale_sweep!`.
5. **State-dependent frames.** A line move along a direction $d(v)$ is reversible
   only if $d$ does not change along the line. Choose it from $(k,x)$ and from
   coordinates the move does not touch, or add an MH correction.

### 0.4 Capacity mobility at ρ₁: pinning

In $\rho_1$ the conditional of a dispatchable capacity given the dispatch is
$$\rho_1(k\mid x,v)\propto\rho_K(k)\,\pi(v\mid x,k)\propto e^{-\beta_kck}\,\frac{\mathbb 1\{k\ge M(v)\}}{Z(x,k)},\qquad
\partial_k\log Z\approx\beta_v\mu ,$$
where $M(v)=\max_\tau p_\tau$ and $\mu\approx c$ is the capacity rent. So $k-M\sim{\rm Exp}(\beta_v c)$,
a margin of **0.57 MW**. In $\rho_0$ the margin is $1/(\beta_kc)=570$ MW.

Consequences:
* Moves of $k$ at fixed physical dispatch ("$k\mid p$", the margin), including
  `kscale` at fixed $p$, move $k$ by about 1 MW at $\rho_1$. Moves of $k$ at fixed
  utilization fail because interior dispatch then leaves the merit order.
* Capacity moves only if the peak dispatch moves with it. A capacity move must be
  **transported**: $k\to k+\delta$, with the dispatch on a peak set $S(x,k)$ moved by
  $+\delta$ and price-setting units compensating. Its acceptance needs
  $\log Z(x,k+\delta)-\log Z(x,k)$, from $\hat b$ or from the integrated mean
  force. This is the double intractability of alternatives.md §7.
  Parametrization only shortens the re-equilibration lag.
* Collapsing $k$ (physical_formulation §3) is exact for $\rho_0$ and $\rho^{\rm pf}$.
  For $\rho_1$ it replaces $\beta_kcM$ by
  $\beta_kcM+\hat b(x,M)+\log(\beta_kc+\partial_k\hat b)$. The capacity marginal is
  then the small difference of two terms of size $\beta_v\mu$, so it needs
  $\partial_k\hat b$ to relative accuracy $\beta_k/\beta_v=10^{-3}$. Use collapse only
  for $\rho_0$ and $\rho^{\rm pf}$. At $\rho_1$, keep $k$ explicit, with regularized walls
  and mean-force dynamics.

---

## 1. Nodal balance (KCL)

**LP.** $\sum_g p_{g}+\sum_s(p^-_s-p^+_s)+\sum_\ell \eta_\ell u_\ell-\ldots-d=0$ per $(n,\tau)$, with dual
$\lambda_{n\tau}$. The code uses the priced form
$\Delta t\,\varphi(r)=\Delta t(c_{\rm shed}r_++c_{\rm dump}r_-)$.

**B0, priced, current.** This form is identical to a δ-balance with shedding and
curtailment as physical units, with simultaneous shedding and curtailment
integrated out:
$$\int\!\!\int ds\,dc\;\delta(s-c-r)\,e^{-\beta_v\Delta t(c_{\rm shed}s+c_{\rm dump}c)}=\frac{e^{-\beta_v\Delta t\varphi(r)}}{\beta_v\Delta t(c_{\rm shed}+c_{\rm dump})}.$$
So the current target is already the physical δ-balance target. It lacks only the
bounds $s\le d$ and $c\le$ curtailable supply.

The problem is the coordinates. With the shed/dump slack as the
dependent unit, every coordinate move hits slope $c_{\rm shed}$: width 0.35 MW.
This is why pair and re-routing moves are needed. The $N\cdot T$ kinks at $r=0$
are intrinsic to having backstops: 14 600 of 50 907 confined directions
(overnight §3). They remain in every reparametrization.

**B1, price-setting dependent unit.** Pick $j(n,\tau)$ and use coordinates
$(p_{-j},r)$ with $p_j=d+\ldots-r$. By rule 2 the Jacobian is 1, so this is a
pure reparametrization. The single-coordinate conditional of unit $i$ then has
slope $c_i-c_j$, the reduced cost, instead of $c_{\rm shed}$. Pair moves with
partner $j$ become coordinate moves and can be swept in parallel colour classes.
Choose $j$ from $(k,x)$, for example the merit-order unit at the zero-temperature
price of a cheap coarse LP or the residual-load rank. Do not choose it from the
current $v$ (rule 5). With a conversion as the dependent unit, include $1/|\eta_j|$.
If $\eta_j$ depends on $x$ (heat-pump COP), the factor enters $S$ and violates E2,
so prefer dependent units with constant coefficients.

The kink moves to the bounds of $j$. When $j$ is at a bound, the effective slope
is again the backstop slope. That is the residual cost of a frame chosen from
$(k,x)$ only.

**B2, graph-Haar balance.** In Haar coordinates of the residual, the cell sum is
the coarse balance and a detail contrast is the imbalance between two child
cells. Natural dependent variables: the cell sum is absorbed by the cell's
price-setting unit, and each contrast by the potential difference across the
cut between the children (flow re-routing). The Jacobian is
$\det$ of the Haar-transformed reduced Laplacian. It is constant for fixed
lines, so this too is a reparametrization. This is the balance frame
that matches the cell moves (`cellpair`) and the coarse levels (§8.5).

**B3, elastic demand (changes the target).** Replace the kink of $\varphi$ by a
convex demand curve $\int_0^{r_+}{\rm VoLL}(s)\,ds$, with VoLL rising to $c_{\rm shed}$.
This is a heterogeneous value of lost load, physically better than a single price.
It removes the kink at $r=0$: a smooth well instead of a confined direction (½
instead of 1 in the equipartition count). It also makes the weather tail of $\hat g$
smoother. Low priority.

**Mixing and cost.** B1 and B2 remove the 0.35 MW steep direction from
coordinate moves without changing the target. A move at $(n,\tau)$ updates
$p_{j(n,\tau)}$ in O(1). A $\theta_n$ move updates the dependent units of $n$ and its
neighbours in O(deg), instead of the dense $L^+$ re-routing (`pairs.jl`, O(N²)) or the
sparse triangular solves (`GridSolver`, O(N)). Steps stay independent.

---

## 2. Lines, KVL, transformers and line expansion

**LP.** PyPSA uses a cycle-based KVL, $\sum_lC_{lc}x_lf_{l\tau}=0$, together with KCL
and $|f_{l\tau}|\le\bar p_{l\tau}s_l$. Losses are optional and piecewise linear
(secants or tangents of $r_lf^2$). Phase shifters add $\alpha_{l\tau}$ to KVL. For
extendable lines (`s_nom_extendable`) $x_l$ is held fixed in the LP and updated
iteratively outside it.

| formulation | coordinates per step | KVL | locality | remarks |
|---|---|---|---|---|
| **L1 angles** (current) | $\theta\in\mathbb R^{N-n_{\rm comp}}$ | by construction | $\theta_n$ touches incident lines and $n$ plus neighbours | constant Jacobian to the cut space; best default |
| **L2 PTDF** | none (injections given) | by construction | dense: O(E) per injection change | removes $N-1$ coordinates per step; good at coarse spatial levels (few cells) |
| **L3 tree plus cycle flows** $f=f_{\rm tree}(P)+C\phi$ | $\phi\in\mathbb R^{E-N+1}$ | $\phi$ is fixed by KVL for AC | — | **natural for transport-model parts** (DC links, pipelines, PSTs), where $\phi$ is free and a cycle move preserves every balance in O(cycle length) |
| **L4 soft KVL** | $f$ on the KCL-affine space with weight $e^{-\frac\gamma2f^\top Xf}$ | as $\gamma\to\infty$ (Thomson: DC flow minimizes $\sum x f^2$) | local | **changes the target** (Gaussian loop flows); only useful as tempering or a coarse model |

**Line limits.**
* (a) Hard $|f|\le s$: congested lines are confined directions, and the exact line sampler handles the walls.
* (b) Penalized at $c_{\rm viol}$ (current): the same, with a soft outside.
* (c) **Soft thermal limit.** The physical limit is conductor temperature, with time constants of minutes to an hour. At hourly or coarser steps a smooth convex overload cost $c_{\rm th}s\,\psi((|f|-s)/s)$ is defensible. It changes the target.
* (d) **Dynamic line rating** $s(x)$ (wind cooling). As a wall it makes Ω depend on $x$ and breaks the score identity. As a priced overload it stays in the cost and is harmless.

**Losses.** Use the exact $\psi=r_lf^2$, split ½ to each end. It is strictly
convex. It breaks the degeneracy between parallel paths, which matters wherever
DC links run parallel to AC corridors, and it removes the artificial secant
breakpoints of PyPSA. Line conditionals become piecewise exp-quadratic:
truncated-Gaussian pieces, still exact, at the cost of erf evaluations.

**Extendable lines.**
* *Fixed impedance (PyPSA).* With the physical flow measure and $s_e\ge s_e^0$, the
  collapse gives $\beta_kc_e\max(s_e^0,\max_\tau|f_{e\tau}|)$: the existing rating is
  free and only the excess is charged (brownfield hinge). This holds for
  $\rho_0$/$\rho^{\rm pf}$. At $\rho_1$ see §0.4. The max couples the steps of one line (§3.2).
* *Impedance scaling with capacity* (parallel circuits): $b_e=b_e^0s_e/s_e^0$ gives
  $f_e=(s_e/s^0_e)b^0_e\Delta\theta_e$. The limit $|f_e|\le s_e$ becomes
  $|b_e^0\Delta\theta_e|\le s_e^0$, an $s$-independent box in θ. So $s$ acts like a
  conductance capacity and $\Delta\theta$ like a utilization. The bilinearity is
  harmless for a sampler, and the target is better justified than the LP's.
  A change of $s_e$ at fixed θ changes residuals at two nodes at every step, O(T), absorbed by B1.
* *Reference measure.* With fixed lines, $d\theta$ and the flow (injection) measure
  differ by a constant. With scaled impedances they differ by
  $\det L_{\rm red}(s)=\sum_{\text{spanning trees}}\prod b_e$ per step (matrix-tree
  theorem), an extensive term $T\log\det L_{\rm red}(s)$. As in §0.1, confined flows
  have $s$-independent widths in MW but widths ∝ $1/b(s)$ in θ. **$d\theta$ is the
  utilization measure of lines.** For extendable lines use the injection measure,
  $d\theta\cdot\det L_{\rm red}(s)$, per step.

**Transformers.** The same as lines ($x$ and tap ratio enter $b$). A phase shifter is one
box coordinate $\alpha_{l\tau}$ per step. It makes its cycles free (like a link in
L3) and touches two residuals.

**Mixing.** L1 plus B1 makes every θ move local and balance-compensated. The slow
directions that remain are long-range transfers, low Laplacian modes, which are the
Haar θ-cell moves of B2. Exact losses and soft limits replace walls by smooth
terms. Extendable lines add one max coupling per line.

---

## 3. Generators

### 3.1 Renewables (time-varying `p_max_pu`, marginal cost ≈ 0)

**LP.** $0\le p_{g\tau}\le A_{g\tau}k_g$. Curtailment is $A k-p$.

**Current.** $p=Ak\omega$ with $\omega\in[0,1]$. A confined renewable coordinate
(at $\omega=1$, pinned by its opportunity cost) has ω-width $w/(A_\tau k)$, so
$S\supset-\sum_{\tau\,\rm conf}\log(A_\tau k)$. That is an entropic force against
capacity (E1) and a per-step weather term (E2): at $\rho_0$ it tilts the weather
toward low availability, and at $\rho_1$ $\hat b$ must learn $\sum_\tau\log A_\tau(x)$.
Co-located renewables (onwind, offwind, solar at one node) share curtailment
along a flat face (E3).

**R1, renewables as negative load (recommended).** Set $p_{g\tau}=A_{g\tau}k_g$
exactly. Curtailment becomes one nodal variable, the negative part of the residual,
priced at $c_{\rm curt}$ (= $c_{\rm dump}$), with bound $r_-\le\sum_{g\in n}A_gk_g$
(+ must-run). This **changes the target slightly**: it removes the curtailment split
among co-located units (a flat or nearly flat face) and the $\log(Ak)$ Jacobians.
It is better justified, since the split is not an alternative anyone reports.
It removes 99 of 567 coordinates per step (17 %; 30 % once storage is in SoC form).
* $k_g$ enters $r_{n\tau}$ linearly at all τ. A $k$ line move has up to $T$ breakpoints
  (sign changes of the residual and bounds of the dependent units), so exact sampling costs O(T log T).
* At fixed other dispatch the move is confined by $c_{\rm shed}A_\tau$ at shortage
  steps, so it needs compensation. With B1 the price-setting units absorb it, and
  with the transport formulation of §8 the absorption is pro rata and built in.
* Different renewables at different nodes can be updated in parallel. They share
  no residual row.

**R2.** Keep $\omega$ but measure in MW, $\log(A_\tau k+\kappa_0)$. This is worse than R1
on every criterion.

### 3.2 Dispatchable generators

**LP.** $\underline g k\le p\le\bar g k$, linear cost.

**D1, physical $p\in[\underline gk,\bar gk]$, explicit $k$.** E1 holds. At $\rho_1$, $k\mid p$ is pinned (§0.4).
Use it together with transported capacity moves.

**D2, collapsed** (physical_formulation §3, valid for $\rho_0$ and $\rho^{\rm pf}$):
$\pi(p)\propto\exp(-\beta_kc\max(k^0,\max_\tau p_\tau)-\beta_v\langle c,p\rangle)$.
With a floor $p\ge\underline g k$ on committed steps, $k$ lies in
$[\max p,\ \min_{\rm on}p/\underline g]$, and the integral is a difference of two
exponentials: still closed form.

**Locality of the max.** A move at step τ changes the capex iff it changes the
maximum. With a top-two cache per unit, the conditional is piecewise linear with
one extra breakpoint at $M_{-\tau}$, and the exact sampler is unchanged.
* **Parallel classes.** Draw an auxiliary $k$ from its conditional,
  $k\sim e^{-\beta_kck}$ on $[M,\bar k]$ (at $\rho_1$: $\times e^{-\hat b(x,k)}$),
  update the class with the wall $p\le k$ (the steps are then independent), and
  discard $k$. This is Gibbs on the joint, so the collapsed sequential moves and
  the augmented parallel ones can be mixed freely.
* **Peak plateaus.** When many steps are near the max (gas at full output
  through a two-week Dunkelflaute) the capex term is a $T_{\rm peak}$-fold tie, and
  raising the capacity needs all of them raised. The **peak-set move**
  $p_\tau\mathrel{+}=\delta$ for $\tau\in S$, where $S$ is the set of top-$q$
  residual-load steps (from $(k,x)$, so the move is reversible), is compensated by
  dependent units. This is the transported capacity move of §0.4.

**Brownfield and co-located vintages.** PyPSA-Eur adds new capacity as a second
component next to the existing one. With equal marginal costs this is a flat
split, with volume ∝ $\min(k^0,k^{\rm new})$ per step (E3). **Merge** them into one unit,
with capex $c\,(k-k^0)_+$ (plus fixed O&M on $k$ if retirement is a decision) and,
for different vintages, the convex merit-order cost
$C(p)=c_{\rm old}\min(p,k^0)+c_{\rm new}(p-k^0)_+$. Replacing the split by its
zero-temperature merit order changes the target and removes nuisance entropy.
Integrating it out exactly (a convolution of truncated exponentials) keeps the
target.

### 3.3 Unit commitment, part load, ramping

* **Binary UC** (`committable`): the feasible set is $\{0\}\cup[\underline gk,\bar gk]$. A Lebesgue
  reference gives "off" zero weight. A reference $w_0\delta_0+dp$ needs a weight
  $w_0$ in MW, an arbitrary modelling constant. Moves between strata are
  reversible-jump moves. Min-up and min-down times need birth/death moves of whole on-runs.
* **Fleet UC (recommended; PyPSA's linearized UC):** committed capacity
  $\kappa_\tau\in[0,k]$ and $p_\tau\in[\underline g\kappa_\tau,\kappa_\tau]$, with start-up cost
  $c_{\rm su}(\kappa_\tau-\kappa_{\tau-1})_+$ and no-load cost $a\kappa_\tau$. Physical
  measure $d\kappa\,dp$ on the cone. A useful coordinate is the headroom
  $h=\kappa-p$ (spinning reserve). The start-up term couples neighbouring steps (two
  colours). Continuous, aggregable, consistent with the hierarchy.
* **Part load and nonconvex heat rates.** Any piecewise-linear cost in $(p,\kappa)$ keeps
  exact line sampling. Nonconvexity makes $\pi(v\mid x,k)$ multimodal. The sampler
  handles this, but the LP is lost as a reference.
* **Ramping** $|p_\tau-p_{\tau-1}|\le r k$: walls on differences, two-colour parallel.
  Binding only at sub-daily steps. Price it as wear $c_r|\Delta p|$ (a kink),
  or leave it out at coarse time levels.

---

## 4. Storage units and stores

**LP** (StorageUnit):
$e_\tau=\eta_{\rm st}^{w}e_{\tau-1}+w(\eta^+p^+_\tau-p^-_\tau/\eta^-+q_\tau-{\rm sp}_\tau)$,
$0\le p^\pm\le P$, $0\le e\le hP$, cyclic or fixed initial level. A Store is $e$ with
its own $E$, connected through charger and discharger links with independent
capacities, or with efficiencies in recent PyPSA. The model's battery and H₂ come
from store plus links. The repo's (H2) inequality "energy may be lost" relaxes the
equality further.

**S1, SoC trajectory** (implemented: `storage = "soc"`, not the default). The
state is $e_\tau$ on instants. With
$\Delta_\tau=e_{\tau+1}-\eta_{\rm st}e_\tau-\Delta t\,q_\tau$:
$p^+_\tau=(\Delta_\tau)_+/(\eta^+\Delta t)$ and $p^-_\tau=\eta^-(-\Delta_\tau)_+/\Delta t$.
* **By construction:** the dynamics and the cyclic condition (instants on a
  cycle), and no simultaneous charge and discharge.
* **Remaining kinks:** $0\le e\le E$; power limits as a Lipschitz bound,
  $-P/\eta^-\le\Delta_\tau/\Delta t\le\eta^+P$; the kink at $\Delta_\tau=0$ (efficiency
  loss).
* **Dimensions:** 3T → T, 234 of 567 coordinates per step in the model.
* **Simultaneous charge and discharge.** This is the LP's way to burn surplus energy
  (an alternative dump with loss fraction $1-\eta^+\eta^-$). It is a flat or soft
  direction of the LP form, and the "storage cycling keeps drifting" of
  local_kernel.md §5 is this direction. It is gone in S1.

**Measure: MWh of state of charge.** By rule 1, Lebesgue in grid-side MW relative
to $de$ is $\prod_\tau(1/\eta^+)^{[\rm ch]}(\eta^-)^{[\rm dis]}$, which depends on
the state. For H₂ ($\eta^+=0.8$, $\eta^-=0.58$) the difference between charging and
discharging is 0.77 nats per step. That is an extensive bias toward charging regimes
with no economic content. $de$ is regime-free and additive under time aggregation.
The code's SoC form with `measure = "physical"` is exactly $de$ in MWh.

**Spill and losses.** The code spills energy beyond the power limit at no cost
(`soc_power`: $\min(-d\eta^-,1)$). For batteries and H₂ this is free destruction of
stored energy, a soft flat direction inherited from the inequality form.
Penalize it, or take it from the dump, for units without inflow. Allow spill only for
units with inflow, and also at a full reservoir: hydro that cannot store its inflow
must either generate or spill, and pricing that spill at $c_{\rm dump}$ makes it
equivalent to the nodal dump.

**Capacity (collapse, for $\rho_0$/$\rho^{\rm pf}$).**
* Fixed ratio $h$ (model): $k\ge\max(\max p^+,\max p^-,\max e/h)$.
* Independent $E$, without standing loss and cyclic: the level offset $e\to e+a$ is a
  flat direction of length $E-R$, where $R=\max e-\min e$. Integrating out $E\ge R$
  together with the offset gives $e^{-\beta_kc_ER}/(\beta_kc_E)^2$: capex on the
  **range**, with a Gamma(2) margin.
* Sample the offset exactly as one coordinate (uniform on its interval, O(1) with a
  range cache). Shifting the whole trajectory is otherwise $T$ local moves.
* With standing losses the map $\Delta\mapsto e$ is invertible
  ($\det(I-\eta_{\rm st}^TS)\ne0$) and there is no offset.

**Mixing and cost.** A change of $e$ at instant τ changes the injection at
steps τ−1 and τ (two-step block, O(1)). A **plateau move** ($e\mathrel{+}=\delta$ on
an interval of instants) moves energy between two distant steps at O(1) cost plus an
O(log T) range check for the bounds (segment tree). This is how seasonal H₂ mixes. The
hierarchical hats in time (v2 §4.6) are plateau moves at block scale. The new couplings
are the range or max for capacity and the cyclic closure.

---

## 5. Links and sector coupling

**LP.** $\underline g k\le p_0\le\bar gk$ is withdrawn at bus0, and
$p_i=\eta_i(\tau)p_0$ is delivered at bus$_i$ ($\eta_i<0$ means consumption, e.g.
electricity for DAC or CO₂ output). Capex is on $k$ (bus0 side).

* **Coordinate:** the throughput $p_0$ in input MW. One coordinate enters several balances.
  Under B1 a link move is a chain move: the link plus the dependent units at each of
  its buses. The cost is O(number of buses) per step.
* **Measure.** In input MW the reference is $x$-independent. If the link is the
  dependent unit at bus$_i$, the factor $1/|\eta_i(\tau)|$ appears (rule 2). For a heat
  pump with ${\rm COP}(T_\tau)$ that puts $-\sum_\tau\log{\rm COP}(x_\tau)$ into $S$,
  violating E2. At $\rho_0$ it tilts the weather toward cold. At $\rho_1$, $\hat b$
  must learn it. Prefer dependent units with constant coefficients at sector buses:
  stores, boilers, resistive heaters.
* **Multi-bus dependent assignment.** Assign one dependent unit to each (bus, step) so
  that the elimination is triangular (an acyclic dependency graph). Then
  evaluation is local and $|\det M_D|=\prod|\eta|$. A cyclic assignment needs a
  small solve per step.
* **Lossy bidirectional links** (PyPSA-Eur: two antiparallel links). Simultaneous use
  is a lossy flat direction, like storage cycling. Replace it by one signed flow
  $u$ with loss $(1-\eta)|u|$ (a kink at 0). This saves one coordinate per link and
  step. The same applies to H₂ and gas pipelines.
* **CHP:** one fuel coordinate plus the operating point in the feasible
  heat/power polygon (back-pressure and extraction lines as walls).
* **Stock buses** (H₂, CO₂ atmosphere): never give them a priced shed or dump.
  A dump at the CO₂ bus is free removal.

---

## 6. Global constraints (CO₂ cap and others)

**LP.** $E(v)=\sum_{g,\tau}w_\tau e_gp_{g\tau}/\eta_g+(\text{store end-point terms})\le\Gamma$,
with dual $\mu$ (the CO₂ price). In sector-coupled networks the same constraint
appears as a CO₂ atmosphere store with $e_{\rm nom}=\Gamma$.

| ensemble | form | locality | marginals |
|---|---|---|---|
| **microcanonical** | $\mathbb 1\{E(v)\le\Gamma\}$ per weather year | one global wall: O(1) with a cached total, but steps in a parallel class are coupled through the remaining budget | the cap binds in **every** weather year, so there is an implicit price $\mu(x,k)$ (spiking in bad years); robust, risk-averse portfolios |
| **canonical** | cost $+\mu E(v)$ | fully local, parallel | emissions fluctuate across years; fluctuations at fixed $(x,k)$ are tiny (std ~ $\sqrt{n_E}/(\beta_v\mu)$ ≈ $2\cdot10^4$ t) |
| **canonical, mean cap** | $\mu$ tuned so that $\mathbb E_\rho E=\Gamma$ (Robbins–Monro on one scalar) | local | the multi-year analogue of a cap with banking (ETS) |
| **random price** | $\mu$ is an auxiliary variable with prior $p(\mu)$ | $v\mid\mu$ local; $\mu\mid v$ exact in 1-D | marginal factor $\mathcal L_p(\beta_v(E-\Gamma))$, a mixture of canonical ensembles, **not** a cap |

PyPSA's single-year LP does not distinguish these ensembles. The weather ensemble
forces a choice. **Recommendation: canonical with μ tuned to the mean cap** (or
fixed at the LP dual). The per-year microcanonical cap is both harder to sample
and economically questionable. Allowance-augmentation tricks
(per-step budgets $b_\tau$ with $\sum b_\tau=\Gamma$) introduce an entropy
$(\Gamma-E)^{T-1}$ that pushes emissions down unless it is divided out, which is
global again. Avoid them. The CO₂-store formulation without extra freedom equals the
microcanonical cap.

Constraints on capacities (`transmission_volume_expansion_limit`,
`tech_capacity_expansion_limit`, cost budgets) are walls in the low-dimensional
$k$-space. Keep them hard, with cached sums.

---

## 7. Capacity expansion, brownfield, and capacities across levels

### 7.1 Box and brownfield

$k\in[k_{\min},k_{\max}]$ with capex $c\,k$ (PyPSA charges `capital_cost·p_nom`
for extendable components). Brownfield is $k=k^0+\Delta k$ with a hinge
$c(k-k^0)_+$, or a convex piecewise-linear cost if fixed O&M on existing capacity
makes retirement a decision. The reference is Lebesgue in MW. A constant rescaling
($k/\bar k$, $c\,k$) is irrelevant. Log-uniform ($dk/k$) is singular at the
legitimate value $k=0$ and implies a scale-free prior that nobody holds.
**Do not use it.**

### 7.2 The physical measure removes κ₀

In utilization coordinates a unit with $k=0$ has an unconfined ω, hence
$\kappa_0=1$ MW in the code. Under the physical measure its dispatch range is
$[0,k]$ and simply collapses at $k=0$. No floor is needed. The penalty weights
$c_{\rm viol}(k+\kappa_0)\,{\rm dist}(\omega)$ become $c_{\rm viol}\,{\rm dist}(p)$ in MW.

### 7.3 Aggregation across levels (E4)

* **Space.** A coarse capacity $K_c=\sum_{n\in c}k_n$ under Lebesgue on the
  fine $k$ has fiber volume $K_c^{n_c-1}/(n_c-1)!$ below the potential caps.
  A coarse model that drops it is biased toward small coarse
  capacities by $(n_c-1)\log K_c$. Add this term, or move in shares coordinates
  $k_n=K_cs_n$ with Jacobian $K_c^{n_c-1}$ (the toy's transported cell moves).
* **Time.** A coarse step carries one confined direction where the fine level
  carries $r$ ($r$ = block ratio). The block mean of $r$ independent confined
  coordinates is Gamma($r$), but the coarse model gives Exp. The coarse level is
  therefore $r$ times too cold in excess. Fix this with $\beta_\ell=\beta_v/r_\ell$, or with the
  fiber entropy $(r_\ell-1)\log(\text{gap to the confining bound})$.

---

## 8. Transport-first formulation

The fundamental variables are energy transport in space (flows from potentials,
links) and in time (storage states of charge). Given the transport, the net
requirement of every node and step is fixed. Shares say which local technology
covers it, and capacities follow from the shares.

### 8.1 Coordinates and map

Primary, per step τ: $\theta_\tau$ (AC, L1), link flows $u_\tau$ (signed, L3 cycles free),
storage states $e_{s,\tau}$ (S1), plus the slow renewable capacities $k_{\rm ren}$
(§8.3). The **requirement** that local dispatchable technologies must supply is
$$
Q_{n\tau}=d_{n\tau}(x)-\sum_{g\in{\rm Ren}(n)}A_{g\tau}(x)k_g+(L\theta_\tau)_n+(B_Cu_\tau)_n+\sum_{s\in n}\big(p^+_s-p^-_s\big)(e_{s,\tau},e_{s,\tau+1}),
$$
with $L\theta$ the net export. Let $D_n$ be the dispatchable technologies at $n$ plus shedding, $m_n=|D_n|$.

* **T1, shares of the requirement** (the proposal as stated):
  $\sigma_{n\tau}\in\Delta^{m_n-1}$, $p_{i,n\tau}=\sigma_{i,n\tau}(Q_{n\tau})_+$,
  ${\rm curt}_{n\tau}=(Q_{n\tau})_-$.
* **T2, shares of gross generation:** additionally an excess $c_{n\tau}\ge0$, with
  $p_i=\sigma_i\big((Q)_++c\big)$ and ${\rm curt}=(Q)_-+c$.

In both, $\sum_ip_i-{\rm curt}=Q$. **The balance holds identically.**

**Derived capacities** (dispatchables, $\bar g$ = `p_max_pu`, brownfield $k^0$):
$k_{i,n}=\max\big(k^0_{i,n},\max_\tau p_{i,n\tau}/\bar g_i\big)$.
Storage: $k_s=\max(\max p^+,\max p^-,\max e/h)$, or the range (§4).
Extendable lines: $s_e=\max(s^0_e,\max|f_e|)$.

**Target** (collapsed, $\rho_0$/$\rho^{\rm pf}$):
$$
\Phi=\beta_k\sum_ic_i(k_i-k_i^0)+\beta_v\Delta t\sum_{n,\tau}\Big[\sum_ic_ip_{i,n\tau}+c_{\rm curt}{\rm curt}_{n\tau}\Big]+\text{kinks}-\log J .
$$

**What holds by construction:** the balance, all capacity constraints of
extendable units, $e\le E$ (when $E$ is derived), non-negativity of dispatch, and the sign
logic (a negative requirement is curtailed).

**What remains as kinks or walls:**
* line limits of fixed lines;
* $e\ge0$;
* the fixed-fleet caps $\sigma_iQ\le\bar g_ik_i^{\rm fix}$ (nuclear, run-of-river, existing
  hydro): walls that couple transport and shares at every step;
* potential caps $p_i\le\bar g_i\bar k_i$;
* $Q=0$ (the curtailment boundary);
* the max of every capex term;
* ramping and UC walls on $\sigma Q$.

### 8.2 Reference measure and implied entropy

The physical target is $d\theta\,du\,de\prod_{n\tau}\big[\prod_idp_i\,d{\rm curt}\,\delta(\sum p-{\rm curt}-Q)\big]$ (§1).

* **T2 is an exact reparametrization of it.** At fixed $Q$,
  $(\sigma,c)\mapsto(p,{\rm curt})$ has Jacobian
  $$J_{\rm T2}=\prod_{n,\tau}\big((Q_{n\tau})_++c_{n\tau}\big)^{m_n-1}$$
  ($p=\sigma P$ with $P=Q_++c$ gives $dp=P^{m-1}dP\,d\sigma$). The term is local: one log per node and step.
  The variable $c$ is the simultaneous generation and curtailment, confined with slope
  $c_i+c_{\rm curt}$ (≈ 8 MW for gas, wide only where every local technology is
  nearly free).
* **T1 with the physical reference is ill-defined.** At $Q<0$ the dispatch is a
  point (0-dimensional), while at $Q>0$ it has $(m-1)$-volume $Q^{m-1}$, in different
  units. $J=Q_+^{m-1}$ gives the surplus stratum zero weight, and any finite weight
  $w^{m-1}$ [MW$^{m-1}$] is an arbitrary constant, like the UC atom of §3.3.
* **T1 with uniform shares** ($d\sigma$, normalized Dirichlet(1)) is continuous and
  dimensionless. Relative to the physical target it carries $\prod Q_+^{-(m_n-1)}$.
  Implied entropy: confined shares have σ-width $w/Q$, so
  $S\supset-\sum_{n\tau}(m_n-1)\log Q_{n\tau}$. This acts on the **transport**, at
  full operational strength: a force $(m-1)/Q$ per MW, i.e. an effective price
  shift of $(m-1)/(\beta_v\Delta tQ)$ ≈ 2 €/MWh at $Q=1$ GW and $m=3$, favouring
  imports into nodes with many technologies.
* **Capacities under T1.** The capacities see this entropy only at their argmax
  steps ($k_i=\sigma_iQ$ at the peak), so the bias on capacity is $O(1/k)$ per unit, not
  $n_{\rm conf}/k$. That is much milder than utilization coordinates.
* **Defensibility.** Uniform shares are the "decision fraction" view. They are not
  invariant under adding a technology ($m$ changes the exponent at every node and step)
  and they penalize large requirements. **Recommended: T2 with $J_{\rm T2}$.**
  T1 is the limit with $c$ removed, which changes the target by dropping the
  generate-and-curtail direction, and with an explicit choice of the surplus weight.
* **Exact line sampling.** $J_{\rm T2}$ is constant along share moves. Along moves
  that change $P$ it adds $(m-1)\log P$ (Gamma-type pieces: slice or
  incomplete-gamma inversion).

### 8.3 Renewables: given first

If renewables were shares, $k_g=\max_\tau\sigma_{g\tau}Q_\tau/A_{g\tau}$. This blows up
where $A_\tau\to0$, and a curtailment-free state needs $\sigma_\tau Q_\tau=A_\tau k$ at
**every** τ. All $T$ maxima are then tied, and the near-optimal set is a curve in a
$T$-dimensional share space: the worst possible coordinates. So renewables are
**given first** (R1). $k_{\rm ren}$ is primary, its output $Ak$ is subtracted in
$Q$, and the shares split the residual.

A move of $k_{\rm ren}$ at fixed (transport, σ, c) lowers $Q_{n\tau}$ by $A_\tau\delta$ at
all τ, and **the local dispatch follows pro rata**. The compensation is built in (it
is v2 §5.1's shear "pro rata to dispatchable capacity"). The derived dispatchable
capacities at the residual peaks follow too. **Wind-for-gas substitution is one
coordinate move**, O(T·m) at one node. In utilization coordinates it needs
`kscale`, ω shifts of gas at all steps and a gas capacity move. At $\rho_1$ it
still needs the $\hat b$/mean-force acceptance of §0.4.

### 8.4 Cost structure and locality

| move | touches | cost of ΔΦ | notes |
|---|---|---|---|
| share edge $\sigma_i\pm t$ at $(n,\tau)$ | one node and step | O(1) | opex linear in $t$; capex piecewise linear with at most 2 extra breakpoints (top-two cache of $\max_\tau p_{i,n\tau}$) |
| $\theta_n$ at τ | $Q$ at $n$ and its neighbours; incident flows | O(deg·m) | dispatch follows pro rata; breakpoints at line limits, $Q=0$, argmax; **no $L^+$ solve** |
| link flow $u_\ell$ at τ | $Q$ at its buses | O(m·buses) | multi-output links give a chain |
| $e_s$ at instant τ | $Q_n$ at τ−1 and τ | O(m) | plus storage maxima (top-two or range cache) |
| plateau $e_s$ on $[\tau_1,\tau_2]$ | $Q_n$ at τ₁−1 and τ₂ | O(m + log T) | seasonal storage |
| $k_{\rm ren}$ | $Q_n$ at all τ | O(T·m) | recompute the dispatchable maxima at $n$ |
| weather at $(n,\tau)$ | $Q_{n\tau}$ | O(m) | dispatch follows (built-in shear) |

**Parallelism.** Steps are independent except for (i) storage (two-step coupling,
parity colours) and (ii) the maxima. For the maxima, use the auxiliary-capacity
augmentation of §3.2 for parallel classes and the collapsed form in sequential
moves. With canonical CO₂ (§6) nothing global remains.

**At $\rho_1$.** Keep $k_i$ explicit for dispatchables, with the regularized wall
$c_{\rm viol}(\sigma_iQ-\bar g_ik_i)_+$, so that the score identity gives the mean force
$\partial_kF=\mathbb E[\partial_kC]$ (§0.4). The transport and share coordinates
are unchanged. Only the capex term is replaced by walls at the current $k$.

### 8.5 Hierarchy

* **Space (graph-Haar).** The cell requirement
  $Q_C=\sum_{n\in C}Q_n=d_C-R_C+(\text{net export across the cuts of }C)+\text{storage}_C$
  does not depend on intra-cell flows. The coarse model is therefore a transport model
  on cells:
  * coarse transport = cut transfers $F_{A|B}=\sum_{e\in{\rm cut}(A,B)}f_e$, driven by the Haar θ-coefficient of the contrast A|B (Galerkin Laplacian $P^\top LP$);
  * coarse shares $\sigma_{C,i}=\sum_{n\in C}p_{i,n}/Q_C$;
  * coarse line limits: $|F_{A|B}|\le\sum_{\rm cut}s_e$, an outer bound, optimistic.

  Details are intra-cell transport (θ details) and intra-cell redistribution of each
  technology's output at fixed cell totals. Define the hierarchy on dispatch
  *energies*, which are extensive and restrict by sums, and derive shares at each
  level.
* **Time (blocks and hats).** Coarse SoC is decimated at block boundaries:
  inter-block energy transport. Details are hierarchical surpluses: intra-block
  shifts. Coarse shares are energy-weighted block shares.
* **Capacities across levels.** $\max_\tau$ of block means $\le$ fine maximum, and
  $\max$ of a cell sum $\le$ sum of node maxima, so coarse capex is optimistic. Correct
  it with a peak factor per block or cell. It is a function of $x$ (intra-block
  variation of $Q$) and therefore cheap. Add the fiber entropy of §7.3 as well.

**Principal flow patterns** (Hofmann, Schäfer, Brown, Hörsch, Schramm, Greiner,
EPL 124, 18005 (2018), arXiv:1807.07771). For DC flows f = H p with the PTDF
H = Ω Kᵀ B⁺, the flow covariance Σ^f = H Σ^p Hᵀ has rank N − 1 (the L − N + 1
cycle directions carry no variance), and in a 1024-node renewable European
model about 8 principal flow patterns ρ^f_k = H v_k/‖H v_k‖ (v_k eigenvectors of
Σ^p HᵀH) capture 95 % of the flow variance at every network size, while the
number of injection components needed grows to ≈ 76; the flow patterns align
with low Laplacian eigenmodes. For the transport-first formulation this
suggests the coarse transport level: a handful of flow patterns (or cut
transfers between few large cells), learned from the weather data through H,
as the level-0 transport coordinates, with graph-Haar θ details below. The
paper's balancing rule (mismatch shared in proportion to the positive mismatch
or the mean load) is the pro-rata compensation of the T2 moves. The paper has
no line limits, storage or optimization.

### 8.6 Mixing

**Removed compared with the current formulation:**
* the 0.35 MW steep balance direction: the balance holds identically, and shedding
  is a share with reduced-cost slope $c_{\rm shed}-c_i$;
* same-node pairs, which become share edge moves;
* cross-node pairs and re-routing, which become θ moves with local pro-rata compensation;
* `cellpair`, which becomes a coarse share or cut-transfer move;
* the storage pair `spair`, which becomes an SoC coordinate move;
* the $k\omega$ ridge and `kscale`: dispatchable $k$ is derived;
* the weather–dispatch blocking of v2 §5.1(4), and the need for compensation in renewable $k$ moves;
* the per-step $\log(Ak)$ and $\log k$ Jacobians.

**Retained or new:**
* **Max couplings** for capex, including peak plateaus. These need peak-set moves
  (§3.2), choosing the set from $x$.
* **Pro-rata, not merit-order, compensation.** At a node with an interior mix, a transport
  move is compensated at the average cost, not the marginal cost. Near-optimal mixes
  are mostly vertices, so this is usually harmless. Where the marginal "technology" is
  import, the move belongs to the neighbour, through transport.
* **The kink at $Q=0$.**
* **Fixed-fleet walls** that couple transport and shares.
* **KVL.** With θ primary, KVL is exact and local. With flows primary (tree plus
  cycles), KVL fixes the cycle flows of AC (back to dense PTDF), so flows should be
  primary only for DC links, links and pipelines.
* **At $\rho_1$:** the capacity pinning of §0.4 for explicit $k$. It is not removed, only
  made milder: since transport and shares are non-centered with respect to $Q$, $v$
  re-equilibrates faster after a $k$ or $x$ move.

### 8.7 Comparison

| | current (ω, priced balance) | physical + price-setting frame (B1, R1, S1) | T1 (shares of $Q$, uniform) | T2 (shares of gross, $J_{\rm T2}$) |
|---|---|---|---|---|
| coordinates per step (model) | 567 (LP storage) / 333 (SoC) | 234 (39 θ + 8 u + 70 dispatchable + 117 e; $r$ replaces $j$) | 39 θ + 8 u + 117 e + Σ(m−1) ≈ 234 | T1 + $N$ ≈ 274 |
| balance | priced, kink per $(n,\tau)$ | exact on hyperplane; kink at bounds of $j$ | identical | identical |
| dispatchable $k$ | explicit, ridge | explicit or collapsed | derived (max) | derived (max) |
| renewables | ω, log(Ak) Jacobians | negative load | given first | given first |
| entropy, $k$-marginal | $-n_{\rm conf}\log k$ (E1 violated) | physical (E1–E3) | mild, at argmax only | physical (exact) |
| entropy, transport/weather | $-\sum\log A_\tau$ (E2 violated) | clean if $j$ has constant η | $-(m-1)\log Q$ per node and step | clean |
| steepest coordinate | 0.35 MW (shed) | reduced cost; shed when $j$ at a bound | reduced cost | reduced cost |
| pair/re-route moves | separate, $L^+$ solves | coordinates; frame from $(k,x)$ | coordinates; frame is part of the state | same as T1 |
| weather moves | blocked without shear | shear needed | built in | built in |
| renewable $k$ moves | kscale + ω | compensated by $j$ | pro rata, built in | pro rata, built in |
| locality / parallel | local, per step | local, per step | local; max coupling | local; max coupling |
| implementation | exists | moderate (frame, R1) | new state layout | new state layout + $c$ |

**Assessment.** T2 (with R1, S1 and θ for AC) is the natural endpoint of the
other recommendations:
* the balance holds by construction;
* the entropy is physical;
* dispatchable capacities are collapsed;
* the compensation frame is part of the state, so there is no reversibility constraint
  on choosing it (unlike B1).

Its costs are a new state layout, the max couplings and the curtailment boundary. Its
building blocks R1, S1 and the physical measure pay off on their own. **Rank: 3,
the target architecture after recommendations 1–2, which it contains.**

---

## 9. Where the LP form biases the sampled ensemble

1. **Normalized coordinates** (utilization, angles for extendable lines, uniform shares):
   $-\sum\log({\rm scale})$ per confined coordinate. This gives extensive entropic forces on capacities
   (§0.1: 10 % of gas capex at $\rho_1$ daily, 2.5× hourly, 104× at $\rho_0$) and per-step weather terms
   that $\hat b$ cannot represent.
2. **Redundant variables:** storage 3T, curtailment versus dump, the curtailment split among
   co-located renewables, brownfield and new vintages, antiparallel lossy links,
   store with independent charger and discharger links, spill below the power limit, "energy
   may be lost" inequalities. Each carries a volume ∝ capacity per step and is a slow
   soft direction. The observed drifts in dump and storage cycling are two of these.
3. **Capacity as a separate upper bound:** the $k$–$p$ ridge, and at $\rho_1$ the
   pinning of $k$ to the peak within 0.6 MW.
4. **Per-year hard caps** (CO₂), which become implicit weather-dependent prices.
5. **Piecewise-linear approximations** (losses, efficiency curves): artificial kinks.
6. **Per-snapshot Lebesgue under snapshot weights or aggregation:** coarse or weighted
   snapshots are effectively colder (§7.3).
7. **ρ₀ with two temperatures** weighs opex $10^3\times$ capex. Any statement about
   capacities from $\rho_0$ or $\rho^{\rm pf}$ needs $\beta_k=\beta_v$, or the $\rho_1$ structure.

---

## 10. Points where the current notes need correction

1. **v2 §2.6 ("use the utilization reference").** This conclusion is wrong. Its "polynomial boost
   $k^{(\beta_k/\beta_v)\bar T}$" under the physical measure assumes that every dispatch coordinate is
   free. Near the optimum they are confined, and confined coordinates have
   $k$-independent MW widths. The utilization reference gives the opposite and larger
   effect, $k^{-n_{\rm conf}}$, plus weather Jacobians. Only genuinely flat faces grow
   with $k$ under the physical measure, and those should be removed by parametrization (E3).
2. **physical_formulation §2 (45×).** This is the $\rho_0$ number. At $\rho_1$ the force
   carries $\beta_k/\beta_v$: about 10 % of capex at daily resolution (with the model's
   $c=4.4\cdot10^4$), but 2.5× at hourly resolution with $\beta_v$ unchanged. It still
   matters, and it depends on resolution unless $\beta_v\propto n_v$.
3. **physical_formulation §3 (collapse, idle margin 250 MW).** This holds for $\rho_0$, whose
   capacity marginal overweights opex $10^3$-fold. At $\rho_1$ the margin is about $1/(\beta_vc)$ ≈ 0.6 MW
   and $k$ is pinned to the peak. The collapse needs $\partial_k\log Z$ to relative accuracy
   $10^{-3}$. `kscale` and $k\mid p$ moves do not mix capacities at $\rho_1$. Transported
   (peak-set) moves with mean-force or $\hat b$ acceptance are needed.
4. **physical_formulation §2 (renewables keep the utilization measure).** This keeps
   $-\sum_\tau\log(A_\tau k)$ in $\log Z$ (E1, E2). Renewables as negative load (R1)
   avoids the dilemma: no dispatch coordinate, no Jacobian.
5. **physical_formulation §5 ("the choice may depend on $(k,x)$ without correction").**
   This is right for unit coefficients only. With efficiencies the Jacobian is
   $1/|\eta_j|$, and with $x$-dependent η it enters $S$. Also, the current priced balance
   already is the δ-balance with (shed, dump) as the dependent unit (§1, B0). A frame change
   is a reparametrization and does not remove the $N\cdot T$ kinks; it only aligns them.
6. **physical_formulation §4 / code (`soc_power`).** Spill beyond the power limit
   is free destruction of stored energy for batteries and H₂. Price it.
7. **Status table.** It is stale. The SoC form exists (`storage = "soc"`). The default is still
   `measure = :utilization`, `storage = :lp`.
8. **mathematical_formulation §1.3, κ₀ = 1 MW.** This is needed only because of utilization
   coordinates (§7.2).

---

## 11. Ranked recommendations

| # | what | expected benefit | risks |
|---|---|---|---|
| 1 | **Renewables as negative load** (R1). Drop the ω fields of onwind, offwind and solar; nodal curtailment = $r_-$ priced at $c_{\rm dump}$; $k_{\rm ren}$ moves with B1 compensation | −17 % of coordinates (−30 % with SoC); removes the curtailment-split flat directions and the $\log(Ak)$ Jacobians (E1, E2), so $\hat b$ need not learn $\sum\log A_\tau$; simpler score identity | loses per-unit curtailment costs; $k_{\rm ren}$ moves cost O(T) and need compensation |
| 2 | **Physical measure and SoC as defaults** (`measure = physical`, `storage = soc`), MWh for SoC, injection MW for links; drop κ₀; price spill for units without inflow; merge co-located vintages; signed lossy links; calibrate $\beta_v\propto n_{\rm conf}$ | removes the $k^{-n_{\rm conf}}$ force (10 % of gas capex daily, 2.5× hourly) and its resolution dependence; −41 % of coordinates; removes the storage-cycling drift | zero-capacity units (handled by the scale-free line sampler); recalibrating $\beta_v$ changes the slack |
| 3 | **Transport-first formulation T2** (§8): θ, u, e and $k_{\rm ren}$ primary; shares of gross generation with $J_{\rm T2}$; derived dispatchable capacities (collapsed for $\rho_0$/$\rho^{\rm pf}$, regularized walls plus mean force at $\rho_1$) | balance by construction; pair, re-route and cell-pair moves become coordinate moves with no $L^+$ solves; built-in weather shear and pro-rata compensation; technology substitution in one move | new state layout; max couplings and peak plateaus; kink at $Q=0$; fixed-fleet walls; not a remedy for the $\rho_1$ pinning |
| 4 | **Transported capacity moves** (peak set $S$ from $(k,x)$, compensated), with $\hat b$ or integrated-mean-force acceptance; interleave with `kscale` (ASIS) | the only route to capacity mixing at $\rho_1$ | needs $\partial_k\log Z$; the bias accuracy is the bottleneck (alternatives §0) |
| 5 | **Price-setting frame B1** if (3) is postponed: $j(n,\tau)$ from the merit order at the coarse-LP prices | single-coordinate conditionals at reduced-cost widths; local θ moves; parallel sweeps of pair moves | frame quality; kinks return when $j$ is at a bound; Jacobian with efficiencies |
| 6 | **Canonical CO₂ price** tuned to the mean cap; no priced dump on stock buses | local and parallel; well-defined multi-year economics | a different ensemble from the per-year cap (state it explicitly) |
| 7 | **Level consistency**: $\beta_\ell=\beta_v/r_\ell$ or fiber entropies, $(n_c-1)\log K_c$ for coarse capacities, peak factors for coarse capex | coarse levels with correct excess and capacity marginals, hence better MLDA and MLMC level differences | approximations of the fiber volume near active sets |
| 8 | **Exact quadratic losses and soft thermal limits** at fine levels; extendable lines via max (brownfield hinge) with optional impedance scaling and the injection measure | breaks path degeneracy; replaces walls by smooth terms; better physics than the LP | erf pieces in the line sampler; a different target |
| 9 | **Fleet UC**, ramping as a penalty at sub-daily levels; elastic demand (B3) | continuous, aggregable nonconvex physics; smoother shedding tail | multimodality; new parameters |

Recommendations 1 and 2 are small changes in `energy.jl`/`local.jl` and the
test for each is already defined: the experiments E1/R2 of alternatives.md,
with excess ratio and drift of dump and storage cycling. Recommendation 3 should be
prototyped on the toy and on 40 nodes × daily against the node kernel
(overnight §4) before it replaces the state layout.
