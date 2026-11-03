# Near-optimal sampling of energy system decisions under weather uncertainty — mathematical setup

Working notes. The document covers, in order:

- the abstract target measures (§1);
- the energy system model, with hard physical constraints and a priced balance relaxation (§2);
- the multilevel Gaussian-mixture weather model (§3);
- the multilevel parametrization of the joint model, from overcomplete generating systems to non-redundant splittings, with nearly mean-free fine-level priors as the middle ground (§4);
- multilevel delayed acceptance (MLDA) on that hierarchy (§5);
- the one normalization that has to be learned (§6).

Spatial levels are kept abstract throughout.

**Conventions.**

- Time fields live on instants ($\mathcal V_T^0$) or steps ($\mathcal V_T^1$) of the time cycle. Pairings in time are $\Delta t$-weighted, so $\langle c,p\rangle$ is an energy cost.
- Spatial operators act on the first tensor factor, temporal ones on the second.
- $(u)_\pm=\max(\pm u,0)$ elementwise, and $\langle\mathbf 1,\cdot\rangle$ sums.

---

## 1. Abstract setting

### 1.1 Ingredients

- **First stage.** $k\in\mathcal K$: investments, fixed before the weather is known. $\mathcal K$ is a box.
- **Second stage.** $v\in\mathcal V$: operation, chosen given the weather.
- **Weather.** $x\in\mathcal X$, with probability measure $\rho^x$ (§3).
- **Hard constraints.** $\Omega(k)\subset\mathcal V$, convex and full-dimensional. The coordinates are chosen so that $\Omega$ does not depend on $x$ (§2.5).
- **Costs.** Annualized capex $\langle c_k,k\rangle$, and operational cost $C_x(k,v)$, which is convex in $v$.
- **Reference measures.** Lebesgue $dk$ on $\mathcal K$ and $dv$ on $\mathcal V$. The choice of coordinates for $v$ is a modelling decision (§2.6). §4.3 adds a multiscale Gaussian alternative.
- **Temperatures.** $\beta_k$ for investments and $\beta_v$ for operation.

### 1.2 Operational Gibbs measure and free energy

$$Z(x,k)=\int_{\Omega(k)}e^{-\beta_vC_x(k,v)}\,dv,\qquad F(x,k)=-\beta_v^{-1}\log Z(x,k),\qquad \pi(dv\mid x,k)=\frac{e^{-\beta_vC_x(k,v)}}{Z(x,k)}\,\mathbb 1_{\Omega(k)}(v)\,dv .$$

### 1.3 Investment measure: expected cost over weather

$$\rho_K(dk)\ \propto\ \exp\!\Big(-\beta_k\big[\langle c_k,k\rangle+\mathbb E_{x\sim\rho^x}F(x,k)\big]\Big)\,\mathbb 1_{\mathcal K}(k)\,dk .$$

- As $\beta_v\to\infty$, $F(x,k)\to\min_vC_x(k,v)$.
- As then $\beta_k\to\infty$, $\rho_K$ concentrates on the solutions of the two-stage stochastic program
$$\min_k\ \langle c_k,k\rangle+\mathbb E_x\min_{v\in\Omega(k)}C_x(k,v).$$

### 1.4 Joint targets

**α = 1 (the target).**
$$\rho_1(dk,dx,dv)=\rho_K(dk)\,\rho^x(dx)\,\pi(dv\mid x,k).$$
- The weather is independent of the decisions and distributed as $\rho^x$.
- Operation is Gibbs given weather and investments.
- Investments follow the expected-cost measure.

**α = 0 (for exposition).**
$$\rho_0(dk,dx,dv)\ \propto\ e^{-\beta_k\langle c_k,k\rangle}\,\rho^x(dx)\,e^{-\beta_vC_x(k,v)}\,\mathbb 1_{\Omega(k)}(v)\,dk\,dv .$$
- Its marginals are $x\mid k\propto\rho^xZ(x,k)$, i.e. weather tilted toward cheap configurations.
- For $\beta_k=\beta_v$, $k\propto e^{-\beta_k\langle c_k,k\rangle}\mathbb E_xZ(x,k)$. This is an annealed average: the planner effectively benefits from favorable weather. α = 1 is the corresponding quenched average.

**Relation between the two.** With
$$b(x,k)=\log Z(x,k)-\tfrac{\beta_k}{\beta_v}\,\Lambda(k),\qquad \Lambda(k)=\mathbb E_{x'\sim\rho^x}\log Z(x',k),$$
we have
$$\rho_1\ \propto\ e^{-\beta_k\langle c_k,k\rangle}\,\rho^x(dx)\,e^{-\beta_vC_x(k,v)}\,e^{-b(x,k)}\,\mathbb 1_{\Omega(k)}(v)\,dk\,dv .$$
To check this, integrate out $v$; that produces $Z(x,k)$, which cancels against $e^{-\log Z}$ and leaves $e^{(\beta_k/\beta_v)\Lambda(k)}\rho^x(dx)=e^{-\beta_k\mathbb E F}\rho^x(dx)$. So **α = 1 is α = 0 with a weather–investment bias $e^{-b}$**. Learning $b$ is the only normalization problem (§6).

### 1.5 Calibrating the temperatures

Near a nondegenerate optimum, every free coordinate carries one unit of $\beta\times$cost (equipartition). Hence
$$\mathbb E[\text{operational excess}]\approx n_v^\ast/\beta_v,\qquad \mathbb E[\text{investment excess}]\approx n_k/\beta_k,$$
where $n_v^\ast$ and $n_k$ count the operational and investment coordinates that are not pinned at the optimum. For target slacks $\varepsilon_v,\varepsilon_k$, choose $\beta_v\approx n_v^\ast/\varepsilon_v$ and $\beta_k\approx n_k/\varepsilon_k$. Typically $n_v^\ast\gg n_k$, so $\beta_v\gg\beta_k$.

### 1.6 Perfect-foresight variant

Treating $k$ as second stage gives $\rho_1^{\rm pf}=\rho^x(dx)\,\pi^{\rm pf}(dk,dv\mid x)$, with bias $\log Z^{\rm pf}(x)$. Its free energy $F^{\rm pf}(x)$ ranks weather by how expensive it is *even if planned for*, which is the Dunkelflaute question. Everything below applies to it with $k$ moved into $v$.

---

## 2. The energy system model

### 2.1 Spaces and operators

**Spaces.** $\mathcal V_N$ (nodes), $\mathcal V_E$ (lines), $\mathcal V_G$ (generators), $\mathcal V_S$ (storage), $\mathcal V_C$ (controllable links).

**Operators.**

- Incidence $B:\mathcal V_E\to\mathcal V_N$, susceptances $W=\mathrm{diag}(1/x_e)$, Laplacian $L=BWB^\top$.
- Component incidences $G$, $H$, $B_C$ into $\mathcal V_N$.
- Temporal derivative $\partial_t:\mathcal V_T^0\to\mathcal V_T^1$ (cyclic).

### 2.2 Variables

| | symbol | space |
|---|---|---|
| capacities | $k=(k_G,k_E,k_C,k_S^E,k_S^P)$ | $\mathcal V_G\oplus\mathcal V_E\oplus\mathcal V_C\oplus\mathcal V_S^2$ |
| potentials | $\theta$ | $\mathcal V_N\otimes\mathcal V_T^1$, modulo constants per step |
| utilizations | $\omega_G$, $\omega_C$, $\omega_\pm$ | step fields on $\mathcal V_G$, $\mathcal V_C$, $\mathcal V_S$ |
| state of charge | $\omega_e$ | $\mathcal V_S\otimes\mathcal V_T^0$ |
| weather | $x=(d,a)$: load, availability | $(\mathcal V_N\oplus\mathcal V_G)\otimes\mathcal V_T^1$ |

Physical quantities are derived from these:
$$p=\bar a\odot k_G\odot\omega_G,\quad u=k_C\odot\omega_C,\quad e=k_S^E\odot\omega_e,\quad p^\pm=k_S^P\odot\omega_\pm,\quad f=WB^\top\theta,$$
with capacities broadcast in time. Availability enters through the data map $\bar a=\min(\max(a,0),1)$, and load through $\bar d=(d)_+$.

### 2.3 Hard constraints: physical capability

$\Omega(k)$ is defined by:

- **(H1) Generators.** $0\le\omega_G\le1$: a generator never produces more than its available capacity.
- **(H2) Storage.** $0\le\omega_e\le1$ and $0\le\omega_\pm\le1$, together with
$$k_S^E\odot\partial_t\omega_e\ \le\ k_S^P\odot(\eta^+\omega_+-\omega_-/\eta^-).$$
  Energy may be lost but never created.
- **(H3) Network.** $|WB^\top\theta|\le k_E$ (thermal ratings) and $-1\le\omega_C\le1$ (links).

Investment bounds are $0\le k\le\bar k$, from technical potentials. KVL is built into $f=WB^\top\theta$, and the storage relation is an inequality. **No equality constraints remain**, so $\Omega(k)$ is full-dimensional.

### 2.4 Priced relaxation: load shedding

The nodal deficit (unserved demand) is
$$r=\bar d+L\theta+B_Cu+H(p^+-p^-)-Gp\ \in\ \mathcal V_N\otimes\mathcal V_T^1 .$$
Balance is not imposed; imbalance is priced:
$$C_x(k,v)=\langle c_G,p\rangle+\langle c_S,p^++p^-\rangle+c_{\rm shed}\langle\mathbf 1,(r)_+\rangle+c_{\rm dump}\langle\mathbf 1,(r)_-\rangle .$$

- $c_{\rm shed}$ is the value of lost load. This is equivalent to backstop load-shedding generators at every node.
- $c_{\rm dump}\in[0,\infty)$ prices surplus dumping. The redundancy between dumping and curtailment (lowering $\omega_G$) is harmless.
- These are the relu penalties from the original notes, but with physical slopes. As $c_{\rm shed},c_{\rm dump}\to\infty$, the exact balance $r=0$ is recovered.
- Keep $c_{\rm dump}$ finite. Otherwise $\{r\ge0\}$ becomes a hard constraint that depends on $x$, and the score identities of §6 pick up boundary terms.

### 2.5 Where the weather enters

Weather enters **only the cost**: $\bar d$ through $r$, and $\bar a$ through $p$. $\Omega(k)$ is independent of $x$. In particular, in utilization coordinates (H1) is the fixed box $[0,1]$.

### 2.6 Choice of reference measure

Lebesgue measure in utilization coordinates and Lebesgue measure in physical units ($p,u,e,p^\pm$) differ by a Jacobian
$$J(x,k)=\prod_{g,t:\,\bar a_{g,t}>0}\bar a_{g,t}k_g\cdot\prod_c k_c^{T}\cdot\prod_s(k^E_s)^{T}(k^P_s)^{2T}$$
that does not depend on $v$. Consequences:

- **Same operational measure.** $\pi(\cdot\mid x,k)$ is identical in both coordinate systems. Only $Z$ changes, by the factor $J$.
- **Same weather marginal.** Under $\rho_1$ it is $\rho^x$ in both cases, because $b$ absorbs any factor in $Z$.
- **Different investment marginal.** In physical units
$$\rho_K^{\rm phys}\ \propto\ \rho_K\cdot\exp\!\big(\tfrac{\beta_k}{\beta_v}\,\mathbb E_x\log J(x,k)\big)\ \propto\ \rho_K\cdot\prod_g k_g^{(\beta_k/\beta_v)\bar T_g}\cdots,$$
  where $\bar T_g$ is the expected number of steps with $\bar a>0$. This is a polynomial boost of high degree towards large capacities: the extensive dispatch entropy in its sharpest form. **Use the utilization reference.**
- **Price of that choice.** In physical units, $\rho_K^{\rm phys}$ is log-concave by Prékopa–Leindler, because the constraints are jointly linear in $(k,p)$. In utilization coordinates, $\rho_K$ need not be log-concave.
- **Multiscale Gaussian reference (option).** An overcomplete parametrization with Gaussian level priors, left uncorrected (§4.3, κ = 0), replaces Lebesgue measure by a projective multiscale Gaussian reference on the fields it covers. This also removes the extensive-entropy issue in those directions.
- **Reference coordinates are not sampling coordinates.** The reference is fixed in utilization coordinates, but a sampler may move in any coordinates as long as the density carries the Jacobian. Two cases:
  - **Investment moves.** In physical coordinates $(p,k)$, with $-\log J(x,k)$ added to the log-density, the constraint $p\le\bar a\,k$ is linear. The curved $k\cdot\omega$ ridge of utilization coordinates then becomes a straight direction: change $k$ at fixed $p$.
  - **Weather moves.** Utilization coordinates keep $\Omega$ independent of $x$, which physical coordinates do not.

  Use whichever coordinates suit each block move; they describe the same measure.

Two directions are left without a capacity-normalized coordinate: line ratings (through the polytope in $\theta$-space) and the storage power-to-energy ratio in (H2).

---

## 3. Weather model: multilevel Gaussian mixture

### 3.1 Anomalies and restriction

- **Anomalies.** $x=\bar x+x'$, where $\bar x$ is the climatological hour-of-year mean over the 60 years.
- **Weather restriction.** $R^w_\ell$ must equal the restriction the energy model applies to its data, so that coarse weather *is* coarse model data:
  - load: like injections, i.e. block sums or averages in time and the injection restriction in space;
  - availability: block averages in time, and the generator restriction $R_G$ in space (fixed weights).
- **Levels.** The weather uses the same generating system as the decisions (§4.1).

### 3.2 Generative model

**Coarsest level: regime-switching linear-Gaussian model in coarse time (step $\tau$).**

- A regime Markov chain $m_\tau$ on $K$ states.
- Coarse weather evolves as
$$z^w_0(\tau)\mid m_\tau,z^w_0(\tau-1)\ \sim\ \mathcal N\big(\Phi_{m_\tau}z^w_0(\tau-1)+\mu_{m_\tau}(s_\tau),\ \Sigma_{m_\tau}\big),$$
  with $s_\tau$ the calendar season.
- Conditional on the regime path $m=(m_\tau)$, everything is Gaussian. The mixture components are weather regimes.
- An optional per-year latent factor captures interannual variability.

**Finer levels: conditional linear-Gaussian details (non-centered).**
$$\xi^w_\ell=A^{m}_\ell z^w_\ell+c^{m}_\ell+(Q^{m}_\ell)^{1/2}w_\ell,\qquad w_\ell\sim\mathcal N(0,I)\ \text{independent of everything coarser}.$$
This is a multiresolution (multiscale autoregressive) Gaussian model on the level tree, mixed over regime paths. It corresponds to the non-redundant limit of §4 (ε = 0). §3.5 gives the overcomplete variant.

### 3.3 Properties

- **Projective by construction.** The level-$\ell$ weather state is $x_\ell=(m,z^w_0,w_0,\dots,w_{\ell-1})$, with density
$$p^w_\ell(x_\ell)=p(m)\,p(z^w_0\mid m)\prod_{j<\ell}\mathcal N(w_j;0,I).$$
  This is the exact marginal of the finest model, so the weather part needs no renormalization.
- **Local.** $Q_\ell^m$ is block-diagonal over the detail blocks, and $A^m_\ell$ couples each block only to coarse variables in its neighborhood. Given the coarse level, detail blocks are independent.
- **Exactly samplable.** Expectations $\mathbb E_{x'\sim\rho^x}[\cdot]$ can be computed by direct sampling.

### 3.4 Fitting and validation

- **Coarsest level.** EM (or Gibbs) for the switching model on coarse coordinates. With daily coarse steps, 60 years give about 22,000 time steps.
- **Details.** Responsibility-weighted linear regression of the detail coordinates on their coarse parents, pooled over all blocks within a season.
- **Validation** on held-out years:
  - regime frequencies and persistence;
  - spatial and temporal correlations per level;
  - the frequency of low-wind, cold multi-day events.

### 3.5 Variant: nested random effects (overcomplete)

Conditionally on the regime path, represent the anomaly as a sum of independent level effects in the generating system of §4.1:
$$x'=\sum_{\ell}P_\ell u^w_\ell,\qquad u^w_\ell\mid m\ \sim\ \mathcal N\big(\mu^w_\ell(m),\ \sigma_\ell^2(m)(I-M_\ell)+\epsilon_\ell^2(m)M_\ell\big),\quad\text{independent across }\ell .$$

- **Interpretation.** Large-scale + country + area + node effects (and week + day + hour effects in time) form a nested random-effects model. The redundancy is harmless: for the weather, the coefficient prior *is* the model.
- **Temporal correlation within a level.** An AR(1) per coefficient can be added without breaking independence across levels.
- **Dependence on the coarse state.** Compared with §3.2, details depend on the coarse state only through the regime ($A^m_\ell=0$). Add the coupling back if validation shows that detail variance depends on the coarse state.
- **Fitting.** Variance components (REML or EM) per regime and season, pooled over blocks. The model is identifiable despite the redundancy because the level covariances have different structures.
- **Consistency with the model data.** With ε = 0, coarse weather equals coarse model data exactly; with ε > 0, approximately.

---

## 4. Multilevel parametrization of the joint model

All fields indexed by space or time are represented in one multilevel generating system: operation, investments per location, and weather anomalies. A single family of level priors interpolates between an overcomplete representation and a non-redundant splitting.

### 4.1 Generating system

For a field space $\mathcal Y$, choose levels $\ell=0,\dots,L$ with coefficient spaces $\mathcal U_\ell$ and synthesis maps $P_\ell:\mathcal U_\ell\to\mathcal Y$:
$$y=\Pi u=\sum_{\ell=0}^LP_\ell u_\ell,\qquad u=(u_0,\dots,u_L)\in\mathcal U=\textstyle\bigoplus_\ell\mathcal U_\ell .$$
Π is onto, and $\ker\Pi$ is the redundancy. Example levels:

| ℓ | space | time |
|---|---|---|
| 0 | large-scale modes (low Laplacian eigenvectors or geographic Fourier modes) | seasonal and synoptic harmonics |
| 1 | country indicators | week indicators |
| 2 | area indicators | day indicators |
| 3 | node identity | hour identity |

- **Space–time fields** use tensor products of spatial and temporal levels along a chosen chain.
- **Step fields** use indicators. **Instant fields** (storage levels) use hat functions, so that $\partial_t$ maps hats to indicators.
- **Restrictions.** Each level ℓ has a model of its own (§4.4), whose variables and data are obtained from fields by a restriction $R_\ell$. Restrictions are nested: $R_{\ell-1}=R^{\ell}_{\ell-1}R_\ell$.

### 4.2 Level priors: from overcomplete to non-redundant

Let $M_\ell$ be the orthogonal projector in $\mathcal U_\ell$ onto $(\ker R_{\ell-1}P_\ell)^\perp$, the part of level ℓ that the next coarser model can see. Then $(I-M_\ell)u_\ell$ is invisible to all coarser models, by nesting. Concretely:

- **Extensive fields** (dispatch, injections, load) are restricted by cell sums, so $(I-M_\ell)u_\ell$ has zero sum over each parent cell.
- **Instant fields** are restricted by decimation, so $(I-M_\ell)u_\ell$ vanishes at the parent instants (hierarchical surplus).
- **Potentials θ** are seen by the coarse model through the coarse balance $E_X^\top L\theta$, so $(I-M_\ell)u_\ell$ is $L$-orthogonal to the coarse potentials.
- **The first partition level** relative to the large-scale modes: $(I-M_1)u_1$ is orthogonal to those modes.

Take independent Gaussian priors per level:
$$q(u)=\prod_\ell\mathcal N(u_\ell;\mu_\ell,Q_\ell),\qquad Q_\ell=\sigma_\ell^2\,(I-M_\ell)+\epsilon_\ell^2\,M_\ell .$$

| | regime | consequence |
|---|---|---|
| $\epsilon_\ell=\sigma_\ell$ | isotropic overcomplete | level ℓ moves coarse aggregates as freely as details |
| $0<\epsilon_\ell\ll\sigma_\ell$ | **middle ground** | aggregates are carried mainly by coarser coefficients; redundant directions are proper but weakly excited |
| $\epsilon_\ell=0$ | non-redundant | Π is invertible on the support of $q$: the exact splitting of §4.6 |

### 4.3 Target in coefficient space

For a field target ρ (the operational, investment and weather parts of §1), define
$$\rho_U(u)\ \propto\ \rho(\Pi u)\ q(u)\ c(\Pi u)^{-\kappa},\qquad c(y)=\mathcal N\Big(y;\ \sum_\ell P_\ell\mu_\ell,\ C_Y\Big),\qquad C_Y=\sum_\ell P_\ell Q_\ell P_\ell^\top,\qquad\kappa\in\{0,1\}.$$

**Proposition.** $\Pi_\#\rho_U\propto\rho\cdot c^{\,1-\kappa}$.

*Proof.* On the fiber $\Pi^{-1}(y)$ the factor $\rho(\Pi u)\,c(\Pi u)^{-\kappa}$ is constant. The fiber integral of $q$ is the density of $\Pi u$ under $q$, which is $c(y)$ up to the constant Jacobian of Π.

- **κ = 1: exact.**
  - The fields follow exactly the target ρ, for any ε.
  - The correction needs $y^\top C_Y^{-1}y$. Coarse levels are low-rank and the node level is diagonal plus low-rank, so Woodbury applies.
  - $C_Y$ is invertible if every direction of $\mathcal Y$ is covered by some level with positive variance.
- **κ = 0: multiscale Gaussian reference.**
  - The field target becomes ρ times a projective Gaussian with BPX-type covariance $C_Y$.
  - This is a modelling choice. It replaces the Lebesgue reference and removes the extensive-entropy issue in the directions it covers.
  - The center $\sum_\ell P_\ell\mu_\ell$ should be meaningful (e.g. the coarsest-level optimum), not zero.
- **Weather.** The coefficient prior is the model (§3.5), so κ = 0 by definition and no correction arises.

### 4.4 States, level models and level increments

- **State at level ℓ.** $s_\ell=u_{\le\ell}$ for every field: weather together with its regime path, investments, and operation. The level-ℓ fields are the truncations $y_\ell=\sum_{j\le\ell}P_ju_j$.
- **Level model.**
$$\pi_\ell(s_\ell)\ \propto\ q_{\le\ell}(u_{\le\ell})\ c_\ell^{-\kappa}\ e^{-\beta_k\langle c_k,k_\ell\rangle}\ e^{-\beta_vC^{(\ell)}_{x_\ell}(k_\ell,v_\ell)}\ \mathbb 1_{\Omega_\ell}(k_\ell,v_\ell)\ e^{-\hat b_\ell-\hat F_\ell}.$$
  - $(k_\ell,v_\ell,x_\ell)=R_\ell y_\ell$.
  - $C^{(\ell)}$ and $\Omega_\ell$ are the energy model at the resolution of level ℓ: coarse network, coarse time, aggregated data.
  - $c_\ell$ is the Gaussian factor of the truncated system.
  - $\hat b_\ell$ is the learned bias (§6), and $\hat F_\ell$ an optional learned renormalization, playing the role of MLDA's adaptive error model.
  - Only $\pi_L$ must be exact.
- **Level increment.** Because $q$ factorizes over levels,
$$\pi_{\ell+1}(s_{\ell+1})=\pi_\ell(s_\ell)\,\mathcal N(u_{\ell+1};\mu_{\ell+1},Q_{\ell+1})\,e^{-\Delta_\ell(s_{\ell+1})},$$
$$\Delta_\ell=\beta_v\big[C^{(\ell+1)}-C^{(\ell)}\big]+\kappa\big[\log c_{\ell+1}-\log c_\ell\big]+\big[\hat b_{\ell+1}-\hat b_\ell\big]+\big[\hat F_{\ell+1}-\hat F_\ell\big]+\infty\cdot\mathbb 1\{\text{infeasible}\}+\text{const}.$$
- **Role of ε.** Level ℓ+1 changes what the level-ℓ model sees only through $M_{\ell+1}u_{\ell+1}$, whose prior scale is $\epsilon_{\ell+1}$.
  - The aggregate seen by the coarse model therefore differs from that of the fine state by $O(\epsilon_{\ell+1})$. This adds to the genuine resolution effects in $\Delta_\ell$ (penalties, data, constraints).
  - MLDA acceptance degrades smoothly as ε grows; ε = 0 removes this contribution.
- **Support condition.** MLDA needs $\pi_\ell>0$ at the coarse modes of every state with $\pi_{\ell+1}>0$; otherwise the fine chain can never accept.
  - For ε = 0 this follows from coarse feasibility of restrictions (§4.6).
  - For ε > 0, the truncated coarse fields of a fine-feasible state can violate coarse hard constraints.
  - So: **keep hard constraints only at the finest level and replace them by steep penalties at coarser levels**, or remove the box constraints altogether with the logit expansion (§4.5).

### 4.5 Bounded fields: logit-space expansion (option)

For utilizations, expand in logit space: $\omega=\sigma(\eta)$ with $\eta=\Pi u$.

- The boxes in (H1), the storage boxes in (H2) and the link boxes then hold at every level automatically. Only line ratings and the storage dynamics remain as constraints.
- With κ = 0, the reference on ω is a multiscale logit-normal. To recover the utilization reference of §2.6, include the Jacobian $\prod\sigma'(\eta)$ and take κ = 1.
- Near-optimal utilizations sit near 0 or 1, i.e. at large $|\eta|$. Set the level means from a coarse optimum and keep node-level variances broad.

### 4.6 The non-redundant limit (ε = 0)

With ε = 0 the hierarchy is an exact splitting, $y_{\ell+1}=E_\ell y_\ell+D_\ell\xi_\ell$, with $D_\ell$ spanning $P_{\ell+1}\,\mathrm{range}(I-M_{\ell+1})$.

**Time.**
- Step fields are block averages: indicators, with block-mean-free details.
- Instant fields are decimated with piecewise-linear interpolation: hats, with hierarchical surpluses.
- θ is block-averaged.
- **(A1) holds exactly for Ω.** The boxes are preserved by averaging and decimation. The storage inequality sums over a block, because capacities are constant in time. Line limits survive averaging by convexity.
- **Costs are only approximately consistent.**
  - For given $k$ and $\omega$, the coarse model uses $\bar a\,k\,\bar\omega$ in place of $k\,\overline{a\omega}$.
  - The data map does not commute with averaging: $\mathrm{clip}(\overline a)\ne\overline{\mathrm{clip}(a)}$.
  - By Jensen, coarse penalties underestimate fine ones, so coarse models are optimistic.
  - $\hat F_\ell$ and the acceptance step absorb these discrepancies.
- **Fibers can be empty**, which shows up as rejections.

**Space.**
- The θ splitting is $L$-orthogonal: $E_X^\top LD_X=0$.
- Component restrictions are compatible with the injection restriction: $E_X^\top G=G_cR_G$, and likewise for $H$ and $B_C$.
- Coarse components inject only into $(\mathrm{im}D_X)^\perp$.
- Merging generators with different availability profiles needs per-class capacities; otherwise (A1) holds only approximately.

---

## 5. MLDA on the joint model

### 5.1 Coarsest level: base sampler

Any $\pi_0$-invariant kernel is valid. A first implementation (§5.4) shows that gradient-based samplers such as reflective HMC and MALA fit this target poorly. The recommended default is a blocked scheme of exact or tuning-free moves, combined as a random-scan mixture so that the kernel stays reversible:

1. **Regimes.** Sample $m\mid u^w_0$ exactly by forward-filtering backward-sampling.
2. **Operation: exact coordinate and directional Gibbs.** For fixed $(k,x)$, the cost is piecewise linear along any line in operation space, in physical and in utilization coordinates alike. The constraints cut the line to an interval. So every line conditional is a piecewise-exponential density on an interval, which can be sampled exactly by inversion: no step size, no rejection, and no dependence on the scale of the near-optimal set. Useful directions:
   - **Single coordinates.** These are local; coordinates that share no cost term or constraint can be updated in parallel (graph colouring).
   - **Balance-preserving directions.** These avoid climbing the steep shedding and dumping penalties:
     - pairs of units at the same node, one up and one down by the same amount;
     - flow re-routing, i.e. potential moves combined with compensating injections;
     - intertemporal storage shifts: charge more at $t$, discharge more at $t'$.
   - **Investment–dispatch directions.** Change $k$ at fixed $p$ in physical coordinates. This is the "carrier scaling" move.

   The direction set must span the space, for irreducibility. Scan the directions randomly.
3. **Investments.** In physical coordinates the conditional of $k_g$ given $p$ is proportional to $k_g^{-n_g}e^{-\beta_kc_gk_g}$ on $[\max_tp_t/\bar a_t,\ \bar k_g]$, where $n_g$ is the number of steps with $\bar a>0$ and the power comes from $-\log J$. Sample it by inversion or by univariate slice sampling. Most investment mobility comes from combining this with the investment–dispatch direction in item 2.
4. **Blocks with a Gaussian prior.** This covers the weather coefficients and, in the overcomplete parametrization, all coefficients. Use elliptical slice sampling: it is exact with respect to the Gaussian prior, tolerates the nonsmooth cost, and needs no tuning.
   - **Coupling through the balance.** Weather moves at fixed operation are blocked, because any load change at fixed dispatch is shed or dumped at a steep price. Sample instead in sheared coordinates $(x,\tilde v)$ with $\tilde v=v-Sx$.
   - **The shear $S$** is a fixed linear map given the variables not being updated, for example load changes allocated pro rata to the dispatchable capacity at each node. A shear is volume-preserving, so Gibbs or elliptical slice sampling in $x$ at fixed $\tilde v$ is valid. It lets dispatch follow the weather.
5. **Optional gradient moves** on a few smooth, low-dimensional coarse coefficients, such as large-scale modes. These are not used for the full state.

### 5.2 Level $\ell\ge1$

Let the current state be $s=(s_C,u_\ell)$, where $s_C=u_{<\ell}$.

1. Draw a subchain length $n\sim\mathcal U\{1,\dots,J_\ell\}$. Run $n$ steps of MLDA at level $\ell-1$ starting from $s_C$, giving $\psi_C$.
2. Propose new level-ℓ coefficients independently of the coarse ones. This independence is required for detailed balance.
   - Use pCN with respect to the level prior, for weather, investments and operation alike:
$$u'_\ell=\mu_\ell+\sqrt{1-\varrho^2}\,(u_\ell-\mu_\ell)+\varrho\,Q_\ell^{1/2}\zeta,\qquad\zeta\sim\mathcal N(0,I).$$
   - In the non-redundant limit with a Lebesgue reference and no Gaussian prior on operation, a symmetric random walk on the operation details, scaled per block, works as well.
3. Accept $\psi=(\psi_C,u'_\ell)$ with
$$\alpha=\min\Big\{1,\ \exp\big(\Delta_{\ell-1}(s)-\Delta_{\ell-1}(\psi)\big)\Big\}.$$
   The level priors cancel against the pCN proposal ratio, and the coarse densities cancel by the MLDA construction.

**Fine-mode moves at fixed coarse modes.** When the near-optimal scale is tiny (§5.4), independent fine proposals are almost always rejected. Mix the two-level delayed-acceptance step with exact directional Gibbs sweeps on the level-ℓ coefficients at fixed $s_C$ (§5.1, item 2).
- A random-scan mixture of $\pi_\ell$-reversible kernels is $\pi_\ell$-reversible, so the level-ℓ chain remains a valid proposal generator for level ℓ+1.
- Fixed-order compositions also work if they are palindromic (forward sweep followed by backward sweep).

### 5.3 Properties

- **Exactness.** With fixed $\hat b_L$ and $\hat F_\ell$, the finest chain is reversible with respect to $\pi_L$ for any subchain lengths (Lykkegaard et al., Thm. 1), provided the support condition of §4.4 holds. With online-learned corrections, diminishing adaptation is required.
- **Estimator.** Use random subchain lengths for proposals, continue the subchains to fixed length $J_\ell$, and form
$$\hat Q=\frac1{N_0}\sum_iQ_0(s_0^i)+\sum_{\ell\ge1}\frac1{N_\ell}\sum_j\big[Q_\ell(s_\ell^j)-Q_{\ell-1}(\psi_{\ell-1}^j)\big].$$
  It is asymptotically unbiased for $\mathbb E_{\pi_L}[Q]$ when $Q_\ell(s_\ell)=Q(R\,y_\ell)$ for a fixed coarse restriction $R$. Its variance grows with ε (§4.4).
- **Parallelism.** Under block locality, the proposals and the evaluation of $\Delta$ factorize over blocks, so they parallelize on a GPU. Across levels, MLDA is sequential.

### 5.4 Why not HMC: implementation notes

**Setting of the first implementation.** The level-0 state has about $3.6\cdot10^4$ coordinates. One gradient costs about as much as one cost evaluation, roughly 3 ms. The step size collapsed to about $10^{-5}$ in utilization units, and the near-optimal slack was of the same order per coordinate.

**Why HMC looked right.**
- Its step size scales as $d^{-1/4}$, against $d^{-1/3}$ for MALA and $d^{-1/2}$ for random-walk Metropolis.
- Gradients are cheap and analytic.
- Reflection off the walls of the polytope keeps hard constraints exactly.
- Gradient-guided joint moves promised to move weather and dispatch together.

**Why it does not work here.**
- **The target is not smooth.**
  - The dimension scaling assumes smooth, well-conditioned targets. Here the log-density is piecewise linear on a polytope.
  - Gradients jump at every kink of the cost, for example where a node switches between shedding and dumping, and at every cost breakpoint.
  - Leapfrog incurs an energy error proportional to the step size times the gradient jump at each kink it crosses, so acceptance forces tiny steps.
  - Reflection is exact at walls, but kinks inside the polytope are not walls.
- **Extreme stiffness.**
  - At near-optimal temperatures the measure is close to a product of exponentials at the optimal vertex (equipartition, §1.5).
  - The scale in each direction is $1/(\beta_vr_i)$, set by the reduced cost $r_i$, so scales differ widely across directions.
  - Trajectories are short, and mixing is slow regardless of the dimension scaling.
- **The slow directions are curved or constrained.**
  - In utilization coordinates the $k\cdot\omega$ ridge is curved.
  - Redispatch at fixed net injection is confined by the balance penalty.
  - A diagonal mass matrix follows neither. A dedicated move that scales $k$ and $\omega$ at fixed $p$ helped investments more than HMC did.

**What the structure suggests instead.**
- **Exact line sampling.** Costs are piecewise linear and constraints are linear, so line conditionals in operation space are piecewise exponential and can be sampled exactly at any scale. Step-size tuning is replaced by the choice of directions (§5.1).
- **Problem-adapted directions.** Balance-preserving pairs, flow re-routing, storage shifts and investment–dispatch moves encode the geometry. This is hit-and-run with adapted directions. Isotropic hit-and-run is also valid, but it mixes in $O(d^2)$ or more steps in high dimension.
- **Elliptical slice sampling** for blocks with a Gaussian prior, with sheared coordinates for the weather.
- **Piecewise-deterministic samplers** (Zig-Zag, Bouncy Particle), if gradients are to be used at all.
  - Their event rates are piecewise constant along piecewise-linear potentials, so they can be simulated exactly.
  - Walls are handled by reflection, and local variants parallelize.
  - They have not been tested here.

**Benchmark needed.** No alternative has yet been compared with HMC. Run the level-0 model alone and measure effective samples per second for three quantities: investment marginals, coarse weather coefficients, and total cost. Compare:
1. HMC;
2. exact coordinate Gibbs;
3. directional Gibbs with the direction set of §5.1;
4. optionally, Zig-Zag.

---

## 6. Learning the bias: the normalization that is needed

### 6.1 Decomposition

$$b(x,k)=\underbrace{\big[\log Z(x,k)-\Lambda(k)\big]}_{b_1(x,k):\ \text{weather dependence}}+\Big(1-\frac{\beta_k}{\beta_v}\Big)\underbrace{\Lambda(k)}_{\text{expected log-partition}} .$$

- For $\beta_k=\beta_v$ only $b_1$ is needed. Any function of $k$ alone cancels.
- In general one also needs $\Lambda(k)$, a function on investment space only.

### 6.2 Score identities

Because $\Omega(k)$ does not depend on $x$ (§2.5):
$$\nabla_x\log Z(x,k)=-\beta_v\,\mathbb E_{\pi(\cdot|x,k)}\big[\nabla_xC_x(k,v)\big],$$
where the gradient passes through the data map ($\bar d$ in $r$, $\bar a$ in $p$; subgradients at kinks).

For capacities that enter only the cost ($k_G$, $k_C$ in utilization coordinates):
$$\partial_k\Lambda=-\beta_v\,\mathbb E_{x\sim\rho^x}\mathbb E_{\pi}\big[\partial_kC_x\big].$$
For $k_E$ and the storage ratio, which also shape $\Omega(k)$, there are additional boundary terms. Estimate those few components by finite differences of $\log Z$ at the coarsest level, or keep those capacities fixed.

### 6.3 Online learning (adaptive biasing force)

- **Parametrization.** Use a form that matches the hierarchy: level-wise, $\hat g=\sum_\ell\hat g_\ell(u^w_{\le\ell},k)$, or block-local, $\hat g=\sum_Bg_B(x_{N(B)},k_B)$. Represent $\hat\Lambda(k)$ on the coarse investment coefficients.
- **Regression.** Regress $\nabla_x\hat g$ on the per-sample scores $-\beta_v\nabla_xC_x(k,v)$, and $\nabla_k\hat\Lambda$ on $-\beta_v\nabla_kC_x(k,v)$, using SGD with decreasing steps.
- **Bias.** Set $\hat b_1=\hat g-\mathbb E_{x'}\hat g(x',k)$, where the expectation uses exact samples from the weather model.
- **Validity of the samples.** The scores are valid at any stage of learning: $v\mid x,k$ is exactly $\pi$ whatever the current bias, because the bias depends only on $(x,k)$. Once the bias is correct, $x\mid k$ is $\rho^x$, so chain averages at fixed $k$ estimate $\partial_k\Lambda$.
- **Across levels.** Learn at the coarsest level first, then add corrections $\hat g_\ell-\hat g_{\ell-1}$ from finer chains.

### 6.4 Diagnostic

Under the exact $\rho_1$, the chain's weather marginal is $\rho^x$. Compare regime frequencies and coarse moments of the chain with exact draws from the weather model. Any discrepancy is residual bias error.

---

## 7. Outputs

- **Investment marginal** $\rho_K$: capacity distributions and near-optimal trade-offs, via the multilevel estimator.
- **Cost.** The expected near-optimal cost and its distribution.
- **Weather-cost map.** At fixed $k$,
$$F(x,k)\approx-\beta_v^{-1}\big(\hat b_1(x,k)+\hat\Lambda(k)\big)$$
  ranks weather by its cost under optimal operation of that system. Per regime, this gives a Dunkelflaute classification.
- **α = 0 comparison.** Drop $\hat b$ and rerun to see how much an "optimistic nature" shifts investments.

## 8. Open choices

1. **Temperatures.** Calibrate $\beta_v$ and $\beta_k$ by §1.5. The single-temperature case avoids $\Lambda$ but freezes investments.
2. **Reference measure.** Utilization coordinates (§2.6) or the multiscale Gaussian reference (§4.3, κ = 0). In either case, line ratings and storage ratios remain uncovered.
3. **Prices.** The value of lost load and $c_{\rm dump}$. These are also the slopes of the regularization.
4. **Generating system.** Choice of large-scale modes (Laplacian eigenvectors or geographic Fourier) and temporal harmonics, partition levels, and the order of space and time levels in the chain.
5. **Fine-level priors.** The schedule of $\epsilon_\ell/\sigma_\ell$ per level and field: redundancy and simple construction against level differences and MLDA acceptance. ε → 0 recovers the exact splitting. Also the level means $\mu_\ell$ and the choice κ ∈ {0, 1}.
6. **Bounded fields.** Logit-space expansion (§4.5) versus direct coordinates with penalties at coarse levels.
7. **Weather model.** Conditional details (§3.2) or random effects (§3.5); number of regimes, autoregressive order, interannual factor, validation of the tails.
8. **Spatial restrictions.** Merging generators with different availability needs per-class capacities, or accepts approximate consistency.
9. **Learned corrections.** Surrogate classes for $\hat g$, $\hat\Lambda$, $\hat F_\ell$, and their adaptation schedules.
10. **Base sampler.**
    - **Default:** exact directional Gibbs with a problem-adapted direction set, plus elliptical slice sampling in sheared coordinates for the weather.
    - **Gradient methods:** only for small, smooth blocks.
    - **Option:** piecewise-deterministic samplers.
    - **To decide:** the benchmark of §5.4, and how to choose and weight the directions.
