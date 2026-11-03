# Near-optimal sampling of energy system decisions under weather uncertainty — mathematical setup

Working notes. The document covers, in order:

- the abstract target measures (§1);
- the energy system model, with hard physical constraints and a priced balance relaxation (§2);
- the multilevel Gaussian-mixture weather model (§3);
- the multilevel structure of the joint model (§4);
- multilevel delayed acceptance (MLDA) on that hierarchy (§5);
- the one normalization that has to be learned (§6).

Spatial coarsenings are kept abstract throughout.

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
- **Reference measures.** Lebesgue $dk$ on $\mathcal K$ and $dv$ on $\mathcal V$. The choice of coordinates for $v$ is a modelling decision (§2.6).
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

Two directions are left without a capacity-normalized coordinate: line ratings (through the polytope in $\theta$-space) and the storage power-to-energy ratio in (H2).

---

## 3. Weather model: multilevel Gaussian mixture

### 3.1 Anomalies and restriction

- **Anomalies.** $x=\bar x+x'$, where $\bar x$ is the climatological hour-of-year mean over the 60 years.
- **Weather restriction.** $R^w_\ell$ must equal the restriction the energy model applies to its data, so that coarse weather *is* coarse model data:
  - load: like injections, i.e. block averages in time and $E_X^\top$ in space;
  - availability: block averages in time, and the generator restriction $R_G$ in space (fixed weights).
- **Splitting.** $x'_{\ell+1}=E^w_\ell z^w_\ell+D^w_\ell\xi^w_\ell$, with $R^w_\ell E^w_\ell=I$ and $R^w_\ell D^w_\ell=0$.

### 3.2 Generative model

**Coarsest level: regime-switching linear-Gaussian model in coarse time (step $\tau$).**

- A regime Markov chain $m_\tau$ on $K$ states.
- Coarse weather evolves as
$$z^w_0(\tau)\mid m_\tau,z^w_0(\tau-1)\ \sim\ \mathcal N\big(\Phi_{m_\tau}z^w_0(\tau-1)+\mu_{m_\tau}(s_\tau),\ \Sigma_{m_\tau}\big),$$
  with $s_\tau$ the calendar season.
- Conditional on the regime path $m=(m_\tau)$, everything is Gaussian. The mixture components are weather regimes.
- An optional per-year latent factor captures interannual variability.

**Details, level by level (non-centered).**
$$\xi^w_\ell=A^{m}_\ell z^w_\ell+c^{m}_\ell+(Q^{m}_\ell)^{1/2}w_\ell,\qquad w_\ell\sim\mathcal N(0,I)\ \text{independent of everything coarser}.$$
This is a multiresolution (multiscale autoregressive) Gaussian model on the level tree, mixed over regime paths.

### 3.3 Properties

- **Projective by construction.** The level-$\ell$ weather state is $x_\ell=(m,z^w_0,w_0,\dots,w_{\ell-1})$, with density
$$p^w_\ell(x_\ell)=p(m)\,p(z^w_0\mid m)\prod_{j<\ell}\mathcal N(w_j;0,I).$$
  This is the exact marginal of the finest model, so the weather part needs no renormalization.
- **Local.** $Q_\ell^m$ is block-diagonal over the detail blocks of §4, and $A^m_\ell$ couples each block only to coarse variables in its neighborhood. Given the coarse level, detail blocks are independent.
- **Exactly samplable.** Expectations $\mathbb E_{x'\sim\rho^x}[\cdot]$ can be computed by direct sampling.

### 3.4 Fitting and validation

- **Coarsest level.** EM (or Gibbs) for the switching model on coarse coordinates. With daily coarse steps, 60 years give about 22,000 time steps.
- **Details.** Responsibility-weighted linear regression of the detail coordinates on their coarse parents, pooled over all blocks within a season.
- **Validation** on held-out years:
  - regime frequencies and persistence;
  - spatial and temporal correlations per level;
  - the frequency of low-wind, cold multi-day events.

---

## 4. Level hierarchy of the joint model

### 4.1 States and splitting

The state at level $\ell$ is $s_\ell=(k_\ell,x_\ell,v_\ell)$:

- $k_\ell$: unchanged by time steps; restricted by $R_K$ in space.
- $x_\ell$: as in §3.3.
- $v_\ell$: coarse operation. In time, step fields are block-averaged, instant fields ($\omega_e$) are decimated, and $\theta$ is block-averaged. In space, the node splitting satisfies (S1)–(S3), and $\theta=E_Xz+D_X\xi$ induces the splitting of $f$.

The fine modes added from level $\ell$ to $\ell+1$ are $\phi_\ell=(w_\ell,\xi^v_\ell)$.

### 4.2 Level models

$$\pi_\ell(s_\ell)\ \propto\ e^{-\beta_k\langle c_k,k_\ell\rangle}\ p^w_\ell(x_\ell)\ e^{-\beta_vC^{(\ell)}_{x_\ell}(k_\ell,v_\ell)}\ \mathbb 1_{\Omega_\ell(k_\ell)}(v_\ell)\ e^{-\hat b_\ell(x_\ell,k_\ell)-\hat F_\ell(s_\ell)} .$$

- $C^{(\ell)}$ is the coarse energy model: operators $L_c,\partial_t^c,G_c,\dots$ and coarse data $(\bar d,\bar a)$ computed from $x_\ell$.
- $\hat b_\ell$ is the learned bias (§6).
- $\hat F_\ell$ is an optional learned renormalization of the coarse model, playing the role of MLDA's adaptive error model.
- Only $\pi_L$ has to be exact. Coarser $\pi_\ell$ are free approximations.

### 4.3 Consistency of the time steps

- **(A1) holds exactly for $\Omega$.**
  - The boxes in (H1)–(H3) are preserved by averaging and by decimation.
  - The storage inequality sums over a block, because the capacities are constant in time.
  - Line limits survive averaging by convexity.
- **Costs are only approximately consistent.**
  - For given $k$ and $\omega$, the coarse model uses $\bar a\,k\,\bar\omega$ in place of $k\,\overline{a\omega}$.
  - The data map does not commute with averaging: $\mathrm{clip}(\overline a)\ne\overline{\mathrm{clip}(a)}$.
  - By Jensen, coarse penalties underestimate fine ones: $\varphi(\bar r)\le\overline{\varphi(r)}$. Coarse models are optimistic.
  - These are exactly the discrepancies that $\hat F_\ell$ and the acceptance step correct.
- **Fibers can be empty.** A coarse-feasible state may have no feasible fine completion. Such proposals are rejected. With priced balance this happens only through (H1)–(H3).

### 4.4 Level increment

Define $\Delta_\ell$ by
$$\pi_{\ell+1}(s_{\ell+1})=\pi_\ell(s_\ell)\,\mathcal N(w_\ell;0,I)\,e^{-\Delta_\ell(s_{\ell+1})}.$$
Then
$$\Delta_\ell=\beta_v\big[C^{(\ell+1)}-C^{(\ell)}\big]+\big[\hat b_{\ell+1}-\hat b_\ell\big]+\big[\hat F_{\ell+1}-\hat F_\ell\big]+\infty\cdot\mathbb 1\{v_{\ell+1}\notin\Omega_{\ell+1}\}+\text{const}.$$
The capex term and the weather prior cancel exactly.

---

## 5. MLDA on the joint model

### 5.1 Coarsest level

Use any $\pi_0$-invariant kernel. A natural choice is blocked Gibbs:

1. Sample the regime path $m\mid z^w_0$ exactly by forward-filtering backward-sampling.
2. Update $(k,z^w_0,v_0)\mid m$ by reflective HMC or MALA, handling the hard constraints by reflection.

### 5.2 Level $\ell\ge1$

Let the current state be $s=(s_C,\phi)$, where $s_C$ is the coarse part and $\phi$ the fine modes.

1. Draw a subchain length $n\sim\mathcal U\{1,\dots,J_\ell\}$. Run $n$ steps of MLDA at level $\ell-1$ starting from $s_C$, giving $\psi_C$.
2. Propose new fine modes independently of the coarse modes. This independence is required for detailed balance.
   - Weather innovations by pCN: $w'=\sqrt{1-\varrho^2}\,w+\varrho\,\zeta$ with $\zeta\sim\mathcal N(0,I)$.
   - Operation details by a symmetric random walk, scaled per block.
3. Accept $\psi=(\psi_C,\psi_F)$ with
$$\alpha=\min\Big\{1,\ \exp\big(\Delta_{\ell-1}(s)-\Delta_{\ell-1}(\psi)\big)\Big\}.$$
   The Gaussian factors cancel against the pCN proposal ratio, and the coarse densities cancel by the MLDA construction.

### 5.3 Properties

- **Exactness.** With fixed $\hat b_L$ and $\hat F_\ell$, the finest chain is reversible with respect to $\pi_L$ for any subchain lengths (Lykkegaard et al., Thm. 1). With online-learned corrections, diminishing adaptation is required.
- **Estimator.** Use random subchain lengths for proposals, continue the subchains to fixed length $J_\ell$, and form
$$\hat Q=\frac1{N_0}\sum_iQ_0(s_0^i)+\sum_{\ell\ge1}\frac1{N_\ell}\sum_j\big[Q_\ell(s_\ell^j)-Q_{\ell-1}(\psi_{\ell-1}^j)\big].$$
  It is asymptotically unbiased for $\mathbb E_{\pi_L}[Q]$, provided $Q$ factors through a fixed coarse level.
- **Parallelism.** Under locality, step 2 and the evaluation of $\Delta$ factorize over blocks, so they parallelize on a GPU. Across levels, MLDA is sequential.

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

- **Parametrization.** Use a local form, $\hat g(x,k)=\sum_B g_B(x_{N(B)},k_B)$, consistent with block locality, and $\hat\Lambda(k)$ on the coarse investment space.
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
2. **Reference measure.** Utilization coordinates as recommended in §2.6; line ratings and storage ratios remain uncovered.
3. **Prices.** The value of lost load and $c_{\rm dump}$. These are also the slopes of the regularization.
4. **Spatial splitting** against (S1)–(S3) and locality. Merging generators with different availability needs per-class capacities, or else accepts an approximate (A1).
5. **Weather model.** Number of regimes, autoregressive order, interannual factor, and validation of the tails.
6. **Proposals.** Random-walk scales per block; whether to add a fitted Gaussian reference for operation details so that pCN can be used there too.
7. **Learned corrections.** Surrogate classes for $\hat g$, $\hat\Lambda$, $\hat F_\ell$, and their adaptation schedules.
