# A toy model with the same difficulties, and why MLDA fails on it

The toy keeps three features of the energy model: a network operator, a
weather-dependent right-hand side, and a near-optimal Gibbs measure at large β.
Everything below is exact or follows from elementary estimates.

---

## 1. The toy

Let $G$ be a connected graph on $N$ nodes with Laplacian $L$ ($L\mathbf 1=0$,
$L$ invertible on $\mathbf 1^\perp$). Fix a cost vector $w\in\mathbf 1^\perp$ and
let the "weather" $c\in\mathbb R^N$ be random. For each $c$ the operator solves
$$
\min_{x\in\mathbf 1^\perp}\ w\cdot x\qquad\text{s.t.}\qquad Lx\le c ,
$$
and the planning objective is $\mathbb E_c[\min_x w\cdot x]$.

**Reduction.** Put $y=Lx$ (a bijection of $\mathbf 1^\perp$ with constant
Jacobian) and $a=L^+w$, so $w\cdot x=a\cdot y$. The constraint is $y\le c$ with
$\mathbf 1\cdot y=0$. With the slack $s=c-y\ge0$ and $S=\mathbf 1\cdot c$:
$$
\min\ a\cdot c-a\cdot s\qquad\text{s.t.}\qquad s\ge0,\ \ \mathbf 1\cdot s=S .
$$
The feasible set is the simplex $\Delta_S=\{s\ge0,\ \mathbf 1\cdot s=S\}$
(nonempty and full-dimensional in $\mathbf 1^\perp$ iff $S>0$). The optimum puts
all slack on the node $i^*=\arg\max_i a_i$:
$$
s^*=S\,e_{i^*},\qquad C^*(c)=a\cdot c-a_{i^*}S=-\sum_j r_jc_j,\qquad r_j=a_{i^*}-a_j\ge0 .
$$
$r_j$ are the reduced costs. Every node except $i^*$ is **pinned** at its bound
$y_j=c_j$; node $i^*$ absorbs the balance $\mathbf 1\cdot y=0$.

| toy | energy model |
|---|---|
| $x$, $y=Lx$ | potentials θ, net injections |
| $c$ | weather-dependent capacities/availability |
| $\mathbf 1\cdot y=0$ | power balance |
| $i^*$ | the marginal unit (price setter) |
| $j\ne i^*$ at $y_j=c_j$ | units at their bounds |
| $r_j$ | merit-order distances (reduced costs) |

**Regularized version.** Replace the constraint by a cost with slope $\mu$:
$$
C^\mu_c(s)=a\cdot c-a\cdot s+\mu\sum_i(s_i)_-,\qquad \mathbf 1\cdot s=S,\ s\in\mathbb R^N .
$$
The Gibbs measure below is proper iff $\mu>\max_j r_j$, and it tends to the hard
version as $\mu\to\infty$.

---

## 2. The near-optimal Gibbs measure

$$
\pi_\beta(ds\mid c)\ \propto\ e^{-\beta(C_c(s)-C^*(c))}\,ds=e^{-\beta\sum_jr_js_j}\,\mathbb 1_{\Delta_S}(s)\,ds
$$
(Lebesgue measure on the hyperplane). Eliminating $s_{i^*}=S-\sum_{j\ne i^*}s_j$:

* **Near-vertex regime** $\beta r_jS\gg1$ for all $j\ne i^*$. The $s_j$, $j\ne i^*$,
  are approximately independent, $s_j\sim{\rm Exp}(\beta r_j)$.
* **Equipartition.** $\mathbb E[\beta(C-C^*)]=\sum_{j\ne i^*}\beta r_j\,\mathbb E s_j\approx N-1$:
  one unit per pinned direction.
* **Free energy.** $Z(c)=\int_{\Delta_S}e^{-\beta r\cdot s}ds=\sum_i\dfrac{e^{-\beta r_iS}}{\prod_{j\ne i}\beta(r_j-r_i)}$
  (distinct $r$); in the near-vertex regime
  $$
  F(c)=-\beta^{-1}\log Z\approx C^*(c)+\beta^{-1}\sum_{j\ne i^*}\log(\beta r_j)=-r\cdot c+\text{const}.
  $$
* **Regularized.** Each pinned $s_j$ has an asymmetric Laplace law: rate
  $\beta r_j$ for $s_j>0$ and $\beta(\mu-r_j)$ for $s_j<0$. The violation width
  $1/(\beta(\mu-r_j))$ is as small as the slack width when $\mu\gg r_j$.

All operational widths are $O(1/(\beta r))$, tiny compared with the scale of
$c$. That is the stiffness of the full problem.

---

## 3. The hierarchy

Partition the nodes into cells $C$ of size $n_C$. Following the restriction of
injections by cell sums, the coarse variable of a cell is its total slack and the
fine variables are the within-cell deviations (the non-redundant splitting):
$$
\sigma_C=\sum_{i\in C}s_i,\qquad \xi_i=s_i-\sigma_C/n_C,\qquad \sum_{i\in C}\xi_i=0 .
$$

**Fibers.** For a given $\sigma$, the admissible details form a scaled simplex:
$$
\Phi(\sigma)=\{\xi:\ \sigma_C/n_C+\xi_i\ge0\ \ \forall i\},\qquad {\rm vol}\,\Phi(\sigma)\propto\prod_C\sigma_C^{\,n_C-1}.
$$

**Fine marginal and conditional.** Take a cell $C\not\ni i^*$ in the near-vertex
regime and, for clarity, equal reduced costs $r_i=r$ in the cell. Then
$e^{-\beta r\cdot s}=e^{-\beta r\sigma_C}$ is constant on the fiber, so
$$
\sigma_C\sim{\rm Gamma}(n_C,\beta r)\quad(\text{mean }n_C/\beta r,\ \text{sd }\sqrt{n_C}/\beta r),\qquad
\xi_C\mid\sigma_C\sim{\rm Uniform}\big(\Phi_C(\sigma_C)\big).
$$
Cells are independent given $\sigma_{C^*}=S-\sum_{C\ne C^*}\sigma_C$. With
unequal $r_i$ the marginal is hypoexponential and the conditional is a tilted
uniform; nothing below depends on equality.

**Coarse model.** The natural coarse model evaluates the cost at the
prolongation ($\xi=0$), $s_i=\sigma_C/n_C$, with the aggregated constraint
$\sigma_C\ge0$ (fine feasible ⇒ coarse feasible):
$$
\pi_0(\sigma)\propto\prod_Ce^{-\beta\bar r_C\sigma_C}\mathbb 1\{\sigma_C\ge0\},\qquad \bar r_C=\tfrac1{n_C}\sum_{i\in C}r_i .
$$

---

## 4. Two-level delayed acceptance

A fine state is $(\sigma,\xi)$. The coarse chain moves $\sigma\to\sigma'$; the
details are kept or proposed independently of $\sigma$; the proposal
$(\sigma',\xi')$ is accepted with
$$
\alpha=\min\Big\{1,\ \frac{\pi_1(\sigma',\xi')\,\pi_0(\sigma)}{\pi_1(\sigma,\xi)\,\pi_0(\sigma')}\cdot\frac{q(\xi\mid\xi')}{q(\xi'\mid\xi)}\Big\}.
$$
Write $\pi_1(\sigma,\xi)=m(\sigma)\,\pi(\xi\mid\sigma)$ with the fine marginal $m$.
For kept details ($\xi'=\xi$):
$$
\alpha=\min\Big\{1,\ \underbrace{\frac{m(\sigma')/\pi_0(\sigma')}{m(\sigma)/\pi_0(\sigma)}}_{\text{(M1) coarse marginal}}\cdot\underbrace{\frac{\pi(\xi\mid\sigma')}{\pi(\xi\mid\sigma)}}_{\text{(M2) detail conditional}}\Big\}.
$$
Both factors fail, independently of each other.

### (M1) The coarse model misses the fiber entropy

From §3, $m(\sigma_C)\propto\sigma_C^{n_C-1}e^{-\beta r\sigma_C}$, while
$\pi_0(\sigma_C)\propto e^{-\beta r\sigma_C}$. The cost terms agree exactly (the
coarse model is Galerkin-exact in cost); the difference is the fiber volume
$\sigma_C^{n_C-1}$, i.e. the entropy of the details:
$$
\log\frac{m(\sigma_C)}{\pi_0(\sigma_C)}=(n_C-1)\log\sigma_C+\text{const}.
$$
The coarse chain lives at $\sigma_C\sim1/\beta r$; the fine marginal lives at
$\sigma_C\sim n_C/\beta r$. A coarse proposal drawn from $\pi_0$ is therefore
typically worse by a factor $n_C^{-(n_C-1)}$, i.e. $\approx e^{-n_C\log n_C}$ per
cell. The coarse model is optimistic in cost and pessimistic in entropy.

*Correction:* a renormalization
$\hat F_C(\sigma_C)=-(n_C-1)\log\sigma_C$ (here in closed form; in general the
free energy of the details given the coarse state) makes $\pi_0=m$. Note that
this is a correction of order $n_C\log n_C$ nats per cell, extensive in the
number of fine variables, not a small adjustment.

### (M2) Even the exact coarse marginal fails

Assume (M1) has been corrected, $\pi_0=m$. Then
$\alpha=\min\{1,\pi(\xi\mid\sigma')/\pi(\xi\mid\sigma)\}$, which factorizes over
cells. A coarse move of natural size is $\delta_C=\sigma'_C-\sigma_C\sim\sigma_C/\sqrt{n_C}$
(the sd of the Gamma law).

*Hard constraints.* With $\pi(\xi\mid\sigma)={\rm vol}\,\Phi(\sigma)^{-1}\mathbb 1_{\Phi(\sigma)}(\xi)$:
$$
\alpha_C=\mathbb 1\{\xi\in\Phi(\sigma')\}\Big(\frac{\sigma_C}{\sigma'_C}\Big)^{n_C-1}.
$$
* **Upward moves** ($\delta_C>0$) keep $\xi$ admissible, but
  $(\sigma/\sigma')^{n-1}\approx e^{-(n-1)\delta/\sigma}\approx e^{-\sqrt n\,z}$
  with $z=\sqrt n\,\delta/\sigma=O(1)$: the kept details are too small for the
  larger total.
* **Downward moves** need $\min_i s_i\ge|\delta|/n$. For uniform shares
  $\min_is_i\approx\sigma/n^2$, so only moves with $|\delta|\lesssim\sigma/n$ are
  admissible, a fraction $O(n^{-1/2})$ of the natural move size.

So the acceptance per cell is $O(n_C^{-1/2})$. For a coarse move that changes
$M$ cells, all of them must stay admissible, and half of the moves go down:
$$
\mathbb E\,\alpha\ \le\ \mathbb P(\text{all moved cells admissible})\ \approx\ \big(\tfrac12+O(n_C^{-1/2})\big)^{M}.
$$

*Regularized.* In the near-vertex regime the shares within a cell behave like
independent asymmetric-Laplace variables, rate $\lambda\approx\beta r$ for
$s_i>0$ and $\beta(\mu-r)$ for $s_i<0$; the fiber constraint acts as the tilt
$\lambda=n/\sigma$. Shifting all shares by $\delta/n$ changes $\log\pi(\xi\mid\sigma)$:

* at first order by $-\lambda\delta$ from the normalization (the fiber volume),
  and by $+\lambda\delta$ from the shares that are already negative (a fraction
  $\approx\lambda/\beta\mu$ of them, each changing its penalty by $\beta\mu\,\delta/n$).
  These cancel.
* At second order, the shares that cross zero contribute
  $-\beta\mu\,n\,p_0\,(\delta/n)^2/2$, with share density $p_0\approx\lambda$ at zero.

Hence, with $z=\sqrt n\,\delta/\sigma=O(1)$,
$$
\log\alpha_C\ \approx\ -\frac{\mu}{2r}\,z^2\qquad(\mu/r\ll\sqrt{n_C}),
$$
in both directions; for $\mu/r\gg\sqrt{n_C}$ the behaviour approaches the hard
case. Since properness requires $\mu>r$, every moved cell costs on average more
than half a nat, and with a physical slope ($\mu\gg r$: value of lost load against
merit-order differences) many nats. The cells are independent, so
$\log\alpha=\sum_C\log\alpha_C$ has mean $\approx-M\,\frac{\mu}{2r}\mathbb E z^2$
and standard deviation $O(\sqrt M)$:
$$
\mathbb E\,\alpha\ \approx\ e^{-c\,M\mu/r}.
$$
A coarse subchain changes all cells, so in both versions the acceptance decays
exponentially in the number of cells.

*Independent detail proposals* ($\xi'\sim q$, e.g. a random walk or pCN) do not
help: they replace the kept $\xi$ by a draw that is admissible and typical for
neither $\sigma$ nor $\sigma'$ unless $q$ itself depends on $\sigma'$, which the
construction forbids.

### Why more levels do not help

Each level transition has the same structure: cells of the coarser partition,
fibers of dimension $n_C-1$, and a sum over the moved cells in $\log\alpha$.
Smaller cells ($n_C\to2$) reduce the hard-case loss per cell to $O(1)$; the
regularized loss per cell, $\approx\mu/2r$, does not depend on $n_C$ at all.
Either way, smaller cells mean more cells, and the exponent is extensive in the
number of cells moved, at every level.

---

## 5. What the toy says about the method

MLDA is efficient when the fine conditional $\pi(\xi\mid\sigma)$ hardly changes
under coarse moves, as for Gaussian or Karhunen–Loève priors with smooth
likelihoods, where it is close to $\mathcal N(0,I)$ whatever the coarse
coefficients. Near an LP vertex the opposite holds: the support (hard) or the
soft edges and the volume (regularized) of the conditional are set by the coarse
state itself. Two conditionals at $\sigma$ and $\sigma'$ overlap by a factor
that is bounded away from one per cell and multiplies over cells.

Ways out that the toy suggests:

* **Scale-covariant splitting.** Write $s_i=\sigma_Cu_i$ with shares $u$ on the
  unit simplex (Jacobian $\sigma_C^{n_C-1}$, which belongs in the coarse
  marginal). For equal $r_i$ the conditional of $u$ given $\sigma$ is
  ${\rm Dirichlet}(1,\dots,1)$, independent of $\sigma$, and kept shares are
  always accepted. With unequal $r_i$ the conditional is tilted by
  $e^{-\beta\sigma_C\sum_i(r_i-\bar r_C)u_i}$, which is independent of $\sigma$
  only if $\beta\sigma_C|r_i-\bar r_C|\ll1$, i.e. $|r_i-\bar r_C|\ll\bar r_C/n_C$:
  cells must be homogeneous in reduced cost (merit-order position), not merely
  in geography.
* **Hierarchical moves under the fine cost.** Moving $\sigma_C$ by a line
  conditional of the fine density, with the details either rescaled or
  re-equilibrated, needs no coarse model and has no rejection; the hierarchy then
  enters as a set of directions, not as a sequence of approximate targets.
* **Local moves.** A coarse move that changes one cell has acceptance
  $\approx e^{-c\mu/r}$ (regularized) or $O(n_C^{-1/2})$ (hard); this does not
  scale to global moves.

---

## 5a. A coarse chain on all degrees of freedom

Delayed acceptance does not require the levels to have different dimensions.
Let the coarse level be a density $\pi_c(\sigma,\xi)$ on the **full** state and
let the coarse kernel be any $\pi_c$-reversible kernel on the full state. A
coarse subchain from $s$ ends at $\psi$, which is accepted with
$$
\alpha=\min\Big\{1,\frac{w(\psi)}{w(s)}\Big\},\qquad w=\pi_f/\pi_c .
$$
Nothing has to be held fixed or proposed independently of the coarse
coordinates. The coarse kernel may change σ and **fix** the details in the same
step, in either of two ways.

* **Redraw.** After a move of σ, draw $\xi\sim g(\cdot\mid\sigma')$, where
  $\pi_c(\sigma,\xi)=\tilde m(\sigma)\,g(\xi\mid\sigma)$. This is a Gibbs step for
  $\pi_c$. It is useful when $g$ factorizes over cells (and nodes) and can be
  sampled exactly.
* **Transport.** Map the details deterministically with the move, as a
  Metropolis step for $\pi_c$ that carries the Jacobian: for example the
  share-preserving map $\xi\mapsto(\sigma'/\sigma)\,\xi$ with Jacobian
  $(\sigma'/\sigma)^{n-1}$, or an active-set-preserving map that keeps pinned
  coordinates at their bounds and moves only the free ones. A state-dependent map
  is allowed if the rule that selects the pinned set gives the same selection
  before and after the move, or if its selection probability enters the ratio.

**Accuracy required of the detail model.** Use shares $u=s/\sigma_C$ and the
coarse detail model $g={\rm Dirichlet}(1,\dots,1)$, independent of σ, with the
coarse marginal $\tilde m$ chosen equal to the fine marginal (it then contains
the fiber entropy of (M1) automatically through the Jacobian $\sigma^{n-1}$ of
the share coordinates). The fine conditional is the tilted Dirichlet
$$
\pi(u\mid\sigma)\ \propto\ \exp\Big(-\beta\sigma\sum_{i\in C}(r_i-\bar r_C)u_i\Big),
$$
so per cell
$\log w=-\beta\sigma\sum_i(r_i-\bar r)u_i-\log N(\sigma)+{\rm const}$. Under a
coarse move $\delta\sim\sigma/\sqrt n$ with $u$ transported unchanged, the mean
part cancels against $\partial_\sigma\log N$, and the fluctuating part is
$\beta\delta\sum_i(r_i-\bar r)u_i$. With $u_i\approx{\rm Exp}(1)/n$ and
$\sigma\approx n/\beta\bar r$, its standard deviation is
$$
\beta\,\frac{\sigma}{\sqrt n}\cdot\frac{\Delta r_C}{\sqrt n}\ =\ \frac{\Delta r_C}{\bar r_C},
\qquad \Delta r_C^2=\frac1{n_C}\sum_{i\in C}(r_i-\bar r_C)^2 .
$$
For a move of $M$ cells $\log(w(\psi)/w(s))$ is approximately Gaussian with
variance $v=\sum_C(\Delta r_C/\bar r_C)^2z_C^2$ and mean $-v/2$, so
$$
\mathbb E\,\alpha\approx2\Phi\big(-\sqrt v/2\big),\qquad v\approx M\,(\Delta r/\bar r)^2 .
$$
Global moves remain possible as long as $\Delta r/\bar r\ll1/\sqrt M$: the
extensivity is still there, but its constant is set by the quality of the
detail model (here the merit-order homogeneity of the cells) rather than being
bounded below by $\tfrac12$ nat per cell. For homogeneous cells the coarse chain
is exact and every proposal is accepted.

**What this buys: convergence, not cheaper evaluation.** $\pi_c$ acts on all
degrees of freedom and need not be cheaper to evaluate than $\pi_f$; in this
setting the evaluation of $\pi_f$ is not the bottleneck anyway. The gain is that
$\pi_c$ **mixes fast**: its aggregates move under a coarse model that contains
the right entropy, and its details are redrawn exactly or transported along the
active set, so a short coarse subchain decorrelates the whole state. The fine
level only filters through $w$. The speed-up of the fine chain is roughly
$$
\frac{\tau_{\rm mix}(\pi_f\text{-chain})}{\tau_{\rm mix}(\pi_c\text{-chain})}\cdot\mathbb E\,\alpha .
$$

**In the energy model** $\pi_c$ would consist of a coarse network for cell
injections and storage totals, together with per-unit local models along the
merit order: the marginal unit of each cell absorbs a change of the cell total,
pinned units stay pinned. Cells must be grouped so that the reduced costs within
a cell are similar, to the tolerance $\Delta r/\bar r\ll1/\sqrt M$.

---

## 6. The same toy shows the weather problem

With $F(c)\approx-r\cdot c+\text{const}$ the joint chain over $(c,s)$ at α = 0
has the weather marginal $\rho^c(c)\,Z(c)\propto\rho^c(c)\,e^{\beta r\cdot c}$.
For $c\sim\mathcal N(\mu_c,\Sigma_c)$ this is $\mathcal N(\mu_c+\beta\Sigma_cr,\ \Sigma_c)$:
a shift of $\beta\,\|\Sigma_c^{1/2}r\|$ standard deviations, huge at near-optimal β.
The exact bias is linear, $b(c)=\beta r\cdot c$. A learned bias
$\hat b(c)=\beta\hat r\cdot c$ leaves the weather at
$\mathcal N(\mu_c+\beta\Sigma_c(r-\hat r),\Sigma_c)$, so the slopes must be learned
to $|\hat r-r|\ll1/(\beta\sigma_c)$: relative precision $1/(\beta r\sigma_c)$.
Any error in $\hat b$ is amplified by β; this is a property of the joint
density, not of the sampler.
