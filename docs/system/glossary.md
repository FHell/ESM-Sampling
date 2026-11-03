# Glossary

Symbols and terms used across the system documents, with the code identifier where one
exists. See [docs/README.md](../README.md) for the document index.

## Symbols

| symbol | meaning | code |
|---|---|---|
| $x$ | weather: one year of load and availability at the finest step | `s.xs[end]` |
| $m$, $z_0$, $w_\ell$ | regime path, level-0 standardized anomalies, innovations of level $\ell$ | `s.m`, `s.z0`, `s.ws` |
| $\rho^x$, $p^w$ | weather law, its density in $(m,z_0,w)$ | `WeatherModel` |
| $k$, $k_0$ | capacities (MW); their level-0 coefficients (in the bias) or existing capacity (brownfield) | `s.k`, `k_at0`; `GenGroup.k0` |
| $v$ | operation: angles $\theta$, link utilizations $u$, dispatch utilizations $\omega$, commitment $c$, states of charge $e$ | `s.v` |
| $K(k)$ | annualized capital cost (€/a) with the box penalty | `CostParts.capex` |
| $C_x(k,v)$ | annual operating cost (€/a) with all priced violations | `CostParts.op` |
| $\beta_v$, $\beta_k$, $r$ | operational and investment temperatures (1/€), $r=\beta_k/\beta_v$ | `P.βv`, `P.βk`, `bias.ratio` |
| $Z_x(k)$, $F_x(k)$ | operational partition function and free energy of year $x$ | — |
| $\Lambda(k)$ | $\mathbb E_x\log Z_x(k)$ | — |
| $\rho_K$, $\rho_0$, $\rho_1$ | investment measure; ensemble without and with bias | — |
| $b$, $\hat b$, $\hat g$, $\hat h$ | exact bias; learned bias and its weather and capacity parts | `bias_term`, `ghat`, `hhat` |
| $\alpha$ | 1 with the bias, 0 without | `use_bias` |
| $J(k)$, $\kappa_0$ | reference-measure density; capacity floor (1 MW) | `log_measure`, `E.κ0` |
| $c_{\rm shed}$, $c_{\rm dump}$, $c_{\rm viol}$, $\gamma_k$ | shed, dump and violation prices; capex multiplier of the capacity box penalty | `E.c_shed`, … |
| $r_{n,\tau}$ | nodal residual, demand minus supply (MW) | `LocalCost.r` |
| $\kappa$ | injection scale of a unit per unit utilization ($Ak$, $\mp k$) | `κunit` |
| $Q$, $P$, $\sigma$, $e$ | requirement, unclamped output of the compensating units, shares, excess | `prorata!` |
| $\beta_{n,c,i}$ | learned response coefficients | `AdaptiveModes.β` |
| $\mu$ | canonical CO₂ price (€/t) | `E.co2` |
| $M_\ell$, $h_\ell$ | basis vectors kept and step length of level $\ell$ | `levels` |
| $T$, $\Delta t$ | steps per year and step length at a level | `lv.T`, `lv.h` |

## Terms

- **Level.** A pair $(M_\ell,h_\ell)$ of spatial and temporal resolution; the sampler runs on
  the finest, spatially complete one.
- **Field.** A node-indexed quantity with a kind (capacity, step, instant), a basis and a box.
- **Atom.** One basis vector of a field at one step.
- **Graph Haar basis.** The sparse orthonormal basis of nested cell indicators and
  child contrasts on the partition network → countries → nodes.
- **Data map.** The clamp of load to $\ge 0$ and availability to $[0,1]$.
- **Line move.** A move along a one-dimensional curve with $t$ drawn exactly from the target
  restricted to it.
- **Primary / secondary breakpoints.** Kinks of the piecewise-linear cost along a line known
  in advance / found as zeros of quantities linear between primary breakpoints.
- **Share-preserving pro-rata compensation.** Dispatch that follows a requirement change with
  fixed shares and excess, with a kink where the node turns to surplus and a Jacobian.
- **Requirement tracking.** The variant of that compensation that follows the actual,
  piecewise-linear requirement when losses make it nonlinear in $t$.
- **Frames.** Per-step predicted binding lines and face-projected transport modes, computed
  from $(x,k)$.
- **Proxy class.** Surplus or deficit of a node and step in the proxy injections of the
  frames; indexes the learned response coefficients.
- **Peak set.** The top 5 % of a node's steps by proxy requirement; the steps a `kpeak` move
  changes.
- **Diminishing adaptation.** Learned tables recomputed only at observation counts $2^j$, so
  the kernel changes less and less.
- **Round** (device). A set of conditionally independent sites updated at once for all chains.
- **Equipartition.** About $1/\beta$ of expected excess cost per cost-confined coordinate.
