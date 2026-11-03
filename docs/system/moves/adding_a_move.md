# Adding a move family

A checklist for adding a move family to the CPU kernel and, later, to the device: the
invariance argument, the preparation function, registration and rates, the Jacobian step,
the required tests, the document rows, and the device mapping with its mock-random-number
tests. The line sampler and the invariance constructions are in
[../sampler.md](../sampler.md) §3 and §5, the testing rules in [../testing.md](../testing.md),
the device structure in [../device.md](../device.md).

## 1. Decide the direction and why it is invariant

Write down the curve $s(t)$ through the current state and which of the three constructions
makes the line draw valid ([../sampler.md](../sampler.md) §5.1):

1. **Fixed direction**: it depends only on $(k,x)$, the network, fixed probabilities and tables
   frozen between adaptation updates. Selection rules (skip if $|\kappa|<\kappa_{\min}$, choose a
   partner among the graph neighbours, pick a mode piece with probability $\propto c_j^2$) must
   use only these.
2. **Coordinate line of a reparametrization**: the direction uses the current operation only
   through quantities that are constant along the curve (as the shares and the excess of the
   share-preserving compensation). State the coordinate change and its Jacobian
   $J_{\rm curve}(t)$.
3. **Group move**: a one-parameter group acting on the state; state the group's Jacobian for the
   generalized Gibbs conditional.

A move at step $\tau$ may read and write only steps $\tau-1$ and $\tau$ (parity classes run in
parallel); longer moves need their own block colouring. Every cost term the move changes must be
convex and piecewise linear in $t$, with its kinks either known in advance (primary
breakpoints) or zeros of quantities linear between them (secondary).

## 2. Implement it (CPU, `src/local.jl` or `src/flowmoves.jl`)

- **Preparation function** `prep_<name>!(rng, ws, P, LM, LC, τ[, jac]; κmin, comp, learn)`,
  returning `false` when the draw gives no valid move. Build the direction with the existing
  helpers: `prep_atom!` (a basis vector), `prep_node!` (a node-value change of a complete field,
  which also writes the coefficient changes), `add_entry!`/`addz_node!` for custom patterns,
  `cell_entries!` for cell-constant changes. Use `κunit` for injection scales.
- **Compensation** through `prorata!` (share-preserving: kinked entries and the Jacobian terms
  in `jac`; set `dyn` for requirement tracking, which `tracking_kinks(E)` decides) or, with
  `learn = (AM = …, FF = …)`, the learned linear rule (`compensate_learned!`, no Jacobian).
  Compensation of a requirement change $g$ at a node goes through these two functions only.
- **Two-step or block moves** (states of charge, profiles): fill one `MoveWS` per step of a
  `BlockMove` (the linked previous step in the SoC form), call `finalize!` on every step, collect
  the breakpoints, and `link_soc!` consecutive steps.
- Use the per-step generator `r` passed into the threaded loop and the scratch of the workspace
  (`gbuf`, `gnodes`, `nbuf`, `lseen`, restored to zero or false after use); allocate nothing per
  move.
- **New cost terms** (a new role or term type) need their evaluation in `eval_move!` (with the
  secondary quantities pushed to `ws.sec`), their kinks in `finalize!`, and either an entry in the
  recorded-term evaluation (`eval_cached!`, the same arithmetic) or exclusion from it
  (`cacheable`).

## 3. Register it

- Add a branch for the family name in the move dispatch of `local_sweep!`, or a sweep function
  of its own for block moves (as `storage_ms_sweep!`), called from the kernel's sweep.
- Add `(:name, rate)` to the family list of the kernels that use it: `plain_families`,
  `flow_families` or the list in `adaptive_sweep!`. The number of random moves is rescaled so
  that the plain families keep their absolute rates; check that this still holds.
- **Jacobian.** If the curve has a non-constant Jacobian, add the name to the tuple of families
  with the independence Metropolis step in `local_sweep!` (or apply `logjac` yourself after
  `sample_line!` in a block sweep), and make sure `jac` is emptied at the start of the
  preparation.
- New configuration keys go into `Config` and through `LocalKernel`/`LocalModel` keyword
  arguments; new learned or frozen tables are built from $(k,x)$ or on the adaptation schedule
  only.

## 4. Test it (CPU)

In `test/test_local.jl`, `test/test_flow.jl` or `test/test_model.jl`, following the existing
loops, with all model terms switched on where they apply and in both storage forms if the move
touches storage:

1. **Incremental cost**: for many random moves, the cost change from `eval_move!`/`mv_eval!` at
   a random $t$ equals the global change from `evaluate!` after the commit.
2. **Caches**: after a sweep with the family, `LocalCost` rebuilt from the state equals the
   incrementally updated one (node values, residuals, flows, cell sums).
3. **Invariants** of a reparametrized curve (shares and excess across the kink, residuals that
   must stay fixed), to round-off.
4. **Line draws against quadrature** (Kolmogorov–Smirnov) if the family brings a new kind of
   term, kink or prior.
5. **Recorded-term evaluation** bitwise equal to the plain evaluation, if the move uses it.
6. If the family joins the adaptive mixture, `test/test_exact.jl` exercises it end to end; run
   it.

## 5. Document it

- A section in the move document of its group under `docs/system/moves/` (direction, selection,
  compensation, Jacobian, cost, use, family name and rate, device status) and a row in
  [README.md](README.md).
- The rate table in [../sampler.md](../sampler.md) §4.2, new keys in
  [../config.md](../config.md), new outputs in [../data.md](../data.md), and its timing in
  [../performance.md](../performance.md) once measured.

## 6. Port it to the device

- **Rule and terms.** Map the family onto the move record: a primary direction plus
  compensation sites with one rule (none, unit, linear, share-preserving). The term types of the
  term batch (`GenTerms`, `UCTerms`, `LinkTerms`, `LineTermsD`, `SocTerms`, balance slots, kinks)
  must cover every term it touches. The dense sampler is exact only if every secondary quantity
  has at most two sign changes along the line; prove it per term type and keep the per-site
  check (`root_count`) passing.
- **Schedule.** Define its rounds of conflict-free sites (`family_rounds(Val(:name), …)`):
  time classes for terms that couple steps, per-node slots, the distance-2 colouring for angle
  moves, disjoint cells. Draw any state-independent choices (units, carriers) on the host.
  Declare its read and write footprint in `FOOT` so that `check_conflicts` verifies every round.
- **Kernels.** `site_cols(Val(:name), …)` (the index columns of a site), `build(Val(:name), st,
  cp, sp, tab, r, k)` returning the term batch and the data for the commit, and `commit` (or
  `commit_tb` when the commit needs the kinks at $t$). Captured scalars must be type-parametric,
  flags passed as `Val`, and `abs(a*b)` written as `max(x,-x)`.
- **Tests with mock random numbers** (a new file in `device/test/`, after `test_comp.jl`): build
  the same explicit sites on the CPU with the CPU preparation functions, drive both sides with the
  same uniforms (`MockRNG` for `sample_line!`, the same values as an input array on the device,
  plus the uniform of the Jacobian step), commit the device's $t$ on both sides, and compare $t$,
  node values, the residual cache and the cost, exactly or to round-off. First on Julia arrays,
  then compiled by Reactant on the CPU backend against the array result. No statistical tests.
- **Driver.** Add the family to the family list of `plan` used by `run_device` only after these
  comparisons pass, and update the status table of [../device.md](../device.md).
