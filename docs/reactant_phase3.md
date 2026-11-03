# Reactant port, phase 3: compensated moves, cell templates, storage blocks, one program per checkpoint

This is the report of the phase-3 device work: it records what was done, how it was verified and
what remains. The system as built is described in [system/device.md](system/device.md); the move
families in [system/moves/](system/moves/). Code is in `device/src/` (`comp.jl`, `cells.jl`,
`blocks.jl`, edits to `terms.jl`, `rounds.jl`, `driver.jl`), tests in `device/test/` (`test_comp.jl`,
`test_program.jl`, `helpers.jl`). Nothing in `src/` was changed.

Verification follows the device rules ([system/testing.md](system/testing.md)):
1. The device code runs on Julia arrays against the CPU reference, with the same explicit moves
   (`prep_node!`, `prorata!`, `prep_tpot!(…; n)`, `cell_entries!`, `prep_storage_ms!`) and the same
   mock uniforms (`MockRNG` for `sample_line!`, u3 for the Jacobian step).
2. The same code is compiled by Reactant on the CPU backend and must equal (1).
3. No statistical tests.

## 1. What was added

**Kinked arguments and the Jacobian step** (`terms.jl`):
* Generator and commitment terms take arguments x₀ + a t + a_k φ_k(t), with
  φ_k = (Q_k + g_k t)₊ − (Q_k)₊.
* `Kinks` holds the kinks of a move (Q, g, P, e, m − 1).
* The kink points and the clamp points on both pieces of every kinked argument are primary
  breakpoints.
* `logjac` and `jac_accept` give the independence step with the third uniform, as in
  `local_sweep!`.

**Families:**

| device family | CPU | rule | sites per round |
|---|---|---|---|
| `:spair` | `prep_spair!` | unit | storage slot j per node × time class; (q1, q2) drawn per site on the host |
| `:tsocL`, `:tsoc` | `prep_tsoc!` (learn / T2) | linear / share-preserving | storage slot j per node × time class |
| `:tpotL`, `:tpot` | `prep_tpot!(…; n)` | linear / share-preserving | distance-2 colour × time class |
| `:cell{q}`, `:cellu{q}`, `:cellpair{q}`, `:cellpairu{q}` | `cell`, `cellpair` at partition level q | additive / balanced | every cell of level q × time class |
| `:sblkL{L}`, `:sblk{L}` (L = 2, 4, 8, 16, 32) | `prep_storage_ms!` (plateau and shapes), one unit | linear / share-preserving | blocks of one parity (offset, stride L) × storage slot j per node |

**Rules shared by the families:**
* *Commitment.* It follows at the main step, and at every step of a block, as on the CPU.
* *Linear coefficients.* They are host tables (`StaticParams.β1`, `β2`; equal split by default),
  with the class from `ChainParams.cls`.
* *Unit choices.* State-independent choices (the unit pair of `:spair`, the carriers of the cell
  moves, the block offset and profile) are drawn on the host when the schedule is built.

**Cell families.**
* *Rounds per level.* They are ∝ (number of cells)^(cell_power − 1), which reproduces the CPU's
  level probabilities ∝ (number of cells)^cell_power: one round per level and class at
  cell_power = 1.
* *Buckets.* Families are bucketed by level, so that each level is padded only to its own
  largest cell. They are also split by commitment (`u`: committable carriers, with commitment
  terms; otherwise none).

**Requirement tracking (decision).** With line losses or lossy links the CPU's share-preserving
compensation uses requirement-tracking kinks: Q_k(t) = Q_k + Δr(t), found by an event walk. The
device does not port these. Such models use the linear rule, whose direction is constant (no
kink, no Jacobian). The share-preserving families (`:tsoc`, `:tpot`, `:sblk…`) refuse them.
Reasons:
* The linear rule is the design's default.
* In the dense form, a bounded event loop would need nested root levels per site
  (requirement → compensating clamps → balance), with no fixed bound.

**Roots per secondary quantity (exactness).**
* *Why the bound changed.* The plain families have ≤ 2 sign changes per secondary quantity
  (proven). With several units compensating at one node, a balance residual can have 3.
* *K roots.* The dense form now keeps K roots per quantity: 2 for the plain families, 4 for the
  others (`nroots`).
* *Exact guard.* A site whose line has more sign changes than K stays at t = 0. The count is a
  property of the whole line, the same from every point on it, so this keeps the kernel
  π-invariant. The tests report the guard rate.

**One program per checkpoint** (`driver.jl`):
* `sweep_program` runs `ns` operation sweeps (traced count, Reactant RNG), then the capacity pass
  (host-drawn proposals), in one compiled call.
* `run_device(cfg; merged=true)` uses it.
* `device_setup(cfg; families=:all)` schedules the plain, compensated, cell and (not yet) block
  families (`device_families`).

## 2. Verification (40 nodes × 6 h, 2 chains)

Tests: `device/test/test_comp.jl`.
* *Cases.* A: all terms, `:spair`, `:tsocL`, `:tpotL`. B: no line losses or lossy links,
  `:tsoc`, `:tpot`. C: all terms, every cell family. D: all terms, `:sblkL2`, `:sblkL4`.
  E: no losses, `:sblk2`, `:sblk4`. Blocks use the plateau and two random shapes.
* *Rounds.* The first two rounds of every family, a prefix of 100–600 sites each.

Each draw is compared three ways:
* **(a)** the device on its own points against CPU `sample_line!`;
* **(c)** the device sampler on the CPU walk's points (plus the device's own kinks outside the
  CPU's range) against the CPU;
* the Jacobian decision on the same t.

Then both sides commit the device's t, and node values, r and cost are compared.

| case | draws | (a) agree | (a) differ (CPU ray splitting) | (c) max \|Δt\| | Jacobian rejections (mismatches) | states vs CPU caches | r | op vs energy_cost |
|---|---|---|---|---|---|---|---|---|
| A | 7 200 | 7 143 | 57 | 3.8e-13 | 0 (0) | bitwise equal | 1.8e-12 MW | 8e-14 |
| B | 4 800 | 3 499 | 1 301 | 4.3e-13 | 189 (0) | ≤ 2.2e-16 | 3.6e-12 MW | 1.1e-13 |
| C | 3 600 | 3 596 | 4 | 7.9e-13 | — | bitwise equal | bitwise | 1.3e-13 |
| D | 1 600 | 1 600 | 0 | 1.1e-12 | — | bitwise equal | bitwise | 1.3e-13 |
| E | 1 600 | 1 518 | 82 | 2.2e-13 | 6 (0) | ≤ 1.1e-16 | 9.1e-13 MW | 1.4e-13 |

**Also checked:**
* *Guard and roots.* The exact guard (more sign changes than kept roots) triggered on 0 sites in
  every case. The largest number of sign changes of one secondary quantity was 3 (tsocL, tpotL,
  cellpair, blocks) and 2 elsewhere.
* *Conflicts.* `check_conflicts` = 0 on the full schedules.
* *Overshoot.* No CPU ray overshoot after the main session's fix of `walk!`.
* *ΔC by itself*, for spair, tsocL and tpotL, agrees with CPU `eval_move!` to ≤ 7e-9 relative.
  That is round-off at |r| ≈ 10⁴ MW.

**Reactant, CPU backend** (one family per program, 2 rounds of ≤ 60 sites), against the
Julia-array run with the same uniforms:

| family | compile | max relative \|reactant − julia\| | peak memory |
|---|---|---|---|
| `:spair` | 155 s | 2.1e-14 | 4.2 GB |
| `:tsocL` | 52 s | 3.4e-14 | 4.4 GB |
| `:tsoc` | 53 s | 3.4e-14 | 4.8 GB |
| `:sblkL4` | 464 s | 1.4e-13 | 11.0 GB |
| `:tpotL`, `:tpot`, cell level 1 | not compiled here (see §3) | | |

The full A–C run with Reactant (all compensated families in one program) reached 12.7 GB without
finishing the tpotL compile, so I stopped it to protect the machine.

**One program per checkpoint** (`test/test_program.jl`, configs/device40.toml, CPU backend):
RESULTS_PROGRAM

## 3. Issues met

* **`sdiv` with a scalar divisor** (`ifelse(Bool, Float64, TracedRNumber)`) is type-unstable under
  tracing. It now returns `zero(a)`.
* **Large cell templates.** One 40-node cell gives about 1 700 points per site in the dense form.
  Two fixes made it fit:
  - the final potential evaluation skips the secondary quantities (`eval_terms(…; sec=false)`);
  - the cell families are bucketed by level and split by commitment.

  The slope-jump form is the real fix for large templates.
* **The CPU walk's 40-nat stop.** Beyond the CPU's last walk point the potential can fall again
  (a T2 kink further out). The device-on-CPU-points check therefore merges the device's own
  kinks outside the CPU's range.

* **Compile size grows with the number of terms.** The term loops of `eval_terms` and
  `primaries` are unrolled at trace time, so the program grows with the term count of a family.
  Measured:
  - `:sblkL4` (5 SoC, 15 generator, 18 commitment terms): 464 s and 11 GB to compile;
  - `:tpotL` (7 lines, 24 generator, 48 commitment terms) did not finish within 20 minutes at
    12.7 GB.

  The fix is structural. Store each term type as one stacked array with a term axis (C, B, 1, n),
  evaluate on (C, B, K, n) and reduce over n: program size becomes independent of n, at the cost
  of the (C, B, K, n) workspace. Alternatively use the slope-jump form for these buckets.
  Until then, `device_setup(cfg; families=:all)` (which includes `:tpotL` and the cell
  templates) is not practical to compile on this machine. The array versions are verified.

## 4. Open

* **Stacked term evaluation** (above): needed before `:tpot(L)`, the coarse cells and long blocks
  can run compiled.
* **Blocks in the driver.** `device_families` does not include the block families yet.
  Cell–class aggregates of the blocks and the proxy and learned shapes as tables are missing.
* **Learned tables.** The linear rule uses static β tables (equal split). Host adaptation
  (`AdaptiveModes` response moments → β upload) is not wired into the driver.
* **Item 4** (templates from localized proxy and learned modes, `tcut`/`acut`; dense
  `tmode`/`amode`): not started.
* **(b) fine-level weather moves on the device: not started.** It needs:
  - raw node weather in the device state (`dnet` is precomputed today);
  - weather entries with the data-map kinks;
  - truncated-Gaussian segments for the prior;
  - host-built propagation tables per coordinate, which depend on the regimes after each host
    FFBS;
  - compensation.

  The CPU compensates with requirement-tracking T2, which the device does not port. The exact
  CPU references without tracking are `comp=false` and, for a device-linear variant, none.
  Recommended first step: uncompensated weather coordinate moves (`comp=false`) as a device
  family, verified against `weather_coord_move!(…; comp=false)` with mock uniforms.
