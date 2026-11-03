# Cell moves: coarse spatial directions at the fine level

Cell moves change one or two carriers uniformly over a cell of a coarser partition. They are
the multigrid part of the move set: the coarse spatial levels act on the fine state through
these directions, which are the cell indicators of the overcomplete Haar frame. See
[README.md](README.md) for the other families and [../model.md](../model.md) §2.2 for the
partitions.

## `cell`: one carrier over a cell

**Direction.** A partition level $q$ (network, countries, nodes, up to the finest level the
current level resolves), a cell $c$ of that level uniformly, and one injecting field $f$
present in the cell (a dispatchable carrier, or storage charge or discharge in the LP form);
every unit of $f$ in $c$ changes its utilization by $t$. The level is drawn with fixed
probabilities proportional to (number of cells)$^{p}$, $p$ = `cell_level_power` (default 1,
which favours finer cells).

**Selection.** Fixed probabilities and the static cell tables (`cellunits`, `cellfields`).
**Compensation.** Additive: the change is the same for every unit of the cell and the finer
structure is kept as an offset; the residuals absorb it. **Cost.** Linear in the cell size.
Rate 0.1.

## `cellpair`: two carriers over a cell

**Direction.** As `cell`, with two distinct carriers $f_1, f_2$ in the cell, changed by
$+t/K_1$ and $-t/K_2$ with $K_j$ the sum of the injection scales of carrier $f_j$ over the
cell. The cell total of the injection is unchanged; node residuals inside the cell
redistribute. This is the transported coarse direction: a share swap of two technologies
over a country.

**Selection.** As `cell`; skipped if $|K_j|<\kappa_{\min}$. **Cost.** Linear in the cell size.
Rate 0.1.

**Use of both.** They move country-scale shares in one draw, which single-unit moves would
need many steps for. They are kept in every kernel.

**Device.** Cell templates `:cell{q}`, `:cellu{q}`, `:cellpair{q}`, `:cellpairu{q}`
(`device/src/cells.jl`). Each family is one partition level q.
* *Round.* A round covers every cell of the level at once (cells of a level are disjoint), times
  the time classes, because the commitment follows.
* *Host choices.* The carriers per cell and step are drawn on the host. The `u` families hold
  the committable carriers.
* *Level probabilities.* Level q gets rounds ∝ $p_q$/(number of cells), which matches the CPU's
  level probabilities $p_q$ (one round per level and class at `cell_power = 1`).
* *Scheduling.* The driver schedules them with `families = :all` ([../device.md](../device.md)).
