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

**Device.** Cell templates `:cell` and `:cellpair` (`device/src/cells.jl`, being added): a
round covers every cell of one partition level at once (cells of a level are disjoint), with
the carriers per cell and step drawn on the host. To match the CPU's level probabilities
$p_q$, level $q$ gets a number of rounds proportional to $p_q$ divided by its number of cells.
The driver does not schedule them yet ([../device.md](../device.md)).
