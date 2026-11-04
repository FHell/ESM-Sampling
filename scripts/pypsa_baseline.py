"""PyPSA-Eur's own optimum and MGA envelopes on its network (campaign H2b; docs/system/validation.md
§2.2, §2.7): the composed network of the PyPSA-Eur build (resources/esm256/networks/composed_2050.nc,
weather year 2013), configured to our scope, solved by PyPSA with HiGHS; then, for each cost slack
eps, the minimum and the maximum of the total capacity of each carrier group subject to
cost <= (1 + eps) * C* (PyPSA's MGA budget: capex + opex including the fixed costs of the existing
fleet, as in `optimize_mga`).

    python scripts/pypsa_baseline.py <config.toml> [key=value ...]

    resolution=24h | 6h | 3h | 1h    time aggregation: averages over blocks, snapshot weightings = h
                                     (PyPSA-Eur's `average_every_nhours`; default from the config)
    eps=0.01,0.02,0.05,0.1          MGA cost slacks (default: none, optimum only)
    carriers=all | onwind,solar,... MGA carrier groups (default all: onwind solar offwind gas battery H2)
    time_limit=<s>                  HiGHS time limit per LP

Configuration keys (TOML; `out` is set by hpc/run.sh, the outputs go to its directory):
    network      path of the composed network; `${VAR}` expanded (default
                 ${ESM_PYPSA_EUR}/resources/esm256/networks/composed_2050.nc)
    resolution   default time step ("24h")
    co2          "cap": keep PyPSA-Eur's CO2Limit (its dual is the CO2 price);
                 "none": drop it; a number: drop it and add that price (EUR/t) to the marginal costs
    load_shedding  EUR/MWh of a shedding generator at every bus (0: none; default 3000, our c_shed)
    line_expansion false: AC lines and DC links at their current ratings (our scope)
    snapshots    optional number of leading snapshots after aggregation (smoke tests only)

Our scope (validation.md §2.2): electricity only (the build has the sector off), the same cluster
count, no line expansion, load shedding at our c_shed, the CO2 cap of the build (our side: co2_cap).
Outputs next to `out`: summary_pypsa.toml (objective, capex/opex, CO2 price, capacities per group
and country, LP size, times, memory), capacity_pypsa.csv, and with eps: mga_pypsa.csv (one row per
solve, appended as they finish; a restarted job skips finished rows), mga_capacity_pypsa.csv,
summary_mga_pypsa.toml. HiGHS threads: ESM_LP_THREADS. Runs under hpc/run.sh in `python` mode.
"""

import os
import re
import resource
import sys
import time
import tomllib

import numpy as np
import pandas as pd
import pypsa
from linopy import LinearExpression, QuadraticExpression

# carrier groups of PyPSA-Eur mapped to our k fields (power capacities, MW)
GROUPS = {
    "onwind": [("Generator", ["onwind"])],
    "solar": [("Generator", ["solar", "solar-hsat", "solar rooftop"])],
    "offwind": [("Generator", ["offwind-ac", "offwind-dc", "offwind-float"])],
    "gas": [("Generator", ["CCGT", "OCGT"])],
    "battery": [("Link", ["battery discharger"])],
    "H2": [("Link", ["H2 Fuel Cell"])],
}
# HiGHS options of the second attempt when a solve does not end optimal (as in src/lp_baseline.jl)
FALLBACK = {"solver": "ipx", "presolve": "off"}
NOM = {"Generator": "p_nom", "Link": "p_nom", "StorageUnit": "p_nom", "Store": "e_nom"}


def log(*a):
    print(*a, flush=True)


def expand(s):
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), s)


def toml_dump(d, path):
    """Minimal TOML writer (nested dicts of scalars and lists)."""

    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            v = float(v)
            return "nan" if np.isnan(v) else ("inf" if np.isinf(v) else repr(v))
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(val(x) for x in v) + "]"
        return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'

    def key(k):
        return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else '"' + k + '"'

    lines = []

    def emit(d, prefix):
        for k, v in d.items():
            if not isinstance(v, dict):
                lines.append(f"{key(k)} = {val(v)}")
        for k, v in d.items():
            if isinstance(v, dict):
                p = prefix + [key(k)]
                lines.append("")
                lines.append("[" + ".".join(p) + "]")
                emit(v, p)

    emit(d, [])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def average_every_nhours(n, offset):
    """Block means of all time series, snapshot weightings summed (PyPSA-Eur's prepare_network)."""
    if offset in ("1h", "1H"):
        return n
    m = n.copy(snapshots=[])
    w = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(w.index)
    m.snapshot_weightings = w
    for c in n.components:
        if not c.dynamic:
            continue
        mc = m.components[c.name]
        for attr, df in c.dynamic.items():
            if not df.empty:
                mc.dynamic[attr] = df.resample(offset).mean()
    return m


def scope(n, cfg):
    """Configure the composed network to our scope (validation.md §2.2)."""
    if not cfg.get("line_expansion", False):
        n.lines["s_nom_extendable"] = False
        dc = n.links.carrier == "DC"
        n.links.loc[dc, "p_nom_extendable"] = False
    co2 = cfg.get("co2", "cap")
    if co2 != "cap" and "CO2Limit-upper" in n.global_constraints.index:
        n.remove("GlobalConstraint", "CO2Limit-upper")
    if isinstance(co2, (int, float)) and not isinstance(co2, bool):
        em = n.generators.carrier.map(n.carriers.co2_emissions).fillna(0.0)
        n.generators["marginal_cost"] += co2 * em / n.generators.efficiency
    cs = float(cfg.get("load_shedding", 3000.0))
    if cs > 0:
        n.add("Carrier", "load shedding")
        peak = n.loads_t.p_set.T.groupby(n.loads.bus).sum().T.max().reindex(n.buses.index).fillna(0.0)
        buses = n.buses.index[n.buses.carrier == "AC"] if "carrier" in n.buses else n.buses.index
        n.add("Generator", buses + " load shedding", bus=buses, carrier="load shedding",
              p_nom=(2 * peak[buses] + 1.0).values, marginal_cost=cs)
    return n


def group_index(n, g, extendable_only):
    """(component, index) pairs of a carrier group."""
    out = []
    for comp, carriers in GROUPS[g]:
        df = n.static(comp)
        sel = df.carrier.isin(carriers)
        if extendable_only:
            sel &= df[NOM[comp] + "_extendable"]
        out.append((comp, df.index[sel]))
    return out


def capacities(n):
    """Optimal capacity per group, total and per country (MW)."""
    country = n.buses.country if "country" in n.buses else pd.Series("all", index=n.buses.index)
    tot, byc = {}, {}
    for g in GROUPS:
        s = pd.Series(0.0, index=sorted(country.unique()))
        for comp, idx in group_index(n, g, False):
            df = n.static(comp)
            col = NOM[comp] + "_opt"
            bus = df.loc[idx, "bus" if comp != "Link" else "bus1" if g in ("battery", "H2") else "bus0"]
            s = s.add(df.loc[idx, col].groupby(bus.map(country)).sum(), fill_value=0.0)
        tot[g] = float(s.sum())
        byc[g] = {k: float(v) for k, v in s.items()}
    return tot, byc


def main():
    cfgpath = sys.argv[1]
    with open(cfgpath, "rb") as f:
        cfg = tomllib.load(f)
    opts = {}
    for a in sys.argv[2:]:
        k, _, v = a.partition("=")
        if k not in ("resolution", "eps", "carriers", "time_limit") or not v:
            sys.exit(f"argument {a}: expected resolution=, eps=, carriers= or time_limit=")
        opts[k] = v
    out = os.path.dirname(cfg.get("out", "results/pypsa/out.h5")) or "."
    os.makedirs(out, exist_ok=True)
    os.environ.setdefault("ESM_PYPSA_EUR", os.path.join(os.environ.get("ESM_PROJ", "."), "external", "pypsa-eur"))
    path = expand(cfg.get("network", "${ESM_PYPSA_EUR}/resources/esm256/networks/composed_2050.nc"))
    res = opts.get("resolution", cfg.get("resolution", "24h"))
    eps = [float(x) for x in opts["eps"].split(",")] if "eps" in opts else []
    groups = list(GROUPS) if opts.get("carriers", "all") == "all" else opts["carriers"].split(",")
    for g in groups:
        if g not in GROUPS:
            sys.exit(f"carrier group {g}: one of {list(GROUPS)}")
    solver_options = {"solver": "ipm", "run_crossover": "off"}
    if os.environ.get("ESM_LP_THREADS"):
        solver_options["threads"] = int(os.environ["ESM_LP_THREADS"])
    if "time_limit" in opts:
        solver_options["time_limit"] = float(opts["time_limit"])

    t0 = time.time()
    n = pypsa.Network(path)
    n = average_every_nhours(n, res)
    if "snapshots" in cfg:
        n.set_snapshots(n.snapshots[: int(cfg["snapshots"])])
    n = scope(n, cfg)
    log(f"PyPSA baseline: {path}, {len(n.buses)} buses, {len(n.snapshots)} snapshots at {res}, "
        f"co2 {cfg.get('co2', 'cap')}, eps {eps}, groups {groups}, HiGHS {solver_options}")

    # ---- the optimum
    tb = time.time()
    status, cond = n.optimize(solver_name="highs", solver_options=solver_options)
    if cond != "optimal":                     # HiPO's tolerance check after postsolve: retry with IPX
        log(f"optimum: {status} {cond}; repeating with IPX without presolve")
        status, cond = n.optimize.solve_model(solver_name="highs", solver_options={**solver_options, **FALLBACK})
    t_opt = time.time() - tb
    if cond != "optimal":
        sys.exit(f"optimum: {status} {cond}")
    st = n.statistics
    capex, opex = float(st.capex().sum()), float(st.opex().sum())
    fixed = float(st.installed_capex().sum())
    total = capex + opex                       # PyPSA's MGA reference cost
    co2p = float(-n.global_constraints.mu.get("CO2Limit-upper", np.nan)) if "mu" in n.global_constraints else np.nan
    tot, byc = capacities(n)
    m = n.model
    gen = n.generators_t.p.T.groupby(n.generators.carrier).sum().T
    w = n.snapshot_weightings.generators
    energy = {c: float((gen[c] * w).sum()) for c in gen.columns}
    summ = {
        "commit": os.environ.get("ESM_COMMIT", ""), "run_id": os.environ.get("ESM_RUN_ID", ""),
        "config": cfgpath, "network": path, "args": " ".join(sys.argv[2:]), "resolution": res,
        "buses": len(n.buses), "snapshots": len(n.snapshots), "co2": str(cfg.get("co2", "cap")),
        "status": f"{status} {cond}", "objective_eur_per_a": float(n.objective + n.objective_constant),
        "capex_eur_per_a": capex, "opex_eur_per_a": opex, "fixed_capex_eur_per_a": fixed,
        "total_cost_eur_per_a": total, "co2_price_eur_per_t": co2p,
        "shed_MWh": energy.get("load shedding", 0.0),
        "energy_MWh": energy,
        "lp_variables": int(m.nvars), "lp_constraints": int(m.ncons),
        "lp_threads": os.environ.get("ESM_LP_THREADS", ""),
        "seconds_setup": tb - t0, "seconds_optimum": t_opt,
        "maxrss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "capacity": {g: {"total_MW": tot[g], "by_country_MW": byc[g]} for g in GROUPS},
    }
    toml_dump(summ, os.path.join(out, "summary_pypsa.toml"))
    with open(os.path.join(out, "capacity_pypsa.csv"), "w") as f:
        f.write("carrier,country,MW\n")
        for g in GROUPS:
            for c, v in byc[g].items():
                f.write(f"{g},{c},{v}\n")
    log(f"optimum: total {total:.6e} EUR/a (objective {summ['objective_eur_per_a']:.6e}), CO2 price {co2p:.1f}, "
        f"{m.nvars} variables, {m.ncons} constraints  [{t_opt:.0f} s]")
    if not eps:
        return 0

    # ---- MGA (PyPSA's budget constraint on one model; resumable through mga_pypsa.csv)
    mpath, cpath = os.path.join(out, "mga_pypsa.csv"), os.path.join(out, "mga_capacity_pypsa.csv")
    head = "eps,target,sense,status,cost_eur,cost_rel_excess,total_MW,seconds\n"
    done, rows = set(), {}
    if os.path.exists(mpath):
        for line in open(mpath).read().splitlines()[1:]:
            c = line.split(",")
            if len(c) == 8:
                rows[(float(c[0]), c[1], c[2])] = c
        done = set(rows)
        keep = [l for l in open(cpath).read().splitlines()[1:]
                if len(l.split(",")) == 6 and (float(l.split(",")[0]), l.split(",")[1], l.split(",")[2]) in done] \
            if os.path.exists(cpath) else []
        with open(cpath, "w") as f:
            f.write("eps,target,sense,carrier,country,MW\n" + "".join(l + "\n" for l in keep))
        log(f"MGA: resuming, {len(done)} solves done")
    else:
        open(mpath, "w").write(head)
        open(cpath, "w").write("eps,target,sense,carrier,country,MW\n")
    m = n.optimize.create_model()
    cost = m.objective
    if not isinstance(cost, (LinearExpression, QuadraticExpression)):
        cost = cost.expression
    m.add_constraints(cost + fixed <= (1 + eps[0]) * total, name="budget")
    for e in eps:
        m.constraints["budget"].rhs = (1 + e) * total - fixed
        for g in groups:
            expr = None
            for comp, idx in group_index(n, g, True):
                if len(idx) == 0:
                    continue
                var = m.variables[f"{comp}-{NOM[comp]}"]          # one dimension: the extendable units
                v = var.sel({var.dims[0]: list(idx)}).sum()
                expr = v if expr is None else expr + v
            for sense in ("min", "max"):
                if (e, g, sense) in done:
                    continue
                tb = time.time()
                if expr is None:
                    stt, cc = "ok", "no extendable units"
                else:
                    m.objective = expr if sense == "min" else -1 * expr
                    stt, cc = n.optimize.solve_model(solver_name="highs", solver_options=solver_options)
                    if cc != "optimal":
                        stt, cc = n.optimize.solve_model(solver_name="highs", solver_options={**solver_options, **FALLBACK})
                ok = cc == "optimal" or expr is None
                # the cost of the solution in PyPSA's convention (capex + opex, fixed costs included)
                c_ = float(n.statistics.capex().sum() + n.statistics.opex().sum()) if ok and expr is not None \
                    else (total if ok else np.nan)
                if ok:
                    tt, bc = capacities(n)
                else:
                    tt, bc = {k: np.nan for k in GROUPS}, {k: {} for k in GROUPS}
                with open(cpath, "a") as f:
                    for gg in GROUPS:
                        for c, v in bc[gg].items():
                            f.write(f"{e},{g},{sense},{gg},{c},{v}\n")
                row = [str(e), g, sense, f"{stt}:{cc}", str(c_), str(c_ / total - 1), str(tt[g]), f"{time.time() - tb:.1f}"]
                with open(mpath, "a") as f:
                    f.write(",".join(row) + "\n")
                rows[(e, g, sense)] = row
                log(f"  eps {e:.3f} {g:8s} {sense}  {stt}:{cc}  total {tt[g]:.4e} MW (optimum {tot[g]:.4e})  "
                    f"cost +{100 * (c_ / total - 1):.4f} %  [{time.time() - tb:.0f} s]")
    env = {}
    for e in eps:
        env[f"eps_{e:g}"] = {g: {"min_MW": float(rows[(e, g, "min")][6]), "max_MW": float(rows[(e, g, "max")][6]),
                                 "optimum_MW": tot[g], "status": [rows[(e, g, "min")][3], rows[(e, g, "max")][3]]}
                             for g in groups}
    toml_dump({"total_cost_eur_per_a": total, "eps": eps, "resolution": res, "envelope": env,
               "seconds_total": time.time() - t0,
               "maxrss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
              os.path.join(out, "summary_mga_pypsa.toml"))
    log(f"MGA done [{time.time() - t0:.0f} s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
