"""Build the model input file for the near-optimal sampler.

Sources (fetched by scripts/download_data.sh and by this script):

* PyPSA-Eur network ``carbon_management.nc`` (PyPSA example networks, 91 AC
  nodes, sector coupled, 2050 net-zero). Only its electricity layer is used,
  at the original node resolution: AC lines, DC links, technology costs,
  technical potentials and the fixed fleet (nuclear, run-of-river, hydro, PHS).
* NASA POWER hourly point data (MERRA-2 wind at 50 m, CERES irradiance, 2 m
  temperature) at every network node, 2001-2022.
* OPSD ``weather_data`` (population weighted national temperature) and OPSD
  ``time_series`` (ENTSO-E load 2015-2019), used to fit a national
  temperature-driven load model that is then driven by node temperatures.

Capacity factors are calibrated per generator so that their mean over the
network's snapshot window equals the mean of the PyPSA-Eur ``p_max_pu``.

Output: HDF5 with network arrays and per-field weather tensors
``weather/<field>/data[year, hour, node]`` (8760 h per year, Feb 29 dropped).
"""

import argparse
import concurrent.futures as cf
import io
import os
import time
import urllib.request
import warnings

import h5py
import numpy as np
import pandas as pd
import pypsa
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

DEFAULT_COUNTRIES = ["DE", "FR", "BE", "NL", "LU", "CH", "AT", "CZ", "PL", "DK"]
YEARS = list(range(2001, 2023))
POWER_URL = ("https://power.larc.nasa.gov/api/temporal/hourly/point?"
             "parameters=WS50M,ALLSKY_SFC_SW_DWN,T2M&community=RE&longitude={lon:.4f}"
             "&latitude={lat:.4f}&start={y0}0101&end={y1}1231&format=CSV&time-standard=UTC")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/processed/model.h5")
    p.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES))
    p.add_argument("--co2-price", type=float, default=150.0, help="EUR/t")
    p.add_argument("--battery-hours", type=float, default=6.0)
    p.add_argument("--h2-hours", type=float, default=168.0)
    p.add_argument("--hydro-inflow-cf", type=float, default=0.3,
                   help="constant reservoir inflow as fraction of turbine power")
    p.add_argument("--ror-cf", type=float, default=0.5)
    p.add_argument("--kbar-dispatch", type=float, default=1.5,
                   help="upper bound for gas/storage power, multiple of nodal peak load")
    p.add_argument("--augment", action="store_true",
                   help="only add the model-completion datasets (line losses, link efficiencies, existing "
                        "capacities, emission factors, brownfield fleets) to the existing --out file")
    p.add_argument("--dc-eta-static", type=float, default=0.98, help="DC link converter efficiency (both ends)")
    p.add_argument("--dc-eta-per-1000km", type=float, default=0.977, help="DC link efficiency per 1000 km")
    p.add_argument("--thermal-avail", type=float, default=0.9, help="availability of fixed thermal fleets")
    p.add_argument("--network", default="",
                   help="composed PyPSA-Eur electricity network (resources/<run>/networks/composed_<year>.nc); "
                        "default: the carbon_management example in --raw")
    return p.parse_args()


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------

def line_loss_aggregates(ln, agg_index):
    """Quadratic loss coefficient alpha (1/MW, loss = alpha f^2 in MW) and length per merged line pair.
    Parallel lines share the flow in proportion to b = 1/x: alpha = sum_i alpha_i (b_i / sum b)^2,
    alpha_i = r_i / v_nom_i^2 (r in Ohm, v in kV)."""
    ln = ln.copy()
    vn = ln.v_nom.replace(0.0, np.nan).fillna(380.0)
    ln["alpha"] = ln.r.fillna(0.0) / vn ** 2
    ln["w"] = ln.b / ln.groupby("pair").b.transform("sum")
    ln["aw"] = ln.alpha * ln.w ** 2
    ln["lw"] = ln.length.fillna(0.0) * ln.w
    g = ln.groupby("pair").agg(alpha=("aw", "sum"), length=("lw", "sum"))
    return g.reindex(agg_index)


def link_aggregates(dc, cap_index, a):
    """Efficiency and length per merged DC pair; lossless links get eta = static * per_1000km^(length/1000)."""
    if len(dc) == 0:
        return np.zeros(0), np.zeros(0)
    g = dc.groupby("pair").agg(eta=("efficiency", "mean"), length=("length", "mean")).reindex(cap_index)
    eta = g.eta.values.copy()
    L = g.length.fillna(0.0).values
    lossless = ~(eta < 0.999999)
    eta[lossless] = a.dc_eta_static * a.dc_eta_per_1000km ** (L[lossless] / 1000.0)
    return eta, L


def completion_defaults(gens, co2_price):
    """Fill the model-completion keys of every generator dict: existing capacity k0 (brownfield floor),
    old-vintage marginal cost mc0 and emission factors ef, ef0 (t CO2 / MWh_el)."""
    for c in gens:
        c.setdefault("ef", 0.0)
        c.setdefault("k0", 0.0)
        c.setdefault("mc0", c["mc"])
        c.setdefault("ef0", c["ef"])
        c.setdefault("brownfield", 0)
    return gens


def extract_network(n, countries, a):
    ac = n.buses[(n.buses.carrier == "AC") & n.buses.country.isin(countries)]
    nodes = list(ac.index)
    loc = n.buses.location.where(n.buses.location != "", n.buses.index)
    node_of = n.buses.index.to_series().map(loc)  # every bus -> its AC location

    def idx(bus):
        return bus.map(node_of)

    net = {"nodes": nodes, "x": ac.x.values, "y": ac.y.values,
           "country": ac.country.values}

    # ---- AC lines (parallel lines between the same pair are merged)
    ln = n.lines[n.lines.bus0.isin(nodes) & n.lines.bus1.isin(nodes) & (n.lines.s_nom_opt > 1.0)].copy()
    ln["pair"] = [tuple(sorted(p)) for p in zip(ln.bus0, ln.bus1)]
    ln["b"] = 1.0 / ln.x_pu_eff
    agg = ln.groupby("pair").agg(b=("b", "sum"), s=("s_nom_opt", "sum"))
    net["line_bus0"] = [nodes.index(p[0]) for p in agg.index]
    net["line_bus1"] = [nodes.index(p[1]) for p in agg.index]
    net["line_b"] = (agg.b / agg.b.mean()).values  # mean line has b = 1
    net["line_s"] = agg.s.values
    la = line_loss_aggregates(ln, agg.index)
    net["line_alpha"], net["line_length"] = la.alpha.fillna(0.0).values, la.length.fillna(0.0).values

    # ---- DC links, bidirectional capacity per pair
    dc = n.links[(n.links.carrier == "DC") & n.links.bus0.isin(nodes) & n.links.bus1.isin(nodes)
                 & (n.links.p_nom_opt > 1.0)].copy()
    dc["pair"] = [tuple(sorted(p)) for p in zip(dc.bus0, dc.bus1)]
    dc["fwd"] = [b0 == p[0] for b0, p in zip(dc.bus0, dc.pair)]
    cap = dc.groupby(["pair", "fwd"]).p_nom_opt.sum().unstack(fill_value=0.0).max(axis=1)
    net["link_bus0"] = [nodes.index(p[0]) for p in cap.index]
    net["link_bus1"] = [nodes.index(p[1]) for p in cap.index]
    net["link_cap"] = cap.values
    net["link_eta"], net["link_length"] = link_aggregates(dc, cap.index, a)

    # ---- generators, one per (node, carrier)
    pmax = n.get_switchable_as_dense("Generator", "p_max_pu")
    w_snap = n.snapshot_weightings.objective
    gas_price = n.generators[n.generators.carrier == "gas"].marginal_cost.mean()
    # emissions are booked on a CO2 bus in the sector network, so use the fuel factor directly
    gas_ef = 0.198  # t CO2 / MWh_th
    g = n.generators.copy()
    g["node"] = idx(g.bus)
    g = g[g.node.isin(nodes)]
    g["kind"] = g.carrier.replace({"solar rooftop": "solar"})
    gens = []
    for (nd, kind), df in g[g.kind.isin(["onwind", "offwind", "solar"])].groupby(["node", "kind"]):
        kmax = df.p_nom_max.replace(np.inf, np.nan).sum()
        if not kmax > 1.0:
            continue
        w = df.p_nom_max.replace(np.inf, 0.0)
        target = float((pmax[df.index].mul(w_snap, axis=0).sum() / w_snap.sum() * w).sum() / w.sum())
        gens.append(dict(node=nd, carrier=kind, weather=kind, invest=True, kbar=kmax, k=0.0,
                         capex=float((df.capital_cost * w).sum() / w.sum()),
                         mc=float(df.marginal_cost.mean()), avail=1.0, cf_target=target))
    for kind, cf_ in [("nuclear", 0.9), ("ror", a.ror_cf)]:
        for nd, df in g[g.kind == kind].groupby("node"):
            if df.p_nom_opt.sum() < 1.0:
                continue
            gens.append(dict(node=nd, carrier=kind, weather="", invest=False,
                             kbar=float(df.p_nom_opt.sum()), k=float(df.p_nom_opt.sum()),
                             capex=0.0, mc=float(df.marginal_cost.mean()), avail=cf_, cf_target=np.nan))
    oc = n.links[n.links.carrier == "OCGT"].copy()
    oc["node"] = idx(oc.bus1)
    for nd, df in oc[oc.node.isin(nodes)].groupby("node"):
        eff = df.efficiency.mean()
        gens.append(dict(node=nd, carrier="gas", weather="", invest=True, kbar=np.nan, k=0.0,
                         capex=float(df.capital_cost.mean() / eff),
                         mc=float((gas_price + gas_ef * a.co2_price) / eff + df.marginal_cost.mean()),
                         avail=1.0, cf_target=np.nan, ef=float(gas_ef / eff)))

    # ---- storage, one per (node, carrier)
    stores = []
    for kind, hours, chn, disn in [("battery", a.battery_hours, "battery charger", "battery discharger"),
                                   ("H2", a.h2_hours, "H2 Electrolysis", "H2 Fuel Cell")]:
        st = n.stores[n.stores.carrier == kind]
        ch = n.links[n.links.carrier == chn]
        dis = n.links[n.links.carrier == disn]
        for nd in nodes:
            s_c, ch_c, dis_c = st[idx(st.bus) == nd], ch[idx(ch.bus0) == nd], dis[idx(dis.bus1) == nd]
            if len(s_c) == 0 or len(ch_c) == 0 or len(dis_c) == 0:
                continue
            eta_out = dis_c.efficiency.mean()
            # discharger p_nom is on its input side: cost per MW_el out = cc / eta
            capex = ch_c.capital_cost.mean() + dis_c.capital_cost.mean() / eta_out \
                + hours * s_c.capital_cost.mean()
            stores.append(dict(node=nd, carrier=kind, invest=True, kbar=np.nan, k=0.0, hours=hours,
                               eta_in=ch_c.efficiency.mean(), eta_out=eta_out, capex=capex,
                               mc=float(ch_c.marginal_cost.mean()), inflow=0.0))
    su = n.storage_units.copy()
    su["node"] = idx(su.bus)
    for (nd, kind), df in su[su.node.isin(nodes)].groupby(["node", "carrier"]):
        p = df.p_nom_opt.sum()
        hours = float((df.max_hours * df.p_nom_opt).sum() / p) if p > 0 else 0.0
        if p < 1.0 or hours < 0.5:  # no usable energy capacity (e.g. LU PHS has max_hours = 0)
            continue
        stores.append(dict(node=nd, carrier=kind, invest=False, kbar=p, k=p,
                           hours=float((df.max_hours * df.p_nom_opt).sum() / p),
                           eta_in=float(df.efficiency_store.mean()),
                           eta_out=float(df.efficiency_dispatch.mean()), capex=0.0,
                           mc=float(df.marginal_cost.mean()),
                           inflow=a.hydro_inflow_cf if kind == "hydro" else 0.0))

    ld = n.loads[n.loads.carrier.isin(["electricity", "industry electricity", "agriculture electricity"])]
    p_set = n.get_switchable_as_dense("Load", "p_set")[ld.index]
    mean_load = (p_set.mul(w_snap, axis=0).sum() / w_snap.sum()).groupby(idx(ld.bus)).sum()
    net["mean_load"] = mean_load.reindex(nodes).fillna(0.0).values
    return net, gens, stores, n.snapshots


def extract_network_elec(n, countries, a):
    """Electricity layer of a composed (unsolved) PyPSA-Eur network: existing line and link
    capacities, renewable potentials and profiles, extendable gas, fixed nuclear/run-of-river,
    battery and H2 (store + links), PHS and reservoir hydro."""
    ac = n.buses[(n.buses.carrier == "AC") & n.buses.country.isin(countries)]
    nodes = list(ac.index)
    nodes_set = set(nodes)
    net = {"nodes": nodes, "x": ac.x.values, "y": ac.y.values, "country": ac.country.values}

    ln = n.lines[n.lines.bus0.isin(nodes_set) & n.lines.bus1.isin(nodes_set) & (n.lines.s_nom > 1.0)].copy()
    ln["pair"] = [tuple(sorted(p)) for p in zip(ln.bus0, ln.bus1)]
    ln["b"] = 1.0 / ln.x_pu_eff
    agg = ln.groupby("pair").agg(b=("b", "sum"), s=("s_nom", "sum"))
    net["line_bus0"] = [nodes.index(p[0]) for p in agg.index]
    net["line_bus1"] = [nodes.index(p[1]) for p in agg.index]
    net["line_b"] = (agg.b / agg.b.mean()).values
    net["line_s"] = agg.s.values
    la = line_loss_aggregates(ln, agg.index)
    net["line_alpha"], net["line_length"] = la.alpha.fillna(0.0).values, la.length.fillna(0.0).values

    dc = n.links[(n.links.carrier == "DC") & n.links.bus0.isin(nodes_set) & n.links.bus1.isin(nodes_set)
                 & (n.links.p_nom > 1.0)].copy()
    dc["pair"] = [tuple(sorted(p)) for p in zip(dc.bus0, dc.bus1)]
    dc["fwd"] = [b0 == p[0] for b0, p in zip(dc.bus0, dc.pair)]
    cap = dc.groupby(["pair", "fwd"]).p_nom.sum().unstack(fill_value=0.0).max(axis=1) if len(dc) else pd.Series(dtype=float)
    net["link_bus0"] = [nodes.index(p[0]) for p in cap.index]
    net["link_bus1"] = [nodes.index(p[1]) for p in cap.index]
    net["link_cap"] = cap.values
    net["link_eta"], net["link_length"] = link_aggregates(dc, cap.index, a)

    pmax = n.get_switchable_as_dense("Generator", "p_max_pu")
    w_snap = n.snapshot_weightings.objective
    gas_ef = 0.198
    g = n.generators[n.generators.bus.isin(nodes_set)].copy()
    kindmap = {"onwind": "onwind", "offwind-ac": "offwind", "offwind-dc": "offwind", "offwind-float": "offwind",
               "solar": "solar", "solar-hsat": "solar", "solar rooftop": "solar",
               "OCGT": "gas", "CCGT": "gas", "nuclear": "nuclear", "ror": "ror"}
    g["kind"] = g.carrier.map(kindmap)
    gens = []
    for (nd, kind), df in g[g.kind.isin(["onwind", "offwind", "solar"])].groupby(["bus", "kind"]):
        kmax = df.p_nom_max.replace(np.inf, np.nan).sum()
        if not kmax > 1.0:
            continue
        w = df.p_nom_max.replace(np.inf, 0.0)
        target = float((pmax[df.index].mul(w_snap, axis=0).sum() / w_snap.sum() * w).sum() / w.sum())
        gens.append(dict(node=nd, carrier=kind, weather=kind, invest=True, kbar=kmax, k=0.0,
                         capex=float((df.capital_cost * w).sum() / w.sum()),
                         mc=float(df.marginal_cost.mean()), avail=1.0, cf_target=target,
                         k0=float(df.p_nom_min.sum())))
    for kind, cf_ in [("nuclear", 0.9), ("ror", a.ror_cf)]:
        for nd, df in g[g.kind == kind].groupby("bus"):
            if df.p_nom.sum() < 1.0:
                continue
            gens.append(dict(node=nd, carrier=kind, weather="", invest=False, kbar=float(df.p_nom.sum()),
                             k=float(df.p_nom.sum()), capex=0.0, mc=float(df.marginal_cost.mean()),
                             avail=cf_, cf_target=np.nan))
    ocgt = g[(g.carrier == "OCGT") & g.p_nom_extendable]
    ref = ocgt if len(ocgt) else g[g.kind == "gas"]
    capex_gas = float(ref.capital_cost.mean()) if len(ref) else 50000.0
    eff_gas = float(ref.efficiency.mean()) if len(ref) else 0.4
    mc_gas = float(ref.marginal_cost.mean()) + gas_ef * a.co2_price / eff_gas if len(ref) else 150.0
    # existing gas plants (CCGT and OCGT with p_nom_min) form the old vintage of the merged gas unit
    ex = g[(g.kind == "gas") & (g.p_nom_min > 0)].copy()
    ex["mc_tot"] = ex.marginal_cost + gas_ef * a.co2_price / ex.efficiency
    ex["ef_el"] = gas_ef / ex.efficiency
    for nd in nodes:
        e = ex[ex.bus == nd]
        k0 = float(e.p_nom_min.sum())
        w = e.p_nom_min / k0 if k0 > 0 else None
        gens.append(dict(node=nd, carrier="gas", weather="", invest=True, kbar=np.nan, k=0.0,
                         capex=capex_gas, mc=mc_gas, avail=1.0, cf_target=np.nan, ef=gas_ef / eff_gas,
                         k0=k0, mc0=float((e.mc_tot * w).sum()) if k0 > 0 else mc_gas,
                         ef0=float((e.ef_el * w).sum()) if k0 > 0 else gas_ef / eff_gas))
    # fixed thermal fleets, used only with brownfield = true
    co2c = n.carriers.co2_emissions
    for kind in ["coal", "lignite", "oil", "biomass", "waste", "geothermal"]:
        for nd, df in g[g.carrier == kind].groupby("bus"):
            p = float(df.p_nom.sum())
            if p < 1.0:
                continue
            eff = float((df.efficiency * df.p_nom).sum() / p)
            ef = float(co2c.get(kind, 0.0)) / max(eff, 1e-3)
            mcv = float((df.marginal_cost * df.p_nom).sum() / p)
            gens.append(dict(node=nd, carrier=kind, weather="", invest=False, kbar=p, k=p, capex=0.0,
                             mc=mcv + ef * a.co2_price, avail=a.thermal_avail, cf_target=np.nan,
                             ef=ef, k0=p, brownfield=1))

    stores = []
    lk = n.links.copy()
    lk["c"] = lk.carrier.str.lower()
    for kind, hours, chn, disn in [("battery", a.battery_hours, "battery charger", "battery discharger"),
                                   ("H2", a.h2_hours, "h2 electrolysis", "h2 fuel cell")]:
        st = n.stores[n.stores.carrier == kind]
        ch = lk[lk.c == chn]
        dis = lk[lk.c == disn]
        if len(st) == 0 or len(ch) == 0 or len(dis) == 0:
            continue
        eta_out = float(dis.efficiency.mean())
        capex = float(ch.capital_cost.mean() + dis.capital_cost.mean() / eta_out + hours * st.capital_cost.mean())
        for nd in nodes:
            if not ((ch.bus0 == nd).any() and (dis.bus1 == nd).any()):
                continue
            stores.append(dict(node=nd, carrier=kind, invest=True, kbar=np.nan, k=0.0, hours=hours,
                               eta_in=float(ch.efficiency.mean()), eta_out=eta_out, capex=capex,
                               mc=float(ch.marginal_cost.mean()), inflow=0.0))
    su = n.storage_units[n.storage_units.bus.isin(nodes_set)]
    for (nd, kind), df in su[su.carrier.isin(["PHS", "hydro"])].groupby(["bus", "carrier"]):
        p = df.p_nom.sum()
        hours = float((df.max_hours * df.p_nom).sum() / p) if p > 0 else 0.0
        if p < 1.0 or hours < 0.5:
            continue
        stores.append(dict(node=nd, carrier=kind, invest=False, kbar=p, k=p, hours=hours,
                           eta_in=float(df.efficiency_store.mean()), eta_out=float(df.efficiency_dispatch.mean()),
                           capex=0.0, mc=float(df.marginal_cost.mean()),
                           inflow=a.hydro_inflow_cf if kind == "hydro" else 0.0))

    ld = n.loads[n.loads.bus.isin(nodes_set)]
    p_set = n.get_switchable_as_dense("Load", "p_set")[ld.index]
    mean_load = (p_set.mul(w_snap, axis=0).sum() / w_snap.sum()).groupby(ld.bus).sum()
    net["mean_load"] = mean_load.reindex(nodes).fillna(0.0).values
    return net, gens, stores, n.snapshots


# --------------------------------------------------------------------------
# weather
# --------------------------------------------------------------------------

def fetch_power(node, lon, lat, cache):
    # keyed by location: node names repeat across networks (e.g. "DE0 0") at different places
    path = os.path.join(cache, f"{lon:.4f}_{lat:.4f}.csv")
    legacy = os.path.join(cache, node.replace(" ", "_") + ".csv")
    if not os.path.exists(path) and os.path.exists(legacy):
        head = open(legacy).read(4000)
        if f"{lat:.4f}" in head and f"{lon:.4f}" in head:      # legacy file for the same location
            os.rename(legacy, path)
    if not os.path.exists(path):
        url = POWER_URL.format(lon=lon, lat=lat, y0=YEARS[0], y1=YEARS[-1])
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=300) as r:
                    txt = r.read().decode()
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"POWER download failed for {node}")
        with open(path, "w") as f:
            f.write(txt)
    body = open(path).read().split("-END HEADER-")[1]
    df = pd.read_csv(io.StringIO(body))
    t = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY, hour=df.HR), utc=True)
    df = df.set_index(t)[["WS50M", "ALLSKY_SFC_SW_DWN", "T2M"]].replace(-999.0, np.nan)
    df = df.interpolate(limit_direction="both")
    return df[~((df.index.month == 2) & (df.index.day == 29))]


def wind_cf(ws50, alpha):
    """Smoothed generic power curve at hub height (100 m), shear exponent 1/7."""
    v = alpha * ws50 * (100 / 50) ** (1 / 7)
    grid = np.linspace(0, 40, 801)
    pc = np.clip((grid**3 - 3.0**3) / (12.0**3 - 3.0**3), 0, 1)
    pc[grid >= 25] = 0.0
    pc = gaussian_filter1d(pc, sigma=1.5 / (grid[1] - grid[0]))  # spread of the fleet
    return np.interp(v, grid, pc)


def calibrate(series_fn, target, window):
    f = lambda s: series_fn(s)[window].mean() - target
    lo, hi = 0.2, 4.0
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return hi
    return brentq(f, lo, hi)


def hourly(df, col="utc_timestamp"):
    t = pd.to_datetime(df[col], utc=True)
    df = df.drop(columns=[col]).set_index(t)
    return df[~((df.index.month == 2) & (df.index.day == 29))]


def load_model_fit(temp, load):
    """load/mean ~ hour-of-week x seasonal harmonics + heating/cooling degrees."""

    def design(index, T):
        Ts = pd.Series(np.asarray(T), index=index).ewm(halflife=24).mean().values
        how = np.asarray(index.dayofweek * 24 + index.hour)
        doy = 2 * np.pi * np.asarray(index.dayofyear) / 365.0
        H = np.zeros((len(index), 168))
        H[np.arange(len(index)), how] = 1.0
        return np.hstack([H, H * np.cos(doy)[:, None], H * np.sin(doy)[:, None],
                          np.maximum(15.0 - Ts, 0)[:, None], np.maximum(Ts - 22.0, 0)[:, None]])

    common = load.dropna().index.intersection(temp.dropna().index)
    common = common[(common.year >= 2015) & (common.year <= 2019)]
    y = (load / load.loc[common].mean()).loc[common].values
    X = design(common, temp.loc[common])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rmse = np.sqrt(np.mean((y - X @ beta) ** 2))
    return (lambda index, T: design(index, T) @ beta), rmse


def to_tensor(values, index):
    out = np.full((len(YEARS), 8760), np.nan)
    for i, y in enumerate(YEARS):
        v = values[index.year == y]
        out[i, : len(v)] = v[:8760]
    return out


def build_weather(raw, net, gens, snaps):
    nodes = net["nodes"]
    cache = os.path.join(raw, "power")
    os.makedirs(cache, exist_ok=True)
    print(f"fetching NASA POWER for {len(nodes)} nodes ...")
    with cf.ThreadPoolExecutor(4) as ex:
        dfs = list(ex.map(lambda i: fetch_power(nodes[i], net["x"][i], net["y"][i], cache), range(len(nodes))))
    index = dfs[0].index
    s0, s1 = pd.Timestamp(snaps[0], tz="UTC"), pd.Timestamp(snaps[-1], tz="UTC") + pd.Timedelta(hours=3)
    window = np.asarray((index >= s0) & (index < s1))

    countries = sorted(set(net["country"]))
    temp = hourly(pd.read_csv(f"{raw}/weather_data.csv",
                              usecols=["utc_timestamp"] + [f"{c}_temperature" for c in countries]))
    ts = hourly(pd.read_csv(f"{raw}/time_series_60min_singleindex.csv",
                            usecols=["utc_timestamp"] + [f"{c}_load_actual_entsoe_transparency" for c in countries]))
    models = {}
    for c in countries:
        models[c], rmse = load_model_fit(temp[f"{c}_temperature"], ts[f"{c}_load_actual_entsoe_transparency"])
        print(f"  load model {c}: relative rmse {rmse:.3f}")

    fields = {}
    shape = np.zeros((len(index), len(nodes)))
    for i in range(len(nodes)):
        shape[:, i] = models[net["country"][i]](index, dfs[i]["T2M"].values)
    # the network's load level refers to its snapshot window
    load = shape * (net["mean_load"] / shape[window].mean(axis=0))
    fields["load"] = (list(range(len(nodes))), np.stack([to_tensor(load[:, i], index) for i in range(len(nodes))], -1))
    print(f"  total mean load {load.mean(axis=0).sum() / 1e3:.1f} GW, peak {load.sum(axis=1).max() / 1e3:.1f} GW")

    for kind in ["onwind", "offwind", "solar"]:
        cols, nidx, means = [], [], []
        for g in gens:
            if g["weather"] != kind:
                continue
            i = nodes.index(g["node"])
            df = dfs[i]
            if kind == "solar":
                fn = lambda s, d=df: np.clip(s * d["ALLSKY_SFC_SW_DWN"].values / 1000.0, 0, 1)
            else:
                fn = lambda s, d=df: wind_cf(d["WS50M"].values, s)
            series = fn(calibrate(fn, g["cf_target"], window))
            means.append(series.mean())
            cols.append(to_tensor(series, index))
            nidx.append(i)
        fields[kind] = (nidx, np.stack(cols, -1))
        print(f"  {kind}: {len(nidx)} nodes, long-term mean CF {np.mean(means):.3f}")
    for f_, (_, arr) in fields.items():
        assert not np.isnan(arr).any(), f"missing data in {f_}"
    return fields


COMPLETION_GEN = ["k0", "mc0", "ef", "ef0"]


def write_completion(f, net, gens, bfg, nodes):
    """Model-completion datasets: line losses, link efficiencies, vintages and emissions, brownfield fleets."""
    for k in ["line_alpha", "line_length", "link_eta", "link_length"]:
        if f"network/{k}" in f:
            del f[f"network/{k}"]
        f[f"network/{k}"] = np.asarray(net[k], dtype=float)
    gdf = pd.DataFrame(gens)
    for col in COMPLETION_GEN:
        if f"generators/{col}" in f:   # a fresh file has no generators group yet
            del f[f"generators/{col}"]
        f[f"generators/{col}"] = gdf[col].values.astype(float)
    if "brownfield" in f:
        del f["brownfield"]
    if len(bfg):
        b = pd.DataFrame(bfg)
        b["bus"] = [nodes.index(c) for c in b.node]
        for col in ["bus", "invest", "kbar", "k", "capex", "mc", "avail"] + COMPLETION_GEN:
            f[f"brownfield/generators/{col}"] = b[col].values.astype(float)
        f["brownfield/generators/carrier"] = b.carrier.values.astype("S")


def augment_file(path, net, gens):
    """Add the completion datasets to an existing model file without touching its other contents."""
    bfg = [c for c in gens if c["brownfield"]]
    gens = [c for c in gens if not c["brownfield"]]
    with h5py.File(path, "r+") as f:
        nodes = [x.decode() for x in f["network/nodes"][()]]
        assert nodes == list(net["nodes"]), "node list differs from the file"
        assert len(f["network/line_s"]) == len(net["line_s"]) and len(f["network/link_cap"]) == len(net["link_cap"])
        assert np.allclose(f["network/line_s"][()], net["line_s"])
        car = [x.decode() for x in f["generators/carrier"][()]]
        bus = f["generators/bus"][()].astype(int)
        assert car == [c["carrier"] for c in gens], "generator rows differ from the file"
        assert list(bus) == [nodes.index(c["node"]) for c in gens]
        write_completion(f, net, gens, bfg, nodes)
    g = pd.DataFrame(gens)
    print(g.groupby("carrier").agg(k0=("k0", "sum"), ef=("ef", "mean"), ef0=("ef0", "mean"), mc0=("mc0", "mean")).to_string())
    if bfg:
        print(pd.DataFrame(bfg).groupby("carrier").agg(n=("node", "size"), k=("k", "sum"), ef=("ef", "mean"), mc=("mc", "mean")).to_string())
    print(f"line alpha*s: median {np.median(np.asarray(net['line_alpha']) * np.asarray(net['line_s'])):.4f}; "
          f"link eta: {np.round(net['link_eta'], 3)}")
    print(f"augmented {path}")


def main():
    a = parse_args()
    countries = a.countries.split(",")
    if a.network:
        n = pypsa.Network(a.network)
        n.calculate_dependent_values()
        net, gens, stores, snaps = extract_network_elec(n, countries, a)
    else:
        n = pypsa.Network(f"{a.raw}/carbon_management.nc")
        n.calculate_dependent_values()
        net, gens, stores, snaps = extract_network(n, countries, a)
    completion_defaults(gens, a.co2_price)
    if a.augment:
        augment_file(a.out, net, gens)
        return
    bfg = [c for c in gens if c["brownfield"]]
    gens = [c for c in gens if not c["brownfield"]]
    fields = build_weather(a.raw, net, gens, snaps)

    nodes = net["nodes"]
    peak = fields["load"][1].max(axis=(0, 1))
    for lst in (gens, stores):
        for c in lst:
            if np.isnan(c["kbar"]):
                c["kbar"] = a.kbar_dispatch * peak[nodes.index(c["node"])]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with h5py.File(a.out, "w") as f:
        f.attrs["source_network"] = n.name
        f.attrs["countries"] = ",".join(countries)
        f.attrs["years"] = np.array(YEARS)
        f.attrs["co2_price"] = a.co2_price
        f["network/nodes"] = np.array(nodes, dtype="S")
        f["network/country"] = np.array(net["country"], dtype="S")
        for k in ["x", "y", "mean_load", "line_bus0", "line_bus1", "line_b", "line_s",
                  "link_bus0", "link_bus1", "link_cap"]:
            f[f"network/{k}"] = np.asarray(net[k], dtype=float)
        write_completion(f, net, gens, bfg, nodes)
        gdf, sdf = pd.DataFrame(gens), pd.DataFrame(stores)
        gdf["bus"] = [nodes.index(c) for c in gdf.node]
        sdf["bus"] = [nodes.index(c) for c in sdf.node]
        for col in ["bus", "invest", "kbar", "k", "capex", "mc", "avail"]:
            f[f"generators/{col}"] = gdf[col].values.astype(float)
        f["generators/carrier"] = gdf.carrier.values.astype("S")
        for col in ["bus", "invest", "kbar", "k", "capex", "mc", "hours", "eta_in", "eta_out", "inflow"]:
            f[f"storage/{col}"] = sdf[col].values.astype(float)
        f["storage/carrier"] = sdf.carrier.values.astype("S")
        for name, (nidx, arr) in fields.items():
            f[f"weather/{name}/nodes"] = np.asarray(nidx, dtype=float)
            f[f"weather/{name}/data"] = arr.astype(np.float32)
    print(pd.DataFrame(gens).groupby("carrier").agg(n=("node", "size"), kbar=("kbar", "sum"), k=("k", "sum"),
                                                     capex=("capex", "mean"), mc=("mc", "mean")).to_string())
    print(pd.DataFrame(stores).groupby("carrier").agg(n=("node", "size"), kbar=("kbar", "sum"),
                                                      hours=("hours", "mean"), capex=("capex", "mean")).to_string())
    print(f"{len(nodes)} nodes, {len(net['line_s'])} lines, {len(net['link_cap'])} DC links")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
