"""Replace the NASA POWER weather of a model file by ERA5 (hourly, 0.25 deg).

    python scripts/prepare_era5.py [--base data/processed/model256.h5]
                                   [--out data/processed/model256_era5.h5] [--years 2003,2010,2013,2017]

Input: the European cut-outs written by scripts/fetch_era5.sh
(``data/raw/era5/europe/<variable>/<variable>_YYYYMM.nc``: 10 m wind u10/v10,
surface solar radiation downwards ssrd, 2 m temperature t2m) and the PyPSA-Eur
resources of the network (``external/pypsa-eur/resources/esm256``).

Everything except the ``weather`` group and the ``years`` attribute is copied from
``--base`` unchanged (network, generators, storage, brownfield fleets and also the
dispatch bounds kbar of gas and storage, which prepare_data.py derived from the
POWER peak load).

Mapping to the nodes (instead of one POWER point at the bus location):

* onwind, offwind, solar: capacity factors per ERA5 cell, averaged over the bus with
  the PyPSA-Eur availability matrix as weights (eligible area per cell and bus,
  ``availability_matrix_<tech>.nc`` on the 0.3 deg cutout grid, remapped
  conservatively to the 0.25 deg ERA5 grid; offwind = ac + dc + float);
* temperature (load model input): population weighted mean over the cells whose
  centre lies in the bus's onshore region (``pop_layout_total.nc``,
  ``onshore_regions.geojson``), nearest cell for regions without a cell centre.

Conversion as in prepare_data.py: the same smoothed power curve at 100 m (10 m
wind extrapolated with shear exponent 1/7), the same PV model
(cf = clip(s * GHI / 1000, 0, 1)), the scale s calibrated per generator by brentq
so that the mean over the network's snapshot window (2013) equals the PyPSA-Eur
p_max_pu mean, and the same OPSD-fitted national temperature-driven load models,
driven by node temperature and scaled to the network's nodal mean load over the
window. The window year (2013) must therefore be among the ERA5 years.

ERA5 conventions: ssrd is accumulated over the hour ending at valid_time (J/m^2);
it is divided by 3600 and labelled with the hour's start, like the instantaneous
fields. Feb 29 is dropped; every year has 8760 hours.
"""

import argparse
import importlib.util
import os
import sys
import time
import warnings

import geopandas as gpd
import h5py
import netCDF4
import numpy as np
import pandas as pd
import pypsa
import scipy.sparse as sp
import xarray as xr
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("prepare_data", os.path.join(HERE, "prepare_data.py"))
pdat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdat)

VARS = {"t2m": "2m_temperature", "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind",
        "ssrd": "surface_solar_radiation_downwards"}
WIND_SHEAR = 1 / 7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="data/processed/model256.h5")
    p.add_argument("--out", default="data/processed/model256_era5.h5")
    p.add_argument("--era5", default="data/raw/era5/europe")
    p.add_argument("--years", default="", help="comma separated; default: all complete years in --era5")
    p.add_argument("--network", default="external/pypsa-eur/resources/esm256/networks/composed_2050.nc")
    p.add_argument("--resources", default="external/pypsa-eur/resources/esm256")
    p.add_argument("--raw", default="data/raw", help="OPSD weather_data.csv and time_series_60min_singleindex.csv")
    p.add_argument("--countries", default=",".join(pdat.DEFAULT_COUNTRIES))
    p.add_argument("--force", action="store_true", help="overwrite --out")
    return p.parse_args()


# --------------------------------------------------------------------------
# ERA5 input
# --------------------------------------------------------------------------

def complete_years(root):
    years = None
    for v in VARS.values():
        d = os.path.join(root, v)
        have = set(os.listdir(d)) if os.path.isdir(d) else set()
        ys = {int(f[-9:-5]) for f in have if f.endswith(".nc")}
        ys = {y for y in ys if all(f"{v}_{y}{m:02d}.nc" in have for m in range(1, 13))}
        years = ys if years is None else years & ys
    return sorted(years or [])


def read_grid(root, year):
    with netCDF4.Dataset(os.path.join(root, VARS["t2m"], f"{VARS['t2m']}_{year}01.nc")) as d:
        return d["lat"][:].data.copy(), d["lon"][:].data.copy()


def read_year(root, var, year):
    """(8760, ncell) float32 and the hourly UTC index (Feb 29 dropped)."""
    name = VARS[var]
    arrs, times = [], []
    for m in range(1, 13):
        with netCDF4.Dataset(os.path.join(root, name, f"{name}_{year}{m:02d}.nc")) as d:
            x = d[var][:]
            arrs.append(np.ma.filled(x, np.nan).reshape(x.shape[0], -1).astype(np.float32))
            times.append(pd.to_datetime(d["time"][:].data, unit="s", utc=True))
    x, t = np.concatenate(arrs), times[0].append(times[1:])
    assert ((t[1:] - t[:-1]) == pd.Timedelta(hours=1)).all() and t[0] == pd.Timestamp(f"{year}-01-01", tz="UTC"), \
        f"{var} {year}: time axis not hourly-contiguous from Jan 1"
    if var == "ssrd":
        # accumulation over the hour ending at t -> mean flux of the hour starting at t - 1 h
        x = np.concatenate([x[1:], x[-1:]]) / 3600.0
    keep = ~((t.month == 2) & (t.day == 29))
    x, t = x[keep], t[keep]
    assert len(t) == 8760, f"{var} {year}: {len(t)} hours"
    assert not np.isnan(x).any(), f"{var} {year}: NaN"
    return x, t


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------

def overlap_1d(edges_a, edges_b):
    """Overlap lengths of intervals a_i x b_j (edges increasing)."""
    lo = np.maximum(edges_a[:-1, None], edges_b[None, :-1])
    hi = np.minimum(edges_a[1:, None], edges_b[None, 1:])
    return np.clip(hi - lo, 0, None)


def edges(c):
    d = np.diff(c).mean()
    return np.concatenate([c - d / 2, [c[-1] + d / 2]])


def remap(field_yx, ys, xs, lat, lon):
    """Conservative remap of an extensive quantity per cell (y, x) to the ERA5 grid (lat, lon)."""
    Oy = overlap_1d(edges(lat), edges(ys)) / np.diff(edges(ys))[None, :]
    Ox = overlap_1d(edges(lon), edges(xs)) / np.diff(edges(xs))[None, :]
    return Oy @ field_yx @ Ox.T


def availability_weights(resources, techs, buses, lat, lon):
    """Sparse (bus x cell) row-normalized weights: eligible area per ERA5 cell."""
    W = None
    for t in techs:
        a = xr.open_dataarray(os.path.join(resources, f"availability_matrix_{t}.nc"))
        ys, xs, A = a.y.values, a.x.values, np.nan_to_num(a.values)
        area = np.cos(np.deg2rad(ys))[:, None] * np.ones((1, len(xs)))   # relative cell area
        idx = {b: i for i, b in enumerate(a.bus.values)}
        rows = np.zeros((len(buses), len(lat) * len(lon)))
        for i, b in enumerate(buses):
            if b in idx:
                rows[i] = remap(A[idx[b]] * area, ys, xs, lat, lon).ravel()
        tot = np.array([(A[idx[b]] * area).sum() if b in idx else 0.0 for b in buses])
        lost = np.where(tot > 0, 1 - rows.sum(1) / np.where(tot > 0, tot, 1), 0.0)
        print(f"  weights {t}: {sum(b in idx for b in buses)} buses, largest share outside the box {lost.max():.3f}")
        W = rows if W is None else W + rows
    s = W.sum(1)
    assert (s > 0).all(), "bus without eligible ERA5 cell"
    return sp.csr_matrix(W / s[:, None])


def population_weights(resources, nodes, x, y, lat, lon):
    p = xr.open_dataarray(os.path.join(resources, "pop_layout_total.nc"))
    pop = remap(np.nan_to_num(p.values), p.y.values, p.x.values, lat, lon)
    LON, LAT = np.meshgrid(lon, lat)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(LON.ravel(), LAT.ravel()), crs="EPSG:4326")
    pts["cell"] = np.arange(len(pts))
    reg = gpd.read_file(os.path.join(resources, "onshore_regions.geojson"))[["name", "geometry"]]
    j = gpd.sjoin(pts, reg.to_crs("EPSG:4326"), predicate="within")
    W = np.zeros((len(nodes), len(lat) * len(lon)))
    pos = {n: i for i, n in enumerate(nodes)}
    for cell, name in zip(j.cell.values, j.name.values):
        if name in pos:
            W[pos[name], cell] = max(pop.ravel()[cell], 1e-6)   # unpopulated cells: tiny weight
    nearest = 0
    for i in range(len(nodes)):
        if W[i].sum() == 0:
            c = int(np.argmin(np.abs(lat - y[i]))) * len(lon) + int(np.argmin(np.abs(lon - x[i])))
            W[i, c] = 1.0
            nearest += 1
    print(f"  population weights: {len(nodes) - nearest} regions with cells, {nearest} by nearest cell")
    return sp.csr_matrix(W / W.sum(1)[:, None])


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------

def wind_cf10(ws10, s):
    """prepare_data.wind_cf for 10 m wind: shear 1/7 from 10 m to 100 m."""
    return pdat.wind_cf(ws10 * (50 / 10) ** WIND_SHEAR, s)


def solar_cf(ghi, s):
    return np.clip(s * ghi / 1000.0, 0, 1)


def calibrated_series(conv, X, w, target, window):
    """Weighted mean over cells of conv(X[:, c], s), s calibrated on the window.

    The window mean is not monotone in s for wind (cut-out at 25 m/s), so the
    smallest root on [0.2, 4] is bracketed on a grid first; prepare_data.calibrate
    tests f(4) < 0 first and then returns s = 4, deep in the cut-out regime, for
    nodes whose target needs a large scale (most POWER offwind nodes). Without a
    root the grid maximum of the window mean is used.
    """
    Xw = X[window]
    f = lambda s: (conv(Xw, s) @ w).mean() - target
    grid = np.linspace(0.2, 4.0, 39)
    fg = []
    for g in grid:                      # scan upwards to the first sign change
        fg.append(f(g))
        if fg[-1] >= 0:
            break
    j = len(fg) - 1
    if fg[j] < 0:
        s = grid[int(np.argmax(fg))]
    elif j == 0:
        s = grid[0]
    else:
        s = brentq(f, grid[j - 1], grid[j], xtol=1e-4)
    return (conv(X, s) @ w).astype(np.float32), s


def main():
    a = parse_args()
    t0 = time.time()
    if os.path.exists(a.out) and not a.force:
        sys.exit(f"{a.out} exists (use --force)")
    assert os.path.abspath(a.out) != os.path.abspath(a.base)
    years = [int(y) for y in a.years.split(",")] if a.years else complete_years(a.era5)
    assert years, f"no complete ERA5 year in {a.era5}"
    print(f"ERA5 years {years}")

    # ---- network: generator targets and nodal mean load, checked against the base file
    countries = a.countries.split(",")
    defaults = dict(co2_price=150.0, battery_hours=6.0, h2_hours=168.0, hydro_inflow_cf=0.3, ror_cf=0.5,
                    kbar_dispatch=1.5, dc_eta_static=0.98, dc_eta_per_1000km=0.977, thermal_avail=0.9)
    n = pypsa.Network(a.network)
    n.calculate_dependent_values()
    net, gens, stores, snaps = pdat.extract_network_elec(n, countries, argparse.Namespace(**defaults))
    pdat.completion_defaults(gens, defaults["co2_price"])
    gens = [c for c in gens if not c["brownfield"]]
    nodes = net["nodes"]
    with h5py.File(a.base, "r") as f:
        assert [s.decode() for s in f["network/nodes"][()]] == list(nodes), "node list differs from the base file"
        assert [s.decode() for s in f["generators/carrier"][()]] == [c["carrier"] for c in gens]
        assert np.allclose(f["network/mean_load"][()], net["mean_load"])
        base_w = {k: (f[f"weather/{k}/nodes"][()].astype(int), f[f"weather/{k}/data"][()]) for k in f["weather"]}
        base_years = list(f.attrs["years"])
    print(f"network: {len(nodes)} nodes, {len(gens)} generators  [{time.time() - t0:.0f}s]")

    lat, lon = read_grid(a.era5, years[0])
    x, y = np.asarray(net["x"]), np.asarray(net["y"])
    assert x.min() > lon.min() and x.max() < lon.max() and y.min() > lat.min() and y.max() < lat.max()

    # ---- weights
    kinds = {"onwind": ["onwind"], "offwind": ["offwind-ac", "offwind-dc", "offwind-float"], "solar": ["solar"]}
    rows = {k: [c for c in gens if c["weather"] == k] for k in kinds}
    for k, lst in rows.items():
        assert [nodes.index(c["node"]) for c in lst] == list(base_w[k][0]), f"{k}: node list differs"
    W = {k: availability_weights(a.resources, kinds[k], [c["node"] for c in rows[k]], lat, lon) for k in kinds}
    Wt = population_weights(a.resources, nodes, x, y, lat, lon)

    # ---- read ERA5: temperature reduced to nodes at once; wind speed and GHI on the used cells
    cells = {"wind": np.unique(np.concatenate([W["onwind"].indices, W["offwind"].indices])),
             "solar": np.unique(W["solar"].indices)}
    T, WS, GHI, index = [], [], [], None
    for yr in years:
        t2m, ti = read_year(a.era5, "t2m", yr)
        T.append(np.asarray(Wt @ t2m.T).T - 273.15)
        u, _ = read_year(a.era5, "u10", yr)
        v, _ = read_year(a.era5, "v10", yr)
        WS.append(np.hypot(u[:, cells["wind"]], v[:, cells["wind"]]))
        del u, v
        r, _ = read_year(a.era5, "ssrd", yr)
        GHI.append(r[:, cells["solar"]])
        del r
        index = ti if index is None else index.append(ti)
        print(f"  read {yr}  [{time.time() - t0:.0f}s]")
    T, WS, GHI = np.concatenate(T), np.concatenate(WS), np.concatenate(GHI)
    s0, s1 = pd.Timestamp(snaps[0], tz="UTC"), pd.Timestamp(snaps[-1], tz="UTC") + pd.Timedelta(hours=3)
    window = np.asarray((index >= s0) & (index < s1))
    assert window.sum() >= 8000, f"snapshot window {s0}..{s1} not covered by the ERA5 years {years}"

    def to_tensor(vals):
        out = np.full((len(years), 8760), np.nan, dtype=np.float32)
        for i, yr in enumerate(years):
            out[i] = vals[index.year == yr]
        return out

    # ---- load: OPSD national models driven by node temperature, per year (ewm restarts each year)
    temp = pdat.hourly(pd.read_csv(f"{a.raw}/weather_data.csv",
                                   usecols=["utc_timestamp"] + [f"{c}_temperature" for c in sorted(set(net["country"]))]))
    ts = pdat.hourly(pd.read_csv(f"{a.raw}/time_series_60min_singleindex.csv",
                                 usecols=["utc_timestamp"] + [f"{c}_load_actual_entsoe_transparency" for c in sorted(set(net["country"]))]))
    models = {}
    for c in sorted(set(net["country"])):
        models[c], rmse = pdat.load_model_fit(temp[f"{c}_temperature"], ts[f"{c}_load_actual_entsoe_transparency"])
        print(f"  load model {c}: relative rmse {rmse:.3f}")
    shape = np.zeros((len(index), len(nodes)))
    for yr in years:
        sel = np.asarray(index.year == yr)
        for i in range(len(nodes)):
            shape[sel, i] = models[net["country"][i]](index[sel], T[sel, i])
    load = shape * (np.asarray(net["mean_load"]) / shape[window].mean(axis=0))
    fields = {"load": (list(range(len(nodes))), np.stack([to_tensor(load[:, i]) for i in range(len(nodes))], -1))}
    print(f"  load: total mean {load.mean(0).sum() / 1e3:.1f} GW, peak {load.sum(1).max() / 1e3:.1f} GW")

    # ---- renewables
    scales = {}
    for k in kinds:
        X, conv, cidx = (WS, wind_cf10, cells["wind"]) if k != "solar" else (GHI, solar_cf, cells["solar"])
        pos = {c: j for j, c in enumerate(cidx)}
        cols, sc, miss = [], [], []
        for r, g in enumerate(rows[k]):
            lo, hi = W[k].indptr[r], W[k].indptr[r + 1]
            cj = np.array([pos[c] for c in W[k].indices[lo:hi]])
            series, s = calibrated_series(conv, X[:, cj], W[k].data[lo:hi], g["cf_target"], window)
            cols.append(to_tensor(series))
            sc.append(s)
            miss.append(float(series[window].mean() - g["cf_target"]))
        fields[k] = (list(base_w[k][0]), np.stack(cols, -1))
        scales[k] = np.array(sc)
        print(f"  {k}: {len(cols)} nodes, scale median {np.median(sc):.2f} [{np.min(sc):.2f}, {np.max(sc):.2f}], "
              f"at bounds {int(np.sum((np.array(sc) <= 0.2) | (np.array(sc) >= 4.0)))}, "
              f"|window mean - target| > 0.005 at {int(np.sum(np.abs(np.array(miss)) > 0.005))}, "
              f"long-term mean CF {fields[k][1].mean():.3f}  [{time.time() - t0:.0f}s]")
    for k, (_, arr) in fields.items():
        assert not np.isnan(arr).any(), f"missing data in {k}"

    # ---- write: base file without weather + ERA5 weather
    tmp = a.out + ".tmp"
    with h5py.File(a.base, "r") as src, h5py.File(tmp, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for g in src:
            if g != "weather":
                src.copy(src[g], dst, name=g)
        dst.attrs["years"] = np.array(years)
        dst.attrs["weather_source"] = ("ERA5 hourly 0.25 deg (u10, v10, ssrd, t2m), availability/population "
                                       "weighted per bus; scripts/prepare_era5.py")
        dst.attrs["weather_base"] = os.path.basename(a.base)
        for name, (nidx, arr) in fields.items():
            dst[f"weather/{name}/nodes"] = np.asarray(nidx, dtype=float)
            dst[f"weather/{name}/data"] = arr.astype(np.float32)
        for k, s in scales.items():
            dst[f"weather_calibration/{k}_scale"] = s
    os.replace(tmp, a.out)
    print(f"wrote {a.out}  [{time.time() - t0:.0f}s]")

    # ---- comparison with the POWER file
    yi = [base_years.index(yr) for yr in years if yr in base_years]
    i13 = base_years.index(2013) if 2013 in base_years else None
    e13 = years.index(2013) if 2013 in years else None
    rows_out = []
    for k in ["onwind", "offwind", "solar", "load"]:
        pb, pe = base_w[k][1], fields[k][1]
        r = dict(field=k, power_all=float(pb.mean()), power_same_years=float(pb[yi].mean()) if yi else np.nan,
                 era5=float(pe.mean()))
        if i13 is not None and e13 is not None:
            r["power_2013"], r["era5_2013"] = float(pb[i13].mean()), float(pe[e13].mean())
            # aggregate hourly series 2013: std and correlation
            ab, ae = pb[i13].sum(-1), pe[e13].sum(-1)
            r["corr_agg_2013"] = float(np.corrcoef(ab, ae)[0, 1])
            r["corr_node_med_2013"] = float(np.median([np.corrcoef(pb[i13][:, j], pe[e13][:, j])[0, 1]
                                                       for j in range(pb.shape[-1])]))
            r["cv_agg_power"], r["cv_agg_era5"] = float(ab.std() / ab.mean()), float(ae.std() / ae.mean())
        if k == "load":
            r["peak_power_GW"] = float(pb[yi].sum(-1).max() / 1e3) if yi else np.nan
            r["peak_era5_GW"] = float(pe.sum(-1).max() / 1e3)
        rows_out.append(r)
    cmp = pd.DataFrame(rows_out).set_index("field")
    print(cmp.round(3).to_string())
    csv = os.path.splitext(a.out)[0] + "_comparison.csv"
    cmp.to_csv(csv)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
