"""Cut a local copy of a global ERA5 monthly file (PIK climate_data_central layout,
``<variable>/<variable>_YYYYMM.nc``, 0.25 deg, lon 0..359.75, lat 90..-90) to the
European bounding box and write it as a compressed NetCDF4 file.

    python scripts/era5_subset.py IN.nc OUT.nc [--lon0 -10.5 --lon1 25.5 --lat0 40.5 --lat1 59.5]

The output has dimensions (time, lat, lon) with lat and lon increasing and lon in
-180..180; the data variable keeps its ERA5 short name (t2m, u10, v10, ssrd), its
units and float32 values. The input is only read; it is deleted by the caller
(scripts/fetch_era5.sh) after a successful write.
"""

import argparse
import os

import netCDF4
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--lon0", type=float, default=-10.5)
    p.add_argument("--lon1", type=float, default=25.5)
    p.add_argument("--lat0", type=float, default=40.5)
    p.add_argument("--lat1", type=float, default=59.5)
    a = p.parse_args()

    src = netCDF4.Dataset(a.inp)
    tname = "valid_time" if "valid_time" in src.variables else "time"
    names = [k for k, v in src.variables.items() if v.ndim == 3]
    assert len(names) == 1, f"expected one 3-d variable, got {names}"
    name = names[0]
    var = src.variables[name]
    lat = src.variables["latitude"][:].data
    lon = src.variables["longitude"][:].data
    lon180 = np.where(lon >= 180.0, lon - 360.0, lon)
    ilat = np.where((lat >= a.lat0 - 1e-6) & (lat <= a.lat1 + 1e-6))[0]
    ilon = np.where((lon180 >= a.lon0 - 1e-6) & (lon180 <= a.lon1 + 1e-6))[0]
    ilon = ilon[np.argsort(lon180[ilon])]          # west of 0 first
    ilat = ilat[np.argsort(lat[ilat])]             # south first
    # read in two contiguous index blocks per axis (netCDF4 slices must be monotone)
    west, east = ilon[lon180[ilon] < 0], ilon[lon180[ilon] >= 0]
    la0, la1 = ilat.min(), ilat.max() + 1
    parts = []
    for blk in (west, east):
        if len(blk):
            parts.append(var[:, la0:la1, blk.min():blk.max() + 1])
    data = np.concatenate([np.ma.filled(x.astype(np.float32), np.nan) for x in parts], axis=2)
    data = data[:, ::-1, :] if lat[la0] > lat[la1 - 1] else data   # lat increasing
    time = src.variables[tname][:]

    tmp = a.out + ".tmp"
    with netCDF4.Dataset(tmp, "w", format="NETCDF4") as dst:
        dst.setncattr("source", os.path.basename(a.inp))
        dst.setncattr("history", "cut to the European box by scripts/era5_subset.py")
        dst.createDimension("time", data.shape[0])
        dst.createDimension("lat", data.shape[1])
        dst.createDimension("lon", data.shape[2])
        t = dst.createVariable("time", "i8", ("time",))
        t.units = src.variables[tname].units
        t.calendar = getattr(src.variables[tname], "calendar", "proleptic_gregorian")
        t[:] = time
        y = dst.createVariable("lat", "f8", ("lat",))
        y.units = "degrees_north"
        y[:] = np.sort(lat[ilat])
        x = dst.createVariable("lon", "f8", ("lon",))
        x.units = "degrees_east"
        x[:] = np.sort(lon180[ilon])
        v = dst.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True, complevel=4, shuffle=True,
                               chunksizes=(min(744, data.shape[0]), data.shape[1], data.shape[2]))
        for k in ["units", "long_name", "GRIB_stepType", "GRIB_shortName"]:
            if k in var.ncattrs():
                v.setncattr(k, var.getncattr(k))
        v[:] = data
    src.close()
    os.replace(tmp, a.out)
    print(f"{a.out}: {name} {data.shape}, {np.isnan(data).sum()} NaN, "
          f"range {np.nanmin(data):.3g}..{np.nanmax(data):.3g}")


if __name__ == "__main__":
    main()
