#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, glob, sys, numbers
sys.path.append("../")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator
import pygeode as pyg
from jfunctions import *
import scipy.interpolate as intp
from scipy.ndimage import uniform_filter1d
import xarray as xr
import dask
from wavenumber_frequency_functions import *
from matplotlib import colors
from tropopause import *

save = True
save_loc = '/home/cwp29/tmp/mpas/crf_on/'

# Avoid giant Dask chunks when selecting latitude/longitude subsets from full lat-lon fields.
dask.config.set({'array.slicing.split_large_chunks': True})


# In[2]:


##### config_microp_re on/off test #####
#root_dir = '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/'
#data_locs = ['spinup/latlon_05deg.nc', 'microp_re_test/latlon_regridded_05deg.nc']
#sim_names = ['default', 'microp_re']

##### CRF on/off test #####
#root_dir = '/store/atmos-adk33/cwp29/mpas-a/data/qu_120km/'
#data_locs = ['control', 'crf_test']
#latlon_file = ['', 'latlon_spinup.nc']
#sim_names = ['crf on', 'crf off']

root_dir = '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/'
data_locs = ['control', 'u_nudge', 'T_nudge']
latlon_file = ['latlon.2000-*.nc', 'latlon.2000-*.nc', 'latlon.2000-*.nc']
zm_latlon_file = ['zm_latlon.2000-*.nc', 'zm_latlon.2000-*.nc', 'zm_latlon.2000-*.nc']
sim_names = ['control', 'nudged u', 'nudged T']

cols = ['k', 'b', 'r']

# Use ndays of data ending at a fixed analysis date
ndays = 200
dt = 6
analysis_end_date = '2000-11-21'

time_format = 'time' #or 'int'


# In[3]:


def load_latlon_dataset(
    data_dir,
    latlon_pattern="latlon.2000-*.nc",
    diag_file="latlon_diag.nc",
    hist_file="latlon_hist.nc",
    cldfrac_file="cldfrac.nc",
    time_format="int",
    dt=6,
    ndays=None,
    end_date=None,
    chunks=None,
    prefer_pattern=False,
    add_cldfrac_file=True,
):
    diag_path = os.path.join(data_dir, diag_file)
    hist_path = os.path.join(data_dir, hist_file)
    cldfrac_path = os.path.join(data_dir, cldfrac_file)
    latlon_files = sorted(glob.glob(os.path.join(data_dir, latlon_pattern)))

    # Open lazily; rechunk later if needed.  For pre-computed zonal means,
    # prefer the explicit zm_latlon pattern over any latlon_diag/latlon_hist files
    # that may also exist in the same directory.
    if prefer_pattern and latlon_files:
        print(data_dir, latlon_pattern)
        ds = xr.open_mfdataset(latlon_files, combine="by_coords", chunks={}, data_vars="minimal", coords="minimal", compat="override")
        print(ds)

    elif os.path.exists(diag_path) and os.path.exists(hist_path):
        dsd = xr.open_dataset(diag_path, chunks={})
        dsh = xr.open_dataset(hist_path, chunks={})

        if time_format == "int":
            dsd = dsd.isel(Time=slice(0, None, 2))
        else:
            dsd = dsd.sel(Time=dsh.Time)

        ds = xr.merge([dsh, dsd], join="exact", compat="override")

    elif latlon_files:
        print(data_dir, latlon_pattern)
        ds = xr.open_mfdataset(latlon_files, combine="by_coords", chunks={}, data_vars="minimal", coords="minimal", compat="override")
        print(ds)

    else:
        raise FileNotFoundError(
            f"Could not find either {latlon_pattern} or both {diag_file} and {hist_file} in {data_dir}"
        )

    # Trim early, before adding extra variables
    if ndays is not None:
        if time_format == "int":
            ntime = ndays * int(24 / dt)
            ds = ds.isel(Time=slice(-ntime, None))
        else:
            t_end = pd.to_datetime(end_date) if end_date is not None else pd.to_datetime(ds.Time.max().item())
            t_start = t_end - pd.Timedelta(days=ndays)
            ds = ds.sel(Time=slice(t_start, t_end))

    if add_cldfrac_file and "cldfrac" not in ds.data_vars and os.path.exists(cldfrac_path):
        ds_cld = xr.open_dataset(cldfrac_path, chunks={})
    
        if "cldfrac" in ds_cld:
            # Align cldfrac to ds.Time; missing times become NaN.
            cldfrac = ds_cld["cldfrac"].reindex(Time=ds.Time)
    
            # Standardise the model-level dimension name to "z".
            if "nVertLevels" in cldfrac.dims:
                cldfrac = cldfrac.rename({"nVertLevels": "z"})
            elif "nVertLevelsP1" in cldfrac.dims:
                raise ValueError("cldfrac appears to be on interface levels, not mid-levels")
            elif "z" not in cldfrac.dims:
                raise ValueError(f"Could not find vertical dimension in cldfrac dims {cldfrac.dims}")
    
            # Replace cldfrac.z with a geometric-height coordinate.
            #
            # zgrid in the full latlon files is interface height:
            #     zgrid(nVertLevelsP1, latitude, longitude)
            #
            # A dimension coordinate must be 1D, so use the horizontal-mean
            # mid-level height as the z coordinate.
            if "zgrid" not in ds.data_vars:
                raise ValueError("Cannot assign geometric z coordinate to cldfrac: zgrid is missing")
    
            zgrid_mean = ds.zgrid.mean(dim=("latitude", "longitude"))
    
            z_lower = zgrid_mean.isel(nVertLevelsP1=slice(0, -1)).values
            z_upper = zgrid_mean.isel(nVertLevelsP1=slice(1, None)).values
            zmid = 0.5 * (z_lower + z_upper)
    
            if cldfrac.sizes["z"] != zmid.size:
                raise ValueError(
                    f"cldfrac has {cldfrac.sizes['z']} vertical levels, "
                    f"but zgrid implies {zmid.size} mid-levels"
                )
    
            cldfrac = cldfrac.assign_coords(z=("z", zmid))
            cldfrac["z"].attrs["units"] = "m"
            cldfrac["z"].attrs["long_name"] = "geometric height of layer midpoints"
    
            ds = ds.assign(cldfrac=cldfrac)

    # Rechunk only at the end
    if chunks is not None:
        ds = ds.chunk(chunks)

    return ds


# In[4]:


dss = []
dss_zm = []

pres = np.array([100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000], dtype=np.float64)
pres /= 100.
pres2 = np.array([
    5000, 7000, 10000, 15000, 17000, 20000, 25000, 27000,
    30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000,
    70000, 75000, 80000, 85000, 92500, 100000
], dtype=np.float64)
pres2 /= 100.

pres_levels = [pres, pres2, pres2]

needed_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw",
    "uzonal_isobaric", "umeridional_isobaric", "omg_isobaric",
    "rh_isobaric", "t_isobaric", "qv_isobaric", "theta",
    "uReconstructZonal", "cldfrac", "olrtoa", "zgrid", "pressure", "vorticity", "ke",
    # Extra 2-D variables for convective/radiative mechanism diagnostics.
    # These are much cheaper than adding full 4-D qv/rho, so load them by default.
    "cape", "cin", "lcl", "lfc", "lh", "hfx",
    "lwdnt", "lwdnb", "lwupb", "lwupt", "swdnt", "swdnb", "swupb", "swupt",
]

check_vars = ['t_isobaric']

# Chunk full latitude-longitude fields spatially as well as in time.
# This avoids loading an entire global field when a diagnostic only needs the tropics.
latlon_chunks = {"Time": 8, "latitude": 60, "longitude": 120}
zm_chunks = {"Time": -1}


def get_time_values(ds):
    """Return a plotting-friendly time coordinate for this dataset only."""
    if issubclass(ds.Time.values.dtype.type, numbers.Integral):
        return ds.Time.values * dt / 24.0
    return ds.Time.values

def vertical_dim(da):
    """Find the model-level vertical dimension in either full lat/lon or zm_latlon data."""
    for dim in ("z", "nVertLevels", "nVertLevelsP1"):
        if dim in da.dims:
            return dim
    raise ValueError(f"No known model-level vertical dimension found in {da.dims!r}")

def midlevel_height_from_latlon(ds, latitude_sel=None, mean_latitude=False):
    """
    Geometric midlevel height from full latlon zgrid.

    From the latlon ncdump, zgrid has dimensions
    (nVertLevelsP1, latitude, longitude), with no Time dimension.
    This is used only for variables that are unavailable in zm_latlon
    and therefore still need an on-the-fly zonal mean from the full latlon files.
    """
    zgrid = ds.zgrid
    if latitude_sel is not None:
        zgrid = zgrid.sel(latitude=latitude_sel)

    zgrid = zgrid.mean(dim="longitude")
    if mean_latitude:
        zgrid = zgrid.mean(dim="latitude")

    # Use raw arrays for the midpoint so xarray does not align face-level
    # coordinates (0..N-1 with 1..N), which would silently drop the first
    # and last interfaces.
    lower = zgrid.isel(nVertLevelsP1=slice(0, -1)).values
    upper = zgrid.isel(nVertLevelsP1=slice(1, None)).values
    zmid_values = 0.5 * (lower + upper)

    if mean_latitude:
        return xr.DataArray(
            zmid_values,
            dims=("z",),
            coords={"z": np.arange(zmid_values.shape[0])},
            name="zmid",
        )

    return xr.DataArray(
        zmid_values,
        dims=("z", "latitude"),
        coords={
            "z": np.arange(zmid_values.shape[0]),
            "latitude": zgrid.latitude,
        },
        name="zmid",
    )


def height_from_zm_latlon(zds):
    """
    Geometric model-level height from precomputed zm_latlon files.

    From the zm_latlon ncdump, model-level zonal-mean variables use
    dimensions (Time, z, latitude), and z is the matching 1D geometric
    height coordinate.
    """
    return zds["z"]


def model_lat_field(da):
    """Return a DataArray ordered as (z, latitude) for latitude-height plots."""
    lev_dim = vertical_dim(da)
    if lev_dim != "z":
        da = da.rename({lev_dim: "z"})
    return da.transpose("z", "latitude")

def lat_height_mesh(field, zmid, lat):
    """Build 2D latitude and geometric-height arrays matching a (z, latitude) field."""
    field = model_lat_field(field)
    zmid = model_lat_field(zmid) if "latitude" in zmid.dims else zmid.rename({vertical_dim(zmid): "z"})
    lat2d = np.broadcast_to(lat[None, :], field.shape)
    zvals = zmid.values
    if zvals.ndim == 1:
        zvals = np.broadcast_to(zvals[:, None], field.shape)
    return field, lat2d, zvals

def standardize_latlon_dataset(ds, needed_vars=None, pres_values=None):
    if needed_vars is not None:
        keep = [v for v in needed_vars if v in ds.data_vars]
        ds = ds[keep]

    rename_map = {}
    if "nIsoLevelsU" in ds.dims:
        rename_map["nIsoLevelsU"] = "pres"
    if "nIsoLevelsT" in ds.dims:
        rename_map["nIsoLevelsT"] = "pres"
    if "nIsoLevelsZ" in ds.dims:
        rename_map["nIsoLevelsZ"] = "pres"
    if rename_map:
        ds = ds.rename(rename_map)

    if "pres" in ds.dims:
        if pres_values is None:
            raise ValueError("pres_values must be provided for datasets with a pres dimension")

        if ds.sizes["pres"] != len(pres_values):
            raise ValueError(
                f"Unexpected pressure axis length: {ds.sizes['pres']} "
                f"(expected {len(pres_values)})"
            )

        ds = ds.assign_coords(pres=("pres", pres_values))
        ds["pres"].attrs["units"] = "hPa"
        ds["pres"].attrs["positive"] = "down"
        ds["pres"].attrs["axis"] = "Z"

    return ds
    
for i, d in enumerate(data_locs):
    """
    if i == 0:
        dsd = xr.open_dataset(os.path.join(root_dir, 'control/latlon_diag.nc'), chunks={'Time': 4})
        dsh = xr.open_dataset(os.path.join(root_dir, 'control/latlon_hist.nc'), chunks={'Time': 4})
        #print(dsd)
        #print(dsh)
        
        # keep only times present in the 6h dataset
        if time_format == 'int':
            dsd_6h = dsd.isel(Time=slice(0, None, 2))
        else:
            dsd_6h = dsd.sel(Time=dsh.Time)
        
        # merge variables into one dataset
        ds = xr.merge([dsh, dsd_6h], join="exact") 
        
    else:
        ds = xr.open_dataset(os.path.join(root_dir, d), chunks={'Time': 4})

    # Dataset before time restriction
    print(ds.data_vars)

    if time_format == 'int':
        ds = ds.sel(Time=slice(-1-ndays*int(24/dt), -1))
    else:
        # Restrict time
        t_end = pd.to_datetime(ds.Time.max().item())
        t_start = t_end - pd.Timedelta(days=ndays)
        print(t_start, t_end)
        ds = ds.sel(Time=slice(t_start, t_end))
    """
    ds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks=latlon_chunks
    )
    ds = standardize_latlon_dataset(ds, needed_vars, pres_values=pres_levels[i])

    zds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=zm_latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks=zm_chunks,
        prefer_pattern=True,
        add_cldfrac_file=False
    )
    zds = standardize_latlon_dataset(zds, needed_vars, pres_values=pres_levels[i])

    # cheaper if data are exactly 6-hourly and aligned
    #ds = ds.coarsen(Time=4, boundary="trim").mean()
    #zds = zds.coarsen(Time=4, boundary="trim").mean()

    print(ds)
    print(zds)
    dss.append(ds)
    dss_zm.append(zds)

def lat_mean(da, lat_dim='latitude'):
    w = np.cos(np.deg2rad(da[lat_dim]))
    return da.weighted(w).mean(dim=lat_dim)
# Plotting helper functions

def plot_Tu_latpres(a, lats, pres, Tbar, ubar):
    T_im = a.contourf(lats, pres, Tbar, levels=np.arange(190, 305, 5), extend='both', cmap='coolwarm')
    u_im_neg = a.contour(lats, pres, ubar, levels=np.arange(-20, 0, 5), colors='k', linestyles=':', linewidths=0.5)
    u_im_zero = a.contour(lats, pres, ubar, levels=[0], colors='k', linewidths=1)
    u_im_pos = a.contour(lats, pres, ubar, levels=np.arange(5, 55, 5), colors='k', linewidths=0.5)
    a.clabel(u_im_pos, u_im_pos.levels, fontsize=6)
    a.clabel(u_im_neg, u_im_neg.levels, fontsize=6)

    return T_im, u_im_neg, u_im_zero, u_im_pos

def plot_rhv_latpres(a, lats, pres, rhbar, vbar, omgbar):
    xi = np.linspace(lats.min(), lats.max(), len(lats))
    yi = np.linspace(pres.min(), pres.max(), len(pres))

    X, Y = np.meshgrid(xi, yi)

    U = intp.interpn((lats, pres), vbar.T, (X, Y))
    V = intp.interpn((lats, pres), omgbar.T, (X, Y))

    rh_im = a.contourf(lats, pres, rhbar, levels=np.arange(0, 105, 5), cmap='BrBG')
    medcirc_im = a.streamplot(X, Y, U/111111., V/100., color='k', linewidth=1, density=1)

    return rh_im, medcirc_im

def isobaric_lat_mean(da):
    """Return an isobaric time-mean field ordered as (pres, latitude)."""
    return da.mean(dim="Time").transpose("pres", "latitude")

def interp_like_control(field, control_field):
    """Interpolate the control field onto another field's pressure/latitude grid before differencing."""
    return control_field.interp(pres=field.pres, latitude=field.latitude)

def set_latheight_labels(ax, xlabel=True):
    ax.set_xlim(-90, 90)
    if xlabel:
        ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Height (km)")

def symmetric_levels(max_abs, ndiv=11):
    """Symmetric colour boundaries with an odd number of colour divisions."""
    return np.linspace(-max_abs, max_abs, ndiv + 1)

def make_diff_norm(levels):
    return colors.BoundaryNorm(levels, plt.get_cmap("bwr").N)

def make_spectrum_cmap(levels):
    """Colormap that leaves normalised power <= 1 nearly white and emphasizes > 1."""
    base = plt.get_cmap("YlGnBu")
    centres = 0.5 * (levels[:-1] + levels[1:])
    cols_spec = []
    for c in centres:
        if c <= 1.0:
            cols_spec.append((1.0, 1.0, 1.0, 1.0))
        else:
            x = (c - 1.0) / (levels[-1] - 1.0)
            cols_spec.append(base(0.15 + 0.85 * x))
    cmap = colors.ListedColormap(cols_spec, name="spec_gt1")
    cmap.set_over(base(1.0))
    cmap.set_under((1.0, 1.0, 1.0, 1.0))
    norm = colors.BoundaryNorm(levels, cmap.N)
    return cmap, norm

# pre-compute CPTs for plotting

CPTs = []

for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
    theta = zds.theta.mean(dim="Time")
    pressure = zds.pressure.mean(dim="Time")

    T = theta * (pressure / 100000.)**(287.0 / 1004.0)
    T = model_lat_field(T)
    pressure = model_lat_field(pressure)
    z = height_from_zm_latlon(zds).values

    ztps = []
    ptps = []
    for j in range(len(zds.latitude.values)):
        ttp, ptp, ztp, _ = tropopause_profile(T.values[:, j], pressure.values[:, j], z)
        ztps.append(ztp)
        ptps.append(ptp)
        print(ztp)

    CPTs.append([ztps, ptps])

CPTs = np.array(CPTs)

# -----------------------------------------------------------------------------
# Additional mechanism diagnostics for QBO-like nudging experiments
# -----------------------------------------------------------------------------
# These diagnostics deliberately do *not* assume that the simulations contain an
# MJO-like mode.  They first test whether zonal/hemispheric symmetry is broken,
# then compare UTLS stability, cloud depth, convective thresholds, radiative
# coupling, and simple moisture-convection lag relationships.
#
# Suggested reading of the figures:
#   * Low zonal-asymmetric variance in precip/OLR means there is probably no
#     useful MJO-like cloud/rain pattern to composite.
#   * A strong T-nudge response in N2, cloud-top height, OLR-per-precipitation,
#     or high-cloud fraction points to the thermal/static-stability pathway.
#   * A strong U-nudge response in zonal asymmetry, k=1-3 power, eddy fluxes, or
#     lag structure points more towards a dynamical/wave-mediated pathway.

# If you want the radiative/convective diagnostics below, make sure these 2-D
# fields are included in needed_vars before the data are loaded:
#     "lwdnt", "lwdnb", "lwupb", "lwupt", "swdnt", "swdnb", "swupb", "swupt",
#     "lh", "hfx", "cape", "cin", "lcl", "lfc"
# The patched version of this script adds them automatically.


def accumulated_precip(ds, component="total"):
    """Accumulated precipitation in mm, including MPAS bucket counters."""
    conv = ds["rainc"] + ds["i_rainc"] * 100.0
    large = ds["rainnc"] + ds["i_rainnc"] * 100.0
    if component == "convective":
        return conv
    if component == "resolved":
        return large
    if component == "total":
        return conv + large
    raise ValueError("component must be 'total', 'convective', or 'resolved'")


def precip_rate(ds, component="total"):
    """Precipitation rate in mm day-1 from accumulated MPAS precipitation."""
    acc = accumulated_precip(ds, component=component)
    if issubclass(acc.Time.values.dtype.type, numbers.Integral):
        rate = acc.diff("Time") / (dt / 24.0)
    else:
        dt_days = acc.Time.diff("Time") / np.timedelta64(1, "D")
        rate = acc.diff("Time") / dt_days
    rate.name = f"precip_rate_{component}"
    rate.attrs["units"] = "mm day-1"
    return rate


def sel_time_like(da, ref):
    """Select exact times from da matching ref.Time; used after precip differencing."""
    return da.sel(Time=ref.Time)


def tropics(da, latmax=10):
    return da.sel(latitude=slice(-latmax, latmax))


def lat_weights(da):
    return np.cos(np.deg2rad(da.latitude))


def area_mean(da, latmax=None):
    x = tropics(da, latmax) if latmax is not None else da
    return x.weighted(lat_weights(x)).mean(dim=[d for d in ("latitude", "longitude") if d in x.dims])


def weighted_var(da, dims, weights=None):
    """Weighted variance over dims, keeping NaNs out."""
    if weights is None:
        m = da.mean(dim=dims, skipna=True)
        return ((da - m) ** 2).mean(dim=dims, skipna=True)
    m = da.weighted(weights).mean(dim=dims, skipna=True)
    return ((da - m) ** 2).weighted(weights).mean(dim=dims, skipna=True)


def hemispheric_components(da, lat_dim="latitude"):
    """Return hemispherically symmetric and antisymmetric parts on the original latitude grid."""
    flipped = da.isel({lat_dim: slice(None, None, -1)})
    flipped = flipped.assign_coords({lat_dim: da[lat_dim].values})
    sym = 0.5 * (da + flipped)
    asym = 0.5 * (da - flipped)
    return sym, asym


def fraction_zonal_asymmetry(da, latmax=10):
    """
    Fraction of tropical anomaly variance in k>0 components.

    The anomaly is formed by removing the local time mean.  The k=0 part is the
    longitude mean of this time-varying anomaly; k>0 is the residual.
    """
    x = tropics(da, latmax=latmax)
    x = x - x.mean(dim="Time", skipna=True)
    xstar = x - x.mean(dim="longitude", skipna=True)
    dims = [d for d in ("Time", "latitude", "longitude") if d in x.dims]
    w = lat_weights(x)
    total = weighted_var(x, dims=dims, weights=w)
    asym = weighted_var(xstar, dims=dims, weights=w)
    return float((asym / total).values)


def fraction_hemispheric_antisymmetry(da):
    """Fraction of anomaly variance in the hemispherically antisymmetric component."""
    x = da - da.mean(dim="Time", skipna=True)
    xs, xa = hemispheric_components(x)
    dims = [d for d in ("Time", "latitude", "longitude") if d in x.dims]
    w = lat_weights(x)
    vs = weighted_var(xs, dims=dims, weights=w)
    va = weighted_var(xa, dims=dims, weights=w)
    return float((va / (vs + va)).values)


def cldfrac_full_with_z(ds):
    """Return full lat/lon cloud fraction with a geometric-height z coordinate."""
    if "cldfrac" not in ds.data_vars:
        return None
    cf = ds.cldfrac
    if "nVertLevels" in cf.dims:
        cf = cf.rename({"nVertLevels": "z"})
    elif "nVertLevelsP1" in cf.dims:
        raise ValueError("cldfrac appears to be on interface levels")
    if "z" not in cf.dims:
        return None
    if "z" not in cf.coords or cf.z.size != cf.sizes["z"] or not np.issubdtype(cf.z.dtype, np.number):
        zmid = midlevel_height_from_latlon(ds, mean_latitude=True)
        cf = cf.assign_coords(z=("z", zmid.values))
    return cf


def high_cloud_fraction(ds, zmin=12000.0):
    """Mean cloud fraction above zmin using full lat/lon cldfrac, if available."""
    cf = cldfrac_full_with_z(ds)
    if cf is None:
        return None
    return cf.where(cf.z >= zmin).mean(dim="z", skipna=True)


def cloud_top_height(ds, thresh=0.05):
    """Column cloud-top height in m from full lat/lon cldfrac, if available."""
    cf = cldfrac_full_with_z(ds)
    if cf is None:
        return None
    z3d = xr.DataArray(cf.z.values, dims=("z",), coords={"z": cf.z})
    top = z3d.where(cf > thresh).max(dim="z", skipna=True)
    top.name = "cloud_top_height"
    top.attrs["units"] = "m"
    return top


def sample_flatten(*arrays, max_points=300000):
    """Flatten matching DataArrays/arrays and take a deterministic stride sample."""
    vals = [np.asarray(a).ravel() for a in arrays]
    mask = np.ones(vals[0].shape, dtype=bool)
    for v in vals:
        mask &= np.isfinite(v)
    vals = [v[mask] for v in vals]
    if vals[0].size > max_points:
        step = int(np.ceil(vals[0].size / max_points))
        vals = [v[::step] for v in vals]
    return vals


def binned_mean(x, y, bins, min_count=20):
    """Mean y in bins of x."""
    x, y = sample_flatten(x, y, max_points=1000000)
    ind = np.digitize(x, bins) - 1
    centres = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(len(centres), np.nan)
    count = np.zeros(len(centres), dtype=int)
    for i in range(len(centres)):
        m = ind == i
        count[i] = np.count_nonzero(m)
        if count[i] >= min_count:
            mean[i] = np.nanmean(y[m])
    return centres, mean, count


def get_dt_days(da):
    """Median output interval in days."""
    if issubclass(da.Time.values.dtype.type, numbers.Integral):
        return dt / 24.0
    return float(np.median(da.Time.diff("Time") / np.timedelta64(1, "D")))


def lag_corr(x, y, max_lag_steps):
    """
    Correlation between x(t) and y(t+lag).

    Positive lag therefore means x leads y.  This is useful for asking whether
    column moisture leads rainfall/OLR changes.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    out = []
    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    for lag in lags:
        if lag < 0:
            xs, ys = x[-lag:], y[:lag]
        elif lag > 0:
            xs, ys = x[:-lag], y[lag:]
        else:
            xs, ys = x, y
        mask = np.isfinite(xs) & np.isfinite(ys)
        if np.count_nonzero(mask) < 3:
            out.append(np.nan)
        else:
            out.append(np.corrcoef(xs[mask], ys[mask])[0, 1])
    return lags, np.asarray(out)


def net_atmospheric_radiation(ds):
    """
    Net atmospheric radiative heating proxy in W m-2.

    Positive means radiative energy convergence into the atmospheric column:
        net_down_TOA - net_down_surface
    using all-sky SW and LW fluxes.
    """
    needed = ["swdnt", "swupt", "lwdnt", "lwupt", "swdnb", "swupb", "lwdnb", "lwupb"]
    if any(v not in ds.data_vars for v in needed):
        return None
    net_toa = (ds.swdnt - ds.swupt) + (ds.lwdnt - ds.lwupt)
    net_sfc = (ds.swdnb - ds.swupb) + (ds.lwdnb - ds.lwupb)
    ratm = net_toa - net_sfc
    ratm.name = "net_atmospheric_radiation"
    ratm.attrs["units"] = "W m-2"
    return ratm


def time_aligned_fields(ds):
    """Common 2-D fields aligned to precipitation-rate time."""
    pr = precip_rate(ds, component="total")
    fields = {"Precip": pr}
    if "olrtoa" in ds:
        fields["OLR"] = sel_time_like(ds.olrtoa, pr)
    if "precipw" in ds:
        fields["PW"] = sel_time_like(ds.precipw, pr)
    if "cape" in ds:
        fields["CAPE"] = sel_time_like(ds.cape, pr)
    if "cin" in ds:
        fields["CIN"] = sel_time_like(ds.cin, pr)
    hc = high_cloud_fraction(ds, zmin=12000.0)
    if hc is not None:
        fields["High cloud"] = sel_time_like(hc, pr)
    # Do not add R_atm here: it is a compound field made from eight fluxes,
    # so constructing it repeatedly inside generic diagnostics can create large
    # Dask graphs.  The dedicated radiative diagnostic below computes it once
    # and samples it conservatively.
    return fields


def strided_matching_samples(*arrays, latmax=30, max_time=24, max_lat=80, max_lon=160):
    """
    Take the same coarse, deterministic stride sample from matching lat-lon-time fields.

    This is deliberately applied before calling .values, so histograms and binned
    relationships do not materialise a full compound Dask array.
    """
    xs = [tropics(a, latmax=latmax) for a in arrays]
    indexers = {}
    ref = xs[0]
    limits = {"Time": max_time, "latitude": max_lat, "longitude": max_lon}
    for dim, max_n in limits.items():
        if dim in ref.dims and ref.sizes[dim] > max_n:
            step = int(np.ceil(ref.sizes[dim] / max_n))
            indexers[dim] = slice(0, None, step)
    vals = [np.asarray(x.isel(indexers).values).ravel() for x in xs]
    mask = np.ones(vals[0].shape, dtype=bool)
    for v in vals:
        mask &= np.isfinite(v)
    return [v[mask] for v in vals]


# In[ ]:

# 1. Symmetry-breaking diagnostics.
# What to look for:
#   * If tropical k>0 variance for precipitation/OLR is very small, do not use
#     MJO-style phase composites.
#   * If nudging increases k>0 or antisymmetric variance, it is changing wave or
#     instability behaviour rather than just the zonal-mean convective state.

symmetry_rows = []
for i, ds in enumerate(dss):
    fields = time_aligned_fields(ds)
    test_fields = {}
    for key in ("Precip", "OLR", "PW", "High cloud"):
        if key in fields:
            test_fields[key] = fields[key]
    if "uzonal_isobaric" in ds:
        try:
            u850 = ds.uzonal_isobaric.sel(pres=850, method="nearest")
            test_fields["u850"] = sel_time_like(u850, fields["Precip"])
        except Exception:
            pass
    for name, da in test_fields.items():
        try:
            symmetry_rows.append({
                "simulation": sim_names[i],
                "variable": name,
                "zonal_asym_fraction_10S10N": fraction_zonal_asymmetry(da, latmax=10),
                "hemispheric_antisym_fraction_global": fraction_hemispheric_antisymmetry(da),
            })
        except Exception as e:
            print(f"Skipping symmetry metric for {sim_names[i]} {name}: {e}")

symmetry_df = pd.DataFrame(symmetry_rows)
print(symmetry_df)

if len(symmetry_df) > 0:
    variables = list(symmetry_df.variable.unique())
    x = np.arange(len(variables))
    width = 0.8 / len(dss)

    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), constrained_layout=True, sharex=True)
    for i, name in enumerate(sim_names):
        sub = symmetry_df[symmetry_df.simulation == name].set_index("variable").reindex(variables)
        ax[0].bar(x + (i - (len(dss)-1)/2)*width, sub.zonal_asym_fraction_10S10N, width, color=cols[i], label=name)
        ax[1].bar(x + (i - (len(dss)-1)/2)*width, sub.hemispheric_antisym_fraction_global, width, color=cols[i])

    ax[0].set_ylabel(r"Tropical $k>0$ variance fraction")
    ax[1].set_ylabel("Global antisymmetric variance fraction")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(variables)
    ax[0].legend(loc="best")
    ax[0].set_title("Symmetry-breaking diagnostics")

    if save: fig.savefig(save_loc + "diag_symmetry_breaking.png", dpi=300)
    print("Done with figure: diag_symmetry_breaking")


# In[ ]:

# 2. Zonal wavenumber variance fractions of tropical-mean anomalies.
# What to look for:
#   * k=0 dominance: mostly zonally symmetric variability.
#   * enhanced k=1-3 precip/OLR/PW variance: a necessary, but not sufficient,
#     condition for MJO-like planetary-scale organisation.

def tropical_zonal_series(da, latmax=10):
    x = tropics(da, latmax=latmax)
    return x.weighted(lat_weights(x)).mean(dim="latitude")


def zonal_wavenumber_variance_fraction(da, latmax=10, max_k=15):
    x = tropical_zonal_series(da, latmax=latmax)
    x = x - x.mean(dim="Time", skipna=True)
    arr = np.asarray(x.values)
    if arr.ndim != 2:
        raise ValueError(f"Expected (Time, longitude), got shape {arr.shape}")
    # Fill occasional NaNs with the longitude mean at that time before FFT.
    if np.any(~np.isfinite(arr)):
        row_mean = np.nanmean(arr, axis=1, keepdims=True)
        arr = np.where(np.isfinite(arr), arr, row_mean)
    fft = np.fft.rfft(arr, axis=1)
    power = np.abs(fft) ** 2
    if power.shape[1] > 2:
        power[:, 1:-1] *= 2.0
    pk = np.nanmean(power, axis=0)
    pk = pk / np.nansum(pk)
    kk = np.arange(pk.size)
    keep = kk <= max_k
    return kk[keep], pk[keep]

fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True, sharey=True)
plot_vars = [("Precip", None), ("OLR", "olrtoa"), ("PW", "precipw")]
for j, (label, vname) in enumerate(plot_vars):
    for i, ds in enumerate(dss):
        try:
            if label == "Precip":
                da = precip_rate(ds, component="total")
            else:
                da = ds[vname]
            k, frac = zonal_wavenumber_variance_fraction(da, latmax=10, max_k=15)
            ax[j].plot(k, frac, marker="o", color=cols[i], label=sim_names[i] if j == 0 else None)
        except Exception as e:
            print(f"Skipping zonal wavenumber plot for {sim_names[i]} {label}: {e}")
    ax[j].set_title(label)
    ax[j].set_xlabel("Zonal wavenumber")
    ax[j].set_xlim(0, 15)
    ax[j].set_yscale("log")
ax[0].set_ylabel("Fraction of tropical anomaly variance")
ax[0].legend(loc="best")

if save: fig.savefig(save_loc + "diag_zonal_wavenumber_variance.png", dpi=300)
print("Done with figure: diag_zonal_wavenumber_variance")


# In[ ]:

# 3. Equatorial Hovmoller anomalies for precipitation, OLR and precipitable water.
# These are better than raw fields for checking whether there is any coherent
# propagating cloud/rain envelope.

hov_vars = [("Precip", None, "mm day$^{-1}$"), ("OLR", "olrtoa", "W m$^{-2}$"), ("PW", "precipw", "kg m$^{-2}$")]
fig, ax = plt.subplots(len(hov_vars), len(dss), figsize=(3.3*len(dss), 7.5), constrained_layout=True, sharex=True)
if len(dss) == 1:
    ax = ax[:, None]

for j, (label, vname, units) in enumerate(hov_vars):
    vmax = None
    fields = []
    for ds in dss:
        if label == "Precip":
            da = precip_rate(ds, component="total")
        else:
            da = ds[vname]
        x = tropical_zonal_series(da, latmax=10)
        # Remove time mean at each longitude and instantaneous zonal mean.
        x = x - x.mean(dim="Time", skipna=True)
        x = x - x.mean(dim="longitude", skipna=True)
        fields.append(x)
        vv = np.nanpercentile(np.abs(x.values), 98)
        vmax = vv if vmax is None else max(vmax, vv)
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0
    levels_hov = np.linspace(-vmax, vmax, 21)
    for i, x in enumerate(fields):
        im = ax[j, i].contourf(x.longitude, x.Time, x.values, levels=levels_hov, cmap="bwr", extend="both")
        ax[j, i].set_title(sim_names[i] if j == 0 else "")
        if i == 0:
            ax[j, i].set_ylabel(f"{label}\ntime")
        else:
            ax[j, i].set_yticklabels([])
        if j == len(hov_vars)-1:
            ax[j, i].set_xlabel("Longitude")
    fig.colorbar(im, ax=ax[j, :], location="right", label=f"{label}' ({units})")

if save: fig.savefig(save_loc + "diag_hovmoller_anomalies.png", dpi=300)
print("Done with figure: diag_hovmoller_anomalies")


# In[ ]:

# 4. Latitude-height static stability N^2.
# What to look for:
#   * Does T nudging reduce/increase N2 near the tropical tropopause?
#   * Does U nudging induce a secondary N2 response despite only nudging wind?

def N2_latheight(zds):
    theta = model_lat_field(zds.theta.mean(dim="Time"))
    z = height_from_zm_latlon(zds).values
    dtheta_dz = np.gradient(theta.values, z, axis=0)
    g = 9.80665
    N2 = xr.DataArray(
        g / theta.values * dtheta_dz,
        dims=theta.dims,
        coords=theta.coords,
        name="N2",
        attrs={"units": "s-2"},
    )
    return N2

fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.3*len(dss)), constrained_layout=True, sharex=True)
fig2, ax2 = plt.subplots(len(dss), 1, figsize=(8, 3.3*len(dss)), constrained_layout=True, sharex=True)
if len(dss) == 1:
    ax = [ax]
    ax2 = [ax2]

control_N2 = None
im = im2 = im3 = None
for i, zds in enumerate(dss_zm):
    N2 = N2_latheight(zds)
    zmid = height_from_zm_latlon(zds)
    field, lat2d, z2d = lat_height_mesh(N2 * 1e4, zmid, zds.latitude.values)
    im = ax[i].contourf(lat2d, z2d/1000., field.values, levels=np.linspace(0, 8, 25), cmap="YlGnBu", extend="max")
    ax[i].plot(zds.latitude.values, CPTs[i][0]/1000., color="k", lw=1, ls=":")
    ax[i].set_title(sim_names[i])
    set_latheight_labels(ax[i], xlabel=(i == len(dss)-1))
    if i < len(dss)-1:
        ax[i].set_xticklabels([])

    if i == 0:
        control_N2 = N2
        im2 = ax2[i].contourf(lat2d, z2d/1000., field.values, levels=np.linspace(0, 8, 25), cmap="YlGnBu", extend="max")
    else:
        cN2 = control_N2.interp(z=N2.z, latitude=N2.latitude)
        diff = (N2 - cN2) * 1e4
        im3 = ax2[i].contourf(lat2d, z2d/1000., diff.values, levels=symmetric_levels(1.0, ndiv=12), cmap="bwr", extend="both")
    ax2[i].plot(zds.latitude.values, CPTs[i][0]/1000., color="k", lw=1, ls=":")
    ax2[i].set_title(sim_names[i])
    set_latheight_labels(ax2[i], xlabel=(i == len(dss)-1))
    if i < len(dss)-1:
        ax2[i].set_xticklabels([])

fig.colorbar(im, ax=ax, location="right", label=r"$N^2$ ($10^{-4}$ s$^{-2}$)")
fig2.colorbar(im2, ax=ax2[0], location="right", label=r"$N^2$ ($10^{-4}$ s$^{-2}$)")
if im3 is not None:
    fig2.colorbar(im3, ax=ax2[1:], location="right", label=r"$\Delta N^2$ ($10^{-4}$ s$^{-2}$)")

if save: fig.savefig(save_loc + "diag_N2_latheight.png", dpi=300)
if save: fig2.savefig(save_loc + "diag_N2_latheight_diff.png", dpi=300)
print("Done with figure: diag_N2_latheight")


# In[ ]:

# 5. Tropopause height and pressure profiles, plus nudged-control differences.
# What to look for:
#   * Tropical tropopause height/pressure shifts are a compact measure of the
#     UTLS thermal response.
#   * A T-nudged shift without a similar U-nudged shift points to a direct
#     thermal pathway.

fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True, sharex=True)
for i, zds in enumerate(dss_zm):
    lat = zds.latitude.values
    ztp_km = CPTs[i][0] / 1000.0
    ptp_hpa = CPTs[i][1] / 100.0
    ax[0, 0].plot(lat, ztp_km, color=cols[i], label=sim_names[i])
    ax[0, 1].plot(lat, ptp_hpa, color=cols[i])
    if i > 0:
        cztp = np.interp(lat, dss_zm[0].latitude.values, CPTs[0][0] / 1000.0)
        cptp = np.interp(lat, dss_zm[0].latitude.values, CPTs[0][1] / 100.0)
        ax[1, 0].plot(lat, ztp_km - cztp, color=cols[i], label=sim_names[i] + " - control")
        ax[1, 1].plot(lat, ptp_hpa - cptp, color=cols[i])

ax[0, 0].set_ylabel("Tropopause height (km)")
ax[0, 1].set_ylabel("Tropopause pressure (hPa)")
ax[1, 0].set_ylabel(r"$\Delta$ height (km)")
ax[1, 1].set_ylabel(r"$\Delta$ pressure (hPa)")
for a in ax.ravel():
    a.set_xlim(-90, 90)
    a.axvline(0, color="0.7", lw=0.5)
ax[1, 0].axhline(0, color="0.7", lw=0.5)
ax[1, 1].axhline(0, color="0.7", lw=0.5)
ax[1, 0].set_xlabel("Latitude")
ax[1, 1].set_xlabel("Latitude")
ax[0, 0].legend(loc="best")
ax[1, 0].legend(loc="best")
ax[0, 1].invert_yaxis()

if save: fig.savefig(save_loc + "diag_tropopause_profiles.png", dpi=300)
print("Done with figure: diag_tropopause_profiles")


# In[ ]:

# 6. High-cloud maps and cloud-top-height distributions.
# What to look for:
#   * More high cloud / higher cloud tops at fixed or similar rainfall suggests
#     a cloud-depth/radiative pathway rather than simply more precipitation.
#   * If this block skips, it means full lat/lon cldfrac was not loaded.

hc_fields = []
for ds in dss:
    hc = high_cloud_fraction(ds, zmin=12000.0)
    hc_fields.append(hc.mean(dim="Time") if hc is not None else None)

if any(h is not None for h in hc_fields):
    fig, ax = plt.subplots(1, len(dss), figsize=(3.6*len(dss), 3.2), constrained_layout=True, sharey=True)
    fig2, ax2 = plt.subplots(1, len(dss), figsize=(3.6*len(dss), 3.2), constrained_layout=True, sharey=True)
    if len(dss) == 1:
        ax = [ax]
        ax2 = [ax2]
    control_hc = None
    im = im2 = im3 = None
    for i, hc in enumerate(hc_fields):
        if hc is None:
            ax[i].set_visible(False)
            ax2[i].set_visible(False)
            continue
        im = ax[i].contourf(hc.longitude, hc.latitude, hc.values, levels=np.linspace(0, 1, 21), cmap="binary", extend="max")
        ax[i].set_title(sim_names[i])
        ax[i].set_xlabel("Longitude")
        if i == 0:
            ax[i].set_ylabel("Latitude")
            control_hc = hc
            im2 = ax2[i].contourf(hc.longitude, hc.latitude, hc.values, levels=np.linspace(0, 1, 21), cmap="binary", extend="max")
        else:
            ax[i].set_yticklabels([])
            c = control_hc.interp(latitude=hc.latitude, longitude=hc.longitude)
            im3 = ax2[i].contourf(hc.longitude, hc.latitude, (hc - c).values, levels=symmetric_levels(0.15, ndiv=12), cmap="bwr", extend="both")
        ax2[i].set_title(sim_names[i])
        ax2[i].set_xlabel("Longitude")
        if i == 0:
            ax2[i].set_ylabel("Latitude")
        else:
            ax2[i].set_yticklabels([])

    fig.colorbar(im, ax=ax[-1], location="right", label="Cloud fraction above 12 km")
    fig2.colorbar(im2, ax=ax2[0], location="right", label="Cloud fraction above 12 km")
    if im3 is not None:
        fig2.colorbar(im3, ax=ax2[1:], location="right", label="High-cloud difference")
    if save: fig.savefig(save_loc + "diag_high_cloud_maps.png", dpi=300)
    if save: fig2.savefig(save_loc + "diag_high_cloud_maps_diff.png", dpi=300)
    print("Done with figure: diag_high_cloud_maps")

cth_available = False
fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
for i, ds in enumerate(dss):
    cth = cloud_top_height(ds, thresh=0.05)
    if cth is None:
        continue
    cth_available = True
    vals = tropics(cth, latmax=10).values.ravel() / 1000.0
    vals = vals[np.isfinite(vals)]
    h, b = np.histogram(vals, bins=np.linspace(0, 25, 101), density=True)
    h = uniform_filter1d(h, size=3)
    ax.plot(0.5*(b[1:]+b[:-1]), h, color=cols[i], label=sim_names[i])
if cth_available:
    ax.set_xlabel("Tropical cloud-top height thresholded at cldfrac > 0.05 (km)")
    ax.set_ylabel("Probability density")
    ax.legend(loc="best")
    if save: fig.savefig(save_loc + "diag_cloud_top_height_hist.png", dpi=300)
    print("Done with figure: diag_cloud_top_height_hist")
else:
    plt.close(fig)
    print("Skipping cloud-top-height histogram: full lat/lon cldfrac unavailable")


# In[ ]:

# 7. Convective distributions: rainfall, OLR, PW, CAPE, CIN.
# What to look for:
#   * A rainfall tail change without a PW/CAPE shift suggests cloud-depth or
#     stability changes rather than just moisture availability.
#   * T nudging that changes OLR/PW/high-cloud distributions more than rainfall
#     is especially suggestive of a radiative/cloud pathway.

hist_specs = [
    ("Precip", lambda ds: precip_rate(ds, component="total"), np.linspace(0, 80, 121), "Rainfall rate (mm day$^{-1}$)", True),
    ("OLR", lambda ds: ds.olrtoa, np.linspace(100, 350, 121), r"OLR (W m$^{-2}$)", False),
    ("PW", lambda ds: ds.precipw, np.linspace(0, 80, 121), r"Precipitable water (kg m$^{-2}$)", False),
    ("CAPE", lambda ds: ds.cape if "cape" in ds else None, np.linspace(0, 5000, 121), r"CAPE (J kg$^{-1}$)", False),
    ("CIN", lambda ds: ds.cin if "cin" in ds else None, np.linspace(-500, 50, 121), r"CIN (J kg$^{-1}$)", False),
]

fig, ax = plt.subplots(1, len(hist_specs), figsize=(3.1*len(hist_specs), 3.4), constrained_layout=True)
for j, (name, func, bins, xlabel, logy) in enumerate(hist_specs):
    any_plotted = False
    for i, ds in enumerate(dss):
        da = func(ds)
        if da is None:
            continue
        vals = tropics(da, latmax=30).values.ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        h, b = np.histogram(vals, bins=bins, density=True)
        h = uniform_filter1d(h, size=3)
        ax[j].plot(0.5*(b[1:]+b[:-1]), h, color=cols[i], label=sim_names[i] if j == 0 else None)
        any_plotted = True
    ax[j].set_title(name)
    ax[j].set_xlabel(xlabel)
    if logy:
        ax[j].set_yscale("log")
    if not any_plotted:
        ax[j].text(0.5, 0.5, "not loaded", transform=ax[j].transAxes, ha="center", va="center")
ax[0].set_ylabel("Probability density")
ax[0].legend(loc="best")

if save: fig.savefig(save_loc + "diag_convective_distributions.png", dpi=300)
print("Done with figure: diag_convective_distributions")


# In[ ]:

# 8. Conditional convective physics.
# What to look for:
#   * Precip(PW): moisture threshold / pickup changes.
#   * OLR(precip): cloud-radiative/depth changes for a given rainfall rate.
#   * High cloud(precip): direct anvil/deep-cloud-depth response.
#   * Precip(CAPE): whether changes are tied to instability rather than UTLS effects.

fig, ax = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
for i, ds in enumerate(dss):
    fields = time_aligned_fields(ds)
    pr = tropics(fields["Precip"], latmax=30)

    if "PW" in fields:
        pw = tropics(fields["PW"], latmax=30)
        centres, mean, count = binned_mean(pw.values, pr.values, bins=np.linspace(0, 80, 41))
        ax[0, 0].plot(centres, mean, color=cols[i], label=sim_names[i])

    if "OLR" in fields:
        olr = tropics(fields["OLR"], latmax=30)
        centres, mean, count = binned_mean(pr.values, olr.values, bins=np.linspace(0, 80, 41))
        ax[0, 1].plot(centres, mean, color=cols[i])

    if "High cloud" in fields:
        hc = tropics(fields["High cloud"], latmax=30)
        centres, mean, count = binned_mean(pr.values, hc.values, bins=np.linspace(0, 80, 41))
        ax[1, 0].plot(centres, mean, color=cols[i])
    else:
        ax[1, 0].text(0.5, 0.5, "full cldfrac\nnot loaded", transform=ax[1, 0].transAxes, ha="center", va="center")

    if "CAPE" in fields:
        cape = tropics(fields["CAPE"], latmax=30)
        centres, mean, count = binned_mean(cape.values, pr.values, bins=np.linspace(0, 5000, 41))
        ax[1, 1].plot(centres, mean, color=cols[i])
    else:
        ax[1, 1].text(0.5, 0.5, "CAPE not loaded", transform=ax[1, 1].transAxes, ha="center", va="center")

ax[0, 0].set_xlabel(r"PW (kg m$^{-2}$)")
ax[0, 0].set_ylabel(r"Mean precipitation (mm day$^{-1}$)")
ax[0, 0].legend(loc="best")
ax[0, 1].set_xlabel(r"Precipitation (mm day$^{-1}$)")
ax[0, 1].set_ylabel(r"Mean OLR (W m$^{-2}$)")
ax[1, 0].set_xlabel(r"Precipitation (mm day$^{-1}$)")
ax[1, 0].set_ylabel("Mean cloud fraction above 12 km")
ax[1, 1].set_xlabel(r"CAPE (J kg$^{-1}$)")
ax[1, 1].set_ylabel(r"Mean precipitation (mm day$^{-1}$)")

if save: fig.savefig(save_loc + "diag_conditional_convective_coupling.png", dpi=300)
print("Done with figure: diag_conditional_convective_coupling")


# In[ ]:

# 9. OLR-precipitation coupling / greenhouse-enhancement-like proxy.
# This is not a formal clear-sky CRF decomposition, but the slope tells you how
# much OLR decreases per unit rainfall anomaly.
# What to look for:
#   * More negative slope in T-nudged run: deeper/more radiatively active clouds
#     per unit precipitation.
#   * Similar slope but shifted precipitation distribution: mostly amplitude/state change.

fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
slope_rows = []
for i, ds in enumerate(dss):
    fields = time_aligned_fields(ds)
    if "OLR" not in fields:
        continue
    pr = tropics(fields["Precip"], latmax=30)
    olr = tropics(fields["OLR"], latmax=30)

    pr_anom = pr - pr.mean(dim="Time", skipna=True)
    olr_anom = olr - olr.mean(dim="Time", skipna=True)
    x, y = sample_flatten(pr_anom.values, olr_anom.values, max_points=150000)
    if x.size < 10:
        continue
    slope, intercept = np.polyfit(x, y, 1)
    slope_rows.append({"simulation": sim_names[i], "dOLR_dP": slope})

    hb = ax[0].hexbin(x, y, gridsize=60, mincnt=1, linewidths=0.0)
    xs = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 100)
    ax[0].plot(xs, slope*xs + intercept, color=cols[i], lw=2, label=f"{sim_names[i]}: {slope:.2f}")

    centres, mean, count = binned_mean(pr_anom.values, olr_anom.values, bins=np.linspace(-30, 60, 46), min_count=50)
    ax[1].plot(centres, mean, color=cols[i], marker="o", ms=3, label=sim_names[i])

ax[0].axhline(0, color="0.7", lw=0.5)
ax[0].axvline(0, color="0.7", lw=0.5)
ax[0].set_xlabel(r"Precipitation anomaly (mm day$^{-1}$)")
ax[0].set_ylabel(r"OLR anomaly (W m$^{-2}$)")
ax[0].legend(loc="best", title=r"slope $dOLR'/dP'$")
ax[1].axhline(0, color="0.7", lw=0.5)
ax[1].axvline(0, color="0.7", lw=0.5)
ax[1].set_xlabel(r"Precipitation anomaly (mm day$^{-1}$)")
ax[1].set_ylabel(r"Binned mean OLR anomaly (W m$^{-2}$)")
ax[1].legend(loc="best")

slope_df = pd.DataFrame(slope_rows)
print(slope_df)
if save: fig.savefig(save_loc + "diag_olr_precip_coupling.png", dpi=300)
print("Done with figure: diag_olr_precip_coupling")


# In[ ]:

# 10. Atmospheric-column radiative heating proxy and coupling to precipitation.
# What to look for:
#   * A systematic R_atm shift suggests the nudging changes column radiative
#     energy input.
#   * A changed R_atm(precip) curve suggests cloud-radiative feedback changes.
#
# Memory note:
#   R_atm is built from eight full latitude-longitude flux fields.  Do not call
#   .values on the full tropical field.  The histogram and binned curve below use
#   a deterministic stride sample before loading data into memory.

rad_needed = ["swdnt", "swupt", "lwdnt", "lwupt", "swdnb", "swupb", "lwdnb", "lwupb"]
have_radiation = any(all(v in ds.data_vars for v in rad_needed) for ds in dss)

if have_radiation:
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for i, ds in enumerate(dss):
        if not all(v in ds.data_vars for v in rad_needed):
            continue

        ratm = net_atmospheric_radiation(ds)
        pr = precip_rate(ds, component="total")
        ratm_pr = sel_time_like(ratm, pr)

        # This is small after area_mean, so loading it is safe.
        tmean = area_mean(ratm, latmax=30).load()
        ax[0].plot(get_time_values(tmean), tmean.values, color=cols[i], label=sim_names[i])

        # Histogram: stride before .values.
        vals, = strided_matching_samples(ratm, latmax=30, max_time=32, max_lat=80, max_lon=160)
        if vals.size > 0:
            h, b = np.histogram(vals, bins=np.linspace(-300, 300, 121), density=True)
            h = uniform_filter1d(h, size=3)
            ax[1].plot(0.5*(b[1:]+b[:-1]), h, color=cols[i])

        # Conditional curve: sample precipitation and R_atm on the same sparse grid.
        pr_vals, ratm_vals = strided_matching_samples(
            pr, ratm_pr, latmax=30, max_time=32, max_lat=80, max_lon=160
        )
        if pr_vals.size > 0:
            centres, mean, count = binned_mean(
                pr_vals, ratm_vals, bins=np.linspace(0, 80, 41), min_count=20
            )
            ax[2].plot(centres, mean, color=cols[i])

        # Encourage Dask/matplotlib to release intermediate references between simulations.
        del ratm, pr, ratm_pr

    ax[0].set_xlabel("Time")
    ax[0].set_ylabel(r"Tropical mean $R_{atm}$ (W m$^{-2}$)")
    ax[0].legend(loc="best")
    ax[1].set_xlabel(r"$R_{atm}$ (W m$^{-2}$)")
    ax[1].set_ylabel("Probability density")
    ax[2].set_xlabel(r"Precipitation (mm day$^{-1}$)")
    ax[2].set_ylabel(r"Mean $R_{atm}$ (W m$^{-2}$)")
    if save: fig.savefig(save_loc + "diag_radiative_heating_precip.png", dpi=300)
    print("Done with figure: diag_radiative_heating_precip")
else:
    print("Skipping radiative-heating diagnostics: radiation flux variables not loaded")


# In[ ]:

# 11. Moisture-convection lag correlations using tropical means.
# What to look for:
#   * PW leading precipitation is a recharge-discharge signature.
#   * If nudging changes the lag of maximum correlation, it changes the timing of
#     moistening relative to convection even without an MJO-like envelope.

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True, sharey=True)
for i, ds in enumerate(dss):
    fields = time_aligned_fields(ds)
    if "PW" not in fields or "OLR" not in fields:
        continue
    pr_ts = area_mean(fields["Precip"], latmax=10)
    pw_ts = area_mean(fields["PW"], latmax=10)
    olr_ts = area_mean(fields["OLR"], latmax=10)

    dt_days_here = get_dt_days(fields["Precip"])
    max_steps = max(1, int(round(10.0 / dt_days_here)))
    lags, c_pr = lag_corr(pw_ts.values, pr_ts.values, max_steps)
    _, c_olr = lag_corr(pw_ts.values, olr_ts.values, max_steps)
    lag_days = lags * dt_days_here

    ax[0].plot(lag_days, c_pr, color=cols[i], label=sim_names[i])
    ax[1].plot(lag_days, c_olr, color=cols[i])

for a in ax:
    a.axvline(0, color="0.7", lw=0.5)
    a.axhline(0, color="0.7", lw=0.5)
    a.set_xlabel("Lag (days; positive means PW leads)")
ax[0].set_ylabel("Correlation")
ax[0].set_title("PW vs precipitation")
ax[1].set_title("PW vs OLR")
ax[0].legend(loc="best")

if save: fig.savefig(save_loc + "diag_tropical_lag_correlations.png", dpi=300)
print("Done with figure: diag_tropical_lag_correlations")

