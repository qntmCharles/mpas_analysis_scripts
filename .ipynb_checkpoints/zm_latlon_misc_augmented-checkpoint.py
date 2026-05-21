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
from wavenumber_frequency_functions import *
from matplotlib import colors
from tropopause import *

save = True
save_loc = '/home/cwp29/tmp/mpas/crf_on/'


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
    # Surface / column convection
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw", "olrtoa",
    "cape", "cin", "lcl", "lfc", "lh", "hfx",

    # Isobaric fields
    "uzonal_isobaric", "umeridional_isobaric", "omg_isobaric",
    "rh_isobaric", "t_isobaric", "qv_isobaric",

    # Model-level thermodynamic/dynamic fields
    "theta", "pressure", "zgrid", "zz",
    "uReconstructZonal", "uReconstructMeridional", "vorticity", "ke",

    # Cloud / condensate fields
    "cldfrac", "qc", "qr", "qi", "qs", "qg",

    # Radiation fluxes
    "lwdnt", "lwdnb", "lwupb", "lwupt",
    "swdnt", "swdnb", "swupb", "swupt",
]

check_vars = ['t_isobaric']



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
        chunks={"Time": -1}
    )
    ds = standardize_latlon_dataset(ds, needed_vars, pres_values=pres_levels[i])

    zds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=zm_latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks={"Time": -1},
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


# ============================================================
# AUGMENTED QBO / CONVECTION MECHANISM DIAGNOSTICS
# ============================================================
# These diagnostics are intentionally framed so that they do not assume an
# MJO-like mode exists.  They first test whether longitudinal or hemispheric
# symmetry is broken, then diagnose the UTLS thermodynamic/cloud/radiative
# pathways that would be relevant to an MJO-QBO mechanism.
#
# Things to look for are written in comments before each block.
# ============================================================


# ------------------------------
# Extra helper functions
# ------------------------------

def finite_or_nan(da, max_abs=1e30):
    """Mask NaNs/Infs and MPAS-style large fill values."""
    return da.where(np.isfinite(da) & (np.abs(da) < max_abs))


def total_precip_accum(ds):
    """Accumulated total precipitation in mm, including bucket counters."""
    return (
        ds["rainnc"] + ds["i_rainnc"] * 100.0
        + ds["rainc"] + ds["i_rainc"] * 100.0
    )


def conv_precip_accum(ds):
    """Accumulated convective precipitation in mm, including bucket counters."""
    return ds["rainc"] + ds["i_rainc"] * 100.0


def gridscale_precip_accum(ds):
    """Accumulated grid-scale precipitation in mm, including bucket counters."""
    return ds["rainnc"] + ds["i_rainnc"] * 100.0


def precip_rate_from_accum(accum):
    """Convert accumulated precipitation to a precipitation rate in mm/day."""
    if issubclass(accum.Time.values.dtype.type, numbers.Integral):
        return accum.diff("Time") / (dt / 24.0)
    dt_days = accum.Time.diff("Time") / np.timedelta64(1, "D")
    return accum.diff("Time") / dt_days


def tropical_mean_latlon(da, lat_bounds=(-10, 10)):
    """Area-weighted latitude mean and arithmetic longitude mean."""
    da = finite_or_nan(da).sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
    w = np.cos(np.deg2rad(da.latitude))
    return da.weighted(w).mean(dim="latitude").mean(dim="longitude")


def tropical_mean_zm(da, lat_bounds=(-10, 10)):
    """Area-weighted tropical mean for zonal-mean fields."""
    da = finite_or_nan(da).sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
    w = np.cos(np.deg2rad(da.latitude))
    return da.weighted(w).mean(dim="latitude")


def interp_control_like(field, control):
    """Interpolate a control field onto another field's coordinates."""
    kwargs = {}
    for coord in ("z", "pres", "latitude", "longitude"):
        if coord in field.coords and coord in control.coords:
            kwargs[coord] = field[coord]
    if kwargs:
        return control.interp(**kwargs, kwargs={"fill_value": np.nan})
    return control


def safe_nanpercentile(arr, q, default=1.0):
    vals = np.asarray(arr)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return default
    out = np.nanpercentile(vals, q)
    if not np.isfinite(out) or out == 0:
        return default
    return out


def binned_statistic_1d(x, y, bins, min_count=20):
    """Return bin centres, mean, interquartile range and counts."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    inds = np.digitize(x, bins) - 1
    xc = 0.5 * (bins[1:] + bins[:-1])
    mean = np.full(len(xc), np.nan)
    p25 = np.full(len(xc), np.nan)
    p75 = np.full(len(xc), np.nan)
    count = np.zeros(len(xc), dtype=int)

    for k in range(len(xc)):
        vals = y[inds == k]
        count[k] = vals.size
        if vals.size >= min_count:
            mean[k] = np.nanmean(vals)
            p25[k] = np.nanpercentile(vals, 25)
            p75[k] = np.nanpercentile(vals, 75)

    return xc, mean, p25, p75, count


def hemi_components_latlon(da, lat_dim="latitude"):
    """Return hemispherically symmetric and antisymmetric components."""
    flipped = da.isel({lat_dim: slice(None, None, -1)})
    flipped = flipped.assign_coords({lat_dim: da[lat_dim].values})
    sym = 0.5 * (da + flipped)
    asym = 0.5 * (da - flipped)
    return sym, asym


def pressure_level_nearest(da, p_hpa):
    """Select nearest pressure level in hPa."""
    return da.sel(pres=p_hpa, method="nearest")


def add_cpt_height(ax, i):
    ax.plot(dss_zm[i].latitude.values, CPTs[i][0] / 1000.0, color="k", lw=1, ls=":")


def get_latheight_field_from_zm(zds, var):
    """Time-mean zonal-mean model-level variable as (z, latitude)."""
    x = finite_or_nan(zds[var]).mean(dim="Time")
    zmid = height_from_zm_latlon(zds)
    return lat_height_mesh(x, zmid, zds.latitude.values)


def get_cldfrac_latheight(ds, zds, i=None, hemisym_control=False):
    """Return cloud fraction as (z, latitude), plus matching mesh arrays."""
    if "cldfrac" in zds.data_vars:
        cld = finite_or_nan(zds.cldfrac).mean(dim="Time")
        zmid = height_from_zm_latlon(zds)
        cld, lat2d, z2d = lat_height_mesh(cld, zmid, zds.latitude.values)
        return cld, lat2d, z2d, zds.latitude.values

    if "cldfrac" in ds.data_vars:
        cld = finite_or_nan(ds.cldfrac).mean(dim=("Time", "longitude"))
        cld = model_lat_field(cld)
        if hemisym_control and i == 0:
            cld = hemispheric_symmetrise_lat(cld)
        zmid = midlevel_height_from_latlon(ds)
        cld, lat2d, z2d = lat_height_mesh(cld, zmid, ds.latitude.values)
        return cld, lat2d, z2d, ds.latitude.values

    raise ValueError("No cldfrac available")


def condensate_latheight(ds, zds, varlist):
    """Return time-mean zonal-mean condensate in g/kg as (z, latitude)."""
    available_zm = [v for v in varlist if v in zds.data_vars]
    available_ll = [v for v in varlist if v in ds.data_vars]

    if len(available_zm) > 0:
        q = sum(finite_or_nan(zds[v]) for v in available_zm).mean(dim="Time") * 1000.0
        zmid = height_from_zm_latlon(zds)
        return lat_height_mesh(q, zmid, zds.latitude.values)

    if len(available_ll) > 0:
        q = sum(finite_or_nan(ds[v]) for v in available_ll).mean(dim=("Time", "longitude")) * 1000.0
        q = model_lat_field(q)
        zmid = midlevel_height_from_latlon(ds)
        return lat_height_mesh(q, zmid, ds.latitude.values)

    raise ValueError(f"None of {varlist} found")


def cloud_top_latheight_from_cldfrac(cld, threshold=0.01):
    """
    Cloud-top proxy from a zonal-mean cloud-fraction section.
    Returns one height per latitude, in km.
    """
    vals = cld.values
    zvals = cld.z.values
    out = np.full(vals.shape[1], np.nan)
    for j in range(vals.shape[1]):
        idx = np.where(vals[:, j] > threshold)[0]
        if idx.size > 0:
            out[j] = zvals[idx[-1]] / 1000.0
    return xr.DataArray(out, dims=("latitude",), coords={"latitude": cld.latitude}, name="cloud_top")


def high_cloud_fraction_from_cldfrac(cld, zmin_km=10.0, zmax_km=18.0):
    mask = (cld.z / 1000.0 >= zmin_km) & (cld.z / 1000.0 <= zmax_km)
    return cld.where(mask, drop=True).mean(dim="z")


def atmospheric_radiative_heating_proxy(ds):
    """
    Approximate all-sky atmospheric radiative heating proxy:
      net TOA minus net surface radiation.

    Check signs carefully before publication; this is intended as a relative
    diagnostic across the simulations.
    """
    required = ["swdnt", "swupt", "swdnb", "swupb", "lwdnt", "lwupt", "lwdnb", "lwupb"]
    missing = [v for v in required if v not in ds.data_vars]
    if missing:
        raise ValueError(f"Missing radiation variables: {missing}")

    net_toa_lw = ds.lwdnt - ds.lwupt
    net_sfc_lw = ds.lwdnb - ds.lwupb
    atm_lw = net_toa_lw - net_sfc_lw

    net_toa_sw = ds.swdnt - ds.swupt
    net_sfc_sw = ds.swdnb - ds.swupb
    atm_sw = net_toa_sw - net_sfc_sw

    return atm_lw, atm_sw, atm_lw + atm_sw


# ============================================================
# 1. Zonal-asymmetry fraction
# ============================================================
# What to look for:
#   - If Var(X*)/Var(X) is small for OLR and precipitation, the simulation is
#     mostly zonal-mean and MJO-style phase diagnostics are probably not useful.
#   - If it is large, symmetric forcing is still allowing spontaneous longitude
#     organisation.

fig, ax = plt.subplots(1, len(dss), figsize=(3.5 * len(dss), 3.2), constrained_layout=True)
if len(dss) == 1:
    ax = [ax]

for i, ds in enumerate(dss):
    ratios = []
    names = []

    fields = []
    if "olrtoa" in ds:
        fields.append(("OLR", finite_or_nan(ds.olrtoa)))
    if "precipw" in ds:
        fields.append(("PW", finite_or_nan(ds.precipw)))

    precip = total_precip_accum(ds)
    pr = precip_rate_from_accum(precip)
    fields.append(("Precip", pr))

    for name, x in fields:
        x = x.sel(latitude=slice(-30, 30))
        xprime = x - x.mean(dim="Time")
        x_zm = xprime.mean(dim="longitude")
        x_eddy = xprime - x_zm
        total_var = (xprime**2).mean(dim=("Time", "latitude", "longitude"))
        eddy_var = (x_eddy**2).mean(dim=("Time", "latitude", "longitude"))
        ratios.append((eddy_var / total_var).values.item())
        names.append(name)

    ax[i].bar(np.arange(len(ratios)), ratios, color=cols[i])
    ax[i].set_title(sim_names[i])
    ax[i].set_ylim(0, 1)
    ax[i].set_xticks(np.arange(len(ratios)))
    ax[i].set_xticklabels(names)
    ax[i].set_ylabel(r"Var($X^*$) / Var($X$)")

fig.suptitle("Tropical zonally asymmetric variance fraction")
if save:
    fig.savefig(save_loc + "comp_zonal_asymmetry_fraction.png", dpi=300)
print("Done with figure: comp_zonal_asymmetry_fraction")


# ============================================================
# 2. Hemispheric symmetric / antisymmetric variance fraction
# ============================================================
# What to look for:
#   - Small antisymmetric variance means hemisphere averaging is justified.
#   - Enhanced antisymmetric variance in a nudged run indicates symmetry breaking.

fig, ax = plt.subplots(1, len(dss), figsize=(3.5 * len(dss), 3.2), constrained_layout=True)
if len(dss) == 1:
    ax = [ax]

for i, ds in enumerate(dss):
    ratios = []
    names = []

    fields = []
    if "olrtoa" in ds:
        fields.append(("OLR", finite_or_nan(ds.olrtoa)))
    if "precipw" in ds:
        fields.append(("PW", finite_or_nan(ds.precipw)))
    fields.append(("Precip", precip_rate_from_accum(total_precip_accum(ds))))

    for name, x in fields:
        x = x.sel(latitude=slice(-60, 60))
        xprime = x - x.mean(dim="Time")
        xs, xa = hemi_components_latlon(xprime)
        sym_var = (xs**2).mean(dim=("Time", "latitude", "longitude"))
        asym_var = (xa**2).mean(dim=("Time", "latitude", "longitude"))
        ratios.append((asym_var / (sym_var + asym_var)).values.item())
        names.append(name)

    ax[i].bar(np.arange(len(ratios)), ratios, color=cols[i])
    ax[i].set_title(sim_names[i])
    ax[i].set_ylim(0, 1)
    ax[i].set_xticks(np.arange(len(ratios)))
    ax[i].set_xticklabels(names)
    ax[i].set_ylabel(r"Var($X_a$) / [Var($X_s$)+Var($X_a$)]")

fig.suptitle("Hemispheric antisymmetric variance fraction")
if save:
    fig.savefig(save_loc + "comp_hemispheric_asymmetry_fraction.png", dpi=300)
print("Done with figure: comp_hemispheric_asymmetry_fraction")


# ============================================================
# 3. Zonal-mean convective state and differences from control
# ============================================================
# What to look for:
#   - Broad OLR/high-cloud changes without a comparable precipitation change
#     suggest altered cloud depth/radiative effects rather than simply stronger rain.
#   - PW/CAPE/CIN shifts indicate that the lower-tropospheric convective regime
#     has changed, not just the UTLS.

surface_vars = [
    ("Precip", None, r"Precipitation (mm day$^{-1}$)"),
    ("olrtoa", "olrtoa", r"OLR (W m$^{-2}$)"),
    ("precipw", "precipw", r"Precipitable water (kg m$^{-2}$)"),
    ("cape", "cape", r"CAPE (J kg$^{-1}$)"),
    ("cin", "cin", r"CIN (J kg$^{-1}$)"),
]

fig, ax = plt.subplots(len(surface_vars), 1, figsize=(8, 2.1 * len(surface_vars)), constrained_layout=True, sharex=True)
fig2, ax2 = plt.subplots(len(surface_vars), 1, figsize=(8, 2.1 * len(surface_vars)), constrained_layout=True, sharex=True)

for j, (label, var, ylabel) in enumerate(surface_vars):
    control = None
    for i, ds in enumerate(dss):
        if label == "Precip":
            x = precip_rate_from_accum(total_precip_accum(ds))
        else:
            if var not in ds.data_vars:
                continue
            x = finite_or_nan(ds[var])

        xm = x.mean(dim=("Time", "longitude"))
        ax[j].plot(ds.latitude.values, xm.values, color=cols[i], label=sim_names[i])

        if i == 0:
            control = xm
            ax2[j].axhline(0, color="k", lw=0.7)
        else:
            c = control.interp(latitude=xm.latitude)
            ax2[j].plot(ds.latitude.values, (xm - c).values, color=cols[i], label=sim_names[i])

    ax[j].set_ylabel(ylabel)
    ax2[j].set_ylabel(r"$\Delta$ " + ylabel)
    for a in (ax[j], ax2[j]):
        a.axvline(0, color="grey", lw=0.5)
        a.set_xlim(-90, 90)

ax[-1].set_xlabel("Latitude")
ax2[-1].set_xlabel("Latitude")
fig.legend(loc="outside lower center", ncol=len(dss))
fig2.legend(loc="outside lower center", ncol=max(1, len(dss) - 1))

if save:
    fig.savefig(save_loc + "comp_zonalmean_convective_state.png", dpi=300)
    fig2.savefig(save_loc + "comp_zonalmean_convective_state_diff.png", dpi=300)
print("Done with figure: comp_zonalmean_convective_state")


# ============================================================
# 4. Latitude-height static stability N^2
# ============================================================
# What to look for:
#   - Reduced N^2 near the tropical tropopause in T_nudge supports a direct
#     static-stability/cloud-depth pathway.
#   - A similar response in u_nudge indicates thermal-wind/dynamical coupling.

fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.4 * len(dss)), constrained_layout=True, sharex=True)
fig2, ax2 = plt.subplots(len(dss), 1, figsize=(8, 3.4 * len(dss)), constrained_layout=True, sharex=True)
if len(dss) == 1:
    ax = [ax]
    ax2 = [ax2]

g = 9.80665
control_N2 = None
im = im2 = im3 = None

for i, zds in enumerate(dss_zm):
    theta = model_lat_field(finite_or_nan(zds.theta).mean(dim="Time"))
    z = height_from_zm_latlon(zds).values
    lat = zds.latitude.values
    dtheta_dz = np.gradient(theta.values, z, axis=0)
    N2_vals = g / theta.values * dtheta_dz
    N2 = xr.DataArray(N2_vals, dims=("z", "latitude"), coords={"z": theta.z.values, "latitude": theta.latitude.values})
    lat2d = np.broadcast_to(lat[None, :], N2.shape)
    z2d = np.broadcast_to(z[:, None], N2.shape)

    im = ax[i].contourf(lat2d, z2d / 1000.0, N2.values * 1e4, levels=np.linspace(-1, 8, 28), cmap="YlGnBu", extend="both")
    add_cpt_height(ax[i], i)
    ax[i].set_title(sim_names[i])
    set_latheight_labels(ax[i], xlabel=(i == len(dss) - 1))
    if i < len(dss) - 1:
        ax[i].set_xticklabels([])

    if i == 0:
        control_N2 = N2
        im2 = ax2[i].contourf(lat2d, z2d / 1000.0, N2.values * 1e4, levels=np.linspace(-1, 8, 28), cmap="YlGnBu", extend="both")
    else:
        c = control_N2.interp(z=N2.z, latitude=N2.latitude, kwargs={"fill_value": np.nan})
        diff = (N2 - c) * 1e4
        im3 = ax2[i].contourf(lat2d, z2d / 1000.0, diff.values, levels=symmetric_levels(1.0, ndiv=11), cmap="bwr", extend="both")

    add_cpt_height(ax2[i], i)
    ax2[i].set_title(sim_names[i])
    set_latheight_labels(ax2[i], xlabel=(i == len(dss) - 1))
    if i < len(dss) - 1:
        ax2[i].set_xticklabels([])

fig.colorbar(im, ax=ax, location="right", label=r"$N^2$ ($10^{-4}$ s$^{-2}$)")
fig2.colorbar(im2, ax=ax2[0], location="right", label=r"$N^2$ ($10^{-4}$ s$^{-2}$)")
if im3 is not None:
    fig2.colorbar(im3, ax=ax2[1:], location="right", label=r"$\Delta N^2$ ($10^{-4}$ s$^{-2}$)")

if save:
    fig.savefig(save_loc + "comp_N2_latheight.png", dpi=300)
    fig2.savefig(save_loc + "comp_N2_latheight_diff.png", dpi=300)
print("Done with figure: comp_N2_latheight")


# ============================================================
# 5. Hydrometeor latitude-height sections
# ============================================================
# What to look for:
#   - More upper-tropospheric ice/snow/graupel in T_nudge without much more rain
#     points to deeper/anvil cloud changes rather than just greater rain amount.

hydro_groups = [
    ("liquid_plus_rain", ["qc", "qr"], r"$q_c+q_r$ (g kg$^{-1}$)"),
    ("ice_plus_snow_graupel", ["qi", "qs", "qg"], r"$q_i+q_s+q_g$ (g kg$^{-1}$)"),
    ("total_condensate", ["qc", "qr", "qi", "qs", "qg"], r"Total condensate (g kg$^{-1}$)"),
]

for outname, varlist, cbar_label in hydro_groups:
    try:
        fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.4 * len(dss)), constrained_layout=True, sharex=True)
        fig2, ax2 = plt.subplots(len(dss), 1, figsize=(8, 3.4 * len(dss)), constrained_layout=True, sharex=True)
        if len(dss) == 1:
            ax = [ax]
            ax2 = [ax2]

        control_q = None
        im = im2 = im3 = None

        # Use a common absolute colour scale from the control where possible.
        q0, _, _ = condensate_latheight(dss[0], dss_zm[0], varlist)
        vmax = safe_nanpercentile(q0.values, 99.5, default=1e-4)
        levs = np.linspace(0, vmax, 21)

        for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
            q, lat2d, z2d = condensate_latheight(ds, zds, varlist)

            im = ax[i].contourf(lat2d, z2d / 1000.0, q.values, levels=levs, cmap="YlGnBu", extend="max")
            add_cpt_height(ax[i], i)
            ax[i].set_title(sim_names[i])
            set_latheight_labels(ax[i], xlabel=(i == len(dss) - 1))
            if i < len(dss) - 1:
                ax[i].set_xticklabels([])

            if i == 0:
                control_q = q
                im2 = ax2[i].contourf(lat2d, z2d / 1000.0, q.values, levels=levs, cmap="YlGnBu", extend="max")
            else:
                c = control_q.interp(z=q.z, latitude=q.latitude, kwargs={"fill_value": np.nan})
                diff = q - c
                dmax = safe_nanpercentile(np.abs(diff.values), 99.0, default=max(vmax * 0.25, 1e-6))
                im3 = ax2[i].contourf(lat2d, z2d / 1000.0, diff.values, levels=symmetric_levels(dmax, ndiv=11), cmap="bwr", extend="both")

            add_cpt_height(ax2[i], i)
            ax2[i].set_title(sim_names[i])
            set_latheight_labels(ax2[i], xlabel=(i == len(dss) - 1))
            if i < len(dss) - 1:
                ax2[i].set_xticklabels([])

        fig.colorbar(im, ax=ax, location="right", label=cbar_label)
        fig2.colorbar(im2, ax=ax2[0], location="right", label=cbar_label)
        if im3 is not None:
            fig2.colorbar(im3, ax=ax2[1:], location="right", label=r"$\Delta$ " + cbar_label)

        if save:
            fig.savefig(save_loc + f"comp_{outname}_latheight.png", dpi=300)
            fig2.savefig(save_loc + f"comp_{outname}_latheight_diff.png", dpi=300)
        print(f"Done with figure: comp_{outname}_latheight")
    except ValueError as e:
        print(f"Skipping {outname}: {e}")


# ============================================================
# 6. Cloud-top proxy and high-cloud fraction from zonal-mean cloud fraction
# ============================================================
# What to look for:
#   - Higher tropical cloud tops / more 10-18 km cloud in T_nudge would support
#     a UTLS-temperature/cloud-depth interpretation.

try:
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), constrained_layout=True, sharex=True)
    fig2, ax2 = plt.subplots(2, 1, figsize=(8, 5.5), constrained_layout=True, sharex=True)

    control_ctop = None
    control_hcld = None

    for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
        cld, lat2d, z2d, lat_for_cpt = get_cldfrac_latheight(ds, zds, i=i, hemisym_control=True)
        ctop = cloud_top_latheight_from_cldfrac(cld, threshold=0.01)
        hcld = high_cloud_fraction_from_cldfrac(cld, zmin_km=10.0, zmax_km=18.0)

        ax[0].plot(cld.latitude, ctop, color=cols[i], label=sim_names[i])
        ax[1].plot(cld.latitude, hcld, color=cols[i], label=sim_names[i])

        if i == 0:
            control_ctop = ctop
            control_hcld = hcld
            ax2[0].axhline(0, color="k", lw=0.7)
            ax2[1].axhline(0, color="k", lw=0.7)
        else:
            ax2[0].plot(cld.latitude, ctop - control_ctop.interp(latitude=ctop.latitude), color=cols[i], label=sim_names[i])
            ax2[1].plot(cld.latitude, hcld - control_hcld.interp(latitude=hcld.latitude), color=cols[i], label=sim_names[i])

    ax[0].set_ylabel("Cloud-top proxy (km)")
    ax[1].set_ylabel("High cloud fraction\n10-18 km")
    ax2[0].set_ylabel(r"$\Delta$ cloud top (km)")
    ax2[1].set_ylabel(r"$\Delta$ high cloud")

    for a in list(ax) + list(ax2):
        a.axvline(0, color="grey", lw=0.5)
        a.set_xlim(-90, 90)
        a.grid(alpha=0.2)
    ax[1].set_xlabel("Latitude")
    ax2[1].set_xlabel("Latitude")
    ax[0].legend(loc="best")
    ax2[0].legend(loc="best")

    if save:
        fig.savefig(save_loc + "comp_cloudtop_highcloud_latitude.png", dpi=300)
        fig2.savefig(save_loc + "comp_cloudtop_highcloud_latitude_diff.png", dpi=300)
    print("Done with figure: comp_cloudtop_highcloud_latitude")
except ValueError as e:
    print(f"Skipping cloud-top/high-cloud diagnostic: {e}")


# ============================================================
# 7. Conditional convective relationships
# ============================================================
# What to look for:
#   - For the same precipitation, lower OLR or higher cloud top in T_nudge implies
#     deeper/more radiatively active cloud, not just more rain.
#   - Different precip(PW/CAPE/CIN) curves imply changed convective threshold.

conditional_specs = [
    ("precipw", r"Precipitable water (kg m$^{-2}$)", np.linspace(0, 80, 41)),
    ("cape", r"CAPE (J kg$^{-1}$)", np.linspace(0, 6000, 41)),
    ("cin", r"CIN (J kg$^{-1}$)", np.linspace(-500, 100, 41)),
]

fig, ax = plt.subplots(1, len(conditional_specs), figsize=(12, 3.4), constrained_layout=True)
if len(conditional_specs) == 1:
    ax = [ax]

for j, (var, xlabel, bins) in enumerate(conditional_specs):
    for i, ds in enumerate(dss):
        if var not in ds.data_vars:
            continue
        pr = precip_rate_from_accum(total_precip_accum(ds))
        x = finite_or_nan(ds[var]).sel(Time=pr.Time)
        x = x.sel(latitude=slice(-20, 20))
        pr_sub = pr.sel(latitude=slice(-20, 20))
        x, pr_sub = xr.align(x, pr_sub, join="inner")
        xc, ym, y25, y75, count = binned_statistic_1d(x.values, pr_sub.values, bins)
        ax[j].plot(xc, ym, color=cols[i], label=sim_names[i])
        ax[j].fill_between(xc, y25, y75, color=cols[i], alpha=0.15)
    ax[j].set_xlabel(xlabel)
    ax[j].set_ylabel(r"Precipitation (mm day$^{-1}$)")
    ax[j].grid(alpha=0.2)
ax[0].legend(loc="best")

if save:
    fig.savefig(save_loc + "comp_precip_conditioned_on_pw_cape_cin.png", dpi=300)
print("Done with figure: comp_precip_conditioned_on_pw_cape_cin")


# OLR and cloud-top conditioned on precipitation.
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
precip_bins = np.linspace(0, 60, 41)

for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
    pr = precip_rate_from_accum(total_precip_accum(ds))
    olr = finite_or_nan(ds.olrtoa).sel(Time=pr.Time)

    pr_trop = pr.sel(latitude=slice(-20, 20))
    olr_trop = olr.sel(latitude=slice(-20, 20))
    pr_trop, olr_trop = xr.align(pr_trop, olr_trop, join="inner")

    xc, ym, y25, y75, count = binned_statistic_1d(pr_trop.values, olr_trop.values, precip_bins)
    ax[0].plot(xc, ym, color=cols[i], label=sim_names[i])
    ax[0].fill_between(xc, y25, y75, color=cols[i], alpha=0.15)

    # A cheaper cloud-top conditional: pair latitude-mean precipitation with the
    # zonal-mean cloud-top proxy.  This is not a gridpoint joint distribution;
    # it asks how the mean cloud-top changes across latitudinal rain regimes.
    try:
        cld, _, _, _ = get_cldfrac_latheight(ds, zds, i=i, hemisym_control=True)
        ctop_lat = cloud_top_latheight_from_cldfrac(cld, threshold=0.01)
        pr_lat = pr.mean(dim=("Time", "longitude"))
        pr_lat, ctop_lat = xr.align(pr_lat, ctop_lat, join="inner")
        xc2, ym2, y252, y752, count2 = binned_statistic_1d(pr_lat.values, ctop_lat.values, precip_bins, min_count=2)
        ax[1].plot(xc2, ym2, color=cols[i], label=sim_names[i])
    except ValueError:
        pass

ax[0].set_xlabel(r"Precipitation (mm day$^{-1}$)")
ax[0].set_ylabel(r"OLR (W m$^{-2}$)")
ax[1].set_xlabel(r"Latitude-mean precipitation (mm day$^{-1}$)")
ax[1].set_ylabel("Cloud-top proxy (km)")
for a in ax:
    a.grid(alpha=0.2)
ax[0].legend(loc="best")

if save:
    fig.savefig(save_loc + "comp_olr_cloudtop_conditioned_on_precip.png", dpi=300)
print("Done with figure: comp_olr_cloudtop_conditioned_on_precip")


# ============================================================
# 8. Radiation budget proxy
# ============================================================
# What to look for:
#   - Tropical LW-atmospheric heating changes consistent with high-cloud changes
#     support a cloud-radiative pathway.

try:
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), constrained_layout=True, sharex=True)
    fig2, ax2 = plt.subplots(3, 1, figsize=(8, 7), constrained_layout=True, sharex=True)
    control_profiles = None

    for i, ds in enumerate(dss):
        atm_lw, atm_sw, atm_rad = atmospheric_radiative_heating_proxy(ds)
        profiles = [
            atm_lw.mean(dim=("Time", "longitude")),
            atm_sw.mean(dim=("Time", "longitude")),
            atm_rad.mean(dim=("Time", "longitude")),
        ]

        for j, prof in enumerate(profiles):
            ax[j].plot(ds.latitude, prof, color=cols[i], label=sim_names[i])

        if i == 0:
            control_profiles = profiles
            for a in ax2:
                a.axhline(0, color="k", lw=0.7)
        else:
            for j, prof in enumerate(profiles):
                c = control_profiles[j].interp(latitude=prof.latitude)
                ax2[j].plot(ds.latitude, prof - c, color=cols[i], label=sim_names[i])

    labels = ["Atmos LW proxy", "Atmos SW proxy", "Atmos net rad proxy"]
    for j in range(3):
        ax[j].set_ylabel(labels[j] + "\n(W m$^{-2}$)")
        ax2[j].set_ylabel(r"$\Delta$ " + labels[j] + "\n(W m$^{-2}$)")
        for a in (ax[j], ax2[j]):
            a.axvline(0, color="grey", lw=0.5)
            a.set_xlim(-90, 90)
            a.grid(alpha=0.2)
    ax[-1].set_xlabel("Latitude")
    ax2[-1].set_xlabel("Latitude")
    fig.legend(loc="outside lower center", ncol=len(dss))
    fig2.legend(loc="outside lower center", ncol=max(1, len(dss) - 1))

    if save:
        fig.savefig(save_loc + "comp_atmos_radiative_heating_proxy.png", dpi=300)
        fig2.savefig(save_loc + "comp_atmos_radiative_heating_proxy_diff.png", dpi=300)
    print("Done with figure: comp_atmos_radiative_heating_proxy")
except ValueError as e:
    print(f"Skipping radiation budget proxy: {e}")


# ============================================================
# 9. OLR anomaly per precipitation anomaly
# ============================================================
# What to look for:
#   - A more negative regression slope means a stronger OLR reduction per rain
#     anomaly, consistent with deeper/high-cloud radiative effects.

fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)

for i, ds in enumerate(dss):
    pr = precip_rate_from_accum(total_precip_accum(ds))
    olr = finite_or_nan(ds.olrtoa).sel(Time=pr.Time)

    pr = pr.sel(latitude=slice(-20, 20))
    olr = olr.sel(latitude=slice(-20, 20))
    pr, olr = xr.align(pr, olr, join="inner")

    pr_anom = pr - pr.mean(dim="Time")
    olr_anom = olr - olr.mean(dim="Time")

    x = pr_anom.values.ravel()
    y = olr_anom.values.ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if x.size == 0:
        continue

    # Subsample only for plotting, not for the regression.
    if x.size > 200000:
        rng = np.random.default_rng(1234)
        idx = rng.choice(x.size, size=200000, replace=False)
        xp = x[idx]
        yp = y[idx]
    else:
        xp = x
        yp = y

    ax.scatter(xp, yp, s=1, alpha=0.04, color=cols[i])
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 100)
    ax.plot(xx, intercept + slope * xx, color=cols[i], label=f"{sim_names[i]}: slope={slope:.2f}")

ax.axhline(0, color="grey", lw=0.5)
ax.axvline(0, color="grey", lw=0.5)
ax.set_xlabel(r"Precipitation anomaly (mm day$^{-1}$)")
ax.set_ylabel(r"OLR anomaly (W m$^{-2}$)")
ax.legend(loc="best", fontsize=8)
ax.grid(alpha=0.2)

if save:
    fig.savefig(save_loc + "comp_olr_precip_anomaly_slope.png", dpi=300)
print("Done with figure: comp_olr_precip_anomaly_slope")


# ============================================================
# 10. Extended tropical-mean time series
# ============================================================
# What to look for:
#   - Coherent changes in PW -> rain -> OLR/high-cloud timing indicate changed
#     convection-radiation coupling even without MJO-like propagation.

fig, ax = plt.subplots(5, 1, figsize=(8, 8), constrained_layout=True, sharex=True)

for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
    pr = precip_rate_from_accum(total_precip_accum(ds))
    pr_trop = tropical_mean_latlon(pr)
    pw_trop = tropical_mean_latlon(ds.precipw)
    olr_trop = tropical_mean_latlon(ds.olrtoa)

    ax[0].plot(get_time_values(pr_trop), pr_trop, color=cols[i], label=sim_names[i])
    ax[1].plot(get_time_values(pw_trop), pw_trop, color=cols[i])
    ax[2].plot(get_time_values(olr_trop), olr_trop, color=cols[i])

    try:
        if "cldfrac" in zds.data_vars:
            cld_ts = finite_or_nan(zds.cldfrac).sel(latitude=slice(-20, 20))
            z = height_from_zm_latlon(zds)
        elif "cldfrac" in ds.data_vars:
            cld_ts = finite_or_nan(ds.cldfrac).sel(latitude=slice(-20, 20)).mean(dim="longitude")
            lev_dim = vertical_dim(cld_ts)
            if lev_dim != "z":
                cld_ts = cld_ts.rename({lev_dim: "z"})
            z = midlevel_height_from_latlon(ds, latitude_sel=slice(-20, 20), mean_latitude=True)
        else:
            cld_ts = None
            z = None

        if cld_ts is not None:
            zvals = z.values / 1000.0
            mask = (zvals >= 10.0) & (zvals <= 18.0)
            high_cld_ts = cld_ts.isel(z=np.where(mask)[0]).mean(dim=[d for d in cld_ts.dims if d not in ("Time",)])
            ax[3].plot(get_time_values(high_cld_ts), high_cld_ts, color=cols[i])
    except Exception as e:
        print(f"Skipping high-cloud time series for {sim_names[i]}: {e}")

    hydro = [v for v in ["qc", "qr", "qi", "qs", "qg"] if v in ds.data_vars]
    if len(hydro) > 0:
        qtot = sum(finite_or_nan(ds[v]) for v in hydro)
        qtot_trop = qtot.sel(latitude=slice(-20, 20)).mean(dim=[d for d in qtot.dims if d not in ("Time",)]) * 1000.0
        ax[4].plot(get_time_values(qtot_trop), qtot_trop, color=cols[i])

ax[0].set_ylabel(r"Rain\n(mm day$^{-1}$)")
ax[1].set_ylabel("PW")
ax[2].set_ylabel("OLR")
ax[3].set_ylabel("High cloud\n10-18 km")
ax[4].set_ylabel(r"Mean condensate\n(g kg$^{-1}$)")
ax[4].set_xlabel("Time")
fig.legend(loc="outside lower center", ncol=len(dss))

if save:
    fig.savefig(save_loc + "comp_tropical_timeseries_extended.png", dpi=300)
print("Done with figure: comp_tropical_timeseries_extended")


# ============================================================
# 11. Hovmoller diagrams for OLR and precipitable water anomalies
# ============================================================
# What to look for:
#   - Coherent eastward propagation would justify MJO-like diagnostics.
#   - Zonal-mean/standing anomalies argue for a mean-state/convective-regime analysis.

hov_vars = [
    ("olrtoa", r"OLR anomaly (W m$^{-2}$)", symmetric_levels(30, ndiv=12), "comp_hovmoller_olr_anom.png"),
    ("precipw", r"PW anomaly (kg m$^{-2}$)", symmetric_levels(4, ndiv=12), "comp_hovmoller_pw_anom.png"),
]

for var, cbar_label, levs, fname in hov_vars:
    if not all(var in ds.data_vars for ds in dss):
        continue
    fig, ax = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 3.2), constrained_layout=True, sharey=True)
    if len(dss) == 1:
        ax = [ax]
    im = None
    for i, ds in enumerate(dss):
        x = finite_or_nan(ds[var]).sel(latitude=slice(-10, 10)).mean(dim="latitude")
        x = x - x.mean(dim="Time")
        im = ax[i].contourf(x.longitude, get_time_values(x), x.values, levels=levs, cmap="bwr", extend="both")
        ax[i].set_title(sim_names[i])
        ax[i].set_xlabel("Longitude")
        if i == 0:
            ax[i].set_ylabel("Time")
        else:
            ax[i].set_yticklabels([])
    fig.colorbar(im, ax=ax[-1], location="right", label=cbar_label)
    if save:
        fig.savefig(save_loc + fname, dpi=300)
    print(f"Done with figure: {fname}")


# ============================================================
# 12. UTLS summary metrics
# ============================================================
# What to look for:
#   - This compactly shows whether the U-nudged and T-nudged runs differ mostly
#     in wind, temperature, shear, stability, or tropopause displacement.

metric_keys = ["T_100hPa", "u_100hPa", "u_70minus200", "N2_14_18km", "CPT_height", "CPT_pressure"]
metrics = {k: [] for k in metric_keys}

for i, zds in enumerate(dss_zm):
    if "t_isobaric" in zds.data_vars:
        T100 = pressure_level_nearest(zds.t_isobaric, 100).sel(latitude=slice(-10, 10)).mean(dim=("Time", "latitude")).values.item()
    else:
        T100 = np.nan

    if "uzonal_isobaric" in zds.data_vars:
        u100 = pressure_level_nearest(zds.uzonal_isobaric, 100).sel(latitude=slice(-10, 10)).mean(dim=("Time", "latitude")).values.item()
        u70 = pressure_level_nearest(zds.uzonal_isobaric, 70).sel(latitude=slice(-10, 10)).mean(dim=("Time", "latitude")).values.item()
        u200 = pressure_level_nearest(zds.uzonal_isobaric, 200).sel(latitude=slice(-10, 10)).mean(dim=("Time", "latitude")).values.item()
        shear = u70 - u200
    else:
        u100 = np.nan
        shear = np.nan

    theta = finite_or_nan(zds.theta).sel(latitude=slice(-10, 10)).mean(dim=("Time", "latitude"))
    z = height_from_zm_latlon(zds).values
    dtheta_dz = np.gradient(theta.values, z)
    N2 = g / theta.values * dtheta_dz
    mask = (z / 1000.0 >= 14.0) & (z / 1000.0 <= 18.0)
    N2_1418 = np.nanmean(N2[mask]) * 1e4

    metrics["T_100hPa"].append(T100)
    metrics["u_100hPa"].append(u100)
    metrics["u_70minus200"].append(shear)
    metrics["N2_14_18km"].append(N2_1418)
    metrics["CPT_height"].append(np.nanmean(CPTs[i][0]) / 1000.0)
    metrics["CPT_pressure"].append(np.nanmean(CPTs[i][1]) / 100.0)

fig, ax = plt.subplots(2, 3, figsize=(10, 5), constrained_layout=True)
ax = ax.ravel()
metric_labels = [
    ("T_100hPa", r"Tropical $T_{100}$ (K)"),
    ("u_100hPa", r"Tropical $u_{100}$ (m s$^{-1}$)"),
    ("u_70minus200", r"$u_{70} - u_{200}$ (m s$^{-1}$)"),
    ("N2_14_18km", r"$N^2_{14-18km}$ ($10^{-4}$ s$^{-2}$)"),
    ("CPT_height", "CPT height (km)"),
    ("CPT_pressure", "CPT pressure (hPa)"),
]

for j, (key, ylabel) in enumerate(metric_labels):
    vals = np.array(metrics[key], dtype=float)
    ax[j].bar(np.arange(len(dss)), vals, color=cols)
    ax[j].set_xticks(np.arange(len(dss)))
    ax[j].set_xticklabels(sim_names, rotation=30, ha="right")
    ax[j].set_ylabel(ylabel)
    for i in range(1, len(dss)):
        if np.isfinite(vals[i]) and np.isfinite(vals[0]):
            ax[j].text(i, vals[i], f"{vals[i] - vals[0]:+.2g}", ha="center", va="bottom", fontsize=8)

if save:
    fig.savefig(save_loc + "comp_utls_summary_metrics.png", dpi=300)
print("Done with figure: comp_utls_summary_metrics")


# ============================================================
# 13. Optional: scalar eastward/westward precipitation power metric
# ============================================================
# What to look for:
#   - Use this only if ndays is long enough for intraseasonal variability.
#   - A large east/west excess-power ratio would be the first hint of MJO-like
#     organisation, but only alongside Hovmoller evidence.

run_augmented_spectra_metrics = False

if run_augmented_spectra_metrics:
    ew_ratios = []
    for d, ds in enumerate(dss):
        precip = total_precip_accum(ds)
        pr = precip_rate_from_accum(precip)
        pr = pr.rename({"latitude": "lat", "longitude": "lon"}).load()

        # spd is samples per day.  With dt=6h, spd=4.
        spd = int(round(24 / dt)) if issubclass(ds.Time.values.dtype.type, numbers.Integral) else int(round(24 / dt))
        raw_spectrum = spacetime_power(
            pr, lataggreg="mean", latitude_bounds=(-10, 10), dosymmetries=True,
            segsize=int(96 / spd), noverlap=int(60 / spd), spd=spd
        )
        sym_smooth = smooth_wavefreq(raw_spectrum.sel(component="symmetric"), freq_ax=1)
        sym_sig = raw_spectrum.sel(component="symmetric") / sym_smooth
        excess = xr.where(sym_sig > 1.0, sym_sig - 1.0, 0.0)

        # Approximate MJO-like box: eastward k=1..3 and 20-100 day periods.
        freq = excess.frequency
        mjo_freq_mask = (freq >= 1 / 100.0) & (freq <= 1 / 20.0)
        east = excess.sel(wavenumber=slice(1, 3)).where(mjo_freq_mask, drop=True).sum().values.item()
        west = excess.sel(wavenumber=slice(-3, -1)).where(mjo_freq_mask, drop=True).sum().values.item()
        ew_ratios.append(east / west if west != 0 else np.nan)

    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    ax.bar(np.arange(len(dss)), ew_ratios, color=cols)
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(dss)))
    ax.set_xticklabels(sim_names, rotation=30, ha="right")
    ax.set_ylabel("Eastward / westward excess power\n(k=1-3, 20-100 d)")
    if save:
        fig.savefig(save_loc + "comp_mjo_box_east_west_power_ratio.png", dpi=300)
    print("Done with figure: comp_mjo_box_east_west_power_ratio")

print("All augmented diagnostics complete.")
