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
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw",
    "uzonal_isobaric", "umeridional_isobaric", "omg_isobaric",
    "rh_isobaric", "t_isobaric", "qv_isobaric", "theta",
    "uReconstructZonal", "cldfrac", "olrtoa", "zgrid", "pressure", "vorticity", "ke"
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
# Thermal-wind and realised-state diagnostics
# ============================================================
#
# WHAT TO LOOK FOR:
# The u-nudged and T-nudged experiments should not be interpreted as
# dynamically independent wind-only and temperature-only perturbations.
# Thermal-wind balance couples the meridional temperature gradient to
# vertical shear of the zonal wind.  These diagnostics therefore ask:
#
#   1. Does the u-nudged run generate a stronger realised TTL cooling than
#      the T-nudged run?
#   2. Is the realised wind anomaly approximately consistent with the
#      realised temperature anomaly through thermal-wind balance?
#   3. Do OLR/precip/PW/RH/omega/cloud responses scale mainly with realised
#      TTL temperature, or is there a residual response that points to a
#      wind/shear/wave pathway?
#
# Notes:
#   * The pressure-coordinate thermal-wind diagnostic is singular at the
#     equator because f -> 0.  The diagnostic is therefore masked inside
#     +/- tw_equator_mask_deg.
#   * This is most useful on the off-equatorial flanks of the tropical
#     anomaly, not exactly at the equator.
#   * The scaled-residual diagnostics assume approximate linearity.  They
#     are not a proof, but they are a good way to avoid accidentally calling
#     a stronger thermal response a direct wind response.


# -----------------------
# General extra helpers
# -----------------------

def total_precip_accum(ds):
    """Accumulated total precipitation in mm, including bucket counters."""
    return (
        ds["rainnc"] + ds["i_rainnc"] * 100.0
        + ds["rainc"] + ds["i_rainc"] * 100.0
    )


def precip_rate_from_accum(accum):
    """Convert accumulated precipitation to mm/day, using this script's dt/time_format logic."""
    if issubclass(accum.Time.values.dtype.type, numbers.Integral):
        return accum.diff("Time") / (dt / 24.0)
    dt_days = accum.Time.diff("Time") / np.timedelta64(1, "D")
    return accum.diff("Time") / dt_days


def tropical_mean_surface(da, lat_bounds=(-10, 10)):
    """Area-weighted tropical mean for lat/lon fields."""
    da = da.sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
    w = np.cos(np.deg2rad(da.latitude))
    dims = [d for d in ("latitude", "longitude") if d in da.dims]
    return da.weighted(w).mean(dim=dims)


def tropical_mean_isobaric(da, p=None, p_bounds=None, lat_bounds=(-10, 10)):
    """Area-weighted tropical mean for isobaric fields."""
    x = da
    if p is not None:
        x = x.sel(pres=p, method="nearest")
    if p_bounds is not None:
        pmin, pmax = min(p_bounds), max(p_bounds)
        x = x.sel(pres=slice(pmin, pmax))
    x = x.sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
    w = np.cos(np.deg2rad(x.latitude))
    dims = [d for d in ("Time", "pres", "latitude") if d in x.dims]
    return x.weighted(w).mean(dim=dims)


def rms_weighted_latpres(da, lat_bounds=(-60, 60), p_bounds=(50, 300)):
    """Weighted RMS over a pressure-latitude box."""
    x = da.sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
    if "pres" in x.dims and p_bounds is not None:
        pmin, pmax = min(p_bounds), max(p_bounds)
        x = x.sel(pres=slice(pmin, pmax))
    w = np.cos(np.deg2rad(x.latitude))
    dims = [d for d in ("pres", "latitude") if d in x.dims]
    return np.sqrt((x**2).weighted(w).mean(dim=dims)).values.item()


def pattern_corr_latpres(a, b, lat_bounds=(-60, 60), p_bounds=(50, 300)):
    """Cosine-latitude weighted pattern correlation over pressure-latitude box."""
    aa, bb = xr.align(a, b, join="inner")
    aa = aa.sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
    bb = bb.sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
    if "pres" in aa.dims and p_bounds is not None:
        pmin, pmax = min(p_bounds), max(p_bounds)
        aa = aa.sel(pres=slice(pmin, pmax))
        bb = bb.sel(pres=slice(pmin, pmax))

    w = np.cos(np.deg2rad(aa.latitude))
    dims = [d for d in ("pres", "latitude") if d in aa.dims]
    aa = aa - aa.weighted(w).mean(dim=dims)
    bb = bb - bb.weighted(w).mean(dim=dims)
    num = (aa * bb).weighted(w).mean(dim=dims)
    den = np.sqrt((aa**2).weighted(w).mean(dim=dims) * (bb**2).weighted(w).mean(dim=dims))
    return (num / den).values.item()


def precip_rate_mean_map(ds):
    """Time-mean precipitation-rate map in mm/day."""
    pr = precip_rate_from_accum(total_precip_accum(ds))
    return pr.mean(dim="Time")


def zmean_time_mean(da):
    """Time mean, zonal mean if longitude is present."""
    dims = [d for d in ("Time", "longitude") if d in da.dims]
    return da.mean(dim=dims)


# -----------------------
# Thermal wind helpers
# -----------------------

def thermal_wind_terms(T_da, u_da, tw_equator_mask_deg=5.0):
    """
    Diagnose pressure-coordinate thermal-wind balance.

    Inputs must have dimensions (pres, latitude), with pres in hPa.

    Approximate relation used:
        d u / d ln(p) = R/(f a) * dT/dphi

    where phi is latitude in radians.  This relation is not valid at the
    equator, so a latitude band around the equator is masked.
    """
    R = 287.0
    omega = 7.2921159e-5
    a = 6.371e6

    T = T_da.transpose("pres", "latitude")
    u = u_da.transpose("pres", "latitude")

    lat_rad = np.deg2rad(T.latitude.values)
    lnp = np.log(T.pres.values)
    f = 2.0 * omega * np.sin(lat_rad)

    dT_dphi = np.gradient(T.values, lat_rad, axis=1)
    du_dlnp = np.gradient(u.values, lnp, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        du_dlnp_tw =  R / (f[None, :] * a) * dT_dphi

    mask = np.abs(T.latitude.values) < tw_equator_mask_deg
    du_dlnp[:, mask] = np.nan
    du_dlnp_tw[:, mask] = np.nan

    resid = du_dlnp - du_dlnp_tw
    coords = {"pres": T.pres, "latitude": T.latitude}

    return (
        xr.DataArray(du_dlnp, dims=("pres", "latitude"), coords=coords, name="du_dlnp_observed"),
        xr.DataArray(du_dlnp_tw, dims=("pres", "latitude"), coords=coords, name="du_dlnp_thermal_wind"),
        xr.DataArray(resid, dims=("pres", "latitude"), coords=coords, name="du_dlnp_residual"),
    )


def get_isobaric_anomaly(zds, zds_control, var):
    """Time-mean isobaric anomaly relative to control, on this run's grid."""
    field = isobaric_lat_mean(zds[var])
    control = isobaric_lat_mean(zds_control[var]).interp(
        pres=field.pres, latitude=field.latitude, kwargs={"fill_value": np.nan}
    )
    return field - control


def mean_TTL_temperature_anomaly(zds, zds_control, p_bounds=(70, 150), lat_bounds=(-10, 10)):
    """Mean realised TTL temperature anomaly relative to control."""
    dT = get_isobaric_anomaly(zds, zds_control, "t_isobaric")
    pmin, pmax = min(p_bounds), max(p_bounds)
    dT = dT.sel(pres=slice(pmin, pmax)).sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
    w = np.cos(np.deg2rad(dT.latitude))
    return dT.weighted(w).mean(dim=("pres", "latitude")).values.item()


def mean_TTL_wind_anomaly(zds, zds_control, p_bounds=(70, 150), lat_bounds=(-10, 10)):
    """Mean realised TTL zonal-wind anomaly relative to control."""
    du = get_isobaric_anomaly(zds, zds_control, "uzonal_isobaric")
    pmin, pmax = min(p_bounds), max(p_bounds)
    du = du.sel(pres=slice(pmin, pmax)).sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
    w = np.cos(np.deg2rad(du.latitude))
    return du.weighted(w).mean(dim=("pres", "latitude")).values.item()


def scaled_residual_latpres(var, alpha, run_u=1, run_T=2, control=0):
    """
    Return dX_u, dX_T, and dX_u - alpha*dX_T for isobaric fields.
    """
    dXu = get_isobaric_anomaly(dss_zm[run_u], dss_zm[control], var)
    dXT = get_isobaric_anomaly(dss_zm[run_T], dss_zm[control], var)
    dXT_on_u = dXT.interp(pres=dXu.pres, latitude=dXu.latitude, kwargs={"fill_value": np.nan})
    resid = dXu - alpha * dXT_on_u
    return dXu, dXT, resid


def scaled_residual_map(var, alpha, run_u=1, run_T=2, control=0):
    """
    Return dX_u, dX_T, and dX_u - alpha*dX_T for surface/latlon fields.
    Special var name 'precip_rate' uses time-mean precipitation rate.
    """
    if var == "precip_rate":
        Xc = precip_rate_mean_map(dss[control])
        Xu = precip_rate_mean_map(dss[run_u])
        XT = precip_rate_mean_map(dss[run_T])
    else:
        Xc = dss[control][var].mean(dim="Time")
        Xu = dss[run_u][var].mean(dim="Time")
        XT = dss[run_T][var].mean(dim="Time")

    dXu = Xu - Xc.interp(latitude=Xu.latitude, longitude=Xu.longitude, kwargs={"fill_value": np.nan})
    dXT = XT - Xc.interp(latitude=XT.latitude, longitude=XT.longitude, kwargs={"fill_value": np.nan})
    dXT_on_u = dXT.interp(latitude=dXu.latitude, longitude=dXu.longitude, kwargs={"fill_value": np.nan})
    resid = dXu - alpha * dXT_on_u
    return dXu, dXT, resid


# -----------------------
# 1. Realised thermal/wind scaling numbers
# -----------------------

TTL_p_bounds = (70, 150)
TTL_lat_bounds = (-10, 10)

dT_TTL_u = mean_TTL_temperature_anomaly(dss_zm[1], dss_zm[0], p_bounds=TTL_p_bounds, lat_bounds=TTL_lat_bounds)
dT_TTL_T = mean_TTL_temperature_anomaly(dss_zm[2], dss_zm[0], p_bounds=TTL_p_bounds, lat_bounds=TTL_lat_bounds)
du_TTL_u = mean_TTL_wind_anomaly(dss_zm[1], dss_zm[0], p_bounds=TTL_p_bounds, lat_bounds=TTL_lat_bounds)
du_TTL_T = mean_TTL_wind_anomaly(dss_zm[2], dss_zm[0], p_bounds=TTL_p_bounds, lat_bounds=TTL_lat_bounds)

alpha_TTL = dT_TTL_u / dT_TTL_T if np.isfinite(dT_TTL_T) and abs(dT_TTL_T) > 1e-12 else np.nan

print("\nThermal scaling based on realised TTL temperature anomaly")
print("--------------------------------------------------------")
print(f"TTL layer: {TTL_p_bounds[0]}-{TTL_p_bounds[1]} hPa, {TTL_lat_bounds[0]} to {TTL_lat_bounds[1]} deg")
print(f"u-nudged mean ΔT_TTL = {dT_TTL_u:.3f} K")
print(f"T-nudged mean ΔT_TTL = {dT_TTL_T:.3f} K")
print(f"alpha_TTL = ΔT_u / ΔT_T = {alpha_TTL:.3f}")
print(f"u-nudged mean Δu_TTL = {du_TTL_u:.3f} m s-1")
print(f"T-nudged mean Δu_TTL = {du_TTL_T:.3f} m s-1")
print("WHAT TO LOOK FOR: if |alpha_TTL| > 1, the u-nudged run has the stronger realised TTL thermal anomaly.")


# -----------------------
# 2. Thermal-wind balance of total fields
# -----------------------

fig, ax = plt.subplots(len(dss_zm), 3, figsize=(10, 3.1 * len(dss_zm)),
                       constrained_layout=True, sharex=True, sharey=True)
if len(dss_zm) == 1:
    ax = np.array([ax])

im_obs = im_tw = im_resid = None
for i, zds in enumerate(dss_zm):
    Tbar = isobaric_lat_mean(zds.t_isobaric)
    ubar = isobaric_lat_mean(zds.uzonal_isobaric)
    du_obs, du_tw, tw_resid = thermal_wind_terms(Tbar, ubar, tw_equator_mask_deg=5.0)

    levels = symmetric_levels(25.0, ndiv=13)
    im_obs = ax[i, 0].contourf(zds.latitude, zds.pres, du_obs, levels=levels, cmap="bwr", extend="both")
    im_tw = ax[i, 1].contourf(zds.latitude, zds.pres, du_tw, levels=levels, cmap="bwr", extend="both")
    im_resid = ax[i, 2].contourf(zds.latitude, zds.pres, tw_resid, levels=levels, cmap="bwr", extend="both")

    for j in range(3):
        ax[i, j].plot(zds.latitude.values, CPTs[i][1] / 100.0, color="k", lw=1, ls=":")
        ax[i, j].annotate(sim_names[i], (-55, 18), fontweight="bold", bbox=dict(facecolor="w", alpha=0.7))
        ax[i, j].set_xlim(-60, 60)
        ax[i, j].set_ylim(1000, 50)
        ax[i, j].set_yscale("log")
        if i == len(dss_zm) - 1:
            ax[i, j].set_xlabel("Latitude")
        else:
            ax[i, j].set_xticklabels([])
    ax[i, 0].set_ylabel("Pressure (hPa)")

ax[0, 0].invert_yaxis()
ax[0, 0].set_title(r"Observed $\partial u/\partial \,\ln p$")
ax[0, 1].set_title("TW-implied shear")
ax[0, 2].set_title("Observed - implied")
fig.colorbar(im_obs, ax=ax[:, 0], location="right", label=r"m s$^{-1}$ per ln p")
fig.colorbar(im_tw, ax=ax[:, 1], location="right", label=r"m s$^{-1}$ per ln p")
fig.colorbar(im_resid, ax=ax[:, 2], location="right", label=r"m s$^{-1}$ per ln p")

if save:
    fig.savefig(save_loc + "comp_thermal_wind_total_fields.png", dpi=300)
print("Done with figure: comp_thermal_wind_total_fields")
print("WHAT TO LOOK FOR: large residuals outside +/-5 deg imply the realised flow is not simply balanced by the realised T gradient.")


# -----------------------
# 3. Thermal-wind balance of anomalies relative to control
# -----------------------

fig, ax = plt.subplots(len(dss_zm), 3, figsize=(10, 3.1 * len(dss_zm)),
                       constrained_layout=True, sharex=True, sharey=True)
if len(dss_zm) == 1:
    ax = np.array([ax])

im_obs = im_tw = im_resid = None
thermal_wind_metric_rows = []

for i, zds in enumerate(dss_zm):
    if i == 0:
        # Plot a blank/control row for consistency.
        dT = isobaric_lat_mean(zds.t_isobaric) * 0.0
        du = isobaric_lat_mean(zds.uzonal_isobaric) * 0.0
    else:
        dT = get_isobaric_anomaly(zds, dss_zm[0], "t_isobaric")
        du = get_isobaric_anomaly(zds, dss_zm[0], "uzonal_isobaric")

    du_obs, du_tw, tw_resid = thermal_wind_terms(dT, du, tw_equator_mask_deg=5.0)

    levels = symmetric_levels(12.0, ndiv=13)
    im_obs = ax[i, 0].contourf(zds.latitude, zds.pres, du_obs, levels=levels, cmap="bwr", extend="both")
    im_tw = ax[i, 1].contourf(zds.latitude, zds.pres, du_tw, levels=levels, cmap="bwr", extend="both")
    im_resid = ax[i, 2].contourf(zds.latitude, zds.pres, tw_resid, levels=levels, cmap="bwr", extend="both")

    if i > 0:
        rms_obs = rms_weighted_latpres(du_obs)
        rms_tw = rms_weighted_latpres(du_tw)
        rms_resid = rms_weighted_latpres(tw_resid)
        corr = pattern_corr_latpres(du_obs, du_tw)
        thermal_wind_metric_rows.append([sim_names[i], rms_obs, rms_tw, rms_resid, corr])

    for j in range(3):
        ax[i, j].plot(zds.latitude.values, CPTs[i][1] / 100.0, color="k", lw=1, ls=":")
        ax[i, j].annotate(sim_names[i], (-55, 18), fontweight="bold", bbox=dict(facecolor="w", alpha=0.7))
        ax[i, j].set_xlim(-60, 60)
        ax[i, j].set_ylim(1000, 50)
        ax[i, j].set_yscale("log")
        if i == len(dss_zm) - 1:
            ax[i, j].set_xlabel("Latitude")
        else:
            ax[i, j].set_xticklabels([])
    ax[i, 0].set_ylabel("Pressure (hPa)")

ax[0, 0].invert_yaxis()
ax[0, 0].set_title(r"Observed anomaly shear")
ax[0, 1].set_title("TW-implied anomaly shear")
ax[0, 2].set_title("Anomaly residual")
fig.colorbar(im_obs, ax=ax[:, 0], location="right", label=r"m s$^{-1}$ per ln p")
fig.colorbar(im_tw, ax=ax[:, 1], location="right", label=r"m s$^{-1}$ per ln p")
fig.colorbar(im_resid, ax=ax[:, 2], location="right", label=r"m s$^{-1}$ per ln p")

if save:
    fig.savefig(save_loc + "comp_thermal_wind_anomaly_fields.png", dpi=300)
print("Done with figure: comp_thermal_wind_anomaly_fields")
print("WHAT TO LOOK FOR: if the u-nudged anomaly shear is well matched by the TW-implied shear, its cold anomaly is part of a balanced adjustment.")

thermal_wind_metrics = pd.DataFrame(
    thermal_wind_metric_rows,
    columns=["simulation", "rms_observed_shear", "rms_TW_implied_shear", "rms_residual", "pattern_corr_obs_vs_TW"]
)
print("\nThermal-wind anomaly metrics over 50-300 hPa and 60S-60N, excluding +/-5 deg:")
print(thermal_wind_metrics.to_string(index=False))
if save:
    thermal_wind_metrics.to_csv(save_loc + "thermal_wind_anomaly_metrics.csv", index=False)


# -----------------------
# 4. Realised-state summary bar chart
# -----------------------

summary_rows = []
for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
    T100 = tropical_mean_isobaric(zds.t_isobaric, p=100, lat_bounds=(-10, 10)).values.item()
    u100 = tropical_mean_isobaric(zds.uzonal_isobaric, p=100, lat_bounds=(-10, 10)).values.item()
    Tttl = tropical_mean_isobaric(zds.t_isobaric, p_bounds=TTL_p_bounds, lat_bounds=(-10, 10)).values.item()
    uttl = tropical_mean_isobaric(zds.uzonal_isobaric, p_bounds=TTL_p_bounds, lat_bounds=(-10, 10)).values.item()

    theta = zds.theta.sel(latitude=slice(-10, 10)).mean(dim=("Time", "latitude"))
    z = height_from_zm_latlon(zds).values
    dtheta_dz = np.gradient(theta.values, z)
    N2 = 9.80665 / theta.values * dtheta_dz
    zmask = (z / 1000.0 >= 14) & (z / 1000.0 <= 18)
    N2_1418 = np.nanmean(N2[zmask]) * 1e4

    olr = tropical_mean_surface(ds.olrtoa, lat_bounds=(-10, 10)).mean(dim="Time").values.item()
    pw = tropical_mean_surface(ds.precipw, lat_bounds=(-10, 10)).mean(dim="Time").values.item()
    pr = tropical_mean_surface(precip_rate_from_accum(total_precip_accum(ds)), lat_bounds=(-10, 10)).mean(dim="Time").values.item()

    summary_rows.append([sim_names[i], T100, u100, Tttl, uttl, N2_1418, olr, pw, pr])

summary_df = pd.DataFrame(
    summary_rows,
    columns=["simulation", "T100_K", "u100_ms-1", "T_TTL_K", "u_TTL_ms-1", "N2_14_18km_1e-4s-2", "OLR_Wm-2", "PW_kgm-2", "precip_mmday-1"]
)

# Add anomalies relative to control for quick inspection
for col in summary_df.columns[1:]:
    summary_df["d" + col] = summary_df[col] - summary_df.loc[0, col]

print("\nRealised-state summary:")
print(summary_df.to_string(index=False))
if save:
    summary_df.to_csv(save_loc + "realised_state_summary.csv", index=False)

plot_cols = [
    ("dT100_K", r"$\Delta T_{100}$ (K)"),
    ("du100_ms-1", r"$\Delta u_{100}$ (m s$^{-1}$)"),
    ("dT_TTL_K", r"$\Delta T_{TTL}$ (K)"),
    ("du_TTL_ms-1", r"$\Delta u_{TTL}$ (m s$^{-1}$)"),
    ("dN2_14_18km_1e-4s-2", r"$\Delta N^2_{14-18 km}$ ($10^{-4}$ s$^{-2}$)"),
    ("dOLR_Wm-2", r"$\Delta$ OLR (W m$^{-2}$)"),
    ("dPW_kgm-2", r"$\Delta$ PW (kg m$^{-2}$)"),
    ("dprecip_mmday-1", r"$\Delta$ precip (mm day$^{-1}$)"),
]

fig, ax = plt.subplots(2, 4, figsize=(11, 5.2), constrained_layout=True)
ax = ax.ravel()
for j, (col, ylabel) in enumerate(plot_cols):
    vals = summary_df[col].values
    ax[j].bar(np.arange(len(sim_names)), vals, color=cols)
    ax[j].axhline(0, color="grey", lw=0.6)
    ax[j].set_xticks(np.arange(len(sim_names)))
    ax[j].set_xticklabels(sim_names, rotation=30, ha="right")
    ax[j].set_ylabel(ylabel)

if save:
    fig.savefig(save_loc + "comp_realised_state_summary_bars.png", dpi=300)
print("Done with figure: comp_realised_state_summary_bars")
print("WHAT TO LOOK FOR: compare dT_TTL and dN2 with OLR/PW/precip. If the u-nudged run is colder and has the larger cloud/radiative response, analyse it as a stronger realised thermal perturbation first.")


# -----------------------
# 5. Response plotted against realised TTL temperature anomaly
# -----------------------

fig, ax = plt.subplots(1, 4, figsize=(12, 3.3), constrained_layout=True)
x = summary_df["dT_TTL_K"].values
scatter_items = [
    ("dOLR_Wm-2", r"$\Delta$ OLR (W m$^{-2}$)"),
    ("dprecip_mmday-1", r"$\Delta$ precip (mm day$^{-1}$)"),
    ("dPW_kgm-2", r"$\Delta$ PW (kg m$^{-2}$)"),
    ("dN2_14_18km_1e-4s-2", r"$\Delta N^2_{14-18km}$"),
]
for j, (col, ylabel) in enumerate(scatter_items):
    y = summary_df[col].values
    for i in range(len(sim_names)):
        ax[j].scatter(x[i], y[i], color=cols[i], s=60)
        ax[j].annotate(sim_names[i], (x[i], y[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax[j].axhline(0, color="grey", lw=0.5)
    ax[j].axvline(0, color="grey", lw=0.5)
    ax[j].set_xlabel(r"$\Delta T_{TTL}$ (K)")
    ax[j].set_ylabel(ylabel)
    ax[j].grid(alpha=0.2)

if save:
    fig.savefig(save_loc + "comp_responses_vs_realised_TTL_temperature.png", dpi=300)
print("Done with figure: comp_responses_vs_realised_TTL_temperature")
print("WHAT TO LOOK FOR: if the two nudged runs line up by ΔT_TTL, the response is probably thermal/static-stability controlled rather than set by the nudged variable label.")


# -----------------------
# 6. Scaled-residual maps for surface/column fields
# -----------------------

surface_residual_specs = [
    ("olrtoa", r"OLR anomaly (W m$^{-2}$)", 20.0, "comp_scaled_residual_olrtoa.png"),
    ("precipw", r"PW anomaly (kg m$^{-2}$)", 4.0, "comp_scaled_residual_precipw.png"),
    ("precip_rate", r"Precip anomaly (mm day$^{-1}$)", 2.0, "comp_scaled_residual_precip.png"),
]

for var, label, max_abs, fname in surface_residual_specs:
    dXu, dXT, resid = scaled_residual_map(var, alpha_TTL)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True, sharex=True, sharey=True)
    levels = symmetric_levels(max_abs, ndiv=13)
    im0 = ax[0].contourf(dXu.longitude, dXu.latitude, dXu, levels=levels, cmap="bwr", extend="both")
    im1 = ax[1].contourf(dXT.longitude, dXT.latitude, dXT, levels=levels, cmap="bwr", extend="both")
    im2 = ax[2].contourf(resid.longitude, resid.latitude, resid, levels=levels, cmap="bwr", extend="both")

    ax[0].set_title(r"u-nudged $\Delta X$")
    ax[1].set_title(r"T-nudged $\Delta X$")
    ax[2].set_title(r"$\Delta X_u - \alpha \Delta X_T$")
    for a in ax:
        a.set_xlabel("Longitude")
        a.set_ylabel("Latitude")
    fig.suptitle(f"Scaled thermal residual for {var}; alpha={alpha_TTL:.2f}")
    fig.colorbar(im2, ax=ax, location="right", label=label)

    if save:
        fig.savefig(save_loc + fname, dpi=300)
    print(f"Done with figure: {fname}")

print("WHAT TO LOOK FOR: small residual maps mean the u-nudged response is largely the T-nudged response scaled by realised TTL cooling. Coherent residuals imply an additional wind/shear/dynamical pathway.")


# -----------------------
# 7. Scaled-residual pressure-latitude sections
# -----------------------

latpres_residual_specs = [
    ("t_isobaric", r"T anomaly (K)", 8.0, "comp_scaled_residual_T_latpres.png"),
    ("uzonal_isobaric", r"u anomaly (m s$^{-1}$)", 8.0, "comp_scaled_residual_u_latpres.png"),
    ("rh_isobaric", r"RH anomaly", 10.0, "comp_scaled_residual_RH_latpres.png"),
    ("omg_isobaric", r"omega anomaly (Pa s$^{-1}$)", 1.0, "comp_scaled_residual_omega_latpres.png"),
]

for var, label, max_abs, fname in latpres_residual_specs:
    if var not in dss_zm[0].data_vars or var not in dss_zm[1].data_vars or var not in dss_zm[2].data_vars:
        print(f"Skipping {var}: not present in all zm datasets")
        continue

    dXu, dXT, resid = scaled_residual_latpres(var, alpha_TTL)
    levels = symmetric_levels(max_abs, ndiv=13)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True, sharex=True, sharey=True)
    im0 = ax[0].contourf(dXu.latitude, dXu.pres, dXu, levels=levels, cmap="bwr", extend="both")
    im1 = ax[1].contourf(dXT.latitude, dXT.pres, dXT, levels=levels, cmap="bwr", extend="both")
    im2 = ax[2].contourf(resid.latitude, resid.pres, resid, levels=levels, cmap="bwr", extend="both")

    for j, a in enumerate(ax):
        a.set_xlim(-60, 60)
        a.set_ylim(1000, 50)
        a.set_yscale("log")
        a.set_xlabel("Latitude")
        a.set_ylabel("Pressure (hPa)")
        if j == 0:
            a.plot(dss_zm[1].latitude.values, CPTs[1][1] / 100.0, color="k", lw=1, ls=":")
        elif j == 1:
            a.plot(dss_zm[2].latitude.values, CPTs[2][1] / 100.0, color="k", lw=1, ls=":")
        else:
            a.plot(dss_zm[1].latitude.values, CPTs[1][1] / 100.0, color="k", lw=1, ls=":")
    ax[0].invert_yaxis()

    ax[0].set_title(r"u-nudged $\Delta X$")
    ax[1].set_title(r"T-nudged $\Delta X$")
    ax[2].set_title(r"$\Delta X_u - \alpha \Delta X_T$")
    fig.suptitle(f"Scaled thermal residual for {var}; alpha={alpha_TTL:.2f}")
    fig.colorbar(im2, ax=ax, location="right", label=label)

    if save:
        fig.savefig(save_loc + fname, dpi=300)
    print(f"Done with figure: {fname}")

print("WHAT TO LOOK FOR: omega/RH residuals are particularly useful. A small T residual but large omega residual suggests comparable thermal forcing but different circulation adjustment.")


# -----------------------
# 8. N2 scaled residual in latitude-height coordinates
# -----------------------


def N2_latheight_from_zds(zds):
    """Time-mean N2 on model height levels as (z, latitude)."""
    theta = model_lat_field(zds.theta.mean(dim="Time"))
    z = height_from_zm_latlon(zds).values
    dtheta_dz = np.gradient(theta.values, z, axis=0)
    N2_vals = 9.80665 / theta.values * dtheta_dz
    return xr.DataArray(
        N2_vals,
        dims=("z", "latitude"),
        coords={"z": theta.z.values, "latitude": theta.latitude.values},
        name="N2",
    )

N2_control = N2_latheight_from_zds(dss_zm[0])
N2_u = N2_latheight_from_zds(dss_zm[1])
N2_T = N2_latheight_from_zds(dss_zm[2])

dN2_u = N2_u - N2_control.interp(z=N2_u.z, latitude=N2_u.latitude, kwargs={"fill_value": np.nan})
dN2_T = N2_T - N2_control.interp(z=N2_T.z, latitude=N2_T.latitude, kwargs={"fill_value": np.nan})
dN2_T_on_u = dN2_T.interp(z=dN2_u.z, latitude=dN2_u.latitude, kwargs={"fill_value": np.nan})
dN2_resid = dN2_u - alpha_TTL * dN2_T_on_u

fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True, sharex=True, sharey=True)
levels = symmetric_levels(1.5, ndiv=13)
lat2d = np.broadcast_to(dN2_u.latitude.values[None, :], dN2_u.shape)
z2d = np.broadcast_to(dN2_u.z.values[:, None], dN2_u.shape)
lat2d_T = np.broadcast_to(dN2_T.latitude.values[None, :], dN2_T.shape)
z2d_T = np.broadcast_to(dN2_T.z.values[:, None], dN2_T.shape)

im0 = ax[0].contourf(lat2d, z2d / 1000.0, dN2_u.values * 1e4, levels=levels, cmap="bwr", extend="both")
im1 = ax[1].contourf(lat2d_T, z2d_T / 1000.0, dN2_T.values * 1e4, levels=levels, cmap="bwr", extend="both")
im2 = ax[2].contourf(lat2d, z2d / 1000.0, dN2_resid.values * 1e4, levels=levels, cmap="bwr", extend="both")

ax[0].plot(dss_zm[1].latitude.values, CPTs[1][0] / 1000.0, color="k", lw=1, ls=":")
ax[1].plot(dss_zm[2].latitude.values, CPTs[2][0] / 1000.0, color="k", lw=1, ls=":")
ax[2].plot(dss_zm[1].latitude.values, CPTs[1][0] / 1000.0, color="k", lw=1, ls=":")

ax[0].set_title(r"u-nudged $\Delta N^2$")
ax[1].set_title(r"T-nudged $\Delta N^2$")
ax[2].set_title(r"$\Delta N^2_u - \alpha \Delta N^2_T$")
for a in ax:
    set_latheight_labels(a)
    a.set_xlim(-60, 60)
    a.set_ylim(0, 25)
fig.suptitle(f"Scaled thermal residual for static stability; alpha={alpha_TTL:.2f}")
fig.colorbar(im2, ax=ax, location="right", label=r"$\Delta N^2$ ($10^{-4}$ s$^{-2}$)")

if save:
    fig.savefig(save_loc + "comp_scaled_residual_N2_latheight.png", dpi=300)
print("Done with figure: comp_scaled_residual_N2_latheight")
print("WHAT TO LOOK FOR: if N2 residual is small, the u-nudged stability response is mostly the stronger realised cooling. If not, wind-induced circulation/shear may be modifying stability independently.")


# -----------------------
# 9. Cloud-fraction scaled residual, using same cldfrac logic as above
# -----------------------


def get_zm_cldfrac_for_run(i):
    """Return time-mean zonal-mean cldfrac as (z, latitude), following the existing script's logic."""
    ds = dss[i]
    zds = dss_zm[i]
    if "cldfrac" in zds.data_vars:
        cld = zds.cldfrac.mean(dim="Time")
        zmid = height_from_zm_latlon(zds)
        cld, _, _ = lat_height_mesh(cld, zmid, zds.latitude.values)
        return cld
    if "cldfrac" in ds.data_vars:
        cld = ds.cldfrac.mean(dim=("Time", "longitude"))
        cld = model_lat_field(cld)
        # Keep the same optional control symmetrisation used in the earlier cloud-fraction section.
        if i == 0 and "hemispheric_symmetrise_lat" in globals():
            cld = hemispheric_symmetrise_lat(cld)
        return cld
    return None

cld_control = get_zm_cldfrac_for_run(0)
cld_u = get_zm_cldfrac_for_run(1)
cld_T = get_zm_cldfrac_for_run(2)

if cld_control is not None and cld_u is not None and cld_T is not None:
    dcld_u = cld_u - cld_control.interp(z=cld_u.z, latitude=cld_u.latitude, kwargs={"fill_value": np.nan})
    dcld_T = cld_T - cld_control.interp(z=cld_T.z, latitude=cld_T.latitude, kwargs={"fill_value": np.nan})
    dcld_T_on_u = dcld_T.interp(z=dcld_u.z, latitude=dcld_u.latitude, kwargs={"fill_value": np.nan})
    dcld_resid = dcld_u - alpha_TTL * dcld_T_on_u

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True, sharex=True, sharey=True)
    levels = symmetric_levels(0.12, ndiv=13)
    lat2d = np.broadcast_to(dcld_u.latitude.values[None, :], dcld_u.shape)
    z2d = np.broadcast_to(dcld_u.z.values[:, None], dcld_u.shape)
    lat2d_T = np.broadcast_to(dcld_T.latitude.values[None, :], dcld_T.shape)
    z2d_T = np.broadcast_to(dcld_T.z.values[:, None], dcld_T.shape)

    im0 = ax[0].contourf(lat2d, z2d / 1000.0, dcld_u.values, levels=levels, cmap="bwr", extend="both")
    im1 = ax[1].contourf(lat2d_T, z2d_T / 1000.0, dcld_T.values, levels=levels, cmap="bwr", extend="both")
    im2 = ax[2].contourf(lat2d, z2d / 1000.0, dcld_resid.values, levels=levels, cmap="bwr", extend="both")

    ax[0].plot(dss_zm[1].latitude.values, CPTs[1][0] / 1000.0, color="k", lw=1, ls=":")
    ax[1].plot(dss_zm[2].latitude.values, CPTs[2][0] / 1000.0, color="k", lw=1, ls=":")
    ax[2].plot(dss_zm[1].latitude.values, CPTs[1][0] / 1000.0, color="k", lw=1, ls=":")

    ax[0].set_title(r"u-nudged $\Delta$cldfrac")
    ax[1].set_title(r"T-nudged $\Delta$cldfrac")
    ax[2].set_title(r"$\Delta C_u - \alpha \Delta C_T$")
    for a in ax:
        set_latheight_labels(a)
        a.set_xlim(-60, 60)
        a.set_ylim(0, 25)
    fig.suptitle(f"Scaled thermal residual for cloud fraction; alpha={alpha_TTL:.2f}")
    fig.colorbar(im2, ax=ax, location="right", label="Cloud-fraction anomaly")

    if save:
        fig.savefig(save_loc + "comp_scaled_residual_cldfrac_latheight.png", dpi=300)
    print("Done with figure: comp_scaled_residual_cldfrac_latheight")
    print("WHAT TO LOOK FOR: a small residual suggests cloud-depth changes mainly scale with realised TTL cooling. A large upper-cloud residual in u-nudged suggests an additional dynamical/shear/wave pathway.")
else:
    print("Skipping cloud-fraction scaled residual: cldfrac not present in all runs")


# -----------------------
# 10. Compact residual summary
# -----------------------

residual_rows = []

# Surface metrics: tropical mean residuals
for var, pretty in [("olrtoa", "OLR"), ("precipw", "PW"), ("precip_rate", "precip")]:
    dXu, dXT, resid = scaled_residual_map(var, alpha_TTL)
    trop_resid = tropical_mean_surface(resid, lat_bounds=(-10, 10)).values.item()
    trop_du = tropical_mean_surface(dXu, lat_bounds=(-10, 10)).values.item()
    trop_dT = tropical_mean_surface(dXT, lat_bounds=(-10, 10)).values.item()
    residual_rows.append([pretty, "surface_tropical_mean", trop_du, trop_dT, trop_resid])

# Isobaric metrics: TTL tropical mean residuals
for var, pretty in [("t_isobaric", "T"), ("uzonal_isobaric", "u"), ("rh_isobaric", "RH"), ("omg_isobaric", "omega")]:
    if var in dss_zm[0].data_vars and var in dss_zm[1].data_vars and var in dss_zm[2].data_vars:
        dXu, dXT, resid = scaled_residual_latpres(var, alpha_TTL)
        pmin, pmax = min(TTL_p_bounds), max(TTL_p_bounds)
        lat0, lat1 = TTL_lat_bounds
        for name, field in [("dXu", dXu), ("dXT", dXT), ("resid", resid)]:
            field_box = field.sel(pres=slice(pmin, pmax)).sel(latitude=slice(lat0, lat1))
            w = np.cos(np.deg2rad(field_box.latitude))
            val = field_box.weighted(w).mean(dim=("pres", "latitude")).values.item()
            if name == "dXu":
                du_val = val
            elif name == "dXT":
                dT_val = val
            else:
                resid_val = val
        residual_rows.append([pretty, "TTL_tropical_mean", du_val, dT_val, resid_val])

residual_df = pd.DataFrame(
    residual_rows,
    columns=["variable", "metric", "u_nudged_delta", "T_nudged_delta", "scaled_residual_u_minus_alphaT"]
)
print("\nScaled residual summary:")
print(residual_df.to_string(index=False))
if save:
    residual_df.to_csv(save_loc + "scaled_thermal_residual_summary.csv", index=False)

print("\nInterpretation guide:")
print("  * If scaled_residual is small: the u-nudged response is mostly the T-nudged response scaled by realised TTL ΔT.")
print("  * If scaled_residual remains large for u/omega/RH but not T/N2: likely extra dynamical/shear/secondary-circulation effect.")
print("  * If scaled_residual remains large for cloud/OLR/precip: the u-nudged run is doing something beyond simple thermal scaling, worth examining with wave/eddy diagnostics.")

