#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os, glob, sys, numbers

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

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
from diagnostic_helpers import *

save = True
save_loc = '/home/cwp29/tmp/mpas/crf_on/final/thermal_wind/'

# Set to [] or None to run all figures.  Otherwise list the numbered figures
# to regenerate, e.g. [4, 5, 6] for the scaled-residual lat-lon maps.
#
# Figure numbers:
#   1:  thermal_wind_total_fields
#   2:  realised_state_summary_bars
#   3:  responses_vs_realised_TTL_temperature
#   4:  scaled_residual_olrtoa
#   5:  scaled_residual_precipw
#   6:  scaled_residual_precip
#   7:  scaled_residual_T_latheight
#   8:  scaled_residual_u_latheight
#   9:  scaled_residual_RH_latheight
#   10: scaled_residual_w_latheight
#   11: scaled_residual_N2_latheight
#   12: scaled_residual_cldfrac_latheight
#   13: thermal_wind_temperature_gradient_total_fields
figures_to_run = []
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)

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

dss = []
dss_zm = []

needed_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw",
    "theta", "pressure", "uReconstructZonal", "relhum", "w",
    "cldfrac", "olrtoa", "zgrid", "vorticity", "ke"
]

check_vars = ["theta", "pressure", "uReconstructZonal"]


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


def run_all_figures():
    return all_figures_requested(figures_to_run)


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
    ds = standardize_latlon_dataset(ds, needed_vars)

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
    zds = standardize_latlon_dataset(zds, needed_vars)

    # cheaper if data are exactly 6-hourly and aligned
    #ds = ds.coarsen(Time=4, boundary="trim").mean()
    #zds = zds.coarsen(Time=4, boundary="trim").mean()

    print(ds)
    print(zds)
    dss.append(ds)
    dss_zm.append(zds)

def annotate_scaled_residual_labels(ax, run_u=1, run_T=2, control=0):
    labels = [
        f"{sim_names[run_u]} - {sim_names[control]}",
        f"{sim_names[run_T]} - {sim_names[control]}",
        "scaled residual",
    ]
    for a, label in zip(ax, labels):
        annotate_panel_label(a, label)

# Pre-compute cold-point tropopause profiles only for figures that draw them.
CPTs = None
if should_run_any([1, 7, 8, 9, 10, 11, 12, 13]):
    CPTs = []

    for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
        theta = hemispheric_fold(zds.theta.mean(dim="Time"), parity="even")
        pressure = hemispheric_fold(zds.pressure.mean(dim="Time"), parity="even")

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
#   2. Are the realised total wind and temperature fields close to
#      thermal-wind balance, and where do residuals remain?
#   3. Do OLR/precip/PW/RH/w/cloud responses scale mainly with realised
#      TTL temperature, or is there a residual response that points to a
#      wind/shear/wave pathway?
#
# Notes:
#   * The height-coordinate thermal-wind diagnostic is singular at the equator
#     because f -> 0.  Plots keep the near-equator region visible but mark
#     +/-5 deg; metrics still mask that band.
#   * This is most useful on the off-equatorial flanks of the tropical
#     anomaly, not exactly at the equator.
#   * The scaled-residual diagnostics assume approximate linearity.  They
#     are not a proof, but they are a good way to avoid accidentally calling
#     a stronger thermal response a direct wind response.

RD = 287.0
CP = 1004.0
P0_PA = 100000.0
G = 9.80665

def tropical_mean_surface(da, lat_bounds=(-10, 10)):
    """Area-weighted tropical mean for lat/lon fields."""
    da = da.sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
    w = np.cos(np.deg2rad(da.latitude))
    dims = [d for d in ("latitude", "longitude") if d in da.dims]
    return da.weighted(w).mean(dim=dims)

def subset_between(da, dim, bounds):
    """Subset a coordinate range without assuming ascending or descending order."""
    if bounds is None or dim not in da.dims:
        return da
    lo, hi = sorted(bounds)
    return da.where((da[dim] >= lo) & (da[dim] <= hi), drop=True)

def coordinate_layer_weights(coord, dim):
    """Positive finite-difference layer weights for a 1-D coordinate."""
    values = np.asarray(coord.values if isinstance(coord, xr.DataArray) else coord, dtype=float)
    if values.size == 1:
        weights = np.ones_like(values)
    else:
        bounds = np.empty(values.size + 1)
        bounds[1:-1] = 0.5 * (values[:-1] + values[1:])
        bounds[0] = values[0] - 0.5 * (values[1] - values[0])
        bounds[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
        weights = np.abs(np.diff(bounds))
    return xr.DataArray(weights, dims=(dim,), coords={dim: values}, name=f"d{dim}_weight")

def latheight_weights(da):
    """Cosine-latitude and layer-thickness weights for latitude-height fields."""
    weight = xr.ones_like(da)
    if "latitude" in da.dims:
        weight = weight * np.cos(np.deg2rad(da.latitude))
    if "z" in da.dims:
        weight = weight * coordinate_layer_weights(da.z, "z")
    return weight

def tropical_mean_latheight(da, z=None, z_bounds=None, lat_bounds=(-10, 10)):
    """Area and dz weighted tropical mean for model-height fields."""
    x = da
    if z is not None:
        x = x.interp(z=z, kwargs={"fill_value": np.nan})
    x = subset_between(x, "z", z_bounds)
    x = subset_between(x, "latitude", lat_bounds)
    dims = [d for d in ("Time", "z", "latitude") if d in x.dims]
    return x.weighted(latheight_weights(x)).mean(dim=dims)

def rms_weighted_latheight(da, lat_bounds=(-60, 60), z_bounds=(10000, 22000)):
    """Weighted RMS over a latitude-height box."""
    x = subset_between(da, "latitude", lat_bounds)
    x = subset_between(x, "z", z_bounds)
    dims = [d for d in ("z", "latitude") if d in x.dims]
    return np.sqrt((x**2).weighted(latheight_weights(x)).mean(dim=dims)).values.item()

def pattern_corr_latheight(a, b, lat_bounds=(-60, 60), z_bounds=(10000, 22000)):
    """Cosine-latitude and dz weighted pattern correlation over a latitude-height box."""
    aa, bb = xr.align(a, b, join="inner")
    aa = subset_between(aa, "latitude", lat_bounds)
    bb = subset_between(bb, "latitude", lat_bounds)
    aa = subset_between(aa, "z", z_bounds)
    bb = subset_between(bb, "z", z_bounds)
    aa, bb = xr.align(aa, bb, join="inner")

    weights = latheight_weights(aa)
    dims = [d for d in ("z", "latitude") if d in aa.dims]
    aa = aa - aa.weighted(weights).mean(dim=dims)
    bb = bb - bb.weighted(weights).mean(dim=dims)
    num = (aa * bb).weighted(weights).mean(dim=dims)
    den = np.sqrt((aa**2).weighted(weights).mean(dim=dims) * (bb**2).weighted(weights).mean(dim=dims))
    return (num / den).values.item()

def precip_rate_mean_map(ds):
    """Time-mean precipitation-rate map in mm/day."""
    pr = precip_rate_from_accum(total_precip_accum(ds), dt=dt)
    return hemispheric_fold(pr.mean(dim="Time"), parity="even")

def zmean_time_mean(da):
    """Time mean, zonal mean if longitude is present."""
    dims = [d for d in ("Time", "longitude") if d in da.dims]
    out = da.mean(dim=dims)
    return hemispheric_fold(out, parity="even")

# -----------------------
# Thermal wind helpers
# -----------------------

def canonical_latheight_var(var):
    aliases = {
        "T": "temperature",
        "t": "temperature",
        "temperature": "temperature",
        "u": "uReconstructZonal",
        "uReconstructZonal": "uReconstructZonal",
        "RH": "relhum",
        "rh": "relhum",
        "relhum": "relhum",
        "w": "w",
        "vertical_velocity": "w",
    }
    return aliases.get(var, var)

def required_latheight_vars(var):
    var = canonical_latheight_var(var)
    if var == "temperature":
        return ("theta", "pressure")
    if var == "w":
        return ("w",)
    return (var,)

def latheight_field_available(zds, var):
    return all(name in zds.data_vars for name in required_latheight_vars(var))

def latheight_var_hemispheric_parity(var):
    var = canonical_latheight_var(var)
    if var in ("temperature", "uReconstructZonal", "relhum", "w"):
        return "even"
    if var in ("uReconstructMeridional", "v", "umeridional"):
        return "odd"
    return "even"

def model_level_temperature(zds):
    theta = finite_or_nan(zds.theta)
    pressure = finite_or_nan(zds.pressure)
    return theta * (pressure / P0_PA) ** (RD / CP)

def latheight_time_mean_field(zds, var):
    """Time-mean zonal-mean model-height field as (z, latitude)."""
    var = canonical_latheight_var(var)
    if not latheight_field_available(zds, var):
        missing = [name for name in required_latheight_vars(var) if name not in zds.data_vars]
        raise ValueError(f"Missing variables for {var}: {missing}")

    if var == "temperature":
        da = model_level_temperature(zds)
    elif var == "w":
        da = vertical_velocity_from_zm(zds)
    else:
        da = finite_or_nan(zds[var])

    if "Time" in da.dims:
        da = da.mean(dim="Time", skipna=True)

    da = model_lat_field(da)
    da = hemispheric_fold(da, parity=latheight_var_hemispheric_parity(var)).load()
    da.name = var
    return da

_latheight_field_cache = {}

def get_latheight_field(run_index, var):
    key = (run_index, canonical_latheight_var(var))
    if key not in _latheight_field_cache:
        _latheight_field_cache[key] = latheight_time_mean_field(dss_zm[run_index], key[1])
    return _latheight_field_cache[key]

def thermal_wind_terms(T_da, u_da, tw_equator_mask_deg=5.0):
    """
    Diagnose height-coordinate thermal-wind balance.

    Inputs must have dimensions (z, latitude), with z in metres.

    Approximate relation used:
        d u / dz = -g/(f a T) * dT/dphi

    where phi is latitude in radians.  This relation is not valid at the
    equator. Set tw_equator_mask_deg to None to keep the near-equator values
    for plotting; otherwise that latitude band is masked.
    """
    earth_rotation_rate = 7.2921159e-5
    a = 6.371e6

    T, u = xr.align(T_da, u_da, join="inner")
    T = T.transpose("z", "latitude")
    u = u.transpose("z", "latitude")

    lat_rad = np.deg2rad(T.latitude.values)
    z = T.z.values
    f = 2.0 * earth_rotation_rate * np.sin(lat_rad)

    dT_dphi = np.gradient(T.values, lat_rad, axis=1)
    du_dz = np.gradient(u.values, z, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        du_dz_tw = -G / (f[None, :] * a * T.values) * dT_dphi

    du_dz = np.where(np.isfinite(du_dz), du_dz, np.nan)
    du_dz_tw = np.where(np.isfinite(du_dz_tw), du_dz_tw, np.nan)
    if tw_equator_mask_deg is not None and tw_equator_mask_deg > 0:
        mask = np.abs(T.latitude.values) < tw_equator_mask_deg
        du_dz[:, mask] = np.nan
        du_dz_tw[:, mask] = np.nan

    resid = du_dz - du_dz_tw
    coords = {"z": T.z, "latitude": T.latitude}

    return (
        xr.DataArray(du_dz, dims=("z", "latitude"), coords=coords, name="du_dz_observed", attrs={"units": "s-1"}),
        xr.DataArray(du_dz_tw, dims=("z", "latitude"), coords=coords, name="du_dz_thermal_wind", attrs={"units": "s-1"}),
        xr.DataArray(resid, dims=("z", "latitude"), coords=coords, name="du_dz_residual", attrs={"units": "s-1"}),
    )

def thermal_wind_temperature_gradient_terms(T_da, u_da, tw_equator_mask_deg=5.0):
    """
    Diagnose thermal-wind balance as a meridional temperature gradient.

    Inputs must have dimensions (z, latitude), with z in metres.  The
    inverted height-coordinate relation is:
        dT/dy = -(f T/g) * du/dz

    where y is northward distance in metres. Set tw_equator_mask_deg to None
    to keep the near-equator values for plotting; otherwise that latitude band
    is masked because the geostrophic thermal-wind approximation is not useful
    there.
    """
    earth_rotation_rate = 7.2921159e-5
    a = 6.371e6

    T, u = xr.align(T_da, u_da, join="inner")
    T = T.transpose("z", "latitude")
    u = u.transpose("z", "latitude")

    lat_rad = np.deg2rad(T.latitude.values)
    z = T.z.values
    f = 2.0 * earth_rotation_rate * np.sin(lat_rad)

    dT_dphi = np.gradient(T.values, lat_rad, axis=1)
    dT_dy = dT_dphi / a
    du_dz = np.gradient(u.values, z, axis=0)
    dT_dy_tw = -(f[None, :] * T.values / G) * du_dz

    dT_dy = np.where(np.isfinite(dT_dy), dT_dy, np.nan)
    dT_dy_tw = np.where(np.isfinite(dT_dy_tw), dT_dy_tw, np.nan)
    if tw_equator_mask_deg is not None and tw_equator_mask_deg > 0:
        mask = np.abs(T.latitude.values) < tw_equator_mask_deg
        dT_dy[:, mask] = np.nan
        dT_dy_tw[:, mask] = np.nan

    resid = dT_dy - dT_dy_tw
    coords = {"z": T.z, "latitude": T.latitude}

    return (
        xr.DataArray(dT_dy, dims=("z", "latitude"), coords=coords, name="dT_dy_observed", attrs={"units": "K m-1"}),
        xr.DataArray(dT_dy_tw, dims=("z", "latitude"), coords=coords, name="dT_dy_thermal_wind", attrs={"units": "K m-1"}),
        xr.DataArray(resid, dims=("z", "latitude"), coords=coords, name="dT_dy_residual", attrs={"units": "K m-1"}),
    )

def add_tw_equator_guides(ax, lat=5.0):
    for x in (-lat, lat):
        ax.axvline(x, color="0.2", lw=0.9, ls="--", alpha=0.9)

def get_latheight_anomaly(run_index, control_index, var):
    """Time-mean latitude-height anomaly relative to control, on this run's grid."""
    field = get_latheight_field(run_index, var)
    control = get_latheight_field(control_index, var).interp(
        z=field.z, latitude=field.latitude, kwargs={"fill_value": np.nan}
    )
    return field - control

def mean_TTL_latheight_anomaly(var, run_index, control_index=0, z_bounds=(14000, 18000), lat_bounds=(-10, 10)):
    """Mean realised TTL anomaly relative to control on model-height levels."""
    dX = get_latheight_anomaly(run_index, control_index, var)
    return tropical_mean_latheight(dX, z_bounds=z_bounds, lat_bounds=lat_bounds).values.item()

def scaled_residual_latheight(var, alpha, run_u=1, run_T=2, control=0):
    """
    Return dX_u, dX_T, and dX_u - alpha*dX_T for model-height fields.
    """
    dXu = get_latheight_anomaly(run_u, control, var)
    dXT = get_latheight_anomaly(run_T, control, var)
    dXT_on_u = dXT.interp(z=dXu.z, latitude=dXu.latitude, kwargs={"fill_value": np.nan})
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
        Xc = hemispheric_fold(dss[control][var].mean(dim="Time"), parity="even")
        Xu = hemispheric_fold(dss[run_u][var].mean(dim="Time"), parity="even")
        XT = hemispheric_fold(dss[run_T][var].mean(dim="Time"), parity="even")

    dXu = Xu - Xc.interp(latitude=Xu.latitude, longitude=Xu.longitude, kwargs={"fill_value": np.nan})
    dXT = XT - Xc.interp(latitude=XT.latitude, longitude=XT.longitude, kwargs={"fill_value": np.nan})
    dXT_on_u = dXT.interp(latitude=dXu.latitude, longitude=dXu.longitude, kwargs={"fill_value": np.nan})
    resid = dXu - alpha * dXT_on_u
    return dXu, dXT, resid

# -----------------------
# 1. Realised thermal/wind scaling numbers
# -----------------------

TTL_z_bounds = (14000, 18000)
TTL_ref_z = 16000
TTL_lat_bounds = (-10, 10)
TW_metric_z_bounds = (10000, 22000)

dT_TTL_u = mean_TTL_latheight_anomaly("temperature", 1, z_bounds=TTL_z_bounds, lat_bounds=TTL_lat_bounds)
dT_TTL_T = mean_TTL_latheight_anomaly("temperature", 2, z_bounds=TTL_z_bounds, lat_bounds=TTL_lat_bounds)
du_TTL_u = mean_TTL_latheight_anomaly("uReconstructZonal", 1, z_bounds=TTL_z_bounds, lat_bounds=TTL_lat_bounds)
du_TTL_T = mean_TTL_latheight_anomaly("uReconstructZonal", 2, z_bounds=TTL_z_bounds, lat_bounds=TTL_lat_bounds)

alpha_TTL = dT_TTL_u / dT_TTL_T if np.isfinite(dT_TTL_T) and abs(dT_TTL_T) > 1e-12 else np.nan

print("\nTemperature scaling based on realised TTL temperature anomaly")
print("--------------------------------------------------------")
print(f"TTL layer: {TTL_z_bounds[0]/1000:.1f}-{TTL_z_bounds[1]/1000:.1f} km, {TTL_lat_bounds[0]} to {TTL_lat_bounds[1]} deg")
print(f"u-nudged mean ΔT_TTL = {dT_TTL_u:.3f} K")
print(f"T-nudged mean ΔT_TTL = {dT_TTL_T:.3f} K")
print(f"alpha_TTL = ΔT_u / ΔT_T = {alpha_TTL:.3f}")
print(f"u-nudged mean Δu_TTL = {du_TTL_u:.3f} m s-1")
print(f"T-nudged mean Δu_TTL = {du_TTL_T:.3f} m s-1")
print("If |alpha_TTL| > 1, the u-nudged run has the stronger realised TTL temperature anomaly.")

# -----------------------
# 2. Thermal-wind balance of total fields
# -----------------------

if should_run_figure(1):
    fig, ax = plt.subplots(len(dss_zm), 3, figsize=(8, 2 * len(dss_zm)), constrained_layout=True, sharex=True, sharey=True)
    if len(dss_zm) == 1:
        ax = np.array([ax])

    im_obs = im_tw = im_resid = None
    thermal_wind_metric_rows = []

    levels = symmetric_levels(8.0, ndiv=16)

    for i, zds in enumerate(dss_zm):
        Tbar = get_latheight_field(i, "temperature")
        ubar = get_latheight_field(i, "uReconstructZonal")
        du_obs, du_tw, tw_resid = thermal_wind_terms(Tbar, ubar, tw_equator_mask_deg=None)
        du_obs_metric, du_tw_metric, tw_resid_metric = thermal_wind_terms(
            Tbar, ubar, tw_equator_mask_deg=5.0
        )

        rms_obs = rms_weighted_latheight(du_obs_metric * 1000.0, z_bounds=TW_metric_z_bounds)
        rms_tw = rms_weighted_latheight(du_tw_metric * 1000.0, z_bounds=TW_metric_z_bounds)
        rms_resid = rms_weighted_latheight(tw_resid_metric * 1000.0, z_bounds=TW_metric_z_bounds)
        corr = pattern_corr_latheight(du_obs_metric, du_tw_metric, z_bounds=TW_metric_z_bounds)
        thermal_wind_metric_rows.append([sim_names[i], rms_obs, rms_tw, rms_resid, corr])

        im_obs = ax[i, 0].contourf(du_obs.latitude, du_obs.z / 1000.0, du_obs * 1000.0, levels=levels, cmap="bwr", extend="both")
        im_tw = ax[i, 1].contourf(du_tw.latitude, du_tw.z / 1000.0, du_tw * 1000.0, levels=levels, cmap="bwr", extend="both")
        im_resid = ax[i, 2].contourf(tw_resid.latitude, tw_resid.z / 1000.0, tw_resid * 1000.0, levels=levels, cmap="bwr", extend="both")

        for j in range(3):
            ax[i, j].plot(zds.latitude.values, CPTs[i][0] / 1000.0, color="k", lw=1, ls=":")
            add_tw_equator_guides(ax[i, j], lat=5.0)
            annotate_panel_label(ax[i, j], sim_names[i])
            set_latheight_labels(ax[i, j], xlabel=(i == len(dss_zm) - 1))
            ax[i, j].set_xlim(-60, 60)
            ax[i, j].set_ylim(0, 25)
            if j != 0:
                ax[i, j].set_ylabel("")
            if i < len(dss_zm) - 1:
                ax[i, j].set_xticklabels([])

    ax[0, 0].set_title(r"Observed $\partial u/\partial z$")
    ax[0, 1].set_title("TW-implied shear")
    ax[0, 2].set_title("Observed - implied")
    add_symmetric_colorbar(fig, im_resid, ax[:, 2], levels, label=r"m s$^{-1}$ km$^{-1}$")

    figsave.save(fig, "thermal_wind_total_fields", number=1)
    print("Done with figure: 01_thermal_wind_total_fields.png")
    print("WHAT TO LOOK FOR: use the residual panels and total-field metrics together. Low residual RMS and high observed-vs-TW pattern correlation mean the realised shear is mostly thermal-wind balanced; coherent residuals outside the dashed +/-5 deg lines point to ageostrophic or transient structure not captured by the local T gradient.")

    thermal_wind_metrics = pd.DataFrame(
        thermal_wind_metric_rows,
        columns=["simulation", "rms_observed_shear_ms-1_km-1", "rms_TW_implied_shear_ms-1_km-1", "rms_residual_ms-1_km-1", "pattern_corr_obs_vs_TW"]
    )
    print(f"\nThermal-wind total-field metrics over {TW_metric_z_bounds[0]/1000:.1f}-{TW_metric_z_bounds[1]/1000:.1f} km and 60S-60N, excluding +/-5 deg:")
    print(thermal_wind_metrics.to_string(index=False))
    if save:
        thermal_wind_metrics.to_csv(save_loc + "thermal_wind_total_metrics.csv", index=False)

# -----------------------
# 2b. Thermal-wind balance inverted to meridional temperature gradients
# -----------------------

if should_run_figure(13):
    fig, ax = plt.subplots(len(dss_zm), 3, figsize=(8, 2 * len(dss_zm)), constrained_layout=True, sharex=True, sharey=True)
    if len(dss_zm) == 1:
        ax = np.array([ax])

    im_obs = im_tw = im_resid = None
    thermal_wind_temperature_gradient_metric_rows = []

    levels = symmetric_levels(8.0, ndiv=16)

    for i, zds in enumerate(dss_zm):
        Tbar = get_latheight_field(i, "temperature")
        ubar = get_latheight_field(i, "uReconstructZonal")
        dTdy_obs, dTdy_tw, dTdy_resid = thermal_wind_temperature_gradient_terms(
            Tbar, ubar, tw_equator_mask_deg=None
        )
        dTdy_obs_metric, dTdy_tw_metric, dTdy_resid_metric = thermal_wind_temperature_gradient_terms(
            Tbar, ubar, tw_equator_mask_deg=5.0
        )

        rms_obs = rms_weighted_latheight(dTdy_obs_metric * 1e6, z_bounds=TW_metric_z_bounds)
        rms_tw = rms_weighted_latheight(dTdy_tw_metric * 1e6, z_bounds=TW_metric_z_bounds)
        rms_resid = rms_weighted_latheight(dTdy_resid_metric * 1e6, z_bounds=TW_metric_z_bounds)
        corr = pattern_corr_latheight(dTdy_obs_metric, dTdy_tw_metric, z_bounds=TW_metric_z_bounds)
        thermal_wind_temperature_gradient_metric_rows.append([sim_names[i], rms_obs, rms_tw, rms_resid, corr])

        im_obs = ax[i, 0].contourf(dTdy_obs.latitude, dTdy_obs.z / 1000.0, dTdy_obs * 1e6, levels=levels, cmap="bwr", extend="both")
        im_tw = ax[i, 1].contourf(dTdy_tw.latitude, dTdy_tw.z / 1000.0, dTdy_tw * 1e6, levels=levels, cmap="bwr", extend="both")
        im_resid = ax[i, 2].contourf(dTdy_resid.latitude, dTdy_resid.z / 1000.0, dTdy_resid * 1e6, levels=levels, cmap="bwr", extend="both")

        for j in range(3):
            ax[i, j].plot(zds.latitude.values, CPTs[i][0] / 1000.0, color="k", lw=1, ls=":")
            add_tw_equator_guides(ax[i, j], lat=5.0)
            annotate_panel_label(ax[i, j], sim_names[i])
            set_latheight_labels(ax[i, j], xlabel=(i == len(dss_zm) - 1))
            ax[i, j].set_xlim(-60, 60)
            ax[i, j].set_ylim(0, 25)
            if j != 0:
                ax[i, j].set_ylabel("")
            if i < len(dss_zm) - 1:
                ax[i, j].set_xticklabels([])

    ax[0, 0].set_title(r"Observed $\partial T/\partial y$")
    ax[0, 1].set_title("TW-implied T gradient")
    ax[0, 2].set_title("Observed - implied")
    add_symmetric_colorbar(fig, im_resid, ax[:, 2], levels, label=r"K per 1000 km")

    figsave.save(fig, "thermal_wind_temperature_gradient_total_fields", number=13)
    print("Done with figure: 13_thermal_wind_temperature_gradient_total_fields.png")
    print("WHAT TO LOOK FOR: this is the same thermal-wind balance viewed the other way round. Low residual RMS means the observed meridional temperature gradient is consistent with the observed vertical shear; coherent residuals outside the dashed +/-5 deg lines indicate thermal gradients not captured by balanced shear.")

    thermal_wind_temperature_gradient_metrics = pd.DataFrame(
        thermal_wind_temperature_gradient_metric_rows,
        columns=["simulation", "rms_observed_dTdy_K_per_1000km", "rms_TW_implied_dTdy_K_per_1000km", "rms_residual_K_per_1000km", "pattern_corr_obs_vs_TW"]
    )
    print(f"\nThermal-wind temperature-gradient metrics over {TW_metric_z_bounds[0]/1000:.1f}-{TW_metric_z_bounds[1]/1000:.1f} km and 60S-60N, excluding +/-5 deg:")
    print(thermal_wind_temperature_gradient_metrics.to_string(index=False))
    if save:
        thermal_wind_temperature_gradient_metrics.to_csv(
            save_loc + "thermal_wind_temperature_gradient_total_metrics.csv", index=False
        )

# -----------------------
# 3. Realised-state summary bar chart
# -----------------------

if should_run_any([2, 3]):
    summary_rows = []
    for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
        Tfield = get_latheight_field(i, "temperature")
        ufield = get_latheight_field(i, "uReconstructZonal")
        T16 = tropical_mean_latheight(Tfield, z=TTL_ref_z, lat_bounds=TTL_lat_bounds).values.item()
        u16 = tropical_mean_latheight(ufield, z=TTL_ref_z, lat_bounds=TTL_lat_bounds).values.item()
        Tttl = tropical_mean_latheight(Tfield, z_bounds=TTL_z_bounds, lat_bounds=TTL_lat_bounds).values.item()
        uttl = tropical_mean_latheight(ufield, z_bounds=TTL_z_bounds, lat_bounds=TTL_lat_bounds).values.item()

        N2 = N2_latheight(zds)
        N2_1418 = tropical_mean_latheight(N2 * 1e4, z_bounds=TTL_z_bounds, lat_bounds=TTL_lat_bounds).values.item()

        olr = tropical_mean_surface(ds.olrtoa, lat_bounds=(-10, 10)).mean(dim="Time").values.item()
        pw = tropical_mean_surface(ds.precipw, lat_bounds=(-10, 10)).mean(dim="Time").values.item()
        pr = tropical_mean_surface(precip_rate_from_accum(total_precip_accum(ds), dt=dt), lat_bounds=(-10, 10)).mean(dim="Time").values.item()

        summary_rows.append([sim_names[i], T16, u16, Tttl, uttl, N2_1418, olr, pw, pr])

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["simulation", "T16km_K", "u16km_ms-1", "T_TTL_K", "u_TTL_ms-1", "N2_14_18km_1e-4s-2", "OLR_Wm-2", "PW_kgm-2", "precip_mmday-1"]
    )

    # Add anomalies relative to control for quick inspection
    for col in summary_df.columns[1:]:
        summary_df["d" + col] = summary_df[col] - summary_df.loc[0, col]

    print("\nRealised-state summary:")
    print(summary_df.to_string(index=False))
    if save:
        summary_df.to_csv(save_loc + "realised_state_summary.csv", index=False)

    if should_run_figure(2):
        plot_cols = [
            ("dT16km_K", r"$\Delta T_{16km}$ (K)"),
            ("du16km_ms-1", r"$\Delta u_{16km}$ (m s$^{-1}$)"),
            ("dT_TTL_K", r"$\Delta T_{14-18km}$ (K)"),
            ("du_TTL_ms-1", r"$\Delta u_{14-18km}$ (m s$^{-1}$)"),
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

        figsave.save(fig, "realised_state_summary_bars", number=2)
        print("Done with figure: 02_realised_state_summary_bars.png")
        print("WHAT TO LOOK FOR: compare dT_TTL and dN2 with OLR/PW/precip. If the u-nudged run is colder and has the larger cloud/radiative response, analyse it as a stronger realised thermal perturbation first.")

    if should_run_figure(3):
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
            ax[j].set_xlabel(r"$\Delta T_{14-18km}$ (K)")
            ax[j].set_ylabel(ylabel)
            ax[j].grid(alpha=0.2)

        figsave.save(fig, "responses_vs_realised_TTL_temperature", number=3)
        print("Done with figure: 03_responses_vs_realised_TTL_temperature.png")
        print("WHAT TO LOOK FOR: if the two nudged runs line up by ΔT_TTL, the response is probably thermal/static-stability controlled rather than set by the nudged variable label.")

# -----------------------
# 5. Scaled-residual maps for surface/column fields
# -----------------------

surface_residual_specs = [
    (4, "olrtoa", r"OLR anomaly (W m$^{-2}$)", 20.0, "04_scaled_residual_olrtoa.png"),
    (5, "precipw", r"PW anomaly (kg m$^{-2}$)", 4.0, "05_scaled_residual_precipw.png"),
    (6, "precip_rate", r"Precip anomaly (mm day$^{-1}$)", 2.0, "06_scaled_residual_precip.png"),
]

for fig_num, var, label, max_abs, fname in surface_residual_specs:
    if not should_run_figure(fig_num):
        continue

    dXu, dXT, resid = scaled_residual_map(var, alpha_TTL)

    fig, ax = plt.subplots(
        1, 4, figsize=(13.5, 3.3), constrained_layout=True, sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 0.8]}
    )
    levels = symmetric_levels(max_abs, ndiv=13)
    im0 = ax[0].contourf(dXu.longitude, dXu.latitude, dXu, levels=levels, cmap="bwr", extend="both")
    im1 = ax[1].contourf(dXT.longitude, dXT.latitude, dXT, levels=levels, cmap="bwr", extend="both")
    im2 = ax[2].contourf(resid.longitude, resid.latitude, resid, levels=levels, cmap="bwr", extend="both")

    dXu_zm = dXu.mean(dim="longitude", skipna=True)
    dXT_zm = dXT.mean(dim="longitude", skipna=True)
    resid_zm = resid.mean(dim="longitude", skipna=True)
    ax[3].plot(dXu_zm, dXu_zm.latitude, color=cols[1], lw=1.5, label=f"{sim_names[1]} - {sim_names[0]}")
    ax[3].plot(dXT_zm, dXT_zm.latitude, color=cols[2], lw=1.5, label=f"{sim_names[2]} - {sim_names[0]}")
    ax[3].plot(resid_zm, resid_zm.latitude, color=cols[0], lw=1.5, label="scaled residual")
    ax[3].axvline(0, color="0.7", lw=0.7)

    ax[0].set_title(r"$\Delta X_u$")
    ax[1].set_title(r"$\Delta X_T$")
    ax[2].set_title(r"$\Delta X_u - \alpha \Delta X_T$")
    ax[3].set_title("Zonal mean")
    annotate_scaled_residual_labels(ax[:3])
    annotate_panel_label(ax[3], "zonal mean")
    for a in ax[:3]:
        a.set_xlabel("Longitude")
    ax[3].set_xlabel(label)
    ax[3].grid(alpha=0.2)
    ax[3].legend(loc="best", fontsize=7)
    ax[0].set_ylabel("Latitude")
    fig.suptitle(f"Scaled residual for {var}; alpha={alpha_TTL:.2f}")
    add_symmetric_colorbar(fig, im2, ax[:3], levels, label=label)

    stem = fname[3:-4] if fname.endswith(".png") and fname[:2].isdigit() else fname.replace(".png", "")
    outname = figsave.save(fig, stem, number=fig_num)
    print(f"Done with figure: {outname}")

if should_run_any([4, 5, 6]):
    print("WHAT TO LOOK FOR: small residual maps mean the u-nudged response is largely the T-nudged response scaled by realised TTL cooling. Coherent residuals imply an additional wind/shear/dynamical pathway.")

# -----------------------
# 6. Scaled-residual latitude-height sections
# -----------------------

latheight_residual_specs = [
    (7, "temperature", r"T anomaly (K)", 8.0, "07_scaled_residual_T_latheight.png"),
    (8, "uReconstructZonal", r"u anomaly (m s$^{-1}$)", 8.0, "08_scaled_residual_u_latheight.png"),
    (9, "relhum", r"RH anomaly (%)", 10.0, "09_scaled_residual_RH_latheight.png"),
    (10, "w", r"w anomaly (m s$^{-1}$)", 0.03, "10_scaled_residual_w_latheight.png"),
]

for fig_num, var, label, max_abs, fname in latheight_residual_specs:
    if not should_run_figure(fig_num):
        continue

    if not all(latheight_field_available(zds, var) for zds in dss_zm[:3]):
        print(f"Skipping {var}: required height-level variables are not present in all zm datasets")
        continue

    dXu, dXT, resid = scaled_residual_latheight(var, alpha_TTL)
    levels = symmetric_levels(max_abs, ndiv=13)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True, sharex=True, sharey=True)
    im0 = ax[0].contourf(dXu.latitude, dXu.z / 1000.0, dXu, levels=levels, cmap="bwr", extend="both")
    im1 = ax[1].contourf(dXT.latitude, dXT.z / 1000.0, dXT, levels=levels, cmap="bwr", extend="both")
    im2 = ax[2].contourf(resid.latitude, resid.z / 1000.0, resid, levels=levels, cmap="bwr", extend="both")

    for j, a in enumerate(ax):
        set_latheight_labels(a)
        a.set_xlim(-60, 60)
        a.set_ylim(0, 25)
        if j != 0:
            a.set_ylabel("")
        if j == 0:
            a.plot(dss_zm[1].latitude.values, CPTs[1][0] / 1000.0, color="k", lw=1, ls=":")
        elif j == 1:
            a.plot(dss_zm[2].latitude.values, CPTs[2][0] / 1000.0, color="k", lw=1, ls=":")
        else:
            a.plot(dss_zm[1].latitude.values, CPTs[1][0] / 1000.0, color="k", lw=1, ls=":")

    ax[0].set_title(r"$\Delta X_u$")
    ax[1].set_title(r"$\Delta X_T$")
    ax[2].set_title(r"$\Delta X_u - \alpha \Delta X_T$")
    annotate_scaled_residual_labels(ax)
    fig.suptitle(f"Scaled residual for {var}; alpha={alpha_TTL:.2f}")
    add_symmetric_colorbar(fig, im2, ax, levels, label=label)

    stem = fname[3:-4] if fname.endswith(".png") and fname[:2].isdigit() else fname.replace(".png", "")
    outname = figsave.save(fig, stem, number=fig_num)
    print(f"Done with figure: {outname}")

if should_run_any([7, 8, 9, 10]):
    print("WHAT TO LOOK FOR: w/RH residuals are particularly useful. A small height-level T residual but large w residual suggests comparable thermal forcing but different circulation adjustment.")

# -----------------------
# 7. N2 scaled residual in latitude-height coordinates
# -----------------------

if should_run_figure(11):
    N2_control = N2_latheight(dss_zm[0])
    N2_u = N2_latheight(dss_zm[1])
    N2_T = N2_latheight(dss_zm[2])

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

    ax[0].set_title(r"$\Delta N^2$")
    ax[1].set_title(r"$\Delta N^2$")
    ax[2].set_title(r"$\Delta N^2_u - \alpha \Delta N^2_T$")
    annotate_scaled_residual_labels(ax)
    for a in ax:
        set_latheight_labels(a)
        a.set_xlim(-60, 60)
        a.set_ylim(0, 25)
    fig.suptitle(f"Scaled thermal residual for static stability; alpha={alpha_TTL:.2f}")
    add_symmetric_colorbar(fig, im2, ax, levels, label=r"$\Delta N^2$ ($10^{-4}$ s$^{-2}$)")

    figsave.save(fig, "scaled_residual_N2_latheight", number=11)
    print("Done with figure: 11_scaled_residual_N2_latheight")
    print("WHAT TO LOOK FOR: if N2 residual is small, the u-nudged stability response is mostly the stronger realised cooling. If not, wind-induced circulation/shear may be modifying stability independently.")

# -----------------------
# 8. Cloud-fraction scaled residual, using same cldfrac logic as above
# -----------------------

if should_run_figure(12):
    def get_zm_cldfrac_for_run(i):
        """Return time-mean zonal-mean cldfrac as (z, latitude), following the existing script's logic."""
        ds = dss[i]
        zds = dss_zm[i]
        if "cldfrac" in zds.data_vars:
            cld = zds.cldfrac.mean(dim="Time")
            cld = hemispheric_fold(cld, parity="even")
            zmid = height_from_zm_latlon(zds)
            cld, _, _ = lat_height_mesh(cld, zmid, zds.latitude.values)
            return cld
        if "cldfrac" in ds.data_vars:
            cld = ds.cldfrac.mean(dim=("Time", "longitude"))
            cld = model_lat_field(cld)
            cld = hemispheric_fold(cld, parity="even")
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

        ax[0].set_title(r"$\Delta$cldfrac")
        ax[1].set_title(r"$\Delta$cldfrac")
        ax[2].set_title(r"$\Delta C_u - \alpha \Delta C_T$")
        annotate_scaled_residual_labels(ax)
        for a in ax:
            set_latheight_labels(a)
            a.set_xlim(-60, 60)
            a.set_ylim(0, 25)
        fig.suptitle(f"Scaled thermal residual for cloud fraction; alpha={alpha_TTL:.2f}")
        add_symmetric_colorbar(fig, im2, ax, levels, label="Cloud-fraction anomaly")

        figsave.save(fig, "scaled_residual_cldfrac_latheight", number=12)
        print("Done with figure: 12_scaled_residual_cldfrac_latheight")
        print("WHAT TO LOOK FOR: a small residual suggests cloud-depth changes mainly scale with realised TTL cooling. A large upper-cloud residual in u-nudged suggests an additional dynamical/shear/wave pathway.")
    else:
        print("Skipping cloud-fraction scaled residual: cldfrac not present in all runs")

# -----------------------
# 9. Compact residual summary
# -----------------------

if run_all_figures():
    residual_rows = []

    # Surface metrics: tropical mean residuals
    for var, pretty in [("olrtoa", "OLR"), ("precipw", "PW"), ("precip_rate", "precip")]:
        dXu, dXT, resid = scaled_residual_map(var, alpha_TTL)
        trop_resid = tropical_mean_surface(resid, lat_bounds=(-10, 10)).values.item()
        trop_du = tropical_mean_surface(dXu, lat_bounds=(-10, 10)).values.item()
        trop_dT = tropical_mean_surface(dXT, lat_bounds=(-10, 10)).values.item()
        residual_rows.append([pretty, "surface_tropical_mean", trop_du, trop_dT, trop_resid])

    # Height-level metrics: TTL tropical mean residuals
    for var, pretty in [("temperature", "T"), ("uReconstructZonal", "u"), ("relhum", "RH"), ("w", "w")]:
        if all(latheight_field_available(zds, var) for zds in dss_zm[:3]):
            dXu, dXT, resid = scaled_residual_latheight(var, alpha_TTL)
            values = []
            for field in (dXu, dXT, resid):
                values.append(
                    tropical_mean_latheight(
                        field, z_bounds=TTL_z_bounds, lat_bounds=TTL_lat_bounds
                    ).values.item()
                )
            residual_rows.append([pretty, "TTL_14_18km_tropical_mean", values[0], values[1], values[2]])

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
    print("  * If scaled_residual remains large for u/w/RH but not T/N2: likely extra dynamical/shear/secondary-circulation effect.")
    print("  * If scaled_residual remains large for cloud/OLR/precip: the u-nudged run is doing something beyond simple thermal scaling, worth examining with wave/eddy diagnostics.")
