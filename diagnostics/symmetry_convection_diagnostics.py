#!/usr/bin/env python
# coding: utf-8

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
import dask
from wavenumber_frequency_functions import *
from matplotlib import colors
from tropopause import *
from diagnostic_helpers import *

save = True
save_loc = '/home/cwp29/tmp/mpas/crf_on/convection/'

# Set to [] or None to run all figures.  Otherwise list the numbered figures
# to regenerate, e.g. [18] for the conditional cloud diagnostics.
#
# Figure numbers:
#   1:  diag_symmetry_breaking
#   2:  diag_zonal_wavenumber_variance
#   3:  diag_hovmoller_anomalies
#   4:  diag_N2_latheight
#   5:  diag_zonal_wind_shear_latheight
#   6:  diag_tropopause_profiles
#   7:  diag_high_cloud_maps
#   8:  diag_cloud_top_height_hist
#   9:  diag_convective_distributions
#   10: diag_olr_precip_coupling
#   11: diag_radiative_heating_precip
#   12: diag_tropical_lag_correlations
#   13: comp_zonalmean_convective_state
#   14: comp_hydrometeor_latheight
#   15: comp_cloudtop_highcloud_latitude
#   16: comp_precip_conditioned_on_pw_cape_cin
#   17: comp_olr_cloudtop_conditioned_on_precip
#   18: comp_cloud_conditioned_on_precip_pw_cape_cin
#   19: comp_atmos_radiative_heating_proxy
#   20: comp_tropical_timeseries_extended
figures_to_run = [8, 18]
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)


def save_figure(number, fig, stem, dpi=300):
    return figsave.save(fig, stem, number=number, dpi=dpi)


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


def run_all_figures():
    return all_figures_requested(figures_to_run)


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

dss = []
dss_zm = []

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
    ds = standardize_latlon_dataset(ds, needed_vars)

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
    zds = standardize_latlon_dataset(zds, needed_vars)

    # cheaper if data are exactly 6-hourly and aligned
    #ds = ds.coarsen(Time=4, boundary="trim").mean()
    #zds = zds.coarsen(Time=4, boundary="trim").mean()

    print(ds)
    print(zds)
    dss.append(ds)
    dss_zm.append(zds)

# Pre-compute CPTs only for figures that draw them.
CPTs = None
if should_run_any([4, 5, 6, 14]):
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

# -----------------------------------------------------------------------------
# Additional helpers preserved from zm_latlon_misc_augmented.py
# -----------------------------------------------------------------------------
# These support the non-duplicated mechanism diagnostics appended below.

g = 9.80665

def conv_precip_accum(ds):
    """Accumulated convective precipitation in mm, including bucket counters."""
    return ds["rainc"] + ds["i_rainc"] * 100.0

def gridscale_precip_accum(ds):
    """Accumulated grid-scale precipitation in mm, including bucket counters."""
    return ds["rainnc"] + ds["i_rainnc"] * 100.0

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

def plot_binned_conditional_curve(
    ax, x, y, bins, color, label=None, min_count=20, latmax=20,
    max_time=48, max_lat=80, max_lon=160, show_spread=True
):
    """Plot a sampled, binned y(x) curve from matching lat-lon-time fields."""
    x, y = xr.align(finite_or_nan(x), finite_or_nan(y), join="inner")
    xv, yv = strided_matching_samples(
        x, y, latmax=latmax, max_time=max_time, max_lat=max_lat, max_lon=max_lon
    )
    if xv.size == 0:
        return False

    xc, ym, y25, y75, count = binned_statistic_1d(xv, yv, bins, min_count=min_count)
    if not np.any(np.isfinite(ym)):
        return False

    ax.plot(xc, ym, color=color, label=label)
    if show_spread:
        ax.fill_between(xc, y25, y75, color=color, alpha=0.15, linewidth=0)
    return True

def hemi_components_latlon(da, lat_dim="latitude"):
    """Return hemispherically symmetric and antisymmetric components."""
    flipped = da.isel({lat_dim: slice(None, None, -1)})
    flipped = flipped.assign_coords({lat_dim: da[lat_dim].values})
    sym = 0.5 * (da + flipped)
    asym = 0.5 * (da - flipped)
    return sym, asym


def hemispheric_symmetrise_lat(da, lat_dim="latitude"):
    """Return the hemispherically symmetric part on the original latitude grid."""
    flipped = da.isel({lat_dim: slice(None, None, -1)})
    flipped = flipped.assign_coords({lat_dim: da[lat_dim].values})
    return 0.5 * (da + flipped)

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

# In[ ]:

# 1. Symmetry-breaking diagnostics.
# What to look for:
#   * If tropical k>0 variance for precipitation/OLR is very small, do not use
#     MJO-style phase composites.
#   * If nudging increases k>0 or antisymmetric variance, it is changing wave or
#     instability behaviour rather than just the zonal-mean convective state.


if should_run_figure(1):
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

        save_figure(1, fig, "symmetry_breaking")
        print("Done with figure: diag_symmetry_breaking")
else:
    symmetry_df = pd.DataFrame()

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

if should_run_figure(2):
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

    save_figure(2, fig, "zonal_wavenumber_variance")
    print("Done with figure: diag_zonal_wavenumber_variance")

# In[ ]:

# 3. Equatorial Hovmoller anomalies for precipitation, OLR and precipitable water.
# These are better than raw fields for checking whether there is any coherent
# propagating cloud/rain envelope.

if should_run_figure(3):
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
            annotate_panel_label(ax[j, i], sim_names[i])
            if i == 0:
                ax[j, i].set_ylabel(f"{label}\ntime")
            else:
                ax[j, i].set_yticklabels([])
            if j == len(hov_vars)-1:
                ax[j, i].set_xlabel("Longitude")
        add_symmetric_colorbar(fig, im, ax[j, :], levels_hov, label=f"{label}' ({units})")

    save_figure(3, fig, "hovmoller_anomalies")
    print("Done with figure: diag_hovmoller_anomalies")

# In[ ]:

# 4. Latitude-height static stability N^2.
# What to look for:
#   * Does T nudging reduce/increase N2 near the tropical tropopause?
#   * Does U nudging induce a secondary N2 response despite only nudging wind?

if should_run_figure(4):
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
        annotate_panel_label(ax[i], sim_names[i])
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
        annotate_panel_label(ax2[i], control_diff_label(sim_names, i))
        set_latheight_labels(ax2[i], xlabel=(i == len(dss)-1))
        if i < len(dss)-1:
            ax2[i].set_xticklabels([])

    fig.colorbar(im, ax=ax, location="right", label=r"$N^2$ ($10^{-4}$ s$^{-2}$)")
    fig2.colorbar(im2, ax=ax2[0], location="right", label=r"$N^2$ ($10^{-4}$ s$^{-2}$)")
    if im3 is not None:
        add_symmetric_colorbar(fig2, im3, ax2[1:], symmetric_levels(1.0, ndiv=12), label=r"$\Delta N^2$ ($10^{-4}$ s$^{-2}$)")

    save_figure(4, fig, "N2_latheight")
    save_figure(4, fig2, "N2_latheight_diff")
    print("Done with figure: diag_N2_latheight")

# In[ ]:

# 4b. Latitude-height zonal wind shear.
# What to look for:
#   * Does U nudging directly sharpen or weaken tropical shear layers?
#   * Does T nudging induce a secondary shear response consistent with
#     thermal-wind adjustment?

def zonal_wind_shear_latheight(zds):
    u = model_lat_field(zds.uReconstructZonal.mean(dim="Time"))
    z = height_from_zm_latlon(zds).values
    du_dz = np.gradient(u.values, z, axis=0)
    return xr.DataArray(
        du_dz,
        dims=u.dims,
        coords=u.coords,
        name="du_dz",
        attrs={"units": "s-1"},
    )

if should_run_figure(5):
    if all("uReconstructZonal" in zds.data_vars for zds in dss_zm):
        fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.3*len(dss)), constrained_layout=True, sharex=True)
        fig2, ax2 = plt.subplots(len(dss), 1, figsize=(8, 3.3*len(dss)), constrained_layout=True, sharex=True)
        if len(dss) == 1:
            ax = [ax]
            ax2 = [ax2]

        control_shear = None
        im = im2 = im3 = None
        shear_levels = symmetric_levels(8.0, ndiv=16)
        shear_diff_levels = symmetric_levels(4.0, ndiv=16)

        for i, zds in enumerate(dss_zm):
            shear = zonal_wind_shear_latheight(zds)
            zmid = height_from_zm_latlon(zds)
            field, lat2d, z2d = lat_height_mesh(shear * 1000.0, zmid, zds.latitude.values)

            im = ax[i].contourf(lat2d, z2d/1000., field.values, levels=shear_levels, cmap="bwr", extend="both")
            ax[i].plot(zds.latitude.values, CPTs[i][0]/1000., color="k", lw=1, ls=":")
            annotate_panel_label(ax[i], sim_names[i])
            set_latheight_labels(ax[i], xlabel=(i == len(dss)-1))
            if i < len(dss)-1:
                ax[i].set_xticklabels([])

            if i == 0:
                control_shear = shear
                im2 = ax2[i].contourf(lat2d, z2d/1000., field.values, levels=shear_levels, cmap="bwr", extend="both")
            else:
                cshear = control_shear.interp(z=shear.z, latitude=shear.latitude)
                diff = (shear - cshear) * 1000.0
                im3 = ax2[i].contourf(lat2d, z2d/1000., diff.values, levels=shear_diff_levels, cmap="bwr", extend="both")
            ax2[i].plot(zds.latitude.values, CPTs[i][0]/1000., color="k", lw=1, ls=":")
            annotate_panel_label(ax2[i], control_diff_label(sim_names, i))
            set_latheight_labels(ax2[i], xlabel=(i == len(dss)-1))
            if i < len(dss)-1:
                ax2[i].set_xticklabels([])

        add_symmetric_colorbar(fig, im, ax, shear_levels, label=r"$\partial u / \partial z$ (m s$^{-1}$ km$^{-1}$)")
        add_symmetric_colorbar(fig2, im2, ax2[0], shear_levels, label=r"$\partial u / \partial z$ (m s$^{-1}$ km$^{-1}$)")
        if im3 is not None:
            add_symmetric_colorbar(fig2, im3, ax2[1:], shear_diff_levels, label=r"$\Delta \partial u / \partial z$ (m s$^{-1}$ km$^{-1}$)")

        save_figure(5, fig, "zonal_wind_shear_latheight")
        save_figure(5, fig2, "zonal_wind_shear_latheight_diff")
        print("Done with figure: diag_zonal_wind_shear_latheight")
    else:
        print("Skipping zonal wind shear latitude-height plots: uReconstructZonal not present in all zm datasets")

# In[ ]:

# 5. Tropopause height and pressure profiles, plus nudged-control differences.
# What to look for:
#   * Tropical tropopause height/pressure shifts are a compact measure of the
#     UTLS thermal response.
#   * A T-nudged shift without a similar U-nudged shift points to a direct
#     thermal pathway.

if should_run_figure(6):
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

    save_figure(6, fig, "tropopause_profiles")
    print("Done with figure: diag_tropopause_profiles")

# In[ ]:

# 6. High-cloud maps and cloud-top-height distributions.
# What to look for:
#   * More high cloud / higher cloud tops at fixed or similar rainfall suggests
#     a cloud-depth/radiative pathway rather than simply more precipitation.
#   * If this block skips, it means full lat/lon cldfrac was not loaded.

if should_run_figure(7):
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
            annotate_panel_label(ax[i], sim_names[i])
            ax[i].set_xlabel("Longitude")
            if i == 0:
                ax[i].set_ylabel("Latitude")
                control_hc = hc
                im2 = ax2[i].contourf(hc.longitude, hc.latitude, hc.values, levels=np.linspace(0, 1, 21), cmap="binary", extend="max")
            else:
                ax[i].set_yticklabels([])
                c = control_hc.interp(latitude=hc.latitude, longitude=hc.longitude)
                im3 = ax2[i].contourf(hc.longitude, hc.latitude, (hc - c).values, levels=symmetric_levels(0.15, ndiv=12), cmap="bwr", extend="both")
            annotate_panel_label(ax2[i], control_diff_label(sim_names, i))
            ax2[i].set_xlabel("Longitude")
            if i == 0:
                ax2[i].set_ylabel("Latitude")
            else:
                ax2[i].set_yticklabels([])

        fig.colorbar(im, ax=ax[-1], location="right", label="Cloud fraction above 12 km")
        fig2.colorbar(im2, ax=ax2[0], location="right", label="Cloud fraction above 12 km")
        if im3 is not None:
            add_symmetric_colorbar(fig2, im3, ax2[1:], symmetric_levels(0.15, ndiv=12), label="High-cloud difference")
        save_figure(7, fig, "high_cloud_maps")
        save_figure(7, fig2, "high_cloud_maps_diff")
        print("Done with figure: diag_high_cloud_maps")

if should_run_figure(8):
    cth_available = False
    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    for i, ds in enumerate(dss):
        cth = cloud_top_height(ds, thresh=0.05)
        if cth is None:
            continue
        cth_available = True
        vals = tropics(cth, latmax=10).values.ravel() / 1000.0
        vals = vals[np.isfinite(vals)]
        h, b = np.histogram(vals, bins=np.linspace(0, 22, 45), density=True)
        h = uniform_filter1d(h, size=3)
        ax.plot(0.5*(b[1:]+b[:-1]), h, color=cols[i], label=sim_names[i])
    if cth_available:
        ax.set_xlabel("Tropical cloud-top height thresholded at cldfrac > 0.05 (km)")
        ax.set_ylabel("Probability density")
        ax.legend(loc="best")
        save_figure(8, fig, "cloud_top_height_hist")
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

if should_run_figure(9):
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

    save_figure(9, fig, "convective_distributions")
    print("Done with figure: diag_convective_distributions")

# In[ ]:

# 9. OLR-precipitation coupling / greenhouse-enhancement-like proxy.
# This is not a formal clear-sky CRF decomposition, but the slope tells you how
# much OLR decreases per unit rainfall anomaly.
# What to look for:
#   * More negative slope in T-nudged run: deeper/more radiatively active clouds
#     per unit precipitation.
#   * Similar slope but shifted precipitation distribution: mostly amplitude/state change.

if should_run_figure(10):
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
    save_figure(10, fig, "olr_precip_coupling")
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

if should_run_figure(11):
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
            ax[0].plot(get_time_values(tmean, dt=dt), tmean.values, color=cols[i], label=sim_names[i])

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
        save_figure(11, fig, "radiative_heating_precip")
        print("Done with figure: diag_radiative_heating_precip")
    else:
        print("Skipping radiative-heating diagnostics: radiation flux variables not loaded")

# In[ ]:

# 11. Moisture-convection lag correlations using tropical means.
# What to look for:
#   * PW leading precipitation is a recharge-discharge signature.
#   * If nudging changes the lag of maximum correlation, it changes the timing of
#     moistening relative to convection even without an MJO-like envelope.

if should_run_figure(12):
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

    save_figure(12, fig, "tropical_lag_correlations")
    print("Done with figure: diag_tropical_lag_correlations")


# ============================================================
# Additional diagnostics preserved from zm_latlon_misc_augmented.py
# ============================================================
# The older misc script overlapped with the diagnostics above in its symmetry,
# N2, Hovmoller, and OLR-precipitation analyses.  Those duplicated analyses are
# intentionally not repeated here; the blocks below preserve the distinct
# convective-state, hydrometeor, cloud-top, radiation-profile, and time-series
# diagnostics.

# ============================================================
# 3. Zonal-mean convective state and differences from control
# ============================================================
# What to look for:
#   - Broad OLR/high-cloud changes without a comparable precipitation change
#     suggest altered cloud depth/radiative effects rather than simply stronger rain.
#   - PW/CAPE/CIN shifts indicate that the lower-tropospheric convective regime
#     has changed, not just the UTLS.

if should_run_figure(13):
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
                x = precip_rate_from_accum(total_precip_accum(ds), dt=dt)
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
        save_figure(13, fig, "zonalmean_convective_state")
        save_figure(13, fig2, "zonalmean_convective_state_diff")
    print("Done with figure: comp_zonalmean_convective_state")

# ============================================================
# 5. Hydrometeor latitude-height sections
# ============================================================
# What to look for:
#   - More upper-tropospheric ice/snow/graupel in T_nudge without much more rain
#     points to deeper/anvil cloud changes rather than just greater rain amount.

if should_run_figure(14):
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
                annotate_panel_label(ax[i], sim_names[i])
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
                annotate_panel_label(ax2[i], control_diff_label(sim_names, i))
                set_latheight_labels(ax2[i], xlabel=(i == len(dss) - 1))
                if i < len(dss) - 1:
                    ax2[i].set_xticklabels([])

            fig.colorbar(im, ax=ax, location="right", label=cbar_label)
            fig2.colorbar(im2, ax=ax2[0], location="right", label=cbar_label)
            if im3 is not None:
                add_symmetric_colorbar(fig2, im3, ax2[1:], im3.levels, label=r"$\Delta$ " + cbar_label)

            if save:
                save_figure(14, fig, f"{outname}_latheight")
                save_figure(14, fig2, f"{outname}_latheight_diff")
            print(f"Done with figure: comp_{outname}_latheight")
        except ValueError as e:
            print(f"Skipping {outname}: {e}")

# ============================================================
# 6. Cloud-top proxy and high-cloud fraction from zonal-mean cloud fraction
# ============================================================
# What to look for:
#   - Higher tropical cloud tops / more 10-18 km cloud in T_nudge would support
#     a UTLS-temperature/cloud-depth interpretation.

if should_run_figure(15):
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
            save_figure(15, fig, "cloudtop_highcloud_latitude")
            save_figure(15, fig2, "cloudtop_highcloud_latitude_diff")
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

if should_run_figure(16):
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
            pr = precip_rate_from_accum(total_precip_accum(ds), dt=dt)
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
        save_figure(16, fig, "precip_conditioned_on_pw_cape_cin")
    print("Done with figure: comp_precip_conditioned_on_pw_cape_cin")

if should_run_figure(17):
    # OLR and cloud-top conditioned on precipitation.
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    precip_bins = np.linspace(0, 60, 41)

    for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
        pr = precip_rate_from_accum(total_precip_accum(ds), dt=dt)
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
        save_figure(17, fig, "olr_cloudtop_conditioned_on_precip")
    print("Done with figure: comp_olr_cloudtop_conditioned_on_precip")

# Full-grid cloud-top and high-cloud diagnostics conditioned on rain and
# convective-environment variables.
if should_run_figure(18):
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.6), constrained_layout=True)
    panel_plotted = [False] * len(ax)

    precip_bins_cloud = np.linspace(0, 80, 41)
    pw_bins_cloud = np.linspace(0, 80, 41)
    cape_bins_cloud = np.linspace(0, 6000, 41)
    cin_bins_cloud = np.linspace(-500, 100, 41)

    for i, ds in enumerate(dss):
        cth = cloud_top_height(ds, thresh=0.05)
        hcld = high_cloud_fraction(ds, zmin=12000.0)
        if cth is None and hcld is None:
            print(f"Skipping conditional cloud diagnostics for {sim_names[i]}: full lat/lon cldfrac unavailable")
            continue

        pr = precip_rate_from_accum(total_precip_accum(ds), dt=dt)
        fields = {"Precip": pr}
        if "precipw" in ds.data_vars:
            fields["PW"] = finite_or_nan(ds.precipw).sel(Time=pr.Time)
        if "cape" in ds.data_vars:
            fields["CAPE"] = finite_or_nan(ds.cape).sel(Time=pr.Time)
        if "cin" in ds.data_vars:
            fields["CIN"] = finite_or_nan(ds.cin).sel(Time=pr.Time)

        if cth is not None:
            cth_km = finite_or_nan(cth).sel(Time=pr.Time) / 1000.0
            panel_plotted[0] |= plot_binned_conditional_curve(
                ax[0], fields["Precip"], cth_km, precip_bins_cloud, cols[i],
                label=sim_names[i], min_count=30, latmax=20
            )
            if "PW" in fields:
                panel_plotted[2] |= plot_binned_conditional_curve(
                    ax[2], fields["PW"], cth_km, pw_bins_cloud, cols[i],
                    label=sim_names[i], min_count=30, latmax=20
                )

        if hcld is not None:
            hcld = finite_or_nan(hcld).sel(Time=pr.Time)
            panel_plotted[1] |= plot_binned_conditional_curve(
                ax[1], fields["Precip"], hcld, precip_bins_cloud, cols[i],
                label=sim_names[i], min_count=30, latmax=20
            )
            if "CAPE" in fields:
                panel_plotted[3] |= plot_binned_conditional_curve(
                    ax[3], fields["CAPE"], hcld, cape_bins_cloud, cols[i],
                    label=sim_names[i], min_count=30, latmax=20
                )
            if "CIN" in fields:
                panel_plotted[4] |= plot_binned_conditional_curve(
                    ax[4], fields["CIN"], hcld, cin_bins_cloud, cols[i],
                    label=sim_names[i], min_count=30, latmax=20
                )

    labels = [
        (r"Precipitation (mm day$^{-1}$)", "Cloud-top height (km)", "CTH vs precip"),
        (r"Precipitation (mm day$^{-1}$)", "High-cloud fraction\nabove 12 km", "High cloud vs precip"),
        (r"Precipitable water (kg m$^{-2}$)", "Cloud-top height (km)", "CTH vs PW"),
        (r"CAPE (J kg$^{-1}$)", "High-cloud fraction\nabove 12 km", "High cloud vs CAPE"),
        (r"CIN (J kg$^{-1}$)", "High-cloud fraction\nabove 12 km", "High cloud vs CIN"),
    ]
    for j, (xlabel, ylabel, title) in enumerate(labels):
        ax[j].set_xlabel(xlabel)
        ax[j].set_ylabel(ylabel)
        ax[j].set_title(title)
        ax[j].grid(alpha=0.2)
        if not panel_plotted[j]:
            ax[j].text(0.5, 0.5, "not loaded", transform=ax[j].transAxes, ha="center", va="center")

    legend_ax = next((a for a, ok in zip(ax, panel_plotted) if ok), ax[0])
    legend_ax.legend(loc="best")
    save_figure(18, fig, "cloud_conditioned_on_precip_pw_cape_cin")
    print("Done with figure: comp_cloud_conditioned_on_precip_pw_cape_cin")

# ============================================================
# 8. Radiation budget proxy
# ============================================================
# What to look for:
#   - Tropical LW-atmospheric heating changes consistent with high-cloud changes
#     support a cloud-radiative pathway.

if should_run_figure(19):
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
            save_figure(19, fig, "atmos_radiative_heating_proxy")
            save_figure(19, fig2, "atmos_radiative_heating_proxy_diff")
        print("Done with figure: comp_atmos_radiative_heating_proxy")
    except ValueError as e:
        print(f"Skipping radiation budget proxy: {e}")

# ============================================================
# 10. Extended tropical-mean time series
# ============================================================
# What to look for:
#   - Coherent changes in PW -> rain -> OLR/high-cloud timing indicate changed
#     convection-radiation coupling even without MJO-like propagation.

if should_run_figure(20):
    fig, ax = plt.subplots(5, 1, figsize=(8, 8), constrained_layout=True, sharex=True)

    for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
        pr = precip_rate_from_accum(total_precip_accum(ds), dt=dt)
        pr_trop = tropical_mean_latlon(pr)
        pw_trop = tropical_mean_latlon(ds.precipw)
        olr_trop = tropical_mean_latlon(ds.olrtoa)

        ax[0].plot(get_time_values(pr_trop, dt=dt), pr_trop, color=cols[i], label=sim_names[i])
        ax[1].plot(get_time_values(pw_trop, dt=dt), pw_trop, color=cols[i])
        ax[2].plot(get_time_values(olr_trop, dt=dt), olr_trop, color=cols[i])

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
                ax[3].plot(get_time_values(high_cld_ts, dt=dt), high_cld_ts, color=cols[i])
        except Exception as e:
            print(f"Skipping high-cloud time series for {sim_names[i]}: {e}")

        hydro = [v for v in ["qc", "qr", "qi", "qs", "qg"] if v in ds.data_vars]
        if len(hydro) > 0:
            qtot = sum(finite_or_nan(ds[v]) for v in hydro)
            qtot_trop = qtot.sel(latitude=slice(-20, 20)).mean(dim=[d for d in qtot.dims if d not in ("Time",)]) * 1000.0
            ax[4].plot(get_time_values(qtot_trop, dt=dt), qtot_trop, color=cols[i])

    ax[0].set_ylabel(r"Rain\n(mm day$^{-1}$)")
    ax[1].set_ylabel("PW")
    ax[2].set_ylabel("OLR")
    ax[3].set_ylabel("High cloud\n10-18 km")
    ax[4].set_ylabel(r"Mean condensate\n(g kg$^{-1}$)")
    ax[4].set_xlabel("Time")
    fig.legend(loc="outside lower center", ncol=len(dss))

    if save:
        save_figure(20, fig, "tropical_timeseries_extended")
    print("Done with figure: comp_tropical_timeseries_extended")

print("All combined symmetry/convection diagnostics complete.")
