#!/usr/bin/env python
# coding: utf-8

"""
Block-bootstrap significance diagnostics for the MPAS-A nudging experiments.

This script is a compact significance wrapper around the diagnostic families
that have proved useful in the other scripts in this directory.  It asks whether
the small nudged-control differences are robust against internal variability,
and whether robust changes look more wind-driven or temperature-driven.

The diagnostics are intentionally computed from the model output here rather
than imported from the other scripts, because those scripts execute full figure
workflows at module top level.
"""

from __future__ import annotations

import os
import sys
import numbers
import warnings

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

try:
    import dask
except ModuleNotFoundError:
    dask = None

from diagnostic_helpers import *

try:
    from waveFilter import kf_filter
except ModuleNotFoundError:
    kf_filter = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = "/home/cwp29/tmp/mpas/crf_on/final/significance_bootstrap/"

# Figure numbers:
#   1: mean-state lat-height bootstrap response maps
#   2: precipitation scalar/PDF bootstrap diagnostics
#   3: cloud profile/cloud-type bootstrap diagnostics
#   4: CCEW wave-box variance bootstrap diagnostics
#
# WHAT TO LOOK FOR:
#   * Treat a response as robust only when the bootstrap confidence interval
#     excludes zero and sign robustness is high across resamples or blocks.
#   * For spatial fields, prioritise coherent latitude-height or physically
#     meaningful regions over isolated robust grid cells.
#   * Compare robust U-control and T-control responses with the U-T contrast:
#     shared thermal/static-stability/cloud-depth structures favour a
#     temperature-driven pathway, while U-only wind/wave residuals favour a
#     wind-driven pathway.
figures_to_run = []
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control", "u_nudge", "T_nudge"]
latlon_file = ["latlon.2000-*.nc", "latlon.2000-*.nc", "latlon.2000-*.nc"]
zm_latlon_file = ["zm_latlon.2000-*.nc", "zm_latlon.2000-*.nc", "zm_latlon.2000-*.nc"]
sim_names = ["control", "nudged u", "nudged T"]
cols = ["k", "b", "r"]

ndays = 200
dt = 6
analysis_end_date = "2000-11-21"
time_format = "time"
obs_per_day = int(round(24 / dt))

BOOTSTRAP_BLOCK_DAYS = 10.0
N_BOOTSTRAP = 1000
FIELD_N_BOOTSTRAP = 600
BOOTSTRAP_SEED = 20240602
MIN_BLOCKS = 6
CI_PERCENTILES = (2.5, 97.5)
ROBUST_SIGN_THRESHOLD = 0.95

TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
DEEP_TROPICS_LAT_BOUNDS = (-10.0, 10.0)
WAVE_FILTER_LAT_BOUNDS = (-10.0, 10.0)

TTL_Z_BOUNDS_M = (14000.0, 18000.0)
HIGH_CLOUD_Z_BOUNDS_M = (12000.0, 18000.0)

PRECIP_STRONG_THRESHOLD = 5.0
PRECIP_WEAK_THRESHOLD = 1.0
PRECIP_HEAVY_THRESHOLD = 20.0
PRECIP_PDF_BINS = np.linspace(0.0, 120.0, 121)

# Cloud-type thresholds copied from convective_pathway_diagnostics.py.
CF_THRESH = 0.05
Z_HIGH_KM = 12.0
Z_LOW_MIN_KM = 2.0
Z_LOW_MAX_KM = 5.0
Z_MID_MIN_KM = 5.0
Z_MID_MAX_KM = 10.0
Z_UPPER_MIN_KM = 10.0
P_DEEP = 5.0
P_WEAK = 1.0
MAX_CLEAR_GAP_KM = 3.0
ANVIL_RADIUS_DEG = 5.0
ANVIL_TIME_STEPS = 1
CLOUD_CLASSIFICATION_CACHE_DIRS = [
    "/home/cwp29/tmp/mpas/crf_on/final/convective_pathways/classification_cache",
    os.path.join(save_loc, "classification_cache"),
]

RD = 287.0
CP = 1004.0
P0_PA = 100000.0
G = 9.80665

COMPARISONS = [
    ("u_minus_control", 1, 0, "nudged u - control"),
    ("T_minus_control", 2, 0, "nudged T - control"),
    ("u_minus_T", 1, 2, "nudged u - nudged T"),
]

LATHEIGHT_DOMAINS = {
    "DeepTropics_TTL_14_18km": dict(lat_bounds=DEEP_TROPICS_LAT_BOUNDS, z_bounds=TTL_Z_BOUNDS_M),
    "Tropics_TTL_14_18km": dict(lat_bounds=TROPICAL_LAT_BOUNDS, z_bounds=TTL_Z_BOUNDS_M),
    "Tropics_highcloud_12_18km": dict(lat_bounds=TROPICAL_LAT_BOUNDS, z_bounds=HIGH_CLOUD_Z_BOUNDS_M),
}

MATCHED_PATTERN_DOMAINS = {
    "TTL_70_200hPa_30S30N": dict(lat_bounds=(-30.0, 30.0), p_bounds=(70.0, 200.0)),
    "UTLS_50_300hPa_40S40N": dict(lat_bounds=(-40.0, 40.0), p_bounds=(50.0, 300.0)),
    "DeepTropics_70_200hPa_10S10N": dict(lat_bounds=(-10.0, 10.0), p_bounds=(70.0, 200.0)),
}

MIS = -999.0
WAVE_BOXES = {
    "Kelvin": dict(k=(1, 14), period=(2.5, 20.0), depth=(8, 90), label="Kelvin", filter_name="Kelvin"),
    "EIG": dict(k=(1, 14), period=(1.8, 4.5), depth=(8, 90), label="EIG", filter_name="IG0"),
    "IG_antisym": dict(k=(-15, -1), period=(1.25, 2.0), depth=(8, 90), label="IG antisym", filter_name="IG2"),
    "AD_sym": dict(k=(-28, -4), period=(5.0, 20.0), depth=(MIS, MIS), label="AD sym", filter_name="AD"),
    "AD_antisym": dict(k=(-28, -4), period=(5.0, 20.0), depth=(MIS, MIS), label="AD antisym", filter_name="AD"),
    "ER": dict(k=(-10, -1), period=(10.0, 40.0), depth=(8, 90), label="ER", filter_name="ER"),
}
WAVE_LATITUDE_PROJECTION = {
    "Kelvin": "symmetric",
    "EIG": "antisymmetric",
    "IG_antisym": "antisymmetric",
    "AD_sym": "symmetric",
    "AD_antisym": "antisymmetric",
    "ER": "symmetric",
}
WAVE_FIELDS = ["Precip", "OLR"]

needed_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw", "olrtoa",
    "qv", "qc", "qr", "qi", "qs", "qg", "relhum", "w",
    "theta", "pressure", "zgrid", "cldfrac",
    "uReconstructZonal", "uReconstructMeridional",
    "t_isobaric", "uzonal_isobaric", "umeridional_isobaric", "rh_isobaric",
    "qv_isobaric",
    "high_cloud_mask", "column_connected_mask", "column_connected_high_cloud_mask",
    "deep_convective_high_cloud_mask", "anvil_high_cloud_mask",
    "cirrus_like_high_cloud_mask", "nonprecipitating_high_cloud_mask",
]

latlon_chunks = {"Time": 8, "latitude": 90, "longitude": 180}
zm_chunks = {"Time": 8}

if dask is not None:
    dask.config.set({"array.slicing.split_large_chunks": True})


# -----------------------------------------------------------------------------
# Bookkeeping and generic bootstrap helpers
# -----------------------------------------------------------------------------

implemented: list[str] = []
skipped_unavailable: list[str] = []
failed: list[str] = []
todo_diagnostics: list[str] = []
scalar_metric_rows: list[dict] = []


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def record_implemented(message):
    print(f"Implemented: {message}")
    implemented.append(message)


def record_skip(message):
    print(f"Skipping: {message}")
    skipped_unavailable.append(message)


def record_failure(message, exc):
    text = f"{message}: {exc}"
    print(f"Failed: {text}")
    failed.append(text)


def record_todo(title, why, implement_first):
    todo_diagnostics.append(
        f"{title}: {why} Implement first in {implement_first}, then reproduce the block metrics here."
    )


def save_table(df, filename):
    if save and len(df) > 0:
        os.makedirs(save_loc, exist_ok=True)
        df.to_csv(os.path.join(save_loc, filename), index=False)


def subset_between(da, dim, bounds):
    if bounds is None or dim not in da.dims:
        return da
    lo, hi = sorted(bounds)
    coord = da[dim]
    return da.where((coord >= lo) & (coord <= hi), drop=True)


def subset_latitude(da, lat_bounds=TROPICAL_LAT_BOUNDS):
    return subset_between(da, "latitude", lat_bounds)


def z_values_m(da):
    z = np.asarray(da.z.values, dtype=float)
    return z if np.nanmax(np.abs(z)) > 100.0 else z * 1000.0


def subset_z_m(da, z_bounds_m):
    if z_bounds_m is None or "z" not in da.dims:
        return da
    z_m = xr.DataArray(z_values_m(da), dims=("z",), coords={"z": da.z})
    lo, hi = sorted(z_bounds_m)
    return da.where((z_m >= lo) & (z_m <= hi), drop=True)


def latitude_weights(da):
    return xr.DataArray(
        np.cos(np.deg2rad(da.latitude.values)),
        dims=("latitude",),
        coords={"latitude": da.latitude},
        name="coslat_weight",
    )


def weighted_mean_dims(da, dims):
    dims = [dim for dim in dims if dim in da.dims]
    if not dims:
        return da
    if "latitude" in da.dims and "latitude" in dims:
        return da.weighted(latitude_weights(da)).mean(dim=dims, skipna=True)
    return da.mean(dim=dims, skipna=True)


def weighted_variance_dims(da, dims):
    dims = [dim for dim in dims if dim in da.dims]
    if not dims:
        return xr.zeros_like(da)
    mean = weighted_mean_dims(da, dims)
    anomaly2 = (da - mean) ** 2
    return weighted_mean_dims(anomaly2, dims)


def time_step_days(da):
    if "Time" not in da.dims or da.sizes["Time"] < 2:
        return dt / 24.0
    values = da.Time.values
    if issubclass(values.dtype.type, numbers.Integral):
        return dt / 24.0
    return float(np.nanmedian(da.Time.diff("Time") / np.timedelta64(1, "D")))


def block_slices(da, block_days=BOOTSTRAP_BLOCK_DAYS):
    if "Time" not in da.dims:
        raise ValueError(f"Cannot block-resample without Time dimension in {da.dims!r}")
    ntime = int(da.sizes["Time"])
    block_len = max(1, int(round(block_days / time_step_days(da))))
    nblocks = ntime // block_len
    if nblocks < MIN_BLOCKS:
        raise ValueError(
            f"Only {nblocks} complete blocks for {getattr(da, 'name', None)!r}; "
            f"need at least {MIN_BLOCKS}. Try a shorter block length."
        )
    return [slice(k * block_len, (k + 1) * block_len) for k in range(nblocks)]


def scalar_from_dataarray(value):
    if isinstance(value, xr.DataArray):
        return float(value.compute().values)
    return float(value)


def block_scalar_values(da, metric_func, block_days=BOOTSTRAP_BLOCK_DAYS):
    vals = []
    for slc in block_slices(da, block_days=block_days):
        vals.append(scalar_from_dataarray(metric_func(da.isel(Time=slc))))
    return np.asarray(vals, dtype=float)


def bootstrap_difference_1d(a, b, rng, n_boot=N_BOOTSTRAP):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < MIN_BLOCKS or b.size < MIN_BLOCKS:
        raise ValueError(f"Insufficient finite blocks ({a.size}, {b.size})")

    response = float(np.nanmean(a) - np.nanmean(b))
    boot = np.empty(n_boot, dtype=float)
    for j in range(n_boot):
        ai = rng.integers(0, a.size, a.size)
        bi = rng.integers(0, b.size, b.size)
        boot[j] = np.nanmean(a[ai]) - np.nanmean(b[bi])

    ci_low, ci_high = np.nanpercentile(boot, CI_PERCENTILES)
    noise = float(np.nanstd(boot, ddof=1))
    sign = np.sign(response)
    sign_robustness = np.nan if sign == 0 else float(np.nanmean(np.sign(boot) == sign))

    n_pair = min(a.size, b.size)
    block_diff = a[:n_pair] - b[:n_pair]
    block_sign_robustness = (
        np.nan if sign == 0 else float(np.nanmean(np.sign(block_diff) == sign))
    )

    return {
        "mean_response": response,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_noise": noise,
        "response_to_noise": response / noise if noise > 0 else np.nan,
        "sign_robustness": sign_robustness,
        "block_sign_robustness": block_sign_robustness,
        "n_blocks_a": int(a.size),
        "n_blocks_b": int(b.size),
        "ci_excludes_zero": bool((ci_low > 0.0) or (ci_high < 0.0)),
        "robust": bool(((ci_low > 0.0) or (ci_high < 0.0)) and sign_robustness >= ROBUST_SIGN_THRESHOLD),
    }


def append_bootstrap_metric_rows(rows, values_by_run, category, diagnostic, metric, units="", extra=None):
    extra = extra or {}
    rng = np.random.default_rng(BOOTSTRAP_SEED + len(rows) + 17)
    for comp_name, run_idx, ref_idx, comp_label in COMPARISONS:
        stats = bootstrap_difference_1d(values_by_run[run_idx], values_by_run[ref_idx], rng)
        row = {
            "category": category,
            "diagnostic": diagnostic,
            "metric": metric,
            "units": units,
            "comparison": comp_name,
            "comparison_label": comp_label,
        }
        row.update(extra)
        row.update(stats)
        rows.append(row)


def interp_spatial_blocks_like(source, target):
    out = source
    interp_kwargs = {}
    for dim in ("z", "pres", "latitude", "longitude", "precip_bin"):
        if dim in out.dims and dim in target.dims and dim in out.coords and dim in target.coords:
            if not np.array_equal(np.asarray(out[dim].values), np.asarray(target[dim].values)):
                interp_kwargs[dim] = target[dim]
    if interp_kwargs:
        out = out.interp(interp_kwargs, kwargs={"fill_value": np.nan})
    return out


def bootstrap_field_difference(a_blocks, b_blocks, rng, n_boot=FIELD_N_BOOTSTRAP):
    b_blocks = interp_spatial_blocks_like(b_blocks, a_blocks)
    template = a_blocks.isel(block=0, drop=True)
    a = np.asarray(a_blocks.values, dtype=np.float64)
    b = np.asarray(b_blocks.values, dtype=np.float64)
    if a.shape[0] < MIN_BLOCKS or b.shape[0] < MIN_BLOCKS:
        raise ValueError(f"Insufficient blocks for field bootstrap: {a.shape[0]}, {b.shape[0]}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        response = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)

    boot = np.empty((n_boot,) + response.shape, dtype=np.float32)
    for j in range(n_boot):
        ai = rng.integers(0, a.shape[0], a.shape[0])
        bi = rng.integers(0, b.shape[0], b.shape[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            boot[j] = np.nanmean(a[ai], axis=0) - np.nanmean(b[bi], axis=0)

    ci_low, ci_high = np.nanpercentile(boot, CI_PERCENTILES, axis=0)
    noise = np.nanstd(boot, axis=0, ddof=1)
    sign = np.sign(response)
    sign_robustness = np.full_like(response, np.nan, dtype=float)
    valid_sign = sign != 0
    sign_robustness[valid_sign] = np.nanmean(np.sign(boot[:, valid_sign]) == sign[valid_sign], axis=0)

    coords = template.coords
    dims = template.dims
    ds = xr.Dataset(
        {
            "mean_response": xr.DataArray(response, dims=dims, coords=coords),
            "ci_low": xr.DataArray(ci_low, dims=dims, coords=coords),
            "ci_high": xr.DataArray(ci_high, dims=dims, coords=coords),
            "bootstrap_noise": xr.DataArray(noise, dims=dims, coords=coords),
            "response_to_noise": xr.DataArray(response / noise, dims=dims, coords=coords),
            "sign_robustness": xr.DataArray(sign_robustness, dims=dims, coords=coords),
        }
    )
    ds["ci_excludes_zero"] = (ds.ci_low > 0.0) | (ds.ci_high < 0.0)
    ds["robust"] = ds.ci_excludes_zero & (ds.sign_robustness >= ROBUST_SIGN_THRESHOLD)
    return ds


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

silence_hdf5_error_stack()

print("Loading lat-lon and zonal-mean datasets")
dss = []
dss_zm = []
for i, d in enumerate(data_locs):
    ds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks=latlon_chunks,
        add_cldfrac_file=True,
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
        add_cldfrac_file=False,
    )
    zds = standardize_latlon_dataset(zds, needed_vars)

    print(f"{sim_names[i]}: latlon vars={len(ds.data_vars)}, zm vars={len(zds.data_vars)}")
    dss.append(ds)
    dss_zm.append(zds)


# -----------------------------------------------------------------------------
# Mean-state pathway: lat-height fields and scalar domain means
# -----------------------------------------------------------------------------

def rename_vertical_to_z(da):
    vdim = vertical_dim(da)
    if vdim != "z":
        da = da.rename({vdim: "z"})
    return da


def zm_temperature(zds):
    missing = [name for name in ("theta", "pressure") if name not in zds.data_vars]
    if missing:
        raise ValueError(f"Missing variables for temperature: {missing}")
    theta = rename_vertical_to_z(finite_or_nan(zds.theta))
    pressure = rename_vertical_to_z(finite_or_nan(zds.pressure))
    out = theta * (pressure / P0_PA) ** (RD / CP)
    out.name = "temperature"
    out.attrs["units"] = "K"
    return out


def zm_static_stability(zds):
    if "theta" not in zds.data_vars:
        raise ValueError("Missing theta for static stability")
    theta = rename_vertical_to_z(finite_or_nan(zds.theta))
    if "z" not in theta.coords:
        raise ValueError("theta has no geometric z coordinate")
    z = np.asarray(theta.z.values, dtype=float)
    coord_to_m = 1.0 if np.nanmax(np.abs(z)) > 100.0 else 1000.0
    dtheta_dz_m = theta.differentiate("z") / coord_to_m
    out = G / theta * dtheta_dz_m
    out.name = "N2"
    out.attrs["units"] = "s-2"
    return out


def zm_lapse_rate(zds):
    temp = zm_temperature(zds)
    z = np.asarray(temp.z.values, dtype=float)
    per_km = 1000.0 if np.nanmax(np.abs(z)) > 100.0 else 1.0
    out = -temp.differentiate("z") * per_km
    out.name = "lapse_rate"
    out.attrs["units"] = "K km-1"
    return out


def zm_vertical_velocity(zds):
    out = vertical_velocity_from_zm(zds)
    out.name = "vertical_velocity"
    return out


def cloud_fraction_time_field(ds, zds):
    source = None
    if "cldfrac" in zds.data_vars:
        source = zds.cldfrac
    elif "cldfrac" in ds.data_vars:
        source = ds.cldfrac
    else:
        raise ValueError("cldfrac is unavailable")
    cf = rename_vertical_to_z(finite_or_nan(source))
    if "z" not in cf.coords:
        raise ValueError("cldfrac has no geometric z coordinate")
    cf.name = "cloud_fraction"
    return cf


def field_from_candidates(zds, candidates, name, scale=1.0):
    for candidate in candidates:
        if candidate in zds.data_vars:
            out = rename_vertical_to_z(finite_or_nan(zds[candidate])) * scale
            out.name = name
            return out
    raise ValueError(f"None of {candidates} available")


LATHEIGHT_FIELD_SPECS = [
    dict(name="temperature", label="Temperature", units="K", diff_fallback=2.0,
         getter=lambda ds, zds: zm_temperature(zds), parity="even"),
    dict(name="zonal_wind", label="Zonal wind", units="m s-1", diff_fallback=5.0,
         getter=lambda ds, zds: field_from_candidates(zds, ("uReconstructZonal",), "zonal_wind"), parity="even"),
    dict(name="static_stability_N2", label="Static stability N2", units="1e-4 s-2", diff_fallback=1.0,
         getter=lambda ds, zds: zm_static_stability(zds) * 1.0e4, parity="even"),
    dict(name="lapse_rate", label="Lapse rate", units="K km-1", diff_fallback=0.5,
         getter=lambda ds, zds: zm_lapse_rate(zds), parity="even"),
    dict(name="relative_humidity", label="Relative humidity", units="%", diff_fallback=10.0,
         getter=lambda ds, zds: field_from_candidates(zds, ("relhum",), "relative_humidity"), parity="even"),
    dict(name="water_vapour", label="Water vapour", units="g kg-1", diff_fallback=1.0,
         getter=lambda ds, zds: field_from_candidates(zds, ("qv",), "water_vapour", scale=1000.0), parity="even"),
    dict(name="vertical_velocity", label="Vertical velocity", units="m s-1", diff_fallback=0.01,
         getter=lambda ds, zds: zm_vertical_velocity(zds), parity="even"),
    dict(name="cloud_fraction", label="Cloud fraction", units="fraction", diff_fallback=0.05,
         getter=cloud_fraction_time_field, parity="even"),
]


def block_latheight_field(da, parity="even", block_days=BOOTSTRAP_BLOCK_DAYS):
    x = rename_vertical_to_z(finite_or_nan(da))
    blocks = []
    for slc in block_slices(x, block_days=block_days):
        xb = x.isel(Time=slc)
        reduce_dims = [dim for dim in ("Time", "longitude") if dim in xb.dims]
        xb = xb.mean(dim=reduce_dims, skipna=True)
        xb = xb.transpose("z", "latitude")
        xb = hemispheric_fold(xb, parity=parity)
        blocks.append(xb.load())
    out = xr.concat(blocks, dim=xr.IndexVariable("block", np.arange(len(blocks))))
    out.name = x.name
    return out


def latheight_domain_values(blocks, lat_bounds, z_bounds):
    x = subset_latitude(blocks, lat_bounds)
    x = subset_z_m(x, z_bounds)
    if x.sizes.get("z", 0) == 0 or x.sizes.get("latitude", 0) == 0:
        raise ValueError("empty lat-height domain")
    return np.asarray(weighted_mean_dims(x, ["z", "latitude"]).values, dtype=float)


def symmetric_field_levels(fields, fallback=1.0, percentile=99.0, ndiv=16):
    vals = []
    for field in fields:
        arr = np.asarray(field, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(np.nanpercentile(np.abs(arr), percentile))
    vmax = max(vals) if vals else fallback
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = fallback
    return symmetric_levels(vmax, ndiv=ndiv)


def plot_latheight_bootstrap(spec, diff_by_comparison):
    if not should_run_figure(1):
        return
    comp_items = [(name, label, ds) for name, _, _, label in COMPARISONS if name in diff_by_comparison]
    if not comp_items:
        return

    fields = [item[2].mean_response.values for item in comp_items]
    levels = symmetric_field_levels(fields, fallback=spec["diff_fallback"])
    fig, axes = plt.subplots(
        1, len(comp_items), figsize=(3.6 * len(comp_items), 3.4),
        constrained_layout=True, sharex=True, sharey=True
    )
    if len(comp_items) == 1:
        axes = [axes]

    last_im = None
    for ax, (comp_name, label, ds) in zip(axes, comp_items):
        z_km = z_values_m(ds.mean_response) / 1000.0
        last_im = ax.contourf(
            ds.latitude, z_km, ds.mean_response,
            levels=levels, cmap="bwr", extend="both"
        )
        robust = ds.robust.where(np.isfinite(ds.mean_response), False)
        ax.contourf(
            ds.latitude, z_km, robust.astype(int),
            levels=[0.5, 1.5], hatches=["///"], colors="none"
        )
        ax.axvline(0.0, color="0.4", lw=0.7)
        ax.set_title(label)
        ax.set_xlabel("Latitude")
        ax.set_ylim(0.0, 25.0)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Height (km)")
    add_symmetric_colorbar(fig, last_im, axes, levels, label=f"{spec['label']} response ({spec['units']})")
    fig.suptitle(
        f"{spec['label']}: mean response; hatching = CI excludes zero and "
        f"bootstrap sign >= {ROBUST_SIGN_THRESHOLD:.2f}"
    )
    figsave.save(fig, f"mean_state_{spec['name']}_bootstrap_latheight", number=1)
    plt.close(fig)


def run_mean_state_pathway():
    for spec in LATHEIGHT_FIELD_SPECS:
        try:
            print(f"\nMean-state bootstrap: {spec['name']}")
            blocks_by_run = {}
            for i in range(len(dss)):
                field = spec["getter"](dss[i], dss_zm[i])
                blocks_by_run[i] = block_latheight_field(field, parity=spec["parity"])

            rng = np.random.default_rng(BOOTSTRAP_SEED + 101)
            diff_by_comp = {}
            for comp_name, run_idx, ref_idx, _ in COMPARISONS:
                diff_by_comp[comp_name] = bootstrap_field_difference(
                    blocks_by_run[run_idx], blocks_by_run[ref_idx], rng
                )
            plot_latheight_bootstrap(spec, diff_by_comp)

            for domain_name, domain in LATHEIGHT_DOMAINS.items():
                values_by_run = {
                    i: latheight_domain_values(
                        blocks_by_run[i],
                        domain["lat_bounds"],
                        domain["z_bounds"],
                    )
                    for i in range(len(dss))
                }
                append_bootstrap_metric_rows(
                    scalar_metric_rows,
                    values_by_run,
                    "mean_state_pathway",
                    spec["name"],
                    domain_name,
                    units=spec["units"],
                    extra={
                        "lat_min": domain["lat_bounds"][0],
                        "lat_max": domain["lat_bounds"][1],
                        "z_min_m": domain["z_bounds"][0],
                        "z_max_m": domain["z_bounds"][1],
                    },
                )
            record_implemented(f"mean-state lat-height bootstrap for {spec['name']}")
        except Exception as exc:
            if "unavailable" in str(exc) or "Missing" in str(exc) or "None of" in str(exc):
                record_skip(f"mean-state {spec['name']} ({exc})")
            else:
                record_failure(f"mean-state {spec['name']}", exc)


# -----------------------------------------------------------------------------
# Precipitation response: mean, variance, tails, and PDFs
# -----------------------------------------------------------------------------

def get_precip_rate(ds):
    if "precip" in ds.data_vars:
        pr = finite_or_nan(ds.precip)
    else:
        missing = [name for name in ("rainnc", "i_rainnc", "rainc", "i_rainc") if name not in ds.data_vars]
        if missing:
            raise ValueError(f"Missing precipitation accumulation variables: {missing}")
        pr = precip_rate_from_accum(total_precip_accum(ds), dt=dt)
    pr.name = "precip"
    pr.attrs["units"] = "mm day-1"
    return finite_or_nan(pr)


def precip_metric_values(pr):
    x = subset_latitude(pr, TROPICAL_LAT_BOUNDS)
    dims = [dim for dim in ("Time", "latitude", "longitude") if dim in x.dims]
    out = {}
    out["tropical_area_time_mean"] = block_scalar_values(
        x,
        lambda xb: weighted_mean_dims(xb, dims),
    )
    out["gridcell_space_time_variance"] = block_scalar_values(
        x,
        lambda xb: weighted_variance_dims(xb, dims),
    )

    def tropical_mean_timeseries_variance(xb):
        ts = weighted_mean_dims(xb, [dim for dim in ("latitude", "longitude") if dim in xb.dims])
        return ts.var(dim="Time", skipna=True)

    out["tropical_mean_timeseries_variance"] = block_scalar_values(x, tropical_mean_timeseries_variance)

    for q in (90.0, 95.0, 99.0):
        out[f"gridcell_p{int(q)}"] = block_scalar_values(
            x,
            lambda xb, q=q: float(np.nanpercentile(np.asarray(xb.values, dtype=float), q)),
        )

    out[f"fraction_gt_{int(PRECIP_HEAVY_THRESHOLD)}mmday"] = block_scalar_values(
        x,
        lambda xb: weighted_mean_dims((xb >= PRECIP_HEAVY_THRESHOLD).astype(float), dims),
    )
    out[f"fraction_lt_{int(PRECIP_WEAK_THRESHOLD)}mmday"] = block_scalar_values(
        x,
        lambda xb: weighted_mean_dims((xb <= PRECIP_WEAK_THRESHOLD).astype(float), dims),
    )
    return out


def precip_pdf_block_values(pr, bins=PRECIP_PDF_BINS):
    x = subset_latitude(pr, TROPICAL_LAT_BOUNDS)
    centres = 0.5 * (bins[:-1] + bins[1:])
    blocks = []
    for slc in block_slices(x):
        vals = np.asarray(x.isel(Time=slc).values, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            hist = np.full(centres.size, np.nan)
        else:
            hist, _ = np.histogram(vals, bins=bins, density=True)
        blocks.append(
            xr.DataArray(hist, dims=("precip_bin",), coords={"precip_bin": centres}, name="precip_pdf")
        )
    return xr.concat(blocks, dim=xr.IndexVariable("block", np.arange(len(blocks))))


def plot_precip_pdf_differences(pdf_diffs):
    if not should_run_figure(2) or not pdf_diffs:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6), constrained_layout=True)
    colors_by_comp = {"u_minus_control": "b", "T_minus_control": "r", "u_minus_T": "0.2"}
    for comp_name, ds in pdf_diffs.items():
        label = next(label for name, _, _, label in COMPARISONS if name == comp_name)
        x = ds.precip_bin.values
        y = ds.mean_response.values
        ax.plot(x, y, color=colors_by_comp.get(comp_name, "k"), lw=1.4, label=label)
        ax.fill_between(x, ds.ci_low.values, ds.ci_high.values, color=colors_by_comp.get(comp_name, "k"), alpha=0.12)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlim(0.0, 80.0)
    ax.set_xlabel("Precipitation (mm day-1)")
    ax.set_ylabel("PDF response")
    ax.set_title("Tropical precipitation PDF bootstrap response")
    ax.legend(frameon=False, fontsize=8)
    figsave.save(fig, "precipitation_pdf_bootstrap_response", number=2)
    plt.close(fig)


def run_precipitation_response():
    try:
        precip_by_run = {i: get_precip_rate(ds) for i, ds in enumerate(dss)}
        metric_by_run = {i: precip_metric_values(precip_by_run[i]) for i in range(len(dss))}
        metric_names = sorted(metric_by_run[0])
        for metric in metric_names:
            values_by_run = {i: metric_by_run[i][metric] for i in range(len(dss))}
            append_bootstrap_metric_rows(
                scalar_metric_rows,
                values_by_run,
                "precipitation_response",
                "precipitation",
                metric,
                units="mm day-1 or derived",
                extra={"lat_min": TROPICAL_LAT_BOUNDS[0], "lat_max": TROPICAL_LAT_BOUNDS[1]},
            )

        pdf_blocks_by_run = {i: precip_pdf_block_values(precip_by_run[i]) for i in range(len(dss))}
        rng = np.random.default_rng(BOOTSTRAP_SEED + 202)
        pdf_diffs = {}
        pdf_rows = []
        for comp_name, run_idx, ref_idx, comp_label in COMPARISONS:
            ds = bootstrap_field_difference(pdf_blocks_by_run[run_idx], pdf_blocks_by_run[ref_idx], rng)
            pdf_diffs[comp_name] = ds
            for k, centre in enumerate(ds.precip_bin.values):
                pdf_rows.append(
                    {
                        "comparison": comp_name,
                        "comparison_label": comp_label,
                        "precip_bin_center_mmday": float(centre),
                        "mean_response": float(ds.mean_response.isel(precip_bin=k).values),
                        "ci_low": float(ds.ci_low.isel(precip_bin=k).values),
                        "ci_high": float(ds.ci_high.isel(precip_bin=k).values),
                        "sign_robustness": float(ds.sign_robustness.isel(precip_bin=k).values),
                        "robust": bool(ds.robust.isel(precip_bin=k).values),
                    }
                )
        save_table(pd.DataFrame(pdf_rows), "precipitation_pdf_bootstrap_response.csv")
        plot_precip_pdf_differences(pdf_diffs)
        record_implemented("precipitation mean, variance, upper-tail, weak-rain fraction, and PDF bootstrap metrics")
        return precip_by_run
    except Exception as exc:
        record_failure("precipitation response bootstrap", exc)
        return {}


# -----------------------------------------------------------------------------
# Cloud-depth and cloud-type response
# -----------------------------------------------------------------------------

def get_cloud_fraction(ds):
    if "cldfrac" not in ds.data_vars:
        return None
    cf = rename_vertical_to_z(finite_or_nan(ds.cldfrac))
    if "z" not in cf.coords:
        return None
    zvals = np.asarray(cf.z.values, dtype=float)
    if np.nanmax(np.abs(zvals)) < 100.0:
        cf = cf.assign_coords(z=cf.z * 1000.0)
    return cf


def cloud_profile_blocks(cf):
    x = subset_latitude(cf, TROPICAL_LAT_BOUNDS)
    blocks = []
    for slc in block_slices(x):
        xb = x.isel(Time=slc)
        reduce_dims = [dim for dim in ("Time", "latitude", "longitude") if dim in xb.dims]
        prof = weighted_mean_dims(xb, reduce_dims).load()
        blocks.append(prof)
    return xr.concat(blocks, dim=xr.IndexVariable("block", np.arange(len(blocks))))


def high_cloud_fraction_blocks(cf):
    high = subset_z_m(cf, HIGH_CLOUD_Z_BOUNDS_M).mean(dim="z", skipna=True)
    high.name = "high_cloud_fraction"
    return block_scalar_values(
        subset_latitude(high, TROPICAL_LAT_BOUNDS),
        lambda xb: weighted_mean_dims(xb, [dim for dim in ("Time", "latitude", "longitude") if dim in xb.dims]),
    )


def plot_cloud_profile_diffs(profile_diffs):
    if not should_run_figure(3) or not profile_diffs:
        return
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.4), constrained_layout=True)
    colors_by_comp = {"u_minus_control": "b", "T_minus_control": "r", "u_minus_T": "0.2"}
    for comp_name, ds in profile_diffs.items():
        label = next(label for name, _, _, label in COMPARISONS if name == comp_name)
        z_km = z_values_m(ds.mean_response) / 1000.0
        ax.plot(ds.mean_response, z_km, color=colors_by_comp.get(comp_name, "k"), lw=1.5, label=label)
        ax.fill_betweenx(z_km, ds.ci_low, ds.ci_high, color=colors_by_comp.get(comp_name, "k"), alpha=0.13)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_ylim(0.0, 25.0)
    ax.set_xlabel("Cloud-fraction response")
    ax.set_ylabel("Height (km)")
    ax.set_title("Tropical vertical cloud-fraction profile")
    ax.legend(frameon=False, fontsize=8)
    figsave.save(fig, "cloud_fraction_profile_bootstrap_response", number=3)
    plt.close(fig)


def available_cloud_times(cf):
    if "Time" not in cf.dims:
        return None
    indexers = {dim: 0 for dim in cf.dims if dim != "Time"}
    valid = cf.isel(indexers).notnull().load()
    return valid.Time.values[np.asarray(valid.values, dtype=bool)]


def z_km_from_cldfrac(cf):
    z = np.asarray(cf.z.values, dtype=float)
    z = z / 1000.0 if np.nanmax(np.abs(z)) > 100.0 else z
    return xr.DataArray(z, dims=("z",), coords={"z": cf.z}, name="z_km")


def any_cloud_in_layer(cloudy, z_km, zmin, zmax):
    return cloudy.where((z_km >= zmin) & (z_km < zmax), drop=True).any(dim="z")


def rolling_or_dilated_max(mask, lat_radius=ANVIL_RADIUS_DEG, lon_radius=ANVIL_RADIUS_DEG, time_radius=ANVIL_TIME_STEPS):
    x = mask.astype(np.int8)
    rolling_kwargs = {}
    if "Time" in x.dims and time_radius > 0:
        rolling_kwargs["Time"] = 2 * int(time_radius) + 1
    if "latitude" in x.dims and lat_radius > 0 and x.sizes.get("latitude", 0) > 1:
        dlat = float(np.nanmedian(np.abs(np.diff(x.latitude.values))))
        rolling_kwargs["latitude"] = 2 * int(np.ceil(lat_radius / dlat)) + 1
    if "longitude" in x.dims and lon_radius > 0 and x.sizes.get("longitude", 0) > 1:
        dlon = float(np.nanmedian(np.abs(np.diff(x.longitude.values))))
        rolling_kwargs["longitude"] = 2 * int(np.ceil(lon_radius / dlon)) + 1
    if not rolling_kwargs:
        return mask
    return x.rolling(**rolling_kwargs, center=True, min_periods=1).max() >= 1


def classify_high_cloud(cldfrac, precip):
    cf = rename_vertical_to_z(finite_or_nan(cldfrac))
    if "z" not in cf.coords:
        raise ValueError("cldfrac has no z coordinate")
    times = available_cloud_times(cf)
    if times is not None:
        cf = cf.sel(Time=times)
    cf, pr = xr.align(cf, precip, join="inner")
    z_km = z_km_from_cldfrac(cf)

    cloudy = cf >= CF_THRESH
    cloud_top_height = z_km.where(cloudy).max(dim="z", skipna=True)
    cloud_top_height.name = "cloud_top_height"

    high_cloud_fraction = cf.where(z_km >= Z_HIGH_KM).mean(dim="z", skipna=True)
    high_cloud_fraction.name = "high_cloud_fraction"
    high_cloud_mask = ((cloud_top_height >= Z_HIGH_KM) | (high_cloud_fraction >= CF_THRESH)).fillna(False)
    high_cloud_mask.name = "high_cloud_mask"

    low_cloud = any_cloud_in_layer(cloudy, z_km, Z_LOW_MIN_KM, Z_LOW_MAX_KM)
    mid_cloud = any_cloud_in_layer(cloudy, z_km, Z_MID_MIN_KM, Z_MID_MAX_KM)
    upper_cloud = any_cloud_in_layer(cloudy, z_km, Z_UPPER_MIN_KM, max(float(z_km.max()), Z_HIGH_KM + 0.1))

    gap_ok = xr.ones_like(high_cloud_mask, dtype=bool)
    if MAX_CLEAR_GAP_KM is not None and MAX_CLEAR_GAP_KM > 0:
        for z0 in np.arange(Z_LOW_MIN_KM, Z_HIGH_KM, MAX_CLEAR_GAP_KM):
            z1 = min(z0 + MAX_CLEAR_GAP_KM, max(float(z_km.max()), Z_HIGH_KM))
            if z1 > z0:
                gap_ok = gap_ok & any_cloud_in_layer(cloudy, z_km, z0, z1 + 1.0e-6)

    column_connected = (high_cloud_mask & low_cloud & mid_cloud & upper_cloud & gap_ok).fillna(False)
    column_connected.name = "column_connected_mask"
    deep_convective = (high_cloud_mask & column_connected & (pr >= P_DEEP)).fillna(False)
    deep_convective.name = "deep_convective_high_cloud_mask"

    near_deep_core = rolling_or_dilated_max(deep_convective).fillna(False)
    anvil = (high_cloud_mask & ~deep_convective & near_deep_core).fillna(False)
    anvil.name = "anvil_high_cloud_mask"
    cirrus_like = (high_cloud_mask & ~column_connected & ~near_deep_core & (pr <= P_WEAK)).fillna(False)
    cirrus_like.name = "cirrus_like_high_cloud_mask"
    nonprecip = (high_cloud_mask & (pr <= P_WEAK)).fillna(False)
    nonprecip.name = "nonprecipitating_high_cloud_mask"

    return {
        "high_cloud_fraction": high_cloud_fraction,
        "high_cloud_mask": high_cloud_mask,
        "column_connected_mask": column_connected,
        "deep_convective_high_cloud_mask": deep_convective,
        "anvil_high_cloud_mask": anvil,
        "cirrus_like_high_cloud_mask": cirrus_like,
        "nonprecipitating_high_cloud_mask": nonprecip,
    }


def existing_cloud_masks(ds, precip_ref):
    aliases = {
        "high_cloud_mask": ("high_cloud_mask",),
        "column_connected_mask": ("column_connected_mask", "column_connected_high_cloud_mask"),
        "deep_convective_high_cloud_mask": ("deep_convective_high_cloud_mask",),
        "anvil_high_cloud_mask": ("anvil_high_cloud_mask",),
        "cirrus_like_high_cloud_mask": ("cirrus_like_high_cloud_mask",),
        "nonprecipitating_high_cloud_mask": ("nonprecipitating_high_cloud_mask",),
    }
    out = {}
    for key, names in aliases.items():
        for name in names:
            if name in ds.data_vars:
                da = ds[name]
                if "Time" in da.dims:
                    da = da.sel(Time=precip_ref.Time)
                out[key] = da
                break
    if len(out) != len(aliases):
        return None

    if "high_cloud_fraction" in ds.data_vars:
        high = finite_or_nan(ds.high_cloud_fraction)
        if "Time" in high.dims:
            high = high.sel(Time=precip_ref.Time)
        out["high_cloud_fraction"] = high
    else:
        cf = get_cloud_fraction(ds)
        if cf is None:
            return None
        if "Time" in cf.dims:
            cf = cf.sel(Time=precip_ref.Time)
        out["high_cloud_fraction"] = subset_z_m(cf, HIGH_CLOUD_Z_BOUNDS_M).mean(dim="z", skipna=True)
    return out


def cloud_masks_from_cache(sim_name, precip_ref):
    filename = f"{sim_name.replace(' ', '_')}_classified_high_cloud.nc"
    aliases = {
        "high_cloud_fraction": ("high_cloud_fraction",),
        "high_cloud_mask": ("high_cloud_mask",),
        "column_connected_mask": ("column_connected_mask", "column_connected_high_cloud_mask"),
        "deep_convective_high_cloud_mask": ("deep_convective_high_cloud_mask",),
        "anvil_high_cloud_mask": ("anvil_high_cloud_mask",),
        "cirrus_like_high_cloud_mask": ("cirrus_like_high_cloud_mask",),
        "nonprecipitating_high_cloud_mask": ("nonprecipitating_high_cloud_mask",),
    }

    for cache_dir in CLOUD_CLASSIFICATION_CACHE_DIRS:
        path = os.path.join(cache_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            cached = xr.open_dataset(path, chunks={})
            out = {}
            for key, names in aliases.items():
                for name in names:
                    if name in cached.data_vars:
                        da = cached[name]
                        if "Time" in da.dims and "Time" in precip_ref.dims:
                            common = da.Time.to_index().intersection(precip_ref.Time.to_index())
                            da = da.sel(Time=common)
                        out[key] = da
                        break
            if len(out) == len(aliases):
                print(f"Using cached high-cloud classification for {sim_name}: {path}")
                return out
            cached.close()
        except Exception as exc:
            print(f"Could not use cached high-cloud classification {path}: {exc}")
    return None


def block_area_fraction_values(da):
    x = subset_latitude(da.astype(float), TROPICAL_LAT_BOUNDS)
    dims = [dim for dim in ("Time", "latitude", "longitude") if dim in x.dims]
    return block_scalar_values(x, lambda xb: weighted_mean_dims(xb, dims))


def run_cloud_response(precip_by_run):
    try:
        cf_by_run = {}
        for i in range(len(dss)):
            cf = get_cloud_fraction(dss_zm[i])
            if cf is None:
                cf = get_cloud_fraction(dss[i])
            if cf is None:
                raise ValueError(f"cldfrac unavailable for {sim_names[i]}")
            cf_by_run[i] = cf

        profile_by_run = {i: cloud_profile_blocks(cf_by_run[i]) for i in range(len(dss))}
        rng = np.random.default_rng(BOOTSTRAP_SEED + 303)
        profile_diffs = {}
        for comp_name, run_idx, ref_idx, _ in COMPARISONS:
            profile_diffs[comp_name] = bootstrap_field_difference(profile_by_run[run_idx], profile_by_run[ref_idx], rng)
        plot_cloud_profile_diffs(profile_diffs)

        high_values_by_run = {i: high_cloud_fraction_blocks(cf_by_run[i]) for i in range(len(dss))}
        append_bootstrap_metric_rows(
            scalar_metric_rows,
            high_values_by_run,
            "cloud_depth_response",
            "cloud_fraction",
            "tropical_high_cloud_fraction_12_18km",
            units="fraction",
        )
        record_implemented("high-cloud fraction and vertical cloud-fraction profile bootstrap metrics")
    except Exception as exc:
        record_failure("cloud fraction profile/depth bootstrap", exc)

    if not precip_by_run:
        record_skip("cloud-type bootstrap (precipitation unavailable for classification)")
        return

    try:
        class_by_run = {}
        for i in range(len(dss)):
            masks = existing_cloud_masks(dss[i], precip_by_run[i])
            if masks is None:
                masks = cloud_masks_from_cache(sim_names[i], precip_by_run[i])
            if masks is None:
                cf = get_cloud_fraction(dss[i])
                if cf is None:
                    raise ValueError(f"full lat-lon cldfrac unavailable for {sim_names[i]}")
                masks = classify_high_cloud(cf, precip_by_run[i])
            class_by_run[i] = masks

        for key in (
            "high_cloud_fraction",
            "high_cloud_mask",
            "column_connected_mask",
            "deep_convective_high_cloud_mask",
            "anvil_high_cloud_mask",
            "cirrus_like_high_cloud_mask",
            "nonprecipitating_high_cloud_mask",
        ):
            values_by_run = {i: block_area_fraction_values(class_by_run[i][key]) for i in range(len(dss))}
            append_bootstrap_metric_rows(
                scalar_metric_rows,
                values_by_run,
                "cloud_type_response",
                "high_cloud_classification",
                key,
                units="fraction",
            )
        record_implemented("column-connected, deep-convective, anvil, cirrus-like, and nonprecipitating high-cloud class bootstrap metrics")
    except Exception as exc:
        record_failure("cloud-type classification bootstrap", exc)
        record_todo(
            "Cloud-type bootstrap refinement",
            "Cloud-type differences matter because high cloud is mechanistically ambiguous without separating deep convective towers from anvil and detached cirrus.",
            "convective_pathway_diagnostics.py by exposing an import-safe cached classification dataset and block-metric helper",
        )


# -----------------------------------------------------------------------------
# Conditional diagnostics: strong/weak precipitation states
# -----------------------------------------------------------------------------

def aligned_optional_field(ds, key, ref):
    aliases = {
        "OLR": ("olrtoa", "olr", "OLR"),
        "PW": ("precipw", "pw", "prw"),
    }
    for name in aliases.get(key, (key,)):
        if name in ds.data_vars:
            out = finite_or_nan(ds[name])
            if "Time" in out.dims and "Time" in ref.dims:
                out = out.sel(Time=ref.Time)
            out.name = key
            return out
    return None


def conditional_area_mean_blocks(target, condition):
    target, condition = xr.align(target, condition, join="inner")
    x = subset_latitude(target, TROPICAL_LAT_BOUNDS)
    c = subset_latitude(condition, TROPICAL_LAT_BOUNDS)
    dims = [dim for dim in ("Time", "latitude", "longitude") if dim in x.dims]
    return block_scalar_values(x, lambda xb: weighted_mean_dims(xb.where(c.sel(Time=xb.Time)), dims))


def run_conditional_precipitation_states(precip_by_run):
    if not precip_by_run:
        record_skip("strong/weak precipitation conditional bootstrap (precipitation unavailable)")
        return
    try:
        targets = {
            "precip": {i: precip_by_run[i] for i in range(len(dss))},
            "OLR": {i: aligned_optional_field(dss[i], "OLR", precip_by_run[i]) for i in range(len(dss))},
            "PW": {i: aligned_optional_field(dss[i], "PW", precip_by_run[i]) for i in range(len(dss))},
        }
        for target_name, by_run in list(targets.items()):
            if any(by_run[i] is None for i in range(len(dss))):
                record_skip(f"conditional {target_name} on precipitation states (variable unavailable)")
                targets.pop(target_name)

        for state_name, condition_func in (
            ("strong_precip_ge_5mmday", lambda pr: pr >= PRECIP_STRONG_THRESHOLD),
            ("weak_precip_le_1mmday", lambda pr: pr <= PRECIP_WEAK_THRESHOLD),
        ):
            for target_name, by_run in targets.items():
                values_by_run = {
                    i: conditional_area_mean_blocks(by_run[i], condition_func(precip_by_run[i]))
                    for i in range(len(dss))
                }
                append_bootstrap_metric_rows(
                    scalar_metric_rows,
                    values_by_run,
                    "conditional_response",
                    target_name,
                    state_name,
                    units=getattr(by_run[0], "units", ""),
                )
        record_implemented("strong-versus-weak precipitation conditional bootstrap metrics for precip/OLR/PW")
        record_todo(
            "Kelvin-active versus inactive conditional bootstrap",
            "State-dependent changes can be hidden in small means, and Kelvin-active sampling directly tests wave-coupled convection.",
            "equatorial_wave_diagnostics.py by exposing cached Kelvin amplitude indices as import-safe block metrics",
        )
        record_todo(
            "Organised versus disorganised convection bootstrap",
            "Organisation-conditioned changes test whether nudging affects intermittency and object clustering rather than only rain amount.",
            "convective_organisation_diagnostics.py by writing a reusable per-time Lorg/object metric table",
        )
    except Exception as exc:
        record_failure("conditional strong/weak precipitation bootstrap", exc)


# -----------------------------------------------------------------------------
# Wave-mediated pathway: integrated CCEW wave-box variance
# -----------------------------------------------------------------------------

def wave_label(wave):
    return WAVE_BOXES[wave]["label"]


def wave_cache_dir():
    path = os.path.join(save_loc, "wave_cache")
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "-").replace(".", "p").replace("-", "m")


def get_wave_filter_field(ds, field_name):
    pr = get_precip_rate(ds)
    if field_name == "Precip":
        return pr
    if field_name == "OLR":
        if "olrtoa" not in ds.data_vars:
            raise ValueError("olrtoa unavailable")
        return finite_or_nan(ds.olrtoa).sel(Time=pr.Time)
    raise ValueError(f"Unsupported wave field {field_name}")


def project_wave_latitude(da, wave_name):
    projection = WAVE_LATITUDE_PROJECTION[wave_name]
    lo, hi = WAVE_FILTER_LAT_BOUNDS
    if projection == "symmetric":
        x = subset_between(da, "latitude", (lo, hi))
        return weighted_mean_dims(x, ["latitude"]).transpose("Time", "longitude")

    half = max(abs(lo), abs(hi))
    north = subset_between(da, "latitude", (0.0, half))
    south = subset_between(da, "latitude", (-half, 0.0))
    north_mean = weighted_mean_dims(north, ["latitude"])
    south_mean = weighted_mean_dims(south, ["latitude"])
    north_mean, south_mean = xr.align(north_mean, south_mean, join="inner")
    if projection == "antisymmetric":
        return (0.5 * (north_mean - south_mean)).transpose("Time", "longitude")
    raise ValueError(f"Unsupported wave projection {projection}")


def filter_projected_wave(projected, wave_name, sim_index, field_name):
    cache_path = os.path.join(
        wave_cache_dir(),
        f"{safe_name(sim_names[sim_index])}_{safe_name(field_name)}_{safe_name(wave_name)}_bootstrap.nc",
    )
    if os.path.exists(cache_path):
        try:
            cached = xr.open_dataset(cache_path)
            if (
                cached.attrs.get("field") == field_name
                and cached.attrs.get("wave") == wave_name
                and cached.attrs.get("source_simulation") == sim_names[sim_index]
            ):
                return cached["wave"]
            cached.close()
        except Exception:
            pass

    if kf_filter is None:
        raise ModuleNotFoundError("waveFilter.kf_filter is unavailable")

    x = projected.load()
    arr = np.asarray(x.values, dtype=np.float32)
    arr = arr - np.nanmean(arr, axis=0, keepdims=True)
    arr[~np.isfinite(arr)] = 0.0

    box = WAVE_BOXES[wave_name]
    k0, k1 = box["k"]
    p0, p1 = box["period"]
    h0, h1 = box["depth"]
    filt = np.zeros((arr.shape[0] + 702, arr.shape[1]), dtype="f")
    filt[: arr.shape[0], :] = arr
    wave = kf_filter(filt, obs_per_day, p0, p1, k0, k1, h0, h1, box["filter_name"])
    out = xr.DataArray(
        wave[: arr.shape[0], :],
        dims=("Time", "longitude"),
        coords={"Time": x.Time, "longitude": x.longitude},
        name="wave",
        attrs={
            "field": field_name,
            "wave": wave_name,
            "wave_label": box["label"],
            "source_simulation": sim_names[sim_index],
            "units": x.attrs.get("units", ""),
        },
    )
    out_ds = out.to_dataset()
    out_ds.attrs.update(out.attrs)
    tmp_cache_path = cache_path + ".tmp"
    out_ds.to_netcdf(tmp_cache_path)
    os.replace(tmp_cache_path, cache_path)
    return xr.open_dataset(cache_path)["wave"]


def block_variance_values(da):
    return block_scalar_values(
        da,
        lambda xb: ((xb - xb.mean(dim="Time", skipna=True)) ** 2).mean(
            dim=[dim for dim in ("Time", "longitude") if dim in xb.dims],
            skipna=True,
        ),
    )


def run_wave_pathway():
    if kf_filter is None:
        record_skip("wave-mediated bootstrap (waveFilter.kf_filter unavailable)")
        record_todo(
            "CCEW wave-box bootstrap",
            "Integrated wave-box variance tests whether nudged responses are tied to organised equatorial wave-coupled convection.",
            "equatorial_wave_diagnostics.py or waveFilter import configuration",
        )
        return

    try:
        wave_rows = []
        for field_name in WAVE_FIELDS:
            projected_by_run_by_wave = {}
            for wave_name in WAVE_BOXES:
                wave_values_by_run = {}
                frac_values_by_run = {}
                for i in range(len(dss)):
                    raw = get_wave_filter_field(dss[i], field_name)
                    projected = project_wave_latitude(raw, wave_name)
                    filtered = filter_projected_wave(projected, wave_name, i, field_name)
                    wave_var = block_variance_values(filtered)
                    total_var = block_variance_values(projected)
                    wave_values_by_run[i] = wave_var
                    frac_values_by_run[i] = wave_var / total_var
                for values, metric in (
                    (wave_values_by_run, "wave_variance"),
                    (frac_values_by_run, "wave_variance_fraction"),
                ):
                    before = len(scalar_metric_rows)
                    append_bootstrap_metric_rows(
                        scalar_metric_rows,
                        values,
                        "wave_mediated_pathway",
                        field_name,
                        metric,
                        units="variance or fraction",
                        extra={
                            "wave": wave_name,
                            "wave_label": wave_label(wave_name),
                            "latitude_projection": WAVE_LATITUDE_PROJECTION[wave_name],
                        },
                    )
                    wave_rows.extend(scalar_metric_rows[before:])
                projected_by_run_by_wave[wave_name] = True

        wave_df = pd.DataFrame(wave_rows)
        save_table(wave_df, "wave_box_variance_bootstrap_metrics.csv")
        plot_wave_summary(wave_df)
        record_implemented("integrated Kelvin/CCEW-filtered Precip and OLR wave-box variance bootstrap metrics")
    except Exception as exc:
        record_failure("wave-mediated pathway bootstrap", exc)


def plot_wave_summary(wave_df):
    if not should_run_figure(4) or len(wave_df) == 0:
        return
    sub = wave_df[
        (wave_df["metric"] == "wave_variance_fraction")
        & (wave_df["comparison"].isin(["u_minus_control", "T_minus_control"]))
    ].copy()
    if len(sub) == 0:
        return
    waves = list(WAVE_BOXES)
    fields = sorted(sub["diagnostic"].unique())
    fig, axes = plt.subplots(len(fields), 1, figsize=(7.2, 3.0 * len(fields)), constrained_layout=True, sharex=True)
    if len(fields) == 1:
        axes = [axes]
    offsets = {"u_minus_control": -0.15, "T_minus_control": 0.15}
    colors_by_comp = {"u_minus_control": "b", "T_minus_control": "r"}
    x = np.arange(len(waves))
    for ax, field in zip(axes, fields):
        for comp in ("u_minus_control", "T_minus_control"):
            y = []
            err_low = []
            err_high = []
            for wave in waves:
                row = sub[(sub["diagnostic"] == field) & (sub["comparison"] == comp) & (sub["wave"] == wave)]
                if len(row) == 0:
                    y.append(np.nan)
                    err_low.append(np.nan)
                    err_high.append(np.nan)
                    continue
                row = row.iloc[0]
                y.append(row["mean_response"])
                err_low.append(row["mean_response"] - row["ci_low"])
                err_high.append(row["ci_high"] - row["mean_response"])
            label = next(label for name, _, _, label in COMPARISONS if name == comp)
            ax.errorbar(
                x + offsets[comp], y, yerr=[err_low, err_high], fmt="o",
                color=colors_by_comp[comp], label=label, capsize=2
            )
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_ylabel(f"{field}\nwave-fraction response")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([WAVE_BOXES[w]["label"] for w in waves], rotation=25, ha="right")
    fig.suptitle("CCEW wave-box variance-fraction bootstrap response")
    figsave.save(fig, "wave_box_variance_fraction_bootstrap_summary", number=4)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Matched-pattern attribution diagnostics
# -----------------------------------------------------------------------------

def block_latpres_field(zds, var, parity="even"):
    if var not in zds.data_vars:
        raise ValueError(f"{var} unavailable")
    da = finite_or_nan(zds[var])
    if "pres" not in da.dims:
        raise ValueError(f"{var} has no pres dimension")
    blocks = []
    for slc in block_slices(da):
        xb = da.isel(Time=slc).mean(dim="Time", skipna=True).transpose("pres", "latitude")
        xb = hemispheric_fold(xb, parity=parity)
        blocks.append(xb.load())
    return xr.concat(blocks, dim=xr.IndexVariable("block", np.arange(len(blocks))))


def subset_latpres_domain(blocks, lat_bounds, p_bounds):
    x = subset_between(blocks, "latitude", lat_bounds)
    x = subset_between(x, "pres", p_bounds)
    if x.sizes.get("pres", 0) == 0 or x.sizes.get("latitude", 0) == 0:
        raise ValueError("empty matched-pattern domain")
    return x


def latpres_weights(template):
    weight = xr.ones_like(template)
    weight = weight * latitude_weights(template)
    weight = weight * pressure_layer_weights(template.pres)
    return np.asarray(weight.values, dtype=float)


def weighted_projection_metrics(target, basis, weight):
    target = np.asarray(target, dtype=float)
    basis = np.asarray(basis, dtype=float)
    valid = np.isfinite(target) & np.isfinite(basis) & np.isfinite(weight) & (weight > 0)
    if np.count_nonzero(valid) < 3:
        return np.nan, np.nan, np.nan, np.nan
    t = target[valid]
    b = basis[valid]
    w = weight[valid]
    numerator = np.sum(t * b * w)
    denominator = np.sum(b * b * w)
    target_norm2 = np.sum(t * t * w)
    basis_norm2 = denominator
    if denominator <= 0 or target_norm2 <= 0 or basis_norm2 <= 0:
        return np.nan, np.nan, np.nan, np.nan
    alpha = numerator / denominator
    residual = t - alpha * b
    residual_norm2 = np.sum(residual * residual * w)
    corr = numerator / np.sqrt(target_norm2 * basis_norm2)
    residual_fraction = np.sqrt(residual_norm2 / target_norm2)
    explained_fraction = 1.0 - residual_norm2 / target_norm2
    return float(alpha), float(corr), float(residual_fraction), float(explained_fraction)


def bootstrap_matched_pattern(blocks_control, blocks_u, blocks_T, domain_name, domain):
    u = subset_latpres_domain(blocks_u, domain["lat_bounds"], domain["p_bounds"])
    T = subset_latpres_domain(blocks_T, domain["lat_bounds"], domain["p_bounds"])
    c = subset_latpres_domain(blocks_control, domain["lat_bounds"], domain["p_bounds"])
    T = interp_spatial_blocks_like(T, u)
    c = interp_spatial_blocks_like(c, u)
    template = u.isel(block=0, drop=True)
    weight = latpres_weights(template)

    u_values = np.asarray(u.values, dtype=float)
    T_values = np.asarray(T.values, dtype=float)
    c_values = np.asarray(c.values, dtype=float)

    mean_du = np.nanmean(u_values, axis=0) - np.nanmean(c_values, axis=0)
    mean_dT = np.nanmean(T_values, axis=0) - np.nanmean(c_values, axis=0)
    alpha, corr, resid, explained = weighted_projection_metrics(mean_du, mean_dT, weight)

    rng = np.random.default_rng(BOOTSTRAP_SEED + 404 + len(domain_name))
    metrics = {"alpha_pattern": [], "weighted_pattern_corr": [], "residual_fraction": [], "explained_fraction": []}
    for _ in range(N_BOOTSTRAP):
        ui = rng.integers(0, u_values.shape[0], u_values.shape[0])
        Ti = rng.integers(0, T_values.shape[0], T_values.shape[0])
        ci_u = rng.integers(0, c_values.shape[0], c_values.shape[0])
        ci_T = rng.integers(0, c_values.shape[0], c_values.shape[0])
        du = np.nanmean(u_values[ui], axis=0) - np.nanmean(c_values[ci_u], axis=0)
        dT = np.nanmean(T_values[Ti], axis=0) - np.nanmean(c_values[ci_T], axis=0)
        a, r, rf, ef = weighted_projection_metrics(du, dT, weight)
        metrics["alpha_pattern"].append(a)
        metrics["weighted_pattern_corr"].append(r)
        metrics["residual_fraction"].append(rf)
        metrics["explained_fraction"].append(ef)

    rows = []
    point = {
        "alpha_pattern": alpha,
        "weighted_pattern_corr": corr,
        "residual_fraction": resid,
        "explained_fraction": explained,
    }
    for metric, vals in metrics.items():
        vals = np.asarray(vals, dtype=float)
        ci_low, ci_high = np.nanpercentile(vals, CI_PERCENTILES)
        rows.append(
            {
                "category": "matched_pattern_attribution",
                "diagnostic": "temperature_latpres",
                "domain": domain_name,
                "metric": metric,
                "mean_response": point[metric],
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "bootstrap_noise": float(np.nanstd(vals, ddof=1)),
                "n_bootstrap": N_BOOTSTRAP,
                "lat_min": domain["lat_bounds"][0],
                "lat_max": domain["lat_bounds"][1],
                "p_min_hPa": min(domain["p_bounds"]),
                "p_max_hPa": max(domain["p_bounds"]),
            }
        )
    return rows


def run_matched_pattern_attribution():
    try:
        blocks_c = block_latpres_field(dss_zm[0], "t_isobaric", parity="even")
        blocks_u = block_latpres_field(dss_zm[1], "t_isobaric", parity="even")
        blocks_T = block_latpres_field(dss_zm[2], "t_isobaric", parity="even")
        rows = []
        for domain_name, domain in MATCHED_PATTERN_DOMAINS.items():
            rows.extend(bootstrap_matched_pattern(blocks_c, blocks_u, blocks_T, domain_name, domain))
        df = pd.DataFrame(rows)
        save_table(df, "matched_pattern_bootstrap_metrics.csv")
        record_implemented("matched-pattern projection bootstrap for u-nudged temperature response onto T-nudged temperature response")
        return df
    except Exception as exc:
        record_failure("matched-pattern attribution bootstrap", exc)
        record_todo(
            "Matched convective/cloud/precipitation pattern projections",
            "These would directly test whether U-nudged or T-nudged responses project more strongly onto the actual precipitation/cloud-change pattern.",
            "matched_pattern_response.py by adding import-safe basis/target projection helpers for surface and cloud fields",
        )
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def concise_metric_table(df):
    if len(df) == 0:
        return
    robust = df[df.get("robust", False) == True].copy()
    print("\nRobust scalar responses (CI excludes zero and bootstrap sign is robust):")
    if len(robust) == 0:
        print("  None by the current thresholds.")
        return
    robust["abs_rtn"] = robust["response_to_noise"].abs()
    cols_to_show = [
        "category", "diagnostic", "metric", "comparison",
        "mean_response", "ci_low", "ci_high", "response_to_noise", "sign_robustness",
    ]
    print(robust.sort_values("abs_rtn", ascending=False)[cols_to_show].head(25).to_string(index=False))


def print_mechanism_cues(metrics_df, matched_df):
    print("\nMechanism-oriented cues:")
    if len(matched_df) > 0:
        pivot = matched_df.pivot_table(index="domain", columns="metric", values="mean_response", aggfunc="first")
        for domain, row in pivot.iterrows():
            corr = row.get("weighted_pattern_corr", np.nan)
            resid = row.get("residual_fraction", np.nan)
            alpha = row.get("alpha_pattern", np.nan)
            print(
                f"  Matched thermal pattern {domain}: alpha={alpha:.2f}, "
                f"corr={corr:.2f}, residual_fraction={resid:.2f}."
            )
        print("  High correlation and low residual_fraction support a shared thermal pattern; coherent robust residuals or wave/wind-only changes support an added wind/dynamical pathway.")
    else:
        print("  Matched-pattern metrics were not available.")

    if len(metrics_df) > 0:
        wave_robust = metrics_df[
            (metrics_df["category"] == "wave_mediated_pathway")
            & (metrics_df["comparison"].isin(["u_minus_control", "T_minus_control"]))
            & (metrics_df.get("robust", False) == True)
        ]
        if len(wave_robust) > 0:
            print("  Robust wave-box changes are present; compare U-control and T-control rows in wave_box_variance_bootstrap_metrics.csv for wind-mediated organisation.")
        thermal_robust = metrics_df[
            (metrics_df["category"] == "mean_state_pathway")
            & (metrics_df["diagnostic"].isin(["temperature", "static_stability_N2", "lapse_rate", "cloud_fraction"]))
            & (metrics_df["comparison"].isin(["u_minus_control", "T_minus_control"]))
            & (metrics_df.get("robust", False) == True)
        ]
        if len(thermal_robust) > 0:
            print("  Robust temperature/static-stability/cloud-depth changes are present; these are the core thermal-pathway checks.")


def print_final_summary(metrics_df, matched_df):
    print("\n" + "=" * 78)
    print("Significance bootstrap summary")
    print("=" * 78)

    print("\nDiagnostics implemented and run:")
    if implemented:
        for item in implemented:
            print(f"  - {item}")
    else:
        print("  - None")

    print("\nDiagnostics skipped because variables or helpers were unavailable:")
    if skipped_unavailable:
        for item in skipped_unavailable:
            print(f"  - {item}")
    else:
        print("  - None")

    print("\nDiagnostics that failed:")
    if failed:
        for item in failed:
            print(f"  - {item}")
    else:
        print("  - None")

    print("\nScientifically useful TODO diagnostics:")
    if todo_diagnostics:
        for item in todo_diagnostics:
            print(f"  - {item}")
    else:
        print("  - None beyond sensitivity tests/threshold refinements.")

    concise_metric_table(metrics_df)
    print_mechanism_cues(metrics_df, matched_df)

    print("\nAnswering the headline question:")
    print("  Use bootstrap_scalar_metrics.csv for scalar robustness, the lat-height figures for coherent structures rather than isolated gridpoints, wave_box_variance_bootstrap_metrics.csv for organised wave-mediated changes, and matched_pattern_bootstrap_metrics.csv for thermal-pattern attribution.")
    print("  Robust U-control changes in zonal wind or CCEW variance that are absent from T-control point toward a wind-driven pathway; robust shared temperature/N2/cloud-depth structures with high matched-pattern correlation point toward a temperature-driven pathway.")


# -----------------------------------------------------------------------------
# Main top-level workflow
# -----------------------------------------------------------------------------

print("\nRunning significance bootstrap diagnostics")
run_mean_state_pathway()
precip_by_run = run_precipitation_response()
run_cloud_response(precip_by_run)
run_conditional_precipitation_states(precip_by_run)
run_wave_pathway()
matched_pattern_df = run_matched_pattern_attribution()

metrics_df = pd.DataFrame(scalar_metric_rows)
save_table(metrics_df, "bootstrap_scalar_metrics.csv")
print_final_summary(metrics_df, matched_pattern_df)
