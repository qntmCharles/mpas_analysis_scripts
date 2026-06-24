#!/usr/bin/env python
# coding: utf-8

"""
Refined high-cloud classification diagnostics for MPAS-A nudging experiments.

This script owns the shared high-cloud classification cache and focuses on a
small set of interpretable tests:

  * high-cloud class budgets and nudged-control differences;
  * observed-minus-control-predicted high-cloud residual maps/zonal means;
  * lat/lon overlay snapshots that validate the class masks visually;
  * object/distance checks for convectively generated versus cirrus-like cloud;
  * lightweight cloud-fraction threshold sensitivity checks.

The diagnostics deliberately avoid tercile splits of N2_TTL.  If N2_TTL is
available, a single common-support range is reported in the residual summary
tables rather than plotted as multiple weakly supported conditional regimes.
"""

import os
import sys
import warnings
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in (".", "..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import dask
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors, pyplot as plt
from scipy import ndimage

from diagnostic_helpers import *
from high_cloud_classification import (
    HYDROMETEOR_NEEDED_VARS,
    HighCloudThresholds,
    cached_classification_is_current as shared_cached_classification_is_current,
    classification_cache_metadata as shared_classification_cache_metadata,
    classification_dataset as shared_classification_dataset,
    classify_high_cloud_from_dataset,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/high_cloud_refined/")

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: high_cloud_class_budget
#   2: residual_high_cloud_maps_zonal
#   3: cloud_classification_validation_snapshots
#   4: cloud_class_distance_to_precip_core
#   5: cloud_class_time_since_precip
#   6: cloud_classification_threshold_sensitivity
figures_to_run = []
figsave = None

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

TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
SNAPSHOT_LAT_BOUNDS = (-20.0, 20.0)

# The classifier uses cldfrac for parent cloud geometry and 3-D hydrometeors for
# physical interpretation. Surface precipitation is a supporting constraint for
# active deep convection, while vapour is intentionally not class-defining.
CF_THRESH = 0.05
Z_HIGH = 12.0
Z_LOW_MIN = 2.0
Z_LOW_MAX = 5.0
Z_MID_MIN = 5.0
Z_MID_MAX = 10.0
Z_UPPER_MIN = 10.0
P_DEEP = 5.0
P_WEAK = 1.0
MAX_CLEAR_GAP = 3.0
DEEP_MIN_DEPTH_KM = 8.0
HYDROMETEOR_PRESENT = 1.0e-8
SUBSTANTIAL_HYDROMETEOR = 1.0e-7
CIRRUS_MAX_DEPTH_KM = 4.0
CONVECTIVE_GENERATED_DISTANCE_KM = 300.0
CIRRUS_MIN_DISTANCE_KM = 500.0

SAVE_CLASSIFICATION_CACHE = True
classification_cache_dir = diagnostic_cache_dir("classification_cache", root_dir=root_dir)
classification_cache_read_dirs = unique_existing_order([classification_cache_dir])

BOOTSTRAP_BLOCK_DAYS = 10.0
N_BOOTSTRAP = 600
BOOTSTRAP_SEED = 20240610
CI_PERCENTILES = (2.5, 97.5)

LATITUDE_BIN_WIDTH = 2.0
MIN_BIN_COUNT = 50

RESIDUAL_FIT_SAMPLE_LIMITS = dict(max_time=120, max_lat=120, max_lon=240)
RESIDUAL_PLOT_SAMPLE_LIMITS = dict(max_time=144, max_lat=120, max_lon=240)
DISTANCE_SAMPLE_LIMITS = dict(max_time=96, max_lat=120, max_lon=240)
LAG_SAMPLE_LIMITS = dict(max_time=192, max_lat=100, max_lon=220)
SENSITIVITY_SAMPLE_LIMITS = dict(max_time=144, max_lat=100, max_lon=220)

DISTANCE_BINS_KM = np.array([0, 100, 250, 500, 750, 1000, 1500, 2500, 4000, 6000, 9000], dtype=float)
LAG_BINS_HOURS = np.array([0, 6, 12, 24, 48, 72, 120, 240, 480], dtype=float)
LAG_NEAR_PRECIP_RADIUS_DEG = 5.0
SENSITIVITY_CF_THRESHOLDS = [0.03, CF_THRESH, 0.10]
SNAPSHOT_LON_WINDOW_DEG = 90.0
SNAPSHOT_Z_MAX_KM = 19.0

latlon_chunks = {"Time": 8, "latitude": 90, "longitude": 180}
zm_chunks = {"Time": -1}
dask.config.set({"array.slicing.split_large_chunks": True})

VAR_ALIASES = {
    "precip": ["precip", "precip_rate", "pr", "rain_rate"],
    "rainc": ["rainc"],
    "rainnc": ["rainnc"],
    "i_rainc": ["i_rainc"],
    "i_rainnc": ["i_rainnc"],
    "olr": ["olr", "olrtoa", "OLR"],
    "pw": ["pw", "precipw", "prw"],
    "cape": ["cape", "CAPE"],
    "cin": ["cin", "CIN"],
    "cldfrac": ["cldfrac", "cloud_fraction", "cl"],
    "theta": ["theta", "potential_temperature"],
    "pressure": ["pressure", "p"],
    "zgrid": ["zgrid"],
    "u_zm": ["uReconstructZonal", "u", "uzonal"],
    "N2": ["N2", "n2", "static_stability"],
    "dudz": ["dudz", "du_dz", "vertical_shear"],
}

needed_vars = sorted(
    set(
        VAR_ALIASES["rainc"]
        + VAR_ALIASES["rainnc"]
        + VAR_ALIASES["i_rainc"]
        + VAR_ALIASES["i_rainnc"]
        + VAR_ALIASES["precip"]
        + VAR_ALIASES["olr"]
        + VAR_ALIASES["pw"]
        + VAR_ALIASES["cape"]
        + VAR_ALIASES["cin"]
        + VAR_ALIASES["cldfrac"]
        + VAR_ALIASES["theta"]
        + VAR_ALIASES["pressure"]
        + VAR_ALIASES["zgrid"]
        + VAR_ALIASES["u_zm"]
        + VAR_ALIASES["N2"]
        + VAR_ALIASES["dudz"]
        + HYDROMETEOR_NEEDED_VARS
    )
)

BUDGET_SPECS = [
    ("high_cloud_mask", "Total high cloud", "0.25"),
    ("active_deep_convection_high_cloud_mask", "Active deep convection", "tab:purple"),
    ("convectively_generated_high_cloud_mask", "Convectively generated high cloud", "tab:orange"),
    ("cirrus_like_high_cloud_mask", "Cirrus-like high cloud", "tab:green"),
    ("unclassified_high_cloud_mask", "Ambiguous/unclassified", "0.55"),
]

RESIDUAL_TARGET_SPECS = [
    ("total", "Total high cloud", "high_cloud_fraction"),
    ("active", "Active deep convection", "active_deep_convection_high_cloud_mask"),
    ("generated", "Convectively generated high cloud", "convectively_generated_high_cloud_mask"),
    ("cirrus", "Cirrus-like high cloud", "cirrus_like_high_cloud_mask"),
]

DISTANCE_CLASS_SPECS = [
    ("active_deep_convection_high_cloud_mask", "Active deep convection", "tab:purple"),
    ("convectively_generated_high_cloud_mask", "Convectively generated high cloud", "tab:orange"),
    ("cirrus_like_high_cloud_mask", "Cirrus-like high cloud", "tab:green"),
]

PRECIP_RATE_COLS = [
    [255 / 255.0, 255 / 255.0, 255 / 255.0],
    [205 / 255.0, 236 / 255.0, 251 / 255.0],
    [157 / 255.0, 218 / 255.0, 247 / 255.0],
    [72 / 255.0, 142 / 255.0, 202 / 255.0],
    [72 / 255.0, 158 / 255.0, 148 / 255.0],
    [73 / 255.0, 181 / 255.0, 70 / 255.0],
    [150 / 255.0, 203 / 255.0, 80 / 255.0],
    [250 / 255.0, 232 / 255.0, 92 / 255.0],
    [248 / 255.0, 184 / 255.0, 73 / 255.0],
    [245 / 255.0, 106 / 255.0, 41 / 255.0],
    [230 / 255.0, 74 / 255.0, 41 / 255.0],
    [211 / 255.0, 31 / 255.0, 40 / 255.0],
    [183 / 255.0, 27 / 255.0, 34 / 255.0],
    [146 / 255.0, 21 / 255.0, 25 / 255.0],
    [110 / 255.0, 15 / 255.0, 15 / 255.0],
]
PRECIP_RATE_LEVELS = np.linspace(0.01, 25.0, len(PRECIP_RATE_COLS))

CLASS_CODE_SPECS = [
    (1, "Ambiguous", "0.70"),
    (2, "Cirrus-like", "tab:green"),
    (3, "Conv. generated", "tab:orange"),
    (4, "Active deep", "tab:purple"),
]


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


def save_figure(number, fig, stem):
    if figsave is None:
        raise RuntimeError("FigureSaver has not been initialised")
    return figsave.save(fig, stem, number=number)


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def save_table(df, stem):
    if save and df is not None and len(df) > 0:
        path = os.path.join(save_loc, f"{stem}.csv")
        df.to_csv(path, index=False)
        progress(f"Wrote {path}")


def write_dataset_with_progress(ds, path, label, encoding=None):
    progress(f"{label}: writing {path}")
    tmp_path = path + ".tmp"
    progress_bar = None
    try:
        from dask.diagnostics import ProgressBar

        progress_bar = ProgressBar()
    except Exception:
        progress_bar = None

    if progress_bar is None:
        ds.to_netcdf(tmp_path, encoding=encoding)
    else:
        with progress_bar:
            ds.to_netcdf(tmp_path, encoding=encoding)
    os.replace(tmp_path, path)
    progress(f"{label}: finished writing {path}")


# -----------------------------------------------------------------------------
# General data and classification helpers
# -----------------------------------------------------------------------------


def find_var(ds, key, required=False):
    for name in VAR_ALIASES.get(key, [key]):
        if name in ds.data_vars:
            return name
    if required:
        raise KeyError(f"Could not find variable for {key!r}; tried {VAR_ALIASES.get(key, [key])}")
    return None


def subset_lat(da, lat_bounds=TROPICAL_LAT_BOUNDS):
    if lat_bounds is None or "latitude" not in da.dims:
        return da
    lo, hi = sorted(lat_bounds)
    return da.sel(latitude=slice(lo, hi))


def latitude_weight_array(da):
    if "latitude" not in da.dims:
        return xr.ones_like(da)
    w = xr.DataArray(
        np.cos(np.deg2rad(da.latitude.values)),
        dims=("latitude",),
        coords={"latitude": da.latitude},
        name="latitude_weight",
    )
    return xr.broadcast(w, da)[0]


def weighted_area_time_mean(da, lat_bounds=TROPICAL_LAT_BOUNDS):
    x = subset_lat(da.astype(float), lat_bounds=lat_bounds)
    dims = [d for d in ("Time", "latitude", "longitude") if d in x.dims]
    if "latitude" not in x.dims:
        return float(x.mean(dim=dims, skipna=True).values)
    weights = np.cos(np.deg2rad(x.latitude))
    return float(x.weighted(weights).mean(dim=dims, skipna=True).values)


def weighted_mean_dims(da, dims):
    dims = [dim for dim in dims if dim in da.dims]
    if not dims:
        return da
    if "latitude" in da.dims and "latitude" in dims:
        weights = np.cos(np.deg2rad(da.latitude))
        return da.weighted(weights).mean(dim=dims, skipna=True)
    return da.mean(dim=dims, skipna=True)


def time_step_days(da):
    if "Time" not in da.dims or da.sizes["Time"] < 2:
        return dt / 24.0
    values = da.Time.values
    if issubclass(values.dtype.type, np.integer):
        return dt / 24.0
    return float(np.nanmedian(da.Time.diff("Time") / np.timedelta64(1, "D")))


def block_slices(da, block_days=BOOTSTRAP_BLOCK_DAYS):
    if "Time" not in da.dims:
        return [slice(None)]
    ntime = int(da.sizes["Time"])
    block_len = max(1, int(round(block_days / time_step_days(da))))
    return [slice(start, min(start + block_len, ntime)) for start in range(0, ntime, block_len)]


def block_area_fraction_values(da, block_days=BOOTSTRAP_BLOCK_DAYS):
    x = subset_lat(da.astype(float), TROPICAL_LAT_BOUNDS)
    values = []
    for slc in block_slices(x, block_days=block_days):
        xb = x.isel(Time=slc) if "Time" in x.dims else x
        values.append(weighted_area_time_mean(xb, lat_bounds=None))
    return np.asarray(values, dtype=float)


def bootstrap_mean_1d(values, rng, n_boot=N_BOOTSTRAP):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan, n_blocks=0)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, values.size, values.size)
        boot[i] = np.nanmean(values[idx])
    ci_low, ci_high = np.nanpercentile(boot, CI_PERCENTILES)
    return dict(mean=float(np.nanmean(values)), ci_low=float(ci_low), ci_high=float(ci_high), n_blocks=int(values.size))


def bootstrap_difference_1d(a, b, rng, n_boot=N_BOOTSTRAP):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan, n_blocks_a=int(a.size), n_blocks_b=int(b.size))
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ai = rng.integers(0, a.size, a.size)
        bi = rng.integers(0, b.size, b.size)
        boot[i] = np.nanmean(a[ai]) - np.nanmean(b[bi])
    ci_low, ci_high = np.nanpercentile(boot, CI_PERCENTILES)
    return dict(
        mean=float(np.nanmean(a) - np.nanmean(b)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_blocks_a=int(a.size),
        n_blocks_b=int(b.size),
    )


def get_precip_rate(ds):
    direct = find_var(ds, "precip")
    if direct is not None:
        pr = finite_or_nan(ds[direct])
        pr.name = "precip"
        return pr

    rainc = find_var(ds, "rainc", required=True)
    rainnc = find_var(ds, "rainnc", required=True)
    i_rainc = find_var(ds, "i_rainc", required=True)
    i_rainnc = find_var(ds, "i_rainnc", required=True)
    accum = (
        ds[rainnc] + ds[i_rainnc] * 100.0
        + ds[rainc] + ds[i_rainc] * 100.0
    )
    pr = precip_rate_from_accum(accum, dt=dt)
    pr.name = "precip"
    pr.attrs["units"] = "mm day-1"
    return finite_or_nan(pr)


def time_aligned_field(ds, key, ref, required=False):
    name = find_var(ds, key, required=required)
    if name is None:
        return None
    out = finite_or_nan(ds[name])
    if "Time" in out.dims and "Time" in ref.dims:
        out = out.sel(Time=ref.Time)
    out.name = key
    return out


def get_cloud_fraction(ds):
    name = find_var(ds, "cldfrac", required=False)
    if name is None:
        warnings.warn("No cloud-fraction variable found; high-cloud diagnostics will be skipped.")
        return None
    cf = finite_or_nan(ds[name])
    zdim = vertical_dim(cf)
    if zdim != "z":
        cf = cf.rename({zdim: "z"})
    if "z" not in cf.coords:
        warnings.warn("Cloud fraction has no usable z coordinate; high-cloud diagnostics will be skipped.")
        return None
    zvals = np.asarray(cf.z.values, dtype=float)
    if zvals.size != cf.sizes["z"] or not np.all(np.isfinite(zvals)):
        warnings.warn("Cloud fraction z coordinate is not usable; high-cloud diagnostics will be skipped.")
        return None
    if np.nanmax(zvals) < 100.0:
        cf = cf.assign_coords(z=cf.z * 1000.0)
    return cf


def available_cloud_fraction_times(cf):
    if "Time" not in cf.dims:
        return None
    indexers = {dim: 0 for dim in cf.dims if dim != "Time"}
    valid = cf.isel(indexers).notnull().load()
    return valid.Time.values[np.asarray(valid.values, dtype=bool)]


def any_cloud_in_layer(cloudy, z_km, zmin, zmax):
    return cloudy.where((z_km >= zmin) & (z_km < zmax), drop=True).any(dim="z")


def rolling_or_dilated_max(mask, lat_radius=0.0, lon_radius=0.0, time_radius=0):
    x = mask.astype(np.int8)
    rolling_kwargs = {}
    if "Time" in x.dims and time_radius > 0:
        rolling_kwargs["Time"] = 2 * int(time_radius) + 1
    if "latitude" in x.dims and lat_radius > 0:
        dlat = float(np.nanmedian(np.abs(np.diff(x.latitude.values))))
        rolling_kwargs["latitude"] = 2 * int(np.ceil(lat_radius / dlat)) + 1
    if "longitude" in x.dims and lon_radius > 0:
        dlon = float(np.nanmedian(np.abs(np.diff(x.longitude.values))))
        rolling_kwargs["longitude"] = 2 * int(np.ceil(lon_radius / dlon)) + 1

    if not rolling_kwargs:
        return mask
    return x.rolling(**rolling_kwargs, center=True, min_periods=1).max() >= 1


def classification_thresholds(cf_thresh=CF_THRESH):
    return HighCloudThresholds(
        cf_thresh=float(cf_thresh),
        z_high_km=Z_HIGH,
        low_cloud_min_km=Z_LOW_MIN,
        low_cloud_max_km=Z_LOW_MAX,
        mid_cloud_min_km=Z_MID_MIN,
        mid_cloud_max_km=Z_MID_MAX,
        upper_cloud_min_km=Z_UPPER_MIN,
        max_clear_gap_km=MAX_CLEAR_GAP,
        deep_min_depth_km=DEEP_MIN_DEPTH_KM,
        surface_precip_deep_mm_day=P_DEEP,
        surface_precip_weak_mm_day=P_WEAK,
        hydrometeor_present_kg_kg=HYDROMETEOR_PRESENT,
        substantial_hydrometeor_kg_kg=SUBSTANTIAL_HYDROMETEOR,
        cirrus_max_depth_km=CIRRUS_MAX_DEPTH_KM,
        convective_generated_max_distance_km=CONVECTIVE_GENERATED_DISTANCE_KM,
        cirrus_min_distance_km=CIRRUS_MIN_DISTANCE_KM,
        require_surface_precip_for_active=True,
    )


def classify_high_cloud(ds, cldfrac, precip, cf_thresh=CF_THRESH):
    """Shared 3-D hydrometeor-aware high-cloud decomposition."""
    return classify_high_cloud_from_dataset(
        ds,
        precip=precip,
        cldfrac=cldfrac,
        thresholds=classification_thresholds(cf_thresh),
        warn_missing=True,
    )


def classification_cache_metadata():
    return shared_classification_cache_metadata(classification_thresholds())


def classification_dataset(classif):
    return shared_classification_dataset(classif, classification_thresholds())


def cached_classification_is_current(cached):
    return shared_cached_classification_is_current(cached, classification_thresholds())


def open_cached_classification(cache_path, sim_name):
    cached = xr.open_dataset(cache_path, chunks={})
    if not cached_classification_is_current(cached):
        progress(f"{sim_name}: cached high-cloud classification is stale; rebuilding {cache_path}")
        cached.close()
        return None
    progress(f"{sim_name}: using cached high-cloud classification {cache_path}")
    return {name: cached[name] for name in cached.data_vars}


def get_or_build_classification(ds, fields, sim_name):
    cache_filename = f"{sim_name.replace(' ', '_')}_classified_high_cloud.nc"
    cache_path = os.path.join(classification_cache_dir, cache_filename)
    if SAVE_CLASSIFICATION_CACHE:
        for read_dir in classification_cache_read_dirs:
            read_path = os.path.join(read_dir, cache_filename)
            if not os.path.exists(read_path):
                continue
            cached = open_cached_classification(read_path, sim_name)
            if cached is not None:
                return cached

    progress(f"{sim_name}: preparing high-cloud classification")
    cf = get_cloud_fraction(ds)
    if cf is None:
        progress(f"{sim_name}: no usable cloud fraction; skipping high-cloud classification")
        return None
    valid_times = available_cloud_fraction_times(cf)
    if valid_times is not None:
        if len(valid_times) == 0:
            warnings.warn(f"{sim_name}: cloud fraction has no valid times after reindexing.")
            return None
        cf = cf.sel(Time=valid_times)
        progress(f"{sim_name}: found {len(valid_times)} valid cloud-fraction times")

    precip = fields["precip"]
    if "Time" in cf.dims and "Time" in precip.dims:
        common_time = cf.Time.to_index().intersection(precip.Time.to_index())
        if len(common_time) == 0:
            warnings.warn(f"{sim_name}: cloud fraction and precipitation have no overlapping times.")
            return None
        cf = cf.sel(Time=common_time)
        precip = precip.sel(Time=common_time)

    progress(f"{sim_name}: building high-cloud classification graph")
    classif = classify_high_cloud(ds, cf, precip)
    progress(f"{sim_name}: high-cloud classification graph built")

    if SAVE_CLASSIFICATION_CACHE:
        fds = classification_dataset(classif)
        encoding = {
            name: {"dtype": "float32"}
            for name in fds.data_vars
            if "fraction" in name or "height" in name
        }
        write_dataset_with_progress(
            fds,
            cache_path,
            label=f"{sim_name}: high-cloud classification cache",
            encoding=encoding,
        )
        cached = open_cached_classification(cache_path, sim_name)
        if cached is not None:
            return cached
    return classif


def compute_ttl_diagnostics(zds, precip_ref):
    """Return optional N2_TTL and dudz_TTL as Time/latitude fields."""
    out = {}
    z_coord = zds["z"] if "z" in zds.coords else None
    if z_coord is None:
        return out

    zvals = np.asarray(z_coord.values, dtype=float)
    zkm = zvals / 1000.0 if np.nanmax(np.abs(zvals)) > 100.0 else zvals
    ttl_mask = xr.DataArray((zkm >= 14.0) & (zkm <= 18.0), dims=("z",), coords={"z": z_coord})

    n2_name = find_var(zds, "N2")
    if n2_name is not None:
        n2 = finite_or_nan(zds[n2_name])
        if vertical_dim(n2) != "z":
            n2 = n2.rename({vertical_dim(n2): "z"})
        out["N2_TTL"] = n2.where(ttl_mask, drop=True).mean(dim="z", skipna=True)
    else:
        theta_name = find_var(zds, "theta")
        if theta_name is not None:
            theta = finite_or_nan(zds[theta_name])
            if vertical_dim(theta) != "z":
                theta = theta.rename({vertical_dim(theta): "z"})
            try:
                dtheta_dz = theta.differentiate("z")
                n2 = 9.80665 / theta * dtheta_dz
                out["N2_TTL"] = n2.where(ttl_mask, drop=True).mean(dim="z", skipna=True)
            except Exception as exc:
                warnings.warn(f"Could not compute N2_TTL from theta: {exc}")

    dudz_name = find_var(zds, "dudz")
    if dudz_name is not None:
        dudz = finite_or_nan(zds[dudz_name])
        if vertical_dim(dudz) != "z":
            dudz = dudz.rename({vertical_dim(dudz): "z"})
        out["dudz_TTL"] = dudz.where(ttl_mask, drop=True).mean(dim="z", skipna=True)
    else:
        u_name = find_var(zds, "u_zm")
        if u_name is not None:
            u = finite_or_nan(zds[u_name])
            if vertical_dim(u) != "z":
                u = u.rename({vertical_dim(u): "z"})
            try:
                dudz = u.differentiate("z")
                out["dudz_TTL"] = dudz.where(ttl_mask, drop=True).mean(dim="z", skipna=True)
            except Exception as exc:
                warnings.warn(f"Could not compute dudz_TTL from uReconstructZonal: {exc}")

    aligned = {}
    for key, da in out.items():
        if "Time" in da.dims:
            da = da.sel(Time=precip_ref.Time)
        da.name = key
        aligned[key] = da
    return aligned


def latitude_field_like(ref):
    lat = xr.DataArray(ref.latitude.values, dims=("latitude",), coords={"latitude": ref.latitude}, name="latitude")
    return xr.broadcast(lat, ref)[0]


def stride_indexers(arrays, max_time=None, max_lat=None, max_lon=None, time_mode="stride"):
    limits = {"Time": max_time, "latitude": max_lat, "longitude": max_lon}
    indexers = {}
    for dim, max_n in limits.items():
        if max_n is None:
            continue
        size = max((da.sizes[dim] for da in arrays if dim in da.dims), default=0)
        if size <= max_n:
            continue
        if dim == "Time" and time_mode == "tail":
            indexers[dim] = slice(size - max_n, None)
        else:
            indexers[dim] = slice(0, None, int(np.ceil(size / max_n)))
    return indexers


def sample_matching_arrays(
    *arrays,
    lat_bounds=TROPICAL_LAT_BOUNDS,
    max_time=96,
    max_lat=96,
    max_lon=192,
    return_weights=True,
    time_mode="stride",
):
    prepared = []
    for da in arrays:
        x = finite_or_nan(da.astype(float)) if hasattr(da, "astype") else da
        x = subset_lat(x, lat_bounds=lat_bounds)
        prepared.append(x)
    prepared = list(xr.align(*prepared, join="inner"))
    indexers = stride_indexers(prepared, max_time=max_time, max_lat=max_lat, max_lon=max_lon, time_mode=time_mode)
    if indexers:
        prepared = [
            da.isel({dim: slc for dim, slc in indexers.items() if dim in da.dims})
            for da in prepared
        ]

    broadcasted = xr.broadcast(*prepared)
    vals = [np.asarray(da.load().values).ravel() for da in broadcasted]
    mask = np.ones(vals[0].shape, dtype=bool)
    for val in vals:
        mask &= np.isfinite(val)

    if return_weights:
        w_da = latitude_weight_array(broadcasted[0])
        weights = np.asarray(w_da.load().values).ravel()
        mask &= np.isfinite(weights) & (weights > 0)
        return [val[mask] for val in vals], weights[mask]
    return [val[mask] for val in vals]


def weighted_quantile(values, quantiles, weights=None):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    ok = np.isfinite(values)
    values = values[ok]
    if weights is None:
        if values.size == 0:
            return np.full_like(quantiles, np.nan, dtype=float)
        return np.nanpercentile(values, quantiles * 100.0)

    weights = np.asarray(weights, dtype=float)[ok]
    ok = weights > 0
    values = values[ok]
    weights = weights[ok]
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) - 0.5 * weights
    cdf /= np.sum(weights)
    return np.interp(quantiles, cdf, values)


def unique_monotonic_bins(edges, fallback=None):
    edges = np.asarray(edges, dtype=float)
    edges = edges[np.isfinite(edges)]
    edges = np.unique(edges)
    if edges.size < 3:
        if fallback is None:
            raise ValueError("Could not construct enough unique bin edges")
        return np.asarray(fallback, dtype=float)
    return edges


def weighted_binned_mean(x, y, bins, weights=None, min_count=MIN_BIN_COUNT):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    x = x[ok]
    y = y[ok]
    weights = weights[ok]
    idx = np.digitize(x, bins) - 1
    centres = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(centres.size, np.nan)
    count = np.zeros(centres.size, dtype=int)
    for j in range(centres.size):
        m = idx == j
        count[j] = int(np.count_nonzero(m))
        if count[j] >= min_count and np.sum(weights[m]) > 0:
            mean[j] = np.sum(weights[m] * y[m]) / np.sum(weights[m])
    return centres, mean, count


def class_overlap_sanity(classifications):
    rows = []
    exclusive_keys = [
        "active_deep_convection_high_cloud_mask",
        "convectively_generated_high_cloud_mask",
        "cirrus_like_high_cloud_mask",
        "unclassified_high_cloud_mask",
    ]
    overlay_keys = [
        "column_connected_mask",
        "nonprecipitating_high_cloud_mask",
    ]
    for sim_name, classif in classifications.items():
        if classif is None:
            continue
        high = classif["high_cloud_mask"].astype(bool)
        high_frac = weighted_area_time_mean(high, TROPICAL_LAT_BOUNDS)
        rows.append(
            dict(
                simulation=sim_name,
                metric="high_cloud_denominator",
                value=high_frac,
                denominator="all tropical grid-cell/time samples",
                note="All class fractions in budget use this domain denominator unless marked within_high_cloud.",
            )
        )
        for key in exclusive_keys + overlay_keys:
            mask = classif[key].astype(bool)
            rows.append(
                dict(
                    simulation=sim_name,
                    metric=f"{key}_domain_fraction",
                    value=weighted_area_time_mean(mask, TROPICAL_LAT_BOUNDS),
                    denominator="all tropical grid-cell/time samples",
                    note="domain occurrence",
                )
            )
            rows.append(
                dict(
                    simulation=sim_name,
                    metric=f"{key}_within_high_cloud",
                    value=weighted_area_time_mean(mask & high, TROPICAL_LAT_BOUNDS) / high_frac if high_frac > 0 else np.nan,
                    denominator="total high-cloud tropical area-time fraction",
                    note="within high-cloud denominator",
                )
            )

        for i, key_a in enumerate(exclusive_keys):
            for key_b in exclusive_keys[i + 1:]:
                overlap = weighted_area_time_mean(
                    classif[key_a].astype(bool) & classif[key_b].astype(bool),
                    TROPICAL_LAT_BOUNDS,
                )
                rows.append(
                    dict(
                        simulation=sim_name,
                        metric=f"overlap_{key_a}__{key_b}",
                        value=overlap,
                        denominator="all tropical grid-cell/time samples",
                        note="science classes and unclassified mask should be mutually exclusive",
                    )
                )

        for key_a in overlay_keys:
            for key_b in exclusive_keys:
                overlap = weighted_area_time_mean(
                    classif[key_a].astype(bool) & classif[key_b].astype(bool),
                    TROPICAL_LAT_BOUNDS,
                )
                rows.append(
                    dict(
                        simulation=sim_name,
                        metric=f"overlap_{key_a}__{key_b}",
                        value=overlap,
                        denominator="all tropical grid-cell/time samples",
                        note="column-connected and nonprecipitating are diagnostic overlays, not exclusive partitions",
                    )
                )

    df = pd.DataFrame(rows)
    save_table(df, "high_cloud_classification_overlap_denominators")
    if len(df) > 0:
        print("\nHigh-cloud class denominator/overlap checks")
        print(df.to_string(index=False, max_rows=240), flush=True)
    return df


# -----------------------------------------------------------------------------
# Figure 1. Class budget
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * The robust signal should appear as more total high cloud in the nudged
#     runs, with the difference carried mainly by active deep convection,
#     convectively generated high cloud, or cirrus-like high cloud.
#   * Ambiguous/unclassified high cloud should stay small; if not, the class
#     thresholds need physical inspection before interpreting differences.
#   * Error bars are block-resampled time-chunk consistency intervals, not
#     formal observational confidence intervals.


def compute_budget_table(classifications):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    block_values = {}
    control_name = sim_names[0]

    for sim_name in sim_names:
        classif = classifications.get(sim_name)
        if classif is None:
            continue
        for key, label, _color in BUDGET_SPECS:
            vals = block_area_fraction_values(classif[key])
            block_values[(sim_name, key)] = vals
            stats = bootstrap_mean_1d(vals, rng)
            rows.append(
                dict(
                    simulation=sim_name,
                    class_key=key,
                    class_label=label,
                    statistic="absolute_fraction",
                    value=weighted_area_time_mean(classif[key], TROPICAL_LAT_BOUNDS),
                    block_mean=stats["mean"],
                    ci_low=stats["ci_low"],
                    ci_high=stats["ci_high"],
                    n_blocks=stats["n_blocks"],
                    denominator="all tropical grid-cell/time samples",
                )
            )

    for sim_name in sim_names[1:]:
        for key, label, _color in BUDGET_SPECS:
            if (sim_name, key) not in block_values or (control_name, key) not in block_values:
                continue
            stats = bootstrap_difference_1d(block_values[(sim_name, key)], block_values[(control_name, key)], rng)
            rows.append(
                dict(
                    simulation=sim_name,
                    class_key=key,
                    class_label=label,
                    statistic="difference_from_control",
                    value=stats["mean"],
                    block_mean=stats["mean"],
                    ci_low=stats["ci_low"],
                    ci_high=stats["ci_high"],
                    n_blocks=min(stats["n_blocks_a"], stats["n_blocks_b"]),
                    denominator="all tropical grid-cell/time samples",
                )
            )

    df = pd.DataFrame(rows)
    save_table(df, "high_cloud_class_budget")
    return df


def draw_errorbarh(ax, y, value, lo, hi, color):
    if not np.isfinite(value) or not np.isfinite(lo) or not np.isfinite(hi):
        return
    ax.errorbar(
        value,
        y,
        xerr=np.array([[value - lo], [hi - value]]),
        color=color,
        fmt="none",
        lw=1.1,
        capsize=2,
        zorder=4,
    )


def plot_class_budget(classifications):
    if not should_run_figure(1):
        return
    progress("Starting Figure 1: high_cloud_class_budget")
    budget = compute_budget_table(classifications)
    if len(budget) == 0:
        print("Skipping Figure 1: high-cloud classifications are unavailable.")
        return

    labels = [label for _key, label, _color in BUDGET_SPECS]
    keys = [key for key, _label, _color in BUDGET_SPECS]
    y = np.arange(len(keys))
    height = 0.22

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 5.3), constrained_layout=True, sharey=True)
    abs_df = budget[budget.statistic == "absolute_fraction"].set_index(["simulation", "class_key"])
    diff_df = budget[budget.statistic == "difference_from_control"].set_index(["simulation", "class_key"])

    for i, sim_name in enumerate(sim_names):
        yy = y + (i - (len(sim_names) - 1) / 2.0) * height
        vals = [
            abs_df.loc[(sim_name, key), "value"] if (sim_name, key) in abs_df.index else np.nan
            for key in keys
        ]
        ax[0].barh(yy, vals, height=height, color=cols[i], alpha=0.82, label=sim_name)
        for j, key in enumerate(keys):
            if (sim_name, key) in abs_df.index:
                row = abs_df.loc[(sim_name, key)]
                draw_errorbarh(ax[0], yy[j], row["value"], row["ci_low"], row["ci_high"], color="0.15")

    for i, sim_name in enumerate(sim_names[1:], start=1):
        yy = y + (i - 1.5) * height
        vals = [
            diff_df.loc[(sim_name, key), "value"] if (sim_name, key) in diff_df.index else np.nan
            for key in keys
        ]
        ax[1].barh(yy, vals, height=height, color=cols[i], alpha=0.82, label=f"{sim_name} - control")
        for j, key in enumerate(keys):
            if (sim_name, key) in diff_df.index:
                row = diff_df.loc[(sim_name, key)]
                draw_errorbarh(ax[1], yy[j], row["value"], row["ci_low"], row["ci_high"], color="0.15")

    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Tropical area-time fraction")
    ax[0].set_title("Absolute coverage")
    ax[0].grid(axis="x", alpha=0.22)
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].axvline(0.0, color="0.45", lw=0.8)
    ax[1].set_xlabel("Difference from control")
    ax[1].set_title("Nudged response")
    ax[1].grid(axis="x", alpha=0.22)
    ax[1].legend(frameon=False, fontsize=8)

    fig.suptitle("High-cloud class budget: domain denominator is tropical area-time")
    save_figure(1, fig, "high_cloud_class_budget")
    plt.close(fig)
    progress("Done with figure: high_cloud_class_budget")


# -----------------------------------------------------------------------------
# Figure 2. Residual high-cloud signal
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Positive nudged residuals mean more high cloud than expected from the
#     control relationship with local precipitation, moisture, instability, and
#     latitude.
#   * The control model is evaluated only inside the robust control feature
#     support; unsupported extrapolated samples are masked.
#   * N2_TTL is not split into terciles here.  If available, one common-support
#     residual summary is written to CSV.


class SimpleRidgeRegressor:
    """Small weighted ridge fallback with linear, squared, and pair terms."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.mean_ = None
        self.scale_ = None
        self.beta_ = None

    def _design(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        if fit:
            self.mean_ = np.nanmean(X, axis=0)
            self.scale_ = np.nanstd(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
        Xn = (X - self.mean_) / self.scale_
        cols = [np.ones(Xn.shape[0]), Xn, Xn ** 2]
        pair_cols = []
        for i in range(Xn.shape[1]):
            for j in range(i + 1, Xn.shape[1]):
                pair_cols.append(Xn[:, i] * Xn[:, j])
        if pair_cols:
            cols.append(np.column_stack(pair_cols))
        return np.column_stack(cols)

    def fit(self, X, y, sample_weight=None):
        D = self._design(X, fit=True)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        sw = np.sqrt(np.maximum(w, 0.0))
        Dw = D * sw[:, None]
        yw = y * sw
        penalty = self.alpha * np.eye(D.shape[1])
        penalty[0, 0] = 0.0
        self.beta_ = np.linalg.solve(Dw.T @ Dw + penalty, Dw.T @ yw)
        return self

    def predict(self, X):
        return self._design(X, fit=False) @ self.beta_


def fit_control_relationship(X, y, weights, target_label):
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    X = X[ok]
    y = y[ok]
    weights = weights[ok]
    progress(f"Fitting control high-cloud model for {target_label}: {X.shape[0]} samples")
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            min_samples_leaf=40,
            random_state=0,
        )
        model.fit(X, y, sample_weight=weights)
        return model
    except Exception as exc:
        progress(f"Falling back to weighted ridge for {target_label}: {exc}")
        model = SimpleRidgeRegressor(alpha=1.0)
        return model.fit(X, y, sample_weight=weights)


def available_residual_features(fields_by_run):
    feature_keys = ["precip_log"]
    for key in ("pw", "cape", "cin"):
        if all(key in fields for fields in fields_by_run):
            feature_keys.append(key)
        else:
            warnings.warn(f"Residual model omits {key.upper()} because it is unavailable in at least one run.")
    feature_keys.append("latitude")
    return feature_keys


def feature_arrays(fields, feature_keys):
    arrays = []
    for key in feature_keys:
        if key == "latitude":
            arrays.append(latitude_field_like(fields["precip"]))
        elif key == "precip_log":
            arrays.append(np.log1p(fields["precip"].clip(min=0.0)))
        else:
            arrays.append(fields[key])
    return arrays


def residual_target_da(classif, target_key):
    if target_key == "total":
        return classif["high_cloud_fraction"].clip(min=0.0, max=1.0)
    if target_key == "active":
        return classif["active_deep_convection_high_cloud_mask"].astype(float)
    if target_key == "generated":
        return classif["convectively_generated_high_cloud_mask"].astype(float)
    if target_key == "cirrus":
        return classif["cirrus_like_high_cloud_mask"].astype(float)
    raise ValueError(target_key)


def model_training_data(fields, classif, target_key, feature_keys):
    arrays = feature_arrays(fields, feature_keys) + [residual_target_da(classif, target_key)]
    vals, weights = sample_matching_arrays(
        *arrays,
        lat_bounds=TROPICAL_LAT_BOUNDS,
        **RESIDUAL_FIT_SAMPLE_LIMITS,
    )
    X = np.column_stack(vals[:-1])
    y = vals[-1]
    return X, y, weights


def feature_support_ranges(X, weights, feature_keys):
    ranges = {}
    for j, key in enumerate(feature_keys):
        lo, hi = weighted_quantile(X[:, j], [0.01, 0.99], weights)
        ranges[key] = (float(lo), float(hi))
    return ranges


def inside_feature_support(X, support_ranges, feature_keys):
    ok = np.ones(X.shape[0], dtype=bool)
    for j, key in enumerate(feature_keys):
        lo, hi = support_ranges[key]
        ok &= np.isfinite(X[:, j]) & (X[:, j] >= lo) & (X[:, j] <= hi)
    return ok


def prepare_residual_grid(fields, classif, target_key, feature_keys, model, support_ranges):
    target = residual_target_da(classif, target_key)
    arrays = feature_arrays(fields, feature_keys) + [target]
    n2_available = "N2_TTL" in fields
    if n2_available:
        arrays.append(fields["N2_TTL"])

    prepared = [subset_lat(finite_or_nan(da.astype(float)), TROPICAL_LAT_BOUNDS) for da in arrays]
    prepared = list(xr.align(*prepared, join="inner"))
    indexers = stride_indexers(prepared, **RESIDUAL_PLOT_SAMPLE_LIMITS, time_mode="stride")
    if indexers:
        prepared = [
            da.isel({dim: slc for dim, slc in indexers.items() if dim in da.dims})
            for da in prepared
        ]
    broadcasted = xr.broadcast(*prepared)

    dims = [dim for dim in ("Time", "latitude", "longitude") if dim in broadcasted[-1].dims]
    broadcasted = [da.transpose(*dims) for da in broadcasted]
    vals = [np.asarray(da.load().values, dtype=float) for da in broadcasted]
    shape = vals[0].shape
    flat = [v.ravel() for v in vals]

    nfeat = len(feature_keys)
    X = np.column_stack(flat[:nfeat])
    y = flat[nfeat]
    finite = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    supported = finite & inside_feature_support(X, support_ranges, feature_keys)

    pred = np.full(y.shape, np.nan, dtype=float)
    if np.any(supported):
        pred[supported] = np.clip(model.predict(X[supported]), 0.0, 1.0)
    residual = np.full(y.shape, np.nan, dtype=float)
    residual[supported] = y[supported] - pred[supported]

    template = broadcasted[nfeat]
    residual_da = xr.DataArray(residual.reshape(shape), dims=dims, coords={dim: template[dim] for dim in dims}, name="residual")
    support_da = xr.DataArray(supported.reshape(shape), dims=dims, coords=residual_da.coords, name="support_mask")
    n2_da = None
    if n2_available:
        n2_values = flat[nfeat + 1].reshape(shape)
        n2_da = xr.DataArray(n2_values, dims=dims, coords=residual_da.coords, name="N2_TTL")
    return residual_da, support_da, n2_da


def n2_common_support(fields_by_run, classifications):
    if not all("N2_TTL" in fields for fields in fields_by_run):
        return None
    per_run = []
    for sim_name, fields in zip(sim_names, fields_by_run):
        classif = classifications.get(sim_name)
        if classif is None:
            continue
        vals, weights = sample_matching_arrays(
            fields["N2_TTL"],
            classif["high_cloud_mask"].astype(float),
            lat_bounds=TROPICAL_LAT_BOUNDS,
            **RESIDUAL_FIT_SAMPLE_LIMITS,
        )
        q02, q98 = weighted_quantile(vals[0], [0.02, 0.98], weights)
        per_run.append((sim_name, float(q02), float(q98)))
    if len(per_run) != len(sim_names):
        return None
    common_lo = max(row[1] for row in per_run)
    common_hi = min(row[2] for row in per_run)
    if not np.isfinite(common_lo) or not np.isfinite(common_hi) or common_hi <= common_lo:
        return dict(common_lo=np.nan, common_hi=np.nan, per_run=per_run)
    return dict(common_lo=common_lo, common_hi=common_hi, per_run=per_run)


def residual_summary_rows(residual_data, support_data, n2_data, n2_support):
    rows = []
    for target_key, target_label, _source in RESIDUAL_TARGET_SPECS:
        for sim_name in sim_names:
            residual = residual_data[(target_key, sim_name)]
            support = support_data[(target_key, sim_name)]
            weights = latitude_weight_array(residual)
            valid = np.isfinite(residual)
            weighted_mean = residual.where(valid).weighted(weights).mean(
                dim=[d for d in ("Time", "latitude", "longitude") if d in residual.dims],
                skipna=True,
            )
            rows.append(
                dict(
                    simulation=sim_name,
                    target=target_label,
                    support_subset="control_feature_support",
                    mean_residual=float(weighted_mean.values),
                    supported_fraction=float(support.astype(float).mean().values),
                    denominator="sampled tropical grid-cell/time samples inside control feature support",
                    N2_TTL_common_lo=np.nan,
                    N2_TTL_common_hi=np.nan,
                )
            )
            n2 = n2_data.get((target_key, sim_name))
            if n2_support is not None and n2 is not None and np.isfinite(n2_support.get("common_lo", np.nan)):
                in_common = valid & (n2 >= n2_support["common_lo"]) & (n2 <= n2_support["common_hi"])
                common_mean = residual.where(in_common).weighted(weights).mean(
                    dim=[d for d in ("Time", "latitude", "longitude") if d in residual.dims],
                    skipna=True,
                )
                rows.append(
                    dict(
                        simulation=sim_name,
                        target=target_label,
                        support_subset="control_feature_and_N2_TTL_common_support",
                        mean_residual=float(common_mean.values),
                        supported_fraction=float(in_common.astype(float).mean().values),
                        denominator="sampled tropical grid-cell/time samples inside control feature support and one common N2_TTL range",
                        N2_TTL_common_lo=float(n2_support["common_lo"]),
                        N2_TTL_common_hi=float(n2_support["common_hi"]),
                    )
                )
    return rows


def plot_residual_diagnostics(fields_by_run, classifications):
    if not should_run_figure(2):
        return
    progress("Starting Figure 2: residual_high_cloud_maps_zonal")
    if any(classifications.get(sim_name) is None for sim_name in sim_names):
        print("Skipping Figure 2: high-cloud classification is unavailable for at least one run.")
        return

    feature_keys = available_residual_features(fields_by_run)
    residual_data = {}
    support_data = {}
    n2_data = {}
    models = {}
    support_ranges_by_target = {}

    for target_key, target_label, _source in RESIDUAL_TARGET_SPECS:
        Xc, yc, wc = model_training_data(fields_by_run[0], classifications[sim_names[0]], target_key, feature_keys)
        support_ranges = feature_support_ranges(Xc, wc, feature_keys)
        model = fit_control_relationship(Xc, yc, wc, target_label)
        models[target_key] = model
        support_ranges_by_target[target_key] = support_ranges
        for i, sim_name in enumerate(sim_names):
            residual, support, n2 = prepare_residual_grid(
                fields_by_run[i],
                classifications[sim_name],
                target_key,
                feature_keys,
                model,
                support_ranges,
            )
            residual_data[(target_key, sim_name)] = residual
            support_data[(target_key, sim_name)] = support
            if n2 is not None:
                n2_data[(target_key, sim_name)] = n2

    support_rows = []
    for target_key, ranges in support_ranges_by_target.items():
        for key, (lo, hi) in ranges.items():
            support_rows.append(dict(target=target_key, feature=key, control_q01=lo, control_q99=hi))
    save_table(pd.DataFrame(support_rows), "residual_control_feature_support")

    n2_support = n2_common_support(fields_by_run, classifications)
    if n2_support is not None:
        n2_rows = [
            dict(simulation=sim_name, N2_TTL_q02=q02, N2_TTL_q98=q98)
            for sim_name, q02, q98 in n2_support["per_run"]
        ]
        for row in n2_rows:
            row["N2_TTL_common_lo"] = n2_support.get("common_lo", np.nan)
            row["N2_TTL_common_hi"] = n2_support.get("common_hi", np.nan)
        save_table(pd.DataFrame(n2_rows), "N2_TTL_common_support_summary")

    summary = pd.DataFrame(residual_summary_rows(residual_data, support_data, n2_data, n2_support))
    save_table(summary, "residual_high_cloud_summary")

    nrows = len(RESIDUAL_TARGET_SPECS)
    ncols = len(sim_names) + 1
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(3.35 * ncols, 2.65 * nrows),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    levels = symmetric_levels(0.12, ndiv=15)
    last_im = None

    for r, (target_key, target_label, _source) in enumerate(RESIDUAL_TARGET_SPECS):
        row_maps = []
        for sim_name in sim_names:
            residual = residual_data[(target_key, sim_name)]
            m = residual.mean(dim="Time", skipna=True) if "Time" in residual.dims else residual
            row_maps.append(m)
        row_abs = np.nanpercentile(np.abs(np.concatenate([np.ravel(m.values) for m in row_maps])), 98)
        row_abs = float(row_abs) if np.isfinite(row_abs) and row_abs > 0 else 0.05
        row_levels = symmetric_levels(max(0.03, row_abs), ndiv=15)

        for c, sim_name in enumerate(sim_names):
            a = ax[r, c]
            m = row_maps[c]
            last_im = a.contourf(
                m.longitude,
                m.latitude,
                m,
                levels=row_levels,
                cmap=difference_cmap(row_levels, cmap="RdBu_r"),
                extend="both",
            )
            if r == 0:
                a.set_title(sim_name)
            if c == 0:
                a.set_ylabel(f"{target_label}\nLatitude")
            if r == nrows - 1:
                a.set_xlabel("Longitude")
            a.set_xlim(-180, 180)
            a.set_ylim(*TROPICAL_LAT_BOUNDS)

        a = ax[r, -1]
        for i, sim_name in enumerate(sim_names):
            residual = residual_data[(target_key, sim_name)]
            dims = [d for d in ("Time", "longitude") if d in residual.dims]
            z = residual.mean(dim=dims, skipna=True)
            a.plot(z.latitude, z, color=cols[i], lw=1.5, label=sim_name if r == 0 else None)
        a.axhline(0.0, color="0.55", lw=0.8)
        a.grid(alpha=0.22)
        if r == 0:
            a.set_title("Zonal residual")
        if r == nrows - 1:
            a.set_xlabel("Latitude")
        a.set_xlim(*TROPICAL_LAT_BOUNDS)
        a.set_ylabel("Obs - control-pred.")

    if last_im is not None:
        fig.colorbar(last_im, ax=ax[:, :-1].ravel().tolist(), label="Residual high-cloud fraction/probability")
    ax[0, -1].legend(frameon=False, fontsize=8)
    subtitle = "masked outside control q01-q99 feature support"
    if n2_support is not None and np.isfinite(n2_support.get("common_lo", np.nan)):
        subtitle += f"; N2_TTL common-support summary: {n2_support['common_lo']:.3g} to {n2_support['common_hi']:.3g}"
    fig.suptitle(f"Observed minus control-predicted high cloud ({subtitle})")
    save_figure(2, fig, "residual_high_cloud_maps_zonal")
    plt.close(fig)
    progress("Done with figure: residual_high_cloud_maps_zonal")


# -----------------------------------------------------------------------------
# Figure 3. Classification validation snapshots
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * The rain panel shows where active precipitating convection sits.
#   * The class panel shows the parent high-cloud mask and the mutually
#     exclusive diagnosed classes. Ambiguous means high cloud above 12 km that
#     did not pass the active, convectively generated, or cirrus-like tests.
#   * The two vertical slices show cloud fraction through height.  The coloured
#     shading between 12 km and the plot top shows the diagnosed column class,
#     so cirrus-like cloud can be checked for shallow or separated structure.


def format_time_value(value):
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.datetime64):
        return np.datetime_as_string(arr, unit="h")
    return str(value)


def representative_validation_time(classif):
    high = subset_lat(classif["high_cloud_mask"].astype(float), TROPICAL_LAT_BOUNDS)
    cirrus = subset_lat(classif["cirrus_like_high_cloud_mask"].astype(float), TROPICAL_LAT_BOUNDS)
    generated = subset_lat(classif["convectively_generated_high_cloud_mask"].astype(float), TROPICAL_LAT_BOUNDS)
    active = subset_lat(classif["active_deep_convection_high_cloud_mask"].astype(float), TROPICAL_LAT_BOUNDS)
    if "Time" not in high.dims:
        return None
    weights = np.cos(np.deg2rad(high.latitude))
    score = (high + 2.0 * cirrus + 1.25 * generated + 1.50 * active).weighted(weights).mean(
        dim=("latitude", "longitude"),
        skipna=True,
    )
    idx = int(score.argmax(dim="Time").values)
    return high.Time.isel(Time=idx).values


def contour_if_present(ax, lon, lat, mask, color, lw=1.2, linestyle="-"):
    values = np.asarray(mask.values if hasattr(mask, "values") else mask, dtype=float)
    if not np.any(values >= 0.5):
        return None
    return ax.contour(lon, lat, values, levels=[0.5], colors=[color], linewidths=lw, linestyles=linestyle)


def precip_rate_cmap_norm():
    cmap = colors.LinearSegmentedColormap.from_list(
        name="project_precip_rate",
        colors=PRECIP_RATE_COLS,
        N=len(PRECIP_RATE_COLS),
    )
    norm = colors.BoundaryNorm(boundaries=PRECIP_RATE_LEVELS, ncolors=len(PRECIP_RATE_COLS) - 1)
    return cmap, norm


def class_code_cmap_norm(alpha=0.82):
    rgba = [(0.0, 0.0, 0.0, 0.0)]
    for _code, _label, color in CLASS_CODE_SPECS:
        rgba.append(colors.to_rgba(color, alpha=alpha))
    cmap = colors.ListedColormap(rgba, name="high_cloud_class_codes")
    norm = colors.BoundaryNorm(np.arange(-0.5, len(CLASS_CODE_SPECS) + 1.5, 1.0), cmap.N)
    return cmap, norm


def class_code_from_masks(high, active, generated, cirrus, unclassified):
    code = np.zeros(high.shape, dtype=np.int16)
    code[np.asarray(high.values) >= 0.5] = 1
    code[np.asarray(unclassified.values) >= 0.5] = 1
    code[np.asarray(cirrus.values) >= 0.5] = 2
    code[np.asarray(generated.values) >= 0.5] = 3
    code[np.asarray(active.values) >= 0.5] = 4
    return xr.DataArray(code, dims=high.dims, coords=high.coords, name="high_cloud_class_code")


def periodic_lon_delta(lon, center_lon):
    return ((np.asarray(lon, dtype=float) - float(center_lon) + 180.0) % 360.0) - 180.0


def wrap_longitude_for_center(da, center_lon):
    if "longitude" not in da.dims:
        return da
    lon = np.asarray(da.longitude.values, dtype=float)
    shifted = periodic_lon_delta(lon, center_lon) + float(center_lon)
    order = np.argsort(shifted)
    return da.assign_coords(longitude=("longitude", shifted)).isel(longitude=order)


def crop_longitude_window(da, center_lon, width_deg=SNAPSHOT_LON_WINDOW_DEG):
    x = wrap_longitude_for_center(da, center_lon)
    half = 0.5 * float(width_deg)
    return x.sel(longitude=slice(float(center_lon) - half, float(center_lon) + half))


def centres_to_edges(x):
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges


def select_time_value(da, time_value):
    try:
        return da.sel(Time=time_value)
    except Exception:
        return da.sel(Time=time_value, method="nearest")


def select_nearest_latitude(da, lat_value):
    lat = np.asarray(da.latitude.values, dtype=float)
    idx = int(np.nanargmin(np.abs(lat - float(lat_value))))
    return da.isel(latitude=idx), float(lat[idx])


def select_nearest_longitude(da, lon_value):
    lon = np.asarray(da.longitude.values, dtype=float)
    idx = int(np.nanargmin(np.abs(periodic_lon_delta(lon, lon_value))))
    return da.isel(longitude=idx), float(lon[idx])


def z_km_dataarray(cf):
    zvals = np.asarray(cf.z.values, dtype=float)
    zkm = zvals / 1000.0 if np.nanmax(np.abs(zvals)) > 100.0 else zvals
    return xr.DataArray(zkm, dims=("z",), coords={"z": cf.z}, name="height_km")


def focus_point_for_snapshot(precip, high, active, generated, cirrus):
    rain_object = (precip >= P_DEEP).astype(float)
    score = (
        0.20 * high.astype(float)
        + 1.50 * active.astype(float)
        + 1.25 * rain_object
        + 1.50 * generated.astype(float)
        + 1.75 * cirrus.astype(float)
    )
    arr = np.asarray(score.fillna(0.0).values, dtype=float)
    if not np.any(arr > 0):
        return 0.0, 0.0
    smooth = ndimage.uniform_filter(arr, size=(7, 17), mode=("nearest", "wrap"))
    iy, ix = np.unravel_index(int(np.nanargmax(smooth)), smooth.shape)
    return float(score.latitude.values[iy]), float(score.longitude.values[ix])


def add_class_band(ax, x, code_values, y0=Z_HIGH, y1=SNAPSHOT_Z_MAX_KM):
    x = np.asarray(x, dtype=float)
    codes = np.asarray(code_values, dtype=float)
    if x.size == 0 or not np.any(codes > 0):
        return
    cmap, norm = class_code_cmap_norm(alpha=0.45)
    masked = np.ma.masked_where(codes <= 0, codes)
    ax.pcolormesh(
        centres_to_edges(x),
        np.array([y0, y1]),
        masked[np.newaxis, :],
        cmap=cmap,
        norm=norm,
        shading="flat",
        zorder=3,
    )


def add_rain_ticks(ax, x, precip_values):
    x = np.asarray(x, dtype=float)
    rain = np.asarray(precip_values, dtype=float) >= P_DEEP
    if not np.any(rain):
        return
    ax.scatter(
        x[rain],
        np.full(np.count_nonzero(rain), 0.35),
        marker="|",
        color="k",
        s=18,
        linewidths=1.0,
        zorder=5,
        label="P >= threshold",
    )


def plot_precip_map(ax, precip, focus_lon, focus_lat):
    cmap, norm = precip_rate_cmap_norm()
    p = crop_longitude_window(precip, focus_lon)
    im = ax.pcolormesh(
        p.longitude,
        p.latitude,
        p.clip(min=0.0),
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    contour_if_present(ax, p.longitude, p.latitude, (p >= P_DEEP).astype(float), "k", lw=0.9)
    ax.axvline(focus_lon, color="0.1", lw=1.0, ls=":")
    ax.axhline(focus_lat, color="0.1", lw=1.0, ls=":")
    ax.set_xlim(float(focus_lon) - 0.5 * SNAPSHOT_LON_WINDOW_DEG, float(focus_lon) + 0.5 * SNAPSHOT_LON_WINDOW_DEG)
    ax.set_ylim(*SNAPSHOT_LAT_BOUNDS)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Rain rate; black line P >= {P_DEEP:g}")
    return im


def plot_class_map(ax, class_code, focus_lon, focus_lat):
    cmap, norm = class_code_cmap_norm(alpha=0.88)
    cls = crop_longitude_window(class_code, focus_lon)
    values = np.ma.masked_where(np.asarray(cls.values) <= 0, np.asarray(cls.values))
    ax.set_facecolor("0.96")
    im = ax.pcolormesh(
        cls.longitude,
        cls.latitude,
        values,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.axvline(focus_lon, color="0.1", lw=1.0, ls=":")
    ax.axhline(focus_lat, color="0.1", lw=1.0, ls=":")
    ax.set_xlim(float(focus_lon) - 0.5 * SNAPSHOT_LON_WINDOW_DEG, float(focus_lon) + 0.5 * SNAPSHOT_LON_WINDOW_DEG)
    ax.set_ylim(*SNAPSHOT_LAT_BOUNDS)
    ax.set_xlabel("Longitude")
    ax.tick_params(labelleft=False)
    ax.set_title("Class mask")
    return im


def plot_cloud_slice(ax, cf_snapshot, class_code, precip, focus_lat, focus_lon, orientation):
    if orientation == "longitude":
        cf_line, actual_lat = select_nearest_latitude(cf_snapshot, focus_lat)
        class_line, _ = select_nearest_latitude(class_code, actual_lat)
        precip_line, _ = select_nearest_latitude(precip, actual_lat)
        cf_line = crop_longitude_window(cf_line, focus_lon)
        class_line = crop_longitude_window(class_line, focus_lon)
        precip_line = crop_longitude_window(precip_line, focus_lon)
        x = np.asarray(cf_line.longitude.values, dtype=float)
        xlabel = "Longitude"
        title = f"Cloud slice at {actual_lat:.1f} deg lat"
        horizontal_dim = "longitude"
    else:
        cf_line, actual_lon = select_nearest_longitude(cf_snapshot, focus_lon)
        class_line, _ = select_nearest_longitude(class_code, actual_lon)
        precip_line, _ = select_nearest_longitude(precip, actual_lon)
        cf_line = subset_lat(cf_line, SNAPSHOT_LAT_BOUNDS)
        class_line = subset_lat(class_line, SNAPSHOT_LAT_BOUNDS)
        precip_line = subset_lat(precip_line, SNAPSHOT_LAT_BOUNDS)
        x = np.asarray(cf_line.latitude.values, dtype=float)
        xlabel = "Latitude"
        title = f"Cloud slice at {actual_lon:.1f} deg lon"
        horizontal_dim = "latitude"

    z_km = z_km_dataarray(cf_line)
    cf_line = cf_line.where(z_km <= SNAPSHOT_Z_MAX_KM, drop=True)
    z_vals = np.asarray(z_km.sel(z=cf_line.z).values, dtype=float)
    field = cf_line.transpose("z", horizontal_dim).values
    im = ax.pcolormesh(
        x,
        z_vals,
        field,
        shading="auto",
        cmap="Greys",
        vmin=0.0,
        vmax=0.6,
    )
    add_class_band(ax, x, class_line.values)
    add_rain_ticks(ax, x, precip_line.values)
    for zline, lw, ls in [(2.0, 0.55, ":"), (5.0, 0.55, ":"), (10.0, 0.55, ":"), (12.0, 0.85, "--")]:
        ax.axhline(zline, color="0.45", lw=lw, ls=ls, zorder=4)
    ax.set_ylim(0.0, SNAPSHOT_Z_MAX_KM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Height (km)")
    ax.set_title(title)
    ax.grid(alpha=0.15, lw=0.4)
    return im


def plot_validation_snapshots(dss, fields_by_run, classifications):
    if not should_run_figure(3):
        return
    progress("Starting Figure 3: cloud_classification_validation_snapshots")
    if not any(classifications.get(sim_name) is not None for sim_name in sim_names):
        print("Skipping Figure 3: high-cloud classification is unavailable.")
        return

    fig, axes = plt.subplots(
        len(sim_names),
        4,
        figsize=(18.0, 3.35 * len(sim_names)),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    if len(sim_names) == 1:
        axes = axes[None, :]

    rows = []
    last_precip_im = None
    last_class_im = None
    last_cf_im = None
    for i, sim_name in enumerate(sim_names):
        ax_rain, ax_class, ax_lon, ax_lat = axes[i]
        classif = classifications.get(sim_name)
        if classif is None:
            for ax in axes[i]:
                ax.text(0.5, 0.5, "classification unavailable", transform=ax.transAxes, ha="center", va="center")
            continue
        snap_time = representative_validation_time(classif)
        if snap_time is None:
            for ax in axes[i]:
                ax.text(0.5, 0.5, "no Time dimension", transform=ax.transAxes, ha="center", va="center")
            continue

        precip = subset_lat(fields_by_run[i]["precip"].sel(Time=snap_time), SNAPSHOT_LAT_BOUNDS).load()
        high = subset_lat(classif["high_cloud_mask"].sel(Time=snap_time).astype(float), SNAPSHOT_LAT_BOUNDS).load()
        active = subset_lat(classif["active_deep_convection_high_cloud_mask"].sel(Time=snap_time).astype(float), SNAPSHOT_LAT_BOUNDS).load()
        generated = subset_lat(classif["convectively_generated_high_cloud_mask"].sel(Time=snap_time).astype(float), SNAPSHOT_LAT_BOUNDS).load()
        cirrus = subset_lat(classif["cirrus_like_high_cloud_mask"].sel(Time=snap_time).astype(float), SNAPSHOT_LAT_BOUNDS).load()
        unclassified = subset_lat(classif["unclassified_high_cloud_mask"].sel(Time=snap_time).astype(float), SNAPSHOT_LAT_BOUNDS).load()
        precip_object = (precip >= P_DEEP).astype(float)
        class_code = class_code_from_masks(high, active, generated, cirrus, unclassified)
        focus_lat, focus_lon = focus_point_for_snapshot(precip, high, active, generated, cirrus)

        last_precip_im = plot_precip_map(ax_rain, precip, focus_lon, focus_lat)
        last_class_im = plot_class_map(ax_class, class_code, focus_lon, focus_lat)
        ax_rain.set_ylabel(f"{sim_name}\nLatitude")

        cf = get_cloud_fraction(dss[i])
        if cf is None:
            ax_lon.text(0.5, 0.5, "cloud fraction unavailable", transform=ax_lon.transAxes, ha="center", va="center")
            ax_lat.text(0.5, 0.5, "cloud fraction unavailable", transform=ax_lat.transAxes, ha="center", va="center")
        else:
            cf_snapshot = select_time_value(cf, snap_time)
            last_cf_im = plot_cloud_slice(ax_lon, cf_snapshot, class_code, precip, focus_lat, focus_lon, "longitude")
            last_cf_im = plot_cloud_slice(ax_lat, cf_snapshot, class_code, precip, focus_lat, focus_lon, "latitude")
            ax_lat.set_ylabel("")
            ax_lat.tick_params(labelleft=False)

        if i == 0:
            ax_rain.set_title(f"Rain rate; black line P >= {P_DEEP:g}")
            ax_class.set_title("Class mask")
        ax_rain.text(
            0.01,
            0.98,
            format_time_value(snap_time),
            transform=ax_rain.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
        )
        rows.append(
            dict(
                simulation=sim_name,
                snapshot_time=format_time_value(snap_time),
                focus_latitude=float(focus_lat),
                focus_longitude=float(focus_lon),
                high_cloud_fraction=weighted_area_time_mean(high, SNAPSHOT_LAT_BOUNDS),
                active_deep_convection_fraction=weighted_area_time_mean(active, SNAPSHOT_LAT_BOUNDS),
                convectively_generated_fraction=weighted_area_time_mean(generated, SNAPSHOT_LAT_BOUNDS),
                cirrus_like_fraction=weighted_area_time_mean(cirrus, SNAPSHOT_LAT_BOUNDS),
                unclassified_fraction=weighted_area_time_mean(unclassified, SNAPSHOT_LAT_BOUNDS),
                precip_object_fraction=weighted_area_time_mean(precip_object, SNAPSHOT_LAT_BOUNDS),
                denominator="snapshot tropical grid cells",
            )
        )

    if last_precip_im is not None:
        fig.colorbar(
            last_precip_im,
            ax=axes[:, 0].ravel().tolist(),
            label=r"Rain rate (mm day$^{-1}$)",
            shrink=0.9,
        )
    if last_class_im is not None:
        cbar = fig.colorbar(
            last_class_im,
            ax=axes[:, 1].ravel().tolist(),
            ticks=[code for code, _label, _color in CLASS_CODE_SPECS],
            shrink=0.9,
        )
        cbar.ax.set_yticklabels([label for _code, label, _color in CLASS_CODE_SPECS])
    if last_cf_im is not None:
        fig.colorbar(
            last_cf_im,
            ax=axes[:, 2:].ravel().tolist(),
            label="Cloud fraction",
            shrink=0.9,
        )
    fig.suptitle(
        "High-cloud classification snapshots: rain, class mask, and vertical cloud slices\n"
        "Parent high cloud comes from cldfrac above 12 km; classes use hydrometeors and object/distance connectivity"
    )
    save_table(pd.DataFrame(rows), "cloud_classification_validation_snapshots")
    save_figure(3, fig, "cloud_classification_validation_snapshots")
    plt.close(fig)
    progress("Done with figure: cloud_classification_validation_snapshots")


# -----------------------------------------------------------------------------
# Figure 4. Distance-to-precipitation/core diagnostics
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Convectively generated high cloud should be concentrated near
#     precipitating or active deep convective objects.
#   * Cirrus-like high cloud should shift toward larger distances if it is
#     truly separated from active convection.
#   * Distances use a regular-grid approximation in kilometres with periodic
#     longitude, so they are a classification sanity check rather than exact
#     great-circle object geometry.


def arrays_for_numpy(arrays, lat_bounds=TROPICAL_LAT_BOUNDS, max_time=96, max_lat=120, max_lon=240, time_mode="stride"):
    prepared = [subset_lat(da, lat_bounds) for da in arrays]
    prepared = list(xr.align(*prepared, join="inner"))
    indexers = stride_indexers(prepared, max_time=max_time, max_lat=max_lat, max_lon=max_lon, time_mode=time_mode)
    if indexers:
        prepared = [
            da.isel({dim: slc for dim, slc in indexers.items() if dim in da.dims})
            for da in prepared
        ]
    broadcasted = xr.broadcast(*prepared)
    dims = [dim for dim in ("Time", "latitude", "longitude") if dim in broadcasted[0].dims]
    broadcasted = [da.transpose(*dims) for da in broadcasted]
    values = [np.asarray(da.load().values) for da in broadcasted]
    coords = {dim: np.asarray(broadcasted[0][dim].values) for dim in dims}
    time_stride = 1
    if "Time" in indexers and isinstance(indexers["Time"], slice) and indexers["Time"].step is not None:
        time_stride = int(indexers["Time"].step)
    return values, coords, dict(indexers=indexers, time_stride=time_stride, dims=dims)


def grid_spacing_km(lat, lon):
    dlat = float(np.nanmedian(np.abs(np.diff(lat)))) if len(lat) > 1 else 1.0
    dlon = float(np.nanmedian(np.abs(np.diff(lon)))) if len(lon) > 1 else 1.0
    lat_scale = 111.2 * dlat
    lon_scale = 111.2 * np.cos(np.deg2rad(float(np.nanmean(lat)))) * dlon
    return lat_scale, max(lon_scale, 1.0)


def distance_to_mask_km(mask2d, dy_km, dx_km):
    mask2d = np.asarray(mask2d, dtype=bool)
    if not np.any(mask2d):
        return np.full(mask2d.shape, np.nan, dtype=float)
    nlon = mask2d.shape[1]
    tiled = np.concatenate([mask2d, mask2d, mask2d], axis=1)
    dist = ndimage.distance_transform_edt(~tiled, sampling=(dy_km, dx_km))
    return dist[:, nlon:2 * nlon]


def distance_histograms_for_run(fields, classif, sim_name):
    arrays = [
        fields["precip"],
        classif["high_cloud_mask"].astype(bool),
        classif["active_deep_convection_high_cloud_mask"].astype(bool),
    ] + [classif[key].astype(bool) for key, _label, _color in DISTANCE_CLASS_SPECS]
    values, coords, meta = arrays_for_numpy(arrays, **DISTANCE_SAMPLE_LIMITS, time_mode="stride")
    precip, high, active = values[:3]
    class_arrays = dict(zip([key for key, _label, _color in DISTANCE_CLASS_SPECS], values[3:]))
    lat = coords["latitude"]
    lon = coords["longitude"]
    dy_km, dx_km = grid_spacing_km(lat, lon)
    weights2d = np.cos(np.deg2rad(lat))[:, None] * np.ones((len(lat), len(lon)))
    centres = 0.5 * (DISTANCE_BINS_KM[:-1] + DISTANCE_BINS_KM[1:])

    accum = {}
    refs = {
        "precip_object": precip >= P_DEEP,
        "active_deep_convection": active,
    }
    for ref_name in refs:
        for key, label, _color in DISTANCE_CLASS_SPECS:
            accum[(ref_name, key)] = dict(high=np.zeros(len(centres)), cls=np.zeros(len(centres)), class_total=0.0)

    ntime = precip.shape[0] if precip.ndim == 3 else 1
    for t in range(ntime):
        high_t = high[t] if high.ndim == 3 else high
        for ref_name, ref_arr in refs.items():
            ref_t = ref_arr[t] if ref_arr.ndim == 3 else ref_arr
            dist = distance_to_mask_km(ref_t, dy_km, dx_km)
            if not np.any(np.isfinite(dist)):
                continue
            bin_index = np.digitize(dist, DISTANCE_BINS_KM) - 1
            for key, label, _color in DISTANCE_CLASS_SPECS:
                cls_t = class_arrays[key][t] if class_arrays[key].ndim == 3 else class_arrays[key]
                target = accum[(ref_name, key)]
                target["class_total"] += float(np.nansum(weights2d[cls_t]))
                for j in range(len(centres)):
                    in_bin = bin_index == j
                    target["high"][j] += float(np.nansum(weights2d[in_bin & high_t]))
                    target["cls"][j] += float(np.nansum(weights2d[in_bin & cls_t]))

    rows = []
    for ref_name in refs:
        for key, label, _color in DISTANCE_CLASS_SPECS:
            target = accum[(ref_name, key)]
            for j, centre in enumerate(centres):
                denom_high = target["high"][j]
                class_total = target["class_total"]
                rows.append(
                    dict(
                        simulation=sim_name,
                        reference=ref_name,
                        class_key=key,
                        class_label=label,
                        distance_bin_min_km=float(DISTANCE_BINS_KM[j]),
                        distance_bin_max_km=float(DISTANCE_BINS_KM[j + 1]),
                        distance_bin_center_km=float(centre),
                        class_given_high_at_distance=target["cls"][j] / denom_high if denom_high > 0 else np.nan,
                        class_distance_pdf=target["cls"][j] / class_total if class_total > 0 else np.nan,
                        high_denominator_weight=float(denom_high),
                        class_numerator_weight=float(target["cls"][j]),
                        class_total_weight=float(class_total),
                        denominator="high-cloud cells in distance bin for class_given_high; class cells for class_distance_pdf",
                        sample_time_stride=meta["time_stride"],
                    )
                )
    return pd.DataFrame(rows)


def plot_distance_diagnostics(fields_by_run, classifications):
    if not should_run_figure(4):
        return
    progress("Starting Figure 4: cloud_class_distance_to_precip_core")
    rows = []
    for i, sim_name in enumerate(sim_names):
        classif = classifications.get(sim_name)
        if classif is None:
            continue
        rows.append(distance_histograms_for_run(fields_by_run[i], classif, sim_name))
    if not rows:
        print("Skipping Figure 4: distance diagnostics unavailable.")
        return
    df = pd.concat(rows, ignore_index=True)
    save_table(df, "cloud_class_distance_to_precip_core")

    fig, ax = plt.subplots(
        len(DISTANCE_CLASS_SPECS),
        2,
        figsize=(10.8, 3.0 * len(DISTANCE_CLASS_SPECS)),
        constrained_layout=True,
        sharex=True,
        sharey="row",
    )
    references = [("precip_object", f"Nearest precip object\nP >= {P_DEEP:g} mm day$^{{-1}}$"), ("active_deep_convection", "Nearest active deep convection")]
    for r, (class_key, class_label, _class_color) in enumerate(DISTANCE_CLASS_SPECS):
        for c, (ref_name, ref_label) in enumerate(references):
            a = ax[r, c]
            for i, sim_name in enumerate(sim_names):
                sub = df[
                    (df.simulation == sim_name)
                    & (df.reference == ref_name)
                    & (df.class_key == class_key)
                ].sort_values("distance_bin_center_km")
                if len(sub) == 0:
                    continue
                a.plot(
                    sub.distance_bin_center_km,
                    sub.class_given_high_at_distance,
                    color=cols[i],
                    marker="o",
                    ms=3,
                    lw=1.4,
                    label=sim_name if (r == 0 and c == 0) else None,
                )
            if r == 0:
                a.set_title(ref_label)
            if c == 0:
                a.set_ylabel(f"{class_label}\nFraction of high cloud")
            if r == len(DISTANCE_CLASS_SPECS) - 1:
                a.set_xlabel("Distance (km)")
            a.grid(alpha=0.22)
            a.set_xscale("symlog", linthresh=100)
            a.set_xlim(0, DISTANCE_BINS_KM[-1])
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("High-cloud class occurrence versus distance to precipitation and active convection")
    save_figure(4, fig, "cloud_class_distance_to_precip_core")
    plt.close(fig)
    progress("Done with figure: cloud_class_distance_to_precip_core")


# -----------------------------------------------------------------------------
# Figure 5. Time since nearby precipitation
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Active and convectively generated cloud should preferentially occur
#     shortly after local or nearby precipitation.
#   * Cirrus-like cloud with very recent nearby precipitation is less isolated,
#     while long-lag cirrus-like cloud is a stronger separated-cirrus candidate.
#   * The lag is limited by sampled time coverage and the model output interval.


def spatial_dilate_2d(mask2d, lat, lon, radius_deg):
    if radius_deg <= 0:
        return np.asarray(mask2d, dtype=bool)
    dlat = float(np.nanmedian(np.abs(np.diff(lat)))) if len(lat) > 1 else radius_deg
    dlon = float(np.nanmedian(np.abs(np.diff(lon)))) if len(lon) > 1 else radius_deg
    ny = 2 * int(np.ceil(radius_deg / max(dlat, 1.0e-6))) + 1
    nx = 2 * int(np.ceil(radius_deg / max(dlon, 1.0e-6))) + 1
    return ndimage.maximum_filter(np.asarray(mask2d, dtype=np.uint8), size=(ny, nx), mode=("nearest", "wrap")) >= 1


def lag_histograms_for_run(fields, classif, sim_name):
    arrays = [
        fields["precip"],
    ] + [classif[key].astype(bool) for key, _label, _color in DISTANCE_CLASS_SPECS]
    values, coords, meta = arrays_for_numpy(arrays, **LAG_SAMPLE_LIMITS, time_mode="tail")
    precip = values[0]
    class_arrays = dict(zip([key for key, _label, _color in DISTANCE_CLASS_SPECS], values[1:]))
    lat = coords["latitude"]
    lon = coords["longitude"]
    weights2d = np.cos(np.deg2rad(lat))[:, None] * np.ones((len(lat), len(lon)))

    step_hours = dt * meta["time_stride"]
    if "Time" in coords and len(coords["Time"]) > 1 and not np.issubdtype(np.asarray(coords["Time"]).dtype, np.integer):
        step_hours = 24.0 * float(np.nanmedian(np.diff(coords["Time"]) / np.timedelta64(1, "D")))
    age_steps = np.full(precip.shape[1:], np.inf, dtype=float)
    centres = 0.5 * (LAG_BINS_HOURS[:-1] + LAG_BINS_HOURS[1:])
    accum = {
        key: dict(bin_weight=np.zeros(len(centres)), class_total=0.0, beyond_weight=0.0)
        for key, _label, _color in DISTANCE_CLASS_SPECS
    }

    for t in range(precip.shape[0]):
        nearby_precip = spatial_dilate_2d(precip[t] >= P_DEEP, lat, lon, LAG_NEAR_PRECIP_RADIUS_DEG)
        age_steps = np.where(nearby_precip, 0.0, age_steps + 1.0)
        age_hours = age_steps * step_hours
        bin_index = np.digitize(age_hours, LAG_BINS_HOURS) - 1
        for key, _label, _color in DISTANCE_CLASS_SPECS:
            cls = class_arrays[key][t]
            accum[key]["class_total"] += float(np.nansum(weights2d[cls]))
            finite_plotted = np.zeros(cls.shape, dtype=bool)
            for j in range(len(centres)):
                in_bin = cls & (bin_index == j)
                finite_plotted |= in_bin
                accum[key]["bin_weight"][j] += float(np.nansum(weights2d[in_bin]))
            accum[key]["beyond_weight"] += float(np.nansum(weights2d[cls & ~finite_plotted]))

    rows = []
    for key, label, _color in DISTANCE_CLASS_SPECS:
        total = accum[key]["class_total"]
        for j, centre in enumerate(centres):
            rows.append(
                dict(
                    simulation=sim_name,
                    class_key=key,
                    class_label=label,
                    lag_bin_min_hours=float(LAG_BINS_HOURS[j]),
                    lag_bin_max_hours=float(LAG_BINS_HOURS[j + 1]),
                    lag_bin_center_hours=float(centre),
                    fraction_of_class=accum[key]["bin_weight"][j] / total if total > 0 else np.nan,
                    class_bin_weight=float(accum[key]["bin_weight"][j]),
                    class_total_weight=float(total),
                    denominator=f"{label} sampled tropical grid-cell/time occurrences",
                    nearby_precip_radius_deg=LAG_NEAR_PRECIP_RADIUS_DEG,
                    sample_time_stride=meta["time_stride"],
                )
            )
        rows.append(
            dict(
                simulation=sim_name,
                class_key=key,
                class_label=label,
                lag_bin_min_hours=float(LAG_BINS_HOURS[-1]),
                lag_bin_max_hours=np.inf,
                lag_bin_center_hours=np.nan,
                fraction_of_class=accum[key]["beyond_weight"] / total if total > 0 else np.nan,
                class_bin_weight=float(accum[key]["beyond_weight"]),
                class_total_weight=float(total),
                denominator=f"{label} sampled tropical grid-cell/time occurrences",
                nearby_precip_radius_deg=LAG_NEAR_PRECIP_RADIUS_DEG,
                sample_time_stride=meta["time_stride"],
            )
        )
    return pd.DataFrame(rows)


def plot_lag_diagnostics(fields_by_run, classifications):
    if not should_run_figure(5):
        return
    progress("Starting Figure 5: cloud_class_time_since_precip")
    rows = []
    for i, sim_name in enumerate(sim_names):
        classif = classifications.get(sim_name)
        if classif is None:
            continue
        rows.append(lag_histograms_for_run(fields_by_run[i], classif, sim_name))
    if not rows:
        print("Skipping Figure 5: lag diagnostics unavailable.")
        return
    df = pd.concat(rows, ignore_index=True)
    save_table(df, "cloud_class_time_since_precip")

    fig, ax = plt.subplots(
        1,
        len(DISTANCE_CLASS_SPECS),
        figsize=(4.0 * len(DISTANCE_CLASS_SPECS), 3.7),
        constrained_layout=True,
        sharey=True,
    )
    if len(DISTANCE_CLASS_SPECS) == 1:
        ax = [ax]
    for c, (class_key, class_label, _color) in enumerate(DISTANCE_CLASS_SPECS):
        a = ax[c]
        for i, sim_name in enumerate(sim_names):
            sub = df[
                (df.simulation == sim_name)
                & (df.class_key == class_key)
                & np.isfinite(df.lag_bin_center_hours)
            ].sort_values("lag_bin_center_hours")
            if len(sub) == 0:
                continue
            a.plot(
                sub.lag_bin_center_hours,
                sub.fraction_of_class,
                color=cols[i],
                marker="o",
                ms=3,
                lw=1.4,
                label=sim_name if c == 0 else None,
            )
        a.set_title(class_label)
        a.set_xlabel("Hours since nearby precip")
        a.set_xscale("symlog", linthresh=12)
        a.grid(alpha=0.22)
    ax[0].set_ylabel("Fraction of class occurrences")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Time since last nearby precipitation (radius {LAG_NEAR_PRECIP_RADIUS_DEG:g} deg, P >= {P_DEEP:g})")
    save_figure(5, fig, "cloud_class_time_since_precip")
    plt.close(fig)
    progress("Done with figure: cloud_class_time_since_precip")


# -----------------------------------------------------------------------------
# Figure 6. Lightweight cldfrac-threshold sensitivity
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * The three science classes should keep the same qualitative interpretation
#     under one lower and one higher cloud-fraction threshold.
#   * Large changes in the dominant class or class share are flagged in the CSV.


def sampled_classification_for_threshold(ds, fields, cf_thresh):
    cf = get_cloud_fraction(ds)
    if cf is None:
        return None
    precip = fields["precip"]
    if "Time" in cf.dims and "Time" in precip.dims:
        common_time = cf.Time.to_index().intersection(precip.Time.to_index())
        if len(common_time) == 0:
            return None
        cf = cf.sel(Time=common_time)
        precip = precip.sel(Time=common_time)
    cf2d = cf.isel(z=0, drop=True)
    aligned_for_sampling = list(xr.align(cf2d, precip, join="inner"))
    indexers = stride_indexers(
        aligned_for_sampling,
        **SENSITIVITY_SAMPLE_LIMITS,
        time_mode="stride",
    )
    if indexers:
        cf = cf.isel({dim: slc for dim, slc in indexers.items() if dim in cf.dims})
        precip = precip.isel({dim: slc for dim, slc in indexers.items() if dim in precip.dims})
    return classify_high_cloud(ds, cf, precip, cf_thresh=cf_thresh)


def threshold_sensitivity_rows(dss, fields_by_run):
    rows = []
    for sim_name, ds, fields in zip(sim_names, dss, fields_by_run):
        for cf_thresh in SENSITIVITY_CF_THRESHOLDS:
            progress(f"{sim_name}: sensitivity classification for cldfrac threshold {cf_thresh:g}")
            classif = sampled_classification_for_threshold(ds, fields, cf_thresh)
            if classif is None:
                continue
            high_frac = weighted_area_time_mean(classif["high_cloud_mask"], TROPICAL_LAT_BOUNDS)
            for key, label, _color in BUDGET_SPECS:
                frac = weighted_area_time_mean(classif[key], TROPICAL_LAT_BOUNDS)
                rows.append(
                    dict(
                        simulation=sim_name,
                        cldfrac_threshold=float(cf_thresh),
                        class_key=key,
                        class_label=label,
                        domain_fraction=float(frac),
                        within_high_cloud_fraction=float(frac / high_frac) if high_frac > 0 else np.nan,
                        high_cloud_denominator=float(high_frac),
                        denominator=(
                            "sampled tropical area-time fraction for domain_fraction; "
                            "sampled high-cloud area-time fraction for within_high_cloud_fraction"
                        ),
                    )
                )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    flags = []
    science_keys = [
        "active_deep_convection_high_cloud_mask",
        "convectively_generated_high_cloud_mask",
        "cirrus_like_high_cloud_mask",
    ]
    for sim_name in sim_names:
        sub = df[(df.simulation == sim_name) & (df.class_key.isin(science_keys))]
        base = sub[np.isclose(sub.cldfrac_threshold, CF_THRESH)]
        if len(base) == 0:
            continue
        base_by_key = base.set_index("class_key")["within_high_cloud_fraction"]
        strong = False
        notes = []
        for key in science_keys:
            vals = sub[sub.class_key == key]["within_high_cloud_fraction"].to_numpy(dtype=float)
            base_val = float(base_by_key.get(key, np.nan))
            spread = float(np.nanmax(vals) - np.nanmin(vals)) if np.any(np.isfinite(vals)) else np.nan
            limit = max(0.05, 0.30 * abs(base_val)) if np.isfinite(base_val) else 0.05
            if np.isfinite(spread) and spread > limit:
                strong = True
                notes.append(f"{key} spread {spread:.3f}")
        dominant = (
            sub.sort_values("within_high_cloud_fraction")
            .groupby("cldfrac_threshold")
            .tail(1)[["cldfrac_threshold", "class_key"]]
        )
        if dominant["class_key"].nunique() > 1:
            strong = True
            notes.append("dominant class changes")
        flags.append(
            dict(
                simulation=sim_name,
                interpretation_changes_strongly=bool(strong),
                note="; ".join(notes) if notes else "classification fractions stable across tested cldfrac thresholds",
            )
        )
    flag_df = pd.DataFrame(flags)
    if len(flag_df) > 0:
        save_table(flag_df, "cloud_classification_threshold_sensitivity_flags")
    return df


def plot_sensitivity_diagnostics(fields_by_run, classifications, dss=None):
    if not should_run_figure(6):
        return
    progress("Starting Figure 6: cloud_classification_threshold_sensitivity")
    if dss is None:
        print("Skipping Figure 6: source datasets unavailable.")
        return
    df = threshold_sensitivity_rows(dss, fields_by_run)
    if len(df) == 0:
        print("Skipping Figure 6: threshold sensitivity unavailable.")
        return
    save_table(df, "cloud_classification_threshold_sensitivity")

    plot_specs = [
        ("high_cloud_mask", "Total high cloud", "domain_fraction"),
        ("active_deep_convection_high_cloud_mask", "Active deep convection", "within_high_cloud_fraction"),
        ("convectively_generated_high_cloud_mask", "Convectively generated", "within_high_cloud_fraction"),
        ("cirrus_like_high_cloud_mask", "Cirrus-like", "within_high_cloud_fraction"),
        ("unclassified_high_cloud_mask", "Ambiguous/unclassified", "within_high_cloud_fraction"),
    ]
    fig, ax = plt.subplots(
        len(plot_specs),
        1,
        figsize=(7.2, 2.2 * len(plot_specs)),
        constrained_layout=True,
        sharex=True,
    )
    if len(plot_specs) == 1:
        ax = [ax]
    for r, (class_key, title, value_col) in enumerate(plot_specs):
        a = ax[r]
        for i, sim_name in enumerate(sim_names):
            sub = df[(df.simulation == sim_name) & (df.class_key == class_key)].sort_values("cldfrac_threshold")
            if len(sub) == 0:
                continue
            a.plot(
                sub.cldfrac_threshold,
                sub[value_col],
                color=cols[i],
                marker="o",
                lw=1.5,
                label=sim_name if r == 0 else None,
            )
        a.axvline(CF_THRESH, color="0.65", lw=0.9, ls=":")
        a.set_ylabel("Domain frac." if value_col == "domain_fraction" else "Frac. of high")
        a.set_title(title)
        a.grid(alpha=0.22)
    ax[-1].set_xlabel("cldfrac threshold")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Lightweight high-cloud classification sensitivity to cldfrac threshold")
    save_figure(6, fig, "cloud_classification_threshold_sensitivity")
    plt.close(fig)
    progress("Done with figure: cloud_classification_threshold_sensitivity")


# -----------------------------------------------------------------------------
# Load data and run diagnostics
# -----------------------------------------------------------------------------


def load_datasets():
    dss = []
    dss_zm = []
    progress("Loading lat-lon and zonal-mean datasets")
    for i, d in enumerate(data_locs):
        progress(f"{sim_names[i]}: opening lat-lon dataset")
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
        progress(f"{sim_names[i]}: lat-lon dataset ready with {len(ds.data_vars)} variables")

        progress(f"{sim_names[i]}: opening zonal-mean dataset")
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
        progress(f"{sim_names[i]}: zonal-mean dataset ready with {len(zds.data_vars)} variables")
        dss.append(ds)
        dss_zm.append(zds)
    return dss, dss_zm


def prepare_fields(dss, dss_zm):
    fields_by_run = []
    progress("Preparing precipitation, thermodynamic, and TTL fields")
    for sim_name, ds, zds in zip(sim_names, dss, dss_zm):
        progress(f"{sim_name}: preparing fields")
        fields = {"precip": get_precip_rate(ds)}
        for key in ("olr", "pw", "cape", "cin"):
            da = time_aligned_field(ds, key, fields["precip"], required=False)
            if da is not None:
                fields[key] = da
        fields.update(compute_ttl_diagnostics(zds, fields["precip"]))
        progress(f"{sim_name}: fields ready ({', '.join(sorted(fields))})")
        fields_by_run.append(fields)
    return fields_by_run


def prepare_classifications(dss, fields_by_run):
    classifications = {}
    if not should_run_any([1, 2, 3, 4, 5, 6]):
        return classifications
    progress("Building or reading high-cloud classifications")
    for sim_name, ds, fields in zip(sim_names, dss, fields_by_run):
        classifications[sim_name] = get_or_build_classification(ds, fields, sim_name)
    if should_run_any([1, 2, 4, 5, 6]):
        class_overlap_sanity(classifications)
    progress("High-cloud classification stage complete")
    return classifications


def main():
    global figsave
    os.makedirs(save_loc, exist_ok=True)
    os.makedirs(classification_cache_dir, exist_ok=True)
    figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)

    dss, dss_zm = load_datasets()
    fields_by_run = prepare_fields(dss, dss_zm)
    classifications = prepare_classifications(dss, fields_by_run)

    plot_class_budget(classifications)
    plot_residual_diagnostics(fields_by_run, classifications)
    plot_validation_snapshots(dss, fields_by_run, classifications)
    plot_distance_diagnostics(fields_by_run, classifications)
    plot_lag_diagnostics(fields_by_run, classifications)
    plot_sensitivity_diagnostics(fields_by_run, classifications, dss=dss)

    progress("Refined high-cloud classification diagnostics complete")


if __name__ == "__main__":
    main()
