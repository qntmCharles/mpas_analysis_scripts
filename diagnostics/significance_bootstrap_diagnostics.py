#!/usr/bin/env python
# coding: utf-8

"""
Block-resampling consistency diagnostics for the MPAS-A nudging experiments.

This script is a compact block-resampling wrapper around diagnostic families
that have proved useful in the other scripts in this directory. It does not
claim formal statistical significance from a short simulation. Instead, it asks
whether small nudged-control differences have the same sign and comparable
magnitude across time chunks, or whether they look like internal variability.

The diagnostics are intentionally computed from the model output here rather
than imported from the other scripts, because those scripts execute full figure
workflows at module top level.
"""

from __future__ import annotations

import json
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
from high_cloud_classification import (
    CLASSIFICATION_CACHE_VERSION,
    HighCloudThresholds,
    cached_classification_is_current as shared_cached_classification_is_current,
)
from equatorial_wave_config import (
    WAVE_BOXES as SHARED_WAVE_BOXES,
    WAVE_LATITUDE_PROJECTION as SHARED_WAVE_LATITUDE_PROJECTION,
)

try:
    from waveFilter import kf_filter
except ModuleNotFoundError:
    kf_filter = None


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/significance_bootstrap/")

# Figure numbers:
#   1: mean-state lat-height block-resampled response maps
#   2: precipitation scalar/PDF block-consistency diagnostics
#   3: cloud profile/cloud-type block-consistency diagnostics
#   4: CCEW wave-box variance block-consistency diagnostics
#   5: convective organisation block-consistency diagnostics
#   6: basic convection PDF block-consistency diagnostics
#   7: vertical wave-power profile block-consistency diagnostics
#   8: wave-amplitude lag-correlation block-consistency diagnostics
#   9: zonal/meridional wavenumber-variance block-consistency diagnostics
#
# WHAT TO LOOK FOR:
#   * Treat a response as robust only when the central block-resampling interval
#     excludes zero and the response sign is consistent across resampled chunks.
#     These are internal-consistency checks, not formal significance tests.
#   * For spatial fields, prioritise coherent latitude-height or physically
#     meaningful regions over isolated cells that pass the chunk-consistency rule.
#   * Compare chunk-consistent U-control and T-control responses with the U-T contrast:
#     shared thermal/static-stability/cloud-depth structures favour a
#     temperature-driven pathway, while U-only wind/wave residuals favour a
#     wind-driven pathway.
figures_to_run = []
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control_new", "u_nudge", "T_nudge"]
latlon_file = ["latlon.2000-*.nc", "latlon.2000-*.nc", "latlon.2000-*.nc"]
zm_latlon_file = ["zm_latlon.2000-*.nc", "zm_latlon.2000-*.nc", "zm_latlon.2000-*.nc"]
sim_names = ["control", "nudged u", "nudged T"]
cols = ["k", "b", "r"]

ndays = 200
dt = 6
analysis_end_date = "2000-11-21"
time_format = "time"
obs_per_day = int(round(24 / dt))

USE_BOOTSTRAP_CACHE = True
FORCE_RECOMPUTE_BOOTSTRAP_CACHE = False

USE_BOOTSTRAP_CACHE = USE_BOOTSTRAP_CACHE and not env_flag("DISABLE_BOOTSTRAP_CACHE")
FORCE_RECOMPUTE_BOOTSTRAP_CACHE = (
    FORCE_RECOMPUTE_BOOTSTRAP_CACHE
    or env_flag("FORCE_RECOMPUTE_BOOTSTRAP_CACHE")
)
BOOTSTRAP_CACHE_DIR = diagnostic_cache_dir("significance_bootstrap_cache", root_dir=root_dir)

BOOTSTRAP_BLOCK_DAYS = 10.0
CLOUD_TYPE_BLOCK_DAYS = 5.0
N_BOOTSTRAP = 1000
FIELD_N_BOOTSTRAP = 600
BOOTSTRAP_SEED = 20240602
MIN_BLOCKS = 6
CI_PERCENTILES = (2.5, 97.5)
ROBUST_SIGN_THRESHOLD = 0.95
# Output columns retain ci_low/ci_high for compatibility. Interpret them as a
# central interval from resampled time chunks, not as a formal confidence
# interval from an independent observational sample.

TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
DEEP_TROPICS_LAT_BOUNDS = (-10.0, 10.0)
WAVE_FILTER_LAT_BOUNDS = (-10.0, 10.0)

TTL_Z_BOUNDS_M = (14000.0, 18000.0)
UPPER_TROPOSPHERE_Z_BOUNDS_M = (12000.0, 18000.0)

PRECIP_STRONG_THRESHOLD = 5.0
PRECIP_WEAK_THRESHOLD = 1.0
PRECIP_HEAVY_THRESHOLD = 20.0
PRECIP_PDF_BINS = np.linspace(0.0, 120.0, 121)
PRECIP_PDF_PLOT_MAX = 35.0
PRECIP_PDF_SYMLOG_LINTHRESH = 1.0e-3

BASIC_CONVECTION_PDF_SPECS = [
    dict(name="Precip", label="Precipitation", units="mm day-1", bins=np.linspace(0.0, 80.0, 121), symlog=True),
    dict(name="OLR", label="OLR", units="W m-2", bins=np.linspace(100.0, 350.0, 121), symlog=False),
    dict(name="PW", label="Precipitable water", units="kg m-2", bins=np.linspace(0.0, 80.0, 121), symlog=False),
    dict(name="CAPE", label="CAPE", units="J kg-1", bins=np.linspace(0.0, 5000.0, 121), symlog=False),
]
WAVENUMBER_FIELDS = ["Precip", "OLR", "PW"]
WAVENUMBER_LAT_BOUNDS = (-10.0, 10.0)
WAVENUMBER_MAX_K = 15

WAVE_PROFILE_WAVES = ["Kelvin", "ER", "Easterly"]
WAVE_PROFILE_FIELDS = [
    dict(name="u", prefix="u", units="m s-1"),
    dict(name="T", prefix="T", units="K"),
    dict(name="w", prefix="w", units="m s-1"),
]
WAVE_PROFILE_LEVELS_HPA = [100, 150, 200, 250, 300, 500, 700, 850]

WAVE_LAG_WAVES = ["Kelvin", "ER", "Easterly"]
WAVE_LAG_TARGET_FIELDS = ["Precip", "OLR", "PW"]
WAVE_LAG_MAX_DAYS = 8.0
WAVE_LAG_BLOCK_DAYS = 30.0

ORGANISATION_OUTPUT_DIR = "/home/cwp29/tmp/mpas/crf_on/final/organisation"
ORGANISATION_TIMESERIES_FILE = os.path.join(ORGANISATION_OUTPUT_DIR, "organisation_lorg_timeseries.csv")
ORGANISATION_TRACKS_FILE = os.path.join(ORGANISATION_OUTPUT_DIR, "organisation_track_lifetimes.csv")
ORGANISATION_METRIC_METADATA_FILE = os.path.join(ORGANISATION_OUTPUT_DIR, "organisation_metric_cache_metadata.json")
ORGANISATION_TRACK_METADATA_FILE = os.path.join(ORGANISATION_OUTPUT_DIR, "organisation_track_persistence_cache_metadata.json")
ORGANISATION_BLOCK_DAYS = 5.0
ORGANISATION_CONDITION_NBINS = 4
ORGANISATION_MIN_CONDITION_SAMPLES = 3
ORGANISATION_METRIC_SPECS = [
    ("lorg_km", "km"),
    ("lorg_normalised", ""),
    ("object_count", "count"),
    ("mean_object_area_km2", "km2"),
    ("median_object_area_km2", "km2"),
    ("largest_object_area_km2", "km2"),
    ("largest_object_fraction", "fraction"),
    ("convective_area_fraction", "fraction"),
]
ORGANISATION_TRACK_METRIC_SPECS = [
    ("lifetime_hours", "h"),
    ("max_area_km2", "km2"),
    ("mean_area_km2", "km2"),
    ("area_time_km2_h", "km2 h"),
    ("precip_area_rate_time", "mm day-1 km2 h"),
    ("starts_as_split", "fraction"),
    ("ends_as_merge", "fraction"),
    ("complete_lifecycle", "fraction"),
]
# Track-persistence rows come from convective_organisation_diagnostics.py, which
# uses PyFLEXTRKR overlap tracking to define object lifetimes. This script does
# the across-time chunk-consistency check; it is not invoking a separate
# PyFLEXTRKR statistical test.

# Shared coarse-grid precipitation-regime cloud thresholds.
CF_THRESH = 0.05
P_HIGH_HPA = 440.0
P_PRECIP_MIN = 1.0
CONVECTIVE_FRACTION_THRESHOLD = 0.50
CLOUD_CLASSIFICATION_CACHE_DIRS = [
    diagnostic_cache_dir("classification_cache", root_dir=root_dir),
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
    "Tropics_upper_troposphere_12_18km": dict(lat_bounds=TROPICAL_LAT_BOUNDS, z_bounds=UPPER_TROPOSPHERE_Z_BOUNDS_M),
}

MATCHED_PATTERN_DOMAINS = {
    "TTL_70_200hPa_30S30N": dict(lat_bounds=(-30.0, 30.0), p_bounds=(70.0, 200.0)),
    "UTLS_50_300hPa_40S40N": dict(lat_bounds=(-40.0, 40.0), p_bounds=(50.0, 300.0)),
    "DeepTropics_70_200hPa_10S10N": dict(lat_bounds=(-10.0, 10.0), p_bounds=(70.0, 200.0)),
}

WAVE_BOXES = {
    name: SHARED_WAVE_BOXES[name]
    for name in ("Kelvin", "EIG", "IG_antisym", "Easterly", "ER")
}
WAVE_LATITUDE_PROJECTION = {
    name: SHARED_WAVE_LATITUDE_PROJECTION[name]
    for name in WAVE_BOXES
}
WAVE_FIELDS = ["Precip", "OLR", "PW"]

needed_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw", "olrtoa", "cape",
    "qv", "relhum", "w",
    "theta", "pressure", "zgrid", "cldfrac",
    "uReconstructZonal", "uReconstructMeridional",
    "t_isobaric", "uzonal_isobaric", "umeridional_isobaric", "rh_isobaric",
    "qv_isobaric",
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


def ensure_bootstrap_cache_dir():
    os.makedirs(BOOTSTRAP_CACHE_DIR, exist_ok=True)
    return BOOTSTRAP_CACHE_DIR


def bootstrap_cache_path(filename):
    return os.path.join(ensure_bootstrap_cache_dir(), filename)


def jsonable_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable_value(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def base_bootstrap_cache_signature(family, extra=None):
    signature = dict(
        cache_version="significance_bootstrap_v1",
        family=family,
        root_dir=root_dir,
        data_locs=list(data_locs),
        latlon_file=list(latlon_file),
        zm_latlon_file=list(zm_latlon_file),
        sim_names=list(sim_names),
        ndays=int(ndays),
        dt=float(dt),
        analysis_end_date=analysis_end_date,
        time_format=time_format,
        bootstrap_block_days=float(BOOTSTRAP_BLOCK_DAYS),
        cloud_type_block_days=float(CLOUD_TYPE_BLOCK_DAYS),
        n_bootstrap=int(N_BOOTSTRAP),
        field_n_bootstrap=int(FIELD_N_BOOTSTRAP),
        bootstrap_seed=int(BOOTSTRAP_SEED),
        min_blocks=int(MIN_BLOCKS),
        ci_percentiles=jsonable_value(CI_PERCENTILES),
        robust_sign_threshold=float(ROBUST_SIGN_THRESHOLD),
        comparisons=jsonable_value(COMPARISONS),
    )
    if extra:
        signature.update(jsonable_value(extra))
    return signature


def read_cache_metadata(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_cache_metadata(path, signature):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(jsonable_value(signature), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def bootstrap_cache_matches(cache_name, signature):
    if not USE_BOOTSTRAP_CACHE:
        return False
    if FORCE_RECOMPUTE_BOOTSTRAP_CACHE:
        print(f"Ignoring cached {cache_name}: force recompute is enabled.")
        return False
    metadata_path = bootstrap_cache_path(f"{cache_name}.metadata.json")
    try:
        cached = read_cache_metadata(metadata_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Ignoring cached {cache_name}: could not read metadata ({exc}).")
        return False
    if cached is None:
        return False
    signature = jsonable_value(signature)
    changed = [
        key
        for key in sorted(signature)
        if cached.get(key) != signature.get(key)
    ]
    if changed:
        shown = ", ".join(changed[:8])
        suffix = "" if len(changed) <= 8 else ", ..."
        print(f"Ignoring cached {cache_name}: metadata changed ({shown}{suffix}).")
        return False
    return True


def load_cached_table(cache_name, signature):
    if not bootstrap_cache_matches(cache_name, signature):
        return None
    path = bootstrap_cache_path(f"{cache_name}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    print(f"Using cached bootstrap table {path}")
    return df


def write_cached_table(cache_name, df, signature):
    if not USE_BOOTSTRAP_CACHE or df is None:
        return
    df.to_csv(bootstrap_cache_path(f"{cache_name}.csv"), index=False)
    write_cache_metadata(bootstrap_cache_path(f"{cache_name}.metadata.json"), signature)


def append_cached_scalar_rows(df):
    if df is not None and len(df) > 0:
        scalar_metric_rows.extend(df.to_dict("records"))


def cached_diff_dataset_filename(cache_name, comp_name):
    return f"{cache_name}_{comp_name}.nc"


def load_cached_diff_datasets(cache_name, signature):
    if not bootstrap_cache_matches(cache_name, signature):
        return None
    metadata = read_cache_metadata(bootstrap_cache_path(f"{cache_name}.metadata.json")) or {}
    comp_names = metadata.get("comparison_names", [name for name, *_ in COMPARISONS])
    out = {}
    for comp_name in comp_names:
        path = bootstrap_cache_path(cached_diff_dataset_filename(cache_name, comp_name))
        if not os.path.exists(path):
            return None
        with xr.open_dataset(path) as ds:
            out[comp_name] = ds.load()
    print(f"Using cached bootstrap response fields for {cache_name}")
    return out


def write_cached_diff_datasets(cache_name, diff_by_comp, signature):
    if not USE_BOOTSTRAP_CACHE or not diff_by_comp:
        return
    for comp_name, ds in diff_by_comp.items():
        path = bootstrap_cache_path(cached_diff_dataset_filename(cache_name, comp_name))
        tmp = path + ".tmp"
        ds.to_netcdf(tmp)
        os.replace(tmp, path)
    metadata = dict(jsonable_value(signature))
    metadata["comparison_names"] = list(diff_by_comp)
    write_cache_metadata(bootstrap_cache_path(f"{cache_name}.metadata.json"), metadata)


def response_dataset_from_table(df, coord_col, coord_name, dim_name=None):
    dim_name = dim_name or coord_name
    sub = df.sort_values(coord_col)
    coord = sub[coord_col].to_numpy(dtype=float)
    data_vars = {}
    for col in ("mean_response", "ci_low", "ci_high", "bootstrap_noise", "response_to_noise", "sign_robustness"):
        if col in sub.columns:
            data_vars[col] = (dim_name, pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float))
    if "robust" in sub.columns:
        robust = sub["robust"]
        if robust.dtype == bool:
            robust_values = robust.to_numpy()
        else:
            robust_values = robust.astype(str).str.lower().isin(["true", "1", "yes", "y"]).to_numpy()
        data_vars["robust"] = (dim_name, robust_values)
    if "ci_excludes_zero" in sub.columns:
        excludes = sub["ci_excludes_zero"]
        if excludes.dtype == bool:
            excludes_values = excludes.to_numpy()
        else:
            excludes_values = excludes.astype(str).str.lower().isin(["true", "1", "yes", "y"]).to_numpy()
        data_vars["ci_excludes_zero"] = (dim_name, excludes_values)
    return xr.Dataset(data_vars, coords={coord_name: (dim_name, coord)})


def precip_pdf_diffs_from_table(df):
    return {
        comp_name: response_dataset_from_table(sub, "precip_bin_center_mmday", "precip_bin")
        for comp_name, sub in df.groupby("comparison")
    }


def basic_pdf_diffs_from_table(df):
    out = {}
    for field_name, field_df in df.groupby("field"):
        out[field_name] = {
            comp_name: response_dataset_from_table(sub, "bin_center", "pdf_bin")
            for comp_name, sub in field_df.groupby("comparison")
        }
    return out


def wavenumber_diffs_from_table(df):
    out = {}
    for (direction, field_name), group in df.groupby(["direction", "field"]):
        out[(direction, field_name)] = {
            comp_name: response_dataset_from_table(sub, "wavenumber", "wavenumber")
            for comp_name, sub in group.groupby("comparison")
        }
    return out


def cloud_profile_diffs_from_table(df):
    return {
        comp_name: response_dataset_from_table(sub, "z_m", "z")
        for comp_name, sub in df.groupby("comparison")
    }


def wave_lag_diffs_from_table(df):
    out = {}
    for (wave_name, target_name), group in df.groupby(["wave", "target"]):
        out[(wave_name, target_name)] = {
            comp_name: response_dataset_from_table(sub, "lag_days", "lag_days")
            for comp_name, sub in group.groupby("comparison")
        }
    return out


def diff_datasets_to_profile_table(diff_by_comp):
    rows = []
    for comp_name, ds in diff_by_comp.items():
        comp_label = next(label for name, _, _, label in COMPARISONS if name == comp_name)
        z_vals = z_values_m(ds.mean_response)
        for zi, z_m in enumerate(z_vals):
            rows.append(
                dict(
                    comparison=comp_name,
                    comparison_label=comp_label,
                    z_m=float(z_m),
                    mean_response=float(ds.mean_response.isel(z=zi).values),
                    ci_low=float(ds.ci_low.isel(z=zi).values),
                    ci_high=float(ds.ci_high.isel(z=zi).values),
                    sign_robustness=float(ds.sign_robustness.isel(z=zi).values),
                    robust=bool(ds.robust.isel(z=zi).values),
                )
            )
    return pd.DataFrame(rows)


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


def assign_midlevel_height_from_zgrid(da, source_ds):
    out = rename_vertical_to_z(da)
    if "z" in out.coords:
        zvals = np.asarray(out.z.values, dtype=float)
        if zvals.size == out.sizes["z"] and np.nanmax(np.abs(zvals)) > 100.0:
            return out

    if "zgrid" not in source_ds.data_vars:
        raise ValueError("zgrid is required to assign geometric height to model-level field")

    zgrid = finite_or_nan(source_ds.zgrid)
    mean_dims = [dim for dim in ("latitude", "longitude") if dim in zgrid.dims]
    zgrid_mean = zgrid.mean(dim=mean_dims, skipna=True).load()
    zmid = 0.5 * (
        zgrid_mean.isel(nVertLevelsP1=slice(0, -1)).values
        + zgrid_mean.isel(nVertLevelsP1=slice(1, None)).values
    )
    if out.sizes["z"] != zmid.size:
        raise ValueError(
            f"model-level field has {out.sizes['z']} levels but zgrid implies {zmid.size}"
        )
    return out.assign_coords(z=("z", zmid))


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


def field_from_candidates(zds, candidates, name, scale=1.0):
    for candidate in candidates:
        if candidate in zds.data_vars:
            out = rename_vertical_to_z(finite_or_nan(zds[candidate])) * scale
            out.name = name
            return out
    raise ValueError(f"None of {candidates} available")


def latlon_field_from_candidates(ds, candidates, name, scale=1.0):
    for candidate in candidates:
        if candidate in ds.data_vars:
            out = assign_midlevel_height_from_zgrid(finite_or_nan(ds[candidate]), ds) * scale
            out.name = name
            return out
    raise ValueError(f"None of {candidates} available")


def water_vapour_field(ds, zds):
    if "qv" in zds.data_vars:
        return field_from_candidates(zds, ("qv",), "water_vapour", scale=1000.0)
    return latlon_field_from_candidates(ds, ("qv",), "water_vapour", scale=1000.0)


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
         getter=water_vapour_field, parity="even"),
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
    comp_items = [
        (name, label, diff_by_comparison[name])
        for name, _, _, label in COMPARISONS
        if name in diff_by_comparison
    ]
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
            levels=levels, cmap=difference_cmap(levels), extend="both"
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
        f"{spec['label']}: mean response; hatching = block-resampling interval excludes zero and "
        f"sign consistency >= {ROBUST_SIGN_THRESHOLD:.2f}"
    )
    figsave.save(fig, f"mean_state_{spec['name']}_bootstrap_latheight", number=1)
    plt.close(fig)


def run_mean_state_pathway():
    for spec in LATHEIGHT_FIELD_SPECS:
        try:
            print(f"\nMean-state bootstrap: {spec['name']}")
            cache_signature = base_bootstrap_cache_signature(
                "mean_state_pathway",
                extra=dict(
                    diagnostic=spec["name"],
                    domains=jsonable_value(LATHEIGHT_DOMAINS),
                    tropical_lat_bounds=list(TROPICAL_LAT_BOUNDS),
                    deep_tropics_lat_bounds=list(DEEP_TROPICS_LAT_BOUNDS),
                ),
            )
            cached_rows = load_cached_table(f"mean_state_{spec['name']}_scalar", cache_signature)
            cached_diffs = load_cached_diff_datasets(f"mean_state_{spec['name']}_diffs", cache_signature)
            if cached_rows is not None and cached_diffs is not None:
                append_cached_scalar_rows(cached_rows)
                plot_latheight_bootstrap(spec, cached_diffs)
                record_implemented(f"cached mean-state lat-height bootstrap for {spec['name']}")
                continue

            row_start = len(scalar_metric_rows)
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
            spec_rows = pd.DataFrame(scalar_metric_rows[row_start:])
            write_cached_table(f"mean_state_{spec['name']}_scalar", spec_rows, cache_signature)
            write_cached_diff_datasets(f"mean_state_{spec['name']}_diffs", diff_by_comp, cache_signature)
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
        plot_mask = x <= PRECIP_PDF_PLOT_MAX
        y = ds.mean_response.values
        ax.plot(
            x[plot_mask],
            y[plot_mask],
            color=colors_by_comp.get(comp_name, "k"),
            lw=1.4,
            label=label,
        )
        ax.fill_between(
            x[plot_mask],
            ds.ci_low.values[plot_mask],
            ds.ci_high.values[plot_mask],
            color=colors_by_comp.get(comp_name, "k"),
            alpha=0.12,
        )
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlim(0.0, PRECIP_PDF_PLOT_MAX)
    ax.set_yscale("symlog", linthresh=PRECIP_PDF_SYMLOG_LINTHRESH)
    ax.grid(alpha=0.18)
    ax.set_xlabel("Precipitation (mm day-1)")
    ax.set_ylabel("PDF response (symlog)")
    ax.set_title("Tropical precipitation PDF block-consistency response, low-rate focus")
    ax.legend(frameon=False, fontsize=8)
    figsave.save(fig, "precipitation_pdf_bootstrap_response", number=2)
    plt.close(fig)


def run_precipitation_response():
    try:
        precip_by_run = {i: get_precip_rate(ds) for i, ds in enumerate(dss)}
        cache_signature = base_bootstrap_cache_signature(
            "precipitation_response",
            extra=dict(
                tropical_lat_bounds=list(TROPICAL_LAT_BOUNDS),
                precip_heavy_threshold=float(PRECIP_HEAVY_THRESHOLD),
                precip_weak_threshold=float(PRECIP_WEAK_THRESHOLD),
                precip_pdf_bins=jsonable_value(PRECIP_PDF_BINS),
            ),
        )
        cached_scalar = load_cached_table("precipitation_response_scalar", cache_signature)
        cached_pdf = load_cached_table("precipitation_pdf_response", cache_signature)
        if cached_scalar is not None and cached_pdf is not None:
            append_cached_scalar_rows(cached_scalar)
            save_table(cached_pdf, "precipitation_pdf_bootstrap_response.csv")
            plot_precip_pdf_differences(precip_pdf_diffs_from_table(cached_pdf))
            record_implemented("cached precipitation response block-consistency metrics")
            return precip_by_run

        row_start = len(scalar_metric_rows)
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
        write_cached_table("precipitation_response_scalar", pd.DataFrame(scalar_metric_rows[row_start:]), cache_signature)
        write_cached_table("precipitation_pdf_response", pd.DataFrame(pdf_rows), cache_signature)
        record_implemented("precipitation mean, variance, upper-tail, weak-rain fraction, and PDF block-consistency metrics")
        return precip_by_run
    except Exception as exc:
        record_failure("precipitation response bootstrap", exc)
        return {}


# -----------------------------------------------------------------------------
# Basic convection distributions: PDFs of PR/OLR/PW/CAPE
# -----------------------------------------------------------------------------

def field_pdf_block_values(da, bins, lat_bounds=TROPICAL_LAT_BOUNDS):
    x = subset_latitude(da, lat_bounds)
    centres = 0.5 * (bins[:-1] + bins[1:])
    blocks = []
    for slc in block_slices(x):
        vals = np.asarray(x.isel(Time=slc).values, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            hist = np.full(centres.size, np.nan)
        else:
            hist, _ = np.histogram(vals, bins=bins, density=True)
        blocks.append(xr.DataArray(hist, dims=("pdf_bin",), coords={"pdf_bin": centres}, name="pdf"))
    return xr.concat(blocks, dim=xr.IndexVariable("block", np.arange(len(blocks))))


def plot_basic_convection_pdf_differences(pdf_diffs_by_field):
    if not should_run_figure(6) or not pdf_diffs_by_field:
        return
    nfields = len(pdf_diffs_by_field)
    ncols = min(2, nfields)
    nrows = int(np.ceil(nfields / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows), constrained_layout=True, squeeze=False)
    colors_by_comp = {"u_minus_control": "b", "T_minus_control": "r", "u_minus_T": "0.2"}
    spec_by_name = {spec["name"]: spec for spec in BASIC_CONVECTION_PDF_SPECS}

    for ax, (field_name, diffs) in zip(axes.ravel(), pdf_diffs_by_field.items()):
        spec = spec_by_name[field_name]
        for comp_name, ds in diffs.items():
            label = next(label for name, _, _, label in COMPARISONS if name == comp_name)
            x = ds.pdf_bin.values
            y = ds.mean_response.values
            color = colors_by_comp.get(comp_name, "k")
            ax.plot(x, y, color=color, lw=1.25, label=label)
            ax.fill_between(x, ds.ci_low.values, ds.ci_high.values, color=color, alpha=0.12)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_yscale("symlog", linthresh=1.0e-4)
        if field_name == "Precip":
            ax.set_xlim(0.0, PRECIP_PDF_PLOT_MAX)
        ax.set_title(spec["label"])
        ax.set_xlabel(f"{spec['label']} ({spec['units']})")
        ax.set_ylabel("PDF response (symlog)")
        ax.grid(alpha=0.18)
    for ax in axes.ravel()[nfields:]:
        ax.set_visible(False)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Basic convective-regime PDF block-consistency responses")
    figsave.save(fig, "basic_convection_pdf_bootstrap_response", number=6)
    plt.close(fig)


def run_basic_convection_pdf_bootstrap():
    try:
        cache_signature = base_bootstrap_cache_signature(
            "basic_convection_pdf",
            extra=dict(
                specs=[
                    dict(name=spec["name"], bins=jsonable_value(spec["bins"]))
                    for spec in BASIC_CONVECTION_PDF_SPECS
                ],
                tropical_lat_bounds=list(TROPICAL_LAT_BOUNDS),
            ),
        )
        cached_rows = load_cached_table("basic_convection_pdf_response", cache_signature)
        if cached_rows is not None:
            save_table(cached_rows, "basic_convection_pdf_bootstrap_response.csv")
            plot_basic_convection_pdf_differences(basic_pdf_diffs_from_table(cached_rows))
            record_implemented("cached basic convection PDF block-consistency metrics")
            return

        rows = []
        pdf_diffs_by_field = {}
        rng = np.random.default_rng(BOOTSTRAP_SEED + 222)
        for spec in BASIC_CONVECTION_PDF_SPECS:
            try:
                blocks_by_run = {
                    i: field_pdf_block_values(get_surface_convection_field(dss[i], spec["name"]), spec["bins"])
                    for i in range(len(dss))
                }
            except Exception as exc:
                record_skip(f"basic convection PDF {spec['name']} ({exc})")
                continue

            diffs = {}
            for comp_name, run_idx, ref_idx, comp_label in COMPARISONS:
                ds = bootstrap_field_difference(blocks_by_run[run_idx], blocks_by_run[ref_idx], rng)
                diffs[comp_name] = ds
                for k, bin_center in enumerate(ds.pdf_bin.values):
                    rows.append(
                        {
                            "field": spec["name"],
                            "field_label": spec["label"],
                            "units": spec["units"],
                            "comparison": comp_name,
                            "comparison_label": comp_label,
                            "bin_center": float(bin_center),
                            "mean_response": float(ds.mean_response.isel(pdf_bin=k).values),
                            "ci_low": float(ds.ci_low.isel(pdf_bin=k).values),
                            "ci_high": float(ds.ci_high.isel(pdf_bin=k).values),
                            "sign_robustness": float(ds.sign_robustness.isel(pdf_bin=k).values),
                            "robust": bool(ds.robust.isel(pdf_bin=k).values),
                        }
                    )
            pdf_diffs_by_field[spec["name"]] = diffs

        if rows:
            rows_df = pd.DataFrame(rows)
            save_table(rows_df, "basic_convection_pdf_bootstrap_response.csv")
            plot_basic_convection_pdf_differences(pdf_diffs_by_field)
            write_cached_table("basic_convection_pdf_response", rows_df, cache_signature)
            record_implemented("basic convection PDF block-consistency metrics for precipitation, OLR, PW, and CAPE where available")
    except Exception as exc:
        record_failure("basic convection PDF bootstrap", exc)


# -----------------------------------------------------------------------------
# Basic organisation spectra: zonal and meridional wavenumber variance fractions
# -----------------------------------------------------------------------------

def spatial_wavenumber_fraction_from_series(x, spatial_dim, max_k=WAVENUMBER_MAX_K):
    x = x - x.mean(dim="Time", skipna=True)
    x = x.transpose("Time", spatial_dim)
    arr = np.asarray(x.values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected (Time, {spatial_dim}), got shape {arr.shape}")
    if np.any(~np.isfinite(arr)):
        row_mean = np.nanmean(arr, axis=1, keepdims=True)
        arr = np.where(np.isfinite(arr), arr, row_mean)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    fft = np.fft.rfft(arr, axis=1)
    power = np.abs(fft) ** 2
    if power.shape[1] > 2:
        power[:, 1:-1] *= 2.0
    pk = np.nanmean(power, axis=0)
    denom = np.nansum(pk)
    pk = pk / denom if denom > 0 else np.full_like(pk, np.nan)
    kk = np.arange(pk.size)
    keep = kk <= max_k
    return kk[keep], pk[keep]


def block_wavenumber_fraction_values(da, direction):
    x = subset_latitude(da, WAVENUMBER_LAT_BOUNDS)
    blocks = []
    for slc in block_slices(x):
        xb = x.isel(Time=slc)
        if direction == "zonal":
            series = weighted_mean_dims(xb, ["latitude"])
            spatial_dim = "longitude"
        elif direction == "meridional":
            series = xb.mean(dim="longitude", skipna=True)
            spatial_dim = "latitude"
        else:
            raise ValueError(f"unknown wavenumber direction {direction}")
        k, frac = spatial_wavenumber_fraction_from_series(series, spatial_dim, max_k=WAVENUMBER_MAX_K)
        blocks.append(
            xr.DataArray(
                frac,
                dims=("wavenumber",),
                coords={"wavenumber": k},
                name=f"{direction}_wavenumber_variance_fraction",
            )
        )
    return xr.concat(blocks, dim=xr.IndexVariable("block", np.arange(len(blocks))))


def plot_wavenumber_bootstrap_differences(diff_by_direction_field):
    if not should_run_figure(9) or not diff_by_direction_field:
        return
    fields = [field for field in WAVENUMBER_FIELDS if any((direction, field) in diff_by_direction_field for direction in ("zonal", "meridional"))]
    directions = ["zonal", "meridional"]
    fig, axes = plt.subplots(len(directions), len(fields), figsize=(3.7 * len(fields), 3.2 * len(directions)), constrained_layout=True, squeeze=False, sharey=True)
    colors_by_comp = {"u_minus_control": "b", "T_minus_control": "r", "u_minus_T": "0.2"}
    for ri, direction in enumerate(directions):
        for ci, field in enumerate(fields):
            ax = axes[ri, ci]
            diffs = diff_by_direction_field.get((direction, field), {})
            for comp_name, ds in diffs.items():
                label = next(label for name, _, _, label in COMPARISONS if name == comp_name)
                k = ds.wavenumber.values
                color = colors_by_comp.get(comp_name, "k")
                ax.plot(k, ds.mean_response.values, color=color, marker="o", ms=3, lw=1.1, label=label)
                ax.fill_between(k, ds.ci_low.values, ds.ci_high.values, color=color, alpha=0.12)
            ax.axhline(0.0, color="0.6", lw=0.8)
            ax.set_title(f"{field}: {direction}")
            ax.set_xlabel("Wavenumber")
            ax.set_xlim(0, WAVENUMBER_MAX_K)
            ax.grid(alpha=0.18)
            if ci == 0:
                ax.set_ylabel("Variance-fraction response")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Tropical wavenumber variance-fraction block-consistency responses")
    figsave.save(fig, "wavenumber_variance_fraction_bootstrap_response", number=9)
    plt.close(fig)


def run_wavenumber_variance_bootstrap():
    try:
        cache_signature = base_bootstrap_cache_signature(
            "wavenumber_variance",
            extra=dict(
                fields=list(WAVENUMBER_FIELDS),
                lat_bounds=list(WAVENUMBER_LAT_BOUNDS),
                max_k=int(WAVENUMBER_MAX_K),
            ),
        )
        cached_rows = load_cached_table("wavenumber_variance_fraction_response", cache_signature)
        if cached_rows is not None:
            save_table(cached_rows, "wavenumber_variance_fraction_bootstrap_response.csv")
            plot_wavenumber_bootstrap_differences(wavenumber_diffs_from_table(cached_rows))
            record_implemented("cached wavenumber variance-fraction block-consistency metrics")
            return

        rows = []
        diffs_by_direction_field = {}
        rng = np.random.default_rng(BOOTSTRAP_SEED + 333)
        for field_name in WAVENUMBER_FIELDS:
            try:
                field_by_run = {i: get_surface_convection_field(dss[i], field_name) for i in range(len(dss))}
            except Exception as exc:
                record_skip(f"wavenumber variance {field_name} ({exc})")
                continue
            for direction in ("zonal", "meridional"):
                blocks_by_run = {
                    i: block_wavenumber_fraction_values(field_by_run[i], direction)
                    for i in range(len(dss))
                }
                diffs = {}
                for comp_name, run_idx, ref_idx, comp_label in COMPARISONS:
                    ds = bootstrap_field_difference(blocks_by_run[run_idx], blocks_by_run[ref_idx], rng)
                    diffs[comp_name] = ds
                    for k_index, wavenumber in enumerate(ds.wavenumber.values):
                        rows.append(
                            {
                                "field": field_name,
                                "direction": direction,
                                "comparison": comp_name,
                                "comparison_label": comp_label,
                                "wavenumber": int(wavenumber),
                                "mean_response": float(ds.mean_response.isel(wavenumber=k_index).values),
                                "ci_low": float(ds.ci_low.isel(wavenumber=k_index).values),
                                "ci_high": float(ds.ci_high.isel(wavenumber=k_index).values),
                                "sign_robustness": float(ds.sign_robustness.isel(wavenumber=k_index).values),
                                "robust": bool(ds.robust.isel(wavenumber=k_index).values),
                            }
                        )
                diffs_by_direction_field[(direction, field_name)] = diffs
        if rows:
            rows_df = pd.DataFrame(rows)
            save_table(rows_df, "wavenumber_variance_fraction_bootstrap_response.csv")
            plot_wavenumber_bootstrap_differences(diffs_by_direction_field)
            write_cached_table("wavenumber_variance_fraction_response", rows_df, cache_signature)
            record_implemented("zonal and meridional wavenumber variance-fraction block-consistency metrics for precipitation/OLR/PW")
    except Exception as exc:
        record_failure("wavenumber variance-fraction bootstrap", exc)


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


def classification_thresholds():
    return HighCloudThresholds(
        cf_thresh=CF_THRESH,
        high_cloud_top_pressure_hpa=P_HIGH_HPA,
        surface_precip_min_mm_day=P_PRECIP_MIN,
        convective_fraction_threshold=CONVECTIVE_FRACTION_THRESHOLD,
    )


def cached_classification_is_current(cached):
    return shared_cached_classification_is_current(cached, classification_thresholds())


REQUIRED_HIGH_CLOUD_CLASSIFICATION_VARS = (
    "high_cloud_fraction",
    "high_cloud_mask",
    "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
    "unclassified_high_cloud_mask",
)


def cloud_masks_from_cache(sim_name, precip_ref):
    filename = f"{sim_name.replace(' ', '_')}_classified_high_cloud.nc"
    checked = []
    for cache_dir in CLOUD_CLASSIFICATION_CACHE_DIRS:
        path = os.path.join(cache_dir, filename)
        checked.append(path)
        if not os.path.exists(path):
            continue
        cached = xr.open_dataset(path, chunks={})
        if not cached_classification_is_current(cached):
            cached.close()
            raise RuntimeError(f"{sim_name}: stale high-cloud classification cache: {path}")
        missing = [
            key for key in REQUIRED_HIGH_CLOUD_CLASSIFICATION_VARS
            if key not in cached.data_vars
        ]
        if missing:
            cached.close()
            raise RuntimeError(
                f"{sim_name}: high-cloud classification cache lacks required shared variables {missing}: {path}"
            )
        out = {}
        for key in REQUIRED_HIGH_CLOUD_CLASSIFICATION_VARS:
            da = cached[key]
            if "Time" in da.dims and "Time" in precip_ref.dims:
                common = da.Time.to_index().intersection(precip_ref.Time.to_index())
                da = da.sel(Time=common)
            if key.endswith("_mask"):
                da = da.fillna(False).astype(bool)
            out[key] = da
        print(f"Using cached high-cloud classification for {sim_name}: {path}")
        return out
    raise FileNotFoundError(
        f"{sim_name}: required shared high-cloud classification cache is missing. "
        f"Run refined_high_cloud_classification_diagnostics.py first. Checked: {checked}"
    )


def block_area_fraction_values(da):
    x = subset_latitude(da.astype(float), TROPICAL_LAT_BOUNDS)
    dims = [dim for dim in ("Time", "latitude", "longitude") if dim in x.dims]
    return block_scalar_values(
        x,
        lambda xb: weighted_mean_dims(xb, dims),
        block_days=CLOUD_TYPE_BLOCK_DAYS,
    )


def run_cloud_response(precip_by_run):
    cache_signature = base_bootstrap_cache_signature(
        "cloud_response",
        extra=dict(
            tropical_lat_bounds=list(TROPICAL_LAT_BOUNDS),
            upper_troposphere_z_bounds_m=list(UPPER_TROPOSPHERE_Z_BOUNDS_M),
            cloud_thresholds=dict(
                **vars(classification_thresholds()),
                cloud_classification_cache_version=CLASSIFICATION_CACHE_VERSION,
                classifier="shared coarse-grid precipitation-regime high-cloud classification",
            ),
        ),
    )
    cached_scalar = load_cached_table("cloud_response_scalar", cache_signature)
    cached_profile = load_cached_table("cloud_profile_response", cache_signature)
    if cached_scalar is not None and cached_profile is not None:
        append_cached_scalar_rows(cached_scalar)
        plot_cloud_profile_diffs(cloud_profile_diffs_from_table(cached_profile))
        record_implemented("cached cloud profile/depth/type block-consistency metrics")
        return

    row_start = len(scalar_metric_rows)
    profile_rows = pd.DataFrame()
    try:
        cf_by_run = {}
        for i in range(len(dss)):
            cf = get_cloud_fraction(dss_zm[i])
            if cf is None:
                raise ValueError(f"cldfrac unavailable for {sim_names[i]}")
            cf_by_run[i] = cf

        profile_by_run = {i: cloud_profile_blocks(cf_by_run[i]) for i in range(len(dss))}
        rng = np.random.default_rng(BOOTSTRAP_SEED + 303)
        profile_diffs = {}
        for comp_name, run_idx, ref_idx, _ in COMPARISONS:
            profile_diffs[comp_name] = bootstrap_field_difference(profile_by_run[run_idx], profile_by_run[ref_idx], rng)
        plot_cloud_profile_diffs(profile_diffs)
        profile_rows = diff_datasets_to_profile_table(profile_diffs)

        record_implemented("vertical cloud-fraction profile block-consistency metrics")
    except Exception as exc:
        record_failure("cloud fraction profile/depth bootstrap", exc)

    if not precip_by_run:
        record_skip("cloud-type bootstrap (precipitation unavailable for classification)")
        return

    try:
        class_by_run = {}
        for i in range(len(dss)):
            class_by_run[i] = cloud_masks_from_cache(sim_names[i], precip_by_run[i])

        for key in (
            "high_cloud_fraction",
            "high_cloud_mask",
            "convective_precipitating_high_cloud_mask",
            "stratiform_precipitating_high_cloud_mask",
            "nonprecipitating_high_cloud_mask",
            "unclassified_high_cloud_mask",
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
        record_implemented("coarse-grid precipitation-regime high-cloud fraction block-consistency metrics")
    except Exception as exc:
        record_failure("cloud-type classification bootstrap", exc)
        record_todo(
            "Cloud-type classification cache",
            "The cloud-type bootstrap requires the shared refined high-cloud classification for every run.",
            "refined_high_cloud_classification_diagnostics.py by running it before this bootstrap script",
        )
        raise

    scalar_rows = pd.DataFrame(scalar_metric_rows[row_start:])
    if len(scalar_rows) > 0:
        write_cached_table("cloud_response_scalar", scalar_rows, cache_signature)
    if len(profile_rows) > 0:
        write_cached_table("cloud_profile_response", profile_rows, cache_signature)


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
        cache_signature = base_bootstrap_cache_signature(
            "conditional_precipitation_states",
            extra=dict(
                tropical_lat_bounds=list(TROPICAL_LAT_BOUNDS),
                precip_strong_threshold=float(PRECIP_STRONG_THRESHOLD),
                precip_weak_threshold=float(PRECIP_WEAK_THRESHOLD),
            ),
        )
        cached_rows = load_cached_table("conditional_precipitation_state_metrics", cache_signature)
        if cached_rows is not None:
            append_cached_scalar_rows(cached_rows)
            record_implemented("cached strong-versus-weak precipitation conditional block-consistency metrics")
            return

        row_start = len(scalar_metric_rows)
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
        write_cached_table(
            "conditional_precipitation_state_metrics",
            pd.DataFrame(scalar_metric_rows[row_start:]),
            cache_signature,
        )
        record_implemented("strong-versus-weak precipitation conditional block-consistency metrics for precip/OLR/PW")
        record_todo(
            "Kelvin-active versus inactive conditional bootstrap",
            "State-dependent changes can be hidden in small means, and Kelvin-active sampling directly tests wave-coupled convection.",
            "equatorial_wave_diagnostics.py by exposing cached Kelvin amplitude indices as import-safe block metrics",
        )
    except Exception as exc:
        record_failure("conditional strong/weak precipitation bootstrap", exc)


# -----------------------------------------------------------------------------
# Convective organisation: cached Lorg/object metrics from organisation script
# -----------------------------------------------------------------------------

ORGANISATION_SOURCE_ORDER = [
    "precip_ge_1mmh",
    "olr_tb241",
    "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud_mask",
    # Sensitivity-only sources from convective_organisation_diagnostics.py.
    "precip_ge_2mmh",
    "precip_ge_8mmh",
    "precip_p95",
    "precip_p99",
    "olr_tb233",
    "olr_cold_p05",
    "high_cloud_mask",
]
ORGANISATION_EXCLUDED_SOURCES = {
    "nonprecipitating_high_cloud_mask",
    "unclassified_high_cloud_mask",
}
ORGANISATION_CONTEXT_SOURCES = {
    "high_cloud_mask",
    "stratiform_precipitating_high_cloud_mask",
}
ORGANISATION_PRIMARY_TRACK_SOURCE = "convective_precipitating_high_cloud_mask"
ORGANISATION_TRACK_SENSITIVITY_SOURCES = {"precip_ge_1mmh", "olr_tb241"}
ORGANISATION_TRACK_CACHE_VERSION = "pyflextrkr_track_persistence_v5_product_contract"


def require_current_organisation_metadata(path, label):
    if not os.path.exists(path):
        raise RuntimeError(
            f"{label} metadata is missing: {path}. "
            "Run convective_organisation_diagnostics.py with the new shared high-cloud classification first."
        )
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    cached_version = metadata.get("cloud_classification_cache_version")
    if cached_version != CLASSIFICATION_CACHE_VERSION:
        raise RuntimeError(
            f"{label} metadata was built with high-cloud classification version "
            f"{cached_version!r}, expected {CLASSIFICATION_CACHE_VERSION!r}. "
            "Run convective_organisation_diagnostics.py before this bootstrap script."
        )
    return metadata


def load_organisation_timeseries():
    if not os.path.exists(ORGANISATION_TIMESERIES_FILE):
        record_skip(
            "convective organisation bootstrap "
            f"({ORGANISATION_TIMESERIES_FILE} not found; run convective_organisation_diagnostics.py first)"
        )
        record_todo(
            "Import-safe organisation metric builder",
            "Organisation changes are scientifically useful only if object metrics can be tested against internal variability without rerunning every figure workflow.",
            "convective_organisation_diagnostics.py by exposing a reusable metric-table builder/cache interface",
        )
        return None

    require_current_organisation_metadata(ORGANISATION_METRIC_METADATA_FILE, "organisation metric")
    df = pd.read_csv(ORGANISATION_TIMESERIES_FILE)
    required = {"source", "source_label", "simulation", "time_days"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"organisation timeseries is missing columns: {missing}")

    df = df[df["simulation"].isin(sim_names)].copy()
    for col in (
        ["time_days", "tropical_mean_precip", "convective_area_fraction"]
        + [metric for metric, _ in ORGANISATION_METRIC_SPECS]
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_organisation_tracks():
    if not os.path.exists(ORGANISATION_TRACKS_FILE):
        raise FileNotFoundError(
            "Required convective-organisation PyFLEXTRKR track product is missing: "
            f"{ORGANISATION_TRACKS_FILE}. Run convective_organisation_diagnostics.py first."
        )

    metadata = require_current_organisation_metadata(
        ORGANISATION_TRACK_METADATA_FILE, "organisation track"
    )
    if metadata.get("cache_version") != ORGANISATION_TRACK_CACHE_VERSION:
        raise RuntimeError(
            "Organisation track metadata does not describe the current "
            f"PyFLEXTRKR product ({metadata.get('cache_version')!r} != "
            f"{ORGANISATION_TRACK_CACHE_VERSION!r}). Run "
            "convective_organisation_diagnostics.py first."
        )
    if metadata.get("primary_track_source") != ORGANISATION_PRIMARY_TRACK_SOURCE:
        raise RuntimeError(
            "Organisation track metadata does not use the classifier's "
            f"primary convective source {ORGANISATION_PRIMARY_TRACK_SOURCE!r}."
        )
    df = pd.read_csv(ORGANISATION_TRACKS_FILE)
    required = {"source", "source_label", "simulation", "track_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"organisation track table is missing columns: {missing}")

    df = df[df["simulation"].isin(sim_names)].copy()
    for col in ["starts_as_split", "ends_as_merge"]:
        if col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(float)
            else:
                text = df[col].astype(str).str.lower()
                df[col] = np.where(text.isin(["true", "1", "yes"]), 1.0, np.where(text.isin(["false", "0", "no"]), 0.0, np.nan))
    if "starts_as_split" in df.columns and "ends_as_merge" in df.columns:
        df["complete_lifecycle"] = np.where(
            np.isfinite(df["starts_as_split"]) & np.isfinite(df["ends_as_merge"]),
            ((df["starts_as_split"] == 0.0) & (df["ends_as_merge"] == 0.0)).astype(float),
            np.nan,
        )
    for col in [metric for metric, _ in ORGANISATION_TRACK_METRIC_SPECS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def organisation_source_label(df, source):
    sub = df[df["source"] == source]
    if len(sub) == 0 or "source_label" not in sub.columns:
        return source
    vals = sub["source_label"].dropna().unique()
    return str(vals[0]) if len(vals) else source


def dataframe_time_block_values(
    sub,
    metric,
    block_days=ORGANISATION_BLOCK_DAYS,
    condition_key=None,
    bin_edges=None,
    bin_index=None,
):
    sub = sub.sort_values("time_days")
    sub = sub[np.isfinite(sub["time_days"].values)]
    if len(sub) == 0:
        raise ValueError(f"no finite times for {metric}")

    unique_times = np.unique(np.asarray(sub["time_days"].values, dtype=float))
    dt_values = np.diff(unique_times)
    dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0.0)]
    dt_days = float(np.nanmedian(dt_values)) if dt_values.size else dt / 24.0
    block_len = max(1, int(round(block_days / dt_days)))
    nblocks = len(sub) // block_len
    if nblocks < MIN_BLOCKS:
        raise ValueError(
            f"Only {nblocks} complete organisation blocks for {metric}; "
            f"need at least {MIN_BLOCKS}."
        )

    vals = []
    for bi in range(nblocks):
        block = sub.iloc[bi * block_len:(bi + 1) * block_len]
        if condition_key is not None:
            lo = bin_edges[bin_index]
            hi = bin_edges[bin_index + 1]
            cond = block[condition_key]
            if bin_index == len(bin_edges) - 2:
                keep = (cond >= lo) & (cond <= hi)
            else:
                keep = (cond >= lo) & (cond < hi)
            block = block[keep]
            if len(block) < ORGANISATION_MIN_CONDITION_SAMPLES:
                vals.append(np.nan)
                continue
        values = np.asarray(block[metric].values, dtype=float)
        values = values[np.isfinite(values)]
        vals.append(np.nanmean(values) if values.size else np.nan)

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < MIN_BLOCKS:
        raise ValueError(
            f"Only {vals.size} finite organisation blocks for {metric}; "
            f"need at least {MIN_BLOCKS}."
        )
    return vals


def organisation_condition_edges(df, source, condition_key):
    sub = df[df["source"] == source]
    values = np.asarray(sub[condition_key].values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < ORGANISATION_CONDITION_NBINS:
        return None
    edges = np.unique(np.nanpercentile(values, np.linspace(0.0, 100.0, ORGANISATION_CONDITION_NBINS + 1)))
    return edges if edges.size >= 3 else None


def ordered_organisation_sources(df):
    present = set(df["source"].dropna().unique())
    usable = present - ORGANISATION_EXCLUDED_SOURCES
    ordered = [source for source in ORGANISATION_SOURCE_ORDER if source in usable]
    ordered.extend(sorted(usable - set(ordered)))
    return ordered


def note_skipped_organisation_sources(df):
    present = set(df["source"].dropna().unique())
    skipped = sorted(present & ORGANISATION_EXCLUDED_SOURCES)
    if skipped:
        record_skip(
            "organisation/Lorg metrics for passive high-cloud classes "
            f"({', '.join(skipped)} are cloud-regime diagnostics, not convective-organisation diagnostics)"
        )


def organisation_source_role(source):
    if source == ORGANISATION_PRIMARY_TRACK_SOURCE:
        return "primary_convective_objects"
    if source in ORGANISATION_TRACK_SENSITIVITY_SOURCES:
        return "object_definition_sensitivity"
    if source in ORGANISATION_CONTEXT_SOURCES:
        return "cloud_context"
    return "convective_proxy"


def plot_organisation_bootstrap_summary(org_df, org_source_labels):
    if not should_run_figure(5) or len(org_df) == 0:
        return

    display_metrics = [
        ("lorg_km", "Lorg (km)"),
        ("object_count", "Object count"),
        ("convective_area_fraction", "Convective area fraction"),
        ("largest_object_fraction", "Largest-object fraction"),
    ]
    plot_df = org_df[
        (org_df["category"] == "convective_organisation")
        & (org_df["metric"].isin([metric for metric, _ in display_metrics]))
    ].copy()
    if len(plot_df) == 0:
        return

    sources = [source for source in org_source_labels if source in set(plot_df["diagnostic"])]
    if not sources:
        sources = sorted(plot_df["diagnostic"].unique())

    fig, axes = plt.subplots(
        len(display_metrics),
        1,
        figsize=(8.2, max(6.0, 1.85 * len(display_metrics) + 0.34 * len(sources))),
        constrained_layout=True,
        squeeze=False,
    )
    offsets = {"u_minus_control": -0.24, "T_minus_control": 0.0, "u_minus_T": 0.24}
    colors_by_comp = {"u_minus_control": "b", "T_minus_control": "r", "u_minus_T": "0.2"}
    ybase = np.arange(len(sources), dtype=float)

    for ax, (metric, label) in zip(axes.ravel(), display_metrics):
        sub_metric = plot_df[plot_df["metric"] == metric]
        for comp_name, _, _, comp_label in COMPARISONS:
            sub = sub_metric[sub_metric["comparison"] == comp_name]
            if len(sub) == 0:
                continue
            yvals = []
            xvals = []
            xerr_low = []
            xerr_high = []
            robust = []
            for si, source in enumerate(sources):
                row = sub[sub["diagnostic"] == source]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                x = float(row["mean_response"])
                yvals.append(ybase[si] + offsets[comp_name])
                xvals.append(x)
                xerr_low.append(max(0.0, x - float(row["ci_low"])))
                xerr_high.append(max(0.0, float(row["ci_high"]) - x))
                robust.append(bool(row["robust"]))
            if not xvals:
                continue
            color = colors_by_comp.get(comp_name, "k")
            ax.errorbar(
                xvals,
                yvals,
                xerr=[xerr_low, xerr_high],
                fmt="o",
                ms=4.0,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2,
                label=comp_label,
            )
            for x, y, is_robust in zip(xvals, yvals, robust):
                if not is_robust:
                    ax.plot(x, y, marker="o", ms=4.5, mfc="white", mec=color, mew=1.0, linestyle="none")
        ax.axvline(0.0, color="0.55", lw=0.8)
        ax.set_yticks(ybase)
        ax.set_yticklabels([org_source_labels.get(source, source) for source in sources])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlabel(f"{label} response")
    axes.ravel()[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Convective/object organisation block-consistency response; filled points pass the chunk-consistency rule")
    figsave.save(fig, "organisation_bootstrap_summary", number=5)
    plt.close(fig)


def run_convective_organisation_bootstrap():
    try:
        tracks = load_organisation_tracks()
        cache_signature = base_bootstrap_cache_signature(
            "convective_organisation",
            extra=dict(
                organisation_timeseries_file=ORGANISATION_TIMESERIES_FILE,
                organisation_tracks_file=ORGANISATION_TRACKS_FILE,
                organisation_block_days=float(ORGANISATION_BLOCK_DAYS),
                condition_nbins=int(ORGANISATION_CONDITION_NBINS),
                min_condition_samples=int(ORGANISATION_MIN_CONDITION_SAMPLES),
                metric_specs=jsonable_value(ORGANISATION_METRIC_SPECS),
                track_metric_specs=jsonable_value(ORGANISATION_TRACK_METRIC_SPECS),
                primary_track_source=ORGANISATION_PRIMARY_TRACK_SOURCE,
                track_sensitivity_sources=sorted(ORGANISATION_TRACK_SENSITIVITY_SOURCES),
                organisation_track_cache_version=ORGANISATION_TRACK_CACHE_VERSION,
                excluded_sources=sorted(ORGANISATION_EXCLUDED_SOURCES),
                cloud_classification_cache_version=CLASSIFICATION_CACHE_VERSION,
            ),
        )
        cached_rows = load_cached_table("organisation_bootstrap_metrics", cache_signature)
        if cached_rows is not None:
            append_cached_scalar_rows(cached_rows)
            save_table(cached_rows, "organisation_bootstrap_metrics.csv")
            source_labels = (
                cached_rows.dropna(subset=["diagnostic", "source_label"])
                .drop_duplicates("diagnostic")
                .set_index("diagnostic")["source_label"]
                .to_dict()
                if {"diagnostic", "source_label"}.issubset(cached_rows.columns)
                else {}
            )
            plot_organisation_bootstrap_summary(cached_rows, source_labels)
            record_implemented("cached convective-organisation block-consistency checks")
            return

        org = load_organisation_timeseries()
        if org is None or len(org) == 0:
            return

        note_skipped_organisation_sources(org)
        sources = ordered_organisation_sources(org)
        source_labels = {source: organisation_source_label(org, source) for source in sources}
        org_row_start = len(scalar_metric_rows)
        skipped_pairs = 0

        for source in sources:
            sub_source = org[org["source"] == source]
            for metric, units in ORGANISATION_METRIC_SPECS:
                if metric not in sub_source.columns:
                    continue
                values_by_run = {}
                for i, sim_name in enumerate(sim_names):
                    sub = sub_source[sub_source["simulation"] == sim_name]
                    try:
                        values_by_run[i] = dataframe_time_block_values(sub, metric)
                    except ValueError:
                        skipped_pairs += 1
                        values_by_run = None
                        break
                if values_by_run is None:
                    continue
                append_bootstrap_metric_rows(
                    scalar_metric_rows,
                    values_by_run,
                    "convective_organisation",
                    source,
                    metric,
                    units=units,
                    extra={
                        "source_label": source_labels[source],
                        "organisation_role": organisation_source_role(source),
                    },
                )

            for condition_key, condition_label in (
                ("tropical_mean_precip", "fixed_tropical_mean_precip"),
                ("convective_area_fraction", "fixed_convective_area_fraction"),
            ):
                if condition_key not in sub_source.columns:
                    continue
                edges = organisation_condition_edges(org, source, condition_key)
                if edges is None:
                    continue
                for bin_index in range(len(edges) - 1):
                    values_by_run = {}
                    for i, sim_name in enumerate(sim_names):
                        sub = sub_source[sub_source["simulation"] == sim_name]
                        try:
                            values_by_run[i] = dataframe_time_block_values(
                                sub,
                                "lorg_km",
                                condition_key=condition_key,
                                bin_edges=edges,
                                bin_index=bin_index,
                            )
                        except ValueError:
                            skipped_pairs += 1
                            values_by_run = None
                            break
                    if values_by_run is None:
                        continue
                    append_bootstrap_metric_rows(
                        scalar_metric_rows,
                        values_by_run,
                        "convective_organisation_conditioned",
                        source,
                        "lorg_km",
                        units="km",
                        extra={
                            "source_label": source_labels[source],
                            "organisation_role": organisation_source_role(source),
                            "condition": condition_label,
                            "bin_index": bin_index,
                            "bin_low": float(edges[bin_index]),
                            "bin_high": float(edges[bin_index + 1]),
                            "bin_center": float(0.5 * (edges[bin_index] + edges[bin_index + 1])),
                        },
                    )

        if tracks is not None and len(tracks) > 0:
            note_skipped_organisation_sources(tracks)
            for source in ordered_organisation_sources(tracks):
                source_labels.setdefault(source, organisation_source_label(tracks, source))
                sub_source = tracks[tracks["source"] == source]
                for metric, units in ORGANISATION_TRACK_METRIC_SPECS:
                    if metric not in sub_source.columns:
                        continue
                    values_by_run = {}
                    for i, sim_name in enumerate(sim_names):
                        vals = np.asarray(sub_source.loc[sub_source["simulation"] == sim_name, metric].values, dtype=float)
                        vals = vals[np.isfinite(vals)]
                        if vals.size < MIN_BLOCKS:
                            skipped_pairs += 1
                            values_by_run = None
                            break
                        values_by_run[i] = vals
                    if values_by_run is None:
                        continue
                    append_bootstrap_metric_rows(
                        scalar_metric_rows,
                        values_by_run,
                        "convective_organisation_persistence",
                        source,
                        metric,
                        units=units,
                        extra={
                            "source_label": source_labels[source],
                            "organisation_role": organisation_source_role(source),
                        },
                    )

        org_df = pd.DataFrame(scalar_metric_rows[org_row_start:])
        if len(org_df) == 0:
            record_skip("convective organisation bootstrap (no source had enough finite time blocks)")
            return
        save_table(org_df, "organisation_bootstrap_metrics.csv")
        plot_organisation_bootstrap_summary(org_df, source_labels)
        write_cached_table("organisation_bootstrap_metrics", org_df, cache_signature)
        message = (
            "convective-organisation block-consistency checks for Lorg/object metrics, PyFLEXTRKR-method "
            "track-persistence metrics, plus Lorg conditioned on tropical-mean precipitation and convective area fraction"
        )
        if skipped_pairs:
            message += f" ({skipped_pairs} source/metric/bin entries skipped for insufficient blocks)"
        record_implemented(message)
    except Exception as exc:
        record_failure("convective organisation bootstrap", exc)
        raise


# -----------------------------------------------------------------------------
# Wave-mediated pathway: integrated CCEW wave-box variance
# -----------------------------------------------------------------------------

def wave_label(wave):
    return WAVE_BOXES[wave]["label"]


def wave_cache_dir():
    path = diagnostic_cache_dir("significance_bootstrap_wave_cache", root_dir=root_dir)
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "-").replace(".", "p").replace("-", "m")


def get_surface_convection_field(ds, field_name, ref=None):
    pr = get_precip_rate(ds)
    if field_name == "Precip":
        return pr
    aliases = {
        "OLR": ("olrtoa", "olr", "OLR"),
        "PW": ("precipw", "pw", "prw"),
        "CAPE": ("cape",),
    }
    for name in aliases.get(field_name, (field_name,)):
        if name in ds.data_vars:
            out = finite_or_nan(ds[name])
            if ref is not None and "Time" in out.dims and "Time" in ref.dims:
                out = out.sel(Time=ref.Time)
            elif "Time" in out.dims and "Time" in pr.dims:
                out = out.sel(Time=pr.Time)
            out.name = field_name
            return out
    raise ValueError(f"{field_name} unavailable")


def get_wave_filter_field(ds, field_name):
    pr = get_precip_rate(ds)
    if field_name in ("Precip", "OLR", "PW", "CAPE"):
        return get_surface_convection_field(ds, field_name, ref=pr)

    pressure_fields = [
        ("w", None),
        ("u", "uzonal_isobaric"),
        ("v", "umeridional_isobaric"),
        ("T", "t_isobaric"),
        ("qv", "qv_isobaric"),
        ("RH", "rh_isobaric"),
    ]
    for prefix, var in pressure_fields:
        if not field_name.startswith(prefix):
            continue
        level_text = field_name[len(prefix):]
        if not level_text:
            continue
        plev = float(level_text)
        if var is None:
            out = vertical_velocity_at_pressure(ds, plev).sel(Time=pr.Time)
        else:
            if var not in ds.data_vars:
                raise ValueError(f"{var} unavailable")
            out = finite_or_nan(ds[var]).sel(pres=plev, method="nearest").sel(Time=pr.Time)
        out.name = field_name
        return out

    raise ValueError(f"Unsupported wave field {field_name}")


def project_wave_latitude(da, wave_name):
    projection = WAVE_LATITUDE_PROJECTION[wave_name]
    lo, hi = WAVE_FILTER_LAT_BOUNDS
    if projection == "symmetric":
        x = subset_between(da, "latitude", (lo, hi))
        return weighted_mean_dims(x, ["latitude"]).transpose("Time", "longitude")
    if projection == "direct":
        x = subset_between(da, "latitude", (lo, hi))
        return x.mean(dim="latitude", skipna=True).transpose("Time", "longitude")

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
        record_skip("wave-mediated block-consistency diagnostics (waveFilter.kf_filter unavailable)")
        record_todo(
            "CCEW wave-box bootstrap",
            "Integrated wave-box variance tests whether nudged responses are tied to organised equatorial wave-coupled convection.",
            "equatorial_wave_diagnostics.py or waveFilter import configuration",
        )
        return

    try:
        cache_signature = base_bootstrap_cache_signature(
            "wave_pathway",
            extra=dict(
                wave_fields=list(WAVE_FIELDS),
                wave_boxes=jsonable_value(WAVE_BOXES),
                latitude_projection=jsonable_value(WAVE_LATITUDE_PROJECTION),
                wave_filter_lat_bounds=list(WAVE_FILTER_LAT_BOUNDS),
            ),
        )
        cached_rows = load_cached_table("wave_box_variance_bootstrap_metrics", cache_signature)
        if cached_rows is not None:
            append_cached_scalar_rows(cached_rows)
            save_table(cached_rows, "wave_box_variance_bootstrap_metrics.csv")
            plot_wave_summary(cached_rows)
            record_implemented("cached integrated CCEW wave-power and variance-fraction block-consistency metrics")
            return

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
        write_cached_table("wave_box_variance_bootstrap_metrics", wave_df, cache_signature)
        record_implemented("integrated Kelvin/CCEW-filtered Precip, OLR, and PW wave-power and variance-fraction block-consistency metrics")
    except Exception as exc:
        record_failure("wave-mediated pathway bootstrap", exc)


def plot_wave_summary(wave_df):
    if not should_run_figure(4) or len(wave_df) == 0:
        return
    sub = wave_df[
        (wave_df["metric"].isin(["wave_variance", "wave_variance_fraction"]))
        & (wave_df["comparison"].isin(["u_minus_control", "T_minus_control"]))
    ].copy()
    if len(sub) == 0:
        return
    waves = list(WAVE_BOXES)
    fields = [field for field in WAVE_FIELDS if field in set(sub["diagnostic"])]
    metrics = [
        ("wave_variance", "Wave-power response"),
        ("wave_variance_fraction", "Wave-fraction response"),
    ]
    fig, axes = plt.subplots(
        len(fields), len(metrics),
        figsize=(7.2 * len(metrics), 3.0 * len(fields)),
        constrained_layout=True,
        sharex=True,
        squeeze=False,
    )
    offsets = {"u_minus_control": -0.15, "T_minus_control": 0.15}
    colors_by_comp = {"u_minus_control": "b", "T_minus_control": "r"}
    x = np.arange(len(waves))
    for ri, field in enumerate(fields):
        for ci, (metric, metric_label) in enumerate(metrics):
            ax = axes[ri, ci]
            metric_sub = sub[(sub["diagnostic"] == field) & (sub["metric"] == metric)]
            for comp in ("u_minus_control", "T_minus_control"):
                y = []
                err_low = []
                err_high = []
                for wave in waves:
                    row = metric_sub[(metric_sub["comparison"] == comp) & (metric_sub["wave"] == wave)]
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
            ax.set_ylabel(f"{field}\n{metric_label}")
            ax.grid(axis="y", alpha=0.2)
            if ri == 0:
                ax.set_title(metric_label)
            if ri == len(fields) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([WAVE_BOXES[w]["label"] for w in waves], rotation=25, ha="right")
            if ri == 0 and ci == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("CCEW wave-box block-consistency response: power changes relative to control and variance fractions")
    figsave.save(fig, "wave_box_variance_fraction_bootstrap_summary", number=4)
    plt.close(fig)


def wave_profile_field_name(field_spec, level_hpa):
    return f"{field_spec['prefix']}{float(level_hpa):g}".replace(".", "p")


def plot_vertical_wave_power_profiles(profile_df):
    if not should_run_figure(7) or len(profile_df) == 0:
        return
    plot_df = profile_df[
        (profile_df["comparison"].isin(["u_minus_control", "T_minus_control"]))
        & (profile_df["metric"] == "wave_variance")
    ].copy()
    if len(plot_df) == 0:
        return
    fields = [spec["name"] for spec in WAVE_PROFILE_FIELDS if spec["name"] in set(plot_df["field"])]
    waves = [wave for wave in WAVE_PROFILE_WAVES if wave in set(plot_df["wave"])]
    fig, axes = plt.subplots(
        len(fields), len(waves),
        figsize=(3.4 * len(waves), 3.0 * len(fields)),
        constrained_layout=True,
        squeeze=False,
        sharey=True,
    )
    colors_by_comp = {"u_minus_control": "b", "T_minus_control": "r"}
    for fi, field in enumerate(fields):
        for wi, wave in enumerate(waves):
            ax = axes[fi, wi]
            for comp in ("u_minus_control", "T_minus_control"):
                sub = plot_df[(plot_df["field"] == field) & (plot_df["wave"] == wave) & (plot_df["comparison"] == comp)].sort_values("pressure_hpa")
                if len(sub) == 0:
                    continue
                color = colors_by_comp[comp]
                label = next(text for name, _, _, text in COMPARISONS if name == comp)
                ax.plot(sub["mean_response"], sub["pressure_hpa"], color=color, marker="o", ms=3, lw=1.1, label=label)
                ax.fill_betweenx(sub["pressure_hpa"], sub["ci_low"], sub["ci_high"], color=color, alpha=0.12)
            ax.axvline(0.0, color="0.6", lw=0.8)
            ax.invert_yaxis()
            ax.set_yscale("log")
            ax.set_ylim(max(WAVE_PROFILE_LEVELS_HPA), min(WAVE_PROFILE_LEVELS_HPA))
            ax.grid(alpha=0.18)
            if fi == 0:
                ax.set_title(wave_label(wave))
            if wi == 0:
                ax.set_ylabel(f"{field}\nPressure (hPa)")
            if fi == len(fields) - 1:
                ax.set_xlabel("Wave-power response")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Vertical wave-power profile block-consistency response relative to control")
    figsave.save(fig, "vertical_wave_power_profile_bootstrap_response", number=7)
    plt.close(fig)


def run_vertical_wave_power_profile_bootstrap():
    if kf_filter is None:
        record_skip("vertical wave-power profile block-consistency diagnostics (waveFilter.kf_filter unavailable)")
        return
    try:
        cache_signature = base_bootstrap_cache_signature(
            "vertical_wave_power_profile",
            extra=dict(
                waves=list(WAVE_PROFILE_WAVES),
                fields=jsonable_value(WAVE_PROFILE_FIELDS),
                levels_hpa=jsonable_value(WAVE_PROFILE_LEVELS_HPA),
                wave_filter_lat_bounds=list(WAVE_FILTER_LAT_BOUNDS),
            ),
        )
        cached_rows = load_cached_table("vertical_wave_power_profile_bootstrap_metrics", cache_signature)
        if cached_rows is not None:
            append_cached_scalar_rows(cached_rows)
            save_table(cached_rows, "vertical_wave_power_profile_bootstrap_metrics.csv")
            plot_vertical_wave_power_profiles(cached_rows)
            record_implemented("cached vertical wave-power profile block-consistency metrics")
            return

        rows = []
        for wave_name in WAVE_PROFILE_WAVES:
            for field_spec in WAVE_PROFILE_FIELDS:
                for level_hpa in WAVE_PROFILE_LEVELS_HPA:
                    field_name = wave_profile_field_name(field_spec, level_hpa)
                    try:
                        values_by_run = {}
                        for i in range(len(dss)):
                            raw = get_wave_filter_field(dss[i], field_name)
                            projected = project_wave_latitude(raw, wave_name)
                            filtered = filter_projected_wave(projected, wave_name, i, field_name)
                            values_by_run[i] = block_variance_values(filtered)
                    except Exception as exc:
                        record_skip(f"vertical wave power {field_name} {wave_name} ({exc})")
                        continue

                    before = len(scalar_metric_rows)
                    append_bootstrap_metric_rows(
                        scalar_metric_rows,
                        values_by_run,
                        "vertical_wave_power_profile",
                        field_spec["name"],
                        "wave_variance",
                        units=f"{field_spec['units']} variance",
                        extra={
                            "field": field_spec["name"],
                            "source_field": field_name,
                            "pressure_hpa": float(level_hpa),
                            "wave": wave_name,
                            "wave_label": wave_label(wave_name),
                            "latitude_projection": WAVE_LATITUDE_PROJECTION[wave_name],
                        },
                    )
                    rows.extend(scalar_metric_rows[before:])
        if rows:
            df = pd.DataFrame(rows)
            save_table(df, "vertical_wave_power_profile_bootstrap_metrics.csv")
            plot_vertical_wave_power_profiles(df)
            write_cached_table("vertical_wave_power_profile_bootstrap_metrics", df, cache_signature)
            record_implemented("vertical wave-power profile block-consistency metrics for selected u/T/w pressure levels and dominant wave boxes")
    except Exception as exc:
        record_failure("vertical wave-power profile bootstrap", exc)


def lag_correlation_for_lags(x, y, lag_steps):
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    out = np.full(lag_steps.size, np.nan)
    for j, lag in enumerate(lag_steps):
        if lag < 0:
            xs, ys = xv[-lag:], yv[:lag]
        elif lag > 0:
            xs, ys = xv[:-lag], yv[lag:]
        else:
            xs, ys = xv, yv
        ok = np.isfinite(xs) & np.isfinite(ys)
        if np.count_nonzero(ok) >= 4:
            out[j] = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
    return out


def block_lag_correlation_values(wave_amp, target_ts):
    wave_amp, target_ts = xr.align(wave_amp, target_ts, join="inner")
    dt_days_here = time_step_days(wave_amp)
    max_steps = max(1, int(round(WAVE_LAG_MAX_DAYS / dt_days_here)))
    lag_steps = np.arange(-max_steps, max_steps + 1)
    lag_days = lag_steps * dt_days_here
    blocks = []
    for slc in block_slices(wave_amp, block_days=WAVE_LAG_BLOCK_DAYS):
        corr = lag_correlation_for_lags(
            wave_amp.isel(Time=slc).values,
            target_ts.isel(Time=slc).values,
            lag_steps,
        )
        blocks.append(
            xr.DataArray(corr, dims=("lag_days",), coords={"lag_days": lag_days}, name="lag_correlation")
        )
    return xr.concat(blocks, dim=xr.IndexVariable("block", np.arange(len(blocks))))


def plot_wave_lag_correlation_differences(lag_diffs):
    if not should_run_figure(8) or not lag_diffs:
        return
    waves = [wave for wave in WAVE_LAG_WAVES if any(key[0] == wave for key in lag_diffs)]
    targets = [target for target in WAVE_LAG_TARGET_FIELDS if any(key[1] == target for key in lag_diffs)]
    fig, axes = plt.subplots(
        len(waves), len(targets),
        figsize=(3.7 * len(targets), 2.5 * len(waves)),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    colors_by_comp = {"u_minus_control": "b", "T_minus_control": "r", "u_minus_T": "0.2"}
    for wi, wave in enumerate(waves):
        for ti, target in enumerate(targets):
            ax = axes[wi, ti]
            diffs = lag_diffs.get((wave, target), {})
            for comp_name, ds in diffs.items():
                label = next(label for name, _, _, label in COMPARISONS if name == comp_name)
                color = colors_by_comp.get(comp_name, "k")
                lag = ds.lag_days.values
                ax.plot(lag, ds.mean_response.values, color=color, lw=1.1, label=label)
                ax.fill_between(lag, ds.ci_low.values, ds.ci_high.values, color=color, alpha=0.12)
            ax.axhline(0.0, color="0.6", lw=0.7)
            ax.axvline(0.0, color="0.6", lw=0.7)
            ax.grid(alpha=0.18)
            if wi == 0:
                ax.set_title(target)
            if ti == 0:
                ax.set_ylabel(f"{wave_label(wave)}\nCorrelation response")
            if wi == len(waves) - 1:
                ax.set_xlabel("Lag (days; + wave leads)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Wave-amplitude lag-correlation block-consistency response")
    figsave.save(fig, "wave_lag_correlation_bootstrap_response", number=8)
    plt.close(fig)


def run_wave_lag_correlation_bootstrap():
    if kf_filter is None:
        record_skip("wave lag-correlation block-consistency diagnostics (waveFilter.kf_filter unavailable)")
        return
    try:
        cache_signature = base_bootstrap_cache_signature(
            "wave_lag_correlation",
            extra=dict(
                waves=list(WAVE_LAG_WAVES),
                targets=list(WAVE_LAG_TARGET_FIELDS),
                max_days=float(WAVE_LAG_MAX_DAYS),
                block_days=float(WAVE_LAG_BLOCK_DAYS),
                wave_filter_lat_bounds=list(WAVE_FILTER_LAT_BOUNDS),
            ),
        )
        cached_rows = load_cached_table("wave_lag_correlation_bootstrap_metrics", cache_signature)
        if cached_rows is not None:
            save_table(cached_rows, "wave_lag_correlation_bootstrap_metrics.csv")
            plot_wave_lag_correlation_differences(wave_lag_diffs_from_table(cached_rows))
            record_implemented("cached wave-amplitude lag-correlation block-consistency metrics")
            return

        rows = []
        lag_diffs = {}
        rng = np.random.default_rng(BOOTSTRAP_SEED + 444)
        for wave_name in WAVE_LAG_WAVES:
            amp_by_run = {}
            for i in range(len(dss)):
                raw = get_wave_filter_field(dss[i], "Precip")
                projected = project_wave_latitude(raw, wave_name)
                filtered = filter_projected_wave(projected, wave_name, i, "Precip")
                amp_by_run[i] = np.sqrt((filtered ** 2).mean(dim="longitude", skipna=True))
                amp_by_run[i].name = "wave_amplitude"

            for target_name in WAVE_LAG_TARGET_FIELDS:
                try:
                    blocks_by_run = {}
                    for i in range(len(dss)):
                        target_raw = get_wave_filter_field(dss[i], target_name)
                        target_projected = project_wave_latitude(target_raw, wave_name)
                        target_ts = target_projected.mean(dim="longitude", skipna=True)
                        blocks_by_run[i] = block_lag_correlation_values(amp_by_run[i], target_ts)
                except Exception as exc:
                    record_skip(f"wave lag correlation {wave_name} {target_name} ({exc})")
                    continue

                diffs = {}
                for comp_name, run_idx, ref_idx, comp_label in COMPARISONS:
                    ds = bootstrap_field_difference(blocks_by_run[run_idx], blocks_by_run[ref_idx], rng)
                    diffs[comp_name] = ds
                    for lag_index, lag_day in enumerate(ds.lag_days.values):
                        rows.append(
                            {
                                "wave": wave_name,
                                "wave_label": wave_label(wave_name),
                                "target": target_name,
                                "comparison": comp_name,
                                "comparison_label": comp_label,
                                "lag_days": float(lag_day),
                                "mean_response": float(ds.mean_response.isel(lag_days=lag_index).values),
                                "ci_low": float(ds.ci_low.isel(lag_days=lag_index).values),
                                "ci_high": float(ds.ci_high.isel(lag_days=lag_index).values),
                                "sign_robustness": float(ds.sign_robustness.isel(lag_days=lag_index).values),
                                "robust": bool(ds.robust.isel(lag_days=lag_index).values),
                            }
                        )
                lag_diffs[(wave_name, target_name)] = diffs
        if rows:
            rows_df = pd.DataFrame(rows)
            save_table(rows_df, "wave_lag_correlation_bootstrap_metrics.csv")
            plot_wave_lag_correlation_differences(lag_diffs)
            write_cached_table("wave_lag_correlation_bootstrap_metrics", rows_df, cache_signature)
            record_implemented("wave-amplitude lag-correlation block-consistency metrics for precipitation/OLR/PW targets")
    except Exception as exc:
        record_failure("wave lag-correlation bootstrap", exc)


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


def pressure_layer_weights(pres):
    if isinstance(pres, xr.DataArray):
        pres_values = np.asarray(pres.values, dtype=float)
        pres_coord = pres_values
    else:
        pres_values = np.asarray(pres, dtype=float)
        pres_coord = pres_values

    if np.any(pres_values <= 0):
        raise ValueError("Pressure values must be positive before taking log(p)")

    if pres_values.size == 1:
        weights = np.ones_like(pres_values)
    else:
        lnp = np.log(pres_values)
        bounds = np.empty(pres_values.size + 1)
        bounds[1:-1] = 0.5 * (lnp[:-1] + lnp[1:])
        bounds[0] = lnp[0] - 0.5 * (lnp[1] - lnp[0])
        bounds[-1] = lnp[-1] + 0.5 * (lnp[-1] - lnp[-2])
        weights = np.abs(np.diff(bounds))

    return xr.DataArray(
        weights,
        dims=("pres",),
        coords={"pres": pres_coord},
        name="dlnp_weight",
    )


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
        cache_signature = base_bootstrap_cache_signature(
            "matched_pattern_attribution",
            extra=dict(
                domains=jsonable_value(MATCHED_PATTERN_DOMAINS),
                source_variable="t_isobaric",
            ),
        )
        cached_rows = load_cached_table("matched_pattern_bootstrap_metrics", cache_signature)
        if cached_rows is not None:
            save_table(cached_rows, "matched_pattern_bootstrap_metrics.csv")
            record_implemented("cached matched-pattern projection bootstrap")
            return cached_rows

        blocks_c = block_latpres_field(dss_zm[0], "t_isobaric", parity="even")
        blocks_u = block_latpres_field(dss_zm[1], "t_isobaric", parity="even")
        blocks_T = block_latpres_field(dss_zm[2], "t_isobaric", parity="even")
        rows = []
        for domain_name, domain in MATCHED_PATTERN_DOMAINS.items():
            rows.extend(bootstrap_matched_pattern(blocks_c, blocks_u, blocks_T, domain_name, domain))
        df = pd.DataFrame(rows)
        save_table(df, "matched_pattern_bootstrap_metrics.csv")
        write_cached_table("matched_pattern_bootstrap_metrics", df, cache_signature)
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
    print("\nChunk-consistent scalar responses (central resampling interval excludes zero and sign consistency is high):")
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
        org_robust = metrics_df[
            (metrics_df["category"].isin([
                "convective_organisation",
                "convective_organisation_conditioned",
                "convective_organisation_persistence",
            ]))
            & (metrics_df["comparison"].isin(["u_minus_control", "T_minus_control"]))
            & (metrics_df.get("robust", False) == True)
        ]
        if len(org_robust) > 0:
            print("  Robust organisation/object/persistence changes are present; use organisation_bootstrap_metrics.csv to check Lorg, object metrics, and PyFLEXTRKR-method track lifetimes.")
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
    print("Block-resampling consistency summary")
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
    print("  Use bootstrap_scalar_metrics.csv for scalar robustness, basic_convection_pdf_bootstrap_response.csv for PR/OLR/PW/CAPE distribution changes, wavenumber_variance_fraction_bootstrap_response.csv for planetary-scale organisation/noise checks, organisation_bootstrap_metrics.csv for Lorg/object robustness, the lat-height figures for coherent structures rather than isolated gridpoints, wave_box_variance_bootstrap_metrics.csv plus vertical_wave_power_profile_bootstrap_metrics.csv and wave_lag_correlation_bootstrap_metrics.csv for organised wave-mediated changes, and matched_pattern_bootstrap_metrics.csv for thermal-pattern attribution.")
    print("  Robust U-control changes in zonal wind or CCEW variance that are absent from T-control point toward a wind-driven pathway; robust shared temperature/N2/cloud-depth structures with high matched-pattern correlation point toward a temperature-driven pathway.")


# -----------------------------------------------------------------------------
# Main top-level workflow
# -----------------------------------------------------------------------------

print("\nRunning block-resampling consistency diagnostics")
run_mean_state_pathway()
precip_by_run = run_precipitation_response()
run_basic_convection_pdf_bootstrap()
run_wavenumber_variance_bootstrap()
run_cloud_response(precip_by_run)
run_conditional_precipitation_states(precip_by_run)
run_convective_organisation_bootstrap()
run_wave_pathway()
run_vertical_wave_power_profile_bootstrap()
run_wave_lag_correlation_bootstrap()
matched_pattern_df = run_matched_pattern_attribution()

metrics_df = pd.DataFrame(scalar_metric_rows)
save_table(metrics_df, "bootstrap_scalar_metrics.csv")
print_final_summary(metrics_df, matched_pattern_df)
