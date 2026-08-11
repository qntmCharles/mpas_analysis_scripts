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

import argparse
import glob
import json
import os
import sys
import warnings
from functools import partial

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import numpy as np
import pandas as pd
import xarray as xr
import dask
from matplotlib import pyplot as plt

from diagnostics.diagnostic_helpers import (
    ANALYSIS_END, ANALYSIS_START, FigureSaver, LATLON_PATTERN,
    ZONAL_MEAN_PATTERN, add_symmetric_colorbar,
    configured_label, diagnostic_cache_dir, diagnostic_output_dir,
    difference_cmap, env_flag,
    figure_should_run, finite_or_nan, hemispheric_fold, jsonable_value,
    latitude_weights, load_latlon_dataset, precip_rate_from_accum,
    native_midlevel_height, pressure_layer_weights, pressure_to_pa,
    read_json_file, safe_name,
    simulation_colour, standardize_latlon_dataset, subset_between,
    source_file_signature,
    subset_latitude, symmetric_levels, time_block_slices, time_step_days,
    total_precip_accum, vertical_dim, vertical_velocity_from_zm,
    write_json_file,
)
from diagnostics.ttl import TTL_HEIGHT_BOUNDS_M
from diagnostics.convection.high_cloud_classification import (
    CLASSIFICATION_CACHE_VERSION,
    HighCloudThresholds,
    cached_classification_is_current as shared_cached_classification_is_current,
)
from diagnostics.waves.equatorial_wave_config import (
    SHORT_LAG_FILTERS,
    SHORT_PERIOD_CRITICAL_LEVEL_FILTERS,
    VERTICAL_STRUCTURE_FILTERS,
    WAVE_BOXES as SHARED_WAVE_BOXES,
    WAVE_CONFIG_VERSION,
    WAVE_LATITUDE_PROJECTION as SHARED_WAVE_LATITUDE_PROJECTION,
)
from diagnostics.waves.cached_wave_products import (
    wave_response_metrics_product_path,
)
from diagnostics.convection.lpo_tracking import (
    PRODUCT_VERSION as LPO_TRACKING_PRODUCT_VERSION,
    publication_output_directory,
    read_standard_event_tables,
    require_publication_products,
)
from diagnostics.waves.wave_metrics import (
    METRICS_VERSION,
    project_latitude_for_wave_filter,
    upward_pressure_velocity,
    wave_filter_field_units,
)
from waveFilter import kf_filter


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def parse_command_line(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=range(1, 10),
        default=None,
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--figures-only",
        action="store_true",
        help="Build selected figures only from current bootstrap caches.",
    )
    cache_mode.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild selected bootstrap caches before plotting.",
    )
    args = parser.parse_args(argv)
    if args.figures:
        args.figures = list(dict.fromkeys(args.figures))
    return args


runtime_args = parse_command_line()
save = True
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/significance_bootstrap/")

# Figure numbers:
#   1: mean-state lat-height block-resampled response maps
#   2: precipitation scalar/PDF block-consistency diagnostics
#   3: cloud profile/cloud-type block-consistency diagnostics
#   4: CCEW wave-box variance block-consistency diagnostics
#   5: LPO precipitation-envelope event-bootstrap diagnostics
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
#     temperature-driven pathway, while U-only wind/wave differences favour a
#     wind-driven pathway.
figures_to_run = runtime_args.figures or [2, 3, 4, 5]
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)
FIGURES_ONLY = bool(runtime_args.figures_only)

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control_new", "u_nudge", "T_nudge"]
sim_names = ["control", "nudged u", "nudged T"]
cols = [simulation_colour(name) for name in sim_names]

dt = 6
obs_per_day = int(round(24 / dt))

USE_BOOTSTRAP_CACHE = True
FORCE_RECOMPUTE_BOOTSTRAP_CACHE = False
BOOTSTRAP_CACHE_VERSION = (
    "significance-bootstrap-v4-corrected-t-nudge-omega"
)
BOOTSTRAP_WAVE_CACHE_VERSION = (
    "significance-bootstrap-wave-v3-native-omega-units"
)
WAVE_SOURCE_CONTRACT_VERSION = "significance-wave-source-v2-native-omega-units"
_rebuilt_wave_cache_paths = set()

USE_BOOTSTRAP_CACHE = USE_BOOTSTRAP_CACHE and not env_flag("DISABLE_BOOTSTRAP_CACHE")
FORCE_RECOMPUTE_BOOTSTRAP_CACHE = (
    FORCE_RECOMPUTE_BOOTSTRAP_CACHE
    or runtime_args.rebuild_cache
    or env_flag("FORCE_RECOMPUTE_BOOTSTRAP_CACHE")
)
if FIGURES_ONLY and FORCE_RECOMPUTE_BOOTSTRAP_CACHE:
    raise SystemExit(
        "Figure-only mode cannot run with forced bootstrap cache rebuilding"
    )
BOOTSTRAP_CACHE_DIR = diagnostic_cache_dir("significance_bootstrap_cache")

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

TTL_Z_BOUNDS_M = TTL_HEIGHT_BOUNDS_M
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

# The main wave analysis includes the slow eastward-intraseasonal band in its
# vertical profiles. This script deliberately limits ordinary short-block
# resampling to filters for which a 10-day block is meaningful.
WAVE_PROFILE_WAVES = tuple(
    name
    for name in VERTICAL_STRUCTURE_FILTERS
    if name in SHORT_PERIOD_CRITICAL_LEVEL_FILTERS
)
WAVE_PROFILE_FIELDS = [
    dict(name="u", prefix="u", units="m s-1"),
    dict(name="T", prefix="T", units="K"),
    dict(name="w", prefix="w", units="Pa s-1"),
]
WAVE_PROFILE_LEVELS_HPA = [100, 150, 200, 250, 300, 500, 700, 850]

WAVE_LAG_WAVES = tuple(SHORT_LAG_FILTERS)
WAVE_LAG_TARGET_FIELDS = ["Precip", "OLR", "PW"]
WAVE_LAG_MAX_DAYS = 8.0
WAVE_LAG_BLOCK_DAYS = 30.0

ORGANISATION_OUTPUT_DIR = publication_output_directory()
ORGANISATION_TRACKS_FILE = str(ORGANISATION_OUTPUT_DIR / "primary")
ORGANISATION_TRACK_METRIC_SPECS = [
    ("duration_hours", "h"),
    ("maximum_area_km2", "km2"),
    ("event_integrated_precipitation", "km2 mm"),
    ("zonal_speed_m_s", "m s-1"),
    ("total_zonal_displacement_km", "km"),
]
# Rows come from the definitive 12-mm/day tropical LPO population. Complete
# events are resampled as indivisible units; track steps are never samples.
LPO_TRACK_SOURCE = "lpo_precipitation_envelope"
LPO_TRACK_POPULATION = "primary_tropical_lpo"
ANALYSIS_TRACK_POPULATIONS = (LPO_TRACK_POPULATION,)
TRACK_POPULATION_LABELS = {
    LPO_TRACK_POPULATION: "Large-scale tropical precipitation envelopes",
}

# Shared coarse-grid precipitation-regime cloud thresholds.
CF_THRESH = 0.05
P_HIGH_HPA = 440.0
P_PRECIP_MIN = 12.0  # 0.5 mm h-1 PF threshold (Feng et al. 2021)
CLOUD_CLASSIFICATION_CACHE_DIR = diagnostic_cache_dir(
    "classification_cache"
)

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
    "DeepTropics_TTL": dict(lat_bounds=DEEP_TROPICS_LAT_BOUNDS, z_bounds=TTL_Z_BOUNDS_M),
    "Tropics_TTL": dict(lat_bounds=TROPICAL_LAT_BOUNDS, z_bounds=TTL_Z_BOUNDS_M),
    "Tropics_upper_troposphere_12_18km": dict(lat_bounds=TROPICAL_LAT_BOUNDS, z_bounds=UPPER_TROPOSPHERE_Z_BOUNDS_M),
}

WAVE_BOXES = {
    name: SHARED_WAVE_BOXES[name]
    for name in SHORT_PERIOD_CRITICAL_LEVEL_FILTERS
}
WAVE_LATITUDE_PROJECTION = {
    name: SHARED_WAVE_LATITUDE_PROJECTION[name]
    for name in WAVE_BOXES
}
WAVE_FIELDS = ["Precip", "OLR", "PW"]
WAVE_RESPONSE_METRIC_OUTPUT = wave_response_metrics_product_path()

latlon_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw", "olrtoa", "cape",
    "qv", "relhum", "w",
    "theta", "pressure", "zgrid", "cldfrac",
    "uReconstructZonal", "uReconstructMeridional",
    "t_isobaric", "uzonal_isobaric", "umeridional_isobaric", "rh_isobaric",
    "qv_isobaric", "omg_isobaric",
]
zonal_mean_vars = [
    "relhum", "w", "theta", "pressure", "cldfrac", "uReconstructZonal",
    "uReconstructMeridional", "t_isobaric", "uzonal_isobaric",
    "umeridional_isobaric", "rh_isobaric", "qv_isobaric",
]

latlon_chunks = {"Time": 8, "latitude": 90, "longitude": 180}
zm_chunks = {"Time": 8}

dask.config.set({"array.slicing.split_large_chunks": True})


# -----------------------------------------------------------------------------
# Bookkeeping and generic bootstrap helpers
# -----------------------------------------------------------------------------

implemented: list[str] = []
failed: list[str] = []
todo_diagnostics: list[str] = []
scalar_metric_rows: list[dict] = []


should_run_figure = partial(figure_should_run, figures_to_run)


def record_implemented(message):
    print(f"Implemented: {message}")
    implemented.append(message)


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


_bootstrap_source_signatures = {}


def bootstrap_source_signature(family):
    """Return the source signature required by one bootstrap family."""

    if family in _bootstrap_source_signatures:
        return _bootstrap_source_signatures[family]
    if family == "lpo_envelope_event_properties":
        paths = sorted(
            glob.glob(
                os.path.join(ORGANISATION_TRACKS_FILE, "**", "*.csv"),
                recursive=True,
            )
        )
    else:
        paths = sorted(
            path
            for directory in data_locs
            for pattern in (LATLON_PATTERN, ZONAL_MEAN_PATTERN)
            for path in glob.glob(os.path.join(root_dir, directory, pattern))
        )
        if family == "cloud_response":
            paths.extend(
                sorted(
                    glob.glob(
                        os.path.join(
                            CLOUD_CLASSIFICATION_CACHE_DIR,
                            "*_classified_high_cloud.nc",
                        )
                    )
                )
            )
    signature = source_file_signature(paths)
    _bootstrap_source_signatures[family] = signature
    return signature


def base_bootstrap_cache_signature(family, extra=None):
    signature = dict(
        cache_version=BOOTSTRAP_CACHE_VERSION,
        family=family,
        root_dir=root_dir,
        data_locs=list(data_locs),
        latlon_pattern=LATLON_PATTERN,
        zonal_mean_pattern=ZONAL_MEAN_PATTERN,
        sim_names=list(sim_names),
        analysis_start_date=ANALYSIS_START,
        dt=float(dt),
        analysis_end_date=ANALYSIS_END,
        bootstrap_block_days=float(BOOTSTRAP_BLOCK_DAYS),
        cloud_type_block_days=float(CLOUD_TYPE_BLOCK_DAYS),
        n_bootstrap=int(N_BOOTSTRAP),
        field_n_bootstrap=int(FIELD_N_BOOTSTRAP),
        bootstrap_seed=int(BOOTSTRAP_SEED),
        min_blocks=int(MIN_BLOCKS),
        ci_percentiles=jsonable_value(CI_PERCENTILES),
        robust_sign_threshold=float(ROBUST_SIGN_THRESHOLD),
        comparisons=jsonable_value(COMPARISONS),
        wave_config_version=WAVE_CONFIG_VERSION,
        source_files=bootstrap_source_signature(family),
    )
    if extra:
        signature.update(jsonable_value(extra))
    return signature


read_cache_metadata = read_json_file
write_cache_metadata = write_json_file


def bootstrap_cache_matches(cache_name, signature):
    if not USE_BOOTSTRAP_CACHE:
        return False
    if FORCE_RECOMPUTE_BOOTSTRAP_CACHE:
        print(f"Ignoring cached {cache_name}: force recompute is enabled.")
        return False
    metadata_path = bootstrap_cache_path(f"{cache_name}.metadata.json")
    cached = read_cache_metadata(metadata_path)
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
        if FIGURES_ONLY:
            raise FileNotFoundError(
                f"Figure-only mode requires current cache {cache_name}"
            )
        return None
    path = bootstrap_cache_path(f"{cache_name}.csv")
    if not os.path.exists(path):
        if FIGURES_ONLY:
            raise FileNotFoundError(
                f"Figure-only mode requires cache file {path}"
            )
        return None
    df = pd.read_csv(path)
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
        if FIGURES_ONLY:
            raise FileNotFoundError(
                f"Figure-only mode requires current cache {cache_name}"
            )
        return None
    metadata = read_cache_metadata(
        bootstrap_cache_path(f"{cache_name}.metadata.json")
    )
    if metadata is None or "comparison_names" not in metadata:
        if FIGURES_ONLY:
            raise FileNotFoundError(
                f"Figure-only mode requires comparison metadata {cache_name}"
            )
        return None
    comp_names = metadata["comparison_names"]
    expected_names = [name for name, *_ in COMPARISONS]
    if comp_names != expected_names:
        if FIGURES_ONLY:
            raise RuntimeError(
                f"Figure-only cache has wrong comparisons: {cache_name}"
            )
        return None
    out = {}
    for comp_name in comp_names:
        path = bootstrap_cache_path(cached_diff_dataset_filename(cache_name, comp_name))
        if not os.path.exists(path):
            if FIGURES_ONLY:
                raise FileNotFoundError(
                    f"Figure-only mode requires cache file {path}"
                )
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
            data_vars[col] = (dim_name, pd.to_numeric(sub[col], errors="raise").to_numpy(dtype=float))
    if "robust" in sub.columns:
        robust = sub["robust"]
        if robust.dtype != bool:
            raise TypeError("robust cache column must be boolean")
        data_vars["robust"] = (dim_name, robust.to_numpy())
    if "ci_excludes_zero" in sub.columns:
        excludes = sub["ci_excludes_zero"]
        if excludes.dtype != bool:
            raise TypeError("ci_excludes_zero cache column must be boolean")
        data_vars["ci_excludes_zero"] = (dim_name, excludes.to_numpy())
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


def z_values_m(da):
    z = np.asarray(da.z.values, dtype=float)
    return z if np.nanmax(np.abs(z)) > 100.0 else z * 1000.0


def subset_z_m(da, z_bounds_m):
    if z_bounds_m is None or "z" not in da.dims:
        return da
    z_m = xr.DataArray(z_values_m(da), dims=("z",), coords={"z": da.z})
    lo, hi = sorted(z_bounds_m)
    return da.where((z_m >= lo) & (z_m <= hi), drop=True)


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


block_slices = partial(
    time_block_slices,
    block_days=BOOTSTRAP_BLOCK_DAYS,
    complete=True,
    min_blocks=MIN_BLOCKS,
    require_time=True,
)


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


def bootstrap_field_difference(a_blocks, b_blocks, rng, n_boot=FIELD_N_BOOTSTRAP):
    b_blocks, _ = xr.align(b_blocks, a_blocks, join="exact", exclude={"block"})
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


dss = []
dss_zm = []


def ensure_datasets_loaded():
    """Open raw histories only after a selected cache misses validation."""

    if dss:
        return
    print("Loading lat-lon and zonal-mean datasets")
    for i, directory in enumerate(data_locs):
        ds = load_latlon_dataset(
            os.path.join(root_dir, directory),
            latlon_pattern=LATLON_PATTERN,
            chunks=latlon_chunks,
        )
        ds = standardize_latlon_dataset(ds, latlon_vars)

        zds = load_latlon_dataset(
            os.path.join(root_dir, directory),
            latlon_pattern=ZONAL_MEAN_PATTERN,
            chunks=zm_chunks,
        )
        zds = standardize_latlon_dataset(zds, zonal_mean_vars)

        print(
            f"{sim_names[i]}: latlon vars={len(ds.data_vars)}, "
            f"zm vars={len(zds.data_vars)}"
        )
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
    zdim = vertical_dim(da)
    height = native_midlevel_height(source_ds, da)
    reduce_dims = [dim for dim in height.dims if dim != zdim]
    height = height.mean(reduce_dims, skipna=True) if reduce_dims else height
    if height.sizes[zdim] != da.sizes[zdim]:
        raise RuntimeError(
            "Native midpoint-height coordinate does not match the field "
            f"levels: {height.sizes[zdim]} versus {da.sizes[zdim]}"
        )
    out = da.assign_coords({zdim: np.asarray(height.values, dtype=float)})
    out = out.rename({zdim: "z"})
    out.z.attrs.update(units="m", long_name="native MPAS mid-level height")
    return out


def require_zonal_mean_height(da):
    """Require the fixed physical-height coordinate in zonal-mean products."""

    out = rename_vertical_to_z(da)
    if "z" not in out.coords:
        raise RuntimeError(
            "Zonal-mean field lacks the required physical 'z' coordinate"
        )
    height = np.asarray(out.z.values, dtype=float)
    if (
        height.ndim != 1
        or height.size != out.sizes["z"]
        or not np.all(np.isfinite(height))
        or not np.all(np.diff(height) > 0)
    ):
        raise RuntimeError(
            "Zonal-mean 'z' coordinate must be finite, one-dimensional, "
            "and strictly increasing"
        )
    out.z.attrs["units"] = "m"
    return out


def zm_temperature(zds):
    missing = [name for name in ("theta", "pressure") if name not in zds.data_vars]
    if missing:
        raise ValueError(f"Missing variables for temperature: {missing}")
    theta = require_zonal_mean_height(finite_or_nan(zds.theta))
    pressure = require_zonal_mean_height(
        pressure_to_pa(finite_or_nan(zds.pressure))
    )
    out = theta * (pressure / P0_PA) ** (RD / CP)
    out.name = "temperature"
    out.attrs["units"] = "K"
    return out


def zm_static_stability(zds):
    if "theta" not in zds.data_vars:
        raise ValueError("Missing theta for static stability")
    theta = require_zonal_mean_height(finite_or_nan(zds.theta))
    dtheta_dz_m = theta.differentiate("z")
    out = G / theta * dtheta_dz_m
    out.name = "N2"
    out.attrs["units"] = "s-2"
    return out


def zm_lapse_rate(zds):
    temp = zm_temperature(zds)
    out = -temp.differentiate("z") * 1000.0
    out.name = "lapse_rate"
    out.attrs["units"] = "K km-1"
    return out


def zm_vertical_velocity(zds):
    out = vertical_velocity_from_zm(zds)
    out.name = "vertical_velocity"
    return out


def water_vapour_field(ds, zds):
    return (assign_midlevel_height_from_zgrid(ds.qv, ds) * 1000.0).rename("water_vapour")


def cloud_fraction_time_field(ds, zds):
    return require_zonal_mean_height(zds.cldfrac).rename("cloud_fraction")


LATHEIGHT_FIELD_SPECS = [
    dict(name="temperature", label="Temperature", units="K", zero_difference_scale=2.0,
         getter=lambda ds, zds: zm_temperature(zds), parity="even"),
    dict(name="zonal_wind", label="Zonal wind", units="m s-1", zero_difference_scale=5.0,
         getter=lambda ds, zds: zds.uReconstructZonal.rename("zonal_wind"), parity="even"),
    dict(name="static_stability_N2", label="Static stability N2", units="1e-4 s-2", zero_difference_scale=1.0,
         getter=lambda ds, zds: zm_static_stability(zds) * 1.0e4, parity="even"),
    dict(name="lapse_rate", label="Lapse rate", units="K km-1", zero_difference_scale=0.5,
         getter=lambda ds, zds: zm_lapse_rate(zds), parity="even"),
    dict(name="relative_humidity", label="Relative humidity", units="%", zero_difference_scale=10.0,
         getter=lambda ds, zds: zds.relhum.rename("relative_humidity"), parity="even"),
    dict(name="water_vapour", label="Water vapour", units="g kg-1", zero_difference_scale=1.0,
         getter=water_vapour_field, parity="even"),
    dict(name="vertical_velocity", label="Vertical velocity", units="m s-1", zero_difference_scale=0.01,
         getter=lambda ds, zds: zm_vertical_velocity(zds), parity="even"),
    dict(name="cloud_fraction", label="Cloud fraction", units="fraction", zero_difference_scale=0.05,
         getter=cloud_fraction_time_field, parity="even"),
]


def block_latheight_field(da, parity="even", block_days=BOOTSTRAP_BLOCK_DAYS):
    x = require_zonal_mean_height(finite_or_nan(da))
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


def symmetric_field_levels(
    fields, zero_difference_scale=1.0, percentile=99.0, ndiv=16
):
    vals = []
    for field in fields:
        arr = np.asarray(field, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(np.nanpercentile(np.abs(arr), percentile))
    if not vals:
        raise RuntimeError("Cannot determine plot levels from non-finite fields")
    vmax = max(vals)
    if not np.isfinite(vmax):
        raise RuntimeError("Cannot determine plot levels from invalid fields")
    if vmax == 0.0:
        vmax = zero_difference_scale
    return symmetric_levels(vmax, ndiv=ndiv)


def plot_latheight_bootstrap(spec, diff_by_comparison):
    if not should_run_figure(1):
        return
    missing = [
        name for name, *_ in COMPARISONS if name not in diff_by_comparison
    ]
    if missing:
        raise RuntimeError(
            f"Lat-height bootstrap plot lacks comparisons {missing}"
        )
    comp_items = [
        (name, label, diff_by_comparison[name])
        for name, _, _, label in COMPARISONS
    ]

    fields = [item[2].mean_response.values for item in comp_items]
    levels = symmetric_field_levels(
        fields,
        zero_difference_scale=spec["zero_difference_scale"],
    )
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

            ensure_datasets_loaded()
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
            record_failure(f"mean-state {spec['name']}", exc)


# -----------------------------------------------------------------------------
# Precipitation response: mean, variance, tails, and PDFs
# -----------------------------------------------------------------------------

def get_precip_rate(ds):
    pr = precip_rate_from_accum(total_precip_accum(ds))
    pr.name = "precip"
    pr.attrs["units"] = "mm day-1"
    return pr


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
            color=colors_by_comp[comp_name],
            lw=1.4,
            label=label,
        )
        ax.fill_between(
            x[plot_mask],
            ds.ci_low.values[plot_mask],
            ds.ci_high.values[plot_mask],
            color=colors_by_comp[comp_name],
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
            return {}

        ensure_datasets_loaded()
        precip_by_run = {
            i: get_precip_rate(ds) for i, ds in enumerate(dss)
        }
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
            color = colors_by_comp[comp_name]
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

        ensure_datasets_loaded()
        rows = []
        pdf_diffs_by_field = {}
        rng = np.random.default_rng(BOOTSTRAP_SEED + 222)
        for spec in BASIC_CONVECTION_PDF_SPECS:
            blocks_by_run = {
                i: field_pdf_block_values(
                    get_surface_convection_field(dss[i], spec["name"]),
                    spec["bins"],
                )
                for i in range(len(dss))
            }

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
            key = (direction, field)
            if key not in diff_by_direction_field:
                raise RuntimeError(
                    f"Missing wavenumber bootstrap response for {direction} "
                    f"{field}"
                )
            diffs = diff_by_direction_field[key]
            for comp_name, ds in diffs.items():
                label = next(label for name, _, _, label in COMPARISONS if name == comp_name)
                k = ds.wavenumber.values
                color = colors_by_comp[comp_name]
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
    fig.suptitle(
        "Deep-tropical wavenumber variance-fraction "
        "block-consistency responses"
    )
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

        ensure_datasets_loaded()
        rows = []
        diffs_by_direction_field = {}
        rng = np.random.default_rng(BOOTSTRAP_SEED + 333)
        for field_name in WAVENUMBER_FIELDS:
            field_by_run = {
                i: get_surface_convection_field(dss[i], field_name)
                for i in range(len(dss))
            }
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
        raise RuntimeError("cldfrac is required for cloud-profile diagnostics")
    return require_zonal_mean_height(finite_or_nan(ds.cldfrac))


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
    if not should_run_figure(3):
        return
    missing = [name for name, *_ in COMPARISONS if name not in profile_diffs]
    if missing:
        raise RuntimeError(
            f"Cloud-profile bootstrap plot lacks comparisons {missing}"
        )
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.4), constrained_layout=True)
    colors_by_comp = {"u_minus_control": "b", "T_minus_control": "r", "u_minus_T": "0.2"}
    for comp_name, ds in profile_diffs.items():
        label = next(label for name, _, _, label in COMPARISONS if name == comp_name)
        z_km = z_values_m(ds.mean_response) / 1000.0
        ax.plot(ds.mean_response, z_km, color=colors_by_comp[comp_name], lw=1.5, label=label)
        ax.fill_betweenx(z_km, ds.ci_low, ds.ci_high, color=colors_by_comp[comp_name], alpha=0.13)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_ylim(0.0, 25.0)
    ax.set_xlabel("Cloud-fraction response")
    ax.set_ylabel("Height (km)")
    ax.set_title("Tropical vertical cloud-fraction profile")
    ax.legend(frameon=False, fontsize=8)
    figsave.save(fig, "cloud_fraction_profile_bootstrap_response", number=3)
    plt.close(fig)


classification_thresholds = partial(
    HighCloudThresholds,
    cf_thresh=CF_THRESH,
    high_cloud_top_pressure_hpa=P_HIGH_HPA,
    surface_precip_min_mm_day=P_PRECIP_MIN,
)
cached_classification_is_current = partial(
    shared_cached_classification_is_current,
    thresholds=classification_thresholds(),
)


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
    path = os.path.join(CLOUD_CLASSIFICATION_CACHE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{sim_name}: required shared high-cloud classification cache "
            f"is missing: {path}. Run "
            "refined_high_cloud_classification_diagnostics.py first."
        )
    cached = xr.open_dataset(path, chunks={})
    if not cached_classification_is_current(cached):
        cached.close()
        raise RuntimeError(
            f"{sim_name}: stale high-cloud classification cache: {path}"
        )
    missing = [
        key for key in REQUIRED_HIGH_CLOUD_CLASSIFICATION_VARS
        if key not in cached.data_vars
    ]
    if missing:
        cached.close()
        raise RuntimeError(
            f"{sim_name}: high-cloud classification cache lacks required "
            f"shared variables {missing}: {path}"
        )
    out = {}
    expected_time = precip_ref.Time.to_index()
    for key in REQUIRED_HIGH_CLOUD_CLASSIFICATION_VARS:
        da = cached[key]
        if "Time" in da.dims:
            actual_time = da.Time.to_index()
            missing_time = expected_time.difference(actual_time)
            if len(missing_time):
                cached.close()
                raise RuntimeError(
                    f"{sim_name}: classification lacks {len(missing_time)} "
                    f"required precipitation times for {key}: {path}"
                )
            da = da.sel(Time=expected_time)
        if key.endswith("_mask"):
            da = da.fillna(False).astype(bool)
        out[key] = da
    print(f"Using cached high-cloud classification for {sim_name}: {path}")
    return out


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
                classification="shared coarse-grid precipitation-regime high-cloud classification",
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

    ensure_datasets_loaded()
    if not precip_by_run:
        precip_by_run = {
            i: get_precip_rate(ds) for i, ds in enumerate(dss)
        }
    row_start = len(scalar_metric_rows)
    profile_rows = pd.DataFrame()
    try:
        cf_by_run = {}
        for i in range(len(dss)):
            cf_by_run[i] = get_cloud_fraction(dss_zm[i])

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
        raise RuntimeError(
            "Cloud-type bootstrap requires precipitation for every simulation"
        )

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

def aligned_field(ds, key, ref):
    variable = {"OLR": "olrtoa", "PW": "precipw"}[key]
    return ds[variable].sel(Time=ref.Time).rename(key)


def conditional_area_mean_blocks(target, condition):
    target, condition = xr.align(target, condition, join="inner")
    x = subset_latitude(target, TROPICAL_LAT_BOUNDS)
    c = subset_latitude(condition, TROPICAL_LAT_BOUNDS)
    dims = [dim for dim in ("Time", "latitude", "longitude") if dim in x.dims]
    return block_scalar_values(x, lambda xb: weighted_mean_dims(xb.where(c.sel(Time=xb.Time)), dims))


def run_conditional_precipitation_states(precip_by_run):
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

        ensure_datasets_loaded()
        if not precip_by_run:
            precip_by_run = {
                i: get_precip_rate(ds) for i, ds in enumerate(dss)
            }
        row_start = len(scalar_metric_rows)
        targets = {
            "precip": {i: precip_by_run[i] for i in range(len(dss))},
            "OLR": {i: aligned_field(dss[i], "OLR", precip_by_run[i]) for i in range(len(dss))},
            "PW": {i: aligned_field(dss[i], "PW", precip_by_run[i]) for i in range(len(dss))},
        }
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
# LPO precipitation-envelope event properties
# -----------------------------------------------------------------------------

ORGANISATION_SOURCE_ORDER = [
    LPO_TRACK_SOURCE,
]
ORGANISATION_PRIMARY_TRACK_SOURCE = LPO_TRACK_SOURCE
ORGANISATION_TRACK_CACHE_VERSION = LPO_TRACKING_PRODUCT_VERSION


def load_lpo_envelope_events():
    publication = require_publication_products()
    df, _lifecycle = read_standard_event_tables(publication / "primary")
    required = {
        "simulation", "event_id", "primary_tropical_sample",
        "reference_in_record_subset",
        *[metric for metric, _ in ORGANISATION_TRACK_METRIC_SPECS],
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"LPO event table is missing columns: {missing}")

    df = df[
        df.primary_tropical_sample.astype(bool)
        & df.reference_in_record_subset.astype(bool)
    ].copy()
    df["simulation"] = df.simulation.map(
        {
            "control": "control",
            "nudged-u": "nudged u",
            "nudged-T": "nudged T",
        }
    )
    if df.simulation.isna().any():
        raise ValueError("LPO event table contains an unknown simulation")
    df["source"] = LPO_TRACK_SOURCE
    df["source_label"] = TRACK_POPULATION_LABELS[LPO_TRACK_POPULATION]
    df[LPO_TRACK_POPULATION] = True
    for col in [metric for metric, _ in ORGANISATION_TRACK_METRIC_SPECS]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def lpo_source_label(df, source):
    sub = df[df["source"] == source]
    if len(sub) == 0 or "source_label" not in sub.columns:
        return source
    vals = sub["source_label"].dropna().unique()
    return str(vals[0]) if len(vals) else source


def ordered_lpo_sources(df):
    present = set(df["source"].dropna().unique())
    unexpected = present - {LPO_TRACK_SOURCE}
    if unexpected:
        raise RuntimeError(
            f"LPO event tables contain unexpected sources: {sorted(unexpected)}"
        )
    ordered = [source for source in ORGANISATION_SOURCE_ORDER if source in present]
    return ordered


def lpo_source_role(source):
    if source == LPO_TRACK_SOURCE:
        return "large_scale_precipitation_envelopes"
    raise RuntimeError(
        f"LPO event tables contain an unexpected source: {source!r}"
    )


def plot_lpo_envelope_bootstrap_summary(org_df, org_source_labels):
    if not should_run_figure(5) or len(org_df) == 0:
        return

    display_metrics = [
        ("duration_hours", "Envelope duration (h)"),
        ("maximum_area_km2", "Maximum envelope area (km2)"),
        (
            "event_integrated_precipitation",
            "Event-integrated precipitation (km2 mm)",
        ),
        ("zonal_speed_m_s", "Mean zonal propagation speed (m s-1)"),
        ("total_zonal_displacement_km", "Net zonal displacement (km)"),
    ]
    plot_df = org_df[
        (org_df["category"] == "lpo_envelope_event_properties")
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
                raise RuntimeError(
                    f"Missing LPO bootstrap comparison {comp_name} for "
                    f"metric {metric}"
                )
            yvals = []
            xvals = []
            xerr_low = []
            xerr_high = []
            robust = []
            for si, source in enumerate(sources):
                row = sub[sub["diagnostic"] == source]
                if len(row) == 0:
                    raise RuntimeError(
                        f"Missing LPO bootstrap row for {source} {metric} "
                        f"{comp_name}"
                    )
                row = row.iloc[0]
                x = float(row["mean_response"])
                yvals.append(ybase[si] + offsets[comp_name])
                xvals.append(x)
                xerr_low.append(max(0.0, x - float(row["ci_low"])))
                xerr_high.append(max(0.0, float(row["ci_high"]) - x))
                robust.append(bool(row["robust"]))
            color = colors_by_comp[comp_name]
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
        ax.set_yticklabels([org_source_labels[source] for source in sources])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlabel(f"{label} response")
    axes.ravel()[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "LPO precipitation-envelope responses with each event weighted equally; "
        "filled points exclude zero"
    )
    figsave.save(fig, "lpo_envelope_bootstrap_summary", number=5)
    plt.close(fig)


def run_lpo_envelope_bootstrap():
    try:
        cache_signature = base_bootstrap_cache_signature(
            "lpo_envelope_event_properties",
            extra=dict(
                organisation_tracks_file=ORGANISATION_TRACKS_FILE,
                track_metric_specs=jsonable_value(ORGANISATION_TRACK_METRIC_SPECS),
                primary_track_source=ORGANISATION_PRIMARY_TRACK_SOURCE,
                organisation_track_cache_version=ORGANISATION_TRACK_CACHE_VERSION,
                cloud_classification_cache_version=CLASSIFICATION_CACHE_VERSION,
                track_populations=list(ANALYSIS_TRACK_POPULATIONS),
                weighting="tracks are resampled equally within each population",
            ),
        )
        cached_rows = load_cached_table(
            "lpo_envelope_bootstrap_metrics", cache_signature
        )
        if cached_rows is not None:
            append_cached_scalar_rows(cached_rows)
            save_table(cached_rows, "lpo_envelope_bootstrap_metrics.csv")
            source_labels = (
                cached_rows.dropna(subset=["diagnostic", "source_label"])
                .drop_duplicates("diagnostic")
                .set_index("diagnostic")["source_label"]
                .to_dict()
            )
            plot_lpo_envelope_bootstrap_summary(cached_rows, source_labels)
            record_implemented("cached LPO-envelope event bootstrap")
            return

        tracks = load_lpo_envelope_events()
        source_labels = {}
        row_start = len(scalar_metric_rows)
        for source in ordered_lpo_sources(tracks):
            for population in ANALYSIS_TRACK_POPULATIONS:
                diagnostic = f"{source}:{population}"
                population_label = TRACK_POPULATION_LABELS[population]
                source_labels[diagnostic] = population_label
                source_tracks = tracks[
                    (tracks.source == source) & tracks[population]
                ]
                for metric, units in ORGANISATION_TRACK_METRIC_SPECS:
                    values_by_run = {}
                    for simulation_index, simulation in enumerate(sim_names):
                        values = source_tracks.loc[
                            source_tracks.simulation == simulation, metric
                        ].to_numpy(dtype=float)
                        values = values[np.isfinite(values)]
                        if values.size < MIN_BLOCKS:
                            raise RuntimeError(
                                f"{source} {population} {metric}: "
                                f"{simulation} has {values.size} finite events; "
                                f"at least {MIN_BLOCKS} are required"
                            )
                        values_by_run[simulation_index] = values
                    append_bootstrap_metric_rows(
                        scalar_metric_rows,
                        values_by_run,
                        "lpo_envelope_event_properties",
                        diagnostic,
                        metric,
                        units=units,
                        extra={
                            "source_label": population_label,
                            "track_population": population,
                            "organisation_role": lpo_source_role(source),
                            "weighting": "tracks resampled equally within population",
                        },
                    )

        result = pd.DataFrame(scalar_metric_rows[row_start:])
        if len(result) == 0:
            raise RuntimeError(
                "LPO-envelope event bootstrap produced no configured metrics"
            )
        save_table(result, "lpo_envelope_bootstrap_metrics.csv")
        plot_lpo_envelope_bootstrap_summary(result, source_labels)
        write_cached_table("lpo_envelope_bootstrap_metrics", result, cache_signature)
        record_implemented(
            "bootstrap of tropical LPO-envelope properties with equal event "
            "weights"
        )
    except Exception as exc:
        record_failure("LPO-envelope event bootstrap", exc)
        raise


# -----------------------------------------------------------------------------
# Wave-mediated pathway: integrated CCEW wave-box variance
# -----------------------------------------------------------------------------

wave_label = partial(configured_label, WAVE_BOXES)


def wave_cache_dir():
    path = diagnostic_cache_dir("significance_bootstrap_wave_cache")
    os.makedirs(path, exist_ok=True)
    return path


def get_surface_convection_field(ds, field_name, ref=None):
    pr = get_precip_rate(ds)
    if field_name == "Precip":
        out = pr
    else:
        variable = {
            "OLR": "olrtoa",
            "PW": "precipw",
            "CAPE": "cape",
        }[field_name]
        time = ref.Time if ref is not None else pr.Time
        out = ds[variable].sel(Time=time).rename(field_name)
    out.attrs = dict(out.attrs)
    out.attrs["units"] = wave_filter_field_units(field_name)
    return out


def get_wave_filter_field(ds, field_name):
    pr = get_precip_rate(ds)
    if field_name in ("Precip", "OLR", "PW", "CAPE"):
        return get_surface_convection_field(ds, field_name, ref=pr)

    pressure_fields = [
        ("w", None),
        ("u", "uzonal_isobaric"),
        ("T", "t_isobaric"),
    ]
    for prefix, var in pressure_fields:
        if not field_name.startswith(prefix):
            continue
        level_text = field_name[len(prefix):]
        if not level_text:
            continue
        plev = float(level_text)
        if var is None:
            out = upward_pressure_velocity(ds, plev, times=pr.Time)
        else:
            out = ds[var].sel(pres=plev).sel(Time=pr.Time)
        out.name = field_name
        out.attrs = dict(out.attrs)
        out.attrs["units"] = wave_filter_field_units(field_name)
        out.attrs["selected_pressure_hpa"] = plev
        return out

    raise ValueError(f"Unsupported wave field {field_name}")


def project_wave_latitude(da, wave_name):
    return project_latitude_for_wave_filter(
        da,
        WAVE_LATITUDE_PROJECTION[wave_name],
        WAVE_FILTER_LAT_BOUNDS,
    )


def filter_projected_wave(projected, wave_name, sim_index, field_name):
    cache_path = os.path.join(
        wave_cache_dir(),
        f"{safe_name(sim_names[sim_index])}_{safe_name(field_name)}_{safe_name(wave_name)}_bootstrap.nc",
    )
    force_this_path = (
        FORCE_RECOMPUTE_BOOTSTRAP_CACHE
        and cache_path not in _rebuilt_wave_cache_paths
    )
    if os.path.exists(cache_path) and not force_this_path:
        try:
            cached = xr.open_dataset(cache_path)
            if (
                cached.attrs.get("cache_version")
                == BOOTSTRAP_WAVE_CACHE_VERSION
                and cached.attrs.get("field") == field_name
                and cached.attrs.get("wave") == wave_name
                and cached.attrs.get("source_simulation") == sim_names[sim_index]
            ):
                return cached["wave"]
            cached.close()
        except Exception:
            pass

    x = projected.load()
    arr = np.asarray(x.values, dtype=np.float32)
    arr = arr - np.nanmean(arr, axis=0, keepdims=True)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(
            f"{sim_names[sim_index]} {field_name} {wave_name}: projected "
            "filter input contains non-finite values"
        )
    if "units" not in x.attrs or not str(x.attrs["units"]).strip():
        raise RuntimeError(
            f"{sim_names[sim_index]} {field_name}: source field lacks units"
        )

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
            "cache_version": BOOTSTRAP_WAVE_CACHE_VERSION,
            "units": x.attrs["units"],
        },
    )
    out_ds = out.to_dataset()
    out_ds.attrs.update(out.attrs)
    tmp_cache_path = cache_path + ".tmp"
    out_ds.to_netcdf(tmp_cache_path)
    os.replace(tmp_cache_path, cache_path)
    _rebuilt_wave_cache_paths.add(cache_path)
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
    try:
        cache_signature = base_bootstrap_cache_signature(
            "wave_pathway",
            extra=dict(
                wave_fields=list(WAVE_FIELDS),
                wave_boxes=jsonable_value(WAVE_BOXES),
                latitude_projection=jsonable_value(WAVE_LATITUDE_PROJECTION),
                wave_filter_lat_bounds=list(WAVE_FILTER_LAT_BOUNDS),
                wave_source_contract_version=WAVE_SOURCE_CONTRACT_VERSION,
            ),
        )
        cached_rows = load_cached_table("wave_box_variance_bootstrap_metrics", cache_signature)
        if cached_rows is not None:
            append_cached_scalar_rows(cached_rows)
            save_table(cached_rows, "wave_box_variance_bootstrap_metrics.csv")
            plot_wave_summary(cached_rows)
            record_implemented("cached integrated CCEW wave-power and variance-fraction block-consistency metrics")
            return

        ensure_datasets_loaded()
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


def consume_wave_response_metrics():
    """Reuse the authoritative shared-metric table instead of re-regressing."""
    if not os.path.exists(WAVE_RESPONSE_METRIC_OUTPUT):
        raise FileNotFoundError(
            f"{WAVE_RESPONSE_METRIC_OUTPUT} is missing; run "
            "equatorial_wave_diagnostics.py Figure 11 before significance "
            "diagnostics"
        )
    table = pd.read_csv(WAVE_RESPONSE_METRIC_OUTPUT)
    required = {
        "wave_metrics_version",
        "wave_config_version",
        "precipitation_rms_mm_day",
        "precipitation_variance_mm2_day2",
        "covariance",
        "regression_slope",
        "sensitivity_subset",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(
            f"Shared wave-response table {WAVE_RESPONSE_METRIC_OUTPUT} "
            f"lacks {missing}"
        )
    versions = set(table.wave_config_version.dropna().astype(str))
    if versions != {WAVE_CONFIG_VERSION}:
        raise RuntimeError(
            f"Stale wave configuration {sorted(versions)} in "
            f"{WAVE_RESPONSE_METRIC_OUTPUT}; expected {WAVE_CONFIG_VERSION}"
        )
    metric_versions = set(table.wave_metrics_version.dropna().astype(str))
    if metric_versions != {METRICS_VERSION}:
        raise RuntimeError(
            f"Stale wave metrics {sorted(metric_versions)} in "
            f"{WAVE_RESPONSE_METRIC_OUTPUT}; expected {METRICS_VERSION}"
        )
    save_table(table, "wave_response_efficiency_metrics_shared.csv")
    record_implemented(
        "shared precipitation-amplitude, filtered-cloud-efficiency, and TOA-longwave-response metrics"
    )


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
                    if len(row) != 1:
                        raise RuntimeError(
                            f"Wave-box figure requires one {field} {metric} "
                            f"{comp} {wave} row; found {len(row)}"
                        )
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
    fields = [spec["name"] for spec in WAVE_PROFILE_FIELDS]
    waves = list(WAVE_PROFILE_WAVES)
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
                    raise RuntimeError(
                        f"Missing vertical wave-power profile for {field} "
                        f"{wave} {comp}"
                    )
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
    try:
        cache_signature = base_bootstrap_cache_signature(
            "vertical_wave_power_profile",
            extra=dict(
                waves=list(WAVE_PROFILE_WAVES),
                fields=jsonable_value(WAVE_PROFILE_FIELDS),
                levels_hpa=jsonable_value(WAVE_PROFILE_LEVELS_HPA),
                wave_filter_lat_bounds=list(WAVE_FILTER_LAT_BOUNDS),
                wave_source_contract_version=WAVE_SOURCE_CONTRACT_VERSION,
            ),
        )
        cached_rows = load_cached_table("vertical_wave_power_profile_bootstrap_metrics", cache_signature)
        if cached_rows is not None:
            append_cached_scalar_rows(cached_rows)
            save_table(cached_rows, "vertical_wave_power_profile_bootstrap_metrics.csv")
            plot_vertical_wave_power_profiles(cached_rows)
            record_implemented("cached vertical wave-power profile block-consistency metrics")
            return

        ensure_datasets_loaded()
        rows = []
        for wave_name in WAVE_PROFILE_WAVES:
            for field_spec in WAVE_PROFILE_FIELDS:
                for level_hpa in WAVE_PROFILE_LEVELS_HPA:
                    field_name = wave_profile_field_name(field_spec, level_hpa)
                    values_by_run = {}
                    for i in range(len(dss)):
                        raw = get_wave_filter_field(dss[i], field_name)
                        projected = project_wave_latitude(raw, wave_name)
                        filtered = filter_projected_wave(
                            projected, wave_name, i, field_name
                        )
                        values_by_run[i] = block_variance_values(filtered)

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
    waves = list(WAVE_LAG_WAVES)
    targets = list(WAVE_LAG_TARGET_FIELDS)
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
            key = (wave, target)
            if key not in lag_diffs:
                raise RuntimeError(
                    f"Missing wave-lag bootstrap response for {wave} {target}"
                )
            diffs = lag_diffs[key]
            for comp_name, ds in diffs.items():
                label = next(label for name, _, _, label in COMPARISONS if name == comp_name)
                color = colors_by_comp[comp_name]
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
    try:
        cache_signature = base_bootstrap_cache_signature(
            "wave_lag_correlation",
            extra=dict(
                waves=list(WAVE_LAG_WAVES),
                targets=list(WAVE_LAG_TARGET_FIELDS),
                max_days=float(WAVE_LAG_MAX_DAYS),
                block_days=float(WAVE_LAG_BLOCK_DAYS),
                wave_filter_lat_bounds=list(WAVE_FILTER_LAT_BOUNDS),
                wave_source_contract_version=WAVE_SOURCE_CONTRACT_VERSION,
            ),
        )
        cached_rows = load_cached_table("wave_lag_correlation_bootstrap_metrics", cache_signature)
        if cached_rows is not None:
            save_table(cached_rows, "wave_lag_correlation_bootstrap_metrics.csv")
            plot_wave_lag_correlation_differences(wave_lag_diffs_from_table(cached_rows))
            record_implemented("cached wave-amplitude lag-correlation block-consistency metrics")
            return

        ensure_datasets_loaded()
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
                blocks_by_run = {}
                for i in range(len(dss)):
                    target_raw = get_wave_filter_field(
                        dss[i], target_name
                    )
                    target_projected = project_wave_latitude(
                        target_raw, wave_name
                    )
                    target_ts = target_projected.mean(
                        dim="longitude", skipna=True
                    )
                    blocks_by_run[i] = block_lag_correlation_values(
                        amp_by_run[i], target_ts
                    )

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
# Reporting
# -----------------------------------------------------------------------------

def concise_metric_table(df):
    if len(df) == 0:
        return
    if "robust" not in df:
        raise RuntimeError("Scalar metric table lacks required 'robust' column")
    robust = df[df["robust"] == True].copy()
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


def print_mechanism_cues(metrics_df):
    print("\nMechanism-oriented cues:")
    if len(metrics_df) > 0:
        if "robust" not in metrics_df:
            raise RuntimeError(
                "Significance metric table lacks required 'robust' column"
            )
        wave_robust = metrics_df[
            (metrics_df["category"] == "wave_mediated_pathway")
            & (metrics_df["comparison"].isin(["u_minus_control", "T_minus_control"]))
            & (metrics_df["robust"] == True)
        ]
        if len(wave_robust) > 0:
            print("  Robust wave-box changes are present; compare U-control and T-control rows in wave_box_variance_bootstrap_metrics.csv for wind-mediated organisation.")
        org_robust = metrics_df[
            (metrics_df["category"].isin([
                "lpo_envelope_event_properties",
            ]))
            & (metrics_df["comparison"].isin(["u_minus_control", "T_minus_control"]))
            & (metrics_df["robust"] == True)
        ]
        if len(org_robust) > 0:
            print(
                "  Robust LPO-envelope property changes are present; use "
                "lpo_envelope_bootstrap_metrics.csv to compare equally "
                "weighted tracked systems."
            )
        thermal_robust = metrics_df[
            (metrics_df["category"] == "mean_state_pathway")
            & (metrics_df["diagnostic"].isin(["temperature", "static_stability_N2", "lapse_rate", "cloud_fraction"]))
            & (metrics_df["comparison"].isin(["u_minus_control", "T_minus_control"]))
            & (metrics_df["robust"] == True)
        ]
        if len(thermal_robust) > 0:
            print("  Robust temperature/static-stability/cloud-depth changes are present; these are the core thermal-pathway checks.")


def print_final_summary(metrics_df):
    print("\n" + "=" * 78)
    print("Block-resampling consistency summary")
    print("=" * 78)

    print("\nDiagnostics implemented and run:")
    if implemented:
        for item in implemented:
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
    print_mechanism_cues(metrics_df)

    print("\nAnswering the headline question:")
    print("  Use bootstrap_scalar_metrics.csv for scalar robustness, basic_convection_pdf_bootstrap_response.csv for PR/OLR/PW/CAPE distribution changes, wavenumber_variance_fraction_bootstrap_response.csv for planetary-scale organisation/noise checks, lpo_envelope_bootstrap_metrics.csv for LPO properties, the lat-height figures for coherent structures rather than isolated gridpoints, and the wave response, vertical-wave-power and lag-correlation tables for organised wave-mediated changes.")
    print("  Compare the independently diagnosed u-nudged and T-nudged changes directly; no scalar thermal-residual attribution is applied.")


# -----------------------------------------------------------------------------
# Main top-level workflow
# -----------------------------------------------------------------------------

print("\nRunning block-resampling consistency diagnostics")
precip_by_run = {}
if should_run_figure(1):
    run_mean_state_pathway()
if should_run_figure(2):
    precip_by_run = run_precipitation_response()
if should_run_figure(3):
    run_cloud_response(precip_by_run)
    run_conditional_precipitation_states(precip_by_run)
if should_run_figure(4):
    consume_wave_response_metrics()
    run_wave_pathway()
if should_run_figure(5):
    run_lpo_envelope_bootstrap()
if should_run_figure(6):
    run_basic_convection_pdf_bootstrap()
if should_run_figure(7):
    run_vertical_wave_power_profile_bootstrap()
if should_run_figure(8):
    run_wave_lag_correlation_bootstrap()
if should_run_figure(9):
    run_wavenumber_variance_bootstrap()

metrics_df = pd.DataFrame(scalar_metric_rows)
save_table(metrics_df, "bootstrap_scalar_metrics.csv")
print_final_summary(metrics_df)
if failed:
    raise RuntimeError(
        "One or more configured significance diagnostics failed:\n"
        + "\n".join(f"  - {message}" for message in failed)
    )
