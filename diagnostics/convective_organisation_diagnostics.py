#!/usr/bin/env python
# coding: utf-8

"""
Convective organisation diagnostics for MPAS-A nudging experiments.

The diagnostics here separate four related questions:

  * Lorg: instantaneous spatial arrangement/aggregation of convective objects.
  * Object size metrics: whether apparent organisation changes are actually
    changes in object size, count, or convective area.
  * Conditional Lorg: whether organisation differs between simulations at a
    fixed tropical-mean precipitation or fixed convective area fraction.
  * Persistence: lifetime/coherence of tracked convective objects.

The Lorg implementation follows the all-neighbour-distance idea of Biagioli and
Tompkins (2023): object centroids are converted to a Ripley/Besag L function,
then compared with a random reference with the same object count and domain.
Longitude is periodic both when labelling objects across the seam and when
measuring distances between centroids.
"""

import json
import os
import sys
import warnings

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (
    _THIS_DIR,
    os.path.abspath(os.path.join(_THIS_DIR, "..")),
    os.path.abspath(os.path.join(_THIS_DIR, "../..")),
    os.path.abspath(os.path.join(_THIS_DIR, "../../..")),
):
    if _path not in sys.path:
        sys.path.append(_path)

import dask
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy import ndimage

from diagnostic_helpers import *
from high_cloud_classification import (
    CLASSIFICATION_CACHE_VERSION,
    HighCloudThresholds,
    cached_classification_is_current as shared_cached_classification_is_current,
)

from pyflextrkr.gettracks import gettracknumbers as pyflextrkr_gettracknumbers
from pyflextrkr.tracksingle import trackclouds as pyflextrkr_trackclouds


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/organisation/")

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: Lorg time series
#   2: Lorg PDF and boxplot
#   3: object count/area/area-fraction distributions
#   4: Lorg conditioned on tropical-mean precipitation
#   5: Lorg conditioned on convective area fraction
#   6: PyFLEXTRKR-method track lifetime survival and boxplots
#   7: long-lived tracked area/precipitation contribution
#   8: PyFLEXTRKR-method track lifetime conditioned on tropical-mean precipitation
#   9: PyFLEXTRKR-method track lifetime conditioned on convective area fraction
#   10: propagation-aware pattern autocorrelation
#
# WHAT TO LOOK FOR:
#   * Robust Lorg or object-count changes at fixed tropical precipitation imply
#     organisation changes beyond a simple mean-rain-rate shift.
#   * Compare object area, count, and convective area fraction before assigning
#     a change to aggregation; apparent Lorg changes can come from object-size
#     changes.
#   * Longer object tracks or larger long-lived area/precipitation fractions
#     mean organised convective patterns persist for longer.
#   * PyFLEXTRKR is used here to define overlap-based tracks and merge/split
#     bookkeeping; the separate block-resampling script tests whether the
#     resulting metric differences are consistent across time chunks.
figures_to_run = []
figsave = None

# Expensive organisation and persistence tables are cached in save_loc. Metadata
# must match active settings; older CSV-only caches can be adopted once if their
# source/simulation columns still fit the current configuration.
USE_ORGANISATION_METRIC_CACHE = True
FORCE_RECOMPUTE_ORGANISATION_CACHE = False
FORCE_RECOMPUTE_PERSISTENCE_CACHE = False
ALLOW_LEGACY_ORGANISATION_CACHE = False

USE_ORGANISATION_METRIC_CACHE = (
    USE_ORGANISATION_METRIC_CACHE
    and not env_flag("DISABLE_ORGANISATION_METRIC_CACHE")
)
ALLOW_LEGACY_ORGANISATION_CACHE = (
    ALLOW_LEGACY_ORGANISATION_CACHE
    and not env_flag("DISABLE_LEGACY_ORGANISATION_CACHE")
)
FORCE_RECOMPUTE_ORGANISATION_CACHE = (
    FORCE_RECOMPUTE_ORGANISATION_CACHE
    or env_flag("FORCE_RECOMPUTE_ORGANISATION_CACHE")
)
FORCE_RECOMPUTE_PERSISTENCE_CACHE = (
    FORCE_RECOMPUTE_PERSISTENCE_CACHE
    or env_flag("FORCE_RECOMPUTE_PERSISTENCE_CACHE")
)

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control", "u_nudge", "T_nudge"]
latlon_file = ["latlon.2000-*.nc", "latlon.2000-*.nc", "latlon.2000-*.nc"]
sim_names = ["control", "nudged u", "nudged T"]
cols = ["k", "b", "r"]

ndays = 200
dt = 6
analysis_end_date = "2000-11-21"
time_format = "time"

TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
PERSISTENCE_LAT_BOUNDS = (-10.0, 10.0)
EARTH_RADIUS_KM = 6371.0
LONGITUDE_PERIOD_DEG = 360.0
STEFAN_BOLTZMANN = 5.670374419e-8

# Default object sources are intentionally small because every source is used
# for Lorg and PyFLEXTRKR tracking. Turn on sensitivity sources only when
# checking threshold robustness.
RUN_SENSITIVITY_OBJECT_SOURCES = False

# MCS-like objects use broad absolute rain-rate and cold-cloud-shield
# thresholds; percentile and very-heavy-rain thresholds are sensitivity tests.
PRIMARY_PRECIP_OBJECT_SPECS = [
    dict(name="precip_ge_1mmh", label="Precip >= 1 mm h$^{-1}$", threshold=24.0, sense="above"),
]
SENSITIVITY_PRECIP_OBJECT_SPECS = [
    dict(name="precip_ge_2mmh", label="Precip >= 2 mm h$^{-1}$", threshold=48.0, sense="above"),
    dict(name="precip_ge_8mmh", label="Precip >= 8 mm h$^{-1}$", threshold=192.0, sense="above"),
    dict(name="precip_p95", label="Precip p95 sensitivity", percentile=95.0, sense="above"),
    dict(name="precip_p99", label="Precip p99 sensitivity", percentile=99.0, sense="above"),
]
PRECIP_OBJECT_SPECS = PRIMARY_PRECIP_OBJECT_SPECS + (
    SENSITIVITY_PRECIP_OBJECT_SPECS if RUN_SENSITIVITY_OBJECT_SOURCES else []
)

PRIMARY_OLR_OBJECT_SPECS = [
    dict(
        name="olr_tb241",
        label=r"OLR $T_b^\ast \leq$ 241 K",
        threshold=STEFAN_BOLTZMANN * 241.0 ** 4,
        field="olr",
        sense="below",
    ),
]
SENSITIVITY_OLR_OBJECT_SPECS = [
    dict(
        name="olr_tb233",
        label=r"OLR $T_b^\ast \leq$ 233 K",
        threshold=STEFAN_BOLTZMANN * 233.0 ** 4,
        field="olr",
        sense="below",
    ),
    dict(name="olr_cold_p05", label="Cold OLR anomaly p5", percentile=5.0, sense="below"),
]
OLR_OBJECT_SPECS = PRIMARY_OLR_OBJECT_SPECS + (
    SENSITIVITY_OLR_OBJECT_SPECS if RUN_SENSITIVITY_OBJECT_SOURCES else []
)

ENABLE_OLR_OBJECTS = True
ENABLE_CLOUD_OBJECTS = True

# The refined high-cloud diagnostic is the single producer of cloud classes.
# Organisation diagnostics fail if that shared cache is missing or stale.
CLOUD_CLASSIFICATION_CACHE_DIRS = [
    diagnostic_cache_dir("classification_cache", root_dir=root_dir),
]

CLOUD_OBJECT_KEYS_PRIMARY = [
    "active_deep_convection_high_cloud_mask",
    "convectively_generated_high_cloud_mask",
]
CLOUD_OBJECT_KEYS_SENSITIVITY = [
    "column_connected_mask",
    "high_cloud_mask",
    "cirrus_like_high_cloud_mask",
    "unclassified_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
]
CLOUD_OBJECT_KEYS = CLOUD_OBJECT_KEYS_PRIMARY + (
    CLOUD_OBJECT_KEYS_SENSITIVITY if RUN_SENSITIVITY_OBJECT_SOURCES else []
)

# Shared high-cloud thresholds used to validate the refined classification cache.
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
ANVIL_RADIUS_DEG = 5.0
ANVIL_TIME_STEPS = 1

MIN_OBJECT_CELLS = 1
MIN_OBJECT_AREA_KM2 = 0.0
PAIR_DISTANCE_NBINS = 80
REFERENCE_REALIZATIONS = 96
REFERENCE_RANDOM_SEED = 202405
MIN_OBJECTS_FOR_LORG = 2
MIN_CONDITION_COUNT = 3
CONDITION_NBINS = 4

# Persistence follows the core PyFLEXTRKR overlap tracker (Feng et al., 2023):
# adjacent objects are linked by bidirectional overlap, with split/merge branches
# recorded separately from the continuously tracked largest branch.
PYFLEXTRKR_WORK_DIR = diagnostic_cache_dir("pyflextrkr_tracking", root_dir=root_dir)
PYFLEXTRKR_NMAXLINKS = 50
PYFLEXTRKR_FILLVAL = -9999
FORCE_RECOMPUTE_PYFLEXTRKR_CACHE = (
    FORCE_RECOMPUTE_PERSISTENCE_CACHE
    or env_flag("FORCE_RECOMPUTE_PYFLEXTRKR_CACHE")
)
TRACK_MIN_OVERLAP_FRACTION = 0.50
TRACK_MIN_AREA_KM2 = 800.0
TRACK_MIN_LIFETIME_STEPS_FOR_SUMMARY = 1
LONG_LIVED_THRESHOLDS_HOURS = [12.0, 24.0, 48.0]
TRACK_CONDITION_NBINS = 4
MIN_TRACK_BIN_COUNT = 3

PERSISTENCE_MAX_LAG_DAYS = 12.0
PERSISTENCE_MAX_LON_SHIFT_CELLS = 12

THRESHOLD_SAMPLE_LIMITS = dict(max_time=160, max_lat=90, max_lon=180)

VAR_ALIASES = {
    "precip": ["precip", "precip_rate", "pr", "rain_rate"],
    "rainc": ["rainc"],
    "rainnc": ["rainnc"],
    "i_rainc": ["i_rainc"],
    "i_rainnc": ["i_rainnc"],
    "olr": ["olr", "olrtoa", "OLR"],
    "pw": ["pw", "precipw", "prw"],
}

SOURCE_LABELS = {
    "precip_ge_1mmh": r"Precip $\geq$ 1 mm h$^{-1}$",
    "precip_ge_2mmh": r"Precip $\geq$ 2 mm h$^{-1}$",
    "precip_ge_8mmh": r"Precip $\geq$ 8 mm h$^{-1}$",
    "precip_p95": "Precip p95 sensitivity",
    "precip_p99": "Precip p99 sensitivity",
    "olr_tb241": r"OLR $T_b^\ast \leq$ 241 K",
    "olr_tb233": r"OLR $T_b^\ast \leq$ 233 K",
    "olr_cold_p05": "Cold OLR anomaly p5",
    "high_cloud_mask": "High cloud",
    "column_connected_mask": "Column-connected high cloud",
    "active_deep_convection_high_cloud_mask": "Active deep convection",
    "convectively_generated_high_cloud_mask": "Convectively generated high cloud",
    "unclassified_high_cloud_mask": "Ambiguous/unclassified high cloud",
    "cirrus_like_high_cloud_mask": "Cirrus-like high cloud",
    "nonprecipitating_high_cloud_mask": "Nonprecipitating high cloud",
}

SOURCE_ORDER = [spec["name"] for spec in PRECIP_OBJECT_SPECS]
SOURCE_ORDER += [spec["name"] for spec in OLR_OBJECT_SPECS]
SOURCE_ORDER += CLOUD_OBJECT_KEYS

needed_vars = sorted(
    set(
        VAR_ALIASES["rainc"]
        + VAR_ALIASES["rainnc"]
        + VAR_ALIASES["i_rainc"]
        + VAR_ALIASES["i_rainnc"]
        + VAR_ALIASES["precip"]
        + VAR_ALIASES["olr"]
        + VAR_ALIASES["pw"]
    )
)

latlon_chunks = {"Time": 8, "latitude": 90, "longitude": 180}
dask.config.set({"array.slicing.split_large_chunks": True})


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


def save_figure(number, fig, stem, dpi=300):
    global figsave
    if figsave is None:
        figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)
    return figsave.save(fig, stem, number=number, dpi=dpi)


# -----------------------------------------------------------------------------
# Metric/persistence caches
# -----------------------------------------------------------------------------

METRIC_CACHE_VERSION = "organisation_metrics_v2"
TRACK_PERSISTENCE_CACHE_VERSION = "pyflextrkr_track_persistence_v2"
PATTERN_PERSISTENCE_CACHE_VERSION = "shifted_pattern_persistence_v1"

ORGANISATION_CACHE_FILES = {
    "metric_metadata": "organisation_metric_cache_metadata.json",
    "track_metadata": "organisation_track_persistence_cache_metadata.json",
    "pattern_metadata": "organisation_pattern_persistence_cache_metadata.json",
    "thresholds": "organisation_object_thresholds.csv",
    "timeseries": "organisation_lorg_timeseries.csv",
    "summary": "organisation_lorg_summary.csv",
    "tracked_objects": "organisation_tracked_objects.csv",
    "track_lifetimes": "organisation_track_lifetimes.csv",
    "long_lived_contribution": "organisation_long_lived_contribution.csv",
    "pattern_persistence": "equatorial_shifted_pattern_autocorrelation.csv",
}

EMPTY_CACHE_COLUMNS = {
    "tracked_objects": [
        "node_id",
        "source",
        "source_label",
        "simulation",
        "Time",
        "time_index",
        "time_days",
        "dt_hours",
        "domain_area_km2",
        "scene_convective_area_fraction",
        "scene_precip_area_rate",
        "label",
        "area_km2",
        "precip_area_rate",
        "track_id",
        "tracking_method",
    ],
    "track_lifetimes": [
        "source",
        "source_label",
        "simulation",
        "track_id",
        "first_time_index",
        "last_time_index",
        "lifetime_hours",
        "n_timesteps",
        "mean_area_km2",
        "max_area_km2",
        "mean_tropical_precip",
        "mean_convective_area_fraction",
    ],
    "long_lived_contribution": [
        "source",
        "source_label",
        "simulation",
        "Time",
        "time_days",
        "threshold_hours",
        "tropical_mean_precip",
        "convective_area_fraction",
        "long_lived_area_fraction_of_domain",
        "long_lived_area_fraction_of_objects",
        "long_lived_precip_fraction_of_objects",
    ],
    "pattern_persistence": [
        "simulation",
        "field",
        "field_label",
        "lag_days",
        "autocorrelation",
    ],
}


def organisation_cache_path(name):
    return os.path.join(save_loc, ORGANISATION_CACHE_FILES[name])


def jsonable_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable_value(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def object_spec_signature(specs):
    keys = ("name", "field", "threshold", "percentile", "sense")
    return [
        {key: jsonable_value(spec[key]) for key in keys if key in spec}
        for spec in specs
    ]


def base_data_cache_signature():
    return dict(
        root_dir=root_dir,
        data_locs=list(data_locs),
        latlon_file=list(latlon_file),
        sim_names=list(sim_names),
        ndays=int(ndays),
        dt=float(dt),
        analysis_end_date=analysis_end_date,
        time_format=time_format,
        tropical_lat_bounds=list(TROPICAL_LAT_BOUNDS),
    )


def object_definition_cache_signature():
    signature = base_data_cache_signature()
    signature.update(
        source_order=list(SOURCE_ORDER),
        precip_object_specs=object_spec_signature(PRECIP_OBJECT_SPECS),
        olr_object_specs=object_spec_signature(OLR_OBJECT_SPECS),
        cloud_object_keys=list(CLOUD_OBJECT_KEYS),
        run_sensitivity_object_sources=bool(RUN_SENSITIVITY_OBJECT_SOURCES),
        enable_olr_objects=bool(ENABLE_OLR_OBJECTS),
        enable_cloud_objects=bool(ENABLE_CLOUD_OBJECTS),
        cloud_classification_source="refined_high_cloud_classification_diagnostics.py",
        cloud_classification_cache_version=CLASSIFICATION_CACHE_VERSION,
        shared_cloud_thresholds=dict(
            cf_thresh=float(CF_THRESH),
            z_high=float(Z_HIGH),
            z_low_min=float(Z_LOW_MIN),
            z_low_max=float(Z_LOW_MAX),
            z_mid_min=float(Z_MID_MIN),
            z_mid_max=float(Z_MID_MAX),
            z_upper_min=float(Z_UPPER_MIN),
            p_deep=float(P_DEEP),
            p_weak=float(P_WEAK),
            max_clear_gap=jsonable_value(MAX_CLEAR_GAP),
            classifier="shared 3-D hydrometeor-aware high-cloud classification",
        ),
        threshold_sample_limits=jsonable_value(THRESHOLD_SAMPLE_LIMITS),
        min_object_cells=int(MIN_OBJECT_CELLS),
        min_object_area_km2=float(MIN_OBJECT_AREA_KM2),
    )
    return signature


def metric_cache_signature():
    signature = object_definition_cache_signature()
    signature.update(
        cache_version=METRIC_CACHE_VERSION,
        pair_distance_nbins=int(PAIR_DISTANCE_NBINS),
        reference_realizations=int(REFERENCE_REALIZATIONS),
        reference_random_seed=int(REFERENCE_RANDOM_SEED),
        min_objects_for_lorg=int(MIN_OBJECTS_FOR_LORG),
    )
    return signature


def track_persistence_cache_signature():
    signature = object_definition_cache_signature()
    signature.update(
        cache_version=TRACK_PERSISTENCE_CACHE_VERSION,
        pyflextrkr_nmaxlinks=int(PYFLEXTRKR_NMAXLINKS),
        pyflextrkr_fillval=int(PYFLEXTRKR_FILLVAL),
        track_min_overlap_fraction=float(TRACK_MIN_OVERLAP_FRACTION),
        track_min_area_km2=float(TRACK_MIN_AREA_KM2),
        track_min_lifetime_steps_for_summary=int(TRACK_MIN_LIFETIME_STEPS_FOR_SUMMARY),
        long_lived_thresholds_hours=[float(v) for v in LONG_LIVED_THRESHOLDS_HOURS],
    )
    return signature


def pattern_persistence_cache_signature():
    signature = base_data_cache_signature()
    signature.update(
        cache_version=PATTERN_PERSISTENCE_CACHE_VERSION,
        persistence_lat_bounds=list(PERSISTENCE_LAT_BOUNDS),
        persistence_max_lag_days=float(PERSISTENCE_MAX_LAG_DAYS),
        persistence_max_lon_shift_cells=int(PERSISTENCE_MAX_LON_SHIFT_CELLS),
    )
    return signature


def cache_force_enabled(kind):
    if kind == "metric":
        return FORCE_RECOMPUTE_ORGANISATION_CACHE
    return FORCE_RECOMPUTE_PERSISTENCE_CACHE


def read_cache_metadata(name):
    path = organisation_cache_path(name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_cache_metadata(name, signature):
    if not save:
        return
    os.makedirs(save_loc, exist_ok=True)
    path = organisation_cache_path(name)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(jsonable_value(signature), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def cache_metadata_status(name, signature, kind, label):
    if not USE_ORGANISATION_METRIC_CACHE:
        return "miss"
    if cache_force_enabled(kind):
        print(f"Ignoring cached {label}: force recompute is enabled.")
        return "miss"
    try:
        cached = read_cache_metadata(name)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Ignoring cached {label}: could not read metadata ({exc}).")
        return "miss"
    if cached is None:
        if ALLOW_LEGACY_ORGANISATION_CACHE:
            print(f"Checking legacy cached {label}: metadata file is missing.")
            return "legacy"
        return "miss"

    signature = jsonable_value(signature)
    changed = [
        key
        for key in sorted(set(cached) | set(signature))
        if cached.get(key) != signature.get(key)
    ]
    if changed:
        shown = ", ".join(changed[:8])
        suffix = "" if len(changed) <= 8 else ", ..."
        print(f"Ignoring cached {label}: metadata changed ({shown}{suffix}).")
        return "miss"
    return "match"


def read_cache_table(name):
    path = organisation_cache_path(name)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=EMPTY_CACHE_COLUMNS.get(name, []))


def write_cache_table(name, df):
    if not save or df is None:
        return
    os.makedirs(save_loc, exist_ok=True)
    if len(df.columns) == 0:
        df = pd.DataFrame(columns=EMPTY_CACHE_COLUMNS.get(name, []))
    df.to_csv(organisation_cache_path(name), index=False)


def frame_has_current_simulations(df, label, require_all=True):
    if len(df) == 0 or "simulation" not in df.columns:
        return True
    expected = set(sim_names)
    actual = set(df["simulation"].dropna().astype(str).unique())
    unexpected = actual - expected
    if unexpected:
        print(f"Ignoring cached {label}: unexpected simulations {sorted(unexpected)}.")
        return False
    if require_all and actual != expected:
        print(f"Ignoring cached {label}: simulations differ from current configuration.")
        return False
    return True


def frame_has_only_current_sources(df, label):
    if len(df) == 0 or "source" not in df.columns:
        return True
    expected = set(SOURCE_ORDER)
    actual = set(df["source"].dropna().astype(str).unique())
    unexpected = actual - expected
    if unexpected:
        print(f"Ignoring cached {label}: unexpected sources {sorted(unexpected)}.")
        return False
    return True


def organisation_metric_summary(metrics_df):
    return (
        metrics_df
        .groupby(["source", "source_label", "simulation"], as_index=False)
        [["lorg_km", "object_count", "convective_area_fraction"]]
        .mean(numeric_only=True)
    )


def load_cached_metric_products():
    metadata_status = cache_metadata_status(
        "metric_metadata",
        metric_cache_signature(),
        "metric",
        "organisation metrics",
    )
    if metadata_status == "miss":
        return None

    metrics_df = read_cache_table("timeseries")
    if metrics_df is None:
        return None
    if len(metrics_df) == 0:
        print("Ignoring cached organisation metrics: time series table is empty.")
        return None
    if not frame_has_current_simulations(metrics_df, "organisation metrics"):
        return None
    if not frame_has_only_current_sources(metrics_df, "organisation metrics"):
        return None

    threshold_table = read_cache_table("thresholds")
    if threshold_table is None:
        threshold_table = pd.DataFrame()
    summary = read_cache_table("summary")
    if summary is None or len(summary) == 0:
        summary = organisation_metric_summary(metrics_df)

    if metadata_status == "legacy":
        write_cache_metadata("metric_metadata", metric_cache_signature())
    print(f"Using cached organisation metrics from {save_loc}")
    return threshold_table, metrics_df, summary


def write_metric_cache(threshold_table, metrics_df, summary):
    write_cache_table("thresholds", threshold_table)
    write_cache_table("timeseries", metrics_df)
    write_cache_table("summary", summary)
    write_cache_metadata("metric_metadata", metric_cache_signature())


def load_cached_track_persistence_products():
    metadata_status = cache_metadata_status(
        "track_metadata",
        track_persistence_cache_signature(),
        "persistence",
        "track persistence metrics",
    )
    if metadata_status == "miss":
        return None

    missing = [
        name
        for name in ("tracked_objects", "track_lifetimes", "long_lived_contribution")
        if not os.path.exists(organisation_cache_path(name))
    ]
    if missing:
        print(f"Ignoring cached track persistence metrics: missing {', '.join(missing)}.")
        return None

    objects_df = read_cache_table("tracked_objects")
    tracks_df = read_cache_table("track_lifetimes")
    contribution_df = read_cache_table("long_lived_contribution")
    for label, frame in (
        ("tracked objects", objects_df),
        ("track lifetimes", tracks_df),
        ("long-lived contribution", contribution_df),
    ):
        if not frame_has_current_simulations(frame, label, require_all=False):
            return None
        if not frame_has_only_current_sources(frame, label):
            return None

    if metadata_status == "legacy":
        write_cache_metadata("track_metadata", track_persistence_cache_signature())
    print(f"Using cached track persistence metrics from {save_loc}")
    return objects_df, tracks_df, contribution_df


def write_track_persistence_cache(objects_df, tracks_df, contribution_df):
    write_cache_table("tracked_objects", objects_df)
    write_cache_table("track_lifetimes", tracks_df)
    write_cache_table("long_lived_contribution", contribution_df)
    write_cache_metadata("track_metadata", track_persistence_cache_signature())


def load_cached_pattern_persistence():
    metadata_status = cache_metadata_status(
        "pattern_metadata",
        pattern_persistence_cache_signature(),
        "persistence",
        "pattern persistence metrics",
    )
    if metadata_status == "miss":
        return None
    persistence_df = read_cache_table("pattern_persistence")
    if persistence_df is None:
        return None
    if not frame_has_current_simulations(persistence_df, "pattern persistence metrics"):
        return None
    if metadata_status == "legacy":
        write_cache_metadata("pattern_metadata", pattern_persistence_cache_signature())
    print(f"Using cached pattern persistence metrics from {save_loc}")
    return persistence_df


def write_pattern_persistence_cache(persistence_df):
    write_cache_table("pattern_persistence", persistence_df)
    write_cache_metadata("pattern_metadata", pattern_persistence_cache_signature())


# -----------------------------------------------------------------------------
# Basic data helpers
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


def get_precip_rate(ds):
    direct = find_var(ds, "precip")
    if direct is not None:
        pr = finite_or_nan(ds[direct])
        pr.name = "precip"
        pr.attrs.setdefault("units", "mm day-1")
        return pr

    accum = total_precip_accum(ds)
    pr = precip_rate_from_accum(accum, dt=dt)
    pr = finite_or_nan(pr)
    pr.name = "precip"
    pr.attrs["units"] = "mm day-1"
    return pr


def select_common_time(da, ref):
    if "Time" not in da.dims or "Time" not in ref.dims:
        return da
    common_time = da.Time.to_index().intersection(ref.Time.to_index())
    return da.sel(Time=common_time)


def time_aligned_field(ds, key, ref, required=False):
    name = find_var(ds, key, required=required)
    if name is None:
        return None
    out = finite_or_nan(ds[name])
    out = select_common_time(out, ref)
    out.name = key
    return out


def time_days_from_coord(time_coord):
    values = np.asarray(time_coord.values)
    if np.issubdtype(values.dtype, np.datetime64):
        return (values - values[0]) / np.timedelta64(1, "D")
    return values.astype(float) * dt / 24.0


def median_time_step_days(da):
    values = np.asarray(da.Time.values)
    if values.size < 2:
        return dt / 24.0
    if np.issubdtype(values.dtype, np.datetime64):
        return float(np.median(np.diff(values) / np.timedelta64(1, "D")))
    return float(np.median(np.diff(values))) * dt / 24.0


def weighted_tropical_mean(da, lat_bounds=TROPICAL_LAT_BOUNDS):
    x = subset_lat(finite_or_nan(da), lat_bounds=lat_bounds)
    dims = [dim for dim in ("latitude", "longitude") if dim in x.dims]
    if "latitude" in x.dims:
        weights = np.cos(np.deg2rad(x.latitude))
        return x.weighted(weights).mean(dim=dims, skipna=True)
    return x.mean(dim=dims, skipna=True)


def stride_for_limits(da, limits):
    dim_names = {"max_time": "Time", "max_lat": "latitude", "max_lon": "longitude"}
    indexers = {}
    for dim, max_size in limits.items():
        dim_name = dim_names.get(dim, dim)
        if dim_name in da.dims and da.sizes[dim_name] > max_size:
            step = int(np.ceil(da.sizes[dim_name] / max_size))
            indexers[dim_name] = slice(0, None, step)
    return da.isel(indexers) if indexers else da


def sampled_values(da, lat_bounds=TROPICAL_LAT_BOUNDS, limits=THRESHOLD_SAMPLE_LIMITS):
    x = subset_lat(finite_or_nan(da), lat_bounds=lat_bounds)
    x = stride_for_limits(x, limits)
    vals = np.asarray(x.values).ravel()
    return vals[np.isfinite(vals)]


def source_label(source_name):
    return SOURCE_LABELS.get(source_name, source_name.replace("_", " "))


def ordered_sources(metrics_df):
    present = set(metrics_df.source.unique())
    ordered = [name for name in SOURCE_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "-").replace(".", "p").replace("-", "m")


# -----------------------------------------------------------------------------
# Grid/object geometry
# -----------------------------------------------------------------------------


def coordinate_edges(values):
    vals = np.asarray(values, dtype=float)
    if vals.size == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5], dtype=float)
    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (vals[:-1] + vals[1:])
    edges[0] = vals[0] - 0.5 * (vals[1] - vals[0])
    edges[-1] = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return edges


def grid_cell_area_km2(lat, lon):
    lat_edges = np.clip(coordinate_edges(lat), -90.0, 90.0)
    lon_edges = coordinate_edges(lon)
    dlon = np.abs(np.diff(lon_edges))
    lat_factor = np.diff(np.sin(np.deg2rad(lat_edges)))
    area = (EARTH_RADIUS_KM ** 2) * np.deg2rad(dlon)[None, :] * np.abs(lat_factor)[:, None]
    return xr.DataArray(
        area,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="cell_area",
        attrs={"units": "km2"},
    )


def domain_area_km2(lat_bounds, lon_width=LONGITUDE_PERIOD_DEG):
    lo, hi = sorted(lat_bounds)
    return (
        EARTH_RADIUS_KM ** 2
        * np.deg2rad(lon_width)
        * (np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo)))
    )


def haversine_distances_km(lat1, lon1, lat2, lon2, lon_period=LONGITUDE_PERIOD_DEG):
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    lon1 = np.asarray(lon1, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    dlat = lat2 - lat1
    dlon_deg = lon2 - lon1
    dlon_deg = (dlon_deg + 0.5 * lon_period) % lon_period - 0.5 * lon_period
    dlon = np.deg2rad(dlon_deg)
    a = (
        np.sin(0.5 * dlat) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * dlon) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def max_domain_distance_km(lat_bounds=TROPICAL_LAT_BOUNDS):
    lo, hi = sorted(lat_bounds)
    lat_samples = np.linspace(lo, hi, 9)
    lon_samples = np.array([0.0, 90.0, 180.0, 270.0])
    lat_grid, lon_grid = np.meshgrid(lat_samples, lon_samples, indexing="ij")
    lat = lat_grid.ravel()
    lon = lon_grid.ravel()
    d = pairwise_distances_km(lat, lon)
    return float(np.nanmax(d)) if d.size else np.nan


def pairwise_distances_km(lat, lon, block=512):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = lat.size
    if n < 2:
        return np.array([], dtype=float)

    chunks = []
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        for j0 in range(i0, n, block):
            j1 = min(j0 + block, n)
            d = haversine_distances_km(
                lat[i0:i1, None],
                lon[i0:i1, None],
                lat[None, j0:j1],
                lon[None, j0:j1],
            )
            if i0 == j0:
                ii = np.triu_indices(i1 - i0, k=1)
                chunks.append(d[ii])
            else:
                chunks.append(d.ravel())
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)


def merge_periodic_longitude_labels(labels):
    """Merge connected components touching both sides of a periodic longitude seam."""
    labels = np.asarray(labels, dtype=np.int32)
    max_label = int(labels.max())
    if max_label == 0 or labels.shape[1] < 2:
        return labels

    parent = np.arange(max_label + 1, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a <= 0 or b <= 0:
            return
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra

    nlat = labels.shape[0]
    for j in range(nlat):
        left = labels[j, 0]
        if left <= 0:
            continue
        for jj in range(max(0, j - 1), min(nlat, j + 2)):
            union(left, labels[jj, -1])

    roots = np.array([find(i) for i in range(max_label + 1)], dtype=np.int32)
    unique_roots = [root for root in np.unique(roots[1:]) if root > 0]
    remap = {root: index + 1 for index, root in enumerate(unique_roots)}
    out = np.zeros_like(labels)
    positive = labels > 0
    out[positive] = np.array([remap[roots[label]] for label in labels[positive]], dtype=np.int32)
    return out


def label_objects_with_properties(mask2d, area2d, precip2d=None):
    """
    Label one thresholded scene and return labels plus object properties.

    The object mask uses 8-connectivity and merges objects across periodic
    longitude. This is the only new object-identification step in this script;
    the masks themselves come from existing precipitation/cloud diagnostics
    wherever available.
    """
    mask2d = mask2d.transpose("latitude", "longitude")
    area2d = area2d.sel(latitude=mask2d.latitude, longitude=mask2d.longitude)
    mask_vals = np.asarray(mask2d.fillna(False).values, dtype=bool)
    area_vals = np.asarray(area2d.values, dtype=float)
    precip_vals = None
    if precip2d is not None:
        precip2d = precip2d.transpose("latitude", "longitude")
        precip2d = precip2d.sel(latitude=mask2d.latitude, longitude=mask2d.longitude)
        precip_vals = np.asarray(precip2d.values, dtype=float)

    if not np.any(mask_vals):
        empty = pd.DataFrame(
            columns=[
                "label",
                "cell_count",
                "area_km2",
                "precip_area_rate",
                "centroid_lat",
                "centroid_lon",
            ]
        )
        return np.zeros(mask_vals.shape, dtype=np.int32), empty

    structure = np.ones((3, 3), dtype=np.int8)
    labels, _ = ndimage.label(mask_vals, structure=structure)
    labels = merge_periodic_longitude_labels(labels)

    lat_vals = np.asarray(mask2d.latitude.values, dtype=float)
    lon_vals = np.asarray(mask2d.longitude.values, dtype=float) % LONGITUDE_PERIOD_DEG
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")

    rows = []
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        obj = labels == label_id
        cell_count = int(np.count_nonzero(obj))
        area = float(np.nansum(area_vals[obj]))
        if cell_count < MIN_OBJECT_CELLS or area < MIN_OBJECT_AREA_KM2:
            labels[obj] = 0
            continue

        weights = area_vals[obj]
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
        if precip_vals is None:
            precip_area_rate = np.nan
        else:
            pr = np.where(np.isfinite(precip_vals[obj]), precip_vals[obj], 0.0)
            precip_area_rate = float(np.nansum(pr * weights))
        centroid_lat = float(np.average(lat_grid[obj], weights=weights))
        theta = 2.0 * np.pi * lon_grid[obj] / LONGITUDE_PERIOD_DEG
        sin_mean = np.average(np.sin(theta), weights=weights)
        cos_mean = np.average(np.cos(theta), weights=weights)
        centroid_lon = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0)

        rows.append(
            dict(
                label=int(label_id),
                cell_count=cell_count,
                area_km2=area,
                precip_area_rate=precip_area_rate,
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
            )
        )

    if not rows:
        return np.zeros(mask_vals.shape, dtype=np.int32), pd.DataFrame(
            columns=[
                "label",
                "cell_count",
                "area_km2",
                "precip_area_rate",
                "centroid_lat",
                "centroid_lon",
            ]
        )

    valid_labels = {row["label"] for row in rows}
    labels[~np.isin(labels, list(valid_labels))] = 0
    return labels.astype(np.int32), pd.DataFrame(rows)


def object_properties_from_mask(mask2d, area2d):
    """Return object centroid/area properties for one thresholded scene."""
    _, objects = label_objects_with_properties(mask2d, area2d)
    return objects


# -----------------------------------------------------------------------------
# Lorg metric
# -----------------------------------------------------------------------------


def ripley_besag_l(pair_distances, n_objects, domain_area, r_edges):
    pair_distances = np.asarray(pair_distances, dtype=float)
    if n_objects < 2 or pair_distances.size == 0:
        return np.full(r_edges.size - 1, np.nan, dtype=float)

    counts, _ = np.histogram(pair_distances, bins=r_edges)
    cumulative_pairs = np.cumsum(counts).astype(float)
    k_r = domain_area * (2.0 * cumulative_pairs) / (n_objects * (n_objects - 1))
    return np.sqrt(np.maximum(k_r, 0.0) / np.pi)


def random_centroids(n_objects, lat_bounds=TROPICAL_LAT_BOUNDS, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = sorted(lat_bounds)
    sin_lo = np.sin(np.deg2rad(lo))
    sin_hi = np.sin(np.deg2rad(hi))
    lat = np.rad2deg(np.arcsin(sin_lo + rng.random(n_objects) * (sin_hi - sin_lo)))
    lon = rng.random(n_objects) * LONGITUDE_PERIOD_DEG
    return lat, lon


def reference_l_function(n_objects, domain_area, r_edges, rng, cache):
    key = (int(n_objects), int(PAIR_DISTANCE_NBINS), tuple(np.round(r_edges, 6)))
    if key in cache:
        return cache[key]

    values = []
    for _ in range(REFERENCE_REALIZATIONS):
        lat, lon = random_centroids(n_objects, rng=rng)
        d = pairwise_distances_km(lat, lon)
        values.append(ripley_besag_l(d, n_objects, domain_area, r_edges))

    ref = np.nanmean(np.vstack(values), axis=0)
    cache[key] = ref
    return ref


def lorg_from_objects(objects, domain_area, r_edges, rng, reference_cache):
    n_objects = len(objects)
    if n_objects < MIN_OBJECTS_FOR_LORG:
        return np.nan, np.nan

    distances = pairwise_distances_km(objects.centroid_lat.values, objects.centroid_lon.values)
    l_obs = ripley_besag_l(distances, n_objects, domain_area, r_edges)
    l_ref = reference_l_function(n_objects, domain_area, r_edges, rng, reference_cache)
    r = r_edges[1:]
    rmax = float(r_edges[-1])
    lorg_km = float(np.trapz(l_obs - l_ref, r) / rmax)
    lorg_normalised = float(lorg_km / rmax)
    return lorg_km, lorg_normalised


def compute_organisation_timeseries(mask, fields, source_name, rng, reference_cache):
    mask = subset_lat(mask.fillna(False), lat_bounds=TROPICAL_LAT_BOUNDS)
    if "Time" not in mask.dims:
        warnings.warn(f"Skipping {source_name}: mask has no Time dimension")
        return pd.DataFrame()

    precip_mean = fields["mean_precip"].sel(Time=mask.Time)
    mask, precip_mean = xr.align(mask, precip_mean, join="inner")
    area2d = grid_cell_area_km2(mask.latitude.values, mask.longitude.values)
    domain_area = float(area2d.sum().values)
    r_edges = np.linspace(0.0, max_domain_distance_km(TROPICAL_LAT_BOUNDS), PAIR_DISTANCE_NBINS + 1)
    time_days = time_days_from_coord(mask.Time)

    rows = []
    for it in range(mask.sizes["Time"]):
        mask2d = mask.isel(Time=it).load()
        objects = object_properties_from_mask(mask2d, area2d)
        object_count = int(len(objects))
        object_area = float(objects.area_km2.sum()) if object_count else 0.0
        largest_area = float(objects.area_km2.max()) if object_count else np.nan
        mean_area = float(objects.area_km2.mean()) if object_count else np.nan
        median_area = float(objects.area_km2.median()) if object_count else np.nan
        largest_object_fraction = largest_area / object_area if object_area > 0.0 else np.nan
        convective_area_fraction = object_area / domain_area if domain_area > 0.0 else np.nan
        lorg_km, lorg_normalised = lorg_from_objects(objects, domain_area, r_edges, rng, reference_cache)

        rows.append(
            dict(
                source=source_name,
                source_label=source_label(source_name),
                Time=mask.Time.values[it],
                time_days=float(time_days[it]),
                lorg_km=lorg_km,
                lorg_normalised=lorg_normalised,
                object_count=object_count,
                mean_object_area_km2=mean_area,
                median_object_area_km2=median_area,
                largest_object_area_km2=largest_area,
                largest_object_fraction=largest_object_fraction,
                convective_area_fraction=convective_area_fraction,
                tropical_mean_precip=float(precip_mean.isel(Time=it).values),
            )
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Object-track persistence
# -----------------------------------------------------------------------------


def filter_track_objects(labels, objects):
    if len(objects) == 0 or TRACK_MIN_AREA_KM2 <= 0.0:
        return labels, objects
    keep = objects[objects.area_km2 >= TRACK_MIN_AREA_KM2].copy()
    if len(keep) == len(objects):
        return labels, objects
    if len(keep) == 0:
        return np.zeros_like(labels), keep

    keep_labels = keep.label.astype(int).values
    filtered = labels.copy()
    filtered[~np.isin(filtered, keep_labels)] = 0
    return filtered, keep.reset_index(drop=True)


def epoch_seconds_from_time_values(time_values):
    stamps = pd.to_datetime(np.asarray(time_values))
    return ((stamps - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).astype(float).values


def pyflextrkr_timestamp_strings(epoch_seconds):
    stamp = pd.to_datetime(float(epoch_seconds), unit="s")
    return stamp.strftime("%Y%m%d"), stamp.strftime("%H%M%S")


def pyflextrkr_track_filename(epoch_seconds):
    stamp = pd.to_datetime(float(epoch_seconds), unit="s")
    return f"track_{stamp.strftime('%Y%m%d')}_{stamp.strftime('%H%M')}.nc"


def pyflextrkr_track_filename_with_seconds(epoch_seconds):
    stamp = pd.to_datetime(float(epoch_seconds), unit="s")
    return f"track_{stamp.strftime('%Y%m%d')}_{stamp.strftime('%H%M%S')}.nc"


def pyflextrkr_work_path(source_name, sim_name):
    safe_sim = sim_name.replace(" ", "_").replace("/", "_")
    source_cache_name = source_name
    if source_name in CLOUD_OBJECT_KEYS:
        source_cache_name = f"{source_name}_{CLASSIFICATION_CACHE_VERSION}"
    safe_source = source_cache_name.replace(" ", "_").replace("/", "_")
    return os.path.join(PYFLEXTRKR_WORK_DIR, safe_source, safe_sim)


def pyflextrkr_tracknumbers_filename(basetimes):
    startdate = pd.to_datetime(float(basetimes[0]), unit="s").strftime("%Y%m%d.%H%M")
    enddate = pd.to_datetime(float(basetimes[-1]), unit="s").strftime("%Y%m%d.%H%M")
    return f"tracknumbers_{startdate}_{enddate}.nc"


def pyflextrkr_tracknumbers_path(work_path, basetimes):
    return os.path.join(work_path, "stats", pyflextrkr_tracknumbers_filename(basetimes))


def expected_pyflextrkr_cloudid_files(tracking_path, basetimes):
    return [
        os.path.join(tracking_path, f"cloudid_{datestr}_{timestr}.nc")
        for datestr, timestr in (pyflextrkr_timestamp_strings(basetime) for basetime in basetimes)
    ]


def expected_pyflextrkr_track_files(tracking_path, basetimes):
    return [
        os.path.join(tracking_path, pyflextrkr_track_filename_with_seconds(basetimes[it + 1]))
        for it in range(len(basetimes) - 1)
    ]


def expected_pyflextrkr_files_available(tracking_path, basetimes):
    expected = expected_pyflextrkr_cloudid_files(tracking_path, basetimes)
    expected.extend(expected_pyflextrkr_track_files(tracking_path, basetimes))
    return bool(expected) and all(os.path.exists(path) for path in expected)


def clear_generated_pyflextrkr_files(tracking_path):
    if not os.path.exists(tracking_path):
        return
    for name in os.listdir(tracking_path):
        if name.endswith(".nc") and (name.startswith("cloudid_") or name.startswith("track_")):
            os.remove(os.path.join(tracking_path, name))


def write_pyflextrkr_cloudid_files(labels_by_time, mask, work_path):
    tracking_path = os.path.join(work_path, "tracking")
    os.makedirs(tracking_path, exist_ok=True)
    clear_generated_pyflextrkr_files(tracking_path)

    lat = np.asarray(mask.latitude.values, dtype=np.float32)
    lon = np.asarray(mask.longitude.values, dtype=np.float32)
    lon2d, lat2d = np.meshgrid(lon, lat)
    basetimes = epoch_seconds_from_time_values(mask.Time.values)
    files = []

    for it, labels in enumerate(labels_by_time):
        labels = np.asarray(labels, dtype=np.int32)
        nfeatures = int(np.nanmax(labels)) if labels.size else 0
        npix_feature = np.bincount(labels.ravel(), minlength=nfeatures + 1)[1:].astype(np.int32)
        datestr, timestr = pyflextrkr_timestamp_strings(basetimes[it])
        outfile = os.path.join(tracking_path, f"cloudid_{datestr}_{timestr}.nc")

        ds = xr.Dataset(
            data_vars=dict(
                base_time=(["time"], np.array([basetimes[it]], dtype=float)),
                longitude=(["lat", "lon"], lon2d),
                latitude=(["lat", "lon"], lat2d),
                feature_number=(["time", "lat", "lon"], labels[None, :, :]),
                nfeatures=(["time"], np.array([nfeatures], dtype=np.int32)),
                npix_feature=(["features"], npix_feature),
            ),
            coords=dict(
                time=(["time"], np.array([basetimes[it]], dtype=float)),
                lat=(["lat"], lat),
                lon=(["lon"], lon),
                features=(["features"], np.arange(1, nfeatures + 1, dtype=np.int32)),
            ),
            attrs=dict(Title="PyFLEXTRKR cloudid file generated from organisation diagnostics"),
        )
        if os.path.exists(outfile):
            os.remove(outfile)
        ds.to_netcdf(outfile, mode="w", format="NETCDF4")
        ds.close()
        files.append(outfile)

    return files, basetimes, tracking_path


def attach_pyflextrkr_track_ids(tracknumbers_file, objects_df):
    ds = xr.open_dataset(tracknumbers_file, mask_and_scale=False, decode_times=False, concat_characters=True)
    track_numbers = np.asarray(ds["track_numbers"].squeeze().values)
    track_status = np.asarray(ds["track_status"].squeeze().values)
    track_merge = np.asarray(ds["track_mergenumbers"].squeeze().values)
    track_split = np.asarray(ds["track_splitnumbers"].squeeze().values)
    track_reset = np.asarray(ds["track_reset"].squeeze().values)
    ds.close()

    objects_df = objects_df.copy()
    track_ids = []
    status_vals = []
    merge_vals = []
    split_vals = []
    reset_vals = []
    for record in objects_df[["time_index", "label"]].itertuples(index=False):
        it = int(record.time_index)
        label_index = int(record.label) - 1
        if it < track_numbers.shape[0] and 0 <= label_index < track_numbers.shape[1]:
            track_id = int(track_numbers[it, label_index])
            status = int(track_status[it, label_index])
            merge = int(track_merge[it, label_index])
            split = int(track_split[it, label_index])
            reset = int(track_reset[it, label_index])
        else:
            track_id = PYFLEXTRKR_FILLVAL
            status = PYFLEXTRKR_FILLVAL
            merge = PYFLEXTRKR_FILLVAL
            split = PYFLEXTRKR_FILLVAL
            reset = PYFLEXTRKR_FILLVAL
        track_ids.append(track_id if track_id > 0 else np.nan)
        status_vals.append(status if status != PYFLEXTRKR_FILLVAL else np.nan)
        merge_vals.append(merge if merge > 0 else np.nan)
        split_vals.append(split if split > 0 else np.nan)
        reset_vals.append(reset if reset != PYFLEXTRKR_FILLVAL else np.nan)

    objects_df["track_id"] = track_ids
    objects_df["pyflextrkr_track_status"] = status_vals
    objects_df["pyflextrkr_track_reset"] = reset_vals
    objects_df["start_split_track_id"] = split_vals
    objects_df["end_merge_track_id"] = merge_vals
    objects_df = objects_df[np.isfinite(objects_df.track_id)].copy()
    objects_df["track_id"] = objects_df.track_id.astype(int)
    objects_df["pyflextrkr_object_label"] = objects_df["label"].astype(int)
    objects_df["tracking_method"] = "pyflextrkr"
    return objects_df


def assign_pyflextrkr_track_ids(labels_by_time, objects_df, mask, source_name, sim_name):
    if len(labels_by_time) < 2:
        raise ValueError("at least two scenes are required for PyFLEXTRKR tracking")

    work_path = pyflextrkr_work_path(source_name, sim_name)
    stats_path = os.path.join(work_path, "stats")
    os.makedirs(stats_path, exist_ok=True)
    tracking_path = os.path.join(work_path, "tracking")
    basetimes = epoch_seconds_from_time_values(mask.Time.values)
    max_label = max(int(np.nanmax(labels)) if np.asarray(labels).size else 0 for labels in labels_by_time)
    config = dict(
        tracking_outpath=tracking_path + os.sep,
        stats_outpath=stats_path + os.sep,
        cloudid_filebase="cloudid_",
        singletrack_filebase="track_",
        tracknumbers_filebase="tracknumbers_",
        feature_varname="feature_number",
        nfeature_varname="nfeatures",
        featuresize_varname="npix_feature",
        timegap=1.5 * median_time_step_days(mask) * 24.0,
        nmaxlinks=PYFLEXTRKR_NMAXLINKS,
        othresh=TRACK_MIN_OVERLAP_FRACTION,
        maxnclouds=max(PYFLEXTRKR_NMAXLINKS, max_label + 2),
        start_basetime=float(basetimes[0]),
        end_basetime=float(basetimes[-1]),
        startdate=pd.to_datetime(float(basetimes[0]), unit="s").strftime("%Y%m%d.%H%M"),
        enddate=pd.to_datetime(float(basetimes[-1]), unit="s").strftime("%Y%m%d.%H%M"),
        fillval=PYFLEXTRKR_FILLVAL,
    )

    tracknumbers_file = pyflextrkr_tracknumbers_path(work_path, basetimes)
    if FORCE_RECOMPUTE_PYFLEXTRKR_CACHE and os.path.exists(tracknumbers_file):
        os.remove(tracknumbers_file)

    if not FORCE_RECOMPUTE_PYFLEXTRKR_CACHE and os.path.exists(tracknumbers_file):
        print(f"Using cached PyFLEXTRKR track numbers for {sim_name} {source_name}: {tracknumbers_file}")
        return attach_pyflextrkr_track_ids(tracknumbers_file, objects_df)

    if (
        not FORCE_RECOMPUTE_PYFLEXTRKR_CACHE
        and expected_pyflextrkr_files_available(tracking_path, basetimes)
    ):
        print(f"Using cached PyFLEXTRKR cloudid/track files for {sim_name} {source_name}: {tracking_path}")
        tracknumbers_file = pyflextrkr_gettracknumbers(config)
        return attach_pyflextrkr_track_ids(tracknumbers_file, objects_df)

    files, basetimes, tracking_path = write_pyflextrkr_cloudid_files(labels_by_time, mask, work_path)
    old_cwd = os.getcwd()
    try:
        os.chdir(tracking_path)
        for it in range(len(files) - 1):
            pyflextrkr_trackclouds(
                (os.path.basename(files[it]), os.path.basename(files[it + 1])),
                (basetimes[it], basetimes[it + 1]),
                config,
            )
            short_name = os.path.join(tracking_path, pyflextrkr_track_filename(basetimes[it + 1]))
            seconds_name = os.path.join(tracking_path, pyflextrkr_track_filename_with_seconds(basetimes[it + 1]))
            if os.path.exists(seconds_name):
                os.remove(seconds_name)
            if os.path.exists(short_name):
                os.replace(short_name, seconds_name)
    finally:
        os.chdir(old_cwd)

    tracknumbers_file = pyflextrkr_gettracknumbers(config)
    return attach_pyflextrkr_track_ids(tracknumbers_file, objects_df)


def build_track_tables(mask, fields, source_name, sim_name):
    """
    Link object masks through time using PyFLEXTRKR-method overlap tracking.

    PyFLEXTRKR links adjacent objects when either forward or backward overlap
    fraction exceeds the threshold. During merges/splits, the largest branch is
    tracked continuously and smaller branches are recorded as merge/split links.
    """
    mask = subset_lat(mask.fillna(False), lat_bounds=TROPICAL_LAT_BOUNDS)
    if "Time" not in mask.dims:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    precip = fields["precip"].sel(Time=mask.Time)
    precip_mean = fields["mean_precip"].sel(Time=mask.Time)
    mask, precip, precip_mean = xr.align(mask, precip, precip_mean, join="inner")

    area2d = grid_cell_area_km2(mask.latitude.values, mask.longitude.values)
    domain_area = float(area2d.sum().values)
    time_days = time_days_from_coord(mask.Time)
    dt_hours = 24.0 * median_time_step_days(mask)

    labels_by_time = []
    rows = []
    node_index = 0

    for it in range(mask.sizes["Time"]):
        labels, objects = label_objects_with_properties(
            mask.isel(Time=it).load(),
            area2d,
            precip2d=precip.isel(Time=it).load(),
        )
        labels, objects = filter_track_objects(labels, objects)
        labels_by_time.append(labels)
        scene_area = float(objects.area_km2.sum()) if len(objects) else 0.0
        scene_precip_area_rate = float(objects.precip_area_rate.sum()) if len(objects) else 0.0
        scene_area_fraction = scene_area / domain_area if domain_area > 0.0 else np.nan

        for record in objects.to_dict("records"):
            node_id = node_index
            node_index += 1
            rows.append(
                dict(
                    node_id=node_id,
                    source=source_name,
                    source_label=source_label(source_name),
                    simulation=sim_name,
                    Time=mask.Time.values[it],
                    time_index=it,
                    time_days=float(time_days[it]),
                    dt_hours=dt_hours,
                    domain_area_km2=domain_area,
                    scene_convective_area_fraction=scene_area_fraction,
                    scene_precip_area_rate=scene_precip_area_rate,
                    tropical_mean_precip=float(precip_mean.isel(Time=it).values),
                    **record,
                )
            )

    objects_df = pd.DataFrame(rows)
    if len(objects_df) == 0:
        return objects_df, pd.DataFrame(), pd.DataFrame()

    objects_df = assign_pyflextrkr_track_ids(labels_by_time, objects_df, mask, source_name, sim_name)
    if len(objects_df) == 0:
        return objects_df, pd.DataFrame(), pd.DataFrame()

    track_time = (
        objects_df
        .groupby(["source", "source_label", "simulation", "track_id", "time_index"], as_index=False)
        .agg(
            Time=("Time", "first"),
            time_days=("time_days", "first"),
            dt_hours=("dt_hours", "first"),
            area_km2=("area_km2", "sum"),
            precip_area_rate=("precip_area_rate", "sum"),
            object_count=("label", "count"),
            tropical_mean_precip=("tropical_mean_precip", "first"),
            scene_convective_area_fraction=("scene_convective_area_fraction", "first"),
        )
    )

    tracks = []
    for (source, label, simulation, track_id), group in track_time.groupby(
        ["source", "source_label", "simulation", "track_id"]
    ):
        group = group.sort_values("time_index")
        lifetime_steps = int(group.time_index.nunique())
        if lifetime_steps < TRACK_MIN_LIFETIME_STEPS_FOR_SUMMARY:
            continue
        lifetime_hours = float(lifetime_steps * dt_hours)
        track_objects = objects_df[objects_df.track_id == track_id]
        start_objects = track_objects[track_objects.time_index == group.time_index.min()]
        end_objects = track_objects[track_objects.time_index == group.time_index.max()]
        split_from = start_objects.start_split_track_id.dropna().values
        merge_into = end_objects.end_merge_track_id.dropna().values
        tracks.append(
            dict(
                source=source,
                source_label=label,
                simulation=simulation,
                track_id=int(track_id),
                start_time=group.Time.iloc[0],
                end_time=group.Time.iloc[-1],
                start_day=float(group.time_days.iloc[0]),
                end_day=float(group.time_days.iloc[-1]),
                lifetime_steps=lifetime_steps,
                lifetime_hours=lifetime_hours,
                max_area_km2=float(group.area_km2.max()),
                mean_area_km2=float(group.area_km2.mean()),
                area_time_km2_h=float(np.nansum(group.area_km2.values * group.dt_hours.values)),
                precip_area_rate_time=float(np.nansum(group.precip_area_rate.values * group.dt_hours.values)),
                mean_tropical_precip=float(group.tropical_mean_precip.mean()),
                mean_convective_area_fraction=float(group.scene_convective_area_fraction.mean()),
                max_object_count=int(group.object_count.max()),
                starts_as_split=bool(split_from.size > 0),
                ends_as_merge=bool(merge_into.size > 0),
                split_from_track_id=float(split_from[0]) if split_from.size > 0 else np.nan,
                merge_into_track_id=float(merge_into[-1]) if merge_into.size > 0 else np.nan,
            )
        )
    tracks_df = pd.DataFrame(tracks)
    if len(tracks_df) == 0:
        return objects_df, tracks_df, pd.DataFrame()

    track_time = track_time.merge(
        tracks_df[["source", "simulation", "track_id", "lifetime_hours"]],
        on=["source", "simulation", "track_id"],
        how="left",
    )

    rows = []
    denom = (
        track_time
        .groupby(["source", "source_label", "simulation", "time_index"], as_index=False)
        .agg(
            Time=("Time", "first"),
            time_days=("time_days", "first"),
            total_area_km2=("area_km2", "sum"),
            total_precip_area_rate=("precip_area_rate", "sum"),
            domain_area_km2=("area_km2", lambda _x: domain_area),
            tropical_mean_precip=("tropical_mean_precip", "first"),
            convective_area_fraction=("scene_convective_area_fraction", "first"),
        )
    )
    for threshold_hours in LONG_LIVED_THRESHOLDS_HOURS:
        long_track_time = track_time[track_time.lifetime_hours >= threshold_hours]
        numer = (
            long_track_time
            .groupby(["source", "simulation", "time_index"], as_index=False)
            .agg(
                long_lived_area_km2=("area_km2", "sum"),
                long_lived_precip_area_rate=("precip_area_rate", "sum"),
            )
        )
        merged = denom.merge(numer, on=["source", "simulation", "time_index"], how="left")
        merged["long_lived_area_km2"] = merged["long_lived_area_km2"].fillna(0.0)
        merged["long_lived_precip_area_rate"] = merged["long_lived_precip_area_rate"].fillna(0.0)
        for record in merged.to_dict("records"):
            total_area = record["total_area_km2"]
            total_precip = record["total_precip_area_rate"]
            rows.append(
                dict(
                    source=record["source"],
                    source_label=record["source_label"],
                    simulation=record["simulation"],
                    Time=record["Time"],
                    time_days=record["time_days"],
                    threshold_hours=float(threshold_hours),
                    tropical_mean_precip=record["tropical_mean_precip"],
                    convective_area_fraction=record["convective_area_fraction"],
                    long_lived_area_fraction_of_domain=record["long_lived_area_km2"] / domain_area,
                    long_lived_area_fraction_of_objects=record["long_lived_area_km2"] / total_area if total_area > 0 else np.nan,
                    long_lived_precip_fraction_of_objects=(
                        record["long_lived_precip_area_rate"] / total_precip if total_precip > 0 else np.nan
                    ),
                )
            )

    return objects_df, tracks_df, pd.DataFrame(rows)


def build_track_persistence_metrics(masks_by_run, fields_by_run):
    object_rows = []
    track_rows = []
    contribution_rows = []
    for i, masks in enumerate(masks_by_run):
        sim_name = sim_names[i]
        for source_name in [name for name in SOURCE_ORDER if name in masks]:
            print(f"Tracking object persistence for {sim_name}: {source_name}")
            objects, tracks, contribution = build_track_tables(
                masks[source_name],
                fields_by_run[i],
                source_name,
                sim_name,
            )
            if len(objects):
                object_rows.append(objects)
            if len(tracks):
                track_rows.append(tracks)
            if len(contribution):
                contribution_rows.append(contribution)

    objects_df = pd.concat(object_rows, ignore_index=True) if object_rows else pd.DataFrame()
    tracks_df = pd.concat(track_rows, ignore_index=True) if track_rows else pd.DataFrame()
    contribution_df = pd.concat(contribution_rows, ignore_index=True) if contribution_rows else pd.DataFrame()
    return objects_df, tracks_df, contribution_df


# -----------------------------------------------------------------------------
# Cloud-type masks
# -----------------------------------------------------------------------------


def has_primary_cloud_masks(masks):
    return all(key in masks for key in CLOUD_OBJECT_KEYS_PRIMARY)


def classification_thresholds():
    return HighCloudThresholds(
        cf_thresh=CF_THRESH,
        z_high_km=Z_HIGH,
        low_cloud_min_km=Z_LOW_MIN,
        low_cloud_max_km=Z_LOW_MAX,
        mid_cloud_min_km=Z_MID_MIN,
        mid_cloud_max_km=Z_MID_MAX,
        upper_cloud_min_km=Z_UPPER_MIN,
        max_clear_gap_km=MAX_CLEAR_GAP,
        surface_precip_deep_mm_day=P_DEEP,
        surface_precip_weak_mm_day=P_WEAK,
    )


def cached_classification_is_current(cached):
    return shared_cached_classification_is_current(cached, classification_thresholds())


def cached_classification_path(sim_name):
    filename = f"{sim_name.replace(' ', '_')}_classified_high_cloud.nc"
    for directory in CLOUD_CLASSIFICATION_CACHE_DIRS:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return None


def cloud_masks_from_cache(sim_name, precip_ref):
    path = cached_classification_path(sim_name)
    if path is None:
        checked = [
            os.path.join(directory, f"{sim_name.replace(' ', '_')}_classified_high_cloud.nc")
            for directory in CLOUD_CLASSIFICATION_CACHE_DIRS
        ]
        raise FileNotFoundError(
            f"{sim_name}: required shared high-cloud classification cache is missing. "
            f"Run refined_high_cloud_classification_diagnostics.py first. Checked: {checked}"
        )

    print(f"Using cached high-cloud classification for {sim_name}: {path}")
    cached = xr.open_dataset(path, chunks={})
    if not cached_classification_is_current(cached):
        cached.close()
        raise RuntimeError(f"{sim_name}: stale high-cloud classification cache: {path}")
    missing = [key for key in CLOUD_OBJECT_KEYS if key not in cached.data_vars]
    if missing:
        cached.close()
        raise RuntimeError(
            f"{sim_name}: high-cloud classification cache lacks required shared variables {missing}: {path}"
        )
    masks = {}
    for key in CLOUD_OBJECT_KEYS:
        da = cached[key]
        da = select_common_time(da, precip_ref)
        masks[key] = da.fillna(False).astype(bool)
    return masks


def cloud_type_masks(ds, fields, sim_name):
    if not ENABLE_CLOUD_OBJECTS:
        return {}

    masks = cloud_masks_from_cache(sim_name, fields["precip"])
    if not has_primary_cloud_masks(masks):
        raise RuntimeError(f"{sim_name}: shared high-cloud classification cache lacks primary cloud-object masks")
    return masks


# -----------------------------------------------------------------------------
# Threshold masks and summaries
# -----------------------------------------------------------------------------


def threshold_from_pooled_runs(fields_by_run, field_key, percentile, transform=None):
    vals_all = []
    for fields in fields_by_run:
        if field_key not in fields:
            continue
        da = fields[field_key]
        if transform is not None:
            da = transform(da)
        vals = sampled_values(da)
        if vals.size:
            vals_all.append(vals)
    if not vals_all:
        return np.nan
    return float(np.nanpercentile(np.concatenate(vals_all), percentile))


def build_threshold_table(fields_by_run):
    rows = []
    for spec in PRECIP_OBJECT_SPECS:
        if "threshold" in spec:
            threshold = float(spec["threshold"])
            row = dict(spec)
            row.update(
                source=spec["name"],
                field="precip",
                threshold=threshold,
                threshold_type="absolute",
                percentile=np.nan,
            )
            rows.append(row)
        else:
            threshold = threshold_from_pooled_runs(fields_by_run, "precip", spec["percentile"])
            row = dict(spec)
            row.update(
                source=spec["name"],
                field="precip",
                threshold=threshold,
                threshold_type="pooled_percentile",
            )
            rows.append(row)

    if ENABLE_OLR_OBJECTS and all("olr" in fields for fields in fields_by_run):
        for spec in OLR_OBJECT_SPECS:
            row = dict(spec)
            if "threshold" in spec:
                threshold = float(spec["threshold"])
                row.update(
                    source=spec["name"],
                    field=spec.get("field", "olr"),
                    threshold=threshold,
                    threshold_type="absolute",
                    percentile=np.nan,
                )
            else:
                threshold = threshold_from_pooled_runs(
                    fields_by_run,
                    "olr",
                    spec["percentile"],
                    transform=lambda da: da - da.mean(dim="Time", skipna=True),
                )
                row.update(
                    source=spec["name"],
                    field="olr_anomaly",
                    threshold=threshold,
                    threshold_type="pooled_percentile",
                )
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_object_masks(fields, threshold_table):
    masks = {}
    for row in threshold_table.to_dict("records"):
        source = row["source"]
        field = row["field"]
        threshold = row["threshold"]
        sense = row["sense"]
        if not np.isfinite(threshold):
            continue
        if field == "precip":
            da = fields["precip"]
        elif field == "olr" and "olr" in fields:
            da = fields["olr"]
        elif field == "olr_anomaly" and "olr" in fields:
            da = fields["olr"] - fields["olr"].mean(dim="Time", skipna=True)
        else:
            continue

        if sense == "above":
            mask = da >= threshold
        elif sense == "below":
            mask = da <= threshold
        else:
            raise ValueError(f"Unknown threshold sense {sense!r}")
        mask.name = source
        masks[source] = mask.fillna(False)
    return masks


def build_fields(ds):
    precip = get_precip_rate(ds)
    fields = {
        "precip": precip,
        "mean_precip": weighted_tropical_mean(precip, lat_bounds=TROPICAL_LAT_BOUNDS),
    }
    olr = time_aligned_field(ds, "olr", precip, required=False)
    if olr is not None:
        fields["olr"] = olr
    pw = time_aligned_field(ds, "pw", precip, required=False)
    if pw is not None:
        fields["pw"] = pw
    return fields


def count_rows_to_frame(rows, stem):
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    print(f"\nSample counts for {stem}")
    print(df.to_string(index=False, max_rows=240))
    if save:
        df.to_csv(os.path.join(save_loc, f"{stem}_bin_counts.csv"), index=False)
    return df


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def source_axis_grid(sources, ncols=2, height_per_row=2.9):
    nsrc = len(sources)
    ncols = min(ncols, max(1, nsrc))
    nrows = int(np.ceil(nsrc / ncols))
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, height_per_row * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis in ax.ravel()[nsrc:]:
        axis.set_visible(False)
    return fig, ax


def plot_lorg_timeseries(metrics_df):
    sources = ordered_sources(metrics_df)
    fig, ax = source_axis_grid(sources)
    for si, source in enumerate(sources):
        axis = ax.ravel()[si]
        ncols = ax.shape[1]
        sub_source = metrics_df[metrics_df.source == source]
        for i, sim_name in enumerate(sim_names):
            sub = sub_source[sub_source.simulation == sim_name]
            if len(sub) == 0:
                continue
            axis.plot(sub.time_days, sub.lorg_km, color=cols[i], lw=1.2, label=sim_name)
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.set_title(source_label(source))
        if si + ncols >= len(sources):
            axis.set_xlabel("Time since analysis start (days)")
        if si % ncols == 0:
            axis.set_ylabel("Lorg (km)")
        axis.grid(alpha=0.2)
    ax.ravel()[0].legend(loc="best")
    fig.suptitle("Lorg: instantaneous spatial aggregation from all object-pair distances")
    save_figure(1, fig, "lorg_timeseries")


def plot_lorg_distribution(metrics_df):
    sources = ordered_sources(metrics_df)
    nrows = len(sources)
    fig, ax = plt.subplots(
        nrows,
        2,
        figsize=(10.5, max(3.0, 2.8 * nrows)),
        constrained_layout=True,
        squeeze=False,
    )

    for si, source in enumerate(sources):
        sub_source = metrics_df[metrics_df.source == source]
        vals_all = sub_source.lorg_km.values
        vals_all = vals_all[np.isfinite(vals_all)]
        if vals_all.size >= 2:
            lo = float(np.nanpercentile(vals_all, 2))
            hi = float(np.nanpercentile(vals_all, 98))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                bins = np.linspace(lo, hi, 24)
            else:
                span = max(1.0, abs(lo) * 0.1)
                bins = np.linspace(lo - span, lo + span, 24)
        else:
            bins = np.linspace(-1.0, 1.0, 12)

        box_values = []
        box_labels = []
        for i, sim_name in enumerate(sim_names):
            vals = sub_source.loc[sub_source.simulation == sim_name, "lorg_km"].values
            vals = vals[np.isfinite(vals)]
            if vals.size:
                ax[si, 0].hist(vals, bins=bins, density=True, histtype="step", color=cols[i], lw=1.4, label=sim_name)
            box_values.append(vals if vals.size else np.array([np.nan]))
            box_labels.append(sim_name)

        ax[si, 1].boxplot(
            box_values,
            labels=box_labels,
            showfliers=False,
            medianprops={"color": "k"},
        )
        for tick, color in zip(ax[si, 1].get_xticklabels(), cols):
            tick.set_color(color)

        ax[si, 0].axvline(0.0, color="0.6", lw=0.8)
        ax[si, 1].axhline(0.0, color="0.6", lw=0.8)
        ax[si, 0].set_ylabel(f"{source_label(source)}\nPDF")
        if si == nrows - 1:
            ax[si, 0].set_xlabel("Lorg (km)")
        ax[si, 1].set_ylabel("Lorg (km)")
        ax[si, 0].grid(alpha=0.2)
        ax[si, 1].grid(alpha=0.2, axis="y")

    ax[0, 0].legend(loc="best")
    fig.suptitle("Distribution of Lorg across timesteps")
    save_figure(2, fig, "lorg_pdf_boxplot")


def plot_object_metric_distributions(metrics_df):
    sources = ordered_sources(metrics_df)
    metric_specs = [
        ("object_count", "Object count"),
        ("mean_object_area_km2", "Mean object area (km$^2$)"),
        ("largest_object_fraction", "Largest-object fraction"),
        ("convective_area_fraction", "Convective area fraction"),
    ]
    fig, ax = plt.subplots(
        len(sources),
        len(metric_specs),
        figsize=(4.1 * len(metric_specs), max(3.0, 2.55 * len(sources))),
        constrained_layout=True,
        squeeze=False,
    )
    for si, source in enumerate(sources):
        sub_source = metrics_df[metrics_df.source == source]
        for mj, (metric, label) in enumerate(metric_specs):
            values = []
            for sim_name in sim_names:
                vals = sub_source.loc[sub_source.simulation == sim_name, metric].values
                vals = vals[np.isfinite(vals)]
                values.append(vals if vals.size else np.array([np.nan]))
            ax[si, mj].boxplot(values, labels=sim_names, showfliers=False, medianprops={"color": "k"})
            for tick, color in zip(ax[si, mj].get_xticklabels(), cols):
                tick.set_color(color)
                tick.set_rotation(25)
                tick.set_ha("right")
            if si == 0:
                ax[si, mj].set_title(label)
            if mj == 0:
                ax[si, mj].set_ylabel(source_label(source))
            ax[si, mj].grid(alpha=0.2, axis="y")
    fig.suptitle("Object size/count metrics: size and amount changes")
    save_figure(3, fig, "object_metric_boxplots")


def conditional_summary(metrics_df, source, condition_key):
    sub_source = metrics_df[metrics_df.source == source]
    cond_vals = sub_source[condition_key].values
    cond_vals = cond_vals[np.isfinite(cond_vals)]
    if cond_vals.size < CONDITION_NBINS:
        return pd.DataFrame()
    edges = np.unique(np.nanpercentile(cond_vals, np.linspace(0.0, 100.0, CONDITION_NBINS + 1)))
    if edges.size < 3:
        return pd.DataFrame()

    rows = []
    for bi in range(edges.size - 1):
        lo, hi = edges[bi], edges[bi + 1]
        for sim_name in sim_names:
            sub = sub_source[sub_source.simulation == sim_name]
            if bi == edges.size - 2:
                in_bin = (sub[condition_key] >= lo) & (sub[condition_key] <= hi)
            else:
                in_bin = (sub[condition_key] >= lo) & (sub[condition_key] < hi)
            vals = sub.loc[in_bin, "lorg_km"].values
            vals = vals[np.isfinite(vals)]
            rows.append(
                dict(
                    source=source,
                    source_label=source_label(source),
                    simulation=sim_name,
                    condition_key=condition_key,
                    bin_index=bi,
                    bin_low=lo,
                    bin_high=hi,
                    bin_center=0.5 * (lo + hi),
                    lorg_mean=np.nanmean(vals) if vals.size >= MIN_CONDITION_COUNT else np.nan,
                    lorg_p25=np.nanpercentile(vals, 25) if vals.size >= MIN_CONDITION_COUNT else np.nan,
                    lorg_p75=np.nanpercentile(vals, 75) if vals.size >= MIN_CONDITION_COUNT else np.nan,
                    sample_count=int(vals.size),
                )
            )
    return pd.DataFrame(rows)


def plot_conditional_lorg(metrics_df, condition_key, number, stem, xlabel):
    sources = ordered_sources(metrics_df)
    summaries = [conditional_summary(metrics_df, source, condition_key) for source in sources]
    summary = pd.concat([df for df in summaries if len(df) > 0], ignore_index=True) if summaries else pd.DataFrame()
    if len(summary) == 0:
        print(f"Skipping {stem}: no finite conditional samples")
        return

    if save:
        summary.to_csv(os.path.join(save_loc, f"{stem}.csv"), index=False)

    fig, ax = source_axis_grid(sources)
    for si, source in enumerate(sources):
        axis = ax.ravel()[si]
        ncols = ax.shape[1]
        sub_source = summary[summary.source == source]
        if len(sub_source) == 0:
            axis.set_visible(False)
            continue
        for i, sim_name in enumerate(sim_names):
            sub = sub_source[sub_source.simulation == sim_name]
            axis.plot(sub.bin_center, sub.lorg_mean, color=cols[i], marker="o", lw=1.5, label=sim_name)
            axis.fill_between(sub.bin_center, sub.lorg_p25, sub.lorg_p75, color=cols[i], alpha=0.12, linewidth=0)
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.set_title(source_label(source))
        if si + ncols >= len(sources):
            axis.set_xlabel(xlabel)
        if si % ncols == 0:
            axis.set_ylabel("Lorg (km)")
        axis.grid(alpha=0.2)
    ax.ravel()[0].legend(loc="best")
    fig.suptitle("Conditional Lorg: organisation compared at fixed convective amount")
    save_figure(number, fig, stem)
    count_rows_to_frame(summary.rename(columns={"sample_count": "count"}).to_dict("records"), stem)


def plot_track_lifetime_distribution(tracks_df):
    if len(tracks_df) == 0:
        print("Skipping track lifetime plot: no tracks")
        return
    sources = ordered_sources(tracks_df)
    fig, ax = plt.subplots(
        len(sources),
        2,
        figsize=(10.6, max(3.2, 2.9 * len(sources))),
        constrained_layout=True,
        squeeze=False,
    )
    for si, source in enumerate(sources):
        sub_source = tracks_df[tracks_df.source == source]
        box_values = []
        box_labels = []
        for i, sim_name in enumerate(sim_names):
            vals = sub_source.loc[sub_source.simulation == sim_name, "lifetime_hours"].values
            vals = vals[np.isfinite(vals)]
            if vals.size:
                x = np.sort(np.unique(vals))
                survival = np.array([np.mean(vals >= xx) for xx in x])
                ax[si, 0].step(x, survival, where="post", color=cols[i], lw=1.5, label=sim_name)
            box_values.append(vals if vals.size else np.array([np.nan]))
            box_labels.append(sim_name)

        ax[si, 1].boxplot(box_values, labels=box_labels, showfliers=False, medianprops={"color": "k"})
        for tick, color in zip(ax[si, 1].get_xticklabels(), cols):
            tick.set_color(color)
            tick.set_rotation(25)
            tick.set_ha("right")
        ax[si, 0].set_ylabel(f"{source_label(source)}\nSurvival fraction")
        ax[si, 0].set_xlabel("Track lifetime (h)")
        ax[si, 1].set_ylabel("Track lifetime (h)")
        ax[si, 0].grid(alpha=0.2)
        ax[si, 1].grid(alpha=0.2, axis="y")
    ax[0, 0].legend(loc="best")
    fig.suptitle("Persistence: lifetime of tracked organised convective patterns")
    save_figure(6, fig, "track_lifetime_survival_boxplot")


def plot_long_lived_contribution(contribution_df):
    if len(contribution_df) == 0:
        print("Skipping long-lived contribution plot: no contribution table")
        return
    sources = ordered_sources(contribution_df)
    fig, ax = source_axis_grid(sources, height_per_row=3.2)
    for si, source in enumerate(sources):
        axis = ax.ravel()[si]
        ncols = ax.shape[1]
        sub_source = contribution_df[contribution_df.source == source]
        for i, sim_name in enumerate(sim_names):
            sub = (
                sub_source[sub_source.simulation == sim_name]
                .groupby("threshold_hours", as_index=False)
                .agg(
                    area_fraction=("long_lived_area_fraction_of_objects", "mean"),
                    precip_fraction=("long_lived_precip_fraction_of_objects", "mean"),
                )
            )
            if len(sub) == 0:
                continue
            axis.plot(sub.threshold_hours, sub.area_fraction, color=cols[i], marker="o", lw=1.5, label=sim_name)
            axis.plot(sub.threshold_hours, sub.precip_fraction, color=cols[i], marker="s", lw=1.1, ls="--")
        axis.set_ylim(0, 1)
        axis.set_title(source_label(source))
        if si + ncols >= len(sources):
            axis.set_xlabel("Minimum track lifetime (h)")
        if si % ncols == 0:
            axis.set_ylabel("Fraction in long-lived tracks")
        axis.grid(alpha=0.2)
    ax.ravel()[0].legend(loc="best")
    fig.suptitle("Long-lived organised contribution: area solid, precipitation dashed")
    save_figure(7, fig, "long_lived_track_contribution")


def conditional_track_summary(tracks_df, source, condition_key):
    sub_source = tracks_df[tracks_df.source == source]
    values = sub_source[condition_key].values
    values = values[np.isfinite(values)]
    if values.size < TRACK_CONDITION_NBINS:
        return pd.DataFrame()
    edges = np.unique(np.nanpercentile(values, np.linspace(0.0, 100.0, TRACK_CONDITION_NBINS + 1)))
    if edges.size < 3:
        return pd.DataFrame()

    rows = []
    for bi in range(edges.size - 1):
        lo, hi = edges[bi], edges[bi + 1]
        for sim_name in sim_names:
            sub = sub_source[sub_source.simulation == sim_name]
            if bi == edges.size - 2:
                in_bin = (sub[condition_key] >= lo) & (sub[condition_key] <= hi)
            else:
                in_bin = (sub[condition_key] >= lo) & (sub[condition_key] < hi)
            vals = sub.loc[in_bin, "lifetime_hours"].values
            vals = vals[np.isfinite(vals)]
            rows.append(
                dict(
                    source=source,
                    source_label=source_label(source),
                    simulation=sim_name,
                    condition_key=condition_key,
                    bin_index=bi,
                    bin_low=lo,
                    bin_high=hi,
                    bin_center=0.5 * (lo + hi),
                    lifetime_mean=np.nanmean(vals) if vals.size >= MIN_TRACK_BIN_COUNT else np.nan,
                    lifetime_p25=np.nanpercentile(vals, 25) if vals.size >= MIN_TRACK_BIN_COUNT else np.nan,
                    lifetime_p75=np.nanpercentile(vals, 75) if vals.size >= MIN_TRACK_BIN_COUNT else np.nan,
                    sample_count=int(vals.size),
                )
            )
    return pd.DataFrame(rows)


def plot_conditional_track_lifetime(tracks_df, condition_key, number, stem, xlabel):
    if len(tracks_df) == 0 or condition_key not in tracks_df.columns:
        print(f"Skipping {stem}: no track lifetime samples")
        return
    sources = ordered_sources(tracks_df)
    summaries = [conditional_track_summary(tracks_df, source, condition_key) for source in sources]
    summary = pd.concat([df for df in summaries if len(df) > 0], ignore_index=True) if summaries else pd.DataFrame()
    if len(summary) == 0:
        print(f"Skipping {stem}: no finite conditional track samples")
        return
    if save:
        summary.to_csv(os.path.join(save_loc, f"{stem}.csv"), index=False)

    fig, ax = source_axis_grid(sources)
    for si, source in enumerate(sources):
        axis = ax.ravel()[si]
        ncols = ax.shape[1]
        sub_source = summary[summary.source == source]
        if len(sub_source) == 0:
            axis.set_visible(False)
            continue
        for i, sim_name in enumerate(sim_names):
            sub = sub_source[sub_source.simulation == sim_name]
            axis.plot(sub.bin_center, sub.lifetime_mean, color=cols[i], marker="o", lw=1.5, label=sim_name)
            axis.fill_between(sub.bin_center, sub.lifetime_p25, sub.lifetime_p75, color=cols[i], alpha=0.12, linewidth=0)
        axis.set_title(source_label(source))
        if si + ncols >= len(sources):
            axis.set_xlabel(xlabel)
        if si % ncols == 0:
            axis.set_ylabel("Track lifetime (h)")
        axis.grid(alpha=0.2)
    ax.ravel()[0].legend(loc="best")
    fig.suptitle("Conditional persistence: track lifetime at fixed convective amount")
    save_figure(number, fig, stem)
    count_rows_to_frame(summary.rename(columns={"sample_count": "count"}).to_dict("records"), stem)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def equatorial_latmean_anomaly(da, lat_bounds=PERSISTENCE_LAT_BOUNDS):
    x = subset_lat(finite_or_nan(da), lat_bounds=lat_bounds)
    if "latitude" in x.dims:
        x = x.weighted(np.cos(np.deg2rad(x.latitude))).mean(dim="latitude", skipna=True)
    if "longitude" not in x.dims:
        return x - x.mean(dim="Time", skipna=True)
    return x - x.mean(dim="Time", skipna=True)


def lag_pattern_autocorrelation(da, max_lag_steps):
    x = np.asarray(da.transpose("Time", ...).load().values, dtype=float)
    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    out = []
    lon_axis = x.ndim - 1 if "longitude" in da.dims else None
    for lag in lags:
        if lag < 0:
            x0 = x[-lag:]
            x1 = x[:lag]
        elif lag > 0:
            x0 = x[:-lag]
            x1 = x[lag:]
        else:
            x0 = x
            x1 = x

        correlations = []
        shifts = [0] if lon_axis is None else range(-PERSISTENCE_MAX_LON_SHIFT_CELLS, PERSISTENCE_MAX_LON_SHIFT_CELLS + 1)
        for shift in shifts:
            y1 = np.roll(x1, int(shift), axis=lon_axis) if lon_axis is not None else x1
            v0 = x0.ravel()
            v1 = y1.ravel()
            ok = np.isfinite(v0) & np.isfinite(v1)
            if np.count_nonzero(ok) >= 3:
                correlations.append(float(np.corrcoef(v0[ok], v1[ok])[0, 1]))
        out.append(np.nanmax(correlations) if correlations else np.nan)
    return lags, np.asarray(out, dtype=float)


def compute_persistence(fields_by_run):
    rows = []
    field_specs = [("precip", "Precipitation")]
    if all("olr" in fields for fields in fields_by_run):
        field_specs.append(("olr", "OLR"))

    for i, fields in enumerate(fields_by_run):
        for field_key, label in field_specs:
            da = equatorial_latmean_anomaly(fields[field_key], lat_bounds=PERSISTENCE_LAT_BOUNDS)
            dt_days = median_time_step_days(da)
            max_lag_steps = max(1, int(round(PERSISTENCE_MAX_LAG_DAYS / dt_days)))
            lags, corr = lag_pattern_autocorrelation(da, max_lag_steps)
            for lag, value in zip(lags, corr):
                rows.append(
                    dict(
                        simulation=sim_names[i],
                        field=field_key,
                        field_label=label,
                        lag_days=float(lag * dt_days),
                        autocorrelation=value,
                    )
                )
    return pd.DataFrame(rows)


def plot_shifted_pattern_autocorrelation(persistence_df):
    fields = list(persistence_df.field.unique())
    fig, ax = plt.subplots(
        1,
        len(fields),
        figsize=(5.2 * len(fields), 3.8),
        constrained_layout=True,
        squeeze=False,
    )
    for fj, field_key in enumerate(fields):
        axis = ax[0, fj]
        sub_field = persistence_df[persistence_df.field == field_key]
        for i, sim_name in enumerate(sim_names):
            sub = sub_field[sub_field.simulation == sim_name]
            axis.plot(sub.lag_days, sub.autocorrelation, color=cols[i], lw=1.6, label=sim_name)
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.axvline(0.0, color="0.6", lw=0.8)
        axis.set_title(sub_field.field_label.iloc[0])
        axis.set_xlabel("Lag (days)")
        if fj == 0:
            axis.set_ylabel("Pattern autocorrelation")
        axis.grid(alpha=0.2)
    ax[0, 0].legend(loc="best")
    fig.suptitle(
        f"Secondary persistence check: longitude-shifted equatorial anomaly correlation over {PERSISTENCE_LAT_BOUNDS[0]:g} to {PERSISTENCE_LAT_BOUNDS[1]:g} deg"
    )
    save_figure(10, fig, "equatorial_shifted_pattern_autocorrelation")


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def load_datasets():
    dss = []
    for i, d in enumerate(data_locs):
        ds = load_latlon_dataset(
            os.path.join(root_dir, d),
            latlon_pattern=latlon_file[i],
            time_format=time_format,
            ndays=ndays,
            end_date=analysis_end_date,
            chunks=latlon_chunks,
            add_cldfrac_file=False,
        )
        ds = standardize_latlon_dataset(ds, needed_vars)
        print(ds)
        dss.append(ds)
    return dss


def build_masks_by_run(dss, fields_by_run, threshold_table):
    masks_by_run = []
    for i, (ds, fields) in enumerate(zip(dss, fields_by_run)):
        sim_name = sim_names[i]
        masks = threshold_object_masks(fields, threshold_table)
        masks.update(cloud_type_masks(ds, fields, sim_name))
        masks_by_run.append(masks)
    return masks_by_run


def build_all_metrics(dss, fields_by_run, threshold_table, masks_by_run=None):
    rng = np.random.default_rng(REFERENCE_RANDOM_SEED)
    reference_cache = {}
    all_rows = []
    if masks_by_run is None:
        masks_by_run = build_masks_by_run(dss, fields_by_run, threshold_table)

    for i, fields in enumerate(fields_by_run):
        sim_name = sim_names[i]
        masks = masks_by_run[i]

        for source_name in [name for name in SOURCE_ORDER if name in masks]:
            print(f"Computing Lorg metrics for {sim_name}: {source_name}")
            metrics = compute_organisation_timeseries(
                masks[source_name],
                fields,
                source_name,
                rng,
                reference_cache,
            )
            if len(metrics) == 0:
                continue
            metrics["simulation"] = sim_name
            all_rows.append(metrics)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def main():
    os.makedirs(save_loc, exist_ok=True)

    dss = None
    fields_by_run = None
    masks_by_run = None

    def ensure_fields():
        nonlocal dss, fields_by_run
        if fields_by_run is None:
            dss = load_datasets()
            fields_by_run = [build_fields(ds) for ds in dss]
        return dss, fields_by_run

    def ensure_masks(threshold_table):
        nonlocal masks_by_run
        if masks_by_run is None:
            dss_local, fields_local = ensure_fields()
            masks_by_run = build_masks_by_run(dss_local, fields_local, threshold_table)
        return masks_by_run

    metric_products = load_cached_metric_products()
    if metric_products is None:
        dss, fields_by_run = ensure_fields()
        threshold_table = build_threshold_table(fields_by_run)
        if len(threshold_table) > 0:
            print("\nOrganisation object thresholds")
            print(threshold_table.to_string(index=False))

        masks_by_run = ensure_masks(threshold_table)
        metrics_df = build_all_metrics(dss, fields_by_run, threshold_table, masks_by_run=masks_by_run)
        if len(metrics_df) == 0:
            print("No organisation metrics were computed.")
            return

        summary = organisation_metric_summary(metrics_df)
        write_metric_cache(threshold_table, metrics_df, summary)
    else:
        threshold_table, metrics_df, summary = metric_products
        if len(threshold_table) > 0:
            print("\nCached organisation object thresholds")
            print(threshold_table.to_string(index=False))

    objects_df = pd.DataFrame()
    tracks_df = pd.DataFrame()
    contribution_df = pd.DataFrame()
    track_persistence_needed = should_run_any([6, 7, 8, 9])
    if track_persistence_needed:
        track_products = load_cached_track_persistence_products()
        if track_products is None:
            if len(threshold_table) == 0:
                _, fields_by_run = ensure_fields()
                threshold_table = build_threshold_table(fields_by_run)
            masks_by_run = ensure_masks(threshold_table)
            _, fields_by_run = ensure_fields()
            objects_df, tracks_df, contribution_df = build_track_persistence_metrics(masks_by_run, fields_by_run)
            write_track_persistence_cache(objects_df, tracks_df, contribution_df)
        else:
            objects_df, tracks_df, contribution_df = track_products

    print("\nOrganisation metric summary")
    print(summary.to_string(index=False))

    if should_run_figure(1):
        plot_lorg_timeseries(metrics_df)
    if should_run_figure(2):
        plot_lorg_distribution(metrics_df)
    if should_run_figure(3):
        plot_object_metric_distributions(metrics_df)
    if should_run_figure(4):
        plot_conditional_lorg(
            metrics_df,
            "tropical_mean_precip",
            4,
            "conditional_lorg_by_tropical_mean_precip",
            r"Tropical-mean precipitation (mm day$^{-1}$)",
        )
    if should_run_figure(5):
        plot_conditional_lorg(
            metrics_df,
            "convective_area_fraction",
            5,
            "conditional_lorg_by_convective_area_fraction",
            "Convective area fraction",
        )

    if should_run_figure(6):
        plot_track_lifetime_distribution(tracks_df)
    if should_run_figure(7):
        plot_long_lived_contribution(contribution_df)
    if should_run_figure(8):
        plot_conditional_track_lifetime(
            tracks_df,
            "mean_tropical_precip",
            8,
            "conditional_track_lifetime_by_tropical_mean_precip",
            r"Track-mean tropical precipitation (mm day$^{-1}$)",
        )
    if should_run_figure(9):
        plot_conditional_track_lifetime(
            tracks_df,
            "mean_convective_area_fraction",
            9,
            "conditional_track_lifetime_by_convective_area_fraction",
            "Track-mean convective area fraction",
        )
    if should_run_figure(10):
        persistence_df = load_cached_pattern_persistence()
        if persistence_df is None:
            _, fields_by_run = ensure_fields()
            persistence_df = compute_persistence(fields_by_run)
            write_pattern_persistence_cache(persistence_df)
        if len(persistence_df) > 0:
            plot_shifted_pattern_autocorrelation(persistence_df)

    print("\nFinished convective_organisation_diagnostics.py")


if __name__ == "__main__":
    main()
