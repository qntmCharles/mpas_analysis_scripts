#!/usr/bin/env python
# coding: utf-8

"""
Convective organisation diagnostics for MPAS-A nudging experiments.

The diagnostics here separate four related questions:

  * Lorg: instantaneous spatial arrangement/aggregation of organised
    convective-system envelopes.
  * Object size metrics: whether apparent organisation changes are actually
    changes in object size, count, or convective area.
  * Conditional Lorg: whether organisation differs between simulations at a
    fixed tropical-mean precipitation or fixed convective area fraction.
  * Persistence: lifetime/coherence of accepted organised-system envelopes.

The Lorg implementation follows the all-neighbour-distance idea of Biagioli and
Tompkins (2023): object centroids are converted to a Ripley/Besag L function,
then compared with a random reference with the same object count and domain.
Longitude is periodic both when labelling objects across the seam and when
measuring distances between centroids.
"""

import argparse
import glob
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
from matplotlib import animation, colors as mpl_colors, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage

from diagnostic_helpers import *
from high_cloud_classification import (
    CLASSIFICATION_CACHE_VERSION,
    HighCloudThresholds,
    cached_classification_is_current as shared_cached_classification_is_current,
)
from pyflextrkr_contract import (
    MIN_ORGANISED_TRACK_DURATION_HOURS,
    MIN_ORGANISED_TRACK_OBSERVATIONS,
    ORGANISED_TRACK_ACCEPTANCE_CRITERIA,
    ORGANISED_TRACK_EMBEDDED_CORE_REQUIREMENT,
    ORGANISED_TRACK_OBJECT_GEOMETRY,
    ORGANISED_ENVELOPE_MIN_RAW_AREA_KM2,
    PRIMARY_ANALYSIS_ROLE,
    PRIMARY_TRACK_SOURCE,
    PYFLEXTRKR_PRODUCT_VERSION,
    PYFLEXTRKR_TRACK_MANIFEST_FILENAME,
    TRACKED_SOURCES,
    product_fingerprint,
    manifest_accepted_mask,
    read_product_metadata,
    read_track_manifest,
    sha256_file,
    sha256_label_sequence,
    time_signature,
    track_manifest_path,
    write_product_metadata,
)

from pyflextrkr.gettracks import gettracknumbers as pyflextrkr_gettracknumbers
from pyflextrkr.idclouds_tbpf import olr_to_tb as pyflextrkr_olr_to_tb
from pyflextrkr.label_and_grow_cold_clouds import (
    label_and_grow_cold_clouds as pyflextrkr_label_and_grow_cold_clouds,
)
from pyflextrkr.ftfunctions import grow_cells as pyflextrkr_grow_cells
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
default_save_loc = save_loc

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: Lorg time series
#   2: Lorg PDF and boxplot
#   3: object count/area/area-fraction distributions
#   4: Lorg conditioned on tropical-mean precipitation
#   5: Lorg conditioned on convective area fraction
#   6: organised-system track lifetime survival and boxplots
#   7: organised-system long-lived area/precipitation contribution
#   8: organised-system lifetime conditioned on tropical-mean precipitation
#   9: organised-system lifetime conditioned on convective area fraction
#   10: propagation-aware pattern autocorrelation
#   11: PyFLEXTRKR cloud-object and track validation snapshots
#   12: separate per-simulation classification and PyFLEXTRKR tracking movies
#   13: PyFLEXTRKR track acceptance plus size/speed diagnostics
#   14: PyFLEXTRKR validation inventory and time-series dashboard
#   15: shared high-cloud classification versus PyFLEXTRKR Tb core/shield comparison
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
#   * The organised convective-system envelope is the primary tracked object set
#     used for persistence and downstream lifecycle/wave composites.  The
#     convective, stratiform, and non-precipitating high-cloud masks are sampled
#     inside each envelope rather than defining the tracked geometry.
#   * The envelope follows the MCS-tracking logic used by PyFLEXTRKR/TAMS:
#     broad cold-cloud shields are tracked, then tracks are accepted only if
#     they persist long enough and contain convective-precipitating high cloud.
#   * In Figure 11, accepted footprints should occupy coherent parts of the
#     source mask and numbered past paths should move continuously.
#   * Figure 12 separates the evolving cloud classification from accepted
#     organised-system tracks.
#   * Figure 13 shows track acceptance counts and diagnostic size/speed
#     distributions; size and speed are not acceptance criteria.
#   * Figure 14 shows whether every requested scene was written and whether
#     object counts, active tracks, lifetimes, speeds, and area changes are
#     plausible before the resulting science metrics are interpreted.
figures_to_run = []
figsave = None

TEST_MODE = False
TEST_TIME_STEPS = 16
TEST_START = None
TEST_REUSE_CACHE = False
TEST_DEFAULT_FIGURES = [11, 12, 13, 14, 15]
SIMULATION_ARGUMENTS = {
    "control": 0,
    "nudged-u": 1,
    "nudged-T": 2,
}

# Expensive organisation and persistence tables are cached in save_loc. Metadata
# must match active settings; metadata-less caches are recomputed.
USE_ORGANISATION_METRIC_CACHE = True
FORCE_RECOMPUTE_ORGANISATION_CACHE = False
FORCE_RECOMPUTE_PERSISTENCE_CACHE = False

USE_ORGANISATION_METRIC_CACHE = (
    USE_ORGANISATION_METRIC_CACHE
    and not env_flag("DISABLE_ORGANISATION_METRIC_CACHE")
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
data_locs = ["control_new", "u_nudge", "T_nudge"]
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
ENABLE_CLOUD_OBJECTS = True

# The refined high-cloud diagnostic is the single producer of cloud classes.
# Organisation diagnostics fail if that shared cache is missing or stale.
CLOUD_CLASSIFICATION_CACHE_DIRS = [
    diagnostic_cache_dir("classification_cache", root_dir=root_dir),
]
CLASSIFICATION_TEST_OUTPUT_ROOT = diagnostic_output_dir(
    "/home/cwp29/tmp/mpas/crf_on/final/high_cloud_refined/"
)

CLOUD_OBJECT_KEYS_PRIMARY = [
    "convective_precipitating_high_cloud_mask",
]
ORGANISED_SYSTEM_SOURCE = PRIMARY_TRACK_SOURCE
CLOUD_OBJECT_KEYS = [ORGANISED_SYSTEM_SOURCE]

# Broad organised-system envelopes are tracked for persistence and
# lifecycle/wave composites. Embedded convective/stratiform/non-precipitating
# cloud classes remain diagnostic attributes sampled inside each envelope.
PYFLEXTRKR_TRACKED_SOURCES = list(TRACKED_SOURCES)
PYFLEXTRKR_VALIDATION_SOURCES = list(PYFLEXTRKR_TRACKED_SOURCES)
PYFLEXTRKR_MOVIE_MAX_FRAMES = 48
PYFLEXTRKR_MOVIE_FPS = 4
PYFLEXTRKR_MOVIE_TRACK_HISTORY_STEPS = 6

# Per-scene cloud-system segmentation uses the PyFLEXTRKR detect-and-spread
# cold-cloud method. OLR is converted to a Tb proxy with PyFLEXTRKR's own
# Yang-and-Slingo-style conversion, then cold cores are grown into warmer anvil
# cloud. Cloud-top pressure remains a diagnostic attribute sampled from the
# shared classification, but it no longer restricts the PyFLEXTRKR object geometry.
PYFLEXTRKR_TB_CORE_K = 225.0
PYFLEXTRKR_TB_COLD_K = 241.0
PYFLEXTRKR_TB_WARM_K = 261.0
PYFLEXTRKR_TB_CLOUD_K = 293.0
PYFLEXTRKR_TB_MIN_VALID_K = 160.0
PYFLEXTRKR_TB_MAX_VALID_K = 330.0
PYFLEXTRKR_TB_SMOOTHING_CELLS = 5
PYFLEXTRKR_MIN_COLD_CORE_PIXELS = 1
PYFLEXTRKR_WARM_ANVIL_EXPANSION = 1

# Shared high-cloud thresholds used to validate the refined classification cache.
CF_THRESH = 0.05
P_HIGH_HPA = 440.0
P_PRECIP_MIN = 1.0
CONVECTIVE_FRACTION = 0.50

MIN_OBJECT_CELLS = 1
MIN_OBJECT_AREA_KM2 = 0.0

# Organised-envelope definition.  PyFLEXTRKR defines the cold-cloud shield
# geometry from Tb; the classification classes are sampled inside those shields.
ENVELOPE_MIN_RAW_AREA_KM2 = ORGANISED_ENVELOPE_MIN_RAW_AREA_KM2

# Accepted cold-cloud systems must be larger and longer lived than individual
# intermittent cold pixels. The 24 h duration is the science default; 6-hourly
# output means this corresponds to four sampled scenes.
MIN_TRACK_DURATION_HOURS = MIN_ORGANISED_TRACK_DURATION_HOURS
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
MIN_TRACK_OBSERVATIONS = MIN_ORGANISED_TRACK_OBSERVATIONS
LONG_LIVED_THRESHOLDS_HOURS = [12.0, 24.0, 48.0]
TRACK_CONDITION_NBINS = 4
MIN_TRACK_BIN_COUNT = 3

PERSISTENCE_MAX_LAG_DAYS = 12.0
PERSISTENCE_MAX_LON_SHIFT_CELLS = 12

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
    "organised_convective_system_mask": "Organised convective-system envelope",
    "high_cloud_mask": "High cloud",
    "convective_precipitating_high_cloud_mask": "Convective precipitating high cloud",
    "stratiform_precipitating_high_cloud_mask": "Stratiform precipitating high cloud",
    "nonprecipitating_high_cloud_mask": "Non-precipitating high cloud",
    "unclassified_high_cloud_mask": "Invalid/unclassified high cloud",
}

SOURCE_ORDER = list(CLOUD_OBJECT_KEYS)

CLASSIFICATION_DIAGNOSTIC_KEYS = [
    "high_cloud_mask",
    "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
    "unclassified_high_cloud_mask",
    "cloud_top_pressure",
]

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


def parse_command_line(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compute convective-organisation metrics and validate PyFLEXTRKR "
            "object segmentation and tracking."
        )
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run one simulation over a short isolated time window.",
    )
    parser.add_argument(
        "--simulation",
        choices=tuple(SIMULATION_ARGUMENTS),
        default="control",
        help="Simulation used by --test (default: control).",
    )
    parser.add_argument(
        "--test-times",
        type=int,
        default=TEST_TIME_STEPS,
        metavar="N",
        help="Number of tracked six-hourly snapshots in --test mode (default: 16).",
    )
    parser.add_argument(
        "--test-start",
        default=None,
        metavar="TIME",
        help="Optional first source time, for example 2000-10-01.",
    )
    parser.add_argument(
        "--reuse-test-cache",
        action="store_true",
        help="Reuse matching short-run metric and PyFLEXTRKR caches in test mode.",
    )
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        choices=range(1, 16),
        default=None,
        metavar="N",
        help="Run only the listed figure numbers (1-15).",
    )
    args = parser.parse_args(argv)
    if args.test and args.test_times < 5:
        parser.error(
            "--test-times must be at least 5 because PyFLEXTRKR's track-number "
            "allocator needs four adjacent-scene files"
        )
    return args


def configure_run(args):
    global TEST_MODE, TEST_TIME_STEPS, TEST_START, TEST_REUSE_CACHE
    global save_loc, figures_to_run, figsave, PYFLEXTRKR_WORK_DIR
    global data_locs, latlon_file, sim_names, cols
    global CLOUD_CLASSIFICATION_CACHE_DIRS
    global FORCE_RECOMPUTE_ORGANISATION_CACHE
    global FORCE_RECOMPUTE_PERSISTENCE_CACHE, FORCE_RECOMPUTE_PYFLEXTRKR_CACHE

    TEST_MODE = bool(args.test)
    TEST_TIME_STEPS = int(args.test_times)
    TEST_START = args.test_start
    TEST_REUSE_CACHE = bool(args.reuse_test_cache)

    if args.figures is not None:
        figures_to_run = list(args.figures)
    elif TEST_MODE:
        figures_to_run = list(TEST_DEFAULT_FIGURES)

    if TEST_MODE:
        index = SIMULATION_ARGUMENTS[args.simulation]
        data_locs = [data_locs[index]]
        latlon_file = [latlon_file[index]]
        sim_names = [sim_names[index]]
        cols = [cols[index]]

        start_label = f"_{TEST_START}" if TEST_START else ""
        test_label = f"{args.simulation}{start_label}_{TEST_TIME_STEPS:03d}times"
        test_label = test_label.replace(":", "").replace(" ", "_")
        save_loc = os.path.join(default_save_loc, "testing", test_label)
        PYFLEXTRKR_WORK_DIR = os.path.join(save_loc, "pyflextrkr_tracking")
        test_classification_dir = os.path.join(
            CLASSIFICATION_TEST_OUTPUT_ROOT,
            "testing",
            test_label,
            "classification_products",
        )
        CLOUD_CLASSIFICATION_CACHE_DIRS = [
            test_classification_dir,
            *CLOUD_CLASSIFICATION_CACHE_DIRS,
        ]

        if not TEST_REUSE_CACHE:
            FORCE_RECOMPUTE_ORGANISATION_CACHE = True
            FORCE_RECOMPUTE_PERSISTENCE_CACHE = True
            FORCE_RECOMPUTE_PYFLEXTRKR_CACHE = True

    figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)


def save_figure(number, fig, stem, dpi=300):
    global figsave
    if figsave is None:
        figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)
    return figsave.save(fig, stem, number=number, dpi=dpi)


def feature_label_cmap():
    cmap = plt.get_cmap("tab20").copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    return cmap


def track_id_label_plot_data(track_grid, mask):
    track_grid = np.asarray(track_grid, dtype=np.int32)
    mask = np.asarray(mask, dtype=bool) & (track_grid > 0)
    track_ids = np.unique(track_grid[mask])
    if track_ids.size == 0:
        cmap = mpl_colors.ListedColormap([(0.0, 0.0, 0.0, 0.0)])
        norm = mpl_colors.BoundaryNorm([-0.5, 0.5], cmap.N)
        return np.zeros(track_grid.shape, dtype=np.int32), cmap, norm

    max_track = int(track_ids.max())
    remap = np.zeros(max_track + 1, dtype=np.int32)
    remap[track_ids] = np.arange(1, track_ids.size + 1, dtype=np.int32)
    codes = remap[np.clip(track_grid, 0, max_track)]
    codes = np.where(mask, codes, 0)
    colors = [(0.0, 0.0, 0.0, 0.0)]
    for track_id in track_ids:
        rgba = list(pyflextrkr_track_color(track_id))
        rgba[3] = 0.82
        colors.append(tuple(rgba))
    cmap = mpl_colors.ListedColormap(colors)
    norm = mpl_colors.BoundaryNorm(
        np.arange(-0.5, track_ids.size + 1.5, 1.0),
        cmap.N,
    )
    return codes, cmap, norm


def pyflextrkr_tb_threshold_label():
    return (
        f"core <= {PYFLEXTRKR_TB_CORE_K:g} K; "
        f"cold <= {PYFLEXTRKR_TB_COLD_K:g} K; "
        f"warm/anvil <= {PYFLEXTRKR_TB_WARM_K:g} K"
    )


# -----------------------------------------------------------------------------
# Metric/persistence caches
# -----------------------------------------------------------------------------

METRIC_CACHE_VERSION = "organisation_metrics_v11_ctp_basin_shield"
TRACK_PERSISTENCE_CACHE_VERSION = "pyflextrkr_ctp_basin_shield_tracks_v13"
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
        "organised_track",
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
        "observed_span_hours",
        "n_timesteps",
        "mean_area_km2",
        "max_area_km2",
        "mean_tropical_precip",
        "mean_convective_area_fraction",
        "median_translation_speed_kmh",
        "p95_translation_speed_kmh",
        "max_translation_speed_kmh",
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
        test_mode=bool(TEST_MODE),
        test_time_steps=int(TEST_TIME_STEPS) if TEST_MODE else None,
        test_start=TEST_START if TEST_MODE else None,
    )


def object_definition_cache_signature():
    signature = base_data_cache_signature()
    signature.update(
        source_order=list(SOURCE_ORDER),
        cloud_object_keys=list(CLOUD_OBJECT_KEYS),
        enable_cloud_objects=bool(ENABLE_CLOUD_OBJECTS),
        cloud_classification_source="refined_high_cloud_classification_diagnostics.py",
        cloud_classification_cache_version=CLASSIFICATION_CACHE_VERSION,
        shared_cloud_thresholds=jsonable_value(vars(classification_thresholds())),
        cloud_classification="0.5-degree MPAS precipitation-regime classification",
        min_object_cells=int(MIN_OBJECT_CELLS),
        min_object_area_km2=float(MIN_OBJECT_AREA_KM2),
        organised_system_source=ORGANISED_SYSTEM_SOURCE,
        object_geometry=ORGANISED_TRACK_OBJECT_GEOMETRY,
        envelope_min_raw_area_km2=float(ENVELOPE_MIN_RAW_AREA_KM2),
        pyflextrkr_tb_core_k=float(PYFLEXTRKR_TB_CORE_K),
        pyflextrkr_tb_cold_k=float(PYFLEXTRKR_TB_COLD_K),
        pyflextrkr_tb_warm_k=float(PYFLEXTRKR_TB_WARM_K),
        pyflextrkr_tb_cloud_k=float(PYFLEXTRKR_TB_CLOUD_K),
        pyflextrkr_tb_min_valid_k=float(PYFLEXTRKR_TB_MIN_VALID_K),
        pyflextrkr_tb_max_valid_k=float(PYFLEXTRKR_TB_MAX_VALID_K),
        pyflextrkr_tb_smoothing_cells=int(PYFLEXTRKR_TB_SMOOTHING_CELLS),
        pyflextrkr_min_cold_core_pixels=int(PYFLEXTRKR_MIN_COLD_CORE_PIXELS),
        pyflextrkr_warm_anvil_expansion=int(PYFLEXTRKR_WARM_ANVIL_EXPANSION),
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
        product_version=PYFLEXTRKR_PRODUCT_VERSION,
        primary_track_source=ORGANISED_SYSTEM_SOURCE,
        tracked_sources=list(PYFLEXTRKR_TRACKED_SOURCES),
        pyflextrkr_nmaxlinks=int(PYFLEXTRKR_NMAXLINKS),
        pyflextrkr_fillval=int(PYFLEXTRKR_FILLVAL),
        track_min_overlap_fraction=float(TRACK_MIN_OVERLAP_FRACTION),
        track_min_area_km2=float(TRACK_MIN_AREA_KM2),
        min_track_observations=int(MIN_TRACK_OBSERVATIONS),
        min_track_duration_hours=float(MIN_TRACK_DURATION_HOURS),
        track_acceptance_criteria=ORGANISED_TRACK_ACCEPTANCE_CRITERIA,
        embedded_convective_core_requirement=ORGANISED_TRACK_EMBEDDED_CORE_REQUIREMENT,
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
        print(f"Ignoring cached {label}: metadata file is missing.")
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


def source_label(source_name):
    return SOURCE_LABELS.get(source_name, source_name.replace("_", " "))


def track_source_role(source_name):
    if source_name == ORGANISED_SYSTEM_SOURCE:
        return PRIMARY_ANALYSIS_ROLE
    raise ValueError(
        f"PyFLEXTRKR output is only produced for {ORGANISED_SYSTEM_SOURCE!r}, "
        f"got {source_name!r}"
    )


def ordered_sources(metrics_df):
    present = set(metrics_df.source.unique())
    ordered = [name for name in SOURCE_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


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


def label_objects_with_properties(mask2d, area2d, precip2d=None, field_slices=None):
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
    context_values = {}
    if field_slices:
        for key, da in field_slices.items():
            if da is None or not {"latitude", "longitude"}.issubset(da.dims):
                continue
            da = da.transpose("latitude", "longitude")
            da = da.sel(latitude=mask2d.latitude, longitude=mask2d.longitude)
            context_values[key] = np.asarray(da.values, dtype=float)

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
    lon_coord = np.asarray(mask2d.longitude.values, dtype=float)
    lon_vals = lon_coord % LONGITUDE_PERIOD_DEG
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
            mean_precip = np.nan
            max_precip = np.nan
        else:
            pr = np.where(np.isfinite(precip_vals[obj]), precip_vals[obj], 0.0)
            precip_area_rate = float(np.nansum(pr * weights))
            mean_precip = float(np.average(pr, weights=weights))
            max_precip = float(np.nanmax(pr)) if pr.size else np.nan
        centroid_lat = float(np.average(lat_grid[obj], weights=weights))
        theta = 2.0 * np.pi * lon_grid[obj] / LONGITUDE_PERIOD_DEG
        sin_mean = np.average(np.sin(theta), weights=weights)
        cos_mean = np.average(np.cos(theta), weights=weights)
        centroid_lon = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0
        if np.nanmin(lon_coord) < 0.0:
            centroid_lon = (centroid_lon + 180.0) % 360.0 - 180.0

        row = dict(
            label=int(label_id),
            cell_count=cell_count,
            area_km2=area,
            precip_area_rate=precip_area_rate,
            mean_precip=mean_precip,
            max_precip=max_precip,
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
        )
        for key, vals in context_values.items():
            v = vals[obj]
            ok = np.isfinite(v)
            if not np.any(ok):
                row[f"mean_{key}"] = np.nan
                row[f"max_{key}"] = np.nan
                row[f"min_{key}"] = np.nan
                continue
            w = weights[ok]
            vv = v[ok]
            row[f"mean_{key}"] = float(np.average(vv, weights=w))
            row[f"max_{key}"] = float(np.nanmax(vv))
            row[f"min_{key}"] = float(np.nanmin(vv))
            if key.endswith("_mask"):
                active = vv > 0.5
                area_active = float(np.nansum(w[active]))
                row[f"area_{key}_km2"] = area_active
                row[f"frac_{key}"] = area_active / area if area > 0.0 else np.nan
        rows.append(row)

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
    if source_name != ORGANISED_SYSTEM_SOURCE:
        raise ValueError(
            f"PyFLEXTRKR output is only produced for {ORGANISED_SYSTEM_SOURCE!r}, "
            f"got {source_name!r}"
        )
    safe_sim = sim_name.replace(" ", "_").replace("/", "_")
    source_cache_name = f"{source_name}_{CLASSIFICATION_CACHE_VERSION}"
    safe_source = source_cache_name.replace(" ", "_").replace("/", "_")
    return os.path.join(PYFLEXTRKR_WORK_DIR, safe_source, safe_sim)


def pyflextrkr_tracknumbers_filename(basetimes):
    startdate = pd.to_datetime(float(basetimes[0]), unit="s").strftime("%Y%m%d.%H%M")
    enddate = pd.to_datetime(float(basetimes[-1]), unit="s").strftime("%Y%m%d.%H%M")
    return f"tracknumbers_{startdate}_{enddate}.nc"


def pyflextrkr_tracknumbers_path(work_path, basetimes):
    return os.path.join(work_path, "stats", pyflextrkr_tracknumbers_filename(basetimes))


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


def build_organised_track_manifest(objects_df):
    """Classify raw PyFLEXTRKR tracks as accepted organised systems or noise."""
    rows = []
    for (source, simulation, track_id), raw_group in objects_df.groupby(
        ["source", "simulation", "track_id"]
    ):
        # During a split/merge, use the largest branch to diagnose the centroid
        # path that downstream composites would follow.
        group = (
            raw_group.sort_values(
                ["time_index", "area_km2"], ascending=[True, False]
            )
            .drop_duplicates("time_index")
            .sort_values("time_index")
        )
        observations = int(len(group))
        speeds = np.array([], dtype=float)
        step_distances = np.array([], dtype=float)
        step_hours = np.array([], dtype=float)
        if observations >= 2:
            step_distances = np.asarray(
                haversine_distances_km(
                    group.centroid_lat.to_numpy(dtype=float)[:-1],
                    group.centroid_lon.to_numpy(dtype=float)[:-1],
                    group.centroid_lat.to_numpy(dtype=float)[1:],
                    group.centroid_lon.to_numpy(dtype=float)[1:],
                ),
                dtype=float,
            )
            step_hours = np.asarray(
                (
                    pd.to_datetime(group.Time.to_numpy()[1:])
                    - pd.to_datetime(group.Time.to_numpy()[:-1])
                )
                / pd.Timedelta(hours=1),
                dtype=float,
            )
            speeds = np.divide(
                step_distances,
                step_hours,
                out=np.full(step_distances.shape, np.nan),
                where=step_hours > 0.0,
            )

        finite_speeds = speeds[np.isfinite(speeds)]
        max_speed = (
            float(np.nanmax(finite_speeds)) if finite_speeds.size else np.nan
        )

        median_dt = (
            float(np.nanmedian(group.dt_hours.to_numpy(dtype=float)))
            if "dt_hours" in group and len(group)
            else float(dt)
        )
        sampling_support_hours = float(observations * median_dt)
        observed_span_hours = (
            float(
                (pd.to_datetime(group.Time.iloc[-1]) - pd.to_datetime(group.Time.iloc[0]))
                / pd.Timedelta(hours=1)
            )
            if observations >= 2
            else 0.0
        )
        max_area = float(group.area_km2.max()) if "area_km2" in group else np.nan
        core_area_col = "area_convective_precipitating_high_cloud_mask_km2"
        max_core_area = (
            float(group[core_area_col].max())
            if core_area_col in group
            else 0.0
        )
        has_embedded_convective_core = bool(max_core_area > 0.0)
        core_fraction_col = "frac_convective_precipitating_high_cloud_mask"
        max_core_fraction = (
            float(group[core_fraction_col].max())
            if core_fraction_col in group
            else 0.0
        )
        mean_core_fraction = (
            float(group[core_fraction_col].mean())
            if core_fraction_col in group
            else 0.0
        )
        max_precip = (
            float(group.max_precip.max())
            if "max_precip" in group
            else np.nan
        )
        centroid_lats = group.centroid_lat.to_numpy(dtype=float)
        finite_lats = centroid_lats[np.isfinite(centroid_lats)]
        max_abs_centroid_lat = (
            float(np.nanmax(np.abs(finite_lats))) if finite_lats.size else np.nan
        )
        rows.append(
            dict(
                source=source,
                simulation=simulation,
                track_id=int(track_id),
                accepted=False,
                rejection_reason="pending",
                observation_count=observations,
                first_time_index=int(group.time_index.iloc[0]),
                last_time_index=int(group.time_index.iloc[-1]),
                observed_span_hours=observed_span_hours,
                sampling_support_hours=sampling_support_hours,
                max_area_km2=max_area,
                has_embedded_convective_core=has_embedded_convective_core,
                max_embedded_convective_core_area_km2=max_core_area,
                max_embedded_convective_core_fraction=max_core_fraction,
                mean_embedded_convective_core_fraction=mean_core_fraction,
                max_precip_mm_day=max_precip,
                min_centroid_lat=float(np.nanmin(finite_lats)) if finite_lats.size else np.nan,
                max_centroid_lat=float(np.nanmax(finite_lats)) if finite_lats.size else np.nan,
                max_abs_centroid_lat=max_abs_centroid_lat,
                median_translation_speed_kmh=(
                    float(np.nanmedian(finite_speeds))
                    if finite_speeds.size
                    else np.nan
                ),
                p95_translation_speed_kmh=(
                    float(np.nanpercentile(finite_speeds, 95))
                    if finite_speeds.size
                    else np.nan
                ),
                max_translation_speed_kmh=max_speed,
                max_step_distance_km=(
                    float(np.nanmax(step_distances))
                    if step_distances.size
                    else np.nan
                ),
                min_track_observations=int(MIN_TRACK_OBSERVATIONS),
                min_track_duration_hours=float(MIN_TRACK_DURATION_HOURS),
                track_acceptance_criteria=ORGANISED_TRACK_ACCEPTANCE_CRITERIA,
                embedded_convective_core_requirement=ORGANISED_TRACK_EMBEDDED_CORE_REQUIREMENT,
            )
        )
        row_reasons = []
        if observations < MIN_TRACK_OBSERVATIONS:
            row_reasons.append(
                f"fewer_than_{MIN_TRACK_OBSERVATIONS}_observations"
            )
        if sampling_support_hours < MIN_TRACK_DURATION_HOURS:
            row_reasons.append("duration_below_threshold")
        if not has_embedded_convective_core:
            row_reasons.append("no_embedded_convective_core")
        rows[-1]["accepted"] = len(row_reasons) == 0
        rows[-1]["rejection_reason"] = ";".join(row_reasons) if row_reasons else "accepted"
    return pd.DataFrame(rows)


def write_organised_track_manifest(work_path, objects_df):
    manifest = build_organised_track_manifest(objects_df)
    path = track_manifest_path(work_path)
    tmp_path = path + ".tmp"
    manifest.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
    return manifest


def accepted_track_ids(manifest):
    if manifest is None or len(manifest) == 0:
        return set()
    accepted = manifest[manifest_accepted_mask(manifest)]
    return set(accepted.track_id.to_numpy(dtype=int))


def record_pyflextrkr_product(
    work_path,
    tracknumbers_file,
    labels_by_time,
    mask,
    source_name,
    sim_name,
    track_manifest,
):
    manifest_path = track_manifest_path(work_path)
    metadata = dict(
        product_version=PYFLEXTRKR_PRODUCT_VERSION,
        source=source_name,
        simulation=sim_name,
        analysis_role=track_source_role(source_name),
        classification_cache_version=CLASSIFICATION_CACHE_VERSION,
        cloudid_labels_sha256=sha256_label_sequence(labels_by_time),
        tracknumbers_file=os.path.basename(tracknumbers_file),
        tracknumbers_sha256=sha256_file(tracknumbers_file),
        track_min_overlap_fraction=float(TRACK_MIN_OVERLAP_FRACTION),
        track_min_area_km2=float(TRACK_MIN_AREA_KM2),
        min_track_observations=int(MIN_TRACK_OBSERVATIONS),
        min_track_duration_hours=float(MIN_TRACK_DURATION_HOURS),
        track_acceptance_criteria=ORGANISED_TRACK_ACCEPTANCE_CRITERIA,
        embedded_convective_core_requirement=ORGANISED_TRACK_EMBEDDED_CORE_REQUIREMENT,
        object_geometry=ORGANISED_TRACK_OBJECT_GEOMETRY,
        envelope_min_raw_area_km2=float(ENVELOPE_MIN_RAW_AREA_KM2),
        pyflextrkr_tb_core_k=float(PYFLEXTRKR_TB_CORE_K),
        pyflextrkr_tb_cold_k=float(PYFLEXTRKR_TB_COLD_K),
        pyflextrkr_tb_warm_k=float(PYFLEXTRKR_TB_WARM_K),
        pyflextrkr_tb_cloud_k=float(PYFLEXTRKR_TB_CLOUD_K),
        pyflextrkr_tb_smoothing_cells=int(PYFLEXTRKR_TB_SMOOTHING_CELLS),
        pyflextrkr_min_cold_core_pixels=int(PYFLEXTRKR_MIN_COLD_CORE_PIXELS),
        pyflextrkr_warm_anvil_expansion=int(PYFLEXTRKR_WARM_ANVIL_EXPANSION),
        track_manifest_file=PYFLEXTRKR_TRACK_MANIFEST_FILENAME,
        track_manifest_sha256=sha256_file(manifest_path),
        raw_track_count=int(len(track_manifest)),
        accepted_track_count=int(manifest_accepted_mask(track_manifest).sum()),
        rejected_track_count=int((~manifest_accepted_mask(track_manifest)).sum()),
        pyflextrkr_nmaxlinks=int(PYFLEXTRKR_NMAXLINKS),
        **time_signature(mask.Time.values),
    )
    return write_product_metadata(work_path, metadata)


def cached_pyflextrkr_product_is_current(
    work_path,
    tracknumbers_file,
    labels_by_time,
    mask,
    source_name,
    sim_name,
):
    metadata = read_product_metadata(work_path)
    if metadata is None or not os.path.exists(tracknumbers_file):
        return False
    expected = dict(
        product_version=PYFLEXTRKR_PRODUCT_VERSION,
        source=source_name,
        simulation=sim_name,
        analysis_role=track_source_role(source_name),
        classification_cache_version=CLASSIFICATION_CACHE_VERSION,
        cloudid_labels_sha256=sha256_label_sequence(labels_by_time),
        tracknumbers_file=os.path.basename(tracknumbers_file),
        track_min_overlap_fraction=float(TRACK_MIN_OVERLAP_FRACTION),
        track_min_area_km2=float(TRACK_MIN_AREA_KM2),
        min_track_observations=int(MIN_TRACK_OBSERVATIONS),
        min_track_duration_hours=float(MIN_TRACK_DURATION_HOURS),
        track_acceptance_criteria=ORGANISED_TRACK_ACCEPTANCE_CRITERIA,
        embedded_convective_core_requirement=ORGANISED_TRACK_EMBEDDED_CORE_REQUIREMENT,
        object_geometry=ORGANISED_TRACK_OBJECT_GEOMETRY,
        envelope_min_raw_area_km2=float(ENVELOPE_MIN_RAW_AREA_KM2),
        pyflextrkr_tb_core_k=float(PYFLEXTRKR_TB_CORE_K),
        pyflextrkr_tb_cold_k=float(PYFLEXTRKR_TB_COLD_K),
        pyflextrkr_tb_warm_k=float(PYFLEXTRKR_TB_WARM_K),
        pyflextrkr_tb_cloud_k=float(PYFLEXTRKR_TB_CLOUD_K),
        pyflextrkr_tb_smoothing_cells=int(PYFLEXTRKR_TB_SMOOTHING_CELLS),
        pyflextrkr_min_cold_core_pixels=int(PYFLEXTRKR_MIN_COLD_CORE_PIXELS),
        pyflextrkr_warm_anvil_expansion=int(PYFLEXTRKR_WARM_ANVIL_EXPANSION),
        pyflextrkr_nmaxlinks=int(PYFLEXTRKR_NMAXLINKS),
        **time_signature(mask.Time.values),
    )
    if any(metadata.get(key) != value for key, value in expected.items()):
        return False
    manifest_path = track_manifest_path(work_path)
    return (
        metadata.get("tracknumbers_sha256") == sha256_file(tracknumbers_file)
        and os.path.exists(manifest_path)
        and metadata.get("track_manifest_sha256") == sha256_file(manifest_path)
    )


def attach_and_record_pyflextrkr_product(
    work_path,
    tracknumbers_file,
    labels_by_time,
    objects_df,
    mask,
    source_name,
    sim_name,
):
    attached = attach_pyflextrkr_track_ids(tracknumbers_file, objects_df)
    manifest = write_organised_track_manifest(work_path, attached)
    record_pyflextrkr_product(
        work_path,
        tracknumbers_file,
        labels_by_time,
        mask,
        source_name,
        sim_name,
        manifest,
    )
    accepted_ids = accepted_track_ids(manifest)
    accepted = (
        attached[attached.track_id.isin(accepted_ids)]
        .sort_values(["track_id", "time_index", "area_km2"], ascending=[True, True, False])
        .drop_duplicates(["track_id", "time_index"])
        .copy()
    )
    accepted["organised_track"] = True
    print(
        f"{sim_name} {source_name}: accepted {len(accepted_ids)} of "
        f"{len(manifest)} raw tracks as organised systems "
        f"(>= {MIN_TRACK_DURATION_HOURS:g} h support and embedded "
        "convective-precipitating cloud at least once)",
        flush=True,
    )
    return accepted


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

    cached_product_current = cached_pyflextrkr_product_is_current(
        work_path,
        tracknumbers_file,
        labels_by_time,
        mask,
        source_name,
        sim_name,
    )
    if not FORCE_RECOMPUTE_PYFLEXTRKR_CACHE and cached_product_current:
        print(f"Using cached PyFLEXTRKR track numbers for {sim_name} {source_name}: {tracknumbers_file}")
        return attach_and_record_pyflextrkr_product(
            work_path, tracknumbers_file, labels_by_time, objects_df, mask,
            source_name, sim_name,
        )

    if os.path.exists(tracknumbers_file):
        print(
            f"Regenerating stale PyFLEXTRKR product for {sim_name} {source_name}: "
            f"{tracknumbers_file}"
        )
        os.remove(tracknumbers_file)

    files, basetimes, tracking_path = write_pyflextrkr_cloudid_files(labels_by_time, mask, work_path)
    old_cwd = os.getcwd()
    try:
        os.chdir(tracking_path)
        for it in range(len(files) - 1):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Converting non-nanosecond precision datetime values.*",
                    category=UserWarning,
                )
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
    return attach_and_record_pyflextrkr_product(
        work_path, tracknumbers_file, labels_by_time, objects_df, mask,
        source_name, sim_name,
    )


def build_track_tables(mask, fields, source_name, sim_name, context_masks=None):
    """
    Link object masks through time using PyFLEXTRKR-method overlap tracking.

    PyFLEXTRKR links adjacent objects when either forward or backward overlap
    fraction exceeds the threshold. During merges/splits, the largest branch is
    tracked continuously and smaller branches are recorded as merge/split links.
    """
    mask = subset_lat(mask.fillna(False), lat_bounds=TROPICAL_LAT_BOUNDS)
    context_masks = context_masks or {}
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
            field_slices={
                key: da.isel(Time=it).load()
                for key, da in context_masks.items()
                if key in (
                    "high_cloud_mask",
                    "convective_precipitating_high_cloud_mask",
                    "stratiform_precipitating_high_cloud_mask",
                    "nonprecipitating_high_cloud_mask",
                    "unclassified_high_cloud_mask",
                    "cloud_top_pressure",
                )
                and "Time" in da.dims
            },
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
                    analysis_role=track_source_role(source_name),
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

    track_time_agg = dict(
        Time=("Time", "first"),
        time_days=("time_days", "first"),
        dt_hours=("dt_hours", "first"),
        area_km2=("area_km2", "sum"),
        precip_area_rate=("precip_area_rate", "sum"),
        mean_precip=("mean_precip", "mean"),
        max_precip=("max_precip", "max"),
        object_count=("label", "count"),
        tropical_mean_precip=("tropical_mean_precip", "first"),
        scene_convective_area_fraction=("scene_convective_area_fraction", "first"),
    )
    for col in objects_df.columns:
        if col in track_time_agg:
            continue
        if col.startswith("area_"):
            track_time_agg[col] = (col, "sum")
        elif col.startswith(("frac_", "mean_")):
            track_time_agg[col] = (col, "mean")
        elif col.startswith("max_"):
            track_time_agg[col] = (col, "max")
        elif col.startswith("min_"):
            track_time_agg[col] = (col, "min")

    track_time = (
        objects_df
        .groupby(["source", "source_label", "simulation", "track_id", "time_index"], as_index=False)
        .agg(**track_time_agg)
    )
    accepted_quality = build_organised_track_manifest(objects_df)
    if not manifest_accepted_mask(accepted_quality).all():
        raise RuntimeError(
            f"{sim_name}: rejected PyFLEXTRKR candidates leaked into the "
            "organised-track tables"
        )
    accepted_quality = accepted_quality.set_index("track_id")

    tracks = []
    for (source, label, simulation, track_id), group in track_time.groupby(
        ["source", "source_label", "simulation", "track_id"]
    ):
        group = group.sort_values("time_index")
        lifetime_steps = int(group.time_index.nunique())
        lifetime_hours = float(lifetime_steps * dt_hours)
        quality = accepted_quality.loc[int(track_id)]
        track_objects = objects_df[objects_df.track_id == track_id]
        start_objects = track_objects[track_objects.time_index == group.time_index.min()]
        end_objects = track_objects[track_objects.time_index == group.time_index.max()]
        split_from = start_objects.start_split_track_id.dropna().values
        merge_into = end_objects.end_merge_track_id.dropna().values
        tracks.append(
            dict(
                source=source,
                source_label=label,
                analysis_role=track_source_role(source),
                simulation=simulation,
                track_id=int(track_id),
                start_time=group.Time.iloc[0],
                end_time=group.Time.iloc[-1],
                start_day=float(group.time_days.iloc[0]),
                end_day=float(group.time_days.iloc[-1]),
                lifetime_steps=lifetime_steps,
                lifetime_hours=lifetime_hours,
                observed_span_hours=float(quality.observed_span_hours),
                max_area_km2=float(group.area_km2.max()),
                mean_area_km2=float(group.area_km2.mean()),
                area_time_km2_h=float(np.nansum(group.area_km2.values * group.dt_hours.values)),
                precip_area_rate_time=float(np.nansum(group.precip_area_rate.values * group.dt_hours.values)),
                max_precip=float(group.max_precip.max()),
                mean_tropical_precip=float(group.tropical_mean_precip.mean()),
                mean_convective_area_fraction=float(group.scene_convective_area_fraction.mean()),
                max_object_count=int(group.object_count.max()),
                median_translation_speed_kmh=float(
                    quality.median_translation_speed_kmh
                ),
                p95_translation_speed_kmh=float(
                    quality.p95_translation_speed_kmh
                ),
                max_translation_speed_kmh=float(
                    quality.max_translation_speed_kmh
                ),
                starts_as_split=bool(split_from.size > 0),
                ends_as_merge=bool(merge_into.size > 0),
                split_from_track_id=float(split_from[0]) if split_from.size > 0 else np.nan,
                merge_into_track_id=float(merge_into[-1]) if merge_into.size > 0 else np.nan,
            )
        )
        for col in group.columns:
            if col.startswith("area_"):
                tracks[-1][f"max_{col}"] = float(group[col].max())
                tracks[-1][f"mean_{col}"] = float(group[col].mean())
            elif col.startswith("frac_"):
                tracks[-1][f"mean_{col}"] = float(group[col].mean())
                tracks[-1][f"max_{col}"] = float(group[col].max())
            elif col.startswith(("mean_", "max_", "min_")) and col not in tracks[-1]:
                values = group[col].to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                tracks[-1][col] = float(np.nanmean(finite)) if finite.size else np.nan
    tracks_df = pd.DataFrame(tracks)
    if len(tracks_df) == 0:
        return objects_df, tracks_df, pd.DataFrame()

    track_time = track_time.merge(
        tracks_df[[
            "source", "simulation", "track_id", "lifetime_hours",
            "max_translation_speed_kmh",
        ]],
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
                    analysis_role=track_source_role(record["source"]),
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
        for source_name in [name for name in PYFLEXTRKR_TRACKED_SOURCES if name in masks]:
            print(
                f"Tracking object persistence for {sim_name}: {source_name}",
                flush=True,
            )
            objects, tracks, contribution = build_track_tables(
                masks[source_name],
                fields_by_run[i],
                source_name,
                sim_name,
                context_masks=masks,
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


def write_pyflextrkr_obsolete_output_report():
    rows = []
    expected_paths = {
        os.path.abspath(pyflextrkr_work_path(source, sim)): (source, sim)
        for source in PYFLEXTRKR_TRACKED_SOURCES
        for sim in sim_names
    }
    candidate_paths = {
        os.path.abspath(path)
        for path in glob.glob(os.path.join(PYFLEXTRKR_WORK_DIR, "*", "*"))
        if os.path.isdir(path)
    }
    for path in sorted(candidate_paths):
        metadata = read_product_metadata(path)
        if path not in expected_paths:
            rows.append(dict(path=path, reason="not a current tracked source/simulation"))
            continue
        if metadata is None:
            rows.append(dict(path=path, reason="missing product metadata"))
            continue
        if metadata.get("product_version") != PYFLEXTRKR_PRODUCT_VERSION:
            rows.append(dict(path=path, reason="obsolete product version"))
            continue
        source, sim = expected_paths[path]
        if (
            metadata.get("source") != source
            or metadata.get("simulation") != sim
            or metadata.get("classification_cache_version")
            != CLASSIFICATION_CACHE_VERSION
        ):
            rows.append(dict(path=path, reason="metadata does not match current product definition"))
            continue
        if metadata.get("product_fingerprint") != product_fingerprint(metadata):
            rows.append(dict(path=path, reason="invalid product fingerprint"))
            continue
        selected = metadata.get("tracknumbers_file", "")
        selected_path = os.path.join(path, "stats", selected)
        if not selected or not os.path.exists(selected_path):
            rows.append(dict(path=path, reason="selected track-number file is missing"))
            continue
        if metadata.get("tracknumbers_sha256") != sha256_file(selected_path):
            rows.append(dict(path=path, reason="selected track-number checksum mismatch"))
        manifest_path = track_manifest_path(path)
        if not os.path.exists(manifest_path):
            rows.append(dict(path=path, reason="organised-track manifest is missing"))
        elif metadata.get("track_manifest_sha256") != sha256_file(manifest_path):
            rows.append(dict(path=path, reason="organised-track manifest checksum mismatch"))
        for track_file in glob.glob(os.path.join(path, "stats", "tracknumbers_*.nc")):
            if os.path.basename(track_file) != selected:
                rows.append(dict(path=track_file, reason="unselected track-number file"))

    report = pd.DataFrame(rows, columns=["path", "reason"])
    report_path = os.path.join(save_loc, "pyflextrkr_obsolete_output_candidates.csv")
    report.to_csv(report_path, index=False)
    if len(report):
        print(
            f"Found {len(report)} PyFLEXTRKR output candidate(s) for manual deletion; "
            f"see {report_path}",
            flush=True,
        )
    else:
        print(f"No obsolete PyFLEXTRKR outputs found; wrote {report_path}", flush=True)
    return report


def write_organised_track_acceptance_report():
    manifests = []
    for sim_name in sim_names:
        work_path = pyflextrkr_work_path(ORGANISED_SYSTEM_SOURCE, sim_name)
        manifest = read_track_manifest(work_path)
        if manifest is None:
            raise RuntimeError(
                f"{sim_name}: organised-track manifest is missing: "
                f"{track_manifest_path(work_path)}"
            )
        manifests.append(manifest)
    combined = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()
    combined_path = os.path.join(save_loc, "pyflextrkr_organised_track_acceptance.csv")
    combined.to_csv(combined_path, index=False)

    summary_rows = []
    for simulation, group in combined.groupby("simulation"):
        accepted = manifest_accepted_mask(group)
        reasons = group.loc[~accepted, "rejection_reason"].astype(str)
        summary_rows.append(
            dict(
                simulation=simulation,
                raw_track_count=int(len(group)),
                accepted_track_count=int(accepted.sum()),
                rejected_track_count=int((~accepted).sum()),
                accepted_fraction=float(accepted.mean()) if len(group) else np.nan,
                rejected_single_snapshot_count=int(
                    reasons.str.contains(
                        f"fewer_than_{MIN_TRACK_OBSERVATIONS}_observations",
                        regex=False,
                    ).sum()
                ),
                rejected_short_duration_count=int(
                    reasons.str.contains("duration_below_threshold", regex=False).sum()
                ),
                rejected_no_embedded_core_count=int(
                    reasons.str.contains("no_embedded_convective_core", regex=False).sum()
                ),
                min_track_observations=int(MIN_TRACK_OBSERVATIONS),
                min_track_duration_hours=float(MIN_TRACK_DURATION_HOURS),
                track_acceptance_criteria=ORGANISED_TRACK_ACCEPTANCE_CRITERIA,
                embedded_convective_core_requirement=ORGANISED_TRACK_EMBEDDED_CORE_REQUIREMENT,
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(save_loc, "pyflextrkr_organised_track_acceptance_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote organised-track acceptance reports: {combined_path}, {summary_path}")
    return combined, summary


# -----------------------------------------------------------------------------
# Cloud-type masks
# -----------------------------------------------------------------------------


def has_primary_cloud_masks(masks):
    return all(key in masks for key in CLOUD_OBJECT_KEYS_PRIMARY)


def classification_thresholds():
    return HighCloudThresholds(
        cf_thresh=CF_THRESH,
        high_cloud_top_pressure_hpa=P_HIGH_HPA,
        surface_precip_min_mm_day=P_PRECIP_MIN,
        convective_fraction_threshold=CONVECTIVE_FRACTION,
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
    missing = [
        key for key in CLASSIFICATION_DIAGNOSTIC_KEYS
        if key not in cached.data_vars
    ]
    if missing:
        cached.close()
        raise RuntimeError(
            f"{sim_name}: high-cloud classification cache lacks required shared variables {missing}: {path}"
        )
    if "Time" not in cached.dims or "Time" not in precip_ref.dims:
        cached.close()
        raise RuntimeError(
            f"{sim_name}: classification and precipitation must both have Time coordinates"
        )
    requested_time = precip_ref.Time.to_index()
    missing_time = requested_time.difference(cached.Time.to_index())
    if len(missing_time) > 0:
        cached.close()
        preview = ", ".join(str(value) for value in missing_time[:4])
        suffix = "" if len(missing_time) <= 4 else ", ..."
        raise RuntimeError(
            f"{sim_name}: shared classification is missing {len(missing_time)} requested "
            f"times ({preview}{suffix}): {path}"
        )
    masks = {}
    for key in CLASSIFICATION_DIAGNOSTIC_KEYS:
        da = cached[key]
        da = select_common_time(da, precip_ref)
        masks[key] = da.fillna(False).astype(bool) if key.endswith("_mask") else finite_or_nan(da)
    return masks


def cloud_type_masks(ds, fields, sim_name):
    if not ENABLE_CLOUD_OBJECTS:
        return {}

    masks = cloud_masks_from_cache(sim_name, fields["precip"])
    if not has_primary_cloud_masks(masks):
        raise RuntimeError(f"{sim_name}: shared high-cloud classification cache lacks primary cloud-object masks")
    return masks


def remove_label_boundaries(labels):
    labels = np.asarray(labels, dtype=np.int32)
    keep = labels > 0
    if not np.any(keep):
        return keep
    boundary = np.zeros(labels.shape, dtype=bool)
    rolled = np.roll(labels, 1, axis=1)
    diff = (labels > 0) & (rolled > 0) & (labels != rolled)
    boundary |= diff
    boundary |= np.roll(diff, -1, axis=1)

    upper = labels[:-1, :]
    lower = labels[1:, :]
    vdiff = (upper > 0) & (lower > 0) & (upper != lower)
    boundary[:-1, :] |= vdiff
    boundary[1:, :] |= vdiff
    return keep & ~boundary


def pyflextrkr_tb_thresholds():
    return np.asarray(
        [
            PYFLEXTRKR_TB_CORE_K,
            PYFLEXTRKR_TB_COLD_K,
            PYFLEXTRKR_TB_WARM_K,
            PYFLEXTRKR_TB_CLOUD_K,
        ],
        dtype=float,
    )


def olr_to_pyflextrkr_tb(olr):
    values = np.asarray(olr, dtype=float)
    tb = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0.0)
    if np.any(valid):
        with np.errstate(invalid="ignore", divide="ignore"):
            tb[valid] = pyflextrkr_olr_to_tb(values[valid])
    valid_tb = (
        np.isfinite(tb)
        & (tb >= PYFLEXTRKR_TB_MIN_VALID_K)
        & (tb <= PYFLEXTRKR_TB_MAX_VALID_K)
    )
    return np.where(valid_tb, tb, np.nan)


def pyflextrkr_label_grow_scene(olr, area2d):
    area = np.asarray(area2d, dtype=float)
    tb = olr_to_pyflextrkr_tb(olr)
    if not np.any(np.isfinite(tb)):
        return np.zeros_like(tb, dtype=np.int32)

    valid_area = area[np.isfinite(tb) & np.isfinite(area)]
    pixel_area_km2 = float(np.nanmedian(valid_area)) if valid_area.size else 1.0
    pixel_radius_km = float(np.sqrt(max(pixel_area_km2, 1.0)))
    clouddata = pyflextrkr_label_and_grow_cold_clouds(
        tb,
        pixel_radius_km,
        pyflextrkr_tb_thresholds(),
        ENVELOPE_MIN_RAW_AREA_KM2,
        PYFLEXTRKR_MIN_COLD_CORE_PIXELS,
        PYFLEXTRKR_TB_SMOOTHING_CELLS,
        0,
    )
    labels = np.asarray(clouddata["final_cloudnumber"], dtype=np.int32)
    if PYFLEXTRKR_WARM_ANVIL_EXPANSION:
        # PyFLEXTRKR's MCS method grows cold-core labels into surrounding warm
        # anvil without merging the seed labels. Use its queue-based grow_cells
        # implementation directly for the warm-anvil step so each MPAS scene is
        # processed in one pass.
        grow_grid = labels.copy()
        excluded = (~np.isfinite(tb)) | (tb > PYFLEXTRKR_TB_WARM_K)
        grow_grid[excluded] = -1
        grow_grid[(~excluded) & (grow_grid <= 0)] = 0
        labels = pyflextrkr_grow_cells(grow_grid).astype(np.int32)
    labels = np.where(np.isfinite(tb), labels, 0).astype(np.int32)
    return merge_periodic_longitude_labels(labels)


def organised_envelope_scene(olr, area2d):
    area = np.asarray(area2d, dtype=float)
    labels = pyflextrkr_label_grow_scene(olr, area)
    if not np.any(labels > 0):
        return np.zeros_like(labels, dtype=bool)

    retained_labels = np.zeros(labels.shape, dtype=np.int32)
    for label_id in np.unique(labels[labels > 0]):
        obj = labels == label_id
        if float(np.nansum(area[obj])) < ENVELOPE_MIN_RAW_AREA_KM2:
            continue
        retained_labels[obj] = int(label_id)

    envelope = remove_label_boundaries(retained_labels)
    clean_labels = merge_periodic_longitude_labels(
        ndimage.label(envelope, structure=np.ones((3, 3), dtype=np.int8))[0]
    )
    cleaned = np.zeros_like(envelope, dtype=bool)
    for label_id in np.unique(clean_labels):
        if label_id <= 0:
            continue
        obj = clean_labels == label_id
        if float(np.nansum(area[obj])) >= ENVELOPE_MIN_RAW_AREA_KM2:
            cleaned[obj] = True
    return cleaned


def build_organised_convective_system_mask(fields, masks):
    if "olr" not in fields:
        raise RuntimeError(
            "Organised convective-system tracking requires OLR/olrtoa so "
            "PyFLEXTRKR can build a Tb-like cold-cloud shield"
        )
    olr = fields["olr"].sel(Time=fields["precip"].Time)
    olr, _precip = xr.align(
        finite_or_nan(olr),
        fields["precip"],
        join="inner",
    )
    olr = subset_lat(olr, lat_bounds=TROPICAL_LAT_BOUNDS)
    area2d = grid_cell_area_km2(olr.latitude.values, olr.longitude.values)
    envelope = xr.apply_ufunc(
        organised_envelope_scene,
        olr,
        area2d,
        input_core_dims=[
            ["latitude", "longitude"],
            ["latitude", "longitude"],
        ],
        output_core_dims=[["latitude", "longitude"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[bool],
    )
    envelope = envelope.assign_coords(
        Time=olr.Time,
        latitude=olr.latitude,
        longitude=olr.longitude,
    )
    envelope.name = ORGANISED_SYSTEM_SOURCE
    envelope.attrs.update(
        long_name="organised convective-system envelope mask",
        description=(
            "PyFLEXTRKR detect-and-spread cold-cloud-system labels generated "
            "from OLR-derived Tb. Convective, stratiform, "
            "non-precipitating, and cloud-top-pressure fields are sampled inside "
            "these objects but do not define their geometry."
        ),
        object_geometry=ORGANISED_TRACK_OBJECT_GEOMETRY,
        olr_to_tb_source="pyflextrkr.idclouds_tbpf.olr_to_tb",
        pyflextrkr_tb_core_k=float(PYFLEXTRKR_TB_CORE_K),
        pyflextrkr_tb_cold_k=float(PYFLEXTRKR_TB_COLD_K),
        pyflextrkr_tb_warm_k=float(PYFLEXTRKR_TB_WARM_K),
        pyflextrkr_tb_cloud_k=float(PYFLEXTRKR_TB_CLOUD_K),
        pyflextrkr_tb_smoothing_cells=int(PYFLEXTRKR_TB_SMOOTHING_CELLS),
        pyflextrkr_min_cold_core_pixels=int(PYFLEXTRKR_MIN_COLD_CORE_PIXELS),
        pyflextrkr_warm_anvil_expansion=int(PYFLEXTRKR_WARM_ANVIL_EXPANSION),
        min_raw_area_km2=float(ENVELOPE_MIN_RAW_AREA_KM2),
    )
    return envelope.astype(bool)


# -----------------------------------------------------------------------------
# Threshold masks and summaries
# -----------------------------------------------------------------------------


def build_threshold_table(fields_by_run):
    return pd.DataFrame(
        [
            dict(
                source=ORGANISED_SYSTEM_SOURCE,
                field="cloud_top_pressure",
                threshold="not used to define PyFLEXTRKR objects",
                threshold_type="diagnostic_attribute",
                percentile=np.nan,
                role="sampled inside tracked cold-cloud shields for interpretation",
                literature_basis="project high-cloud classification; retained for comparison with earlier diagnostics",
            ),
            dict(
                source=ORGANISED_SYSTEM_SOURCE,
                field="OLR-derived brightness temperature",
                threshold=pyflextrkr_tb_threshold_label(),
                threshold_type="PyFLEXTRKR_default_style",
                percentile=np.nan,
                role="labels cold cores and grows them into warmer anvil cloud using PyFLEXTRKR detect-and-spread",
                literature_basis="PyFLEXTRKR MCS workflow: OLR-to-Tb proxy plus cold-core/anvil detect-and-spread segmentation",
            ),
            dict(
                source=ORGANISED_SYSTEM_SOURCE,
                field="convective_precipitating_high_cloud_mask",
                threshold="accepted track must contain any convective-precipitating high cloud during its lifetime",
                threshold_type="classification_contract",
                percentile=np.nan,
                role="simple lifetime filter tying cold-cloud shields to active convection without tracking intermittent cores",
                literature_basis="PyFLEXTRKR-style cold-cloud shield tracking with convective-feature attributes sampled inside the shield",
            ),
            dict(
                source=ORGANISED_SYSTEM_SOURCE,
                field="precip",
                threshold="not used for object acceptance",
                threshold_type="diagnostic_attribute",
                percentile=np.nan,
                role="sampled inside tracked cold-cloud shields; no precip-coverage requirement is imposed",
                literature_basis="precipitation-feature screening is diagnostic here because the target is general cold-cloud/convection lifecycle, not only strict MCSs",
            ),
            dict(
                source=ORGANISED_SYSTEM_SOURCE,
                field="envelope_area",
                threshold=f"raw PyFLEXTRKR feature >= {ENVELOPE_MIN_RAW_AREA_KM2:g} km2",
                threshold_type="segmentation_minimum",
                percentile=np.nan,
                role="removes isolated cold-pixel fragments before tracking; not used to accept or reject tracks",
                literature_basis="PyFLEXTRKR cold-cloud feature detection requires a minimum feature area; track-size distributions are diagnostic only here",
            ),
            dict(
                source=ORGANISED_SYSTEM_SOURCE,
                field="duration",
                threshold=f">= {MIN_TRACK_DURATION_HOURS:g} h support ({MIN_TRACK_OBSERVATIONS} snapshots)",
                threshold_type="science_default",
                percentile=np.nan,
                role="organized-system persistence requirement",
                literature_basis="MCS duration filtering; 24 h preferred for robust 6-hourly lifecycle composites",
            ),
            dict(
                source=ORGANISED_SYSTEM_SOURCE,
                field="convective_precipitating_high_cloud",
                threshold="any positive convective-precipitating high-cloud area during track lifetime",
                threshold_type="classification_contract",
                percentile=np.nan,
                role="keeps accepted lifecycle objects connected to active convection without tuning an area-fraction threshold",
                literature_basis="simple classification attribute sampled within PyFLEXTRKR cold-cloud shields",
            ),
            dict(
                source=ORGANISED_SYSTEM_SOURCE,
                field="centroid_latitude",
                threshold="not used for track acceptance",
                threshold_type="diagnostic_attribute",
                percentile=np.nan,
                role="reported to show whether accepted tracks sit in the equatorial wave-coupling domain",
                literature_basis="source scenes are tropical; latitude is checked diagnostically rather than imposed as an additional track filter",
            ),
            dict(
                source=ORGANISED_SYSTEM_SOURCE,
                field="centroid_step_speed",
                threshold="not used for track acceptance",
                threshold_type="diagnostic_attribute",
                percentile=np.nan,
                role="reported to reveal discontinuous PyFLEXTRKR links; not used to remove tracks",
                literature_basis="centroid speed is shape-sensitive for broad evolving cloud shields, so it is a validation diagnostic rather than a hard filter",
            ),
        ]
    )


def threshold_object_masks(fields, threshold_table):
    return {}


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
    tracks_df = tracks_df[tracks_df.source == ORGANISED_SYSTEM_SOURCE]
    if len(tracks_df) == 0:
        print("Skipping track lifetime plot: no accepted organised-system tracks")
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
    fig.suptitle(
        "Track lifetime: organised convective-system envelopes",
        fontsize=12,
    )
    save_figure(6, fig, "track_lifetime_survival_boxplot")


def plot_long_lived_contribution(contribution_df):
    if len(contribution_df) == 0:
        print("Skipping long-lived contribution plot: no contribution table")
        return
    contribution_df = contribution_df[
        contribution_df.source == ORGANISED_SYSTEM_SOURCE
    ]
    if len(contribution_df) == 0:
        print("Skipping long-lived contribution plot: no accepted organised-system tracks")
        return
    sources = ordered_sources(contribution_df)
    fig, ax = source_axis_grid(sources, height_per_row=3.2)
    fig.set_size_inches(7.2, 4.2)
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
    fig.suptitle(
        "Organised convective-system tracks: area solid, precipitation dashed",
        fontsize=11,
    )
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
    tracks_df = tracks_df[tracks_df.source == ORGANISED_SYSTEM_SOURCE]
    if len(tracks_df) == 0:
        print(f"Skipping {stem}: no accepted organised-system tracks")
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


def pyflextrkr_cloudid_files(source_name, sim_name):
    work_path = pyflextrkr_work_path(source_name, sim_name)
    return sorted(glob.glob(os.path.join(work_path, "tracking", "cloudid_*.nc")))


def pyflextrkr_scene_track_grid(source_name, sim_name, time_index, labels):
    labels = np.asarray(labels, dtype=np.int32)
    track_grid = np.zeros(labels.shape, dtype=np.int32)
    work_path = pyflextrkr_work_path(source_name, sim_name)
    metadata = read_product_metadata(work_path)
    if metadata is None:
        return track_grid
    tracknumbers_name = metadata.get("tracknumbers_file")
    if not tracknumbers_name:
        return track_grid
    tracknumbers_file = os.path.join(work_path, "stats", tracknumbers_name)
    if not os.path.exists(tracknumbers_file):
        return track_grid
    with xr.open_dataset(tracknumbers_file, decode_times=False, mask_and_scale=False) as ds:
        track_numbers = np.asarray(ds["track_numbers"].squeeze().values)
    if track_numbers.ndim != 2 or time_index >= track_numbers.shape[0]:
        return track_grid
    max_label = int(labels.max()) if labels.size else 0
    if max_label <= 0:
        return track_grid
    label_to_track = np.zeros(max_label + 1, dtype=np.int32)
    scene_tracks = np.asarray(track_numbers[time_index], dtype=np.int32)
    count = min(max_label, scene_tracks.size)
    if count > 0:
        label_to_track[1:count + 1] = scene_tracks[:count]
    return label_to_track[np.clip(labels, 0, max_label)]


def pyflextrkr_scene_label_status(source_name, sim_name, time_index, labels):
    """
    Return per-grid-cell candidate status from PyFLEXTRKR manifest bookkeeping.

    Codes:
      0: background/no candidate
      1: accepted organised-system track
      2: rejected because it is short lived in this 6-hourly product
      3: rejected for another reason or not linked into a valid track
    """
    labels = np.asarray(labels, dtype=np.int32)
    status = np.zeros(labels.shape, dtype=np.int8)
    work_path = pyflextrkr_work_path(source_name, sim_name)
    manifest = read_track_manifest(work_path)
    metadata = read_product_metadata(work_path)
    if manifest is None or metadata is None:
        status[labels > 0] = 3
        return status, set(), set(), set()

    accepted_ids = accepted_track_ids(manifest)
    accepted = manifest_accepted_mask(manifest)
    reasons = manifest.rejection_reason.astype(str)
    short_lived = reasons.str.contains(
        f"fewer_than_{MIN_TRACK_OBSERVATIONS}_observations",
        regex=False,
    ) | reasons.str.contains("duration_below_threshold", regex=False)
    short_ids = set(manifest.loc[(~accepted) & short_lived, "track_id"].astype(int))
    rejected_ids = set(manifest.loc[~accepted, "track_id"].astype(int))

    track_grid = pyflextrkr_scene_track_grid(source_name, sim_name, time_index, labels)
    candidate = labels > 0
    status[candidate] = 3
    if accepted_ids:
        status[np.isin(track_grid, list(accepted_ids)) & candidate] = 1
    if short_ids:
        status[np.isin(track_grid, list(short_ids)) & candidate] = 2
    return status, accepted_ids, short_ids, rejected_ids


def overlay_status_mask(ax, lon, lat, mask, color, alpha):
    if not np.any(mask):
        return
    cmap = mpl_colors.ListedColormap([color])
    ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_where(~mask, np.ones(mask.shape, dtype=float)),
        cmap=cmap,
        shading="auto",
        alpha=alpha,
        zorder=4,
        rasterized=True,
    )


def pyflextrkr_track_color(track_id):
    cmap = plt.get_cmap("tab20")
    return cmap((int(track_id) - 1) % cmap.N)


def annotate_track_id(ax, lon_value, lat_value, track_id, color, fontsize=6.0):
    ax.annotate(
        str(int(track_id)),
        xy=(float(lon_value), float(lat_value)),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor=color, alpha=0.88, pad=0.45),
        zorder=12,
        clip_on=True,
    )


def plot_centroid_tracks_overview(
    ax,
    sim_objects,
    lon,
    lat,
    max_label_count=80,
    highlight_track_ids=None,
):
    highlight_track_ids = set() if highlight_track_ids is None else {
        int(track_id) for track_id in highlight_track_ids
    }
    if sim_objects is None or len(sim_objects) == 0:
        ax.text(
            0.5, 0.5, "No accepted tracks", transform=ax.transAxes,
            ha="center", va="center",
        )
        return
    ranked = (
        sim_objects.groupby("track_id")["area_km2"]
        .max()
        .sort_values(ascending=False)
    )
    labelled_ids = set(ranked.index[:max_label_count].astype(int))
    labelled_ids.update(highlight_track_ids)
    for i, (track_id, track) in enumerate(sim_objects.groupby("track_id")):
        track = (
            track.sort_values(["time_index", "area_km2"], ascending=[True, False])
            .drop_duplicates("time_index")
            .sort_values("time_index")
        )
        if len(track) == 0:
            continue
        x = np.asarray(track.centroid_lon, dtype=float)
        y = np.asarray(track.centroid_lat, dtype=float)
        jumps = np.where(np.abs(np.diff(x)) > 180.0)[0]
        x_plot = x.copy()
        y_plot = y.copy()
        for jump_index in jumps[::-1]:
            x_plot = np.insert(x_plot, jump_index + 1, np.nan)
            y_plot = np.insert(y_plot, jump_index + 1, np.nan)
        color = pyflextrkr_track_color(track_id)
        highlighted = int(track_id) in highlight_track_ids
        ax.plot(
            x_plot,
            y_plot,
            color=color,
            lw=1.55 if highlighted else 0.85,
            alpha=0.98 if highlighted else 0.55,
            zorder=8 if highlighted else 5,
        )
        ax.scatter(
            [x[-1]],
            [y[-1]],
            color=color,
            s=24 if highlighted else 12,
            zorder=9 if highlighted else 6,
        )
        if int(track_id) in labelled_ids:
            annotate_track_id(ax, x[-1], y[-1], track_id, color, fontsize=5.8)
    suffix = "" if len(ranked) <= max_label_count else f"; largest {max_label_count} labelled"
    highlight_note = (
        f"; selected snapshot tracks highlighted ({len(highlight_track_ids)})"
        if highlight_track_ids
        else ""
    )
    ax.set_title(
        "All accepted centroid tracks over "
        f"{sim_objects.time_index.nunique()} snapshots{highlight_note}; "
        f"labels are track IDs{suffix}"
    )


def plot_pyflextrkr_track_validation(objects_df, fields_by_run, masks_by_run):
    if not should_run_figure(11):
        return
    if objects_df is None or len(objects_df) == 0:
        objects_df = pd.DataFrame()
    object_sources = set(objects_df.source) if "source" in objects_df.columns else set()
    mask_sources = {
        source
        for masks in masks_by_run
        for source in PYFLEXTRKR_VALIDATION_SOURCES
        if source in masks
    }
    sources = [
        source
        for source in PYFLEXTRKR_VALIDATION_SOURCES
        if source in object_sources or source in mask_sources
    ]
    if not sources:
        print("Skipping Figure 11: no current PyFLEXTRKR source masks are available.")
        return

    validation_rows = []
    precip_cmap, precip_norm = precip_rate_cmap_norm()
    status_handles = [
        Patch(facecolor="#1a9850", alpha=0.52, label="Accepted organised systems"),
        Patch(facecolor="#fdae61", alpha=0.46, label="Rejected: short lived"),
        Patch(facecolor="#756bb1", alpha=0.42, label="Rejected: no convective-precipitating high cloud / unlinked"),
        Patch(
            facecolor="0.65",
            alpha=0.45,
            label="Accepted object fill uses matching track-ID colour",
        ),
        Line2D([0], [0], color="#c51b8a", lw=1.0, label="Convective-precipitating high cloud"),
        Line2D([0], [0], color="tab:blue", marker="o", markersize=3, lw=1.0, label="Accepted centroid path and ID"),
    ]
    for source_name in sources:
        fig, axes = plt.subplots(
            3 * len(sim_names),
            1,
            figsize=(15.0, max(7.4, 6.2 * len(sim_names))),
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes = axes.reshape(len(sim_names), 3)
        last_precip_im = None
        for c, sim_name in enumerate(sim_names):
            ax_precip = axes[c, 0]
            ax_ctp = axes[c, 1]
            ax_tracks = axes[c, 2]
            if {"source", "simulation"}.issubset(objects_df.columns):
                sub = objects_df[
                    (objects_df.source == source_name)
                    & (objects_df.simulation == sim_name)
                ].copy()
            else:
                sub = pd.DataFrame()
            files = pyflextrkr_cloudid_files(source_name, sim_name)
            if not files:
                for ax in (ax_precip, ax_ctp, ax_tracks):
                    ax.text(
                        0.5,
                        0.5,
                        "No current PyFLEXTRKR scene",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                continue
            if len(sub) and "time_index" in sub.columns:
                snapshot_metric = None
                for metric in (
                    "area_convective_precipitating_high_cloud_mask_km2",
                    "precip_area_rate",
                    "area_km2",
                ):
                    if metric in sub.columns:
                        snapshot_metric = metric
                        break
                if snapshot_metric is not None:
                    scores = sub.groupby("time_index")[snapshot_metric].sum()
                    time_index = int(scores.idxmax())
                    snapshot_selection = snapshot_metric
                else:
                    counts = sub.groupby("time_index").size()
                    time_index = int(counts.idxmax())
                    snapshot_selection = "accepted_label_count"
            else:
                time_index = 0
                snapshot_selection = "first_scene"
            if time_index >= len(files):
                for ax in (ax_precip, ax_ctp, ax_tracks):
                    ax.text(
                        0.5,
                        0.5,
                        "cloudid/time index mismatch",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                continue

            with xr.open_dataset(files[time_index], decode_times=False, mask_and_scale=False) as cloud:
                labels = np.asarray(cloud["feature_number"].squeeze().values, dtype=np.int32)
                lat = np.asarray(cloud["lat"].values, dtype=float)
                lon = np.asarray(cloud["lon"].values, dtype=float)
            precip = fields_by_run[c]["precip"].isel(Time=time_index)
            precip = precip.sel(
                latitude=xr.DataArray(lat, dims="latitude"),
                longitude=xr.DataArray(lon, dims="longitude"),
                method="nearest",
            ).load()
            last_precip_im = ax_precip.pcolormesh(
                lon,
                lat,
                precip.clip(min=0.0),
                cmap=precip_cmap,
                norm=precip_norm,
                shading="auto",
                alpha=0.82,
                rasterized=True,
            )
            source_mask = tracking_grid_field(
                masks_by_run[c][source_name], time_index, lat, lon
            ).astype(bool)
            source_values = np.asarray(source_mask.values, dtype=bool)
            core_values = None
            if "convective_precipitating_high_cloud_mask" in masks_by_run[c]:
                core_mask = tracking_grid_field(
                    masks_by_run[c]["convective_precipitating_high_cloud_mask"],
                    time_index,
                    lat,
                    lon,
                ).astype(bool)
                core_values = np.asarray(core_mask.values, dtype=bool)
            accepted_scene = (
                sub[sub.time_index == time_index].copy()
                if len(sub) and "time_index" in sub.columns
                else pd.DataFrame()
            )
            accepted_labels = (
                set(accepted_scene.label.to_numpy(dtype=int))
                if len(accepted_scene) and "label" in accepted_scene.columns
                else set()
            )
            status, _accepted_ids, short_ids, rejected_ids = pyflextrkr_scene_label_status(
                source_name, sim_name, time_index, labels
            )
            track_grid = pyflextrkr_scene_track_grid(
                source_name, sim_name, time_index, labels
            )
            accepted_values = (status == 1) | np.isin(labels, list(accepted_labels))
            short_values = (status == 2) & ~accepted_values
            other_rejected_values = (status == 3) & ~accepted_values
            candidate_values = labels > 0
            overlay_status_mask(ax_precip, lon, lat, other_rejected_values, "#756bb1", 0.42)
            overlay_status_mask(ax_precip, lon, lat, short_values, "#fdae61", 0.46)
            overlay_status_mask(ax_precip, lon, lat, accepted_values, "#1a9850", 0.52)
            if np.any(accepted_values):
                ax_precip.contour(
                    lon, lat, accepted_values.astype(float), levels=[0.5],
                    colors="0.05", linewidths=1.0, zorder=8,
                )
            if np.any(short_values):
                ax_precip.contour(
                    lon, lat, short_values.astype(float), levels=[0.5],
                    colors="#d95f02", linewidths=0.8, zorder=7,
                )
            if np.any(other_rejected_values):
                ax_precip.contour(
                    lon, lat, other_rejected_values.astype(float), levels=[0.5],
                    colors="#542788", linewidths=0.8, zorder=7,
                )
            if np.any(candidate_values):
                overlay_status_mask(
                    ax_ctp, lon, lat, other_rejected_values, "#756bb1", 0.34
                )
                overlay_status_mask(
                    ax_ctp, lon, lat, short_values, "#fdae61", 0.38
                )
                label_codes, label_cmap, label_norm = track_id_label_plot_data(
                    track_grid,
                    accepted_values,
                )
                ax_ctp.pcolormesh(
                    lon,
                    lat,
                    np.ma.masked_where(label_codes <= 0, label_codes),
                    cmap=label_cmap,
                    norm=label_norm,
                    shading="auto",
                    zorder=5,
                    rasterized=True,
                )
            if core_values is not None and np.any(core_values):
                for ax in (ax_precip, ax_ctp):
                    ax.contour(
                        lon,
                        lat,
                        core_values.astype(float),
                        levels=[0.5],
                        colors="#c51b8a",
                        linewidths=0.9,
                    )
            if np.any(accepted_values):
                ax_ctp.contour(
                    lon, lat, accepted_values.astype(float), levels=[0.5],
                    colors="0.05", linewidths=1.1,
                )

            source_cells = int(np.count_nonzero(source_values))
            accepted_cells = int(np.count_nonzero(accepted_values))
            accepted_source_cells = int(
                np.count_nonzero(source_values & accepted_values)
            )
            organised_source_percent = (
                100.0 * accepted_source_cells / source_cells
                if source_cells
                else np.nan
            )
            raw_candidate_label_count = int(np.unique(labels[candidate_values]).size)
            accepted_candidate_label_count = int(np.unique(labels[accepted_values]).size)
            short_lived_rejected_label_count = int(np.unique(labels[short_values]).size)
            other_rejected_label_count = int(np.unique(labels[other_rejected_values]).size)
            validation_rows.append(
                dict(
                    row_type="scene_summary",
                    simulation=sim_name,
                    source=source_name,
                    time_index=time_index,
                    Time=precip.Time.values,
                    object_label=np.nan,
                    track_id=np.nan,
                    area_km2=np.nan,
                    centroid_lat=np.nan,
                    centroid_lon=np.nan,
                    cloudid_file=files[time_index],
                    source_mask_cell_count=source_cells,
                    accepted_organised_cell_count=accepted_cells,
                    raw_candidate_label_count=raw_candidate_label_count,
                    accepted_candidate_label_count=accepted_candidate_label_count,
                    short_lived_rejected_label_count=short_lived_rejected_label_count,
                    other_rejected_label_count=other_rejected_label_count,
                    snapshot_selection=snapshot_selection,
                    source_mask_in_accepted_tracks_percent=organised_source_percent,
                )
            )

            scene = (
                accepted_scene.nlargest(12, "area_km2")
                if len(accepted_scene) and "area_km2" in accepted_scene.columns
                else pd.DataFrame()
            )
            selected_track_ids = (
                set(scene.track_id.astype(int))
                if len(scene) and "track_id" in scene.columns
                else set()
            )
            if len(scene) and {"track_id", "time_index"}.issubset(sub.columns):
                history = sub[
                    sub.track_id.isin(selected_track_ids)
                    & (sub.time_index >= time_index - 4)
                    & (sub.time_index <= time_index)
                ]
                for track_id, track in history.groupby("track_id"):
                    track = track.sort_values("time_index")
                    x = np.asarray(track.centroid_lon, dtype=float)
                    y = np.asarray(track.centroid_lat, dtype=float)
                    jumps = np.abs(np.diff(x)) > 180.0
                    x_plot = x.astype(float)
                    y_plot = y.astype(float)
                    for jump_index in np.where(jumps)[0]:
                        x_plot = np.insert(x_plot, jump_index + 1, np.nan)
                        y_plot = np.insert(y_plot, jump_index + 1, np.nan)
                    for ax in (ax_precip, ax_ctp):
                        color = pyflextrkr_track_color(track_id)
                        ax.plot(x_plot, y_plot, color=color, lw=1.05, alpha=0.9, zorder=9)
                        ax.scatter(
                            [float(track.centroid_lon.iloc[-1])],
                            [float(track.centroid_lat.iloc[-1])],
                            color=color, s=16, zorder=10,
                        )
            plot_centroid_tracks_overview(
                ax_tracks,
                sub,
                lon,
                lat,
                highlight_track_ids=selected_track_ids,
            )
            for record in scene.itertuples(index=False):
                color = pyflextrkr_track_color(record.track_id)
                for label_ax in (ax_precip, ax_ctp):
                    annotate_track_id(
                        label_ax,
                        record.centroid_lon,
                        record.centroid_lat,
                        record.track_id,
                        color,
                        fontsize=6.2,
                    )
                validation_rows.append(
                    dict(
                        row_type="accepted_object",
                        simulation=sim_name,
                        source=source_name,
                        time_index=time_index,
                        Time=record.Time,
                        object_label=int(record.label),
                        track_id=int(record.track_id),
                        area_km2=float(record.area_km2),
                        centroid_lat=float(record.centroid_lat),
                        centroid_lon=float(record.centroid_lon),
                        cloudid_file=files[time_index],
                        source_mask_cell_count=source_cells,
                        accepted_organised_cell_count=accepted_cells,
                        raw_candidate_label_count=raw_candidate_label_count,
                        accepted_candidate_label_count=accepted_candidate_label_count,
                        short_lived_rejected_label_count=short_lived_rejected_label_count,
                        other_rejected_label_count=other_rejected_label_count,
                        snapshot_selection=snapshot_selection,
                        source_mask_in_accepted_tracks_percent=organised_source_percent,
                    )
                )
            for ax in (ax_precip, ax_ctp, ax_tracks):
                ax.set_xlim(float(lon.min()), float(lon.max()))
                ax.set_ylim(*TROPICAL_LAT_BOUNDS)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("Longitude")
                ax.grid(alpha=0.12, lw=0.4)
            ax_precip.set_ylabel("Latitude")
            ax_ctp.set_ylabel("Latitude")
            ax_tracks.set_ylabel("Latitude")
            ax_precip.set_title(
                f"{sim_name}, {pd.to_datetime(precip.Time.values):%Y-%m-%d %H:%M}: precip + candidate status\n"
                f"accepted {accepted_candidate_label_count}; "
                f"short-lived {short_lived_rejected_label_count}; other rejected {other_rejected_label_count}"
            )
            ax_ctp.set_title(
                "PyFLEXTRKR candidates: accepted objects use track-ID colours\n"
                f"{raw_candidate_label_count} candidates; Tb core/cold/warm <= "
                f"{PYFLEXTRKR_TB_CORE_K:g}/{PYFLEXTRKR_TB_COLD_K:g}/{PYFLEXTRKR_TB_WARM_K:g} K"
            )

        if last_precip_im is not None:
            fig.colorbar(
                last_precip_im,
                ax=axes[:, 0].ravel().tolist(),
                label=r"Precipitation (mm day$^{-1}$)",
                shrink=0.86,
            )
        fig.suptitle(
            "PyFLEXTRKR cold-cloud candidates and accepted lifecycle tracks: "
            f">= {MIN_TRACK_DURATION_HOURS:g} h support plus embedded convective-precipitating high cloud",
            fontsize=12,
        )
        fig.legend(
            handles=status_handles,
            loc="outside lower center",
            ncol=3,
            frameon=True,
            fontsize=7,
        )
        save_figure(
            11,
            fig,
            f"pyflextrkr_track_validation_{source_name}",
        )
        plt.close(fig)
    pd.DataFrame(validation_rows).to_csv(
        os.path.join(save_loc, "pyflextrkr_cloud_object_track_validation.csv"),
        index=False,
    )


def pyflextrkr_track_step_table(objects_df):
    rows = []
    if objects_df is None or len(objects_df) == 0:
        return pd.DataFrame()
    for (source, simulation, track_id), group in objects_df.groupby(
        ["source", "simulation", "track_id"]
    ):
        # Merge/split bookkeeping can leave more than one branch at a time;
        # validate continuity on the largest branch used for visual track paths.
        group = (
            group.sort_values(["time_index", "area_km2"], ascending=[True, False])
            .drop_duplicates("time_index")
            .sort_values("time_index")
        )
        if len(group) < 2:
            continue
        previous = group.iloc[:-1]
        current = group.iloc[1:]
        distance = haversine_distances_km(
            previous.centroid_lat.to_numpy(),
            previous.centroid_lon.to_numpy(),
            current.centroid_lat.to_numpy(),
            current.centroid_lon.to_numpy(),
        )
        time_delta_hours = (
            pd.to_datetime(current.Time.to_numpy())
            - pd.to_datetime(previous.Time.to_numpy())
        ) / pd.Timedelta(hours=1)
        time_delta_hours = np.asarray(time_delta_hours, dtype=float)
        area_ratio = np.maximum(
            current.area_km2.to_numpy() / previous.area_km2.to_numpy(),
            previous.area_km2.to_numpy() / current.area_km2.to_numpy(),
        )
        for index, (dist, hours, ratio) in enumerate(
            zip(np.asarray(distance), time_delta_hours, area_ratio)
        ):
            rows.append(
                dict(
                    source=source,
                    source_label=source_label(source),
                    simulation=simulation,
                    track_id=int(track_id),
                    from_time_index=int(previous.time_index.iloc[index]),
                    to_time_index=int(current.time_index.iloc[index]),
                    time_delta_hours=float(hours),
                    centroid_distance_km=float(dist),
                    translation_speed_km_h=float(dist / hours) if hours > 0 else np.nan,
                    area_change_factor=float(ratio),
                )
            )
    return pd.DataFrame(rows)


def pyflextrkr_validation_inventory(
    objects_df,
    tracks_df,
    masks_by_run,
):
    steps = pyflextrkr_track_step_table(objects_df)
    rows = []
    for sim_index, sim_name in enumerate(sim_names):
        masks = masks_by_run[sim_index]
        sources = [source for source in PYFLEXTRKR_TRACKED_SOURCES if source in masks]
        for source in sources:
            mask = masks[source]
            expected_scenes = int(mask.sizes.get("Time", 0))
            files = pyflextrkr_cloudid_files(source, sim_name)
            objects = objects_df[
                (objects_df.simulation == sim_name) & (objects_df.source == source)
            ] if len(objects_df) else pd.DataFrame()
            tracks = tracks_df[
                (tracks_df.simulation == sim_name) & (tracks_df.source == source)
            ] if len(tracks_df) else pd.DataFrame()
            source_steps = steps[
                (steps.simulation == sim_name) & (steps.source == source)
            ] if len(steps) else pd.DataFrame()

            issues = []
            if len(files) != expected_scenes:
                issues.append(
                    f"cloudid files {len(files)} != expected scenes {expected_scenes}"
                )
            if len(objects) == 0:
                issues.append("no tracked objects")
            if len(tracks) == 0:
                issues.append("no tracks")
            invalid_time_indices = 0
            if len(objects) and expected_scenes > 0:
                invalid_time_indices = int(
                    np.count_nonzero(
                        (objects.time_index.to_numpy(dtype=int) < 0)
                        | (objects.time_index.to_numpy(dtype=int) >= expected_scenes)
                    )
                )
                if invalid_time_indices:
                    issues.append(f"{invalid_time_indices} invalid object time indices")
            multistep_tracks = int(
                np.count_nonzero(tracks.lifetime_steps.to_numpy(dtype=float) > 1)
            ) if len(tracks) and "lifetime_steps" in tracks else 0
            status = "FAIL" if issues else ("WARN" if multistep_tracks == 0 else "PASS")
            if not issues and multistep_tracks == 0:
                issues.append("no multi-snapshot tracks")

            object_counts = (
                objects.groupby("time_index").size().to_numpy(dtype=float)
                if len(objects)
                else np.array([], dtype=float)
            )
            lifetimes = (
                tracks.lifetime_hours.to_numpy(dtype=float)
                if len(tracks) and "lifetime_hours" in tracks
                else np.array([], dtype=float)
            )
            speeds = (
                source_steps.translation_speed_km_h.to_numpy(dtype=float)
                if len(source_steps)
                else np.array([], dtype=float)
            )
            area_changes = (
                source_steps.area_change_factor.to_numpy(dtype=float)
                if len(source_steps)
                else np.array([], dtype=float)
            )
            rows.append(
                dict(
                    simulation=sim_name,
                    source=source,
                    source_label=source_label(source),
                    analysis_role=track_source_role(source),
                    validation_status=status,
                    issue="; ".join(issues) if issues else "none",
                    expected_scene_count=expected_scenes,
                    cloudid_file_count=len(files),
                    cloudid_scene_coverage_percent=(
                        100.0 * len(files) / expected_scenes
                        if expected_scenes > 0
                        else np.nan
                    ),
                    scenes_with_tracked_objects=(
                        int(objects.time_index.nunique()) if len(objects) else 0
                    ),
                    tracked_object_count=int(len(objects)),
                    track_count=int(len(tracks)),
                    multistep_track_count=multistep_tracks,
                    median_objects_per_nonempty_scene=(
                        float(np.nanmedian(object_counts)) if object_counts.size else np.nan
                    ),
                    p95_objects_per_nonempty_scene=(
                        float(np.nanpercentile(object_counts, 95)) if object_counts.size else np.nan
                    ),
                    median_track_lifetime_hours=(
                        float(np.nanmedian(lifetimes)) if lifetimes.size else np.nan
                    ),
                    p95_track_lifetime_hours=(
                        float(np.nanpercentile(lifetimes, 95)) if lifetimes.size else np.nan
                    ),
                    tracks_starting_as_split=(
                        int(np.count_nonzero(tracks.starts_as_split))
                        if len(tracks) and "starts_as_split" in tracks
                        else 0
                    ),
                    tracks_ending_as_merge=(
                        int(np.count_nonzero(tracks.ends_as_merge))
                        if len(tracks) and "ends_as_merge" in tracks
                        else 0
                    ),
                    median_translation_speed_km_h=(
                        float(np.nanmedian(speeds)) if speeds.size else np.nan
                    ),
                    p95_translation_speed_km_h=(
                        float(np.nanpercentile(speeds, 95)) if speeds.size else np.nan
                    ),
                    p95_area_change_factor=(
                        float(np.nanpercentile(area_changes, 95))
                        if area_changes.size
                        else np.nan
                    ),
                    invalid_object_time_index_count=invalid_time_indices,
                )
            )
    return pd.DataFrame(rows)


def plot_pyflextrkr_validation_dashboard(
    objects_df,
    tracks_df,
    fields_by_run,
    masks_by_run,
):
    if not should_run_figure(14):
        return
    inventory = pyflextrkr_validation_inventory(
        objects_df, tracks_df, masks_by_run
    )
    inventory.to_csv(
        os.path.join(save_loc, "pyflextrkr_validation_inventory.csv"), index=False
    )
    steps = pyflextrkr_track_step_table(objects_df)

    for sim_index, sim_name in enumerate(sim_names):
        sources = [
            source
            for source in PYFLEXTRKR_TRACKED_SOURCES
            if source in masks_by_run[sim_index]
        ]
        colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, max(1, len(sources))))
        fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.6), constrained_layout=True)
        for source, color in zip(sources, colors):
            objects = objects_df[
                (objects_df.simulation == sim_name) & (objects_df.source == source)
            ] if len(objects_df) else pd.DataFrame()
            if len(objects):
                index = np.arange(int(masks_by_run[sim_index][source].sizes["Time"]))
                object_count = objects.groupby("time_index").size().reindex(index, fill_value=0)
                active_tracks = (
                    objects.groupby("time_index").track_id.nunique().reindex(index, fill_value=0)
                )
                axes[0, 0].plot(index, object_count, color=color, lw=1.5, label=source_label(source))
                axes[0, 1].plot(index, active_tracks, color=color, lw=1.5, label=source_label(source))

        lifetime_data = []
        speed_data = []
        labels = []
        box_colors = []
        for source, color in zip(sources, colors):
            tracks = tracks_df[
                (tracks_df.simulation == sim_name) & (tracks_df.source == source)
            ] if len(tracks_df) else pd.DataFrame()
            source_steps = steps[
                (steps.simulation == sim_name) & (steps.source == source)
            ] if len(steps) else pd.DataFrame()
            lifetime = (
                tracks.lifetime_hours.to_numpy(dtype=float)
                if len(tracks) and "lifetime_hours" in tracks
                else np.array([], dtype=float)
            )
            speed = (
                source_steps.translation_speed_km_h.to_numpy(dtype=float)
                if len(source_steps)
                else np.array([], dtype=float)
            )
            if lifetime.size or speed.size:
                labels.append(source_label(source))
                lifetime_data.append(lifetime[np.isfinite(lifetime)] if lifetime.size else np.array([np.nan]))
                speed_data.append(speed[np.isfinite(speed)] if speed.size else np.array([np.nan]))
                box_colors.append(color)

        for axis, values, title, xlabel in (
            (axes[1, 0], lifetime_data, "Track lifetime", "Lifetime (hours)"),
            (
                axes[1, 1],
                speed_data,
                "Consecutive-snapshot centroid translation speed",
                "Translation speed (km h$^{-1}$)",
            ),
        ):
            if values:
                boxes = axis.boxplot(
                    values,
                    vert=False,
                    tick_labels=labels,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch, color in zip(boxes["boxes"], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.65)
            else:
                axis.text(0.5, 0.5, "No values", transform=axis.transAxes, ha="center", va="center")
            axis.set_title(title)
            axis.set_xlabel(xlabel)
            axis.grid(axis="x", alpha=0.2)

        axes[0, 0].set_title("Tracked objects per snapshot")
        axes[0, 1].set_title("Active track IDs per snapshot")
        for axis in axes[0]:
            axis.set_xlabel("Snapshot index (6-hour spacing)")
            axis.set_ylabel("Count")
            axis.grid(alpha=0.2)
        axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
        sim_inventory = inventory[inventory.simulation == sim_name]
        status_counts = sim_inventory.validation_status.value_counts().to_dict()
        fig.suptitle(
            f"PyFLEXTRKR validation overview: {sim_name} | "
            f"PASS {status_counts.get('PASS', 0)}, WARN {status_counts.get('WARN', 0)}, "
            f"FAIL {status_counts.get('FAIL', 0)}"
        )
        safe_sim = sim_name.replace(" ", "_")
        save_figure(14, fig, f"pyflextrkr_validation_dashboard_{safe_sim}")
        plt.close(fig)


def plot_pyflextrkr_track_step_validation(objects_df):
    if not should_run_figure(13):
        return
    steps = pyflextrkr_track_step_table(objects_df)
    if len(steps) == 0:
        print("Skipping accepted-track step-speed plot: no accepted consecutive PyFLEXTRKR steps")
    else:
        steps.to_csv(os.path.join(save_loc, "pyflextrkr_track_step_validation.csv"), index=False)
        summaries = []
        for (source, simulation), group in steps.groupby(["source", "simulation"]):
            values = group.translation_speed_km_h.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            summaries.append(
                dict(
                    source=source,
                    source_label=source_label(source),
                    simulation=simulation,
                    step_count=int(values.size),
                    median_speed_km_h=float(np.nanmedian(values)) if values.size else np.nan,
                    p95_speed_km_h=float(np.nanpercentile(values, 95)) if values.size else np.nan,
                    max_speed_km_h=float(np.nanmax(values)) if values.size else np.nan,
                    p95_area_change_factor=float(np.nanpercentile(group.area_change_factor, 95)),
                )
            )
        pd.DataFrame(summaries).to_csv(
            os.path.join(save_loc, "pyflextrkr_track_step_validation_summary.csv"), index=False
        )

        sources = [source for source in PYFLEXTRKR_VALIDATION_SOURCES if source in set(steps.source)]
        for source in sources:
            fig, axes = plt.subplots(
                1, len(sim_names), figsize=(13.5, 3.7), constrained_layout=True,
                sharex=True, sharey=True, squeeze=False,
            )
            for i, simulation in enumerate(sim_names):
                ax = axes[0, i]
                values = steps[
                    (steps.source == source) & (steps.simulation == simulation)
                ].translation_speed_km_h.to_numpy(dtype=float)
                values = np.sort(values[np.isfinite(values)])
                if values.size:
                    y = np.arange(1, values.size + 1) / values.size
                    ax.plot(values, y, color=cols[i], lw=1.7)
                    ax.text(
                        0.98, 0.05,
                        f"median {np.nanmedian(values):.1f}\np95 {np.nanpercentile(values, 95):.1f} km h$^{{-1}}$",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    )
                ax.set_title(simulation)
                ax.set_xlabel("Centroid translation speed (km h$^{-1}$)")
                ax.grid(alpha=0.2)
            axes[0, 0].set_ylabel("Cumulative fraction of track steps")
            fig.suptitle("Accepted organised-track consecutive-snapshot centroid speeds")
            save_figure(13, fig, f"pyflextrkr_track_step_continuity_{source}")
            plt.close(fig)

    manifests = []
    for simulation in sim_names:
        work_path = pyflextrkr_work_path(ORGANISED_SYSTEM_SOURCE, simulation)
        manifest = read_track_manifest(work_path)
        if manifest is not None:
            manifests.append(manifest)
    if not manifests:
        return
    manifest = pd.concat(manifests, ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), constrained_layout=True)
    x = np.arange(len(sim_names))
    bottoms = np.zeros(len(sim_names), dtype=float)
    categories = (
        ("accepted", "Accepted", "tab:green"),
        ("single", f"< {MIN_TRACK_OBSERVATIONS} observations", "0.55"),
        ("duration", f"< {MIN_TRACK_DURATION_HOURS:g} h", "0.70"),
        ("no_core", "no convective-precipitating high cloud", "tab:blue"),
        ("other", "Other rejected/unlinked", "tab:orange"),
    )
    for category, label, color in categories:
        counts = []
        for simulation in sim_names:
            sub = manifest[manifest.simulation == simulation]
            accepted = manifest_accepted_mask(sub)
            reasons = sub.rejection_reason.astype(str)
            single_rejection = reasons.str.contains(
                f"fewer_than_{MIN_TRACK_OBSERVATIONS}_observations",
                regex=False,
            )
            duration_rejection = reasons.str.contains(
                "duration_below_threshold", regex=False
            )
            no_core_rejection = reasons.str.contains("no_embedded_convective_core", regex=False)
            if category == "accepted":
                mask = accepted
            elif category == "single":
                mask = (~accepted) & single_rejection
            elif category == "duration":
                mask = (~accepted) & ~single_rejection & duration_rejection
            elif category == "no_core":
                mask = (
                    (~accepted)
                    & ~single_rejection
                    & ~duration_rejection
                    & no_core_rejection
                )
            else:
                mask = (
                    (~accepted)
                    & ~single_rejection
                    & ~duration_rejection
                    & ~no_core_rejection
                )
            counts.append(int(mask.sum()))
        axes[0].bar(x, counts, bottom=bottoms, color=color, label=label)
        bottoms += np.asarray(counts, dtype=float)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sim_names)
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=0.8)
    axes[0].set_ylabel("Raw PyFLEXTRKR track count (log scale)")
    axes[0].set_title("Track acceptance")
    axes[0].legend(frameon=False, fontsize=8)

    for i, simulation in enumerate(sim_names):
        values = manifest.loc[
            manifest.simulation == simulation, "max_area_km2"
        ].to_numpy(dtype=float)
        values = np.sort(values[np.isfinite(values) & (values > 0.0)])
        if values.size:
            axes[1].plot(
                values,
                np.arange(1, values.size + 1) / values.size,
                color=cols[i], lw=1.7, label=simulation,
            )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Raw-track maximum area (km$^2$)")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title("Track-size diagnostic")
    axes[1].legend(frameon=False, fontsize=8)

    for i, simulation in enumerate(sim_names):
        values = manifest.loc[
            manifest.simulation == simulation, "max_translation_speed_kmh"
        ].to_numpy(dtype=float)
        values = np.sort(values[np.isfinite(values)])
        if values.size:
            axes[2].plot(
                values,
                np.arange(1, values.size + 1) / values.size,
                color=cols[i], lw=1.7, label=simulation,
            )
    axes[2].set_xlabel("Raw-track maximum step speed (km h$^{-1}$)")
    axes[2].set_ylabel("Cumulative fraction")
    axes[2].set_title("Centroid-speed diagnostic")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"Acceptance criteria: >= {MIN_TRACK_DURATION_HOURS:g} h; "
        "any embedded convective-precipitating high cloud; "
        "area and speed shown as diagnostics only",
        fontsize=11,
    )
    save_figure(13, fig, "pyflextrkr_organised_track_acceptance_qc")
    plt.close(fig)


def pyflextrkr_tb_class_from_labels(labels, olr_values):
    tb = olr_to_pyflextrkr_tb(olr_values)
    labels = np.asarray(labels, dtype=np.int32)
    code = np.zeros(labels.shape, dtype=np.int8)
    candidate = (labels > 0) & np.isfinite(tb)
    code[candidate & (tb <= PYFLEXTRKR_TB_CORE_K)] = 1
    code[
        candidate
        & (tb > PYFLEXTRKR_TB_CORE_K)
        & (tb <= PYFLEXTRKR_TB_COLD_K)
    ] = 2
    code[
        candidate
        & (tb > PYFLEXTRKR_TB_COLD_K)
        & (tb <= PYFLEXTRKR_TB_WARM_K)
    ] = 3
    return code, tb


def classification_pyflextrkr_overlap_table(class_code, pyflex_code, area2d):
    class_code = np.asarray(class_code)
    pyflex_code = np.asarray(pyflex_code)
    area = np.asarray(area2d, dtype=float)
    rows = []
    classification_labels = {
        1: "convective precipitating",
        2: "stratiform precipitating",
        3: "non-precipitating",
        4: "unclassified",
    }
    pyflex_labels = {
        1: f"Tb core <= {PYFLEXTRKR_TB_CORE_K:g} K",
        2: f"cold shield <= {PYFLEXTRKR_TB_COLD_K:g} K",
        3: f"warm shield <= {PYFLEXTRKR_TB_WARM_K:g} K",
    }
    total_area = float(np.nansum(area[class_code > 0]))
    for class_id, class_label in classification_labels.items():
        class_mask = class_code == class_id
        class_area = float(np.nansum(area[class_mask]))
        for pyflex_id, pyflex_label in pyflex_labels.items():
            overlap_area = float(np.nansum(area[class_mask & (pyflex_code == pyflex_id)]))
            rows.append(
                dict(
                    classification_id=class_id,
                    classification_label=class_label,
                    pyflextrkr_id=pyflex_id,
                    pyflextrkr_label=pyflex_label,
                    overlap_area_km2=overlap_area,
                    percent_of_classification_area=(
                        100.0 * overlap_area / class_area if class_area > 0 else np.nan
                    ),
                    percent_of_total_classification_area=(
                        100.0 * overlap_area / total_area if total_area > 0 else np.nan
                    ),
                )
            )
    return pd.DataFrame(rows)


def plot_classification_pyflextrkr_comparison(objects_df, fields_by_run, masks_by_run):
    if not should_run_figure(15):
        return

    class_cmap, class_norm = high_cloud_class_cmap_norm(alpha=0.86)
    class_handles = high_cloud_class_legend_handles()
    pyflex_colors = ["white", "#b2182b", "#fdae61", "#67a9cf"]
    pyflex_cmap = mpl_colors.ListedColormap(pyflex_colors)
    pyflex_norm = mpl_colors.BoundaryNorm(np.arange(-0.5, 4.5, 1.0), pyflex_cmap.N)
    pyflex_handles = [
        Patch(facecolor=pyflex_colors[1], label=f"PyFLEXTRKR Tb core <= {PYFLEXTRKR_TB_CORE_K:g} K"),
        Patch(facecolor=pyflex_colors[2], label=f"PyFLEXTRKR cold shield <= {PYFLEXTRKR_TB_COLD_K:g} K"),
        Patch(facecolor=pyflex_colors[3], label=f"PyFLEXTRKR warm shield <= {PYFLEXTRKR_TB_WARM_K:g} K"),
    ]

    fig, axes = plt.subplots(
        3 * len(sim_names),
        1,
        figsize=(15.0, max(8.5, 7.2 * len(sim_names))),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.reshape(len(sim_names), 3)
    overlap_rows = []

    for c, sim_name in enumerate(sim_names):
        files = pyflextrkr_cloudid_files(ORGANISED_SYSTEM_SOURCE, sim_name)
        if not files:
            for ax in axes[c]:
                ax.text(0.5, 0.5, "No current PyFLEXTRKR scene", transform=ax.transAxes, ha="center", va="center")
            continue

        if objects_df is not None and len(objects_df) and {"source", "simulation"}.issubset(objects_df.columns):
            sub = objects_df[
                (objects_df.source == ORGANISED_SYSTEM_SOURCE)
                & (objects_df.simulation == sim_name)
            ].copy()
        else:
            sub = pd.DataFrame()
        if len(sub) and "area_convective_precipitating_high_cloud_mask_km2" in sub.columns:
            scores = sub.groupby("time_index")["area_convective_precipitating_high_cloud_mask_km2"].sum()
            time_index = int(scores.idxmax())
        elif len(sub) and "time_index" in sub.columns:
            time_index = int(sub.groupby("time_index").size().idxmax())
        else:
            candidate_indices = np.linspace(
                0,
                len(files) - 1,
                num=min(len(files), PYFLEXTRKR_MOVIE_MAX_FRAMES),
                dtype=int,
            )
            best_score = -1
            time_index = 0
            for candidate_index in np.unique(candidate_indices):
                with xr.open_dataset(
                    files[int(candidate_index)],
                    decode_times=False,
                    mask_and_scale=False,
                ) as candidate_cloud:
                    candidate_labels = np.asarray(
                        candidate_cloud["feature_number"].squeeze().values,
                        dtype=np.int32,
                    )
                candidate_status, *_ = pyflextrkr_scene_label_status(
                    ORGANISED_SYSTEM_SOURCE,
                    sim_name,
                    int(candidate_index),
                    candidate_labels,
                )
                score = int(np.count_nonzero(candidate_status == 1))
                if score > best_score:
                    best_score = score
                    time_index = int(candidate_index)
        time_index = min(time_index, len(files) - 1)

        with xr.open_dataset(files[time_index], decode_times=False, mask_and_scale=False) as cloud:
            labels = np.asarray(cloud["feature_number"].squeeze().values, dtype=np.int32)
            lat = np.asarray(cloud["lat"].values, dtype=float)
            lon = np.asarray(cloud["lon"].values, dtype=float)

        class_code = cloud_class_code_for_movie(masks_by_run[c], time_index, lat, lon)
        olr = tracking_grid_field(fields_by_run[c]["olr"], time_index, lat, lon)
        pyflex_code, _tb = pyflextrkr_tb_class_from_labels(labels, np.asarray(olr, dtype=float))
        status, _accepted_ids, _short_ids, _rejected_ids = pyflextrkr_scene_label_status(
            ORGANISED_SYSTEM_SOURCE, sim_name, time_index, labels
        )
        accepted_values = status == 1
        area2d = grid_cell_area_km2(lat, lon)
        overlap = classification_pyflextrkr_overlap_table(class_code, pyflex_code, area2d)
        overlap.insert(0, "simulation", sim_name)
        overlap.insert(1, "time_index", time_index)
        overlap.insert(2, "Time", fields_by_run[c]["precip"].Time.values[time_index])
        overlap_rows.append(overlap)

        ax_class, ax_pyflex, ax_matrix = axes[c]
        ax_class.pcolormesh(
            lon,
            lat,
            np.ma.masked_where(class_code <= 0, class_code),
            cmap=class_cmap,
            norm=class_norm,
            shading="auto",
            rasterized=True,
        )
        ax_pyflex.pcolormesh(
            lon,
            lat,
            np.ma.masked_where(pyflex_code <= 0, pyflex_code),
            cmap=pyflex_cmap,
            norm=pyflex_norm,
            shading="auto",
            rasterized=True,
        )
        if np.any(accepted_values):
            for ax in (ax_class, ax_pyflex):
                ax.contour(
                    lon,
                    lat,
                    accepted_values.astype(float),
                    levels=[0.5],
                    colors="k",
                    linewidths=0.8,
                )
        for ax in (ax_class, ax_pyflex):
            ax.set_xlim(float(lon.min()), float(lon.max()))
            ax.set_ylim(*TROPICAL_LAT_BOUNDS)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(alpha=0.12, lw=0.4)

        ax_class.set_title(
            f"{sim_name}, {pd.to_datetime(fields_by_run[c]['precip'].Time.values[time_index]):%Y-%m-%d %H:%M}\n"
            "Shared precip-regime high-cloud classification"
        )
        ax_pyflex.set_title("PyFLEXTRKR Tb core/shield classes; black: accepted tracks")

        matrix = overlap.pivot(
            index="classification_label",
            columns="pyflextrkr_label",
            values="percent_of_classification_area",
        ).reindex(
            index=[
                "convective precipitating",
                "stratiform precipitating",
                "non-precipitating",
                "unclassified",
            ],
            columns=[
                f"Tb core <= {PYFLEXTRKR_TB_CORE_K:g} K",
                f"cold shield <= {PYFLEXTRKR_TB_COLD_K:g} K",
                f"warm shield <= {PYFLEXTRKR_TB_WARM_K:g} K",
            ],
        )
        im = ax_matrix.imshow(matrix.to_numpy(dtype=float), vmin=0.0, vmax=100.0, cmap="viridis")
        ax_matrix.set_xticks(np.arange(matrix.shape[1]))
        ax_matrix.set_xticklabels(["Tb core", "cold shield", "warm shield"], rotation=30, ha="right")
        ax_matrix.set_yticks(np.arange(matrix.shape[0]))
        ax_matrix.set_yticklabels(["convective", "stratiform", "non-precip.", "unclassified"])
        ax_matrix.set_title("Area overlap: % of each classification class")
        for iy in range(matrix.shape[0]):
            for ix in range(matrix.shape[1]):
                value = matrix.to_numpy(dtype=float)[iy, ix]
                if np.isfinite(value):
                    ax_matrix.text(ix, iy, f"{value:.0f}", ha="center", va="center", color="white" if value > 50 else "black", fontsize=8)
        fig.colorbar(im, ax=ax_matrix, label="% of classification class area", shrink=0.82)

    if overlap_rows:
        pd.concat(overlap_rows, ignore_index=True).to_csv(
            os.path.join(save_loc, "pyflextrkr_classification_overlap_snapshot.csv"),
            index=False,
        )
    fig.legend(
        handles=class_handles + pyflex_handles,
        loc="outside lower center",
        ncol=3,
        frameon=True,
        fontsize=7,
    )
    fig.suptitle(
        "Shared high-cloud classification versus PyFLEXTRKR Tb core/shield labels",
        fontsize=12,
    )
    save_figure(15, fig, "pyflextrkr_classification_comparison")
    plt.close(fig)


def tracking_grid_field(da, time_index, lat, lon):
    field = da.isel(Time=time_index) if "Time" in da.dims else da
    return field.sel(
        latitude=xr.DataArray(lat, dims="latitude"),
        longitude=xr.DataArray(lon, dims="longitude"),
        method="nearest",
    ).load()


def cloud_class_code_for_movie(masks, time_index, lat, lon):
    template = tracking_grid_field(masks["high_cloud_mask"], time_index, lat, lon)
    code = np.zeros(template.shape, dtype=np.int8)
    class_specs = (
        ("convective_precipitating_high_cloud_mask", 1),
        ("stratiform_precipitating_high_cloud_mask", 2),
        ("nonprecipitating_high_cloud_mask", 3),
        ("unclassified_high_cloud_mask", 4),
    )
    for key, value in class_specs:
        if key in masks:
            mask = tracking_grid_field(masks[key], time_index, lat, lon)
            code[np.asarray(mask, dtype=bool)] = value
    return code


def draw_pyflextrkr_movie_object_panel(
    ax,
    source_name,
    sim_objects,
    cloudid_file,
    precip_values,
    time_index,
    lat,
    lon,
):
    with xr.open_dataset(cloudid_file, decode_times=False, mask_and_scale=False) as cloud:
        labels = np.asarray(cloud["feature_number"].squeeze().values, dtype=np.int32)
    source_objects = sim_objects[sim_objects.source == source_name]
    all_scene_objects = source_objects[source_objects.time_index == time_index]
    accepted_labels = set(all_scene_objects.label.to_numpy(dtype=int))
    accepted_footprint = np.isin(labels, list(accepted_labels))
    precip_cmap, precip_norm = precip_rate_cmap_norm()
    ax.pcolormesh(
        lon,
        lat,
        np.maximum(precip_values, 0.0),
        cmap=precip_cmap,
        norm=precip_norm,
        shading="auto",
        alpha=0.68,
        rasterized=True,
    )
    ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_where(labels <= 0, labels),
        cmap=feature_label_cmap(),
        shading="auto",
        alpha=0.28,
        rasterized=True,
    )
    ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_where(~accepted_footprint, labels),
        cmap=feature_label_cmap(),
        shading="auto",
        alpha=0.62,
        rasterized=True,
    )

    scene = all_scene_objects.nlargest(10, "area_km2")
    track_ids = set(scene.track_id.astype(int))
    history = source_objects[
        source_objects.track_id.isin(track_ids)
        & (source_objects.time_index >= time_index - PYFLEXTRKR_MOVIE_TRACK_HISTORY_STEPS)
        & (source_objects.time_index <= time_index)
    ]
    for _track_id, track in history.groupby("track_id"):
        track = track.sort_values("time_index")
        x = np.asarray(track.centroid_lon, dtype=float)
        y = np.asarray(track.centroid_lat, dtype=float)
        if x.size > 1:
            jumps = np.where(np.abs(np.diff(x)) > 180.0)[0]
            for jump_index in jumps[::-1]:
                x = np.insert(x, jump_index + 1, np.nan)
                y = np.insert(y, jump_index + 1, np.nan)
        ax.plot(x, y, color="k", lw=1.0, alpha=0.85)
        if len(track):
            ax.scatter(
                [float(track.centroid_lon.iloc[-1])],
                [float(track.centroid_lat.iloc[-1])],
                color="k", s=12, zorder=6,
            )
    for record in scene.itertuples(index=False):
        ax.text(
            float(record.centroid_lon),
            float(record.centroid_lat),
            str(int(record.track_id)),
            fontsize=6,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="0.2", alpha=0.78, pad=0.4),
            clip_on=True,
        )
    ax.set_title(
        f"Raw candidates; accepted organised systems ({len(all_scene_objects.track_id.unique())} tracks)",
        fontsize=9,
    )
    ax.set_xlim(float(lon.min()), float(lon.max()))
    ax.set_ylim(*TROPICAL_LAT_BOUNDS)


def save_pyflextrkr_animation(fig, update, nframes, output_path, title):
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=nframes,
        interval=1000.0 / PYFLEXTRKR_MOVIE_FPS,
        blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=PYFLEXTRKR_MOVIE_FPS,
        codec="libx264",
        bitrate=2200,
        extra_args=[
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
        ],
        metadata={"title": title},
    )
    print(f"Writing PyFLEXTRKR validation movie: {output_path}", flush=True)
    movie.save(output_path, writer=writer, dpi=120)
    plt.close(fig)


def plot_pyflextrkr_validation_movies(objects_df, fields_by_run, masks_by_run):
    if not should_run_figure(12):
        return
    if objects_df is None or len(objects_df) == 0:
        raise RuntimeError("Figure 12 requires current PyFLEXTRKR object tracks")

    precip_cmap, precip_norm = precip_rate_cmap_norm()
    class_cmap, class_norm = high_cloud_class_cmap_norm(alpha=0.82)
    class_legend = high_cloud_class_legend_handles()
    ctp_cmap = plt.get_cmap("magma_r")
    ctp_norm = mpl_colors.Normalize(vmin=80.0, vmax=P_HIGH_HPA)
    manifest_rows = []

    for sim_index, sim_name in enumerate(sim_names):
        sim_objects = objects_df[objects_df.simulation == sim_name].copy()
        files_by_source = {
            source: pyflextrkr_cloudid_files(source, sim_name)
            for source in PYFLEXTRKR_VALIDATION_SOURCES
        }
        missing_sources = [source for source, files in files_by_source.items() if not files]
        if missing_sources:
            raise RuntimeError(
                f"{sim_name}: Figure 12 requires current PyFLEXTRKR files for {missing_sources}"
            )
        ntime = min(
            int(fields_by_run[sim_index]["precip"].sizes["Time"]),
            *(len(files) for files in files_by_source.values()),
        )
        counts = (
            sim_objects[sim_objects.source.isin(PYFLEXTRKR_VALIDATION_SOURCES)]
            .groupby("time_index")
            .size()
        )
        peak_index = int(counts.idxmax()) if len(counts) else ntime // 2
        nframes = min(PYFLEXTRKR_MOVIE_MAX_FRAMES, ntime)
        start = max(0, min(peak_index - nframes // 2, ntime - nframes))
        frame_indices = list(range(start, start + nframes))

        first_source = PYFLEXTRKR_VALIDATION_SOURCES[0]
        with xr.open_dataset(
            files_by_source[first_source][frame_indices[0]],
            decode_times=False,
            mask_and_scale=False,
        ) as first_cloud:
            lat = np.asarray(first_cloud["lat"].values, dtype=float)
            lon = np.asarray(first_cloud["lon"].values, dtype=float)

        def finish_axes(axes):
            for ax in np.atleast_1d(axes).ravel():
                ax.set_xlim(float(lon.min()), float(lon.max()))
                ax.set_ylim(*TROPICAL_LAT_BOUNDS)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

        def frame_time(time_index):
            return pd.to_datetime(fields_by_run[sim_index]["precip"].Time.values[time_index])

        safe_sim = sim_name.replace(" ", "_")

        class_fig, class_axes = plt.subplots(
            3, 1, figsize=(12.0, 9.2), constrained_layout=True, sharex=True
        )
        class_fig.colorbar(
            plt.cm.ScalarMappable(norm=precip_norm, cmap=precip_cmap),
            ax=class_axes[0], label=r"Precipitation (mm day$^{-1}$)", shrink=0.80,
        )
        class_fig.colorbar(
            plt.cm.ScalarMappable(norm=ctp_norm, cmap=ctp_cmap),
            ax=class_axes[2], label="Cloud-top pressure (hPa)", shrink=0.80,
        )

        def update_classification(frame_number):
            time_index = frame_indices[frame_number]
            for ax in class_axes:
                ax.clear()
            precip = tracking_grid_field(fields_by_run[sim_index]["precip"], time_index, lat, lon)
            precip_values = np.asarray(precip, dtype=float)
            class_axes[0].pcolormesh(
                lon, lat, np.maximum(precip_values, 0.0),
                cmap=precip_cmap, norm=precip_norm, shading="auto", rasterized=True,
            )
            class_axes[0].set_title("Precipitation")

            class_code = cloud_class_code_for_movie(
                masks_by_run[sim_index], time_index, lat, lon
            )
            class_axes[1].pcolormesh(
                lon, lat, class_code, cmap=class_cmap, norm=class_norm,
                shading="auto", rasterized=True,
            )
            class_axes[1].set_title("Shared high-cloud classification")
            class_axes[1].legend(handles=class_legend, loc="lower left", fontsize=7)

            ctp = tracking_grid_field(
                masks_by_run[sim_index]["cloud_top_pressure"], time_index, lat, lon
            )
            ctp_values = np.asarray(ctp, dtype=float)
            ctp_values = np.ma.masked_where(class_code <= 0, ctp_values)
            class_axes[2].pcolormesh(
                lon, lat, ctp_values, cmap=ctp_cmap, norm=ctp_norm,
                shading="auto", rasterized=True,
            )
            class_axes[2].set_title("Cloud-top pressure within high cloud")
            finish_axes(class_axes)
            class_fig.suptitle(f"{sim_name} | {frame_time(time_index)}")
            return []

        class_path = os.path.join(save_loc, f"12_cloud_classification_{safe_sim}.mp4")
        save_pyflextrkr_animation(
            class_fig, update_classification, len(frame_indices), class_path,
            f"Cloud classification: {sim_name}",
        )
        manifest_rows.append(dict(simulation=sim_name, movie_type="classification", output_file=class_path))

        for movie_type, sources in (
            ("organised_convective_tracking", PYFLEXTRKR_TRACKED_SOURCES),
        ):
            track_fig, track_axes = plt.subplots(
                len(sources), 1, figsize=(12.0, 3.4 * len(sources)),
                constrained_layout=True, squeeze=False, sharex=True,
            )
            track_axes = track_axes[:, 0]
            track_fig.colorbar(
                plt.cm.ScalarMappable(norm=precip_norm, cmap=precip_cmap),
                ax=track_axes.tolist(), label=r"Precipitation (mm day$^{-1}$)", shrink=0.80,
            )

            def update_tracking(frame_number, axes=track_axes, source_names=tuple(sources), fig=track_fig):
                time_index = frame_indices[frame_number]
                precip = tracking_grid_field(
                    fields_by_run[sim_index]["precip"], time_index, lat, lon
                )
                precip_values = np.asarray(precip, dtype=float)
                for source_name, ax in zip(source_names, axes):
                    ax.clear()
                    draw_pyflextrkr_movie_object_panel(
                        ax, source_name, sim_objects,
                        files_by_source[source_name][time_index], precip_values,
                        time_index, lat, lon,
                    )
                finish_axes(axes)
                fig.suptitle(f"{sim_name} | {frame_time(time_index)}")
                return []

            track_path = os.path.join(save_loc, f"12_{movie_type}_{safe_sim}.mp4")
            save_pyflextrkr_animation(
                track_fig, update_tracking, len(frame_indices), track_path,
                f"{movie_type.replace('_', ' ').title()}: {sim_name}",
            )
            manifest_rows.append(dict(simulation=sim_name, movie_type=movie_type, output_file=track_path))

        for row in manifest_rows:
            if row["simulation"] == sim_name:
                row.update(
                    frame_count=len(frame_indices),
                    first_time_index=frame_indices[0],
                    last_time_index=frame_indices[-1],
                    fps=PYFLEXTRKR_MOVIE_FPS,
                )

    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(save_loc, "pyflextrkr_validation_movie_manifest.csv"),
        index=False,
    )


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def select_test_source_window(ds, sim_name):
    """Select N+1 source snapshots so accumulated rain can yield N rates."""
    if "Time" not in ds.dims:
        raise RuntimeError(f"{sim_name}: test mode requires a Time dimension")
    index = ds.Time.to_index()
    source_count = TEST_TIME_STEPS + 1
    if len(index) < source_count:
        raise RuntimeError(
            f"{sim_name}: test mode requested {source_count} source snapshots, "
            f"but only {len(index)} are available"
        )
    first = 0
    if TEST_START is not None:
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("--test-start requires a datetime Time coordinate")
        candidates = np.flatnonzero(index >= pd.Timestamp(TEST_START))
        if candidates.size == 0:
            raise ValueError(
                f"{sim_name}: --test-start {TEST_START!r} is after the available data"
            )
        first = int(candidates[0])

    selected = None
    for start in range(first, len(index) - source_count + 1):
        candidate = index[start:start + source_count]
        if isinstance(candidate, pd.DatetimeIndex):
            if not np.all(np.diff(candidate.values) == np.timedelta64(int(dt), "h")):
                continue
        selected = candidate
        break
    if selected is None:
        raise RuntimeError(
            f"{sim_name}: could not find {source_count} contiguous {dt:g}-hourly snapshots"
        )
    print(
        f"{sim_name}: test source window {selected[0]} to {selected[-1]} "
        f"({source_count} source snapshots)",
        flush=True,
    )
    return ds.sel(Time=selected)


def trim_test_fields(fields, sim_name):
    precip = fields["precip"]
    if "Time" not in precip.dims or precip.sizes["Time"] < TEST_TIME_STEPS:
        raise RuntimeError(
            f"{sim_name}: test mode produced fewer than {TEST_TIME_STEPS} precipitation times"
        )
    tracked_times = precip.Time.isel(Time=slice(0, TEST_TIME_STEPS))
    out = {}
    for key, da in fields.items():
        if "Time" in da.dims:
            missing = tracked_times.to_index().difference(da.Time.to_index())
            if len(missing) > 0:
                raise RuntimeError(
                    f"{sim_name}: {key} is missing {len(missing)} requested test times"
                )
            da = da.sel(Time=tracked_times)
        out[key] = da
    print(
        f"{sim_name}: tracking {TEST_TIME_STEPS} snapshots from "
        f"{tracked_times.values[0]} to {tracked_times.values[-1]}",
        flush=True,
    )
    return out


def materialize_test_tracking_inputs(fields_by_run, masks_by_run):
    """Load each short tropical test window once for all tracked sources."""
    if not TEST_MODE:
        return
    for sim_name, fields, masks in zip(sim_names, fields_by_run, masks_by_run):
        print(
            f"{sim_name}: loading the short tropical fields and masks once",
            flush=True,
        )
        precip = subset_lat(
            fields["precip"], lat_bounds=TROPICAL_LAT_BOUNDS
        ).load()
        fields["precip"] = precip
        fields["mean_precip"] = weighted_tropical_mean(
            precip, lat_bounds=TROPICAL_LAT_BOUNDS
        ).load()
        test_mask_names = list(dict.fromkeys([
            *[name for name in PYFLEXTRKR_TRACKED_SOURCES if name in masks],
            *[name for name in CLASSIFICATION_DIAGNOSTIC_KEYS if name in masks],
        ]))
        for source_name in test_mask_names:
            print(f"{sim_name}: loading test mask {source_name}", flush=True)
            masks[source_name] = subset_lat(
                masks[source_name], lat_bounds=TROPICAL_LAT_BOUNDS
            ).load()


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
        )
        ds = standardize_latlon_dataset(ds, needed_vars)
        if TEST_MODE:
            ds = select_test_source_window(ds, sim_names[i])
        print(ds)
        dss.append(ds)
    return dss


def build_masks_by_run(dss, fields_by_run, threshold_table):
    masks_by_run = []
    for i, (ds, fields) in enumerate(zip(dss, fields_by_run)):
        sim_name = sim_names[i]
        masks = threshold_object_masks(fields, threshold_table)
        masks.update(cloud_type_masks(ds, fields, sim_name))
        masks[ORGANISED_SYSTEM_SOURCE] = build_organised_convective_system_mask(
            fields,
            masks,
        )
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


def main(argv=None):
    args = parse_command_line(argv)
    configure_run(args)
    os.makedirs(save_loc, exist_ok=True)
    if TEST_MODE:
        print(
            f"TEST MODE: simulation={sim_names[0]}, tracked_times={TEST_TIME_STEPS}, "
            f"figures={figures_to_run}, reuse_cache={TEST_REUSE_CACHE}, output={save_loc}",
            flush=True,
        )

    dss = None
    fields_by_run = None
    masks_by_run = None

    def ensure_fields():
        nonlocal dss, fields_by_run
        if fields_by_run is None:
            dss = load_datasets()
            fields_by_run = [build_fields(ds) for ds in dss]
            if TEST_MODE:
                fields_by_run = [
                    trim_test_fields(fields, sim_name)
                    for fields, sim_name in zip(fields_by_run, sim_names)
                ]
                rows = []
                for sim_name, fields in zip(sim_names, fields_by_run):
                    rows.append(
                        dict(
                            simulation=sim_name,
                            tracked_snapshot_count=int(fields["precip"].sizes["Time"]),
                            first_tracked_time=str(fields["precip"].Time.values[0]),
                            last_tracked_time=str(fields["precip"].Time.values[-1]),
                            snapshot_spacing_hours=dt,
                            output_directory=save_loc,
                            pyflextrkr_work_directory=PYFLEXTRKR_WORK_DIR,
                            reused_test_cache=TEST_REUSE_CACHE,
                        )
                    )
                pd.DataFrame(rows).to_csv(
                    os.path.join(save_loc, "pyflextrkr_test_run_manifest.csv"),
                    index=False,
                )
        return dss, fields_by_run

    def ensure_masks(threshold_table):
        nonlocal masks_by_run
        if masks_by_run is None:
            dss_local, fields_local = ensure_fields()
            masks_by_run = build_masks_by_run(dss_local, fields_local, threshold_table)
        return masks_by_run

    threshold_table = pd.DataFrame()
    metrics_df = pd.DataFrame()
    summary = pd.DataFrame()
    metric_products_needed = should_run_any([1, 2, 3, 4, 5])
    if metric_products_needed:
        metric_products = load_cached_metric_products()
        if metric_products is None:
            dss, fields_by_run = ensure_fields()
            threshold_table = build_threshold_table(fields_by_run)
            if len(threshold_table) > 0:
                print("\nOrganisation object thresholds")
                print(threshold_table.to_string(index=False))

            masks_by_run = ensure_masks(threshold_table)
            metrics_df = build_all_metrics(
                dss, fields_by_run, threshold_table, masks_by_run=masks_by_run
            )
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
    track_persistence_needed = should_run_any([6, 7, 8, 9, 11, 12, 13, 14])
    if track_persistence_needed:
        if len(threshold_table) == 0:
            _, fields_by_run = ensure_fields()
            threshold_table = build_threshold_table(fields_by_run)
        track_products = load_cached_track_persistence_products()
        if track_products is None:
            masks_by_run = ensure_masks(threshold_table)
            _, fields_by_run = ensure_fields()
            materialize_test_tracking_inputs(fields_by_run, masks_by_run)
            objects_df, tracks_df, contribution_df = build_track_persistence_metrics(masks_by_run, fields_by_run)
            write_track_persistence_cache(objects_df, tracks_df, contribution_df)
        else:
            objects_df, tracks_df, contribution_df = track_products
            if TEST_MODE and should_run_any([11, 12, 14]):
                masks_by_run = ensure_masks(threshold_table)
                _, fields_by_run = ensure_fields()
                materialize_test_tracking_inputs(fields_by_run, masks_by_run)

        write_pyflextrkr_obsolete_output_report()
        write_organised_track_acceptance_report()
        if should_run_any([11, 12, 13, 14]):
            _, fields_by_run = ensure_fields()
            masks_by_run = ensure_masks(threshold_table)

    if len(summary) > 0:
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
            "Tracked-source object area fraction",
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
            "Track-mean source-object area fraction",
        )
    if should_run_figure(10):
        persistence_df = load_cached_pattern_persistence()
        if persistence_df is None:
            _, fields_by_run = ensure_fields()
            persistence_df = compute_persistence(fields_by_run)
            write_pattern_persistence_cache(persistence_df)
        if len(persistence_df) > 0:
            plot_shifted_pattern_autocorrelation(persistence_df)
    if should_run_figure(11):
        _, fields_by_run = ensure_fields()
        masks_by_run = ensure_masks(threshold_table)
        plot_pyflextrkr_track_validation(
            objects_df, fields_by_run, masks_by_run
        )
    if should_run_figure(12):
        _, fields_by_run = ensure_fields()
        masks_by_run = ensure_masks(threshold_table)
        plot_pyflextrkr_validation_movies(objects_df, fields_by_run, masks_by_run)
    if should_run_figure(13):
        plot_pyflextrkr_track_step_validation(objects_df)
    if should_run_figure(14):
        _, fields_by_run = ensure_fields()
        masks_by_run = ensure_masks(threshold_table)
        plot_pyflextrkr_validation_dashboard(
            objects_df, tracks_df, fields_by_run, masks_by_run
        )
    if should_run_figure(15):
        _, fields_by_run = ensure_fields()
        masks_by_run = ensure_masks(threshold_table)
        plot_classification_pyflextrkr_comparison(
            objects_df, fields_by_run, masks_by_run
        )

    print("\nFinished convective_organisation_diagnostics.py")


if __name__ == "__main__":
    main()
