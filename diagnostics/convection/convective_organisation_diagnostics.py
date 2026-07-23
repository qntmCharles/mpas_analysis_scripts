#!/usr/bin/env python
# coding: utf-8

"""
Persistent deep-convective-system and Feng-MCS diagnostics for MPAS-A.

The primary analysis tracks the exact shared PyFLEXTRKR 241-K cold shields and
reports a broad, nested population of persistent core-bearing systems and the
strict subset satisfying Feng et al. (2021) 50-km cold-shield/precipitation-
feature criteria. It reports the same simple event properties for both sets;
the other figures validate segmentation, temporal links, population selection,
and the relation to shared high-cloud classes.
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings
from functools import partial

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

from diagnostics.diagnostic_helpers import (
    ANALYSIS_END, ANALYSIS_START, FigureSaver, LATLON_PATTERN,
    DEFAULT_ORGANISATION_TRACKING_MODE,
    MIN_ORGANISED_TRACK_DURATION_HOURS,
    MIN_ORGANISED_TRACK_OBSERVATIONS, ORGANISED_ENVELOPE_MIN_RAW_AREA_KM2,
    FENG_MCS_COLD_SHIELD_MIN_AREA_KM2, FENG_MCS_CRITERIA_PERCENTILE,
    FENG_MCS_PF_AREA_INTERCEPT_KM2, FENG_MCS_PF_AREA_SLOPE_KM2_PER_H,
    FENG_MCS_PF_MEAN_RAIN_INTERCEPT_MM_H,
    FENG_MCS_PF_MEAN_RAIN_SLOPE_MM_H_PER_H,
    FENG_MCS_PF_MIN_MAJOR_AXIS_KM, FENG_MCS_PF_MIN_RAIN_MM_H,
    FENG_MCS_PF_SKEW_INTERCEPT, FENG_MCS_PF_SKEW_SLOPE_PER_H,
    ORGANISED_TRACK_ACCEPTANCE_CRITERIA,
    ORGANISED_TRACK_EMBEDDED_CORE_REQUIREMENT,
    ORGANISED_TRACK_OBJECT_GEOMETRY, PRIMARY_ANALYSIS_ROLE,
    PERSISTENT_DCS_ACCEPTANCE_CRITERIA,
    PRIMARY_TRACK_SOURCE, PYFLEXTRKR_PRODUCT_VERSION,
    PYFLEXTRKR_TRACK_MANIFEST_FILENAME, TRACKED_SOURCES,
    TRACK_POPULATION_LABELS, TRACK_POPULATIONS,
    QUICK_TEST_LAT_BOUNDS, QUICK_TEST_LON_BOUNDS,
    convective_organisation_cache_dir, diagnostic_cache_dir,
    diagnostic_output_dir, env_flag, figure_should_run,
    high_cloud_class_cmap_norm, high_cloud_class_legend_handles,
    grid_cell_area_km2, haversine_distances_km, jsonable_value,
    load_latlon_dataset, manifest_accepted_mask, manifest_population_mask,
    median_time_step_days, metadata_reader, metadata_writer,
    precip_rate_cmap_norm, precipitation_rate_from_dataset,
    product_fingerprint, progress, read_product_metadata, read_track_manifest,
    sha256_file, sha256_label_sequence,
    standardize_latlon_dataset, subset_latitude,
    time_days_from_coord, time_signature, track_manifest_path,
    weighted_tropical_mean, write_product_metadata,
)
from diagnostics.convection.high_cloud_classification import (
    CLASSIFICATION_CACHE_VERSION,
    HighCloudThresholds,
    cached_classification_is_current as shared_cached_classification_is_current,
    corrected_cached_warm_shield_masks,
)
from pyflextrkr.gettracks import gettracknumbers as pyflextrkr_gettracknumbers
from pyflextrkr.tracksingle import trackclouds as pyflextrkr_trackclouds
from pyflextrkr.tracksingle_drift import trackclouds as pyflextrkr_trackclouds_drift
from pyflextrkr.advection_tiles import calc_mean_advection
from pyflextrkr.ft_utilities import match_drift_times


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/organisation/")
default_save_loc = save_loc

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: persistent-DCS and Feng-MCS population/property figures
#   2: PyFLEXTRKR cloud-object and track validation snapshots
#   3: track acceptance plus size/speed validation
#   4: high-cloud classes versus exclusive PyFLEXTRKR scene regions
#   5: per-simulation evolving classification and track-population movies
#
# WHAT TO LOOK FOR:
#   * Event rate, cold-shield coverage, lifetime and PF properties are reported
#     identically for the broad persistent-DCS and strict Feng-MCS populations.
#   * The organised convective-system envelope is the primary tracked object set
#     used for persistence and downstream lifecycle/wave composites.  The
#     convective, stratiform, and non-precipitating high-cloud masks are sampled
#     inside each envelope rather than defining the tracked geometry.
#   * The envelope follows the Feng et al. 50-km MCS method: 241-K cold-cloud
#     shields are tracked, then the cold-shield and precipitation-feature
#     criteria must hold in at least two consecutive six-hour scenes.
#   * In Figure 2, accepted footprints should occupy coherent parts of the
#     source mask and numbered past paths should move continuously.
#   * Figure 3 shows candidate and accepted object counts through time, plus
#     track acceptance totals and diagnostic size/speed distributions; size and
#     speed are not acceptance criteria.
#   * Figure 4 separates the cold core, cold-shield shell, and warm-shield
#     shell used to describe each PyFLEXTRKR scene.
#   * Figure 5 animates the same acceptance and class/shield decomposition for
#     a window of up to 48 consecutive six-hour scenes per simulation.
figures_to_run = [1]
figsave = None

TEST_MODE = False
TEST_TIME_STEPS = 16
TEST_START = None
TEST_REUSE_CACHE = False
TEST_DEFAULT_FIGURES = [1, 2, 3, 4, 5]
SIMULATION_ARGUMENTS = {
    "control": 0,
    "nudged-u": 1,
    "nudged-T": 2,
}

FORCE_RECOMPUTE_PERSISTENCE_CACHE = False
FORCE_RECOMPUTE_PERSISTENCE_CACHE = (
    FORCE_RECOMPUTE_PERSISTENCE_CACHE
    or env_flag("FORCE_RECOMPUTE_PERSISTENCE_CACHE")
)

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control_new", "u_nudge", "T_nudge"]
sim_names = ["control", "nudged u", "nudged T"]
cols = ["k", "b", "r"]

dt = 6

TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
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
PYFLEXTRKR_WARM_ANVIL_EXPANSION = 0
VALIDATION_SCENE_SEARCH_LIMIT = 24
PYFLEXTRKR_MOVIE_MAX_FRAMES = 48
PYFLEXTRKR_MOVIE_FPS = 4
PYFLEXTRKR_MOVIE_TRACK_HISTORY_STEPS = 6

# Shared high-cloud thresholds used to validate the refined classification cache.
CF_THRESH = 0.05
P_HIGH_HPA = 440.0
P_PRECIP_MIN = 12.0  # 0.5 mm h-1 PF threshold (Feng et al. 2021)

MIN_OBJECT_CELLS = 1
MIN_OBJECT_AREA_KM2 = 0.0

# Organised-envelope definition.  PyFLEXTRKR defines the cold-cloud shield
# geometry from Tb; the classification classes are sampled inside those shields.
ENVELOPE_MIN_RAW_AREA_KM2 = ORGANISED_ENVELOPE_MIN_RAW_AREA_KM2

# Feng et al. require the CCS and PF criteria continuously for at least 6 h.
# With this analysis' six-hour output, this is represented conservatively by
# at least two consecutive sampled scenes.
MIN_TRACK_DURATION_HOURS = MIN_ORGANISED_TRACK_DURATION_HOURS

# Persistence follows the core PyFLEXTRKR overlap tracker (Feng et al., 2023):
# adjacent objects are linked by bidirectional overlap, with split/merge branches
# recorded separately from the continuously tracked largest branch.
TRACKING_MODE = DEFAULT_ORGANISATION_TRACKING_MODE
ORGANISATION_CACHE_DIR = convective_organisation_cache_dir(
    TRACKING_MODE, root_dir=root_dir
)
PYFLEXTRKR_WORK_DIR = diagnostic_cache_dir(
    "pyflextrkr_tracking", TRACKING_MODE, root_dir=root_dir
)
PYFLEXTRKR_NMAXLINKS = 50
PYFLEXTRKR_FILLVAL = -9999
FORCE_RECOMPUTE_PYFLEXTRKR_CACHE = (
    FORCE_RECOMPUTE_PERSISTENCE_CACHE
    or env_flag("FORCE_RECOMPUTE_PYFLEXTRKR_CACHE")
)
TRACK_MIN_OVERLAP_FRACTION = 0.50
TRACK_MIN_AREA_KM2 = 800.0
MIN_TRACK_OBSERVATIONS = MIN_ORGANISED_TRACK_OBSERVATIONS
PYFLEXTRKR_ADVECTION_MEDIAN_FILTER_LENGTH = 3
PYFLEXTRKR_ADVECTION_MAX_MOVEMENT_MPS = 60.0

SOURCE_LABELS = {
    "organised_convective_system_mask": "Organised convective-system envelope",
    "high_cloud_mask": "High cloud",
    "convective_precipitating_high_cloud_mask": "PyFLEXTRKR convective core",
    "stratiform_precipitating_high_cloud_mask": "Stratiform precipitating high cloud",
    "nonprecipitating_high_cloud_mask": "Non-precipitating high cloud",
    "unclassified_high_cloud_mask": "Invalid/unclassified high cloud",
}

SOURCE_ORDER = list(CLOUD_OBJECT_KEYS)

CLASSIFICATION_DIAGNOSTIC_KEYS = [
    "pyflextrkr_core_mask",
    "pyflextrkr_cold_cloud_shield_mask",
    "pyflextrkr_core_bearing_cold_shield_mask",
    "pyflextrkr_warm_cloud_shield_mask",
    "pyflextrkr_core_bearing_warm_shield_mask",
    "pyflextrkr_cold_cloud_shield_label",
    "high_cloud_mask",
    "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
    "unclassified_high_cloud_mask",
    "cloud_top_pressure",
]


def feng_mcs_metadata():
    return dict(
        feng_mcs_cold_shield_min_area_km2=float(FENG_MCS_COLD_SHIELD_MIN_AREA_KM2),
        feng_mcs_pf_min_rain_mm_h=float(FENG_MCS_PF_MIN_RAIN_MM_H),
        feng_mcs_pf_min_major_axis_km=float(FENG_MCS_PF_MIN_MAJOR_AXIS_KM),
        feng_mcs_pf_area_intercept_km2=float(FENG_MCS_PF_AREA_INTERCEPT_KM2),
        feng_mcs_pf_area_slope_km2_per_h=float(FENG_MCS_PF_AREA_SLOPE_KM2_PER_H),
        feng_mcs_pf_mean_rain_intercept_mm_h=float(FENG_MCS_PF_MEAN_RAIN_INTERCEPT_MM_H),
        feng_mcs_pf_mean_rain_slope_mm_h_per_h=float(FENG_MCS_PF_MEAN_RAIN_SLOPE_MM_H_PER_H),
        feng_mcs_pf_skew_intercept=float(FENG_MCS_PF_SKEW_INTERCEPT),
        feng_mcs_pf_skew_slope_per_h=float(FENG_MCS_PF_SKEW_SLOPE_PER_H),
        feng_mcs_criteria_percentile=int(FENG_MCS_CRITERIA_PERCENTILE),
    )

latlon_vars = [
    "rainc", "rainnc", "i_rainc", "i_rainnc", "olrtoa", "precipw",
]

latlon_chunks = {"Time": 8, "latitude": 90, "longitude": 180}
dask.config.set({"array.slicing.split_large_chunks": True})


should_run_figure = partial(figure_should_run, figures_to_run)


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
        "--qc",
        action="store_true",
        help="Add the three PyFLEXTRKR validation figures to the primary MCS figure.",
    )
    parser.add_argument(
        "--tracking-mode",
        choices=("standard", "drift"),
        default=DEFAULT_ORGANISATION_TRACKING_MODE,
        help=(
            "PyFLEXTRKR adjacent-scene linker. 'drift' applies PyFLEXTRKR's "
            "domain-mean motion correction before the published 0.5 "
            "bidirectional-overlap test; 'standard' omits motion correction."
        ),
    )
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        choices=range(1, 6),
        default=None,
        metavar="N",
        help="Run only the listed figure numbers (1-5; Figure 5 writes movies).",
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
    global ORGANISATION_CACHE_DIR
    global data_locs, sim_names, cols
    global CLOUD_CLASSIFICATION_CACHE_DIRS
    global FORCE_RECOMPUTE_PERSISTENCE_CACHE, FORCE_RECOMPUTE_PYFLEXTRKR_CACHE
    global TRACKING_MODE

    TEST_MODE = bool(args.test)
    TEST_TIME_STEPS = int(args.test_times)
    TEST_START = args.test_start
    TEST_REUSE_CACHE = bool(args.reuse_test_cache)
    TRACKING_MODE = str(args.tracking_mode)

    if TRACKING_MODE == "drift" and not TEST_MODE:
        save_loc = os.path.join(default_save_loc, "tracking_sensitivity_drift")

    if args.figures is not None:
        figures_to_run[:] = args.figures
    elif args.qc:
        figures_to_run[:] = [1, 2, 3, 4, 5]
    elif TEST_MODE:
        figures_to_run[:] = TEST_DEFAULT_FIGURES

    if TEST_MODE:
        index = SIMULATION_ARGUMENTS[args.simulation]
        data_locs = [data_locs[index]]
        sim_names = [sim_names[index]]
        cols = [cols[index]]

        start_label = f"_{TEST_START}" if TEST_START else ""
        mode_label = "_drift" if TRACKING_MODE == "drift" else ""
        test_label = (
            f"{args.simulation}{start_label}_{TEST_TIME_STEPS:03d}times{mode_label}"
        )
        test_label = test_label.replace(":", "").replace(" ", "_")
        save_loc = os.path.join(default_save_loc, "testing", test_label)
        ORGANISATION_CACHE_DIR = diagnostic_cache_dir(
            "organisation_cache", "testing", test_label, root_dir=root_dir
        )
        PYFLEXTRKR_WORK_DIR = diagnostic_cache_dir(
            "pyflextrkr_tracking", "testing", test_label, root_dir=root_dir
        )
        test_classification_dir = os.path.join(
            CLASSIFICATION_TEST_OUTPUT_ROOT,
            "testing",
            test_label,
            "classification_products",
        )
        classification_dirs = [test_classification_dir]
        if TRACKING_MODE == "drift":
            standard_label = (
                f"{args.simulation}{start_label}_{TEST_TIME_STEPS:03d}times"
            ).replace(":", "").replace(" ", "_")
            classification_dirs.append(
                os.path.join(
                    CLASSIFICATION_TEST_OUTPUT_ROOT,
                    "testing",
                    standard_label,
                    "classification_products",
                )
            )
        CLOUD_CLASSIFICATION_CACHE_DIRS = [
            *classification_dirs,
            *CLOUD_CLASSIFICATION_CACHE_DIRS,
        ]

        if not TEST_REUSE_CACHE:
            FORCE_RECOMPUTE_PERSISTENCE_CACHE = True
            FORCE_RECOMPUTE_PYFLEXTRKR_CACHE = True
    else:
        ORGANISATION_CACHE_DIR = convective_organisation_cache_dir(
            TRACKING_MODE, root_dir=root_dir
        )
        PYFLEXTRKR_WORK_DIR = diagnostic_cache_dir(
            "pyflextrkr_tracking", TRACKING_MODE, root_dir=root_dir
        )

    figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)


def save_figure(number, fig, stem, dpi=300):
    global figsave
    if figsave is None:
        figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)
    return figsave.save(fig, stem, number=number, dpi=dpi)


def format_duration(seconds):
    """Format elapsed/remaining seconds compactly for captured job logs."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def report_loop_progress(label, completed, total, started, target_updates=20):
    """Report roughly ``target_updates`` evenly spaced updates for a loop."""
    if total <= 0 or completed <= 0:
        return
    interval = max(1, (total + target_updates - 1) // target_updates)
    if completed != 1 and completed != total and completed % interval != 0:
        return
    elapsed = time.monotonic() - started
    eta = elapsed / completed * (total - completed)
    progress(
        f"{label}: {completed}/{total} ({100.0 * completed / total:.0f}%); "
        f"elapsed {format_duration(elapsed)}, ETA {format_duration(eta)}"
    )


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
        f"core < {PYFLEXTRKR_TB_CORE_K:g} K; "
        f"cold < {PYFLEXTRKR_TB_COLD_K:g} K; "
        f"warm/anvil < {PYFLEXTRKR_TB_WARM_K:g} K"
    )


# -----------------------------------------------------------------------------
# Metric/persistence caches
# -----------------------------------------------------------------------------

TRACK_PERSISTENCE_CACHE_VERSION = "pyflextrkr_dual_population_tracks_v15"

ORGANISATION_CACHE_FILES = {
    "track_metadata": "organisation_track_persistence_cache_metadata.json",
    "tracked_objects": "organisation_tracked_objects.csv",
    "track_lifetimes": "organisation_track_lifetimes.csv",
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
        "persistent_dcs",
        "feng_mcs",
        "tracking_method",
    ],
    "track_lifetimes": [
        "source",
        "source_label",
        "simulation",
        "track_id",
        "persistent_dcs",
        "feng_mcs",
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
}


def organisation_cache_path(name):
    return os.path.join(ORGANISATION_CACHE_DIR, ORGANISATION_CACHE_FILES[name])


def base_data_cache_signature():
    return dict(
        root_dir=root_dir,
        data_locs=list(data_locs),
        latlon_pattern=LATLON_PATTERN,
        sim_names=list(sim_names),
        analysis_start_date=ANALYSIS_START,
        dt=float(dt),
        analysis_end_date=ANALYSIS_END,
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
        **feng_mcs_metadata(),
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
        persistent_dcs_acceptance_criteria=PERSISTENT_DCS_ACCEPTANCE_CRITERIA,
        track_populations=list(TRACK_POPULATIONS),
        tracking_mode=TRACKING_MODE,
        pyflextrkr_advection_median_filter_length=(
            int(PYFLEXTRKR_ADVECTION_MEDIAN_FILTER_LENGTH)
            if TRACKING_MODE == "drift" else None
        ),
        pyflextrkr_advection_max_movement_mps=(
            float(PYFLEXTRKR_ADVECTION_MAX_MOVEMENT_MPS)
            if TRACKING_MODE == "drift" else None
        ),
    )
    return signature


read_cache_metadata = metadata_reader(organisation_cache_path)
write_cache_metadata = metadata_writer(organisation_cache_path, enabled=save)


def cache_metadata_status(name, signature, label):
    if FORCE_RECOMPUTE_PERSISTENCE_CACHE:
        print(f"Ignoring cached {label}: force recompute is enabled.")
        return "miss"
    cached = read_cache_metadata(name)
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
    return pd.read_csv(path)


def write_cache_table(name, df):
    if not save or df is None:
        return
    os.makedirs(ORGANISATION_CACHE_DIR, exist_ok=True)
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


def load_cached_track_persistence_products():
    metadata_status = cache_metadata_status(
        "track_metadata",
        track_persistence_cache_signature(),
        "MCS track products",
    )
    if metadata_status == "miss":
        return None

    missing = [
        name
        for name in ("tracked_objects", "track_lifetimes")
        if not os.path.exists(organisation_cache_path(name))
    ]
    if missing:
        print(f"Ignoring cached track persistence metrics: missing {', '.join(missing)}.")
        return None

    objects_df = read_cache_table("tracked_objects")
    tracks_df = read_cache_table("track_lifetimes")
    for label, frame in (
        ("tracked objects", objects_df),
        ("track lifetimes", tracks_df),
    ):
        if not frame_has_current_simulations(frame, label, require_all=False):
            return None
        if not frame_has_only_current_sources(frame, label):
            return None

    progress(
        f"Using cached dual-population track products from {ORGANISATION_CACHE_DIR}: "
        f"{len(objects_df)} retained scene objects, {len(tracks_df)} retained tracks"
    )
    return objects_df, tracks_df


def write_track_persistence_cache(objects_df, tracks_df):
    os.makedirs(ORGANISATION_CACHE_DIR, exist_ok=True)
    write_cache_table("tracked_objects", objects_df)
    write_cache_table("track_lifetimes", tracks_df)
    write_cache_metadata("track_metadata", track_persistence_cache_signature())
    progress(f"Wrote dual-population track cache to {ORGANISATION_CACHE_DIR}")


# -----------------------------------------------------------------------------
# Basic data helpers
# -----------------------------------------------------------------------------

subset_lat = partial(subset_latitude, lat_bounds=TROPICAL_LAT_BOUNDS)
get_precip_rate = precipitation_rate_from_dataset
weighted_tropical_mean = partial(
    weighted_tropical_mean,
    lat_bounds=TROPICAL_LAT_BOUNDS,
)


def source_label(source_name):
    return SOURCE_LABELS.get(source_name, source_name.replace("_", " "))


def track_source_role(source_name):
    if source_name == ORGANISED_SYSTEM_SOURCE:
        return PRIMARY_ANALYSIS_ROLE
    raise ValueError(
        f"PyFLEXTRKR output is only produced for {ORGANISED_SYSTEM_SOURCE!r}, "
        f"got {source_name!r}"
    )


# -----------------------------------------------------------------------------
# Grid/object geometry
# -----------------------------------------------------------------------------

grid_cell_area_km2 = partial(grid_cell_area_km2, earth_radius_km=EARTH_RADIUS_KM)
haversine_distances_km = partial(
    haversine_distances_km,
    lon_period=LONGITUDE_PERIOD_DEG,
    earth_radius_km=EARTH_RADIUS_KM,
)


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


def precipitation_feature_properties(
    object_mask, precip_mm_day, area_km2, lat_grid, lon_grid
):
    """Properties of the largest contiguous 0.5 mm h-1 PF in one shield."""
    empty = dict(
        pf_area_km2=0.0,
        pf_major_axis_km=0.0,
        pf_mean_rain_mm_h=np.nan,
        pf_max_rain_mm_h=np.nan,
        pf_rain_skewness=np.nan,
        pf_cell_count=0,
    )
    if precip_mm_day is None:
        return empty
    rain_mm_h = np.asarray(precip_mm_day, dtype=float) / 24.0
    pf_mask = (
        object_mask
        & np.isfinite(rain_mm_h)
        & (rain_mm_h >= FENG_MCS_PF_MIN_RAIN_MM_H)
    )
    if not np.any(pf_mask):
        return empty
    pf_labels = ndimage.label(pf_mask, structure=np.ones((3, 3), dtype=np.int8))[0]
    pf_labels = merge_periodic_longitude_labels(pf_labels)
    candidates = []
    for label_id in np.unique(pf_labels[pf_labels > 0]):
        region = pf_labels == label_id
        candidates.append((float(np.nansum(area_km2[region])), int(label_id)))
    pf_area, largest_label = max(candidates)
    region = pf_labels == largest_label
    weights = np.asarray(area_km2[region], dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
    rain = rain_mm_h[region]
    mean_rain = float(np.average(rain, weights=weights))
    max_rain = float(np.nanmax(rain))
    std_rain = float(np.nanstd(rain))
    skewness = (
        float(np.nanmean(((rain - np.nanmean(rain)) / std_rain) ** 3))
        if rain.size >= 3 and std_rain > 0.0
        else 0.0
    )

    lat = lat_grid[region]
    lon = lon_grid[region] % LONGITUDE_PERIOD_DEG
    lat0 = float(np.average(lat, weights=weights))
    theta = np.deg2rad(lon)
    lon0 = np.rad2deg(
        np.arctan2(
            np.average(np.sin(theta), weights=weights),
            np.average(np.cos(theta), weights=weights),
        )
    ) % LONGITUDE_PERIOD_DEG
    dlon = (lon - lon0 + 180.0) % 360.0 - 180.0
    x = 111.195 * np.cos(np.deg2rad(lat0)) * dlon
    y = 111.195 * (lat - lat0)
    xy = np.column_stack([x, y])
    covariance = (
        np.cov(xy, rowvar=False, aweights=weights, bias=True)
        if xy.shape[0] >= 2
        else np.zeros((2, 2), dtype=float)
    )
    if np.ndim(covariance) == 0:
        max_variance = 0.0
    else:
        max_variance = float(np.nanmax(np.linalg.eigvalsh(covariance)))
    cell_scale = float(np.sqrt(np.nanmedian(weights)))
    # Equivalent to the regionprops 4*sqrt(lambda) convention, augmented by
    # within-cell variance so a coarse-grid PF has its finite physical width.
    major_axis = 4.0 * np.sqrt(max(0.0, max_variance) + cell_scale**2 / 12.0)
    return dict(
        pf_area_km2=float(pf_area),
        pf_major_axis_km=float(major_axis),
        pf_mean_rain_mm_h=mean_rain,
        pf_max_rain_mm_h=max_rain,
        pf_rain_skewness=skewness,
        pf_cell_count=int(np.count_nonzero(region)),
    )


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
            **precipitation_feature_properties(
                obj, precip_vals, area_vals, lat_grid, lon_grid
            ),
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


def write_pyflextrkr_cloudid_files(
    labels_by_time,
    mask,
    work_path,
    progress_label="PyFLEXTRKR cloud-ID files",
):
    tracking_path = os.path.join(work_path, "tracking")
    os.makedirs(tracking_path, exist_ok=True)
    clear_generated_pyflextrkr_files(tracking_path)

    lat = np.asarray(mask.latitude.values, dtype=np.float32)
    lon = np.asarray(mask.longitude.values, dtype=np.float32)
    lon2d, lat2d = np.meshgrid(lon, lat)
    basetimes = epoch_seconds_from_time_values(mask.Time.values)
    files = []
    started = time.monotonic()
    progress(f"{progress_label}: writing {len(labels_by_time)} scene files")

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
        report_loop_progress(
            progress_label,
            it + 1,
            len(labels_by_time),
            started,
        )

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
    """Assign raw PyFLEXTRKR tracks to persistent-DCS and Feng-MCS sets."""
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
        observed_run = 0
        max_observed_run = 0
        previous_observed_index = None
        for time_index in group.time_index.to_numpy(dtype=int):
            observed_run = (
                observed_run + 1
                if previous_observed_index is not None
                and time_index == previous_observed_index + 1
                else 1
            )
            max_observed_run = max(max_observed_run, observed_run)
            previous_observed_index = time_index
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
        feng_lifetime_hours = max(observed_span_hours, median_dt)
        pf_area_threshold = (
            FENG_MCS_PF_AREA_INTERCEPT_KM2
            + FENG_MCS_PF_AREA_SLOPE_KM2_PER_H * feng_lifetime_hours
        )
        pf_mean_rain_threshold = (
            FENG_MCS_PF_MEAN_RAIN_INTERCEPT_MM_H
            + FENG_MCS_PF_MEAN_RAIN_SLOPE_MM_H_PER_H * feng_lifetime_hours
        )
        pf_skew_threshold = (
            FENG_MCS_PF_SKEW_INTERCEPT
            + FENG_MCS_PF_SKEW_SLOPE_PER_H * feng_lifetime_hours
        )
        required_pf_columns = (
            "pf_area_km2",
            "pf_major_axis_km",
            "pf_mean_rain_mm_h",
            "pf_rain_skewness",
        )
        if all(name in group for name in required_pf_columns):
            feng_scene_criteria = (
                (group.area_km2.to_numpy(dtype=float) > FENG_MCS_COLD_SHIELD_MIN_AREA_KM2)
                & (group.pf_major_axis_km.to_numpy(dtype=float) > FENG_MCS_PF_MIN_MAJOR_AXIS_KM)
                & (group.pf_area_km2.to_numpy(dtype=float) > pf_area_threshold)
                & (group.pf_mean_rain_mm_h.to_numpy(dtype=float) > pf_mean_rain_threshold)
                & (group.pf_rain_skewness.to_numpy(dtype=float) > pf_skew_threshold)
            )
        else:
            feng_scene_criteria = np.zeros(observations, dtype=bool)
        max_qualifying_run = 0
        current_run = 0
        previous_index = None
        for qualifies, time_index in zip(
            feng_scene_criteria, group.time_index.to_numpy(dtype=int)
        ):
            if qualifies and (
                previous_index is None or time_index == previous_index + 1
            ):
                current_run += 1
            elif qualifies:
                current_run = 1
            else:
                current_run = 0
            max_qualifying_run = max(max_qualifying_run, current_run)
            previous_index = time_index
        passes_feng_continuity = max_qualifying_run >= MIN_TRACK_OBSERVATIONS
        max_area = float(group.area_km2.max()) if "area_km2" in group else np.nan
        core_area_col = (
            "area_pyflextrkr_core_mask_km2"
            if "area_pyflextrkr_core_mask_km2" in group
            else "area_convective_precipitating_high_cloud_mask_km2"
        )
        max_core_area = (
            float(group[core_area_col].max())
            if core_area_col in group
            else 0.0
        )
        has_embedded_convective_core = bool(max_core_area > 0.0)
        core_fraction_col = (
            "frac_pyflextrkr_core_mask"
            if "frac_pyflextrkr_core_mask" in group
            else "frac_convective_precipitating_high_cloud_mask"
        )
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
                feng_mcs=False,
                persistent_dcs=False,
                rejection_reason="pending",
                persistent_dcs_rejection_reason="pending",
                observation_count=observations,
                max_consecutive_observed_scenes=int(max_observed_run),
                first_time_index=int(group.time_index.iloc[0]),
                last_time_index=int(group.time_index.iloc[-1]),
                observed_span_hours=observed_span_hours,
                sampling_support_hours=sampling_support_hours,
                feng_lifetime_hours=feng_lifetime_hours,
                feng_criteria_percentile=int(FENG_MCS_CRITERIA_PERCENTILE),
                qualifying_scene_count=int(np.count_nonzero(feng_scene_criteria)),
                max_consecutive_qualifying_scenes=int(max_qualifying_run),
                pf_area_threshold_km2=float(pf_area_threshold),
                pf_mean_rain_threshold_mm_h=float(pf_mean_rain_threshold),
                pf_skewness_threshold=float(pf_skew_threshold),
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
            row_reasons.append(f"fewer_than_{MIN_TRACK_OBSERVATIONS}_observations")
        if not np.any(feng_scene_criteria):
            row_reasons.append("no_scene_passes_feng_50km_pf_criteria")
        elif not passes_feng_continuity:
            row_reasons.append("feng_criteria_not_met_in_two_consecutive_scenes")
        feng_mcs = len(row_reasons) == 0
        rows[-1]["feng_mcs"] = feng_mcs
        # Retain ``accepted`` as a backwards-compatible alias for the strict,
        # published Feng population. It must never mean the broader DCS set.
        rows[-1]["accepted"] = feng_mcs
        rows[-1]["rejection_reason"] = (
            ";".join(row_reasons) if row_reasons else "accepted_feng_mcs"
        )

        dcs_reasons = []
        if max_observed_run < MIN_TRACK_OBSERVATIONS:
            dcs_reasons.append(
                f"fewer_than_{MIN_TRACK_OBSERVATIONS}_consecutive_observations"
            )
        if not has_embedded_convective_core:
            dcs_reasons.append("no_225K_convective_core_in_track")
        # The broader population is deliberately nested: every published Feng
        # MCS remains present even if an auxiliary core-area diagnostic is
        # missing, alongside the less intense persistent core-bearing DCSs.
        persistent_dcs = feng_mcs or len(dcs_reasons) == 0
        rows[-1]["persistent_dcs"] = persistent_dcs
        rows[-1]["persistent_dcs_rejection_reason"] = (
            "accepted_feng_mcs"
            if feng_mcs
            else (
                ";".join(dcs_reasons)
                if dcs_reasons else "accepted_persistent_dcs"
            )
        )
        rows[-1]["persistent_dcs_acceptance_criteria"] = (
            PERSISTENT_DCS_ACCEPTANCE_CRITERIA
        )
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


def retained_track_ids(manifest):
    """IDs retained in at least one downstream-analysis population."""
    if manifest is None or len(manifest) == 0:
        return set()
    retained = np.zeros(len(manifest), dtype=bool)
    for population in TRACK_POPULATIONS:
        retained |= manifest_population_mask(manifest, population).to_numpy()
    return set(manifest.loc[retained, "track_id"].to_numpy(dtype=int))


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
        persistent_dcs_acceptance_criteria=PERSISTENT_DCS_ACCEPTANCE_CRITERIA,
        track_populations=list(TRACK_POPULATIONS),
        tracking_mode=TRACKING_MODE,
        pyflextrkr_advection_median_filter_length=(
            int(PYFLEXTRKR_ADVECTION_MEDIAN_FILTER_LENGTH)
            if TRACKING_MODE == "drift" else None
        ),
        pyflextrkr_advection_max_movement_mps=(
            float(PYFLEXTRKR_ADVECTION_MAX_MOVEMENT_MPS)
            if TRACKING_MODE == "drift" else None
        ),
        object_geometry=ORGANISED_TRACK_OBJECT_GEOMETRY,
        envelope_min_raw_area_km2=float(ENVELOPE_MIN_RAW_AREA_KM2),
        pyflextrkr_tb_core_k=float(PYFLEXTRKR_TB_CORE_K),
        pyflextrkr_tb_cold_k=float(PYFLEXTRKR_TB_COLD_K),
        pyflextrkr_tb_warm_k=float(PYFLEXTRKR_TB_WARM_K),
        pyflextrkr_tb_cloud_k=float(PYFLEXTRKR_TB_CLOUD_K),
        pyflextrkr_tb_smoothing_cells=int(PYFLEXTRKR_TB_SMOOTHING_CELLS),
        pyflextrkr_min_cold_core_pixels=int(PYFLEXTRKR_MIN_COLD_CORE_PIXELS),
        pyflextrkr_warm_anvil_expansion=int(PYFLEXTRKR_WARM_ANVIL_EXPANSION),
        **feng_mcs_metadata(),
        track_manifest_file=PYFLEXTRKR_TRACK_MANIFEST_FILENAME,
        track_manifest_sha256=sha256_file(manifest_path),
        raw_track_count=int(len(track_manifest)),
        accepted_track_count=int(manifest_accepted_mask(track_manifest).sum()),
        persistent_dcs_track_count=int(
            manifest_population_mask(track_manifest, "persistent_dcs").sum()
        ),
        feng_mcs_track_count=int(
            manifest_population_mask(track_manifest, "feng_mcs").sum()
        ),
        retained_track_count=int(len(retained_track_ids(track_manifest))),
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
        persistent_dcs_acceptance_criteria=PERSISTENT_DCS_ACCEPTANCE_CRITERIA,
        track_populations=list(TRACK_POPULATIONS),
        tracking_mode=TRACKING_MODE,
        pyflextrkr_advection_median_filter_length=(
            int(PYFLEXTRKR_ADVECTION_MEDIAN_FILTER_LENGTH)
            if TRACKING_MODE == "drift" else None
        ),
        pyflextrkr_advection_max_movement_mps=(
            float(PYFLEXTRKR_ADVECTION_MAX_MOVEMENT_MPS)
            if TRACKING_MODE == "drift" else None
        ),
        object_geometry=ORGANISED_TRACK_OBJECT_GEOMETRY,
        envelope_min_raw_area_km2=float(ENVELOPE_MIN_RAW_AREA_KM2),
        pyflextrkr_tb_core_k=float(PYFLEXTRKR_TB_CORE_K),
        pyflextrkr_tb_cold_k=float(PYFLEXTRKR_TB_COLD_K),
        pyflextrkr_tb_warm_k=float(PYFLEXTRKR_TB_WARM_K),
        pyflextrkr_tb_cloud_k=float(PYFLEXTRKR_TB_CLOUD_K),
        pyflextrkr_tb_smoothing_cells=int(PYFLEXTRKR_TB_SMOOTHING_CELLS),
        pyflextrkr_min_cold_core_pixels=int(PYFLEXTRKR_MIN_COLD_CORE_PIXELS),
        pyflextrkr_warm_anvil_expansion=int(PYFLEXTRKR_WARM_ANVIL_EXPANSION),
        **feng_mcs_metadata(),
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


def cached_pyflextrkr_tracknumbers_are_reusable(
    work_path,
    tracknumbers_file,
    labels_by_time,
    mask,
    source_name,
    sim_name,
):
    """Reuse old selection-schema caches when the actual tracking is unchanged."""
    metadata = read_product_metadata(work_path)
    if metadata is None or not os.path.exists(tracknumbers_file):
        return False
    cached_tracking_mode = metadata.get("tracking_mode", "standard")
    expected = {
        "source": source_name,
        "simulation": sim_name,
        "classification_cache_version": CLASSIFICATION_CACHE_VERSION,
        "cloudid_labels_sha256": sha256_label_sequence(labels_by_time),
        "tracknumbers_file": os.path.basename(tracknumbers_file),
        "track_min_overlap_fraction": float(TRACK_MIN_OVERLAP_FRACTION),
        "track_min_area_km2": float(TRACK_MIN_AREA_KM2),
        **time_signature(mask.Time.values),
    }
    return (
        cached_tracking_mode == TRACKING_MODE
        and all(metadata.get(key) == value for key, value in expected.items())
        and metadata.get("tracknumbers_sha256") == sha256_file(tracknumbers_file)
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
    retained_ids = retained_track_ids(manifest)
    retained = (
        attached[attached.track_id.isin(retained_ids)]
        .sort_values(["track_id", "time_index", "area_km2"], ascending=[True, True, False])
        .drop_duplicates(["track_id", "time_index"])
        .copy()
    )
    membership = manifest.set_index("track_id")[["persistent_dcs", "feng_mcs"]]
    retained = retained.join(membership, on="track_id")
    retained["organised_track"] = True
    feng_count = int(manifest_population_mask(manifest, "feng_mcs").sum())
    dcs_count = int(manifest_population_mask(manifest, "persistent_dcs").sum())
    progress(
        f"{sim_name} {source_name}: retained {len(retained_ids)} of "
        f"{len(manifest)} raw tracks: {dcs_count} persistent DCS and "
        f"{feng_count} Feng-defined MCSs"
    )
    return retained


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

    drift_data = None

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
        progress(
            f"{sim_name} {source_name}: using cached PyFLEXTRKR track numbers "
            f"from {tracknumbers_file}"
        )
        return attach_and_record_pyflextrkr_product(
            work_path, tracknumbers_file, labels_by_time, objects_df, mask,
            source_name, sim_name,
        )

    if (
        not FORCE_RECOMPUTE_PYFLEXTRKR_CACHE
        and cached_pyflextrkr_tracknumbers_are_reusable(
            work_path,
            tracknumbers_file,
            labels_by_time,
            mask,
            source_name,
            sim_name,
        )
    ):
        progress(
            f"{sim_name} {source_name}: reusing unchanged PyFLEXTRKR links; "
            "rebuilding only the dual-population manifest"
        )
        return attach_and_record_pyflextrkr_product(
            work_path,
            tracknumbers_file,
            labels_by_time,
            objects_df,
            mask,
            source_name,
            sim_name,
        )

    if os.path.exists(tracknumbers_file):
        print(
            f"Regenerating stale PyFLEXTRKR product for {sim_name} {source_name}: "
            f"{tracknumbers_file}"
        )
        os.remove(tracknumbers_file)

    files, basetimes, tracking_path = write_pyflextrkr_cloudid_files(
        labels_by_time,
        mask,
        work_path,
        progress_label=f"{sim_name} {source_name}: cloud-ID files",
    )
    if TRACKING_MODE == "drift":
        # PyFLEXTRKR's supported motion-corrected linker estimates one mean
        # displacement per adjacent pair, shifts the reference labels, then
        # applies the unchanged 0.5 bidirectional-overlap test. This remains a
        # sensitivity mode because a single tropical-domain drift cannot
        # represent systems propagating in different directions.
        cell_area = grid_cell_area_km2(
            mask.latitude.values, mask.longitude.values
        )
        pixel_radius_km = float(np.sqrt(np.nanmedian(cell_area.values)))
        config.update(
            ref_varname="feature_number",
            advection_field_threshold=0.5,
            advection_mask_method="greater",
            pixel_radius=pixel_radius_km,
            advection_tiles=[1, 1],
            advection_med_filt_len=int(
                PYFLEXTRKR_ADVECTION_MEDIAN_FILTER_LENGTH
            ),
            advection_max_movement_mps=float(
                PYFLEXTRKR_ADVECTION_MAX_MOVEMENT_MPS
            ),
            advection_buffer=10,
            advection_size_threshold=10,
            datatimeresolution=float(median_time_step_days(mask) * 24.0),
            run_parallel=0,
        )
        progress(
            f"{sim_name} {source_name}: estimating PyFLEXTRKR domain-mean "
            "adjacent-scene drift (sensitivity mode)"
        )
        drift_file = calc_mean_advection(config)
        date_strings = [
            pd.to_datetime(float(value), unit="s").strftime("%Y%m%d")
            for value in basetimes
        ]
        time_strings = [
            pd.to_datetime(float(value), unit="s").strftime("%H%M%S")
            for value in basetimes
        ]
        drift_datetimes, xdrifts, ydrifts = match_drift_times(
            date_strings, time_strings, driftfile=drift_file
        )
        drift_data = list(zip(drift_datetimes, xdrifts, ydrifts))
        # Some netCDF4/HDF5 builds print thousands of harmless missing-
        # quantization-attribute diagnostics when tracksingle_drift opens the
        # cloud-ID files. Silence only HDF5's C error stack; Python exceptions
        # from genuine file or tracking failures still propagate normally.
        try:
            import h5py
            h5py._errors.silence_errors()
        except (ImportError, AttributeError):
            pass
        pd.DataFrame(
            {
                "reference_time": pd.to_datetime(
                    np.asarray(basetimes[:-1], dtype=float), unit="s"
                ),
                "x_shift_grid_cells": xdrifts,
                "y_shift_grid_cells": ydrifts,
                "pixel_radius_km": pixel_radius_km,
                "tracking_mode": TRACKING_MODE,
            }
        ).to_csv(
            os.path.join(work_path, "pyflextrkr_advection_drift.csv"),
            index=False,
        )
    old_cwd = os.getcwd()
    pair_count = len(files) - 1
    pair_started = time.monotonic()
    progress(
        f"{sim_name} {source_name}: tracking {pair_count} adjacent-scene pairs"
    )
    try:
        os.chdir(tracking_path)
        for it in range(pair_count):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Converting non-nanosecond precision datetime values.*",
                    category=UserWarning,
                )
                pair = (
                    os.path.basename(files[it]),
                    os.path.basename(files[it + 1]),
                )
                pair_times = (basetimes[it], basetimes[it + 1])
                if TRACKING_MODE == "drift":
                    pyflextrkr_trackclouds_drift(
                        pair, pair_times, config, drift_data=drift_data[it]
                    )
                else:
                    pyflextrkr_trackclouds(pair, pair_times, config)
            if TRACKING_MODE == "standard":
                short_name = os.path.join(
                    tracking_path,
                    pyflextrkr_track_filename(basetimes[it + 1]),
                )
                seconds_name = os.path.join(
                    tracking_path,
                    pyflextrkr_track_filename_with_seconds(basetimes[it + 1]),
                )
                if os.path.exists(seconds_name):
                    os.remove(seconds_name)
                if os.path.exists(short_name):
                    os.replace(short_name, seconds_name)
            report_loop_progress(
                f"{sim_name} {source_name}: adjacent-scene tracking",
                it + 1,
                pair_count,
                pair_started,
            )
    finally:
        os.chdir(old_cwd)

    progress(f"{sim_name} {source_name}: assembling PyFLEXTRKR track numbers")
    tracknumbers_file = pyflextrkr_gettracknumbers(config)
    progress(f"{sim_name} {source_name}: PyFLEXTRKR track numbers ready")
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
        return pd.DataFrame(), pd.DataFrame()

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
    scene_count = mask.sizes["Time"]
    scene_started = time.monotonic()
    progress(
        f"{sim_name} {source_name}: extracting objects and PF properties from "
        f"{scene_count} scenes"
    )

    for it in range(scene_count):
        labels, objects = label_objects_with_properties(
            mask.isel(Time=it).load(),
            area2d,
            precip2d=precip.isel(Time=it).load(),
            field_slices={
                key: da.isel(Time=it).load()
                for key, da in context_masks.items()
                if key in (
                    "pyflextrkr_core_mask",
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
        report_loop_progress(
            f"{sim_name} {source_name}: scene object extraction",
            it + 1,
            scene_count,
            scene_started,
        )

    objects_df = pd.DataFrame(rows)
    if len(objects_df) == 0:
        progress(f"{sim_name} {source_name}: no trackable objects found")
        return objects_df, pd.DataFrame()

    progress(
        f"{sim_name} {source_name}: extracted {len(objects_df)} scene objects; "
        "starting overlap tracking"
    )
    objects_df = assign_pyflextrkr_track_ids(labels_by_time, objects_df, mask, source_name, sim_name)
    if len(objects_df) == 0:
        progress(f"{sim_name} {source_name}: no objects remained after tracking")
        return objects_df, pd.DataFrame()

    track_time_agg = dict(
        Time=("Time", "first"),
        time_days=("time_days", "first"),
        dt_hours=("dt_hours", "first"),
        area_km2=("area_km2", "sum"),
        precip_area_rate=("precip_area_rate", "sum"),
        mean_precip=("mean_precip", "mean"),
        max_precip=("max_precip", "max"),
        pf_area_km2=("pf_area_km2", "max"),
        pf_major_axis_km=("pf_major_axis_km", "max"),
        pf_mean_rain_mm_h=("pf_mean_rain_mm_h", "mean"),
        pf_max_rain_mm_h=("pf_max_rain_mm_h", "max"),
        pf_rain_skewness=("pf_rain_skewness", "mean"),
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
    retained_quality = build_organised_track_manifest(objects_df)
    retained_membership = np.zeros(len(retained_quality), dtype=bool)
    for population in TRACK_POPULATIONS:
        retained_membership |= manifest_population_mask(
            retained_quality, population
        ).to_numpy()
    if not retained_membership.all():
        raise RuntimeError(
            f"{sim_name}: rejected PyFLEXTRKR candidates leaked into the "
            "retained dual-population track tables"
        )
    retained_quality = retained_quality.set_index("track_id")

    tracks = []
    for (source, label, simulation, track_id), group in track_time.groupby(
        ["source", "source_label", "simulation", "track_id"]
    ):
        group = group.sort_values("time_index")
        lifetime_steps = int(group.time_index.nunique())
        quality = retained_quality.loc[int(track_id)]
        lifetime_hours = float(quality.feng_lifetime_hours)
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
                persistent_dcs=bool(quality.persistent_dcs),
                feng_mcs=bool(quality.feng_mcs),
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
                max_pf_area_km2=float(group.pf_area_km2.max()),
                mean_pf_area_km2=float(group.pf_area_km2.mean()),
                max_pf_major_axis_km=float(group.pf_major_axis_km.max()),
                mean_pf_rain_mm_h=float(group.pf_mean_rain_mm_h.mean()),
                max_pf_rain_mm_h=float(group.pf_max_rain_mm_h.max()),
                mean_pf_rain_skewness=float(group.pf_rain_skewness.mean()),
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
    progress(
        f"{sim_name} {source_name}: track tables ready with "
        f"{len(objects_df)} retained scene objects and {len(tracks_df)} retained tracks"
    )
    return objects_df, tracks_df


def build_track_persistence_metrics(masks_by_run, fields_by_run):
    object_rows = []
    track_rows = []
    task_count = sum(
        name in masks
        for masks in masks_by_run
        for name in PYFLEXTRKR_TRACKED_SOURCES
    )
    task_index = 0
    started = time.monotonic()
    progress(f"Computing dual-population track products for {task_count} simulation/source tasks")
    for i, masks in enumerate(masks_by_run):
        sim_name = sim_names[i]
        for source_name in [name for name in PYFLEXTRKR_TRACKED_SOURCES if name in masks]:
            task_index += 1
            progress(
                f"Tracking task {task_index}/{task_count}: {sim_name} {source_name}"
            )
            objects, tracks = build_track_tables(
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
            report_loop_progress(
                "MCS simulation/source tasks",
                task_index,
                task_count,
                started,
            )

    objects_df = pd.concat(object_rows, ignore_index=True) if object_rows else pd.DataFrame()
    tracks_df = pd.concat(track_rows, ignore_index=True) if track_rows else pd.DataFrame()
    progress(
        f"Dual-population track calculation complete: {len(objects_df)} retained "
        f"scene objects, {len(tracks_df)} retained tracks"
    )
    return objects_df, tracks_df


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
        persistent = manifest_population_mask(group, "persistent_dcs")
        retained = accepted | persistent
        reasons = group.loc[~accepted, "rejection_reason"].astype(str)
        summary_rows.append(
            dict(
                simulation=simulation,
                raw_track_count=int(len(group)),
                accepted_track_count=int(accepted.sum()),
                rejected_track_count=int((~accepted).sum()),
                accepted_fraction=float(accepted.mean()) if len(group) else np.nan,
                feng_mcs_track_count=int(accepted.sum()),
                persistent_dcs_track_count=int(persistent.sum()),
                persistent_dcs_not_feng_count=int((persistent & ~accepted).sum()),
                retained_track_count=int(retained.sum()),
                retained_fraction=float(retained.mean()) if len(group) else np.nan,
                rejected_single_snapshot_count=int(
                    reasons.str.contains(
                        f"fewer_than_{MIN_TRACK_OBSERVATIONS}_observations",
                        regex=False,
                    ).sum()
                ),
                rejected_no_qualifying_feng_scene_count=int(
                    reasons.str.contains("no_scene_passes_feng_50km_pf_criteria", regex=False).sum()
                ),
                rejected_noncontinuous_feng_criteria_count=int(
                    reasons.str.contains("feng_criteria_not_met_in_two_consecutive_scenes", regex=False).sum()
                ),
                min_track_observations=int(MIN_TRACK_OBSERVATIONS),
                min_track_duration_hours=float(MIN_TRACK_DURATION_HOURS),
                track_acceptance_criteria=ORGANISED_TRACK_ACCEPTANCE_CRITERIA,
                embedded_convective_core_requirement=ORGANISED_TRACK_EMBEDDED_CORE_REQUIREMENT,
                persistent_dcs_acceptance_criteria=PERSISTENT_DCS_ACCEPTANCE_CRITERIA,
                tracking_mode=TRACKING_MODE,
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(save_loc, "pyflextrkr_organised_track_acceptance_summary.csv")
    summary.to_csv(summary_path, index=False)
    if TRACKING_MODE == "drift":
        standard_directory = (
            save_loc[:-len("_drift")]
            if TEST_MODE and save_loc.endswith("_drift")
            else default_save_loc
        )
        standard_path = os.path.join(
            standard_directory,
            "pyflextrkr_organised_track_acceptance_summary.csv",
        )
        if os.path.exists(standard_path):
            standard = pd.read_csv(standard_path).copy()
            standard["comparison_tracking_mode"] = "standard"
            current = summary.copy()
            current["comparison_tracking_mode"] = "drift"
            comparison = pd.concat([standard, current], ignore_index=True)
            for metric in (
                "raw_track_count",
                "persistent_dcs_track_count",
                "feng_mcs_track_count",
                "retained_track_count",
            ):
                baseline = standard.set_index("simulation")[metric]
                comparison[f"{metric}_difference_from_standard"] = [
                    float(getattr(row, metric) - baseline.get(row.simulation, np.nan))
                    for row in comparison.itertuples(index=False)
                ]
            comparison_path = os.path.join(
                save_loc, "pyflextrkr_tracking_mode_comparison.csv"
            )
            comparison.to_csv(comparison_path, index=False)
            print(
                "Wrote standard-versus-drift tracking comparison: "
                f"{comparison_path}"
            )
    print(f"Wrote organised-track acceptance reports: {combined_path}, {summary_path}")
    return combined, summary


# -----------------------------------------------------------------------------
# Cloud-type masks
# -----------------------------------------------------------------------------


def has_primary_cloud_masks(masks):
    return all(key in masks for key in CLOUD_OBJECT_KEYS_PRIMARY)


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

    progress(f"{sim_name}: using cached high-cloud classification from {path}")
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
    corrected_warm, reconstructed = corrected_cached_warm_shield_masks(cached)
    if reconstructed:
        progress(
            f"{sim_name}: reconstructing corrected warm-shield validation "
            "masks lazily; retaining the expensive classification cache"
        )
    masks = {}
    for key in CLASSIFICATION_DIAGNOSTIC_KEYS:
        da = corrected_warm.get(key, cached[key])
        da = da.sel(Time=precip_ref.Time)
        masks[key] = da.fillna(False).astype(bool) if key.endswith("_mask") else da
    return masks


def cloud_type_masks(ds, fields, sim_name):
    if not ENABLE_CLOUD_OBJECTS:
        return {}

    masks = cloud_masks_from_cache(sim_name, fields["precip"])
    if not has_primary_cloud_masks(masks):
        raise RuntimeError(f"{sim_name}: shared high-cloud classification cache lacks primary cloud-object masks")
    return masks


def build_organised_convective_system_mask(fields, masks):
    """Reuse the exact cold-shield mask written by the shared classifier."""
    key = "pyflextrkr_cold_cloud_shield_mask"
    if key not in masks:
        raise RuntimeError(
            "The shared classification cache lacks the PyFLEXTRKR cold-cloud "
            "shield. Rebuild it with refined_high_cloud_classification_diagnostics.py."
        )
    envelope, _precip = xr.align(
        masks[key].fillna(False).astype(bool), fields["precip"], join="inner"
    )
    envelope = subset_lat(envelope, lat_bounds=TROPICAL_LAT_BOUNDS)
    envelope.name = ORGANISED_SYSTEM_SOURCE
    envelope.attrs.update(
        long_name="organised convective-system envelope mask",
        description=(
            "Exact shared PyFLEXTRKR OLR-derived Tb cold shield: 225-K core "
            "detection spread to the connected 241-K shield. Feng et al. MCS "
            "criteria are applied after overlap tracking."
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
        shared_cache_variable=key,
    )
    return envelope.astype(bool)


# -----------------------------------------------------------------------------
# Threshold masks and summaries
# -----------------------------------------------------------------------------


def build_threshold_table():
    rows = [
        ("OLR-derived Tb core", f"< {PYFLEXTRKR_TB_CORE_K:g} K", "scene segmentation"),
        ("connected cold-cloud shield", f"< {PYFLEXTRKR_TB_COLD_K:g} K", "scene segmentation"),
        ("persistent DCS nesting", "all Feng-defined MCSs are included", "broader population"),
        ("persistent DCS duration", f">= {MIN_TRACK_OBSERVATIONS} consecutive six-hour scenes", "broader population"),
        ("persistent DCS core", f">= 1 scene containing a {PYFLEXTRKR_TB_CORE_K:g}-K core", "broader population"),
        ("adjacent-scene link", f">= {TRACK_MIN_OVERLAP_FRACTION:g} forward or backward overlap", f"PyFLEXTRKR {TRACKING_MODE} tracking"),
        ("cold-shield area", f"> {FENG_MCS_COLD_SHIELD_MIN_AREA_KM2:g} km2", "MCS acceptance"),
        ("PF rain definition", f">= {FENG_MCS_PF_MIN_RAIN_MM_H:g} mm h-1", "PF identification"),
        ("PF major axis", f"> {FENG_MCS_PF_MIN_MAJOR_AXIS_KM:g} km", "MCS acceptance"),
        ("PF area", f"> {FENG_MCS_PF_AREA_INTERCEPT_KM2:g} + {FENG_MCS_PF_AREA_SLOPE_KM2_PER_H:g} * lifetime_h km2", "MCS acceptance"),
        ("PF mean rain", f"> {FENG_MCS_PF_MEAN_RAIN_INTERCEPT_MM_H:g} + {FENG_MCS_PF_MEAN_RAIN_SLOPE_MM_H_PER_H:g} * lifetime_h mm h-1", "MCS acceptance"),
        ("PF rain skewness", f"> {FENG_MCS_PF_SKEW_INTERCEPT:g} + {FENG_MCS_PF_SKEW_SLOPE_PER_H:g} * lifetime_h", "MCS acceptance"),
        ("criteria continuity", f">= {MIN_TRACK_OBSERVATIONS} consecutive six-hour scenes", "six-hourly adaptation of >=6 h continuity"),
    ]
    return pd.DataFrame([
        dict(
            source=ORGANISED_SYSTEM_SOURCE,
            field=field,
            threshold=threshold,
            threshold_type=role,
            percentile=FENG_MCS_CRITERIA_PERCENTILE if "PF " in field and field != "PF rain definition" else np.nan,
            role=role,
            literature_basis=(
                "Feng et al. (2023), GMD, PyFLEXTRKR DCS-first tracking"
                if "persistent DCS" in field or "adjacent-scene" in field
                else "Feng et al. (2021), doi:10.1175/JCLI-D-20-0136.1; 50-km configuration"
            ),
        )
        for field, threshold, role in rows
    ])


def build_fields(ds):
    precip = get_precip_rate(ds)
    return {
        "precip": precip,
        "mean_precip": weighted_tropical_mean(precip, lat_bounds=TROPICAL_LAT_BOUNDS),
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _event_bootstrap_interval(values, statistic=np.nanmedian, seed=20240720, n_boot=1000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for iboot in range(n_boot):
        boot[iboot] = statistic(values[rng.integers(0, values.size, values.size)])
    return float(statistic(values)), *np.nanpercentile(boot, [2.5, 97.5]).tolist()


def population_exposure(sim_objects, fields):
    """Return full analysed exposure and population cold-shield coverage."""
    precip = subset_lat(fields["precip"], lat_bounds=TROPICAL_LAT_BOUNDS)
    analysed_steps = int(precip.sizes.get("Time", 0))
    step_hours = 24.0 * median_time_step_days(precip) if analysed_steps else np.nan
    total_hours = analysed_steps * step_hours if np.isfinite(step_hours) else np.nan
    duration_days = total_hours / 24.0 if total_hours > 0 else np.nan
    domain_area = float(
        grid_cell_area_km2(
            precip.latitude.values,
            precip.longitude.values,
        ).sum()
    )
    area_time = (
        float(np.nansum(sim_objects.area_km2 * sim_objects.dt_hours))
        if len(sim_objects)
        else 0.0
    )
    coverage = (
        area_time / (domain_area * total_hours)
        if domain_area > 0 and total_hours > 0
        else np.nan
    )
    return dict(
        duration_days=duration_days,
        total_hours=total_hours,
        domain_area_km2=domain_area,
        cold_shield_area_time_km2_h=area_time,
        mcs_coverage=coverage,
    )


def plot_convective_system_population_properties(objects_df, tracks_df, fields_by_run):
    """Plot the same event-equal summary for both retained populations."""
    if not should_run_figure(1):
        return
    if (
        tracks_df is None
        or len(tracks_df) == 0
        or "source" not in tracks_df.columns
    ):
        print("Skipping Figure 1: no retained convective-system tracks")
        return
    tracks = tracks_df[tracks_df.source == ORGANISED_SYSTEM_SOURCE].copy()
    objects = (
        objects_df[objects_df.source == ORGANISED_SYSTEM_SOURCE].copy()
        if objects_df is not None
        and len(objects_df) > 0
        and "source" in objects_df.columns
        else pd.DataFrame()
    )
    if len(tracks) == 0:
        print("Skipping Figure 1: no retained convective-system tracks")
        return
    required = {
        "lifetime_hours", "max_area_km2", "max_pf_area_km2",
        "mean_pf_rain_mm_h", "mean_pf_rain_skewness",
    }
    missing = sorted(required.difference(tracks.columns))
    if missing:
        raise RuntimeError(
            "Track table predates the dual-population/PF schema; rebuild the "
            f"organisation cache (missing {missing})"
        )

    summary_rows = []
    event_metrics = (
        ("lifetime_hours", "lifetime", "h"),
        ("max_area_km2", "maximum cold-shield area", "km2"),
        ("max_pf_area_km2", "maximum PF area", "km2"),
        ("mean_pf_rain_mm_h", "mean PF rain rate", "mm h-1"),
        ("mean_pf_rain_skewness", "PF rain-rate skewness", "1"),
        ("max_translation_speed_kmh", "maximum translation speed", "km h-1"),
    )
    for population in TRACK_POPULATIONS:
        population_label = TRACK_POPULATION_LABELS[population]
        population_tracks = tracks[
            manifest_population_mask(tracks, population)
        ].copy()
        population_objects = objects[
            manifest_population_mask(objects, population)
        ].copy() if len(objects) else objects
        if len(population_tracks) == 0:
            progress(f"Figure 1: no {population_label.lower()} to plot")
            continue

        rates = []
        coverage = []
        for isim, (sim_name, fields) in enumerate(zip(sim_names, fields_by_run)):
            progress(
                f"Figure 1 ({population_label}): summarising {sim_name} "
                f"({isim + 1}/{len(sim_names)})"
            )
            sim_tracks = population_tracks[
                population_tracks.simulation == sim_name
            ]
            sim_objects = population_objects[
                population_objects.simulation == sim_name
            ]
            exposure = population_exposure(sim_objects, fields)
            duration_days = exposure["duration_days"]
            shield_coverage = exposure["mcs_coverage"]
            event_rate = (
                100.0 * len(sim_tracks) / duration_days
                if duration_days > 0 else np.nan
            )
            rates.append(event_rate)
            coverage.append(shield_coverage)
            common = dict(
                track_population=population,
                track_population_label=population_label,
                simulation=sim_name,
                analysed_time_days=duration_days,
                event_count=int(len(sim_tracks)),
            )
            summary_rows.append(dict(
                **common, metric="event rate", value=event_rate,
                ci_low=np.nan, ci_high=np.nan, units="events per 100 days",
                denominator="population tracks divided by the full analysed period",
            ))
            summary_rows.append(dict(
                **common, metric="cold-shield coverage", value=shield_coverage,
                ci_low=np.nan, ci_high=np.nan, units="fraction",
                denominator="population cold-shield area-time divided by full tropical domain area-time",
            ))
            for column, metric, units in event_metrics:
                value, lo, hi = _event_bootstrap_interval(
                    sim_tracks[column], seed=20240720 + isim
                )
                summary_rows.append(dict(
                    **common, metric=f"median {metric}", value=value,
                    ci_low=lo, ci_high=hi, units=units,
                    denominator="one equally weighted track; bootstrap resamples tracks",
                ))

        fig, axes = plt.subplots(
            2, 2, figsize=(11.4, 8.4), constrained_layout=True
        )
        xpos = np.arange(len(sim_names))
        axes[0, 0].bar(xpos - 0.18, rates, width=0.36, color=cols)
        coverage_axis = axes[0, 0].twinx()
        coverage_axis.bar(
            xpos + 0.18, np.asarray(coverage) * 100.0,
            width=0.36, color=cols, alpha=0.45,
        )
        axes[0, 0].set(
            xticks=xpos, xticklabels=sim_names,
            ylabel="Events per 100 days", title="(a) Frequency and coverage",
        )
        coverage_axis.set_ylabel("Cold-shield area-time (%)")

        lifetime_values = [
            population_tracks.loc[
                population_tracks.simulation == name, "lifetime_hours"
            ].dropna().values
            for name in sim_names
        ]
        axes[0, 1].boxplot(lifetime_values, labels=sim_names, showfliers=False)
        axes[0, 1].set(
            ylabel="Lifetime (h)",
            title="(b) Lifetime (one value per track)",
        )
        for isim, sim_name in enumerate(sim_names):
            sub = population_tracks[population_tracks.simulation == sim_name]
            axes[1, 0].scatter(
                sub.max_area_km2, sub.max_pf_area_km2,
                s=18, alpha=0.55, color=cols[isim], label=sim_name,
            )
            axes[1, 1].scatter(
                sub.mean_pf_rain_mm_h, sub.mean_pf_rain_skewness,
                s=np.clip(sub.lifetime_hours, 8, 90), alpha=0.5,
                color=cols[isim], label=sim_name,
            )
        axes[1, 0].set(
            xscale="log", yscale="log",
            xlabel="Maximum cold-shield area (km$^2$)",
            ylabel="Maximum PF area (km$^2$)",
            title="(c) Cloud-shield / PF size",
        )
        axes[1, 1].set(
            xlabel="Mean PF rain rate (mm h$^{-1}$)",
            ylabel="PF rain-rate skewness",
            title="(d) Precipitation-feature rain distribution\n(marker size: lifetime)",
        )
        for ax in axes.ravel():
            ax.grid(alpha=0.2)
        axes[1, 0].legend(frameon=False, fontsize=8)
        axes[1, 1].legend(frameon=False, fontsize=8)
        method_line = (
            "Feng MCSs plus other 241-K shield tracks with at least two "
            "consecutive 6-h scenes and a 225-K core"
            if population == "persistent_dcs"
            else "Feng et al. (2021) 50-km CCS/PF criteria"
        )
        fig.suptitle(
            f"{population_label}: population and properties\n{method_line}; "
            "track-property panels weight each event equally"
        )
        save_figure(1, fig, f"{population}_population_properties")
        plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(save_loc, "convective_system_population_property_summary.csv"),
        index=False,
    )


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
      1: Feng-defined MCS track
      2: persistent deep-convective-system track that is not a Feng MCS
      3: neither retained population, or not linked into a valid track
    """
    labels = np.asarray(labels, dtype=np.int32)
    status = np.zeros(labels.shape, dtype=np.int8)
    work_path = pyflextrkr_work_path(source_name, sim_name)
    manifest = read_track_manifest(work_path)
    metadata = read_product_metadata(work_path)
    if manifest is None or metadata is None:
        status[labels > 0] = 3
        return status, set(), set(), set()

    feng = manifest_population_mask(manifest, "feng_mcs")
    persistent = manifest_population_mask(manifest, "persistent_dcs")
    feng_ids = set(manifest.loc[feng, "track_id"].astype(int))
    persistent_only_ids = set(
        manifest.loc[persistent & ~feng, "track_id"].astype(int)
    )
    rejected_ids = set(
        manifest.loc[~persistent & ~feng, "track_id"].astype(int)
    )

    track_grid = pyflextrkr_scene_track_grid(source_name, sim_name, time_index, labels)
    candidate = labels > 0
    status[candidate] = 3
    if persistent_only_ids:
        status[np.isin(track_grid, list(persistent_only_ids)) & candidate] = 2
    if feng_ids:
        status[np.isin(track_grid, list(feng_ids)) & candidate] = 1
    return status, feng_ids, persistent_only_ids, rejected_ids


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
            0.5, 0.5, "No retained tracks", transform=ax.transAxes,
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
        "All retained centroid tracks over "
        f"{sim_objects.time_index.nunique()} snapshots{highlight_note}; "
        f"labels are track IDs{suffix}"
    )


def plot_pyflextrkr_track_validation(objects_df, fields_by_run, masks_by_run):
    if not should_run_figure(2):
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
        print("Skipping Figure 2: no current PyFLEXTRKR source masks are available.")
        return

    validation_rows = []
    precip_cmap, precip_norm = precip_rate_cmap_norm()
    status_handles = [
        Patch(facecolor="#1a9850", alpha=0.52, label="Feng-defined MCS tracks"),
        Patch(
            facecolor="#2b8cbe",
            alpha=0.46,
            label="Persistent deep-convective systems (not Feng MCSs)",
        ),
        Patch(facecolor="#fdae61", alpha=0.42, label="Other candidate or unlinked"),
        Patch(
            facecolor="0.65",
            alpha=0.45,
            label="Feng MCS fill uses matching track-ID colour",
        ),
        Line2D([0], [0], color="#c51b8a", lw=1.0, label="PyFLEXTRKR 225-K core"),
        Line2D([0], [0], color="tab:blue", marker="o", markersize=3, lw=1.0, label="Retained-track centroid path and ID"),
    ]
    for source_name in sources:
        for c, sim_name in enumerate(sim_names):
            progress(
                f"Figure 2 {source_label(source_name)}: preparing {sim_name} figure "
                f"({c + 1}/{len(sim_names)})"
            )
            fig = plt.figure(figsize=(14.5, 9.0), constrained_layout=True)
            grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.25))
            ax_precip = fig.add_subplot(grid[0, 0])
            ax_ctp = fig.add_subplot(grid[0, 1])
            ax_tracks = fig.add_subplot(grid[1, :])
            last_precip_im = None
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
                fig.suptitle(f"{sim_name}: no current PyFLEXTRKR scene")
                safe_sim = sim_name.replace(" ", "_")
                save_figure(
                    2,
                    fig,
                    f"pyflextrkr_track_validation_{source_name}_{safe_sim}",
                )
                plt.close(fig)
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
                    snapshot_selection = "retained_label_count"
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
            retained_scene = (
                sub[sub.time_index == time_index].copy()
                if len(sub) and "time_index" in sub.columns
                else pd.DataFrame()
            )
            status, _feng_ids, _persistent_only_ids, _rejected_ids = pyflextrkr_scene_label_status(
                source_name, sim_name, time_index, labels
            )
            track_grid = pyflextrkr_scene_track_grid(
                source_name, sim_name, time_index, labels
            )
            accepted_values = status == 1
            short_values = status == 2
            other_rejected_values = status == 3
            candidate_values = labels > 0
            overlay_status_mask(ax_precip, lon, lat, other_rejected_values, "#fdae61", 0.42)
            overlay_status_mask(ax_precip, lon, lat, short_values, "#2b8cbe", 0.46)
            overlay_status_mask(ax_precip, lon, lat, accepted_values, "#1a9850", 0.52)
            if np.any(accepted_values):
                ax_precip.contour(
                    lon, lat, accepted_values.astype(float), levels=[0.5],
                    colors="0.05", linewidths=1.0, zorder=8,
                )
            if np.any(short_values):
                ax_precip.contour(
                    lon, lat, short_values.astype(float), levels=[0.5],
                    colors="#0868ac", linewidths=0.8, zorder=7,
                )
            if np.any(other_rejected_values):
                ax_precip.contour(
                    lon, lat, other_rejected_values.astype(float), levels=[0.5],
                    colors="#d95f02", linewidths=0.8, zorder=7,
                )
            if np.any(candidate_values):
                overlay_status_mask(
                    ax_ctp, lon, lat, other_rejected_values, "#fdae61", 0.34
                )
                overlay_status_mask(
                    ax_ctp, lon, lat, short_values, "#2b8cbe", 0.38
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
            feng_continuity_rejected_label_count = int(
                np.unique(labels[short_values]).size
            )
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
                    feng_mcs_candidate_label_count=accepted_candidate_label_count,
                    persistent_dcs_only_candidate_label_count=(
                        feng_continuity_rejected_label_count
                    ),
                    feng_continuity_rejected_label_count=feng_continuity_rejected_label_count,
                    other_rejected_label_count=other_rejected_label_count,
                    snapshot_selection=snapshot_selection,
                    source_mask_in_accepted_tracks_percent=organised_source_percent,
                )
            )

            scene = (
                retained_scene.nlargest(12, "area_km2")
                if len(retained_scene) and "area_km2" in retained_scene.columns
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
                        row_type="retained_object",
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
                        feng_mcs_candidate_label_count=accepted_candidate_label_count,
                        persistent_dcs_only_candidate_label_count=(
                            feng_continuity_rejected_label_count
                        ),
                        feng_continuity_rejected_label_count=feng_continuity_rejected_label_count,
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
                f"Feng MCS {accepted_candidate_label_count}; persistent DCS only "
                f"{feng_continuity_rejected_label_count}; other {other_rejected_label_count}"
            )
            ax_ctp.set_title(
                "PyFLEXTRKR candidates: Feng MCS objects use track-ID colours\n"
                f"{raw_candidate_label_count} candidates; Tb core/cold/warm <= "
                f"{PYFLEXTRKR_TB_CORE_K:g}/{PYFLEXTRKR_TB_COLD_K:g}/{PYFLEXTRKR_TB_WARM_K:g} K"
            )

            if last_precip_im is not None:
                fig.colorbar(
                    last_precip_im,
                    ax=ax_precip,
                    orientation="horizontal",
                    label=r"Precipitation (mm day$^{-1}$)",
                    shrink=0.58,
                    aspect=30,
                    pad=0.14,
                )
            fig.suptitle(
                f"{sim_name}: PyFLEXTRKR cold-cloud candidates and retained populations\n"
                "Green: Feng 50-km MCS; blue: persistent core-bearing DCS not meeting Feng PF criteria",
                fontsize=12,
            )
            fig.legend(
                handles=status_handles,
                loc="outside lower center",
                ncol=3,
                frameon=True,
                fontsize=7,
            )
            safe_sim = sim_name.replace(" ", "_")
            save_figure(
                2,
                fig,
                f"pyflextrkr_track_validation_{source_name}_{safe_sim}",
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


def pyflextrkr_object_count_timeline():
    """Count raw, persistent-DCS, and Feng-MCS objects in every scene."""
    rows = []
    for isim, sim_name in enumerate(sim_names):
        files = pyflextrkr_cloudid_files(ORGANISED_SYSTEM_SOURCE, sim_name)
        work_path = pyflextrkr_work_path(ORGANISED_SYSTEM_SOURCE, sim_name)
        manifest = read_track_manifest(work_path)
        metadata = read_product_metadata(work_path)
        if not files or manifest is None or metadata is None:
            continue
        tracknumbers_name = metadata.get("tracknumbers_file", "")
        tracknumbers_path = os.path.join(work_path, "stats", tracknumbers_name)
        if not tracknumbers_name or not os.path.exists(tracknumbers_path):
            continue
        with xr.open_dataset(
            tracknumbers_path,
            decode_times=False,
            mask_and_scale=False,
        ) as track_ds:
            track_numbers = np.asarray(track_ds["track_numbers"].squeeze().values)
        if track_numbers.ndim != 2:
            continue

        feng = manifest_population_mask(manifest, "feng_mcs")
        persistent = manifest_population_mask(manifest, "persistent_dcs")
        feng_ids = set(manifest.loc[feng, "track_id"].astype(int))
        persistent_ids = set(manifest.loc[persistent, "track_id"].astype(int))
        scene_count = min(len(files), track_numbers.shape[0])
        started = time.monotonic()
        progress(
            f"Figure 3 timeline: counting {sim_name} objects in "
            f"{scene_count} scenes"
        )
        for time_index in range(scene_count):
            with xr.open_dataset(
                files[time_index],
                decode_times=False,
                mask_and_scale=False,
            ) as cloud:
                labels = np.asarray(
                    cloud["feature_number"].squeeze().values,
                    dtype=np.int32,
                )
                base_time = float(np.asarray(cloud["base_time"]).squeeze())
            object_labels = np.unique(labels[labels > 0])
            valid = (
                (object_labels > 0)
                & (object_labels <= track_numbers.shape[1])
            )
            object_track_ids = np.full(
                object_labels.shape,
                PYFLEXTRKR_FILLVAL,
                dtype=np.int64,
            )
            object_track_ids[valid] = track_numbers[
                time_index,
                object_labels[valid] - 1,
            ].astype(np.int64)
            feng_objects = np.isin(object_track_ids, list(feng_ids))
            persistent_objects = np.isin(
                object_track_ids, list(persistent_ids)
            )
            retained_objects = feng_objects | persistent_objects
            raw_count = int(object_labels.size)
            feng_count = int(np.count_nonzero(feng_objects))
            persistent_count = int(np.count_nonzero(persistent_objects))
            retained_count = int(np.count_nonzero(retained_objects))
            rows.append(
                dict(
                    simulation=sim_name,
                    time_index=time_index,
                    Time=pd.to_datetime(base_time, unit="s"),
                    raw_candidate_object_count=raw_count,
                    accepted_object_count=feng_count,
                    feng_mcs_object_count=feng_count,
                    persistent_dcs_object_count=persistent_count,
                    retained_population_object_count=retained_count,
                    rejected_object_count=raw_count - retained_count,
                    active_feng_mcs_track_count=int(
                        np.unique(object_track_ids[feng_objects]).size
                    ),
                    active_persistent_dcs_track_count=int(
                        np.unique(object_track_ids[persistent_objects]).size
                    ),
                    feng_mcs_object_fraction=(
                        feng_count / raw_count if raw_count else np.nan
                    ),
                    persistent_dcs_object_fraction=(
                        persistent_count / raw_count if raw_count else np.nan
                    ),
                    denominator="raw PyFLEXTRKR cloud objects in each six-hour scene",
                )
            )
            report_loop_progress(
                f"Figure 3 timeline {sim_name}",
                time_index + 1,
                scene_count,
                started,
            )
    return pd.DataFrame(rows)


def plot_pyflextrkr_object_count_timeline(timeline):
    """Plot raw and both retained population counts through time."""
    if timeline is None or len(timeline) == 0:
        progress("Skipping Figure 3 object timeline: no scene counts are available")
        return
    timeline.to_csv(
        os.path.join(save_loc, "pyflextrkr_object_acceptance_timeline.csv"),
        index=False,
    )
    fig, axes = plt.subplots(
        len(sim_names),
        1,
        figsize=(13.5, max(3.4, 3.0 * len(sim_names))),
        constrained_layout=True,
        sharex=True,
        squeeze=False,
    )
    for isim, sim_name in enumerate(sim_names):
        ax = axes[isim, 0]
        sub = timeline[timeline.simulation == sim_name].sort_values("Time")
        if len(sub) == 0:
            ax.text(0.5, 0.5, "No scene counts", transform=ax.transAxes, ha="center")
            continue
        times = pd.to_datetime(sub.Time)
        raw = sub.raw_candidate_object_count.to_numpy(dtype=float)
        feng = sub.feng_mcs_object_count.to_numpy(dtype=float)
        persistent = sub.persistent_dcs_object_count.to_numpy(dtype=float)
        ax.plot(times, raw, color="0.15", lw=1.1, label="All candidate objects")
        ax.plot(
            times,
            persistent,
            color="#2b8cbe",
            lw=1.25,
            label="Objects on persistent DCS tracks",
        )
        ax.plot(
            times,
            feng,
            color="tab:green",
            lw=1.35,
            label="Objects on Feng-defined MCS tracks",
        )
        denominator = np.nansum(raw)
        persistent_fraction = (
            float(np.nansum(persistent) / denominator)
            if denominator > 0.0 else np.nan
        )
        feng_fraction = (
            float(np.nansum(feng) / denominator)
            if denominator > 0.0 else np.nan
        )
        ax.text(
            0.99,
            0.95,
            f"mean per scene: {np.nanmean(raw):.1f} candidates, "
            f"{np.nanmean(persistent):.1f} persistent DCS, "
            f"{np.nanmean(feng):.1f} Feng MCS\n"
            f"candidate shares: {100.0 * persistent_fraction:.1f}% DCS, "
            f"{100.0 * feng_fraction:.1f}% Feng",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85),
        )
        ax.set_ylabel("Objects per scene")
        ax.set_title(sim_name)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    axes[-1, 0].set_xlabel("Simulation time")
    fig.suptitle(
        "PyFLEXTRKR cloud objects through time\n"
        "Broad set: all Feng MCSs plus >=2 consecutive 6-h scenes with a 225-K core; strict set: published 50-km PF criteria"
    )
    save_figure(3, fig, "pyflextrkr_object_acceptance_timeline")
    plt.close(fig)


def plot_pyflextrkr_track_step_validation(objects_df):
    if not should_run_figure(3):
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
            fig.suptitle("Retained-track consecutive-snapshot centroid speeds")
            save_figure(3, fig, f"pyflextrkr_track_step_continuity_{source}")
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
    plot_pyflextrkr_object_count_timeline(pyflextrkr_object_count_timeline())
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), constrained_layout=True)
    x = np.arange(len(sim_names))
    bottoms = np.zeros(len(sim_names), dtype=float)
    categories = (
        ("feng", "Feng-defined MCS", "tab:green"),
        ("persistent_only", "Persistent DCS, not Feng MCS", "#2b8cbe"),
        ("single_core", f"Core-bearing but < {MIN_TRACK_OBSERVATIONS} consecutive scenes", "0.55"),
        ("no_core", "No 225-K core in track", "tab:purple"),
        ("other", "Other candidate/unlinked", "tab:orange"),
    )
    for category, label, color in categories:
        counts = []
        for simulation in sim_names:
            sub = manifest[manifest.simulation == simulation]
            feng = manifest_population_mask(sub, "feng_mcs")
            persistent = manifest_population_mask(sub, "persistent_dcs")
            retained = feng | persistent
            has_core = sub.has_embedded_convective_core.astype(bool)
            short_core = (
                (~retained)
                & has_core
                & (sub.max_consecutive_observed_scenes < MIN_TRACK_OBSERVATIONS)
            )
            if category == "feng":
                mask = feng
            elif category == "persistent_only":
                mask = persistent & ~feng
            elif category == "single_core":
                mask = short_core
            elif category == "no_core":
                mask = (~retained) & ~has_core
            else:
                mask = (~retained) & has_core & ~short_core
            counts.append(int(mask.sum()))
        axes[0].bar(x, counts, bottom=bottoms, color=color, label=label)
        bottoms += np.asarray(counts, dtype=float)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sim_names)
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=0.8)
    axes[0].set_ylabel("Raw PyFLEXTRKR track count (log scale)")
    axes[0].set_title("Track populations")
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
        "PyFLEXTRKR cold-cloud tracks: persistent core-bearing DCS and "
        "Feng et al. (2021) 50-km MCS subset",
        fontsize=11,
    )
    save_figure(3, fig, "pyflextrkr_organised_track_acceptance_qc")
    plt.close(fig)


def classification_pyflextrkr_overlap_table(class_code, pyflex_code, area2d):
    class_code = np.asarray(class_code)
    pyflex_code = np.asarray(pyflex_code)
    area = np.asarray(area2d, dtype=float)
    rows = []
    classification_labels = {
        1: "PyFLEXTRKR convective core",
        2: "stratiform precipitating",
        3: "non-precipitating",
        4: "unclassified",
    }
    pyflex_labels = {
        1: f"cold core: Tb < {PYFLEXTRKR_TB_CORE_K:g} K",
        2: f"cold-shield shell: {PYFLEXTRKR_TB_CORE_K:g}-{PYFLEXTRKR_TB_COLD_K:g} K",
        3: f"warm-shield shell: {PYFLEXTRKR_TB_COLD_K:g}-{PYFLEXTRKR_TB_WARM_K:g} K",
    }
    # pyflex_code contains the mutually exclusive displayed scene regions:
    # cold core, cold-shield shell, and warm-shield shell.
    pyflex_membership = {
        1: pyflex_code == 1,
        2: pyflex_code == 2,
        3: pyflex_code == 3,
    }
    total_area = float(np.nansum(area[class_code > 0]))
    for class_id, class_label in classification_labels.items():
        class_mask = class_code == class_id
        class_area = float(np.nansum(area[class_mask]))
        for pyflex_id, pyflex_label in pyflex_labels.items():
            overlap_area = float(
                np.nansum(area[class_mask & pyflex_membership[pyflex_id]])
            )
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
                    pyflextrkr_membership="mutually exclusive scene region",
                )
            )
    return pd.DataFrame(rows)


def plot_classification_pyflextrkr_comparison(objects_df, fields_by_run, masks_by_run):
    if not should_run_figure(4):
        return

    class_cmap, class_norm = high_cloud_class_cmap_norm(alpha=0.86)
    class_handles = high_cloud_class_legend_handles()
    pyflex_colors = ["white", "#b2182b", "#fdae61", "#67a9cf"]
    pyflex_cmap = mpl_colors.ListedColormap(pyflex_colors)
    pyflex_norm = mpl_colors.BoundaryNorm(np.arange(-0.5, 4.5, 1.0), pyflex_cmap.N)
    pyflex_handles = [
        Patch(facecolor=pyflex_colors[1], label=f"cold core: Tb < {PYFLEXTRKR_TB_CORE_K:g} K"),
        Patch(facecolor=pyflex_colors[2], label=f"cold-shield shell: {PYFLEXTRKR_TB_CORE_K:g}-{PYFLEXTRKR_TB_COLD_K:g} K"),
        Patch(facecolor=pyflex_colors[3], label=f"warm-shield shell: {PYFLEXTRKR_TB_COLD_K:g}-{PYFLEXTRKR_TB_WARM_K:g} K"),
    ]

    overlap_rows = []

    for c, sim_name in enumerate(sim_names):
        progress(
            f"Figure 4: preparing {sim_name} comparison "
            f"({c + 1}/{len(sim_names)})"
        )
        fig = plt.figure(figsize=(14.5, 8.8), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.9))
        ax_class = fig.add_subplot(grid[0, 0])
        ax_pyflex = fig.add_subplot(grid[0, 1])
        ax_matrix = fig.add_subplot(grid[1, :])
        files = pyflextrkr_cloudid_files(ORGANISED_SYSTEM_SOURCE, sim_name)
        if not files:
            for ax in (ax_class, ax_pyflex, ax_matrix):
                ax.text(0.5, 0.5, "No current PyFLEXTRKR scene", transform=ax.transAxes, ha="center", va="center")
            safe_sim = sim_name.replace(" ", "_")
            save_figure(4, fig, f"pyflextrkr_classification_comparison_{safe_sim}")
            plt.close(fig)
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
                num=min(len(files), VALIDATION_SCENE_SEARCH_LIMIT),
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
                score = int(np.count_nonzero(np.isin(candidate_status, (1, 2))))
                if score > best_score:
                    best_score = score
                    time_index = int(candidate_index)
        time_index = min(time_index, len(files) - 1)

        with xr.open_dataset(files[time_index], decode_times=False, mask_and_scale=False) as cloud:
            labels = np.asarray(cloud["feature_number"].squeeze().values, dtype=np.int32)
            lat = np.asarray(cloud["lat"].values, dtype=float)
            lon = np.asarray(cloud["lon"].values, dtype=float)

        class_code = cloud_class_code_for_movie(masks_by_run[c], time_index, lat, lon)
        core_grid = tracking_grid_field(
            masks_by_run[c]["pyflextrkr_core_mask"], time_index, lat, lon
        )
        cold_grid = tracking_grid_field(
            masks_by_run[c]["pyflextrkr_core_bearing_cold_shield_mask"],
            time_index, lat, lon,
        )
        warm_grid = tracking_grid_field(
            masks_by_run[c]["pyflextrkr_core_bearing_warm_shield_mask"],
            time_index, lat, lon,
        )
        pyflex_code = np.zeros(labels.shape, dtype=np.int8)
        pyflex_code[np.asarray(warm_grid, dtype=bool)] = 3
        pyflex_code[np.asarray(cold_grid, dtype=bool)] = 2
        pyflex_code[np.asarray(core_grid, dtype=bool)] = 1
        status, _accepted_ids, _short_ids, _rejected_ids = pyflextrkr_scene_label_status(
            ORGANISED_SYSTEM_SOURCE, sim_name, time_index, labels
        )
        accepted_values = status == 1
        persistent_only_values = status == 2
        area2d = grid_cell_area_km2(lat, lon)
        overlap = classification_pyflextrkr_overlap_table(class_code, pyflex_code, area2d)
        overlap.insert(0, "simulation", sim_name)
        overlap.insert(1, "time_index", time_index)
        overlap.insert(2, "Time", fields_by_run[c]["precip"].Time.values[time_index])
        overlap_rows.append(overlap)

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
        if np.any(persistent_only_values):
            for ax in (ax_class, ax_pyflex):
                ax.contour(
                    lon,
                    lat,
                    persistent_only_values.astype(float),
                    levels=[0.5],
                    colors="#0868ac",
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
            "Cloud classes: core, stratiform, non-precipitating",
            fontsize=9,
        )
        ax_pyflex.set_title(
            "Exclusive PyFLEXTRKR scene regions\n"
            "225-K core / 241-K cold shell / 261-K warm shell",
            fontsize=9,
        )

        matrix = overlap.pivot(
            index="classification_label",
            columns="pyflextrkr_label",
            values="percent_of_classification_area",
        ).reindex(
            index=[
                "PyFLEXTRKR convective core",
                "stratiform precipitating",
                "non-precipitating",
                "unclassified",
            ],
            columns=[
                f"cold core: Tb < {PYFLEXTRKR_TB_CORE_K:g} K",
                f"cold-shield shell: {PYFLEXTRKR_TB_CORE_K:g}-{PYFLEXTRKR_TB_COLD_K:g} K",
                f"warm-shield shell: {PYFLEXTRKR_TB_COLD_K:g}-{PYFLEXTRKR_TB_WARM_K:g} K",
            ],
        )
        im = ax_matrix.imshow(
            matrix.to_numpy(dtype=float),
            vmin=0.0,
            vmax=100.0,
            cmap="viridis",
            aspect="auto",
        )
        ax_matrix.set_xticks(np.arange(matrix.shape[1]))
        ax_matrix.set_xticklabels(
            ["225-K cold core", "241-K cold shell", "261-K warm shell"],
            rotation=30,
            ha="right",
        )
        ax_matrix.set_yticks(np.arange(matrix.shape[0]))
        ax_matrix.set_yticklabels(["core", "stratiform", "non-precip.", "unclassified"])
        ax_matrix.set_title(
            "Class area in each exclusive scene region (%)",
            fontsize=9,
        )
        for iy in range(matrix.shape[0]):
            for ix in range(matrix.shape[1]):
                value = matrix.to_numpy(dtype=float)[iy, ix]
                if np.isfinite(value):
                    ax_matrix.text(ix, iy, f"{value:.0f}", ha="center", va="center", color="white" if value > 50 else "black", fontsize=8)
        fig.colorbar(
            im,
            ax=ax_matrix,
            label="% of classification class area",
            shrink=0.55,
            aspect=25,
            pad=0.02,
        )

        fig.legend(
            handles=class_handles + pyflex_handles + [
                Line2D([0], [0], color="k", lw=1.0, label="Feng-defined MCS outline"),
                Line2D([0], [0], color="#0868ac", lw=1.0, label="Persistent DCS-only outline"),
            ],
            loc="outside lower center",
            ncol=3,
            frameon=True,
            fontsize=8,
        )
        fig.suptitle(
            f"{sim_name}: shared high-cloud classes versus PyFLEXTRKR "
            "cold-core and shield-shell regions",
            fontsize=12,
        )
        safe_sim = sim_name.replace(" ", "_")
        save_figure(4, fig, f"pyflextrkr_classification_comparison_{safe_sim}")
        plt.close(fig)

    if overlap_rows:
        pd.concat(overlap_rows, ignore_index=True).to_csv(
            os.path.join(save_loc, "pyflextrkr_classification_overlap_snapshot.csv"),
            index=False,
        )


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


def save_pyflextrkr_animation(fig, update, nframes, output_path, title):
    """Write one MP4 with progress reporting for long validation renders."""
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "Figure 5 requires ffmpeg, but Matplotlib cannot find an ffmpeg writer"
        )
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
        bitrate=2400,
        extra_args=[
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
        ],
        metadata={"title": title},
    )
    started = time.monotonic()
    progress(f"Writing Figure 5 movie with {nframes} frames: {output_path}")

    def movie_progress(frame_number, frame_count):
        report_loop_progress(
            f"Figure 5 movie {os.path.basename(output_path)}",
            frame_number + 1,
            frame_count,
            started,
        )

    movie.save(
        output_path,
        writer=writer,
        dpi=115,
        progress_callback=movie_progress,
    )
    plt.close(fig)
    progress(f"Finished Figure 5 movie: {output_path}")


def plot_pyflextrkr_validation_movies(objects_df, fields_by_run, masks_by_run):
    """Animate candidate acceptance and its relation to the cloud classes."""
    if not should_run_figure(5):
        return
    if objects_df is None:
        objects_df = pd.DataFrame()

    precip_cmap, precip_norm = precip_rate_cmap_norm()
    class_cmap, class_norm = high_cloud_class_cmap_norm(alpha=0.86)
    class_handles = high_cloud_class_legend_handles()
    pyflex_colors = ["white", "#b2182b", "#fdae61", "#67a9cf"]
    pyflex_cmap = mpl_colors.ListedColormap(pyflex_colors)
    pyflex_norm = mpl_colors.BoundaryNorm(
        np.arange(-0.5, 4.5, 1.0),
        pyflex_cmap.N,
    )
    status_handles = [
        Patch(facecolor="#1a9850", alpha=0.55, label="Feng-defined MCS track"),
        Patch(
            facecolor="#2b8cbe",
            alpha=0.50,
            label="Persistent DCS track (not Feng MCS)",
        ),
        Patch(
            facecolor="#fdae61",
            alpha=0.46,
            label="Other candidate or unlinked",
        ),
    ]
    pyflex_handles = [
        Patch(facecolor=pyflex_colors[1], label=f"cold core < {PYFLEXTRKR_TB_CORE_K:g} K"),
        Patch(facecolor=pyflex_colors[2], label=f"cold-shield shell to {PYFLEXTRKR_TB_COLD_K:g} K"),
        Patch(facecolor=pyflex_colors[3], label=f"warm-shield shell to {PYFLEXTRKR_TB_WARM_K:g} K"),
    ]
    manifest_rows = []

    for sim_index, sim_name in enumerate(sim_names):
        files = pyflextrkr_cloudid_files(ORGANISED_SYSTEM_SOURCE, sim_name)
        if not files:
            progress(f"Skipping Figure 5 for {sim_name}: no PyFLEXTRKR scene files")
            continue
        ntime = min(
            int(fields_by_run[sim_index]["precip"].sizes["Time"]),
            len(files),
        )
        if ntime == 0:
            continue
        sim_objects = (
            objects_df[
                (objects_df.simulation == sim_name)
                & (objects_df.source == ORGANISED_SYSTEM_SOURCE)
            ].copy()
            if len(objects_df)
            and {"simulation", "source"}.issubset(objects_df.columns)
            else pd.DataFrame()
        )
        retained_counts = (
            sim_objects.groupby("time_index").size()
            if len(sim_objects) and "time_index" in sim_objects.columns
            else pd.Series(dtype=float)
        )
        peak_index = int(retained_counts.idxmax()) if len(retained_counts) else ntime // 2
        nframes = min(PYFLEXTRKR_MOVIE_MAX_FRAMES, ntime)
        start = max(0, min(peak_index - nframes // 2, ntime - nframes))
        frame_indices = list(range(start, start + nframes))

        with xr.open_dataset(
            files[frame_indices[0]],
            decode_times=False,
            mask_and_scale=False,
        ) as first_cloud:
            lat = np.asarray(first_cloud["lat"].values, dtype=float)
            lon = np.asarray(first_cloud["lon"].values, dtype=float)

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(13.0, 10.5),
            constrained_layout=True,
            sharex=True,
        )
        fig.colorbar(
            plt.cm.ScalarMappable(norm=precip_norm, cmap=precip_cmap),
            ax=axes[0],
            label=r"Precipitation (mm day$^{-1}$)",
            shrink=0.48,
            aspect=24,
            pad=0.015,
        )

        def update(frame_number):
            time_index = frame_indices[frame_number]
            for ax in axes:
                ax.clear()
            with xr.open_dataset(
                files[time_index],
                decode_times=False,
                mask_and_scale=False,
            ) as cloud:
                labels = np.asarray(
                    cloud["feature_number"].squeeze().values,
                    dtype=np.int32,
                )
            precip = tracking_grid_field(
                fields_by_run[sim_index]["precip"],
                time_index,
                lat,
                lon,
            )
            precip_values = np.asarray(precip, dtype=float)
            status, _accepted_ids, _criteria_ids, _rejected_ids = (
                pyflextrkr_scene_label_status(
                    ORGANISED_SYSTEM_SOURCE,
                    sim_name,
                    time_index,
                    labels,
                )
            )
            accepted_values = status == 1
            criteria_rejected_values = status == 2
            other_rejected_values = status == 3

            axes[0].pcolormesh(
                lon,
                lat,
                np.maximum(precip_values, 0.0),
                cmap=precip_cmap,
                norm=precip_norm,
                shading="auto",
                rasterized=True,
            )
            overlay_status_mask(
                axes[0], lon, lat, other_rejected_values, "#fdae61", 0.46
            )
            overlay_status_mask(
                axes[0], lon, lat, criteria_rejected_values, "#2b8cbe", 0.50
            )
            overlay_status_mask(
                axes[0], lon, lat, accepted_values, "#1a9850", 0.55
            )
            for values, color, linewidth in (
                (other_rejected_values, "#d95f02", 0.65),
                (criteria_rejected_values, "#0868ac", 0.72),
                (accepted_values, "0.05", 0.90),
            ):
                if np.any(values):
                    axes[0].contour(
                        lon,
                        lat,
                        values.astype(float),
                        levels=[0.5],
                        colors=color,
                        linewidths=linewidth,
                    )

            scene = (
                sim_objects[sim_objects.time_index == time_index].nlargest(
                    10, "area_km2"
                )
                if len(sim_objects) and "time_index" in sim_objects.columns
                else pd.DataFrame()
            )
            selected_ids = (
                set(scene.track_id.astype(int))
                if len(scene) and "track_id" in scene.columns
                else set()
            )
            if selected_ids:
                history = sim_objects[
                    sim_objects.track_id.isin(selected_ids)
                    & (sim_objects.time_index >= time_index - PYFLEXTRKR_MOVIE_TRACK_HISTORY_STEPS)
                    & (sim_objects.time_index <= time_index)
                ]
                for track_id, track in history.groupby("track_id"):
                    track = track.sort_values("time_index")
                    x = track.centroid_lon.to_numpy(dtype=float)
                    y = track.centroid_lat.to_numpy(dtype=float)
                    for jump_index in np.where(np.abs(np.diff(x)) > 180.0)[0][::-1]:
                        x = np.insert(x, jump_index + 1, np.nan)
                        y = np.insert(y, jump_index + 1, np.nan)
                    axes[0].plot(x, y, color="k", lw=0.9, alpha=0.85)
                    axes[0].scatter(x[-1:], y[-1:], color="k", s=10, zorder=8)
                for record in scene.itertuples(index=False):
                    annotate_track_id(
                        axes[0],
                        record.centroid_lon,
                        record.centroid_lat,
                        record.track_id,
                        "0.1",
                        fontsize=5.5,
                    )
            axes[0].set_title(
                "PyFLEXTRKR candidates over precipitation: "
                f"{np.unique(labels[labels > 0]).size} total, "
                f"{np.unique(labels[criteria_rejected_values]).size} persistent DCS only, "
                f"{np.unique(labels[accepted_values]).size} Feng MCS"
            )
            axes[0].legend(
                handles=status_handles,
                loc="lower left",
                ncol=3,
                fontsize=7,
                frameon=True,
            )

            class_code = cloud_class_code_for_movie(
                masks_by_run[sim_index],
                time_index,
                lat,
                lon,
            )
            axes[1].pcolormesh(
                lon,
                lat,
                np.ma.masked_where(class_code <= 0, class_code),
                cmap=class_cmap,
                norm=class_norm,
                shading="auto",
                rasterized=True,
            )
            axes[1].set_title("Dependent high-cloud classification")
            axes[1].legend(
                handles=class_handles,
                loc="lower left",
                ncol=4,
                fontsize=7,
                frameon=True,
            )

            core = tracking_grid_field(
                masks_by_run[sim_index]["pyflextrkr_core_mask"],
                time_index,
                lat,
                lon,
            )
            cold = tracking_grid_field(
                masks_by_run[sim_index]["pyflextrkr_core_bearing_cold_shield_mask"],
                time_index,
                lat,
                lon,
            )
            warm = tracking_grid_field(
                masks_by_run[sim_index]["pyflextrkr_core_bearing_warm_shield_mask"],
                time_index,
                lat,
                lon,
            )
            pyflex_code = np.zeros(labels.shape, dtype=np.int8)
            pyflex_code[np.asarray(warm, dtype=bool)] = 3
            pyflex_code[np.asarray(cold, dtype=bool)] = 2
            pyflex_code[np.asarray(core, dtype=bool)] = 1
            axes[2].pcolormesh(
                lon,
                lat,
                np.ma.masked_where(pyflex_code <= 0, pyflex_code),
                cmap=pyflex_cmap,
                norm=pyflex_norm,
                shading="auto",
                rasterized=True,
            )
            if np.any(accepted_values):
                axes[2].contour(
                    lon,
                    lat,
                    accepted_values.astype(float),
                    levels=[0.5],
                    colors="k",
                    linewidths=0.8,
                )
            if np.any(criteria_rejected_values):
                axes[2].contour(
                    lon,
                    lat,
                    criteria_rejected_values.astype(float),
                    levels=[0.5],
                    colors="#0868ac",
                    linewidths=0.8,
                )
            axes[2].set_title(
                "PyFLEXTRKR cold-core and shield regions "
                "(black: Feng MCS; blue: persistent DCS only)"
            )
            axes[2].legend(
                handles=pyflex_handles,
                loc="lower left",
                ncol=3,
                fontsize=7,
                frameon=True,
            )

            for ax in axes:
                ax.set_xlim(float(lon.min()), float(lon.max()))
                ax.set_ylim(*TROPICAL_LAT_BOUNDS)
                ax.set_aspect("equal", adjustable="box")
                ax.set_ylabel("Latitude")
                ax.grid(alpha=0.12, lw=0.4)
            axes[-1].set_xlabel("Longitude")
            timestamp = pd.to_datetime(
                fields_by_run[sim_index]["precip"].Time.values[time_index]
            )
            fig.suptitle(f"{sim_name} | {timestamp:%Y-%m-%d %H:%M}")
            return []

        safe_sim = sim_name.replace(" ", "_")
        output_path = os.path.join(
            save_loc,
            f"05_pyflextrkr_validation_{safe_sim}.mp4",
        )
        save_pyflextrkr_animation(
            fig,
            update,
            len(frame_indices),
            output_path,
            f"PyFLEXTRKR and high-cloud validation: {sim_name}",
        )
        manifest_rows.append(
            dict(
                simulation=sim_name,
                movie_type="dual_population_candidates_and_cloud_classification",
                output_file=output_path,
                frame_count=len(frame_indices),
                first_time_index=frame_indices[0],
                last_time_index=frame_indices[-1],
                fps=PYFLEXTRKR_MOVIE_FPS,
            )
        )

    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(save_loc, "pyflextrkr_validation_movie_manifest.csv"),
        index=False,
    )


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
    test_ds = subset_latitude(
        ds.sel(Time=selected), lat_bounds=QUICK_TEST_LAT_BOUNDS
    ).sel(longitude=slice(*QUICK_TEST_LON_BOUNDS))
    print(
        f"{sim_name}: test domain {QUICK_TEST_LAT_BOUNDS[0]:g} to "
        f"{QUICK_TEST_LAT_BOUNDS[1]:g} deg latitude, "
        f"{QUICK_TEST_LON_BOUNDS[0]:g} to {QUICK_TEST_LON_BOUNDS[1]:g} deg longitude",
        flush=True,
    )
    return test_ds


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
        test_mask_names = list(dict.fromkeys([
            *[name for name in PYFLEXTRKR_TRACKED_SOURCES if name in masks],
            *[name for name in CLASSIFICATION_DIAGNOSTIC_KEYS if name in masks],
        ]))
        already_loaded = (
            getattr(fields["precip"], "chunks", None) is None
            and all(getattr(masks[name], "chunks", None) is None for name in test_mask_names)
        )
        if already_loaded:
            progress(f"{sim_name}: reusing in-memory short tropical tracking inputs")
            continue
        progress(f"{sim_name}: loading the short tropical fields and masks once")
        precip = subset_lat(
            fields["precip"], lat_bounds=TROPICAL_LAT_BOUNDS
        ).load()
        fields["precip"] = precip
        fields["mean_precip"] = weighted_tropical_mean(
            precip, lat_bounds=TROPICAL_LAT_BOUNDS
        ).load()
        mask_started = time.monotonic()
        for index, source_name in enumerate(test_mask_names, start=1):
            masks[source_name] = subset_lat(
                masks[source_name], lat_bounds=TROPICAL_LAT_BOUNDS
            ).load()
            report_loop_progress(
                f"{sim_name}: test masks loaded",
                index,
                len(test_mask_names),
                mask_started,
            )
        progress(f"{sim_name}: short tropical tracking inputs ready")


def load_datasets():
    dss = []
    for i, d in enumerate(data_locs):
        sim_name = sim_names[i]
        progress(
            f"{sim_name}: opening lat-lon dataset ({i + 1}/{len(data_locs)})"
        )
        ds = load_latlon_dataset(
            os.path.join(root_dir, d),
            latlon_pattern=LATLON_PATTERN,
            chunks=latlon_chunks,
        )
        ds = standardize_latlon_dataset(ds, latlon_vars)
        if TEST_MODE:
            ds = select_test_source_window(ds, sim_name)
        dimension_summary = ", ".join(
            f"{name}={size}" for name, size in ds.sizes.items()
        )
        progress(f"{sim_name}: dataset ready ({dimension_summary})")
        dss.append(ds)
    return dss


def build_masks_by_run(dss, fields_by_run):
    masks_by_run = []
    for i, (ds, fields) in enumerate(zip(dss, fields_by_run)):
        sim_name = sim_names[i]
        progress(
            f"{sim_name}: loading shared high-cloud masks "
            f"({i + 1}/{len(sim_names)})"
        )
        masks = cloud_type_masks(ds, fields, sim_name)
        masks[ORGANISED_SYSTEM_SOURCE] = build_organised_convective_system_mask(
            fields,
            masks,
        )
        masks_by_run.append(masks)
        progress(f"{sim_name}: organisation tracking masks ready")
    return masks_by_run


def main(argv=None):
    run_started = time.monotonic()
    args = parse_command_line(argv)
    configure_run(args)
    os.makedirs(save_loc, exist_ok=True)
    if TEST_MODE:
        progress(
            f"TEST MODE: simulation={sim_names[0]}, tracked_times={TEST_TIME_STEPS}, "
            f"figures={figures_to_run}, reuse_cache={TEST_REUSE_CACHE}, "
            f"output={save_loc}, object_cache={ORGANISATION_CACHE_DIR}"
        )
    else:
        progress(
            f"Starting convective-organisation diagnostics for "
            f"{len(sim_names)} simulations; figures={figures_to_run}; "
            f"object_cache={ORGANISATION_CACHE_DIR}"
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
                pd.DataFrame([
                    dict(
                        simulation=sim_name,
                        tracked_snapshot_count=int(fields["precip"].sizes["Time"]),
                        first_tracked_time=str(fields["precip"].Time.values[0]),
                        last_tracked_time=str(fields["precip"].Time.values[-1]),
                        snapshot_spacing_hours=dt,
                        output_directory=save_loc,
                        organisation_cache_directory=ORGANISATION_CACHE_DIR,
                        pyflextrkr_work_directory=PYFLEXTRKR_WORK_DIR,
                        tracking_mode=TRACKING_MODE,
                        reused_test_cache=TEST_REUSE_CACHE,
                    )
                    for sim_name, fields in zip(sim_names, fields_by_run)
                ]).to_csv(
                    os.path.join(save_loc, "pyflextrkr_test_run_manifest.csv"),
                    index=False,
                )
        return dss, fields_by_run

    def ensure_masks():
        nonlocal masks_by_run
        if masks_by_run is None:
            dss_local, fields_local = ensure_fields()
            masks_by_run = build_masks_by_run(dss_local, fields_local)
        return masks_by_run

    threshold_table = build_threshold_table()
    threshold_table.to_csv(
        os.path.join(save_loc, "convective_system_tracking_method_thresholds.csv"), index=False
    )
    progress("Wrote convective-system tracking and population method table")

    progress("Checking the dual-population track cache")
    track_products = load_cached_track_persistence_products()
    if track_products is None:
        progress("No current dual-population track cache; starting reconstruction")
        _, fields_by_run = ensure_fields()
        masks_by_run = ensure_masks()
        materialize_test_tracking_inputs(fields_by_run, masks_by_run)
        objects_df, tracks_df = build_track_persistence_metrics(
            masks_by_run, fields_by_run
        )
        write_track_persistence_cache(objects_df, tracks_df)
    else:
        objects_df, tracks_df = track_products

    write_pyflextrkr_obsolete_output_report()
    write_organised_track_acceptance_report()

    if should_run_figure(1):
        progress("Starting Figure 1: persistent-DCS and Feng-MCS properties")
        _, fields_by_run = ensure_fields()
        plot_convective_system_population_properties(
            objects_df, tracks_df, fields_by_run
        )
        progress("Figure 1 stage complete")
    if should_run_figure(2):
        progress("Starting Figure 2: PyFLEXTRKR track-validation snapshots")
        _, fields_by_run = ensure_fields()
        masks_by_run = ensure_masks()
        if TEST_MODE:
            materialize_test_tracking_inputs(fields_by_run, masks_by_run)
        plot_pyflextrkr_track_validation(objects_df, fields_by_run, masks_by_run)
        progress("Figure 2 stage complete")
    if should_run_figure(3):
        progress("Starting Figure 3: track size and speed validation")
        plot_pyflextrkr_track_step_validation(objects_df)
        progress("Figure 3 stage complete")
    if should_run_figure(4):
        progress("Starting Figure 4: cloud-class/PyFLEXTRKR comparison")
        _, fields_by_run = ensure_fields()
        masks_by_run = ensure_masks()
        if TEST_MODE:
            materialize_test_tracking_inputs(fields_by_run, masks_by_run)
        plot_classification_pyflextrkr_comparison(
            objects_df, fields_by_run, masks_by_run
        )
        progress("Figure 4 stage complete")
    if should_run_figure(5):
        progress("Starting Figure 5: evolving PyFLEXTRKR validation movies")
        _, fields_by_run = ensure_fields()
        masks_by_run = ensure_masks()
        if TEST_MODE:
            materialize_test_tracking_inputs(fields_by_run, masks_by_run)
        plot_pyflextrkr_validation_movies(
            objects_df,
            fields_by_run,
            masks_by_run,
        )
        progress("Figure 5 stage complete")

    progress(
        "Finished convective_organisation_diagnostics.py; total elapsed "
        f"{format_duration(time.monotonic() - run_started)}"
    )


if __name__ == "__main__":
    main()
