#!/usr/bin/env python
# coding: utf-8

"""
Refined high-cloud classification diagnostics for MPAS-A nudging experiments.

This script owns the shared high-cloud classification, including the
deep-convective attribution derived from instantaneous PyFLEXTRKR cold-core
scene masks. Its primary
figures report class occurrence/composition, cloud-top-pressure distributions,
TTL-reaching frequency, tropical-mean vertical cloud profiles,
class-conditioned OLR distributions, spatial occurrence responses, and rain-matched cloud
depth/OLR responses. Core/cold-shield/warm-shield scene regions and vertical
slices provide the Figure 3 validation view in both normal and test runs.

The expensive full-record reductions are versioned independently of the
rendering code below ``high_cloud_diagnostics_cache`` in the data repository.
After one cache-building run, ``--figures-only --figures ...`` can regenerate
Figures 1, 2, 4, 5, or 6 without opening the simulation fields. Figure 3 is an
instantaneous source-data validation view and therefore still opens only the
selected source slices and classification fields.
"""

import argparse
import gc
import json
import os
import sys
import time
from functools import partial

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in (".", "..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import dask
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors, pyplot as plt, ticker
from matplotlib.lines import Line2D
from scipy import ndimage

from diagnostics.diagnostic_helpers import (
    FigureSaver, HIGH_CLOUD_CLASS_SPECS, LATLON_PATTERN,
    QUICK_TEST_LON_BOUNDS,
    diagnostic_cache_dir, diagnostic_output_dir, difference_cmap,
    figure_should_run, hemispheric_fold,
    high_cloud_class_cmap_norm, high_cloud_class_legend_handles,
    latitude_is_hemispherically_symmetric,
    load_latlon_dataset, precip_rate_cmap_norm,
    precipitation_rate_from_dataset, progress,
    sample_matching_arrays, standardize_latlon_dataset,
    simulation_colour, symmetric_levels,
    subset_latitude, time_block_slices, weighted_area_time_mean,
)
from diagnostics.convection.high_cloud_classification import (
    CLASSIFICATION_CACHE_VERSION,
    HighCloudThresholds,
    cached_classification_is_current as shared_cached_classification_is_current,
    classification_dataset as shared_classification_dataset,
    classify_high_cloud_from_dataset,
    corrected_cached_warm_shield_masks,
    standardize_cloud_fraction,
    standardize_pressure_hpa,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/high_cloud_refined/")
default_save_loc = save_loc

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: high_cloud_occurrence_and_composition and high_cloud_regime_attribution
#   2: tropical/subtropical cloud-top and OLR distributions
#   3: cloud_classification_validation_snapshots
#   4: high_cloud_vertical_contribution_profiles
#   5: high_cloud_class_spatial_response
#   6: rain_matched_high_cloud_structure_response
figures_to_run = [1, 2, 3, 4, 5, 6]
figsave = None

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control_new", "u_nudge", "T_nudge"]
sim_names = ["control", "nudged u", "nudged T"]
cols = [simulation_colour(name) for name in sim_names]

dt = 6

TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
SNAPSHOT_LAT_BOUNDS = (-20.0, 20.0)
HIGH_CLOUD_TEST_LAT_BOUNDS = (-40.0, 40.0)
CLOUD_STRUCTURE_DOMAIN_SPECS = (
    ("tropical", (0.0, 20.0), "20°S–20°N"),
    ("subtropical", (20.0, 40.0), "20–40° in both hemispheres"),
)
CLOUD_TOP_PRESSURE_BIN_WIDTH_HPA = 5.0
CLOUD_TOP_PRESSURE_DISPLAY_BIN_WIDTH_HPA = 20.0
OLR_BIN_WIDTH_W_M2 = 5.0

# Cloud-top pressure avoids tying tropical high cloud to one geometric height.
# The 440 hPa parent follows the standard ISCCP high-cloud boundary. Continuous
# cloud-top pressure is retained for the UTLS cloud-deepening analysis.
CF_THRESH = 0.05
P_HIGH_HPA = 440.0
P_PRECIP_MIN = 12.0  # 0.5 mm h-1 PF threshold (Feng et al. 2021)
WRITE_CLASSIFICATION_PRODUCT = True
classification_cache_dir = diagnostic_cache_dir("classification_cache")
DERIVED_CACHE_VERSION = "high-cloud-derived-diagnostics-v2-folded-domains"
derived_cache_dir = diagnostic_cache_dir("high_cloud_diagnostics_cache")
DERIVED_CACHE_SIGNATURE = None
REBUILD_DERIVED_CACHE = False
FIGURES_ONLY = False

TEST_MODE = False
TEST_TIME_STEPS = 16
TEST_START = None
TEST_DEFAULT_FIGURES = [1, 2, 3, 4, 5, 6]
SIMULATION_ARGUMENTS = {
    "control": 0,
    "nudged-u": 1,
    "nudged-T": 2,
}

# The shared classification product is computed in bounded blocks. Eight times
# matches the source chunking; limiting the worker pool prevents several large
# 3-D cloud/pressure intermediates from accumulating ahead of the NetCDF writer.
CLASSIFICATION_WRITE_TIME_BATCH = 8
CLASSIFICATION_WRITE_WORKERS = 2

BOOTSTRAP_BLOCK_DAYS = 10.0
N_BOOTSTRAP = 600
BOOTSTRAP_SEED = 20240610
CI_PERCENTILES = (2.5, 97.5)

CLOUD_STRUCTURE_TIME_BATCH = 4
# Match the source Time chunk so each expensive 3-D cloud-fraction chunk is read
# once while still yielding regular progress updates in a full-period run.
VERTICAL_PROFILE_TIME_BATCH = 8
RAIN_CONDITIONED_TIME_BATCH_DAYS = BOOTSTRAP_BLOCK_DAYS
MIN_RAIN_CONDITIONED_BLOCKS = 4
RAIN_RATE_BINS_MM_H = np.array(
    [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0,
     3.0, 5.0, 8.0, np.inf]
)
SNAPSHOT_LON_WINDOW_DEG = 90.0
SNAPSHOT_Z_MAX_KM = 19.0
CLOUD_TOP_HEIGHT_MAP_MIN_KM = 6.0
CLOUD_TOP_HEIGHT_MAP_MAX_KM = 20.0
PROFILE_Z_BOUNDS_KM = (10.0, 20.0)
PROFILE_COMPUTE_Z_BOUNDS_KM = (5.0, 20.0)
UTLS_LAYER_BOUNDS_KM = (14.0, 18.0)
latlon_chunks = {"Time": 8, "latitude": 90, "longitude": 180}
dask.config.set({"array.slicing.split_large_chunks": True})

latlon_vars = [
    "rainc", "rainnc", "i_rainc", "i_rainnc",
    "olrtoa", "cldfrac", "pressure", "zgrid",
]

_CLASS_COLOURS = {
    code: colour for code, _label, colour in HIGH_CLOUD_CLASS_SPECS
}
BUDGET_SPECS = [
    ("high_cloud_mask", "Total high cloud", "0.25"),
    ("convective_precipitating_high_cloud_mask", "Deep convection", _CLASS_COLOURS[1]),
    ("stratiform_precipitating_high_cloud_mask", "Stratiform cloud", _CLASS_COLOURS[2]),
    ("nonprecipitating_high_cloud_mask", "Non-precipitating cloud", _CLASS_COLOURS[3]),
]

SPATIAL_CLASS_SPECS = BUDGET_SPECS
RAIN_CONDITIONED_CLASS_SPECS = BUDGET_SPECS

REGIME_SPECS = [
    ("convective_precipitating_high_cloud_mask", "Deep convection", _CLASS_COLOURS[1]),
    ("stratiform_precipitating_high_cloud_mask", "Stratiform cloud", _CLASS_COLOURS[2]),
    ("nonprecipitating_high_cloud_mask", "Non-precipitating cloud", _CLASS_COLOURS[3]),
]

PRECIPITATING_COMPOSITION_SPECS = [
    ("rainy_pyflextrkr_core_mask", "Deep convection", _CLASS_COLOURS[1]),
    ("stratiform_precipitating_high_cloud_mask", "Stratiform cloud", _CLASS_COLOURS[2]),
]

CIRRUS_COMPARISON_SPECS = [
    (
        "noncore_high_cloud_aggregate",
        "Stratiform + non-precipitating cloud",
        "tab:purple",
    ),
    (
        "nonprecipitating_high_cloud_mask",
        "Non-precipitating cloud",
        _CLASS_COLOURS[3],
    ),
]

NONPRECIP_SHIELD_SPECS = [
    (
        "nonprecip_within_core_bearing_warm_shield",
        "Inside 261-K convective shield",
        "tab:orange",
    ),
    (
        "nonprecip_outside_core_bearing_warm_shield",
        "Outside 261-K convective shield",
        "0.45",
    ),
]

# Published percentages used only where their denominator is explicit. The
# AIRS/IASI value is the closest like-for-like check on the parent high-cloud
# mask. The MODIS result helps compare the core share, but its optical-
# depth classification is not equivalent to the PyFLEXTRKR 225-K core.
LITERATURE_CLOUD_PERCENTAGES = [
    dict(
        metric="total_high_cloud_domain_occurrence",
        plotted_class_key="high_cloud_mask",
        value_percent=35.0,
        bound="approximately",
        comparison_role="closest like-for-like observational benchmark",
        definition="all upper-tropospheric cloud with cloud-top pressure < 440 hPa",
        denominator="all observations from 30S-30N",
        region="30S-30N",
        grid="0.5 degree",
        reference="Stubenrauch et al. (2023), Atmospheric Chemistry and Physics",
        doi="10.5194/acp-23-5867-2023",
    ),
    dict(
        metric="total_high_cloud_domain_occurrence",
        plotted_class_key="high_cloud_mask",
        value_percent=22.7,
        bound="reported tropical mean",
        comparison_role="observational benchmark with lower thin-cloud sensitivity",
        definition="MODIS high ice cloud classified using the ISCCP pressure boundary",
        denominator="all daytime observations from 30S-30N",
        region="30S-30N",
        grid="MODIS Collection 4 level-3 products",
        reference="Hong et al. (2007), Journal of Applied Meteorology and Climatology",
        doi="10.1175/2007JAMC1583.1",
    ),
    dict(
        metric="deep_convective_share_of_high_cloud",
        plotted_class_key="convective_precipitating_high_cloud_mask",
        value_percent=20.0,
        bound="upper bound (<20%)",
        comparison_role="optical-class comparison; not an equivalent class definition",
        definition="MODIS/ISCCP optical-depth deep-convective class",
        denominator="total MODIS high-cloud amount from 30S-30N",
        region="30S-30N",
        grid="MODIS Collection 4 level-3 products",
        reference="Hong et al. (2007), Journal of Applied Meteorology and Climatology",
        doi="10.1175/2007JAMC1583.1",
    ),
    dict(
        metric="cirriform_share_of_high_cloud",
        plotted_class_key="not plotted against one model class",
        value_percent=80.0,
        bound="lower bound (>80%)",
        comparison_role="optical-class comparison; cirriform spans more than one model class",
        definition="MODIS/ISCCP cirriform optical-depth classes",
        denominator="total MODIS high-cloud amount from 30S-30N",
        region="30S-30N",
        grid="MODIS Collection 4 level-3 products",
        reference="Hong et al. (2007), Journal of Applied Meteorology and Climatology",
        doi="10.1175/2007JAMC1583.1",
    ),
    dict(
        metric="tropical_cirrus_domain_occurrence",
        plotted_class_key="nonprecipitating_high_cloud_mask",
        value_percent=38.4,
        bound="lower endpoint of retrieval range",
        comparison_role="cirrus-occurrence comparison; closest latitude band but not an equivalent class",
        definition="CALIOP NASA level-2 non-opaque cirrus with maximum temperature < -40 C and optical depth > 0.001",
        denominator="all profiles from 15S-15N",
        region="15S-15N",
        grid="CALIOP profile sampling, June 2006-December 2008",
        reference="Martins et al. (2011), Journal of Geophysical Research: Atmospheres",
        doi="10.1029/2010JD014519",
    ),
    dict(
        metric="tropical_cirrus_domain_occurrence",
        plotted_class_key="nonprecipitating_high_cloud_mask",
        value_percent=43.5,
        bound="upper endpoint of retrieval range",
        comparison_role="cirrus-occurrence comparison; closest latitude band but not an equivalent class",
        definition="CALIOP selectively extracted non-opaque cirrus with maximum temperature < -40 C and optical depth > 0.001",
        denominator="all profiles from 15S-15N",
        region="15S-15N",
        grid="CALIOP profile sampling, June 2006-December 2008",
        reference="Martins et al. (2011), Journal of Geophysical Research: Atmospheres",
        doi="10.1029/2010JD014519",
    ),
    dict(
        metric="convective_share_of_tropical_raining_area",
        plotted_class_key="rainy_pyflextrkr_core_mask",
        value_percent=27.0,
        bound="derived complement of reported 73% stratiform area",
        comparison_role="closest radar-area benchmark; classification method differs",
        definition="TRMM PR convective rain area with reflectivity >= 17 dBZ",
        denominator="total tropical area covered by rain",
        region="20S-20N",
        grid="TRMM Precipitation Radar native sampling",
        reference="Schumacher and Houze (2003), Journal of Climate",
        doi="10.1175/1520-0442(2003)016<1739:SRITTA>2.0.CO;2",
    ),
    dict(
        metric="stratiform_share_of_tropical_raining_area",
        plotted_class_key="stratiform_precipitating_high_cloud_mask",
        value_percent=73.0,
        bound="reported mean",
        comparison_role="closest radar-area benchmark; classification method differs",
        definition="TRMM PR stratiform rain area with reflectivity >= 17 dBZ",
        denominator="total tropical area covered by rain",
        region="20S-20N",
        grid="TRMM Precipitation Radar native sampling",
        reference="Schumacher and Houze (2003), Journal of Climate",
        doi="10.1175/1520-0442(2003)016<1739:SRITTA>2.0.CO;2",
    ),
]

LITERATURE_CLOUD_STRUCTURE_VALUES = [
    dict(
        metric="mean_cloud_top_pressure",
        cloud_type="cirriform",
        value=281.4,
        units="hPa",
        definition="MODIS ISCCP cirriform high-cloud optical classes",
        region="30S-30N",
        reference="Hong et al. (2007), Journal of Applied Meteorology and Climatology",
        doi="10.1175/2007JAMC1583.1",
    ),
    dict(
        metric="mean_cloud_top_pressure",
        cloud_type="Deep convection",
        value=223.3,
        units="hPa",
        definition="MODIS ISCCP deep-convective high-cloud optical class",
        region="30S-30N",
        reference="Hong et al. (2007), Journal of Applied Meteorology and Climatology",
        doi="10.1175/2007JAMC1583.1",
    ),
    dict(
        metric="class_conditioned_olr",
        cloud_type="thin cirrus",
        value=260.0,
        units="W m-2",
        definition="OLR associated with optically thin tropical-ocean cirrus in a high-resolution model",
        region="tropical ocean",
        reference="Seiki et al. (2015), Geophysical Research Letters",
        doi="10.1002/2015GL064282",
    ),
    dict(
        metric="deep_convection_olr_threshold",
        cloud_type="deep convective system",
        value=205.0,
        units="W m-2",
        definition="satellite OLR threshold used to identify tropical deep-convective systems",
        region="0-20N identification domain; globally applied threshold",
        reference="Gu et al. (2002), Journal of Geophysical Research: Atmospheres",
        doi="10.1029/2002JD002089",
    ),
]

# Persist only fields used by the scientific and validation consumers.
CLASSIFICATION_CACHE_VARS = (
    "brightness_temperature",
    "pyflextrkr_valid_tb_mask",
    "pyflextrkr_core_mask",
    "pyflextrkr_cold_cloud_shield_mask",
    "pyflextrkr_core_bearing_cold_shield_mask",
    "pyflextrkr_warm_cloud_shield_mask",
    "pyflextrkr_core_bearing_warm_shield_mask",
    "pyflextrkr_cold_cloud_shield_label",
    "cloud_top_pressure",
    "cloud_top_height",
    "high_cloud_fraction",
    "high_cloud_mask",
    "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
    "unclassified_high_cloud_mask",
    "high_cloud_regime_code",
)

should_run_figure = partial(figure_should_run, figures_to_run)


def parse_command_line(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build the shared high-cloud classification product and its "
            "validation diagnostics."
        )
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run one simulation over a short time window using isolated data-"
            "repository caches and a testing figure directory."
        ),
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
        help="Number of classified timesteps in --test mode (default: 16).",
    )
    parser.add_argument(
        "--test-start",
        default=None,
        metavar="TIME",
        help="Optional first test time, for example 2000-10-01.",
    )
    parser.add_argument(
        "--no-write-classification",
        dest="write_classification",
        action="store_false",
        help=(
            "Reuse a current classification product without deleting or replacing it; "
            "fail if the product is missing, stale, or incomplete."
        ),
    )
    parser.add_argument(
        "--qc",
        action="store_true",
        help="Compatibility option selecting all figures, including Figure 3.",
    )
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        choices=range(1, 7),
        default=None,
        metavar="N",
        help="Run only the listed figure numbers (1-6).",
    )
    derived_group = parser.add_mutually_exclusive_group()
    derived_group.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Render from current classification and derived diagnostic caches; "
            "fail rather than recomputing an absent or stale derived product."
        ),
    )
    derived_group.add_argument(
        "--rebuild-derived-cache",
        action="store_true",
        help=(
            "Recompute the requested derived diagnostic products from the "
            "classification cache before rendering."
        ),
    )
    parser.set_defaults(write_classification=True)
    args = parser.parse_args(argv)
    if args.test_times < 1:
        parser.error("--test-times must be at least 1")
    return args


def configure_run(args):
    global TEST_MODE, TEST_TIME_STEPS, TEST_START
    global WRITE_CLASSIFICATION_PRODUCT, classification_cache_dir
    global derived_cache_dir, REBUILD_DERIVED_CACHE, FIGURES_ONLY
    global save_loc, figures_to_run
    global data_locs, sim_names, cols

    TEST_MODE = bool(args.test)
    TEST_TIME_STEPS = int(args.test_times)
    TEST_START = args.test_start
    FIGURES_ONLY = bool(args.figures_only)
    REBUILD_DERIVED_CACHE = bool(args.rebuild_derived_cache)
    WRITE_CLASSIFICATION_PRODUCT = (
        bool(args.write_classification) and not FIGURES_ONLY
    )

    if args.figures is not None:
        figures_to_run[:] = args.figures
    elif args.qc:
        figures_to_run[:] = [1, 2, 3, 4, 5, 6]
    elif TEST_MODE:
        figures_to_run[:] = TEST_DEFAULT_FIGURES

    if not TEST_MODE:
        return

    index = SIMULATION_ARGUMENTS[args.simulation]
    data_locs = [data_locs[index]]
    sim_names = [sim_names[index]]
    cols = [cols[index]]

    start_label = f"_{TEST_START}" if TEST_START else ""
    test_label = (
        f"{args.simulation}{start_label}_{TEST_TIME_STEPS:03d}times_lat40"
    )
    test_label = test_label.replace(":", "").replace(" ", "_")
    save_loc = os.path.join(default_save_loc, "testing", test_label)
    # Test products are isolated from publication caches but remain in the data
    # repository; only rendered figures belong below ~/tmp.
    classification_cache_dir = diagnostic_cache_dir(
        "classification_cache", "testing", test_label
    )
    derived_cache_dir = diagnostic_cache_dir(
        "high_cloud_diagnostics_cache", "testing", test_label
    )


def save_figure(number, fig, stem):
    if figsave is None:
        raise RuntimeError("FigureSaver has not been initialised")
    return figsave.save(fig, stem, number=number)


def save_table(df, stem):
    if save and df is not None and len(df) > 0:
        path = os.path.join(save_loc, f"{stem}.csv")
        df.to_csv(path, index=False)
        progress(f"Wrote {path}")


def _json_ready(value):
    """Return a deterministic JSON representation of cache metadata."""

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def derived_cache_configuration():
    """Return the non-time-dependent part of the derived-cache signature."""

    return _json_ready(
        {
            "derived_cache_version": DERIVED_CACHE_VERSION,
            "classification_cache_version": CLASSIFICATION_CACHE_VERSION,
            "test_mode": bool(TEST_MODE),
            "simulations": list(sim_names),
            "tropical_latitude_bounds": TROPICAL_LAT_BOUNDS,
            "cloud_structure_domains": CLOUD_STRUCTURE_DOMAIN_SPECS,
            "cloud_top_pressure_bin_width_hpa": (
                CLOUD_TOP_PRESSURE_BIN_WIDTH_HPA
            ),
            "olr_bin_width_w_m2": OLR_BIN_WIDTH_W_M2,
            "hemispheric_treatment": (
                "paired-hemisphere mean for mean scalar fields; "
                "cosine-weighted pooled absolute-latitude bands for PDFs"
            ),
            "cloud_fraction_threshold": CF_THRESH,
            "high_cloud_top_pressure_hpa": P_HIGH_HPA,
            "surface_precipitation_threshold_mm_day": P_PRECIP_MIN,
            "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
            "bootstrap_samples": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "profile_compute_height_bounds_km": PROFILE_COMPUTE_Z_BOUNDS_KM,
            "utls_layer_bounds_km": UTLS_LAYER_BOUNDS_KM,
            "rain_rate_bins_mm_h": RAIN_RATE_BINS_MM_H,
        }
    )


def initialise_derived_cache_signature(dss):
    """Describe every scientific input that invalidates derived diagnostics."""

    global DERIVED_CACHE_SIGNATURE
    time_axes = {}
    for sim_name, ds in zip(sim_names, dss):
        values = np.asarray(ds.Time.values)
        time_axes[sim_name] = {
            "count": int(values.size),
            "first": format_time_value(values[0]),
            "last": format_time_value(values[-1]),
        }
    DERIVED_CACHE_SIGNATURE = derived_cache_configuration()
    DERIVED_CACHE_SIGNATURE["time_axes"] = time_axes
    os.makedirs(derived_cache_dir, exist_ok=True)


def required_derived_products():
    """Return the cached products needed by the selected non-snapshot figures."""

    products = []
    if should_run_figure(1):
        products.append(("high_cloud_class_budget", "csv"))
    if should_run_figure(2) or should_run_figure(4):
        products.append(("high_cloud_vertical_contribution_profiles", "nc"))
    if should_run_figure(2):
        products.append(("high_cloud_structure_distributions", "nc"))
    if should_run_figure(5):
        products.append(("high_cloud_class_occurrence_maps", "nc"))
    if should_run_figure(6):
        products.append(("rain_matched_high_cloud_response", "csv"))
    return products


def initialise_derived_cache_signature_from_products(products):
    """Validate cache metadata without opening the full simulation datasets."""

    global DERIVED_CACHE_SIGNATURE
    expected = derived_cache_configuration()
    metadata_paths = [
        os.path.join(derived_cache_dir, f"{stem}.metadata.json")
        for stem, _suffix in products
    ]
    for metadata_path in metadata_paths:
        if not os.path.isfile(metadata_path):
            continue
        with open(metadata_path, encoding="utf-8") as stream:
            signature = json.load(stream)
        mismatches = [
            key for key, value in expected.items()
            if signature.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                f"Stale derived high-cloud cache metadata {metadata_path}; "
                "configuration differs for " + ", ".join(mismatches)
            )
        if "time_axes" not in signature:
            raise RuntimeError(
                f"Incomplete derived high-cloud cache metadata {metadata_path}: "
                "time_axes is missing"
            )
        DERIVED_CACHE_SIGNATURE = signature
        progress(
            "Validated derived high-cloud cache directly from "
            f"{metadata_path}"
        )
        return
    missing = metadata_paths[0] if metadata_paths else derived_cache_dir
    raise RuntimeError(
        f"No derived high-cloud cache metadata is available at {missing}; "
        "run without --figures-only once to build the selected products"
    )


def _derived_product_paths(stem, suffix):
    if DERIVED_CACHE_SIGNATURE is None:
        raise RuntimeError("Derived cache signature has not been initialised")
    product = os.path.join(derived_cache_dir, f"{stem}.{suffix}")
    metadata = os.path.join(derived_cache_dir, f"{stem}.metadata.json")
    return product, metadata


def _read_current_derived_metadata(metadata_path):
    if REBUILD_DERIVED_CACHE or not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    return metadata if metadata == DERIVED_CACHE_SIGNATURE else None


def _write_json_atomic(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(tmp_path, path)


def load_or_compute_derived_table(stem, compute):
    """Load a current CSV diagnostic product or compute it exactly once."""

    path, metadata_path = _derived_product_paths(stem, "csv")
    if os.path.isfile(path) and _read_current_derived_metadata(metadata_path):
        progress(f"Using cached high-cloud table {path}")
        return pd.read_csv(path)
    if FIGURES_ONLY:
        raise RuntimeError(
            f"Missing or stale derived high-cloud table {path}; run without "
            "--figures-only once to build it"
        )
    progress(f"Building derived high-cloud table {path}")
    table = compute()
    if table is None or not isinstance(table, pd.DataFrame):
        raise TypeError(f"{stem}: derived table builder did not return a DataFrame")
    tmp_path = path + ".tmp"
    table.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
    _write_json_atomic(metadata_path, DERIVED_CACHE_SIGNATURE)
    progress(f"Wrote derived high-cloud table {path}")
    return table


def load_or_compute_derived_dataset(stem, compute):
    """Load a current NetCDF diagnostic product or compute it exactly once."""

    path, metadata_path = _derived_product_paths(stem, "nc")
    if os.path.isfile(path) and _read_current_derived_metadata(metadata_path):
        progress(f"Using cached high-cloud dataset {path}")
        with xr.open_dataset(path) as cached:
            return cached.load()
    if FIGURES_ONLY:
        raise RuntimeError(
            f"Missing or stale derived high-cloud dataset {path}; run without "
            "--figures-only once to build it"
        )
    progress(f"Building derived high-cloud dataset {path}")
    dataset = compute()
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"{stem}: derived dataset builder did not return a Dataset")
    dataset = dataset.copy()
    dataset.attrs.update(
        derived_cache_version=DERIVED_CACHE_VERSION,
        derived_cache_signature=json.dumps(
            DERIVED_CACHE_SIGNATURE, sort_keys=True
        ),
    )
    tmp_path = path + ".tmp"
    dataset.to_netcdf(tmp_path)
    os.replace(tmp_path, path)
    _write_json_atomic(metadata_path, DERIVED_CACHE_SIGNATURE)
    progress(f"Wrote derived high-cloud dataset {path}")
    return dataset


def write_classification_method_table():
    rows = [
        dict(component="parent high cloud", implementation=f"cldfrac >= {CF_THRESH:g}; cloud-top pressure <= {P_HIGH_HPA:g} hPa", origin="ISCCP pressure category plus stated model-cloud cutoff", reference="Rossow and Schiffer (1999), BAMS", doi="10.1175/1520-0477(1999)080<2261:AIUCFI>2.0.CO;2"),
        dict(component="convective core", implementation="PyFLEXTRKR OLR-derived Tb < 225 K core pixels", origin="published 50-km configuration", reference="Feng et al. (2021), Journal of Climate", doi="10.1175/JCLI-D-20-0136.1"),
        dict(component="PyFLEXTRKR scene regions", implementation="mutually exclusive display of Tb < 225 K cold core, 241-K cold-shield shell, and connected 261-K warm-shield shell; shields do not define high-cloud classes", origin="published core/cold thresholds plus a displayed warm-shield sensitivity", reference="Feng et al. (2021); PyFLEXTRKR", doi="10.1175/JCLI-D-20-0136.1"),
        dict(component="precipitation feature", implementation=f"surface rain >= {P_PRECIP_MIN / 24.0:g} mm h-1 ({P_PRECIP_MIN:g} mm day-1)", origin="published PF definition", reference="Feng et al. (2021), Journal of Climate", doi="10.1175/JCLI-D-20-0136.1"),
        dict(component="stratiform high cloud", implementation=f"all non-core high cloud with surface rain >= {P_PRECIP_MIN / 24.0:g} mm h-1", origin="baseline core/rain decomposition", reference="Feng et al. (2021) PF rain threshold", doi="10.1175/JCLI-D-20-0136.1"),
        dict(component="non-precipitating high cloud", implementation=f"all remaining non-core high cloud with surface rain < {P_PRECIP_MIN / 24.0:g} mm h-1", origin="baseline core/rain decomposition", reference="this analysis", doi="not applicable"),
        dict(component="vertical rain attribution", implementation="none in the baseline; surface rain is associated with the high-cloud column", origin="deliberately deferred sensitivity", reference="not applicable", doi="not applicable"),
        dict(component="spatial class response", implementation="unfiltered time-mean occurrence difference for each high-cloud class; grey contours show the control occurrence", origin="class-aware analogue of published cloud-regime occurrence maps", reference="Jin et al. (2023), Nature Communications", doi="10.1038/s41467-023-39465-7"),
        dict(component="rain-matched cloud structure", implementation="cloud-top pressure and OLR conditioned on the same surface-rain-rate bins; nudged-minus-control response from 10-day block ratios", origin="combines published cloud-depth/radiative property comparisons with precipitation matching", reference="Jin et al. (2023), Nature Communications; Sakaeda et al. (2020), JGR Atmospheres", doi="10.1038/s41467-023-39465-7; 10.1029/2020JD033196"),
    ]
    save_table(pd.DataFrame(rows), "cloud_classification_method_and_references")


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def encode_time_for_netcdf(values, nc_variable):
    """Encode one decoded xarray time block using the product's CF metadata."""
    values = np.asarray(values)
    if values.size == 0 or not (
        np.issubdtype(values.dtype, np.datetime64) or values.dtype == object
    ):
        return values

    units = getattr(nc_variable, "units", None)
    if not units:
        raise ValueError("The classification product Time variable has no CF units")

    from netCDF4 import date2num

    calendar = getattr(nc_variable, "calendar", "standard")
    if np.issubdtype(values.dtype, np.datetime64):
        dates = pd.to_datetime(values).to_pydatetime()
    else:
        dates = values.tolist()
    return date2num(dates, units=units, calendar=calendar)


def write_time_batched_dataset(
    ds,
    path,
    label,
    encoding=None,
    time_batch=CLASSIFICATION_WRITE_TIME_BATCH,
    num_workers=CLASSIFICATION_WRITE_WORKERS,
):
    """Write an atomic NetCDF product without evaluating one monolithic graph.

    The zero-length template establishes an unlimited Time dimension and the
    complete schema. Each block is then computed before opening the output file,
    so input reads and NetCDF output writes cannot build an unbounded queue of
    live 3-D intermediates.
    """
    if "Time" not in ds.dims:
        raise ValueError(f"{label}: time-batched writer requires a Time dimension")
    ntime = int(ds.sizes["Time"])
    if ntime == 0:
        raise ValueError(f"{label}: refusing to write a product with no times")
    if time_batch < 1 or num_workers < 1:
        raise ValueError("time_batch and num_workers must both be positive")

    from netCDF4 import Dataset

    tmp_path = path + ".tmp"
    if os.path.exists(tmp_path):
        progress(f"{label}: removing incomplete temporary product {tmp_path}")
        os.remove(tmp_path)

    progress(
        f"{label}: initialising {path} for {ntime} times "
        f"(batch={time_batch}, workers={num_workers})"
    )
    template = ds.isel(Time=slice(0, 0))
    writer_encoding = dict(encoding or {})
    time_values = np.asarray(ds.Time.values)
    time_is_datetime = np.issubdtype(time_values.dtype, np.datetime64) or (
        time_values.dtype == object
        and time_values.size > 0
        and hasattr(time_values.flat[0], "year")
    )
    if time_is_datetime:
        # An empty datetime coordinate otherwise defaults to integer days;
        # appending sub-daily values would silently truncate them.
        time_encoding = {
            "dtype": "float64",
            "units": ds.Time.encoding.get(
                "units", "hours since 1970-01-01 00:00:00"
            ),
        }
        calendar = ds.Time.encoding.get("calendar")
        if calendar is None and time_values.dtype == object:
            calendar = getattr(time_values.flat[0], "calendar", None)
        if calendar is not None:
            time_encoding["calendar"] = calendar
        writer_encoding["Time"] = time_encoding
    template.to_netcdf(
        tmp_path,
        encoding=writer_encoding,
        unlimited_dims=["Time"],
    )

    started = time.monotonic()
    nblocks = (ntime + time_batch - 1) // time_batch
    for block_number, start in enumerate(range(0, ntime, time_batch), start=1):
        stop = min(start + time_batch, ntime)
        block_started = time.monotonic()
        progress(
            f"{label}: computing block {block_number}/{nblocks} "
            f"(times {start + 1}-{stop})"
        )
        block = ds.isel(Time=slice(start, stop)).load(
            scheduler="threads",
            num_workers=num_workers,
        )
        computed = time.monotonic()

        with Dataset(tmp_path, mode="a") as target:
            for name, variable in block.variables.items():
                if "Time" not in variable.dims:
                    continue
                if name not in target.variables:
                    raise KeyError(f"{label}: template is missing variable {name!r}")
                values = variable.values
                if name == "Time":
                    values = encode_time_for_netcdf(values, target.variables[name])
                indexer = tuple(
                    slice(start, stop) if dim == "Time" else slice(None)
                    for dim in variable.dims
                )
                target.variables[name][indexer] = values
            target.sync()

        elapsed = time.monotonic() - started
        completed_times = stop
        eta = elapsed * (ntime - completed_times) / completed_times
        progress(
            f"{label}: wrote {completed_times}/{ntime} times "
            f"({100.0 * completed_times / ntime:.1f}%); "
            f"compute {format_duration(computed - block_started)}, "
            f"write {format_duration(time.monotonic() - computed)}, "
            f"elapsed {format_duration(elapsed)}, ETA {format_duration(eta)}"
        )
        del block
        gc.collect()

    os.replace(tmp_path, path)
    progress(f"{label}: finished writing {path}")


def classification_cache_encoding(ds):
    chunk_targets = {"Time": 8, "latitude": 90, "longitude": 180}
    encoding = {}
    for name, da in ds.data_vars.items():
        entry = dict(zlib=True, complevel=1, shuffle=True)
        if np.issubdtype(da.dtype, np.floating):
            entry["dtype"] = "float32"
        entry["chunksizes"] = tuple(
            min(int(da.sizes[dim]), chunk_targets.get(dim, int(da.sizes[dim])))
            for dim in da.dims
        )
        encoding[name] = entry
    return encoding


# -----------------------------------------------------------------------------
# General data and classification helpers
# -----------------------------------------------------------------------------

subset_lat = subset_latitude
get_precip_rate = precipitation_rate_from_dataset


def select_absolute_latitude_band(da, absolute_latitude_bounds):
    """Pool paired hemispheres without averaging nonlinear PDF samples."""

    if "latitude" not in da.dims:
        raise ValueError(
            f"Cannot select an absolute-latitude band from "
            f"{getattr(da, 'name', None)!r}: latitude is absent"
        )
    if not latitude_is_hemispherically_symmetric(da.latitude):
        raise ValueError(
            f"Cannot form a paired-hemisphere sample from "
            f"{getattr(da, 'name', None)!r}: latitude is not symmetric"
        )
    lower, upper = sorted(float(value) for value in absolute_latitude_bounds)
    absolute_latitude = np.abs(np.asarray(da.latitude.values, dtype=float))
    indices = np.flatnonzero(
        (absolute_latitude >= lower) & (absolute_latitude <= upper)
    )
    if indices.size == 0:
        raise ValueError(
            f"No latitudes are available in the requested paired band "
            f"{lower:g}–{upper:g}°"
        )
    out = da.isel(latitude=indices)
    out.attrs.update(getattr(da, "attrs", {}))
    out.attrs["hemispheric_sampling"] = (
        f"cosine-weighted pooled samples at {lower:g}–{upper:g}° absolute latitude"
    )
    return out


block_slices = partial(
    time_block_slices,
    block_days=BOOTSTRAP_BLOCK_DAYS,
)


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


def bootstrap_ratio_1d(numerator, denominator, rng, n_boot=N_BOOTSTRAP):
    """Paired block bootstrap for a share with a common time denominator."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    n = min(numerator.size, denominator.size)
    numerator = numerator[:n]
    denominator = denominator[:n]
    finite = np.isfinite(numerator) & np.isfinite(denominator)
    numerator = numerator[finite]
    denominator = denominator[finite]
    if numerator.size == 0 or np.nanmean(denominator) <= 0.0:
        return dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan, n_blocks=0)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        index = rng.integers(0, numerator.size, numerator.size)
        sampled_denominator = np.nanmean(denominator[index])
        boot[i] = (
            np.nanmean(numerator[index]) / sampled_denominator
            if sampled_denominator > 0.0 else np.nan
        )
    ci_low, ci_high = np.nanpercentile(boot, CI_PERCENTILES)
    return dict(
        mean=float(np.nanmean(numerator) / np.nanmean(denominator)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_blocks=int(numerator.size),
    )


def get_cloud_fraction(ds):
    return ds.cldfrac


def full_class_structure_statistics(
    fields,
    classif,
    sim_name,
    absolute_latitude_bounds=(0.0, 20.0),
    domain_label="20°S–20°N",
    ctp_bins=None,
    olr_bins=None,
    time_batch=CLOUD_STRUCTURE_TIME_BATCH,
):
    """Accumulate class PDFs from a paired absolute-latitude band."""
    class_keys = [key for key, _label, _color in REGIME_SPECS]
    aligned = xr.align(
        select_absolute_latitude_band(
            classif["cloud_top_pressure"], absolute_latitude_bounds
        ),
        select_absolute_latitude_band(
            fields["olr"], absolute_latitude_bounds
        ),
        *[
            select_absolute_latitude_band(
                classif[key].astype(bool), absolute_latitude_bounds
            )
            for key in class_keys
        ],
        join="inner",
    )
    cloud_top_pressure, olr = aligned[:2]
    class_masks = dict(zip(class_keys, aligned[2:]))

    field_names = (
        "cloud_top_pressure",
        "olr",
    )
    accumulators = {
        key: {
            "class_weight": 0.0,
            "ttl_weight": 0.0,
            **{f"{name}_sum": 0.0 for name in field_names},
            **{f"{name}_weight": 0.0 for name in field_names},
            "ctp_histogram": (
                np.zeros(len(ctp_bins) - 1, dtype=float)
                if ctp_bins is not None else None
            ),
            "olr_histogram": (
                np.zeros(len(olr_bins) - 1, dtype=float)
                if olr_bins is not None else None
            ),
        }
        for key in class_keys
    }
    latitude_weights = np.cos(
        np.deg2rad(np.asarray(cloud_top_pressure.latitude.values, dtype=float))
    )[None, :, None]
    domain_weight = (
        cloud_top_pressure.sizes["Time"]
        * cloud_top_pressure.sizes["longitude"]
        * float(np.sum(latitude_weights))
    )
    ntime = cloud_top_pressure.sizes["Time"]
    time_slices = [
        slice(start, min(start + time_batch, ntime))
        for start in range(0, ntime, time_batch)
    ]
    started = time.monotonic()
    progress(
        f"{sim_name}: computing {domain_label} cloud structure in "
        f"{len(time_slices)} blocks of at most {time_batch} times"
    )

    for iblock, time_slice in enumerate(time_slices, start=1):
        block_reference = cloud_top_pressure.isel(Time=time_slice)
        block = xr.Dataset(
            {
                "cloud_top_pressure": block_reference,
                "olr": olr.isel(Time=time_slice),
                **{
                    key: class_masks[key].isel(Time=time_slice)
                    for key in class_keys
                },
            }
        ).load()
        block_values = {
            "cloud_top_pressure": np.asarray(
                block["cloud_top_pressure"].values, dtype=float
            ),
            "olr": np.asarray(block["olr"].values, dtype=float),
        }
        sample_weights = np.broadcast_to(
            latitude_weights, block_values["cloud_top_pressure"].shape
        )

        for key in class_keys:
            mask_values = np.asarray(block[key].values, dtype=bool)
            accumulator = accumulators[key]
            accumulator["class_weight"] += float(
                np.sum(sample_weights[mask_values])
            )
            ttl = (
                mask_values
                & np.isfinite(block_values["cloud_top_pressure"])
                & (block_values["cloud_top_pressure"] <= 180.0)
            )
            accumulator["ttl_weight"] += float(np.sum(sample_weights[ttl]))
            for field_name, values in block_values.items():
                valid = mask_values & np.isfinite(values)
                weights = sample_weights[valid]
                accumulator[f"{field_name}_sum"] += float(
                    np.sum(values[valid] * weights)
                )
                accumulator[f"{field_name}_weight"] += float(np.sum(weights))
            if ctp_bins is not None:
                valid = mask_values & np.isfinite(
                    block_values["cloud_top_pressure"]
                )
                histogram, _ = np.histogram(
                    block_values["cloud_top_pressure"][valid],
                    bins=ctp_bins,
                    weights=sample_weights[valid],
                )
                accumulator["ctp_histogram"] += histogram
            if olr_bins is not None:
                valid = mask_values & np.isfinite(block_values["olr"])
                histogram, _ = np.histogram(
                    block_values["olr"][valid],
                    bins=olr_bins,
                    weights=sample_weights[valid],
                )
                accumulator["olr_histogram"] += histogram

        del block, block_values
        elapsed = time.monotonic() - started
        eta = elapsed / iblock * (len(time_slices) - iblock)
        progress(
            f"{sim_name} {domain_label}: cloud-structure block "
            f"{iblock}/{len(time_slices)} "
            f"({100.0 * iblock / len(time_slices):.0f}%); "
            f"elapsed {format_duration(elapsed)}, ETA {format_duration(eta)}"
        )
        if iblock % 10 == 0:
            gc.collect()

    statistics = {}
    for key, accumulator in accumulators.items():
        class_weight = accumulator["class_weight"]
        result = {
            "occurrence_fraction": (
                class_weight / domain_weight if domain_weight > 0.0 else np.nan
            ),
            "ttl_fraction": (
                accumulator["ttl_weight"] / class_weight
                if class_weight > 0.0 else np.nan
            ),
        }
        for field_name in field_names:
            weight = accumulator[f"{field_name}_weight"]
            result[f"mean_{field_name}"] = (
                accumulator[f"{field_name}_sum"] / weight
                if weight > 0.0 else np.nan
            )
        for histogram_name in ("ctp_histogram", "olr_histogram"):
            histogram = accumulator[histogram_name]
            if histogram is not None:
                result[histogram_name] = (
                    histogram / histogram.sum()
                    if histogram.sum() > 0.0 else histogram
                )
        statistics[key] = result
    progress(f"{sim_name}: {domain_label} cloud-structure statistics ready")
    return statistics


def classification_thresholds(
    cf_thresh=CF_THRESH,
    precip_min=P_PRECIP_MIN,
):
    return HighCloudThresholds(
        cf_thresh=float(cf_thresh),
        high_cloud_top_pressure_hpa=P_HIGH_HPA,
        surface_precip_min_mm_day=float(precip_min),
    )


def classify_high_cloud(
    ds,
    cldfrac,
    fields,
    cf_thresh=CF_THRESH,
    precip_min=P_PRECIP_MIN,
):
    """Shared 0.5-degree high-cloud precipitation-regime decomposition."""
    return classify_high_cloud_from_dataset(
        ds,
        precip=fields["precip"],
        cldfrac=cldfrac,
        thresholds=classification_thresholds(
            cf_thresh,
            precip_min=precip_min,
        ),
    )


classification_dataset = partial(
    shared_classification_dataset,
    thresholds=classification_thresholds(),
)


cached_classification_is_current = partial(
    shared_cached_classification_is_current,
    thresholds=classification_thresholds(),
)


def open_cached_classification(cache_path, sim_name, expected_time=None):
    cached = xr.open_dataset(cache_path, chunks={})
    if not cached_classification_is_current(cached):
        progress(f"{sim_name}: cached high-cloud classification is stale: {cache_path}")
        cached.close()
        return None
    missing = [name for name in CLASSIFICATION_CACHE_VARS if name not in cached.data_vars]
    if missing:
        progress(
            f"{sim_name}: current-version classification cache is incomplete "
            f"({', '.join(missing)}): {cache_path}"
        )
        cached.close()
        return None
    if expected_time is not None:
        cached_time = np.asarray(cached.Time.values) if "Time" in cached.coords else None
        expected_time = np.asarray(expected_time)
        if (
            cached_time is None
            or cached_time.shape != expected_time.shape
            or not np.array_equal(cached_time, expected_time)
        ):
            cached_count = 0 if cached_time is None else cached_time.size
            progress(
                f"{sim_name}: classification cache has the wrong Time axis "
                f"({cached_count} cached, {expected_time.size} expected): {cache_path}"
            )
            cached.close()
            return None
    variables = {name: cached[name] for name in cached.data_vars}
    corrected_warm, reconstructed = corrected_cached_warm_shield_masks(cached)
    variables.update(corrected_warm)
    if reconstructed:
        progress(
            f"{sim_name}: reconstructing corrected warm-shield validation "
            "masks lazily; retaining the expensive classification cache"
        )
    progress(f"{sim_name}: using cached high-cloud classification {cache_path}")
    return variables


def classification_product_path(sim_name):
    filename = f"{sim_name.replace(' ', '_')}_classified_high_cloud.nc"
    return os.path.join(classification_cache_dir, filename)


def get_or_build_classification(ds, fields, sim_name):
    cache_path = classification_product_path(sim_name)
    precip = fields["precip"]

    if os.path.exists(cache_path):
        cached = open_cached_classification(
            cache_path,
            sim_name,
            expected_time=precip.Time.values,
        )
        if cached is not None:
            return cached
        if not WRITE_CLASSIFICATION_PRODUCT:
            raise RuntimeError(
                f"{sim_name}: --no-write-classification requires a current, "
                f"complete classification product: {cache_path}. Run without "
                "--no-write-classification to rebuild it."
            )
        progress(
            f"{sim_name}: rebuilding stale or incomplete high-cloud "
            f"classification {cache_path}"
        )
    else:
        if not WRITE_CLASSIFICATION_PRODUCT:
            raise RuntimeError(
                f"{sim_name}: --no-write-classification requires an existing "
                f"classification product: {cache_path}. Run without "
                "--no-write-classification to create it."
            )
        progress(
            f"{sim_name}: no high-cloud classification cache found; "
            f"building {cache_path}"
        )

    progress(f"{sim_name}: preparing high-cloud classification")
    cf = get_cloud_fraction(ds).sel(Time=precip.Time)

    progress(f"{sim_name}: building high-cloud classification graph")
    classif = classify_high_cloud(ds, cf, {"precip": precip})
    progress(f"{sim_name}: high-cloud classification graph built")

    if WRITE_CLASSIFICATION_PRODUCT:
        cached_classif = {name: classif[name] for name in CLASSIFICATION_CACHE_VARS}
        fds = classification_dataset(cached_classif)
        encoding = classification_cache_encoding(fds)
        write_time_batched_dataset(
            fds,
            cache_path,
            label=f"{sim_name}: high-cloud classification product",
            encoding=encoding,
        )
        cached = open_cached_classification(
            cache_path,
            sim_name,
            expected_time=precip.Time.values,
        )
        if cached is not None:
            return cached
    return classif


def class_overlap_sanity(classifications):
    rows = []
    exclusive_keys = [
        "convective_precipitating_high_cloud_mask",
        "stratiform_precipitating_high_cloud_mask",
        "nonprecipitating_high_cloud_mask",
        "unclassified_high_cloud_mask",
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
        for key in exclusive_keys:
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

        class_union = xr.zeros_like(high, dtype=bool)
        for key in exclusive_keys:
            class_union = class_union | classif[key].astype(bool)
        missing_from_classes = weighted_area_time_mean(
            high & ~class_union, TROPICAL_LAT_BOUNDS
        )
        classified_outside_high = weighted_area_time_mean(
            class_union & ~high, TROPICAL_LAT_BOUNDS
        )
        rows.extend(
            [
                dict(
                    simulation=sim_name,
                    metric="high_cloud_missing_from_exhaustive_classes",
                    value=missing_from_classes,
                    denominator="all tropical grid-cell/time samples",
                    note="must be exactly zero",
                ),
                dict(
                    simulation=sim_name,
                    metric="classified_columns_outside_high_cloud",
                    value=classified_outside_high,
                    denominator="all tropical grid-cell/time samples",
                    note="must be exactly zero",
                ),
            ]
        )
        if missing_from_classes > 1.0e-12 or classified_outside_high > 1.0e-12:
            raise RuntimeError(
                f"{sim_name}: high-cloud classes fail exhaustive closure "
                f"(missing={missing_from_classes:.3e}, outside={classified_outside_high:.3e})"
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
#     runs, with the difference carried mainly by PyFLEXTRKR cores,
#     stratiform precipitating, or non-precipitating high cloud.
#   * Invalid/unclassified high cloud should be near zero; a substantial value
#     indicates missing precipitation-component data rather than a fourth regime.
#   * Error bars are block-resampled time-chunk consistency intervals, not
#     formal observational confidence intervals.


def compute_budget_table(classifications, fields_by_run):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    block_values = {}
    control_name = sim_names[0]
    fields_by_name = dict(zip(sim_names, fields_by_run))

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

        high_key = "high_cloud_mask"
        high_fraction = weighted_area_time_mean(
            classif[high_key], TROPICAL_LAT_BOUNDS
        )
        for key, label, _color in REGIME_SPECS:
            stats = bootstrap_ratio_1d(
                block_values[(sim_name, key)],
                block_values[(sim_name, high_key)],
                rng,
            )
            class_fraction = weighted_area_time_mean(
                classif[key], TROPICAL_LAT_BOUNDS
            )
            rows.append(
                dict(
                    simulation=sim_name,
                    class_key=key,
                    class_label=label,
                    statistic="share_of_high_cloud",
                    value=(
                        class_fraction / high_fraction
                        if high_fraction > 0.0 else np.nan
                    ),
                    block_mean=stats["mean"],
                    ci_low=stats["ci_low"],
                    ci_high=stats["ci_high"],
                    n_blocks=stats["n_blocks"],
                    denominator="total tropical high-cloud area-time occurrence",
                )
            )

        stratiform_mask = classif[
            "stratiform_precipitating_high_cloud_mask"
        ].astype(bool)
        nonprecip_mask = classif["nonprecipitating_high_cloud_mask"].astype(bool)
        noncore_mask = stratiform_mask | nonprecip_mask
        noncore_blocks = block_area_fraction_values(noncore_mask)
        noncore_stats = bootstrap_ratio_1d(
            noncore_blocks,
            block_values[(sim_name, high_key)],
            rng,
        )
        noncore_fraction = weighted_area_time_mean(
            noncore_mask, TROPICAL_LAT_BOUNDS
        )
        rows.append(
            dict(
                simulation=sim_name,
                class_key="noncore_high_cloud_aggregate",
                class_label="Non-core aggregate",
                statistic="share_of_high_cloud",
                value=(
                    noncore_fraction / high_fraction
                    if high_fraction > 0.0 else np.nan
                ),
                block_mean=noncore_stats["mean"],
                ci_low=noncore_stats["ci_low"],
                ci_high=noncore_stats["ci_high"],
                n_blocks=noncore_stats["n_blocks"],
                denominator="total tropical high-cloud area-time occurrence",
            )
        )
        nonprecip_of_noncore_stats = bootstrap_ratio_1d(
            block_values[(sim_name, "nonprecipitating_high_cloud_mask")],
            noncore_blocks,
            rng,
        )
        nonprecip_fraction = weighted_area_time_mean(
            nonprecip_mask, TROPICAL_LAT_BOUNDS
        )
        rows.append(
            dict(
                simulation=sim_name,
                class_key="nonprecipitating_high_cloud_mask",
                class_label="Non-precipitating",
                statistic="share_of_noncore_high_cloud",
                value=(
                    nonprecip_fraction / noncore_fraction
                    if noncore_fraction > 0.0 else np.nan
                ),
                block_mean=nonprecip_of_noncore_stats["mean"],
                ci_low=nonprecip_of_noncore_stats["ci_low"],
                ci_high=nonprecip_of_noncore_stats["ci_high"],
                n_blocks=nonprecip_of_noncore_stats["n_blocks"],
                denominator="stratiform plus non-precipitating tropical high-cloud area-time occurrence",
            )
        )

        shield_key = "pyflextrkr_core_bearing_warm_shield_mask"
        if shield_key in classif:
            nonprecip_aligned, shield = xr.align(
                nonprecip_mask,
                classif[shield_key].fillna(False).astype(bool),
                join="inner",
            )
            nonprecip_shield_masks = {
                "nonprecip_within_core_bearing_warm_shield": (
                    nonprecip_aligned & shield
                ),
                "nonprecip_outside_core_bearing_warm_shield": (
                    nonprecip_aligned & ~shield
                ),
            }
            nonprecip_blocks = block_area_fraction_values(nonprecip_aligned)
            nonprecip_aligned_fraction = weighted_area_time_mean(
                nonprecip_aligned, TROPICAL_LAT_BOUNDS
            )
            for key, label, _color in NONPRECIP_SHIELD_SPECS:
                mask = nonprecip_shield_masks[key]
                mask_fraction = weighted_area_time_mean(
                    mask, TROPICAL_LAT_BOUNDS
                )
                stats = bootstrap_ratio_1d(
                    block_area_fraction_values(mask),
                    nonprecip_blocks,
                    rng,
                )
                rows.append(
                    dict(
                        simulation=sim_name,
                        class_key=key,
                        class_label=label,
                        statistic="share_of_nonprecipitating_high_cloud",
                        value=(
                            mask_fraction / nonprecip_aligned_fraction
                            if nonprecip_aligned_fraction > 0.0 else np.nan
                        ),
                        block_mean=stats["mean"],
                        ci_low=stats["ci_low"],
                        ci_high=stats["ci_high"],
                        n_blocks=stats["n_blocks"],
                        denominator="tropical non-precipitating high-cloud area-time occurrence",
                    )
                )

        precip = fields_by_name[sim_name]["precip"]
        core, stratiform, precip = xr.align(
            classif["convective_precipitating_high_cloud_mask"].astype(bool),
            classif["stratiform_precipitating_high_cloud_mask"].astype(bool),
            precip,
            join="inner",
        )
        rainy_core = (
            core & precip.notnull() & (precip >= P_PRECIP_MIN)
        ).fillna(False)
        rainy_core.name = "rainy_pyflextrkr_core_mask"
        precipitating_high_cloud = rainy_core | stratiform
        precipitating_fraction = weighted_area_time_mean(
            precipitating_high_cloud, TROPICAL_LAT_BOUNDS
        )
        precipitating_blocks = block_area_fraction_values(
            precipitating_high_cloud
        )
        precipitating_masks = {
            "rainy_pyflextrkr_core_mask": rainy_core,
            "stratiform_precipitating_high_cloud_mask": stratiform,
        }
        for key, label, _color in PRECIPITATING_COMPOSITION_SPECS:
            mask = precipitating_masks[key]
            mask_fraction = weighted_area_time_mean(mask, TROPICAL_LAT_BOUNDS)
            stats = bootstrap_ratio_1d(
                block_area_fraction_values(mask),
                precipitating_blocks,
                rng,
            )
            rows.append(
                dict(
                    simulation=sim_name,
                    class_key=key,
                    class_label=label,
                    statistic="share_of_precipitating_high_cloud",
                    value=(
                        mask_fraction / precipitating_fraction
                        if precipitating_fraction > 0.0 else np.nan
                    ),
                    block_mean=stats["mean"],
                    ci_low=stats["ci_low"],
                    ci_high=stats["ci_high"],
                    n_blocks=stats["n_blocks"],
                    denominator=(
                        "tropical high-cloud area-time with surface rain >= "
                        f"{P_PRECIP_MIN / 24.0:g} mm h-1"
                    ),
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
    save_table(
        pd.DataFrame(LITERATURE_CLOUD_PERCENTAGES),
        "high_cloud_literature_validation_values",
    )
    return df


def draw_errorbarh(ax, y, value, lo, hi, color):
    if not np.isfinite(value) or not np.isfinite(lo) or not np.isfinite(hi):
        return
    ax.errorbar(
        value,
        y,
        xerr=np.array([[max(value - lo, 0.0)], [max(hi - value, 0.0)]]),
        color=color,
        fmt="none",
        lw=1.1,
        capsize=2,
        zorder=4,
    )


def label_percentage(ax, x, y, suffix=""):
    if not np.isfinite(x):
        return
    ax.annotate(
        f"{x:.1f}{suffix}",
        xy=(x, y),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=6.5,
        color="0.12",
        clip_on=False,
        zorder=6,
    )


BAR_CATEGORY_LABELS = {
    "Total high cloud": "Total\nhigh cloud",
    "Deep convection": "Deep\nconvection",
    "Stratiform cloud": "Stratiform\ncloud",
    "Non-precipitating cloud": "Non-precipitating\ncloud",
    (
        "Stratiform + non-precipitating cloud"
    ): "Stratiform +\nnon-precipitating cloud",
    "Inside 261-K convective shield": "Inside 261-K\nconvective shield",
    "Outside 261-K convective shield": "Outside 261-K\nconvective shield",
}


def two_line_bar_labels(labels):
    """Return compact category labels without removing scientific meaning."""

    return [BAR_CATEGORY_LABELS.get(label, label) for label in labels]


def draw_literature_legend(legend_axis, source_axis, simulation_names):
    """Put a panel's literature markers in a dedicated boxed legend."""

    handles, labels = source_axis.get_legend_handles_labels()
    literature_entries = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label not in simulation_names
    ]
    legend_axis.axis("off")
    if not literature_entries:
        return
    legend_axis.legend(
        [entry[0] for entry in literature_entries],
        [entry[1] for entry in literature_entries],
        loc="center",
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="0.55",
        framealpha=1.0,
        fontsize=6.5,
    )


def plot_class_budget(classifications, fields_by_run, budget=None):
    if not should_run_figure(1):
        return
    progress("Starting Figure 1: high_cloud_class_budget")
    if budget is None:
        budget = compute_budget_table(classifications, fields_by_run)
    save_table(budget, "high_cloud_class_budget")
    if len(budget) == 0:
        print("Skipping Figure 1: high-cloud classifications are unavailable.")
        return

    occurrence_labels = [label for _key, label, _color in BUDGET_SPECS]
    occurrence_keys = [key for key, _label, _color in BUDGET_SPECS]
    composition_labels = [label for _key, label, _color in REGIME_SPECS]
    composition_keys = [key for key, _label, _color in REGIME_SPECS]
    precip_labels = [
        label for _key, label, _color in PRECIPITATING_COMPOSITION_SPECS
    ]
    precip_keys = [
        key for key, _label, _color in PRECIPITATING_COMPOSITION_SPECS
    ]
    cirrus_comparison_labels = [
        label for _key, label, _color in CIRRUS_COMPARISON_SPECS
    ]
    cirrus_comparison_keys = [
        key for key, _label, _color in CIRRUS_COMPARISON_SPECS
    ]
    shield_association_labels = [
        label for _key, label, _color in NONPRECIP_SHIELD_SPECS
    ]
    shield_association_keys = [
        key for key, _label, _color in NONPRECIP_SHIELD_SPECS
    ]
    occurrence_y = np.arange(len(occurrence_keys))
    composition_y = np.arange(len(composition_keys))
    precip_y = np.arange(len(precip_keys))
    cirrus_comparison_y = np.arange(len(cirrus_comparison_keys))
    shield_association_y = np.arange(len(shield_association_keys))
    height = 0.22

    has_differences = len(sim_names) > 1
    population_fig = plt.figure(
        figsize=(12.0, 6.2), constrained_layout=True
    )
    population_grid = population_fig.add_gridspec(
        3, 3, height_ratios=(1.0, 0.34, 0.12)
    )
    occurrence_ax = population_fig.add_subplot(population_grid[0, 0])
    difference_ax = population_fig.add_subplot(population_grid[0, 1])
    composition_ax = population_fig.add_subplot(population_grid[0, 2])
    population_literature_axes = [
        population_fig.add_subplot(population_grid[1, column])
        for column in range(3)
    ]
    population_legend_ax = population_fig.add_subplot(population_grid[2, :])
    population_legend_ax.axis("off")

    attribution_fig = plt.figure(
        figsize=(12.0, 6.2), constrained_layout=True
    )
    attribution_grid = attribution_fig.add_gridspec(
        3, 3, height_ratios=(1.0, 0.24, 0.12)
    )
    precip_ax = attribution_fig.add_subplot(attribution_grid[0, 0])
    cirrus_comparison_ax = attribution_fig.add_subplot(
        attribution_grid[0, 1]
    )
    shield_association_ax = attribution_fig.add_subplot(
        attribution_grid[0, 2]
    )
    attribution_literature_axes = [
        attribution_fig.add_subplot(attribution_grid[1, column])
        for column in range(3)
    ]
    attribution_legend_ax = attribution_fig.add_subplot(
        attribution_grid[2, :]
    )
    attribution_legend_ax.axis("off")
    abs_df = budget[budget.statistic == "absolute_fraction"].set_index(["simulation", "class_key"])
    composition_df = budget[budget.statistic == "share_of_high_cloud"].set_index(["simulation", "class_key"])
    precip_composition_df = budget[
        budget.statistic == "share_of_precipitating_high_cloud"
    ].set_index(["simulation", "class_key"])
    shield_association_df = budget[
        budget.statistic == "share_of_nonprecipitating_high_cloud"
    ].set_index(["simulation", "class_key"])
    diff_df = budget[budget.statistic == "difference_from_control"].set_index(["simulation", "class_key"])

    for i, sim_name in enumerate(sim_names):
        occurrence_yy = occurrence_y + (i - (len(sim_names) - 1) / 2.0) * height
        occurrence_values = [
            100.0 * abs_df.loc[(sim_name, key), "value"] if (sim_name, key) in abs_df.index else np.nan
            for key in occurrence_keys
        ]
        occurrence_ax.barh(
            occurrence_yy, occurrence_values, height=height,
            color=cols[i], alpha=0.82, label=sim_name,
        )
        for j, key in enumerate(occurrence_keys):
            if (sim_name, key) in abs_df.index:
                row = abs_df.loc[(sim_name, key)]
                draw_errorbarh(
                    occurrence_ax, occurrence_yy[j], 100.0 * row["value"],
                    100.0 * row["ci_low"], 100.0 * row["ci_high"], color="0.15"
                )
                label_percentage(
                    occurrence_ax, 100.0 * row["value"], occurrence_yy[j]
                )

        composition_yy = composition_y + (i - (len(sim_names) - 1) / 2.0) * height
        composition_values = [
            100.0 * composition_df.loc[(sim_name, key), "value"]
            if (sim_name, key) in composition_df.index else np.nan
            for key in composition_keys
        ]
        composition_ax.barh(
            composition_yy, composition_values, height=height,
            color=cols[i], alpha=0.82, label=sim_name,
        )
        for j, key in enumerate(composition_keys):
            if (sim_name, key) in composition_df.index:
                row = composition_df.loc[(sim_name, key)]
                draw_errorbarh(
                    composition_ax, composition_yy[j], 100.0 * row["value"],
                    100.0 * row["ci_low"], 100.0 * row["ci_high"], color="0.15"
                )
                label_percentage(
                    composition_ax, 100.0 * row["value"], composition_yy[j]
                )

        precip_yy = precip_y + (i - (len(sim_names) - 1) / 2.0) * height
        precip_values = [
            100.0 * precip_composition_df.loc[(sim_name, key), "value"]
            if (sim_name, key) in precip_composition_df.index else np.nan
            for key in precip_keys
        ]
        precip_ax.barh(
            precip_yy, precip_values, height=height,
            color=cols[i], alpha=0.82, label=sim_name,
        )
        for j, key in enumerate(precip_keys):
            if (sim_name, key) in precip_composition_df.index:
                row = precip_composition_df.loc[(sim_name, key)]
                draw_errorbarh(
                    precip_ax, precip_yy[j], 100.0 * row["value"],
                    100.0 * row["ci_low"], 100.0 * row["ci_high"], color="0.15"
                )
                label_percentage(
                    precip_ax, 100.0 * row["value"], precip_yy[j]
                )

        cirrus_yy = (
            cirrus_comparison_y
            + (i - (len(sim_names) - 1) / 2.0) * height
        )
        cirrus_values = [
            100.0 * composition_df.loc[(sim_name, key), "value"]
            if (sim_name, key) in composition_df.index else np.nan
            for key in cirrus_comparison_keys
        ]
        cirrus_comparison_ax.barh(
            cirrus_yy, cirrus_values, height=height,
            color=cols[i], alpha=0.82, label=sim_name,
        )
        for j, key in enumerate(cirrus_comparison_keys):
            if (sim_name, key) in composition_df.index:
                row = composition_df.loc[(sim_name, key)]
                draw_errorbarh(
                    cirrus_comparison_ax, cirrus_yy[j], 100.0 * row["value"],
                    100.0 * row["ci_low"], 100.0 * row["ci_high"], color="0.15"
                )
                label_percentage(
                    cirrus_comparison_ax, 100.0 * row["value"], cirrus_yy[j]
                )

        shield_yy = (
            shield_association_y
            + (i - (len(sim_names) - 1) / 2.0) * height
        )
        shield_values = [
            100.0 * shield_association_df.loc[(sim_name, key), "value"]
            if (sim_name, key) in shield_association_df.index else np.nan
            for key in shield_association_keys
        ]
        shield_association_ax.barh(
            shield_yy, shield_values, height=height,
            color=cols[i], alpha=0.82, label=sim_name,
        )
        for j, key in enumerate(shield_association_keys):
            if (sim_name, key) in shield_association_df.index:
                row = shield_association_df.loc[(sim_name, key)]
                draw_errorbarh(
                    shield_association_ax, shield_yy[j], 100.0 * row["value"],
                    100.0 * row["ci_low"], 100.0 * row["ci_high"], color="0.15"
                )
                label_percentage(
                    shield_association_ax, 100.0 * row["value"], shield_yy[j]
                )

    if has_differences:
        for i, sim_name in enumerate(sim_names[1:], start=1):
            yy = occurrence_y + (i - 1.5) * height
            vals = [
                100.0 * diff_df.loc[(sim_name, key), "value"] if (sim_name, key) in diff_df.index else np.nan
                for key in occurrence_keys
            ]
            difference_ax.barh(
                yy, vals, height=height, color=cols[i], alpha=0.82,
                label=f"{sim_name} - control",
            )
            for j, key in enumerate(occurrence_keys):
                if (sim_name, key) in diff_df.index:
                    row = diff_df.loc[(sim_name, key)]
                    draw_errorbarh(
                        difference_ax, yy[j], 100.0 * row["value"],
                        100.0 * row["ci_low"], 100.0 * row["ci_high"], color="0.15"
                    )

    occurrence_ax.scatter(
        [35.0], [occurrence_y[0]], marker="D", s=52,
        facecolor="white", edgecolor="#7a4e00", linewidth=1.4, zorder=7,
        label=(
            "AIRS/IASI high cloud — Stubenrauch et al. (2023)"
        ),
    )
    occurrence_ax.annotate(
        "35", (35.0, occurrence_y[0]), xytext=(0, 8),
        textcoords="offset points", ha="center", va="bottom",
        fontsize=6.5, color="#7a4e00",
    )
    occurrence_ax.scatter(
        [22.7], [occurrence_y[0]], marker="v", s=54,
        facecolor="white", edgecolor="#375a7f", linewidth=1.3, zorder=7,
        label=(
            "MODIS high ice cloud — Hong et al. (2007)"
        ),
    )
    occurrence_ax.annotate(
        "22.7", (22.7, occurrence_y[0]), xytext=(0, -10),
        textcoords="offset points", ha="center", va="top",
        fontsize=6.5, color="#375a7f",
    )
    nonprecip_occurrence_y = occurrence_y[
        occurrence_keys.index("nonprecipitating_high_cloud_mask")
    ]
    occurrence_ax.plot(
        [38.4, 43.5],
        [nonprecip_occurrence_y, nonprecip_occurrence_y],
        color="#7a4e00", lw=4.0, alpha=0.85, solid_capstyle="round",
        zorder=7,
        label=(
            "CALIOP cirrus — Martins et al. (2011)"
        ),
    )
    occurrence_ax.scatter(
        [38.4, 43.5], [nonprecip_occurrence_y, nonprecip_occurrence_y],
        marker="|", s=70, color="#7a4e00", linewidth=1.1, zorder=8,
    )
    deep_convection_y = composition_y[
        composition_keys.index("convective_precipitating_high_cloud_mask")
    ]
    composition_ax.scatter(
        [20.0], [deep_convection_y], marker="<", s=60,
        facecolor="white", edgecolor="#7a4e00", linewidth=1.4, zorder=7,
        label="MODIS deep cloud — Hong et al. (2007)",
    )
    composition_ax.annotate(
        "<20", (20.0, deep_convection_y), xytext=(0, 8),
        textcoords="offset points", ha="center", va="bottom",
        fontsize=6.5, color="#7a4e00",
    )
    precip_ax.scatter(
        [27.0, 73.0], precip_y, marker="D", s=52,
        facecolor="white", edgecolor="#7a4e00", linewidth=1.4, zorder=7,
        label=(
            "TRMM rain regimes — Schumacher & Houze (2003)"
        ),
    )
    for value, y_value in zip((27.0, 73.0), precip_y):
        precip_ax.annotate(
            f"{value:.0f}", (value, y_value), xytext=(0, 8),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=6.5, color="#7a4e00",
        )
    cirrus_comparison_ax.scatter(
        [80.0], [cirrus_comparison_y[0]], marker=">", s=60,
        facecolor="white", edgecolor="#7a4e00", linewidth=1.4, zorder=7,
        label=(
            "MODIS cirriform cloud — Hong et al. (2007)"
        ),
    )
    cirrus_comparison_ax.annotate(
        ">80", (80.0, cirrus_comparison_y[0]), xytext=(0, 8),
        textcoords="offset points", ha="center", va="bottom",
        fontsize=6.5, color="#7a4e00",
    )

    occurrence_ax.set_yticks(occurrence_y)
    occurrence_ax.set_yticklabels(two_line_bar_labels(occurrence_labels))
    occurrence_ax.invert_yaxis()
    occurrence_ax.set_xlabel("Tropical area-time occurrence (%)")
    occurrence_ax.set_title("(a) High-cloud occurrence")
    occurrence_ax.grid(axis="x", alpha=0.22)
    occurrence_ax.margins(x=0.14)

    composition_ax.set_yticks(composition_y)
    composition_ax.set_yticklabels(two_line_bar_labels(composition_labels))
    composition_ax.invert_yaxis()
    composition_ax.set_xlabel("Share of total high cloud (%)")
    composition_ax.set_title("(c) High-cloud composition")
    composition_ax.grid(axis="x", alpha=0.22)
    composition_ax.margins(x=0.14)

    precip_ax.set_yticks(precip_y)
    precip_ax.set_yticklabels(two_line_bar_labels(precip_labels))
    precip_ax.invert_yaxis()
    precip_ax.set_xlim(0.0, 108.0)
    precip_ax.set_xlabel("Share of precipitating high cloud (%)")
    precip_ax.set_title("(a) Precipitating-cloud composition")
    precip_ax.grid(axis="x", alpha=0.22)

    if has_differences:
        difference_ax.set_yticks(occurrence_y)
        difference_ax.set_yticklabels(
            two_line_bar_labels(occurrence_labels)
        )
        difference_ax.invert_yaxis()
        difference_ax.axvline(0.0, color="0.45", lw=0.8)
        difference_ax.set_xlabel("Change from control (percentage points)")
        difference_ax.set_title("(b) Cloud-class occurrence change")
        difference_ax.grid(axis="x", alpha=0.22)
        difference_ax.legend(frameon=False, fontsize=8)
    else:
        difference_ax.axis("off")
        difference_ax.text(
            0.5, 0.5,
            "Single-simulation test\n(no response relative to control)",
            transform=difference_ax.transAxes,
            ha="center", va="center", color="0.35", fontsize=10,
        )

    cirrus_comparison_ax.set_yticks(cirrus_comparison_y)
    cirrus_comparison_ax.set_yticklabels(
        two_line_bar_labels(cirrus_comparison_labels)
    )
    cirrus_comparison_ax.invert_yaxis()
    cirrus_comparison_ax.set_xlim(0.0, 108.0)
    cirrus_comparison_ax.set_xlabel("Share of total high cloud (%)")
    cirrus_comparison_ax.set_title("(b) Cirrus comparison")
    cirrus_comparison_ax.grid(axis="x", alpha=0.22)

    shield_association_ax.set_yticks(shield_association_y)
    shield_association_ax.set_yticklabels(
        two_line_bar_labels(shield_association_labels)
    )
    shield_association_ax.invert_yaxis()
    shield_association_ax.set_xlim(0.0, 108.0)
    shield_association_ax.set_xlabel(
        "Share of non-precipitating cloud (%)"
    )
    shield_association_ax.set_title(
        "(c) Non-precipitating cloud location"
    )
    shield_association_ax.grid(axis="x", alpha=0.22)
    if len(shield_association_df) == 0:
        shield_association_ax.text(
            0.5, 0.5, "Shield association unavailable",
            transform=shield_association_ax.transAxes,
            ha="center", va="center", color="0.35", fontsize=9,
        )

    population_literature_axes[1].axis("off")
    for source_axis, legend_axis in (
        (occurrence_ax, population_literature_axes[0]),
        (composition_ax, population_literature_axes[2]),
    ):
        draw_literature_legend(legend_axis, source_axis, sim_names)
    for source_axis, legend_axis in zip(
        (precip_ax, cirrus_comparison_ax, shield_association_ax),
        attribution_literature_axes,
    ):
        draw_literature_legend(legend_axis, source_axis, sim_names)

    simulation_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=7,
            label=sim_name,
        )
        for sim_name, color in zip(sim_names, cols)
    ]
    for legend_axis in (population_legend_ax, attribution_legend_ax):
        legend_axis.legend(
            handles=simulation_handles,
            loc="center",
            ncol=len(simulation_handles),
            frameon=False,
            fontsize=8,
            title="Simulations",
            title_fontsize=8,
        )
    all_axes = (
        occurrence_ax,
        difference_ax,
        composition_ax,
        precip_ax,
        cirrus_comparison_ax,
        shield_association_ax,
    )
    for axis in all_axes:
        axis.title.set_fontsize(10)
        axis.tick_params(labelsize=8)

    population_fig.suptitle(
        "High-cloud occurrence and composition (20°S–20°N)\n"
        "error bars: 95% block intervals; markers: literature values",
        fontsize=11,
    )
    attribution_fig.suptitle(
        "High-cloud regime attribution (20°S–20°N)\n"
        f"Panel (a): rain ≥ {P_PRECIP_MIN / 24.0:g} mm h$^{{-1}}$; "
        "error bars: 95% block intervals; markers: literature values",
        fontsize=11,
    )
    save_figure(
        1,
        population_fig,
        "high_cloud_occurrence_and_composition",
    )
    save_figure(
        1,
        attribution_fig,
        "high_cloud_regime_attribution",
    )
    plt.close(population_fig)
    plt.close(attribution_fig)
    progress("Done with Figure 1 high-cloud budget figures")


# -----------------------------------------------------------------------------
# Figure 3. Classification validation snapshots
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * The rain panel shows the precipitating columns used by the regime split.
#   * The class panel should partition nearly every valid high-cloud column into
#     convective, stratiform, or non-precipitating cloud.
#   * The two vertical slices show cloud fraction through height, with the
#     diagnosed column regime shown as a narrow annotation band.


def format_time_value(value):
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.datetime64):
        return np.datetime_as_string(arr, unit="h")
    return str(value)


def representative_validation_time(classif):
    high = subset_lat(classif["high_cloud_mask"].astype(float), TROPICAL_LAT_BOUNDS)
    nonprecipitating = subset_lat(classif["nonprecipitating_high_cloud_mask"].astype(float), TROPICAL_LAT_BOUNDS)
    stratiform = subset_lat(classif["stratiform_precipitating_high_cloud_mask"].astype(float), TROPICAL_LAT_BOUNDS)
    convective = subset_lat(classif["convective_precipitating_high_cloud_mask"].astype(float), TROPICAL_LAT_BOUNDS)
    if "Time" not in high.dims:
        return None
    weights = np.cos(np.deg2rad(high.latitude))
    score = (high + nonprecipitating + 1.25 * stratiform + 1.75 * convective).weighted(weights).mean(
        dim=("latitude", "longitude"),
        skipna=True,
    )
    idx = int(score.argmax(dim="Time").values)
    return high.Time.isel(Time=idx).values


def class_code_from_masks(high, convective, stratiform, nonprecipitating, unclassified):
    code = np.zeros(high.shape, dtype=np.int16)
    code[np.asarray(convective.values) >= 0.5] = 1
    code[np.asarray(stratiform.values) >= 0.5] = 2
    code[np.asarray(nonprecipitating.values) >= 0.5] = 3
    code[np.asarray(unclassified.values) >= 0.5] = 4
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


def focus_point_for_snapshot(precip, high, convective, stratiform, nonprecipitating):
    rain_object = (precip >= P_PRECIP_MIN).astype(float)
    score = (
        0.20 * high.astype(float)
        + 1.75 * convective.astype(float)
        + 1.25 * rain_object
        + 1.50 * stratiform.astype(float)
        + 0.75 * nonprecipitating.astype(float)
    )
    arr = np.asarray(score.fillna(0.0).values, dtype=float)
    if not np.any(arr > 0):
        return 0.0, 0.0
    smooth = ndimage.uniform_filter(arr, size=(7, 17), mode=("nearest", "wrap"))
    iy, ix = np.unravel_index(int(np.nanargmax(smooth)), smooth.shape)
    return float(score.latitude.values[iy]), float(score.longitude.values[ix])


def add_class_band(ax, x, code_values, y0=17.5, y1=SNAPSHOT_Z_MAX_KM):
    x = np.asarray(x, dtype=float)
    codes = np.asarray(code_values, dtype=float)
    if x.size == 0 or not np.any(codes > 0):
        return
    cmap, norm = high_cloud_class_cmap_norm(alpha=0.45)
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
    rain = np.asarray(precip_values, dtype=float) >= P_PRECIP_MIN
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
        label=f"rainc + rainnc >= {P_PRECIP_MIN:g} mm day$^{{-1}}$",
    )


def classification_threshold_summary():
    return (
        f"High cloud: CF ≥ {CF_THRESH:g}; CTP ≤ {P_HIGH_HPA:g} hPa\n"
        "Deep convection: high cloud intersecting a PyFLEXTRKR Tb < 225 K core; "
        f"stratiform high cloud: other high cloud with rain ≥ {P_PRECIP_MIN / 24.0:g} mm h-1; "
        "non-precipitating high cloud: remainder"
    )


def overlay_class_tint(ax, class_code, alpha=0.20):
    cmap, norm = high_cloud_class_cmap_norm(alpha=alpha)
    values = np.asarray(class_code.values)
    masked = np.ma.masked_where(values <= 0, values)
    return ax.pcolormesh(
        class_code.longitude,
        class_code.latitude,
        masked,
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
        zorder=2,
    )


def overlay_high_cloud_tint(ax, high_cloud, alpha=0.32):
    values = np.asarray(high_cloud.values, dtype=float)
    masked = np.ma.masked_where(values < 0.5, np.ones_like(values))
    cmap = colors.ListedColormap([colors.to_rgba("#00a6ca", alpha=alpha)])
    return ax.pcolormesh(
        high_cloud.longitude,
        high_cloud.latitude,
        masked,
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
        zorder=2,
    )


def plot_precip_map(ax, precip, class_code, focus_lon, focus_lat):
    cmap, norm = precip_rate_cmap_norm()
    p = crop_longitude_window(precip, focus_lon)
    classes = crop_longitude_window(class_code, focus_lon)
    im = ax.pcolormesh(
        p.longitude,
        p.latitude,
        p.clip(min=0.0),
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    overlay_class_tint(ax, classes)
    ax.axvline(focus_lon, color="0.1", lw=1.0, ls=":")
    ax.axhline(focus_lat, color="0.1", lw=1.0, ls=":")
    ax.set_xlim(float(p.longitude.min()), float(p.longitude.max()))
    ax.set_ylim(*SNAPSHOT_LAT_BOUNDS)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("P = rainc + rainnc; high-cloud class tint")
    return im


def plot_cloud_top_height_map(
    ax,
    cloud_top_height,
    high_cloud,
    focus_lon,
    focus_lat,
):
    height = crop_longitude_window(cloud_top_height, focus_lon)
    high = crop_longitude_window(high_cloud, focus_lon)
    norm = colors.Normalize(
        vmin=CLOUD_TOP_HEIGHT_MAP_MIN_KM,
        vmax=CLOUD_TOP_HEIGHT_MAP_MAX_KM,
    )
    ax.set_facecolor("0.96")
    im = ax.pcolormesh(
        height.longitude,
        height.latitude,
        height,
        shading="auto",
        cmap="magma",
        norm=norm,
    )
    overlay_high_cloud_tint(ax, high)
    ax.axvline(focus_lon, color="0.1", lw=1.0, ls=":")
    ax.axhline(focus_lat, color="0.1", lw=1.0, ls=":")
    ax.set_xlim(float(height.longitude.min()), float(height.longitude.max()))
    ax.set_ylim(*SNAPSHOT_LAT_BOUNDS)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Cloud-top height; cyan tint = high-cloud mask")
    return im


def plot_pyflextrkr_hierarchy_map(
    ax, core, cold_shield, warm_shield, focus_lon, focus_lat
):
    core = crop_longitude_window(core, focus_lon)
    cold = crop_longitude_window(cold_shield, focus_lon)
    warm = crop_longitude_window(warm_shield, focus_lon)
    core_values = np.asarray(core, dtype=bool)
    cold_values = np.asarray(cold, dtype=bool)
    warm_values = np.asarray(warm, dtype=bool)
    cold_shell = cold_values & ~core_values
    warm_shell = warm_values & ~cold_values
    code = np.zeros(core.shape, dtype=np.int8)
    code[warm_shell] = 1
    code[cold_shell] = 2
    code[core_values] = 3
    cmap = colors.ListedColormap(["white", "#67a9cf", "#fdae61", "#b2182b"])
    norm = colors.BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap.N)
    im = ax.pcolormesh(
        core.longitude, core.latitude, np.ma.masked_where(code == 0, code),
        shading="auto", cmap=cmap, norm=norm,
    )
    ax.axvline(focus_lon, color="0.1", lw=1.0, ls=":")
    ax.axhline(focus_lat, color="0.1", lw=1.0, ls=":")
    ax.set_xlim(float(core.longitude.min()), float(core.longitude.max()))
    ax.set_ylim(*SNAPSHOT_LAT_BOUNDS)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("PyFLEXTRKR scene: warm-shield shell / cold-shield shell / cold core")
    return im


def plot_class_map(ax, class_code, focus_lon, focus_lat):
    cmap, norm = high_cloud_class_cmap_norm(alpha=0.88)
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
    ax.set_xlim(float(cls.longitude.min()), float(cls.longitude.max()))
    ax.set_ylim(*SNAPSHOT_LAT_BOUNDS)
    ax.set_xlabel("Longitude")
    ax.tick_params(labelleft=False)
    ax.set_title("High-cloud column class")
    return im


def save_validation_map_figure(
    sim_name,
    snap_time,
    precip,
    high_cloud,
    pyflextrkr_core,
    core_bearing_cold_shield,
    core_bearing_warm_shield,
    cloud_top_height,
    class_code,
    focus_lon,
    focus_lat,
):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), constrained_layout=False)
    fig.subplots_adjust(
        left=0.06,
        right=0.94,
        bottom=0.15,
        top=0.80,
        wspace=0.24,
        hspace=0.32,
    )
    precip_im = plot_precip_map(
        axes[0, 0], precip, class_code, focus_lon, focus_lat
    )
    hierarchy_im = plot_pyflextrkr_hierarchy_map(
        axes[0, 1], pyflextrkr_core, core_bearing_cold_shield,
        core_bearing_warm_shield, focus_lon, focus_lat
    )
    height_im = plot_cloud_top_height_map(
        axes[1, 0], cloud_top_height, high_cloud, focus_lon, focus_lat
    )
    plot_class_map(axes[1, 1], class_code, focus_lon, focus_lat)
    axes[0, 1].set_ylabel("Latitude")
    axes[1, 1].set_ylabel("Latitude")
    axes[0, 0].text(
        0.01,
        0.98,
        format_time_value(snap_time),
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78),
    )
    fig.colorbar(
        precip_im,
        ax=axes[0, 0],
        label="Rain (mm day$^{-1}$)",
        fraction=0.046,
        pad=0.025,
    )
    fig.colorbar(
        hierarchy_im, ax=axes[0, 1], ticks=[1, 2, 3],
        label="1 warm; 2 cold shield; 3 core",
        fraction=0.046, pad=0.025,
    )
    fig.colorbar(
        height_im,
        ax=axes[1, 0],
        label="Cloud-top height (km)",
        fraction=0.046,
        pad=0.025,
    )
    fig.legend(
        handles=high_cloud_class_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=len(HIGH_CLOUD_CLASS_SPECS),
        frameon=False,
        title="High-cloud class",
    )
    fig.suptitle(
        f"{sim_name}: high-cloud classification at {format_time_value(snap_time)}\n"
        f"{classification_threshold_summary()}",
        fontsize=9,
    )
    safe_sim = sim_name.replace(" ", "_")
    stem = (
        "cloud_classification_validation_maps"
        if len(sim_names) == 1
        else f"cloud_classification_validation_maps_{safe_sim}"
    )
    save_figure(3, fig, stem)
    plt.close(fig)


def plot_cloud_slice(
    ax,
    cf_snapshot,
    class_code,
    precip,
    focus_lat,
    focus_lon,
    orientation,
    pressure_snapshot,
):
    if orientation == "longitude":
        cf_line, actual_lat = select_nearest_latitude(cf_snapshot, focus_lat)
        class_line, _ = select_nearest_latitude(class_code, actual_lat)
        precip_line, _ = select_nearest_latitude(precip, actual_lat)
        pressure_line = select_nearest_latitude(pressure_snapshot, actual_lat)[0]
        cf_line = crop_longitude_window(cf_line, focus_lon)
        class_line = crop_longitude_window(class_line, focus_lon)
        precip_line = crop_longitude_window(precip_line, focus_lon)
        pressure_line = crop_longitude_window(pressure_line, focus_lon)
        x = np.asarray(cf_line.longitude.values, dtype=float)
        xlabel = "Longitude"
        title = f"cldfrac at latitude {actual_lat:.1f} deg"
        horizontal_dim = "longitude"
    else:
        cf_line, actual_lon = select_nearest_longitude(cf_snapshot, focus_lon)
        class_line, _ = select_nearest_longitude(class_code, actual_lon)
        precip_line, _ = select_nearest_longitude(precip, actual_lon)
        pressure_line = select_nearest_longitude(pressure_snapshot, actual_lon)[0]
        cf_line = subset_lat(cf_line, SNAPSHOT_LAT_BOUNDS)
        class_line = subset_lat(class_line, SNAPSHOT_LAT_BOUNDS)
        precip_line = subset_lat(precip_line, SNAPSHOT_LAT_BOUNDS)
        pressure_line = subset_lat(pressure_line, SNAPSHOT_LAT_BOUNDS)
        x = np.asarray(cf_line.latitude.values, dtype=float)
        xlabel = "Latitude"
        title = f"cldfrac at longitude {actual_lon:.1f} deg"
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
    pressure_values = pressure_line.sel(z=cf_line.z).transpose("z", horizontal_dim).values
    evidence_handles = []
    for level, color, label in (
        (P_HIGH_HPA, "tab:blue", f"{P_HIGH_HPA:g} hPa high-cloud boundary"),
    ):
        finite_pressure = pressure_values[np.isfinite(pressure_values)]
        if finite_pressure.size and finite_pressure.min() <= level <= finite_pressure.max():
            ax.contour(
                x, z_vals, pressure_values, levels=[level], colors=color,
                linewidths=0.9, linestyles="--",
            )
            evidence_handles.append(Line2D([0], [0], color=color, lw=1.0, ls="--", label=label))
    add_class_band(ax, x, class_line.values)
    add_rain_ticks(ax, x, precip_line.values)
    for zline, lw, ls in [(2.0, 0.55, ":"), (5.0, 0.55, ":"), (10.0, 0.55, ":")]:
        ax.axhline(zline, color="0.45", lw=lw, ls=ls, zorder=4)
    ax.set_ylim(0.0, SNAPSHOT_Z_MAX_KM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Height (km)")
    mean_pressure_profile = np.nanmean(pressure_values, axis=1)
    add_height_pressure_axis(
        ax,
        z_vals,
        mean_pressure_profile,
        primary="height",
    )
    ax.set_title(title)
    ax.grid(alpha=0.15, lw=0.4)
    if evidence_handles:
        ax.legend(handles=evidence_handles, loc="upper right", fontsize=6, frameon=True)
    return im


def snapshot_pressure(ds, cf_snapshot, snap_time):
    raw = select_time_value(ds.pressure, snap_time)
    return standardize_pressure_hpa(raw, cf_snapshot)


def plot_validation_snapshots(dss, fields_by_run, classifications):
    if not should_run_figure(3):
        return
    progress("Starting Figure 3: cloud_classification_validation_snapshots")
    if not any(classifications.get(sim_name) is not None for sim_name in sim_names):
        print("Skipping Figure 3: high-cloud classification is unavailable.")
        return

    slice_fig, slice_axes = plt.subplots(
        len(sim_names),
        2,
        figsize=(12.0, 3.6 * len(sim_names)),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    if len(sim_names) == 1:
        slice_axes = slice_axes[None, :]

    rows = []
    last_cf_im = None
    for i, sim_name in enumerate(sim_names):
        ax_lon, ax_lat = slice_axes[i]
        classif = classifications.get(sim_name)
        if classif is None:
            for ax in slice_axes[i]:
                ax.text(0.5, 0.5, "classification unavailable", transform=ax.transAxes, ha="center", va="center")
            continue
        snap_time = representative_validation_time(classif)
        if snap_time is None:
            for ax in slice_axes[i]:
                ax.text(0.5, 0.5, "no Time dimension", transform=ax.transAxes, ha="center", va="center")
            continue

        precip = subset_lat(
            fields_by_run[i]["precip"].sel(Time=snap_time),
            SNAPSHOT_LAT_BOUNDS,
        ).load()
        high = subset_lat(classif["high_cloud_mask"].sel(Time=snap_time).astype(float), SNAPSHOT_LAT_BOUNDS).load()
        pyflextrkr_core = subset_lat(classif["pyflextrkr_core_mask"].sel(Time=snap_time).astype(bool), SNAPSHOT_LAT_BOUNDS).load()
        core_bearing_cold_shield = subset_lat(classif["pyflextrkr_core_bearing_cold_shield_mask"].sel(Time=snap_time).astype(bool), SNAPSHOT_LAT_BOUNDS).load()
        core_bearing_warm_shield = subset_lat(classif["pyflextrkr_core_bearing_warm_shield_mask"].sel(Time=snap_time).astype(bool), SNAPSHOT_LAT_BOUNDS).load()
        cloud_top_height = subset_lat(
            classif["cloud_top_height"].sel(Time=snap_time),
            SNAPSHOT_LAT_BOUNDS,
        ).load()
        convective = subset_lat(classif["convective_precipitating_high_cloud_mask"].sel(Time=snap_time).astype(float), SNAPSHOT_LAT_BOUNDS).load()
        stratiform = subset_lat(classif["stratiform_precipitating_high_cloud_mask"].sel(Time=snap_time).astype(float), SNAPSHOT_LAT_BOUNDS).load()
        nonprecipitating = subset_lat(classif["nonprecipitating_high_cloud_mask"].sel(Time=snap_time).astype(float), SNAPSHOT_LAT_BOUNDS).load()
        unclassified = subset_lat(classif["unclassified_high_cloud_mask"].sel(Time=snap_time).astype(float), SNAPSHOT_LAT_BOUNDS).load()
        precip_object = (precip >= P_PRECIP_MIN).astype(float)
        class_code = class_code_from_masks(high, convective, stratiform, nonprecipitating, unclassified)
        focus_lat, focus_lon = focus_point_for_snapshot(
            precip, high, convective, stratiform, nonprecipitating
        )

        save_validation_map_figure(
            sim_name,
            snap_time,
            precip,
            high,
            pyflextrkr_core,
            core_bearing_cold_shield,
            core_bearing_warm_shield,
            cloud_top_height,
            class_code,
            focus_lon,
            focus_lat,
        )

        cf = get_cloud_fraction(dss[i])
        if cf is None:
            ax_lon.text(0.5, 0.5, "cloud fraction unavailable", transform=ax_lon.transAxes, ha="center", va="center")
            ax_lat.text(0.5, 0.5, "cloud fraction unavailable", transform=ax_lat.transAxes, ha="center", va="center")
        else:
            cf_snapshot = select_time_value(cf, snap_time)
            pressure_snapshot = snapshot_pressure(dss[i], cf_snapshot, snap_time)
            last_cf_im = plot_cloud_slice(
                ax_lon, cf_snapshot, class_code, precip, focus_lat, focus_lon,
                "longitude", pressure_snapshot,
            )
            last_cf_im = plot_cloud_slice(
                ax_lat, cf_snapshot, class_code, precip, focus_lat, focus_lon,
                "latitude", pressure_snapshot,
            )
            ax_lat.set_ylabel("")
            ax_lat.tick_params(labelleft=False)

        rows.append(
            dict(
                simulation=sim_name,
                snapshot_time=format_time_value(snap_time),
                focus_latitude=float(focus_lat),
                focus_longitude=float(focus_lon),
                high_cloud_fraction=weighted_area_time_mean(high, SNAPSHOT_LAT_BOUNDS),
                pyflextrkr_convective_core_fraction=weighted_area_time_mean(convective, SNAPSHOT_LAT_BOUNDS),
                stratiform_precipitating_fraction=weighted_area_time_mean(stratiform, SNAPSHOT_LAT_BOUNDS),
                nonprecipitating_fraction=weighted_area_time_mean(nonprecipitating, SNAPSHOT_LAT_BOUNDS),
                unclassified_fraction=weighted_area_time_mean(unclassified, SNAPSHOT_LAT_BOUNDS),
                precip_object_fraction=weighted_area_time_mean(precip_object, SNAPSHOT_LAT_BOUNDS),
                denominator="snapshot tropical grid cells",
            )
        )

    if last_cf_im is not None:
        slice_fig.colorbar(
            last_cf_im,
            ax=slice_axes.ravel().tolist(),
            label="cldfrac",
            shrink=0.9,
        )
    slice_fig.suptitle(
        f"cldfrac vertical sections through the selected map location; dashed line: {P_HIGH_HPA:g} hPa\n"
        "Top band gives the high-cloud column class; bottom ticks mark rainc + rainnc "
        f">= {P_PRECIP_MIN:g} mm day$^{{-1}}$"
    )
    save_table(pd.DataFrame(rows), "cloud_classification_validation_snapshots")
    save_figure(3, slice_fig, "cloud_classification_validation_vertical_slices")
    plt.close(slice_fig)
    progress("Done with figure: cloud_classification_validation_snapshots")



# -----------------------------------------------------------------------------
# Load data and run diagnostics
# -----------------------------------------------------------------------------


def test_time_window(ds, sim_name):
    """Select N+1 contiguous source times, yielding N precipitation rates."""
    cf = get_cloud_fraction(ds)
    if cf is None or "Time" not in cf.dims:
        raise RuntimeError(f"{sim_name}: test mode requires time-varying cloud fraction")

    full_index = ds.Time.to_index()
    start = 0
    if TEST_START is not None:
        if not isinstance(full_index, pd.DatetimeIndex):
            raise ValueError("--test-start requires a datetime Time coordinate")
        requested = pd.Timestamp(TEST_START)
        candidates = np.flatnonzero(full_index >= requested)
        if candidates.size == 0:
            raise ValueError(
                f"{sim_name}: --test-start {TEST_START!r} is after the loaded data"
            )
        start = int(candidates[0])

    source_count = TEST_TIME_STEPS + 1
    stop = start + source_count
    if stop > len(full_index):
        raise ValueError(
            f"{sim_name}: only {len(full_index) - start} source times are available at or "
            f"after {TEST_START or full_index[0]}; {source_count} are required"
        )
    selected = full_index[start:stop]

    progress(
        f"{sim_name}: test window uses {TEST_TIME_STEPS} classified times from "
        f"{format_time_value(selected[1])} to {format_time_value(selected[-1])}"
    )
    test_ds = subset_latitude(
        ds.sel(Time=selected), lat_bounds=HIGH_CLOUD_TEST_LAT_BOUNDS
    ).sel(longitude=slice(*QUICK_TEST_LON_BOUNDS))
    progress(
        f"{sim_name}: test domain is {HIGH_CLOUD_TEST_LAT_BOUNDS[0]:g} to "
        f"{HIGH_CLOUD_TEST_LAT_BOUNDS[1]:g} deg latitude and "
        f"{QUICK_TEST_LON_BOUNDS[0]:g} to {QUICK_TEST_LON_BOUNDS[1]:g} deg longitude"
    )
    return test_ds, selected


def load_datasets():
    dss = []
    progress("Loading lat-lon datasets")
    for i, d in enumerate(data_locs):
        progress(f"{sim_names[i]}: opening lat-lon dataset")
        ds = load_latlon_dataset(
            os.path.join(root_dir, d),
            latlon_pattern=LATLON_PATTERN,
            chunks=latlon_chunks,
        )
        ds = standardize_latlon_dataset(ds, latlon_vars)
        if TEST_MODE:
            ds, _selected_test_times = test_time_window(ds, sim_names[i])
        progress(f"{sim_names[i]}: lat-lon dataset ready with {len(ds.data_vars)} variables")
        dss.append(ds)
    return dss


def prepare_fields(dss):
    fields_by_run = []
    progress("Preparing precipitation and OLR fields")
    for sim_name, ds in zip(sim_names, dss):
        progress(f"{sim_name}: preparing fields")
        precip = get_precip_rate(ds)
        fields = {
            "precip": precip,
            "olr": ds["olrtoa"].sel(Time=precip.Time).rename("olr"),
        }
        progress(f"{sim_name}: fields ready ({', '.join(sorted(fields))})")
        fields_by_run.append(fields)
    return fields_by_run


def prepare_classifications(dss, fields_by_run):
    classifications = {}
    progress("Checking high-cloud classification products")
    for sim_name, ds, fields in zip(sim_names, dss, fields_by_run):
        classifications[sim_name] = get_or_build_classification(ds, fields, sim_name)
    if not FIGURES_ONLY:
        overlap = load_or_compute_derived_table(
            "high_cloud_classification_overlap_denominators",
            lambda: class_overlap_sanity(classifications),
        )
        save_table(overlap, "high_cloud_classification_overlap_denominators")
    progress("High-cloud classification stage complete")
    return classifications


def layer_mean_from_profile(profile, bounds_km=UTLS_LAYER_BOUNDS_KM):
    """Linearly integrate a profile over a geometric-height layer."""
    z_km = np.asarray(profile.z.values, dtype=float) / 1000.0
    values = np.asarray(profile.values, dtype=float)
    finite = np.isfinite(z_km) & np.isfinite(values)
    z_km = z_km[finite]
    values = values[finite]
    if z_km.size < 2:
        return float(values[0]) if values.size == 1 else np.nan
    lower, upper = map(float, bounds_km)
    if z_km.min() > lower or z_km.max() < upper:
        return np.nan
    inside = (z_km > lower) & (z_km < upper)
    sample_z = np.concatenate(([lower], z_km[inside], [upper]))
    sample_values = np.interp(sample_z, z_km, values)
    return float(np.trapz(sample_values, sample_z) / (upper - lower))


def _linear_interp_with_extrapolation(values, xp, fp):
    """One-dimensional interpolation with linear end extrapolation."""
    values = np.asarray(values, dtype=float)
    scalar = values.ndim == 0
    values = np.atleast_1d(values)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    result = np.interp(values, xp, fp)
    if xp.size > 1:
        lower = values < xp[0]
        upper = values > xp[-1]
        result[lower] = fp[0] + (
            (values[lower] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
        )
        result[upper] = fp[-1] + (
            (values[upper] - xp[-1])
            * (fp[-1] - fp[-2])
            / (xp[-1] - xp[-2])
        )
    return result[0] if scalar else result


def add_height_pressure_axis(ax, height_km, pressure_hpa, *, primary):
    """Add a model-derived height/pressure companion axis."""
    height = np.asarray(height_km, dtype=float)
    pressure = np.asarray(pressure_hpa, dtype=float)
    finite = np.isfinite(height) & np.isfinite(pressure) & (pressure > 0.0)
    height = height[finite]
    pressure = pressure[finite]
    order = np.argsort(height)
    height = height[order]
    pressure = np.minimum.accumulate(pressure[order])
    keep = np.r_[True, np.diff(height) > 0.0]
    height = height[keep]
    pressure = pressure[keep]
    if height.size < 2 or np.any(np.diff(pressure) >= 0.0):
        raise ValueError("Height-pressure profile must decrease monotonically")
    log_pressure = np.log(pressure)

    def height_to_pressure(values):
        return np.exp(
            _linear_interp_with_extrapolation(values, height, log_pressure)
        )

    def pressure_to_height(values):
        values = np.maximum(np.asarray(values, dtype=float), 1.0e-6)
        return _linear_interp_with_extrapolation(
            np.log(values),
            log_pressure[::-1],
            height[::-1],
        )

    if primary == "height":
        companion = ax.secondary_yaxis(
            "right", functions=(height_to_pressure, pressure_to_height)
        )
        companion.set_ylabel("Pressure (hPa)")
        companion.set_yscale("log")
        companion.yaxis.set_major_formatter(ticker.ScalarFormatter())
    elif primary == "pressure":
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        companion = ax.secondary_yaxis(
            "right", functions=(pressure_to_height, height_to_pressure)
        )
        companion.set_ylabel("Height (km)")
    else:
        raise ValueError(f"Unknown primary vertical coordinate {primary!r}")
    return companion


def compute_vertical_contribution_profiles(
    dss,
    classifications,
    time_batch=VERTICAL_PROFILE_TIME_BATCH,
):
    """Domain means of cloud fraction after selecting each column class."""
    profiles = {}
    rows = []
    summary_rows = []
    class_keys = [key for key, _label, _color in REGIME_SPECS]
    class_labels = {key: label for key, label, _color in REGIME_SPECS}

    progress("Computing vertical cloud profiles by column class and domain")
    for ds, sim_name in zip(dss, sim_names):
        classif = classifications.get(sim_name)
        if classif is None:
            continue
        full_cf = standardize_cloud_fraction(ds.cldfrac)
        full_pressure = standardize_pressure_hpa(ds.pressure, full_cf)
        domain_profiles = []
        for domain_key, latitude_bounds, domain_label in (
            CLOUD_STRUCTURE_DOMAIN_SPECS
        ):
            cf = select_absolute_latitude_band(
                full_cf, latitude_bounds
            ).sel(
                z=slice(
                    1000.0 * PROFILE_COMPUTE_Z_BOUNDS_KM[0],
                    1000.0 * PROFILE_COMPUTE_Z_BOUNDS_KM[1],
                )
            )
            pressure = select_absolute_latitude_band(
                full_pressure, latitude_bounds
            ).sel(z=cf.z)
            masks = [
                select_absolute_latitude_band(
                    classif[key].astype(bool), latitude_bounds
                )
                for key in class_keys
            ]
            aligned = xr.align(cf, pressure, *masks, join="inner")
            cf_aligned, pressure_aligned, mask_aligned = (
                aligned[0],
                aligned[1],
                aligned[2:],
            )
            cf_aligned = cf_aligned.transpose(
                "Time", "z", "latitude", "longitude"
            )
            pressure_aligned = pressure_aligned.transpose(
                "Time", "z", "latitude", "longitude"
            )
            mask_aligned = [
                mask.transpose("Time", "latitude", "longitude")
                for mask in mask_aligned
            ]

            ntime = cf_aligned.sizes["Time"]
            time_slices = [
                slice(start, min(start + time_batch, ntime))
                for start in range(0, ntime, time_batch)
            ]
            numerators = np.zeros(
                (len(class_keys), cf_aligned.sizes["z"]), dtype=float
            )
            denominators = np.zeros_like(numerators)
            pressure_numerator = np.zeros(
                cf_aligned.sizes["z"], dtype=float
            )
            pressure_denominator = np.zeros_like(pressure_numerator)
            latitude_weights = np.cos(
                np.deg2rad(
                    np.asarray(cf_aligned.latitude.values, dtype=float)
                )
            )[None, None, :, None]
            started = time.monotonic()
            progress(
                f"{sim_name}: computing {domain_label} vertical cloud "
                f"profiles in {len(time_slices)} blocks of at most "
                f"{time_batch} times"
            )

            for iblock, time_slice in enumerate(time_slices, start=1):
                block = xr.Dataset(
                    {
                        "cloud_fraction": cf_aligned.isel(Time=time_slice),
                        "pressure_hpa": pressure_aligned.isel(Time=time_slice),
                        **{
                            key: mask.isel(Time=time_slice)
                            for key, mask in zip(class_keys, mask_aligned)
                        },
                    }
                ).load()
                cloud_fraction = np.asarray(block["cloud_fraction"].values)
                pressure_values = np.asarray(block["pressure_hpa"].values)
                sample_weights = np.broadcast_to(
                    latitude_weights, cloud_fraction.shape
                )
                finite_pressure = (
                    np.isfinite(pressure_values) & (pressure_values > 0.0)
                )
                pressure_numerator += np.sum(
                    np.where(
                        finite_pressure,
                        pressure_values * sample_weights,
                        0.0,
                    ),
                    axis=(0, 2, 3),
                    dtype=np.float64,
                )
                pressure_denominator += np.sum(
                    np.where(finite_pressure, sample_weights, 0.0),
                    axis=(0, 2, 3),
                    dtype=np.float64,
                )

                # Samples outside a class remain zero contributions to the
                # all-column mean; missing selected cloud is excluded.
                for iclass, key in enumerate(class_keys):
                    mask = np.asarray(
                        block[key].values, dtype=bool
                    )[:, None, :, :]
                    contribution = np.where(mask, cloud_fraction, 0.0)
                    finite = np.isfinite(contribution)
                    numerators[iclass] += np.sum(
                        np.where(
                            finite,
                            contribution * sample_weights,
                            0.0,
                        ),
                        axis=(0, 2, 3),
                        dtype=np.float64,
                    )
                    denominators[iclass] += np.sum(
                        np.where(finite, sample_weights, 0.0),
                        axis=(0, 2, 3),
                        dtype=np.float64,
                    )
                    del contribution, finite, mask

                del block, cloud_fraction, pressure_values, sample_weights
                elapsed = time.monotonic() - started
                eta = elapsed / iblock * (len(time_slices) - iblock)
                progress(
                    f"{sim_name} {domain_label}: vertical-profile block "
                    f"{iblock}/{len(time_slices)} "
                    f"({100.0 * iblock / len(time_slices):.0f}%); "
                    f"elapsed {format_duration(elapsed)}, "
                    f"ETA {format_duration(eta)}"
                )
                if iblock % 10 == 0:
                    gc.collect()

            profile = xr.DataArray(
                np.divide(
                    numerators,
                    denominators,
                    out=np.full_like(numerators, np.nan),
                    where=denominators > 0.0,
                ),
                dims=("class_key", "z"),
                coords={"class_key": class_keys, "z": cf_aligned.z},
            )
            profile = profile.assign_coords(
                pressure_hpa=(
                    "z",
                    np.divide(
                        pressure_numerator,
                        pressure_denominator,
                        out=np.full_like(pressure_numerator, np.nan),
                        where=pressure_denominator > 0.0,
                    ),
                )
            ).expand_dims(latitude_domain=[domain_key])
            profile = profile.assign_coords(
                domain_label=("latitude_domain", [domain_label]),
                absolute_latitude_lower=(
                    "latitude_domain", [float(latitude_bounds[0])]
                ),
                absolute_latitude_upper=(
                    "latitude_domain", [float(latitude_bounds[1])]
                ),
            )
            profile.name = "domain_mean_cldfrac_contribution"
            profile.attrs.update(
                long_name=(
                    "domain-mean cloud fraction contributed by columns in "
                    "each high-cloud class"
                ),
                units="1",
                denominator="all grid-cell/time samples in the latitude band",
                hemispheric_sampling=(
                    "cosine-weighted pooled paired absolute-latitude band"
                ),
                data_scope="cloud-radiation-on simulations only",
            )
            domain_profiles.append(profile)

            for key in class_keys:
                class_profile = profile.sel(
                    latitude_domain=domain_key, class_key=key
                )
                for z_value, value in zip(
                    class_profile.z.values, class_profile.values
                ):
                    rows.append(
                        dict(
                            simulation=sim_name,
                            latitude_domain=domain_key,
                            domain_label=domain_label,
                            statistic="absolute_profile",
                            reference_simulation="none",
                            class_key=key,
                            cloud_class=class_labels[key],
                            height_km=float(z_value) / 1000.0,
                            domain_mean_cldfrac_contribution=float(value),
                            domain_mean_cldfrac_contribution_percent=(
                                100.0 * float(value)
                            ),
                            denominator=(
                                "all grid-cell/time samples in the "
                                "paired latitude band"
                            ),
                            data_scope="crf_on only",
                        )
                    )
                layer_mean = layer_mean_from_profile(class_profile)
                summary_rows.append(
                    dict(
                        simulation=sim_name,
                        latitude_domain=domain_key,
                        domain_label=domain_label,
                        statistic="absolute_layer_mean",
                        reference_simulation="none",
                        class_key=key,
                        cloud_class=class_labels[key],
                        layer_bottom_km=UTLS_LAYER_BOUNDS_KM[0],
                        layer_top_km=UTLS_LAYER_BOUNDS_KM[1],
                        domain_mean_cldfrac_contribution=layer_mean,
                        domain_mean_cldfrac_contribution_percent=(
                            100.0 * layer_mean
                        ),
                        denominator=(
                            "all grid-cell/time samples in the paired "
                            "latitude band and geometric layer depth"
                        ),
                        data_scope="crf_on only",
                    )
                )
        profiles[sim_name] = xr.concat(
            domain_profiles, dim="latitude_domain", join="exact"
        )

    if sim_names and sim_names[0] in profiles:
        control_name = sim_names[0]
        control = profiles[control_name]
        for sim_name in sim_names[1:]:
            if sim_name not in profiles:
                continue
            response = profiles[sim_name] - control
            for domain_key, _latitude_bounds, domain_label in (
                CLOUD_STRUCTURE_DOMAIN_SPECS
            ):
                for key in class_keys:
                    class_response = response.sel(
                        latitude_domain=domain_key, class_key=key
                    )
                    for z_value, value in zip(
                        class_response.z.values, class_response.values
                    ):
                        rows.append(
                            dict(
                                simulation=sim_name,
                                latitude_domain=domain_key,
                                domain_label=domain_label,
                                statistic="difference_from_control",
                                reference_simulation=control_name,
                                class_key=key,
                                cloud_class=class_labels[key],
                                height_km=float(z_value) / 1000.0,
                                domain_mean_cldfrac_contribution=float(value),
                                domain_mean_cldfrac_contribution_percent=(
                                    100.0 * float(value)
                                ),
                                denominator=(
                                    "all grid-cell/time samples in the "
                                    "paired latitude band"
                                ),
                                data_scope="crf_on only",
                            )
                        )
                    layer_mean = layer_mean_from_profile(class_response)
                    summary_rows.append(
                        dict(
                            simulation=sim_name,
                            latitude_domain=domain_key,
                            domain_label=domain_label,
                            statistic="difference_from_control",
                            reference_simulation=control_name,
                            class_key=key,
                            cloud_class=class_labels[key],
                            layer_bottom_km=UTLS_LAYER_BOUNDS_KM[0],
                            layer_top_km=UTLS_LAYER_BOUNDS_KM[1],
                            domain_mean_cldfrac_contribution=layer_mean,
                            domain_mean_cldfrac_contribution_percent=(
                                100.0 * layer_mean
                            ),
                            denominator=(
                                "all grid-cell/time samples in the paired "
                                "latitude band and geometric layer depth"
                            ),
                            data_scope="crf_on only",
                        )
                    )

    save_table(pd.DataFrame(rows), "high_cloud_vertical_contribution_profiles")
    save_table(pd.DataFrame(summary_rows), "utls_high_cloud_contribution_summary")
    progress("Domain-mean vertical cloud profiles ready")
    return profiles


def vertical_profiles_to_dataset(profiles):
    """Combine per-simulation profiles into one stable cache dataset."""

    if not profiles:
        raise RuntimeError("No vertical high-cloud profiles were calculated")
    combined = xr.concat(
        [
            profile.expand_dims(simulation=[sim_name])
            for sim_name, profile in profiles.items()
        ],
        dim="simulation",
        join="exact",
    )
    return combined.to_dataset(name="domain_mean_cldfrac_contribution")


def vertical_profiles_from_dataset(dataset):
    """Restore the plotting interface from the stable profile cache."""

    profile = dataset["domain_mean_cldfrac_contribution"]
    return {
        str(sim_name): profile.sel(simulation=sim_name, drop=True).load()
        for sim_name in profile.simulation.values
    }


def get_or_compute_vertical_profiles(dss, classifications):
    dataset = load_or_compute_derived_dataset(
        "high_cloud_vertical_contribution_profiles",
        lambda: vertical_profiles_to_dataset(
            compute_vertical_contribution_profiles(dss, classifications)
        ),
    )
    return vertical_profiles_from_dataset(dataset)


def plot_vertical_contribution_profiles(profiles):
    if not should_run_figure(4):
        return
    progress("Starting Figure 4: high_cloud_vertical_contribution_profiles")
    if not profiles:
        print("Skipping Figure 4: vertical profiles are unavailable.")
        return

    fig, axes = plt.subplots(
        2, len(REGIME_SPECS), figsize=(12.0, 7.2), constrained_layout=True,
        sharey=True, squeeze=False,
    )
    control_name = sim_names[0]
    for j, (key, label, class_color) in enumerate(REGIME_SPECS):
        ax_absolute = axes[0, j]
        ax_response = axes[1, j]
        for sim_name, color in zip(sim_names, cols):
            if sim_name not in profiles:
                continue
            profile = profiles[sim_name].sel(
                latitude_domain="tropical", class_key=key
            )
            height_km = np.asarray(profile.z.values, dtype=float) / 1000.0
            ax_absolute.plot(
                100.0 * profile.values, height_km,
                color=color, lw=1.8, label=sim_name,
            )
            if sim_name != control_name and control_name in profiles:
                response = profile - profiles[control_name].sel(
                    latitude_domain="tropical", class_key=key
                )
                ax_response.plot(
                    100.0 * response.values, height_km,
                    color=color, lw=1.8, label=f"{sim_name} - {control_name}",
                )
        for ax in (ax_absolute, ax_response):
            ax.axhspan(
                UTLS_LAYER_BOUNDS_KM[0], UTLS_LAYER_BOUNDS_KM[1],
                color="0.7", alpha=0.16, zorder=0,
            )
            ax.set_ylim(*PROFILE_Z_BOUNDS_KM)
            ax.grid(alpha=0.2)
            add_height_pressure_axis(
                ax,
                np.asarray(profile.z.values, dtype=float) / 1000.0,
                np.asarray(profile.pressure_hpa.values, dtype=float),
                primary="height",
            )
        ax_absolute.set_title(label, color=class_color, fontsize=10)
        ax_response.axvline(0.0, color="0.45", lw=0.8)
        if len(sim_names) == 1:
            ax_response.text(
                0.5, 0.5, "Single-simulation test:\nno control difference",
                transform=ax_response.transAxes, ha="center", va="center", fontsize=8,
            )

    axes[0, 0].set_ylabel("Height (km)")
    axes[1, 0].set_ylabel("Height (km)")
    for ax in axes[0, :]:
        ax.set_xlabel("Tropical-mean cloud fraction (%)")
    for ax in axes[1, :]:
        ax.set_xlabel("Nudged − control (percentage points)")
    axes[0, -1].legend(frameon=False, fontsize=8)
    if len(sim_names) > 1:
        axes[1, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Vertical cloud-fraction contribution by high-cloud class\n"
        "Class masks are zero outside identified columns; grey band marks the "
        "14–18 km UTLS layer",
        fontsize=11,
    )
    save_figure(4, fig, "high_cloud_vertical_contribution_profiles")
    plt.close(fig)
    progress("Done with figure: high_cloud_vertical_contribution_profiles")


# -----------------------------------------------------------------------------
# Figure 5. Spatial response of each high-cloud class
# -----------------------------------------------------------------------------


def time_mean_class_occurrence_maps(classifications):
    """Load exact time-mean occurrence maps from the shared class products."""
    maps = {}
    zonal_rows = []
    control_name = sim_names[0]
    for sim_name in sim_names:
        classif = classifications.get(sim_name)
        if classif is None:
            continue
        occurrence = xr.Dataset(
            {
                key: subset_lat(classif[key].astype(float), TROPICAL_LAT_BOUNDS)
                for key, _label, _color in SPATIAL_CLASS_SPECS
            }
        ).mean("Time", skipna=True)
        occurrence = xr.Dataset(
            {
                key: hemispheric_fold(occurrence[key], parity="even")
                for key, _label, _color in SPATIAL_CLASS_SPECS
            }
        ).load()
        occurrence.attrs["hemispheric_fold"] = (
            "even paired-hemisphere average before plotting"
        )
        maps[sim_name] = occurrence
        for key, label, _color in SPATIAL_CLASS_SPECS:
            zonal = occurrence[key].mean("longitude", skipna=True)
            control_zonal = (
                maps[control_name][key].mean("longitude", skipna=True)
                if control_name in maps else None
            )
            for latitude, value in zip(zonal.latitude.values, zonal.values):
                control_value = (
                    float(control_zonal.sel(latitude=latitude).values)
                    if control_zonal is not None else np.nan
                )
                zonal_rows.append(
                    dict(
                        simulation=sim_name,
                        class_key=key,
                        cloud_class=label,
                        latitude=float(latitude),
                        occurrence_percent=100.0 * float(value),
                        difference_from_control_percentage_points=(
                            100.0 * (float(value) - control_value)
                            if np.isfinite(control_value) else np.nan
                        ),
                        temporal_filter="none",
                        statistic="time-mean zonal-mean grid-column occurrence",
                    )
                )
    save_table(pd.DataFrame(zonal_rows), "high_cloud_class_zonal_response")
    return maps


def occurrence_maps_to_dataset(maps):
    if not maps:
        raise RuntimeError("No high-cloud occurrence maps were calculated")
    return xr.concat(
        [
            occurrence.expand_dims(simulation=[sim_name])
            for sim_name, occurrence in maps.items()
        ],
        dim="simulation",
        join="exact",
    )


def occurrence_maps_from_dataset(dataset):
    return {
        str(sim_name): dataset.sel(simulation=sim_name, drop=True).load()
        for sim_name in dataset.simulation.values
    }


def get_or_compute_occurrence_maps(classifications):
    dataset = load_or_compute_derived_dataset(
        "high_cloud_class_occurrence_maps",
        lambda: occurrence_maps_to_dataset(
            time_mean_class_occurrence_maps(classifications)
        ),
    )
    return occurrence_maps_from_dataset(dataset)


def control_occurrence_contour_levels(class_key, values_percent):
    """Return readable, class-specific control-occurrence contour levels."""
    preferred = {
        "high_cloud_mask": (20.0, 40.0, 60.0, 80.0),
        "convective_precipitating_high_cloud_mask": (1.0, 3.0, 5.0, 10.0),
        "stratiform_precipitating_high_cloud_mask": (5.0, 10.0, 20.0, 30.0),
        "nonprecipitating_high_cloud_mask": (20.0, 40.0, 60.0, 80.0),
    }[class_key]
    finite = np.asarray(values_percent, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return []
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    return [level for level in preferred if lo < level < hi]


def shared_zonal_mean_limits(fields_percent, difference=False):
    """Return one zonal-mean x range for comparable simulation panels."""

    finite_values = []
    for field in fields_percent:
        zonal = field.mean("longitude", skipna=True)
        values = np.asarray(zonal.values, dtype=float)
        finite_values.append(values[np.isfinite(values)])
    finite_values = [values for values in finite_values if values.size]
    pooled = (
        np.concatenate(finite_values)
        if finite_values
        else np.asarray([], dtype=float)
    )
    if difference:
        limit = max(
            0.02,
            1.08 * float(np.nanmax(np.abs(pooled)))
            if pooled.size else 0.02,
        )
        return (-limit, limit)
    upper = max(
        0.1,
        1.08 * float(np.nanmax(pooled)) if pooled.size else 0.1,
    )
    return (0.0, upper)


def plot_zonal_mean_axis(
    ax,
    field_percent,
    difference=False,
    x_limits=None,
):
    """Plot a longitude mean in a subplot aligned with its map."""
    zonal = field_percent.mean("longitude", skipna=True)
    values = np.asarray(zonal.values, dtype=float)
    ax.plot(values, zonal.latitude.values, color="0.10", lw=1.25)
    if x_limits is None:
        x_limits = shared_zonal_mean_limits(
            [field_percent],
            difference=difference,
        )
    ax.set_xlim(*x_limits)
    if difference:
        ax.axvline(0.0, color="0.55", lw=0.6)
    ax.set_ylim(float(zonal.latitude.min()), float(zonal.latitude.max()))
    ax.set_xlabel(
        "Change (pp)" if difference else "Occurrence (%)",
        fontsize=8,
        labelpad=1,
    )
    ax.tick_params(
        axis="x", labelsize=7, length=2, pad=1, labelrotation=25
    )
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(axis="x", alpha=0.15)
    return ax


def plot_class_spatial_response(classifications, occurrence_maps=None):
    """Plot unfiltered spatial occurrence or nudging-minus-control response."""
    if not should_run_figure(5):
        return
    progress("Starting Figure 5: high_cloud_class_spatial_response")
    if occurrence_maps is None:
        occurrence_maps = time_mean_class_occurrence_maps(classifications)
    if not occurrence_maps:
        print("Skipping Figure 5: high-cloud occurrence maps are unavailable.")
        return

    control_name = sim_names[0]
    control = occurrence_maps[control_name]
    class_specs = SPATIAL_CLASS_SPECS
    contrast_names = [name for name in sim_names[1:] if name in occurrence_maps]

    if not contrast_names:
        fig = plt.figure(figsize=(12.0, 4.0), constrained_layout=True)
        grid = fig.add_gridspec(
            1,
            2 * len(class_specs) + 1,
            width_ratios=[3.0, 1.0] * len(class_specs) + [0.18],
        )
        colorbar_ax = fig.add_subplot(grid[0, -1])
        values_by_class = {
            key: 100.0 * control[key] for key, _label, _color in class_specs
        }
        pooled = np.concatenate(
            [
                np.ravel(np.asarray(values.values, dtype=float))
                for values in values_by_class.values()
            ]
        )
        finite = pooled[np.isfinite(pooled)]
        vmax = max(
            1.0,
            float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0,
        )
        image = None
        for column, (key, label, class_color) in enumerate(class_specs):
            ax = fig.add_subplot(grid[0, 2 * column])
            mean_ax = fig.add_subplot(
                grid[0, 2 * column + 1], sharey=ax
            )
            values = values_by_class[key]
            image = ax.pcolormesh(
                values.longitude, values.latitude, values,
                shading="auto", cmap="Blues", vmin=0.0, vmax=vmax,
            )
            plot_zonal_mean_axis(mean_ax, values, difference=False)
            ax.set_title(label, fontsize=9, color=class_color)
            ax.set_xlabel("Longitude")
            if column == 0:
                ax.set_ylabel("Latitude")
            else:
                ax.tick_params(labelleft=False)
        fig.colorbar(
            image,
            cax=colorbar_ax,
            label="Time occurrence (%)",
        )
        fig.suptitle(
            f"{control_name}: high-cloud occurrence"
        )
    else:
        fig = plt.figure(
            figsize=(12.0, 3.4 * len(contrast_names)),
            constrained_layout=True,
        )
        grid = fig.add_gridspec(
            len(contrast_names),
            2 * len(class_specs) + 1,
            width_ratios=[3.0, 1.0] * len(class_specs) + [0.18],
        )
        colorbar_ax = fig.add_subplot(grid[:, -1])
        differences = {
            (sim_name, key): 100.0 * (occurrence_maps[sim_name][key] - control[key])
            for sim_name in contrast_names
            for key, _label, _color in class_specs
        }
        pooled = np.concatenate(
            [
                np.ravel(np.asarray(field.values, dtype=float))
                for field in differences.values()
            ]
        )
        finite = np.abs(pooled[np.isfinite(pooled)])
        limit = max(
            0.05,
            float(np.nanpercentile(finite, 99.0)) if finite.size else 0.05,
        )
        difference_levels = symmetric_levels(limit, ndiv=11)
        difference_colormap = difference_cmap(difference_levels)
        norm = colors.BoundaryNorm(
            difference_levels, ncolors=difference_colormap.N
        )
        zonal_limits = {
            key: shared_zonal_mean_limits(
                [
                    differences[(sim_name, key)]
                    for sim_name in contrast_names
                ],
                difference=True,
            )
            for key, _label, _color in class_specs
        }
        image = None
        for column, (key, label, class_color) in enumerate(class_specs):
            control_percent = 100.0 * control[key]
            contour_levels = control_occurrence_contour_levels(
                key, control_percent.values
            )
            for row, sim_name in enumerate(contrast_names):
                ax = fig.add_subplot(grid[row, 2 * column])
                mean_ax = fig.add_subplot(
                    grid[row, 2 * column + 1], sharey=ax
                )
                difference = differences[(sim_name, key)]
                image = ax.pcolormesh(
                    difference.longitude, difference.latitude, difference,
                    shading="auto", cmap=difference_colormap, norm=norm,
                )
                plot_zonal_mean_axis(
                    mean_ax,
                    difference,
                    difference=True,
                    x_limits=zonal_limits[key],
                )
                if contour_levels:
                    ax.contour(
                        control_percent.longitude,
                        control_percent.latitude,
                        control_percent,
                        levels=contour_levels,
                        colors="0.25",
                        linewidths=0.6,
                    )
                    if row == 0:
                        ax.text(
                            0.01, 0.02,
                            "control contours: "
                            + ", ".join(f"{level:g}" for level in contour_levels)
                            + "%",
                            transform=ax.transAxes,
                            ha="left", va="bottom", fontsize=6.5, color="0.2",
                            bbox=dict(
                                facecolor="white", edgecolor="none", alpha=0.72,
                                pad=1.5,
                            ),
                        )
                if row == 0:
                    ax.set_title(label, fontsize=9, color=class_color)
                if column == 0:
                    ax.set_ylabel(f"{sim_name} − control\nLatitude")
                else:
                    ax.tick_params(labelleft=False)
                if row == len(contrast_names) - 1:
                    ax.set_xlabel("Longitude")
        fig.colorbar(
            image,
            cax=colorbar_ax,
            label="Occurrence change (percentage points)",
        )
        fig.suptitle(
            "High-cloud occurrence response\n"
            "Shading: nudged − control; contours: control; right panels: zonal mean"
        )

    save_figure(5, fig, "high_cloud_class_spatial_response")
    plt.close(fig)
    progress("Done with figure: high_cloud_class_spatial_response")


# -----------------------------------------------------------------------------
# Figure 6. Cloud depth and OLR response at matched surface rain rate
# -----------------------------------------------------------------------------


def rain_bin_centres():
    centres = []
    for lower, upper in zip(RAIN_RATE_BINS_MM_H[:-1], RAIN_RATE_BINS_MM_H[1:]):
        centres.append(
            0.5 * (lower + upper) if np.isfinite(upper) else 1.35 * lower
        )
    return np.asarray(centres, dtype=float)


def rain_conditioned_block_sums(classifications, fields_by_run):
    """Accumulate exact area-weighted sums in 10-day time blocks."""
    rows = []
    fields_by_name = dict(zip(sim_names, fields_by_run))
    class_keys = [key for key, _label, _color in RAIN_CONDITIONED_CLASS_SPECS]
    for sim_name in sim_names:
        if sim_name not in classifications:
            continue
        classif = classifications[sim_name]
        fields = fields_by_name[sim_name]
        aligned = xr.align(
            subset_lat(classif["cloud_top_pressure"], TROPICAL_LAT_BOUNDS),
            subset_lat(fields["olr"], TROPICAL_LAT_BOUNDS),
            subset_lat(fields["precip"] / 24.0, TROPICAL_LAT_BOUNDS),
            *[
                subset_lat(classif[key].astype(bool), TROPICAL_LAT_BOUNDS)
                for key in class_keys
            ],
            join="inner",
        )
        cloud_top_pressure, olr, rain_rate = aligned[:3]
        masks = dict(zip(class_keys, aligned[3:]))
        data = xr.Dataset(
            {
                "cloud_top_pressure": cloud_top_pressure,
                "olr": olr,
                "rain_rate": rain_rate,
                **masks,
            }
        )
        slices = time_block_slices(
            rain_rate,
            block_days=RAIN_CONDITIONED_TIME_BATCH_DAYS,
        )
        progress(
            f"{sim_name}: rain-matched high-cloud properties in {len(slices)} "
            f"blocks of about {RAIN_CONDITIONED_TIME_BATCH_DAYS:g} days"
        )
        latitude_weights = np.cos(
            np.deg2rad(np.asarray(rain_rate.latitude.values, dtype=float))
        )[None, :, None]
        for block_index, time_slice in enumerate(slices):
            block = data.isel(Time=time_slice).load()
            rain_values = np.asarray(block["rain_rate"].values, dtype=float)
            weights = np.broadcast_to(latitude_weights, rain_values.shape)
            field_values = {
                "cloud_top_pressure": np.asarray(
                    block["cloud_top_pressure"].values, dtype=float
                ),
                "olr": np.asarray(block["olr"].values, dtype=float),
            }
            block_times = block.Time.values
            for key, label, _color in RAIN_CONDITIONED_CLASS_SPECS:
                class_mask = np.asarray(block[key].values, dtype=bool)
                for bin_index, (lower, upper) in enumerate(
                    zip(RAIN_RATE_BINS_MM_H[:-1], RAIN_RATE_BINS_MM_H[1:])
                ):
                    in_bin = (
                        np.isfinite(rain_values)
                        & (rain_values >= lower)
                        & (
                            (rain_values < upper)
                            if np.isfinite(upper) else True
                        )
                    )
                    for property_key, values in field_values.items():
                        valid = class_mask & in_bin & np.isfinite(values)
                        valid_weights = weights[valid]
                        rows.append(
                            dict(
                                simulation=sim_name,
                                class_key=key,
                                cloud_class=label,
                                property_key=property_key,
                                block_index=int(block_index),
                                block_start=format_time_value(block_times[0]),
                                block_end=format_time_value(block_times[-1]),
                                rain_bin_index=int(bin_index),
                                rain_bin_lower_mm_h=float(lower),
                                rain_bin_upper_mm_h=(
                                    float(upper) if np.isfinite(upper) else np.inf
                                ),
                                rain_bin_centre_mm_h=float(
                                    rain_bin_centres()[bin_index]
                                ),
                                weighted_sum=float(
                                    np.sum(values[valid] * valid_weights)
                                ),
                                area_time_weight=float(np.sum(valid_weights)),
                                grid_cell_time_count=int(np.count_nonzero(valid)),
                            )
                        )
            if (
                block_index == 0
                or (block_index + 1) % 5 == 0
                or block_index + 1 == len(slices)
            ):
                progress(
                    f"{sim_name}: rain-matched block {block_index + 1}/{len(slices)}"
                )
            del block, field_values
        gc.collect()
    table = pd.DataFrame(rows)
    save_table(table, "rain_matched_high_cloud_block_sums")
    return table


def bootstrap_weighted_ratio(block_rows, rng):
    sums = block_rows.weighted_sum.to_numpy(dtype=float)
    weights = block_rows.area_time_weight.to_numpy(dtype=float)
    finite = np.isfinite(sums) & np.isfinite(weights) & (weights > 0.0)
    sums = sums[finite]
    weights = weights[finite]
    if sums.size == 0:
        return np.nan, np.nan, np.nan, 0
    value = float(np.sum(sums) / np.sum(weights))
    boot = np.empty(N_BOOTSTRAP, dtype=float)
    for iboot in range(N_BOOTSTRAP):
        index = rng.integers(0, sums.size, sums.size)
        boot[iboot] = np.sum(sums[index]) / np.sum(weights[index])
    ci_low, ci_high = np.nanpercentile(boot, CI_PERCENTILES)
    return value, float(ci_low), float(ci_high), int(sums.size)


def bootstrap_paired_weighted_ratio_difference(sim_rows, control_rows, rng):
    paired = sim_rows.merge(
        control_rows,
        on="block_index",
        suffixes=("_simulation", "_control"),
    )
    valid = (
        np.isfinite(paired.weighted_sum_simulation)
        & np.isfinite(paired.weighted_sum_control)
        & np.isfinite(paired.area_time_weight_simulation)
        & np.isfinite(paired.area_time_weight_control)
        & (paired.area_time_weight_simulation > 0.0)
        & (paired.area_time_weight_control > 0.0)
    )
    paired = paired.loc[valid]
    if len(paired) == 0:
        return np.nan, np.nan, np.nan, 0
    sim_sum = paired.weighted_sum_simulation.to_numpy(dtype=float)
    sim_weight = paired.area_time_weight_simulation.to_numpy(dtype=float)
    control_sum = paired.weighted_sum_control.to_numpy(dtype=float)
    control_weight = paired.area_time_weight_control.to_numpy(dtype=float)
    value = float(
        np.sum(sim_sum) / np.sum(sim_weight)
        - np.sum(control_sum) / np.sum(control_weight)
    )
    boot = np.empty(N_BOOTSTRAP, dtype=float)
    for iboot in range(N_BOOTSTRAP):
        index = rng.integers(0, len(paired), len(paired))
        boot[iboot] = (
            np.sum(sim_sum[index]) / np.sum(sim_weight[index])
            - np.sum(control_sum[index]) / np.sum(control_weight[index])
        )
    ci_low, ci_high = np.nanpercentile(boot, CI_PERCENTILES)
    return value, float(ci_low), float(ci_high), int(len(paired))


def aggregate_rain_conditioned_response(block_table):
    rng = np.random.default_rng(BOOTSTRAP_SEED + 61)
    rows = []
    group_keys = ["class_key", "cloud_class", "property_key", "rain_bin_index"]
    for sim_name in sim_names:
        simulation_table = block_table[block_table.simulation == sim_name]
        for key_values, group in simulation_table.groupby(group_keys, sort=False):
            value, ci_low, ci_high, n_blocks = bootstrap_weighted_ratio(group, rng)
            first = group.iloc[0]
            rows.append(
                dict(
                    statistic="absolute",
                    simulation=sim_name,
                    contrast="",
                    class_key=key_values[0],
                    cloud_class=key_values[1],
                    property_key=key_values[2],
                    rain_bin_index=int(key_values[3]),
                    rain_bin_lower_mm_h=first.rain_bin_lower_mm_h,
                    rain_bin_upper_mm_h=first.rain_bin_upper_mm_h,
                    rain_bin_centre_mm_h=first.rain_bin_centre_mm_h,
                    value=value,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    n_blocks=n_blocks,
                    sample_count=int(group.grid_cell_time_count.sum()),
                    control_sample_count=np.nan,
                )
            )

    control_name = sim_names[0]
    for sim_name in sim_names[1:]:
        for key_values, sim_group in block_table[
            block_table.simulation == sim_name
        ].groupby(group_keys, sort=False):
            control_group = block_table[
                (block_table.simulation == control_name)
                & (block_table.class_key == key_values[0])
                & (block_table.property_key == key_values[2])
                & (block_table.rain_bin_index == key_values[3])
            ]
            value, ci_low, ci_high, n_blocks = (
                bootstrap_paired_weighted_ratio_difference(
                    sim_group, control_group, rng
                )
            )
            first = sim_group.iloc[0]
            rows.append(
                dict(
                    statistic="difference_from_control",
                    simulation=sim_name,
                    contrast=f"{sim_name} - {control_name}",
                    class_key=key_values[0],
                    cloud_class=key_values[1],
                    property_key=key_values[2],
                    rain_bin_index=int(key_values[3]),
                    rain_bin_lower_mm_h=first.rain_bin_lower_mm_h,
                    rain_bin_upper_mm_h=first.rain_bin_upper_mm_h,
                    rain_bin_centre_mm_h=first.rain_bin_centre_mm_h,
                    value=value,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    n_blocks=n_blocks,
                    sample_count=int(sim_group.grid_cell_time_count.sum()),
                    control_sample_count=int(
                        control_group.grid_cell_time_count.sum()
                    ),
                )
            )
    out = pd.DataFrame(rows)
    out["temporal_filter"] = "none"
    out["conditioning"] = "surface rain-rate bin"
    out["interval"] = "95% 10-day block-bootstrap consistency interval"
    out["block_support"] = np.where(
        out.statistic == "difference_from_control",
        "paired 10-day blocks containing class samples in both simulations",
        "10-day blocks containing class samples",
    )
    save_table(out, "rain_matched_high_cloud_response")
    return out


def plot_rain_matched_high_cloud_response(
    classifications,
    fields_by_run,
    response=None,
):
    if not should_run_figure(6):
        return
    progress("Starting Figure 6: rain_matched_high_cloud_structure_response")
    if response is None:
        block_table = rain_conditioned_block_sums(
            classifications, fields_by_run
        )
        response = aggregate_rain_conditioned_response(block_table)
    if len(response) == 0:
        print("Skipping Figure 6: rain-matched high-cloud statistics are unavailable.")
        return

    has_contrasts = len(sim_names) > 1
    statistic = "difference_from_control" if has_contrasts else "absolute"
    plotted = response[response.statistic == statistic]
    property_specs = (
        (
            "cloud_top_pressure",
            "Cloud-top pressure response (hPa)\nnegative = higher cloud top"
            if has_contrasts else "Cloud-top pressure (hPa)",
        ),
        (
            "olr",
            "OLR response (W m$^{-2}$)\nnegative = less outgoing longwave radiation"
            if has_contrasts else "OLR (W m$^{-2}$)",
        ),
    )
    fig, axes = plt.subplots(
        2, len(RAIN_CONDITIONED_CLASS_SPECS),
        figsize=(12.0, 6.8), constrained_layout=True,
        sharex=True, squeeze=False,
    )
    line_names = sim_names[1:] if has_contrasts else sim_names
    line_colors = cols[1:] if has_contrasts else cols
    required_blocks = 1 if TEST_MODE else MIN_RAIN_CONDITIONED_BLOCKS
    for column, (class_key, class_label, class_color) in enumerate(
        RAIN_CONDITIONED_CLASS_SPECS
    ):
        axes[0, column].set_title(
            class_label, fontsize=10, color=class_color
        )
        for row, (property_key, ylabel) in enumerate(property_specs):
            ax = axes[row, column]
            for line_name, color in (
                zip(line_names, line_colors)
            ):
                values = plotted[
                    (plotted.class_key == class_key)
                    & (plotted.property_key == property_key)
                    & (plotted.simulation == line_name)
                    & (plotted.n_blocks >= required_blocks)
                ].sort_values("rain_bin_index")
                if len(values) == 0:
                    continue
                x = values.rain_bin_centre_mm_h.to_numpy(dtype=float)
                y = values.value.to_numpy(dtype=float)
                lo = values.ci_low.to_numpy(dtype=float)
                hi = values.ci_high.to_numpy(dtype=float)
                label = (
                    f"{line_name} - {sim_names[0]}" if has_contrasts else line_name
                )
                ax.plot(x, y, color=color, lw=1.7, marker="o", ms=3, label=label)
                ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0)
            if has_contrasts:
                ax.axhline(0.0, color="0.45", lw=0.8)
            ax.axvline(
                P_PRECIP_MIN / 24.0, color="0.35", lw=0.9, ls="--"
            )
            ax.set_xscale("symlog", linthresh=0.05, linscale=0.6, base=10)
            ax.set_xlim(0.0, 11.0)
            ax.set_xticks([0.025, 0.1, 0.5, 1.0, 5.0, 10.0])
            ax.set_xticklabels(["0.025", "0.1", "0.5", "1", "5", "8+"])
            ax.grid(alpha=0.2)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Surface rain rate (mm h$^{-1}$)")
        axes[0, column].invert_yaxis()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for ax in axes.ravel():
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    if handles and has_contrasts:
        axes[0, 0].legend(handles, labels, loc="best", frameon=False, fontsize=8)
    title_prefix = "Nudging response of" if has_contrasts else "Control values of"
    fig.suptitle(
        f"{title_prefix} cloud depth and OLR by surface rain rate\n"
        f"Dashed line: {P_PRECIP_MIN / 24.0:g} mm h$^{{-1}}$ threshold "
        "for precipitation features; shading: 95% block interval",
        fontsize=11,
    )
    save_figure(6, fig, "rain_matched_high_cloud_structure_response")
    plt.close(fig)
    progress("Done with figure: rain_matched_high_cloud_structure_response")


def compute_cloud_structure_product(fields_by_run, classifications):
    """Compute the reusable cloud-top and OLR distribution product."""

    ctp_bins = np.arange(
        0.0,
        P_HIGH_HPA + CLOUD_TOP_PRESSURE_BIN_WIDTH_HPA,
        CLOUD_TOP_PRESSURE_BIN_WIDTH_HPA,
    )
    olr_bins = np.arange(
        80.0,
        330.0 + OLR_BIN_WIDTH_W_M2,
        OLR_BIN_WIDTH_W_M2,
    )
    class_keys = [key for key, _label, _color in REGIME_SPECS]
    class_labels = [label for _key, label, _color in REGIME_SPECS]
    domain_keys = [
        key for key, _bounds, _label in CLOUD_STRUCTURE_DOMAIN_SPECS
    ]
    domain_labels = [
        label for _key, _bounds, label in CLOUD_STRUCTURE_DOMAIN_SPECS
    ]
    shape = (len(sim_names), len(domain_keys), len(class_keys))
    scalar = {
        name: np.full(shape, np.nan)
        for name in (
            "occurrence_fraction",
            "ttl_reaching_fraction",
            "mean_cloud_top_pressure_hpa",
            "mean_olr_w_m2",
        )
    }
    ctp_histogram = np.full(shape + (len(ctp_bins) - 1,), np.nan)
    olr_histogram = np.full(shape + (len(olr_bins) - 1,), np.nan)
    fields_by_name = dict(zip(sim_names, fields_by_run))
    for isim, sim_name in enumerate(sim_names):
        for idomain, (
            _domain_key,
            latitude_bounds,
            domain_label,
        ) in enumerate(CLOUD_STRUCTURE_DOMAIN_SPECS):
            statistics = full_class_structure_statistics(
                fields_by_name[sim_name],
                classifications[sim_name],
                sim_name,
                absolute_latitude_bounds=latitude_bounds,
                domain_label=domain_label,
                ctp_bins=ctp_bins,
                olr_bins=olr_bins,
            )
            for iclass, key in enumerate(class_keys):
                values = statistics[key]
                index = (isim, idomain, iclass)
                scalar["occurrence_fraction"][index] = values[
                    "occurrence_fraction"
                ]
                scalar["ttl_reaching_fraction"][index] = values[
                    "ttl_fraction"
                ]
                scalar["mean_cloud_top_pressure_hpa"][index] = values[
                    "mean_cloud_top_pressure"
                ]
                scalar["mean_olr_w_m2"][index] = values["mean_olr"]
                ctp_histogram[index] = values["ctp_histogram"]
                olr_histogram[index] = values["olr_histogram"]
            del statistics
            gc.collect()
    return xr.Dataset(
        {
            **{
                name: (
                    ("simulation", "latitude_domain", "class_key"),
                    values,
                )
                for name, values in scalar.items()
            },
            "cloud_top_pressure_histogram": (
                (
                    "simulation",
                    "latitude_domain",
                    "class_key",
                    "cloud_top_pressure_bin",
                ),
                ctp_histogram,
            ),
            "olr_histogram": (
                (
                    "simulation",
                    "latitude_domain",
                    "class_key",
                    "olr_bin",
                ),
                olr_histogram,
            ),
        },
        coords={
            "simulation": sim_names,
            "latitude_domain": domain_keys,
            "domain_label": ("latitude_domain", domain_labels),
            "absolute_latitude_lower": (
                "latitude_domain",
                [
                    bounds[0]
                    for _key, bounds, _label
                    in CLOUD_STRUCTURE_DOMAIN_SPECS
                ],
            ),
            "absolute_latitude_upper": (
                "latitude_domain",
                [
                    bounds[1]
                    for _key, bounds, _label
                    in CLOUD_STRUCTURE_DOMAIN_SPECS
                ],
            ),
            "class_key": class_keys,
            "cloud_class": ("class_key", class_labels),
            "cloud_top_pressure_bin": 0.5 * (ctp_bins[:-1] + ctp_bins[1:]),
            "olr_bin": 0.5 * (olr_bins[:-1] + olr_bins[1:]),
        },
        attrs={
            "latitude_domains": str(CLOUD_STRUCTURE_DOMAIN_SPECS),
            "hemispheric_sampling": (
                "cosine-weighted pooled paired absolute-latitude bands; "
                "individual cloud-top and OLR samples are not averaged "
                "between hemispheres"
            ),
            "cloud_top_pressure_bin_width_hpa": (
                CLOUD_TOP_PRESSURE_BIN_WIDTH_HPA
            ),
            "olr_bin_width_w_m2": OLR_BIN_WIDTH_W_M2,
            "occurrence_denominator": (
                "all area-time samples in each paired latitude band"
            ),
            "response_filter": "none",
        },
    )


def cloud_structure_metrics_table(structure):
    rows = []
    for sim_name in structure.simulation.values:
        for domain_key in structure.latitude_domain.values:
            domain_label = str(
                structure.domain_label.sel(
                    latitude_domain=domain_key
                ).item()
            )
            for key, label, _color in REGIME_SPECS:
                sample = structure.sel(
                    simulation=sim_name,
                    latitude_domain=domain_key,
                    class_key=key,
                )
                rows.append(
                    dict(
                        simulation=str(sim_name),
                        latitude_domain=str(domain_key),
                        domain_label=domain_label,
                        class_key=key,
                        cloud_class=label,
                        occurrence_fraction=float(
                            sample.occurrence_fraction
                        ),
                        ttl_reaching_fraction_ctp_le_180_hpa=float(
                            sample.ttl_reaching_fraction
                        ),
                        class_conditioned_olr_w_m2=float(
                            sample.mean_olr_w_m2
                        ),
                        mean_cloud_top_pressure_hpa=float(
                            sample.mean_cloud_top_pressure_hpa
                        ),
                        response_filter="none",
                        occurrence_denominator=(
                            "all area-time samples in paired latitude band"
                        ),
                        ttl_share_denominator=(
                            "area-time occurrence of this cloud class"
                        ),
                        olr_condition=(
                            "unfiltered grid columns in which this cloud "
                            "class occurs"
                        ),
                        temporal_spatial_sampling=(
                            f"full {domain_label} analysis period; "
                            "cosine-weighted pooled hemispheres; time-blocked"
                        ),
                    )
                )
    return pd.DataFrame(rows)


def coarsen_histogram_for_plot(
    bin_centres,
    histogram,
    target_bin_width,
):
    """Combine adjacent regular histogram bins for a clearer plotted PDF."""

    centres = np.asarray(bin_centres, dtype=float)
    values = np.asarray(histogram, dtype=float)
    if centres.ndim != 1 or values.shape[-1] != centres.size:
        raise ValueError(
            "Histogram values must end with the one-dimensional bin axis"
        )
    if centres.size < 2:
        return centres, values
    spacing = np.diff(centres)
    source_width = float(np.median(spacing))
    if not np.allclose(spacing, source_width):
        raise ValueError("Histogram bin centres must be regularly spaced")
    factor = int(round(float(target_bin_width) / source_width))
    if factor < 1 or not np.isclose(
        factor * source_width,
        target_bin_width,
    ):
        raise ValueError(
            "Display bin width must be a positive integer multiple of the "
            "cached bin width"
        )
    if centres.size % factor:
        raise ValueError(
            "Cached histogram bins cannot be evenly combined at the "
            "requested display width"
        )
    if factor == 1:
        return centres, values
    output_shape = values.shape[:-1] + (centres.size // factor, factor)
    return (
        centres.reshape(-1, factor).mean(axis=1),
        values.reshape(output_shape).sum(axis=-1),
    )


def regular_histogram_bin_edges(bin_centres):
    """Return exact edges for regularly spaced histogram centres."""

    centres = np.asarray(bin_centres, dtype=float)
    if centres.ndim != 1 or centres.size < 2:
        raise ValueError("At least two one-dimensional bin centres are required")
    spacing = np.diff(centres)
    width = float(np.median(spacing))
    if not np.allclose(spacing, width):
        raise ValueError("Histogram bin centres must be regularly spaced")
    return np.concatenate(
        ([centres[0] - 0.5 * width], centres + 0.5 * width)
    )


def plot_literature_high_cloud_structure(
    fields_by_run,
    classifications,
    vertical_profiles,
    structure=None,
):
    """Class-specific cloud-top-pressure and OLR distributions."""
    if not should_run_figure(2):
        return
    progress("Starting Figure 2: literature_high_cloud_structure")
    class_specs = REGIME_SPECS
    if structure is None:
        structure = compute_cloud_structure_product(
            fields_by_run, classifications
        )
    cached_ctp_centres = np.asarray(
        structure.cloud_top_pressure_bin.values, dtype=float
    )
    ctp_centres, _ = coarsen_histogram_for_plot(
        cached_ctp_centres,
        np.zeros_like(cached_ctp_centres),
        CLOUD_TOP_PRESSURE_DISPLAY_BIN_WIDTH_HPA,
    )
    ctp_edges = regular_histogram_bin_edges(ctp_centres)
    olr_centres = np.asarray(structure.olr_bin.values, dtype=float)
    rows = cloud_structure_metrics_table(structure)
    rows["cloud_top_pdf_display_bin_width_hpa"] = (
        CLOUD_TOP_PRESSURE_DISPLAY_BIN_WIDTH_HPA
    )
    rows["cloud_top_pdf_display_style"] = "unsmoothed histogram staircase"

    literature_color = "#7a4e00"
    ctp_literature = {
        0: (
            223.3,
            "MODIS deep cloud — Hong et al. (2007)",
        ),
        2: (
            281.4,
            "MODIS cirriform cloud — Hong et al. (2007)",
        ),
    }
    olr_literature = {
        0: (
            205.0,
            "Deep-cloud criterion — Gu et al. (2002)",
        ),
        2: (
            260.0,
            "Thin cirrus — Seiki et al. (2015)",
        ),
    }
    coordinate_profiles = vertical_profiles.get(sim_names[0])
    if (
        coordinate_profiles is None
        or "pressure_hpa" not in coordinate_profiles.coords
    ):
        raise RuntimeError(
            "Figure 2 requires the model-derived mean pressure-height profile"
        )
    save_table(rows, "literature_high_cloud_structure_metrics")
    save_table(
        pd.DataFrame(LITERATURE_CLOUD_STRUCTURE_VALUES),
        "high_cloud_structure_literature_values",
    )
    panel_letters = ("a", "b", "c")
    for domain_key, _latitude_bounds, domain_label in (
        CLOUD_STRUCTURE_DOMAIN_SPECS
    ):
        vertical_fig, ctp_axes = plt.subplots(
            1,
            3,
            figsize=(12.0, 4.6),
            constrained_layout=True,
            sharey=True,
        )
        olr_fig, olr_axes = plt.subplots(
            1,
            3,
            figsize=(12.0, 4.2),
            constrained_layout=True,
            sharey=True,
        )
        for isim, sim_name in enumerate(sim_names):
            for column, (key, _label, _color) in enumerate(class_specs):
                class_statistics = structure.sel(
                    simulation=sim_name,
                    latitude_domain=domain_key,
                    class_key=key,
                )
                ctp_mean = float(
                    class_statistics.mean_cloud_top_pressure_hpa
                )
                olr_mean = float(class_statistics.mean_olr_w_m2)
                _, ctp_histogram = coarsen_histogram_for_plot(
                    cached_ctp_centres,
                    class_statistics.cloud_top_pressure_histogram.values,
                    CLOUD_TOP_PRESSURE_DISPLAY_BIN_WIDTH_HPA,
                )
                ctp_axes[column].stairs(
                    100.0 * ctp_histogram,
                    ctp_edges,
                    orientation="horizontal",
                    color=cols[isim],
                    lw=1.8,
                    label=sim_name,
                )
                ctp_axes[column].axhline(
                    ctp_mean,
                    color=cols[isim],
                    lw=0.9,
                    ls="--",
                    alpha=0.8,
                )
                olr_axes[column].plot(
                    olr_centres,
                    100.0 * class_statistics.olr_histogram.values,
                    color=cols[isim],
                    lw=1.8,
                    label=sim_name,
                )
                olr_axes[column].axvline(
                    olr_mean,
                    color=cols[isim],
                    lw=0.9,
                    ls="--",
                    alpha=0.8,
                )
                ttl_text = (
                    f"{sim_name}: "
                    f"{100.0 * float(class_statistics.ttl_reaching_fraction):.1f}%"
                )
                ctp_axes[column].text(
                    0.98,
                    0.98 - 0.065 * isim,
                    ttl_text,
                    transform=ctp_axes[column].transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    color=cols[isim],
                )

        coordinate_profile = coordinate_profiles.sel(
            latitude_domain=domain_key
        )
        height_km = (
            np.asarray(coordinate_profile.z.values, dtype=float) / 1000.0
        )
        pressure_hpa = np.asarray(
            coordinate_profile.pressure_hpa.values, dtype=float
        )
        for column, (_key, label, class_color) in enumerate(class_specs):
            ctp_ax = ctp_axes[column]
            olr_ax = olr_axes[column]
            ctp_ax.axhspan(
                ctp_centres[0],
                180.0,
                color="#8da0cb",
                alpha=0.10,
                zorder=0,
            )
            ctp_ax.set_ylim(P_HIGH_HPA, max(10.0, ctp_centres[0]))
            ctp_ax.set_xlabel(
                "Probability per "
                f"{CLOUD_TOP_PRESSURE_DISPLAY_BIN_WIDTH_HPA:g} hPa bin (%)"
            )
            ctp_ax.set_title(
                f"({panel_letters[column]}) {label}",
                fontsize=10,
                color=class_color,
            )
            if column == 0:
                ctp_ax.set_ylabel("Cloud-top pressure (hPa)")
            add_height_pressure_axis(
                ctp_ax,
                height_km,
                pressure_hpa,
                primary="pressure",
            )
            if column in ctp_literature:
                value, citation = ctp_literature[column]
                ctp_ax.axhline(
                    value,
                    color=literature_color,
                    lw=1.3,
                    ls="-.",
                    label=citation,
                )
            ctp_ax.text(
                0.02,
                0.98,
                "TTL-reaching share\n(CTP ≤ 180 hPa)",
                transform=ctp_ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
                color="0.28",
            )
            ctp_ax.legend(frameon=False, fontsize=7, loc="best")

            olr_ax.set_xlabel("OLR (W m$^{-2}$)")
            olr_ax.set_title(
                f"({panel_letters[column]}) {label}",
                fontsize=10,
                color=class_color,
            )
            if column == 0:
                olr_ax.set_ylabel(
                    f"Probability per {OLR_BIN_WIDTH_W_M2:g} W m$^{{-2}}$ "
                    "bin (%)"
                )
            if column in olr_literature:
                value, citation = olr_literature[column]
                olr_ax.axvline(
                    value,
                    color=(
                        literature_color if column == 0 else "#375a7f"
                    ),
                    lw=1.3,
                    ls="-.",
                    label=citation,
                )
            olr_ax.legend(frameon=False, fontsize=7, loc="best")
            ctp_ax.grid(alpha=0.2)
            olr_ax.grid(alpha=0.2)

        vertical_fig.suptitle(
            f"Cloud-top distributions ({domain_label})",
            fontsize=11,
        )
        olr_fig.suptitle(
            f"High-cloud OLR distributions ({domain_label})",
            fontsize=11,
        )
        save_figure(
            2,
            vertical_fig,
            f"high_cloud_vertical_structure_{domain_key}",
        )
        save_figure(
            2,
            olr_fig,
            f"high_cloud_olr_distributions_{domain_key}",
        )
        plt.close(vertical_fig)
        plt.close(olr_fig)
    progress("Done with Figure 2 cloud structure and OLR distributions")


def main(argv=None):
    global figsave
    args = parse_command_line(argv)
    configure_run(args)
    os.makedirs(save_loc, exist_ok=True)
    os.makedirs(classification_cache_dir, exist_ok=True)
    figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)
    progress(
        "Selected high-cloud figures: "
        + ", ".join(str(number) for number in figures_to_run)
    )
    if TEST_MODE:
        progress(
            f"TEST MODE: simulation={sim_names[0]}, classified_times={TEST_TIME_STEPS}, "
            f"write_test_product={WRITE_CLASSIFICATION_PRODUCT}, output={save_loc}"
        )
    elif not WRITE_CLASSIFICATION_PRODUCT:
        progress("Classification products will be preserved and reused after version/schema validation")
    else:
        progress(
            "Classification products will be reused when current and rebuilt "
            "only when missing, stale, incomplete, or on the wrong Time axis"
        )
    if FIGURES_ONLY:
        progress(
            "FIGURES ONLY: derived products must already exist and match the "
            "current scientific configuration"
        )
    elif REBUILD_DERIVED_CACHE:
        progress("Requested derived high-cloud cache rebuild")
    else:
        progress(
            "Derived high-cloud products will be reused from the data repository"
        )
    write_classification_method_table()

    products = required_derived_products()
    cache_only_render = FIGURES_ONLY and not should_run_figure(3)
    if cache_only_render:
        progress(
            "Rendering selected figures directly from derived products; "
            "simulation fields and classification grids will not be opened"
        )
        initialise_derived_cache_signature_from_products(products)
        dss = []
        fields_by_run = []
        classifications = {}
    else:
        dss = load_datasets()
        initialise_derived_cache_signature(dss)
        fields_by_run = prepare_fields(dss)
        classifications = prepare_classifications(dss, fields_by_run)
    vertical_profiles = (
        get_or_compute_vertical_profiles(dss, classifications)
        if should_run_figure(2) or should_run_figure(4)
        else {}
    )
    budget = (
        load_or_compute_derived_table(
            "high_cloud_class_budget",
            lambda: compute_budget_table(classifications, fields_by_run),
        )
        if should_run_figure(1)
        else None
    )
    structure = (
        load_or_compute_derived_dataset(
            "high_cloud_structure_distributions",
            lambda: compute_cloud_structure_product(
                fields_by_run, classifications
            ),
        )
        if should_run_figure(2)
        else None
    )
    occurrence_maps = (
        get_or_compute_occurrence_maps(classifications)
        if should_run_figure(5)
        else None
    )
    rain_response = None
    if should_run_figure(6):
        if FIGURES_ONLY:
            rain_response = load_or_compute_derived_table(
                "rain_matched_high_cloud_response",
                lambda: pd.DataFrame(),
            )
        else:
            rain_blocks = load_or_compute_derived_table(
                "rain_matched_high_cloud_block_sums",
                lambda: rain_conditioned_block_sums(
                    classifications, fields_by_run
                ),
            )
            rain_response = load_or_compute_derived_table(
                "rain_matched_high_cloud_response",
                lambda: aggregate_rain_conditioned_response(rain_blocks),
            )

    plot_class_budget(classifications, fields_by_run, budget=budget)
    plot_literature_high_cloud_structure(
        fields_by_run,
        classifications,
        vertical_profiles,
        structure=structure,
    )
    plot_validation_snapshots(dss, fields_by_run, classifications)
    plot_vertical_contribution_profiles(vertical_profiles)
    plot_class_spatial_response(
        classifications,
        occurrence_maps=occurrence_maps,
    )
    plot_rain_matched_high_cloud_response(
        classifications,
        fields_by_run,
        response=rain_response,
    )

    progress("Refined high-cloud classification diagnostics complete")


if __name__ == "__main__":
    main()
