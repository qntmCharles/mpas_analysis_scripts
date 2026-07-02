#!/usr/bin/env python
# coding: utf-8

"""
Refined high-cloud classification diagnostics for MPAS-A nudging experiments.

This script owns the shared high-cloud classification product and focuses on a
small set of interpretable tests:

  * high-cloud class budgets and nudged-control differences;
  * observed-minus-control-predicted high-cloud residual maps/zonal means;
  * lat/lon overlay snapshots that validate the class masks visually;
  * explicit precipitation-source, vertical-velocity, and cloud-top validation;
  * lightweight cloud-fraction threshold sensitivity checks.
  * all-time lat/lon classification-validation movies.

The diagnostics deliberately avoid tercile splits of N2_TTL.  If N2_TTL is
available, a single common-support range is reported in the residual summary
tables rather than plotted as multiple weakly supported conditional regimes.
"""

import argparse
import gc
import os
import sys
import time
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
from matplotlib import animation, colors, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage

from diagnostic_helpers import *
from high_cloud_classification import (
    HighCloudThresholds,
    cached_classification_is_current as shared_cached_classification_is_current,
    classification_cache_metadata as shared_classification_cache_metadata,
    classification_dataset as shared_classification_dataset,
    classify_high_cloud_from_dataset,
    precip_component_rate_from_dataset,
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
#   1: high_cloud_class_budget
#   2: residual_high_cloud_maps_zonal
#   3: cloud_classification_validation_snapshots
#   4: cloud_regime_vertical_validation
#   5: cloud_regime_precip_and_cloud_top_validation
#   6: cloud_classification_threshold_sensitivity
#   7: cloud_classification_validation_movies
figures_to_run = []
figsave = None

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

TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
SNAPSHOT_LAT_BOUNDS = (-20.0, 20.0)

# Cloud-top pressure avoids tying tropical high cloud to one geometric height.
# The 440 hPa parent follows the standard ISCCP high-cloud boundary. Continuous
# cloud-top pressure is retained for the QBO-MJO cloud-deepening analysis.
CF_THRESH = 0.05
P_HIGH_HPA = 440.0
P_PRECIP_MIN = 1.0
P_HEAVY = 10.0
W_STRONG = 0.10
CONVECTIVE_FRACTION = 0.50
CONVECTIVE_EVIDENCE_REQUIRED = 2

WRITE_CLASSIFICATION_PRODUCT = True
classification_cache_dir = diagnostic_cache_dir("classification_cache", root_dir=root_dir)

TEST_MODE = False
TEST_TIME_STEPS = 16
TEST_START = None
TEST_DEFAULT_FIGURES = [1, 3, 4, 5, 6, 7]
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

LATITUDE_BIN_WIDTH = 2.0
MIN_BIN_COUNT = 50

RESIDUAL_FIT_SAMPLE_LIMITS = dict(max_time=120, max_lat=120, max_lon=240)
RESIDUAL_PLOT_SAMPLE_LIMITS = dict(max_time=144, max_lat=120, max_lon=240)
PROCESS_SAMPLE_LIMITS = dict(max_time=144, max_lat=120, max_lon=240)
SENSITIVITY_SAMPLE_LIMITS = dict(max_time=144, max_lat=100, max_lon=220)

SENSITIVITY_CF_THRESHOLDS = [0.03, CF_THRESH, 0.10]
SENSITIVITY_PRECIP_MIN_THRESHOLDS = [0.5, P_PRECIP_MIN, 2.0]
SENSITIVITY_INDICATOR_SETTINGS = [
    (5.0, 0.05),
    (P_HEAVY, W_STRONG),
    (20.0, 0.20),
]
SENSITIVITY_CONVECTIVE_FRACTIONS = [0.33, CONVECTIVE_FRACTION, 0.67]
SNAPSHOT_LON_WINDOW_DEG = 90.0
SNAPSHOT_Z_MAX_KM = 19.0
CLOUD_TOP_HEIGHT_MAP_MIN_KM = 6.0
CLOUD_TOP_HEIGHT_MAP_MAX_KM = 20.0
CLOUD_TOP_PRESSURE_BINS_HPA = np.arange(0.0, P_HIGH_HPA + 1.0, 40.0)
CLOUD_TOP_HISTOGRAM_TIME_BATCH = 32
VALIDATION_MOVIE_FPS = 6
VALIDATION_MOVIE_DPI = 110

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
    "w": ["w", "vertical_velocity"],
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
        + VAR_ALIASES["w"]
        + VAR_ALIASES["zgrid"]
        + VAR_ALIASES["u_zm"]
        + VAR_ALIASES["N2"]
        + VAR_ALIASES["dudz"]
    )
)

BUDGET_SPECS = [
    ("high_cloud_mask", "Total high cloud", "0.25"),
    ("convective_precipitating_high_cloud_mask", "Convective precipitating", "tab:red"),
    ("stratiform_precipitating_high_cloud_mask", "Stratiform precipitating", "tab:blue"),
    ("nonprecipitating_high_cloud_mask", "Non-precipitating", "tab:green"),
    ("unclassified_high_cloud_mask", "Invalid/unclassified", "0.55"),
]

RESIDUAL_TARGET_SPECS = [
    ("total", "Total high cloud", "high_cloud_fraction"),
    ("convective", "Convective precipitating", "convective_precipitating_high_cloud_mask"),
    ("stratiform", "Stratiform precipitating", "stratiform_precipitating_high_cloud_mask"),
    ("nonprecipitating", "Non-precipitating", "nonprecipitating_high_cloud_mask"),
]

REGIME_SPECS = [
    ("convective_precipitating_high_cloud_mask", "Convective precipitating", "tab:red"),
    ("stratiform_precipitating_high_cloud_mask", "Stratiform precipitating", "tab:blue"),
    ("nonprecipitating_high_cloud_mask", "Non-precipitating", "tab:green"),
]

# Persist only fields shared by downstream diagnostics. Full-resolution process
# evidence would make this product unnecessarily large; Figures 4-5 recompute it
# on the lightweight validation sample below.
CLASSIFICATION_CACHE_VARS = (
    "cloud_top_pressure",
    "cloud_top_height",
    "column_max_updraft",
    "high_cloud_fraction",
    "high_cloud_mask",
    "precipitating_high_cloud_mask",
    "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
    "unclassified_high_cloud_mask",
    "high_cloud_regime_code",
    "convective_evidence_count",
)

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
    (1, "Convective precipitating", "tab:red"),
    (2, "Stratiform precipitating", "tab:blue"),
    (3, "Non-precipitating", "tab:green"),
    (4, "Invalid/unclassified", "0.70"),
]


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


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
        help="Run one simulation over a short time window in an isolated output directory.",
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
        "--figures",
        type=int,
        nargs="+",
        choices=range(1, 8),
        default=None,
        metavar="N",
        help="Run only the listed figure numbers (1-7).",
    )
    parser.set_defaults(write_classification=True)
    args = parser.parse_args(argv)
    if args.test_times < 1:
        parser.error("--test-times must be at least 1")
    return args


def configure_run(args):
    global TEST_MODE, TEST_TIME_STEPS, TEST_START
    global WRITE_CLASSIFICATION_PRODUCT, classification_cache_dir
    global save_loc, figures_to_run
    global data_locs, latlon_file, zm_latlon_file, sim_names, cols

    TEST_MODE = bool(args.test)
    TEST_TIME_STEPS = int(args.test_times)
    TEST_START = args.test_start
    WRITE_CLASSIFICATION_PRODUCT = bool(args.write_classification)

    if args.figures is not None:
        figures_to_run = list(args.figures)
    elif TEST_MODE:
        figures_to_run = list(TEST_DEFAULT_FIGURES)

    if not TEST_MODE:
        return

    index = SIMULATION_ARGUMENTS[args.simulation]
    data_locs = [data_locs[index]]
    latlon_file = [latlon_file[index]]
    zm_latlon_file = [zm_latlon_file[index]]
    sim_names = [sim_names[index]]
    cols = [cols[index]]

    start_label = f"_{TEST_START}" if TEST_START else ""
    test_label = f"{args.simulation}{start_label}_{TEST_TIME_STEPS:03d}times"
    test_label = test_label.replace(":", "").replace(" ", "_")
    save_loc = os.path.join(default_save_loc, "testing", test_label)
    # Test products are deliberately isolated from products consumed downstream.
    classification_cache_dir = os.path.join(save_loc, "classification_products")


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


def write_classification_method_table():
    rows = [
        dict(
            component="cloudy model level",
            implementation=f"cldfrac >= {CF_THRESH:g}",
            threshold_origin="analysis convention; not percentile-derived",
            calibration_population="none",
            published_comparator=(
                "ISCCP supplies cloud-top-pressure classes, not a model-level "
                "cldfrac cutoff"
            ),
            reason_for_difference=(
                "A finite cldfrac cutoff is needed to convert fractional MPAS cloud "
                "into a diagnosed cloud top"
            ),
            sensitivity_values="cldfrac = 0.03, 0.05, 0.10",
            reference="Rossow and Schiffer (1999), BAMS / ISCCP",
            url="https://doi.org/10.1175/1520-0477(1999)080%3C2261:AIUCFI%3E2.0.CO;2",
            caveat="The central value is not claimed as a published threshold.",
        ),
        dict(
            component="high-cloud boundary",
            implementation=f"cloud-top pressure <= {P_HIGH_HPA:g} hPa",
            threshold_origin="published categorical boundary",
            calibration_population="none",
            published_comparator="ISCCP high-cloud boundary: cloud-top pressure <= 440 hPa",
            reason_for_difference="none; the published pressure boundary is used directly",
            sensitivity_values="not varied; retained as the fixed published boundary",
            reference="Rossow and Schiffer (1999), BAMS / ISCCP",
            url="https://doi.org/10.1175/1520-0477(1999)080%3C2261:AIUCFI%3E2.0.CO;2",
            caveat="Pressure, rather than a fixed geometric height, defines high cloud.",
        ),
        dict(
            component="precipitation occurrence",
            implementation=f"rainc + rainnc >= {P_PRECIP_MIN:g} mm day-1",
            threshold_origin="adapted literature convention; not percentile-derived",
            calibration_population="none",
            published_comparator=(
                "Kooperman et al. compare 0.1 and 1.0 mm day-1 dry-day thresholds "
                "for daily approximately 1-degree precipitation"
            ),
            reason_for_difference=(
                "The larger published comparison threshold is used to suppress drizzle, "
                "but applied here to a rate derived from precipitation-accumulation "
                "increments between 6-hourly instantaneous snapshots"
            ),
            sensitivity_values="rainc + rainnc = 0.5, 1.0, 2.0 mm day-1",
            reference="Kooperman et al. (2016), JAMES",
            url="https://doi.org/10.1002/2015MS000574",
            caveat="This is an adaptation, not a direct reproduction of their daily statistic.",
        ),
        dict(
            component="convective rain-intensity indicator",
            implementation=f"rainc + rainnc >= {P_HEAVY:g} mm day-1",
            threshold_origin=(
                "rounded percentile-informed threshold from the preliminary coarse-grid sample"
            ),
            calibration_population=(
                "approximately the 68th percentile of sampled precipitating high-cloud "
                "columns in the preliminary calibration sample"
            ),
            published_comparator=(
                "Schumacher and Houze describe convective rain >5 mm h-1; "
                "Shu et al. test a 20 mm h-1 surface-rain criterion"
            ),
            reason_for_difference=(
                f"The implemented {P_HEAVY:g} mm day-1 is {P_HEAVY / 24.0:.3g} mm h-1; "
                "0.5-degree spatial averaging and the use of precipitation-accumulation "
                "increments between snapshots dilute instantaneous kilometre-scale rates. "
                "cldfrac and w themselves are instantaneous snapshots"
            ),
            sensitivity_values=(
                "tested jointly with w: (5 mm day-1, 0.05 m s-1), "
                "(10, 0.10), (20, 0.20)"
            ),
            reference="Schumacher and Houze (2003); Shu et al. (2020)",
            url="https://doi.org/10.1029/2019JD031510",
            caveat=(
                "The original exploratory sample was not persisted as a calibration "
                "artifact. The approximate rank records the development-time diagnostic; "
                "current runs write realised ranks and do not retune by simulation."
            ),
        ),
        dict(
            component="convective vertical-velocity indicator",
            implementation=(
                f"max(w) over levels with cldfrac >= {CF_THRESH:g} is >= "
                f"{W_STRONG:g} m s-1"
            ),
            threshold_origin=(
                "rounded percentile-informed threshold from the preliminary coarse-grid sample"
            ),
            calibration_population=(
                "approximately the 90th percentile of sampled max(w) over cloudy levels "
                "in the preliminary calibration sample"
            ),
            published_comparator=(
                "Schumacher and Houze describe intermittent |w| >1 m s-1; Shu et al. "
                "test 3 m s-1 column and 0.5 m s-1 low-level criteria"
            ),
            reason_for_difference=(
                "Resolved w is strongly damped by 0.5-degree averaging and is diagnosed "
                "over all model levels satisfying the cldfrac cutoff, not within a resolved core"
            ),
            sensitivity_values=(
                "tested jointly with rain: (0.05 m s-1, 5 mm day-1), "
                "(0.10, 10), (0.20, 20)"
            ),
            reference="Schumacher and Houze (2003); Shu et al. (2020)",
            url="https://doi.org/10.1029/2019JD031510",
            caveat=(
                "The original exploratory sample was not persisted as a calibration "
                "artifact. The approximate rank records the development-time diagnostic; "
                "current runs write realised ranks and do not retune by simulation."
            ),
        ),
        dict(
            component="parameterized-rain indicator",
            implementation=f"rainc / (rainc + rainnc) >= {CONVECTIVE_FRACTION:g}",
            threshold_origin="equal-contribution convention; not percentile-derived",
            calibration_population="none",
            published_comparator=(
                "Kysely et al. evaluate parameterized convective and grid-scale model "
                "precipitation as process categories; they do not establish a universal "
                "0.5 physical convective/stratiform boundary"
            ),
            reason_for_difference=(
                "0.5 means rainc exceeds rainnc and is used only as one supporting indicator"
            ),
            sensitivity_values="rainc fraction = 0.33, 0.50, 0.67",
            reference="Kysely et al. (2016), Climate Dynamics",
            url="https://doi.org/10.1007/s00382-015-2580-7",
            caveat=(
                "rainc and rainnc identify parameterized and grid-scale model processes, "
                "not physical convective and stratiform rain. Neither source can determine "
                "the class by itself."
            ),
        ),
        dict(
            component="convective evidence rule",
            implementation=(
                f"convective when at least {CONVECTIVE_EVIDENCE_REQUIRED} of the 3 "
                "rain-intensity, vertical-velocity, and parameterized-rain indicators pass"
            ),
            threshold_origin="analysis design; not a published algorithm",
            calibration_population="none",
            published_comparator=(
                "Published radar/cloud-resolving methods combine intensity, texture, "
                "reflectivity, and vertical-motion evidence at much finer resolution"
            ),
            reason_for_difference=(
                "The 2-of-3 rule prevents any one coarse-grid indicator from defining convection"
            ),
            sensitivity_values="all three constituent thresholds are varied as listed above",
            reference="Schumacher and Houze (2003); Shu et al. (2020)",
            url="https://doi.org/10.1175/1520-0442(2003)016%3C1739:SRITTA%3E2.0.CO;2",
            caveat="The rule is intentionally simple and exhaustive, but remains an analysis choice.",
        ),
        dict(
            component="exhaustive class closure",
            implementation=(
                "non-precipitating if rain is below the occurrence threshold; "
                "otherwise convective if the evidence rule passes and stratiform if it does not"
            ),
            threshold_origin="logical closure; no additional threshold",
            calibration_population="none",
            published_comparator="not applicable",
            reason_for_difference="not applicable",
            sensitivity_values="closure is checked for every tested threshold set",
            reference="Schumacher and Houze (2003), Journal of Climate",
            url="https://doi.org/10.1175/1520-0442(2003)016%3C1739:SRITTA%3E2.0.CO;2",
            caveat="Unclassified is retained only for missing precipitation-component data.",
        ),
        dict(
            component="object tracking",
            implementation="PyFLEXTRKR 2-D connected features and bidirectional overlap linking",
            threshold_origin="published tracking framework with resolution-dependent configuration",
            calibration_population="not part of the cloud-class threshold calibration",
            published_comparator="PyFLEXTRKR methodology",
            reason_for_difference="tracking settings are documented by the organisation diagnostic",
            sensitivity_values="not part of Figure 6 cloud-class sensitivity",
            reference="Feng et al. (2023), Geoscientific Model Development",
            url="https://doi.org/10.5194/gmd-16-2753-2023",
            caveat="Segmentation and overlap thresholds remain resolution dependent and are exposed in snapshots and movies.",
        ),
    ]
    sensitivity_components = {
        "cloudy model level",
        "precipitation occurrence",
        "convective rain-intensity indicator",
        "convective vertical-velocity indicator",
        "parameterized-rain indicator",
        "convective evidence rule",
        "exhaustive class closure",
    }
    percentile_components = {
        "precipitation occurrence",
        "convective rain-intensity indicator",
        "convective vertical-velocity indicator",
        "parameterized-rain indicator",
    }
    for row in rows:
        row["sensitivity_output"] = (
            "06_cloud_classification_threshold_sensitivity.png and "
            "cloud_classification_threshold_sensitivity.csv"
            if row["component"] in sensitivity_components
            else "not included in the cloud-threshold sensitivity figure"
        )
        row["realised_percentile_output"] = (
            "cloud_classification_threshold_percentile_ranks.csv"
            if row["component"] in percentile_components
            else "not percentile-calibrated"
        )
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


def classification_thresholds(
    cf_thresh=CF_THRESH,
    precip_min=P_PRECIP_MIN,
    heavy_precip=P_HEAVY,
    strong_updraft=W_STRONG,
    convective_fraction=CONVECTIVE_FRACTION,
):
    return HighCloudThresholds(
        cf_thresh=float(cf_thresh),
        high_cloud_top_pressure_hpa=P_HIGH_HPA,
        surface_precip_min_mm_day=float(precip_min),
        heavy_precip_mm_day=float(heavy_precip),
        strong_updraft_m_s=float(strong_updraft),
        convective_fraction_threshold=float(convective_fraction),
        convective_evidence_required=int(CONVECTIVE_EVIDENCE_REQUIRED),
    )


def classify_high_cloud(
    ds,
    cldfrac,
    fields,
    cf_thresh=CF_THRESH,
    precip_min=P_PRECIP_MIN,
    heavy_precip=P_HEAVY,
    strong_updraft=W_STRONG,
    convective_fraction=CONVECTIVE_FRACTION,
    include_process_diagnostics=False,
):
    """Shared 0.5-degree high-cloud precipitation-regime decomposition."""
    return classify_high_cloud_from_dataset(
        ds,
        precip=fields["precip"],
        cldfrac=cldfrac,
        convective_precip=fields["convective_precip"],
        grid_scale_precip=fields["grid_scale_precip"],
        thresholds=classification_thresholds(
            cf_thresh,
            precip_min=precip_min,
            heavy_precip=heavy_precip,
            strong_updraft=strong_updraft,
            convective_fraction=convective_fraction,
        ),
        warn_missing=False,
        dt_hours=dt,
        include_process_diagnostics=include_process_diagnostics,
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
    progress(f"{sim_name}: using cached high-cloud classification {cache_path}")
    return {name: cached[name] for name in cached.data_vars}


def classification_product_path(sim_name):
    filename = f"{sim_name.replace(' ', '_')}_classified_high_cloud.nc"
    return os.path.join(classification_cache_dir, filename)


def invalidate_classification_products():
    """Ensure an interrupted rebuild cannot expose an older final product."""
    for sim_name in sim_names:
        path = classification_product_path(sim_name)
        for candidate in (path, path + ".tmp"):
            if os.path.exists(candidate):
                progress(f"{sim_name}: removing previous classification product {candidate}")
                os.remove(candidate)


def get_or_build_classification(ds, fields, sim_name):
    cache_path = classification_product_path(sim_name)

    if not WRITE_CLASSIFICATION_PRODUCT:
        if not os.path.exists(cache_path):
            raise RuntimeError(
                f"{sim_name}: --no-write-classification requires an existing "
                f"classification product: {cache_path}. Run without "
                "--no-write-classification to create it."
            )
        cached = open_cached_classification(cache_path, sim_name)
        if cached is None:
            raise RuntimeError(
                f"{sim_name}: --no-write-classification requires a current, complete "
                f"classification product: {cache_path}. Run without "
                "--no-write-classification to rebuild it."
            )
        return cached

    progress(f"{sim_name}: preparing high-cloud classification")
    cf = get_cloud_fraction(ds)
    if cf is None:
        progress(f"{sim_name}: no usable cloud fraction; skipping high-cloud classification")
        return None

    precip = fields["precip"]
    if "Time" in cf.dims and "Time" in precip.dims:
        common_time = cf.Time.to_index().intersection(precip.Time.to_index())
        if len(common_time) == 0:
            warnings.warn(f"{sim_name}: cloud fraction and precipitation have no overlapping times.")
            return None
        cf = cf.sel(Time=common_time)
        precip = precip.sel(Time=common_time)

    progress(f"{sim_name}: building high-cloud classification graph")
    class_fields = dict(fields)
    class_fields["precip"] = precip
    class_fields["convective_precip"] = fields["convective_precip"].sel(Time=precip.Time)
    class_fields["grid_scale_precip"] = fields["grid_scale_precip"].sel(Time=precip.Time)
    classif = classify_high_cloud(ds, cf, class_fields)
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
        "convective_precipitating_high_cloud_mask",
        "stratiform_precipitating_high_cloud_mask",
        "nonprecipitating_high_cloud_mask",
        "unclassified_high_cloud_mask",
    ]
    overlay_keys = ["precipitating_high_cloud_mask"]
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
                        note="precipitating high cloud is the union of convective and stratiform regimes",
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
#     runs, with the difference carried mainly by convective precipitating,
#     stratiform precipitating, or non-precipitating high cloud.
#   * Invalid/unclassified high cloud should be near zero; a substantial value
#     indicates missing precipitation-component data rather than a fourth regime.
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

    has_differences = len(sim_names) > 1
    ncols = 2 if has_differences else 1
    fig, ax = plt.subplots(
        1,
        ncols,
        figsize=(11.4 if has_differences else 6.2, 5.3),
        constrained_layout=True,
        sharey=True,
        squeeze=False,
    )
    ax = ax[0]
    abs_df = budget[budget.statistic == "absolute_fraction"].set_index(["simulation", "class_key"])
    diff_df = budget[budget.statistic == "difference_from_control"].set_index(["simulation", "class_key"])

    for i, sim_name in enumerate(sim_names):
        yy = y + (i - (len(sim_names) - 1) / 2.0) * height
        vals = [
            100.0 * abs_df.loc[(sim_name, key), "value"] if (sim_name, key) in abs_df.index else np.nan
            for key in keys
        ]
        ax[0].barh(yy, vals, height=height, color=cols[i], alpha=0.82, label=sim_name)
        for j, key in enumerate(keys):
            if (sim_name, key) in abs_df.index:
                row = abs_df.loc[(sim_name, key)]
                draw_errorbarh(
                    ax[0], yy[j], 100.0 * row["value"],
                    100.0 * row["ci_low"], 100.0 * row["ci_high"], color="0.15"
                )

    if has_differences:
        for i, sim_name in enumerate(sim_names[1:], start=1):
            yy = y + (i - 1.5) * height
            vals = [
                100.0 * diff_df.loc[(sim_name, key), "value"] if (sim_name, key) in diff_df.index else np.nan
                for key in keys
            ]
            ax[1].barh(yy, vals, height=height, color=cols[i], alpha=0.82, label=f"{sim_name} - control")
            for j, key in enumerate(keys):
                if (sim_name, key) in diff_df.index:
                    row = diff_df.loc[(sim_name, key)]
                    draw_errorbarh(
                        ax[1], yy[j], 100.0 * row["value"],
                        100.0 * row["ci_low"], 100.0 * row["ci_high"], color="0.15"
                    )

    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Tropical area-time (%)")
    ax[0].set_title("Class occurrence")
    ax[0].grid(axis="x", alpha=0.22)
    ax[0].legend(frameon=False, fontsize=8)

    if has_differences:
        ax[1].axvline(0.0, color="0.45", lw=0.8)
        ax[1].set_xlabel("Simulation minus control (percentage points)")
        ax[1].set_title("Difference from control")
        ax[1].grid(axis="x", alpha=0.22)
        ax[1].legend(frameon=False, fontsize=8)

    fig.suptitle("High-cloud class occurrence as a percentage of tropical area-time")
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
    if target_key == "convective":
        return classif["convective_precipitating_high_cloud_mask"].astype(float)
    if target_key == "stratiform":
        return classif["stratiform_precipitating_high_cloud_mask"].astype(float)
    if target_key == "nonprecipitating":
        return classif["nonprecipitating_high_cloud_mask"].astype(float)
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
    last_im = None

    for r, (target_key, target_label, _source) in enumerate(RESIDUAL_TARGET_SPECS):
        row_maps = []
        for sim_name in sim_names:
            residual = residual_data[(target_key, sim_name)]
            m = 100.0 * (
                residual.mean(dim="Time", skipna=True)
                if "Time" in residual.dims
                else residual
            )
            row_maps.append(m)
        row_abs = np.nanpercentile(np.abs(np.concatenate([np.ravel(m.values) for m in row_maps])), 98)
        row_abs = float(row_abs) if np.isfinite(row_abs) and row_abs > 0 else 0.05
        row_levels = symmetric_levels(max(3.0, row_abs), ndiv=15)

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
            z = 100.0 * residual.mean(dim=dims, skipna=True)
            a.plot(z.latitude, z, color=cols[i], lw=1.5, label=sim_name if r == 0 else None)
        a.axhline(0.0, color="0.55", lw=0.8)
        a.grid(alpha=0.22)
        if r == 0:
            a.set_title("Time- and longitude-mean residual")
        if r == nrows - 1:
            a.set_xlabel("Latitude")
        a.set_xlim(*TROPICAL_LAT_BOUNDS)
        a.set_ylabel("Actual minus control-model prediction\n(percentage points)")

    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=ax[:, :-1].ravel().tolist(),
            label="Actual minus control-model prediction (percentage points)",
        )
    ax[0, -1].legend(frameon=False, fontsize=8)
    subtitle = "masked outside control q01-q99 feature support"
    if n2_support is not None and np.isfinite(n2_support.get("common_lo", np.nan)):
        subtitle += f"; N2_TTL common-support summary: {n2_support['common_lo']:.3g} to {n2_support['common_hi']:.3g}"
    fig.suptitle(f"Actual minus control-model-predicted high cloud ({subtitle})")
    save_figure(2, fig, "residual_high_cloud_maps_zonal")
    plt.close(fig)
    progress("Done with figure: residual_high_cloud_maps_zonal")


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


def class_legend_handles():
    return [
        Patch(facecolor=color, label=label)
        for _code, label, color in CLASS_CODE_SPECS
    ]


def classification_threshold_summary():
    return (
        f"High: cldfrac >= {CF_THRESH:g}, cloud-top p <= {P_HIGH_HPA:g} hPa | "
        f"Convective: >={CONVECTIVE_EVIDENCE_REQUIRED}/3 "
        f"[P >= {P_HEAVY:g} mm/d, wmax >= {W_STRONG:g} m/s, "
        f"rainc/P >= {CONVECTIVE_FRACTION:g}]"
    )


def overlay_class_tint(ax, class_code, alpha=0.20):
    cmap, norm = class_code_cmap_norm(alpha=alpha)
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
    ax.set_xlim(float(focus_lon) - 0.5 * SNAPSHOT_LON_WINDOW_DEG, float(focus_lon) + 0.5 * SNAPSHOT_LON_WINDOW_DEG)
    ax.set_ylim(*SNAPSHOT_LAT_BOUNDS)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("P = rainc + rainnc; high-cloud class tint")
    return im


def plot_updraft_map(ax, column_max_w, class_code, focus_lon, focus_lat):
    w = crop_longitude_window(column_max_w, focus_lon)
    classes = crop_longitude_window(class_code, focus_lon)
    norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=0.4)
    im = ax.pcolormesh(
        w.longitude,
        w.latitude,
        w,
        shading="auto",
        cmap="RdBu_r",
        norm=norm,
    )
    overlay_class_tint(ax, classes)
    ax.axvline(focus_lon, color="0.1", lw=1.0, ls=":")
    ax.axhline(focus_lat, color="0.1", lw=1.0, ls=":")
    ax.set_xlim(float(focus_lon) - 0.5 * SNAPSHOT_LON_WINDOW_DEG, float(focus_lon) + 0.5 * SNAPSHOT_LON_WINDOW_DEG)
    ax.set_ylim(*SNAPSHOT_LAT_BOUNDS)
    ax.set_xlabel("Longitude")
    ax.tick_params(labelleft=False)
    ax.set_title(
        f"max(w) over cldfrac >= {CF_THRESH:g} levels; high-cloud class tint"
    )
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
    ax.set_xlim(
        float(focus_lon) - 0.5 * SNAPSHOT_LON_WINDOW_DEG,
        float(focus_lon) + 0.5 * SNAPSHOT_LON_WINDOW_DEG,
    )
    ax.set_ylim(*SNAPSHOT_LAT_BOUNDS)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Cloud-top height; cyan tint = high-cloud mask")
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
    ax.set_title("High-cloud column class")
    return im


def save_validation_map_figure(
    sim_name,
    snap_time,
    precip,
    high_cloud,
    column_max_w,
    cloud_top_height,
    class_code,
    focus_lon,
    focus_lat,
):
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 8.8), constrained_layout=False)
    fig.subplots_adjust(
        left=0.06,
        right=0.94,
        bottom=0.15,
        top=0.87,
        wspace=0.24,
        hspace=0.32,
    )
    precip_im = plot_precip_map(
        axes[0, 0], precip, class_code, focus_lon, focus_lat
    )
    updraft_im = plot_updraft_map(
        axes[0, 1], column_max_w, class_code, focus_lon, focus_lat
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
        label="rainc + rainnc (mm day$^{-1}$)",
        fraction=0.046,
        pad=0.025,
    )
    fig.colorbar(
        updraft_im,
        ax=axes[0, 1],
        label=f"max(w) over levels with cldfrac >= {CF_THRESH:g} (m s$^{{-1}}$)",
        fraction=0.046,
        pad=0.025,
    )
    fig.colorbar(
        height_im,
        ax=axes[1, 0],
        label="Cloud-top height (km)",
        fraction=0.046,
        pad=0.025,
    )
    fig.legend(
        handles=class_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=len(CLASS_CODE_SPECS),
        frameon=False,
        title="High-cloud column class (tint in top row)",
    )
    fig.suptitle(
        f"{sim_name}: high-cloud classification at {format_time_value(snap_time)}\n"
        f"{classification_threshold_summary()}"
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
    ax.set_title(title)
    ax.grid(alpha=0.15, lw=0.4)
    if evidence_handles:
        ax.legend(handles=evidence_handles, loc="upper right", fontsize=6, frameon=True)
    return im


def snapshot_pressure(ds, cf_snapshot, snap_time):
    name = find_var(ds, "pressure", required=True)
    raw = select_time_value(finite_or_nan(ds[name]), snap_time)
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
        figsize=(12.5, 3.6 * len(sim_names)),
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
            (
                fields_by_run[i]["convective_precip"]
                + fields_by_run[i]["grid_scale_precip"]
            ).sel(Time=snap_time),
            SNAPSHOT_LAT_BOUNDS,
        ).load()
        high = subset_lat(classif["high_cloud_mask"].sel(Time=snap_time).astype(float), SNAPSHOT_LAT_BOUNDS).load()
        column_max_w = subset_lat(
            classif["column_max_updraft"].sel(Time=snap_time),
            SNAPSHOT_LAT_BOUNDS,
        ).load()
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
            column_max_w,
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
                convective_precipitating_fraction=weighted_area_time_mean(convective, SNAPSHOT_LAT_BOUNDS),
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
# Figures 4-5. Cloud-top and precipitation-source validation
# -----------------------------------------------------------------------------


REGIME_CODE_LABELS = {
    1: "Convective precipitating",
    2: "Stratiform precipitating",
    3: "Non-precipitating high cloud",
    4: "Invalid",
}
REGIME_CODE_COLORS = {1: "tab:red", 2: "tab:blue", 3: "tab:green", 4: "0.55"}


def sampled_regime_table(sim_name, classif):
    keys = [
        "cloud_top_pressure",
        "cloud_top_height",
        "column_max_updraft",
        "high_cloud_fraction",
        "surface_precip_rate",
        "convective_precip_rate",
        "grid_scale_precip_rate",
        "convective_precip_fraction",
        "precip_source_dominance_index",
        "heavy_precipitation_evidence",
        "strong_updraft_evidence",
        "parameterized_source_evidence",
        "convective_evidence_count",
        "high_cloud_regime_code",
        "high_cloud_mask",
    ]
    arrays = [subset_lat(classif[key], TROPICAL_LAT_BOUNDS) for key in keys]
    arrays = list(xr.align(*arrays, join="inner"))
    indexers = stride_indexers(
        arrays,
        **PROCESS_SAMPLE_LIMITS,
        time_mode="stride",
    )
    if indexers:
        arrays = [
            da.isel({dim: slc for dim, slc in indexers.items() if dim in da.dims})
            for da in arrays
        ]
    sampled = xr.Dataset(dict(zip(keys, arrays))).load()
    frame = sampled.to_dataframe().reset_index()
    frame = frame[frame.high_cloud_mask >= 0.5].copy()
    frame["simulation"] = sim_name
    frame["regime"] = frame.high_cloud_regime_code.map(REGIME_CODE_LABELS)
    return frame


def regime_process_summary(frame):
    variables = [
        "cloud_top_pressure",
        "cloud_top_height",
        "high_cloud_fraction",
        "surface_precip_rate",
        "column_max_updraft",
        "convective_precip_fraction",
        "precip_source_dominance_index",
        "convective_evidence_count",
    ]
    rows = []
    for (simulation, regime), group in frame.groupby(["simulation", "regime"], dropna=False):
        for variable in variables:
            values = pd.to_numeric(group[variable], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                dict(
                    simulation=simulation,
                    regime=regime,
                    variable=variable,
                    count=int(values.size),
                    median=float(np.nanmedian(values)) if values.size else np.nan,
                    p10=float(np.nanpercentile(values, 10)) if values.size else np.nan,
                    p90=float(np.nanpercentile(values, 90)) if values.size else np.nan,
                )
            )
    return pd.DataFrame(rows)


def weighted_percentile_rank(values, threshold, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return np.nan
    return float(100.0 * weights[valid & (values <= threshold)].sum() / weights[valid].sum())


def threshold_percentile_table(frame):
    """Report realised ranks; these diagnose, but do not retune, common thresholds."""
    if frame is None or len(frame) == 0:
        return pd.DataFrame()
    rows = []
    specs = [
        dict(
            indicator="precipitation occurrence",
            variable="surface_precip_rate",
            threshold=P_PRECIP_MIN,
            population="all sampled tropical high-cloud columns",
            percentile_selected=False,
            preliminary_rank="not percentile-selected",
        ),
        dict(
            indicator="convective rain intensity",
            variable="surface_precip_rate",
            threshold=P_HEAVY,
            population="sampled tropical precipitating high-cloud columns (P >= 1 mm day-1)",
            percentile_selected=True,
            preliminary_rank="approximately p68",
        ),
        dict(
            indicator="convective vertical velocity",
            variable="column_max_updraft",
            threshold=W_STRONG,
            population="all sampled tropical high-cloud columns with finite max(w)",
            percentile_selected=True,
            preliminary_rank="approximately p90",
        ),
        dict(
            indicator="parameterized-rain contribution",
            variable="convective_precip_fraction",
            threshold=CONVECTIVE_FRACTION,
            population="sampled tropical precipitating high-cloud columns (P >= 1 mm day-1)",
            percentile_selected=False,
            preliminary_rank="not percentile-selected; equal-contribution boundary",
        ),
    ]
    for sim_name in sim_names:
        sim = frame[frame.simulation == sim_name].copy()
        if len(sim) == 0:
            continue
        if "latitude" in sim.columns:
            latitude = pd.to_numeric(sim["latitude"], errors="coerce").to_numpy(dtype=float)
            base_weights = np.cos(np.deg2rad(latitude))
        else:
            base_weights = np.ones(len(sim), dtype=float)
        precip = pd.to_numeric(sim.surface_precip_rate, errors="coerce").to_numpy(dtype=float)
        for spec in specs:
            values = pd.to_numeric(sim[spec["variable"]], errors="coerce").to_numpy(dtype=float)
            population_mask = np.ones(len(sim), dtype=bool)
            if "precipitating" in spec["population"]:
                population_mask &= precip >= P_PRECIP_MIN
            rank = weighted_percentile_rank(
                values[population_mask],
                spec["threshold"],
                base_weights[population_mask],
            )
            rows.append(
                dict(
                    simulation=sim_name,
                    indicator=spec["indicator"],
                    variable=spec["variable"],
                    implemented_threshold=float(spec["threshold"]),
                    calibration_was_percentile_informed=bool(spec["percentile_selected"]),
                    preliminary_calibration_rank=spec["preliminary_rank"],
                    realised_weighted_percentile_rank=rank,
                    realised_sample_count=int(
                        np.count_nonzero(
                            population_mask
                            & np.isfinite(values)
                            & np.isfinite(base_weights)
                            & (base_weights > 0.0)
                        )
                    ),
                    percentile_population=spec["population"],
                    weighting="cosine(latitude)",
                    role=(
                        "diagnostic only; one common threshold is retained across simulations"
                    ),
                )
            )
    return pd.DataFrame(rows)


def build_process_validation_frame(dss, fields_by_run):
    frames = []
    for sim_name, ds, fields in zip(sim_names, dss, fields_by_run):
        progress(f"{sim_name}: preparing sampled cloud-top validation")
        classif = sampled_classification_for_threshold(
            ds,
            fields,
            CF_THRESH,
            sample_limits=PROCESS_SAMPLE_LIMITS,
            include_process_diagnostics=False,
        )
        if classif is not None:
            frames.append(sampled_regime_table(sim_name, classif))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_regime_process_validation(frame):
    if not should_run_figure(4):
        return
    progress("Starting Figure 4: cloud_regime_vertical_validation")
    if frame is None or len(frame) == 0:
        return
    save_table(regime_process_summary(frame), "cloud_regime_process_summary")
    specs = [
        ("cloud_top_pressure", f"Pressure of highest level with cldfrac >= {CF_THRESH:g} (hPa)", False),
        ("surface_precip_rate", "rainc + rainnc (mm day$^{-1}$)", True),
        ("column_max_updraft", f"max(w) where cldfrac >= {CF_THRESH:g} (m s$^{{-1}}$)", False),
        ("convective_precip_fraction", "rainc / (rainc + rainnc)", False),
    ]
    regime_codes = [1, 2, 3]
    fig, axes = plt.subplots(
        len(sim_names),
        len(specs),
        figsize=(15.5, max(3.8, 2.8 * len(sim_names))),
        constrained_layout=True,
        squeeze=False,
    )
    for i, sim_name in enumerate(sim_names):
        sub = frame[frame.simulation == sim_name]
        for j, (variable, label, log_transform) in enumerate(specs):
            ax = axes[i, j]
            values_by_regime = []
            for code in regime_codes:
                values = pd.to_numeric(
                    sub.loc[sub.high_cloud_regime_code == code, variable], errors="coerce"
                ).to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if log_transform:
                    values = np.log10(np.maximum(values, 1.0e-6))
                values_by_regime.append(values)
            bp = ax.boxplot(values_by_regime, patch_artist=True, showfliers=False)
            for patch, code in zip(bp["boxes"], regime_codes):
                patch.set_facecolor(REGIME_CODE_COLORS[code])
                patch.set_alpha(0.65)
            ax.set_xticks([1, 2, 3], ["Conv.", "Strat.", "Non-precip."])
            ax.set_ylabel(("log10 " if log_transform else "") + label)
            if variable == "cloud_top_pressure":
                ax.invert_yaxis()
            elif variable == "surface_precip_rate":
                ax.axhline(np.log10(P_HEAVY), color="0.2", lw=0.9, ls="--")
            elif variable == "column_max_updraft":
                ax.axhline(W_STRONG, color="0.2", lw=0.9, ls="--")
            elif variable == "convective_precip_fraction":
                ax.axhline(CONVECTIVE_FRACTION, color="0.2", lw=0.9, ls="--")
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.text(-0.30, 0.5, sim_name, transform=ax.transAxes, rotation=90, va="center")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"Inputs to the {CONVECTIVE_EVIDENCE_REQUIRED}-of-3 convective classification, grouped by assigned high-cloud class"
    )
    save_figure(4, fig, "cloud_regime_vertical_validation")
    plt.close(fig)
    progress("Done with figure: cloud_regime_vertical_validation")


def cloud_top_pressure_budget_rows(classifications):
    """Return class contributions to tropical area-time in cloud-top-pressure bins."""
    rows = []
    profile_specs = REGIME_SPECS + [
        ("unclassified_high_cloud_mask", "Invalid/unclassified", "0.55")
    ]
    needed = ["cloud_top_pressure"] + [key for key, _label, _color in profile_specs]
    for sim_name in sim_names:
        classif = classifications.get(sim_name)
        if classif is None:
            continue
        data = xr.Dataset(
            {key: subset_lat(classif[key], TROPICAL_LAT_BOUNDS) for key in needed}
        )
        if "Time" not in data.dims:
            data = data.expand_dims(Time=[0])
        lat_weights = np.cos(np.deg2rad(np.asarray(data.latitude.values, dtype=float)))
        histograms = {
            key: np.zeros(len(CLOUD_TOP_PRESSURE_BINS_HPA) - 1, dtype=float)
            for key, _label, _color in profile_specs
        }
        class_weight_totals = {
            key: 0.0 for key, _label, _color in profile_specs
        }
        denominator = 0.0
        for start in range(0, data.sizes["Time"], CLOUD_TOP_HISTOGRAM_TIME_BATCH):
            stop = min(start + CLOUD_TOP_HISTOGRAM_TIME_BATCH, data.sizes["Time"])
            block = data.isel(Time=slice(start, stop)).load()
            cloud_top_pressure = np.asarray(
                block["cloud_top_pressure"].transpose("Time", "latitude", "longitude").values,
                dtype=float,
            )
            weights = np.broadcast_to(
                lat_weights[None, :, None], cloud_top_pressure.shape
            )
            denominator += float(
                (stop - start) * data.sizes["longitude"] * lat_weights.sum()
            )
            for key, _label, _color in profile_specs:
                mask = np.asarray(
                    block[key].transpose("Time", "latitude", "longitude").values,
                    dtype=bool,
                )
                valid = mask & np.isfinite(cloud_top_pressure)
                class_weight_totals[key] += float(weights[mask].sum())
                counts, _ = np.histogram(
                    cloud_top_pressure[valid],
                    bins=CLOUD_TOP_PRESSURE_BINS_HPA,
                    weights=weights[valid],
                )
                histograms[key] += counts
            del block, cloud_top_pressure, weights

        if denominator <= 0.0:
            continue
        for key, label, _color in profile_specs:
            omitted = class_weight_totals[key] - histograms[key].sum()
            if abs(omitted) > max(1.0e-8 * denominator, 1.0e-8):
                raise RuntimeError(
                    f"{sim_name}: cloud-top-pressure bins omit {omitted / denominator:.3e} "
                    f"of tropical area-time for {label}"
                )
        for key, label, _color in profile_specs:
            for lower, upper, weighted_count in zip(
                CLOUD_TOP_PRESSURE_BINS_HPA[:-1],
                CLOUD_TOP_PRESSURE_BINS_HPA[1:],
                histograms[key],
            ):
                rows.append(
                    dict(
                        simulation=sim_name,
                        class_key=key,
                        class_label=label,
                        cloud_top_pressure_lower_hpa=float(lower),
                        cloud_top_pressure_upper_hpa=float(upper),
                        cloud_top_pressure_midpoint_hpa=float(0.5 * (lower + upper)),
                        tropical_area_time_percent=float(100.0 * weighted_count / denominator),
                        denominator="all tropical grid cells and times, cosine-latitude weighted",
                    )
                )
    return pd.DataFrame(rows)


def plot_precip_component_validation(frame, classifications):
    if not should_run_figure(5):
        return
    progress("Starting Figure 5: cloud_regime_precip_and_cloud_top_validation")
    if frame is None or len(frame) == 0:
        return
    pressure_frame = cloud_top_pressure_budget_rows(classifications)
    save_table(pressure_frame, "cloud_top_pressure_class_area_time_budget")
    fig, axes = plt.subplots(
        3,
        len(sim_names),
        figsize=(max(6.2, 4.7 * len(sim_names)), 12.0),
        constrained_layout=True,
        squeeze=False,
    )
    rng = np.random.default_rng(20240630)
    for i, sim_name in enumerate(sim_names):
        sub = frame[frame.simulation == sim_name]
        for code in (3, 2, 1):
            group = sub[sub.high_cloud_regime_code == code]
            if len(group) > 5000:
                group = group.iloc[rng.choice(len(group), size=5000, replace=False)]
            color = REGIME_CODE_COLORS[code]
            label = REGIME_CODE_LABELS[code]
            axes[0, i].scatter(
                np.log10(np.maximum(group.surface_precip_rate, 1.0e-4)),
                group.column_max_updraft,
                s=5,
                alpha=0.28,
                color=color,
                label=label,
            )
            axes[1, i].scatter(
                np.log10(np.maximum(group.grid_scale_precip_rate, 1.0e-4)),
                np.log10(np.maximum(group.convective_precip_rate, 1.0e-4)),
                s=5,
                alpha=0.28,
                color=color,
                label=label,
            )
        axes[0, i].set_title(sim_name)
        axes[0, i].axvline(np.log10(P_HEAVY), color="0.2", lw=0.9, ls="--")
        axes[0, i].axhline(W_STRONG, color="0.2", lw=0.9, ls="--")
        axes[0, i].set_xlabel("log10(rainc + rainnc [mm day$^{-1}$])")
        axes[0, i].set_ylabel(f"max(w) where cldfrac >= {CF_THRESH:g} (m s$^{{-1}}$)")

        finite_sources = np.concatenate(
            [
                pd.to_numeric(sub.grid_scale_precip_rate, errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(sub.convective_precip_rate, errors="coerce").to_numpy(dtype=float),
            ]
        )
        finite_sources = finite_sources[np.isfinite(finite_sources)]
        source_upper = max(
            1.0,
            float(np.ceil(np.log10(max(np.nanpercentile(finite_sources, 99.5), 1.0))))
            if finite_sources.size
            else 1.0,
        )
        source_lower = -4.0
        axes[1, i].plot(
            [source_lower, source_upper],
            [source_lower, source_upper],
            color="0.15",
            lw=1.0,
            ls="--",
            label=f"rainc/(rainc + rainnc) = {CONVECTIVE_FRACTION:g}",
        )
        rainnc_curve = np.geomspace(1.0e-4, P_HEAVY - 1.0e-4, 250)
        rainc_curve = P_HEAVY - rainnc_curve
        axes[1, i].plot(
            np.log10(rainnc_curve),
            np.log10(rainc_curve),
            color="#b21878",
            lw=1.0,
            ls=":",
            label=f"rainc + rainnc = {P_HEAVY:g} mm day$^{{-1}}$",
        )
        axes[1, i].set_xlim(source_lower, source_upper)
        axes[1, i].set_ylim(source_lower, source_upper)
        axes[1, i].set_aspect("equal", adjustable="box")
        axes[1, i].set_xlabel("log10(rainnc [mm day$^{-1}$])")
        axes[1, i].set_ylabel("log10(rainc [mm day$^{-1}$])")

        pressure_sub = pressure_frame[pressure_frame.simulation == sim_name]
        left = np.zeros(len(CLOUD_TOP_PRESSURE_BINS_HPA) - 1, dtype=float)
        pressure_midpoints = 0.5 * (
            CLOUD_TOP_PRESSURE_BINS_HPA[:-1] + CLOUD_TOP_PRESSURE_BINS_HPA[1:]
        )
        for class_key, label, color in REGIME_SPECS + [
            ("unclassified_high_cloud_mask", "Invalid/unclassified", "0.55")
        ]:
            values = (
                pressure_sub[pressure_sub.class_key == class_key]
                .set_index("cloud_top_pressure_midpoint_hpa")
                .reindex(pressure_midpoints)
                .tropical_area_time_percent
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            axes[2, i].barh(
                pressure_midpoints,
                values,
                left=left,
                height=32.0,
                color=color,
                edgecolor="none",
                label=label,
            )
            left += values
        axes[2, i].set_ylim(P_HIGH_HPA, 0.0)
        axes[2, i].set_xlabel("Tropical area-time (%) per 40 hPa cloud-top-pressure bin")
        axes[2, i].set_ylabel("Cloud-top pressure (hPa)")
        axes[0, i].grid(alpha=0.18)
        axes[1, i].grid(alpha=0.18)
        axes[2, i].grid(axis="x", alpha=0.18)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=7)
    axes[2, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Classification evidence by assigned high-cloud class\n"
        "Absolute tropical area-time occurrence by cloud-top pressure"
    )
    save_figure(5, fig, "cloud_regime_precip_and_cloud_top_validation")
    plt.close(fig)
    progress("Done with figure: cloud_regime_precip_and_cloud_top_validation")


# -----------------------------------------------------------------------------
# Figure 6. Lightweight definition sensitivity
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * The three science classes should keep the same qualitative interpretation
#     under the small requested tests of cldfrac, precipitation occurrence,
#     paired rain/w evidence, and rainc contribution thresholds.
#   * Only cldfrac requires re-reading 3-D model fields; the other tests
#     reclassify already sampled 2-D evidence fields.
#   * Large changes in the dominant class or class share are flagged in the CSV.


def sampled_classification_for_threshold(
    ds,
    fields,
    cf_thresh,
    precip_min=P_PRECIP_MIN,
    convective_fraction=CONVECTIVE_FRACTION,
    sample_limits=None,
    include_process_diagnostics=False,
):
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
        **(sample_limits or SENSITIVITY_SAMPLE_LIMITS),
        time_mode="stride",
    )
    if indexers:
        cf = cf.isel({dim: slc for dim, slc in indexers.items() if dim in cf.dims})
        precip = precip.isel({dim: slc for dim, slc in indexers.items() if dim in precip.dims})
    sampled_fields = dict(fields)
    sampled_fields["precip"] = precip
    for key in ("convective_precip", "grid_scale_precip"):
        da = fields[key].sel(Time=precip.Time)
        if indexers:
            da = da.isel({dim: slc for dim, slc in indexers.items() if dim in da.dims})
        sampled_fields[key] = da
    return classify_high_cloud(
        ds,
        cf,
        sampled_fields,
        cf_thresh=cf_thresh,
        precip_min=precip_min,
        convective_fraction=convective_fraction,
        include_process_diagnostics=include_process_diagnostics,
    )


def reclassify_convective_indicators(
    classif,
    heavy_precip=P_HEAVY,
    strong_updraft=W_STRONG,
    precip_min=P_PRECIP_MIN,
    convective_fraction=CONVECTIVE_FRACTION,
):
    """Reapply cheap 2-D thresholds using already-computed coarse fields."""
    high = classif["high_cloud_mask"].astype(bool)
    valid_high = (
        classif["convective_precipitating_high_cloud_mask"].astype(bool)
        | classif["stratiform_precipitating_high_cloud_mask"].astype(bool)
        | classif["nonprecipitating_high_cloud_mask"].astype(bool)
    )
    precipitating = classif["surface_precip_rate"] >= precip_min
    count = (
        (classif["surface_precip_rate"] >= heavy_precip).astype("int8")
        + (classif["column_max_updraft"] >= strong_updraft).astype("int8")
        + (classif["convective_precip_fraction"] >= convective_fraction).astype("int8")
    )
    convective = valid_high & precipitating & (count >= CONVECTIVE_EVIDENCE_REQUIRED)
    stratiform = valid_high & precipitating & ~convective
    nonprecipitating = valid_high & ~precipitating
    out = dict(classif)
    out.update(
        convective_precipitating_high_cloud_mask=convective,
        stratiform_precipitating_high_cloud_mask=stratiform,
        nonprecipitating_high_cloud_mask=nonprecipitating,
        precipitating_high_cloud_mask=valid_high & precipitating,
        convective_evidence_count=count,
    )
    return out


def append_sensitivity_rows(
    rows,
    sim_name,
    classif,
    parameter,
    threshold_value,
    threshold_label,
    cf_thresh,
    precip_min,
    heavy_precip,
    strong_updraft,
    convective_fraction,
):
    high_frac = weighted_area_time_mean(
        classif["high_cloud_mask"], TROPICAL_LAT_BOUNDS
    )
    for key, label, _color in BUDGET_SPECS:
        frac = weighted_area_time_mean(classif[key], TROPICAL_LAT_BOUNDS)
        rows.append(
            dict(
                simulation=sim_name,
                sensitivity_parameter=parameter,
                threshold_value=float(threshold_value),
                threshold_label=threshold_label,
                cldfrac_threshold=float(cf_thresh),
                precipitation_threshold_mm_day=float(precip_min),
                heavy_precipitation_threshold_mm_day=float(heavy_precip),
                strong_updraft_threshold_m_s=float(strong_updraft),
                convective_source_fraction=float(convective_fraction),
                convective_evidence_required=int(CONVECTIVE_EVIDENCE_REQUIRED),
                class_key=key,
                class_label=label,
                domain_fraction=float(frac),
                within_high_cloud_fraction=(
                    float(frac / high_frac) if high_frac > 0 else np.nan
                ),
                high_cloud_denominator=float(high_frac),
                denominator=(
                    "sampled tropical area-time fraction for domain_fraction; "
                    "sampled high-cloud area-time fraction for within_high_cloud_fraction"
                ),
            )
        )


def threshold_sensitivity_rows(dss, fields_by_run):
    rows = []
    for sim_name, ds, fields in zip(sim_names, dss, fields_by_run):
        baseline = None
        for cf_thresh in SENSITIVITY_CF_THRESHOLDS:
            progress(f"{sim_name}: sensitivity classification for cldfrac={cf_thresh:g}")
            classif = sampled_classification_for_threshold(
                ds,
                fields,
                cf_thresh,
            )
            if classif is None:
                continue
            if np.isclose(cf_thresh, CF_THRESH):
                baseline = classif
            append_sensitivity_rows(
                rows,
                sim_name,
                classif,
                "cldfrac_threshold",
                cf_thresh,
                f"{cf_thresh:g}",
                cf_thresh,
                P_PRECIP_MIN,
                P_HEAVY,
                W_STRONG,
                CONVECTIVE_FRACTION,
            )

        if baseline is None:
            continue
        for precip_min in SENSITIVITY_PRECIP_MIN_THRESHOLDS:
            label = f"rainc + rainnc >= {precip_min:g} mm day-1"
            progress(f"{sim_name}: precipitation-occurrence sensitivity {label}")
            trial = (
                baseline
                if np.isclose(precip_min, P_PRECIP_MIN)
                else reclassify_convective_indicators(
                    baseline, precip_min=precip_min
                )
            )
            append_sensitivity_rows(
                rows,
                sim_name,
                trial,
                "precipitation_occurrence_threshold",
                precip_min,
                label,
                CF_THRESH,
                precip_min,
                P_HEAVY,
                W_STRONG,
                CONVECTIVE_FRACTION,
            )

        for setting_index, (heavy_precip, strong_updraft) in enumerate(
            SENSITIVITY_INDICATOR_SETTINGS
        ):
            label = (
                f"rainc + rainnc >= {heavy_precip:g} mm day-1; "
                f"max(w) over levels with cldfrac >= {CF_THRESH:g} >= "
                f"{strong_updraft:g} m s-1"
            )
            progress(
                f"{sim_name}: indicator sensitivity {label}"
            )
            trial = (
                baseline
                if np.isclose(heavy_precip, P_HEAVY)
                and np.isclose(strong_updraft, W_STRONG)
                else reclassify_convective_indicators(
                    baseline, heavy_precip, strong_updraft
                )
            )
            append_sensitivity_rows(
                rows,
                sim_name,
                trial,
                "convective_indicator_thresholds",
                setting_index,
                label,
                CF_THRESH,
                P_PRECIP_MIN,
                heavy_precip,
                strong_updraft,
                CONVECTIVE_FRACTION,
            )

        for convective_fraction in SENSITIVITY_CONVECTIVE_FRACTIONS:
            label = (
                f"rainc / (rainc + rainnc) >= {convective_fraction:g}"
            )
            progress(f"{sim_name}: parameterized-rain sensitivity {label}")
            trial = (
                baseline
                if np.isclose(convective_fraction, CONVECTIVE_FRACTION)
                else reclassify_convective_indicators(
                    baseline, convective_fraction=convective_fraction
                )
            )
            append_sensitivity_rows(
                rows,
                sim_name,
                trial,
                "convective_source_fraction",
                convective_fraction,
                label,
                CF_THRESH,
                P_PRECIP_MIN,
                P_HEAVY,
                W_STRONG,
                convective_fraction,
            )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    flags = []
    science_keys = [
        "convective_precipitating_high_cloud_mask",
        "stratiform_precipitating_high_cloud_mask",
        "nonprecipitating_high_cloud_mask",
    ]
    for sim_name in sim_names:
        for parameter in df.sensitivity_parameter.unique():
            sub = df[
                (df.simulation == sim_name)
                & (df.sensitivity_parameter == parameter)
                & (df.class_key.isin(science_keys))
            ]
            if len(sub) == 0:
                continue
            strong = False
            notes = []
            for key in science_keys:
                vals = sub[sub.class_key == key]["within_high_cloud_fraction"].to_numpy(dtype=float)
                spread = float(np.nanmax(vals) - np.nanmin(vals)) if np.any(np.isfinite(vals)) else np.nan
                if np.isfinite(spread) and spread > 0.10:
                    strong = True
                    notes.append(f"{key} spread {spread:.3f}")
            dominant = (
                sub.sort_values("within_high_cloud_fraction")
                .groupby("threshold_value")
                .tail(1)[["threshold_value", "class_key"]]
            )
            if dominant["class_key"].nunique() > 1:
                strong = True
                notes.append("dominant class changes")
            flags.append(
                dict(
                    simulation=sim_name,
                    sensitivity_parameter=parameter,
                    interpretation_changes_strongly=bool(strong),
                    note="; ".join(notes) if notes else "class shares stable across tested values",
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

    class_specs = [
        ("high_cloud_mask", "Total high cloud", "0.15"),
        ("convective_precipitating_high_cloud_mask", "Convective precipitating", "tab:red"),
        ("stratiform_precipitating_high_cloud_mask", "Stratiform precipitating", "tab:blue"),
        ("nonprecipitating_high_cloud_mask", "Non-precipitating", "tab:green"),
        ("unclassified_high_cloud_mask", "Invalid", "0.55"),
    ]
    parameter_specs = [
        (
            "cldfrac_threshold",
            "Cloudy-level cldfrac",
            CF_THRESH,
        ),
        (
            "precipitation_occurrence_threshold",
            "Precipitation occurrence",
            P_PRECIP_MIN,
        ),
        (
            "convective_indicator_thresholds",
            "Joint rain and w indicators",
            1.0,
        ),
        (
            "convective_source_fraction",
            "rainc contribution",
            CONVECTIVE_FRACTION,
        ),
    ]
    fig, ax = plt.subplots(
        len(sim_names),
        len(parameter_specs),
        figsize=(18.5, max(4.2, 3.2 * len(sim_names))),
        constrained_layout=True,
        sharey=True,
        squeeze=False,
    )
    for i, sim_name in enumerate(sim_names):
        for j, (parameter, title, baseline_value) in enumerate(parameter_specs):
            a = ax[i, j]
            for class_key, label, color in class_specs:
                sub = df[
                    (df.simulation == sim_name)
                    & (df.sensitivity_parameter == parameter)
                    & (df.class_key == class_key)
                ].sort_values("threshold_value")
                if len(sub) == 0:
                    continue
                a.plot(
                    sub.threshold_value,
                    100.0 * sub.domain_fraction,
                    color=color,
                    marker="o",
                    lw=1.5,
                    label=label if i == 0 and j == 0 else None,
                )
            a.set_ylim(bottom=0.0)
            a.grid(alpha=0.22)
            a.axvline(baseline_value, color="0.4", lw=0.9, ls=":")
            if parameter == "cldfrac_threshold":
                a.set_xlabel("cldfrac threshold defining a cloudy model level")
            elif parameter == "precipitation_occurrence_threshold":
                a.set_xlabel("rainc + rainnc threshold (mm day$^{-1}$)")
            elif parameter == "convective_indicator_thresholds":
                a.set_xticks(
                    range(len(SENSITIVITY_INDICATOR_SETTINGS)),
                    [
                        f"P >= {rain:g}\nwmax >= {updraft:g}"
                        for rain, updraft in SENSITIVITY_INDICATOR_SETTINGS
                    ],
                )
                a.set_xlabel(
                    f"P = rainc + rainnc (mm day$^{{-1}}$)\n"
                    f"wmax = max(w) over cldfrac >= {CF_THRESH:g} levels (m s$^{{-1}}$)"
                )
                a.tick_params(axis="x", labelsize=8)
            else:
                a.set_xlabel("rainc / (rainc + rainnc) threshold")
            if j == 0:
                a.set_ylabel(f"{sim_name}\nTropical area-time (%)")
            if i == 0:
                a.set_title(title)
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Tropical area-time occurrence under the tested classification thresholds")
    save_figure(6, fig, "cloud_classification_threshold_sensitivity")
    plt.close(fig)
    progress("Done with figure: cloud_classification_threshold_sensitivity")


# -----------------------------------------------------------------------------
# Figure 7. All-time lat/lon classification validation movies
# -----------------------------------------------------------------------------


def plot_classification_validation_movies(fields_by_run, classifications):
    if not should_run_figure(7):
        return
    progress("Starting Figure 7: cloud_classification_validation_movies")
    precip_cmap, precip_norm = precip_rate_cmap_norm()
    updraft_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=0.4)
    height_norm = colors.Normalize(
        vmin=CLOUD_TOP_HEIGHT_MAP_MIN_KM,
        vmax=CLOUD_TOP_HEIGHT_MAP_MAX_KM,
    )
    class_cmap, class_norm = class_code_cmap_norm(alpha=0.92)
    class_legend = class_legend_handles()
    manifest_rows = []

    for i, sim_name in enumerate(sim_names):
        classif = classifications.get(sim_name)
        if classif is None:
            raise RuntimeError(
                f"{sim_name}: Figure 7 requires the shared classification product"
            )
        component_precip = (
            fields_by_run[i]["convective_precip"]
            + fields_by_run[i]["grid_scale_precip"]
        )
        classification_time = classif["high_cloud_mask"].Time
        common_times = classification_time.to_index().intersection(
            component_precip.Time.to_index()
        )
        if len(common_times) == 0:
            raise RuntimeError(
                f"{sim_name}: classification and precipitation have no common times"
            )

        fig, axes = plt.subplots(2, 2, figsize=(16.0, 8.8), constrained_layout=False)
        fig.subplots_adjust(
            left=0.06,
            right=0.94,
            bottom=0.15,
            top=0.88,
            wspace=0.24,
            hspace=0.30,
        )
        fig.colorbar(
            plt.cm.ScalarMappable(norm=precip_norm, cmap=precip_cmap),
            ax=axes[0, 0],
            label="rainc + rainnc (mm day$^{-1}$)",
            fraction=0.046,
            pad=0.025,
        )
        fig.colorbar(
            plt.cm.ScalarMappable(norm=updraft_norm, cmap="RdBu_r"),
            ax=axes[0, 1],
            label=f"max(w) over levels with cldfrac >= {CF_THRESH:g} (m s$^{{-1}}$)",
            fraction=0.046,
            pad=0.025,
        )
        fig.colorbar(
            plt.cm.ScalarMappable(norm=height_norm, cmap="magma"),
            ax=axes[1, 0],
            label="Cloud-top height (km)",
            fraction=0.046,
            pad=0.025,
        )
        fig.legend(
            handles=class_legend,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=len(CLASS_CODE_SPECS),
            frameon=False,
            title="High-cloud column class (tint in top row)",
        )

        def update(frame_index):
            time_value = common_times[frame_index]
            for ax in axes.ravel():
                ax.clear()
            precip = subset_lat(
                component_precip.sel(Time=time_value), TROPICAL_LAT_BOUNDS
            ).load()
            frame = xr.Dataset(
                {
                    key: subset_lat(classif[key].sel(Time=time_value), TROPICAL_LAT_BOUNDS)
                    for key in (
                        "high_cloud_mask",
                        "cloud_top_height",
                        "column_max_updraft",
                        "convective_precipitating_high_cloud_mask",
                        "stratiform_precipitating_high_cloud_mask",
                        "nonprecipitating_high_cloud_mask",
                        "unclassified_high_cloud_mask",
                    )
                }
            ).load()
            high = frame["high_cloud_mask"].astype(float)
            class_code = class_code_from_masks(
                high,
                frame["convective_precipitating_high_cloud_mask"],
                frame["stratiform_precipitating_high_cloud_mask"],
                frame["nonprecipitating_high_cloud_mask"],
                frame["unclassified_high_cloud_mask"],
            )
            lon = np.asarray(precip.longitude.values, dtype=float)
            lat = np.asarray(precip.latitude.values, dtype=float)

            axes[0, 0].pcolormesh(
                lon,
                lat,
                precip.clip(min=0.0),
                cmap=precip_cmap,
                norm=precip_norm,
                shading="auto",
                rasterized=True,
            )
            overlay_class_tint(axes[0, 0], class_code)
            axes[0, 0].set_title("P = rainc + rainnc; high-cloud class tint")

            axes[0, 1].pcolormesh(
                lon,
                lat,
                frame["column_max_updraft"],
                cmap="RdBu_r",
                norm=updraft_norm,
                shading="auto",
                rasterized=True,
            )
            overlay_class_tint(axes[0, 1], class_code)
            axes[0, 1].set_title(
                f"max(w) over cldfrac >= {CF_THRESH:g} levels; high-cloud class tint"
            )

            cloud_top_height = frame["cloud_top_height"]
            axes[1, 0].set_facecolor("0.96")
            axes[1, 0].pcolormesh(
                lon,
                lat,
                cloud_top_height,
                cmap="magma",
                norm=height_norm,
                shading="auto",
                rasterized=True,
            )
            overlay_high_cloud_tint(axes[1, 0], high)
            axes[1, 0].set_title("Cloud-top height; cyan tint = high-cloud mask")

            class_values = np.ma.masked_where(
                np.asarray(class_code.values) <= 0,
                np.asarray(class_code.values),
            )
            axes[1, 1].set_facecolor("0.96")
            axes[1, 1].pcolormesh(
                lon,
                lat,
                class_values,
                cmap=class_cmap,
                norm=class_norm,
                shading="auto",
                rasterized=True,
            )
            axes[1, 1].set_title("High-cloud column class")

            for ax in axes.ravel():
                ax.set_xlim(float(lon.min()), float(lon.max()))
                ax.set_ylim(*TROPICAL_LAT_BOUNDS)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
            fig.suptitle(
                f"{sim_name} | {format_time_value(time_value)}\n"
                f"{classification_threshold_summary()}"
            )
            if frame_index % 25 == 0 or frame_index == len(common_times) - 1:
                progress(
                    f"{sim_name}: rendering movie frame {frame_index + 1}/{len(common_times)}"
                )
            return []

        movie = animation.FuncAnimation(
            fig,
            update,
            frames=len(common_times),
            interval=1000.0 / VALIDATION_MOVIE_FPS,
            blit=False,
        )
        safe_sim = sim_name.replace(" ", "_")
        output_path = os.path.join(
            save_loc, f"7_cloud_classification_validation_{safe_sim}.mp4"
        )
        writer = animation.FFMpegWriter(
            fps=VALIDATION_MOVIE_FPS,
            codec="libx264",
            bitrate=2200,
            extra_args=[
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
            ],
            metadata={"title": f"High-cloud classification validation: {sim_name}"},
        )
        progress(f"Writing classification validation movie: {output_path}")
        movie.save(output_path, writer=writer, dpi=VALIDATION_MOVIE_DPI)
        plt.close(fig)
        manifest_rows.append(
            dict(
                simulation=sim_name,
                output_file=output_path,
                frame_count=len(common_times),
                first_time=format_time_value(common_times[0]),
                last_time=format_time_value(common_times[-1]),
                frame_interval_hours=dt,
                fps=VALIDATION_MOVIE_FPS,
                precipitation_threshold_mm_day=P_HEAVY,
                max_w_threshold_m_s=W_STRONG,
                cldfrac_threshold=CF_THRESH,
            )
        )

    save_table(
        pd.DataFrame(manifest_rows), "cloud_classification_validation_movie_manifest"
    )
    progress("Done with Figure 7: cloud_classification_validation_movies")


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
    return ds.sel(Time=selected), selected


def subset_zonal_mean_test_window(zds, selected_times, sim_name):
    if "Time" not in zds.dims:
        return zds
    available = zds.Time.to_index()
    selected = pd.Index(selected_times)
    common = selected.intersection(available)
    if len(common) == 0:
        warnings.warn(f"{sim_name}: no zonal-mean times overlap the test window")
        return zds.isel(Time=slice(0, 0))
    return zds.sel(Time=common)


def load_datasets():
    dss = []
    dss_zm = []
    progress("Loading lat-lon and zonal-mean datasets")
    for i, d in enumerate(data_locs):
        selected_test_times = None
        progress(f"{sim_names[i]}: opening lat-lon dataset")
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
            ds, selected_test_times = test_time_window(ds, sim_names[i])
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
        )
        zds = standardize_latlon_dataset(zds, needed_vars)
        if TEST_MODE:
            zds = subset_zonal_mean_test_window(
                zds, selected_test_times, sim_names[i]
            )
        progress(f"{sim_names[i]}: zonal-mean dataset ready with {len(zds.data_vars)} variables")
        dss.append(ds)
        dss_zm.append(zds)
    return dss, dss_zm


def prepare_fields(dss, dss_zm):
    fields_by_run = []
    progress("Preparing precipitation, thermodynamic, and TTL fields")
    for sim_name, ds, zds in zip(sim_names, dss, dss_zm):
        progress(f"{sim_name}: preparing fields")
        precip = get_precip_rate(ds)
        convective_precip = precip_component_rate_from_dataset(
            ds, "convective", dt_hours=dt
        )
        grid_scale_precip = precip_component_rate_from_dataset(
            ds, "grid_scale", dt_hours=dt
        )
        common_time = precip.Time.to_index()
        common_time = common_time.intersection(convective_precip.Time.to_index())
        common_time = common_time.intersection(grid_scale_precip.Time.to_index())
        if len(common_time) == 0:
            raise RuntimeError(f"{sim_name}: precipitation components have no common times")
        fields = {
            "precip": precip.sel(Time=common_time),
            "convective_precip": convective_precip.sel(Time=common_time),
            "grid_scale_precip": grid_scale_precip.sel(Time=common_time),
        }
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
    if not should_run_any([1, 2, 3, 4, 5, 6, 7]):
        return classifications
    if WRITE_CLASSIFICATION_PRODUCT:
        invalidate_classification_products()
    progress("Building high-cloud classification products")
    for sim_name, ds, fields in zip(sim_names, dss, fields_by_run):
        classifications[sim_name] = get_or_build_classification(ds, fields, sim_name)
    if should_run_any([1, 2, 4, 5, 6, 7]):
        class_overlap_sanity(classifications)
    progress("High-cloud classification stage complete")
    return classifications


def main(argv=None):
    global figsave
    args = parse_command_line(argv)
    configure_run(args)
    os.makedirs(save_loc, exist_ok=True)
    os.makedirs(classification_cache_dir, exist_ok=True)
    figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)
    if TEST_MODE:
        progress(
            f"TEST MODE: simulation={sim_names[0]}, classified_times={TEST_TIME_STEPS}, "
            f"write_test_product={WRITE_CLASSIFICATION_PRODUCT}, output={save_loc}"
        )
    elif not WRITE_CLASSIFICATION_PRODUCT:
        progress("Classification products will be preserved and reused after version/schema validation")
    write_classification_method_table()

    dss, dss_zm = load_datasets()
    fields_by_run = prepare_fields(dss, dss_zm)
    classifications = prepare_classifications(dss, fields_by_run)

    plot_class_budget(classifications)
    plot_residual_diagnostics(fields_by_run, classifications)
    plot_validation_snapshots(dss, fields_by_run, classifications)
    process_frame = (
        build_process_validation_frame(dss, fields_by_run)
        if should_run_any([4, 5])
        else pd.DataFrame()
    )
    if len(process_frame) > 0:
        save_table(
            threshold_percentile_table(process_frame),
            "cloud_classification_threshold_percentile_ranks",
        )
    plot_regime_process_validation(process_frame)
    plot_precip_component_validation(process_frame, classifications)
    plot_sensitivity_diagnostics(fields_by_run, classifications, dss=dss)
    plot_classification_validation_movies(fields_by_run, classifications)

    progress("Refined high-cloud classification diagnostics complete")


if __name__ == "__main__":
    main()
