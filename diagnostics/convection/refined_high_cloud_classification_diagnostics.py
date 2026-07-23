#!/usr/bin/env python
# coding: utf-8

"""
Refined high-cloud classification diagnostics for MPAS-A nudging experiments.

This script owns the shared high-cloud/PyFLEXTRKR scene product. Its primary
figures report class occurrence/composition, cloud-top-pressure distributions,
TTL-reaching frequency, domain-contribution vertical cloud profiles,
class-conditioned OLR, spatial occurrence responses, and rain-matched cloud
depth/OLR responses. Core/cold-shield/warm-shield scene regions and vertical
slices provide the validation view behind ``--qc``.
"""

import argparse
import gc
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
from matplotlib import colors, pyplot as plt
from matplotlib.lines import Line2D
from scipy import ndimage

from diagnostics.diagnostic_helpers import (
    FigureSaver, HIGH_CLOUD_CLASS_SPECS, LATLON_PATTERN,
    QUICK_TEST_LAT_BOUNDS, QUICK_TEST_LON_BOUNDS,
    diagnostic_cache_dir, diagnostic_output_dir, figure_should_run,
    high_cloud_class_cmap_norm, high_cloud_class_legend_handles,
    load_latlon_dataset, precip_rate_cmap_norm,
    precipitation_rate_from_dataset, progress,
    sample_matching_arrays, standardize_latlon_dataset,
    subset_latitude, time_block_slices, weighted_area_time_mean,
)
from diagnostics.convection.high_cloud_classification import (
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
#   1: high_cloud_class_budget
#   2: literature_high_cloud_structure (CTP, TTL, OLR, vertical profiles)
#   3: cloud_classification_validation_snapshots
#   4: high_cloud_vertical_contribution_profiles
#   5: high_cloud_class_spatial_response
#   6: rain_matched_high_cloud_structure_response
figures_to_run = [1, 2, 4, 5, 6]
figsave = None

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control_new", "u_nudge", "T_nudge"]
sim_names = ["control", "nudged u", "nudged T"]
cols = ["k", "b", "r"]

dt = 6

TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
SNAPSHOT_LAT_BOUNDS = (-20.0, 20.0)

# Cloud-top pressure avoids tying tropical high cloud to one geometric height.
# The 440 hPa parent follows the standard ISCCP high-cloud boundary. Continuous
# cloud-top pressure is retained for the UTLS cloud-deepening analysis.
CF_THRESH = 0.05
P_HIGH_HPA = 440.0
P_PRECIP_MIN = 12.0  # 0.5 mm h-1 PF threshold (Feng et al. 2021)
WRITE_CLASSIFICATION_PRODUCT = True
classification_cache_dir = diagnostic_cache_dir("classification_cache", root_dir=root_dir)

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
UTLS_LAYER_BOUNDS_KM = (14.0, 18.0)
latlon_chunks = {"Time": 8, "latitude": 90, "longitude": 180}
dask.config.set({"array.slicing.split_large_chunks": True})

latlon_vars = [
    "rainc", "rainnc", "i_rainc", "i_rainnc",
    "olrtoa", "cldfrac", "pressure", "t_isobaric", "zgrid",
]

BUDGET_SPECS = [
    ("high_cloud_mask", "Total high cloud", "0.25"),
    ("convective_precipitating_high_cloud_mask", "PyFLEXTRKR convective core", "tab:red"),
    ("stratiform_precipitating_high_cloud_mask", "Stratiform precipitating", "tab:blue"),
    ("nonprecipitating_high_cloud_mask", "Non-precipitating", "tab:green"),
]

SPATIAL_CLASS_SPECS = BUDGET_SPECS
RAIN_CONDITIONED_CLASS_SPECS = BUDGET_SPECS

REGIME_SPECS = [
    ("convective_precipitating_high_cloud_mask", "PyFLEXTRKR convective core", "tab:red"),
    ("stratiform_precipitating_high_cloud_mask", "Stratiform precipitating", "tab:blue"),
    ("nonprecipitating_high_cloud_mask", "Non-precipitating", "tab:green"),
]

PRECIPITATING_COMPOSITION_SPECS = [
    ("rainy_pyflextrkr_core_mask", "Rainy PyFLEXTRKR core", "tab:red"),
    ("stratiform_precipitating_high_cloud_mask", "Stratiform precipitating", "tab:blue"),
]

CIRRUS_COMPARISON_SPECS = [
    ("noncore_high_cloud_aggregate", "Non-core aggregate", "tab:purple"),
    ("nonprecipitating_high_cloud_mask", "Non-precipitating", "tab:green"),
]

NONPRECIP_SHIELD_SPECS = [
    (
        "nonprecip_within_core_bearing_warm_shield",
        "Within core-bearing 261-K shield",
        "tab:orange",
    ),
    (
        "nonprecip_outside_core_bearing_warm_shield",
        "Outside core-bearing 261-K shield",
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
        cloud_type="deep convective",
        value=223.3,
        units="hPa",
        definition="MODIS ISCCP deep-convective high-cloud optical class",
        region="30S-30N",
        reference="Hong et al. (2007), Journal of Applied Meteorology and Climatology",
        doi="10.1175/2007JAMC1583.1",
    ),
    dict(
        metric="mean_cloud_top_temperature",
        cloud_type="cirriform",
        value=231.0,
        units="K",
        definition="MODIS ISCCP cirriform high-cloud optical classes",
        region="30S-30N",
        reference="Hong et al. (2007), Journal of Applied Meteorology and Climatology",
        doi="10.1175/2007JAMC1583.1",
    ),
    dict(
        metric="mean_cloud_top_temperature",
        cloud_type="deep convective",
        value=234.9,
        units="K",
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
        "--qc",
        action="store_true",
        help="Add the classification map and vertical-slice validation (Figure 3).",
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
    parser.set_defaults(write_classification=True)
    args = parser.parse_args(argv)
    if args.test_times < 1:
        parser.error("--test-times must be at least 1")
    return args


def configure_run(args):
    global TEST_MODE, TEST_TIME_STEPS, TEST_START
    global WRITE_CLASSIFICATION_PRODUCT, classification_cache_dir
    global save_loc, figures_to_run
    global data_locs, sim_names, cols

    TEST_MODE = bool(args.test)
    TEST_TIME_STEPS = int(args.test_times)
    TEST_START = args.test_start
    WRITE_CLASSIFICATION_PRODUCT = bool(args.write_classification)

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
    test_label = f"{args.simulation}{start_label}_{TEST_TIME_STEPS:03d}times"
    test_label = test_label.replace(":", "").replace(" ", "_")
    save_loc = os.path.join(default_save_loc, "testing", test_label)
    # Test products are deliberately isolated from products consumed downstream.
    classification_cache_dir = os.path.join(save_loc, "classification_products")


def save_figure(number, fig, stem):
    if figsave is None:
        raise RuntimeError("FigureSaver has not been initialised")
    return figsave.save(fig, stem, number=number)


def save_table(df, stem):
    if save and df is not None and len(df) > 0:
        path = os.path.join(save_loc, f"{stem}.csv")
        df.to_csv(path, index=False)
        progress(f"Wrote {path}")


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


def diagnosed_cloud_top_temperature(ds, classif):
    """Interpolate model isobaric temperature to diagnosed cloud-top pressure."""
    if "t_isobaric" not in ds:
        raise KeyError("t_isobaric is required for cloud-top temperature")
    temperature = ds["t_isobaric"]
    if "pres" not in temperature.dims:
        raise ValueError(
            "t_isobaric must have a 'pres' dimension after lat-lon standardisation"
        )
    temperature, cloud_top_pressure = xr.align(
        temperature,
        classif["cloud_top_pressure"],
        join="inner",
    )
    # xarray requires a monotonic interpolation coordinate.  The source files
    # currently store pressure from the top downward, but sorting makes this
    # robust to either ordering.
    temperature = temperature.sortby("pres")
    out = temperature.interp(pres=cloud_top_pressure)
    out.name = "diagnosed_cloud_top_temperature"
    out.attrs.update(
        units="K",
        long_name="model isobaric temperature at diagnosed cloud-top pressure",
        method=(
            "linear interpolation of t_isobaric to the pressure of the highest "
            f"model layer with cloud fraction at or above {CF_THRESH:g}"
        ),
    )
    return out


def full_class_structure_statistics(
    ds,
    fields,
    classif,
    sim_name,
    ctp_bins=None,
    time_batch=CLOUD_STRUCTURE_TIME_BATCH,
):
    """Accumulate exact tropical class statistics in memory-bounded time blocks."""
    class_keys = [key for key, _label, _color in REGIME_SPECS]
    aligned = xr.align(
        subset_lat(classif["cloud_top_pressure"], TROPICAL_LAT_BOUNDS),
        subset_lat(fields["olr"], TROPICAL_LAT_BOUNDS),
        subset_lat(classif["brightness_temperature"], TROPICAL_LAT_BOUNDS),
        *[
            subset_lat(classif[key].astype(bool), TROPICAL_LAT_BOUNDS)
            for key in class_keys
        ],
        join="inner",
    )
    cloud_top_pressure, olr, brightness_temperature = aligned[:3]
    class_masks = dict(zip(class_keys, aligned[3:]))
    tropical_ds = subset_lat(ds, TROPICAL_LAT_BOUNDS).sel(
        latitude=cloud_top_pressure.latitude,
        longitude=cloud_top_pressure.longitude,
    )

    field_names = (
        "cloud_top_pressure",
        "olr",
        "brightness_temperature",
        "cloud_top_temperature",
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
        f"{sim_name}: computing full-period cloud structure in "
        f"{len(time_slices)} blocks of at most {time_batch} times"
    )

    for iblock, time_slice in enumerate(time_slices, start=1):
        block_reference = cloud_top_pressure.isel(Time=time_slice)
        block_ds = tropical_ds.sel(Time=block_reference.Time)
        cloud_top_temperature = diagnosed_cloud_top_temperature(
            block_ds,
            {"cloud_top_pressure": block_reference},
        ).load()
        block = xr.Dataset(
            {
                "cloud_top_pressure": block_reference,
                "olr": olr.isel(Time=time_slice),
                "brightness_temperature": brightness_temperature.isel(Time=time_slice),
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
            "brightness_temperature": np.asarray(
                block["brightness_temperature"].values, dtype=float
            ),
            "cloud_top_temperature": np.asarray(
                cloud_top_temperature.values, dtype=float
            ),
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

        del block, block_values, cloud_top_temperature
        elapsed = time.monotonic() - started
        eta = elapsed / iblock * (len(time_slices) - iblock)
        progress(
            f"{sim_name}: cloud structure block {iblock}/{len(time_slices)} "
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
        histogram = accumulator["ctp_histogram"]
        if histogram is not None:
            result["ctp_histogram"] = (
                histogram / histogram.sum()
                if histogram.sum() > 0.0 else histogram
            )
        statistics[key] = result
    progress(f"{sim_name}: full-period cloud-structure statistics ready")
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


def plot_class_budget(classifications, fields_by_run):
    if not should_run_figure(1):
        return
    progress("Starting Figure 1: high_cloud_class_budget")
    budget = compute_budget_table(classifications, fields_by_run)
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
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(19.0, 9.0),
        constrained_layout=True,
        squeeze=False,
    )
    (
        occurrence_ax,
        composition_ax,
        precip_ax,
        difference_ax,
        cirrus_comparison_ax,
        shield_association_ax,
    ) = axes.ravel()
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
        label="CIRS all UT cloud: ~35% (30S-30N; CTP < 440 hPa)",
    )
    occurrence_ax.annotate(
        "35", (35.0, occurrence_y[0]), xytext=(0, 8),
        textcoords="offset points", ha="center", va="bottom",
        fontsize=6.5, color="#7a4e00",
    )
    occurrence_ax.scatter(
        [22.7], [occurrence_y[0]], marker="v", s=54,
        facecolor="white", edgecolor="#375a7f", linewidth=1.3, zorder=7,
        label="MODIS daytime high ice cloud: 22.7% (30S-30N)",
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
        label="CALIOP cirrus occurrence: 38-44% (15S-15N)",
    )
    occurrence_ax.scatter(
        [38.4, 43.5], [nonprecip_occurrence_y, nonprecip_occurrence_y],
        marker="|", s=70, color="#7a4e00", linewidth=1.1, zorder=8,
    )
    precip_ax.scatter(
        [27.0, 73.0], precip_y, marker="D", s=52,
        facecolor="white", edgecolor="#7a4e00", linewidth=1.4, zorder=7,
        label="TRMM PR raining area: 27% convective / 73% stratiform (20S-20N)",
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
        label="MODIS high-cloud optical classes: cirriform >80% (30S-30N)",
    )
    cirrus_comparison_ax.annotate(
        ">80", (80.0, cirrus_comparison_y[0]), xytext=(0, 8),
        textcoords="offset points", ha="center", va="bottom",
        fontsize=6.5, color="#7a4e00",
    )

    occurrence_ax.set_yticks(occurrence_y)
    occurrence_ax.set_yticklabels(occurrence_labels)
    occurrence_ax.invert_yaxis()
    occurrence_ax.set_xlabel("Tropical area-time (%)")
    occurrence_ax.set_title("(a) Domain occurrence")
    occurrence_ax.grid(axis="x", alpha=0.22)
    occurrence_ax.legend(frameon=False, fontsize=7)
    occurrence_ax.margins(x=0.14)

    composition_ax.set_yticks(composition_y)
    composition_ax.set_yticklabels(composition_labels)
    composition_ax.invert_yaxis()
    composition_ax.set_xlabel("Share of all diagnosed high cloud (%)")
    composition_ax.set_title("(b) High-cloud composition")
    composition_ax.grid(axis="x", alpha=0.22)
    composition_ax.legend(frameon=False, fontsize=7)
    composition_ax.margins(x=0.14)

    precip_ax.set_yticks(precip_y)
    precip_ax.set_yticklabels(precip_labels)
    precip_ax.invert_yaxis()
    precip_ax.set_xlim(0.0, 108.0)
    precip_ax.set_xlabel("Share of precipitating high-cloud area (%)")
    precip_ax.set_title("(c) Rainy-core/stratiform validation")
    precip_ax.grid(axis="x", alpha=0.22)
    precip_ax.legend(frameon=False, fontsize=7)

    if has_differences:
        difference_ax.set_yticks(occurrence_y)
        difference_ax.set_yticklabels(occurrence_labels)
        difference_ax.invert_yaxis()
        difference_ax.axvline(0.0, color="0.45", lw=0.8)
        difference_ax.set_xlabel("Simulation minus control (percentage points)")
        difference_ax.set_title("(d) Domain-occurrence response")
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
    cirrus_comparison_ax.set_yticklabels(cirrus_comparison_labels)
    cirrus_comparison_ax.invert_yaxis()
    cirrus_comparison_ax.set_xlim(0.0, 108.0)
    cirrus_comparison_ax.set_xlabel("Share of all diagnosed high cloud (%)")
    cirrus_comparison_ax.set_title("(e) Cirrus comparison")
    cirrus_comparison_ax.grid(axis="x", alpha=0.22)
    cirrus_comparison_ax.legend(frameon=False, fontsize=7)

    shield_association_ax.set_yticks(shield_association_y)
    shield_association_ax.set_yticklabels(shield_association_labels)
    shield_association_ax.invert_yaxis()
    shield_association_ax.set_xlim(0.0, 108.0)
    shield_association_ax.set_xlabel("Share of non-precipitating high cloud (%)")
    shield_association_ax.set_title("(f) Non-precipitating shield association")
    shield_association_ax.grid(axis="x", alpha=0.22)
    if len(shield_association_df) > 0:
        shield_association_ax.legend(frameon=False, fontsize=7)
    else:
        shield_association_ax.text(
            0.5, 0.5, "Shield association unavailable",
            transform=shield_association_ax.transAxes,
            ha="center", va="center", color="0.35", fontsize=9,
        )

    fig.suptitle(
        "Cloud-radiation-on high-cloud occurrence and composition\n"
        "Model bars use 20S-20N. Panel (b) uses all high cloud; panel (c) uses only "
        f"high-cloud columns with rain >= {P_PRECIP_MIN / 24.0:g} mm h$^{{-1}}$. "
        "Literature symbols use the denominators stated in their legends.",
        fontsize=11,
    )
    save_figure(1, fig, "high_cloud_class_budget")
    plt.close(fig)
    progress("Done with figure: high_cloud_class_budget")


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
        f"High cloud: cldfrac >= {CF_THRESH:g}, CTP <= {P_HIGH_HPA:g} hPa\n"
        "Core: PyFLEXTRKR Tb < 225 K; "
        f"stratiform: other high cloud with rain >= {P_PRECIP_MIN / 24.0:g} mm h-1; "
        "non-precipitating: leftover\n"
        "Cold-core, cold-shield-shell, and warm-shield-shell regions are shown for PyFLEXTRKR validation and do not alter the three cloud classes"
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
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 8.8), constrained_layout=False)
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
        label="rainc + rainnc (mm day$^{-1}$)",
        fraction=0.046,
        pad=0.025,
    )
    fig.colorbar(
        hierarchy_im, ax=axes[0, 1], ticks=[1, 2, 3],
        label="1 warm-shield shell; 2 cold-shield shell; 3 cold core",
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
        title="High-cloud column class (tint in top row)",
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
        ds.sel(Time=selected), lat_bounds=QUICK_TEST_LAT_BOUNDS
    ).sel(longitude=slice(*QUICK_TEST_LON_BOUNDS))
    progress(
        f"{sim_name}: test domain is {QUICK_TEST_LAT_BOUNDS[0]:g} to "
        f"{QUICK_TEST_LAT_BOUNDS[1]:g} deg latitude and "
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
    class_overlap_sanity(classifications)
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


def compute_vertical_contribution_profiles(
    dss,
    classifications,
    time_batch=VERTICAL_PROFILE_TIME_BATCH,
):
    """Domain-mean cldfrac contributed by columns in each exclusive class."""
    profiles = {}
    rows = []
    summary_rows = []
    class_keys = [key for key, _label, _color in REGIME_SPECS]
    class_labels = {key: label for key, label, _color in REGIME_SPECS}

    progress("Computing domain-contribution vertical cloud profiles")
    for ds, sim_name in zip(dss, sim_names):
        classif = classifications.get(sim_name)
        if classif is None:
            continue
        cf = standardize_cloud_fraction(ds.cldfrac)
        cf = subset_lat(cf, TROPICAL_LAT_BOUNDS).sel(
            z=slice(1000.0 * PROFILE_Z_BOUNDS_KM[0], 1000.0 * PROFILE_Z_BOUNDS_KM[1])
        )
        masks = [
            subset_lat(classif[key].astype(bool), TROPICAL_LAT_BOUNDS)
            for key in class_keys
        ]
        aligned = xr.align(cf, *masks, join="inner")
        cf_aligned, mask_aligned = aligned[0], aligned[1:]
        cf_aligned = cf_aligned.transpose("Time", "z", "latitude", "longitude")
        mask_aligned = [
            mask.transpose("Time", "latitude", "longitude")
            for mask in mask_aligned
        ]

        ntime = cf_aligned.sizes["Time"]
        time_slices = [
            slice(start, min(start + time_batch, ntime))
            for start in range(0, ntime, time_batch)
        ]
        numerators = np.zeros((len(class_keys), cf_aligned.sizes["z"]), dtype=float)
        denominators = np.zeros_like(numerators)
        latitude_weights = np.cos(
            np.deg2rad(np.asarray(cf_aligned.latitude.values, dtype=float))
        )[None, None, :, None]
        started = time.monotonic()
        progress(
            f"{sim_name}: computing vertical cloud profiles in "
            f"{len(time_slices)} blocks of at most {time_batch} times"
        )

        for iblock, time_slice in enumerate(time_slices, start=1):
            block = xr.Dataset(
                {
                    "cloud_fraction": cf_aligned.isel(Time=time_slice),
                    **{
                        key: mask.isel(Time=time_slice)
                        for key, mask in zip(class_keys, mask_aligned)
                    },
                }
            ).load()
            cloud_fraction = np.asarray(block["cloud_fraction"].values)
            sample_weights = np.broadcast_to(latitude_weights, cloud_fraction.shape)

            # This exactly reproduces cf.where(mask, other=0).weighted(...).mean():
            # samples outside the class contribute zero to the numerator and
            # remain in the all-column denominator, while missing cloud fraction
            # inside the selected class is excluded from both.
            for iclass, key in enumerate(class_keys):
                mask = np.asarray(block[key].values, dtype=bool)[:, None, :, :]
                contribution = np.where(mask, cloud_fraction, 0.0)
                finite = np.isfinite(contribution)
                numerators[iclass] += np.sum(
                    np.where(finite, contribution * sample_weights, 0.0),
                    axis=(0, 2, 3),
                    dtype=np.float64,
                )
                denominators[iclass] += np.sum(
                    np.where(finite, sample_weights, 0.0),
                    axis=(0, 2, 3),
                    dtype=np.float64,
                )
                del contribution, finite, mask

            del block, cloud_fraction, sample_weights
            elapsed = time.monotonic() - started
            eta = elapsed / iblock * (len(time_slices) - iblock)
            progress(
                f"{sim_name}: vertical-profile block {iblock}/{len(time_slices)} "
                f"({100.0 * iblock / len(time_slices):.0f}%); "
                f"elapsed {format_duration(elapsed)}, ETA {format_duration(eta)}"
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
        profile.name = "domain_mean_cldfrac_contribution"
        profile.attrs.update(
            long_name="domain-mean cloud fraction contributed by columns in each high-cloud class",
            units="1",
            denominator="all tropical grid-cell/time samples",
            data_scope="cloud-radiation-on simulations only",
        )
        profiles[sim_name] = profile

        for key in class_keys:
            class_profile = profile.sel(class_key=key)
            for z_value, value in zip(class_profile.z.values, class_profile.values):
                rows.append(
                    dict(
                        simulation=sim_name,
                        statistic="absolute_profile",
                        reference_simulation="none",
                        class_key=key,
                        cloud_class=class_labels[key],
                        height_km=float(z_value) / 1000.0,
                        domain_mean_cldfrac_contribution=float(value),
                        domain_mean_cldfrac_contribution_percent=100.0 * float(value),
                        denominator="all tropical grid-cell/time samples",
                        data_scope="crf_on only",
                    )
                )
            layer_mean = layer_mean_from_profile(class_profile)
            summary_rows.append(
                dict(
                    simulation=sim_name,
                    statistic="absolute_layer_mean",
                    reference_simulation="none",
                    class_key=key,
                    cloud_class=class_labels[key],
                    layer_bottom_km=UTLS_LAYER_BOUNDS_KM[0],
                    layer_top_km=UTLS_LAYER_BOUNDS_KM[1],
                    domain_mean_cldfrac_contribution=layer_mean,
                    domain_mean_cldfrac_contribution_percent=100.0 * layer_mean,
                    denominator="all tropical grid-cell/time samples and geometric layer depth",
                    data_scope="crf_on only",
                )
            )

    if sim_names and sim_names[0] in profiles:
        control_name = sim_names[0]
        control = profiles[control_name]
        for sim_name in sim_names[1:]:
            if sim_name not in profiles:
                continue
            response = profiles[sim_name] - control
            for key in class_keys:
                class_response = response.sel(class_key=key)
                for z_value, value in zip(class_response.z.values, class_response.values):
                    rows.append(
                        dict(
                            simulation=sim_name,
                            statistic="difference_from_control",
                            reference_simulation=control_name,
                            class_key=key,
                            cloud_class=class_labels[key],
                            height_km=float(z_value) / 1000.0,
                            domain_mean_cldfrac_contribution=float(value),
                            domain_mean_cldfrac_contribution_percent=100.0 * float(value),
                            denominator="all tropical grid-cell/time samples",
                            data_scope="crf_on only",
                        )
                    )
                layer_mean = layer_mean_from_profile(class_response)
                summary_rows.append(
                    dict(
                        simulation=sim_name,
                        statistic="difference_from_control",
                        reference_simulation=control_name,
                        class_key=key,
                        cloud_class=class_labels[key],
                        layer_bottom_km=UTLS_LAYER_BOUNDS_KM[0],
                        layer_top_km=UTLS_LAYER_BOUNDS_KM[1],
                        domain_mean_cldfrac_contribution=layer_mean,
                        domain_mean_cldfrac_contribution_percent=100.0 * layer_mean,
                        denominator="all tropical grid-cell/time samples and geometric layer depth",
                        data_scope="crf_on only",
                    )
                )

    save_table(pd.DataFrame(rows), "high_cloud_vertical_contribution_profiles")
    save_table(pd.DataFrame(summary_rows), "utls_high_cloud_contribution_summary")
    progress("Domain-contribution vertical cloud profiles ready")
    return profiles


def plot_vertical_contribution_profiles(profiles):
    if not should_run_figure(4):
        return
    progress("Starting Figure 4: high_cloud_vertical_contribution_profiles")
    if not profiles:
        print("Skipping Figure 4: vertical profiles are unavailable.")
        return

    fig, axes = plt.subplots(
        2, len(REGIME_SPECS), figsize=(13.2, 8.0), constrained_layout=True,
        sharey=True, squeeze=False,
    )
    control_name = sim_names[0]
    for j, (key, label, class_color) in enumerate(REGIME_SPECS):
        ax_absolute = axes[0, j]
        ax_response = axes[1, j]
        for sim_name, color in zip(sim_names, cols):
            if sim_name not in profiles:
                continue
            profile = profiles[sim_name].sel(class_key=key)
            height_km = np.asarray(profile.z.values, dtype=float) / 1000.0
            ax_absolute.plot(
                100.0 * profile.values, height_km,
                color=color, lw=1.8, label=sim_name,
            )
            if sim_name != control_name and control_name in profiles:
                response = profile - profiles[control_name].sel(class_key=key)
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
        ax_absolute.set_title(label, color=class_color)
        ax_response.axvline(0.0, color="0.45", lw=0.8)
        if len(sim_names) == 1:
            ax_response.text(
                0.5, 0.5, "Single-simulation test:\nno control difference",
                transform=ax_response.transAxes, ha="center", va="center", fontsize=8,
            )

    axes[0, 0].set_ylabel("Height (km)")
    axes[1, 0].set_ylabel("Height (km)")
    for ax in axes[0, :]:
        ax.set_xlabel("Domain contribution (%)")
    for ax in axes[1, :]:
        ax.set_xlabel("Simulation minus control (percentage points)")
    axes[0, -1].legend(frameon=False, fontsize=8)
    if len(sim_names) > 1:
        axes[1, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Cloud-radiation-on simulations: vertical cloud amount from each high-cloud column class\n"
        "Unfiltered tropical means; grey band marks 14-18 km; profiles use the full-domain denominator",
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
        ).mean("Time", skipna=True).load()
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


def plot_class_spatial_response(classifications):
    """Plot unfiltered spatial occurrence or nudging-minus-control response."""
    if not should_run_figure(5):
        return
    progress("Starting Figure 5: high_cloud_class_spatial_response")
    occurrence_maps = time_mean_class_occurrence_maps(classifications)
    if not occurrence_maps:
        print("Skipping Figure 5: high-cloud occurrence maps are unavailable.")
        return

    control_name = sim_names[0]
    control = occurrence_maps[control_name]
    class_specs = SPATIAL_CLASS_SPECS
    contrast_names = [name for name in sim_names[1:] if name in occurrence_maps]

    if not contrast_names:
        fig, axes = plt.subplots(
            1, len(class_specs), figsize=(17.0, 3.8),
            constrained_layout=True, squeeze=False, sharey=True,
        )
        for column, (key, label, _color) in enumerate(class_specs):
            ax = axes[0, column]
            values = 100.0 * control[key]
            finite = np.asarray(values.values, dtype=float)
            vmax = max(1.0, float(np.nanpercentile(finite, 99.0)))
            image = ax.pcolormesh(
                values.longitude, values.latitude, values,
                shading="auto", cmap="YlGnBu", vmin=0.0, vmax=vmax,
            )
            fig.colorbar(image, ax=ax, label="Occurrence (%)", shrink=0.82)
            ax.set_title(label)
            ax.set_xlabel("Longitude")
        axes[0, 0].set_ylabel("Latitude")
        fig.suptitle(
            f"{control_name}: time-mean high-cloud column occurrence "
            f"({TROPICAL_LAT_BOUNDS[0]:g} to {TROPICAL_LAT_BOUNDS[1]:g} degrees latitude; unfiltered)"
        )
    else:
        fig, axes = plt.subplots(
            len(contrast_names), len(class_specs),
            figsize=(18.0, 3.5 * len(contrast_names) + 1.0),
            constrained_layout=True, squeeze=False, sharex=True, sharey=True,
        )
        differences = {
            (sim_name, key): 100.0 * (occurrence_maps[sim_name][key] - control[key])
            for sim_name in contrast_names
            for key, _label, _color in class_specs
        }
        for column, (key, label, _color) in enumerate(class_specs):
            combined = np.concatenate(
                [
                    np.ravel(np.asarray(differences[(name, key)].values, dtype=float))
                    for name in contrast_names
                ]
            )
            finite = np.abs(combined[np.isfinite(combined)])
            limit = max(
                0.05,
                float(np.nanpercentile(finite, 99.0)) if finite.size else 0.05,
            )
            norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
            column_images = []
            control_percent = 100.0 * control[key]
            contour_levels = control_occurrence_contour_levels(
                key, control_percent.values
            )
            for row, sim_name in enumerate(contrast_names):
                ax = axes[row, column]
                difference = differences[(sim_name, key)]
                image = ax.pcolormesh(
                    difference.longitude, difference.latitude, difference,
                    shading="auto", cmap="RdBu_r", norm=norm,
                )
                column_images.append(image)
                if contour_levels:
                    contours = ax.contour(
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
                    ax.set_title(label)
                if column == 0:
                    ax.set_ylabel(f"{sim_name} - {control_name}\nLatitude")
                if row == len(contrast_names) - 1:
                    ax.set_xlabel("Longitude")
            fig.colorbar(
                column_images[-1], ax=axes[:, column], shrink=0.82,
                label="Occurrence response (percentage points)",
            )
        fig.suptitle(
            "Time-mean high-cloud column-occurrence response "
            f"({TROPICAL_LAT_BOUNDS[0]:g} to {TROPICAL_LAT_BOUNDS[1]:g} degrees latitude; unfiltered)\n"
            "Shading: nudged minus control; grey contours: control occurrence (%)"
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


def plot_rain_matched_high_cloud_response(classifications, fields_by_run):
    if not should_run_figure(6):
        return
    progress("Starting Figure 6: rain_matched_high_cloud_structure_response")
    block_table = rain_conditioned_block_sums(classifications, fields_by_run)
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
        figsize=(17.2, 7.2), constrained_layout=True,
        sharex=True, squeeze=False,
    )
    line_names = sim_names[1:] if has_contrasts else sim_names
    line_colors = cols[1:] if has_contrasts else cols
    required_blocks = 1 if TEST_MODE else MIN_RAIN_CONDITIONED_BLOCKS
    for column, (class_key, class_label, _class_color) in enumerate(
        RAIN_CONDITIONED_CLASS_SPECS
    ):
        axes[0, column].set_title(class_label)
        for row, (property_key, ylabel) in enumerate(property_specs):
            ax = axes[row, column]
            for line_name, color in zip(line_names, line_colors):
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
        f"{title_prefix} cloud depth and outgoing longwave radiation at matched surface rain rate\n"
        f"Unfiltered {TROPICAL_LAT_BOUNDS[0]:g} to {TROPICAL_LAT_BOUNDS[1]:g} degree columns; "
        f"dashed line: {P_PRECIP_MIN / 24.0:g} mm h$^{{-1}}$ class threshold; "
        f"shading: 95% 10-day block-bootstrap interval; >= {required_blocks} block(s) required",
        fontsize=11,
    )
    save_figure(6, fig, "rain_matched_high_cloud_structure_response")
    plt.close(fig)
    progress("Done with figure: rain_matched_high_cloud_structure_response")


def plot_literature_high_cloud_structure(
    dss, fields_by_run, classifications, vertical_profiles
):
    """Unfiltered cloud-top, TTL, vertical-profile, and longwave diagnostics."""
    if not should_run_figure(2):
        return
    progress("Starting Figure 2: literature_high_cloud_structure")
    class_specs = REGIME_SPECS
    ctp_bins = np.arange(0.0, P_HIGH_HPA + 20.0, 20.0)
    ctp_centres = 0.5 * (ctp_bins[:-1] + ctp_bins[1:])
    rows = []

    fig, axes = plt.subplots(2, 3, figsize=(16.8, 8.2), constrained_layout=True)
    ctp_distribution_ax = axes[0, 0]
    ttl_ax = axes[0, 1]
    mean_ctp_ax = axes[0, 2]
    olr_ax = axes[1, 0]
    vertical_profile_ax = axes[1, 1]
    mean_ctt_ax = axes[1, 2]
    control_name = sim_names[0]
    x = np.arange(len(class_specs))
    width = 0.8 / max(1, len(sim_names))
    simulation_linestyles = ("-", "--", ":", "-.")
    for isim, (sim_name, ds, fields) in enumerate(
        zip(sim_names, dss, fields_by_run)
    ):
        classif = classifications[sim_name]
        statistics = full_class_structure_statistics(
            ds,
            fields,
            classif,
            sim_name,
            ctp_bins=ctp_bins,
        )
        ttl_fractions = [statistics[key]["ttl_fraction"] for key, *_ in class_specs]
        olr_means = [statistics[key]["mean_olr"] for key, *_ in class_specs]
        ctp_means = [
            statistics[key]["mean_cloud_top_pressure"] for key, *_ in class_specs
        ]
        ctt_means = [
            statistics[key]["mean_cloud_top_temperature"] for key, *_ in class_specs
        ]
        olr_tb_means = [
            statistics[key]["mean_brightness_temperature"] for key, *_ in class_specs
        ]
        for key, label, _color in class_specs:
            class_statistics = statistics[key]
            ctt_mean = class_statistics["mean_cloud_top_temperature"]
            olr_tb_mean = class_statistics["mean_brightness_temperature"]
            rows.append(dict(
                simulation=sim_name,
                class_key=key,
                cloud_class=label,
                occurrence_fraction=class_statistics["occurrence_fraction"],
                ttl_reaching_fraction_ctp_le_180_hpa=class_statistics["ttl_fraction"],
                class_conditioned_olr_w_m2=class_statistics["mean_olr"],
                mean_cloud_top_pressure_hpa=class_statistics["mean_cloud_top_pressure"],
                mean_diagnosed_cloud_top_temperature_k=ctt_mean,
                mean_olr_equivalent_brightness_temperature_k=olr_tb_mean,
                mean_olr_tb_minus_cloud_top_temperature_k=(
                    olr_tb_mean - ctt_mean
                    if np.isfinite(olr_tb_mean) and np.isfinite(ctt_mean)
                    else np.nan
                ),
                response_filter="none",
                occurrence_denominator="all tropical area-time samples",
                ttl_share_denominator="area-time occurrence of this cloud class",
                olr_condition="unfiltered grid columns in which this cloud class occurs",
                cloud_top_temperature_method=(
                    "linear interpolation of model isobaric temperature to the "
                    f"highest cldfrac >= {CF_THRESH:g} layer's pressure"
                ),
                temporal_spatial_sampling="full 20S-20N analysis period; time-blocked",
            ))
        for key, _label, color in class_specs:
            ctp_distribution_ax.plot(
                statistics[key]["ctp_histogram"],
                ctp_centres,
                color=color,
                ls=simulation_linestyles[isim % len(simulation_linestyles)],
                lw=1.8,
            )
        del statistics
        gc.collect()
        offset = (isim - (len(sim_names) - 1) / 2.0) * width
        ttl_ax.bar(
            x + offset, ttl_fractions, width=width,
            color=cols[isim], label=sim_name,
        )
        olr_ax.bar(
            x + offset, olr_means, width=width,
            color=cols[isim], label=sim_name,
        )
        mean_ctp_ax.scatter(
            x + offset, ctp_means, s=30, marker="o",
            color=cols[isim], label=sim_name, zorder=4,
        )
        mean_ctt_ax.scatter(
            x + offset, ctt_means, s=30, marker="o",
            color=cols[isim], label=f"{sim_name}: cloud-top T", zorder=4,
        )
        mean_ctt_ax.scatter(
            x + offset, olr_tb_means, s=34, marker="s",
            facecolor="white", edgecolor=cols[isim], linewidth=1.2,
            label=f"{sim_name}: OLR-equivalent Tb", zorder=4,
        )
        for xpos, ctt_mean, olr_tb_mean in zip(
            x + offset, ctt_means, olr_tb_means
        ):
            if np.isfinite(ctt_mean) and np.isfinite(olr_tb_mean):
                mean_ctt_ax.plot(
                    [xpos, xpos], [ctt_mean, olr_tb_mean],
                    color=cols[isim], lw=0.9, alpha=0.55, zorder=2,
                )

    core_x = x[0]
    nonprecip_x = x[2]
    literature_color = "#7a4e00"
    mean_ctp_ax.scatter(
        [core_x, nonprecip_x], [223.3, 281.4],
        marker="D", s=48, facecolor="white", edgecolor=literature_color,
        linewidth=1.3, zorder=5,
        label="MODIS 30S-30N: deep convective 223.3; cirriform 281.4 hPa",
    )
    mean_ctt_ax.scatter(
        [core_x, nonprecip_x], [234.9, 231.0],
        marker="D", s=48, facecolor="white", edgecolor=literature_color,
        linewidth=1.3, zorder=5,
        label="MODIS 30S-30N: deep convective 234.9; cirriform 231.0 K",
    )
    olr_ax.scatter(
        [core_x], [205.0], marker="D", s=48,
        facecolor="white", edgecolor=literature_color, linewidth=1.3,
        zorder=5,
        label="Satellite deep-convection threshold: 205 W m$^{-2}$ (0-20N)",
    )
    olr_ax.scatter(
        [nonprecip_x], [260.0], marker=">", s=55,
        facecolor="white", edgecolor="#375a7f", linewidth=1.3,
        zorder=5,
        label="Thin-cirrus OLR: ~260 W m$^{-2}$ (tropical-ocean model)",
    )

    if control_name in vertical_profiles:
        for key, label, color in class_specs:
            profile = vertical_profiles[control_name].sel(class_key=key)
            vertical_profile_ax.plot(
                100.0 * profile, profile.z / 1000.0,
                color=color, lw=1.8, label=label,
            )
        vertical_profile_ax.axhspan(
            UTLS_LAYER_BOUNDS_KM[0], UTLS_LAYER_BOUNDS_KM[1],
            color="0.7", alpha=0.16,
        )

    ctp_distribution_ax.invert_yaxis()
    ctp_distribution_ax.set(
        xlabel="Share of class occurrence per 20-hPa bin",
        ylabel="Cloud-top pressure (hPa)",
        title="(a) Within-class cloud-top distributions",
    )
    class_legend = ctp_distribution_ax.legend(
        handles=[
            Line2D([0], [0], color=color, lw=1.8, label=label)
            for _key, label, color in class_specs
        ],
        loc="lower right",
        frameon=False,
        fontsize=7,
    )
    ctp_distribution_ax.add_artist(class_legend)
    ctp_distribution_ax.legend(
        handles=[
            Line2D(
                [0], [0], color="0.15",
                ls=simulation_linestyles[index % len(simulation_linestyles)],
                lw=1.5, label=sim_name,
            )
            for index, sim_name in enumerate(sim_names)
        ],
        loc="upper right",
        frameon=False,
        fontsize=7,
    )
    ttl_ax.set(
        title="(b) Share of each class reaching the TTL",
        ylabel="Class occurrence with CTP <= 180 hPa (fraction)",
        xticks=x,
        xticklabels=[label for _key, label, _color in class_specs],
    )
    mean_ctp_ax.set(
        title="(c) Mean cloud-top pressure",
        ylabel="Cloud-top pressure (hPa)",
        xticks=x,
        xticklabels=[label for _key, label, _color in class_specs],
    )
    mean_ctp_ax.set_ylim(P_HIGH_HPA, 100.0)
    olr_ax.set(
        title="(d) Mean OLR where each class occurs",
        ylabel="Unfiltered class-conditioned OLR (W m$^{-2}$)",
        xticks=x,
        xticklabels=[label for _key, label, _color in class_specs],
    )
    vertical_profile_ax.set(
        xlabel="Contribution to domain-mean cloud fraction (%)",
        ylabel="Height (km)",
        title=f"(e) Domain cloud amount from each class ({control_name})",
    )
    mean_ctt_ax.set(
        title="(f) Cloud-top and OLR-equivalent temperature",
        ylabel="Temperature (K)",
        xticks=x,
        xticklabels=[label for _key, label, _color in class_specs],
    )
    for ax in (ttl_ax, mean_ctp_ax, olr_ax, mean_ctt_ax):
        ax.tick_params(axis="x", rotation=18)
        ax.legend(frameon=False, fontsize=8)
    if control_name in vertical_profiles:
        vertical_profile_ax.legend(frameon=False, fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Unfiltered high-cloud height and longwave structure (model bars/points: 20S-20N)\n"
        "Circles: model isobaric temperature interpolated to diagnosed cloud-top pressure; "
        "squares: PyFLEXTRKR OLR-equivalent brightness temperature"
    )
    save_table(pd.DataFrame(rows), "literature_high_cloud_structure_metrics")
    save_table(
        pd.DataFrame(LITERATURE_CLOUD_STRUCTURE_VALUES),
        "high_cloud_structure_literature_values",
    )
    save_figure(2, fig, "literature_high_cloud_structure")
    plt.close(fig)
    progress("Done with figure: literature_high_cloud_structure")


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
    else:
        progress(
            "Classification products will be reused when current and rebuilt "
            "only when missing, stale, incomplete, or on the wrong Time axis"
        )
    write_classification_method_table()

    dss = load_datasets()
    fields_by_run = prepare_fields(dss)
    classifications = prepare_classifications(dss, fields_by_run)
    vertical_profiles = (
        compute_vertical_contribution_profiles(dss, classifications)
        if should_run_figure(2) or should_run_figure(4)
        else {}
    )

    plot_class_budget(classifications, fields_by_run)
    plot_literature_high_cloud_structure(
        dss, fields_by_run, classifications, vertical_profiles
    )
    plot_validation_snapshots(dss, fields_by_run, classifications)
    plot_vertical_contribution_profiles(vertical_profiles)
    plot_class_spatial_response(classifications)
    plot_rain_matched_high_cloud_response(classifications, fields_by_run)

    progress("Refined high-cloud classification diagnostics complete")


if __name__ == "__main__":
    main()
