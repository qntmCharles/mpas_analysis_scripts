#!/usr/bin/env python
"""Diagnose circulation, tropical width, and momentum and moisture transport.

The analysis treats both hemispheres as independent aquaplanet realizations.
It saves each hemispheric scalar and their absolute-latitude mean.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

_THIS_DIR = Path(__file__).resolve().parent
for _parent in (_THIS_DIR.parents[1], _THIS_DIR.parents[2]):
    if str(_parent) not in sys.path:
        sys.path.append(str(_parent))

import numpy as np
import pandas as pd
import xarray as xr
import dask
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt

from diagnostics.diagnostic_helpers import (
    ANALYSIS_END,
    ANALYSIS_START,
    FigureSaver,
    LATLON_PATTERN,
    QUICK_TEST_LON_BOUNDS,
    ZONAL_MEAN_PATTERN,
    annotate_panel_label,
    cache_manifest,
    cache_manifest_matches,
    diagnostic_cache_dir,
    diagnostic_profile_cache_dir,
    diagnostic_output_dir,
    difference_cmap,
    hemispheric_fold,
    load_latlon_dataset,
    load_netcdf,
    progress,
    read_json_file,
    simulation_colour,
    standardize_latlon_dataset,
    subset_between,
    symmetric_percentile_levels_for_fields,
    time_block_slices,
    write_csv_atomic,
    write_json_file,
    write_netcdf_atomic,
)
from diagnostics.analysis_config import (
    ALL_SIMULATION_ARCHIVES,
    ALL_SIMULATION_LABELS,
    BLOCK_LENGTH_DAYS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_PERCENTILES,
    FIELD_BOOTSTRAP_REPLICATES,
    SIMULATION_ARCHIVES,
    SIMULATION_LABELS,
    SIMULATION_PROFILES,
    SIMULATION_ROOT,
    canonical_simulation_name,
    profile_cache_directory,
)
from diagnostics.uncertainty import independent_bootstrap_field_difference
from diagnostics.mean_state.vapour_transport_budget import (
    mixing_ratio_to_specific_humidity,
)


FIGURES = (1, 2, 3, 4, 5, 6, 7, 8)
SIMULATIONS = tuple(
    (SIMULATION_LABELS[name], directory)
    for name, directory in SIMULATION_ARCHIVES.items()
)


COMPARISON_NAMES = {
    "control": ("control", "control"),
    "nudged-u": ("u", "U"),
    "nudged-T": ("t", "T"),
    "nudged-T-from-u": ("t_from_u", "T from U"),
}


COMPARISONS = [("control", 0, None, "control")]
COMPARISONS.extend(
    (
        f"{COMPARISON_NAMES[name][0]}_minus_control",
        index,
        0,
        f"{COMPARISON_NAMES[name][1]} - control",
    )
    for index, name in enumerate(tuple(SIMULATION_ARCHIVES)[1:], start=1)
)
COMPARISONS.extend(
    (
        f"u_minus_{COMPARISON_NAMES[name][0]}",
        1,
        index,
        f"U - {COMPARISON_NAMES[name][1]}",
    )
    for index, name in enumerate(tuple(SIMULATION_ARCHIVES)[2:], start=2)
)
COMPARISONS = tuple(COMPARISONS)

ROOT_DIR = SIMULATION_ROOT
SAVE_LOCATION = diagnostic_output_dir(
    "/home/cwp29/tmp/mpas/crf_on/final/circulation/"
)
CACHE_ROOT = Path(diagnostic_profile_cache_dir("circulation"))
SIMULATION_CACHE_ROOT = Path(diagnostic_cache_dir("circulation", "simulations"))
CACHE_SCHEMA = {
    "circulation": "circulation-v5",
    "eddy": "circulation-eddy-v3",
    "moisture": "circulation-moisture-v3",
    "tem": "circulation-tem-v4",
}
CACHE_FILES = {
    "circulation": (
        "circulation_fields.nc",
        "circulation_metrics.nc",
        "circulation_metric_blocks.nc",
    ),
    "eddy": (
        "eddy_momentum_fields.nc",
        "eddy_momentum_metrics.nc",
        "eddy_momentum_metric_blocks.nc",
    ),
    "moisture": (
        "moisture_transport_fields.nc",
        "moisture_transport_profile_blocks.nc",
    ),
    "tem": (
        "tem_circulation_budget_fields.nc",
        "tem_budget_metrics.nc",
        "tem_budget_metric_blocks.nc",
    ),
}

PSI_METHOD = "Psi_300_700"
STJ_METHOD = "adjusted_peak"
EDJ_METHOD = "peak"
TPB_METHOD = "max_gradient"
BLOCK_DAYS = BLOCK_LENGTH_DAYS["circulation"]
MIN_BLOCKS = 6
N_BOOTSTRAP = BOOTSTRAP_REPLICATES
CI_PERCENTILES = CONFIDENCE_PERCENTILES
ROBUST_SIGN_THRESHOLD = 0.95
EARTH_RADIUS_M = 6_371_000.0
EARTH_ANGULAR_VELOCITY_S1 = 7.2921159e-5
GRAVITY_M_S2 = 9.80665
SECONDS_PER_DAY = 86_400.0
MOMENTUM_BALANCE_TOP_HPA = 150.0
MOMENTUM_BALANCE_BOTTOM_HPA = 300.0
TEM_EARTH_RADIUS_M = 6_371_229.0
TEM_ANGULAR_VELOCITY_S1 = 2.0 * np.pi / (
    (23.0 * 60.0 + 56.0) * 60.0 + 4.1
)
TEM_STJ_LAYER_TOP_HPA = 150.0
TEM_STJ_LAYER_BOTTOM_HPA = 300.0
TEM_STJ_LATITUDE_BOUNDS = (15.0, 45.0)
TEM_COLOUR_SCALE_HEIGHT_BOUNDS_M = (1_000.0, 30_000.0)
TEM_FIGURE_5_6_COLOUR_SCALE_FACTOR = 1.25
TEM_MMC_DECOMPOSITION_COLOUR_PERCENTILE = 90.0
EP_FLUX_ARROW_LENGTH_M = 180_000.0
MOISTURE_BOTTOM_PRESSURE_HPA = 1000.0
MOISTURE_BOTTOM_SOURCE_HPA = 950.0
DEFAULT_TEST_TIMES = 5

TEM_ARCHIVE_VARIABLES = (
    "uReconstructZonal",
    "uReconstructMeridional",
    "pressure",
    "rho_zm",
    "vres",
    "wres",
    "EPFd",
    "EPFd_y",
    "EPFd_z",
    "Fy",
    "Fz",
)
TEM_BUDGET_COMPONENTS = (
    "a_mmc",
    "a_ep",
    "unresolved_direct_momentum_residual",
    "a_tendency",
)
TEM_BOOTSTRAP_FIELDS = (
    "residual_streamfunction",
    "vres",
    "wres",
    "a_vres",
    "a_wres",
    "a_mmc",
    "a_ep",
    "a_ep_meridional",
    "a_ep_vertical",
    "ep_flux_meridional",
    "ep_flux_vertical",
    "unresolved_direct_momentum_residual",
    "a_tendency",
    "stj_layer_a_mmc",
    "stj_layer_a_ep",
    "stj_layer_unresolved_direct_momentum_residual",
    "stj_layer_a_tendency",
)

HEMISPHERES = ("SH", "NH", "mean")
HEMISPHERE_BOUNDS = {"SH": (-70.0, -0.5), "NH": (0.5, 70.0)}

METRIC_SPECS = {
    "hadley_edge_latitude": {
        "units": "degrees_north_absolute",
        "method": f"PyTropD PSI: {PSI_METHOD}",
        "definition": "Absolute latitude of the Hadley-cell edge.",
    },
    "hadley_strength": {
        "units": "kg s-1",
        "method": "absolute extremum of the 300-700 hPa pressure-mean streamfunction",
        "definition": "Magnitude of the hemispheric Hadley-cell overturning maximum.",
    },
    "hadley_max_latitude": {
        "units": "degrees_north_absolute",
        "method": "absolute extremum of the 300-700 hPa pressure-mean streamfunction",
        "definition": "Absolute latitude of maximum Hadley-cell overturning.",
    },
    "stj_latitude": {
        "units": "degrees_north_absolute",
        "method": f"PyTropD STJ: {STJ_METHOD}",
        "definition": "Absolute latitude of the subtropical jet.",
    },
    "stj_strength": {
        "units": "m s-1",
        "method": "pressure-mean 100-400 hPa zonal wind at the PyTropD STJ",
        "definition": "Pressure-weighted subtropical-jet strength.",
    },
    "edj_latitude": {
        "units": "degrees_north_absolute",
        "method": f"PyTropD EDJ: {EDJ_METHOD}",
        "definition": "Absolute latitude of the eddy-driven jet.",
    },
    "edj_strength": {
        "units": "m s-1",
        "method": "850 hPa zonal wind at the PyTropD EDJ",
        "definition": "Eddy-driven-jet strength.",
    },
    "tropopause_break_latitude": {
        "units": "degrees_north_absolute",
        "method": f"PyTropD TPB: {TPB_METHOD}",
        "definition": "Absolute latitude of the tropopause break.",
    },
    "stj_core_pressure": {
        "units": "hPa",
        "method": "100-400 hPa zonal-wind maximum at the PyTropD STJ",
        "definition": "Pressure of the upper-tropospheric STJ wind maximum.",
    },
    "stj_core_strength": {
        "units": "m s-1",
        "method": "100-400 hPa zonal-wind maximum at the PyTropD STJ",
        "definition": "Strength of the upper-tropospheric STJ wind maximum.",
    },
    "eddy_flux_max_latitude": {
        "units": "degrees_north_absolute",
        "method": "gridpoint maximum of 200-250 hPa poleward eddy momentum flux",
        "definition": "Absolute latitude of maximum upper-tropospheric poleward u-prime-v-prime.",
    },
}


def parse_command_line(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Use a bounded subset of the actual simulations and isolated "
            "test caches and figures."
        ),
    )
    parser.add_argument(
        "--test-times",
        type=int,
        default=DEFAULT_TEST_TIMES,
        metavar="N",
        help="Number of six-hourly times in test mode.",
    )
    parser.add_argument(
        "--test-start",
        default=None,
        metavar="TIME",
        help="Optional first six-hourly time in the test subset.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=FIGURES,
        default=None,
        help="Build only these numbered figures.",
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--figures-only",
        action="store_true",
        help="Build figures only from current compact circulation caches.",
    )
    cache_mode.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild selected compact circulation caches before plotting.",
    )
    args = parser.parse_args(argv)
    if args.test_times < 1:
        parser.error("--test-times must be at least 1")
    if not args.test and args.test_start is not None:
        parser.error("--test-start requires --test")
    if args.test_start is not None:
        try:
            start = pd.Timestamp(args.test_start)
        except (TypeError, ValueError):
            parser.error("--test-start must be a valid date or time")
        if pd.isna(start):
            parser.error("--test-start must be a valid date or time")
        args.test_start = start.strftime("%Y-%m-%d %H:%M:%S")
    return args


def load_pytropd() -> Any:
    """Load PyTropD with its SciPy 1.14 compatibility alias."""

    import scipy.integrate

    if not hasattr(scipy.integrate, "cumtrapz"):
        scipy.integrate.cumtrapz = scipy.integrate.cumulative_trapezoid
    import pytropd

    return pytropd


def pytropd_version() -> str:
    try:
        return version("pytropd")
    except PackageNotFoundError:
        return "unavailable"


def requested_families(figures: Sequence[int]) -> set[str]:
    selected = set(figures)
    families = {"circulation"} if selected & set(FIGURES) else set()
    if selected & {3, 4}:
        families.add("eddy")
    if 4 in selected:
        families.add("moisture")
    if selected & {5, 6, 7, 8}:
        families.add("tem")
    return families


def family_source_paths(
    family: str,
    simulations: Sequence[tuple[str, str]] = SIMULATIONS,
) -> list[str]:
    pattern = (
        ZONAL_MEAN_PATTERN
        if family in {"circulation", "tem"}
        else LATLON_PATTERN
    )
    return [
        path
        for _label, directory in simulations
        for path in sorted(glob.glob(str(ROOT_DIR / directory / pattern)))
    ]


def family_manifest(
    family: str,
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
    include_sources: bool = True,
    simulations: Sequence[tuple[str, str]] = SIMULATIONS,
) -> dict:
    configuration = {
        "family": family,
        "simulations": [list(item) for item in simulations],
        "analysis_start": ANALYSIS_START,
        "analysis_end": ANALYSIS_END,
        "block_days": None if test else BLOCK_DAYS,
        "minimum_blocks": 1 if test else MIN_BLOCKS,
        "hemisphere_handling": "independent metrics and absolute-latitude mean",
        "test_subset": bool(test),
    }
    configuration["bootstrap_uncertainty"] = {
        "method": "independent complete-block resampling by simulation",
        "confidence_percentiles": list(CI_PERCENTILES),
        "scalar_replicates": N_BOOTSTRAP,
        "field_replicates": FIELD_BOOTSTRAP_REPLICATES,
        "formal_intervals_in_test_mode": False,
    }
    if test:
        configuration.update(
            test_times=int(test_times),
            test_start=test_start,
        )
    if family == "circulation":
        configuration.update(
            psi_method=PSI_METHOD,
            stj_method=STJ_METHOD,
            edj_method=EDJ_METHOD,
            tpb_method=TPB_METHOD,
            spatial_parity={
                "streamfunction": "odd",
                "zonal_wind": "even",
                "tropopause_pressure": "even",
                "mean_meridional_flow_torque": "even",
            },
            mean_meridional_flow_torque=(
                "-(f + zonal-mean relative vorticity) * "
                "zonal-mean meridional wind"
            ),
            cached_uncertainty_fields=[
                "streamfunction",
                "zonal_wind",
                "upper_zonal_wind",
                "zonal_wind_850",
                "upper_mean_meridional_flow_torque",
            ],
        )
    elif family == "eddy":
        configuration.update(
            eddy_definition="instantaneous zonal-mean anomalies",
            upper_tropospheric_layer_hpa=[200.0, 250.0],
            spatial_parity={
                "eddy_momentum_flux": "odd",
                "eddy_momentum_flux_convergence": "even",
            },
            cached_uncertainty_fields=[
                "eddy_momentum_flux",
                "eddy_momentum_flux_convergence",
                "upper_eddy_momentum_flux",
                "upper_eddy_momentum_flux_convergence",
            ],
        )
        if test:
            configuration["test_longitude_bounds"] = list(
                QUICK_TEST_LON_BOUNDS
            )
    elif family == "moisture":
        configuration.pop("hemisphere_handling")
        configuration.update(
            archived_vapour_variable="qv_isobaric",
            humidity_conversion="q = qv / (1 + qv)",
            mean_circulation_definition="[v_bar] [q_bar]",
            eddy_definition=(
                "[overline(v q)] - [v_bar] [q_bar]; stationary and "
                "transient departures combined"
            ),
            cached_uncertainty_fields=[
                f"{kind}_{component}"
                for kind in ("transport", "convergence")
                for component in MOISTURE_FLUX_COMPONENTS
            ]
            + [
                "section_total_meridional_moisture_flux",
                "mean_meridional_wind",
                "mean_specific_humidity",
            ],
        )
        if test:
            configuration["test_longitude_bounds"] = list(
                QUICK_TEST_LON_BOUNDS
            )
    elif family == "tem":
        configuration.update(
            required_archived_fields=[
                "uReconstructZonal",
                "uReconstructMeridional",
                "pressure",
                "rho_zm",
                "vres",
                "wres",
                "EPFd",
                "EPFd_y",
                "EPFd_z",
                "Fy",
                "Fz",
            ],
            residual_streamfunction=(
                "2*pi*a*cos(latitude)*integral_z^top(rho_zm*vres dz)"
            ),
            residual_streamfunction_top_boundary="zero at the archive top",
            tem_meridional_acceleration=(
                "vres * [f - d(u*cos(latitude))/d(latitude) "
                "/ (a*cos(latitude))]"
            ),
            tem_meridional_discretization=(
                "producer m_dy = f - (du/dlatitude - tan(latitude)*u)/a"
            ),
            tem_vertical_acceleration="-wres * du/dz",
            resolved_ep_acceleration="EPFd, copied without sign reversal",
            ep_flux_vectors=(
                "archived Fy and Fz, transformed with the local pressure "
                "coordinate Jacobian for plotting"
            ),
            ep_flux_arrow_reference=(
                "95th-percentile physical vector magnitude, shared across "
                "all response panels"
            ),
            ep_flux_arrow_length_m=EP_FLUX_ARROW_LENGTH_M,
            unresolved_direct_residual=(
                "du/dt - (A_MMC + EPFd); not a diagnosed nudging tendency"
            ),
            tendency_derivative=(
                "second-order centred finite differences with second-order "
                "one-sided record endpoints"
            ),
            tem_constants={
                "earth_radius_m": TEM_EARTH_RADIUS_M,
                "angular_velocity_s-1": TEM_ANGULAR_VELOCITY_S1,
            },
            stj_profile_pressure_hpa=[
                TEM_STJ_LAYER_TOP_HPA,
                TEM_STJ_LAYER_BOTTOM_HPA,
            ],
            stj_summary_absolute_latitude_degrees=list(
                TEM_STJ_LATITUDE_BOUNDS
            ),
            derivative_boundary_handling=(
                "exclude first and last height and latitude points"
            ),
            nonfinite_handling=(
                "mask non-finite archived and derived TEM values before "
                "climatology and block calculations"
            ),
            stipple_display=(
                "one actual pointwise-significant grid point per coarse "
                "12-by-18 display cell"
            ),
            spatial_parity={
                "residual_streamfunction": "odd",
                "vres": "odd",
                "wres": "even",
                "ep_flux_meridional": "odd",
                "ep_flux_vertical": "even",
                "momentum_accelerations": "even",
            },
            cached_uncertainty_fields=list(TEM_BOOTSTRAP_FIELDS),
        )
    else:
        raise ValueError(f"Unknown circulation cache family: {family}")
    return cache_manifest(
        "mean_state.circulation_diagnostics",
        schema=CACHE_SCHEMA[family],
        configuration=configuration,
        dependencies=(
            {"pytropd": pytropd_version()}
            if family in {"circulation", "eddy"}
            else {}
        ),
        source_paths=(
            family_source_paths(family, simulations)
            if include_sources
            else None
        ),
    )


def test_run_label(test_times: int, test_start: str | None) -> str:
    start = (
        pd.Timestamp(test_start).strftime("%Y%m%dT%H%M")
        if test_start is not None
        else "record-start"
    )
    lon_min, lon_max = QUICK_TEST_LON_BOUNDS
    return (
        f"all_{start}_{int(test_times):03d}times_"
        f"lon{lon_min:g}_{lon_max:g}"
    )


def run_cache_root(
    test: bool,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
) -> Path:
    if not test:
        return CACHE_ROOT
    return CACHE_ROOT / "testing" / test_run_label(test_times, test_start)


def run_simulation_cache_root(
    test: bool,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
) -> Path:
    """Return the profile-independent simulation cache root."""

    if not test:
        return SIMULATION_CACHE_ROOT
    return (
        SIMULATION_CACHE_ROOT
        / "testing"
        / test_run_label(test_times, test_start)
    )


def simulation_cache_root(
    simulation: str,
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
) -> Path:
    return run_simulation_cache_root(test, test_times, test_start) / (
        canonical_simulation_name(simulation)
    )


def profile_simulations(profile: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (ALL_SIMULATION_LABELS[name], ALL_SIMULATION_ARCHIVES[name])
        for name in SIMULATION_PROFILES[profile]
    )


def aggregate_cache_root(
    profile: str,
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
) -> Path:
    root = Path(profile_cache_directory("circulation", profile=profile))
    if test:
        root = root / "testing" / test_run_label(test_times, test_start)
    return root


def run_save_location(
    test: bool,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
) -> str:
    if not test:
        return SAVE_LOCATION
    return os.path.join(
        SAVE_LOCATION,
        "testing",
        test_run_label(test_times, test_start),
    )


def cache_path(filename: str, cache_root: Path | None = None) -> Path:
    return (CACHE_ROOT if cache_root is None else cache_root) / filename


def manifest_path(family: str, cache_root: Path | None = None) -> Path:
    root = CACHE_ROOT if cache_root is None else cache_root
    return root / f"{family}.manifest.json"


def family_cache_is_current(
    family: str,
    manifest: dict | None = None,
    cache_root: Path | None = None,
    validate_sources: bool = True,
) -> bool:
    if not all(
        cache_path(name, cache_root).is_file() for name in CACHE_FILES[family]
    ):
        return False
    try:
        expected = manifest or family_manifest(family)
        saved = read_json_file(str(manifest_path(family, cache_root)))
    except (OSError, ValueError):
        return False
    return cache_manifest_matches(
        saved, expected, compare_sources=validate_sources
    )


def pressure_mean(
    field: xr.DataArray,
    lower_hpa: float,
    upper_hpa: float,
) -> xr.DataArray:
    """Return a pressure-weighted mean between two pressure bounds."""

    layer = subset_between(field.sortby("pres"), "pres", (lower_hpa, upper_hpa))
    if layer.sizes.get("pres", 0) < 2:
        raise ValueError(
            f"Pressure mean needs two levels between {lower_hpa:g} and {upper_hpa:g} hPa"
        )
    depth = float(layer.pres.values[-1] - layer.pres.values[0])
    return layer.integrate("pres") / depth


def fold_to_absolute(field: xr.DataArray, parity: str) -> xr.DataArray:
    """Fold one global field and retain its nonnegative-latitude half."""

    folded = hemispheric_fold(field, parity=parity)
    folded = folded.where(folded.latitude >= 0.0, drop=True)
    folded = folded.rename(latitude="absolute_latitude")
    folded.absolute_latitude.attrs.update(
        units="degrees_north_absolute",
        long_name="absolute latitude",
    )
    return folded


def mean_meridional_flow_torque(
    zonal_wind: xr.DataArray,
    meridional_wind: xr.DataArray,
    radius_m: float = EARTH_RADIUS_M,
    angular_velocity_s1: float = EARTH_ANGULAR_VELOCITY_S1,
) -> xr.DataArray:
    """Return -(f + zonal-mean relative vorticity) times zonal-mean v."""

    if "latitude" not in zonal_wind.dims:
        raise ValueError("Mean meridional-flow torque needs a latitude dimension")
    zonal_wind, meridional_wind = xr.align(
        zonal_wind,
        meridional_wind,
        join="exact",
    )
    meridional_wind = meridional_wind.transpose(*zonal_wind.dims)
    latitude = np.asarray(zonal_wind.latitude.values, dtype=float)
    if latitude.size < 3:
        raise ValueError("Mean meridional-flow torque needs three latitude points")
    phi = np.deg2rad(latitude)
    cos_phi = np.cos(phi)
    axis = zonal_wind.get_axis_num("latitude")
    coordinate_shape = [
        latitude.size if index == axis else 1
        for index in range(zonal_wind.ndim)
    ]
    cos_broadcast = np.reshape(cos_phi, coordinate_shape)
    weighted_wind = np.asarray(zonal_wind.values, dtype=float) * cos_broadcast
    relative_vorticity = -np.gradient(
        weighted_wind,
        phi,
        axis=axis,
        edge_order=2,
    ) / (radius_m * cos_broadcast)
    coriolis_parameter = np.reshape(
        2.0 * angular_velocity_s1 * np.sin(phi),
        coordinate_shape,
    )
    values = -(coriolis_parameter + relative_vorticity) * np.asarray(
        meridional_wind.values,
        dtype=float,
    )
    output = xr.DataArray(
        values,
        dims=zonal_wind.dims,
        coords=zonal_wind.coords,
        name="mean_meridional_flow_torque",
    )
    output.attrs.update(
        units="m s-2",
        long_name="absolute-vorticity torque on the mean meridional flow",
        formula="-(f + zonal-mean relative vorticity) * zonal-mean v",
        sign_convention="positive for eastward acceleration",
    )
    return output


def streamfunction_from_wind(
    meridional_wind: xr.DataArray,
    pytropd_api: Any | None = None,
) -> xr.DataArray:
    api = pytropd_api or load_pytropd()
    wind = meridional_wind.transpose("latitude", "pres").sortby("latitude")
    wind = wind.sortby("pres")
    values = api.TropD_Calculate_StreamFunction(
        np.asarray(wind.values, dtype=float),
        np.asarray(wind.latitude.values, dtype=float),
        np.asarray(wind.pres.values, dtype=float),
    )
    return xr.DataArray(
        values,
        dims=("latitude", "pres"),
        coords={"latitude": wind.latitude, "pres": wind.pres},
        name="streamfunction",
        attrs={"units": "kg s-1", "long_name": "meridional mass streamfunction"},
    )


def tropopause_pressure(
    temperature: xr.DataArray,
    pytropd_api: Any | None = None,
) -> xr.DataArray:
    api = pytropd_api or load_pytropd()
    field = temperature.transpose("latitude", "pres").sortby("latitude")
    field = field.sortby("pres")
    values = api.TropD_Calculate_TropopauseHeight(
        np.asarray(field.values, dtype=float),
        np.asarray(field.pres.values, dtype=float),
    )
    return xr.DataArray(
        np.asarray(values).squeeze(),
        dims=("latitude",),
        coords={"latitude": field.latitude},
        name="tropopause_pressure",
        attrs={"units": "hPa", "long_name": "WMO lapse-rate tropopause pressure"},
    )


def _scalar_metric_result(value: Any) -> float:
    array = np.asarray(value, dtype=float).squeeze()
    if array.size != 1:
        raise ValueError(f"Expected one hemispheric metric value, got shape {array.shape}")
    return float(array)


def _hemisphere_result(result: Any) -> dict[str, float]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("PyTropD must return SH and NH metric values")
    return {
        "SH": _scalar_metric_result(result[0]),
        "NH": _scalar_metric_result(result[1]),
    }


def diagnose_pytropd_latitudes(
    zonal_wind: xr.DataArray,
    meridional_wind: xr.DataArray,
    temperature: xr.DataArray,
    pytropd_api: Any | None = None,
) -> dict[str, dict[str, float]]:
    """Call the four required PyTropD definitions without variants."""

    api = pytropd_api or load_pytropd()
    u = zonal_wind.transpose("latitude", "pres").sortby("latitude").sortby("pres")
    v = meridional_wind.transpose("latitude", "pres").sortby("latitude").sortby("pres")
    t = temperature.transpose("latitude", "pres").sortby("latitude").sortby("pres")
    latitude = np.asarray(u.latitude.values, dtype=float)
    pressure = np.asarray(u.pres.values, dtype=float)
    output = {
        "hadley_edge_latitude": _hemisphere_result(
            api.TropD_Metric_PSI(
                np.asarray(v.values, dtype=float),
                latitude,
                pressure,
                method=PSI_METHOD,
            )
        ),
        "stj_latitude": _hemisphere_result(
            api.TropD_Metric_STJ(
                np.asarray(u.values, dtype=float),
                latitude,
                pressure,
                method=STJ_METHOD,
            )
        ),
        "edj_latitude": _hemisphere_result(
            api.TropD_Metric_EDJ(
                np.asarray(u.values, dtype=float),
                latitude,
                pressure,
                method=EDJ_METHOD,
            )
        ),
        "tropopause_break_latitude": _hemisphere_result(
            api.TropD_Metric_TPB(
                np.asarray(t.values, dtype=float),
                latitude,
                pressure,
                method=TPB_METHOD,
            )
        ),
    }
    for values in output.values():
        for hemisphere in ("SH", "NH"):
            values[hemisphere] = abs(values[hemisphere])
    return output


def _profile_at_latitude(field: xr.DataArray, latitude: float) -> xr.DataArray:
    return field.sortby("latitude").interp(latitude=float(latitude))


def _hadley_extremum(
    psi_layer: xr.DataArray,
    hemisphere: str,
    edge_latitude: float,
) -> tuple[float, float]:
    signed_edge = -abs(edge_latitude) if hemisphere == "SH" else abs(edge_latitude)
    bounds = (signed_edge, -0.5) if hemisphere == "SH" else (0.5, signed_edge)
    cell = subset_between(psi_layer, "latitude", bounds)
    if cell.sizes.get("latitude", 0) == 0 or not np.isfinite(cell.values).any():
        return np.nan, np.nan
    index = int(np.nanargmax(np.abs(cell.values)))
    return float(abs(cell.latitude.values[index])), float(abs(cell.values[index]))


def circulation_metric_values(
    zonal_wind: xr.DataArray,
    meridional_wind: xr.DataArray,
    temperature: xr.DataArray,
    pytropd_api: Any | None = None,
) -> tuple[dict[str, dict[str, float]], xr.DataArray, xr.DataArray]:
    api = pytropd_api or load_pytropd()
    locations = diagnose_pytropd_latitudes(
        zonal_wind,
        meridional_wind,
        temperature,
        pytropd_api=api,
    )
    psi = streamfunction_from_wind(meridional_wind, pytropd_api=api)
    psi_layer = pressure_mean(psi, 300.0, 700.0)
    upper_wind = pressure_mean(zonal_wind, 100.0, 400.0)
    wind_850 = zonal_wind.sortby("pres").interp(pres=850.0)

    values = {name: dict(item) for name, item in locations.items()}
    for name in (
        "hadley_strength",
        "hadley_max_latitude",
        "stj_strength",
        "edj_strength",
        "stj_core_pressure",
        "stj_core_strength",
    ):
        values[name] = {}

    for hemisphere in ("SH", "NH"):
        sign = -1.0 if hemisphere == "SH" else 1.0
        edge = locations["hadley_edge_latitude"][hemisphere]
        maximum_latitude, strength = _hadley_extremum(
            psi_layer,
            hemisphere,
            edge,
        )
        values["hadley_max_latitude"][hemisphere] = maximum_latitude
        values["hadley_strength"][hemisphere] = strength

        stj_signed = sign * locations["stj_latitude"][hemisphere]
        edj_signed = sign * locations["edj_latitude"][hemisphere]
        values["stj_strength"][hemisphere] = float(
            _profile_at_latitude(upper_wind, stj_signed).values
        )
        values["edj_strength"][hemisphere] = float(
            _profile_at_latitude(wind_850, edj_signed).values
        )

        core_profile = subset_between(
            _profile_at_latitude(zonal_wind, stj_signed),
            "pres",
            (100.0, 400.0),
        )
        core_index = int(np.nanargmax(core_profile.values))
        values["stj_core_pressure"][hemisphere] = float(
            core_profile.pres.values[core_index]
        )
        values["stj_core_strength"][hemisphere] = float(
            core_profile.values[core_index]
        )

    return values, psi, tropopause_pressure(temperature, pytropd_api=api)


def add_hemisphere_mean(
    values: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    output = {name: dict(hemispheres) for name, hemispheres in values.items()}
    for hemispheres in output.values():
        pair = np.asarray([hemispheres.get("SH"), hemispheres.get("NH")], dtype=float)
        hemispheres["mean"] = float(np.nanmean(pair))
    return output


def make_metric_dataset(
    simulations: Sequence[str],
    records: Sequence[dict[str, dict[str, float]]],
) -> xr.Dataset:
    metrics = list(records[0])
    values = np.full((len(simulations), len(HEMISPHERES), len(metrics)), np.nan)
    disagreement = np.full((len(simulations), len(metrics)), np.nan)
    for simulation_index, record in enumerate(records):
        for metric_index, metric in enumerate(metrics):
            hemispheres = add_hemisphere_mean({metric: record[metric]})[metric]
            for hemisphere_index, hemisphere in enumerate(HEMISPHERES):
                values[simulation_index, hemisphere_index, metric_index] = hemispheres[
                    hemisphere
                ]
            disagreement[simulation_index, metric_index] = abs(
                hemispheres["NH"] - hemispheres["SH"]
            )
    units = [METRIC_SPECS[name]["units"] for name in metrics]
    methods = [METRIC_SPECS[name]["method"] for name in metrics]
    definitions = [METRIC_SPECS[name]["definition"] for name in metrics]
    return xr.Dataset(
        {
            "value": (
                ("simulation", "hemisphere", "metric"),
                values,
            ),
            "hemisphere_disagreement": (
                ("simulation", "metric"),
                disagreement,
            ),
        },
        coords={
            "simulation": list(simulations),
            "hemisphere": list(HEMISPHERES),
            "metric": metrics,
            "metric_units": ("metric", units),
            "metric_method": ("metric", methods),
            "metric_definition": ("metric", definitions),
        },
        attrs={
            "hemisphere_mean": "arithmetic mean of absolute-latitude NH and SH values",
            "hemisphere_disagreement": "absolute NH-minus-SH difference",
        },
    )


def make_metric_block_dataset(
    simulations: Sequence[str],
    records: Sequence[Sequence[dict[str, dict[str, float]]]],
) -> xr.Dataset:
    nblocks = min(len(item) for item in records)
    metrics = list(records[0][0])
    values = np.full(
        (len(simulations), nblocks, len(HEMISPHERES), len(metrics)),
        np.nan,
    )
    for simulation_index, simulation_records in enumerate(records):
        for block_index, record in enumerate(simulation_records[:nblocks]):
            record_with_mean = add_hemisphere_mean(record)
            for metric_index, metric in enumerate(metrics):
                for hemisphere_index, hemisphere in enumerate(HEMISPHERES):
                    values[
                        simulation_index,
                        block_index,
                        hemisphere_index,
                        metric_index,
                    ] = record_with_mean[metric][hemisphere]
    return xr.Dataset(
        {
            "value": (
                ("simulation", "block", "hemisphere", "metric"),
                values,
            )
        },
        coords={
            "simulation": list(simulations),
            "block": np.arange(nblocks),
            "hemisphere": list(HEMISPHERES),
            "metric": metrics,
            "metric_units": (
                "metric",
                [METRIC_SPECS[name]["units"] for name in metrics],
            ),
        },
        attrs={"block_days": BLOCK_DAYS},
    )


def circulation_fields_for_mean(
    zonal_wind: xr.DataArray,
    meridional_wind: xr.DataArray,
    psi: xr.DataArray,
    tropopause: xr.DataArray,
) -> xr.Dataset:
    folded_wind = fold_to_absolute(
        zonal_wind.transpose("pres", "latitude"),
        "even",
    )
    folded_psi = fold_to_absolute(psi.transpose("pres", "latitude"), "odd")
    folded_tropopause = fold_to_absolute(tropopause, "even")
    folded_torque = fold_to_absolute(
        mean_meridional_flow_torque(
            zonal_wind,
            meridional_wind,
        ).transpose("pres", "latitude"),
        "even",
    )
    upper_wind = pressure_mean(folded_wind, 100.0, 400.0)
    wind_850 = folded_wind.sortby("pres").interp(pres=850.0)
    core_layer = subset_between(folded_wind, "pres", (100.0, 400.0))
    return xr.Dataset(
        {
            "streamfunction": folded_psi,
            "zonal_wind": folded_wind,
            "tropopause_pressure": folded_tropopause,
            "mean_meridional_flow_torque": folded_torque,
            "upper_zonal_wind": upper_wind,
            "zonal_wind_850": wind_850,
            "upper_mean_meridional_flow_torque": pressure_mean(
                folded_torque,
                MOMENTUM_BALANCE_TOP_HPA,
                MOMENTUM_BALANCE_BOTTOM_HPA,
            ),
            "upper_wind_max_pressure": core_layer.idxmax("pres", skipna=True),
            "upper_wind_max_strength": core_layer.max("pres", skipna=True),
        }
    )


def _time_mean(dataset: xr.Dataset) -> xr.Dataset:
    return dataset.mean("Time", skipna=True).load()


def test_time_window(
    dataset: xr.Dataset,
    simulation: str,
    test_times: int,
    test_start: str | None,
) -> xr.Dataset:
    """Select a short contiguous window from the actual analysis record."""

    time_index = dataset.Time.to_index()
    start_index = 0
    if test_start is not None:
        requested = pd.Timestamp(test_start)
        candidates = np.flatnonzero(time_index >= requested)
        if candidates.size == 0:
            raise ValueError(
                f"{simulation}: --test-start {test_start!r} is after the "
                "analysis record"
            )
        start_index = int(candidates[0])
    stop_index = start_index + int(test_times)
    if stop_index > len(time_index):
        available = len(time_index) - start_index
        raise ValueError(
            f"{simulation}: test mode requests {test_times} times, but only "
            f"{available} are available"
        )
    selected = time_index[start_index:stop_index]
    progress(
        f"{simulation}: test window uses {len(selected)} times from "
        f"{selected[0]} to {selected[-1]}"
    )
    return dataset.sel(Time=selected)


def diagnostic_block_slices(
    field: xr.DataArray,
    test: bool,
) -> list[slice]:
    if test:
        return [slice(0, field.sizes["Time"])]
    return time_block_slices(
        field,
        block_days=BLOCK_DAYS,
        complete=True,
        min_blocks=MIN_BLOCKS,
        require_time=True,
    )


def load_zonal_dataset(
    directory: str,
    simulation: str = "simulation",
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
) -> xr.Dataset:
    raw = load_latlon_dataset(
        ROOT_DIR / directory,
        latlon_pattern=ZONAL_MEAN_PATTERN,
        chunks={"Time": 40},
    )
    dataset = standardize_latlon_dataset(
        raw,
        ["uzonal_isobaric", "umeridional_isobaric", "t_isobaric"],
    )
    if test:
        dataset = test_time_window(
            dataset,
            simulation,
            test_times,
            test_start,
        )
    return dataset


def _unit_key(units: str) -> str:
    """Return a compact unit spelling for archive validation."""

    return (
        str(units)
        .lower()
        .replace(" ", "")
        .replace("{", "")
        .replace("}", "")
        .replace("^", "")
        .replace("**", "")
    )


def validate_tem_archive(dataset: xr.Dataset) -> None:
    """Validate the archived TEM variables, dimensions, and acceleration units."""

    missing = [name for name in TEM_ARCHIVE_VARIABLES if name not in dataset]
    if missing:
        raise KeyError(
            "Required archived TEM variables are missing: " + ", ".join(missing)
        )
    expected_dimensions = ("Time", "z", "latitude")
    for name in TEM_ARCHIVE_VARIABLES:
        if dataset[name].dims != expected_dimensions:
            raise ValueError(
                f"{name!r} uses {dataset[name].dims!r}; expected "
                f"{expected_dimensions!r}"
            )
    expected_units = {
        "uReconstructZonal": "ms-1",
        "uReconstructMeridional": "ms-1",
        "pressure": "pa",
        "rho_zm": "kgm-3",
        "vres": "ms-1",
        "wres": "ms-1",
        "EPFd": "ms-2",
        "EPFd_y": "ms-2",
        "EPFd_z": "ms-2",
        "Fy": "kgs-2",
        "Fz": "kgs-2",
    }
    for name, expected in expected_units.items():
        archived = _unit_key(dataset[name].attrs.get("units", ""))
        if archived != expected:
            raise ValueError(
                f"{name!r} uses units {dataset[name].attrs.get('units')!r}; "
                f"expected {expected!r}"
            )
    epfd_long_name = dataset.EPFd.attrs.get("long_name", "").lower()
    if "acceleration due to ep flux convergence" not in epfd_long_name:
        raise ValueError(
            "EPFd metadata must identify acceleration due to EP flux convergence"
        )
    z = np.asarray(dataset.z.values, dtype=float)
    if z.ndim != 1 or z.size < 3 or not np.all(np.diff(z) > 0.0):
        raise ValueError("TEM height coordinate must increase on at least three levels")
    if float(np.nanmax(z)) < 1_000.0:
        raise ValueError("TEM height coordinate must use metres")


def load_tem_dataset(
    directory: str,
    simulation: str = "simulation",
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
) -> xr.Dataset:
    """Load only archived height-coordinate TEM and mean-flow fields."""

    raw = load_latlon_dataset(
        ROOT_DIR / directory,
        latlon_pattern=ZONAL_MEAN_PATTERN,
        chunks={"Time": 40, "z": -1, "latitude": -1},
    )
    dataset = standardize_latlon_dataset(raw, TEM_ARCHIVE_VARIABLES)
    if test:
        dataset = test_time_window(
            dataset,
            simulation,
            test_times,
            test_start,
        )
    validate_tem_archive(dataset)
    dataset.z.attrs.update(
        units="m",
        long_name="geometric height above the model lower boundary",
    )
    return dataset


def differentiate_physical_coordinate(
    field: xr.DataArray,
    dimension: str,
    coordinate_values: np.ndarray,
) -> xr.DataArray:
    """Differentiate against one coordinate expressed in physical units."""

    coordinate = np.asarray(coordinate_values, dtype=np.float64)
    if coordinate.shape != (field.sizes[dimension],):
        raise ValueError(
            f"Derivative coordinate for {dimension!r} has shape "
            f"{coordinate.shape!r}"
        )
    if coordinate.size < 2 or not np.all(np.diff(coordinate) > 0.0):
        raise ValueError(f"Derivative coordinate {dimension!r} must increase")
    edge_order = 2 if coordinate.size >= 3 else 1
    original_coordinate = field[dimension]
    working = field
    if hasattr(working.data, "chunks"):
        working = working.chunk({dimension: -1})
    derivative = working.assign_coords(
        {dimension: coordinate}
    ).differentiate(dimension, edge_order=edge_order)
    return derivative.assign_coords(
        {dimension: original_coordinate}
    ).transpose(*field.dims)


def tem_absolute_vorticity_factor(
    zonal_wind: xr.DataArray,
    radius_m: float = TEM_EARTH_RADIUS_M,
    angular_velocity_s1: float = TEM_ANGULAR_VELOCITY_S1,
) -> xr.DataArray:
    """Return the TEM absolute-vorticity factor multiplying residual v."""

    latitude_radians = np.deg2rad(
        np.asarray(zonal_wind.latitude.values, dtype=np.float64)
    )
    latitude = xr.DataArray(
        latitude_radians,
        dims="latitude",
        coords={"latitude": zonal_wind.latitude},
    )
    wind = zonal_wind.astype(np.float64)
    derivative = differentiate_physical_coordinate(
        wind,
        "latitude",
        latitude_radians,
    )
    factor = (
        2.0 * angular_velocity_s1 * np.sin(latitude)
        - (derivative - np.tan(latitude) * wind) / radius_m
    )
    factor.name = "tem_absolute_vorticity_factor"
    factor.attrs.update(
        units="s-1",
        long_name="TEM absolute-vorticity factor",
        formula="f - d(u*cos(latitude))/d(latitude)/(a*cos(latitude))",
        implementation=(
            "exact discrete m_dy expression from compute_tem_latlon_to_zm.py"
        ),
    )
    return factor


def archived_zonal_wind_tendency(zonal_wind: xr.DataArray) -> xr.DataArray:
    """Calculate du/dt from the archived time series in SI units."""

    time = zonal_wind.Time.values
    if not np.issubdtype(time.dtype, np.datetime64):
        raise TypeError("Archived wind tendency needs decoded datetime values")
    seconds = np.asarray(
        (time - time[0]) / np.timedelta64(1, "s"),
        dtype=np.float64,
    )
    tendency = differentiate_physical_coordinate(
        zonal_wind.astype(np.float64),
        "Time",
        seconds,
    )
    tendency.name = "a_tendency"
    tendency.attrs.update(
        units="m s-2",
        long_name="archived zonal-mean wind tendency",
        derivative=(
            "centred finite difference with one-sided archive endpoints"
        ),
        sign_convention="positive for eastward acceleration",
    )
    return tendency


def _reverse_cumulative_trapezoid(
    values: np.ndarray,
    height_m: np.ndarray,
) -> np.ndarray:
    intervals = 0.5 * (values[..., :-1] + values[..., 1:])
    intervals = intervals * np.diff(height_m)
    output = np.zeros_like(values, dtype=np.float64)
    output[..., :-1] = np.cumsum(intervals[..., ::-1], axis=-1)[..., ::-1]
    return output


def mass_streamfunction_from_height_transport(
    meridional_wind: xr.DataArray,
    density: xr.DataArray,
    radius_m: float = TEM_EARTH_RADIUS_M,
) -> xr.DataArray:
    """Integrate height-coordinate meridional mass transport from the top."""

    meridional_wind, density = xr.align(
        meridional_wind.astype(np.float64),
        density.astype(np.float64),
        join="exact",
    )
    height = np.asarray(meridional_wind.z.values, dtype=np.float64)
    transport = density * meridional_wind
    integral = xr.apply_ufunc(
        _reverse_cumulative_trapezoid,
        transport,
        xr.DataArray(height, dims="z", coords={"z": meridional_wind.z}),
        input_core_dims=[["z"], ["z"]],
        output_core_dims=[["z"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    integral = integral.assign_coords(z=meridional_wind.z).transpose(
        *meridional_wind.dims
    )
    latitude = xr.DataArray(
        np.deg2rad(np.asarray(meridional_wind.latitude.values, dtype=float)),
        dims="latitude",
        coords={"latitude": meridional_wind.latitude},
    )
    streamfunction = (
        2.0 * np.pi * radius_m * np.cos(latitude) * integral
    ).transpose(*meridional_wind.dims)
    streamfunction.name = "mass_streamfunction"
    streamfunction.attrs.update(
        units="kg s-1",
        integration="from each height to the archive top",
        top_boundary="zero mass streamfunction",
        sign_convention=(
            "positive for northward mass transport below in the NH; "
            "matches the Eulerian pressure-coordinate convention"
        ),
    )
    return streamfunction


def mask_spatial_derivative_boundaries(field: xr.DataArray) -> xr.DataArray:
    """Mask one-sided spatial-derivative points at every archive boundary."""

    condition: xr.DataArray | bool = True
    for dimension in ("z", "latitude"):
        if dimension not in field.dims:
            continue
        coordinate = field[dimension]
        interior = xr.DataArray(
            np.arange(coordinate.size),
            dims=dimension,
            coords={dimension: coordinate},
        )
        condition = condition & (interior > 0) & (
            interior < coordinate.size - 1
        )
    output = field.where(condition)
    output.attrs.update(field.attrs)
    output.attrs["boundary_handling"] = (
        "first and last height and latitude points excluded"
    )
    return output


def mask_nonfinite_tem_values(field: xr.DataArray) -> xr.DataArray:
    """Mask non-finite TEM values before any cache calculation."""

    output = field.where(np.isfinite(field))
    output.attrs.update(field.attrs)
    output.attrs["nonfinite_handling"] = "non-finite values masked"
    return output


def tem_instantaneous_diagnostics(dataset: xr.Dataset) -> xr.Dataset:
    """Build the archived-resolution TEM circulation and momentum budget."""

    validate_tem_archive(dataset)
    zonal_wind = mask_nonfinite_tem_values(
        dataset.uReconstructZonal.astype(np.float64)
    )
    meridional_wind = mask_nonfinite_tem_values(
        dataset.uReconstructMeridional.astype(np.float64)
    )
    density = mask_nonfinite_tem_values(dataset.rho_zm.astype(np.float64))
    pressure = mask_nonfinite_tem_values(dataset.pressure.astype(np.float64))
    vres = mask_nonfinite_tem_values(dataset.vres.astype(np.float64))
    wres = mask_nonfinite_tem_values(dataset.wres.astype(np.float64))
    a_ep = mask_nonfinite_tem_values(dataset.EPFd.astype(np.float64))
    a_ep_meridional = mask_nonfinite_tem_values(
        dataset.EPFd_y.astype(np.float64)
    )
    a_ep_vertical = mask_nonfinite_tem_values(
        dataset.EPFd_z.astype(np.float64)
    )
    ep_flux_meridional = mask_nonfinite_tem_values(
        dataset.Fy.astype(np.float64)
    )
    ep_flux_vertical = mask_nonfinite_tem_values(
        dataset.Fz.astype(np.float64)
    )
    height = np.asarray(dataset.z.values, dtype=np.float64)
    absolute_vorticity = tem_absolute_vorticity_factor(zonal_wind)
    vertical_shear = differentiate_physical_coordinate(zonal_wind, "z", height)

    a_vres = mask_nonfinite_tem_values(vres * absolute_vorticity)
    a_wres = mask_nonfinite_tem_values(-wres * vertical_shear)
    a_mmc = mask_nonfinite_tem_values(a_vres + a_wres)
    a_tendency = mask_nonfinite_tem_values(
        archived_zonal_wind_tendency(zonal_wind)
    )
    unresolved = mask_nonfinite_tem_values(
        a_tendency - (a_mmc + a_ep)
    )
    ep_closure = mask_nonfinite_tem_values(
        a_ep - (a_ep_meridional + a_ep_vertical)
    )
    residual_streamfunction = mask_nonfinite_tem_values(
        mass_streamfunction_from_height_transport(vres, density)
    )
    eulerian_streamfunction = mask_nonfinite_tem_values(
        mass_streamfunction_from_height_transport(meridional_wind, density)
    )

    diagnostics = xr.Dataset(
        {
            "residual_streamfunction": residual_streamfunction,
            "eulerian_streamfunction_height": eulerian_streamfunction,
            "vres": vres,
            "wres": wres,
            "pressure_hpa": pressure / 100.0,
            "a_vres": a_vres,
            "a_wres": a_wres,
            "a_mmc": a_mmc,
            "a_ep": a_ep,
            "a_ep_meridional": a_ep_meridional,
            "a_ep_vertical": a_ep_vertical,
            "ep_flux_meridional": ep_flux_meridional,
            "ep_flux_vertical": ep_flux_vertical,
            "a_tendency": a_tendency,
            "unresolved_direct_momentum_residual": unresolved,
            "ep_component_closure": ep_closure,
        }
    )
    diagnostics.residual_streamfunction.attrs["long_name"] = (
        "TEM residual mass streamfunction"
    )
    diagnostics.eulerian_streamfunction_height.attrs["long_name"] = (
        "Eulerian height-coordinate mass streamfunction"
    )
    diagnostics.vres.attrs.update(
        units="m s-1",
        long_name="TEM residual meridional velocity",
        sign_convention="positive northward",
    )
    diagnostics.wres.attrs.update(
        units="m s-1",
        long_name="TEM residual vertical velocity",
        sign_convention="positive upward",
    )
    diagnostics.pressure_hpa.attrs.update(
        units="hPa",
        long_name="archived pressure on height levels",
    )
    acceleration_metadata = {
        "units": "m s-2",
        "sign_convention": "positive for eastward acceleration",
    }
    diagnostics.a_vres.attrs.update(
        acceleration_metadata,
        long_name="residual-meridional-circulation acceleration",
        formula="vres * TEM absolute-vorticity factor",
        discretization=(
            "uses the exact m_dy finite-difference expression from the TEM producer"
        ),
    )
    diagnostics.a_wres.attrs.update(
        acceleration_metadata,
        long_name="residual-vertical-circulation acceleration",
        formula="-wres * du/dz",
    )
    diagnostics.a_mmc.attrs.update(
        acceleration_metadata,
        long_name="resolved TEM residual-circulation acceleration",
        formula="A_vres + A_wres",
    )
    diagnostics.a_ep.attrs.update(
        acceleration_metadata,
        long_name="resolved EP-flux-convergence acceleration",
        source="archived EPFd copied without sign reversal",
    )
    diagnostics.a_ep_meridional.attrs.update(
        acceleration_metadata,
        long_name="meridional EP-flux-convergence acceleration",
        source="archived EPFd_y copied without sign reversal",
    )
    diagnostics.a_ep_vertical.attrs.update(
        acceleration_metadata,
        long_name="vertical EP-flux-convergence acceleration",
        source="archived EPFd_z copied without sign reversal",
    )
    diagnostics.ep_flux_meridional.attrs.update(
        units="kg s-2",
        long_name="meridional Eliassen-Palm flux",
        source="archived Fy copied without reconstruction",
        sign_convention="positive northward",
    )
    diagnostics.ep_flux_vertical.attrs.update(
        units="kg s-2",
        long_name="vertical Eliassen-Palm flux",
        source="archived Fz copied without reconstruction",
        sign_convention="positive upward",
    )
    diagnostics.unresolved_direct_momentum_residual.attrs.update(
        acceleration_metadata,
        long_name="unresolved/direct zonal-momentum-budget residual",
        formula="du/dt - (A_MMC + A_EP)",
        interpretation=(
            "contains direct nudging, parameterized forcing, numerical effects, "
            "unresolved temporal variability, and closure error"
        ),
        warning="This residual is not a diagnosed nudging tendency.",
    )
    diagnostics.ep_component_closure.attrs.update(
        acceleration_metadata,
        long_name="EPFd component closure error",
        formula="EPFd - (EPFd_y + EPFd_z)",
    )
    diagnostics.attrs.update(
        tem_formulation="height-coordinate TEM from compute_tem_latlon_to_zm.py",
        archived_tem_reused=1,
        nonfinite_handling=(
            "non-finite archived and derived values masked before averaging"
        ),
    )
    return diagnostics


def height_layer_mean(
    field: xr.DataArray,
    pressure_hpa: xr.DataArray,
    top_hpa: float = TEM_STJ_LAYER_TOP_HPA,
    bottom_hpa: float = TEM_STJ_LAYER_BOTTOM_HPA,
) -> xr.DataArray:
    """Return a height-weighted mean inside a pressure-defined layer."""

    field, pressure_hpa = xr.align(field, pressure_hpa, join="exact")
    height_weights = xr.DataArray(
        np.gradient(np.asarray(field.z.values, dtype=np.float64)),
        dims="z",
        coords={"z": field.z},
    )
    layer = (pressure_hpa >= top_hpa) & (pressure_hpa <= bottom_hpa)
    valid_weights = height_weights.where(layer & field.notnull())
    profile = (field * valid_weights).sum("z", skipna=True) / valid_weights.sum(
        "z", skipna=True
    )
    profile.attrs.update(field.attrs)
    profile.attrs.update(
        vertical_average="height weighted within archived mean pressure bounds",
        pressure_bounds_hpa=f"{top_hpa:g}-{bottom_hpa:g}",
    )
    return profile


def tem_climatology_fields(diagnostics: xr.Dataset) -> xr.Dataset:
    """Fold time-mean TEM fields and cache the STJ-layer profiles."""

    mean = diagnostics.mean("Time", skipna=True, keep_attrs=True)
    parities = {
        "residual_streamfunction": "odd",
        "eulerian_streamfunction_height": "odd",
        "vres": "odd",
        "wres": "even",
        "pressure_hpa": "even",
        "a_vres": "even",
        "a_wres": "even",
        "a_mmc": "even",
        "a_ep": "even",
        "a_ep_meridional": "even",
        "a_ep_vertical": "even",
        "ep_flux_meridional": "odd",
        "ep_flux_vertical": "even",
        "a_tendency": "even",
        "unresolved_direct_momentum_residual": "even",
        "ep_component_closure": "even",
    }
    boundary_fields = set(parities) - {"pressure_hpa"}
    folded = {}
    for name, parity in parities.items():
        field = mean[name].transpose("z", "latitude")
        if name in boundary_fields:
            field = mask_spatial_derivative_boundaries(field)
        folded[name] = fold_to_absolute(field, parity)
    fields = xr.Dataset(folded)
    for name in TEM_BUDGET_COMPONENTS:
        profile_name = f"stj_layer_{name}"
        fields[profile_name] = height_layer_mean(
            fields[name],
            fields.pressure_hpa,
        )
        fields[profile_name].attrs["long_name"] = (
            f"STJ-layer latitude profile of {fields[name].attrs['long_name']}"
        )
    fields.z.attrs.update(units="m", long_name="geometric height")
    fields.attrs.update(
        spatial_hemisphere_treatment="absolute-latitude fold",
        residual_streamfunction_parity="odd",
        vres_parity="odd",
        wres_parity="even",
        momentum_acceleration_parity="even",
        stj_profile_pressure_bounds_hpa=(
            f"{TEM_STJ_LAYER_TOP_HPA:g}-{TEM_STJ_LAYER_BOTTOM_HPA:g}"
        ),
    )
    return fields


def _tem_region_weights(field: xr.DataArray) -> xr.DataArray:
    latitude_weights = np.cos(np.deg2rad(field.absolute_latitude))
    height_weights = xr.DataArray(
        np.gradient(np.asarray(field.z.values, dtype=np.float64)),
        dims="z",
        coords={"z": field.z},
    )
    return latitude_weights * height_weights


def _weighted_field_statistics(
    field: xr.DataArray,
    mask: xr.DataArray,
) -> dict[str, xr.DataArray]:
    weights = _tem_region_weights(field).where(mask & field.notnull())
    denominator = weights.sum(("z", "absolute_latitude"), skipna=True)
    signed_mean = (field * weights).sum(
        ("z", "absolute_latitude"), skipna=True
    ) / denominator
    mean_absolute = (np.abs(field) * weights).sum(
        ("z", "absolute_latitude"), skipna=True
    ) / denominator
    rms = np.sqrt(
        (field**2 * weights).sum(
            ("z", "absolute_latitude"), skipna=True
        )
        / denominator
    )
    maximum_absolute = np.abs(field.where(mask)).max(
        ("z", "absolute_latitude"), skipna=True
    )
    return {
        "signed_mean": signed_mean,
        "mean_absolute": mean_absolute,
        "rms": rms,
        "maximum_absolute": maximum_absolute,
    }


def _weighted_pattern_correlation(
    first: xr.DataArray,
    second: xr.DataArray,
    mask: xr.DataArray,
) -> xr.DataArray:
    first, second = xr.align(first, second, join="exact")
    weights = _tem_region_weights(first).where(
        mask & first.notnull() & second.notnull()
    )
    denominator = weights.sum(("z", "absolute_latitude"), skipna=True)
    first_mean = (first * weights).sum(
        ("z", "absolute_latitude"), skipna=True
    ) / denominator
    second_mean = (second * weights).sum(
        ("z", "absolute_latitude"), skipna=True
    ) / denominator
    first_anomaly = first - first_mean
    second_anomaly = second - second_mean
    covariance = (first_anomaly * second_anomaly * weights).sum(
        ("z", "absolute_latitude"), skipna=True
    )
    first_variance = (first_anomaly**2 * weights).sum(
        ("z", "absolute_latitude"), skipna=True
    )
    second_variance = (second_anomaly**2 * weights).sum(
        ("z", "absolute_latitude"), skipna=True
    )
    return covariance / np.sqrt(first_variance * second_variance)


def tem_budget_metrics(fields: xr.Dataset) -> xr.Dataset:
    """Summarize budget magnitudes in the upper-tropospheric STJ region."""

    latitude = fields.absolute_latitude
    pressure = fields.pressure_hpa
    lat_min, lat_max = TEM_STJ_LATITUDE_BOUNDS
    region = (
        (latitude >= lat_min)
        & (latitude <= lat_max)
        & (pressure >= TEM_STJ_LAYER_TOP_HPA)
        & (pressure <= TEM_STJ_LAYER_BOTTOM_HPA)
    )
    statistics = {
        name: []
        for name in ("signed_mean", "mean_absolute", "rms", "maximum_absolute")
    }
    for component in TEM_BUDGET_COMPONENTS:
        values = _weighted_field_statistics(fields[component], region)
        for statistic, value in values.items():
            statistics[statistic].append(value)
    metrics = xr.Dataset(
        {
            statistic: xr.concat(
                values,
                dim=xr.IndexVariable(
                    "component",
                    list(TEM_BUDGET_COMPONENTS),
                ),
            )
            for statistic, values in statistics.items()
        }
    )
    closure = _weighted_field_statistics(fields.ep_component_closure, region)[
        "rms"
    ]
    ep_rms = metrics.rms.sel(component="a_ep")
    metrics["ep_component_closure_rms"] = closure
    metrics["ep_component_closure_relative_rms"] = closure / xr.where(
        ep_rms > 0.0,
        ep_rms,
        np.nan,
    )
    hadley_region = (
        (latitude >= 5.0)
        & (latitude <= 35.0)
        & (pressure >= 300.0)
        & (pressure <= 700.0)
    )
    metrics["residual_eulerian_streamfunction_pattern_correlation"] = (
        _weighted_pattern_correlation(
            fields.residual_streamfunction,
            fields.eulerian_streamfunction_height,
            hadley_region,
        )
    )
    for name in ("signed_mean", "mean_absolute", "rms", "maximum_absolute"):
        metrics[name].attrs.update(
            units="m s-2",
            region="upper-tropospheric STJ region",
        )
    metrics.ep_component_closure_rms.attrs["units"] = "m s-2"
    metrics.ep_component_closure_relative_rms.attrs["units"] = "1"
    metrics.residual_eulerian_streamfunction_pattern_correlation.attrs.update(
        units="1",
        region="5-35 degrees absolute latitude and 300-700 hPa",
        interpretation=(
            "positive values indicate similar residual and Eulerian patterns"
        ),
    )
    metrics.attrs.update(
        region=(
            f"{lat_min:g}-{lat_max:g} degrees absolute latitude and "
            f"{TEM_STJ_LAYER_TOP_HPA:g}-{TEM_STJ_LAYER_BOTTOM_HPA:g} hPa"
        ),
        residual_interpretation=(
            "unresolved/direct momentum-budget residual; not nudging tendency"
        ),
    )
    return metrics


def build_tem_products(
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
    simulations: Sequence[tuple[str, str]] = SIMULATIONS,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Build compact TEM circulation, budget, and block-summary products."""

    simulation_names = [item[0] for item in simulations]
    field_records = []
    metric_records = []
    block_records = []
    for simulation, directory in simulations:
        progress(f"{simulation}: loading archived height-coordinate TEM fields")
        dataset = load_tem_dataset(
            directory,
            simulation=simulation,
            test=test,
            test_times=test_times,
            test_start=test_start,
        )
        try:
            diagnostics = tem_instantaneous_diagnostics(dataset)
            fields = tem_climatology_fields(diagnostics)
            metrics = tem_budget_metrics(fields)
            slices = diagnostic_block_slices(
                dataset.uReconstructZonal,
                test=test,
            )
            block_field_items = [
                tem_climatology_fields(diagnostics.isel(Time=block_slice))
                for block_slice in slices
            ]
            block_fields = xr.concat(
                [item[list(TEM_BOOTSTRAP_FIELDS)] for item in block_field_items],
                dim=xr.IndexVariable("block", np.arange(len(slices))),
            )
            block_metrics = xr.concat(
                [tem_budget_metrics(item) for item in block_field_items],
                dim=xr.IndexVariable("block", np.arange(len(slices))),
            )
            blocks = xr.merge([block_metrics, block_fields])
            progress(f"{simulation}: calculating compact TEM budget products")
            with dask.config.set(scheduler="threads", num_workers=2):
                fields, metrics, blocks = dask.compute(fields, metrics, blocks)
            relative_closure = float(
                metrics.ep_component_closure_relative_rms.values
            )
            if not np.isfinite(relative_closure) or relative_closure > 1.0e-4:
                raise ValueError(
                    f"{simulation}: archived EPFd components do not close; "
                    f"relative RMS is {relative_closure:.3g}"
                )
            field_records.append(fields)
            metric_records.append(metrics)
            block_records.append(blocks)
        finally:
            dataset.close()

    fields = xr.concat(
        field_records,
        dim=xr.IndexVariable("simulation", simulation_names),
        combine_attrs="drop_conflicts",
    )
    metrics = xr.concat(
        metric_records,
        dim=xr.IndexVariable("simulation", simulation_names),
        combine_attrs="drop_conflicts",
    )
    metric_blocks = xr.concat(
        block_records,
        dim=xr.IndexVariable("simulation", simulation_names),
        combine_attrs="drop_conflicts",
    )
    common_attrs = {
        "test_subset": int(test),
        "source": "archived zm_latlon TEM fields",
        "unresolved_residual_is_nudging_tendency": 0,
        "nonfinite_handling": (
            "non-finite archived and derived values masked before averaging"
        ),
    }
    fields.attrs.update(common_attrs)
    metrics.attrs.update(common_attrs)
    metric_blocks.attrs.update(
        common_attrs,
        block_definition=(
            "complete bounded test window"
            if test
            else f"complete {BLOCK_DAYS:g}-day block"
        ),
    )
    if not test:
        metric_blocks.attrs["block_days"] = BLOCK_DAYS
    return fields, metrics, metric_blocks


def build_circulation_products(
    pytropd_api: Any | None = None,
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
    simulations: Sequence[tuple[str, str]] = SIMULATIONS,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    api = pytropd_api or load_pytropd()
    simulation_names = [item[0] for item in simulations]
    field_records = []
    metric_records = []
    block_records = []
    block_field_records = []
    for simulation, directory in simulations:
        progress(f"{simulation}: loading zonal-mean circulation fields")
        dataset = load_zonal_dataset(
            directory,
            simulation=simulation,
            test=test,
            test_times=test_times,
            test_start=test_start,
        )
        try:
            mean = _time_mean(dataset)
            metrics, psi, tropopause = circulation_metric_values(
                mean.uzonal_isobaric,
                mean.umeridional_isobaric,
                mean.t_isobaric,
                pytropd_api=api,
            )
            metric_records.append(metrics)
            field_records.append(
                circulation_fields_for_mean(
                    mean.uzonal_isobaric,
                    mean.umeridional_isobaric,
                    psi,
                    tropopause,
                ).load()
            )

            blocks = []
            block_fields = []
            slices = diagnostic_block_slices(
                dataset.uzonal_isobaric,
                test=test,
            )
            for block_index, block_slice in enumerate(slices):
                progress(
                    f"{simulation}: diagnosing circulation block "
                    f"{block_index + 1}/{len(slices)}"
                )
                block = _time_mean(dataset.isel(Time=block_slice))
                block_metrics, block_psi, block_tropopause = circulation_metric_values(
                    block.uzonal_isobaric,
                    block.umeridional_isobaric,
                    block.t_isobaric,
                    pytropd_api=api,
                )
                blocks.append(block_metrics)
                block_fields.append(
                    circulation_fields_for_mean(
                        block.uzonal_isobaric,
                        block.umeridional_isobaric,
                        block_psi,
                        block_tropopause,
                    )[
                        [
                            "streamfunction",
                            "zonal_wind",
                            "upper_zonal_wind",
                            "zonal_wind_850",
                            "upper_mean_meridional_flow_torque",
                        ]
                    ]
                )
            block_records.append(blocks)
            block_field_records.append(
                xr.concat(
                    block_fields,
                    dim=xr.IndexVariable("block", np.arange(len(block_fields))),
                ).load()
            )
        finally:
            dataset.close()

    fields = xr.concat(
        field_records,
        dim=xr.IndexVariable("simulation", simulation_names),
        combine_attrs="drop_conflicts",
    )
    metrics = make_metric_dataset(simulation_names, metric_records)
    metric_blocks = make_metric_block_dataset(simulation_names, block_records)
    profile_blocks = xr.concat(
        block_field_records,
        dim=xr.IndexVariable("simulation", simulation_names),
        combine_attrs="drop_conflicts",
    )
    metric_blocks = xr.merge([metric_blocks, profile_blocks])
    fields.attrs.update(
        spatial_hemisphere_treatment="absolute-latitude fold",
        streamfunction_parity="odd",
        zonal_wind_parity="even",
        mean_meridional_flow_torque_parity="even",
        momentum_torque_sign_convention="positive for eastward acceleration",
        tropopause_parity="even",
        test_subset=int(test),
    )
    metrics.attrs["test_subset"] = int(test)
    if test:
        metric_blocks.attrs.pop("block_days", None)
    metric_blocks.attrs.update(
        test_subset=int(test),
        block_definition=(
            "complete bounded test window"
            if test
            else f"complete {BLOCK_DAYS:g}-day block"
        ),
    )
    return fields, metrics, metric_blocks


def instantaneous_eddy_momentum_flux(
    zonal_wind: xr.DataArray,
    meridional_wind: xr.DataArray,
) -> xr.DataArray:
    """Return zonal-mean u-prime-v-prime at each time."""

    u = zonal_wind.astype(np.float64)
    v = meridional_wind.astype(np.float64)
    u_mean = u.mean("longitude", skipna=True)
    v_mean = v.mean("longitude", skipna=True)
    flux = (u * v).mean("longitude", skipna=True) - u_mean * v_mean
    flux.name = "eddy_momentum_flux"
    flux.attrs.update(
        units="m2 s-2",
        long_name="zonal-mean eddy momentum flux",
        eddy_definition="departure from the instantaneous zonal mean",
    )
    return flux


def eddy_momentum_flux_convergence(
    flux: xr.DataArray,
    radius_m: float = EARTH_RADIUS_M,
) -> xr.DataArray:
    """Return spherical horizontal convergence of zonal-mean u-prime-v-prime."""

    if "latitude" not in flux.dims:
        raise ValueError("Eddy momentum flux needs a latitude dimension")
    latitude = np.asarray(flux.latitude.values, dtype=float)
    phi = np.deg2rad(latitude)
    cos_phi = np.cos(phi)
    axis = flux.get_axis_num("latitude")
    weighted = np.asarray(flux.values, dtype=float) * np.reshape(
        cos_phi**2,
        [latitude.size if index == axis else 1 for index in range(flux.ndim)],
    )
    gradient = np.gradient(weighted, phi, axis=axis, edge_order=2)
    denominator = radius_m * np.reshape(
        cos_phi**2,
        [latitude.size if index == axis else 1 for index in range(flux.ndim)],
    )
    values = -gradient / denominator
    output = xr.DataArray(values, dims=flux.dims, coords=flux.coords)
    output.name = "eddy_momentum_flux_convergence"
    output.attrs.update(
        units="m s-2",
        long_name="horizontal eddy momentum-flux convergence",
        formula="-1/(a cos(phi)^2) d(cos(phi)^2 u-prime-v-prime)/dphi",
        sign_convention="positive for eastward acceleration",
    )
    return output


def eddy_transport_position(
    upper_flux: xr.DataArray,
) -> dict[str, float]:
    values = {}
    for hemisphere in ("SH", "NH"):
        bounds = HEMISPHERE_BOUNDS[hemisphere]
        field = subset_between(upper_flux, "latitude", bounds)
        poleward = -field if hemisphere == "SH" else field
        poleward = subset_between(poleward, "latitude", (-70.0, 70.0))
        poleward = poleward.where(abs(poleward.latitude) >= 10.0, drop=True)
        if poleward.sizes.get("latitude", 0) == 0 or not np.isfinite(
            poleward.values
        ).any():
            values[hemisphere] = np.nan
            continue
        index = int(np.nanargmax(poleward.values))
        values[hemisphere] = float(abs(poleward.latitude.values[index]))
    return values


def eddy_block_fields(blocks: xr.DataArray) -> xr.Dataset:
    """Return folded momentum-flux fields for each complete time block."""

    folded_flux = fold_to_absolute(
        blocks.transpose("block", "pres", "latitude"),
        "odd",
    )
    folded_convergence = fold_to_absolute(
        eddy_momentum_flux_convergence(blocks).transpose(
            "block",
            "pres",
            "latitude",
        ),
        "even",
    )
    return xr.Dataset(
        {
            "eddy_momentum_flux": folded_flux,
            "eddy_momentum_flux_convergence": folded_convergence,
            "upper_eddy_momentum_flux": pressure_mean(
                folded_flux,
                200.0,
                250.0,
            ),
            "upper_eddy_momentum_flux_convergence": pressure_mean(
                folded_convergence,
                MOMENTUM_BALANCE_TOP_HPA,
                MOMENTUM_BALANCE_BOTTOM_HPA,
            ),
        }
    )


def eddy_fields_from_blocks(blocks: xr.DataArray) -> xr.Dataset:
    return eddy_block_fields(blocks).mean("block", skipna=True, keep_attrs=True)


def load_eddy_dataset(
    directory: str,
    simulation: str = "simulation",
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
) -> xr.Dataset:
    raw = load_latlon_dataset(
        ROOT_DIR / directory,
        latlon_pattern=LATLON_PATTERN,
        chunks={"Time": 4, "latitude": 180, "longitude": 360},
    )
    dataset = standardize_latlon_dataset(
        raw,
        ["uzonal_isobaric", "umeridional_isobaric"],
    )
    if test:
        dataset = test_time_window(
            dataset,
            simulation,
            test_times,
            test_start,
        )
        lon_min, lon_max = QUICK_TEST_LON_BOUNDS
        dataset = dataset.where(
            (dataset.longitude >= lon_min) & (dataset.longitude <= lon_max),
            drop=True,
        )
        progress(
            f"{simulation}: test eddy domain uses {lon_min:g} to "
            f"{lon_max:g} degrees longitude"
        )
    return dataset


def load_moisture_dataset(
    directory: str,
    simulation: str = "simulation",
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
) -> xr.Dataset:
    """Load only the archived fields needed for moisture transport."""

    raw = load_latlon_dataset(
        ROOT_DIR / directory,
        latlon_pattern=LATLON_PATTERN,
        chunks={"Time": 4, "latitude": 180, "longitude": 360},
    )
    dataset = standardize_latlon_dataset(
        raw,
        ["umeridional_isobaric", "qv_isobaric"],
    )
    if test:
        dataset = test_time_window(
            dataset,
            simulation,
            test_times,
            test_start,
        )
        lon_min, lon_max = QUICK_TEST_LON_BOUNDS
        dataset = dataset.where(
            (dataset.longitude >= lon_min)
            & (dataset.longitude <= lon_max),
            drop=True,
        )
        progress(
            f"{simulation}: test moisture domain uses {lon_min:g} to "
            f"{lon_max:g} degrees longitude"
        )
    return dataset


def moisture_flux_statistics(
    meridional_wind: xr.DataArray,
    vapour_mixing_ratio: xr.DataArray,
) -> xr.Dataset:
    """Return compact statistics for the zonal-mean moisture-flux split."""

    wind = meridional_wind.astype(np.float64)
    humidity = mixing_ratio_to_specific_humidity(
        vapour_mixing_ratio.astype(np.float64)
    )
    valid = np.isfinite(wind) & np.isfinite(humidity)
    wind = wind.where(valid)
    humidity = humidity.where(valid)
    mean_dimensions = tuple(
        dim for dim in ("Time", "longitude") if dim in wind.dims
    )
    if not mean_dimensions:
        raise ValueError("Moisture transport needs Time or longitude")

    statistics = xr.Dataset(
        {
            "total_meridional_moisture_flux": (wind * humidity).mean(
                mean_dimensions,
                skipna=True,
            ),
            "mean_meridional_wind": wind.mean(
                mean_dimensions,
                skipna=True,
            ),
            "mean_specific_humidity": humidity.mean(
                mean_dimensions,
                skipna=True,
            ),
        }
    )
    statistics["mean_meridional_moisture_flux"] = (
        statistics.mean_meridional_wind
        * statistics.mean_specific_humidity
    )
    statistics["eddy_meridional_moisture_flux"] = (
        statistics.total_meridional_moisture_flux
        - statistics.mean_meridional_moisture_flux
    )
    statistics.total_meridional_moisture_flux.attrs.update(
        units="m s-1",
        long_name="zonal-time-mean meridional moisture flux",
        sign_convention="positive northward",
    )
    statistics.mean_meridional_wind.attrs.update(
        units="m s-1",
        long_name="zonal-time-mean meridional wind",
        sign_convention="positive northward",
    )
    statistics.mean_specific_humidity.attrs.update(
        units="kg kg-1",
        long_name="zonal-time-mean specific humidity",
        source_conversion="qv_isobaric / (1 + qv_isobaric)",
    )
    statistics.mean_meridional_moisture_flux.attrs.update(
        units="m s-1",
        long_name="mean-circulation meridional moisture flux",
        definition="[v_bar] [q_bar]",
        sign_convention="positive northward",
    )
    statistics.eddy_meridional_moisture_flux.attrs.update(
        units="m s-1",
        long_name="eddy meridional moisture flux",
        definition="[overline(v q)] - [v_bar] [q_bar]",
        sign_convention="positive northward",
    )
    return statistics


def build_moisture_products(
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
    simulations: Sequence[tuple[str, str]] = SIMULATIONS,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build compact moisture-flux statistics for each simulation."""

    simulation_names = [item[0] for item in simulations]
    records = []
    block_records = []
    for simulation, directory in simulations:
        progress(f"{simulation}: loading winds and humidity for moisture transport")
        dataset = load_moisture_dataset(
            directory,
            simulation=simulation,
            test=test,
            test_times=test_times,
            test_start=test_start,
        )
        try:
            progress(f"{simulation}: calculating compact moisture-flux statistics")
            slices = diagnostic_block_slices(
                dataset.umeridional_isobaric,
                test=test,
            )
            fields = moisture_flux_statistics(
                dataset.umeridional_isobaric,
                dataset.qv_isobaric,
            )
            block_statistics = xr.concat(
                [
                    moisture_flux_statistics(
                        dataset.umeridional_isobaric.isel(Time=block_slice),
                        dataset.qv_isobaric.isel(Time=block_slice),
                    )
                    for block_slice in slices
                ],
                dim=xr.IndexVariable("block", np.arange(len(slices))),
            )
            with dask.config.set(scheduler="threads", num_workers=2):
                fields, block_statistics = dask.compute(
                    fields,
                    block_statistics,
                )
            records.append(fields)
            block_records.append(
                moisture_profile_block_fields(block_statistics).load()
            )
        finally:
            dataset.close()

    fields = xr.concat(
        records,
        dim=xr.IndexVariable("simulation", simulation_names),
        combine_attrs="drop_conflicts",
    )
    fields.attrs.update(
        humidity_definition=(
            "specific humidity converted from archived MPAS dry-air "
            "vapour mixing ratio"
        ),
        moisture_flux_decomposition=(
            "total = mean circulation + combined stationary/transient eddy"
        ),
        test_subset=int(test),
    )
    profile_blocks = xr.concat(
        block_records,
        dim=xr.IndexVariable("simulation", simulation_names),
        combine_attrs="drop_conflicts",
    )
    profile_blocks.attrs.update(
        test_subset=int(test),
        block_definition=(
            "complete bounded test window"
            if test
            else f"complete {BLOCK_DAYS:g}-day block"
        ),
    )
    if not test:
        profile_blocks.attrs["block_days"] = BLOCK_DAYS
    return fields, profile_blocks


def build_eddy_products(
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
    simulations: Sequence[tuple[str, str]] = SIMULATIONS,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    simulation_names = [item[0] for item in simulations]
    field_records = []
    metric_records = []
    block_metric_records = []
    block_profile_records = []
    for simulation, directory in simulations:
        progress(f"{simulation}: loading winds for eddy momentum transport")
        dataset = load_eddy_dataset(
            directory,
            simulation=simulation,
            test=test,
            test_times=test_times,
            test_start=test_start,
        )
        try:
            flux = instantaneous_eddy_momentum_flux(
                dataset.uzonal_isobaric,
                dataset.umeridional_isobaric,
            )
            slices = diagnostic_block_slices(
                flux,
                test=test,
            )
            progress(f"{simulation}: calculating {len(slices)} eddy blocks")
            with dask.config.set(scheduler="threads", num_workers=2):
                block_flux = xr.concat(
                    [
                        flux.isel(Time=item).mean("Time", skipna=True)
                        for item in slices
                    ],
                    dim=xr.IndexVariable("block", np.arange(len(slices))),
                ).load()
            block_fields = eddy_block_fields(block_flux)
            field_records.append(
                block_fields.mean("block", skipna=True, keep_attrs=True).load()
            )
            block_profile_records.append(
                block_fields[
                    [
                        "eddy_momentum_flux",
                        "eddy_momentum_flux_convergence",
                        "upper_eddy_momentum_flux",
                        "upper_eddy_momentum_flux_convergence",
                    ]
                ]
            )

            blocks = []
            for block_index in range(block_flux.sizes["block"]):
                upper = pressure_mean(
                    block_flux.isel(block=block_index, drop=True),
                    200.0,
                    250.0,
                )
                blocks.append(
                    {"eddy_flux_max_latitude": eddy_transport_position(upper)}
                )
            block_metric_records.append(blocks)
            mean_upper = pressure_mean(
                block_flux.mean("block", skipna=True),
                200.0,
                250.0,
            )
            metric_records.append(
                {"eddy_flux_max_latitude": eddy_transport_position(mean_upper)}
            )
        finally:
            dataset.close()

    fields = xr.concat(
        field_records,
        dim=xr.IndexVariable("simulation", simulation_names),
        combine_attrs="drop_conflicts",
    )
    metrics = make_metric_dataset(simulation_names, metric_records)
    metric_blocks = make_metric_block_dataset(
        simulation_names,
        block_metric_records,
    )
    profile_blocks = xr.concat(
        block_profile_records,
        dim=xr.IndexVariable("simulation", simulation_names),
        combine_attrs="drop_conflicts",
    )
    metric_blocks = xr.merge([metric_blocks, profile_blocks])
    fields.attrs.update(
        eddy_definition="departure from the instantaneous zonal mean",
        eddy_momentum_flux_parity="odd",
        eddy_momentum_flux_convergence_parity="even",
        test_subset=int(test),
    )
    metrics.attrs["test_subset"] = int(test)
    if test:
        metric_blocks.attrs.pop("block_days", None)
    metric_blocks.attrs.update(
        test_subset=int(test),
        block_definition=(
            "complete bounded test window"
            if test
            else f"complete {BLOCK_DAYS:g}-day block"
        ),
    )
    return fields, metrics, metric_blocks


def save_family(
    family: str,
    products: tuple[xr.Dataset, ...],
    manifest: dict,
    cache_root: Path | None = None,
) -> None:
    root = CACHE_ROOT if cache_root is None else cache_root
    root.mkdir(parents=True, exist_ok=True)
    if len(products) != len(CACHE_FILES[family]):
        raise ValueError(
            f"{family} produced {len(products)} cache files; "
            f"expected {len(CACHE_FILES[family])}"
        )
    for dataset, filename in zip(products, CACHE_FILES[family]):
        dataset = dataset.copy()
        dataset.attrs.update(
            cache_schema=CACHE_SCHEMA[family],
            manifest_sha256=manifest["manifest_sha256"],
            cache_manifest=json.dumps(manifest, sort_keys=True),
        )
        write_netcdf_atomic(dataset, cache_path(filename, root))
    write_json_file(str(manifest_path(family, root)), manifest)


def load_family(
    family: str,
    cache_root: Path | None = None,
) -> tuple[xr.Dataset, ...]:
    datasets = []
    for filename in CACHE_FILES[family]:
        datasets.append(load_netcdf(cache_path(filename, cache_root)))
    return tuple(datasets)


def migrate_simulation_family_from_aggregate(
    family: str,
    simulation: tuple[str, str],
    *,
    test: bool,
    test_times: int,
    test_start: str | None,
    validate_sources: bool,
    target_root: Path,
) -> bool:
    """Seed one simulation family from a current profile aggregate."""

    label, _directory = simulation
    canonical = canonical_simulation_name(label)
    for profile, names in SIMULATION_PROFILES.items():
        if canonical not in names:
            continue
        simulations = profile_simulations(profile)
        root = aggregate_cache_root(
            profile,
            test=test,
            test_times=test_times,
            test_start=test_start,
        )
        aggregate_manifest = family_manifest(
            family,
            test=test,
            test_times=test_times,
            test_start=test_start,
            include_sources=validate_sources,
            simulations=simulations,
        )
        if not family_cache_is_current(
            family,
            aggregate_manifest,
            root,
            validate_sources=validate_sources,
        ):
            continue
        selected = tuple(
            dataset.sel(simulation=[label])
            if "simulation" in dataset.dims
            else dataset
            for dataset in load_family(family, root)
        )
        manifest = family_manifest(
            family,
            test=test,
            test_times=test_times,
            test_start=test_start,
            simulations=(simulation,),
        )
        save_family(family, selected, manifest, target_root)
        progress(
            f"migrated {family} products for {label} from the "
            f"{profile} aggregate cache"
        )
        return True
    return False


def assemble_simulation_families(
    families: Sequence[tuple[xr.Dataset, ...]],
) -> tuple[xr.Dataset, ...]:
    """Combine reusable one-simulation products for profile plotting."""

    product_count = len(families[0])
    if any(len(products) != product_count for products in families):
        raise ValueError("Simulation caches contain different product counts")
    return tuple(
        xr.concat(
            [products[index] for products in families],
            dim="simulation",
            combine_attrs="drop_conflicts",
        )
        for index in range(product_count)
    )


def prepare_family(
    family: str,
    figures_only: bool,
    rebuild_cache: bool,
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
    cache_root: Path | None = None,
) -> tuple[xr.Dataset, ...]:
    del cache_root
    products_by_simulation = []
    missing = []
    for simulation in SIMULATIONS:
        label, _directory = simulation
        root = simulation_cache_root(
            label,
            test=test,
            test_times=test_times,
            test_start=test_start,
        )
        manifest = family_manifest(
            family,
            test=test,
            test_times=test_times,
            test_start=test_start,
            include_sources=not figures_only,
            simulations=(simulation,),
        )
        current = not rebuild_cache and family_cache_is_current(
            family,
            manifest,
            root,
            validate_sources=not figures_only,
        )
        if not current and not rebuild_cache:
            current = migrate_simulation_family_from_aggregate(
                family,
                simulation,
                test=test,
                test_times=test_times,
                test_start=test_start,
                validate_sources=not figures_only,
                target_root=root,
            )
        if current:
            progress(f"using current {family} cache for {label}")
            products_by_simulation.append(load_family(family, root))
        else:
            missing.append((simulation, root))

    if figures_only and missing:
        names = ", ".join(simulation[0] for simulation, _root in missing)
        raise FileNotFoundError(
            f"Figure-only mode requires current {family} caches for: {names}"
        )

    for simulation, root in missing:
        label, _directory = simulation
        progress(f"building {family} cache for {label}")
        if family == "circulation":
            products = build_circulation_products(
                test=test,
                test_times=test_times,
                test_start=test_start,
                simulations=(simulation,),
            )
        elif family == "eddy":
            products = build_eddy_products(
                test=test,
                test_times=test_times,
                test_start=test_start,
                simulations=(simulation,),
            )
        elif family == "moisture":
            products = build_moisture_products(
                test=test,
                test_times=test_times,
                test_start=test_start,
                simulations=(simulation,),
            )
        elif family == "tem":
            products = build_tem_products(
                test=test,
                test_times=test_times,
                test_start=test_start,
                simulations=(simulation,),
            )
        else:
            raise ValueError(f"Unknown circulation cache family: {family}")
        manifest = family_manifest(
            family,
            test=test,
            test_times=test_times,
            test_start=test_start,
            simulations=(simulation,),
        )
        save_family(family, products, manifest, root)
        products_by_simulation.append(products)

    by_label = {
        str(products[0].simulation.values[0]): products
        for products in products_by_simulation
    }
    ordered = [by_label[label] for label, _directory in SIMULATIONS]
    return assemble_simulation_families(ordered)


def combine_metric_products(
    products: Sequence[xr.Dataset],
) -> xr.Dataset:
    if len(products) == 1:
        return products[0]
    return xr.concat(products, dim="metric", combine_attrs="drop_conflicts")


def metrics_table(metrics: xr.Dataset) -> pd.DataFrame:
    units = {
        str(metric): str(unit)
        for metric, unit in zip(metrics.metric.values, metrics.metric_units.values)
    }
    methods = {
        str(metric): str(method)
        for metric, method in zip(metrics.metric.values, metrics.metric_method.values)
    }
    definitions = {
        str(metric): str(definition)
        for metric, definition in zip(
            metrics.metric.values,
            metrics.metric_definition.values,
        )
    }
    rows = []
    for simulation in metrics.simulation.values:
        for metric in metrics.metric.values:
            disagreement = float(
                metrics.hemisphere_disagreement.sel(
                    simulation=simulation,
                    metric=metric,
                ).values
            )
            for hemisphere in metrics.hemisphere.values:
                rows.append(
                    {
                        "simulation": str(simulation),
                        "hemisphere": str(hemisphere),
                        "metric": str(metric),
                        "value": float(
                            metrics.value.sel(
                                simulation=simulation,
                                hemisphere=hemisphere,
                                metric=metric,
                            ).values
                        ),
                        "units": units[str(metric)],
                        "method": methods[str(metric)],
                        "definition": definitions[str(metric)],
                        "hemisphere_disagreement": disagreement,
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_metric_response(
    run_values: np.ndarray,
    reference_values: np.ndarray,
    primary_response: float,
    rng: np.random.Generator,
) -> dict[str, float | int | bool | str]:
    run_values = np.asarray(run_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    run_values = run_values[np.isfinite(run_values)]
    reference_values = reference_values[np.isfinite(reference_values)]
    if run_values.size < MIN_BLOCKS or reference_values.size < MIN_BLOCKS:
        raise ValueError(
            f"Insufficient finite blocks ({run_values.size}, {reference_values.size})"
        )
    boot = np.empty(N_BOOTSTRAP, dtype=float)
    for index in range(N_BOOTSTRAP):
        run_index = rng.integers(0, run_values.size, run_values.size)
        reference_index = rng.integers(
            0,
            reference_values.size,
            reference_values.size,
        )
        boot[index] = np.nanmean(run_values[run_index]) - np.nanmean(
            reference_values[reference_index]
        )
    ci_low, ci_high = np.nanpercentile(boot, CI_PERCENTILES)
    noise = float(np.nanstd(boot, ddof=1))
    sign = np.sign(primary_response)
    sign_robustness = (
        np.nan if sign == 0 else float(np.nanmean(np.sign(boot) == sign))
    )
    independent_differences = (
        run_values[:, np.newaxis] - reference_values[np.newaxis, :]
    )
    block_sign = (
        np.nan
        if sign == 0
        else float(np.nanmean(np.sign(independent_differences) == sign))
    )
    excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
    return {
        "response": float(primary_response),
        "block_mean_response": float(
            np.nanmean(run_values) - np.nanmean(reference_values)
        ),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_noise": noise,
        "response_to_noise": primary_response / noise if noise > 0 else np.nan,
        "sign_robustness": sign_robustness,
        "independent_block_sign_probability": block_sign,
        "n_blocks_run": int(run_values.size),
        "n_blocks_reference": int(reference_values.size),
        "ci_excludes_zero": excludes_zero,
        "robust": bool(excludes_zero and sign_robustness >= ROBUST_SIGN_THRESHOLD),
        "bootstrap_replicates": N_BOOTSTRAP,
        "confidence_percentiles": (
            f"{CI_PERCENTILES[0]:g},{CI_PERCENTILES[1]:g}"
        ),
        "resampling_method": (
            "independent complete-block resampling by simulation"
        ),
    }


def metric_response_table(
    metrics: xr.Dataset,
    metric_blocks: xr.Dataset,
) -> pd.DataFrame:
    rows = []
    units = {
        str(metric): str(unit)
        for metric, unit in zip(metrics.metric.values, metrics.metric_units.values)
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for comparison, run_index, reference_index, label in COMPARISONS[1:]:
        run = str(metrics.simulation.values[run_index])
        reference = str(metrics.simulation.values[reference_index])
        for hemisphere in metrics.hemisphere.values:
            for metric in metrics.metric.values:
                run_value = float(
                    metrics.value.sel(
                        simulation=run,
                        hemisphere=hemisphere,
                        metric=metric,
                    ).values
                )
                reference_value = float(
                    metrics.value.sel(
                        simulation=reference,
                        hemisphere=hemisphere,
                        metric=metric,
                    ).values
                )
                run_blocks = metric_blocks.value.sel(
                    simulation=run,
                    hemisphere=hemisphere,
                    metric=metric,
                ).values
                reference_blocks = metric_blocks.value.sel(
                    simulation=reference,
                    hemisphere=hemisphere,
                    metric=metric,
                ).values
                stats = bootstrap_metric_response(
                    run_blocks,
                    reference_blocks,
                    run_value - reference_value,
                    rng,
                )
                row = {
                    "comparison": comparison,
                    "comparison_label": label,
                    "run": run,
                    "reference": reference,
                    "hemisphere": str(hemisphere),
                    "metric": str(metric),
                    "units": units[str(metric)],
                }
                row.update(stats)
                rows.append(row)
    return pd.DataFrame(rows)


def save_publication_products(
    output_directory: Path,
    metrics: xr.Dataset,
    metric_blocks: xr.Dataset,
    include_responses: bool = True,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    write_netcdf_atomic(metrics, output_directory / "circulation_metrics.nc")
    write_csv_atomic(
        metrics_table(metrics),
        output_directory / "circulation_metrics.csv",
        index=False,
    )
    if include_responses:
        write_csv_atomic(
            metric_response_table(metrics, metric_blocks),
            output_directory / "circulation_metric_responses.csv",
            index=False,
        )


def comparison_fields(
    fields: xr.Dataset,
    variable: str,
) -> list[xr.DataArray]:
    output = []
    for _name, run_index, reference_index, _label in COMPARISONS:
        run = fields[variable].isel(simulation=run_index, drop=True)
        if reference_index is None:
            output.append(run)
        else:
            reference = fields[variable].isel(
                simulation=reference_index,
                drop=True,
            )
            output.append((run - reference).load())
    return output


def bootstrap_field_response(
    block_fields: xr.Dataset,
    variable: str,
    run_index: int,
    reference_index: int,
    seed_offset: int,
) -> xr.Dataset:
    """Return a pointwise response interval from independent climate blocks."""

    def finite_blocks(field: xr.DataArray) -> tuple[np.ndarray, xr.DataArray]:
        field = field.transpose("block", ...)
        values = np.asarray(field.values, dtype=float)
        values = np.where(np.isfinite(values), values, np.nan)
        spatial_axes = tuple(range(1, values.ndim))
        valid = np.isfinite(values).any(axis=spatial_axes)
        return values[valid], field.isel(block=0, drop=True)

    run_values, template = finite_blocks(
        block_fields[variable].isel(simulation=run_index, drop=True)
    )
    reference_values, reference_template = finite_blocks(
        block_fields[variable].isel(simulation=reference_index, drop=True)
    )
    if template.dims != reference_template.dims:
        raise ValueError(f"{variable}: response block dimensions do not match")
    if run_values.shape[0] < MIN_BLOCKS or reference_values.shape[0] < MIN_BLOCKS:
        raise ValueError(
            f"{variable}: insufficient finite blocks "
            f"({run_values.shape[0]}, {reference_values.shape[0]})"
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Mean of empty slice",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        estimate, low, high = independent_bootstrap_field_difference(
            run_values,
            reference_values,
            rng=np.random.default_rng(BOOTSTRAP_SEED + int(seed_offset)),
            n_resamples=FIELD_BOOTSTRAP_REPLICATES,
            percentiles=CI_PERCENTILES,
        )
    result = xr.Dataset(
        {
            "block_mean_response": template.copy(data=estimate),
            "ci_low": template.copy(data=low),
            "ci_high": template.copy(data=high),
        }
    )
    result.attrs.update(
        method="independent complete-block bootstrap by simulation",
        confidence_percentiles=list(CI_PERCENTILES),
        bootstrap_replicates=FIELD_BOOTSTRAP_REPLICATES,
        n_blocks_run=int(run_values.shape[0]),
        n_blocks_reference=int(reference_values.shape[0]),
    )
    return result


def _plot_profile_bootstrap_interval(
    axis: Any,
    interval: xr.Dataset,
    colour: str,
) -> None:
    latitude = interval.ci_low.absolute_latitude
    axis.fill_between(
        latitude,
        interval.ci_low,
        interval.ci_high,
        color=colour,
        alpha=0.16,
        linewidth=0.0,
    )


def _sparse_bootstrap_stipple_points(
    significant: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    max_rows: int = 12,
    max_columns: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """Select one actual significant point from each coarse display cell."""

    significant = np.asarray(significant, dtype=bool)
    horizontal = np.asarray(horizontal, dtype=float)
    vertical = np.asarray(vertical, dtype=float)
    if significant.shape != horizontal.shape or significant.shape != vertical.shape:
        raise ValueError("Stippling arrays need matching two-dimensional shapes")
    valid = significant & np.isfinite(horizontal) & np.isfinite(vertical)
    selected_rows: list[int] = []
    selected_columns: list[int] = []
    row_groups = np.array_split(
        np.arange(valid.shape[0]),
        min(max_rows, valid.shape[0]),
    )
    column_groups = np.array_split(
        np.arange(valid.shape[1]),
        min(max_columns, valid.shape[1]),
    )
    for rows in row_groups:
        for columns in column_groups:
            candidates = np.argwhere(valid[np.ix_(rows, columns)])
            if candidates.size == 0:
                continue
            row_centre = 0.5 * (rows.size - 1)
            column_centre = 0.5 * (columns.size - 1)
            distance = (
                (candidates[:, 0] - row_centre) ** 2
                + (candidates[:, 1] - column_centre) ** 2
            )
            selected = candidates[int(np.argmin(distance))]
            selected_rows.append(int(rows[selected[0]]))
            selected_columns.append(int(columns[selected[1]]))
    return (
        horizontal[selected_rows, selected_columns],
        vertical[selected_rows, selected_columns],
    )


def _plot_bootstrap_exclusion_hatching(
    axis: Any,
    interval: xr.Dataset,
    vertical_coordinate: xr.DataArray | None = None,
) -> None:
    excludes_zero = (interval.ci_low > 0.0) | (interval.ci_high < 0.0)
    if vertical_coordinate is not None:
        excludes_zero, vertical_coordinate = xr.align(
            excludes_zero,
            vertical_coordinate,
            join="exact",
        )
        excludes_zero = excludes_zero.transpose("z", "absolute_latitude")
        vertical_coordinate = vertical_coordinate.transpose(
            "z", "absolute_latitude"
        )
        if vertical_coordinate.dims != ("z", "absolute_latitude"):
            raise ValueError(
                "TEM pressure coordinates need z and absolute_latitude dimensions"
            )
        horizontal = np.broadcast_to(
            np.asarray(excludes_zero.absolute_latitude.values)[None, :],
            vertical_coordinate.shape,
        )
        vertical = np.asarray(vertical_coordinate.values, dtype=float)
    elif "z" in excludes_zero.dims:
        excludes_zero = excludes_zero.transpose("z", "absolute_latitude")
        horizontal = np.broadcast_to(
            np.asarray(excludes_zero.absolute_latitude.values)[None, :],
            excludes_zero.shape,
        )
        vertical = np.broadcast_to(
            np.asarray(excludes_zero.z.values)[:, None] / 1_000.0,
            excludes_zero.shape,
        )
    elif "pres" in excludes_zero.dims:
        excludes_zero = excludes_zero.transpose("pres", "absolute_latitude")
        horizontal = np.broadcast_to(
            np.asarray(excludes_zero.absolute_latitude.values)[None, :],
            excludes_zero.shape,
        )
        vertical = np.broadcast_to(
            np.asarray(excludes_zero.pres.values)[:, None],
            excludes_zero.shape,
        )
    else:
        raise ValueError("Bootstrap map needs a z or pres coordinate")
    stipple_x, stipple_y = _sparse_bootstrap_stipple_points(
        np.asarray(excludes_zero.values, dtype=bool),
        horizontal,
        vertical,
    )
    axis.scatter(
        stipple_x,
        stipple_y,
        s=6.0,
        marker=".",
        color="0.12",
        alpha=0.65,
        linewidths=0.0,
        zorder=2.5,
    )


MOISTURE_FLUX_COMPONENTS = (
    "total_meridional_moisture_flux",
    "mean_meridional_moisture_flux",
    "eddy_meridional_moisture_flux",
)


def complete_moisture_pressure_column(fields: xr.Dataset) -> xr.Dataset:
    """Use one fixed lower boundary without selective 1000 hPa sampling."""

    pressure = np.asarray(fields.pres.values, dtype=float)
    for level in (MOISTURE_BOTTOM_SOURCE_HPA, MOISTURE_BOTTOM_PRESSURE_HPA):
        if not np.any(np.isclose(pressure, level)):
            raise ValueError(f"Moisture fields need the {level:g} hPa level")

    completed = xr.Dataset(coords={
        name: coordinate
        for name, coordinate in fields.coords.items()
        if name != "pres"
    })
    for name in (
        "total_meridional_moisture_flux",
        "mean_meridional_wind",
        "mean_specific_humidity",
    ):
        source = fields[name]
        above_bottom = source.where(
            source.pres < MOISTURE_BOTTOM_PRESSURE_HPA,
            drop=True,
        )
        bottom = source.sel(
            pres=MOISTURE_BOTTOM_SOURCE_HPA,
            drop=True,
        ).expand_dims(pres=[MOISTURE_BOTTOM_PRESSURE_HPA])
        completed[name] = xr.concat(
            [above_bottom, bottom],
            dim="pres",
        ).transpose(*source.dims)
        completed[name].attrs.update(source.attrs)
        completed[name].attrs["bottom_boundary_treatment"] = (
            f"{MOISTURE_BOTTOM_SOURCE_HPA:g} hPa value held to "
            f"{MOISTURE_BOTTOM_PRESSURE_HPA:g} hPa"
        )
    completed["mean_meridional_moisture_flux"] = (
        completed.mean_meridional_wind
        * completed.mean_specific_humidity
    )
    completed["eddy_meridional_moisture_flux"] = (
        completed.total_meridional_moisture_flux
        - completed.mean_meridional_moisture_flux
    )
    completed.attrs.update(fields.attrs)
    completed.attrs["bottom_boundary_treatment"] = (
        f"fixed {MOISTURE_BOTTOM_PRESSURE_HPA:g} hPa lower boundary; "
        f"{MOISTURE_BOTTOM_SOURCE_HPA:g} hPa statistics held constant below"
    )
    return completed


def fold_equatorward_transport(field: xr.DataArray) -> xr.DataArray:
    """Fold a northward flux so positive values point toward the equator."""

    folded = -fold_to_absolute(field, "odd")
    folded.name = field.name
    folded.attrs.update(field.attrs)
    folded.attrs["folded_sign_convention"] = "positive equatorward"
    return folded


def pressure_integrated_moisture_transport(
    flux: xr.DataArray,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> xr.DataArray:
    """Integrate specific-humidity transport through pressure coordinates."""

    if "pres" not in flux.dims:
        raise ValueError("Moisture flux needs a pressure dimension")
    pressure = np.asarray(flux.pres.values, dtype=float)
    if pressure.size < 2 or not np.all(np.isfinite(pressure)):
        raise ValueError("Moisture transport needs two finite pressure levels")
    ordered = flux.sortby("pres")

    def integrate_column(values: np.ndarray, pressure_hpa: np.ndarray) -> float:
        finite = np.isfinite(values) & np.isfinite(pressure_hpa)
        if finite.sum() < 2:
            return np.nan
        return float(np.trapz(values[finite], pressure_hpa[finite]))

    integrated = xr.apply_ufunc(
        integrate_column,
        ordered,
        ordered.pres,
        input_core_dims=[["pres"], ["pres"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    integrated = integrated * 100.0 / gravity_m_s2
    integrated.name = flux.name
    integrated.attrs.update(
        units="kg m-1 s-1",
        long_name="vertically integrated meridional moisture transport",
        integration="integral(flux dp) / g",
        pressure_conversion="100 Pa per hPa",
        missing_level_handling=(
            "trapezoidal integration across finite pressure levels in each column"
        ),
        sign_convention="positive northward",
    )
    return integrated


def zonal_mean_moisture_flux_convergence(
    northward_transport: xr.DataArray,
    radius_m: float = EARTH_RADIUS_M,
) -> xr.DataArray:
    """Return convergence of a zonal-mean meridional moisture transport."""

    if "latitude" not in northward_transport.dims:
        raise ValueError("Moisture transport needs a latitude dimension")
    transport = northward_transport.sortby("latitude")
    latitude = np.asarray(transport.latitude.values, dtype=float)
    if latitude.size < 3:
        raise ValueError("Moisture convergence needs three latitude points")
    phi = np.deg2rad(latitude)
    cos_phi = np.cos(phi)
    axis = transport.get_axis_num("latitude")
    shape = [
        latitude.size if index == axis else 1
        for index in range(transport.ndim)
    ]
    weighted = np.asarray(transport.values, dtype=float) * np.reshape(
        cos_phi,
        shape,
    )
    gradient = np.gradient(weighted, phi, axis=axis, edge_order=2)
    values = -gradient / (radius_m * np.reshape(cos_phi, shape))
    convergence = xr.DataArray(
        values,
        dims=transport.dims,
        coords=transport.coords,
        name=transport.name,
    )
    convergence.attrs.update(
        units="kg m-2 s-1",
        long_name="vertically integrated moisture-flux convergence",
        formula="-1/(a cos(phi)) d(cos(phi) F_phi)/dphi",
        sign_convention="positive for convergence",
    )
    return convergence


def folded_moisture_sections(fields: xr.Dataset) -> xr.Dataset:
    """Return equatorward-positive moisture-flux sections."""

    return xr.Dataset(
        {
            name: fold_equatorward_transport(fields[name])
            for name in MOISTURE_FLUX_COMPONENTS
        }
    )


def folded_moisture_profiles(
    fields: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Return folded transports and their scalar convergences."""

    northward = xr.Dataset(
        {
            name: pressure_integrated_moisture_transport(fields[name])
            for name in MOISTURE_FLUX_COMPONENTS
        }
    )
    transport = xr.Dataset(
        {
            name: fold_equatorward_transport(northward[name])
            for name in MOISTURE_FLUX_COMPONENTS
        }
    )
    convergence = xr.Dataset(
        {
            name: fold_to_absolute(
                zonal_mean_moisture_flux_convergence(northward[name]),
                "even",
            )
            for name in MOISTURE_FLUX_COMPONENTS
        }
    )
    for field in convergence.data_vars.values():
        field.attrs.update(
            units="kg m-2 s-1",
            sign_convention="positive for convergence",
        )
    return transport, convergence


def moisture_profile_block_fields(block_statistics: xr.Dataset) -> xr.Dataset:
    """Return compact moisture uncertainty fields for each time block."""

    completed = complete_moisture_pressure_column(block_statistics)
    transports, convergences = folded_moisture_profiles(completed)
    fields = {}
    for component in MOISTURE_FLUX_COMPONENTS:
        fields[f"transport_{component}"] = transports[component]
        fields[f"convergence_{component}"] = convergences[component]
    fields["section_total_meridional_moisture_flux"] = (
        fold_equatorward_transport(
            completed.total_meridional_moisture_flux
        )
    )
    fields["mean_meridional_wind"] = completed.mean_meridional_wind
    fields["mean_specific_humidity"] = completed.mean_specific_humidity
    result = xr.Dataset(fields)
    result.attrs.update(
        uncertainty_source="complete time blocks",
        bootstrap_method="independent resampling by simulation",
    )
    return result


def mean_circulation_response_components(
    fields: xr.Dataset,
    run_index: int,
    reference_index: int,
) -> xr.Dataset:
    """Split the mean-flux response around its comparison reference state."""

    run_wind = fields.mean_meridional_wind.isel(
        simulation=run_index,
        drop=True,
    )
    run_humidity = fields.mean_specific_humidity.isel(
        simulation=run_index,
        drop=True,
    )
    reference_wind = fields.mean_meridional_wind.isel(
        simulation=reference_index,
        drop=True,
    )
    reference_humidity = fields.mean_specific_humidity.isel(
        simulation=reference_index,
        drop=True,
    )
    wind_change = run_wind - reference_wind
    humidity_change = run_humidity - reference_humidity
    sections = xr.Dataset(
        {
            "wind_contribution": wind_change * reference_humidity,
            "humidity_contribution": reference_wind * humidity_change,
            "interaction": wind_change * humidity_change,
        }
    )
    sections["mean_circulation_change"] = (
        sections.wind_contribution
        + sections.humidity_contribution
        + sections.interaction
    )
    profiles = xr.Dataset(
        {
            name: fold_equatorward_transport(
                pressure_integrated_moisture_transport(sections[name])
            )
            for name in sections.data_vars
        }
    )
    profiles.attrs.update(
        reference_simulation=str(fields.simulation.values[reference_index]),
        decomposition=(
            "delta(v q) = delta(v) q_ref + v_ref delta(q) "
            "+ delta(v) delta(q)"
        ),
    )
    return profiles


def bootstrap_moisture_attribution_intervals(
    block_fields: xr.Dataset,
    run_index: int,
    reference_index: int,
    seed_offset: int,
) -> dict[str, xr.Dataset]:
    """Bootstrap the nonlinear mean-circulation moisture decomposition."""

    def climate_blocks(index: int) -> tuple[np.ndarray, np.ndarray]:
        selected = block_fields.isel(simulation=index, drop=True)
        wind = np.asarray(
            selected.mean_meridional_wind.transpose(
                "block", "pres", "latitude"
            ).values,
            dtype=float,
        )
        humidity = np.asarray(
            selected.mean_specific_humidity.transpose(
                "block", "pres", "latitude"
            ).values,
            dtype=float,
        )
        valid = np.isfinite(wind).any(axis=(1, 2)) & np.isfinite(
            humidity
        ).any(axis=(1, 2))
        return wind[valid], humidity[valid]

    def integrated_equatorward(section: np.ndarray) -> np.ndarray:
        order = np.argsort(pressure)
        ordered_pressure = pressure[order]
        ordered_section = section[order]
        northward = np.full(latitude.size, np.nan, dtype=float)
        for latitude_index in range(latitude.size):
            column = ordered_section[:, latitude_index]
            finite = np.isfinite(column) & np.isfinite(ordered_pressure)
            if finite.sum() >= 2:
                northward[latitude_index] = (
                    np.trapz(column[finite], ordered_pressure[finite])
                    * 100.0
                    / GRAVITY_M_S2
                )
        folded = -0.5 * (northward - northward[::-1])
        return folded[latitude >= 0.0]

    def sampled_mean(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        sample = values[indices]
        count = np.isfinite(sample).sum(axis=0)
        return np.divide(
            np.nansum(sample, axis=0),
            count,
            out=np.full(sample.shape[1:], np.nan, dtype=float),
            where=count > 0,
        )

    run_wind, run_humidity = climate_blocks(run_index)
    reference_wind, reference_humidity = climate_blocks(reference_index)
    if (
        run_wind.shape[0] < MIN_BLOCKS
        or reference_wind.shape[0] < MIN_BLOCKS
    ):
        raise ValueError(
            "Moisture attribution has insufficient finite blocks "
            f"({run_wind.shape[0]}, {reference_wind.shape[0]})"
        )
    pressure = np.asarray(block_fields.pres.values, dtype=float)
    latitude = np.asarray(block_fields.latitude.values, dtype=float)
    if not np.allclose(latitude, -latitude[::-1]):
        raise ValueError("Moisture attribution needs symmetric latitudes")
    absolute_latitude = latitude[latitude >= 0.0]
    names = tuple(_MEAN_RESPONSE_STYLES)
    replicates = {
        name: np.full(
            (FIELD_BOOTSTRAP_REPLICATES, absolute_latitude.size),
            np.nan,
            dtype=np.float32,
        )
        for name in names
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(seed_offset))
    for replicate in range(FIELD_BOOTSTRAP_REPLICATES):
        run_sample = rng.integers(0, run_wind.shape[0], run_wind.shape[0])
        reference_sample = rng.integers(
            0,
            reference_wind.shape[0],
            reference_wind.shape[0],
        )
        sampled_run_wind = sampled_mean(run_wind, run_sample)
        sampled_run_humidity = sampled_mean(run_humidity, run_sample)
        sampled_reference_wind = sampled_mean(
            reference_wind, reference_sample
        )
        sampled_reference_humidity = sampled_mean(
            reference_humidity, reference_sample
        )
        wind_change = sampled_run_wind - sampled_reference_wind
        humidity_change = (
            sampled_run_humidity - sampled_reference_humidity
        )
        sections = {
            "wind_contribution": (
                wind_change * sampled_reference_humidity
            ),
            "humidity_contribution": (
                sampled_reference_wind * humidity_change
            ),
            "interaction": wind_change * humidity_change,
        }
        sections["mean_circulation_change"] = sum(sections.values())
        for name, section in sections.items():
            replicates[name][replicate] = integrated_equatorward(section)

    intervals = {}
    for name, values in replicates.items():
        low, high = np.nanpercentile(values, CI_PERCENTILES, axis=0)
        intervals[name] = xr.Dataset(
            {
                "ci_low": ("absolute_latitude", low),
                "ci_high": ("absolute_latitude", high),
            },
            coords={"absolute_latitude": absolute_latitude},
            attrs={
                "method": (
                    "independent complete-block bootstrap by simulation"
                ),
                "bootstrap_replicates": FIELD_BOOTSTRAP_REPLICATES,
                "confidence_percentiles": list(CI_PERCENTILES),
            },
        )
    return intervals


def tropical_profile_magnitude(
    profile: xr.DataArray,
    latitude_limit: float = 30.0,
) -> float:
    """Return a cosine-weighted absolute magnitude within the tropics."""

    tropical = subset_between(
        profile.sortby("absolute_latitude"),
        "absolute_latitude",
        (0.0, latitude_limit),
    )
    latitude = np.asarray(tropical.absolute_latitude.values, dtype=float)
    values = np.asarray(tropical.values, dtype=float)
    finite = np.isfinite(latitude) & np.isfinite(values)
    if finite.sum() < 2:
        return np.nan
    latitude_radians = np.deg2rad(latitude[finite])
    weighted = np.abs(values[finite]) * np.cos(latitude_radians)
    return float(np.trapz(weighted, latitude_radians))


def mean_circulation_transport_share(
    transport_profiles: xr.Dataset,
    comparison_index: int = 1,
) -> float:
    """Measure the mean term's share of tropical U-response magnitudes."""

    mean_response = comparison_fields(
        transport_profiles,
        "mean_meridional_moisture_flux",
    )[comparison_index]
    eddy_response = comparison_fields(
        transport_profiles,
        "eddy_meridional_moisture_flux",
    )[comparison_index]
    mean_magnitude = tropical_profile_magnitude(mean_response)
    eddy_magnitude = tropical_profile_magnitude(eddy_response)
    denominator = mean_magnitude + eddy_magnitude
    if not np.isfinite(denominator) or denominator <= 0.0:
        return np.nan
    return mean_magnitude / denominator


def _metric_value(
    metrics: xr.Dataset,
    simulation_index: int,
    metric: str,
) -> float:
    return float(
        metrics.value.isel(simulation=simulation_index).sel(
            hemisphere="mean",
            metric=metric,
        ).values
    )


def _comparison_styles(
    run_index: int,
    reference_index: int | None,
) -> list[tuple[int, dict]]:
    run_style = {
        "color": simulation_colour(SIMULATIONS[run_index][0]),
        "linestyle": "-",
        "alpha": 1.0,
    }
    if reference_index is None:
        return [(run_index, run_style)]
    reference_style = {
        "color": simulation_colour(SIMULATIONS[reference_index][0]),
        "linestyle": "--",
        "alpha": 0.9,
    }
    return [(run_index, run_style), (reference_index, reference_style)]


def _set_latpres_axis(axis: Any, ylabel: bool) -> None:
    axis.set_xlim(0.0, 70.0)
    axis.set_yscale("log")
    axis.set_ylim(1000.0, 50.0)
    axis.set_xlabel("Absolute latitude (degrees)")
    if ylabel:
        axis.set_ylabel("Pressure (hPa)")


def plot_streamfunction(
    fields: xr.Dataset,
    metrics: xr.Dataset,
    field_blocks: xr.Dataset,
    show_bootstrap: bool,
) -> plt.Figure:
    comparison = [item / 1.0e9 for item in comparison_fields(fields, "streamfunction")]
    raw_levels = symmetric_percentile_levels_for_fields(
        [comparison[0]],
        percentile=99.0,
        ndiv=13,
        zero_field_scale=200.0,
    )
    difference_levels = symmetric_percentile_levels_for_fields(
        comparison[1:],
        percentile=99.0,
        ndiv=13,
        zero_field_scale=20.0,
    )
    fig, axes = plt.subplots(
        1,
        len(COMPARISONS),
        figsize=(3.9 * len(COMPARISONS), 3.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    images = []
    control = comparison[0]
    for column, ((_name, run_index, reference_index, label), field) in enumerate(
        zip(COMPARISONS, comparison)
    ):
        axis = axes[column]
        levels = raw_levels if column == 0 else difference_levels
        image = axis.contourf(
            field.absolute_latitude,
            field.pres,
            field,
            levels=levels,
            cmap=difference_cmap(levels),
            extend="both",
        )
        images.append(image)
        if column > 0:
            axis.contour(
                control.absolute_latitude,
                control.pres,
                control,
                levels=raw_levels[::2],
                colors="0.35",
                linewidths=0.45,
                alpha=0.55,
            )
        if show_bootstrap and reference_index is not None:
            interval = bootstrap_field_response(
                field_blocks,
                "streamfunction",
                run_index,
                reference_index,
                1_000 + column,
            )
            _plot_bootstrap_exclusion_hatching(axis, interval)
        for simulation_index, style in _comparison_styles(
            run_index,
            reference_index,
        ):
            tropopause = fields.tropopause_pressure.isel(
                simulation=simulation_index
            )
            axis.plot(
                tropopause.absolute_latitude,
                tropopause,
                linewidth=1.4,
                label="Tropopause" if column == 0 else None,
                **style,
            )
            edge = _metric_value(
                metrics,
                simulation_index,
                "hadley_edge_latitude",
            )
            axis.axvline(
                edge,
                linewidth=1.2,
                label="HC edge" if column == 0 else None,
                **style,
            )
        annotate_panel_label(axis, label)
        _set_latpres_axis(axis, ylabel=column == 0)
    axes[0].legend(loc="lower left", fontsize=7, frameon=False)
    fig.colorbar(images[0], ax=axes[0], label=r"Streamfunction ($10^9$ kg s$^{-1}$)")
    fig.colorbar(
        images[-1],
        ax=axes[1:],
        label=r"Streamfunction response ($10^9$ kg s$^{-1}$)",
    )
    uncertainty_note = (
        "; stippling marks pointwise 95% intervals excluding zero"
        if show_bootstrap
        else ""
    )
    fig.suptitle(
        "Hadley circulation and tropical-edge responses" + uncertainty_note
    )
    return fig


_LOCATION_MARKERS = {
    "hadley_edge_latitude": ("s", "HC edge"),
    "stj_latitude": ("o", "STJ core"),
    "edj_latitude": ("^", "EDJ"),
    "tropopause_break_latitude": ("D", "TPB"),
}


def _location_pressure(
    fields: xr.Dataset,
    metrics: xr.Dataset,
    simulation_index: int,
    metric: str,
) -> float:
    if metric == "hadley_edge_latitude":
        return 500.0
    if metric == "edj_latitude":
        return 850.0
    if metric == "stj_latitude":
        return _metric_value(metrics, simulation_index, "stj_core_pressure")
    latitude = _metric_value(metrics, simulation_index, metric)
    tropopause = fields.tropopause_pressure.isel(simulation=simulation_index)
    return float(tropopause.interp(absolute_latitude=latitude).values)


def _plot_location_markers(
    axis: Any,
    fields: xr.Dataset,
    metrics: xr.Dataset,
    run_index: int,
    reference_index: int | None,
    legend: bool,
) -> None:
    styles = _comparison_styles(run_index, reference_index)
    for simulation_position, (simulation_index, style) in enumerate(styles):
        for metric, (marker, label) in _LOCATION_MARKERS.items():
            latitude = _metric_value(metrics, simulation_index, metric)
            pressure = _location_pressure(
                fields,
                metrics,
                simulation_index,
                metric,
            )
            axis.scatter(
                latitude,
                pressure,
                s=34,
                marker=marker,
                facecolor=style["color"] if simulation_position == 0 else "none",
                edgecolor=style["color"],
                linewidth=1.0,
                zorder=5,
                label=label if legend and simulation_position == 0 else None,
            )


def plot_jet_and_width(
    fields: xr.Dataset,
    metrics: xr.Dataset,
    field_blocks: xr.Dataset,
    show_bootstrap: bool,
) -> plt.Figure:
    wind_fields = comparison_fields(fields, "zonal_wind")
    upper_profiles = comparison_fields(fields, "upper_zonal_wind")
    lower_profiles = comparison_fields(fields, "zonal_wind_850")
    raw_levels = symmetric_percentile_levels_for_fields(
        [wind_fields[0]],
        percentile=99.5,
        ndiv=15,
        zero_field_scale=40.0,
    )
    difference_levels = symmetric_percentile_levels_for_fields(
        wind_fields[1:],
        percentile=99.0,
        ndiv=13,
        zero_field_scale=5.0,
    )
    fig, axes = plt.subplots(
        2,
        len(COMPARISONS),
        figsize=(3.9 * len(COMPARISONS), 7.0),
        constrained_layout=True,
        sharex="col",
    )
    images = []
    control = wind_fields[0]
    for column, (
        (_name, run_index, reference_index, label),
        wind,
        upper,
        lower,
    ) in enumerate(zip(COMPARISONS, wind_fields, upper_profiles, lower_profiles)):
        levels = raw_levels if column == 0 else difference_levels
        image = axes[0, column].contourf(
            wind.absolute_latitude,
            wind.pres,
            wind,
            levels=levels,
            cmap=difference_cmap(levels),
            extend="both",
        )
        images.append(image)
        if column > 0:
            axes[0, column].contour(
                control.absolute_latitude,
                control.pres,
                control,
                levels=raw_levels[::2],
                colors="0.35",
                linewidths=0.45,
                alpha=0.55,
            )
        if show_bootstrap and reference_index is not None:
            map_interval = bootstrap_field_response(
                field_blocks,
                "zonal_wind",
                run_index,
                reference_index,
                1_500 + column,
            )
            _plot_bootstrap_exclusion_hatching(
                axes[0, column], map_interval
            )
        for simulation_index, style in _comparison_styles(
            run_index,
            reference_index,
        ):
            tropopause = fields.tropopause_pressure.isel(
                simulation=simulation_index
            )
            axes[0, column].plot(
                tropopause.absolute_latitude,
                tropopause,
                linewidth=1.2,
                **style,
            )
        _plot_location_markers(
            axes[0, column],
            fields,
            metrics,
            run_index,
            reference_index,
            legend=column == 0,
        )
        annotate_panel_label(axes[0, column], label)
        _set_latpres_axis(axes[0, column], ylabel=column == 0)

        axes[1, column].plot(
            upper.absolute_latitude,
            upper,
            color="tab:purple",
            linewidth=1.6,
            label="100-400 hPa",
        )
        axes[1, column].plot(
            lower.absolute_latitude,
            lower,
            color="tab:green",
            linewidth=1.5,
            linestyle="--",
            label="850 hPa",
        )
        if show_bootstrap and reference_index is not None:
            upper_interval = bootstrap_field_response(
                field_blocks,
                "upper_zonal_wind",
                run_index,
                reference_index,
                2_000 + column,
            )
            lower_interval = bootstrap_field_response(
                field_blocks,
                "zonal_wind_850",
                run_index,
                reference_index,
                2_100 + column,
            )
            _plot_profile_bootstrap_interval(
                axes[1, column], upper_interval, "tab:purple"
            )
            _plot_profile_bootstrap_interval(
                axes[1, column], lower_interval, "tab:green"
            )
        axes[1, column].axhline(0.0, color="0.3", linewidth=0.6)
        for simulation_index, style in _comparison_styles(
            run_index,
            reference_index,
        ):
            for metric in (
                "hadley_edge_latitude",
                "stj_latitude",
                "edj_latitude",
                "tropopause_break_latitude",
            ):
                axes[1, column].axvline(
                    _metric_value(metrics, simulation_index, metric),
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=0.75,
                    alpha=0.65,
                )
        axes[1, column].set_xlim(0.0, 70.0)
        axes[1, column].set_xlabel("Absolute latitude (degrees)")
        if column == 0:
            axes[1, column].set_ylabel(r"Zonal wind (m s$^{-1}$)")
            axes[0, column].legend(loc="lower left", fontsize=7, frameon=False)
            axes[1, column].legend(loc="best", fontsize=8, frameon=False)
    fig.colorbar(images[0], ax=axes[0, 0], label=r"Zonal wind (m s$^{-1}$)")
    fig.colorbar(
        images[-1],
        ax=axes[0, 1:],
        label=r"Zonal-wind response (m s$^{-1}$)",
    )
    uncertainty_note = (
        "; stippling and shading show pointwise 95% block-bootstrap results"
        if show_bootstrap
        else ""
    )
    fig.suptitle("Jet structure and tropical-width responses" + uncertainty_note)
    return fig


def _plot_transport_location_guides(
    axis: Any,
    profile: xr.DataArray,
    circulation_metrics: xr.Dataset,
    eddy_metrics: xr.Dataset,
    run_index: int,
    reference_index: int | None,
) -> None:
    for simulation_index, style in _comparison_styles(run_index, reference_index):
        for metric, line_style in (
            ("hadley_edge_latitude", ":"),
            ("stj_latitude", "-"),
            ("edj_latitude", "--"),
        ):
            axis.axvline(
                _metric_value(circulation_metrics, simulation_index, metric),
                color=style["color"],
                linestyle=line_style,
                linewidth=0.8,
                alpha=0.7,
            )
        position = _metric_value(
            eddy_metrics,
            simulation_index,
            "eddy_flux_max_latitude",
        )
        axis.axvline(
            position,
            color=style["color"],
            linestyle="-.",
            linewidth=1.5,
            alpha=0.95,
        )
        axis.scatter(
            position,
            float(profile.interp(absolute_latitude=position).values),
            marker="o",
            s=24,
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )


def plot_eddy_momentum_transport(
    eddy_fields: xr.Dataset,
    circulation_metrics: xr.Dataset,
    eddy_metrics: xr.Dataset,
    field_blocks: xr.Dataset,
    show_bootstrap: bool,
) -> plt.Figure:
    flux_fields = comparison_fields(eddy_fields, "eddy_momentum_flux")
    convergence_fields = comparison_fields(
        eddy_fields,
        "eddy_momentum_flux_convergence",
    )
    profiles = comparison_fields(eddy_fields, "upper_eddy_momentum_flux")
    flux_raw_levels = symmetric_percentile_levels_for_fields(
        [flux_fields[0]], percentile=99.0, ndiv=13, zero_field_scale=20.0
    )
    flux_diff_levels = symmetric_percentile_levels_for_fields(
        flux_fields[1:], percentile=99.0, ndiv=13, zero_field_scale=3.0
    )
    convergence_scaled = [item * 86400.0 for item in convergence_fields]
    convergence_raw_levels = symmetric_percentile_levels_for_fields(
        [convergence_scaled[0]], percentile=99.0, ndiv=13, zero_field_scale=2.0
    )
    convergence_diff_levels = symmetric_percentile_levels_for_fields(
        convergence_scaled[1:], percentile=99.0, ndiv=13, zero_field_scale=0.5
    )
    fig, axes = plt.subplots(
        3,
        len(COMPARISONS),
        figsize=(3.9 * len(COMPARISONS), 9.0),
        constrained_layout=True,
        sharex="col",
    )
    flux_images = []
    convergence_images = []
    for column, ((_name, run_index, reference_index, label), flux, convergence, profile) in enumerate(
        zip(COMPARISONS, flux_fields, convergence_scaled, profiles)
    ):
        flux_levels = flux_raw_levels if column == 0 else flux_diff_levels
        convergence_levels = (
            convergence_raw_levels if column == 0 else convergence_diff_levels
        )
        flux_image = axes[0, column].contourf(
            flux.absolute_latitude,
            flux.pres,
            flux,
            levels=flux_levels,
            cmap=difference_cmap(flux_levels),
            extend="both",
        )
        convergence_image = axes[1, column].contourf(
            convergence.absolute_latitude,
            convergence.pres,
            convergence,
            levels=convergence_levels,
            cmap=difference_cmap(convergence_levels),
            extend="both",
        )
        flux_images.append(flux_image)
        convergence_images.append(convergence_image)
        if column > 0:
            axes[0, column].contour(
                flux_fields[0].absolute_latitude,
                flux_fields[0].pres,
                flux_fields[0],
                levels=flux_raw_levels[::2],
                colors="0.35",
                linewidths=0.4,
                alpha=0.5,
            )
            axes[1, column].contour(
                convergence_scaled[0].absolute_latitude,
                convergence_scaled[0].pres,
                convergence_scaled[0],
                levels=convergence_raw_levels[::2],
                colors="0.35",
                linewidths=0.4,
                alpha=0.5,
            )
        if show_bootstrap and reference_index is not None:
            flux_interval = bootstrap_field_response(
                field_blocks,
                "eddy_momentum_flux",
                run_index,
                reference_index,
                3_100 + column,
            )
            convergence_interval = bootstrap_field_response(
                field_blocks,
                "eddy_momentum_flux_convergence",
                run_index,
                reference_index,
                3_200 + column,
            )
            _plot_bootstrap_exclusion_hatching(
                axes[0, column], flux_interval
            )
            _plot_bootstrap_exclusion_hatching(
                axes[1, column], convergence_interval
            )
        for row in (0, 1):
            annotate_panel_label(axes[row, column], label)
            _set_latpres_axis(axes[row, column], ylabel=column == 0)
        axes[2, column].plot(
            profile.absolute_latitude,
            profile,
            color="tab:purple",
            linewidth=1.6,
        )
        if show_bootstrap and reference_index is not None:
            interval = bootstrap_field_response(
                field_blocks,
                "upper_eddy_momentum_flux",
                run_index,
                reference_index,
                3_000 + column,
            )
            _plot_profile_bootstrap_interval(
                axes[2, column], interval, "tab:purple"
            )
        axes[2, column].axhline(0.0, color="0.3", linewidth=0.6)
        _plot_transport_location_guides(
            axes[2, column],
            profile,
            circulation_metrics,
            eddy_metrics,
            run_index,
            reference_index,
        )
        axes[2, column].set_xlim(0.0, 70.0)
        axes[2, column].set_xlabel("Absolute latitude (degrees)")
        if column == 0:
            axes[2, column].set_ylabel(r"200-250 hPa $\overline{u'v'}$ (m$^2$ s$^{-2}$)")
            axes[2, column].legend(
                handles=[
                    Line2D([0], [0], color="0.3", linestyle=":", label="HC edge"),
                    Line2D([0], [0], color="0.3", linestyle="-", label="STJ"),
                    Line2D([0], [0], color="0.3", linestyle="--", label="EDJ"),
                    Line2D(
                        [0],
                        [0],
                        color="0.3",
                        linestyle="-.",
                        marker="o",
                        label="Eddy-flux maximum",
                    ),
                ],
                loc="best",
                fontsize=7,
                frameon=False,
            )
    fig.colorbar(
        flux_images[0],
        ax=axes[0, 0],
        label=r"$\overline{u'v'}$ (m$^2$ s$^{-2}$)",
    )
    fig.colorbar(
        flux_images[-1],
        ax=axes[0, 1:],
        label=r"$\overline{u'v'}$ response (m$^2$ s$^{-2}$)",
    )
    fig.colorbar(
        convergence_images[0],
        ax=axes[1, 0],
        label=r"Convergence (m s$^{-1}$ day$^{-1}$)",
    )
    fig.colorbar(
        convergence_images[-1],
        ax=axes[1, 1:],
        label=r"Convergence response (m s$^{-1}$ day$^{-1}$)",
    )
    uncertainty_note = (
        "; stippling and shading show pointwise 95% block-bootstrap results"
        if show_bootstrap
        else ""
    )
    fig.suptitle(
        "Eddy momentum transport and circulation locations" + uncertainty_note
    )
    return fig


_MOISTURE_COMPONENT_STYLES = {
    "total_meridional_moisture_flux": {
        "color": "0.12",
        "linestyle": "-",
        "label": "Total",
    },
    "mean_meridional_moisture_flux": {
        "color": "#5e3c99",
        "linestyle": "-",
        "label": "Mean circulation",
    },
    "eddy_meridional_moisture_flux": {
        "color": "#1b9e77",
        "linestyle": "--",
        "label": "Eddy residual",
    },
}


_MEAN_RESPONSE_STYLES = {
    "mean_circulation_change": {
        "color": "0.12",
        "linestyle": "-",
        "label": "Mean-circulation change",
    },
    "wind_contribution": {
        "color": "#5e3c99",
        "linestyle": "-",
        "label": r"Wind: $\Delta\bar v\,\bar q_{ref}$",
    },
    "humidity_contribution": {
        "color": "#a6611a",
        "linestyle": "--",
        "label": r"Humidity: $\bar v_{ref}\,\Delta\bar q$",
    },
    "interaction": {
        "color": "#018571",
        "linestyle": ":",
        "label": "Interaction",
    },
}


_MOMENTUM_BALANCE_STYLES = {
    "eddy_momentum_flux_convergence": {
        "color": "#d95f02",
        "linestyle": "-",
        "label": "Eddy-flux convergence",
    },
    "mean_meridional_flow_torque": {
        "color": "#1b9e77",
        "linestyle": "--",
        "label": r"$-(f+\bar\zeta)\bar v$",
    },
}


def _plot_hadley_edge_guides(
    axis: Any,
    metrics: xr.Dataset,
    run_index: int,
    reference_index: int | None,
) -> None:
    for simulation_index, style in _comparison_styles(
        run_index,
        reference_index,
    ):
        axis.axvline(
            _metric_value(
                metrics,
                simulation_index,
                "hadley_edge_latitude",
            ),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=0.8,
            alpha=0.55,
        )


def _profile_response_limit(fields: Sequence[xr.DataArray]) -> float:
    values = np.concatenate(
        [np.asarray(field.values, dtype=float).ravel() for field in fields]
    )
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(finite, 99.0))
    return limit if limit > 0.0 else 1.0


def plot_moisture_transport(
    moisture_fields: xr.Dataset,
    circulation_fields: xr.Dataset,
    eddy_fields: xr.Dataset,
    circulation_metrics: xr.Dataset,
    moisture_blocks: xr.Dataset,
    circulation_blocks: xr.Dataset,
    eddy_blocks: xr.Dataset,
    show_bootstrap: bool,
) -> plt.Figure:
    """Plot Hadley, moisture, and upper-level momentum responses."""

    diagnostic_fields = complete_moisture_pressure_column(moisture_fields)
    sections = folded_moisture_sections(diagnostic_fields)
    transports, convergences = folded_moisture_profiles(diagnostic_fields)
    section_comparisons = comparison_fields(
        sections,
        "total_meridional_moisture_flux",
    )
    streamfunction_comparisons = [
        field / 1.0e9
        for field in comparison_fields(circulation_fields, "streamfunction")
    ]
    transport_comparisons = {
        name: comparison_fields(transports, name)
        for name in MOISTURE_FLUX_COMPONENTS
    }
    convergence_comparisons = {
        name: [
            field * SECONDS_PER_DAY
            for field in comparison_fields(convergences, name)
        ]
        for name in MOISTURE_FLUX_COMPONENTS
    }
    momentum_balance_comparisons = {
        "eddy_momentum_flux_convergence": [
            field * SECONDS_PER_DAY
            for field in comparison_fields(
                eddy_fields,
                "upper_eddy_momentum_flux_convergence",
            )
        ],
        "mean_meridional_flow_torque": [
            field * SECONDS_PER_DAY
            for field in comparison_fields(
                circulation_fields,
                "upper_mean_meridional_flow_torque",
            )
        ],
    }
    mean_share = mean_circulation_transport_share(transports)
    test_subset = bool(moisture_fields.attrs.get("test_subset", 0))
    show_decomposition = bool(mean_share > 0.5) or test_subset
    momentum_row = 4 if show_decomposition else 3
    row_count = momentum_row + 1

    scaled_sections = [field * 1000.0 for field in section_comparisons]
    raw_levels = symmetric_percentile_levels_for_fields(
        [scaled_sections[0]],
        percentile=99.0,
        ndiv=13,
        zero_field_scale=0.02,
    )
    difference_levels = symmetric_percentile_levels_for_fields(
        scaled_sections[1:],
        percentile=99.0,
        ndiv=13,
        zero_field_scale=0.005,
    )
    stream_raw_levels = symmetric_percentile_levels_for_fields(
        [streamfunction_comparisons[0]],
        percentile=99.0,
        ndiv=9,
        zero_field_scale=200.0,
    )
    stream_difference_levels = symmetric_percentile_levels_for_fields(
        streamfunction_comparisons[1:],
        percentile=99.0,
        ndiv=9,
        zero_field_scale=20.0,
    )

    fig, axes = plt.subplots(
        row_count,
        len(COMPARISONS),
        figsize=(3.9 * len(COMPARISONS), 2.85 * row_count + 0.8),
        constrained_layout=True,
        sharex="col",
        squeeze=False,
    )
    section_images = []
    for column, (
        (_name, run_index, reference_index, label),
        section,
        streamfunction,
    ) in enumerate(
        zip(COMPARISONS, scaled_sections, streamfunction_comparisons)
    ):
        levels = raw_levels if column == 0 else difference_levels
        stream_levels = (
            stream_raw_levels
            if column == 0
            else stream_difference_levels
        )
        image = axes[0, column].contourf(
            section.absolute_latitude,
            section.pres,
            section,
            levels=levels,
            cmap=difference_cmap(levels, cmap="BrBG"),
            extend="both",
        )
        section_images.append(image)
        streamfunction_for_contour = streamfunction
        if show_bootstrap and reference_index is not None:
            streamfunction_interval = bootstrap_field_response(
                circulation_blocks,
                "streamfunction",
                run_index,
                reference_index,
                3_700 + column,
            )
            streamfunction_for_contour = streamfunction.where(
                (streamfunction_interval.ci_low > 0.0)
                | (streamfunction_interval.ci_high < 0.0)
            )
        axes[0, column].contour(
            streamfunction_for_contour.absolute_latitude,
            streamfunction_for_contour.pres,
            streamfunction_for_contour,
            levels=stream_levels[1::2],
            colors="#5e3c99",
            linewidths=0.65,
        )
        if show_bootstrap and reference_index is not None:
            section_interval = bootstrap_field_response(
                moisture_blocks,
                "section_total_meridional_moisture_flux",
                run_index,
                reference_index,
                3_800 + column,
            )
            _plot_bootstrap_exclusion_hatching(
                axes[0, column], section_interval
            )
        annotate_panel_label(axes[0, column], label)
        _set_latpres_axis(axes[0, column], ylabel=column == 0)

        for variable, style in _MOISTURE_COMPONENT_STYLES.items():
            transport = transport_comparisons[variable][column]
            convergence = convergence_comparisons[variable][column]
            axes[1, column].plot(
                transport.absolute_latitude,
                transport,
                linewidth=1.6,
                **style,
            )
            axes[2, column].plot(
                convergence.absolute_latitude,
                convergence,
                linewidth=1.6,
                **style,
            )
            if show_bootstrap and reference_index is not None:
                transport_interval = bootstrap_field_response(
                    moisture_blocks,
                    f"transport_{variable}",
                    run_index,
                    reference_index,
                    4_000
                    + 100 * list(_MOISTURE_COMPONENT_STYLES).index(variable)
                    + column,
                )
                convergence_interval = bootstrap_field_response(
                    moisture_blocks,
                    f"convergence_{variable}",
                    run_index,
                    reference_index,
                    4_300
                    + 100 * list(_MOISTURE_COMPONENT_STYLES).index(variable)
                    + column,
                )
                _plot_profile_bootstrap_interval(
                    axes[1, column],
                    transport_interval,
                    style["color"],
                )
                _plot_profile_bootstrap_interval(
                    axes[2, column],
                    convergence_interval * SECONDS_PER_DAY,
                    style["color"],
                )
        for row in (1, 2):
            axes[row, column].axhline(0.0, color="0.45", linewidth=0.6)
            axes[row, column].set_xlim(0.0, 70.0)
            _plot_hadley_edge_guides(
                axes[row, column],
                circulation_metrics,
                run_index,
                reference_index,
            )
        axes[2, column].set_xlabel("Absolute latitude (degrees)")

        for variable, style in _MOMENTUM_BALANCE_STYLES.items():
            profile = momentum_balance_comparisons[variable][column]
            axes[momentum_row, column].plot(
                profile.absolute_latitude,
                profile,
                linewidth=1.6,
                **style,
            )
            if show_bootstrap and reference_index is not None:
                if variable == "eddy_momentum_flux_convergence":
                    blocks = eddy_blocks
                    block_variable = "upper_eddy_momentum_flux_convergence"
                else:
                    blocks = circulation_blocks
                    block_variable = "upper_mean_meridional_flow_torque"
                interval = bootstrap_field_response(
                    blocks,
                    block_variable,
                    run_index,
                    reference_index,
                    4_700
                    + 100 * list(_MOMENTUM_BALANCE_STYLES).index(variable)
                    + column,
                )
                _plot_profile_bootstrap_interval(
                    axes[momentum_row, column],
                    interval * SECONDS_PER_DAY,
                    style["color"],
                )
        axes[momentum_row, column].axhline(
            0.0,
            color="0.45",
            linewidth=0.6,
        )
        axes[momentum_row, column].set_xlim(0.0, 70.0)
        axes[momentum_row, column].set_xlabel("Absolute latitude (degrees)")
        _plot_hadley_edge_guides(
            axes[momentum_row, column],
            circulation_metrics,
            run_index,
            reference_index,
        )

    axes[1, 0].set_ylabel(
        "Equatorward moisture transport\n"
        r"(kg m$^{-1}$ s$^{-1}$)"
    )
    axes[2, 0].set_ylabel(
        "Moisture-flux convergence\n"
        r"(mm day$^{-1}$)"
    )
    axes[1, 0].legend(loc="best", fontsize=7, frameon=False)
    axes[momentum_row, 0].set_ylabel(
        "150-300 hPa acceleration\n"
        r"(m s$^{-1}$ day$^{-1}$; eastward +)"
    )
    axes[momentum_row, 0].legend(
        loc="best",
        fontsize=7,
        frameon=False,
    )
    transport_limit = _profile_response_limit(
        [
            transport_comparisons[name][column]
            for name in MOISTURE_FLUX_COMPONENTS
            for column in range(1, len(COMPARISONS))
        ]
    )
    convergence_limit = _profile_response_limit(
        [
            convergence_comparisons[name][column]
            for name in MOISTURE_FLUX_COMPONENTS
            for column in range(1, len(COMPARISONS))
        ]
    )
    momentum_limit = _profile_response_limit(
        [
            momentum_balance_comparisons[name][column]
            for name in _MOMENTUM_BALANCE_STYLES
            for column in range(1, len(COMPARISONS))
        ]
    )
    for column in range(1, len(COMPARISONS)):
        axes[1, column].set_ylim(-1.08 * transport_limit, 1.08 * transport_limit)
        axes[2, column].set_ylim(-1.08 * convergence_limit, 1.08 * convergence_limit)
        axes[momentum_row, column].set_ylim(
            -1.08 * momentum_limit,
            1.08 * momentum_limit,
        )

    if show_decomposition:
        control_mean = transports.mean_meridional_moisture_flux.isel(
            simulation=0,
            drop=True,
        )
        axes[3, 0].plot(
            control_mean.absolute_latitude,
            control_mean,
            color="#5e3c99",
            linewidth=1.7,
            label="Control mean circulation",
        )
        axes[3, 0].legend(loc="best", fontsize=7, frameon=False)
        attribution_fields = []
        for column, (
            _name,
            run_index,
            reference_index,
            _label,
        ) in enumerate(COMPARISONS[1:], start=1):
            if reference_index is None:
                continue
            attribution = mean_circulation_response_components(
                diagnostic_fields,
                run_index,
                reference_index,
            )
            attribution_intervals = (
                bootstrap_moisture_attribution_intervals(
                    moisture_blocks,
                    run_index,
                    reference_index,
                    4_900 + column,
                )
                if show_bootstrap
                else {}
            )
            for variable, style in _MEAN_RESPONSE_STYLES.items():
                profile = attribution[variable]
                attribution_fields.append(profile)
                axes[3, column].plot(
                    profile.absolute_latitude,
                    profile,
                    linewidth=1.6,
                    **style,
                )
                if show_bootstrap:
                    _plot_profile_bootstrap_interval(
                        axes[3, column],
                        attribution_intervals[variable],
                        style["color"],
                    )
            reference = str(attribution.attrs["reference_simulation"])
            axes[3, column].text(
                0.03,
                0.05,
                f"Fixed reference: {reference}",
                transform=axes[3, column].transAxes,
                fontsize=7,
                va="bottom",
            )
            axes[3, column].legend(loc="best", fontsize=6.5, frameon=False)
        attribution_limit = _profile_response_limit(attribution_fields)
        for column in range(1, len(COMPARISONS)):
            axes[3, column].set_ylim(
                -1.08 * attribution_limit,
                1.08 * attribution_limit,
            )
        for column, (
            _name,
            run_index,
            reference_index,
            _label,
        ) in enumerate(COMPARISONS):
            axes[3, column].axhline(0.0, color="0.45", linewidth=0.6)
            axes[3, column].set_xlim(0.0, 70.0)
            axes[3, column].set_xlabel("Absolute latitude (degrees)")
            _plot_hadley_edge_guides(
                axes[3, column],
                circulation_metrics,
                run_index,
                reference_index,
            )
        axes[3, 0].set_ylabel(
            "Mean-term attribution\n"
            r"(kg m$^{-1}$ s$^{-1}$)"
        )

    fig.colorbar(
        section_images[0],
        ax=axes[0, 0],
        label=(
            "Equatorward moisture flux\n"
            r"(g kg$^{-1}$ m s$^{-1}$)"
        ),
    )
    fig.colorbar(
        section_images[-1],
        ax=axes[0, 1:],
        label=(
            "Equatorward moisture-flux response\n"
            r"(g kg$^{-1}$ m s$^{-1}$)"
        ),
    )
    stream_handle = Line2D(
        [0],
        [0],
        color="#5e3c99",
        linewidth=0.8,
        label=r"Streamfunction contours ($10^9$ kg s$^{-1}$)",
    )
    axes[0, 0].legend(
        handles=[stream_handle],
        loc="lower left",
        fontsize=7,
        frameon=False,
    )
    share_text = (
        f"{mean_share:.0%}"
        if np.isfinite(mean_share)
        else "not available"
    )
    uncertainty_note = (
        "; stippling and shading show pointwise 95% block-bootstrap results"
        if show_bootstrap
        else ""
    )
    fig.suptitle(
        "Hadley circulation, moisture transport, and upper-level momentum balance; "
        f"mean-term share of tropical U - control response: {share_text}"
        + uncertainty_note
    )
    return fig


def _plot_tem_location_guides(
    axis: Any,
    circulation_metrics: xr.Dataset,
    run_index: int,
    reference_index: int | None,
    legend: bool = False,
) -> None:
    for position, (simulation_index, style) in enumerate(
        _comparison_styles(run_index, reference_index)
    ):
        for metric, line_style, label in (
            ("hadley_edge_latitude", ":", "HC edge"),
            ("stj_latitude", "-", "STJ"),
        ):
            axis.axvline(
                _metric_value(circulation_metrics, simulation_index, metric),
                color=style["color"],
                linestyle=line_style,
                linewidth=0.9,
                alpha=0.75,
                label=label if legend and position == 0 else None,
            )


def _tem_reference_pressure_grid(tem_fields: xr.Dataset) -> xr.DataArray:
    """Return one common pressure grid for every TEM comparison panel."""

    pressure = tem_fields.pressure_hpa.mean("simulation", skipna=True).transpose(
        "z",
        "absolute_latitude",
    )
    values = np.asarray(pressure.values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("TEM plot pressure must be finite and positive")
    if np.any(np.diff(values, axis=0) >= 0.0):
        raise ValueError("TEM plot pressure must decrease with height")
    pressure.attrs.update(
        units="hPa",
        long_name="common TEM plotting pressure",
        construction="mean across simulation climatologies at each height and latitude",
    )
    return pressure


def _tem_contour_coordinates(
    field: xr.DataArray,
    pressure: xr.DataArray,
) -> tuple[np.ndarray, xr.DataArray]:
    """Return two-dimensional latitude and pressure contour coordinates."""

    field, pressure = xr.align(field, pressure, join="exact")
    if field.dims != ("z", "absolute_latitude"):
        field = field.transpose("z", "absolute_latitude")
    latitude = np.broadcast_to(
        np.asarray(field.absolute_latitude.values)[None, :],
        field.shape,
    )
    return latitude, pressure


def _tem_colour_scale_fields(
    fields: Sequence[xr.DataArray],
) -> list[xr.DataArray]:
    """Select the derivative-safe interior used to set TEM colour limits."""

    lower, upper = TEM_COLOUR_SCALE_HEIGHT_BOUNDS_M
    selected = []
    for field in fields:
        if "z" not in field.dims:
            raise ValueError("TEM colour-scale fields must contain height")
        interior = field.where((field.z >= lower) & (field.z <= upper), drop=True)
        if interior.sizes.get("z", 0) == 0:
            raise ValueError(
                "TEM colour-scale height bounds do not overlap the archive"
            )
        selected.append(interior)
    return selected


def plot_tem_residual_circulation(
    tem_fields: xr.Dataset,
    circulation_metrics: xr.Dataset,
    field_blocks: xr.Dataset,
    show_bootstrap: bool,
) -> plt.Figure:
    """Plot residual streamfunction and residual velocities."""

    specifications = (
        (
            "residual_streamfunction",
            1.0e-9,
            r"Residual streamfunction ($10^9$ kg s$^{-1}$)",
            200.0,
            20.0,
        ),
        ("vres", 1.0, r"$v^*$ (m s$^{-1}$)", 1.0, 0.2),
        ("wres", 1.0e3, r"$w^*$ (mm s$^{-1}$)", 1.0, 0.2),
    )
    comparisons = {
        variable: [
            field * scale
            for field in comparison_fields(tem_fields, variable)
        ]
        for variable, scale, _label, _raw_zero, _difference_zero in specifications
    }
    levels = {}
    for variable, _scale, _label, raw_zero, difference_zero in specifications:
        fields = comparisons[variable]
        levels[variable] = (
            TEM_FIGURE_5_6_COLOUR_SCALE_FACTOR
            * symmetric_percentile_levels_for_fields(
                _tem_colour_scale_fields([fields[0]]),
                percentile=99.0,
                ndiv=13,
                zero_field_scale=raw_zero,
            ),
            TEM_FIGURE_5_6_COLOUR_SCALE_FACTOR
            * symmetric_percentile_levels_for_fields(
                _tem_colour_scale_fields(fields[1:]),
                percentile=99.0,
                ndiv=13,
                zero_field_scale=difference_zero,
            ),
        )

    pressure_grid = _tem_reference_pressure_grid(tem_fields)
    fig, axes = plt.subplots(
        len(specifications),
        len(COMPARISONS),
        figsize=(3.9 * len(COMPARISONS), 9.5),
        constrained_layout=True,
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    images: list[list[Any]] = [[] for _ in specifications]
    for row, (variable, _scale, label, _raw_zero, _difference_zero) in enumerate(
        specifications
    ):
        control = comparisons[variable][0]
        control_horizontal, control_vertical = _tem_contour_coordinates(
            control,
            pressure_grid,
        )
        for column, (
            (_name, run_index, reference_index, comparison_label),
            field,
        ) in enumerate(zip(COMPARISONS, comparisons[variable])):
            contour_levels = levels[variable][0 if column == 0 else 1]
            horizontal, vertical = _tem_contour_coordinates(
                field,
                pressure_grid,
            )
            image = axes[row, column].contourf(
                horizontal,
                vertical,
                field,
                levels=contour_levels,
                cmap=difference_cmap(contour_levels),
                extend="both",
            )
            images[row].append(image)
            if column > 0:
                axes[row, column].contour(
                    control_horizontal,
                    control_vertical,
                    control,
                    levels=levels[variable][0][::2],
                    colors="0.4",
                    linewidths=0.4,
                    alpha=0.45,
                )
            if show_bootstrap and reference_index is not None:
                interval = bootstrap_field_response(
                    field_blocks,
                    variable,
                    run_index,
                    reference_index,
                    5_000 + 100 * row + column,
                )
                _plot_bootstrap_exclusion_hatching(
                    axes[row, column],
                    interval,
                    pressure_grid,
                )
            _plot_tem_location_guides(
                axes[row, column],
                circulation_metrics,
                run_index,
                reference_index,
                legend=row == 0 and column == 0,
            )
            _set_latpres_axis(axes[row, column], ylabel=column == 0)
            if row == 0:
                annotate_panel_label(axes[row, column], comparison_label)
        axes[row, 0].text(
            -0.24,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            rotation=90.0,
            va="center",
            ha="center",
        )
        fig.colorbar(images[row][0], ax=axes[row, 0], label=label)
        fig.colorbar(
            images[row][-1],
            ax=axes[row, 1:],
            label=f"{label} response",
        )
    axes[0, 0].legend(loc="lower right", fontsize=7, frameon=False)
    uncertainty_note = (
        "; stippling marks pointwise 95% intervals excluding zero"
        if show_bootstrap
        else ""
    )
    fig.suptitle(
        "TEM residual circulation linking the UTLS and upper Hadley branch"
        + uncertainty_note
    )
    return fig


_TEM_BUDGET_LABELS = {
    "a_vres": r"$A_{v^*}$",
    "a_wres": r"$A_{w^*}$",
    "a_mmc": r"$A_{MMC}=A_{v^*}+A_{w^*}$",
    "a_ep": r"$A_{EP}$",
    "a_ep_meridional": r"$A_{EP,y}$",
    "a_ep_vertical": r"$A_{EP,z}$",
    "unresolved_direct_momentum_residual": (
        r"$R$: unresolved/direct budget residual"
    ),
    "a_tendency": r"$A_{tendency}$",
}

_TEM_PROFILE_STYLES = {
    "a_mmc": {"color": "#1b9e77", "linestyle": "-"},
    "a_ep": {"color": "#7570b3", "linestyle": "--"},
    "unresolved_direct_momentum_residual": {
        "color": "#d95f02",
        "linestyle": "-.",
    },
}


def _field_rms(field: xr.DataArray) -> float:
    values = np.asarray(field.values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(finite**2)))


def tendency_is_informative(tem_fields: xr.Dataset) -> bool:
    tendency = comparison_fields(tem_fields, "a_tendency")[1:]
    resolved = [
        field
        for variable in (
            "a_mmc",
            "a_ep",
            "unresolved_direct_momentum_residual",
        )
        for field in comparison_fields(tem_fields, variable)[1:]
    ]
    tendency_scale = max((_field_rms(field) for field in tendency), default=0.0)
    resolved_scale = max((_field_rms(field) for field in resolved), default=0.0)
    return resolved_scale > 0.0 and tendency_scale >= 0.1 * resolved_scale


def plot_tem_momentum_budget(
    tem_fields: xr.Dataset,
    circulation_metrics: xr.Dataset,
    field_blocks: xr.Dataset,
    show_bootstrap: bool,
) -> plt.Figure:
    """Plot the resolved TEM momentum budget and unresolved residual."""

    field_names = [
        "a_mmc",
        "a_ep",
        "unresolved_direct_momentum_residual",
    ]
    show_tendency = tendency_is_informative(tem_fields)
    if show_tendency:
        field_names.append("a_tendency")
    scaled = {
        name: [
            field * SECONDS_PER_DAY
            for field in comparison_fields(tem_fields, name)
        ]
        for name in field_names
    }
    raw_levels = (
        TEM_FIGURE_5_6_COLOUR_SCALE_FACTOR
        * symmetric_percentile_levels_for_fields(
            _tem_colour_scale_fields(
                [scaled[name][0] for name in field_names]
            ),
            percentile=99.0,
            ndiv=13,
            zero_field_scale=1.0,
        )
    )
    response_levels = (
        TEM_FIGURE_5_6_COLOUR_SCALE_FACTOR
        * symmetric_percentile_levels_for_fields(
            _tem_colour_scale_fields(
                [
                    field
                    for name in field_names
                    for field in scaled[name][1:]
                ]
            ),
            percentile=99.0,
            ndiv=13,
            zero_field_scale=0.2,
        )
    )
    pressure_grid = _tem_reference_pressure_grid(tem_fields)
    row_count = len(field_names) + 1
    fig, axes = plt.subplots(
        row_count,
        len(COMPARISONS),
        figsize=(3.9 * len(COMPARISONS), 2.65 * row_count + 0.8),
        constrained_layout=True,
        sharex="col",
        squeeze=False,
    )
    images: list[list[Any]] = [[] for _ in field_names]
    for row, name in enumerate(field_names):
        control = scaled[name][0]
        control_horizontal, control_vertical = _tem_contour_coordinates(
            control,
            pressure_grid,
        )
        control_contour_levels = symmetric_percentile_levels_for_fields(
            _tem_colour_scale_fields([control]),
            percentile=97.0,
            ndiv=9,
            zero_field_scale=0.2,
        )
        for column, (
            (_comparison_name, run_index, reference_index, comparison_label),
            field,
        ) in enumerate(zip(COMPARISONS, scaled[name])):
            contour_levels = raw_levels if column == 0 else response_levels
            horizontal, vertical = _tem_contour_coordinates(
                field,
                pressure_grid,
            )
            image = axes[row, column].contourf(
                horizontal,
                vertical,
                field,
                levels=contour_levels,
                cmap=difference_cmap(contour_levels),
                extend="both",
            )
            images[row].append(image)
            if column > 0:
                axes[row, column].contour(
                    control_horizontal,
                    control_vertical,
                    control,
                    levels=control_contour_levels[::2],
                    colors="0.4",
                    linewidths=0.4,
                    alpha=0.45,
                )
            if show_bootstrap and reference_index is not None:
                interval = bootstrap_field_response(
                    field_blocks,
                    name,
                    run_index,
                    reference_index,
                    6_000 + 100 * row + column,
                )
                _plot_bootstrap_exclusion_hatching(
                    axes[row, column],
                    interval,
                    pressure_grid,
                )
            _plot_tem_location_guides(
                axes[row, column],
                circulation_metrics,
                run_index,
                reference_index,
                legend=False,
            )
            _set_latpres_axis(axes[row, column], ylabel=column == 0)
            if row == 0:
                annotate_panel_label(axes[row, column], comparison_label)
        axes[row, 0].text(
            -0.24,
            0.5,
            _TEM_BUDGET_LABELS[name],
            transform=axes[row, 0].transAxes,
            rotation=90.0,
            va="center",
            ha="center",
        )
        fig.colorbar(
            images[row][0],
            ax=axes[row, 0],
            label=r"Acceleration (m s$^{-1}$ day$^{-1}$)",
        )
        fig.colorbar(
            images[row][-1],
            ax=axes[row, 1:],
            label=r"Response (m s$^{-1}$ day$^{-1}$)",
        )

    profile_row = row_count - 1
    profile_fields = {
        name: [
            field * SECONDS_PER_DAY
            for field in comparison_fields(tem_fields, f"stj_layer_{name}")
        ]
        for name in _TEM_PROFILE_STYLES
    }
    profile_values = []
    for column, (
        _comparison_name,
        run_index,
        reference_index,
        _comparison_label,
    ) in enumerate(COMPARISONS):
        axis = axes[profile_row, column]
        axis.axvspan(
            TEM_STJ_LATITUDE_BOUNDS[0],
            TEM_STJ_LATITUDE_BOUNDS[1],
            color="0.5",
            alpha=0.06,
        )
        for name, style in _TEM_PROFILE_STYLES.items():
            profile = profile_fields[name][column]
            profile_values.append(profile)
            axis.plot(
                profile.absolute_latitude,
                profile,
                linewidth=1.6,
                label=_TEM_BUDGET_LABELS[name],
                **style,
            )
            if show_bootstrap and reference_index is not None:
                interval = bootstrap_field_response(
                    field_blocks,
                    f"stj_layer_{name}",
                    run_index,
                    reference_index,
                    6_500
                    + 100 * list(_TEM_PROFILE_STYLES).index(name)
                    + column,
                )
                _plot_profile_bootstrap_interval(
                    axis,
                    interval * SECONDS_PER_DAY,
                    style["color"],
                )
        axis.axhline(0.0, color="0.35", linewidth=0.6)
        _plot_tem_location_guides(
            axis,
            circulation_metrics,
            run_index,
            reference_index,
            legend=column == 0,
        )
        axis.set_xlim(0.0, 70.0)
        axis.set_xlabel("Absolute latitude (degrees)")
    profile_limit = _profile_response_limit(profile_values)
    for axis in axes[profile_row, :]:
        axis.set_ylim(-1.08 * profile_limit, 1.08 * profile_limit)
    axes[profile_row, 0].set_ylabel(
        f"{TEM_STJ_LAYER_TOP_HPA:g}-{TEM_STJ_LAYER_BOTTOM_HPA:g} hPa\n"
        r"acceleration (m s$^{-1}$ day$^{-1}$)"
    )
    axes[profile_row, 0].legend(loc="best", fontsize=6.5, frameon=False)
    tendency_note = (
        "Archived mean tendency shown"
        if show_tendency
        else "Archived mean tendency is cached and small"
    )
    uncertainty_note = (
        "; stippling and shading show pointwise 95% block-bootstrap results"
        if show_bootstrap
        else ""
    )
    fig.suptitle(
        "Resolved TEM zonal-momentum budget and unresolved/direct residual; "
        + tendency_note
        + uncertainty_note
    )
    return fig


def plot_tem_mmc_decomposition(
    tem_fields: xr.Dataset,
    circulation_metrics: xr.Dataset,
    field_blocks: xr.Dataset,
    show_bootstrap: bool,
) -> plt.Figure:
    """Plot the meridional and vertical parts of the MMC acceleration."""

    field_names = ("a_vres", "a_wres")
    scaled = {
        name: [
            field * SECONDS_PER_DAY
            for field in comparison_fields(tem_fields, name)
        ]
        for name in field_names
    }
    raw_levels = symmetric_percentile_levels_for_fields(
        _tem_colour_scale_fields([scaled[name][0] for name in field_names]),
        percentile=TEM_MMC_DECOMPOSITION_COLOUR_PERCENTILE,
        ndiv=13,
        zero_field_scale=1.0,
    )
    response_levels = symmetric_percentile_levels_for_fields(
        _tem_colour_scale_fields(
            [field for name in field_names for field in scaled[name][1:]]
        ),
        percentile=TEM_MMC_DECOMPOSITION_COLOUR_PERCENTILE,
        ndiv=13,
        zero_field_scale=0.2,
    )
    pressure_grid = _tem_reference_pressure_grid(tem_fields)
    fig, axes = plt.subplots(
        len(field_names),
        len(COMPARISONS),
        figsize=(3.9 * len(COMPARISONS), 6.8),
        constrained_layout=True,
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    control_images = []
    response_images = []
    for row, name in enumerate(field_names):
        control = scaled[name][0]
        control_horizontal, control_vertical = _tem_contour_coordinates(
            control,
            pressure_grid,
        )
        control_contour_levels = symmetric_percentile_levels_for_fields(
            _tem_colour_scale_fields([control]),
            percentile=97.0,
            ndiv=9,
            zero_field_scale=0.2,
        )
        for column, (
            (_comparison_name, run_index, reference_index, comparison_label),
            field,
        ) in enumerate(zip(COMPARISONS, scaled[name])):
            levels = raw_levels if column == 0 else response_levels
            horizontal, vertical = _tem_contour_coordinates(
                field,
                pressure_grid,
            )
            image = axes[row, column].contourf(
                horizontal,
                vertical,
                field,
                levels=levels,
                cmap=difference_cmap(levels),
                extend="both",
            )
            if column == 0:
                control_images.append(image)
            else:
                response_images.append(image)
                axes[row, column].contour(
                    control_horizontal,
                    control_vertical,
                    control,
                    levels=control_contour_levels[::2],
                    colors="0.4",
                    linewidths=0.4,
                    alpha=0.45,
                )
            if show_bootstrap and reference_index is not None:
                interval = bootstrap_field_response(
                    field_blocks,
                    name,
                    run_index,
                    reference_index,
                    7_000 + 100 * row + column,
                )
                _plot_bootstrap_exclusion_hatching(
                    axes[row, column],
                    interval,
                    pressure_grid,
                )
            _plot_tem_location_guides(
                axes[row, column],
                circulation_metrics,
                run_index,
                reference_index,
                legend=row == 0 and column == 0,
            )
            _set_latpres_axis(axes[row, column], ylabel=column == 0)
            if row == 0:
                annotate_panel_label(axes[row, column], comparison_label)
        axes[row, 0].text(
            -0.24,
            0.5,
            _TEM_BUDGET_LABELS[name],
            transform=axes[row, 0].transAxes,
            rotation=90.0,
            va="center",
            ha="center",
        )
    axes[0, 0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.colorbar(
        control_images[0],
        ax=axes[:, 0],
        label=r"Acceleration (m s$^{-1}$ day$^{-1}$)",
    )
    fig.colorbar(
        response_images[0],
        ax=axes[:, 1:],
        label=r"Response (m s$^{-1}$ day$^{-1}$)",
    )
    uncertainty_note = (
        "; stippling marks pointwise 95% intervals excluding zero"
        if show_bootstrap
        else ""
    )
    fig.suptitle(
        "TEM residual-circulation acceleration decomposition"
        + uncertainty_note
    )
    return fig


def _ep_flux_reference(
    meridional_fields: Sequence[xr.DataArray],
    vertical_fields: Sequence[xr.DataArray],
) -> float:
    """Return one robust physical EP-flux scale for several panels."""

    magnitudes = []
    for meridional, vertical in zip(meridional_fields, vertical_fields):
        meridional, vertical = xr.align(meridional, vertical, join="exact")
        magnitude = np.hypot(meridional, vertical)
        selected = _tem_colour_scale_fields([magnitude])[0]
        values = np.asarray(selected.values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            magnitudes.append(finite)
    if not magnitudes:
        raise ValueError("EP-flux arrows need finite archived Fy and Fz values")
    reference = float(np.nanpercentile(np.concatenate(magnitudes), 95.0))
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("EP-flux arrow reference must be positive")
    return reference


def ep_flux_plot_components(
    meridional_flux: xr.DataArray,
    vertical_flux: xr.DataArray,
    pressure_hpa: xr.DataArray,
    reference_flux: float,
    target_arrow_length_m: float = EP_FLUX_ARROW_LENGTH_M,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Transform physical EP flux into latitude and pressure plot increments."""

    meridional_flux, vertical_flux, pressure_hpa = xr.align(
        meridional_flux,
        vertical_flux,
        pressure_hpa,
        join="exact",
    )
    if reference_flux <= 0.0 or not np.isfinite(reference_flux):
        raise ValueError("EP-flux arrow reference must be finite and positive")
    height = np.asarray(meridional_flux.z.values, dtype=float)
    pressure_values = np.asarray(pressure_hpa.values, dtype=float)
    pressure_gradient = xr.DataArray(
        np.gradient(pressure_values, height, axis=0, edge_order=2),
        dims=pressure_hpa.dims,
        coords=pressure_hpa.coords,
    )
    displacement_per_flux = target_arrow_length_m / reference_flux
    metres_per_degree = TEM_EARTH_RADIUS_M * np.deg2rad(1.0)
    latitude_increment = (
        meridional_flux * displacement_per_flux / metres_per_degree
    )
    pressure_increment = (
        vertical_flux * displacement_per_flux * pressure_gradient
    )
    latitude_increment.attrs.update(
        units="degrees latitude",
        plotting_transform="physical meridional displacement",
    )
    pressure_increment.attrs.update(
        units="hPa",
        plotting_transform="physical vertical displacement times dp/dz",
    )
    return latitude_increment, pressure_increment


def _plot_ep_flux_arrows(
    axis: Any,
    meridional_flux: xr.DataArray,
    vertical_flux: xr.DataArray,
    pressure_hpa: xr.DataArray,
    reference_flux: float,
    add_key: bool,
) -> Any:
    """Plot sparse EP-flux arrows on a logarithmic pressure axis."""

    latitude_increment, pressure_increment = ep_flux_plot_components(
        meridional_flux,
        vertical_flux,
        pressure_hpa,
        reference_flux,
    )
    latitude = np.asarray(meridional_flux.absolute_latitude.values, dtype=float)
    median_pressure = np.nanmedian(
        np.asarray(pressure_hpa.values, dtype=float),
        axis=1,
    )
    height_indices = np.flatnonzero(
        (median_pressure >= 70.0) & (median_pressure <= 900.0)
    )
    latitude_indices = np.flatnonzero((latitude >= 0.0) & (latitude <= 70.0))
    if height_indices.size == 0 or latitude_indices.size == 0:
        raise ValueError("EP-flux arrows do not overlap the plotted domain")
    height_stride = max(1, int(np.ceil(height_indices.size / 8.0)))
    latitude_stride = max(1, int(np.ceil(latitude_indices.size / 10.0)))
    height_indices = height_indices[::height_stride]
    latitude_indices = latitude_indices[::latitude_stride]
    indexer = np.ix_(height_indices, latitude_indices)
    horizontal = np.broadcast_to(latitude[None, :], pressure_hpa.shape)[indexer]
    vertical = np.asarray(pressure_hpa.values, dtype=float)[indexer]
    u_component = np.asarray(latitude_increment.values, dtype=float)[indexer]
    v_component = np.asarray(pressure_increment.values, dtype=float)[indexer]
    magnitude = np.hypot(
        np.asarray(meridional_flux.values, dtype=float)[indexer],
        np.asarray(vertical_flux.values, dtype=float)[indexer],
    )
    visible = magnitude >= 0.04 * reference_flux
    u_component = np.where(visible, u_component, np.nan)
    v_component = np.where(visible, v_component, np.nan)
    arrows = axis.quiver(
        horizontal,
        vertical,
        u_component,
        v_component,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="mid",
        color="0.12",
        alpha=0.78,
        width=0.0032,
        headwidth=3.6,
        headlength=4.4,
        headaxislength=4.0,
        zorder=4,
    )
    if add_key:
        key_length = EP_FLUX_ARROW_LENGTH_M / (
            TEM_EARTH_RADIUS_M * np.deg2rad(1.0)
        )
        axis.quiverkey(
            arrows,
            0.98,
            0.08,
            key_length,
            rf"{reference_flux:.2g} kg s$^{{-2}}$",
            coordinates="axes",
            labelpos="W",
            fontproperties={"size": 6.5},
        )
    return arrows


def _plot_hadley_streamfunction_contours(
    axis: Any,
    streamfunction: xr.DataArray,
    levels: np.ndarray,
) -> None:
    """Overlay signed Eulerian Hadley streamfunction contours."""

    axis.contour(
        streamfunction.absolute_latitude,
        streamfunction.pres,
        streamfunction,
        levels=levels[::2],
        colors="#5e3c99",
        linestyles=["--" if level < 0.0 else "-" for level in levels[::2]],
        linewidths=0.7,
        alpha=0.85,
        zorder=3,
    )


def plot_tem_ep_flux_propagation(
    tem_fields: xr.Dataset,
    circulation_fields: xr.Dataset,
    circulation_metrics: xr.Dataset,
    tem_blocks: xr.Dataset,
    circulation_blocks: xr.Dataset,
    show_bootstrap: bool,
) -> plt.Figure:
    """Plot EP-flux divergence, its components, and propagation vectors."""

    field_names = ("a_ep", "a_ep_meridional", "a_ep_vertical")
    scaled = {
        name: [
            field * SECONDS_PER_DAY
            for field in comparison_fields(tem_fields, name)
        ]
        for name in field_names
    }
    raw_levels = symmetric_percentile_levels_for_fields(
        _tem_colour_scale_fields([scaled[name][0] for name in field_names]),
        percentile=99.0,
        ndiv=13,
        zero_field_scale=1.0,
    )
    response_levels = symmetric_percentile_levels_for_fields(
        _tem_colour_scale_fields(
            [field for name in field_names for field in scaled[name][1:]]
        ),
        percentile=99.0,
        ndiv=13,
        zero_field_scale=0.2,
    )
    meridional_flux = comparison_fields(tem_fields, "ep_flux_meridional")
    vertical_flux = comparison_fields(tem_fields, "ep_flux_vertical")
    raw_flux_reference = _ep_flux_reference(
        [meridional_flux[0]],
        [vertical_flux[0]],
    )
    response_flux_reference = _ep_flux_reference(
        meridional_flux[1:],
        vertical_flux[1:],
    )
    streamfunction = [
        field / 1.0e9
        for field in comparison_fields(circulation_fields, "streamfunction")
    ]
    stream_raw_levels = symmetric_percentile_levels_for_fields(
        [streamfunction[0]],
        percentile=99.0,
        ndiv=9,
        zero_field_scale=200.0,
    )
    stream_response_levels = symmetric_percentile_levels_for_fields(
        streamfunction[1:],
        percentile=99.0,
        ndiv=9,
        zero_field_scale=20.0,
    )
    pressure_grid = _tem_reference_pressure_grid(tem_fields)
    fig, axes = plt.subplots(
        len(field_names),
        len(COMPARISONS),
        figsize=(3.9 * len(COMPARISONS), 9.5),
        constrained_layout=True,
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    control_images = []
    response_images = []
    for row, name in enumerate(field_names):
        control = scaled[name][0]
        control_horizontal, control_vertical = _tem_contour_coordinates(
            control,
            pressure_grid,
        )
        control_contour_levels = symmetric_percentile_levels_for_fields(
            _tem_colour_scale_fields([control]),
            percentile=97.0,
            ndiv=9,
            zero_field_scale=0.2,
        )
        for column, (
            (_comparison_name, run_index, reference_index, comparison_label),
            field,
        ) in enumerate(zip(COMPARISONS, scaled[name])):
            levels = raw_levels if column == 0 else response_levels
            horizontal, vertical = _tem_contour_coordinates(
                field,
                pressure_grid,
            )
            image = axes[row, column].contourf(
                horizontal,
                vertical,
                field,
                levels=levels,
                cmap=difference_cmap(levels),
                extend="both",
            )
            if column == 0:
                control_images.append(image)
            else:
                response_images.append(image)
                axes[row, column].contour(
                    control_horizontal,
                    control_vertical,
                    control,
                    levels=control_contour_levels[::2],
                    colors="0.45",
                    linewidths=0.35,
                    alpha=0.35,
                )
            if show_bootstrap and reference_index is not None:
                interval = bootstrap_field_response(
                    tem_blocks,
                    name,
                    run_index,
                    reference_index,
                    8_000 + 100 * row + column,
                )
                _plot_bootstrap_exclusion_hatching(
                    axes[row, column],
                    interval,
                    pressure_grid,
                )
            if row == 0:
                meridional_flux_for_arrows = meridional_flux[column]
                vertical_flux_for_arrows = vertical_flux[column]
                if show_bootstrap and reference_index is not None:
                    meridional_flux_interval = bootstrap_field_response(
                        tem_blocks,
                        "ep_flux_meridional",
                        run_index,
                        reference_index,
                        8_700 + column,
                    )
                    vertical_flux_interval = bootstrap_field_response(
                        tem_blocks,
                        "ep_flux_vertical",
                        run_index,
                        reference_index,
                        8_800 + column,
                    )
                    robust_vector = (
                        (meridional_flux_interval.ci_low > 0.0)
                        | (meridional_flux_interval.ci_high < 0.0)
                        | (vertical_flux_interval.ci_low > 0.0)
                        | (vertical_flux_interval.ci_high < 0.0)
                    )
                    meridional_flux_for_arrows = (
                        meridional_flux_for_arrows.where(robust_vector)
                    )
                    vertical_flux_for_arrows = (
                        vertical_flux_for_arrows.where(robust_vector)
                    )
                _plot_ep_flux_arrows(
                    axes[row, column],
                    meridional_flux_for_arrows,
                    vertical_flux_for_arrows,
                    pressure_grid,
                    (
                        raw_flux_reference
                        if column == 0
                        else response_flux_reference
                    ),
                    add_key=column in {0, 1},
                )
                streamfunction_for_contour = streamfunction[column]
                if show_bootstrap and reference_index is not None:
                    stream_interval = bootstrap_field_response(
                        circulation_blocks,
                        "streamfunction",
                        run_index,
                        reference_index,
                        8_500 + column,
                    )
                    streamfunction_for_contour = streamfunction_for_contour.where(
                        (stream_interval.ci_low > 0.0)
                        | (stream_interval.ci_high < 0.0)
                    )
                _plot_hadley_streamfunction_contours(
                    axes[row, column],
                    streamfunction_for_contour,
                    (
                        stream_raw_levels
                        if column == 0
                        else stream_response_levels
                    ),
                )
            _plot_tem_location_guides(
                axes[row, column],
                circulation_metrics,
                run_index,
                reference_index,
                legend=row == 0 and column == 0,
            )
            _set_latpres_axis(axes[row, column], ylabel=column == 0)
            if row == 0:
                annotate_panel_label(axes[row, column], comparison_label)
        axes[row, 0].text(
            -0.24,
            0.5,
            _TEM_BUDGET_LABELS[name],
            transform=axes[row, 0].transAxes,
            rotation=90.0,
            va="center",
            ha="center",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="#5e3c99",
            linewidth=0.8,
            label=r"HC streamfunction ($10^9$ kg s$^{-1}$)",
        )
    )
    labels.append(r"HC streamfunction ($10^9$ kg s$^{-1}$)")
    axes[0, 0].legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        fontsize=6.5,
        frameon=False,
    )
    fig.colorbar(
        control_images[0],
        ax=axes[:, 0],
        label=r"EP-flux acceleration (m s$^{-1}$ day$^{-1}$)",
    )
    fig.colorbar(
        response_images[0],
        ax=axes[:, 1:],
        label=r"EP-flux acceleration response (m s$^{-1}$ day$^{-1}$)",
    )
    uncertainty_note = (
        "; stippling and vector masking use pointwise 95% intervals"
        if show_bootstrap
        else ""
    )
    fig.suptitle(
        "Resolved EP-flux propagation and acceleration decomposition"
        + uncertainty_note
    )
    return fig


def tem_budget_metrics_table(metrics: xr.Dataset) -> pd.DataFrame:
    rows = []
    for simulation in metrics.simulation.values:
        selected = metrics.sel(simulation=simulation)
        for component in selected.component.values:
            rows.append(
                {
                    "simulation": str(simulation),
                    "component": str(component),
                    "region": selected.attrs.get("region", ""),
                    "signed_mean_m_s_day": float(
                        selected.signed_mean.sel(component=component).values
                        * SECONDS_PER_DAY
                    ),
                    "mean_absolute_m_s_day": float(
                        selected.mean_absolute.sel(component=component).values
                        * SECONDS_PER_DAY
                    ),
                    "rms_m_s_day": float(
                        selected.rms.sel(component=component).values
                        * SECONDS_PER_DAY
                    ),
                    "maximum_absolute_m_s_day": float(
                        selected.maximum_absolute.sel(component=component).values
                        * SECONDS_PER_DAY
                    ),
                    "unresolved_residual_is_nudging_tendency": False,
                }
            )
    return pd.DataFrame(rows)


def tem_budget_response_table(
    metrics: xr.Dataset,
    metric_blocks: xr.Dataset,
) -> pd.DataFrame:
    """Return block-bootstrap uncertainty for STJ-region budget responses."""

    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 6_000)
    statistics = ("signed_mean", "mean_absolute", "rms", "maximum_absolute")
    for comparison, run_index, reference_index, label in COMPARISONS[1:]:
        run = str(metrics.simulation.values[run_index])
        reference = str(metrics.simulation.values[reference_index])
        for component in metrics.component.values:
            for statistic in statistics:
                run_value = float(
                    metrics[statistic]
                    .sel(simulation=run, component=component)
                    .values
                    * SECONDS_PER_DAY
                )
                reference_value = float(
                    metrics[statistic]
                    .sel(simulation=reference, component=component)
                    .values
                    * SECONDS_PER_DAY
                )
                run_blocks = (
                    metric_blocks[statistic]
                    .sel(simulation=run, component=component)
                    .values
                    * SECONDS_PER_DAY
                )
                reference_blocks = (
                    metric_blocks[statistic]
                    .sel(simulation=reference, component=component)
                    .values
                    * SECONDS_PER_DAY
                )
                stats = bootstrap_metric_response(
                    run_blocks,
                    reference_blocks,
                    run_value - reference_value,
                    rng,
                )
                row = {
                    "comparison": comparison,
                    "comparison_label": label,
                    "run": run,
                    "reference": reference,
                    "component": str(component),
                    "statistic": statistic,
                    "region": metrics.attrs.get("region", ""),
                    "units": "m s-1 day-1",
                    "unresolved_residual_is_nudging_tendency": False,
                }
                row.update(stats)
                rows.append(row)
    return pd.DataFrame(rows)


def report_tem_budget_validation(metrics: xr.Dataset) -> None:
    """Report closure checks and typical STJ-region magnitudes."""

    for simulation in metrics.simulation.values:
        selected = metrics.sel(simulation=simulation)
        magnitudes = {
            str(component): float(
                selected.rms.sel(component=component).values
                * SECONDS_PER_DAY
            )
            for component in selected.component.values
        }
        progress(
            f"{simulation}: STJ-region RMS acceleration "
            + ", ".join(
                f"{name}={value:.3g} m s-1 day-1"
                for name, value in magnitudes.items()
            )
        )
        progress(
            f"{simulation}: EP component relative closure RMS="
            f"{float(selected.ep_component_closure_relative_rms.values):.3g}; "
            "unresolved/direct residual is not a nudging tendency"
        )


def _weighted_profile_cosine_similarity(
    first: xr.DataArray,
    second: xr.DataArray,
    latitude_bounds: tuple[float, float] = (5.0, 35.0),
) -> float:
    """Return a latitude-weighted signed similarity for two profiles."""

    first = first.sortby("absolute_latitude")
    second = second.sortby("absolute_latitude").interp(
        absolute_latitude=first.absolute_latitude
    )
    latitude = np.asarray(first.absolute_latitude.values, dtype=float)
    first_values = np.asarray(first.values, dtype=float)
    second_values = np.asarray(second.values, dtype=float)
    valid = (
        np.isfinite(first_values)
        & np.isfinite(second_values)
        & (latitude >= latitude_bounds[0])
        & (latitude <= latitude_bounds[1])
    )
    if not np.any(valid):
        return np.nan
    weights = np.cos(np.deg2rad(latitude[valid]))
    numerator = np.sum(weights * first_values[valid] * second_values[valid])
    denominator = np.sqrt(
        np.sum(weights * first_values[valid] ** 2)
        * np.sum(weights * second_values[valid] ** 2)
    )
    if denominator <= 0.0:
        return np.nan
    return float(numerator / denominator)


def validate_tem_streamfunction_sign(
    tem_fields: xr.Dataset,
    circulation_fields: xr.Dataset,
) -> dict[str, float]:
    """Compare control height streamfunctions with the existing pressure result."""

    control_label = next(
        label
        for label, _directory in SIMULATIONS
        if canonical_simulation_name(label) == "control"
    )
    tem_control = tem_fields.sel(simulation=control_label)
    circulation_control = circulation_fields.sel(simulation=control_label)
    pressure_streamfunction = pressure_mean(
        circulation_control.streamfunction,
        300.0,
        700.0,
    )
    scores = {}
    for name in ("eulerian_streamfunction_height", "residual_streamfunction"):
        height_streamfunction = height_layer_mean(
            tem_control[name],
            tem_control.pressure_hpa,
            300.0,
            700.0,
        )
        score = _weighted_profile_cosine_similarity(
            height_streamfunction,
            pressure_streamfunction,
        )
        if not np.isfinite(score) or score <= 0.0:
            raise ValueError(
                f"Control {name} has an inconsistent sign relative to the "
                "existing Eulerian pressure-coordinate streamfunction"
            )
        scores[name] = score
    return scores


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_command_line(argv)
    figures = tuple(dict.fromkeys(args.figures or FIGURES))
    families = requested_families(figures)
    cache_root = run_cache_root(
        args.test,
        args.test_times,
        args.test_start,
    )
    save_location = run_save_location(
        args.test,
        args.test_times,
        args.test_start,
    )
    if args.test:
        progress(f"circulation test cache root: {cache_root}")
        progress(f"circulation test figures: {save_location}")
    products = {
        family: prepare_family(
            family,
            figures_only=args.figures_only,
            rebuild_cache=args.rebuild_cache,
            test=args.test,
            test_times=args.test_times,
            test_start=args.test_start,
            cache_root=cache_root,
        )
        for family in ("circulation", "eddy", "moisture", "tem")
        if family in families
    }
    circulation_fields, circulation_metrics, circulation_blocks = products[
        "circulation"
    ]
    metric_products = [circulation_metrics]
    block_products = [circulation_blocks[["value"]]]
    if "eddy" in products:
        _eddy_fields, eddy_metrics, eddy_blocks = products["eddy"]
        metric_products.append(eddy_metrics)
        block_products.append(eddy_blocks[["value"]])
    combined_metrics = combine_metric_products(metric_products)
    combined_blocks = combine_metric_products(block_products)
    combined_metrics.attrs.update(
        test_subset=int(args.test),
        test_times=int(args.test_times) if args.test else 0,
        test_start=args.test_start or "",
    )
    combined_blocks.attrs.update(
        test_subset=int(args.test),
        test_times=int(args.test_times) if args.test else 0,
        test_start=args.test_start or "",
    )

    saver = FigureSaver(
        save_location,
        enabled=True,
        figures_to_run=figures,
    )
    save_publication_products(
        Path(saver.save_loc),
        combined_metrics,
        combined_blocks,
        include_responses=not args.test,
    )
    if "tem" in products:
        _tem_fields, tem_metrics, tem_blocks = products["tem"]
        report_tem_budget_validation(tem_metrics)
        sign_scores = validate_tem_streamfunction_sign(
            _tem_fields,
            circulation_fields,
        )
        progress(
            "control streamfunction sign agreement with the existing "
            "Eulerian pressure result: "
            + ", ".join(
                f"{name}={score:.3f}"
                for name, score in sign_scores.items()
            )
        )
        write_csv_atomic(
            tem_budget_metrics_table(tem_metrics),
            Path(saver.save_loc) / "tem_budget_magnitudes.csv",
            index=False,
        )
        if not args.test:
            write_csv_atomic(
                tem_budget_response_table(tem_metrics, tem_blocks),
                Path(saver.save_loc)
                / "tem_budget_response_uncertainty.csv",
                index=False,
            )
    if args.test:
        progress("test mode omits formal bootstrap response intervals")
    if 1 in figures:
        saver.save(
            plot_streamfunction(
                circulation_fields,
                circulation_metrics,
                circulation_blocks,
                show_bootstrap=not args.test,
            ),
            "streamfunction_tropical_edge_responses",
            number=1,
        )
    if 2 in figures:
        saver.save(
            plot_jet_and_width(
                circulation_fields,
                circulation_metrics,
                circulation_blocks,
                show_bootstrap=not args.test,
            ),
            "jet_structure_tropical_width_responses",
            number=2,
        )
    if 3 in figures:
        eddy_fields, eddy_metrics, eddy_blocks = products["eddy"]
        saver.save(
            plot_eddy_momentum_transport(
                eddy_fields,
                circulation_metrics,
                eddy_metrics,
                eddy_blocks,
                show_bootstrap=not args.test,
            ),
            "eddy_momentum_transport_responses",
            number=3,
        )
    if 4 in figures:
        moisture_fields, moisture_blocks = products["moisture"]
        eddy_fields, _eddy_metrics, eddy_blocks = products["eddy"]
        saver.save(
            plot_moisture_transport(
                moisture_fields,
                circulation_fields,
                eddy_fields,
                circulation_metrics,
                moisture_blocks,
                circulation_blocks,
                eddy_blocks,
                show_bootstrap=not args.test,
            ),
            "hadley_moisture_transport_responses",
            number=4,
        )
    if 5 in figures:
        tem_fields, _tem_metrics, tem_blocks = products["tem"]
        saver.save(
            plot_tem_residual_circulation(
                tem_fields,
                circulation_metrics,
                tem_blocks,
                show_bootstrap=not args.test,
            ),
            "tem_residual_circulation_responses",
            number=5,
        )
    if 6 in figures:
        tem_fields, _tem_metrics, tem_blocks = products["tem"]
        saver.save(
            plot_tem_momentum_budget(
                tem_fields,
                circulation_metrics,
                tem_blocks,
                show_bootstrap=not args.test,
            ),
            "tem_zonal_momentum_budget_responses",
            number=6,
        )
    if 7 in figures:
        tem_fields, _tem_metrics, tem_blocks = products["tem"]
        saver.save(
            plot_tem_mmc_decomposition(
                tem_fields,
                circulation_metrics,
                tem_blocks,
                show_bootstrap=not args.test,
            ),
            "tem_mmc_acceleration_decomposition_responses",
            number=7,
        )
    if 8 in figures:
        tem_fields, _tem_metrics, tem_blocks = products["tem"]
        saver.save(
            plot_tem_ep_flux_propagation(
                tem_fields,
                circulation_fields,
                circulation_metrics,
                tem_blocks,
                circulation_blocks,
                show_bootstrap=not args.test,
            ),
            "tem_ep_flux_propagation_responses",
            number=8,
        )
    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
