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
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
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
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib import pyplot as plt
from matplotlib import patheffects
from scipy.ndimage import gaussian_filter
from numba import njit

from diagnostics.diagnostic_helpers import (
    ANALYSIS_END,
    ANALYSIS_START,
    FigureSaver,
    LATLON_PATTERN,
    QUICK_TEST_LON_BOUNDS,
    ZONAL_MEAN_PATTERN,
    add_symmetric_colorbar,
    align_vertical_velocity_to_reference,
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
    nice_levels,
    progress,
    read_json_file,
    simulation_colour,
    standardize_latlon_dataset,
    subset_between,
    symmetric_percentile_levels_for_fields,
    time_block_slices,
    total_precip_accum,
    vertical_dim,
    write_csv_atomic,
    write_json_file,
    write_netcdf_atomic,
)
from diagnostics.analysis_config import (
    simulation_cache_label,
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
from diagnostics.cache_support import relabel_cached_simulations
from diagnostics.mean_state.circulation_bootstrap import (
    cached_interval,
    hemisphere_samples,
    response_distribution,
    response_interval,
)
from diagnostics.mean_state.vapour_transport_budget import (
    mixing_ratio_to_specific_humidity,
)
from diagnostics.mean_state.nudging_region import (
    NudgingRegion,
    load_nudging_regions,
    nudging_region_report,
)

from diagnostics.tem_core import TEM_FORMULATION, TEM_VERSION, log_pressure_height
from diagnostics.mean_state import circulation_mechanism as mechanism
from diagnostics.mean_state import circulation_lead_lag as lead_lag
from diagnostics.tem_archive import (
    height_coordinate_tem, load_stored_tem, tem_archive_root, tem_source_paths,
)

TEM_ARCHIVE_ROOT = None


FIGURES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15)
SIMULATIONS = tuple(
    (SIMULATION_LABELS[name], directory)
    for name, directory in SIMULATION_ARCHIVES.items()
)


COMPARISON_NAMES = {
    "control": ("control", ALL_SIMULATION_LABELS["control"]),
    "nudged-u": ("u", ALL_SIMULATION_LABELS["nudged-u"]),
    "nudged-T": ("t", ALL_SIMULATION_LABELS["nudged-T"]),
    "nudged-T-from-u": ("t_from_u", ALL_SIMULATION_LABELS["nudged-T-from-u"]),
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
        f"{ALL_SIMULATION_LABELS['nudged-u']} - {COMPARISON_NAMES[name][1]}",
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
    "circulation": "circulation-v8-native-inputs",
    "eddy": "circulation-eddy-v5-native-inputs",
    "moisture": "circulation-moisture-v8-native-inputs",
    "tem": "circulation-tem-v8-stored-conservative",
    "mechanism": "circulation-tem-mechanism-v4-mmc-decomposition",
    "lead_lag": "circulation-lead-lag-v1-fixed-area-daily",
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
    "mechanism": ("tem_mechanism_fields.nc", "tem_mechanism_blocks.nc"),
    "lead_lag": ("circulation_daily_timing.nc",),
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
F_EFF_MINIMUM_ABSOLUTE_VRES_M_S = 1.0e-2
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
    "a_density_storage",
    "a_mmc",
    "a_ep",
    "unresolved_direct_momentum_residual",
    "a_tendency",
)
TEM_BOOTSTRAP_FIELDS = (
    "a_density_storage", "stj_layer_a_density_storage",
    "residual_streamfunction",
    "residual_streamfunction_phi_gradient",
    "residual_streamfunction_z_gradient",
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
    "ep_flux_meridional_normalized",
    "ep_flux_vertical_normalized",
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
    parser.add_argument("--tem-archive-root", help="Root of stored TEM simulation archives")
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
    if selected & {4, 10, 12}:
        families.add("moisture")
    if selected & {5, 6, 7, 8, 9, 10, 13}:
        families.add("tem")
    if selected & {12, 14}:
        families.add("mechanism")
    if 15 in selected:
        families.add("lead_lag")
    return families


def family_source_paths(
    family: str,
    simulations: Sequence[tuple[str, str]] = SIMULATIONS,
) -> list[str]:
    if family in {"tem", "mechanism", "lead_lag"}:
        paths = [path for _label, directory in simulations
                 for path in tem_source_paths(directory, TEM_FORMULATION, TEM_ARCHIVE_ROOT)]
        if not paths:
            raise FileNotFoundError("Stored TEM fields are missing; run compute_tem_latlon_to_zm.py first")
        if family in {"mechanism", "lead_lag"}:
            paths.extend(path for _label, directory in simulations
                         for path in sorted(glob.glob(str(ROOT_DIR / directory / LATLON_PATTERN))))
        if family == "lead_lag":
            paths.extend(sorted(glob.glob(str(
                ROOT_DIR / ALL_SIMULATION_ARCHIVES["control"] / ZONAL_MEAN_PATTERN))))
        return paths
    pattern = (
        ZONAL_MEAN_PATTERN
        if family in {"circulation", "tem"}
        else LATLON_PATTERN
    )
    paths = [
        path
        for _label, directory in simulations
        for path in sorted(glob.glob(str(ROOT_DIR / directory / pattern)))
    ]
    # Pressure diagnostics share a reference grid from the control's native
    # pressure profile. Include that archive in every affected manifest.
    paths.extend(sorted(glob.glob(str(
        ROOT_DIR / ALL_SIMULATION_ARCHIVES["control"] / ZONAL_MEAN_PATTERN
    ))))
    return sorted(set(paths))


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
        "simulations": [[simulation_cache_label(label), directory] for label, directory in simulations],
        "analysis_start": ANALYSIS_START,
        "analysis_end": ANALYSIS_END,
        "block_days": None if test else BLOCK_DAYS,
        "minimum_blocks": 1 if test else MIN_BLOCKS,
        "hemisphere_handling": "independent SH/NH blocks and equal-weight hemisphere mean",
        "test_subset": bool(test),
    }
    configuration["bootstrap_uncertainty"] = {
        "method": "independent complete-block resampling by simulation and hemisphere",
        "equator_handling": "shared block draws across hemispheres",
        "minimum_finite_blocks_per_point": MIN_BLOCKS,
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
    if family in {"circulation", "eddy", "moisture"}:
        configuration.update(
            vertical_source="native model levels; instantaneous pressure remapping before temporal/zonal statistics",
            pressure_reference="control zonal-mean native pressure at analysis start, averaged across latitude; retain archived pressure anchors",
            display_vertical_axis="logarithmic pressure in hPa",
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
                "temperature": "even",
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
                "temperature",
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
        configuration.update(
            archived_vapour_variable="qv",
            archived_meridional_velocity_variable="uReconstructMeridional",
            humidity_conversion="q = qv / (1 + qv)",
            mean_circulation_definition="[v_bar] [q_bar]",
            upward_moisture_flux_definition="[-omega_bar] [q_bar]; omega in Pa s-1",
            archived_vertical_velocity_variable="omg_isobaric",
            pressure_velocity_sampling="archived isobaric omega interpolated to the common pressure grid; omega remains an isobaric exception",
            archived_height_velocity_variable="w",
            height_velocity_interpolation="native interface w averaged to midlevels, then interpolated using instantaneous native pressure",
            combined_moisture_flux_definition="(equatorward mean v + upward mean w) * mean q; jointly finite samples",
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
                "section_mean_meridional_moisture_flux",
                "mean_meridional_wind",
                "mean_specific_humidity",
                "section_mean_upward_moisture_flux",
                "mean_upward_pressure_velocity",
                "mean_upward_specific_humidity",
                "mean_upward_velocity",
                "mean_height_upward_specific_humidity",
                "mean_combined_meridional_wind",
                "mean_combined_upward_velocity",
                "mean_combined_specific_humidity",
                "section_mean_equatorward_upward_moisture_flux",
            ],
        )
        if test:
            configuration["test_longitude_bounds"] = list(
                QUICK_TEST_LON_BOUNDS
            )
    elif family == "tem":
        configuration.update(
            upstream_tem_version=TEM_VERSION,
            tem_formulation=TEM_FORMULATION,
            tem_archive_root=str(tem_archive_root(TEM_FORMULATION, TEM_ARCHIVE_ROOT).resolve()),
            density_storage="-d<rho'u'>/dt / <rho> for HPE(z); zero in log-pressure",
            flux_divergence="density inside latitude and vertical derivatives",
            hemisphere_order="calculate each hemisphere before presentation folding",
        )
        configuration.update(
            required_archived_fields=[
                "u_bar",
                "v_bar",
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
                "2*pi*a*cos(latitude)*integral_top^p(vres dp)/g in log-pressure; "
                "2*pi*a*cos(latitude)*integral_z^top(rho_zm*vres dz) in geometric-height"
            ),
            residual_streamfunction_top_boundary="zero at the archive top",
            tem_meridional_acceleration=(
                "vres * [f - d(u*cos(latitude))/d(latitude) "
                "/ (a*cos(latitude))]"
            ),
            tem_meridional_discretization=(
                "producer m_dy = f - (du/dlatitude - tan(latitude)*u)/a"
            ),
            tem_meridional_factor_components=(
                "f = 2*Omega*sin(latitude); relative vorticity = "
                "-(du/dlatitude - tan(latitude)*u)/a; "
                "both archived as raw folded climatologies"
            ),
            tem_vertical_acceleration="-wres * du/dz",
            resolved_ep_acceleration="EPFd, copied without sign reversal",
            ep_flux_vectors=(
                "Jucker (2021), section 3 and Table 2; "
                "instantaneous Fy/(rho*a*cos(phi)), Fz/(rho*a*cos(phi)); "
                "cos(phi)^2 weighted conservative flux transformed with the full "
                "latitude/log-pressure Jacobian and final panel dimensions"
            ),
            ep_flux_arrow_reference=(
                "linear common component multiplier; 95th-percentile display "
                "magnitude shared across response panels"
            ),
            residual_circulation_vectors=(
                "Hollowed et al. (2025), Appendix C, equation C12; "
                "tangents (-dPsi/dY, dPsi/dX) to residual mass streamfunction "
                "in final display inches; one linear length calibration across "
                "all circulation panels; no regional amplitude multipliers"
            ),
            unresolved_direct_residual=(
                "du/dt - (A_MMC + EPFd + density_storage); not a diagnosed nudging tendency"
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
                "pointwise 95% interval excludes zero; fixed spatial thinning"
            ),
            spatial_parity={
                "residual_streamfunction": "odd",
                "residual_streamfunction_phi_gradient": "even",
                "residual_streamfunction_z_gradient": "odd",
                "vres": "odd",
                "wres": "even",
                "ep_flux_meridional": "odd",
                "ep_flux_vertical": "even",
                "ep_flux_meridional_normalized": "odd",
                "ep_flux_vertical_normalized": "even",
                "momentum_accelerations": "even",
            },
            cached_uncertainty_fields=list(TEM_BOOTSTRAP_FIELDS),
        )
    elif family in {"mechanism", "lead_lag"}:
        configuration.update(
            upstream_tem_version=TEM_VERSION,
            tem_formulation=TEM_FORMULATION,
            tem_archive_root=str(tem_archive_root(TEM_FORMULATION, TEM_ARCHIVE_ROOT).resolve()),
            region_latitude_degrees=list(mechanism.REGION_LATITUDE),
            region_pressure_hpa=list(mechanism.REGION_PRESSURE),
            quadrature_latitudes=mechanism.REGION_LATITUDES.tolist(),
            quadrature_pressures=mechanism.REGION_PRESSURES.tolist(),
            hc_latitude_degrees=list(mechanism.HC_LATITUDE),
            hc_pressure_hpa=list(mechanism.HC_PRESSURE),
            hc_sampling="native archived height/latitude grid; mask each state's mean pressure to 200-700 hPa",
            precipitation_latitude_degrees=list(mechanism.PRECIP_LATITUDE),
            precipitation="MPAS bucket-corrected accumulation; exact endpoints; right-ended archived intervals",
            precipitation_spatial_sampling="interpolate 10S/equator/10N before hemisphere folding; retain native interior latitudes",
            regional_weighting="common control rho*(-dz/dp)*cos(phi), with exact box boundaries",
            mmc_decomposition="products of matched mean vres/eta and wres/u_z; retain temporal covariance and interpolation remainder",
            cached_regional_primitives=list(mechanism.PRIMITIVES),
            ep_boundary_budget="outward angular-momentum flux; integral rho*a*cos(phi)*EPFd dV",
            ep_boundary_components="meridional boundary flux includes Fy through sloping pressure surfaces; vertical boundary flux uses Fz only",
            bootstrap="paired control/experiment dates; independent hemisphere draws; joint primitive resampling",
        )
        if family == "lead_lag":
            configuration.update(
                daily_tem_primitives=list(lead_lag.TEM_PRIMITIVES),
                daily_sampling="four right-ended six-hour samples; exact precipitation accumulation endpoints",
                hc_latitude_degrees=list(lead_lag.HC_LATITUDE),
                hc_pressure_hpa=list(lead_lag.HC_PRESSURE_HPA),
                hc_sampling="fixed signed cos(latitude)*dp area mean; native pressure remapping before daily mean; no maxima",
                moisture_pressure_hpa=list(lead_lag.MOISTURE_PRESSURE_HPA),
                moisture_latitude_degrees=list(lead_lag.TROPICAL_LATITUDE),
                moisture_convergence="native qv/v on 850-950 hPa; daily-zonal mean v*q; exact boundary inflow, including equator",
                moisture_input_height_ceiling_m=6000.0,
                pw="archived precipw; tropical cosine-area mean",
                bootstrap="joint daily dates; established circular moving blocks of aligned lag pairs; no artificial block-join neighbours",
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
    root = CACHE_ROOT if cache_root is None else cache_root
    if filename in CACHE_FILES.get("tem", ()):
        root = root / "tem" / TEM_FORMULATION
    return root / filename


def manifest_path(family: str, cache_root: Path | None = None) -> Path:
    root = CACHE_ROOT if cache_root is None else cache_root
    if family == "tem":
        root = root / "tem" / TEM_FORMULATION
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


def fold_to_absolute(
    field: xr.DataArray, parity: str, *, keep_hemispheres: bool = False,
) -> xr.DataArray:
    """Reflect hemispheres, optionally retaining separate bootstrap samples."""

    if keep_hemispheres:
        return hemisphere_samples(field, parity)

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
        (len(simulations), nblocks, 2, len(metrics)),
        np.nan,
    )
    for simulation_index, simulation_records in enumerate(records):
        for block_index, record in enumerate(simulation_records[:nblocks]):
            for metric_index, metric in enumerate(metrics):
                for hemisphere_index, hemisphere in enumerate(HEMISPHERES[:2]):
                    values[
                        simulation_index,
                        block_index,
                        hemisphere_index,
                        metric_index,
                    ] = record[metric][hemisphere]
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
            "hemisphere": list(HEMISPHERES[:2]),
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
    temperature: xr.DataArray,
    psi: xr.DataArray,
    tropopause: xr.DataArray,
    *,
    keep_hemispheres: bool = False,
) -> xr.Dataset:
    folded_wind = fold_to_absolute(
        zonal_wind.transpose("pres", "latitude"),
        "even",
        keep_hemispheres=keep_hemispheres,
    )
    folded_psi = fold_to_absolute(
        psi.transpose("pres", "latitude"), "odd", keep_hemispheres=keep_hemispheres,
    )
    folded_temperature = fold_to_absolute(
        temperature.transpose("pres", "latitude"),
        "even",
        keep_hemispheres=keep_hemispheres,
    )
    folded_tropopause = fold_to_absolute(tropopause, "even", keep_hemispheres=keep_hemispheres)
    folded_torque = fold_to_absolute(
        mean_meridional_flow_torque(
            zonal_wind,
            meridional_wind,
        ).transpose("pres", "latitude"),
        "even",
        keep_hemispheres=keep_hemispheres,
    )
    upper_wind = pressure_mean(folded_wind, 100.0, 400.0)
    wind_850 = folded_wind.sortby("pres").interp(pres=850.0)
    core_layer = subset_between(folded_wind, "pres", (100.0, 400.0))
    return xr.Dataset(
        {
            "streamfunction": folded_psi,
            "zonal_wind": folded_wind,
            "temperature": folded_temperature,
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


@lru_cache(maxsize=1)
def native_reference_pressure_levels_hpa() -> np.ndarray:
    """Retain native-profile resolution on a common grid for pressure diagnostics."""

    files = sorted(glob.glob(str(
        ROOT_DIR / ALL_SIMULATION_ARCHIVES["control"] / ZONAL_MEAN_PATTERN
    )))
    if not files:
        raise FileNotFoundError("Native control zonal-mean files are required for the pressure reference")
    with xr.open_mfdataset(files, chunks={}, data_vars="minimal", coords="minimal", compat="override") as source:
        profile = source.pressure.sel(Time=ANALYSIS_START).mean("latitude", skipna=False).compute()
        native = np.asarray(profile.values, dtype=float) / 100.0
        anchors = np.asarray(source.pres.values, dtype=float)
    if not np.isfinite(native).all() or np.any(native <= 0.0):
        raise ValueError("Native control pressure reference requires positive finite levels")
    native = native[(native >= anchors.min()) & (native <= anchors.max())]
    targets = np.unique(np.r_[native, anchors])
    progress(f"pressure diagnostics: native control profile supplies {len(targets)} common pressure nodes, including diagnostic boundaries")
    return targets


def native_field_on_pressure_levels(
    field: xr.DataArray, pressure: xr.DataArray, targets_hpa: np.ndarray,
) -> xr.DataArray:
    """Remap native columns before averaging; never extrapolate beyond a column."""

    dimension = vertical_dim(pressure)
    field, pressure = xr.align(field, pressure, join="exact")
    if hasattr(field.data, "chunks"):
        field = field.chunk({dimension: -1})
        pressure = pressure.chunk({dimension: -1})
    field = mask_nonfinite_tem_values(field.astype(np.float64))
    pressure = mask_nonfinite_tem_values(pressure.astype(np.float64))
    result = xr.apply_ufunc(
        _interpolate_native_velocity_columns, pressure, field,
        input_core_dims=[[dimension], [dimension]], output_core_dims=[["pres"]],
        kwargs={"targets_hpa": np.asarray(targets_hpa, dtype=float)},
        dask="parallelized", output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"pres": len(targets_hpa)}, "allow_rechunk": True},
    ).assign_coords(pres=targets_hpa)
    result = result.transpose("Time", "pres", "latitude", *(
        ["longitude"] if "longitude" in result.dims else []))
    result.attrs.update(field.attrs, source_variable=field.name,
                        vertical_source="native model levels; instantaneous pressure interpolation; no extrapolation")
    result.pres.attrs.update(units="hPa")
    return result


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
    targets = native_reference_pressure_levels_hpa()
    dataset = xr.Dataset({
        output: native_field_on_pressure_levels(raw[source], raw.pressure, targets)
        for output, source in (("uzonal_isobaric", "uReconstructZonal"),
                               ("umeridional_isobaric", "uReconstructMeridional"),
                               ("t_isobaric", "t"))
    })
    dataset.set_close(raw.close)
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
    *,
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> xr.Dataset:
    """Load versioned upstream TEM and its matching mean-flow fields."""

    dataset = load_stored_tem(directory, formulation=TEM_FORMULATION, root=TEM_ARCHIVE_ROOT,
                              start=start, end=end)
    dataset = height_coordinate_tem(dataset)
    if test:
        dataset = test_time_window(
            dataset,
            simulation,
            test_times,
            test_start,
        )
    validate_tem_archive(dataset)
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


def mass_streamfunction_from_pressure_transport(wind, pressure):
    """Integrate pressure-coordinate mass transport from the archive top."""
    wind, pressure = xr.align(wind.astype(float), pressure.astype(float), join="exact")
    integral = xr.apply_ufunc(
        _reverse_cumulative_trapezoid, wind, -pressure,
        input_core_dims=[["z"], ["z"]], output_core_dims=[["z"]],
        vectorize=True, dask="parallelized", output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    ).assign_coords(z=wind.z).transpose(*wind.dims)
    result = 2*np.pi*TEM_EARTH_RADIUS_M*np.cos(np.deg2rad(wind.latitude.astype(float)))*integral/GRAVITY_M_S2
    result.attrs.update(units="kg s-1", integration="pressure integral from the archive top",
                        top_boundary="zero mass streamfunction")
    return result.transpose(*wind.dims)


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
    zonal_wind = mask_nonfinite_tem_values(dataset.get("u_bar", dataset.uReconstructZonal).astype(np.float64))
    meridional_wind = mask_nonfinite_tem_values(dataset.get("v_bar", dataset.uReconstructMeridional).astype(np.float64))
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
    absolute_vorticity = (dataset.absolute_vorticity if "absolute_vorticity" in dataset
                          else tem_absolute_vorticity_factor(zonal_wind))
    coriolis_parameter = (
        xr.zeros_like(absolute_vorticity)
        + 2.0
        * TEM_ANGULAR_VELOCITY_S1
        * np.sin(np.deg2rad(zonal_wind.latitude.astype(float)))
    )
    relative_vorticity = absolute_vorticity - coriolis_parameter
    vertical_shear = dataset.u_z if "u_z" in dataset else differentiate_physical_coordinate(zonal_wind, "z", height)

    a_vres = mask_nonfinite_tem_values(vres * absolute_vorticity)
    a_wres = mask_nonfinite_tem_values(-wres * vertical_shear)
    a_mmc = mask_nonfinite_tem_values(a_vres + a_wres)
    a_tendency = mask_nonfinite_tem_values(
        archived_zonal_wind_tendency(zonal_wind)
    )
    a_density_storage = xr.zeros_like(a_tendency)
    if "density_wind_covariance" in dataset:
        a_density_storage = -archived_zonal_wind_tendency(dataset.density_wind_covariance) / density
    unresolved = mask_nonfinite_tem_values(a_tendency - (a_mmc + a_ep + a_density_storage))
    ep_closure = mask_nonfinite_tem_values(
        a_ep - (a_ep_meridional + a_ep_vertical)
    )
    pressure_formulation = dataset.attrs.get("tem_formulation") == "log-pressure"
    streamfunction = mass_streamfunction_from_pressure_transport if pressure_formulation else mass_streamfunction_from_height_transport
    transport_coordinate = pressure if pressure_formulation else density
    residual_streamfunction = mask_nonfinite_tem_values(streamfunction(vres, transport_coordinate))
    # Reconstruct the conservative divergence of the actual stored fluxes.
    cos_phi = np.cos(np.deg2rad(dataset.latitude.astype(float)))
    flux_normalizer = density.where(density > 0.0) * TEM_EARTH_RADIUS_M * cos_phi
    normalized_y = mask_nonfinite_tem_values(ep_flux_meridional / flux_normalizer)
    normalized_z = mask_nonfinite_tem_values(ep_flux_vertical / flux_normalizer)
    normalized_ep_closure = a_ep - (
        differentiate_physical_coordinate(density * cos_phi**2 * normalized_y,
            "latitude", np.deg2rad(np.asarray(dataset.latitude.values, dtype=float)))
        / (TEM_EARTH_RADIUS_M * density * cos_phi**2)
        + differentiate_physical_coordinate(density * normalized_z, "z", height) / density
    )
    psi_phi = differentiate_physical_coordinate(
        residual_streamfunction, "latitude", np.deg2rad(np.asarray(dataset.latitude.values, dtype=float)),
    )
    psi_z = differentiate_physical_coordinate(residual_streamfunction, "z", height)
    eulerian_streamfunction = mask_nonfinite_tem_values(
        streamfunction(meridional_wind, transport_coordinate)
    )

    diagnostics = xr.Dataset(
        {
            "residual_streamfunction": residual_streamfunction,
            "residual_streamfunction_phi_gradient": psi_phi,
            "residual_streamfunction_z_gradient": psi_z,
            "eulerian_streamfunction_height": eulerian_streamfunction,
            "vres": vres,
            "wres": wres,
            "u_z": mask_nonfinite_tem_values(vertical_shear),
            "pressure_hpa": pressure / 100.0,
            "coriolis_parameter": coriolis_parameter,
            "relative_vorticity": relative_vorticity,
            "a_vres": a_vres,
            "a_wres": a_wres,
            "a_mmc": a_mmc,
            "a_ep": a_ep,
            "a_ep_meridional": a_ep_meridional,
            "a_ep_vertical": a_ep_vertical,
            "ep_flux_meridional": ep_flux_meridional,
            "ep_flux_vertical": ep_flux_vertical,
            "ep_flux_meridional_normalized": normalized_y,
            "ep_flux_vertical_normalized": normalized_z,
            "ep_normalized_operator_closure": normalized_ep_closure,
            "a_tendency": a_tendency,
            "a_density_storage": a_density_storage,
            "unresolved_direct_momentum_residual": unresolved,
            "ep_component_closure": ep_closure,
        }
    )
    diagnostics.residual_streamfunction.attrs["long_name"] = (
        "TEM residual mass streamfunction"
    )
    diagnostics.u_z.attrs.update(units="s-1", long_name="native vertical zonal-wind shear", formula="du/dz")
    diagnostics.ep_normalized_operator_closure.attrs.update(
        units="m s-2", long_name="archived EPFd minus normalized flux divergence",
        formula="EPFd - [d(rho*cos(phi)^2*M)/dphi/(a*rho*cos(phi)^2) + d(rho*N)/dz/rho]",
    )
    for name, units, formula in (
        ("residual_streamfunction_phi_gradient", "kg s-1 rad-1", "dPsi_res/dphi"),
        ("residual_streamfunction_z_gradient", "kg s-1 m-1", "dPsi_res/dz"),
    ):
        diagnostics[name].attrs.update(units=units, formula=formula)
    for name, component in (
        ("ep_flux_meridional_normalized", "Fy"),
        ("ep_flux_vertical_normalized", "Fz"),
    ):
        diagnostics[name].attrs.update(
            units="m2 s-2", formula=f"{component}/(rho_zm*a*cos(phi))",
            averaging="normalize instantaneous archived flux before time averaging",
            divergence_operator=(
                "d(rho*cos(phi)^2*M)/dphi/(a*rho*cos(phi)^2) + d(rho*N)/dz/rho"
            ),
        )
    diagnostics.eulerian_streamfunction_height.attrs["long_name"] = (
        "Eulerian mass streamfunction in the selected TEM coordinate"
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
    diagnostics.coriolis_parameter.attrs.update(
        units="s-1",
        long_name="Coriolis parameter",
        formula="f = 2*Omega*sin(latitude)",
    )
    diagnostics.relative_vorticity.attrs.update(
        units="s-1",
        long_name="zonal-mean relative vorticity from the meridional zonal-wind gradient",
        formula="zeta = -[du/dlatitude - u*tan(latitude)]/a",
        discretization="exact TEM producer m_dy wind-gradient term",
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
        formula="du/dt - (A_MMC + A_EP + A_density_storage)",
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
        tem_formulation=dataset.attrs.get("tem_formulation", "geometric-height"),
        archived_tem_reused=1,
        nonfinite_handling=(
            "non-finite archived and derived values masked before averaging"
        ),
    )
    diagnostics.attrs["tem_formulation"] = dataset.attrs.get("tem_formulation", "geometric-height")
    diagnostics.a_density_storage.attrs.update(units="m s-2", long_name="Density-wind covariance storage acceleration")
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
    total_weight = valid_weights.sum("z", skipna=True)
    profile = (field * valid_weights).sum("z", skipna=True) / total_weight.where(total_weight > 0)
    profile.attrs.update(field.attrs)
    profile.attrs.update(
        vertical_average="height weighted within archived mean pressure bounds",
        pressure_bounds_hpa=f"{top_hpa:g}-{bottom_hpa:g}",
    )
    return profile


def tem_climatology_fields(
    diagnostics: xr.Dataset, *, keep_hemispheres: bool = False,
) -> xr.Dataset:
    """Fold time-mean TEM fields and cache the STJ-layer profiles."""

    mean = diagnostics.mean("Time", skipna=diagnostics.attrs.get("tem_formulation") != "log-pressure", keep_attrs=True)
    parities = {
        "residual_streamfunction": "odd",
        "residual_streamfunction_phi_gradient": "even",
        "residual_streamfunction_z_gradient": "odd",
        "eulerian_streamfunction_height": "odd",
        "vres": "odd",
        "wres": "even",
        "pressure_hpa": "even",
        "coriolis_parameter": "odd",
        "relative_vorticity": "odd",
        "a_vres": "even",
        "a_wres": "even",
        "a_mmc": "even",
        "a_ep": "even",
        "a_ep_meridional": "even",
        "a_ep_vertical": "even",
        "ep_flux_meridional": "odd",
        "ep_flux_vertical": "even",
        "ep_flux_meridional_normalized": "odd",
        "ep_flux_vertical_normalized": "even",
        "a_tendency": "even",
        "a_density_storage": "even",
        "unresolved_direct_momentum_residual": "even",
        "ep_component_closure": "even",
        "ep_normalized_operator_closure": "even",
    }
    boundary_fields = set(parities) - {"pressure_hpa"}
    folded = {}
    for name, parity in parities.items():
        field = mean[name].transpose("z", "latitude")
        if name in boundary_fields:
            field = mask_spatial_derivative_boundaries(field)
        folded[name] = fold_to_absolute(field, parity, keep_hemispheres=keep_hemispheres)
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
    fields.z.attrs.update(units="m", long_name=("log-pressure height" if diagnostics.attrs.get("tem_formulation") == "log-pressure" else "geometric height"))
    fields.attrs["tem_formulation"] = diagnostics.attrs.get("tem_formulation", "geometric-height")
    fields.attrs.update(
        spatial_hemisphere_treatment=("separate hemispheres" if keep_hemispheres else "absolute-latitude fold"),
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
    operator_closure = _weighted_field_statistics(fields.ep_normalized_operator_closure, region)["rms"]
    metrics["ep_normalized_operator_closure_rms"] = operator_closure
    metrics["ep_normalized_operator_closure_relative_rms"] = operator_closure / ep_rms.where(ep_rms > 0.0)
    metrics.ep_normalized_operator_closure_rms.attrs["units"] = "m s-2"
    metrics.ep_normalized_operator_closure_relative_rms.attrs["units"] = "1"
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
        progress(f"{simulation}: loading stored {TEM_FORMULATION} TEM fields")
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
                tem_climatology_fields(
                    diagnostics.isel(Time=block_slice), keep_hemispheres=True,
                )
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
        "source": "versioned upstream TEM fields",
        "upstream_tem_version": TEM_VERSION,
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


def build_mechanism_products(
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
    simulations: Sequence[tuple[str, str]] = SIMULATIONS,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Cache small regional TEM states and exact precipitation block means."""

    if BLOCK_DAYS != 10.0:
        raise ValueError("The mechanism figure requires the configured 10-day circulation blocks")
    records, block_records = [], []
    for simulation, directory in simulations:
        progress(f"{simulation}: loading stored {TEM_FORMULATION} TEM for the fixed mechanism box")
        archive = load_tem_dataset(directory, simulation, test, test_times, test_start)
        surface = None
        try:
            times = np.asarray(archive.Time.values, dtype="datetime64[ns]")
            progress(f"{simulation}: mechanism source has {archive.sizes['z']} archived vertical levels; region {mechanism.REGION_DESCRIPTION}")
            dt = np.timedelta64(6, "h")
            if len(times) > 1 and not np.all(np.diff(times) == dt):
                raise ValueError("Mechanism blocks require the complete six-hourly archive")
            slices = diagnostic_block_slices(archive.vres, test=test)
            first = times[0] - dt
            surface = load_latlon_dataset(
                ROOT_DIR / directory, LATLON_PATTERN, chunks={"Time": 40},
                start=str(first), end=str(times[-1]),
            )
            endpoints = np.unique(np.r_[first, times[-1],
                [times[item.start] - dt for item in slices],
                [times[item.stop - 1] for item in slices]])
            progress(f"{simulation}: reducing tropical precipitation at {len(endpoints)} block endpoints")
            accumulation = mechanism.tropical_precipitation(
                total_precip_accum(surface.sel(Time=endpoints))
            ).compute()
            diagnostics = tem_instantaneous_diagnostics(archive)
            ends = sorted({0, len(times), *(item.stop for item in slices)})
            full_sum = None
            cached_blocks = []
            for left, right in zip(ends[:-1], ends[1:]):
                progress(f"{simulation}: mechanism reduction {left + 1}-{right}/{len(times)} timestamps")
                with dask.config.set(scheduler="threads", num_workers=2):
                    compact = mechanism.regional_tem_samples(
                        diagnostics.isel(Time=slice(left, right)),
                        archive.isel(Time=slice(left, right)),
                    ).mean("Time", skipna=False, keep_attrs=True).compute()
                weighted = compact * (right - left)
                full_sum = weighted if full_sum is None else full_sum + weighted
                if any(item.start == left and item.stop == right for item in slices):
                    compact["precipitation"] = mechanism.precipitation_window(
                        accumulation, times[left] - dt, times[right - 1],
                    )
                    compact["block_start"] = xr.DataArray(times[left] - dt)
                    compact["block_end"] = xr.DataArray(times[right - 1])
                    cached_blocks.append(compact)
            fields = full_sum / len(times)
            fields["precipitation"] = mechanism.precipitation_window(accumulation, first, times[-1])
            blocks = xr.concat(cached_blocks, dim=xr.IndexVariable("block", np.arange(len(slices))))
            for item in (fields, blocks):
                mechanism.annotate_products(item)
                item.attrs.update(
                    test_subset=int(test), block_days=10.0,
                    tem_formulation=TEM_FORMULATION,
                    region=mechanism.REGION_DESCRIPTION,
                    weighting="common control rho*(-dz/dp)*cos(phi); hemisphere means have equal weight",
                    block_pairing="match experiment/control block_start and block_end exactly",
                    precipitation="10S-10N cosine-area mean; bucket-corrected exact right-ended interval endpoints",
                    hc_definition="maximum magnitude of each simulation's mean Psi*, separately by hemisphere; 0-25 degrees, 200-700 hPa",
                    ep_boundary_budget="outward EP torque; integral rho*a*cos(phi)*EPFd dV",
                )
            records.append(fields)
            block_records.append(blocks)
            progress(f"{simulation}: compact mechanism products complete")
        finally:
            archive.close()
            if surface is not None:
                surface.close()
    coordinate = xr.IndexVariable("simulation", [item[0] for item in simulations])
    return (xr.concat(records, dim=coordinate, combine_attrs="drop_conflicts"),
            xr.concat(block_records, dim=coordinate, combine_attrs="drop_conflicts"))


def build_daily_timing_products(
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
    simulations: Sequence[tuple[str, str]] = SIMULATIONS,
) -> tuple[xr.Dataset]:
    """Stream native daily indices into the existing per-simulation cache architecture."""

    reference = native_reference_pressure_levels_hpa()
    lower, upper = lead_lag.MOISTURE_PRESSURE_HPA
    targets = np.unique(np.r_[lower, reference[(reference > lower) & (reference < upper)], upper])
    records = []
    for simulation, directory in simulations:
        progress(f"{simulation}: loading native TEM for daily production timing")
        archive = load_tem_dataset(directory, simulation, test, test_times, test_start)
        surface = None
        try:
            times = np.asarray(archive.Time.values, dtype="datetime64[ns]")
            days = lead_lag.daily_slices(times)
            first = times[0] - np.timedelta64(6, "h")
            surface = load_latlon_dataset(ROOT_DIR / directory, LATLON_PATTERN, chunks={"Time": 1},
                                          start=str(first), end=str(times[-1]))
            endpoints = np.r_[first, [times[day.stop - 1] for day in days]]
            progress(f"{simulation}: tropical precipitation reduction at {len(endpoints)} daily endpoints")
            with dask.config.set(scheduler="threads", num_workers=1):
                accumulation = mechanism.tropical_precipitation(total_precip_accum(
                    surface.sel(Time=endpoints))).compute()
            latitude = np.asarray(surface.latitude.values)
            boundary_rows = np.unique([index for boundary in (-10.0, 0.0, 10.0)
                                       for index in (np.searchsorted(latitude, boundary)-1,
                                                     np.searchsorted(latitude, boundary))])
            if boundary_rows.min() < 0 or boundary_rows.max() >= len(latitude):
                raise ValueError("Native moisture archive must bracket 10S, the equator, and 10N")
            # The low-level layer requires only lower-height columns. Retain
            # their native levels and reject unbracketed targets, with no extrapolation.
            levels = np.flatnonzero(np.asarray(archive.z) <= 6000.0)
            if surface.sizes["nVertLevels"] != archive.sizes["z"] or levels.size < 2:
                raise ValueError("Native TEM and moisture model levels must match")
            native = surface[["pressure", "qv", "uReconstructMeridional"]].isel(
                latitude=boundary_rows, nVertLevels=levels)
            diagnostics = tem_instantaneous_diagnostics(archive)
            daily = []
            for number, day in enumerate(days, start=1):
                started = perf_counter()
                progress(f"{simulation}: daily timing {number}/{len(days)}, {str(endpoints[number-1])[:10]}")
                sampled = mechanism.regional_tem_samples(diagnostics.isel(Time=day), archive.isel(Time=day))
                compact = lead_lag.daily_tem_state(sampled, diagnostics.isel(Time=day), archive.isel(Time=day))
                raw = native.sel(Time=times[day])
                wind = native_field_on_pressure_levels(raw.uReconstructMeridional, raw.pressure, targets)
                qv = native_field_on_pressure_levels(raw.qv, raw.pressure, targets)
                mean_v = wind.mean(("Time", "longitude"), skipna=False)
                mean_q = mixing_ratio_to_specific_humidity(qv).mean(("Time", "longitude"), skipna=False)
                compact["moisture_convergence"] = lead_lag.low_level_convergence(
                    mean_v, mean_q, radius_m=EARTH_RADIUS_M, gravity_m_s2=GRAVITY_M_S2)
                compact["pw"] = mechanism.tropical_precipitation(
                    surface.precipw.sel(Time=times[day])).mean("Time", skipna=False)
                compact["precipitation"] = mechanism.precipitation_window(
                    accumulation, endpoints[number-1], endpoints[number])
                with dask.config.set(scheduler="threads", num_workers=1):
                    compact = compact.compute()
                for name in (*lead_lag.TEM_PRIMITIVES, "hc_streamfunction", "moisture_convergence", "pw", "precipitation"):
                    if not np.isfinite(compact[name].values).all():
                        raise ValueError(f"{simulation}: nonfinite daily {name}; native coverage is required")
                compact.pw.attrs.update(units="kg m-2", source="archived precipw", definition="10S-10N cosine-area mean")
                compact = compact.expand_dims(Time=[endpoints[number-1] + np.timedelta64(12, "h")])
                compact["day_start"] = xr.DataArray([endpoints[number-1]], dims="Time")
                compact["day_end"] = xr.DataArray([endpoints[number]], dims="Time")
                daily.append(compact)
                progress(f"{simulation}: daily timing {number}/{len(days)} complete ({perf_counter()-started:.1f} s)")
            result = xr.concat(daily, dim="Time", join="exact", combine_attrs="drop_conflicts")
            result.attrs.update(test_subset=int(test), region=mechanism.REGION_DESCRIPTION,
                                hc_region="fixed signed 5-15 degrees, 150-850 hPa",
                                daily_definition="four right-ended six-hourly samples; coordinate at interval midpoint",
                                analysis_start=ANALYSIS_START, analysis_end=ANALYSIS_END)
            records.append(result.expand_dims(simulation=[simulation]))
        finally:
            archive.close()
            if surface is not None:
                surface.close()
    return (xr.concat(records, dim="simulation", join="exact", combine_attrs="drop_conflicts"),)


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
                    mean.t_isobaric,
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
                        block.t_isobaric,
                        block_psi,
                        block_tropopause,
                        keep_hemispheres=True,
                    )[
                        [
                            "streamfunction",
                            "zonal_wind",
                            "temperature",
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


def eddy_block_fields(
    blocks: xr.DataArray, *, keep_hemispheres: bool = False,
) -> xr.Dataset:
    """Return folded momentum-flux fields for each complete time block."""

    folded_flux = fold_to_absolute(
        blocks.transpose("block", "pres", "latitude"),
        "odd",
        keep_hemispheres=keep_hemispheres,
    )
    folded_convergence = fold_to_absolute(
        eddy_momentum_flux_convergence(blocks).transpose(
            "block",
            "pres",
            "latitude",
        ),
        "even",
        keep_hemispheres=keep_hemispheres,
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
        chunks={"Time": 1},
    )
    targets = native_reference_pressure_levels_hpa()
    dataset = xr.Dataset({
        output: native_field_on_pressure_levels(raw[source], raw.pressure, targets)
        for output, source in (("uzonal_isobaric", "uReconstructZonal"),
                               ("umeridional_isobaric", "uReconstructMeridional"))
    })
    dataset.set_close(raw.close)
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
        chunks={"Time": 1},
    )
    targets = native_reference_pressure_levels_hpa()
    # The native m/s velocity supplies the combined moisture diagnostic.
    # Archived omega remains necessary for the existing Pa/s diagnostics.
    omega = standardize_latlon_dataset(raw, ["omg_isobaric"]).omg_isobaric.interp(pres=targets)
    dataset = xr.Dataset({
        "umeridional_isobaric": native_field_on_pressure_levels(raw.uReconstructMeridional, raw.pressure, targets),
        "qv_isobaric": native_field_on_pressure_levels(raw.qv, raw.pressure, targets),
        "omg_isobaric": omega, "w": raw.w, "pressure": raw.pressure,
    })
    dataset.set_close(raw.close)
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


@njit(cache=True, nogil=True)
def _interpolate_native_velocity_columns(
    pressure_pa: np.ndarray, velocity: np.ndarray, targets_hpa: np.ndarray,
) -> np.ndarray:
    """Interpolate a whole chunk, with native levels on the final array axis."""

    nlevels = pressure_pa.shape[-1]
    pressure = np.ascontiguousarray(pressure_pa).reshape((-1, nlevels))
    wind = np.ascontiguousarray(velocity).reshape((-1, nlevels))
    result = np.full((pressure.shape[0], len(targets_hpa)), np.nan)
    p = np.empty(nlevels)
    w = np.empty(nlevels)
    sorted_p = np.empty(nlevels)
    sorted_w = np.empty(nlevels)
    for column in range(pressure.shape[0]):
        count = 0
        for level in range(nlevels):
            if np.isfinite(pressure[column, level]) and np.isfinite(wind[column, level]):
                p[count] = pressure[column, level] / 100.0
                w[count] = wind[column, level]
                count += 1
        if count < 2:
            continue
        ascending = True
        descending = True
        for level in range(1, count):
            ascending = ascending and p[level] > p[level - 1]
            descending = descending and p[level] < p[level - 1]
        if ascending:
            order = np.arange(count)
        elif descending:
            order = np.arange(count - 1, -1, -1)
        else:
            order = np.argsort(p[:count])
        unique_count = 0
        for level in order:
            if unique_count == 0 or p[level] != sorted_p[unique_count - 1]:
                sorted_p[unique_count] = p[level]
                sorted_w[unique_count] = w[level]
                unique_count += 1
        if unique_count < 2:
            continue
        result[column] = np.interp(targets_hpa, sorted_p[:unique_count], sorted_w[:unique_count])
        for target in range(len(targets_hpa)):
            if targets_hpa[target] < sorted_p[0] or targets_hpa[target] > sorted_p[unique_count - 1]:
                result[column, target] = np.nan
    return result.reshape(pressure_pa.shape[:-1] + (len(targets_hpa),))


def _native_upward_velocity_on_pressure_levels(dataset: xr.Dataset) -> xr.DataArray:
    """Interpolate native m/s velocity before zonal or temporal averaging."""

    pressure = dataset.pressure.astype(np.float64)
    velocity = align_vertical_velocity_to_reference(dataset.w.astype(np.float64), pressure)
    level_dimension = vertical_dim(pressure)
    targets = np.asarray(dataset.pres.values, dtype=float)
    # Storage chunks bound horizontal reads; each interpolation needs all levels.
    pressure = pressure.chunk({level_dimension: -1})
    velocity = velocity.chunk({level_dimension: -1})

    upward = xr.apply_ufunc(
        _interpolate_native_velocity_columns, pressure, velocity,
        input_core_dims=[[level_dimension], [level_dimension]],
        output_core_dims=[["pres"]], dask="parallelized",
        kwargs={"targets_hpa": targets},
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"pres": targets.size}, "allow_rechunk": True},
    ).assign_coords(pres=dataset.pres).transpose(*dataset.omg_isobaric.dims)
    upward.attrs.update(
        units="m s-1", source_variable="w", sign_convention="positive upward",
        vertical_interpolation="native interface average followed by instantaneous pressure interpolation; no extrapolation",
    )
    return upward


def _moisture_moment_sums(
    meridional_wind: xr.DataArray,
    vapour_mixing_ratio: xr.DataArray,
    pressure_velocity: xr.DataArray,
    upward_height_velocity: xr.DataArray,
) -> xr.Dataset:
    """Reduce paired finite samples to additive sums and counts."""

    wind = meridional_wind.astype(np.float64)
    humidity = mixing_ratio_to_specific_humidity(
        vapour_mixing_ratio.astype(np.float64)
    )
    upward_velocity = -pressure_velocity.astype(np.float64)
    height_velocity = upward_height_velocity.astype(np.float64)
    height_valid = np.isfinite(height_velocity) & np.isfinite(humidity)
    combined_valid = height_valid & np.isfinite(wind)
    combined_wind = wind.where(combined_valid)
    combined_upward = height_velocity.where(combined_valid)
    combined_humidity = humidity.where(combined_valid)
    height_humidity = humidity.where(height_valid)
    height_velocity = height_velocity.where(height_valid)
    vertical_valid = np.isfinite(upward_velocity) & np.isfinite(humidity)
    vertical_humidity = humidity.where(vertical_valid)
    upward_velocity = upward_velocity.where(vertical_valid)
    valid = np.isfinite(wind) & np.isfinite(humidity)
    wind = wind.where(valid)
    humidity = humidity.where(valid)
    mean_dimensions = tuple(
        dim for dim in ("Time", "longitude") if dim in wind.dims
    )
    if not mean_dimensions:
        raise ValueError("Moisture transport needs Time or longitude")

    return xr.Dataset({
        "horizontal_count": valid.sum(mean_dimensions),
        "vertical_count": vertical_valid.sum(mean_dimensions),
        "height_vertical_count": height_valid.sum(mean_dimensions),
        "combined_count": combined_valid.sum(mean_dimensions),
        "vq_sum": (wind * humidity).sum(mean_dimensions, skipna=True),
        "v_sum": wind.sum(mean_dimensions, skipna=True),
        "q_sum": humidity.sum(mean_dimensions, skipna=True),
        "upward_sum": upward_velocity.sum(mean_dimensions, skipna=True),
        "vertical_q_sum": vertical_humidity.sum(mean_dimensions, skipna=True),
        "height_upward_sum": height_velocity.sum(mean_dimensions, skipna=True),
        "height_vertical_q_sum": height_humidity.sum(mean_dimensions, skipna=True),
        "combined_v_sum": combined_wind.sum(mean_dimensions, skipna=True),
        "combined_w_sum": combined_upward.sum(mean_dimensions, skipna=True),
        "combined_q_sum": combined_humidity.sum(mean_dimensions, skipna=True),
    })


def moisture_flux_statistics(
    meridional_wind: xr.DataArray,
    vapour_mixing_ratio: xr.DataArray,
    pressure_velocity: xr.DataArray,
    upward_height_velocity: xr.DataArray,
) -> xr.Dataset:
    """Return compact statistics for the zonal-mean moisture-flux split."""

    return _moisture_statistics_from_moments(_moisture_moment_sums(
        meridional_wind, vapour_mixing_ratio, pressure_velocity, upward_height_velocity,
    ))


def _moisture_statistics_from_moments(moments: xr.Dataset) -> xr.Dataset:
    """Recover the original means with sample-count weighting."""

    horizontal_count = moments.horizontal_count.where(moments.horizontal_count > 0)
    vertical_count = moments.vertical_count.where(moments.vertical_count > 0)
    height_count = moments.height_vertical_count.where(moments.height_vertical_count > 0)
    combined_count = moments.combined_count.where(moments.combined_count > 0)
    statistics = xr.Dataset({
        "total_meridional_moisture_flux": moments.vq_sum / horizontal_count,
        "mean_meridional_wind": moments.v_sum / horizontal_count,
        "mean_specific_humidity": moments.q_sum / horizontal_count,
        "mean_upward_pressure_velocity": moments.upward_sum / vertical_count,
        "mean_upward_specific_humidity": moments.vertical_q_sum / vertical_count,
        "mean_upward_velocity": moments.height_upward_sum / height_count,
        "mean_height_upward_specific_humidity": moments.height_vertical_q_sum / height_count,
        "mean_combined_meridional_wind": moments.combined_v_sum / combined_count,
        "mean_combined_upward_velocity": moments.combined_w_sum / combined_count,
        "mean_combined_specific_humidity": moments.combined_q_sum / combined_count,
    })
    statistics["mean_meridional_moisture_flux"] = (
        statistics.mean_meridional_wind
        * statistics.mean_specific_humidity
    )
    statistics["eddy_meridional_moisture_flux"] = (
        statistics.total_meridional_moisture_flux
        - statistics.mean_meridional_moisture_flux
    )
    statistics["mean_upward_moisture_flux"] = (
        statistics.mean_upward_pressure_velocity
        * statistics.mean_upward_specific_humidity
    )
    statistics["mean_height_upward_moisture_flux"] = (
        statistics.mean_upward_velocity * statistics.mean_height_upward_specific_humidity
    )
    for name in ("mean_upward_velocity", "mean_combined_upward_velocity"):
        statistics[name].attrs.update(
            units="m s-1", sign_convention="positive upward",
            source_variable="w", vertical_interpolation="instantaneous native pressure before averaging",
        )
    statistics.mean_combined_meridional_wind.attrs.update(units="m s-1", sign_convention="positive northward")
    for name in ("mean_height_upward_specific_humidity", "mean_combined_specific_humidity"):
        statistics[name].attrs.update(units="kg kg-1", sample_mask="jointly finite velocity and humidity")
    statistics.mean_height_upward_moisture_flux.attrs.update(
        units="m s-1", definition="mean(w) * mean(q)", sign_convention="positive upward",
    )
    statistics.mean_upward_pressure_velocity.attrs.update(
        units="Pa s-1", source_variable="omg_isobaric",
        long_name="zonal-time-mean upward pressure velocity",
        sign_convention="positive upward (-omega)",
    )
    statistics.mean_upward_specific_humidity.attrs.update(
        units="kg kg-1", long_name="specific humidity on valid vertical-velocity samples",
        source_conversion="qv_isobaric / (1 + qv_isobaric)",
    )
    statistics.mean_upward_moisture_flux.attrs.update(
        units="Pa s-1", long_name="mean-circulation upward moisture flux",
        definition="[-omega_bar] [q_bar]", sign_convention="positive upward",
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


def _stream_moisture_statistics(
    dataset: xr.Dataset, slices: Sequence[slice], batch_times: int = 4,
    *, progress_label: str = "moisture",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute full-record and original block statistics in bounded time batches.

    Include incomplete tails in the full mean. Sum paired finite counts instead
    of averaging batch means, so missing samples retain the original weighting.
    Keep only one raw-data reduction graph and compact accumulators in memory.
    """

    ntime = dataset.sizes["Time"]
    bounds = [(item.start, item.stop) for item in slices]
    edges = sorted({0, ntime, *(point for pair in bounds for point in pair)})
    full_moments = None
    block_moments = [None] * len(slices)
    started = last_report = perf_counter()
    progress(
        f"{progress_label}: reducing {ntime} archived timestamps in batches of {batch_times}; "
        f"{len(slices)} complete blocks; starting native-w interpolation and flux reduction"
    )
    with dask.config.set(scheduler="threads", num_workers=1):
        for left, right in zip(edges[:-1], edges[1:]):
            block_index = next((index for index, (start, stop) in enumerate(bounds)
                                if start <= left and right <= stop), None)
            for start in range(left, right, batch_times):
                stop = min(start + batch_times, right)
                batch = dataset.isel(Time=slice(start, stop))
                moments = _moisture_moment_sums(
                    batch.umeridional_isobaric, batch.qv_isobaric, batch.omg_isobaric,
                    _native_upward_velocity_on_pressure_levels(batch),
                ).compute()
                full_moments = moments if full_moments is None else full_moments + moments
                if block_index is not None:
                    previous = block_moments[block_index]
                    block_moments[block_index] = moments if previous is None else previous + moments
                now = perf_counter()
                if start == 0 or stop == ntime or now - last_report >= 30.0:
                    progress(
                        f"{progress_label}: moisture reduction {stop}/{ntime} timestamps "
                        f"({100.0 * stop / ntime:.1f}%); elapsed {now - started:.1f} s"
                    )
                    last_report = now
    return _moisture_statistics_from_moments(full_moments), xr.concat(
        [_moisture_statistics_from_moments(item) for item in block_moments],
        dim=xr.IndexVariable("block", np.arange(len(slices))),
    )


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
            fields, block_statistics = _stream_moisture_statistics(
                dataset, slices, progress_label=simulation,
            )
            progress(f"{simulation}: assembling compact moisture sections and profiles")
            records.append(fields)
            block_records.append(
                moisture_profile_block_fields(block_statistics).load()
            )
            progress(f"{simulation}: compact moisture products complete")
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


def _stream_eddy_block_flux(
    flux: xr.DataArray,
    slices: Sequence[slice],
    *,
    progress_label: str,
) -> xr.DataArray:
    """Reduce one timestamp at a time and retain only compact block means."""

    records = []
    started = last_report = perf_counter()
    total = sum(len(range(*item.indices(flux.sizes["Time"]))) for item in slices)
    completed = 0
    with dask.config.set(scheduler="threads", num_workers=2):
        for block_index, item in enumerate(slices):
            summed = counts = template = None
            for index in range(*item.indices(flux.sizes["Time"])):
                sample = flux.isel(Time=index, drop=True).compute()
                values = np.asarray(sample.values)
                finite = ~np.isnan(values)
                if summed is None:
                    template = sample
                    summed = np.zeros_like(values, dtype=np.float64)
                    counts = np.zeros_like(values, dtype=np.int64)
                summed += np.where(finite, values, 0.0)
                counts += finite
                completed += 1
                now = perf_counter()
                if completed == 1 or now - last_report >= 30.0:
                    progress(
                        f"{progress_label}: eddy reduction {completed}/{total} timestamps; "
                        f"block {block_index + 1}/{len(slices)}; elapsed {now - started:.1f} s"
                    )
                    last_report = now
            if template is None:
                raise ValueError("Eddy time blocks must contain at least one timestamp")
            mean = np.full_like(summed, np.nan)
            np.divide(summed, counts, out=mean, where=counts > 0)
            records.append(template.copy(data=mean))
            progress(
                f"{progress_label}: eddy block {block_index + 1}/{len(slices)} complete; "
                f"{completed}/{total} timestamps; elapsed {perf_counter() - started:.1f} s"
            )
    return xr.concat(records, dim=xr.IndexVariable("block", np.arange(len(records))))


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
            progress(
                f"{simulation}: calculating {len(slices)} eddy blocks; "
                "reading one timestamp at a time with stored spatial chunks"
            )
            block_flux = _stream_eddy_block_flux(
                flux, slices, progress_label=simulation,
            )
            block_fields = eddy_block_fields(block_flux, keep_hemispheres=True)
            field_records.append(
                eddy_fields_from_blocks(block_flux).load()
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
        datasets.append(relabel_cached_simulations(load_netcdf(cache_path(filename, cache_root))))
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
        elif family == "mechanism":
            products = build_mechanism_products(
                test=test, test_times=test_times, test_start=test_start,
                simulations=(simulation,),
            )
        elif family == "lead_lag":
            products = build_daily_timing_products(
                test=test, test_times=test_times, test_start=test_start,
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
    run_hemispheres = [row[np.isfinite(row)] for row in np.atleast_2d(run_values)]
    reference_hemispheres = [row[np.isfinite(row)] for row in np.atleast_2d(reference_values)]
    run_count = min(map(len, run_hemispheres))
    reference_count = min(map(len, reference_hemispheres))
    if min(run_count, reference_count) < MIN_BLOCKS:
        raise ValueError(
            f"Insufficient finite blocks per hemisphere ({run_count}, {reference_count})"
        )
    boot = np.empty(N_BOOTSTRAP, dtype=float)
    for index in range(N_BOOTSTRAP):
        boot[index] = np.mean([
            row[rng.integers(row.size, size=row.size)].mean() for row in run_hemispheres
        ]) - np.mean([
            row[rng.integers(row.size, size=row.size)].mean() for row in reference_hemispheres
        ])
    run_values = np.concatenate(run_hemispheres)
    reference_values = np.concatenate(reference_hemispheres)
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
            np.mean([row.mean() for row in run_hemispheres])
            - np.mean([row.mean() for row in reference_hemispheres])
        ),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_noise": noise,
        "response_to_noise": primary_response / noise if noise > 0 else np.nan,
        "sign_robustness": sign_robustness,
        "independent_block_sign_probability": block_sign,
        "n_blocks_run": int(run_count),
        "n_blocks_reference": int(reference_count),
        "n_hemispheres_run": len(run_hemispheres),
        "n_hemispheres_reference": len(reference_hemispheres),
        "ci_excludes_zero": excludes_zero,
        "robust": bool(excludes_zero and sign_robustness >= ROBUST_SIGN_THRESHOLD),
        "bootstrap_replicates": N_BOOTSTRAP,
        "confidence_percentiles": (
            f"{CI_PERCENTILES[0]:g},{CI_PERCENTILES[1]:g}"
        ),
        "resampling_method": (
            "independent complete-block resampling by simulation and hemisphere"
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
                    hemisphere=list(HEMISPHERES[:2]) if hemisphere == "mean" else [hemisphere],
                    metric=metric,
                ).transpose("hemisphere", "block").values
                reference_blocks = metric_blocks.value.sel(
                    simulation=reference,
                    hemisphere=list(HEMISPHERES[:2]) if hemisphere == "mean" else [hemisphere],
                    metric=metric,
                ).transpose("hemisphere", "block").values
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


def _bootstrap_response(
    block_fields, variables, run_index, reference_index, statistic, *, key,
    template_variable=None,
):
    """Calculate each response interval once for all figures."""

    def calculate():
        estimate, samples, template, metadata = response_distribution(
            block_fields, variables, run_index, reference_index, statistic,
            n_resamples=FIELD_BOOTSTRAP_REPLICATES,
            minimum_blocks=MIN_BLOCKS, seed=BOOTSTRAP_SEED,
        )
        if template_variable is not None:
            template = block_fields[template_variable].isel(
                simulation=run_index, block=0, hemisphere=0, drop=True,
            )
        return response_interval(estimate, samples, template, metadata, CI_PERCENTILES)

    return cached_interval(
        block_fields,
        (key, run_index, reference_index, FIELD_BOOTSTRAP_REPLICATES,
         MIN_BLOCKS, BOOTSTRAP_SEED, tuple(CI_PERCENTILES)),
        calculate,
    )


def bootstrap_field_response(
    block_fields: xr.Dataset,
    variable: str,
    run_index: int,
    reference_index: int,
) -> xr.Dataset:
    """Bootstrap the difference between equal-weight hemispheric means."""

    if variable == "section_mean_equatorward_upward_moisture_flux":
        def combined_flux(values):
            wind, upward, humidity = values
            return ((-wind + upward) * humidity).mean(axis=0)

        return _bootstrap_response(
            block_fields,
            ["mean_combined_meridional_wind", "mean_combined_upward_velocity",
             "mean_combined_specific_humidity"],
            run_index, reference_index,
            lambda run, reference: combined_flux(run) - combined_flux(reference),
            key=("moisture_field", variable), template_variable=variable,
        )
    if variable == "section_mean_upward_moisture_flux":
        def upward_flux(values):
            velocity, humidity = values
            return (velocity * humidity).mean(axis=0)

        return _bootstrap_response(
            block_fields,
            ["mean_upward_pressure_velocity", "mean_upward_specific_humidity"],
            run_index, reference_index,
            lambda run, reference: upward_flux(run) - upward_flux(reference),
            key=("moisture_field", variable), template_variable=variable,
        )
    if "mean_specific_humidity" in block_fields:
        return _bootstrap_moisture_field_response(
            block_fields, variable, run_index, reference_index,
        )
    return _bootstrap_response(
        block_fields, [variable], run_index, reference_index,
        lambda run, reference: run[0].mean(axis=0) - reference[0].mean(axis=0),
        key=("field", variable),
    )


def bootstrap_ratio_field_response(
    block_fields: xr.Dataset,
    numerator: str,
    denominator: str,
    run_index: int,
    reference_index: int,
    minimum_absolute_denominator: float,
) -> xr.Dataset:
    """Recompute the plotted ratio of hemispheric means in every replicate."""

    def ratio(values):
        numerator_mean, denominator_mean = values.mean(axis=1)
        return np.divide(
            numerator_mean, denominator_mean,
            out=np.full(numerator_mean.shape, np.nan),
            where=np.abs(denominator_mean) >= minimum_absolute_denominator,
        )

    interval = _bootstrap_response(
        block_fields, [numerator, denominator], run_index, reference_index,
        lambda run, reference: ratio(run) - ratio(reference),
        key=("ratio", numerator, denominator, minimum_absolute_denominator),
    )
    interval.attrs.update(
        statistic="ratio of hemispheric means",
        numerator=numerator,
        denominator=denominator,
        minimum_absolute_denominator=minimum_absolute_denominator,
    )
    return interval


def _plot_profile_bootstrap_interval(
    axis: Any,
    interval: xr.Dataset,
    colour: str,
    alpha: float = 0.16,
) -> None:
    latitude = interval.ci_low.absolute_latitude
    axis.fill_between(
        latitude,
        interval.ci_low,
        interval.ci_high,
        color=colour,
        alpha=alpha,
        linewidth=0.0,
    )


def _sparse_bootstrap_stipple_points(
    significant: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    valid: np.ndarray | None = None,
    max_rows: int = 24,
    max_columns: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Display pointwise significance on a fixed, regularly sampled grid."""

    significant = np.asarray(significant, dtype=bool)
    horizontal = np.asarray(horizontal, dtype=float)
    vertical = np.asarray(vertical, dtype=float)
    if significant.ndim != 2 or significant.shape != horizontal.shape or significant.shape != vertical.shape:
        raise ValueError("Stippling arrays need matching two-dimensional shapes")
    if valid is not None:
        if np.shape(valid) != significant.shape:
            raise ValueError("Stippling validity mask needs a matching shape")
        significant = significant & valid
    significant = significant & np.isfinite(horizontal) & np.isfinite(vertical)
    rows = slice(None, None, max(1, int(np.ceil(significant.shape[0] / max_rows))))
    columns = slice(None, None, max(1, int(np.ceil(significant.shape[1] / max_columns))))
    selected = significant[rows, columns]
    return horizontal[rows, columns][selected], vertical[rows, columns][selected]


def _plot_bootstrap_stippling(
    axis: Any,
    interval: xr.Dataset,
    vertical_coordinate: xr.DataArray | None = None,
) -> None:
    excludes_zero = (interval.ci_low > 0.0) | (interval.ci_high < 0.0)
    valid_interval = np.isfinite(interval.ci_low) & np.isfinite(interval.ci_high)
    if vertical_coordinate is not None:
        excludes_zero, valid_interval, vertical_coordinate = xr.align(
            excludes_zero,
            valid_interval,
            vertical_coordinate,
            join="exact",
        )
        excludes_zero = excludes_zero.transpose("z", "absolute_latitude")
        valid_interval = valid_interval.transpose("z", "absolute_latitude")
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
        valid_interval = valid_interval.transpose("z", "absolute_latitude")
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
        valid_interval = valid_interval.transpose("pres", "absolute_latitude")
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
        valid=np.asarray(valid_interval.values, dtype=bool),
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
        "mean_upward_pressure_velocity",
        "mean_upward_specific_humidity",
        "mean_upward_velocity",
        "mean_height_upward_specific_humidity",
        "mean_combined_meridional_wind",
        "mean_combined_upward_velocity",
        "mean_combined_specific_humidity",
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
    completed["mean_upward_moisture_flux"] = (
        completed.mean_upward_pressure_velocity
        * completed.mean_upward_specific_humidity
    )
    completed.mean_upward_moisture_flux.attrs.update(
        units="Pa s-1", long_name="mean-circulation upward moisture flux",
        definition="[-omega_bar] [q_bar]", sign_convention="positive upward",
    )
    completed["mean_height_upward_moisture_flux"] = (
        completed.mean_upward_velocity * completed.mean_height_upward_specific_humidity
    )
    completed.mean_height_upward_moisture_flux.attrs.update(
        units="m s-1", definition="mean(w) * mean(q)", sign_convention="positive upward",
    )
    completed.attrs.update(fields.attrs)
    completed.attrs["bottom_boundary_treatment"] = (
        f"fixed {MOISTURE_BOTTOM_PRESSURE_HPA:g} hPa lower boundary; "
        f"{MOISTURE_BOTTOM_SOURCE_HPA:g} hPa statistics held constant below"
    )
    return completed


def fold_equatorward_transport(
    field: xr.DataArray, *, keep_hemispheres: bool = False,
) -> xr.DataArray:
    """Fold a northward flux so positive values point toward the equator."""

    folded = -fold_to_absolute(field, "odd", keep_hemispheres=keep_hemispheres)
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


def equatorward_transport_convergence(
    equatorward_transport: xr.DataArray,
    radius_m: float = EARTH_RADIUS_M,
) -> xr.DataArray:
    """Return convergence from an absolute-latitude equatorward transport."""

    if "absolute_latitude" not in equatorward_transport.dims:
        raise ValueError(
            "Equatorward transport needs an absolute_latitude dimension"
        )
    transport = equatorward_transport.sortby("absolute_latitude")
    latitude = np.asarray(transport.absolute_latitude.values, dtype=float)
    if latitude.size < 2:
        raise ValueError("Moisture convergence needs two latitude points")
    phi = np.deg2rad(latitude)
    cos_phi = np.cos(phi)
    axis = transport.get_axis_num("absolute_latitude")
    shape = [
        latitude.size if index == axis else 1
        for index in range(transport.ndim)
    ]
    weighted = np.asarray(transport.values, dtype=float) * np.reshape(
        cos_phi,
        shape,
    )
    edge_order = 2 if latitude.size >= 3 else 1
    gradient = np.gradient(weighted, phi, axis=axis, edge_order=edge_order)
    values = gradient / (radius_m * np.reshape(cos_phi, shape))
    convergence = xr.DataArray(
        values,
        dims=transport.dims,
        coords=transport.coords,
        name=transport.name,
    )
    convergence.attrs.update(
        units="kg m-2 s-1",
        long_name="moisture-flux convergence",
        formula=(
            "1/(a cos(abs(phi))) d(cos(abs(phi)) F_equatorward)/dabs(phi)"
        ),
        sign_convention="positive for convergence",
    )
    return convergence


def folded_moisture_sections(fields: xr.Dataset) -> xr.Dataset:
    """Return equatorward and upward mean-circulation moisture sections."""

    sections = xr.Dataset(
        {
            name: fold_equatorward_transport(fields[name])
            for name in MOISTURE_FLUX_COMPONENTS
        } | {
            "mean_upward_moisture_flux": fold_to_absolute(fields.mean_upward_moisture_flux, "even"),
            "mean_height_upward_moisture_flux": fold_to_absolute(fields.mean_height_upward_moisture_flux, "even"),
        }
    )
    sections["mean_equatorward_upward_moisture_flux"] = ((
        fold_equatorward_transport(fields.mean_combined_meridional_wind, keep_hemispheres=True)
        + fold_to_absolute(fields.mean_combined_upward_velocity, "even", keep_hemispheres=True)
    ) * fold_to_absolute(fields.mean_combined_specific_humidity, "even", keep_hemispheres=True)).mean("hemisphere")
    sections.mean_equatorward_upward_moisture_flux.attrs.update(
        units="m s-1", definition="(mean(v_equatorward) + mean(w)) * mean(q)",
        sign_convention="positive equatorward plus upward",
        sample_mask="jointly finite v, w and q",
    )
    return sections


def folded_moisture_profiles(
    fields: xr.Dataset, *, keep_hemispheres: bool = False,
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
            name: fold_equatorward_transport(northward[name], keep_hemispheres=keep_hemispheres)
            for name in MOISTURE_FLUX_COMPONENTS
        }
    )
    convergence = xr.Dataset(
        {
            name: fold_to_absolute(
                zonal_mean_moisture_flux_convergence(northward[name]),
                "even", keep_hemispheres=keep_hemispheres,
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
    transports, convergences = folded_moisture_profiles(completed, keep_hemispheres=True)
    fields = {}
    for component in MOISTURE_FLUX_COMPONENTS:
        fields[f"transport_{component}"] = transports[component]
        fields[f"convergence_{component}"] = convergences[component]
    fields["section_total_meridional_moisture_flux"] = (
        fold_equatorward_transport(
            completed.total_meridional_moisture_flux, keep_hemispheres=True,
        )
    )
    fields["section_mean_meridional_moisture_flux"] = (
        fold_equatorward_transport(
            completed.mean_meridional_moisture_flux, keep_hemispheres=True,
        )
    )
    fields["mean_meridional_wind"] = hemisphere_samples(completed.mean_meridional_wind, "odd")
    fields["mean_specific_humidity"] = hemisphere_samples(completed.mean_specific_humidity, "even")
    fields["mean_upward_pressure_velocity"] = hemisphere_samples(completed.mean_upward_pressure_velocity, "even")
    fields["mean_upward_specific_humidity"] = hemisphere_samples(completed.mean_upward_specific_humidity, "even")
    fields["section_mean_upward_moisture_flux"] = hemisphere_samples(completed.mean_upward_moisture_flux, "even")
    fields["mean_upward_velocity"] = hemisphere_samples(completed.mean_upward_velocity, "even")
    fields["mean_height_upward_specific_humidity"] = hemisphere_samples(completed.mean_height_upward_specific_humidity, "even")
    for name, parity in (("mean_combined_meridional_wind", "odd"),
                         ("mean_combined_upward_velocity", "even"),
                         ("mean_combined_specific_humidity", "even")):
        fields[name] = hemisphere_samples(completed[name], parity)
    fields["section_mean_equatorward_upward_moisture_flux"] = (
        -fields["mean_combined_meridional_wind"] + fields["mean_combined_upward_velocity"]
    ) * fields["mean_combined_specific_humidity"]
    fields["section_mean_equatorward_upward_moisture_flux"].attrs.update(
        units="m s-1", definition="(mean(v_equatorward) + mean(w)) * mean(q)",
        sign_convention="positive equatorward plus upward",
    )
    result = xr.Dataset(fields)
    result.attrs.update(
        uncertainty_source="complete time blocks",
        bootstrap_method="independent resampling by simulation and hemisphere",
    )
    return result


def _integrated_moisture_values(section: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Integrate arrays ending in (pressure, latitude), retaining finite columns."""

    order = np.argsort(pressure)
    pressure = pressure[order]
    columns = np.moveaxis(section[..., order, :], -2, -1)
    flat = columns.reshape(-1, pressure.size)
    finite = np.isfinite(flat)
    integrated = np.trapz(flat, pressure, axis=-1)
    for index in np.flatnonzero(~finite.all(axis=-1)):
        valid = finite[index]
        integrated[index] = (
            np.trapz(flat[index, valid], pressure[valid])
            if valid.sum() >= 2 else np.nan
        )
    return integrated.reshape(columns.shape[:-1]) * 100.0 / GRAVITY_M_S2


def _bootstrap_moisture_field_response(blocks, variable, run_index, reference_index):
    """Recompute mean and eddy fluxes from joint wind/humidity draws."""

    kind, component = variable.split("_", 1)
    latitude = np.asarray(blocks.absolute_latitude.values)
    pressure = np.asarray(blocks.pres.values)

    def climate(values):
        total, wind, humidity = values
        mean_flux = -wind * humidity  # Equatorward-positive after the odd reflection.
        section = {
            "total_meridional_moisture_flux": total,
            "mean_meridional_moisture_flux": mean_flux,
            "eddy_meridional_moisture_flux": total - mean_flux,
        }[component]
        if kind == "section":
            return section.mean(axis=0)
        transport = _integrated_moisture_values(section, pressure).mean(axis=0)
        if kind == "transport":
            return transport
        # Reflect the folded transport to preserve the global derivative stencil.
        positive = latitude > 0.0
        signed_latitude = np.concatenate([-latitude[positive][::-1], latitude])
        extended = np.concatenate([-transport[positive][::-1], transport])
        phi = np.deg2rad(signed_latitude)
        convergence = np.gradient(extended * np.cos(phi), phi, edge_order=2)
        convergence /= EARTH_RADIUS_M * np.cos(phi)
        return convergence[-latitude.size:]

    return _bootstrap_response(
        blocks,
        ["section_total_meridional_moisture_flux", "mean_meridional_wind", "mean_specific_humidity"],
        run_index, reference_index,
        lambda run, reference: climate(run) - climate(reference),
        key=("moisture_field", variable), template_variable=variable,
    )


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
) -> dict[str, xr.Dataset]:
    """Recompute moisture products with joint wind/humidity draws per hemisphere."""

    def calculate():
        names = tuple(_MEAN_RESPONSE_STYLES)
        all_names = names + tuple(f"convergence_{name}" for name in names)
        latitude = block_fields.absolute_latitude
        pressure = np.asarray(block_fields.pres.values)
        phi = np.deg2rad(latitude.values)
        cosine = np.cos(phi)

        def statistic(run, reference):
            wind_change = run[0] - reference[0]
            humidity_change = run[1] - reference[1]
            sections = {
                "wind_contribution": wind_change * reference[1],
                "humidity_contribution": reference[0] * humidity_change,
                "interaction": wind_change * humidity_change,
            }
            sections["mean_circulation_change"] = sum(sections.values())
            # Corrected wind is poleward-positive in both hemispheres.
            # Integrate products within each hemisphere before averaging.
            profiles = []
            for name in names:
                transport = -_integrated_moisture_values(sections[name], pressure)
                profiles.append(transport.mean(axis=0))
            profiles = np.asarray(profiles)
            convergence = np.gradient(
                profiles * cosine, phi, axis=-1,
                edge_order=2 if latitude.size >= 3 else 1,
            ) / (EARTH_RADIUS_M * cosine)
            return np.concatenate([profiles, convergence], axis=0)

        estimate, samples, _template, metadata = response_distribution(
            block_fields, ["mean_meridional_wind", "mean_specific_humidity"],
            run_index, reference_index, statistic,
            n_resamples=FIELD_BOOTSTRAP_REPLICATES,
            minimum_blocks=MIN_BLOCKS, seed=BOOTSTRAP_SEED,
        )
        template = xr.DataArray(
            np.empty((len(all_names), latitude.size)),
            dims=("term", "absolute_latitude"),
            coords={"term": list(all_names), "absolute_latitude": latitude},
        )
        intervals = response_interval(estimate, samples, template, metadata, CI_PERCENTILES)
        return {
            name: intervals.sel(term=name, drop=True).assign_attrs(
                units="kg m-2 s-1" if name.startswith("convergence_") else "kg m-1 s-1",
            ) for name in all_names
        }

    return cached_interval(
        block_fields,
        ("moisture_attribution", run_index, reference_index, FIELD_BOOTSTRAP_REPLICATES,
         MIN_BLOCKS, BOOTSTRAP_SEED, tuple(CI_PERCENTILES)),
        calculate,
    )


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


HC_EDGE_LINE_STYLE = {
    "color": "0.3", "linestyle": ":", "linewidth": 0.9, "alpha": 0.85,
}


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
    axis.set_xticks(np.arange(0.0, 71.0, 10.0))
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
            )
            _plot_bootstrap_stippling(axis, interval)
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
                label="HC edge" if column == 0 else None,
                **HC_EDGE_LINE_STYLE,
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
        "; stippling shows pointwise 95% intervals excluding zero"
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
            )
            _plot_bootstrap_stippling(
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
            )
            lower_interval = bootstrap_field_response(
                field_blocks,
                "zonal_wind_850",
                run_index,
                reference_index,
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
                    **(HC_EDGE_LINE_STYLE if metric == "hadley_edge_latitude" else {
                        "color": style["color"], "linestyle": style["linestyle"],
                        "linewidth": 0.75, "alpha": 0.65,
                    }),
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
        "; stippling shows pointwise 95% intervals excluding zero; shading shows intervals"
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
                **(HC_EDGE_LINE_STYLE if metric == "hadley_edge_latitude" else {
                    "color": style["color"], "linestyle": line_style,
                    "linewidth": 0.8, "alpha": 0.7,
                }),
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
            )
            convergence_interval = bootstrap_field_response(
                field_blocks,
                "eddy_momentum_flux_convergence",
                run_index,
                reference_index,
            )
            _plot_bootstrap_stippling(
                axes[0, column], flux_interval
            )
            _plot_bootstrap_stippling(
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
                    Line2D([0], [0], label="HC edge", **HC_EDGE_LINE_STYLE),
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
        "; stippling shows pointwise 95% intervals excluding zero; shading shows intervals"
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
    for simulation_index, _style in _comparison_styles(
        run_index,
        reference_index,
    ):
        axis.axvline(
            _metric_value(
                metrics,
                simulation_index,
                "hadley_edge_latitude",
            ),
            **HC_EDGE_LINE_STYLE,
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
            )
            _plot_bootstrap_stippling(
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
                )
                convergence_interval = bootstrap_field_response(
                    moisture_blocks,
                    f"convergence_{variable}",
                    run_index,
                    reference_index,
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
        "; stippling shows pointwise 95% intervals excluding zero; shading shows intervals"
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
    show_stj: bool = True,
) -> None:
    for position, (simulation_index, style) in enumerate(
        _comparison_styles(run_index, reference_index)
    ):
        for metric, line_style, label in (
            ("hadley_edge_latitude", ":", "HC edge"),
            ("stj_latitude", "-", "STJ"),
        ):
            if metric == "stj_latitude" and not show_stj:
                continue
            axis.axvline(
                _metric_value(circulation_metrics, simulation_index, metric),
                label=label if legend and position == 0 else None,
                **(HC_EDGE_LINE_STYLE if metric == "hadley_edge_latitude" else {
                    "color": style["color"], "linestyle": line_style,
                    "linewidth": 0.9, "alpha": 0.75,
                }),
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
        tem_formulation=tem_fields.attrs.get("tem_formulation", "geometric-height"),
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


def _plot_nudging_region_guides(
    axis: Any,
    pressure: xr.DataArray,
    simulations: Sequence[str],
    regions: dict[str, tuple[NudgingRegion, ...]],
) -> list[Line2D]:
    """Map analytic e-fold contours through the common TEM pressure grid."""

    if pressure.attrs.get("tem_formulation", "geometric-height") != "geometric-height":
        raise ValueError("Nudging contours require geometric height, not log-pressure height")
    handles = []
    for simulation in dict.fromkeys(canonical_simulation_name(name) for name in simulations):
        for region in regions[simulation]:
            latitude, height = region.contour()
            # Use the nearest equatorial pressure when the grid omits zero latitude.
            lookup_latitude = np.maximum(latitude, float(pressure.absolute_latitude.min()))
            mapped = pressure.interp(
                z=xr.DataArray(height, dims="vertex"),
                absolute_latitude=xr.DataArray(lookup_latitude, dims="vertex"),
            )
            variable = r"$u$" if region.variable == "u" else r"$\theta$"
            label = f"{ALL_SIMULATION_LABELS[simulation]} kernel ($e^{{-1}}$)"
            if len(regions[simulation]) > 1:
                label = f"{ALL_SIMULATION_LABELS[simulation]} {variable} kernel ($e^{{-1}}$)"
            line, = axis.plot(
                latitude, mapped, color=simulation_colour(simulation),
                linestyle=":" if simulation == "control" else (
                    "-." if region.variable == "u" else "--"
                ),
                linewidth=1.25, zorder=9.2, label=label,
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
            )
            handles.append(line)
    return handles


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
                )
                _plot_bootstrap_stippling(
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
        "; stippling shows pointwise 95% intervals excluding zero"
        if show_bootstrap
        else ""
    )
    fig.suptitle(
        "TEM residual circulation linking the UTLS and upper Hadley branch"
        + uncertainty_note
    )
    return fig


_TEM_BUDGET_LABELS = {
    "a_density_storage": "Density-wind covariance storage",
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
    "a_density_storage": {"color": "#e7298a", "linestyle": ":"},
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
    nudging_regions: dict[str, tuple[NudgingRegion, ...]] | None = None,
) -> plt.Figure:
    """Plot the resolved TEM momentum budget and unresolved residual."""

    field_names = [
        "a_mmc",
        "a_ep",
        "unresolved_direct_momentum_residual",
    ]
    if "a_density_storage" in tem_fields and _field_rms(tem_fields.a_density_storage) > 1e-15:
        field_names.insert(2, "a_density_storage")
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
    # Keep every budget response comparable with the EP acceleration in figure 10.
    response_levels = _ep_acceleration_response_levels(tem_fields)
    pressure_grid = _tem_reference_pressure_grid(tem_fields)
    if nudging_regions is None:
        nudging_regions = load_nudging_regions(ROOT_DIR, tem_fields.simulation.values)
    nudging_handles = {}
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
            colourmap = difference_cmap(
                contour_levels, cmap="RdBu_r", white_zero_boundary=True,
            )
            image = axes[row, column].contourf(
                horizontal,
                vertical,
                field,
                levels=contour_levels,
                cmap=colourmap,
                norm=BoundaryNorm(contour_levels, colourmap.N),
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
                )
                _plot_bootstrap_stippling(
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
            participants = [str(tem_fields.simulation.values[run_index])]
            if reference_index is not None:
                participants.append(str(tem_fields.simulation.values[reference_index]))
            for handle in _plot_nudging_region_guides(
                axes[row, column], pressure_grid, participants, nudging_regions,
            ):
                nudging_handles[handle.get_label()] = handle
            _set_latpres_axis(axes[row, column], ylabel=column == 0)
            if row == 0:
                annotate_panel_label(axes[row, column], comparison_label)
                axes[row, column].texts[-1].set_zorder(10)
                axes[row, column].texts[-1].set_position((0.97, 0.95))
                axes[row, column].texts[-1].set_ha("right")
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
        for name in _TEM_PROFILE_STYLES if f"stj_layer_{name}" in tem_fields
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
        axis.set_xticks(np.arange(0.0, 71.0, 10.0))
        axis.set_xlabel("Absolute latitude (degrees)")
    profile_limit = _profile_response_limit(profile_values)
    for axis in axes[profile_row, :]:
        axis.set_ylim(-1.08 * profile_limit, 1.08 * profile_limit)
    axes[profile_row, 0].set_ylabel(
        f"{TEM_STJ_LAYER_TOP_HPA:g}-{TEM_STJ_LAYER_BOTTOM_HPA:g} hPa\n"
        r"acceleration (m s$^{-1}$ day$^{-1}$)"
    )
    axes[profile_row, 0].legend(loc="best", fontsize=6.5, frameon=False)
    if nudging_handles:
        fig.legend(
            list(nudging_handles.values()), list(nudging_handles),
            loc="outside lower center", ncol=len(nudging_handles),
            fontsize=8, frameon=False,
        )
    tendency_note = (
        "Archived mean tendency shown"
        if show_tendency
        else "Archived mean tendency is cached and small"
    )
    uncertainty_note = (
        "; stippling shows pointwise 95% intervals excluding zero; shading shows intervals"
        if show_bootstrap
        else ""
    )
    fig.suptitle(
        "Resolved TEM zonal-momentum budget and unresolved/direct residual\n"
        + tendency_note
        + uncertainty_note,
        fontsize=12,
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
                )
                _plot_bootstrap_stippling(
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
        "; stippling shows pointwise 95% intervals excluding zero"
        if show_bootstrap
        else ""
    )
    fig.suptitle(
        "TEM residual-circulation acceleration decomposition"
        + uncertainty_note
    )
    return fig


def effective_coriolis_factor_from_cached_terms(
    fields: xr.Dataset,
) -> xr.DataArray:
    """Return f_eff = A_vres / vres from compact cached TEM terms."""

    a_vres, vres = xr.align(fields.a_vres, fields.vres, join="exact")
    valid = (
        np.isfinite(a_vres)
        & np.isfinite(vres)
        & (np.abs(vres) >= F_EFF_MINIMUM_ABSOLUTE_VRES_M_S)
    )
    result = (a_vres / vres).where(valid)
    result.name = "f_eff"
    result.attrs.update(
        units="s-1",
        long_name="effective Coriolis factor",
        formula="f_eff = A_vres / vres",
        source="derived from compact cached TEM terms",
        denominator_mask=(
            f"abs(vres) >= {F_EFF_MINIMUM_ABSOLUTE_VRES_M_S:g} m s-1"
        ),
    )
    return result


def plot_tem_effective_coriolis_factor(
    tem_fields: xr.Dataset,
    circulation_metrics: xr.Dataset,
    field_blocks: xr.Dataset,
    show_bootstrap: bool,
) -> plt.Figure:
    """Plot f_eff from cached meridional acceleration and velocity."""

    effective_fields = tem_fields.assign(
        f_eff=effective_coriolis_factor_from_cached_terms(tem_fields)
    )
    fields = [
        field * SECONDS_PER_DAY
        for field in comparison_fields(effective_fields, "f_eff")
    ]
    raw_levels = symmetric_percentile_levels_for_fields(
        _tem_colour_scale_fields([fields[0]]),
        percentile=99.0,
        ndiv=13,
        zero_field_scale=10.0,
    )
    response_levels = symmetric_percentile_levels_for_fields(
        _tem_colour_scale_fields(fields[1:]),
        percentile=95.0,
        ndiv=13,
        zero_field_scale=1.0,
    )
    pressure_grid = _tem_reference_pressure_grid(tem_fields)
    fig, axes = plt.subplots(
        1,
        len(COMPARISONS),
        figsize=(3.9 * len(COMPARISONS), 3.8),
        constrained_layout=True,
        sharex="col",
        sharey=True,
        squeeze=False,
    )
    control = fields[0]
    control_horizontal, control_vertical = _tem_contour_coordinates(
        control,
        pressure_grid,
    )
    images = []
    for column, (
        (_name, run_index, reference_index, comparison_label),
        field,
    ) in enumerate(zip(COMPARISONS, fields)):
        axis = axes[0, column]
        levels = raw_levels if column == 0 else response_levels
        horizontal, vertical = _tem_contour_coordinates(field, pressure_grid)
        image = axis.contourf(
            horizontal,
            vertical,
            field,
            levels=levels,
            cmap=difference_cmap(levels),
            extend="both",
        )
        images.append(image)
        if column > 0:
            axis.contour(
                control_horizontal,
                control_vertical,
                control,
                levels=raw_levels[::2],
                colors="0.4",
                linewidths=0.4,
                alpha=0.45,
            )
        if show_bootstrap and reference_index is not None:
            interval = bootstrap_ratio_field_response(
                field_blocks,
                "a_vres",
                "vres",
                run_index,
                reference_index,
                F_EFF_MINIMUM_ABSOLUTE_VRES_M_S,
            )
            _plot_bootstrap_stippling(axis, interval, pressure_grid)
        _plot_tem_location_guides(
            axis,
            circulation_metrics,
            run_index,
            reference_index,
            legend=column == 0,
        )
        _set_latpres_axis(axis, ylabel=column == 0)
        annotate_panel_label(axis, comparison_label)
    axes[0, 0].legend(loc="lower right", fontsize=7, frameon=False)
    label = r"$f_{\mathrm{eff}}=A_{v^*}/v^*$ (day$^{-1}$)"
    fig.colorbar(images[0], ax=axes[:, 0], label=label)
    fig.colorbar(images[-1], ax=axes[:, 1:], label=f"{label} response")
    uncertainty_note = (
        "; stippling shows pointwise 95% intervals excluding zero"
        if show_bootstrap
        else ""
    )
    fig.suptitle(
        r"Effective Coriolis factor $f_{\mathrm{eff}}=A_{v^*}/v^*$"
        + uncertainty_note
    )
    return fig


def _quiver_example_value(reference: float) -> float:
    """Choose a readable example magnitude from the 1, 2, 5 sequence."""

    target = reference
    decade = 10.0 ** np.floor(np.log10(target))
    candidates = decade * np.array([1.0, 2.0, 5.0, 10.0])
    return float(candidates[np.argmin(np.abs(np.log(candidates / target)))])


def _quiver_example_label(symbol: str, value: float, units: str) -> str:
    """Label an example vector with its physical component magnitude."""

    if value >= 1.0e4 or value < 1.0e-2:
        exponent = int(np.floor(np.log10(value)))
        number = rf"{value / 10.0**exponent:g}\times10^{{{exponent}}}"
    else:
        number = f"{value:g}"
    return rf"${symbol}={number}$ {units}"


def _add_component_quiver_keys(
    axis: Any,
    horizontal_fraction: float,
    vertical_fraction: float,
    horizontal_label: str,
    vertical_label: str,
    *,
    width: float,
    common_units: str | None = None,
    offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[Any, Any]:
    """Draw two calibrated reference arrows and adjacent labels in one shaded box."""

    from matplotlib.transforms import Affine2D

    key_transform = Affine2D().translate(*offset) + axis.transAxes
    horizontal_x = 0.45 + horizontal_fraction / 2.0
    vertical_x = horizontal_x
    label_left = max(0.57, 0.47 + horizontal_fraction)
    label_x = (label_left + 0.96) / 2.0
    box_top = 0.985
    texts = [axis.text(
        label_x, 0.0, label, transform=key_transform,
        ha="center", va="center", fontsize=5.5, color="0.08", zorder=9,
    ) for label in (horizontal_label, vertical_label)]
    unit_text = None
    if common_units is not None:
        unit_text = axis.text(
            0.705, 0.0, common_units, transform=key_transform,
            ha="center", va="center", fontsize=4.5, color="0.08", zorder=9,
        )
    # Size the box from the rendered text, including mathematical superscripts.
    axis_height = axis.get_window_extent().height
    text_heights = [text.get_window_extent().height / axis_height for text in texts]
    row_heights = [text_heights[0], max(text_heights[1], vertical_fraction)]
    padding = 2.0 * axis.figure.dpi / 72.0 / axis_height
    units_height = unit_text.get_window_extent().height / axis_height if unit_text else 0.0
    box_height = sum(row_heights) + 3.0 * padding
    if unit_text is not None:
        box_height += units_height + padding
    box_bottom = box_top - box_height
    horizontal_y = box_top - padding - row_heights[0] / 2.0
    vertical_y = horizontal_y - row_heights[0] / 2.0 - padding - row_heights[1] / 2.0
    for text, y in zip(texts, (horizontal_y, vertical_y)):
        text.set_position((label_x, y))
    if unit_text is not None:
        unit_text.set_position((0.705, box_bottom + padding + units_height / 2.0))
    axis.add_patch(Rectangle(
        (0.43, box_bottom), 0.55, box_height,
        transform=key_transform, facecolor="0.94", edgecolor="0.65",
        linewidth=0.4, alpha=0.95, zorder=8, clip_on=False,
    ))
    keys = []
    for vertical, x, y, length in (
        (False, horizontal_x, horizontal_y, horizontal_fraction),
        (True, vertical_x, vertical_y, vertical_fraction),
    ):
        arrow = axis.quiver(
            [x], [y], [0.0 if vertical else length], [length if vertical else 0.0],
            transform=key_transform, angles="uv", pivot="mid",
            scale_units="height" if vertical else "width", scale=1.0,
            width=max(width, 0.0045), headwidth=3.8, headlength=4.5,
            minlength=0, color="0.08", edgecolor="0.08", linewidth=0.0,
            alpha=1.0, zorder=9, clip_on=False,
        )
        arrow.set_gid("vertical_reference_vector" if vertical else "horizontal_reference_vector")
        keys.append(arrow)
    return tuple(keys)


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
    meridional_flux = comparison_fields(tem_fields, "ep_flux_meridional_normalized")
    vertical_flux = comparison_fields(tem_fields, "ep_flux_vertical_normalized")
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
    vector_requests = []
    tropopause = circulation_fields.tropopause_pressure.isel(simulation=0, drop=True)
    vector_grids = [_publication_vector_grid(
        fy, fz, pressure_grid, tropopause, "full", kind="ep",
    ) for fy, fz in zip(meridional_flux, vertical_flux)]
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
                )
                _plot_publication_scalar_hatching(
                    axes[row, column],
                    interval,
                    pressure_grid,
                    latitude_limit=70.0,
                )
            if row == 0:
                vector_requests.append(dict(
                    axis=axes[row, column], group="control" if column == 0 else "response",
                    grid=vector_grids[column], reference_grid=vector_grids[column],
                    pressure=pressure_grid,
                    intervals=_bootstrap_vector_intervals(
                        tem_blocks, pressure_grid, "ep", run_index, reference_index, 70.0,
                    ) if show_bootstrap and reference_index is not None else [],
                ))
                streamfunction_for_contour = streamfunction[column]
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
        loc="lower left",
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
        "; hatching shows insignificant responses (pointwise 95% intervals include zero)"
        if show_bootstrap
        else ""
    )
    fig.suptitle(
        "Resolved EP-flux propagation and acceleration decomposition"
        + uncertainty_note
    )
    _draw_vector_requests(fig, vector_requests)
    return fig


def _publication_comparison_fields(
    fields: xr.Dataset,
    variable: str,
) -> list[xr.DataArray]:
    """Return control and the two responses used by the synthesis figures."""

    return comparison_fields(fields, variable)[:3]


def _publication_shading_levels(boundaries: np.ndarray) -> np.ndarray:
    """Round publication colour boundaries to the shared readable interval sequence."""

    rounded = nice_levels(float(boundaries[0]), float(boundaries[-1]), n=boundaries.size)
    tolerance = float(np.min(np.diff(rounded))) * 1.0e-10
    rounded[np.isclose(rounded, 0.0, rtol=0.0, atol=tolerance)] = 0.0
    return rounded


def _ep_acceleration_response_levels(tem_fields: xr.Dataset) -> np.ndarray:
    """Share the figure 10 EP acceleration scale with figure 6 budget responses."""

    responses = [
        field.sel(absolute_latitude=slice(0.0, 50.0)) * SECONDS_PER_DAY
        for field in _publication_comparison_fields(tem_fields, "a_ep")[1:]
    ]
    return _publication_shading_levels(
        symmetric_percentile_levels_for_fields(
            _tem_colour_scale_fields(responses),
            percentile=98.0, ndiv=17, zero_field_scale=0.01,
        )
    )


def _percentile_levels(
    field: xr.DataArray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    count: int = 14,
) -> np.ndarray:
    """Return finite, evenly spaced percentile levels for an absolute field."""

    values = np.asarray(field.values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Colour-scale field has no finite values")
    lower, upper = np.nanpercentile(
        finite,
        [lower_percentile, upper_percentile],
    )
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("Colour-scale percentiles do not span a finite range")
    return np.linspace(float(lower), float(upper), count)


def _nonzero_contour_levels(levels: np.ndarray, count: int = 7) -> np.ndarray:
    """Return a sparse set of signed contour levels without the zero contour."""

    values = np.asarray(levels, dtype=float)
    if values.size > count:
        indices = np.linspace(0, values.size - 1, count).round().astype(int)
        values = values[np.unique(indices)]
    scale = float(np.nanmax(np.abs(values))) if values.size else 0.0
    tolerance = max(np.finfo(float).eps, 1.0e-10 * scale)
    return values[np.abs(values) > tolerance]


def _plot_signed_contours(
    axis: Any,
    horizontal: xr.DataArray | np.ndarray,
    vertical: xr.DataArray | np.ndarray,
    field: xr.DataArray,
    levels: np.ndarray,
    colour: str = "0.12",
    linewidth: float = 0.75,
    alpha: float = 0.85,
    count: int = 7,
) -> Any:
    """Plot sparse signed contours with dashed negative values."""

    selected = _nonzero_contour_levels(levels, count=count)
    contours = axis.contour(
        horizontal,
        vertical,
        field,
        levels=selected,
        colors=colour,
        linestyles=["--" if value < 0.0 else "-" for value in selected],
        linewidths=linewidth,
        alpha=alpha,
        zorder=3,
    )
    return contours


def _publication_pressure_levels(top_hpa: float = 50.0) -> np.ndarray:
    """Use a regular log-pressure display grid within the plotted domain."""

    return np.geomspace(top_hpa, 1000.0, 121)


def _publication_pressure_field(
    field: xr.DataArray,
    pressure_levels: np.ndarray,
    native_pressure: xr.DataArray | None = None,
    *,
    nearest: bool = False,
) -> xr.DataArray:
    """Interpolate linearly in log pressure, preserving component sums and missing regions."""

    if native_pressure is None:
        field = field.transpose("pres", "absolute_latitude")
        source_pressure = np.broadcast_to(
            np.asarray(field.pres.values, dtype=float)[:, None],
            field.shape,
        )
    else:
        field, native_pressure = xr.align(field, native_pressure, join="exact")
        field = field.transpose("z", "absolute_latitude")
        source_pressure = np.asarray(
            native_pressure.transpose("z", "absolute_latitude").values,
            dtype=float,
        )
    source = np.asarray(field.values, dtype=float)
    output = np.full(
        (pressure_levels.size, field.sizes["absolute_latitude"]),
        np.nan,
        dtype=float,
    )
    target = np.log(pressure_levels)
    if native_pressure is not None and native_pressure.attrs.get("tem_formulation") == "log-pressure" and not nearest:
        # Preserve explicit legacy pressure-grid interpolation without extrapolation.
        if not np.allclose(source_pressure, source_pressure[:, :1], atol=1e-10, rtol=0):
            raise ValueError("Log-pressure TEM requires common pressure surfaces")
        return field.rename(z="pres").assign_coords(pres=source_pressure[:, 0]).sortby("pres").interp(pres=pressure_levels)
    for column in range(source.shape[1]):
        valid = (
            np.isfinite(source_pressure[:, column])
            & (source_pressure[:, column] > 0.0)
        )
        if np.count_nonzero(valid) < 2:
            continue
        order = np.argsort(source_pressure[valid, column])
        pressure = np.log(source_pressure[valid, column][order])
        values = source[valid, column][order]
        unique = np.r_[True, np.diff(pressure) > 0.0]
        if np.count_nonzero(unique) < 2:
            continue
        if nearest:
            # Keep categorical significance on its nearest native log-pressure level.
            pressure, values = pressure[unique], values[unique]
            indices = np.abs(target[:, None] - pressure[None, :]).argmin(axis=1)
            inside = (target >= pressure[0]) & (target <= pressure[-1])
            output[inside, column] = values[indices[inside]]
        else:
            pressure, values = pressure[unique], values[unique]
            # Preserve missing regions instead of interpolating across invalid TEM cells.
            finite = np.isfinite(values)
            edges = np.diff(np.r_[False, finite, False].astype(int))
            for begin, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if stop - begin < 2:
                    continue
                inside = (target >= pressure[begin]) & (target <= pressure[stop - 1])
                output[inside, column] = np.interp(
                    target[inside], pressure[begin:stop], values[begin:stop],
                )
    return xr.DataArray(
        output,
        dims=("pres", "absolute_latitude"),
        coords={
            "pres": pressure_levels,
            "absolute_latitude": field.absolute_latitude,
        },
        attrs=field.attrs,
    )


def _publication_smooth_contours(
    field: xr.DataArray,
    latitude_sigma: float = 3.0,
) -> xr.DataArray:
    """Smooth line overlays across adjacent display pixels only."""

    values = np.asarray(field.values, dtype=float)
    valid = np.isfinite(values)
    smoothing = (1.5, latitude_sigma)
    weights = gaussian_filter(valid.astype(float), sigma=smoothing)
    smoothed = gaussian_filter(
        np.where(valid, values, 0.0), sigma=smoothing
    )
    smoothed = np.divide(
        smoothed,
        weights,
        out=np.full_like(smoothed, np.nan),
        where=weights > 0.8,
    )
    return field.copy(data=smoothed)


def _plot_publication_jet_and_tropopause(
    axis: Any,
    circulation_fields: xr.Dataset,
    run_index: int,
    reference_index: int | None,
    pressure_levels: np.ndarray,
    *,
    show_stj: bool = True,
    show_tropopause: bool = True,
    stj_levels: tuple[float, ...] = (30.0, 40.0),
    tropopause_colours: tuple[str, str] = ("#a05a00", "0.4"),
) -> None:
    """Locate the STJ with wind isotachs and trace each tropopause."""

    if show_stj:
        wind = circulation_fields.zonal_wind.isel(simulation=run_index)
        jet = _publication_smooth_contours(
            _publication_pressure_field(wind, pressure_levels)
        )
        jet = jet.where(
            (jet.pres >= 100.0)
            & (jet.pres <= 400.0)
            & (jet.absolute_latitude >= 10.0)
            & (jet.absolute_latitude <= 60.0)
        )
        finite = jet.values[np.isfinite(jet.values)]
        visible_levels = tuple(level for level in stj_levels
                               if finite.size and finite.min() < level < finite.max())
        if visible_levels:
            axis.contour(
                jet.absolute_latitude,
                jet.pres,
                jet,
                levels=visible_levels,
                colors="#087e8b",
                linestyles=tuple("--" if level == 30.0 else "-" for level in visible_levels),
                linewidths=tuple(1.15 if level == 30.0 else 1.45 for level in visible_levels),
                zorder=6,
            )
    if not show_tropopause:
        return
    for simulation_index, style in _comparison_styles(
        run_index, reference_index
    ):
        tropopause = circulation_fields.tropopause_pressure.isel(
            simulation=simulation_index
        )
        axis.plot(
            tropopause.absolute_latitude,
            tropopause,
            color=tropopause_colours[0 if simulation_index == run_index else 1],
            linestyle=style["linestyle"],
            linewidth=1.3 if simulation_index == run_index else 0.8,
            alpha=1.0 if simulation_index == run_index else 0.7,
            zorder=7,
        )


def _publication_jet_tropopause_handles(
    *,
    show_stj: bool = True,
    stj_levels: tuple[float, ...] = (30.0, 40.0),
    tropopause_colours: tuple[str, str] = ("#a05a00", "0.4"),
) -> tuple[list[Line2D], list[str]]:
    """Return shared guide labels for the synthesis figures."""

    handles = [
        Line2D([0], [0], color=tropopause_colours[0], linewidth=1.3),
        Line2D([0], [0], color=tropopause_colours[1], linestyle="--", linewidth=0.8),
    ]
    labels = ["Nudged tropopause", "Control tropopause"]
    if show_stj:
        handles = [
            Line2D(
                [0], [0], color="#087e8b",
                linestyle="--" if level == 30.0 else "-",
                linewidth=1.15 if level == 30.0 else 1.45,
            )
            for level in stj_levels
        ] + handles
        labels = [f"STJ: absolute wind {level:g} m/s" for level in stj_levels] + labels
    return handles, labels


def _publication_vector_components(
    first: xr.DataArray,
    second: xr.DataArray,
    pressure_hpa: xr.DataArray,
    *,
    kind: str,
    latitude_span: float = 70.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Apply Jucker (2021) or Hollowed et al. (2025), Appendix C.

    X=phi/delta_phi and Y=-log(p)/log(20). Include the meridional
    pressure gradient in the full coordinate Jacobian. For EP flux, first=M
    and second=N are the instantaneous density-normalized archived fluxes.
    For circulation, they are dPsi/dphi and dPsi/dz.
    Multiply EP components by panel width and height, respectively.
    Divide circulation components by panel height and width, respectively.
    Both components then receive one common linear arrow-length multiplier.
    Jucker section 3 and Table 2 supply the EP coordinate transformation.
    Hollowed equation C12 supplies the rotated display gradient of Psi.
    No altitude-dependent or component-dependent amplitude multiplier applies.
    """

    first, second, pressure_hpa = xr.align(first, second, pressure_hpa, join="exact")
    pressure_hpa = pressure_hpa.transpose(..., "z", "absolute_latitude")
    log_pressure = np.log(pressure_hpa)
    q_z = differentiate_physical_coordinate(log_pressure, "z", pressure_hpa.z.values)
    q_phi = differentiate_physical_coordinate(
        log_pressure, "absolute_latitude",
        np.deg2rad(pressure_hpa.absolute_latitude.values),
    )
    if not bool((np.isfinite(q_z) & (q_z < 0.0)).all()):
        raise ValueError("Vector pressure mapping must decrease monotonically with height")
    delta_phi = np.deg2rad(latitude_span)
    if kind == "circulation":
        return np.log(20.0) * second / q_z, delta_phi * (first - second * q_phi / q_z)
    if kind == "ep":
        cos_squared = np.cos(np.deg2rad(pressure_hpa.absolute_latitude)) ** 2
        factor = 2.0 * np.pi / GRAVITY_M_S2
        # Jucker equations 4, 8, 12: hat(Fphi), hat(Fp), dX/dphi, dY/dp.
        # Fp = p * [dlog(p)/dz * N + dlog(p)/dphi * M/a].
        return (
            (factor * TEM_EARTH_RADIUS_M**2 * cos_squared * first / delta_phi).transpose(*first.dims),
            (-factor * TEM_EARTH_RADIUS_M**3 * cos_squared
             * (q_z * second + q_phi * first / TEM_EARTH_RADIUS_M)
             / np.log(20.0)).transpose(*first.dims),
        )
    raise ValueError(f"Unknown vector kind: {kind}")


def _publication_vector_grid(
    meridional: xr.DataArray,
    vertical: xr.DataArray,
    pressure_hpa: xr.DataArray,
    control_tropopause: xr.DataArray,
    region: str,
    *,
    latitude_limit_degrees: float = 70.0,
    kind: str,
) -> xr.Dataset:
    """Sample vectors uniformly in log pressure, using a common control tropopause."""

    if region not in {"stratosphere", "troposphere", "full"}:
        raise ValueError(f"Unknown vector region: {region}")
    pressures = (
        np.geomspace(60.0, 300.0, 12)
        if region == "stratosphere"
        else np.geomspace(60.0, 925.0, 18)
    )
    latitudes = np.arange(5.0, latitude_limit_degrees, 5.0)
    horizontal, upward = _publication_vector_components(
        meridional, vertical, pressure_hpa,
        kind=kind, latitude_span=latitude_limit_degrees,
    )
    grid = xr.Dataset({
        name: _publication_pressure_field(field, pressures, pressure_hpa).interp(
            absolute_latitude=latitudes
        )
        for name, field in (("horizontal", horizontal), ("upward", upward))
    })
    tropopause = control_tropopause.interp(absolute_latitude=latitudes)
    if region == "stratosphere":
        grid = grid.where(grid.pres < tropopause)
    elif region == "troposphere":
        grid = grid.where(grid.pres >= tropopause)
    grid.attrs["vector_region"] = region
    grid.attrs["vector_kind"] = kind
    grid.attrs["latitude_span_degrees"] = latitude_limit_degrees
    grid.attrs["pressure_span_log"] = np.log(20.0)
    return grid


def _bootstrap_vector_intervals(
    blocks: xr.Dataset, pressure: xr.DataArray, kind: str,
    run: int, control: int, latitude_span: float,
) -> list[xr.Dataset]:
    """Bootstrap transformed components with the existing draws and estimators.

    Transform joint primitive block fields before resampling. This retains their
    covariance, including both terms in the pressure-coordinate vertical flux.
    """

    names = (
        ("residual_streamfunction_phi_gradient", "residual_streamfunction_z_gradient")
        if kind == "circulation" else
        ("ep_flux_meridional_normalized", "ep_flux_vertical_normalized")
    )
    horizontal, upward = _publication_vector_components(
        blocks[names[0]], blocks[names[1]], pressure,
        kind=kind, latitude_span=latitude_span,
    )
    transformed = xr.Dataset({"horizontal": horizontal, "upward": upward})
    return [bootstrap_field_response(transformed, name, run, control)
            for name in ("horizontal", "upward")]


def _display_vector_grid(grid: xr.Dataset, axis: Any) -> xr.Dataset:
    """Apply the final axis ranges and dimensions to the paper transformations."""

    width, height = axis.get_window_extent().size / axis.figure.dpi
    pressure_ratio = latitude_ratio = 1.0
    if "pressure_span_log" in grid.attrs:
        pressure_limits = sorted(axis.get_ylim())
        latitude_limits = sorted(axis.get_xlim())
        pressure_ratio = np.log(pressure_limits[1] / pressure_limits[0]) / grid.attrs["pressure_span_log"]
        latitude_ratio = (latitude_limits[1] - latitude_limits[0]) / grid.attrs["latitude_span_degrees"]
        grid = grid.where(
            (grid.pres >= pressure_limits[0]) & (grid.pres <= pressure_limits[1])
            & (grid.absolute_latitude >= latitude_limits[0])
            & (grid.absolute_latitude <= latitude_limits[1])
        )
    if grid.attrs.get("vector_kind") == "ep":
        return xr.Dataset({"horizontal": grid.horizontal * width / latitude_ratio,
                           "upward": grid.upward * height / pressure_ratio}, attrs=grid.attrs)
    return xr.Dataset({"horizontal": grid.horizontal * pressure_ratio / height,
                       "upward": grid.upward * latitude_ratio / width}, attrs=grid.attrs)


def _draw_vector_requests(fig: plt.Figure, requests: Sequence[dict]) -> None:
    """Calibrate shared linear scales after colourbars and legends fix the layout."""

    fig.canvas.draw()
    # Keep the calibrated panel geometry fixed when savefig changes the DPI.
    fig.set_layout_engine("none")
    references = {}
    for request in requests:
        request["display_grid"] = _display_vector_grid(request["grid"], request["axis"])
        references.setdefault(request["group"], []).append(
            _display_vector_grid(request["reference_grid"], request["axis"])
        )
    references = {name: _publication_vector_reference(grids)
                  for name, grids in references.items()}
    reference_length_inches = 0.075 * min(
        min(request["axis"].get_window_extent().size / fig.dpi)
        for request in requests
    )
    for request in requests:
        _plot_publication_vector_arrows(
            request["axis"], request["display_grid"], references[request["group"]],
            request["intervals"], request["pressure"],
            add_key=False,
            reference_length_inches=reference_length_inches,
            minimum_reference_fraction=0.0,
            fade_below_pressure_hpa=request.get("fade"),
            maximum_pressure_hpa=request.get("maximum_pressure_hpa"),
        )


def _publication_vector_reference(grids: Sequence[xr.Dataset]) -> float:
    """Use one linear vector scale across both experiment responses."""

    speeds = np.concatenate([
        np.hypot(grid.horizontal.values, grid.upward.values).ravel()
        for grid in grids
    ])
    finite = speeds[np.isfinite(speeds)]
    if not finite.size:
        raise ValueError("Publication vectors need a positive finite reference")
    reference = float(np.nanpercentile(finite, 95.0))
    # A valid zero response has no arrows and does not require recalibration.
    return reference if reference > 0.0 else 1.0


def _publication_significance_grid(
    interval: xr.Dataset,
    pressures: np.ndarray,
    latitudes: np.ndarray,
    native_pressure: xr.DataArray | None = None,
) -> xr.DataArray:
    """Resample native pointwise significance without creating new significant cells."""

    valid = np.isfinite(interval.ci_low) & np.isfinite(interval.ci_high)
    significant = ((interval.ci_low > 0.0) | (interval.ci_high < 0.0)).where(valid)
    return _publication_pressure_field(
        significant.astype(float), pressures, native_pressure, nearest=True,
    ).interp(absolute_latitude=latitudes, method="nearest") == 1.0


def _publication_insignificance_grid(
    interval: xr.Dataset,
    pressures: np.ndarray,
    latitudes: np.ndarray,
    native_pressure: xr.DataArray | None = None,
) -> xr.DataArray:
    """Resample finite intervals containing zero without hatching missing estimates."""

    valid = np.isfinite(interval.ci_low) & np.isfinite(interval.ci_high)
    insignificant = ((interval.ci_low <= 0.0) & (interval.ci_high >= 0.0)).where(valid)
    return _publication_pressure_field(
        insignificant.astype(float), pressures, native_pressure, nearest=True,
    ).interp(absolute_latitude=latitudes, method="nearest") == 1.0


def _plot_publication_scalar_hatching(
    axis: Any,
    interval: xr.Dataset,
    native_pressure: xr.DataArray | None = None,
    *,
    latitude_limit: float = 50.0,
    contrast_shading: tuple[xr.DataArray, Any] | None = None,
    background_shading: tuple[xr.DataArray, Any] | None = None,
    hatch: str = "/",
) -> Any:
    """Hatch insignificant responses uniformly in log pressure, retaining native classifications."""

    dot_pressure = np.geomspace(55.0, 975.0, 193)
    dot_latitude = np.arange(0.625, latitude_limit, 1.25)
    insignificant = _publication_insignificance_grid(
        interval, dot_pressure, dot_latitude, native_pressure,
    ).values
    latitude_edges = np.r_[dot_latitude - 0.625, dot_latitude[-1] + 0.625]
    log_pressure = np.log(dot_pressure)
    pressure_edges = np.exp(np.r_[
        log_pressure[0] - np.diff(log_pressure)[0] / 2.0,
        (log_pressure[:-1] + log_pressure[1:]) / 2.0,
        log_pressure[-1] + np.diff(log_pressure)[-1] / 2.0,
    ])
    hatch_layers = [(insignificant, "0.35", 0.4, 0.25)]
    if contrast_shading is not None:
        shaded_field, image = contrast_shading
        shading_values = _publication_pressure_field(
            shaded_field, dot_pressure,
        ).interp(absolute_latitude=dot_latitude).values
        rgba = image.to_rgba(shading_values)
        background_rgb = np.ones_like(rgba[..., :3])
        if background_shading is not None:
            background_field, background_image = background_shading
            background_values = _publication_pressure_field(
                background_field, dot_pressure,
            ).interp(absolute_latitude=dot_latitude).values
            background_rgba = background_image.to_rgba(background_values)
            background_rgb = (
                background_rgba[..., :3] * background_rgba[..., 3:4]
                + (1.0 - background_rgba[..., 3:4])
            )
        alpha = rgba[..., 3:4]
        if image.get_alpha() is not None:
            alpha = alpha * image.get_alpha()
        rgb = rgba[..., :3] * alpha + background_rgb * (1.0 - alpha)
        luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
        dark_background = luminance < 0.6
        # Change only hatch contrast; retain exactly the same insignificance mask.
        hatch_layers = [
            (insignificant & ~dark_background, "0.25", 0.65, 0.30),
            (insignificant & dark_background, "white", 0.85, 0.30),
        ]
    artists = []
    for mask, colour, opacity, line_width in hatch_layers:
        artists.append(axis.pcolor(
            latitude_edges, pressure_edges,
            np.ma.masked_where(~mask, np.ones(insignificant.shape)),
            shading="flat", facecolor="none", edgecolor=colour,
            hatch=hatch * 2, hatch_linewidth=line_width, linewidth=0.0,
            alpha=opacity, zorder=2.5,
        ))
    return artists[0]


def _plot_publication_vector_arrows(
    axis: Any,
    grid: xr.Dataset,
    reference: float,
    intervals: Sequence[xr.Dataset],
    pressure_hpa: xr.DataArray,
    *,
    add_key: bool = True,
    minimum_reference_fraction: float = 0.10,
    fade_below_pressure_hpa: float | None = None,
    maximum_pressure_hpa: float | None = None,
    reference_length_inches: float | None = None,
    key_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Render transformed vectors with one multiplier and calibrated example arrows.

    Circulation lengths represent streamfunction gradients, not velocities.
    EP lengths represent the conservative, normalized flux of the archived
    EPFd operator. There is no independent component magnification or length clipping.
    """

    if maximum_pressure_hpa is not None:
        grid = grid.where(grid.pres <= maximum_pressure_hpa, drop=True)
    x, p = np.meshgrid(grid.absolute_latitude.values, grid.pres.values)
    significant = np.zeros(x.shape, dtype=bool)
    for interval in intervals:
        significant |= _publication_significance_grid(
            interval, grid.pres.values, grid.absolute_latitude.values, pressure_hpa,
        ).values
    colours = np.where(significant.ravel(), "0.08", "0.55") if intervals else "0.08"
    width, height = axis.get_window_extent().size / axis.figure.dpi
    if reference_length_inches is None:
        reference_length_inches = 0.075 * min(width, height)
    displacement_per_unit = reference_length_inches / reference
    visible = np.hypot(grid.horizontal, grid.upward) >= minimum_reference_fraction * reference
    horizontal = grid.horizontal.where(visible).values
    upward = grid.upward.where(visible).values
    opacity = (
        np.where(p.ravel() > fade_below_pressure_hpa, 0.08, 1.0)
        if fade_below_pressure_hpa is not None else 1.0
    )
    latitude_span = np.diff(axis.get_xlim())[0]
    pressure_span = np.log(max(axis.get_ylim()) / min(axis.get_ylim()))
    arrows = axis.quiver(
        x, p,
        horizontal * displacement_per_unit * latitude_span / width,
        p * np.expm1(-upward * displacement_per_unit * pressure_span / height),
        angles="xy", scale_units="xy", scale=1.0,
        pivot="mid", color=colours, edgecolor="white", linewidth=0.35,
        alpha=opacity, width=0.0045, headwidth=3.8, headlength=4.5,
        minlength=0, zorder=5,
    )
    if maximum_pressure_hpa is not None:
        top_pressure = min(axis.get_ylim())
        bottom_pressure = min(maximum_pressure_hpa, max(axis.get_ylim()))
        arrows.set_clip_path(Rectangle(
            (0.0, top_pressure), 1.0, bottom_pressure - top_pressure,
            transform=axis.get_yaxis_transform(),
        ))
    if add_key:
        example = _quiver_example_value(0.5 * reference)
        exponent = int(np.floor(np.log10(example)))
        labelled_example = example / 10.0**exponent
        kind = grid.attrs["vector_kind"]
        symbols = (r"C_x", r"C_y") if kind == "circulation" else (r"\mathcal{F}_x", r"\mathcal{F}_y")
        units = r"kg s$^{-1}$ in$^{-1}$" if kind == "circulation" else r"m$^3$ in"
        _add_component_quiver_keys(
            axis, example * displacement_per_unit / width,
            example * displacement_per_unit / height,
            _quiver_example_label(symbols[0], labelled_example, ""),
            _quiver_example_label(symbols[1], labelled_example, ""), width=0.0045,
            common_units=rf"$10^{{{exponent}}}$ {units}",
            offset=key_offset,
        )


def _publication_synthesis_colourbars(
    fig: plt.Figure, axes: np.ndarray, images: dict,
) -> None:
    """Place four colourbars above the boxed footer legend."""

    labels = [
        r"$\Delta T$ (K)",
        r"$\Delta\mathrm{EPFd}$" + "\n" + r"(m s$^{-1}$ day$^{-1}$)",
        r"$\Delta\Psi$ ($10^9$ kg s$^{-1}$)",
        r"$\Delta$ equatorward + upward moisture flux" + "\n" + r"(g kg$^{-1}$ m s$^{-1}$)",
    ]
    shading_images = [images[column] for column in range(4)]
    left, right, gap = 0.055, 0.985, 0.025
    width = (right - left - (len(labels) - 1) * gap) / len(labels)
    for index, (image, label) in enumerate(zip(shading_images, labels)):
        cax = fig.add_axes((left + index * (width + gap), 0.18, width, 0.014))
        cax.set_in_layout(False)
        colourbar = add_symmetric_colorbar(
            fig, image, axes.ravel(), image.levels, label, cax=cax,
            location="bottom", nticks=4, spacing="uniform", ticks_at_boundaries=True,
        )
        colourbar.ax.tick_params(labelsize=6, pad=1.5)
        colourbar.ax.xaxis.label.set_fontsize(6.5)
        colourbar.ax.xaxis.labelpad = 2.0


def _publication_synthesis_legends(
    fig: plt.Figure, guide_handles: Sequence[Any], guide_labels: Sequence[str],
    show_bootstrap: bool,
) -> None:
    """Place physical guides and uncertainty symbols in one boxed footer legend."""

    guides = dict(zip(guide_labels, guide_handles))
    handles = list(guides.values())
    labels = [label.replace("STJ: absolute wind", "STJ:") for label in guides]
    if show_bootstrap:
        handles.extend([
            Line2D([0], [0], color="0.08", marker=">", linewidth=0.8, markersize=4),
            Line2D([0], [0], color="0.55", marker=">", linewidth=0.8, markersize=4),
            Patch(facecolor="none", edgecolor="0.35", hatch="//",
                  hatch_linewidth=0.25, linewidth=0.0, alpha=0.4),
        ])
        labels.extend(["Vector: ≥1 significant", "Vector: neither significant",
                       "Shading: insignificant (95% bootstrap)"])
    legend = fig.legend(
        handles, labels, loc="lower left", bbox_to_anchor=(0.04, 0.008, 0.945, 0.0),
        ncol=min(5, int(np.ceil(len(handles) / 2))), mode="expand", fontsize=6.2, frameon=True,
        facecolor="white", edgecolor="0.35", framealpha=1.0,
        handlelength=1.6, handletextpad=0.4, columnspacing=0.8,
        labelspacing=0.5, borderaxespad=0.0,
    )
    legend.set_in_layout(False)


def plot_publication_thermal_residual_circulation(
    circulation_fields: xr.Dataset,
    circulation_metrics: xr.Dataset,
    circulation_blocks: xr.Dataset,
    tem_fields: xr.Dataset,
    tem_blocks: xr.Dataset,
    moisture_fields: xr.Dataset,
    moisture_blocks: xr.Dataset,
    show_bootstrap: bool,
    nudging_regions: dict[str, tuple[NudgingRegion, ...]] | None = None,
) -> plt.Figure:
    """Compare thermal, stratospheric, wave, Hadley, and moisture responses."""

    started = perf_counter()
    progress("Figure 10: preparing temperature, EPFd, streamfunction, and moisture responses")
    specs = COMPARISONS[1:3]
    latitude_limit = 50.0
    box_style = mechanism.REGION_BOX_STYLE
    pressure_grid = _tem_reference_pressure_grid(tem_fields)
    if nudging_regions is None:
        nudging_regions = load_nudging_regions(ROOT_DIR, tem_fields.simulation.values)
    nudging_handles = {}
    pressures = _publication_pressure_levels()
    tropopause = circulation_fields.tropopause_pressure.isel(simulation=0)
    tropopause_colours = ("#16843b", "#16843b")
    sections = folded_moisture_sections(complete_moisture_pressure_column(moisture_fields))
    responses = {
        "temperature": _publication_comparison_fields(circulation_fields, "temperature")[1:],
        "a_ep": [field * SECONDS_PER_DAY for field in
                 _publication_comparison_fields(tem_fields, "a_ep")[1:]],
        "streamfunction": [field / 1.0e9 for field in
                           _publication_comparison_fields(circulation_fields, "streamfunction")[1:]],
        "moisture": [field * 1000.0 for field in
                     _publication_comparison_fields(sections, "mean_equatorward_upward_moisture_flux")[1:]],
    }
    scalar_sources = {
        "temperature": (circulation_blocks, "temperature"),
        "a_ep": (tem_blocks, "a_ep"),
        "streamfunction": (circulation_blocks, "streamfunction"),
        "moisture": (moisture_blocks, "section_mean_equatorward_upward_moisture_flux"),
    }
    scalar_intervals = {}
    if show_bootstrap:
        scalar_labels = dict(temperature="temperature", a_ep="EPFd",
                             streamfunction="streamfunction", moisture="combined moisture flux")
        for name, (blocks, variable) in scalar_sources.items():
            scalar_intervals[name] = []
            for _name, run, control, label in specs:
                display_label = label.replace("T from U", "T")
                step_started = perf_counter()
                progress(f"Figure 10: {display_label}, {scalar_labels[name]} bootstrap interval")
                scalar_intervals[name].append(
                    bootstrap_field_response(blocks, variable, run, control)
                )
                progress(
                    f"Figure 10: {display_label}, {scalar_labels[name]} interval complete "
                    f"({perf_counter() - step_started:.1f} s)"
                )
    progress("Figure 10: preparing colour scales and conservative vector grids")
    levels = {
        name: symmetric_percentile_levels_for_fields(
            [field.where((field.pres >= 50) & (field.pres <= 1000), drop=True).sel(
                absolute_latitude=slice(0.0, latitude_limit)
             ) for field in fields],
            percentile=98.0 if name != "moisture" else 95.0,
            ndiv=17, zero_field_scale=0.01,
        )
        for name, fields in responses.items()
        if name != "a_ep"
    }
    levels = {name: _publication_shading_levels(boundaries)
              for name, boundaries in levels.items()}
    levels["a_ep"] = _ep_acceleration_response_levels(tem_fields)
    vector_fields = {
        name: _publication_comparison_fields(tem_fields, name)[1:]
        for name in ("residual_streamfunction_phi_gradient", "residual_streamfunction_z_gradient",
                     "ep_flux_meridional_normalized", "ep_flux_vertical_normalized")
    }
    vector_grids = {"ep": [
        _publication_vector_grid(
            fy, fz, pressure_grid, tropopause, "full",
            latitude_limit_degrees=latitude_limit,
            kind="ep",
        )
        for fy, fz in zip(vector_fields["ep_flux_meridional_normalized"],
                          vector_fields["ep_flux_vertical_normalized"])
    ]}
    # Retain the full-depth scale for the Hadley and moisture panels.
    residual_grids = [
        _publication_vector_grid(
            v, w, pressure_grid, tropopause, "full",
            latitude_limit_degrees=latitude_limit,
            kind="circulation",
        )
        for v, w in zip(vector_fields["residual_streamfunction_phi_gradient"],
                        vector_fields["residual_streamfunction_z_gradient"])
    ]
    # Calibrate column 1 from its visible upper-level vectors, shared across rows.
    upper_residual_grids = [
        grid.where(grid.pres <= 100.0, drop=True) for grid in residual_grids
    ]
    winds = _publication_comparison_fields(circulation_fields, "zonal_wind")
    wind_levels = symmetric_percentile_levels_for_fields(
        [wind.sel(absolute_latitude=slice(0.0, latitude_limit)) for wind in winds[1:]],
        percentile=99.0, ndiv=17, zero_field_scale=1.0,
    )
    fig, axes = plt.subplots(
        2, 4, figsize=(10.0, 5.2), sharex=True, sharey="col",
        constrained_layout=True, squeeze=False,
    )
    titles = (
        "Temperature; zonal wind\nResidual circulation",
        "EPFd; EP flux\nZonal wind", "Hadley streamfunction\nResidual circulation",
        "Equatorward + upward moisture flux\nResidual circulation",
    )
    shading = {
        0: "temperature", 1: "a_ep", 2: "streamfunction", 3: "moisture",
    }
    images = {}
    vector_requests = []
    for row, (_name, run, control, label) in enumerate(specs):
        display_label = label.replace("T from U", "T")
        intervals = {}
        if show_bootstrap:
            for kind in ("ep", "circulation"):
                step_started = perf_counter()
                progress(f"Figure 10: {display_label}, {kind} vector bootstrap intervals")
                intervals[kind] = _bootstrap_vector_intervals(
                    tem_blocks, pressure_grid, kind, run, control, latitude_limit,
                )
                progress(
                    f"Figure 10: {display_label}, {kind} vector intervals complete "
                    f"({perf_counter() - step_started:.1f} s)"
                )
        wind_plot = _publication_smooth_contours(
            _publication_pressure_field(winds[row + 1], pressures)
        )
        for column, axis in enumerate(axes[row]):
            panel_letter = chr(97 + row * axes.shape[1] + column)
            panel_title = titles[column].replace("\n", "; ")
            step_started = perf_counter()
            progress(f"Figure 10: drawing panel {panel_letter}, {display_label}, {panel_title}")
            _set_latpres_axis(axis, ylabel=column == 0)
            axis.set_xlim(0.0, latitude_limit)
            if column in shading:
                name = shading[column]
                native_pressure = pressure_grid if name == "a_ep" else None
                field = _publication_pressure_field(responses[name][row], pressures, native_pressure)
                colourmap = difference_cmap(
                    levels[name], cmap="BrBG" if name == "moisture" else "RdBu_r",
                    white_zero_boundary=True,
                )
                images[column] = axis.contourf(
                    field.absolute_latitude, field.pres, field,
                    levels=levels[name], cmap=colourmap,
                    norm=BoundaryNorm(levels[name], colourmap.N), extend="both",
                    alpha=1.0, zorder=1.0,
                )
                if show_bootstrap:
                    _plot_publication_scalar_hatching(
                        axis, scalar_intervals[name][row], native_pressure,
                        latitude_limit=latitude_limit,
                        contrast_shading=(field, images[column]) if name == "moisture" else None,
                    )
            if column in (0, 1, 2):
                contours = _plot_signed_contours(
                    axis, wind_plot.absolute_latitude, wind_plot.pres, wind_plot,
                    wind_levels, count=wind_levels.size,
                )
                axis.clabel(
                    contours, contours.levels[::2], fontsize=6, inline=True, fmt="%g",
                )
            kind = "ep" if column == 1 else "circulation"
            vector_grid = (
                upper_residual_grids[row] if column == 0 else
                vector_grids["ep"][row] if column == 1 else residual_grids[row]
            )
            vector_requests.append(dict(
                axis=axis, group="circulation_upper" if column == 0 else kind,
                pressure=pressure_grid, grid=vector_grid, reference_grid=vector_grid,
                intervals=intervals[kind] if show_bootstrap else [],
                maximum_pressure_hpa=100.0 if column == 0 else None,
            ))
            _plot_publication_jet_and_tropopause(
                axis, circulation_fields, run, control, pressures,
                stj_levels=(40.0,), tropopause_colours=tropopause_colours,
            )
            _plot_tem_location_guides(
                axis, circulation_metrics, run, None,
                legend=row == 0 and column == 0, show_stj=False,
            )
            if column == 1:
                axis.add_patch(Rectangle(
                    (mechanism.REGION_LATITUDE[0], mechanism.REGION_PRESSURE[0]),
                    mechanism.REGION_LATITUDE[1] - mechanism.REGION_LATITUDE[0],
                    mechanism.REGION_PRESSURE[1] - mechanism.REGION_PRESSURE[0],
                    fill=False, zorder=9, **box_style,
                ))
            for handle in _plot_nudging_region_guides(
                axis, pressure_grid,
                [str(tem_fields.simulation.values[run]), str(tem_fields.simulation.values[control])],
                nudging_regions,
            ):
                nudging_handles[handle.get_label()] = handle
            axis.set_xticks((0, 10, 20, 30, 40, 50))
            axis.set_yticks((1000, 700, 500, 300, 200, 100, 50))
            axis.set_yticklabels(("1000", "700", "500", "300", "200", "100", "50"))
            axis.tick_params(axis="both", which="major", labelsize=7)
            axis.set_xlabel("Latitude (°)" if row == 1 else "")
            axis.xaxis.label.set_fontsize(7)
            axis.yaxis.label.set_fontsize(7)
            annotate_panel_label(axis, f"({chr(97 + row * axes.shape[1] + column)}) {display_label}")
            axis.texts[-1].set_fontsize(7)
            axis.texts[-1].set_zorder(10)
            axis.texts[-1].set_position((0.97, 0.95))
            axis.texts[-1].set_ha("right")
            axis.texts[-1].get_bbox_patch().set_alpha(0.95)
            if row == 0:
                axis.set_title(titles[column], fontsize=8, pad=6)
            progress(f"Figure 10: panel {panel_letter} complete ({perf_counter() - step_started:.1f} s)")
    progress("Figure 10: arranging colourbars and boxed legend")
    _publication_synthesis_colourbars(fig, axes, images)
    handles, guide_labels = axes[0, 0].get_legend_handles_labels()
    jet_handles, jet_labels = _publication_jet_tropopause_handles(
        stj_levels=(40.0,), tropopause_colours=tropopause_colours,
    )
    handles.extend(jet_handles)
    guide_labels.extend(jet_labels)
    handles.extend([
        Line2D([0], [0], color="0.12", linewidth=0.8),
    ])
    guide_labels.append("Wind anomaly (negative dashed)")
    handles.append(Patch(fill=False, **box_style))
    guide_labels.append("Mechanism box")
    handles.extend(nudging_handles.values())
    guide_labels.extend(nudging_handles)
    _publication_synthesis_legends(
        fig, handles, guide_labels, show_bootstrap,
    )
    fig.get_layout_engine().set(rect=(0.0, 0.24, 1.0, 0.76))
    progress("Figure 10: calibrating and drawing vectors at the final panel dimensions")
    _draw_vector_requests(fig, vector_requests)
    progress(f"Figure 10: plotting complete ({perf_counter() - started:.1f} s); ready for export")
    return fig


def _plot_publication_hadley_edges(
    axis: Any,
    circulation_metrics: xr.Dataset,
) -> None:
    """Mark the response simulations with the shared HC-edge convention."""

    for index, (_name, run, _reference, label) in enumerate(COMPARISONS[1:3]):
        latitude = _metric_value(circulation_metrics, run, "hadley_edge_latitude")
        axis.axvline(latitude, **HC_EDGE_LINE_STYLE)
        axis.annotate(
            label.replace("T from U", "T").split(" - ")[0],
            xy=(latitude, 0.985), xycoords=axis.get_xaxis_transform(),
            xytext=(-3 if index == 0 else 3, -2), textcoords="offset points",
            ha="right" if index == 0 else "left", va="top", fontsize=6,
            color=HC_EDGE_LINE_STYLE["color"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5),
        )


def plot_publication_moisture_transport(
    moisture_fields: xr.Dataset,
    circulation_metrics: xr.Dataset,
    moisture_blocks: xr.Dataset,
    show_bootstrap: bool,
) -> plt.Figure:
    """Compare total transport decomposition with its mean-circulation breakdown."""

    diagnostic_fields = complete_moisture_pressure_column(moisture_fields)
    transports, _convergences = folded_moisture_profiles(diagnostic_fields)
    comparison_specs = COMPARISONS[1:3]
    transport_responses = {
        name: _publication_comparison_fields(transports, name)[1:]
        for name in MOISTURE_FLUX_COMPONENTS
    }
    component_styles = {
        name: dict(
            linestyle=linestyle,
            label=_MOISTURE_COMPONENT_STYLES[name]["label"],
        )
        for name, linestyle in (
            ("total_meridional_moisture_flux", "-"),
            ("mean_meridional_moisture_flux", "--"),
            ("eddy_meridional_moisture_flux", ":"),
        )
    }
    attribution_styles = {
        "mean_circulation_change": component_styles["mean_meridional_moisture_flux"],
        "wind_contribution": dict(linestyle="-", label=r"Wind: $\Delta\bar v\,\bar q_{control}$"),
        "humidity_contribution": dict(linestyle="-.",
                                      label=r"Humidity: $\bar v_{control}\,\Delta\bar q$"),
        "interaction": dict(linestyle=":",
                            label=r"Interaction: $\Delta\bar v\,\Delta\bar q$"),
    }
    fig, axes = plt.subplots(
        1, 2, figsize=(10.0, 3.8), sharex=True, sharey=True,
        constrained_layout=True, squeeze=False,
    )
    transport_axis, attribution_axis = axes[0]
    panel_handles = [[], []]
    all_profiles = []
    interval_opacity = 0.10
    for experiment, (_name, run, reference, label) in enumerate(comparison_specs):
        response_label = label.replace("T from U", "T").replace(" - ", " − ")
        colour = simulation_colour(SIMULATIONS[run][0])
        attribution = mean_circulation_response_components(diagnostic_fields, run, reference)
        attribution_intervals = (
            bootstrap_moisture_attribution_intervals(moisture_blocks, run, reference)
            if show_bootstrap else {}
        )
        for panel, (axis, styles) in enumerate(((transport_axis, component_styles),
                                               (attribution_axis, attribution_styles))):
            for name, style in styles.items():
                profile = transport_responses[name][experiment] if panel == 0 else attribution[name]
                all_profiles.append(profile.sel(absolute_latitude=slice(0.0, 50.0)))
                line, = axis.plot(
                    profile.absolute_latitude, profile, color=colour,
                    linestyle=style["linestyle"], linewidth=1.6, zorder=3,
                    label=f"{response_label}: {style['label']}",
                )
                panel_handles[panel].append(line)
                if show_bootstrap:
                    interval = (
                        bootstrap_field_response(moisture_blocks, f"transport_{name}", run, reference)
                        if panel == 0 else attribution_intervals[name]
                    )
                    _plot_profile_bootstrap_interval(axis, interval, colour, alpha=interval_opacity)
    limit = 1.08 * _profile_response_limit(all_profiles)
    for panel, axis in enumerate(axes[0]):
        axis.axhline(0.0, color="0.45", linewidth=0.6, zorder=0)
        axis.set_xlim(0.0, 50.0)
        axis.set_xticks((0, 10, 20, 30, 40, 50))
        axis.set_ylim(-limit, limit)
        axis.set_xlabel("Latitude (°)", fontsize=7)
        axis.tick_params(axis="both", labelsize=7)
        _plot_publication_hadley_edges(axis, circulation_metrics)
        annotate_panel_label(axis, ("(a) Total transport decomposition", "(b) Mean-circulation breakdown")[panel])
        axis.texts[-1].set_fontsize(7)
        axis.texts[-1].set_zorder(10)
        guides = [Line2D([0], [0], label="HC edge", **HC_EDGE_LINE_STYLE)]
        if show_bootstrap:
            guides.append(Patch(facecolor="0.15", alpha=interval_opacity, edgecolor="none", label="95% bootstrap interval"))
        # Group U and T entries by column, with one shared guide below each group.
        halfway = len(panel_handles[panel]) // 2
        handles = (panel_handles[panel][:halfway] + guides[:1]
                   + panel_handles[panel][halfway:] + guides[1:])
        legend = fig.legend(
            handles=handles, loc="lower left", ncol=2, mode="expand",
            bbox_to_anchor=(0.055 if panel == 0 else 0.54, 0.008, 0.43 if panel == 0 else 0.445, 0.0),
            fontsize=6.2, frameon=True, facecolor="white", edgecolor="0.35", framealpha=1.0,
            handlelength=2.5, handletextpad=0.4, columnspacing=0.8,
            labelspacing=0.5, borderaxespad=0.0,
        )
        legend.set_in_layout(False)
    transport_axis.set_ylabel(
        "Equatorward moisture-transport response\n" + r"(kg m$^{-1}$ s$^{-1}$)", fontsize=7,
    )
    fig.get_layout_engine().set(rect=(0.0, 0.30, 1.0, 0.70))
    return fig


def plot_mechanism_check(means, block_values, intervals) -> plt.Figure:
    """Show TEM balance, wave/HC checks, EP torque, and U-response MMC decompositions."""

    fig = plt.figure(figsize=(10.0, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.85))
    budget_axis = fig.add_subplot(grid[0, 0])
    flow_axis = fig.add_subplot(grid[0, 1])
    association_grid = grid[1, 0].subgridspec(1, 2, wspace=0.08)
    ep_hc_axis = fig.add_subplot(association_grid[0, 0])
    precip_hc_axis = fig.add_subplot(association_grid[0, 1], sharey=ep_hc_axis)
    boundary_axis = fig.add_subplot(grid[1, 1])
    av_axis = fig.add_subplot(grid[2, 0])
    aw_axis = fig.add_subplot(grid[2, 1], sharey=av_axis)
    styles = (("#d95f02", "o", "U − control"), ("#2166ac", "^", "T − control"))

    def interval_lines(axis, x, value, name, ci, color, factor=1.0):
        if name not in ci:
            return
        low, high = np.asarray(ci[name]) * factor
        axis.vlines(x, low, high, color=color, linewidth=0.9, zorder=5)
        axis.hlines((low, high), x - 0.04, x + 0.04, color=color, linewidth=0.9, zorder=5)

    def scatter(axis, index, xname, yname, xfactor, yfactor):
        color, marker, _label = styles[index]
        records, mean, ci = block_values[index], means[index], intervals[index]
        axis.scatter([item[xname] * xfactor for item in records],
                     [item[yname] * yfactor for item in records],
                     color=color, marker=marker, s=13, alpha=0.38, linewidths=0, zorder=2)
        x, y = mean[xname] * xfactor, mean[yname] * yfactor
        if xname in ci:
            axis.plot(np.asarray(ci[xname]) * xfactor, [y, y], color=color, linewidth=1.1, zorder=4)
            axis.plot(np.asarray(ci[xname]) * xfactor, [y, y], "|", color=color, markersize=5, zorder=4)
        if yname in ci:
            axis.plot([x, x], np.asarray(ci[yname]) * yfactor, color=color, linewidth=1.1, zorder=4)
            axis.plot([x, x], np.asarray(ci[yname]) * yfactor, "_", color=color, markersize=5, zorder=4)
        axis.scatter([x], [y], color=color, marker=marker, s=70,
                     edgecolors="white", linewidths=0.8, zorder=6)

    budget_names = ["a_ep", "a_vres", "a_wres", "unresolved_direct_momentum_residual", "a_tendency"]
    budget_labels = ["EPFd", r"$A_{v^*}$", r"$A_{w^*}$", "Unresolved/\ndirect residual", "Tendency"]
    if any(abs(mean["a_density_storage"]) > 1e-15 for mean in means):
        budget_names.append("a_density_storage")
        budget_labels.append("Density\nstorage")
    for index, (color, marker, _label) in enumerate(styles):
        offset = (-0.18, 0.18)[index]
        x = np.arange(len(budget_names)) + offset
        values = [means[index][name] * SECONDS_PER_DAY for name in budget_names]
        budget_axis.bar(x, values, width=0.32, color=color, alpha=0.85,
                        hatch="" if index == 0 else "//", edgecolor=color, linewidth=0.6)
        for point, value, name in zip(x, values, budget_names):
            interval_lines(budget_axis, point, value, name, intervals[index], color, SECONDS_PER_DAY)
        scatter(flow_axis, index, "predicted_vres_adjusted", "observed_vres", 1.0, 1.0)
        scatter(ep_hc_axis, index, "predicted_vres", "hc_strength", 1.0, 1e-9)
        scatter(precip_hc_axis, index, "precipitation", "hc_strength", 1.0, 1e-9)
        boundary_names = [f"boundary_{name}" for name in (*mechanism.BOUNDARIES, "net")]
        positions = np.arange(len(boundary_names)) + offset
        widths = [0.32] * 4 + [0.16]
        values = [means[index][name] * SECONDS_PER_DAY for name in boundary_names]
        boundary_axis.bar(positions, values, width=widths, color=color, alpha=0.85,
                          hatch="" if index == 0 else "//", edgecolor=color, linewidth=0.6)
        for point, value, name in zip(positions, values, boundary_names):
            interval_lines(boundary_axis, point, value, name, intervals[index], color, SECONDS_PER_DAY)
        boundary_axis.plot(positions[-1], means[index]["torque_a_ep"] * SECONDS_PER_DAY,
                           marker="D", markersize=4, markerfacecolor="white", markeredgecolor=color, zorder=6)
    budget_axis.set_xticks(np.arange(len(budget_names)), budget_labels, fontsize=7)
    budget_axis.set_ylabel(r"Acceleration response (m s$^{-1}$ day$^{-1}$)", fontsize=8)
    # Decompose matched mean states; expose covariance/remapping instead of forcing three-term closure.
    decomposition_labels = {
        "av": [r"$\eta_C\Delta v^*$" + "\nFlow", r"$v_C^*\Delta\eta$" + "\nVorticity",
               r"$\Delta v^*\Delta\eta$" + "\nInteraction"],
        "aw": [r"$-\Delta w^*(u_z)_C$" + "\nFlow", r"$-w_C^*\Delta u_z$" + "\nShear",
               r"$-\Delta w^*\Delta u_z$" + "\nInteraction"],
    }
    for axis, prefix, total in ((av_axis, "av", "a_vres"), (aw_axis, "aw", "a_wres")):
        names = (*mechanism.MMC_DECOMPOSITIONS[prefix], f"{prefix}_mean_product_remainder", total)
        labels = [*decomposition_labels[prefix], "Covariance/\nremap remainder",
                  r"$\Delta A_{v^*}$" + "\nTotal" if prefix == "av" else r"$\Delta A_{w^*}$" + "\nTotal"]
        for position, name in enumerate(names):
            value = means[0][name] * SECONDS_PER_DAY
            color = "0.45" if position == 3 else styles[0][0]
            axis.bar(position, value, width=0.65, color="white" if position == 4 else color,
                     alpha=1.0 if position == 4 else 0.85, edgecolor=color, linewidth=0.9,
                     hatch="//" if position == 3 else None)
            interval_lines(axis, position, value, name, intervals[0], "0.15", SECONDS_PER_DAY)
        axis.set_xticks(np.arange(len(names)), labels, fontsize=7)
        axis.margins(x=0.07)
    av_axis.set_ylabel(r"Acceleration response (m s$^{-1}$ day$^{-1}$)", fontsize=8)
    limits = [0.0]
    for mean, records, ci in zip(means, block_values, intervals):
        for name in ("predicted_vres_adjusted", "observed_vres"):
            limits.extend([mean[name], *[record[name] for record in records]])
            if name in ci:
                limits.extend(ci[name])
    limits = np.asarray(limits)
    limits = limits[np.isfinite(limits)]
    lower, upper = limits.min(), limits.max()
    margin = 0.04 * max(upper - lower, 0.01)
    flow_axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin],
                   color="0.35", linestyle="--", linewidth=0.8, zorder=1)
    flow_axis.set_xlim(lower - margin, upper + margin)
    flow_axis.set_ylim(lower - margin, upper + margin)
    flow_axis.set_xlabel("Vorticity-adjusted EP prediction\n" +
                        r"$\langle(-\Delta\mathrm{EPFd}-v_C^*\Delta\eta)/\eta_E\rangle_R$ (m s$^{-1}$)", fontsize=8)
    flow_axis.set_ylabel(r"Observed $\langle\Delta v^*\rangle_R$ (m s$^{-1}$)", fontsize=8)
    ep_hc_axis.set_xlabel(r"$V_{\rm EP}=\langle-\Delta\mathrm{EPFd}/\eta_C\rangle_R$" + "\n" + r"(m s$^{-1}$)", fontsize=8)
    precip_hc_axis.set_xlabel(r"$\Delta P_{10^\circ S-10^\circ N}$" + "\n" + r"(mm day$^{-1}$)", fontsize=8)
    ep_hc_axis.set_ylabel(r"HC-strength response ($10^9$ kg s$^{-1}$)", fontsize=8)
    precip_hc_axis.tick_params(labelleft=False)
    boundary_axis.set_xticks(np.arange(5), [f"Lower\n{mechanism.REGION_PRESSURE[1]:g} hPa",
                            f"Upper\n{mechanism.REGION_PRESSURE[0]:g} hPa",
                            f"Equatorward\n{mechanism.REGION_LATITUDE[0]:g}°",
                            f"Poleward\n{mechanism.REGION_LATITUDE[1]:g}°", "Net"], fontsize=7)
    boundary_axis.set_ylabel("Outward EP torque / control mass moment\n" + r"(m s$^{-1}$ day$^{-1}$)", fontsize=8)
    for axis, title in ((budget_axis, "(a) Fixed-box TEM momentum balance"),
                        (flow_axis, "(b) Wave forcing predicts residual flow?"),
                        (ep_hc_axis, "(c) HC versus EP-predicted flow"),
                        (precip_hc_axis, "HC versus precipitation"),
                        (boundary_axis, "(d) EP-flux boundary budget"),
                        (av_axis, r"(e) U − control: $A_{v^*}$ decomposition"),
                        (aw_axis, r"(f) U − control: $A_{w^*}$ decomposition")):
        axis.axhline(0.0, color="0.55", linewidth=0.6, zorder=0)
        axis.tick_params(labelsize=7)
        axis.set_title(title, fontsize=8, loc="left", pad=8)
    handles = [Line2D([], [], marker=marker, color=color, linestyle="none", markersize=5, label=label)
               for color, marker, label in styles]
    handles.extend([
        Line2D([], [], marker="o", color="0.35", linestyle="none", alpha=0.4, markersize=3, label="10-day paired blocks"),
        Line2D([], [], marker="o", color="0.35", linestyle="none", markersize=7, label="Full-run mean"),
        Line2D([], [], color="0.35", linewidth=1, marker="|", label="95% paired-block bootstrap interval"),
        Line2D([], [], marker="D", markerfacecolor="white", color="0.35", linestyle="none", markersize=4, label="Volume-integrated EPFd (d)"),
        Line2D([], [], color="0.35", linestyle="--", linewidth=0.8, label="1:1 (b)"),
    ])
    if not any(intervals):
        handles = [handle for handle in handles if "bootstrap" not in handle.get_label()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=3,
               fontsize=7, frameon=True, facecolor="white", framealpha=1.0, edgecolor="0.35")
    fig.get_layout_engine().set(rect=(0.0, 0.10, 1.0, 0.90))
    return fig


def save_mechanism_check(fields, blocks, saver, *, test: bool) -> None:
    """Compute compact comparisons, report checks, and export Figure 14."""

    progress("Figure 14: fixed-box comparisons and matched-block bootstrap")
    specs = COMPARISONS[1:3]
    means, block_values, intervals = mechanism.mechanism_analysis(
        fields, blocks, specs, resamples=N_BOOTSTRAP, minimum_blocks=MIN_BLOCKS,
        seed=BOOTSTRAP_SEED, percentiles=CI_PERCENTILES, bootstrap=not test,
    )
    summary, block_table, associations = mechanism.mechanism_tables(means, block_values, intervals, specs)
    block_table["block_start"] = [str(value) for _ in specs for value in blocks.block_start.isel(simulation=0).values]
    block_table["block_end"] = [str(value) for _ in specs for value in blocks.block_end.isel(simulation=0).values]
    for name, table in (("summary", summary), ("blocks", block_table), ("associations", associations)):
        write_csv_atomic(table, Path(saver.save_loc) / f"mechanism_check_{name}.csv", index=False)
    caption = [
        "Quantitative mechanism checks for U minus control and T minus control.",
        f"R = {mechanism.REGION_DESCRIPTION}, identical for both experiments.",
        "(a) Eastward-positive regional TEM accelerations use common control mass/area weights; hemispheres receive equal weight.",
        "The unresolved/direct budget residual includes direct and parameterized forcing, numerical effects, and closure error; it is not the nudging tendency.",
        "(b) Vorticity-adjusted EP-predicted flow averages (-delta EPFd-vres_C*delta eta)/eta_E pointwise, with E denoting U or T.",
        "This prediction assumes delta A_v=-delta EPFd; both eta_C and eta_E use the corresponding matched mean states.",
        "Experiment-vorticity magnitudes below 1e-8 s^-1 are excluded from panel (b); its retained control mass fraction is exported.",
        "Small symbols represent matched non-overlapping 10-day blocks; large symbols represent full-record means.",
        "Intervals use paired-date joint primitive resampling, with independent hemisphere draws; nonlinear quantities are recomputed in each draw.",
        "(c) HC strength uses the original control-vorticity predictor V_EP = <-delta EPFd/eta_C>_R.",
        "HC strength is the difference of native-grid maximum |mean Psi*| in 0-25 degrees, 200-700 hPa, calculated separately in each simulation and hemisphere.",
        "Precipitation is the 10S-10N cosine-area mean from exact bucket-corrected accumulation endpoints, in mm/day.",
        "Block associations are discriminating checks, not causal estimates.",
        "(d) Outward EP-flux boundary torques sum to the flux net. Diamonds show independently integrated archived EPFd.",
        "The divergence theorem applies to integral rho*a*cos(phi)*EPFd dV; plotted torques divide by the control integral rho*a*cos(phi) dV.",
        "This angular-momentum weighting differs from panel (a)'s mass/area weighting. Raw torques and component integrals are also exported.",
        "Reference for spherical TEM divergence: https://doi.org/10.5194/acp-22-197-2022, equations 6-7.",
        "(e,f) U minus control MMC decompositions use matched mean states, common control mass weights, and eastward-positive m/s/day.",
        "Delta A_v mean-product terms: eta_C*delta vres, vres_C*delta eta, delta vres*delta eta.",
        "Delta A_w mean-product terms: -delta wres*(u_z)_C, -wres_C*delta u_z, -delta wres*delta u_z.",
        "Grey hatched bars retain the temporal covariance and spatial-interpolation remainder; outlined bars show the cached mean instantaneous acceleration response.",
        "The three mean-product terms plus the reported remainder reproduce each full-record acceleration response.",
        "Bootstrap draws recompute every decomposition product from jointly resampled primitives; the existing paired method remains unchanged.",
    ]
    if test:
        caption.append("Test mode uses its bounded sample window and omits formal bootstrap intervals.")
    for prefix, total in (("av", "a_vres"), ("aw", "a_wres")):
        entries = []
        for name in (*mechanism.MMC_DECOMPOSITIONS[prefix], f"{prefix}_mean_product_remainder", total):
            text = f"{name}={means[0][name] * SECONDS_PER_DAY:.4g}"
            if name in intervals[0]:
                low, high = np.asarray(intervals[0][name]) * SECONDS_PER_DAY
                text += f" (95% interval {low:.4g} to {high:.4g})"
            entries.append(text)
        caption.append("U minus control, m/s/day: " + "; ".join(entries) + ".")
    for values, ci, (_name, _run, _control, label) in zip(means, intervals, specs):
        balance_text = []
        for name, symbol in (("balance_ep_v", "B_EP-v"), ("balance_w_v", "B_w/v")):
            text = f"{symbol}={values[name]:.3g}"
            if name in ci:
                text += f" (95% interval {ci[name][0]:.3g} to {ci[name][1]:.3g})"
            balance_text.append(text)
        caption.append(label.replace("T from U", "T") + ": " + "; ".join(balance_text) + ".")
        caption.append(
            label.replace("T from U", "T") + ": boundary-minus-volume = "
            f"{values['boundary_closure_error'] * SECONDS_PER_DAY:.4g} m/s/day; "
            f"relative to boundary magnitudes = {100 * values['boundary_closure_relative']:.2g}%."
        )
    (Path(saver.save_loc) / "mechanism_check_caption.txt").write_text("\n".join(caption) + "\n")
    for values, (_name, _run, _control, label) in zip(means, specs):
        progress(
            f"Figure 14 {label}: B_EP-v={values['balance_ep_v']:.3g}, B_w/v={values['balance_w_v']:.3g}; "
            f"valid regional mass={100 * values['valid_mass_fraction']:.1f}%"
        )
        progress(f"Figure 14 {label}: EP-predicted flow (m/s): "
                 f"vorticity-adjusted={values['predicted_vres_adjusted']:.4g}, "
                 f"control-vorticity={values['predicted_vres']:.4g}, observed={values['observed_vres']:.4g}; "
                 f"adjusted prediction valid mass={100 * values['adjusted_prediction_valid_mass_fraction']:.1f}%")
        progress(
            f"Figure 14 {label}: A_v decomposition (m/s/day): "
            + ", ".join(f"{name}={values[name] * SECONDS_PER_DAY:.4g}" for name in
                        ("av_flow", "av_vorticity", "av_interaction", "av_mean_product_remainder"))
        )
        progress(
            f"Figure 14 {label}: A_w decomposition (m/s/day): "
            + ", ".join(f"{name}={values[name] * SECONDS_PER_DAY:.4g}" for name in
                        (*mechanism.MMC_DECOMPOSITIONS["aw"], "aw_mean_product_remainder"))
        )
        progress(f"Figure 14 {label}: signed contribution/total ratios: "
                 f"A_v flow={values['av_flow_fraction']:.3g}; A_w shear={values['aw_shear_fraction']:.3g}")
        progress(
            f"Figure 14 {label}: TEM closure={values['budget_closure'] * SECONDS_PER_DAY:.3g}, "
            f"boundary minus volume={values['boundary_closure_error'] * SECONDS_PER_DAY:.3g} m/s/day; "
            f"relative boundary error={100 * values['boundary_closure_relative']:.2g}%; "
            f"EP-y={values['torque_a_ep_meridional'] * SECONDS_PER_DAY:.3g}, "
            f"EP-z={values['torque_a_ep_vertical'] * SECONDS_PER_DAY:.3g} m/s/day"
        )
        progress(
            f"Figure 14 {label}: boundary-component errors: "
            f"meridional={values['boundary_meridional_closure_error'] * SECONDS_PER_DAY:.3g}, "
            f"vertical={values['boundary_vertical_closure_error'] * SECONDS_PER_DAY:.3g} m/s/day"
        )
        if values["boundary_closure_relative"] > 0.05:
            progress(f"Figure 14 {label}: boundary closure error exceeds 5% of boundary magnitudes; do not interpret as a closed EP budget")
    progress("Figure 14 block associations (not causal estimates):\n" + associations.to_string(index=False))
    saver.save(plot_mechanism_check(means, block_values, intervals),
               "publication_mechanism_check", number=14)


def mechanism_moisture_responses(fields, blocks, *, bootstrap):
    """Reuse cached full-column moisture statistics and established response intervals."""
    completed = complete_moisture_pressure_column(fields)
    transports, convergences = folded_moisture_profiles(completed)
    profiles, intervals = [], []
    for _name, run, control, _label in COMPARISONS[1:3]:
        response = transports.isel(simulation=run, drop=True) - transports.isel(simulation=control, drop=True)
        attribution = mean_circulation_response_components(completed, run, control)
        response = xr.merge([response, attribution])
        response["mean_flux_convergence"] = (
            convergences.mean_meridional_moisture_flux.isel(simulation=run, drop=True)
            - convergences.mean_meridional_moisture_flux.isel(simulation=control, drop=True)
        ) * SECONDS_PER_DAY
        for name in response:
            response[name].attrs["units"] = "mm day-1" if name == "mean_flux_convergence" else "kg m-1 s-1"
        ci = {}
        if bootstrap:
            ci.update(bootstrap_moisture_attribution_intervals(blocks, run, control))
            for name in MOISTURE_FLUX_COMPONENTS:
                ci[name] = bootstrap_field_response(blocks, f"transport_{name}", run, control)
            ci["mean_flux_convergence"] = bootstrap_field_response(
                blocks, "convergence_mean_meridional_moisture_flux", run, control,
            ) * SECONDS_PER_DAY
        profiles.append(response)
        intervals.append(ci)
    return profiles, intervals


def plot_mechanism_summary(means, momentum_intervals, profiles, moisture_intervals, circulation_metrics):
    """Summarize fixed-box momentum changes and full-column moisture attribution."""
    fig = plt.figure(figsize=(10.0, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.9, 1.1))
    budget_axis = fig.add_subplot(grid[0, 0])
    decomposition_grid = grid[0, 1].subgridspec(1, 2, wspace=0.08)
    av_axis = fig.add_subplot(decomposition_grid[0, 0])
    aw_axis = fig.add_subplot(decomposition_grid[0, 1], sharey=av_axis)
    transport_axis = fig.add_subplot(grid[1, 0])
    attribution_axis = fig.add_subplot(grid[1, 1], sharex=transport_axis, sharey=transport_axis)
    colours = [simulation_colour(SIMULATIONS[run][0]) for _name, run, _control, _label in COMPARISONS[1:3]]
    labels = [label.replace("T from U", "T").replace(" - ", " − ")
              for _name, _run, _control, label in COMPARISONS[1:3]]

    def bar(axis, x, name, index, *, width=0.32, outlined=False, remainder=False):
        colour = "0.5" if remainder else colours[index]
        value = means[index][name] * SECONDS_PER_DAY
        axis.bar(x, value, width=width, color="white" if outlined else colour,
                 edgecolor=colour, linewidth=0.9, alpha=1.0 if outlined else 0.85,
                 hatch="//" if remainder else None, zorder=2)
        if name in momentum_intervals[index]:
            low, high = np.asarray(momentum_intervals[index][name]) * SECONDS_PER_DAY
            axis.vlines(x, low, high, color="0.15", linewidth=0.8, zorder=4)
            axis.hlines([low, high], x-0.04, x+0.04, color="0.15", linewidth=0.8, zorder=4)

    budget_names = ("a_ep", "a_vres", "a_wres", "unresolved_direct_momentum_residual")
    for index in range(2):
        for position, name in enumerate(budget_names):
            bar(budget_axis, position + (-0.18, 0.18)[index], name, index)
    budget_axis.set_xticks(range(4), [r"$\Delta\mathrm{EPFd}$", r"$\Delta A_{v^*}$", r"$\Delta A_{w^*}$",
                                    "Unresolved/direct\n" + r"$\Delta R$"])
    budget_axis.set_title(r"(a) TEM balance: $\Delta\mathrm{EPFd},\;\Delta A_{v^*},\;\Delta A_{w^*},\;\Delta R$",
                          loc="left", fontsize=8)
    budget_axis.set_ylabel(r"Acceleration response (m s$^{-1}$ day$^{-1}$)", fontsize=7)
    for axis, prefix, total, ticks in (
        (av_axis, "av", "a_vres", [r"$\eta_C\Delta v^*$", r"$v_C^*\Delta\eta$", r"$\Delta v^*\Delta\eta$",
                                   "Remainder", r"$\Delta A_{v^*}$"]),
        (aw_axis, "aw", "a_wres", [r"$-\Delta w^*(u_z)_C$", r"$-w_C^*\Delta u_z$", r"$-\Delta w^*\Delta u_z$",
                                   "Remainder", r"$\Delta A_{w^*}$"]),
    ):
        names = (*mechanism.MMC_DECOMPOSITIONS[prefix], f"{prefix}_mean_product_remainder", total)
        for position, name in enumerate(names):
            bar(axis, position, name, 0, width=0.65, outlined=position == 4, remainder=position == 3)
        axis.set_xticks(range(5), ticks, rotation=35, ha="right", fontsize=7)
        axis.margins(x=0.08)
    av_axis.set_title(r"(b) U − control: $\Delta(\eta v^*)$", loc="left", fontsize=8)
    aw_axis.set_title(r"$\Delta(-w^*u_z)$ decomposition", loc="left", fontsize=8)
    aw_axis.tick_params(labelleft=False)
    av_axis.set_ylabel(r"m s$^{-1}$ day$^{-1}$", fontsize=7)

    transport_styles = {
        "total_meridional_moisture_flux": ("-", "Total"),
        "mean_meridional_moisture_flux": ("--", "Mean circulation"),
        "eddy_meridional_moisture_flux": (":", "Eddy"),
    }
    attribution_styles = {
        "mean_circulation_change": ("--", "Mean circulation"),
        "wind_contribution": ("-", r"Wind: $\Delta\bar v\,\bar q_C$"),
        "humidity_contribution": ("-.", r"Humidity: $\bar v_C\Delta\bar q$"),
        "interaction": (":", r"Interaction: $\Delta\bar v\Delta\bar q$"),
    }
    displayed = []
    for index, (profile, ci) in enumerate(zip(profiles, moisture_intervals)):
        for axis, styles in ((transport_axis, transport_styles), (attribution_axis, attribution_styles)):
            for name, (linestyle, _label) in styles.items():
                field = profile[name]
                displayed.append(field.sel(absolute_latitude=slice(0.0, 50.0)))
                axis.plot(field.absolute_latitude, field, color=colours[index], linestyle=linestyle,
                          linewidth=1.5, label=f"{labels[index]}: {_label}", zorder=3)
                if name in ci:
                    _plot_profile_bootstrap_interval(axis, ci[name], colours[index], alpha=0.10)
    limit = 1.10 * _profile_response_limit(displayed)
    for axis, title, styles in (
        (transport_axis, r"(c) $\Delta\overline{vq}=\Delta(\bar v\bar q)+\Delta\overline{v'q'}$", transport_styles),
        (attribution_axis, r"(d) $\Delta(\bar v\bar q)=\Delta\bar v\,\bar q_C+\bar v_C\Delta\bar q+\Delta\bar v\Delta\bar q$", attribution_styles),
    ):
        axis.set_title(title, loc="left", fontsize=8)
        axis.set_xlim(0.0, 50.0)
        axis.set_ylim(-limit, limit)
        axis.set_xticks([0, 10, 20, 30, 40, 50])
        axis.set_xlabel("Latitude (°)", fontsize=7)
        _plot_publication_hadley_edges(axis, circulation_metrics)
        handles = [Line2D([], [], color="0.25", linestyle=style, linewidth=1.5, label=label)
                   for style, label in styles.values()]
        axis.legend(handles=handles, loc="lower left", fontsize=6.2, ncol=2,
                    frameon=True, facecolor="white", edgecolor="0.35", framealpha=1.0)
    transport_axis.set_ylabel("Equatorward transport response\n" + r"(kg m$^{-1}$ s$^{-1}$)", fontsize=7)
    attribution_axis.tick_params(labelleft=False)
    for axis in fig.axes:
        axis.axhline(0.0, color="0.55", linewidth=0.6, zorder=0)
        axis.tick_params(labelsize=7)
    handles = [Patch(facecolor=colour, edgecolor=colour, label=label) for colour, label in zip(colours, labels)]
    handles.append(Line2D([], [], label="HC edge", **HC_EDGE_LINE_STYLE))
    if any(momentum_intervals):
        handles.append(Line2D([], [], color="0.25", marker="|", label="95% block-bootstrap intervals"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=len(handles),
               fontsize=6.2, frameon=True, facecolor="white", edgecolor="0.35", framealpha=1.0)
    fig.get_layout_engine().set(rect=(0.0, 0.07, 1.0, 0.93))
    return fig


def save_mechanism_summary(fields, blocks, moisture_fields, moisture_blocks, circulation_metrics, saver, *, test):
    """Export Figure 12 and quantitative tables using existing compact cache families."""
    progress("Figure 12: TEM balance and decompositions from mechanism cache")
    specs = COMPARISONS[1:3]
    means, per_block, momentum_intervals = mechanism.mechanism_analysis(
        fields, blocks, specs, resamples=N_BOOTSTRAP, minimum_blocks=MIN_BLOCKS,
        seed=BOOTSTRAP_SEED, percentiles=CI_PERCENTILES, bootstrap=not test,
        progress_label="Figure 12",
    )
    progress("Figure 12: moisture profiles and attribution bootstrap from compact cache")
    profiles, moisture_intervals = mechanism_moisture_responses(moisture_fields, moisture_blocks, bootstrap=not test)
    summary, _block_table, _associations = mechanism.mechanism_tables(means, per_block, momentum_intervals, specs)
    root = Path(saver.save_loc)
    write_csv_atomic(summary, root / "mechanism_summary_momentum.csv", index=False)
    rows = []
    for profile, intervals, (_name, _run, _control, label) in zip(profiles, moisture_intervals, specs):
        for name in profile:
            field = profile[name]
            for index, latitude in enumerate(field.absolute_latitude.values):
                ci = intervals.get(name)
                rows.append(dict(comparison=label.replace("T from U", "T"), diagnostic=name,
                                 absolute_latitude=float(latitude), full_record_mean=float(field.isel(absolute_latitude=index)),
                                 ci_low=float(ci.ci_low.isel(absolute_latitude=index)) if ci is not None else np.nan,
                                 ci_high=float(ci.ci_high.isel(absolute_latitude=index)) if ci is not None else np.nan,
                                 units=field.attrs["units"]))
    write_csv_atomic(pd.DataFrame(rows), root / "mechanism_summary_moisture.csv", index=False)
    caption = [
        f"Figure 12: U minus control and T minus control. Momentum box: {mechanism.REGION_DESCRIPTION}.",
        "(a) EPFd, A_v, A_w and the unresolved/direct momentum-budget residual use common control mass/area weights.",
        "The residual includes direct and parameterized forcing, numerical effects and closure error; it is not the nudging tendency.",
        "Tendency and density-storage terms remain in the exported momentum table; four bars alone do not imply a steady closed balance.",
        "(b) U minus control: A_v = eta_C*delta vres + vres_C*delta eta + delta vres*delta eta.",
        "A_w = -delta wres*(u_z)_C - wres_C*delta u_z - delta wres*delta u_z.",
        "Grey hatched remainders retain temporal covariance and pressure-remapping effects; outlined bars show total acceleration responses.",
        "(c,d) Full-column integral(flux dp)/g; positive folded transport points equatorward. Hemispheres receive equal weight.",
        f"Pressure coverage: {float(moisture_fields.pres.min()):g}-{MOISTURE_BOTTOM_PRESSURE_HPA:g} hPa; the existing fixed 950-to-1000 hPa boundary treatment applies.",
        "(c) Total = mean circulation + combined stationary/transient eddy transport.",
        "(d) delta(v_bar*q_bar) = delta v_bar*q_C + v_C*delta q_bar + delta v_bar*delta q_bar.",
        "The exported moisture table retains convergence of the vertically integrated mean-circulation flux response; positive means convergence, in mm/day.",
        "TEM intervals retain paired-date primitive resampling with independent hemisphere draws, as in Figure 14.",
        "Moisture intervals retain the established independent simulation/hemisphere block resampling and recompute wind-humidity products.",
        "All intervals use the established 95% percentile method. The new figure does not change existing uncertainty methods.",
    ]
    for mean, (_name, _run, _control, label) in zip(means, specs):
        caption.append(f"{label}: A_v flow/total={mean['av_flow_fraction']:.4g}; A_w shear/total={mean['aw_shear_fraction']:.4g}.")
    if test:
        caption.append("TEST SUBSET: no formal uncertainty or production-period interpretation.")
    (root / "mechanism_summary_caption.txt").write_text("\n".join(caption) + "\n")
    saver.save(plot_mechanism_summary(means, momentum_intervals, profiles, moisture_intervals, circulation_metrics),
               "publication_figure2_moisture_transport_decomposition", number=12)


def plot_daily_predictive_tests(regressions: pd.DataFrame, asymmetries: pd.DataFrame, *, show_bootstrap: bool) -> plt.Figure:
    """Show separate/joint predictors and integrated HC/P asymmetry for each simulation."""
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.8), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=(1.2, 0.9, 1.1)))
    axes[1].sharey(axes[0])
    labels = list(regressions.comparison.unique())
    colours = [simulation_colour(SIMULATIONS[index][0]) for index in range(len(labels))]
    markers = ("s", "o", "^")

    def estimate(axis, position, value, low, high, colour, marker):
        if show_bootstrap and np.isfinite(low) and np.isfinite(high):
            axis.vlines(position, low, high, color=colour, linewidth=1.25)
            axis.hlines([low, high], position-0.035, position+0.035, color=colour, linewidth=1.25)
        axis.plot(position, value, color=colour, marker=marker, markersize=5, linestyle="none")

    for index, (label, colour, marker) in enumerate(zip(labels, colours, markers)):
        offset = (index - (len(labels)-1)/2) * 0.20
        selected = regressions[(regressions.comparison == label) & (regressions.treatment == "daily_anomaly")]
        for position, (model, _target, _predictor) in enumerate(lead_lag.PREDICTIVE_MODELS):
            row = selected[selected.model == model].iloc[0]
            estimate(axes[0], position+offset, row.standardized_cross_coefficient,
                     row.standardized_ci_low, row.standardized_ci_high, colour, marker)
        joint = selected[selected.model == lead_lag.JOINT_PREDICTIVE_MODEL[0]]
        for position, predictor in enumerate(lead_lag.JOINT_PREDICTIVE_MODEL[2]):
            row = joint[joint.predictor == predictor].iloc[0]
            estimate(axes[1], position+offset, row.standardized_cross_coefficient,
                     row.standardized_ci_low, row.standardized_ci_high, colour, marker)
        selected = asymmetries[(asymmetries.comparison == label) & (asymmetries.treatment == "daily_anomaly")
                               & (asymmetries["first"] == "hc_streamfunction") & (asymmetries.second == "precipitation")]
        selected = selected.sort_values("L_days")
        axes[2].plot(selected.L_days, selected.integrated_asymmetry, color=colour, marker=marker,
                     markersize=4, linewidth=1.0)
        if show_bootstrap:
            axes[2].fill_between(selected.L_days, selected.ci_low, selected.ci_high,
                                 color=colour, alpha=0.12, linewidth=0)
    for axis in axes:
        axis.axhline(0.0, color="0.65", linewidth=0.6)
        axis.tick_params(labelsize=8)
        axis.margins(y=0.15)
    axes[0].set_title("(a) Separate predictors", loc="left", fontsize=9)
    axes[0].set_xticks(np.arange(3), [r"$P_t\rightarrow\Psi_{t+1}$", r"$\Psi_t\rightarrow P_{t+1}$",
                                     r"$V_{\rm EP,t}\rightarrow\Psi_{t+1}$"])
    axes[0].set_xlim(-0.5, 2.5)
    axes[0].set_ylabel("Cross-predictor coefficient (standardized)", fontsize=8)
    axes[0].set_xlabel("Intercept and target persistence included", fontsize=7)
    axes[1].set_title("(b) Joint predictors of HC", loc="left", fontsize=9)
    axes[1].set_xticks([0, 1], [r"$b$: $V_{\rm EP}$", r"$c$: $P$"])
    axes[1].set_xlim(-0.5, 1.5)
    axes[1].tick_params(labelleft=False)
    axes[1].set_xlabel(r"$\Psi_{t+1}=a\Psi_t+bV_{\rm EP,t}+cP_t+\epsilon$", fontsize=8)
    axes[2].set_title("(c) Integrated HC/P asymmetry", loc="left", fontsize=9)
    axes[2].set_xticks(range(1, lead_lag.MAX_LAG_DAYS+1))
    axes[2].set_xlim(0.8, lead_lag.MAX_LAG_DAYS+0.2)
    axes[2].set_ylabel(r"$A(L)$; positive: HC leads P", fontsize=8)
    axes[2].set_xlabel(r"Integration limit $L$ (days)", fontsize=8)
    handles = [Line2D([], [], color=colour, linestyle="none", marker=marker, markersize=5, label=label)
               for label, colour, marker in zip(labels, colours, markers)]
    if show_bootstrap:
        handles.append(axes[0].errorbar([np.nan], [np.nan], yerr=[1.0], fmt="none", ecolor="0.25", capsize=3,
                                       label="95% 10-day block-bootstrap interval"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=len(handles),
               fontsize=7, frameon=True, facecolor="white", edgecolor="0.35", framealpha=1.0)
    fig.get_layout_engine().set(rect=(0.0, 0.13, 1.0, 0.87))
    return fig


def save_daily_lead_lag(daily: xr.Dataset, saver, *, test: bool) -> None:
    """Export cached daily predictive tests, asymmetries, and supplementary lag curves."""
    progress("Figure 15: separate/joint regressions and integrated lead-lag asymmetry bootstrap")
    all_anomalies = lead_lag.daily_anomalies(daily)
    responses = xr.concat([
        all_anomalies.isel(simulation=run_index, drop=True).expand_dims(
            comparison=[label.replace(" - control", "").replace("T from U", "T")])
        for _name, run_index, _control_index, label in COMPARISONS[:3]
    ], dim="comparison", join="exact")
    regressions, asymmetries = lead_lag.predictive_analysis(
        responses, block_days=int(BLOCK_DAYS), resamples=N_BOOTSTRAP,
        minimum_blocks=MIN_BLOCKS, seed=BOOTSTRAP_SEED, percentiles=CI_PERCENTILES, bootstrap=not test)
    progress("Figure 15: predictive fits complete; supplementary lag checks")
    curves, summary = lead_lag.lag_analysis(
        responses, block_days=int(BLOCK_DAYS), resamples=N_BOOTSTRAP,
        minimum_blocks=MIN_BLOCKS, seed=BOOTSTRAP_SEED, percentiles=CI_PERCENTILES, bootstrap=not test)
    for table in (regressions, asymmetries, curves, summary):
        table["anomaly_reference"] = "own full-record simulation time mean"
    root = Path(saver.save_loc)
    anomaly_table = responses.to_dataframe().reset_index().assign(anomaly_reference="own full-record simulation time mean")
    write_csv_atomic(anomaly_table, root / "lead_lag_daily_anomalies.csv", index=False)
    # Refresh the legacy filename too, so prior exports cannot imply control differences.
    write_csv_atomic(anomaly_table, root / "lead_lag_daily_responses.csv", index=False)
    raw = daily[["hc_streamfunction", "moisture_convergence", "pw", "precipitation", "day_start", "day_end"]]
    write_csv_atomic(raw.to_dataframe().reset_index(), root / "lead_lag_daily_absolute_indices.csv", index=False)
    write_csv_atomic(curves, root / "lead_lag_correlations.csv", index=False)
    write_csv_atomic(summary, root / "lead_lag_summary.csv", index=False)
    write_csv_atomic(regressions, root / "lead_lag_predictive_regressions.csv", index=False)
    write_csv_atomic(asymmetries, root / "lead_lag_correlation_asymmetry.csv", index=False)
    write_csv_atomic(asymmetries, root / "lead_lag_integrated_asymmetry.csv", index=False)
    # Retain all simulations, including control, as separate within-run checks.
    absolute = all_anomalies.rename(simulation="comparison")
    absolute_curves, absolute_summary = lead_lag.lag_analysis(
        absolute, block_days=int(BLOCK_DAYS), resamples=N_BOOTSTRAP,
        minimum_blocks=MIN_BLOCKS, seed=BOOTSTRAP_SEED, percentiles=CI_PERCENTILES,
        bootstrap=not test, pairs=lead_lag.PAIRS[1:])
    for table in (absolute_curves, absolute_summary):
        table["anomaly_reference"] = "own full-record simulation time mean"
    write_csv_atomic(absolute_curves, root / "lead_lag_absolute_correlations.csv", index=False)
    write_csv_atomic(absolute_summary, root / "lead_lag_absolute_summary.csv", index=False)
    absolute_regressions, absolute_asymmetries = lead_lag.predictive_analysis(
        absolute, block_days=int(BLOCK_DAYS), resamples=N_BOOTSTRAP,
        minimum_blocks=MIN_BLOCKS, seed=BOOTSTRAP_SEED, percentiles=CI_PERCENTILES,
        bootstrap=not test, models=lead_lag.PREDICTIVE_MODELS[:2], pairs=lead_lag.PAIRS[1:])
    for table in (absolute_regressions, absolute_asymmetries):
        table["anomaly_reference"] = "own full-record simulation time mean"
    write_csv_atomic(absolute_regressions, root / "lead_lag_absolute_predictive_regressions.csv", index=False)
    write_csv_atomic(absolute_asymmetries, root / "lead_lag_absolute_correlation_asymmetry.csv", index=False)
    caption = [
        f"Production daily timing checks: {ANALYSIS_START} to {ANALYSIS_END}; {daily.sizes['Time']} complete 24-hour intervals.",
        "Every index uses its simulation's own full-record production time mean as the climatology; no control subtraction enters Figure 15.",
        "Each day contains four right-ended six-hour samples; precipitation uses exact bucket-corrected accumulation endpoints.",
        f"V_EP uses the existing mechanism box: {mechanism.REGION_DESCRIPTION}.",
        "V_EP averages (-EPFd_prime-vres_climatology*eta_prime)/eta_daily pointwise, using the same simulation's fixed climatological mass weights.",
        "Primes denote departures from that simulation's own pointwise time mean. The regional V_EP series also has its temporal mean removed.",
        "Psi_HC is the fixed signed cos(latitude)*dp mean of native residual streamfunction over 5-15 degrees, 150-850 hPa.",
        "Native pressure remapping precedes daily averaging; SH streamfunction reverses sign before equal hemisphere averaging.",
        "The HC region is fixed across dates and simulations; no daily maxima or moving regions enter this index.",
        "Moisture convergence is the tropical 10S-10N mean-circulation boundary inflow integrated over 850-950 hPa.",
        "Its layer remains within native model coverage, with no surface extrapolation. Units are mm/day; positive means convergence.",
        "PW uses archived precipw; PW and precipitation use 10S-10N cosine-area means.",
        "Correlation at lag L is corr(first(t), second(t+L)); positive lag means the first index leads.",
        "Negative lag in supplementary HC/P curves means precipitation leads HC; positive lag means HC leads precipitation.",
        "All available non-wrapped production pairs enter each lag; n_pairs is exported for each correlation.",
        "Panel (a) fits target(t+1)=intercept+a*target(t)+b*predictor(t), separately for P to HC, HC to P, and V_EP to HC.",
        "Panel (a) shows separate-model cross-predictor coefficients in standardized units, using each complete daily series' fixed mean and standard deviation.",
        "Physical-unit b, intercept, persistence coefficients, and in-sample R-squared increments are retained in CSVs.",
        "A significantly positive b requires the two-sided 95% block-bootstrap interval to lie above zero; no textbook p-values enter this test.",
        "Form daily regression tuples before resampling; the final day supplies no next-day target and is never joined to the first day.",
        "Panel (b) fits Psi(t+1)=intercept+a*Psi(t)+b*V_EP(t)+c*P(t) and reports standardized b and c with block-bootstrap intervals.",
        "Each joint coefficient conditions on both HC persistence and the other cross-predictor; singular designs remain undefined.",
        "Panel (c) shows A(L)=(1/L)*sum_tau=1..L[C(+tau)-C(-tau)] for HC versus P, with L=1..5 days.",
        "Positive A(L) favours HC-leading associations; negative A(L) favours precipitation-leading associations.",
        "Average the lag differences within each bootstrap draw; joint dates preserve covariance across directions and lags.",
        "The plot uses unsmoothed within-simulation daily anomalies; separate linear-detrend sensitivities and full lag curves remain in CSVs.",
        "Legend labels control, U and T denote separate simulations, each centred on its own production mean.",
        "Uncertainty uses the established circular moving-block generator with 10-day blocks and jointly sampled dates across all indices and experiments.",
        "Lag pairs are formed before resampling; artificial neighbours across bootstrap block joins never enter a lag calculation.",
        "Intervals are pointwise; scanning eleven lags does not supply an independent causal or multiple-testing significance claim.",
        "The leading-minus-following contrast is max r(+1..+5) minus max r(-5..-1), with a joint-bootstrap interval.",
        "For HC versus P, a negative contrast favours precipitation-leading associations; a positive contrast favours HC-leading associations.",
        "Intervals are pointwise, without a multiple-model correction. In-sample R-squared increments do not measure held-out forecast skill.",
        "A broad peak, interval crossing zero, or trend-sensitive result leaves temporal ordering unresolved. Conditional prediction does not establish causality.",
    ]
    if test:
        caption.append("TEST SUBSET: bounded intervals only; no formal uncertainty or production-period interpretation.")
    primary = summary[(summary['first'] == "hc_streamfunction") & (summary.second == "precipitation")]
    caption.append("Supplementary HC/P lag-curve summary:\n" + primary.to_string(index=False))
    absolute_primary = absolute_summary[(absolute_summary['first'] == "hc_streamfunction")
                                        & (absolute_summary.second == "precipitation")]
    caption.append("Within-simulation HC versus P sensitivity:\n" + absolute_primary.to_string(index=False))
    report_columns = ["comparison", "model", "predictor", "conditioning_predictors", "standardized_cross_coefficient", "standardized_ci_low",
                      "standardized_ci_high", "significantly_positive", "persistence_coefficient", "status"]
    predictive_primary = regressions[regressions.treatment == "daily_anomaly"][report_columns]
    absolute_predictive_primary = absolute_regressions[absolute_regressions.treatment == "daily_anomaly"][report_columns]
    asymmetry_primary = asymmetries[(asymmetries.treatment == "daily_anomaly")
                                    & (asymmetries["first"] == "hc_streamfunction")
                                    & (asymmetries.second == "precipitation")]
    progress("Figure 15: persistence-adjusted predictive tests:\n" + predictive_primary.to_string(index=False))
    progress("Figure 15: within-simulation predictive checks:\n" + absolute_predictive_primary.to_string(index=False))
    progress("Figure 15: HC/P integrated lead-lag asymmetries:\n" + asymmetry_primary.to_string(index=False))
    caption.append("Persistence-adjusted predictive tests:\n" + predictive_primary.to_string(index=False))
    caption.append("Within-simulation predictive checks:\n" + absolute_predictive_primary.to_string(index=False))
    caption.append("HC/P integrated lead-lag asymmetries:\n" + asymmetry_primary.to_string(index=False))
    (root / "lead_lag_caption.txt").write_text("\n".join(caption) + "\n")
    saver.save(plot_daily_predictive_tests(regressions, asymmetries, show_bootstrap=not test), "production_daily_lead_lag", number=15)


def plot_tem_meridional_factor_components(
    tem_fields: xr.Dataset,
    circulation_fields: xr.Dataset,
    circulation_metrics: xr.Dataset,
) -> plt.Figure:
    """Plot the raw Coriolis and relative-vorticity factors in A_vres."""

    pressure_grid = _tem_reference_pressure_grid(tem_fields)
    pressure_levels = _publication_pressure_levels()
    simulations = (0, 1, 2)
    coriolis = [
        tem_fields.coriolis_parameter.isel(simulation=index) * 1.0e5
        for index in simulations
    ]
    relative_vorticity = [
        tem_fields.relative_vorticity.isel(simulation=index) * 1.0e5
        for index in simulations
    ]
    coriolis_levels = np.arange(0.0, 15.0, 1.0)
    vorticity_levels = symmetric_percentile_levels_for_fields(
        _tem_colour_scale_fields(relative_vorticity),
        percentile=98.0,
        ndiv=17,
        zero_field_scale=1.0,
    )
    fig, axes = plt.subplots(
        2,
        len(simulations),
        figsize=(13.0, 7.6),
        constrained_layout=True,
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    images = []
    for row, (fields, levels, cmap) in enumerate(
        (
            (coriolis, coriolis_levels, "viridis"),
            (relative_vorticity, vorticity_levels, difference_cmap(vorticity_levels)),
        )
    ):
        for column, (simulation_index, field) in enumerate(
            zip(simulations, fields)
        ):
            axis = axes[row, column]
            plot_field = _publication_pressure_field(
                field,
                pressure_levels,
                pressure_grid,
            )
            image = axis.contourf(
                plot_field.absolute_latitude,
                plot_field.pres,
                plot_field,
                levels=levels,
                cmap=cmap,
                extend="max" if row == 0 else "both",
            )
            if column == 0:
                images.append(image)
            _plot_publication_jet_and_tropopause(
                axis,
                circulation_fields,
                simulation_index,
                None,
                pressure_levels,
            )
            _plot_tem_location_guides(
                axis,
                circulation_metrics,
                simulation_index,
                None,
                legend=row == 0 and column == 0,
                show_stj=False,
            )
            _set_latpres_axis(axis, ylabel=column == 0)
            if row == 0:
                annotate_panel_label(axis, SIMULATIONS[simulation_index][0])
        axes[row, 0].text(
            -0.24,
            0.5,
            (r"$f$" if row == 0 else r"$\zeta_{\bar u}$"),
            transform=axes[row, 0].transAxes,
            rotation=90.0,
            va="center",
            ha="center",
        )
        fig.colorbar(
            images[row],
            ax=axes[row, :],
            label=(
                r"Coriolis parameter ($10^{-5}$ s$^{-1}$)"
                if row == 0
                else r"Zonal-mean relative vorticity ($10^{-5}$ s$^{-1}$)"
            ),
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color="#087e8b", linestyle="--", linewidth=1.15),
            Line2D([0], [0], color="#087e8b", linewidth=1.45),
            Line2D([0], [0], color="#a05a00", linewidth=1.3),
        ]
    )
    labels.extend(["STJ wind: 30 m/s", "STJ wind: 40 m/s", "Tropopause"])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        fontsize=8,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
    )
    fig.get_layout_engine().set(rect=(0.0, 0.08, 1.0, 0.92))
    fig.suptitle(
        r"Raw TEM meridional acceleration factors: "
        r"$A_{v^*}=v^*(f+\zeta_{\bar u})$"
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

    def values(statistic, simulation, component):
        field = metric_blocks[statistic].sel(simulation=simulation, component=component)
        if "hemisphere" in field.dims:
            field = field.sel(hemisphere=list(HEMISPHERES[:2])).transpose("hemisphere", "block")
        return field.values * SECONDS_PER_DAY

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
                run_blocks = values(statistic, run, component)
                reference_blocks = values(statistic, reference, component)
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
            "normalized flux-operator relative closure RMS="
            f"{float(selected.ep_normalized_operator_closure_relative_rms.values):.3g}; "
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
    global TEM_ARCHIVE_ROOT
    args = parse_command_line(argv)
    TEM_ARCHIVE_ROOT = args.tem_archive_root
    figures = tuple(dict.fromkeys(args.figures or FIGURES))
    if args.test and 15 in figures and (args.test_times < 52 or args.test_times % 4):
        raise ValueError("Figure 15 test mode requires --test-times >= 52, divisible by four (at least 13 complete days)")
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
        for family in ("circulation", "eddy", "moisture", "tem", "mechanism", "lead_lag")
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
    nudging_regions = None
    if set(figures) & {6, 10}:
        nudging_regions = load_nudging_regions(ROOT_DIR, circulation_fields.simulation.values)
        write_json_file(
            Path(saver.save_loc) / "nudging_regions.json",
            nudging_region_report(nudging_regions),
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
                nudging_regions=nudging_regions,
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
    if 9 in figures:
        tem_fields, _tem_metrics, tem_blocks = products["tem"]
        saver.save(
            plot_tem_effective_coriolis_factor(
                tem_fields,
                circulation_metrics,
                tem_blocks,
                show_bootstrap=not args.test,
            ),
            "tem_effective_coriolis_factor_responses",
            number=9,
        )
    if 10 in figures:
        tem_fields, _tem_metrics, tem_blocks = products["tem"]
        moisture_fields, moisture_blocks = products["moisture"]
        figure = plot_publication_thermal_residual_circulation(
            circulation_fields,
            circulation_metrics,
            circulation_blocks,
            tem_fields,
            tem_blocks,
            moisture_fields,
            moisture_blocks,
            show_bootstrap=not args.test,
            nudging_regions=nudging_regions,
        )
        progress("Figure 10: exporting PNG")
        export_started = perf_counter()
        saver.save(
            figure,
            "publication_figure1_circulation_wave_moisture_responses",
            number=10,
        )
        progress(f"Figure 10: export complete ({perf_counter() - export_started:.1f} s)")
    if 12 in figures:
        save_mechanism_summary(*products["mechanism"], *products["moisture"], circulation_metrics, saver, test=args.test)
    if 14 in figures:
        save_mechanism_check(*products["mechanism"], saver, test=args.test)
    if 15 in figures:
        save_daily_lead_lag(*products["lead_lag"], saver, test=args.test)
    if 13 in figures:
        tem_fields, _tem_metrics, _tem_blocks = products["tem"]
        saver.save(
            plot_tem_meridional_factor_components(
                tem_fields,
                circulation_fields,
                circulation_metrics,
            ),
            "tem_meridional_factor_components_raw",
            number=13,
        )
    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
