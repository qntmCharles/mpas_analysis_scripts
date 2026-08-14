#!/usr/bin/env python
"""Diagnose circulation, tropical width, jets, and eddy momentum transport.

The analysis treats both hemispheres as independent aquaplanet realizations.
It saves each hemispheric scalar and their absolute-latitude mean.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
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
    diagnostic_output_dir,
    difference_cmap,
    hemispheric_fold,
    load_latlon_dataset,
    netcdf_safe_dataset,
    progress,
    read_json_file,
    simulation_colour,
    standardize_latlon_dataset,
    subset_between,
    symmetric_percentile_levels_for_fields,
    time_block_slices,
    write_json_file,
)
from diagnostics.analysis_config import (
    BLOCK_LENGTH_DAYS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_PERCENTILES,
    SIMULATION_ARCHIVES,
    SIMULATION_ROOT,
)


FIGURES = (1, 2, 3)
SIMULATIONS = (
    ("control", SIMULATION_ARCHIVES["control"]),
    ("nudged u", SIMULATION_ARCHIVES["nudged-u"]),
    ("nudged T", SIMULATION_ARCHIVES["nudged-T"]),
)
COMPARISONS = (
    ("control", 0, None, "control"),
    ("u_minus_control", 1, 0, "U - control"),
    ("t_minus_control", 2, 0, "T - control"),
    ("u_minus_t", 1, 2, "U - T"),
)

ROOT_DIR = SIMULATION_ROOT
SAVE_LOCATION = diagnostic_output_dir(
    "/home/cwp29/tmp/mpas/crf_on/final/circulation/"
)
CACHE_ROOT = Path(diagnostic_cache_dir("circulation"))
CACHE_SCHEMA = {
    "circulation": "circulation-v1",
    "eddy": "circulation-eddy-v1",
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
DEFAULT_TEST_TIMES = 5

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
    if 3 in selected:
        families.add("eddy")
    return families


def family_source_paths(family: str) -> list[str]:
    pattern = ZONAL_MEAN_PATTERN if family == "circulation" else LATLON_PATTERN
    return [
        path
        for _label, directory in SIMULATIONS
        for path in sorted(glob.glob(str(ROOT_DIR / directory / pattern)))
    ]


def family_manifest(
    family: str,
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
    include_sources: bool = True,
) -> dict:
    configuration = {
        "family": family,
        "simulations": [list(item) for item in SIMULATIONS],
        "analysis_start": ANALYSIS_START,
        "analysis_end": ANALYSIS_END,
        "block_days": None if test else BLOCK_DAYS,
        "minimum_blocks": 1 if test else MIN_BLOCKS,
        "hemisphere_handling": "independent metrics and absolute-latitude mean",
        "test_subset": bool(test),
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
            },
        )
    else:
        configuration.update(
            eddy_definition="instantaneous zonal-mean anomalies",
            upper_tropospheric_layer_hpa=[200.0, 250.0],
            spatial_parity={
                "eddy_momentum_flux": "odd",
                "eddy_momentum_flux_convergence": "even",
            },
        )
        if test:
            configuration["test_longitude_bounds"] = list(
                QUICK_TEST_LON_BOUNDS
            )
    return cache_manifest(
        "mean_state.circulation_diagnostics",
        schema=CACHE_SCHEMA[family],
        configuration=configuration,
        dependencies={"pytropd": pytropd_version()},
        source_paths=family_source_paths(family) if include_sources else None,
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


def atomic_write_dataset(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    netcdf_safe_dataset(dataset).to_netcdf(temporary)
    os.replace(temporary, path)


def atomic_write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


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
    psi: xr.DataArray,
    tropopause: xr.DataArray,
) -> xr.Dataset:
    folded_wind = fold_to_absolute(
        zonal_wind.transpose("pres", "latitude"),
        "even",
    )
    folded_psi = fold_to_absolute(psi.transpose("pres", "latitude"), "odd")
    folded_tropopause = fold_to_absolute(tropopause, "even")
    upper_wind = pressure_mean(folded_wind, 100.0, 400.0)
    wind_850 = folded_wind.sortby("pres").interp(pres=850.0)
    core_layer = subset_between(folded_wind, "pres", (100.0, 400.0))
    return xr.Dataset(
        {
            "streamfunction": folded_psi,
            "zonal_wind": folded_wind,
            "tropopause_pressure": folded_tropopause,
            "upper_zonal_wind": upper_wind,
            "zonal_wind_850": wind_850,
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


def build_circulation_products(
    pytropd_api: Any | None = None,
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    api = pytropd_api or load_pytropd()
    simulation_names = [item[0] for item in SIMULATIONS]
    field_records = []
    metric_records = []
    block_records = []
    for simulation, directory in SIMULATIONS:
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
                    psi,
                    tropopause,
                ).load()
            )

            blocks = []
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
                block_metrics, _psi, _tropopause = circulation_metric_values(
                    block.uzonal_isobaric,
                    block.umeridional_isobaric,
                    block.t_isobaric,
                    pytropd_api=api,
                )
                blocks.append(block_metrics)
            block_records.append(blocks)
        finally:
            dataset.close()

    fields = xr.concat(
        field_records,
        dim=xr.IndexVariable("simulation", simulation_names),
        combine_attrs="drop_conflicts",
    )
    metrics = make_metric_dataset(simulation_names, metric_records)
    metric_blocks = make_metric_block_dataset(simulation_names, block_records)
    fields.attrs.update(
        spatial_hemisphere_treatment="absolute-latitude fold",
        streamfunction_parity="odd",
        zonal_wind_parity="even",
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


def eddy_fields_from_blocks(blocks: xr.DataArray) -> xr.Dataset:
    mean_flux = blocks.mean("block", skipna=True)
    convergence_blocks = xr.concat(
        [
            eddy_momentum_flux_convergence(blocks.isel(block=index, drop=True))
            for index in range(blocks.sizes["block"])
        ],
        dim=xr.IndexVariable("block", blocks.block.values),
    )
    mean_convergence = convergence_blocks.mean("block", skipna=True)
    folded_flux = fold_to_absolute(mean_flux.transpose("pres", "latitude"), "odd")
    folded_convergence = fold_to_absolute(
        mean_convergence.transpose("pres", "latitude"),
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
        }
    )


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
        chunks={"Time": 4, "latitude": 60, "longitude": 120},
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


def build_eddy_products(
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    simulation_names = [item[0] for item in SIMULATIONS]
    field_records = []
    metric_records = []
    block_metric_records = []
    for simulation, directory in SIMULATIONS:
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
            block_flux = xr.concat(
                [flux.isel(Time=item).mean("Time", skipna=True) for item in slices],
                dim=xr.IndexVariable("block", np.arange(len(slices))),
            ).load()
            field_records.append(eddy_fields_from_blocks(block_flux).load())

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
    products: tuple[xr.Dataset, xr.Dataset, xr.Dataset],
    manifest: dict,
    cache_root: Path | None = None,
) -> None:
    root = CACHE_ROOT if cache_root is None else cache_root
    root.mkdir(parents=True, exist_ok=True)
    for dataset, filename in zip(products, CACHE_FILES[family]):
        dataset = dataset.copy()
        dataset.attrs.update(
            cache_schema=CACHE_SCHEMA[family],
            manifest_sha256=manifest["manifest_sha256"],
            cache_manifest=json.dumps(manifest, sort_keys=True),
        )
        atomic_write_dataset(dataset, cache_path(filename, root))
    write_json_file(str(manifest_path(family, root)), manifest)


def load_family(
    family: str,
    cache_root: Path | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    datasets = []
    for filename in CACHE_FILES[family]:
        with xr.open_dataset(cache_path(filename, cache_root)) as cached:
            datasets.append(cached.load())
    return tuple(datasets)  # type: ignore[return-value]


def prepare_family(
    family: str,
    figures_only: bool,
    rebuild_cache: bool,
    test: bool = False,
    test_times: int = DEFAULT_TEST_TIMES,
    test_start: str | None = None,
    cache_root: Path | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    root = CACHE_ROOT if cache_root is None else cache_root
    manifest = family_manifest(
        family,
        test=test,
        test_times=test_times,
        test_start=test_start,
        include_sources=not figures_only,
    )
    current = not rebuild_cache and family_cache_is_current(
        family,
        manifest,
        root,
        validate_sources=not figures_only,
    )
    if current:
        progress(f"using current {family} cache in {root}")
        return load_family(family, root)
    if figures_only:
        raise FileNotFoundError(
            f"Figure-only mode requires a current {family} cache in {root}"
        )
    products = (
        build_circulation_products(
            test=test,
            test_times=test_times,
            test_start=test_start,
        )
        if family == "circulation"
        else build_eddy_products(
            test=test,
            test_times=test_times,
            test_start=test_start,
        )
    )
    save_family(family, products, manifest, root)
    return products


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
) -> dict[str, float | int | bool]:
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
    atomic_write_dataset(metrics, output_directory / "circulation_metrics.nc")
    atomic_write_table(
        metrics_table(metrics),
        output_directory / "circulation_metrics.csv",
    )
    if include_responses:
        atomic_write_table(
            metric_response_table(metrics, metric_blocks),
            output_directory / "circulation_metric_responses.csv",
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
        4,
        figsize=(15.5, 3.8),
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
    fig.suptitle("Hadley circulation and tropical-edge responses")
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
        4,
        figsize=(15.5, 7.0),
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
    fig.suptitle("Jet structure and tropical-width responses")
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
        4,
        figsize=(15.5, 9.0),
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
        for row in (0, 1):
            annotate_panel_label(axes[row, column], label)
            _set_latpres_axis(axes[row, column], ylabel=column == 0)
        axes[2, column].plot(
            profile.absolute_latitude,
            profile,
            color="tab:purple",
            linewidth=1.6,
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
    fig.suptitle("Eddy momentum transport and circulation locations")
    return fig


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
        for family in ("circulation", "eddy")
        if family in families
    }
    circulation_fields, circulation_metrics, circulation_blocks = products[
        "circulation"
    ]
    metric_products = [circulation_metrics]
    block_products = [circulation_blocks]
    if "eddy" in products:
        _eddy_fields, eddy_metrics, eddy_blocks = products["eddy"]
        metric_products.append(eddy_metrics)
        block_products.append(eddy_blocks)
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
    if args.test:
        progress("test mode omits formal bootstrap response intervals")
    if 1 in figures:
        saver.save(
            plot_streamfunction(circulation_fields, circulation_metrics),
            "streamfunction_tropical_edge_responses",
            number=1,
        )
    if 2 in figures:
        saver.save(
            plot_jet_and_width(circulation_fields, circulation_metrics),
            "jet_structure_tropical_width_responses",
            number=2,
        )
    if 3 in figures:
        eddy_fields, eddy_metrics, _eddy_blocks = products["eddy"]
        saver.save(
            plot_eddy_momentum_transport(
                eddy_fields,
                circulation_metrics,
                eddy_metrics,
            ),
            "eddy_momentum_transport_responses",
            number=3,
        )
    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
