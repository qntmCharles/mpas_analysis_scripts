#!/usr/bin/env python3
"""Diagnose realised thermal-wind balance in the three MPAS-A simulations.

This script retains the physically direct parts of the former
``thermal_wind_scaled_response.py`` diagnostic:

* Figure 1: observed and thermal-wind-implied vertical shear;
* Figure 2: direct realised-state differences from control;
* Figure 13: observed and shear-implied meridional temperature gradients;
* Figure 14: absolute gradient/shear fields and direct experiment responses.

The residuals in Figures 1 and 13 are local thermal-wind balance residuals
(observed minus balance-implied fields within one simulation). They are not a
scalar attribution between experiments. No response is scaled by a realised
temperature change, and no scalar thermal-residual analysis is performed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
for _parent in (
    _THIS_DIR.parent,
    _THIS_DIR.parent.parent,
    _THIS_DIR.parent.parent.parent,
):
    if str(_parent) not in sys.path:
        sys.path.append(str(_parent))

from tropopause import tropopause_profile

from diagnostics.diagnostic_helpers import (
    ANALYSIS_END,
    ANALYSIS_START,
    FigureSaver,
    LATLON_PATTERN,
    N2_latheight,
    ZONAL_MEAN_PATTERN,
    add_symmetric_colorbar,
    annotate_panel_label,
    cache_manifest,
    cache_manifest_matches,
    control_diff_label,
    diagnostic_cache_dir,
    diagnostic_profile_cache_dir,
    diagnostic_output_dir,
    difference_cmap,
    hemispheric_fold,
    load_latlon_dataset,
    load_netcdf,
    model_lat_field,
    precip_rate_from_accum,
    progress,
    read_json_file,
    set_latheight_labels,
    simulation_colour,
    standardize_latlon_dataset,
    subset_between,
    symmetric_levels,
    symmetric_percentile_levels_for_fields,
    time_block_slices,
    total_precip_accum,
    write_csv_atomic,
    write_json_file,
    write_netcdf_atomic,
)
from diagnostics.ttl import TTL_HEIGHT_BOUNDS_M
from diagnostics.analysis_config import (
    ALL_SIMULATION_ARCHIVES,
    ALL_SIMULATION_PUBLICATION_LABELS,
    BLOCK_LENGTH_DAYS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_PERCENTILES,
    SIMULATION_ARCHIVES,
    SIMULATION_PROFILES,
    SIMULATION_PUBLICATION_LABELS,
    SIMULATION_ROOT,
    profile_cache_directory,
    simulation_cache_label,
)
from diagnostics.cache_support import relabel_cached_simulations
from diagnostics.uncertainty import independent_bootstrap_difference


ROOT_DIR = SIMULATION_ROOT
SIMULATIONS = tuple(
    (
        SIMULATION_PUBLICATION_LABELS[name],
        directory,
        simulation_colour(name),
    )
    for name, directory in SIMULATION_ARCHIVES.items()
)
FIGURES = (1, 2, 13, 14)
SAVE_LOCATION = diagnostic_output_dir(
    "/home/cwp29/tmp/mpas/crf_on/final/thermal_wind/"
)
THERMAL_WIND_CACHE_SCHEMA = "thermal-wind-v2-block-uncertainty"
THERMAL_WIND_CACHE_ROOT = diagnostic_cache_dir("thermal_wind")
THERMAL_WIND_PROFILE_CACHE_ROOT = diagnostic_profile_cache_dir("thermal_wind")
TEST_TIME_COUNT: int | None = None
THERMAL_TIME_CHUNKS = {"Time": 40}

RD = 287.0
CP = 1004.0
P0_PA = 100_000.0
G = 9.80665
EARTH_ROTATION_RATE = 7.2921159e-5
EARTH_RADIUS_M = 6.371e6

TTL_Z_BOUNDS = TTL_HEIGHT_BOUNDS_M
TTL_REFERENCE_HEIGHT_M = 16_000.0
DEEP_TROPICAL_TTL_LATITUDE_BOUNDS = (-10.0, 10.0)
THERMAL_WIND_METRIC_Z_BOUNDS = (10_000.0, 22_000.0)
THERMAL_WIND_METRIC_LAT_BOUNDS = (-60.0, 60.0)
EQUATOR_MASK_DEGREES = 5.0

LATLON_VARIABLES = (
    "rainnc",
    "i_rainnc",
    "rainc",
    "i_rainc",
    "precipw",
    "olrtoa",
)
ZONAL_MEAN_VARIABLES = (
    "theta",
    "pressure",
    "uReconstructZonal",
)


def coordinate_layer_weights(
    coordinate: xr.DataArray | np.ndarray,
    dimension: str,
) -> xr.DataArray:
    """Return positive cell-width weights for a one-dimensional coordinate."""

    values = np.asarray(
        coordinate.values if isinstance(coordinate, xr.DataArray) else coordinate,
        dtype=float,
    )
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{dimension!r} coordinate must be non-empty and 1-D")
    if values.size == 1:
        weights = np.ones_like(values)
    else:
        bounds = np.empty(values.size + 1)
        bounds[1:-1] = 0.5 * (values[:-1] + values[1:])
        bounds[0] = values[0] - 0.5 * (values[1] - values[0])
        bounds[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
        weights = np.abs(np.diff(bounds))
    return xr.DataArray(
        weights,
        dims=(dimension,),
        coords={dimension: values},
        name=f"d{dimension}_weight",
    )


def latheight_weights(field: xr.DataArray) -> xr.DataArray:
    """Return cosine-latitude and height-layer weights."""

    weights = xr.ones_like(field)
    if "latitude" in field.dims:
        weights = weights * np.cos(np.deg2rad(field.latitude))
    if "z" in field.dims:
        weights = weights * coordinate_layer_weights(field.z, "z")
    return weights


def deep_tropical_mean_surface(
    field: xr.DataArray,
    latitude_bounds: tuple[float, float] = DEEP_TROPICAL_TTL_LATITUDE_BOUNDS,
) -> xr.DataArray:
    """Return a cosine-latitude-weighted deep-tropical mean."""

    selected = subset_between(field, "latitude", latitude_bounds)
    dimensions = [
        dimension
        for dimension in ("latitude", "longitude")
        if dimension in selected.dims
    ]
    return selected.weighted(
        np.cos(np.deg2rad(selected.latitude))
    ).mean(dimensions)


def deep_tropical_mean_latheight(
    field: xr.DataArray,
    *,
    height: float | None = None,
    height_bounds: tuple[float, float] | None = None,
    latitude_bounds: tuple[float, float] = DEEP_TROPICAL_TTL_LATITUDE_BOUNDS,
) -> xr.DataArray:
    """Return a weighted mean over a latitude-height region."""

    selected = field
    if height is not None:
        selected = selected.interp(z=height, kwargs={"fill_value": np.nan})
    selected = subset_between(selected, "z", height_bounds)
    selected = subset_between(selected, "latitude", latitude_bounds)
    dimensions = [
        dimension
        for dimension in ("Time", "z", "latitude")
        if dimension in selected.dims
    ]
    return selected.weighted(latheight_weights(selected)).mean(dimensions)


def rms_weighted_latheight(
    field: xr.DataArray,
    *,
    latitude_bounds: tuple[float, float] = THERMAL_WIND_METRIC_LAT_BOUNDS,
    height_bounds: tuple[float, float] = THERMAL_WIND_METRIC_Z_BOUNDS,
) -> float:
    """Return a weighted RMS over a latitude-height region."""

    selected = subset_between(field, "latitude", latitude_bounds)
    selected = subset_between(selected, "z", height_bounds)
    dimensions = [
        dimension
        for dimension in ("z", "latitude")
        if dimension in selected.dims
    ]
    value = np.sqrt(
        (selected**2)
        .weighted(latheight_weights(selected))
        .mean(dimensions)
    )
    return float(value.values)


def pattern_correlation_latheight(
    first: xr.DataArray,
    second: xr.DataArray,
    *,
    latitude_bounds: tuple[float, float] = THERMAL_WIND_METRIC_LAT_BOUNDS,
    height_bounds: tuple[float, float] = THERMAL_WIND_METRIC_Z_BOUNDS,
) -> float:
    """Return a weighted spatial pattern correlation."""

    first, second = xr.align(first, second, join="exact")
    first = subset_between(first, "latitude", latitude_bounds)
    second = subset_between(second, "latitude", latitude_bounds)
    first = subset_between(first, "z", height_bounds)
    second = subset_between(second, "z", height_bounds)
    weights = latheight_weights(first)
    dimensions = [
        dimension
        for dimension in ("z", "latitude")
        if dimension in first.dims
    ]
    first = first - first.weighted(weights).mean(dimensions)
    second = second - second.weighted(weights).mean(dimensions)
    numerator = (first * second).weighted(weights).mean(dimensions)
    denominator = np.sqrt(
        (first**2).weighted(weights).mean(dimensions)
        * (second**2).weighted(weights).mean(dimensions)
    )
    return float((numerator / denominator).values)


def model_level_temperature(dataset: xr.Dataset) -> xr.DataArray:
    """Return temperature on native model levels."""

    return dataset.theta * (dataset.pressure / P0_PA) ** (RD / CP)


def latheight_time_mean_field(
    dataset: xr.Dataset,
    variable: str,
) -> xr.DataArray:
    """Return a folded time-mean zonal-mean field as (z, latitude)."""

    if variable == "temperature":
        field = (
            dataset.temperature
            if "temperature" in dataset
            else model_level_temperature(dataset)
        )
    else:
        field = dataset[variable]
    if "Time" in field.dims:
        field = field.mean("Time", skipna=True)
    field = model_lat_field(field)
    field = hemispheric_fold(field, parity="even").load()
    field.name = variable
    return field


def thermal_wind_terms(
    temperature: xr.DataArray,
    zonal_wind: xr.DataArray,
    *,
    equator_mask_degrees: float | None = EQUATOR_MASK_DEGREES,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return observed shear, thermal-wind-implied shear and imbalance.

    The height-coordinate relation is

    ``du/dz = -g/(f a T) dT/dphi``.

    It is not valid at the equator. Set ``equator_mask_degrees=None`` only
    when retaining the near-equatorial region for plotting.
    """

    temperature, zonal_wind = xr.align(
        temperature,
        zonal_wind,
        join="exact",
    )
    temperature = temperature.transpose("z", "latitude")
    zonal_wind = zonal_wind.transpose("z", "latitude")
    latitude_radians = np.deg2rad(temperature.latitude.values)
    coriolis = 2.0 * EARTH_ROTATION_RATE * np.sin(latitude_radians)
    observed = np.gradient(
        zonal_wind.values,
        temperature.z.values,
        axis=0,
    )
    temperature_gradient = np.gradient(
        temperature.values,
        latitude_radians,
        axis=1,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = (
            -G
            / (
                coriolis[None, :]
                * EARTH_RADIUS_M
                * temperature.values
            )
            * temperature_gradient
        )
    observed = np.where(np.isfinite(observed), observed, np.nan)
    implied = np.where(np.isfinite(implied), implied, np.nan)
    if equator_mask_degrees is not None and equator_mask_degrees > 0:
        mask = np.abs(temperature.latitude.values) < equator_mask_degrees
        observed[:, mask] = np.nan
        implied[:, mask] = np.nan
    coordinates = {
        "z": temperature.z,
        "latitude": temperature.latitude,
    }
    observed = xr.DataArray(
        observed,
        dims=("z", "latitude"),
        coords=coordinates,
        name="du_dz_observed",
        attrs={"units": "s-1"},
    )
    implied = xr.DataArray(
        implied,
        dims=("z", "latitude"),
        coords=coordinates,
        name="du_dz_thermal_wind",
        attrs={"units": "s-1"},
    )
    imbalance = (observed - implied).rename("du_dz_imbalance")
    imbalance.attrs["units"] = "s-1"
    return observed, implied, imbalance


def thermal_wind_temperature_gradient_terms(
    temperature: xr.DataArray,
    zonal_wind: xr.DataArray,
    *,
    equator_mask_degrees: float | None = EQUATOR_MASK_DEGREES,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return observed, shear-implied and imbalanced meridional T gradients."""

    temperature, zonal_wind = xr.align(
        temperature,
        zonal_wind,
        join="exact",
    )
    temperature = temperature.transpose("z", "latitude")
    zonal_wind = zonal_wind.transpose("z", "latitude")
    latitude_radians = np.deg2rad(temperature.latitude.values)
    coriolis = 2.0 * EARTH_ROTATION_RATE * np.sin(latitude_radians)
    observed = (
        np.gradient(temperature.values, latitude_radians, axis=1)
        / EARTH_RADIUS_M
    )
    wind_shear = np.gradient(
        zonal_wind.values,
        temperature.z.values,
        axis=0,
    )
    implied = (
        -(coriolis[None, :] * temperature.values / G) * wind_shear
    )
    observed = np.where(np.isfinite(observed), observed, np.nan)
    implied = np.where(np.isfinite(implied), implied, np.nan)
    if equator_mask_degrees is not None and equator_mask_degrees > 0:
        mask = np.abs(temperature.latitude.values) < equator_mask_degrees
        observed[:, mask] = np.nan
        implied[:, mask] = np.nan
    coordinates = {
        "z": temperature.z,
        "latitude": temperature.latitude,
    }
    observed = xr.DataArray(
        observed,
        dims=("z", "latitude"),
        coords=coordinates,
        name="dT_dy_observed",
        attrs={"units": "K m-1"},
    )
    implied = xr.DataArray(
        implied,
        dims=("z", "latitude"),
        coords=coordinates,
        name="dT_dy_thermal_wind",
        attrs={"units": "K m-1"},
    )
    imbalance = (observed - implied).rename("dT_dy_imbalance")
    imbalance.attrs["units"] = "K m-1"
    return observed, implied, imbalance


def load_inputs(
    *,
    need_latlon: bool,
) -> tuple[list[xr.Dataset], list[xr.Dataset]]:
    """Load the fixed simulation inputs required by requested figures."""

    latlon_datasets: list[xr.Dataset] = []
    zonal_datasets: list[xr.Dataset] = []
    for label, directory, _colour in SIMULATIONS:
        progress(f"{label}: loading thermal-wind inputs")
        if need_latlon:
            dataset = load_latlon_dataset(
                ROOT_DIR / directory,
                latlon_pattern=LATLON_PATTERN,
                chunks=THERMAL_TIME_CHUNKS,
            )
            latlon_datasets.append(
                standardize_latlon_dataset(dataset, LATLON_VARIABLES)
            )
        zonal = load_latlon_dataset(
            ROOT_DIR / directory,
            latlon_pattern=ZONAL_MEAN_PATTERN,
            chunks=THERMAL_TIME_CHUNKS,
        )
        zonal_datasets.append(
            standardize_latlon_dataset(zonal, ZONAL_MEAN_VARIABLES)
        )
    return latlon_datasets, zonal_datasets


def thermal_wind_source_paths(directory: str, pattern: str) -> list[str]:
    return sorted(glob.glob(str(ROOT_DIR / directory / pattern)))


def thermal_wind_manifest(
    kind: str,
    *,
    directory: str | None = None,
    include_sources=True,
    simulations: Sequence[tuple[str, str, str]] = SIMULATIONS,
):
    patterns = (
        (LATLON_PATTERN, ZONAL_MEAN_PATTERN)
        if kind == "realised_state_summary"
        else (ZONAL_MEAN_PATTERN,)
    )
    directories = (
        [directory]
        if directory is not None
        else [simulation[1] for simulation in simulations]
    )
    source_paths = [
        path
        for source_directory in directories
        for pattern in patterns
        for path in thermal_wind_source_paths(source_directory, pattern)
    ]
    configuration = {
        "kind": kind,
        "directory": directory,
        "analysis_start": ANALYSIS_START,
        "analysis_end": ANALYSIS_END,
        "simulations": [
            (simulation_cache_label(simulation[0], publication=True), simulation[1])
            for simulation in simulations
            if directory is None or simulation[1] == directory
        ],
        "test_times": TEST_TIME_COUNT,
        "ttl_bounds_m": TTL_Z_BOUNDS,
        "equator_mask_degrees": EQUATOR_MASK_DEGREES,
    }
    if kind == "realised_state_summary":
        configuration.update(
            block_days=BLOCK_LENGTH_DAYS["thermal_wind"],
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            confidence_percentiles=list(CONFIDENCE_PERCENTILES),
        )
    return cache_manifest(
        "mean_state.thermal_wind",
        schema=THERMAL_WIND_CACHE_SCHEMA,
        configuration=configuration,
        source_paths=source_paths if include_sources else None,
    )


def thermal_wind_cache_is_current(
    path: Path, manifest: dict, *, validate_sources=True
) -> bool:
    if not path.is_file():
        return False
    try:
        with xr.open_dataset(path) as cached:
            if cached.attrs.get("cache_schema") != THERMAL_WIND_CACHE_SCHEMA:
                return False
            saved = json.loads(cached.attrs.get("cache_manifest", "null"))
            expected_configuration = manifest.get("configuration", {})
            if expected_configuration.get("directory") is not None:
                saved = dict(saved)
                saved["configuration"] = dict(
                    saved.get("configuration", {})
                )
                saved["configuration"]["simulations"] = (
                    expected_configuration["simulations"]
                )
                saved["manifest_sha256"] = manifest["manifest_sha256"]
            return cache_manifest_matches(
                saved, manifest, compare_sources=validate_sources
            )
    except (OSError, ValueError):
        return False


def write_thermal_wind_cache(
    dataset: xr.Dataset, path: Path, manifest: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = dataset.copy()
    dataset.attrs.update(
        cache_schema=THERMAL_WIND_CACHE_SCHEMA,
        manifest_sha256=manifest["manifest_sha256"],
        cache_manifest=json.dumps(manifest, sort_keys=True),
    )
    write_netcdf_atomic(dataset, path)


def prepare_zonal_mean_caches(
    *, figures_only: bool = False, rebuild_cache: bool = False
) -> list[xr.Dataset]:
    datasets = []
    for label, directory, _colour in SIMULATIONS:
        path = Path(THERMAL_WIND_CACHE_ROOT) / f"{directory}_zonal_mean.nc"
        manifest = thermal_wind_manifest(
            "zonal_time_mean", directory=directory,
            include_sources=not figures_only,
        )
        current = thermal_wind_cache_is_current(
            path, manifest, validate_sources=not figures_only
        )
        if current and not rebuild_cache:
            progress(f"{label}: using thermal-wind cache {path}")
            datasets.append(load_netcdf(path))
            continue
        if figures_only:
            raise FileNotFoundError(
                "Figure-only mode requires a current thermal-wind cache: "
                f"{path}"
            )
        raw = load_latlon_dataset(
            ROOT_DIR / directory,
            latlon_pattern=ZONAL_MEAN_PATTERN,
            chunks=THERMAL_TIME_CHUNKS,
        )
        try:
            raw = standardize_latlon_dataset(raw, ZONAL_MEAN_VARIABLES)
            if TEST_TIME_COUNT is not None:
                raw = raw.isel(Time=slice(0, TEST_TIME_COUNT))
            temperature_mean = model_level_temperature(raw).mean(
                "Time", skipna=True
            ).load()
            mean = raw.mean("Time", skipna=True).load()
            mean["temperature"] = temperature_mean
        finally:
            raw.close()
        write_thermal_wind_cache(mean, path, manifest)
        datasets.append(mean)
    return datasets


def realised_summary_cache_paths() -> tuple[Path, Path]:
    root = Path(THERMAL_WIND_PROFILE_CACHE_ROOT)
    return (
        root / "realised_state_summary.csv",
        root / "realised_state_summary.metadata.json",
    )


def realised_block_cache_paths() -> tuple[Path, Path]:
    root = Path(THERMAL_WIND_PROFILE_CACHE_ROOT)
    return (
        root / "realised_state_blocks.csv",
        root / "realised_state_response_uncertainty.csv",
    )


def realised_simulation_cache_paths(directory: str) -> tuple[Path, Path, Path]:
    root = Path(THERMAL_WIND_CACHE_ROOT)
    return (
        root / f"{directory}_realised_state_summary.csv",
        root / f"{directory}_realised_state_blocks.csv",
        root / f"{directory}_realised_state.metadata.json",
    )


def thermal_profile_simulations(
    profile: str,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            ALL_SIMULATION_PUBLICATION_LABELS[name],
            ALL_SIMULATION_ARCHIVES[name],
            simulation_colour(name),
        )
        for name in SIMULATION_PROFILES[profile]
    )


def thermal_profile_cache_root(profile: str) -> Path:
    parts = (
        ("testing", f"all_{TEST_TIME_COUNT:03d}times")
        if TEST_TIME_COUNT is not None
        else ()
    )
    return profile_cache_directory("thermal_wind", *parts, profile=profile)


def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    write_csv_atomic(table, path, index=False)


def migrate_realised_simulation_cache(
    simulation: tuple[str, str, str],
    *,
    validate_sources: bool,
    skip_uncertainty: bool,
) -> bool:
    """Seed one realised-state cache from a current profile aggregate."""

    label, directory, _colour = simulation
    canonical = next(
        name
        for name, archive in ALL_SIMULATION_ARCHIVES.items()
        if archive == directory
    )
    for profile, names in SIMULATION_PROFILES.items():
        if canonical not in names:
            continue
        donor_simulations = thermal_profile_simulations(profile)
        root = thermal_profile_cache_root(profile)
        table_path = root / "realised_state_summary.csv"
        blocks_path = root / "realised_state_blocks.csv"
        metadata_path = root / "realised_state_summary.metadata.json"
        if not table_path.is_file() or not metadata_path.is_file():
            continue
        if not skip_uncertainty and not blocks_path.is_file():
            continue
        expected = thermal_wind_manifest(
            "realised_state_summary",
            include_sources=validate_sources,
            simulations=donor_simulations,
        )
        if not cache_manifest_matches(
            read_json_file(metadata_path),
            expected,
            compare_sources=validate_sources,
        ):
            continue
        table = relabel_cached_simulations(pd.read_csv(table_path))
        table = table[table.simulation == label].copy()
        if table.empty:
            continue
        blocks = (
            pd.DataFrame()
            if skip_uncertainty
            else relabel_cached_simulations(pd.read_csv(blocks_path))
        )
        if not blocks.empty:
            blocks = blocks[blocks.simulation == label].copy()
        target_table, target_blocks, target_metadata = (
            realised_simulation_cache_paths(directory)
        )
        _atomic_csv(table, target_table)
        if not skip_uncertainty:
            _atomic_csv(blocks, target_blocks)
        manifest = thermal_wind_manifest(
            "realised_state_summary",
            directory=directory,
            simulations=(simulation,),
        )
        write_json_file(target_metadata, manifest)
        progress(
            f"{label}: migrated realised-state products from {profile}"
        )
        return True
    return False


def prepare_realised_state_summary(
    zonal_datasets: Sequence[xr.Dataset],
    *,
    figures_only: bool = False,
    rebuild_cache: bool = False,
    skip_uncertainty: bool = False,
) -> pd.DataFrame:
    table_path, metadata_path = realised_summary_cache_paths()
    blocks_path, uncertainty_path = realised_block_cache_paths()
    simulation_tables = []
    simulation_blocks = []
    block_days = BLOCK_LENGTH_DAYS["thermal_wind"]
    for index, (label, directory, _colour) in enumerate(SIMULATIONS):
        simulation = SIMULATIONS[index]
        sim_table_path, sim_blocks_path, sim_metadata_path = (
            realised_simulation_cache_paths(directory)
        )
        sim_manifest = thermal_wind_manifest(
            "realised_state_summary",
            directory=directory,
            include_sources=not figures_only,
            simulations=(simulation,),
        )
        current = (
            not rebuild_cache
            and sim_table_path.is_file()
            and (skip_uncertainty or sim_blocks_path.is_file())
            and cache_manifest_matches(
                read_json_file(sim_metadata_path),
                sim_manifest,
                compare_sources=not figures_only,
            )
        )
        if not current and not rebuild_cache:
            current = migrate_realised_simulation_cache(
                simulation,
                validate_sources=not figures_only,
                skip_uncertainty=skip_uncertainty,
            )
        if current:
            progress(f"{label}: using realised-state simulation cache")
            simulation_tables.append(relabel_cached_simulations(pd.read_csv(sim_table_path)))
            if not skip_uncertainty:
                simulation_blocks.append(relabel_cached_simulations(pd.read_csv(sim_blocks_path)))
            continue
        if figures_only:
            raise FileNotFoundError(
                "Figure-only mode requires a current realised-state cache: "
                f"{sim_table_path}"
            )
        progress(f"{label}: loading realised-state inputs")
        latlon = load_latlon_dataset(
            ROOT_DIR / directory,
            latlon_pattern=LATLON_PATTERN,
            chunks=THERMAL_TIME_CHUNKS,
        )
        zonal = load_latlon_dataset(
            ROOT_DIR / directory,
            latlon_pattern=ZONAL_MEAN_PATTERN,
            chunks=THERMAL_TIME_CHUNKS,
        )
        try:
            latlon = standardize_latlon_dataset(latlon, LATLON_VARIABLES)
            zonal = standardize_latlon_dataset(zonal, ZONAL_MEAN_VARIABLES)
            if TEST_TIME_COUNT is not None:
                latlon = latlon.isel(Time=slice(0, TEST_TIME_COUNT))
                zonal = zonal.isel(Time=slice(0, TEST_TIME_COUNT))
            sim_rows = [
                realised_state_row(label, latlon, zonal_datasets[index])
            ]
            sim_block_rows = []
            if not skip_uncertainty:
                latlon_blocks = time_block_slices(
                    latlon.olrtoa, block_days=block_days, complete=True
                )
                zonal_blocks = time_block_slices(
                    zonal.theta, block_days=block_days, complete=True
                )
                for block_index, (latlon_slice, zonal_slice) in enumerate(
                    zip(latlon_blocks, zonal_blocks, strict=True)
                ):
                    row = realised_state_row(
                        label,
                        latlon.isel(Time=latlon_slice),
                        zonal.isel(Time=zonal_slice),
                    )
                    row["block"] = block_index
                    row["block_days"] = block_days
                    sim_block_rows.append(row)
        finally:
            latlon.close()
            zonal.close()
        sim_table = pd.DataFrame(sim_rows)
        sim_blocks = pd.DataFrame(sim_block_rows)
        _atomic_csv(sim_table, sim_table_path)
        if not skip_uncertainty:
            _atomic_csv(sim_blocks, sim_blocks_path)
        write_json_file(
            sim_metadata_path,
            thermal_wind_manifest(
                "realised_state_summary",
                directory=directory,
                simulations=(simulation,),
            ),
        )
        simulation_tables.append(sim_table)
        if not skip_uncertainty:
            simulation_blocks.append(sim_blocks)

    table = add_control_differences(
        pd.concat(simulation_tables, ignore_index=True)
    )
    blocks = (
        pd.DataFrame()
        if skip_uncertainty
        else pd.concat(simulation_blocks, ignore_index=True)
    )
    uncertainty = (
        pd.DataFrame()
        if skip_uncertainty
        else realised_state_response_uncertainty(blocks)
    )
    if not figures_only:
        _atomic_csv(table, table_path)
        _atomic_csv(blocks, blocks_path)
        _atomic_csv(uncertainty, uncertainty_path)
        write_json_file(
            metadata_path,
            thermal_wind_manifest("realised_state_summary"),
        )
    return table


def cold_point_heights(zonal_datasets: Sequence[xr.Dataset]) -> list[np.ndarray]:
    """Calculate the time-mean cold-point height profile for each simulation."""

    profiles: list[np.ndarray] = []
    for dataset in zonal_datasets:
        time_mean = (
            dataset.mean("Time", skipna=True)
            if "Time" in dataset.dims
            else dataset
        )
        theta = hemispheric_fold(
            time_mean.theta,
            parity="even",
        )
        pressure = hemispheric_fold(
            time_mean.pressure,
            parity="even",
        )
        temperature = model_lat_field(
            theta * (pressure / P0_PA) ** (RD / CP)
        )
        pressure = model_lat_field(pressure)
        heights = []
        for latitude_index in range(dataset.sizes["latitude"]):
            _temperature, _pressure, height, _index = tropopause_profile(
                temperature.values[:, latitude_index],
                pressure.values[:, latitude_index],
                dataset.z.values,
            )
            heights.append(height)
        profiles.append(np.asarray(heights, dtype=float))
    return profiles


def add_equator_guides(axis, latitude: float = EQUATOR_MASK_DEGREES) -> None:
    """Mark the region where the geostrophic relation is not interpreted."""

    for value in (-latitude, latitude):
        axis.axvline(value, color="0.2", lw=0.9, ls="--", alpha=0.9)


def _balance_metrics(
    observed: xr.DataArray,
    implied: xr.DataArray,
    imbalance: xr.DataArray,
    *,
    scale: float,
) -> tuple[float, float, float, float]:
    return (
        rms_weighted_latheight(observed * scale),
        rms_weighted_latheight(implied * scale),
        rms_weighted_latheight(imbalance * scale),
        pattern_correlation_latheight(observed, implied),
    )


def plot_thermal_wind_shear_balance(
    zonal_datasets: Sequence[xr.Dataset],
    cold_points: Sequence[np.ndarray],
) -> tuple[plt.Figure, pd.DataFrame]:
    """Build Figure 1 and its within-simulation balance metrics."""

    figure, axes = plt.subplots(
        len(zonal_datasets),
        3,
        figsize=(8, 2 * len(zonal_datasets)),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    levels = symmetric_levels(8.0, ndiv=16)
    image = None
    rows = []
    for index, dataset in enumerate(zonal_datasets):
        temperature = latheight_time_mean_field(dataset, "temperature")
        wind = latheight_time_mean_field(dataset, "uReconstructZonal")
        plotted = thermal_wind_terms(
            temperature,
            wind,
            equator_mask_degrees=None,
        )
        metrics = thermal_wind_terms(
            temperature,
            wind,
            equator_mask_degrees=EQUATOR_MASK_DEGREES,
        )
        rows.append(
            (
                SIMULATIONS[index][0],
                *_balance_metrics(*metrics, scale=1000.0),
            )
        )
        for column, field in enumerate(plotted):
            image = axes[index, column].contourf(
                field.latitude,
                field.z / 1000.0,
                field * 1000.0,
                levels=levels,
                cmap=difference_cmap(levels),
                extend="both",
            )
            axes[index, column].plot(
                dataset.latitude,
                cold_points[index] / 1000.0,
                color="k",
                lw=1,
                ls=":",
            )
            add_equator_guides(axes[index, column])
            annotate_panel_label(
                axes[index, column],
                SIMULATIONS[index][0],
            )
            set_latheight_labels(
                axes[index, column],
                xlabel=index == len(zonal_datasets) - 1,
                ylabel=column == 0,
            )
            axes[index, column].set_xlim(-60, 60)
            axes[index, column].set_ylim(0, 25)
    axes[0, 0].set_title(r"Observed $\partial u/\partial z$")
    axes[0, 1].set_title("Thermal-wind implied")
    axes[0, 2].set_title("Observed - implied")
    add_symmetric_colorbar(
        figure,
        image,
        axes[:, 2],
        levels,
        label=r"m s$^{-1}$ km$^{-1}$",
    )
    metrics = pd.DataFrame(
        rows,
        columns=(
            "simulation",
            "rms_observed_shear_ms-1_km-1",
            "rms_implied_shear_ms-1_km-1",
            "rms_balance_imbalance_ms-1_km-1",
            "pattern_correlation_observed_implied",
        ),
    )
    return figure, metrics


def realised_state_row(
    simulation: str,
    latlon: xr.Dataset,
    zonal: xr.Dataset,
) -> dict[str, float | str]:
    """Return one realised-state summary row."""

    temperature = latheight_time_mean_field(zonal, "temperature")
    wind = latheight_time_mean_field(zonal, "uReconstructZonal")
    n2 = N2_latheight(zonal)
    precipitation = precip_rate_from_accum(total_precip_accum(latlon))
    return {
        "simulation": simulation,
        "T16km_K": float(
            deep_tropical_mean_latheight(
                temperature, height=TTL_REFERENCE_HEIGHT_M
            ).values
        ),
        "u16km_ms-1": float(
            deep_tropical_mean_latheight(
                wind, height=TTL_REFERENCE_HEIGHT_M
            ).values
        ),
        "T_TTL_K": float(
            deep_tropical_mean_latheight(
                temperature, height_bounds=TTL_Z_BOUNDS
            ).values
        ),
        "u_TTL_ms-1": float(
            deep_tropical_mean_latheight(
                wind, height_bounds=TTL_Z_BOUNDS
            ).values
        ),
        "N2_TTL_1e-4s-2": float(
            deep_tropical_mean_latheight(
                n2 * 1.0e4, height_bounds=TTL_Z_BOUNDS
            ).values
        ),
        "OLR_Wm-2": float(
            deep_tropical_mean_surface(latlon.olrtoa).mean("Time").values
        ),
        "PW_kgm-2": float(
            deep_tropical_mean_surface(latlon.precipw).mean("Time").values
        ),
        "precip_mmday-1": float(
            deep_tropical_mean_surface(precipitation).mean("Time").values
        ),
    }


def add_control_differences(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for column in tuple(table.columns[1:]):
        table[f"d{column}"] = table[column] - table.loc[0, column]
    return table


def realised_state_summary(
    latlon_datasets: Sequence[xr.Dataset],
    zonal_datasets: Sequence[xr.Dataset],
) -> pd.DataFrame:
    """Return absolute realised states and direct differences from control."""

    rows = [
        realised_state_row(SIMULATIONS[index][0], latlon, zonal)
        for index, (latlon, zonal) in enumerate(
            zip(latlon_datasets, zonal_datasets, strict=True)
        )
    ]
    return add_control_differences(pd.DataFrame(rows))


def realised_state_response_uncertainty(blocks: pd.DataFrame) -> pd.DataFrame:
    """Return independent-block intervals for selected realised-state metrics."""

    metrics = [
        column
        for column in blocks.columns
        if column not in {"simulation", "block", "block_days"}
    ]
    control = SIMULATIONS[0][0]
    perturbations = [simulation[0] for simulation in SIMULATIONS[1:]]
    comparisons = [
        (simulation, control) for simulation in perturbations
    ]
    if perturbations:
        comparisons.extend(
            (perturbations[0], simulation)
            for simulation in perturbations[1:]
        )
    rows = []
    for metric_index, metric in enumerate(metrics):
        for comparison_index, (simulation, reference) in enumerate(comparisons):
            simulation_values = blocks.loc[
                blocks.simulation == simulation, metric
            ].to_numpy(dtype=float)
            reference_values = blocks.loc[
                blocks.simulation == reference, metric
            ].to_numpy(dtype=float)
            result = independent_bootstrap_difference(
                simulation_values,
                reference_values,
                rng=np.random.default_rng(
                    BOOTSTRAP_SEED + 6000 + 10 * metric_index
                    + comparison_index
                ),
                n_resamples=BOOTSTRAP_REPLICATES,
            )
            rows.append(
                {
                    "metric": metric,
                    "simulation": simulation,
                    "reference": reference,
                    "mean_response": result["estimate"],
                    "ci_low": result["interval_low"],
                    "ci_high": result["interval_high"],
                    "sign_probability": result["sign_probability"],
                    "n_simulation_blocks": result["n_simulation"],
                    "n_reference_blocks": result["n_control"],
                    "block_days": BLOCK_LENGTH_DAYS["thermal_wind"],
                    "resampling": "independent complete time blocks by simulation",
                }
            )
    return pd.DataFrame(rows)


def plot_realised_state_summary(
    table: pd.DataFrame,
    uncertainty: pd.DataFrame | None = None,
) -> plt.Figure:
    """Build Figure 2 from direct experiment-minus-control differences."""

    specifications = (
        ("dT16km_K", r"$\Delta T_{16km}$ (K)"),
        ("du16km_ms-1", r"$\Delta u_{16km}$ (m s$^{-1}$)"),
        ("dT_TTL_K", r"$\Delta T_{TTL}$ (K)"),
        ("du_TTL_ms-1", r"$\Delta u_{TTL}$ (m s$^{-1}$)"),
        (
            "dN2_TTL_1e-4s-2",
            r"$\Delta N^2_{TTL}$ ($10^{-4}$ s$^{-2}$)",
        ),
        ("dOLR_Wm-2", r"$\Delta$ OLR (W m$^{-2}$)"),
        ("dPW_kgm-2", r"$\Delta$ PW (kg m$^{-2}$)"),
        ("dprecip_mmday-1", r"$\Delta$ precipitation (mm day$^{-1}$)"),
    )
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(11, 5.2),
        constrained_layout=True,
    )
    positions = np.arange(len(SIMULATIONS))
    labels = [simulation[0] for simulation in SIMULATIONS]
    colours = [simulation[2] for simulation in SIMULATIONS]
    for axis, (column, label) in zip(
        axes.ravel(),
        specifications,
        strict=True,
    ):
        axis.bar(positions, table[column], color=colours)
        if uncertainty is not None and not uncertainty.empty:
            metric = column[1:]
            selected = uncertainty[
                (uncertainty.metric == metric)
                & (uncertainty.reference == SIMULATIONS[0][0])
            ].set_index("simulation")
            for position, simulation in enumerate(labels[1:], start=1):
                if simulation in selected.index:
                    interval = selected.loc[simulation]
                    axis.vlines(
                        position,
                        interval.ci_low,
                        interval.ci_high,
                        color="0.15",
                        linewidth=1.3,
                    )
        axis.axhline(0, color="0.5", lw=0.6)
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Realised mean-state differences from control")
    return figure


def plot_temperature_gradient_balance(
    zonal_datasets: Sequence[xr.Dataset],
    cold_points: Sequence[np.ndarray],
) -> tuple[plt.Figure, pd.DataFrame]:
    """Build Figure 13 and its within-simulation balance metrics."""

    figure, axes = plt.subplots(
        len(zonal_datasets),
        3,
        figsize=(8, 2 * len(zonal_datasets)),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    levels = symmetric_levels(8.0, ndiv=16)
    image = None
    rows = []
    for index, dataset in enumerate(zonal_datasets):
        temperature = latheight_time_mean_field(dataset, "temperature")
        wind = latheight_time_mean_field(dataset, "uReconstructZonal")
        plotted = thermal_wind_temperature_gradient_terms(
            temperature,
            wind,
            equator_mask_degrees=None,
        )
        metrics = thermal_wind_temperature_gradient_terms(
            temperature,
            wind,
            equator_mask_degrees=EQUATOR_MASK_DEGREES,
        )
        rows.append(
            (
                SIMULATIONS[index][0],
                *_balance_metrics(*metrics, scale=1.0e6),
            )
        )
        for column, field in enumerate(plotted):
            image = axes[index, column].contourf(
                field.latitude,
                field.z / 1000.0,
                field * 1.0e6,
                levels=levels,
                cmap=difference_cmap(levels),
                extend="both",
            )
            axes[index, column].plot(
                dataset.latitude,
                cold_points[index] / 1000.0,
                color="k",
                lw=1,
                ls=":",
            )
            add_equator_guides(axes[index, column])
            annotate_panel_label(
                axes[index, column],
                SIMULATIONS[index][0],
            )
            set_latheight_labels(
                axes[index, column],
                xlabel=index == len(zonal_datasets) - 1,
                ylabel=column == 0,
            )
            axes[index, column].set_xlim(-60, 60)
            axes[index, column].set_ylim(0, 25)
    axes[0, 0].set_title(r"Observed $\partial T/\partial y$")
    axes[0, 1].set_title("Shear-implied gradient")
    axes[0, 2].set_title("Observed - implied")
    add_symmetric_colorbar(
        figure,
        image,
        axes[:, 2],
        levels,
        label=r"K $(10^3$ km)$^{-1}$",
    )
    metrics = pd.DataFrame(
        rows,
        columns=(
            "simulation",
            "rms_observed_dTdy_K_per_1000km",
            "rms_implied_dTdy_K_per_1000km",
            "rms_balance_imbalance_K_per_1000km",
            "pattern_correlation_observed_implied",
        ),
    )
    return figure, metrics


def gradient_shear_fields(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """Return observed temperature-gradient and shear fields."""

    temperature = latheight_time_mean_field(dataset, "temperature")
    wind = latheight_time_mean_field(dataset, "uReconstructZonal")
    shear, _implied_shear, _imbalance = thermal_wind_terms(
        temperature,
        wind,
        equator_mask_degrees=None,
    )
    gradient, _implied_gradient, _imbalance = (
        thermal_wind_temperature_gradient_terms(
            temperature,
            wind,
            equator_mask_degrees=None,
        )
    )
    return {
        "dT_dy": (gradient * 1.0e6).rename("dT_dy_K_per_1000km"),
        "du_dz": (shear * 1000.0).rename("du_dz_ms-1_per_km"),
    }


GRADIENT_SHEAR_SPECIFICATIONS = (
    (
        "dT_dy",
        r"$\partial T/\partial y$",
        r"K $(10^3$ km)$^{-1}$",
        "coolwarm",
        8.0,
        2.0,
    ),
    (
        "du_dz",
        r"$\partial u/\partial z$",
        r"m s$^{-1}$ km$^{-1}$",
        "bwr",
        8.0,
        2.0,
    ),
)


def plot_gradient_shear_total_fields(
    zonal_datasets: Sequence[xr.Dataset],
    cold_points: Sequence[np.ndarray],
    fields: Sequence[dict[str, xr.DataArray]] | None = None,
) -> plt.Figure:
    """Build the absolute gradient and shear companion to Figure 14."""

    if fields is None:
        fields = [gradient_shear_fields(dataset) for dataset in zonal_datasets]
    figure, axes = plt.subplots(
        len(GRADIENT_SHEAR_SPECIFICATIONS),
        len(SIMULATIONS),
        figsize=(3.8 * len(SIMULATIONS), 2.9 * len(GRADIENT_SHEAR_SPECIFICATIONS)),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row, (
        key,
        label,
        units,
        colourmap,
        raw_fallback,
        _difference_fallback,
    ) in enumerate(GRADIENT_SHEAR_SPECIFICATIONS):
        raw_levels = symmetric_percentile_levels_for_fields(
            [simulation_fields[key] for simulation_fields in fields],
            percentile=99.0,
            ndiv=12,
            zero_field_scale=raw_fallback,
        )
        image = None
        for column, simulation_fields in enumerate(fields):
            plotted = simulation_fields[key]
            image = axes[row, column].contourf(
                plotted.latitude,
                plotted.z / 1000.0,
                plotted,
                levels=raw_levels,
                cmap=colourmap,
                extend="both",
            )
            axes[row, column].plot(
                zonal_datasets[column].latitude,
                cold_points[column] / 1000.0,
                color="k",
                linewidth=1.0,
                linestyle=":",
            )
            add_equator_guides(axes[row, column])
            annotate_panel_label(axes[row, column], SIMULATIONS[column][0])
            set_latheight_labels(
                axes[row, column],
                xlabel=row == len(GRADIENT_SHEAR_SPECIFICATIONS) - 1,
                ylabel=column == 0,
            )
            axes[row, column].set_xlim(-60.0, 60.0)
            axes[row, column].set_ylim(0.0, 25.0)
            if column == 0:
                axes[row, column].set_ylabel(f"{label}\nHeight (km)")
        figure.colorbar(
            image,
            ax=axes[row, :],
            location="right",
            label=f"{label} ({units})",
        )
    figure.suptitle("Observed thermal gradient and vertical shear")
    return figure


def plot_gradient_shear_differences(
    zonal_datasets: Sequence[xr.Dataset],
    cold_points: Sequence[np.ndarray],
    fields: Sequence[dict[str, xr.DataArray]] | None = None,
) -> plt.Figure:
    """Build Figure 14 using direct nudged-minus-control differences."""

    if fields is None:
        fields = [gradient_shear_fields(dataset) for dataset in zonal_datasets]
    figure, axes = plt.subplots(
        len(GRADIENT_SHEAR_SPECIFICATIONS),
        len(SIMULATIONS),
        figsize=(3.8 * len(SIMULATIONS), 2.9 * len(GRADIENT_SHEAR_SPECIFICATIONS)),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row, (
        key,
        label,
        units,
        colourmap,
        raw_fallback,
        difference_fallback,
    ) in enumerate(GRADIENT_SHEAR_SPECIFICATIONS):
        control = fields[0][key]
        differences = [
            xr.align(fields[index][key], control, join="exact")[0]
            - xr.align(fields[index][key], control, join="exact")[1]
            for index in range(1, len(SIMULATIONS))
        ]
        raw_levels = symmetric_percentile_levels_for_fields(
            [control],
            percentile=99.0,
            ndiv=12,
            zero_field_scale=raw_fallback,
        )
        difference_levels = symmetric_percentile_levels_for_fields(
            differences,
            percentile=99.0,
            ndiv=12,
            zero_field_scale=difference_fallback,
        )
        control_image = difference_image = None
        for column in range(len(SIMULATIONS)):
            axis = axes[row, column]
            if column == 0:
                plotted = control
                control_image = axis.contourf(
                    plotted.latitude,
                    plotted.z / 1000.0,
                    plotted,
                    levels=raw_levels,
                    cmap=colourmap,
                    extend="both",
                )
            else:
                plotted = differences[column - 1]
                difference_image = axis.contourf(
                    plotted.latitude,
                    plotted.z / 1000.0,
                    plotted,
                    levels=difference_levels,
                    cmap=difference_cmap(difference_levels),
                    extend="both",
                )
            axis.plot(
                zonal_datasets[column].latitude,
                cold_points[column] / 1000.0,
                color="k",
                lw=1,
                ls=":",
            )
            add_equator_guides(axis)
            annotate_panel_label(
                axis,
                control_diff_label(
                    [simulation[0] for simulation in SIMULATIONS],
                    column,
                ),
            )
            set_latheight_labels(
                axis,
                xlabel=row == len(GRADIENT_SHEAR_SPECIFICATIONS) - 1,
                ylabel=column == 0,
            )
            axis.set_xlim(-60, 60)
            axis.set_ylim(0, 25)
            if column == 0:
                axis.set_ylabel(f"{label}\nHeight (km)")
        figure.colorbar(
            control_image,
            ax=axes[row, 0],
            location="right",
            label=f"{label} ({units})",
        )
        add_symmetric_colorbar(
            figure,
            difference_image,
            axes[row, 1:],
            difference_levels,
            label=f"{label} difference ({units})",
        )
    figure.suptitle("Observed thermal gradient and vertical shear")
    return figure


def parse_command_line(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose thermal-wind balance and direct realised-state "
            "differences without scalar thermal attribution."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use bounded actual-data subsets and isolated test products.",
    )
    parser.add_argument("--test-times", type=int, default=8)
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=FIGURES,
        default=None,
        help="Figures to regenerate; omit to build all retained figures.",
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--figures-only",
        action="store_true",
        help="Build figures only from current compact thermal-wind caches.",
    )
    cache_mode.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild compact thermal-wind caches before plotting.",
    )
    args = parser.parse_args(argv)
    if args.figures:
        args.figures = list(dict.fromkeys(args.figures))
    if args.test_times < 3:
        parser.error("--test-times must be at least 3")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    global SAVE_LOCATION, TEST_TIME_COUNT, THERMAL_WIND_CACHE_ROOT
    global THERMAL_WIND_PROFILE_CACHE_ROOT
    args = parse_command_line(argv)
    if args.test:
        TEST_TIME_COUNT = int(args.test_times)
        test_label = f"all_{TEST_TIME_COUNT:03d}times"
        THERMAL_WIND_CACHE_ROOT = diagnostic_cache_dir(
            "thermal_wind", "testing", test_label
        )
        THERMAL_WIND_PROFILE_CACHE_ROOT = diagnostic_profile_cache_dir(
            "thermal_wind", "testing", test_label
        )
        SAVE_LOCATION = os.path.join(SAVE_LOCATION, "testing", test_label)
    figures = set(args.figures or FIGURES)
    progress(
        "Starting thermal-wind diagnostics "
        f"(figures={','.join(str(number) for number in sorted(figures))})"
    )
    zonal_datasets = prepare_zonal_mean_caches(
        figures_only=args.figures_only,
        rebuild_cache=args.rebuild_cache,
    )
    saver = FigureSaver(
        SAVE_LOCATION,
        enabled=True,
        figures_to_run=figures,
    )
    cold_points = (
        cold_point_heights(zonal_datasets)
        if figures.intersection({1, 13, 14})
        else []
    )
    try:
        if 1 in figures:
            progress("Starting Figure 1: thermal-wind shear balance")
            figure, metrics = plot_thermal_wind_shear_balance(
                zonal_datasets,
                cold_points,
            )
            saver.save(figure, "thermal_wind_total_fields", number=1)
            plt.close(figure)
            metrics.to_csv(
                os.path.join(
                    saver.save_loc,
                    "thermal_wind_total_metrics.csv",
                ),
                index=False,
            )
        if 2 in figures:
            progress("Starting Figure 2: realised-state differences")
            summary = prepare_realised_state_summary(
                zonal_datasets,
                figures_only=args.figures_only,
                rebuild_cache=args.rebuild_cache,
                skip_uncertainty=args.test,
            )
            summary.to_csv(
                os.path.join(saver.save_loc, "realised_state_summary.csv"),
                index=False,
            )
            _blocks_path, uncertainty_path = realised_block_cache_paths()
            try:
                uncertainty = relabel_cached_simulations(pd.read_csv(uncertainty_path))
            except pd.errors.EmptyDataError:
                uncertainty = pd.DataFrame()
            uncertainty.to_csv(
                os.path.join(
                    saver.save_loc,
                    "realised_state_response_uncertainty.csv",
                ),
                index=False,
            )
            figure = plot_realised_state_summary(summary, uncertainty)
            saver.save(figure, "realised_state_summary_bars", number=2)
            plt.close(figure)
        if 13 in figures:
            progress(
                "Starting Figure 13: thermal-wind temperature-gradient balance"
            )
            figure, metrics = plot_temperature_gradient_balance(
                zonal_datasets,
                cold_points,
            )
            saver.save(
                figure,
                "thermal_wind_temperature_gradient_total_fields",
                number=13,
            )
            plt.close(figure)
            metrics.to_csv(
                os.path.join(
                    saver.save_loc,
                    "thermal_wind_temperature_gradient_total_metrics.csv",
                ),
                index=False,
            )
        if 14 in figures:
            progress(
                "Starting Figure 14: absolute and response gradient/shear fields"
            )
            gradient_shear = [
                gradient_shear_fields(dataset) for dataset in zonal_datasets
            ]
            figure = plot_gradient_shear_total_fields(
                zonal_datasets,
                cold_points,
                fields=gradient_shear,
            )
            saver.save(
                figure,
                "thermal_wind_gradient_shear_latheight",
                number=14,
            )
            plt.close(figure)
            figure = plot_gradient_shear_differences(
                zonal_datasets,
                cold_points,
                fields=gradient_shear,
            )
            saver.save(
                figure,
                "thermal_wind_gradient_shear_latheight_diff",
                number=14,
            )
            plt.close(figure)
    finally:
        for dataset in zonal_datasets:
            dataset.close()
    progress("Thermal-wind diagnostics complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
