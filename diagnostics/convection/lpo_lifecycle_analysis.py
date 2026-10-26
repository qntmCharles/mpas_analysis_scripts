#!/usr/bin/env python3
"""Analyse cloud, environment and wave lifecycles within tracked LPOs.

``lpo_tracking.py`` is the sole producer of LPO identities, masks, tracks,
precipitation diagnostics and lifecycle reference times.  This script consumes
that stable interface and owns every downstream lifecycle calculation:

* instantaneous cloud populations and cloud depth inside each saved envelope;
* absolute environmental sampling and experiment-minus-control responses;
* native-level vertical cloud-fraction profiles;
* Kelvin, mixed Rossby-gravity, equatorial-Rossby, TD-type and
  eastward-intraseasonal fields sampled along the saved LPO path;
* event summaries, independent-event composites and lifecycle figure selectors.

Local model means and vertical velocity statistics use envelope pixels with
concurrent unsmoothed precipitation above 1 mm/day. Cloud statistics,
condensate integrals, rainfall totals and native cloud profiles use the
complete envelope. Unfiltered surface and thermodynamic sections use the
same rain mask. Cloud and wave maps retain their surrounding context.

No tracking is performed here and no upstream-LPT internal file is read.
Lifecycle tables, profiles and composite datasets are written beside the
saved LPO products in the fixed data repository. Only rendered figures are
written below the usual ``~/tmp`` figure tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import dask
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.lines import Line2D
from scipy import ndimage

_THIS_DIR = Path(__file__).resolve().parent
for _parent in (_THIS_DIR.parent, _THIS_DIR.parent.parent, _THIS_DIR.parent.parent.parent):
    if str(_parent) not in sys.path:
        sys.path.append(str(_parent))

from diagnostics.diagnostic_helpers import (
    HIGH_CLOUD_CLASS_SPECS,
    SIMULATION_COLOURS,
)
from diagnostics.convection.lpo_tracking import (
    DEFAULT_ROOT,
    DEFAULT_TEST_DAYS,
    DEFAULT_TEST_MINIMUM_DURATION_DAYS,
    LIFECYCLE_LAGS_HOURS,
    MINIMUM_TEST_DAYS,
    MPAS_LPT_GRID_SPACING_DEGREES,
    PRODUCT_VERSION,
    SIMULATION_DIRS,
    SIMULATION_LABELS,
    VALIDATION_RUN_NAMES,
    _event_product_path,
    _area_sum,
    _load_classification,
    _load_simulation_dataset,
    _read_standard_masks,
    _require_mapping_fields,
    actual_test_output_directory,
    diagnose_envelope_steps,
    lpo_tracking_figure_root,
    lpo_tracking_output_root,
    require_actual_test_products,
    require_publication_products,
)
from diagnostics.diagnostic_helpers import (
    FigureSaver,
    LATLON_PATTERN,
    SPECIFIC_HUMIDITY_CMAP,
    blue_sky_white_cloud_cmap,
    difference_cmap,
    grid_cell_area_km2,
    native_layer_interfaces,
    native_midlevel_height,
    netcdf_safe_dataset,
    precip_rate_cmap_norm,
    precipitation_rate_from_dataset,
    pressure_to_pa,
    progress,
    source_file_signature,
    symmetric_levels,
    vertical_dim,
    write_csv_atomic,
    write_netcdf_atomic,
)
from diagnostics.analysis_config import (
    ACTIVE_SIMULATION_PROFILE,
    DEFAULT_SIMULATION_PROFILE,
    SIMULATION_LABELS as CONFIGURED_SIMULATION_LABELS,
    cache_directory,
)
from diagnostics.waves.cached_wave_products import (
    filtered_wave_cache_path,
    load_existing_filtered_wave,
    load_existing_kelvin_dispersion,
    load_existing_lpo_wave_sections,
    lpo_wave_section_cache_path,
)
from diagnostics.waves.equatorial_wave_config import (
    LPO_LIFECYCLE_FILTERS,
    WAVE_BOXES,
    WAVE_CONFIG_VERSION,
)
from diagnostics.waves.wave_metrics import (
    blockwise_buoyancy_frequency_squared,
    buoyancy_frequency_squared,
    sample_masked_field,
)
from diagnostics.ttl import (
    TTL_BASE_PRESSURE_HPA,
    TTL_HEIGHT_BOUNDS_KM,
    TTL_PRESSURE_BOUNDS_HPA,
    TTL_REFERENCE,
    TTL_TOP_PRESSURE_HPA,
)
from diagnostics.convection.lpo_lifecycle_sections import (
    CLOUD_CLASS_CODES,
    CLOUD_CLASS_NAMES,
    COMPOSITE_CACHE_VERSION,
    DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES,
    LIFECYCLE_RAIN_THRESHOLD_MM_DAY,
    LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES,
    PHASE_ALIGNMENT_WINDOWS_HOURS,
    SECTION_CACHE_VERSION,
    SURFACE_VARIABLES,
    VERTICAL_VARIABLES,
    _default_window_event_cube,
    _event_cube,
    build_review_response_composites,
    build_simulation_composites,
    build_unfiltered_event_sections,
    combine_simulation_composites,
    event_equal_statistics,
    independent_event_contrast,
)


LIFECYCLE_ANALYSIS_VERSION = (
    "lpo-lifecycle-analysis-v12-rain-conditioned"
)
# Intensive local model statistics use raining columns. Extensive quantities
# and cloud population statistics retain the complete envelope.
ENVELOPE_MODEL_FIELDS = {"high_cloud_ice_mass", "high_cloud_snow_mass"}
LOCAL_SAMPLING_DEFINITION = (
    "local model means and vertical velocity statistics: LPO envelope with "
    "concurrent unsmoothed precipitation > 1 mm/day; cloud statistics, "
    "condensate integrals and rainfall totals: complete LPO envelope; "
    "unfiltered surface and thermodynamic sections use the same rain mask"
)
LIFECYCLE_FIGURES = (*range(4, 46), "45a", *range(46, 54))
WAVE_PHASE_PROFILE_FIGURE = 51
WAVE_COUPLING_LIFECYCLE_FIGURE = 52
WAVE_COUPLING_RESPONSE_FIGURE = 53
LIFECYCLE_STEP_FILENAME = "lpo_lifecycle_steps.csv"
LIFECYCLE_SUMMARY_FILENAME = "lpo_lifecycle_event_summary.csv"
LIFECYCLE_PROFILE_FILENAME = "lpo_lifecycle_vertical_profiles.csv"
LIFECYCLE_CONFIGURATION_FILENAME = "lpo_lifecycle_configuration.json"
LIFECYCLE_PROFILE_SUFFIX = (
    ""
    if ACTIVE_SIMULATION_PROFILE == DEFAULT_SIMULATION_PROFILE
    else f".{ACTIVE_SIMULATION_PROFILE}"
)
LIFECYCLE_COMPOSITE_FILENAME = (
    f"lpo_lifecycle_event_equal_composites{LIFECYCLE_PROFILE_SUFFIX}.nc"
)
LIFECYCLE_SECTION_FILENAME = "lpo_lifecycle_event_sections.nc"
LIFECYCLE_2D_COMPOSITE_FILENAME = (
    f"lpo_lifecycle_2d_composites{LIFECYCLE_PROFILE_SUFFIX}.nc"
)
LIFECYCLE_EVENT_CACHE_DIRECTORY = "lpo_lifecycle_event_cache"
LIFECYCLE_BASELINE_TIME_BATCH_SIZE = 32
LIFECYCLE_VERTICAL_BASELINE_TIME_BATCH_SIZE = 8
TEST_EVENTS_PER_SIMULATION = 2
TEST_MODE = False
DEEP_CLOUD_TOP_PERCENTILE = 10
TIMING_BOOTSTRAP_RESAMPLES = 1000
TIMING_BOOTSTRAP_SEED = 20260804
# Convert km2 mm to km3 for figure display.
RAIN_VOLUME_DISPLAY_SCALE = 1.0e-6
RAIN_VOLUME_LABEL = r"Integrated rain volume (km$^3$)"
RAIN_VOLUME_TITLE = "Integrated rain volume"
RAIN_VOLUME_UNITS = r"km$^3$"
LIFECYCLE_LAG_LABEL = "Lag from precipitation peak (h)"
NEAR_PEAK_LAGS_HOURS = (-12, -6, 0, 6, 12)
SPATIAL_STAGE_WINDOWS = (
    ("Pre-peak", (-42, -36, -30, -24, -18)),
    ("Near-peak", NEAR_PEAK_LAGS_HOURS),
    ("Post-peak", (18, 24, 30, 36, 42)),
)

SIMULATION_ROWS = tuple(SIMULATION_DIRS)
RESPONSE_ROWS = tuple(
    simulation for simulation in SIMULATION_ROWS if simulation != "control"
)
WAVE_SIMULATION_LABELS = {
    simulation: CONFIGURED_SIMULATION_LABELS[simulation]
    for simulation in SIMULATION_ROWS
}
WAVE_FIELD_DEFINITIONS = {
    "Precip": ("precip_mm_day", "mm day-1"),
    "PW": ("pw_kg_m2", "kg m-2"),
    "OLR": ("olr_w_m2", "W m-2"),
    "T100": ("temperature_100hpa_K", "K"),
    "u100": ("zonal_wind_100hpa_m_s", "m s-1"),
}
HIGH_CLOUD_CLASS_COLOURS = {
    code: colour for code, _label, colour in HIGH_CLOUD_CLASS_SPECS
}


def _wave_key(wave_name: str) -> str:
    return {
        "Eastward_intraseasonal": "eastward_intraseasonal",
        "Kelvin": "kelvin",
        "MRG": "mixed_rossby_gravity",
        "ER": "equatorial_rossby",
        "Easterly": "td_type",
    }[wave_name]


WAVE_LIFECYCLE_FIELDS = {
    (wave_name, cache_field): (
        f"{_wave_key(wave_name)}_filtered_{suffix}"
    )
    for wave_name in LPO_LIFECYCLE_FILTERS
    for cache_field, (suffix, _units) in WAVE_FIELD_DEFINITIONS.items()
}
LIFECYCLE_COMPOSITE_VARIABLES = (
    "area_integrated_24h_precip_km2_mm",
    "envelope_area_km2",
    "raining_area_fraction_of_envelope",
    "total_high_cloud_area_km2",
    "active_deep_convective_area_km2",
    "stratiform_precipitating_high_cloud_area_km2",
    "nonprecipitating_high_cloud_area_km2",
    "median_cloud_top_pressure_hpa",
    "p10_cloud_top_pressure_hpa",
    "ttl_reaching_high_cloud_fraction_of_envelope",
    "olr_w_m2",
    "temperature_100hpa",
    "temperature_100hpa_event_anomaly",
    "temperature_100_200hpa",
    "temperature_100_200hpa_event_anomaly",
    "layer_mean_n2_ttl",
    "layer_mean_n2_ttl_event_anomaly",
    "zonal_wind_100hpa",
    "zonal_wind_100hpa_event_anomaly",
    "utls_shear_u100_minus_u250",
    "utls_shear_u100_minus_u250_event_anomaly",
    *WAVE_LIFECYCLE_FIELDS.values(),
)


def label_count_periodic(
    mask: np.ndarray, *, periodic_longitude: bool = True
) -> int:
    """Count 8-connected clusters, optionally merging the longitude seam."""

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return 0
    if not periodic_longitude:
        return int(count)
    parent = np.arange(count + 1)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        if left and right:
            parent[find(right)] = find(left)

    for row in range(mask.shape[0]):
        for offset in (-1, 0, 1):
            other = (row + offset) % mask.shape[0]
            if 0 <= row + offset < mask.shape[0]:
                union(labels[row, 0], labels[other, -1])
    return len({find(value) for value in range(1, count + 1)})


CLASS_MASKS = {
    "active_deep_convective": "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud": "stratiform_precipitating_high_cloud_mask",
    "nonprecipitating_high_cloud": "nonprecipitating_high_cloud_mask",
    "total_high_cloud": "high_cloud_mask",
}

REQUIRED_CLASSIFICATION_FIELDS = (
    *CLASS_MASKS.values(),
    "cloud_top_height",
    "cloud_top_pressure",
)

REQUIRED_EVENT_MODEL_FIELDS = (
    "olr",
    "w",
    "high_cloud_ice_mass",
    "high_cloud_snow_mass",
    "temperature_100hpa",
    "temperature_100hpa_event_anomaly",
    "temperature_100_200hpa",
    "temperature_100_200hpa_event_anomaly",
    "utls_temperature_anomaly",
    "tropopause_static_stability",
    "tropopause_static_stability_event_anomaly",
    "zonal_wind_100hpa",
    "zonal_wind_100hpa_event_anomaly",
    "zonal_wind_250hpa",
    "zonal_wind_250hpa_event_anomaly",
    "upper_tropospheric_zonal_wind",
    "upper_tropospheric_zonal_wind_event_anomaly",
    "utls_shear_u100_minus_u250",
    "utls_shear_u100_minus_u250_event_anomaly",
    "deep_layer_shear_u200_minus_u850",
    "deep_layer_shear_u200_minus_u850_event_anomaly",
    "layer_mean_n2_ttl",
    "layer_mean_n2_ttl_event_anomaly",
    "cold_point_tropopause_pressure_hpa",
    "cold_point_tropopause_pressure_hpa_event_anomaly",
    "cold_point_tropopause_height_m",
    "cold_point_tropopause_height_m_event_anomaly",
    "precipitable_water_kg_m2",
    "precipitable_water_kg_m2_event_anomaly",
    "olr_event_anomaly",
)


def diagnose_event_steps(
    event_mask: xr.DataArray,
    precipitation_mm_day: xr.DataArray,
    classification: Mapping[str, xr.DataArray],
    model_fields: Mapping[str, xr.DataArray],
    *,
    event_id: str,
    simulation: str,
    spatial_buffer_degrees: float | None = 0.0,
) -> pd.DataFrame:
    """Diagnose instantaneous cloud populations inside one tracked envelope."""

    precipitation, event_mask = xr.align(
        precipitation_mm_day, event_mask.astype(bool), join="inner"
    )
    if precipitation.sizes.get("Time", 0) == 0:
        raise RuntimeError(
            f"{simulation}/{event_id}: LPT mask and precipitation have no "
            "common times"
        )
    periodic_longitude = True
    if spatial_buffer_degrees is not None:
        support = np.asarray(event_mask.any("Time").values, dtype=bool)
        latitude_support = np.flatnonzero(support.any(axis=1))
        longitude_support = np.flatnonzero(support.any(axis=0))
        if latitude_support.size and longitude_support.size:
            latitude_step = abs(
                float(np.median(np.diff(event_mask.latitude.values)))
            )
            longitude_step = abs(
                float(np.median(np.diff(event_mask.longitude.values)))
            )
            latitude_buffer = int(
                np.ceil(spatial_buffer_degrees / latitude_step)
            )
            longitude_buffer = int(
                np.ceil(spatial_buffer_degrees / longitude_step)
            )
            latitude_indices = np.arange(
                max(0, latitude_support.min() - latitude_buffer),
                min(
                    event_mask.sizes["latitude"],
                    latitude_support.max() + latitude_buffer + 1,
                ),
            )
            longitude_count = event_mask.sizes["longitude"]
            crosses_periodic_seam = (
                longitude_support[0] == 0
                and longitude_support[-1] == longitude_count - 1
            )
            if crosses_periodic_seam:
                # Retaining the complete longitude axis preserves both
                # spherical cell areas and true periodic adjacency.
                longitude_indices = np.arange(longitude_count)
            else:
                longitude_indices = np.arange(
                    max(0, longitude_support.min() - longitude_buffer),
                    min(
                        longitude_count,
                        longitude_support.max() + longitude_buffer + 1,
                    ),
                )
            periodic_longitude = longitude_indices.size == longitude_count
            event_mask = event_mask.isel(
                latitude=latitude_indices,
                longitude=longitude_indices,
            )
            precipitation = precipitation.isel(
                latitude=latitude_indices,
                longitude=longitude_indices,
            )
    envelope_steps = diagnose_envelope_steps(
        event_mask,
        precipitation,
        event_id=event_id,
        simulation=simulation,
    )
    envelope_rows = {
        pd.Timestamp(row["Time"]): row
        for row in envelope_steps.to_dict(orient="records")
    }
    _require_mapping_fields(
        classification,
        REQUIRED_CLASSIFICATION_FIELDS,
        product="shared high-cloud classification",
    )
    _require_mapping_fields(
        model_fields,
        REQUIRED_EVENT_MODEL_FIELDS,
        product="projected MPAS lifecycle diagnostics",
    )

    def load_event_horizontal_fields(
        source: Mapping[str, xr.DataArray],
        *,
        source_label: str,
    ) -> dict[str, xr.DataArray]:
        """Load each reusable event slice once instead of once per track step."""

        loaded: dict[str, xr.DataArray] = {}
        allowed_dimensions = {"Time", "latitude", "longitude"}
        for name, field in source.items():
            if not {"latitude", "longitude"}.issubset(field.dims):
                continue
            if set(field.dims) - allowed_dimensions:
                continue
            indexers = {
                dimension: precipitation[dimension]
                for dimension in allowed_dimensions
                if dimension in field.dims
            }
            # Pressure-level extraction retains scalar ``pres`` coordinates
            # (for example 100 and 250 hPa). They are useful provenance on an
            # individual field but conflict when several fields are combined
            # for one shared Dask load.
            loaded[name] = field.sel(indexers).reset_coords(drop=True)
        if not loaded:
            return loaded
        progress(
            f"{simulation}/{event_id}: loading {len(loaded)} "
            f"{source_label} fields for {precipitation.sizes['Time']} times, "
            f"{precipitation.sizes['latitude']} latitudes, and "
            f"{precipitation.sizes['longitude']} longitudes"
        )
        loaded_dataset = xr.Dataset(loaded).load()
        progress(
            f"{simulation}/{event_id}: loaded {source_label} event fields"
        )
        return {
            name: loaded_dataset[name]
            for name in loaded
        }

    # Several diagnostics include lazy time/zonal-mean anomalies. Without
    # this event-level materialisation those means are recomputed separately
    # for every six-hour track step, making even a short smoke test needlessly
    # expensive.
    classification = load_event_horizontal_fields(
        classification,
        source_label="classification",
    )
    model_fields = load_event_horizontal_fields(
        model_fields,
        source_label="model",
    )
    area = grid_cell_area_km2(
        precipitation.latitude.values, precipitation.longitude.values
    )
    area = xr.DataArray(
        area,
        dims=("latitude", "longitude"),
        coords={
            "latitude": precipitation.latitude,
            "longitude": precipitation.longitude,
        },
    )
    rows: list[dict] = []
    for time_value in pd.to_datetime(event_mask.Time.values):
        mask = event_mask.sel(Time=time_value)
        precip = precipitation.sel(Time=time_value)
        rain_mask = mask & (precip > LIFECYCLE_RAIN_THRESHOLD_MM_DAY)
        row = dict(envelope_rows[pd.Timestamp(time_value)])
        envelope_area = row["envelope_area_km2"]
        row["raining_area_fraction_of_envelope"] = (
            _area_sum(rain_mask, area) / envelope_area
            if envelope_area > 0 else np.nan
        )
        precip_volume = row["area_integrated_precip_km2_mm_day"]
        for short_name, variable_name in CLASS_MASKS.items():
            class_mask = classification[variable_name].sel(Time=time_value) & mask
            row[f"{short_name}_area_km2"] = _area_sum(class_mask, area)
            if short_name == "active_deep_convective":
                row["active_cluster_count"] = label_count_periodic(
                    np.asarray(class_mask.values, dtype=bool),
                    periodic_longitude=periodic_longitude,
                )
        conv_mask = classification[
            "convective_precipitating_high_cloud_mask"
        ].sel(Time=time_value) & mask
        row["convective_precip_area_km2"] = _area_sum(
            conv_mask & (precip > 0), area
        )
        row["convective_precip_volume_km2_mm_day"] = float(
            (precip.where(conv_mask, 0.0) * area).sum()
        )
        stratiform = classification[
            "stratiform_precipitating_high_cloud_mask"
        ]
        row["stratiform_precip_fraction"] = (
            float((precip.where(stratiform.sel(Time=time_value) & mask, 0.0) * area).sum())
            / precip_volume
            if precip_volume > 0
            else np.nan
        )
        high_mask = classification["high_cloud_mask"].sel(
            Time=time_value
        ) & mask
        cloud_top_height = classification["cloud_top_height"].sel(
            Time=time_value
        )
        row["mean_cloud_top_height"] = sample_masked_field(
            cloud_top_height, mask, area
        )["mean"]
        row["maximum_cloud_top_height"] = float(
            cloud_top_height.where(mask).max(skipna=True)
        )
        cloud_top_pressure = classification["cloud_top_pressure"].sel(
            Time=time_value
        )
        pressure_statistics = sample_masked_field(
            cloud_top_pressure,
            mask,
            area,
            quantiles=(0.1, 0.5),
        )
        row["median_cloud_top_pressure_hpa"] = pressure_statistics["q50"]
        row["p10_cloud_top_pressure_hpa"] = pressure_statistics["q10"]
        ttl_mask = high_mask & (
            cloud_top_pressure <= TTL_BASE_PRESSURE_HPA
        )
        ttl_area = _area_sum(ttl_mask, area)
        row["ttl_reaching_high_cloud_area_km2"] = ttl_area
        row["ttl_reaching_high_cloud_fraction_of_envelope"] = (
            ttl_area / envelope_area if envelope_area > 0 else np.nan
        )
        total_high_area = row["total_high_cloud_area_km2"]
        row["total_high_cloud_fraction_of_envelope"] = (
            total_high_area / envelope_area
            if envelope_area > 0 and np.isfinite(total_high_area)
            else np.nan
        )
        row["area_integrated_high_cloud_occurrence_km2"] = total_high_area
        for variable_name, statistic, output_name in (
            ("olr", "mean", "olr_w_m2"),
            ("w", "max", "maximum_vertical_velocity"),
            ("w", "quantile", "p95_vertical_velocity"),
            ("high_cloud_ice_mass", "sum", "high_cloud_ice_mass"),
            ("high_cloud_snow_mass", "sum", "high_cloud_snow_mass"),
            ("temperature_100hpa", "mean", "temperature_100hpa"),
            ("temperature_100hpa_event_anomaly", "mean", "temperature_100hpa_event_anomaly"),
            ("temperature_100_200hpa", "mean", "temperature_100_200hpa"),
            ("temperature_100_200hpa_event_anomaly", "mean", "temperature_100_200hpa_event_anomaly"),
            ("utls_temperature_anomaly", "mean", "utls_temperature_anomaly"),
            ("tropopause_static_stability", "mean", "tropopause_static_stability"),
            ("tropopause_static_stability_event_anomaly", "mean", "tropopause_static_stability_event_anomaly"),
            ("zonal_wind_100hpa", "mean", "zonal_wind_100hpa"),
            ("zonal_wind_100hpa_event_anomaly", "mean", "zonal_wind_100hpa_event_anomaly"),
            ("zonal_wind_250hpa", "mean", "zonal_wind_250hpa"),
            ("zonal_wind_250hpa_event_anomaly", "mean", "zonal_wind_250hpa_event_anomaly"),
            ("upper_tropospheric_zonal_wind", "mean", "upper_tropospheric_zonal_wind"),
            ("upper_tropospheric_zonal_wind_event_anomaly", "mean", "upper_tropospheric_zonal_wind_event_anomaly"),
            ("utls_shear_u100_minus_u250", "mean", "utls_shear_u100_minus_u250"),
            ("utls_shear_u100_minus_u250_event_anomaly", "mean", "utls_shear_u100_minus_u250_event_anomaly"),
            ("deep_layer_shear_u200_minus_u850", "mean", "deep_layer_shear_u200_minus_u850"),
            ("deep_layer_shear_u200_minus_u850_event_anomaly", "mean", "deep_layer_shear_u200_minus_u850_event_anomaly"),
            ("layer_mean_n2_ttl", "mean", "layer_mean_n2_ttl"),
            ("layer_mean_n2_ttl_event_anomaly", "mean", "layer_mean_n2_ttl_event_anomaly"),
            ("cold_point_tropopause_pressure_hpa", "mean", "cold_point_tropopause_pressure_hpa"),
            ("cold_point_tropopause_pressure_hpa_event_anomaly", "mean", "cold_point_tropopause_pressure_hpa_event_anomaly"),
            ("cold_point_tropopause_height_m", "mean", "cold_point_tropopause_height_m"),
            ("cold_point_tropopause_height_m_event_anomaly", "mean", "cold_point_tropopause_height_m_event_anomaly"),
            ("precipitable_water_kg_m2", "mean", "precipitable_water_kg_m2"),
            ("precipitable_water_kg_m2_event_anomaly", "mean", "precipitable_water_kg_m2_event_anomaly"),
            ("olr_event_anomaly", "mean", "olr_event_anomaly"),
        ):
            field = model_fields[variable_name]
            if "Time" in field.dims:
                field = field.sel(Time=time_value)
            sampling_mask = (
                mask if variable_name in ENVELOPE_MODEL_FIELDS else rain_mask
            )
            if statistic == "mean":
                value = sample_masked_field(
                    field, sampling_mask, area
                )["mean"]
            elif statistic == "max":
                value = field.where(sampling_mask).max(skipna=True)
            elif statistic == "quantile":
                value = sample_masked_field(
                    field, sampling_mask, area, quantiles=(0.95,)
                )["q95"]
            else:
                value = sample_masked_field(
                    field, sampling_mask, area
                )["area_integral"]
                if output_name in {
                    "high_cloud_ice_mass",
                    "high_cloud_snow_mass",
                }:
                    value = value * 1.0e6
            row[output_name] = float(value)
        row["olr_per_area_integrated_precipitation"] = (
            row["olr_w_m2"] / precip_volume
            if precip_volume > 0
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _profile_value_at_index(profile: np.ndarray, index: float) -> float:
    """Select one profile value in an apply_ufunc-compatible kernel."""

    values = np.asarray(profile, dtype=float)
    if not np.isfinite(index):
        return np.nan
    integer_index = int(index)
    if integer_index < 0 or integer_index >= values.size:
        return np.nan
    return float(values[integer_index])


def _bounded_time_longitude_means(
    fields: Mapping[str, xr.DataArray],
    *,
    batch_size: int,
    progress_label: str,
) -> dict[str, xr.DataArray]:
    """Calculate full-record zonal-time means in explicit memory bounds."""

    if not fields:
        return {}
    if batch_size < 1:
        raise ValueError("Lifecycle baseline batch size must be positive")
    first = next(iter(fields.values()))
    if not {"Time", "longitude"}.issubset(first.dims):
        raise ValueError("Lifecycle baseline fields require Time and longitude")
    time_count = first.sizes["Time"]
    for name, field in fields.items():
        if not {"Time", "longitude"}.issubset(field.dims):
            raise ValueError(
                f"Lifecycle baseline field {name!r} requires Time and longitude"
            )
        if not field.indexes["Time"].equals(first.indexes["Time"]):
            raise ValueError(
                f"Lifecycle baseline field {name!r} has different times"
            )

    total_sums: dict[str, xr.DataArray] = {}
    total_counts: dict[str, xr.DataArray] = {}
    batch_count = (time_count + batch_size - 1) // batch_size
    for batch_number, start in enumerate(
        range(0, time_count, batch_size), start=1
    ):
        stop = min(start + batch_size, time_count)
        progress(
            f"{progress_label}: baseline time batch {batch_number}/"
            f"{batch_count} ({start + 1}-{stop}/{time_count})"
        )
        partial_variables: dict[str, xr.DataArray] = {}
        for name, field in fields.items():
            block = field.isel(Time=slice(start, stop)).reset_coords(drop=True)
            dimensions = ("Time", "longitude")
            partial_variables[f"{name}__sum"] = block.sum(
                dimensions, skipna=True
            )
            partial_variables[f"{name}__count"] = block.notnull().sum(
                dimensions
            )
        partial = xr.Dataset(partial_variables).load()
        for name in fields:
            block_sum = partial[f"{name}__sum"]
            block_count = partial[f"{name}__count"]
            if name not in total_sums:
                total_sums[name] = block_sum.copy(deep=True)
                total_counts[name] = block_count.copy(deep=True)
            else:
                total_sums[name] += block_sum
                total_counts[name] += block_count

    baselines: dict[str, xr.DataArray] = {}
    for name in fields:
        baseline = total_sums[name] / total_counts[name].where(
            total_counts[name] > 0
        )
        baseline.name = f"{name}_time_zonal_mean"
        baselines[name] = baseline
    return baselines


def _ttl_layer_mean_n2(
    theta: xr.DataArray,
    height: xr.DataArray,
    layer_thickness: xr.DataArray,
    vertical_dimension: str,
    *,
    progress_label: str,
) -> xr.DataArray:
    """Calculate TTL N² from only relevant levels and exact gradient halos."""

    lower_m = 1000.0 * TTL_HEIGHT_BOUNDS_KM[0]
    upper_m = 1000.0 * TTL_HEIGHT_BOUNDS_KM[1]
    in_layer = (height >= lower_m) & (height <= upper_m)
    reduction_dimensions = [
        dimension
        for dimension in in_layer.dims
        if dimension != vertical_dimension
    ]
    # Scan the height support in bounded time batches. A whole-record
    # reduction can schedule many native columns concurrently before the
    # first environmental baseline starts.
    level_support = None
    time_count = in_layer.sizes.get("Time", 1)
    batch_size = LIFECYCLE_VERTICAL_BASELINE_TIME_BATCH_SIZE
    for start in range(0, time_count, batch_size):
        stop = min(start + batch_size, time_count)
        progress(
            f"{progress_label}: scanning TTL height support "
            f"{start + 1}-{stop}/{time_count}"
        )
        block = (
            in_layer.isel(Time=slice(start, stop))
            if "Time" in in_layer.dims else in_layer
        )
        support = (
            block.any(reduction_dimensions)
            if reduction_dimensions else block
        ).compute(scheduler="synchronous")
        level_support = (
            support if level_support is None else level_support | support
        )
    if level_support is None:
        raise RuntimeError("No times are available for TTL height support")
    selected_levels = np.flatnonzero(
        np.asarray(level_support.values, dtype=bool)
    )
    if not selected_levels.size:
        raise RuntimeError(
            "No MPAS native levels intersect the configured TTL height layer"
        )
    support_start = max(0, int(selected_levels[0]) - 1)
    support_stop = min(
        theta.sizes[vertical_dimension],
        int(selected_levels[-1]) + 2,
    )
    progress(
        f"{progress_label}: TTL N2 uses native levels "
        f"{support_start + 1}-{support_stop}/"
        f"{theta.sizes[vertical_dimension]}"
    )
    selector = {vertical_dimension: slice(support_start, support_stop)}
    theta_support = theta.isel(selector)
    height_support = height.isel(selector)
    thickness_support = layer_thickness.isel(selector)
    in_layer_support = in_layer.isel(selector)
    n2 = blockwise_buoyancy_frequency_squared(
        theta_support,
        height_support,
        vertical_dimension,
        mask_nonpositive=False,
    )
    valid_layer = in_layer_support & np.isfinite(n2)
    return (
        (n2 * thickness_support)
        .where(valid_layer)
        .sum(vertical_dimension, skipna=True)
        / thickness_support.where(valid_layer).sum(
            vertical_dimension, skipna=True
        )
    )


def project_model_fields(
    dataset: xr.Dataset,
    precipitation: xr.DataArray,
    *,
    progress_label: str = "lifecycle",
) -> dict:
    """Project the fixed MPAS input contract onto stable lifecycle names."""

    required = (
        "olrtoa",
        "w",
        "cldfrac",
        "precipw",
        "t_isobaric",
        "uzonal_isobaric",
        "theta",
        "pressure",
        "zgrid",
        "rho",
        "qi",
        "qs",
    )
    missing = sorted(set(required) - set(dataset.data_vars))
    if missing:
        raise RuntimeError(
            "MPAS lifecycle input lacks required preferred fields: "
            f"{missing}"
        )
    for variable in ("t_isobaric", "uzonal_isobaric"):
        if "pres" not in dataset[variable].dims:
            raise RuntimeError(
                f"MPAS lifecycle input {variable!r} lacks the standardised "
                "'pres' dimension"
            )

    fields: dict[str, xr.DataArray] = {}
    environmental_names: list[str] = []

    def add_environment(name: str, field: xr.DataArray) -> None:
        field = field.sel(Time=precipitation.Time)
        fields[name] = field
        environmental_names.append(name)

    add_environment("olr", dataset.olrtoa)
    vertical = vertical_dim(dataset.w)
    fields["w"] = dataset.w.sel(Time=precipitation.Time).max(
        vertical, skipna=True
    )
    fields["cloud_fraction_profile"] = dataset.cldfrac.sel(
        Time=precipitation.Time
    )
    add_environment("precipitable_water_kg_m2", dataset.precipw)

    temperature = dataset.t_isobaric.sel(Time=precipitation.Time)
    add_environment(
        "temperature_100hpa",
        temperature.sel(pres=100.0, method="nearest"),
    )
    add_environment(
        "temperature_100_200hpa",
        temperature.where(
            (temperature.pres >= 100.0) & (temperature.pres <= 200.0),
            drop=True,
        ).mean("pres", skipna=True),
    )
    utls = temperature.where(
        (temperature.pres >= 100.0) & (temperature.pres <= 200.0),
        drop=True,
    ).mean("pres", skipna=True)
    t100 = temperature.sel(pres=100.0, method="nearest")
    t200 = temperature.sel(pres=200.0, method="nearest")
    theta100 = t100 * (1000.0 / float(t100.pres)) ** 0.286
    theta200 = t200 * (1000.0 / float(t200.pres)) ** 0.286
    stability = (theta100 - theta200).rename(
        "tropopause_static_stability"
    )
    stability.attrs.update(
        units="K per 100 hPa",
        definition="theta(near 100 hPa) minus theta(near 200 hPa)",
    )
    add_environment("tropopause_static_stability", stability)
    pressure_coordinate = xr.DataArray(
        temperature.pres.values,
        dims=("pres",),
        coords={"pres": temperature.pres},
    )
    valid_temperature_profile = temperature.notnull().any("pres")
    cold_temperature = temperature.fillna(np.inf).min("pres").where(
        valid_temperature_profile
    )
    cold_pressure_candidates = pressure_coordinate.broadcast_like(
        temperature
    ).where(temperature == cold_temperature)
    cold_pressure = cold_pressure_candidates.fillna(np.inf).min(
        "pres"
    ).where(valid_temperature_profile)
    add_environment(
        "cold_point_tropopause_pressure_hpa",
        cold_pressure,
    )

    zonal_wind = dataset.uzonal_isobaric.sel(Time=precipitation.Time)
    add_environment(
        "zonal_wind_100hpa",
        zonal_wind.sel(pres=100.0, method="nearest"),
    )
    add_environment(
        "zonal_wind_250hpa",
        zonal_wind.sel(pres=250.0, method="nearest"),
    )
    add_environment(
        "upper_tropospheric_zonal_wind",
        zonal_wind.where(
            (zonal_wind.pres >= 150.0) & (zonal_wind.pres <= 300.0),
            drop=True,
        ).mean("pres", skipna=True),
    )
    add_environment(
        "utls_shear_u100_minus_u250",
        zonal_wind.sel(pres=100.0, method="nearest")
        - zonal_wind.sel(pres=250.0, method="nearest"),
    )
    add_environment(
        "deep_layer_shear_u200_minus_u850",
        zonal_wind.sel(pres=200.0, method="nearest")
        - zonal_wind.sel(pres=850.0, method="nearest"),
    )

    theta = dataset.theta.sel(Time=precipitation.Time)
    zdim = vertical_dim(theta)
    height = native_midlevel_height(dataset, theta)
    lower, upper = native_layer_interfaces(dataset, theta)
    layer_thickness = upper - lower
    theta, height, layer_thickness = xr.align(
        theta, height, layer_thickness, join="exact"
    )
    layer_n2 = _ttl_layer_mean_n2(
        theta,
        height,
        layer_thickness,
        zdim,
        progress_label=progress_label,
    )
    layer_n2.attrs.update(
        units="s-2",
        definition=(
            "native-layer-thickness-weighted N2 over MPAS mid-levels "
            f"between {TTL_HEIGHT_BOUNDS_KM[0]:g} and "
            f"{TTL_HEIGHT_BOUNDS_KM[1]:g} km"
        ),
        vertical_coordinate="midpoints of zgrid interfaces",
    )
    add_environment("layer_mean_n2_ttl", layer_n2)
    pressure_pa = pressure_to_pa(dataset.pressure).sel(
        Time=precipitation.Time
    )
    native_temperature = theta * (pressure_pa / 100_000.0) ** (
        287.0 / 1004.0
    )
    valid_native_temperature_profile = native_temperature.notnull().any(zdim)
    cold_index = native_temperature.fillna(np.inf).argmin(zdim)
    cold_height = xr.apply_ufunc(
        _profile_value_at_index,
        height,
        cold_index,
        input_core_dims=[[zdim], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    cold_height = cold_height.where(valid_native_temperature_profile)
    cold_height.attrs.update(
        units="m",
        vertical_coordinate="midpoints of zgrid interfaces",
    )
    add_environment("cold_point_tropopause_height_m", cold_height)

    density = dataset.rho.sel(Time=precipitation.Time)
    lower, upper = native_layer_interfaces(dataset, density)
    dz = upper - lower
    high = pressure_to_pa(dataset.pressure).sel(
        Time=precipitation.Time
    ) <= 44_000.0
    for source_name, output_name in (
        ("qi", "high_cloud_ice_mass"),
        ("qs", "high_cloud_snow_mass"),
    ):
        fields[output_name] = (
            dataset[source_name].sel(Time=precipitation.Time)
            * density
            * dz
        ).where(high).sum(zdim, skipna=True)

    # Event anomalies retain each experiment's altered mean state. Calculate
    # the full-record baselines in explicit batches to prevent Dask from
    # materialising several complete native-level columns concurrently.
    expensive_names = {
        "layer_mean_n2_ttl",
        "cold_point_tropopause_height_m",
    }
    baseline_groups = (
        ("surface environment", ("olr", "precipitable_water_kg_m2")),
        (
            "temperature environment",
            (
                "temperature_100hpa",
                "temperature_100_200hpa",
                "tropopause_static_stability",
                "cold_point_tropopause_pressure_hpa",
            ),
        ),
        (
            "wind environment",
            (
                "zonal_wind_100hpa",
                "zonal_wind_250hpa",
                "upper_tropospheric_zonal_wind",
                "utls_shear_u100_minus_u250",
                "deep_layer_shear_u200_minus_u850",
            ),
        ),
    )
    grouped_names = {
        name for _label, names in baseline_groups for name in names
    }
    expected_standard_names = set(environmental_names) - expensive_names
    if grouped_names != expected_standard_names:
        raise RuntimeError(
            "Lifecycle baseline groups do not match environmental fields: "
            f"missing={sorted(expected_standard_names - grouped_names)}, "
            f"extra={sorted(grouped_names - expected_standard_names)}"
        )
    baselines: dict[str, xr.DataArray] = {}
    for group_label, names in baseline_groups:
        baselines.update(
            _bounded_time_longitude_means(
                {name: fields[name] for name in names},
                batch_size=LIFECYCLE_BASELINE_TIME_BATCH_SIZE,
                progress_label=f"{progress_label}: {group_label}",
            )
        )
    for name in (
        "layer_mean_n2_ttl",
        "cold_point_tropopause_height_m",
    ):
        baselines.update(
            _bounded_time_longitude_means(
                {name: fields[name]},
                batch_size=LIFECYCLE_VERTICAL_BASELINE_TIME_BATCH_SIZE,
                progress_label=f"{progress_label}: {name}",
            )
        )
    fields["utls_temperature_anomaly"] = (
        utls - baselines["temperature_100_200hpa"]
    )
    for name in environmental_names:
        anomaly_name = (
            "olr_event_anomaly"
            if name == "olr"
            else f"{name}_event_anomaly"
        )
        fields[anomaly_name] = fields[name] - baselines[name]
    _require_mapping_fields(
        fields,
        (*REQUIRED_EVENT_MODEL_FIELDS, "cloud_fraction_profile"),
        product="projected MPAS lifecycle diagnostics",
    )
    return fields


def event_lifecycle_summary(
    lifecycle: pd.DataFrame,
) -> pd.DataFrame:
    """Return rejected-edge timing and environment summaries."""

    required = {
        "simulation",
        "event_id",
        "lag_hours",
        "area_integrated_24h_precip_km2_mm",
        "active_deep_convective_area_km2",
        "stratiform_precipitating_high_cloud_area_km2",
        "nonprecipitating_high_cloud_area_km2",
        "total_high_cloud_area_km2",
        "ttl_reaching_high_cloud_fraction_of_envelope",
        "olr_w_m2",
        "median_cloud_top_pressure_hpa",
        "p10_cloud_top_pressure_hpa",
        "temperature_100hpa",
        "temperature_100hpa_event_anomaly",
        "temperature_100_200hpa",
        "temperature_100_200hpa_event_anomaly",
        "layer_mean_n2_ttl",
        "layer_mean_n2_ttl_event_anomaly",
        "utls_shear_u100_minus_u250",
        "utls_shear_u100_minus_u250_event_anomaly",
    }
    missing = sorted(required - set(lifecycle.columns))
    if missing:
        raise RuntimeError(
            "Lifecycle summary lacks required total-cloud/environment fields: "
            f"{missing}"
        )
    if "primary_tropical_sample" in lifecycle:
        lifecycle = lifecycle[lifecycle.primary_tropical_sample.astype(bool)]
    if "reference_in_record_subset" in lifecycle:
        lifecycle = lifecycle[
            lifecycle.reference_in_record_subset.astype(bool)
        ]
    rows: list[dict] = []
    extrema = {
        "active_peak_lag_hours": "active_deep_convective_area_km2",
        "stratiform_peak_lag_hours": (
            "stratiform_precipitating_high_cloud_area_km2"
        ),
        "nonprecip_high_cloud_peak_lag_hours": (
            "nonprecipitating_high_cloud_area_km2"
        ),
        "total_high_cloud_peak_lag_hours": "total_high_cloud_area_km2",
        "ttl_reaching_fraction_peak_lag_hours": (
            "ttl_reaching_high_cloud_fraction_of_envelope"
        ),
        "olr_minimum_lag_hours": "olr_w_m2",
        "deepest_cloud_top_pressure_lag_hours": (
            "p10_cloud_top_pressure_hpa"
        ),
    }
    for event_id, event in lifecycle.groupby("event_id", sort=False):
        event = event.sort_values("lag_hours")
        window = event[
            event.lag_hours.between(
                float(np.min(LIFECYCLE_LAGS_HOURS)),
                float(np.max(LIFECYCLE_LAGS_HOURS)),
            )
        ].copy()
        row = {
            "simulation": event.simulation.iloc[0],
            "event_id": event_id,
        }
        if window.empty:
            continue
        row["window_minimum_lag_hours"] = float(window.lag_hours.min())
        row["window_maximum_lag_hours"] = float(window.lag_hours.max())
        for output_name, column in extrema.items():
            smoothed = window[column].rolling(
                window=3,
                center=True,
                min_periods=1,
            ).mean()
            if not np.isfinite(smoothed).any():
                row[output_name] = np.nan
                row[f"{output_name}_censored"] = True
                continue
            index = (
                smoothed.idxmin()
                if column in {"olr_w_m2", "p10_cloud_top_pressure_hpa"}
                else smoothed.idxmax()
            )
            lag = float(window.loc[index, "lag_hours"])
            censored = bool(
                np.isclose(lag, window.lag_hours.min())
                or np.isclose(lag, window.lag_hours.max())
            )
            row[output_name] = np.nan if censored else lag
            row[f"{output_name}_censored"] = censored
        post_peak = window[window.lag_hours > 0.0]
        smoothed_anvil = post_peak[
            "nonprecipitating_high_cloud_area_km2"
        ].rolling(window=3, center=True, min_periods=1).mean()
        if np.isfinite(smoothed_anvil).any():
            index = smoothed_anvil.idxmax()
            lag = float(window.loc[index, "lag_hours"])
            censored = bool(np.isclose(lag, window.lag_hours.max()))
            row["anvil_delay_hours"] = np.nan if censored else lag
            row["anvil_delay_hours_censored"] = censored
        else:
            row["anvil_delay_hours"] = np.nan
            row["anvil_delay_hours_censored"] = True
        active_peak = np.nanmax(event.active_deep_convective_area_km2)
        post_high = _finite_mean(
            post_peak.total_high_cloud_area_km2.to_numpy(),
            axis=0,
        )
        row["post_peak_high_cloud_to_peak_active_area_ratio"] = (
            post_high / active_peak if active_peak > 0 else np.nan
        )
        row["integrated_high_cloud_area_hours"] = float(
            np.trapz(
                event.total_high_cloud_area_km2.values,
                event.lag_hours.values,
            )
        )
        peak_rain = np.nanmax(event.area_integrated_24h_precip_km2_mm)
        post_total = _finite_mean(
            post_peak.total_high_cloud_area_km2.to_numpy(),
            axis=0,
        )
        row["post_peak_total_high_cloud_to_peak_precipitation_ratio"] = (
            post_total / peak_rain if peak_rain > 0 else np.nan
        )
        for environmental in (
            "temperature_100hpa",
            "temperature_100hpa_event_anomaly",
            "temperature_100_200hpa",
            "temperature_100_200hpa_event_anomaly",
            "layer_mean_n2_ttl",
            "layer_mean_n2_ttl_event_anomaly",
            "utls_shear_u100_minus_u250",
            "utls_shear_u100_minus_u250_event_anomaly",
        ):
            row[f"pre_peak_mean_{environmental}"] = float(
                event.loc[event.lag_hours < 0, environmental].mean()
            )
            row[f"post_peak_mean_{environmental}"] = float(
                event.loc[event.lag_hours > 0, environmental].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def normalized_lifetime_composite(
    lifecycle: pd.DataFrame,
    variables: Sequence[str],
    *,
    bin_count: int = 20,
    n_bootstrap: int = TIMING_BOOTSTRAP_RESAMPLES,
    seed: int = TIMING_BOOTSTRAP_SEED,
) -> xr.Dataset:
    """Map full tracks to equal normalized-lifetime bins."""

    selected = lifecycle.copy()
    for flag in ("primary_tropical_sample", "reference_in_record_subset"):
        if flag in selected:
            selected = selected[selected[flag].astype(bool)]
    simulations = [
        name for name in SIMULATION_DIRS if name in set(selected.simulation)
    ]
    lifetime_percent = np.linspace(
        50.0 / bin_count,
        100.0 - 50.0 / bin_count,
        bin_count,
    )
    means = np.full((len(simulations), bin_count, len(variables)), np.nan)
    low = np.full_like(means, np.nan)
    high = np.full_like(means, np.nan)
    counts = np.zeros(len(simulations), dtype=int)
    peak_percent = np.full(len(simulations), np.nan)
    rng = np.random.default_rng(seed)
    for simulation_index, simulation in enumerate(simulations):
        events = []
        event_peaks = []
        simulation_rows = selected[selected.simulation == simulation]
        for _event_id, event in simulation_rows.groupby(
            "event_id", sort=False
        ):
            event = event.sort_values("Time")
            if len(event) < 2:
                continue
            time = pd.to_datetime(event.Time)
            elapsed = np.asarray(
                (time - time.iloc[0]) / (time.iloc[-1] - time.iloc[0]),
                dtype=float,
            )
            if not np.isfinite(elapsed).all():
                continue
            event_values = np.full((bin_count, len(variables)), np.nan)
            for variable_index, variable in enumerate(variables):
                values = np.asarray(event[variable], dtype=float)
                valid = np.isfinite(values)
                if valid.sum() >= 2:
                    event_values[:, variable_index] = np.interp(
                        lifetime_percent / 100.0,
                        elapsed[valid],
                        values[valid],
                    )
            rain = np.asarray(
                event.area_integrated_24h_precip_km2_mm,
                dtype=float,
            )
            if np.isfinite(rain).any():
                event_peaks.append(100.0 * elapsed[np.nanargmax(rain)])
            events.append(event_values)
        if not events:
            continue
        cube = np.asarray(events)
        counts[simulation_index] = len(cube)
        means[simulation_index] = _finite_mean(cube, axis=0)
        bootstrap = np.full((n_bootstrap, bin_count, len(variables)), np.nan)
        for draw_index in range(n_bootstrap):
            draw = rng.integers(0, len(cube), len(cube))
            bootstrap[draw_index] = _finite_mean(cube[draw], axis=0)
        low[simulation_index], high[simulation_index] = _finite_percentiles(
            bootstrap, [2.5, 97.5]
        )
        peak_percent[simulation_index] = float(np.nanmedian(event_peaks))
    return xr.Dataset(
        {
            "mean": (("simulation", "lifetime_percent", "variable"), means),
            "ci_low": (("simulation", "lifetime_percent", "variable"), low),
            "ci_high": (("simulation", "lifetime_percent", "variable"), high),
            "event_count": (("simulation",), counts),
            "median_precipitation_peak_percent": (
                ("simulation",), peak_percent
            ),
        },
        coords={
            "simulation": simulations,
            "lifetime_percent": lifetime_percent,
            "variable": list(variables),
        },
        attrs={
            "lifetime_definition": "0% genesis; 100% termination",
            "lifetime_bins": int(bin_count),
            "bootstrap_resamples": int(n_bootstrap),
        },
    )


def _finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    """Return a NaN-aware mean without warning for empty output cells."""

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    result = np.full(np.shape(total), np.nan, dtype=float)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _finite_percentiles(
    values: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    """Return column percentiles while preserving wholly empty cells."""

    values = np.asarray(values, dtype=float)
    flat = values.reshape(values.shape[0], -1)
    result = np.full((len(quantiles), flat.shape[1]), np.nan)
    valid_columns = np.isfinite(flat).any(axis=0)
    if valid_columns.any():
        result[:, valid_columns] = np.nanpercentile(
            flat[:, valid_columns],
            quantiles,
            axis=0,
        )
    return result.reshape((len(quantiles),) + values.shape[1:])


def bootstrap_event_composite(
    lifecycle: pd.DataFrame,
    variables: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    seed: int = 20260727,
) -> xr.Dataset:
    """Build composites by resampling independent LPO event identities."""

    if TEST_MODE:
        n_bootstrap = 0

    selected = lifecycle[
        lifecycle.in_primary_lifecycle_window.astype(bool)
    ].copy()
    if "primary_tropical_sample" in selected:
        selected = selected[selected.primary_tropical_sample.astype(bool)]
    if "reference_in_record_subset" in selected:
        selected = selected[selected.reference_in_record_subset.astype(bool)]
    lags = np.asarray(
        sorted(selected.lag_hours.dropna().astype(int).unique())
        if "test_lifecycle_sample" in selected
        else LIFECYCLE_LAGS_HOURS,
        dtype=int,
    )
    simulations = [
        name for name in SIMULATION_DIRS if name in set(selected.simulation)
    ]
    shape = (len(simulations), len(lags), len(variables))
    mean = np.full(shape, np.nan)
    area_weighted = np.full(shape, np.nan)
    precipitation_weighted = np.full(shape, np.nan)
    low = np.full(shape, np.nan)
    high = np.full(shape, np.nan)
    count = np.zeros(len(simulations), dtype=int)
    rng = np.random.default_rng(seed)
    bootstrap_by_simulation: dict[str, np.ndarray] = {}
    for isim, simulation in enumerate(simulations):
        sim = selected[selected.simulation == simulation]
        event_ids = sim.event_id.unique()
        count[isim] = len(event_ids)
        if not len(event_ids):
            continue
        cube = np.full(
            (len(event_ids), len(lags), len(variables)),
            np.nan,
        )
        area_weights = np.full(len(event_ids), np.nan)
        precipitation_weights = np.full(len(event_ids), np.nan)
        for ievent, event_id in enumerate(event_ids):
            event = sim[sim.event_id == event_id].set_index("lag_hours")
            if "envelope_area_km2" in event:
                area_weights[ievent] = float(
                    np.nanmax(event.envelope_area_km2)
                )
            if "area_integrated_precip_km2_mm_day" in event:
                precipitation_weights[ievent] = float(
                    np.nansum(event.area_integrated_precip_km2_mm_day)
                )
            for ilag, lag in enumerate(lags):
                if lag not in event.index:
                    continue
                for ivar, variable in enumerate(variables):
                    if variable in event:
                        cube[ievent, ilag, ivar] = event.loc[lag, variable]
        mean[isim] = _finite_mean(cube, axis=0)
        for weights, destination in (
            (area_weights, area_weighted),
            (precipitation_weights, precipitation_weighted),
        ):
            for ilag in range(len(lags)):
                for ivar in range(len(variables)):
                    values = cube[:, ilag, ivar]
                    valid = (
                        np.isfinite(values)
                        & np.isfinite(weights)
                        & (weights > 0)
                    )
                    if valid.any():
                        destination[isim, ilag, ivar] = np.average(
                            values[valid],
                            weights=weights[valid],
                        )
        boot = np.empty(
            (n_bootstrap, len(lags), len(variables))
        )
        for index in range(n_bootstrap):
            draw = rng.integers(0, len(event_ids), len(event_ids))
            boot[index] = _finite_mean(cube[draw], axis=0)
        bootstrap_by_simulation[simulation] = boot
        if n_bootstrap:
            low[isim], high[isim] = _finite_percentiles(
                boot, [2.5, 97.5]
            )
    data_vars = {
        "event_equal_mean": (
            ("simulation", "lag_hours", "variable"),
            mean,
        ),
        "area_weighted_mean": (
            ("simulation", "lag_hours", "variable"),
            area_weighted,
        ),
        "precipitation_weighted_mean": (
            ("simulation", "lag_hours", "variable"),
            precipitation_weighted,
        ),
        "ci_low": (("simulation", "lag_hours", "variable"), low),
        "ci_high": (("simulation", "lag_hours", "variable"), high),
        "event_count": (("simulation",), count),
    }
    contrasts = [
        simulation
        for simulation in RESPONSE_ROWS
        if simulation in bootstrap_by_simulation
        and "control" in bootstrap_by_simulation
    ]
    if contrasts:
        contrast_shape = (len(contrasts), len(lags), len(variables))
        difference_mean = np.full(contrast_shape, np.nan)
        difference_low = np.full(contrast_shape, np.nan)
        difference_high = np.full(contrast_shape, np.nan)
        contrast_event_count = np.zeros((len(contrasts), 2), dtype=int)
        simulation_index = {
            name: index for index, name in enumerate(simulations)
        }
        for icontrast, simulation in enumerate(contrasts):
            difference_mean[icontrast] = (
                mean[simulation_index[simulation]]
                - mean[simulation_index["control"]]
            )
            bootstrap_difference = (
                bootstrap_by_simulation[simulation]
                - bootstrap_by_simulation["control"]
            )
            if n_bootstrap:
                (
                    difference_low[icontrast],
                    difference_high[icontrast],
                ) = _finite_percentiles(
                    bootstrap_difference,
                    [2.5, 97.5],
                )
            contrast_event_count[icontrast] = (
                count[simulation_index[simulation]],
                count[simulation_index["control"]],
            )
        data_vars.update(
            {
                "difference_mean": (
                    ("contrast", "lag_hours", "variable"),
                    difference_mean,
                ),
                "difference_ci_low": (
                    ("contrast", "lag_hours", "variable"),
                    difference_low,
                ),
                "difference_ci_high": (
                    ("contrast", "lag_hours", "variable"),
                    difference_high,
                ),
                "contrast_event_count": (
                    ("contrast", "contrast_member"),
                    contrast_event_count,
                ),
            }
        )
    return xr.Dataset(
        data_vars,
        coords={
            "simulation": simulations,
            "lag_hours": lags,
            "variable": list(variables),
            **({"contrast": contrasts} if contrasts else {}),
            **(
                {"contrast_member": ["experiment", "control"]}
                if contrasts
                else {}
            ),
        },
        attrs={
            "sampling_unit": "LPO event",
            "bootstrap": f"{n_bootstrap} independent-event resamples",
            "event_weighting": "equal",
            "contrast_bootstrap": (
                "independent event resampling within each simulation"
            ),
        },
    )


def _sample_centroid_series(table: pd.DataFrame, wave: xr.DataArray) -> np.ndarray:
    """Sample a longitude-time wave cache at saved LPO centroid positions."""

    wave = wave.transpose("Time", "longitude")
    wave_times = pd.DatetimeIndex(pd.to_datetime(wave.Time.values))
    requested = pd.DatetimeIndex(pd.to_datetime(table.Time))
    time_indices = wave_times.get_indexer(requested)
    if np.any(time_indices < 0):
        missing = requested[time_indices < 0][:3].astype(str).tolist()
        raise RuntimeError(
            "Wave cache does not contain exact LPO lifecycle times: "
            f"{missing}"
        )
    longitude = np.asarray(wave.longitude.values, dtype=float)
    requested_lon = np.asarray(
        table.precip_weighted_centroid_lon.values, dtype=float
    )
    longitude_indices = np.asarray(
        [
            np.argmin(np.abs(((longitude - value + 180.0) % 360.0) - 180.0))
            for value in requested_lon
        ],
        dtype=int,
    )
    sample_dimension = "lpo_lifecycle_sample"
    sampled = wave.isel(
        Time=xr.DataArray(time_indices, dims=sample_dimension),
        longitude=xr.DataArray(longitude_indices, dims=sample_dimension),
    ).load()
    return np.asarray(sampled.values)


def _tracking_timing_columns(table: pd.DataFrame) -> list[str]:
    """Return columns whose values remain authoritative in tracking output."""

    preferred = (
        "simulation",
        "event_id",
        "Time",
        "envelope_area_km2",
        "area_integrated_precip_km2_mm_day",
        "area_integrated_24h_precip_km2_mm",
        "mean_precip_mm_day",
        "maximum_precip_mm_day",
        "precip_weighted_centroid_lon",
        "precip_weighted_centroid_lat",
        "reference_time",
        "lag_hours",
        "complete_primary_lag_window",
        "in_primary_lifecycle_window",
        "primary_tropical_sample",
        "reference_in_record_subset",
        "tropical_consecutive_hours",
        "test_lifecycle_sample",
    )
    return [column for column in preferred if column in table]


def _select_test_events(
    events: pd.DataFrame,
    timing: pd.DataFrame,
) -> list[str]:
    """Choose a small, representative actual-data event sample for figures."""

    candidates = events.copy()
    for flag in ("primary_tropical_sample", "reference_in_record_subset"):
        if flag in candidates:
            candidates = candidates[candidates[flag].astype(bool)]
    if "test_method_example" in candidates and candidates.test_method_example.any():
        candidates = candidates[candidates.test_method_example.astype(bool)]
    if candidates.empty:
        raise RuntimeError(
            "The tracking test produced no LPO suitable for lifecycle "
            "sampling; use a later/longer --test window."
        )
    ranking = (
        timing[timing.event_id.isin(candidates.event_id)]
        .groupby("event_id")
        .area_integrated_24h_precip_km2_mm.max()
        .sort_values()
    )
    centre = float(ranking.median())
    return (
        (ranking - centre)
        .abs()
        .sort_values()
        .head(TEST_EVENTS_PER_SIMULATION)
        .index.tolist()
    )


def _selected_mask_latitude_slice(
    latitude: xr.DataArray,
    masks: Mapping[str, xr.DataArray],
    event_ids: Sequence[str],
) -> slice:
    """Return the model-grid slice spanning all selected event-mask domains."""

    if not event_ids:
        raise RuntimeError("Lifecycle sampling requires at least one LPO event")
    missing = [event_id for event_id in event_ids if event_id not in masks]
    if missing:
        raise KeyError(f"Missing saved LPO masks for events {missing}")
    mask_latitudes = [
        np.asarray(masks[event_id].latitude.values, dtype=float)
        for event_id in event_ids
    ]
    if any(values.ndim != 1 or not values.size for values in mask_latitudes):
        raise RuntimeError("Saved LPO masks require non-empty 1-D latitude")
    lower = min(float(np.nanmin(values)) for values in mask_latitudes)
    upper = max(float(np.nanmax(values)) for values in mask_latitudes)
    model_latitude = np.asarray(latitude.values, dtype=float)
    if model_latitude.ndim != 1 or model_latitude.size < 2:
        raise RuntimeError("MPAS lifecycle data require a 1-D latitude grid")
    lower_index = int(np.nanargmin(np.abs(model_latitude - lower)))
    upper_index = int(np.nanargmin(np.abs(model_latitude - upper)))
    spacing = float(np.nanmedian(np.abs(np.diff(model_latitude))))
    tolerance = max(1.0e-6, spacing * 1.0e-3)
    for bound, index in ((lower, lower_index), (upper, upper_index)):
        mismatch = abs(float(model_latitude[index]) - bound)
        if mismatch > tolerance:
            raise RuntimeError(
                "Saved LPO mask latitude is not on the authoritative MPAS "
                f"grid: mask={bound}, nearest={model_latitude[index]}, "
                f"difference={mismatch}"
            )
    return slice(
        min(lower_index, upper_index),
        max(lower_index, upper_index) + 1,
    )


def _merge_authoritative_timing(
    diagnosed: pd.DataFrame,
    tracking_timing: pd.DataFrame,
) -> pd.DataFrame:
    timing_columns = _tracking_timing_columns(tracking_timing)
    timing = tracking_timing[timing_columns].copy()
    timing["Time"] = pd.to_datetime(timing.Time)
    diagnosed = diagnosed.copy()
    diagnosed["Time"] = pd.to_datetime(diagnosed.Time)
    replace = [
        column
        for column in timing_columns
        if column not in {"simulation", "event_id", "Time"}
    ]
    diagnosed = diagnosed.drop(columns=replace, errors="ignore")
    return diagnosed.merge(
        timing,
        on=["simulation", "event_id", "Time"],
        how="left",
        validate="one_to_one",
    )


def _sample_native_cloud_profiles(
    dataset: xr.Dataset,
    masks: Mapping[str, xr.DataArray],
    timing: pd.DataFrame,
    event_ids: Sequence[str],
    simulation: str,
) -> pd.DataFrame:
    """Area-average native cloud, height and pressure profiles in each LPO."""

    def align_midlevel_field(
        field: xr.DataArray,
        reference: xr.DataArray,
        *,
        name: str,
    ) -> xr.DataArray:
        """Align one native midpoint field to the reference ``z`` axis."""

        source_dimension = vertical_dim(field)
        if source_dimension == "nVertLevelsP1":
            raise ValueError(f"{name} uses interface levels, not midpoints")
        source_levels = field.sizes[source_dimension]
        target_levels = reference.sizes["z"]
        if source_levels != target_levels:
            raise ValueError(
                f"{name} has {source_levels} native levels, but cldfrac has "
                f"{target_levels}"
            )
        if source_dimension != "z":
            field = field.rename({source_dimension: "z"})
        return field.assign_coords(z=reference.z)

    def select_event_grid(
        field: xr.DataArray, mask: xr.DataArray
    ) -> xr.DataArray:
        """Select only event-grid coordinates that the source field owns."""

        indexers = {
            dimension: mask[dimension]
            for dimension in ("Time", "latitude", "longitude")
            if dimension in field.dims
        }
        return field.sel(indexers)

    rows: list[pd.DataFrame] = []
    cloud = dataset.cldfrac
    cloud_dimension = vertical_dim(cloud)
    if cloud_dimension == "nVertLevelsP1":
        raise ValueError("cldfrac uses interface levels, not midpoints")
    if cloud_dimension != "z":
        cloud = cloud.rename({cloud_dimension: "z"})
    if "z" not in cloud.coords:
        cloud = cloud.assign_coords(z=np.arange(cloud.sizes["z"]))
    height = native_midlevel_height(dataset, cloud)
    height = align_midlevel_field(height, cloud, name="native height")
    pressure_hpa = align_midlevel_field(
        pressure_to_pa(dataset.pressure) / 100.0,
        cloud,
        name="pressure",
    )
    for event_id in event_ids:
        event_timing = timing[timing.event_id == event_id].copy()
        event_timing["Time"] = pd.to_datetime(event_timing.Time)
        mask = masks[event_id].sel(
            Time=event_timing.Time.values,
        ).astype(bool)
        event_cloud, mask = xr.align(
            cloud.sel(Time=event_timing.Time.values),
            mask,
            join="inner",
        )
        event_height = select_event_grid(height, mask)
        event_pressure = select_event_grid(pressure_hpa, mask)
        area = xr.DataArray(
            grid_cell_area_km2(mask.latitude.values, mask.longitude.values),
            dims=("latitude", "longitude"),
            coords={"latitude": mask.latitude, "longitude": mask.longitude},
        )
        weights = area.where(mask)
        denominator = weights.sum(("latitude", "longitude"), skipna=True)
        safe_denominator = denominator.where(denominator > 0)
        profile_fields = xr.align(
            (
                (event_cloud * weights).sum(
                    ("latitude", "longitude"), skipna=True
                )
                / safe_denominator
            ),
            (
                (event_height * weights).sum(
                    ("latitude", "longitude"), skipna=True
                )
                / safe_denominator
            ),
            (
                (event_pressure * weights).sum(
                    ("latitude", "longitude"), skipna=True
                )
                / safe_denominator
            ),
            join="exact",
        )
        profiles = xr.Dataset(
            dict(
                zip(
                    ("cloud_fraction", "height_m", "pressure_hpa"),
                    profile_fields,
                    strict=True,
                )
            )
        ).assign_coords(
            vertical_level_index=(
                "z",
                np.arange(cloud.sizes["z"], dtype=int),
            )
        ).load()
        frame = profiles.to_dataframe().reset_index()
        frame["simulation"] = simulation
        frame["event_id"] = event_id
        frame = frame.drop(columns=["z"])
        frame = frame.merge(
            event_timing[["Time", "lag_hours"]],
            on="Time",
            how="left",
            validate="many_to_one",
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _add_wave_lifecycle_fields(
    lifecycle: pd.DataFrame,
    *,
    test: bool = False,
) -> pd.DataFrame:
    lifecycle = lifecycle.copy()
    for wave_name in LPO_LIFECYCLE_FILTERS:
        progress(f"Sampling {wave_name} caches along saved LPO centroids")
        for cache_field in WAVE_FIELD_DEFINITIONS:
            wave = load_existing_filtered_wave(
                WAVE_SIMULATION_LABELS[lifecycle.simulation.iloc[0]],
                cache_field,
                wave_name,
                test=test,
            )
            lifecycle[WAVE_LIFECYCLE_FIELDS[(wave_name, cache_field)]] = (
                _sample_centroid_series(lifecycle, wave)
            )
    return lifecycle


def _wave_event_summary(lifecycle: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for event_id, event in lifecycle.groupby("event_id", sort=False):
        row = {"event_id": event_id}
        for wave_name in LPO_LIFECYCLE_FILTERS:
            prefix = _wave_key(wave_name)
            for cache_field in ("Precip", "PW", "T100", "u100"):
                field = WAVE_LIFECYCLE_FIELDS[(wave_name, cache_field)]
                valid = np.isfinite(event[field])
                row[f"{prefix}_{cache_field.lower()}_peak_lag_hours"] = (
                    float(event.loc[event.loc[valid, field].idxmax(), "lag_hours"])
                    if valid.any()
                    else np.nan
                )
            field = WAVE_LIFECYCLE_FIELDS[(wave_name, "OLR")]
            valid = np.isfinite(event[field])
            row[f"{prefix}_olr_minimum_lag_hours"] = (
                float(event.loc[event.loc[valid, field].idxmin(), "lag_hours"])
                if valid.any()
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _file_sha256(path: Path) -> str:
    """Return the content hash for one cache product."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically."""

    write_csv_atomic(frame, path, index=False)


def _metadata_hash_is_valid(metadata: Mapping[str, object]) -> bool:
    """Validate the embedded hash for one metadata document."""

    expected = metadata.get("configuration_sha256")
    unhashed = dict(metadata)
    unhashed.pop("configuration_sha256", None)
    encoded = json.dumps(
        unhashed, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return expected == hashlib.sha256(encoded).hexdigest()


def _event_timing_sha256(timing: pd.DataFrame) -> str:
    """Hash every authoritative input column for one event."""

    ordered = timing.copy()
    for column in ordered.columns:
        if pd.api.types.is_datetime64_any_dtype(ordered[column]):
            ordered[column] = ordered[column].dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )
    ordered = ordered.reindex(sorted(ordered.columns), axis=1)
    sort_columns = [
        column for column in ("event_id", "Time", "lag_hours")
        if column in ordered
    ]
    if sort_columns:
        ordered = ordered.sort_values(sort_columns, kind="stable")
    encoded = ordered.to_json(
        orient="records", date_format="iso", double_precision=15
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _lifecycle_dependency_hashes(
    standard: Path,
    simulation: str,
    event_ids: Sequence[str],
    *,
    test: bool,
) -> dict[str, str]:
    """Hash calculation and wave-cache source file identities."""

    tracking_paths = [
        standard / "lpt_event_relative_diagnostics.csv",
        standard / "lpt_events.csv",
        standard / "lpt_track_steps.csv",
        standard / "lpt_event_mask_index.csv",
    ]
    mask_index = pd.read_csv(
        standard / "lpt_event_mask_index.csv", dtype={"event_id": str}
    ).set_index("event_id")
    missing_masks = sorted(set(map(str, event_ids)) - set(mask_index.index))
    if missing_masks:
        raise RuntimeError(
            f"{standard}: mask index lacks lifecycle events {missing_masks[:3]}"
        )
    mask_paths = [
        standard / str(mask_index.loc[str(event_id), "mask_file"])
        for event_id in event_ids
    ]
    source_directory = Path(DEFAULT_ROOT) / SIMULATION_DIRS[simulation]
    model_paths = sorted(source_directory.glob(LATLON_PATTERN))
    classification_path = cache_directory(
        "classification",
        f"{simulation.replace('-', '_')}_classified_high_cloud.nc",
    )
    calculation_paths = tracking_paths + mask_paths + model_paths + [
        classification_path
    ]
    calculation_hash = source_file_signature(calculation_paths)["sha256"]
    wave_paths = [
        Path(
            filtered_wave_cache_path(
                WAVE_SIMULATION_LABELS[simulation],
                cache_field,
                wave_name,
                test=test,
            )
        )
        for wave_name in LPO_LIFECYCLE_FILTERS
        for cache_field in WAVE_FIELD_DEFINITIONS
    ]
    bundle_hash = source_file_signature(
        calculation_paths + wave_paths
    )["sha256"]
    return {"calculation": calculation_hash, "bundle": bundle_hash}


def _event_cache_paths(
    standard: Path, event_id: str
) -> tuple[Path, Path, Path]:
    token = hashlib.sha256(str(event_id).encode("utf-8")).hexdigest()[:20]
    root = standard / LIFECYCLE_EVENT_CACHE_DIRECTORY
    return (
        root / f"{token}.steps.csv",
        root / f"{token}.profiles.csv",
        root / f"{token}.json",
    )


def _require_valid_native_profiles(
    profiles: pd.DataFrame,
    steps: pd.DataFrame,
) -> None:
    """Require one native-level profile row for each step and level."""

    key_columns = ["event_id", "Time"]
    profile_key_columns = key_columns + ["vertical_level_index"]
    missing_steps = set(key_columns) - set(steps.columns)
    missing_profiles = set(profile_key_columns) - set(profiles.columns)
    if missing_steps or missing_profiles:
        raise ValueError(
            "Lifecycle profile tables lack required keys: "
            f"steps={sorted(missing_steps)}, "
            f"profiles={sorted(missing_profiles)}"
        )
    if steps.empty or profiles.empty:
        raise ValueError("Lifecycle steps and native profiles must be non-empty")

    step_keys = steps[key_columns].copy()
    profile_keys = profiles[profile_key_columns].copy()
    step_keys["event_id"] = step_keys.event_id.astype(str)
    profile_keys["event_id"] = profile_keys.event_id.astype(str)
    step_keys["Time"] = pd.to_datetime(step_keys.Time)
    profile_keys["Time"] = pd.to_datetime(profile_keys.Time)
    if step_keys.duplicated().any():
        raise ValueError("Lifecycle steps contain duplicate event-time keys")

    numeric_levels = pd.to_numeric(
        profile_keys.vertical_level_index,
        errors="raise",
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric_levels)):
        raise ValueError("Native vertical level indices must be finite")
    integer_levels = numeric_levels.astype(np.int64)
    if np.any(numeric_levels != integer_levels) or np.any(integer_levels < 0):
        raise ValueError("Native vertical level indices must be non-negative integers")
    profile_keys["vertical_level_index"] = integer_levels
    if profile_keys.duplicated().any():
        raise ValueError(
            "Native profiles contain duplicate event-time-level keys"
        )

    expected_step_keys = set(map(tuple, step_keys.to_numpy()))
    actual_step_keys = set(
        map(tuple, profile_keys[key_columns].drop_duplicates().to_numpy())
    )
    if actual_step_keys != expected_step_keys:
        raise ValueError("Native profile event-time keys differ from lifecycle steps")

    levels = np.unique(integer_levels)
    expected_levels = np.arange(levels.size, dtype=np.int64)
    if not np.array_equal(levels, expected_levels):
        raise ValueError("Native vertical level indices must form a zero-based axis")
    group_sizes = profile_keys.groupby(key_columns, sort=False).size()
    if not np.all(group_sizes.to_numpy() == levels.size):
        raise ValueError(
            "Each lifecycle step must contain every native vertical level"
        )


def _cleanup_event_checkpoints(standard: Path) -> None:
    """Remove intermediate checkpoints after final-bundle validation."""

    root = standard / LIFECYCLE_EVENT_CACHE_DIRECTORY
    if root.is_dir():
        shutil.rmtree(root)
        progress(f"Removed completed lifecycle checkpoints: {root}")


def _read_event_checkpoint(
    standard: Path,
    *,
    simulation: str,
    event_id: str,
    timing_sha256: str,
    calculation_source_sha256: str,
    test: bool,
    force_recompute: bool,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load one complete and current event checkpoint."""

    if force_recompute:
        return None
    steps_path, profiles_path, metadata_path = _event_cache_paths(
        standard, event_id
    )
    if not all(path.is_file() for path in (steps_path, profiles_path, metadata_path)):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "lifecycle_analysis_version": LIFECYCLE_ANALYSIS_VERSION,
            "tracking_product_version": PRODUCT_VERSION,
            "simulation": simulation,
            "event_id": str(event_id),
            "timing_sha256": timing_sha256,
            "calculation_source_sha256": calculation_source_sha256,
            "test_product": bool(test),
        }
        if not _metadata_hash_is_valid(metadata):
            return None
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        output_hashes = metadata.get("output_sha256", {})
        if output_hashes != {
            "steps": _file_sha256(steps_path),
            "profiles": _file_sha256(profiles_path),
        }:
            return None
        steps = pd.read_csv(steps_path, dtype={"event_id": str})
        profiles = pd.read_csv(profiles_path, dtype={"event_id": str})
        steps["Time"] = pd.to_datetime(steps.Time)
        profiles["Time"] = pd.to_datetime(profiles.Time)
        if set(steps.event_id) != {str(event_id)}:
            return None
        if set(profiles.event_id) != {str(event_id)}:
            return None
        _require_valid_native_profiles(profiles, steps)
        return steps, profiles
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_event_checkpoint(
    standard: Path,
    *,
    simulation: str,
    event_id: str,
    timing_sha256: str,
    calculation_source_sha256: str,
    test: bool,
    steps: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    """Commit one event checkpoint after both calculations finish."""

    _require_valid_native_profiles(profiles, steps)
    steps_path, profiles_path, metadata_path = _event_cache_paths(
        standard, event_id
    )
    _atomic_write_csv(steps, steps_path)
    _atomic_write_csv(profiles, profiles_path)
    _write_lifecycle_metadata_file(
        metadata_path,
        {
            "lifecycle_analysis_version": LIFECYCLE_ANALYSIS_VERSION,
            "tracking_product_version": PRODUCT_VERSION,
            "simulation": simulation,
            "event_id": str(event_id),
            "timing_sha256": timing_sha256,
            "calculation_source_sha256": calculation_source_sha256,
            "test_product": bool(test),
            "step_columns": list(steps.columns),
            "profile_columns": list(profiles.columns),
            "output_sha256": {
                "steps": _file_sha256(steps_path),
                "profiles": _file_sha256(profiles_path),
            },
        },
    )


def _read_simulation_bundle(
    standard: Path,
    *,
    simulation: str,
    event_ids: Sequence[str],
    bundle_source_sha256: str,
    test: bool,
    force_recompute: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load a validated lifecycle table bundle."""

    if force_recompute:
        return None
    paths = {
        "steps": standard / LIFECYCLE_STEP_FILENAME,
        "summary": standard / LIFECYCLE_SUMMARY_FILENAME,
        "profiles": standard / LIFECYCLE_PROFILE_FILENAME,
    }
    metadata_path = standard / LIFECYCLE_CONFIGURATION_FILENAME
    if not metadata_path.is_file() or any(
        not path.is_file() for path in paths.values()
    ):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not _metadata_hash_is_valid(metadata):
            return None
        expected = {
            "lifecycle_analysis_version": LIFECYCLE_ANALYSIS_VERSION,
            "tracking_product_version": PRODUCT_VERSION,
            "wave_config_version": WAVE_CONFIG_VERSION,
            "source_simulation": simulation,
            "bundle_source_sha256": bundle_source_sha256,
            "test_product": bool(test),
            "selected_event_ids": list(map(str, event_ids)),
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        if metadata.get("output_sha256") != {
            key: _file_sha256(path) for key, path in paths.items()
        }:
            return None
        lifecycle = pd.read_csv(paths["steps"], dtype={"event_id": str})
        summary = pd.read_csv(paths["summary"], dtype={"event_id": str})
        profiles = pd.read_csv(paths["profiles"], dtype={"event_id": str})
        lifecycle["Time"] = pd.to_datetime(lifecycle.Time)
        profiles["Time"] = pd.to_datetime(profiles.Time)
        if list(lifecycle.columns) != metadata.get("lifecycle_step_columns"):
            return None
        if list(summary.columns) != metadata.get("lifecycle_summary_columns"):
            return None
        if list(profiles.columns) != metadata.get("vertical_profile_columns"):
            return None
        expected_ids = set(map(str, event_ids))
        if set(lifecycle.event_id) != expected_ids:
            return None
        if set(summary.event_id) != expected_ids:
            return None
        if set(profiles.event_id) != expected_ids:
            return None
        _require_valid_native_profiles(profiles, lifecycle)
        return lifecycle, summary, profiles
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_lifecycle_metadata(
    standard: Path,
    *,
    simulation: str,
    event_ids: Sequence[str],
    calculation_source_sha256: str,
    bundle_source_sha256: str,
    lifecycle: pd.DataFrame,
    summary: pd.DataFrame,
    profiles: pd.DataFrame,
    test_event_ids: Sequence[str] | None,
    environment_anomaly_baseline: str,
) -> None:
    metadata = {
        "lifecycle_analysis_version": LIFECYCLE_ANALYSIS_VERSION,
        "tracking_product_version": PRODUCT_VERSION,
        "source_simulation": simulation,
        "selected_event_ids": list(map(str, event_ids)),
        "calculation_source_sha256": calculation_source_sha256,
        "bundle_source_sha256": bundle_source_sha256,
        "wave_config_version": WAVE_CONFIG_VERSION,
        "wave_filters": {
            wave: WAVE_BOXES[wave] for wave in LPO_LIFECYCLE_FILTERS
        },
        "wave_sampling_geometry": (
            "saved precipitation-weighted LPO centroid with a cached relative-"
            "longitude stencil; wave filtering does not alter tracking or "
            "event selection"
        ),
        "local_sampling_geometry": LOCAL_SAMPLING_DEFINITION,
        "rain_condition_threshold_mm_day": LIFECYCLE_RAIN_THRESHOLD_MM_DAY,
        "section_cache_version": SECTION_CACHE_VERSION,
        "composite_2d_cache_version": COMPOSITE_CACHE_VERSION,
        "section_default_longitude_half_width_degrees": (
            DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
        ),
        "section_longitude_half_width_sensitivity_degrees": list(
            LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES
        ),
        "section_literature_basis": (
            "Lim and Son lag-height/lag-latitude presentation adapted from "
            "longitude to LPO lag; Rios-Berrios filtered-wave phase metadata "
            "is diagnostic only and is not used for unfiltered composites"
        ),
        "cloud_context_geometry": (
            "broad event-centred saved context; cloud classes are not "
            "truncated to the LPO envelope"
        ),
        "event_weighting": "one equal weight per available LPO event",
        "absolute_environment": True,
        "section_composite_fields": "absolute sampled model values",
        "section_response_definition": (
            "independent-event experiment composite minus control composite"
        ),
        "ttl_pressure_bounds_hpa": list(TTL_PRESSURE_BOUNDS_HPA),
        "ttl_height_bounds_km": list(TTL_HEIGHT_BOUNDS_KM),
        "ttl_reaching_cloud_top_pressure_hpa": TTL_BASE_PRESSURE_HPA,
        "ttl_reference": TTL_REFERENCE,
        "environment_anomaly_baseline": environment_anomaly_baseline,
        "total_high_cloud_is_primary": True,
        "cloud_classes_are_attribution_terms": True,
        "lifecycle_step_columns": list(lifecycle.columns),
        "lifecycle_summary_columns": list(summary.columns),
        "vertical_profile_columns": list(profiles.columns),
        "test_event_ids": list(test_event_ids or ()),
        "test_product": test_event_ids is not None,
        "output_sha256": {
            "steps": _file_sha256(standard / LIFECYCLE_STEP_FILENAME),
            "summary": _file_sha256(standard / LIFECYCLE_SUMMARY_FILENAME),
            "profiles": _file_sha256(standard / LIFECYCLE_PROFILE_FILENAME),
        },
    }
    path = standard / LIFECYCLE_CONFIGURATION_FILENAME
    _write_lifecycle_metadata_file(path, metadata)


def _write_lifecycle_metadata_file(
    path: Path,
    metadata: Mapping[str, object],
) -> None:
    """Write lifecycle metadata with a hash of its configuration fields."""

    output = dict(metadata)
    output.pop("configuration_sha256", None)
    encoded = json.dumps(
        output, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    output["configuration_sha256"] = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_run_composite(
    run_directory: Path,
    lifecycle: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    test: bool,
) -> None:
    """Write aggregate lifecycle products from saved event-level tables."""

    missing = sorted(
        set(LIFECYCLE_COMPOSITE_VARIABLES) - set(lifecycle.columns)
    )
    if missing:
        raise RuntimeError(
            f"{run_directory}: lifecycle schema lacks required fields {missing}"
        )
    composite_input = lifecycle.copy()
    if test:
        composite_input["test_lifecycle_sample"] = True
    composite = bootstrap_event_composite(
        composite_input,
        LIFECYCLE_COMPOSITE_VARIABLES,
    )
    composite.attrs.update(
        {
            "lifecycle_analysis_version": LIFECYCLE_ANALYSIS_VERSION,
            "tracking_product_version": PRODUCT_VERSION,
            "wave_config_version": WAVE_CONFIG_VERSION,
            "wave_filters": json.dumps(
                {wave: WAVE_BOXES[wave] for wave in LPO_LIFECYCLE_FILTERS},
                sort_keys=True,
            ),
            "event_weighting": "one equal weight per LPO event",
            "local_sampling_geometry": LOCAL_SAMPLING_DEFINITION,
            "rain_condition_threshold_mm_day": LIFECYCLE_RAIN_THRESHOLD_MM_DAY,
            "test_partial_sequences_allowed": bool(test),
        }
    )
    composite_path = run_directory / LIFECYCLE_COMPOSITE_FILENAME
    write_netcdf_atomic(composite, composite_path)
    write_csv_atomic(
        summary,
        run_directory / LIFECYCLE_SUMMARY_FILENAME,
        index=False,
    )


def _section_latitude_slice(
    latitude: xr.DataArray,
    event_slice: slice,
) -> slice:
    """Include every event grid plus the full section-sampling latitude band."""

    values = np.asarray(latitude.values, dtype=float)
    event_values = values[event_slice]
    lower = min(float(np.nanmin(event_values)), -35.0)
    upper = max(float(np.nanmax(event_values)), 35.0)
    indices = np.flatnonzero((values >= lower) & (values <= upper))
    if not indices.size:
        raise RuntimeError(
            "MPAS lifecycle latitude grid does not cover the required "
            "35S--35N section domain"
        )
    return slice(int(indices.min()), int(indices.max()) + 1)


def _atomic_write_netcdf(dataset: xr.Dataset, path: Path) -> None:
    write_netcdf_atomic(dataset, path)


def _event_section_cache_is_current(
    path: Path,
    *,
    simulation: str,
    test: bool,
    tracking_timing_path: Path,
    calculation_source_sha256: str | None = None,
    force_recompute: bool = False,
) -> bool:
    if force_recompute:
        return False
    if not path.is_file() or path.stat().st_mtime < tracking_timing_path.stat().st_mtime:
        return False
    try:
        with xr.open_dataset(path) as cached:
            return (
                cached.attrs.get("cache_version") == SECTION_CACHE_VERSION
                and cached.attrs.get("lifecycle_analysis_version")
                == LIFECYCLE_ANALYSIS_VERSION
                and cached.attrs.get("tracking_product_version")
                == PRODUCT_VERSION
                and cached.attrs.get("source_simulation") == simulation
                and (
                    calculation_source_sha256 is None
                    or cached.attrs.get("calculation_source_sha256")
                    == calculation_source_sha256
                )
                and int(cached.attrs.get("test_subset", -1)) == int(test)
                and "cloud_class_indicator" in cached
                and "background_zonal_wind" in cached
                and "native_pressure_hpa" in cached
                and "unfiltered_temperature" in cached
                and "relative_longitude" in cached.sizes
                and "ray_unfiltered_static_stability" in cached
                and "matsuno_relative_vorticity_anomaly" in cached
            )
    except (OSError, ValueError):
        return False


def _align_wave_sections_to_unfiltered_events(
    wave_sections: xr.Dataset,
    event_sections: xr.Dataset,
) -> xr.Dataset:
    """Use identical event/lag/time samples in filtered and unfiltered means."""

    wave_keys = {
        (
            str(event_id),
            int(lag),
            pd.Timestamp(time_value),
        ): index
        for index, (event_id, lag, time_value) in enumerate(
            zip(
                wave_sections.event_id.values,
                wave_sections.lag_hours.values,
                wave_sections.Time.values,
            )
        )
    }
    requested = [
        (str(event_id), int(lag), pd.Timestamp(time_value))
        for event_id, lag, time_value in zip(
            event_sections.event_id.values,
            event_sections.lag_hours.values,
            event_sections.Time.values,
        )
    ]
    missing = [key for key in requested if key not in wave_keys]
    if missing:
        raise RuntimeError(
            "Compact wave cache does not contain the authoritative LPO "
            f"event/lag samples; first missing keys are {missing[:3]}"
        )
    positions = [wave_keys[key] for key in requested]
    return wave_sections.isel(
        event_step=xr.DataArray(positions, dims="event_step")
    ).assign_coords(event_step=np.arange(len(positions)))


def _add_conditioned_static_stability(
    composite: xr.Dataset,
    kelvin_reference: xr.Dataset,
) -> xr.Dataset:
    """Subtract the unconditional Figure-13 N2 profile from peak LPO N2."""

    primary = kelvin_reference.sel(
        zonal_wavenumber=1, vertical_wavelength_km=10.0
    )
    backgrounds = {
        "ray_unconditional_background_zonal_wind": [],
        "ray_unconditional_buoyancy_frequency": [],
        "ray_unconditional_static_stability": [],
        "ray_unconditional_pressure_hpa": [],
    }
    source_names = {
        "ray_unconditional_background_zonal_wind": "background_zonal_wind",
        "ray_unconditional_buoyancy_frequency": "buoyancy_frequency",
        "ray_unconditional_static_stability": "static_stability",
        "ray_unconditional_pressure_hpa": "background_pressure_hpa",
    }
    for simulation in SIMULATION_DIRS:
        selected = primary.sel(
            simulation=WAVE_SIMULATION_LABELS[simulation]
        )
        for output_name, source_name in source_names.items():
            source = selected[source_name].interp(
                height_m=np.asarray(composite.ray_height_m, dtype=float)
            )
            backgrounds[output_name].append(
                source.rename({"height_m": "ray_height_m"}).reset_coords(
                    drop=True
                )
            )
    for output_name, profiles in backgrounds.items():
        composite[output_name] = xr.concat(
            profiles,
            dim=xr.IndexVariable(
                "simulation", np.asarray(list(SIMULATION_DIRS), dtype=str)
            ),
        )
    composite["ray_lpo_conditioned_static_stability_anomaly"] = (
        composite.ray_unfiltered_static_stability
        - composite.ray_unconditional_static_stability
    )
    composite.ray_unconditional_background_zonal_wind.attrs.update(
        units="m s-1",
        definition="unconditional tropical-mean U from equatorial wave Figure 13",
    )
    composite.ray_unconditional_buoyancy_frequency.attrs.update(
        units="s-1",
        definition="unconditional tropical-mean N from equatorial wave Figure 13",
    )
    composite.ray_unconditional_static_stability.attrs.update(
        units="s-2",
        definition=(
            "unconditional tropical-mean static stability from equatorial "
            "wave Figure 13"
        ),
    )
    composite.ray_unconditional_pressure_hpa.attrs.update(
        units="hPa",
        definition=(
            "unconditional tropical-mean pressure from equatorial wave Figure 13"
        ),
    )
    composite.ray_lpo_conditioned_static_stability_anomaly.attrs.update(
        units="s-2",
        definition="lag-zero LPO-conditioned N2 minus Figure-13 background N2",
    )
    return composite


def analyse_run_lifecycles(
    run_directory: Path,
    *,
    test: bool = False,
    force_recompute_caches: bool = False,
    simulations: Sequence[str] | None = None,
    finalize: bool = True,
) -> None:
    """Build lifecycle-owned products from one tracking run directory."""

    selected_simulations = tuple(simulations or SIMULATION_DIRS)
    unknown_simulations = sorted(
        set(selected_simulations) - set(SIMULATION_DIRS)
    )
    if unknown_simulations:
        raise ValueError(f"Unknown lifecycle simulations: {unknown_simulations}")
    if finalize and set(selected_simulations) != set(SIMULATION_DIRS):
        raise ValueError("Lifecycle finalization requires every simulation")
    combined_lifecycle: list[pd.DataFrame] = []
    combined_summary: list[pd.DataFrame] = []
    build_two_dimensional = run_directory.name == "primary"
    simulation_composites: dict[str, xr.Dataset] = {}
    section_paths: dict[str, Path] = {}
    kelvin_reference = (
        load_existing_kelvin_dispersion(test=test)
        if build_two_dimensional and finalize
        else None
    )
    for simulation in selected_simulations:
        standard = run_directory / simulation / "standard"
        timing_path = standard / "lpt_event_relative_diagnostics.csv"
        event_path = standard / "lpt_events.csv"
        if not timing_path.is_file() or not event_path.is_file():
            raise FileNotFoundError(
                f"Missing stable tracking products below {standard}"
            )
        tracking_timing = pd.read_csv(timing_path, dtype={"event_id": str})
        tracking_timing["Time"] = pd.to_datetime(tracking_timing.Time)
        events = pd.read_csv(event_path, dtype={"event_id": str})
        if test:
            event_ids = _select_test_events(events, tracking_timing)
        else:
            event_ids = events.event_id.tolist()
        selected_timing = tracking_timing[
            tracking_timing.event_id.isin(event_ids)
        ].copy()
        geometric_path = standard / "lpt_track_steps.csv"
        if not geometric_path.is_file():
            raise FileNotFoundError(
                f"Missing geometric LPT track steps {geometric_path}"
            )
        geometric = pd.read_csv(
            geometric_path,
            dtype={"event_id": str},
            usecols=[
                "event_id", "Time", "centroid_lon", "centroid_lat",
            ],
        )
        geometric["Time"] = pd.to_datetime(geometric.Time)
        selected_timing = selected_timing.merge(
            geometric,
            on=["event_id", "Time"],
            how="left",
            validate="one_to_one",
        )
        if selected_timing[["centroid_lon", "centroid_lat"]].isna().any().any():
            raise RuntimeError(
                f"{standard}: lifecycle times lack geometric centroids"
            )
        if test:
            environment_anomaly_baseline = (
                "selected actual-data LPO diagnostic times; test figure "
                "smoke-test product only"
            )
        else:
            environment_anomaly_baseline = "complete simulation record"
        dependency_hashes = _lifecycle_dependency_hashes(
            standard, simulation, event_ids, test=test
        )
        bundle = _read_simulation_bundle(
            standard,
            simulation=simulation,
            event_ids=event_ids,
            bundle_source_sha256=dependency_hashes["bundle"],
            test=test,
            force_recompute=force_recompute_caches,
        )
        if bundle is not None:
            lifecycle, summary, profiles = bundle
            expected_steps = {
                (str(row.event_id), pd.Timestamp(row.Time))
                for row in selected_timing.itertuples()
            }
            cached_steps = set(
                zip(
                    lifecycle.event_id.astype(str),
                    pd.to_datetime(lifecycle.Time),
                )
            )
            if cached_steps != expected_steps:
                bundle = None
            else:
                _cleanup_event_checkpoints(standard)
        section_path = standard / LIFECYCLE_SECTION_FILENAME
        section_current = build_two_dimensional and _event_section_cache_is_current(
            section_path,
            simulation=simulation,
            test=test,
            tracking_timing_path=timing_path,
            calculation_source_sha256=dependency_hashes["calculation"],
            force_recompute=force_recompute_caches,
        )
        event_checkpoints: dict[
            str, tuple[pd.DataFrame, pd.DataFrame]
        ] = {}
        if bundle is None:
            for event_id in event_ids:
                event_timing = selected_timing[
                    selected_timing.event_id == event_id
                ]
                checkpoint = _read_event_checkpoint(
                    standard,
                    simulation=simulation,
                    event_id=event_id,
                    timing_sha256=_event_timing_sha256(event_timing),
                    calculation_source_sha256=dependency_hashes[
                        "calculation"
                    ],
                    test=test,
                    force_recompute=force_recompute_caches,
                )
                if checkpoint is not None:
                    event_checkpoints[event_id] = checkpoint
        missing_event_ids = [
            event_id for event_id in event_ids
            if event_id not in event_checkpoints
        ] if bundle is None else []
        needs_source_data = bool(missing_event_ids) or (
            build_two_dimensional and not section_current
        )
        dataset: xr.Dataset | None = None
        masks: Mapping[str, xr.DataArray] | None = None
        full_precipitation: xr.DataArray | None = None
        classification: Mapping[str, xr.DataArray] | None = None
        if needs_source_data:
            masks = _read_standard_masks(standard)
            progress(
                f"{run_directory.name}/{simulation}: opening lifecycle "
                "source data"
            )
            dataset = _load_simulation_dataset(
                simulation,
                include_lifecycle_fields=True,
            )
            latitude_slice = _selected_mask_latitude_slice(
                dataset.latitude,
                masks,
                event_ids,
            )
            if build_two_dimensional:
                latitude_slice = _section_latitude_slice(
                    dataset.latitude, latitude_slice
                )
            dataset = dataset.isel(latitude=latitude_slice)
            progress(
                f"{run_directory.name}/{simulation}: using LPO latitude "
                f"support {float(dataset.latitude.min()):.2f} to "
                f"{float(dataset.latitude.max()):.2f} degrees"
            )
            full_precipitation = precipitation_rate_from_dataset(dataset)
            diagnostic_times = pd.DatetimeIndex(
                selected_timing.Time.drop_duplicates().sort_values()
            )
            classification = _load_classification(
                simulation,
                full_precipitation.sel(
                    Time=diagnostic_times.values
                ).Time,
                REQUIRED_CLASSIFICATION_FIELDS,
            )
        if bundle is None:
            if missing_event_ids:
                assert dataset is not None
                assert masks is not None
                assert full_precipitation is not None
                assert classification is not None
                diagnostic_times = pd.DatetimeIndex(
                    selected_timing.Time.drop_duplicates().sort_values()
                )
                precipitation = full_precipitation.sel(
                    Time=diagnostic_times.values
                )
                projection_precipitation = (
                    precipitation if test else full_precipitation
                )
                progress(
                    f"{run_directory.name}/{simulation}: projecting model "
                    "fields and environment baselines"
                )
                model_fields = project_model_fields(
                    dataset,
                    projection_precipitation,
                    progress_label=f"{run_directory.name}/{simulation}",
                )
                progress(
                    f"{run_directory.name}/{simulation}: environment "
                    "baselines complete"
                )
                for event_number, event_id in enumerate(
                    event_ids, start=1
                ):
                    if event_id in event_checkpoints:
                        progress(
                            f"{run_directory.name}/{simulation}: reusing "
                            f"event {event_number}/{len(event_ids)} {event_id}"
                        )
                        continue
                    progress(
                        f"{run_directory.name}/{simulation}: lifecycle "
                        f"event {event_number}/{len(event_ids)} {event_id}"
                    )
                    event_timing = selected_timing[
                        selected_timing.event_id == event_id
                    ].copy()
                    diagnosed = diagnose_event_steps(
                        masks[event_id],
                        precipitation,
                        classification,
                        model_fields,
                        event_id=event_id,
                        simulation=simulation,
                    )
                    progress(
                        f"{run_directory.name}/{simulation}: event "
                        f"{event_number}/{len(event_ids)} horizontal "
                        "diagnostics complete"
                    )
                    steps = _merge_authoritative_timing(
                        diagnosed, event_timing
                    )
                    event_profiles = _sample_native_cloud_profiles(
                        dataset,
                        masks,
                        event_timing,
                        [event_id],
                        simulation,
                    )
                    progress(
                        f"{run_directory.name}/{simulation}: event "
                        f"{event_number}/{len(event_ids)} native profiles "
                        "complete"
                    )
                    _write_event_checkpoint(
                        standard,
                        simulation=simulation,
                        event_id=event_id,
                        timing_sha256=_event_timing_sha256(event_timing),
                        calculation_source_sha256=dependency_hashes[
                            "calculation"
                        ],
                        test=test,
                        steps=steps,
                        profiles=event_profiles,
                    )
                    event_checkpoints[event_id] = (
                        steps, event_profiles
                    )
                    progress(
                        f"{run_directory.name}/{simulation}: event "
                        f"{event_number}/{len(event_ids)} checkpoint saved"
                    )
            event_rows = [
                event_checkpoints[event_id][0] for event_id in event_ids
            ]
            profile_rows = [
                event_checkpoints[event_id][1] for event_id in event_ids
            ]
            lifecycle = pd.concat(event_rows, ignore_index=True)
            profiles = pd.concat(profile_rows, ignore_index=True)
            lifecycle = _add_wave_lifecycle_fields(lifecycle, test=test)
            summary = event_lifecycle_summary(lifecycle)
            summary = summary.merge(
                _wave_event_summary(lifecycle),
                on="event_id",
                how="left",
                validate="one_to_one",
            )
            _require_valid_native_profiles(profiles, lifecycle)
            _atomic_write_csv(
                lifecycle, standard / LIFECYCLE_STEP_FILENAME
            )
            _atomic_write_csv(
                summary, standard / LIFECYCLE_SUMMARY_FILENAME
            )
            _atomic_write_csv(
                profiles, standard / LIFECYCLE_PROFILE_FILENAME
            )
            _write_lifecycle_metadata(
                standard,
                simulation=simulation,
                event_ids=event_ids,
                calculation_source_sha256=dependency_hashes["calculation"],
                bundle_source_sha256=dependency_hashes["bundle"],
                lifecycle=lifecycle,
                summary=summary,
                profiles=profiles,
                test_event_ids=event_ids if test else None,
                environment_anomaly_baseline=environment_anomaly_baseline,
            )
            validated_bundle = _read_simulation_bundle(
                standard,
                simulation=simulation,
                event_ids=event_ids,
                bundle_source_sha256=dependency_hashes["bundle"],
                test=test,
                force_recompute=False,
            )
            if validated_bundle is None:
                raise RuntimeError(
                    f"{run_directory.name}/{simulation}: final lifecycle "
                    "bundle failed validation; checkpoints remain for restart"
                )
            _cleanup_event_checkpoints(standard)
        else:
            progress(
                f"{run_directory.name}/{simulation}: reusing complete "
                "lifecycle table bundle"
            )
        if build_two_dimensional:
            section_paths[simulation] = section_path
            if section_current:
                progress(
                    f"{run_directory.name}/{simulation}: reusing event "
                    "section cache"
                )
                with xr.open_dataset(section_path) as saved_sections:
                    event_sections = saved_sections.load()
            else:
                assert dataset is not None
                assert full_precipitation is not None
                assert classification is not None
                progress(
                    f"{run_directory.name}/{simulation}: building broad "
                    "event sections and cloud context"
                )
                event_sections = build_unfiltered_event_sections(
                    dataset,
                    full_precipitation,
                    classification,
                    selected_timing,
                    primary=run_directory,
                    simulation=simulation,
                    test=test,
                )
                event_sections.attrs.update(
                    lifecycle_analysis_version=LIFECYCLE_ANALYSIS_VERSION,
                    tracking_product_version=PRODUCT_VERSION,
                    calculation_source_sha256=dependency_hashes[
                        "calculation"
                    ],
                )
                _atomic_write_netcdf(event_sections, section_path)
                progress(
                    f"{run_directory.name}/{simulation}: wrote event "
                    f"section cache {section_path}"
                )
            if finalize:
                assert kelvin_reference is not None
                progress(
                    f"{run_directory.name}/{simulation}: building "
                    "simulation event composites"
                )
                wave_sections = load_existing_lpo_wave_sections(
                    simulation,
                    tracking_product_version=PRODUCT_VERSION,
                    test=test,
                )
                wave_sections = _align_wave_sections_to_unfiltered_events(
                    wave_sections, event_sections
                )
                simulation_composites[simulation] = (
                    build_simulation_composites(
                        event_sections,
                        wave_sections,
                        primary_k_m1=float(
                            kelvin_reference.attrs["primary_k_m-1"]
                        ),
                        primary_m_m1=float(
                            kelvin_reference.attrs["primary_m_m-1"]
                        ),
                        test=test,
                        background_profile=kelvin_reference.sel(
                            simulation=WAVE_SIMULATION_LABELS[simulation],
                            zonal_wavenumber=1,
                            vertical_wavelength_km=10.0,
                        ),
                    )
                )
                progress(
                    f"{run_directory.name}/{simulation}: simulation event "
                    "composites complete"
                )
        combined_lifecycle.append(lifecycle)
        combined_summary.append(summary)
        if dataset is not None:
            dataset.close()

    if not finalize:
        progress(
            f"{run_directory.name}: selected simulation calculation "
            "complete; shared composites remain for the final run"
        )
        return
    lifecycle = pd.concat(combined_lifecycle, ignore_index=True)
    summary = pd.concat(combined_summary, ignore_index=True)
    progress(f"{run_directory.name}: building event-equal lifecycle composite")
    _write_run_composite(
        run_directory,
        lifecycle,
        summary,
        test=test,
    )
    progress(f"{run_directory.name}: event-equal lifecycle composite complete")
    if build_two_dimensional:
        assert kelvin_reference is not None
        progress(f"{run_directory.name}: combining two-dimensional composites")
        composite_2d = combine_simulation_composites(
            simulation_composites
        )
        composite_2d = _add_conditioned_static_stability(
            composite_2d, kelvin_reference
        )
        progress(
            f"{run_directory.name}: calculating two-dimensional response "
            "uncertainty"
        )
        response_composites = build_review_response_composites(
            section_paths,
            wave_section_paths={
                simulation: Path(
                    lpo_wave_section_cache_path(simulation, test=test)
                )
                for simulation in SIMULATION_DIRS
            },
            lags=np.asarray(composite_2d.lag_hours, dtype=int),
            primary_k_m1=float(kelvin_reference.attrs["primary_k_m-1"]),
            primary_m_m1=float(kelvin_reference.attrs["primary_m_m-1"]),
        )
        composite_2d = xr.merge(
            [composite_2d, response_composites],
            compat="no_conflicts",
        )
        composite_2d.attrs.update(
            lifecycle_analysis_version=LIFECYCLE_ANALYSIS_VERSION,
            tracking_product_version=PRODUCT_VERSION,
            wave_config_version=WAVE_CONFIG_VERSION,
        )
        _atomic_write_netcdf(
            composite_2d,
            run_directory / LIFECYCLE_2D_COMPOSITE_FILENAME,
        )
        progress(
            f"{run_directory.name}: wrote combined two-dimensional "
            "lifecycle composite"
        )


def require_lifecycle_products(run_directory: Path) -> None:
    missing: list[str] = []
    require_sections = run_directory.name == "primary"
    for simulation in SIMULATION_DIRS:
        standard = run_directory / simulation / "standard"
        dependency_hashes: dict[str, str] | None = None
        for filename in (
            LIFECYCLE_STEP_FILENAME,
            LIFECYCLE_SUMMARY_FILENAME,
            LIFECYCLE_PROFILE_FILENAME,
            LIFECYCLE_CONFIGURATION_FILENAME,
            *((LIFECYCLE_SECTION_FILENAME,) if require_sections else ()),
        ):
            path = standard / filename
            if not path.is_file():
                missing.append(str(path))
        metadata_path = standard / LIFECYCLE_CONFIGURATION_FILENAME
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not _metadata_hash_is_valid(metadata):
                raise RuntimeError(
                    f"{metadata_path} has invalid configuration metadata"
                )
            if metadata.get("lifecycle_analysis_version") != LIFECYCLE_ANALYSIS_VERSION:
                raise RuntimeError(
                    f"{metadata_path} predates {LIFECYCLE_ANALYSIS_VERSION}"
                )
            if metadata.get("tracking_product_version") != PRODUCT_VERSION:
                raise RuntimeError(
                    f"{metadata_path} does not match {PRODUCT_VERSION}"
                )
            if set(metadata.get("wave_filters", {})) != set(
                LPO_LIFECYCLE_FILTERS
            ):
                raise RuntimeError(
                    f"{metadata_path} does not contain the configured "
                    "lifecycle wave filters"
                )
            event_ids = metadata.get("selected_event_ids")
            if not isinstance(event_ids, list) or not event_ids:
                raise RuntimeError(
                    f"{metadata_path} lacks its selected event identities"
                )
            dependency_hashes = _lifecycle_dependency_hashes(
                standard,
                simulation,
                event_ids,
                test=bool(metadata.get("test_product", False)),
            )
            if (
                metadata.get("calculation_source_sha256")
                != dependency_hashes["calculation"]
                or metadata.get("bundle_source_sha256")
                != dependency_hashes["bundle"]
            ):
                raise RuntimeError(
                    f"{metadata_path} has stale lifecycle dependencies"
                )
            product_paths = {
                "steps": standard / LIFECYCLE_STEP_FILENAME,
                "summary": standard / LIFECYCLE_SUMMARY_FILENAME,
                "profiles": standard / LIFECYCLE_PROFILE_FILENAME,
            }
            if all(path.is_file() for path in product_paths.values()):
                expected_hashes = {
                    key: _file_sha256(path)
                    for key, path in product_paths.items()
                }
                if metadata.get("output_sha256") != expected_hashes:
                    raise RuntimeError(
                        f"{metadata_path} does not match its lifecycle tables"
                    )
        section_path = standard / LIFECYCLE_SECTION_FILENAME
        if require_sections and section_path.is_file():
            with xr.open_dataset(section_path) as section:
                if section.attrs.get("cache_version") != SECTION_CACHE_VERSION:
                    raise RuntimeError(
                        f"{section_path} predates {SECTION_CACHE_VERSION}"
                    )
                if (
                    section.attrs.get("tracking_product_version")
                    != PRODUCT_VERSION
                ):
                    raise RuntimeError(
                        f"{section_path} does not match {PRODUCT_VERSION}"
                    )
                if (
                    dependency_hashes is None
                    or section.attrs.get("calculation_source_sha256")
                    != dependency_hashes["calculation"]
                ):
                    raise RuntimeError(
                        f"{section_path} has stale lifecycle dependencies"
                    )
    if require_sections:
        composite_path = run_directory / LIFECYCLE_2D_COMPOSITE_FILENAME
        if not composite_path.is_file():
            missing.append(str(composite_path))
        else:
            with xr.open_dataset(composite_path) as composite:
                if composite.attrs.get("cache_version") != COMPOSITE_CACHE_VERSION:
                    raise RuntimeError(
                        f"{composite_path} predates {COMPOSITE_CACHE_VERSION}"
                    )
                if (
                    composite.attrs.get("tracking_product_version")
                    != PRODUCT_VERSION
                ):
                    raise RuntimeError(
                        f"{composite_path} does not match {PRODUCT_VERSION}"
                    )
    if missing:
        raise FileNotFoundError(
            "Missing LPO lifecycle products:\n"
            + "\n".join(f"  {path}" for path in missing)
        )


def read_lifecycle_products(
    run_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_lifecycle_products(run_directory)
    lifecycle = pd.concat(
        [
            pd.read_csv(
                run_directory
                / simulation
                / "standard"
                / LIFECYCLE_STEP_FILENAME
            )
            for simulation in SIMULATION_DIRS
        ],
        ignore_index=True,
    )
    summary = pd.concat(
        [
            pd.read_csv(
                run_directory
                / simulation
                / "standard"
                / LIFECYCLE_SUMMARY_FILENAME
            )
            for simulation in SIMULATION_DIRS
        ],
        ignore_index=True,
    )
    return lifecycle, summary


def _test_event_products_are_current(run_directory: Path) -> bool:
    """Validate reusable test tables and event-section caches."""

    for simulation in SIMULATION_DIRS:
        standard = run_directory / simulation / "standard"
        required = (
            LIFECYCLE_STEP_FILENAME,
            LIFECYCLE_SUMMARY_FILENAME,
            LIFECYCLE_PROFILE_FILENAME,
            LIFECYCLE_CONFIGURATION_FILENAME,
            LIFECYCLE_SECTION_FILENAME,
            "lpt_event_relative_diagnostics.csv",
            "lpt_events.csv",
        )
        if any(not (standard / filename).is_file() for filename in required):
            return False
        metadata_path = standard / LIFECYCLE_CONFIGURATION_FILENAME
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if (
            metadata.get("lifecycle_analysis_version")
            != LIFECYCLE_ANALYSIS_VERSION
            or metadata.get("tracking_product_version") != PRODUCT_VERSION
            or not metadata.get("test_product", False)
            or not _metadata_hash_is_valid(metadata)
        ):
            return False
        event_ids = metadata.get("selected_event_ids")
        if not isinstance(event_ids, list) or not event_ids:
            return False
        try:
            dependency_hashes = _lifecycle_dependency_hashes(
                standard, simulation, event_ids, test=True
            )
        except (OSError, RuntimeError, FileNotFoundError, ValueError):
            return False
        if _read_simulation_bundle(
            standard,
            simulation=simulation,
            event_ids=event_ids,
            bundle_source_sha256=dependency_hashes["bundle"],
            test=True,
            force_recompute=False,
        ) is None:
            return False
        for tracking_filename in (
            "lpt_event_relative_diagnostics.csv",
            "lpt_events.csv",
        ):
            if (
                standard / tracking_filename
            ).stat().st_mtime > metadata_path.stat().st_mtime:
                return False
        try:
            with xr.open_dataset(
                standard / LIFECYCLE_SECTION_FILENAME
            ) as section:
                if (
                    section.attrs.get("cache_version") != SECTION_CACHE_VERSION
                    or section.attrs.get("tracking_product_version")
                    != PRODUCT_VERSION
                    or section.attrs.get("calculation_source_sha256")
                    != dependency_hashes["calculation"]
                ):
                    return False
            load_existing_lpo_wave_sections(
                simulation,
                tracking_product_version=PRODUCT_VERSION,
                test=True,
            ).close()
        except (OSError, RuntimeError, FileNotFoundError, ValueError):
            return False
    return True


def _rebuild_test_2d_composite_from_saved_sections(
    run_directory: Path,
) -> None:
    """Rebuild a test aggregate without reopening the source simulations."""

    kelvin_reference = load_existing_kelvin_dispersion(test=True)
    primary_k_m1 = float(kelvin_reference.attrs["primary_k_m-1"])
    primary_m_m1 = float(kelvin_reference.attrs["primary_m_m-1"])
    simulation_composites: dict[str, xr.Dataset] = {}
    section_paths: dict[str, Path] = {}
    for simulation in SIMULATION_DIRS:
        section_path = (
            run_directory
            / simulation
            / "standard"
            / LIFECYCLE_SECTION_FILENAME
        )
        section_paths[simulation] = section_path
        with xr.open_dataset(section_path) as saved_sections:
            event_sections = saved_sections.load()
        wave_sections = load_existing_lpo_wave_sections(
            simulation,
            tracking_product_version=PRODUCT_VERSION,
            test=True,
        )
        try:
            aligned_wave_sections = _align_wave_sections_to_unfiltered_events(
                wave_sections, event_sections
            )
            simulation_composites[simulation] = build_simulation_composites(
                event_sections,
                aligned_wave_sections,
                primary_k_m1=primary_k_m1,
                primary_m_m1=primary_m_m1,
                test=True,
                background_profile=kelvin_reference.sel(
                    simulation=WAVE_SIMULATION_LABELS[simulation],
                    zonal_wavenumber=1,
                    vertical_wavelength_km=10.0,
                ),
            )
        finally:
            wave_sections.close()
            event_sections.close()
    composite_2d = combine_simulation_composites(simulation_composites)
    composite_2d = _add_conditioned_static_stability(
        composite_2d, kelvin_reference
    )
    response_composites = build_review_response_composites(
        section_paths,
        wave_section_paths={
            simulation: Path(
                lpo_wave_section_cache_path(simulation, test=True)
            )
            for simulation in SIMULATION_DIRS
        },
        lags=np.asarray(composite_2d.lag_hours, dtype=int),
        primary_k_m1=primary_k_m1,
        primary_m_m1=primary_m_m1,
    )
    composite_2d = xr.merge(
        [composite_2d, response_composites], compat="no_conflicts"
    )
    composite_2d.attrs.update(
        lifecycle_analysis_version=LIFECYCLE_ANALYSIS_VERSION,
        tracking_product_version=PRODUCT_VERSION,
        wave_config_version=WAVE_CONFIG_VERSION,
    )
    _atomic_write_netcdf(
        composite_2d,
        run_directory / LIFECYCLE_2D_COMPOSITE_FILENAME,
    )
    kelvin_reference.close()


def _synchronize_test_composite_metadata(run_directory: Path) -> None:
    """Record the aggregate cache version in each test configuration."""

    for simulation in SIMULATION_DIRS:
        path = (
            run_directory
            / simulation
            / "standard"
            / LIFECYCLE_CONFIGURATION_FILENAME
        )
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("composite_2d_cache_version") == COMPOSITE_CACHE_VERSION:
            continue
        metadata["composite_2d_cache_version"] = COMPOSITE_CACHE_VERSION
        _write_lifecycle_metadata_file(path, metadata)


def reuse_test_lifecycle_products(run_directory: Path) -> bool:
    """Rebuild test aggregates from current saved event products if possible."""

    if not _test_event_products_are_current(run_directory):
        return False
    composite_2d_path = run_directory / LIFECYCLE_2D_COMPOSITE_FILENAME
    rebuild_2d = not composite_2d_path.is_file()
    try:
        if not rebuild_2d:
            with xr.open_dataset(composite_2d_path) as composite_2d:
                rebuild_2d = (
                    composite_2d.attrs.get("cache_version")
                    != COMPOSITE_CACHE_VERSION
                    or composite_2d.attrs.get("tracking_product_version")
                    != PRODUCT_VERSION
                    or "ray_lpo_conditioned_static_stability_anomaly"
                    not in composite_2d
                )
    except OSError:
        rebuild_2d = True
    if rebuild_2d:
        progress(
            f"{run_directory.name}: rebuilding current test 2-D aggregate "
            "from saved event sections"
        )
        _rebuild_test_2d_composite_from_saved_sections(run_directory)
    _synchronize_test_composite_metadata(run_directory)
    try:
        require_lifecycle_products(run_directory)
    except (FileNotFoundError, RuntimeError):
        return False
    progress(
        f"{run_directory.name}: reusing current saved test event products"
    )
    lifecycle, summary = read_lifecycle_products(run_directory)
    _write_run_composite(
        run_directory,
        lifecycle,
        summary,
        test=True,
    )
    return True


def _plot_composite_grid(
    lifecycle: pd.DataFrame,
    variables: Sequence[tuple[str, str]],
    title: str,
):
    display_scale = {
        "area_integrated_24h_precip_km2_mm": (
            RAIN_VOLUME_DISPLAY_SCALE
        ),
        "envelope_area_km2": 1.0e-6,
        "total_high_cloud_area_km2": 1.0e-6,
        "active_deep_convective_area_km2": 1.0e-6,
        "stratiform_precipitating_high_cloud_area_km2": 1.0e-6,
        "nonprecipitating_high_cloud_area_km2": 1.0e-6,
    }
    composite = bootstrap_event_composite(
        lifecycle, [name for name, _label in variables]
    )
    fig, axes = plt.subplots(
        len(variables),
        2,
        figsize=(11, 2.25 * len(variables)),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
    )
    lags = composite.lag_hours
    for row, (variable, label) in enumerate(variables):
        scale = display_scale.get(variable, 1.0)
        for simulation in SIMULATION_DIRS:
            mean = composite.event_equal_mean.sel(
                simulation=simulation, variable=variable
            )
            low = composite.ci_low.sel(
                simulation=simulation, variable=variable
            )
            high = composite.ci_high.sel(
                simulation=simulation, variable=variable
            )
            count = int(composite.event_count.sel(simulation=simulation))
            axes[row, 0].plot(
                lags,
                scale * mean,
                color=SIMULATION_COLOURS[simulation],
                label=f"{SIMULATION_LABELS[simulation]} (n={count})",
            )
            axes[row, 0].fill_between(
                lags, scale * low, scale * high,
                color=SIMULATION_COLOURS[simulation], alpha=0.12
            )
            if simulation != "control":
                contrast = composite.difference_mean.sel(
                    contrast=simulation, variable=variable
                )
                contrast_low = composite.difference_ci_low.sel(
                    contrast=simulation, variable=variable
                )
                contrast_high = composite.difference_ci_high.sel(
                    contrast=simulation, variable=variable
                )
                axes[row, 1].plot(
                    lags,
                    scale * contrast,
                    color=SIMULATION_COLOURS[simulation],
                    label=f"{SIMULATION_LABELS[simulation]} − control",
                )
                axes[row, 1].fill_between(
                    lags,
                    scale * contrast_low,
                    scale * contrast_high,
                    color=SIMULATION_COLOURS[simulation],
                    alpha=0.12,
                )
        axes[row, 0].set_ylabel(label)
        axes[row, 1].axhline(0, color="0.5", lw=0.7)
        for axis in axes[row]:
            axis.axvline(0, color="0.5", ls="--", lw=0.7)
            axis.grid(alpha=0.2)
    axes[0, 0].set_title("Composite")
    axes[0, 1].set_title("Difference")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[-1, 0].set_xlabel(LIFECYCLE_LAG_LABEL)
    axes[-1, 1].set_xlabel(LIFECYCLE_LAG_LABEL)
    fig.suptitle(title)
    return fig


def _plot_lifecycle(lifecycle: pd.DataFrame):
    figure = _plot_composite_grid(
        lifecycle,
        (
            (
                "area_integrated_24h_precip_km2_mm",
                RAIN_VOLUME_LABEL,
            ),
            (
                "total_high_cloud_area_km2",
                r"Total high-cloud area ($10^6$ km$^2$)",
            ),
            (
                "active_deep_convective_area_km2",
                r"Active deep-convective area ($10^6$ km$^2$)",
            ),
            (
                "stratiform_precipitating_high_cloud_area_km2",
                r"Stratiform high-cloud area ($10^6$ km$^2$)",
            ),
            (
                "nonprecipitating_high_cloud_area_km2",
                (
                    "Nonprecipitating high-cloud area "
                    r"(anvil/cirriform proxy; $10^6$ km$^2$)"
                ),
            ),
            ("olr_w_m2", r"OLR (W m$^{-2}$)"),
        ),
        "LPO lifecycle",
    )
    return figure


def _plot_lifecycle_timing(summary: pd.DataFrame):
    """Show the four main lifecycle milestones as a compact forest plot."""

    metrics = (
        ("active_peak_lag_hours", "Active-core peak"),
        ("stratiform_peak_lag_hours", "Stratiform peak"),
        (
            "nonprecip_high_cloud_peak_lag_hours",
            "Non-precipitating high-cloud peak",
        ),
        (
            "ttl_reaching_fraction_peak_lag_hours",
            "TTL-reaching fraction peak",
        ),
    )
    fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    rng = np.random.default_rng(TIMING_BOOTSTRAP_SEED)
    offsets = np.linspace(-0.22, 0.22, len(SIMULATION_DIRS))
    positions = np.arange(len(metrics), dtype=float)[::-1]
    for metric_index, ((column, _title), position) in enumerate(
        zip(metrics, positions, strict=True)
    ):
        for simulation_index, simulation in enumerate(SIMULATION_DIRS):
            subset = summary.loc[summary.simulation == simulation]
            values = np.asarray(subset[column].dropna(), dtype=float)
            y = position + offsets[simulation_index]
            if len(values):
                median = float(np.median(values))
                bootstrap_median = np.asarray(
                    [
                        np.median(
                            rng.choice(values, size=len(values), replace=True)
                        )
                        for _ in range(TIMING_BOOTSTRAP_RESAMPLES)
                    ]
                )
                ci_low, ci_high = np.percentile(
                    bootstrap_median, [2.5, 97.5]
                )
                axis.errorbar(
                    median,
                    y,
                    xerr=[[median - ci_low], [ci_high - median]],
                    fmt="o",
                    color=SIMULATION_COLOURS[simulation],
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    capsize=2.0,
                    label=(
                        SIMULATION_LABELS[simulation]
                        if metric_index == 0 else None
                    ),
                    zorder=4,
                )
            censored_column = f"{column}_censored"
            rejected = (
                int(subset[censored_column].fillna(False).astype(bool).sum())
                if censored_column in subset
                else int(len(subset) - len(values))
            )
            axis.text(
                0.995,
                y,
                f"n={len(values)}, r={rejected}",
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=6.2,
                color=SIMULATION_COLOURS[simulation],
            )
    axis.set(
        yticks=positions,
        yticklabels=[label for _column, label in metrics],
        xlim=(-78, 78),
        xlabel=LIFECYCLE_LAG_LABEL,
    )
    axis.axvline(0, color="0.5", lw=0.7)
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, loc="lower left")
    axis.text(
        0.995,
        1.01,
        "n: retained; r: edge-rejected",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    fig.suptitle("LPO lifecycle milestones")
    return fig


def _plot_normalized_lifetime(lifecycle: pd.DataFrame):
    variables = (
        (
            "area_integrated_24h_precip_km2_mm",
            RAIN_VOLUME_TITLE,
            RAIN_VOLUME_UNITS,
            RAIN_VOLUME_DISPLAY_SCALE,
        ),
        (
            "precipitable_water_kg_m2",
            "Precipitable water",
            r"kg m$^{-2}$",
            1.0,
        ),
        ("olr_w_m2", "Outgoing longwave radiation", r"W m$^{-2}$", 1.0),
        (
            "active_deep_convective_area_km2",
            "Active cloud area",
            r"$10^6$ km$^2$",
            1.0e-6,
        ),
        (
            "stratiform_precipitating_high_cloud_area_km2",
            "Stratiform cloud area",
            r"$10^6$ km$^2$",
            1.0e-6,
        ),
        (
            "nonprecipitating_high_cloud_area_km2",
            "Nonprecipitating high-cloud area",
            r"$10^6$ km$^2$",
            1.0e-6,
        ),
    )
    composite = normalized_lifetime_composite(
        lifecycle, [name for name, _title, _unit, _scale in variables]
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(14, 7.5), constrained_layout=True, sharex=True
    )
    for axis, (variable, title, unit, scale) in zip(axes.flat, variables):
        for simulation in SIMULATION_DIRS:
            mean = composite["mean"].sel(
                simulation=simulation, variable=variable
            ) * scale
            low = composite.ci_low.sel(
                simulation=simulation, variable=variable
            ) * scale
            high = composite.ci_high.sel(
                simulation=simulation, variable=variable
            ) * scale
            count = int(composite.event_count.sel(simulation=simulation))
            axis.plot(
                composite.lifetime_percent, mean,
                color=SIMULATION_COLOURS[simulation],
                label=f"{SIMULATION_LABELS[simulation]} (n={count})",
            )
            axis.fill_between(
                composite.lifetime_percent, low, high,
                color=SIMULATION_COLOURS[simulation], alpha=0.12,
            )
            peak = float(
                composite.median_precipitation_peak_percent.sel(
                    simulation=simulation
                )
            )
            axis.axvline(
                peak, color=SIMULATION_COLOURS[simulation],
                lw=0.7, ls=":",
            )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Normalized lifetime (%)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Whole-track lifecycle")
    return fig


def _centred_event_field(
    field: xr.DataArray,
    centroid_lon: float,
    centroid_lat: float,
) -> xr.DataArray:
    """Place one event field on the fixed LPO-relative 0.5-degree grid.

    Signed north-south orientation is retained deliberately: tropical LPOs can
    cross the equator, so assigning and reflecting them by a single event
    hemisphere would create an arbitrary discontinuity in their lifecycle.
    """

    if set(field.dims) != {"latitude", "longitude"}:
        raise ValueError(
            "LPO-centred fields require exactly latitude and longitude; "
            f"got {field.dims}"
        )
    field = field.transpose("latitude", "longitude")
    latitude = np.asarray(field.latitude.values, dtype=float)
    longitude = np.asarray(field.longitude.values, dtype=float)
    for name, coordinate in (
        ("latitude", latitude),
        ("longitude", longitude),
    ):
        spacing = np.diff(coordinate)
        if (
            not np.isfinite(spacing).all()
            or not np.allclose(
                spacing,
                MPAS_LPT_GRID_SPACING_DEGREES,
                rtol=0.0,
                atol=1.0e-3,
            )
        ):
            raise ValueError(
                f"LPO-centred {name} must use the fixed "
                f"{MPAS_LPT_GRID_SPACING_DEGREES:g}-degree MPAS grid"
            )
    values = np.asarray(field.values, dtype=float)
    ix = int(
        np.argmin(
            np.abs(
                ((longitude - centroid_lon + 180.0) % 360.0) - 180.0
            )
        )
    )
    iy = int(np.argmin(np.abs(latitude - centroid_lat)))
    target_x = values.shape[1] // 2
    values = np.roll(values, target_x - ix, axis=1)
    relative_longitude = (
        np.arange(values.shape[1], dtype=float) - target_x
    ) * MPAS_LPT_GRID_SPACING_DEGREES
    relative_latitude = (
        np.arange(values.shape[0], dtype=float) - iy
    ) * MPAS_LPT_GRID_SPACING_DEGREES
    return xr.DataArray(
        values,
        dims=("relative_latitude", "relative_longitude"),
        coords={
            "relative_latitude": relative_latitude,
            "relative_longitude": relative_longitude,
        },
    )


def _event_equal_centred_mean(
    fields: Sequence[xr.DataArray],
) -> xr.DataArray:
    """Average available events after aligning relative spatial grids."""

    if not fields:
        raise ValueError("An event-centred composite requires at least one field")
    return xr.concat(
        fields,
        dim="event",
        join="outer",
        coords="minimal",
        compat="override",
    ).mean("event", skipna=True)


def _plot_event_centred_evolution(
    primary: Path,
    lifecycle: pd.DataFrame,
):
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12, 10),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    precipitation_mappable = None
    precipitation_cmap, precipitation_norm = precip_rate_cmap_norm()
    for row, simulation in enumerate(SIMULATION_DIRS):
        sim = lifecycle[
            (lifecycle.simulation == simulation)
            & lifecycle.primary_tropical_sample.astype(bool)
            & lifecycle.reference_in_record_subset.astype(bool)
        ]
        stage_precipitation = {
            label: [] for label, _lags in SPATIAL_STAGE_WINDOWS
        }
        stage_regimes = {
            label: {1: [], 2: [], 3: []}
            for label, _lags in SPATIAL_STAGE_WINDOWS
        }
        for event_id in sim.event_id.unique():
            event = sim[sim.event_id == event_id].set_index("lag_hours")
            path = _event_product_path(primary, simulation, event_id)
            with xr.open_dataset(path) as product:
                if "cloud_regime_context" not in product:
                    continue
                for label, stage_lags in SPATIAL_STAGE_WINDOWS:
                    if not set(stage_lags).issubset(event.index):
                        continue
                    event_precipitation: list[xr.DataArray] = []
                    event_regimes = {1: [], 2: [], 3: []}
                    for lag in stage_lags:
                        sample = event.loc[lag]
                        if isinstance(sample, pd.DataFrame):
                            sample = sample.iloc[-1]
                        time_value = pd.Timestamp(sample.Time)
                        precip = product.raw_precipitation_mm_day.sel(
                            Time=time_value
                        )
                        regime = product.cloud_regime_context.sel(
                            Time=time_value
                        )
                        lon = float(sample.precip_weighted_centroid_lon)
                        lat = float(sample.precip_weighted_centroid_lat)
                        event_precipitation.append(
                            _centred_event_field(precip, lon, lat)
                        )
                        for code in event_regimes:
                            event_regimes[code].append(
                                _centred_event_field(
                                    (regime == code).astype(float),
                                    lon,
                                    lat,
                                )
                            )
                    stage_precipitation[label].append(
                        _event_equal_centred_mean(event_precipitation)
                    )
                    for code in event_regimes:
                        stage_regimes[label][code].append(
                            _event_equal_centred_mean(event_regimes[code])
                        )
        for column, (label, stage_lags) in enumerate(SPATIAL_STAGE_WINDOWS):
            precipitation_collect = stage_precipitation[label]
            regimes_collect = stage_regimes[label]
            axis = axes[row, column]
            if precipitation_collect:
                mean_precip = _event_equal_centred_mean(
                    precipitation_collect
                )
                x = mean_precip.relative_longitude.values
                y = mean_precip.relative_latitude.values
                precipitation_mappable = axis.pcolormesh(
                    x,
                    y,
                    mean_precip.values,
                    cmap=precipitation_cmap,
                    norm=precipitation_norm,
                    shading="auto",
                )
                for code in (1, 2, 3):
                    colour = HIGH_CLOUD_CLASS_COLOURS[code]
                    frequency = _event_equal_centred_mean(
                        regimes_collect[code]
                    )
                    if np.isfinite(frequency).any() and (
                        np.nanmax(frequency) > 0.1
                    ):
                        axis.contour(
                            x,
                            y,
                            frequency.values,
                            levels=[0.1],
                            colors=colour,
                            linewidths=1.0,
                        )
            axis.axvline(0, color="0.5", lw=0.5)
            axis.axhline(0, color="0.5", lw=0.5)
            axis.set_xlim(-30.0, 30.0)
            axis.set_ylim(-20.0, 20.0)
            if row == 0:
                axis.set_title(
                    f"{label}\n{min(stage_lags):+d} to "
                    f"{max(stage_lags):+d} h"
                )
            if column == 0:
                axis.set_ylabel(
                    f"{SIMULATION_LABELS[simulation]}\nrelative latitude (°)"
                )
    for axis in axes[-1]:
        axis.set_xlabel("Longitude relative to precipitation centroid (°)")
    if precipitation_mappable is not None:
        fig.colorbar(
            precipitation_mappable,
            ax=axes,
            pad=0.015,
            label="Precipitation (mm day$^{-1}$)",
        )
    axes[0, 0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=HIGH_CLOUD_CLASS_COLOURS[1],
                label="active deep convection",
            ),
            Line2D(
                [0],
                [0],
                color=HIGH_CLOUD_CLASS_COLOURS[2],
                label="stratiform high cloud",
            ),
            Line2D(
                [0],
                [0],
                color=HIGH_CLOUD_CLASS_COLOURS[3],
                label="non-precipitating high cloud",
            ),
        ],
        loc="upper left",
        fontsize=6.5,
        frameon=True,
        title="10% occurrence contours",
    )
    fig.suptitle(
        "LPO-centred precipitation and cloud regimes\n"
        "Event-first six-hourly window means"
    )
    return fig


def _plot_environment(lifecycle: pd.DataFrame):
    return _plot_composite_grid(
        lifecycle,
        (
            (
                "temperature_100hpa",
                "100-hPa temperature (K)",
            ),
            (
                "temperature_100hpa_event_anomaly",
                "100-hPa event anomaly (K)",
            ),
            (
                "layer_mean_n2_ttl",
                r"TTL $N^2$ (s$^{-2}$)",
            ),
            (
                "layer_mean_n2_ttl_event_anomaly",
                r"$N^2$ event anomaly (s$^{-2}$)",
            ),
            (
                "zonal_wind_100hpa",
                r"100-hPa zonal wind (m s$^{-1}$)",
            ),
            (
                "utls_shear_u100_minus_u250",
                r"UTLS shear $u_{100}-u_{250}$ (m s$^{-1}$)",
            ),
        ),
        "UTLS environment",
    )


def _plot_cloud_depth_lifecycle(lifecycle: pd.DataFrame):
    """Plot absolute and attributable cloud-depth lifecycle measures."""

    definitions = (
        (
            "median_cloud_top_pressure_hpa",
            "Median cloud-top pressure (hPa)",
        ),
        (
            "p10_cloud_top_pressure_hpa",
            f"P{DEEP_CLOUD_TOP_PERCENTILE} cloud-top pressure (hPa)",
        ),
        (
            "ttl_reaching_high_cloud_fraction_of_envelope",
            (
                "TTL-reaching fraction "
                f"(CTP ≤ {TTL_BASE_PRESSURE_HPA:g} hPa)"
            ),
        ),
    )
    composite = bootstrap_event_composite(
        lifecycle, [variable for variable, _label in definitions]
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 7.0), constrained_layout=True, sharex=True
    )
    for column, (variable, label) in enumerate(definitions):
        for simulation in SIMULATION_DIRS:
            colour = SIMULATION_COLOURS[simulation]
            mean = composite.event_equal_mean.sel(
                simulation=simulation, variable=variable
            )
            low = composite.ci_low.sel(
                simulation=simulation, variable=variable
            )
            high = composite.ci_high.sel(
                simulation=simulation, variable=variable
            )
            count = int(composite.event_count.sel(simulation=simulation))
            axes[0, column].plot(
                composite.lag_hours, mean, color=colour,
                label=f"{SIMULATION_LABELS[simulation]} (n={count})",
            )
            axes[0, column].fill_between(
                composite.lag_hours, low, high, color=colour, alpha=0.12
            )
            if simulation == "control":
                continue
            difference = composite.difference_mean.sel(
                contrast=simulation, variable=variable
            )
            difference_low = composite.difference_ci_low.sel(
                contrast=simulation, variable=variable
            )
            difference_high = composite.difference_ci_high.sel(
                contrast=simulation, variable=variable
            )
            axes[1, column].plot(
                composite.lag_hours, difference, color=colour,
                label=f"{SIMULATION_LABELS[simulation]} − control",
            )
            axes[1, column].fill_between(
                composite.lag_hours, difference_low, difference_high,
                color=colour, alpha=0.14,
            )
        axes[0, column].set_title(label)
        axes[1, column].axhline(0.0, color="0.45", lw=0.7)
        for row in range(2):
            axes[row, column].axvline(0.0, color="0.45", lw=0.7, ls="--")
            axes[row, column].grid(alpha=0.2)
        if column < 2:
            axes[0, column].invert_yaxis()
            axes[1, column].invert_yaxis()
        axes[1, column].set_xlabel(LIFECYCLE_LAG_LABEL)
    axes[0, 0].set_ylabel("Composite")
    axes[1, 0].set_ylabel("Difference")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Cloud-depth pathway")
    return fig


def _native_plot_coordinates(
    composite: xr.Dataset,
    simulation: str,
) -> tuple[np.ndarray, np.ndarray]:
    pressure = composite.native_pressure_hpa.sel(
        simulation=simulation,
        longitude_half_width_degrees=DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES,
    ).transpose("native_level", "lag_hours")
    lag = np.broadcast_to(
        np.asarray(pressure.lag_hours)[None, :], pressure.shape
    )
    return lag, _fill_native_pressure_coordinates(
        np.asarray(pressure, dtype=float),
        context=f"{simulation} native lag-pressure plot",
    )


def _shade_ttl_pressure(axis) -> None:
    """Mark the shared 150--70 hPa TTL reference layer."""

    axis.axhspan(
        TTL_TOP_PRESSURE_HPA,
        TTL_BASE_PRESSURE_HPA,
        color="0.7",
        alpha=0.12,
        zorder=2,
    )


def _fill_native_pressure_coordinates(
    pressure: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Fill unavailable lag coordinates from each native level's median."""

    pressure = np.asarray(pressure, dtype=float).copy()
    if pressure.ndim != 2:
        raise ValueError(f"{context} pressure coordinates must be 2-D")
    for level in range(pressure.shape[0]):
        valid = np.isfinite(pressure[level]) & (pressure[level] > 0.0)
        if not np.any(valid):
            raise RuntimeError(
                f"{context} has no finite pressure at native level {level}"
            )
        pressure[level, ~valid] = np.median(pressure[level, valid])
    return pressure


def _stipple_significance(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    significant: xr.DataArray | np.ndarray,
    *,
    stride: int = 2,
) -> None:
    mask = np.asarray(significant, dtype=bool)
    sampled = np.zeros_like(mask, dtype=bool)
    sampled[::stride, ::stride] = mask[::stride, ::stride]
    axis.scatter(
        x[sampled], y[sampled], s=2.0, c="black", marker=".",
        linewidths=0.0, rasterized=True,
    )


NEAR_PEAK_VERTICAL_VARIABLES = (
    ("unfiltered_cloud_fraction", "Cloud fraction", ""),
    ("unfiltered_temperature", "Temperature", "K"),
    ("unfiltered_static_stability", "Static stability", r"s$^{-2}$"),
    (
        "unfiltered_upward_pressure_velocity",
        "Ascent",
        r"Pa s$^{-1}$",
    ),
)


def _near_peak_event_profiles(sections: xr.Dataset) -> xr.Dataset:
    """Average five near-peak samples within each complete LPO event."""

    variables = (
        "native_pressure_hpa",
        *(name for name, _title, _unit in NEAR_PEAK_VERTICAL_VARIABLES),
    )
    cubes = {}
    for variable in variables:
        cubes[variable] = _default_window_event_cube(
            sections, variable, lags=NEAR_PEAK_LAGS_HOURS
        )
    pressure = cubes["native_pressure_hpa"]
    complete = pressure.notnull().any("native_level").all("lag_hours")
    if not bool(complete.any()):
        raise RuntimeError(
            "No LPO event contains every near-peak vertical sample"
        )
    profiles = {
        variable: cube.where(complete, drop=True).mean(
            "lag_hours", skipna=False
        )
        for variable, cube in cubes.items()
    }
    output = xr.Dataset(profiles)
    output.attrs.update(
        event_count=int(complete.sum()),
        event_weighting="one equal weight per complete LPO event",
        lag_hours=json.dumps(NEAR_PEAK_LAGS_HOURS),
        longitude_half_width_degrees=(
            DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
        ),
    )
    return output


def _plot_near_peak_vertical_profiles(primary: Path):
    """Plot absolute and attributable profiles near precipitation peak."""

    profiles = {}
    for simulation in SIMULATION_DIRS:
        path = (
            primary
            / simulation
            / "standard"
            / LIFECYCLE_SECTION_FILENAME
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing cached lifecycle event sections {path}"
            )
        with xr.open_dataset(path) as sections:
            profiles[simulation] = _near_peak_event_profiles(
                sections
            ).load()

    fig, axes = plt.subplots(
        2,
        len(NEAR_PEAK_VERTICAL_VARIABLES),
        figsize=(13.5, 7.2),
        constrained_layout=True,
        sharey=True,
    )
    control_pressure = profiles["control"].native_pressure_hpa.mean("event")
    for column, (variable, title, unit) in enumerate(
        NEAR_PEAK_VERTICAL_VARIABLES
    ):
        absolute_axis = axes[0, column]
        response_axis = axes[1, column]
        for simulation_index, simulation in enumerate(SIMULATION_DIRS):
            source = profiles[simulation]
            statistics = event_equal_statistics(
                source[variable],
                seed=TIMING_BOOTSTRAP_SEED + simulation_index,
            )
            pressure = source.native_pressure_hpa.mean("event")
            colour = SIMULATION_COLOURS[simulation]
            absolute_axis.plot(
                statistics["mean"],
                pressure,
                color=colour,
                label=(
                    f"{SIMULATION_LABELS[simulation]} "
                    f"(n={source.sizes['event']})"
                ),
            )
            absolute_axis.fill_betweenx(
                pressure,
                statistics.ci_low,
                statistics.ci_high,
                color=colour,
                alpha=0.14,
            )
            if simulation == "control":
                continue
            contrast = independent_event_contrast(
                source[variable],
                profiles["control"][variable],
                seed=TIMING_BOOTSTRAP_SEED + simulation_index,
            )
            response_axis.plot(
                contrast["mean"],
                control_pressure,
                color=colour,
                label=f"{SIMULATION_LABELS[simulation]} − control",
            )
            response_axis.fill_betweenx(
                control_pressure,
                contrast.ci_low,
                contrast.ci_high,
                color=colour,
                alpha=0.14,
            )
        for axis in (absolute_axis, response_axis):
            _shade_ttl_pressure(axis)
            axis.set_yscale("log")
            axis.set_ylim(400.0, 50.0)
            axis.grid(alpha=0.18)
        response_axis.axvline(0.0, color="0.45", lw=0.7)
        if variable == "unfiltered_upward_pressure_velocity":
            absolute_axis.axvline(0.0, color="0.45", lw=0.7)
        absolute_axis.set_title(
            f"{title}\n"
            + ("LPO envelope" if variable == "unfiltered_cloud_fraction"
               else "Raining LPO columns (>1 mm/day)")
        )
        response_axis.set_xlabel(
            "Difference" if not unit else f"Difference ({unit})"
        )
        absolute_axis.set_xlabel(
            "Absolute" if not unit else f"Absolute ({unit})"
        )
    axes[0, 0].set_ylabel("Absolute\nPressure (hPa)")
    axes[1, 0].set_ylabel("Response\nPressure (hPa)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Near-peak LPO vertical profiles\n"
        "Event-first mean at −12, −6, 0, +6, and +12 h; "
        "±5° longitude"
    )
    return fig


def _plot_lag_pressure_mechanism(
    composite: xr.Dataset,
    lifecycle: pd.DataFrame,
):
    """Plot native-level cloud, temperature, and stability responses."""

    columns = (
        (
            "unfiltered_cloud_fraction",
            "response_unfiltered_cloud_fraction",
            "Cloud fraction",
            None,
        ),
        (
            "unfiltered_temperature",
            "response_unfiltered_temperature",
            "Temperature",
            "K",
        ),
        (
            "unfiltered_static_stability",
            "response_unfiltered_static_stability",
            r"Static stability",
            r"s$^{-2}$",
        ),
    )
    rows = SIMULATION_ROWS
    cloud_top = bootstrap_event_composite(
        lifecycle,
        ["median_cloud_top_pressure_hpa", "p10_cloud_top_pressure_hpa"],
    )
    fig, axes = plt.subplots(
        len(rows), 3,
        figsize=(13, 3.0 * len(rows)), constrained_layout=True,
        sharex=True, sharey=True,
    )
    response_mappables = [None, None, None]
    absolute_mappables = [None, None, None]
    response_limits = {}
    for _absolute, response, _title, _unit in columns:
        response_limits[response] = _symmetric_limit(
            composite[f"{response}_mean"]
        )
    for row, simulation in enumerate(rows):
        x, pressure = _native_plot_coordinates(
            composite, "control" if row else simulation
        )
        for column, (absolute, response, title, unit) in enumerate(columns):
            axis = axes[row, column]
            if row == 0:
                field = composite[absolute].sel(
                    simulation="control",
                    longitude_half_width_degrees=(
                        DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
                    ),
                ).transpose("native_level", "lag_hours")
                if column == 0:
                    kwargs = {"cmap": "Blues", "vmin": 0.0, "vmax": 1.0}
                elif column == 1:
                    kwargs = {
                        "cmap": "coolwarm",
                        "norm": mpl_colors.Normalize(
                            vmin=190.0,
                            vmax=310.0,
                        ),
                    }
                else:
                    limit = _symmetric_limit(field)
                    kwargs = {
                        "cmap": "RdBu_r", "vmin": -limit, "vmax": limit
                    }
                mappable = axis.pcolormesh(
                    x, pressure, field, shading="auto", **kwargs
                )
                absolute_mappables[column] = mappable
            else:
                field = composite[f"{response}_mean"].sel(
                    contrast=simulation
                ).transpose("native_level", "lag_hours")
                limit = response_limits[response]
                mappable = axis.pcolormesh(
                    x, pressure, field, cmap="RdBu_r",
                    vmin=-limit, vmax=limit, shading="auto",
                )
                response_mappables[column] = mappable
                significant = composite[
                    f"{response}_pointwise_significant"
                ].sel(contrast=simulation).transpose(
                    "native_level", "lag_hours"
                )
                _stipple_significance(axis, x, pressure, significant)
            if column == 0 and row == 0:
                for variable, style, label in (
                    (
                        "median_cloud_top_pressure_hpa", "solid",
                        "Median CTP",
                    ),
                    (
                        "p10_cloud_top_pressure_hpa", "dashed",
                        "P10 CTP",
                    ),
                ):
                    line = cloud_top.event_equal_mean.sel(
                        simulation="control", variable=variable
                    )
                    axis.plot(
                        cloud_top.lag_hours, line, color="black",
                        ls=style, lw=1.0, label=label if row == 0 else None,
                    )
            _shade_ttl_pressure(axis)
            axis.axvline(0.0, color="0.35", lw=0.7)
            axis.set_yscale("log")
            axis.set_ylim(400.0, 50.0)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                row_label = (
                    SIMULATION_LABELS[simulation]
                    if row == 0
                    else f"{SIMULATION_LABELS[simulation]} − control"
                )
                axis.set_ylabel(f"{row_label}\nPressure (hPa)")
            if row == len(rows) - 1:
                axis.set_xlabel(LIFECYCLE_LAG_LABEL)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for column, (_absolute, _response, _title, unit) in enumerate(columns):
        control_label = "Control" if unit is None else f"Control ({unit})"
        difference_label = (
            "Difference" if unit is None else f"Difference ({unit})"
        )
        fig.colorbar(
            absolute_mappables[column], ax=axes[0, column],
            label=control_label, pad=0.01,
        )
        fig.colorbar(
            response_mappables[column], ax=axes[1:, column],
            label=difference_label, pad=0.01,
        )
    fig.suptitle("Cloud-depth and UTLS pathway")
    return fig


def _plot_wave_lifecycle(lifecycle: pd.DataFrame, wave_name: str):
    prefix = _wave_key(wave_name)
    variables = (
        (
            WAVE_LIFECYCLE_FIELDS[(wave_name, "Precip")],
            "Filtered precipitation (mm day$^{-1}$)",
        ),
        (
            WAVE_LIFECYCLE_FIELDS[(wave_name, "PW")],
            r"Filtered PW (kg m$^{-2}$)",
        ),
        (
            WAVE_LIFECYCLE_FIELDS[(wave_name, "OLR")],
            r"Filtered OLR (W m$^{-2}$)",
        ),
        (
            WAVE_LIFECYCLE_FIELDS[(wave_name, "T100")],
            "Filtered 100-hPa temperature (K)",
        ),
        (
            WAVE_LIFECYCLE_FIELDS[(wave_name, "u100")],
            r"Filtered 100-hPa zonal wind (m s$^{-1}$)",
        ),
    )
    label = WAVE_BOXES[wave_name].get("label", prefix.replace("_", " "))
    return _plot_composite_grid(
        lifecycle,
        variables,
        f"{label} sampled along LPO lifecycles",
    )


def _symmetric_limit(field: xr.DataArray, percentile: float = 98.0) -> float:
    values = np.abs(np.asarray(field, dtype=float))
    values = values[np.isfinite(values)]
    if not values.size:
        return 1.0
    limit = float(np.percentile(values, percentile))
    return limit if limit > 0.0 else 1.0


def _symmetric_cmap_norm(
    field: xr.DataArray,
    *,
    percentile: float = 98.0,
):
    """Return the project difference palette and readable boundaries."""

    levels = symmetric_levels(
        _symmetric_limit(field, percentile=percentile),
        ndiv=11,
    )
    cmap = difference_cmap(levels)
    norm = mpl_colors.BoundaryNorm(levels, cmap.N)
    return cmap, norm


def _cloud_stage_lags(composite: xr.Dataset) -> tuple[int, ...]:
    """Use canonical stages, or their distinct nearest test-subset lags."""

    available = np.asarray(composite.lag_hours.values, dtype=int)
    if all(value in set(available) for value in (-48, 0, 48)):
        return (-48, 0, 48)
    selected = []
    for target in (-48, 0, 48):
        nearest = int(available[np.argmin(np.abs(available - target))])
        if nearest not in selected:
            selected.append(nearest)
    return tuple(selected)


def _plot_cloud_modal_class(composite: xr.Dataset):
    """Show modal classified high cloud with agreement and envelope context."""

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb

    stages = _cloud_stage_lags(composite)
    fig, axes = plt.subplots(
        len(SIMULATION_DIRS), len(stages),
        figsize=(12, 9.2), constrained_layout=True,
        sharex=True, sharey=True,
    )
    axes = np.asarray(axes).reshape(len(SIMULATION_DIRS), len(stages))
    for row, simulation in enumerate(SIMULATION_DIRS):
        source = composite.sel(
            simulation=simulation,
            cloud_relative_longitude=slice(-30.0, 30.0),
            cloud_relative_latitude=slice(-20.0, 20.0),
        )
        for column, lag in enumerate(stages):
            axis = axes[row, column]
            mode = source.modal_high_cloud_class.sel(lag_hours=lag)
            agreement = source.modal_high_cloud_class_agreement.sel(
                lag_hours=lag
            )
            rgba = np.ones((*mode.shape, 4), dtype=float)
            rgba[..., 3] = 1.0
            strength = np.clip(np.asarray(agreement), 0.5, 1.0)
            for code in CLOUD_CLASS_CODES:
                selected = np.asarray(mode == code)
                colour = np.asarray(to_rgb(HIGH_CLOUD_CLASS_COLOURS[code]))
                for channel in range(3):
                    rgba[..., channel][selected] = (
                        1.0
                        - strength[selected]
                        * (1.0 - colour[channel])
                    )
            x = source.cloud_relative_longitude
            y = source.cloud_relative_latitude
            axis.imshow(
                rgba,
                origin="lower",
                extent=(float(x.min()), float(x.max()), float(y.min()), float(y.max())),
                interpolation="nearest",
                aspect="auto",
            )
            envelope = source.lpo_envelope_occurrence.sel(lag_hours=lag)
            envelope_values = np.asarray(envelope, dtype=float)
            if (
                np.any(np.isfinite(envelope_values))
                and np.nanmax(envelope_values) >= 0.25
            ):
                axis.contour(
                    x, y, envelope,
                    levels=[0.25, 0.5],
                    colors=["0.35", "black"],
                    linewidths=[0.7, 1.0],
                )
            axis.axhline(0.0, color="0.5", lw=0.5)
            axis.axvline(0.0, color="0.5", lw=0.5)
            if row == 0:
                axis.set_title(f"{lag:+d} h")
            if column == 0:
                axis.set_ylabel(
                    f"{SIMULATION_LABELS[simulation]}\n"
                    "Latitude from centroid (°)"
                )
    for axis in axes[-1]:
        axis.set_xlabel("Longitude from centroid (°)")
    strengths = (0.58, 0.78, 1.0)
    colours = []
    tick_labels = []
    short_names = ("Active", "Stratiform", "Nonprecipitating")
    terciles = ("0.50–0.67", "0.67–0.83", "0.83–1.00")
    for code, short_name in zip(CLOUD_CLASS_CODES, short_names, strict=True):
        base = np.asarray(to_rgb(HIGH_CLOUD_CLASS_COLOURS[code]))
        for strength, tercile in zip(strengths, terciles, strict=True):
            colours.append(1.0 - strength * (1.0 - base))
            tick_labels.append(f"{short_name}\n{tercile}")
    modal_cmap = ListedColormap(colours)
    modal_norm = BoundaryNorm(np.arange(0.5, 10.0, 1.0), modal_cmap.N)
    colorbar = fig.colorbar(
        ScalarMappable(norm=modal_norm, cmap=modal_cmap),
        ax=axes, orientation="horizontal", pad=0.055, fraction=0.055,
        ticks=np.arange(1, 10),
        label="Modal class and agreement",
    )
    colorbar.ax.set_xticklabels(tick_labels, fontsize=6)
    fig.legend(
        handles=[
            Line2D(
                [0], [0], color="0.35", lw=0.7,
                label="25% envelope occurrence",
            ),
            Line2D(
                [0], [0], color="black", lw=1.0,
                label="50% envelope occurrence",
            ),
        ],
        loc="outside lower center", ncol=2, frameon=False,
    )
    fig.suptitle("Modal high-cloud regime")
    return fig


def _plot_cloud_occurrences(composite: xr.Dataset, simulation: str):
    stages = _cloud_stage_lags(composite)
    fields = (
        ("total_high_cloud_occurrence", None, "Total high cloud"),
        ("cloud_class_occurrence", CLOUD_CLASS_NAMES[0], "Active deep convection"),
        ("cloud_class_occurrence", CLOUD_CLASS_NAMES[1], "Stratiform high cloud"),
        ("cloud_class_occurrence", CLOUD_CLASS_NAMES[2], "Non-precipitating high cloud"),
    )
    source = composite.sel(
        simulation=simulation,
        cloud_relative_longitude=slice(-30.0, 30.0),
        cloud_relative_latitude=slice(-20.0, 20.0),
    )
    fig, axes = plt.subplots(
        len(stages), len(fields), figsize=(12, 8.4),
        constrained_layout=True, sharex=True, sharey=True,
    )
    axes = np.asarray(axes).reshape(len(stages), len(fields))
    total_mappable = None
    class_mappable = None
    for row, lag in enumerate(stages):
        for column, (variable, class_name, title) in enumerate(fields):
            axis = axes[row, column]
            field = source[variable].sel(lag_hours=lag)
            if class_name is not None:
                field = field.sel(cloud_class=class_name)
            is_total = class_name is None
            mappable = axis.pcolormesh(
                source.cloud_relative_longitude,
                source.cloud_relative_latitude,
                field,
                cmap="YlGnBu", vmin=0.0,
                vmax=1.0 if is_total else 0.6, shading="auto",
            )
            if is_total:
                total_mappable = mappable
            else:
                class_mappable = mappable
            envelope = source.lpo_envelope_occurrence.sel(lag_hours=lag)
            envelope_values = np.asarray(envelope, dtype=float)
            if (
                np.any(np.isfinite(envelope_values))
                and np.nanmax(envelope_values) >= 0.25
            ):
                axis.contour(
                    source.cloud_relative_longitude,
                    source.cloud_relative_latitude,
                    envelope,
                    levels=[0.25], colors="black", linewidths=0.8,
                )
            axis.axhline(0.0, color="0.55", lw=0.45)
            axis.axvline(0.0, color="0.55", lw=0.45)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"{lag:+d} h\nLatitude from centroid (°)")
            if row == len(stages) - 1:
                axis.set_xlabel("Longitude from centroid (°)")
    if total_mappable is not None:
        fig.colorbar(
            total_mappable, ax=axes[:, 0], label="Total occurrence",
            pad=0.015,
        )
    if class_mappable is not None:
        fig.colorbar(
            class_mappable, ax=axes[:, 1:], label="Class occurrence",
            pad=0.015,
        )
    fig.suptitle(
        f"{SIMULATION_LABELS[simulation]} broad cloud occurrence; "
        "black contour is 25% envelope occurrence"
    )
    return fig


def _plot_cloud_occurrence_response(
    composite: xr.Dataset,
    *,
    significance_variable: str = "pointwise_significant",
):
    """Plot background-relative cloud-occurrence experiment responses."""

    classes = (
        ("total_high_cloud", "Total high cloud"),
        (CLOUD_CLASS_NAMES[0], "Active deep convection"),
        (CLOUD_CLASS_NAMES[1], "Stratiform high cloud"),
        (CLOUD_CLASS_NAMES[2], "Nonprecipitating high cloud"),
    )
    source = composite.sel(
        cloud_relative_longitude=slice(-30.0, 30.0),
        cloud_relative_latitude=slice(-20.0, 20.0),
    )
    stages = _cloud_stage_lags(source)
    contrasts = RESPONSE_ROWS
    fig, axes = plt.subplots(
        len(stages) * len(contrasts), len(classes),
        figsize=(13, 14), constrained_layout=True,
        sharex=True, sharey=True,
    )
    longitude = np.asarray(source.cloud_relative_longitude, dtype=float)
    latitude = np.asarray(source.cloud_relative_latitude, dtype=float)
    x, y = np.meshgrid(longitude, latitude)
    mappables = [None] * len(classes)
    for column, (class_name, _title) in enumerate(classes):
        selected = source.response_cloud_excess_occurrence_mean.sel(
            response_cloud_class=class_name
        )
        limit = _symmetric_limit(selected)
        for stage_index, lag in enumerate(stages):
            for contrast_index, contrast in enumerate(contrasts):
                row = stage_index * len(contrasts) + contrast_index
                axis = axes[row, column]
                field = selected.sel(contrast=contrast, lag_hours=lag)
                mappables[column] = axis.pcolormesh(
                    longitude, latitude, field, cmap="RdBu_r",
                    vmin=-limit, vmax=limit, shading="auto",
                )
                significant = source[
                    f"response_cloud_excess_occurrence_{significance_variable}"
                ].sel(
                    contrast=contrast,
                    lag_hours=lag,
                    response_cloud_class=class_name,
                )
                _stipple_significance(axis, x, y, significant, stride=3)
                axis.axhline(0.0, color="0.5", lw=0.45)
                axis.axvline(0.0, color="0.5", lw=0.45)
                if column == 0:
                    axis.set_ylabel(
                        f"{SIMULATION_LABELS[contrast]} − control\n"
                        f"{lag:+d} h; latitude (°)"
                    )
                if row == 0:
                    axis.set_title(_title)
                if row == len(stages) * len(contrasts) - 1:
                    axis.set_xlabel("Longitude from centroid (°)")
    for column, (_class_name, title) in enumerate(classes):
        fig.colorbar(
            mappables[column], ax=axes[:, column],
            label=f"{title} excess-occurrence difference", pad=0.005,
        )
    mask_label = (
        "pointwise"
        if significance_variable == "pointwise_significant"
        else "5% FDR"
    )
    fig.suptitle(f"Far-field-relative cloud occurrence ({mask_label})")
    return fig


def _plot_surface_sections(
    composite: xr.Dataset,
    *,
    response: bool = False,
    wave_name: str | None = None,
):
    if response and wave_name is not None:
        raise ValueError("Filtered response sections require a separate cache")
    definitions = (
        ("precipitation", "Precipitation", r"mm day$^{-1}$"),
        ("precipitable_water", "Precipitable water", r"kg m$^{-2}$"),
        ("olr", "Outgoing longwave radiation", r"W m$^{-2}$"),
    )
    prefix = "filtered" if wave_name else "unfiltered"
    variables = [f"{prefix}_{name}" for name, _title, _unit in definitions]
    rows = RESPONSE_ROWS if response else SIMULATION_ROWS
    fig, axes = plt.subplots(
        len(rows), len(variables),
        figsize=(12, 6.2 if response else 8.2), constrained_layout=True,
        sharex=True, sharey=True,
    )
    for column, (variable, (_name, title, unit)) in enumerate(
        zip(variables, definitions)
    ):
        if response:
            selected = composite[f"response_{variable}_mean"]
        else:
            selected = composite[variable].sel(
                longitude_half_width_degrees=(
                    DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
                )
            )
            if wave_name:
                selected = selected.sel(wave_band=wave_name)
        symmetric = response or wave_name is not None
        if symmetric:
            cmap, norm = _symmetric_cmap_norm(selected)
            extend = "both"
        elif variable.endswith("precipitation"):
            cmap, norm = precip_rate_cmap_norm()
            extend = "max"
        elif variable.endswith("precipitable_water"):
            cmap = "YlGnBu"
            norm = mpl_colors.Normalize(vmin=0.0, vmax=80.0)
            extend = "max"
        else:
            cmap = "Blues"
            norm = mpl_colors.PowerNorm(
                gamma=0.5,
                vmin=200.0,
                vmax=300.0,
            )
            extend = "both"
        mappable = None
        for row, simulation in enumerate(rows):
            axis = axes[row, column]
            dimension = "contrast" if response else "simulation"
            field = selected.sel({dimension: simulation}).transpose(
                "relative_latitude", "lag_hours"
            )
            mappable = axis.pcolormesh(
                field.lag_hours,
                field.relative_latitude,
                field,
                cmap=cmap,
                norm=norm,
                shading="auto",
            )
            axis.axvline(0.0, color="0.35", lw=0.7)
            axis.axhline(0.0, color="0.55", lw=0.5)
            if response:
                significant = composite[
                    f"response_{variable}_pointwise_significant"
                ].sel(contrast=simulation).transpose(
                    "relative_latitude", "lag_hours"
                )
                lag_grid, latitude_grid = np.meshgrid(
                    significant.lag_hours,
                    significant.relative_latitude,
                )
                _stipple_significance(
                    axis, lag_grid, latitude_grid, significant
                )
            if row == 0:
                axis.set_title(title)
            if column == 0:
                row_label = (
                    f"{SIMULATION_LABELS[simulation]} − control"
                    if response
                    else SIMULATION_LABELS[simulation]
                )
                axis.set_ylabel(
                    f"{row_label}\n"
                    "Latitude from centroid (°)"
                )
        fig.colorbar(
            mappable,
            ax=axes[:, column],
            label=unit,
            pad=0.01,
            extend=extend,
        )
    context = (
        WAVE_BOXES[wave_name].get("label", wave_name)
        if wave_name else "Unfiltered; raining LPO columns (>1 mm/day)"
    )
    fig.suptitle(
        f"{context}; ±{DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES:g}° "
        "longitude mean"
    )
    fig.supxlabel(LIFECYCLE_LAG_LABEL)
    return fig


def _plot_vertical_sections(
    composite: xr.Dataset,
    *,
    response: bool = False,
    wave_name: str | None = None,
):
    if response and wave_name is not None:
        raise ValueError("Filtered response sections require a separate cache")
    prefix = "filtered" if wave_name else "unfiltered"
    names = {
        key: f"{prefix}_{key}"
        for key in (
            "temperature", "upward_pressure_velocity",
            "specific_humidity", "relative_humidity",
            "static_stability",
        )
    }
    selections = {}
    for key, variable in names.items():
        if response:
            field = composite[f"response_{variable}_mean"]
        else:
            field = composite[variable].sel(
                longitude_half_width_degrees=(
                    DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
                )
            )
            if wave_name:
                field = field.sel(wave_band=wave_name)
        selections[key] = field
    symmetric = response or wave_name is not None
    shade_styles = {}
    for key in ("temperature", "specific_humidity", "static_stability"):
        if symmetric:
            cmap, norm = _symmetric_cmap_norm(selections[key])
            shade_styles[key] = (
                SPECIFIC_HUMIDITY_CMAP
                if key == "specific_humidity"
                else cmap,
                norm,
            )
        elif key == "temperature":
            shade_styles[key] = (
                "coolwarm",
                mpl_colors.Normalize(vmin=190.0, vmax=310.0),
            )
        elif key == "specific_humidity":
            shade_styles[key] = (
                SPECIFIC_HUMIDITY_CMAP,
                mpl_colors.LogNorm(vmin=0.001, vmax=20.0),
            )
        else:
            finite = np.asarray(selections[key], dtype=float)
            finite = finite[np.isfinite(finite) & (finite > 0.0)]
            lower, upper = (
                np.percentile(finite, [2.0, 98.0])
                if finite.size
                else (1.0e-6, 1.0e-4)
            )
            shade_styles[key] = (
                "viridis",
                mpl_colors.LogNorm(
                    vmin=max(float(lower), 1.0e-8),
                    vmax=max(float(upper), float(lower) * 1.01),
                ),
            )
    contour_levels = {}
    for key in ("upward_pressure_velocity", "relative_humidity"):
        values = np.asarray(selections[key], dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            contour_levels[key] = np.asarray([])
        elif np.nanmin(finite) < 0.0 < np.nanmax(finite):
            limit = _symmetric_limit(selections[key], percentile=90.0)
            contour_levels[key] = np.asarray(
                [-limit, -0.5 * limit, 0.0, 0.5 * limit, limit]
            )
        else:
            contour_levels[key] = np.unique(
                np.percentile(finite, [25.0, 50.0, 75.0])
            )
    rows = RESPONSE_ROWS if response else SIMULATION_ROWS
    definitions = (
        (
            "temperature", "upward_pressure_velocity",
            "Temperature with ascent", "K",
        ),
        (
            "specific_humidity", "relative_humidity",
            "Specific humidity with relative humidity", r"g kg$^{-1}$",
        ),
        (
            "static_stability", None,
            "Static stability", r"$N^2$ (s$^{-2}$)",
        ),
    )
    has_envelope_precipitation = "lpo_envelope_precipitation" in composite
    fig = plt.figure(
        figsize=(15, 8.0 if response else 10.0),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        len(rows) + 1,
        len(definitions),
        height_ratios=[1.0] * len(rows) + [0.42],
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[row, column]) for column in range(3)]
            for row in range(len(rows))
        ]
    )
    precipitation_axes = []
    for column in range(len(definitions)):
        precipitation_axes.append(
            fig.add_subplot(
                grid[-1, column],
                sharex=axes[-1, column],
                sharey=(None if column == 0 else precipitation_axes[0]),
            )
        )
    precipitation_axes = np.asarray(precipitation_axes)
    column_mappables = [None] * len(definitions)
    for row, simulation in enumerate(rows):
        for column, (shade_key, contour_key, title, unit) in enumerate(
            definitions
        ):
            axis = axes[row, column]
            dimension = "contrast" if response else "simulation"
            shade_source = selections[shade_key].sel({dimension: simulation})
            contour_source = (
                selections[contour_key].sel({dimension: simulation})
                if contour_key is not None
                else None
            )
            if wave_name:
                vertical_dimension = "native_level"
                pressure = np.asarray(
                    composite.filtered_native_pressure_hpa.sel(
                        simulation=simulation,
                        longitude_half_width_degrees=(
                            DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
                        ),
                    ).transpose("native_level", "lag_hours"),
                    dtype=float,
                )
            elif response:
                vertical_dimension = "native_level"
                pressure = np.asarray(
                    composite.native_pressure_hpa.sel(
                        simulation="control",
                        longitude_half_width_degrees=(
                            DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
                        ),
                    ).transpose("native_level", "lag_hours"),
                    dtype=float,
                )
            else:
                vertical_dimension = "native_level"
                pressure = np.asarray(
                    composite.native_pressure_hpa.sel(
                        simulation=simulation,
                        longitude_half_width_degrees=(
                            DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
                        ),
                    ).transpose("native_level", "lag_hours"),
                    dtype=float,
                )
            shade = shade_source.transpose(vertical_dimension, "lag_hours")
            contour = (
                contour_source.transpose(vertical_dimension, "lag_hours")
                if contour_source is not None
                else None
            )
            available_levels = np.any(
                np.isfinite(pressure) & (pressure > 0.0), axis=1
            )
            if np.count_nonzero(available_levels) < 2:
                raise RuntimeError(
                    f"{simulation} {wave_name or 'unfiltered'} "
                    "lag-pressure plot has fewer than two pressure levels"
                )
            if not np.all(available_levels):
                level_indices = np.flatnonzero(available_levels)
                pressure = pressure[available_levels]
                shade = shade.isel({vertical_dimension: level_indices})
                if contour is not None:
                    contour = contour.isel(
                        {vertical_dimension: level_indices}
                    )
            pressure = _fill_native_pressure_coordinates(
                pressure,
                context=(
                    f"{simulation} {wave_name or 'unfiltered'} "
                    "lag-pressure plot"
                ),
            )
            lag_grid = np.broadcast_to(
                np.asarray(shade.lag_hours)[None, :], shade.shape
            )
            cmap, norm = shade_styles[shade_key]
            mappable = axis.pcolormesh(
                lag_grid, pressure, shade,
                cmap=cmap,
                norm=norm,
                shading="auto",
            )
            column_mappables[column] = mappable
            levels = (
                contour_levels[contour_key]
                if contour_key is not None
                else np.asarray([])
            )
            if contour is not None and levels.size:
                negative = levels[levels < 0.0]
                positive = levels[levels > 0.0]
                if negative.size:
                    axis.contour(
                        lag_grid, pressure, contour, levels=negative,
                        colors="0.25", linestyles="dashed", linewidths=0.75,
                    )
                if positive.size:
                    axis.contour(
                        lag_grid, pressure, contour, levels=positive,
                        colors="black", linestyles="solid", linewidths=0.75,
                    )
                if np.any(np.isclose(levels, 0.0)):
                    axis.contour(
                        lag_grid, pressure, contour, levels=[0.0],
                        colors="black", linestyles="solid", linewidths=1.35,
                    )
            if response:
                significant = composite[
                    f"response_{names[shade_key]}_pointwise_significant"
                ].sel(contrast=simulation).transpose(
                    vertical_dimension, "lag_hours"
                )
                _stipple_significance(
                    axis, lag_grid, pressure, significant
                )
            _shade_ttl_pressure(axis)
            axis.set_yscale("log")
            axis.set_ylim(1000, 50)
            axis.axvline(0.0, color="0.35", lw=0.7)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                row_label = (
                    f"{SIMULATION_LABELS[simulation]} − control"
                    if response
                    else SIMULATION_LABELS[simulation]
                )
                axis.set_ylabel(
                    f"{row_label}\nPressure (hPa)"
                )
            if row == len(rows) - 1 and not has_envelope_precipitation:
                axis.set_xlabel("Lag (h)")
    for column, (_shade_key, _contour_key, _title, unit) in enumerate(
        definitions
    ):
        fig.colorbar(
            column_mappables[column],
            ax=[*axes[:, column], precipitation_axes[column]],
            label=unit,
            pad=0.01,
            extend="both",
        )
    context = (
        WAVE_BOXES[wave_name].get("label", wave_name)
        if wave_name else "Unfiltered; raining LPO columns (>1 mm/day)"
    )
    if has_envelope_precipitation:
        precipitation = composite.lpo_envelope_precipitation
        control = precipitation.sel(simulation="control")
        for column, precipitation_axis in enumerate(precipitation_axes):
            if response:
                for simulation in rows:
                    curve = precipitation.sel(simulation=simulation) - control
                    precipitation_axis.plot(
                        curve.lag_hours,
                        RAIN_VOLUME_DISPLAY_SCALE * curve,
                        color=SIMULATION_COLOURS[simulation],
                        label=f"{SIMULATION_LABELS[simulation]} − control",
                    )
                precipitation_axis.axhline(0.0, color="0.4", lw=0.7)
            else:
                for simulation in SIMULATION_DIRS:
                    curve = precipitation.sel(simulation=simulation)
                    precipitation_axis.plot(
                        curve.lag_hours,
                        RAIN_VOLUME_DISPLAY_SCALE * curve,
                        color=SIMULATION_COLOURS[simulation],
                        label=SIMULATION_LABELS[simulation],
                    )
                    if "lpo_envelope_precipitation_ci_low" in composite:
                        low = composite.lpo_envelope_precipitation_ci_low.sel(
                            simulation=simulation
                        )
                        high = composite.lpo_envelope_precipitation_ci_high.sel(
                            simulation=simulation
                        )
                        precipitation_axis.fill_between(
                            curve.lag_hours,
                            RAIN_VOLUME_DISPLAY_SCALE * low,
                            RAIN_VOLUME_DISPLAY_SCALE * high,
                            color=SIMULATION_COLOURS[simulation],
                            alpha=0.15,
                        )
            precipitation_axis.axvline(0.0, color="0.35", lw=0.7)
            precipitation_axis.set_xlabel(LIFECYCLE_LAG_LABEL)
            if column == 0:
                precipitation_axis.set_ylabel(RAIN_VOLUME_LABEL)
                precipitation_axis.legend(
                    frameon=False, ncol=1, fontsize=7
                )
            else:
                precipitation_axis.tick_params(labelleft=False)
    else:
        for precipitation_axis in precipitation_axes:
            precipitation_axis.set_axis_off()
    ascent_text = ", ".join(
        f"{value:.2g}"
        for value in contour_levels["upward_pressure_velocity"]
    )
    humidity_text = ", ".join(
        f"{value:.2g}" for value in contour_levels["relative_humidity"]
    )
    suffix = (
        "; LPO-envelope precipitation shown below"
        if has_envelope_precipitation else ""
    )
    fig.suptitle(f"{context}{suffix}")
    fig.legend(
        handles=[
            Line2D([0], [0], color="black", lw=0.75, ls="solid"),
            Line2D([0], [0], color="0.25", lw=0.75, ls="dashed"),
            Line2D([0], [0], color="black", lw=1.35, ls="solid"),
        ],
        labels=("positive", "negative", "zero"),
        title=(
            f"Contours: ascent [{ascent_text}] Pa s⁻¹; "
            f"RH [{humidity_text}] %"
        ),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=6.5,
        title_fontsize=6.5,
    )
    return fig


def _plot_unfiltered_vertical_response(
    composite: xr.Dataset,
    *,
    significance_variable: str = "pointwise_significant",
):
    """Plot unfiltered native-level experiment-minus-control sections."""

    definitions = (
        (
            "unfiltered_temperature",
            "unfiltered_upward_pressure_velocity",
            "Temperature with ascent response",
            "K",
        ),
        (
            "unfiltered_specific_humidity",
            "unfiltered_relative_humidity",
            "Specific-humidity with RH response",
            r"g kg$^{-1}$",
        ),
    )
    contrasts = RESPONSE_ROWS
    x, pressure = _native_plot_coordinates(composite, "control")
    fig, axes = plt.subplots(
        len(contrasts), 2,
        figsize=(12, 3.6 * len(contrasts)), constrained_layout=True,
        sharex=True, sharey=True,
    )
    mappables = [None, None]
    for column, (shade_name, contour_name, title, unit) in enumerate(definitions):
        shade_stem = f"response_{shade_name}"
        contour_stem = f"response_{contour_name}"
        shade_cmap, shade_norm = _symmetric_cmap_norm(
            composite[f"{shade_stem}_mean"]
        )
        if shade_name == "unfiltered_specific_humidity":
            shade_cmap = SPECIFIC_HUMIDITY_CMAP
        contour_limit = _symmetric_limit(
            composite[f"{contour_stem}_mean"], percentile=90.0
        )
        levels = np.asarray(
            [-contour_limit, -0.5 * contour_limit, 0.0,
             0.5 * contour_limit, contour_limit]
        )
        for row, contrast in enumerate(contrasts):
            axis = axes[row, column]
            shade = composite[f"{shade_stem}_mean"].sel(
                contrast=contrast
            ).transpose("native_level", "lag_hours")
            contour = composite[f"{contour_stem}_mean"].sel(
                contrast=contrast
            ).transpose("native_level", "lag_hours")
            mappables[column] = axis.pcolormesh(
                x,
                pressure,
                shade,
                cmap=shade_cmap,
                norm=shade_norm,
                shading="auto",
            )
            axis.contour(
                x, pressure, contour, levels=levels[levels < 0.0],
                colors="0.25", linestyles="dashed", linewidths=0.75,
            )
            axis.contour(
                x, pressure, contour, levels=levels[levels > 0.0],
                colors="black", linestyles="solid", linewidths=0.75,
            )
            axis.contour(
                x, pressure, contour, levels=[0.0], colors="black",
                linestyles="solid", linewidths=1.35,
            )
            significant = composite[
                f"{shade_stem}_{significance_variable}"
            ].sel(contrast=contrast).transpose("native_level", "lag_hours")
            _stipple_significance(axis, x, pressure, significant)
            _shade_ttl_pressure(axis)
            axis.set_yscale("log")
            axis.set_ylim(1000.0, 50.0)
            axis.axvline(0.0, color="0.4", lw=0.7)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(
                    f"{SIMULATION_LABELS[contrast]} − control\nPressure (hPa)"
                )
            if row == len(contrasts) - 1:
                axis.set_xlabel(LIFECYCLE_LAG_LABEL)
        fig.colorbar(
            mappables[column],
            ax=axes[:, column],
            label=unit,
            pad=0.01,
            extend="both",
        )
    mask_label = (
        "pointwise"
        if significance_variable == "pointwise_significant"
        else "5% FDR"
    )
    fig.suptitle(
        f"Vertical response ({mask_label}); raining LPO columns (>1 mm/day)"
    )
    return fig


def _plot_kelvin_reference_absolute(composite: xr.Dataset):
    variables = (
        ("kelvin_reference_cgx", r"$C_{gx}$ (m s$^{-1}$)"),
        ("kelvin_reference_cgz", r"$C_{gz}$ (m s$^{-1}$)"),
        ("kelvin_reference_cgz_over_cgx", r"$C_{gz}/C_{gx}$"),
    )
    fig, axes = plt.subplots(
        len(SIMULATION_DIRS), len(variables), figsize=(12, 8.2),
        constrained_layout=True, sharex=True, sharey=True,
    )
    for column, (variable, label) in enumerate(variables):
        selected = composite[variable].sel(
            zonal_wavenumber=1, vertical_wavelength_km=10.0
        )
        limit = _symmetric_limit(selected)
        mappable = None
        for row, simulation in enumerate(SIMULATION_DIRS):
            axis = axes[row, column]
            field = selected.sel(simulation=simulation).transpose(
                "height_m", "lag_hours"
            )
            mappable = axis.pcolormesh(
                field.lag_hours, field.height_m / 1000.0, field,
                cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto",
            )
            axis.axvline(0.0, color="0.35", lw=0.7)
            axis.axhspan(14.0, 20.0, color="0.85", alpha=0.18)
            if row == 0:
                axis.set_title(label)
            if column == 0:
                axis.set_ylabel(
                    f"{SIMULATION_LABELS[simulation]}\nHeight (km)"
                )
            if row == len(SIMULATION_DIRS) - 1:
                axis.set_xlabel(LIFECYCLE_LAG_LABEL)
        fig.colorbar(mappable, ax=axes[:, column], label=label, pad=0.01)
    fig.suptitle("Kelvin reference (k=1; 10-km vertical wavelength)")
    return fig


def _plot_kelvin_reference_lifecycle(composite: xr.Dataset):
    """Plot the UTLS Kelvin reference and attributable responses."""

    if "response_kelvin_reference_cgx_mean" not in composite:
        return _plot_kelvin_reference_absolute(composite)
    variables = (
        (
            "kelvin_reference_cgx", "response_kelvin_reference_cgx",
            r"$C_{gx}$ (m s$^{-1}$)",
        ),
        (
            "kelvin_reference_cgz", "response_kelvin_reference_cgz",
            r"$C_{gz}$ (m s$^{-1}$)",
        ),
        (
            "kelvin_reference_cgz_over_cgx",
            "response_kelvin_reference_cgz_over_cgx",
            r"$C_{gz}/C_{gx}$",
        ),
    )
    rows = SIMULATION_ROWS
    fig, axes = plt.subplots(
        len(rows), 3,
        figsize=(12.5, 2.85 * len(rows)), constrained_layout=True,
        sharex=True, sharey=True,
    )
    for column, (absolute, response, label) in enumerate(variables):
        selected = composite[absolute].sel(
            simulation="control", zonal_wavenumber=1,
            vertical_wavelength_km=10.0,
            height_m=slice(10_000.0, 25_000.0),
        ).transpose("height_m", "lag_hours")
        finite = np.asarray(selected, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size and np.all(finite >= 0.0):
            absolute_kwargs = {
                "cmap": "viridis", "vmin": 0.0,
                "vmax": float(np.percentile(finite, 98.0)),
            }
        elif finite.size and np.all(finite <= 0.0):
            absolute_kwargs = {
                "cmap": "viridis_r",
                "vmin": float(np.percentile(finite, 2.0)), "vmax": 0.0,
            }
        else:
            limit = _symmetric_limit(selected)
            absolute_kwargs = {
                "cmap": "RdBu_r", "vmin": -limit, "vmax": limit,
            }
        response_field = composite[f"{response}_mean"].sel(
            height_m=slice(10_000.0, 25_000.0)
        )
        response_limit = _symmetric_limit(response_field)
        absolute_mappable = None
        response_mappable = None
        for row, simulation in enumerate(rows):
            axis = axes[row, column]
            if row == 0:
                field = selected
                absolute_mappable = axis.pcolormesh(
                    field.lag_hours, field.height_m / 1000.0, field,
                    shading="auto", **absolute_kwargs,
                )
            else:
                field = response_field.sel(contrast=simulation).transpose(
                    "height_m", "lag_hours"
                )
                response_mappable = axis.pcolormesh(
                    field.lag_hours, field.height_m / 1000.0, field,
                    cmap="RdBu_r", vmin=-response_limit,
                    vmax=response_limit, shading="auto",
                )
                significant = composite[
                    f"{response}_pointwise_significant"
                ].sel(
                    contrast=simulation,
                    height_m=slice(10_000.0, 25_000.0),
                ).transpose("height_m", "lag_hours")
                lag_grid, height_grid = np.meshgrid(
                    field.lag_hours, field.height_m / 1000.0
                )
                _stipple_significance(
                    axis, lag_grid, height_grid, significant
                )
            axis.axvline(0.0, color="0.35", lw=0.7)
            axis.axhspan(14.0, 20.0, color="0.85", alpha=0.16)
            if row == 0:
                axis.set_title(label)
            if column == 0:
                row_label = (
                    "Control"
                    if row == 0
                    else f"{SIMULATION_LABELS[simulation]} − control"
                )
                axis.set_ylabel(f"{row_label}\nHeight (km)")
            if row == len(rows) - 1:
                axis.set_xlabel(LIFECYCLE_LAG_LABEL)
        fig.colorbar(
            absolute_mappable,
            ax=axes[0, column],
            label=f"Control {label}",
            pad=0.01,
        )
        fig.colorbar(
            response_mappable,
            ax=axes[1:, column],
            label=f"Difference {label}",
            pad=0.01,
        )
    fig.suptitle(
        "Dry Kelvin reference, zonal wavenumber 1, and 10-km vertical "
        r"wavelength; $|C_{gx}|\leq10^{-3}$ m s$^{-1}$ is masked"
    )
    return fig


def _plot_group_velocity_sensitivity(composite: xr.Dataset):
    """Summarize wavenumber and vertical-wavelength sensitivity."""

    variables = (
        (
            "kelvin_reference_cgx",
            "response_kelvin_sensitivity_cgx",
            r"$C_{gx}$",
        ),
        (
            "kelvin_reference_cgz",
            "response_kelvin_sensitivity_cgz",
            r"$C_{gz}$",
        ),
        (
            "kelvin_reference_cgz_over_cgx",
            "response_kelvin_sensitivity_cgz_over_cgx",
            r"$C_{gz}/C_{gx}$",
        ),
    )
    rows = SIMULATION_ROWS
    fig, axes = plt.subplots(
        len(rows), 3,
        figsize=(11, 3.15 * len(rows)), constrained_layout=True,
    )
    for column, (variable, response, label) in enumerate(variables):
        control = composite[variable].sel(
            simulation="control",
            height_m=slice(14_000.0, 20_000.0),
            lag_hours=slice(-24, 24),
        ).mean(("height_m", "lag_hours"), skipna=True)
        response_fields = composite[f"{response}_mean"].sel(
            height_m=slice(14_000.0, 20_000.0),
            lag_hours=slice(-24, 24),
        ).mean(("height_m", "lag_hours"), skipna=True)
        control_limit = _symmetric_limit(control, percentile=100.0)
        response_limit = _symmetric_limit(
            response_fields, percentile=100.0
        )
        for row, simulation in enumerate(rows):
            field = (
                control
                if row == 0
                else response_fields.sel(contrast=simulation)
            )
            limit = control_limit if row == 0 else response_limit
            image = axes[row, column].imshow(
                np.asarray(field).T,
                origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit,
                aspect="auto",
            )
            axes[row, column].set_xticks(range(3), field.zonal_wavenumber.values)
            axes[row, column].set_yticks(
                range(3), field.vertical_wavelength_km.values
            )
            for k_index in range(3):
                for wavelength_index in range(3):
                    value = float(field[k_index, wavelength_index])
                    axes[row, column].text(
                        k_index, wavelength_index, f"{value:.2g}",
                        ha="center", va="center", fontsize=7,
                    )
            if row == 0:
                axes[row, column].set_title(label)
            if column == 0:
                row_label = (
                    "Control"
                    if row == 0
                    else f"{SIMULATION_LABELS[simulation]} − control"
                )
                axes[row, column].set_ylabel(
                    f"{row_label}\n"
                    "Vertical wavelength (km)"
                )
            if row == len(rows) - 1:
                axes[row, column].set_xlabel("Zonal wavenumber")
            fig.colorbar(image, ax=axes[row, column], pad=0.01)
    fig.suptitle("Kelvin-reference sensitivity (14–20 km; ±24 h mean)")
    return fig


def _plot_averaging_sensitivity(composite: xr.Dataset):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    definitions = (
        (
            "unfiltered_precipitation",
            {"relative_latitude": slice(-2.5, 2.5)},
            "Near-centroid precipitation",
            r"mm day$^{-1}$",
        ),
        (
            "unfiltered_upward_pressure_velocity",
            {"target_pressure_hpa": 500.0},
            "500-hPa ascent",
            r"Pa s$^{-1}$",
        ),
        (
            "unfiltered_temperature",
            {"target_pressure_hpa": 200.0},
            "200-hPa temperature",
            "K",
        ),
    )
    control = composite.sel(simulation="control")
    for axis, (variable, selection, title, unit) in zip(axes, definitions):
        if "relative_latitude" in selection:
            field = control[
                "unfiltered_precipitation_near_centroid_latitude_mean"
            ]
        else:
            field = control[variable]
            target = selection["target_pressure_hpa"]
            pressure = control.native_pressure_hpa.mean(
                ("longitude_half_width_degrees", "lag_hours"),
                skipna=True,
            )
            native_level = int(
                pressure.native_level[
                    np.nanargmin(np.abs(np.asarray(pressure) - target))
                ]
            )
            field = field.sel(native_level=native_level)
        for width in LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES:
            axis.plot(
                field.lag_hours,
                field.sel(longitude_half_width_degrees=width),
                label=("centroid point" if width == 0 else f"±{width:g}°"),
            )
        axis.axhline(0.0, color="0.6", lw=0.6)
        axis.axvline(0.0, color="0.6", lw=0.6)
        axis.grid(alpha=0.2)
        axis.set(title=title, xlabel="Lag (h)", ylabel=unit)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Longitude-averaging sensitivity; raining LPO columns (>1 mm/day)"
    )
    return fig


def _draw_standard_signed_contours(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    field: xr.DataArray,
    limit: float,
) -> None:
    """Draw common signed contour levels with the required line styles."""

    values = np.asarray(field, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return
    levels = np.asarray(
        [-limit, -0.5 * limit, 0.0, 0.5 * limit, limit]
    )
    negative = levels[(levels < 0.0) & (levels >= finite.min())]
    positive = levels[(levels > 0.0) & (levels <= finite.max())]
    if negative.size:
        axis.contour(
            x, y, field, levels=negative, colors="0.25",
            linestyles="dashed", linewidths=0.75,
        )
    if positive.size:
        axis.contour(
            x, y, field, levels=positive, colors="black",
            linestyles="solid", linewidths=0.75,
        )
    if finite.min() <= 0.0 <= finite.max():
        axis.contour(
            x, y, field, levels=[0.0], colors="black",
            linestyles="solid", linewidths=1.35,
        )


def _plot_phase_pressure_structure(
    composite: xr.Dataset,
    wave_name: str,
):
    """Plot phase-pressure structure and experiment responses."""

    columns = (
        (
            "filtered_temperature",
            "filtered_upward_pressure_velocity",
            "Temperature with ascent",
            "K",
            r"Pa s$^{-1}$",
        ),
        (
            "filtered_specific_humidity",
            "filtered_relative_humidity",
            "Specific humidity with relative humidity",
            r"g kg$^{-1}$",
            "%",
        ),
        (
            "filtered_zonal_wind",
            None,
            "Zonal wind",
            r"m s$^{-1}$",
            None,
        ),
    )
    rows = SIMULATION_ROWS
    phase = np.asarray(composite.wave_phase, dtype=float)
    fig, axes = plt.subplots(
        len(rows), 3,
        figsize=(13, 3.0 * len(rows)), constrained_layout=True,
        sharex=True, sharey=True,
    )
    for column, (
        shade_name,
        contour_name,
        title,
        shade_unit,
        contour_unit,
    ) in enumerate(columns):
        absolute_stem = f"phase_pressure_{shade_name}"
        response_stem = f"response_phase_pressure_{shade_name}"
        absolute_limit = _symmetric_limit(
            composite[absolute_stem].sel(
                simulation="control", wave_band=wave_name
            )
        )
        response_limit = _symmetric_limit(
            composite[f"{response_stem}_mean"].sel(wave_band=wave_name)
        )
        if contour_name is not None:
            absolute_contour_stem = (
                f"phase_pressure_{contour_name}"
            )
            response_contour_stem = (
                f"response_phase_pressure_{contour_name}"
            )
            absolute_contour_limit = _symmetric_limit(
                composite[absolute_contour_stem].sel(
                    simulation="control", wave_band=wave_name
                ),
                percentile=90.0,
            )
            response_contour_limit = _symmetric_limit(
                composite[f"{response_contour_stem}_mean"].sel(
                    wave_band=wave_name
                ),
                percentile=90.0,
            )
        absolute_mappable = None
        response_mappable = None
        pressure_grid = np.asarray(
            composite[
                "phase_pressure_filtered_native_pressure_hpa"
            ].sel(
                simulation="control", wave_band=wave_name
            ).transpose("native_level", "wave_phase"),
            dtype=float,
        )
        fallback_pressure = composite.filtered_native_pressure_hpa.sel(
            simulation="control"
        )
        fallback_dimensions = tuple(
            dimension
            for dimension in fallback_pressure.dims
            if dimension != "native_level"
        )
        fallback_pressure = np.asarray(
            fallback_pressure.mean(
                fallback_dimensions, skipna=True
            ).transpose("native_level"),
            dtype=float,
        )
        pressure_grid = np.where(
            np.isfinite(pressure_grid) & (pressure_grid > 0.0),
            pressure_grid,
            fallback_pressure[:, None],
        )
        valid_native_levels = np.all(
            np.isfinite(pressure_grid) & (pressure_grid > 0.0), axis=1
        )
        native_level_indices = np.flatnonzero(valid_native_levels)
        if native_level_indices.size < 2:
            raise RuntimeError(
                f"{wave_name} has fewer than two finite native pressure "
                "levels for phase-pressure plotting"
            )
        pressure_grid = pressure_grid[valid_native_levels]
        phase_grid = np.broadcast_to(phase[None, :], pressure_grid.shape)
        for row, simulation in enumerate(rows):
            axis = axes[row, column]
            if row == 0:
                shade = composite[absolute_stem].sel(
                    simulation="control", wave_band=wave_name
                ).isel(native_level=native_level_indices).transpose(
                    "native_level", "wave_phase"
                )
                shade_limit = absolute_limit
                if contour_name is not None:
                    contour = composite[absolute_contour_stem].sel(
                        simulation="control", wave_band=wave_name
                    ).isel(native_level=native_level_indices).transpose(
                        "native_level", "wave_phase"
                    )
                    contour_limit = absolute_contour_limit
            else:
                shade = composite[f"{response_stem}_mean"].sel(
                    contrast=simulation, wave_band=wave_name
                ).isel(native_level=native_level_indices).transpose(
                    "native_level", "wave_phase"
                )
                shade_limit = response_limit
                if contour_name is not None:
                    contour = composite[
                        f"{response_contour_stem}_mean"
                    ].sel(
                        contrast=simulation, wave_band=wave_name
                    ).isel(native_level=native_level_indices).transpose(
                        "native_level", "wave_phase"
                    )
                    contour_limit = response_contour_limit
            mappable = axis.pcolormesh(
                phase_grid,
                pressure_grid,
                shade,
                cmap=(
                    SPECIFIC_HUMIDITY_CMAP
                    if shade_name == "filtered_specific_humidity"
                    else "RdBu_r"
                ),
                vmin=-shade_limit,
                vmax=shade_limit,
                shading="auto",
            )
            if row == 0:
                absolute_mappable = mappable
            else:
                response_mappable = mappable
                significant = composite[
                    f"{response_stem}_pointwise_significant"
                ].sel(
                    contrast=simulation, wave_band=wave_name
                ).isel(native_level=native_level_indices).transpose(
                    "native_level", "wave_phase"
                )
                _stipple_significance(
                    axis, phase_grid, pressure_grid, significant, stride=1
                )
            if contour_name is not None:
                _draw_standard_signed_contours(
                    axis,
                    phase_grid,
                    pressure_grid,
                    contour,
                    contour_limit,
                )
            _shade_ttl_pressure(axis)
            axis.set_yscale("log")
            axis.set_ylim(1000.0, 50.0)
            axis.set_xticks(np.arange(1, 9))
            if row == 0:
                axis.set_title(title)
            if column == 0:
                row_label = (
                    "Control"
                    if row == 0
                    else f"{SIMULATION_LABELS[simulation]} − control"
                )
                axis.set_ylabel(f"{row_label}\nPressure (hPa)")
            if row == len(rows) - 1:
                axis.set_xlabel("Filtered precipitation phase")
        fig.colorbar(
            absolute_mappable,
            ax=axes[0, column],
            label=f"Control shading ({shade_unit})",
            pad=0.01,
        )
        fig.colorbar(
            response_mappable,
            ax=axes[1:, column],
            label=f"Difference shading ({shade_unit})",
            pad=0.01,
        )
        if contour_name is not None:
            axes[0, column].text(
                0.02,
                0.03,
                f"Contours: {contour_unit}",
                transform=axes[0, column].transAxes,
                fontsize=7,
                va="bottom",
            )
    label = WAVE_BOXES[wave_name].get("label", wave_name)
    fig.suptitle(f"{label} strong-wave phase composites")
    return fig


def _plot_wave_phase_alignment(composite: xr.Dataset, wave_name: str):
    fig, axes = plt.subplots(
        len(SIMULATION_DIRS), 3, figsize=(12.5, 8.0),
        constrained_layout=True, sharex="col",
    )
    occurrence_mappable = None
    for row, simulation in enumerate(SIMULATION_DIRS):
        source = composite.sel(simulation=simulation, wave_band=wave_name)
        phase_fields = (
            "wave_phase_occurrence",
            "strong_wave_phase_occurrence",
            "random_strong_wave_phase_occurrence",
        )
        missing = [name for name in phase_fields if name not in source]
        if missing:
            raise RuntimeError(
                f"{wave_name} phase cache lacks matched baselines {missing}"
            )
        for column, variable in enumerate(phase_fields):
            occurrence = source[variable].transpose(
                "wave_phase", "lag_hours"
            )
            occurrence_mappable = axes[row, column].pcolormesh(
                occurrence.lag_hours, occurrence.wave_phase, occurrence,
                cmap="magma", vmin=0.0, vmax=1.0, shading="auto",
            )
            axes[row, column].set_yticks(
                range(1, 9),
                [str(phase_number) for phase_number in range(1, 9)],
            )
            axes[row, column].axvline(0.0, color="white", lw=0.7)
            if (
                column == 1
                and "strong_wave_phase_excess_over_random_significant"
                in source
            ):
                significant = source[
                    "strong_wave_phase_excess_over_random_significant"
                ].transpose("wave_phase", "lag_hours")
                phase_grid, lag_grid = np.meshgrid(
                    significant.wave_phase,
                    significant.lag_hours,
                    indexing="ij",
                )
                mask = np.asarray(significant, dtype=bool)
                axes[row, column].scatter(
                    lag_grid[mask],
                    phase_grid[mask],
                    s=10.0,
                    c="white",
                    edgecolors="black",
                    marker="o",
                    linewidths=0.35,
                )
        axes[row, 0].set_ylabel(f"{SIMULATION_LABELS[simulation]}\nPhase")
    axes[0, 0].set_title("All LPO events")
    axes[0, 1].set_title("Strong-wave LPO events")
    axes[0, 2].set_title("Matched random baseline")
    for axis in axes[-1]:
        axis.set_xlabel(LIFECYCLE_LAG_LABEL)
    if occurrence_mappable is not None:
        fig.colorbar(
            occurrence_mappable, ax=axes,
            label="Phase probability", pad=0.01,
        )
    fig.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=4,
                markerfacecolor="white", markeredgecolor="black",
                label="Different from random baseline",
            )
        ],
        loc="outside lower center",
        frameon=False,
    )
    label = WAVE_BOXES[wave_name].get("label", wave_name)
    fig.suptitle(f"{label} phase alignment")
    return fig


def _phase_alignment_window_label(wave_name: str) -> str:
    """Return the compact lag-window label for one wave band."""

    window = PHASE_ALIGNMENT_WINDOWS_HOURS[wave_name]
    if len(window) == 1:
        return "lag 0 h"
    return f"±{max(window)} h mean"


def _plot_cross_wave_phase_alignment(composite: xr.Dataset):
    """Compare near-peak phase profiles and complementary coupling metrics."""

    coupling_stem = "phase_window_alignment_strength"
    locking_stem = "phase_window_phase_locking_strength"
    required = {
        "phase_window_excess_over_random",
        "phase_window_excess_over_random_ci_low",
        "phase_window_excess_over_random_ci_high",
        coupling_stem,
        f"{coupling_stem}_ci_low",
        f"{coupling_stem}_ci_high",
        locking_stem,
        f"{locking_stem}_ci_low",
        f"{locking_stem}_ci_high",
        "strong_wave_event_count",
    }
    missing = sorted(required - set(composite.data_vars))
    if missing:
        raise RuntimeError(
            f"Cross-wave phase comparison lacks variables {missing}"
        )
    wave_colours = {
        wave_name: plt.get_cmap("tab10")(index)
        for index, wave_name in enumerate(LPO_LIFECYCLE_FILTERS)
    }
    interval_names = (
        "phase_window_excess_over_random_ci_low",
        "phase_window_excess_over_random_ci_high",
    )
    intervals = [
        np.asarray(composite[name], dtype=float)
        for name in interval_names
    ]
    finite = np.concatenate(
        [values[np.isfinite(values)] for values in intervals]
    )
    probability_limit = (
        max(0.05, float(np.nanmax(np.abs(finite))))
        if finite.size else 0.05
    )
    coupling_intervals = [
        np.asarray(composite[f"{coupling_stem}_{bound}"], dtype=float)
        for bound in ("ci_low", "ci_high")
    ]
    finite_coupling = np.concatenate(
        [values[np.isfinite(values)] for values in coupling_intervals]
    )
    coupling_limit = (
        max(0.05, float(np.nanmax(np.abs(finite_coupling))))
        if finite_coupling.size else 0.05
    )
    locking_intervals = [
        np.asarray(composite[f"{locking_stem}_{bound}"], dtype=float)
        for bound in ("ci_low", "ci_high")
    ]
    finite_locking = np.concatenate(
        [values[np.isfinite(values)] for values in locking_intervals]
    )
    locking_limit = (
        max(0.05, float(np.nanmax(np.abs(finite_locking))))
        if finite_locking.size else 0.05
    )
    fig, axes = plt.subplots(
        len(SIMULATION_DIRS),
        3,
        figsize=(16.2, 2.8 * len(SIMULATION_DIRS)),
        constrained_layout=True,
        sharex="col",
    )
    phase = np.asarray(composite.wave_phase, dtype=float)
    closed_phase = np.r_[phase, phase[0] + 8.0]
    wave_positions = np.arange(len(LPO_LIFECYCLE_FILTERS))[::-1]
    for row, simulation in enumerate(SIMULATION_DIRS):
        phase_axis, coupling_axis, locking_axis = axes[row]
        counts = []
        for wave_position, wave_name in zip(
            wave_positions, LPO_LIFECYCLE_FILTERS, strict=True
        ):
            source = composite.sel(
                simulation=simulation,
                wave_band=wave_name,
            )
            mean = np.asarray(
                source.phase_window_excess_over_random, dtype=float
            )
            low = np.asarray(
                source.phase_window_excess_over_random_ci_low, dtype=float
            )
            high = np.asarray(
                source.phase_window_excess_over_random_ci_high, dtype=float
            )
            colour = wave_colours[wave_name]
            label = WAVE_BOXES[wave_name].get("label", wave_name)
            phase_axis.plot(
                closed_phase,
                np.r_[mean, mean[0]],
                color=colour,
                label=label,
            )
            phase_axis.fill_between(
                closed_phase,
                np.r_[low, low[0]],
                np.r_[high, high[0]],
                color=colour,
                alpha=0.12,
            )
            coupling = float(source[coupling_stem])
            coupling_low = float(source[f"{coupling_stem}_ci_low"])
            coupling_high = float(source[f"{coupling_stem}_ci_high"])
            coupling_axis.errorbar(
                coupling,
                wave_position,
                xerr=np.asarray(
                    [
                        [max(0.0, coupling - coupling_low)],
                        [max(0.0, coupling_high - coupling)],
                    ]
                ),
                fmt="o",
                color=colour,
                markeredgecolor="black",
                markeredgewidth=0.4,
                capsize=2.0,
            )
            locking = float(source[locking_stem])
            locking_low = float(source[f"{locking_stem}_ci_low"])
            locking_high = float(source[f"{locking_stem}_ci_high"])
            locking_axis.errorbar(
                locking,
                wave_position,
                xerr=np.asarray(
                    [
                        [max(0.0, locking - locking_low)],
                        [max(0.0, locking_high - locking)],
                    ]
                ),
                fmt="o",
                color=colour,
                markeredgecolor="black",
                markeredgewidth=0.4,
                capsize=2.0,
            )
            window = PHASE_ALIGNMENT_WINDOWS_HOURS[wave_name]
            count_values = np.asarray(
                source.strong_wave_event_count.sel(
                    lag_hours=list(window)
                ),
                dtype=float,
            )
            available = count_values[
                np.isfinite(count_values) & (count_values > 0.0)
            ]
            counts.append(
                int(np.median(available)) if available.size else 0
            )
        coupling_axis.set_yticks(
            wave_positions,
            [
                f"{WAVE_BOXES[wave_name].get('label', wave_name)} "
                f"({_phase_alignment_window_label(wave_name)}; "
                f"median n={count})"
                for wave_name, count in zip(
                    LPO_LIFECYCLE_FILTERS, counts, strict=True
                )
            ],
        )
        locking_axis.set_yticks(wave_positions, [""] * len(wave_positions))
        phase_axis.axhline(0.0, color="0.45", lw=0.7)
        phase_axis.set_ylim(
            -1.05 * probability_limit, 1.05 * probability_limit
        )
        phase_axis.set_ylabel(
            f"{SIMULATION_LABELS[simulation]}\nExcess probability"
        )
        coupling_axis.axvline(0.0, color="0.45", lw=0.7)
        coupling_axis.text(
            0.02,
            0.95,
            "Total shifted phase probability",
            transform=coupling_axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        locking_axis.axvline(0.0, color="0.45", lw=0.7)
        locking_axis.text(
            0.02,
            0.95,
            "LPO concentration − matched random",
            transform=locking_axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        for axis in (phase_axis, coupling_axis, locking_axis):
            axis.grid(alpha=0.2)
    for coupling_axis in axes[:, 1]:
        coupling_axis.set_xlim(0.0, min(1.0, 1.05 * coupling_limit))
    for locking_axis in axes[:, 2]:
        locking_axis.set_xlim(
            -min(1.0, 1.05 * locking_limit),
            min(1.0, 1.05 * locking_limit),
        )
    axes[0, 0].set_title("Temporally averaged excess phase probability")
    axes[0, 1].set_title("Near-peak distribution separation")
    axes[0, 2].set_title("Near-peak phase-locking strength")
    axes[-1, 0].set_xlabel("Filtered precipitation phase")
    axes[-1, 1].set_xlabel("Probability shifted from matched random")
    axes[-1, 2].set_xlabel("Circular concentration above matched random")
    axes[-1, 0].set_xticks(np.arange(1, 10))
    axes[-1, 0].set_xticklabels([*map(str, range(1, 9)), "1"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(LPO_LIFECYCLE_FILTERS),
        frameon=False,
    )
    fig.suptitle(
        "Wave–convection phase alignment near the LPO precipitation peak"
    )
    return fig


def _plot_wave_coupling_lifecycle(composite: xr.Dataset):
    """Plot phase-distribution separation through the LPO lifecycle."""

    stem = "wave_phase_alignment_strength"
    required = {stem, f"{stem}_ci_low", f"{stem}_ci_high"}
    missing = sorted(required - set(composite.data_vars))
    if missing:
        raise RuntimeError(
            f"Wave-coupling lifecycle lacks variables {missing}"
        )
    interval_values = np.concatenate(
        [
            np.asarray(composite[f"{stem}_{bound}"], dtype=float).ravel()
            for bound in ("ci_low", "ci_high")
        ]
    )
    finite = interval_values[np.isfinite(interval_values)]
    probability_limit = (
        min(1.0, max(0.05, float(np.nanmax(finite))))
        if finite.size else 0.05
    )
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(11.5, 9.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()
    for axis, wave_name in zip(
        axes_flat, LPO_LIFECYCLE_FILTERS, strict=False
    ):
        window = PHASE_ALIGNMENT_WINDOWS_HOURS[wave_name]
        if len(window) > 1:
            axis.axvspan(
                min(window), max(window), color="0.9", alpha=0.45
            )
        for simulation in SIMULATION_DIRS:
            source = composite.sel(
                simulation=simulation,
                wave_band=wave_name,
            )
            colour = SIMULATION_COLOURS[simulation]
            axis.plot(
                source.lag_hours,
                source[stem],
                color=colour,
                label=SIMULATION_LABELS[simulation],
            )
            axis.fill_between(
                source.lag_hours,
                source[f"{stem}_ci_low"],
                source[f"{stem}_ci_high"],
                color=colour,
                alpha=0.12,
            )
        axis.axhline(0.0, color="0.45", lw=0.7)
        axis.axvline(0.0, color="0.45", lw=0.7)
        axis.set_ylim(
            0.0, min(1.0, 1.05 * probability_limit)
        )
        label = WAVE_BOXES[wave_name].get("label", wave_name)
        axis.set_title(
            f"{label} ({_phase_alignment_window_label(wave_name)})"
        )
        axis.grid(alpha=0.2)
    for axis in axes_flat[len(LPO_LIFECYCLE_FILTERS):]:
        axis.set_axis_off()
    fig.supxlabel(LIFECYCLE_LAG_LABEL)
    fig.supylabel(
        "Phase-alignment strength: probability shifted from matched random"
    )
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(SIMULATION_DIRS),
        frameon=False,
    )
    fig.suptitle("Wave phase alignment through the LPO lifecycle")
    return fig


def _plot_wave_coupling_response(composite: xr.Dataset):
    """Plot direct simulation changes in near-peak wave coupling."""

    metrics = (
        (
            "phase_window_alignment_strength",
            "Distribution separation",
        ),
        (
            "phase_window_phase_locking_strength",
            "Phase-locking strength",
        ),
    )
    required = {
        f"response_{stem}_{statistic}"
        for stem, _title in metrics
        for statistic in (
            "mean", "ci_low", "ci_high", "pointwise_significant"
        )
    }
    missing = sorted(required - set(composite.data_vars))
    if missing:
        raise RuntimeError(
            f"Wave-coupling response comparison lacks variables {missing}"
        )
    contrasts = [
        simulation
        for simulation in RESPONSE_ROWS
        if simulation in composite.contrast.values
    ]
    if not contrasts:
        raise RuntimeError("Wave-coupling response has no simulations")
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(11.5, 4.8),
        constrained_layout=True,
        sharey=True,
    )
    wave_positions = np.arange(len(LPO_LIFECYCLE_FILTERS))[::-1]
    offsets = np.linspace(0.12, -0.12, len(contrasts))
    for axis, (stem, title) in zip(axes, metrics, strict=True):
        response_stem = f"response_{stem}"
        interval_values = np.concatenate(
            [
                np.asarray(
                    composite[f"{response_stem}_{bound}"], dtype=float
                ).ravel()
                for bound in ("ci_low", "ci_high")
            ]
        )
        finite = interval_values[np.isfinite(interval_values)]
        limit = (
            max(0.1, 1.05 * float(np.nanmax(np.abs(finite))))
            if finite.size else 0.1
        )
        for offset, simulation in zip(offsets, contrasts, strict=True):
            source = composite.sel(contrast=simulation)
            colour = SIMULATION_COLOURS[simulation]
            for wave_position, wave_name in zip(
                wave_positions, LPO_LIFECYCLE_FILTERS, strict=True
            ):
                point = float(
                    source[f"{response_stem}_mean"].sel(
                        wave_band=wave_name
                    )
                )
                low = float(
                    source[f"{response_stem}_ci_low"].sel(
                        wave_band=wave_name
                    )
                )
                high = float(
                    source[f"{response_stem}_ci_high"].sel(
                        wave_band=wave_name
                    )
                )
                significant = bool(
                    source[f"{response_stem}_pointwise_significant"].sel(
                        wave_band=wave_name
                    )
                )
                axis.errorbar(
                    point,
                    wave_position + offset,
                    xerr=np.asarray(
                        [
                            [max(0.0, point - low)],
                            [max(0.0, high - point)],
                        ]
                    ),
                    fmt="o",
                    color=colour,
                    markerfacecolor=colour if significant else "white",
                    markeredgecolor=colour,
                    capsize=2.0,
                    label=(
                        SIMULATION_LABELS[simulation]
                        if wave_position == wave_positions[0]
                        else None
                    ),
                )
        axis.axvline(0.0, color="0.35", lw=0.8)
        axis.set_xlim(-limit, limit)
        axis.set_title(title)
        axis.set_xlabel("Change from control")
        axis.grid(alpha=0.2)
    axes[0].set_yticks(
        wave_positions,
        [
            f"{WAVE_BOXES[wave_name].get('label', wave_name)} "
            f"({_phase_alignment_window_label(wave_name)})"
            for wave_name in LPO_LIFECYCLE_FILTERS
        ],
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(contrasts),
        frameon=False,
    )
    fig.suptitle(
        "Changes in near-peak LPO–wave coupling\n"
        "Filled markers: 95% bootstrap interval excludes zero"
    )
    return fig


def _plot_wave_phase_validation(composite: xr.Dataset):
    """Collect supporting phase-coherence, sample, and amplitude measures."""

    fig, axes = plt.subplots(
        len(LPO_LIFECYCLE_FILTERS),
        4,
        figsize=(14.5, 9.0),
        constrained_layout=True,
        sharex=True,
    )
    for row, wave_name in enumerate(LPO_LIFECYCLE_FILTERS):
        for simulation in SIMULATION_DIRS:
            source = composite.sel(
                simulation=simulation,
                wave_band=wave_name,
            )
            colour = SIMULATION_COLOURS[simulation]
            label = SIMULATION_LABELS[simulation]
            axes[row, 0].plot(
                source.lag_hours,
                source.wave_phase_circular_concentration,
                color=colour,
                label=label,
            )
            axes[row, 0].fill_between(
                source.lag_hours,
                source.wave_phase_circular_concentration_ci_low,
                source.wave_phase_circular_concentration_ci_high,
                color=colour,
                alpha=0.12,
            )
            axes[row, 1].plot(
                source.lag_hours,
                source.strong_wave_fraction,
                color=colour,
            )
            axes[row, 2].plot(
                source.lag_hours,
                source.strong_wave_event_count,
                color=colour,
            )
            axes[row, 3].plot(
                source.lag_hours,
                source.wave_normalized_amplitude,
                color=colour,
            )
            axes[row, 3].fill_between(
                source.lag_hours,
                source.wave_normalized_amplitude_q25,
                source.wave_normalized_amplitude_q75,
                color=colour,
                alpha=0.12,
            )
        axes[row, 0].set_ylim(0.0, 1.0)
        axes[row, 1].set_ylim(0.0, 1.0)
        axes[row, 3].axhline(1.0, color="0.4", lw=0.7, ls="--")
        axes[row, 0].set_ylabel(
            WAVE_BOXES[wave_name].get("label", wave_name)
        )
        for axis in axes[row]:
            axis.axvline(0.0, color="0.5", lw=0.7)
            axis.grid(alpha=0.2)
    for axis, title in zip(
        axes[0],
        (
            "Strong-event phase coherence",
            "Strong-wave fraction",
            "Strong-wave event count",
            "Normalized amplitude median and IQR",
        ),
        strict=True,
    ):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel(LIFECYCLE_LAG_LABEL)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(SIMULATION_DIRS),
        frameon=False,
    )
    fig.suptitle("Wave validation")
    return fig


def _plot_matsuno_gill_response(composite: xr.Dataset):
    """Plot eastward-LPO wind and vorticity anomaly composites."""

    pressures = (850.0, 200.0)
    rows = SIMULATION_ROWS
    display_vorticity = [
        composite.matsuno_relative_vorticity_anomaly.sel(
            simulation=simulation
        )
        for simulation in rows
    ]
    limit = _symmetric_limit(xr.concat(display_vorticity, dim="panel"))
    fig, axes = plt.subplots(
        len(rows), len(pressures),
        figsize=(12, 10), constrained_layout=True,
        sharex=True, sharey=True,
    )
    mappable = None
    for row, simulation in enumerate(rows):
        included = int(
            composite.matsuno_eastward_event_count.sel(
                simulation=simulation
            )
        )
        total = int(
            composite.matsuno_total_event_count.sel(
                simulation=simulation
            )
        )
        for column, pressure in enumerate(pressures):
            axis = axes[row, column]
            fields = {}
            for key in (
                "matsuno_relative_vorticity_anomaly",
                "matsuno_zonal_wind_anomaly",
                "matsuno_meridional_wind_anomaly",
            ):
                source = composite[key].sel(
                    simulation=simulation,
                    matsuno_pressure_hpa=pressure,
                )
                fields[key] = source
            vorticity = fields[
                "matsuno_relative_vorticity_anomaly"
            ].transpose(
                "matsuno_relative_latitude",
                "matsuno_relative_longitude",
            )
            mappable = axis.pcolormesh(
                vorticity.matsuno_relative_longitude,
                vorticity.matsuno_relative_latitude,
                1.0e5 * vorticity,
                cmap="RdBu_r",
                vmin=-1.0e5 * limit,
                vmax=1.0e5 * limit,
                shading="auto",
            )
            stride = 4
            u = fields["matsuno_zonal_wind_anomaly"].transpose(
                "matsuno_relative_latitude",
                "matsuno_relative_longitude",
            )
            v = fields["matsuno_meridional_wind_anomaly"].transpose(
                "matsuno_relative_latitude",
                "matsuno_relative_longitude",
            )
            finite_vectors = np.isfinite(u.values) & np.isfinite(v.values)
            if included > 0 and finite_vectors.any():
                axis.quiver(
                    u.matsuno_relative_longitude[::stride],
                    u.matsuno_relative_latitude[::stride],
                    u.values[::stride, ::stride],
                    v.values[::stride, ::stride],
                    color="black",
                    scale=None,
                    width=0.0022,
                )
            else:
                short_sample = bool(
                    composite.attrs.get("test_subset", False)
                )
                message = "No eastward events"
                if short_sample:
                    message += "\nin short test sample"
                axis.text(
                    0.5,
                    0.5,
                    message,
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.axhline(0.0, color="0.4", lw=0.6)
            axis.axvline(0.0, color="0.4", lw=0.6)
            if row == 0:
                axis.set_title(f"{pressure:g} hPa")
            if column == 0:
                label = SIMULATION_LABELS[simulation]
                axis.set_ylabel(
                    f"{label}\nLatitude from centroid (°)\n"
                    f"eastward n={included}/{total}"
                )
            if row == len(rows) - 1:
                axis.set_xlabel("Longitude from centroid (°)")
    fig.colorbar(
        mappable,
        ax=axes,
        label=r"Relative vorticity anomaly ($10^{-5}$ s$^{-1}$)",
        extend="both",
        pad=0.01,
    )
    fig.suptitle(
        "Matsuno-Gill response at geometric LPO centroids\n"
        "Own-simulation climatologies; eastward geometric propagation only"
    )
    return fig


def _draw_peak_rays(
    axis,
    composite: xr.Dataset,
    simulation: str,
    wave_name: str,
    *,
    include_control: bool,
) -> None:
    event_key = f"{wave_name.lower()}_ray_relative_longitude"
    background_key = (
        f"{wave_name.lower()}_background_ray_relative_longitude"
    )

    def draw(
        source_simulation: str,
        *,
        key: str,
        pressure_name: str,
        color: str,
        linestyle: str,
    ) -> None:
        longitude = composite[key].sel(simulation=source_simulation)
        pressure = composite[pressure_name].sel(
            simulation=source_simulation
        )
        if "ray_relative_longitude" in pressure.dims:
            pressure = pressure.mean(
                "ray_relative_longitude", skipna=True
            )
        for launch in longitude.ray_launch_longitude:
            axis.plot(
                longitude.sel(ray_launch_longitude=launch),
                pressure,
                color=color,
                linestyle=linestyle,
                lw=0.8,
                alpha=0.85,
            )

    def draw_pair(source_simulation: str, linestyle: str) -> None:
        draw(
            source_simulation,
            key=event_key,
            pressure_name="ray_pressure_hpa",
            color="tab:purple",
            linestyle=linestyle,
        )
        draw(
            source_simulation,
            key=background_key,
            pressure_name="ray_unconditional_pressure_hpa",
            color="tab:green",
            linestyle=linestyle,
        )

    if include_control:
        draw_pair("control", "--")
    draw_pair(simulation, "-")


def _plot_peak_longitude_height(
    composite: xr.Dataset,
    variable: str,
    *,
    title: str,
    units: str,
    wave_name: str = "Kelvin",
    control_cmap=None,
    control_limits: tuple[float, float] | None = None,
    differences_to_control: bool = True,
):
    """Plot peak longitude-height sections for each simulation."""

    rows = SIMULATION_ROWS
    source = composite[variable]
    if "ray_wave_band" in source.dims:
        source = source.sel(ray_wave_band=wave_name)
    control = source.sel(simulation="control")
    displayed = {
        simulation: (
            source.sel(simulation=simulation)
            if simulation == "control" or not differences_to_control
            else source.sel(simulation=simulation) - control
        )
        for simulation in rows
    }
    control_values = np.asarray(displayed["control"], dtype=float)
    control_finite = control_values[np.isfinite(control_values)]
    positive_control = bool(
        control_finite.size and np.nanmin(control_finite) >= 0.0
    )
    if positive_control or control_cmap is not None:
        if control_limits is None:
            lower = (
                min(0.0, float(np.percentile(control_finite, 2.0)))
                if control_finite.size
                else 0.0
            )
            upper = (
                float(np.percentile(control_finite, 98.0))
                if control_finite.size
                else 1.0
            )
        else:
            lower, upper = control_limits
        control_style = {
            "cmap": control_cmap or "viridis",
            "vmin": lower,
            "vmax": upper if upper > lower else lower + 1.0,
        }
    else:
        limit = _symmetric_limit(displayed["control"])
        control_style = {
            "cmap": "RdBu_r", "vmin": -limit, "vmax": limit,
        }
    response_limit = _symmetric_limit(
        xr.concat(
            [displayed[simulation] for simulation in RESPONSE_ROWS],
            dim="contrast",
        )
    )
    response_style = {
        "cmap": "RdBu_r",
        "vmin": -response_limit,
        "vmax": response_limit,
    }
    fig = plt.figure(
        figsize=(11, 4.0 * len(rows)), constrained_layout=True
    )
    grid_rows = 1 + 2 * len(rows)
    grid = fig.add_gridspec(
        grid_rows,
        2,
        height_ratios=(0.13, *((1.0, 0.24) * len(rows))),
        width_ratios=(1.0, 0.035),
    )
    title_axis = fig.add_subplot(grid[0, :])
    title_axis.set_axis_off()
    mappables = []
    for row, simulation in enumerate(rows):
        axis = fig.add_subplot(grid[2 * row + 1, 0])
        precipitation_axis = fig.add_subplot(
            grid[2 * row + 2, 0], sharex=axis
        )
        field = displayed[simulation].transpose(
            "ray_height_m", "ray_relative_longitude"
        )
        pressure = composite.ray_pressure_hpa.sel(
            simulation=simulation
        ).transpose("ray_height_m", "ray_relative_longitude")
        pressure_values = np.asarray(pressure, dtype=float)
        available_heights = np.any(
            np.isfinite(pressure_values) & (pressure_values > 0.0), axis=1
        )
        if np.count_nonzero(available_heights) < 2:
            raise RuntimeError(
                f"{simulation} {wave_name} ray plot has fewer than two "
                "pressure levels"
            )
        height_indices = np.flatnonzero(available_heights)
        field = field.isel(ray_height_m=height_indices)
        pressure_values = _fill_native_pressure_coordinates(
            pressure_values[available_heights],
            context=f"{simulation} {wave_name} ray plot",
        )
        longitude_grid = np.broadcast_to(
            np.asarray(field.ray_relative_longitude)[None, :], field.shape
        )
        style = control_style if simulation == "control" else response_style
        mappable = axis.pcolormesh(
            longitude_grid,
            pressure_values,
            field,
            shading="auto",
            **style,
        )
        mappables.append(mappable)
        cloud = composite.ray_nonprecipitating_cloud_occurrence.sel(
            simulation=simulation
        )
        if simulation != "control" and differences_to_control:
            cloud = cloud - composite.ray_nonprecipitating_cloud_occurrence.sel(
                simulation="control"
            )
        cloud = cloud.transpose("ray_height_m", "ray_relative_longitude")
        cloud = cloud.isel(ray_height_m=height_indices)
        cloud_limit = _symmetric_limit(cloud, percentile=90.0)
        if simulation == "control" or not differences_to_control:
            cloud_values = np.asarray(cloud, dtype=float)
            cloud_values = cloud_values[
                np.isfinite(cloud_values) & (cloud_values > 0.0)
            ]
            levels = (
                np.unique(np.percentile(cloud_values, [60.0, 80.0, 95.0]))
                if cloud_values.size
                else np.asarray([])
            )
        else:
            levels = np.asarray(
                [-cloud_limit, -0.5 * cloud_limit,
                 0.5 * cloud_limit, cloud_limit]
            )
        if levels.size and np.any(np.isfinite(cloud)):
            negative = levels[levels < 0.0]
            positive = levels[levels > 0.0]
            if negative.size:
                axis.contour(
                    longitude_grid, pressure_values, cloud,
                    levels=negative, colors="0.25",
                    linestyles="dashed", linewidths=0.7,
                )
            if positive.size:
                axis.contour(
                    longitude_grid, pressure_values, cloud,
                    levels=positive, colors="black",
                    linestyles="solid", linewidths=0.7,
                )
        _draw_peak_rays(
            axis,
            composite,
            simulation,
            wave_name,
            include_control=simulation != "control",
        )
        axis.set_yscale("log")
        axis.set_ylim(1000.0, 50.0)
        axis.set_xlim(
            float(field.ray_relative_longitude.min()),
            float(field.ray_relative_longitude.max()),
        )
        label = (
            "Control"
            if simulation == "control"
            else (
                f"{SIMULATION_LABELS[simulation]} − control"
                if differences_to_control
                else SIMULATION_LABELS[simulation]
            )
        )
        axis.set_ylabel(f"{label}\nPressure (hPa)")
        axis.tick_params(labelbottom=False)

        precipitation = composite.ray_lpo_precipitation.sel(
            simulation=simulation
        )
        if simulation != "control" and differences_to_control:
            precipitation = precipitation - composite.ray_lpo_precipitation.sel(
                simulation="control"
            )
            precipitation_axis.axhline(0.0, color="0.4", lw=0.6)
        precipitation_axis.plot(
            precipitation.ray_relative_longitude,
            precipitation,
            color=SIMULATION_COLOURS[simulation],
            lw=1.3,
        )
        precipitation_axis.set_ylabel("Rain\n(mm day⁻¹)")
        if row == len(rows) - 1:
            precipitation_axis.set_xlabel("Longitude from centroid (°)")
    control_colorbar_axis = fig.add_subplot(grid[1, 1])
    response_colorbar_axis = fig.add_subplot(grid[3:, 1])
    fig.colorbar(
        mappables[0], cax=control_colorbar_axis,
        label=f"Control {units}",
    )
    fig.colorbar(
        mappables[-1], cax=response_colorbar_axis,
        label=(
            f"Difference {units}"
            if differences_to_control
            else f"Simulation {units}"
        ),
        extend="both",
    )
    title_axis.text(
        0.5,
        0.5,
        f"{title} at geometric-centroid precipitation peak\n"
        "Non-precipitating high-cloud contours; rays launch at 200 hPa; "
        "purple: event-local U,N; green: unconditional U,N",
        ha="center",
        va="center",
        fontsize=plt.rcParams["figure.titlesize"],
        fontweight=plt.rcParams["figure.titleweight"],
        transform=title_axis.transAxes,
    )
    return fig


def build_lifecycle_figures(
    mode_directory: Path,
    *,
    figures: Sequence[int | str],
    figure_output: Path,
    test: bool = False,
) -> None:
    primary = mode_directory / "primary"
    lifecycle, summary = read_lifecycle_products(primary)
    savers: dict[str, FigureSaver] = {}

    def save(folder: str, number: int, stem: str, figure) -> None:
        if folder not in savers:
            savers[folder] = FigureSaver(
                str(Path(figure_output) / folder), enabled=True
            )
        savers[folder].save(figure, stem, number=number)
        plt.close(figure)

    if 4 in figures:
        progress("Starting lpo_lifecycle Figure 1: cloud populations")
        save(
            "lpo_lifecycle", 1, "cloud_population_lifecycle",
            _plot_lifecycle(lifecycle),
        )
    if 5 in figures:
        progress("Starting lpo_lifecycle Figure 2: lifecycle timing")
        save(
            "lpo_lifecycle", 2, "lifecycle_timing",
            _plot_lifecycle_timing(summary),
        )
    if 6 in figures:
        progress("Starting lpo_lifecycle Figure 3: spatial evolution")
        save(
            "lpo_lifecycle", 3, "centred_spatial_evolution",
            _plot_event_centred_evolution(primary, lifecycle),
        )
    if 7 in figures:
        progress("Starting lpo_environment Figure 1: UTLS environment")
        save(
            "lpo_environment", 1, "utls_environment",
            _plot_environment(lifecycle),
        )
        for subfigure, wave_name in zip(
            "abcde", LPO_LIFECYCLE_FILTERS, strict=True
        ):
            progress(
                "Starting lpo_wave_alignment Figure 1"
                f"{subfigure}: {wave_name} lifecycle"
            )
            save(
                "lpo_wave_alignment", 1,
                f"{subfigure}_lpo_{_wave_key(wave_name)}_lifecycle",
                _plot_wave_lifecycle(lifecycle, wave_name),
            )
    if 8 in figures:
        progress("Starting lpo_lifecycle Figure 4: normalized track")
        save(
            "lpo_lifecycle", 4, "normalized_whole_track_lifecycle",
            _plot_normalized_lifetime(lifecycle),
        )
    if any(number == "45a" or number >= 9 for number in figures):
        composite_path = primary / LIFECYCLE_2D_COMPOSITE_FILENAME
        if not composite_path.is_file():
            raise FileNotFoundError(
                f"Missing cached 2-D lifecycle product {composite_path}"
            )
        with xr.open_dataset(composite_path) as composite:
            if composite.attrs.get("cache_version") != COMPOSITE_CACHE_VERSION:
                raise RuntimeError(
                    f"{composite_path} predates {COMPOSITE_CACHE_VERSION}"
                )
            if (
                composite.attrs.get("tracking_product_version")
                != PRODUCT_VERSION
            ):
                raise RuntimeError(
                    f"{composite_path} does not match {PRODUCT_VERSION}"
                )
            if 9 in figures:
                progress("Starting lpo_lifecycle Figure 5: modal cloud class")
                save(
                    "lpo_lifecycle", 5, "modal_high_cloud_context",
                    _plot_cloud_modal_class(composite),
                )
            if 10 in figures:
                for subfigure, simulation in zip(
                    "abcdefghijklmnopqrstuvwxyz", SIMULATION_ROWS
                ):
                    progress(
                        "Starting lpo_lifecycle Figure 6"
                        f"{subfigure}: {simulation} cloud occurrence"
                    )
                    save(
                        "lpo_lifecycle", 6,
                        (
                            f"{subfigure}_lpo_{simulation}_"
                            "cloud_class_occurrence"
                        ),
                        _plot_cloud_occurrences(composite, simulation),
                    )
            if 11 in figures:
                progress("Starting lpo_environment Figure 2: surface absolute")
                save(
                    "lpo_environment", 2, "unfiltered_surface_absolute",
                    _plot_surface_sections(composite),
                )
            if 12 in figures:
                progress("Starting lpo_environment Figure 3: surface responses")
                save(
                    "lpo_environment", 3, "unfiltered_surface_responses",
                    _plot_surface_sections(composite, response=True),
                )
            if 13 in figures:
                progress("Starting lpo_environment Figure 4: vertical absolute")
                save(
                    "lpo_environment", 4, "unfiltered_vertical_absolute",
                    _plot_vertical_sections(composite),
                )
            if 14 in figures:
                progress("Starting lpo_environment Figure 5: vertical responses")
                save(
                    "lpo_environment", 5, "unfiltered_vertical_responses",
                    _plot_vertical_sections(composite, response=True),
                )
            if 15 in figures:
                progress(
                    "Starting lpo_wave_propagation Figure 1: "
                    "Kelvin group velocities"
                )
                save(
                    "lpo_wave_propagation", 1,
                    "kelvin_reference_group_velocity",
                    _plot_kelvin_reference_lifecycle(composite),
                )
            if 16 in figures:
                progress(
                    "Starting lpo_sensitivity Figure 1: averaging region"
                )
                save(
                    "lpo_sensitivity", 1,
                    "longitude_averaging_sensitivity",
                    _plot_averaging_sensitivity(composite),
                )
            legacy_wave_names = tuple(
                wave_name for wave_name in LPO_LIFECYCLE_FILTERS
                if wave_name != "MRG"
            )
            for offset, wave_name in enumerate(legacy_wave_names):
                surface_number = 17 + offset
                vertical_number = 21 + offset
                phase_number = 25 + offset
                phase_pressure_number = 36 + offset
                key = _wave_key(wave_name)
                subfigure = "abcde"[
                    LPO_LIFECYCLE_FILTERS.index(wave_name)
                ]
                if surface_number in figures:
                    progress(
                        "Starting lpo_wave_alignment Figure 2"
                        f"{subfigure}: {wave_name} lag-latitude"
                    )
                    save(
                        "lpo_wave_alignment", 2,
                        f"{subfigure}_lpo_{key}_filtered_surface",
                        _plot_surface_sections(
                            composite, wave_name=wave_name
                        ),
                    )
                if vertical_number in figures:
                    progress(
                        "Starting lpo_wave_alignment Figure 3"
                        f"{subfigure}: {wave_name} lag-pressure"
                    )
                    save(
                        "lpo_wave_alignment", 3,
                        f"{subfigure}_lpo_{key}_filtered_vertical",
                        _plot_vertical_sections(
                            composite, wave_name=wave_name
                        ),
                    )
                if phase_number in figures:
                    progress(
                        "Starting lpo_wave_alignment Figure 4"
                        f"{subfigure}: {wave_name} phase"
                    )
                    save(
                        "lpo_wave_alignment", 4,
                        f"{subfigure}_lpo_{key}_phase_alignment",
                        _plot_wave_phase_alignment(composite, wave_name),
                    )
                if phase_pressure_number in figures:
                    progress(
                        "Starting lpo_wave_alignment Figure 5"
                        f"{subfigure}: {wave_name} phase-pressure"
                    )
                    save(
                        "lpo_wave_alignment", 5,
                        f"{subfigure}_lpo_{key}_phase_pressure_structure",
                        _plot_phase_pressure_structure(
                            composite, wave_name
                        ),
                    )
            mrg_figure_map = {
                47: ("surface", _plot_surface_sections),
                48: ("vertical", _plot_vertical_sections),
                49: ("phase", _plot_wave_phase_alignment),
                50: ("phase_pressure", _plot_phase_pressure_structure),
            }
            for number, (kind, plotter) in mrg_figure_map.items():
                if number not in figures:
                    continue
                subfigure = "abcde"[
                    LPO_LIFECYCLE_FILTERS.index("MRG")
                ]
                local_number = {
                    "surface": 2,
                    "vertical": 3,
                    "phase": 4,
                    "phase_pressure": 5,
                }[kind]
                progress(
                    "Starting lpo_wave_alignment Figure "
                    f"{local_number}{subfigure}: MRG {kind}"
                )
                if kind in {"surface", "vertical"}:
                    figure = plotter(composite, wave_name="MRG")
                else:
                    figure = plotter(composite, "MRG")
                save(
                    "lpo_wave_alignment", local_number,
                    f"{subfigure}_lpo_mrg_{kind}", figure,
                )
            if 40 in figures:
                progress(
                    "Starting lpo_sensitivity Figure 3: wave-phase validation"
                )
                save(
                    "lpo_sensitivity", 3,
                    "wave_phase_validation_metrics",
                    _plot_wave_phase_validation(composite),
                )
            if 29 in figures:
                progress("Starting lpo_lifecycle Figure 7: cloud depth")
                save(
                    "lpo_lifecycle", 7, "cloud_depth_lifecycle",
                    _plot_cloud_depth_lifecycle(lifecycle),
                )
            if 30 in figures:
                progress(
                    "Starting lpo_environment Figure 6: near-peak profiles"
                )
                save(
                    "lpo_environment", 6, "near_peak_vertical_profiles",
                    _plot_near_peak_vertical_profiles(primary),
                )
            if 31 in figures:
                progress(
                    "Starting lpo_environment Figure 7: cloud responses"
                )
                save(
                    "lpo_environment", 7,
                    "cloud_occurrence_responses_pointwise",
                    _plot_cloud_occurrence_response(composite),
                )
            if 32 in figures:
                progress(
                    "Starting lpo_environment Figure 8: FDR cloud responses"
                )
                save(
                    "lpo_environment", 8,
                    "cloud_occurrence_responses_fdr",
                    _plot_cloud_occurrence_response(
                        composite,
                        significance_variable="fdr_significant",
                    ),
                )
            if 33 in figures:
                progress(
                    "Starting lpo_environment Figure 9: FDR vertical responses"
                )
                save(
                    "lpo_environment", 9,
                    "unfiltered_vertical_responses_fdr",
                    _plot_unfiltered_vertical_response(
                        composite,
                        significance_variable="fdr_significant",
                    ),
                )
            if 34 in figures:
                progress(
                    "Starting lpo_wave_propagation Figure 2: "
                    "absolute group velocities"
                )
                save(
                    "lpo_wave_propagation", 2,
                    "kelvin_reference_group_velocity_absolute",
                    _plot_kelvin_reference_absolute(composite),
                )
            if 35 in figures:
                progress(
                    "Starting lpo_sensitivity Figure 2: group velocities"
                )
                save(
                    "lpo_sensitivity", 2,
                    "kelvin_reference_group_velocity_sensitivity",
                    _plot_group_velocity_sensitivity(composite),
                )
            if 41 in figures:
                progress(
                    "Starting lpo_wave_propagation Figure 3: Matsuno-Gill"
                )
                save(
                    "lpo_wave_propagation", 3, "matsuno_gill_response",
                    _plot_matsuno_gill_response(composite),
                )
            if 42 in figures:
                progress(
                    "Starting lpo_wave_propagation Figure 4a: Kelvin rays"
                )
                save(
                    "lpo_wave_propagation", 4,
                    "a_kelvin_peak_temperature_rays",
                    _plot_peak_longitude_height(
                        composite,
                        "ray_filtered_temperature",
                        title="Kelvin-filtered temperature",
                        units="temperature (K)",
                        wave_name="Kelvin",
                    ),
                )
            if 43 in figures:
                progress(
                    "Starting lpo_wave_propagation Figure 4b: stability rays"
                )
                save(
                    "lpo_wave_propagation", 4,
                    "b_peak_static_stability_rays",
                    _plot_peak_longitude_height(
                        composite,
                        "ray_unfiltered_static_stability",
                        title="Unfiltered static stability",
                        units=r"$N^2$ (s$^{-2}$)",
                        control_cmap="YlGnBu",
                    ),
                )
            if 44 in figures:
                progress(
                    "Starting lpo_wave_propagation Figure 4c: cloud rays"
                )
                save(
                    "lpo_wave_propagation", 4,
                    "c_peak_high_cloud_occurrence_rays",
                    _plot_peak_longitude_height(
                        composite,
                        "ray_high_cloud_occurrence",
                        title="Unfiltered high-cloud occurrence",
                        units="occurrence",
                        control_cmap=blue_sky_white_cloud_cmap(),
                    ),
                )
            if 45 in figures:
                progress(
                    "Starting lpo_wave_propagation Figure 4d: N2 responses"
                )
                save(
                    "lpo_wave_propagation", 4,
                    "d_conditioned_static_stability_anomaly",
                    _plot_peak_longitude_height(
                        composite,
                        "ray_lpo_conditioned_static_stability_anomaly",
                        title=(
                            "LPO-conditioned static stability minus "
                            "unconditional tropical background"
                        ),
                        units=r"$N^2$ anomaly (s$^{-2}$)",
                    ),
                )
            if "45a" in figures:
                progress(
                    "Starting lpo_wave_propagation Figure 4e: absolute N2"
                )
                save(
                    "lpo_wave_propagation", 4,
                    "e_conditioned_static_stability_anomaly_absolute",
                    _plot_peak_longitude_height(
                        composite,
                        "ray_lpo_conditioned_static_stability_anomaly",
                        title=(
                            "LPO-conditioned static stability minus "
                            "unconditional tropical background"
                        ),
                        units=r"$N^2$ anomaly (s$^{-2}$)",
                        differences_to_control=False,
                    ),
                )
            if 46 in figures:
                progress(
                    "Starting lpo_wave_propagation Figure 4f: MRG rays"
                )
                save(
                    "lpo_wave_propagation", 4,
                    "f_mrg_peak_temperature_rays",
                    _plot_peak_longitude_height(
                        composite,
                        "ray_filtered_temperature",
                        title="MRG-filtered temperature",
                        units="temperature (K)",
                        wave_name="MRG",
                    ),
                )
            if WAVE_PHASE_PROFILE_FIGURE in figures:
                progress(
                    "Starting lpo_wave_alignment Figure 6: cross-wave phase"
                )
                save(
                    "lpo_wave_alignment", 6,
                    "cross_wave_phase_alignment",
                    _plot_cross_wave_phase_alignment(composite),
                )
            if WAVE_COUPLING_LIFECYCLE_FIGURE in figures:
                progress(
                    "Starting lpo_wave_alignment Figure 7: coupling lifecycle"
                )
                save(
                    "lpo_wave_alignment", 7,
                    "wave_coupling_lifecycle",
                    _plot_wave_coupling_lifecycle(composite),
                )
            if WAVE_COUPLING_RESPONSE_FIGURE in figures:
                progress(
                    "Starting lpo_wave_alignment Figure 8: coupling responses"
                )
                save(
                    "lpo_wave_alignment", 8,
                    "wave_coupling_responses",
                    _plot_wave_coupling_response(composite),
                )


def build_validation_lifecycle_figure(
    validation_directory: Path,
    *,
    figure_output: Path,
) -> None:
    """Compare lifecycle sensitivity products owned here."""

    run_groups = (
        (
            "Precipitation-threshold sensitivity",
            ("threshold_9", "threshold_12", "threshold_15"),
            "area_integrated_24h_precip_km2_mm",
            RAIN_VOLUME_LABEL,
            RAIN_VOLUME_DISPLAY_SCALE,
        ),
        (
            "Precipitation-threshold sensitivity",
            ("threshold_9", "threshold_12", "threshold_15"),
            "total_high_cloud_area_km2",
            r"Total high-cloud area (km$^2$)",
            1.0,
        ),
        (
            "Record-subset sensitivity",
            ("threshold_12", "record_first_half", "record_second_half"),
            "area_integrated_24h_precip_km2_mm",
            RAIN_VOLUME_LABEL,
            RAIN_VOLUME_DISPLAY_SCALE,
        ),
        (
            "Temporal-mean sensitivity",
            ("running_mean_48", "running_mean_60", "threshold_12"),
            "area_integrated_24h_precip_km2_mm",
            RAIN_VOLUME_LABEL,
            RAIN_VOLUME_DISPLAY_SCALE,
        ),
        (
            "Gaussian-filter sensitivity",
            ("gaussian_sigma_3", "threshold_12", "gaussian_sigma_7"),
            "area_integrated_24h_precip_km2_mm",
            RAIN_VOLUME_LABEL,
            RAIN_VOLUME_DISPLAY_SCALE,
        ),
    )
    labels = {
        "threshold_9": "9 mm day$^{-1}$",
        "threshold_12": "primary",
        "threshold_15": "15 mm day$^{-1}$",
        "record_first_half": "first half",
        "record_second_half": "second half",
        "running_mean_48": "48 h",
        "running_mean_60": "60 h",
        "gaussian_sigma_3": "σ = 3°",
        "gaussian_sigma_7": "σ = 7°",
    }
    primary_labels = {
        "Precipitation-threshold sensitivity": "12 mm day$^{-1}$",
        "Record-subset sensitivity": "full record",
        "Temporal-mean sensitivity": "72 h",
        "Gaussian-filter sensitivity": "σ = 5°",
    }
    datasets: dict[str, xr.Dataset] = {}
    try:
        for run_name in {
            run_name
            for _title, runs, _variable, _units, _scale in run_groups
            for run_name in runs
        }:
            path = (
                validation_directory
                / run_name
                / LIFECYCLE_COMPOSITE_FILENAME
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing lifecycle validation composite: {path}"
                )
            datasets[run_name] = xr.open_dataset(path)

        fig, axes = plt.subplots(
            len(run_groups),
            len(SIMULATION_DIRS),
            figsize=(12, 12),
            constrained_layout=True,
            sharex=True,
        )
        for row, (title, runs, variable, units, scale) in enumerate(
            run_groups
        ):
            for column, simulation in enumerate(SIMULATION_DIRS):
                axis = axes[row, column]
                for run_name in runs:
                    dataset = datasets[run_name]
                    if variable not in set(dataset.variable.values.astype(str)):
                        raise RuntimeError(
                            f"{run_name}: lifecycle composite lacks "
                            f"{variable!r}"
                        )
                    if simulation not in set(
                        dataset.simulation.values.astype(str)
                    ):
                        continue
                    count = int(
                        dataset.event_count.sel(simulation=simulation).values
                    )
                    curve = dataset.event_equal_mean.sel(
                        simulation=simulation,
                        variable=variable,
                    )
                    axis.plot(
                        dataset.lag_hours,
                        scale * curve,
                        label=(
                            (
                                primary_labels[title]
                                if run_name == "threshold_12"
                                else labels[run_name]
                            )
                            + " "
                            f"(n={count})"
                        ),
                    )
                axis.axvline(0, color="0.55", lw=0.7)
                axis.grid(alpha=0.2)
                if row == 0:
                    axis.set_title(SIMULATION_LABELS[simulation])
                if column == 0:
                    axis.set_ylabel(f"{title}\n{units}")
                if row == len(run_groups) - 1:
                    axis.set_xlabel(LIFECYCLE_LAG_LABEL)
                if column == len(SIMULATION_DIRS) - 1:
                    axis.legend(frameon=False, fontsize=7)
        fig.suptitle(
            "LPO lifecycle sensitivity"
        )
        saver = FigureSaver(
            str(Path(figure_output) / "lpo_tracking"),
            enabled=True,
        )
        saver.save(
            fig,
            "lpo_lifecycle_validation_sensitivity",
            number=6,
        )
        plt.close(fig)
    finally:
        for dataset in datasets.values():
            dataset.close()


def _parse_lifecycle_figure(value: str) -> int | str:
    """Parse one numeric figure or the Figure 45a identifier."""

    if value.lower() == "45a":
        return "45a"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid lifecycle figure: {value!r}"
        ) from exc


def parse_command_line(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample cloud, environment and wave properties through saved LPO "
            "lifecycles and build independent-event composites."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("publication", "validation"),
        default="publication",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Use the isolated actual-data tracking subset and sample at most "
            f"{TEST_EVENTS_PER_SIMULATION} representative LPOs per simulation."
        ),
    )
    parser.add_argument("--test-start", default=None, metavar="TIME")
    parser.add_argument(
        "--test-days",
        type=int,
        default=DEFAULT_TEST_DAYS,
        metavar="N",
    )
    parser.add_argument(
        "--test-min-duration-days",
        type=float,
        default=DEFAULT_TEST_MINIMUM_DURATION_DAYS,
        metavar="DAYS",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        type=_parse_lifecycle_figure,
        choices=LIFECYCLE_FIGURES,
        default=None,
        help=(
            "Select legacy lifecycle figure identifiers. Output files use "
            "folder-local figure numbers."
        ),
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Build figures only from current lifecycle products. Fail when "
            "a required product is missing or stale."
        ),
    )
    cache_mode.add_argument(
        "--rebuild-cache",
        "--force-recompute-caches",
        dest="rebuild_cache",
        action="store_true",
        help=(
            "Ignore lifecycle table bundles, event checkpoints, event "
            "sections, and derived composites. Rebuild every product."
        ),
    )
    parser.add_argument(
        "--allow-stale-lpo-products",
        action="store_true",
        help=(
            "Use existing publication LPO tables with an older product "
            "version."
        ),
    )
    parser.add_argument(
        "--simulation",
        choices=tuple(SIMULATION_DIRS),
        default=None,
        help=(
            "Calculate one publication simulation and stop before shared "
            "composites and figures. Run separate simulations concurrently, "
            "then run this script normally to finalize."
        ),
    )
    args = parser.parse_args(argv)
    if args.test and args.mode != "publication":
        parser.error("--test uses isolated publication-shaped products")
    if args.test and args.test_days < MINIMUM_TEST_DAYS:
        parser.error(f"--test-days must be at least {MINIMUM_TEST_DAYS}")
    if args.allow_stale_lpo_products and (
        args.test or args.mode != "publication"
    ):
        parser.error("--allow-stale-lpo-products supports publication mode only")
    if args.simulation and (args.test or args.mode != "publication"):
        parser.error("--simulation supports publication mode only")
    if args.simulation and args.figures_only:
        parser.error("--simulation cannot be combined with --figures-only")
    if args.simulation and args.figures:
        parser.error("--simulation does not render figures")
    if args.figures:
        args.figures = list(dict.fromkeys(args.figures))
    return args


def _main(argv: Sequence[str] | None = None) -> int:
    global TEST_MODE
    args = parse_command_line(argv)
    TEST_MODE = bool(args.test)
    progress(
        "Starting LPO lifecycle analysis "
        f"({'actual-data test' if args.test else args.mode} mode)"
    )
    if args.test:
        mode_directory = require_actual_test_products(
            start=args.test_start,
            days=args.test_days,
            minimum_duration_days=args.test_min_duration_days,
        )
        primary_directory = mode_directory / "primary"
        if args.figures_only:
            require_lifecycle_products(primary_directory)
        elif (
            args.rebuild_cache
            or not reuse_test_lifecycle_products(primary_directory)
        ):
            analyse_run_lifecycles(
                primary_directory,
                test=True,
                force_recompute_caches=args.rebuild_cache,
            )
        figure_output = actual_test_output_directory(
            lpo_tracking_figure_root(),
            start=args.test_start,
            days=args.test_days,
            minimum_duration_hours=int(
                round(24.0 * args.test_min_duration_days)
            ),
        )
        progress(f"Actual-data LPO lifecycle test data: {primary_directory}")
        progress(f"Actual-data LPO lifecycle test figures: {figure_output}")
        figures = args.figures or list(LIFECYCLE_FIGURES)
        build_lifecycle_figures(
            mode_directory,
            figures=figures,
            figure_output=figure_output,
            test=True,
        )
    elif args.mode == "publication":
        mode_directory = require_publication_products(
            allow_stale_product_version=args.allow_stale_lpo_products
        )
        primary_directory = mode_directory / "primary"
        if args.figures_only:
            require_lifecycle_products(primary_directory)
        elif args.simulation:
            analyse_run_lifecycles(
                primary_directory,
                force_recompute_caches=args.rebuild_cache,
                simulations=(args.simulation,),
                finalize=False,
            )
            progress(
                f"Lifecycle products complete for {args.simulation}. "
                "Run without --simulation after every simulation completes."
            )
            return 0
        else:
            analyse_run_lifecycles(
                primary_directory,
                force_recompute_caches=args.rebuild_cache,
            )
        figures = args.figures or list(LIFECYCLE_FIGURES)
        build_lifecycle_figures(
            mode_directory,
            figures=figures,
            figure_output=lpo_tracking_figure_root(),
        )
    else:
        mode_directory = lpo_tracking_output_root() / "validation"
        if not mode_directory.is_dir():
            raise FileNotFoundError(
                f"Missing LPO validation products: {mode_directory}"
            )
        for run_name in VALIDATION_RUN_NAMES:
            run_directory = mode_directory / run_name
            if not run_directory.is_dir():
                raise FileNotFoundError(
                    f"Missing configured LPO validation run: {run_directory}"
                )
            if args.figures_only:
                require_lifecycle_products(run_directory)
            else:
                analyse_run_lifecycles(
                    run_directory,
                    force_recompute_caches=args.rebuild_cache,
                )
        build_validation_lifecycle_figure(
            mode_directory,
            figure_output=lpo_tracking_figure_root(),
        )
    progress("LPO lifecycle analysis complete")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run lifecycle calculations with bounded Dask task concurrency."""

    with dask.config.set(scheduler="synchronous"):
        return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
