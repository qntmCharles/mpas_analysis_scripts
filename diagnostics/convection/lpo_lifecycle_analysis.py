#!/usr/bin/env python3
"""Analyse cloud, environment and wave lifecycles within tracked LPOs.

``lpo_tracking.py`` is the sole producer of LPO identities, masks, tracks,
precipitation diagnostics and lifecycle reference times.  This script consumes
that stable interface and owns every downstream lifecycle calculation:

* instantaneous cloud populations and cloud depth inside each saved envelope;
* absolute and own-mean-anomaly environmental sampling;
* native-level vertical cloud-fraction profiles;
* Kelvin, equatorial-Rossby, TD-type and eastward-intraseasonal filtered
  fields sampled along the precipitation-weighted envelope centroid;
* event summaries, event-equal composites and lifecycle Figures 4--7.

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
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
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
    grid_cell_area_km2,
    native_layer_interfaces,
    native_midlevel_height,
    netcdf_safe_dataset,
    precip_rate_cmap_norm,
    precipitation_rate_from_dataset,
    pressure_to_pa,
    progress,
    vertical_dim,
)
from diagnostics.waves.cached_wave_products import load_existing_filtered_wave
from diagnostics.waves.equatorial_wave_config import (
    LPO_LIFECYCLE_FILTERS,
    WAVE_BOXES,
    WAVE_CONFIG_VERSION,
)
from diagnostics.waves.wave_metrics import (
    buoyancy_frequency_squared,
    sample_masked_field,
)


LIFECYCLE_ANALYSIS_VERSION = "lpo-lifecycle-analysis-v3-mask-grid-union"
LIFECYCLE_FIGURES = (4, 5, 6, 7)
VALIDATION_LIFECYCLE_FIGURE = 8
LIFECYCLE_STEP_FILENAME = "lpo_lifecycle_steps.csv"
LIFECYCLE_SUMMARY_FILENAME = "lpo_lifecycle_event_summary.csv"
LIFECYCLE_PROFILE_FILENAME = "lpo_lifecycle_vertical_profiles.csv"
LIFECYCLE_CONFIGURATION_FILENAME = "lpo_lifecycle_configuration.json"
LIFECYCLE_COMPOSITE_FILENAME = "lpo_lifecycle_event_equal_composites.nc"
TEST_EVENTS_PER_SIMULATION = 2

WAVE_SIMULATION_LABELS = {
    "control": "control",
    "nudged-u": "nudged u",
    "nudged-T": "nudged T",
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
    "area_integrated_24h_precip_km2_mm_day",
    "envelope_area_km2",
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
    "layer_mean_n2_14_18km",
    "layer_mean_n2_14_18km_event_anomaly",
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
    "layer_mean_n2_14_18km",
    "layer_mean_n2_14_18km_event_anomaly",
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
        loaded_dataset = xr.Dataset(loaded).load()
        return {
            name: loaded_dataset[name]
            for name in loaded
        }

    # Several diagnostics include lazy time/zonal-mean anomalies. Without
    # this event-level materialisation those means are recomputed separately
    # for every six-hour track step, making even a short smoke test needlessly
    # expensive.
    classification = load_event_horizontal_fields(classification)
    model_fields = load_event_horizontal_fields(model_fields)
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
        row = dict(envelope_rows[pd.Timestamp(time_value)])
        envelope_area = row["envelope_area_km2"]
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
            cloud_top_height, high_mask, area
        )["mean"]
        row["maximum_cloud_top_height"] = float(
            cloud_top_height.where(high_mask).max(skipna=True)
        )
        cloud_top_pressure = classification["cloud_top_pressure"].sel(
            Time=time_value
        )
        pressure_statistics = sample_masked_field(
            cloud_top_pressure,
            high_mask,
            area,
            quantiles=(0.1, 0.5),
        )
        row["median_cloud_top_pressure_hpa"] = pressure_statistics["q50"]
        row["p10_cloud_top_pressure_hpa"] = pressure_statistics["q10"]
        ttl_mask = high_mask & (cloud_top_pressure <= 180.0)
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
            ("layer_mean_n2_14_18km", "mean", "layer_mean_n2_14_18km"),
            ("layer_mean_n2_14_18km_event_anomaly", "mean", "layer_mean_n2_14_18km_event_anomaly"),
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
                conv_mask
                if variable_name == "w"
                else mask
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


def project_model_fields(dataset: xr.Dataset, precipitation: xr.DataArray) -> dict:
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
    fields["utls_temperature_anomaly"] = (
        utls - utls.mean(("Time", "longitude"), skipna=True)
    )
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
    n2 = xr.apply_ufunc(
        buoyancy_frequency_squared,
        theta,
        height,
        input_core_dims=[[zdim], [zdim]],
        output_core_dims=[[zdim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    in_layer = (height >= 14_000.0) & (height <= 18_000.0)
    valid_layer = in_layer & np.isfinite(n2)
    layer_n2 = (
        (n2 * layer_thickness).where(valid_layer).sum(zdim, skipna=True)
        / layer_thickness.where(valid_layer).sum(zdim, skipna=True)
    )
    layer_n2.attrs.update(
        units="s-2",
        definition=(
            "native-layer-thickness-weighted N2 over MPAS mid-levels "
            "between 14 and 18 km"
        ),
        vertical_coordinate="midpoints of zgrid interfaces",
    )
    add_environment("layer_mean_n2_14_18km", layer_n2)
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

    # Event anomalies retain each experiment's altered mean state by removing
    # only that simulation's own time-mean zonal mean at the same latitude.
    # Materialise these compact latitude-only baselines once per simulation;
    # otherwise their full reductions are rebuilt independently for every LPO.
    for name in environmental_names:
        field = fields[name]
        dimensions = [dim for dim in ("Time", "longitude") if dim in field.dims]
        if not dimensions:
            raise RuntimeError(
                f"Cannot form event anomaly for {name!r}: expected Time "
                "and/or longitude dimension"
            )
        anomaly_name = (
            "olr_event_anomaly"
            if name == "olr"
            else f"{name}_event_anomaly"
        )
        baseline = field.mean(
            dimensions, skipna=True
        ).reset_coords(drop=True).load()
        fields[anomaly_name] = field - baseline
    _require_mapping_fields(
        fields,
        (*REQUIRED_EVENT_MODEL_FIELDS, "cloud_fraction_profile"),
        product="projected MPAS lifecycle diagnostics",
    )
    return fields


def event_lifecycle_summary(
    lifecycle: pd.DataFrame,
    rainfall_decay_fraction: float = 0.25,
) -> pd.DataFrame:
    """Return event-level cloud timing, conversion and environment summaries."""

    required = {
        "simulation",
        "event_id",
        "lag_hours",
        "area_integrated_24h_precip_km2_mm_day",
        "active_deep_convective_area_km2",
        "stratiform_precipitating_high_cloud_area_km2",
        "nonprecipitating_high_cloud_area_km2",
        "total_high_cloud_area_km2",
        "ttl_reaching_high_cloud_fraction_of_envelope",
        "olr_w_m2",
        "median_cloud_top_pressure_hpa",
        "temperature_100hpa",
        "temperature_100hpa_event_anomaly",
        "temperature_100_200hpa",
        "temperature_100_200hpa_event_anomaly",
        "layer_mean_n2_14_18km",
        "layer_mean_n2_14_18km_event_anomaly",
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
    columns = {
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
        "deepest_median_cloud_top_pressure_lag_hours": (
            "median_cloud_top_pressure_hpa"
        ),
    }
    for event_id, event in lifecycle.groupby("event_id", sort=False):
        event = event.sort_values("lag_hours")
        row = {
            "simulation": event.simulation.iloc[0],
            "event_id": event_id,
            "rainfall_decay_fraction": rainfall_decay_fraction,
        }
        for output_name, column in columns.items():
            if not np.isfinite(event[column]).any():
                row[output_name] = np.nan
                continue
            index = (
                event[column].idxmin()
                if column in {"olr_w_m2", "median_cloud_top_pressure_hpa"}
                else event[column].idxmax()
            )
            row[output_name] = float(event.loc[index, "lag_hours"])
        rain = event.area_integrated_24h_precip_km2_mm_day
        peak = float(np.nanmax(rain)) if np.isfinite(rain).any() else np.nan
        after = event[event.lag_hours >= 0]
        below = after[
            after.area_integrated_24h_precip_km2_mm_day
            < rainfall_decay_fraction * peak
        ]
        persistence_column = "total_high_cloud_area_km2"
        if len(below) and np.isfinite(event[persistence_column]).any():
            decay_time = float(below.lag_hours.iloc[0])
            cloudy_after = after[
                (after.lag_hours >= decay_time)
                & (
                    after[persistence_column]
                    >= rainfall_decay_fraction
                    * np.nanmax(event[persistence_column])
                )
            ]
            row["high_cloud_persistence_after_rainfall_hours"] = (
                float(cloudy_after.lag_hours.max() - decay_time)
                if len(cloudy_after)
                else 0.0
            )
        else:
            row["high_cloud_persistence_after_rainfall_hours"] = np.nan
        active_peak = np.nanmax(event.active_deep_convective_area_km2)
        post_high = _finite_mean(
            after.loc[
                after.lag_hours > 0, persistence_column
            ].to_numpy(),
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
        peak_rain = np.nanmax(
            event.area_integrated_24h_precip_km2_mm_day
        )
        post_total = _finite_mean(
            event.loc[
                event.lag_hours > 0,
                "total_high_cloud_area_km2",
            ].to_numpy(),
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
            "layer_mean_n2_14_18km",
            "layer_mean_n2_14_18km_event_anomaly",
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
    """Build event-equal composites by resampling complete LPO sequences."""

    selected = lifecycle[
        lifecycle.complete_primary_lag_window.astype(bool)
        & lifecycle.in_primary_lifecycle_window.astype(bool)
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
        for simulation in ("nudged-u", "nudged-T")
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
            "sampling_unit": "complete LPO event",
            "bootstrap": f"{n_bootstrap} complete-event resamples",
            "event_weighting": "equal",
            "contrast_bootstrap": (
                "independent complete-event resampling within each simulation"
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
        "area_integrated_24h_precip_km2_mm_day",
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
    complete_ids = set(
        timing.loc[
            timing.complete_primary_lag_window.astype(bool), "event_id"
        ]
    )
    complete = candidates[candidates.event_id.isin(complete_ids)]
    if not complete.empty:
        candidates = complete
    if candidates.empty:
        raise RuntimeError(
            "The tracking test produced no LPO suitable for lifecycle "
            "sampling; use a later/longer --test window."
        )
    ranking = (
        timing[timing.event_id.isin(candidates.event_id)]
        .groupby("event_id")
        .area_integrated_24h_precip_km2_mm_day.max()
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
    height = native_midlevel_height(dataset, cloud)
    pressure_hpa = pressure_to_pa(dataset.pressure) / 100.0
    level_dimension = vertical_dim(cloud)
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
        profiles = xr.Dataset(
            {
                "cloud_fraction": (
                    (event_cloud * weights).sum(
                        ("latitude", "longitude"), skipna=True
                    )
                    / safe_denominator
                ),
                "height_m": (
                    (event_height * weights).sum(
                        ("latitude", "longitude"), skipna=True
                    )
                    / safe_denominator
                ),
                "pressure_hpa": (
                    (event_pressure * weights).sum(
                        ("latitude", "longitude"), skipna=True
                    )
                    / safe_denominator
                ),
            }
        ).load()
        frame = profiles.to_dataframe().reset_index()
        frame["simulation"] = simulation
        frame["event_id"] = event_id
        frame["vertical_level_index"] = frame[level_dimension].astype(int)
        frame = frame.drop(columns=[level_dimension])
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


def _write_lifecycle_metadata(
    standard: Path,
    *,
    lifecycle: pd.DataFrame,
    summary: pd.DataFrame,
    profiles: pd.DataFrame,
    test_event_ids: Sequence[str] | None,
    environment_anomaly_baseline: str,
) -> None:
    metadata = {
        "lifecycle_analysis_version": LIFECYCLE_ANALYSIS_VERSION,
        "tracking_product_version": PRODUCT_VERSION,
        "wave_config_version": WAVE_CONFIG_VERSION,
        "wave_filters": {
            wave: WAVE_BOXES[wave] for wave in LPO_LIFECYCLE_FILTERS
        },
        "wave_sampling_geometry": (
            "saved precipitation-weighted LPO centroid; wave filtering does "
            "not alter tracking or event selection"
        ),
        "local_sampling_geometry": "saved LPO envelope mask",
        "event_weighting": "one equal weight per complete LPO event",
        "absolute_environment": True,
        "own_mean_zonal_mean_event_anomalies": True,
        "environment_anomaly_baseline": environment_anomaly_baseline,
        "total_high_cloud_is_primary": True,
        "cloud_classes_are_attribution_terms": True,
        "lifecycle_step_columns": list(lifecycle.columns),
        "lifecycle_summary_columns": list(summary.columns),
        "vertical_profile_columns": list(profiles.columns),
        "test_event_ids": list(test_event_ids or ()),
        "test_product": test_event_ids is not None,
    }
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    metadata["configuration_sha256"] = hashlib.sha256(encoded).hexdigest()
    path = standard / LIFECYCLE_CONFIGURATION_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
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
    if test and not composite_input.complete_primary_lag_window.astype(bool).any():
        # Figure-only smoke tests may contain no complete ±48-hour event.
        # Preserve the actual lags and mark this explicitly in metadata rather
        # than pretending the result is a publication composite.
        composite_input["complete_primary_lag_window"] = True
        composite_input["in_primary_lifecycle_window"] = True
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
            "test_partial_sequences_allowed": bool(test),
        }
    )
    composite_path = run_directory / LIFECYCLE_COMPOSITE_FILENAME
    composite_tmp = composite_path.with_suffix(
        composite_path.suffix + ".tmp"
    )
    netcdf_safe_dataset(composite).to_netcdf(composite_tmp)
    os.replace(composite_tmp, composite_path)
    summary.to_csv(
        run_directory / LIFECYCLE_SUMMARY_FILENAME,
        index=False,
    )


def analyse_run_lifecycles(
    run_directory: Path,
    *,
    test: bool = False,
) -> None:
    """Build lifecycle-owned products from one tracking run directory."""

    combined_lifecycle: list[pd.DataFrame] = []
    combined_summary: list[pd.DataFrame] = []
    for simulation in SIMULATION_DIRS:
        standard = run_directory / simulation / "standard"
        timing_path = standard / "lpt_event_relative_diagnostics.csv"
        event_path = standard / "lpt_events.csv"
        if not timing_path.is_file() or not event_path.is_file():
            raise FileNotFoundError(
                f"Missing stable tracking products below {standard}"
            )
        tracking_timing = pd.read_csv(timing_path)
        tracking_timing["Time"] = pd.to_datetime(tracking_timing.Time)
        events = pd.read_csv(event_path)
        if test:
            event_ids = _select_test_events(events, tracking_timing)
        else:
            event_ids = events.event_id.tolist()
        selected_timing = tracking_timing[
            tracking_timing.event_id.isin(event_ids)
        ].copy()
        masks = _read_standard_masks(standard)
        progress(
            f"{run_directory.name}/{simulation}: sampling cloud and "
            f"environment lifecycles for {len(event_ids)} LPOs"
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
        dataset = dataset.isel(latitude=latitude_slice)
        progress(
            f"{run_directory.name}/{simulation}: using LPO latitude support "
            f"{float(dataset.latitude.min()):.2f} to "
            f"{float(dataset.latitude.max()):.2f} degrees"
        )
        full_precipitation = precipitation_rate_from_dataset(dataset)
        diagnostic_times = pd.DatetimeIndex(
            selected_timing.Time.drop_duplicates().sort_values()
        )
        precipitation = full_precipitation.sel(Time=diagnostic_times.values)
        classification = _load_classification(
            simulation,
            precipitation.Time,
            REQUIRED_CLASSIFICATION_FIELDS,
        )
        if test:
            projection_precipitation = precipitation
            environment_anomaly_baseline = (
                "selected actual-data LPO diagnostic times; test figure "
                "smoke-test product only"
            )
        else:
            projection_precipitation = full_precipitation
            environment_anomaly_baseline = "complete simulation record"
        progress(
            f"{run_directory.name}/{simulation}: projecting model fields and "
            "environment baselines"
        )
        model_fields = project_model_fields(
            dataset,
            projection_precipitation,
        )
        event_rows: list[pd.DataFrame] = []
        for event_number, event_id in enumerate(event_ids, start=1):
            progress(
                f"{run_directory.name}/{simulation}: lifecycle event "
                f"{event_number}/{len(event_ids)} {event_id}"
            )
            diagnosed = diagnose_event_steps(
                masks[event_id],
                precipitation,
                classification,
                model_fields,
                event_id=event_id,
                simulation=simulation,
            )
            event_rows.append(
                _merge_authoritative_timing(diagnosed, selected_timing)
            )
        lifecycle = pd.concat(event_rows, ignore_index=True)
        lifecycle = _add_wave_lifecycle_fields(lifecycle, test=test)
        summary = event_lifecycle_summary(lifecycle)
        summary = summary.merge(
            _wave_event_summary(lifecycle),
            on="event_id",
            how="left",
            validate="one_to_one",
        )
        profiles = _sample_native_cloud_profiles(
            dataset,
            masks,
            selected_timing,
            event_ids,
            simulation,
        )
        lifecycle.to_csv(standard / LIFECYCLE_STEP_FILENAME, index=False)
        summary.to_csv(standard / LIFECYCLE_SUMMARY_FILENAME, index=False)
        profiles.to_csv(standard / LIFECYCLE_PROFILE_FILENAME, index=False)
        _write_lifecycle_metadata(
            standard,
            lifecycle=lifecycle,
            summary=summary,
            profiles=profiles,
            test_event_ids=event_ids if test else None,
            environment_anomaly_baseline=environment_anomaly_baseline,
        )
        combined_lifecycle.append(lifecycle)
        combined_summary.append(summary)
        dataset.close()

    lifecycle = pd.concat(combined_lifecycle, ignore_index=True)
    summary = pd.concat(combined_summary, ignore_index=True)
    _write_run_composite(
        run_directory,
        lifecycle,
        summary,
        test=test,
    )


def require_lifecycle_products(run_directory: Path) -> None:
    missing: list[str] = []
    for simulation in SIMULATION_DIRS:
        standard = run_directory / simulation / "standard"
        for filename in (
            LIFECYCLE_STEP_FILENAME,
            LIFECYCLE_SUMMARY_FILENAME,
            LIFECYCLE_PROFILE_FILENAME,
            LIFECYCLE_CONFIGURATION_FILENAME,
        ):
            path = standard / filename
            if not path.is_file():
                missing.append(str(path))
        metadata_path = standard / LIFECYCLE_CONFIGURATION_FILENAME
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("lifecycle_analysis_version") != LIFECYCLE_ANALYSIS_VERSION:
                raise RuntimeError(
                    f"{metadata_path} predates {LIFECYCLE_ANALYSIS_VERSION}"
                )
            if set(metadata.get("wave_filters", {})) != set(
                LPO_LIFECYCLE_FILTERS
            ):
                raise RuntimeError(
                    f"{metadata_path} does not contain the configured "
                    "lifecycle wave filters"
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


def reuse_test_lifecycle_products(run_directory: Path) -> bool:
    """Rebuild test aggregates from current saved event products if possible."""

    try:
        require_lifecycle_products(run_directory)
    except (FileNotFoundError, RuntimeError):
        return False
    for simulation in SIMULATION_DIRS:
        standard = run_directory / simulation / "standard"
        metadata_path = standard / LIFECYCLE_CONFIGURATION_FILENAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("test_product", False):
            return False
        for tracking_filename in (
            "lpt_event_relative_diagnostics.csv",
            "lpt_events.csv",
        ):
            if (
                standard / tracking_filename
            ).stat().st_mtime > metadata_path.stat().st_mtime:
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
                mean,
                color=SIMULATION_COLOURS[simulation],
                label=f"{SIMULATION_LABELS[simulation]} (n={count})",
            )
            axes[row, 0].fill_between(
                lags, low, high, color=SIMULATION_COLOURS[simulation], alpha=0.12
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
                    contrast,
                    color=SIMULATION_COLOURS[simulation],
                    label=f"{SIMULATION_LABELS[simulation]} − control",
                )
                axes[row, 1].fill_between(
                    lags,
                    contrast_low,
                    contrast_high,
                    color=SIMULATION_COLOURS[simulation],
                    alpha=0.12,
                )
        axes[row, 0].set_ylabel(label)
        axes[row, 1].axhline(0, color="0.5", lw=0.7)
        for axis in axes[row]:
            axis.axvline(0, color="0.5", ls="--", lw=0.7)
            axis.grid(alpha=0.2)
    axes[0, 0].set_title("Absolute event-equal composite")
    axes[0, 1].set_title("Experiment minus control")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[-1, 0].set_xlabel("Lag from peak envelope precipitation (hours)")
    axes[-1, 1].set_xlabel("Lag from peak envelope precipitation (hours)")
    fig.suptitle(title)
    return fig


def _plot_lifecycle(lifecycle: pd.DataFrame):
    return _plot_composite_grid(
        lifecycle,
        (
            (
                "area_integrated_24h_precip_km2_mm_day",
                "Integrated 24-h precipitation",
            ),
            ("total_high_cloud_area_km2", "Total high-cloud area"),
            (
                "active_deep_convective_area_km2",
                "Active deep-convective area",
            ),
            (
                "stratiform_precipitating_high_cloud_area_km2",
                "Stratiform high-cloud area",
            ),
            (
                "nonprecipitating_high_cloud_area_km2",
                "Non-precipitating high-cloud area",
            ),
            ("olr_w_m2", r"OLR (W m$^{-2}$)"),
        ),
        "LPO precipitation and cloud-population lifecycle",
    )


def _plot_lifecycle_timing(summary: pd.DataFrame):
    metrics = (
        ("active_peak_lag_hours", "Active-core peak"),
        ("stratiform_peak_lag_hours", "Stratiform peak"),
        (
            "nonprecip_high_cloud_peak_lag_hours",
            "Non-precipitating high-cloud peak",
        ),
        ("total_high_cloud_peak_lag_hours", "Total high-cloud peak"),
        (
            "deepest_median_cloud_top_pressure_lag_hours",
            "Deepest median cloud top",
        ),
        (
            "ttl_reaching_fraction_peak_lag_hours",
            "TTL-reaching fraction peak",
        ),
        ("olr_minimum_lag_hours", "OLR minimum"),
        (
            "high_cloud_persistence_after_rainfall_hours",
            "Post-rain high-cloud persistence",
        ),
        (
            "post_peak_high_cloud_to_peak_active_area_ratio",
            "Post-peak high cloud / peak active area",
        ),
    )
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for axis, (column, title) in zip(axes.flat, metrics):
        for index, simulation in enumerate(SIMULATION_DIRS):
            values = summary.loc[
                summary.simulation == simulation, column
            ].dropna()
            axis.scatter(
                np.full(len(values), index),
                values,
                s=14,
                alpha=0.55,
                color=SIMULATION_COLOURS[simulation],
            )
        axis.axhline(0, color="0.5", lw=0.7)
        axis.set(
            xticks=range(3),
            xticklabels=[SIMULATION_LABELS[name] for name in SIMULATION_DIRS],
            title=title,
        )
    fig.suptitle("Lifecycle timing and conversion; one point per LPO event")
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
    """Average complete events after aligning their relative spatial grids."""

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
    stages = (-48, 0, 48)
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
            & lifecycle.complete_primary_lag_window.astype(bool)
            & lifecycle.primary_tropical_sample.astype(bool)
            & lifecycle.reference_in_record_subset.astype(bool)
        ]
        for column, lag in enumerate(stages):
            precipitation_collect: list[xr.DataArray] = []
            regimes_collect = {1: [], 2: [], 3: []}
            for event_id in sim.event_id.unique():
                event = sim[sim.event_id == event_id]
                sample = event[event.lag_hours == lag]
                if sample.empty:
                    continue
                path = _event_product_path(primary, simulation, event_id)
                with xr.open_dataset(path) as product:
                    if "cloud_regime_context" not in product:
                        continue
                    time_value = pd.Timestamp(sample.Time.iloc[0])
                    precip = product.raw_precipitation_mm_day.sel(Time=time_value)
                    regime = product.cloud_regime_context.sel(Time=time_value)
                    lon = float(sample.precip_weighted_centroid_lon.iloc[0])
                    lat = float(sample.precip_weighted_centroid_lat.iloc[0])
                    precipitation_collect.append(
                        _centred_event_field(precip, lon, lat)
                    )
                    for code in regimes_collect:
                        regimes_collect[code].append(
                            _centred_event_field(
                                (regime == code).astype(float),
                                lon,
                                lat,
                            )
                        )
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
            if row == 0:
                axis.set_title(f"{lag:+d} h")
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
            label="Event-equal precipitation (mm day$^{-1}$)",
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
    )
    fig.suptitle("LPO-centred precipitation and cloud-regime evolution")
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
                "layer_mean_n2_14_18km",
                r"14–18-km $N^2$ (s$^{-2}$)",
            ),
            (
                "layer_mean_n2_14_18km_event_anomaly",
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
        "LPO lifecycle UTLS environment",
    )


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


def build_lifecycle_figures(
    mode_directory: Path,
    *,
    figures: Sequence[int],
    figure_output: Path,
    test: bool = False,
) -> None:
    primary = mode_directory / "primary"
    lifecycle, summary = read_lifecycle_products(primary)
    if (
        test
        and not lifecycle.empty
        and not lifecycle.complete_primary_lag_window.astype(bool).any()
    ):
        # A deliberately short actual-data test may not span the complete
        # publication lag window.  Keep the saved flags authoritative, but
        # let the selected event fragments exercise every figure builder.
        lifecycle = lifecycle.copy()
        lifecycle["complete_primary_lag_window"] = True
        lifecycle["in_primary_lifecycle_window"] = True
    saver = FigureSaver(
        str(figure_output),
        enabled=True,
        figures_to_run=set(figures),
    )

    def save(number: int, stem: str, figure) -> None:
        saver.save(figure, stem, number=number)
        plt.close(figure)

    if 4 in figures:
        progress("Starting Figure 4: LPO cloud-population lifecycle")
        save(4, "lpo_cloud_population_lifecycle", _plot_lifecycle(lifecycle))
    if 5 in figures:
        progress("Starting Figure 5: LPO lifecycle timing and conversion")
        save(5, "lpo_lifecycle_timing", _plot_lifecycle_timing(summary))
    if 6 in figures:
        progress("Starting Figure 6: LPO-centred spatial evolution")
        save(
            6,
            "lpo_centred_spatial_evolution",
            _plot_event_centred_evolution(primary, lifecycle),
        )
    if 7 in figures:
        progress("Starting Figure 7: LPO UTLS environment")
        save(7, "lpo_utls_environment", _plot_environment(lifecycle))
        for wave_name in LPO_LIFECYCLE_FILTERS:
            progress(f"Starting Figure 7: {wave_name} LPO lifecycle")
            save(
                7,
                f"lpo_{_wave_key(wave_name)}_lifecycle",
                _plot_wave_lifecycle(lifecycle, wave_name),
            )


def build_validation_lifecycle_figure(
    validation_directory: Path,
    *,
    figure_output: Path,
) -> None:
    """Compare event-equal lifecycle sensitivity products owned here."""

    run_groups = (
        (
            "Precipitation-threshold sensitivity",
            ("threshold_9", "threshold_12", "threshold_15"),
            "area_integrated_24h_precip_km2_mm_day",
            r"Envelope-integrated precipitation (km$^2$ mm day$^{-1}$)",
        ),
        (
            "Precipitation-threshold sensitivity",
            ("threshold_9", "threshold_12", "threshold_15"),
            "total_high_cloud_area_km2",
            r"Total high-cloud area (km$^2$)",
        ),
        (
            "Record-subset sensitivity",
            ("threshold_12", "record_first_half", "record_second_half"),
            "area_integrated_24h_precip_km2_mm_day",
            r"Envelope-integrated precipitation (km$^2$ mm day$^{-1}$)",
        ),
    )
    labels = {
        "threshold_9": "9 mm day$^{-1}$",
        "threshold_12": "12 mm day$^{-1}$ / full record",
        "threshold_15": "15 mm day$^{-1}$",
        "record_first_half": "first half",
        "record_second_half": "second half",
    }
    datasets: dict[str, xr.Dataset] = {}
    try:
        for run_name in {
            run_name
            for _title, runs, _variable, _units in run_groups
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
            figsize=(14, 10),
            constrained_layout=True,
            sharex=True,
        )
        for row, (title, runs, variable, units) in enumerate(run_groups):
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
                        curve,
                        label=f"{labels[run_name]} (n={count})",
                    )
                axis.axvline(0, color="0.55", lw=0.7)
                axis.grid(alpha=0.2)
                if row == 0:
                    axis.set_title(SIMULATION_LABELS[simulation])
                if column == 0:
                    axis.set_ylabel(f"{title}\n{units}")
                if row == len(run_groups) - 1:
                    axis.set_xlabel("Lag from peak precipitation (h)")
                if column == len(SIMULATION_DIRS) - 1:
                    axis.legend(frameon=False, fontsize=7)
        fig.suptitle(
            "LPO lifecycle sensitivity; complete events retain equal weight"
        )
        saver = FigureSaver(
            str(figure_output),
            enabled=True,
            figures_to_run={VALIDATION_LIFECYCLE_FIGURE},
        )
        saver.save(
            fig,
            "lpo_lifecycle_validation_sensitivity",
            number=VALIDATION_LIFECYCLE_FIGURE,
        )
        plt.close(fig)
    finally:
        for dataset in datasets.values():
            dataset.close()


def parse_command_line(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample cloud, environment and wave properties through saved LPO "
            "lifecycles and build event-equal composites."
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
        type=int,
        choices=LIFECYCLE_FIGURES,
        default=None,
    )
    args = parser.parse_args(argv)
    if args.test and args.mode != "publication":
        parser.error("--test uses isolated publication-shaped products")
    if args.test and args.test_days < MINIMUM_TEST_DAYS:
        parser.error(f"--test-days must be at least {MINIMUM_TEST_DAYS}")
    if args.figures:
        args.figures = list(dict.fromkeys(args.figures))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_command_line(argv)
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
        if not reuse_test_lifecycle_products(primary_directory):
            analyse_run_lifecycles(primary_directory, test=True)
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
        mode_directory = require_publication_products()
        analyse_run_lifecycles(mode_directory / "primary")
        figures = args.figures or list(LIFECYCLE_FIGURES)
        build_lifecycle_figures(
            mode_directory,
            figures=figures,
            figure_output=lpo_tracking_figure_root() / "publication",
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
            analyse_run_lifecycles(run_directory)
        build_validation_lifecycle_figure(
            mode_directory,
            figure_output=lpo_tracking_figure_root() / "validation",
        )
    progress("LPO lifecycle analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
