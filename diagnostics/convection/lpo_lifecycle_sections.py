"""Import-safe event-section and composite helpers for LPO lifecycles.

The tracking diagnostic remains the sole owner of event identities, masks and
reference times.  This module only samples those saved events and reduces
complete event sequences with one equal weight per event.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from diagnostics.convection.lpo_tracking import (
    LIFECYCLE_LAGS_HOURS,
    MPAS_LPT_GRID_SPACING_DEGREES,
    _event_product_path,
)
from diagnostics.diagnostic_helpers import (
    native_midlevel_height,
    vertical_dim,
)
from diagnostics.waves.equatorial_wave_config import LPO_LIFECYCLE_FILTERS
from diagnostics.waves.wave_metrics import (
    buoyancy_frequency_squared,
    kelvin_group_velocities,
)


SECTION_CACHE_VERSION = "lpo-lifecycle-sections-v2-corrected-t-nudge-omega"
COMPOSITE_CACHE_VERSION = (
    "lpo-lifecycle-2d-composites-v2-corrected-t-nudge-omega"
)

DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES = 5.0
LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES = (0.0, 2.5, 3.75, 5.0)
RELATIVE_LONGITUDE_DEGREES = np.arange(-7.5, 7.51, 0.5)
RELATIVE_LATITUDE_DEGREES = np.arange(-10.0, 10.01, 0.5)
CLOUD_RELATIVE_LONGITUDE_DEGREES = np.arange(-20.0, 20.01, 0.5)
CLOUD_RELATIVE_LATITUDE_DEGREES = np.arange(-15.0, 15.01, 0.5)
PRESSURE_HPA = np.asarray(
    [
        50, 70, 100, 150, 200, 250, 300, 400,
        500, 600, 700, 800, 850, 900, 950, 1000,
    ],
    dtype=float,
)
HEIGHT_M = np.arange(0.0, 25_000.1, 250.0)
TROPICAL_LATITUDE_BOUNDS = (-10.0, 10.0)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260731

CLOUD_CLASS_CODES = (1, 2, 3)
CLOUD_CLASS_NAMES = (
    "active_deep_convection",
    "stratiform_precipitating_high_cloud",
    "nonprecipitating_high_cloud",
)
SURFACE_VARIABLES = (
    "unfiltered_precipitation",
    "unfiltered_precipitable_water",
    "unfiltered_olr",
)
VERTICAL_VARIABLES = (
    "unfiltered_temperature",
    "unfiltered_upward_pressure_velocity",
    "unfiltered_specific_humidity",
    "unfiltered_relative_humidity",
)
FILTERED_SURFACE_VARIABLES = (
    "filtered_precipitation",
    "filtered_precipitable_water",
    "filtered_olr",
)
FILTERED_VERTICAL_VARIABLES = (
    "filtered_temperature",
    "filtered_upward_pressure_velocity",
    "filtered_specific_humidity",
    "filtered_relative_humidity",
)


def _ascending_slice(coordinate: xr.DataArray, lower: float, upper: float):
    """Return a slice that respects an ascending or descending coordinate."""

    first = float(coordinate.values[0])
    last = float(coordinate.values[-1])
    return slice(lower, upper) if first < last else slice(upper, lower)


def _exact_time_indices(source_time, requested_time) -> np.ndarray:
    source = pd.DatetimeIndex(pd.to_datetime(source_time))
    requested = pd.DatetimeIndex(pd.to_datetime(requested_time))
    indices = source.get_indexer(requested)
    if np.any(indices < 0):
        missing = requested[indices < 0]
        raise RuntimeError(
            "LPO section source does not contain exact saved event times: "
            + ", ".join(str(value) for value in missing[:4])
        )
    return indices


def _nearest_indices(source, targets, *, periodic=False):
    source = np.asarray(source, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if periodic:
        distance = np.abs(
            ((source[None, :] - targets[..., None] + 180.0) % 360.0)
            - 180.0
        )
    else:
        distance = np.abs(source[None, :] - targets[..., None])
    indices = distance.argmin(axis=-1)
    mismatch = np.take_along_axis(
        distance, indices[..., None], axis=-1
    )[..., 0]
    return indices, mismatch


def sample_event_relative_surface(
    field: xr.DataArray,
    steps: pd.DataFrame,
    *,
    relative_latitude=RELATIVE_LATITUDE_DEGREES,
    relative_longitude=RELATIVE_LONGITUDE_DEGREES,
) -> xr.DataArray:
    """Sample one Time/latitude/longitude field on an LPO-relative grid."""

    required = {"Time", "latitude", "longitude"}
    if not required.issubset(field.dims):
        raise ValueError(
            f"Event-relative surface sampling requires {required}; got "
            f"{field.dims}"
        )
    field = field.transpose("Time", "latitude", "longitude")
    time_indices = _exact_time_indices(field.Time.values, steps.Time)
    target_latitude = (
        np.asarray(steps.precip_weighted_centroid_lat, dtype=float)[:, None]
        + np.asarray(relative_latitude, dtype=float)[None, :]
    )
    target_longitude = (
        np.asarray(steps.precip_weighted_centroid_lon, dtype=float)[:, None]
        + np.asarray(relative_longitude, dtype=float)[None, :]
    )
    latitude_indices, latitude_mismatch = _nearest_indices(
        field.latitude.values, target_latitude
    )
    longitude_indices, longitude_mismatch = _nearest_indices(
        field.longitude.values, target_longitude, periodic=True
    )
    tolerance = 0.51 * MPAS_LPT_GRID_SPACING_DEGREES
    if np.nanmax(longitude_mismatch) > tolerance:
        raise RuntimeError(
            "Model longitude grid is incompatible with the fixed LPO grid"
        )
    sampled = field.isel(
        Time=xr.DataArray(time_indices, dims="event_step"),
        latitude=xr.DataArray(
            latitude_indices,
            dims=("event_step", "relative_latitude"),
        ),
        longitude=xr.DataArray(
            longitude_indices,
            dims=("event_step", "relative_longitude"),
        ),
    ).transpose("event_step", "relative_latitude", "relative_longitude")
    valid_latitude = xr.DataArray(
        latitude_mismatch <= tolerance,
        dims=("event_step", "relative_latitude"),
    )
    sampled = sampled.where(valid_latitude).reset_coords(drop=True).load()
    return sampled.assign_coords(
        event_step=np.arange(len(steps)),
        relative_latitude=np.asarray(relative_latitude, dtype=float),
        relative_longitude=np.asarray(relative_longitude, dtype=float),
    )


def sample_tropical_pressure_sections(
    field: xr.DataArray,
    steps: pd.DataFrame,
) -> xr.DataArray:
    """Sample 10S--10N mean pressure profiles on the longitude stencil."""

    required = {"Time", "pres", "latitude", "longitude"}
    if not required.issubset(field.dims):
        raise ValueError(
            f"Pressure-section field requires {required}; got {field.dims}"
        )
    tropical = field.sel(
        latitude=_ascending_slice(
            field.latitude, *TROPICAL_LATITUDE_BOUNDS
        ),
        pres=PRESSURE_HPA,
    ).mean("latitude", skipna=True).transpose("Time", "pres", "longitude")
    time_indices = _exact_time_indices(tropical.Time.values, steps.Time)
    target_longitude = (
        np.asarray(steps.precip_weighted_centroid_lon, dtype=float)[:, None]
        + RELATIVE_LONGITUDE_DEGREES[None, :]
    )
    longitude_indices, mismatch = _nearest_indices(
        tropical.longitude.values, target_longitude, periodic=True
    )
    if np.nanmax(mismatch) > 0.51 * MPAS_LPT_GRID_SPACING_DEGREES:
        raise RuntimeError(
            "Model longitude grid is incompatible with the fixed LPO grid"
        )
    sampled = tropical.isel(
        Time=xr.DataArray(time_indices, dims="event_step"),
        longitude=xr.DataArray(
            longitude_indices,
            dims=("event_step", "relative_longitude"),
        ),
    ).transpose("event_step", "pres", "relative_longitude")
    return sampled.reset_coords(drop=True).load().rename(
        {"pres": "pressure_hpa"}
    ).assign_coords(
        event_step=np.arange(len(steps)),
        pressure_hpa=PRESSURE_HPA,
        relative_longitude=RELATIVE_LONGITUDE_DEGREES,
    )


def _sample_tropical_native_profiles(
    field: xr.DataArray,
    steps: pd.DataFrame,
) -> xr.DataArray:
    """Sample native profiles after the same 10S--10N latitude mean."""

    zdim = vertical_dim(field)
    required = {"latitude", "longitude", zdim}
    if not required.issubset(field.dims):
        raise ValueError(
            f"Native-section field requires {required}; got {field.dims}"
        )
    tropical = field.sel(
        latitude=_ascending_slice(
            field.latitude, *TROPICAL_LATITUDE_BOUNDS
        )
    ).mean("latitude", skipna=True)
    if "Time" in tropical.dims:
        time_indices = _exact_time_indices(tropical.Time.values, steps.Time)
        tropical = tropical.isel(
            Time=xr.DataArray(time_indices, dims="event_step")
        )
    else:
        tropical = tropical.expand_dims(event_step=np.arange(len(steps)))
    target_longitude = (
        np.asarray(steps.precip_weighted_centroid_lon, dtype=float)[:, None]
        + RELATIVE_LONGITUDE_DEGREES[None, :]
    )
    longitude_indices, mismatch = _nearest_indices(
        tropical.longitude.values, target_longitude, periodic=True
    )
    if np.nanmax(mismatch) > 0.51 * MPAS_LPT_GRID_SPACING_DEGREES:
        raise RuntimeError(
            "Model longitude grid is incompatible with the fixed LPO grid"
        )
    sampled = tropical.isel(
        longitude=xr.DataArray(
            longitude_indices,
            dims=("event_step", "relative_longitude"),
        )
    ).transpose("event_step", "relative_longitude", zdim)
    return sampled.reset_coords(drop=True).load().assign_coords(
        event_step=np.arange(len(steps)),
        relative_longitude=RELATIVE_LONGITUDE_DEGREES,
    )


def native_background_sections(
    dataset: xr.Dataset,
    steps: pd.DataFrame,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return event-local U and N on the fixed height coordinate."""

    required = {"theta", "uReconstructZonal", "zgrid"}
    missing = sorted(required - set(dataset.data_vars))
    if missing:
        raise RuntimeError(
            f"Kelvin-reference lifecycle sections require {missing}"
        )
    theta = dataset.theta
    height = native_midlevel_height(dataset, theta)
    wind = dataset.uReconstructZonal
    sampled_theta = _sample_tropical_native_profiles(theta, steps)
    sampled_height = _sample_tropical_native_profiles(height, steps)
    sampled_wind = _sample_tropical_native_profiles(wind, steps)
    shape = (
        len(steps),
        len(RELATIVE_LONGITUDE_DEGREES),
        len(HEIGHT_M),
    )
    output_u = np.full(shape, np.nan, dtype=np.float32)
    output_n = np.full(shape, np.nan, dtype=np.float32)
    for event_index in range(len(steps)):
        for longitude_index in range(len(RELATIVE_LONGITUDE_DEGREES)):
            theta_profile = np.asarray(
                sampled_theta[event_index, longitude_index], dtype=float
            )
            height_profile = np.asarray(
                sampled_height[event_index, longitude_index], dtype=float
            )
            wind_profile = np.asarray(
                sampled_wind[event_index, longitude_index], dtype=float
            )
            finite = (
                np.isfinite(theta_profile)
                & np.isfinite(height_profile)
                & np.isfinite(wind_profile)
            )
            if finite.sum() < 3:
                continue
            order = np.argsort(height_profile[finite])
            z = height_profile[finite][order]
            th = theta_profile[finite][order]
            u = wind_profile[finite][order]
            unique = np.r_[True, np.diff(z) > 0.0]
            z, th, u = z[unique], th[unique], u[unique]
            if len(z) < 3:
                continue
            n2 = buoyancy_frequency_squared(th, z)
            n = np.sqrt(np.maximum(n2, 0.0))
            inside = (HEIGHT_M >= z[0]) & (HEIGHT_M <= z[-1])
            output_u[event_index, longitude_index, inside] = np.interp(
                HEIGHT_M[inside], z, u
            )
            output_n[event_index, longitude_index, inside] = np.interp(
                HEIGHT_M[inside], z, n
            )
    coords = {
        "event_step": np.arange(len(steps)),
        "relative_longitude": RELATIVE_LONGITUDE_DEGREES,
        "height_m": HEIGHT_M,
    }
    dimensions = ("event_step", "relative_longitude", "height_m")
    return (
        xr.DataArray(output_u, dims=dimensions, coords=coords),
        xr.DataArray(output_n, dims=dimensions, coords=coords),
    )


def _centred_context_field(
    field: xr.DataArray,
    centroid_lon: float,
    centroid_lat: float,
) -> xr.DataArray:
    """Put a saved event-product field on its native relative grid."""

    field = field.transpose("latitude", "longitude")
    latitude = np.asarray(field.latitude, dtype=float)
    longitude = np.asarray(field.longitude, dtype=float)
    values = np.asarray(field, dtype=float)
    longitude_index = int(
        np.argmin(
            np.abs(((longitude - centroid_lon + 180.0) % 360.0) - 180.0)
        )
    )
    latitude_index = int(np.argmin(np.abs(latitude - centroid_lat)))
    centre_index = values.shape[1] // 2
    values = np.roll(values, centre_index - longitude_index, axis=1)
    return xr.DataArray(
        values,
        dims=("cloud_relative_latitude", "cloud_relative_longitude"),
        coords={
            "cloud_relative_latitude": (
                np.arange(values.shape[0]) - latitude_index
            ) * MPAS_LPT_GRID_SPACING_DEGREES,
            "cloud_relative_longitude": (
                np.arange(values.shape[1]) - centre_index
            ) * MPAS_LPT_GRID_SPACING_DEGREES,
        },
    )


def cloud_context_sections(
    primary: Path,
    simulation: str,
    steps: pd.DataFrame,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Sample broad cloud context; the LPO envelope is only an overlay."""

    shape = (
        len(steps),
        len(CLOUD_CLASS_CODES),
        len(CLOUD_RELATIVE_LATITUDE_DEGREES),
        len(CLOUD_RELATIVE_LONGITUDE_DEGREES),
    )
    class_indicator = np.full(shape, np.nan, dtype=np.float32)
    envelope_indicator = np.full(
        (len(steps), shape[2], shape[3]), np.nan, dtype=np.float32
    )
    coverage = np.zeros_like(envelope_indicator, dtype=np.int8)
    step_lookup = {
        str(event_id): group
        for event_id, group in steps.groupby("event_id", sort=False)
    }
    for event_id, event_steps in step_lookup.items():
        path = _event_product_path(primary, simulation, event_id)
        with xr.open_dataset(path) as product:
            required = {"cloud_regime_context", "lpt_envelope_mask"}
            missing = sorted(required - set(product.data_vars))
            if missing:
                raise RuntimeError(
                    f"Broad cloud composite requires {missing} in {path}"
                )
            for row in event_steps.itertuples():
                step_index = int(row.event_step)
                regime = _centred_context_field(
                    product.cloud_regime_context.sel(Time=row.Time),
                    float(row.precip_weighted_centroid_lon),
                    float(row.precip_weighted_centroid_lat),
                ).reindex(
                    cloud_relative_latitude=CLOUD_RELATIVE_LATITUDE_DEGREES,
                    cloud_relative_longitude=CLOUD_RELATIVE_LONGITUDE_DEGREES,
                )
                envelope = _centred_context_field(
                    product.lpt_envelope_mask.sel(Time=row.Time),
                    float(row.precip_weighted_centroid_lon),
                    float(row.precip_weighted_centroid_lat),
                ).reindex(
                    cloud_relative_latitude=CLOUD_RELATIVE_LATITUDE_DEGREES,
                    cloud_relative_longitude=CLOUD_RELATIVE_LONGITUDE_DEGREES,
                )
                valid = np.isfinite(regime)
                coverage[step_index] = valid.astype(np.int8)
                envelope_indicator[step_index] = np.where(
                    valid, np.asarray(envelope, dtype=float), np.nan
                )
                for class_index, code in enumerate(CLOUD_CLASS_CODES):
                    class_indicator[step_index, class_index] = np.where(
                        valid, np.asarray(regime == code, dtype=float), np.nan
                    )
    coords = {
        "event_step": np.arange(len(steps)),
        "cloud_class": np.asarray(CLOUD_CLASS_NAMES, dtype=str),
        "cloud_relative_latitude": CLOUD_RELATIVE_LATITUDE_DEGREES,
        "cloud_relative_longitude": CLOUD_RELATIVE_LONGITUDE_DEGREES,
    }
    return (
        xr.DataArray(
            class_indicator,
            dims=(
                "event_step", "cloud_class", "cloud_relative_latitude",
                "cloud_relative_longitude",
            ),
            coords=coords,
        ),
        xr.DataArray(
            envelope_indicator,
            dims=(
                "event_step", "cloud_relative_latitude",
                "cloud_relative_longitude",
            ),
            coords={key: value for key, value in coords.items() if key != "cloud_class"},
        ),
        xr.DataArray(
            coverage,
            dims=(
                "event_step", "cloud_relative_latitude",
                "cloud_relative_longitude",
            ),
            coords={key: value for key, value in coords.items() if key != "cloud_class"},
        ),
    )


def _own_climatology_anomaly(
    field: xr.DataArray,
    *,
    baseline_times: Sequence[np.datetime64] | None = None,
) -> xr.DataArray:
    dimensions = [
        dimension
        for dimension in ("Time", "longitude")
        if dimension in field.dims
    ]
    if not dimensions:
        raise ValueError("Own-climatology anomaly requires Time/longitude")
    baseline_source = field
    if baseline_times is not None and "Time" in field.dims:
        baseline_source = field.sel(Time=np.asarray(baseline_times))
    baseline = baseline_source.mean(dimensions, skipna=True).load()
    return field - baseline


def select_section_steps(steps: pd.DataFrame, *, test=False) -> pd.DataFrame:
    """Retain authoritative LPO samples used by the 2-D composite cache."""

    selected = steps.copy()
    for flag in (
        "primary_tropical_sample",
        "reference_in_record_subset",
        "in_primary_lifecycle_window",
    ):
        if flag in selected:
            selected = selected[selected[flag].astype(bool)]
    if not test and "complete_primary_lag_window" in selected:
        selected = selected[
            selected.complete_primary_lag_window.astype(bool)
        ]
    selected = selected.sort_values(
        ["event_id", "lag_hours", "Time"]
    ).drop_duplicates(["event_id", "lag_hours"], keep="last")
    selected = selected.reset_index(drop=True)
    selected["event_step"] = np.arange(len(selected))
    if selected.empty:
        raise RuntimeError("No authoritative LPO steps remain for 2-D sections")
    return selected


def build_unfiltered_event_sections(
    dataset: xr.Dataset,
    precipitation: xr.DataArray,
    steps: pd.DataFrame,
    *,
    primary: Path,
    simulation: str,
    test: bool,
) -> xr.Dataset:
    """Build the slow, event-level unfiltered section cache once."""

    steps = select_section_steps(steps, test=test)
    baseline_times = (
        np.unique(pd.to_datetime(steps.Time).values)
        if test
        else None
    )
    required = {
        "precipw", "olrtoa", "t_isobaric", "omg_isobaric",
        "qv_isobaric", "rh_isobaric", "theta", "uReconstructZonal",
        "zgrid",
    }
    missing = sorted(required - set(dataset.data_vars))
    if missing:
        raise RuntimeError(
            f"Unfiltered LPO sections require preferred MPAS fields {missing}"
        )
    variables: dict[str, xr.DataArray | tuple] = {
        "event_id": (("event_step",), steps.event_id.astype(str)),
        "Time": (("event_step",), pd.to_datetime(steps.Time)),
        "lag_hours": (("event_step",), steps.lag_hours.astype(int)),
        "precip_weighted_centroid_lon": (
            ("event_step",),
            steps.precip_weighted_centroid_lon.astype(float),
        ),
        "precip_weighted_centroid_lat": (
            ("event_step",),
            steps.precip_weighted_centroid_lat.astype(float),
        ),
    }
    for flag in (
        "complete_primary_lag_window",
        "primary_tropical_sample",
        "reference_in_record_subset",
        "in_primary_lifecycle_window",
    ):
        if flag in steps:
            variables[flag] = (
                ("event_step",), steps[flag].astype(bool).values
            )
    surface_sources = {
        "unfiltered_precipitation": precipitation,
        "unfiltered_precipitable_water": dataset.precipw,
        "unfiltered_olr": dataset.olrtoa,
    }
    for name, source in surface_sources.items():
        variables[name] = sample_event_relative_surface(source, steps)
        variables[f"{name}_own_climatology_anomaly"] = (
            sample_event_relative_surface(
                _own_climatology_anomaly(
                    source, baseline_times=baseline_times
                ),
                steps,
            )
        )
    pressure_sources = {
        "unfiltered_temperature": dataset.t_isobaric,
        "unfiltered_upward_pressure_velocity": -dataset.omg_isobaric,
        "unfiltered_specific_humidity": 1000.0 * dataset.qv_isobaric,
        "unfiltered_relative_humidity": dataset.rh_isobaric,
    }
    for name, source in pressure_sources.items():
        variables[name] = sample_tropical_pressure_sections(source, steps)
        variables[f"{name}_own_climatology_anomaly"] = (
            sample_tropical_pressure_sections(
                _own_climatology_anomaly(
                    source, baseline_times=baseline_times
                ),
                steps,
            )
        )
    background_u, background_n = native_background_sections(dataset, steps)
    variables["background_zonal_wind"] = background_u
    variables["buoyancy_frequency"] = background_n
    cloud_classes, envelope, coverage = cloud_context_sections(
        primary, simulation, steps
    )
    variables["cloud_class_indicator"] = cloud_classes
    variables["lpo_envelope_indicator"] = envelope
    variables["cloud_context_coverage"] = coverage
    output = xr.Dataset(
        variables,
        coords={
            "event_step": np.arange(len(steps)),
            "relative_latitude": RELATIVE_LATITUDE_DEGREES,
            "relative_longitude": RELATIVE_LONGITUDE_DEGREES,
            "pressure_hpa": PRESSURE_HPA,
            "height_m": HEIGHT_M,
            "cloud_class": np.asarray(CLOUD_CLASS_NAMES, dtype=str),
            "cloud_relative_latitude": CLOUD_RELATIVE_LATITUDE_DEGREES,
            "cloud_relative_longitude": CLOUD_RELATIVE_LONGITUDE_DEGREES,
        },
        attrs={
            "cache_version": SECTION_CACHE_VERSION,
            "source_simulation": simulation,
            "test_subset": int(test),
            "event_weighting": "one equal weight per complete LPO event",
            "default_longitude_half_width_degrees": (
                DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
            ),
            "longitude_half_width_sensitivity_degrees": json.dumps(
                LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES
            ),
            "vertical_latitude_projection": "10S--10N arithmetic mean",
            "own_climatology_baseline": (
                "selected actual-data event times (smoke-test product only)"
                if test
                else "complete simulation record"
            ),
            "cloud_sampling_region": (
                "broad event-centred context; never truncated by the LPO "
                "envelope; unavailable saved context is NaN"
            ),
            "lpo_envelope_role": "context contour only",
            "literature_geometry": (
                "5-degree centroid-relative zonal half-width follows the "
                "Lim-and-Son composite profile radius; point, 2.5, 3.75 and "
                "5 degree products are retained as sensitivity choices"
            ),
        },
    )
    units = {
        "unfiltered_precipitation": "mm day-1",
        "unfiltered_precipitable_water": "kg m-2",
        "unfiltered_olr": "W m-2",
        "unfiltered_temperature": "K",
        "unfiltered_upward_pressure_velocity": "Pa s-1",
        "unfiltered_specific_humidity": "g kg-1",
        "unfiltered_relative_humidity": "%",
        "background_zonal_wind": "m s-1",
        "buoyancy_frequency": "s-1",
    }
    for name, unit in units.items():
        output[name].attrs["units"] = unit
        anomaly = f"{name}_own_climatology_anomaly"
        if anomaly in output:
            output[anomaly].attrs["units"] = unit
    output.unfiltered_upward_pressure_velocity.attrs[
        "sign_convention"
    ] = "positive values represent ascent (-omega)"
    return output


def longitude_window_mean(
    field: xr.DataArray,
    half_width_degrees: float,
) -> xr.DataArray:
    """Mean one cached stencil with reproducible box-cell weighting."""

    if "relative_longitude" not in field.dims:
        raise ValueError("Longitude-window mean requires relative_longitude")
    coordinate = np.asarray(field.relative_longitude, dtype=float)
    if half_width_degrees == 0.0:
        weights = np.isclose(coordinate, 0.0).astype(float)
    else:
        spacing = float(np.nanmedian(np.abs(np.diff(coordinate))))
        cell_lower = coordinate - 0.5 * spacing
        cell_upper = coordinate + 0.5 * spacing
        weights = np.maximum(
            0.0,
            np.minimum(cell_upper, half_width_degrees)
            - np.maximum(cell_lower, -half_width_degrees),
        )
    if not np.any(weights > 0.0):
        raise ValueError(
            f"No stencil cells lie in +/-{half_width_degrees:g} degrees"
        )
    weights = xr.DataArray(
        weights,
        dims="relative_longitude",
        coords={"relative_longitude": field.relative_longitude},
    )
    return field.weighted(weights).mean(
        "relative_longitude", skipna=True
    )


def _event_cube(
    dataset: xr.Dataset,
    variable: str,
    *,
    lags: Sequence[int],
) -> xr.DataArray:
    """Arrange one event-step variable into event/lag sequence form."""

    if variable not in dataset:
        raise KeyError(f"Event-section cache lacks {variable!r}")
    data = dataset[variable]
    if "event_step" not in data.dims:
        raise ValueError(f"{variable!r} has no event_step dimension")
    event_ids = np.asarray(dataset.event_id.values).astype(str)
    lag_values = np.asarray(dataset.lag_hours.values, dtype=int)
    complete_events = []
    lag_set = set(int(value) for value in lags)
    for event_id in dict.fromkeys(event_ids):
        available = set(lag_values[event_ids == event_id])
        if lag_set.issubset(available):
            complete_events.append(event_id)
    if not complete_events:
        raise RuntimeError(
            f"No complete event sequence is available for {variable!r}"
        )
    trailing_dims = tuple(dim for dim in data.dims if dim != "event_step")
    trailing_shape = tuple(data.sizes[dim] for dim in trailing_dims)
    cube = np.full(
        (len(complete_events), len(lags), *trailing_shape),
        np.nan,
        dtype=float,
    )
    values = np.asarray(data, dtype=float)
    for event_index, event_id in enumerate(complete_events):
        event_positions = np.flatnonzero(event_ids == event_id)
        lookup = {
            int(lag_values[position]): int(position)
            for position in event_positions
        }
        for lag_index, lag in enumerate(lags):
            cube[event_index, lag_index] = values[lookup[int(lag)]]
    coords = {
        "event": np.asarray(complete_events, dtype=str),
        "lag_hours": np.asarray(lags, dtype=int),
        **{dim: data[dim] for dim in trailing_dims},
    }
    return xr.DataArray(
        cube,
        dims=("event", "lag_hours", *trailing_dims),
        coords=coords,
    )


def event_equal_statistics(
    cube: xr.DataArray,
    *,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> xr.Dataset:
    """Mean and percentile interval from complete-event resamples."""

    if "event" not in cube.dims:
        raise ValueError("Event-equal statistics require an event dimension")
    values = np.asarray(cube, dtype=float)
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    mean = np.divide(
        np.where(finite, values, 0.0).sum(axis=0),
        count,
        out=np.full(values.shape[1:], np.nan),
        where=count > 0,
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.full((n_bootstrap, *values.shape[1:]), np.nan)
    for draw_index in range(n_bootstrap):
        indices = rng.integers(0, values.shape[0], values.shape[0])
        drawn = values[indices]
        drawn_finite = np.isfinite(drawn)
        drawn_count = drawn_finite.sum(axis=0)
        bootstrap[draw_index] = np.divide(
            np.where(drawn_finite, drawn, 0.0).sum(axis=0),
            drawn_count,
            out=np.full(values.shape[1:], np.nan),
            where=drawn_count > 0,
        )
    flat = bootstrap.reshape(n_bootstrap, -1)
    percentiles = np.full((2, flat.shape[1]), np.nan)
    valid = np.isfinite(flat).any(axis=0)
    if valid.any():
        percentiles[:, valid] = np.nanpercentile(
            flat[:, valid], [2.5, 97.5], axis=0
        )
    percentiles = percentiles.reshape((2, *values.shape[1:]))
    dimensions = tuple(dim for dim in cube.dims if dim != "event")
    coords = {dim: cube[dim] for dim in dimensions}
    return xr.Dataset(
        {
            "mean": (dimensions, mean),
            "ci_low": (dimensions, percentiles[0]),
            "ci_high": (dimensions, percentiles[1]),
            "finite_event_count": (dimensions, count.astype(np.int32)),
        },
        coords=coords,
        attrs={
            "event_count": int(cube.sizes["event"]),
            "bootstrap_resamples": int(n_bootstrap),
            "sampling_unit": "complete LPO event sequence",
        },
    )


def _mean_for_windows(
    dataset: xr.Dataset,
    variable: str,
    *,
    lags: Sequence[int],
) -> tuple[xr.DataArray, xr.Dataset]:
    means = []
    default_cube = None
    for half_width in LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES:
        windowed = longitude_window_mean(dataset[variable], half_width)
        temporary = xr.Dataset(
            {variable: windowed},
            coords={
                "event_id": dataset.event_id,
                "lag_hours": dataset.lag_hours,
            },
        )
        cube = _event_cube(temporary, variable, lags=lags)
        means.append(cube.mean("event", skipna=True))
        if np.isclose(half_width, DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES):
            default_cube = cube
    if default_cube is None:
        raise RuntimeError("Default longitude window is absent")
    return (
        xr.concat(
            means,
            dim=xr.IndexVariable(
                "longitude_half_width_degrees",
                np.asarray(LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES),
            ),
        ),
        event_equal_statistics(default_cube),
    )


def _phase_composites(
    wave_sections: xr.Dataset,
    *,
    lags: Sequence[int],
) -> xr.Dataset:
    angle = _event_cube(
        wave_sections, "wave_phase_angle_degrees", lags=lags
    )
    phase = _event_cube(wave_sections, "wave_phase_number", lags=lags)
    amplitude = _event_cube(
        wave_sections, "wave_normalized_amplitude", lags=lags
    )
    strong = _event_cube(
        wave_sections, "wave_strong_amplitude", lags=lags
    ).astype(bool)
    phase_numbers = np.arange(1, 9, dtype=int)
    occurrence = xr.concat(
        [(phase == value).mean("event") for value in phase_numbers],
        dim=xr.IndexVariable("wave_phase", phase_numbers),
    ).transpose("lag_hours", "wave_band", "wave_phase")
    strong_occurrence = xr.concat(
        [
            ((phase == value) & strong).sum("event")
            / strong.sum("event").where(strong.sum("event") > 0)
            for value in phase_numbers
        ],
        dim=xr.IndexVariable("wave_phase", phase_numbers),
    ).transpose("lag_hours", "wave_band", "wave_phase")
    radians = np.deg2rad(angle)
    cosine = np.cos(radians).mean("event", skipna=True)
    sine = np.sin(radians).mean("event", skipna=True)
    concentration = np.hypot(cosine, sine)
    mean_angle = np.mod(np.rad2deg(np.arctan2(sine, cosine)), 360.0)
    return xr.Dataset(
        {
            "wave_phase_occurrence": occurrence,
            "strong_wave_phase_occurrence": strong_occurrence,
            "wave_phase_circular_concentration": concentration,
            "wave_mean_phase_angle_degrees": mean_angle,
            "wave_normalized_amplitude": amplitude.mean(
                "event", skipna=True
            ),
            "strong_wave_fraction": strong.mean("event"),
        },
        attrs={
            "phase_role": (
                "diagnostic of filtered-wave alignment only; LPO lag remains "
                "the compositing coordinate and unfiltered composites have "
                "no phase conditioning"
            )
        },
    )


def build_simulation_composites(
    event_sections: xr.Dataset,
    wave_sections: xr.Dataset,
    *,
    primary_k_m1: float,
    primary_m_m1: float,
    test: bool,
) -> xr.Dataset:
    """Reduce one simulation's raw event caches into fast plot products."""

    if test:
        event_ids = np.asarray(event_sections.event_id).astype(str)
        lag_values = np.asarray(event_sections.lag_hours, dtype=int)
        common_lags: set[int] | None = None
        for event_id in np.unique(event_ids):
            event_lags = set(lag_values[event_ids == event_id])
            common_lags = (
                event_lags
                if common_lags is None
                else common_lags.intersection(event_lags)
            )
        if not common_lags:
            raise ValueError("Test events have no common lifecycle lags")
        lags = np.asarray(sorted(common_lags), dtype=int)
    else:
        lags = np.asarray(LIFECYCLE_LAGS_HOURS, dtype=int)
    output = xr.Dataset()
    for variable in (
        *SURFACE_VARIABLES,
        *(f"{name}_own_climatology_anomaly" for name in SURFACE_VARIABLES),
        *VERTICAL_VARIABLES,
        *(f"{name}_own_climatology_anomaly" for name in VERTICAL_VARIABLES),
    ):
        mean, statistics = _mean_for_windows(
            event_sections, variable, lags=lags
        )
        output[variable] = mean
        output[f"{variable}_ci_low"] = statistics.ci_low
        output[f"{variable}_ci_high"] = statistics.ci_high
        output[f"{variable}_finite_event_count"] = (
            statistics.finite_event_count
        )

    for variable in FILTERED_SURFACE_VARIABLES + FILTERED_VERTICAL_VARIABLES:
        mean, statistics = _mean_for_windows(
            wave_sections, variable, lags=lags
        )
        output[variable] = mean
        output[f"{variable}_ci_low"] = statistics.ci_low
        output[f"{variable}_ci_high"] = statistics.ci_high

    background_cubes = {}
    for variable in ("background_zonal_wind", "buoyancy_frequency"):
        means = []
        default_cube = None
        for width in LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES:
            windowed = longitude_window_mean(event_sections[variable], width)
            temporary = xr.Dataset(
                {variable: windowed},
                coords={
                    "event_id": event_sections.event_id,
                    "lag_hours": event_sections.lag_hours,
                },
            )
            cube = _event_cube(temporary, variable, lags=lags)
            means.append(cube.mean("event", skipna=True))
            if np.isclose(width, DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES):
                default_cube = cube
        output[variable] = xr.concat(
            means,
            dim=xr.IndexVariable(
                "longitude_half_width_degrees",
                np.asarray(LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES),
            ),
        )
        background_cubes[variable] = default_cube

    zonal_wavenumbers = np.asarray([1, 2, 3], dtype=int)
    vertical_wavelengths = np.asarray([5.0, 10.0, 15.0])
    u_cube = background_cubes["background_zonal_wind"]
    n_cube = background_cubes["buoyancy_frequency"]
    group_shape = (
        len(zonal_wavenumbers),
        len(vertical_wavelengths),
        *u_cube.shape,
    )
    group = {
        name: np.full(group_shape, np.nan)
        for name in (
            "kelvin_reference_cgx",
            "kelvin_reference_cgz",
            "kelvin_reference_cgz_over_cgx",
        )
    }
    for k_index, zonal_wavenumber in enumerate(zonal_wavenumbers):
        for m_index, wavelength in enumerate(vertical_wavelengths):
            result = kelvin_group_velocities(
                np.asarray(u_cube),
                np.asarray(n_cube),
                zonal_wavenumber_m1=primary_k_m1 * zonal_wavenumber,
                vertical_wavenumber_m1=primary_m_m1 * (10.0 / wavelength),
                minimum_abs_cgx_m_s=0.1,
            )
            group["kelvin_reference_cgx"][k_index, m_index] = result[
                "cgx_m_s"
            ]
            group["kelvin_reference_cgz"][k_index, m_index] = result[
                "cgz_m_s"
            ]
            group["kelvin_reference_cgz_over_cgx"][k_index, m_index] = (
                result["ray_slope"]
            )
    group_dims = (
        "zonal_wavenumber", "vertical_wavelength_km", *u_cube.dims
    )
    group_coords = {
        "zonal_wavenumber": zonal_wavenumbers,
        "vertical_wavelength_km": vertical_wavelengths,
        **{dimension: u_cube[dimension] for dimension in u_cube.dims},
    }
    for variable, values in group.items():
        output[variable] = xr.DataArray(
            values, dims=group_dims, coords=group_coords
        ).mean("event", skipna=True)

    cloud_cube = _event_cube(
        event_sections, "cloud_class_indicator", lags=lags
    )
    envelope_cube = _event_cube(
        event_sections, "lpo_envelope_indicator", lags=lags
    )
    occurrence = cloud_cube.mean("event", skipna=True)
    total_occurrence = occurrence.sum(
        "cloud_class", skipna=True, min_count=1
    )
    class_index = occurrence.fillna(-1.0).argmax("cloud_class")
    modal_code = xr.DataArray(
        np.asarray(CLOUD_CLASS_CODES, dtype=np.int8),
        dims="cloud_class",
        coords={"cloud_class": occurrence.cloud_class},
    ).isel(cloud_class=class_index)
    modal_agreement = (
        occurrence.max("cloud_class", skipna=True)
        / total_occurrence.where(total_occurrence > 0.0)
    )
    modal_code = modal_code.where(total_occurrence >= 0.1, 0).astype(np.int8)
    output["cloud_class_occurrence"] = occurrence
    output["total_high_cloud_occurrence"] = total_occurrence
    output["modal_high_cloud_class"] = modal_code
    output["modal_high_cloud_class_agreement"] = modal_agreement
    output["lpo_envelope_occurrence"] = envelope_cube.mean(
        "event", skipna=True
    )
    output = xr.merge([output, _phase_composites(wave_sections, lags=lags)])
    output["precipitation_wave_rms"] = wave_sections.precipitation_wave_rms
    output.attrs.update(
        {
            "cache_version": COMPOSITE_CACHE_VERSION,
            "source_simulation": event_sections.attrs["source_simulation"],
            "test_subset": int(test),
            "event_weighting": "one equal weight per complete LPO event",
            "bootstrap": (
                f"{BOOTSTRAP_RESAMPLES} complete-event sequence resamples; "
                "intervals stored for the default 5-degree longitude window"
            ),
            "default_longitude_half_width_degrees": (
                DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
            ),
            "kelvin_reference_method": (
                "Lim-and-Son dry Kelvin reference using event-local U and N; "
                "not a generic moist-mode group velocity"
            ),
            "primary_k_m-1": float(primary_k_m1),
            "primary_m_m-1": float(primary_m_m1),
            "cloud_mode_definition": (
                "most frequent classified high-cloud regime across events; "
                "shown only where total high-cloud occurrence is at least 0.1"
            ),
            "cloud_mode_agreement_definition": (
                "modal class occurrence divided by total classified "
                "high-cloud occurrence"
            ),
        }
    )
    return output


def combine_simulation_composites(
    composites: Mapping[str, xr.Dataset],
) -> xr.Dataset:
    """Combine independently built simulation products without reweighting."""

    ordered = list(composites)
    test_flags = {
        bool(composites[name].attrs.get("test_subset", False))
        for name in ordered
    }
    if len(test_flags) != 1:
        raise ValueError("Cannot mix test and publication composites")
    test_subset = test_flags.pop()
    result = xr.concat(
        [composites[name] for name in ordered],
        dim=xr.IndexVariable("simulation", ordered),
        join="inner" if test_subset else "exact",
        combine_attrs="drop_conflicts",
    )
    if result.sizes.get("lag_hours", 0) == 0:
        raise ValueError("Simulation composites have no common test lags")
    result.attrs.update(
        {
            "cache_version": COMPOSITE_CACHE_VERSION,
            "test_subset": int(test_subset),
            "simulation_comparison": (
                "each simulation composited against its own tracked events "
                "and own climatology"
            ),
            "literature_presentation": (
                "lag-latitude and lag-pressure adaptations of Lim-and-Son "
                "and Rios-Berrios phase/vertical-section conventions; LPO "
                "lag is not interpreted as filtered-wave phase"
            ),
        }
    )
    return result
