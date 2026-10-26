"""Import-safe event-section and composite helpers for LPO lifecycles.

The tracking diagnostic remains the sole owner of event identities, masks and
reference times.  This module only samples those saved events and reduces
available event sequences with one equal weight per event at each lag.
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
    align_vertical_velocity_to_reference,
    mpas_moist_air_density,
    native_midlevel_height,
    pressure_to_pa,
    progress,
    vertical_dim,
)
from diagnostics.waves.equatorial_wave_config import (
    EASTWARD_INTRASEASONAL_FILTER,
    LPO_LIFECYCLE_FILTERS,
)
from diagnostics.waves.wave_metrics import (
    blockwise_buoyancy_frequency_squared,
    buoyancy_frequency_squared,
    kelvin_group_velocities,
    mrg_group_velocities,
)


SECTION_CACHE_VERSION = "lpo-lifecycle-sections-v11-raining-properties"
COMPOSITE_CACHE_VERSION = (
    "lpo-lifecycle-2d-composites-v20-raining-properties"
)
LIFECYCLE_RAIN_THRESHOLD_MM_DAY = 1.0
PROPERTY_SAMPLING_DEFINITION = (
    "unfiltered surface and thermodynamic sections: saved LPO envelope "
    "with concurrent unsmoothed precipitation > 1 mm/day; cloud fraction: "
    "complete saved envelope; cloud and wave context retain surrounding fields"
)

DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES = 5.0
LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES = (0.0, 2.5, 3.75, 5.0)
RELATIVE_LONGITUDE_DEGREES = np.arange(-7.5, 7.51, 0.5)
RELATIVE_LATITUDE_DEGREES = np.arange(-10.0, 10.01, 0.5)
CLOUD_RELATIVE_LONGITUDE_DEGREES = np.arange(-150.0, 150.01, 0.5)
CLOUD_RELATIVE_LATITUDE_DEGREES = np.arange(-20.0, 20.01, 0.5)
CLOUD_PLOT_LONGITUDE_BOUNDS = (-30.0, 30.0)
CLOUD_PLOT_LATITUDE_BOUNDS = (-20.0, 20.0)
CLOUD_BACKGROUND_SECTORS = (
    (60.0, 150.0),
    (45.0, 90.0),
    (90.0, 150.0),
)
HEIGHT_M = np.arange(0.0, 25_000.1, 250.0)
RAY_HEIGHT_M = np.arange(0.0, 25_000.1, 500.0)
RAY_RELATIVE_LONGITUDE_DEGREES = np.arange(-60.0, 60.01, 1.0)
RAY_RELATIVE_LATITUDE_DEGREES = np.arange(-10.0, 10.01, 0.5)
RAY_LAUNCH_LONGITUDE_DEGREES = np.arange(-60.0, 60.01, 20.0)
MATSUNO_RELATIVE_LONGITUDE_DEGREES = np.arange(-30.0, 30.01, 1.0)
MATSUNO_RELATIVE_LATITUDE_DEGREES = np.arange(-20.0, 20.01, 1.0)
EARTH_RADIUS_M = 6.371e6
MRG_ZONAL_WAVENUMBER = -2
MRG_VERTICAL_WAVELENGTH_M = 10_000.0
DEEP_TROPICAL_LATITUDE_BOUNDS = (-10.0, 10.0)
AVERAGING_SENSITIVITY_LATITUDE_BOUNDS = (-2.5, 2.5)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260731
PHASE_ALIGNMENT_WINDOWS_HOURS = {
    EASTWARD_INTRASEASONAL_FILTER: (-12, -6, 0, 6, 12),
    "Kelvin": (-12, -6, 0, 6, 12),
    "MRG": (-6, 0, 6),
    "Easterly": (0,),
    "ER": (-6, 0, 6),
}


def cosine_weighted_latitude_mean(
    field: xr.DataArray,
    physical_latitude: xr.DataArray,
    dimension: str,
) -> xr.DataArray:
    """Average one spatial latitude dimension using physical latitudes."""
    weights = np.cos(np.deg2rad(physical_latitude)).clip(
        min=0.0
    ).fillna(0.0)
    return field.weighted(weights).mean(dimension, skipna=True)


def event_relative_latitude_mean(
    field: xr.DataArray,
    centre_latitude: xr.DataArray,
    dimension: str = "relative_latitude",
) -> xr.DataArray:
    """Average an event-relative field using each event's absolute latitude."""
    physical_latitude = centre_latitude + field[dimension]
    return cosine_weighted_latitude_mean(
        field, physical_latitude, dimension
    )

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
    "unfiltered_cloud_fraction",
    "unfiltered_static_stability",
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
    "filtered_zonal_wind",
    "filtered_static_stability",
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


def sample_centroid_latitude_native_sections(
    field: xr.DataArray,
    steps: pd.DataFrame,
    *,
    relative_latitude=RELATIVE_LATITUDE_DEGREES,
    relative_longitude=RELATIVE_LONGITUDE_DEGREES,
    latitude_column="precip_weighted_centroid_lat",
    longitude_column="precip_weighted_centroid_lon",
    horizontal_mask: xr.DataArray | None = None,
    return_weights: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """Sample native profiles within 10 degrees of each LPO latitude."""

    zdim = vertical_dim(field)
    required = {"latitude", "longitude", zdim}
    if not required.issubset(field.dims):
        raise ValueError(
            f"Native-section field requires {required}; got {field.dims}"
        )
    if "Time" in field.dims:
        field = field.transpose("Time", "latitude", "longitude", zdim)
        time_indices = _exact_time_indices(field.Time.values, steps.Time)
        source = field.isel(
            Time=xr.DataArray(time_indices, dims="event_step")
        )
    else:
        field = field.transpose("latitude", "longitude", zdim)
        source = field.expand_dims(event_step=np.arange(len(steps)))
    target_latitude = (
        np.asarray(steps[latitude_column], dtype=float)[:, None]
        + np.asarray(relative_latitude, dtype=float)[None, :]
    )
    target_longitude = (
        np.asarray(steps[longitude_column], dtype=float)[:, None]
        + np.asarray(relative_longitude, dtype=float)[None, :]
    )
    latitude_indices, latitude_mismatch = _nearest_indices(
        field.latitude.values, target_latitude
    )
    longitude_indices, longitude_mismatch = _nearest_indices(
        field.longitude.values, target_longitude, periodic=True
    )
    tolerance = 0.51 * MPAS_LPT_GRID_SPACING_DEGREES
    if (
        np.nanmax(latitude_mismatch) > tolerance
        or np.nanmax(longitude_mismatch) > tolerance
    ):
        raise RuntimeError(
            "Model grid does not cover an LPO-centred native section"
        )
    sampled = source.isel(
        latitude=xr.DataArray(
            latitude_indices,
            dims=("event_step", "relative_latitude"),
        ),
        longitude=xr.DataArray(
            longitude_indices,
            dims=("event_step", "relative_longitude"),
        ),
    )
    sampled = sampled.reset_coords(drop=True).assign_coords(
        event_step=np.arange(len(steps)),
        relative_latitude=np.asarray(relative_latitude, dtype=float),
        relative_longitude=np.asarray(relative_longitude, dtype=float),
    )
    if horizontal_mask is not None:
        sampled = sampled.where(horizontal_mask)
    latitude_weights = xr.DataArray(
        np.clip(np.cos(np.deg2rad(target_latitude)), 0.0, None),
        dims=("event_step", "relative_latitude"),
        coords={
            "event_step": sampled.event_step,
            "relative_latitude": sampled.relative_latitude,
        },
    )
    mean = sampled.weighted(latitude_weights).mean(
        "relative_latitude", skipna=True
    )
    reduced = xr.Dataset({"mean": mean})
    if return_weights:
        # Keep the contributing area for the later longitude mean.
        reduced["weight"] = latitude_weights.where(
            sampled.notnull(), 0.0
        ).sum("relative_latitude")
    reduced = reduced.transpose(
        "event_step", zdim, "relative_longitude"
    ).rename({zdim: "native_level"}).assign_coords(
        native_level=np.arange(source.sizes[zdim])
    ).load()
    if return_weights:
        return reduced["mean"], reduced["weight"]
    return reduced["mean"]


def sample_event_relative_surface(
    field: xr.DataArray,
    steps: pd.DataFrame,
    *,
    relative_latitude=RELATIVE_LATITUDE_DEGREES,
    relative_longitude=RELATIVE_LONGITUDE_DEGREES,
    latitude_column="precip_weighted_centroid_lat",
    longitude_column="precip_weighted_centroid_lon",
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
        np.asarray(steps[latitude_column], dtype=float)[:, None]
        + np.asarray(relative_latitude, dtype=float)[None, :]
    )
    target_longitude = (
        np.asarray(steps[longitude_column], dtype=float)[:, None]
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


def _sample_deep_tropical_native_profiles(
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
    deep_tropical = field.sel(
        latitude=_ascending_slice(
            field.latitude, *DEEP_TROPICAL_LATITUDE_BOUNDS
        )
    )
    deep_tropical = cosine_weighted_latitude_mean(
        deep_tropical, deep_tropical.latitude, "latitude"
    )
    if "Time" in deep_tropical.dims:
        time_indices = _exact_time_indices(
            deep_tropical.Time.values, steps.Time
        )
        deep_tropical = deep_tropical.isel(
            Time=xr.DataArray(time_indices, dims="event_step")
        )
    else:
        deep_tropical = deep_tropical.expand_dims(
            event_step=np.arange(len(steps))
        )
    target_longitude = (
        np.asarray(steps.precip_weighted_centroid_lon, dtype=float)[:, None]
        + RELATIVE_LONGITUDE_DEGREES[None, :]
    )
    longitude_indices, mismatch = _nearest_indices(
        deep_tropical.longitude.values, target_longitude, periodic=True
    )
    if np.nanmax(mismatch) > 0.51 * MPAS_LPT_GRID_SPACING_DEGREES:
        raise RuntimeError(
            "Model longitude grid is incompatible with the fixed LPO grid"
        )
    sampled = deep_tropical.isel(
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
    sampled_theta = _sample_deep_tropical_native_profiles(theta, steps)
    sampled_height = _sample_deep_tropical_native_profiles(height, steps)
    sampled_wind = _sample_deep_tropical_native_profiles(wind, steps)
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


def geometric_prepeak_speed_m_s(event: pd.DataFrame) -> float:
    """Return geometric-centroid zonal speed over the 48 hours before peak."""

    required = {"Time", "lag_hours", "centroid_lon", "centroid_lat"}
    missing = sorted(required - set(event.columns))
    if missing:
        raise ValueError(f"Geometric propagation requires {missing}")
    window = event[event.lag_hours.between(-48.0, 0.0)].sort_values("Time")
    time = pd.DatetimeIndex(pd.to_datetime(window.Time))
    longitude = np.asarray(window.centroid_lon, dtype=float)
    latitude = np.asarray(window.centroid_lat, dtype=float)
    if not len(time):
        return np.nan
    elapsed_hours = np.asarray(
        (time - time[0]) / pd.Timedelta(hours=1), dtype=float
    )
    valid = (
        np.isfinite(elapsed_hours)
        & np.isfinite(longitude)
        & np.isfinite(latitude)
    )
    if valid.sum() < 2 or np.ptp(elapsed_hours[valid]) <= 0.0:
        return np.nan
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(longitude[valid])))
    degrees_per_hour = np.polyfit(
        elapsed_hours[valid], unwrapped, 1
    )[0]
    mean_latitude = float(np.mean(latitude[valid]))
    return float(
        degrees_per_hour
        * (np.pi / 180.0)
        * EARTH_RADIUS_M
        * np.cos(np.deg2rad(mean_latitude))
        / 3600.0
    )


def peak_geometric_steps(steps: pd.DataFrame) -> pd.DataFrame:
    """Return one lag-zero row and geometric speed for each saved event."""

    required = {"event_id", "lag_hours", "centroid_lon", "centroid_lat"}
    missing = sorted(required - set(steps.columns))
    if missing:
        raise ValueError(f"Peak geometric sections require {missing}")
    speeds = {
        str(event_id): geometric_prepeak_speed_m_s(event)
        for event_id, event in steps.groupby("event_id", sort=False)
    }
    peak = steps[np.isclose(steps.lag_hours, 0.0)].copy()
    peak = peak.sort_values(["event_id", "Time"]).drop_duplicates(
        "event_id", keep="last"
    )
    peak["lpo_zonal_speed_48h_m_s"] = peak.event_id.astype(str).map(speeds)
    peak["eastward_geometric_propagation"] = (
        peak.lpo_zonal_speed_48h_m_s > 0.0
    )
    return peak.reset_index(drop=True)


def native_peak_ray_sections(
    dataset: xr.Dataset,
    peak_steps: pd.DataFrame,
) -> xr.Dataset:
    """Sample event-local U, N, N2, and pressure on geometric height."""

    theta = dataset.theta
    pressure_hpa = pressure_to_pa(dataset.pressure) / 100.0
    height = native_midlevel_height(dataset, theta)
    fields = {
        "theta": theta,
        "height": height,
        "wind": dataset.uReconstructZonal,
        "pressure": pressure_hpa,
    }
    sampled = {
        name: sample_centroid_latitude_native_sections(
            field,
            peak_steps,
            relative_latitude=RAY_RELATIVE_LATITUDE_DEGREES,
            relative_longitude=RAY_RELATIVE_LONGITUDE_DEGREES,
            latitude_column="centroid_lat",
            longitude_column="centroid_lon",
        )
        for name, field in fields.items()
    }
    shape = (
        len(peak_steps), len(RAY_HEIGHT_M),
        len(RAY_RELATIVE_LONGITUDE_DEGREES),
    )
    values = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in ("wind", "n", "n2", "pressure")
    }
    for event_index in range(len(peak_steps)):
        for longitude_index in range(len(RAY_RELATIVE_LONGITUDE_DEGREES)):
            source = {
                name: np.asarray(
                    field[event_index, :, longitude_index], dtype=float
                )
                for name, field in sampled.items()
            }
            finite = np.logical_and.reduce(
                [np.isfinite(source[name]) for name in source]
            )
            if finite.sum() < 3:
                continue
            order = np.argsort(source["height"][finite])
            z = source["height"][finite][order]
            unique = np.r_[True, np.diff(z) > 0.0]
            z = z[unique]
            if len(z) < 3:
                continue
            theta_profile = source["theta"][finite][order][unique]
            wind_profile = source["wind"][finite][order][unique]
            pressure_profile = source["pressure"][finite][order][unique]
            n2_profile = buoyancy_frequency_squared(
                theta_profile, z, mask_nonpositive=False
            )
            inside = (RAY_HEIGHT_M >= z[0]) & (RAY_HEIGHT_M <= z[-1])
            values["wind"][event_index, inside, longitude_index] = np.interp(
                RAY_HEIGHT_M[inside], z, wind_profile
            )
            values["n2"][event_index, inside, longitude_index] = np.interp(
                RAY_HEIGHT_M[inside], z, n2_profile
            )
            interpolated_n2 = values["n2"][
                event_index, inside, longitude_index
            ]
            values["n"][event_index, inside, longitude_index] = np.sqrt(
                np.where(interpolated_n2 > 0.0, interpolated_n2, np.nan)
            )
            values["pressure"][event_index, inside, longitude_index] = (
                np.interp(RAY_HEIGHT_M[inside], z, pressure_profile)
            )
    dims = ("peak_event", "ray_height_m", "ray_relative_longitude")
    coords = {
        "peak_event": np.arange(len(peak_steps)),
        "ray_height_m": RAY_HEIGHT_M,
        "ray_relative_longitude": RAY_RELATIVE_LONGITUDE_DEGREES,
    }
    output = xr.Dataset(
        {
            "ray_background_zonal_wind": (dims, values["wind"]),
            "ray_buoyancy_frequency": (dims, values["n"]),
            "ray_unfiltered_static_stability": (dims, values["n2"]),
            "ray_pressure_hpa": (dims, values["pressure"]),
        },
        coords=coords,
    )
    output.ray_background_zonal_wind.attrs["units"] = "m s-1"
    output.ray_buoyancy_frequency.attrs["units"] = "s-1"
    output.ray_unfiltered_static_stability.attrs["units"] = "s-2"
    output.ray_unfiltered_static_stability.attrs[
        "nonpositive_values"
    ] = "retained"
    output.ray_pressure_hpa.attrs["units"] = "hPa"
    return output


def cloud_top_height_occurrence(
    classification: Mapping[str, xr.DataArray],
    peak_steps: pd.DataFrame,
    mask_name: str,
) -> xr.DataArray:
    """Bin latitude-mean cloud occurrence by geometric cloud-top height."""

    height = sample_event_relative_surface(
        classification["cloud_top_height"],
        peak_steps,
        relative_latitude=RAY_RELATIVE_LATITUDE_DEGREES,
        relative_longitude=RAY_RELATIVE_LONGITUDE_DEGREES,
        latitude_column="centroid_lat",
        longitude_column="centroid_lon",
    )
    indicator = sample_event_relative_surface(
        classification[mask_name].astype(float),
        peak_steps,
        relative_latitude=RAY_RELATIVE_LATITUDE_DEGREES,
        relative_longitude=RAY_RELATIVE_LONGITUDE_DEGREES,
        latitude_column="centroid_lat",
        longitude_column="centroid_lon",
    )
    edges = np.concatenate(
        (
            [RAY_HEIGHT_M[0] - 250.0],
            0.5 * (RAY_HEIGHT_M[:-1] + RAY_HEIGHT_M[1:]),
            [RAY_HEIGHT_M[-1] + 250.0],
        )
    )
    output = np.full(
        (
            len(peak_steps), len(RAY_HEIGHT_M),
            len(RAY_RELATIVE_LONGITUDE_DEGREES),
        ),
        np.nan,
        dtype=np.float32,
    )
    height_units = str(
        classification["cloud_top_height"].attrs.get("units", "")
    ).strip().lower()
    if height_units in {"km", "kilometre", "kilometres", "kilometer", "kilometers"}:
        height_scale = 1000.0
    elif height_units in {"m", "metre", "metres", "meter", "meters"}:
        height_scale = 1.0
    else:
        raise ValueError(
            "Cloud-top height requires explicit metre or kilometre units"
        )
    height_values = height_scale * np.asarray(height, dtype=float)
    indicator_values = np.asarray(indicator, dtype=float)
    physical_latitude = (
        np.asarray(peak_steps.centroid_lat, dtype=float)[:, None]
        + np.asarray(RAY_RELATIVE_LATITUDE_DEGREES, dtype=float)[None, :]
    )
    latitude_weights = np.cos(np.deg2rad(physical_latitude))
    for event_index in range(len(peak_steps)):
        for longitude_index in range(len(RAY_RELATIVE_LONGITUDE_DEGREES)):
            valid = np.isfinite(indicator_values[event_index, :, longitude_index])
            if not valid.any():
                continue
            weights = latitude_weights[event_index]
            denominator = np.sum(weights[valid])
            if denominator <= 0.0:
                continue
            cloud_height = height_values[event_index, :, longitude_index]
            cloud = indicator_values[event_index, :, longitude_index] > 0.5
            for height_index in range(len(RAY_HEIGHT_M)):
                in_bin = (
                    cloud
                    & np.isfinite(cloud_height)
                    & (cloud_height >= edges[height_index])
                    & (cloud_height < edges[height_index + 1])
                )
                output[event_index, height_index, longitude_index] = (
                    np.sum(weights[in_bin]) / denominator
                )
    return xr.DataArray(
        output,
        dims=("peak_event", "ray_height_m", "ray_relative_longitude"),
        coords={
            "peak_event": np.arange(len(peak_steps)),
            "ray_height_m": RAY_HEIGHT_M,
            "ray_relative_longitude": RAY_RELATIVE_LONGITUDE_DEGREES,
        },
        attrs={
            "units": "1",
            "latitude_weighting": "cosine of physical latitude",
        },
    )


def relative_vorticity_from_wind(
    zonal_wind: xr.DataArray,
    meridional_wind: xr.DataArray,
) -> xr.DataArray:
    """Calculate spherical relative vorticity on a regular lat-lon grid."""

    zonal_wind, meridional_wind = xr.align(
        zonal_wind, meridional_wind, join="exact"
    )
    latitude = np.deg2rad(np.asarray(zonal_wind.latitude, dtype=float))
    longitude = np.deg2rad(np.asarray(zonal_wind.longitude, dtype=float))
    if len(latitude) < 3 or len(longitude) < 3:
        raise ValueError("Relative vorticity requires at least three grid cells")
    u = np.asarray(zonal_wind, dtype=float)
    v = np.asarray(meridional_wind, dtype=float)
    latitude_axis = zonal_wind.get_axis_num("latitude")
    longitude_axis = zonal_wind.get_axis_num("longitude")
    longitude_spacing = float(np.median(np.diff(longitude)))
    dvd_lambda = (
        np.roll(v, -1, axis=longitude_axis)
        - np.roll(v, 1, axis=longitude_axis)
    ) / (2.0 * longitude_spacing)
    cosine = np.cos(latitude)
    cosine_shape = [1] * u.ndim
    cosine_shape[latitude_axis] = len(cosine)
    u_cosine = u * cosine.reshape(cosine_shape)
    du_cosine_dlatitude = np.gradient(
        u_cosine, latitude, axis=latitude_axis, edge_order=2
    )
    denominator = EARTH_RADIUS_M * cosine.reshape(cosine_shape)
    vorticity = (dvd_lambda - du_cosine_dlatitude) / denominator
    return xr.DataArray(
        vorticity,
        dims=zonal_wind.dims,
        coords=zonal_wind.coords,
        name="relative_vorticity",
        attrs={"units": "s-1"},
    )


def matsuno_gill_peak_sections(
    dataset: xr.Dataset,
    peak_steps: pd.DataFrame,
) -> xr.Dataset:
    """Return eastward-event wind and vorticity anomalies at 850 and 200 hPa."""

    eastward = peak_steps[
        peak_steps.eastward_geometric_propagation.astype(bool)
    ].reset_index(drop=True)
    pressure_levels = np.asarray([850.0, 200.0])
    coords = {
        "matsuno_event": np.arange(len(eastward)),
        "matsuno_pressure_hpa": pressure_levels,
        "matsuno_relative_latitude": MATSUNO_RELATIVE_LATITUDE_DEGREES,
        "matsuno_relative_longitude": MATSUNO_RELATIVE_LONGITUDE_DEGREES,
    }
    dims = (
        "matsuno_event", "matsuno_pressure_hpa",
        "matsuno_relative_latitude", "matsuno_relative_longitude",
    )
    if eastward.empty:
        empty = np.full(tuple(len(coords[dim]) for dim in dims), np.nan)
        return xr.Dataset(
            {
                "matsuno_zonal_wind_anomaly": (dims, empty.copy()),
                "matsuno_meridional_wind_anomaly": (dims, empty.copy()),
                "matsuno_relative_vorticity_anomaly": (dims, empty.copy()),
                "matsuno_event_id": (("matsuno_event",), np.asarray([], str)),
            },
            coords=coords,
        )
    u = dataset.uzonal_isobaric.sel(pres=pressure_levels)
    v = dataset.umeridional_isobaric.sel(pres=pressure_levels)
    peak_times = np.sort(pd.to_datetime(eastward.Time).unique())
    u_anomaly = u.sel(Time=peak_times) - u.mean("Time", skipna=True)
    v_anomaly = v.sel(Time=peak_times) - v.mean("Time", skipna=True)
    vorticity_anomaly = relative_vorticity_from_wind(u_anomaly, v_anomaly)

    def sample(field):
        levels = []
        for pressure in pressure_levels:
            selected = sample_event_relative_surface(
                field.sel(pres=pressure),
                eastward,
                relative_latitude=MATSUNO_RELATIVE_LATITUDE_DEGREES,
                relative_longitude=MATSUNO_RELATIVE_LONGITUDE_DEGREES,
                latitude_column="centroid_lat",
                longitude_column="centroid_lon",
            ).rename(
                {
                    "event_step": "matsuno_event",
                    "relative_latitude": "matsuno_relative_latitude",
                    "relative_longitude": "matsuno_relative_longitude",
                }
            )
            levels.append(selected)
        return xr.concat(
            levels,
            dim=xr.IndexVariable("matsuno_pressure_hpa", pressure_levels),
        ).transpose(*dims)

    output = xr.Dataset(
        {
            "matsuno_zonal_wind_anomaly": sample(u_anomaly),
            "matsuno_meridional_wind_anomaly": sample(v_anomaly),
            "matsuno_relative_vorticity_anomaly": sample(vorticity_anomaly),
            "matsuno_event_id": (
                ("matsuno_event",), eastward.event_id.astype(str).values
            ),
        },
        coords=coords,
    )
    output.matsuno_zonal_wind_anomaly.attrs["units"] = "m s-1"
    output.matsuno_meridional_wind_anomaly.attrs["units"] = "m s-1"
    output.matsuno_relative_vorticity_anomaly.attrs["units"] = "s-1"
    return output


def build_peak_event_sections(
    dataset: xr.Dataset,
    precipitation: xr.DataArray,
    classification: Mapping[str, xr.DataArray],
    steps: pd.DataFrame,
) -> xr.Dataset:
    """Build compact lag-zero geometric sections for rays and Matsuno-Gill."""

    peak = peak_geometric_steps(steps)
    output = native_peak_ray_sections(dataset, peak)
    precipitation_section = sample_event_relative_surface(
        precipitation,
        peak,
        relative_latitude=RAY_RELATIVE_LATITUDE_DEGREES,
        relative_longitude=RAY_RELATIVE_LONGITUDE_DEGREES,
        latitude_column="centroid_lat",
        longitude_column="centroid_lon",
    )
    precipitation_section = event_relative_latitude_mean(
        precipitation_section,
        xr.DataArray(
            np.asarray(peak.centroid_lat, dtype=float),
            dims="event_step",
        ),
    ).rename(
        {
            "event_step": "peak_event",
            "relative_longitude": "ray_relative_longitude",
        }
    )
    output["ray_lpo_precipitation"] = precipitation_section
    output["ray_high_cloud_occurrence"] = cloud_top_height_occurrence(
        classification, peak, "high_cloud_mask"
    )
    output["ray_nonprecipitating_cloud_occurrence"] = (
        cloud_top_height_occurrence(
            classification, peak, "nonprecipitating_high_cloud_mask"
        )
    )
    output["peak_event_id"] = (
        ("peak_event",), peak.event_id.astype(str).values
    )
    output["peak_time"] = (("peak_event",), pd.to_datetime(peak.Time))
    output["geometric_centroid_lon"] = (
        ("peak_event",), peak.centroid_lon.astype(float).values
    )
    output["geometric_centroid_lat"] = (
        ("peak_event",), peak.centroid_lat.astype(float).values
    )
    output["lpo_zonal_speed_48h_m_s"] = (
        ("peak_event",), peak.lpo_zonal_speed_48h_m_s.astype(float).values
    )
    output = xr.merge(
        [output, matsuno_gill_peak_sections(dataset, peak)],
        compat="no_conflicts",
    )
    output.attrs.update(
        peak_event_count=len(peak),
        matsuno_eastward_event_count=int(
            peak.eastward_geometric_propagation.sum()
        ),
        ray_centroid="geometric LPT centroid",
        ray_speed_window="linear regression from lag -48 to 0 hours",
        matsuno_anomaly_baseline="own-simulation complete-record climatology",
    )
    return output


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
    selected = selected.sort_values(
        ["event_id", "lag_hours", "Time"]
    ).drop_duplicates(["event_id", "lag_hours"], keep="last")
    selected = selected.reset_index(drop=True)
    selected["event_step"] = np.arange(len(selected))
    if selected.empty:
        raise RuntimeError("No authoritative LPO steps remain for 2-D sections")
    return selected


def property_envelope_sections(
    primary: Path, simulation: str, steps: pd.DataFrame
) -> xr.DataArray:
    """Sample saved envelopes without loading the broad cloud context."""

    values = np.zeros(
        (len(steps), len(RELATIVE_LATITUDE_DEGREES),
         len(RELATIVE_LONGITUDE_DEGREES)), dtype=bool,
    )
    for event_id, event_steps in steps.groupby("event_id", sort=False):
        with xr.open_dataset(_event_product_path(primary, simulation, event_id)) as product:
            for row in event_steps.itertuples():
                envelope = _centred_context_field(
                    product.lpt_envelope_mask.sel(Time=row.Time),
                    float(row.precip_weighted_centroid_lon),
                    float(row.precip_weighted_centroid_lat),
                ).reindex(
                    cloud_relative_latitude=RELATIVE_LATITUDE_DEGREES,
                    cloud_relative_longitude=RELATIVE_LONGITUDE_DEGREES,
                    fill_value=0.0,
                )
                values[int(row.event_step)] = np.asarray(envelope) > 0.5
    return xr.DataArray(
        values,
        dims=("event_step", "relative_latitude", "relative_longitude"),
        coords={
            "event_step": np.arange(len(steps)),
            "relative_latitude": RELATIVE_LATITUDE_DEGREES,
            "relative_longitude": RELATIVE_LONGITUDE_DEGREES,
        },
    )


def build_unfiltered_event_sections(
    dataset: xr.Dataset,
    precipitation: xr.DataArray,
    classification: Mapping[str, xr.DataArray],
    steps: pd.DataFrame,
    *,
    primary: Path,
    simulation: str,
    test: bool,
) -> xr.Dataset:
    """Build the slow, event-level unfiltered section cache once."""

    steps = select_section_steps(steps, test=test)
    required = {
        "precipw", "olrtoa", "theta", "pressure", "rho", "w", "qv",
        "qc", "qr", "qi", "qs", "qg",
        "relhum", "cldfrac", "uReconstructZonal", "zgrid",
        "uzonal_isobaric", "umeridional_isobaric",
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
    if "area_integrated_24h_precip_km2_mm" in steps:
        variables["lpo_envelope_precipitation"] = (
            ("event_step",),
            steps.area_integrated_24h_precip_km2_mm.astype(float),
        )
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
    progress(f"{simulation}: sampling lifecycle property envelope")
    property_envelope = property_envelope_sections(primary, simulation, steps)
    sampled_precipitation = sample_event_relative_surface(precipitation, steps)
    rain_mask = property_envelope & (
        sampled_precipitation > LIFECYCLE_RAIN_THRESHOLD_MM_DAY
    )
    variables["property_envelope_mask"] = property_envelope
    variables["property_rain_mask"] = rain_mask
    surface_sources = {
        "unfiltered_precipitable_water": dataset.precipw,
        "unfiltered_olr": dataset.olrtoa,
    }
    variables["unfiltered_precipitation"] = sampled_precipitation.where(rain_mask)
    for name, source in surface_sources.items():
        progress(f"{simulation}: sampling lifecycle sections for {name}")
        variables[name] = sample_event_relative_surface(source, steps).where(rain_mask)
    pressure_pa = pressure_to_pa(dataset.pressure)
    theta = dataset.theta
    native_temperature = theta * (pressure_pa / 100_000.0) ** (
        287.0 / 1004.0
    )
    vapour_specific_humidity = dataset.qv / (1.0 + dataset.qv)
    height = native_midlevel_height(dataset, theta)
    zdim = vertical_dim(theta)
    native_n2 = blockwise_buoyancy_frequency_squared(
        theta,
        height,
        zdim,
        mask_nonpositive=False,
    )
    native_sources = {
        "unfiltered_temperature": native_temperature,
        "unfiltered_upward_pressure_velocity": (
            9.80665
            * mpas_moist_air_density(dataset)
            * align_vertical_velocity_to_reference(
                dataset.w, dataset.rho
            )
        ),
        "unfiltered_specific_humidity": 1000.0 * vapour_specific_humidity,
        "unfiltered_relative_humidity": dataset.relhum,
        "unfiltered_cloud_fraction": dataset.cldfrac,
        "unfiltered_static_stability": native_n2,
    }
    progress(f"{simulation}: sampling native pressure lifecycle sections")
    native_sources["native_pressure_hpa"] = pressure_pa / 100.0
    for name, source in native_sources.items():
        progress(f"{simulation}: sampling lifecycle sections for {name}")
        sampling_mask = (
            property_envelope if name == "unfiltered_cloud_fraction" else rain_mask
        )
        mean, weight = sample_centroid_latitude_native_sections(
            source, steps, horizontal_mask=sampling_mask, return_weights=True
        )
        weight_name = f"{name}_latitude_weight"
        mean.attrs["longitude_sample_weight"] = weight_name
        variables[name] = mean
        variables[weight_name] = weight
    progress(f"{simulation}: sampling background ray environments")
    background_u, background_n = native_background_sections(dataset, steps)
    variables["background_zonal_wind"] = background_u
    variables["buoyancy_frequency"] = background_n
    progress(f"{simulation}: sampling lifecycle cloud context")
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
            "height_m": HEIGHT_M,
            "cloud_class": np.asarray(CLOUD_CLASS_NAMES, dtype=str),
            "cloud_relative_latitude": CLOUD_RELATIVE_LATITUDE_DEGREES,
            "cloud_relative_longitude": CLOUD_RELATIVE_LONGITUDE_DEGREES,
        },
        attrs={
            "cache_version": SECTION_CACHE_VERSION,
            "source_simulation": simulation,
            "test_subset": int(test),
            "event_weighting": "one equal weight per available LPO event",
            "default_longitude_half_width_degrees": (
                DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
            ),
            "longitude_half_width_sensitivity_degrees": json.dumps(
                LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES
            ),
            "vertical_latitude_projection": (
                "+/-10 degrees around each saved LPO centroid latitude"
            ),
            "vertical_coordinate": (
                "native MPAS levels with sampled pressure in hPa"
            ),
            "composite_field_definition": "absolute sampled model values",
            "property_sampling_definition": PROPERTY_SAMPLING_DEFINITION,
            "local_precipitation_threshold_mm_day": LIFECYCLE_RAIN_THRESHOLD_MM_DAY,
            "cloud_sampling_region": (
                "broad event-centred context; never truncated by the LPO "
                "envelope; unavailable saved context is NaN"
            ),
            "lpo_envelope_role": (
                "sampling mask for unfiltered properties; contour for broad context"
            ),
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
        "unfiltered_cloud_fraction": "1",
        "unfiltered_static_stability": "s-2",
        "native_pressure_hpa": "hPa",
        "background_zonal_wind": "m s-1",
        "buoyancy_frequency": "s-1",
    }
    for name, unit in units.items():
        output[name].attrs["units"] = unit
    for name in (*SURFACE_VARIABLES, *VERTICAL_VARIABLES, "native_pressure_hpa"):
        output[name].attrs["sampling_region"] = (
            "complete LPO envelope" if name == "unfiltered_cloud_fraction"
            else "LPO envelope with concurrent precipitation > 1 mm/day"
        )
    output.unfiltered_relative_humidity.attrs[
        "source_variable"
    ] = "relhum"
    output.unfiltered_static_stability.attrs[
        "nonpositive_values"
    ] = "retained"
    output.unfiltered_upward_pressure_velocity.attrs[
        "sign_convention"
    ] = "positive values represent ascent (-omega = rho g w)"
    peak_sections = build_peak_event_sections(
        dataset, precipitation, classification, steps
    )
    output = xr.merge([output, peak_sections], compat="no_conflicts")
    output.attrs.update(peak_sections.attrs)
    return output


def longitude_window_mean(
    field: xr.DataArray,
    half_width_degrees: float,
    *,
    sample_weights: xr.DataArray | None = None,
    return_weights: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
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
    if sample_weights is not None:
        weights = weights * sample_weights.fillna(0.0)
    mean = field.weighted(weights).mean(
        "relative_longitude", skipna=True
    )
    if return_weights:
        return mean, weights.where(field.notnull(), 0.0).sum("relative_longitude")
    return mean


def _section_window_mean(
    dataset: xr.Dataset, variable: str, half_width_degrees: float
) -> xr.DataArray:
    """Combine native latitude and longitude weights before event averaging."""

    field = dataset[variable]
    weight_name = field.attrs.get("longitude_sample_weight")
    return longitude_window_mean(
        field, half_width_degrees,
        sample_weights=dataset[weight_name] if weight_name else None,
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
    available_events = list(dict.fromkeys(event_ids))
    if not available_events:
        raise RuntimeError(
            f"No event sequence is available for {variable!r}"
        )
    trailing_dims = tuple(dim for dim in data.dims if dim != "event_step")
    trailing_shape = tuple(data.sizes[dim] for dim in trailing_dims)
    cube = np.full(
        (len(available_events), len(lags), *trailing_shape),
        np.nan,
        dtype=float,
    )
    values = np.asarray(data, dtype=float)
    for event_index, event_id in enumerate(available_events):
        event_positions = np.flatnonzero(event_ids == event_id)
        lookup = {
            int(lag_values[position]): int(position)
            for position in event_positions
        }
        for lag_index, lag in enumerate(lags):
            if int(lag) in lookup:
                cube[event_index, lag_index] = values[lookup[int(lag)]]
    coords = {
        "event": np.asarray(available_events, dtype=str),
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
    """Mean and percentile interval from independent-event resamples."""

    if "event" not in cube.dims:
        raise ValueError(
            "Independent-event statistics require an event dimension"
        )
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
            "sampling_unit": "LPO event",
        },
    )


def _finite_axis0_mean(values: np.ndarray) -> np.ndarray:
    """Return a finite mean along axis zero without empty-slice warnings."""

    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    return np.divide(
        np.where(finite, values, 0.0).sum(axis=0),
        count,
        out=np.full(values.shape[1:], np.nan),
        where=count > 0,
    )


def _finite_axis0_percentiles(
    values: np.ndarray,
    quantiles: Sequence[float],
) -> np.ndarray:
    """Return axis-zero percentiles without all-NaN warnings."""

    values = np.asarray(values, dtype=float)
    flat = values.reshape(values.shape[0], -1)
    output = np.full((len(quantiles), flat.shape[1]), np.nan)
    valid = np.isfinite(flat).any(axis=0)
    if valid.any():
        output[:, valid] = np.nanpercentile(
            flat[:, valid], quantiles, axis=0
        )
    return output.reshape((len(quantiles), *values.shape[1:]))


def _benjamini_hochberg_mask(
    p_values: np.ndarray,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    """Return a Benjamini--Hochberg false-discovery-rate mask."""

    flat = np.asarray(p_values, dtype=float).ravel()
    finite_positions = np.flatnonzero(np.isfinite(flat))
    output = np.zeros(flat.shape, dtype=bool)
    if not finite_positions.size:
        return output.reshape(np.shape(p_values))
    finite = flat[finite_positions]
    order = np.argsort(finite)
    ordered = finite[order]
    thresholds = alpha * np.arange(1, len(ordered) + 1) / len(ordered)
    accepted = np.flatnonzero(ordered <= thresholds)
    if accepted.size:
        cutoff = ordered[accepted[-1]]
        output[finite_positions] = finite <= cutoff
    return output.reshape(np.shape(p_values))


def independent_event_contrast(
    experiment: xr.DataArray,
    control: xr.DataArray,
    *,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    chunk_cells: int = 20_000,
) -> xr.Dataset:
    """Return an independent-sample event contrast and robustness masks."""

    if "event" not in experiment.dims or "event" not in control.dims:
        raise ValueError("Independent contrasts require event dimensions")
    dimensions = tuple(dim for dim in experiment.dims if dim != "event")
    if tuple(dim for dim in control.dims if dim != "event") != dimensions:
        raise ValueError("Independent contrast dimensions do not match")
    experiment, control = xr.align(
        experiment, control, join="exact", exclude="event"
    )
    experiment = experiment.transpose("event", *dimensions)
    control = control.transpose("event", *dimensions)
    experiment_values = np.asarray(experiment, dtype=float)
    control_values = np.asarray(control, dtype=float)
    shape = experiment_values.shape[1:]
    experiment_flat = experiment_values.reshape(
        experiment_values.shape[0], -1
    )
    control_flat = control_values.reshape(control_values.shape[0], -1)
    difference = (
        _finite_axis0_mean(experiment_values)
        - _finite_axis0_mean(control_values)
    )
    low = np.full(experiment_flat.shape[1], np.nan)
    high = np.full_like(low, np.nan)
    p_value = np.full_like(low, np.nan)
    rng = np.random.default_rng(seed)
    experiment_draws = rng.integers(
        0, experiment_values.shape[0],
        size=(n_bootstrap, experiment_values.shape[0]),
    )
    control_draws = rng.integers(
        0, control_values.shape[0],
        size=(n_bootstrap, control_values.shape[0]),
    )
    for start in range(0, experiment_flat.shape[1], chunk_cells):
        stop = min(start + chunk_cells, experiment_flat.shape[1])
        bootstrap = np.full((n_bootstrap, stop - start), np.nan, dtype=np.float32)
        experiment_chunk = experiment_flat[:, start:stop]
        control_chunk = control_flat[:, start:stop]
        for draw_index in range(n_bootstrap):
            bootstrap[draw_index] = (
                _finite_axis0_mean(
                    experiment_chunk[experiment_draws[draw_index]]
                )
                - _finite_axis0_mean(
                    control_chunk[control_draws[draw_index]]
                )
            )
        valid = np.isfinite(bootstrap).any(axis=0)
        if valid.any():
            chunk_low = low[start:stop]
            chunk_high = high[start:stop]
            chunk_p_value = p_value[start:stop]
            chunk_low[valid] = np.nanpercentile(
                bootstrap[:, valid], 2.5, axis=0
            )
            chunk_high[valid] = np.nanpercentile(
                bootstrap[:, valid], 97.5, axis=0
            )
            valid_bootstrap = np.isfinite(bootstrap[:, valid])
            denominator = valid_bootstrap.sum(axis=0)
            nonpositive = np.divide(
                ((bootstrap[:, valid] <= 0.0) & valid_bootstrap).sum(axis=0),
                denominator,
                out=np.full(denominator.shape, np.nan),
                where=denominator > 0,
            )
            nonnegative = np.divide(
                ((bootstrap[:, valid] >= 0.0) & valid_bootstrap).sum(axis=0),
                denominator,
                out=np.full(denominator.shape, np.nan),
                where=denominator > 0,
            )
            chunk_p_value[valid] = np.minimum(
                1.0, 2.0 * np.minimum(nonpositive, nonnegative)
            )
    low = low.reshape(shape)
    high = high.reshape(shape)
    p_value = p_value.reshape(shape)
    pointwise = (low > 0.0) | (high < 0.0)
    fdr = _benjamini_hochberg_mask(p_value)
    coords = {dimension: experiment[dimension] for dimension in dimensions}
    return xr.Dataset(
        {
            "mean": (dimensions, difference),
            "ci_low": (dimensions, low),
            "ci_high": (dimensions, high),
            "pointwise_significant": (dimensions, pointwise),
            "fdr_significant": (dimensions, fdr),
            "bootstrap_p_value": (dimensions, p_value),
        },
        coords=coords,
        attrs={
            "bootstrap_resamples": int(n_bootstrap),
            "sampling_unit": "independent LPO event",
            "experiment_event_count": int(experiment.sizes["event"]),
            "control_event_count": int(control.sizes["event"]),
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
        windowed = _section_window_mean(dataset, variable, half_width)
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


def _event_latitude_mean_for_windows(
    dataset: xr.Dataset,
    variable: str,
    *,
    lags: Sequence[int],
    latitude_bounds: tuple[float, float],
) -> xr.DataArray:
    """Reduce relative latitude per event before the equal-event mean."""
    means = []
    lower, upper = sorted(map(float, latitude_bounds))
    for half_width in LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES:
        windowed, longitude_weight = longitude_window_mean(
            dataset[variable], half_width, return_weights=True
        )
        latitude = dataset.precip_weighted_centroid_lat + windowed.relative_latitude
        weights = np.cos(np.deg2rad(latitude)).clip(min=0.0) * longitude_weight
        windowed = windowed.sel(relative_latitude=slice(lower, upper))
        per_step = windowed.weighted(
            weights.sel(relative_latitude=slice(lower, upper)).fillna(0.0)
        ).mean("relative_latitude", skipna=True)
        temporary = xr.Dataset(
            {variable: per_step},
            coords={
                "event_id": dataset.event_id,
                "lag_hours": dataset.lag_hours,
            },
        )
        cube = _event_cube(temporary, variable, lags=lags)
        means.append(cube.mean("event", skipna=True))
    return xr.concat(
        means,
        dim=xr.IndexVariable(
            "longitude_half_width_degrees",
            np.asarray(LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES),
        ),
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
    strong_raw = _event_cube(
        wave_sections, "wave_strong_amplitude", lags=lags
    )
    strong = strong_raw.fillna(0.0).astype(bool)
    has_random_reference = {
        "random_wave_phase_number",
        "random_wave_strong_amplitude",
    }.issubset(wave_sections.data_vars)
    if has_random_reference:
        random_phase = _event_cube(
            wave_sections, "random_wave_phase_number", lags=lags
        )
        random_strong = _event_cube(
            wave_sections, "random_wave_strong_amplitude", lags=lags
        ).fillna(0.0).astype(bool)
        random_valid = np.isfinite(random_phase)
        random_strong_valid = random_strong & random_valid
    valid_phase = np.isfinite(phase)
    phase_numbers = np.arange(1, 9, dtype=int)
    occurrence = xr.concat(
        [
            (phase == value).sum("event")
            / valid_phase.sum("event").where(valid_phase.sum("event") > 0)
            for value in phase_numbers
        ],
        dim=xr.IndexVariable("wave_phase", phase_numbers),
    ).transpose("lag_hours", "wave_band", "wave_phase")
    strong_valid = strong & valid_phase
    valid_event_count = valid_phase.sum("event")
    strong_fraction = (
        strong_valid.sum("event")
        / valid_event_count.where(valid_event_count > 0)
    )
    strong_occurrence = xr.concat(
        [
            ((phase == value) & strong_valid).sum("event")
            / strong_valid.sum("event").where(strong_valid.sum("event") > 0)
            for value in phase_numbers
        ],
        dim=xr.IndexVariable("wave_phase", phase_numbers),
    ).transpose("lag_hours", "wave_band", "wave_phase")
    if has_random_reference:
        random_occurrence = xr.concat(
            [
                ((random_phase == value) & random_strong_valid).sum("event")
                / random_strong_valid.sum("event").where(
                    random_strong_valid.sum("event") > 0
                )
                for value in phase_numbers
            ],
            dim=xr.IndexVariable("wave_phase", phase_numbers),
        ).transpose("lag_hours", "wave_band", "wave_phase")
    radians = np.deg2rad(angle)
    weight = strong_valid.astype(float)
    denominator = weight.sum("event").where(weight.sum("event") > 0)
    cosine = (np.cos(radians) * weight).sum("event", skipna=True) / denominator
    sine = (np.sin(radians) * weight).sum("event", skipna=True) / denominator
    concentration = np.hypot(cosine, sine)
    mean_angle = np.mod(np.rad2deg(np.arctan2(sine, cosine)), 360.0)
    amplitude_values = np.asarray(amplitude, dtype=float)
    amplitude_percentiles = _finite_axis0_percentiles(
        amplitude_values, [25.0, 50.0, 75.0]
    )
    amplitude_median = amplitude_percentiles[1]
    amplitude_quartiles = amplitude_percentiles[[0, 2]]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_occurrence = np.full(
        (
            BOOTSTRAP_RESAMPLES,
            len(lags),
            wave_sections.sizes["wave_band"],
            len(phase_numbers),
        ),
        np.nan,
        dtype=np.float32,
    )
    bootstrap_concentration = np.full(
        (
            BOOTSTRAP_RESAMPLES,
            len(lags),
            wave_sections.sizes["wave_band"],
        ),
        np.nan,
        dtype=np.float32,
    )
    bootstrap_reference_difference = (
        np.full_like(bootstrap_occurrence, np.nan)
        if has_random_reference
        else None
    )
    phase_values = np.asarray(phase, dtype=float)
    angle_values = np.asarray(angle, dtype=float)
    strong_values = np.asarray(strong_valid, dtype=bool)
    if has_random_reference:
        random_phase_values = np.asarray(random_phase, dtype=float)
        random_strong_values = np.asarray(
            random_strong_valid, dtype=bool
        )
    for draw_index in range(BOOTSTRAP_RESAMPLES):
        draw = rng.integers(0, phase.sizes["event"], phase.sizes["event"])
        drawn_phase = phase_values[draw]
        drawn_angle = np.deg2rad(angle_values[draw])
        drawn_strong = strong_values[draw]
        if has_random_reference:
            drawn_random_phase = random_phase_values[draw]
            drawn_random_strong = random_strong_values[draw]
            random_strong_count = drawn_random_strong.sum(axis=0)
        strong_count = drawn_strong.sum(axis=0)
        for phase_index, phase_number in enumerate(phase_numbers):
            actual_probability = np.divide(
                ((drawn_phase == phase_number) & drawn_strong).sum(axis=0),
                strong_count,
                out=np.full(strong_count.shape, np.nan),
                where=strong_count > 0,
            )
            bootstrap_occurrence[draw_index, ..., phase_index] = (
                actual_probability
            )
            if has_random_reference:
                random_probability = np.divide(
                    (
                        (drawn_random_phase == phase_number)
                        & drawn_random_strong
                    ).sum(axis=0),
                    random_strong_count,
                    out=np.full(random_strong_count.shape, np.nan),
                    where=random_strong_count > 0,
                )
                bootstrap_reference_difference[
                    draw_index, ..., phase_index
                ] = actual_probability - random_probability
        cosine_mean = np.divide(
            np.where(drawn_strong, np.cos(drawn_angle), 0.0).sum(axis=0),
            strong_count,
            out=np.full(strong_count.shape, np.nan),
            where=strong_count > 0,
        )
        sine_mean = np.divide(
            np.where(drawn_strong, np.sin(drawn_angle), 0.0).sum(axis=0),
            strong_count,
            out=np.full(strong_count.shape, np.nan),
            where=strong_count > 0,
        )
        bootstrap_concentration[draw_index] = np.hypot(
            cosine_mean, sine_mean
        )
    occurrence_interval = _finite_axis0_percentiles(
        bootstrap_occurrence, [2.5, 97.5]
    )
    concentration_interval = _finite_axis0_percentiles(
        bootstrap_concentration, [2.5, 97.5]
    )
    output = xr.Dataset(
        {
            "wave_phase_occurrence": occurrence,
            "strong_wave_phase_occurrence": strong_occurrence,
            "wave_phase_circular_concentration": concentration,
            "wave_mean_phase_angle_degrees": mean_angle,
            "wave_normalized_amplitude": (
                ("lag_hours", "wave_band"), amplitude_median
            ),
            "wave_normalized_amplitude_q25": (
                ("lag_hours", "wave_band"), amplitude_quartiles[0]
            ),
            "wave_normalized_amplitude_q75": (
                ("lag_hours", "wave_band"), amplitude_quartiles[1]
            ),
            "strong_wave_fraction": strong_fraction,
            "strong_wave_event_count": strong_valid.sum("event"),
            "all_wave_event_count": valid_event_count,
            "strong_wave_phase_occurrence_ci_low": (
                ("lag_hours", "wave_band", "wave_phase"),
                occurrence_interval[0],
            ),
            "strong_wave_phase_occurrence_ci_high": (
                ("lag_hours", "wave_band", "wave_phase"),
                occurrence_interval[1],
            ),
            "wave_phase_circular_concentration_ci_low": (
                ("lag_hours", "wave_band"), concentration_interval[0]
            ),
            "wave_phase_circular_concentration_ci_high": (
                ("lag_hours", "wave_band"), concentration_interval[1]
            ),
        },
        attrs={
            "phase_role": (
                "diagnostic of filtered-wave alignment only; LPO lag remains "
                "the compositing coordinate and unfiltered composites have "
                "no phase conditioning"
            ),
            "strong_wave_threshold": "normalized phase-space amplitude > 1",
            "amplitude_summary": "event median and interquartile range",
            "phase_bootstrap": (
                f"{BOOTSTRAP_RESAMPLES} independent-event resamples"
            ),
        },
    )
    if has_random_reference:
        reference_interval = _finite_axis0_percentiles(
            bootstrap_reference_difference, [2.5, 97.5]
        )
        output["random_strong_wave_phase_occurrence"] = random_occurrence
        output["random_strong_wave_event_count"] = (
            random_strong_valid.sum("event")
        )
        output["strong_wave_phase_excess_over_random"] = (
            strong_occurrence - random_occurrence
        )
        output["strong_wave_phase_excess_over_random_ci_low"] = (
            ("lag_hours", "wave_band", "wave_phase"),
            reference_interval[0],
        )
        output["strong_wave_phase_excess_over_random_ci_high"] = (
            ("lag_hours", "wave_band", "wave_phase"),
            reference_interval[1],
        )
        output["strong_wave_phase_excess_over_random_significant"] = (
            (
                output.strong_wave_phase_excess_over_random_ci_low > 0.0
            )
            | (
                output.strong_wave_phase_excess_over_random_ci_high < 0.0
            )
        )
        phase_excess = strong_occurrence - random_occurrence
        alignment_strength = (
            0.5 * np.abs(phase_excess).sum("wave_phase")
        ).where(np.isfinite(phase_excess).any("wave_phase"))
        bootstrap_alignment_strength = 0.5 * np.abs(
            bootstrap_reference_difference
        ).sum(axis=-1)
        no_bootstrap_phase = ~np.isfinite(
            bootstrap_reference_difference
        ).any(axis=-1)
        bootstrap_alignment_strength[no_bootstrap_phase] = np.nan
        alignment_bootstrap_centre = _finite_axis0_mean(
            bootstrap_alignment_strength
        )
        alignment_residual_interval = _finite_axis0_percentiles(
            bootstrap_alignment_strength - alignment_bootstrap_centre,
            [2.5, 97.5],
        )
        alignment_strength_interval = np.clip(
            np.asarray(alignment_strength)[None, ...]
            + alignment_residual_interval,
            0.0,
            1.0,
        )
        output["wave_phase_alignment_strength"] = alignment_strength
        output["wave_phase_alignment_strength_ci_low"] = (
            ("lag_hours", "wave_band"), alignment_strength_interval[0]
        )
        output["wave_phase_alignment_strength_ci_high"] = (
            ("lag_hours", "wave_band"), alignment_strength_interval[1]
        )
        lag_values = np.asarray(lags, dtype=int)
        wave_names = wave_sections.wave_band.values.astype(str)
        missing_windows = sorted(
            set(wave_names) - set(PHASE_ALIGNMENT_WINDOWS_HOURS)
        )
        if missing_windows:
            raise RuntimeError(
                "Phase-alignment summary lacks lag windows for "
                f"{missing_windows}"
            )
        summary_excess = np.full(
            (len(wave_names), len(phase_numbers)), np.nan
        )
        bootstrap_summary_excess = np.full(
            (BOOTSTRAP_RESAMPLES, len(wave_names), len(phase_numbers)),
            np.nan,
            dtype=np.float32,
        )
        summary_alignment_strength = np.full(len(wave_names), np.nan)
        bootstrap_summary_alignment_strength = np.full(
            (BOOTSTRAP_RESAMPLES, len(wave_names)),
            np.nan,
            dtype=np.float32,
        )
        summary_phase_locking_strength = np.full(len(wave_names), np.nan)
        bootstrap_summary_phase_locking_strength = np.full(
            (BOOTSTRAP_RESAMPLES, len(wave_names)),
            np.nan,
            dtype=np.float32,
        )
        phase_vectors = np.exp(
            2.0j * np.pi * (phase_numbers - 1.0) / len(phase_numbers)
        )
        strong_values = np.asarray(strong_occurrence, dtype=float)
        random_values = np.asarray(random_occurrence, dtype=float)
        bootstrap_random_occurrence = (
            bootstrap_occurrence - bootstrap_reference_difference
        )
        excess_values = np.asarray(
            strong_occurrence - random_occurrence, dtype=float
        )
        for wave_index, wave_name in enumerate(wave_names):
            window = PHASE_ALIGNMENT_WINDOWS_HOURS[wave_name]
            missing_lags = sorted(set(window) - set(lag_values))
            if missing_lags:
                raise RuntimeError(
                    f"{wave_name} phase-alignment window lacks lags "
                    f"{missing_lags}"
                )
            lag_indices = np.flatnonzero(np.isin(lag_values, window))
            summary_excess[wave_index] = _finite_axis0_mean(
                excess_values[lag_indices, wave_index]
            )
            bootstrap_summary_excess[:, wave_index] = _finite_axis0_mean(
                bootstrap_reference_difference[
                    :, lag_indices, wave_index, :
                ].transpose(1, 0, 2)
            )
            summary_alignment_strength[wave_index] = 0.5 * np.sum(
                np.abs(summary_excess[wave_index])
            )
            bootstrap_summary_alignment_strength[:, wave_index] = (
                0.5 * np.sum(
                    np.abs(bootstrap_summary_excess[:, wave_index]),
                    axis=-1,
                )
            )
            summary_strong = _finite_axis0_mean(
                strong_values[lag_indices, wave_index]
            )
            summary_random = _finite_axis0_mean(
                random_values[lag_indices, wave_index]
            )
            if (
                np.isfinite(summary_strong).all()
                and np.isfinite(summary_random).all()
            ):
                summary_phase_locking_strength[wave_index] = (
                    np.abs(np.sum(summary_strong * phase_vectors))
                    - np.abs(np.sum(summary_random * phase_vectors))
                )
            bootstrap_summary_strong = _finite_axis0_mean(
                bootstrap_occurrence[
                    :, lag_indices, wave_index, :
                ].transpose(1, 0, 2)
            )
            bootstrap_summary_random = _finite_axis0_mean(
                bootstrap_random_occurrence[
                    :, lag_indices, wave_index, :
                ].transpose(1, 0, 2)
            )
            valid_bootstrap = (
                np.isfinite(bootstrap_summary_strong).all(axis=-1)
                & np.isfinite(bootstrap_summary_random).all(axis=-1)
            )
            bootstrap_summary_phase_locking_strength[
                valid_bootstrap, wave_index
            ] = (
                np.abs(
                    np.sum(
                        bootstrap_summary_strong[valid_bootstrap]
                        * phase_vectors,
                        axis=-1,
                    )
                )
                - np.abs(
                    np.sum(
                        bootstrap_summary_random[valid_bootstrap]
                        * phase_vectors,
                        axis=-1,
                    )
                )
            )
        summary_interval = _finite_axis0_percentiles(
            bootstrap_summary_excess, [2.5, 97.5]
        )
        summary_alignment_bootstrap_centre = _finite_axis0_mean(
            bootstrap_summary_alignment_strength
        )
        summary_alignment_residual_interval = _finite_axis0_percentiles(
            bootstrap_summary_alignment_strength
            - summary_alignment_bootstrap_centre,
            [2.5, 97.5],
        )
        summary_alignment_strength_interval = np.clip(
            summary_alignment_strength[None, ...]
            + summary_alignment_residual_interval,
            0.0,
            1.0,
        )
        phase_locking_bootstrap_centre = _finite_axis0_mean(
            bootstrap_summary_phase_locking_strength
        )
        phase_locking_residual_interval = _finite_axis0_percentiles(
            bootstrap_summary_phase_locking_strength
            - phase_locking_bootstrap_centre,
            [2.5, 97.5],
        )
        phase_locking_strength_interval = np.clip(
            summary_phase_locking_strength[None, ...]
            + phase_locking_residual_interval,
            -1.0,
            1.0,
        )
        output["phase_window_excess_over_random"] = (
            ("wave_band", "wave_phase"), summary_excess
        )
        output["phase_window_excess_over_random_ci_low"] = (
            ("wave_band", "wave_phase"), summary_interval[0]
        )
        output["phase_window_excess_over_random_ci_high"] = (
            ("wave_band", "wave_phase"), summary_interval[1]
        )
        output["phase_window_alignment_strength"] = (
            ("wave_band",), summary_alignment_strength
        )
        output["phase_window_alignment_strength_ci_low"] = (
            ("wave_band",), summary_alignment_strength_interval[0]
        )
        output["phase_window_alignment_strength_ci_high"] = (
            ("wave_band",), summary_alignment_strength_interval[1]
        )
        output["phase_window_phase_locking_strength"] = (
            ("wave_band",), summary_phase_locking_strength
        )
        output["phase_window_phase_locking_strength_ci_low"] = (
            ("wave_band",), phase_locking_strength_interval[0]
        )
        output["phase_window_phase_locking_strength_ci_high"] = (
            ("wave_band",), phase_locking_strength_interval[1]
        )
        output["phase_window_alignment_strength_bootstrap"] = (
            ("phase_bootstrap_sample", "wave_band"),
            bootstrap_summary_alignment_strength,
        )
        output["phase_window_phase_locking_strength_bootstrap"] = (
            ("phase_bootstrap_sample", "wave_band"),
            bootstrap_summary_phase_locking_strength,
        )
        output.attrs["random_phase_reference"] = (
            "matched tropical trajectories moving at the simulation mean "
            "LPO zonal speed"
        )
        output.attrs["random_phase_display"] = (
            "the same strong-wave phase-versus-lag probability as the LPO "
            "panel, evaluated on matched random trajectories"
        )
        output.attrs["phase_alignment_windows_hours"] = json.dumps(
            PHASE_ALIGNMENT_WINDOWS_HOURS
        )
        output.attrs["phase_window_average"] = (
            "equal-weight mean of lag-specific strong-wave phase probability "
            "minus its matched-random probability"
        )
        output.attrs["phase_alignment_strength"] = (
            "total variation distance between the strong-wave LPO and "
            "matched-random eight-phase probability distributions; equals "
            "half the sum of absolute phase-probability differences"
        )
        output.attrs["phase_locking_strength"] = (
            "circular resultant length of the temporally averaged strong-wave "
            "LPO phase distribution minus the corresponding matched-random "
            "resultant length; positive values indicate additional phase "
            "concentration at LPOs"
        )
        output.attrs["phase_window_bootstrap"] = (
            f"{BOOTSTRAP_RESAMPLES} complete-LPO-trajectory resamples; phase "
            "probabilities, alignment strengths, phase-locking strengths, and "
            "temporal means are recalculated in every draw; nonlinear "
            "strength intervals use bias-centred bootstrap deviations"
        )
    return output


def _independent_bootstrap_metric_response(
    experiment_point: xr.DataArray,
    control_point: xr.DataArray,
    experiment_bootstrap: xr.DataArray,
    control_bootstrap: xr.DataArray,
    *,
    seed: int,
) -> xr.Dataset:
    """Contrast nonlinear metrics using independently paired bootstrap draws."""

    experiment_point, control_point = xr.align(
        experiment_point, control_point, join="exact"
    )
    dimensions = experiment_point.dims
    bootstrap_dimension = "phase_bootstrap_sample"
    expected_dimensions = (bootstrap_dimension, *dimensions)
    experiment_bootstrap = experiment_bootstrap.transpose(
        *expected_dimensions
    )
    control_bootstrap = control_bootstrap.transpose(*expected_dimensions)
    if experiment_bootstrap.shape != control_bootstrap.shape:
        raise ValueError("Phase-metric bootstrap shapes do not match")
    sample_count = experiment_bootstrap.sizes[bootstrap_dimension]
    rng = np.random.default_rng(seed)
    experiment_order = rng.permutation(sample_count)
    control_order = rng.permutation(sample_count)
    bootstrap_difference = (
        np.asarray(experiment_bootstrap, dtype=float)[experiment_order]
        - np.asarray(control_bootstrap, dtype=float)[control_order]
    )
    point = np.asarray(experiment_point - control_point, dtype=float)
    interval = _finite_axis0_percentiles(
        bootstrap_difference, [2.5, 97.5]
    )
    valid = np.isfinite(bootstrap_difference)
    denominator = valid.sum(axis=0)
    nonpositive = np.divide(
        ((bootstrap_difference <= 0.0) & valid).sum(axis=0),
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 0,
    )
    nonnegative = np.divide(
        ((bootstrap_difference >= 0.0) & valid).sum(axis=0),
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 0,
    )
    p_value = np.minimum(
        1.0, 2.0 * np.minimum(nonpositive, nonnegative)
    )
    pointwise = (interval[0] > 0.0) | (interval[1] < 0.0)
    coordinates = {
        dimension: experiment_point[dimension] for dimension in dimensions
    }
    return xr.Dataset(
        {
            "mean": (dimensions, point),
            "ci_low": (dimensions, interval[0]),
            "ci_high": (dimensions, interval[1]),
            "pointwise_significant": (dimensions, pointwise),
            "fdr_significant": (
                dimensions, _benjamini_hochberg_mask(p_value)
            ),
            "bootstrap_p_value": (dimensions, p_value),
        },
        coords=coordinates,
        attrs={
            "bootstrap_resamples": int(sample_count),
            "sampling_unit": "independent complete LPO trajectory",
            "interval": "two-sided 95% percentile interval",
        },
    )


def _phase_pressure_event_cube(
    wave_sections: xr.Dataset,
    variable: str,
) -> xr.DataArray:
    """Pool strong-wave samples with equal total event weight per phase."""

    field = longitude_window_mean(
        wave_sections[variable], DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
    )
    vertical_dimension = (
        "native_level" if "native_level" in field.dims else "pressure_hpa"
    )
    if "wave_band" not in field.dims:
        field = field.expand_dims(wave_band=wave_sections.wave_band)
    field = field.transpose(
        "event_step", vertical_dimension, "wave_band"
    )
    event_ids = np.asarray(wave_sections.event_id).astype(str)
    unique_events = np.asarray(list(dict.fromkeys(event_ids)), dtype=str)
    phase = np.asarray(wave_sections.wave_phase_number, dtype=float)
    strong = np.asarray(wave_sections.wave_strong_amplitude, dtype=bool)
    values = np.asarray(field, dtype=float)
    phase_numbers = np.arange(1, 9, dtype=int)
    output = np.full(
        (
            len(unique_events),
            wave_sections.sizes["wave_band"],
            len(phase_numbers),
            wave_sections.sizes[vertical_dimension],
        ),
        np.nan,
    )
    for event_index, event_id in enumerate(unique_events):
        event_positions = np.flatnonzero(event_ids == event_id)
        for wave_index in range(wave_sections.sizes["wave_band"]):
            for phase_index, phase_number in enumerate(phase_numbers):
                positions = event_positions[
                    strong[event_positions, wave_index]
                    & (
                        phase[event_positions, wave_index]
                        == phase_number
                    )
                ]
                if positions.size:
                    output[event_index, wave_index, phase_index] = (
                        _finite_axis0_mean(values[positions, :, wave_index])
                    )
    return xr.DataArray(
        output,
        dims=("event", "wave_band", "wave_phase", vertical_dimension),
        coords={
            "event": unique_events,
            "wave_band": wave_sections.wave_band,
            "wave_phase": phase_numbers,
            vertical_dimension: wave_sections[vertical_dimension],
        },
    )


def _phase_pressure_composites(wave_sections: xr.Dataset) -> xr.Dataset:
    output = xr.Dataset()
    for variable_index, variable in enumerate(
        FILTERED_VERTICAL_VARIABLES
    ):
        cube = _phase_pressure_event_cube(wave_sections, variable)
        statistics = event_equal_statistics(
            cube, seed=BOOTSTRAP_SEED + 300 + variable_index
        )
        stem = f"phase_pressure_{variable}"
        output[stem] = statistics["mean"]
        output[f"{stem}_ci_low"] = statistics.ci_low
        output[f"{stem}_ci_high"] = statistics.ci_high
        output[f"{stem}_event_count"] = statistics.finite_event_count
    pressure_cube = _phase_pressure_event_cube(
        wave_sections, "filtered_native_pressure_hpa"
    )
    output["phase_pressure_filtered_native_pressure_hpa"] = (
        pressure_cube.mean("event", skipna=True)
    )
    output.attrs.update(
        {
            "phase_pressure_selection": (
                "normalized filtered-precipitation amplitude greater than one"
            ),
        }
    )
    return output


def theoretical_ray_event_paths(
    event_sections: xr.Dataset,
    wave_name: str,
    *,
    kelvin_k_m1: float = 1.57e-7,
    vertical_wavenumber_m1: float = -2.0 * np.pi / 10_000.0,
    background_profile: xr.Dataset | None = None,
) -> xr.DataArray:
    """Trace event rays from 200 hPa in the moving LPO reference frame."""

    if wave_name not in {"Kelvin", "MRG"}:
        raise ValueError(f"No theoretical LPO ray for {wave_name}")
    height = np.asarray(event_sections.ray_height_m, dtype=float)
    if background_profile is None:
        central = event_sections.sel(
            ray_relative_longitude=slice(-5.0, 5.0)
        )
        wind = np.asarray(
            central.ray_background_zonal_wind.mean(
                "ray_relative_longitude", skipna=True
            ),
            dtype=float,
        )
        buoyancy_frequency = np.asarray(
            central.ray_buoyancy_frequency.mean(
                "ray_relative_longitude", skipna=True
            ),
            dtype=float,
        )
        pressure = np.asarray(
            central.ray_pressure_hpa.mean(
                "ray_relative_longitude", skipna=True
            ),
            dtype=float,
        )
        profile_source = "event-local U and N within 5 degrees of centroid"
    else:
        required = {
            "background_zonal_wind",
            "buoyancy_frequency",
            "background_pressure_hpa",
        }
        missing = sorted(required - set(background_profile.data_vars))
        if missing or "height_m" not in background_profile.coords:
            raise ValueError(
                "Unconditional ray background lacks "
                f"{missing or ['height_m coordinate']}"
            )
        source_height = np.asarray(
            background_profile.height_m, dtype=float
        )

        def interpolate(variable: str) -> np.ndarray:
            values = np.asarray(background_profile[variable], dtype=float)
            valid = np.isfinite(source_height) & np.isfinite(values)
            if valid.sum() < 2:
                return np.full(height.shape, np.nan)
            return np.interp(
                height,
                source_height[valid],
                values[valid],
                left=np.nan,
                right=np.nan,
            )

        background_wind = interpolate("background_zonal_wind")
        background_n = interpolate("buoyancy_frequency")
        background_pressure = interpolate("background_pressure_hpa")
        event_count = event_sections.sizes["peak_event"]
        wind = np.broadcast_to(background_wind, (event_count, len(height)))
        buoyancy_frequency = np.broadcast_to(
            background_n, (event_count, len(height))
        )
        pressure = np.broadcast_to(
            background_pressure, (event_count, len(height))
        )
        profile_source = "unconditional deep-tropical U and N from wave Figure 13"
    result = np.full(
        (
            event_sections.sizes["peak_event"],
            len(RAY_LAUNCH_LONGITUDE_DEGREES),
            len(height),
        ),
        np.nan,
        dtype=np.float32,
    )
    for event_index in range(event_sections.sizes["peak_event"]):
        u = wind[event_index]
        n = buoyancy_frequency[event_index]
        p = pressure[event_index]
        speed = float(event_sections.lpo_zonal_speed_48h_m_s[event_index])
        latitude = float(event_sections.geometric_centroid_lat[event_index])
        valid = (
            np.isfinite(height) & np.isfinite(u) & np.isfinite(n)
            & np.isfinite(p) & (p > 0.0)
        )
        if valid.sum() < 3 or not np.isfinite(speed):
            continue
        start = int(np.nanargmin(np.where(valid, np.abs(p - 200.0), np.nan)))
        trace_indices = np.flatnonzero(
            valid & (height >= height[start]) & (p >= 50.0)
        )
        if len(trace_indices) < 2:
            continue
        if wave_name == "Kelvin":
            velocity = kelvin_group_velocities(
                u[trace_indices],
                n[trace_indices],
                zonal_wavenumber_m1=kelvin_k_m1,
                vertical_wavenumber_m1=vertical_wavenumber_m1,
            )
        else:
            velocity = mrg_group_velocities(
                u[trace_indices],
                n[trace_indices],
                zonal_wavenumber_m1=(
                    MRG_ZONAL_WAVENUMBER / EARTH_RADIUS_M
                ),
                vertical_wavenumber_m1=(
                    -2.0 * np.pi / MRG_VERTICAL_WAVELENGTH_M
                ),
            )
        relative_cgx = np.asarray(velocity["cgx_m_s"]) - speed
        cgz = np.asarray(velocity["cgz_m_s"])
        dx_dz = np.full(relative_cgx.shape, np.nan)
        np.divide(
            relative_cgx,
            cgz,
            out=dx_dz,
            where=np.isfinite(cgz) & (np.abs(cgz) > 1.0e-8),
        )
        displacement_m = np.zeros(len(trace_indices), dtype=float)
        for index in range(1, len(trace_indices)):
            if not np.isfinite(dx_dz[index - 1:index + 1]).all():
                displacement_m[index:] = np.nan
                break
            dz = height[trace_indices[index]] - height[trace_indices[index - 1]]
            displacement_m[index] = (
                displacement_m[index - 1]
                + 0.5 * (dx_dz[index - 1] + dx_dz[index]) * dz
            )
        longitude_displacement = np.rad2deg(
            displacement_m
            / (EARTH_RADIUS_M * np.cos(np.deg2rad(latitude)))
        )
        for launch_index, launch in enumerate(
            RAY_LAUNCH_LONGITUDE_DEGREES
        ):
            result[event_index, launch_index, trace_indices] = (
                launch + longitude_displacement
            )
    return xr.DataArray(
        result,
        dims=("peak_event", "ray_launch_longitude", "ray_height_m"),
        coords={
            "peak_event": event_sections.peak_event,
            "ray_launch_longitude": RAY_LAUNCH_LONGITUDE_DEGREES,
            "ray_height_m": event_sections.ray_height_m,
        },
        attrs={
            "wave_name": wave_name,
            "launch_pressure_hpa": 200.0,
            "horizontal_reference_frame": "Cgx minus geometric LPO speed",
            "lpo_speed_window_hours": "-48 to 0",
            "background_profile": profile_source,
        },
    )


def _add_peak_simulation_composites(
    output: xr.Dataset,
    event_sections: xr.Dataset,
    wave_sections: xr.Dataset,
    *,
    primary_k_m1: float,
    primary_m_m1: float,
    background_profile: xr.Dataset | None = None,
) -> None:
    """Add geometric peak composites when the current caches provide them."""

    required_event_variables = {
        "peak_event_id",
        "ray_background_zonal_wind",
        "ray_buoyancy_frequency",
        "ray_unfiltered_static_stability",
        "ray_pressure_hpa",
        "ray_high_cloud_occurrence",
        "ray_nonprecipitating_cloud_occurrence",
        "ray_lpo_precipitation",
        "lpo_zonal_speed_48h_m_s",
        "geometric_centroid_lat",
        "matsuno_zonal_wind_anomaly",
        "matsuno_meridional_wind_anomaly",
        "matsuno_relative_vorticity_anomaly",
    }
    required_wave_variables = {"peak_event_id", "ray_filtered_temperature"}
    if "peak_event_id" not in event_sections:
        return
    if "peak_event_id" not in wave_sections:
        return
    missing_event_variables = sorted(
        required_event_variables - set(event_sections.data_vars)
    )
    missing_wave_variables = sorted(
        required_wave_variables - set(wave_sections.data_vars)
    )
    if missing_event_variables or missing_wave_variables:
        raise RuntimeError(
            "Geometric peak caches lack event variables "
            f"{missing_event_variables} or wave variables "
            f"{missing_wave_variables}"
        )
    required_attributes = {
        "matsuno_eastward_event_count", "peak_event_count"
    }
    missing_attributes = sorted(
        required_attributes - set(event_sections.attrs)
    )
    if missing_attributes:
        raise RuntimeError(
            f"Geometric peak event cache lacks attributes {missing_attributes}"
        )
    event_peak_ids = event_sections.peak_event_id.values.astype(str)
    wave_peak_lookup = {
        str(event_id): index
        for index, event_id in enumerate(wave_sections.peak_event_id.values)
    }
    missing_peak_ids = [
        event_id for event_id in event_peak_ids
        if event_id not in wave_peak_lookup
    ]
    if missing_peak_ids:
        raise RuntimeError(
            "Filtered peak sections lack geometric LPO events "
            f"{missing_peak_ids[:3]}"
        )
    wave_peak = wave_sections.isel(
        peak_event=xr.DataArray(
            [wave_peak_lookup[event_id] for event_id in event_peak_ids],
            dims="peak_event",
        )
    ).assign_coords(peak_event=event_sections.peak_event)
    for variable in (
        "ray_unfiltered_static_stability",
        "ray_pressure_hpa",
        "ray_high_cloud_occurrence",
        "ray_nonprecipitating_cloud_occurrence",
        "ray_lpo_precipitation",
    ):
        source = event_sections[variable]
        output[variable] = source.mean("peak_event", skipna=True)
        output[f"{variable}_finite_event_count"] = source.notnull().sum(
            "peak_event"
        )
    output["ray_filtered_temperature"] = (
        wave_peak.ray_filtered_temperature.mean("peak_event", skipna=True)
    )
    output["ray_filtered_temperature_finite_event_count"] = (
        wave_peak.ray_filtered_temperature.notnull().sum("peak_event")
    )
    for wave_name in ("Kelvin", "MRG"):
        rays = theoretical_ray_event_paths(
            event_sections,
            wave_name,
            kelvin_k_m1=primary_k_m1,
            vertical_wavenumber_m1=primary_m_m1,
        )
        key = wave_name.lower()
        output[f"{key}_ray_relative_longitude"] = rays.mean(
            "peak_event", skipna=True
        )
        output[f"{key}_ray_finite_event_count"] = rays.notnull().sum(
            "peak_event"
        )
        if background_profile is not None:
            background_rays = theoretical_ray_event_paths(
                event_sections,
                wave_name,
                kelvin_k_m1=primary_k_m1,
                vertical_wavenumber_m1=primary_m_m1,
                background_profile=background_profile,
            )
            output[f"{key}_background_ray_relative_longitude"] = (
                background_rays.mean("peak_event", skipna=True)
            )
            output[f"{key}_background_ray_finite_event_count"] = (
                background_rays.notnull().sum("peak_event")
            )
    for variable in (
        "matsuno_zonal_wind_anomaly",
        "matsuno_meridional_wind_anomaly",
        "matsuno_relative_vorticity_anomaly",
    ):
        output[variable] = event_sections[variable].mean(
            "matsuno_event", skipna=True
        )
    output["matsuno_eastward_event_count"] = xr.DataArray(
        int(event_sections.attrs["matsuno_eastward_event_count"])
    )
    output["matsuno_total_event_count"] = xr.DataArray(
        int(event_sections.attrs["peak_event_count"])
    )


def build_simulation_composites(
    event_sections: xr.Dataset,
    wave_sections: xr.Dataset,
    *,
    primary_k_m1: float,
    primary_m_m1: float,
    test: bool,
    background_profile: xr.Dataset | None = None,
) -> xr.Dataset:
    """Reduce one simulation's raw event caches into fast plot products."""

    lags = np.asarray(LIFECYCLE_LAGS_HOURS, dtype=int)
    output = xr.Dataset()
    for variable in (
        *SURFACE_VARIABLES,
        *VERTICAL_VARIABLES,
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

    latitude_mean_sections = event_sections
    if "precip_weighted_centroid_lat" not in latitude_mean_sections:
        if not test:
            raise RuntimeError(
                "Event sections lack precip_weighted_centroid_lat"
            )
        latitude_mean_sections = latitude_mean_sections.assign(
            precip_weighted_centroid_lat=(
                "event_step",
                np.zeros(latitude_mean_sections.sizes["event_step"]),
            )
        )
    output["unfiltered_precipitation_near_centroid_latitude_mean"] = (
        _event_latitude_mean_for_windows(
            latitude_mean_sections,
            "unfiltered_precipitation",
            lags=lags,
            latitude_bounds=AVERAGING_SENSITIVITY_LATITUDE_BOUNDS,
        )
    )
    output[
        "unfiltered_precipitation_near_centroid_latitude_mean"
    ].attrs.update(
        units="mm day-1",
        latitude_bounds_degrees=json.dumps(
            AVERAGING_SENSITIVITY_LATITUDE_BOUNDS
        ),
        latitude_weighting="cosine of absolute event latitude",
    )

    if "lpo_envelope_precipitation" in event_sections:
        envelope_precipitation_cube = _event_cube(
            event_sections, "lpo_envelope_precipitation", lags=lags
        )
        envelope_precipitation_statistics = event_equal_statistics(
            envelope_precipitation_cube
        )
        output["lpo_envelope_precipitation"] = (
            envelope_precipitation_statistics["mean"]
        )
        output["lpo_envelope_precipitation_ci_low"] = (
            envelope_precipitation_statistics.ci_low
        )
        output["lpo_envelope_precipitation_ci_high"] = (
            envelope_precipitation_statistics.ci_high
        )

    pressure_mean, pressure_statistics = _mean_for_windows(
        event_sections, "native_pressure_hpa", lags=lags
    )
    output["native_pressure_hpa"] = pressure_mean
    output["native_pressure_hpa_finite_event_count"] = (
        pressure_statistics.finite_event_count
    )

    for variable in FILTERED_SURFACE_VARIABLES + FILTERED_VERTICAL_VARIABLES:
        mean, statistics = _mean_for_windows(
            wave_sections, variable, lags=lags
        )
        output[variable] = mean
        output[f"{variable}_ci_low"] = statistics.ci_low
        output[f"{variable}_ci_high"] = statistics.ci_high

    _add_peak_simulation_composites(
        output,
        event_sections,
        wave_sections,
        primary_k_m1=primary_k_m1,
        primary_m_m1=primary_m_m1,
        background_profile=background_profile,
    )

    filtered_pressure_mean, filtered_pressure_statistics = _mean_for_windows(
        wave_sections, "filtered_native_pressure_hpa", lags=lags
    )
    output["filtered_native_pressure_hpa"] = filtered_pressure_mean
    output["filtered_native_pressure_hpa_finite_event_count"] = (
        filtered_pressure_statistics.finite_event_count
    )

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
                minimum_abs_cgx_m_s=1.0e-3,
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
    background_relative_occurrence = []
    for inner, outer in CLOUD_BACKGROUND_SECTORS:
        distance = np.abs(cloud_cube.cloud_relative_longitude)
        far_field = cloud_cube.where(
            (distance >= inner) & (distance <= outer), drop=True
        ).mean("cloud_relative_longitude", skipna=True)
        excess = cloud_cube - far_field
        background_relative_occurrence.append(
            excess.mean("event", skipna=True)
        )
    background_sector = np.asarray(
        [f"{inner:g}-{outer:g}" for inner, outer in CLOUD_BACKGROUND_SECTORS],
        dtype=str,
    )
    class_excess = xr.concat(
        background_relative_occurrence,
        dim=xr.IndexVariable("background_sector", background_sector),
    )
    total_excess = class_excess.sum(
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
    modal_code = modal_code.where(
        (total_occurrence >= 0.20) & (modal_agreement >= 0.50), 0
    ).astype(np.int8)
    output["cloud_class_occurrence"] = occurrence
    output["total_high_cloud_occurrence"] = total_occurrence
    output["cloud_class_excess_occurrence"] = class_excess
    output["total_high_cloud_excess_occurrence"] = total_excess
    output["cloud_event_support_count"] = cloud_cube.notnull().sum(
        "event"
    ).min("cloud_class")
    output["modal_high_cloud_class"] = modal_code
    output["modal_high_cloud_class_agreement"] = modal_agreement
    output["lpo_envelope_occurrence"] = envelope_cube.mean(
        "event", skipna=True
    )
    output = xr.merge(
        [
            output,
            _phase_composites(wave_sections, lags=lags),
            _phase_pressure_composites(wave_sections),
        ]
    )
    output["precipitation_wave_rms"] = wave_sections.precipitation_wave_rms
    output.attrs.update(
        {
            "cache_version": COMPOSITE_CACHE_VERSION,
            "source_simulation": event_sections.attrs["source_simulation"],
            "test_subset": int(test),
            "event_weighting": "one equal weight per available LPO event",
            "bootstrap": (
                f"{BOOTSTRAP_RESAMPLES} independent-event resamples; "
                "intervals stored for the default 5-degree longitude window"
            ),
            "phase_pressure_selection": (
                "normalized filtered-precipitation amplitude greater than one"
            ),
            "default_longitude_half_width_degrees": (
                DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
            ),
            "property_sampling_definition": PROPERTY_SAMPLING_DEFINITION,
            "local_precipitation_threshold_mm_day": LIFECYCLE_RAIN_THRESHOLD_MM_DAY,
            "kelvin_reference_method": (
                "Lim-and-Son dry Kelvin reference using event-local U and N; "
                "not a generic moist-mode group velocity"
            ),
            "ray_profile_comparison": (
                "event-local U and N compared with unconditional deep-tropical "
                "U and N; both retain each event's propagation speed and latitude"
            ),
            "primary_k_m-1": float(primary_k_m1),
            "primary_m_m-1": float(primary_m_m1),
            "cloud_mode_definition": (
                "most frequent classified high-cloud regime across events; "
                "shown where total high-cloud occurrence is at least 0.20 "
                "and modal agreement is at least 0.50"
            ),
            "cloud_mode_agreement_definition": (
                "modal class occurrence divided by total classified "
                "high-cloud occurrence"
            ),
            "cloud_background_definition": (
                "same-time and same-relative-latitude mean over both "
                "60-150 degree far-field longitude sectors"
            ),
            "cloud_background_sensitivity_sectors": json.dumps(
                CLOUD_BACKGROUND_SECTORS
            ),
        }
    )
    return output


def _default_window_event_cube(
    dataset: xr.Dataset,
    variable: str,
    *,
    lags: Sequence[int],
) -> xr.DataArray:
    windowed = _section_window_mean(
        dataset, variable, DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
    )
    temporary = xr.Dataset(
        {variable: windowed},
        coords={
            "event_id": dataset.event_id,
            "lag_hours": dataset.lag_hours,
        },
    )
    return _event_cube(temporary, variable, lags=lags)


def _far_field_excess_cube(
    dataset: xr.Dataset,
    *,
    lags: Sequence[int],
    background_sector: tuple[float, float] = CLOUD_BACKGROUND_SECTORS[0],
) -> xr.DataArray:
    cube = _event_cube(dataset, "cloud_class_indicator", lags=lags)
    distance = np.abs(cube.cloud_relative_longitude)
    inner, outer = background_sector
    background = cube.where(
        (distance >= inner) & (distance <= outer), drop=True
    ).mean("cloud_relative_longitude", skipna=True)
    excess = (cube - background).sel(
        cloud_relative_longitude=slice(*CLOUD_PLOT_LONGITUDE_BOUNDS),
        cloud_relative_latitude=slice(*CLOUD_PLOT_LATITUDE_BOUNDS),
    )
    total = excess.sum("cloud_class", skipna=True, min_count=1).expand_dims(
        response_cloud_class=["total_high_cloud"]
    )
    classes = excess.rename(
        {"cloud_class": "response_cloud_class"}
    )
    return xr.concat([total, classes], dim="response_cloud_class")


def _group_velocity_event_cubes(
    dataset: xr.Dataset,
    *,
    lags: Sequence[int],
    primary_k_m1: float,
    primary_m_m1: float,
) -> dict[str, xr.DataArray]:
    wind = _default_window_event_cube(
        dataset, "background_zonal_wind", lags=lags
    )
    buoyancy_frequency = _default_window_event_cube(
        dataset, "buoyancy_frequency", lags=lags
    )
    result = kelvin_group_velocities(
        np.asarray(wind),
        np.asarray(buoyancy_frequency),
        zonal_wavenumber_m1=primary_k_m1,
        vertical_wavenumber_m1=primary_m_m1,
        minimum_abs_cgx_m_s=1.0e-3,
    )
    return {
        "cgx": xr.DataArray(
            result["cgx_m_s"], dims=wind.dims, coords=wind.coords
        ),
        "cgz": xr.DataArray(
            result["cgz_m_s"], dims=wind.dims, coords=wind.coords
        ),
        "cgz_over_cgx": xr.DataArray(
            result["ray_slope"], dims=wind.dims, coords=wind.coords
        ),
    }


def _group_velocity_sensitivity_event_cubes(
    dataset: xr.Dataset,
    *,
    lags: Sequence[int],
    primary_k_m1: float,
    primary_m_m1: float,
) -> dict[str, xr.DataArray]:
    """Return event cubes for the configured three-by-three sensitivity."""

    wind = _default_window_event_cube(
        dataset, "background_zonal_wind", lags=lags
    )
    buoyancy_frequency = _default_window_event_cube(
        dataset, "buoyancy_frequency", lags=lags
    )
    zonal_wavenumbers = np.asarray([1, 2, 3], dtype=int)
    vertical_wavelengths = np.asarray([5.0, 10.0, 15.0])
    shape = (
        len(zonal_wavenumbers),
        len(vertical_wavelengths),
        *wind.shape,
    )
    values = {
        name: np.full(shape, np.nan)
        for name in ("cgx", "cgz", "cgz_over_cgx")
    }
    for k_index, zonal_wavenumber in enumerate(zonal_wavenumbers):
        for wavelength_index, wavelength in enumerate(vertical_wavelengths):
            result = kelvin_group_velocities(
                np.asarray(wind),
                np.asarray(buoyancy_frequency),
                zonal_wavenumber_m1=primary_k_m1 * zonal_wavenumber,
                vertical_wavenumber_m1=(
                    primary_m_m1 * (10.0 / wavelength)
                ),
                minimum_abs_cgx_m_s=1.0e-3,
            )
            values["cgx"][k_index, wavelength_index] = result["cgx_m_s"]
            values["cgz"][k_index, wavelength_index] = result["cgz_m_s"]
            values["cgz_over_cgx"][k_index, wavelength_index] = result[
                "ray_slope"
            ]
    dimensions = (
        "zonal_wavenumber",
        "vertical_wavelength_km",
        *wind.dims,
    )
    coordinates = {
        "zonal_wavenumber": zonal_wavenumbers,
        "vertical_wavelength_km": vertical_wavelengths,
        **{dimension: wind[dimension] for dimension in wind.dims},
    }
    return {
        name: xr.DataArray(
            field, dims=dimensions, coords=coordinates
        )
        for name, field in values.items()
    }


def _store_contrast_results(
    output: xr.Dataset,
    stem: str,
    results: Mapping[str, xr.Dataset],
) -> None:
    contrast_names = list(results)
    for statistic in (
        "mean", "ci_low", "ci_high", "pointwise_significant",
        "fdr_significant", "bootstrap_p_value",
    ):
        output[f"{stem}_{statistic}"] = xr.concat(
            [results[name][statistic] for name in contrast_names],
            dim=xr.IndexVariable("contrast", contrast_names),
        )


def build_review_response_composites(
    section_paths: Mapping[str, Path],
    *,
    wave_section_paths: Mapping[str, Path],
    lags: Sequence[int],
    primary_k_m1: float,
    primary_m_m1: float,
) -> xr.Dataset:
    """Build requested experiment-minus-control response products."""

    sections = {
        simulation: xr.open_dataset(path)
        for simulation, path in section_paths.items()
    }
    wave_sections = {
        simulation: xr.open_dataset(path)
        for simulation, path in wave_section_paths.items()
    }
    output = xr.Dataset()
    try:
        contrasts = [
            simulation
            for simulation in sections
            if simulation != "control"
            if simulation in sections and "control" in sections
        ]
        response_variables = (
            *SURFACE_VARIABLES,
            *VERTICAL_VARIABLES,
        )
        for variable_index, variable in enumerate(response_variables):
            cubes = {
                simulation: _default_window_event_cube(
                    dataset, variable, lags=lags
                )
                for simulation, dataset in sections.items()
            }
            results = {
                simulation: independent_event_contrast(
                    cubes[simulation], cubes["control"],
                    seed=BOOTSTRAP_SEED + variable_index,
                )
                for simulation in contrasts
            }
            _store_contrast_results(output, f"response_{variable}", results)

        requested_stages = list(
            dict.fromkeys(
                int(
                    np.asarray(lags)[
                        np.argmin(np.abs(np.asarray(lags) - target))
                    ]
                )
                for target in (-48, 0, 48)
            )
        )
        cloud_cubes = {
            simulation: _far_field_excess_cube(
                dataset, lags=requested_stages
            )
            for simulation, dataset in sections.items()
        }
        cloud_results = {
            simulation: independent_event_contrast(
                cloud_cubes[simulation], cloud_cubes["control"],
                seed=BOOTSTRAP_SEED + 100,
            )
            for simulation in contrasts
        }
        _store_contrast_results(
            output, "response_cloud_excess_occurrence", cloud_results
        )

        group_cubes = {
            simulation: _group_velocity_event_cubes(
                dataset,
                lags=lags,
                primary_k_m1=primary_k_m1,
                primary_m_m1=primary_m_m1,
            )
            for simulation, dataset in sections.items()
        }
        for variable_index, variable in enumerate(
            ("cgx", "cgz", "cgz_over_cgx")
        ):
            results = {
                simulation: independent_event_contrast(
                    group_cubes[simulation][variable],
                    group_cubes["control"][variable],
                    seed=BOOTSTRAP_SEED + 200 + variable_index,
                )
                for simulation in contrasts
            }
            _store_contrast_results(
                output, f"response_kelvin_reference_{variable}", results
            )
        sensitivity_group_cubes = {
            simulation: _group_velocity_sensitivity_event_cubes(
                dataset,
                lags=lags,
                primary_k_m1=primary_k_m1,
                primary_m_m1=primary_m_m1,
            )
            for simulation, dataset in sections.items()
        }
        for variable_index, variable in enumerate(
            ("cgx", "cgz", "cgz_over_cgx")
        ):
            results = {
                simulation: independent_event_contrast(
                    sensitivity_group_cubes[simulation][variable],
                    sensitivity_group_cubes["control"][variable],
                    seed=BOOTSTRAP_SEED + 250 + variable_index,
                )
                for simulation in contrasts
            }
            _store_contrast_results(
                output,
                f"response_kelvin_sensitivity_{variable}",
                results,
            )
        for variable_index, variable in enumerate(
            FILTERED_VERTICAL_VARIABLES
        ):
            phase_cubes = {
                simulation: _phase_pressure_event_cube(dataset, variable)
                for simulation, dataset in wave_sections.items()
            }
            results = {
                simulation: independent_event_contrast(
                    phase_cubes[simulation],
                    phase_cubes["control"],
                    seed=BOOTSTRAP_SEED + 400 + variable_index,
                )
                for simulation in contrasts
            }
            _store_contrast_results(
                output,
                f"response_phase_pressure_{variable}",
                results,
            )
    finally:
        for dataset in sections.values():
            dataset.close()
        for dataset in wave_sections.values():
            dataset.close()
    output.attrs.update(
        {
            "response_bootstrap": (
                f"{BOOTSTRAP_RESAMPLES} independent-event resamples"
            ),
            "pointwise_interval": "two-sided 95% percentile interval",
            "fdr_method": "Benjamini-Hochberg at alpha=0.05",
            "cloud_background_sector_degrees": "60-150 in both directions",
            "group_velocity_ratio_mask": "abs(Cgx) <= 1e-3 m s-1",
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
    phase_metric_stems = (
        "phase_window_alignment_strength",
        "phase_window_phase_locking_strength",
    )
    contrast_names = [name for name in ordered if name != "control"]
    phase_bootstraps_available = "control" in ordered and all(
        stem in result and f"{stem}_bootstrap" in result
        for stem in phase_metric_stems
    )
    if phase_bootstraps_available and contrast_names:
        for metric_index, stem in enumerate(phase_metric_stems):
            responses = {
                simulation: _independent_bootstrap_metric_response(
                    result[stem].sel(simulation=simulation),
                    result[stem].sel(simulation="control"),
                    result[f"{stem}_bootstrap"].sel(
                        simulation=simulation
                    ),
                    result[f"{stem}_bootstrap"].sel(
                        simulation="control"
                    ),
                    seed=(
                        BOOTSTRAP_SEED
                        + 600
                        + 10 * metric_index
                        + contrast_index
                    ),
                )
                for contrast_index, simulation in enumerate(contrast_names)
            }
            _store_contrast_results(
                result, f"response_{stem}", responses
            )
    result.attrs.update(
        {
            "cache_version": COMPOSITE_CACHE_VERSION,
            "test_subset": int(test_subset),
            "simulation_comparison": (
                "absolute values use each simulation's tracked events; "
                "responses are independent-event experiment-minus-control "
                "contrasts of those absolute values"
            ),
            "literature_presentation": (
                "lag-latitude and lag-pressure adaptations of Lim-and-Son "
                "and Rios-Berrios phase/vertical-section conventions; LPO "
                "lag is not interpreted as filtered-wave phase"
            ),
            "phase_metric_response_bootstrap": (
                f"{BOOTSTRAP_RESAMPLES} independently paired complete-LPO-"
                "trajectory bootstrap draws; two-sided 95% percentile "
                "intervals"
            ),
        }
    )
    return result
