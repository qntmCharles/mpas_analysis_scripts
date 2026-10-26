#!/usr/bin/env python3
"""Vertically integrated vapour transport and tropopause-exchange budgets.

Column water and horizontal transport are integrated on the native MPAS
vertical layers retained in the latitude-longitude archive.

The sign convention is source-positive for the tropospheric column::

    storage = evaporation - precipitation + convergence + top_influx + residual

``top_influx`` is positive downward into the column.  At a moving upper
boundary it is ``rho_v * (Dz_top/Dt - w_top)``; the material motion of the
boundary is therefore included rather than diagnosing ``-rho_v*w`` alone.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
for _relative in ("..", "../..", "../../.."):
    _path = str((_THIS_DIR / _relative).resolve())
    if _path not in sys.path:
        sys.path.append(_path)

import dask
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors as mpl_colors
from matplotlib import pyplot as plt
from matplotlib import ticker as mpl_ticker
from tropopause import tropopause_profile

from diagnostics.diagnostic_helpers import (
    ANALYSIS_END,
    ANALYSIS_START,
    FigureSaver,
    LATLON_PATTERN,
    SPECIFIC_HUMIDITY_CMAP,
    cache_manifest,
    diagnostic_cache_dir,
    diagnostic_profile_cache_dir,
    diagnostic_output_dir,
    finite_or_nan,
    hemispheric_fold,
    lat_mean,
    load_latlon_dataset,
    native_layer_interfaces,
    native_midlevel_height,
    progress,
    spherical_flux_divergence,
    spherical_horizontal_advection,
    simulation_colour,
    standardize_latlon_dataset,
    total_precip_accum,
    write_csv_atomic,
    write_netcdf_atomic,
)
from diagnostics.analysis_config import (
    canonical_simulation_name,
    SIMULATION_ARCHIVES,
    SIMULATION_LABELS as PROJECT_SIMULATION_LABELS,
    SIMULATION_ROOT,
    simulation_cache_name,
    simulation_cache_label,
)


RD = 287.0
CP = 1004.0
P0_PA = 100_000.0
LV = 2.5e6
SECONDS_PER_DAY = 86_400.0
EARTH_RADIUS_M = 6_371_000.0
FIXED_TOP_PA = 20_000.0
SCHEMA_VERSION = 18
EXECUTION_LAYOUT_VERSION = 2
MAX_TROPOPAUSE_SURFACE_JUMP_M = 2_000.0
PLOT_LATITUDE_BOUNDS = (-40.0, 40.0)
TEST_LATITUDE_BOUNDS = (-42.0, 42.0)

BOUNDARIES = ("fixed_200hpa", "cold_point")
MOVING_BOUNDARIES = ("cold_point",)
COMPONENT_NAMES = (
    "temperature",
    "relative_humidity",
    "wind",
    "density",
    "transient",
    "geometry",
    "total",
)

ROOT_DIR = str(SIMULATION_ROOT)
DATA_LOCS = tuple(SIMULATION_ARCHIVES.values())
SIM_NAMES = tuple(PROJECT_SIMULATION_LABELS.values())
SIM_COLORS = tuple(simulation_colour(name) for name in SIM_NAMES)
COMPARISON_NAMES = tuple(
    f"{directory}_minus_control" for directory in DATA_LOCS[1:]
)
COMPARISON_LABELS = tuple(
    f"{simulation} - {SIM_NAMES[0]}" for simulation in SIM_NAMES[1:]
)
COMPARISON_COLORS = SIM_COLORS[1:]
SAVE_LOC = diagnostic_output_dir(
    "/home/cwp29/tmp/mpas/crf_on/final/vapour_transport_budget/"
)
CACHE_ROOT = diagnostic_cache_dir("vapour_transport")
PROFILE_CACHE_ROOT = diagnostic_profile_cache_dir("vapour_transport")
DATA_LOC = diagnostic_cache_dir("vapour_transport_data")
PROFILE_DATA_LOC = diagnostic_profile_cache_dir("vapour_transport_data")

NEEDED_VARS = (
    "theta", "pressure", "zgrid", "rho", "qv", "relhum", "w",
    "qc", "qr", "qi", "qs", "qg",
    "uReconstructZonal", "uReconstructMeridional",
    "mslp", "precipw", "lh",
    "rainnc", "i_rainnc", "rainc", "i_rainc",
)

CHUNKS = {
    "Time": 4,
    "nVertLevels": 75,
    "nVertLevelsP1": 76,
    "latitude": 120,
    "longitude": 240,
}
STATE_CHUNKS = {
    "nVertLevels": 75,
    "nVertLevelsP1": 76,
    "latitude": 120,
    "longitude": 240,
}
DASK_NUM_WORKERS = 1
LATITUDE_BATCH_SIZE = 120
LATITUDE_HALO_POINTS = 2
TIME_BATCH_SIZE = 24
TIME_HALO_POINTS = 2


def mixing_ratio_to_specific_humidity(mixing_ratio):
    """Convert MPAS dry-air vapour mixing ratio to specific humidity."""
    return mixing_ratio / (1.0 + mixing_ratio)


def saturation_specific_humidity(temperature, pressure_pa):
    """Phase-aware saturation specific humidity using Murphy--Koop vapour pressure."""
    temperature = xr.where(temperature > 120.0, temperature, np.nan)
    log_t = np.log(temperature)
    log_es_ice = (
        9.550426 - 5723.265 / temperature
        + 3.53068 * log_t - 0.00728332 * temperature
    )
    log_es_liquid = (
        54.842763 - 6763.22 / temperature - 4.210 * log_t
        + 0.000367 * temperature
        + np.tanh(0.0415 * (temperature - 218.8))
        * (
            53.878 - 1331.22 / temperature - 9.44523 * log_t
            + 0.014025 * temperature
        )
    )
    vapour_pressure = np.exp(
        xr.where(temperature < 273.15, log_es_ice, log_es_liquid)
    )
    epsilon = 0.622
    qsat = epsilon * vapour_pressure / (
        pressure_pa - (1.0 - epsilon) * vapour_pressure
    )
    qsat.name = "saturation_specific_humidity"
    qsat.attrs["units"] = "kg kg-1"
    return qsat


def saturation_mixing_ratio(temperature, pressure_pa):
    """Saturation vapour mixing ratio relative to dry air."""
    qsat = saturation_specific_humidity(temperature, pressure_pa)
    result = qsat / (1.0 - qsat)
    result.name = "saturation_mixing_ratio"
    result.attrs["units"] = "kg kg-1"
    return result


def archived_relative_humidity_fraction(ds):
    """Return the native MPAS relative-humidity field as a fraction."""

    relative_humidity = finite_or_nan(ds.relhum)
    units = str(ds.relhum.attrs.get("units", "")).strip().lower()
    if units in {"%", "percent", "percentage"}:
        relative_humidity = relative_humidity / 100.0
    elif units not in {"1", "fraction", "dimensionless"}:
        raise ValueError(f"Unsupported relhum units: {units!r}")
    relative_humidity.name = "relative_humidity"
    relative_humidity.attrs.update(
        units="1",
        source_variable="relhum",
        source_definition="native MPAS archived relative humidity",
    )
    return relative_humidity


def _parabolic_vertex(x1, x2, x3, y1, y2, y3):
    denominator = (x1 - x2) * (x1 - x3) * (x2 - x3)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denominator
        b = (
            x3 * x3 * (y1 - y2)
            + x2 * x2 * (y3 - y1)
            + x1 * x1 * (y2 - y3)
        ) / denominator
        vertex = -b / (2.0 * a)
    valid = np.isfinite(vertex) & np.isfinite(a) & (a > 0.0)
    vertex = np.where(valid & (vertex >= x1) & (vertex <= x3), vertex, x2)
    c = y2 - a * x2 * x2 - b * x2
    value = np.where(vertex == x2, y2, a * vertex * vertex + b * vertex + c)
    return vertex, value


def _cold_point_block(temperature, pressure, height):
    """Vectorized NumPy kernel returning the cold-point tropopause."""
    shape = temperature.shape[:-1]
    nlevel = temperature.shape[-1]
    temp = np.asarray(temperature, dtype=float).reshape((-1, nlevel))
    pres = np.asarray(pressure, dtype=float).reshape((-1, nlevel))
    z = np.broadcast_to(height, temperature.shape).astype(float).reshape((-1, nlevel))
    nprofile = temp.shape[0]

    cold_mask = (
        np.isfinite(temp) & np.isfinite(pres) & np.isfinite(z)
        & (z >= 7_000.0) & (z <= 20_000.0)
    )
    cold_values = np.where(cold_mask, temp, np.inf)
    cold_index = np.argmin(cold_values, axis=1)
    rows = np.arange(nprofile)
    cold_valid = (
        np.isfinite(cold_values[rows, cold_index])
        & (cold_index > 0) & (cold_index < nlevel - 1)
    )
    lower = np.clip(cold_index - 1, 0, nlevel - 1)
    upper = np.clip(cold_index + 1, 0, nlevel - 1)
    cold_z, cold_t = _parabolic_vertex(
        z[rows, lower], z[rows, cold_index], z[rows, upper],
        temp[rows, lower], temp[rows, cold_index], temp[rows, upper],
    )
    use_upper = cold_z >= z[rows, cold_index]
    p0 = np.where(use_upper, pres[rows, cold_index], pres[rows, lower])
    p1 = np.where(use_upper, pres[rows, upper], pres[rows, cold_index])
    z0 = np.where(use_upper, z[rows, cold_index], z[rows, lower])
    z1 = np.where(use_upper, z[rows, upper], z[rows, cold_index])
    with np.errstate(divide="ignore", invalid="ignore"):
        cold_p = p0 + (cold_z - z0) * (p1 - p0) / (z1 - z0)
    cold_t = np.where(cold_valid, cold_t, np.nan)
    cold_p = np.where(cold_valid, cold_p, np.nan)
    cold_z = np.where(cold_valid, cold_z, np.nan)
    cold_flag = np.where(cold_valid, 0.0, 1.0)

    return tuple(
        values.reshape(shape)
        for values in (cold_t, cold_p, cold_z, cold_flag)
    )


def _wmo_tropopause_pressure_profile(temperature, pressure, height):
    """Return the WMO tropopause using the shared mean-state diagnostic."""
    temperature = np.asarray(temperature, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    height = np.asarray(height, dtype=float)
    valid = (
        np.isfinite(temperature)
        & np.isfinite(pressure)
        & np.isfinite(height)
    )
    if valid.sum() < 4:
        return np.nan
    temperature = temperature[valid]
    pressure = pressure[valid]
    height = height[valid]
    order = np.argsort(height)
    temperature = temperature[order]
    pressure = pressure[order]
    height = height[order]
    if np.any(np.diff(height) <= 0.0):
        return np.nan
    _temperature, tropopause_pressure, _height, _flag = tropopause_profile(
        temperature, pressure, height
    )
    return tropopause_pressure


def wmo_tropopause_pressure_from_mean_profile(product):
    """Match the hemispherically folded WMO diagnostic in mean_state."""
    temperature = hemispheric_fold(
        product.native_temperature_zonal_mean, parity="even"
    )
    pressure_profile = hemispheric_fold(
        product.native_pressure_zonal_mean, parity="even"
    )
    height = hemispheric_fold(
        product.native_midlevel_height_zonal_mean, parity="even"
    )
    pressure = xr.apply_ufunc(
        _wmo_tropopause_pressure_profile,
        temperature,
        pressure_profile,
        height,
        input_core_dims=[
            ["nVertLevels"], ["nVertLevels"], ["nVertLevels"]
        ],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    pressure.name = "wmo_tropopause_pressure"
    pressure.attrs.update(
        units="Pa",
        diagnostic=(
            "WMO lapse-rate tropopause diagnosed with the shared "
            "tropopause_profile routine from hemispherically folded, "
            "time-mean, zonal-mean native-level profiles"
        ),
    )
    return pressure


def _native_interpolation_weights_block(coordinate, target):
    """Find one native-level bracket and fraction for every target column."""
    coordinate = np.asarray(coordinate, dtype=float)
    target = np.asarray(target, dtype=float)
    nlevel = coordinate.shape[-1]
    if nlevel < 2:
        raise ValueError("native interpolation requires at least two levels")
    shape = np.broadcast_shapes(
        coordinate.shape[:-1], target.shape
    )
    coordinate = np.broadcast_to(coordinate, shape + (nlevel,))
    target = np.broadcast_to(target, shape)

    x0 = coordinate[..., :-1]
    x1 = coordinate[..., 1:]
    lower = np.minimum(x0, x1)
    upper = np.maximum(x0, x1)
    within = (
        np.isfinite(target)[..., None]
        & np.isfinite(x0) & np.isfinite(x1)
        & (target[..., None] >= lower)
        & (target[..., None] <= upper)
        & (x1 != x0)
    )
    valid = np.any(within, axis=-1)
    lower_index = np.argmax(within, axis=-1).astype(np.int16)
    safe_index = np.where(valid, lower_index, 0)
    selected_x0 = np.take_along_axis(
        x0, safe_index[..., None], axis=-1
    )[..., 0]
    selected_x1 = np.take_along_axis(
        x1, safe_index[..., None], axis=-1
    )[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = (target - selected_x0) / (selected_x1 - selected_x0)
    return (
        np.where(valid, lower_index, -1).astype(np.int16),
        np.where(valid, fraction, np.nan),
    )


def _apply_native_interpolation_block(field, lower_index, fraction):
    """Apply precomputed native interpolation weights to one field."""
    field = np.asarray(field, dtype=float)
    lower_index = np.asarray(lower_index)
    fraction = np.asarray(fraction, dtype=float)
    nlevel = field.shape[-1]
    shape = np.broadcast_shapes(
        field.shape[:-1], lower_index.shape, fraction.shape
    )
    field = np.broadcast_to(field, shape + (nlevel,))
    lower_index = np.broadcast_to(lower_index, shape)
    fraction = np.broadcast_to(fraction, shape)
    safe_index = np.clip(lower_index, 0, nlevel - 2)
    f0 = np.take_along_axis(
        field, safe_index[..., None], axis=-1
    )[..., 0]
    f1 = np.take_along_axis(
        field, (safe_index + 1)[..., None], axis=-1
    )[..., 0]
    result = f0 + fraction * (f1 - f0)
    return np.where(
        (lower_index >= 0) & np.isfinite(fraction)
        & np.isfinite(f0) & np.isfinite(f1),
        result,
        np.nan,
    )


def native_interpolation_weights(coordinate, target, dim="nVertLevels"):
    """Return reusable lower-level indices and fractions for a target surface."""
    if dim not in coordinate.dims:
        raise ValueError(f"{dim!r} must be present in coordinate")
    target = (
        target
        if isinstance(target, xr.DataArray)
        else xr.DataArray(float(target))
    )
    lower_index, fraction = xr.apply_ufunc(
        _native_interpolation_weights_block,
        coordinate,
        target,
        input_core_dims=[[dim], []],
        output_core_dims=[[], []],
        dask="parallelized",
        vectorize=False,
        output_dtypes=[np.int16, float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    return lower_index, fraction


def interpolate_native_with_weights(
    field, lower_index, fraction, dim="nVertLevels",
):
    """Interpolate a native field using a previously diagnosed bracket."""
    if dim not in field.dims:
        raise ValueError(f"{dim!r} must be present in field")
    result = xr.apply_ufunc(
        _apply_native_interpolation_block,
        field,
        lower_index,
        fraction,
        input_core_dims=[[dim], [], []],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=False,
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    result.name = getattr(field, "name", None)
    result.attrs.update(getattr(field, "attrs", {}))
    return result


def interpolate_native_profile(field, coordinate, target, dim="nVertLevels"):
    """Linearly interpolate a native profile to a varying target coordinate."""
    if dim not in field.dims or dim not in coordinate.dims:
        raise ValueError(f"{dim!r} must be present in field and coordinate")
    if field.sizes[dim] != coordinate.sizes[dim]:
        raise ValueError("field and coordinate must have the same vertical size")

    lower_index, fraction = native_interpolation_weights(
        coordinate, target, dim=dim
    )
    return interpolate_native_with_weights(
        field, lower_index, fraction, dim=dim
    )


def native_height_at_pressure(ds, target_pressure_pa):
    """Interpolate native mid-level height linearly in log pressure."""
    height = native_midlevel_height(ds, ds.pressure)
    log_pressure = np.log(finite_or_nan(ds.pressure))
    target = (
        target_pressure_pa
        if isinstance(target_pressure_pa, xr.DataArray)
        else xr.DataArray(float(target_pressure_pa))
    )
    result = interpolate_native_profile(height, log_pressure, np.log(target))
    result.attrs.update(units="m", long_name="native-grid interpolated height")
    return result


def native_bounded_height_integral(field, ds, top_height,
                                   dim="nVertLevels"):
    """Integrate a native layer-centred volume density up to ``top_height``.

    Layers wholly below the boundary receive their full geometric thickness;
    the layer intersected by the boundary receives a fractional thickness.
    ``field`` should already include density (for example ``rho*qv`` or
    ``rho*qv*u``).
    """
    if dim not in field.dims:
        raise ValueError(f"{dim!r} is not a dimension of {field.dims!r}")
    overlap = native_layer_overlap(field, ds, top_height, dim=dim)
    contribution = (field * overlap).where(overlap > 0.0, 0.0)
    result = contribution.sum(dim, skipna=False).where(np.isfinite(top_height))
    result.name = getattr(field, "name", None)
    return result


def native_layer_overlap(field, ds, top_height, dim="nVertLevels"):
    """Return native-layer geometric thickness below a varying top surface."""
    if dim not in field.dims:
        raise ValueError(f"{dim!r} is not a dimension of {field.dims!r}")
    lower, upper = native_layer_interfaces(ds, field)
    thickness = upper - lower
    clipped_top = xr.where(top_height < upper, top_height, upper)
    overlap = (clipped_top - lower).clip(min=0.0)
    return xr.where(overlap < thickness, overlap, thickness)


def diagnose_tropopauses(ds):
    temperature = ds.theta * (ds.pressure / P0_PA) ** (RD / CP)
    temperature = finite_or_nan(temperature)
    pressure = finite_or_nan(ds.pressure)
    height = native_midlevel_height(ds, temperature)

    outputs = xr.apply_ufunc(
        _cold_point_block,
        temperature,
        pressure,
        height,
        input_core_dims=[["nVertLevels"], ["nVertLevels"], ["nVertLevels"]],
        output_core_dims=[[]] * 4,
        dask="parallelized",
        vectorize=False,
        output_dtypes=[float] * 4,
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    names = (
        "cold_point_temperature", "cold_point_pressure",
        "cold_point_height", "cold_point_flag",
    )
    result = xr.Dataset(dict(zip(names, outputs)))
    for prefix in MOVING_BOUNDARIES:
        valid = (
            np.isfinite(result[f"{prefix}_pressure"])
            & (result[f"{prefix}_pressure"] >= 4_000.0)
            & (result[f"{prefix}_pressure"] <= 45_000.0)
            & (result[f"{prefix}_height"] >= 5_000.0)
            & (result[f"{prefix}_height"] <= 22_000.0)
        )
        for suffix in ("temperature", "pressure", "height"):
            result[f"{prefix}_{suffix}"] = result[f"{prefix}_{suffix}"].where(valid)
        result[f"{prefix}_flag"] = xr.where(valid, result[f"{prefix}_flag"], 1.0)

    result.cold_point_pressure.attrs.update(units="Pa", positive="down")
    result.cold_point_height.attrs["units"] = "m"
    result.cold_point_temperature.attrs["units"] = "K"
    return result


def time_tendency(field, step=1):
    """Centred time tendency using records ``step`` places apart."""
    seconds = (
        field.Time.shift(Time=-step) - field.Time.shift(Time=step)
    ) / np.timedelta64(1, "s")
    return (field.shift(Time=-step) - field.shift(Time=step)) / seconds


def trapezoidal_time_weights(time):
    """Global sample weights for a trapezoidal mean."""
    if time.size < 2:
        raise ValueError("at least two times are required for a time mean")
    times = np.asarray(time.values)
    intervals = np.diff(times) / np.timedelta64(1, "s")
    if np.any(~np.isfinite(intervals)) or np.any(intervals <= 0.0):
        raise ValueError("Time must be finite and strictly increasing")
    weights = np.empty(times.size, dtype=float)
    weights[0] = 0.5 * intervals[0]
    weights[-1] = 0.5 * intervals[-1]
    if times.size > 2:
        weights[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
    return xr.DataArray(
        weights, dims="Time", coords={"Time": time}, name="time_weight"
    )


def trapezoidal_time_mean(field):
    """Time mean consistent with the interval spanned by endpoint storage."""
    if "Time" not in field.dims:
        raise ValueError("field must contain a Time dimension")
    weights = trapezoidal_time_weights(field.Time)
    valid_weights = weights.where(np.isfinite(field))
    denominator = valid_weights.sum("Time", skipna=True)
    result = (field * weights).sum("Time", skipna=True) / denominator.where(
        denominator > 0.0
    )
    result.name = getattr(field, "name", None)
    result.attrs.update(getattr(field, "attrs", {}))
    return result


def boundary_arrays(ds, tropopauses):
    template = xr.full_like(ds.mslp, FIXED_TOP_PA, dtype=float)
    fixed_height = native_height_at_pressure(ds, template)
    pressure = xr.concat(
        [template, tropopauses.cold_point_pressure],
        dim=xr.IndexVariable("boundary", list(BOUNDARIES)),
    )
    height = xr.concat(
        [fixed_height, tropopauses.cold_point_height],
        dim=xr.IndexVariable("boundary", list(BOUNDARIES)),
    )
    temperature = xr.concat(
        [xr.full_like(template, np.nan), tropopauses.cold_point_temperature],
        dim=xr.IndexVariable("boundary", list(BOUNDARIES)),
    )
    return pressure, height, temperature


def moving_surface_continuity(height, time_step=1):
    """Where a diagnosed tropopause is a locally differentiable surface.

    A branch change in a diagnosed tropopause can create a multi-kilometre
    jump between adjacent samples or grid points. A flux through such a
    discontinuous diagnostic surface is undefined, so it is masked rather
    than interpreted as very large boundary motion.
    """
    temporal_jump = np.abs(
        height.shift(Time=-time_step) - height.shift(Time=time_step)
    )
    latitude_jump = xr.concat(
        [
            np.abs(height - height.shift(latitude=1)),
            np.abs(height - height.shift(latitude=-1)),
        ],
        dim="neighbour",
    ).max("neighbour", skipna=False)
    longitude_jump = xr.concat(
        [
            np.abs(height - height.roll(longitude=1, roll_coords=False)),
            np.abs(height - height.roll(longitude=-1, roll_coords=False)),
        ],
        dim="neighbour",
    ).max("neighbour", skipna=False)
    return (
        np.isfinite(height)
        & (temporal_jump <= MAX_TROPOPAUSE_SURFACE_JUMP_M)
        & (latitude_jump <= MAX_TROPOPAUSE_SURFACE_JUMP_M)
        & (longitude_jump <= MAX_TROPOPAUSE_SURFACE_JUMP_M)
    )


def _core_latitude_subset(obj, core_latitude):
    if core_latitude is None:
        return obj
    return obj.sel(latitude=core_latitude)


def instantaneous_budget(ds, tropopauses, core_latitude=None):
    """Construct instantaneous terms, loading native halos only when required."""
    top_pressure_halo, top_height_halo, top_temperature_halo = boundary_arrays(
        ds, tropopauses
    )
    core_ds = _core_latitude_subset(ds, core_latitude)
    top_pressure = _core_latitude_subset(top_pressure_halo, core_latitude)
    top_height = _core_latitude_subset(top_height_halo, core_latitude)
    top_temperature = _core_latitude_subset(
        top_temperature_halo, core_latitude
    )

    native_mixing_ratio = finite_or_nan(core_ds.qv)
    native_u = finite_or_nan(core_ds.uReconstructZonal)
    native_v = finite_or_nan(core_ds.uReconstructMeridional)
    vapour_density = finite_or_nan(core_ds.rho) * native_mixing_ratio

    native_height = native_midlevel_height(core_ds, native_mixing_ratio)
    layer_overlap = native_layer_overlap(
        vapour_density, core_ds, top_height
    )
    column = (vapour_density * layer_overlap).sum(
        "nVertLevels", skipna=False
    ).where(np.isfinite(top_height))
    ivt_u = (vapour_density * native_u * layer_overlap).sum(
        "nVertLevels", skipna=False
    ).where(np.isfinite(top_height))
    ivt_v = (vapour_density * native_v * layer_overlap).sum(
        "nVertLevels", skipna=False
    ).where(np.isfinite(top_height))

    native_lower_index, native_fraction = native_interpolation_weights(
        native_height, top_height
    )
    vapour_density_top = interpolate_native_with_weights(
        vapour_density, native_lower_index, native_fraction
    )
    native_u_top = interpolate_native_with_weights(
        native_u, native_lower_index, native_fraction
    )
    native_v_top = interpolate_native_with_weights(
        native_v, native_lower_index, native_fraction
    )
    w_lower_index, w_fraction = native_interpolation_weights(
        finite_or_nan(core_ds.zgrid), top_height, dim="nVertLevelsP1"
    )
    native_w_top = interpolate_native_with_weights(
        finite_or_nan(core_ds.w),
        w_lower_index,
        w_fraction,
        dim="nVertLevelsP1",
    )

    height_local_motion = []
    height_local_motion_24h = []
    height_horizontal_motion = []
    continuity_masks = []
    continuity_masks_24h = []
    derivative_height = top_height_halo
    if derivative_height.chunks is not None:
        derivative_height = derivative_height.chunk({"latitude": -1})
    for boundary in BOUNDARIES:
        height_halo = derivative_height.sel(boundary=boundary, drop=True)
        height = _core_latitude_subset(height_halo, core_latitude)
        local = time_tendency(height, step=1)
        local24 = time_tendency(height, step=2)
        horizontal = spherical_horizontal_advection(
            height_halo,
            native_u_top.sel(boundary=boundary, drop=True),
            native_v_top.sel(boundary=boundary, drop=True),
        )
        if boundary in MOVING_BOUNDARIES:
            continuity = _core_latitude_subset(
                moving_surface_continuity(height_halo, time_step=1),
                core_latitude,
            )
            continuity24 = _core_latitude_subset(
                moving_surface_continuity(height_halo, time_step=2),
                core_latitude,
            )
        else:
            # The fixed-pressure surface has no branch ambiguity.  Its common
            # validity is set only by the centred time-difference support.
            continuity = np.isfinite(local)
            continuity24 = np.isfinite(local24)
        local = local.where(continuity)
        local24 = local24.where(continuity24)
        horizontal = horizontal.where(continuity)
        height_local_motion.append(local)
        height_local_motion_24h.append(local24)
        height_horizontal_motion.append(horizontal)
        continuity_masks.append(continuity)
        continuity_masks_24h.append(continuity24)

    height_local_motion = xr.concat(
        height_local_motion,
        dim=xr.IndexVariable("boundary", list(BOUNDARIES)),
    )
    height_local_motion_24h = xr.concat(
        height_local_motion_24h,
        dim=xr.IndexVariable("boundary", list(BOUNDARIES)),
    )
    height_horizontal_motion = xr.concat(
        height_horizontal_motion,
        dim=xr.IndexVariable("boundary", list(BOUNDARIES)),
    )
    continuity = xr.concat(
        continuity_masks,
        dim=xr.IndexVariable("boundary", list(BOUNDARIES)),
    )
    continuity24 = xr.concat(
        continuity_masks_24h,
        dim=xr.IndexVariable("boundary", list(BOUNDARIES)),
    )
    raw_top_vertical = -vapour_density_top * native_w_top
    top_vertical = raw_top_vertical.where(continuity)
    top_local = vapour_density_top * height_local_motion
    top_local_24h = vapour_density_top * height_local_motion_24h
    top_horizontal = vapour_density_top * height_horizontal_motion
    top_influx = top_vertical + top_local + top_horizontal
    top_influx_24h = (
        raw_top_vertical.where(continuity24)
        + top_local_24h
        + (vapour_density_top * height_horizontal_motion).where(continuity24)
    )

    lower_interface, upper_interface = native_layer_interfaces(
        core_ds, vapour_density
    )
    full_column = (
        vapour_density * (upper_interface - lower_interface)
    ).sum("nVertLevels", skipna=False)

    return xr.Dataset(
        {
            "column_vapour": column,
            "ivt_u": ivt_u,
            "ivt_v": ivt_v,
            "top_influx": top_influx,
            "top_influx_24h": top_influx_24h,
            "top_vertical_velocity_component": top_vertical,
            "top_local_motion_component": top_local,
            "top_horizontal_motion_component": top_horizontal,
            "top_flux_surface_valid": continuity,
            "boundary_pressure": top_pressure,
            "boundary_height": top_height,
            "boundary_temperature": top_temperature,
            "full_column_vapour": full_column,
        }
    )


def derive_products(ds, simulation):
    progress(f"{simulation}: diagnosing instantaneous tropopauses")
    tropopauses = diagnose_tropopauses(ds)
    instantaneous = instantaneous_budget(ds, tropopauses)

    progress(f"{simulation}: constructing mean budget products")
    ivt_u_mean = trapezoidal_time_mean(instantaneous.ivt_u)
    ivt_v_mean = trapezoidal_time_mean(instantaneous.ivt_v)
    convergence = -spherical_flux_divergence(ivt_u_mean, ivt_v_mean)

    elapsed_seconds = float(
        (ds.Time.values[-1] - ds.Time.values[0]) / np.timedelta64(1, "s")
    )
    storage = (
        instantaneous.column_vapour.isel(Time=-1)
        - instantaneous.column_vapour.isel(Time=0)
    ) / elapsed_seconds
    evaporation = trapezoidal_time_mean(finite_or_nan(ds.lh) / LV)
    accumulated_precipitation = total_precip_accum(ds)
    precipitation = (
        accumulated_precipitation.isel(Time=-1)
        - accumulated_precipitation.isel(Time=0)
    ) / elapsed_seconds

    condensate_endpoints = ds.isel(Time=[0, -1])
    layer_coord = condensate_endpoints.qv.nVertLevels
    lower_interface = condensate_endpoints.zgrid.isel(
        nVertLevelsP1=slice(0, -1)
    ).rename(
        {"nVertLevelsP1": "nVertLevels"}
    ).assign_coords(nVertLevels=layer_coord)
    upper_interface = condensate_endpoints.zgrid.isel(
        nVertLevelsP1=slice(1, None)
    ).rename(
        {"nVertLevelsP1": "nVertLevels"}
    ).assign_coords(nVertLevels=layer_coord)
    layer_thickness = np.abs(upper_interface - lower_interface)
    condensate_mixing_ratio = sum(
        finite_or_nan(condensate_endpoints[name])
        for name in ("qc", "qr", "qi", "qs", "qg")
    )
    condensate_column = (
        finite_or_nan(condensate_endpoints.rho)
        * condensate_mixing_ratio * layer_thickness
    ).sum("nVertLevels", skipna=True)
    condensate_storage = (
        condensate_column.isel(Time=-1) - condensate_column.isel(Time=0)
    ) / elapsed_seconds

    top_influx = trapezoidal_time_mean(instantaneous.top_influx)
    top_influx_24h = trapezoidal_time_mean(instantaneous.top_influx_24h)
    residual = storage - (evaporation - precipitation + convergence + top_influx)
    conventional_residual = storage - (evaporation - precipitation + convergence)
    closure_scale = (
        np.abs(storage) + np.abs(evaporation) + np.abs(precipitation)
        + np.abs(convergence) + np.abs(top_influx)
    )
    budget_closure_fraction = xr.where(
        closure_scale > 0.0, np.abs(residual) / closure_scale, np.nan
    )

    native_vapour_density = finite_or_nan(ds.rho) * finite_or_nan(ds.qv)
    native_vapour_density_mean = trapezoidal_time_mean(native_vapour_density)
    native_u_mean = trapezoidal_time_mean(finite_or_nan(ds.uReconstructZonal))
    native_v_mean = trapezoidal_time_mean(
        finite_or_nan(ds.uReconstructMeridional)
    )
    fixed_top_height_mean = instantaneous.boundary_height.sel(
        boundary="fixed_200hpa", drop=True
    ).pipe(trapezoidal_time_mean)
    mean_flow_u = native_bounded_height_integral(
        native_vapour_density_mean * native_u_mean,
        ds,
        fixed_top_height_mean,
    )
    mean_flow_v = native_bounded_height_integral(
        native_vapour_density_mean * native_v_mean,
        ds,
        fixed_top_height_mean,
    )
    transient_u = ivt_u_mean.sel(boundary="fixed_200hpa", drop=True) - mean_flow_u
    transient_v = ivt_v_mean.sel(boundary="fixed_200hpa", drop=True) - mean_flow_v

    native_temperature = finite_or_nan(
        ds.theta * (ds.pressure / P0_PA) ** (RD / CP)
    )
    native_specific_humidity = mixing_ratio_to_specific_humidity(
        finite_or_nan(ds.qv)
    )
    native_relative_humidity = archived_relative_humidity_fraction(ds)
    native_temperature_zonal = trapezoidal_time_mean(
        native_temperature
    ).mean("longitude", skipna=True)
    native_specific_humidity_zonal = trapezoidal_time_mean(
        native_specific_humidity.mean("longitude", skipna=True)
    )
    native_relative_humidity_zonal = trapezoidal_time_mean(
        native_relative_humidity.mean("longitude", skipna=True)
    )
    native_pressure_zonal = trapezoidal_time_mean(
        finite_or_nan(ds.pressure)
    ).mean("longitude", skipna=True)
    native_height = native_midlevel_height(ds, native_temperature)
    if "Time" in native_height.dims:
        native_height = trapezoidal_time_mean(native_height)
    native_height_zonal = native_height.mean("longitude", skipna=True)
    zmid = lat_mean(native_height_zonal)

    rates = xr.Dataset(
        {
            "storage": storage * SECONDS_PER_DAY,
            "moisture_convergence": convergence * SECONDS_PER_DAY,
            "evaporation": evaporation * SECONDS_PER_DAY,
            "precipitation": precipitation * SECONDS_PER_DAY,
            "top_influx": top_influx * SECONDS_PER_DAY,
            "top_influx_24h": top_influx_24h * SECONDS_PER_DAY,
            "residual": residual * SECONDS_PER_DAY,
            "conventional_residual": conventional_residual * SECONDS_PER_DAY,
            "condensate_storage": condensate_storage * SECONDS_PER_DAY,
        }
    )
    for variable in rates.data_vars:
        rates[variable].attrs["units"] = "mm day-1"

    full_column_vapour_mean = trapezoidal_time_mean(
        instantaneous.full_column_vapour
    )
    archived_precipw_mean = trapezoidal_time_mean(finite_or_nan(ds.precipw))
    native_precipw_bias = full_column_vapour_mean - archived_precipw_mean
    relative_precipw_error = xr.where(
        np.abs(archived_precipw_mean) > 0.0,
        np.abs(native_precipw_bias) / np.abs(archived_precipw_mean),
        np.nan,
    )
    products = xr.merge(
        [
            rates,
            xr.Dataset(
                {
                    "column_vapour_mean": trapezoidal_time_mean(
                        instantaneous.column_vapour
                    ),
                    "ivt_u_mean": ivt_u_mean,
                    "ivt_v_mean": ivt_v_mean,
                    "boundary_pressure_mean": trapezoidal_time_mean(
                        instantaneous.boundary_pressure
                    ),
                    "boundary_height_mean": trapezoidal_time_mean(
                        instantaneous.boundary_height
                    ),
                    "boundary_temperature_mean": trapezoidal_time_mean(
                        instantaneous.boundary_temperature
                    ),
                    "cold_point_valid_fraction": np.isfinite(
                        tropopauses.cold_point_pressure
                    ).mean("Time"),
                    "top_vertical_velocity_component": trapezoidal_time_mean(
                        instantaneous.top_vertical_velocity_component
                    ) * SECONDS_PER_DAY,
                    "top_local_motion_component": trapezoidal_time_mean(
                        instantaneous.top_local_motion_component
                    ) * SECONDS_PER_DAY,
                    "top_horizontal_motion_component": trapezoidal_time_mean(
                        instantaneous.top_horizontal_motion_component
                    ) * SECONDS_PER_DAY,
                    "full_column_vapour_mean": full_column_vapour_mean,
                    "archived_precipw_mean": archived_precipw_mean,
                    "precipw_integration_bias": native_precipw_bias,
                    "relative_precipw_error": relative_precipw_error,
                    "budget_closure_fraction": budget_closure_fraction,
                    "fixed_mean_flow_ivt_u": mean_flow_u,
                    "fixed_mean_flow_ivt_v": mean_flow_v,
                    "fixed_transient_ivt_u": transient_u,
                    "fixed_transient_ivt_v": transient_v,
                    "native_temperature_zonal_mean": native_temperature_zonal,
                    "native_specific_humidity_zonal_mean": (
                        native_specific_humidity_zonal
                    ),
                    "native_relative_humidity_zonal_mean": (
                        native_relative_humidity_zonal
                    ),
                    "native_pressure_zonal_mean": native_pressure_zonal,
                    "native_midlevel_height_zonal_mean": native_height_zonal,
                    "native_midlevel_height": zmid,
                    "top_flux_valid_fraction": (
                        instantaneous.top_flux_surface_valid.mean(
                            "Time", skipna=True
                        )
                    ),
                }
            ),
        ],
        compat="override",
    )
    for name in (
        "column_vapour_mean", "full_column_vapour_mean",
        "archived_precipw_mean", "precipw_integration_bias",
    ):
        products[name].attrs["units"] = "kg m-2"
    for name in (
        "relative_precipw_error", "budget_closure_fraction",
    ):
        products[name].attrs["units"] = "1"
    for name in ("ivt_u_mean", "ivt_v_mean", "fixed_mean_flow_ivt_u",
                 "fixed_mean_flow_ivt_v", "fixed_transient_ivt_u",
                 "fixed_transient_ivt_v"):
        products[name].attrs["units"] = "kg m-1 s-1"
    for name in (
        "top_vertical_velocity_component", "top_local_motion_component",
        "top_horizontal_motion_component",
    ):
        products[name].attrs["units"] = "mm day-1"
    products.native_temperature_zonal_mean.attrs["units"] = "K"
    products.native_specific_humidity_zonal_mean.attrs["units"] = "kg kg-1"
    products.native_relative_humidity_zonal_mean.attrs["units"] = "1"
    products.native_relative_humidity_zonal_mean.attrs[
        "source_variable"
    ] = "relhum"
    products.native_pressure_zonal_mean.attrs["units"] = "Pa"
    products.native_midlevel_height_zonal_mean.attrs["units"] = "m"
    products.top_flux_valid_fraction.attrs["units"] = "1"
    products.attrs.update(
        schema_version=SCHEMA_VERSION,
        cache_kind="mean_products",
        simulation=simulation,
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        budget_equation=(
            "storage = evaporation - precipitation + moisture_convergence "
            "+ top_influx + residual"
        ),
        top_influx_sign="positive downward into the diagnosed column",
        column_integration=(
            "native MPAS model levels: sum(rho_d * qv_dry_mixing_ratio * dz), "
            "with a fractional layer at the upper boundary"
        ),
        ivt_integration=(
            "native MPAS model levels: sum(rho_d * qv_dry_mixing_ratio * "
            "horizontal_wind * dz), with a fractional upper layer"
        ),
        top_flux_method=(
            "native height coordinate: rho_d*qv*(dz_top/dt + "
            "u_top*grad_x(z_top) + v_top*grad_y(z_top) - w_top)"
        ),
        top_flux_surface_filter=(
            "moving-tropopause flux is retained only where the surface is "
            "finite and adjacent time, latitude, and longitude tropopause "
            f"heights differ by no more than {MAX_TROPOPAUSE_SURFACE_JUMP_M:g} m"
        ),
        closure_fraction_definition=(
            "abs(residual) / (abs(storage) + abs(evaporation) + "
            "abs(precipitation) + abs(moisture_convergence) + abs(top_influx))"
        ),
        horizontal_grid_note=(
            "native vertical levels retained, but fields were archived on the "
            "regular latitude-longitude analysis grid"
        ),
        time_averaging=(
            "trapezoidal integration of instantaneous native fields over the "
            "endpoint storage interval; precipitation from the exact accumulated "
            "difference"
        ),
        sampling_interval_hours=float(
            np.median(np.diff(ds.Time.values) / np.timedelta64(1, "h"))
        ),
        averaging_window_days=float(elapsed_seconds / SECONDS_PER_DAY),
    )
    return products


def cache_path(simulation, test=False):
    suffix = "_test" if test else ""
    safe = simulation_cache_name(simulation)
    return os.path.join(CACHE_ROOT, f"{safe}{suffix}_mean_products.nc")


def state_cache_path(simulation, test=False):
    suffix = "_test" if test else ""
    safe = simulation_cache_name(simulation)
    return os.path.join(CACHE_ROOT, f"{safe}{suffix}_native_mean_state.nc")


def attribution_cache_path(test=False):
    suffix = "_test" if test else ""
    return os.path.join(
        PROFILE_CACHE_ROOT, f"vapour_transport_attribution{suffix}.nc"
    )


def attribution_cache_manifest(test=False):
    source_paths = []
    for simulation in SIM_NAMES:
        source_paths.extend(
            [
                cache_path(simulation, test=test),
                state_cache_path(simulation, test=test),
            ]
        )
    return cache_manifest(
        "vapour_transport_budget.transport_attribution",
        schema=SCHEMA_VERSION,
        configuration={"test": bool(test)},
        source_paths=source_paths,
    )


def attribution_cache_is_current(path, test=False):
    if not os.path.exists(path):
        return False
    try:
        expected = attribution_cache_manifest(test=test)
        with xr.open_dataset(path) as cached:
            return (
                int(cached.attrs.get("schema_version", -1))
                == SCHEMA_VERSION
                and cached.attrs.get("cache_kind")
                == "transport_attribution"
                and cached.attrs.get("manifest_sha256")
                == expected["manifest_sha256"]
            )
    except (FileNotFoundError, OSError, ValueError):
        return False


def simulation_cache_manifest(data_loc, simulation, test=False, test_times=5):
    source_paths = sorted(
        glob.glob(os.path.join(ROOT_DIR, data_loc, LATLON_PATTERN))
    )
    return cache_manifest(
        "vapour_transport_budget.simulation_products",
        schema=f"{SCHEMA_VERSION}.{EXECUTION_LAYOUT_VERSION}",
        configuration={
            # Retain the historical manifest identity when plot labels change.
            "simulation": simulation_cache_label(simulation),
            "data_loc": data_loc,
            "analysis_start": ANALYSIS_START,
            "analysis_end": ANALYSIS_END,
            "needed_vars": list(NEEDED_VARS),
            "test": bool(test),
            "test_times": int(test_times) if test else None,
            "test_latitude_bounds": list(TEST_LATITUDE_BOUNDS) if test else None,
        },
        source_paths=source_paths,
    )


def cache_is_current(
    path, simulation, cache_kind, expected_manifest_sha256=None
):
    if not os.path.exists(path):
        return False
    try:
        with xr.open_dataset(path) as cached:
            return (
                int(cached.attrs.get("schema_version", -1)) == SCHEMA_VERSION
                and (
                    cached.attrs.get("simulation") == simulation
                    or canonical_simulation_name(cached.attrs.get("simulation"))
                    == canonical_simulation_name(simulation)
                )
                and cached.attrs.get("analysis_start") == ANALYSIS_START
                and cached.attrs.get("analysis_end") == ANALYSIS_END
                and cached.attrs.get("cache_kind") == cache_kind
                and (
                    expected_manifest_sha256 is None
                    or cached.attrs.get("source_manifest_sha256")
                    == expected_manifest_sha256
                )
            )
    except (OSError, ValueError, KeyError):
        return False


def write_dataset_cache(dataset, path, description):
    progress(f"writing {description} cache {path}")
    write_netcdf_atomic(dataset, path, num_workers=DASK_NUM_WORKERS)


def write_final_data_products(
    attribution, products, data_loc=PROFILE_DATA_LOC, test=False
):
    """Write retained data products below the central cache root."""
    os.makedirs(data_loc, exist_ok=True)
    suffix = "_test" if test else ""
    attribution_path = os.path.join(
        data_loc, f"vapour_transport_attribution{suffix}.nc"
    )
    summary_path = os.path.join(
        data_loc, f"vapour_transport_budget_summary{suffix}.csv"
    )
    write_dataset_cache(
        attribution, attribution_path, "vapour-transport attribution"
    )
    write_csv_atomic(
        summary_table(products, attribution), summary_path, index=False
    )
    progress(f"writing vapour-budget summary {summary_path}")
    return attribution_path, summary_path


def bounded_compute(*objects):
    """Compute with a small fixed thread pool to bound active chunk memory."""
    return dask.compute(
        *objects, scheduler="threads", num_workers=DASK_NUM_WORKERS
    )


def load_simulation_dataset(data_loc, test=False, test_times=5):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r'The specified chunks separate the stored chunks along '
                r'dimension "(latitude|longitude)".*'
            ),
            category=UserWarning,
        )
        ds = load_latlon_dataset(
            os.path.join(ROOT_DIR, data_loc),
            LATLON_PATTERN,
            chunks=CHUNKS,
            direct_chunks=True,
        )
    ds = standardize_latlon_dataset(ds, NEEDED_VARS)
    if test:
        ds = ds.isel(Time=slice(0, int(test_times))).sel(
            latitude=slice(*TEST_LATITUDE_BOUNDS)
        )
    return ds


def native_mean_state(ds):
    """Native-level time means needed for exact flux-product attribution."""
    mixing_ratio = finite_or_nan(ds.qv)
    density = finite_or_nan(ds.rho)
    zonal_wind = finite_or_nan(ds.uReconstructZonal)
    meridional_wind = finite_or_nan(ds.uReconstructMeridional)
    temperature = finite_or_nan(
        ds.theta * (ds.pressure / P0_PA) ** (RD / CP)
    )
    state = xr.Dataset(
        {
            "mixing_ratio": trapezoidal_time_mean(mixing_ratio),
            "density": trapezoidal_time_mean(density),
            "pressure": trapezoidal_time_mean(finite_or_nan(ds.pressure)),
            "temperature": trapezoidal_time_mean(temperature),
            "zonal_wind": trapezoidal_time_mean(zonal_wind),
            "meridional_wind": trapezoidal_time_mean(meridional_wind),
            "vapour_density": trapezoidal_time_mean(density * mixing_ratio),
            "zonal_vapour_flux": trapezoidal_time_mean(
                density * mixing_ratio * zonal_wind
            ),
            "meridional_vapour_flux": trapezoidal_time_mean(
                density * mixing_ratio * meridional_wind
            ),
            "zgrid": finite_or_nan(ds.zgrid),
        }
    )
    state.attrs.update(
        schema_version=SCHEMA_VERSION,
        cache_kind="native_mean_state",
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        vertical_grid="native MPAS model levels",
        time_averaging="trapezoidal mean over the endpoint storage interval",
    )
    return state


def _time_batches(ds, batch_size=TIME_BATCH_SIZE):
    """Yield unique core time samples with a two-sample diagnostic halo."""
    time_size = ds.sizes["Time"]
    if batch_size < 1:
        raise ValueError("time batch size must be positive")
    for start in range(0, time_size, batch_size):
        stop = min(start + batch_size, time_size)
        halo_start = max(0, start - TIME_HALO_POINTS)
        halo_stop = min(time_size, stop + TIME_HALO_POINTS)
        yield (
            start,
            stop,
            halo_start,
            ds.isel(Time=slice(halo_start, halo_stop)),
        )


def _weighted_time_parts(field, weights):
    """Weighted numerator and valid-data denominator for one time block."""
    valid = np.isfinite(field)
    numerator = (field.where(valid, 0.0) * weights).sum(
        "Time", skipna=True
    )
    denominator = weights.where(valid).sum("Time", skipna=True)
    return numerator, denominator


def _add_weighted_statistic(statistics, name, field, weights):
    numerator, denominator = _weighted_time_parts(field, weights)
    statistics[f"{name}__weighted_sum"] = numerator
    statistics[f"{name}__weight_sum"] = denominator


def _time_batch_statistics(
    subset, global_weights, core_start, core_stop, halo_start,
    core_latitude=None,
):
    """Sufficient statistics for exact reconstruction of all time means."""
    tropopauses = diagnose_tropopauses(subset)
    instantaneous = instantaneous_budget(
        subset, tropopauses, core_latitude=core_latitude
    )
    analysis_subset = _core_latitude_subset(subset, core_latitude)

    local_start = core_start - halo_start
    local_stop = local_start + (core_stop - core_start)
    core_slice = slice(local_start, local_stop)
    weights = global_weights.sel(Time=subset.Time).copy()
    retain = np.zeros(weights.size, dtype=bool)
    retain[core_slice] = True
    weights = weights.where(
        xr.DataArray(retain, dims="Time", coords={"Time": subset.Time}),
        0.0,
    )

    mixing_ratio = finite_or_nan(analysis_subset.qv)
    density = finite_or_nan(analysis_subset.rho)
    zonal_wind = finite_or_nan(analysis_subset.uReconstructZonal)
    meridional_wind = finite_or_nan(
        analysis_subset.uReconstructMeridional
    )
    temperature = finite_or_nan(
        analysis_subset.theta
        * (analysis_subset.pressure / P0_PA) ** (RD / CP)
    )
    specific_humidity = mixing_ratio_to_specific_humidity(mixing_ratio)
    relative_humidity = archived_relative_humidity_fraction(analysis_subset)

    statistics = xr.Dataset()
    sources = {
        "column_vapour": instantaneous.column_vapour,
        "ivt_u": instantaneous.ivt_u,
        "ivt_v": instantaneous.ivt_v,
        "top_influx": instantaneous.top_influx,
        "top_influx_24h": instantaneous.top_influx_24h,
        "boundary_pressure": instantaneous.boundary_pressure,
        "boundary_height": instantaneous.boundary_height,
        "boundary_temperature": instantaneous.boundary_temperature,
        "top_vertical_velocity_component": (
            instantaneous.top_vertical_velocity_component
        ),
        "top_local_motion_component": (
            instantaneous.top_local_motion_component
        ),
        "top_horizontal_motion_component": (
            instantaneous.top_horizontal_motion_component
        ),
        "full_column_vapour": instantaneous.full_column_vapour,
        "evaporation": finite_or_nan(analysis_subset.lh) / LV,
        "archived_precipw": finite_or_nan(analysis_subset.precipw),
        "mixing_ratio": mixing_ratio,
        "density": density,
        "pressure": finite_or_nan(analysis_subset.pressure),
        "temperature": temperature,
        "zonal_wind": zonal_wind,
        "meridional_wind": meridional_wind,
        "vapour_density": density * mixing_ratio,
        "zonal_vapour_flux": density * mixing_ratio * zonal_wind,
        "meridional_vapour_flux": density * mixing_ratio * meridional_wind,
        "specific_humidity_zonal": specific_humidity.mean(
            "longitude", skipna=True
        ),
        "relative_humidity_zonal": relative_humidity.mean(
            "longitude", skipna=True
        ),
    }
    for name, field in sources.items():
        _add_weighted_statistic(statistics, name, field, weights)

    statistics["cold_point_valid_count"] = np.isfinite(
        _core_latitude_subset(
            tropopauses.cold_point_pressure, core_latitude
        ).isel(Time=core_slice)
    ).sum("Time")
    statistics["top_flux_valid_count"] = (
        instantaneous.top_flux_surface_valid.isel(Time=core_slice)
        .astype(np.int64)
        .sum("Time")
    )
    statistics["sample_count"] = xr.DataArray(core_stop - core_start)
    return statistics


def _accumulate_statistics_in_place(total, increment):
    """Add one computed time block without allocating a third full dataset."""
    if set(total.data_vars) != set(increment.data_vars):
        raise ValueError("time-batch statistic variables do not match")
    for name in total.data_vars:
        left = total[name]
        right = increment[name]
        if left.dims != right.dims or left.shape != right.shape:
            raise ValueError(
                f"time-batch statistic {name!r} has inconsistent geometry"
            )
        np.add(left.data, right.data, out=left.data)
    return total


def _endpoint_statistics(ds, core_latitude=None):
    """Only the exact first/last fields required by interval tendencies."""
    endpoints = ds.isel(Time=[0, -1])
    tropopauses = diagnose_tropopauses(endpoints)
    _, top_height, _ = boundary_arrays(endpoints, tropopauses)
    endpoints = _core_latitude_subset(endpoints, core_latitude)
    top_height = _core_latitude_subset(top_height, core_latitude)
    vapour_density = (
        finite_or_nan(endpoints.rho) * finite_or_nan(endpoints.qv)
    )
    column_vapour = native_bounded_height_integral(
        vapour_density, endpoints, top_height
    )

    layer_coord = endpoints.qv.nVertLevels
    lower_interface = endpoints.zgrid.isel(
        nVertLevelsP1=slice(0, -1)
    ).rename({"nVertLevelsP1": "nVertLevels"}).assign_coords(
        nVertLevels=layer_coord
    )
    upper_interface = endpoints.zgrid.isel(
        nVertLevelsP1=slice(1, None)
    ).rename({"nVertLevelsP1": "nVertLevels"}).assign_coords(
        nVertLevels=layer_coord
    )
    condensate = sum(
        finite_or_nan(endpoints[name])
        for name in ("qc", "qr", "qi", "qs", "qg")
    )
    condensate_column = (
        finite_or_nan(endpoints.rho)
        * condensate
        * np.abs(upper_interface - lower_interface)
    ).sum("nVertLevels", skipna=True)
    return xr.Dataset(
        {
            "column_vapour_endpoint": column_vapour,
            "condensate_column_endpoint": condensate_column,
            "precipitation_accum_endpoint": total_precip_accum(endpoints),
            "zgrid": finite_or_nan(endpoints.zgrid),
        }
    )


def _means_from_statistics(statistics):
    names = {
        name.removesuffix("__weighted_sum")
        for name in statistics.data_vars
        if name.endswith("__weighted_sum")
    }
    return {
        name: (
            statistics[f"{name}__weighted_sum"]
            / statistics[f"{name}__weight_sum"].where(
                statistics[f"{name}__weight_sum"] > 0.0
            )
        )
        for name in names
    }


def _set_product_metadata(products, ds, simulation, elapsed_seconds):
    for name in (
        "column_vapour_mean", "full_column_vapour_mean",
        "archived_precipw_mean", "precipw_integration_bias",
    ):
        products[name].attrs["units"] = "kg m-2"
    for name in ("relative_precipw_error", "budget_closure_fraction"):
        products[name].attrs["units"] = "1"
    for name in (
        "ivt_u_mean", "ivt_v_mean", "fixed_mean_flow_ivt_u",
        "fixed_mean_flow_ivt_v", "fixed_transient_ivt_u",
        "fixed_transient_ivt_v",
    ):
        products[name].attrs["units"] = "kg m-1 s-1"
    for name in (
        "top_vertical_velocity_component", "top_local_motion_component",
        "top_horizontal_motion_component",
    ):
        products[name].attrs["units"] = "mm day-1"
    products.native_temperature_zonal_mean.attrs["units"] = "K"
    products.native_specific_humidity_zonal_mean.attrs["units"] = "kg kg-1"
    products.native_relative_humidity_zonal_mean.attrs["units"] = "1"
    products.native_relative_humidity_zonal_mean.attrs[
        "source_variable"
    ] = "relhum"
    products.native_pressure_zonal_mean.attrs["units"] = "Pa"
    products.native_midlevel_height_zonal_mean.attrs["units"] = "m"
    products.top_flux_valid_fraction.attrs["units"] = "1"
    products.attrs.update(
        schema_version=SCHEMA_VERSION,
        cache_kind="mean_products",
        simulation=simulation,
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        budget_equation=(
            "storage = evaporation - precipitation + moisture_convergence "
            "+ top_influx + residual"
        ),
        top_influx_sign="positive downward into the diagnosed column",
        column_integration=(
            "native MPAS model levels: sum(rho_d * qv_dry_mixing_ratio * dz), "
            "with a fractional layer at the upper boundary"
        ),
        ivt_integration=(
            "native MPAS model levels: sum(rho_d * qv_dry_mixing_ratio * "
            "horizontal_wind * dz), with a fractional upper layer"
        ),
        top_flux_method=(
            "native height coordinate: rho_d*qv*(dz_top/dt + "
            "u_top*grad_x(z_top) + v_top*grad_y(z_top) - w_top)"
        ),
        top_flux_surface_filter=(
            "moving-tropopause flux is retained only where the surface is "
            "finite and adjacent time, latitude, and longitude tropopause "
            f"heights differ by no more than {MAX_TROPOPAUSE_SURFACE_JUMP_M:g} m"
        ),
        closure_fraction_definition=(
            "abs(residual) / (abs(storage) + abs(evaporation) + "
            "abs(precipitation) + abs(moisture_convergence) + abs(top_influx))"
        ),
        horizontal_grid_note=(
            "native vertical levels retained, but fields were archived on the "
            "regular latitude-longitude analysis grid"
        ),
        time_averaging=(
            "global trapezoidal sample weights accumulated in bounded time "
            "blocks; precipitation and storage from exact endpoints"
        ),
        sampling_interval_hours=float(
            np.median(np.diff(ds.Time.values) / np.timedelta64(1, "h"))
        ),
        averaging_window_days=float(elapsed_seconds / SECONDS_PER_DAY),
    )
    return products


def _products_and_state_from_statistics(
    statistics, endpoints, ds, simulation,
):
    means = _means_from_statistics(statistics)
    elapsed_seconds = float(
        (ds.Time.values[-1] - ds.Time.values[0]) / np.timedelta64(1, "s")
    )
    storage = (
        endpoints.column_vapour_endpoint.isel(Time=-1)
        - endpoints.column_vapour_endpoint.isel(Time=0)
    ) / elapsed_seconds
    precipitation = (
        endpoints.precipitation_accum_endpoint.isel(Time=-1)
        - endpoints.precipitation_accum_endpoint.isel(Time=0)
    ) / elapsed_seconds
    condensate_storage = (
        endpoints.condensate_column_endpoint.isel(Time=-1)
        - endpoints.condensate_column_endpoint.isel(Time=0)
    ) / elapsed_seconds
    # Horizontal convergence is deliberately finalised only after all
    # archive-aligned latitude cores have been concatenated. Computing it here
    # would either use one-sided derivatives at every internal tile edge or
    # require loading all native 3-D fields in the derivative halos.
    convergence = xr.zeros_like(storage)
    residual = storage - (
        means["evaporation"] - precipitation
        + convergence + means["top_influx"]
    )
    conventional_residual = storage - (
        means["evaporation"] - precipitation + convergence
    )
    closure_scale = (
        np.abs(storage) + np.abs(means["evaporation"])
        + np.abs(precipitation) + np.abs(convergence)
        + np.abs(means["top_influx"])
    )
    closure_fraction = xr.where(
        closure_scale > 0.0, np.abs(residual) / closure_scale, np.nan
    )

    state = xr.Dataset(
        {
            "mixing_ratio": means["mixing_ratio"],
            "density": means["density"],
            "pressure": means["pressure"],
            "temperature": means["temperature"],
            "zonal_wind": means["zonal_wind"],
            "meridional_wind": means["meridional_wind"],
            "vapour_density": means["vapour_density"],
            "zonal_vapour_flux": means["zonal_vapour_flux"],
            "meridional_vapour_flux": means["meridional_vapour_flux"],
            "zgrid": endpoints.zgrid,
        }
    )
    state.attrs.update(
        schema_version=SCHEMA_VERSION,
        cache_kind="native_mean_state",
        simulation=simulation,
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        vertical_grid="native MPAS model levels",
        time_averaging=(
            "global trapezoidal sample weights accumulated in bounded time blocks"
        ),
    )

    fixed_top_height = means["boundary_height"].sel(
        boundary="fixed_200hpa", drop=True
    )
    mean_flow_u = native_bounded_height_integral(
        state.vapour_density * state.zonal_wind, state, fixed_top_height
    )
    mean_flow_v = native_bounded_height_integral(
        state.vapour_density * state.meridional_wind, state, fixed_top_height
    )
    full_column_vapour = means["full_column_vapour"]
    precipw_bias = full_column_vapour - means["archived_precipw"]
    relative_precipw_error = xr.where(
        np.abs(means["archived_precipw"]) > 0.0,
        np.abs(precipw_bias) / np.abs(means["archived_precipw"]),
        np.nan,
    )
    native_height_zonal = native_midlevel_height(
        state, state.temperature
    ).mean("longitude", skipna=True)
    sample_count = statistics.sample_count

    rates = xr.Dataset(
        {
            "storage": storage * SECONDS_PER_DAY,
            "moisture_convergence": convergence * SECONDS_PER_DAY,
            "evaporation": means["evaporation"] * SECONDS_PER_DAY,
            "precipitation": precipitation * SECONDS_PER_DAY,
            "top_influx": means["top_influx"] * SECONDS_PER_DAY,
            "top_influx_24h": means["top_influx_24h"] * SECONDS_PER_DAY,
            "residual": residual * SECONDS_PER_DAY,
            "conventional_residual": (
                conventional_residual * SECONDS_PER_DAY
            ),
            "condensate_storage": condensate_storage * SECONDS_PER_DAY,
        }
    )
    for variable in rates.data_vars:
        rates[variable].attrs["units"] = "mm day-1"

    products = xr.merge(
        [
            rates,
            xr.Dataset(
                {
                    "column_vapour_mean": means["column_vapour"],
                    "ivt_u_mean": means["ivt_u"],
                    "ivt_v_mean": means["ivt_v"],
                    "boundary_pressure_mean": means["boundary_pressure"],
                    "boundary_height_mean": means["boundary_height"],
                    "boundary_temperature_mean": means[
                        "boundary_temperature"
                    ],
                    "cold_point_valid_fraction": (
                        statistics.cold_point_valid_count / sample_count
                    ),
                    "top_vertical_velocity_component": (
                        means["top_vertical_velocity_component"]
                        * SECONDS_PER_DAY
                    ),
                    "top_local_motion_component": (
                        means["top_local_motion_component"] * SECONDS_PER_DAY
                    ),
                    "top_horizontal_motion_component": (
                        means["top_horizontal_motion_component"]
                        * SECONDS_PER_DAY
                    ),
                    "full_column_vapour_mean": full_column_vapour,
                    "archived_precipw_mean": means["archived_precipw"],
                    "precipw_integration_bias": precipw_bias,
                    "relative_precipw_error": relative_precipw_error,
                    "budget_closure_fraction": closure_fraction,
                    "fixed_mean_flow_ivt_u": mean_flow_u,
                    "fixed_mean_flow_ivt_v": mean_flow_v,
                    "fixed_transient_ivt_u": (
                        means["ivt_u"].sel(
                            boundary="fixed_200hpa", drop=True
                        ) - mean_flow_u
                    ),
                    "fixed_transient_ivt_v": (
                        means["ivt_v"].sel(
                            boundary="fixed_200hpa", drop=True
                        ) - mean_flow_v
                    ),
                    "native_temperature_zonal_mean": (
                        means["temperature"].mean("longitude", skipna=True)
                    ),
                    "native_specific_humidity_zonal_mean": (
                        means["specific_humidity_zonal"]
                    ),
                    "native_relative_humidity_zonal_mean": (
                        means["relative_humidity_zonal"]
                    ),
                    "native_pressure_zonal_mean": (
                        means["pressure"].mean("longitude", skipna=True)
                    ),
                    "native_midlevel_height_zonal_mean": native_height_zonal,
                    "native_midlevel_height": lat_mean(native_height_zonal),
                    "top_flux_valid_fraction": (
                        statistics.top_flux_valid_count / sample_count
                    ),
                }
            ),
        ],
        compat="override",
    )
    products = _set_product_metadata(
        products, ds, simulation, elapsed_seconds
    )
    products.attrs["horizontal_budget_finalized"] = 0
    return products, state


def _finalize_horizontal_budget(products):
    """Diagnose convergence and closure once on the complete latitude grid."""
    if products.chunks is not None:
        products = products.chunk({"latitude": -1})
    convergence = -spherical_flux_divergence(
        products.ivt_u_mean, products.ivt_v_mean
    ) * SECONDS_PER_DAY
    residual = products.storage - (
        products.evaporation - products.precipitation
        + convergence + products.top_influx
    )
    conventional_residual = products.storage - (
        products.evaporation - products.precipitation + convergence
    )
    closure_scale = (
        np.abs(products.storage) + np.abs(products.evaporation)
        + np.abs(products.precipitation) + np.abs(convergence)
        + np.abs(products.top_influx)
    )
    products["moisture_convergence"] = convergence
    products["residual"] = residual
    products["conventional_residual"] = conventional_residual
    products["budget_closure_fraction"] = xr.where(
        closure_scale > 0.0, np.abs(residual) / closure_scale, np.nan
    )
    for name in (
        "moisture_convergence", "residual", "conventional_residual"
    ):
        products[name].attrs["units"] = "mm day-1"
    products.budget_closure_fraction.attrs["units"] = "1"
    products.attrs["horizontal_budget_finalized"] = 1
    return products


def _latitude_batches(ds, batch_size=LATITUDE_BATCH_SIZE):
    """Yield core latitude bands with a two-point derivative halo."""
    latitude_size = ds.sizes["latitude"]
    if batch_size < 1:
        raise ValueError("latitude batch size must be positive")
    for start in range(0, latitude_size, batch_size):
        stop = min(start + batch_size, latitude_size)
        halo_start = max(0, start - LATITUDE_HALO_POINTS)
        halo_stop = min(
            latitude_size, stop + LATITUDE_HALO_POINTS
        )
        core_latitude = ds.latitude.isel(latitude=slice(start, stop))
        subset = ds.isel(latitude=slice(halo_start, halo_stop))
        # Preserve the archive's native 120-row chunks for all 3-D fields.
        # Only the diagnosed 2-D boundary height is rechunked before taking
        # latitude derivatives, so halo rows do not force every native field
        # to read an entire neighbouring storage chunk.
        yield start, stop, core_latitude, subset


def _combine_product_latitude_batches(batches):
    """Join product bands and reconstruct the sole latitude-reduced profile."""
    if not batches:
        raise ValueError("at least one product latitude batch is required")
    height_attrs = batches[0].native_midlevel_height.attrs.copy()
    without_global_height = [
        batch.drop_vars("native_midlevel_height") for batch in batches
    ]
    combined = xr.concat(
        without_global_height,
        dim="latitude",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="exact",
    ).sortby("latitude")
    combined["native_midlevel_height"] = lat_mean(
        combined.native_midlevel_height_zonal_mean
    )
    combined.native_midlevel_height.attrs.update(height_attrs)
    return combined


def compute_latitude_slab_in_time_batches(
    ds, simulation, time_batch_size=TIME_BATCH_SIZE, core_latitude=None,
):
    """Compute one spatial slab with a bounded graph for each time block."""
    global_weights = trapezoidal_time_weights(ds.Time)
    statistics = None
    number_of_batches = (
        ds.sizes["Time"] + time_batch_size - 1
    ) // time_batch_size
    for number, (
        core_start, core_stop, halo_start, subset,
    ) in enumerate(
        _time_batches(ds, batch_size=time_batch_size), start=1
    ):
        if number == 1 or number == number_of_batches or number % 4 == 0:
            progress(
                f"{simulation}: time batch {number}/{number_of_batches} "
                f"({str(ds.Time.values[core_start])[:10]} to "
                f"{str(ds.Time.values[core_stop - 1])[:10]})"
            )
        lazy_statistics = _time_batch_statistics(
            subset,
            global_weights,
            core_start,
            core_stop,
            halo_start,
            core_latitude=core_latitude,
        )
        computed = bounded_compute(lazy_statistics)[0]
        statistics = (
            computed
            if statistics is None
            else _accumulate_statistics_in_place(statistics, computed)
        )
        del lazy_statistics, computed, subset
        gc.collect()

    endpoints = bounded_compute(
        _endpoint_statistics(ds, core_latitude=core_latitude)
    )[0]
    products, state = _products_and_state_from_statistics(
        statistics, endpoints, ds, simulation
    )
    del statistics, endpoints
    gc.collect()
    return products, state


def compute_simulation_in_latitude_batches(
    ds, simulation, need_products=True, need_state=True,
    batch_size=LATITUDE_BATCH_SIZE,
    time_batch_size=TIME_BATCH_SIZE,
):
    """Compute compact outputs in bounded latitude bands.

    A two-point halo supplies centred meridional derivatives, second-order
    domain-edge gradients, and tropopause-continuity checks. Only the
    halo-free core is retained, so the concatenated result is numerically
    equivalent to the global calculation.
    """
    product_batches = []
    state_batches = []
    number_of_batches = (
        ds.sizes["latitude"] + batch_size - 1
    ) // batch_size
    for number, (_start, _stop, core_latitude, subset) in enumerate(
        _latitude_batches(ds, batch_size=batch_size), start=1
    ):
        progress(
            f"{simulation}: latitude batch {number}/{number_of_batches} "
            f"({float(core_latitude.min()):.2f} to "
            f"{float(core_latitude.max()):.2f} degrees)"
        )
        computed_products, computed_state = (
            compute_latitude_slab_in_time_batches(
                subset,
                simulation,
                time_batch_size=time_batch_size,
                core_latitude=core_latitude,
            )
        )
        if need_products:
            product_batches.append(
                computed_products.sel(latitude=core_latitude)
            )
        if need_state:
            state_batches.append(
                computed_state.sel(latitude=core_latitude)
            )
        del computed_products, computed_state, subset
        gc.collect()

    products = (
        _finalize_horizontal_budget(
            _combine_product_latitude_batches(product_batches)
        )
        if need_products else None
    )
    state = (
        xr.concat(
            state_batches,
            dim="latitude",
            data_vars="all",
            coords="minimal",
            compat="override",
            join="exact",
        ).sortby("latitude")
        if need_state else None
    )
    if products is not None:
        products.attrs["memory_strategy"] = (
            f"{batch_size}-point latitude batches with "
            f"{LATITUDE_HALO_POINTS}-point derivative halos; "
            f"{time_batch_size}-sample time batches with "
            f"{TIME_HALO_POINTS}-sample tendency halos"
        )
    if state is not None:
        state.attrs["memory_strategy"] = (
            f"{batch_size}-point latitude batches with "
            f"{LATITUDE_HALO_POINTS}-point derivative halos; "
            f"{time_batch_size}-sample time batches with "
            f"{TIME_HALO_POINTS}-sample tendency halos"
        )
    return products, state


def build_simulation_caches_on_disk(
    ds, simulation, product_path, state_path,
    need_products=True, need_state=True,
    latitude_batch_size=LATITUDE_BATCH_SIZE,
    time_batch_size=TIME_BATCH_SIZE,
    source_manifest_sha256=None,
):
    """Spill completed latitude slabs and assemble final caches from disk."""
    working_root = os.path.dirname(
        product_path if need_products else state_path
    )
    os.makedirs(working_root, exist_ok=True)
    try:
        safe_simulation = simulation_cache_name(simulation)
    except KeyError:
        # Synthetic datasets also use this bounded disk-backed calculation.
        safe_simulation = simulation.replace(" ", "_")
    product_parts = []
    state_parts = []
    number_of_batches = (
        ds.sizes["latitude"] + latitude_batch_size - 1
    ) // latitude_batch_size
    temporary_dir = os.path.join(
        working_root,
        (
            f".{safe_simulation}_vapour_parts_schema{SCHEMA_VERSION}"
            f"_layout{EXECUTION_LAYOUT_VERSION}"
            f"_lat{latitude_batch_size}_time{time_batch_size}"
        ),
    )
    os.makedirs(temporary_dir, exist_ok=True)
    completed = False
    try:
        for number, (
            _start, _stop, core_latitude, subset,
        ) in enumerate(
            _latitude_batches(ds, batch_size=latitude_batch_size), start=1
        ):
            product_part = os.path.join(
                temporary_dir, f"products_{number:03d}.nc"
            )
            state_part = os.path.join(
                temporary_dir, f"state_{number:03d}.nc"
            )
            product_ready = (
                need_products
                and cache_is_current(
                    product_part, simulation, "mean_products"
                )
            )
            state_ready = (
                need_state
                and cache_is_current(
                    state_part, simulation, "native_mean_state"
                )
            )
            if (not need_products or product_ready) and (
                not need_state or state_ready
            ):
                progress(
                    f"{simulation}: reusing completed latitude batch "
                    f"{number}/{number_of_batches}"
                )
                if need_products:
                    product_parts.append(product_part)
                if need_state:
                    state_parts.append(state_part)
                continue
            progress(
                f"{simulation}: latitude batch {number}/{number_of_batches} "
                f"({float(core_latitude.min()):.2f} to "
                f"{float(core_latitude.max()):.2f} degrees)"
            )
            products, state = compute_latitude_slab_in_time_batches(
                subset,
                simulation,
                time_batch_size=time_batch_size,
                core_latitude=core_latitude,
            )
            if need_products:
                if not product_ready:
                    write_dataset_cache(
                        products.sel(latitude=core_latitude),
                        product_part,
                        f"latitude-batch {number} product",
                    )
                product_parts.append(product_part)
            if need_state:
                if not state_ready:
                    write_dataset_cache(
                        state.sel(latitude=core_latitude),
                        state_part,
                        f"latitude-batch {number} state",
                    )
                state_parts.append(state_part)
            del products, state, subset
            gc.collect()

        result_products = None
        if need_products:
            opened_products = [
                xr.open_dataset(path, chunks=STATE_CHUNKS)
                for path in product_parts
            ]
            try:
                combined_products = _combine_product_latitude_batches(
                    opened_products
                )
                combined_products = _finalize_horizontal_budget(
                    combined_products
                )
                combined_products.attrs["memory_strategy"] = (
                    f"disk-backed {latitude_batch_size}-point latitude "
                    f"batches and {time_batch_size}-sample time batches"
                )
                if source_manifest_sha256 is not None:
                    combined_products.attrs["source_manifest_sha256"] = (
                        source_manifest_sha256
                    )
                write_dataset_cache(
                    combined_products,
                    product_path,
                    "vapour-budget product",
                )
            finally:
                for dataset in opened_products:
                    dataset.close()
            with xr.open_dataset(product_path) as cached:
                result_products = cached.load()

        if need_state:
            opened_states = [
                xr.open_dataset(path, chunks=STATE_CHUNKS)
                for path in state_parts
            ]
            try:
                combined_state = xr.concat(
                    opened_states,
                    dim="latitude",
                    data_vars="all",
                    coords="minimal",
                    compat="override",
                    join="exact",
                ).sortby("latitude")
                combined_state.attrs["memory_strategy"] = (
                    f"disk-backed {latitude_batch_size}-point latitude "
                    f"batches and {time_batch_size}-sample time batches"
                )
                if source_manifest_sha256 is not None:
                    combined_state.attrs["source_manifest_sha256"] = (
                        source_manifest_sha256
                    )
                write_dataset_cache(
                    combined_state,
                    state_path,
                    "native mean-state",
                )
            finally:
                for dataset in opened_states:
                    dataset.close()
        completed = True
        return result_products
    finally:
        if completed:
            shutil.rmtree(temporary_dir)


def prepare_simulation(
    data_loc,
    simulation,
    force=False,
    test=False,
    test_times=5,
    require_current=False,
):
    """Build one simulation's compact caches, then release its raw history."""
    product_path = cache_path(simulation, test=test)
    state_path = state_cache_path(simulation, test=test)
    manifest = (
        None
        if require_current
        else simulation_cache_manifest(
            data_loc,
            simulation,
            test=test,
            test_times=test_times,
        )
    )
    manifest_sha256 = (
        None if manifest is None else manifest["manifest_sha256"]
    )
    product_current = (
        not force
        and cache_is_current(
            product_path,
            simulation,
            "mean_products",
            expected_manifest_sha256=manifest_sha256,
        )
    )
    state_current = (
        not force
        and cache_is_current(
            state_path,
            simulation,
            "native_mean_state",
            expected_manifest_sha256=manifest_sha256,
        )
    )
    if require_current and not (product_current and state_current):
        missing = []
        if not product_current:
            missing.append(product_path)
        if not state_current:
            missing.append(state_path)
        raise FileNotFoundError(
            "Figure-only mode requires current vapour caches:\n"
            + "\n".join(f"  {path}" for path in missing)
        )

    if product_current:
        progress(f"{simulation}: using product cache {product_path}")
        with xr.open_dataset(product_path) as cached:
            products = cached.load()
    else:
        products = None

    if product_current and state_current:
        progress(f"{simulation}: using native-state cache {state_path}")
        return products

    ds = load_simulation_dataset(
        data_loc, test=test, test_times=test_times
    )
    try:
        progress(
            f"{simulation}: computing latitude-batched compact caches with "
            f"{DASK_NUM_WORKERS} Dask workers"
        )
        computed_products = build_simulation_caches_on_disk(
            ds,
            simulation,
            product_path,
            state_path,
            need_products=not product_current,
            need_state=not state_current,
            source_manifest_sha256=manifest_sha256,
        )
        if not product_current:
            products = computed_products
    finally:
        ds.close()
        del ds
        gc.collect()
    return products


def _attribution_profiles(state0, state1):
    state0, state1 = xr.align(state0, state1, join="exact")
    r0 = state0.mixing_ratio
    r1 = state1.mixing_ratio
    rho0 = state0.density
    rho1 = state1.density
    u0 = state0.zonal_wind
    u1 = state1.zonal_wind
    v0 = state0.meridional_wind
    v1 = state1.meridional_wind
    pressure_bar = 0.5 * (state0.pressure + state1.pressure)
    rsat0 = saturation_mixing_ratio(state0.temperature, pressure_bar)
    rsat1 = saturation_mixing_ratio(state1.temperature, pressure_bar)
    rh0 = r0 / rsat0
    rh1 = r1 / rsat1

    dr_temperature = 0.5 * (rh0 + rh1) * (rsat1 - rsat0)
    dr_rh = 0.5 * (rsat0 + rsat1) * (rh1 - rh0)
    rbar = 0.5 * (r0 + r1)
    rhobar = 0.5 * (rho0 + rho1)
    ubar = 0.5 * (u0 + u1)
    vbar = 0.5 * (v0 + v1)

    covariance_u0 = state0.zonal_vapour_flux - rho0 * r0 * u0
    covariance_u1 = state1.zonal_vapour_flux - rho1 * r1 * u1
    covariance_v0 = state0.meridional_vapour_flux - rho0 * r0 * v0
    covariance_v1 = state1.meridional_vapour_flux - rho1 * r1 * v1
    covariance_column0 = state0.vapour_density - rho0 * r0
    covariance_column1 = state1.vapour_density - rho1 * r1

    flux_u_profiles = {
        "temperature": rhobar * ubar * dr_temperature,
        "relative_humidity": rhobar * ubar * dr_rh,
        "wind": rhobar * rbar * (u1 - u0),
        "density": 0.5 * (r0 * u0 + r1 * u1) * (rho1 - rho0),
        "transient": covariance_u1 - covariance_u0,
    }
    flux_v_profiles = {
        "temperature": rhobar * vbar * dr_temperature,
        "relative_humidity": rhobar * vbar * dr_rh,
        "wind": rhobar * rbar * (v1 - v0),
        "density": 0.5 * (r0 * v0 + r1 * v1) * (rho1 - rho0),
        "transient": covariance_v1 - covariance_v0,
    }
    column_profiles = {
        "temperature": rhobar * dr_temperature,
        "relative_humidity": rhobar * dr_rh,
        "wind": xr.zeros_like(rbar),
        "density": rbar * (rho1 - rho0),
        "transient": covariance_column1 - covariance_column0,
    }
    geometry = xr.Dataset({"zgrid": 0.5 * (state0.zgrid + state1.zgrid)})
    return flux_u_profiles, flux_v_profiles, column_profiles, geometry


def _integrated_attribution_component(
    profile_u, profile_v, column_profile, geometry, control, experiment,
):
    boundary_datasets = []
    for boundary in BOUNDARIES:
        top_height = 0.5 * (
            control.boundary_height_mean.sel(boundary=boundary, drop=True)
            + experiment.boundary_height_mean.sel(
                boundary=boundary, drop=True
            )
        )
        component_u = native_bounded_height_integral(
            profile_u, geometry, top_height
        )
        component_v = native_bounded_height_integral(
            profile_v, geometry, top_height
        )
        component_column = native_bounded_height_integral(
            column_profile, geometry, top_height
        )
        boundary_datasets.append(
            xr.Dataset(
                {
                    "ivt_change_u": component_u,
                    "ivt_change_v": component_v,
                    "convergence_change": (
                        -spherical_flux_divergence(component_u, component_v)
                        * SECONDS_PER_DAY
                    ),
                    "column_vapour_change": component_column,
                }
            )
        )
    return xr.concat(
        boundary_datasets,
        dim=xr.IndexVariable("boundary", list(BOUNDARIES)),
    )


def _actual_attribution_change(control, experiment):
    boundary_datasets = []
    for boundary in BOUNDARIES:
        actual_u = (
            experiment.ivt_u_mean.sel(boundary=boundary, drop=True)
            - control.ivt_u_mean.sel(boundary=boundary, drop=True)
        )
        actual_v = (
            experiment.ivt_v_mean.sel(boundary=boundary, drop=True)
            - control.ivt_v_mean.sel(boundary=boundary, drop=True)
        )
        boundary_datasets.append(
            xr.Dataset(
                {
                    "ivt_change_u": actual_u,
                    "ivt_change_v": actual_v,
                    "convergence_change": (
                        -spherical_flux_divergence(actual_u, actual_v)
                        * SECONDS_PER_DAY
                    ),
                    "column_vapour_change": (
                        experiment.column_vapour_mean.sel(
                            boundary=boundary, drop=True
                        )
                        - control.column_vapour_mean.sel(
                            boundary=boundary, drop=True
                        )
                    ),
                }
            )
        )
    return xr.concat(
        boundary_datasets,
        dim=xr.IndexVariable("boundary", list(BOUNDARIES)),
    )


def _assemble_attribution(base_components, actual):
    explained = xr.concat(
        [
            component.expand_dims(component=[name])
            for name, component in zip(COMPONENT_NAMES[:5], base_components)
        ],
        dim="component",
    )
    geometry = (actual - explained.sum("component")).expand_dims(
        component=["geometry"]
    )
    total = actual.expand_dims(component=["total"])
    return xr.concat([explained, geometry, total], dim="component").sel(
        component=list(COMPONENT_NAMES)
    )


def transport_attribution(states, products):
    """Attribute changes from native mean states; remains lazy when chunked."""
    if len(states) != len(products):
        raise ValueError("states and products must contain the same simulations")
    if "mixing_ratio" not in states[0]:
        states = [native_mean_state(ds) for ds in states]
    comparison_datasets = []
    for index, experiment in enumerate(products[1:], start=1):
        profiles_u, profiles_v, columns, geometry = _attribution_profiles(
            states[0], states[index]
        )
        base_components = [
            _integrated_attribution_component(
                profiles_u[name], profiles_v[name], columns[name], geometry,
                products[0], experiment,
            )
            for name in COMPONENT_NAMES[:5]
        ]
        comparison_datasets.append(
            _assemble_attribution(
                base_components,
                _actual_attribution_change(products[0], experiment),
            )
        )
    attribution = xr.concat(
        comparison_datasets,
        dim=xr.IndexVariable("comparison", list(COMPARISON_NAMES)),
    )
    attribution.convergence_change.attrs["units"] = "mm day-1"
    attribution.column_vapour_change.attrs["units"] = "kg m-2"
    attribution.attrs.update(
        vertical_grid="native MPAS model levels",
        decomposition=(
            "exact symmetric decomposition of time-mean rho_d*r_v*wind; "
            "geometry is the residual from boundary and layer geometry"
        ),
    )
    return attribution


def compute_cached_transport_attribution(products, test=False):
    """Compute one comparison and one physical component at a time."""
    comparison_datasets = []
    control_path = state_cache_path(SIM_NAMES[0], test=test)
    with xr.open_dataset(control_path, chunks=STATE_CHUNKS) as control_state:
        for index, comparison_name in enumerate(COMPARISON_NAMES, start=1):
            experiment_path = state_cache_path(SIM_NAMES[index], test=test)
            progress(f"{comparison_name}: opening cached native mean states")
            with xr.open_dataset(
                experiment_path, chunks=STATE_CHUNKS
            ) as experiment_state:
                profiles_u, profiles_v, columns, geometry = (
                    _attribution_profiles(control_state, experiment_state)
                )
                base_components = []
                for component in COMPONENT_NAMES[:5]:
                    progress(
                        f"{comparison_name}: computing {component} attribution"
                    )
                    lazy_component = _integrated_attribution_component(
                        profiles_u[component],
                        profiles_v[component],
                        columns[component],
                        geometry,
                        products[0],
                        products[index],
                    )
                    base_components.append(
                        bounded_compute(lazy_component)[0]
                    )
                    del lazy_component
                    gc.collect()
                comparison_datasets.append(
                    _assemble_attribution(
                        base_components,
                        _actual_attribution_change(
                            products[0], products[index]
                        ),
                    ).expand_dims(comparison=[comparison_name])
                )
            gc.collect()
    attribution = xr.concat(comparison_datasets, dim="comparison").sel(
        comparison=list(COMPARISON_NAMES)
    )
    attribution.convergence_change.attrs["units"] = "mm day-1"
    attribution.column_vapour_change.attrs["units"] = "kg m-2"
    attribution.attrs.update(
        schema_version=SCHEMA_VERSION,
        cache_kind="transport_attribution",
        manifest_sha256=attribution_cache_manifest(test=test)[
            "manifest_sha256"
        ],
        cache_manifest=json.dumps(
            attribution_cache_manifest(test=test), sort_keys=True
        ),
        vertical_grid="native MPAS model levels",
        decomposition=(
            "exact symmetric decomposition of time-mean rho_d*r_v*wind; "
            "geometry is the residual from boundary and layer geometry"
        ),
        memory_strategy=(
            "cached native mean states; one experiment and one physical "
            "component computed at a time"
        ),
    )
    return attribution


def deep_tropical_mean(da, bounds=(-10.0, 10.0)):
    subset = da.sel(latitude=slice(*bounds))
    dims = [dim for dim in ("latitude", "longitude") if dim in subset.dims]
    weights = np.cos(np.deg2rad(subset.latitude))
    return subset.weighted(weights).mean(dims, skipna=True)


def direct_budget_responses(products):
    """Return direct experiment-minus-control budget responses."""
    terms = (
        "storage", "evaporation", "precipitation", "moisture_convergence",
        "top_influx", "condensate_storage", "residual",
    )
    boundary_arrays = []
    for boundary in BOUNDARIES:
        term_arrays = []
        for term in terms:
            control = products[0][term]
            if "boundary" in control.dims:
                control = control.sel(boundary=boundary, drop=True)
            deltas = []
            for experiment in products[1:]:
                experiment_field = experiment[term]
                if "boundary" in experiment_field.dims:
                    experiment_field = experiment_field.sel(
                        boundary=boundary, drop=True
                    )
                deltas.append(experiment_field - control)
            response_fields = list(deltas)
            response_fields.extend(
                deltas[0] - delta for delta in deltas[1:]
            )
            response_names = list(COMPARISON_NAMES)
            response_names.extend(
                f"{DATA_LOCS[1]}_minus_{directory}"
                for directory in DATA_LOCS[2:]
            )
            term_arrays.append(
                xr.concat(
                    response_fields,
                    dim=xr.IndexVariable("response", response_names),
                )
            )
        boundary_arrays.append(
            xr.concat(term_arrays, dim=xr.IndexVariable("budget_term", list(terms)))
        )
    out = xr.concat(
        boundary_arrays, dim=xr.IndexVariable("boundary", list(BOUNDARIES))
    )
    out.name = "budget_term_response"
    out.attrs.update(
        units="mm day-1",
        response_definition=(
            "direct experiment differences; no scalar thermal scaling"
        ),
    )
    return out


def summary_table(products, attribution):
    rows = []
    terms = (
        "storage", "evaporation", "precipitation", "moisture_convergence",
        "top_influx", "condensate_storage", "residual",
    )
    for simulation, product in zip(SIM_NAMES, products):
        for boundary in BOUNDARIES:
            for term in terms:
                field = product[term]
                if "boundary" in field.dims:
                    field = field.sel(boundary=boundary, drop=True)
                rows.append(
                    {
                        "simulation": simulation,
                        "boundary": boundary,
                        "term": term,
                        "global_mean_mm_day": float(lat_mean(field.mean("longitude"))),
                        "deep_tropical_mean_mm_day": float(
                            deep_tropical_mean(field)
                        ),
                    }
                )

    budget_responses = direct_budget_responses(products)
    response_labels = list(budget_responses.response.values)
    for boundary in BOUNDARIES:
        for response in response_labels:
            for term in terms:
                response_field = budget_responses.sel(
                    boundary=boundary,
                    budget_term=term,
                    response=response,
                )
                rows.append(
                    {
                        "simulation": response,
                        "boundary": boundary,
                        "term": f"{term}_response",
                        "global_mean_mm_day": float(
                            lat_mean(response_field.mean("longitude"))
                        ),
                        "deep_tropical_mean_mm_day": float(
                            deep_tropical_mean(response_field)
                        ),
                    }
                )
        for component in COMPONENT_NAMES:
            values = {
                comparison: float(
                    deep_tropical_mean(
                        attribution.convergence_change.sel(
                            comparison=comparison,
                            boundary=boundary,
                            component=component,
                        )
                    )
                )
                for comparison in COMPARISON_NAMES
            }
            comparison_values = list(values.items())
            comparison_values.extend(
                (
                    f"{DATA_LOCS[1]}_minus_{directory}",
                    values[COMPARISON_NAMES[0]] - values[comparison],
                )
                for directory, comparison in zip(
                    DATA_LOCS[2:], COMPARISON_NAMES[1:], strict=True
                )
            )
            for comparison, value in comparison_values:
                rows.append(
                    {
                        "simulation": comparison,
                        "boundary": boundary,
                        "term": f"convergence_{component}",
                        "global_mean_mm_day": np.nan,
                        "deep_tropical_mean_mm_day": value,
                    }
                )
    return pd.DataFrame(rows)


BOUNDARY_LABELS = {
    "fixed_200hpa": "200 hPa",
    "cold_point": "cold-point tropopause",
}

BUDGET_PLOT_TERMS = (
    ("storage", 1.0, r"$\partial W/\partial t$", "storage"),
    ("evaporation", 1.0, r"$E$", "evaporation"),
    ("precipitation", -1.0, r"$-P$", "precipitation sink"),
    (
        "moisture_convergence", 1.0,
        r"$-\nabla_h\!\cdot\mathbf{IVT}$", "horizontal convergence",
    ),
    ("top_influx", 1.0, r"$F_t$", "top influx"),
    ("residual", 1.0, r"$R$", "closure residual"),
)


def plot_latitude_subset(field):
    """Select the shared 40S-40N plotting domain when latitude is present."""
    if "latitude" not in field.dims:
        return field
    return field.sel(latitude=slice(*PLOT_LATITUDE_BOUNDS))


def _budget_field(product, variable, boundary, sign=1.0):
    field = product[variable]
    if "boundary" in field.dims:
        field = field.sel(boundary=boundary, drop=True)
    return sign * field


def _symmetric_limit(fields, percentile=99.0):
    values = np.concatenate(
        [np.asarray(field.values, dtype=float).ravel() for field in fields]
    )
    vmax = np.nanpercentile(np.abs(values), percentile)
    return vmax if np.isfinite(vmax) and vmax > 0.0 else 1.0


def _colorbar_extend(image):
    return getattr(image, "vapour_budget_extend", "neither")


def map_field(ax, field, title, symmetric=True, vmin=None, vmax=None):
    field = plot_latitude_subset(field)
    values = np.asarray(field.values, dtype=float)
    if symmetric:
        if vmax is None:
            vmax = np.nanpercentile(np.abs(values), 99.0)
            vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("0.82")
        cmap.set_under(cmap(0.0))
        cmap.set_over(cmap(1.0))
        image = ax.pcolormesh(
            field.longitude, field.latitude, field,
            shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
        )
        image.vapour_budget_extend = "both"
    else:
        if vmin is None:
            vmin = np.nanpercentile(values, 1.0)
        if vmax is None:
            vmax = np.nanpercentile(values, 99.0)
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("0.82")
        cmap.set_under(cmap(0.0))
        cmap.set_over(cmap(1.0))
        image = ax.pcolormesh(
            field.longitude, field.latitude, field,
            shading="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        )
        image.vapour_budget_extend = "max"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.axhline(0.0, color="0.2", lw=0.75, ls="--", alpha=0.85)
    return image


def plot_tropopause_overview(products, figsave):
    fig = plt.figure(figsize=(15, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.10))
    axes = [
        fig.add_subplot(grid[0, column], sharey=None if column == 0 else None)
        for column in range(3)
    ]
    for axis in axes[1:]:
        axis.sharey(axes[0])
    legend_axis = fig.add_subplot(grid[1, :])
    legend_axis.axis("off")

    latitude_bounds = slice(*PLOT_LATITUDE_BOUNDS)
    control = products[0]
    background_pressure = (
        control.native_pressure_zonal_mean / 100.0
    ).sel(latitude=latitude_bounds).transpose("nVertLevels", "latitude")
    fields = (
        control.native_temperature_zonal_mean.sel(
            latitude=latitude_bounds
        ).transpose("nVertLevels", "latitude"),
        (
            1.0e6 * control.native_specific_humidity_zonal_mean
        ).sel(latitude=latitude_bounds).transpose(
            "nVertLevels", "latitude"
        ),
        (
            100.0 * control.native_relative_humidity_zonal_mean
        ).clip(0.0, 100.0).sel(latitude=latitude_bounds).transpose(
            "nVertLevels", "latitude"
        ),
    )
    latitude_2d = np.broadcast_to(
        background_pressure.latitude.values[None, :],
        background_pressure.shape,
    )
    latitude_limits = (
        max(
            PLOT_LATITUDE_BOUNDS[0],
            float(background_pressure.latitude.min()),
        ),
        min(
            PLOT_LATITUDE_BOUNDS[1],
            float(background_pressure.latitude.max()),
        ),
    )
    specifications = (
        {
            "title": "(a) Control temperature",
            "label": "Temperature (K)",
            "levels": np.arange(190.0, 257.5, 2.5),
            "cmap": "coolwarm",
            "extend": "both",
            "norm": None,
        },
        {
            "title": "(b) Control specific humidity",
            "label": "Specific humidity (mg kg$^{-1}$)",
            "levels": np.asarray(
                (1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
                 100.0, 200.0, 500.0, 1_000.0)
            ),
            "cmap": SPECIFIC_HUMIDITY_CMAP,
            "extend": "both",
            "norm": mpl_colors.LogNorm(vmin=1.0, vmax=1_000.0),
        },
        {
            "title": "(c) Control relative humidity",
            "label": "Relative humidity (%)",
            "levels": np.arange(0.0, 105.0, 5.0),
            "cmap": "BrBG",
            "extend": "neither",
            "norm": None,
        },
    )
    for ax, field, specification in zip(axes, fields, specifications):
        image = ax.contourf(
            latitude_2d,
            background_pressure,
            field,
            levels=specification["levels"],
            cmap=specification["cmap"],
            norm=specification["norm"],
            extend=specification["extend"],
        )
        fig.colorbar(
            image,
            ax=ax,
            label=specification["label"],
            extend=specification["extend"],
            pad=0.02,
            shrink=0.76,
            aspect=25,
        )
        ax.axhline(
            FIXED_TOP_PA / 100.0,
            color="0.25",
            lw=1.4,
            ls=":",
            label="fixed 200 hPa",
        )
        for simulation, color, product in zip(
            SIM_NAMES, SIM_COLORS, products
        ):
            cold_pressure = (
                product.boundary_pressure_mean.sel(
                    boundary="cold_point", drop=True
                ).mean("longitude", skipna=True)
                .sel(latitude=latitude_bounds) / 100.0
            )
            wmo_pressure = (
                wmo_tropopause_pressure_from_mean_profile(product)
                .sel(latitude=latitude_bounds) / 100.0
            )
            ax.plot(
                cold_pressure.latitude,
                cold_pressure,
                color=color,
                lw=1.5,
                ls="-",
                label=f"{simulation}: cold point",
            )
            ax.plot(
                wmo_pressure.latitude,
                wmo_pressure,
                color=color,
                lw=1.35,
                ls="--",
                label=f"{simulation}: WMO (mean profile)",
            )
        ax.set_title(specification["title"])
        ax.set_yscale("log")
        ax.set_ylim(400.0, 30.0)
        ax.set_yticks(
            (400.0, 300.0, 200.0, 100.0, 70.0, 50.0, 40.0, 30.0)
        )
        ax.set_yticklabels(
            ("400", "300", "200", "100", "70", "50", "40", "30")
        )
        ax.set_xlabel("Latitude")
        ax.set_xlim(*latitude_limits)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Pressure (hPa)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_axis.legend(
        handles,
        labels,
        loc="center",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "Zonal-mean upper-tropospheric structure and tropopause definitions"
    )
    figsave.save(fig, "tropopause_definitions", number=1)
    plt.close(fig)


def budget_change_fields(products, boundary):
    """Return the two experiment-minus-control maps for every budget term."""
    return {
        term: [
            plot_latitude_subset(
                _budget_field(experiment, term, boundary, sign)
                - _budget_field(products[0], term, boundary, sign)
            )
            for experiment in products[1:]
        ]
        for term, sign, _symbol, _description in BUDGET_PLOT_TERMS
    }


def budget_change_colour_limit(products, boundaries=BOUNDARIES):
    """Use one robust colour scale across terms, comparisons, and boundaries."""
    fields = []
    for boundary in boundaries:
        for term_fields in budget_change_fields(products, boundary).values():
            fields.extend(term_fields)
    return _symmetric_limit(fields)


def plot_budget_changes(
    products, figsave, boundary, number, shared_vmax=None,
):
    comparison_count = len(COMPARISON_NAMES)
    fig, axes = plt.subplots(
        2 * comparison_count, 6,
        figsize=(18, 6.0 * comparison_count), constrained_layout=True,
        sharey=True,
        gridspec_kw={"width_ratios": (3.6, 1.0, 3.6, 1.0, 3.6, 1.0)},
    )
    comparison_labels = COMPARISON_LABELS
    change_fields = budget_change_fields(products, boundary)
    all_fields = [
        field for fields in change_fields.values() for field in fields
    ]
    if shared_vmax is None:
        shared_vmax = _symmetric_limit(all_fields)
    map_axes = []
    shared_image = None
    for column, (term, sign, symbol, description) in enumerate(BUDGET_PLOT_TERMS):
        block_row, block_column = divmod(column, 3)
        first_row = comparison_count * block_row
        map_column = 2 * block_column
        fields = change_fields[term]
        profile_limit = _symmetric_limit(
            [field.mean("longitude", skipna=True) for field in fields],
            percentile=100.0,
        )
        for row, (label, color, field) in enumerate(
            zip(comparison_labels, COMPARISON_COLORS, fields)
        ):
            figure_row = first_row + row
            map_axis = axes[figure_row, map_column]
            profile_axis = axes[figure_row, map_column + 1]
            map_axes.append(map_axis)
            image = map_field(
                map_axis, field,
                f"{symbol}\n{description}" if row == 0 else "",
                symmetric=True, vmax=shared_vmax,
            )
            if shared_image is None:
                shared_image = image
            defined_fraction = float(np.isfinite(field).mean())
            if defined_fraction < 0.9995:
                map_axis.text(
                    0.02, 0.05,
                    f"defined: {100.0 * defined_fraction:.1f}%",
                    transform=map_axis.transAxes, va="bottom", ha="left",
                    fontsize=7, color="0.2",
                    bbox={
                        "facecolor": "white", "alpha": 0.75,
                        "edgecolor": "none",
                    },
                )
            zonal_mean = field.mean("longitude", skipna=True)
            profile_axis.plot(
                zonal_mean, zonal_mean.latitude, color=color, lw=1.4,
            )
            profile_axis.axvline(0.0, color="0.35", lw=0.7)
            profile_axis.axhline(
                0.0, color="0.2", lw=0.75, ls="--", alpha=0.85
            )
            profile_axis.grid(True, axis="x", alpha=0.25)
            profile_axis.set_title("Longitude mean" if row == 0 else "", fontsize=8)
            profile_axis.tick_params(labelleft=False)
            map_axis.text(
                0.02, 0.96, label, transform=map_axis.transAxes,
                va="top", ha="left", fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )
            map_axis.set_ylabel("Latitude" if block_column == 0 else "")
            if block_column != 0:
                map_axis.tick_params(labelleft=False)
            if row == 0:
                map_axis.tick_params(labelbottom=False)
                map_axis.set_xlabel("")
                profile_axis.tick_params(labelbottom=False)
            else:
                profile_axis.set_xlabel("mm day$^{-1}$")
        for row in range(comparison_count):
            axes[first_row + row, map_column + 1].set_xlim(
                -profile_limit, profile_limit
            )
    fig.colorbar(
        shared_image,
        ax=map_axes,
        orientation="horizontal",
        location="bottom",
        shrink=0.56,
        aspect=42,
        pad=0.025,
        extend=_colorbar_extend(shared_image),
        label="Budget-term change (mm day$^{-1}$)",
    )
    fig.suptitle(
        r"$\partial W/\partial t = E - P - \nabla_h\!\cdot\mathbf{IVT} "
        r"+ F_t + R$"
        f"   ({BOUNDARY_LABELS[boundary]}; experiment - control)",
        fontsize=13,
    )
    figsave.save(fig, f"budget_changes_{boundary}", number=number)
    plt.close(fig)


def plot_attribution(attribution, figsave):
    component_specs = (
        ("temperature", r"$C_T$", "temperature via saturation humidity"),
        ("relative_humidity", r"$C_{RH}$", "relative humidity"),
        ("wind", r"$C_{\mathbf{u}}$", "horizontal wind"),
        ("density", r"$C_\rho$", "dry-air density"),
        ("transient", r"$C^\prime$", "time covariance"),
        ("geometry", r"$C_{rem}$", "geometry + interaction remainder"),
        ("total", r"$\Delta C$", "actual total change"),
    )
    comparison_count = len(COMPARISON_NAMES)
    fig, axes = plt.subplots(
        2 * comparison_count,
        8,
        figsize=(20, 5.75 * comparison_count),
        constrained_layout=True,
        sharey=True,
        gridspec_kw={
            "width_ratios": (3.6, 1.0, 3.6, 1.0, 3.6, 1.0, 3.6, 1.0)
        },
    )
    comparison_labels = COMPARISON_LABELS
    component_fields = {
        component: [
            plot_latitude_subset(
                attribution.convergence_change.sel(
                    comparison=comparison,
                    boundary="cold_point",
                    component=component,
                )
            )
            for comparison in COMPARISON_NAMES
        ]
        for component, _symbol, _description in component_specs
    }
    shared_vmax = _symmetric_limit(
        [
            field
            for fields in component_fields.values()
            for field in fields
        ]
    )
    map_axes = []
    shared_image = None
    for column, (component, symbol, description) in enumerate(component_specs):
        block_row, block_column = divmod(column, 4)
        first_row = comparison_count * block_row
        map_column = 2 * block_column
        fields = component_fields[component]
        profile_limit = _symmetric_limit(
            [field.mean("longitude", skipna=True) for field in fields],
            percentile=100.0,
        )
        for row, (comparison_label, color, field) in enumerate(
            zip(comparison_labels, COMPARISON_COLORS, fields)
        ):
            figure_row = first_row + row
            axis = axes[figure_row, map_column]
            profile_axis = axes[figure_row, map_column + 1]
            map_axes.append(axis)
            image = map_field(
                axis, field,
                f"{symbol}\n{description}" if row == 0 else "",
                symmetric=True, vmax=shared_vmax,
            )
            if shared_image is None:
                shared_image = image
            axis.text(
                0.02, 0.96, comparison_label,
                transform=axis.transAxes, va="top", ha="left", fontsize=7,
                bbox={
                    "facecolor": "white", "alpha": 0.72,
                    "edgecolor": "none",
                },
            )
            zonal_mean = field.mean("longitude", skipna=True)
            profile_axis.plot(
                zonal_mean,
                zonal_mean.latitude,
                color=color,
                lw=1.4,
            )
            profile_axis.axvline(0.0, color="0.35", lw=0.7)
            profile_axis.axhline(
                0.0, color="0.2", lw=0.75, ls="--", alpha=0.85
            )
            profile_axis.grid(True, axis="x", alpha=0.25)
            profile_axis.xaxis.set_major_locator(
                mpl_ticker.MaxNLocator(nbins=3)
            )
            profile_axis.xaxis.set_major_formatter(
                mpl_ticker.FuncFormatter(
                    lambda value, _position: f"{value:.1g}"
                )
            )
            profile_axis.set_title(
                "Longitude mean" if row == 0 else "", fontsize=8
            )
            profile_axis.tick_params(labelleft=False)
            axis.set_ylabel("Latitude" if block_column == 0 else "")
            if block_column != 0:
                axis.tick_params(labelleft=False)
            if row == 0:
                axis.tick_params(labelbottom=False)
                axis.set_xlabel("")
                profile_axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("Longitude")
                profile_axis.set_xlabel("mm day$^{-1}$")
        for row in range(comparison_count):
            axes[first_row + row, map_column + 1].set_xlim(
                -profile_limit, profile_limit
            )
    for row in range(comparison_count, 2 * comparison_count):
        for column in (6, 7):
            axes[row, column].set_visible(False)
    fig.colorbar(
        shared_image,
        ax=map_axes,
        orientation="horizontal",
        location="bottom",
        shrink=0.46,
        aspect=34,
        pad=0.025,
        extend=_colorbar_extend(shared_image),
        label=(
            r"$C_i$ (attributed component) or $\Delta C$ (diagnosed total) "
            r"(mm day$^{-1}$)"
        ),
    )
    fig.suptitle(
        r"Attribution of the cold-point-column moisture-convergence change" "\n"
        r"$\Delta C=C_T+C_{RH}+C_{\mathbf{u}}+C_\rho+C^\prime+C_{rem}$; "
        r"$C=-\nabla_h\!\cdot\mathbf{IVT}$",
        fontsize=12,
    )
    figsave.save(fig, "cold_point_convergence_attribution", number=4)
    plt.close(fig)


def plot_native_deep_tropical_profiles(products, figsave):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, sharey=True)
    variables = (
        (
            "native_specific_humidity_zonal_mean", 1.0e6,
            "Specific humidity (mg kg$^{-1}$)",
        ),
        ("native_temperature_zonal_mean", 1.0, "Temperature (K)"),
        ("native_relative_humidity_zonal_mean", 100.0, "Relative humidity (%)"),
    )
    for simulation, color, product in zip(SIM_NAMES, SIM_COLORS, products):
        pressure = (
            deep_tropical_mean(product.native_pressure_zonal_mean) / 100.0
        )
        for ax, (variable, scale, label) in zip(axes, variables):
            profile = deep_tropical_mean(product[variable]) * scale
            if variable == "native_relative_humidity_zonal_mean":
                profile = profile.clip(0.0, 100.0)
            valid_layer = (
                np.isfinite(profile) & np.isfinite(pressure)
                & (pressure >= 30.0) & (pressure <= 300.0)
            )
            if variable == "native_specific_humidity_zonal_mean":
                valid_layer = valid_layer & (profile > 0.0)
            ax.plot(
                profile.where(valid_layer, drop=True),
                pressure.where(valid_layer, drop=True),
                color=color, lw=1.5, label=simulation,
            )
            ax.set_xlabel(label)
            ax.grid(alpha=0.2)

    for ax in axes:
        ax.axhline(
            FIXED_TOP_PA / 100.0,
            color="0.25",
            ls=":",
            lw=1.0,
            label="fixed 200 hPa boundary" if ax is axes[-1] else None,
        )
    for simulation, color, product in zip(SIM_NAMES, SIM_COLORS, products):
        cold_point_pressure = float(
            deep_tropical_mean(
                product.boundary_pressure_mean.sel(
                    boundary="cold_point", drop=True
                )
            )
        ) / 100.0
        for ax in axes:
            ax.axhline(
                cold_point_pressure,
                color=color,
                ls="--",
                lw=1.15,
                label=(
                    f"{simulation}: cold-point $p_t$"
                    if ax is axes[-1] else None
                ),
            )
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(300.0, 30.0)
        ax.set_yticks((300.0, 200.0, 100.0, 70.0, 50.0, 30.0))
        ax.set_yticklabels(("300", "200", "100", "70", "50", "30"))
    axes[0].set_xscale("log")
    axes[-1].set_xlim(0.0, 100.0)
    axes[0].set_ylabel("Native-level mean pressure (hPa)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        fontsize=7,
        frameon=False,
    )
    fig.suptitle(
        r"10$^\circ$S-10$^\circ$N native-level moisture and thermal structure; "
        r"lines show each simulation's mean cold-point $p_t$"
    )
    figsave.save(fig, "native_deep_tropical_moisture_profiles", number=5)
    plt.close(fig)


def plot_deep_tropical_summary(products, figsave):
    terms = BUDGET_PLOT_TERMS
    fig, axes = plt.subplots(1, len(BOUNDARIES), figsize=(13, 4), constrained_layout=True)
    x = np.arange(len(terms))
    width = 0.8 / len(SIM_NAMES)
    for ax, boundary in zip(axes, BOUNDARIES):
        for index, (simulation, color, product) in enumerate(
            zip(SIM_NAMES, SIM_COLORS, products)
        ):
            values = [
                float(
                    deep_tropical_mean(
                        _budget_field(product, term, boundary, sign)
                    )
                )
                for term, sign, _symbol, _description in terms
            ]
            ax.bar(
                x + (index - (len(SIM_NAMES) - 1) / 2.0) * width,
                values, width,
                color=color, label=simulation,
            )
        ax.axhline(0.0, color="0.3", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([term[2] for term in terms])
        ax.set_title(BOUNDARY_LABELS[boundary])
        ax.set_ylabel("mm day$^{-1}$")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.suptitle(
        r"10$^\circ$S-10$^\circ$N mean signed budget contributions: "
        r"$\partial_tW=E-P-\nabla_h\!\cdot\mathbf{IVT}+F_t+R$"
    )
    figsave.save(fig, "deep_tropical_budget_summary", number=6)
    plt.close(fig)


def plot_top_exchange_components(products, figsave):
    components = (
        ("top_vertical_velocity_component", r"$F_w=-\rho_v w_t$"),
        ("top_local_motion_component", r"$F_{local}=\rho_v\,\partial_t z_t$"),
        (
            "top_horizontal_motion_component",
            r"$F_{slope}=\rho_v\,\mathbf{u}_h\!\cdot\nabla_h z_t$",
        ),
        ("top_influx", r"$F_t=F_w+F_{local}+F_{slope}$"),
    )
    fig, axes = plt.subplots(
        1, len(components), figsize=(14, 3.8),
        constrained_layout=True, sharex=True, squeeze=False,
    )
    axes = axes[0]
    latitude_limits = (
        max(PLOT_LATITUDE_BOUNDS[0], float(products[0].latitude.min())),
        min(PLOT_LATITUDE_BOUNDS[1], float(products[0].latitude.max())),
    )
    boundary = "cold_point"
    for column, (variable, title) in enumerate(components):
        for simulation, color, product in zip(SIM_NAMES, SIM_COLORS, products):
            field = product[variable].sel(boundary=boundary)
            longitude_coverage = np.isfinite(field).mean("longitude")
            zonal_mean = field.mean(
                "longitude", skipna=True
            ).where(longitude_coverage >= 0.5).sel(
                latitude=slice(*PLOT_LATITUDE_BOUNDS)
            )
            axes[column].plot(
                zonal_mean.latitude, zonal_mean, color=color, label=simulation,
            )
        axes[column].axhline(0.0, color="0.3", lw=0.7)
        axes[column].axvline(
            0.0, color="0.2", lw=0.75, ls="--", alpha=0.85
        )
        axes[column].set_title(title, fontsize=9)
        axes[column].set_xlabel("Latitude")
        axes[column].set_xlim(*latitude_limits)
        axes[column].grid(alpha=0.2)
    axes[0].set_ylabel("mm day$^{-1}$")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Zonal-mean native-level vapour flux through the moving cold-point "
        "tropopause (kinematic decomposition after Wei 1987); "
        "positive is downward"
    )
    figsave.save(fig, "cross_tropopause_flux_components", number=7)
    plt.close(fig)


def plot_ivt_component_overview(
    products, figsave, component, number, stem
):
    if component == "zonal":
        component_symbol = r"IVT_u"
        component_description = "Zonal IVT"
        positive_description = "eastward"
        negative_description = "westward"

        def component_field(u, v):
            return u
    elif component == "poleward":
        component_symbol = r"IVT_{\rm pole}"
        component_description = "Poleward IVT"
        positive_description = "poleward"
        negative_description = "equatorward"

        def component_field(u, v):
            hemisphere = xr.where(v.latitude >= 0.0, 1.0, -1.0)
            return hemisphere * v
    else:
        raise ValueError(f"unsupported IVT component: {component}")

    fig, axes = plt.subplots(
        len(SIM_NAMES), 2 * len(BOUNDARIES),
        figsize=(16, 3.3 * len(SIM_NAMES)),
        constrained_layout=True,
        sharey=True,
        gridspec_kw={"width_ratios": (3.6, 1.0, 3.6, 1.0)},
    )
    row_labels = (SIM_NAMES[0], *COMPARISON_LABELS)
    stride_lat = max(products[0].sizes["latitude"] // 24, 1)
    stride_lon = max(products[0].sizes["longitude"] // 36, 1)
    for row, row_label in enumerate(row_labels):
        vector_fields = []
        colour_fields = []
        control_zonal_fields = []
        for boundary in BOUNDARIES:
            control_u = plot_latitude_subset(
                products[0].ivt_u_mean.sel(boundary=boundary)
            )
            control_v = plot_latitude_subset(
                products[0].ivt_v_mean.sel(boundary=boundary)
            )
            if row == 0:
                u = control_u
                v = control_v
            else:
                experiment_u = plot_latitude_subset(
                    products[row].ivt_u_mean.sel(boundary=boundary)
                )
                experiment_v = plot_latitude_subset(
                    products[row].ivt_v_mean.sel(boundary=boundary)
                )
                u = experiment_u - control_u
                v = experiment_v - control_v
            colour_field = component_field(u, v)
            vector_fields.append((u, v))
            colour_fields.append(colour_field)
            control_zonal_fields.append(
                component_field(control_u, control_v)
            )

        row_vmax = _symmetric_limit(colour_fields)
        row_images = []
        map_axes = []
        for column, (
            boundary, (u, v), colour_field, control_zonal
        ) in enumerate(
            zip(
                BOUNDARIES, vector_fields, colour_fields,
                control_zonal_fields,
            )
        ):
            map_axis = axes[row, 2 * column]
            profile_axis = axes[row, 2 * column + 1]
            map_axes.append(map_axis)
            image = map_field(
                map_axis, colour_field,
                f"{row_label}: {BOUNDARY_LABELS[boundary]}",
                symmetric=True,
                vmax=row_vmax,
            )
            row_images.append(image)
            defined_fraction = float(np.isfinite(colour_field).mean())
            if defined_fraction < 0.9995:
                map_axis.text(
                    0.02, 0.03, f"defined: {100.0 * defined_fraction:.1f}%",
                    transform=map_axis.transAxes,
                    fontsize=7, color="0.2",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            map_axis.contour(
                control_zonal.longitude,
                control_zonal.latitude,
                control_zonal,
                levels=[0.0],
                colors="0.2",
                linewidths=1.0,
                linestyles="--",
            )
            map_axis.quiver(
                u.longitude.isel(longitude=slice(None, None, stride_lon)),
                u.latitude.isel(latitude=slice(None, None, stride_lat)),
                u.isel(latitude=slice(None, None, stride_lat),
                       longitude=slice(None, None, stride_lon)),
                v.isel(latitude=slice(None, None, stride_lat),
                       longitude=slice(None, None, stride_lon)),
                color="k", alpha=0.65, scale=None,
            )
            zonal_mean = colour_field.mean("longitude", skipna=True)
            profile_axis.plot(
                zonal_mean,
                zonal_mean.latitude,
                color=SIM_COLORS[row],
                lw=1.7,
            )
            profile_axis.axvline(
                0.0, color="0.35", lw=0.8, ls="--", alpha=0.9
            )
            profile_axis.axhline(
                0.0, color="0.2", lw=0.75, ls="--", alpha=0.85
            )
            profile_axis.set_xlim(-row_vmax, row_vmax)
            profile_axis.set_title("Zonal mean", fontsize=9)
            profile_axis.set_xlabel(
                rf"${component_symbol}$"
                if row == 0
                else rf"$\Delta {component_symbol}$"
            )
            if column > 0:
                profile_axis.tick_params(labelleft=False)
        fig.colorbar(
            row_images[0],
            ax=list(axes[row, :]),
            shrink=0.58,
            aspect=36,
            pad=0.025,
            extend=_colorbar_extend(row_images[0]),
            label=(
                rf"{component_description}, ${component_symbol}$ "
                r"(kg m$^{-1}$ s$^{-1}$)"
                if row == 0
                else rf"{component_description} change, "
                rf"$\Delta {component_symbol}$ "
                r"(kg m$^{-1}$ s$^{-1}$)"
            ),
        )
    fig.suptitle(
        rf"Vertically integrated vapour transport: shading is signed "
        rf"${component_symbol}$ or $\Delta {component_symbol}$ "
        f"(red {positive_description}, blue {negative_description}); "
        r"arrows are "
        r"$\mathbf{IVT}$ or $\Delta\mathbf{IVT}$; dashed contours mark "
        rf"control ${component_symbol}=0$"
    )
    figsave.save(fig, stem, number=number)
    plt.close(fig)


def plot_ivt_overview(products, figsave):
    plot_ivt_component_overview(
        products, figsave,
        component="zonal",
        number=8,
        stem="ivt_vectors_and_magnitude",
    )


def plot_poleward_ivt_overview(products, figsave):
    plot_ivt_component_overview(
        products, figsave,
        component="poleward",
        number=9,
        stem="ivt_vectors_and_poleward_transport",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--figures-only",
        action="store_true",
        help="Build figures only from current compact caches.",
    )
    cache_mode.add_argument(
        "--rebuild-cache",
        "--force-cache",
        dest="rebuild_cache",
        action="store_true",
        help="Rebuild compact products, mean states, and attribution.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a small deep-tropical/time subset",
    )
    parser.add_argument("--test-times", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--figures", nargs="+", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    global SAVE_LOC
    args = parse_args(argv)
    if args.test_times < 2:
        raise ValueError("--test-times must be at least 2")
    if args.test:
        SAVE_LOC = os.path.join(
            SAVE_LOC, "testing", f"all_{args.test_times:03d}times"
        )
    products = [
        prepare_simulation(
            data_loc, simulation, force=args.rebuild_cache,
            test=args.test, test_times=args.test_times,
            require_current=args.figures_only,
        )
        for data_loc, simulation in zip(DATA_LOCS, SIM_NAMES)
    ]
    attribution_path = attribution_cache_path(test=args.test)
    attribution_current = attribution_cache_is_current(
        attribution_path, test=args.test
    )
    if attribution_current and not args.rebuild_cache:
        progress(f"using transport-attribution cache {attribution_path}")
        with xr.open_dataset(attribution_path) as cached:
            attribution = cached.load()
    elif args.figures_only:
        raise FileNotFoundError(
            "Figure-only mode requires a current attribution cache: "
            f"{attribution_path}"
        )
    else:
        attribution = compute_cached_transport_attribution(
            products, test=args.test
        )
        write_dataset_cache(
            attribution,
            attribution_path,
            "vapour-transport attribution",
        )
    attribution["budget_term_response"] = direct_budget_responses(products)

    os.makedirs(SAVE_LOC, exist_ok=True)
    attribution.attrs["response_comparison"] = (
        "direct experiment-minus-control and u-nudged-minus-T-nudged-family "
        "differences; no scalar thermal-residual attribution"
    )
    if not args.no_save:
        write_final_data_products(
            attribution, products, test=args.test
        )

    figsave = FigureSaver(
        SAVE_LOC, enabled=not args.no_save, figures_to_run=args.figures
    )
    plot_tropopause_overview(products, figsave)
    shared_budget_vmax = budget_change_colour_limit(products)
    plot_budget_changes(
        products, figsave, "fixed_200hpa", 2,
        shared_vmax=shared_budget_vmax,
    )
    plot_budget_changes(
        products, figsave, "cold_point", 3,
        shared_vmax=shared_budget_vmax,
    )
    plot_attribution(attribution, figsave)
    plot_native_deep_tropical_profiles(products, figsave)
    plot_deep_tropical_summary(products, figsave)
    plot_top_exchange_components(products, figsave)
    plot_ivt_overview(products, figsave)
    plot_poleward_ivt_overview(products, figsave)


if __name__ == "__main__":
    main()
