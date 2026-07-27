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
from matplotlib import pyplot as plt

from diagnostics.diagnostic_helpers import (
    ANALYSIS_END,
    ANALYSIS_START,
    FigureSaver,
    LATLON_PATTERN,
    diagnostic_cache_dir,
    diagnostic_output_dir,
    finite_or_nan,
    lat_mean,
    load_latlon_dataset,
    netcdf_safe_dataset,
    progress,
    spherical_flux_divergence,
    spherical_horizontal_advection,
    standardize_latlon_dataset,
    total_precip_accum,
)


RD = 287.0
CP = 1004.0
P0_PA = 100_000.0
LV = 2.5e6
SECONDS_PER_DAY = 86_400.0
EARTH_RADIUS_M = 6_371_000.0
FIXED_TOP_PA = 20_000.0
SCHEMA_VERSION = 15
MAX_TROPOPAUSE_SURFACE_JUMP_M = 2_000.0
PLOT_LATITUDE_BOUNDS = (-40.0, 40.0)
TEST_LATITUDE_BOUNDS = (-42.0, 42.0)

BOUNDARIES = ("fixed_200hpa", "cold_point")
MOVING_BOUNDARIES = ("cold_point",)
COMPARISON_NAMES = ("u_nudge_minus_control", "T_nudge_minus_control")
COMPONENT_NAMES = (
    "temperature",
    "relative_humidity",
    "wind",
    "density",
    "transient",
    "geometry",
    "total",
)

ROOT_DIR = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on"
DATA_LOCS = ("control_new", "u_nudge", "T_nudge")
SIM_NAMES = ("control", "nudged u", "nudged T")
SIM_COLORS = ("k", "b", "r")
COMPARISON_COLORS = ("b", "r")
SAVE_LOC = diagnostic_output_dir(
    "/home/cwp29/tmp/mpas/crf_on/final/vapour_transport_budget/"
)
CACHE_ROOT = diagnostic_cache_dir("vapour_transport_budget_cache", root_dir=ROOT_DIR)
DATA_LOC = diagnostic_cache_dir("vapour_transport_budget_data", root_dir=ROOT_DIR)

NEEDED_VARS = (
    "theta", "pressure", "zgrid", "rho", "qv", "w",
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
LATITUDE_BATCH_SIZE = 20
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


def native_midlevel_height(ds, template):
    lower, upper = native_layer_interfaces(ds, template)
    return 0.5 * (lower + upper)


def native_layer_interfaces(ds, template):
    """Return geometrically ordered native-layer interfaces.

    MPAS stores vapour as a dry-air mixing ratio and ``rho`` as dry-air
    density at cell centres.  The corresponding native layer water mass is
    therefore ``rho * qv * dz``; no dry-to-moist mixing-ratio conversion is
    required for this height-coordinate integral.
    """
    level_coord = template["nVertLevels"]
    first = ds.zgrid.isel(nVertLevelsP1=slice(0, -1)).rename(
        {"nVertLevelsP1": "nVertLevels"}
    ).assign_coords(nVertLevels=level_coord)
    second = ds.zgrid.isel(nVertLevelsP1=slice(1, None)).rename(
        {"nVertLevelsP1": "nVertLevels"}
    ).assign_coords(nVertLevels=level_coord)
    lower = xr.where(first <= second, first, second)
    upper = xr.where(first <= second, second, first)
    return lower, upper


def _interpolate_native_block(field, coordinate, target):
    """NumPy kernel for native-profile interpolation with the level axis last."""
    field = np.asarray(field, dtype=float)
    coordinate = np.asarray(coordinate, dtype=float)
    target = np.asarray(target, dtype=float)
    nlevel = field.shape[-1]
    shape = np.broadcast_shapes(
        field.shape[:-1], coordinate.shape[:-1], target.shape
    )
    field = np.broadcast_to(field, shape + (nlevel,))
    coordinate = np.broadcast_to(coordinate, shape + (nlevel,))
    target = np.broadcast_to(target, shape)
    result = np.full(shape, np.nan, dtype=float)
    for index in range(nlevel - 1):
        x0 = coordinate[..., index]
        x1 = coordinate[..., index + 1]
        f0 = field[..., index]
        f1 = field[..., index + 1]
        lower = np.minimum(x0, x1)
        upper = np.maximum(x0, x1)
        valid = (
            np.isfinite(target) & np.isfinite(x0) & np.isfinite(x1)
            & np.isfinite(f0) & np.isfinite(f1)
            & (target >= lower) & (target <= upper) & (x1 != x0)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            value = f0 + (target - x0) * (f1 - f0) / (x1 - x0)
        result = np.where(valid, value, result)
    return result


def interpolate_native_profile(field, coordinate, target, dim="nVertLevels"):
    """Linearly interpolate a native profile to a varying target coordinate."""
    if dim not in field.dims or dim not in coordinate.dims:
        raise ValueError(f"{dim!r} must be present in field and coordinate")
    if field.sizes[dim] != coordinate.sizes[dim]:
        raise ValueError("field and coordinate must have the same vertical size")

    target = target if isinstance(target, xr.DataArray) else xr.DataArray(float(target))
    result = xr.apply_ufunc(
        _interpolate_native_block,
        field,
        coordinate,
        target,
        input_core_dims=[[dim], [dim], []],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=False,
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    result.name = getattr(field, "name", None)
    result.attrs.update(getattr(field, "attrs", {}))
    return result


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
    lower, upper = native_layer_interfaces(ds, field)
    thickness = upper - lower
    clipped_top = xr.where(top_height < upper, top_height, upper)
    overlap = (clipped_top - lower).clip(min=0.0)
    overlap = xr.where(overlap < thickness, overlap, thickness)
    contribution = xr.where(overlap > 0.0, field * overlap, 0.0)
    result = contribution.sum(dim, skipna=False).where(np.isfinite(top_height))
    result.name = getattr(field, "name", None)
    return result


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


def instantaneous_budget(ds, tropopauses):
    native_mixing_ratio = finite_or_nan(ds.qv)
    native_u = finite_or_nan(ds.uReconstructZonal)
    native_v = finite_or_nan(ds.uReconstructMeridional)
    vapour_density = finite_or_nan(ds.rho) * native_mixing_ratio

    top_pressure, top_height, top_temperature = boundary_arrays(ds, tropopauses)

    column = native_bounded_height_integral(vapour_density, ds, top_height)
    ivt_u = native_bounded_height_integral(
        vapour_density * native_u, ds, top_height
    )
    ivt_v = native_bounded_height_integral(
        vapour_density * native_v, ds, top_height
    )

    native_height = native_midlevel_height(ds, native_mixing_ratio)
    vapour_density_top = interpolate_native_profile(
        vapour_density, native_height, top_height
    )
    native_u_top = interpolate_native_profile(native_u, native_height, top_height)
    native_v_top = interpolate_native_profile(native_v, native_height, top_height)
    native_w_top = interpolate_native_profile(
        finite_or_nan(ds.w),
        finite_or_nan(ds.zgrid),
        top_height,
        dim="nVertLevelsP1",
    )

    height_local_motion = []
    height_local_motion_24h = []
    height_horizontal_motion = []
    continuity_masks = []
    continuity_masks_24h = []
    for boundary in BOUNDARIES:
        height = top_height.sel(boundary=boundary, drop=True)
        local = time_tendency(height, step=1)
        local24 = time_tendency(height, step=2)
        horizontal = spherical_horizontal_advection(
            height,
            native_u_top.sel(boundary=boundary, drop=True),
            native_v_top.sel(boundary=boundary, drop=True),
        )
        if boundary in MOVING_BOUNDARIES:
            continuity = moving_surface_continuity(height, time_step=1)
            continuity24 = moving_surface_continuity(height, time_step=2)
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

    lower_interface, upper_interface = native_layer_interfaces(ds, vapour_density)
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

    layer_coord = ds.qv.nVertLevels
    lower_interface = ds.zgrid.isel(nVertLevelsP1=slice(0, -1)).rename(
        {"nVertLevelsP1": "nVertLevels"}
    ).assign_coords(nVertLevels=layer_coord)
    upper_interface = ds.zgrid.isel(nVertLevelsP1=slice(1, None)).rename(
        {"nVertLevelsP1": "nVertLevels"}
    ).assign_coords(nVertLevels=layer_coord)
    layer_thickness = np.abs(upper_interface - lower_interface)
    condensate_mixing_ratio = sum(
        finite_or_nan(ds[name]) for name in ("qc", "qr", "qi", "qs", "qg")
    )
    condensate_column = (
        finite_or_nan(ds.rho) * condensate_mixing_ratio * layer_thickness
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
    native_relative_humidity = native_specific_humidity / (
        saturation_specific_humidity(native_temperature, finite_or_nan(ds.pressure))
    )
    native_temperature_zonal = trapezoidal_time_mean(
        native_temperature
    ).mean("longitude", skipna=True)
    native_specific_humidity_zonal = trapezoidal_time_mean(
        native_specific_humidity
    ).mean("longitude", skipna=True)
    native_relative_humidity_zonal = trapezoidal_time_mean(
        native_relative_humidity
    ).mean("longitude", skipna=True)
    native_pressure_zonal = trapezoidal_time_mean(
        finite_or_nan(ds.pressure)
    ).mean("longitude", skipna=True)
    native_height = native_midlevel_height(ds, native_temperature)
    if "Time" in native_height.dims:
        native_height = trapezoidal_time_mean(native_height)
    native_height_zonal = native_height.mean("longitude", skipna=True)
    zmid = native_height_zonal.mean("latitude", skipna=True)

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
                    "condensate_column_mean": trapezoidal_time_mean(
                        condensate_column
                    ),
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
        "condensate_column_mean",
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
    safe = simulation.replace(" ", "_")
    return os.path.join(CACHE_ROOT, f"{safe}{suffix}_mean_products.nc")


def state_cache_path(simulation, test=False):
    suffix = "_test" if test else ""
    safe = simulation.replace(" ", "_")
    return os.path.join(CACHE_ROOT, f"{safe}{suffix}_native_mean_state.nc")


def cache_is_current(path, simulation, cache_kind):
    if not os.path.exists(path):
        return False
    try:
        with xr.open_dataset(path) as cached:
            return (
                int(cached.attrs.get("schema_version", -1)) == SCHEMA_VERSION
                and cached.attrs.get("simulation") == simulation
                and cached.attrs.get("analysis_start") == ANALYSIS_START
                and cached.attrs.get("analysis_end") == ANALYSIS_END
                and cached.attrs.get("cache_kind") == cache_kind
            )
    except (OSError, ValueError):
        return False


def write_dataset_cache(dataset, path, description):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    progress(f"writing {description} cache {path}")
    netcdf_safe_dataset(dataset).to_netcdf(temporary)
    os.replace(temporary, path)


def write_final_data_products(attribution, products, alpha, data_loc=DATA_LOC):
    """Write retained data products beside the simulations, not the figures."""
    os.makedirs(data_loc, exist_ok=True)
    attribution_path = os.path.join(
        data_loc, "vapour_transport_attribution.nc"
    )
    summary_path = os.path.join(
        data_loc, "vapour_transport_budget_summary.csv"
    )
    write_dataset_cache(
        attribution, attribution_path, "vapour-transport attribution"
    )
    summary_table(products, attribution, alpha).to_csv(
        summary_path, index=False
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
):
    """Sufficient statistics for exact reconstruction of all time means."""
    tropopauses = diagnose_tropopauses(subset)
    instantaneous = instantaneous_budget(subset, tropopauses)

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

    layer_coord = subset.qv.nVertLevels
    lower_interface = subset.zgrid.isel(
        nVertLevelsP1=slice(0, -1)
    ).rename({"nVertLevelsP1": "nVertLevels"}).assign_coords(
        nVertLevels=layer_coord
    )
    upper_interface = subset.zgrid.isel(
        nVertLevelsP1=slice(1, None)
    ).rename({"nVertLevelsP1": "nVertLevels"}).assign_coords(
        nVertLevels=layer_coord
    )
    layer_thickness = np.abs(upper_interface - lower_interface)
    condensate_mixing_ratio = sum(
        finite_or_nan(subset[name]) for name in ("qc", "qr", "qi", "qs", "qg")
    )
    condensate_column = (
        finite_or_nan(subset.rho)
        * condensate_mixing_ratio
        * layer_thickness
    ).sum("nVertLevels", skipna=True)

    mixing_ratio = finite_or_nan(subset.qv)
    density = finite_or_nan(subset.rho)
    zonal_wind = finite_or_nan(subset.uReconstructZonal)
    meridional_wind = finite_or_nan(subset.uReconstructMeridional)
    temperature = finite_or_nan(
        subset.theta * (subset.pressure / P0_PA) ** (RD / CP)
    )
    specific_humidity = mixing_ratio_to_specific_humidity(mixing_ratio)
    relative_humidity = specific_humidity / saturation_specific_humidity(
        temperature, finite_or_nan(subset.pressure)
    )

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
        "evaporation": finite_or_nan(subset.lh) / LV,
        "archived_precipw": finite_or_nan(subset.precipw),
        "condensate_column": condensate_column,
        "mixing_ratio": mixing_ratio,
        "density": density,
        "pressure": finite_or_nan(subset.pressure),
        "temperature": temperature,
        "zonal_wind": zonal_wind,
        "meridional_wind": meridional_wind,
        "vapour_density": density * mixing_ratio,
        "zonal_vapour_flux": density * mixing_ratio * zonal_wind,
        "meridional_vapour_flux": density * mixing_ratio * meridional_wind,
        "specific_humidity": specific_humidity,
        "relative_humidity": relative_humidity,
    }
    for name, field in sources.items():
        _add_weighted_statistic(statistics, name, field, weights)

    statistics["cold_point_valid_count"] = np.isfinite(
        tropopauses.cold_point_pressure.isel(Time=core_slice)
    ).sum("Time")
    statistics["top_flux_valid_count"] = (
        instantaneous.top_flux_surface_valid.isel(Time=core_slice)
        .astype(np.int64)
        .sum("Time")
    )
    statistics["sample_count"] = xr.DataArray(core_stop - core_start)
    return statistics


def _endpoint_statistics(ds):
    """Only the exact first/last fields required by interval tendencies."""
    endpoints = ds.isel(Time=[0, -1])
    tropopauses = diagnose_tropopauses(endpoints)
    _, top_height, _ = boundary_arrays(endpoints, tropopauses)
    vapour_density = finite_or_nan(endpoints.rho) * finite_or_nan(endpoints.qv)
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
            "zgrid": finite_or_nan(ds.zgrid),
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
        "condensate_column_mean",
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
    convergence = -spherical_flux_divergence(
        means["ivt_u"], means["ivt_v"]
    )
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
                    "condensate_column_mean": means["condensate_column"],
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
                        means["specific_humidity"].mean(
                            "longitude", skipna=True
                        )
                    ),
                    "native_relative_humidity_zonal_mean": (
                        means["relative_humidity"].mean(
                            "longitude", skipna=True
                        )
                    ),
                    "native_pressure_zonal_mean": (
                        means["pressure"].mean("longitude", skipna=True)
                    ),
                    "native_midlevel_height_zonal_mean": native_height_zonal,
                    "native_midlevel_height": native_height_zonal.mean(
                        "latitude", skipna=True
                    ),
                    "top_flux_valid_fraction": (
                        statistics.top_flux_valid_count / sample_count
                    ),
                }
            ),
        ],
        compat="override",
    )
    return (
        _set_product_metadata(products, ds, simulation, elapsed_seconds),
        state,
    )


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
        # Slicing across the archive's 120-row chunks can leave a two-row
        # fragment. Dask's second-order latitude derivative requires every
        # chunk to contain at least three rows. Each bounded slab is small, so
        # making latitude a single chunk is both safe and deterministic.
        subset = subset.chunk({"latitude": -1})
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
    combined["native_midlevel_height"] = (
        combined.native_midlevel_height_zonal_mean.mean(
            "latitude", skipna=True
        )
    )
    combined.native_midlevel_height.attrs.update(height_attrs)
    return combined


def compute_latitude_slab_in_time_batches(
    ds, simulation, time_batch_size=TIME_BATCH_SIZE,
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
        )
        computed = bounded_compute(lazy_statistics)[0]
        statistics = (
            computed
            if statistics is None
            else statistics + computed
        )
        del lazy_statistics, computed, subset
        gc.collect()

    endpoints = bounded_compute(_endpoint_statistics(ds))[0]
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
                subset, simulation, time_batch_size=time_batch_size
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
        _combine_product_latitude_batches(product_batches)
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
            f"{TIME_BATCH_SIZE}-sample time batches with "
            f"{TIME_HALO_POINTS}-sample tendency halos"
        )
    if state is not None:
        state.attrs["memory_strategy"] = (
            f"{batch_size}-point latitude batches with "
            f"{LATITUDE_HALO_POINTS}-point derivative halos; "
            f"{TIME_BATCH_SIZE}-sample time batches with "
            f"{TIME_HALO_POINTS}-sample tendency halos"
        )
    return products, state


def build_simulation_caches_on_disk(
    ds, simulation, product_path, state_path,
    need_products=True, need_state=True,
    latitude_batch_size=LATITUDE_BATCH_SIZE,
    time_batch_size=TIME_BATCH_SIZE,
):
    """Spill completed latitude slabs and assemble final caches from disk."""
    working_root = os.path.dirname(
        product_path if need_products else state_path
    )
    os.makedirs(working_root, exist_ok=True)
    safe_simulation = simulation.replace(" ", "_")
    product_parts = []
    state_parts = []
    number_of_batches = (
        ds.sizes["latitude"] + latitude_batch_size - 1
    ) // latitude_batch_size
    temporary_dir = os.path.join(
        working_root,
        f".{safe_simulation}_vapour_parts_schema{SCHEMA_VERSION}",
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
                subset, simulation, time_batch_size=time_batch_size
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
                combined_products.attrs["memory_strategy"] = (
                    f"disk-backed {latitude_batch_size}-point latitude "
                    f"batches and {time_batch_size}-sample time batches"
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


def prepare_simulation(data_loc, simulation, force=False, test=False,
                       test_times=5):
    """Build one simulation's compact caches, then release its raw history."""
    product_path = cache_path(simulation, test=test)
    state_path = state_cache_path(simulation, test=test)
    product_current = (
        not force
        and cache_is_current(product_path, simulation, "mean_products")
    )
    state_current = (
        not force
        and cache_is_current(state_path, simulation, "native_mean_state")
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


def tropical_mean(da, bounds=(-10.0, 10.0)):
    subset = da.sel(latitude=slice(*bounds))
    dims = [dim for dim in ("latitude", "longitude") if dim in subset.dims]
    weights = np.cos(np.deg2rad(subset.latitude))
    return subset.weighted(weights).mean(dims, skipna=True)


def realised_ttl_scaling(products):
    values = []
    for product in products:
        temperature = product.native_temperature_zonal_mean
        height = product.native_midlevel_height
        masked = temperature.where((height >= 14_000.0) & (height <= 18_000.0))
        values.append(float(tropical_mean(masked).mean("nVertLevels", skipna=True)))
    delta_u = values[1] - values[0]
    delta_t = values[2] - values[0]
    alpha = delta_u / delta_t if np.isfinite(delta_t) and abs(delta_t) > 1e-12 else np.nan
    return alpha, values


def scaled_budget_responses(products, alpha):
    """Return u, T, and temperature-scaled residual maps for every budget term."""
    terms = (
        "storage", "evaporation", "precipitation", "moisture_convergence",
        "top_influx", "condensate_storage", "residual",
    )
    boundary_arrays = []
    for boundary in BOUNDARIES:
        term_arrays = []
        for term in terms:
            control = products[0][term]
            u_experiment = products[1][term]
            t_experiment = products[2][term]
            if "boundary" in control.dims:
                control = control.sel(boundary=boundary, drop=True)
                u_experiment = u_experiment.sel(boundary=boundary, drop=True)
                t_experiment = t_experiment.sel(boundary=boundary, drop=True)
            delta_u = u_experiment - control
            delta_t = t_experiment - control
            term_arrays.append(
                xr.concat(
                    [delta_u, delta_t, delta_u - alpha * delta_t],
                    dim=xr.IndexVariable(
                        "response",
                        ["u_nudge_change", "T_nudge_change", "temperature_scaled_residual"],
                    ),
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
        temperature_scaling_alpha=float(alpha),
    )
    return out


def summary_table(products, attribution, alpha):
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
                        "tropical_mean_mm_day": float(tropical_mean(field)),
                    }
                )

    scaled_budget = scaled_budget_responses(products, alpha)
    for boundary in BOUNDARIES:
        for term in terms:
            scaled_field = scaled_budget.sel(
                boundary=boundary,
                budget_term=term,
                response="temperature_scaled_residual",
            )
            rows.append(
                {
                    "simulation": "temperature_scaled_attribution",
                    "boundary": boundary,
                    "term": f"{term}_scaled_residual",
                    "global_mean_mm_day": float(
                        lat_mean(scaled_field.mean("longitude"))
                    ),
                    "tropical_mean_mm_day": float(tropical_mean(scaled_field)),
                }
            )
        for component in COMPONENT_NAMES:
            u_value = float(
                tropical_mean(
                    attribution.convergence_change.sel(
                        comparison=COMPARISON_NAMES[0], boundary=boundary,
                        component=component,
                    )
                )
            )
            t_value = float(
                tropical_mean(
                    attribution.convergence_change.sel(
                        comparison=COMPARISON_NAMES[1], boundary=boundary,
                        component=component,
                    )
                )
            )
            rows.append(
                {
                    "simulation": "temperature_scaled_attribution",
                    "boundary": boundary,
                    "term": f"convergence_{component}",
                    "global_mean_mm_day": np.nan,
                    "tropical_mean_mm_day": u_value - alpha * t_value,
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
    return image


def plot_tropopause_overview(products, figsave):
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.8), constrained_layout=True,
        sharex=True, sharey=True,
    )
    latitude_bounds = slice(*PLOT_LATITUDE_BOUNDS)
    control = products[0]
    relative_humidity = (
        100.0 * control.native_relative_humidity_zonal_mean
    ).clip(0.0, 100.0).sel(latitude=latitude_bounds).transpose(
        "nVertLevels", "latitude"
    )
    background_pressure = (
        control.native_pressure_zonal_mean / 100.0
    ).sel(latitude=latitude_bounds).transpose("nVertLevels", "latitude")
    latitude_2d = np.broadcast_to(
        relative_humidity.latitude.values[None, :], relative_humidity.shape
    )
    latitude_limits = (
        max(PLOT_LATITUDE_BOUNDS[0], float(relative_humidity.latitude.min())),
        min(PLOT_LATITUDE_BOUNDS[1], float(relative_humidity.latitude.max())),
    )
    humidity_levels = np.arange(0.0, 105.0, 5.0)
    images = []
    for ax, boundary, panel_label in zip(axes, BOUNDARIES, ("(a)", "(b)")):
        image = ax.contourf(
            latitude_2d, background_pressure, relative_humidity,
            levels=humidity_levels, cmap="YlGnBu",
        )
        images.append(image)
        ax.contour(
            latitude_2d, background_pressure, relative_humidity,
            levels=(20.0, 50.0, 80.0), colors="0.35",
            linewidths=0.45, alpha=0.55,
        )
        if boundary == "fixed_200hpa":
            ax.axhline(
                FIXED_TOP_PA / 100.0, color="k", lw=1.6,
                label="fixed boundary",
            )
        else:
            for simulation, color, product in zip(
                SIM_NAMES, SIM_COLORS, products
            ):
                tropopause_pressure = product.boundary_pressure_mean.sel(
                    boundary=boundary
                ).mean(
                    "longitude", skipna=True
                ).sel(latitude=latitude_bounds) / 100.0
                ax.plot(
                    tropopause_pressure.latitude, tropopause_pressure,
                    color=color, lw=1.6, label=simulation,
                )
        ax.set_title(f"{panel_label} {BOUNDARY_LABELS[boundary]}")
        ax.set_yscale("log")
        ax.set_ylim(400.0, 40.0)
        ax.set_yticks((400.0, 300.0, 200.0, 100.0, 70.0, 50.0, 40.0))
        ax.set_yticklabels(("400", "300", "200", "100", "70", "50", "40"))
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    for ax in axes:
        ax.set_xlabel("Latitude")
        ax.set_xlim(*latitude_limits)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Pressure (hPa)")
    fig.colorbar(
        images[0], ax=list(axes), label="Control relative humidity (%)",
        shrink=0.92,
    )
    fig.suptitle(
        "Zonal-mean upper-tropospheric structure and analysis upper boundary"
    )
    figsave.save(fig, "tropopause_definitions", number=1)
    plt.close(fig)


def plot_budget_changes(products, figsave, boundary, number):
    fig, axes = plt.subplots(
        4, 6, figsize=(18, 12), constrained_layout=True,
        sharey=True,
        gridspec_kw={"width_ratios": (3.6, 1.0, 3.6, 1.0, 3.6, 1.0)},
    )
    comparison_labels = ("nudged u - control", "nudged T - control")
    for column, (term, sign, symbol, description) in enumerate(BUDGET_PLOT_TERMS):
        block_row, block_column = divmod(column, 3)
        first_row = 2 * block_row
        map_column = 2 * block_column
        fields = [
            plot_latitude_subset(
                _budget_field(experiment, term, boundary, sign)
                - _budget_field(products[0], term, boundary, sign)
            )
            for experiment in products[1:]
        ]
        vmax = _symmetric_limit(fields)
        images = []
        map_axes = []
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
                symmetric=True, vmax=vmax,
            )
            images.append(image)
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
        profile_limit = _symmetric_limit(
            [field.mean("longitude", skipna=True) for field in fields],
            percentile=100.0,
        )
        for row in range(2):
            axes[first_row + row, map_column + 1].set_xlim(
                -profile_limit, profile_limit
            )
        fig.colorbar(
            images[0], ax=map_axes, orientation="horizontal",
            location="bottom", shrink=0.82, pad=0.03,
            extend=_colorbar_extend(images[0]), label="mm day$^{-1}$",
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
    fig = plt.figure(figsize=(15, 10.5), constrained_layout=True)
    outer = fig.add_gridspec(2, 4)
    comparison_labels = ("nudged u - control", "nudged T - control")
    for column, (component, symbol, description) in enumerate(component_specs):
        block_row, block_column = divmod(column, 4)
        inner = outer[block_row, block_column].subgridspec(
            3, 1, height_ratios=(1.0, 1.0, 0.07), hspace=0.08
        )
        fields = [
            plot_latitude_subset(
                attribution.convergence_change.sel(
                    comparison=comparison, boundary="cold_point",
                    component=component,
                )
            )
            for comparison in COMPARISON_NAMES
        ]
        vmax = _symmetric_limit(fields)
        images = []
        for row, (comparison_label, field) in enumerate(
            zip(comparison_labels, fields)
        ):
            axis = fig.add_subplot(inner[row, 0])
            image = map_field(
                axis, field,
                f"{symbol}\n{description}" if row == 0 else "",
                symmetric=True, vmax=vmax,
            )
            images.append(image)
            axis.set_ylabel(f"{comparison_label}\nLatitude")
            if row == 0:
                axis.tick_params(labelbottom=False)
                axis.set_xlabel("")
        colorbar_axis = fig.add_subplot(inner[2, 0])
        fig.colorbar(
            images[0], cax=colorbar_axis, orientation="horizontal",
            extend=_colorbar_extend(images[0]),
            label="mm day$^{-1}$",
        )
    blank_axis = fig.add_subplot(outer[1, 3])
    blank_axis.axis("off")
    fig.suptitle(
        r"Attribution of the cold-point-column moisture-convergence change" "\n"
        r"$\Delta C=C_T+C_{RH}+C_{\mathbf{u}}+C_\rho+C^\prime+C_{rem}$; "
        r"$C=-\nabla_h\!\cdot\mathbf{IVT}$",
        fontsize=12,
    )
    figsave.save(fig, "cold_point_convergence_attribution", number=5)
    plt.close(fig)


def plot_native_tropical_profiles(products, figsave):
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
        pressure = tropical_mean(product.native_pressure_zonal_mean) / 100.0
        for ax, (variable, scale, label) in zip(axes, variables):
            profile = tropical_mean(product[variable]) * scale
            if variable == "native_relative_humidity_zonal_mean":
                profile = profile.clip(0.0, 100.0)
            valid_layer = (
                np.isfinite(profile) & np.isfinite(pressure)
                & (pressure >= 30.0) & (pressure <= 300.0)
            )
            ax.plot(
                profile.where(valid_layer, drop=True),
                pressure.where(valid_layer, drop=True),
                color=color, lw=1.5, label=simulation,
            )
            ax.set_xlabel(label)
            ax.grid(alpha=0.2)

    for boundary, linestyle in (("fixed_200hpa", ":"), ("cold_point", "--")):
        pressures = [
            float(
                tropical_mean(
                    product.boundary_pressure_mean.sel(
                        boundary=boundary, drop=True
                    )
                )
            ) / 100.0
            for product in products
        ]
        for ax in axes:
            if boundary == "cold_point":
                ax.axhspan(
                    min(pressures), max(pressures), color="0.5", alpha=0.10
                )
            reference_label = (
                "fixed 200 hPa boundary"
                if boundary == "fixed_200hpa"
                else "control cold-point $p_t$"
            )
            ax.axhline(
                pressures[0], color="0.25", ls=linestyle, lw=0.9,
                label=reference_label if ax is axes[-1] else None,
            )
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(300.0, 30.0)
        ax.set_yticks((300.0, 200.0, 100.0, 70.0, 50.0, 30.0))
        ax.set_yticklabels(("300", "200", "100", "70", "50", "30"))
    axes[-1].set_xlim(0.0, 100.0)
    axes[0].set_ylabel("Native-level mean pressure (hPa)")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.suptitle(
        r"10$^\circ$S-10$^\circ$N native-level moisture and thermal structure; "
        r"the band spans cold-point $p_t$ across simulations"
    )
    figsave.save(fig, "native_tropical_moisture_profiles", number=6)
    plt.close(fig)


def plot_tropical_summary(products, figsave):
    terms = BUDGET_PLOT_TERMS
    fig, axes = plt.subplots(1, len(BOUNDARIES), figsize=(13, 4), constrained_layout=True)
    x = np.arange(len(terms))
    width = 0.24
    for ax, boundary in zip(axes, BOUNDARIES):
        for index, (simulation, color, product) in enumerate(
            zip(SIM_NAMES, SIM_COLORS, products)
        ):
            values = [
                float(tropical_mean(_budget_field(product, term, boundary, sign)))
                for term, sign, _symbol, _description in terms
            ]
            ax.bar(
                x + (index - 1) * width, values, width,
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
    figsave.save(fig, "tropical_budget_summary", number=7)
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
        axes[column].set_title(title, fontsize=9)
        axes[column].set_xlabel("Latitude")
        axes[column].set_xlim(*latitude_limits)
        axes[column].grid(alpha=0.2)
    axes[0].set_ylabel("mm day$^{-1}$")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Zonal-mean native-level vapour flux relative to the cold-point "
        "tropopause; positive is downward"
    )
    figsave.save(fig, "cross_tropopause_flux_components", number=8)
    plt.close(fig)


def plot_ivt_overview(products, figsave):
    fig, axes = plt.subplots(3, len(BOUNDARIES), figsize=(10, 10), constrained_layout=True)
    row_labels = ("control", "nudged u - control", "nudged T - control")
    stride_lat = max(products[0].sizes["latitude"] // 24, 1)
    stride_lon = max(products[0].sizes["longitude"] // 36, 1)
    for row, row_label in enumerate(row_labels):
        vector_fields = []
        colour_fields = []
        for boundary in BOUNDARIES:
            if row == 0:
                u = plot_latitude_subset(
                    products[0].ivt_u_mean.sel(boundary=boundary)
                )
                v = plot_latitude_subset(
                    products[0].ivt_v_mean.sel(boundary=boundary)
                )
                colour_field = np.hypot(u, v)
            else:
                u = plot_latitude_subset(
                    products[row].ivt_u_mean.sel(boundary=boundary)
                    - products[0].ivt_u_mean.sel(boundary=boundary)
                )
                v = plot_latitude_subset(
                    products[row].ivt_v_mean.sel(boundary=boundary)
                    - products[0].ivt_v_mean.sel(boundary=boundary)
                )
                colour_field = np.sign(
                    u * products[0].ivt_u_mean.sel(boundary=boundary)
                    + v * products[0].ivt_v_mean.sel(boundary=boundary)
                ) * np.hypot(u, v)
            vector_fields.append((u, v))
            colour_fields.append(colour_field)

        row_vmax = _symmetric_limit(colour_fields)
        row_images = []
        for column, (boundary, (u, v), colour_field) in enumerate(
            zip(BOUNDARIES, vector_fields, colour_fields)
        ):
            if row == 0:
                image = map_field(
                    axes[row, column], colour_field,
                    f"{row_label}: {boundary}", symmetric=False,
                    vmin=0.0, vmax=row_vmax,
                )
            else:
                image = map_field(
                    axes[row, column], colour_field,
                    f"{row_label}: {boundary}", symmetric=True,
                    vmax=row_vmax,
                )
            row_images.append(image)
            defined_fraction = float(np.isfinite(colour_field).mean())
            if defined_fraction < 0.9995:
                axes[row, column].text(
                    0.02, 0.03, f"defined: {100.0 * defined_fraction:.1f}%",
                    transform=axes[row, column].transAxes,
                    fontsize=7, color="0.2",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            axes[row, column].quiver(
                u.longitude.isel(longitude=slice(None, None, stride_lon)),
                u.latitude.isel(latitude=slice(None, None, stride_lat)),
                u.isel(latitude=slice(None, None, stride_lat),
                       longitude=slice(None, None, stride_lon)),
                v.isel(latitude=slice(None, None, stride_lat),
                       longitude=slice(None, None, stride_lon)),
                color="k", alpha=0.65, scale=None,
            )
        fig.colorbar(
            row_images[0], ax=list(axes[row, :]), shrink=0.86,
            extend=_colorbar_extend(row_images[0]),
            label="kg m$^{-1}$ s$^{-1}$",
        )
    fig.suptitle("Mean IVT and experiment-minus-control IVT; kg m$^{-1}$ s$^{-1}$")
    figsave.save(fig, "ivt_vectors_and_magnitude", number=9)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--test", action="store_true", help="Run a small tropical/time subset")
    parser.add_argument("--test-times", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--figures", nargs="+", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    products = [
        prepare_simulation(
            data_loc, simulation, force=args.force_cache,
            test=args.test, test_times=args.test_times,
        )
        for data_loc, simulation in zip(DATA_LOCS, SIM_NAMES)
    ]
    attribution = compute_cached_transport_attribution(
        products, test=args.test
    )
    alpha, ttl_temperatures = realised_ttl_scaling(products)
    attribution["budget_term_response"] = scaled_budget_responses(
        products, alpha
    )
    progress(
        "realised TTL temperatures: "
        + ", ".join(f"{name}={value:.3f} K" for name, value in zip(SIM_NAMES, ttl_temperatures))
        + f"; alpha={alpha:.3f}"
    )

    os.makedirs(SAVE_LOC, exist_ok=True)
    attribution.attrs.update(
        temperature_scaling_alpha=float(alpha),
        temperature_scaling_definition="14-18 km, 10S-10N realised mean temperature",
    )
    if not args.no_save:
        write_final_data_products(attribution, products, alpha)

    figsave = FigureSaver(
        SAVE_LOC, enabled=not args.no_save, figures_to_run=args.figures
    )
    plot_tropopause_overview(products, figsave)
    plot_budget_changes(products, figsave, "fixed_200hpa", 2)
    plot_budget_changes(products, figsave, "cold_point", 3)
    plot_attribution(attribution, figsave)
    plot_native_tropical_profiles(products, figsave)
    plot_tropical_summary(products, figsave)
    plot_top_exchange_components(products, figsave)
    plot_ivt_overview(products, figsave)


if __name__ == "__main__":
    main()
