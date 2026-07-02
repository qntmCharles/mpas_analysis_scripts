#!/usr/bin/env python
"""
Repair or regrid isobaric pressure levels.

The affected control files contain 22 isobaric slots whose true pressures are
1-650 hPa.  This script writes fixed copies with eight appended lower-level
pressure surfaces: 700, 750, 800, 850, 900, 925, 950, and 1000 hPa.

Pass --target-levels to replace the isobaric grid entirely.  In that mode all
isobaric fields are recomputed from the model-level fields in the full latlon
files and written on exactly the requested pressure grid.

Examples
--------
Dry-run a tiny full-latlon subset:

    python fix_control_isobaric_levels.py \\
        --latlon-files '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/control/latlon.2000-05.nc' \\
        --output-dir /tmp/control_isobaric_test \\
        --time-slice 0:2 --lat-slice 170:190 --lon-slice 350:370

Repair both full latlon and zonal-mean files:

    python fix_control_isobaric_levels.py \\
        --latlon-files '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/control/latlon.2000-*.nc' \\
        --output-dir /store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/control_fixed \\
        --zm-files '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/control/zm_latlon.2000-*.nc' \\
        --zm-output-dir /store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/control_fixed

Regrid files to a shared 22-level pressure grid:

    python fix_control_isobaric_levels.py \\
        --latlon-files '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/control/latlon.2000-*.nc' \\
        --output-dir /store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/control_regridded \\
        --target-levels 50 70 100 150 170 200 250 270 300 350 400 450 500 550 600 650 700 750 800 850 925 1000

Two-step preview and patch workflow:

    python fix_control_isobaric_levels.py \\
        --latlon-files '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/control/latlon.2000-05.nc' \\
        --output-dir /tmp/control_isobaric_overlay \\
        --isobaric-only --use-default-target-levels \\
        --old-levels '1,2,3,5,7,10,20,30,50,70,100,150,200,250,300,350,400,450,500,550,600,650' \\
        --comparison-dir /tmp/control_isobaric_overlay/figures

    python fix_control_isobaric_levels.py \\
        --latlon-files '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/control/latlon.2000-05.nc' \\
        --zm-files '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/control/zm_latlon.2000-05.nc' \\
        --patch-existing-from-dir /tmp/control_isobaric_overlay
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
from pathlib import Path

import numpy as np

# Ceph/FUSE-backed filesystems can reject HDF5's default file-locking path
# when opening large NetCDF-4 files read-write for in-place patching.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from netCDF4 import Dataset


CONTROL_LEVELS_HPA = np.array(
    [
        1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0, 70.0,
        100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0,
        500.0, 550.0, 600.0, 650.0,
    ],
    dtype=np.float64,
)
EXTRA_LEVELS_HPA = np.array([700.0, 750.0, 800.0, 850.0, 900.0, 925.0, 950.0, 1000.0], dtype=np.float64)
FIXED_LEVELS_HPA = np.concatenate([CONTROL_LEVELS_HPA, EXTRA_LEVELS_HPA])
DEFAULT_TARGET_LEVELS_HPA = np.array(
    [
        7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0,
        300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 850.0, 900.0, 950.0, 1000.0
    ],
    dtype=np.float64,
)
LEGACY_FILE_LABELS_HPA = np.array(
    [
        1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0, 70.0,
        100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0,
        700.0, 850.0, 925.0, 1000.0,
    ],
    dtype=np.float64,
)
LEGACY_NUDGED_LEVELS_HPA = np.array(
    [
        50.0, 70.0, 100.0, 150.0, 170.0, 200.0, 250.0, 270.0,
        300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0,
        700.0, 750.0, 800.0, 850.0, 925.0, 1000.0,
    ],
    dtype=np.float64,
)

R_D = 287.0
C_P = 1004.0
P0_PA = 100000.0
G = 9.80665
FILL_THRESHOLD = 1.0e30
PERCENT_REFERENCE_SCALE_PERCENTILE = 98.0

ISO_DIMS = ("nIsoLevelsT", "nIsoLevelsU", "nIsoLevelsZ")
ISO_LEVEL_VAR_BY_DIM = {
    "nIsoLevelsT": "t_iso_levels",
    "nIsoLevelsU": "u_iso_levels",
    "nIsoLevelsZ": "z_iso_levels",
}
LEVEL_VAR_NAMES = set(ISO_LEVEL_VAR_BY_DIM.values())

LATLON_ISOBARIC_SPECS = {
    "t_isobaric": ("nIsoLevelsT", "temperature"),
    "rh_isobaric": ("nIsoLevelsT", "relhum"),
    "qv_isobaric": ("nIsoLevelsT", "qv"),
    "uzonal_isobaric": ("nIsoLevelsU", "uReconstructZonal"),
    "umeridional_isobaric": ("nIsoLevelsU", "uReconstructMeridional"),
    "z_isobaric": ("nIsoLevelsZ", "zmid"),
    "omg_isobaric": ("nIsoLevelsZ", "omega_hydrostatic"),
}

ZM_ISOBARIC_SPECS = {
    name: spec for name, spec in LATLON_ISOBARIC_SPECS.items() if name != "z_isobaric"
}

DEFAULT_COMPARISON_VARS = tuple(LATLON_ISOBARIC_SPECS)
DEFAULT_DETECTION_VARS = (
    "t_isobaric",
    "uzonal_isobaric",
    "umeridional_isobaric",
    "rh_isobaric",
    "qv_isobaric",
    "z_isobaric",
)


def parse_slice(text):
    if text is None:
        return slice(None)
    parts = text.split(":")
    if len(parts) > 3:
        raise argparse.ArgumentTypeError(f"Bad slice {text!r}")
    values = [int(p) if p else None for p in parts]
    while len(values) < 3:
        values.append(None)
    slc = slice(*values)
    if slc.step not in (None, 1):
        raise argparse.ArgumentTypeError("Slice steps other than 1 are not supported")
    return slc


def selected_len(size, slc):
    start, stop, step = slc.indices(size)
    return len(range(start, stop, step))


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            paths.append(pattern)
    return [Path(path) for path in paths]


def parse_level_values(values):
    if values is None:
        return None

    tokens = []
    for value in values:
        tokens.extend(value.replace(",", " ").split())
    if not tokens:
        raise argparse.ArgumentTypeError("--target-levels needs at least one pressure level")

    try:
        levels = np.array([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Could not parse --target-levels: {values}") from exc

    if not np.all(np.isfinite(levels)):
        raise argparse.ArgumentTypeError("--target-levels must be finite numbers")
    if np.any(levels <= 0.0):
        raise argparse.ArgumentTypeError("--target-levels must be positive hPa values")
    if np.unique(levels).size != levels.size:
        raise argparse.ArgumentTypeError("--target-levels contains duplicate pressure levels")

    return levels


def get_fill_value(var):
    return var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None


def as_float_array(data, fill_value=None):
    if np.ma.isMaskedArray(data):
        arr = data.astype(np.float64).filled(np.nan)
    else:
        arr = np.asarray(data, dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    if fill_value is not None and np.isfinite(fill_value):
        arr[np.abs(arr - fill_value) == 0] = np.nan
    arr[np.abs(arr) > FILL_THRESHOLD] = np.nan
    return arr


def old_data_array(data, fill_value=None):
    if np.ma.isMaskedArray(data):
        if fill_value is None:
            return data.filled(np.nan)
        return data.filled(fill_value)
    return np.asarray(data)


def output_data_array(data, dtype, fill_value=None):
    arr = np.asarray(data)
    if np.issubdtype(np.dtype(dtype), np.floating):
        arr = arr.astype(dtype, copy=False)
        if fill_value is not None:
            arr = np.where(np.isfinite(arr), arr, np.asarray(fill_value, dtype=dtype))
        return arr
    return arr.astype(dtype, copy=False)


def levels_for_source_variable(src, var_name, iso_dim, old_levels_hpa=None):
    if old_levels_hpa is not None:
        levels = np.asarray(old_levels_hpa, dtype=np.float64)
    else:
        levels = source_levels_for_dim(src, iso_dim)
        if levels is None and iso_dim in src.dimensions and len(src.dimensions[iso_dim]) == CONTROL_LEVELS_HPA.size:
            levels = CONTROL_LEVELS_HPA
        if levels is None:
            raise ValueError(f"Could not determine old pressure levels for {src.filepath()}:{var_name}")

    if iso_dim in src.dimensions and len(src.dimensions[iso_dim]) != levels.size:
        raise ValueError(
            f"{src.filepath()}:{var_name} has {len(src.dimensions[iso_dim])} old levels, "
            f"but {levels.size} old levels were supplied/found"
        )
    return levels


def normalised_slices(ds, args, include_lon=True):
    slices = {}
    for dim, requested in (
        ("Time", args.time_slice),
        ("latitude", args.lat_slice),
        ("longitude", args.lon_slice if include_lon else slice(None)),
    ):
        if dim in ds.dimensions and requested != slice(None):
            size = len(ds.dimensions[dim])
            start, stop, step = requested.indices(size)
            if step != 1:
                raise ValueError(f"{dim} slice steps other than 1 are not supported")
            slices[dim] = slice(start, stop, step)
    return slices


def planned_dimension_sizes(src, dim_slices, output_levels, replace_levels=False):
    sizes = {}
    for name, dim in src.dimensions.items():
        size = len(dim)
        if name in ISO_DIMS and (replace_levels or size == CONTROL_LEVELS_HPA.size):
            sizes[name] = output_levels.size
        elif name == "pres" and (replace_levels or size == CONTROL_LEVELS_HPA.size):
            sizes[name] = output_levels.size
        elif name in dim_slices:
            sizes[name] = selected_len(size, dim_slices[name])
        else:
            sizes[name] = size
    return sizes


def create_dimensions(src, dst, dim_slices, output_levels, replace_levels=False):
    sizes = planned_dimension_sizes(src, dim_slices, output_levels, replace_levels)
    for name, dim in src.dimensions.items():
        if dim.isunlimited() and name == "Time":
            dst.createDimension(name, None)
        else:
            dst.createDimension(name, sizes[name])
    return sizes


def variable_creation_kwargs(src_var, planned_sizes):
    kwargs = {}
    fill_value = get_fill_value(src_var)
    if fill_value is not None:
        kwargs["fill_value"] = fill_value

    filters = src_var.filters()
    if filters.get("zlib"):
        kwargs["zlib"] = True
        kwargs["complevel"] = filters.get("complevel", 4)
        kwargs["shuffle"] = filters.get("shuffle", True)
    if filters.get("fletcher32"):
        kwargs["fletcher32"] = True

    chunking = src_var.chunking()
    if chunking != "contiguous":
        chunksizes = []
        for dim, chunk in zip(src_var.dimensions, chunking):
            chunksizes.append(min(int(chunk), planned_sizes[dim]))
        kwargs["chunksizes"] = tuple(chunksizes)

    return kwargs


def copy_attrs(src_obj, dst_obj, skip=("_FillValue",)):
    for attr in src_obj.ncattrs():
        if attr in skip:
            continue
        dst_obj.setncattr(attr, src_obj.getncattr(attr))


def create_variable_like(src, dst, name, planned_sizes):
    src_var = src.variables[name]
    kwargs = variable_creation_kwargs(src_var, planned_sizes)
    dst_var = dst.createVariable(name, src_var.datatype, src_var.dimensions, **kwargs)
    copy_attrs(src_var, dst_var)
    return dst_var


def create_isobaric_only_dimensions(src, dst, dim_slices, output_levels, include_lon=True):
    sizes = {}
    coord_dims = ("Time", "latitude", "longitude") if include_lon else ("Time", "latitude")

    for name in coord_dims:
        if name not in src.dimensions:
            continue
        dim = src.dimensions[name]
        size = selected_len(len(dim), dim_slices[name]) if name in dim_slices else len(dim)
        if dim.isunlimited() and name == "Time":
            dst.createDimension(name, None)
        else:
            dst.createDimension(name, size)
        sizes[name] = size

    if "pres" in src.dimensions:
        dst.createDimension("pres", output_levels.size)
        sizes["pres"] = output_levels.size
    else:
        for name in ISO_DIMS:
            if name in src.dimensions:
                dst.createDimension(name, output_levels.size)
                sizes[name] = output_levels.size

    return sizes


def create_coordinate_variables(src, dst, planned_sizes, names):
    for name in names:
        if name in src.variables and name in dst.dimensions:
            create_variable_like(src, dst, name, planned_sizes)


def add_history(src, dst, output_levels, replace_levels=False):
    copy_attrs(src, dst, skip=("history",))
    stamp = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    if replace_levels:
        action = f"regridded isobaric variables to {output_levels.tolist()} hPa"
    else:
        action = f"appended control-run isobaric levels {EXTRA_LEVELS_HPA.tolist()} hPa"
    message = f"{stamp}: {action} using {Path(__file__).name}"
    old_history = src.getncattr("history") if "history" in src.ncattrs() else ""
    dst.setncattr("history", f"{old_history}\n{message}" if old_history else message)


def create_all_variables(src, dst, planned_sizes, special_names=()):
    special = set(special_names)
    for name in src.variables:
        if name in special:
            continue
        create_variable_like(src, dst, name, planned_sizes)


def create_full_level_variables(dst, output_levels):
    for dim_name, var_name in ISO_LEVEL_VAR_BY_DIM.items():
        if dim_name not in dst.dimensions:
            continue
        if var_name not in dst.variables:
            var = dst.createVariable(var_name, "f8", (dim_name,))
        else:
            var = dst.variables[var_name]
        var[:] = output_levels
        var.setncattr("units", "hPa")
        var.setncattr("positive", "down")
        var.setncattr("axis", "Z")
        var.setncattr("long_name", "isobaric pressure level")


def create_zm_pres_variable(src, dst, output_levels):
    src_var = src.variables["pres"]
    var = dst.createVariable("pres", src_var.datatype, ("pres",))
    copy_attrs(src_var, var)
    var[:] = output_levels


def iter_time_chunks(src, dim_slices, chunk_size):
    size = len(src.dimensions["Time"])
    slc = dim_slices.get("Time", slice(None))
    start, stop, step = slc.indices(size)
    if step != 1:
        raise ValueError("Time slice steps other than 1 are not supported")
    for src_start in range(start, stop, chunk_size):
        src_stop = min(src_start + chunk_size, stop)
        yield slice(src_start, src_stop), slice(src_start - start, src_stop - start)


def source_key_for_dims(dims, dim_slices, time_slice=None):
    key = []
    for dim in dims:
        if dim == "Time" and time_slice is not None:
            key.append(time_slice)
        elif dim in dim_slices:
            key.append(dim_slices[dim])
        else:
            key.append(slice(None))
    return tuple(key)


def dest_key_for_dims(dims, time_slice=None):
    key = []
    for dim in dims:
        if dim == "Time" and time_slice is not None:
            key.append(time_slice)
        else:
            key.append(slice(None))
    return tuple(key)


def copy_variable_data(src_var, dst_var, dim_slices, time_chunk):
    dims = src_var.dimensions
    if "Time" not in dims:
        src_key = source_key_for_dims(dims, dim_slices)
        dst_var[:] = src_var[src_key]
        return

    parent = src_var.group()
    for src_time, dst_time in iter_time_chunks(parent, dim_slices, time_chunk):
        src_key = source_key_for_dims(dims, dim_slices, src_time)
        dst_key = dest_key_for_dims(dims, dst_time)
        dst_var[dst_key] = src_var[src_key]


def copy_non_isobaric_data(src, dst, dim_slices, time_chunk, skip_names):
    for name, src_var in src.variables.items():
        if name in skip_names:
            continue
        copy_variable_data(src_var, dst.variables[name], dim_slices, time_chunk)


def read_model_var(src, name, time_slice, dim_slices):
    if name not in src.variables:
        raise KeyError(f"Missing required source variable {name!r} in {src.filepath()}")
    var = src.variables[name]
    key = source_key_for_dims(var.dimensions, dim_slices, time_slice)
    return as_float_array(var[key], get_fill_value(var))


def read_zmid(src, dim_slices):
    if "zgrid" not in src.variables:
        raise KeyError(f"Missing required source variable 'zgrid' in {src.filepath()}")
    var = src.variables["zgrid"]
    key = source_key_for_dims(var.dimensions, dim_slices)
    zgrid = as_float_array(var[key], get_fill_value(var))
    return 0.5 * (zgrid[:-1, ...] + zgrid[1:, ...])


def vertical_interp_pressure(pressure_pa_desc, values_desc, target_hpa, extrapolate=False):
    pressure = pressure_pa_desc[:, ::-1, ...]
    values = values_desc[:, ::-1, ...]
    nt, nz = pressure.shape[:2]
    horizontal_shape = pressure.shape[2:]
    out = np.full((nt, target_hpa.size) + horizontal_shape, np.nan, dtype=np.float64)

    for level_index, target in enumerate(target_hpa * 100.0):
        idx = np.sum(pressure <= target, axis=1) - 1
        if extrapolate:
            valid = np.ones_like(idx, dtype=bool)
        else:
            valid = (idx >= 0) & (idx < nz - 1)

        idx = np.clip(idx, 0, nz - 2)
        idx0 = idx[:, None, ...]
        idx1 = (idx + 1)[:, None, ...]

        p0 = np.take_along_axis(pressure, idx0, axis=1)[:, 0, ...]
        p1 = np.take_along_axis(pressure, idx1, axis=1)[:, 0, ...]
        v0 = np.take_along_axis(values, idx0, axis=1)[:, 0, ...]
        v1 = np.take_along_axis(values, idx1, axis=1)[:, 0, ...]

        valid &= np.isfinite(p0) & np.isfinite(p1) & np.isfinite(v0) & np.isfinite(v1) & (p1 != p0)
        interp = v0 + (target - p0) * (v1 - v0) / (p1 - p0)
        out[:, level_index, ...] = np.where(valid, interp, np.nan)

    return out


def interp_constant_pressure_axis(data, old_levels_hpa, target_levels_hpa):
    data = np.asarray(data, dtype=np.float64)
    old_levels = np.asarray(old_levels_hpa, dtype=np.float64)
    target_levels = np.asarray(target_levels_hpa, dtype=np.float64)
    out = np.full((data.shape[0], target_levels.size) + data.shape[2:], np.nan, dtype=np.float64)

    order = np.argsort(old_levels)
    p = old_levels[order]
    values = data[:, order, ...]

    for level_index, target in enumerate(target_levels):
        exact = np.where(np.isclose(p, target, rtol=0.0, atol=1.0e-6))[0]
        if exact.size:
            out[:, level_index, ...] = values[:, exact[0], ...]
            continue
        if target < p[0] or target > p[-1]:
            continue
        lower = np.searchsorted(p, target) - 1
        lower = np.clip(lower, 0, p.size - 2)
        upper = lower + 1
        p0 = p[lower]
        p1 = p[upper]
        v0 = values[:, lower, ...]
        v1 = values[:, upper, ...]
        valid = np.isfinite(v0) & np.isfinite(v1) & (p1 != p0)
        interp = v0 + (target - p0) * (v1 - v0) / (p1 - p0)
        out[:, level_index, ...] = np.where(valid, interp, np.nan)

    return out


def overlay_stored_omega_where_available(src, time_slice, dim_slices, target_levels, target_data, old_levels_hpa=None):
    name = "omg_isobaric"
    if name not in src.variables:
        return target_data

    src_var = src.variables[name]
    iso_dim = LATLON_ISOBARIC_SPECS[name][0]
    iso_axis = src_var.dimensions.index(iso_dim)
    old_levels = levels_for_source_variable(src, name, iso_dim, old_levels_hpa)
    key = source_key_for_dims(src_var.dimensions, dim_slices, time_slice)
    old_data = as_float_array(src_var[key], get_fill_value(src_var))
    old_data = np.moveaxis(old_data, iso_axis, 1)
    old_on_target = interp_constant_pressure_axis(old_data, old_levels, target_levels)
    return np.where(np.isfinite(old_on_target), old_on_target, target_data)


def compute_interpolated_fields(lat_src, time_slice, dim_slices, zmid, target_levels, extrapolate=False):
    pressure = read_model_var(lat_src, "pressure", time_slice, dim_slices)
    fields = {}

    for out_name, (_, source_kind) in LATLON_ISOBARIC_SPECS.items():
        if source_kind == "temperature":
            theta = read_model_var(lat_src, "theta", time_slice, dim_slices)
            source = theta * (pressure / P0_PA) ** (R_D / C_P)
        elif source_kind == "zmid":
            source = np.broadcast_to(zmid[None, ...], pressure.shape)
        elif source_kind == "omega_hydrostatic":
            rho = read_model_var(lat_src, "rho", time_slice, dim_slices)
            w = read_model_var(lat_src, "w", time_slice, dim_slices)
            w_mid = 0.5 * (w[:, :-1, ...] + w[:, 1:, ...])
            source = -rho * G * w_mid
        else:
            source = read_model_var(lat_src, source_kind, time_slice, dim_slices)

        fields[out_name] = vertical_interp_pressure(pressure, source, target_levels, extrapolate)

    return fields


def source_values_for_isobaric_var(lat_src, var_name, time_slice, dim_slices, zmid, pressure):
    source_kind = LATLON_ISOBARIC_SPECS[var_name][1]
    if source_kind == "temperature":
        theta = read_model_var(lat_src, "theta", time_slice, dim_slices)
        return theta * (pressure / P0_PA) ** (R_D / C_P)
    if source_kind == "zmid":
        return np.broadcast_to(zmid[None, ...], pressure.shape)
    if source_kind == "omega_hydrostatic":
        rho = read_model_var(lat_src, "rho", time_slice, dim_slices)
        w = read_model_var(lat_src, "w", time_slice, dim_slices)
        w_mid = 0.5 * (w[:, :-1, ...] + w[:, 1:, ...])
        return -rho * G * w_mid
    return read_model_var(lat_src, source_kind, time_slice, dim_slices)


def assign_expanded_isobaric(src_var, dst_var, src_key, dst_key, extra, iso_axis, output_levels):
    fill_value = get_fill_value(src_var)
    old = old_data_array(src_var[src_key], fill_value)
    out_shape = list(old.shape)
    out_shape[iso_axis] = output_levels.size

    if fill_value is None and np.issubdtype(old.dtype, np.floating):
        out = np.full(out_shape, np.nan, dtype=old.dtype)
    else:
        out = np.full(out_shape, fill_value, dtype=old.dtype)

    old_slice = [slice(None)] * old.ndim
    old_slice[iso_axis] = slice(0, CONTROL_LEVELS_HPA.size)
    out[tuple(old_slice)] = old

    extra_slice = [slice(None)] * old.ndim
    extra_slice[iso_axis] = slice(CONTROL_LEVELS_HPA.size, output_levels.size)
    out[tuple(extra_slice)] = output_data_array(extra, old.dtype, fill_value)

    dst_var[dst_key] = out


def assign_regridded_isobaric(src_var, dst_var, dst_key, target_data):
    fill_value = get_fill_value(src_var)
    dst_var[dst_key] = output_data_array(target_data, src_var.dtype, fill_value)


def repair_latlon_isobaric_variables(
    src,
    dst,
    dim_slices,
    time_chunk,
    output_levels,
    replace_levels=False,
    extrapolate=False,
    old_levels_hpa=None,
):
    zmid = read_zmid(src, dim_slices)
    for src_time, dst_time in iter_time_chunks(src, dim_slices, time_chunk):
        interp_levels = output_levels if replace_levels else EXTRA_LEVELS_HPA
        interpolated = compute_interpolated_fields(src, src_time, dim_slices, zmid, interp_levels, extrapolate)
        if replace_levels:
            interpolated["omg_isobaric"] = overlay_stored_omega_where_available(
                src,
                src_time,
                dim_slices,
                output_levels,
                interpolated["omg_isobaric"],
                old_levels_hpa,
            )
        for name, (iso_dim, _) in LATLON_ISOBARIC_SPECS.items():
            if name not in src.variables:
                continue
            src_var = src.variables[name]
            dst_var = dst.variables[name]
            if not replace_levels and len(src.dimensions[iso_dim]) != CONTROL_LEVELS_HPA.size:
                raise ValueError(
                    f"{src.filepath()}:{name} has {len(src.dimensions[iso_dim])} levels; "
                    f"expected {CONTROL_LEVELS_HPA.size}"
                )
            iso_axis = src_var.dimensions.index(iso_dim)
            dst_key = dest_key_for_dims(dst_var.dimensions, dst_time)
            if replace_levels:
                assign_regridded_isobaric(src_var, dst_var, dst_key, interpolated[name])
            else:
                src_key = source_key_for_dims(src_var.dimensions, dim_slices, src_time)
                assign_expanded_isobaric(
                    src_var, dst_var, src_key, dst_key, interpolated[name], iso_axis, output_levels
                )


def repair_zm_isobaric_variables(
    lat_src,
    zm_src,
    zm_dst,
    lat_dim_slices,
    zm_dim_slices,
    time_chunk,
    output_levels,
    replace_levels=False,
    extrapolate=False,
    old_levels_hpa=None,
):
    zmid = read_zmid(lat_src, lat_dim_slices)
    for lat_src_time, zm_dst_time in iter_time_chunks(zm_src, zm_dim_slices, time_chunk):
        interp_levels = output_levels if replace_levels else EXTRA_LEVELS_HPA
        fields_full = compute_interpolated_fields(lat_src, lat_src_time, lat_dim_slices, zmid, interp_levels, extrapolate)
        if replace_levels:
            fields_full["omg_isobaric"] = overlay_stored_omega_where_available(
                lat_src,
                lat_src_time,
                lat_dim_slices,
                output_levels,
                fields_full["omg_isobaric"],
                old_levels_hpa,
            )
        fields_zm = {name: nanmean_last_axis(values) for name, values in fields_full.items()}

        for name in ZM_ISOBARIC_SPECS:
            if name not in zm_src.variables:
                continue
            src_var = zm_src.variables[name]
            dst_var = zm_dst.variables[name]
            iso_axis = src_var.dimensions.index("pres")
            dst_key = dest_key_for_dims(dst_var.dimensions, zm_dst_time)
            if replace_levels:
                assign_regridded_isobaric(src_var, dst_var, dst_key, fields_zm[name])
            else:
                src_key = source_key_for_dims(src_var.dimensions, zm_dim_slices, lat_src_time)
                assign_expanded_isobaric(
                    src_var, dst_var, src_key, dst_key, fields_zm[name], iso_axis, output_levels
                )


def old_levels_for_variable(src, var_name, args, is_zm=False):
    if args.old_levels_hpa is not None:
        return np.asarray(args.old_levels_hpa, dtype=np.float64)

    if is_zm:
        levels = source_levels_for_dim(src, "pres")
        dim_name = "pres"
    else:
        dim_name = LATLON_ISOBARIC_SPECS[var_name][0]
        levels = source_levels_for_dim(src, dim_name)

    if levels is None and dim_name in src.dimensions and len(src.dimensions[dim_name]) == CONTROL_LEVELS_HPA.size:
        levels = CONTROL_LEVELS_HPA
    if levels is None and dim_name in src.dimensions:
        levels = np.arange(len(src.dimensions[dim_name]), dtype=np.float64)

    return np.asarray(levels, dtype=np.float64)


def output_levels_from_file(ds, is_zm=False):
    if is_zm and "pres" in ds.variables:
        return np.asarray(ds.variables["pres"][:], dtype=np.float64)
    for name in ISO_LEVEL_VAR_BY_DIM.values():
        if name in ds.variables:
            return np.asarray(ds.variables[name][:], dtype=np.float64)
    raise ValueError(f"Could not find output pressure levels in {ds.filepath()}")


def lat_pres_mean(ds, var_name, dim_slices, time_chunk):
    var = ds.variables[var_name]
    dims = var.dimensions
    keep_dims = {"latitude", "pres", "nIsoLevelsT", "nIsoLevelsU", "nIsoLevelsZ"}
    total = None
    count = None

    if "Time" in dims:
        time_iter = iter_time_chunks(ds, dim_slices, time_chunk)
    else:
        time_iter = [(None, None)]

    for src_time, _ in time_iter:
        key = source_key_for_dims(dims, dim_slices, src_time)
        data = as_float_array(var[key], get_fill_value(var))
        reduce_axes = tuple(i for i, dim in enumerate(dims) if dim not in keep_dims)
        valid = np.isfinite(data)
        chunk_sum = np.nansum(data, axis=reduce_axes)
        chunk_count = np.sum(valid, axis=reduce_axes)

        remaining_dims = [dim for dim in dims if dim in keep_dims]
        level_dim = next(dim for dim in remaining_dims if dim != "latitude")
        desired_order = [level_dim, "latitude"]
        if remaining_dims != desired_order:
            order = [remaining_dims.index(dim) for dim in desired_order]
            chunk_sum = np.transpose(chunk_sum, order)
            chunk_count = np.transpose(chunk_count, order)

        if total is None:
            total = chunk_sum.astype(np.float64)
            count = chunk_count.astype(np.int64)
        else:
            total += chunk_sum
            count += chunk_count

    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)


def match_old_to_new_shared_levels(old_field, old_levels, new_levels):
    out = np.full((new_levels.size, old_field.shape[1]), np.nan, dtype=np.float64)
    for new_index, new_level in enumerate(new_levels):
        matches = np.where(np.isclose(old_levels, new_level, rtol=0.0, atol=1.0e-6))[0]
        if matches.size:
            out[new_index, :] = old_field[matches[0], :]
    return out


def nanmean_last_axis(values):
    valid = np.isfinite(values)
    total = np.nansum(values, axis=-1)
    count = np.sum(valid, axis=-1)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)


def nanmean_axis(values, axis):
    valid = np.isfinite(values)
    total = np.nansum(values, axis=axis)
    count = np.sum(valid, axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)


def percent_difference(values, reference):
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    finite_reference = np.abs(reference[np.isfinite(reference)])
    if finite_reference.size:
        scale = np.nanpercentile(finite_reference, PERCENT_REFERENCE_SCALE_PERCENTILE)
    else:
        finite_values = np.abs(values[np.isfinite(values)])
        scale = np.nanpercentile(finite_values, PERCENT_REFERENCE_SCALE_PERCENTILE) if finite_values.size else np.nan
    if not np.isfinite(scale) or scale <= 0.0:
        scale = np.finfo(np.float64).tiny
    valid = np.isfinite(values) & np.isfinite(reference)
    out = np.full(np.broadcast_shapes(values.shape, reference.shape), np.nan, dtype=np.float64)
    np.divide(values - reference, scale, out=out, where=valid)
    return 100.0 * out


def comparison_variables(args, ordered_names, available_names):
    if args.comparison_vars:
        names = [name.strip() for part in args.comparison_vars for name in part.replace(",", " ").split()]
    else:
        names = list(ordered_names)
    return [name for name in names if name in available_names]


def plot_comparison_figure(original_path, fixed_path, args, is_zm=False):
    if args.comparison_dir is None:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    args.comparison_dir.mkdir(parents=True, exist_ok=True)
    with Dataset(original_path, "r") as old_ds, Dataset(fixed_path, "r") as new_ds:
        specs = ZM_ISOBARIC_SPECS if is_zm else LATLON_ISOBARIC_SPECS
        available_names = set(specs) & old_ds.variables.keys() & new_ds.variables.keys()
        names = comparison_variables(args, specs, available_names)
        if not names:
            print(f"No comparison variables found for {original_path}")
            return

        old_slices = normalised_slices(old_ds, args, include_lon=not is_zm)
        # The fixed file already contains any requested subset, so do not slice it
        # a second time.
        new_slices = {}
        lat = np.asarray(new_ds.variables["latitude"][source_key_for_dims(("latitude",), new_slices)], dtype=np.float64)
        new_levels = output_levels_from_file(new_ds, is_zm=is_zm)

        fig, axes = plt.subplots(
            len(names),
            3,
            figsize=(12.6, max(2.8 * len(names), 3.4)),
            squeeze=False,
            constrained_layout=True,
        )

        for row, name in enumerate(names):
            old_levels = old_levels_for_variable(old_ds, name, args, is_zm=is_zm)
            old_mean = lat_pres_mean(old_ds, name, old_slices, args.time_chunk)
            new_mean = lat_pres_mean(new_ds, name, new_slices, args.time_chunk)
            if old_levels.size != old_mean.shape[0]:
                raise ValueError(
                    f"{name} has {old_mean.shape[0]} old levels, but {old_levels.size} "
                    "were supplied/found for the comparison figure"
                )
            old_on_new = match_old_to_new_shared_levels(old_mean, old_levels, new_levels)
            diff_percent = percent_difference(new_mean, old_on_new)

            finite_vals = np.concatenate([old_mean[np.isfinite(old_mean)], new_mean[np.isfinite(new_mean)]])
            if finite_vals.size:
                vmin, vmax = np.nanpercentile(finite_vals, [2, 98])
                if vmin == vmax:
                    vmin, vmax = np.nanmin(finite_vals), np.nanmax(finite_vals)
                if vmin == vmax:
                    delta = abs(vmin) * 0.01 if vmin != 0 else 1.0
                    vmin, vmax = vmin - delta, vmax + delta
            else:
                vmin, vmax = 0.0, 1.0
            dmax = (
                np.nanpercentile(np.abs(diff_percent[np.isfinite(diff_percent)]), 98)
                if np.any(np.isfinite(diff_percent))
                else 1.0
            )
            dmax = dmax if np.isfinite(dmax) and dmax > 0 else 1.0

            abs_levels = np.linspace(vmin, vmax, 21)
            diff_levels = np.linspace(-dmax, dmax, 21)
            y_min = float(np.nanmin([np.nanmin(old_levels), np.nanmin(new_levels)]))
            y_max = float(np.nanmax([np.nanmax(old_levels), np.nanmax(new_levels)]))
            x_min = float(np.nanmin(lat))
            x_max = float(np.nanmax(lat))
            im_abs = None
            im_diff = None
            panels = (
                (old_mean, old_levels, "old stored", "viridis", abs_levels),
                (new_mean, new_levels, "new target", "viridis", abs_levels),
                (diff_percent, new_levels, "% diff of ref scale", "RdBu_r", diff_levels),
            )
            for col, (field, levels, title, cmap, contour_levels) in enumerate(panels):
                ax = axes[row, col]
                im = ax.contourf(lat, levels, field, levels=contour_levels, cmap=cmap, extend="both")
                if col < 2:
                    im_abs = im
                else:
                    im_diff = im
                ax.set_yscale("log")
                ax.set_ylim(y_max, y_min)
                ax.set_xlim(x_min, x_max)
                if row == 0:
                    ax.set_title(title)
                if col == 0:
                    ax.set_ylabel(f"{name}\npressure (hPa)")
                if row == len(names) - 1:
                    ax.set_xlabel("latitude")
            fig.colorbar(im_abs, ax=axes[row, :2], shrink=0.78, label=name)
            fig.colorbar(im_diff, ax=axes[row, 2], shrink=0.78, label=f"{name} difference (% ref scale)")

        prefix = "zm_" if is_zm else ""
        out = args.comparison_dir / f"{prefix}{Path(original_path).stem}_isobaric_comparison.png"
        fig.savefig(out, dpi=args.comparison_dpi)
        plt.close(fig)
        print(f"Wrote comparison figure: {out}")


def levels_match(values, target_levels):
    return values is not None and values.size == target_levels.size and np.allclose(values, target_levels)


def source_levels_for_dim(src, dim_name):
    if dim_name == "pres" and "pres" in src.variables:
        return np.asarray(src.variables["pres"][:], dtype=np.float64)
    level_var = ISO_LEVEL_VAR_BY_DIM.get(dim_name)
    if level_var and level_var in src.variables:
        return np.asarray(src.variables[level_var][:], dtype=np.float64)
    return None


def check_file_needs_repair(src, iso_dim_name, output_levels, replace_levels=False):
    if iso_dim_name not in src.dimensions:
        return False
    size = len(src.dimensions[iso_dim_name])

    if replace_levels:
        source_levels = source_levels_for_dim(src, iso_dim_name)
        if size == output_levels.size and levels_match(source_levels, output_levels):
            return False
        return True

    if size == output_levels.size and levels_match(source_levels_for_dim(src, iso_dim_name), output_levels):
        return False
    if size != CONTROL_LEVELS_HPA.size:
        raise ValueError(
            f"{src.filepath()} has {size} {iso_dim_name} levels; expected "
            f"{CONTROL_LEVELS_HPA.size} or {output_levels.size}"
        )
    return True


def prepare_output_path(input_path, output_dir, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    if output_path.resolve() == input_path.resolve():
        raise ValueError(
            f"Refusing to write repaired data over the input file {input_path}. "
            "Choose a different output directory."
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    return output_path


def copy_variable_in_place(src_var, dst_var, time_chunk):
    if src_var.shape != dst_var.shape:
        raise ValueError(
            f"Cannot patch {dst_var.name}: original shape {dst_var.shape} "
            f"does not match fixed shape {src_var.shape}"
        )
    if "Time" not in src_var.dimensions:
        dst_var[:] = src_var[:]
        return
    time_size = src_var.shape[src_var.dimensions.index("Time")]
    for start in range(0, time_size, time_chunk):
        stop = min(start + time_chunk, time_size)
        key = tuple(slice(start, stop) if dim == "Time" else slice(None) for dim in src_var.dimensions)
        dst_var[key] = src_var[key]


def ensure_level_variable(dst, name, dim_name, template_var, values):
    if name in dst.variables:
        var = dst.variables[name]
    else:
        var = dst.createVariable(name, template_var.datatype, (dim_name,))
        copy_attrs(template_var, var)
    if var.shape != values.shape:
        raise ValueError(f"Cannot patch {name}: original shape {var.shape} does not match fixed shape {values.shape}")
    var[:] = values


def patch_existing_file(original_path, fixed_path, args, is_zm=False):
    if not fixed_path.exists():
        raise FileNotFoundError(f"Fixed isobaric file not found: {fixed_path}")

    specs = ZM_ISOBARIC_SPECS if is_zm else LATLON_ISOBARIC_SPECS
    print(f"Patching existing file: {original_path}")
    with Dataset(original_path, "r+") as dst, Dataset(fixed_path, "r") as fixed:
        for name in specs:
            if name not in fixed.variables:
                continue
            if name not in dst.variables:
                raise KeyError(f"{original_path} does not contain variable {name!r}")
            copy_variable_in_place(fixed.variables[name], dst.variables[name], args.time_chunk)

        if is_zm:
            if "pres" not in fixed.variables:
                raise KeyError(f"{fixed_path} does not contain 'pres'")
            ensure_level_variable(dst, "pres", "pres", fixed.variables["pres"], fixed.variables["pres"][:])
        else:
            for dim_name, level_name in ISO_LEVEL_VAR_BY_DIM.items():
                if level_name in fixed.variables:
                    ensure_level_variable(dst, level_name, dim_name, fixed.variables[level_name], fixed.variables[level_name][:])

        old_history = dst.getncattr("history") if "history" in dst.ncattrs() else ""
        stamp = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        message = f"{stamp}: patched isobaric variables from {fixed_path}"
        dst.setncattr("history", f"{old_history}\n{message}" if old_history else message)


def patch_existing_files(latlon_files, zm_files, args):
    fixed_dir = args.patch_existing_from_dir
    for path in latlon_files:
        patch_existing_file(path, fixed_dir / path.name, args, is_zm=False)
    for path in zm_files:
        patch_existing_file(path, fixed_dir / path.name, args, is_zm=True)


def plot_existing_comparison_figures(latlon_files, zm_files, args):
    fixed_dir = args.comparison_only_from_dir
    if args.comparison_domain in ("all", "latlon"):
        for path in latlon_files:
            fixed_path = fixed_dir / path.name
            if not fixed_path.exists():
                raise FileNotFoundError(f"Fixed isobaric file not found: {fixed_path}")
            plot_comparison_figure(path, fixed_path, args, is_zm=False)
    if args.comparison_domain in ("all", "zm"):
        for path in zm_files:
            fixed_path = fixed_dir / path.name
            if not fixed_path.exists():
                raise FileNotFoundError(f"Fixed zonal-mean isobaric file not found: {fixed_path}")
            plot_comparison_figure(path, fixed_path, args, is_zm=True)


def zonal_mean_chunk(data, dims):
    lon_axis = dims.index("longitude")
    zonal = nanmean_axis(data, axis=lon_axis)
    remaining_dims = [dim for dim in dims if dim != "longitude"]
    level_dims = [dim for dim in remaining_dims if dim in ISO_DIMS]
    if not level_dims:
        raise ValueError(f"Could not find isobaric dimension in {dims}")
    desired_order = [dim for dim in ("Time", level_dims[0], "latitude") if dim in remaining_dims]
    if remaining_dims != desired_order:
        zonal = np.transpose(zonal, [remaining_dims.index(dim) for dim in desired_order])
    return zonal


def verify_latlon_zm_file(latlon_fixed_path, zm_fixed_path, args):
    if not latlon_fixed_path.exists():
        raise FileNotFoundError(f"Fixed latlon file not found for check: {latlon_fixed_path}")
    if not zm_fixed_path.exists():
        raise FileNotFoundError(f"Fixed zonal-mean file not found for check: {zm_fixed_path}")

    lines = [f"Latlon-to-zm check: {latlon_fixed_path.name} -> {zm_fixed_path.name}"]
    with Dataset(latlon_fixed_path, "r") as lat_ds, Dataset(zm_fixed_path, "r") as zm_ds:
        zm_levels = source_levels_for_dim(zm_ds, "pres")
        for name, (iso_dim, _) in ZM_ISOBARIC_SPECS.items():
            if name not in lat_ds.variables or name not in zm_ds.variables:
                continue

            lat_levels = source_levels_for_dim(lat_ds, iso_dim)
            if lat_levels is not None and zm_levels is not None:
                if lat_levels.shape != zm_levels.shape or not np.allclose(lat_levels, zm_levels):
                    raise ValueError(f"{name}: fixed latlon levels do not match fixed zm levels")

            lat_var = lat_ds.variables[name]
            zm_var = zm_ds.variables[name]
            if "longitude" not in lat_var.dimensions:
                raise ValueError(f"{latlon_fixed_path}:{name} has no longitude dimension")

            max_abs_pct = np.nan
            sum_sq_pct = 0.0
            count_pct = 0
            for src_time, _ in iter_time_chunks(lat_ds, {}, args.time_chunk):
                lat_key = source_key_for_dims(lat_var.dimensions, {}, src_time)
                zm_key = source_key_for_dims(zm_var.dimensions, {}, src_time)
                lat_data = as_float_array(lat_var[lat_key], get_fill_value(lat_var))
                zm_data = as_float_array(zm_var[zm_key], get_fill_value(zm_var))
                lat_zm = zonal_mean_chunk(lat_data, list(lat_var.dimensions))
                if lat_zm.shape != zm_data.shape:
                    raise ValueError(f"{name}: zonal mean shape {lat_zm.shape} does not match zm shape {zm_data.shape}")

                diff_pct = percent_difference(lat_zm, zm_data)
                finite = np.isfinite(diff_pct)
                if np.any(finite):
                    abs_diff_pct = np.abs(diff_pct[finite])
                    chunk_max = float(np.nanmax(abs_diff_pct))
                    max_abs_pct = chunk_max if not np.isfinite(max_abs_pct) else max(max_abs_pct, chunk_max)
                    sum_sq_pct += float(np.nansum(diff_pct[finite] ** 2))
                    count_pct += int(np.sum(finite))

            rms_pct = (sum_sq_pct / count_pct) ** 0.5 if count_pct else np.nan
            lines.append(
                f"  {name}: max_abs_pct_ref_scale={max_abs_pct:.6g}, rms_pct_ref_scale={rms_pct:.6g}, n={count_pct}"
            )

    for line in lines:
        print(line)
    if args.comparison_dir is not None:
        args.comparison_dir.mkdir(parents=True, exist_ok=True)
        with (args.comparison_dir / "latlon_zm_check.txt").open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


def verify_latlon_zm_outputs(latlon_files, zm_files, latlon_fixed_dir, zm_fixed_dir, args):
    for latlon_path, zm_path in zip(latlon_files, zm_files):
        verify_latlon_zm_file(latlon_fixed_dir / latlon_path.name, zm_fixed_dir / zm_path.name, args)


def compact_detection_slice(size, requested, count, centered=False):
    if requested != slice(None):
        start, stop, step = requested.indices(size)
        if step != 1:
            raise ValueError("Detection slices with steps other than 1 are not supported")
        return slice(start, stop, step)
    stop_count = min(size, max(1, count))
    if centered:
        start = max(0, size // 2 - stop_count // 2)
        stop = min(size, start + stop_count)
        start = max(0, stop - stop_count)
        return slice(start, stop)
    return slice(0, stop_count)


def detection_slices(ds, args, include_lon=True, full_lon=False):
    slices = {}
    if "Time" in ds.dimensions:
        slices["Time"] = compact_detection_slice(
            len(ds.dimensions["Time"]), args.time_slice, args.detect_time_count, centered=False
        )
    if "latitude" in ds.dimensions:
        slices["latitude"] = compact_detection_slice(
            len(ds.dimensions["latitude"]), args.lat_slice, args.detect_lat_count, centered=True
        )
    if include_lon and "longitude" in ds.dimensions:
        if full_lon:
            slices["longitude"] = slice(None)
        else:
            slices["longitude"] = compact_detection_slice(
                len(ds.dimensions["longitude"]), args.lon_slice, args.detect_lon_count, centered=True
            )
    return slices


def slice_label(slc):
    return f"{slc.start if slc.start is not None else ''}:{slc.stop if slc.stop is not None else ''}"


def parse_detection_variables(args, is_zm=False):
    if args.detect_vars:
        names = [name.strip() for part in args.detect_vars for name in part.replace(",", " ").split()]
    else:
        names = list(DEFAULT_DETECTION_VARS)
    specs = ZM_ISOBARIC_SPECS if is_zm else LATLON_ISOBARIC_SPECS
    return [name for name in names if name in specs]


def add_candidate_level_set(candidates, name, levels, nlev=None):
    if levels is None:
        return
    values = np.asarray(levels, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        return
    if np.unique(values).size != values.size:
        return
    if nlev is not None and values.size != nlev:
        return
    for _, existing in candidates:
        if existing.size == values.size and np.allclose(existing, values, rtol=0.0, atol=1.0e-8):
            return
    candidates.append((name, values))


def candidate_level_sets(ds, args, is_zm=False, nlev=None):
    candidates = []
    if is_zm:
        add_candidate_level_set(candidates, "file:pres", source_levels_for_dim(ds, "pres"), nlev=nlev)
    else:
        for dim_name, level_name in ISO_LEVEL_VAR_BY_DIM.items():
            add_candidate_level_set(candidates, f"file:{level_name}", source_levels_for_dim(ds, dim_name), nlev=nlev)
    add_candidate_level_set(candidates, "control_actual_1_650", CONTROL_LEVELS_HPA, nlev=nlev)
    add_candidate_level_set(candidates, "legacy_file_labels_1_1000", LEGACY_FILE_LABELS_HPA, nlev=nlev)
    add_candidate_level_set(candidates, "legacy_nudged_50_1000", LEGACY_NUDGED_LEVELS_HPA, nlev=nlev)
    add_candidate_level_set(candidates, "default_target", DEFAULT_TARGET_LEVELS_HPA, nlev=nlev)
    if args.target_levels_hpa is not None:
        add_candidate_level_set(candidates, "requested_target", args.target_levels_hpa, nlev=nlev)
    return candidates


def probe_levels_from_candidates(candidates):
    if not candidates:
        return np.array([], dtype=np.float64)
    return np.unique(np.concatenate([levels for _, levels in candidates])).astype(np.float64)


def level_indices(probe_levels, levels):
    indices = []
    for level in levels:
        matches = np.where(np.isclose(probe_levels, level, rtol=0.0, atol=1.0e-8))[0]
        if not matches.size:
            raise ValueError(f"Internal error: level {level} missing from probe grid")
        indices.append(int(matches[0]))
    return indices


def score_arrays_percent(stored, reference):
    pct = percent_difference(stored, reference)
    finite = np.isfinite(pct)
    count = int(np.sum(finite))
    if not count:
        return {"rms": np.nan, "max": np.nan, "sum_sq": 0.0, "count": 0, "coverage": 0.0}
    values = pct[finite]
    return {
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "max": float(np.nanmax(np.abs(values))),
        "sum_sq": float(np.nansum(values ** 2)),
        "count": count,
        "coverage": count / stored.size,
    }


def stored_isobaric_sample(ds, var_name, dim_slices, time_slice, is_zm=False):
    var = ds.variables[var_name]
    key = source_key_for_dims(var.dimensions, dim_slices, time_slice)
    data = as_float_array(var[key], get_fill_value(var))
    iso_dim = "pres" if is_zm else LATLON_ISOBARIC_SPECS[var_name][0]
    iso_axis = var.dimensions.index(iso_dim)
    return np.moveaxis(data, iso_axis, 1)


def candidate_scores_for_variable(stored, interpolated_probe, probe_levels, candidates):
    scores = {}
    for name, levels in candidates:
        idx = level_indices(probe_levels, levels)
        scores[name] = score_arrays_percent(stored, interpolated_probe[:, idx, ...])
    return scores


def best_slot_levels(stored, interpolated_probe, probe_levels):
    best = []
    best_rms = []
    for slot in range(stored.shape[1]):
        slot_scores = []
        for probe_index, level in enumerate(probe_levels):
            score = score_arrays_percent(stored[:, slot, ...], interpolated_probe[:, probe_index, ...])
            slot_scores.append((score["rms"], level))
        finite = [(rms, level) for rms, level in slot_scores if np.isfinite(rms)]
        if not finite:
            best.append(np.nan)
            best_rms.append(np.nan)
            continue
        rms, level = min(finite, key=lambda item: item[0])
        best.append(float(level))
        best_rms.append(float(rms))
    return np.asarray(best, dtype=np.float64), np.asarray(best_rms, dtype=np.float64)


def aggregate_candidate_scores(var_scores):
    aggregate = {}
    for scores in var_scores.values():
        for name, score in scores.items():
            agg = aggregate.setdefault(name, {"sum_sq": 0.0, "count": 0, "max": np.nan, "coverages": []})
            agg["sum_sq"] += score["sum_sq"]
            agg["count"] += score["count"]
            if np.isfinite(score["max"]):
                agg["max"] = score["max"] if not np.isfinite(agg["max"]) else max(agg["max"], score["max"])
            agg["coverages"].append(score["coverage"])
    for agg in aggregate.values():
        agg["rms"] = float(np.sqrt(agg["sum_sq"] / agg["count"])) if agg["count"] else np.nan
        agg["coverage"] = float(np.mean(agg["coverages"])) if agg["coverages"] else np.nan
    return aggregate


def format_levels(levels):
    return "[" + ", ".join(f"{level:g}" if np.isfinite(level) else "nan" for level in levels) + "]"


def report_detection_result(title, nlev, candidates, var_scores, slot_levels_by_var, stated_levels=None, sample_label=None):
    aggregate = aggregate_candidate_scores(var_scores)
    ranked = sorted(
        aggregate.items(),
        key=lambda item: item[1]["rms"] if np.isfinite(item[1]["rms"]) else np.inf,
    )
    print(title)
    if sample_label:
        print(f"  sample: {sample_label}")
    if stated_levels:
        for name, levels in stated_levels:
            print(f"  stated {name}: {format_levels(levels)}")
    print(f"  stored slots: {nlev}")
    if ranked:
        best_name, best_score = ranked[0]
        print(
            f"  best candidate: {best_name} "
            f"(rms_pct_ref_scale={best_score['rms']:.6g}, coverage={best_score['coverage']:.3f})"
        )
        print("  candidate scores:")
        for name, score in ranked:
            print(
                f"    {name}: rms_pct_ref_scale={score['rms']:.6g}, "
                f"max_abs_pct_ref_scale={score['max']:.6g}, coverage={score['coverage']:.3f}, n={score['count']}"
            )
    else:
        print("  no candidate levels could be scored")

    if var_scores:
        print("  variable bests:")
        for var_name, scores in var_scores.items():
            ranked_var = sorted(
                scores.items(),
                key=lambda item: item[1]["rms"] if np.isfinite(item[1]["rms"]) else np.inf,
            )
            best_name, score = ranked_var[0]
            print(f"    {var_name}: {best_name} rms_pct_ref_scale={score['rms']:.6g} coverage={score['coverage']:.3f}")

    if slot_levels_by_var:
        levels = np.vstack([levels for levels, _ in slot_levels_by_var.values()])
        inferred = np.nanmedian(levels, axis=0)
        spread = np.nanmax(levels, axis=0) - np.nanmin(levels, axis=0)
        print(f"  inferred slot levels median across variables: {format_levels(inferred)}")
        if np.nanmax(spread) > 1.0e-6:
            print(f"  inferred slot-level spread across variables: {format_levels(spread)}")


def stated_latlon_levels(ds):
    levels = []
    for dim_name, level_name in ISO_LEVEL_VAR_BY_DIM.items():
        values = source_levels_for_dim(ds, dim_name)
        if values is not None:
            levels.append((level_name, values))
    return levels


def stated_zm_levels(ds):
    levels = source_levels_for_dim(ds, "pres")
    return [("pres", levels)] if levels is not None else []


def detect_latlon_pressure_levels(path, args):
    with Dataset(path, "r") as lat_src:
        dim_slices = detection_slices(lat_src, args, include_lon=True, full_lon=False)
        time_slice = dim_slices.get("Time", slice(None))
        variables = [name for name in parse_detection_variables(args, is_zm=False) if name in lat_src.variables]
        if not variables:
            print(f"Pressure-level diagnosis for {path}: no requested isobaric variables found")
            return
        nlev = max(len(lat_src.dimensions[LATLON_ISOBARIC_SPECS[name][0]]) for name in variables)
        candidates = candidate_level_sets(lat_src, args, is_zm=False, nlev=nlev)
        probe_levels = probe_levels_from_candidates(candidates)
        if not probe_levels.size:
            print(f"Pressure-level diagnosis for {path}: no candidate level sets have {nlev} levels")
            return

        pressure = read_model_var(lat_src, "pressure", time_slice, dim_slices)
        zmid = read_zmid(lat_src, dim_slices)
        var_scores = {}
        slot_levels_by_var = {}
        for name in variables:
            stored = stored_isobaric_sample(lat_src, name, dim_slices, time_slice, is_zm=False)
            if stored.shape[1] != nlev:
                continue
            source = source_values_for_isobaric_var(lat_src, name, time_slice, dim_slices, zmid, pressure)
            interpolated = vertical_interp_pressure(pressure, source, probe_levels, args.extrapolate)
            var_scores[name] = candidate_scores_for_variable(stored, interpolated, probe_levels, candidates)
            slot_levels_by_var[name] = best_slot_levels(stored, interpolated, probe_levels)

        sample = ", ".join(f"{dim}={slice_label(slc)}" for dim, slc in dim_slices.items())
        report_detection_result(
            f"Pressure-level diagnosis for full latlon file: {path}",
            nlev,
            candidates,
            var_scores,
            slot_levels_by_var,
            stated_levels=stated_latlon_levels(lat_src),
            sample_label=sample,
        )


def detect_zm_pressure_levels(latlon_path, zm_path, args):
    with Dataset(latlon_path, "r") as lat_src, Dataset(zm_path, "r") as zm_src:
        zm_slices = detection_slices(zm_src, args, include_lon=False)
        lat_slices = detection_slices(lat_src, args, include_lon=True, full_lon=True)
        if "Time" in zm_slices:
            lat_slices["Time"] = zm_slices["Time"]
        if "latitude" in zm_slices:
            lat_slices["latitude"] = zm_slices["latitude"]
        time_slice = zm_slices.get("Time", slice(None))
        variables = [name for name in parse_detection_variables(args, is_zm=True) if name in zm_src.variables]
        if not variables:
            print(f"Pressure-level diagnosis for {zm_path}: no requested zonal-mean isobaric variables found")
            return
        nlev = len(zm_src.dimensions["pres"])
        candidates = candidate_level_sets(zm_src, args, is_zm=True, nlev=nlev)
        probe_levels = probe_levels_from_candidates(candidates)
        if not probe_levels.size:
            print(f"Pressure-level diagnosis for {zm_path}: no candidate level sets have {nlev} levels")
            return

        pressure = read_model_var(lat_src, "pressure", time_slice, lat_slices)
        zmid = read_zmid(lat_src, lat_slices)
        var_scores = {}
        slot_levels_by_var = {}
        for name in variables:
            stored = stored_isobaric_sample(zm_src, name, zm_slices, time_slice, is_zm=True)
            source = source_values_for_isobaric_var(lat_src, name, time_slice, lat_slices, zmid, pressure)
            interpolated_full = vertical_interp_pressure(pressure, source, probe_levels, args.extrapolate)
            interpolated = nanmean_last_axis(interpolated_full)
            var_scores[name] = candidate_scores_for_variable(stored, interpolated, probe_levels, candidates)
            slot_levels_by_var[name] = best_slot_levels(stored, interpolated, probe_levels)

        sample = ", ".join(f"{dim}={slice_label(slc)}" for dim, slc in zm_slices.items())
        report_detection_result(
            f"Pressure-level diagnosis for zonal-mean file: {zm_path}",
            nlev,
            candidates,
            var_scores,
            slot_levels_by_var,
            stated_levels=stated_zm_levels(zm_src),
            sample_label=sample + " (full longitude mean from latlon source)",
        )


def detect_pressure_levels(latlon_files, zm_files, args):
    for path in latlon_files:
        detect_latlon_pressure_levels(path, args)
    if zm_files:
        for latlon_path, zm_path in zip(latlon_files, zm_files):
            detect_zm_pressure_levels(latlon_path, zm_path, args)


def repair_latlon_file(input_path, output_dir, args):
    with Dataset(input_path, "r") as src:
        output_levels = args.output_levels_hpa
        replace_levels = args.target_levels_hpa is not None
        if not check_file_needs_repair(src, "nIsoLevelsT", output_levels, replace_levels):
            print(f"Skipping already-fixed full file: {input_path}")
            return
        output_path = prepare_output_path(input_path, output_dir, args.overwrite)
        dim_slices = normalised_slices(src, args, include_lon=True)
        if args.isobaric_only:
            planned_sizes = create_sizes = {}
            for name in ("Time", "latitude", "longitude"):
                if name in src.dimensions:
                    create_sizes[name] = selected_len(len(src.dimensions[name]), dim_slices[name]) if name in dim_slices else len(src.dimensions[name])
            for name in ISO_DIMS:
                if name in src.dimensions:
                    create_sizes[name] = output_levels.size
            planned_sizes = create_sizes
        else:
            planned_sizes = planned_dimension_sizes(src, dim_slices, output_levels, replace_levels)

        print(f"Writing {'isobaric-only ' if args.isobaric_only else ''}fixed full file: {output_path}")
        with Dataset(output_path, "w", format=src.file_format) as dst:
            if args.isobaric_only:
                planned_sizes = create_isobaric_only_dimensions(src, dst, dim_slices, output_levels, include_lon=True)
            else:
                create_dimensions(src, dst, dim_slices, output_levels, replace_levels)
            add_history(src, dst, output_levels, replace_levels)
            if args.isobaric_only:
                create_coordinate_variables(src, dst, planned_sizes, ("Time", "latitude", "longitude"))
                for name in LATLON_ISOBARIC_SPECS:
                    if name in src.variables:
                        create_variable_like(src, dst, name, planned_sizes)
            else:
                create_all_variables(src, dst, planned_sizes, special_names=LEVEL_VAR_NAMES)
            create_full_level_variables(dst, output_levels)
            skip = set(LATLON_ISOBARIC_SPECS) | LEVEL_VAR_NAMES
            if args.isobaric_only:
                for name in ("Time", "latitude", "longitude"):
                    if name in src.variables and name in dst.variables:
                        copy_variable_data(src.variables[name], dst.variables[name], dim_slices, args.time_chunk)
            else:
                copy_non_isobaric_data(src, dst, dim_slices, args.time_chunk, skip)
            repair_latlon_isobaric_variables(
                src,
                dst,
                dim_slices,
                args.time_chunk,
                output_levels,
                replace_levels,
                args.extrapolate,
                args.old_levels_hpa,
            )
        if args.comparison_domain in ("all", "latlon"):
            plot_comparison_figure(input_path, output_path, args, is_zm=False)


def repair_zm_file(latlon_path, zm_path, output_dir, args):
    with Dataset(latlon_path, "r") as lat_src, Dataset(zm_path, "r") as zm_src:
        output_levels = args.output_levels_hpa
        replace_levels = args.target_levels_hpa is not None
        if not check_file_needs_repair(zm_src, "pres", output_levels, replace_levels):
            print(f"Skipping already-fixed zonal-mean file: {zm_path}")
            return
        output_path = prepare_output_path(zm_path, output_dir, args.overwrite)
        lat_slices = normalised_slices(lat_src, args, include_lon=True)
        if "longitude" in lat_slices:
            print("Ignoring --lon-slice for zonal-mean repair so zm output remains a true zonal mean")
            lat_slices = dict(lat_slices)
            lat_slices.pop("longitude")
        zm_slices = normalised_slices(zm_src, args, include_lon=False)
        if args.isobaric_only:
            planned_sizes = {}
            for name in ("Time", "latitude"):
                planned_sizes[name] = selected_len(len(zm_src.dimensions[name]), zm_slices[name]) if name in zm_slices else len(zm_src.dimensions[name])
            planned_sizes["pres"] = output_levels.size
        else:
            planned_sizes = planned_dimension_sizes(zm_src, zm_slices, output_levels, replace_levels)

        print(f"Writing {'isobaric-only ' if args.isobaric_only else ''}fixed zonal-mean file: {output_path}")
        with Dataset(output_path, "w", format=zm_src.file_format) as dst:
            if args.isobaric_only:
                planned_sizes = create_isobaric_only_dimensions(zm_src, dst, zm_slices, output_levels, include_lon=False)
            else:
                create_dimensions(zm_src, dst, zm_slices, output_levels, replace_levels)
            add_history(zm_src, dst, output_levels, replace_levels)
            if args.isobaric_only:
                create_coordinate_variables(zm_src, dst, planned_sizes, ("Time", "latitude"))
                for name in ZM_ISOBARIC_SPECS:
                    if name in zm_src.variables:
                        create_variable_like(zm_src, dst, name, planned_sizes)
            else:
                create_all_variables(zm_src, dst, planned_sizes, special_names=("pres",))
            create_zm_pres_variable(zm_src, dst, output_levels)
            skip = set(ZM_ISOBARIC_SPECS) | {"pres"}
            if args.isobaric_only:
                for name in ("Time", "latitude"):
                    if name in zm_src.variables and name in dst.variables:
                        copy_variable_data(zm_src.variables[name], dst.variables[name], zm_slices, args.time_chunk)
            else:
                copy_non_isobaric_data(zm_src, dst, zm_slices, args.time_chunk, skip)
            repair_zm_isobaric_variables(
                lat_src,
                zm_src,
                dst,
                lat_slices,
                zm_slices,
                args.time_chunk,
                output_levels,
                replace_levels,
                args.extrapolate,
                args.old_levels_hpa,
            )
        if args.comparison_domain in ("all", "zm"):
            plot_comparison_figure(zm_path, output_path, args, is_zm=True)


def print_plan(latlon_files, zm_files, args):
    replace_levels = args.target_levels_hpa is not None
    print("Isobaric level repair plan")
    print(f"  mode:          {'target regrid' if replace_levels else 'append missing control levels'}")
    if not replace_levels:
        print(f"  append levels: {EXTRA_LEVELS_HPA.tolist()} hPa")
    print(f"  output levels: {args.output_levels_hpa.tolist()} hPa")
    print(f"  full files:    {len(latlon_files)}")
    if args.output_dir:
        print(f"  full output:   {args.output_dir}")
    if zm_files:
        print(f"  zm files:      {len(zm_files)}")
        print(f"  zm output:     {args.zm_output_dir}")
    print(f"  time slice:    {args.time_slice}")
    print(f"  lat slice:     {args.lat_slice}")
    print(f"  lon slice:     {args.lon_slice}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latlon-files", nargs="+", required=True, help="Full latlon NetCDF files or quoted globs")
    parser.add_argument("--output-dir", type=Path, help="Directory for repaired full latlon copies")
    parser.add_argument("--zm-files", nargs="+", help="Paired zm_latlon NetCDF files or quoted globs")
    parser.add_argument("--zm-output-dir", type=Path, help="Directory for repaired zm_latlon copies")
    parser.add_argument(
        "--isobaric-only",
        action="store_true",
        help="Write only coordinates, pressure-level metadata, and fixed isobaric variables",
    )
    parser.add_argument(
        "--patch-existing-from-dir",
        type=Path,
        help="Patch input files in place from fixed isobaric-only files with matching basenames in this directory",
    )
    parser.add_argument(
        "--comparison-only-from-dir",
        type=Path,
        help="Generate comparison figures from existing fixed isobaric-only files in this directory without writing NetCDF files",
    )
    parser.add_argument(
        "--detect-pressure-levels",
        action="store_true",
        help=(
            "Read-only mode: infer the real stored isobaric pressure levels by comparing "
            "the pressure-level variables with interpolation from native model-level fields"
        ),
    )
    parser.add_argument("--time-slice", type=parse_slice, default=slice(None), help="Python slice for Time, e.g. 0:2")
    parser.add_argument("--lat-slice", type=parse_slice, default=slice(None), help="Python slice for latitude")
    parser.add_argument("--lon-slice", type=parse_slice, default=slice(None), help="Python slice for longitude")
    parser.add_argument(
        "--target-levels",
        nargs="+",
        help=(
            "Replace the isobaric grid with these hPa levels instead of only appending "
            "missing control levels. Accepts space-separated values or a quoted comma-separated list."
        ),
    )
    parser.add_argument(
        "--use-default-target-levels",
        action="store_true",
        help=f"Shortcut for --target-levels {' '.join(str(x).rstrip('0').rstrip('.') for x in DEFAULT_TARGET_LEVELS_HPA)}",
    )
    parser.add_argument(
        "--old-levels",
        nargs="+",
        help="Previous hPa levels for comparison plots when the source file has no pressure-level metadata",
    )
    parser.add_argument("--comparison-dir", type=Path, help="Directory for old/new pressure-level comparison figures")
    parser.add_argument(
        "--comparison-domain",
        choices=("all", "latlon", "zm"),
        default="all",
        help="Which comparison figures to generate",
    )
    parser.add_argument(
        "--check-latlon-zm",
        action="store_true",
        help="Chunk through fixed latlon files and check their zonal means against the fixed zm files",
    )
    parser.add_argument(
        "--comparison-vars",
        nargs="+",
        help="Variables to include in comparison figures, space-separated or comma-separated",
    )
    parser.add_argument(
        "--detect-vars",
        nargs="+",
        help=(
            "Variables to use in --detect-pressure-levels, space-separated or comma-separated. "
            f"Defaults to {', '.join(DEFAULT_DETECTION_VARS)}"
        ),
    )
    parser.add_argument("--comparison-dpi", type=int, default=130, help="DPI for comparison figures")
    parser.add_argument("--detect-time-count", type=int, default=2, help="Number of time records sampled by --detect-pressure-levels")
    parser.add_argument("--detect-lat-count", type=int, default=24, help="Number of latitudes sampled by --detect-pressure-levels")
    parser.add_argument("--detect-lon-count", type=int, default=32, help="Number of longitudes sampled by --detect-pressure-levels")
    parser.add_argument("--time-chunk", type=int, default=1, help="Number of time records processed at once")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without writing files")
    parser.add_argument(
        "--extrapolate",
        action="store_true",
        help="Linearly extrapolate outside the model-level pressure range instead of writing missing values",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    patch_mode = args.patch_existing_from_dir is not None
    comparison_only_mode = args.comparison_only_from_dir is not None
    detect_mode = args.detect_pressure_levels
    if args.target_levels and args.use_default_target_levels:
        parser.error("Use either --target-levels or --use-default-target-levels, not both")
    if args.time_chunk < 1:
        parser.error("--time-chunk must be >= 1")
    if args.detect_time_count < 1 or args.detect_lat_count < 1 or args.detect_lon_count < 1:
        parser.error("--detect-time-count, --detect-lat-count, and --detect-lon-count must be >= 1")
    if not detect_mode and not patch_mode and not comparison_only_mode and not args.output_dir and not args.zm_output_dir:
        parser.error("Provide --output-dir and/or --zm-output-dir")
    if args.zm_files and not args.zm_output_dir and not detect_mode and not patch_mode and not comparison_only_mode:
        parser.error("--zm-files requires --zm-output-dir")
    if args.zm_output_dir and not args.zm_files:
        parser.error("--zm-output-dir requires --zm-files")
    if args.comparison_domain == "zm" and not args.zm_files:
        parser.error("--comparison-domain zm requires --zm-files")
    if args.check_latlon_zm and not args.zm_files:
        parser.error("--check-latlon-zm requires --zm-files")
    if args.check_latlon_zm and not comparison_only_mode and not (args.output_dir and args.zm_output_dir):
        parser.error("--check-latlon-zm requires --output-dir and --zm-output-dir outside comparison-only mode")
    if patch_mode and comparison_only_mode:
        parser.error("Use either --patch-existing-from-dir or --comparison-only-from-dir, not both")
    if detect_mode and (patch_mode or comparison_only_mode or args.output_dir or args.zm_output_dir):
        parser.error("--detect-pressure-levels cannot be combined with repair, patch, comparison-only, or output modes")
    if patch_mode and (args.output_dir or args.zm_output_dir):
        parser.error("--patch-existing-from-dir cannot be combined with output directories")
    if comparison_only_mode and (args.output_dir or args.zm_output_dir):
        parser.error("--comparison-only-from-dir cannot be combined with output directories")
    if comparison_only_mode and args.comparison_dir is None:
        parser.error("--comparison-only-from-dir requires --comparison-dir")
    if patch_mode and any(slc != slice(None) for slc in (args.time_slice, args.lat_slice, args.lon_slice)):
        parser.error("Do not use subset slices when patching existing files in place")

    try:
        args.old_levels_hpa = parse_level_values(args.old_levels)
        if args.use_default_target_levels:
            args.target_levels_hpa = DEFAULT_TARGET_LEVELS_HPA.copy()
        else:
            args.target_levels_hpa = parse_level_values(args.target_levels)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    args.output_levels_hpa = args.target_levels_hpa if args.target_levels_hpa is not None else FIXED_LEVELS_HPA

    latlon_files = expand_paths(args.latlon_files)
    zm_files = expand_paths(args.zm_files) if args.zm_files else []
    if zm_files and len(zm_files) != len(latlon_files):
        parser.error(f"Got {len(latlon_files)} latlon files but {len(zm_files)} zm files")

    if detect_mode:
        detect_pressure_levels(latlon_files, zm_files, args)
        return 0
    if patch_mode:
        patch_existing_files(latlon_files, zm_files, args)
        return 0
    if comparison_only_mode:
        plot_existing_comparison_figures(latlon_files, zm_files, args)
        if args.check_latlon_zm:
            verify_latlon_zm_outputs(
                latlon_files,
                zm_files,
                args.comparison_only_from_dir,
                args.comparison_only_from_dir,
                args,
            )
        return 0

    print_plan(latlon_files, zm_files, args)
    if args.dry_run:
        return 0

    if args.output_dir:
        for path in latlon_files:
            repair_latlon_file(path, args.output_dir, args)

    if zm_files:
        for latlon_path, zm_path in zip(latlon_files, zm_files):
            repair_zm_file(latlon_path, zm_path, args.zm_output_dir, args)

    if args.check_latlon_zm:
        verify_latlon_zm_outputs(latlon_files, zm_files, args.output_dir, args.zm_output_dir, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
