#!/usr/bin/env python
"""
Recover the corrupted ``omg_isobaric`` field in the crf_on T-nudge files.

This is intentionally a one-case repair script. It uses the fixed T-nudge
source and staging directories below and refuses files that do not match the
known bad isobaric remapping.

The previous repair interpreted the original 22 omega slots using the wrong
pressure labels. Most original values still survive at identifiable positions
in the current 22-level field. This script:

* reverses that known mapping;
* retains the recovered stored omega from 200 to 1000 hPa;
* reconstructs 925 hPa independently from the current 900 and 950 hPa slots;
* uses ``omega = -rho * g * w`` on native levels only for 7 to 150 hPa, whose
  original stored values were lost; and
* writes the zonal-mean recovery directly from the repaired full field.

First create and inspect staged omega-only files:

    python fix_t_nudge_omg_isobaric.py stage

This also writes the same old-stored / new-target / percentage-difference
comparison figures as ``fix_control_isobaric_levels.py``, restricted to
``omg_isobaric``.

Then explicitly patch only ``omg_isobaric`` in the original files:

    python fix_t_nudge_omg_isobaric.py patch
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from types import SimpleNamespace

# These very large NetCDF-4 files live on Ceph, where HDF5 file locking can
# make a valid read-write open fail with the uninformative ``NetCDF: HDF
# error``.  This must be set before importing netCDF4, as in the authoritative
# general isobaric-repair script.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
from netCDF4 import Dataset

from fix_control_isobaric_levels import (
    DEFAULT_TARGET_LEVELS_HPA,
    G,
    as_float_array,
    copy_attrs,
    get_fill_value,
    interp_constant_pressure_axis,
    nanmean_last_axis,
    output_data_array,
    plot_comparison_figure,
    score_arrays_percent,
    source_key_for_dims,
    variable_creation_kwargs,
    vertical_interp_pressure,
)


SOURCE_DIRECTORY = Path(
    "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/T_nudge"
)
STAGING_DIRECTORY = Path(
    "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
    "T_nudge_omg_isobaric_recovery"
)
DEFAULT_FIGURE_DIRECTORY = Path(
    "/home/cwp29/tmp/mpas/crf_on/final/isobaric_recovery/T_nudge"
)
FIGURE_DIRECTORY = DEFAULT_FIGURE_DIRECTORY
TARGET_LEVELS_HPA = DEFAULT_TARGET_LEVELS_HPA.copy()

# Exact surviving values after the known erroneous interpolation:
# recovered true source pressure -> current target-grid pressure.
SURVIVING_TRUE_LEVELS_HPA = np.array(
    [
        170.0,
        200.0,
        250.0,
        270.0,
        300.0,
        350.0,
        400.0,
        450.0,
        500.0,
        550.0,
        600.0,
        650.0,
        700.0,
        750.0,
        800.0,
        850.0,
        1000.0,
    ],
    dtype=np.float64,
)
SURVIVING_CURRENT_LEVELS_HPA = np.array(
    [
        7.0,
        10.0,
        20.0,
        30.0,
        50.0,
        70.0,
        100.0,
        150.0,
        200.0,
        250.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        850.0,
        1000.0,
    ],
    dtype=np.float64,
)
RECOVERED_STORED_MINIMUM_HPA = 200.0
RECOVERY_VERSION = "t_nudge_omg_isobaric_recovery_v1"
EXPECTED_925_RELATIVE_RMS_MAX = 1.0e-5
MAPPING_IMPROVEMENT_FACTOR_MAX = 0.75


def utc_stamp():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def exact_level_indices(levels_hpa, requested_hpa):
    levels = np.asarray(levels_hpa, dtype=np.float64)
    indices = []
    for requested in np.asarray(requested_hpa, dtype=np.float64):
        matches = np.flatnonzero(
            np.isclose(levels, requested, rtol=0.0, atol=1.0e-8)
        )
        if matches.size != 1:
            raise ValueError(
                f"Expected exactly one {requested:g} hPa slot in "
                f"{levels.tolist()}, found {matches.size}"
            )
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int64)


def relative_rms_difference(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if not np.any(valid):
        return np.nan
    difference_rms = float(np.sqrt(np.mean((first[valid] - second[valid]) ** 2)))
    scale_values = np.abs(0.5 * (first[valid] + second[valid]))
    scale = float(np.nanpercentile(scale_values, 98.0))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.nan
    return difference_rms / scale


def recover_surviving_source_omega(corrupted_target_omega):
    """Recover surviving original omega slots and the independently encoded 925 hPa."""

    corrupted = np.asarray(corrupted_target_omega, dtype=np.float64)
    if corrupted.ndim < 2 or corrupted.shape[1] != TARGET_LEVELS_HPA.size:
        raise ValueError(
            "Corrupted omega must have pressure as axis 1 with "
            f"{TARGET_LEVELS_HPA.size} levels; got {corrupted.shape}"
        )

    current_indices = exact_level_indices(
        TARGET_LEVELS_HPA,
        SURVIVING_CURRENT_LEVELS_HPA,
    )
    surviving = corrupted[:, current_indices, ...]

    index_850, index_900, index_950, index_1000 = exact_level_indices(
        TARGET_LEVELS_HPA,
        [850.0, 900.0, 950.0, 1000.0],
    )
    recovered_925_from_900 = (
        3.0 * corrupted[:, index_900, ...] - corrupted[:, index_850, ...]
    ) / 2.0
    recovered_925_from_950 = (
        3.0 * corrupted[:, index_950, ...] - corrupted[:, index_1000, ...]
    ) / 2.0
    mismatch = relative_rms_difference(
        recovered_925_from_900,
        recovered_925_from_950,
    )
    recovered_925 = 0.5 * (
        recovered_925_from_900 + recovered_925_from_950
    )

    insertion = int(np.searchsorted(SURVIVING_TRUE_LEVELS_HPA, 925.0))
    source_levels = np.insert(SURVIVING_TRUE_LEVELS_HPA, insertion, 925.0)
    source_values = np.concatenate(
        [
            surviving[:, :insertion, ...],
            recovered_925[:, None, ...],
            surviving[:, insertion:, ...],
        ],
        axis=1,
    )
    return source_levels, source_values, mismatch


def native_target_omega(pressure, rho, w):
    """Calculate omega on the target grid from native pressure, density and w."""

    pressure = np.asarray(pressure, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if pressure.shape != rho.shape:
        raise ValueError(
            f"pressure and rho shapes differ: {pressure.shape} and {rho.shape}"
        )
    if (
        w.shape[0] != pressure.shape[0]
        or w.shape[1] != pressure.shape[1] + 1
        or w.shape[2:] != pressure.shape[2:]
    ):
        raise ValueError(
            "w must use the matching native interface grid; got "
            f"pressure {pressure.shape} and w {w.shape}"
        )
    w_mid = 0.5 * (w[:, :-1, ...] + w[:, 1:, ...])
    omega_native = -rho * G * w_mid
    return vertical_interp_pressure(
        pressure,
        omega_native,
        TARGET_LEVELS_HPA,
        extrapolate=False,
    )


def recover_target_omega(corrupted_target_omega, native_omega):
    """Combine recovered stored omega with native interpolation only where lost."""

    source_levels, source_values, mismatch = recover_surviving_source_omega(
        corrupted_target_omega
    )
    stored_on_target = interp_constant_pressure_axis(
        source_values,
        source_levels,
        TARGET_LEVELS_HPA,
    )
    use_stored = TARGET_LEVELS_HPA >= RECOVERED_STORED_MINIMUM_HPA
    recovered = np.asarray(native_omega, dtype=np.float64).copy()
    recovered[:, use_stored, ...] = stored_on_target[:, use_stored, ...]
    return recovered, mismatch


def write_original_style_comparison_figures(
    lat_path,
    zm_path,
    lat_output,
    zm_output,
    time_chunk,
):
    """Reuse the general repair script's comparison figures for omega only."""

    figure_args = SimpleNamespace(
        comparison_dir=FIGURE_DIRECTORY,
        comparison_vars=["omg_isobaric"],
        comparison_dpi=200,
        old_levels_hpa=None,
        time_slice=slice(None),
        lat_slice=slice(None),
        lon_slice=slice(None),
        time_chunk=time_chunk,
    )
    plot_comparison_figure(
        lat_path,
        lat_output,
        figure_args,
        is_zm=False,
    )
    plot_comparison_figure(
        zm_path,
        zm_output,
        figure_args,
        is_zm=True,
    )


def require_target_coordinate(ds, is_zm=False):
    coordinate_name = "pres" if is_zm else "z_iso_levels"
    dimension_name = "pres" if is_zm else "nIsoLevelsZ"
    if coordinate_name not in ds.variables:
        raise KeyError(f"{ds.filepath()} has no {coordinate_name!r} coordinate")
    levels = np.asarray(ds.variables[coordinate_name][:], dtype=np.float64)
    if levels.shape != TARGET_LEVELS_HPA.shape or not np.allclose(
        levels,
        TARGET_LEVELS_HPA,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError(
            f"{ds.filepath()} does not use the expected current target grid: "
            f"{levels.tolist()}"
        )
    if (
        dimension_name not in ds.dimensions
        or len(ds.dimensions[dimension_name]) != TARGET_LEVELS_HPA.size
    ):
        raise ValueError(
            f"{ds.filepath()} has an unexpected {dimension_name} dimension"
        )


def read_chunk(var, time_slice, latitude_slice, include_longitude=True):
    slices = {"latitude": latitude_slice}
    if include_longitude and "longitude" not in var.dimensions:
        raise ValueError(f"{var.name} has no longitude dimension")
    key = source_key_for_dims(var.dimensions, slices, time_slice)
    return as_float_array(var[key], get_fill_value(var))


def validate_required_latlon_structure(ds):
    require_target_coordinate(ds, is_zm=False)
    required = {
        "pressure": ("Time", "nVertLevels", "latitude", "longitude"),
        "rho": ("Time", "nVertLevels", "latitude", "longitude"),
        "w": ("Time", "nVertLevelsP1", "latitude", "longitude"),
        "omg_isobaric": (
            "Time",
            "nIsoLevelsZ",
            "latitude",
            "longitude",
        ),
    }
    for name, expected_dims in required.items():
        if name not in ds.variables:
            raise KeyError(f"{ds.filepath()} is missing required variable {name!r}")
        if ds.variables[name].dimensions != expected_dims:
            raise ValueError(
                f"{ds.filepath()}:{name} has dimensions "
                f"{ds.variables[name].dimensions}; expected {expected_dims}"
            )


def validate_pair_coordinates(lat_ds, zm_ds):
    require_target_coordinate(zm_ds, is_zm=True)
    expected_dims = ("Time", "pres", "latitude")
    if (
        "omg_isobaric" not in zm_ds.variables
        or zm_ds.variables["omg_isobaric"].dimensions != expected_dims
    ):
        raise ValueError(
            f"{zm_ds.filepath()}:omg_isobaric must use {expected_dims}"
        )
    for name in ("Time", "latitude"):
        lat_values = np.asarray(lat_ds.variables[name][:])
        zm_values = np.asarray(zm_ds.variables[name][:])
        if lat_values.shape != zm_values.shape or not np.allclose(
            lat_values,
            zm_values,
            equal_nan=True,
        ):
            raise ValueError(
                f"Paired files disagree on {name}: "
                f"{lat_ds.filepath()} and {zm_ds.filepath()}"
            )


def validate_known_corruption(lat_path, zm_path):
    """Check that a file has the expected bad mapping before any output or patch."""

    with Dataset(lat_path, "r") as lat_ds, Dataset(zm_path, "r") as zm_ds:
        validate_required_latlon_structure(lat_ds)
        validate_pair_coordinates(lat_ds, zm_ds)

        latitude_size = len(lat_ds.dimensions["latitude"])
        longitude_size = len(lat_ds.dimensions["longitude"])
        lat_start = max(0, latitude_size // 2 - 6)
        lon_start = max(0, longitude_size // 2 - 8)
        lat_slice = slice(lat_start, min(latitude_size, lat_start + 12))
        lon_slice = slice(lon_start, min(longitude_size, lon_start + 16))
        slices = {"latitude": lat_slice, "longitude": lon_slice}
        time_slice = slice(0, 1)

        def sample(name):
            var = lat_ds.variables[name]
            key = source_key_for_dims(var.dimensions, slices, time_slice)
            return as_float_array(var[key], get_fill_value(var))

        current = sample("omg_isobaric")
        native = native_target_omega(
            sample("pressure"),
            sample("rho"),
            sample("w"),
        )
        recovered, mismatch = recover_target_omega(current, native)
        stored_range = TARGET_LEVELS_HPA >= RECOVERED_STORED_MINIMUM_HPA
        labelled_score = score_arrays_percent(
            current[:, stored_range, ...],
            native[:, stored_range, ...],
        )
        recovered_score = score_arrays_percent(
            recovered[:, stored_range, ...],
            native[:, stored_range, ...],
        )

        if (
            not np.isfinite(mismatch)
            or mismatch > EXPECTED_925_RELATIVE_RMS_MAX
        ):
            raise ValueError(
                f"{lat_path} does not match the expected interpolation signature: "
                f"the two independent 925 hPa recoveries differ by {mismatch:.6g}"
            )
        if (
            not np.isfinite(labelled_score["rms"])
            or not np.isfinite(recovered_score["rms"])
            or recovered_score["rms"]
            >= MAPPING_IMPROVEMENT_FACTOR_MAX * labelled_score["rms"]
        ):
            raise ValueError(
                f"{lat_path} does not match the known bad pressure mapping "
                "(or may already be fixed): current labelled rms="
                f"{labelled_score['rms']:.6g}%, recovered-mapping rms="
                f"{recovered_score['rms']:.6g}%"
            )

        print(
            f"Validated known omega remapping: {lat_path.name}; "
            f"labelled rms={labelled_score['rms']:.3g}%, "
            f"recovered rms={recovered_score['rms']:.3g}%, "
            f"925 consistency={mismatch:.3g}"
        )


def paired_files():
    latlon_files = sorted(SOURCE_DIRECTORY.glob("latlon.2000-*.nc"))
    if not latlon_files:
        raise FileNotFoundError(
            f"No latlon.2000-*.nc files found in {SOURCE_DIRECTORY}"
        )
    pairs = []
    for lat_path in latlon_files:
        zm_path = SOURCE_DIRECTORY / lat_path.name.replace(
            "latlon.",
            "zm_latlon.",
            1,
        )
        if not zm_path.exists():
            raise FileNotFoundError(f"Missing paired zonal-mean file {zm_path}")
        pairs.append((lat_path, zm_path))
    return pairs


def source_fingerprint(path):
    stat = path.stat()
    return str(stat.st_size), str(stat.st_mtime_ns)


def copy_coordinate(src, dst, name):
    if name not in src.variables:
        return
    src_var = src.variables[name]
    dst_var = dst.createVariable(
        name,
        src_var.datatype,
        src_var.dimensions,
        **variable_creation_kwargs(
            src_var,
            {dim: len(src.dimensions[dim]) for dim in src_var.dimensions},
        ),
    )
    copy_attrs(src_var, dst_var)
    dst_var[:] = src_var[:]


def create_staged_dataset(src, path, is_zm=False):
    omega = src.variables["omg_isobaric"]
    for dim_name in omega.dimensions:
        dim = src.dimensions[dim_name]
        dst_size = None if dim.isunlimited() else len(dim)
        path.createDimension(dim_name, dst_size)
    coordinate_names = (
        ("Time", "pres", "latitude")
        if is_zm
        else ("Time", "z_iso_levels", "latitude", "longitude")
    )
    for name in coordinate_names:
        copy_coordinate(src, path, name)
    planned_sizes = {
        dim: len(src.dimensions[dim])
        for dim in omega.dimensions
    }
    dst_omega = path.createVariable(
        "omg_isobaric",
        omega.datatype,
        omega.dimensions,
        **variable_creation_kwargs(omega, planned_sizes),
    )
    copy_attrs(omega, dst_omega)
    copy_attrs(src, path)
    path.setncattr("recovery_version", RECOVERY_VERSION)
    path.setncattr(
        "recovery_method",
        "recovered stored omega retained at 200-1000 hPa; "
        "native pressure/rho/w interpolation used at 7-150 hPa",
    )
    path.setncattr(
        "native_interpolation_levels_hpa",
        ",".join(
            f"{level:g}"
            for level in TARGET_LEVELS_HPA[
                TARGET_LEVELS_HPA < RECOVERED_STORED_MINIMUM_HPA
            ]
        ),
    )
    path.setncattr(
        "recovered_stored_levels_hpa",
        ",".join(
            f"{level:g}"
            for level in TARGET_LEVELS_HPA[
                TARGET_LEVELS_HPA >= RECOVERED_STORED_MINIMUM_HPA
            ]
        ),
    )
    path.setncattr("created_utc", utc_stamp())
    path.setncattr("source_is_zonal_mean", int(is_zm))


def stage_pair(lat_path, zm_path, time_chunk, latitude_chunk, overwrite):
    STAGING_DIRECTORY.mkdir(parents=True, exist_ok=True)
    lat_output = STAGING_DIRECTORY / lat_path.name
    zm_output = STAGING_DIRECTORY / zm_path.name
    for output in (lat_output, zm_output):
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"{output} exists; pass --overwrite to replace staged files"
            )

    lat_tmp = lat_output.with_name(f".{lat_output.name}.tmp-{os.getpid()}")
    zm_tmp = zm_output.with_name(f".{zm_output.name}.tmp-{os.getpid()}")
    for temporary in (lat_tmp, zm_tmp):
        if temporary.exists():
            temporary.unlink()

    try:
        with (
            Dataset(lat_path, "r") as lat_src,
            Dataset(zm_path, "r") as zm_src,
            Dataset(lat_tmp, "w", format=lat_src.file_format) as lat_dst,
            Dataset(zm_tmp, "w", format=zm_src.file_format) as zm_dst,
        ):
            validate_required_latlon_structure(lat_src)
            validate_pair_coordinates(lat_src, zm_src)
            create_staged_dataset(lat_src, lat_dst, is_zm=False)
            create_staged_dataset(zm_src, zm_dst, is_zm=True)
            for dst, source_path in (
                (lat_dst, lat_path),
                (zm_dst, zm_path),
            ):
                size, mtime_ns = source_fingerprint(source_path)
                dst.setncattr("source_path", str(source_path))
                dst.setncattr("source_size_bytes", size)
                dst.setncattr("source_mtime_ns", mtime_ns)

            nt = len(lat_src.dimensions["Time"])
            nlat = len(lat_src.dimensions["latitude"])
            for time_start in range(0, nt, time_chunk):
                time_stop = min(nt, time_start + time_chunk)
                time_slice = slice(time_start, time_stop)
                print(
                    f"  {lat_path.name}: times {time_start}:{time_stop} of {nt}"
                )
                for lat_start in range(0, nlat, latitude_chunk):
                    lat_stop = min(nlat, lat_start + latitude_chunk)
                    latitude_slice = slice(lat_start, lat_stop)
                    current = read_chunk(
                        lat_src.variables["omg_isobaric"],
                        time_slice,
                        latitude_slice,
                    )
                    pressure = read_chunk(
                        lat_src.variables["pressure"],
                        time_slice,
                        latitude_slice,
                    )
                    rho = read_chunk(
                        lat_src.variables["rho"],
                        time_slice,
                        latitude_slice,
                    )
                    w = read_chunk(
                        lat_src.variables["w"],
                        time_slice,
                        latitude_slice,
                    )
                    native = native_target_omega(pressure, rho, w)
                    recovered, mismatch = recover_target_omega(current, native)
                    if (
                        not np.isfinite(mismatch)
                        or mismatch > EXPECTED_925_RELATIVE_RMS_MAX
                    ):
                        raise ValueError(
                            f"{lat_path}: 925 hPa recovery mismatch {mismatch:.6g} "
                            f"at Time={time_start}:{time_stop}, "
                            f"latitude={lat_start}:{lat_stop}"
                        )

                    lat_key = (
                        time_slice,
                        slice(None),
                        latitude_slice,
                        slice(None),
                    )
                    lat_dst.variables["omg_isobaric"][lat_key] = output_data_array(
                        recovered,
                        lat_dst.variables["omg_isobaric"].dtype,
                        get_fill_value(lat_dst.variables["omg_isobaric"]),
                    )
                    zm_key = (time_slice, slice(None), latitude_slice)
                    zonal_mean = nanmean_last_axis(recovered)
                    zm_dst.variables["omg_isobaric"][zm_key] = output_data_array(
                        zonal_mean,
                        zm_dst.variables["omg_isobaric"].dtype,
                        get_fill_value(zm_dst.variables["omg_isobaric"]),
                    )

        os.replace(lat_tmp, lat_output)
        os.replace(zm_tmp, zm_output)
        print(f"Wrote staged recovery {lat_output}")
        print(f"Wrote staged recovery {zm_output}")
        write_original_style_comparison_figures(
            lat_path,
            zm_path,
            lat_output,
            zm_output,
            time_chunk,
        )
    finally:
        for temporary in (lat_tmp, zm_tmp):
            if temporary.exists():
                temporary.unlink()


def validate_staged_file(source_path, staged_path):
    if not staged_path.exists():
        raise FileNotFoundError(
            f"Missing staged recovery {staged_path}; run stage mode first"
        )
    with Dataset(source_path, "r") as source, Dataset(staged_path, "r") as staged:
        if staged.getncattr("recovery_version") != RECOVERY_VERSION:
            raise ValueError(
                f"{staged_path} has recovery version "
                f"{staged.getncattr('recovery_version')!r}, expected "
                f"{RECOVERY_VERSION!r}"
            )
        size, mtime_ns = source_fingerprint(source_path)
        if (
            staged.getncattr("source_path") != str(source_path)
            or staged.getncattr("source_size_bytes") != size
            or staged.getncattr("source_mtime_ns") != mtime_ns
        ):
            raise ValueError(
                f"{source_path} changed after {staged_path} was created; "
                "rerun stage mode"
            )
        source_omega = source.variables["omg_isobaric"]
        staged_omega = staged.variables["omg_isobaric"]
        if (
            source_omega.dimensions != staged_omega.dimensions
            or source_omega.shape != staged_omega.shape
        ):
            raise ValueError(
                f"Staged omega structure does not match {source_path}"
            )


def validate_source_is_patchable(source_path):
    """Confirm read-write NetCDF access before any source field is changed."""

    try:
        with Dataset(source_path, "r+") as source:
            if "omg_isobaric" not in source.variables:
                raise KeyError(f"{source_path} has no omg_isobaric variable")
    except OSError as exc:
        raise OSError(
            f"Cannot open {source_path} read-write. Ensure no diagnostic is "
            "currently reading the T_nudge files and that HDF5 file locking "
            "was disabled before netCDF4 was imported."
        ) from exc


def patch_file(source_path, staged_path, time_chunk, latitude_chunk):
    with Dataset(source_path, "r+") as source, Dataset(staged_path, "r") as staged:
        source_omega = source.variables["omg_isobaric"]
        staged_omega = staged.variables["omg_isobaric"]
        nt = source_omega.shape[0]
        nlat = source_omega.shape[-2] if "longitude" in source_omega.dimensions else source_omega.shape[-1]
        for time_start in range(0, nt, time_chunk):
            time_stop = min(nt, time_start + time_chunk)
            for lat_start in range(0, nlat, latitude_chunk):
                lat_stop = min(nlat, lat_start + latitude_chunk)
                if "longitude" in source_omega.dimensions:
                    key = (
                        slice(time_start, time_stop),
                        slice(None),
                        slice(lat_start, lat_stop),
                        slice(None),
                    )
                else:
                    key = (
                        slice(time_start, time_stop),
                        slice(None),
                        slice(lat_start, lat_stop),
                    )
                source_omega[key] = staged_omega[key]
        source_omega.setncattr("recovery_version", RECOVERY_VERSION)
        source_omega.setncattr(
            "recovery_method",
            "recovered stored omega retained at 200-1000 hPa; "
            "native pressure/rho/w interpolation used at 7-150 hPa",
        )
        old_history = (
            source.getncattr("history") if "history" in source.ncattrs() else ""
        )
        message = (
            f"{utc_stamp()}: patched only omg_isobaric from {staged_path} "
            f"using {Path(__file__).name}"
        )
        source.setncattr(
            "history",
            f"{old_history}\n{message}" if old_history else message,
        )
    print(f"Patched omg_isobaric only: {source_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=("stage", "patch"),
        help="Stage corrected omega-only files, or patch originals from them",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=1,
        help="Number of time records processed together",
    )
    parser.add_argument(
        "--latitude-chunk",
        type=int,
        default=30,
        help="Number of latitudes processed together",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing staged files (stage mode only)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.time_chunk < 1 or args.latitude_chunk < 1:
        raise SystemExit("--time-chunk and --latitude-chunk must be at least 1")
    if args.mode == "patch" and args.overwrite:
        raise SystemExit("--overwrite applies only to stage mode")

    pairs = paired_files()
    if args.mode == "stage":
        print(f"Source:  {SOURCE_DIRECTORY}")
        print(f"Staging: {STAGING_DIRECTORY}")
        print(f"Files:   {len(pairs)} full/zonal-mean pairs")
        print("Preflighting every file before creating staged output")
        for lat_path, zm_path in pairs:
            validate_known_corruption(lat_path, zm_path)
        for lat_path, zm_path in pairs:
            stage_pair(
                lat_path,
                zm_path,
                args.time_chunk,
                args.latitude_chunk,
                args.overwrite,
            )
        return 0

    print(
        "Validating every staged file and every read-write source open before "
        "patching the first source file"
    )
    for lat_path, zm_path in pairs:
        validate_staged_file(lat_path, STAGING_DIRECTORY / lat_path.name)
        validate_staged_file(zm_path, STAGING_DIRECTORY / zm_path.name)
        validate_source_is_patchable(lat_path)
        validate_source_is_patchable(zm_path)
    print(
        "Preflight complete. Do not run diagnostics that read T_nudge until "
        "patching has finished."
    )
    for lat_path, zm_path in pairs:
        patch_file(
            lat_path,
            STAGING_DIRECTORY / lat_path.name,
            args.time_chunk,
            args.latitude_chunk,
        )
        patch_file(
            zm_path,
            STAGING_DIRECTORY / zm_path.name,
            args.time_chunk,
            args.latitude_chunk,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
