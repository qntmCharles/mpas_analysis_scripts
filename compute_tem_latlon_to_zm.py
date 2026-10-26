#!/usr/bin/env python
"""Calculate versioned, unfolded TEM fields before downstream analysis caches.

Store mass-flux HPE(z) TEM in simulation-local companion files.
Retain the pressure producer only for explicit legacy archive work. See diagnostics/tem_core.py for equations.
"""

from __future__ import annotations

import argparse
import faulthandler
import glob
import os
from pathlib import Path
from time import perf_counter
import uuid
import warnings

if __name__ == "__main__":
    print("TEM producer: loading numerical libraries and archive helpers", flush=True)
    # Avoid background stack inspection during numerical and NetCDF operations.
    # Keep a traceback for an actual fatal error in the failing thread.
    faulthandler.enable(all_threads=False)

import numpy as np
from dask.diagnostics import ProgressBar
import xarray as xr


from diagnostics.tem_core import (
    R, CP_DRY_AIR, M_AIR, EARTH_RADIUS, OMEGA, RD, KAPPA, P0_PA,
    TEM_FORMULATION, TEM_VERSION, ISOBARIC_FIELDS, PRESSURE_REQUIRED, compute_height_tem, compute_pressure_tem,
    differentiate_wrt,
)
from diagnostics.cache_support import (
    cache_manifest, cache_manifest_matches, materialize, read_json_file,
    write_json_file, write_netcdf_atomic,
)
from diagnostics.archive_coordinates import standardize_latlon_dataset
from diagnostics.analysis_config import LATLON_PATTERN
from diagnostics.tem_native import (
    model_level_field, zfull_from_latlon, midlevel_vertical_velocity, native_tem_fields,
    open_native_dataset, iter_native_frames, NATIVE_FIELDS, WATER_SPECIES, PREPARED_FIELDS,
)

REFERENCE_SCRIPT = "/data/phh1-hosted/cwp29/mpas-a/analysis/TEM/compute_tem_new.py"
TEM_VARIABLES = (
    "t",
    "rho_zm",
    "upvp",
    "upwp",
    "vptp",
    "vpthp",
    "u_bar", "v_bar", "w_bar", "theta_bar", "u_z", "theta_z", "absolute_vorticity",
    "momentum_meridional", "momentum_vertical", "heat_meridional", "heat_vertical",
    "EPFd_momentum", "EPFd_heat", "density_wind_covariance", "rho_dry_zm",
    "uv", "uw", "vtheta", "valid_zonal_sample",
    "Fy",
    "Fz",
    "EPFd",
    "EPFd_y",
    "EPFd_z",
    "vres",
    "wres",
)
TIME_DIM_CANDIDATES = ("Time", "time")


def finite_or_nan(da: xr.DataArray, max_abs: float = 1e30) -> xr.DataArray:
    """Mask NaNs, infinities, and MPAS-style huge fill values."""
    return da.where(np.isfinite(da) & (np.abs(da) < max_abs))


def vertical_dim(da: xr.DataArray) -> str:
    for dim in ("z", "nVertLevels", "nVertLevelsP1"):
        if dim in da.dims:
            return dim
    raise ValueError(f"No known vertical dimension found in {da.name!r}: {da.dims!r}")


def time_dim(obj: xr.Dataset | xr.DataArray) -> str:
    for dim in TIME_DIM_CANDIDATES:
        if dim in obj.dims:
            return dim
    raise ValueError(f"No known time dimension found in {obj.dims!r}")


def latitude_is_hemispherically_symmetric(latitude, rtol: float = 1e-7, atol: float = 1e-8) -> bool:
    values = np.asarray(latitude.values if isinstance(latitude, xr.DataArray) else latitude, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        return False
    return np.allclose(values, -values[::-1], rtol=rtol, atol=atol)


def hemispheric_fold(da: xr.DataArray, lat_dim: str = "latitude", parity: str = "even") -> xr.DataArray:
    """Average paired hemispheres with even or odd reflection parity."""
    if lat_dim not in da.dims:
        return da
    if parity not in ("even", "odd"):
        raise ValueError(f"Unknown hemispheric parity {parity!r}; expected 'even' or 'odd'")
    if not latitude_is_hemispherically_symmetric(da[lat_dim]):
        warnings.warn(
            f"Skipping hemispheric fold for {da.name!r}: {lat_dim!r} is not symmetric about the equator."
        )
        return da

    reflected = da.isel({lat_dim: slice(None, None, -1)})
    reflected = reflected.assign_coords({lat_dim: da[lat_dim].values})
    sign = 1.0 if parity == "even" else -1.0
    out = 0.5 * (da + sign * reflected)
    out.name = da.name
    out.attrs.update(da.attrs)
    out.attrs["hemispheric_fold"] = f"{parity} paired-hemisphere average"
    return out


def require_matching_coord(ds: xr.Dataset, target: xr.DataArray, dim: str) -> xr.Dataset:
    if dim not in ds.dims:
        raise ValueError(f"{dim!r} is missing from {ds.encoding.get('source', 'dataset')}")
    if ds.sizes[dim] != target.size:
        raise ValueError(f"{dim!r} has size {ds.sizes[dim]}, expected {target.size}")

    if dim in ds.coords:
        source_values = ds[dim].values
        target_values = target.values
        if np.issubdtype(source_values.dtype, np.datetime64) or np.issubdtype(
            target_values.dtype, np.datetime64
        ):
            matches = np.array_equal(source_values, target_values)
        else:
            matches = np.allclose(source_values, target_values, rtol=1e-7, atol=1e-8)
        if not matches:
            raise ValueError(f"{dim!r} coordinate does not match the target zm file")

    return ds.assign_coords({dim: target})


def open_dataset(path: str | Path, **kwargs) -> xr.Dataset:
    """Open a dataset, falling back to raw time values if CF time decoding fails."""
    # Later Dask rechunking must read slices, not retain whole monthly arrays.
    kwargs.setdefault("cache", False)
    try:
        return xr.open_dataset(path, **kwargs)
    except ValueError as exc:
        if "Failed to decode variable" not in str(exc):
            raise
        return xr.open_dataset(path, decode_times=False, **kwargs)


def compute_tem_dataset(
    latlon_ds: xr.Dataset,
    zm_coords: dict[str, xr.DataArray],
    hemispheric_fold_fields: bool = False,
) -> xr.Dataset:
    """Calculate mass-flux HPE(z) TEM before any presentation folding."""
    target_time_dim = "Time" if "Time" in zm_coords else "time"
    source_time_dim = time_dim(latlon_ds)
    if source_time_dim != target_time_dim:
        latlon_ds = latlon_ds.rename({source_time_dim: target_time_dim})
    latlon_ds = latlon_ds.sel({target_time_dim: zm_coords[target_time_dim]})
    latlon_ds = require_matching_coord(latlon_ds, zm_coords["latitude"], "latitude")
    z = zm_coords["z"]
    fields = native_tem_fields(latlon_ds, z)
    fields["rho_dry_zm"] = model_level_field(latlon_ds, "rho", z).mean("longitude", skipna=False)
    return tem_from_prepared_fields(fields, hemispheric_fold_fields)


def tem_from_prepared_fields(fields, hemispheric_fold_fields=False):
    """Compute TEM eagerly from one prepared sample or a small reference dataset."""
    out = compute_height_tem(*(fields[name] for name in PREPARED_FIELDS))
    out["rho_dry_zm"] = fields.rho_dry_zm
    out.rho_dry_zm.attrs.update(units="kg m-3", long_name="Zonal mean dry-air density",
                               tem_version=TEM_VERSION, tem_formulation="geometric-height")
    out.attrs["density_convention"] = fields.attrs["density_convention"]
    if hemispheric_fold_fields:
        odd = {"v_bar", "vres", "Fy", "uv", "vtheta", "upvp", "vpthp", "vptp", "absolute_vorticity"}
        for name in out:
            out[name] = hemispheric_fold(out[name], parity="odd" if name in odd else "even")
        out.attrs["hemispheric_fold"] = "completed diagnostics"
    out.attrs["legacy_reference_script"] = REFERENCE_SCRIPT
    return out


def write_height_tem_companion(latlon_path, output_path, args, zm_coords):
    """Read, calculate, and append each time sample once, then publish atomically."""
    from netCDF4 import Dataset
    from xarray.coding.times import encode_cf_datetime

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    manifest = producer_manifest(latlon_path, args)
    target_time = "Time" if "Time" in zm_coords else "time"
    writer = None
    try:
        with open_native_dataset(latlon_path) as raw:
            names = [*NATIVE_FIELDS, *[name for name in WATER_SPECIES if name in raw]]
            source = standardize_latlon_dataset(raw, names)
            if "zgrid" in raw:
                source["zgrid"] = xr.DataArray(zfull_from_latlon(raw), dims="nVertLevelsP1")
            if target_time != "Time":
                source = source.rename(Time=target_time)
            time_indices = source.get_index(target_time).get_indexer(zm_coords[target_time].values)
            if not time_indices.size or np.any(time_indices < 0):
                raise ValueError("Native TEM source lacks requested time samples")
            if np.all(np.diff(time_indices) == 1):
                source = source.isel({target_time: slice(int(time_indices[0]), int(time_indices[-1])+1)})
            else:
                source = source.isel({target_time: time_indices})
            source = require_matching_coord(source, zm_coords["latitude"], "latitude")
            if not source.sizes[target_time]:
                raise ValueError("Native TEM source has no requested time samples")
            print(f"  Sequential TEM calculation: {source.sizes[target_time]} samples; complete height columns", flush=True)
            frames = iter(iter_native_frames(source, zm_coords["z"], scheduler=args.scheduler))
            for index in range(source.sizes[target_time]):
                started = perf_counter()
                print(f"  Native read {index+1}/{source.sizes[target_time]}", flush=True)
                fields = next(frames)
                read_seconds = perf_counter()-started
                calculation_start = perf_counter()
                result = tem_from_prepared_fields(fields, getattr(args, "hemispheric_fold", False))
                result.attrs["manifest_sha256"] = manifest["manifest_sha256"]
                if writer is None:
                    encoding = output_encoding(result, args.output_dtype)
                    if np.issubdtype(result[target_time].dtype, np.datetime64):
                        encoding[target_time] = {"dtype": "float64"}
                    write_netcdf_atomic(result, temporary, encoding=encoding,
                                        unlimited_dims=[target_time], materialize_first=False)
                    writer = Dataset(temporary, "a")
                else:
                    for name, field in result.data_vars.items():
                        variable = writer[name]
                        key = tuple(slice(index, index+1) if dim == target_time else slice(None)
                                    for dim in variable.dimensions)
                        variable[key] = field.transpose(*variable.dimensions).values
                    time_variable = writer[target_time]
                    times = result[target_time].values
                    if np.issubdtype(times.dtype, np.datetime64):
                        times = encode_cf_datetime(times, time_variable.units,
                                                   getattr(time_variable, "calendar", "standard"),
                                                   dtype=np.float64)[0]
                    time_variable[index:index+1] = times
                del result, fields
                print(f"  TEM sample {index+1}/{source.sizes[target_time]}: read {read_seconds:.1f} s; "
                      f"calculate and store {perf_counter()-calculation_start:.1f} s", flush=True)
        writer.close()
        writer = None
        os.replace(temporary, output_path)
        write_json_file(str(output_path)+".metadata.json", manifest)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
    print(f"Stored native-height TEM: {output_path}", flush=True)

def load_zm_coords(zm_path: str | Path) -> tuple[dict[str, xr.DataArray], set[str]]:
    with open_dataset(zm_path) as zds:
        zds_time_dim = time_dim(zds)
        missing = [coord for coord in ("latitude", "z") if coord not in zds]
        if missing:
            raise ValueError(f"{zm_path} is missing required coordinates: {missing}")
        coords = {coord: zds[coord].load() for coord in (zds_time_dim, "latitude", "z")}
        existing = set(zds.data_vars)
    return coords, existing


def chunks_for(ds: xr.Dataset, args: argparse.Namespace) -> dict[str, int]:
    requested = {
        "Time": args.time_chunk,
        "time": args.time_chunk,
        "latitude": args.latitude_chunk,
        "longitude": args.longitude_chunk,
        "nVertLevels": -1,
        "nVertLevelsP1": -1,
        "z": -1,
        "nIsoLevelsT": -1, "nIsoLevelsU": -1, "nIsoLevelsZ": -1,
    }
    return {dim: chunk for dim, chunk in requested.items() if dim in ds.dims and chunk is not None}


def output_encoding(ds: xr.Dataset, dtype: str | None) -> dict[str, dict[str, object]]:
    encoding: dict[str, dict[str, object]] = {}
    for name in ds.data_vars:
        enc: dict[str, object] = {}
        if dtype:
            enc["dtype"] = dtype
        encoding[name] = enc
    if "valid_zonal_sample" in encoding:
        encoding["valid_zonal_sample"] = {"dtype": "int8", "_FillValue": None}
    return encoding


def compute_with_scheduler(ds: xr.Dataset, scheduler: str | None) -> xr.Dataset:
    if scheduler is None:
        return ds.load()
    import dask

    with dask.config.set(scheduler=scheduler):
        return ds.load()


def to_netcdf_with_scheduler(ds: xr.Dataset, scheduler: str | None, *args, **kwargs) -> None:
    if scheduler is None:
        ds.to_netcdf(*args, **kwargs)
        return
    import dask

    with dask.config.set(scheduler=scheduler):
        ds.to_netcdf(*args, **kwargs)


def variables_to_write(tem_ds: xr.Dataset, existing: set[str], overwrite: bool) -> xr.Dataset:
    conflicts = sorted(name for name in tem_ds.data_vars if name in existing)
    if conflicts and not overwrite:
        raise ValueError(
            "Target zm file already contains TEM variables "
            f"{conflicts}; rerun with --overwrite to replace them"
        )
    return tem_ds


def ensure_tem_variables(zm_path: Path, tem_ds: xr.Dataset, dtype: str, overwrite: bool) -> None:
    from netCDF4 import Dataset

    with Dataset(zm_path, "a") as nc:
        # NetCDF cannot change an existing variable's type during append.
        for name, da in tem_ds.data_vars.items():
            if name not in nc.variables:
                continue
            var = nc.variables[name]
            expected_dtype = np.dtype("int8" if name == "valid_zonal_sample" else dtype)
            if np.dtype(var.dtype) != expected_dtype:
                raise ValueError(f"Existing {name!r} uses {var.dtype}; requested {expected_dtype}. "
                                 "Use --output-dir to write a corrected copy with the requested precision.")
        for name, da in tem_ds.data_vars.items():
            dims = tuple(da.dims)
            missing_dims = [dim for dim in dims if dim not in nc.dimensions]
            if missing_dims:
                raise ValueError(f"{zm_path} is missing dimensions needed for {name}: {missing_dims}")

            if name in nc.variables:
                var = nc.variables[name]
                if not overwrite:
                    raise ValueError(f"{zm_path} already contains {name!r}; rerun with --overwrite")
                if tuple(var.dimensions) != dims:
                    raise ValueError(f"Existing {name!r} has dimensions {var.dimensions}, expected {dims}")
            else:
                var = nc.createVariable(name, "int8" if name == "valid_zonal_sample" else dtype, dims)

            for attr, value in da.attrs.items():
                var.setncattr(attr, value)


def write_tem_block(zm_path: Path, tem_ds: xr.Dataset, start: int, stop: int, dtype: str) -> None:
    from netCDF4 import Dataset

    with Dataset(zm_path, "a") as nc:
        for name, da in tem_ds.data_vars.items():
            var = nc.variables[name]
            values = da.transpose(*var.dimensions).values.astype(dtype, copy=False)
            var[start:stop, :, :] = values


def append_tem_to_zm_by_time_blocks(
    latlon_ds: xr.Dataset,
    zm_path: Path,
    args: argparse.Namespace,
    zm_coords: dict[str, xr.DataArray],
    existing: set[str],
) -> None:
    target_time_dim = "Time" if "Time" in zm_coords else "time"
    time_coord = zm_coords[target_time_dim]
    ntime = time_coord.size
    block_size = max(1, int(args.time_block_size))
    created = False

    for start in range(0, ntime, block_size):
        stop = min(start + block_size, ntime)
        print(f"  Computing time block {start}:{stop} of {ntime}")
        block_coords = dict(zm_coords)
        block_coords[target_time_dim] = time_coord.isel({target_time_dim: slice(start, stop)})
        tem_block = compute_tem_dataset(
            latlon_ds,
            block_coords,
            hemispheric_fold_fields=getattr(args, "hemispheric_fold", False),
        )
        tem_block = variables_to_write(tem_block, existing, args.overwrite)

        if not created:
            ensure_tem_variables(zm_path, tem_block, args.output_dtype, args.overwrite)
            created = True

        loaded = compute_with_scheduler(tem_block, args.scheduler)
        write_tem_block(zm_path, loaded, start, stop, args.output_dtype)
        loaded.close()


def append_tem_to_zm(
    latlon_path: str | Path,
    zm_path: str | Path,
    args: argparse.Namespace,
    output_path: str | Path | None = None,
) -> None:
    latlon_path = Path(latlon_path)
    zm_path = Path(zm_path)
    output_path = Path(output_path) if output_path is not None else zm_path

    if getattr(args, "resume", False) and completed_tem_output(latlon_path, output_path, args):
        print(f"Reusing completed native-height TEM: {output_path}", flush=True)
        return

    if output_path != zm_path and output_path.exists() and not (args.overwrite or args.dry_run):
        raise FileExistsError(f"{output_path} already exists; rerun with --overwrite")

    print(f"Processing {latlon_path.name} -> {output_path}")
    metadata_path = Path(str(output_path) + ".metadata.json")
    zm_coords, existing = load_zm_coords(zm_path)

    if args.dry_run:
        conflicts = sorted(name for name in TEM_VARIABLES if name in existing)
        if conflicts:
            print(f"  Existing TEM variables: {', '.join(conflicts)}")
        return

    if output_path != zm_path:
        write_height_tem_companion(latlon_path, output_path, args, zm_coords)
        return

    if metadata_path.exists():
        metadata_path.unlink()

    latlon_ds = open_dataset(latlon_path, chunks=None)
    try:
        latlon_ds = latlon_ds.chunk(chunks_for(latlon_ds, args))
        if output_path == zm_path:
            if args.time_block_size is not None:
                append_tem_to_zm_by_time_blocks(latlon_ds, zm_path, args, zm_coords, existing)
            else:
                tem_ds = compute_tem_dataset(
                    latlon_ds,
                    zm_coords,
                    hemispheric_fold_fields=getattr(args, "hemispheric_fold", False),
                )
                tem_ds = variables_to_write(tem_ds, existing, args.overwrite)
                ensure_tem_variables(zm_path, tem_ds, args.output_dtype, args.overwrite)
                encoding = output_encoding(tem_ds, args.output_dtype)
                # Do not rewrite coordinate variables or global attributes when appending.
                to_write = tem_ds.drop_vars(
                    [coord for coord in ("Time", "time", "latitude", "z") if coord in tem_ds.coords]
                )
                to_write.attrs = {}
                to_netcdf_with_scheduler(
                    to_write,
                    args.scheduler,
                    zm_path,
                    mode="a",
                    engine="netcdf4",
                    encoding=encoding,
                    compute=True,
                )
    finally:
        latlon_ds.close()
    write_json_file(metadata_path, producer_manifest(latlon_path, args))


def producer_manifest(path, args):
    return cache_manifest(
        "compute_tem_latlon_to_zm", schema=TEM_VERSION,
        configuration={"formulation": args.tem_formulation,
                       "hemispheric_fold": getattr(args, "hemispheric_fold", False),
                       "output_dtype": args.output_dtype,
                       "start": getattr(args, "start", None), "end": getattr(args, "end", None)},
        source_paths=[path],
    )


def completed_tem_output(source_path, output_path, args):
    """Reuse only a complete archive with the current inputs and scientific settings."""
    if not output_path.is_file():
        return False
    expected = producer_manifest(source_path, args)
    try:
        saved = read_json_file(str(output_path) + ".metadata.json")
        if not cache_manifest_matches(saved, expected):
            return False
        with open_dataset(output_path, chunks=None) as stored:
            required = set(PRESSURE_REQUIRED)
            if args.tem_formulation == "geometric-height":
                required.add("density_wind_covariance")
            if not required.issubset(stored):
                return False
            if stored.attrs.get("manifest_sha256") != expected["manifest_sha256"]:
                return False
            return (
                stored.EPFd.attrs.get("tem_version") == TEM_VERSION
                and stored.EPFd.attrs.get("tem_formulation") == args.tem_formulation
                and all(stored[name].dtype == np.dtype(args.output_dtype)
                        for name in required if name != "valid_zonal_sample")
            )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def write_pressure_tem(path, output_path, args):
    if args.output_dtype != "float64":
        raise ValueError("Pressure TEM requires --output-dtype float64 for decomposition closure")
    if args.hemispheric_fold:
        raise ValueError("Pressure TEM archives require unfolded hemispheres")
    if getattr(args, "resume", False) and completed_tem_output(path, output_path, args):
        print(f"Reusing completed pressure TEM: {output_path}", flush=True)
        return
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite")
    print(f"Pressure TEM: {path} -> {output_path}", flush=True)
    if args.dry_run:
        return
    metadata_path = Path(str(output_path) + ".metadata.json")
    if metadata_path.exists():
        metadata_path.unlink()
    print("  Opening source NetCDF", flush=True)
    with open_dataset(path, chunks=None) as raw:
        print("  Preparing pressure coordinates and time selection", flush=True)
        dataset = standardize_latlon_dataset(raw, ISOBARIC_FIELDS)
        if args.start or args.end:
            dataset = dataset.sel(Time=slice(args.start, args.end))
        if dataset.sizes["Time"] == 0:
            print("  No records in the requested period; skipping", flush=True)
            return
        dataset = dataset.chunk({"Time": args.time_chunk, "latitude": args.latitude_chunk,
                                 "longitude": args.longitude_chunk, "pres": -1})
        print(f"  Building TEM computation: {dataset.sizes['Time']} records, "
              f"{dataset.sizes['pres']} levels, {dataset.sizes['latitude']} latitudes", flush=True)
        result = compute_pressure_tem(dataset)
        manifest = producer_manifest(path, args)
        result.attrs["manifest_sha256"] = manifest["manifest_sha256"]
        encoding = output_encoding(result, args.output_dtype)
        encoding["valid_zonal_sample"] = {"dtype": "int8", "_FillValue": None}
        print(f"  Computing pressure TEM (scheduler: {args.scheduler})", flush=True)
        with ProgressBar(dt=10):
            result = materialize(result, scheduler=args.scheduler, num_workers=1)
        print("  Writing completed fields to NetCDF", flush=True)
        write_netcdf_atomic(result, output_path, encoding=encoding, materialize_first=False,
                            scheduler=args.scheduler, num_workers=1)
    write_json_file(metadata_path, manifest)
    print(f"Stored pressure TEM: {output_path}", flush=True)


def infer_zm_path(latlon_path: str | Path, zm_pattern: str | None = None) -> Path | None:
    latlon_path = Path(latlon_path)
    name = latlon_path.name
    if name.startswith("latlon"):
        candidate = latlon_path.with_name("zm_latlon" + name[len("latlon") :])
        if candidate.exists():
            return candidate

    if zm_pattern is not None:
        candidates = sorted(latlon_path.parent.glob(zm_pattern))
        if len(candidates) == 1:
            return candidates[0]
    return None


def find_pairs(data_dir: str | Path, latlon_pattern: str, zm_pattern: str | None) -> list[tuple[Path, Path]]:
    data_dir = Path(data_dir)
    latlon_files = sorted(Path(path) for path in glob.glob(str(data_dir / latlon_pattern)))
    pairs: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    for latlon_path in latlon_files:
        zm_path = infer_zm_path(latlon_path, zm_pattern)
        if zm_path is None:
            missing.append(latlon_path)
        else:
            pairs.append((latlon_path, zm_path))

    if missing:
        missing_text = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Could not infer zm file for:\n  {missing_text}")
    if not pairs:
        raise FileNotFoundError(f"No files matched {data_dir / latlon_pattern}")
    return pairs


def output_path_for(zm_path: Path, data_dir: Path, output_dir: str | Path | None) -> Path:
    output_dir = Path(output_dir) if output_dir is not None else data_dir
    try:
        rel = zm_path.relative_to(data_dir)
    except ValueError:
        rel = Path(zm_path.name)
    if output_dir.absolute() == data_dir.absolute():
        rel = rel.with_name("tem_geometric_height" + rel.name[len("zm_latlon"):])
    return output_dir / rel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate and store native-height TEM from MPAS latlon histories."
        )
    )
    parser.add_argument("data_dir", help="Source latlon directory; native-height mode also requires matching zm_latlon files")
    parser.add_argument(
        "--latlon-pattern", default=LATLON_PATTERN,
        help="Glob for source latlon files; default: dated production files. Select spin-up files explicitly.",
    )
    parser.add_argument(
        "--zm-pattern",
        default=None,
        help="Optional glob used only when the matching zm_latlon name cannot be inferred",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write TEM files to this directory; default: the source simulation directory",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing TEM variables/output files")
    parser.add_argument("--resume", action="store_true", help="Reuse completed output with matching source provenance and scientific settings")
    parser.add_argument("--dry-run", action="store_true", help="List matched files without computing")
    parser.add_argument(
        "--no-hemispheric-fold",
        action="store_true",
        help="Store unfolded hemispheres (the default).",
    )
    parser.add_argument("--hemispheric-fold", action="store_true", help="Fold completed diagnostics; analysis owners require unfolded archives.")
    parser.add_argument("--tem-formulation", choices=("log-pressure", "geometric-height"), default=TEM_FORMULATION,
                        help="Use geometric-height for current analyses; log-pressure serves legacy archive work")
    parser.add_argument("--start", help="First time for log-pressure output")
    parser.add_argument("--end", help="Last time for log-pressure output")
    parser.add_argument("--time-chunk", type=int, default=1, help="Legacy pressure/in-place Dask chunk length for Time")
    parser.add_argument("--latitude-chunk", type=int, default=30, help="Legacy pressure/in-place Dask chunk length for latitude")
    parser.add_argument("--longitude-chunk", type=int, default=180, help="Legacy pressure/in-place Dask chunk length for longitude")
    parser.add_argument(
        "--time-block-size",
        type=int,
        default=1,
        help="Number of time samples to compute and append at once for in-place writes",
    )
    parser.add_argument(
        "--full-file-write",
        action="store_true",
        help="Disable block writing and let xarray append all times in one operation",
    )
    parser.add_argument(
        "--scheduler",
        choices=("single-threaded", "synchronous", "threads", "processes"),
        default="single-threaded",
        help="Dask scheduler used during compute/write",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float64"),
        default="float64",
        help="NetCDF dtype for TEM variables; float64 preserves decomposition closure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hemispheric_fold and args.no_hemispheric_fold:
        raise ValueError("Choose only one hemispheric-fold option")
    if args.full_file_write:
        args.time_block_size = None
    data_dir = Path(args.data_dir)
    print(f"Scanning source directory: {data_dir}", flush=True)
    if args.tem_formulation == "log-pressure":
        output_dir = Path(args.output_dir) if args.output_dir is not None else data_dir
        paths = sorted(data_dir.glob(args.latlon_pattern))
        if not paths:
            raise FileNotFoundError(f"No files match {args.latlon_pattern} in {data_dir}")
        print(f"Found {len(paths)} pressure-source file(s)", flush=True)
        for path in paths:
            name = "tem_log_pressure" + path.name[len("latlon"):]
            write_pressure_tem(path, output_dir / name, args)
        return
    pairs = find_pairs(data_dir, args.latlon_pattern, args.zm_pattern)
    print(f"Found {len(pairs)} latlon/zm pair(s)")
    for latlon_path, zm_path in pairs:
        out_path = output_path_for(zm_path, data_dir, args.output_dir)
        append_tem_to_zm(latlon_path, zm_path, args, output_path=out_path)
    print("Done.")


if __name__ == "__main__":
    main()
