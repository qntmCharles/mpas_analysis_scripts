#!/usr/bin/env python
"""Compute height-coordinate TEM diagnostics from latlon files and append to zm files.

This is an xarray/dask translation of
/data/phh1-hosted/cwp29/mpas-a/analysis/TEM/compute_tem_new.py.
The formulas are kept aligned with the PyGeode script, while derivatives use
xarray's nonuniform-coordinate finite differences.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import warnings

import numpy as np
from scipy.interpolate import CubicSpline
import xarray as xr


R = 8.31447215
CP_DRY_AIR = 1005.7
M_AIR = 0.0289644
EARTH_RADIUS = 6371229.0
OMEGA = 2.0 * np.pi / ((23 * 60 + 56) * 60 + 4.1)
RD = R / M_AIR
KAPPA = RD / CP_DRY_AIR
P0_PA = 100000.0

REFERENCE_SCRIPT = "/data/phh1-hosted/cwp29/mpas-a/analysis/TEM/compute_tem_new.py"
TEM_VARIABLES = (
    "t",
    "upvp",
    "upwp",
    "vptp",
    "vpthp",
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


def differentiate_wrt(
    da: xr.DataArray,
    dim: str,
    coordinate_values: np.ndarray | xr.DataArray,
    edge_order: int = 2,
) -> xr.DataArray:
    """Differentiate with respect to a 1D coordinate expressed in physical units."""
    if dim not in da.dims:
        raise ValueError(f"Cannot differentiate {da.name!r}; missing dimension {dim!r}")

    coord = np.asarray(coordinate_values, dtype=np.float64)
    if coord.ndim != 1 or coord.size != da.sizes[dim]:
        raise ValueError(
            f"Derivative coordinate for {dim!r} must be one-dimensional with "
            f"{da.sizes[dim]} values; got shape {coord.shape!r}"
        )
    if da.sizes[dim] <= edge_order:
        raise ValueError(
            f"Need more than edge_order={edge_order} points along {dim!r} to differentiate"
        )

    original_coord = da[dim]
    if hasattr(da.data, "chunks"):
        da = da.chunk({dim: -1})
    out = da.assign_coords({dim: coord}).differentiate(dim, edge_order=edge_order)
    out = out.assign_coords({dim: original_coord})
    out.name = f"d{da.name}" if da.name else None
    return out.transpose(*da.dims)


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
    try:
        return xr.open_dataset(path, **kwargs)
    except ValueError as exc:
        if "Failed to decode variable" not in str(exc):
            raise
        return xr.open_dataset(path, decode_times=False, **kwargs)


def model_level_field(ds: xr.Dataset, name: str, z_coord: xr.DataArray) -> xr.DataArray:
    if name not in ds:
        raise ValueError(f"Missing required variable {name!r}")

    da = finite_or_nan(ds[name])
    lev_dim = vertical_dim(da)
    if lev_dim == "nVertLevelsP1":
        raise ValueError(f"{name!r} is on interface levels; expected model levels")
    if da.sizes[lev_dim] != z_coord.size:
        raise ValueError(
            f"{name!r} has {da.sizes[lev_dim]} vertical levels, expected {z_coord.size}"
        )

    if lev_dim != "z":
        da = da.rename({lev_dim: "z"})
    da = da.assign_coords(z=z_coord)
    return da.transpose(time_dim(da), "z", "latitude", "longitude")


def zfull_from_latlon(ds: xr.Dataset) -> np.ndarray:
    if "zgrid" not in ds:
        raise ValueError("Missing required variable 'zgrid' for vertical interpolation of w")
    zgrid = finite_or_nan(ds["zgrid"])
    if "latitude" in zgrid.dims:
        zgrid = zgrid.sel(latitude=0.0, method="nearest")
    if "longitude" in zgrid.dims:
        zgrid = zgrid.sel(longitude=0.0, method="nearest")
    zgrid = zgrid.squeeze()
    z_dim = vertical_dim(zgrid)
    if zgrid.sizes[z_dim] < 4:
        raise ValueError("Need at least four interface levels for cubic-spline w interpolation")
    return np.asarray(zgrid.values, dtype=np.float64)


def _natural_cubic_interp(values: np.ndarray, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
    return CubicSpline(x_in, values, axis=-1, bc_type="natural", extrapolate=False)(x_out)


def midlevel_vertical_velocity(ds: xr.Dataset, z_coord: xr.DataArray) -> xr.DataArray:
    if "w" not in ds:
        raise ValueError("Missing required variable 'w'")

    w = finite_or_nan(ds["w"])
    lev_dim = vertical_dim(w)
    if w.sizes[lev_dim] == z_coord.size + 1:
        z_full = zfull_from_latlon(ds)
        if z_full.size != w.sizes[lev_dim]:
            raise ValueError(
                f"zgrid has {z_full.size} interface levels, but 'w' has {w.sizes[lev_dim]}"
            )
        if hasattr(w.data, "chunks"):
            w = w.chunk({lev_dim: -1})
        w = xr.apply_ufunc(
            _natural_cubic_interp,
            w,
            xr.DataArray(z_full, dims=(lev_dim,)),
            xr.DataArray(np.asarray(z_coord.values, dtype=np.float64), dims=("z",)),
            input_core_dims=[[lev_dim], [lev_dim], ["z"]],
            output_core_dims=[["z"]],
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={"output_sizes": {"z": z_coord.size}, "allow_rechunk": True},
        )
    elif w.sizes[lev_dim] != z_coord.size:
        raise ValueError(f"'w' has {w.sizes[lev_dim]} levels, expected {z_coord.size} or {z_coord.size + 1}")

    if lev_dim in w.dims and lev_dim != "z":
        w = w.rename({lev_dim: "z"})
    w = w.assign_coords(z=z_coord)
    w.name = "w"
    return w.transpose(time_dim(w), "z", "latitude", "longitude")


def compute_tem_dataset(
    latlon_ds: xr.Dataset,
    zm_coords: dict[str, xr.DataArray],
    hemispheric_fold_fields: bool = True,
) -> xr.Dataset:
    """Compute the variables written by compute_tem_new.py on height levels."""
    target_time_dim = "Time" if "Time" in zm_coords else "time"
    source_time_dim = time_dim(latlon_ds)
    if source_time_dim != target_time_dim:
        latlon_ds = latlon_ds.rename({source_time_dim: target_time_dim})

    time = zm_coords[target_time_dim]
    latitude = zm_coords["latitude"]
    z = zm_coords["z"]

    latlon_ds = latlon_ds.sel({target_time_dim: time})
    latlon_ds = require_matching_coord(latlon_ds, latitude, "latitude")

    u = model_level_field(latlon_ds, "uReconstructZonal", z)
    v = model_level_field(latlon_ds, "uReconstructMeridional", z)
    theta = model_level_field(latlon_ds, "theta", z)
    rho = model_level_field(latlon_ds, "rho", z)
    pressure = model_level_field(latlon_ds, "pressure", z)
    w = midlevel_vertical_velocity(latlon_ds, z)

    temperature = theta * (pressure / P0_PA) ** KAPPA
    temperature.name = "t"

    print("  Computing zonal means")
    u_zm_raw = u.mean("longitude", skipna=True)
    v_zm_raw = v.mean("longitude", skipna=True)
    theta_zm_raw = theta.mean("longitude", skipna=True)
    t_zm_raw = temperature.mean("longitude", skipna=True)
    w_zm_raw = w.mean("longitude", skipna=True)
    rho_zm_raw = rho.mean("longitude", skipna=True)

    print("  Computing eddy fluxes")
    upvp = ((u - u_zm_raw) * (v - v_zm_raw)).mean("longitude", skipna=True)
    upwp = ((u - u_zm_raw) * (w - w_zm_raw)).mean("longitude", skipna=True)
    vptp = ((temperature - t_zm_raw) * (v - v_zm_raw)).mean("longitude", skipna=True)
    vpthp = ((theta - theta_zm_raw) * (v - v_zm_raw)).mean("longitude", skipna=True)

    if hemispheric_fold_fields:
        print("  Applying hemispheric paired averages")
        u_zm = hemispheric_fold(u_zm_raw, parity="even")
        v_zm = hemispheric_fold(v_zm_raw, parity="odd")
        theta_zm = hemispheric_fold(theta_zm_raw, parity="even")
        t_zm = hemispheric_fold(t_zm_raw, parity="even")
        w_zm = hemispheric_fold(w_zm_raw, parity="even")
        rho_zm = hemispheric_fold(rho_zm_raw, parity="even")
        upvp = hemispheric_fold(upvp, parity="odd")
        upwp = hemispheric_fold(upwp, parity="even")
        vptp = hemispheric_fold(vptp, parity="odd")
        vpthp = hemispheric_fold(vpthp, parity="odd")
    else:
        u_zm = u_zm_raw
        v_zm = v_zm_raw
        theta_zm = theta_zm_raw
        t_zm = t_zm_raw
        w_zm = w_zm_raw
        rho_zm = rho_zm_raw

    print("  Computing TEM derivatives and EP fluxes")
    z_values = np.asarray(z.values, dtype=np.float64)
    phi_values = np.deg2rad(np.asarray(latitude.values, dtype=np.float64))
    phi = xr.DataArray(phi_values, dims=("latitude",), coords={"latitude": latitude})

    u_dz = differentiate_wrt(u_zm, "z", z_values)
    theta_dz = differentiate_wrt(theta_zm, "z", z_values)
    u_dphi = differentiate_wrt(u_zm, "latitude", phi_values)

    h_flux = vpthp / theta_dz
    h_flux.name = "h_flux"

    m_dy = 2.0 * OMEGA * np.sin(phi) - (u_dphi - np.tan(phi) * u_zm) / EARTH_RADIUS
    m_eddy = u_dz * h_flux - upvp
    v_eddy = m_dy * h_flux - upwp

    cos_phi = np.cos(phi)
    fy = rho_zm * EARTH_RADIUS * cos_phi * m_eddy
    fz = rho_zm * EARTH_RADIUS * cos_phi * v_eddy

    dfy = (
        differentiate_wrt(m_eddy, "latitude", phi_values)
        - 2.0 * np.tan(phi) * m_eddy
    ) / EARTH_RADIUS
    dfz = differentiate_wrt(v_eddy, "z", z_values)
    epfd = dfy + dfz

    vres = v_zm - differentiate_wrt(h_flux, "z", z_values)
    wres = w_zm + (
        differentiate_wrt(h_flux, "latitude", phi_values)
        - np.tan(phi) * h_flux
    ) / EARTH_RADIUS

    out = xr.Dataset(
        {
            "t": t_zm,
            "upvp": upvp,
            "upwp": upwp,
            "vptp": vptp,
            "vpthp": vpthp,
            "Fy": fy,
            "Fz": fz,
            "EPFd": epfd,
            "EPFd_y": dfy,
            "EPFd_z": dfz,
            "vres": vres,
            "wres": wres,
        }
    )

    for name in out.data_vars:
        out[name] = out[name].transpose(target_time_dim, "z", "latitude")

    out["t"].attrs.update(long_name="Zonal mean temperature", units="K")
    out["upvp"].attrs.update(long_name="Meridional momentum flux", units="m2 s-2")
    out["upwp"].attrs.update(long_name="Vertical momentum flux", units="m2 s-2")
    out["vptp"].attrs.update(long_name="Meridional temperature flux", units="K m s-1")
    out["vpthp"].attrs.update(long_name="Meridional potential temperature flux", units="K m s-1")
    out["Fy"].attrs.update(long_name="Meridional EP flux", units="kg s-2")
    out["Fz"].attrs.update(long_name="Vertical EP flux", units="kg s-2")
    out["EPFd"].attrs.update(long_name="Acceleration due to EP flux convergence", units="m s-2")
    out["EPFd_y"].attrs.update(
        long_name="Meridional component of acceleration due to EP flux convergence",
        units="m s-2",
    )
    out["EPFd_z"].attrs.update(
        long_name="Vertical component of acceleration due to EP flux convergence",
        units="m s-2",
    )
    out["vres"].attrs.update(long_name="Meridional residual velocity", units="m s-1")
    out["wres"].attrs.update(long_name="Vertical residual velocity", units="m s-1")

    out.attrs.update(
        description="Height-coordinate TEM diagnostics computed from latlon files",
        reference_script=REFERENCE_SCRIPT,
        hemispheric_fold="enabled" if hemispheric_fold_fields else "disabled",
    )
    return out


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
    }
    return {dim: chunk for dim, chunk in requested.items() if dim in ds.dims and chunk is not None}


def output_encoding(ds: xr.Dataset, dtype: str | None) -> dict[str, dict[str, object]]:
    encoding: dict[str, dict[str, object]] = {}
    for name in ds.data_vars:
        enc: dict[str, object] = {}
        if dtype:
            enc["dtype"] = dtype
        encoding[name] = enc
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
                var = nc.createVariable(name, dtype, dims)

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
            hemispheric_fold_fields=not args.no_hemispheric_fold,
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

    print(f"Processing {latlon_path.name} -> {output_path}")
    zm_coords, existing = load_zm_coords(zm_path)

    if args.dry_run:
        conflicts = sorted(name for name in TEM_VARIABLES if name in existing)
        if conflicts:
            print(f"  Existing TEM variables: {', '.join(conflicts)}")
        return

    latlon_ds = open_dataset(latlon_path, chunks={})
    try:
        latlon_ds = latlon_ds.chunk(chunks_for(latlon_ds, args))
        if output_path == zm_path:
            if args.time_block_size is not None:
                append_tem_to_zm_by_time_blocks(latlon_ds, zm_path, args, zm_coords, existing)
            else:
                tem_ds = compute_tem_dataset(
                    latlon_ds,
                    zm_coords,
                    hemispheric_fold_fields=not args.no_hemispheric_fold,
                )
                tem_ds = variables_to_write(tem_ds, existing, args.overwrite)
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
        else:
            tem_ds = compute_tem_dataset(
                latlon_ds,
                zm_coords,
                hemispheric_fold_fields=not args.no_hemispheric_fold,
            )
            tem_ds = variables_to_write(tem_ds, existing, args.overwrite)
            encoding = output_encoding(tem_ds, args.output_dtype)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"{output_path} already exists; rerun with --overwrite")
            with open_dataset(zm_path, chunks={}) as zds:
                merged = xr.merge([zds, tem_ds], compat="override")
                to_netcdf_with_scheduler(
                    merged,
                    args.scheduler,
                    output_path,
                    mode="w",
                    engine="netcdf4",
                    encoding=encoding,
                    compute=True,
                )
    finally:
        latlon_ds.close()


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


def output_path_for(zm_path: Path, data_dir: Path, output_dir: str | Path | None) -> Path | None:
    if output_dir is None:
        return None
    output_dir = Path(output_dir)
    try:
        rel = zm_path.relative_to(data_dir)
    except ValueError:
        rel = Path(zm_path.name)
    return output_dir / rel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute TEM diagnostics from MPAS latlon files and append the "
            "resulting zonal-mean variables to matching zm_latlon files."
        )
    )
    parser.add_argument("data_dir", help="Directory containing paired latlon and zm_latlon files")
    parser.add_argument("--latlon-pattern", default="latlon*.nc", help="Glob for source latlon files")
    parser.add_argument(
        "--zm-pattern",
        default=None,
        help="Optional glob used only when the matching zm_latlon name cannot be inferred",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write augmented zm files to this directory instead of appending in place",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing TEM variables/output files")
    parser.add_argument("--dry-run", action="store_true", help="List matched files without computing")
    parser.add_argument(
        "--no-hemispheric-fold",
        action="store_true",
        help="Keep raw hemispheres instead of applying parity-aware paired-hemisphere averages",
    )
    parser.add_argument("--time-chunk", type=int, default=1, help="Dask chunk length for Time")
    parser.add_argument("--latitude-chunk", type=int, default=30, help="Dask chunk length for latitude")
    parser.add_argument("--longitude-chunk", type=int, default=180, help="Dask chunk length for longitude")
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
        default="float32",
        help="NetCDF dtype for appended TEM variables; calculations remain xarray/dask expressions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_file_write:
        args.time_block_size = None
    data_dir = Path(args.data_dir)
    pairs = find_pairs(data_dir, args.latlon_pattern, args.zm_pattern)
    print(f"Found {len(pairs)} latlon/zm pair(s)")
    for latlon_path, zm_path in pairs:
        out_path = output_path_for(zm_path, data_dir, args.output_dir)
        append_tem_to_zm(latlon_path, zm_path, args, output_path=out_path)
    print("Done.")


if __name__ == "__main__":
    main()
