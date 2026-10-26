"""Native MPAS fields shared by the TEM producer and spectral decomposition."""

import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline

from diagnostics.tem_core import finite_or_nan

NATIVE_FIELDS = ("uReconstructZonal", "uReconstructMeridional", "w", "theta", "rho", "pressure")
WATER_SPECIES = ("qv", "qc", "qi", "qr", "qs", "qg")
PREPARED_FIELDS = ("u", "v", "w", "theta", "density", "pressure")


def open_native_dataset(path):
    """Open native fields without caching complete uncompressed HDF5 chunks."""
    from netCDF4 import Dataset

    handle = Dataset(path, mode="r")
    try:
        for name in (*NATIVE_FIELDS, *WATER_SPECIES):
            if name not in handle.variables:
                continue
            variable = handle[name]
            filters = variable.filters() or {}
            filtered = any(value for key, value in filters.items() if key != "complevel")
            if variable.chunking() != "contiguous" and not filtered:
                variable.set_var_chunk_cache(size=0)
        return xr.open_dataset(xr.backends.NetCDF4DataStore(handle), chunks=None, cache=False)
    except Exception:
        if handle.isopen():
            handle.close()
        raise


def iter_native_frames(source, z, latitude=None, *, scheduler="threads", output_height=None, requested_fields=None):
    """Read each native time sample once using stored horizontal chunk boundaries.

    Complete interface columns preserve the global vertical-velocity spline.
    Optional output heights limit model-level reads without truncating that spline.
    Spatial selection follows field preparation, so storage chunks stay intact.
    Only one prepared time sample remains in memory at each yield.
    """
    from diagnostics.cache_support import materialize

    stored = source.uReconstructZonal.encoding.get("chunksizes")
    target_height = z
    if output_height is not None:
        indices = z.get_index("z").get_indexer(np.asarray(output_height))
        if not indices.size or np.any(indices < 0) or np.any(np.diff(indices) != 1):
            raise ValueError("Native output heights require a contiguous subset of the height grid")
        target_height = z.isel(z=indices)
        source = source.isel({vertical_dim(source.uReconstructZonal): slice(int(indices[0]), int(indices[-1])+1)})
    chunks = {dimension: -1 for dimension in source.dims}
    chunks[time_dim(source)] = 1
    for dimension in ("latitude", "longitude"):
        chunks[dimension] = (stored[source.uReconstructZonal.dims.index(dimension)]
                             if stored else source.sizes[dimension])
    # Read the producer's reference interface column before spatial selection.
    if "zgrid" in source and (requested_fields is None or "w" in requested_fields):
        source = source.assign(zgrid=xr.DataArray(zfull_from_latlon(source), dims="nVertLevelsP1"))
    for index in range(source.sizes[time_dim(source)]):
        native = source.isel({time_dim(source): slice(index, index + 1)}).chunk(chunks)
        fields = native_tem_fields(native, target_height, requested_fields=requested_fields)
        if requested_fields is None:
            fields["rho_dry_zm"] = model_level_field(native, "rho", target_height).mean("longitude", skipna=False)
        if latitude is not None:
            fields = fields.sel(latitude=latitude)
        yield materialize(fields, scheduler=scheduler, num_workers=2)


def vertical_dim(field):
    for dimension in ("z", "nVertLevels", "nVertLevelsP1"):
        if dimension in field.dims:
            return dimension
    raise ValueError(f"No native vertical dimension in {field.name!r}")


def time_dim(obj):
    for dimension in ("Time", "time"):
        if dimension in obj.dims:
            return dimension
    raise ValueError("Native TEM fields require a time dimension")


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
    zgrid = ds["zgrid"]
    if "latitude" in zgrid.dims:
        zgrid = zgrid.sel(latitude=0.0, method="nearest")
    if "longitude" in zgrid.dims:
        zgrid = zgrid.sel(longitude=0.0, method="nearest")
    for dimension in ("Time", "time"):
        if dimension in zgrid.dims:
            zgrid = zgrid.isel({dimension: 0})
    zgrid = finite_or_nan(zgrid.squeeze())
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
    if lev_dim == "nVertLevelsP1" or w.sizes[lev_dim] == z_coord.size + 1:
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


def native_tem_fields(dataset, z, *, requested_fields=None):
    """Use the producer's interpolation and total moist-air density convention."""
    requested = PREPARED_FIELDS if requested_fields is None else tuple(requested_fields)
    if not requested or not set(requested).issubset(PREPARED_FIELDS):
        raise ValueError("Native field selection must use known prepared fields")
    names = {"u": "uReconstructZonal", "v": "uReconstructMeridional", "theta": "theta", "pressure": "pressure"}
    result = xr.Dataset({name: model_level_field(dataset, source, z)
                         for name, source in names.items() if name in requested})
    if "w" in requested:
        result["w"] = midlevel_vertical_velocity(dataset, z)
    present = [name for name in WATER_SPECIES if name in dataset]
    if present and len(present) != len(WATER_SPECIES):
        raise ValueError("Total mass density requires qv, qc, qi, qr, qs, and qg")
    if "density" in requested:
        density = model_level_field(dataset, "rho", z)
        if present:
            density = density * (1 + sum(model_level_field(dataset, name, z) for name in WATER_SPECIES))
        result["density"] = density
    result.attrs["density_convention"] = ("dry density times (1 + all six water mixing ratios)"
                                          if present else "supplied dry density; no water species")
    return result
