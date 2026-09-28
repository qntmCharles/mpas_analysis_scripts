import glob
import json
import os
from datetime import datetime

import dask
import dask.array as dsa
import numpy as np
import pandas as pd
from matplotlib import colors, pyplot as plt, ticker
from matplotlib.patches import Patch
import scipy.interpolate as intp
import xarray as xr

from diagnostics.ttl import TTL_HEIGHT_BOUNDS_M
from diagnostics.analysis_config import (
    ANALYSIS_END,
    ANALYSIS_START,
    DEFAULT_FIGURE_ROOT,
    DEFAULT_FIGURE_VERSION,
    DIAGNOSTIC_FIGURE_ROOT_ENV,
    DIAGNOSTIC_FIGURE_VERSION_ENV,
    DIAGNOSTIC_OUTPUT_ROOT_ENV,
    LATLON_PATTERN,
    QUICK_TEST_LAT_BOUNDS,
    QUICK_TEST_LON_BOUNDS,
    SIMULATION_COLOURS,
    SIMULATION_ROOT,
    ZONAL_MEAN_PATTERN,
    cache_directory,
    profile_cache_directory,
    simulation_colour,
)
from diagnostics.cache_support import (
    cache_manifest,
    cache_manifest_matches,
    jsonable_value,
    load_netcdf,
    materialize,
    metadata_reader,
    metadata_writer,
    netcdf_safe_attribute,
    netcdf_safe_attrs,
    netcdf_safe_dataset,
    read_json_file,
    source_file_signature,
    write_csv_atomic,
    write_json_file,
    write_netcdf_atomic,
)


DEFAULT_SIMULATION_PARENT_DIR = str(SIMULATION_ROOT)
DEFAULT_DIAGNOSTIC_OUTPUT_ROOT = str(DEFAULT_FIGURE_ROOT)
DEFAULT_DIAGNOSTIC_FIGURE_VERSION = DEFAULT_FIGURE_VERSION
SPECIFIC_HUMIDITY_CMAP = "BrBG"


def simulation_parent_dir():
    """Return the parent directory containing the simulation subdirectories."""
    return os.path.abspath(DEFAULT_SIMULATION_PARENT_DIR)


def diagnostic_cache_dir(family, *parts):
    """Return a path from the central diagnostic cache layout."""

    return os.fspath(cache_directory(family, *parts))


def diagnostic_profile_cache_dir(family, *parts):
    """Return a cache path for products that combine simulations."""

    return os.fspath(profile_cache_directory(family, *parts))


def diagnostic_output_dir(default_save_loc, root_dir=DEFAULT_DIAGNOSTIC_OUTPUT_ROOT):
    """Return a diagnostic output directory, optionally remapped under an override root."""
    override_root = os.environ.get(DIAGNOSTIC_OUTPUT_ROOT_ENV) or os.environ.get(DIAGNOSTIC_FIGURE_ROOT_ENV)
    if not override_root:
        return default_save_loc

    default_path = os.path.abspath(os.path.expanduser(str(default_save_loc)))
    default_root = os.path.abspath(os.path.expanduser(str(root_dir)))
    override_root = os.path.abspath(os.path.expanduser(override_root))

    override_rel = os.path.relpath(default_path, override_root)
    if override_rel == "." or (
        override_rel != os.pardir and not override_rel.startswith(os.pardir + os.sep)
    ):
        return default_path.rstrip(os.sep) + os.sep

    rel_path = os.path.relpath(default_path, default_root)
    if rel_path == os.pardir or rel_path.startswith(os.pardir + os.sep):
        rel_path = os.path.basename(default_path.rstrip(os.sep))

    return os.path.join(override_root, rel_path) + os.sep


def figure_release_directory(target_dir, version):
    """Place one runner release above its diagnostic output directories."""

    configured_root = (
        os.environ.get(DIAGNOSTIC_OUTPUT_ROOT_ENV)
        or os.environ.get(DIAGNOSTIC_FIGURE_ROOT_ENV)
        or DEFAULT_DIAGNOSTIC_OUTPUT_ROOT
    )
    configured_root = os.path.abspath(os.path.expanduser(configured_root))
    target_dir = os.path.abspath(os.path.expanduser(str(target_dir)))
    relative = os.path.relpath(target_dir, configured_root)
    parts = relative.split(os.sep)
    if parts[0] == "final" and os.pardir not in parts:
        diagnostic_parts = parts[1:]
        if diagnostic_parts[:2] == ["organisation_lpt", "publication"]:
            diagnostic_parts = [
                "organisation_lpt",
                *diagnostic_parts[2:],
            ]
        return os.path.join(
            configured_root, "final", version, *diagnostic_parts
        )
    return os.path.join(target_dir, version)


ISOBARIC_DIM_LEVEL_VAR = {
    "nIsoLevelsT": "t_iso_levels",
    "nIsoLevelsU": "u_iso_levels",
    "nIsoLevelsZ": "z_iso_levels",
}


def standardize_cldfrac_vertical_coordinate(ds):
    """Give cldfrac a representative geometric-height ``z`` coordinate."""
    if "cldfrac" not in ds.data_vars:
        return ds

    cldfrac = ds["cldfrac"]
    if "nVertLevels" in cldfrac.dims:
        cldfrac = cldfrac.rename({"nVertLevels": "z"})
    elif "nVertLevelsP1" in cldfrac.dims:
        raise ValueError("cldfrac appears to be on interface levels, not mid-levels")
    elif "z" not in cldfrac.dims:
        raise ValueError(f"Could not find vertical dimension in cldfrac dims {cldfrac.dims}")

    if "zgrid" not in ds.data_vars:
        if "z" in cldfrac.coords:
            return ds.assign(cldfrac=cldfrac)
        raise ValueError(
            "Cannot assign geometric z coordinate to cldfrac: zgrid is missing"
        )

    zgrid_mean = ds.zgrid
    for dimension in ("Time", "longitude"):
        if dimension in zgrid_mean.dims:
            zgrid_mean = zgrid_mean.mean(dimension, skipna=True)
    if "latitude" in zgrid_mean.dims:
        zgrid_mean = lat_mean(zgrid_mean)
    remaining = [
        dimension
        for dimension in zgrid_mean.dims
        if dimension != "nVertLevelsP1"
    ]
    if remaining:
        zgrid_mean = zgrid_mean.mean(remaining, skipna=True)
    zmid = 0.5 * (
        zgrid_mean.isel(nVertLevelsP1=slice(0, -1)).values
        + zgrid_mean.isel(nVertLevelsP1=slice(1, None)).values
    )
    if cldfrac.sizes["z"] != zmid.size:
        raise ValueError(
            f"cldfrac has {cldfrac.sizes['z']} vertical levels, "
            f"but zgrid implies {zmid.size} mid-levels"
        )
    cldfrac = cldfrac.assign_coords(z=("z", zmid))
    cldfrac["z"].attrs["units"] = "m"
    cldfrac["z"].attrs["long_name"] = (
        "time-horizontal-mean geometric height of layer midpoints"
    )
    return ds.assign(cldfrac=cldfrac)


def load_latlon_dataset(
    data_dir,
    latlon_pattern,
    chunks,
    direct_chunks=False,
    start=ANALYSIS_START,
    end=ANALYSIS_END,
):
    latlon_files = sorted(glob.glob(os.path.join(data_dir, latlon_pattern)))
    if not latlon_files:
        raise FileNotFoundError(f"No files match {latlon_pattern!r} in {data_dir}")
    print(data_dir, latlon_pattern)
    ds = xr.open_mfdataset(
        latlon_files,
        combine="by_coords",
        chunks=chunks if direct_chunks else {},
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    if not np.issubdtype(ds.Time.dtype, np.datetime64):
        raise TypeError(f"Expected decoded datetime Time coordinate, got {ds.Time.dtype}")
    expected_time = pd.date_range(
        start=start,
        end=end,
        freq="6h",
    )
    ds = ds.sel(Time=expected_time)
    if not ds.Time.to_index().equals(expected_time):
        raise ValueError("Input does not have the expected complete 6-hourly Time axis")
    return ds if direct_chunks else ds.chunk(chunks)


def get_time_values(ds):
    return ds.Time.values


def finite_or_nan(da, max_abs=1e30):
    """Mask NaNs, infinities, and MPAS-style large fill values."""
    return da.where(np.isfinite(da) & (np.abs(da) < max_abs))


def safe_nanpercentile(values, q, zero_value=1.0):
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("Cannot calculate a percentile from non-finite data")
    out = np.nanpercentile(vals, q)
    if not np.isfinite(out):
        raise ValueError("Calculated percentile is non-finite")
    if out == 0:
        return zero_value
    return out


def binned_mean_1d(x, y, bins, min_count=10):
    """Bin y by x and return bin centres, mean, p25, p75, and counts."""
    xv = np.asarray(x).ravel()
    yv = np.asarray(y).ravel()
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[ok]
    yv = yv[ok]
    idx = np.digitize(xv, bins) - 1
    xc = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(xc.size, np.nan)
    p25 = np.full(xc.size, np.nan)
    p75 = np.full(xc.size, np.nan)
    count = np.zeros(xc.size, dtype=int)
    for k in range(xc.size):
        vals = yv[idx == k]
        count[k] = vals.size
        if vals.size >= min_count:
            mean[k] = np.nanmean(vals)
            p25[k] = np.nanpercentile(vals, 25)
            p75[k] = np.nanpercentile(vals, 75)
    return xc, mean, p25, p75, count


def total_precip_accum(ds):
    """Accumulated total precipitation in mm, including MPAS bucket counters."""
    return (
        ds["rainnc"] + ds["i_rainnc"] * 100.0
        + ds["rainc"] + ds["i_rainc"] * 100.0
    )


def precip_rate_from_accum(accum):
    """Convert accumulated precipitation to a precipitation rate in mm day-1."""
    dt_days = accum.Time.diff("Time") / np.timedelta64(1, "D")
    return accum.diff("Time") / dt_days


def vertical_dim(da):
    for dim in ("z", "nVertLevels", "nVertLevelsP1"):
        if dim in da.dims:
            return dim
    raise ValueError(f"No known model-level vertical dimension found in {da.dims!r}")


def midlevel_height_from_latlon(ds, latitude_sel=None, mean_latitude=False):
    zgrid = ds.zgrid
    if latitude_sel is not None:
        zgrid = zgrid.sel(latitude=latitude_sel)
    zgrid = zgrid.mean(dim="longitude")
    if mean_latitude:
        zgrid = lat_mean(zgrid)

    zmid_values = 0.5 * (
        zgrid.isel(nVertLevelsP1=slice(0, -1)).values
        + zgrid.isel(nVertLevelsP1=slice(1, None)).values
    )
    if mean_latitude:
        return xr.DataArray(zmid_values, dims=("z",), coords={"z": np.arange(zmid_values.shape[0])}, name="zmid")
    return xr.DataArray(
        zmid_values,
        dims=("z", "latitude"),
        coords={"z": np.arange(zmid_values.shape[0]), "latitude": zgrid.latitude},
        name="zmid",
    )


def align_vertical_velocity_to_reference(w, reference):
    """Align standard vertical velocity to a model-level reference field."""
    w_dim = vertical_dim(w)
    target_dim = vertical_dim(reference)

    source_levels = w.sizes[w_dim]
    target_levels = reference.sizes[target_dim]
    if source_levels == target_levels + 1:
        lower = w.isel({w_dim: slice(0, -1)}).rename(
            {w_dim: target_dim}
        )
        upper = w.isel({w_dim: slice(1, None)}).rename(
            {w_dim: target_dim}
        )
        target_coordinate = reference[target_dim]
        lower = lower.assign_coords({target_dim: target_coordinate})
        upper = upper.assign_coords({target_dim: target_coordinate})
        w = 0.5 * (lower + upper)
        w_dim = target_dim
    elif source_levels != target_levels:
        raise ValueError(
            "Vertical velocity must use the same native mid-level grid or "
            "bounding interfaces; got "
            f"{source_levels} velocity levels and {target_levels} targets"
        )

    if w_dim != target_dim:
        w = w.rename({w_dim: target_dim})
    if target_dim in reference.coords:
        w = w.assign_coords({target_dim: reference[target_dim]})
    return w


def native_layer_interfaces(ds, template):
    """Return MPAS native-layer interfaces aligned to a mid-level field."""

    if "zgrid" not in ds.data_vars:
        raise ValueError("zgrid is required for native geometric height")
    level_dim = vertical_dim(template)
    if level_dim == "nVertLevelsP1":
        raise ValueError("template must be defined at native layer midpoints")
    if ds.zgrid.sizes.get("nVertLevelsP1") != template.sizes[level_dim] + 1:
        raise ValueError(
            "zgrid interface count is inconsistent with the native mid-level "
            f"field: {ds.zgrid.sizes.get('nVertLevelsP1')} interfaces for "
            f"{template.sizes[level_dim]} levels"
        )

    target_coord = template[level_dim]
    first = ds.zgrid.isel(nVertLevelsP1=slice(0, -1)).rename(
        {"nVertLevelsP1": level_dim}
    ).assign_coords({level_dim: target_coord})
    second = ds.zgrid.isel(nVertLevelsP1=slice(1, None)).rename(
        {"nVertLevelsP1": level_dim}
    ).assign_coords({level_dim: target_coord})
    lower = xr.where(first <= second, first, second)
    upper = xr.where(first <= second, second, first)
    return lower, upper


def native_midlevel_height(ds, template):
    """Return full MPAS native mid-level height aligned to ``template``."""

    lower, upper = native_layer_interfaces(ds, template)
    height = 0.5 * (lower + upper)
    height.name = "native_midlevel_height"
    height.attrs.update(
        units="m",
        long_name="geometric height of MPAS native layer midpoint",
        source="midpoint of zgrid interfaces",
    )
    return height


def pressure_to_pa(pressure):
    """Return pressure in Pa without corrupting valid low-pressure levels."""

    units = str(pressure.attrs.get("units", "")).strip().lower()
    normalised = units.replace(" ", "")
    if normalised in {"hpa", "hectopascal", "hectopascals", "mb", "mbar"}:
        out = pressure * 100.0
    elif normalised in {"pa", "pascal", "pascals"}:
        out = pressure
    else:
        raise ValueError(
            "Pressure must declare Pa or hPa units; got "
            f"{units!r}"
        )
    out.attrs.update(getattr(pressure, "attrs", {}))
    out.attrs["units"] = "Pa"
    return out


MPAS_MOISTURE_SCALARS = ("qv", "qc", "qr", "qi", "qs", "qg")


def mpas_moist_air_density(dataset, total_moisture=None):
    """Return MPAS moist density from dry density and all moisture scalars."""

    required = ("rho", *MPAS_MOISTURE_SCALARS)
    missing = [name for name in required if name not in dataset.data_vars]
    if missing:
        raise ValueError(f"MPAS moist density requires fields {missing}")
    if total_moisture is None:
        total_moisture = sum(
            (finite_or_nan(dataset[name]) for name in MPAS_MOISTURE_SCALARS),
            start=xr.zeros_like(finite_or_nan(dataset.rho)),
        )
    density = finite_or_nan(dataset.rho) * (1.0 + total_moisture)
    density.name = "moist_air_density"
    density.attrs.update(
        units="kg m-3",
        dry_density_source="rho",
        formulation="rho_m = rho_d * (1 + qv + qc + qr + qi + qs + qg)",
    )
    return density


def vertical_velocity_from_zm(zds):
    """Return standard zonal-mean vertical velocity from a zm dataset."""
    missing = [name for name in ("w", "pressure") if name not in zds.data_vars]
    if missing:
        raise ValueError(
            "Zonal-mean vertical velocity requires the standard MPAS fields "
            f"'w' and 'pressure'; missing {missing}"
        )

    w = finite_or_nan(zds.w)
    w = align_vertical_velocity_to_reference(
        w, finite_or_nan(zds.pressure)
    )

    w.name = "w"
    w.attrs.update(units="m s-1", long_name="standard vertical velocity")
    return w


def _interp_column_to_pressure(pressure_pa, values, target_hpa):
    pressure_hpa = np.asarray(pressure_pa, dtype=float) / 100.0
    vals = np.asarray(values, dtype=float)
    ok = np.isfinite(pressure_hpa) & np.isfinite(vals)
    if np.count_nonzero(ok) < 2:
        return np.nan

    p_ok = pressure_hpa[ok]
    vals_ok = vals[ok]
    order = np.argsort(p_ok)
    p_sorted = p_ok[order]
    vals_sorted = vals_ok[order]
    p_unique, unique_idx = np.unique(p_sorted, return_index=True)
    if p_unique.size < 2:
        return np.nan
    return np.interp(float(target_hpa), p_unique, vals_sorted[unique_idx], left=np.nan, right=np.nan)


def vertical_velocity_at_pressure(ds, pressure_hpa):
    """Interpolate standard vertical velocity to one pressure level in hPa."""
    missing = [name for name in ("w", "pressure") if name not in ds.data_vars]
    if missing:
        raise ValueError(f"Missing variables for pressure-level w: {missing}")

    pressure = finite_or_nan(ds.pressure)
    w = align_vertical_velocity_to_reference(finite_or_nan(ds.w), pressure)
    pressure, w = xr.align(pressure, w, join="inner")
    z_dim = vertical_dim(pressure)

    out = xr.apply_ufunc(
        _interp_column_to_pressure,
        pressure,
        w,
        input_core_dims=[[z_dim], [z_dim]],
        output_core_dims=[[]],
        kwargs={"target_hpa": float(pressure_hpa)},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    out.name = f"w{float(pressure_hpa):g}".replace(".", "p")
    out.attrs.update(
        units="m s-1",
        long_name="standard vertical velocity interpolated to pressure level",
        selected_pressure_hpa=float(pressure_hpa),
    )
    return out


def isobaric_vertical_velocity_from_zm(zds, pressure_levels_hpa=None, hemispheric_parity="even"):
    """Interpolate time-mean zonal-mean w to the dataset's pressure levels."""
    if "pressure" not in zds.data_vars:
        raise ValueError("pressure is required to interpolate zonal-mean w to pressure levels")

    if pressure_levels_hpa is None:
        if "pres" not in zds.coords:
            raise ValueError("No pressure levels supplied and zds has no pres coordinate")
        pressure_levels_hpa = zds.pres.values

    w = vertical_velocity_from_zm(zds)
    pressure = finite_or_nan(zds.pressure)
    if "Time" in w.dims:
        w = w.mean(dim="Time", skipna=True)
    if "Time" in pressure.dims:
        pressure = pressure.mean(dim="Time", skipna=True)

    w = model_lat_field(w)
    pressure_hpa = model_lat_field(pressure) / 100.0
    target = np.asarray(pressure_levels_hpa, dtype=float)
    out = np.full((target.size, w.sizes["latitude"]), np.nan, dtype=float)

    for j in range(w.sizes["latitude"]):
        p_col = np.asarray(pressure_hpa.isel(latitude=j).values, dtype=float)
        w_col = np.asarray(w.isel(latitude=j).values, dtype=float)
        ok = np.isfinite(p_col) & np.isfinite(w_col)
        if np.count_nonzero(ok) < 2:
            continue
        p_ok = p_col[ok]
        w_ok = w_col[ok]
        order = np.argsort(p_ok)
        p_sorted = p_ok[order]
        w_sorted = w_ok[order]
        p_unique, unique_idx = np.unique(p_sorted, return_index=True)
        if p_unique.size < 2:
            continue
        out[:, j] = np.interp(target, p_unique, w_sorted[unique_idx], left=np.nan, right=np.nan)

    da = xr.DataArray(
        out,
        dims=("pres", "latitude"),
        coords={"pres": target, "latitude": w.latitude},
        name="w",
        attrs={"units": "m s-1", "long_name": "standard vertical velocity interpolated to pressure levels"},
    )
    return hemispheric_fold(da, parity=hemispheric_parity)


def model_lat_field(da):
    lev_dim = vertical_dim(da)
    if lev_dim != "z":
        da = da.rename({lev_dim: "z"})
    return da.transpose("z", "latitude")


def lat_height_mesh(field, zmid, lat):
    field = model_lat_field(field)
    zmid = model_lat_field(zmid) if "latitude" in zmid.dims else zmid.rename({vertical_dim(zmid): "z"})
    lat2d = np.broadcast_to(lat[None, :], field.shape)
    zvals = zmid.values
    if zvals.ndim == 1:
        zvals = np.broadcast_to(zvals[:, None], field.shape)
    return field, lat2d, zvals


def N2_latheight(zds, g=9.80665):
    """Time-mean static stability on model height levels as (z, latitude)."""
    theta_source = (
        zds.theta.mean(dim="Time")
        if "Time" in zds.theta.dims
        else zds.theta
    )
    theta = model_lat_field(theta_source)
    z = zds.z.values
    dtheta_dz = np.gradient(theta.values, z, axis=0)
    n2 = xr.DataArray(
        g / theta.values * dtheta_dz,
        dims=theta.dims,
        coords=theta.coords,
        name="N2",
        attrs={"units": "s-2"},
    )
    return hemispheric_fold(n2, parity="even")


def pressure_values_from_dataset(ds):
    """Return the fixed 22-level pressure coordinate, checking exact agreement."""
    if "pres" in ds.dims:
        return np.asarray(ds.pres.values, dtype=float)

    axes = []
    for dim, level_var in ISOBARIC_DIM_LEVEL_VAR.items():
        if dim in ds.dims:
            values = np.asarray(ds[level_var].values, dtype=float)
            if values.shape != (ds.sizes[dim],):
                raise ValueError(f"{level_var} must be one-dimensional on {dim}")
            axes.append(values)

    if not axes:
        return None

    if any(not np.array_equal(axes[0], values) for values in axes[1:]):
        raise ValueError("In-file isobaric pressure axes disagree")
    return axes[0]


def standardize_latlon_dataset(ds, needed_vars=None):
    if needed_vars is not None:
        missing = [name for name in needed_vars if name not in ds.data_vars]
        if missing:
            raise KeyError(f"Required variables are missing: {missing}")
        source = ds
        selected = ds[list(needed_vars)]
        level_vars = [
            level_var
            for dim, level_var in ISOBARIC_DIM_LEVEL_VAR.items()
            if dim in selected.dims and level_var not in needed_vars
        ]
        ds = source[list(needed_vars) + level_vars]
    file_pres_values = pressure_values_from_dataset(ds)
    rename_map = {}
    for dim in ISOBARIC_DIM_LEVEL_VAR:
        if dim in ds.dims:
            rename_map[dim] = "pres"
    if rename_map:
        ds = ds.rename(rename_map)
    if "pres" in ds.dims:
        if file_pres_values is None:
            raise ValueError("Dataset has isobaric dimensions but no trustworthy in-file pressure axis")
        if ds.sizes["pres"] != len(file_pres_values):
            raise ValueError(f"Unexpected pressure axis length: {ds.sizes['pres']} (expected {len(file_pres_values)})")
        ds = ds.assign_coords(pres=("pres", file_pres_values))
        ds["pres"].attrs.update(units="hPa", positive="down", axis="Z")
    if "rh_isobaric" in ds.data_vars:
        relative_humidity = ds.rh_isobaric.copy(deep=False)
        archived_units = relative_humidity.attrs.get("units")
        if archived_units and archived_units not in {"%", "percent"}:
            relative_humidity.attrs["archived_units"] = archived_units
        relative_humidity.attrs["units"] = "%"
        relative_humidity.attrs["quantity"] = "relative humidity"
        ds = ds.assign(rh_isobaric=relative_humidity)
    return standardize_cldfrac_vertical_coordinate(ds)


def lat_mean(da, lat_dim="latitude"):
    return da.weighted(np.cos(np.deg2rad(da[lat_dim]))).mean(
        dim=lat_dim, skipna=True
    )


def cosine_weighted_latitude_coarsen(
    da,
    window,
    lat_dim="latitude",
    boundary="trim",
):
    """Coarsen latitude with cosine weights and a finite-data denominator."""
    coarse_dim = f"{lat_dim}_coarse"
    window_dim = f"{lat_dim}_window"
    constructed = da.coarsen(
        {lat_dim: int(window)}, boundary=boundary
    ).construct(**{lat_dim: (coarse_dim, window_dim)})
    weights = np.cos(np.deg2rad(constructed[lat_dim])).clip(min=0.0)
    coarse_latitude = constructed[lat_dim].mean(window_dim)
    result = constructed.weighted(weights).mean(window_dim, skipna=True)
    return result.rename({coarse_dim: lat_dim}).assign_coords(
        {lat_dim: coarse_latitude.values}
    )


def cosine_weighted_power_input(da, lat_dim="latitude"):
    """Scale paired latitudes so a spectral latitude sum is area-weighted."""
    latitude = da[lat_dim]
    weights = np.cos(np.deg2rad(latitude)).clip(min=0.0)
    north_total = weights.where(latitude > 0.0, drop=True).sum()
    south_total = weights.where(latitude < 0.0, drop=True).sum()
    if not np.isclose(
        float(north_total), float(south_total), rtol=1.0e-7, atol=1.0e-12
    ):
        raise ValueError(
            "Cosine-weighted symmetric spectra require paired hemispheres"
        )
    normalised = weights.where(latitude != 0.0, 0.0) / north_total
    return da * np.sqrt(normalised)


def latitude_is_hemispherically_symmetric(latitude, rtol=1e-7, atol=1e-8):
    """Return True when a latitude coordinate can be paired across the equator."""
    values = np.asarray(latitude.values if isinstance(latitude, xr.DataArray) else latitude, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        return False
    return np.allclose(values, -values[::-1], rtol=rtol, atol=atol)


def hemispheric_fold(da, lat_dim="latitude", parity="even"):
    """
    Average paired hemispheres on the original latitude grid.

    ``parity="even"`` uses 0.5 * (NH + SH), appropriate for scalars, zonal wind,
    vertical velocity, and energies. ``parity="odd"`` flips the reflected sign,
    appropriate for northward vectors and meridional fluxes.
    """
    if parity is None:
        return da
    if da is None:
        raise ValueError("Cannot hemispherically fold a missing field")
    if lat_dim not in da.dims:
        raise ValueError(
            f"Cannot hemispherically fold {getattr(da, 'name', None)!r}: "
            f"dimension {lat_dim!r} is absent from {da.dims!r}"
        )
    if parity not in ("even", "odd"):
        raise ValueError(f"Unknown hemispheric parity {parity!r}; expected 'even' or 'odd'")
    if not latitude_is_hemispherically_symmetric(da[lat_dim]):
        raise ValueError(
            f"Cannot hemispherically fold {getattr(da, 'name', None)!r}: "
            f"{lat_dim!r} is not symmetric about the equator"
        )

    reflected = da.isel({lat_dim: slice(None, None, -1)})
    reflected = reflected.assign_coords({lat_dim: da[lat_dim].values})
    sign = 1.0 if parity == "even" else -1.0
    out = 0.5 * (da + sign * reflected)
    out.name = da.name
    out.attrs.update(getattr(da, "attrs", {}))
    out.attrs["hemispheric_fold"] = f"{parity} paired-hemisphere average"
    return out


def plot_Tu_latpres(a, lats, pres, Tbar, ubar):
    T_im = a.contourf(lats, pres, Tbar, levels=np.arange(190, 305, 5), extend="both", cmap="coolwarm")
    u_im_neg = a.contour(lats, pres, ubar, levels=np.arange(-20, 0, 5), colors="k", linestyles=":", linewidths=0.5)
    u_im_zero = a.contour(lats, pres, ubar, levels=[0], colors="k", linewidths=1)
    u_im_pos = a.contour(lats, pres, ubar, levels=np.arange(5, 55, 5), colors="k", linewidths=0.5)
    a.clabel(u_im_pos, u_im_pos.levels, fontsize=6)
    a.clabel(u_im_neg, u_im_neg.levels, fontsize=6)
    return T_im, u_im_neg, u_im_zero, u_im_pos


def plot_rhw_latpres(a, lats, pres, rhbar, vbar, omega_hpa_s):
    """Plot RH and circulation using archived pressure velocity."""

    rh_im = a.contourf(lats, pres, rhbar, levels=np.arange(0, 105, 5), cmap="BrBG")
    xi = np.linspace(lats.min(), lats.max(), len(lats))
    yi = np.linspace(pres.min(), pres.max(), len(pres))
    X, Y = np.meshgrid(xi, yi)

    V_meridional = intp.interpn((lats, pres), vbar.T, (X, Y))
    dpdt_hpa_s = intp.interpn((lats, pres), omega_hpa_s.T, (X, Y))

    medcirc_im = a.streamplot(
        X,
        Y,
        V_meridional / 111111.0,
        dpdt_hpa_s,
        color="k",
        linewidth=1,
        density=1,
    )
    return rh_im, medcirc_im


def isobaric_lat_mean(da, hemispheric_parity=None):
    out = da.mean(dim="Time").transpose("pres", "latitude")
    return hemispheric_fold(out, parity=hemispheric_parity)


def set_latheight_labels(ax, xlabel=True, ylabel=True):
    ax.set_xlim(-90, 90)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    if xlabel:
        ax.set_xlabel("Latitude (degrees)")
    if ylabel:
        ax.set_ylabel("Height (km)")


def set_latpres_axis(ax, xlabel=True, ylabel=True, xlim=(-60, 60), ylim=(1000, 50)):
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel("Latitude")
    if ylabel:
        ax.set_ylabel("Pressure (hPa)")


def annotate_panel_label(ax, label):
    ax.annotate(
        label,
        (0.03, 0.95),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontweight="bold",
        bbox=dict(facecolor="w", alpha=0.7),
    )


def control_diff_label(sim_names, i, control=0):
    return sim_names[i] if i == control else f"{sim_names[i]} - {sim_names[control]}"


def _nice_step(value):
    """Round a positive interval up to a familiar 1, 2, 2.5, 5 sequence."""
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / 10.0 ** exponent
    nice_fraction = next(candidate for candidate in (1.0, 2.0, 2.5, 5.0, 10.0) if fraction <= candidate)
    return nice_fraction * 10.0 ** exponent


def nice_levels(vmin, vmax, n=17):
    """Return outward-rounded linear contour boundaries with readable increments."""
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if np.isclose(vmin, vmax, rtol=1.0e-12, atol=0.0):
        width = abs(vmin) * 0.1 if vmin else 1.0
        vmin, vmax = vmin - width, vmax + width
    step = _nice_step((vmax - vmin) / max(int(n) - 1, 1))
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    return np.arange(lo, hi + 0.5 * step, step)


def positive_levels(maximum, n=17):
    return nice_levels(0.0, maximum, n=n)


def symmetric_levels(max_abs, ndiv=11):
    """Readable symmetric levels with one interval centred on zero."""
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    ndiv = max(int(ndiv), 3)
    if ndiv % 2 == 0:
        ndiv += 1
    step = _nice_step(float(max_abs) / (ndiv / 2.0))
    return (np.arange(ndiv + 1) - ndiv / 2.0) * step


def format_colorbar_tick(value):
    if np.isclose(value, 0.0, atol=1e-12):
        return "0"
    return f"{value:.3g}"


def symmetric_colorbar_ticks(levels, nticks=5):
    max_abs = float(np.nanmax(np.abs(levels)))
    ticks = ticker.MaxNLocator(nbins=nticks, symmetric=True).tick_values(-max_abs, max_abs)
    ticks = ticks[np.abs(ticks) <= max_abs + 1.0e-12]
    return ticks if ticks.size >= 3 else np.array([-max_abs, 0.0, max_abs])


def add_symmetric_colorbar(fig, mappable, ax, levels, label, location="right", nticks=5, **kwargs):
    ticks = symmetric_colorbar_ticks(levels, nticks=nticks)
    cbar = fig.colorbar(mappable, ax=ax, location=location, label=label, ticks=ticks, **kwargs)
    cbar.set_ticklabels([format_colorbar_tick(tick) for tick in ticks])
    return cbar


PRECIP_RATE_COLORS = [
    [1.0, 1.0, 1.0],
    [205 / 255.0, 236 / 255.0, 251 / 255.0],
    [157 / 255.0, 218 / 255.0, 247 / 255.0],
    [72 / 255.0, 142 / 255.0, 202 / 255.0],
    [72 / 255.0, 158 / 255.0, 148 / 255.0],
    [73 / 255.0, 181 / 255.0, 70 / 255.0],
    [150 / 255.0, 203 / 255.0, 80 / 255.0],
    [250 / 255.0, 232 / 255.0, 92 / 255.0],
    [248 / 255.0, 184 / 255.0, 73 / 255.0],
    [245 / 255.0, 106 / 255.0, 41 / 255.0],
    [230 / 255.0, 74 / 255.0, 41 / 255.0],
    [211 / 255.0, 31 / 255.0, 40 / 255.0],
    [183 / 255.0, 27 / 255.0, 34 / 255.0],
    [146 / 255.0, 21 / 255.0, 25 / 255.0],
    [110 / 255.0, 15 / 255.0, 15 / 255.0],
]
PRECIP_RATE_LEVELS = np.linspace(0.01, 25.0, len(PRECIP_RATE_COLORS))

HIGH_CLOUD_CLASS_SPECS = (
    (1, "Deep convection", "tab:red"),
    (2, "Stratiform cloud", "tab:blue"),
    (3, "Non-precipitating cloud", "tab:green"),
    (4, "Unclassified cloud", "0.70"),
)


def precip_rate_cmap_norm():
    cmap = colors.LinearSegmentedColormap.from_list(
        "project_precip_rate", PRECIP_RATE_COLORS, N=len(PRECIP_RATE_COLORS)
    )
    norm = colors.BoundaryNorm(
        PRECIP_RATE_LEVELS, ncolors=len(PRECIP_RATE_COLORS) - 1
    )
    return cmap, norm


def blue_sky_white_cloud_cmap():
    """Return the shared sequential palette for cloud fraction or occurrence."""

    return colors.LinearSegmentedColormap.from_list(
        "blue_sky_white_cloud",
        ("#154c79", "#5fa8d3", "#d9edf7", "#ffffff"),
    )


def high_cloud_class_cmap_norm(alpha=0.82):
    rgba = [(0.0, 0.0, 0.0, 0.0)]
    for _code, _label, color in HIGH_CLOUD_CLASS_SPECS:
        rgba.append(colors.to_rgba(color, alpha=alpha))
    cmap = colors.ListedColormap(rgba, name="high_cloud_class_codes")
    norm = colors.BoundaryNorm(
        np.arange(-0.5, len(HIGH_CLOUD_CLASS_SPECS) + 1.5, 1.0),
        cmap.N,
    )
    return cmap, norm


def high_cloud_class_legend_handles():
    return [
        Patch(facecolor=color, label=label)
        for _code, label, color in HIGH_CLOUD_CLASS_SPECS
    ]


def difference_cmap(levels, cmap="bwr"):
    """Discrete diverging colormap with the zero-containing interval in white."""
    levels = np.asarray(levels, dtype=float)
    base = plt.get_cmap(cmap)
    n_intervals = len(levels) - 1
    rgba = base(np.linspace(0.0, 1.0, n_intervals))
    centre = np.flatnonzero((levels[:-1] < 0.0) & (levels[1:] > 0.0))
    if centre.size:
        rgba[centre] = (1.0, 1.0, 1.0, 1.0)
    out = colors.ListedColormap(rgba, name=f"{base.name}_white_zero")
    out.set_under(base(0.0))
    out.set_over(base(1.0))
    out.set_bad(base.get_bad())
    return out


# Minimum ratio between the troposphere and stratosphere colour scales before
# a difference panel is worth splitting into two colorbars.
SPLIT_SCALE_RATIO = 3.0


def tropopause_on_latitudes(tropopause_values, tropopause_latitudes, target_latitudes):
    """Interpolate a per-latitude tropopause curve onto a field's latitude grid."""
    return np.interp(
        np.asarray(target_latitudes, dtype=float),
        np.asarray(tropopause_latitudes, dtype=float),
        np.asarray(tropopause_values, dtype=float),
    )


def tropopause_region_masks(vertical2d, tropopause, coord="height"):
    """Boolean (troposphere, stratosphere) masks for a (nlev, nlat) vertical mesh.

    ``tropopause`` gives the tropopause of each latitude column in the same
    units as ``vertical2d``: height in m, or pressure in hPa with
    ``coord="pressure"``.
    """
    vert = np.asarray(vertical2d, dtype=float)
    tp = np.asarray(tropopause, dtype=float)
    if tp.ndim == 0:
        tp = np.full(vert.shape[1], float(tp))
    tp2d = np.broadcast_to(tp[None, :], vert.shape)
    tropo = vert >= tp2d if coord == "pressure" else vert <= tp2d
    return tropo, ~tropo


def region_symmetric_levels(
    value_mask_pairs, ndiv=10, zero_field_scale=1.0, percentile=None
):
    """Symmetric contour levels from the masked region of one or more fields."""
    pooled = []
    for values, mask in value_mask_pairs:
        vals = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
        pooled.append(vals[np.isfinite(vals)])
    pooled = np.concatenate(pooled) if pooled else np.array([], dtype=float)
    if pooled.size == 0:
        raise ValueError(
            "Cannot calculate regional contour levels from non-finite data"
        )
    if percentile is None:
        vmax = np.max(np.abs(pooled))
    else:
        vmax = np.percentile(np.abs(pooled), percentile)
    if not np.isfinite(vmax):
        raise ValueError("Calculated regional contour scale is non-finite")
    if vmax == 0.0:
        vmax = zero_field_scale
    return symmetric_levels(vmax, ndiv=ndiv)


def split_scales_warranted(tropo_levels, strat_levels, ratio=SPLIT_SCALE_RATIO):
    """True when troposphere and stratosphere need visibly different scales."""
    tropo_max = float(np.max(np.abs(tropo_levels)))
    strat_max = float(np.max(np.abs(strat_levels)))
    if not (np.isfinite(tropo_max) and np.isfinite(strat_max)):
        return False
    if min(tropo_max, strat_max) <= 0:
        return False
    return max(tropo_max / strat_max, strat_max / tropo_max) >= ratio


def contourf_split_difference(ax, x2d, y2d, values, tropo_mask,
                              tropo_levels, strat_levels, cmap="bwr"):
    """Difference panel with separate troposphere/stratosphere colour scales.

    The region with the wider scale is drawn over the full domain so the cells
    straddling the tropopause stay coloured (they sit near white on the wide
    scale); the tighter-scale region is then overlaid with its own scale.
    Returns (tropo_im, strat_im).
    """
    vals = np.asarray(values, dtype=float)
    tropo_mask = np.asarray(tropo_mask, dtype=bool)
    strat_is_wide = float(np.max(np.abs(strat_levels))) >= float(np.max(np.abs(tropo_levels)))
    base_levels = strat_levels if strat_is_wide else tropo_levels
    over_levels = tropo_levels if strat_is_wide else strat_levels
    over_mask = tropo_mask if strat_is_wide else ~tropo_mask
    base_im = ax.contourf(x2d, y2d, vals, levels=base_levels,
                          cmap=difference_cmap(base_levels, cmap), extend="both")
    over_im = ax.contourf(x2d, y2d, np.where(over_mask, vals, np.nan),
                          levels=over_levels,
                          cmap=difference_cmap(over_levels, cmap), extend="both")
    return (over_im, base_im) if strat_is_wide else (base_im, over_im)


def add_split_difference_colorbars(fig, tropo_im, strat_im, ax, tropo_levels,
                                   strat_levels, label, nticks=4, shrink_each=0.47):
    """Two stacked colorbars right of ``ax``: stratosphere above troposphere."""
    strat_ticks = symmetric_colorbar_ticks(strat_levels, nticks=nticks)
    strat_cbar = fig.colorbar(
        strat_im, ax=ax, location="right", shrink=shrink_each,
        anchor=(0.0, 1.0), ticks=strat_ticks, label=label,
    )
    strat_cbar.set_ticklabels([format_colorbar_tick(t) for t in strat_ticks])
    strat_cbar.ax.set_title("strat.", fontsize=8)

    # The lower bar lives in an inset of the upper (layout-managed) bar so the
    # pair stays one constrained-layout column, stacked vertically.
    gap = 1.0 - 2.0 * shrink_each
    tropo_cax = strat_cbar.ax.inset_axes([0.0, -(shrink_each + gap) / shrink_each, 1.0, 1.0])
    tropo_ticks = symmetric_colorbar_ticks(tropo_levels, nticks=nticks)
    tropo_cbar = fig.colorbar(tropo_im, cax=tropo_cax, ticks=tropo_ticks)
    tropo_cbar.set_ticklabels([format_colorbar_tick(t) for t in tropo_ticks])
    tropo_cbar.ax.set_title("trop.", fontsize=8)
    return tropo_cbar, strat_cbar


def make_spectrum_cmap(levels):
    base = plt.get_cmap("YlGnBu")
    centres = 0.5 * (levels[:-1] + levels[1:])
    cols_spec = []
    for c in centres:
        if c <= 1.0:
            cols_spec.append((1.0, 1.0, 1.0, 1.0))
        else:
            x = (c - 1.0) / (levels[-1] - 1.0)
            cols_spec.append(base(0.15 + 0.85 * x))
    cmap = colors.ListedColormap(cols_spec, name="spec_gt1")
    cmap.set_over(base(1.0))
    cmap.set_under((1.0, 1.0, 1.0, 1.0))
    norm = colors.BoundaryNorm(levels, cmap.N)
    return cmap, norm


def figure_should_run(figures_to_run, number):
    return figures_to_run is None or len(figures_to_run) == 0 or number in figures_to_run


def figure_should_run_any(figures_to_run, numbers):
    return figures_to_run is None or len(figures_to_run) == 0 or any(number in figures_to_run for number in numbers)


def all_figures_requested(figures_to_run):
    return figures_to_run is None or len(figures_to_run) == 0


class FigureSaver:
    """Save numbered figures beneath a versioned output directory."""

    def __init__(
        self,
        save_loc,
        enabled=True,
        start=1,
        dpi=300,
        figures_to_run=None,
        version=None,
        date_stamp=None,
    ):
        self.target_dir = diagnostic_output_dir(save_loc)
        environment_version = os.environ.get(DIAGNOSTIC_FIGURE_VERSION_ENV)
        selected_version = str(
            version
            or date_stamp
            or environment_version
            or DEFAULT_DIAGNOSTIC_FIGURE_VERSION
        )
        if (
            selected_version in {".", ".."}
            or os.path.basename(selected_version) != selected_version
        ):
            raise ValueError("Figure version must be one directory name")
        self.version = selected_version
        # Retain the historical attribute for callers that inspect it.
        self.date_stamp = selected_version
        if environment_version and version is None and date_stamp is None:
            self.save_loc = figure_release_directory(
                self.target_dir, selected_version
            )
        else:
            self.save_loc = os.path.join(self.target_dir, selected_version)
        self.enabled = enabled
        self.index = start
        self.dpi = dpi
        self.figures_to_run = figures_to_run
        os.makedirs(self.save_loc, exist_ok=True)

    def save(self, fig, stem, number=None, dpi=None):
        if number is None:
            number = self.index
            self.index += 1
        fname = f"{number:02d}_{stem}.png"
        if self.enabled and figure_should_run(self.figures_to_run, number):
            output_path = os.path.join(self.save_loc, fname)
            fig.savefig(output_path, dpi=self.dpi if dpi is None else dpi)
            print(f"Done with figure file: {output_path}", flush=True)
            return output_path
        return os.path.join(self.save_loc, fname)


def numbered_figure_saver(figure_saver):
    """Bind a FigureSaver to the common ``(number, fig, stem)`` call shape."""
    return lambda number, fig, stem: figure_saver.save(fig, stem, number=number)


# -----------------------------------------------------------------------------
# Shared runtime, data-selection, and cache helpers
# -----------------------------------------------------------------------------


def env_flag(name, default=False):
    """Read a conventional truthy/falsey environment flag."""
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def progress(message):
    """Print timestamped progress for long diagnostics and captured logs."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def safe_name(text):
    """Return the filename-safe convention used by diagnostic cache products."""
    return str(text).replace(" ", "_").replace("/", "-").replace(".", "p").replace("-", "m")


def configured_label(mapping, name):
    """Return a configured display label, falling back to the key itself."""

    return mapping.get(name, {}).get("label", name)


def subset_between(da, dim, bounds):
    """Subset a coordinate range without assuming coordinate ordering."""
    if bounds is None or dim not in da.dims:
        return da
    lo, hi = sorted(bounds)
    return da.where((da[dim] >= lo) & (da[dim] <= hi), drop=True)


def pressure_layer_weights(pressure):
    """Return positive pressure-layer weights proportional to d ln(p)."""
    values = np.asarray(
        pressure.values if isinstance(pressure, xr.DataArray) else pressure,
        dtype=float,
    )
    if np.any(values <= 0):
        raise ValueError("Pressure values must be positive before taking log(p)")
    if values.size == 1:
        weights = np.ones_like(values)
    else:
        log_pressure = np.log(values)
        bounds = np.empty(values.size + 1)
        bounds[1:-1] = 0.5 * (log_pressure[:-1] + log_pressure[1:])
        bounds[0] = log_pressure[0] - 0.5 * (log_pressure[1] - log_pressure[0])
        bounds[-1] = log_pressure[-1] + 0.5 * (log_pressure[-1] - log_pressure[-2])
        weights = np.abs(np.diff(bounds))
    return xr.DataArray(
        weights,
        dims=("pres",),
        coords={"pres": values},
        name="dlnp_weight",
    )


def bounded_pressure_integral(field, pressure_pa, top_pressure_pa,
                              bottom_pressure_pa, dim="pres"):
    """Integrate a pressure-level field between spatially varying bounds.

    The pressure coordinate must be one-dimensional.  Values at partial layer
    endpoints are linearly interpolated, and bounds outside the supplied
    pressure range are clipped.  The returned integral retains the pressure
    units supplied by ``pressure_pa`` (normally Pa).
    """
    if dim not in field.dims:
        raise ValueError(f"{dim!r} is not a dimension of {field.dims!r}")

    pressure = np.asarray(
        pressure_pa.values if isinstance(pressure_pa, xr.DataArray) else pressure_pa,
        dtype=float,
    )
    if pressure.ndim != 1 or pressure.size != field.sizes[dim]:
        raise ValueError("pressure_pa must be one-dimensional and match the pressure dimension")
    if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0.0):
        raise ValueError("pressure_pa must contain finite positive values")

    order = np.argsort(pressure)
    pressure = pressure[order]
    if np.any(np.diff(pressure) <= 0.0):
        raise ValueError("pressure_pa must contain unique levels")
    field = field.isel({dim: order})

    top = (
        top_pressure_pa
        if isinstance(top_pressure_pa, xr.DataArray)
        else xr.DataArray(float(top_pressure_pa))
    )
    bottom = (
        bottom_pressure_pa
        if isinstance(bottom_pressure_pa, xr.DataArray)
        else xr.DataArray(float(bottom_pressure_pa))
    )
    top, bottom = xr.broadcast(top, bottom)
    result = xr.zeros_like(top + bottom, dtype=float)
    for index in range(pressure.size - 1):
        p0 = float(pressure[index])
        p1 = float(pressure[index + 1])
        lo = xr.where(top > p0, top, p0)
        hi = xr.where(bottom < p1, bottom, p1)
        width = (hi - lo).clip(min=0.0)

        f0 = field.isel({dim: index}, drop=True)
        f1 = field.isel({dim: index + 1}, drop=True)
        slope = (f1 - f0) / (p1 - p0)
        flo = f0 + slope * (lo - p0)
        fhi = f0 + slope * (hi - p0)
        contribution = 0.5 * (flo + fhi) * width
        result = result + contribution.where(width > 0.0, 0.0)

    result.name = getattr(field, "name", None)
    result.attrs.update(getattr(field, "attrs", {}))
    return result


def interpolate_isobaric_to_pressure(field, target_pressure_hpa, dim="pres"):
    """Linearly interpolate a fixed-isobaric field to a varying pressure."""
    if dim not in field.dims:
        raise ValueError(f"{dim!r} is not a dimension of {field.dims!r}")
    pressure = np.asarray(field[dim].values, dtype=float)
    if pressure.ndim != 1 or np.any(~np.isfinite(pressure)):
        raise ValueError("The isobaric coordinate must be one-dimensional and finite")

    order = np.argsort(pressure)
    pressure = pressure[order]
    field = field.isel({dim: order})
    target = target_pressure_hpa
    out = xr.full_like(target, np.nan, dtype=float)
    for index in range(pressure.size - 1):
        p0 = float(pressure[index])
        p1 = float(pressure[index + 1])
        fraction = (target - p0) / (p1 - p0)
        value = (
            field.isel({dim: index}, drop=True)
            + fraction
            * (
                field.isel({dim: index + 1}, drop=True)
                - field.isel({dim: index}, drop=True)
            )
        )
        out = xr.where((target >= p0) & (target <= p1), value, out)
    out.name = getattr(field, "name", None)
    out.attrs.update(getattr(field, "attrs", {}))
    return out


def spherical_flux_divergence(zonal_flux, meridional_flux,
                              radius=6_371_000.0,
                              latitude="latitude", longitude="longitude"):
    """Return horizontal divergence of a vector flux on a regular lat-lon grid."""
    for dim in (latitude, longitude):
        if dim not in zonal_flux.dims or dim not in meridional_flux.dims:
            raise ValueError(f"Both fluxes must contain {dim!r}")

    lon = np.asarray(zonal_flux[longitude].values, dtype=float)
    if lon.size < 3:
        raise ValueError("At least three longitude points are required")
    lon_step = np.deg2rad(np.diff(lon))
    median_lon_step = float(np.median(lon_step))
    if not np.allclose(lon_step, median_lon_step, rtol=5e-5, atol=1e-12):
        raise ValueError("Longitude must be regularly spaced")

    coslat = np.cos(np.deg2rad(zonal_flux[latitude]))
    d_zonal_dlambda = (
        zonal_flux.roll({longitude: -1}, roll_coords=False)
        - zonal_flux.roll({longitude: 1}, roll_coords=False)
    ) / (2.0 * median_lon_step)
    d_meridional_cos_dphi = (
        meridional_flux * coslat
    ).differentiate(latitude, edge_order=2) * (180.0 / np.pi)

    divergence = (d_zonal_dlambda + d_meridional_cos_dphi) / (radius * coslat)
    divergence.name = "horizontal_flux_divergence"
    divergence.attrs.update(
        long_name="spherical horizontal flux divergence",
        earth_radius_m=float(radius),
    )
    return divergence


def spherical_horizontal_advection(scalar, zonal_wind, meridional_wind,
                                    radius=6_371_000.0,
                                    latitude="latitude", longitude="longitude"):
    """Return ``u/(a cos(phi))*d/dlambda + v/a*d/dphi`` for a scalar."""
    lon = np.asarray(scalar[longitude].values, dtype=float)
    if lon.size < 3:
        raise ValueError("At least three longitude points are required")
    lon_step = np.deg2rad(np.diff(lon))
    median_lon_step = float(np.median(lon_step))
    if not np.allclose(lon_step, median_lon_step, rtol=5e-5, atol=1e-12):
        raise ValueError("Longitude must be regularly spaced")

    coslat = np.cos(np.deg2rad(scalar[latitude]))
    dscalar_dlambda = (
        scalar.roll({longitude: -1}, roll_coords=False)
        - scalar.roll({longitude: 1}, roll_coords=False)
    ) / (2.0 * median_lon_step)
    dscalar_dphi = scalar.differentiate(latitude, edge_order=2) * (180.0 / np.pi)
    out = (
        zonal_wind * dscalar_dlambda / (radius * coslat)
        + meridional_wind * dscalar_dphi / radius
    )
    out.name = "horizontal_advection"
    out.attrs["earth_radius_m"] = float(radius)
    return out


def latitude_weights(latitude_or_field):
    """Return non-negative cosine-latitude weights on the latitude dimension."""
    latitude = (
        latitude_or_field.latitude
        if isinstance(latitude_or_field, xr.DataArray)
        and "latitude" in latitude_or_field.dims
        and latitude_or_field.ndim != 1
        else latitude_or_field
    )
    values = np.asarray(
        latitude.values if isinstance(latitude, xr.DataArray) else latitude,
        dtype=float,
    )
    weights = np.maximum(np.cos(np.deg2rad(values)), 0.0)
    return xr.DataArray(
        weights,
        dims=("latitude",),
        coords={"latitude": values},
        name="coslat_weight",
    )


def time_step_days(da):
    """Return the median decoded-datetime Time increment in days."""
    if "Time" not in da.dims or da.sizes["Time"] < 2:
        return 0.25
    values = da.Time.values
    if not np.issubdtype(values.dtype, np.datetime64):
        raise TypeError(f"Expected decoded datetime Time coordinate, got {values.dtype}")
    deltas = np.diff(values) / np.timedelta64(1, "D")
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if not deltas.size:
        raise ValueError("Time coordinate has no positive increments")
    return float(np.median(deltas))


def time_block_slices(
    da,
    block_days,
    complete=False,
    min_blocks=0,
    require_time=False,
):
    """Construct time-block slices for resampling or bounded calculations."""
    if "Time" not in da.dims:
        if require_time:
            raise ValueError(f"Cannot block-resample without Time dimension in {da.dims!r}")
        return [slice(None)]
    length = max(
        1,
        int(
            round(
                block_days
                / time_step_days(da)
            )
        ),
    )
    ntime = int(da.sizes["Time"])
    if complete:
        nblocks = ntime // length
        if nblocks < min_blocks:
            raise ValueError(
                f"Only {nblocks} complete blocks for {getattr(da, 'name', None)!r}; "
                f"need at least {min_blocks}. Try a shorter block length."
            )
        return [slice(index * length, (index + 1) * length) for index in range(nblocks)]
    return [slice(start, min(start + length, ntime)) for start in range(0, ntime, length)]


def add_period_reference_lines_frequency(ax, periods):
    """Annotate frequency axes with horizontal reference-period guides."""
    _, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ymin, ymax = sorted((y0, y1))
    for period in periods:
        frequency = 1.0 / period
        if ymin <= frequency <= ymax:
            ax.axhline(frequency, color="0.25", lw=0.8, ls=":", alpha=0.8)
            ax.text(
                x1,
                frequency,
                f"{period:g} d",
                ha="right",
                va="bottom",
                fontsize=7,
                color="0.25",
            )


def count_rows_to_frame(rows, stem, save=False, save_loc=None):
    """Report diagnostic sample counts and optionally save them as CSV."""
    frame = pd.DataFrame(rows)
    if len(frame) == 0:
        return frame
    print(f"\nSample counts for {stem}")
    print(frame.to_string(index=False, max_rows=240))
    if save:
        frame.to_csv(os.path.join(save_loc, f"{stem}_bin_counts.csv"), index=False)
    return frame


def subset_latitude(da, lat_bounds=None):
    """Subset a required latitude dimension to configured physical bounds."""
    if lat_bounds is None:
        return da
    if "latitude" not in da.dims:
        raise ValueError(
            f"Cannot apply latitude bounds to {getattr(da, 'name', None)!r}: "
            f"'latitude' is absent from {da.dims!r}"
        )
    lo, hi = sorted(lat_bounds)
    return da.where((da.latitude >= lo) & (da.latitude <= hi), drop=True)


def latitude_weight_array(da):
    """Broadcast cosine-latitude weights to the shape of a field."""
    if "latitude" not in da.dims:
        return xr.ones_like(da)
    weights = xr.DataArray(
        np.cos(np.deg2rad(da.latitude.values)),
        dims=("latitude",),
        coords={"latitude": da.latitude},
        name="latitude_weight",
    )
    return xr.broadcast(weights, da)[0]


def weighted_area_time_mean(da, lat_bounds=None):
    """Return a cosine-latitude-weighted mean over available map/time axes."""
    field = subset_latitude(da.astype(float), lat_bounds=lat_bounds)
    dims = [dim for dim in ("Time", "latitude", "longitude") if dim in field.dims]
    if "latitude" not in field.dims:
        return float(field.mean(dim=dims, skipna=True).values)
    weights = np.cos(np.deg2rad(field.latitude))
    return float(field.weighted(weights).mean(dim=dims, skipna=True).values)


def time_aligned_variable(
    ds,
    name,
    reference,
):
    """Load a required variable and select the reference timestamps exactly."""
    out = ds[name]
    if "Time" in out.dims and "Time" in reference.dims:
        out = out.sel(Time=reference.Time)
    return out


def precipitation_rate_from_dataset(ds):
    """Return total precipitation rate from the four MPAS accumulators."""
    precip = precip_rate_from_accum(total_precip_accum(ds))
    precip.name = "precip"
    precip.attrs["units"] = "mm day-1"
    return precip


def time_days_from_coord(time_coord):
    """Convert a decoded datetime coordinate to elapsed days."""
    values = np.asarray(time_coord.values)
    if not np.issubdtype(values.dtype, np.datetime64):
        raise TypeError(f"Expected decoded datetime Time coordinate, got {values.dtype}")
    return (values - values[0]) / np.timedelta64(1, "D")


def median_time_step_days(da):
    """Return the median Time increment in days."""
    values = np.asarray(da.Time.values)
    if values.size < 2:
        return 0.25
    if not np.issubdtype(values.dtype, np.datetime64):
        raise TypeError(f"Expected decoded datetime Time coordinate, got {values.dtype}")
    return float(np.median(np.diff(values) / np.timedelta64(1, "D")))


def weighted_latitude_band_mean(da, lat_bounds):
    """Return a cosine-latitude-weighted horizontal mean."""
    field = subset_latitude(finite_or_nan(da), lat_bounds=lat_bounds)
    dims = [dim for dim in ("latitude", "longitude") if dim in field.dims]
    if "latitude" in field.dims:
        weights = np.cos(np.deg2rad(field.latitude))
        return field.weighted(weights).mean(dim=dims, skipna=True)
    return field.mean(dim=dims, skipna=True)


def latitude_field_like(reference):
    """Broadcast a field's latitude coordinate across all of its dimensions."""
    latitude = xr.DataArray(
        reference.latitude.values,
        dims=("latitude",),
        coords={"latitude": reference.latitude},
        name="latitude",
    )
    return xr.broadcast(latitude, reference)[0]


def coordinate_edges(values):
    """Construct cell edges from one-dimensional cell centres."""
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def grid_cell_area_km2(lat, lon, earth_radius_km=6371.0):
    """Return spherical latitude/longitude grid-cell areas in square km."""
    lat_edges = np.clip(coordinate_edges(lat), -90.0, 90.0)
    lon_edges = coordinate_edges(lon)
    dlon = np.abs(np.diff(lon_edges))
    lat_factor = np.diff(np.sin(np.deg2rad(lat_edges)))
    area = earth_radius_km ** 2 * np.deg2rad(dlon)[None, :] * np.abs(lat_factor)[:, None]
    return xr.DataArray(
        area,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="cell_area",
        attrs={"units": "km2"},
    )


def haversine_distances_km(
    lat1,
    lon1,
    lat2,
    lon2,
    lon_period=360.0,
    earth_radius_km=6371.0,
):
    """Return periodic great-circle distances in km for broadcastable inputs."""
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    lon1 = np.asarray(lon1, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    dlat = lat2 - lat1
    dlon_deg = (lon2 - lon1 + 0.5 * lon_period) % lon_period - 0.5 * lon_period
    dlon = np.deg2rad(dlon_deg)
    haversine = (
        np.sin(0.5 * dlat) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * dlon) ** 2
    )
    return 2.0 * earth_radius_km * np.arcsin(np.minimum(1.0, np.sqrt(haversine)))


def weighted_quantile(values, quantiles, weights=None):
    """Return weighted quantiles using midpoint positions in the weight CDF."""
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    valid = np.isfinite(values)
    values = values[valid]
    if weights is None:
        if values.size == 0:
            return np.full_like(quantiles, np.nan, dtype=float)
        return np.nanpercentile(values, quantiles * 100.0)

    weights = np.asarray(weights, dtype=float)[valid]
    positive = weights > 0
    values = values[positive]
    weights = weights[positive]
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) - 0.5 * weights
    cdf /= np.sum(weights)
    return np.interp(quantiles, cdf, values)


def sample_matching_arrays(
    *arrays,
    lat_bounds,
    max_time=96,
    max_lat=96,
    max_lon=192,
    return_weights=True,
    time_mode="stride",
):
    """Align, stride, broadcast, and flatten matching xarray arrays."""
    prepared = [
        subset_latitude(
            finite_or_nan(array.astype(float)) if hasattr(array, "astype") else array,
            lat_bounds=lat_bounds,
        )
        for array in arrays
    ]
    prepared = list(xr.align(*prepared, join="inner"))
    indexers = stride_indexers(
        prepared,
        max_time=max_time,
        max_lat=max_lat,
        max_lon=max_lon,
        time_mode=time_mode,
    )
    if indexers:
        prepared = [
            array.isel({dim: slc for dim, slc in indexers.items() if dim in array.dims})
            for array in prepared
        ]

    broadcasted = xr.broadcast(*prepared)
    values = [np.asarray(array.values).ravel() for array in broadcasted]
    valid = np.ones(values[0].shape, dtype=bool)
    for value in values:
        valid &= np.isfinite(value)
    if return_weights:
        weights = np.asarray(latitude_weight_array(broadcasted[0]).values).ravel()
        valid &= np.isfinite(weights) & (weights > 0)
        return [value[valid] for value in values], weights[valid]
    return [value[valid] for value in values]


def stride_indexers(
    arrays,
    max_time=None,
    max_lat=None,
    max_lon=None,
    time_mode="stride",
):
    """Build bounded-size indexers shared by flattened sampling routines."""
    limits = {"Time": max_time, "latitude": max_lat, "longitude": max_lon}
    indexers = {}
    for dim, maximum in limits.items():
        if maximum is None:
            continue
        size = max((array.sizes[dim] for array in arrays if dim in array.dims), default=0)
        if size <= maximum:
            continue
        if dim == "Time" and time_mode == "tail":
            indexers[dim] = slice(size - maximum, None)
        else:
            indexers[dim] = slice(0, None, int(np.ceil(size / maximum)))
    return indexers


def compute_ttl_diagnostics(zonal_dataset, precip_reference):
    """Derive TTL static stability and shear from the fixed zonal schema."""
    z_coord = zonal_dataset.z
    ttl_mask = xr.DataArray(
        (z_coord >= TTL_HEIGHT_BOUNDS_M[0])
        & (z_coord <= TTL_HEIGHT_BOUNDS_M[1]),
        dims=("z",),
        coords={"z": z_coord},
    )
    theta = zonal_dataset.theta
    wind = zonal_dataset.uReconstructZonal
    output = {
        "N2_TTL": (9.80665 / theta * theta.differentiate("z"))
        .where(ttl_mask, drop=True)
        .mean(dim="z", skipna=True),
        "dudz_TTL": wind.differentiate("z")
        .where(ttl_mask, drop=True)
        .mean(dim="z", skipna=True),
    }
    for name, array in output.items():
        output[name] = array.sel(Time=precip_reference.Time).rename(name)
    return output


def residual_feature_arrays(fields, feature_keys):
    """Build the common environmental predictors for cloud residual models."""
    arrays = []
    for key in feature_keys:
        if key == "latitude":
            arrays.append(latitude_field_like(fields["precip"]))
        elif key == "precip_log":
            arrays.append(np.log1p(fields["precip"].clip(min=0.0)))
        else:
            arrays.append(fields[key])
    return arrays


def residual_target_array(classification, target_key):
    """Return one of the shared high-cloud residual-model targets."""
    if target_key == "total":
        return classification["high_cloud_fraction"].clip(min=0.0, max=1.0)
    variable = {
        "convective": "convective_precipitating_high_cloud_mask",
        "stratiform": "stratiform_precipitating_high_cloud_mask",
        "nonprecipitating": "nonprecipitating_high_cloud_mask",
    }.get(target_key)
    if variable is None:
        raise ValueError(target_key)
    return classification[variable].astype(float)


def residual_model_training_data(
    fields,
    classification,
    target_key,
    feature_keys,
    lat_bounds,
    sample_limits,
):
    """Build the common weighted design matrix for cloud residual models."""
    arrays = residual_feature_arrays(fields, feature_keys)
    arrays.append(residual_target_array(classification, target_key))
    values, weights = sample_matching_arrays(
        *arrays,
        lat_bounds=lat_bounds,
        **sample_limits,
    )
    return np.column_stack(values[:-1]), values[-1], weights


def available_residual_features():
    """Return the fixed residual-model predictor set."""
    return ["precip_log", "pw", "cape", "cin", "latitude"]


def fit_control_relationship(
    X,
    y,
    weights,
    target_label,
    logger=print,
    report_backend=False,
):
    """Fit the shared control high-cloud model."""
    valid = (
        np.all(np.isfinite(X), axis=1)
        & np.isfinite(y)
        & np.isfinite(weights)
        & (weights > 0)
    )
    X = X[valid]
    y = y[valid]
    weights = weights[valid]
    logger(f"Fitting control high-cloud model for {target_label}: {X.shape[0]} samples")
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        min_samples_leaf=40,
        random_state=0,
    )
    model.fit(X, y, sample_weight=weights)
    if report_backend:
        logger(f"  used HistGradientBoostingRegressor for {target_label}")
    return model


def symmetric_percentile_levels_for_fields(
    fields,
    percentile=99.0,
    ndiv=12,
    zero_field_scale=1.0,
):
    """Build shared symmetric levels from the finite values in several fields."""
    values = []
    for field in fields:
        if field is None:
            raise ValueError("Cannot calculate contour levels for a missing field")
        finite = np.asarray(field.values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            values.append(np.abs(finite))
    if not values:
        raise ValueError("Cannot calculate contour levels from non-finite data")
    vmax = np.nanpercentile(np.concatenate(values), percentile)
    if not np.isfinite(vmax):
        raise ValueError("Calculated contour scale is non-finite")
    if vmax == 0.0:
        vmax = zero_field_scale
    return symmetric_levels(vmax, ndiv=ndiv)


# -----------------------------------------------------------------------------
# Memory-bounded mean-state precipitation helpers
# -----------------------------------------------------------------------------


DEFAULT_DEEP_TROPICAL_LAT_BOUNDS = (-10.0, 10.0)


def accumulated_precipitation_components(ds):
    """Return total, convective, and resolved accumulated rain as float32."""
    hundred = np.float32(100.0)
    convective = ds["rainc"].astype(np.float32) + ds["i_rainc"].astype(np.float32) * hundred
    resolved = ds["rainnc"].astype(np.float32) + ds["i_rainnc"].astype(np.float32) * hundred
    return (convective + resolved).astype(np.float32), convective, resolved


def domain_fields(
    field,
    deep_tropical_lat_bounds=DEFAULT_DEEP_TROPICAL_LAT_BOUNDS,
):
    """Return lazy global and deep-tropical views without loading either."""
    lat_name = "latitude" if "latitude" in field.dims else "lat"
    lo, hi = deep_tropical_lat_bounds
    latitude = field[lat_name]
    deep_tropical = field.where(
        (latitude >= lo) & (latitude <= hi), drop=True
    )
    return field, deep_tropical


def _histogram_task(field, bins, value_range):
    values = field.data
    if not isinstance(values, dsa.Array):
        return np.histogram(np.asarray(values), bins=bins, range=value_range)
    return dsa.histogram(values.ravel(), bins=bins, range=value_range)


def domain_histograms(
    field,
    bins,
    value_range=None,
    deep_tropical_lat_bounds=DEFAULT_DEEP_TROPICAL_LAT_BOUNDS,
):
    """Compute global and deep-tropical histograms without loading the field."""
    tasks = [
        _histogram_task(domain, bins=bins, value_range=value_range)
        for domain in domain_fields(field, deep_tropical_lat_bounds)
    ]
    count_tasks = [task[0] for task in tasks]
    counts = (
        dask.compute(*count_tasks)
        if any(isinstance(task, dsa.Array) for task in count_tasks)
        else tuple(np.asarray(task) for task in count_tasks)
    )
    edges = np.asarray(tasks[0][1])
    return tuple((np.asarray(count), edges) for count in counts)


def relative_frequency_from_histogram(counts, edges, smooth):
    """Apply the diagnostic's smoothing and zero-bin normalisation."""
    from scipy.ndimage import uniform_filter1d

    smoothed = uniform_filter1d(np.asarray(counts, dtype=float), size=smooth)
    scale = smoothed[0] if smoothed[0] > 0 else np.nanmax(smoothed)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, smoothed / scale


def duration_domain_histograms(
    field,
    threshold,
    bins,
    value_range,
    deep_tropical_lat_bounds=DEFAULT_DEEP_TROPICAL_LAT_BOUNDS,
    time_block=8,
):
    """Return global/deep-tropical wet-spell histograms with bounded memory."""
    lat_name = "latitude" if "latitude" in field.dims else "lat"
    lon_name = "longitude" if "longitude" in field.dims else "lon"
    time_name = "Time" if "Time" in field.dims else "time"
    ordered = field.transpose(time_name, lat_name, lon_name)
    edges = np.histogram_bin_edges([], bins=bins, range=value_range)
    global_counts = np.zeros(edges.size - 1, dtype=np.int64)
    deep_tropical_counts = np.zeros_like(global_counts)
    latitudes = np.asarray(ordered[lat_name].values)
    lo, hi = deep_tropical_lat_bounds
    deep_tropical_mask = (latitudes >= lo) & (latitudes <= hi)
    run_dtype = np.int16 if ordered.sizes[time_name] <= np.iinfo(np.int16).max else np.int32
    run_lengths = np.zeros(
        (ordered.sizes[lat_name], ordered.sizes[lon_name]),
        dtype=run_dtype,
    )
    for time_start in range(0, ordered.sizes[time_name], time_block):
        time_stop = min(time_start + time_block, ordered.sizes[time_name])
        values = ordered.isel({time_name: slice(time_start, time_stop)}).values
        wet_block = np.isfinite(values) & (values > threshold)
        for wet in wet_block:
            run_lengths[wet] += 1
            ended = (~wet) & (run_lengths > 0)
            if not ended.any():
                continue
            global_counts += np.histogram(run_lengths[ended], bins=edges)[0]
            deep_tropical_ended = ended & deep_tropical_mask[:, None]
            if deep_tropical_ended.any():
                deep_tropical_counts += np.histogram(
                    run_lengths[deep_tropical_ended], bins=edges
                )[0]
            run_lengths[ended] = 0
    return (global_counts, edges), (deep_tropical_counts, edges)
