"""Shared archive coordinate normalization without plotting dependencies."""

import numpy as np


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
