import glob
import os
import numbers

import numpy as np
import pandas as pd
from matplotlib import colors, pyplot as plt
import scipy.interpolate as intp
import xarray as xr


ISOBARIC_DIM_LEVEL_VAR = {
    "nIsoLevelsT": "t_iso_levels",
    "nIsoLevelsU": "u_iso_levels",
    "nIsoLevelsZ": "z_iso_levels",
}


def silence_hdf5_error_stack():
    """Disable HDF5 C-library error-stack printing, without hiding Python errors."""
    try:
        import h5py
    except Exception:
        return False

    try:
        h5py._errors.silence_errors()
    except Exception:
        return False
    return True


def standardize_cldfrac_vertical_coordinate(ds):
    """Give cldfrac a geometric-height z coordinate regardless of its source."""
    if "cldfrac" not in ds.data_vars:
        return ds

    cldfrac = ds["cldfrac"]
    if "nVertLevels" in cldfrac.dims:
        cldfrac = cldfrac.rename({"nVertLevels": "z"})
    elif "nVertLevelsP1" in cldfrac.dims:
        raise ValueError("cldfrac appears to be on interface levels, not mid-levels")
    elif "z" not in cldfrac.dims:
        raise ValueError(f"Could not find vertical dimension in cldfrac dims {cldfrac.dims}")

    if "z" in cldfrac.coords:
        zvals = np.asarray(cldfrac["z"].values)
        if zvals.size == cldfrac.sizes["z"] and np.nanmax(zvals) > 100.0:
            return ds.assign(cldfrac=cldfrac)

    if "zgrid" not in ds.data_vars:
        raise ValueError("Cannot assign geometric z coordinate to cldfrac: zgrid is missing")

    zgrid_mean = ds.zgrid.mean(dim=("latitude", "longitude"))
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
    cldfrac["z"].attrs["long_name"] = "geometric height of layer midpoints"
    return ds.assign(cldfrac=cldfrac)


def load_latlon_dataset(
    data_dir,
    latlon_pattern="latlon.2000-*.nc",
    diag_file="latlon_diag.nc",
    hist_file="latlon_hist.nc",
    cldfrac_file="cldfrac.nc",
    time_format="int",
    dt=6,
    ndays=None,
    end_date=None,
    chunks=None,
    prefer_pattern=False,
    add_cldfrac_file=True,
):
    diag_path = os.path.join(data_dir, diag_file)
    hist_path = os.path.join(data_dir, hist_file)
    cldfrac_path = os.path.join(data_dir, cldfrac_file)
    latlon_files = sorted(glob.glob(os.path.join(data_dir, latlon_pattern)))

    if prefer_pattern and latlon_files:
        print(data_dir, latlon_pattern)
        ds = xr.open_mfdataset(latlon_files, combine="by_coords", chunks={}, data_vars="minimal", coords="minimal", compat="override")
        print(ds)
    elif os.path.exists(diag_path) and os.path.exists(hist_path):
        dsd = xr.open_dataset(diag_path, chunks={})
        dsh = xr.open_dataset(hist_path, chunks={})
        dsd = dsd.isel(Time=slice(0, None, 2)) if time_format == "int" else dsd.sel(Time=dsh.Time)
        ds = xr.merge([dsh, dsd], join="exact", compat="override")
    elif latlon_files:
        print(data_dir, latlon_pattern)
        ds = xr.open_mfdataset(latlon_files, combine="by_coords", chunks={}, data_vars="minimal", coords="minimal", compat="override")
        print(ds)
    else:
        raise FileNotFoundError(
            f"Could not find either {latlon_pattern} or both {diag_file} and {hist_file} in {data_dir}"
        )

    if ndays is not None:
        if time_format == "int":
            ds = ds.isel(Time=slice(-ndays * int(24 / dt), None))
        else:
            t_end = pd.to_datetime(end_date) if end_date is not None else pd.to_datetime(ds.Time.max().item())
            ds = ds.sel(Time=slice(t_end - pd.Timedelta(days=ndays), t_end))

    if add_cldfrac_file and "cldfrac" not in ds.data_vars and os.path.exists(cldfrac_path):
        ds_cld = xr.open_dataset(cldfrac_path, chunks={})
        if "cldfrac" in ds_cld:
            cldfrac = ds_cld["cldfrac"].reindex(Time=ds.Time)
            ds = ds.assign(cldfrac=cldfrac)

    ds = standardize_cldfrac_vertical_coordinate(ds)

    return ds.chunk(chunks) if chunks is not None else ds


def get_time_values(ds, dt=6):
    if issubclass(ds.Time.values.dtype.type, numbers.Integral):
        return ds.Time.values * dt / 24.0
    return ds.Time.values


def finite_or_nan(da, max_abs=1e30):
    """Mask NaNs, infinities, and MPAS-style large fill values."""
    return da.where(np.isfinite(da) & (np.abs(da) < max_abs))


def total_precip_accum(ds):
    """Accumulated total precipitation in mm, including MPAS bucket counters."""
    return (
        ds["rainnc"] + ds["i_rainnc"] * 100.0
        + ds["rainc"] + ds["i_rainc"] * 100.0
    )


def precip_rate_from_accum(accum, dt=6):
    """Convert accumulated precipitation to a precipitation rate in mm day-1."""
    if issubclass(accum.Time.values.dtype.type, numbers.Integral):
        return accum.diff("Time") / (dt / 24.0)
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
        zgrid = zgrid.mean(dim="latitude")

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


def height_from_zm_latlon(zds):
    return zds["z"]


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
    theta = model_lat_field(zds.theta.mean(dim="Time"))
    z = height_from_zm_latlon(zds).values
    dtheta_dz = np.gradient(theta.values, z, axis=0)
    return xr.DataArray(
        g / theta.values * dtheta_dz,
        dims=theta.dims,
        coords=theta.coords,
        name="N2",
        attrs={"units": "s-2"},
    )


def _pressure_axis_values(ds, name, dim):
    if name not in ds.variables:
        raise ValueError(f"Dataset has dimension {dim!r} but no in-file pressure variable {name!r}")
    axis = ds[name]
    if dim not in axis.dims:
        raise ValueError(f"Pressure variable {name!r} is not defined on dimension {dim!r}; dims are {axis.dims!r}")
    values = np.asarray(axis.values, dtype=np.float64)
    values = np.squeeze(values)
    if values.ndim != 1:
        raise ValueError(f"Pressure variable {name!r} must be one-dimensional; shape is {axis.shape!r}")
    if values.size != ds.sizes[dim]:
        raise ValueError(
            f"Pressure variable {name!r} has {values.size} levels, "
            f"but dimension {dim!r} has {ds.sizes[dim]}"
        )
    return values


def pressure_values_from_dataset(ds):
    """Return the verified in-file pressure coordinate in hPa, if present."""
    axes = []
    if "pres" in ds.dims:
        axes.append(("pres", "pres", _pressure_axis_values(ds, "pres", "pres")))
        for level_var in ISOBARIC_DIM_LEVEL_VAR.values():
            if level_var in ds.variables and "pres" in ds[level_var].dims:
                axes.append(("pres", level_var, _pressure_axis_values(ds, level_var, "pres")))

    for dim, level_var in ISOBARIC_DIM_LEVEL_VAR.items():
        if dim in ds.dims:
            axes.append((dim, level_var, _pressure_axis_values(ds, level_var, dim)))

    if not axes:
        return None

    ref_dim, ref_name, ref_values = axes[0]
    for dim, name, values in axes[1:]:
        if ref_values.shape != values.shape or not np.allclose(ref_values, values, rtol=1e-8, atol=1e-10):
            raise ValueError(
                "In-file isobaric pressure axes disagree: "
                f"{ref_name!r} on {ref_dim!r} differs from {name!r} on {dim!r}"
            )
    return ref_values


def standardize_latlon_dataset(ds, needed_vars=None):
    if needed_vars is not None:
        level_vars = [
            v for v in ISOBARIC_DIM_LEVEL_VAR.values()
            if v in ds.variables and v not in needed_vars
        ]
        ds = ds[[v for v in needed_vars if v in ds.data_vars] + level_vars]
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
    return ds


def lat_mean(da, lat_dim="latitude"):
    return da.weighted(np.cos(np.deg2rad(da[lat_dim]))).mean(dim=lat_dim)


def plot_Tu_latpres(a, lats, pres, Tbar, ubar):
    T_im = a.contourf(lats, pres, Tbar, levels=np.arange(190, 305, 5), extend="both", cmap="coolwarm")
    u_im_neg = a.contour(lats, pres, ubar, levels=np.arange(-20, 0, 5), colors="k", linestyles=":", linewidths=0.5)
    u_im_zero = a.contour(lats, pres, ubar, levels=[0], colors="k", linewidths=1)
    u_im_pos = a.contour(lats, pres, ubar, levels=np.arange(5, 55, 5), colors="k", linewidths=0.5)
    a.clabel(u_im_pos, u_im_pos.levels, fontsize=6)
    a.clabel(u_im_neg, u_im_neg.levels, fontsize=6)
    return T_im, u_im_neg, u_im_zero, u_im_pos


def plot_rhv_latpres(a, lats, pres, rhbar, vbar, omgbar):
    xi = np.linspace(lats.min(), lats.max(), len(lats))
    yi = np.linspace(pres.min(), pres.max(), len(pres))
    X, Y = np.meshgrid(xi, yi)
    U = intp.interpn((lats, pres), vbar.T, (X, Y))
    V = intp.interpn((lats, pres), omgbar.T, (X, Y))
    rh_im = a.contourf(lats, pres, rhbar, levels=np.arange(0, 105, 5), cmap="BrBG")
    medcirc_im = a.streamplot(X, Y, U / 111111.0, V / 100.0, color="k", linewidth=1, density=1)
    return rh_im, medcirc_im


def isobaric_lat_mean(da):
    return da.mean(dim="Time").transpose("pres", "latitude")


def interp_like_control(field, control_field):
    return control_field.interp(pres=field.pres, latitude=field.latitude)


def set_latheight_labels(ax, xlabel=True):
    ax.set_xlim(-90, 90)
    if xlabel:
        ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Height (km)")


def set_latpres_axis(ax, xlabel=True, ylabel=True, xlim=(-60, 60), ylim=(1000, 50)):
    ax.set_xlim(*xlim)
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


def symmetric_levels(max_abs, ndiv=11):
    return np.linspace(-max_abs, max_abs, ndiv + 1)


def format_colorbar_tick(value):
    if np.isclose(value, 0.0, atol=1e-12):
        return "0"
    return f"{value:.3g}"


def symmetric_colorbar_ticks(levels, nticks=5):
    max_abs = float(np.nanmax(np.abs(levels)))
    ticks = np.linspace(-max_abs, max_abs, nticks)
    ticks[np.isclose(ticks, 0.0, atol=1e-12)] = 0.0
    return ticks


def add_symmetric_colorbar(fig, mappable, ax, levels, label, location="right", nticks=5, **kwargs):
    ticks = symmetric_colorbar_ticks(levels, nticks=nticks)
    cbar = fig.colorbar(mappable, ax=ax, location=location, label=label, ticks=ticks, **kwargs)
    cbar.set_ticklabels([format_colorbar_tick(tick) for tick in ticks])
    return cbar


def make_diff_norm(levels):
    return colors.BoundaryNorm(levels, plt.get_cmap("bwr").N)


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
    def __init__(self, save_loc, enabled=True, start=1, dpi=300, figures_to_run=None):
        self.save_loc = save_loc
        self.enabled = enabled
        self.index = start
        self.dpi = dpi
        self.figures_to_run = figures_to_run
        os.makedirs(save_loc, exist_ok=True)

    def save(self, fig, stem, number=None, dpi=None):
        if number is None:
            number = self.index
            self.index += 1
        fname = f"{number:02d}_{stem}.png"
        if self.enabled and figure_should_run(self.figures_to_run, number):
            fig.savefig(os.path.join(self.save_loc, fname), dpi=self.dpi if dpi is None else dpi)
        return fname
