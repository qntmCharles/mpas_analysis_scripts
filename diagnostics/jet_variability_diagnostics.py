#!/usr/bin/env python
# coding: utf-8

"""
Jet variability diagnostics for the 200-day MPAS aquaplanet integrations.

The script provides two complementary checks for jets becoming trapped in
particular configurations:

  * fixed-level zonal-mean wind, by default u at 700 hPa, with latitude-time
    wind fields, hemispheric jet-latitude tracks, and latitude-resolved
    autocorrelation of the wind fluctuations;
  * leading EOFs of zonal-mean u in each hemisphere, interpreted as annular-mode
    zonal indices, with EOF patterns, index time series, and autocorrelation.
  * model-height latitude-height jet structure, with nudged-control differences
    and subtropical jet-core markers.

The default configuration mirrors the other diagnostics in this directory.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr
import dask

try:
    from diagnostic_helpers import *
except ModuleNotFoundError:
    from diagnostics.diagnostic_helpers import *


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/jet_variability/")

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: fixed_level_zonal_wind_hovmoller
#   2: fixed_level_zonal_wind_autocorrelation
#   3: annular_mode_eof_patterns
#   4: annular_mode_indices_and_autocorrelation
#   5: time_mean_zonal_wind_profiles
#   6: subtropical_jet_latheight_structure
#
# WHAT TO LOOK FOR:
#   * Long jet autocorrelation or persistent jet-latitude anomalies indicate
#     slow internal variability that can contaminate small nudged-control
#     differences.
#   * EOF/annular-mode patterns identify whether differences are broad
#     hemispheric circulation shifts or local tropical/UTLS responses.
#   * If U-nudged changes jet variability without a matching T-nudged change,
#     that supports a wind-driven or shear-mediated pathway.
figures_to_run = [6]
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


def run_all_figures():
    return all_figures_requested(figures_to_run)


root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control", "u_nudge", "T_nudge"]
latlon_file = ["latlon.2000-*.nc", "latlon.2000-*.nc", "latlon.2000-*.nc"]
zm_latlon_file = ["zm_latlon.2000-*.nc", "zm_latlon.2000-*.nc", "zm_latlon.2000-*.nc"]
sim_names = ["control", "nudged u", "nudged T"]
cols = ["k", "b", "r"]

ndays = 200
dt = 6
analysis_end_date = "2000-11-21"
time_format = "time"

fixed_pressure_level = 250.0
jet_profile_pressure_levels = [700.0, 200.0]
time_average_days = 1.0
max_autocorr_lag_days = 80.0
autocorr_efold_threshold = np.exp(-1.0)

# EOFs are computed separately in each hemisphere. The latitude bounds avoid the
# equatorial transition region, where the annular-mode interpretation is weak.
hemisphere_lat_bounds = {
    "SH": (-85.0, -15.0),
    "NH": (15.0, 85.0),
}
hemisphere_jet_bounds = {
    "SH": (-85.0, -5.0),
    "NH": (5.0, 85.0),
}
subtropical_jet_lat_bounds = {
    "SH": (-45.0, -10.0),
    "NH": (10.0, 45.0),
}
subtropical_jet_height_bounds = (8000.0, 18000.0)
jet_structure_plot_height_bounds = (0.0, 25000.0)
eof_pressure_bounds = (1000.0, 100.0)
eof_min_valid_fraction = 0.95

latlon_chunks = {"Time": -1, "latitude": 60, "longitude": 120}
needed_vars = ["uzonal_isobaric"]
needed_vars_zm = ["uReconstructZonal"]

dask.config.set({"array.slicing.split_large_chunks": True})


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def safe_name(value):
    return str(value).replace(" ", "_").replace("/", "-").replace(".", "p").replace("-", "m")


def pressure_name(value):
    return safe_name(f"{float(value):g}")


def tight_layout_with_right_margin(fig, right=0.86, top=0.94):
    fig.tight_layout(rect=[0.0, 0.0, right, top])


def add_right_margin_colorbar(fig, mappable, axes, label, pad=0.015, width=0.025):
    axes_flat = [axis for axis in np.asarray(axes, dtype=object).ravel() if axis.get_visible()]
    if not axes_flat:
        return None
    boxes = [axis.get_position() for axis in axes_flat]
    x0 = max(box.x1 for box in boxes) + pad
    y0 = min(box.y0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    cax = fig.add_axes([x0, y0, width, y1 - y0])
    return fig.colorbar(mappable, cax=cax, label=label)


def subset_between(da, dim, bounds):
    """Subset a coordinate range without assuming ascending coordinates."""
    lo, hi = sorted(bounds)
    coord = da[dim]
    return da.where((coord >= lo) & (coord <= hi), drop=True)


def time_step_days(da):
    if "Time" not in da.dims or da.sizes["Time"] < 2:
        return dt / 24.0

    times = da.Time.values
    if np.issubdtype(times.dtype, np.number):
        deltas = np.diff(np.asarray(times, dtype=float)) * dt / 24.0
    elif np.issubdtype(times.dtype, np.datetime64):
        deltas = np.diff(times) / np.timedelta64(1, "D")
    else:
        times = pd.to_datetime(times)
        deltas = np.asarray(np.diff(times) / pd.Timedelta(days=1), dtype=float)

    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    return float(np.nanmedian(deltas)) if deltas.size else dt / 24.0


def time_days_since_start(da):
    times = da.Time.values
    if times.size == 0:
        return np.asarray([])
    if np.issubdtype(times.dtype, np.number):
        return (np.asarray(times, dtype=float) - float(times[0])) * dt / 24.0
    if np.issubdtype(times.dtype, np.datetime64):
        return np.asarray((times - times[0]) / np.timedelta64(1, "D"), dtype=float)
    times = pd.to_datetime(times)
    return np.asarray((times - times[0]) / pd.Timedelta(days=1), dtype=float)


def time_average(da, days=time_average_days):
    """Coarsen to regular multi-sample means while keeping the original calendar."""
    if days is None or days <= 0:
        return da
    step_days = time_step_days(da)
    window = int(round(days / step_days))
    if window <= 1:
        return da
    return da.coarsen(Time=window, boundary="trim").mean()


def pressure_layer_weights(pres):
    """Positive pressure weights proportional to d ln(p), with p in hPa."""
    if isinstance(pres, xr.DataArray):
        pres_values = np.asarray(pres.values, dtype=float)
        pres_coord = pres_values
    else:
        pres_values = np.asarray(pres, dtype=float)
        pres_coord = pres_values

    if np.any(pres_values <= 0):
        raise ValueError("Pressure values must be positive before taking log(p)")

    if pres_values.size == 1:
        weights = np.ones_like(pres_values)
    else:
        lnp = np.log(pres_values)
        bounds = np.empty(pres_values.size + 1)
        bounds[1:-1] = 0.5 * (lnp[:-1] + lnp[1:])
        bounds[0] = lnp[0] - 0.5 * (lnp[1] - lnp[0])
        bounds[-1] = lnp[-1] + 0.5 * (lnp[-1] - lnp[-2])
        weights = np.abs(np.diff(bounds))

    return xr.DataArray(weights, dims=("pres",), coords={"pres": pres_coord}, name="dlnp_weight")


def latitude_weights(latitude):
    values = np.asarray(latitude.values if isinstance(latitude, xr.DataArray) else latitude, dtype=float)
    weights = np.cos(np.deg2rad(values))
    weights = np.where(weights > 0, weights, 0.0)
    return xr.DataArray(weights, dims=("latitude",), coords={"latitude": values}, name="coslat_weight")


def pressure_latitude_weights(field):
    pw, lw = xr.broadcast(pressure_layer_weights(field.pres), latitude_weights(field.latitude))
    return (pw * lw).transpose("pres", "latitude")


def weighted_pressure_mean(field):
    weights = pressure_layer_weights(field.pres)
    valid = np.isfinite(field)
    numerator = (field * weights).where(valid).sum(dim="pres", skipna=True)
    denominator = weights.where(valid).sum(dim="pres", skipna=True)
    return numerator / denominator


def symmetric_percentile_levels_for_fields(fields, percentile=99.0, ndiv=12, fallback=1.0):
    vals = []
    for field in fields:
        if field is None:
            continue
        arr = np.asarray(field.values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(np.abs(arr))

    if vals:
        vmax = np.nanpercentile(np.concatenate(vals), percentile)
    else:
        vmax = fallback
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = fallback
    return symmetric_levels(vmax, ndiv=ndiv)


def model_height_time_mean_zonal_u(zds):
    if "uReconstructZonal" not in zds.data_vars:
        raise ValueError("uReconstructZonal is required for model-height jet structure")

    u = finite_or_nan(zds.uReconstructZonal)
    mean_dims = [dim for dim in ("Time", "longitude") if dim in u.dims]
    if mean_dims:
        u = u.mean(dim=mean_dims, skipna=True)
    u = model_lat_field(u).load()
    u.name = "u_time_mean_zonal_mean"
    u.attrs.update(units="m s-1", long_name="time-mean zonal-mean zonal wind")
    return u


def subtropical_jet_core(field, lat_bounds, z_bounds=subtropical_jet_height_bounds):
    domain = subset_between(field, "latitude", lat_bounds)
    domain = subset_between(domain, "z", z_bounds)
    if domain.sizes.get("latitude", 0) == 0 or domain.sizes.get("z", 0) == 0:
        return None

    values = np.asarray(domain.transpose("z", "latitude").values, dtype=float)
    if not np.isfinite(values).any():
        return None

    z_index, lat_index = np.unravel_index(np.nanargmax(values), values.shape)
    return {
        "latitude": float(domain.latitude.values[lat_index]),
        "height_m": float(domain.z.values[z_index]),
        "u_ms-1": float(values[z_index, lat_index]),
    }


def draw_subtropical_search_boxes(ax):
    z0 = subtropical_jet_height_bounds[0] / 1000.0
    z1 = subtropical_jet_height_bounds[1] / 1000.0
    for bounds in subtropical_jet_lat_bounds.values():
        x0, x1 = bounds
        ax.plot(
            [x0, x1, x1, x0, x0],
            [z0, z0, z1, z1, z0],
            color="0.15",
            lw=0.7,
            ls=":",
            alpha=0.8,
        )


def plot_jet_core_marker(ax, core, color, marker="o", label=None, zorder=4):
    if core is None:
        return
    if marker == "x":
        ax.scatter(
            core["latitude"],
            core["height_m"] / 1000.0,
            s=38,
            marker=marker,
            color=color,
            linewidth=1.0,
            label=label,
            zorder=zorder,
        )
        return

    ax.scatter(
        core["latitude"],
        core["height_m"] / 1000.0,
        s=34,
        marker=marker,
        facecolor=color,
        edgecolor="0.05",
        linewidth=0.8,
        label=label,
        zorder=zorder,
    )


def autocorrelation_1d(values, max_lag_steps, min_pairs=8):
    x = np.asarray(values, dtype=float)
    out = np.full(max_lag_steps + 1, np.nan)
    for lag in range(max_lag_steps + 1):
        if lag == 0:
            xs = x
            ys = x
        else:
            xs = x[:-lag]
            ys = x[lag:]
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < min_pairs:
            continue
        xs = xs[ok] - np.nanmean(xs[ok])
        ys = ys[ok] - np.nanmean(ys[ok])
        denom = np.sqrt(np.sum(xs * xs) * np.sum(ys * ys))
        if denom > 0:
            out[lag] = np.sum(xs * ys) / denom
    return out


def autocorrelation_timeseries(da, max_lag_days=max_autocorr_lag_days):
    step_days = time_step_days(da)
    max_steps = min(da.sizes["Time"] - 1, int(round(max_lag_days / step_days)))
    lags = np.arange(max_steps + 1)
    corr = autocorrelation_1d(da.values, max_steps)
    return xr.DataArray(
        corr,
        dims=("lag_days",),
        coords={"lag_days": lags * step_days},
        name="autocorrelation",
    )


def autocorrelation_by_latitude(da, max_lag_days=max_autocorr_lag_days):
    step_days = time_step_days(da)
    max_steps = min(da.sizes["Time"] - 1, int(round(max_lag_days / step_days)))
    lags = np.arange(max_steps + 1)

    values = da.transpose("Time", "latitude").load().values
    corr = np.full((max_steps + 1, da.sizes["latitude"]), np.nan)
    for j in range(da.sizes["latitude"]):
        corr[:, j] = autocorrelation_1d(values[:, j], max_steps)

    return xr.DataArray(
        corr,
        dims=("lag_days", "latitude"),
        coords={"lag_days": lags * step_days, "latitude": da.latitude},
        name="autocorrelation",
    )


def efolding_time(ac, threshold=autocorr_efold_threshold):
    """First lag where autocorrelation drops below threshold, linearly interpolated."""
    if "latitude" in ac.dims:
        vals = ac.transpose("lag_days", "latitude").values
        out = np.full(ac.sizes["latitude"], np.nan)
        for j in range(ac.sizes["latitude"]):
            out[j] = efolding_time(ac.isel(latitude=j), threshold=threshold).item()
        return xr.DataArray(out, dims=("latitude",), coords={"latitude": ac.latitude}, name="efolding_time_days")

    lags = np.asarray(ac.lag_days.values, dtype=float)
    vals = np.asarray(ac.values, dtype=float)
    ok = np.isfinite(vals)
    if ok.sum() == 0:
        return xr.DataArray(np.nan, name="efolding_time_days")
    vals = vals[ok]
    lags = lags[ok]
    crossed = np.where(vals <= threshold)[0]
    if crossed.size == 0:
        return xr.DataArray(np.nan, name="efolding_time_days")
    idx = crossed[0]
    if idx == 0:
        tau = lags[0]
    else:
        x0, x1 = vals[idx - 1], vals[idx]
        t0, t1 = lags[idx - 1], lags[idx]
        if np.isclose(x0, x1):
            tau = t1
        else:
            tau = t0 + (threshold - x0) * (t1 - t0) / (x1 - x0)
    return xr.DataArray(float(tau), name="efolding_time_days")


def ac_at_lag(ac, lag_days):
    if ac.sizes["lag_days"] == 0:
        return np.nan
    return float(ac.interp(lag_days=float(lag_days)).values)


def longest_boolean_run(mask):
    best = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def longest_same_sign_run_days(values, step_days):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    pos = finite & (values >= 0)
    neg = finite & (values < 0)
    return max(longest_boolean_run(pos), longest_boolean_run(neg)) * step_days


def save_dataset(ds, filename):
    if save:
        ds.to_netcdf(os.path.join(save_loc, filename))


# -----------------------------------------------------------------------------
# Data loading and wind diagnostics
# -----------------------------------------------------------------------------

dss = []
for i, d in enumerate(data_locs):
    ds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks=latlon_chunks,
        add_cldfrac_file=False,
    )
    ds = standardize_latlon_dataset(ds, needed_vars)
    print(ds)
    dss.append(ds)


dss_zm = []
if should_run_figure(6):
    for i, d in enumerate(data_locs):
        zds = load_latlon_dataset(
            os.path.join(root_dir, d),
            latlon_pattern=zm_latlon_file[i],
            time_format=time_format,
            ndays=ndays,
            end_date=analysis_end_date,
            chunks={"Time": -1},
            prefer_pattern=True,
            add_cldfrac_file=False,
        )
        zds = standardize_latlon_dataset(zds, needed_vars_zm)
        print(zds)
        dss_zm.append(zds)


def zonal_mean_u_at_level(ds, plev=fixed_pressure_level):
    if "uzonal_isobaric" not in ds.data_vars:
        raise ValueError("uzonal_isobaric is required for fixed-level jet diagnostics")
    u = finite_or_nan(ds.uzonal_isobaric).sel(pres=plev, method="nearest")
    selected_pressure = float(u.pres.values)
    if "longitude" in u.dims:
        u = u.mean(dim="longitude", skipna=True)
    u = time_average(u.transpose("Time", "latitude")).load()
    u.name = "u_zonal_mean"
    u.attrs.update(
        units="m s-1",
        selected_pressure_hpa=selected_pressure,
        time_average_days=time_average_days,
    )
    return u


def time_mean_zonal_mean_u_at_level(ds, plev):
    u = zonal_mean_u_at_level(ds, plev=plev)
    u_mean = u.mean(dim="Time", skipna=True).load()
    u_mean.name = "u_time_mean_zonal_mean"
    u_mean.attrs.update(u.attrs)
    u_mean.attrs["long_name"] = "time-mean zonal-mean zonal wind"
    return u_mean


def zonal_mean_u_profile(ds):
    if "uzonal_isobaric" not in ds.data_vars:
        raise ValueError("uzonal_isobaric is required for annular-mode diagnostics")
    u = finite_or_nan(ds.uzonal_isobaric)
    if "longitude" in u.dims:
        u = u.mean(dim="longitude", skipna=True)
    u = time_average(u.transpose("Time", "pres", "latitude")).load()
    u.name = "u_zonal_mean_profile"
    u.attrs.update(units="m s-1", time_average_days=time_average_days)
    return u


def hemisphere_jet_latitudes(u_lat_time):
    tracks = {}
    for hemisphere, bounds in hemisphere_jet_bounds.items():
        hemi = subset_between(u_lat_time, "latitude", bounds)
        if hemi.sizes.get("latitude", 0) < 2:
            continue
        tracks[hemisphere] = hemi.idxmax(dim="latitude", skipna=True).load()
    return tracks


def jet_latitude_summary(simulation, hemisphere, jet_lat, step_days):
    vals = np.asarray(jet_lat.values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "simulation": simulation,
            "hemisphere": hemisphere,
            "mean_latitude": np.nan,
            "std_latitude": np.nan,
            "min_latitude": np.nan,
            "max_latitude": np.nan,
            "modal_bin_center": np.nan,
            "modal_bin_fraction": np.nan,
            "longest_modal_residence_days": np.nan,
        }

    bins = np.arange(-90.0, 95.0, 5.0)
    hist, edges = np.histogram(vals, bins=bins)
    modal_idx = int(np.argmax(hist))
    modal_lo, modal_hi = edges[modal_idx], edges[modal_idx + 1]
    in_modal = (np.asarray(jet_lat.values, dtype=float) >= modal_lo) & (
        np.asarray(jet_lat.values, dtype=float) < modal_hi
    )
    return {
        "simulation": simulation,
        "hemisphere": hemisphere,
        "mean_latitude": float(np.nanmean(vals)),
        "std_latitude": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0,
        "min_latitude": float(np.nanmin(vals)),
        "max_latitude": float(np.nanmax(vals)),
        "modal_bin_center": float(0.5 * (modal_lo + modal_hi)),
        "modal_bin_fraction": float(hist[modal_idx] / vals.size),
        "longest_modal_residence_days": float(longest_boolean_run(in_modal) * step_days),
    }


fixed_level = {}
fixed_level_anomaly = {}
fixed_level_autocorr = {}
fixed_level_efold = {}
jet_latitudes = {}
fixed_summary_rows = []
jet_summary_rows = []

if should_run_any([1, 2]):
    for i, ds in enumerate(dss):
        try:
            u700 = zonal_mean_u_at_level(ds)
            anomaly = u700 - u700.mean(dim="Time", skipna=True)
            ac = autocorrelation_by_latitude(anomaly)
            tau = efolding_time(ac)
            tracks = hemisphere_jet_latitudes(u700)

            fixed_level[i] = u700
            fixed_level_anomaly[i] = anomaly
            fixed_level_autocorr[i] = ac
            fixed_level_efold[i] = tau
            jet_latitudes[i] = tracks

            finite_tau = tau.values[np.isfinite(tau.values)]
            if finite_tau.size:
                max_idx = int(np.nanargmax(tau.values))
                max_tau = float(tau.values[max_idx])
                max_tau_lat = float(tau.latitude.values[max_idx])
                median_tau = float(np.nanmedian(finite_tau))
            else:
                max_tau = np.nan
                max_tau_lat = np.nan
                median_tau = np.nan

            fixed_summary_rows.append(
                {
                    "simulation": sim_names[i],
                    "pressure_hpa": u700.attrs["selected_pressure_hpa"],
                    "time_average_days": time_average_days,
                    "median_efolding_time_days": median_tau,
                    "max_efolding_time_days": max_tau,
                    "max_efolding_time_latitude": max_tau_lat,
                }
            )

            step_days = time_step_days(u700)
            for hemisphere, track in tracks.items():
                jet_summary_rows.append(jet_latitude_summary(sim_names[i], hemisphere, track, step_days))
        except Exception as exc:
            print(f"Skipping fixed-level jet diagnostics for {sim_names[i]}: {exc}")

    if save and fixed_level:
        fixed_simulations = [sim_names[i] for i in fixed_level]
        fixed_ds = xr.concat(
            [
                xr.Dataset(
                    {
                        "u_zonal_mean": fixed_level[i],
                        "u_anomaly": fixed_level_anomaly[i],
                    }
                )
                for i in fixed_level
            ],
            dim=xr.IndexVariable("simulation", fixed_simulations),
            join="outer",
        )
        save_dataset(fixed_ds, f"u{pressure_name(fixed_pressure_level)}_zonal_mean_timeseries.nc")

        ac_simulations = [sim_names[i] for i in fixed_level_autocorr]
        ac_ds = xr.concat(
            [
                xr.Dataset(
                    {
                        "autocorrelation": fixed_level_autocorr[i],
                        "efolding_time_days": fixed_level_efold[i],
                    }
                )
                for i in fixed_level_autocorr
            ],
            dim=xr.IndexVariable("simulation", ac_simulations),
            join="outer",
        )
        save_dataset(ac_ds, f"u{pressure_name(fixed_pressure_level)}_zonal_mean_autocorrelation.nc")

        pd.DataFrame(fixed_summary_rows).to_csv(
            os.path.join(save_loc, f"u{pressure_name(fixed_pressure_level)}_persistence_summary.csv"),
            index=False,
        )
        pd.DataFrame(jet_summary_rows).to_csv(
            os.path.join(save_loc, f"u{pressure_name(fixed_pressure_level)}_jet_latitude_summary.csv"),
            index=False,
        )


# -----------------------------------------------------------------------------
# Annular-mode EOF diagnostics
# -----------------------------------------------------------------------------

def orient_annular_mode(eof, index, hemisphere):
    """
    Set EOF sign so positive index strengthens the poleward side of the pattern
    relative to the equatorward side.
    """
    vertical_mean = weighted_pressure_mean(eof)
    split_lat = float(np.nanmedian(vertical_mean.latitude.values))
    if hemisphere == "NH":
        poleward = vertical_mean.where(vertical_mean.latitude >= split_lat, drop=True)
        equatorward = vertical_mean.where(vertical_mean.latitude < split_lat, drop=True)
    else:
        poleward = vertical_mean.where(vertical_mean.latitude <= split_lat, drop=True)
        equatorward = vertical_mean.where(vertical_mean.latitude > split_lat, drop=True)

    contrast = float(poleward.mean(skipna=True).values - equatorward.mean(skipna=True).values)
    if np.isfinite(contrast) and contrast < 0:
        eof_attrs = eof.attrs.copy()
        index_attrs = index.attrs.copy()
        eof = -eof
        index = -index
        eof.attrs = eof_attrs
        index.attrs = index_attrs
        return eof, index
    return eof, index


def leading_annular_mode(zm_u, hemisphere, lat_bounds, pressure_bounds=eof_pressure_bounds):
    field = subset_between(zm_u, "latitude", lat_bounds)
    field = subset_between(field, "pres", pressure_bounds)
    field = field.transpose("Time", "pres", "latitude")
    if field.sizes["Time"] < 3:
        raise ValueError("Need at least three time samples for an EOF")
    if field.sizes["latitude"] < 2 or field.sizes["pres"] < 2:
        raise ValueError("EOF domain must contain at least two latitudes and two pressure levels")

    anomaly = field - field.mean(dim="Time", skipna=True)
    weights_2d = pressure_latitude_weights(anomaly.isel(Time=0))

    stacked = anomaly.stack(point=("pres", "latitude")).transpose("Time", "point")
    weights_flat = weights_2d.stack(point=("pres", "latitude")).transpose("point")
    valid_fraction = np.isfinite(stacked).mean(dim="Time").load()
    weights_values = weights_flat.load().values
    valid = (
        np.asarray(valid_fraction.values >= eof_min_valid_fraction, dtype=bool)
        & np.isfinite(weights_values)
        & (weights_values > 0)
    )
    if valid.sum() < 2:
        raise ValueError("Not enough valid weighted points in the EOF domain")

    x = stacked.isel(point=valid).load()
    values = np.asarray(x.values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    values = values - values.mean(axis=0, keepdims=True)

    sqrt_weights = np.sqrt(weights_values[valid])
    weighted_values = values * sqrt_weights[None, :]

    u_svd, singular_values, vt = np.linalg.svd(weighted_values, full_matrices=False)
    if singular_values.size == 0 or not np.isfinite(singular_values[0]):
        raise ValueError("EOF decomposition failed")

    pc = u_svd[:, 0] * singular_values[0]
    pc_std = float(np.nanstd(pc, ddof=1))
    if not np.isfinite(pc_std) or pc_std == 0:
        raise ValueError("Leading PC has zero variance")

    index_values = pc / pc_std
    eof_values = vt[0, :] / sqrt_weights * pc_std
    variance_fraction = float(singular_values[0] ** 2 / np.sum(singular_values ** 2))

    template = stacked.isel(Time=0).copy(data=np.full(stacked.sizes["point"], np.nan))
    template.values[valid] = eof_values
    eof = template.unstack("point").transpose("pres", "latitude")
    eof.name = f"{hemisphere}_annular_mode_eof"
    eof.attrs.update(
        units="m s-1 per index standard deviation",
        variance_fraction=variance_fraction,
        hemisphere=hemisphere,
        lat_bounds=f"{lat_bounds[0]},{lat_bounds[1]}",
        pressure_bounds_hpa=f"{pressure_bounds[0]},{pressure_bounds[1]}",
    )

    index = xr.DataArray(
        index_values,
        dims=("Time",),
        coords={"Time": field.Time},
        name=f"{hemisphere}_zonal_index",
        attrs={
            "units": "standard deviations",
            "variance_fraction": variance_fraction,
            "hemisphere": hemisphere,
        },
    )

    eof, index = orient_annular_mode(eof, index, hemisphere)
    return eof, index, variance_fraction


annular_results = {}
annular_summary_rows = []
annular_index_rows = []

if should_run_any([3, 4]):
    for i, ds in enumerate(dss):
        try:
            zm_u = zonal_mean_u_profile(ds)
        except Exception as exc:
            print(f"Skipping annular-mode diagnostics for {sim_names[i]}: {exc}")
            continue

        for hemisphere, lat_bounds in hemisphere_lat_bounds.items():
            try:
                eof, index, variance_fraction = leading_annular_mode(zm_u, hemisphere, lat_bounds)
                ac = autocorrelation_timeseries(index)
                tau = efolding_time(ac)
                step_days = time_step_days(index)

                annular_results[(i, hemisphere)] = {
                    "eof": eof,
                    "index": index,
                    "autocorrelation": ac,
                    "efolding_time_days": float(tau.values),
                    "variance_fraction": variance_fraction,
                }

                annular_summary_rows.append(
                    {
                        "simulation": sim_names[i],
                        "hemisphere": hemisphere,
                        "variance_fraction": variance_fraction,
                        "efolding_time_days": float(tau.values),
                        "autocorrelation_10d": ac_at_lag(ac, 10.0),
                        "autocorrelation_30d": ac_at_lag(ac, 30.0),
                        "fraction_abs_index_gt_1": float(np.nanmean(np.abs(index.values) > 1.0)),
                        "longest_same_sign_run_days": longest_same_sign_run_days(index.values, step_days),
                    }
                )

                tdays = time_days_since_start(index)
                for time_value, time_day, index_value in zip(index.Time.values, tdays, index.values):
                    annular_index_rows.append(
                        {
                            "simulation": sim_names[i],
                            "hemisphere": hemisphere,
                            "Time": time_value,
                            "time_days": time_day,
                            "zonal_index": index_value,
                        }
                    )
            except Exception as exc:
                print(f"Skipping {hemisphere} annular-mode diagnostics for {sim_names[i]}: {exc}")

    if save and annular_results:
        members = list(annular_results)
        mode_member = xr.IndexVariable("mode_member", np.arange(len(members)))
        member_simulations = [sim_names[i] for i, hemisphere in members]
        member_hemispheres = [hemisphere for i, hemisphere in members]

        pattern_da = xr.concat(
            [annular_results[member]["eof"].rename("eof") for member in members],
            dim=mode_member,
            join="outer",
        )
        pattern_ds = pattern_da.to_dataset()
        pattern_ds = pattern_ds.assign_coords(
            simulation=("mode_member", member_simulations),
            hemisphere=("mode_member", member_hemispheres),
        )
        save_dataset(pattern_ds, "annular_mode_eof_patterns.nc")

        index_da = xr.concat(
            [annular_results[member]["index"].rename("zonal_index") for member in members],
            dim=mode_member,
            join="outer",
        )
        autocorr_da = xr.concat(
            [annular_results[member]["autocorrelation"].rename("autocorrelation") for member in members],
            dim=mode_member,
            join="outer",
        )
        ac_ds = xr.Dataset({"zonal_index": index_da, "autocorrelation": autocorr_da})
        ac_ds = ac_ds.assign_coords(
            simulation=("mode_member", member_simulations),
            hemisphere=("mode_member", member_hemispheres),
        )
        save_dataset(ac_ds, "annular_mode_indices_autocorrelation.nc")

        pd.DataFrame(annular_summary_rows).to_csv(os.path.join(save_loc, "annular_mode_summary.csv"), index=False)
        pd.DataFrame(annular_index_rows).to_csv(os.path.join(save_loc, "annular_mode_zonal_indices.csv"), index=False)


jet_mean_profiles = {}
if should_run_figure(5):
    for i, ds in enumerate(dss):
        for plev in jet_profile_pressure_levels:
            try:
                jet_mean_profiles[(i, plev)] = time_mean_zonal_mean_u_at_level(ds, plev)
            except Exception as exc:
                print(f"Skipping {plev:g} hPa jet profile for {sim_names[i]}: {exc}")

    if save and jet_mean_profiles:
        members = list(jet_mean_profiles)
        profile_da = xr.concat(
            [jet_mean_profiles[member].rename("u_zonal_mean") for member in members],
            dim=xr.IndexVariable("profile_member", np.arange(len(members))),
            join="outer",
        )
        profile_ds = profile_da.to_dataset()
        profile_ds = profile_ds.assign_coords(
            simulation=("profile_member", [sim_names[i] for i, _ in members]),
            requested_pressure_hpa=("profile_member", [plev for _, plev in members]),
            selected_pressure_hpa=(
                "profile_member",
                [jet_mean_profiles[member].attrs["selected_pressure_hpa"] for member in members],
            ),
        )
        save_dataset(profile_ds, "u700_u200_time_mean_zonal_profiles.nc")


jet_structure_fields = {}
jet_core_lookup = {}
jet_core_rows = []
if should_run_figure(6):
    for i, zds in enumerate(dss_zm):
        try:
            field = model_height_time_mean_zonal_u(zds)
            jet_structure_fields[i] = field

            for hemisphere, bounds in subtropical_jet_lat_bounds.items():
                core = subtropical_jet_core(field, bounds)
                jet_core_lookup[(i, hemisphere)] = core
                if core is None:
                    continue
                control_core = jet_core_lookup.get((0, hemisphere))
                row = {
                    "simulation": sim_names[i],
                    "hemisphere": hemisphere,
                    "core_latitude": core["latitude"],
                    "core_height_km": core["height_m"] / 1000.0,
                    "core_u_ms-1": core["u_ms-1"],
                }
                if i == 0 or control_core is None:
                    row.update(
                        {
                            "delta_core_latitude_vs_control": 0.0 if i == 0 else np.nan,
                            "delta_core_height_km_vs_control": 0.0 if i == 0 else np.nan,
                            "delta_core_u_ms-1_vs_control": 0.0 if i == 0 else np.nan,
                        }
                    )
                else:
                    row.update(
                        {
                            "delta_core_latitude_vs_control": core["latitude"] - control_core["latitude"],
                            "delta_core_height_km_vs_control": (core["height_m"] - control_core["height_m"]) / 1000.0,
                            "delta_core_u_ms-1_vs_control": core["u_ms-1"] - control_core["u_ms-1"],
                        }
                    )
                jet_core_rows.append(row)
        except Exception as exc:
            print(f"Skipping model-height jet-structure diagnostics for {sim_names[i]}: {exc}")

    if save and jet_structure_fields:
        members = list(jet_structure_fields)
        structure_da = xr.concat(
            [jet_structure_fields[i].rename("u_zonal_mean") for i in members],
            dim=xr.IndexVariable("simulation", [sim_names[i] for i in members]),
            join="outer",
        )
        save_dataset(structure_da.to_dataset(), "subtropical_jet_latheight_structure.nc")

        if jet_core_rows:
            pd.DataFrame(jet_core_rows).to_csv(
                os.path.join(save_loc, "subtropical_jet_core_summary.csv"),
                index=False,
            )


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

if should_run_figure(1) and fixed_level:
    all_u = np.concatenate([np.ravel(v.values) for v in fixed_level.values()])
    vmax = np.nanpercentile(np.abs(all_u[np.isfinite(all_u)]), 98) if np.isfinite(all_u).any() else 1.0
    vmax = max(vmax, 1.0)
    levels = symmetric_levels(vmax, ndiv=23)

    fig, ax = plt.subplots(len(fixed_level), 1, figsize=(10, 2.8 * len(fixed_level)), sharex=True, sharey=True)
    ax = np.atleast_1d(ax)
    im = None
    for row, i in enumerate(fixed_level):
        u = fixed_level[i]
        tdays = time_days_since_start(u)
        im = ax[row].contourf(
            tdays,
            u.latitude,
            u.transpose("latitude", "Time").values,
            levels=levels,
            cmap=difference_cmap(levels, cmap="RdBu_r"),
            extend="both",
        )
        if np.nanmin(u.values) < 0.0 < np.nanmax(u.values):
            ax[row].contour(
                tdays,
                u.latitude,
                u.transpose("latitude", "Time").values,
                levels=[0.0],
                colors="0.15",
                linewidths=0.6,
            )
        for hemisphere, track in jet_latitudes.get(i, {}).items():
            ax[row].plot(tdays, track.values, color="w", lw=1.6)
            ax[row].plot(tdays, track.values, color="0.1", lw=0.7, label=hemisphere if row == 0 else None)
        ax[row].axhline(0, color="0.2", lw=0.6)
        ax[row].set_ylabel("Latitude")
        annotate_panel_label(ax[row], sim_names[i])
    ax[-1].set_xlabel("Time since analysis start (days)")
    fig.suptitle(f"Zonal-mean u at {fixed_pressure_level:g} hPa")
    if jet_latitudes:
        ax[0].legend(loc="upper right", frameon=True)
    tight_layout_with_right_margin(fig)
    if im is not None:
        add_right_margin_colorbar(fig, im, ax, label="m s$^{-1}$")
    figsave.save(fig, f"u{pressure_name(fixed_pressure_level)}_zonal_wind_hovmoller", number=1)
    print("Done with figure: fixed_level_zonal_wind_hovmoller")


if should_run_figure(2) and fixed_level_autocorr:
    fig, ax = plt.subplots(len(fixed_level_autocorr), 1, figsize=(9, 2.8 * len(fixed_level_autocorr)), sharex=True, sharey=True)
    ax = np.atleast_1d(ax)
    levels = np.linspace(-0.4, 1.0, 29)
    im = None
    for row, i in enumerate(fixed_level_autocorr):
        ac = fixed_level_autocorr[i]
        im = ax[row].contourf(
            ac.lag_days,
            ac.latitude,
            ac.transpose("latitude", "lag_days").values,
            levels=levels,
            cmap="RdBu_r",
            extend="both",
        )
        ax[row].contour(ac.lag_days, ac.latitude, ac.transpose("latitude", "lag_days").values,
                        levels=[autocorr_efold_threshold], colors="k", linewidths=1.0)
        tau = fixed_level_efold[i]
        ax[row].plot(tau.values, tau.latitude.values, color="0.05", lw=1.2)
        ax[row].set_ylabel("Latitude")
        annotate_panel_label(ax[row], sim_names[i])
    ax[-1].set_xlabel("Lag (days)")
    fig.suptitle(f"Autocorrelation of zonal-mean u anomalies at {fixed_pressure_level:g} hPa")
    tight_layout_with_right_margin(fig)
    if im is not None:
        add_right_margin_colorbar(fig, im, ax, label="correlation")
    figsave.save(fig, f"u{pressure_name(fixed_pressure_level)}_zonal_wind_autocorrelation", number=2)
    print("Done with figure: fixed_level_zonal_wind_autocorrelation")


if should_run_figure(3) and annular_results:
    hemispheres = list(hemisphere_lat_bounds)
    pattern_values = np.concatenate([np.ravel(result["eof"].values) for result in annular_results.values()])
    vmax = np.nanpercentile(np.abs(pattern_values[np.isfinite(pattern_values)]), 98) if np.isfinite(pattern_values).any() else 1.0
    vmax = max(vmax, 0.5)
    levels = symmetric_levels(vmax, ndiv=23)

    fig, ax = plt.subplots(len(dss), len(hemispheres), figsize=(4.4 * len(hemispheres), 2.8 * len(dss)), squeeze=False)
    im = None
    for row in range(len(dss)):
        for col, hemisphere in enumerate(hemispheres):
            axis = ax[row, col]
            result = annular_results.get((row, hemisphere))
            if result is None:
                axis.set_axis_off()
                continue
            eof = result["eof"]
            im = axis.contourf(
                eof.latitude,
                eof.pres,
                eof.values,
                levels=levels,
                cmap=difference_cmap(levels, cmap="RdBu_r"),
                extend="both",
            )
            axis.contour(eof.latitude, eof.pres, eof.values, levels=[0.0], colors="k", linewidths=0.8)
            set_latpres_axis(
                axis,
                xlabel=(row == len(dss) - 1),
                ylabel=(col == 0),
                xlim=hemisphere_lat_bounds[hemisphere],
                ylim=(max(eof_pressure_bounds), min(eof_pressure_bounds)),
            )
            annotate_panel_label(axis, f"{sim_names[row]} {hemisphere}")
            axis.set_title(f"EOF1 {100 * result['variance_fraction']:.1f}%")
    fig.suptitle("Hemispheric annular-mode EOFs of zonal-mean u")
    tight_layout_with_right_margin(fig)
    if im is not None:
        add_right_margin_colorbar(fig, im, ax, label="m s$^{-1}$ per index std. dev.")
    figsave.save(fig, "annular_mode_eof_patterns", number=3)
    print("Done with figure: annular_mode_eof_patterns")


if should_run_figure(4) and annular_results:
    hemispheres = list(hemisphere_lat_bounds)
    fig, ax = plt.subplots(2, len(hemispheres), figsize=(5.6 * len(hemispheres), 5.8), squeeze=False)
    for col, hemisphere in enumerate(hemispheres):
        for i in range(len(dss)):
            result = annular_results.get((i, hemisphere))
            if result is None:
                continue
            index = result["index"]
            ac = result["autocorrelation"]
            tdays = time_days_since_start(index)
            ax[0, col].plot(tdays, index.values, color=cols[i], lw=1.0, label=sim_names[i])
            ax[1, col].plot(ac.lag_days.values, ac.values, color=cols[i], lw=1.4)
        ax[0, col].axhline(0, color="0.3", lw=0.7)
        ax[0, col].set_title(f"{hemisphere} zonal index")
        if col == 0:
            ax[0, col].set_ylabel("Index")
        ax[1, col].axhline(0, color="0.3", lw=0.7)
        ax[1, col].axhline(autocorr_efold_threshold, color="0.3", lw=0.8, ls=":")
        ax[1, col].set_xlabel("Lag (days)")
        if col == 0:
            ax[1, col].set_ylabel("Autocorrelation")
        ax[1, col].set_ylim(-0.4, 1.05)
    ax[0, 0].legend(loc="upper right", frameon=False)
    fig.suptitle("Annular-mode zonal indices and persistence")
    tight_layout_with_right_margin(fig, right=0.98)
    figsave.save(fig, "annular_mode_indices_and_autocorrelation", number=4)
    print("Done with figure: annular_mode_indices_and_autocorrelation")


if should_run_figure(5) and jet_mean_profiles:
    fig, ax = plt.subplots(
        1,
        len(jet_profile_pressure_levels),
        figsize=(5.0 * len(jet_profile_pressure_levels), 3.8),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    ax = ax[0]

    for col, plev in enumerate(jet_profile_pressure_levels):
        axis = ax[col]
        selected_pressures = [
            jet_mean_profiles[(i, plev)].attrs["selected_pressure_hpa"]
            for i in range(len(dss))
            if (i, plev) in jet_mean_profiles
        ]
        title_pressure = np.nanmedian(selected_pressures) if selected_pressures else plev

        for i in range(len(dss)):
            profile = jet_mean_profiles.get((i, plev))
            if profile is None:
                continue
            axis.plot(profile.latitude.values, profile.values, color=cols[i], lw=1.5, label=sim_names[i])
            for bounds in hemisphere_jet_bounds.values():
                hemi = subset_between(profile, "latitude", bounds)
                if hemi.sizes.get("latitude", 0) == 0 or not np.isfinite(hemi.values).any():
                    continue
                jet_lat = float(hemi.idxmax(dim="latitude", skipna=True).values)
                jet_u = float(hemi.sel(latitude=jet_lat).values)
                axis.scatter(jet_lat, jet_u, color=cols[i], s=22, edgecolor="w", linewidth=0.5, zorder=3)

        axis.axhline(0, color="0.3", lw=0.7)
        axis.axvline(0, color="0.3", lw=0.7, ls=":")
        axis.set_title(f"{title_pressure:g} hPa")
        axis.set_xlabel("Latitude")
        axis.grid(True, lw=0.4, alpha=0.5)

    ax[0].set_ylabel(r"Zonal-mean u (m s$^{-1}$)")
    ax[0].legend(loc="best", frameon=False)
    fig.suptitle("Time-mean zonal-mean u")
    figsave.save(fig, "u700_u200_time_mean_zonal_profiles", number=5)
    print("Done with figure: time_mean_zonal_wind_profiles")


if should_run_figure(6) and jet_structure_fields:
    if 0 not in jet_structure_fields:
        print("Skipping subtropical jet latitude-height structure: control field is missing")
    else:
        control_field = jet_structure_fields[0]
        diff_fields = {}
        for i, field in jet_structure_fields.items():
            control_interp = control_field.interp(
                z=field.z,
                latitude=field.latitude,
                kwargs={"fill_value": np.nan},
            )
            diff_fields[i] = (field - control_interp).load()

        raw_levels = symmetric_percentile_levels_for_fields(
            list(jet_structure_fields.values()),
            percentile=99.0,
            ndiv=16,
            fallback=25.0,
        )
        diff_levels = symmetric_percentile_levels_for_fields(
            [field for i, field in diff_fields.items() if i != 0],
            percentile=99.0,
            ndiv=12,
            fallback=3.0,
        )

        ncols = len(dss_zm)
        fig, ax = plt.subplots(
            2,
            ncols,
            figsize=(3.9 * ncols, 6.2),
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )

        raw_im = diff_im = None
        for i in range(ncols):
            field = jet_structure_fields.get(i)
            if field is None:
                ax[0, i].set_visible(False)
                ax[1, i].set_visible(False)
                continue

            raw_im = ax[0, i].contourf(
                field.latitude.values,
                field.z.values / 1000.0,
                field.values,
                levels=raw_levels,
                cmap="RdBu_r",
                extend="both",
            )
            if np.nanmin(field.values) < 0.0 < np.nanmax(field.values):
                ax[0, i].contour(
                    field.latitude.values,
                    field.z.values / 1000.0,
                    field.values,
                    levels=[0.0],
                    colors="0.15",
                    linewidths=0.7,
                )

            diff = diff_fields[i]
            diff_im = ax[1, i].contourf(
                diff.latitude.values,
                diff.z.values / 1000.0,
                diff.values,
                levels=diff_levels,
                cmap=difference_cmap(diff_levels),
                extend="both",
            )
            if i != 0 and np.nanmin(diff.values) < 0.0 < np.nanmax(diff.values):
                ax[1, i].contour(
                    diff.latitude.values,
                    diff.z.values / 1000.0,
                    diff.values,
                    levels=[0.0],
                    colors="0.15",
                    linewidths=0.7,
                )

            for row in range(2):
                axis = ax[row, i]
                draw_subtropical_search_boxes(axis)
                annotate_panel_label(axis, sim_names[i] if row == 0 else control_diff_label(sim_names, i))
                set_latheight_labels(axis, xlabel=(row == 1), ylabel=(i == 0))
                axis.set_xlim(-70, 70)
                axis.set_ylim(
                    jet_structure_plot_height_bounds[0] / 1000.0,
                    jet_structure_plot_height_bounds[1] / 1000.0,
                )
                if i != 0:
                    axis.tick_params(labelleft=False)
                if row == 0:
                    axis.tick_params(labelbottom=False)

            for hemisphere in subtropical_jet_lat_bounds:
                run_core = jet_core_lookup.get((i, hemisphere))
                control_core = jet_core_lookup.get((0, hemisphere))
                plot_jet_core_marker(ax[0, i], run_core, color="gold", marker="o")

                if i == 0:
                    plot_jet_core_marker(ax[1, i], control_core, color="gold", marker="o")
                else:
                    plot_jet_core_marker(ax[1, i], control_core, color="0.05", marker="x")
                    plot_jet_core_marker(ax[1, i], run_core, color="gold", marker="o")
                    if control_core is not None and run_core is not None:
                        ax[1, i].plot(
                            [control_core["latitude"], run_core["latitude"]],
                            [control_core["height_m"] / 1000.0, run_core["height_m"] / 1000.0],
                            color="0.05",
                            lw=0.8,
                            alpha=0.9,
                            zorder=3,
                        )

        ax[0, 0].set_ylabel("Zonal wind\nHeight (km)")
        ax[1, 0].set_ylabel("Difference\nHeight (km)")
        if raw_im is not None:
            fig.colorbar(raw_im, ax=ax[0, :], location="right", label=r"u (m s$^{-1}$)")
        if diff_im is not None:
            add_symmetric_colorbar(
                fig,
                diff_im,
                ax[1, :],
                diff_levels,
                label=r"u difference from control (m s$^{-1}$)",
            )

        fig.suptitle("Subtropical jet latitude-height structure")
        figsave.save(fig, "subtropical_jet_latheight_structure", number=6)
        print("Done with figure: subtropical_jet_latheight_structure")
        print("WHAT TO LOOK FOR: top panels show each run's time-mean jet structure; bottom panels isolate nudged-control differences. Boxes define the subtropical jet-core search region, filled markers are each panel's core, and x markers in nudged panels show the control core.")
