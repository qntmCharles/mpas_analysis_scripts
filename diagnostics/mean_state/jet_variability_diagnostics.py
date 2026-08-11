#!/usr/bin/env python
# coding: utf-8

"""
Jet variability diagnostics for the 200-day MPAS aquaplanet integrations.

The script provides two complementary checks for jets becoming trapped in
particular configurations:

  * fixed-level zonal-mean wind, by default u at 250 hPa, with latitude-time
    wind fields, hemispheric jet-latitude tracks, and latitude-resolved
    autocorrelation of the wind fluctuations;
  * leading EOFs of zonal-mean u in each hemisphere, interpreted as annular-mode
    zonal indices, with EOF patterns, index time series, and autocorrelation.

Time-mean jet profiles and jet-core structure belong to
``circulation_diagnostics.py``.

The default configuration mirrors the other diagnostics in this directory.
"""

import argparse
import glob
import os
import sys
from functools import partial

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

from diagnostics.diagnostic_helpers import (
    FigureSaver, LATLON_PATTERN,
    annotate_panel_label, cache_manifest, cache_manifest_matches,
    diagnostic_cache_dir, diagnostic_output_dir,
    difference_cmap, figure_should_run, figure_should_run_any,
    latitude_weights, load_latlon_dataset,
    pressure_layer_weights, safe_name, set_latpres_axis,
    simulation_colour, standardize_latlon_dataset, subset_between, symmetric_levels,
    time_step_days, read_json_file,
    write_json_file,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=range(1, 5),
        default=None,
        help="Build only these numbered figures.",
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--figures-only",
        action="store_true",
        help="Build figures only from current compact jet caches.",
    )
    cache_mode.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild selected compact jet caches before plotting.",
    )
    return parser.parse_args(argv)


runtime_args = parse_args()
save = True
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/jet_variability/")

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: fixed_level_zonal_wind_hovmoller
#   2: fixed_level_zonal_wind_autocorrelation
#   3: annular_mode_eof_patterns
#   4: annular_mode_indices_and_autocorrelation
#
# WHAT TO LOOK FOR:
#   * Long jet autocorrelation or persistent jet-latitude anomalies indicate
#     slow internal variability that can contaminate small nudged-control
#     differences.
#   * EOF/annular-mode patterns identify whether differences are broad
#     hemispheric circulation shifts or local tropical/UTLS responses.
#   * If U-nudged changes jet variability without a matching T-nudged change,
#     that supports a wind-driven or shear-mediated pathway.
figures_to_run = runtime_args.figures or []
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)


should_run_figure = partial(figure_should_run, figures_to_run)
should_run_any = partial(figure_should_run_any, figures_to_run)


root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control_new", "u_nudge", "T_nudge"]
sim_names = ["control", "nudged u", "nudged T"]
cols = [simulation_colour(name) for name in sim_names]


fixed_pressure_level = 250.0
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
eof_pressure_bounds = (1000.0, 100.0)
eof_min_valid_fraction = 0.95

latlon_chunks = {"Time": -1, "latitude": 60, "longitude": 120}
needed_vars = ["uzonal_isobaric"]

dask.config.set({"array.slicing.split_large_chunks": True})

JET_CACHE_ROOT = diagnostic_cache_dir("jet_variability_cache")
JET_CACHE_SCHEMA = "jet-variability-v1"
if not runtime_args.figures_only:
    os.makedirs(JET_CACHE_ROOT, exist_ok=True)

JET_CACHE_FILES = {
    "fixed": (
        f"u{safe_name(f'{fixed_pressure_level:g}')}_zonal_mean_timeseries.nc",
        f"u{safe_name(f'{fixed_pressure_level:g}')}_zonal_mean_autocorrelation.nc",
        f"u{safe_name(f'{fixed_pressure_level:g}')}_persistence_summary.csv",
        f"u{safe_name(f'{fixed_pressure_level:g}')}_jet_latitude_summary.csv",
    ),
    "annular": (
        "annular_mode_eof_patterns.nc",
        "annular_mode_indices_autocorrelation.nc",
        "annular_mode_summary.csv",
        "annular_mode_zonal_indices.csv",
    ),
}


def requested_cache_families():
    selected = set(figures_to_run or range(1, 5))
    families = set()
    if selected & {1, 2}:
        families.add("fixed")
    if selected & {3, 4}:
        families.add("annular")
    return families


def jet_source_paths(family):
    return [
        path
        for data_loc in data_locs
        for path in sorted(glob.glob(os.path.join(root_dir, data_loc, LATLON_PATTERN)))
    ]


def jet_cache_manifest(family):
    configuration = {
        "family": family,
        "simulations": list(sim_names),
        "fixed_pressure_level_hpa": fixed_pressure_level,
        "time_average_days": time_average_days,
        "max_autocorr_lag_days": max_autocorr_lag_days,
        "hemisphere_lat_bounds": hemisphere_lat_bounds,
        "hemisphere_jet_bounds": hemisphere_jet_bounds,
        "eof_pressure_bounds_hpa": eof_pressure_bounds,
        "eof_min_valid_fraction": eof_min_valid_fraction,
    }
    return cache_manifest(
        "jet_variability_diagnostics.py",
        schema=JET_CACHE_SCHEMA,
        configuration=configuration,
        source_paths=jet_source_paths(family),
    )


def jet_cache_path(filename):
    return os.path.join(JET_CACHE_ROOT, filename)


def jet_manifest_path(family):
    return jet_cache_path(f"{family}.manifest.json")


def jet_cache_is_current(family):
    if runtime_args.rebuild_cache:
        return False
    if not all(os.path.isfile(jet_cache_path(name)) for name in JET_CACHE_FILES[family]):
        return False
    try:
        expected = jet_cache_manifest(family)
        saved = read_json_file(jet_manifest_path(family))
    except (OSError, ValueError):
        return False
    return cache_manifest_matches(saved, expected)


requested_families = requested_cache_families()
current_cache_families = {
    family for family in requested_families if jet_cache_is_current(family)
}
stale_cache_families = requested_families - current_cache_families
if runtime_args.figures_only and stale_cache_families:
    stale = ", ".join(sorted(stale_cache_families))
    raise RuntimeError(
        f"Figure-only mode requires current jet caches for: {stale}. "
        "Run without --figures-only to rebuild them."
    )


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

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


def time_days_since_start(da):
    times = da.Time.values
    if times.size == 0:
        return np.asarray([])
    if not np.issubdtype(times.dtype, np.datetime64):
        raise TypeError(f"Expected decoded datetime Time coordinate, got {times.dtype}")
    return np.asarray((times - times[0]) / np.timedelta64(1, "D"), dtype=float)


def time_average(da, days=time_average_days):
    """Coarsen to regular multi-sample means while keeping the original calendar."""
    if days is None or days <= 0:
        return da
    step_days = time_step_days(da)
    window = int(round(days / step_days))
    if window <= 1:
        return da
    return da.coarsen(Time=window, boundary="trim").mean()


def pressure_latitude_weights(field):
    pw, lw = xr.broadcast(pressure_layer_weights(field.pres), latitude_weights(field.latitude))
    return (pw * lw).transpose("pres", "latitude")


def weighted_pressure_mean(field):
    weights = pressure_layer_weights(field.pres)
    valid = np.isfinite(field)
    numerator = (field * weights).where(valid).sum(dim="pres", skipna=True)
    denominator = weights.where(valid).sum(dim="pres", skipna=True)
    return numerator / denominator


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
        path = jet_cache_path(filename)
        temporary = path + ".tmp"
        ds.to_netcdf(temporary)
        os.replace(temporary, path)


def save_table(table, filename):
    if save:
        path = jet_cache_path(filename)
        temporary = path + ".tmp"
        table.to_csv(temporary, index=False)
        os.replace(temporary, path)


# -----------------------------------------------------------------------------
# Data loading and wind diagnostics
# -----------------------------------------------------------------------------

dss = [None] * len(data_locs)
if stale_cache_families & {"fixed", "annular"}:
    for i, d in enumerate(data_locs):
        ds = load_latlon_dataset(
            os.path.join(root_dir, d),
            latlon_pattern=LATLON_PATTERN,
            chunks=latlon_chunks,
        )
        ds = standardize_latlon_dataset(ds, needed_vars)
        print(ds)
        dss[i] = ds

def zonal_mean_u_at_level(ds, plev=fixed_pressure_level):
    if "uzonal_isobaric" not in ds.data_vars:
        raise ValueError("uzonal_isobaric is required for fixed-level jet diagnostics")
    u = ds.uzonal_isobaric.sel(pres=plev)
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


def zonal_mean_u_profile(ds):
    if "uzonal_isobaric" not in ds.data_vars:
        raise ValueError("uzonal_isobaric is required for annular-mode diagnostics")
    u = ds.uzonal_isobaric
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

if "fixed" in current_cache_families:
    with xr.open_dataset(
        jet_cache_path(
            f"u{pressure_name(fixed_pressure_level)}_zonal_mean_timeseries.nc"
        )
    ) as cached:
        fixed_cache = cached.load()
    with xr.open_dataset(
        jet_cache_path(
            f"u{pressure_name(fixed_pressure_level)}_zonal_mean_autocorrelation.nc"
        )
    ) as cached:
        autocorr_cache = cached.load()
    for i, simulation in enumerate(sim_names):
        if simulation not in fixed_cache.simulation.values:
            continue
        fixed_level[i] = fixed_cache.u_zonal_mean.sel(simulation=simulation)
        fixed_level_anomaly[i] = fixed_cache.u_anomaly.sel(simulation=simulation)
        fixed_level_autocorr[i] = autocorr_cache.autocorrelation.sel(
            simulation=simulation
        )
        fixed_level_efold[i] = autocorr_cache.efolding_time_days.sel(
            simulation=simulation
        )
        jet_latitudes[i] = hemisphere_jet_latitudes(fixed_level[i])
    print(f"Using current fixed-level jet cache in {JET_CACHE_ROOT}")

if should_run_any([1, 2]) and "fixed" in stale_cache_families:
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

        save_table(
            pd.DataFrame(fixed_summary_rows),
            f"u{pressure_name(fixed_pressure_level)}_persistence_summary.csv",
        )
        save_table(
            pd.DataFrame(jet_summary_rows),
            f"u{pressure_name(fixed_pressure_level)}_jet_latitude_summary.csv",
        )
        write_json_file(jet_manifest_path("fixed"), jet_cache_manifest("fixed"))


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

    poleward_mean = poleward.weighted(
        latitude_weights(poleward.latitude)
    ).mean("latitude", skipna=True)
    equatorward_mean = equatorward.weighted(
        latitude_weights(equatorward.latitude)
    ).mean("latitude", skipna=True)
    contrast = float(poleward_mean.values - equatorward_mean.values)
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

if "annular" in current_cache_families:
    with xr.open_dataset(jet_cache_path("annular_mode_eof_patterns.nc")) as cached:
        pattern_cache = cached.load()
    with xr.open_dataset(
        jet_cache_path("annular_mode_indices_autocorrelation.nc")
    ) as cached:
        index_cache = cached.load()
    for member in pattern_cache.mode_member.values:
        pattern_member = pattern_cache.sel(mode_member=member)
        simulation = str(pattern_member.simulation.item())
        hemisphere = str(pattern_member.hemisphere.item())
        if simulation not in sim_names:
            continue
        i = sim_names.index(simulation)
        index_member = index_cache.sel(mode_member=member)
        eof = pattern_member.eof.drop_vars(
            ["mode_member", "simulation", "hemisphere"], errors="ignore"
        )
        index = index_member.zonal_index.drop_vars(
            ["mode_member", "simulation", "hemisphere"], errors="ignore"
        )
        autocorrelation = index_member.autocorrelation.drop_vars(
            ["mode_member", "simulation", "hemisphere"], errors="ignore"
        )
        annular_results[(i, hemisphere)] = {
            "eof": eof,
            "index": index,
            "autocorrelation": autocorrelation,
            "efolding_time_days": float(efolding_time(autocorrelation).values),
            "variance_fraction": float(eof.attrs["variance_fraction"]),
        }
    print(f"Using current annular-mode jet cache in {JET_CACHE_ROOT}")

if should_run_any([3, 4]) and "annular" in stale_cache_families:
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

        save_table(pd.DataFrame(annular_summary_rows), "annular_mode_summary.csv")
        save_table(
            pd.DataFrame(annular_index_rows),
            "annular_mode_zonal_indices.csv",
        )
        write_json_file(jet_manifest_path("annular"), jet_cache_manifest("annular"))


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
            cmap=difference_cmap(levels),
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
