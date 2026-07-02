#!/usr/bin/env python
# coding: utf-8

"""
Conditional convective/cloud pathway diagnostics for MPAS-A nudging experiments.

This script asks whether precipitation and high-cloud changes in the control,
u-nudged, and T-nudged experiments are better explained by:

  * local thermodynamic controls such as rain rate, PW, CAPE, CIN and latitude;
  * direct or balanced UTLS thermal/static-stability changes; or
  * wind/shear/wave effects, especially in organised equatorial disturbances.

The high-cloud diagnostics deliberately separate convective precipitating convection,
stratiform precipitating high cloud, and non-precipitating high cloud.  High cloud
with a high cloud top is not automatically interpreted as deep precipitating convection.
"""

import os
import sys
import warnings
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import dask
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from diagnostic_helpers import *
from high_cloud_classification import (
    HighCloudThresholds,
    cached_classification_is_current as shared_cached_classification_is_current,
    classification_cache_metadata as shared_classification_cache_metadata,
)

try:
    from waveFilter import kf_filter
except ModuleNotFoundError:
    kf_filter = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/convective_pathways/")

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: conditional_highcloud_vs_precip
#   2: conditional_cloudtop_pressure_vs_precip_pw
#   3: conditional_precip_pw_by_wave_amplitude_and_phase
#   4: residual_highcloud_thermo_controls
#   5: cloud_classification_summary
#   6: cloud_classification_mean_maps
#   7: cloud_classification_snapshots
#   8: conditional_nonprecipitating_highcloud_pw_by_lagged_precip
#   9: wave_conditioned_cloud_classes_and_cloud_top_pressure
figures_to_run = []
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control_new", "u_nudge", "T_nudge"]
latlon_file = ["latlon.2000-*.nc", "latlon.2000-*.nc", "latlon.2000-*.nc"]
zm_latlon_file = ["zm_latlon.2000-*.nc", "zm_latlon.2000-*.nc", "zm_latlon.2000-*.nc"]
sim_names = ["control", "nudged u", "nudged T"]
cols = ["k", "b", "r"]

ndays = 200
dt = 6
analysis_end_date = "2000-11-21"
time_format = "time"
obs_per_day = int(round(24 / dt))

TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
WAVE_FILTER_LAT_BOUNDS = (-10.0, 10.0)

# High-cloud classification thresholds.  Keep these near the top so threshold
# sensitivity tests do not require editing the classification logic.
CF_THRESH = 0.05
P_HIGH_HPA = 440.0
P_PRECIP_MIN = 1.0
CONVECTIVE_FRACTION = 0.50
LAGGED_PRECIP_TIME_STEPS = 1

# Conditional sampling and binning controls.  These diagnostics use high-time
# gridded fields, but they stride before materialising values to keep memory
# predictable.
MIN_BIN_COUNT = 50
PRECIP_NBINS = 14
PW_NBINS = 14
LATITUDE_BIN_WIDTH = 2.0
LOW_COUNT_MARKER_Y = 0.03

PLOT_SAMPLE_LIMITS = dict(max_time=96, max_lat=96, max_lon=192)
RESIDUAL_FIT_SAMPLE_LIMITS = dict(max_time=120, max_lat=120, max_lon=240)
RESIDUAL_PLOT_SAMPLE_LIMITS = dict(max_time=120, max_lat=120, max_lon=240)

# This diagnostic consumes the shared high-cloud classification written by
# refined_high_cloud_classification_diagnostics.py. Missing or stale caches are
# fatal so all analyses use exactly the same physical classes.
classification_cache_dir = diagnostic_cache_dir("classification_cache", root_dir=root_dir)
classification_cache_read_dirs = unique_existing_order([classification_cache_dir])

cache_dir = diagnostic_cache_dir("convective_pathways_cache", root_dir=root_dir)
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(classification_cache_dir, exist_ok=True)

# Robust variable aliases.  The first available variable in each list is used.
VAR_ALIASES = {
    "precip": ["precip", "precip_rate", "pr", "rain_rate"],
    "rainc": ["rainc"],
    "rainnc": ["rainnc"],
    "i_rainc": ["i_rainc"],
    "i_rainnc": ["i_rainnc"],
    "olr": ["olr", "olrtoa", "OLR"],
    "pw": ["pw", "precipw", "prw"],
    "cape": ["cape", "CAPE"],
    "cin": ["cin", "CIN"],
    "theta": ["theta", "potential_temperature"],
    "pressure": ["pressure", "p"],
    "zgrid": ["zgrid"],
    "u_zm": ["uReconstructZonal", "u", "uzonal"],
    "u_isobaric": ["uzonal_isobaric", "u_isobaric"],
    "N2": ["N2", "n2", "static_stability"],
    "dudz": ["dudz", "du_dz", "vertical_shear"],
}

needed_vars = sorted(
    set(
        VAR_ALIASES["rainc"]
        + VAR_ALIASES["rainnc"]
        + VAR_ALIASES["i_rainc"]
        + VAR_ALIASES["i_rainnc"]
        + VAR_ALIASES["precip"]
        + VAR_ALIASES["olr"]
        + VAR_ALIASES["pw"]
        + VAR_ALIASES["cape"]
        + VAR_ALIASES["cin"]
        + VAR_ALIASES["theta"]
        + VAR_ALIASES["pressure"]
        + VAR_ALIASES["zgrid"]
        + VAR_ALIASES["u_zm"]
        + VAR_ALIASES["u_isobaric"]
        + VAR_ALIASES["N2"]
        + VAR_ALIASES["dudz"]
    )
)

latlon_chunks = {"Time": 8, "latitude": 90, "longitude": 180}
zm_chunks = {"Time": -1}

dask.config.set({"array.slicing.split_large_chunks": True})


# These wave boxes intentionally match the names and Kelvin definition used in
# equatorial_wave_diagnostics.py.  The low-frequency eastward box is configurable
# here because no import-safe project-level MJO helper exists; it is labelled as
# "MJO-like/eastward LF" rather than treated as proof of an aquaplanet MJO.
MIS = -999.0
WAVE_BOXES = {
    "Kelvin": dict(
        k=(1, 14), period=(2.5, 20.0), depth=(8, 90),
        label="Kelvin", filter_name="Kelvin",
    ),
    "MJO_like": dict(
        k=(1, 5), period=(30.0, 96.0), depth=(MIS, MIS),
        label="MJO-like/eastward LF", filter_name="None",
    ),
}
WAVE_BANDS_FOR_FIG3 = ["Kelvin", "MJO_like"]
WAVE_FILTER_FIELD = "olr"  # falls back to precipitation if OLR is unavailable
WAVE_PHASE_THRESHOLD_STD = 0.5

CLOUD_CATEGORY_SPECS = [
    ("high_cloud_mask", "Total high cloud", "0.25"),
    ("convective_precipitating_high_cloud_mask", "Convective precipitating", "tab:red"),
    ("stratiform_precipitating_high_cloud_mask", "Stratiform precipitating", "tab:blue"),
    ("nonprecipitating_high_cloud_mask", "Non-precipitating", "tab:green"),
    ("unclassified_high_cloud_mask", "Invalid/unclassified", "0.55"),
]

MAP_CATEGORY_SPECS = [
    ("high_cloud_mask", "Total high cloud"),
    ("convective_precipitating_high_cloud_mask", "Convective precipitating"),
    ("stratiform_precipitating_high_cloud_mask", "Stratiform precipitating"),
    ("nonprecipitating_high_cloud_mask", "Non-precipitating"),
]


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


def run_all_figures():
    return all_figures_requested(figures_to_run)


def save_figure(number, fig, stem):
    return figsave.save(fig, stem, number=number)


def progress(message):
    """Print timestamped progress; useful when stdout is captured to a log."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def write_dataset_with_progress(ds, path, label, encoding=None):
    """Write a NetCDF file through a temporary path with optional dask progress."""
    progress(f"{label}: writing {path}")
    tmp_path = path + ".tmp"
    progress_bar = None
    if dask is not None:
        try:
            from dask.diagnostics import ProgressBar
            progress_bar = ProgressBar()
        except Exception:
            progress_bar = None

    if progress_bar is None:
        ds.to_netcdf(tmp_path, encoding=encoding)
    else:
        with progress_bar:
            ds.to_netcdf(tmp_path, encoding=encoding)
    os.replace(tmp_path, path)
    progress(f"{label}: finished writing {path}")


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def find_var(ds, key, required=False):
    for name in VAR_ALIASES.get(key, [key]):
        if name in ds.data_vars:
            return name
    if required:
        raise KeyError(f"Could not find variable for {key!r}; tried {VAR_ALIASES.get(key, [key])}")
    return None


def subset_lat(da, lat_bounds=TROPICAL_LAT_BOUNDS):
    if lat_bounds is None or "latitude" not in da.dims:
        return da
    lo, hi = sorted(lat_bounds)
    return da.sel(latitude=slice(lo, hi))


def latitude_weight_array(da):
    if "latitude" not in da.dims:
        return xr.ones_like(da)
    w = xr.DataArray(
        np.cos(np.deg2rad(da.latitude.values)),
        dims=("latitude",),
        coords={"latitude": da.latitude},
        name="latitude_weight",
    )
    return xr.broadcast(w, da)[0]


def weighted_area_time_mean(da, lat_bounds=TROPICAL_LAT_BOUNDS):
    x = subset_lat(da.astype(float), lat_bounds=lat_bounds)
    dims = [d for d in ("Time", "latitude", "longitude") if d in x.dims]
    weights = np.cos(np.deg2rad(x.latitude)) if "latitude" in x.dims else None
    if weights is None:
        return float(x.mean(dim=dims, skipna=True).values)
    return float(x.weighted(weights).mean(dim=dims, skipna=True).values)


def weighted_lat_mean(da, lat_bounds=WAVE_FILTER_LAT_BOUNDS):
    x = subset_lat(da, lat_bounds=lat_bounds)
    if "latitude" not in x.dims:
        return x
    return lat_mean(x)


def get_precip_rate(ds):
    direct = find_var(ds, "precip")
    if direct is not None:
        pr = finite_or_nan(ds[direct])
        pr.name = "precip"
        return pr

    rainc = find_var(ds, "rainc", required=True)
    rainnc = find_var(ds, "rainnc", required=True)
    i_rainc = find_var(ds, "i_rainc", required=True)
    i_rainnc = find_var(ds, "i_rainnc", required=True)
    accum = (
        ds[rainnc] + ds[i_rainnc] * 100.0
        + ds[rainc] + ds[i_rainc] * 100.0
    )
    pr = precip_rate_from_accum(accum, dt=dt)
    pr.name = "precip"
    pr.attrs["units"] = "mm day-1"
    return finite_or_nan(pr)


def time_aligned_field(ds, key, ref, required=False):
    name = find_var(ds, key, required=required)
    if name is None:
        return None
    out = finite_or_nan(ds[name])
    if "Time" in out.dims and "Time" in ref.dims:
        out = out.sel(Time=ref.Time)
    out.name = key
    return out


def classification_thresholds():
    return HighCloudThresholds(
        cf_thresh=CF_THRESH,
        high_cloud_top_pressure_hpa=P_HIGH_HPA,
        surface_precip_min_mm_day=P_PRECIP_MIN,
        convective_fraction_threshold=CONVECTIVE_FRACTION,
    )


def classification_cache_metadata():
    return shared_classification_cache_metadata(classification_thresholds())


def cached_classification_is_current(cached):
    return shared_cached_classification_is_current(cached, classification_thresholds())


REQUIRED_CLASSIFICATION_VARS = (
    "cloud_top_pressure",
    "cloud_top_height",
    "high_cloud_fraction",
    "high_cloud_mask",
    "precipitating_high_cloud_mask",
    "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
    "unclassified_high_cloud_mask",
)


def open_cached_classification(cache_path, sim_name):
    cached = xr.open_dataset(cache_path, chunks={})
    if not cached_classification_is_current(cached):
        cached.close()
        raise RuntimeError(f"{sim_name}: stale high-cloud classification cache: {cache_path}")
    missing = [name for name in REQUIRED_CLASSIFICATION_VARS if name not in cached.data_vars]
    if missing:
        cached.close()
        raise RuntimeError(
            f"{sim_name}: high-cloud classification cache lacks required shared variables "
            f"{missing}: {cache_path}"
        )
    progress(f"{sim_name}: using cached high-cloud classification {cache_path}")
    return {name: cached[name] for name in cached.data_vars}


def load_required_classification(ds, fields, sim_name):
    cache_filename = f"{sim_name.replace(' ', '_')}_classified_high_cloud.nc"
    checked_paths = []
    for read_dir in classification_cache_read_dirs:
        read_path = os.path.join(read_dir, cache_filename)
        checked_paths.append(read_path)
        if os.path.exists(read_path):
            return open_cached_classification(read_path, sim_name)
    raise FileNotFoundError(
        f"{sim_name}: required shared high-cloud classification cache is missing. "
        f"Run refined_high_cloud_classification_diagnostics.py first. Checked: {checked_paths}"
    )


def print_class_coverage(classifications):
    progress("Computing high-cloud class tropical coverage")
    rows = []
    for sim_name, classif in classifications.items():
        if classif is None:
            continue
        for key in (
            "high_cloud_mask",
            "convective_precipitating_high_cloud_mask",
            "stratiform_precipitating_high_cloud_mask",
            "nonprecipitating_high_cloud_mask",
            "unclassified_high_cloud_mask",
        ):
            rows.append(
                dict(
                    simulation=sim_name,
                    class_name=key,
                    tropical_area_time_fraction=weighted_area_time_mean(classif[key], TROPICAL_LAT_BOUNDS),
                )
            )

    coverage = pd.DataFrame(rows)
    if len(coverage) == 0:
        return coverage

    print("\nHigh-cloud class tropical area/time coverage")
    print(coverage.to_string(index=False), flush=True)
    if save:
        coverage.to_csv(os.path.join(save_loc, "high_cloud_class_coverage.csv"), index=False)
        progress(f"Wrote {os.path.join(save_loc, 'high_cloud_class_coverage.csv')}")

    for sim_name in coverage.simulation.unique():
        sub = coverage[coverage.simulation == sim_name].set_index("class_name")
        high = sub.loc["high_cloud_mask", "tropical_area_time_fraction"]
        if high <= 0:
            continue
        class_fracs = sub.drop(index="high_cloud_mask")["tropical_area_time_fraction"] / high
        if class_fracs.max() > 0.85:
            dominant = class_fracs.idxmax()
            warnings.warn(
                f"{sim_name}: high-cloud classification is dominated by {dominant} "
                f"({100.0 * class_fracs.max():.1f}% of high-cloud coverage)."
            )
    return coverage


def class_overlap_sanity(classifications):
    progress("Computing high-cloud class overlap/denominator sanity checks")
    rows = []
    exclusive_keys = [
        "convective_precipitating_high_cloud_mask",
        "stratiform_precipitating_high_cloud_mask",
        "nonprecipitating_high_cloud_mask",
        "unclassified_high_cloud_mask",
    ]
    for sim_name, classif in classifications.items():
        if classif is None:
            continue
        high = classif["high_cloud_mask"].astype(bool)
        high_frac = weighted_area_time_mean(high, TROPICAL_LAT_BOUNDS)
        rows.append(dict(simulation=sim_name, metric="high_cloud_denominator", value=high_frac))

        for key in exclusive_keys + ["precipitating_high_cloud_mask"]:
            mask = classif[key].astype(bool)
            rows.append(dict(simulation=sim_name, metric=f"{key}_coverage", value=weighted_area_time_mean(mask, TROPICAL_LAT_BOUNDS)))
            rows.append(dict(simulation=sim_name, metric=f"{key}_within_high_cloud", value=weighted_area_time_mean(mask & high, TROPICAL_LAT_BOUNDS) / high_frac if high_frac > 0 else np.nan))

        for i, key_a in enumerate(exclusive_keys):
            for key_b in exclusive_keys[i + 1:]:
                overlap = weighted_area_time_mean(classif[key_a].astype(bool) & classif[key_b].astype(bool), TROPICAL_LAT_BOUNDS)
                rows.append(dict(simulation=sim_name, metric=f"overlap_{key_a}__{key_b}", value=overlap))

    df = pd.DataFrame(rows)
    if len(df) > 0:
        print("\nHigh-cloud class denominator/overlap sanity checks")
        print(df.to_string(index=False), flush=True)
        if save:
            path = os.path.join(save_loc, "high_cloud_classification_sanity.csv")
            df.to_csv(path, index=False)
            progress(f"Wrote {path}")
    return df


def zonal_occurrence(da):
    x = subset_lat(da.astype(float), TROPICAL_LAT_BOUNDS)
    dims = [dim for dim in ("Time", "longitude") if dim in x.dims]
    return x.mean(dim=dims, skipna=True)


def time_mean_map(da):
    x = subset_lat(da.astype(float), TROPICAL_LAT_BOUNDS)
    if "Time" in x.dims:
        x = x.mean(dim="Time", skipna=True)
    return x


def representative_snapshot_time(classif):
    high = subset_lat(classif["high_cloud_mask"].astype(float), TROPICAL_LAT_BOUNDS)
    if "Time" not in high.dims:
        return None
    weights = np.cos(np.deg2rad(high.latitude))
    coverage = high.weighted(weights).mean(dim=("latitude", "longitude"), skipna=True)
    idx = int(coverage.argmax(dim="Time").values)
    return high.Time.isel(Time=idx).values


def compute_ttl_diagnostics(zds, precip_ref):
    """Return optional N2_TTL and dudz_TTL as Time/latitude fields."""
    out = {}
    z_coord = zds["z"] if "z" in zds.coords else None
    if z_coord is None:
        return out

    zvals = np.asarray(z_coord.values, dtype=float)
    zkm = zvals / 1000.0 if np.nanmax(np.abs(zvals)) > 100.0 else zvals
    ttl_mask = xr.DataArray((zkm >= 14.0) & (zkm <= 18.0), dims=("z",), coords={"z": z_coord})

    n2_name = find_var(zds, "N2")
    if n2_name is not None:
        n2 = finite_or_nan(zds[n2_name])
        if vertical_dim(n2) != "z":
            n2 = n2.rename({vertical_dim(n2): "z"})
        out["N2_TTL"] = n2.where(ttl_mask, drop=True).mean(dim="z", skipna=True)
    else:
        theta_name = find_var(zds, "theta")
        if theta_name is not None:
            theta = finite_or_nan(zds[theta_name])
            if vertical_dim(theta) != "z":
                theta = theta.rename({vertical_dim(theta): "z"})
            try:
                dtheta_dz = theta.differentiate("z")
                n2 = 9.80665 / theta * dtheta_dz
                out["N2_TTL"] = n2.where(ttl_mask, drop=True).mean(dim="z", skipna=True)
            except Exception as exc:
                warnings.warn(f"Could not compute N2_TTL from theta: {exc}")

    dudz_name = find_var(zds, "dudz")
    if dudz_name is not None:
        dudz = finite_or_nan(zds[dudz_name])
        if vertical_dim(dudz) != "z":
            dudz = dudz.rename({vertical_dim(dudz): "z"})
        out["dudz_TTL"] = dudz.where(ttl_mask, drop=True).mean(dim="z", skipna=True)
    else:
        u_name = find_var(zds, "u_zm")
        if u_name is not None:
            u = finite_or_nan(zds[u_name])
            if vertical_dim(u) != "z":
                u = u.rename({vertical_dim(u): "z"})
            try:
                dudz = u.differentiate("z")
                out["dudz_TTL"] = dudz.where(ttl_mask, drop=True).mean(dim="z", skipna=True)
            except Exception as exc:
                warnings.warn(f"Could not compute dudz_TTL from uReconstructZonal: {exc}")

    aligned = {}
    for key, da in out.items():
        if "Time" in da.dims:
            da = da.sel(Time=precip_ref.Time)
        da.name = key
        aligned[key] = da
    return aligned


def latitude_field_like(ref):
    lat = xr.DataArray(ref.latitude.values, dims=("latitude",), coords={"latitude": ref.latitude}, name="latitude")
    return xr.broadcast(lat, ref)[0]


def sample_matching_arrays(
    *arrays,
    lat_bounds=TROPICAL_LAT_BOUNDS,
    max_time=96,
    max_lat=96,
    max_lon=192,
    return_weights=True,
):
    """Align, subset, stride and flatten matching xarray arrays."""
    prepared = []
    for da in arrays:
        x = finite_or_nan(da.astype(float)) if hasattr(da, "astype") else da
        x = subset_lat(x, lat_bounds=lat_bounds)
        prepared.append(x)
    prepared = list(xr.align(*prepared, join="inner"))
    indexers = {}
    limits = {"Time": max_time, "latitude": max_lat, "longitude": max_lon}
    for dim, max_n in limits.items():
        if max_n is None:
            continue
        size = max((da.sizes[dim] for da in prepared if dim in da.dims), default=0)
        if size > max_n:
            indexers[dim] = slice(0, None, int(np.ceil(size / max_n)))

    if indexers:
        prepared = [
            da.isel({dim: slc for dim, slc in indexers.items() if dim in da.dims})
            for da in prepared
        ]

    broadcasted = xr.broadcast(*prepared)
    vals = [np.asarray(da.values).ravel() for da in broadcasted]

    mask = np.ones(vals[0].shape, dtype=bool)
    for val in vals:
        mask &= np.isfinite(val)

    if return_weights:
        w_da = latitude_weight_array(broadcasted[0])
        weights = np.asarray(w_da.values).ravel()
        mask &= np.isfinite(weights) & (weights > 0)
        return [val[mask] for val in vals], weights[mask]
    return [val[mask] for val in vals]


def weighted_quantile(values, quantiles, weights=None):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    ok = np.isfinite(values)
    values = values[ok]
    if weights is None:
        if values.size == 0:
            return np.full_like(quantiles, np.nan, dtype=float)
        return np.nanpercentile(values, quantiles * 100.0)

    weights = np.asarray(weights, dtype=float)[ok]
    ok = weights > 0
    values = values[ok]
    weights = weights[ok]
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) - 0.5 * weights
    cdf /= np.sum(weights)
    return np.interp(quantiles, cdf, values)


def unique_monotonic_bins(edges, fallback=None):
    edges = np.asarray(edges, dtype=float)
    edges = edges[np.isfinite(edges)]
    edges = np.unique(edges)
    if edges.size < 3:
        if fallback is None:
            raise ValueError("Could not construct enough unique bin edges")
        return np.asarray(fallback, dtype=float)
    return edges


def fallback_bins_for_key(key, vals, nbins):
    if key == "precip":
        vmax = max(80.0, float(np.nanpercentile(vals, 99.5)))
        core = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
        return np.unique(np.concatenate([core[core < vmax], [vmax]]))
    if key == "pw":
        return np.linspace(float(np.nanpercentile(vals, 1.0)), float(np.nanpercentile(vals, 99.0)), nbins + 1)
    return np.linspace(float(np.nanmin(vals)), float(np.nanmax(vals)), nbins + 1)


def pooled_weighted_bins(field_dicts, key, nbins, lat_bounds=TROPICAL_LAT_BOUNDS, limits=PLOT_SAMPLE_LIMITS):
    vals_all = []
    weights_all = []
    for fields in field_dicts:
        if key not in fields:
            continue
        vals, weights = sample_matching_arrays(fields[key], lat_bounds=lat_bounds, **limits)
        vals_all.append(vals[0])
        weights_all.append(weights)
    if not vals_all:
        return None
    vals = np.concatenate(vals_all)
    weights = np.concatenate(weights_all)
    q = np.linspace(0.01, 0.99, nbins + 1)
    edges = weighted_quantile(vals, q, weights)
    return unique_monotonic_bins(edges, fallback=fallback_bins_for_key(key, vals, nbins))


def pooled_tercile_edges(field_dicts, key, lat_bounds=TROPICAL_LAT_BOUNDS, limits=PLOT_SAMPLE_LIMITS):
    vals_all = []
    weights_all = []
    for fields in field_dicts:
        if key not in fields:
            continue
        vals, weights = sample_matching_arrays(fields[key], lat_bounds=lat_bounds, **limits)
        vals_all.append(vals[0])
        weights_all.append(weights)
    if not vals_all:
        return None
    vals = np.concatenate(vals_all)
    weights = np.concatenate(weights_all)
    q = weighted_quantile(vals, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], weights)
    q[0] = np.nanmin(vals)
    q[-1] = np.nanmax(vals)
    return unique_monotonic_bins(q)


def pooled_classified_values(field_dicts, classifications, key, lat_bounds=TROPICAL_LAT_BOUNDS, limits=PLOT_SAMPLE_LIMITS):
    """
    Pool one predictor only over samples with classified cloud data.

    This keeps the control run on its shorter cloud-fraction record while still
    using the full available cloud record for the nudged simulations.
    """
    vals_all = []
    weights_all = []
    for sim_name, fields in zip(sim_names, field_dicts):
        classif = classifications.get(sim_name)
        if classif is None or key not in fields:
            continue
        vals, weights = sample_matching_arrays(
            fields[key],
            classif["high_cloud_mask"].astype(float),
            lat_bounds=lat_bounds,
            **limits,
        )
        vals_all.append(vals[0])
        weights_all.append(weights)
    if not vals_all:
        return None, None
    return np.concatenate(vals_all), np.concatenate(weights_all)


def pooled_weighted_bins_for_classified(field_dicts, classifications, key, nbins, lat_bounds=TROPICAL_LAT_BOUNDS, limits=PLOT_SAMPLE_LIMITS):
    vals, weights = pooled_classified_values(field_dicts, classifications, key, lat_bounds=lat_bounds, limits=limits)
    if vals is None:
        return None
    q = np.linspace(0.01, 0.99, nbins + 1)
    edges = weighted_quantile(vals, q, weights)
    return unique_monotonic_bins(edges, fallback=fallback_bins_for_key(key, vals, nbins))


def pooled_tercile_edges_for_classified(field_dicts, classifications, key, lat_bounds=TROPICAL_LAT_BOUNDS, limits=PLOT_SAMPLE_LIMITS):
    vals, weights = pooled_classified_values(field_dicts, classifications, key, lat_bounds=lat_bounds, limits=limits)
    if vals is None:
        return None
    q = weighted_quantile(vals, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], weights)
    q[0] = np.nanmin(vals)
    q[-1] = np.nanmax(vals)
    return unique_monotonic_bins(q)


def weighted_binned_mean(x, y, bins, weights=None, min_count=MIN_BIN_COUNT):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    x = x[ok]
    y = y[ok]
    weights = weights[ok]
    idx = np.digitize(x, bins) - 1
    centres = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(centres.size, np.nan)
    count = np.zeros(centres.size, dtype=int)
    for j in range(centres.size):
        m = idx == j
        count[j] = int(np.count_nonzero(m))
        if count[j] >= min_count and np.sum(weights[m]) > 0:
            mean[j] = np.sum(weights[m] * y[m]) / np.sum(weights[m])
    return centres, mean, count


def weighted_binned_fraction(x, numerator, bins, denominator=None, weights=None, min_count=MIN_BIN_COUNT):
    x = np.asarray(x, dtype=float)
    numerator = np.asarray(numerator, dtype=bool)
    if denominator is None:
        denominator = np.ones_like(numerator, dtype=bool)
    else:
        denominator = np.asarray(denominator, dtype=bool)
    weights = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(weights) & (weights > 0) & denominator
    x = x[ok]
    numerator = numerator[ok]
    weights = weights[ok]
    idx = np.digitize(x, bins) - 1
    centres = 0.5 * (bins[:-1] + bins[1:])
    frac = np.full(centres.size, np.nan)
    count = np.zeros(centres.size, dtype=int)
    for j in range(centres.size):
        m = idx == j
        count[j] = int(np.count_nonzero(m))
        if count[j] >= min_count and np.sum(weights[m]) > 0:
            frac[j] = np.sum(weights[m] * numerator[m].astype(float)) / np.sum(weights[m])
    return centres, frac, count


def plot_supported_curve(
    ax,
    centres,
    values,
    count,
    color,
    label=None,
    linestyle="-",
    marker="o",
    min_count=MIN_BIN_COUNT,
    low_count_marker=True,
    **kwargs,
):
    """Plot only sample-supported bins and mark bins that were present but undersampled."""
    centres = np.asarray(centres)
    values = np.asarray(values, dtype=float)
    count = np.asarray(count, dtype=int)
    supported = np.isfinite(values) & (count >= min_count)
    if np.any(supported):
        ax.plot(
            centres[supported],
            values[supported],
            color=color,
            ls=linestyle,
            marker=marker,
            ms=3,
            lw=1.6,
            label=label,
            **kwargs,
        )
    elif label is not None:
        # Keep the legend honest even if this run has no supported bins.
        ax.plot([], [], color=color, ls=linestyle, marker=marker, label=label, **kwargs)

    weak = (~supported) & (count > 0)
    if low_count_marker and np.any(weak):
        ax.scatter(
            centres[weak],
            np.full(np.count_nonzero(weak), LOW_COUNT_MARKER_Y),
            marker="|",
            s=22,
            color=color,
            alpha=0.35,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=2,
        )
    return supported


def add_low_count_note(ax, min_count=MIN_BIN_COUNT):
    ax.text(
        0.99,
        0.02,
        f"ticks: n<{min_count}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="0.35",
    )


def count_rows_to_frame(rows, stem):
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    print(f"\nSample counts for {stem}")
    print(df.to_string(index=False, max_rows=240))
    if save:
        df.to_csv(os.path.join(save_loc, f"{stem}_bin_counts.csv"), index=False)
    return df


def write_metadata_table():
    rows = [dict(threshold=name, value=value) for name, value in classification_cache_metadata().items()]
    df = pd.DataFrame(rows)
    if save:
        path = os.path.join(save_loc, "high_cloud_classification_thresholds.csv")
        df.to_csv(path, index=False)
        progress(f"Wrote {path}")
    return df


def condition_support_table(field_dicts, classifications, condition_key, edges, stem):
    """Count samples each simulation contributes to each conditioning bin."""
    rows = []
    labels = condition_labels(edges, condition_key)
    for sim_name, fields in zip(sim_names, field_dicts):
        classif = classifications.get(sim_name)
        if classif is None or condition_key not in fields:
            continue
        vals, weights = sample_matching_arrays(
            fields[condition_key],
            classif["high_cloud_mask"].astype(float),
            lat_bounds=TROPICAL_LAT_BOUNDS,
            **PLOT_SAMPLE_LIMITS,
        )
        cond = vals[0]
        for j, label in enumerate(labels):
            lo, hi = edges[j], edges[j + 1]
            mask = (cond >= lo) & (cond <= hi if j == len(labels) - 1 else cond < hi)
            rows.append(
                dict(
                    simulation=sim_name,
                    condition=condition_key,
                    bin_label=label,
                    bin_min=lo,
                    bin_max=hi,
                    sample_count=int(np.count_nonzero(mask)),
                    weighted_sample=float(np.sum(weights[mask])),
                )
            )
    df = pd.DataFrame(rows)
    if len(df) > 0:
        print(f"\nCondition-bin support for {stem}")
        print(df.to_string(index=False), flush=True)
        if save:
            path = os.path.join(save_loc, f"{stem}_condition_support.csv")
            df.to_csv(path, index=False)
            progress(f"Wrote {path}")
    return df


def common_support_condition_edges(field_dicts, classifications, key, nbins=3):
    """
    Prefer edges inside the range sampled by all simulations.

    For N2_TTL this avoids interpreting a nudged-only stability regime as a
    like-for-like comparison.  If the robust per-run ranges do not overlap, the
    caller gets pooled occupancy bins and should label the panels accordingly.
    """
    if key != "N2_TTL":
        return pooled_tercile_edges_for_classified(field_dicts, classifications, key), "pooled occupancy", None

    per_run = []
    for sim_name, fields in zip(sim_names, field_dicts):
        classif = classifications.get(sim_name)
        if classif is None or key not in fields:
            continue
        vals, weights = sample_matching_arrays(
            fields[key],
            classif["high_cloud_mask"].astype(float),
            lat_bounds=TROPICAL_LAT_BOUNDS,
            **PLOT_SAMPLE_LIMITS,
        )
        q02, q98 = weighted_quantile(vals[0], [0.02, 0.98], weights)
        per_run.append((sim_name, q02, q98))

    if len(per_run) == len(sim_names):
        common_lo = max(row[1] for row in per_run)
        common_hi = min(row[2] for row in per_run)
        if np.isfinite(common_lo) and np.isfinite(common_hi) and common_hi > common_lo:
            vals, weights = pooled_classified_values(field_dicts, classifications, key)
            if vals is not None:
                in_common = (vals >= common_lo) & (vals <= common_hi)
                if np.any(in_common):
                    q = weighted_quantile(vals[in_common], np.linspace(0, 1, nbins + 1), weights[in_common])
                    q[0], q[-1] = common_lo, common_hi
                    return unique_monotonic_bins(q), "common-support", (common_lo, common_hi, per_run)

    return pooled_tercile_edges_for_classified(field_dicts, classifications, key), "regime-occupancy", None


def condition_labels(edges, name):
    nbin = max(len(edges) - 1, 0)
    if nbin == 3:
        prefixes = ["low", "mid", "high"]
    else:
        prefixes = [f"bin {i + 1}" for i in range(nbin)]
    return [f"{prefix} {name}" for prefix in prefixes]


# -----------------------------------------------------------------------------
# Wave-filter helpers for Figure 3
# -----------------------------------------------------------------------------


def safe_name(x):
    return str(x).replace(" ", "_").replace("/", "-").replace(".", "p").replace("-", "m")


def wave_label(wave_name):
    return WAVE_BOXES[wave_name].get("label", wave_name)


def active_phase_sign(field_key):
    """Positive active_index means wave phase favours convection."""
    return -1.0 if field_key == "olr" else 1.0


def wave_cache_path(sim_index, field_key, wave_name):
    return os.path.join(
        cache_dir,
        f"{safe_name(sim_names[sim_index])}_{field_key}_{wave_name}_latmean_{WAVE_FILTER_LAT_BOUNDS[0]:g}_{WAVE_FILTER_LAT_BOUNDS[1]:g}.nc"
        .replace("-", "m"),
    )


def filter_latmean_wave(fields, sim_index, field_key, wave_name):
    if kf_filter is None:
        raise ModuleNotFoundError("waveFilter.kf_filter is unavailable")
    if field_key not in fields:
        raise ValueError(f"{field_key} is not available for wave filtering")

    cache_path = wave_cache_path(sim_index, field_key, wave_name)
    if os.path.exists(cache_path):
        progress(f"{sim_names[sim_index]} {wave_name}: using cached {field_key} wave filter {cache_path}")
        with xr.open_dataset(cache_path) as cached:
            return cached["wave"].load()

    box = WAVE_BOXES[wave_name]
    k0, k1 = box["k"]
    p0, p1 = box["period"]
    h0, h1 = box["depth"]
    filter_name = box.get("filter_name", wave_name)

    da = weighted_lat_mean(fields[field_key], lat_bounds=WAVE_FILTER_LAT_BOUNDS)
    da = da.transpose("Time", "longitude")
    progress(f"{sim_names[sim_index]} {wave_name}: filtering {field_key} over {WAVE_FILTER_LAT_BOUNDS}")
    obs_time = da.Time.load()
    lon = da.longitude.load()
    obs = np.asarray(da.load().values, dtype="f")

    filt_time = np.arange(0, obs.shape[0] + 702)
    filt_data = np.zeros((len(filt_time), obs.shape[1]), dtype="f")
    filt_data[: obs.shape[0], :] = obs

    wave_full = kf_filter(filt_data, obs_per_day, p0, p1, k0, k1, h0, h1, filter_name)
    wave = xr.DataArray(
        wave_full[: obs.shape[0], :],
        dims=("Time", "longitude"),
        coords={"Time": obs_time, "longitude": lon},
        name="wave",
        attrs=dict(
            field=field_key,
            wave_name=wave_name,
            filter_name=filter_name,
            units=fields[field_key].attrs.get("units", ""),
        ),
    )
    tmp_path = cache_path + ".tmp"
    wave.to_dataset().to_netcdf(tmp_path)
    os.replace(tmp_path, cache_path)
    progress(f"{sim_names[sim_index]} {wave_name}: wrote wave cache {cache_path}")
    return wave


def wave_conditioned_cloud_table(fields_by_run, classifications_by_run):
    rows = []
    class_specs = [
        ("convective_precipitating_high_cloud_mask", "Convective precipitating"),
        ("stratiform_precipitating_high_cloud_mask", "Stratiform precipitating"),
        ("nonprecipitating_high_cloud_mask", "Non-precipitating"),
    ]
    condition_specs = [
        ("suppressed", lambda phase: phase <= -WAVE_PHASE_THRESHOLD_STD),
        ("neutral", lambda phase: np.abs(phase) < WAVE_PHASE_THRESHOLD_STD),
        ("active", lambda phase: phase >= WAVE_PHASE_THRESHOLD_STD),
    ]
    for wave_name in WAVE_BANDS_FOR_FIG3:
        for i, (fields, sim_name) in enumerate(zip(fields_by_run, sim_names)):
            classif = classifications_by_run.get(sim_name)
            if classif is None:
                continue
            field_key = WAVE_FILTER_FIELD if WAVE_FILTER_FIELD in fields else "precip"
            wave = filter_latmean_wave(fields, i, field_key, wave_name)
            sigma = float(wave.std(dim=("Time", "longitude"), skipna=True).values)
            if not np.isfinite(sigma) or sigma <= 0.0:
                continue
            phase = active_phase_sign(field_key) * wave / sigma
            for condition, selector in condition_specs:
                condition_mask = selector(phase)
                for class_key, class_label in class_specs:
                    mask, condition_aligned = xr.align(
                        classif[class_key].astype(float), condition_mask, join="inner"
                    )
                    fraction = weighted_area_time_mean(mask.where(condition_aligned))
                    ctp, class_bool, condition_ctp = xr.align(
                        classif["cloud_top_pressure"],
                        classif[class_key].astype(bool),
                        condition_mask,
                        join="inner",
                    )
                    mean_ctp = weighted_area_time_mean(
                        ctp.where(class_bool & condition_ctp)
                    )
                    rows.append(
                        dict(
                            wave=wave_name,
                            wave_label=wave_label(wave_name),
                            simulation=sim_name,
                            condition=condition,
                            class_key=class_key,
                            class_label=class_label,
                            conditional_area_fraction=fraction,
                            mean_cloud_top_pressure_hpa=mean_ctp,
                            denominator="tropical area-time within wave condition",
                        )
                    )
    return pd.DataFrame(rows)


def plot_wave_conditioned_cloud_diagnostics(table):
    if len(table) == 0:
        return
    table.to_csv(
        os.path.join(save_loc, "wave_conditioned_cloud_classes_and_cloud_top_pressure.csv"),
        index=False,
    )
    conditions = ["suppressed", "neutral", "active"]
    classes = list(table.class_key.drop_duplicates())
    labels = {
        key: table.loc[table.class_key == key, "class_label"].iloc[0]
        for key in classes
    }
    x = np.arange(len(classes))
    width = 0.8 / len(sim_names)
    for wave_name in WAVE_BANDS_FOR_FIG3:
        wave_table = table[table.wave == wave_name]
        if len(wave_table) == 0:
            continue
        for metric, ylabel, stem, invert in (
            ("conditional_area_fraction", "Conditional tropical area-time fraction", "class_fraction", False),
            ("mean_cloud_top_pressure_hpa", "Mean cloud-top pressure (hPa)", "cloud_top_pressure", True),
        ):
            fig, axes = plt.subplots(
                1, len(conditions), figsize=(12.5, 3.8), constrained_layout=True, sharey=True
            )
            for c, condition in enumerate(conditions):
                ax = axes[c]
                sub = wave_table[wave_table.condition == condition]
                for i, sim_name in enumerate(sim_names):
                    values = []
                    for class_key in classes:
                        row = sub[(sub.simulation == sim_name) & (sub.class_key == class_key)]
                        values.append(float(row[metric].iloc[0]) if len(row) else np.nan)
                    ax.bar(
                        x + (i - (len(sim_names) - 1) / 2) * width,
                        values,
                        width,
                        color=cols[i],
                        label=sim_name if c == 0 else None,
                    )
                ax.set_title(condition.title())
                ax.set_xticks(x)
                ax.set_xticklabels([labels[key] for key in classes], rotation=25, ha="right")
                ax.grid(axis="y", alpha=0.2)
                if invert:
                    ax.invert_yaxis()
            axes[0].set_ylabel(ylabel)
            axes[0].legend(frameon=False)
            fig.suptitle(f"{wave_label(wave_name)}-conditioned high cloud")
            save_figure(9, fig, f"wave_conditioned_{stem}_{wave_name}")
            plt.close(fig)


# -----------------------------------------------------------------------------
# Residual model helpers
# -----------------------------------------------------------------------------


class SimpleRidgeRegressor:
    """Small weighted ridge fallback with linear, squared, and pair terms."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.mean_ = None
        self.scale_ = None
        self.beta_ = None

    def _design(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        if fit:
            self.mean_ = np.nanmean(X, axis=0)
            self.scale_ = np.nanstd(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
        Xn = (X - self.mean_) / self.scale_
        cols = [np.ones(Xn.shape[0]), Xn]
        cols.append(Xn ** 2)
        pair_cols = []
        for i in range(Xn.shape[1]):
            for j in range(i + 1, Xn.shape[1]):
                pair_cols.append(Xn[:, i] * Xn[:, j])
        if pair_cols:
            cols.append(np.column_stack(pair_cols))
        return np.column_stack(cols)

    def fit(self, X, y, sample_weight=None):
        D = self._design(X, fit=True)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        sw = np.sqrt(np.maximum(w, 0.0))
        Dw = D * sw[:, None]
        yw = y * sw
        penalty = self.alpha * np.eye(D.shape[1])
        penalty[0, 0] = 0.0
        self.beta_ = np.linalg.solve(Dw.T @ Dw + penalty, Dw.T @ yw)
        return self

    def predict(self, X):
        return self._design(X, fit=False) @ self.beta_


def fit_control_relationship(X, y, weights, target_label):
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    X = X[ok]
    y = y[ok]
    weights = weights[ok]
    print(f"Fitting control high-cloud model for {target_label}: {X.shape[0]} samples")

    try:
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
        print(f"  used HistGradientBoostingRegressor for {target_label}")
        return model
    except Exception as exc:
        print(f"  falling back to weighted ridge for {target_label}: {exc}")
        model = SimpleRidgeRegressor(alpha=1.0)
        return model.fit(X, y, sample_weight=weights)


def feature_arrays(fields, feature_keys):
    arrays = []
    for key in feature_keys:
        if key == "latitude":
            arrays.append(latitude_field_like(fields["precip"]))
        elif key == "precip_log":
            arrays.append(np.log1p(fields["precip"].clip(min=0.0)))
        else:
            arrays.append(fields[key])
    return arrays


def feature_matrix_from_values(feature_keys, values):
    cols_out = []
    for key, val in zip(feature_keys, values):
        if key == "precip_log":
            cols_out.append(np.log1p(np.maximum(val, 0.0)))
        else:
            cols_out.append(val)
    return np.column_stack(cols_out)


def residual_target_da(classif, target_key):
    if target_key == "total":
        return classif["high_cloud_fraction"].clip(min=0.0, max=1.0)
    if target_key == "convective":
        return classif["convective_precipitating_high_cloud_mask"].astype(float)
    if target_key == "stratiform":
        return classif["stratiform_precipitating_high_cloud_mask"].astype(float)
    if target_key == "nonprecipitating":
        return classif["nonprecipitating_high_cloud_mask"].astype(float)
    raise ValueError(target_key)


def available_residual_features(fields_by_run):
    feature_keys = ["precip_log"]
    for key in ("pw", "cape", "cin"):
        if all(key in fields for fields in fields_by_run):
            feature_keys.append(key)
        else:
            warnings.warn(f"Residual model omits {key.upper()} because it is unavailable in at least one run.")
    feature_keys.append("latitude")
    return feature_keys


def model_training_data(fields, classif, target_key, feature_keys):
    target = residual_target_da(classif, target_key)
    arrays = feature_arrays(fields, feature_keys) + [target]
    vals, weights = sample_matching_arrays(
        *arrays,
        lat_bounds=TROPICAL_LAT_BOUNDS,
        **RESIDUAL_FIT_SAMPLE_LIMITS,
    )
    X = np.column_stack(vals[:-1])
    y = vals[-1]
    return X, y, weights


def residual_sample_data(fields, classif, target_key, feature_keys, bin_keys):
    target = residual_target_da(classif, target_key)
    bin_arrays = []
    for key in bin_keys:
        if key == "latitude":
            bin_arrays.append(latitude_field_like(fields["precip"]))
        else:
            bin_arrays.append(fields[key])
    arrays = feature_arrays(fields, feature_keys) + [target] + bin_arrays
    vals, weights = sample_matching_arrays(
        *arrays,
        lat_bounds=TROPICAL_LAT_BOUNDS,
        **RESIDUAL_PLOT_SAMPLE_LIMITS,
    )
    n_features = len(feature_keys)
    X = np.column_stack(vals[:n_features])
    y = vals[n_features]
    bins = {key: vals[n_features + 1 + j] for j, key in enumerate(bin_keys)}
    return X, y, bins, weights


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------


dss = []
dss_zm = []
progress("Loading lat-lon and zonal-mean datasets")
for i, d in enumerate(data_locs):
    progress(f"{sim_names[i]}: opening lat-lon dataset")
    ds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=latlon_file[i],
            time_format=time_format,
            ndays=ndays,
            end_date=analysis_end_date,
            chunks=latlon_chunks,
        )
    ds = standardize_latlon_dataset(ds, needed_vars)
    progress(f"{sim_names[i]}: lat-lon dataset ready with {len(ds.data_vars)} variables")

    progress(f"{sim_names[i]}: opening zonal-mean dataset")
    zds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=zm_latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks=zm_chunks,
        prefer_pattern=True,
    )
    zds = standardize_latlon_dataset(zds, needed_vars)
    progress(f"{sim_names[i]}: zonal-mean dataset ready with {len(zds.data_vars)} variables")

    print(ds)
    print(zds)
    dss.append(ds)
    dss_zm.append(zds)


fields_by_run = []
progress("Preparing precipitation, thermodynamic, and TTL fields")
for sim_name, ds, zds in zip(sim_names, dss, dss_zm):
    progress(f"{sim_name}: preparing fields")
    fields = {"precip": get_precip_rate(ds)}
    for key in ("olr", "pw", "cape", "cin"):
        da = time_aligned_field(ds, key, fields["precip"], required=False)
        if da is not None:
            fields[key] = da

    ttl_fields = compute_ttl_diagnostics(zds, fields["precip"])
    fields.update(ttl_fields)
    progress(f"{sim_name}: fields ready ({', '.join(sorted(fields))})")
    fields_by_run.append(fields)


classifications_by_run = {}
classification_coverage = pd.DataFrame()
classification_sanity = pd.DataFrame()
if should_run_any([1, 2, 4, 5, 6, 7, 8]):
    progress("Building or reading high-cloud classifications")
    for sim_name, ds, fields in zip(sim_names, dss, fields_by_run):
        classifications_by_run[sim_name] = load_required_classification(ds, fields, sim_name)
    write_metadata_table()
    classification_coverage = print_class_coverage(classifications_by_run)
    classification_sanity = class_overlap_sanity(classifications_by_run)
    progress("High-cloud classification stage complete")


# -----------------------------------------------------------------------------
# Figure 5. Cloud-classification summary and zonal occurrence
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * The bar plot shows tropical area-time occurrence; total high cloud is the
#     denominator for class-fraction diagnostics.
#   * The zonal curves show whether a class response is confined to the equator,
#     flanks, or broad tropical belt.
#   * Non-precipitating means high cloud with total rainc+rainnc below the shared
#     precipitation threshold; it is a precipitation regime, not a detailed
#     microphysical cloud-type claim.

if should_run_figure(5):
    progress("Starting Figure 5: cloud_classification_summary")
    if len(classification_coverage) == 0:
        print("Skipping Figure 5: classification coverage is unavailable.")
    else:
        fig, ax = plt.subplots(
            1, 1 + len(sim_names),
            figsize=(4.8 + 3.6 * len(sim_names), 4.6),
            constrained_layout=True,
            sharey=False,
        )
        class_keys = [key for key, _label, _color in CLOUD_CATEGORY_SPECS]
        class_labels = [label for _key, label, _color in CLOUD_CATEGORY_SPECS]
        class_colors = [color for _key, _label, color in CLOUD_CATEGORY_SPECS]
        x = np.arange(len(class_keys))
        width = 0.8 / len(sim_names)
        cov = classification_coverage.set_index(["simulation", "class_name"])

        for i, sim_name in enumerate(sim_names):
            vals = [
                cov.loc[(sim_name, key), "tropical_area_time_fraction"]
                if (sim_name, key) in cov.index else np.nan
                for key in class_keys
            ]
            ax[0].bar(
                x + (i - (len(sim_names) - 1) / 2) * width,
                vals,
                width,
                color=cols[i],
                label=sim_name,
            )
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(class_labels, rotation=35, ha="right")
        ax[0].set_ylabel("Tropical area-time fraction")
        ax[0].set_title("Class coverage\n(total high cloud is denominator)")
        ax[0].grid(axis="y", alpha=0.2)
        ax[0].legend(loc="best", fontsize=7)

        for i, sim_name in enumerate(sim_names):
            a = ax[i + 1]
            classif = classifications_by_run.get(sim_name)
            if classif is None:
                a.text(0.5, 0.5, "not available", transform=a.transAxes, ha="center", va="center")
                continue
            for key, label, color in CLOUD_CATEGORY_SPECS:
                try:
                    zocc = zonal_occurrence(classif[key])
                    a.plot(zocc.latitude, zocc, color=color, lw=1.4, label=label if i == 0 else None)
                except Exception as exc:
                    print(f"Skipping zonal occurrence {sim_name} {key}: {exc}")
            a.set_title(sim_name)
            a.set_xlabel("Latitude")
            a.set_xlim(*TROPICAL_LAT_BOUNDS)
            a.grid(alpha=0.2)
            if i == 0:
                a.set_ylabel("Occurrence")
                a.legend(loc="best", fontsize=6)
        fig.suptitle("High-cloud classification visibility and denominator check")
        save_figure(5, fig, "cloud_classification_summary")
        progress("Done with figure: cloud_classification_summary")


# -----------------------------------------------------------------------------
# Figure 6. Time-mean maps of selected cloud classes
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Spatially coherent class differences help distinguish cloud-type changes
#     from a few sparsely sampled conditional bins.
#   * The total high-cloud map is the denominator visibility check for the class
#     maps and fractions.

if should_run_figure(6):
    progress("Starting Figure 6: cloud_classification_mean_maps")
    if not any(classifications_by_run.get(sim_name) is not None for sim_name in sim_names):
        print("Skipping Figure 6: high-cloud classification is unavailable.")
    else:
        fig, ax = plt.subplots(
            len(sim_names), len(MAP_CATEGORY_SPECS),
            figsize=(3.4 * len(MAP_CATEGORY_SPECS), 2.8 * len(sim_names)),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        if len(sim_names) == 1:
            ax = ax[None, :]
        last_im = None
        levels = np.linspace(0, 0.5, 11)
        for i, sim_name in enumerate(sim_names):
            classif = classifications_by_run.get(sim_name)
            for j, (key, label) in enumerate(MAP_CATEGORY_SPECS):
                a = ax[i, j]
                if classif is None:
                    a.text(0.5, 0.5, "not available", transform=a.transAxes, ha="center", va="center")
                else:
                    try:
                        m = time_mean_map(classif[key])
                        last_im = a.contourf(
                            m.longitude,
                            m.latitude,
                            m,
                            levels=levels,
                            cmap="YlGnBu",
                            extend="max",
                        )
                    except Exception as exc:
                        a.text(0.5, 0.5, f"skipped\n{exc}", transform=a.transAxes, ha="center", va="center", fontsize=7)
                if i == 0:
                    a.set_title(label)
                if j == 0:
                    a.set_ylabel(f"{sim_name}\nLatitude")
                if i == len(sim_names) - 1:
                    a.set_xlabel("Longitude")
                a.set_xlim(-180, 180)
                a.set_ylim(*TROPICAL_LAT_BOUNDS)
        if last_im is not None:
            fig.colorbar(last_im, ax=ax.ravel().tolist(), label="Time-mean occurrence")
        fig.suptitle("Time-mean occurrence maps of high-cloud classes")
        save_figure(6, fig, "cloud_classification_mean_maps")
        progress("Done with figure: cloud_classification_mean_maps")


# -----------------------------------------------------------------------------
# Figure 7. Representative class snapshots
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Snapshots are a sanity check on whether the masks look meteorologically
#     sensible: convective precipitating convection should be compact, stratiform precipitating
#     cloud should lie near or downstream of active objects, and non-precipitating
#     masks should be separated by construction.

if should_run_figure(7):
    progress("Starting Figure 7: cloud_classification_snapshots")
    if not any(classifications_by_run.get(sim_name) is not None for sim_name in sim_names):
        print("Skipping Figure 7: high-cloud classification is unavailable.")
    else:
        snapshot_specs = [
            ("high_cloud_mask", "Total high cloud"),
            ("convective_precipitating_high_cloud_mask", "Convective precipitating"),
            ("stratiform_precipitating_high_cloud_mask", "Stratiform precipitating"),
            ("nonprecipitating_high_cloud_mask", "Non-precipitating"),
        ]
        fig, ax = plt.subplots(
            len(sim_names), len(snapshot_specs),
            figsize=(3.4 * len(snapshot_specs), 2.8 * len(sim_names)),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        if len(sim_names) == 1:
            ax = ax[None, :]
        for i, sim_name in enumerate(sim_names):
            classif = classifications_by_run.get(sim_name)
            if classif is None:
                for j, (_key, label) in enumerate(snapshot_specs):
                    a = ax[i, j]
                    a.text(0.5, 0.5, "not available", transform=a.transAxes, ha="center", va="center")
                    if i == 0:
                        a.set_title(label)
                    if j == 0:
                        a.set_ylabel(f"{sim_name}\nnot available")
                    if i == len(sim_names) - 1:
                        a.set_xlabel("Longitude")
                    a.set_xlim(-180, 180)
                    a.set_ylim(*TROPICAL_LAT_BOUNDS)
                continue
            snap_time = representative_snapshot_time(classif)
            for j, (key, label) in enumerate(snapshot_specs):
                a = ax[i, j]
                if snap_time is None:
                    a.text(0.5, 0.5, "no Time dim", transform=a.transAxes, ha="center", va="center")
                    continue
                m = subset_lat(classif[key].sel(Time=snap_time).astype(float), TROPICAL_LAT_BOUNDS)
                a.pcolormesh(m.longitude, m.latitude, m, vmin=0, vmax=1, cmap="Greys", shading="auto")
                if i == 0:
                    a.set_title(label)
                if j == 0:
                    a.set_ylabel(f"{sim_name}\n{np.datetime_as_string(np.asarray(snap_time), unit='h')}")
                if i == len(sim_names) - 1:
                    a.set_xlabel("Longitude")
                a.set_xlim(-180, 180)
                a.set_ylim(*TROPICAL_LAT_BOUNDS)
        fig.suptitle("Representative high-cloud class snapshots (time of max tropical high-cloud coverage)")
        save_figure(7, fig, "cloud_classification_snapshots")
        progress("Done with figure: cloud_classification_snapshots")


# -----------------------------------------------------------------------------
# Figure 1. High cloud vs precipitation, conditioned on PW or TTL N2
# -----------------------------------------------------------------------------
# Interpretation:
#   * If u-nudged has more high cloud at fixed precipitation and fixed PW/N2_TTL,
#     the response is not just due to sampling wetter or more stable columns.
#   * If T-nudged matches u-nudged at fixed N2_TTL, this supports the
#     thermal/static-stability pathway.
#   * If u-nudged remains distinct from T-nudged at fixed N2_TTL, this supports
#     a wind/shear/wave pathway.
#   * If the signal is mainly non-precipitating or stratiform precipitating high cloud
#     rather than convective precipitating convection, do not interpret it as simply deeper
#     precipitating convection.
#   * Non-precipitating is not plotted against local precipitation here
#     because weak local precipitation is part of that category definition.
#     Non-precipitating and nonprecipitating high cloud are diagnosed in the
#     classification summary figures and should be interpreted as
#     occurrence/lifetime/proximity information, not as a local rain-rate
#     response curve.

if should_run_figure(1):
    progress("Starting Figure 1: conditional_highcloud_vs_precip")
    available_condition = "N2_TTL" if all("N2_TTL" in f for f in fields_by_run) else "pw"
    if not all(available_condition in f for f in fields_by_run):
        print("Skipping Figure 1: neither N2_TTL nor PW is available in all runs.")
    else:
        if available_condition == "N2_TTL":
            condition_label = r"$N^2_{TTL}$"
            cond_edges, support_mode, support_bounds = common_support_condition_edges(
                fields_by_run, classifications_by_run, available_condition
            )
        else:
            condition_label = "PW"
            cond_edges = pooled_tercile_edges_for_classified(
                fields_by_run, classifications_by_run, available_condition
            )
            support_mode, support_bounds = "pooled occupancy", None
        precip_bins = pooled_weighted_bins_for_classified(
            fields_by_run, classifications_by_run, "precip", PRECIP_NBINS
        )
        if cond_edges is None or precip_bins is None:
            print("Skipping Figure 1: could not construct condition or precipitation bins.")
        else:
            condition_support = condition_support_table(
                fields_by_run, classifications_by_run, available_condition, cond_edges,
                "conditional_highcloud_vs_precip",
            )
            if available_condition == "N2_TTL" and support_mode != "common-support":
                warnings.warn(
                    "Figure 1 N2_TTL panels are regime-occupancy diagnostics: "
                    "robust N2_TTL ranges do not overlap across all simulations."
                )

            response_specs = [
                ("total high cloud", lambda c: c["high_cloud_fraction"], "High-cloud fraction, cloud top <= 440 hPa"),
                ("convective precipitating", lambda c: c["convective_precipitating_high_cloud_mask"].astype(float), "Convective-precipitating high-cloud probability"),
                ("stratiform precipitating", lambda c: c["stratiform_precipitating_high_cloud_mask"].astype(float), "Stratiform precipitating high-cloud probability"),
            ]

            labels = condition_labels(cond_edges, condition_label)
            n_condition_bins = len(labels)
            fig, ax = plt.subplots(
                len(response_specs), n_condition_bins,
                figsize=(12.0, 8.0),
                constrained_layout=True,
                sharex=True,
                sharey="row",
            )
            if n_condition_bins == 1:
                ax = ax[:, None]
            count_rows = []

            for r, (resp_name, resp_getter, ylabel) in enumerate(response_specs):
                for c, label in enumerate(labels):
                    lo, hi = cond_edges[c], cond_edges[c + 1]
                    for i, sim_name in enumerate(sim_names):
                        classif = classifications_by_run.get(sim_name)
                        if classif is None:
                            continue
                        fields = fields_by_run[i]
                        vals, weights = sample_matching_arrays(
                            fields["precip"], resp_getter(classif), fields[available_condition],
                            lat_bounds=TROPICAL_LAT_BOUNDS,
                            **PLOT_SAMPLE_LIMITS,
                        )
                        pr_v, y_v, cond_v = vals
                        cond_mask = (cond_v >= lo) & (cond_v <= hi if c == n_condition_bins - 1 else cond_v < hi)
                        centres, mean, count = weighted_binned_mean(
                            pr_v[cond_mask], y_v[cond_mask], precip_bins,
                            weights=weights[cond_mask],
                            min_count=MIN_BIN_COUNT,
                        )
                        plot_supported_curve(
                            ax[r, c], centres, mean, count, cols[i],
                            label=sim_name if (r == 0 and c == 0) else None,
                        )
                        for centre, n in zip(centres, count):
                            count_rows.append(
                                dict(figure=1, simulation=sim_name, response=resp_name,
                                     condition=label, condition_mode=support_mode,
                                     precip_bin_center=centre, sample_count=n)
                            )

                    if r == 0:
                        title = f"{label}\n{lo:.3g} to {hi:.3g}"
                        if available_condition == "N2_TTL":
                            title += f"\n{support_mode}"
                        ax[r, c].set_title(title)
                        if available_condition == "N2_TTL" and support_mode != "common-support":
                            ax[r, c].set_facecolor("#fff8e8")
                    if c == 0:
                        ax[r, c].set_ylabel(ylabel)
                    if r == len(response_specs) - 1:
                        ax[r, c].set_xlabel(r"Precipitation (mm day$^{-1}$)")
                    ax[r, c].grid(alpha=0.2)
                    add_low_count_note(ax[r, c])

            ax[0, 0].legend(loc="best")
            title = f"High cloud vs precipitation conditioned on {condition_label}"
            if available_condition == "N2_TTL" and support_bounds is not None:
                title += f" ({support_mode}: {support_bounds[0]:.3g} to {support_bounds[1]:.3g})"
            elif available_condition == "N2_TTL":
                title += " (regime occupancy, not common support)"
            fig.suptitle(title)
            save_figure(1, fig, "conditional_highcloud_vs_precip")
            count_rows_to_frame(count_rows, "conditional_highcloud_vs_precip")
            progress("Done with figure: conditional_highcloud_vs_precip")


# -----------------------------------------------------------------------------
# Figure 2. Cloud-top pressure vs precipitation, conditioned on PW
# -----------------------------------------------------------------------------
# Interpretation:
#   * Lower cloud-top pressure in u-nudged at fixed precipitation and PW implies altered
#     vertical cloud expression independent of local rain rate and moisture.
#   * A difference between convective and stratiform curves distinguishes the
#     strong-rain/ascent convective pathway from weaker precipitating high cloud.
#   * If it appears mainly outside convective precipitation, it suggests changes
#     in stratiform precipitating or non-precipitating cloud lifetime, detrainment, or
#     upper-level cloud-radiative effects.

if should_run_figure(2):
    progress("Starting Figure 2: conditional_cloudtop_pressure_vs_precip_pw")
    if not all("pw" in f for f in fields_by_run):
        print("Skipping Figure 2: PW is unavailable in at least one run.")
    elif not any(classifications_by_run.get(sim_name) is not None for sim_name in sim_names):
        print("Skipping Figure 2: high-cloud classification is unavailable.")
    else:
        pw_edges = pooled_tercile_edges_for_classified(fields_by_run, classifications_by_run, "pw")
        precip_bins = pooled_weighted_bins_for_classified(
            fields_by_run, classifications_by_run, "precip", PRECIP_NBINS
        )

        class_specs = [
            ("all high", "high_cloud_mask", "-", "All high cloud"),
            ("convective precipitating", "convective_precipitating_high_cloud_mask", "--", "Convective precipitating"),
            ("stratiform precipitating", "stratiform_precipitating_high_cloud_mask", ":", "Stratiform precipitating"),
        ]

        labels = condition_labels(pw_edges, "PW")
        n_condition_bins = len(labels)
        fig, ax = plt.subplots(2, n_condition_bins, figsize=(12.0, 6.6), constrained_layout=True, sharex=True)
        if n_condition_bins == 1:
            ax = ax[:, None]
        count_rows = []

        for c, label in enumerate(labels):
            lo, hi = pw_edges[c], pw_edges[c + 1]
            for i, sim_name in enumerate(sim_names):
                classif = classifications_by_run.get(sim_name)
                if classif is None:
                    continue
                fields = fields_by_run[i]
                vals, weights = sample_matching_arrays(
                    fields["precip"], fields["pw"], classif["cloud_top_pressure"], classif["high_cloud_mask"].astype(float),
                    classif["convective_precipitating_high_cloud_mask"].astype(float),
                    classif["stratiform_precipitating_high_cloud_mask"].astype(float),
                    lat_bounds=TROPICAL_LAT_BOUNDS,
                    **PLOT_SAMPLE_LIMITS,
                )
                pr_v, pw_v, ctp_v, high_v, active_v, generated_v = vals
                cond_mask = (pw_v >= lo) & (pw_v <= hi if c == n_condition_bins - 1 else pw_v < hi)

                masks = {
                    "high_cloud_mask": high_v >= 0.5,
                    "convective_precipitating_high_cloud_mask": active_v >= 0.5,
                    "stratiform_precipitating_high_cloud_mask": generated_v >= 0.5,
                }
                for class_label, mask_key, linestyle, _pretty in class_specs:
                    valid = cond_mask & masks[mask_key]
                    centres, mean, count = weighted_binned_mean(
                        pr_v[valid], ctp_v[valid], precip_bins,
                        weights=weights[valid],
                        min_count=MIN_BIN_COUNT,
                    )
                    plot_supported_curve(
                        ax[0, c], centres, mean, count, cols[i], linestyle=linestyle,
                        label=f"{sim_name}, {class_label}" if c == 0 else None,
                    )
                    for centre, n in zip(centres, count):
                        count_rows.append(
                            dict(figure=2, panel="cloud_top", simulation=sim_name,
                                 class_name=class_label, pw_condition=label,
                                 precip_bin_center=centre, sample_count=n)
                        )

                high_den = (high_v >= 0.5) & cond_mask
                for frac_label, numerator, linestyle in [
                    ("convective fraction", active_v >= 0.5, "--"),
                    ("stratiform precipitating fraction", generated_v >= 0.5, ":"),
                ]:
                    centres, frac, count = weighted_binned_fraction(
                        pr_v, numerator, precip_bins,
                        denominator=high_den,
                        weights=weights,
                        min_count=MIN_BIN_COUNT,
                    )
                    plot_supported_curve(
                        ax[1, c], centres, frac, count, cols[i], linestyle=linestyle,
                        label=f"{sim_name}, {frac_label}" if c == 0 else None,
                    )
                    for centre, n in zip(centres, count):
                        count_rows.append(
                            dict(figure=2, panel="class_fraction", simulation=sim_name,
                                 class_name=frac_label, denominator="all high cloud in PW/precip bin",
                                 pw_condition=label,
                                 precip_bin_center=centre, sample_count=n)
                        )

            ax[0, c].set_title(f"{label}\n{lo:.3g} to {hi:.3g} kg m$^{{-2}}$")
            ax[1, c].set_xlabel(r"Precipitation (mm day$^{-1}$)")
            for rr in (0, 1):
                ax[rr, c].grid(alpha=0.2)
                add_low_count_note(ax[rr, c])

        for c in range(n_condition_bins):
            ax[0, c].invert_yaxis()
        ax[0, 0].set_ylabel("Mean cloud-top pressure (hPa)")
        ax[1, 0].set_ylabel("Fraction of high-cloud samples\n(denominator: all high cloud)")
        ax[0, 0].legend(loc="best", fontsize=7)
        ax[1, 0].legend(loc="best", fontsize=7)
        fig.suptitle("Cloud-top pressure and high-cloud class fractions at fixed PW")
        save_figure(2, fig, "conditional_cloudtop_pressure_vs_precip_pw")
        count_rows_to_frame(count_rows, "conditional_cloudtop_pressure_vs_precip_pw")
        progress("Done with figure: conditional_cloudtop_pressure_vs_precip_pw")


# -----------------------------------------------------------------------------
# Figure 8. Non-precipitating high cloud vs PW by lagged precipitation
# -----------------------------------------------------------------------------
# Interpretation:
#   * This avoids plotting non-precipitating cloud against current local precipitation.
#   * A stronger non-precipitating occurrence after previous-timestep rain suggests
#     a precipitation-memory or detrainment effect.

if should_run_figure(8):
    progress("Starting Figure 8: conditional_nonprecipitating_highcloud_pw_by_lagged_precip")
    if not all("pw" in f for f in fields_by_run):
        print("Skipping Figure 8: PW is unavailable in at least one run.")
    elif not any(classifications_by_run.get(sim_name) is not None for sim_name in sim_names):
        print("Skipping Figure 8: high-cloud classification is unavailable.")
    else:
        pw_bins = pooled_weighted_bins_for_classified(
            fields_by_run, classifications_by_run, "pw", PW_NBINS
        )
        target_specs = [
            ("non-precipitating", "nonprecipitating_high_cloud_mask", "Non-precipitating high-cloud occurrence"),
        ]
        lag_specs = [
            (f"previous P < {P_PRECIP_MIN:g}", "dry_lagged_precip", lambda lag: lag < P_PRECIP_MIN),
            (f"previous P >= {P_PRECIP_MIN:g}", "wet_lagged_precip", lambda lag: lag >= P_PRECIP_MIN),
        ]
        fig, ax = plt.subplots(
            len(target_specs), len(lag_specs),
            figsize=(8.8, 5.8),
            constrained_layout=True,
            sharex=True,
            sharey="row",
        )
        count_rows = []
        for r, (target_label, target_key, ylabel) in enumerate(target_specs):
            for c, (lag_label, lag_id, lag_mask_func) in enumerate(lag_specs):
                for i, sim_name in enumerate(sim_names):
                    fields = fields_by_run[i]
                    classif = classifications_by_run.get(sim_name)
                    if classif is None:
                        continue
                    lagged_precip = fields["precip"].shift(Time=LAGGED_PRECIP_TIME_STEPS)
                    vals, weights = sample_matching_arrays(
                        fields["pw"],
                        lagged_precip,
                        classif[target_key].astype(float),
                        lat_bounds=TROPICAL_LAT_BOUNDS,
                        **PLOT_SAMPLE_LIMITS,
                    )
                    pw_v, lag_v, y_v = vals
                    split = lag_mask_func(lag_v)
                    centres, mean, count = weighted_binned_mean(
                        pw_v[split],
                        y_v[split],
                        pw_bins,
                        weights=weights[split],
                        min_count=MIN_BIN_COUNT,
                    )
                    plot_supported_curve(
                        ax[r, c], centres, mean, count, cols[i],
                        label=sim_name if (r == 0 and c == 0) else None,
                    )
                    for centre, n in zip(centres, count):
                        count_rows.append(
                            dict(
                                figure=8,
                                simulation=sim_name,
                                target=target_label,
                                split=lag_id,
                                pw_bin_center=centre,
                                sample_count=n,
                                denominator="all samples in PW bin and lagged-precip split",
                            )
                        )
                ax[r, c].set_title(lag_label if r == 0 else "")
                ax[r, c].grid(alpha=0.2)
                add_low_count_note(ax[r, c])
                if c == 0:
                    ax[r, c].set_ylabel(ylabel)
                if r == len(target_specs) - 1:
                    ax[r, c].set_xlabel(r"PW (kg m$^{-2}$)")
        ax[0, 0].legend(loc="best")
        fig.suptitle(
            "Non-precipitating high cloud vs PW split by previous-timestep precipitation\n"
            "Current local precipitation is not used as the conditioning axis"
        )
        save_figure(8, fig, "conditional_nonprecipitating_highcloud_pw_by_lagged_precip")
        count_rows_to_frame(count_rows, "conditional_nonprecipitating_highcloud_pw_by_lagged_precip")
        progress("Done with figure: conditional_nonprecipitating_highcloud_pw_by_lagged_precip")


# -----------------------------------------------------------------------------
# Figure 3. Precipitation vs PW split by wave amplitude and phase
# -----------------------------------------------------------------------------
# Interpretation:
#   * If u-nudged changes the precipitation-PW curve mainly during strong wave
#     activity, this supports a wind/wave-mediated organisation mechanism.
#   * If T-nudged behaves similarly during strong wave activity, the wave
#     response may be mediated through thermal/static-stability changes.
#   * If u-nudged changes wave-active convection while T-nudged does not, that
#     is strong evidence for a wind/shear/wave-filtering pathway.

if should_run_figure(3):
    progress("Starting Figure 3: conditional_precip_pw_by_wave_amplitude_and_phase")
    if kf_filter is None:
        print("Skipping Figure 3: waveFilter.kf_filter is unavailable.")
    elif not all("pw" in f for f in fields_by_run):
        print("Skipping Figure 3: PW is unavailable in at least one run.")
    else:
        pw_bins = pooled_weighted_bins(fields_by_run, "pw", PW_NBINS)
        row_specs = [
            ("weak amplitude", "amplitude", "weak"),
            ("strong amplitude", "amplitude", "strong"),
            ("suppressed phase", "phase", "suppressed"),
            ("neutral phase", "phase", "neutral"),
            ("active phase", "phase", "active"),
        ]
        fig, ax = plt.subplots(
            len(row_specs), len(WAVE_BANDS_FOR_FIG3),
            figsize=(5.2 * len(WAVE_BANDS_FOR_FIG3), 10.0),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        if len(WAVE_BANDS_FOR_FIG3) == 1:
            ax = ax[:, None]
        count_rows = []

        for wj, wave_name in enumerate(WAVE_BANDS_FOR_FIG3):
            amp_by_run = []
            active_index_by_run = []
            for i, fields in enumerate(fields_by_run):
                field_key = WAVE_FILTER_FIELD if WAVE_FILTER_FIELD in fields else "precip"
                try:
                    wave = filter_latmean_wave(fields, i, field_key, wave_name)
                    std = float(wave.std(dim=("Time", "longitude"), skipna=True).values)
                    wave_norm = wave / std if std > 0 else xr.zeros_like(wave)
                    amp = np.abs(wave_norm)
                    active_index = active_phase_sign(field_key) * wave_norm
                    active_index.attrs["phase_convention"] = (
                        "positive means convectively active; for OLR this is negative filtered OLR"
                    )
                    amp_by_run.append(amp)
                    active_index_by_run.append(active_index)
                except Exception as exc:
                    print(f"Skipping {sim_names[i]} {wave_name} wave split: {exc}")
                    amp_by_run.append(None)
                    active_index_by_run.append(None)

            amp_samples = []
            amp_weights = []
            for fields, amp in zip(fields_by_run, amp_by_run):
                if amp is None:
                    continue
                vals, weights = sample_matching_arrays(
                    amp, fields["pw"], fields["precip"],
                    lat_bounds=TROPICAL_LAT_BOUNDS,
                    **PLOT_SAMPLE_LIMITS,
                )
                amp_samples.append(vals[0])
                amp_weights.append(weights)
            if not amp_samples:
                continue
            amp_edges = weighted_quantile(
                np.concatenate(amp_samples), [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
                np.concatenate(amp_weights),
            )

            for i, (fields, amp, active_index) in enumerate(zip(fields_by_run, amp_by_run, active_index_by_run)):
                if amp is None or active_index is None:
                    continue
                vals, weights = sample_matching_arrays(
                    fields["pw"], fields["precip"], amp, active_index,
                    lat_bounds=TROPICAL_LAT_BOUNDS,
                    **PLOT_SAMPLE_LIMITS,
                )
                pw_v, pr_v, amp_v, phase_v = vals
                split_masks = [
                    ("weak", "amplitude", amp_v <= amp_edges[1], 0),
                    ("strong", "amplitude", amp_v >= amp_edges[2], 1),
                    ("suppressed", "phase", phase_v <= -WAVE_PHASE_THRESHOLD_STD, 2),
                    ("neutral", "phase", np.abs(phase_v) < WAVE_PHASE_THRESHOLD_STD, 3),
                    ("active", "phase", phase_v >= WAVE_PHASE_THRESHOLD_STD, 4),
                ]
                for split_label, split_type, mask, row in split_masks:
                    centres, mean, count = weighted_binned_mean(
                        pw_v[mask], pr_v[mask], pw_bins,
                        weights=weights[mask],
                        min_count=MIN_BIN_COUNT,
                    )
                    plot_supported_curve(
                        ax[row, wj], centres, mean, count, cols[i],
                        label=sim_names[i] if (wj == 0 and row == 0) else None,
                    )
                    for centre, n in zip(centres, count):
                        count_rows.append(
                            dict(figure=3, simulation=sim_names[i], wave=wave_name,
                                 split_type=split_type, split_label=split_label,
                                 pw_bin_center=centre, sample_count=n)
                        )

            ax[0, wj].set_title(wave_label(wave_name))
            for row, (row_label, _split_type, _split_label) in enumerate(row_specs):
                ax[row, wj].set_xlabel(r"PW (kg m$^{-2}$)")
                ax[row, wj].grid(alpha=0.2)
                add_low_count_note(ax[row, wj])
        for row, (row_label, _split_type, _split_label) in enumerate(row_specs):
            ax[row, 0].set_ylabel(f"{row_label}\nPrecipitation (mm day$^{{-1}}$)")
        ax[0, 0].legend(loc="best")
        fig.suptitle(
            "Precipitation-PW relationship split by filtered wave amplitude and phase\n"
            f"Phase threshold: |active-index| >= {WAVE_PHASE_THRESHOLD_STD:g} std; active OLR phase is negative filtered OLR"
        )
        save_figure(3, fig, "conditional_precip_pw_by_wave_amplitude")
        save_figure(3, fig, "conditional_precip_pw_by_wave_amplitude_and_phase")
        wave_count_df = count_rows_to_frame(count_rows, "conditional_precip_pw_by_wave_amplitude_and_phase")
        if save and len(wave_count_df) > 0:
            wave_count_df.to_csv(
                os.path.join(save_loc, "conditional_precip_pw_by_wave_amplitude_bin_counts.csv"),
                index=False,
            )
        progress("Done with figure: conditional_precip_pw_by_wave_amplitude_and_phase")


# -----------------------------------------------------------------------------
# Figure 9. Wave-conditioned cloud classes and cloud-top pressure
# -----------------------------------------------------------------------------

if should_run_figure(9):
    progress("Starting Figure 9: wave_conditioned_cloud_classes_and_cloud_top_pressure")
    if kf_filter is None:
        print("Skipping Figure 9: waveFilter.kf_filter is unavailable.")
    else:
        wave_cloud_table = wave_conditioned_cloud_table(
            fields_by_run, classifications_by_run
        )
        plot_wave_conditioned_cloud_diagnostics(wave_cloud_table)
        progress("Done with figure: wave_conditioned_cloud_classes_and_cloud_top_pressure")


# -----------------------------------------------------------------------------
# Figure 4. Residual high cloud after removing local thermodynamic controls
# -----------------------------------------------------------------------------
# Interpretation:
#   * Residuals near zero for both nudged runs mean high-cloud differences are
#     mostly explained by local rain/PW/CAPE/CIN/latitude.
#   * If T-nudged residuals resemble u-nudged residuals when binned by N2_TTL,
#     this supports a thermal/static-stability pathway.
#   * If u-nudged has a distinct residual at fixed N2_TTL, especially organised
#     by dudz_TTL, this supports a wind/shear/wave pathway.
#   * Residuals in convective precipitating convection imply deeper or more vertically
#     connected precipitating convection; residuals in stratiform precipitating or
#     non-precipitating cloud imply changes in detrainment, lifetime, or upper-level
#     cloud-radiative feedback rather than simply deeper convective cores.

if should_run_figure(4):
    progress("Starting Figure 4: residual_highcloud_thermo_controls")
    if any(classifications_by_run.get(sim_name) is None for sim_name in sim_names):
        print("Skipping Figure 4: high-cloud classification is unavailable.")
    else:
        feature_keys = available_residual_features(fields_by_run)
        bin_keys = ["latitude"]
        if all("N2_TTL" in fields for fields in fields_by_run):
            bin_keys.append("N2_TTL")
        if all("dudz_TTL" in fields for fields in fields_by_run):
            bin_keys.append("dudz_TTL")

        target_specs = [
            ("total", "Total high cloud"),
            ("convective", "Convective precipitating"),
            ("stratiform", "Stratiform precipitating"),
            ("nonprecipitating", "Non-precipitating"),
        ]

        residual_samples = {}
        models = {}
        for target_key, target_label in target_specs:
            Xc, yc, wc = model_training_data(
                fields_by_run[0], classifications_by_run[sim_names[0]], target_key, feature_keys
            )
            models[target_key] = fit_control_relationship(Xc, yc, wc, target_label)

            for i, sim_name in enumerate(sim_names):
                X, y, bins, weights = residual_sample_data(
                    fields_by_run[i], classifications_by_run[sim_name], target_key, feature_keys, bin_keys
                )
                pred = np.clip(models[target_key].predict(X), 0.0, 1.0)
                residual = y - pred
                residual_samples[(target_key, sim_name)] = dict(
                    residual=residual,
                    bins=bins,
                    weights=weights,
                )
                mean_resid = np.sum(weights * residual) / np.sum(weights)
                print(f"Mean residual for {sim_name} {target_label}: {mean_resid:.4f}")

        fig, ax = plt.subplots(
            len(target_specs), len(bin_keys),
            figsize=(4.2 * len(bin_keys), 8.0),
            constrained_layout=True,
            sharey="row",
        )
        if len(target_specs) == 1:
            ax = ax[None, :]
        if len(bin_keys) == 1:
            ax = ax[:, None]

        count_rows = []
        for r, (target_key, target_label) in enumerate(target_specs):
            for c, bin_key in enumerate(bin_keys):
                pooled_vals = []
                pooled_weights = []
                for sim_name in sim_names:
                    sample = residual_samples[(target_key, sim_name)]
                    pooled_vals.append(sample["bins"][bin_key])
                    pooled_weights.append(sample["weights"])
                pooled_vals = np.concatenate(pooled_vals)
                pooled_weights = np.concatenate(pooled_weights)
                if bin_key == "latitude":
                    bins = np.arange(TROPICAL_LAT_BOUNDS[0], TROPICAL_LAT_BOUNDS[1] + LATITUDE_BIN_WIDTH, LATITUDE_BIN_WIDTH)
                    xlabel = "Latitude (deg)"
                elif bin_key == "N2_TTL":
                    bins = unique_monotonic_bins(weighted_quantile(pooled_vals, np.linspace(0.02, 0.98, 11), pooled_weights))
                    xlabel = r"$N^2_{TTL}$ (s$^{-2}$)"
                else:
                    bins = unique_monotonic_bins(weighted_quantile(pooled_vals, np.linspace(0.02, 0.98, 11), pooled_weights))
                    xlabel = r"$du/dz_{TTL}$ (s$^{-1}$)"

                for i, sim_name in enumerate(sim_names):
                    sample = residual_samples[(target_key, sim_name)]
                    centres, mean, count = weighted_binned_mean(
                        sample["bins"][bin_key],
                        sample["residual"],
                        bins,
                        weights=sample["weights"],
                        min_count=MIN_BIN_COUNT,
                    )
                    plot_supported_curve(
                        ax[r, c], centres, mean, count, cols[i],
                        label=sim_name if (r == 0 and c == 0) else None,
                    )
                    for centre, n in zip(centres, count):
                        count_rows.append(
                            dict(figure=4, simulation=sim_name, target=target_label,
                                 bin_variable=bin_key, bin_center=centre, sample_count=n)
                        )

                ax[r, c].axhline(0, color="0.6", lw=0.8)
                ax[r, c].grid(alpha=0.2)
                add_low_count_note(ax[r, c])
                if r == 0:
                    ax[r, c].set_title(f"Residual vs {bin_key}")
                if c == 0:
                    ax[r, c].set_ylabel(f"{target_label}\nobserved - control-predicted")
                if r == len(target_specs) - 1:
                    ax[r, c].set_xlabel(xlabel)

        ax[0, 0].legend(loc="best")
        fig.suptitle("High-cloud residuals after removing control local-thermodynamic relationship")
        save_figure(4, fig, "residual_highcloud_thermo_controls")
        count_rows_to_frame(count_rows, "residual_highcloud_thermo_controls")
        progress("Done with figure: residual_highcloud_thermo_controls")
