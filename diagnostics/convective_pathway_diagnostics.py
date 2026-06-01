#!/usr/bin/env python
# coding: utf-8

"""
Conditional convective/cloud pathway diagnostics for MPAS-A nudging experiments.

This script asks whether precipitation and high-cloud changes in the control,
u-nudged, and T-nudged experiments are better explained by:

  * local thermodynamic controls such as rain rate, PW, CAPE, CIN and latitude;
  * direct or balanced UTLS thermal/static-stability changes; or
  * wind/shear/wave effects, especially in organised equatorial disturbances.

The high-cloud diagnostics deliberately separate column-connected high cloud
from anvil and detached cirrus-like high cloud.  High cloud above 12 km is not
automatically interpreted as deep precipitating convection.
"""

import os
import sys
import warnings

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

try:
    from waveFilter import kf_filter
except ModuleNotFoundError:
    kf_filter = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = "/home/cwp29/tmp/mpas/crf_on/final/convective_pathways/"

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: conditional_highcloud_vs_precip
#   2: conditional_cloudtop_vs_precip_pw
#   3: conditional_precip_pw_by_wave_amplitude
#   4: residual_highcloud_thermo_controls
figures_to_run = []
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)

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
obs_per_day = int(round(24 / dt))

TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
WAVE_FILTER_LAT_BOUNDS = (-10.0, 10.0)

# High-cloud classification thresholds.  Keep these near the top so threshold
# sensitivity tests do not require editing the classification logic.
CF_THRESH = 0.05
Z_HIGH = 12.0       # km
Z_LOW_MIN = 2.0     # km
Z_LOW_MAX = 5.0     # km
Z_MID_MIN = 5.0     # km
Z_MID_MAX = 10.0    # km
Z_UPPER_MIN = 10.0  # km
P_DEEP = 5.0        # mm day-1
P_WEAK = 1.0        # mm day-1
MAX_CLEAR_GAP = 3.0 # km
ANVIL_RADIUS_DEG = 5.0
ANVIL_TIME_STEPS = 1

# Conditional sampling and binning controls.  These diagnostics use high-time
# gridded fields, but they stride before materialising values to keep memory
# predictable.
MIN_BIN_COUNT = 50
PRECIP_NBINS = 14
PW_NBINS = 14
LATITUDE_BIN_WIDTH = 2.0

PLOT_SAMPLE_LIMITS = dict(max_time=96, max_lat=96, max_lon=192)
RESIDUAL_FIT_SAMPLE_LIMITS = dict(max_time=120, max_lat=120, max_lon=240)
RESIDUAL_PLOT_SAMPLE_LIMITS = dict(max_time=120, max_lat=120, max_lon=240)

SAVE_CLASSIFICATION_CACHE = False
classification_cache_dir = os.path.join(save_loc, "classification_cache")

cache_dir = "/store/atmos-adk33/cwp29/cache/convective_pathways/"
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
    "cldfrac": ["cldfrac", "cloud_fraction", "cl"],
    "cloud_top_height": ["cloud_top_height", "cth", "cloudtop", "cloud_top"],
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
        + VAR_ALIASES["cldfrac"]
        + VAR_ALIASES["cloud_top_height"]
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


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


def run_all_figures():
    return all_figures_requested(figures_to_run)


def save_figure(number, fig, stem):
    """Save both project-style numbered and task-requested exact filenames."""
    figsave.save(fig, stem, number=number)
    if save and should_run_figure(number):
        fig.savefig(os.path.join(save_loc, f"{stem}.png"), dpi=300)


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


def get_cloud_fraction(ds):
    name = find_var(ds, "cldfrac", required=False)
    if name is None:
        warnings.warn("No cloud-fraction variable found; high-cloud diagnostics will be skipped.")
        return None
    cf = finite_or_nan(ds[name])
    zdim = vertical_dim(cf)
    if zdim != "z":
        cf = cf.rename({zdim: "z"})
    if "z" not in cf.coords:
        warnings.warn("Cloud fraction has no usable z coordinate; high-cloud diagnostics will be skipped.")
        return None
    zvals = np.asarray(cf.z.values, dtype=float)
    if zvals.size != cf.sizes["z"] or not np.all(np.isfinite(zvals)):
        warnings.warn("Cloud fraction z coordinate is not usable; high-cloud diagnostics will be skipped.")
        return None
    if np.nanmax(zvals) < 100.0:
        cf = cf.assign_coords(z=cf.z * 1000.0)
    return cf


def available_cloud_fraction_times(cf):
    """Return times where the attached cloud-fraction file has real data."""
    if "Time" not in cf.dims:
        return None
    indexers = {dim: 0 for dim in cf.dims if dim != "Time"}
    valid = cf.isel(indexers).notnull().load()
    return valid.Time.values[np.asarray(valid.values, dtype=bool)]


def z_in_km(da):
    zvals = np.asarray(da.z.values, dtype=float)
    return xr.DataArray(
        zvals / 1000.0 if np.nanmax(np.abs(zvals)) > 100.0 else zvals,
        dims=("z",),
        coords={"z": da.z},
        name="z_km",
    )


def any_cloud_in_layer(cloudy, z_km, zmin, zmax):
    return cloudy.where((z_km >= zmin) & (z_km < zmax), drop=True).any(dim="z")


def rolling_or_dilated_max(mask, lat_radius=ANVIL_RADIUS_DEG, lon_radius=ANVIL_RADIUS_DEG, time_radius=ANVIL_TIME_STEPS):
    """Simple time/lat/lon dilation for identifying recently convective anvils."""
    x = mask.astype(np.int8)
    rolling_kwargs = {}
    if "Time" in x.dims and time_radius > 0:
        rolling_kwargs["Time"] = 2 * int(time_radius) + 1
    if "latitude" in x.dims and lat_radius > 0:
        dlat = float(np.nanmedian(np.abs(np.diff(x.latitude.values))))
        rolling_kwargs["latitude"] = 2 * int(np.ceil(lat_radius / dlat)) + 1
    if "longitude" in x.dims and lon_radius > 0:
        dlon = float(np.nanmedian(np.abs(np.diff(x.longitude.values))))
        rolling_kwargs["longitude"] = 2 * int(np.ceil(lon_radius / dlon)) + 1

    if not rolling_kwargs:
        return mask
    return x.rolling(**rolling_kwargs, center=True, min_periods=1).max() >= 1


def classify_high_cloud(
    cldfrac,
    z,
    precip,
    cf_thresh=CF_THRESH,
    z_high=Z_HIGH,
    z_low_min=Z_LOW_MIN,
    z_low_max=Z_LOW_MAX,
    z_mid_min=Z_MID_MIN,
    z_mid_max=Z_MID_MAX,
    z_upper_min=Z_UPPER_MIN,
    p_deep=P_DEEP,
    p_weak=P_WEAK,
    max_clear_gap=MAX_CLEAR_GAP,
    anvil_radius_deg=ANVIL_RADIUS_DEG,
    anvil_time_steps=ANVIL_TIME_STEPS,
):
    """
    Classify high cloud into column-connected, anvil, and cirrus-like categories.

    A high cloud column must have low, mid, and upper cloud to count as
    column-connected.  The optional clear-gap test requires cloud in every
    max_clear_gap-km slab between the lower cloud layer and high cloud.
    """
    cf = cldfrac
    pr = precip
    if "Time" in cf.dims and "Time" in pr.dims:
        common_time = cf.Time.to_index().intersection(pr.Time.to_index())
        if len(common_time) == 0:
            raise ValueError("cloud fraction and precipitation have no overlapping times")
        cf = cf.sel(Time=common_time)
        pr = pr.sel(Time=common_time)
    cf, pr = xr.align(cf, pr, join="inner")
    z_km = z / 1000.0 if float(z.max()) > 100.0 else z
    z_km = xr.DataArray(np.asarray(z_km, dtype=float), dims=("z",), coords={"z": cf.z})

    cloudy = cf >= cf_thresh
    cloud_top_height = z_km.where(cloudy).max(dim="z", skipna=True)
    cloud_top_height.name = "cloud_top_height"
    cloud_top_height.attrs["units"] = "km"

    high_cloud_fraction = cf.where(z_km >= z_high).mean(dim="z", skipna=True)
    high_cloud_fraction.name = "high_cloud_fraction"
    high_cloud_mask = ((cloud_top_height >= z_high) | (high_cloud_fraction >= cf_thresh)).fillna(False)
    high_cloud_mask.name = "high_cloud_mask"

    low_cloud = any_cloud_in_layer(cloudy, z_km, z_low_min, z_low_max)
    mid_cloud = any_cloud_in_layer(cloudy, z_km, z_mid_min, z_mid_max)
    upper_cloud = any_cloud_in_layer(cloudy, z_km, z_upper_min, max(float(z_km.max()), z_high + 0.1))

    gap_ok = xr.ones_like(high_cloud_mask, dtype=bool)
    if max_clear_gap is not None and max_clear_gap > 0:
        slab_starts = np.arange(z_low_min, z_high, max_clear_gap)
        for z0 in slab_starts:
            z1 = min(z0 + max_clear_gap, max(float(z_km.max()), z_high))
            if z1 <= z0:
                continue
            gap_ok = gap_ok & any_cloud_in_layer(cloudy, z_km, z0, z1 + 1.0e-6)

    column_connected_mask = (
        high_cloud_mask & low_cloud & mid_cloud & upper_cloud & gap_ok
    ).fillna(False)
    column_connected_mask.name = "column_connected_high_cloud_mask"

    deep_convective_high_cloud_mask = (
        high_cloud_mask & column_connected_mask & (pr >= p_deep)
    ).fillna(False)
    deep_convective_high_cloud_mask.name = "deep_convective_high_cloud_mask"

    near_deep_core = rolling_or_dilated_max(
        deep_convective_high_cloud_mask,
        lat_radius=anvil_radius_deg,
        lon_radius=anvil_radius_deg,
        time_radius=anvil_time_steps,
    ).fillna(False)
    near_deep_core.name = "near_deep_core"

    anvil_high_cloud_mask = (
        high_cloud_mask & ~deep_convective_high_cloud_mask & near_deep_core
    ).fillna(False)
    anvil_high_cloud_mask.name = "anvil_high_cloud_mask"

    cirrus_like_high_cloud_mask = (
        high_cloud_mask & ~column_connected_mask & ~near_deep_core & (pr <= p_weak)
    ).fillna(False)
    cirrus_like_high_cloud_mask.name = "cirrus_like_high_cloud_mask"

    nonprecipitating_high_cloud = (high_cloud_mask & (pr <= p_weak)).fillna(False)
    nonprecipitating_high_cloud.name = "nonprecipitating_high_cloud_mask"

    return {
        "cloud_top_height": cloud_top_height,
        "high_cloud_fraction": high_cloud_fraction,
        "high_cloud_mask": high_cloud_mask,
        "column_connected_mask": column_connected_mask,
        "deep_convective_high_cloud_mask": deep_convective_high_cloud_mask,
        "near_deep_core": near_deep_core,
        "anvil_high_cloud_mask": anvil_high_cloud_mask,
        "cirrus_like_high_cloud_mask": cirrus_like_high_cloud_mask,
        "nonprecipitating_high_cloud_mask": nonprecipitating_high_cloud,
    }


def classification_dataset(classif):
    return xr.Dataset({key: da for key, da in classif.items()})


def get_or_build_classification(ds, fields, sim_name):
    cache_path = os.path.join(classification_cache_dir, f"{sim_name.replace(' ', '_')}_classified_high_cloud.nc")
    if SAVE_CLASSIFICATION_CACHE and os.path.exists(cache_path):
        cached = xr.open_dataset(cache_path, chunks={})
        return {name: cached[name] for name in cached.data_vars}

    cf = get_cloud_fraction(ds)
    if cf is None:
        return None
    valid_times = available_cloud_fraction_times(cf)
    if valid_times is not None:
        if len(valid_times) == 0:
            warnings.warn(f"{sim_name}: cloud fraction has no valid times after reindexing.")
            return None
        cf = cf.sel(Time=valid_times)

    precip = fields["precip"]
    if "Time" in cf.dims and "Time" in precip.dims:
        common_time = cf.Time.to_index().intersection(precip.Time.to_index())
        if len(common_time) == 0:
            warnings.warn(f"{sim_name}: cloud fraction and precipitation have no overlapping times.")
            return None
        print(
            f"{sim_name}: high-cloud classification uses {len(common_time)} "
            f"cloud/precip times (valid cloud times={cf.sizes['Time']}, "
            f"precip-rate times={precip.sizes['Time']})."
        )
        cf = cf.sel(Time=common_time)
        precip = precip.sel(Time=common_time)

    classif = classify_high_cloud(cf, cf.z, precip)

    if SAVE_CLASSIFICATION_CACHE:
        encoding = {name: {"dtype": "float32"} for name in classif if "fraction" in name or "height" in name}
        tmp_path = cache_path + ".tmp"
        classification_dataset(classif).to_netcdf(tmp_path, encoding=encoding)
        os.replace(tmp_path, cache_path)

    return classif


def print_class_coverage(classifications):
    rows = []
    for sim_name, classif in classifications.items():
        if classif is None:
            continue
        for key in (
            "high_cloud_mask",
            "column_connected_mask",
            "deep_convective_high_cloud_mask",
            "anvil_high_cloud_mask",
            "cirrus_like_high_cloud_mask",
            "nonprecipitating_high_cloud_mask",
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
    print(coverage.to_string(index=False))
    if save:
        coverage.to_csv(os.path.join(save_loc, "high_cloud_class_coverage.csv"), index=False)

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


def count_rows_to_frame(rows, stem):
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    print(f"\nSample counts for {stem}")
    print(df.to_string(index=False, max_rows=240))
    if save:
        df.to_csv(os.path.join(save_loc, f"{stem}_bin_counts.csv"), index=False)
    return df


def condition_labels(edges, name):
    return [
        f"low {name}",
        f"mid {name}",
        f"high {name}",
    ]


# -----------------------------------------------------------------------------
# Wave-filter helpers for Figure 3
# -----------------------------------------------------------------------------


def safe_name(x):
    return str(x).replace(" ", "_").replace("/", "-").replace(".", "p").replace("-", "m")


def wave_label(wave_name):
    return WAVE_BOXES[wave_name].get("label", wave_name)


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
        with xr.open_dataset(cache_path) as cached:
            return cached["wave"].load()

    box = WAVE_BOXES[wave_name]
    k0, k1 = box["k"]
    p0, p1 = box["period"]
    h0, h1 = box["depth"]
    filter_name = box.get("filter_name", wave_name)

    da = weighted_lat_mean(fields[field_key], lat_bounds=WAVE_FILTER_LAT_BOUNDS)
    da = da.transpose("Time", "longitude")
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
    return wave


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
    if target_key == "connected":
        return classif["column_connected_mask"].astype(float)
    if target_key == "cirrus":
        return classif["cirrus_like_high_cloud_mask"].astype(float)
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
for i, d in enumerate(data_locs):
    ds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks=latlon_chunks,
        add_cldfrac_file=True,
    )
    ds = standardize_latlon_dataset(ds, needed_vars)

    zds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=zm_latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks=zm_chunks,
        prefer_pattern=True,
        add_cldfrac_file=False,
    )
    zds = standardize_latlon_dataset(zds, needed_vars)

    print(ds)
    print(zds)
    dss.append(ds)
    dss_zm.append(zds)


fields_by_run = []
for ds, zds in zip(dss, dss_zm):
    fields = {"precip": get_precip_rate(ds)}
    for key in ("olr", "pw", "cape", "cin"):
        da = time_aligned_field(ds, key, fields["precip"], required=False)
        if da is not None:
            fields[key] = da

    ttl_fields = compute_ttl_diagnostics(zds, fields["precip"])
    fields.update(ttl_fields)
    fields_by_run.append(fields)


classifications_by_run = {}
if should_run_any([1, 2, 4]):
    for sim_name, ds, fields in zip(sim_names, dss, fields_by_run):
        classifications_by_run[sim_name] = get_or_build_classification(ds, fields, sim_name)
    print_class_coverage(classifications_by_run)


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
#   * If the signal is mainly cirrus-like or anvil high cloud rather than
#     column-connected cloud, do not interpret it as simply deeper precipitating
#     convection.

if should_run_figure(1):
    available_condition = "N2_TTL" if all("N2_TTL" in f for f in fields_by_run) else "pw"
    if not all(available_condition in f for f in fields_by_run):
        print("Skipping Figure 1: neither N2_TTL nor PW is available in all runs.")
    else:
        if available_condition == "N2_TTL":
            condition_label = r"$N^2_{TTL}$"
        else:
            condition_label = "PW"
        cond_edges = pooled_tercile_edges_for_classified(
            fields_by_run, classifications_by_run, available_condition
        )
        precip_bins = pooled_weighted_bins_for_classified(
            fields_by_run, classifications_by_run, "precip", PRECIP_NBINS
        )

        response_specs = [
            ("total high cloud", lambda c: c["high_cloud_fraction"], "High-cloud fraction above 12 km"),
            ("column-connected deep", lambda c: c["deep_convective_high_cloud_mask"].astype(float), "Deep connected high-cloud probability"),
            ("cirrus-like", lambda c: c["cirrus_like_high_cloud_mask"].astype(float), "Cirrus-like high-cloud probability"),
        ]

        fig, ax = plt.subplots(
            len(response_specs), 3,
            figsize=(12.0, 8.0),
            constrained_layout=True,
            sharex=True,
            sharey="row",
        )
        count_rows = []
        labels = condition_labels(cond_edges, condition_label)

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
                    cond_mask = (cond_v >= lo) & (cond_v <= hi if c == 2 else cond_v < hi)
                    centres, mean, count = weighted_binned_mean(
                        pr_v[cond_mask], y_v[cond_mask], precip_bins,
                        weights=weights[cond_mask],
                        min_count=MIN_BIN_COUNT,
                    )
                    ax[r, c].plot(centres, mean, color=cols[i], lw=1.6, label=sim_name if (r == 0 and c == 0) else None)
                    for centre, n in zip(centres, count):
                        count_rows.append(
                            dict(figure=1, simulation=sim_name, response=resp_name,
                                 condition=label, precip_bin_center=centre, sample_count=n)
                        )

                if r == 0:
                    ax[r, c].set_title(f"{label}\n{lo:.3g} to {hi:.3g}")
                if c == 0:
                    ax[r, c].set_ylabel(ylabel)
                if r == len(response_specs) - 1:
                    ax[r, c].set_xlabel(r"Precipitation (mm day$^{-1}$)")
                ax[r, c].grid(alpha=0.2)

        ax[0, 0].legend(loc="best")
        fig.suptitle(f"High cloud vs precipitation conditioned on {condition_label}")
        save_figure(1, fig, "conditional_highcloud_vs_precip")
        count_rows_to_frame(count_rows, "conditional_highcloud_vs_precip")
        print("Done with figure: conditional_highcloud_vs_precip")


# -----------------------------------------------------------------------------
# Figure 2. Cloud-top height vs precipitation, conditioned on PW
# -----------------------------------------------------------------------------
# Interpretation:
#   * Higher cloud tops in u-nudged at fixed precipitation and PW imply altered
#     vertical cloud expression independent of local rain rate and moisture.
#   * If the increase appears only for column-connected cloud, it suggests
#     deeper precipitating convection.
#   * If it appears mainly for non-column-connected cloud, it suggests changes
#     in anvil/cirrus lifetime, detrainment, or upper-level cloud-radiative
#     effects.

if should_run_figure(2):
    if not all("pw" in f for f in fields_by_run):
        print("Skipping Figure 2: PW is unavailable in at least one run.")
    else:
        pw_edges = pooled_tercile_edges_for_classified(fields_by_run, classifications_by_run, "pw")
        precip_bins = pooled_weighted_bins_for_classified(
            fields_by_run, classifications_by_run, "precip", PRECIP_NBINS
        )

        class_specs = [
            ("all high", "high_cloud_mask", "-", "All high cloud"),
            ("column-connected", "column_connected_mask", "--", "Column-connected high cloud"),
            ("cirrus-like", "cirrus_like_high_cloud_mask", ":", "Cirrus-like high cloud"),
        ]

        fig, ax = plt.subplots(2, 3, figsize=(12.0, 6.6), constrained_layout=True, sharex=True)
        count_rows = []
        labels = condition_labels(pw_edges, "PW")

        for c, label in enumerate(labels):
            lo, hi = pw_edges[c], pw_edges[c + 1]
            for i, sim_name in enumerate(sim_names):
                classif = classifications_by_run.get(sim_name)
                if classif is None:
                    continue
                fields = fields_by_run[i]
                vals, weights = sample_matching_arrays(
                    fields["precip"], fields["pw"], classif["cloud_top_height"], classif["high_cloud_mask"].astype(float),
                    classif["column_connected_mask"].astype(float), classif["cirrus_like_high_cloud_mask"].astype(float),
                    lat_bounds=TROPICAL_LAT_BOUNDS,
                    **PLOT_SAMPLE_LIMITS,
                )
                pr_v, pw_v, cth_v, high_v, conn_v, cirr_v = vals
                cond_mask = (pw_v >= lo) & (pw_v <= hi if c == 2 else pw_v < hi)

                masks = {
                    "high_cloud_mask": high_v >= 0.5,
                    "column_connected_mask": conn_v >= 0.5,
                    "cirrus_like_high_cloud_mask": cirr_v >= 0.5,
                }
                for class_label, mask_key, linestyle, _pretty in class_specs:
                    valid = cond_mask & masks[mask_key]
                    centres, mean, count = weighted_binned_mean(
                        pr_v[valid], cth_v[valid], precip_bins,
                        weights=weights[valid],
                        min_count=MIN_BIN_COUNT,
                    )
                    ax[0, c].plot(
                        centres, mean, color=cols[i], ls=linestyle, lw=1.5,
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
                    ("column-connected fraction", conn_v >= 0.5, "--"),
                    ("cirrus-like fraction", cirr_v >= 0.5, ":"),
                ]:
                    centres, frac, count = weighted_binned_fraction(
                        pr_v, numerator, precip_bins,
                        denominator=high_den,
                        weights=weights,
                        min_count=MIN_BIN_COUNT,
                    )
                    ax[1, c].plot(
                        centres, frac, color=cols[i], ls=linestyle, lw=1.5,
                        label=f"{sim_name}, {frac_label}" if c == 0 else None,
                    )
                    for centre, n in zip(centres, count):
                        count_rows.append(
                            dict(figure=2, panel="class_fraction", simulation=sim_name,
                                 class_name=frac_label, pw_condition=label,
                                 precip_bin_center=centre, sample_count=n)
                        )

            ax[0, c].set_title(f"{label}\n{lo:.3g} to {hi:.3g} kg m$^{{-2}}$")
            ax[1, c].set_xlabel(r"Precipitation (mm day$^{-1}$)")
            for rr in (0, 1):
                ax[rr, c].grid(alpha=0.2)

        ax[0, 0].set_ylabel("Mean cloud-top height (km)")
        ax[1, 0].set_ylabel("Fraction of high-cloud samples")
        ax[0, 0].legend(loc="best", fontsize=7)
        ax[1, 0].legend(loc="best", fontsize=7)
        fig.suptitle("Cloud-top height and high-cloud class fractions at fixed PW")
        save_figure(2, fig, "conditional_cloudtop_vs_precip_pw")
        count_rows_to_frame(count_rows, "conditional_cloudtop_vs_precip_pw")
        print("Done with figure: conditional_cloudtop_vs_precip_pw")


# -----------------------------------------------------------------------------
# Figure 3. Precipitation vs PW split by wave amplitude
# -----------------------------------------------------------------------------
# Interpretation:
#   * If u-nudged changes the precipitation-PW curve mainly during strong wave
#     activity, this supports a wind/wave-mediated organisation mechanism.
#   * If T-nudged behaves similarly during strong wave activity, the wave
#     response may be mediated through thermal/static-stability changes.
#   * If u-nudged changes wave-active convection while T-nudged does not, that
#     is strong evidence for a wind/shear/wave-filtering pathway.

if should_run_figure(3):
    if kf_filter is None:
        print("Skipping Figure 3: waveFilter.kf_filter is unavailable.")
    elif not all("pw" in f for f in fields_by_run):
        print("Skipping Figure 3: PW is unavailable in at least one run.")
    else:
        pw_bins = pooled_weighted_bins(fields_by_run, "pw", PW_NBINS)
        fig, ax = plt.subplots(
            2, len(WAVE_BANDS_FOR_FIG3),
            figsize=(5.2 * len(WAVE_BANDS_FOR_FIG3), 6.2),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        if len(WAVE_BANDS_FOR_FIG3) == 1:
            ax = ax[:, None]
        count_rows = []

        for wj, wave_name in enumerate(WAVE_BANDS_FOR_FIG3):
            amp_by_run = []
            for i, fields in enumerate(fields_by_run):
                field_key = WAVE_FILTER_FIELD if WAVE_FILTER_FIELD in fields else "precip"
                try:
                    wave = filter_latmean_wave(fields, i, field_key, wave_name)
                    amp = np.abs(wave)
                    std = float(amp.std(dim=("Time", "longitude"), skipna=True).values)
                    if std > 0:
                        amp = amp / std
                    amp_by_run.append(amp)
                except Exception as exc:
                    print(f"Skipping {sim_names[i]} {wave_name} wave split: {exc}")
                    amp_by_run.append(None)

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

            for i, (fields, amp) in enumerate(zip(fields_by_run, amp_by_run)):
                if amp is None:
                    continue
                vals, weights = sample_matching_arrays(
                    fields["pw"], fields["precip"], amp,
                    lat_bounds=TROPICAL_LAT_BOUNDS,
                    **PLOT_SAMPLE_LIMITS,
                )
                pw_v, pr_v, amp_v = vals
                split_specs = [
                    ("weak", amp_v <= amp_edges[1], 0),
                    ("strong", amp_v >= amp_edges[2], 1),
                ]
                for split_label, mask, row in split_specs:
                    centres, mean, count = weighted_binned_mean(
                        pw_v[mask], pr_v[mask], pw_bins,
                        weights=weights[mask],
                        min_count=MIN_BIN_COUNT,
                    )
                    ax[row, wj].plot(centres, mean, color=cols[i], lw=1.6, label=sim_names[i] if wj == 0 else None)
                    for centre, n in zip(centres, count):
                        count_rows.append(
                            dict(figure=3, simulation=sim_names[i], wave=wave_name,
                                 amplitude_split=split_label, pw_bin_center=centre, sample_count=n)
                        )

            ax[0, wj].set_title(f"{wave_label(wave_name)}\nweak wave amplitude")
            ax[1, wj].set_title("strong wave amplitude")
            for row in (0, 1):
                ax[row, wj].set_xlabel(r"PW (kg m$^{-2}$)")
                ax[row, wj].grid(alpha=0.2)
        ax[0, 0].set_ylabel(r"Precipitation (mm day$^{-1}$)")
        ax[1, 0].set_ylabel(r"Precipitation (mm day$^{-1}$)")
        ax[0, 0].legend(loc="best")
        fig.suptitle("Precipitation-PW relationship split by filtered wave amplitude")
        save_figure(3, fig, "conditional_precip_pw_by_wave_amplitude")
        count_rows_to_frame(count_rows, "conditional_precip_pw_by_wave_amplitude")
        print("Done with figure: conditional_precip_pw_by_wave_amplitude")


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
#   * Residuals in column-connected cloud imply deeper or more vertically
#     connected convection; residuals in cirrus/anvil-like cloud imply changes
#     in detrainment, anvil/cirrus lifetime, or upper-level cloud-radiative
#     feedback rather than simply deeper convective cores.

if should_run_figure(4):
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
            ("connected", "Column-connected high cloud"),
            ("cirrus", "Cirrus-like high cloud"),
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
                    ax[r, c].plot(centres, mean, color=cols[i], lw=1.6, label=sim_name if (r == 0 and c == 0) else None)
                    for centre, n in zip(centres, count):
                        count_rows.append(
                            dict(figure=4, simulation=sim_name, target=target_label,
                                 bin_variable=bin_key, bin_center=centre, sample_count=n)
                        )

                ax[r, c].axhline(0, color="0.6", lw=0.8)
                ax[r, c].grid(alpha=0.2)
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
        print("Done with figure: residual_highcloud_thermo_controls")
