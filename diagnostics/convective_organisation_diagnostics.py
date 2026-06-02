#!/usr/bin/env python
# coding: utf-8

"""
Convective organisation diagnostics for MPAS-A nudging experiments.

The diagnostics here separate four related questions:

  * Lorg: instantaneous spatial arrangement/aggregation of convective objects.
  * Object size metrics: whether apparent organisation changes are actually
    changes in object size, count, or convective area.
  * Conditional Lorg: whether organisation differs between simulations at a
    fixed tropical-mean precipitation or fixed convective area fraction.
  * Persistence: lifetime/coherence of equatorial convective or OLR anomalies.

The Lorg implementation follows the all-neighbour-distance idea of Biagioli and
Tompkins (2023): object centroids are converted to a Ripley/Besag L function,
then compared with a random reference with the same object count and domain.
Longitude is periodic both when labelling objects across the seam and when
measuring distances between centroids.
"""

import os
import sys
import warnings

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (
    _THIS_DIR,
    os.path.abspath(os.path.join(_THIS_DIR, "..")),
    os.path.abspath(os.path.join(_THIS_DIR, "../..")),
    os.path.abspath(os.path.join(_THIS_DIR, "../../..")),
):
    if _path not in sys.path:
        sys.path.append(_path)

import dask
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy import ndimage

from diagnostic_helpers import *


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = "/home/cwp29/tmp/mpas/crf_on/final/organisation/"

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: Lorg time series
#   2: Lorg PDF and boxplot
#   3: object count/area/area-fraction distributions
#   4: Lorg conditioned on tropical-mean precipitation
#   5: Lorg conditioned on convective area fraction
#   6: persistence/decorrelation curves
figures_to_run = []
figsave = None

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control", "u_nudge", "T_nudge"]
latlon_file = ["latlon.2000-*.nc", "latlon.2000-*.nc", "latlon.2000-*.nc"]
sim_names = ["control", "nudged u", "nudged T"]
cols = ["k", "b", "r"]

ndays = 200
dt = 6
analysis_end_date = "2000-11-21"
time_format = "time"

TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
PERSISTENCE_LAT_BOUNDS = (-10.0, 10.0)
EARTH_RADIUS_KM = 6371.0
LONGITUDE_PERIOD_DEG = 360.0

# Thresholds are deliberately near the top of the file so sensitivity tests can
# switch between p95/p99 precipitation or cold-OLR anomaly objects without
# changing the object-identification or Lorg code.
PRECIP_OBJECT_SPECS = [
    dict(name="precip_p95", label="Precip p95", percentile=95.0, sense="above"),
    dict(name="precip_p99", label="Precip p99", percentile=99.0, sense="above"),
]
OLR_OBJECT_SPECS = [
    dict(name="olr_cold_p05", label="Cold OLR anomaly p5", percentile=5.0, sense="below"),
]

ENABLE_OLR_OBJECTS = True
ENABLE_CLOUD_OBJECTS = True
ENABLE_FALLBACK_CLOUD_CLASSIFICATION = True
SAVE_CLOUD_CLASSIFICATION_CACHE = False

# If convective_pathway_diagnostics.py has written classified high-cloud masks,
# this script uses those masks rather than rebuilding the decomposition.
CLOUD_CLASSIFICATION_CACHE_DIRS = [
    "/home/cwp29/tmp/mpas/crf_on/final/convective_pathways/classification_cache",
    os.path.join(save_loc, "classification_cache"),
]

CLOUD_OBJECT_KEYS = [
    "high_cloud_mask",
    "column_connected_mask",
    "deep_convective_high_cloud_mask",
    "anvil_high_cloud_mask",
    "cirrus_like_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
]

# Fallback cloud decomposition thresholds. These mirror the definitions used in
# convective_pathway_diagnostics.py and equatorial_wave_diagnostics.py. They are
# only used if no precomputed/cached cloud-type masks are available.
CF_THRESH = 0.05
Z_HIGH = 12.0
Z_LOW_MIN = 2.0
Z_LOW_MAX = 5.0
Z_MID_MIN = 5.0
Z_MID_MAX = 10.0
Z_UPPER_MIN = 10.0
P_DEEP = 5.0
P_WEAK = 1.0
MAX_CLEAR_GAP = 3.0
ANVIL_RADIUS_DEG = 5.0
ANVIL_TIME_STEPS = 1

MIN_OBJECT_CELLS = 1
MIN_OBJECT_AREA_KM2 = 0.0
PAIR_DISTANCE_NBINS = 80
REFERENCE_REALIZATIONS = 96
REFERENCE_RANDOM_SEED = 202405
MIN_OBJECTS_FOR_LORG = 2
MIN_CONDITION_COUNT = 3
CONDITION_NBINS = 4
PERSISTENCE_MAX_LAG_DAYS = 12.0

THRESHOLD_SAMPLE_LIMITS = dict(max_time=160, max_lat=90, max_lon=180)

VAR_ALIASES = {
    "precip": ["precip", "precip_rate", "pr", "rain_rate"],
    "rainc": ["rainc"],
    "rainnc": ["rainnc"],
    "i_rainc": ["i_rainc"],
    "i_rainnc": ["i_rainnc"],
    "olr": ["olr", "olrtoa", "OLR"],
    "pw": ["pw", "precipw", "prw"],
    "cldfrac": ["cldfrac", "cloud_fraction", "cl"],
    "zgrid": ["zgrid"],
    "high_cloud_mask": ["high_cloud_mask"],
    "column_connected_mask": [
        "column_connected_mask",
        "column_connected_high_cloud_mask",
    ],
    "deep_convective_high_cloud_mask": ["deep_convective_high_cloud_mask"],
    "anvil_high_cloud_mask": ["anvil_high_cloud_mask"],
    "cirrus_like_high_cloud_mask": ["cirrus_like_high_cloud_mask"],
    "nonprecipitating_high_cloud_mask": ["nonprecipitating_high_cloud_mask"],
}

SOURCE_LABELS = {
    "precip_p95": "Precip p95",
    "precip_p99": "Precip p99",
    "olr_cold_p05": "Cold OLR anomaly p5",
    "high_cloud_mask": "High cloud",
    "column_connected_mask": "Column-connected high cloud",
    "deep_convective_high_cloud_mask": "Deep convective high cloud",
    "anvil_high_cloud_mask": "Anvil high cloud",
    "cirrus_like_high_cloud_mask": "Cirrus-like high cloud",
    "nonprecipitating_high_cloud_mask": "Nonprecipitating high cloud",
}

SOURCE_ORDER = [
    "precip_p95",
    "precip_p99",
    "olr_cold_p05",
    "deep_convective_high_cloud_mask",
    "column_connected_mask",
    "high_cloud_mask",
    "anvil_high_cloud_mask",
    "cirrus_like_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
]

needed_vars = sorted(
    set(
        VAR_ALIASES["rainc"]
        + VAR_ALIASES["rainnc"]
        + VAR_ALIASES["i_rainc"]
        + VAR_ALIASES["i_rainnc"]
        + VAR_ALIASES["precip"]
        + VAR_ALIASES["olr"]
        + VAR_ALIASES["pw"]
        + VAR_ALIASES["cldfrac"]
        + VAR_ALIASES["zgrid"]
        + sum((VAR_ALIASES[key] for key in CLOUD_OBJECT_KEYS), [])
    )
)

latlon_chunks = {"Time": 8, "latitude": 90, "longitude": 180}
dask.config.set({"array.slicing.split_large_chunks": True})


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


def save_figure(number, fig, stem, dpi=300):
    global figsave
    if figsave is None:
        figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)
    return figsave.save(fig, stem, number=number, dpi=dpi)


# -----------------------------------------------------------------------------
# Basic data helpers
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


def get_precip_rate(ds):
    direct = find_var(ds, "precip")
    if direct is not None:
        pr = finite_or_nan(ds[direct])
        pr.name = "precip"
        pr.attrs.setdefault("units", "mm day-1")
        return pr

    accum = total_precip_accum(ds)
    pr = precip_rate_from_accum(accum, dt=dt)
    pr = finite_or_nan(pr)
    pr.name = "precip"
    pr.attrs["units"] = "mm day-1"
    return pr


def select_common_time(da, ref):
    if "Time" not in da.dims or "Time" not in ref.dims:
        return da
    common_time = da.Time.to_index().intersection(ref.Time.to_index())
    return da.sel(Time=common_time)


def time_aligned_field(ds, key, ref, required=False):
    name = find_var(ds, key, required=required)
    if name is None:
        return None
    out = finite_or_nan(ds[name])
    out = select_common_time(out, ref)
    out.name = key
    return out


def time_days_from_coord(time_coord):
    values = np.asarray(time_coord.values)
    if np.issubdtype(values.dtype, np.datetime64):
        return (values - values[0]) / np.timedelta64(1, "D")
    return values.astype(float) * dt / 24.0


def median_time_step_days(da):
    values = np.asarray(da.Time.values)
    if values.size < 2:
        return dt / 24.0
    if np.issubdtype(values.dtype, np.datetime64):
        return float(np.median(np.diff(values) / np.timedelta64(1, "D")))
    return float(np.median(np.diff(values))) * dt / 24.0


def weighted_tropical_mean(da, lat_bounds=TROPICAL_LAT_BOUNDS):
    x = subset_lat(finite_or_nan(da), lat_bounds=lat_bounds)
    dims = [dim for dim in ("latitude", "longitude") if dim in x.dims]
    if "latitude" in x.dims:
        weights = np.cos(np.deg2rad(x.latitude))
        return x.weighted(weights).mean(dim=dims, skipna=True)
    return x.mean(dim=dims, skipna=True)


def stride_for_limits(da, limits):
    dim_names = {"max_time": "Time", "max_lat": "latitude", "max_lon": "longitude"}
    indexers = {}
    for dim, max_size in limits.items():
        dim_name = dim_names.get(dim, dim)
        if dim_name in da.dims and da.sizes[dim_name] > max_size:
            step = int(np.ceil(da.sizes[dim_name] / max_size))
            indexers[dim_name] = slice(0, None, step)
    return da.isel(indexers) if indexers else da


def sampled_values(da, lat_bounds=TROPICAL_LAT_BOUNDS, limits=THRESHOLD_SAMPLE_LIMITS):
    x = subset_lat(finite_or_nan(da), lat_bounds=lat_bounds)
    x = stride_for_limits(x, limits)
    vals = np.asarray(x.values).ravel()
    return vals[np.isfinite(vals)]


def source_label(source_name):
    return SOURCE_LABELS.get(source_name, source_name.replace("_", " "))


def ordered_sources(metrics_df):
    present = set(metrics_df.source.unique())
    ordered = [name for name in SOURCE_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "-").replace(".", "p").replace("-", "m")


# -----------------------------------------------------------------------------
# Grid/object geometry
# -----------------------------------------------------------------------------


def coordinate_edges(values):
    vals = np.asarray(values, dtype=float)
    if vals.size == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5], dtype=float)
    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (vals[:-1] + vals[1:])
    edges[0] = vals[0] - 0.5 * (vals[1] - vals[0])
    edges[-1] = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return edges


def grid_cell_area_km2(lat, lon):
    lat_edges = np.clip(coordinate_edges(lat), -90.0, 90.0)
    lon_edges = coordinate_edges(lon)
    dlon = np.abs(np.diff(lon_edges))
    lat_factor = np.diff(np.sin(np.deg2rad(lat_edges)))
    area = (EARTH_RADIUS_KM ** 2) * np.deg2rad(dlon)[None, :] * np.abs(lat_factor)[:, None]
    return xr.DataArray(
        area,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="cell_area",
        attrs={"units": "km2"},
    )


def domain_area_km2(lat_bounds, lon_width=LONGITUDE_PERIOD_DEG):
    lo, hi = sorted(lat_bounds)
    return (
        EARTH_RADIUS_KM ** 2
        * np.deg2rad(lon_width)
        * (np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo)))
    )


def haversine_distances_km(lat1, lon1, lat2, lon2, lon_period=LONGITUDE_PERIOD_DEG):
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    lon1 = np.asarray(lon1, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    dlat = lat2 - lat1
    dlon_deg = lon2 - lon1
    dlon_deg = (dlon_deg + 0.5 * lon_period) % lon_period - 0.5 * lon_period
    dlon = np.deg2rad(dlon_deg)
    a = (
        np.sin(0.5 * dlat) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * dlon) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def max_domain_distance_km(lat_bounds=TROPICAL_LAT_BOUNDS):
    lo, hi = sorted(lat_bounds)
    lat_samples = np.linspace(lo, hi, 9)
    lon_samples = np.array([0.0, 90.0, 180.0, 270.0])
    lat_grid, lon_grid = np.meshgrid(lat_samples, lon_samples, indexing="ij")
    lat = lat_grid.ravel()
    lon = lon_grid.ravel()
    d = pairwise_distances_km(lat, lon)
    return float(np.nanmax(d)) if d.size else np.nan


def pairwise_distances_km(lat, lon, block=512):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = lat.size
    if n < 2:
        return np.array([], dtype=float)

    chunks = []
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        for j0 in range(i0, n, block):
            j1 = min(j0 + block, n)
            d = haversine_distances_km(
                lat[i0:i1, None],
                lon[i0:i1, None],
                lat[None, j0:j1],
                lon[None, j0:j1],
            )
            if i0 == j0:
                ii = np.triu_indices(i1 - i0, k=1)
                chunks.append(d[ii])
            else:
                chunks.append(d.ravel())
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)


def merge_periodic_longitude_labels(labels):
    """Merge connected components touching both sides of a periodic longitude seam."""
    labels = np.asarray(labels, dtype=np.int32)
    max_label = int(labels.max())
    if max_label == 0 or labels.shape[1] < 2:
        return labels

    parent = np.arange(max_label + 1, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a <= 0 or b <= 0:
            return
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra

    nlat = labels.shape[0]
    for j in range(nlat):
        left = labels[j, 0]
        if left <= 0:
            continue
        for jj in range(max(0, j - 1), min(nlat, j + 2)):
            union(left, labels[jj, -1])

    roots = np.array([find(i) for i in range(max_label + 1)], dtype=np.int32)
    unique_roots = [root for root in np.unique(roots[1:]) if root > 0]
    remap = {root: index + 1 for index, root in enumerate(unique_roots)}
    out = np.zeros_like(labels)
    positive = labels > 0
    out[positive] = np.array([remap[roots[label]] for label in labels[positive]], dtype=np.int32)
    return out


def object_properties_from_mask(mask2d, area2d):
    """
    Label one thresholded scene and return object centroid/area properties.

    The object mask uses 8-connectivity and merges objects across periodic
    longitude. This is the only new object-identification step in this script;
    the masks themselves come from existing precipitation/cloud diagnostics
    wherever available.
    """
    mask2d = mask2d.transpose("latitude", "longitude")
    area2d = area2d.sel(latitude=mask2d.latitude, longitude=mask2d.longitude)
    mask_vals = np.asarray(mask2d.fillna(False).values, dtype=bool)
    area_vals = np.asarray(area2d.values, dtype=float)

    if not np.any(mask_vals):
        return pd.DataFrame(columns=["label", "cell_count", "area_km2", "centroid_lat", "centroid_lon"])

    structure = np.ones((3, 3), dtype=np.int8)
    labels, _ = ndimage.label(mask_vals, structure=structure)
    labels = merge_periodic_longitude_labels(labels)

    lat_vals = np.asarray(mask2d.latitude.values, dtype=float)
    lon_vals = np.asarray(mask2d.longitude.values, dtype=float) % LONGITUDE_PERIOD_DEG
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")

    rows = []
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        obj = labels == label_id
        cell_count = int(np.count_nonzero(obj))
        area = float(np.nansum(area_vals[obj]))
        if cell_count < MIN_OBJECT_CELLS or area < MIN_OBJECT_AREA_KM2:
            continue

        weights = area_vals[obj]
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
        centroid_lat = float(np.average(lat_grid[obj], weights=weights))
        theta = 2.0 * np.pi * lon_grid[obj] / LONGITUDE_PERIOD_DEG
        sin_mean = np.average(np.sin(theta), weights=weights)
        cos_mean = np.average(np.cos(theta), weights=weights)
        centroid_lon = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0)

        rows.append(
            dict(
                label=int(label_id),
                cell_count=cell_count,
                area_km2=area,
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
            )
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Lorg metric
# -----------------------------------------------------------------------------


def ripley_besag_l(pair_distances, n_objects, domain_area, r_edges):
    pair_distances = np.asarray(pair_distances, dtype=float)
    if n_objects < 2 or pair_distances.size == 0:
        return np.full(r_edges.size - 1, np.nan, dtype=float)

    counts, _ = np.histogram(pair_distances, bins=r_edges)
    cumulative_pairs = np.cumsum(counts).astype(float)
    k_r = domain_area * (2.0 * cumulative_pairs) / (n_objects * (n_objects - 1))
    return np.sqrt(np.maximum(k_r, 0.0) / np.pi)


def random_centroids(n_objects, lat_bounds=TROPICAL_LAT_BOUNDS, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = sorted(lat_bounds)
    sin_lo = np.sin(np.deg2rad(lo))
    sin_hi = np.sin(np.deg2rad(hi))
    lat = np.rad2deg(np.arcsin(sin_lo + rng.random(n_objects) * (sin_hi - sin_lo)))
    lon = rng.random(n_objects) * LONGITUDE_PERIOD_DEG
    return lat, lon


def reference_l_function(n_objects, domain_area, r_edges, rng, cache):
    key = (int(n_objects), int(PAIR_DISTANCE_NBINS), tuple(np.round(r_edges, 6)))
    if key in cache:
        return cache[key]

    values = []
    for _ in range(REFERENCE_REALIZATIONS):
        lat, lon = random_centroids(n_objects, rng=rng)
        d = pairwise_distances_km(lat, lon)
        values.append(ripley_besag_l(d, n_objects, domain_area, r_edges))

    ref = np.nanmean(np.vstack(values), axis=0)
    cache[key] = ref
    return ref


def lorg_from_objects(objects, domain_area, r_edges, rng, reference_cache):
    n_objects = len(objects)
    if n_objects < MIN_OBJECTS_FOR_LORG:
        return np.nan, np.nan

    distances = pairwise_distances_km(objects.centroid_lat.values, objects.centroid_lon.values)
    l_obs = ripley_besag_l(distances, n_objects, domain_area, r_edges)
    l_ref = reference_l_function(n_objects, domain_area, r_edges, rng, reference_cache)
    r = r_edges[1:]
    rmax = float(r_edges[-1])
    lorg_km = float(np.trapz(l_obs - l_ref, r) / rmax)
    lorg_normalised = float(lorg_km / rmax)
    return lorg_km, lorg_normalised


def compute_organisation_timeseries(mask, fields, source_name, rng, reference_cache):
    mask = subset_lat(mask.fillna(False), lat_bounds=TROPICAL_LAT_BOUNDS)
    if "Time" not in mask.dims:
        warnings.warn(f"Skipping {source_name}: mask has no Time dimension")
        return pd.DataFrame()

    precip_mean = fields["mean_precip"].sel(Time=mask.Time)
    mask, precip_mean = xr.align(mask, precip_mean, join="inner")
    area2d = grid_cell_area_km2(mask.latitude.values, mask.longitude.values)
    domain_area = float(area2d.sum().values)
    r_edges = np.linspace(0.0, max_domain_distance_km(TROPICAL_LAT_BOUNDS), PAIR_DISTANCE_NBINS + 1)
    time_days = time_days_from_coord(mask.Time)

    rows = []
    for it in range(mask.sizes["Time"]):
        mask2d = mask.isel(Time=it).load()
        objects = object_properties_from_mask(mask2d, area2d)
        object_count = int(len(objects))
        object_area = float(objects.area_km2.sum()) if object_count else 0.0
        largest_area = float(objects.area_km2.max()) if object_count else np.nan
        mean_area = float(objects.area_km2.mean()) if object_count else np.nan
        median_area = float(objects.area_km2.median()) if object_count else np.nan
        largest_object_fraction = largest_area / object_area if object_area > 0.0 else np.nan
        convective_area_fraction = object_area / domain_area if domain_area > 0.0 else np.nan
        lorg_km, lorg_normalised = lorg_from_objects(objects, domain_area, r_edges, rng, reference_cache)

        rows.append(
            dict(
                source=source_name,
                source_label=source_label(source_name),
                Time=mask.Time.values[it],
                time_days=float(time_days[it]),
                lorg_km=lorg_km,
                lorg_normalised=lorg_normalised,
                object_count=object_count,
                mean_object_area_km2=mean_area,
                median_object_area_km2=median_area,
                largest_object_area_km2=largest_area,
                largest_object_fraction=largest_object_fraction,
                convective_area_fraction=convective_area_fraction,
                tropical_mean_precip=float(precip_mean.isel(Time=it).values),
            )
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Cloud-type masks
# -----------------------------------------------------------------------------


def existing_cloud_masks_from_dataset(ds, precip_ref):
    masks = {}
    for key in CLOUD_OBJECT_KEYS:
        name = find_var(ds, key, required=False)
        if name is None:
            continue
        da = ds[name]
        da = select_common_time(da, precip_ref)
        masks[key] = da.fillna(False).astype(bool)
    return masks


def cached_classification_path(sim_name):
    filename = f"{sim_name.replace(' ', '_')}_classified_high_cloud.nc"
    for directory in CLOUD_CLASSIFICATION_CACHE_DIRS:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return None


def cloud_masks_from_cache(sim_name, precip_ref):
    path = cached_classification_path(sim_name)
    if path is None:
        return {}

    print(f"Using cached high-cloud classification for {sim_name}: {path}")
    cached = xr.open_dataset(path, chunks={})
    masks = {}
    for key in CLOUD_OBJECT_KEYS:
        if key not in cached.data_vars:
            continue
        da = cached[key]
        da = select_common_time(da, precip_ref)
        masks[key] = da.fillna(False).astype(bool)
    return masks


def get_cloud_fraction(ds):
    name = find_var(ds, "cldfrac", required=False)
    if name is None:
        return None
    cf = finite_or_nan(ds[name])
    zdim = vertical_dim(cf)
    if zdim != "z":
        cf = cf.rename({zdim: "z"})
    if "z" not in cf.coords:
        warnings.warn("Cloud fraction has no usable z coordinate; cloud-object Lorg will be skipped.")
        return None
    zvals = np.asarray(cf.z.values, dtype=float)
    if zvals.size != cf.sizes["z"] or not np.all(np.isfinite(zvals)):
        warnings.warn("Cloud fraction z coordinate is not usable; cloud-object Lorg will be skipped.")
        return None
    if np.nanmax(zvals) < 100.0:
        cf = cf.assign_coords(z=cf.z * 1000.0)
    return cf


def available_cloud_fraction_times(cf):
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
    x = mask.astype(np.int8)
    rolling_kwargs = {}
    if "Time" in x.dims and time_radius > 0:
        rolling_kwargs["Time"] = 2 * int(time_radius) + 1
    if "latitude" in x.dims and lat_radius > 0 and x.sizes.get("latitude", 0) > 1:
        dlat = float(np.nanmedian(np.abs(np.diff(x.latitude.values))))
        rolling_kwargs["latitude"] = 2 * int(np.ceil(lat_radius / dlat)) + 1
    if "longitude" in x.dims and lon_radius > 0 and x.sizes.get("longitude", 0) > 1:
        dlon = float(np.nanmedian(np.abs(np.diff(x.longitude.values))))
        rolling_kwargs["longitude"] = 2 * int(np.ceil(lon_radius / dlon)) + 1

    if not rolling_kwargs:
        return mask
    return x.rolling(**rolling_kwargs, center=True, min_periods=1).max() >= 1


def classify_high_cloud_masks(cldfrac, precip):
    """
    Fallback high-cloud decomposition used only when precomputed categories are
    unavailable. The categories mirror convective_pathway_diagnostics.py.
    """
    cf = cldfrac
    valid_times = available_cloud_fraction_times(cf)
    if valid_times is not None:
        cf = cf.sel(Time=valid_times)
    if "Time" in cf.dims and "Time" in precip.dims:
        common_time = cf.Time.to_index().intersection(precip.Time.to_index())
        if len(common_time) == 0:
            return {}
        cf = cf.sel(Time=common_time)
        precip = precip.sel(Time=common_time)
    cf, pr = xr.align(cf, precip, join="inner")
    z_km = z_in_km(cf)

    cloudy = cf >= CF_THRESH
    cloud_top_height = z_km.where(cloudy).max(dim="z", skipna=True)
    high_cloud_fraction = cf.where(z_km >= Z_HIGH).mean(dim="z", skipna=True)
    high_cloud_mask = ((cloud_top_height >= Z_HIGH) | (high_cloud_fraction >= CF_THRESH)).fillna(False)
    high_cloud_mask.name = "high_cloud_mask"

    low_cloud = any_cloud_in_layer(cloudy, z_km, Z_LOW_MIN, Z_LOW_MAX)
    mid_cloud = any_cloud_in_layer(cloudy, z_km, Z_MID_MIN, Z_MID_MAX)
    upper_cloud = any_cloud_in_layer(cloudy, z_km, Z_UPPER_MIN, max(float(z_km.max()), Z_HIGH + 0.1))

    gap_ok = xr.ones_like(high_cloud_mask, dtype=bool)
    if MAX_CLEAR_GAP is not None and MAX_CLEAR_GAP > 0:
        slab_starts = np.arange(Z_LOW_MIN, Z_HIGH, MAX_CLEAR_GAP)
        for z0 in slab_starts:
            z1 = min(z0 + MAX_CLEAR_GAP, max(float(z_km.max()), Z_HIGH))
            if z1 <= z0:
                continue
            gap_ok = gap_ok & any_cloud_in_layer(cloudy, z_km, z0, z1 + 1.0e-6)

    column_connected = (high_cloud_mask & low_cloud & mid_cloud & upper_cloud & gap_ok).fillna(False)
    column_connected.name = "column_connected_high_cloud_mask"

    deep_convective = (high_cloud_mask & column_connected & (pr >= P_DEEP)).fillna(False)
    deep_convective.name = "deep_convective_high_cloud_mask"

    near_deep_core = rolling_or_dilated_max(
        deep_convective,
        lat_radius=ANVIL_RADIUS_DEG,
        lon_radius=ANVIL_RADIUS_DEG,
        time_radius=ANVIL_TIME_STEPS,
    ).fillna(False)

    anvil = (high_cloud_mask & ~deep_convective & near_deep_core).fillna(False)
    anvil.name = "anvil_high_cloud_mask"

    cirrus_like = (high_cloud_mask & ~column_connected & ~near_deep_core & (pr <= P_WEAK)).fillna(False)
    cirrus_like.name = "cirrus_like_high_cloud_mask"

    nonprecipitating = (high_cloud_mask & (pr <= P_WEAK)).fillna(False)
    nonprecipitating.name = "nonprecipitating_high_cloud_mask"

    return {
        "high_cloud_mask": high_cloud_mask,
        "column_connected_mask": column_connected,
        "deep_convective_high_cloud_mask": deep_convective,
        "anvil_high_cloud_mask": anvil,
        "cirrus_like_high_cloud_mask": cirrus_like,
        "nonprecipitating_high_cloud_mask": nonprecipitating,
    }


def maybe_write_cloud_cache(sim_name, masks):
    if not SAVE_CLOUD_CLASSIFICATION_CACHE or not masks:
        return
    cache_dir = CLOUD_CLASSIFICATION_CACHE_DIRS[-1]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{sim_name.replace(' ', '_')}_classified_high_cloud.nc")
    tmp_path = path + ".tmp"
    xr.Dataset(masks).to_netcdf(tmp_path)
    os.replace(tmp_path, path)


def cloud_type_masks(ds, fields, sim_name):
    if not ENABLE_CLOUD_OBJECTS:
        return {}

    masks = existing_cloud_masks_from_dataset(ds, fields["precip"])
    if masks:
        print(f"Using in-dataset cloud-type masks for {sim_name}: {', '.join(sorted(masks))}")
        return masks

    masks = cloud_masks_from_cache(sim_name, fields["precip"])
    if masks:
        return masks

    if not ENABLE_FALLBACK_CLOUD_CLASSIFICATION:
        return {}

    cf = get_cloud_fraction(ds)
    if cf is None:
        return {}

    print(f"Building fallback high-cloud classification for {sim_name}")
    masks = classify_high_cloud_masks(cf, fields["precip"])
    maybe_write_cloud_cache(sim_name, masks)
    return masks


# -----------------------------------------------------------------------------
# Threshold masks and summaries
# -----------------------------------------------------------------------------


def threshold_from_pooled_runs(fields_by_run, field_key, percentile, transform=None):
    vals_all = []
    for fields in fields_by_run:
        if field_key not in fields:
            continue
        da = fields[field_key]
        if transform is not None:
            da = transform(da)
        vals = sampled_values(da)
        if vals.size:
            vals_all.append(vals)
    if not vals_all:
        return np.nan
    return float(np.nanpercentile(np.concatenate(vals_all), percentile))


def build_threshold_table(fields_by_run):
    rows = []
    for spec in PRECIP_OBJECT_SPECS:
        threshold = threshold_from_pooled_runs(fields_by_run, "precip", spec["percentile"])
        rows.append(dict(source=spec["name"], field="precip", threshold=threshold, **spec))

    if ENABLE_OLR_OBJECTS and all("olr" in fields for fields in fields_by_run):
        for spec in OLR_OBJECT_SPECS:
            threshold = threshold_from_pooled_runs(
                fields_by_run,
                "olr",
                spec["percentile"],
                transform=lambda da: da - da.mean(dim="Time", skipna=True),
            )
            rows.append(dict(source=spec["name"], field="olr_anomaly", threshold=threshold, **spec))
    return pd.DataFrame(rows)


def threshold_object_masks(fields, threshold_table):
    masks = {}
    for row in threshold_table.to_dict("records"):
        source = row["source"]
        field = row["field"]
        threshold = row["threshold"]
        sense = row["sense"]
        if not np.isfinite(threshold):
            continue
        if field == "precip":
            da = fields["precip"]
        elif field == "olr_anomaly" and "olr" in fields:
            da = fields["olr"] - fields["olr"].mean(dim="Time", skipna=True)
        else:
            continue

        if sense == "above":
            mask = da >= threshold
        elif sense == "below":
            mask = da <= threshold
        else:
            raise ValueError(f"Unknown threshold sense {sense!r}")
        mask.name = source
        masks[source] = mask.fillna(False)
    return masks


def build_fields(ds):
    precip = get_precip_rate(ds)
    fields = {
        "precip": precip,
        "mean_precip": weighted_tropical_mean(precip, lat_bounds=TROPICAL_LAT_BOUNDS),
    }
    olr = time_aligned_field(ds, "olr", precip, required=False)
    if olr is not None:
        fields["olr"] = olr
    pw = time_aligned_field(ds, "pw", precip, required=False)
    if pw is not None:
        fields["pw"] = pw
    return fields


def count_rows_to_frame(rows, stem):
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    print(f"\nSample counts for {stem}")
    print(df.to_string(index=False, max_rows=240))
    if save:
        df.to_csv(os.path.join(save_loc, f"{stem}_bin_counts.csv"), index=False)
    return df


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def source_axis_grid(sources, ncols=2, height_per_row=2.9):
    nsrc = len(sources)
    ncols = min(ncols, max(1, nsrc))
    nrows = int(np.ceil(nsrc / ncols))
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, height_per_row * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis in ax.ravel()[nsrc:]:
        axis.set_visible(False)
    return fig, ax


def plot_lorg_timeseries(metrics_df):
    sources = ordered_sources(metrics_df)
    fig, ax = source_axis_grid(sources)
    for si, source in enumerate(sources):
        axis = ax.ravel()[si]
        sub_source = metrics_df[metrics_df.source == source]
        for i, sim_name in enumerate(sim_names):
            sub = sub_source[sub_source.simulation == sim_name]
            if len(sub) == 0:
                continue
            axis.plot(sub.time_days, sub.lorg_km, color=cols[i], lw=1.2, label=sim_name)
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.set_title(source_label(source))
        axis.set_xlabel("Time since analysis start (days)")
        axis.set_ylabel("Lorg (km)")
        axis.grid(alpha=0.2)
    ax.ravel()[0].legend(loc="best")
    fig.suptitle("Lorg: instantaneous spatial aggregation from all object-pair distances")
    save_figure(1, fig, "lorg_timeseries")


def plot_lorg_distribution(metrics_df):
    sources = ordered_sources(metrics_df)
    nrows = len(sources)
    fig, ax = plt.subplots(
        nrows,
        2,
        figsize=(10.5, max(3.0, 2.8 * nrows)),
        constrained_layout=True,
        squeeze=False,
    )

    for si, source in enumerate(sources):
        sub_source = metrics_df[metrics_df.source == source]
        vals_all = sub_source.lorg_km.values
        vals_all = vals_all[np.isfinite(vals_all)]
        if vals_all.size >= 2:
            lo = float(np.nanpercentile(vals_all, 2))
            hi = float(np.nanpercentile(vals_all, 98))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                bins = np.linspace(lo, hi, 24)
            else:
                span = max(1.0, abs(lo) * 0.1)
                bins = np.linspace(lo - span, lo + span, 24)
        else:
            bins = np.linspace(-1.0, 1.0, 12)

        box_values = []
        box_labels = []
        for i, sim_name in enumerate(sim_names):
            vals = sub_source.loc[sub_source.simulation == sim_name, "lorg_km"].values
            vals = vals[np.isfinite(vals)]
            if vals.size:
                ax[si, 0].hist(vals, bins=bins, density=True, histtype="step", color=cols[i], lw=1.4, label=sim_name)
            box_values.append(vals if vals.size else np.array([np.nan]))
            box_labels.append(sim_name)

        ax[si, 1].boxplot(
            box_values,
            labels=box_labels,
            showfliers=False,
            medianprops={"color": "k"},
        )
        for tick, color in zip(ax[si, 1].get_xticklabels(), cols):
            tick.set_color(color)

        ax[si, 0].axvline(0.0, color="0.6", lw=0.8)
        ax[si, 1].axhline(0.0, color="0.6", lw=0.8)
        ax[si, 0].set_ylabel(f"{source_label(source)}\nPDF")
        ax[si, 0].set_xlabel("Lorg (km)")
        ax[si, 1].set_ylabel("Lorg (km)")
        ax[si, 0].grid(alpha=0.2)
        ax[si, 1].grid(alpha=0.2, axis="y")

    ax[0, 0].legend(loc="best")
    fig.suptitle("Distribution of Lorg across timesteps")
    save_figure(2, fig, "lorg_pdf_boxplot")


def plot_object_metric_distributions(metrics_df):
    sources = ordered_sources(metrics_df)
    metric_specs = [
        ("object_count", "Object count"),
        ("mean_object_area_km2", "Mean object area (km$^2$)"),
        ("largest_object_fraction", "Largest-object fraction"),
        ("convective_area_fraction", "Convective area fraction"),
    ]
    fig, ax = plt.subplots(
        len(sources),
        len(metric_specs),
        figsize=(4.1 * len(metric_specs), max(3.0, 2.55 * len(sources))),
        constrained_layout=True,
        squeeze=False,
    )
    for si, source in enumerate(sources):
        sub_source = metrics_df[metrics_df.source == source]
        for mj, (metric, label) in enumerate(metric_specs):
            values = []
            for sim_name in sim_names:
                vals = sub_source.loc[sub_source.simulation == sim_name, metric].values
                vals = vals[np.isfinite(vals)]
                values.append(vals if vals.size else np.array([np.nan]))
            ax[si, mj].boxplot(values, labels=sim_names, showfliers=False, medianprops={"color": "k"})
            for tick, color in zip(ax[si, mj].get_xticklabels(), cols):
                tick.set_color(color)
                tick.set_rotation(25)
                tick.set_ha("right")
            if si == 0:
                ax[si, mj].set_title(label)
            if mj == 0:
                ax[si, mj].set_ylabel(source_label(source))
            ax[si, mj].grid(alpha=0.2, axis="y")
    fig.suptitle("Object size/count metrics: size and amount changes")
    save_figure(3, fig, "object_metric_boxplots")


def conditional_summary(metrics_df, source, condition_key):
    sub_source = metrics_df[metrics_df.source == source]
    cond_vals = sub_source[condition_key].values
    cond_vals = cond_vals[np.isfinite(cond_vals)]
    if cond_vals.size < CONDITION_NBINS:
        return pd.DataFrame()
    edges = np.unique(np.nanpercentile(cond_vals, np.linspace(0.0, 100.0, CONDITION_NBINS + 1)))
    if edges.size < 3:
        return pd.DataFrame()

    rows = []
    for bi in range(edges.size - 1):
        lo, hi = edges[bi], edges[bi + 1]
        for sim_name in sim_names:
            sub = sub_source[sub_source.simulation == sim_name]
            if bi == edges.size - 2:
                in_bin = (sub[condition_key] >= lo) & (sub[condition_key] <= hi)
            else:
                in_bin = (sub[condition_key] >= lo) & (sub[condition_key] < hi)
            vals = sub.loc[in_bin, "lorg_km"].values
            vals = vals[np.isfinite(vals)]
            rows.append(
                dict(
                    source=source,
                    source_label=source_label(source),
                    simulation=sim_name,
                    condition_key=condition_key,
                    bin_index=bi,
                    bin_low=lo,
                    bin_high=hi,
                    bin_center=0.5 * (lo + hi),
                    lorg_mean=np.nanmean(vals) if vals.size >= MIN_CONDITION_COUNT else np.nan,
                    lorg_p25=np.nanpercentile(vals, 25) if vals.size >= MIN_CONDITION_COUNT else np.nan,
                    lorg_p75=np.nanpercentile(vals, 75) if vals.size >= MIN_CONDITION_COUNT else np.nan,
                    sample_count=int(vals.size),
                )
            )
    return pd.DataFrame(rows)


def plot_conditional_lorg(metrics_df, condition_key, number, stem, xlabel):
    sources = ordered_sources(metrics_df)
    summaries = [conditional_summary(metrics_df, source, condition_key) for source in sources]
    summary = pd.concat([df for df in summaries if len(df) > 0], ignore_index=True) if summaries else pd.DataFrame()
    if len(summary) == 0:
        print(f"Skipping {stem}: no finite conditional samples")
        return

    if save:
        summary.to_csv(os.path.join(save_loc, f"{stem}.csv"), index=False)

    fig, ax = source_axis_grid(sources)
    for si, source in enumerate(sources):
        axis = ax.ravel()[si]
        sub_source = summary[summary.source == source]
        if len(sub_source) == 0:
            axis.set_visible(False)
            continue
        for i, sim_name in enumerate(sim_names):
            sub = sub_source[sub_source.simulation == sim_name]
            axis.plot(sub.bin_center, sub.lorg_mean, color=cols[i], marker="o", lw=1.5, label=sim_name)
            axis.fill_between(sub.bin_center, sub.lorg_p25, sub.lorg_p75, color=cols[i], alpha=0.12, linewidth=0)
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.set_title(source_label(source))
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Lorg (km)")
        axis.grid(alpha=0.2)
    ax.ravel()[0].legend(loc="best")
    fig.suptitle("Conditional Lorg: organisation compared at fixed convective amount")
    save_figure(number, fig, stem)
    count_rows_to_frame(summary.rename(columns={"sample_count": "count"}).to_dict("records"), stem)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def equatorial_latmean_anomaly(da, lat_bounds=PERSISTENCE_LAT_BOUNDS):
    x = subset_lat(finite_or_nan(da), lat_bounds=lat_bounds)
    if "latitude" in x.dims:
        x = x.weighted(np.cos(np.deg2rad(x.latitude))).mean(dim="latitude", skipna=True)
    if "longitude" not in x.dims:
        return x - x.mean(dim="Time", skipna=True)
    return x - x.mean(dim="Time", skipna=True)


def lag_pattern_autocorrelation(da, max_lag_steps):
    x = np.asarray(da.transpose("Time", ...).load().values, dtype=float)
    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    out = []
    for lag in lags:
        if lag < 0:
            x0 = x[-lag:]
            x1 = x[:lag]
        elif lag > 0:
            x0 = x[:-lag]
            x1 = x[lag:]
        else:
            x0 = x
            x1 = x
        v0 = x0.ravel()
        v1 = x1.ravel()
        ok = np.isfinite(v0) & np.isfinite(v1)
        if np.count_nonzero(ok) < 3:
            out.append(np.nan)
        else:
            out.append(float(np.corrcoef(v0[ok], v1[ok])[0, 1]))
    return lags, np.asarray(out, dtype=float)


def compute_persistence(fields_by_run):
    rows = []
    field_specs = [("precip", "Precipitation")]
    if all("olr" in fields for fields in fields_by_run):
        field_specs.append(("olr", "OLR"))

    for i, fields in enumerate(fields_by_run):
        for field_key, label in field_specs:
            da = equatorial_latmean_anomaly(fields[field_key], lat_bounds=PERSISTENCE_LAT_BOUNDS)
            dt_days = median_time_step_days(da)
            max_lag_steps = max(1, int(round(PERSISTENCE_MAX_LAG_DAYS / dt_days)))
            lags, corr = lag_pattern_autocorrelation(da, max_lag_steps)
            for lag, value in zip(lags, corr):
                rows.append(
                    dict(
                        simulation=sim_names[i],
                        field=field_key,
                        field_label=label,
                        lag_days=float(lag * dt_days),
                        autocorrelation=value,
                    )
                )
    return pd.DataFrame(rows)


def plot_persistence(persistence_df):
    fields = list(persistence_df.field.unique())
    fig, ax = plt.subplots(
        1,
        len(fields),
        figsize=(5.2 * len(fields), 3.8),
        constrained_layout=True,
        squeeze=False,
    )
    for fj, field_key in enumerate(fields):
        axis = ax[0, fj]
        sub_field = persistence_df[persistence_df.field == field_key]
        for i, sim_name in enumerate(sim_names):
            sub = sub_field[sub_field.simulation == sim_name]
            axis.plot(sub.lag_days, sub.autocorrelation, color=cols[i], lw=1.6, label=sim_name)
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.axvline(0.0, color="0.6", lw=0.8)
        axis.set_title(sub_field.field_label.iloc[0])
        axis.set_xlabel("Lag (days)")
        axis.set_ylabel("Pattern autocorrelation")
        axis.grid(alpha=0.2)
    ax[0, 0].legend(loc="best")
    fig.suptitle(
        f"Persistence: equatorial anomaly decorrelation over {PERSISTENCE_LAT_BOUNDS[0]:g} to {PERSISTENCE_LAT_BOUNDS[1]:g} deg"
    )
    save_figure(6, fig, "equatorial_persistence_autocorrelation")


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def load_datasets():
    dss = []
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
        print(ds)
        dss.append(ds)
    return dss


def build_all_metrics(dss, fields_by_run, threshold_table):
    rng = np.random.default_rng(REFERENCE_RANDOM_SEED)
    reference_cache = {}
    all_rows = []

    for i, (ds, fields) in enumerate(zip(dss, fields_by_run)):
        sim_name = sim_names[i]
        masks = threshold_object_masks(fields, threshold_table)
        masks.update(cloud_type_masks(ds, fields, sim_name))

        for source_name in [name for name in SOURCE_ORDER if name in masks]:
            print(f"Computing Lorg metrics for {sim_name}: {source_name}")
            metrics = compute_organisation_timeseries(
                masks[source_name],
                fields,
                source_name,
                rng,
                reference_cache,
            )
            if len(metrics) == 0:
                continue
            metrics["simulation"] = sim_name
            all_rows.append(metrics)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def main():
    os.makedirs(save_loc, exist_ok=True)
    dss = load_datasets()
    fields_by_run = [build_fields(ds) for ds in dss]

    threshold_table = build_threshold_table(fields_by_run)
    if len(threshold_table) > 0:
        print("\nOrganisation object thresholds")
        print(threshold_table.to_string(index=False))
        if save:
            threshold_table.to_csv(os.path.join(save_loc, "organisation_object_thresholds.csv"), index=False)

    metrics_df = build_all_metrics(dss, fields_by_run, threshold_table)
    if len(metrics_df) == 0:
        print("No organisation metrics were computed.")
        return

    print("\nOrganisation metric summary")
    summary = (
        metrics_df
        .groupby(["source", "source_label", "simulation"], as_index=False)
        [["lorg_km", "object_count", "convective_area_fraction"]]
        .mean(numeric_only=True)
    )
    print(summary.to_string(index=False))

    if save:
        metrics_df.to_csv(os.path.join(save_loc, "organisation_lorg_timeseries.csv"), index=False)
        summary.to_csv(os.path.join(save_loc, "organisation_lorg_summary.csv"), index=False)

    if should_run_figure(1):
        plot_lorg_timeseries(metrics_df)
    if should_run_figure(2):
        plot_lorg_distribution(metrics_df)
    if should_run_figure(3):
        plot_object_metric_distributions(metrics_df)
    if should_run_figure(4):
        plot_conditional_lorg(
            metrics_df,
            "tropical_mean_precip",
            4,
            "conditional_lorg_by_tropical_mean_precip",
            r"Tropical-mean precipitation (mm day$^{-1}$)",
        )
    if should_run_figure(5):
        plot_conditional_lorg(
            metrics_df,
            "convective_area_fraction",
            5,
            "conditional_lorg_by_convective_area_fraction",
            "Convective area fraction",
        )

    if should_run_figure(6):
        persistence_df = compute_persistence(fields_by_run)
        if save and len(persistence_df) > 0:
            persistence_df.to_csv(os.path.join(save_loc, "equatorial_persistence_autocorrelation.csv"), index=False)
        if len(persistence_df) > 0:
            plot_persistence(persistence_df)

    print("\nFinished convective_organisation_diagnostics.py")


if __name__ == "__main__":
    main()
