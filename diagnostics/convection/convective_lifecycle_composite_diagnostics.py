#!/usr/bin/env python
# coding: utf-8

"""
Convective lifecycle composites centred on tracked organised convective systems.

This diagnostic requires the PyFLEXTRKR output generated from the shared
organised convective-system envelope product. Its event set is the first
accepted appearance of those envelope tracks, while embedded
convective/stratiform/non-precipitating high-cloud fractions remain composite
diagnostics sampled inside and around the tracked systems.

# WHAT TO LOOK FOR:
#   * In the lag-longitude Hovmollers, a moisture signal that peaks earlier or
#     later than lag 0 diagnoses preconditioning timing rather than only mean PW.
#   * TTL N2 and shear anomalies before lag 0 separate static-stability and wind
#     pathways for the T-nudged and U-nudged experiments.
#   * Post-onset precipitation, cloud-top, and high-cloud persistence diagnose
#     whether nudging changes storm longevity rather than just onset amplitude.
#   * Track-following Kelvin, easterly-wave, and ER composites show whether
#     convection samples different moisture, circulation, and cloud anomalies.
"""

import glob
import gc
import hashlib
import json
import os
import sys
import warnings
from functools import partial

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (_THIS_DIR,):
    if _path not in sys.path:
        sys.path.append(_path)
for _rel_path in ("..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import dask
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from diagnostics.diagnostic_helpers import *
from diagnostics.convection.high_cloud_classification import (
    CLASSIFICATION_CACHE_VERSION,
    HighCloudThresholds,
    cached_classification_is_current as shared_cached_classification_is_current,
)
from diagnostics.waves.equatorial_wave_config import (
    CORE_WAVE_NAMES,
    WAVE_BOXES,
    WAVE_VARIABILITY_FIELDS,
)
try:
    from waveFilter import kf_filter
except ModuleNotFoundError:
    kf_filter = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/convective_lifecycle/")

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: lifecycle Hovmoller composites
#   2: difference-lifecycle fingerprints
#   3: event-centred spatial composites
#   4: wave-filtered fields following tracked organised-system centres
#   5: track-property lifecycle statistics
figures_to_run = []
figsave = None

USE_LIFECYCLE_CACHE = True
FORCE_RECOMPUTE_LIFECYCLE_CACHE = False

USE_LIFECYCLE_CACHE = USE_LIFECYCLE_CACHE and not env_flag("DISABLE_LIFECYCLE_CACHE")
FORCE_RECOMPUTE_LIFECYCLE_CACHE = (
    FORCE_RECOMPUTE_LIFECYCLE_CACHE
    or env_flag("FORCE_RECOMPUTE_LIFECYCLE_CACHE")
)

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
EVENT_LAT_BOUNDS = (-15.0, 15.0)
WAVE_FILTER_LAT_BOUNDS = (-10.0, 10.0)
EVENT_LAT_HALF_WIDTH = 7.5
FINGERPRINT_LON_WINDOW_DEG = 10.0

LAG_MIN_DAYS = -5.0
LAG_MAX_DAYS = 5.0
REL_LON_GRID = np.arange(-90.0, 90.0 + 1.0e-6, 2.0)
SPATIAL_REL_LON_GRID = np.arange(-50.0, 50.0 + 1.0e-6, 2.0)
SPATIAL_REL_LAT_GRID = np.arange(-20.0, 20.0 + 1.0e-6, 2.0)
SPATIAL_LAGS_DAYS = [-2.0, -1.0, 0.0, 1.0, 2.0]

# Lifecycle event selection. The organiser diagnostic owns track acceptance;
# this script only keeps the tropical events relevant to wave-coupling
# composites. Object-size diversity is shown in track-property diagnostics,
# not by branching the main composite definition.
MIN_EVENT_LIFETIME_HOURS = 18.0
MAX_EVENTS_PER_RUN = None

MIN_OBJECT_CELLS = 1
MIN_OBJECT_AREA_KM2 = 0.0

CF_THRESH = 0.05
P_HIGH_HPA = 440.0
P_PRECIP_MIN = 1.0
CONVECTIVE_FRACTION = 0.50

EARTH_RADIUS_KM = 6371.0
LONGITUDE_PERIOD_DEG = 360.0

LIFECYCLE_WAVE_NAMES = tuple(CORE_WAVE_NAMES)
LIFECYCLE_WAVE_FIELDS = tuple(WAVE_VARIABILITY_FIELDS)

latlon_chunks = {"Time": 8, "latitude": 90, "longitude": 180}
zm_chunks = {"Time": -1}
dask.config.set({"array.slicing.split_large_chunks": True})

CLASSIFICATION_CACHE_DIRS = [
    diagnostic_cache_dir("classification_cache", root_dir=root_dir),
]

PYFLEXTRKR_SEARCH_ROOTS = [
    diagnostic_cache_dir("pyflextrkr_tracking", root_dir=root_dir),
]

CACHE_DIR = diagnostic_cache_dir("convective_lifecycle_cache", root_dir=root_dir)

VAR_ALIASES = {
    "precip": ["precip", "precip_rate", "pr", "rain_rate"],
    "rainc": ["rainc"],
    "rainnc": ["rainnc"],
    "i_rainc": ["i_rainc"],
    "i_rainnc": ["i_rainnc"],
    "olr": ["olr", "olrtoa", "OLR"],
    "pw": ["pw", "precipw", "prw"],
    "theta": ["theta", "potential_temperature"],
    "N2": ["N2", "n2", "static_stability"],
    "dudz": ["dudz", "du_dz", "vertical_shear"],
    "u_zm": ["uReconstructZonal", "u", "uzonal"],
    "u_isobaric": ["uzonal_isobaric", "u_isobaric"],
    "zgrid": ["zgrid"],
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
        + VAR_ALIASES["theta"]
        + VAR_ALIASES["N2"]
        + VAR_ALIASES["dudz"]
        + VAR_ALIASES["u_zm"]
        + VAR_ALIASES["u_isobaric"]
        + VAR_ALIASES["zgrid"]
        + [
            "cldfrac",
            "uzonal_isobaric",
            "umeridional_isobaric",
            "t_isobaric",
            "pressure",
            "w",
        ]
    )
)


# -----------------------------------------------------------------------------
# Small shared helpers
# -----------------------------------------------------------------------------


def save_figure(number, fig, stem, dpi=300):
    global figsave
    if figsave is None:
        figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)
    figsave.save(fig, stem, number=number, dpi=dpi)


should_run_figure = partial(figure_should_run, figures_to_run)


# -----------------------------------------------------------------------------
# Store-backed derived-product caches
# -----------------------------------------------------------------------------

TRACK_EVENT_CACHE_VERSION = "lifecycle_track_events_v5_organised_manifest"
COMPOSITE_CACHE_VERSION = "lifecycle_composites_v5_lazy_fields_lat_subset"
TRACK_PROPERTY_CACHE_VERSION = "lifecycle_track_properties_v3_nonprecipitating"
ANOMALY_BASELINE_CACHE_VERSION = "lifecycle_anomaly_baseline_v1"

PYFLEXTRKR_PRODUCT_FINGERPRINTS = {}


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def cache_path(filename):
    return os.path.join(ensure_cache_dir(), filename)


def base_cache_signature():
    return dict(
        root_dir=root_dir,
        data_locs=list(data_locs),
        latlon_file=list(latlon_file),
        zm_latlon_file=list(zm_latlon_file),
        sim_names=list(sim_names),
        ndays=int(ndays),
        dt=float(dt),
        analysis_end_date=analysis_end_date,
        time_format=time_format,
        tropical_lat_bounds=list(TROPICAL_LAT_BOUNDS),
    )


def object_filter_signature(source_name):
    signature = base_cache_signature()
    signature.update(
        source_name=source_name,
        event_lat_bounds=list(EVENT_LAT_BOUNDS),
        min_event_lifetime_hours=float(MIN_EVENT_LIFETIME_HOURS),
        lifecycle_event_selection=(
            "first accepted tropical organised-system track observation; "
            "no area, precip, or separation filter"
        ),
        max_events_per_run=jsonable_value(MAX_EVENTS_PER_RUN),
        cloud_classification_cache_version=CLASSIFICATION_CACHE_VERSION,
        min_object_cells=int(MIN_OBJECT_CELLS),
        min_object_area_km2=float(MIN_OBJECT_AREA_KM2),
        pyflextrkr_product_fingerprints=dict(PYFLEXTRKR_PRODUCT_FINGERPRINTS),
    )
    return signature


def track_event_cache_signature(source_name):
    signature = object_filter_signature(source_name)
    signature.update(cache_version=TRACK_EVENT_CACHE_VERSION)
    return signature


def composite_cache_signature(source_name, spec_keys, include_spatial):
    signature = object_filter_signature(source_name)
    signature.update(
        cache_version=COMPOSITE_CACHE_VERSION,
        spec_keys=list(spec_keys),
        include_spatial=bool(include_spatial),
        lag_min_days=float(LAG_MIN_DAYS),
        lag_max_days=float(LAG_MAX_DAYS),
        rel_lon_grid=jsonable_value(REL_LON_GRID),
        spatial_rel_lon_grid=jsonable_value(SPATIAL_REL_LON_GRID),
        spatial_rel_lat_grid=jsonable_value(SPATIAL_REL_LAT_GRID),
        spatial_lags_days=jsonable_value(SPATIAL_LAGS_DAYS),
        event_lat_half_width=float(EVENT_LAT_HALF_WIDTH),
        fingerprint_lon_window_deg=float(FINGERPRINT_LON_WINDOW_DEG),
        lifecycle_wave_names=list(LIFECYCLE_WAVE_NAMES),
        lifecycle_wave_fields=list(LIFECYCLE_WAVE_FIELDS),
        wave_boxes={name: jsonable_value(WAVE_BOXES[name]) for name in LIFECYCLE_WAVE_NAMES},
        composite_weighting="event_equal",
        track_centering="tracked_centroid_with_endpoint_clamping_v1",
        composite_field_lat_bounds=jsonable_value(COMPOSITE_FIELD_LAT_BOUNDS),
        anomaly_baseline_cache_version=ANOMALY_BASELINE_CACHE_VERSION,
    )
    return signature


def track_property_cache_signature(source_name, step_days):
    signature = object_filter_signature(source_name)
    signature.update(
        cache_version=TRACK_PROPERTY_CACHE_VERSION,
        step_days=float(step_days),
    )
    return signature


read_cache_metadata = metadata_reader(cache_path)
write_cache_metadata = metadata_writer(cache_path)


def cache_metadata_matches(filename, signature, label):
    if not USE_LIFECYCLE_CACHE:
        return False
    if FORCE_RECOMPUTE_LIFECYCLE_CACHE:
        progress(f"Ignoring cached {label}: force recompute is enabled.")
        return False
    try:
        cached = read_cache_metadata(filename)
    except (OSError, json.JSONDecodeError) as exc:
        progress(f"Ignoring cached {label}: could not read metadata ({exc}).")
        return False
    if cached is None:
        return False
    signature = jsonable_value(signature)
    changed = [
        key
        for key in sorted(signature)
        if cached.get(key) != signature.get(key)
    ]
    if changed:
        shown = ", ".join(changed[:8])
        suffix = "" if len(changed) <= 8 else ", ..."
        progress(f"Ignoring cached {label}: metadata changed ({shown}{suffix}).")
        return False
    return True


def write_cache_csv(df, filename):
    df = df if df is not None else pd.DataFrame()
    df.to_csv(cache_path(filename), index=False)


def read_cache_csv(filename):
    path = cache_path(filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_cache_dataset(ds, filename):
    path = cache_path(filename)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        netcdf_safe_dataset(ds).to_netcdf(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_cache_dataset(filename):
    path = cache_path(filename)
    if not os.path.exists(path):
        return None
    try:
        with xr.open_dataset(path) as ds:
            return ds.load()
    except (OSError, RuntimeError, ValueError) as exc:
        progress(f"Ignoring unreadable cached dataset {path}: {exc}")
        return None


def track_event_cache_files():
    return {
        "events": "lifecycle_onset_events.csv",
        "objects": "lifecycle_track_objects.csv",
        "track_time": "lifecycle_track_time.csv",
        "tracks": "lifecycle_track_summary.csv",
        "metadata": "lifecycle_track_event_cache_metadata.json",
    }


def load_cached_track_event_tables(source_name):
    files = track_event_cache_files()
    if not cache_metadata_matches(files["metadata"], track_event_cache_signature(source_name), "lifecycle track/event tables"):
        return None
    tables = {key: read_cache_csv(filename) for key, filename in files.items() if key != "metadata"}
    if any(value is None for value in tables.values()):
        return None
    if "simulation" in tables["events"].columns:
        events_by_run = {sim: tables["events"][tables["events"].simulation == sim].copy() for sim in sim_names}
    else:
        events_by_run = {sim: pd.DataFrame() for sim in sim_names}
    if "simulation" in tables["track_time"].columns:
        track_time_by_run = {sim: tables["track_time"][tables["track_time"].simulation == sim].copy() for sim in sim_names}
    else:
        track_time_by_run = {sim: pd.DataFrame() for sim in sim_names}
    progress(f"Using cached lifecycle track/event tables from {ensure_cache_dir()}")
    return events_by_run, track_time_by_run, tables["events"], tables["objects"], tables["track_time"], tables["tracks"]


def write_track_event_cache(source_name, events_all, objects_all, track_time_all, tracks_all):
    files = track_event_cache_files()
    write_cache_csv(events_all, files["events"])
    write_cache_csv(objects_all, files["objects"])
    write_cache_csv(track_time_all, files["track_time"])
    write_cache_csv(tracks_all, files["tracks"])
    write_cache_metadata(files["metadata"], track_event_cache_signature(source_name))


def composite_cache_metadata_file():
    return "lifecycle_composite_cache_metadata.json"


def composite_cache_filename(sim_name, var, kind, lag_day=None):
    base = f"lifecycle_{safe_name(sim_name)}_{safe_name(var)}_{kind}"
    if lag_day is not None:
        base += f"_{float(lag_day):+g}d".replace("+", "p").replace("-", "m").replace(".", "p")
    return base + ".nc"


def composite_cache_token(source_name, spec_keys, include_spatial):
    signature = composite_cache_signature(source_name, spec_keys, include_spatial)
    payload = json.dumps(jsonable_value(signature), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def spatial_composite_expected(var, include_spatial):
    return include_spatial and var in (
        "precip",
        "pw_anom",
        "convective_precipitating",
        "stratiform_precipitating",
        "nonprecipitating",
        "cloud_top_pressure",
        "cloud_top_height",
        "N2_TTL_anom",
    )


def load_cached_composite_product(sim, var, include_spatial, expected_token):
    hov = read_cache_dataset(composite_cache_filename(sim, var, "hov"))
    scalar = read_cache_dataset(composite_cache_filename(sim, var, "scalar"))
    if hov is None or scalar is None:
        return None
    if (
        hov.attrs.get("lifecycle_composite_cache_token") != expected_token
        or scalar.attrs.get("lifecycle_composite_cache_token") != expected_token
    ):
        return None

    spatial = {}
    if spatial_composite_expected(var, include_spatial):
        for lag_day in SPATIAL_LAGS_DAYS:
            ds = read_cache_dataset(
                composite_cache_filename(sim, var, "spatial", lag_day=lag_day)
            )
            if (
                ds is None
                or ds.attrs.get("lifecycle_composite_cache_token") != expected_token
            ):
                return None
            spatial[lag_day] = ds
    return {"hov": hov, "scalar": scalar}, spatial


def with_composite_cache_token(ds, token):
    out = ds.copy(deep=False)
    out.attrs = dict(ds.attrs)
    out.attrs["lifecycle_composite_cache_token"] = token
    return out


def write_composite_product_cache(sim, var, product, spatial, cache_token):
    write_cache_dataset(
        with_composite_cache_token(product["hov"], cache_token),
        composite_cache_filename(sim, var, "hov"),
    )
    write_cache_dataset(
        with_composite_cache_token(product["scalar"], cache_token),
        composite_cache_filename(sim, var, "scalar"),
    )
    for lag_day, ds in spatial.items():
        write_cache_dataset(
            with_composite_cache_token(ds, cache_token),
            composite_cache_filename(sim, var, "spatial", lag_day=lag_day),
        )


def write_composite_cache_manifest(
    source_name,
    spec_keys,
    include_spatial,
    event_counts,
    available_specs_by_sim,
):
    signature = composite_cache_signature(source_name, spec_keys, include_spatial)
    signature["event_counts"] = {
        sim: int(event_counts.get(sim, 0)) for sim in sim_names
    }
    signature["available_specs_by_sim"] = {
        sim: list(available_specs_by_sim.get(sim, [])) for sim in sim_names
    }
    signature["composite_cache_token"] = composite_cache_token(
        source_name,
        spec_keys,
        include_spatial,
    )
    write_cache_metadata(composite_cache_metadata_file(), signature)


def load_cached_composites(source_name, spec_keys, include_spatial):
    signature = composite_cache_signature(source_name, spec_keys, include_spatial)
    if not cache_metadata_matches(composite_cache_metadata_file(), signature, "lifecycle composites"):
        return None
    metadata = read_cache_metadata(composite_cache_metadata_file()) or {}
    event_counts = metadata.get("event_counts", {})
    available_specs_by_sim = metadata.get("available_specs_by_sim", {})
    expected_token = composite_cache_token(source_name, spec_keys, include_spatial)
    composites = {}
    spatial_products = {}
    for sim in sim_names:
        composites[sim] = {"event_count": int(event_counts.get(sim, 0))}
        spatial_products[sim] = {}
        for var in available_specs_by_sim.get(sim, spec_keys):
            cached = load_cached_composite_product(
                sim,
                var,
                include_spatial,
                expected_token,
            )
            if cached is None:
                return None
            product, spatial = cached
            composites[sim][var] = product
            if spatial:
                spatial_products[sim][var] = spatial
    progress(f"Using cached lifecycle composites from {ensure_cache_dir()}")
    return composites, spatial_products


def track_property_cache_files():
    return {
        "metadata": "lifecycle_track_property_cache_metadata.json",
        "samples": "lifecycle_track_property_samples.csv",
        "summary": "lifecycle_track_property_summary.csv",
    }


def load_cached_track_property_lifecycle(source_name, step_days):
    files = track_property_cache_files()
    if not cache_metadata_matches(files["metadata"], track_property_cache_signature(source_name, step_days), "track-property lifecycle"):
        return None
    lifecycle = read_cache_csv(files["samples"])
    summary = read_cache_csv(files["summary"])
    if lifecycle is None or summary is None:
        return None
    progress(f"Using cached track-property lifecycle from {ensure_cache_dir()}")
    return lifecycle, summary


def write_track_property_lifecycle_cache(source_name, step_days, lifecycle, summary):
    files = track_property_cache_files()
    write_cache_csv(lifecycle, files["samples"])
    write_cache_csv(summary, files["summary"])
    write_cache_metadata(files["metadata"], track_property_cache_signature(source_name, step_days))


find_var = partial(find_variable, aliases=VAR_ALIASES)
subset_lat = partial(subset_latitude, lat_bounds=TROPICAL_LAT_BOUNDS)
time_aligned_field = partial(time_aligned_variable, aliases=VAR_ALIASES)
get_precip_rate = partial(
    precipitation_rate_from_dataset,
    aliases=VAR_ALIASES,
    dt_hours=dt,
)
weighted_tropical_mean = partial(
    weighted_tropical_mean,
    lat_bounds=TROPICAL_LAT_BOUNDS,
)
time_days_from_coord = partial(time_days_from_coord, dt_hours=dt)
median_time_step_days = partial(median_time_step_days, dt_hours=dt)


def lag_steps_for_reference(ref):
    step_days = median_time_step_days(ref)
    lag_days = np.arange(LAG_MIN_DAYS, LAG_MAX_DAYS + 0.5 * step_days, step_days)
    lag_steps = np.rint(lag_days / step_days).astype(int)
    lag_days = lag_steps.astype(float) * step_days
    return lag_days, lag_steps, step_days


grid_cell_area_km2 = partial(grid_cell_area_km2, earth_radius_km=EARTH_RADIUS_KM)
haversine_distances_km = partial(
    haversine_distances_km,
    lon_period=LONGITUDE_PERIOD_DEG,
    earth_radius_km=EARTH_RADIUS_KM,
)


def circular_mean_lon(lon, weights=None):
    lon = np.asarray(lon, dtype=float) % 360.0
    if weights is None:
        weights = np.ones(lon.shape, dtype=float)
    theta = 2.0 * np.pi * lon / LONGITUDE_PERIOD_DEG
    sin_mean = np.average(np.sin(theta), weights=weights)
    cos_mean = np.average(np.cos(theta), weights=weights)
    return float(np.rad2deg(np.arctan2(sin_mean, cos_mean)) % LONGITUDE_PERIOD_DEG)


# -----------------------------------------------------------------------------
# Cloud classification and TTL fields
# -----------------------------------------------------------------------------


classification_thresholds = partial(
    HighCloudThresholds,
    cf_thresh=CF_THRESH,
    high_cloud_top_pressure_hpa=P_HIGH_HPA,
    surface_precip_min_mm_day=P_PRECIP_MIN,
    convective_fraction_threshold=CONVECTIVE_FRACTION,
)
cached_classification_is_current = partial(
    shared_cached_classification_is_current,
    thresholds=classification_thresholds(),
)


REQUIRED_CLASSIFICATION_VARS = (
    "cloud_top_pressure",
    "cloud_top_height",
    "high_cloud_fraction",
    "high_cloud_mask",
    "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
    "unclassified_high_cloud_mask",
)


def classification_cache_names(sim_name):
    safe = sim_name.replace(" ", "_")
    return [f"{safe}_classified_high_cloud.nc"]


def open_classification_cache(sim_name, precip_ref):
    checked = []
    for cache_dir_i in CLASSIFICATION_CACHE_DIRS:
        for fname in classification_cache_names(sim_name):
            path = os.path.join(cache_dir_i, fname)
            checked.append(path)
            if not os.path.exists(path):
                continue
            progress(f"{sim_name}: using high-cloud classification cache {path}")
            cached = xr.open_dataset(path, chunks={})
            if not cached_classification_is_current(cached):
                cached.close()
                raise RuntimeError(f"{sim_name}: stale high-cloud classification cache: {path}")
            missing = [name for name in REQUIRED_CLASSIFICATION_VARS if name not in cached.data_vars]
            if missing:
                cached.close()
                raise RuntimeError(
                    f"{sim_name}: high-cloud classification cache lacks required shared variables "
                    f"{missing}: {path}"
                )
            out = {}
            for name in cached.data_vars:
                da = cached[name]
                if name.endswith("_mask"):
                    da = da.astype(bool)
                out[name] = select_common_time(da, precip_ref)
            return out
    raise FileNotFoundError(
        f"{sim_name}: required shared high-cloud classification cache is missing. "
        f"Run refined_high_cloud_classification_diagnostics.py first. Checked: {checked}"
    )


def load_required_classification(ds, fields, sim_name):
    return open_classification_cache(sim_name, fields["precip"])


def infer_midlevel_z_from_dataset(ds, size):
    if "zgrid" not in ds.data_vars:
        return None
    zgrid = finite_or_nan(ds["zgrid"])
    if "nVertLevelsP1" not in zgrid.dims:
        return None
    if "Time" in zgrid.dims:
        zgrid = zgrid.isel(Time=0)
    mean_dims = [dim for dim in zgrid.dims if dim != "nVertLevelsP1"]
    if mean_dims:
        zgrid = zgrid.mean(dim=mean_dims, skipna=True)
    zvals = 0.5 * (
        zgrid.isel(nVertLevelsP1=slice(0, -1)).values
        + zgrid.isel(nVertLevelsP1=slice(1, None)).values
    )
    if len(zvals) != size:
        return None
    return np.asarray(zvals, dtype=float)


def prepare_vertical_field(ds, da):
    try:
        dim = vertical_dim(da)
    except ValueError:
        return da
    if dim != "z":
        da = da.rename({dim: "z"})
    needs_z = "z" not in da.coords
    if not needs_z:
        zvals = np.asarray(da.z.values, dtype=float)
        needs_z = zvals.size != da.sizes["z"] or not np.all(np.isfinite(zvals))
    if needs_z:
        zvals = infer_midlevel_z_from_dataset(ds, da.sizes["z"])
        if zvals is not None:
            da = da.assign_coords(z=("z", zvals))
            da["z"].attrs.update(units="m", long_name="inferred geometric height of layer midpoints")
    return da


def lifecycle_wave_field(ds, precip, field_name):
    """Return the same field definition used by equatorial-wave variability."""
    if field_name == "Precip":
        out = precip
    elif field_name == "OLR":
        name = find_var(ds, "olr", required=True)
        out = finite_or_nan(ds[name])
    elif field_name == "PW":
        name = find_var(ds, "pw", required=True)
        out = finite_or_nan(ds[name])
    elif field_name == "cldfrac":
        if "cldfrac" not in ds.data_vars:
            raise ValueError("cldfrac is required by the shared wave-field definition")
        cloud = prepare_vertical_field(ds, finite_or_nan(ds["cldfrac"]))
        zdim = vertical_dim(cloud)
        out = cloud.mean(dim=zdim, skipna=True)
    else:
        prefixes = (
            ("w", None),
            ("u", "uzonal_isobaric"),
            ("v", "umeridional_isobaric"),
            ("T", "t_isobaric"),
        )
        out = None
        for prefix, variable in prefixes:
            if not field_name.startswith(prefix):
                continue
            level = float(field_name[len(prefix):])
            if variable is None:
                out = vertical_velocity_at_pressure(ds, level)
            else:
                if variable not in ds.data_vars:
                    raise ValueError(
                        f"{variable} is required for shared wave field {field_name}"
                    )
                out = finite_or_nan(ds[variable]).sel(pres=level, method="nearest")
            break
        if out is None:
            raise ValueError(f"Unrecognised shared wave field: {field_name}")
    out = select_common_time(out, precip)
    out.name = field_name
    return out


def _z_or_pres_layer_mask(da, z_bounds_km=(14.0, 18.0), pres_bounds_hpa=(70.0, 150.0)):
    dim = vertical_dim(da) if any(dim in da.dims for dim in ("z", "nVertLevels", "nVertLevelsP1")) else None
    if dim is not None:
        if dim != "z":
            da = da.rename({dim: "z"})
        if "z" not in da.coords:
            return da, None, "z"
        zvals = np.asarray(da.z.values, dtype=float)
        zkm = zvals / 1000.0 if np.nanmax(np.abs(zvals)) > 100.0 else zvals
        mask = xr.DataArray((zkm >= z_bounds_km[0]) & (zkm <= z_bounds_km[1]), dims=("z",), coords={"z": da.z})
        return da, mask, "z"
    if "pres" in da.dims:
        pres = np.asarray(da.pres.values, dtype=float)
        lo, hi = sorted(pres_bounds_hpa)
        mask = xr.DataArray((pres >= lo) & (pres <= hi), dims=("pres",), coords={"pres": da.pres})
        return da, mask, "pres"
    return da, None, None


def layer_mean(da, name, z_bounds_km=(14.0, 18.0), pres_bounds_hpa=(70.0, 150.0)):
    da, mask, dim = _z_or_pres_layer_mask(finite_or_nan(da), z_bounds_km, pres_bounds_hpa)
    if mask is None or dim is None or not bool(mask.any()):
        return None
    out = da.where(mask, drop=True).mean(dim=dim, skipna=True)
    out.name = name
    return out


def compute_ttl_fields(source_ds, precip_ref):
    out = {}

    n2_name = find_var(source_ds, "N2")
    if n2_name is not None:
        n2 = layer_mean(prepare_vertical_field(source_ds, source_ds[n2_name]), "N2_TTL")
        if n2 is not None:
            out["N2_TTL"] = n2
    else:
        theta_name = find_var(source_ds, "theta")
        theta_da = prepare_vertical_field(source_ds, source_ds[theta_name]) if theta_name is not None else None
        if theta_da is not None and "z" in theta_da.dims and "z" in theta_da.coords:
            try:
                theta = finite_or_nan(theta_da)
                n2 = 9.80665 / theta * theta.differentiate("z")
                n2 = layer_mean(n2, "N2_TTL")
                if n2 is not None:
                    out["N2_TTL"] = n2
            except Exception as exc:
                warnings.warn(f"Could not compute N2_TTL from theta: {exc}")

    u_name = find_var(source_ds, "u_zm") or find_var(source_ds, "u_isobaric")
    if u_name is not None:
        u_da = prepare_vertical_field(source_ds, source_ds[u_name])
        u_ttl = layer_mean(u_da, "u_TTL")
        if u_ttl is not None:
            out["u_TTL"] = u_ttl

    dudz_name = find_var(source_ds, "dudz")
    if dudz_name is not None:
        dudz = layer_mean(prepare_vertical_field(source_ds, source_ds[dudz_name]), "dudz_TTL")
        if dudz is not None:
            out["dudz_TTL"] = dudz
    elif u_name is not None:
        try:
            u = finite_or_nan(prepare_vertical_field(source_ds, source_ds[u_name]))
            if "z" in u.dims and "z" in u.coords:
                dudz = layer_mean(u.differentiate("z"), "dudz_TTL")
                if dudz is not None:
                    out["dudz_TTL"] = dudz
        except Exception as exc:
            warnings.warn(f"Could not compute dudz_TTL from u: {exc}")

    aligned = {}
    for key, da in out.items():
        if "Time" in da.dims and "Time" in precip_ref.dims:
            da = select_common_time(da, precip_ref)
        aligned[key] = da
    return aligned


# -----------------------------------------------------------------------------
# Object and track tables
# -----------------------------------------------------------------------------


def object_properties_from_labels(labels, lat, lon, area2d, field_slices=None):
    labels = np.asarray(labels, dtype=np.int32)
    if labels.ndim != 2:
        labels = np.squeeze(labels)
    if labels.size == 0 or int(np.nanmax(labels)) <= 0:
        return pd.DataFrame()

    area_vals = np.asarray(area2d.values, dtype=float)
    lat_grid, lon_grid = np.meshgrid(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float), indexing="ij")
    field_values = {}
    if field_slices:
        for key, da in field_slices.items():
            if da is None or not {"latitude", "longitude"}.issubset(da.dims):
                continue
            sub = da.transpose("latitude", "longitude")
            sub = sub.sel(latitude=lat, longitude=lon)
            field_values[key] = np.asarray(sub.values, dtype=float)

    rows = []
    for label_id in np.unique(labels):
        if label_id <= 0:
            continue
        obj = labels == int(label_id)
        cell_count = int(np.count_nonzero(obj))
        area = float(np.nansum(area_vals[obj]))
        if cell_count < MIN_OBJECT_CELLS or area < MIN_OBJECT_AREA_KM2:
            continue

        weights = area_vals[obj]
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
        row = dict(
            label=int(label_id),
            cell_count=cell_count,
            area_km2=area,
            centroid_lat=float(np.average(lat_grid[obj], weights=weights)),
            centroid_lon=circular_mean_lon(lon_grid[obj], weights=weights),
        )
        for key, vals in field_values.items():
            v = vals[obj]
            ok = np.isfinite(v)
            if np.any(ok):
                w = weights[ok]
                row[f"mean_{key}"] = float(np.average(v[ok], weights=w))
                row[f"max_{key}"] = float(np.nanmax(v))
                row[f"min_{key}"] = float(np.nanmin(v))
                row[f"area_integral_{key}"] = float(np.nansum(v[ok] * w))
                if key.endswith("_mask") or key == "stratiform_precipitating":
                    present = v[ok] > 0.5
                    row[f"frac_{key}"] = float(np.average(present, weights=w))
                    row[f"area_{key}_km2"] = float(np.nansum(w[present]))
            else:
                row[f"mean_{key}"] = np.nan
                row[f"max_{key}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def chararray_to_strings(array):
    arr = np.asarray(array)
    if arr.dtype.kind in ("S", "U") and arr.ndim == 2:
        return ["".join(x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in row).strip() for row in arr]
    return [str(x) for x in arr.ravel()]


def find_pyflextrkr_work_path(source_name, sim_name):
    if source_name != PRIMARY_TRACK_SOURCE:
        raise ValueError(
            f"Lifecycle composites require {PRIMARY_TRACK_SOURCE!r}, got {source_name!r}"
        )
    safe_sim = sim_name.replace(" ", "_").replace("/", "_")
    source_candidates = [f"{source_name}_{CLASSIFICATION_CACHE_VERSION}"]
    candidates = []
    for root in PYFLEXTRKR_SEARCH_ROOTS:
        for source_candidate in source_candidates:
            safe_source = source_candidate.replace(" ", "_").replace("/", "_")
            candidates.extend([
                os.path.join(root, safe_source, safe_sim),
                os.path.join(root, safe_source, sim_name),
                os.path.join(root, safe_source, safe_name(sim_name)),
            ])
    for path in candidates:
        if glob.glob(os.path.join(path, "stats", "tracknumbers_*.nc")) and glob.glob(os.path.join(path, "tracking", "cloudid_*.nc")):
            return path
    return None


def require_pyflextrkr_work_path(source_name, sim_name):
    work_path = find_pyflextrkr_work_path(source_name, sim_name)
    if work_path is None:
        raise FileNotFoundError(
            f"{sim_name}: required PyFLEXTRKR output for {source_name} is missing. "
            "Run convective_organisation_diagnostics.py first."
        )
    return work_path


def validate_pyflextrkr_product(source_name, sim_name, fields):
    work_path = require_pyflextrkr_work_path(source_name, sim_name)
    metadata = read_product_metadata(work_path)
    if metadata is None:
        raise RuntimeError(
            f"{sim_name}: PyFLEXTRKR output lacks required product metadata in "
            f"{work_path}. Rerun convective_organisation_diagnostics.py; this "
            "directory is an obsolete-output candidate."
        )
    expected = dict(
        product_version=PYFLEXTRKR_PRODUCT_VERSION,
        source=source_name,
        simulation=sim_name,
        analysis_role=PRIMARY_ANALYSIS_ROLE,
        classification_cache_version=CLASSIFICATION_CACHE_VERSION,
        min_track_observations=MIN_ORGANISED_TRACK_OBSERVATIONS,
        min_track_duration_hours=MIN_ORGANISED_TRACK_DURATION_HOURS,
        track_acceptance_criteria=ORGANISED_TRACK_ACCEPTANCE_CRITERIA,
        embedded_convective_core_requirement=ORGANISED_TRACK_EMBEDDED_CORE_REQUIREMENT,
        object_geometry=ORGANISED_TRACK_OBJECT_GEOMETRY,
        envelope_min_raw_area_km2=ORGANISED_ENVELOPE_MIN_RAW_AREA_KM2,
        **time_signature(fields["precip"].Time.values),
    )
    changed = [key for key, value in expected.items() if metadata.get(key) != value]
    if changed:
        raise RuntimeError(
            f"{sim_name}: stale or incorrect PyFLEXTRKR product ({', '.join(changed)}): "
            f"{work_path}. Rerun convective_organisation_diagnostics.py."
        )
    if metadata.get("product_fingerprint") != product_fingerprint(metadata):
        raise RuntimeError(f"{sim_name}: invalid PyFLEXTRKR product fingerprint: {work_path}")
    tracknumbers_name = metadata.get("tracknumbers_file", "")
    tracknumbers_path = os.path.join(work_path, "stats", tracknumbers_name)
    if not tracknumbers_name or not os.path.exists(tracknumbers_path):
        raise RuntimeError(
            f"{sim_name}: the track-number file named by PyFLEXTRKR metadata is missing: "
            f"{tracknumbers_path}"
        )
    if metadata.get("tracknumbers_sha256") != sha256_file(tracknumbers_path):
        raise RuntimeError(
            f"{sim_name}: PyFLEXTRKR track-number checksum mismatch: {tracknumbers_path}"
        )
    manifest_path = track_manifest_path(work_path)
    manifest = read_track_manifest(work_path)
    if manifest is None:
        raise RuntimeError(
            f"{sim_name}: required organised-track manifest is missing: {manifest_path}"
        )
    if metadata.get("track_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError(
            f"{sim_name}: organised-track manifest checksum mismatch: {manifest_path}"
        )
    required_manifest_columns = {
        "source", "simulation", "track_id", "accepted", "rejection_reason",
        "observation_count", "sampling_support_hours", "max_area_km2",
        "has_embedded_convective_core", "max_embedded_convective_core_area_km2",
        "max_embedded_convective_core_fraction", "max_abs_centroid_lat",
        "max_translation_speed_kmh",
    }
    missing_manifest_columns = required_manifest_columns - set(manifest.columns)
    if missing_manifest_columns:
        raise RuntimeError(
            f"{sim_name}: organised-track manifest lacks columns "
            f"{sorted(missing_manifest_columns)}: {manifest_path}"
        )
    if set(manifest.source.astype(str)) != {source_name} or set(
        manifest.simulation.astype(str)
    ) != {sim_name}:
        raise RuntimeError(
            f"{sim_name}: organised-track manifest source/simulation mismatch: "
            f"{manifest_path}"
        )
    if manifest.track_id.duplicated().any():
        raise RuntimeError(f"{sim_name}: duplicate track IDs in {manifest_path}")
    accepted_mask = manifest_accepted_mask(manifest)
    accepted_manifest = manifest[accepted_mask]
    if (
        (accepted_manifest.observation_count < MIN_ORGANISED_TRACK_OBSERVATIONS).any()
        or (
            accepted_manifest.sampling_support_hours
            < MIN_ORGANISED_TRACK_DURATION_HOURS
        ).any()
        or (~accepted_manifest.has_embedded_convective_core.astype(bool)).any()
    ):
        raise RuntimeError(
            f"{sim_name}: organised-track manifest contains accepted tracks that "
            "violate duration/core constraints"
        )
    if metadata.get("accepted_track_count") != int(accepted_mask.sum()):
        raise RuntimeError(
            f"{sim_name}: accepted-track count disagrees with product metadata"
        )
    expected_basetimes = epoch_seconds(fields["precip"].Time.values)
    with xr.open_dataset(
        tracknumbers_path,
        mask_and_scale=False,
        decode_times=False,
        concat_characters=True,
    ) as track_ds:
        file_basetimes = np.asarray(track_ds["basetimes"].values).astype(np.int64).ravel()
        track_time_count = int(track_ds["track_numbers"].sizes.get("nfiles", -1))
        listed_cloudids = (
            chararray_to_strings(track_ds["cloudid_files"].values)
            if "cloudid_files" in track_ds
            else []
        )
    if not np.array_equal(file_basetimes, expected_basetimes):
        raise RuntimeError(
            f"{sim_name}: timestamps inside {tracknumbers_path} do not exactly "
            "match the instantaneous 6-hourly model output"
        )
    if track_time_count not in (-1, expected_basetimes.size):
        raise RuntimeError(
            f"{sim_name}: PyFLEXTRKR track array has {track_time_count} times, "
            f"expected {expected_basetimes.size}"
        )
    if len(listed_cloudids) != expected_basetimes.size:
        raise RuntimeError(
            f"{sim_name}: PyFLEXTRKR lists {len(listed_cloudids)} cloudid files, "
            f"expected {expected_basetimes.size}"
        )
    PYFLEXTRKR_PRODUCT_FINGERPRINTS[sim_name] = metadata["product_fingerprint"]
    return work_path, tracknumbers_path, metadata, manifest


def load_pyflextrkr_objects(source_name, sim_name, fields):
    work_path, tracknumbers_path, _, manifest = validate_pyflextrkr_product(
        source_name, sim_name, fields
    )
    accepted_ids = set(
        manifest.loc[manifest_accepted_mask(manifest), "track_id"].to_numpy(dtype=int)
    )
    progress(f"{sim_name}: reading PyFLEXTRKR track links for {source_name}: {tracknumbers_path}")
    ds_track = xr.open_dataset(tracknumbers_path, mask_and_scale=False, decode_times=False, concat_characters=True)
    track_numbers = np.asarray(ds_track["track_numbers"].squeeze().values)
    if track_numbers.ndim == 1:
        track_numbers = track_numbers[None, :]
    basetimes = np.asarray(ds_track["basetimes"].values).astype(np.int64).ravel()
    if "cloudid_files" in ds_track:
        cloud_files = chararray_to_strings(ds_track["cloudid_files"].values)
    else:
        cloud_files = []
    ds_track.close()

    tracking_path = os.path.join(work_path, "tracking")
    precip = fields["precip"]
    expected_basetimes = epoch_seconds(precip.Time.values)
    time_count = precip.sizes.get("Time", 0)
    if time_count < 1:
        raise RuntimeError(f"{sim_name}: no model timestamps are available for lifecycle tracking")
    if track_numbers.shape[0] != time_count:
        raise RuntimeError(
            f"{sim_name}: PyFLEXTRKR track-number time count {track_numbers.shape[0]} "
            f"does not match model time count {time_count}"
        )
    if basetimes.size != time_count or not np.array_equal(basetimes, expected_basetimes):
        raise RuntimeError(
            f"{sim_name}: PyFLEXTRKR basetimes do not exactly match the instantaneous "
            "6-hourly model timestamps"
        )
    if len(np.unique(basetimes)) != time_count or np.any(np.diff(basetimes) <= 0):
        raise RuntimeError(f"{sim_name}: PyFLEXTRKR basetimes are duplicated or unordered")
    if not cloud_files:
        raise RuntimeError(f"{sim_name}: track-number product does not list cloudid files")
    if len(cloud_files) != time_count:
        raise RuntimeError(
            f"{sim_name}: PyFLEXTRKR cloudid file count {len(cloud_files)} does not "
            f"match model time count {time_count}"
        )

    rows = []
    dt_hours = 24.0 * median_time_step_days(precip)
    time_days = time_days_from_coord(precip.Time)

    for it in range(time_count):
        path = os.path.join(tracking_path, cloud_files[it])
        if not os.path.exists(path):
            raise FileNotFoundError(f"{sim_name}: required PyFLEXTRKR cloudid file is missing: {path}")
        with xr.open_dataset(path, decode_times=False, mask_and_scale=False) as cloud:
            cloud_basetime = int(np.asarray(cloud["base_time"].values).ravel()[0])
            if cloud_basetime != int(expected_basetimes[it]):
                raise RuntimeError(
                    f"{sim_name}: cloudid timestamp mismatch at model index {it}: {path}"
                )
            labels = np.asarray(cloud["feature_number"].squeeze().values, dtype=np.int32)
            lat_name = "lat" if "lat" in cloud.coords else "latitude"
            lon_name = "lon" if "lon" in cloud.coords else "longitude"
            lat = np.asarray(cloud[lat_name].values, dtype=float)
            lon = np.asarray(cloud[lon_name].values, dtype=float)
            if lat.ndim == 2:
                lat = lat[:, 0]
            if lon.ndim == 2:
                lon = lon[0, :]

        lat_da = xr.DataArray(lat, dims=("latitude",), name="latitude")
        lon_da = xr.DataArray(lon, dims=("longitude",), name="longitude")
        area2d = grid_cell_area_km2(lat, lon)
        field_slices = {}
        for key in (
            "precip", "pw", "cloud_top_pressure", "cloud_top_height",
            "high_cloud_fraction", "stratiform_precipitating_high_cloud_mask",
            "convective_precipitating_high_cloud_mask",
            "nonprecipitating_high_cloud_mask",
        ):
            if key not in fields or not {"latitude", "longitude"}.issubset(fields[key].dims):
                continue
            field_slices[key] = fields[key].isel(Time=it).sel(latitude=lat_da, longitude=lon_da, method="nearest").load()

        objects = object_properties_from_labels(labels, lat, lon, area2d, field_slices)
        if len(objects) == 0:
            continue
        for record in objects.to_dict("records"):
            label_index = int(record["label"]) - 1
            if label_index < 0 or label_index >= track_numbers.shape[1]:
                continue
            track_id = int(track_numbers[it, label_index])
            if track_id <= 0 or track_id not in accepted_ids:
                continue
            rows.append(
                dict(
                    source=source_name,
                    simulation=sim_name,
                    Time=precip.Time.values[it],
                    pyflextrkr_basetime=int(basetimes[it]),
                    time_index=it,
                    time_days=float(time_days[it]),
                    dt_hours=dt_hours,
                    track_id=track_id,
                    tracking_method="pyflextrkr_output",
                    onset_flag=True,
                    **record,
                )
            )

    objects_df = pd.DataFrame(rows)
    if len(objects_df):
        objects_df = (
            objects_df.sort_values(
                ["track_id", "time_index", "area_km2"],
                ascending=[True, True, False],
            )
            .drop_duplicates(["track_id", "time_index"])
            .reset_index(drop=True)
        )
    track_time, tracks_df = summarise_tracks(objects_df)
    return objects_df, track_time, tracks_df


def summarise_tracks(objects_df):
    if objects_df is None or len(objects_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    agg = dict(
        Time=("Time", "first"),
        time_days=("time_days", "first"),
        dt_hours=("dt_hours", "first"),
        area_km2=("area_km2", "sum"),
        centroid_lat=("centroid_lat", "mean"),
        centroid_lon=("centroid_lon", circular_mean_lon),
        object_count=("label", "count"),
        onset_flag=("onset_flag", "max"),
        tracking_method=("tracking_method", "first"),
    )
    for col in objects_df.columns:
        if col.startswith("mean_"):
            agg[col] = (col, "mean")
        elif col.startswith("max_"):
            agg[col] = (col, "max")
        elif col.startswith("min_"):
            agg[col] = (col, "min")
        elif col.startswith("frac_"):
            agg[col] = (col, "mean")
        elif col.startswith("area_"):
            agg[col] = (col, "sum")
        elif col.startswith("area_integral_"):
            agg[col] = (col, "sum")

    track_time = (
        objects_df
        .groupby(["source", "simulation", "track_id", "time_index"], as_index=False)
        .agg(**agg)
        .sort_values(["source", "simulation", "track_id", "time_index"])
    )
    if {"area_integral_precip", "area_km2"}.issubset(track_time.columns):
        track_time["mean_precip"] = np.where(
            track_time.area_km2.to_numpy(dtype=float) > 0.0,
            track_time.area_integral_precip.to_numpy(dtype=float)
            / track_time.area_km2.to_numpy(dtype=float),
            np.nan,
        )

    speed_values = np.full(len(track_time), np.nan)
    for _, idx in track_time.groupby(["source", "simulation", "track_id"]).groups.items():
        g = track_time.loc[idx].sort_values("time_index")
        pos = g.index.values
        if len(g) < 2:
            continue
        d = haversine_distances_km(
            g.centroid_lat.values[:-1],
            g.centroid_lon.values[:-1],
            g.centroid_lat.values[1:],
            g.centroid_lon.values[1:],
        )
        dh = np.diff(g.time_days.values) * 24.0
        speed = np.where(dh > 0, d / dh, np.nan)
        speed_values[track_time.index.get_indexer(pos[1:])] = speed
    track_time["propagation_speed_kmh"] = speed_values

    tracks = []
    for (source, simulation, track_id), group in track_time.groupby(["source", "simulation", "track_id"]):
        group = group.sort_values("time_index")
        dt_hours = float(group.dt_hours.median()) if "dt_hours" in group else dt
        lifetime_steps = int(group.time_index.nunique())
        lifetime_hours = float(lifetime_steps * dt_hours)
        row = dict(
            source=source,
            simulation=simulation,
            track_id=int(track_id),
            start_time=group.Time.iloc[0],
            end_time=group.Time.iloc[-1],
            start_time_index=int(group.time_index.iloc[0]),
            end_time_index=int(group.time_index.iloc[-1]),
            lifetime_steps=lifetime_steps,
            lifetime_hours=lifetime_hours,
            max_area_km2=float(group.area_km2.max()),
            mean_area_km2=float(group.area_km2.mean()),
            mean_propagation_speed_kmh=float(group.propagation_speed_kmh.mean(skipna=True)),
            max_propagation_speed_kmh=float(group.propagation_speed_kmh.max(skipna=True)),
            tracking_method=group.tracking_method.iloc[0],
        )
        if "area_integral_precip" in group:
            row["precip_volume_proxy_km2_mm"] = float(
                (group.area_integral_precip * group.dt_hours / 24.0).sum(skipna=True)
            )
            row["max_precip_volume_rate_proxy_km2_mm_day"] = float(
                group.area_integral_precip.max(skipna=True)
            )
        for key in ("precip", "cloud_top_pressure", "cloud_top_height", "high_cloud_fraction", "pw"):
            if f"max_{key}" in group:
                row[f"max_{key}"] = float(group[f"max_{key}"].max(skipna=True))
            if f"min_{key}" in group:
                row[f"min_{key}"] = float(group[f"min_{key}"].min(skipna=True))
            if f"mean_{key}" in group:
                row[f"mean_{key}"] = float(group[f"mean_{key}"].mean(skipna=True))
        tracks.append(row)
    return track_time, pd.DataFrame(tracks)


def build_track_tables_for_source(source_name, fields, sim_name):
    if source_name != PRIMARY_TRACK_SOURCE:
        raise ValueError(
            f"Lifecycle composites require the primary tracked source "
            f"{PRIMARY_TRACK_SOURCE!r}, got {source_name!r}"
        )
    return load_pyflextrkr_objects(source_name, sim_name, fields)


def select_onset_events(track_time, tracks_df, source_name, sim_name):
    if len(track_time) == 0 or len(tracks_df) == 0:
        return pd.DataFrame()
    merged = track_time.merge(
        tracks_df[[
            "source", "simulation", "track_id", "lifetime_hours", "max_area_km2",
            "mean_propagation_speed_kmh", "tracking_method"
        ] + [
            c for c in (
                "max_precip",
                "mean_precip",
                "precip_volume_proxy_km2_mm",
                "max_precip_volume_rate_proxy_km2_mm_day",
                "min_cloud_top_pressure",
                "max_cloud_top_height",
                "max_high_cloud_fraction",
            )
            if c in tracks_df.columns
        ]],
        on=["source", "simulation", "track_id"],
        how="left",
        suffixes=("", "_track"),
    )

    rows = []
    for track_id, group in merged.groupby("track_id"):
        group = group.sort_values("time_index")
        onset_group = group[group.onset_flag.astype(bool)]
        if len(onset_group) == 0:
            continue
        onset = onset_group.iloc[0]
        lifetime_hours = float(onset.get("lifetime_hours", np.nan))
        max_area = float(onset.get("max_area_km2_track", onset.get("max_area_km2", np.nan)))
        max_precip = float(onset.get("max_precip_track", onset.get("max_precip", np.nan)))
        if (not np.isfinite(max_precip)) and "max_precip" in group:
            max_precip = float(group.max_precip.max(skipna=True))
        if lifetime_hours < MIN_EVENT_LIFETIME_HOURS:
            continue
        if not (EVENT_LAT_BOUNDS[0] <= float(onset.centroid_lat) <= EVENT_LAT_BOUNDS[1]):
            continue
        precip_volume = float(
            onset.get("precip_volume_proxy_km2_mm_track", onset.get("precip_volume_proxy_km2_mm", np.nan))
        )
        if not np.isfinite(precip_volume):
            precip_volume = float(max_area * max_precip) if np.isfinite(max_area) and np.isfinite(max_precip) else np.nan
        rows.append(
            dict(
                simulation=sim_name,
                source=source_name,
                track_id=int(track_id),
                Time=onset.Time,
                onset_time_index=int(onset.time_index),
                time_days=float(onset.time_days),
                centroid_lat=float(onset.centroid_lat),
                centroid_lon=float(onset.centroid_lon),
                center_lon=float(onset.centroid_lon),
                lifetime_hours=lifetime_hours,
                max_area_km2=max_area,
                max_precip=max_precip,
                mean_precip=float(onset.get("mean_precip_track", onset.get("mean_precip", np.nan))),
                precip_volume_proxy_km2_mm=precip_volume,
                max_precip_volume_rate_proxy_km2_mm_day=float(
                    onset.get(
                        "max_precip_volume_rate_proxy_km2_mm_day_track",
                        onset.get("max_precip_volume_rate_proxy_km2_mm_day", np.nan),
                    )
                ),
                min_cloud_top_pressure=float(group.min_cloud_top_pressure.min(skipna=True)) if "min_cloud_top_pressure" in group else np.nan,
                max_cloud_top_height=float(group.max_cloud_top_height.max(skipna=True)) if "max_cloud_top_height" in group else np.nan,
                max_high_cloud_fraction=float(group.max_high_cloud_fraction.max(skipna=True)) if "max_high_cloud_fraction" in group else np.nan,
                mean_propagation_speed_kmh=float(onset.get("mean_propagation_speed_kmh", np.nan)),
                tracking_method=onset.get("tracking_method", ""),
            )
        )

    events = pd.DataFrame(rows)
    if len(events) == 0:
        return events
    events = events.sort_values("time_days").reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1)
    if MAX_EVENTS_PER_RUN is not None and len(events) > MAX_EVENTS_PER_RUN:
        events = events.iloc[:MAX_EVENTS_PER_RUN].copy()
    return events


def build_track_path_lookup(track_time):
    paths = {}
    if track_time is None or len(track_time) == 0:
        return paths
    for track_id, group in track_time.groupby("track_id"):
        group = group.sort_values("time_index").drop_duplicates("time_index")
        paths[int(track_id)] = dict(
            time_index=group.time_index.to_numpy(dtype=int),
            latitude=group.centroid_lat.to_numpy(dtype=float),
            longitude_unwrapped=np.rad2deg(
                np.unwrap(np.deg2rad(group.centroid_lon.to_numpy(dtype=float)))
            ),
        )
    return paths


def track_center_at_index(event, time_index, track_paths):
    track_id = int(event.track_id)
    if track_id not in track_paths:
        raise RuntimeError(f"No tracked centroid path is available for track {track_id}")
    path = track_paths[track_id]
    indices = path["time_index"]
    if time_index < indices[0]:
        pos = 0
        method = "pre_genesis_endpoint"
        latitude = path["latitude"][pos]
        longitude = path["longitude_unwrapped"][pos]
    elif time_index > indices[-1]:
        pos = -1
        method = "post_lysis_endpoint"
        latitude = path["latitude"][pos]
        longitude = path["longitude_unwrapped"][pos]
    else:
        exact = np.where(indices == int(time_index))[0]
        if exact.size:
            pos = int(exact[0])
            method = "tracked_centroid"
            latitude = path["latitude"][pos]
            longitude = path["longitude_unwrapped"][pos]
        else:
            method = "interpolated_tracked_centroid"
            latitude = np.interp(time_index, indices, path["latitude"])
            longitude = np.interp(time_index, indices, path["longitude_unwrapped"])
    return float(latitude), float(longitude % 360.0), method


def build_composite_centre_table(
    events_by_run,
    track_time_by_run,
    fields_by_run,
    lag_days,
    lag_steps,
):
    rows = []
    for sim, fields in zip(sim_names, fields_by_run):
        events = events_by_run.get(sim, pd.DataFrame())
        paths = build_track_path_lookup(track_time_by_run.get(sim, pd.DataFrame()))
        for event in events.itertuples(index=False):
            for lag_day, step in zip(lag_days, lag_steps):
                it = int(event.onset_time_index) + int(step)
                if it < 0 or it >= fields["precip"].sizes.get("Time", 0):
                    continue
                lat, lon, method = track_center_at_index(event, it, paths)
                rows.append(
                    dict(
                        simulation=sim,
                        event_id=int(event.event_id),
                        track_id=int(event.track_id),
                        lag_days=float(lag_day),
                        time_index=it,
                        Time=fields["precip"].Time.values[it],
                        center_latitude=lat,
                        center_longitude=lon,
                        center_method=method,
                    )
                )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Wave filters and composite-field preparation
# -----------------------------------------------------------------------------


def anomaly(da, label=None):
    if "Time" in da.dims:
        baseline = da.mean(dim="Time", skipna=True)
        if getattr(baseline, "chunks", None) is not None:
            progress(f"Computing reusable anomaly baseline for {label or da.name or 'field'}")
            baseline = baseline.compute()
        out = da - baseline
    else:
        baseline = da.mean(skipna=True)
        if getattr(baseline, "chunks", None) is not None:
            baseline = baseline.compute()
        out = da - baseline
    out.attrs.update(getattr(da, "attrs", {}))
    return out


def anomaly_baseline_cache_filename(sim_name, field_key):
    return f"anomaly_baseline_{safe_name(sim_name)}_{safe_name(field_key)}.nc"


def anomaly_baseline_signature(da, sim_name, field_key):
    signature = dict(
        cache_version=ANOMALY_BASELINE_CACHE_VERSION,
        source_simulation=sim_name,
        field_key=field_key,
        root_dir=root_dir,
        ndays=int(ndays),
        analysis_end_date=analysis_end_date,
        time_format=time_format,
        dims_after_mean={
            dim: int(size)
            for dim, size in da.sizes.items()
            if dim != "Time"
        },
    )
    if "Time" in da.dims:
        signature.update(time_signature(da.Time.values))
    for coord_name in ("latitude", "longitude", "pres", "z"):
        if coord_name not in da.coords:
            continue
        values = np.asarray(da[coord_name].values, dtype=float)
        finite = values[np.isfinite(values)]
        signature[f"{coord_name}_size"] = int(values.size)
        signature[f"{coord_name}_first"] = float(values[0]) if values.size else None
        signature[f"{coord_name}_last"] = float(values[-1]) if values.size else None
        signature[f"{coord_name}_min"] = float(finite.min()) if finite.size else None
        signature[f"{coord_name}_max"] = float(finite.max()) if finite.size else None
    return signature


def signature_token(signature):
    payload = json.dumps(
        jsonable_value(signature),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cached_anomaly(da, sim_name, field_key, label=None):
    if "Time" not in da.dims:
        return anomaly(da, label=label)

    signature = anomaly_baseline_signature(da, sim_name, field_key)
    token = signature_token(signature)
    filename = anomaly_baseline_cache_filename(sim_name, field_key)
    cached = read_cache_dataset(filename)
    if (
        cached is not None
        and "baseline" in cached
        and cached.attrs.get("anomaly_baseline_cache_token") == token
    ):
        progress(f"{sim_name}: using cached {field_key} anomaly baseline")
        baseline = cached["baseline"]
    else:
        baseline = da.mean(dim="Time", skipna=True)
        if getattr(baseline, "chunks", None) is not None:
            progress(f"Computing reusable anomaly baseline for {label or da.name or field_key}")
            baseline = baseline.compute()
        baseline = baseline.rename("baseline")
        ds = baseline.to_dataset()
        ds.attrs.update(
            anomaly_baseline_cache_token=token,
            anomaly_baseline_signature=json.dumps(
                jsonable_value(signature),
                sort_keys=True,
            ),
        )
        write_cache_dataset(ds, filename)
        progress(f"{sim_name}: cached {field_key} anomaly baseline")

    out = da - baseline
    out.attrs.update(getattr(da, "attrs", {}))
    return out


def wave_cache_path(sim_name, field_key, wave_name):
    cache_dir = diagnostic_cache_dir("wave_cache", root_dir=root_dir)
    os.makedirs(cache_dir, exist_ok=True)
    projection = "direct" if wave_name == "Easterly" else "symmetric"
    lat_tag = (
        f"{WAVE_FILTER_LAT_BOUNDS[0]:g}_{WAVE_FILTER_LAT_BOUNDS[1]:g}"
        .replace("-", "m")
        .replace(".", "p")
    )
    return os.path.join(
        cache_dir,
        f"{safe_name(sim_name)}_{safe_name(field_key)}_{safe_name(wave_name)}_"
        f"{lat_tag}_avgLat_{projection}.nc",
    )


def wave_cache_is_current(wave, sim_name, field_key, wave_name, wave_box):
    projection = "direct" if wave_name == "Easterly" else "symmetric"
    return (
        wave.attrs.get("field") == field_key
        and wave.attrs.get("wave_name") == wave_name
        and wave.attrs.get("source_simulation") == sim_name
        and wave.attrs.get("filter_name") == wave_box["filter_name"]
        and wave.attrs.get("filter_lat_bounds")
        == f"{WAVE_FILTER_LAT_BOUNDS[0]:g},{WAVE_FILTER_LAT_BOUNDS[1]:g}"
        and wave.attrs.get("latitude_projection") == projection
        and wave.attrs.get("wave_box")
        == json.dumps(jsonable_value(wave_box), sort_keys=True)
    )


def filter_wave_latmean(fields, sim_name, field_key, wave_name, wave_box):
    if field_key not in fields:
        return None
    cache_path = wave_cache_path(sim_name, field_key, wave_name)
    if os.path.exists(cache_path):
        with xr.open_dataset(cache_path) as cached:
            wave = cached["wave"]
            if wave_cache_is_current(
                wave, sim_name, field_key, wave_name, wave_box
            ):
                progress(f"{sim_name}: using shared {wave_name} {field_key} filter cache")
                return wave.load()
        progress(f"{sim_name}: regenerating stale {wave_name} {field_key} filter cache")
    if kf_filter is None:
        raise ModuleNotFoundError(
            "waveFilter.kf_filter is required for lifecycle wave composites"
        )

    da = subset_lat(fields[field_key], lat_bounds=WAVE_FILTER_LAT_BOUNDS)
    if "latitude" in da.dims:
        da = da.mean(dim="latitude", skipna=True)
    if "longitude" not in da.dims:
        return None
    da = da.transpose("Time", "longitude")
    obs_time = da.Time.load()
    lon = da.longitude.load()
    obs = np.asarray(da.load().values, dtype="f")
    obs = obs - np.nanmean(obs, axis=0, keepdims=True)
    obs = np.where(np.isfinite(obs), obs, 0.0)

    k0, k1 = wave_box["k"]
    p0, p1 = wave_box["period"]
    h0, h1 = wave_box["depth"]
    filter_name = wave_box["filter_name"]
    filt_time = np.arange(0, obs.shape[0] + 702)
    filt_data = np.zeros((len(filt_time), obs.shape[1]), dtype="f")
    filt_data[: obs.shape[0], :] = obs

    progress(f"{sim_name}: filtering {field_key} for {wave_name}")
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
            source_simulation=sim_name,
            filter_lat_bounds=f"{WAVE_FILTER_LAT_BOUNDS[0]:g},{WAVE_FILTER_LAT_BOUNDS[1]:g}",
            latitude_projection=("direct" if wave_name == "Easterly" else "symmetric"),
            avgLat=1,
            wave_box=json.dumps(jsonable_value(wave_box), sort_keys=True),
            units=fields[field_key].attrs.get("units", ""),
        ),
    )
    if field_key[0] in ("u", "v", "w", "T") and field_key[1:].replace(".", "", 1).isdigit():
        wave.attrs["cache_version"] = "pressure-axis-fixed"
    tmp = cache_path + f".tmp.{os.getpid()}"
    netcdf_safe_dataset(wave.to_dataset()).to_netcdf(tmp)
    os.replace(tmp, cache_path)
    return wave


WAVE_FIELD_LABELS = {
    "Precip": "precipitation",
    "OLR": "OLR",
    "PW": "PW",
    "cldfrac": "vertically averaged cloud fraction",
    "u850": "850-hPa zonal wind",
    "u250": "250-hPa zonal wind",
    "u100": "100-hPa zonal wind",
    "v850": "850-hPa meridional wind",
    "v250": "250-hPa meridional wind",
    "w500": "500-hPa vertical velocity",
    "w250": "250-hPa vertical velocity",
    "T250": "250-hPa temperature",
    "T100": "100-hPa temperature",
}


WAVE_FIELD_UNITS = {
    "Precip": "mm day$^{-1}$",
    "OLR": "W m$^{-2}$",
    "PW": "kg m$^{-2}$",
    "cldfrac": "fraction",
    "u850": "m s$^{-1}$",
    "u250": "m s$^{-1}$",
    "u100": "m s$^{-1}$",
    "v850": "m s$^{-1}$",
    "v250": "m s$^{-1}$",
    "w500": "m s$^{-1}$",
    "w250": "m s$^{-1}$",
    "T250": "K",
    "T100": "K",
}


def wave_composite_key(wave_name, field_name):
    return f"wave_{wave_name.lower()}_{field_name.lower()}"


WAVE_COMPOSITE_FIELDS = {
    wave_composite_key(wave_name, field_name): (wave_name, field_name)
    for wave_name in LIFECYCLE_WAVE_NAMES
    for field_name in LIFECYCLE_WAVE_FIELDS
}

# Lifecycle composites only sample tracks in the deep tropics plus the local
# latitude window around those tracks. Keeping composite-only fields to this
# latitude range avoids expensive global anomaly baselines for TTL diagnostics.
COMPOSITE_FIELD_LAT_BOUNDS = (
    EVENT_LAT_BOUNDS[0] - max(EVENT_LAT_HALF_WIDTH, float(np.nanmax(np.abs(SPATIAL_REL_LAT_GRID)))) - 5.0,
    EVENT_LAT_BOUNDS[1] + max(EVENT_LAT_HALF_WIDTH, float(np.nanmax(np.abs(SPATIAL_REL_LAT_GRID)))) + 5.0,
)


def trim_to_composite_latitudes(da):
    if da is None or "latitude" not in da.dims:
        return da
    return subset_lat(da, lat_bounds=COMPOSITE_FIELD_LAT_BOUNDS)


def available_composite_keys(fields):
    available = ["precip"] if "precip" in fields else []
    core_requirements = {
        "pw_anom": "pw",
        "convective_precipitating": "convective_precipitating_high_cloud_mask",
        "stratiform_precipitating": "stratiform_precipitating_high_cloud_mask",
        "nonprecipitating": "nonprecipitating_high_cloud_mask",
        "cloud_top_height": "cloud_top_height",
        "cloud_top_pressure": "cloud_top_pressure",
        "N2_TTL_anom": "N2_TTL",
        "u_TTL_anom": "u_TTL",
        "dudz_TTL_anom": "dudz_TTL",
    }
    for key, field_key in core_requirements.items():
        if field_key in fields:
            available.append(key)
    for key, (_, field_name) in WAVE_COMPOSITE_FIELDS.items():
        if field_name in fields:
            available.append(key)
    return available


def composite_spec_metadata(key, fields=None):
    if key == "precip":
        return dict(label="Precipitation", units="mm day$^{-1}$", cmap="YlGnBu", anomaly=False)
    if key == "pw_anom":
        units = fields["pw"].attrs.get("units", "") if fields and "pw" in fields else ""
        return dict(label="PW anomaly", units=units, cmap="BrBG", anomaly=True)
    if key == "convective_precipitating":
        return dict(label="Convective precipitating high cloud", units="fraction", cmap="Reds", anomaly=False)
    if key == "stratiform_precipitating":
        return dict(label="Stratiform precipitating high cloud", units="fraction", cmap="YlGnBu", anomaly=False)
    if key == "nonprecipitating":
        return dict(label="Non-precipitating high cloud", units="fraction", cmap="Greys", anomaly=False)
    if key == "cloud_top_height":
        return dict(label="Cloud-top height", units="km", cmap="magma", anomaly=False)
    if key == "cloud_top_pressure":
        return dict(label="Cloud-top pressure", units="hPa", cmap="magma_r", anomaly=False)
    if key == "N2_TTL_anom":
        return dict(label=r"$N^2_{TTL}$ anomaly", units="s$^{-2}$", cmap="PuOr_r", anomaly=True)
    if key == "u_TTL_anom":
        return dict(label=r"$u_{TTL}$ anomaly", units="m s$^{-1}$", cmap="RdBu_r", anomaly=True)
    if key == "dudz_TTL_anom":
        return dict(label=r"$du/dz_{TTL}$ anomaly", units="s$^{-1}$", cmap="RdBu_r", anomaly=True)
    if key in WAVE_COMPOSITE_FIELDS:
        wave_name, field_name = WAVE_COMPOSITE_FIELDS[key]
        return dict(
            label=f"{wave_name}-filtered {WAVE_FIELD_LABELS[field_name]}",
            units=WAVE_FIELD_UNITS[field_name],
            cmap="RdBu_r",
            anomaly=True,
            wave_name=wave_name,
            wave_field=field_name,
        )
    raise KeyError(f"Unrecognised composite variable: {key}")


def build_composite_spec(fields, sim_name, key):
    meta = composite_spec_metadata(key, fields)
    if key == "precip":
        da = fields["precip"]
    elif key == "pw_anom":
        da = cached_anomaly(
            trim_to_composite_latitudes(fields["pw"]),
            sim_name,
            "PW",
            label=f"{sim_name} PW",
        )
    elif key == "convective_precipitating":
        da = fields["convective_precipitating_high_cloud_mask"].astype(float)
    elif key == "stratiform_precipitating":
        da = fields["stratiform_precipitating_high_cloud_mask"].astype(float)
    elif key == "nonprecipitating":
        da = fields["nonprecipitating_high_cloud_mask"].astype(float)
    elif key == "cloud_top_height":
        da = fields["cloud_top_height"]
    elif key == "cloud_top_pressure":
        da = fields["cloud_top_pressure"]
    elif key == "N2_TTL_anom":
        da = cached_anomaly(
            trim_to_composite_latitudes(fields["N2_TTL"]),
            sim_name,
            "N2_TTL",
            label=f"{sim_name} N2_TTL",
        )
    elif key == "u_TTL_anom":
        da = cached_anomaly(
            trim_to_composite_latitudes(fields["u_TTL"]),
            sim_name,
            "u_TTL",
            label=f"{sim_name} u_TTL",
        )
    elif key == "dudz_TTL_anom":
        da = cached_anomaly(
            trim_to_composite_latitudes(fields["dudz_TTL"]),
            sim_name,
            "dudz_TTL",
            label=f"{sim_name} dudz_TTL",
        )
    elif key in WAVE_COMPOSITE_FIELDS:
        wave_name, field_name = WAVE_COMPOSITE_FIELDS[key]
        da = filter_wave_latmean(
            fields,
            sim_name,
            field_name,
            wave_name,
            WAVE_BOXES[wave_name],
        )
        if da is None:
            return None
        fields[key] = da
    else:
        return None

    spec = dict(meta)
    spec["da"] = trim_to_composite_latitudes(da)
    return spec


# -----------------------------------------------------------------------------
# Event-centred compositing
# -----------------------------------------------------------------------------


def lon_series_relative_to_event(series, center_lon, rel_lon_grid):
    lon = np.asarray(series.longitude.values, dtype=float) % 360.0
    vals = np.asarray(series.values, dtype=float)
    order = np.argsort(lon)
    lon = lon[order]
    vals = vals[order]
    lon_ext = np.concatenate([lon - 360.0, lon, lon + 360.0])
    vals_ext = np.concatenate([vals, vals, vals])
    target = (float(center_lon) % 360.0) + rel_lon_grid
    return np.interp(target, lon_ext, vals_ext, left=np.nan, right=np.nan)


def lat_window_mean(slice_da, center_lat, half_width=EVENT_LAT_HALF_WIDTH):
    if "latitude" not in slice_da.dims:
        return slice_da
    lo = max(float(slice_da.latitude.min()), center_lat - half_width)
    hi = min(float(slice_da.latitude.max()), center_lat + half_width)
    sub = slice_da.sel(latitude=slice(lo, hi))
    if sub.sizes.get("latitude", 0) == 0:
        sub = subset_lat(slice_da, lat_bounds=EVENT_LAT_BOUNDS)
    if "latitude" in sub.dims:
        sub = sub.weighted(np.cos(np.deg2rad(sub.latitude))).mean(dim="latitude", skipna=True)
    return sub


def composite_lag_products(
    da,
    events,
    track_paths,
    lag_days,
    lag_steps,
    rel_lon_grid=REL_LON_GRID,
):
    hov_sums = np.zeros((len(lag_steps), len(rel_lon_grid)), dtype=float)
    hov_sums2 = np.zeros_like(hov_sums)
    hov_counts = np.zeros_like(hov_sums)
    scalar_sums = np.zeros(len(lag_steps), dtype=float)
    scalar_sums2 = np.zeros_like(scalar_sums)
    scalar_counts = np.zeros_like(scalar_sums)
    no_longitude = "longitude" not in da.dims

    for event in events.itertuples(index=False):
        for ilag, step in enumerate(lag_steps):
            it = int(event.onset_time_index) + int(step)
            if it < 0 or it >= da.sizes.get("Time", 0):
                continue

            center_lat, center_lon, _ = track_center_at_index(
                event, it, track_paths
            )

            x = da.isel(Time=it)
            x = lat_window_mean(x, center_lat).load()
            if "longitude" in x.dims:
                vals = lon_series_relative_to_event(x, center_lon, rel_lon_grid)
                rel_lon = ((x.longitude.values.astype(float) - center_lon + 180.0) % 360.0) - 180.0
                mask = np.abs(rel_lon) <= FINGERPRINT_LON_WINDOW_DEG
                scalar_value = (
                    float(x.isel(longitude=mask).mean(dim="longitude", skipna=True).values)
                    if np.any(mask)
                    else np.nan
                )
            else:
                scalar_value = float(x.values)
                vals = np.full(len(rel_lon_grid), scalar_value, dtype=float)

            ok = np.isfinite(vals)
            hov_sums[ilag, ok] += vals[ok]
            hov_sums2[ilag, ok] += vals[ok] ** 2
            hov_counts[ilag, ok] += 1.0

            if np.isfinite(scalar_value):
                scalar_sums[ilag] += scalar_value
                scalar_sums2[ilag] += scalar_value ** 2
                scalar_counts[ilag] += 1.0

    hov_mean = np.full_like(hov_sums, np.nan)
    np.divide(hov_sums, hov_counts, out=hov_mean, where=hov_counts > 0)
    hov_var = np.full_like(hov_sums, np.nan)
    hov_valid = hov_counts > 1
    hov_var[hov_valid] = (
        hov_sums2[hov_valid] - hov_sums[hov_valid] ** 2 / hov_counts[hov_valid]
    ) / (hov_counts[hov_valid] - 1)
    hov_stderr = np.full_like(hov_sums, np.nan)
    hov_stderr[hov_valid] = np.sqrt(hov_var[hov_valid] / hov_counts[hov_valid])
    hov_rms = np.full_like(hov_sums, np.nan)
    np.divide(hov_sums2, hov_counts, out=hov_rms, where=hov_counts > 0)
    hov_rms = np.sqrt(hov_rms)
    hov = xr.Dataset(
        data_vars=dict(
            mean=(("lag_days", "relative_longitude"), hov_mean),
            count=(("lag_days", "relative_longitude"), hov_counts),
            stderr=(("lag_days", "relative_longitude"), hov_stderr),
            rms=(("lag_days", "relative_longitude"), hov_rms),
        ),
        coords=dict(lag_days=lag_days, relative_longitude=rel_lon_grid),
        attrs=dict(
            no_longitude=bool(no_longitude),
            composite_weighting="event_equal",
        ),
    )

    scalar_mean = np.full_like(scalar_sums, np.nan)
    np.divide(scalar_sums, scalar_counts, out=scalar_mean, where=scalar_counts > 0)
    scalar_var = np.full_like(scalar_sums, np.nan)
    scalar_valid = scalar_counts > 1
    scalar_var[scalar_valid] = (
        scalar_sums2[scalar_valid]
        - scalar_sums[scalar_valid] ** 2 / scalar_counts[scalar_valid]
    ) / (scalar_counts[scalar_valid] - 1)
    scalar_stderr = np.full_like(scalar_sums, np.nan)
    scalar_stderr[scalar_valid] = np.sqrt(
        scalar_var[scalar_valid] / scalar_counts[scalar_valid]
    )
    scalar_rms = np.full_like(scalar_sums, np.nan)
    np.divide(scalar_sums2, scalar_counts, out=scalar_rms, where=scalar_counts > 0)
    scalar_rms = np.sqrt(scalar_rms)
    scalar = xr.Dataset(
        data_vars=dict(
            mean=("lag_days", scalar_mean),
            count=("lag_days", scalar_counts),
            stderr=("lag_days", scalar_stderr),
            rms=("lag_days", scalar_rms),
        ),
        coords=dict(lag_days=lag_days),
        attrs=dict(composite_weighting="event_equal"),
    )
    return {"hov": hov, "scalar": scalar}


def event_centred_patch(
    da,
    it,
    center_lat,
    center_lon,
    rel_lat_grid=SPATIAL_REL_LAT_GRID,
    rel_lon_grid=SPATIAL_REL_LON_GRID,
):
    if not {"latitude", "longitude"}.issubset(da.dims):
        return None
    x = da.isel(Time=it)
    lon360 = (x.longitude.astype(float) % 360.0)
    x = x.assign_coords(longitude=lon360).sortby("longitude")
    x_ext = xr.concat(
        [
            x.assign_coords(longitude=x.longitude - 360.0),
            x,
            x.assign_coords(longitude=x.longitude + 360.0),
        ],
        dim="longitude",
    )
    target_lat = float(center_lat) + rel_lat_grid
    target_lon = (float(center_lon) % 360.0) + rel_lon_grid
    patch = x_ext.interp(latitude=target_lat, longitude=target_lon)
    patch = patch.assign_coords(latitude=rel_lat_grid, longitude=rel_lon_grid)
    patch = patch.rename(latitude="relative_latitude", longitude="relative_longitude")
    return patch.load()


def composite_spatial(da, events, track_paths, selected_lags_days, step_days):
    lag_steps = [int(round(lag / step_days)) for lag in selected_lags_days]
    out = {}
    for lag_day, step in zip(selected_lags_days, lag_steps):
        sums = np.zeros((len(SPATIAL_REL_LAT_GRID), len(SPATIAL_REL_LON_GRID)), dtype=float)
        sums2 = np.zeros_like(sums)
        counts = np.zeros_like(sums)
        for event in events.itertuples(index=False):
            it = int(event.onset_time_index) + step
            if it < 0 or it >= da.sizes.get("Time", 0):
                continue
            center_lat, center_lon, _ = track_center_at_index(
                event, it, track_paths
            )
            patch = event_centred_patch(da, it, center_lat, center_lon)
            if patch is None:
                continue
            vals = np.asarray(patch.values, dtype=float)
            ok = np.isfinite(vals)
            sums[ok] += vals[ok]
            sums2[ok] += vals[ok] ** 2
            counts[ok] += 1.0
        mean = np.full_like(sums, np.nan)
        np.divide(sums, counts, out=mean, where=counts > 0)
        out[lag_day] = xr.Dataset(
            data_vars=dict(mean=(("relative_latitude", "relative_longitude"), mean), count=(("relative_latitude", "relative_longitude"), counts)),
            coords=dict(relative_latitude=SPATIAL_REL_LAT_GRID, relative_longitude=SPATIAL_REL_LON_GRID),
        )
    return out


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def finite_percentile(arrays, q, default=1.0):
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays if a is not None])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return default
    value = float(np.nanpercentile(vals, q))
    return value if np.isfinite(value) and value != 0 else default


def plot_hovmoller_group(composites, specs, var_names, stem_suffix):
    cols_to_plot = sim_names + ["nudged u - control", "nudged T - control"]
    fig, ax = plt.subplots(
        len(var_names),
        len(cols_to_plot),
        figsize=(3.0 * len(cols_to_plot), 2.35 * len(var_names)),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    if len(var_names) == 1:
        ax = ax[None, :]

    for r, var in enumerate(var_names):
        spec = specs[var]
        sim_arrays = [composites[sim][var]["hov"].mean.values for sim in sim_names if var in composites[sim]]
        vmax = finite_percentile(sim_arrays, 98.0, default=1.0)
        vmin = finite_percentile(sim_arrays, 2.0, default=0.0)
        if spec.get("anomaly", False):
            lim = finite_percentile([np.abs(a) for a in sim_arrays], 98.0, default=1.0)
            levels = symmetric_levels(lim, ndiv=15)
            cmap = spec.get("cmap", "RdBu_r")
        else:
            levels = nice_levels(vmin, vmax, n=17)
            cmap = spec.get("cmap", "YlGnBu")
        diff_arrays = []
        if var in composites["control"]:
            ctrl = composites["control"][var]["hov"].mean
            for sim in sim_names[1:]:
                if var in composites[sim]:
                    diff_arrays.append((composites[sim][var]["hov"].mean - ctrl).values)
        diff_lim = finite_percentile([np.abs(a) for a in diff_arrays], 98.0, default=1.0)
        diff_levels = symmetric_levels(diff_lim, ndiv=15)

        for c, label in enumerate(cols_to_plot):
            a = ax[r, c]
            if c < len(sim_names):
                sim = sim_names[c]
                if var not in composites[sim]:
                    a.set_axis_off()
                    continue
                ds = composites[sim][var]["hov"]
                plot_cmap = difference_cmap(levels, cmap=cmap) if spec.get("anomaly", False) else cmap
                im = a.contourf(ds.relative_longitude, ds.lag_days, ds.mean, levels=levels, cmap=plot_cmap, extend="both")
                if r == 0:
                    n = int(composites[sim]["event_count"])
                    a.set_title(f"{sim}\nN={n}", fontsize=9)
            else:
                sim = sim_names[c - 2]
                if var not in composites[sim] or var not in composites["control"]:
                    a.set_axis_off()
                    continue
                ds = composites[sim][var]["hov"].mean - composites["control"][var]["hov"].mean
                im = a.contourf(ds.relative_longitude, ds.lag_days, ds, levels=diff_levels, cmap=difference_cmap(diff_levels, cmap="RdBu_r"), extend="both")
                if r == 0:
                    a.set_title(label, fontsize=9)
            a.axhline(0.0, color="0.2", lw=0.7)
            a.axvline(0.0, color="0.2", lw=0.7)
            if c == 0:
                a.set_ylabel(f"{spec['label']}\nLag (days)")
            if r == len(var_names) - 1:
                a.set_xlabel("Relative longitude")
            if c == len(cols_to_plot) - 1:
                cbar = fig.colorbar(im, ax=ax[r, :], location="right", shrink=0.85)
                cbar.set_label(spec.get("units", ""))
    fig.suptitle("Event-equal track-following lifecycle composites", fontsize=11)
    save_figure(1, fig, f"lifecycle_hovmoller_composites_{stem_suffix}")
    plt.close(fig)


def plot_difference_fingerprints(composites, specs, var_names):
    fig, ax = plt.subplots(len(var_names), 1, figsize=(8.0, 2.0 * len(var_names)), constrained_layout=True, sharex=True)
    ax = np.atleast_1d(ax)
    ctrl = composites["control"]
    for r, var in enumerate(var_names):
        a = ax[r]
        if var not in ctrl:
            a.set_axis_off()
            continue
        cmean = ctrl[var]["scalar"].mean
        cse = ctrl[var]["scalar"].stderr
        for i, sim in enumerate(sim_names[1:], start=1):
            if var not in composites[sim]:
                continue
            ds = composites[sim][var]["scalar"]
            diff = ds.mean - cmean
            se = np.sqrt(ds.stderr ** 2 + cse ** 2)
            a.plot(ds.lag_days, diff, color=cols[i], lw=1.6, label=f"{sim} - control")
            a.fill_between(ds.lag_days.values, (diff - 1.96 * se).values, (diff + 1.96 * se).values, color=cols[i], alpha=0.18, lw=0)
        a.axvline(0.0, color="0.25", lw=0.8)
        a.axhline(0.0, color="0.25", lw=0.7)
        a.set_ylabel(f"{specs[var]['label']}\ndifference")
        if r == 0:
            a.legend(frameon=False, loc="best")
    ax[-1].set_xlabel("Lag (days)")
    fig.suptitle("Event-equal lifecycle differences from control", fontsize=11)
    save_figure(2, fig, "difference_lifecycle_fingerprints")
    plt.close(fig)


def plot_spatial_composites(spatial_products, specs, var_names):
    for var in var_names:
        available_sims = [sim for sim in sim_names if var in spatial_products.get(sim, {})]
        if not available_sims:
            continue
        arrays = []
        for sim in available_sims:
            arrays.extend([ds.mean.values for ds in spatial_products[sim][var].values()])
        if specs[var].get("anomaly", False):
            lim = finite_percentile([np.abs(a) for a in arrays], 98.0, default=1.0)
            levels = symmetric_levels(lim, ndiv=15)
            cmap = specs[var].get("cmap", "RdBu_r")
        else:
            vmin = finite_percentile(arrays, 2.0, default=0.0)
            vmax = finite_percentile(arrays, 98.0, default=1.0)
            levels = nice_levels(vmin, vmax, n=17)
            cmap = specs[var].get("cmap", "YlGnBu")

        fig, ax = plt.subplots(
            len(available_sims),
            len(SPATIAL_LAGS_DAYS),
            figsize=(2.65 * len(SPATIAL_LAGS_DAYS), 2.35 * len(available_sims)),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        ax = np.atleast_2d(ax)
        im = None
        for r, sim in enumerate(available_sims):
            for c, lag in enumerate(SPATIAL_LAGS_DAYS):
                a = ax[r, c]
                ds = spatial_products[sim][var][lag]
                im = a.contourf(ds.relative_longitude, ds.relative_latitude, ds.mean, levels=levels, cmap=cmap, extend="both")
                a.axhline(0.0, color="0.25", lw=0.6)
                a.axvline(0.0, color="0.25", lw=0.6)
                if r == 0:
                    a.set_title(f"{lag:+g} d", fontsize=9)
                if c == 0:
                    a.set_ylabel(sim)
                if r == len(available_sims) - 1:
                    a.set_xlabel("Rel. lon")
        if im is not None:
            cbar = fig.colorbar(im, ax=ax, location="right", shrink=0.9)
            cbar.set_label(specs[var].get("units", ""))
        fig.suptitle(f"Event-equal spatial composite: {specs[var]['label']}", fontsize=12)
        save_figure(3, fig, f"event_centred_spatial_composites_{var}")
        plt.close(fig)


def wave_filter_title(wave_name):
    box = WAVE_BOXES[wave_name]
    k0, k1 = box["k"]
    p0, p1 = box["period"]
    direction = "E" if k0 > 0 else "W"
    return (
        f"{wave_name}\n{direction} k={min(abs(k0), abs(k1))}--"
        f"{max(abs(k0), abs(k1))}; {p0:g}--{p1:g} d"
    )


def wave_composite_curve_table(composites):
    rows = []
    for sim in sim_names:
        for wave_name in LIFECYCLE_WAVE_NAMES:
            for field_name in LIFECYCLE_WAVE_FIELDS:
                key = wave_composite_key(wave_name, field_name)
                if key not in composites.get(sim, {}):
                    continue
                ds = composites[sim][key]["scalar"]
                for lag, mean, rms, stderr, count in zip(
                    ds.lag_days.values,
                    ds["mean"].values,
                    ds.rms.values,
                    ds.stderr.values,
                    ds["count"].values,
                ):
                    rows.append(
                        dict(
                            simulation=sim,
                            wave=wave_name,
                            field=field_name,
                            lag_days=float(lag),
                            mean=float(mean),
                            rms=float(rms),
                            stderr=float(stderr),
                            object_count=int(count),
                            longitude_half_width_deg=float(FINGERPRINT_LON_WINDOW_DEG),
                            centering="tracked centroid; endpoints held before genesis/after lysis",
                        )
                    )
    return pd.DataFrame(rows)


def wave_composite_method_table():
    rows = []
    for wave_name in LIFECYCLE_WAVE_NAMES:
        box = WAVE_BOXES[wave_name]
        rows.append(
            dict(
                wave=wave_name,
                direction=("eastward" if box["k"][0] > 0 else "westward"),
                zonal_wavenumber_min=min(abs(box["k"][0]), abs(box["k"][1])),
                zonal_wavenumber_max=max(abs(box["k"][0]), abs(box["k"][1])),
                period_min_days=min(box["period"]),
                period_max_days=max(box["period"]),
                equivalent_depth_min_m=(
                    np.nan if box["depth"][0] < 0 else box["depth"][0]
                ),
                equivalent_depth_max_m=(
                    np.nan if box["depth"][1] < 0 else box["depth"][1]
                ),
                latitude_band="10S-10N mean",
                fields=", ".join(LIFECYCLE_WAVE_FIELDS),
                composite_center="moving PyFLEXTRKR centroid",
                reference=(
                    "Rios-Berrios et al. 2023 J Climate"
                    if wave_name == "Easterly"
                    else "Wheeler-Kiladis convention"
                ),
            )
        )
    return pd.DataFrame(rows)


def plot_wave_filtered_track_composites(composites, specs):
    field_groups = (
        ("convective_moisture_cloud", ("Precip", "OLR", "PW", "cldfrac")),
        ("winds_vertical_motion", ("u850", "u250", "u100", "v850", "v250", "w500", "w250")),
        ("temperature", ("T250", "T100")),
    )
    for stem, field_names in field_groups:
        available = [
            field_name
            for field_name in field_names
            if any(
                wave_composite_key(wave_name, field_name) in composites.get(sim, {})
                for sim in sim_names
                for wave_name in LIFECYCLE_WAVE_NAMES
            )
        ]
        if not available:
            continue
        fig, axes = plt.subplots(
            len(available),
            len(LIFECYCLE_WAVE_NAMES),
            figsize=(4.0 * len(LIFECYCLE_WAVE_NAMES), 2.25 * len(available)),
            constrained_layout=True,
            sharex=True,
            squeeze=False,
        )
        for row, field_name in enumerate(available):
            row_arrays = []
            for wave_name in LIFECYCLE_WAVE_NAMES:
                key = wave_composite_key(wave_name, field_name)
                for sim in sim_names:
                    if key in composites.get(sim, {}):
                        row_arrays.append(composites[sim][key]["scalar"]["mean"].values)
            limit = finite_percentile(
                [np.abs(values) for values in row_arrays], 99.0, default=1.0
            )
            for col, wave_name in enumerate(LIFECYCLE_WAVE_NAMES):
                ax = axes[row, col]
                key = wave_composite_key(wave_name, field_name)
                for sim_index, sim in enumerate(sim_names):
                    if key not in composites.get(sim, {}):
                        continue
                    ds = composites[sim][key]["scalar"]
                    ax.plot(
                        ds.lag_days,
                        ds["mean"],
                        color=cols[sim_index],
                        lw=1.6,
                        label=sim if row == 0 and col == 0 else None,
                    )
                    ax.fill_between(
                        ds.lag_days.values,
                        (ds["mean"] - ds.stderr).values,
                        (ds["mean"] + ds.stderr).values,
                        color=cols[sim_index],
                        alpha=0.12,
                        linewidth=0,
                    )
                ax.axvline(0.0, color="0.2", lw=0.8)
                ax.axhline(0.0, color="0.5", lw=0.6)
                ax.set_ylim(-limit, limit)
                if row == 0:
                    ax.set_title(wave_filter_title(wave_name), fontsize=9)
                if col == 0:
                    units = WAVE_FIELD_UNITS[field_name]
                    ax.set_ylabel(f"{WAVE_FIELD_LABELS[field_name]}\n({units})")
                if row == len(available) - 1:
                    ax.set_xlabel("Lag from tracked onset (days)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(sim_names), frameon=False)
        fig.suptitle(
            "Event-equal wave-filtered fields within "
            f"{FINGERPRINT_LON_WINDOW_DEG:g} deg of tracked organised-system centres",
            fontsize=11,
        )
        save_figure(4, fig, f"wave_filtered_track_composites_{stem}")
        plt.close(fig)


def build_track_property_lifecycle(track_time_by_run, events_by_run, step_days):
    rows = []
    for sim in sim_names:
        tt = track_time_by_run.get(sim, pd.DataFrame())
        ev = events_by_run.get(sim, pd.DataFrame())
        if len(tt) == 0 or len(ev) == 0:
            continue
        onset = ev[["track_id", "onset_time_index", "source"]].copy()
        merged = tt.merge(onset, on=["track_id", "source"], how="inner")
        merged["lag_days"] = (merged.time_index - merged.onset_time_index) * step_days
        for record in merged.to_dict("records"):
            rows.append(record)
    lifecycle = pd.DataFrame(rows)
    if len(lifecycle) == 0:
        return lifecycle, pd.DataFrame()

    props = [
        ("area_km2", "object area"),
        ("propagation_speed_kmh", "propagation speed"),
        ("mean_precip", "precipitation intensity"),
        ("min_cloud_top_pressure", "minimum cloud-top pressure"),
        ("max_cloud_top_height", "cloud-top height"),
        ("mean_high_cloud_fraction", "high-cloud fraction"),
    ]
    summary_rows = []
    for sim in sim_names:
        sub = lifecycle[lifecycle.simulation == sim]
        for lag, group in sub.groupby("lag_days"):
            for key, label in props:
                if key not in group:
                    continue
                vals = group[key].values.astype(float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                summary_rows.append(
                    dict(
                        simulation=sim,
                        lag_days=float(lag),
                        property=key,
                        label=label,
                        mean=float(np.nanmean(vals)),
                        p25=float(np.nanpercentile(vals, 25)),
                        p75=float(np.nanpercentile(vals, 75)),
                        count=int(vals.size),
                    )
                )
    return lifecycle, pd.DataFrame(summary_rows)


def plot_track_property_lifecycle(track_summary, tracks_all):
    if len(track_summary) == 0:
        print("Skipping Figure 5: no track lifecycle properties available.")
        return
    props = [
        ("area_km2", "Object area (km$^2$)"),
        ("propagation_speed_kmh", "Speed (km h$^{-1}$)"),
        ("mean_precip", "Precip. intensity"),
        ("max_cloud_top_height", "Cloud-top height"),
        ("mean_high_cloud_fraction", "High-cloud fraction"),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.6), constrained_layout=True)
    ax = ax.ravel()
    for p, (key, ylabel) in enumerate(props):
        a = ax[p]
        sub = track_summary[track_summary.property == key]
        for i, sim in enumerate(sim_names):
            s = sub[sub.simulation == sim].sort_values("lag_days")
            if len(s) == 0:
                continue
            a.plot(s.lag_days, s["mean"], color=cols[i], label=sim)
            a.fill_between(s.lag_days, s.p25, s.p75, color=cols[i], alpha=0.18, lw=0)
        a.axvline(0, color="0.25", lw=0.8)
        a.set_ylabel(ylabel)
        a.set_xlabel("Lag (days)")
    a = ax[-1]
    for i, sim in enumerate(sim_names):
        sub = tracks_all[tracks_all.simulation == sim]
        if len(sub) == 0:
            continue
        a.hist(sub.lifetime_hours, bins=20, histtype="step", density=True, color=cols[i], label=sim)
    a.set_xlabel("Lifetime (h)")
    a.set_ylabel("Density")
    a.legend(frameon=False)
    save_figure(5, fig, "track_property_lifecycle_statistics")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def load_datasets():
    dss = []
    zss = []
    for i, data_loc in enumerate(data_locs):
        progress(f"Loading {sim_names[i]} lat-lon fields")
        ds = load_latlon_dataset(
            os.path.join(root_dir, data_loc),
            latlon_pattern=latlon_file[i],
            time_format=time_format,
            ndays=ndays,
            end_date=analysis_end_date,
            chunks=latlon_chunks,
        )
        ds = standardize_latlon_dataset(ds, needed_vars)
        dss.append(ds)

        progress(f"Loading {sim_names[i]} TTL/zonal fields")
        zds = load_latlon_dataset(
            os.path.join(root_dir, data_loc),
            latlon_pattern=zm_latlon_file[i],
            time_format=time_format,
            ndays=ndays,
            end_date=analysis_end_date,
            chunks=zm_chunks,
        )
        zds = standardize_latlon_dataset(zds, needed_vars)
        zss.append(zds)
    return dss, zss


def build_fields(ds, zds, sim_name):
    precip = get_precip_rate(ds)
    fields = {
        "precip": precip,
        "mean_precip": weighted_tropical_mean(precip, lat_bounds=TROPICAL_LAT_BOUNDS),
    }
    for key in ("pw", "olr"):
        da = time_aligned_field(ds, key, precip, required=False)
        if da is not None:
            fields[key] = da

    classif = load_required_classification(ds, fields, sim_name)
    for key in (
        "cloud_top_pressure", "cloud_top_height", "high_cloud_fraction",
        "high_cloud_mask", "stratiform_precipitating_high_cloud_mask",
        "convective_precipitating_high_cloud_mask",
        "nonprecipitating_high_cloud_mask", "unclassified_high_cloud_mask",
    ):
        if key in classif:
            fields[key] = classif[key]

    for field_name in LIFECYCLE_WAVE_FIELDS:
        fields[field_name] = lifecycle_wave_field(ds, precip, field_name)

    ttl = compute_ttl_fields(ds, precip)
    ttl_fallback = compute_ttl_fields(zds, precip)
    for key, da in ttl_fallback.items():
        ttl.setdefault(key, da)
    fields.update(ttl)
    progress(f"{sim_name}: available lifecycle fields: {', '.join(sorted(fields))}")
    return fields


def write_csv(df, filename):
    if save and df is not None and len(df) > 0:
        path = os.path.join(save_loc, filename)
        df.to_csv(path, index=False)
        progress(f"Wrote {path}")


def main():
    os.makedirs(save_loc, exist_ok=True)
    dss, zss = load_datasets()

    fields_by_run = []
    for sim_name, ds, zds in zip(sim_names, dss, zss):
        fields_by_run.append(build_fields(ds, zds, sim_name))
    fields_by_sim = dict(zip(sim_names, fields_by_run))

    source_name = PRIMARY_TRACK_SOURCE
    for sim_name, fields in zip(sim_names, fields_by_run):
        validate_pyflextrkr_product(source_name, sim_name, fields)
    cached_tables = load_cached_track_event_tables(source_name)
    if cached_tables is None:
        all_objects = []
        all_track_time = []
        all_tracks = []
        events_by_run = {}
        track_time_by_run = {}

        for sim_name, fields in zip(sim_names, fields_by_run):
            objects_df, track_time, tracks_df = build_track_tables_for_source(source_name, fields, sim_name)
            if len(objects_df) == 0:
                print(f"{sim_name}: no tracked objects found for {source_name}")
                events_by_run[sim_name] = pd.DataFrame()
                track_time_by_run[sim_name] = pd.DataFrame()
                continue
            events = select_onset_events(track_time, tracks_df, source_name, sim_name)
            events_by_run[sim_name] = events
            track_time_by_run[sim_name] = track_time
            all_objects.append(objects_df)
            all_track_time.append(track_time)
            all_tracks.append(tracks_df)
            progress(f"{sim_name}: selected {len(events)} onset events from {len(tracks_df)} tracks")

        events_all = pd.concat(events_by_run.values(), ignore_index=True) if events_by_run else pd.DataFrame()
        objects_all = pd.concat(all_objects, ignore_index=True) if all_objects else pd.DataFrame()
        track_time_all = pd.concat(all_track_time, ignore_index=True) if all_track_time else pd.DataFrame()
        tracks_all = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
        write_track_event_cache(source_name, events_all, objects_all, track_time_all, tracks_all)
    else:
        events_by_run, track_time_by_run, events_all, objects_all, track_time_all, tracks_all = cached_tables

    write_csv(events_all, "convective_lifecycle_onset_events.csv")
    write_csv(objects_all, "convective_lifecycle_track_objects.csv")
    write_csv(track_time_all, "convective_lifecycle_track_time.csv")
    write_csv(tracks_all, "convective_lifecycle_track_summary.csv")

    if len(events_all) == 0:
        print("No onset events passed the filters; lifecycle composites were not built.")
        return

    lag_days, lag_steps, step_days = lag_steps_for_reference(fields_by_run[0]["precip"])
    centre_table = build_composite_centre_table(
        events_by_run,
        track_time_by_run,
        fields_by_run,
        lag_days,
        lag_steps,
    )
    write_csv(centre_table, "convective_lifecycle_composite_centres.csv")
    available_specs_by_run = {
        sim_name: set(available_composite_keys(fields))
        for sim_name, fields in zip(sim_names, fields_by_run)
    }
    spec_keys = []
    preferred_keys = [
        "precip", "pw_anom", "convective_precipitating", "stratiform_precipitating", "nonprecipitating",
        "cloud_top_pressure", "cloud_top_height", "N2_TTL_anom",
        "u_TTL_anom", "dudz_TTL_anom",
    ]
    preferred_keys.extend(
        wave_composite_key(wave_name, field_name)
        for wave_name in LIFECYCLE_WAVE_NAMES
        for field_name in LIFECYCLE_WAVE_FIELDS
    )
    for key in preferred_keys:
        if any(key in available_specs_by_run[sim] for sim in sim_names):
            spec_keys.append(key)
    specs = {}
    for key in spec_keys:
        for sim in sim_names:
            if key in available_specs_by_run[sim]:
                fields = fields_by_sim[sim]
                specs[key] = composite_spec_metadata(key, fields)
                break
    include_spatial = should_run_figure(3)
    expected_composite_token = composite_cache_token(
        source_name,
        spec_keys,
        include_spatial,
    )
    composite_signature_valid = cache_metadata_matches(
        composite_cache_metadata_file(),
        composite_cache_signature(source_name, spec_keys, include_spatial),
        "lifecycle composites",
    )
    cached_composites = (
        load_cached_composites(source_name, spec_keys, include_spatial)
        if composite_signature_valid
        else None
    )
    if cached_composites is None:
        event_counts = {
            sim: len(events_by_run.get(sim, pd.DataFrame())) for sim in sim_names
        }
        available_specs_by_sim = {
            sim: [
                var
                for var in spec_keys
                if event_counts[sim] > 0 and var in available_specs_by_run[sim]
            ]
            for sim in sim_names
        }
        # Write the signature before the expensive loop. Completed per-variable
        # products can then be resumed after an interrupted run.
        write_composite_cache_manifest(
            source_name,
            spec_keys,
            include_spatial,
            event_counts,
            available_specs_by_sim,
        )
        composites = {}
        spatial_products = {}
        for sim_name, fields in zip(sim_names, fields_by_run):
            events = events_by_run.get(sim_name, pd.DataFrame())
            track_paths = build_track_path_lookup(
                track_time_by_run.get(sim_name, pd.DataFrame())
            )
            composites[sim_name] = {"event_count": len(events)}
            spatial_products[sim_name] = {}
            if len(events) == 0:
                continue
            for var in spec_keys:
                if var not in available_specs_by_run[sim_name]:
                    continue
                cached_product = load_cached_composite_product(
                    sim_name,
                    var,
                    include_spatial,
                    expected_composite_token,
                )
                if cached_product is not None:
                    product, spatial = cached_product
                    composites[sim_name][var] = product
                    if spatial:
                        spatial_products[sim_name][var] = spatial
                    progress(
                        f"{sim_name}: using cached {var} lifecycle composites"
                    )
                    continue
                spec = build_composite_spec(fields, sim_name, var)
                if spec is None:
                    continue
                da = spec["da"]
                progress(f"{sim_name}: compositing {var}")
                progress(
                    f"{sim_name}: {var} event-equal lag-longitude/scalar composites"
                )
                product = composite_lag_products(
                    da,
                    events,
                    track_paths,
                    lag_days,
                    lag_steps,
                )
                composites[sim_name][var] = product
                spatial = {}
                if (
                    spatial_composite_expected(var, include_spatial)
                    and {"latitude", "longitude"}.issubset(da.dims)
                ):
                    progress(f"{sim_name}: {var} spatial composites")
                    spatial = composite_spatial(
                        da,
                        events,
                        track_paths,
                        SPATIAL_LAGS_DAYS,
                        step_days,
                    )
                    spatial_products[sim_name][var] = spatial
                write_composite_product_cache(
                    sim_name,
                    var,
                    product,
                    spatial,
                    expected_composite_token,
                )
                progress(f"{sim_name}: cached {var} lifecycle composites")
                del da, spec, product, spatial
                gc.collect()
        write_composite_cache_manifest(
            source_name,
            spec_keys,
            include_spatial,
            event_counts,
            available_specs_by_sim,
        )
    else:
        composites, spatial_products = cached_composites

    if should_run_figure(1):
        first_group = [v for v in ("precip", "pw_anom", "convective_precipitating", "stratiform_precipitating", "nonprecipitating", "cloud_top_pressure", "cloud_top_height") if v in specs]
        second_group = [v for v in ("N2_TTL_anom", "u_TTL_anom", "dudz_TTL_anom") if v in specs]
        if first_group:
            plot_hovmoller_group(composites, specs, first_group, "core")
        if second_group:
            plot_hovmoller_group(composites, specs, second_group, "ttl_wave")

    if should_run_figure(2):
        diff_vars = [v for v in ("pw_anom", "precip", "convective_precipitating", "stratiform_precipitating", "nonprecipitating", "cloud_top_pressure", "cloud_top_height", "N2_TTL_anom", "dudz_TTL_anom") if v in specs]
        plot_difference_fingerprints(composites, specs, diff_vars)

    if should_run_figure(3):
        spatial_vars = [v for v in ("precip", "pw_anom", "convective_precipitating", "stratiform_precipitating", "nonprecipitating", "cloud_top_pressure", "cloud_top_height", "N2_TTL_anom") if v in specs]
        plot_spatial_composites(spatial_products, specs, spatial_vars)

    if should_run_figure(4):
        write_csv(
            wave_composite_method_table(),
            "convective_lifecycle_wave_filter_methods.csv",
        )
        wave_curve_table = wave_composite_curve_table(composites)
        write_csv(
            wave_curve_table,
            "convective_lifecycle_wave_filtered_track_composites.csv",
        )
        plot_wave_filtered_track_composites(composites, specs)

    if should_run_figure(5):
        cached_lifecycle = load_cached_track_property_lifecycle(source_name, step_days)
        if cached_lifecycle is None:
            lifecycle, track_summary = build_track_property_lifecycle(track_time_by_run, events_by_run, step_days)
            write_track_property_lifecycle_cache(source_name, step_days, lifecycle, track_summary)
        else:
            lifecycle, track_summary = cached_lifecycle
        write_csv(lifecycle, "convective_lifecycle_track_property_samples.csv")
        write_csv(track_summary, "convective_lifecycle_track_property_summary.csv")
        plot_track_property_lifecycle(track_summary, tracks_all)

    print("\nFinished convective_lifecycle_composite_diagnostics.py")


if __name__ == "__main__":
    main()
