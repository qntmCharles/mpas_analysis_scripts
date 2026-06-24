#!/usr/bin/env python
# coding: utf-8

"""
Convective lifecycle composites centred on tracked deep-convective onset.

This diagnostic uses existing high-cloud classification caches and
PyFLEXTRKR-style track outputs where they are available.  Its main event set is
the first appearance of tracked deep, column-connected high cloud; a
precipitation-only onset set is retained as a sensitivity option.

# WHAT TO LOOK FOR:
#   * In the lag-longitude Hovmollers, a moisture signal that peaks earlier or
#     later than lag 0 diagnoses preconditioning timing rather than only mean PW.
#   * TTL N2 and shear anomalies before lag 0 separate static-stability and wind
#     pathways for the T-nudged and U-nudged experiments.
#   * Post-onset precipitation, cloud-top, and high-cloud persistence diagnose
#     whether nudging changes storm longevity rather than just onset amplitude.
#   * Kelvin phase histograms test whether event onsets remain locked to the
#     same wave phase across simulations.
"""

import glob
import json
import os
import sys
import warnings
from datetime import datetime

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
from matplotlib import colors, pyplot as plt
from scipy import ndimage
from scipy.signal import hilbert

from diagnostic_helpers import *
from high_cloud_classification import (
    CLASSIFICATION_CACHE_VERSION,
    HighCloudThresholds,
    cached_classification_is_current as shared_cached_classification_is_current,
)

try:
    from waveFilter import kf_filter
except ModuleNotFoundError:
    kf_filter = None


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/convective_lifecycle/")

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: lifecycle Hovmoller composites
#   2: mechanistic fingerprint summary panels
#   3: difference-lifecycle fingerprints
#   4: event-centred spatial composites
#   5: wave-convection phase-locking diagnostics
#   6: track-property lifecycle statistics
#   7: mechanistic synthesis figure
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

# Event and object filters.  The defaults are deliberately simple so threshold
# sensitivity can be changed near the top of the file.
EVENT_SET = "deep"  # "deep" or "all_precip"
MIN_EVENT_LIFETIME_HOURS = 18.0
MIN_EVENT_AREA_KM2 = 800.0
MIN_EVENT_MAX_PRECIP_MM_DAY = 5.0
MIN_EVENT_SEPARATION_HOURS = 24.0
MIN_EVENT_SEPARATION_KM = 1200.0
MAX_EVENTS_PER_RUN = None

PRECIP_ONSET_THRESHOLD_MM_DAY = 24.0  # 1 mm h-1
DEEP_PRECIP_THRESHOLD_MM_DAY = 5.0
MIN_OBJECT_CELLS = 1
MIN_OBJECT_AREA_KM2 = 0.0

CF_THRESH = 0.05
Z_HIGH = 12.0       # km
Z_LOW_MIN = 2.0     # km
Z_LOW_MAX = 5.0     # km
Z_MID_MIN = 5.0     # km
Z_MID_MAX = 10.0    # km
Z_UPPER_MIN = 10.0  # km
P_WEAK = 1.0        # mm day-1
MAX_CLEAR_GAP = 3.0 # km
ANVIL_RADIUS_DEG = 5.0
ANVIL_TIME_STEPS = 1

EARTH_RADIUS_KM = 6371.0
LONGITUDE_PERIOD_DEG = 360.0

KELVIN_FILTER = dict(k=(1, 14), period=(2.5, 20.0), depth=(8, 90), filter_name="Kelvin")
MJO_LIKE_FILTER = dict(k=(1, 5), period=(30.0, 96.0), depth=(-999.0, -999.0), filter_name="None")
WAVE_FILTER_FIELD = "precip"

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
    )
)


# -----------------------------------------------------------------------------
# Small shared helpers
# -----------------------------------------------------------------------------


def progress(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def save_figure(number, fig, stem, dpi=300):
    global figsave
    if figsave is None:
        figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)
    figsave.save(fig, stem, number=number, dpi=dpi)


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


# -----------------------------------------------------------------------------
# Store-backed derived-product caches
# -----------------------------------------------------------------------------

TRACK_EVENT_CACHE_VERSION = "lifecycle_track_events_v1"
COMPOSITE_CACHE_VERSION = "lifecycle_composites_v1"
PHASE_CACHE_VERSION = "lifecycle_kelvin_phase_v1"
TRACK_PROPERTY_CACHE_VERSION = "lifecycle_track_properties_v1"


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def cache_path(filename):
    return os.path.join(ensure_cache_dir(), filename)


def jsonable_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable_value(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
        event_set=EVENT_SET,
        event_lat_bounds=list(EVENT_LAT_BOUNDS),
        min_event_lifetime_hours=float(MIN_EVENT_LIFETIME_HOURS),
        min_event_area_km2=float(MIN_EVENT_AREA_KM2),
        min_event_max_precip_mm_day=float(MIN_EVENT_MAX_PRECIP_MM_DAY),
        min_event_separation_hours=float(MIN_EVENT_SEPARATION_HOURS),
        min_event_separation_km=float(MIN_EVENT_SEPARATION_KM),
        max_events_per_run=jsonable_value(MAX_EVENTS_PER_RUN),
        precip_onset_threshold_mm_day=float(PRECIP_ONSET_THRESHOLD_MM_DAY),
        deep_precip_threshold_mm_day=float(DEEP_PRECIP_THRESHOLD_MM_DAY),
        cloud_classification_cache_version=CLASSIFICATION_CACHE_VERSION,
        min_object_cells=int(MIN_OBJECT_CELLS),
        min_object_area_km2=float(MIN_OBJECT_AREA_KM2),
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
        wave_filter_field=WAVE_FILTER_FIELD,
        kelvin_filter=jsonable_value(KELVIN_FILTER),
        mjo_like_filter=jsonable_value(MJO_LIKE_FILTER),
    )
    return signature


def phase_cache_signature(source_name):
    signature = object_filter_signature(source_name)
    signature.update(
        cache_version=PHASE_CACHE_VERSION,
        wave_filter_field=WAVE_FILTER_FIELD,
        kelvin_filter=jsonable_value(KELVIN_FILTER),
    )
    return signature


def track_property_cache_signature(source_name, step_days):
    signature = object_filter_signature(source_name)
    signature.update(
        cache_version=TRACK_PROPERTY_CACHE_VERSION,
        step_days=float(step_days),
    )
    return signature


def read_cache_metadata(filename):
    path = cache_path(filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_cache_metadata(filename, signature):
    path = cache_path(filename)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(jsonable_value(signature), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


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
    tmp = path + ".tmp"
    ds.to_netcdf(tmp)
    os.replace(tmp, path)


def read_cache_dataset(filename):
    path = cache_path(filename)
    if not os.path.exists(path):
        return None
    with xr.open_dataset(path) as ds:
        return ds.load()


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


def load_cached_composites(source_name, spec_keys, include_spatial):
    signature = composite_cache_signature(source_name, spec_keys, include_spatial)
    if not cache_metadata_matches(composite_cache_metadata_file(), signature, "lifecycle composites"):
        return None
    metadata = read_cache_metadata(composite_cache_metadata_file()) or {}
    event_counts = metadata.get("event_counts", {})
    available_specs_by_sim = metadata.get("available_specs_by_sim", {})
    composites = {}
    spatial_products = {}
    for sim in sim_names:
        composites[sim] = {"event_count": int(event_counts.get(sim, 0))}
        spatial_products[sim] = {}
        for var in available_specs_by_sim.get(sim, spec_keys):
            hov = read_cache_dataset(composite_cache_filename(sim, var, "hov"))
            scalar = read_cache_dataset(composite_cache_filename(sim, var, "scalar"))
            if hov is None or scalar is None:
                return None
            composites[sim][var] = {"hov": hov, "scalar": scalar}
            if include_spatial:
                spatial_lags = {}
                for lag_day in SPATIAL_LAGS_DAYS:
                    ds = read_cache_dataset(composite_cache_filename(sim, var, "spatial", lag_day=lag_day))
                    if ds is not None:
                        spatial_lags[lag_day] = ds
                if spatial_lags:
                    spatial_products[sim][var] = spatial_lags
    progress(f"Using cached lifecycle composites from {ensure_cache_dir()}")
    return composites, spatial_products


def write_composite_cache(source_name, spec_keys, include_spatial, composites, spatial_products):
    event_counts = {}
    available_specs_by_sim = {}
    for sim in sim_names:
        event_counts[sim] = int(composites.get(sim, {}).get("event_count", 0))
        available_specs_by_sim[sim] = [
            var
            for var in spec_keys
            if var in composites.get(sim, {}) and isinstance(composites[sim].get(var), dict)
        ]
        for var in spec_keys:
            product = composites.get(sim, {}).get(var)
            if product is None:
                continue
            write_cache_dataset(product["hov"], composite_cache_filename(sim, var, "hov"))
            write_cache_dataset(product["scalar"], composite_cache_filename(sim, var, "scalar"))
            if include_spatial:
                for lag_day, ds in spatial_products.get(sim, {}).get(var, {}).items():
                    write_cache_dataset(ds, composite_cache_filename(sim, var, "spatial", lag_day=lag_day))
    signature = composite_cache_signature(source_name, spec_keys, include_spatial)
    signature["event_counts"] = event_counts
    signature["available_specs_by_sim"] = available_specs_by_sim
    write_cache_metadata(composite_cache_metadata_file(), signature)


def phase_cache_files():
    return {
        "metadata": "lifecycle_kelvin_phase_cache_metadata.json",
        "stats": "lifecycle_kelvin_phase_stats.csv",
        "series": "lifecycle_kelvin_phase_series.csv",
    }


def load_cached_phase_products(source_name):
    files = phase_cache_files()
    if not cache_metadata_matches(files["metadata"], phase_cache_signature(source_name), "Kelvin phase diagnostics"):
        return None
    phase_stats = read_cache_csv(files["stats"])
    phase_table = read_cache_csv(files["series"])
    if phase_stats is None or phase_table is None:
        return None
    phase_series = {
        sim: sub["phase_rad"].to_numpy(dtype=float)
        for sim, sub in phase_table.groupby("simulation")
    } if len(phase_table) else {}
    progress(f"Using cached Kelvin phase diagnostics from {ensure_cache_dir()}")
    return phase_stats, phase_series


def write_phase_cache(source_name, phase_stats, phase_series):
    files = phase_cache_files()
    write_cache_csv(phase_stats, files["stats"])
    rows = []
    for sim, phases in phase_series.items():
        for phase in np.asarray(phases, dtype=float):
            rows.append(dict(simulation=sim, phase_rad=float(phase)))
    write_cache_csv(pd.DataFrame(rows), files["series"])
    write_cache_metadata(files["metadata"], phase_cache_signature(source_name))


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


def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "-").replace(".", "p").replace("-", "m")


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


def weighted_lat_mean(da, lat_bounds=TROPICAL_LAT_BOUNDS):
    x = subset_lat(da, lat_bounds=lat_bounds)
    if "latitude" not in x.dims:
        return x
    return x.weighted(np.cos(np.deg2rad(x.latitude))).mean(dim="latitude", skipna=True)


def weighted_tropical_mean(da, lat_bounds=TROPICAL_LAT_BOUNDS):
    x = subset_lat(finite_or_nan(da), lat_bounds=lat_bounds)
    dims = [dim for dim in ("latitude", "longitude") if dim in x.dims]
    if "latitude" in x.dims:
        weights = np.cos(np.deg2rad(x.latitude))
        return x.weighted(weights).mean(dim=dims, skipna=True)
    return x.mean(dim=dims, skipna=True)


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


def lag_steps_for_reference(ref):
    step_days = median_time_step_days(ref)
    lag_days = np.arange(LAG_MIN_DAYS, LAG_MAX_DAYS + 0.5 * step_days, step_days)
    lag_steps = np.rint(lag_days / step_days).astype(int)
    lag_days = lag_steps.astype(float) * step_days
    return lag_days, lag_steps, step_days


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


def haversine_distances_km(lat1, lon1, lat2, lon2, lon_period=LONGITUDE_PERIOD_DEG):
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    lon1 = np.asarray(lon1, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    dlat = lat2 - lat1
    dlon_deg = (lon2 - lon1 + 0.5 * lon_period) % lon_period - 0.5 * lon_period
    dlon = np.deg2rad(dlon_deg)
    a = np.sin(0.5 * dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * dlon) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def circular_mean_lon(lon, weights=None):
    lon = np.asarray(lon, dtype=float) % 360.0
    if weights is None:
        weights = np.ones(lon.shape, dtype=float)
    theta = 2.0 * np.pi * lon / LONGITUDE_PERIOD_DEG
    sin_mean = np.average(np.sin(theta), weights=weights)
    cos_mean = np.average(np.cos(theta), weights=weights)
    return float(np.rad2deg(np.arctan2(sin_mean, cos_mean)) % LONGITUDE_PERIOD_DEG)


def merge_periodic_longitude_labels(labels):
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

    for j in range(labels.shape[0]):
        left = labels[j, 0]
        if left <= 0:
            continue
        for jj in range(max(0, j - 1), min(labels.shape[0], j + 2)):
            union(left, labels[jj, -1])

    roots = np.array([find(i) for i in range(max_label + 1)], dtype=np.int32)
    unique_roots = [root for root in np.unique(roots[1:]) if root > 0]
    remap = {root: index + 1 for index, root in enumerate(unique_roots)}
    out = np.zeros_like(labels)
    positive = labels > 0
    out[positive] = np.array([remap[roots[label]] for label in labels[positive]], dtype=np.int32)
    return out


# -----------------------------------------------------------------------------
# Cloud classification and TTL fields
# -----------------------------------------------------------------------------


def classification_thresholds():
    return HighCloudThresholds(
        cf_thresh=CF_THRESH,
        z_high_km=Z_HIGH,
        low_cloud_min_km=Z_LOW_MIN,
        low_cloud_max_km=Z_LOW_MAX,
        mid_cloud_min_km=Z_MID_MIN,
        mid_cloud_max_km=Z_MID_MAX,
        upper_cloud_min_km=Z_UPPER_MIN,
        max_clear_gap_km=MAX_CLEAR_GAP,
        surface_precip_deep_mm_day=DEEP_PRECIP_THRESHOLD_MM_DAY,
        surface_precip_weak_mm_day=P_WEAK,
    )


def cached_classification_is_current(cached):
    return shared_cached_classification_is_current(cached, classification_thresholds())


REQUIRED_CLASSIFICATION_VARS = (
    "cloud_top_height",
    "high_cloud_fraction",
    "high_cloud_mask",
    "column_connected_mask",
    "active_deep_convection_high_cloud_mask",
    "convectively_generated_high_cloud_mask",
    "cirrus_like_high_cloud_mask",
    "unclassified_high_cloud_mask",
)


def classification_cache_names(sim_name):
    safe = sim_name.replace(" ", "_")
    return [
        f"{safe}_classified_high_cloud.nc",
        f"{safe.lower()}_classified_high_cloud.nc",
        f"{safe_name(sim_name)}_classified_high_cloud.nc",
    ]


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
                if name.endswith("_mask") or name == "column_connected_mask":
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
                if key.endswith("_mask") or key in ("column_connected", "deep_convective"):
                    row[f"frac_{key}"] = float(np.average(v[ok] > 0.5, weights=w))
            else:
                row[f"mean_{key}"] = np.nan
                row[f"max_{key}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def label_objects_with_properties(mask2d, area2d, field_slices=None):
    mask2d = mask2d.transpose("latitude", "longitude")
    area2d = area2d.sel(latitude=mask2d.latitude, longitude=mask2d.longitude)
    mask_vals = np.asarray(mask2d.fillna(False).values, dtype=bool)
    if not np.any(mask_vals):
        return np.zeros(mask_vals.shape, dtype=np.int32), pd.DataFrame()
    labels, _ = ndimage.label(mask_vals, structure=np.ones((3, 3), dtype=np.int8))
    labels = merge_periodic_longitude_labels(labels)
    objects = object_properties_from_labels(labels, mask2d.latitude.values, mask2d.longitude.values, area2d, field_slices)
    if len(objects) == 0:
        return np.zeros(mask_vals.shape, dtype=np.int32), objects
    valid = set(objects.label.astype(int).values)
    labels[~np.isin(labels, list(valid))] = 0
    return labels.astype(np.int32), objects.reset_index(drop=True)


def nearest_previous_track(current_labels, previous_labels, previous_label_to_track, overlap_fraction=0.2):
    out = {}
    current_positive = np.unique(current_labels[current_labels > 0])
    for label_id in current_positive:
        obj = current_labels == label_id
        prev = previous_labels[obj]
        prev = prev[prev > 0]
        if prev.size == 0:
            continue
        labels, counts = np.unique(prev, return_counts=True)
        best_index = int(np.argmax(counts))
        best_label = int(labels[best_index])
        frac = float(counts[best_index] / np.count_nonzero(obj))
        if frac >= overlap_fraction and best_label in previous_label_to_track:
            out[int(label_id)] = previous_label_to_track[best_label]
    return out


def track_objects_from_mask(mask, fields, source_name, sim_name, onset_flag_field=None):
    mask = subset_lat(mask.fillna(False).astype(bool), lat_bounds=TROPICAL_LAT_BOUNDS)
    if "Time" not in mask.dims:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    field_keys = [
        "precip",
        "pw",
        "cloud_top_height",
        "high_cloud_fraction",
        "column_connected_mask",
        "active_deep_convection_high_cloud_mask",
    ]
    available = {key: fields[key] for key in field_keys if key in fields and {"latitude", "longitude"}.issubset(fields[key].dims)}
    align_inputs = [mask] + [available[key] for key in available]
    aligned = xr.align(*align_inputs, join="inner")
    mask = aligned[0]
    available = {key: aligned[i + 1] for i, key in enumerate(available)}

    area2d = grid_cell_area_km2(mask.latitude.values, mask.longitude.values)
    time_days = time_days_from_coord(mask.Time)
    dt_hours = 24.0 * median_time_step_days(mask)

    rows = []
    labels_by_time = []
    next_track_id = 1
    previous_labels = None
    previous_label_to_track = {}

    for it in range(mask.sizes["Time"]):
        field_slices = {key: da.isel(Time=it).load() for key, da in available.items()}
        labels, objects = label_objects_with_properties(mask.isel(Time=it).load(), area2d, field_slices)
        track_lookup = {}
        if previous_labels is not None:
            track_lookup.update(nearest_previous_track(labels, previous_labels, previous_label_to_track))
        for label_id in np.unique(labels):
            if label_id <= 0:
                continue
            if int(label_id) not in track_lookup:
                track_lookup[int(label_id)] = next_track_id
                next_track_id += 1
        labels_by_time.append(labels)
        previous_labels = labels
        previous_label_to_track = track_lookup

        if len(objects) == 0:
            continue
        for record in objects.to_dict("records"):
            label_id = int(record["label"])
            rows.append(
                dict(
                    source=source_name,
                    simulation=sim_name,
                    Time=mask.Time.values[it],
                    time_index=it,
                    time_days=float(time_days[it]),
                    dt_hours=dt_hours,
                    track_id=int(track_lookup[label_id]),
                    tracking_method="local_overlap",
                    onset_flag=True if onset_flag_field is None else bool(record.get(f"frac_{onset_flag_field}", 0.0) > 0.0),
                    **record,
                )
            )

    objects_df = pd.DataFrame(rows)
    track_time, tracks_df = summarise_tracks(objects_df)
    return objects_df, track_time, tracks_df


def chararray_to_strings(array):
    arr = np.asarray(array)
    if arr.dtype.kind in ("S", "U") and arr.ndim == 2:
        return ["".join(x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in row).strip() for row in arr]
    return [str(x) for x in arr.ravel()]


def find_pyflextrkr_work_path(source_name, sim_name):
    safe_sim = sim_name.replace(" ", "_").replace("/", "_")
    source_candidates = [source_name]
    if source_name.endswith("_high_cloud_mask") or source_name in {"high_cloud_mask", "column_connected_mask"}:
        source_candidates.insert(0, f"{source_name}_{CLASSIFICATION_CACHE_VERSION}")
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


def load_pyflextrkr_objects(source_name, sim_name, fields):
    work_path = find_pyflextrkr_work_path(source_name, sim_name)
    if work_path is None:
        return None, None, None

    stats_files = sorted(glob.glob(os.path.join(work_path, "stats", "tracknumbers_*.nc")))
    tracknumbers_path = stats_files[-1]
    progress(f"{sim_name}: reading PyFLEXTRKR track links for {source_name}: {tracknumbers_path}")
    ds_track = xr.open_dataset(tracknumbers_path, mask_and_scale=False, decode_times=False, concat_characters=True)
    track_numbers = np.asarray(ds_track["track_numbers"].squeeze().values)
    if track_numbers.ndim == 1:
        track_numbers = track_numbers[None, :]
    basetimes = np.asarray(ds_track["basetimes"].values).astype(float)
    if "cloudid_files" in ds_track:
        cloud_files = chararray_to_strings(ds_track["cloudid_files"].values)
    else:
        cloud_files = []
    ds_track.close()

    tracking_path = os.path.join(work_path, "tracking")
    if not cloud_files:
        cloud_files = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(tracking_path, "cloudid_*.nc")))]

    precip = fields["precip"]
    time_count = min(len(cloud_files), precip.sizes.get("Time", len(cloud_files)), track_numbers.shape[0])
    if time_count < 1:
        return None, None, None

    rows = []
    dt_hours = 24.0 * median_time_step_days(precip)
    time_days = time_days_from_coord(precip.Time)

    for it in range(time_count):
        path = os.path.join(tracking_path, cloud_files[it])
        if not os.path.exists(path):
            continue
        with xr.open_dataset(path, decode_times=False, mask_and_scale=False) as cloud:
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
        for key in ("precip", "pw", "cloud_top_height", "high_cloud_fraction", "column_connected_mask", "active_deep_convection_high_cloud_mask"):
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
            if track_id <= 0:
                continue
            rows.append(
                dict(
                    source=source_name,
                    simulation=sim_name,
                    Time=precip.Time.values[it],
                    pyflextrkr_basetime=basetimes[it] if it < basetimes.size else np.nan,
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
        elif col.startswith("frac_"):
            agg[col] = (col, "mean")

    track_time = (
        objects_df
        .groupby(["source", "simulation", "track_id", "time_index"], as_index=False)
        .agg(**agg)
        .sort_values(["source", "simulation", "track_id", "time_index"])
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
        for key in ("precip", "cloud_top_height", "high_cloud_fraction", "pw"):
            if f"max_{key}" in group:
                row[f"max_{key}"] = float(group[f"max_{key}"].max(skipna=True))
            if f"mean_{key}" in group:
                row[f"mean_{key}"] = float(group[f"mean_{key}"].mean(skipna=True))
        tracks.append(row)
    return track_time, pd.DataFrame(tracks)


def build_track_tables_for_source(source_name, fields, sim_name):
    objects_df, track_time, tracks_df = load_pyflextrkr_objects(source_name, sim_name, fields)
    if objects_df is not None and len(objects_df) > 0:
        return objects_df, track_time, tracks_df

    if source_name == "active_deep_convection_high_cloud_mask" and "active_deep_convection_high_cloud_mask" in fields:
        progress(f"{sim_name}: building local overlap tracks for active deep convection")
        return track_objects_from_mask(fields["active_deep_convection_high_cloud_mask"], fields, source_name, sim_name)

    if source_name == "precip_ge_1mmh":
        progress(f"{sim_name}: building local overlap tracks for precipitation sensitivity")
        mask = fields["precip"] >= PRECIP_ONSET_THRESHOLD_MM_DAY
        return track_objects_from_mask(mask, fields, source_name, sim_name)

    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def filter_separated_events(candidates):
    if len(candidates) == 0:
        return candidates
    candidates = candidates.sort_values(["time_days", "max_precip", "max_area_km2"], ascending=[True, False, False])
    kept = []
    for row in candidates.to_dict("records"):
        too_close = False
        for prev in kept:
            dt_hours_i = abs(row["time_days"] - prev["time_days"]) * 24.0
            if dt_hours_i > MIN_EVENT_SEPARATION_HOURS:
                continue
            dist = float(haversine_distances_km(row["centroid_lat"], row["centroid_lon"], prev["centroid_lat"], prev["centroid_lon"]))
            if dist <= MIN_EVENT_SEPARATION_KM:
                too_close = True
                break
        if not too_close:
            kept.append(row)
    return pd.DataFrame(kept)


def select_onset_events(track_time, tracks_df, source_name, sim_name):
    if len(track_time) == 0 or len(tracks_df) == 0:
        return pd.DataFrame()
    merged = track_time.merge(
        tracks_df[[
            "source", "simulation", "track_id", "lifetime_hours", "max_area_km2",
            "mean_propagation_speed_kmh", "tracking_method"
        ] + [c for c in ("max_precip", "max_cloud_top_height", "max_high_cloud_fraction") if c in tracks_df.columns]],
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
        if np.isfinite(max_area) and max_area < MIN_EVENT_AREA_KM2:
            continue
        if np.isfinite(max_precip) and max_precip < MIN_EVENT_MAX_PRECIP_MM_DAY:
            continue
        if not (EVENT_LAT_BOUNDS[0] <= float(onset.centroid_lat) <= EVENT_LAT_BOUNDS[1]):
            continue
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
                max_cloud_top_height=float(group.max_cloud_top_height.max(skipna=True)) if "max_cloud_top_height" in group else np.nan,
                max_high_cloud_fraction=float(group.max_high_cloud_fraction.max(skipna=True)) if "max_high_cloud_fraction" in group else np.nan,
                mean_propagation_speed_kmh=float(onset.get("mean_propagation_speed_kmh", np.nan)),
                tracking_method=onset.get("tracking_method", ""),
            )
        )

    events = filter_separated_events(pd.DataFrame(rows))
    if len(events) == 0:
        return events
    events = events.sort_values("time_days").reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1)
    if MAX_EVENTS_PER_RUN is not None and len(events) > MAX_EVENTS_PER_RUN:
        events = events.iloc[:MAX_EVENTS_PER_RUN].copy()
    return events


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


def wave_cache_path(sim_name, field_key, wave_name):
    cache_dir = ensure_cache_dir()
    return os.path.join(
        cache_dir,
        f"{safe_name(sim_name)}_{field_key}_{wave_name}_latmean_{WAVE_FILTER_LAT_BOUNDS[0]:g}_{WAVE_FILTER_LAT_BOUNDS[1]:g}.nc".replace("-", "m"),
    )


def filter_wave_latmean(fields, sim_name, field_key, wave_name, wave_box):
    if field_key not in fields:
        return None
    cache_path = wave_cache_path(sim_name, field_key, wave_name)
    if os.path.exists(cache_path):
        with xr.open_dataset(cache_path) as cached:
            return cached["wave"].load()
    if kf_filter is None:
        warnings.warn("waveFilter.kf_filter is unavailable; Kelvin/MJO lifecycle diagnostics will be skipped.")
        return None

    da = weighted_lat_mean(fields[field_key], lat_bounds=WAVE_FILTER_LAT_BOUNDS)
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
        attrs=dict(field=field_key, wave_name=wave_name, filter_name=filter_name, units=fields[field_key].attrs.get("units", "")),
    )
    tmp = cache_path + ".tmp"
    wave.to_dataset().to_netcdf(tmp)
    os.replace(tmp, cache_path)
    return wave


def prepare_composite_fields(fields, sim_name):
    specs = {}
    specs["precip"] = dict(da=fields["precip"], label="Precipitation", units="mm day$^{-1}$", cmap="YlGnBu", anomaly=False)
    if "pw" in fields:
        specs["pw_anom"] = dict(
            da=anomaly(fields["pw"], f"{sim_name} PW"),
            label="PW anomaly",
            units=fields["pw"].attrs.get("units", ""),
            cmap="BrBG",
            anomaly=True,
        )
    if "column_connected_mask" in fields:
        specs["column_connected"] = dict(da=fields["column_connected_mask"].astype(float), label="Column-connected high cloud", units="fraction", cmap="YlGnBu", anomaly=False)
    if "cloud_top_height" in fields:
        specs["cloud_top_height"] = dict(da=fields["cloud_top_height"], label="Cloud-top height", units="km", cmap="magma", anomaly=False)
    if "N2_TTL" in fields:
        specs["N2_TTL_anom"] = dict(
            da=anomaly(fields["N2_TTL"], f"{sim_name} N2_TTL"),
            label=r"$N^2_{TTL}$ anomaly",
            units="s$^{-2}$",
            cmap="PuOr_r",
            anomaly=True,
        )
    if "u_TTL" in fields:
        specs["u_TTL_anom"] = dict(
            da=anomaly(fields["u_TTL"], f"{sim_name} u_TTL"),
            label=r"$u_{TTL}$ anomaly",
            units="m s$^{-1}$",
            cmap="RdBu_r",
            anomaly=True,
        )
    if "dudz_TTL" in fields:
        specs["dudz_TTL_anom"] = dict(
            da=anomaly(fields["dudz_TTL"], f"{sim_name} dudz_TTL"),
            label=r"$du/dz_{TTL}$ anomaly",
            units="s$^{-1}$",
            cmap="RdBu_r",
            anomaly=True,
        )

    kelvin = filter_wave_latmean(fields, sim_name, WAVE_FILTER_FIELD, "Kelvin", KELVIN_FILTER)
    if kelvin is not None:
        specs["kelvin_anom"] = dict(da=kelvin, label="Filtered Kelvin anomaly", units=kelvin.attrs.get("units", ""), cmap="RdBu_r", anomaly=True)
        fields["kelvin_anom"] = kelvin
    mjo = filter_wave_latmean(fields, sim_name, WAVE_FILTER_FIELD, "MJO_like", MJO_LIKE_FILTER)
    if mjo is not None:
        fields["mjo_like_anom"] = mjo
    return specs


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


def composite_lag_lon(da, events, lag_days, lag_steps, rel_lon_grid=REL_LON_GRID):
    sums = np.zeros((len(lag_steps), len(rel_lon_grid)), dtype=float)
    sums2 = np.zeros_like(sums)
    counts = np.zeros_like(sums)
    no_longitude = "longitude" not in da.dims

    for event in events.itertuples(index=False):
        for ilag, step in enumerate(lag_steps):
            it = int(event.onset_time_index) + int(step)
            if it < 0 or it >= da.sizes.get("Time", 0):
                continue
            x = da.isel(Time=it)
            x = lat_window_mean(x, float(event.centroid_lat))
            if "longitude" in x.dims:
                vals = lon_series_relative_to_event(x.load(), float(event.center_lon), rel_lon_grid)
            else:
                value = float(x.load().values)
                vals = np.full(len(rel_lon_grid), value, dtype=float)
            ok = np.isfinite(vals)
            sums[ilag, ok] += vals[ok]
            sums2[ilag, ok] += vals[ok] ** 2
            counts[ilag, ok] += 1.0

    mean = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=mean, where=counts > 0)
    var = np.full_like(sums, np.nan)
    valid = counts > 1
    var[valid] = (sums2[valid] - sums[valid] ** 2 / counts[valid]) / (counts[valid] - 1)
    stderr = np.full_like(sums, np.nan)
    stderr[valid] = np.sqrt(var[valid] / counts[valid])
    out = xr.Dataset(
        data_vars=dict(
            mean=(("lag_days", "relative_longitude"), mean),
            count=(("lag_days", "relative_longitude"), counts),
            stderr=(("lag_days", "relative_longitude"), stderr),
        ),
        coords=dict(lag_days=lag_days, relative_longitude=rel_lon_grid),
        attrs=dict(no_longitude=bool(no_longitude)),
    )
    return out


def scalar_at_event_lag(da, event, it, lon_window=FINGERPRINT_LON_WINDOW_DEG):
    x = da.isel(Time=it)
    x = lat_window_mean(x, float(event.centroid_lat))
    if "longitude" not in x.dims:
        return float(x.load().values)
    rel_lon = ((x.longitude.values.astype(float) - float(event.center_lon) + 180.0) % 360.0) - 180.0
    mask = np.abs(rel_lon) <= lon_window
    if not np.any(mask):
        return np.nan
    return float(x.isel(longitude=mask).mean(dim="longitude", skipna=True).load().values)


def composite_lag_scalar(da, events, lag_days, lag_steps):
    sums = np.zeros(len(lag_steps), dtype=float)
    sums2 = np.zeros_like(sums)
    counts = np.zeros_like(sums)
    for event in events.itertuples(index=False):
        for ilag, step in enumerate(lag_steps):
            it = int(event.onset_time_index) + int(step)
            if it < 0 or it >= da.sizes.get("Time", 0):
                continue
            value = scalar_at_event_lag(da, event, it)
            if not np.isfinite(value):
                continue
            sums[ilag] += value
            sums2[ilag] += value ** 2
            counts[ilag] += 1.0
    mean = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=mean, where=counts > 0)
    var = np.full_like(sums, np.nan)
    valid = counts > 1
    var[valid] = (sums2[valid] - sums[valid] ** 2 / counts[valid]) / (counts[valid] - 1)
    stderr = np.full_like(sums, np.nan)
    stderr[valid] = np.sqrt(var[valid] / counts[valid])
    return xr.Dataset(
        data_vars=dict(mean=("lag_days", mean), count=("lag_days", counts), stderr=("lag_days", stderr)),
        coords=dict(lag_days=lag_days),
    )


def composite_lag_products(da, events, lag_days, lag_steps, rel_lon_grid=REL_LON_GRID):
    hov_sums = np.zeros((len(lag_steps), len(rel_lon_grid)), dtype=float)
    hov_sums2 = np.zeros_like(hov_sums)
    hov_counts = np.zeros_like(hov_sums)
    scalar_sums = np.zeros(len(lag_steps), dtype=float)
    scalar_sums2 = np.zeros_like(scalar_sums)
    scalar_counts = np.zeros_like(scalar_sums)
    no_longitude = "longitude" not in da.dims

    for event in events.itertuples(index=False):
        center_lon = float(event.center_lon)
        for ilag, step in enumerate(lag_steps):
            it = int(event.onset_time_index) + int(step)
            if it < 0 or it >= da.sizes.get("Time", 0):
                continue

            x = da.isel(Time=it)
            x = lat_window_mean(x, float(event.centroid_lat)).load()
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
    hov_var[hov_valid] = (hov_sums2[hov_valid] - hov_sums[hov_valid] ** 2 / hov_counts[hov_valid]) / (hov_counts[hov_valid] - 1)
    hov_stderr = np.full_like(hov_sums, np.nan)
    hov_stderr[hov_valid] = np.sqrt(hov_var[hov_valid] / hov_counts[hov_valid])
    hov = xr.Dataset(
        data_vars=dict(
            mean=(("lag_days", "relative_longitude"), hov_mean),
            count=(("lag_days", "relative_longitude"), hov_counts),
            stderr=(("lag_days", "relative_longitude"), hov_stderr),
        ),
        coords=dict(lag_days=lag_days, relative_longitude=rel_lon_grid),
        attrs=dict(no_longitude=bool(no_longitude)),
    )

    scalar_mean = np.full_like(scalar_sums, np.nan)
    np.divide(scalar_sums, scalar_counts, out=scalar_mean, where=scalar_counts > 0)
    scalar_var = np.full_like(scalar_sums, np.nan)
    scalar_valid = scalar_counts > 1
    scalar_var[scalar_valid] = (
        scalar_sums2[scalar_valid] - scalar_sums[scalar_valid] ** 2 / scalar_counts[scalar_valid]
    ) / (scalar_counts[scalar_valid] - 1)
    scalar_stderr = np.full_like(scalar_sums, np.nan)
    scalar_stderr[scalar_valid] = np.sqrt(scalar_var[scalar_valid] / scalar_counts[scalar_valid])
    scalar = xr.Dataset(
        data_vars=dict(
            mean=("lag_days", scalar_mean),
            count=("lag_days", scalar_counts),
            stderr=("lag_days", scalar_stderr),
        ),
        coords=dict(lag_days=lag_days),
    )
    return hov, scalar


def event_centred_patch(da, event, it, rel_lat_grid=SPATIAL_REL_LAT_GRID, rel_lon_grid=SPATIAL_REL_LON_GRID):
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
    target_lat = float(event.centroid_lat) + rel_lat_grid
    target_lon = (float(event.center_lon) % 360.0) + rel_lon_grid
    patch = x_ext.interp(latitude=target_lat, longitude=target_lon)
    patch = patch.assign_coords(latitude=rel_lat_grid, longitude=rel_lon_grid)
    patch = patch.rename(latitude="relative_latitude", longitude="relative_longitude")
    return patch.load()


def composite_spatial(da, events, selected_lags_days, step_days):
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
            patch = event_centred_patch(da, event, it)
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
    save_figure(1, fig, f"lifecycle_hovmoller_composites_{stem_suffix}")
    plt.close(fig)


def normalized_curve(y):
    y = np.asarray(y, dtype=float)
    scale = np.nanmax(np.abs(y))
    if not np.isfinite(scale) or scale == 0:
        return y
    return y / scale


def milestone_lag(ds, kind="max", before_onset=False):
    y = np.asarray(ds.mean.values, dtype=float)
    x = np.asarray(ds.lag_days.values, dtype=float)
    mask = np.isfinite(y)
    if before_onset:
        mask &= x <= 0.0
    if not np.any(mask):
        return np.nan, np.nan
    idxs = np.where(mask)[0]
    if kind == "min":
        idx = idxs[int(np.nanargmin(y[mask]))]
    else:
        idx = idxs[int(np.nanargmax(y[mask]))]
    return float(x[idx]), float(y[idx])


def plot_fingerprint_summary(composites, specs, var_names):
    ncols = 3
    nrows = int(np.ceil(len(var_names) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(11.5, 2.7 * nrows), constrained_layout=True, sharex=True)
    ax = np.asarray(ax).reshape(nrows, ncols)
    milestone_rows = []
    for k, var in enumerate(var_names):
        a = ax.flat[k]
        spec = specs[var]
        for i, sim in enumerate(sim_names):
            if var not in composites[sim]:
                continue
            ds = composites[sim][var]["scalar"]
            y = normalized_curve(ds.mean.values)
            a.plot(ds.lag_days, y, color=cols[i], label=sim, lw=1.6)
            if var == "pw_anom":
                lag, val = milestone_lag(ds, before_onset=True)
                marker = "^"
                name = "moisture preconditioning peak"
            elif var == "precip":
                lag, val = milestone_lag(ds)
                marker = "o"
                name = "maximum precipitation"
            elif var == "cloud_top_height":
                lag, val = milestone_lag(ds)
                marker = "s"
                name = "maximum cloud depth"
            elif var == "column_connected":
                lag, val = milestone_lag(ds)
                marker = "D"
                name = "maximum high-cloud coverage"
            else:
                lag, val = milestone_lag(ds)
                marker = "."
                name = f"{spec['label']} extremum"
            if np.isfinite(lag):
                a.scatter([lag], [normalized_curve([val])[0] if False else np.interp(lag, ds.lag_days, y)], color=cols[i], marker=marker, s=25, zorder=5)
                milestone_rows.append(dict(simulation=sim, variable=var, milestone=name, lag_days=lag, value=val))
        a.axvline(0.0, color="0.25", lw=0.8)
        a.axhline(0.0, color="0.75", lw=0.5)
        a.set_title(spec["label"], fontsize=10)
        if k % ncols == 0:
            a.set_ylabel("normalised")
        else:
            a.tick_params(labelleft=False)
        if a.get_subplotspec().is_last_row():
            a.set_xlabel("Lag (days)")
    for a in ax.flat[len(var_names):]:
        a.set_axis_off()
    handles, labels = ax.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(sim_names), frameon=False)
    save_figure(2, fig, "mechanistic_fingerprint_summary")
    plt.close(fig)
    return pd.DataFrame(milestone_rows)


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
    save_figure(3, fig, "difference_lifecycle_fingerprints")
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
        fig.suptitle(specs[var]["label"], fontsize=12)
        save_figure(4, fig, f"event_centred_spatial_composites_{var}")
        plt.close(fig)


def compute_kelvin_phase_stats(fields_by_run, events_by_run):
    rows = []
    phase_series = {}
    for sim, fields in zip(sim_names, fields_by_run):
        if "kelvin_anom" not in fields or sim not in events_by_run:
            continue
        wav = fields["kelvin_anom"].transpose("Time", "longitude").load()
        analytic = hilbert(np.asarray(wav.values, dtype=float), axis=0)
        phase = np.angle(analytic)
        amp = np.abs(analytic)
        lon_vals = np.asarray(wav.longitude.values, dtype=float)
        sim_phases = []
        sim_amps = []
        for event in events_by_run[sim].itertuples(index=False):
            it = int(event.onset_time_index)
            if it < 0 or it >= phase.shape[0]:
                continue
            rel = np.abs(((lon_vals - event.center_lon + 180.0) % 360.0) - 180.0)
            j = int(np.nanargmin(rel))
            sim_phases.append(float(phase[it, j]))
            sim_amps.append(float(amp[it, j]))
        if not sim_phases:
            continue
        sim_phases = np.asarray(sim_phases)
        sim_amps = np.asarray(sim_amps)
        vector = np.nanmean(np.exp(1j * sim_phases))
        rows.append(
            dict(
                simulation=sim,
                event_count=len(sim_phases),
                preferred_phase_rad=float(np.angle(vector)),
                vector_strength=float(np.abs(vector)),
                mean_kelvin_amplitude=float(np.nanmean(sim_amps)),
            )
        )
        phase_series[sim] = sim_phases
    return pd.DataFrame(rows), phase_series


def plot_phase_locking(phase_stats, phase_series):
    if not phase_series:
        print("Skipping Figure 5: no Kelvin-filtered phase series available.")
        return
    bins = np.linspace(-np.pi, np.pi, 17)
    fig = plt.figure(figsize=(10.5, 4.2), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2, projection="polar")
    for i, sim in enumerate(sim_names):
        if sim not in phase_series:
            continue
        phases = phase_series[sim]
        hist, edges = np.histogram(phases, bins=bins, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax0.step(centres, hist, where="mid", color=cols[i], label=sim)
        row = phase_stats[phase_stats.simulation == sim]
        if len(row):
            theta = float(row.preferred_phase_rad.iloc[0])
            r = float(row.vector_strength.iloc[0])
            ax1.plot([theta, theta], [0, r], color=cols[i], lw=2.0, label=sim)
            ax1.scatter([theta], [r], color=cols[i], s=35)
    ax0.set_xlabel("Kelvin phase at onset (rad)")
    ax0.set_ylabel("Probability density")
    ax0.legend(frameon=False)
    ax1.set_title("Preferred phase vector")
    ax1.set_ylim(0, 1)
    save_figure(5, fig, "wave_convection_phase_locking")
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
        print("Skipping Figure 6: no track lifecycle properties available.")
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
    save_figure(6, fig, "track_property_lifecycle_statistics")
    plt.close(fig)


def plot_synthesis(composites, specs, milestone_table, phase_stats):
    synth_vars = [v for v in ("pw_anom", "precip", "column_connected", "N2_TTL_anom", "dudz_TTL_anom", "kelvin_anom") if v in specs]
    if not synth_vars:
        return
    fig, ax = plt.subplots(len(synth_vars), 1, figsize=(8.8, 1.75 * len(synth_vars)), constrained_layout=True, sharex=True)
    ax = np.atleast_1d(ax)
    for r, var in enumerate(synth_vars):
        a = ax[r]
        for i, sim in enumerate(sim_names):
            if var not in composites[sim]:
                continue
            ds = composites[sim][var]["scalar"]
            y = normalized_curve(ds.mean.values)
            a.plot(ds.lag_days, y, color=cols[i], label=sim if r == 0 else None, lw=1.7)
        a.axvline(0.0, color="0.2", lw=0.8)
        a.axhline(0.0, color="0.75", lw=0.5)
        a.set_ylabel(specs[var]["label"])
    ax[-1].set_xlabel("Lag from tracked onset (days)")
    ax[0].legend(frameon=False, ncol=len(sim_names), loc="upper right")
    if len(milestone_table):
        text_rows = []
        for sim in sim_names[1:]:
            for variable, label in (("precip", "Pmax"), ("cloud_top_height", "CTHmax"), ("pw_anom", "PWpre")):
                c = milestone_table[(milestone_table.simulation == "control") & (milestone_table.variable == variable)]
                s = milestone_table[(milestone_table.simulation == sim) & (milestone_table.variable == variable)]
                if len(c) and len(s):
                    text_rows.append(f"{sim} {label}: {s.lag_days.iloc[0] - c.lag_days.iloc[0]:+.1f} d")
        if text_rows:
            ax[0].text(0.01, 0.97, "\n".join(text_rows[:6]), transform=ax[0].transAxes, va="top", ha="left", fontsize=8, bbox=dict(facecolor="w", alpha=0.75, edgecolor="0.8"))
    if len(phase_stats):
        phase_text = []
        for row in phase_stats.to_dict("records"):
            phase_text.append(f"{row['simulation']}: R={row['vector_strength']:.2f}")
        ax[-1].text(0.99, 0.05, "\n".join(phase_text), transform=ax[-1].transAxes, va="bottom", ha="right", fontsize=8, bbox=dict(facecolor="w", alpha=0.75, edgecolor="0.8"))
    save_figure(7, fig, "mechanistic_synthesis_lifecycle")
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
            add_cldfrac_file=False,
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
            add_cldfrac_file=False,
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
    for key in ("cloud_top_height", "high_cloud_fraction", "high_cloud_mask", "column_connected_mask", "active_deep_convection_high_cloud_mask"):
        if key in classif:
            fields[key] = classif[key]

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

    source_name = "active_deep_convection_high_cloud_mask" if EVENT_SET == "deep" else "precip_ge_1mmh"
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
    specs_by_run = {}
    for sim_name, fields in zip(sim_names, fields_by_run):
        specs_by_run[sim_name] = prepare_composite_fields(fields, sim_name)
    spec_keys = []
    for key in ("precip", "pw_anom", "column_connected", "cloud_top_height", "N2_TTL_anom", "u_TTL_anom", "dudz_TTL_anom", "kelvin_anom"):
        if any(key in specs_by_run[sim] for sim in sim_names):
            spec_keys.append(key)
    specs = {}
    for key in spec_keys:
        for sim in sim_names:
            if key in specs_by_run[sim]:
                specs[key] = specs_by_run[sim][key]
                break

    include_spatial = should_run_figure(4)
    cached_composites = load_cached_composites(source_name, spec_keys, include_spatial)
    if cached_composites is None:
        composites = {}
        spatial_products = {}
        for sim_name, fields in zip(sim_names, fields_by_run):
            events = events_by_run.get(sim_name, pd.DataFrame())
            composites[sim_name] = {"event_count": len(events)}
            spatial_products[sim_name] = {}
            if len(events) == 0:
                continue
            for var in spec_keys:
                if var not in specs_by_run[sim_name]:
                    continue
                da = specs_by_run[sim_name][var]["da"]
                progress(f"{sim_name}: compositing {var}")
                progress(f"{sim_name}: {var} lag-longitude and scalar composites")
                hov, scalar = composite_lag_products(da, events, lag_days, lag_steps)
                composites[sim_name][var] = {
                    "hov": hov,
                    "scalar": scalar,
                }
                if (
                    include_spatial
                    and {"latitude", "longitude"}.issubset(da.dims)
                    and var in ("precip", "pw_anom", "column_connected", "cloud_top_height", "N2_TTL_anom")
                ):
                    progress(f"{sim_name}: {var} spatial composites")
                    spatial_products[sim_name][var] = composite_spatial(da, events, SPATIAL_LAGS_DAYS, step_days)
        write_composite_cache(source_name, spec_keys, include_spatial, composites, spatial_products)
    else:
        composites, spatial_products = cached_composites

    if should_run_figure(1):
        first_group = [v for v in ("precip", "pw_anom", "column_connected", "cloud_top_height") if v in specs]
        second_group = [v for v in ("N2_TTL_anom", "u_TTL_anom", "dudz_TTL_anom", "kelvin_anom") if v in specs]
        if first_group:
            plot_hovmoller_group(composites, specs, first_group, "core")
        if second_group:
            plot_hovmoller_group(composites, specs, second_group, "ttl_wave")

    milestone_table = pd.DataFrame()
    if should_run_figure(2):
        milestone_table = plot_fingerprint_summary(composites, specs, spec_keys)
        write_csv(milestone_table, "convective_lifecycle_milestones.csv")

    if should_run_figure(3):
        diff_vars = [v for v in ("pw_anom", "precip", "column_connected", "cloud_top_height", "N2_TTL_anom", "dudz_TTL_anom", "kelvin_anom") if v in specs]
        plot_difference_fingerprints(composites, specs, diff_vars)

    if should_run_figure(4):
        spatial_vars = [v for v in ("precip", "pw_anom", "column_connected", "cloud_top_height", "N2_TTL_anom") if v in specs]
        plot_spatial_composites(spatial_products, specs, spatial_vars)

    phase_stats = pd.DataFrame()
    phase_series = {}
    if should_run_figure(5) or should_run_figure(7):
        cached_phase = load_cached_phase_products(source_name)
        if cached_phase is None:
            phase_stats, phase_series = compute_kelvin_phase_stats(fields_by_run, events_by_run)
            write_phase_cache(source_name, phase_stats, phase_series)
        else:
            phase_stats, phase_series = cached_phase
        write_csv(phase_stats, "convective_lifecycle_kelvin_phase_stats.csv")
    if should_run_figure(5):
        plot_phase_locking(phase_stats, phase_series)

    if should_run_figure(6):
        cached_lifecycle = load_cached_track_property_lifecycle(source_name, step_days)
        if cached_lifecycle is None:
            lifecycle, track_summary = build_track_property_lifecycle(track_time_by_run, events_by_run, step_days)
            write_track_property_lifecycle_cache(source_name, step_days, lifecycle, track_summary)
        else:
            lifecycle, track_summary = cached_lifecycle
        write_csv(lifecycle, "convective_lifecycle_track_property_samples.csv")
        write_csv(track_summary, "convective_lifecycle_track_property_summary.csv")
        plot_track_property_lifecycle(track_summary, tracks_all)

    if should_run_figure(7):
        if len(milestone_table) == 0:
            milestone_table = plot_fingerprint_summary(composites, specs, spec_keys)
            write_csv(milestone_table, "convective_lifecycle_milestones.csv")
        plot_synthesis(composites, specs, milestone_table, phase_stats)

    print("\nFinished convective_lifecycle_composite_diagnostics.py")


if __name__ == "__main__":
    main()
