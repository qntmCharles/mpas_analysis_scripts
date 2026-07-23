#!/usr/bin/env python
"""Event-equal cloud-type evolution through two convective-system populations.

This is a deliberately narrow companion to the organisation and pathway
diagnostics. It reads their retained PyFLEXTRKR object table and reports how
cold-shield area, precipitation, convective core, stratiform cloud, and
non-precipitating high cloud evolve from genesis to lysis. A second analysis
samples the existing Kelvin, easterly-wave, and ER filtered-field caches along
the moving track centres; it does not define another object set or rerun filters.
Unfiltered precipitation, precipitable water, and OLR are sampled in the same
fixed-radius region following each track. They are shown both through normalized
lifecycle phase and at actual six-hourly lags from the local precipitation peak.
"""

import argparse
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from diagnostics.diagnostic_helpers import (
    LATLON_PATTERN, FigureSaver, TRACK_POPULATION_LABELS, TRACK_POPULATIONS,
    convective_organisation_cache_dir, diagnostic_output_dir,
    load_latlon_dataset, precipitation_rate_from_dataset, simulation_parent_dir,
    standardize_latlon_dataset, subset_latitude, table_boolean_mask,
)
from diagnostics.waves.cached_wave_products import load_existing_filtered_wave
from diagnostics.waves.equatorial_wave_config import (
    CORE_WAVE_NAMES, WAVE_BOXES, WAVE_LATITUDE_PROJECTION,
)


save = True
save_loc = diagnostic_output_dir(
    "/home/cwp29/tmp/mpas/crf_on/final/convective_lifecycle/"
)
organisation_output_dir = convective_organisation_cache_dir()
figures_to_run = [1, 2, 3, 4]
sim_names = ["control", "nudged u", "nudged T"]
cols = ["k", "b", "r"]
data_locs = ["control_new", "u_nudge", "T_nudge"]
PHASE_GRID = np.linspace(0.0, 1.0, 11)
WAVE_NAMES = tuple(CORE_WAVE_NAMES)
WAVE_LAT_BOUNDS = (-10.0, 10.0)
WAVE_LONGITUDE_HALF_WIDTH_DEG = 10.0
TRACK_ENVIRONMENT_RADIUS_KM = 500.0
EARTH_RADIUS_KM = 6371.0
PEAK_LAG_WINDOW_HOURS = 48.0
UNFILTERED_SAMPLE_VERSION = "fixed_radius_500km_v1"
UNFILTERED_SOURCE = "unfiltered MPAS lat-lon output"
RAW_LATLON_VARS = (
    "rainc", "rainnc", "i_rainc", "i_rainnc", "precipw", "olrtoa",
)
RAW_FIELD_SPECS = (
    ("Precip", "Precipitation", "mm day$^{-1}$"),
    ("PW", "Precipitable water", "kg m$^{-2}$"),
    ("OLR", "Outgoing longwave radiation", "W m$^{-2}$"),
)

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"

WAVE_FIELD_SPECS = (
    ("Precip", "Precipitation", "mm day$^{-1}$"),
    ("PW", "Precipitable water", "kg m$^{-2}$"),
    ("OLR", "Outgoing longwave radiation", "W m$^{-2}$"),
    ("u850", "850-hPa zonal wind", "m s$^{-1}$"),
    ("u250", "250-hPa zonal wind", "m s$^{-1}$"),
    ("T100", "100-hPa temperature", "K"),
)

CORE_AREA = "area_convective_precipitating_high_cloud_mask_km2"
STRATIFORM_AREA = "area_stratiform_precipitating_high_cloud_mask_km2"
NONPRECIP_AREA = "area_nonprecipitating_high_cloud_mask_km2"

PROPERTY_SPECS = (
    ("cold_shield_area_km2", "Cold-shield area", "km$^2$"),
    ("mean_precip_mm_day", "Mean precipitation", "mm day$^{-1}$"),
    ("convective_core_fraction", "Convective-core fraction", "% of shield"),
    ("stratiform_fraction", "Stratiform fraction", "% of shield"),
    ("nonprecipitating_fraction", "Non-precipitating fraction", "% of shield"),
    ("anvil_to_core_area_ratio", "Stratiform + non-precip. / core", "area ratio"),
)


def parse_command_line(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot event-equal lifecycles of persistent DCS and Feng MCS tracks."
    )
    parser.add_argument(
        "--organisation-dir",
        default=organisation_output_dir,
        help="Data-side cache directory containing organisation_tracked_objects.csv.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use a small deterministic subset of retained tracks.",
    )
    parser.add_argument(
        "--simulation",
        choices=("control", "nudged-u", "nudged-T", "all"),
        default="all",
        help="Simulation retained in test mode (default: all).",
    )
    parser.add_argument(
        "--test-tracks",
        type=int,
        default=20,
        metavar="N",
        help="Maximum tracks per population/simulation in test mode (default: 20).",
    )
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        choices=(1, 2, 3, 4),
        default=None,
        help=(
            "Figures: 1 cloud-type lifecycle; 2 cached track-following wave "
            "fields; 3 unfiltered track-following lifecycle; 4 unfiltered "
            "timing relative to local precipitation maximum."
        ),
    )
    parser.add_argument(
        "--rebuild-unfiltered-cache",
        action="store_true",
        help="Resample raw precipitation, PW, and OLR even if the CSV cache is current.",
    )
    return parser.parse_args(argv)


def load_track_objects(directory):
    path = os.path.join(directory, "organisation_tracked_objects.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run convective_organisation_diagnostics.py first."
        )
    objects = pd.read_csv(path)
    required = {
        "simulation", "track_id", "Time", "time_index", "area_km2",
        "centroid_lat", "centroid_lon", "precip_area_rate", CORE_AREA,
        STRATIFORM_AREA, NONPRECIP_AREA,
        *TRACK_POPULATIONS,
    }
    missing = sorted(required - set(objects.columns))
    if missing:
        raise RuntimeError(
            "Track object table predates the dual-population lifecycle schema; "
            f"missing {missing}. Rebuild the organisation product."
        )
    return objects


def expand_track_populations(objects):
    """Create analysis rows for each population without duplicating cache data."""
    frames = []
    for population in TRACK_POPULATIONS:
        selected = objects[table_boolean_mask(objects, population)].copy()
        selected["track_population"] = population
        selected["track_population_label"] = TRACK_POPULATION_LABELS[population]
        frames.append(selected)
    return (
        pd.concat(frames, ignore_index=True)
        if frames else objects.iloc[0:0].copy()
    )


def select_test_tracks(objects, simulation="all", maximum=20):
    if simulation != "all":
        label = {"control": "control", "nudged-u": "nudged u", "nudged-T": "nudged T"}[simulation]
        objects = objects[objects.simulation == label]
    selected = []
    group_keys = ["simulation"]
    if "track_population" in objects:
        group_keys.append("track_population")
    for _key, group in objects.groupby(group_keys):
        track_ids = np.sort(group.track_id.unique())[:maximum]
        selected.append(group[group.track_id.isin(track_ids)])
    return pd.concat(selected, ignore_index=True) if selected else objects.iloc[0:0]


def aggregate_track_times(objects):
    if "track_population" not in objects:
        objects = objects.copy()
        objects["track_population"] = "feng_mcs"
        objects["track_population_label"] = TRACK_POPULATION_LABELS["feng_mcs"]
    rows = []
    keys = ["track_population", "track_population_label", "simulation", "track_id", "time_index"]
    for (population, population_label, simulation, track_id, time_index), group in objects.groupby(keys):
        area = float(group.area_km2.sum())
        precip_area_rate = float(group.precip_area_rate.sum())
        core_area = float(group[CORE_AREA].sum())
        stratiform_area = float(group[STRATIFORM_AREA].sum())
        nonprecip_area = float(group[NONPRECIP_AREA].sum())
        lon_weights = group.area_km2.to_numpy(dtype=float)
        centroid_lat = np.nan
        if "centroid_lat" in group:
            centroid_lats = group.centroid_lat.to_numpy(dtype=float)
            valid_lat = np.isfinite(centroid_lats) & np.isfinite(lon_weights) & (lon_weights > 0.0)
            if np.any(valid_lat):
                centroid_lat = float(np.average(centroid_lats[valid_lat], weights=lon_weights[valid_lat]))
        lon_radians = np.deg2rad(group.centroid_lon.to_numpy(dtype=float))
        centroid_lon = np.rad2deg(
            np.arctan2(
                np.nansum(np.sin(lon_radians) * lon_weights),
                np.nansum(np.cos(lon_radians) * lon_weights),
            )
        ) % 360.0
        rows.append(
            dict(
                simulation=simulation,
                track_population=population,
                track_population_label=population_label,
                track_id=track_id,
                Time=(pd.to_datetime(group.Time.iloc[0]) if "Time" in group else pd.NaT),
                time_index=int(time_index),
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                cold_shield_area_km2=area,
                mean_precip_mm_day=(precip_area_rate / area if area > 0.0 else np.nan),
                convective_core_fraction=(core_area / area if area > 0.0 else np.nan),
                stratiform_fraction=(stratiform_area / area if area > 0.0 else np.nan),
                nonprecipitating_fraction=(nonprecip_area / area if area > 0.0 else np.nan),
                anvil_to_core_area_ratio=(
                    (stratiform_area + nonprecip_area) / core_area
                    if core_area > 0.0 else np.nan
                ),
            )
        )
    return pd.DataFrame(rows)


def interpolate_event_lifecycles(track_time, phase_grid=PHASE_GRID):
    rows = []
    property_keys = [key for key, _label, _units in PROPERTY_SPECS]
    group_keys = ["track_population", "track_population_label", "simulation", "track_id"]
    for (population, population_label, simulation, track_id), group in track_time.groupby(group_keys):
        group = group.sort_values("time_index")
        if len(group) < 2:
            continue
        time = group.time_index.to_numpy(dtype=float)
        span = time[-1] - time[0]
        if span <= 0.0:
            continue
        phase = (time - time[0]) / span
        for key in property_keys:
            values = group[key].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if finite.sum() < 2:
                interpolated = np.full(phase_grid.shape, np.nan)
            else:
                interpolated = np.interp(phase_grid, phase[finite], values[finite])
            if key.endswith("_fraction"):
                interpolated *= 100.0
            for lifecycle_phase, value in zip(phase_grid, interpolated):
                rows.append(
                    dict(
                        simulation=simulation,
                        track_population=population,
                        track_population_label=population_label,
                        track_id=track_id,
                        lifecycle_phase=float(lifecycle_phase),
                        property=key,
                        value=float(value),
                    )
                )
    return pd.DataFrame(rows)


def lifecycle_summary(samples):
    if len(samples) == 0:
        return pd.DataFrame()
    return (
        samples.groupby(
            ["track_population", "track_population_label", "simulation", "lifecycle_phase", "property"], as_index=False
        )
        .value.agg(
            median="median",
            p25=lambda values: values.quantile(0.25),
            p75=lambda values: values.quantile(0.75),
            event_count="count",
        )
    )


def plot_lifecycle(summary, output_directory):
    if len(summary) == 0:
        print("No accepted multi-scene MCS tracks are available for the lifecycle figure.")
        return
    figure_saver = FigureSaver(
        output_directory, enabled=save, figures_to_run=figures_to_run
    )
    for population in TRACK_POPULATIONS:
        population_summary = summary[summary.track_population == population]
        if len(population_summary) == 0:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.0), constrained_layout=True)
        for axis, (key, label, units) in zip(axes.ravel(), PROPERTY_SPECS):
            subset = population_summary[population_summary.property == key]
            for simulation, color in zip(sim_names, cols):
                values = subset[subset.simulation == simulation].sort_values(
                    "lifecycle_phase"
                )
                if len(values) == 0:
                    continue
                axis.plot(values.lifecycle_phase, values["median"], color=color, lw=1.7, label=simulation)
                axis.fill_between(values.lifecycle_phase, values.p25, values.p75, color=color, alpha=0.16, lw=0)
            axis.set(title=label, xlabel="Normalized lifecycle (0 first, 1 last sampled scene)", ylabel=units)
            axis.grid(alpha=0.2)
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(
            f"Event-equal lifecycle: {TRACK_POPULATION_LABELS[population]}\n"
            "Lines: median across tracks; shading: interquartile range",
            fontsize=11,
        )
        figure_saver.save(fig, f"{population}_normalized_cloud_type_lifecycle", number=1)
        plt.close(fig)


def great_circle_distance_km(latitude, longitude, centre_lat, centre_lon):
    """Distance from every regular-grid point to a track centre."""
    lat2d, lon2d = np.meshgrid(
        np.asarray(latitude, dtype=float),
        np.asarray(longitude, dtype=float),
        indexing="ij",
    )
    lat1 = np.deg2rad(float(centre_lat))
    lat2 = np.deg2rad(lat2d)
    dlat = lat2 - lat1
    dlon = np.deg2rad((lon2d - float(centre_lon) + 180.0) % 360.0 - 180.0)
    haversine = (
        np.sin(0.5 * dlat) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * dlon) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(
        np.sqrt(np.clip(haversine, 0.0, 1.0))
    )


def track_centered_field_means(
    scene,
    centre_lat,
    centre_lon,
    radius_km=TRACK_ENVIRONMENT_RADIUS_KM,
):
    """Cosine-area means in one fixed-radius aperture for all raw fields."""
    distance = great_circle_distance_km(
        scene.latitude.values,
        scene.longitude.values,
        centre_lat,
        centre_lon,
    )
    aperture = distance <= float(radius_km)
    weights = np.cos(np.deg2rad(scene.latitude.values))[:, None]
    means = {}
    for field_name, _label, _units in RAW_FIELD_SPECS:
        values = np.asarray(
            scene[field_name].transpose("latitude", "longitude").values,
            dtype=float,
        )
        valid = aperture & np.isfinite(values) & (np.abs(values) < 1.0e30)
        field_weights = np.broadcast_to(weights, values.shape)[valid]
        means[field_name] = (
            float(np.average(values[valid], weights=field_weights))
            if np.any(valid) and np.nansum(field_weights) > 0.0
            else np.nan
        )
    return means


def open_unfiltered_fields(simulation):
    """Open only the six source variables needed for local raw-field sampling."""
    simulation_index = sim_names.index(simulation)
    source_root = simulation_parent_dir(root_dir)
    dataset = load_latlon_dataset(
        os.path.join(source_root, data_locs[simulation_index]),
        latlon_pattern=LATLON_PATTERN,
        chunks={"Time": 8, "latitude": 60, "longitude": 120},
    )
    dataset = standardize_latlon_dataset(dataset, RAW_LATLON_VARS)
    precip = precipitation_rate_from_dataset(dataset)
    fields = xr.Dataset(
        {
            "Precip": precip,
            "PW": dataset.precipw.sel(Time=precip.Time),
            "OLR": dataset.olrtoa.sel(Time=precip.Time),
        }
    )
    return dataset, fields


def add_track_relative_coordinates(samples):
    """Attach elapsed hours and normalized phase without changing event weight."""
    if len(samples) == 0:
        return samples.copy()
    frames = []
    group_keys = [
        "track_population", "track_population_label", "simulation", "track_id"
    ]
    for _key, group in samples.groupby(group_keys, sort=False):
        group = group.sort_values("Time").copy()
        times = pd.to_datetime(group.Time)
        elapsed_hours = (
            times.to_numpy() - times.iloc[0].to_datetime64()
        ) / np.timedelta64(1, "h")
        span_hours = float(elapsed_hours[-1]) if len(elapsed_hours) else 0.0
        group["elapsed_hours_from_first_scene"] = elapsed_hours.astype(float)
        group["lifecycle_phase"] = (
            elapsed_hours / span_hours if span_hours > 0.0 else np.nan
        )
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else samples.iloc[0:0].copy()


def sample_unfiltered_track_environment(track_time):
    """Sample unfiltered precipitation, PW, and OLR around moving track centres."""
    rows = []
    for simulation in sim_names:
        sim_tracks = track_time[track_time.simulation == simulation].copy()
        if len(sim_tracks) == 0:
            continue
        sim_tracks["Time"] = pd.to_datetime(sim_tracks.Time)
        if sim_tracks.Time.isna().any():
            raise RuntimeError(f"{simulation}: retained tracks contain invalid timestamps")
        if not np.all(
            np.isfinite(sim_tracks.centroid_lat.to_numpy(dtype=float))
            & np.isfinite(sim_tracks.centroid_lon.to_numpy(dtype=float))
        ):
            raise RuntimeError(f"{simulation}: retained tracks contain invalid centroids")

        print(
            f"{simulation}: opening raw precipitation, PW, and OLR for "
            f"{sim_tracks.track_id.nunique()} retained track IDs",
            flush=True,
        )
        dataset, fields = open_unfiltered_fields(simulation)
        try:
            requested_times = pd.DatetimeIndex(sim_tracks.Time.unique()).sort_values()
            available_times = pd.DatetimeIndex(fields.Time.values)
            missing = requested_times.difference(available_times)
            if len(missing):
                raise RuntimeError(
                    f"{simulation}: raw fields are missing {len(missing)} tracked times; "
                    f"first missing time is {missing[0]}"
                )

            angular_padding = np.rad2deg(TRACK_ENVIRONMENT_RADIUS_KM / EARTH_RADIUS_KM)
            latitude_bounds = (
                float(sim_tracks.centroid_lat.min()) - angular_padding,
                float(sim_tracks.centroid_lat.max()) + angular_padding,
            )
            fields = subset_latitude(fields, lat_bounds=latitude_bounds)
            scene_groups = list(sim_tracks.groupby("Time", sort=True))
            started = time.monotonic()
            report_every = max(1, len(scene_groups) // 20)
            for scene_number, (timestamp, group) in enumerate(scene_groups, start=1):
                scene = fields.sel(Time=np.datetime64(timestamp)).load()
                centre_cache = {}
                for record in group.itertuples(index=False):
                    centre_key = (
                        round(float(record.centroid_lat), 8),
                        round(float(record.centroid_lon), 8),
                    )
                    if centre_key not in centre_cache:
                        centre_cache[centre_key] = track_centered_field_means(
                            scene,
                            record.centroid_lat,
                            record.centroid_lon,
                        )
                    rows.append(
                        dict(
                            simulation=record.simulation,
                            track_population=record.track_population,
                            track_population_label=record.track_population_label,
                            track_id=record.track_id,
                            Time=pd.Timestamp(timestamp),
                            time_index=int(record.time_index),
                            centroid_lat=float(record.centroid_lat),
                            centroid_lon=float(record.centroid_lon),
                            **centre_cache[centre_key],
                            sampling_radius_km=float(TRACK_ENVIRONMENT_RADIUS_KM),
                            sampling_version=UNFILTERED_SAMPLE_VERSION,
                            source=UNFILTERED_SOURCE,
                        )
                    )
                if scene_number % report_every == 0 or scene_number == len(scene_groups):
                    elapsed = time.monotonic() - started
                    print(
                        f"{simulation}: sampled {scene_number}/{len(scene_groups)} "
                        f"tracked scenes ({100.0 * scene_number / len(scene_groups):.0f}%; "
                        f"{elapsed / 60.0:.1f} min)",
                        flush=True,
                    )
        finally:
            dataset.close()
    return add_track_relative_coordinates(pd.DataFrame(rows))


def _sample_key_set(frame):
    if len(frame) == 0:
        return set()
    times = pd.to_datetime(frame.Time).to_numpy(dtype="datetime64[ns]").astype("int64")
    return set(
        zip(
            frame.simulation.astype(str),
            frame.track_population.astype(str),
            frame.track_id.astype(str),
            frame.time_index.astype(int),
            times,
            np.round(frame.centroid_lat.to_numpy(dtype=float), 8),
            np.round(frame.centroid_lon.to_numpy(dtype=float), 8),
        )
    )


def cached_unfiltered_samples_are_current(samples, track_time):
    required = {
        "simulation", "track_population", "track_population_label", "track_id",
        "Time", "time_index", "centroid_lat", "centroid_lon", "Precip", "PW",
        "OLR", "sampling_radius_km", "sampling_version", "source",
        "elapsed_hours_from_first_scene", "lifecycle_phase",
    }
    if len(samples) == 0 or not required.issubset(samples.columns):
        return False
    if set(samples.sampling_version.dropna().astype(str)) != {UNFILTERED_SAMPLE_VERSION}:
        return False
    if not np.allclose(
        samples.sampling_radius_km.to_numpy(dtype=float),
        TRACK_ENVIRONMENT_RADIUS_KM,
    ):
        return False
    return _sample_key_set(samples) == _sample_key_set(track_time)


def load_or_build_unfiltered_samples(
    track_time,
    output_directory,
    rebuild=False,
):
    cache_path = os.path.join(
        output_directory,
        "convective_system_track_following_unfiltered_environment_samples.csv",
    )
    if not rebuild and os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, parse_dates=["Time"])
        except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
            print(f"Ignoring unreadable unfiltered lifecycle cache: {exc}", flush=True)
        else:
            if cached_unfiltered_samples_are_current(cached, track_time):
                print(f"Using current unfiltered lifecycle cache: {cache_path}", flush=True)
                return cached
            print("Unfiltered lifecycle cache is stale; resampling raw fields.", flush=True)
    samples = sample_unfiltered_track_environment(track_time)
    samples.to_csv(cache_path, index=False)
    print(f"Wrote unfiltered lifecycle cache: {cache_path}", flush=True)
    return samples


def interpolate_unfiltered_lifecycles(samples, phase_grid=PHASE_GRID):
    rows = []
    group_keys = [
        "track_population", "track_population_label", "simulation", "track_id"
    ]
    for (population, population_label, simulation, track_id), group in samples.groupby(group_keys):
        group = group.sort_values("lifecycle_phase")
        phase = group.lifecycle_phase.to_numpy(dtype=float)
        if np.isfinite(phase).sum() < 2:
            continue
        for field_name, _label, units in RAW_FIELD_SPECS:
            values = group[field_name].to_numpy(dtype=float)
            finite = np.isfinite(phase) & np.isfinite(values)
            if finite.sum() < 2:
                continue
            interpolated = np.interp(phase_grid, phase[finite], values[finite])
            for lifecycle_phase, value in zip(phase_grid, interpolated):
                rows.append(
                    dict(
                        simulation=simulation,
                        track_population=population,
                        track_population_label=population_label,
                        track_id=track_id,
                        field=field_name,
                        units=units,
                        lifecycle_phase=float(lifecycle_phase),
                        value=float(value),
                    )
                )
    return pd.DataFrame(rows)


def summarize_environment_samples(samples, coordinate):
    if len(samples) == 0:
        return pd.DataFrame()
    return (
        samples.groupby(
            [
                "track_population", "track_population_label", "simulation",
                "field", "units", coordinate,
            ],
            as_index=False,
        )
        .value.agg(
            median="median",
            p25=lambda values: values.quantile(0.25),
            p75=lambda values: values.quantile(0.75),
            event_count="count",
        )
    )


def _extremum_time(group, field, mode):
    values = group[field].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return pd.NaT
    finite_positions = np.flatnonzero(finite)
    relative_index = (
        int(np.argmax(values[finite])) if mode == "maximum"
        else int(np.argmin(values[finite]))
    )
    return pd.Timestamp(group.iloc[finite_positions[relative_index]].Time)


def peak_aligned_environment_samples(
    samples,
    lag_window_hours=PEAK_LAG_WINDOW_HOURS,
):
    """Return actual-hour samples and extrema offsets centred on precipitation."""
    aligned_rows = []
    offset_rows = []
    group_keys = [
        "track_population", "track_population_label", "simulation", "track_id"
    ]
    for (population, population_label, simulation, track_id), group in samples.groupby(group_keys):
        group = group.sort_values("Time").copy()
        group["Time"] = pd.to_datetime(group.Time)
        precip_peak = _extremum_time(group, "Precip", "maximum")
        if pd.isna(precip_peak):
            continue
        pw_peak = _extremum_time(group, "PW", "maximum")
        olr_minimum = _extremum_time(group, "OLR", "minimum")
        lag_hours = (
            group.Time.to_numpy() - precip_peak.to_datetime64()
        ) / np.timedelta64(1, "h")
        for field_name, _label, units in RAW_FIELD_SPECS:
            values = group[field_name].to_numpy(dtype=float)
            for lag_hour, value in zip(lag_hours, values):
                if np.isfinite(value) and abs(float(lag_hour)) <= lag_window_hours:
                    aligned_rows.append(
                        dict(
                            simulation=simulation,
                            track_population=population,
                            track_population_label=population_label,
                            track_id=track_id,
                            field=field_name,
                            units=units,
                            lag_hours_from_precipitation_maximum=float(lag_hour),
                            value=float(value),
                        )
                    )
        offset_rows.append(
            dict(
                simulation=simulation,
                track_population=population,
                track_population_label=population_label,
                track_id=track_id,
                precipitation_maximum_time=precip_peak,
                pw_maximum_minus_precipitation_maximum_hours=(
                    (pw_peak - precip_peak) / pd.Timedelta(hours=1)
                    if not pd.isna(pw_peak) else np.nan
                ),
                olr_minimum_minus_precipitation_maximum_hours=(
                    (olr_minimum - precip_peak) / pd.Timedelta(hours=1)
                    if not pd.isna(olr_minimum) else np.nan
                ),
                first_detection_minus_precipitation_maximum_hours=(
                    (group.Time.iloc[0] - precip_peak) / pd.Timedelta(hours=1)
                ),
                last_detection_minus_precipitation_maximum_hours=(
                    (group.Time.iloc[-1] - precip_peak) / pd.Timedelta(hours=1)
                ),
                timing_window="first to last detected scene",
                temporal_resolution_hours=float(
                    np.nanmedian(np.diff(group.Time) / np.timedelta64(1, "h"))
                    if len(group) > 1 else np.nan
                ),
            )
        )
    return pd.DataFrame(aligned_rows), pd.DataFrame(offset_rows)


TIMING_OFFSET_SPECS = (
    (
        "pw_maximum_minus_precipitation_maximum_hours",
        "PW maximum",
    ),
    (
        "olr_minimum_minus_precipitation_maximum_hours",
        "OLR minimum",
    ),
)


def timing_offset_summary(offsets):
    rows = []
    if len(offsets) == 0:
        return pd.DataFrame()
    group_keys = [
        "track_population", "track_population_label", "simulation"
    ]
    for (population, population_label, simulation), group in offsets.groupby(group_keys):
        for metric, label in TIMING_OFFSET_SPECS:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            rows.append(
                dict(
                    simulation=simulation,
                    track_population=population,
                    track_population_label=population_label,
                    metric=metric,
                    metric_label=label,
                    median_hours=float(np.median(values)),
                    p25_hours=float(np.percentile(values, 25)),
                    p75_hours=float(np.percentile(values, 75)),
                    event_count=int(values.size),
                    sign_convention="negative leads local precipitation maximum",
                )
            )
    return pd.DataFrame(rows)


def unfiltered_method_table():
    return pd.DataFrame(
        [
            dict(
                fields="precipitation rate; precipitable water; outgoing longwave radiation",
                source=UNFILTERED_SOURCE,
                spatial_sampling=(
                    f"cosine-area mean within {TRACK_ENVIRONMENT_RADIUS_KM:g} km "
                    "of moving cold-shield centroid"
                ),
                normalized_lifecycle="0 first detected scene; 1 last detected scene",
                timing_anchor="maximum local precipitation during detected lifecycle",
                timing_resolution="6 hours",
                timing_window="first to last detected scene",
                event_weighting="each track weighted equally",
                populations="persistent DCS and nested Feng-criteria MCS",
                olr_interpretation=(
                    "OLR-derived brightness temperature contributes to cold-shield "
                    "identification; OLR timing is not independent tracking validation"
                ),
                cache_policy=(
                    "reuse samples only when method, retained track times, and moving "
                    "centroids are unchanged"
                ),
            )
        ]
    )


def plot_unfiltered_lifecycle(summary, output_directory):
    if len(summary) == 0:
        print("No unfiltered track-following lifecycle samples are available.")
        return
    figure_saver = FigureSaver(
        output_directory, enabled=save, figures_to_run=figures_to_run
    )
    for population in TRACK_POPULATIONS:
        population_rows = summary[summary.track_population == population]
        if len(population_rows) == 0:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
        for axis, (field_name, label, units) in zip(axes, RAW_FIELD_SPECS):
            field_rows = population_rows[population_rows.field == field_name]
            for simulation, color in zip(sim_names, cols):
                values = field_rows[field_rows.simulation == simulation].sort_values(
                    "lifecycle_phase"
                )
                if len(values) == 0:
                    continue
                axis.plot(
                    values.lifecycle_phase,
                    values["median"],
                    color=color,
                    lw=1.7,
                    label=f"{simulation} (n={int(values.event_count.max())})",
                )
                axis.fill_between(
                    values.lifecycle_phase,
                    values.p25,
                    values.p75,
                    color=color,
                    alpha=0.15,
                    lw=0,
                )
            axis.set(
                title=label,
                xlabel="Normalized lifecycle (0 first, 1 last sampled scene)",
                ylabel=units,
            )
            axis.grid(alpha=0.2)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(
            f"Unfiltered local environment following {TRACK_POPULATION_LABELS[population].lower()}\n"
            f"Area mean within {TRACK_ENVIRONMENT_RADIUS_KM:g} km; median and interquartile range across tracks",
            fontsize=11,
        )
        figure_saver.save(
            fig,
            f"{population}_unfiltered_track_following_lifecycle",
            number=3,
        )
        plt.close(fig)


def plot_peak_aligned_timing(summary, offset_summary, output_directory):
    if len(summary) == 0:
        print("No precipitation-peak-aligned lifecycle samples are available.")
        return
    figure_saver = FigureSaver(
        output_directory, enabled=save, figures_to_run=figures_to_run
    )
    lag_key = "lag_hours_from_precipitation_maximum"
    for population in TRACK_POPULATIONS:
        population_rows = summary[summary.track_population == population]
        if len(population_rows) == 0:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
        for axis, (field_name, label, units) in zip(axes.ravel()[:3], RAW_FIELD_SPECS):
            field_rows = population_rows[population_rows.field == field_name]
            for simulation, color in zip(sim_names, cols):
                values = field_rows[field_rows.simulation == simulation].sort_values(lag_key)
                if len(values) == 0:
                    continue
                zero_lag = values[np.isclose(values[lag_key], 0.0)]
                event_count = (
                    int(zero_lag.event_count.iloc[0])
                    if len(zero_lag) else int(values.event_count.max())
                )
                axis.plot(
                    values[lag_key],
                    values["median"],
                    color=color,
                    lw=1.7,
                    label=f"{simulation} (n={event_count})",
                )
                axis.fill_between(values[lag_key], values.p25, values.p75, color=color, alpha=0.15, lw=0)
            axis.axvline(0.0, color="0.45", lw=0.8)
            axis.set(
                title=label,
                xlabel="Lag from local precipitation maximum (hours)",
                ylabel=units,
                xlim=(-PEAK_LAG_WINDOW_HOURS, PEAK_LAG_WINDOW_HOURS),
            )
            axis.grid(alpha=0.2)
        axes[0, 0].legend(frameon=False, fontsize=8)

        offset_axis = axes[1, 1]
        offset_rows = (
            offset_summary[offset_summary.track_population == population]
            if len(offset_summary) and "track_population" in offset_summary
            else pd.DataFrame()
        )
        metric_positions = {label: index for index, (_key, label) in enumerate(TIMING_OFFSET_SPECS)}
        simulation_offsets = np.linspace(-0.18, 0.18, len(sim_names))
        for simulation, color, y_offset in zip(sim_names, cols, simulation_offsets):
            values = (
                offset_rows[offset_rows.simulation == simulation]
                if "simulation" in offset_rows else pd.DataFrame()
            )
            first = True
            for record in values.itertuples(index=False):
                y = metric_positions[record.metric_label] + y_offset
                offset_axis.errorbar(
                    record.median_hours,
                    y,
                    xerr=[
                        [record.median_hours - record.p25_hours],
                        [record.p75_hours - record.median_hours],
                    ],
                    fmt="o",
                    color=color,
                    capsize=3,
                    ms=4,
                    label=simulation if first else None,
                )
                first = False
        offset_axis.axvline(0.0, color="0.45", lw=0.8)
        offset_axis.set(
            title="Per-track extrema during detected lifecycle",
            xlabel="Lag from local precipitation maximum (hours; negative leads)",
            yticks=list(metric_positions.values()),
            yticklabels=list(metric_positions.keys()),
            xlim=(-PEAK_LAG_WINDOW_HOURS, PEAK_LAG_WINDOW_HOURS),
        )
        offset_axis.grid(axis="x", alpha=0.2)
        if len(offset_rows):
            offset_axis.legend(frameon=False, fontsize=8)
        fig.suptitle(
            f"Timing of the unfiltered local environment for {TRACK_POPULATION_LABELS[population].lower()}\n"
            f"Six-hourly fields within {TRACK_ENVIRONMENT_RADIUS_KM:g} km; each track aligned on its precipitation maximum",
            fontsize=11,
        )
        fig.text(
            0.5,
            0.005,
            "Only first-to-last detected scenes are sampled. OLR-derived brightness temperature helps define the cold shield, so OLR timing describes shield evolution rather than independent validation.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.3",
        )
        figure_saver.save(
            fig,
            f"{population}_unfiltered_environment_precipitation_peak_timing",
            number=4,
        )
        plt.close(fig)


def track_following_wave_samples(track_time):
    """Sample existing filtered fields near moving tracks in both populations."""
    rows = []
    for simulation in sim_names:
        sim_tracks = track_time[track_time.simulation == simulation]
        if len(sim_tracks) == 0:
            continue
        for wave_name in WAVE_NAMES:
            for field_name, _label, units in WAVE_FIELD_SPECS:
                wave = load_existing_filtered_wave(
                    root_dir, simulation, field_name, wave_name
                )
                longitude = np.asarray(wave.longitude.values, dtype=float)
                for (population, population_label, track_id), group in sim_tracks.groupby(
                    ["track_population", "track_population_label", "track_id"]
                ):
                    group = group.sort_values("time_index")
                    if len(group) < 2:
                        continue
                    time_index = group.time_index.to_numpy(dtype=int)
                    valid_time = (
                        (time_index >= 0) & (time_index < wave.sizes["Time"])
                    )
                    if valid_time.sum() < 2:
                        continue
                    time_index = time_index[valid_time]
                    centres = group.centroid_lon.to_numpy(dtype=float)[valid_time]
                    values = []
                    for itime, centre in zip(time_index, centres):
                        distance = np.abs(
                            (longitude - centre + 180.0) % 360.0 - 180.0
                        )
                        in_window = distance <= WAVE_LONGITUDE_HALF_WIDTH_DEG
                        values.append(
                            float(
                                wave.isel(Time=int(itime), longitude=in_window)
                                .mean("longitude", skipna=True).values
                            )
                        )
                    values = np.asarray(values, dtype=float)
                    phase = (time_index - time_index[0]) / (
                        time_index[-1] - time_index[0]
                    )
                    finite = np.isfinite(values)
                    if finite.sum() < 2:
                        continue
                    interpolated = np.interp(
                        PHASE_GRID, phase[finite], values[finite]
                    )
                    for lifecycle_phase, value in zip(PHASE_GRID, interpolated):
                        rows.append(
                            dict(
                                simulation=simulation,
                                track_population=population,
                                track_population_label=population_label,
                                track_id=track_id,
                                wave=wave_name,
                                field=field_name,
                                units=units,
                                lifecycle_phase=float(lifecycle_phase),
                                filtered_value=float(value),
                                longitude_half_width_deg=WAVE_LONGITUDE_HALF_WIDTH_DEG,
                                source="existing equatorial_wave_diagnostics.py cache",
                            )
                        )
    return pd.DataFrame(rows)


def track_following_wave_summary(samples):
    if len(samples) == 0:
        return pd.DataFrame()
    return (
        samples.groupby(
            ["track_population", "track_population_label", "simulation", "wave", "field", "units", "lifecycle_phase"],
            as_index=False,
        )
        .filtered_value.agg(
            median="median",
            p25=lambda values: values.quantile(0.25),
            p75=lambda values: values.quantile(0.75),
            event_count="count",
        )
    )


def wave_method_table():
    rows = []
    for wave_name in WAVE_NAMES:
        box = WAVE_BOXES[wave_name]
        for field_name, label, units in WAVE_FIELD_SPECS:
            rows.append(
                dict(
                    wave=wave_name,
                    field=field_name,
                    field_label=label,
                    units=units,
                    zonal_wavenumber_range=str(box["k"]),
                    period_days=str(box["period"]),
                    equivalent_depth_m=str(box["depth"]),
                    latitude_projection=WAVE_LATITUDE_PROJECTION[wave_name],
                    latitude_bounds="10S-10N",
                    composite="filtered field within 10 longitude degrees of moving track centre",
                    cache_policy="read existing equatorial-wave cache; never regenerate",
                )
            )
    return pd.DataFrame(rows)


def plot_track_following_wave_lifecycle(summary, output_directory):
    if len(summary) == 0:
        print("No track-following filtered-wave lifecycle samples are available.")
        return
    figure_saver = FigureSaver(
        output_directory, enabled=save, figures_to_run=figures_to_run
    )
    for population in TRACK_POPULATIONS:
        for wave_name in WAVE_NAMES:
            wave_rows = summary[
                (summary.wave == wave_name)
                & (summary.track_population == population)
            ]
            if len(wave_rows) == 0:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.0), constrained_layout=True)
            for axis, (field_name, label, units) in zip(axes.ravel(), WAVE_FIELD_SPECS):
                field_rows = wave_rows[wave_rows.field == field_name]
                for simulation, color in zip(sim_names, cols):
                    values = field_rows[field_rows.simulation == simulation].sort_values("lifecycle_phase")
                    if len(values) == 0:
                        continue
                    axis.plot(values.lifecycle_phase, values["median"], color=color, lw=1.7, label=simulation)
                    axis.fill_between(values.lifecycle_phase, values.p25, values.p75, color=color, alpha=0.14, lw=0)
                axis.axhline(0.0, color="0.45", lw=0.8)
                axis.set(title=label, xlabel="Normalized lifecycle (0 first, 1 last sampled scene)", ylabel=units)
                axis.grid(alpha=0.2)
            axes[0, 0].legend(frameon=False, fontsize=8)
            fig.suptitle(
                f"{wave_name}-band environment following {TRACK_POPULATION_LABELS[population].lower()}\n"
                "Existing 10S-10N filtered cache sampled within 10 degrees of moving track centres",
                fontsize=11,
            )
            figure_saver.save(
                fig,
                f"{population}_{wave_name.lower()}_track_following_wave_environment",
                number=2,
            )
            plt.close(fig)


def main(argv=None):
    global figures_to_run
    args = parse_command_line(argv)
    if args.figures is not None:
        figures_to_run = list(args.figures)
    elif args.test:
        figures_to_run = [1, 3, 4]
    output_directory = save_loc
    if args.test:
        output_directory = os.path.join(save_loc, "testing")
    os.makedirs(output_directory, exist_ok=True)

    objects = expand_track_populations(load_track_objects(args.organisation_dir))
    if args.test:
        objects = select_test_tracks(
            objects, simulation=args.simulation, maximum=args.test_tracks
        )
        print(
            f"TEST MODE: {objects.track_id.nunique()} retained track IDs across "
            f"{objects.track_population.nunique()} populations from "
            f"{', '.join(sorted(objects.simulation.unique())) or 'no simulations'}"
        )
    track_time = aggregate_track_times(objects)
    samples = interpolate_event_lifecycles(track_time)
    summary = lifecycle_summary(samples)
    track_time.to_csv(
        os.path.join(output_directory, "convective_system_lifecycle_track_time.csv"), index=False
    )
    samples.to_csv(
        os.path.join(output_directory, "convective_system_normalized_lifecycle_event_samples.csv"),
        index=False,
    )
    summary.to_csv(
        os.path.join(output_directory, "convective_system_normalized_lifecycle_summary.csv"),
        index=False,
    )
    if 1 in figures_to_run:
        plot_lifecycle(summary, output_directory)
    if 2 in figures_to_run:
        wave_samples = track_following_wave_samples(track_time)
        wave_samples.to_csv(
            os.path.join(
                output_directory, "convective_system_track_following_wave_environment_samples.csv"
            ),
            index=False,
        )
        wave_summary = track_following_wave_summary(wave_samples)
        wave_summary.to_csv(
            os.path.join(
                output_directory, "convective_system_track_following_wave_environment_summary.csv"
            ),
            index=False,
        )
        wave_method_table().to_csv(
            os.path.join(output_directory, "convective_system_track_following_wave_method.csv"),
            index=False,
        )
        plot_track_following_wave_lifecycle(wave_summary, output_directory)
    if any(number in figures_to_run for number in (3, 4)):
        unfiltered_method_table().to_csv(
            os.path.join(
                output_directory,
                "convective_system_unfiltered_environment_method.csv",
            ),
            index=False,
        )
        unfiltered_samples = load_or_build_unfiltered_samples(
            track_time,
            output_directory,
            rebuild=args.rebuild_unfiltered_cache,
        )
        normalized_samples = interpolate_unfiltered_lifecycles(unfiltered_samples)
        normalized_summary = summarize_environment_samples(
            normalized_samples,
            "lifecycle_phase",
        )
        normalized_summary.to_csv(
            os.path.join(
                output_directory,
                "convective_system_unfiltered_normalized_lifecycle_summary.csv",
            ),
            index=False,
        )
        aligned_samples, timing_offsets = peak_aligned_environment_samples(
            unfiltered_samples
        )
        aligned_summary = summarize_environment_samples(
            aligned_samples,
            "lag_hours_from_precipitation_maximum",
        )
        aligned_summary.to_csv(
            os.path.join(
                output_directory,
                "convective_system_unfiltered_precipitation_peak_aligned_summary.csv",
            ),
            index=False,
        )
        timing_offsets.to_csv(
            os.path.join(
                output_directory,
                "convective_system_unfiltered_peak_timing_by_track.csv",
            ),
            index=False,
        )
        offsets_summary = timing_offset_summary(timing_offsets)
        offsets_summary.to_csv(
            os.path.join(
                output_directory,
                "convective_system_unfiltered_peak_timing_summary.csv",
            ),
            index=False,
        )
        if 3 in figures_to_run:
            plot_unfiltered_lifecycle(normalized_summary, output_directory)
        if 4 in figures_to_run:
            plot_peak_aligned_timing(
                aligned_summary,
                offsets_summary,
                output_directory,
            )
    print("\nFinished convective_lifecycle_composite_diagnostics.py")


if __name__ == "__main__":
    main()
